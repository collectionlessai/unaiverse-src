// Here we define the file messages: a direct message whose files travel next to it, read from disk
// while they are sent and written to disk while they arrive.
package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/libp2p/go-libp2p/core/peer"
)

// A file message is a message (the protobuf bytes of Python, whose FILE samples have no
// content) plus the files of those samples, each named by the key of its sample. It goes to
// a peer whose HELLO said CAP_FILE_STREAM as chunk frames (chunked.go) whose FIRST chunk has
// chunkFlagFile, carrying one logical stream:
//
//	[u32 manifest length][manifest: JSON {"msg_len": N, "files": [{"key": k, "size": s}, ...]}]
//	[the message: N bytes][file 1: s1 bytes]...[file K: sK bytes][SHA-256 of file 1]...[of file K]
//
// The total length of the FIRST chunk is the length of the whole stream (64 bits, so no
// field limits a file). The sender reads the files from disk chunk by chunk, hashing them,
// and starts again from chunk 0, reading them again, when a write fails, as for any chunked
// message. A file whose size or modification time is no longer what it was when the message
// was queued drops the message (reason file_changed), instead of sending a mixed file. The
// receiver keeps the message in memory and writes each file to a .part file in its incoming
// folder, hashing it; at the end it checks the hashes and queues the message with the paths
// of its files, which Python then moves to their place (Downloads).

const (
	chunkFlagFile       = 1 << 7 // In the flags of a FIRST chunk: a file message
	fileManifestMax     = 64 << 10
	fileMaxFilesPerMsg  = 64
	fileHashLen         = sha256.Size
	fileIncomingPattern = "*.part"
)

// The settings of the file messages, set by ConfigureSendQueues: fileIncomingDir empty means
// no file messages at all; maxFileSize 0 means no limit, on both sides.
var (
	fileIncomingDir string
	acceptFiles     bool
	maxFileSize     int64
)

// errFileChanged is the error of a file message whose files are not the ones that were queued:
// the message is dropped, not written again.
var errFileChanged = errors.New("file_changed")

// permanentSendError is the error of a write that would fail the same way if tried again: the
// writer drops the message instead of retrying it.
type permanentSendError struct {
	reason string
	err    error
}

func (e *permanentSendError) Error() string { return fmt.Sprintf("%s: %v", e.reason, e.err) }
func (e *permanentSendError) Unwrap() error { return e.err }

// fileRef is a file of a message being sent: where it is, and what it was when the message
// was queued.
type fileRef struct {
	Key     string `json:"key"`
	Path    string `json:"path"`
	Name    string `json:"name"` // As the receiver shows it in its progress events (the message has the real one)
	size    int64
	modTime time.Time
}

// filePayload is a file message in a send queue.
type filePayload struct {
	channel string
	msg     []byte
	files   []fileRef
	attempt uint8 // Of its next start; only the writer of the queue touches it
}

// fileManifest is the head of the logical stream of a file message.
type fileManifest struct {
	MsgLen int64          `json:"msg_len"`
	Files  []manifestFile `json:"files"`
}

// ReceivedFile is a file of a received message, as Python gets it with the message.
type ReceivedFile struct {
	Key  string `json:"key"`
	Path string `json:"path"`
	Size int64  `json:"size"`
}

// newFilePayload checks the files of a message to send and records what they are now.
func newFilePayload(channel string, msg []byte, files []fileRef) (*filePayload, error) {
	if len(files) == 0 || len(files) > fileMaxFilesPerMsg {
		return nil, fmt.Errorf("a file message has 1 to %d files, not %d", fileMaxFilesPerMsg, len(files))
	}
	seen := make(map[string]bool, len(files))
	for i := range files {
		if files[i].Key == "" || len(files[i].Key) > 255 || seen[files[i].Key] || len(files[i].Name) > 255 {
			return nil, fmt.Errorf("file key %q empty, too long or repeated, or its name too long", files[i].Key)
		}
		seen[files[i].Key] = true
		info, err := os.Stat(files[i].Path)
		if err != nil {
			return nil, err
		}
		if !info.Mode().IsRegular() {
			return nil, fmt.Errorf("%s is not a regular file", files[i].Path)
		}
		files[i].size, files[i].modTime = info.Size(), info.ModTime()
	}
	return &filePayload{channel: channel, msg: msg, files: files}, nil
}

// manifestFile is a file in the manifest of a file message.
type manifestFile struct {
	Key  string `json:"key"`
	Size int64  `json:"size"`
	Name string `json:"name,omitempty"`
}

// names are the names of the files of the message.
func (fp *filePayload) names() []string {
	names := make([]string, len(fp.files))
	for i, f := range fp.files {
		names[i] = f.Name
	}
	return names
}

// manifest is the manifest of the message.
func (fp *filePayload) manifest() []byte {
	var m fileManifest
	m.MsgLen = int64(len(fp.msg))
	for _, f := range fp.files {
		m.Files = append(m.Files, manifestFile{Key: f.Key, Size: f.size, Name: f.Name})
	}
	b, _ := json.Marshal(m)
	return b
}

// fileBytes is the size of all the files of the message.
func (fp *filePayload) fileBytes() int64 {
	var n int64
	for _, f := range fp.files {
		n += f.size
	}
	return n
}

// streamLen is the length of the logical stream of a message with this manifest.
func streamLen(manifestLen int, m *fileManifest) uint64 {
	n := uint64(4+manifestLen) + uint64(m.MsgLen)
	for _, f := range m.Files {
		n += uint64(f.Size) + fileHashLen
	}
	return n
}

// fileStream reads the logical stream of a file message, hashing the files as they are read.
type fileStream struct {
	prefix  *bytes.Reader
	fp      *filePayload
	opened  []*os.File
	current int // Index of the file being read, len(files) for the hashes
	reader  io.Reader
	read    int64
	hasher  hash.Hash
	sums    []byte
	tail    *bytes.Reader
}

// openFileStream opens the logical stream of a message from its start, checking that its files
// are still the ones that were queued.
func openFileStream(fp *filePayload) (*fileStream, error) {
	for _, f := range fp.files {
		info, err := os.Stat(f.Path)
		if err != nil || info.Size() != f.size || !info.ModTime().Equal(f.modTime) {
			return nil, &permanentSendError{reason: "file_changed", err: fmt.Errorf("%s: %w", f.Path, errFileChanged)}
		}
	}
	man := fp.manifest()
	prefix := make([]byte, 0, 4+len(man)+len(fp.msg))
	prefix = binary.BigEndian.AppendUint32(prefix, uint32(len(man)))
	prefix = append(prefix, man...)
	prefix = append(prefix, fp.msg...)
	return &fileStream{prefix: bytes.NewReader(prefix), fp: fp}, nil
}

// Read reads the next bytes of the logical stream.
func (s *fileStream) Read(p []byte) (int, error) {
	if s.prefix.Len() > 0 {
		return s.prefix.Read(p)
	}
	for s.current < len(s.fp.files) {
		ref := s.fp.files[s.current]
		if s.reader == nil {
			f, err := os.Open(ref.Path)
			if err != nil {
				return 0, &permanentSendError{reason: "file_changed", err: err}
			}
			s.opened = append(s.opened, f)
			s.hasher, s.read = sha256.New(), 0
			s.reader = io.TeeReader(io.LimitReader(f, ref.size), s.hasher)
		}
		n, err := s.reader.Read(p)
		s.read += int64(n)
		if n > 0 {
			return n, nil
		}
		if err == io.EOF {
			if s.read != ref.size {
				return 0, &permanentSendError{reason: "file_changed", err: fmt.Errorf("%s ended at %d of %d bytes: %w", ref.Path, s.read, ref.size, errFileChanged)}
			}
			s.sums = s.hasher.Sum(s.sums)
			s.opened[len(s.opened)-1].Close()
			s.current++
			s.reader = nil
			continue
		}
		return 0, err
	}
	if s.tail == nil {
		s.tail = bytes.NewReader(s.sums)
	}
	return s.tail.Read(p)
}

// Close closes the files the stream opened.
func (s *fileStream) Close() {
	for _, f := range s.opened {
		f.Close()
	}
}

// writeFileMessage writes a file message from its first chunk with the next attempt, telling
// progress (when not nil) the bytes written after each chunk.
func writeFileMessage(seq uint64, fp *filePayload, write func(frame []byte) error, progress func(sent, total uint64, attempt uint8)) error {
	attempt := fp.attempt
	fp.attempt = (fp.attempt + 1) & chunkAttemptMask
	stream, err := openFileStream(fp)
	if err != nil {
		return err
	}
	defer stream.Close()
	man := fp.manifest()
	total := uint64(4+len(man)+len(fp.msg)) + uint64(fp.fileBytes()) + uint64(len(fp.files))*fileHashLen
	return writeChunkStream(seq, attempt, true, fp.channel, total, stream, write, progress)
}

// incomingDirOf is the folder of the .part files of an instance (its own, since the instances
// of a machine can share the app folder).
func incomingDirOf(ni *NodeInstance) string {
	if fileIncomingDir == "" || ni.host == nil {
		return fileIncomingDir // An instance with no host (a test) uses the folder itself
	}
	return filepath.Join(fileIncomingDir, ni.host.ID().String())
}

// fileSink receives the logical stream of a file message: the message in memory, each file in
// a .part file, hashed while it is written.
type fileSink struct {
	dir      string
	total    uint64
	head     []byte // The manifest length and the manifest, while they arrive
	manLen   int    // -1 until known
	man      fileManifest
	msg      []byte
	current  int // Index of the file being written, len(files) for the hashes
	written  int64
	file     *os.File
	hasher   hash.Hash
	paths    []string
	sums     []byte
	gotSums  []byte
	reserved int64 // What the message counts in the reassembly caps once the manifest is known
}

// newFileSink starts the reception of a file message of the given total length.
func newFileSink(dir string, total uint64) *fileSink {
	return &fileSink{dir: dir, total: total, manLen: -1}
}

// write adds the next bytes of the logical stream. It returns the bytes the message now counts
// in the reassembly caps (once, when the manifest is known), or an error that drops it.
func (s *fileSink) write(data []byte) (int64, error) {
	var reserve int64
	for len(data) > 0 {
		switch {
		case s.manLen < 0 || len(s.head) < 4+s.manLen:
			need := 4
			if s.manLen >= 0 {
				need = 4 + s.manLen
			}
			take := min(need-len(s.head), len(data))
			s.head = append(s.head, data[:take]...)
			data = data[take:]
			if s.manLen < 0 && len(s.head) == 4 {
				s.manLen = int(binary.BigEndian.Uint32(s.head))
				if s.manLen == 0 || s.manLen > fileManifestMax {
					return 0, fmt.Errorf("manifest of %d bytes", s.manLen)
				}
			}
			if s.manLen >= 0 && len(s.head) == 4+s.manLen {
				if err := s.checkManifest(); err != nil {
					return 0, err
				}
				s.reserved, reserve = s.man.MsgLen, s.man.MsgLen
				s.msg = make([]byte, 0, s.man.MsgLen)
			}
		case int64(len(s.msg)) < s.man.MsgLen:
			take := int(min(s.man.MsgLen-int64(len(s.msg)), int64(len(data))))
			s.msg = append(s.msg, data[:take]...)
			data = data[take:]
		case s.current < len(s.man.Files):
			if s.file == nil {
				f, err := os.CreateTemp(s.dir, fileIncomingPattern)
				if err != nil {
					return 0, err
				}
				s.file, s.hasher, s.written = f, sha256.New(), 0
				s.paths = append(s.paths, f.Name())
			}
			size := s.man.Files[s.current].Size
			take := int(min(size-s.written, int64(len(data))))
			if take > 0 {
				if _, err := s.file.Write(data[:take]); err != nil {
					return 0, err
				}
				s.hasher.Write(data[:take])
				s.written += int64(take)
				data = data[take:]
			}
			if s.written == size {
				if err := s.file.Close(); err != nil {
					s.file = nil
					return 0, err
				}
				s.file = nil
				s.sums = s.hasher.Sum(s.sums)
				s.current++
			}
		default:
			take := min(len(s.man.Files)*fileHashLen-len(s.gotSums), len(data))
			if take <= 0 {
				return 0, fmt.Errorf("data past the end of the message")
			}
			s.gotSums = append(s.gotSums, data[:take]...)
			data = data[take:]
		}
	}
	return reserve, nil
}

// names are the names of the files, once the manifest is known (nil before).
func (s *fileSink) names() []string {
	if s.manLen < 0 || len(s.head) < 4+s.manLen {
		return nil
	}
	names := make([]string, len(s.man.Files))
	for i, f := range s.man.Files {
		names[i] = f.Name
	}
	return names
}

// checkManifest checks the manifest against the total length and the limits.
func (s *fileSink) checkManifest() error {
	if err := json.Unmarshal(s.head[4:], &s.man); err != nil {
		return fmt.Errorf("manifest: %w", err)
	}
	if s.man.MsgLen < 0 || len(s.man.Files) == 0 || len(s.man.Files) > fileMaxFilesPerMsg {
		return fmt.Errorf("manifest with a message of %d bytes and %d files", s.man.MsgLen, len(s.man.Files))
	}
	if limit := chunkedMaxMessageSize; limit > 0 && s.man.MsgLen > limit {
		return fmt.Errorf("message of %d bytes above the limit (%d)", s.man.MsgLen, limit)
	}
	for _, f := range s.man.Files {
		if f.Size < 0 || f.Key == "" || len(f.Key) > 255 || len(f.Name) > 255 {
			return fmt.Errorf("file %q of %d bytes", f.Key, f.Size)
		}
		if maxFileSize > 0 && f.Size > maxFileSize {
			return fmt.Errorf("file of %d bytes above the limit (%d)", f.Size, maxFileSize)
		}
	}
	if streamLen(s.manLen, &s.man) != s.total {
		return fmt.Errorf("manifest says %d bytes, the message %d", streamLen(s.manLen, &s.man), s.total)
	}
	return nil
}

// finish checks the hashes of a complete message and returns it with its files.
func (s *fileSink) finish() ([]byte, []ReceivedFile, error) {
	if s.manLen < 0 || s.current != len(s.man.Files) || len(s.gotSums) != len(s.man.Files)*fileHashLen {
		return nil, nil, fmt.Errorf("incomplete file message")
	}
	if !bytes.Equal(s.sums, s.gotSums) {
		return nil, nil, fmt.Errorf("hash mismatch")
	}
	files := make([]ReceivedFile, len(s.man.Files))
	for i, f := range s.man.Files {
		files[i] = ReceivedFile{Key: f.Key, Path: s.paths[i], Size: f.Size}
	}
	s.paths = nil // Handed over: discard no longer removes them
	return s.msg, files, nil
}

// discard removes what the sink wrote.
func (s *fileSink) discard() {
	if s.file != nil {
		s.file.Close()
		s.file = nil
	}
	removeFiles(s.paths)
	s.paths = nil
}

// removeFiles removes files that nobody will receive.
func removeFiles(paths []string) {
	for _, p := range paths {
		if err := os.Remove(p); err != nil && !os.IsNotExist(err) {
			logger.Warnf("[GO] ⚠️ Could not remove %s: %v\n", p, err)
		}
	}
}

// receivedFilePaths are the paths of the files of a queued message.
func receivedFilePaths(files []ReceivedFile) []string {
	paths := make([]string, len(files))
	for i, f := range files {
		paths[i] = f.Path
	}
	return paths
}

// queueFileMessage queues a file message for a peer, after the checks that can refuse it at
// once: the peer must take file messages, and the message and the files must be within the
// limits of both sides. It returns as queueDirectMessage.
func queueFileMessage(ni *NodeInstance, pid peer.ID, channel string, msg []byte, files []fileRef) (uint64, string, string, error) {
	if pid == ni.host.ID() {
		return 0, SendCodeSelfSend, "Attempt to send direct message to self is invalid", nil
	}
	if ni.sendStore == nil || chunkSize <= 0 || fileIncomingDir == "" {
		return 0, SendCodeNotConfigured, "File messages not configured (ConfigureSendQueues)", nil
	}
	limits := ni.limitsOf(pid)
	refuse := func(err error) (uint64, string, string, error) {
		return 0, SendCodeTooLarge, fmt.Sprintf("Cannot send a file message to %s", pid), err
	}
	if limits.caps&capFileStream == 0 {
		return 0, SendCodeNoFileStream, fmt.Sprintf("%s does not take file messages", pid), nil
	}
	if len(channel) > 255 {
		return refuse(fmt.Errorf("channel name exceeds 255 bytes: %s", channel))
	}
	if limit := chunkedLimitFor(limits); limit > 0 && int64(len(msg)) > limit {
		return refuse(fmt.Errorf("message of %d bytes above the limit (%d)", len(msg), limit))
	}
	fp, err := newFilePayload(channel, msg, files)
	if err != nil {
		return refuse(err)
	}
	for _, f := range fp.files {
		for _, limit := range []int64{maxFileSize, limits.maxFileSize} {
			if limit > 0 && f.size > limit {
				return refuse(fmt.Errorf("file %s of %d bytes above the limit (%d) of the peer or of this node", f.Path, f.size, limit))
			}
		}
	}
	seq, err := ni.sendStore.enqueue(pid, &queuedFrame{file: fp})
	if err != nil {
		return 0, SendCodeQueueFull, fmt.Sprintf("File message to %s refused", pid), err
	}
	logger.Debugf("[GO] 📥 Instance %d: Queued file message to %s (seq %d, %d files, %d bytes).\n", ni.instanceIndex, pid, seq, len(files), fp.fileBytes())
	return seq, "", fmt.Sprintf("File message to %s queued.", pid), nil
}
