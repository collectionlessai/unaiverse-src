// Here we define the chunked frames: direct messages larger than a chunk, on the same chat stream.
package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"sync"
	"time"

	"github.com/libp2p/go-libp2p/core/peer"
	pwebrtc "github.com/pion/webrtc/v4"
)

// A direct message larger than chunkSize, to a peer whose HELLO said CAP_CHUNKED_FRAMES,
// travels as consecutive chunk frames on the chat stream, in the per-peer order of the
// send queue (all the chunks of a message before the next message). A chunk frame is
// told apart from a legacy frame by the high bit of its 4-byte length (a legacy frame
// is always below 2 GB):
//
//	[u32 chunkFrameFlag | body length][u64 seq][u32 index][u8 flags]
//	FIRST only: [u64 total length][u8 channel length][channel]
//	[data]
//
// The chunks travel on the WebRTC DataChannel of the peer when one is open (as the small
// messages do, so that the order of the two kinds is kept), else on the chat stream; a
// DataChannel carries each chunk frame in its own pieces (writeFrameToWebRTCDataChannel).
//
// The receiver reassembles one message per peer, keyed by its sequence number, and
// delivers it whole when the LAST chunk arrives. When a write fails, the writer sends
// the message again from its first chunk (a written chunk may never have reached the
// receiver, which cannot say what it got), on a new stream or on the chat stream when the
// DataChannel failed. Bits 2 to 6 of the flags are the attempt: every new start of a
// message has the next one (modulo 32). A FIRST chunk of a newer attempt of the message
// under reassembly starts it over; a chunk of another attempt or message is a stale chunk
// of the path that failed, still arriving on it, and is dropped alone. A peer that never
// said CAP_CHUNKED_FRAMES (older nodes, browsers) keeps getting legacy frames, up to
// MaxMessageSize.

const (
	capChunkedFrames = 1 << 0 // CAP_CHUNKED_FRAMES of sess_msg.py
	capFileStream    = 1 << 1 // CAP_FILE_STREAM of sess_msg.py

	chunkFrameFlag   = uint32(1) << 31
	chunkFlagFirst   = 1 << 0
	chunkFlagLast    = 1 << 1
	chunkAttemptBits = 2           // The attempt is (flags >> chunkAttemptBits) & chunkAttemptMask
	chunkAttemptMask = 0x1f        // Five bits (bit 7 is chunkFlagFile, file.go)
	chunkFixedHeader = 8 + 4 + 1   // seq, index, flags
	chunkFirstExtra  = 8 + 1 + 255 // total, channel length, channel (at most)
)

// popBatchMaxBytes: a pop stops adding messages past this (its reply has 32-bit lengths);
// a variable so that a test can lower it.
var popBatchMaxBytes = 1536 * 1024 * 1024

// The limits of the chunked frames, set by ConfigureSendQueues (all from Python, no
// defaults here): chunkSize 0 means no chunked frames at all.
var (
	chunkSize                  int
	chunkedMaxMessageSize      int64 // 0: no limit
	reassemblyMaxBytesPerPeer  int64 // 0: chunkedMaxMessageSize
	reassemblyMaxBytesTotal    int64 // 0: no limit
	reassemblyTimeout          time.Duration
	chunkedFramesConfiguredErr = fmt.Errorf("chunked frames not configured")
)

// peerLimits is what a peer can receive, as its Session told Python.
type peerLimits struct {
	caps           uint32
	maxMessageSize int64 // From its profile: 0 no limit, -1 unknown (legacy)
	maxFileSize    int64 // From its profile: 0 no limit
}

// setPeerLimits records what a peer can receive.
func (ni *NodeInstance) setPeerLimits(pid peer.ID, caps uint32, maxMessageSize, maxFileSize int64) {
	ni.peerLimitsMu.Lock()
	defer ni.peerLimitsMu.Unlock()
	if ni.peerLimits == nil {
		ni.peerLimits = make(map[peer.ID]peerLimits)
	}
	ni.peerLimits[pid] = peerLimits{caps: caps, maxMessageSize: maxMessageSize, maxFileSize: maxFileSize}
}

// limitsOf is what a peer can receive (legacy when unknown).
func (ni *NodeInstance) limitsOf(pid peer.ID) peerLimits {
	ni.peerLimitsMu.RLock()
	defer ni.peerLimitsMu.RUnlock()
	if l, ok := ni.peerLimits[pid]; ok {
		return l
	}
	return peerLimits{caps: 0, maxMessageSize: -1}
}

// chunkedLimitFor is the largest message that may go to a peer as chunk frames: the
// smallest of our maximum and the peer's (0: no limit).
func chunkedLimitFor(l peerLimits) int64 {
	limit := chunkedMaxMessageSize
	if l.maxMessageSize > 0 && (limit == 0 || l.maxMessageSize < limit) {
		limit = l.maxMessageSize
	}
	return limit
}

// usesChunks tells whether a message of the given size goes to the peer as chunk frames.
func usesChunks(l peerLimits, channel string, payloadLen int) bool {
	return chunkSize > 0 && l.caps&capChunkedFrames != 0 && 1+len(channel)+payloadLen > chunkSize
}

// buildChunkFrame builds one chunk frame of the given attempt.
func buildChunkFrame(seq uint64, index uint32, attempt uint8, first, last bool, total uint64, channel string, data []byte) []byte {
	return buildChunkFrameOf(seq, index, attempt, false, first, last, total, channel, data)
}

// buildChunkFrameOf builds one chunk frame of the given attempt, of a file message or not.
func buildChunkFrameOf(seq uint64, index uint32, attempt uint8, file, first, last bool, total uint64, channel string, data []byte) []byte {
	flags := (attempt & chunkAttemptMask) << chunkAttemptBits
	if file && first {
		flags |= chunkFlagFile
	}
	if first {
		flags |= chunkFlagFirst
	}
	if last {
		flags |= chunkFlagLast
	}
	bodyLen := chunkFixedHeader + len(data)
	if first {
		bodyLen += 8 + 1 + len(channel)
	}
	buf := new(bytes.Buffer)
	buf.Grow(4 + bodyLen)
	_ = binary.Write(buf, binary.BigEndian, chunkFrameFlag|uint32(bodyLen))
	_ = binary.Write(buf, binary.BigEndian, seq)
	_ = binary.Write(buf, binary.BigEndian, index)
	buf.WriteByte(flags)
	if first {
		_ = binary.Write(buf, binary.BigEndian, total)
		buf.WriteByte(byte(len(channel)))
		buf.WriteString(channel)
	}
	buf.Write(data)
	return buf.Bytes()
}

// writeChunkedMessage writes a message as chunk frames, from its first chunk (see
// writeLargeMessage), with no progress events. The writer of a send queue does the same
// with its progress events (newInstanceSendStore); it is the only one that touches cp.attempt.
func writeChunkedMessage(ni *NodeInstance, pid peer.ID, seq uint64, cp *chunkedPayload) error {
	return writeLargeMessage(ni, pid, seq, transferChunked, int64(len(cp.payload)), func(write func([]byte) error) error {
		return writeChunks(seq, cp, write, nil)
	})
}

// writeLargeMessage writes a message as chunk frames, with start, on the DataChannel of the
// peer when one is open, else on its chat stream; each frame has the write deadline of its
// own size. A DataChannel that fails in the middle of the message is left for the chat
// stream, where the message starts again with the next attempt (start takes it each time);
// a permanent error ends it at once.
func writeLargeMessage(ni *NodeInstance, pid peer.ID, seq uint64, kind string, size int64, start func(write func([]byte) error) error) error {
	if chunkSize <= 0 {
		return chunkedFramesConfiguredErr
	}
	if wconn := openDataChannel(ni, pid); wconn != nil {
		err := start(func(frame []byte) error { return writeChunkToDataChannel(ni, wconn, frame) })
		if err == nil {
			logger.Infof("[GO] 📤 Instance %d: Sent a %s message (%d bytes) to %s via WebRTC DataChannel\n", ni.instanceIndex, kind, size, pid)
			return nil
		}
		var permanent *permanentSendError
		if errors.As(err, &permanent) {
			return err
		}
		logger.Warnf("[GO] ⚠️ Instance %d: WebRTC send of message %d to %s failed (%v), sending it again on a stream\n", ni.instanceIndex, seq, pid, err)
	}
	return start(func(frame []byte) error {
		message, err := sendFrameOnChatStream(ni, pid, frame)
		if err != nil {
			return fmt.Errorf("%s %w", message, err)
		}
		return nil
	})
}

// writeChunks writes all the chunks of an in-memory message with the next attempt, telling
// progress (when not nil) the bytes written after each chunk.
func writeChunks(seq uint64, cp *chunkedPayload, write func(frame []byte) error, progress func(sent, total uint64, attempt uint8)) error {
	attempt := cp.attempt
	cp.attempt = (cp.attempt + 1) & chunkAttemptMask
	return writeChunkStream(seq, attempt, false, cp.channel, uint64(len(cp.payload)), bytes.NewReader(cp.payload), write, progress)
}

// writeChunkStream writes the total bytes of src as the chunks of one attempt of a message,
// one frame at a time with write, stopping at the first error, and tells progress (when not
// nil) the bytes written after each chunk.
func writeChunkStream(seq uint64, attempt uint8, file bool, channel string, total uint64, src io.Reader, write func(frame []byte) error, progress func(sent, total uint64, attempt uint8)) error {
	buf := make([]byte, min(uint64(chunkSize-chunkFixedHeader), max(total, 1)))
	var sent uint64
	for index := uint32(0); ; index++ {
		n := int(min(uint64(len(buf)), total-sent))
		if _, err := io.ReadFull(src, buf[:n]); err != nil {
			return fmt.Errorf("chunk %d of message %d (attempt %d): %w", index, seq, attempt, err)
		}
		sent += uint64(n)
		first, last := index == 0, sent == total
		frame := buildChunkFrameOf(seq, index, attempt, file, first, last, total, channel, buf[:n])
		if err := write(frame); err != nil {
			return fmt.Errorf("chunk %d of message %d (attempt %d): %w", index, seq, attempt, err)
		}
		if progress != nil {
			progress(sent, total, attempt)
		}
		if last {
			return nil
		}
	}
}

// dataChannelPoll is how often a chunk waiting for room on a DataChannel looks again.
const dataChannelPoll = 2 * time.Millisecond

// writeChunkToDataChannel writes one chunk frame on a DataChannel once what the channel
// still has to send is at most one chunk: a DataChannel takes all that it is given into
// memory, so a large message must not be handed to it at once. The wait has the write
// deadline of the frame, as on a chat stream; a channel that closes ends it.
func writeChunkToDataChannel(ni *NodeInstance, wconn *WebRTCConn, frame []byte) error {
	var deadline time.Time
	if timeout := chatWriteTimeout(len(frame)); timeout > 0 {
		deadline = time.Now().Add(timeout)
	}
	for wconn.dc.BufferedAmount() > uint64(chunkSize) {
		if wconn.dc.ReadyState() != pwebrtc.DataChannelStateOpen {
			return fmt.Errorf("the DataChannel closed")
		}
		if !deadline.IsZero() && time.Now().After(deadline) {
			return fmt.Errorf("the DataChannel did not drain in time (%d bytes still to send)", wconn.dc.BufferedAmount())
		}
		select {
		case <-ni.ctx.Done():
			return ni.ctx.Err()
		case <-time.After(dataChannelPoll):
		}
	}
	return writeFrameToWebRTCDataChannel(wconn, frame)
}

// partialMessage is a message under reassembly.
type partialMessage struct {
	seq      uint64
	attempt  uint8
	next     uint32
	total    uint64
	channel  string
	protocol string    // Of the path of its FIRST chunk, as the message is queued with
	buf      []byte    // An in-memory message
	sink     *fileSink // A file message (file.go)
	got      uint64    // Bytes received
	reserved int64     // What it counts in reassemblyStore.reserved
	lastSeen time.Time
	throttle progressThrottle // Of its receive_progress events
}

// reassemblyStore holds the messages under reassembly of an instance, one per peer.
type reassemblyStore struct {
	mu       sync.Mutex
	byPeer   map[peer.ID]*partialMessage
	reserved int64 // Sum of the in-memory bytes declared under reassembly
}

// newReassemblyStore returns the reassembly store of an instance.
func newReassemblyStore() *reassemblyStore {
	return &reassemblyStore{byPeer: make(map[peer.ID]*partialMessage)}
}

// dropLocked forgets the message under reassembly of a peer. The caller holds r.mu.
func (r *reassemblyStore) dropLocked(pid peer.ID) {
	if p, ok := r.byPeer[pid]; ok {
		r.reserved -= p.reserved
		if p.sink != nil {
			p.sink.discard()
		}
		delete(r.byPeer, pid)
	}
}

// Drop forgets the message under reassembly of a peer (it was purged).
func (r *reassemblyStore) Drop(pid peer.ID) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.dropLocked(pid)
}

// sweep forgets, every few seconds, the messages whose next chunk did not come in time.
func (r *reassemblyStore) sweep(ctx context.Context, ni *NodeInstance) {
	defer recoverGoroutine("reassemblySweeper")
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case now := <-ticker.C:
			r.mu.Lock()
			for pid, p := range r.byPeer {
				if reassemblyTimeout > 0 && now.Sub(p.lastSeen) > reassemblyTimeout {
					logger.Warnf("[GO] ⏳ Instance %d: message %d from %s timed out under reassembly (%d of %d bytes).\n", ni.instanceIndex, p.seq, pid, p.got, p.total)
					r.failLocked(ni, pid, "timeout")
				}
			}
			r.mu.Unlock()
		}
	}
}

// readChunkFrame reads the body of one chunk frame from a chat stream and adds it to the
// message under reassembly of the peer, storing the message when it is complete. It
// returns false when the stream must be reset (a protocol violation or a limit).
func readChunkFrame(ni *NodeInstance, s io.Reader, from peer.ID, bodyLen uint32) bool {
	if !chunkBodyLenOK(ni, from, bodyLen) {
		return false
	}
	body := make([]byte, bodyLen)
	if _, err := io.ReadFull(s, body); err != nil {
		logger.Errorf("[GO] ❌ Instance %d: error reading a chunk frame from %s: %v\n", ni.instanceIndex, from, err)
		return false
	}
	return addChunk(ni, from, body, UnaiverseChatProtocol)
}

// isChunkFrame tells whether a whole frame (its length included) is a chunk frame.
func isChunkFrame(frame []byte) bool {
	return len(frame) >= 4 && binary.BigEndian.Uint32(frame[0:4])&chunkFrameFlag != 0
}

// receiveDataChannelChunk adds a chunk frame that came whole on a DataChannel (its length
// included) to the message under reassembly of the peer. A frame that breaks the rules
// is dropped, with the message under reassembly: the channel has no stream to reset.
func receiveDataChannelChunk(ni *NodeInstance, from peer.ID, frame []byte) {
	bodyLen := binary.BigEndian.Uint32(frame[0:4]) &^ chunkFrameFlag
	if !chunkBodyLenOK(ni, from, bodyLen) || int(bodyLen) != len(frame)-4 {
		logger.Errorf("[GO] ❌ Instance %d: chunk frame of %d bytes (%d declared) from %s on the WebRTC DataChannel refused.\n", ni.instanceIndex, len(frame)-4, bodyLen, from)
		ni.reassembly.Drop(from)
		return
	}
	addChunk(ni, from, frame[4:], UnaiverseWebRTCSignalProtocol)
}

// chunkBodyLenOK tells whether the body of a chunk frame can be as long as it says.
func chunkBodyLenOK(ni *NodeInstance, from peer.ID, bodyLen uint32) bool {
	if chunkSize <= 0 || bodyLen < chunkFixedHeader || int(bodyLen) > chunkSize+chunkFirstExtra {
		logger.Errorf("[GO] ❌ Instance %d: chunk frame of %d bytes from %s refused (chunk size %d).\n", ni.instanceIndex, bodyLen, from, chunkSize)
		return false
	}
	return true
}

// newerAttempt tells whether attempt a comes after attempt b (modulo 64, half the circle).
func newerAttempt(a, b uint8) bool {
	d := (a - b) & chunkAttemptMask
	return d != 0 && d < (chunkAttemptMask+1)/2
}

// addChunk adds the body of one chunk frame, which came on the path of the given protocol,
// to the message under reassembly of the peer, storing the message when it is complete. It
// returns false on a protocol violation or a limit: the message under reassembly is gone.
func addChunk(ni *NodeInstance, from peer.ID, body []byte, protocol string) bool {
	seq := binary.BigEndian.Uint64(body[0:8])
	index := binary.BigEndian.Uint32(body[8:12])
	flags := body[12]
	attempt := (flags >> chunkAttemptBits) & chunkAttemptMask
	rest := body[chunkFixedHeader:]

	r := ni.reassembly
	if r == nil {
		return false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	p := r.byPeer[from]

	if flags&chunkFlagFirst != 0 {
		if index != 0 || len(rest) < 9 {
			return false
		}
		if p != nil && p.seq == seq && !newerAttempt(attempt, p.attempt) {
			logger.Debugf("[GO] Instance %d: FIRST chunk of attempt %d of message %d from %s, under reassembly at attempt %d: stale, dropped.\n", ni.instanceIndex, attempt, seq, from, p.attempt)
			return true
		}
		total := binary.BigEndian.Uint64(rest[0:8])
		channelLen := int(rest[8])
		if len(rest) < 9+channelLen {
			return false
		}
		channel := string(rest[9 : 9+channelLen])
		data := rest[9+channelLen:]
		if isSessionProtocol(channel) {
			return false
		}

		// A new message, or the same one sent again from its first chunk after a broken stream
		r.dropLocked(from)
		if uint64(len(data)) > total {
			return false
		}
		p = &partialMessage{seq: seq, attempt: attempt, next: 1, protocol: protocol, total: total, channel: channel, lastSeen: time.Now()}
		if flags&chunkFlagFile != 0 {
			dir := incomingDirOf(ni)
			if dir == "" || !acceptFiles {
				logger.Errorf("[GO] ❌ Instance %d: file message %d from %s refused (files not accepted). Resetting.\n", ni.instanceIndex, seq, from)
				recordTransferEvent(ni, p.event(SendEventReceiveDropped, from, "files_not_accepted"))
				return false
			}
			if err := os.MkdirAll(dir, 0o700); err != nil {
				logger.Errorf("[GO] ❌ Instance %d: cannot create %s: %v. Resetting.\n", ni.instanceIndex, dir, err)
				recordTransferEvent(ni, p.event(SendEventReceiveDropped, from, "disk_error"))
				return false
			}
			p.sink = newFileSink(dir, total)
		} else {
			if !r.fitsLocked(int64(total)) {
				logger.Errorf("[GO] ❌ Instance %d: message %d of %d bytes from %s is above the limits. Resetting.\n", ni.instanceIndex, seq, total, from)
				recordTransferEvent(ni, p.event(SendEventReceiveDropped, from, "too_large"))
				return false
			}
			// The declared size at once (the caps above count it): growing the buffer as the chunks
			// came copied the message again at each doubling, and pages not yet written cost no memory
			p.buf = make([]byte, 0, total)
			p.reserved = int64(total)
			r.reserved += p.reserved
		}
		r.byPeer[from] = p
		if err := r.addLocked(p, data); err != nil {
			logger.Errorf("[GO] ❌ Instance %d: message %d from %s dropped: %v. Resetting.\n", ni.instanceIndex, seq, from, err)
			r.failLocked(ni, from, "refused")
			return false
		}
	} else {
		if p == nil || p.seq != seq || p.attempt != attempt {
			logger.Debugf("[GO] Instance %d: chunk %d of attempt %d of message %d from %s, not under reassembly: stale, dropped.\n", ni.instanceIndex, index, attempt, seq, from)
			return true
		}
		if index < p.next {
			return true // A duplicate: already added
		}
		if index != p.next || p.got+uint64(len(rest)) > p.total {
			logger.Errorf("[GO] ❌ Instance %d: chunk %d of message %d from %s out of place. Resetting.\n", ni.instanceIndex, index, seq, from)
			r.failLocked(ni, from, "out_of_place")
			return false
		}
		if err := r.addLocked(p, rest); err != nil {
			logger.Errorf("[GO] ❌ Instance %d: message %d from %s dropped: %v. Resetting.\n", ni.instanceIndex, seq, from, err)
			r.failLocked(ni, from, "refused")
			return false
		}
		p.next++
		p.lastSeen = time.Now()
	}
	if p.throttle.due(p.lastSeen, p.got, p.total) {
		recordTransferEvent(ni, p.event(SendEventReceiveProgress, from, ""))
	}

	if flags&chunkFlagLast != 0 {
		if p.got != p.total {
			logger.Errorf("[GO] ❌ Instance %d: message %d from %s ended at %d of %d bytes. Resetting.\n", ni.instanceIndex, seq, from, p.got, p.total)
			r.failLocked(ni, from, "wrong_length")
			return false
		}
		channel, payload, protocol := p.channel, p.buf, p.protocol
		if p.sink != nil {
			msg, files, err := p.sink.finish()
			if err != nil {
				logger.Errorf("[GO] ❌ Instance %d: file message %d from %s dropped: %v. Resetting.\n", ni.instanceIndex, seq, from, err)
				r.failLocked(ni, from, "hash_mismatch")
				return false
			}
			r.dropLocked(from)
			logger.Infof("[GO] 📨 Instance %d: Received a file message (%d bytes, %d files) on channel '%s' from %s.\n", ni.instanceIndex, len(msg), len(files), channel, from)
			storeReceivedMessageWithFiles(ni, from, channel, protocol, msg, files)
			return true
		}
		r.dropLocked(from)
		logger.Infof("[GO] 📨 Instance %d: Received a chunked message (%d bytes) on channel '%s' from %s.\n", ni.instanceIndex, len(payload), channel, from)
		storeReceivedMessage(ni, from, channel, protocol, payload)
	}
	return true
}

// Kinds of the messages that report their progress.
const (
	transferChunked = "chunked"
	transferFile    = "file"
)

// progressMinInterval is how often at most a message reports its progress (and only once it
// moved by 1% at least); a variable, so that a test can change it.
var progressMinInterval = 500 * time.Millisecond

// progressThrottle decides when a message reports its progress: at its first chunk, at its
// last, and in between when it moved by 1% at least and progressMinInterval has passed.
type progressThrottle struct {
	started   bool
	lastBytes uint64
	lastAt    time.Time
}

// due tells whether the progress of a message is to be reported now, and if so takes note.
func (t *progressThrottle) due(now time.Time, done, total uint64) bool {
	report := !t.started || done >= total ||
		(total > 0 && done*100/total > t.lastBytes*100/total && now.Sub(t.lastAt) >= progressMinInterval)
	if report {
		t.started, t.lastBytes, t.lastAt = true, done, now
	}
	return report
}

// recordTransferEvent records a receive event of an instance among its send events (one
// PopSendEvents gives both); an instance with no send queues records none.
func recordTransferEvent(ni *NodeInstance, ev *SendEvent) {
	if ni.sendStore == nil {
		return
	}
	ni.sendStore.mu.Lock()
	defer ni.sendStore.mu.Unlock()
	ni.sendStore.recordLocked(ev)
}

// event is a receive event about a message under reassembly.
func (p *partialMessage) event(typ string, from peer.ID, reason string) *SendEvent {
	ev := &SendEvent{Type: typ, PeerID: from.String(), Seq: p.seq, Bytes: int64(p.got), Total: int64(p.total),
		Attempt: int(p.attempt), Kind: transferChunked, Reason: reason}
	if p.sink != nil {
		ev.Kind, ev.Files = transferFile, p.sink.names()
	}
	return ev
}

// failLocked drops the message under reassembly of a peer that broke a rule or a limit, and
// records why. The caller holds r.mu.
func (r *reassemblyStore) failLocked(ni *NodeInstance, pid peer.ID, reason string) {
	if p, ok := r.byPeer[pid]; ok {
		recordTransferEvent(ni, p.event(SendEventReceiveDropped, pid, reason))
	}
	r.dropLocked(pid)
}

// fitsLocked tells whether an in-memory message of the given size fits the reassembly caps.
// The caller holds r.mu.
func (r *reassemblyStore) fitsLocked(size int64) bool {
	perPeer := reassemblyMaxBytesPerPeer
	if perPeer == 0 {
		perPeer = chunkedMaxMessageSize
	}
	return !(chunkedMaxMessageSize > 0 && size > chunkedMaxMessageSize) &&
		!(perPeer > 0 && size > perPeer) &&
		!(reassemblyMaxBytesTotal > 0 && r.reserved+size > reassemblyMaxBytesTotal)
}

// addLocked adds the data of a chunk to a message under reassembly: to its buffer, or to its
// file sink, whose in-memory part then counts in the caps. The caller holds r.mu.
func (r *reassemblyStore) addLocked(p *partialMessage, data []byte) error {
	p.got += uint64(len(data))
	if p.sink == nil {
		p.buf = append(p.buf, data...)
		return nil
	}
	reserve, err := p.sink.write(data)
	if err != nil {
		return err
	}
	if reserve > 0 {
		if !r.fitsLocked(reserve) {
			return fmt.Errorf("a message of %d bytes is above the limits", reserve)
		}
		p.reserved = reserve
		r.reserved += reserve
	}
	return nil
}
