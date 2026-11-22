// Tests of the file messages: the logical stream, its reception on disk, and the queue.
package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/peer"
)

// withFiles turns the file messages on for one test, with an incoming folder of its own.
func withFiles(t *testing.T, size int, maxFile int64) string {
	t.Helper()
	withChunks(t, size, 0)
	prev := []any{fileIncomingDir, acceptFiles, maxFileSize}
	fileIncomingDir, acceptFiles, maxFileSize = t.TempDir(), true, maxFile
	t.Cleanup(func() {
		fileIncomingDir, acceptFiles, maxFileSize = prev[0].(string), prev[1].(bool), prev[2].(int64)
	})
	return fileIncomingDir
}

// writeTemp writes a file for a test and returns its path.
func writeTemp(t *testing.T, name string, data []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// fileFrames are the chunk frames of the next attempt of a file message.
func fileFrames(t *testing.T, seq uint64, fp *filePayload) [][]byte {
	t.Helper()
	var frames [][]byte
	if err := writeFileMessage(seq, fp, func(frame []byte) error {
		frames = append(frames, append([]byte(nil), frame...))
		return nil
	}, nil); err != nil {
		t.Fatalf("writeFileMessage: %v", err)
	}
	return frames
}

// partFiles are the .part files left in a folder.
func partFiles(t *testing.T, dir string) []string {
	t.Helper()
	matches, _ := filepath.Glob(filepath.Join(dir, fileIncomingPattern))
	return matches
}

// A message with two files (one empty) arrives with its files on disk, whole.
func TestFileMessageArrivesWithItsFiles(t *testing.T) {
	dir := withFiles(t, 1024, 0)
	ni := newChunkInstance(t)
	big := patterned(5000)
	files := []fileRef{{Key: "doc", Path: writeTemp(t, "a.bin", big)}, {Key: "empty", Path: writeTemp(t, "b.bin", nil)}}
	fp, err := newFilePayload("A::dm:B-misc", []byte("the message"), files)
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range fileFrames(t, 2, fp) {
		if !feed(t, ni, peer.ID("alice"), f) {
			t.Fatal("a chunk of a valid file message was refused")
		}
	}
	popped := ni.messageStore.popNetworkBatch()
	if len(popped) != 1 || string(popped[0].Data) != "the message" || len(popped[0].Files) != 2 {
		t.Fatalf("got %d messages, want the message once with its 2 files", len(popped))
	}
	for i, want := range [][]byte{big, nil} {
		got, err := os.ReadFile(popped[0].Files[i].Path)
		if err != nil || !bytes.Equal(got, want) || popped[0].Files[i].Key != files[i].Key || filepath.Dir(popped[0].Files[i].Path) != dir {
			t.Fatalf("file %d: %d bytes in %s, key %q (err %v)", i, len(got), popped[0].Files[i].Path, popped[0].Files[i].Key, err)
		}
	}
	if ni.reassembly.reserved != 0 || len(ni.reassembly.byPeer) != 0 {
		t.Fatal("the reassembly state was not released")
	}
}

// A message sent again from its first chunk (next attempt) leaves no file of the broken one.
func TestFileMessageRestartLeavesNoPartFile(t *testing.T) {
	dir := withFiles(t, 1024, 0)
	ni := newChunkInstance(t)
	fp, _ := newFilePayload("A::dm:B-misc", []byte("m"), []fileRef{{Key: "k", Path: writeTemp(t, "a.bin", patterned(4000))}})
	broken := fileFrames(t, 3, fp)
	feed(t, ni, peer.ID("alice"), broken[0])
	feed(t, ni, peer.ID("alice"), broken[1])
	for _, f := range fileFrames(t, 3, fp) {
		feed(t, ni, peer.ID("alice"), f)
	}
	popped := ni.messageStore.popNetworkBatch()
	if len(popped) != 1 || len(partFiles(t, dir)) != 1 {
		t.Fatalf("%d messages and %d .part files, want 1 and 1 (the delivered one)", len(popped), len(partFiles(t, dir)))
	}
}

// A file that arrives changed (a byte flipped on the way) fails its hash: nothing is
// delivered and nothing is left on disk.
func TestFileMessageWithAWrongHashIsDropped(t *testing.T) {
	dir := withFiles(t, 1024, 0)
	ni := newChunkInstance(t)
	fp, _ := newFilePayload("A::dm:B-misc", []byte("m"), []fileRef{{Key: "k", Path: writeTemp(t, "a.bin", patterned(3000))}})
	frames := fileFrames(t, 4, fp)
	frames[1][len(frames[1])-1] ^= 0xff
	results := []bool{}
	for _, f := range frames {
		results = append(results, feed(t, ni, peer.ID("alice"), f))
	}
	if results[len(results)-1] {
		t.Fatal("the last chunk of a corrupted file was accepted")
	}
	if popped := ni.messageStore.popNetworkBatch(); len(popped) != 0 || len(partFiles(t, dir)) != 0 {
		t.Fatalf("%d messages and %d .part files after a hash mismatch", len(popped), len(partFiles(t, dir)))
	}
}

// The receiver refuses a file message when it does not take files, or a file above its
// maximum, before it writes anything.
func TestFileMessageRefusedByTheReceiver(t *testing.T) {
	dir := withFiles(t, 1024, 1000)
	ni := newChunkInstance(t)
	fp, _ := newFilePayload("A::dm:B-misc", []byte("m"), []fileRef{{Key: "k", Path: writeTemp(t, "a.bin", patterned(3000))}})
	if feed(t, ni, peer.ID("alice"), fileFrames(t, 5, fp)[0]) {
		t.Fatal("a file above the maximum was accepted")
	}
	maxFileSize, acceptFiles = 0, false
	if feed(t, ni, peer.ID("alice"), fileFrames(t, 6, fp)[0]) {
		t.Fatal("a file message was accepted by a node that takes no files")
	}
	if len(partFiles(t, dir)) != 0 || len(ni.reassembly.byPeer) != 0 {
		t.Fatal("a refused file message left something behind")
	}
}

// A dropped message under reassembly (purge, timeout) removes its .part file.
func TestDroppedFileMessageRemovesItsPartFile(t *testing.T) {
	dir := withFiles(t, 1024, 0)
	ni := newChunkInstance(t)
	fp, _ := newFilePayload("A::dm:B-misc", []byte("m"), []fileRef{{Key: "k", Path: writeTemp(t, "a.bin", patterned(4000))}})
	frames := fileFrames(t, 7, fp)
	feed(t, ni, peer.ID("alice"), frames[0])
	feed(t, ni, peer.ID("alice"), frames[1])
	if len(partFiles(t, dir)) != 1 {
		t.Fatalf("%d .part files while the file arrives, want 1", len(partFiles(t, dir)))
	}
	ni.reassembly.Drop(peer.ID("alice"))
	if len(partFiles(t, dir)) != 0 {
		t.Fatal("the .part file outlived its message")
	}
}

// A file changed after its message was queued is not sent: the writer gets a permanent error.
func TestChangedFileIsNotSent(t *testing.T) {
	withFiles(t, 1024, 0)
	path := writeTemp(t, "a.bin", patterned(3000))
	fp, _ := newFilePayload("A::dm:B-misc", []byte("m"), []fileRef{{Key: "k", Path: path}})
	if err := os.WriteFile(path, patterned(3001), 0o600); err != nil {
		t.Fatal(err)
	}
	err := writeFileMessage(1, fp, func([]byte) error { return nil }, nil)
	var permanent *permanentSendError
	if !errors.As(err, &permanent) || permanent.reason != "file_changed" {
		t.Fatalf("got %v, want a permanent file_changed error", err)
	}
}

// A message the store does not keep (its channel queue full) has its files removed.
func TestEvictedFileMessageRemovesItsFiles(t *testing.T) {
	ni := newTestInstance(t)
	maxChannelQueueLen = 1
	path := writeTemp(t, "x.part", []byte("x"))
	storeReceivedMessageWithFiles(ni, peer.ID("alice"), "A::dm:B-misc", UnaiverseChatProtocol, []byte("1"), []ReceivedFile{{Key: "k", Path: path, Size: 1}})
	storeReceivedMessage(ni, peer.ID("alice"), "A::dm:B-misc", UnaiverseChatProtocol, []byte("2"))
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatal("the file of the evicted message is still there")
	}
}

// A permanent error drops only the message at the head of the queue; the next one is written.
func TestPermanentErrorDropsOnlyTheHead(t *testing.T) {
	f := newFakeTransport()
	store, _ := newTestSendStore(t, testSendConfig(), f)
	store.writeFile = func(pid peer.ID, seq uint64, fp *filePayload) error {
		return &permanentSendError{reason: "file_changed", err: errFileChanged}
	}
	pid := testPeer(t, 0)
	fileSeq, err := store.enqueue(pid, &queuedFrame{file: &filePayload{channel: "c", msg: []byte("m")}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Enqueue(pid, []byte("next")); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "the next message written", func() bool { return len(f.writtenFrames()) == 1 })
	time.Sleep(10 * time.Millisecond)
	events := store.PopEvents()
	dropped := eventsOf(events, SendEventQueueDropped)
	if len(dropped) != 1 || dropped[0].FromSeq != fileSeq || dropped[0].ToSeq != fileSeq || dropped[0].Reason != "file_changed" {
		t.Fatalf("drop events %+v, want one for seq %d alone (file_changed)", dropped, fileSeq)
	}
	if len(eventsOf(events, SendEventWritten)) != 1 {
		t.Fatal("the next message was not reported written")
	}
}

// The throttle: the first and the last report always, in between 1% and the interval both.
func TestProgressThrottle(t *testing.T) {
	prev := progressMinInterval
	progressMinInterval = time.Second
	t.Cleanup(func() { progressMinInterval = prev })
	var th progressThrottle
	t0 := time.Unix(1000, 0)
	for _, c := range []struct {
		at   time.Duration
		done uint64
		want bool
	}{{0, 10, true}, {2 * time.Second, 15, false}, {500 * time.Millisecond, 500, false}, {2 * time.Second, 500, true},
		{2100 * time.Millisecond, 600, false}, {2200 * time.Millisecond, 1000, true}} {
		if got := th.due(t0.Add(c.at), c.done, 1000); got != c.want {
			t.Fatalf("at %v with %d of 1000: due %v, want %v", c.at, c.done, got, c.want)
		}
	}
}

// The progress of a message is one event until it is popped (the latest), and a new attempt
// reports again at once.
func TestSenderProgressIsMergedPerMessage(t *testing.T) {
	store, _ := newTestSendStore(t, testSendConfig(), newFakeTransport())
	now := time.Unix(1000, 0)
	store.now = func() time.Time { return now }
	pid := testPeer(t, 0)
	report := store.progressOf(pid, 3, transferFile, []string{"a.bin"})
	report(100, 1000, 0)
	now = now.Add(time.Second)
	report(600, 1000, 0)
	report(1000, 1000, 0)
	events := eventsOf(store.PopEvents(), SendEventProgress)
	if len(events) != 1 || events[0].Bytes != 1000 || events[0].Total != 1000 || events[0].Seq != 3 ||
		events[0].Kind != transferFile || len(events[0].Files) != 1 {
		t.Fatalf("progress events %+v, want one at 1000 of 1000", events)
	}
	report(10, 1000, 1)
	if events := eventsOf(store.PopEvents(), SendEventProgress); len(events) != 1 || events[0].Attempt != 1 || events[0].Bytes != 10 {
		t.Fatalf("a new attempt reported %+v", events)
	}
}

// The receiver reports the progress of a file message with the names of its files, up to the
// whole message, and why it dropped one.
func TestReceiverReportsProgressAndDrops(t *testing.T) {
	withFiles(t, 1024, 0)
	prev := progressMinInterval
	progressMinInterval = 0
	t.Cleanup(func() { progressMinInterval = prev })
	ni := newChunkInstance(t)
	ni.sendStore, _ = newTestSendStore(t, testSendConfig(), newFakeTransport())
	files := []fileRef{{Key: "k", Path: writeTemp(t, "a.bin", patterned(5000)), Name: "a.bin"}}
	fp, _ := newFilePayload("A::dm:B-misc", []byte("m"), files)
	var seen []*SendEvent
	for _, f := range fileFrames(t, 8, fp) {
		feed(t, ni, peer.ID("alice"), f)
		for _, ev := range eventsOf(ni.sendStore.PopEvents(), SendEventReceiveProgress) {
			if ev.Kind != transferFile || len(ev.Files) != 1 || ev.Files[0] != "a.bin" || ev.Seq != 8 {
				t.Fatalf("progress event %+v", ev)
			}
			seen = append(seen, ev)
		}
	}
	if len(seen) < 3 || seen[len(seen)-1].Bytes != seen[len(seen)-1].Total {
		t.Fatalf("%d progress events, the last at %d of %d", len(seen), seen[len(seen)-1].Bytes, seen[len(seen)-1].Total)
	}
	for i := 1; i < len(seen); i++ {
		if seen[i].Bytes <= seen[i-1].Bytes {
			t.Fatal("the progress went back")
		}
	}
	if popped := ni.messageStore.popNetworkBatch(); len(popped) != 1 {
		t.Fatalf("%d messages delivered, want 1", len(popped))
	}

	frames := fileFrames(t, 9, fp)
	frames[1][len(frames[1])-1] ^= 0xff
	for _, f := range frames {
		feed(t, ni, peer.ID("alice"), f)
	}
	dropped := eventsOf(ni.sendStore.PopEvents(), SendEventReceiveDropped)
	if len(dropped) != 1 || dropped[0].Reason != "hash_mismatch" || dropped[0].Seq != 9 {
		t.Fatalf("drop events %+v, want one for seq 9 (hash_mismatch)", dropped)
	}
}
