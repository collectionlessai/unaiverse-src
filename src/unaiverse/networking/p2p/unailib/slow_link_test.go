// Tests of the writes on a slow link: the stall deadline of a chat frame, and the size of the chunks written.
package main

import (
	"bytes"
	"errors"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/peer"
)

// slowWriter takes a write after delay, unless its write deadline comes first: a stream whose link drains slowly
// (delay > 0), or not at all (stuck).
type slowWriter struct {
	mu       sync.Mutex
	delay    time.Duration
	stuck    bool
	deadline time.Time
	got      bytes.Buffer
	writes   int
}

// SetWriteDeadline sets the deadline of the writes to come (zero: none).
func (w *slowWriter) SetWriteDeadline(t time.Time) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.deadline = t
	return nil
}

// Write takes p after the delay, or fails with os.ErrDeadlineExceeded when the deadline comes first.
func (w *slowWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	deadline, delay, stuck := w.deadline, w.delay, w.stuck
	w.mu.Unlock()
	wait := delay
	if stuck {
		wait = time.Hour
	}
	if !deadline.IsZero() && time.Until(deadline) < wait {
		time.Sleep(max(time.Until(deadline), 0))
		return 0, os.ErrDeadlineExceeded
	}
	time.Sleep(wait)
	w.mu.Lock()
	defer w.mu.Unlock()
	w.writes++
	w.got.Write(p)
	return len(p), nil
}

// withWriteDeadlines sets the write deadlines of the chat frames for one test.
func withWriteDeadlines(t *testing.T, stall, base time.Duration, rate int64) {
	t.Helper()
	prevStall, prevBase, prevRate := chatWriteStallTimeout, chatWriteBaseTimeout, chatWriteMinBytesPerSec
	chatWriteStallTimeout, chatWriteBaseTimeout, chatWriteMinBytesPerSec = stall, base, rate
	t.Cleanup(func() {
		chatWriteStallTimeout, chatWriteBaseTimeout, chatWriteMinBytesPerSec = prevStall, prevBase, prevRate
	})
}

func TestASlowLinkThatKeepsDrainingIsWaitedFor(t *testing.T) {
	// 1 MiB in pieces of 256 KiB, one every 60 ms: about 240 ms in all, with a stall deadline of 150 ms. The size rule
	// with the same base (150 ms) and a rate of 1 MiB/s would have given the frame 1.15 s; a rate of 64 MiB/s, 165 ms
	withWriteDeadlines(t, 150*time.Millisecond, 0, 0)
	w := &slowWriter{delay: 60 * time.Millisecond}
	frame := bytes.Repeat([]byte{7}, 1<<20)
	started := time.Now()
	if err := writeChatFrame(nil, w, frame); err != nil {
		t.Fatalf("a slow link that keeps draining failed: %v", err)
	}
	if took := time.Since(started); took < 150*time.Millisecond {
		t.Fatalf("the frame took %v: not longer than the stall deadline, the test proves nothing", took)
	}
	if !bytes.Equal(w.got.Bytes(), frame) || w.writes != 4 {
		t.Fatalf("got %d bytes in %d writes, want the frame in 4 pieces", w.got.Len(), w.writes)
	}
	if !w.deadline.IsZero() {
		t.Fatal("the stream kept the deadline of the frame")
	}
}

func TestALinkThatTakesNothingFailsAfterTheStallDeadline(t *testing.T) {
	withWriteDeadlines(t, 100*time.Millisecond, 0, 0)
	w := &slowWriter{stuck: true}
	started := time.Now()
	err := writeChatFrame(nil, w, bytes.Repeat([]byte{7}, 1<<20))
	if !errors.Is(err, os.ErrDeadlineExceeded) {
		t.Fatalf("got %v, want the deadline error", err)
	}
	if took := time.Since(started); took > 2*time.Second {
		t.Fatalf("it failed after %v, long after the stall deadline", took)
	}
}

func TestWithoutAStallDeadlineTheSizeRuleHolds(t *testing.T) {
	// The older configuration (no write_stall_timeout_ms): one deadline for the whole frame, its base plus its size
	// at the rate. 1 MiB in 4 pieces of 60 ms does not fit in 100 ms + 1 MiB at 64 MiB/s (about 116 ms)
	withWriteDeadlines(t, 0, 100*time.Millisecond, 64<<20)
	w := &slowWriter{delay: 60 * time.Millisecond}
	frame := bytes.Repeat([]byte{7}, 1<<20)
	if err := writeChatFrame(nil, w, frame); err != nil {
		t.Fatalf("one write of the whole frame within its deadline failed: %v", err)
	}
	if w.writes != 1 {
		t.Fatalf("got %d writes, want the frame written at once", w.writes)
	}
}

func TestTheChunksWrittenHaveTheSendChunkSize(t *testing.T) {
	withChunks(t, 64*1024, 0)
	prev := sendChunkSize
	t.Cleanup(func() { sendChunkSize = prev })

	payload := bytes.Repeat([]byte{1, 2, 3}, 40*1024) // 120 KiB
	count := func() int {
		n := 0
		cp := &chunkedPayload{channel: "c", payload: payload}
		if err := writeChunks(1, cp, func([]byte) error { n++; return nil }, nil); err != nil {
			t.Fatal(err)
		}
		return n
	}
	sendChunkSize = 0
	if n := count(); n != 2 {
		t.Fatalf("without a send chunk size: %d frames, want 2 (chunks of 64 KiB)", n)
	}
	sendChunkSize = 16 * 1024
	if n := count(); n != 8 {
		t.Fatalf("with a send chunk size of 16 KiB: %d frames, want 8", n)
	}
	sendChunkSize = 1 << 20
	if n := count(); n != 2 {
		t.Fatalf("a send chunk size above the chunk size is not used: %d frames, want 2", n)
	}
}

func TestSmallerChunksAreReassembled(t *testing.T) {
	// A receiver takes chunks smaller than its chunk size: the sender's own (a browser sends chunks of 1 MiB)
	withChunks(t, 64*1024, 0)
	prev := sendChunkSize
	t.Cleanup(func() { sendChunkSize = prev })
	sendChunkSize = 16 * 1024

	ni := newChunkInstance(t)
	payload := bytes.Repeat([]byte{4, 5, 6, 7}, 30*1024)
	cp := &chunkedPayload{channel: "A::dm:B-misc", payload: payload}
	var frames [][]byte
	if err := writeChunks(9, cp, func(f []byte) error { frames = append(frames, append([]byte(nil), f...)); return nil }, nil); err != nil {
		t.Fatal(err)
	}
	if len(frames) != 8 {
		t.Fatalf("%d frames, want 8 chunks of 16 KiB", len(frames))
	}
	for _, f := range frames {
		if !feed(t, ni, peer.ID("alice"), f) {
			t.Fatal("a smaller chunk was refused")
		}
	}
	popped := ni.messageStore.popNetworkBatch()
	if len(popped) != 1 || !bytes.Equal(popped[0].Data, payload) {
		t.Fatalf("got %d messages, want the whole payload once", len(popped))
	}
}
