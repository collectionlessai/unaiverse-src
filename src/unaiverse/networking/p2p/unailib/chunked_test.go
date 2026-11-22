// Tests of the chunked frames: reassembly on the receiver, its limits, and the pop budget.
package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/peer"
)

// withChunks sets the limits of the chunked frames for one test.
func withChunks(t *testing.T, size int, maxMessage int64) {
	t.Helper()
	prev := []any{chunkSize, chunkedMaxMessageSize, reassemblyMaxBytesPerPeer, reassemblyMaxBytesTotal, reassemblyTimeout}
	chunkSize, chunkedMaxMessageSize, reassemblyMaxBytesPerPeer, reassemblyMaxBytesTotal = size, maxMessage, 0, 0
	reassemblyTimeout = time.Minute
	t.Cleanup(func() {
		chunkSize, chunkedMaxMessageSize = prev[0].(int), prev[1].(int64)
		reassemblyMaxBytesPerPeer, reassemblyMaxBytesTotal = prev[2].(int64), prev[3].(int64)
		reassemblyTimeout = prev[4].(time.Duration)
	})
}

// chunkFrames cuts a payload into the chunk frames of the first attempt of writeChunks.
func chunkFrames(seq uint64, channel string, payload []byte) [][]byte {
	return attemptFrames(seq, 0, channel, payload)
}

// attemptFrames cuts a payload into the chunk frames of the given attempt of writeChunks.
func attemptFrames(seq uint64, attempt uint8, channel string, payload []byte) [][]byte {
	room := chunkSize - chunkFixedHeader
	var frames [][]byte
	for offset, index := 0, uint32(0); ; index++ {
		end := offset + room
		if end > len(payload) {
			end = len(payload)
		}
		frames = append(frames, buildChunkFrame(seq, index, attempt, index == 0, end == len(payload), uint64(len(payload)), channel, payload[offset:end]))
		offset = end
		if end == len(payload) {
			return frames
		}
	}
}

// feed hands a chunk frame to the receiver as readChatStream does; false: stream reset.
func feed(t *testing.T, ni *NodeInstance, from peer.ID, frame []byte) bool {
	t.Helper()
	length := binary.BigEndian.Uint32(frame[0:4])
	if length&chunkFrameFlag == 0 {
		t.Fatalf("not a chunk frame")
	}
	return readChunkFrame(ni, bytes.NewReader(frame[4:]), from, length&^chunkFrameFlag)
}

func newChunkInstance(t *testing.T) *NodeInstance {
	ni := newTestInstance(t)
	ni.reassembly = newReassemblyStore()
	return ni
}

// A message cut into chunks arrives whole, as one message on its channel.
func TestChunkedMessageIsReassembled(t *testing.T) {
	withChunks(t, 1024, 0)
	ni := newChunkInstance(t)
	alice := peer.ID("alice")
	payload := bytes.Repeat([]byte("0123456789"), 300) // 3000 bytes: 3 chunks
	frames := chunkFrames(7, "A::dm:B-misc", payload)
	if len(frames) != 3 {
		t.Fatalf("%d frames, want 3", len(frames))
	}
	for _, f := range frames {
		if !feed(t, ni, alice, f) {
			t.Fatal("a valid chunk was refused")
		}
	}
	popped := ni.messageStore.popNetworkBatch()
	if len(popped) != 1 || !bytes.Equal(popped[0].Data, payload) {
		t.Fatalf("got %d messages, want the whole payload once", len(popped))
	}
	if len(ni.reassembly.byPeer) != 0 || ni.reassembly.reserved != 0 {
		t.Fatal("the reassembly state was not released")
	}
}

// After a broken stream the writer sends the message again from its first chunk, with the
// next attempt: the partial copy is replaced, and only the complete one is delivered, once.
func TestAFirstChunkAgainRestartsTheMessage(t *testing.T) {
	withChunks(t, 1024, 0)
	ni := newChunkInstance(t)
	alice := peer.ID("alice")
	payload := bytes.Repeat([]byte("x"), 2500)
	frames := chunkFrames(3, "A::dm:B-misc", payload)
	feed(t, ni, alice, frames[0])
	feed(t, ni, alice, frames[1])
	for _, f := range attemptFrames(3, 1, "A::dm:B-misc", payload) {
		if !feed(t, ni, alice, f) {
			t.Fatal("the restarted message was refused")
		}
	}
	popped := ni.messageStore.popNetworkBatch()
	if len(popped) != 1 || len(popped[0].Data) != len(payload) {
		t.Fatalf("got %d messages, want one of %d bytes", len(popped), len(payload))
	}
}

// A duplicated chunk is dropped; a chunk out of place resets; one with no FIRST is dropped.
func TestDuplicateAndOutOfPlaceChunks(t *testing.T) {
	withChunks(t, 1024, 0)
	ni := newChunkInstance(t)
	alice := peer.ID("alice")
	frames := chunkFrames(1, "A::dm:B-misc", bytes.Repeat([]byte("y"), 3500))
	feed(t, ni, alice, frames[0])
	feed(t, ni, alice, frames[1])
	if !feed(t, ni, alice, frames[1]) {
		t.Fatal("a duplicate must be dropped, not reset")
	}
	if feed(t, ni, alice, frames[3]) {
		t.Fatal("a chunk out of place was accepted")
	}
	if len(ni.reassembly.byPeer) != 0 {
		t.Fatal("the broken message was kept")
	}
	if !feed(t, ni, peer.ID("bob"), frames[2]) || len(ni.reassembly.byPeer) != 0 {
		t.Fatal("a chunk with no FIRST must be dropped alone")
	}
	if popped := ni.messageStore.popNetworkBatch(); len(popped) != 0 {
		t.Fatalf("%d messages delivered from broken ones", len(popped))
	}
}

// When the DataChannel fails in the middle of a message, the message starts again with the
// next attempt on the chat stream while chunks of the first attempt still arrive on the
// channel: they are dropped, the FIRST one included, and the message is delivered once.
// Attempts wrap around after 31.
func TestStaleChunksOfAnOlderAttemptAreDropped(t *testing.T) {
	withChunks(t, 1024, 0)
	ni := newChunkInstance(t)
	alice := peer.ID("alice")
	payload := bytes.Repeat([]byte("s"), 3500) // 4 chunks
	for _, attempts := range [][2]uint8{{0, 1}, {31, 0}} {
		old, cur := attemptFrames(9, attempts[0], "A::dm:B-misc", payload), attemptFrames(9, attempts[1], "A::dm:B-misc", payload)
		order := [][]byte{old[0], old[1], cur[0], old[2], cur[1], old[0], old[3], cur[2], cur[3]}
		for i, f := range order {
			if !feed(t, ni, alice, f) {
				t.Fatalf("attempts %v: frame %d of the mix reset the message", attempts, i)
			}
		}
		popped := ni.messageStore.popNetworkBatch()
		if len(popped) != 1 || !bytes.Equal(popped[0].Data, payload) {
			t.Fatalf("attempts %v: got %d messages, want the whole payload once", attempts, len(popped))
		}
		if len(ni.reassembly.byPeer) != 0 {
			t.Fatalf("attempts %v: stale chunks left a message under reassembly", attempts)
		}
	}
}

// The order of attempts on the circle of 32.
func TestNewerAttempt(t *testing.T) {
	for _, c := range []struct {
		a, b  uint8
		newer bool
	}{{1, 0, true}, {0, 0, false}, {0, 1, false}, {0, 31, true}, {15, 0, true}, {16, 0, false}, {5, 22, true}, {5, 20, false}} {
		if got := newerAttempt(c.a, c.b); got != c.newer {
			t.Errorf("newerAttempt(%d, %d) = %v, want %v", c.a, c.b, got, c.newer)
		}
	}
}

// writeChunks gives every start of a message the next attempt, and its frames put back
// together are the message.
func TestWriteChunksTakesTheNextAttempt(t *testing.T) {
	withChunks(t, 1024, 0)
	ni := newChunkInstance(t)
	cp := &chunkedPayload{channel: "A::dm:B-misc", payload: bytes.Repeat([]byte("w"), 2100), attempt: 31}
	var frames [][]byte
	collect := func(frame []byte) error { frames = append(frames, frame); return nil }
	if err := writeChunks(4, cp, collect, nil); err != nil || cp.attempt != 0 {
		t.Fatalf("first start: err %v, next attempt %d (want 0 after 31)", err, cp.attempt)
	}
	first := len(frames)
	if err := writeChunks(4, cp, collect, nil); err != nil || cp.attempt != 1 {
		t.Fatalf("second start: err %v, next attempt %d", err, cp.attempt)
	}
	if first != 3 || len(frames) != 6 || frames[0][16]>>chunkAttemptBits != 31 || frames[3][16]>>chunkAttemptBits != 0 {
		t.Fatalf("%d frames per start, attempts %d and %d", first, frames[0][16]>>chunkAttemptBits, frames[3][16]>>chunkAttemptBits)
	}
	for _, f := range frames {
		feed(t, ni, peer.ID("alice"), f)
	}
	if popped := ni.messageStore.popNetworkBatch(); len(popped) != 2 {
		t.Fatalf("got %d messages, want one per complete start", len(popped))
	}
}

// A declared total above the local maximum resets at once, before any data is kept.
func TestTotalAboveTheLimitResets(t *testing.T) {
	withChunks(t, 1024, 2000)
	ni := newChunkInstance(t)
	frames := chunkFrames(1, "A::dm:B-misc", bytes.Repeat([]byte("z"), 5000))
	if feed(t, ni, peer.ID("alice"), frames[0]) {
		t.Fatal("a message above the limit was accepted")
	}
	if ni.reassembly.reserved != 0 {
		t.Fatal("bytes were reserved for a refused message")
	}
}

// A message whose next chunk does not come in time is forgotten by the sweeper.
func TestStalledMessageTimesOut(t *testing.T) {
	withChunks(t, 1024, 0)
	reassemblyTimeout = 50 * time.Millisecond
	ni := newChunkInstance(t)
	frames := chunkFrames(1, "A::dm:B-misc", bytes.Repeat([]byte("w"), 3000))
	feed(t, ni, peer.ID("alice"), frames[0])
	ni.reassembly.mu.Lock()
	ni.reassembly.byPeer[peer.ID("alice")].lastSeen = time.Now().Add(-time.Second)
	ni.reassembly.mu.Unlock()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go ni.reassembly.sweep(ctx, ni)
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		ni.reassembly.mu.Lock()
		n := len(ni.reassembly.byPeer)
		ni.reassembly.mu.Unlock()
		if n == 0 {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatal("the stalled message was not forgotten")
}

// A pop stops adding messages past its byte budget, and always takes at least one.
func TestPopStopsAtItsByteBudget(t *testing.T) {
	ni := newChunkInstance(t)
	prev := popBatchMaxBytes
	popBatchMaxBytes = 150
	t.Cleanup(func() { popBatchMaxBytes = prev })
	for i := 0; i < 3; i++ {
		storeReceivedMessage(ni, peer.ID("alice"), "A::dm:B-misc", "", bytes.Repeat([]byte("m"), 100))
	}
	for i, want := range []int{1, 1, 1, 0} {
		if got := len(ni.messageStore.popNetworkBatch()); got != want {
			t.Fatalf("pop %d gave %d messages, want %d", i, got, want)
		}
	}
}

// Who gets chunks: a peer that said so, for a message larger than a chunk, within the
// smallest of the two maximums.
func TestWhoGetsChunks(t *testing.T) {
	withChunks(t, 1024, 5000)
	legacy := peerLimits{caps: 0, maxMessageSize: -1}
	modern := peerLimits{caps: capChunkedFrames, maxMessageSize: 3000}
	if usesChunks(legacy, "c", 4000) || usesChunks(modern, "c", 500) || !usesChunks(modern, "c", 4000) {
		t.Fatal("wrong choice of frames")
	}
	if chunkedLimitFor(modern) != 3000 || chunkedLimitFor(peerLimits{caps: capChunkedFrames, maxMessageSize: 0}) != 5000 {
		t.Fatal("wrong limit")
	}
}
