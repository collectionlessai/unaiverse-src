// Tests of the send queues of the direct messages: the caps and the low-water mark, the
// order, the retry of a frame whose write failed, the drops of a whole queue, the end of
// the writers, the events, and the write deadline of a chat frame.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
)

// testSendConfig is a configuration with room for everything, that a test narrows.
func testSendConfig() SendQueueConfig {
	return SendQueueConfig{
		MaxBytesPerPeer: 1 << 20,
		MaxMsgsPerPeer:  64,
		MaxBytesTotal:   4 << 20,
		LowWater:        0.5,
		MaxWaitAge:      10 * time.Second,
		MaxDialAttempts: 3,
		RetryBackoff:    []time.Duration{time.Millisecond},
		MaxEvents:       1000,
	}
}

// fakeTransport is the transport of a store under test. A write takes the next result of
// the script (nil when the script is over) and, while hold is set, waits for a release
// first. It records every frame written, including those that failed, and how many
// writers are in a write at once.
type fakeTransport struct {
	mu       sync.Mutex
	script   []error
	hold     bool
	release  chan struct{}
	started  chan string
	attempts []string
	written  []string
	inWrite  int
	maxIn    int
	duration time.Duration
}

func newFakeTransport() *fakeTransport {
	return &fakeTransport{release: make(chan struct{}, 1000), started: make(chan string, 1000)}
}

func (f *fakeTransport) write(pid peer.ID, frame []byte) error {
	f.mu.Lock()
	f.inWrite++
	if f.inWrite > f.maxIn {
		f.maxIn = f.inWrite
	}
	hold, duration := f.hold, f.duration
	f.mu.Unlock()
	f.started <- string(frame)
	if hold {
		<-f.release
	}
	if duration > 0 {
		time.Sleep(duration)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.inWrite--
	f.attempts = append(f.attempts, string(frame))
	var err error
	if len(f.script) > 0 {
		err, f.script = f.script[0], f.script[1:]
	}
	if err == nil {
		f.written = append(f.written, string(frame))
	}
	return err
}

func (f *fakeTransport) writtenFrames() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.written...)
}

func (f *fakeTransport) attemptedFrames() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.attempts...)
}

// waitStarted waits for the next write to start and returns its frame.
func (f *fakeTransport) waitStarted(t *testing.T) string {
	t.Helper()
	select {
	case frame := <-f.started:
		return frame
	case <-time.After(5 * time.Second):
		t.Fatal("no write started")
		return ""
	}
}

// newTestSendStore returns a store on the fake transport, whose writers stop at the end
// of the test.
func newTestSendStore(t *testing.T, cfg SendQueueConfig, f *fakeTransport) (*SendStore, context.CancelFunc) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(func() {
		cancel()

		// A writer held by the fake must be let go, or it outlives the test.
		f.mu.Lock()
		f.hold = false
		f.mu.Unlock()
		for i := 0; i < 100; i++ {
			select {
			case f.release <- struct{}{}:
			default:
			}
		}
	})
	return newSendStore(ctx, cfg, f.write), cancel
}

// testPeer returns a peer ID that is valid, the n-th of a test.
func testPeer(t *testing.T, n int) peer.ID {
	t.Helper()
	ids := []string{
		"12D3KooWEXse6zf6g6rsFckJqXA6SWWrENNB3oN8tG5qPEPWV5do",
		"12D3KooWKSQXVsqfSDJ1MrAD3b4FMCbtAaqKusKKsUXY2QNn2yr6",
	}
	pid, err := peer.Decode(ids[n])
	if err != nil {
		t.Fatalf("peer id: %v", err)
	}
	return pid
}

// mustEnqueue queues a frame and fails the test when it is refused.
func mustEnqueue(t *testing.T, store *SendStore, pid peer.ID, frame string) uint64 {
	t.Helper()
	seq, err := store.Enqueue(pid, []byte(frame))
	if err != nil {
		t.Fatalf("frame %q refused: %v", frame, err)
	}
	return seq
}

// refusedFor queues a frame that must be refused and returns the reason.
func refusedFor(t *testing.T, store *SendStore, pid peer.ID, frame string) string {
	t.Helper()
	_, err := store.Enqueue(pid, []byte(frame))
	refusal, ok := err.(*sendRefusal)
	if !ok {
		t.Fatalf("frame %q: want a refusal, got %v", frame, err)
	}
	return refusal.reason
}

// waitFor polls a condition, for the outcomes of the writer goroutines.
func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(time.Millisecond)
	}
}

// queueState returns the length of the queue of the peer and whether a writer runs.
func queueState(store *SendStore, pid peer.ID) (int, bool) {
	store.mu.Lock()
	defer store.mu.Unlock()
	q := store.queues[pid]
	if q == nil {
		return 0, false
	}
	return q.frames.Len(), q.running
}

// eventsOf returns the events of the given type among those popped.
func eventsOf(events []*SendEvent, typ string) []*SendEvent {
	var out []*SendEvent
	for _, ev := range events {
		if ev.Type == typ {
			out = append(out, ev)
		}
	}
	return out
}

func sameFrames(t *testing.T, what string, got []string, want ...string) {
	t.Helper()
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("%s: got %v, want %v", what, got, want)
	}
}

func TestSendQueueRefusesAtTheMessageCapOfThePeer(t *testing.T) {
	f := newFakeTransport()
	f.hold = true
	cfg := testSendConfig()
	cfg.MaxMsgsPerPeer = 3
	store, _ := newTestSendStore(t, cfg, f)
	pid := testPeer(t, 0)

	for i, frame := range []string{"a", "b", "c"} {
		if seq := mustEnqueue(t, store, pid, frame); seq != uint64(i+1) {
			t.Fatalf("frame %q got seq %d, want %d", frame, seq, i+1)
		}
	}
	f.waitStarted(t)
	if reason := refusedFor(t, store, pid, "d"); reason != SendRefusedPeerMsgs {
		t.Fatalf("refused for %q, want %q", reason, SendRefusedPeerMsgs)
	}

	// What was queued is untouched by the refusal.
	for i := 0; i < 3; i++ {
		f.release <- struct{}{}
	}
	waitFor(t, "the queue to drain", func() bool { n, _ := queueState(store, pid); return n == 0 })
	sameFrames(t, "written", f.writtenFrames(), "a", "b", "c")
}

func TestSendQueueRefusesAtTheByteCapOfThePeer(t *testing.T) {
	f := newFakeTransport()
	f.hold = true
	cfg := testSendConfig()
	cfg.MaxBytesPerPeer = 10
	store, _ := newTestSendStore(t, cfg, f)
	pid := testPeer(t, 0)

	mustEnqueue(t, store, pid, "123456")
	mustEnqueue(t, store, pid, "7890")
	if reason := refusedFor(t, store, pid, "x"); reason != SendRefusedPeerBytes {
		t.Fatalf("refused for %q, want %q", reason, SendRefusedPeerBytes)
	}
	if reason := refusedFor(t, store, pid, "12345678901"); reason != SendRefusedTooLarge {
		t.Fatalf("a frame larger than the cap was refused for %q, want %q", reason, SendRefusedTooLarge)
	}
}

func TestSendQueueRefusesAtTheByteCapOfAllThePeers(t *testing.T) {
	f := newFakeTransport()
	f.hold = true
	cfg := testSendConfig()
	cfg.MaxBytesPerPeer = 10
	cfg.MaxBytesTotal = 15
	store, _ := newTestSendStore(t, cfg, f)
	a, b := testPeer(t, 0), testPeer(t, 1)

	mustEnqueue(t, store, a, "1234567890")
	mustEnqueue(t, store, b, "12345")
	if reason := refusedFor(t, store, b, "6"); reason != SendRefusedTotalBytes {
		t.Fatalf("refused for %q, want %q", reason, SendRefusedTotalBytes)
	}

	// Once the global cap refused, every peer waits for the low-water mark of the global
	// cap (7 bytes here): peer a drains its 10 bytes and the total falls to 5.
	f.release <- struct{}{}
	f.release <- struct{}{}
	waitFor(t, "the queues to drain", func() bool {
		na, _ := queueState(store, a)
		nb, _ := queueState(store, b)
		return na == 0 && nb == 0
	})
	mustEnqueue(t, store, b, "6")
}

func TestSendQueueRefusesUntilTheLowWaterMark(t *testing.T) {
	f := newFakeTransport()
	f.hold = true
	cfg := testSendConfig()
	cfg.MaxMsgsPerPeer = 4
	cfg.LowWater = 0.5
	store, _ := newTestSendStore(t, cfg, f)
	pid := testPeer(t, 0)

	for _, frame := range []string{"1", "2", "3", "4"} {
		mustEnqueue(t, store, pid, frame)
	}
	if reason := refusedFor(t, store, pid, "x"); reason != SendRefusedPeerMsgs {
		t.Fatalf("refused for %q, want %q", reason, SendRefusedPeerMsgs)
	}

	// Three left: there is room under the cap, but not yet down to the low-water mark.
	f.release <- struct{}{}
	waitFor(t, "one write", func() bool { n, _ := queueState(store, pid); return n == 3 })
	if reason := refusedFor(t, store, pid, "x"); reason != SendRefusedLowWater {
		t.Fatalf("refused for %q, want %q", reason, SendRefusedLowWater)
	}

	// Two left: half of the cap, accepted again.
	f.release <- struct{}{}
	waitFor(t, "two writes", func() bool { n, _ := queueState(store, pid); return n == 2 })
	if seq := mustEnqueue(t, store, pid, "5"); seq != 5 {
		t.Fatalf("seq %d, want 5: refused frames take no number", seq)
	}

	events := store.PopEvents()
	refused := eventsOf(events, SendEventRefused)
	if len(refused) != 2 || refused[0].Count != 1 || refused[1].Count != 1 {
		t.Fatalf("refused events %+v, want two of one frame each (a written event lies between)", refused)
	}
}

func TestSendQueueWritesInOrderWithOneWriterPerPeer(t *testing.T) {
	f := newFakeTransport()
	f.duration = 100 * time.Microsecond
	store, _ := newTestSendStore(t, testSendConfig(), f)
	pid := testPeer(t, 0)

	var want []string
	var wg sync.WaitGroup
	var mu sync.Mutex
	seqs := make(map[uint64]string)
	for i := 0; i < 50; i++ {
		frame := fmt.Sprintf("%02d", i)
		want = append(want, frame)
		seq := mustEnqueue(t, store, pid, frame)
		if seq != uint64(i+1) {
			t.Fatalf("frame %s got seq %d", frame, seq)
		}

		// Queue from other goroutines too, now and then: still one writer.
		if i%10 == 0 {
			wg.Add(1)
			go func() {
				defer wg.Done()
				mu.Lock()
				defer mu.Unlock()
				if s, err := store.Enqueue(testPeer(t, 1), []byte("other")); err == nil {
					seqs[s] = "other"
				}
			}()
		}
	}
	wg.Wait()
	waitFor(t, "the queue to drain", func() bool { n, running := queueState(store, pid); return n == 0 && !running })

	var got []string
	for _, frame := range f.writtenFrames() {
		if frame != "other" {
			got = append(got, frame)
		}
	}
	sameFrames(t, "written", got, want...)
	if f.maxIn > 2 {
		t.Fatalf("%d writes at once, want at most one per peer", f.maxIn)
	}

	written := eventsOf(store.PopEvents(), SendEventWritten)
	last := uint64(0)
	for _, ev := range written {
		if ev.PeerID == pid.String() {
			last = ev.Seq
		}
	}
	if last != 50 {
		t.Fatalf("written up to %d, want 50", last)
	}
}

func TestSendQueueWritesAFailedFrameAgainBeforeTheNextOnes(t *testing.T) {
	f := newFakeTransport()
	f.script = []error{fmt.Errorf("stream reset"), fmt.Errorf("dial failed")}
	store, _ := newTestSendStore(t, testSendConfig(), f)
	pid := testPeer(t, 0)

	mustEnqueue(t, store, pid, "a")
	mustEnqueue(t, store, pid, "b")
	waitFor(t, "the queue to drain", func() bool { n, _ := queueState(store, pid); return n == 0 })

	sameFrames(t, "attempts", f.attemptedFrames(), "a", "a", "a", "b")
	sameFrames(t, "written", f.writtenFrames(), "a", "b")

	var types []string
	for _, ev := range store.PopEvents() {
		types = append(types, fmt.Sprintf("%s:%d", ev.Type, ev.Seq))
	}
	sameFrames(t, "events", types, "write_failed:1", "write_failed:1", "recovered:1", "written:2")
}

func TestSendQueueIsDroppedWholeAfterTheMaxAttempts(t *testing.T) {
	f := newFakeTransport()
	f.script = []error{fmt.Errorf("e1"), fmt.Errorf("e2"), fmt.Errorf("e3")}
	f.hold = true
	cfg := testSendConfig()
	cfg.MaxDialAttempts = 3
	store, _ := newTestSendStore(t, cfg, f)
	pid := testPeer(t, 0)

	mustEnqueue(t, store, pid, "a")
	mustEnqueue(t, store, pid, "b")
	mustEnqueue(t, store, pid, "c")
	for i := 0; i < 3; i++ {
		f.waitStarted(t)
		f.release <- struct{}{}
	}
	waitFor(t, "the writer to end", func() bool { n, running := queueState(store, pid); return n == 0 && !running })

	dropped := eventsOf(store.PopEvents(), SendEventQueueDropped)
	if len(dropped) != 1 {
		t.Fatalf("%d drops, want 1", len(dropped))
	}
	d := dropped[0]
	if d.Reason != SendDroppedMaxAttempts || d.FromSeq != 1 || d.ToSeq != 3 || d.Count != 3 || d.Bytes != 3 || d.Error != "e3" {
		t.Fatalf("drop %+v, want max_attempts of seq 1..3 (3 frames, 3 bytes) with the last error", d)
	}
	sameFrames(t, "written", f.writtenFrames())

	// The queue takes frames again, with the next numbers and a new writer.
	f.mu.Lock()
	f.hold = false
	f.mu.Unlock()
	if seq := mustEnqueue(t, store, pid, "d"); seq != 4 {
		t.Fatalf("seq %d after the drop, want 4", seq)
	}
	waitFor(t, "the new frame", func() bool { return len(f.writtenFrames()) == 1 })
	sameFrames(t, "written", f.writtenFrames(), "d")
}

func TestSendQueueIsDroppedWholeWhenItsHeadWaitedTooLong(t *testing.T) {
	f := newFakeTransport()
	f.script = []error{fmt.Errorf("e1"), fmt.Errorf("e2")}
	cfg := testSendConfig()
	cfg.MaxDialAttempts = 100
	cfg.MaxWaitAge = 150 * time.Millisecond
	cfg.RetryBackoff = []time.Duration{time.Hour}
	store, _ := newTestSendStore(t, cfg, f)
	pid := testPeer(t, 0)

	start := time.Now()
	mustEnqueue(t, store, pid, "a")
	mustEnqueue(t, store, pid, "b")
	waitFor(t, "the writer to end", func() bool { n, running := queueState(store, pid); return n == 0 && !running })
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Fatalf("dropped after %v: the backoff was not cut at the age limit", elapsed)
	}
	dropped := eventsOf(store.PopEvents(), SendEventQueueDropped)
	if len(dropped) != 1 || dropped[0].Reason != SendDroppedMaxAge || dropped[0].FromSeq != 1 || dropped[0].ToSeq != 2 {
		t.Fatalf("drops %+v, want one max_age of seq 1..2", dropped)
	}
}

func TestSendQueueDoesNotExpireAFrameWhoseWriteIsInProgress(t *testing.T) {
	f := newFakeTransport()
	f.duration = 300 * time.Millisecond
	cfg := testSendConfig()
	cfg.MaxWaitAge = 100 * time.Millisecond
	store, _ := newTestSendStore(t, cfg, f)
	pid := testPeer(t, 0)

	// Each write takes three times the age limit: the queue progresses, so nothing expires.
	mustEnqueue(t, store, pid, "a")
	mustEnqueue(t, store, pid, "b")
	waitFor(t, "the writer to end", func() bool { n, running := queueState(store, pid); return n == 0 && !running })
	sameFrames(t, "written", f.writtenFrames(), "a", "b")
	if dropped := eventsOf(store.PopEvents(), SendEventQueueDropped); len(dropped) != 0 {
		t.Fatalf("drops %+v, want none", dropped)
	}
}

func TestDropSendQueueEmptiesItAtOnce(t *testing.T) {
	f := newFakeTransport()
	f.hold = true
	store, _ := newTestSendStore(t, testSendConfig(), f)
	pid := testPeer(t, 0)

	mustEnqueue(t, store, pid, "a")
	mustEnqueue(t, store, pid, "b")
	mustEnqueue(t, store, pid, "c")
	f.waitStarted(t)
	if n := store.Drop(pid); n != 3 {
		t.Fatalf("dropped %d, want 3", n)
	}
	if n, _ := queueState(store, pid); n != 0 {
		t.Fatalf("%d frames left after the drop", n)
	}

	// The write in progress ends; its outcome is not reported, and the writer ends.
	f.release <- struct{}{}
	waitFor(t, "the writer to end", func() bool { _, running := queueState(store, pid); return !running })
	events := store.PopEvents()
	if len(eventsOf(events, SendEventWritten)) != 0 {
		t.Fatalf("events %+v: a frame of a dropped queue was reported as written", events)
	}
	dropped := eventsOf(events, SendEventQueueDropped)
	if len(dropped) != 1 || dropped[0].Reason != SendDroppedExplicit || dropped[0].FromSeq != 1 || dropped[0].ToSeq != 3 {
		t.Fatalf("drops %+v, want one explicit of seq 1..3", dropped)
	}
	store.mu.Lock()
	total := store.totalBytes
	store.mu.Unlock()
	if total != 0 {
		t.Fatalf("%d bytes still counted after the drop", total)
	}
}

func TestDropSendQueueCutsTheBackoffShort(t *testing.T) {
	f := newFakeTransport()
	f.script = []error{fmt.Errorf("e1")}
	cfg := testSendConfig()
	cfg.RetryBackoff = []time.Duration{time.Hour}
	store, _ := newTestSendStore(t, cfg, f)
	pid := testPeer(t, 0)

	mustEnqueue(t, store, pid, "a")
	waitFor(t, "the failed write", func() bool { return len(f.attemptedFrames()) == 1 })
	store.Drop(pid)
	waitFor(t, "the writer to end", func() bool { _, running := queueState(store, pid); return !running })
}

func TestSendWriterEndsWhenTheInstanceCloses(t *testing.T) {
	f := newFakeTransport()
	f.script = []error{fmt.Errorf("e1")}
	cfg := testSendConfig()
	cfg.RetryBackoff = []time.Duration{time.Hour}
	store, cancel := newTestSendStore(t, cfg, f)
	pid := testPeer(t, 0)

	mustEnqueue(t, store, pid, "a")
	waitFor(t, "the failed write", func() bool { return len(f.attemptedFrames()) == 1 })
	cancel()
	waitFor(t, "the writer to end", func() bool { _, running := queueState(store, pid); return !running })
	if attempts := f.attemptedFrames(); len(attempts) != 1 {
		t.Fatalf("%d attempts, want 1: nothing is written after the close", len(attempts))
	}
}

func TestSendEventsAreMergedAndBounded(t *testing.T) {
	f := newFakeTransport()
	f.hold = true
	cfg := testSendConfig()
	cfg.MaxMsgsPerPeer = 1
	cfg.MaxEvents = 2
	store, _ := newTestSendStore(t, cfg, f)
	a, b := testPeer(t, 0), testPeer(t, 1)

	mustEnqueue(t, store, a, "a1")
	for i := 0; i < 5; i++ {
		refusedFor(t, store, a, "x")
	}
	events := store.PopEvents()
	if len(events) != 1 || events[0].Type != SendEventRefused || events[0].Count != 5 || events[0].Bytes != 5 {
		t.Fatalf("events %+v, want one refused event of 5 frames", events)
	}

	// Three events of different peers and types in a buffer of two: the oldest goes.
	mustEnqueue(t, store, b, "b1")
	refusedFor(t, store, b, "x")
	f.release <- struct{}{}
	f.release <- struct{}{}
	waitFor(t, "the queues to drain", func() bool {
		na, _ := queueState(store, a)
		nb, _ := queueState(store, b)
		return na == 0 && nb == 0
	})
	refusedFor(t, store, a, strings.Repeat("x", 1<<20+1))
	events = store.PopEvents()
	if len(events) != 3 || events[0].Type != SendEventEventsLost || events[0].Count < 1 {
		t.Fatalf("events %+v, want events_lost first and the two newest", events)
	}
	if out, err := json.Marshal(events); err != nil || !strings.Contains(string(out), `"type":"events_lost"`) {
		t.Fatalf("events JSON %s: %v", out, err)
	}
}

// The deadline of a chat frame: toward a peer that does not read, a frame larger than the
// stream window waits until the deadline, and the send of the queue then fails and closes
// the stream, instead of waiting with no end.
func TestChatFrameWriteIsBoundedByItsDeadline(t *testing.T) {
	prevBase, prevRate := chatWriteBaseTimeout, chatWriteMinBytesPerSec
	chatWriteBaseTimeout, chatWriteMinBytesPerSec = 300*time.Millisecond, 1<<40
	t.Cleanup(func() { chatWriteBaseTimeout, chatWriteMinBytesPerSec = prevBase, prevRate })
	prevMax := MaxMessageSize
	MaxMessageSize = 64 << 20
	t.Cleanup(func() { MaxMessageSize = prevMax })

	alice := newChatNode(t)
	stalled := newLoopbackHost(t)
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	stalled.SetStreamHandler(UnaiverseChatProtocol, func(s network.Stream) {
		<-release
		s.Reset()
	})
	directLink(t, alice.host, stalled)

	start := time.Now()
	done := make(chan error, 1)
	go func() {
		_, err := sendOnChatStream(alice, stalled.ID(), chatTestChannel, make([]byte, 16<<20))
		done <- err
	}()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("a frame nobody read was reported as written")
		}
	case <-time.After(10 * time.Second):
		t.Fatal("the write is still waiting: no deadline")
	}
	if elapsed := time.Since(start); elapsed > 3*time.Second {
		t.Fatalf("the write failed after %v, the deadline is %v", elapsed, chatWriteBaseTimeout)
	}
	if chatEntry(alice, stalled.ID()) != nil {
		t.Fatal("the stream whose write failed is still the chat stream of the peer")
	}
}

func TestChatWriteTimeoutScalesWithTheFrameSize(t *testing.T) {
	prevBase, prevRate := chatWriteBaseTimeout, chatWriteMinBytesPerSec
	t.Cleanup(func() { chatWriteBaseTimeout, chatWriteMinBytesPerSec = prevBase, prevRate })

	chatWriteBaseTimeout, chatWriteMinBytesPerSec = 10*time.Second, 1<<20
	if got := chatWriteTimeout(100); got < 10*time.Second || got > 10*time.Second+time.Millisecond {
		t.Fatalf("small frame: %v, want about 10s", got)
	}
	if got := chatWriteTimeout(50 << 20); got != 60*time.Second {
		t.Fatalf("50 MiB frame: %v, want 60s", got)
	}
	chatWriteBaseTimeout = 0
	if got := chatWriteTimeout(50 << 20); got != 0 {
		t.Fatalf("not configured: %v, want no deadline", got)
	}
}

// End to end on two hosts: the queued messages reach the peer in order, over the chat
// stream that the writer dials.
func TestQueuedDirectMessagesReachThePeerInOrder(t *testing.T) {
	prevMax := MaxMessageSize
	MaxMessageSize = 8 << 20
	t.Cleanup(func() { MaxMessageSize = prevMax })
	prevBase, prevRate := chatWriteBaseTimeout, chatWriteMinBytesPerSec
	chatWriteBaseTimeout, chatWriteMinBytesPerSec = 10*time.Second, 1<<20
	t.Cleanup(func() { chatWriteBaseTimeout, chatWriteMinBytesPerSec = prevBase, prevRate })
	prevChannels, prevQueueLen := maxUniqueChannels, maxChannelQueueLen
	maxUniqueChannels, maxChannelQueueLen = 64, 64
	t.Cleanup(func() { maxUniqueChannels, maxChannelQueueLen = prevChannels, prevQueueLen })

	alice, bob := newChatNode(t), newChatNode(t)
	alice.sendStore = newSendStore(alice.ctx, testSendConfig(), func(pid peer.ID, frame []byte) error {
		return writeQueuedFrame(alice, pid, frame)
	})
	directLink(t, alice.host, bob.host)

	if _, code, _, _ := queueDirectMessage(alice, alice.host.ID(), chatTestChannel, []byte("me")); code != SendCodeSelfSend {
		t.Fatalf("send to self: code %q, want %q", code, SendCodeSelfSend)
	}
	var want []string
	for i := 1; i <= 20; i++ {
		text := fmt.Sprintf("m%02d", i)
		want = append(want, text)
		seq, code, message, err := queueDirectMessage(alice, bob.host.ID(), chatTestChannel, []byte(text))
		if code != "" || seq != uint64(i) {
			t.Fatalf("message %s: seq %d, code %q: %s: %v", text, seq, code, message, err)
		}
	}
	waitQueued(t, bob, chatTestChannel, want...)
}
