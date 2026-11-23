package main

// The receiver stops a message (its Session's STOP_SENDING): StopMessage takes a waiting
// message out of the queue, and stops the one being written before its next chunk; both are
// dropped alone, with reason "receiver_stopped", and the queue goes on with the next ones.

import (
	"fmt"
	"testing"

	"github.com/libp2p/go-libp2p/core/peer"
)

func TestStopMessageTakesAWaitingMessageOutOfTheQueue(t *testing.T) {
	f := newFakeTransport()
	f.hold = true
	store, _ := newTestSendStore(t, testSendConfig(), f)
	pid := testPeer(t, 0)

	mustEnqueue(t, store, pid, "a")
	seqB := mustEnqueue(t, store, pid, "b")
	mustEnqueue(t, store, pid, "c")
	f.waitStarted(t) // "a" is being written
	if !store.StopMessage(pid, seqB) {
		t.Fatal("StopMessage of a waiting message said it was not there")
	}

	f.mu.Lock()
	f.hold = false
	f.mu.Unlock()
	f.release <- struct{}{}
	waitFor(t, "the queue to empty", func() bool { n, running := queueState(store, pid); return n == 0 && !running })
	sameFrames(t, "written", f.writtenFrames(), "a", "c")

	dropped := eventsOf(store.PopEvents(), SendEventQueueDropped)
	if len(dropped) != 1 || dropped[0].Reason != SendDroppedReceiverStopped || dropped[0].FromSeq != seqB || dropped[0].ToSeq != seqB || dropped[0].Count != 1 {
		t.Fatalf("drops %+v, want one receiver_stopped of seq %d", dropped, seqB)
	}
	store.mu.Lock()
	total := store.totalBytes
	store.mu.Unlock()
	if total != 0 {
		t.Fatalf("%d bytes still counted", total)
	}
}

func TestStopMessageStopsTheMessageBeingWrittenBeforeItsNextChunk(t *testing.T) {
	f := newFakeTransport()
	store, _ := newTestSendStore(t, testSendConfig(), f)
	pid := testPeer(t, 0)

	// A large message of five chunks: the writer waits after its first one
	chunks := 0
	firstWritten := make(chan struct{})
	goOn := make(chan struct{})
	store.writeChunked = func(pid peer.ID, seq uint64, cp *chunkedPayload) error {
		write := store.stoppable(pid, seq, func(frame []byte) error { chunks++; return nil })
		for i := 0; i < 5; i++ {
			if err := write([]byte{byte(i)}); err != nil {
				return fmt.Errorf("chunk %d of message %d: %w", i, seq, err)
			}
			if i == 0 {
				close(firstWritten)
				<-goOn
			}
		}
		return nil
	}
	seq, err := store.EnqueueChunked(pid, "c", make([]byte, 100))
	if err != nil {
		t.Fatalf("large message refused: %v", err)
	}
	mustEnqueue(t, store, pid, "after")

	<-firstWritten
	if !store.StopMessage(pid, seq) {
		t.Fatal("StopMessage of the message being written said it was not there")
	}
	close(goOn)
	waitFor(t, "the queue to empty", func() bool { n, running := queueState(store, pid); return n == 0 && !running })

	if chunks != 1 {
		t.Fatalf("%d chunks written, want 1 (the one before the stop)", chunks)
	}
	sameFrames(t, "written after it", f.writtenFrames(), "after")
	events := store.PopEvents()
	dropped := eventsOf(events, SendEventQueueDropped)
	if len(dropped) != 1 || dropped[0].Reason != SendDroppedReceiverStopped || dropped[0].FromSeq != seq {
		t.Fatalf("drops %+v, want one receiver_stopped of seq %d", dropped, seq)
	}
	for _, ev := range eventsOf(events, SendEventWritten) {
		if ev.Seq == seq {
			t.Fatalf("the stopped message was reported as written: %+v", ev)
		}
	}
	if len(eventsOf(events, SendEventWriteFailed)) != 0 {
		t.Fatal("a stop is not a failed write: it must not be retried")
	}
}

func TestStopMessageOfAMessageNotInTheQueueChangesNothing(t *testing.T) {
	f := newFakeTransport()
	store, _ := newTestSendStore(t, testSendConfig(), f)
	pid := testPeer(t, 0)

	if store.StopMessage(pid, 1) {
		t.Fatal("a peer with no queue has nothing to stop")
	}
	seq := mustEnqueue(t, store, pid, "a")
	waitFor(t, "the frame to be written", func() bool { n, running := queueState(store, pid); return n == 0 && !running })
	if store.StopMessage(pid, seq) {
		t.Fatal("a message already written cannot be stopped")
	}
	if store.StopMessage(pid, 0) {
		t.Fatal("seq 0 names no message")
	}
	if len(eventsOf(store.PopEvents(), SendEventQueueDropped)) != 0 {
		t.Fatal("nothing was dropped")
	}
}
