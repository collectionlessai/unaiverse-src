// Tests of the message store: what one PopMessages call hands to the caller.
package main

import (
	"testing"

	"github.com/libp2p/go-libp2p/core/peer"
)

// newTestInstance returns an instance with an empty message store and room in its queues
// (the limits are package-level and normally set by InitializeLibrary).
func newTestInstance(t *testing.T) *NodeInstance {
	t.Helper()
	prevQueueLen, prevChannels := maxChannelQueueLen, maxUniqueChannels
	maxChannelQueueLen, maxUniqueChannels = 64, 64
	t.Cleanup(func() { maxChannelQueueLen, maxUniqueChannels = prevQueueLen, prevChannels })
	return &NodeInstance{messageStore: newMessageStore()}
}

// payloadsOn returns the payloads of the popped messages that carry the given protocol, in
// the order they were popped.
func payloadsOn(popped []*QueuedMessage, protocolID string) []string {
	var out []string
	for _, msg := range popped {
		if msg.Protocol == protocolID {
			out = append(out, string(msg.Data))
		}
	}
	return out
}

func sameStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// The frames of a Session protocol sit in one queue shared by every peer. A pop must hand
// over all of them, oldest first: one frame per pop would make the GOAWAY of a peer wait
// behind the keep-alives of all the others, one caller cycle each.
func TestPopBatchDrainsTheSessionQueues(t *testing.T) {
	ni := newTestInstance(t)
	alice, bob := peer.ID("alice"), peer.ID("bob")

	// Queued as handleSessionStream does: the protocol ID is the channel key.
	storeReceivedMessage(ni, alice, SessionControlProtocol, SessionControlProtocol, []byte("ping-alice"))
	storeReceivedMessage(ni, bob, SessionControlProtocol, SessionControlProtocol, []byte("pong-bob"))
	storeReceivedMessage(ni, bob, SessionControlProtocol, SessionControlProtocol, []byte("ping-bob"))
	storeReceivedMessage(ni, alice, SessionControlProtocol, SessionControlProtocol, []byte("goaway-alice"))
	storeReceivedMessage(ni, alice, SessionHelloProtocol, SessionHelloProtocol, []byte("hello-alice"))
	storeReceivedMessage(ni, bob, SessionHelloProtocol, SessionHelloProtocol, []byte("hello-bob"))

	popped := ni.messageStore.popBatch()

	if len(popped) != 6 {
		t.Fatalf("expected the 6 queued Session frames in one pop, got %d", len(popped))
	}
	control := payloadsOn(popped, SessionControlProtocol)
	if !sameStrings(control, []string{"ping-alice", "pong-bob", "ping-bob", "goaway-alice"}) {
		t.Errorf("control frames out of order or missing: %v", control)
	}
	hello := payloadsOn(popped, SessionHelloProtocol)
	if !sameStrings(hello, []string{"hello-alice", "hello-bob"}) {
		t.Errorf("hello frames out of order or missing: %v", hello)
	}
	if len(ni.messageStore.messagesByChannel) != 0 {
		t.Errorf("drained queues must be dropped, %d left", len(ni.messageStore.messagesByChannel))
	}
	if again := ni.messageStore.popBatch(); len(again) != 0 {
		t.Errorf("nothing was left to pop, got %d", len(again))
	}
}

// Chat and pubsub channels keep their pace of one message per channel per pop, whatever
// the Session queues hold.
func TestPopBatchKeepsOneMessagePerChatChannel(t *testing.T) {
	ni := newTestInstance(t)
	alice := peer.ID("alice")

	for _, text := range []string{"a1", "a2", "a3"} {
		storeReceivedMessage(ni, alice, "alice::dm:me-stream_sample", UnaiverseChatProtocol, []byte(text))
	}
	storeReceivedMessage(ni, alice, "alice::ps:topic", "", []byte("t1"))
	storeReceivedMessage(ni, alice, SessionControlProtocol, SessionControlProtocol, []byte("ping"))
	storeReceivedMessage(ni, alice, SessionControlProtocol, SessionControlProtocol, []byte("goaway"))

	first := ni.messageStore.popBatch()
	if len(first) != 4 {
		t.Fatalf("expected one message per chat/pubsub channel plus the 2 Session frames, got %d", len(first))
	}
	if chat := payloadsOn(first, UnaiverseChatProtocol); !sameStrings(chat, []string{"a1"}) {
		t.Errorf("first pop of the chat channel: %v", chat)
	}

	second := ni.messageStore.popBatch()
	if chat := payloadsOn(second, UnaiverseChatProtocol); len(second) != 1 || !sameStrings(chat, []string{"a2"}) {
		t.Errorf("second pop: %d messages, chat %v", len(second), chat)
	}
	third := ni.messageStore.popBatch()
	if chat := payloadsOn(third, UnaiverseChatProtocol); len(third) != 1 || !sameStrings(chat, []string{"a3"}) {
		t.Errorf("third pop: %d messages, chat %v", len(third), chat)
	}
	if len(ni.messageStore.messagesByChannel) != 0 {
		t.Errorf("every queue should be gone, %d left", len(ni.messageStore.messagesByChannel))
	}
}
