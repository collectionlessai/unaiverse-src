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

// Stream channels are the PubSub topics and the direct-message channels of content type
// stream_sample; every other channel (other content types, the Session protocols) is not.
func TestIsStreamChannel(t *testing.T) {
	cases := map[string]bool{
		"12D3A::ps:smoHfHa":                         true,
		"12D3A::dm:12D3B-stream_sample~lecture1":    true,
		"12D3A::dm:12D3B-stream_sample":             true,
		"12D3A::dm:12D3B-interaction":               false,
		"12D3A::dm:12D3B-interaction_status~x":      false,
		"12D3A::dm:12D3B-role_suggestion":           false,
		SessionControlProtocol:                      false,
		"12D3A::dm:12D3B-stream_samples_are_not_it": false,
	}
	for channel, want := range cases {
		if got := isStreamChannel(channel); got != want {
			t.Errorf("isStreamChannel(%q) = %v, want %v", channel, got, want)
		}
	}
}

// The network pass takes everything but the streams, all of it; the tick takes one
// message per stream channel and leaves the rest of the stream in the store.
func TestNetworkAndStreamPopsSplitTheChannels(t *testing.T) {
	ni := newTestInstance(t)
	alice := peer.ID("alice")
	stream, inter := "A::ps:sig", "A::dm:B-interaction"
	for _, d := range []string{"s1", "s2", "s3"} {
		storeReceivedMessage(ni, alice, stream, "", []byte(d))
	}
	storeReceivedMessage(ni, alice, inter, "", []byte("i1"))
	storeReceivedMessage(ni, alice, inter, "", []byte("i2"))
	storeReceivedMessage(ni, alice, SessionControlProtocol, SessionControlProtocol, []byte("ping"))

	net := ni.messageStore.popNetworkBatch()
	if got := payloadsOn(net, ""); !sameStrings(got, []string{"i1", "i2"}) {
		t.Fatalf("network pass took %v from the chat channels, want [i1 i2]", got)
	}
	if got := payloadsOn(net, SessionControlProtocol); !sameStrings(got, []string{"ping"}) {
		t.Fatalf("network pass took %v from the Session queue, want [ping]", got)
	}
	for _, want := range []string{"s1", "s2", "s3"} {
		got := payloadsOn(ni.messageStore.popStreamBatch(), "")
		if !sameStrings(got, []string{want}) {
			t.Fatalf("tick took %v, want [%s]", got, want)
		}
	}
	if again := ni.messageStore.popStreamBatch(); len(again) != 0 {
		t.Fatalf("an empty stream gave %d messages", len(again))
	}
}

// A full chat/pubsub queue drops its oldest message; a full Session queue still refuses
// the newest frame (it is shared by all the peers).
func TestFullQueueDropsTheOldestExceptForSessions(t *testing.T) {
	ni := newTestInstance(t)
	maxChannelQueueLen = 3
	alice := peer.ID("alice")
	for _, d := range []string{"s1", "s2", "s3", "s4", "s5"} {
		storeReceivedMessage(ni, alice, "A::ps:sig", "", []byte(d))
		storeReceivedMessage(ni, alice, SessionControlProtocol, SessionControlProtocol, []byte("p"+d))
	}
	var kept []string
	for _, msg := range ni.messageStore.popStreamBatch() {
		kept = append(kept, string(msg.Data))
	}
	for b := ni.messageStore.popStreamBatch(); len(b) > 0; b = ni.messageStore.popStreamBatch() {
		kept = append(kept, string(b[0].Data))
	}
	if !sameStrings(kept, []string{"s3", "s4", "s5"}) {
		t.Fatalf("a full stream kept %v, want the newest [s3 s4 s5]", kept)
	}
	if got := payloadsOn(ni.messageStore.popNetworkBatch(), SessionControlProtocol); !sameStrings(got, []string{"ps1", "ps2", "ps3"}) {
		t.Fatalf("a full Session queue kept %v, want the oldest [ps1 ps2 ps3]", got)
	}
}
