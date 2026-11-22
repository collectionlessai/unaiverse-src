// Tests of the chunked frames on a WebRTC DataChannel, between two Go nodes on the loopback.
package main

import (
	"bytes"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/network"
)

// chunkedNodes are two linked nodes that take chunked frames, the remote one also on its
// chat stream.
func chunkedNodes(t *testing.T, size int) (local, remote *sessionNode, localConn *WebRTCConn) {
	t.Helper()
	useSessionTestSettings(t, 8<<20)
	withChunks(t, size, 0)
	local, remote = newSessionNode(t), newSessionNode(t)
	for _, node := range []*sessionNode{local, remote} {
		node.ni.reassembly = newReassemblyStore()
	}
	remote.ni.host.SetStreamHandler(UnaiverseChatProtocol, func(s network.Stream) { handleStream(remote.ni, s) })
	localConn, _ = linkNodes(t, local, remote)
	return local, remote, localConn
}

// patterned is a payload whose bytes say where they are, so that a misplaced chunk shows.
func patterned(n int) []byte {
	payload := make([]byte, n)
	for i := range payload {
		payload[i] = byte(i % 251)
	}
	return payload
}

// A large message goes on the open DataChannel, as the small ones around it, and arrives
// whole and in their order, queued as a DataChannel message.
func TestChunkedMessageCrossesTheDataChannelInOrder(t *testing.T) {
	local, remote, _ := chunkedNodes(t, 256<<10)
	channel := "A::dm:B-misc"
	before, after := []byte("before"), []byte("after")
	large := patterned(3 << 20) // 12 chunks
	cp := &chunkedPayload{channel: channel, payload: large}

	write := func(data []byte) {
		frame, err := buildDirectMessageFrame(channel, data)
		if err != nil {
			t.Fatalf("frame: %v", err)
		}
		if err := writeQueuedFrame(local.ni, remote.id(), frame); err != nil {
			t.Fatalf("small message: %v", err)
		}
	}
	// What the channel holds to send, seen every millisecond while the large message is written.
	localConn := openDataChannel(local.ni, remote.id())
	stop, peak := make(chan struct{}), make(chan uint64, 1)
	go func() {
		var most uint64
		for {
			select {
			case <-stop:
				peak <- most
				return
			case <-time.After(time.Millisecond):
				most = max(most, localConn.dc.BufferedAmount())
			}
		}
	}()
	write(before)
	if err := writeChunkedMessage(local.ni, remote.id(), 1, cp); err != nil {
		t.Fatalf("chunked message: %v", err)
	}
	write(after)
	close(stop)
	if most := <-peak; most > uint64(2*chunkSize+chunkFirstExtra) {
		t.Fatalf("the DataChannel held %d bytes to send, more than two chunks (%d)", most, chunkSize)
	}

	got := remote.popped(t, 3)[channel]
	if len(got) != 3 || !bytes.Equal(got[0].Data, before) || !bytes.Equal(got[1].Data, large) || !bytes.Equal(got[2].Data, after) {
		t.Fatalf("got %d messages on the channel, want before, the large one whole, after", len(got))
	}
	for _, m := range got {
		if m.Protocol != UnaiverseWebRTCSignalProtocol {
			t.Fatalf("a message was queued as %q, not as one of the DataChannel", m.Protocol)
		}
	}
	if cp.attempt != 1 {
		t.Fatalf("the message took %d starts, want one", cp.attempt)
	}
}

// A DataChannel that closes in the middle of a message leaves it to the chat stream, where it
// starts again with the next attempt: it arrives once and whole, whatever of the first
// attempt was still on its way.
func TestChunkedMessageFallsBackToTheStreamWhenTheDataChannelCloses(t *testing.T) {
	local, remote, localConn := chunkedNodes(t, 1<<20)
	channel := "A::dm:B-misc"
	large := patterned(48 << 20)
	cp := &chunkedPayload{channel: channel, payload: large}

	done := make(chan error, 1)
	go func() { done <- writeChunkedMessage(local.ni, remote.id(), 5, cp) }()
	waitUntil(t, "the first chunks on the DataChannel", func() bool {
		r := remote.ni.reassembly
		r.mu.Lock()
		defer r.mu.Unlock()
		p := r.byPeer[local.id()]
		return p != nil && p.protocol == UnaiverseWebRTCSignalProtocol && p.next >= 2
	})
	if err := localConn.dc.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("the message was not written: %v", err)
		}
	case <-time.After(60 * time.Second):
		t.Fatal("the write did not end")
	}

	got := remote.popped(t, 1)[channel]
	if len(got) != 1 || !bytes.Equal(got[0].Data, large) || got[0].Protocol != UnaiverseChatProtocol {
		t.Fatalf("got %d messages, want the whole message once, from the chat stream", len(got))
	}
	if cp.attempt != 2 {
		t.Fatalf("the message took %d starts, want two (DataChannel, then stream)", cp.attempt)
	}
	time.Sleep(200 * time.Millisecond)
	if again := remote.ni.messageStore.popNetworkBatch(); len(again) != 0 {
		t.Fatalf("%d more messages arrived", len(again))
	}
}
