// Tests of the send of a Session frame: what it delivers and how long it may take.
package main

import (
	"context"
	"encoding/binary"
	"io"
	"net"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/peerstore"
	"github.com/libp2p/go-libp2p/core/test"
	ma "github.com/multiformats/go-multiaddr"
	manet "github.com/multiformats/go-multiaddr/net"
)

// How far past SessionSendTimeout a send may return: the time the library takes to notice
// the deadline and unwind.
const sessionSendSlack = time.Second

func newLoopbackHost(t *testing.T) host.Host {
	t.Helper()
	h, err := libp2p.New(libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
	if err != nil {
		t.Fatalf("host: %v", err)
	}
	t.Cleanup(func() { h.Close() })
	return h
}

// newSendingInstance returns an instance that can send Session frames from a host of its own
// (the size limit is package-level and normally set by InitializeLibrary).
func newSendingInstance(t *testing.T) *NodeInstance {
	t.Helper()
	prevMax := MaxMessageSize
	MaxMessageSize = 8 << 20
	t.Cleanup(func() { MaxMessageSize = prevMax })
	return &NodeInstance{
		host:          newLoopbackHost(t),
		ctx:           context.Background(),
		friendlyPeers: make(map[peer.ID]ExtendedPeerInfo),
	}
}

// timedSend runs sendSessionFrame and fails the test if it is still waiting well after
// the limit, so that a send with no bound fails here instead of hanging the suite.
func timedSend(t *testing.T, ni *NodeInstance, pid peer.ID, payload []byte) (time.Duration, error) {
	t.Helper()
	start := time.Now()
	done := make(chan error, 1)
	go func() { done <- sendSessionFrame(ni, pid, SessionControlProtocol, payload) }()
	select {
	case err := <-done:
		return time.Since(start), err
	case <-time.After(SessionSendTimeout + 5*sessionSendSlack):
		t.Fatalf("send still waiting %v after the limit of %v", 5*sessionSendSlack, SessionSendTimeout)
		return 0, nil
	}
}

func TestSessionSendDeliversOneFrame(t *testing.T) {
	ni := newSendingInstance(t)
	receiver := newLoopbackHost(t)
	received := make(chan []byte, 1)
	receiver.SetStreamHandler(SessionControlProtocol, func(s network.Stream) {
		defer s.Close()
		var length uint32
		if err := binary.Read(s, binary.BigEndian, &length); err != nil {
			return
		}
		frame := make([]byte, length)
		if _, err := io.ReadFull(s, frame); err == nil {
			received <- frame
		}
	})
	if err := ni.host.Connect(context.Background(), peer.AddrInfo{ID: receiver.ID(), Addrs: receiver.Addrs()}); err != nil {
		t.Fatalf("connect: %v", err)
	}

	if _, err := timedSend(t, ni, receiver.ID(), []byte("ping")); err != nil {
		t.Fatalf("send: %v", err)
	}
	select {
	case frame := <-received:
		if string(frame) != "ping" {
			t.Fatalf("received %q, want %q", frame, "ping")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the frame did not arrive")
	}
	if _, ok := ni.friendlyPeers[receiver.ID()]; !ok {
		t.Fatal("the receiver was not surfaced to the application pools")
	}
}

// With no connection to the peer the send has to dial first. A peer that accepts the TCP
// connection and then says nothing (a frozen process does exactly that: the kernel answers,
// the node does not) must cost the caller SessionSendTimeout and no more.
func TestSessionSendGivesUpOnAPeerThatDoesNotAnswer(t *testing.T) {
	ni := newSendingInstance(t)
	silent, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	t.Cleanup(func() { silent.Close() })
	go func() {
		for {
			conn, err := silent.Accept()
			if err != nil {
				return
			}
			t.Cleanup(func() { conn.Close() })
		}
	}()
	addr, err := manet.FromNetAddr(silent.Addr())
	if err != nil {
		t.Fatalf("address: %v", err)
	}
	pid := test.RandPeerIDFatal(t)
	ni.host.Peerstore().AddAddrs(pid, []ma.Multiaddr{addr}, peerstore.TempAddrTTL)

	elapsed, err := timedSend(t, ni, pid, []byte("ping"))
	if err == nil {
		t.Fatal("the send toward a silent peer succeeded")
	}
	if elapsed < SessionSendTimeout-sessionSendSlack || elapsed > SessionSendTimeout+sessionSendSlack {
		t.Fatalf("the send returned after %v, want about %v", elapsed, SessionSendTimeout)
	}
}

// With a connection in place the stream opens at once and the write is what can wait: a
// frame larger than the stream window, toward a peer that does not read it, blocks until
// the window opens. The write must share the budget of the send.
func TestSessionSendBoundsTheWrite(t *testing.T) {
	ni := newSendingInstance(t)
	receiver := newLoopbackHost(t)
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	receiver.SetStreamHandler(SessionControlProtocol, func(s network.Stream) {
		<-release
		s.Reset()
	})
	if err := ni.host.Connect(context.Background(), peer.AddrInfo{ID: receiver.ID(), Addrs: receiver.Addrs()}); err != nil {
		t.Fatalf("connect: %v", err)
	}

	elapsed, err := timedSend(t, ni, receiver.ID(), make([]byte, 4<<20))
	if err == nil {
		t.Fatal("a frame nobody read was reported as sent")
	}
	if elapsed > SessionSendTimeout+sessionSendSlack {
		t.Fatalf("the send returned after %v, the limit is %v", elapsed, SessionSendTimeout)
	}
}

// A peer that took the connection and the stream, and then stops answering (a process frozen after its last frame)
// never answers the multistream proposal of the stream. The send must not wait for that answer: the caller gets the
// send back within its budget, and the stream is still given back to the resource manager once the answer is given
// up. The peer's protocols are known here, as they always are after identify, so the stream is negotiated lazily.
func TestSessionSendDoesNotWaitForAPeerThatNeverAnswersTheStream(t *testing.T) {
	for _, listen := range []string{"/ip4/127.0.0.1/udp/0/quic-v1", "/ip4/127.0.0.1/tcp/0"} {
		t.Run(listen, func(t *testing.T) {
			ni := newSendingInstance(t)
			sender, err := libp2p.New(libp2p.ListenAddrStrings(listen))
			if err != nil {
				t.Fatalf("host: %v", err)
			}
			t.Cleanup(func() { sender.Close() })
			ni.host = sender
			receiver, err := libp2p.New(libp2p.ListenAddrStrings(listen))
			if err != nil {
				t.Fatalf("host: %v", err)
			}
			t.Cleanup(func() { receiver.Close() })
			receiver.SetStreamHandler(SessionControlProtocol, func(s network.Stream) { s.Close() })
			if err := sender.Connect(context.Background(), peer.AddrInfo{ID: receiver.ID(), Addrs: receiver.Addrs()}); err != nil {
				t.Fatalf("connect: %v", err)
			}
			deadline := time.Now().Add(5 * time.Second)
			for {
				supported, _ := sender.Peerstore().SupportsProtocols(receiver.ID(), SessionControlProtocol)
				if len(supported) > 0 {
					break
				}
				if time.Now().After(deadline) {
					t.Fatal("identify did not tell the sender the receiver's protocols")
				}
				time.Sleep(20 * time.Millisecond)
			}

			// From now on the receiver takes every stream and says nothing on it.
			release := make(chan struct{})
			t.Cleanup(func() { close(release) })
			receiver.Network().SetStreamHandler(func(s network.Stream) {
				<-release
				s.Reset()
			})

			elapsed, err := timedSend(t, ni, receiver.ID(), []byte("ping"))
			if err != nil {
				t.Fatalf("send: %v", err)
			}
			if elapsed > sessionSendSlack {
				t.Fatalf("the send returned after %v: it waited for the peer's answer", elapsed)
			}
			_, out := openStreamsTo(t, sender, receiver.ID())
			for end := time.Now().Add(SessionSendTimeout + sessionSendSlack); out > 0 && time.Now().Before(end); {
				time.Sleep(50 * time.Millisecond)
				_, out = openStreamsTo(t, sender, receiver.ID())
			}
			if out > 0 {
				t.Fatalf("%d outbound streams still counted after the budget of the send", out)
			}
		})
	}
}
