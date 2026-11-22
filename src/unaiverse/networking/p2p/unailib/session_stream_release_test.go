// Tests of the streams of the Session frames: each frame goes on a stream of its own, and the stream must be given
// back to the resource manager once the frame is through. A stream that stays counted adds up, frame after frame
// (a keep-alive every few seconds), until the limit of streams per protocol and peer is reached: from then on no
// Session frame can be sent to that peer, and its Session is lost.
package main

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/peerstore"
)

// newReleaseTestInstance returns an instance on a host that listens on the given address only, with the stream
// handlers of the Session protocols installed (as setupDirectMessageHandler does).
func newReleaseTestInstance(t *testing.T, listen string) *NodeInstance {
	t.Helper()
	h, err := libp2p.New(libp2p.ListenAddrStrings(listen))
	if err != nil {
		t.Fatalf("host: %v", err)
	}
	t.Cleanup(func() { h.Close() })
	ni := newTestInstance(t)
	ni.host = h
	ni.ctx = context.Background()
	ni.friendlyPeers = make(map[peer.ID]ExtendedPeerInfo)
	h.SetStreamHandler(SessionControlProtocol, func(s network.Stream) {
		handleSessionStream(ni, s, SessionControlProtocol)
	})
	return ni
}

// openStreamsTo gives how many streams h counts towards peer pid (inbound, outbound).
func openStreamsTo(t *testing.T, h host.Host, pid peer.ID) (int, int) {
	t.Helper()
	var stat network.ScopeStat
	if err := h.Network().ResourceManager().ViewPeer(pid, func(s network.PeerScope) error {
		stat = s.Stat()
		return nil
	}); err != nil {
		t.Fatalf("view peer: %v", err)
	}
	return stat.NumStreamsInbound, stat.NumStreamsOutbound
}

// waitForStreams waits until both hosts count at most max streams towards each other, and gives the last counts.
func waitForStreams(t *testing.T, a, b *NodeInstance, max int, within time.Duration) (int, int, int, int) {
	t.Helper()
	deadline := time.Now().Add(within)
	for {
		aIn, aOut := openStreamsTo(t, a.host, b.host.ID())
		bIn, bOut := openStreamsTo(t, b.host, a.host.ID())
		if (aIn+aOut <= max && bIn+bOut <= max) || time.Now().After(deadline) {
			return aIn, aOut, bIn, bOut
		}
		time.Sleep(50 * time.Millisecond)
	}
}

// Many Session frames, one after the other, on each transport: once they are through, no stream of theirs is still
// counted, on either side.
func TestSessionFrameStreamsAreReleased(t *testing.T) {
	prevMax := MaxMessageSize
	MaxMessageSize = 8 << 20
	t.Cleanup(func() { MaxMessageSize = prevMax })

	for _, listen := range []string{"/ip4/127.0.0.1/udp/0/quic-v1", "/ip4/127.0.0.1/tcp/0"} {
		t.Run(listen, func(t *testing.T) {
			sender := newReleaseTestInstance(t, listen)
			receiver := newReleaseTestInstance(t, listen)
			sender.host.Peerstore().AddAddrs(receiver.host.ID(), receiver.host.Addrs(), peerstore.PermanentAddrTTL)
			receiver.host.Peerstore().AddAddrs(sender.host.ID(), sender.host.Addrs(), peerstore.PermanentAddrTTL)

			const frames = 200
			for i := 0; i < frames; i++ {
				if err := sendSessionFrame(sender, receiver.host.ID(), SessionControlProtocol,
					[]byte(fmt.Sprintf("frame %d", i))); err != nil {
					t.Fatalf("frame %d not sent: %v", i, err)
				}
			}
			sIn, sOut, rIn, rOut := waitForStreams(t, sender, receiver, 0, 5*time.Second)
			if sIn+sOut+rIn+rOut != 0 {
				t.Fatalf("after %d frames the streams are still counted: sender in=%d out=%d, receiver in=%d out=%d",
					frames, sIn, sOut, rIn, rOut)
			}
		})
	}
}
