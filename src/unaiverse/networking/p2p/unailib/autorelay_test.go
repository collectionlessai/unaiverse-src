// Tests of the AutoRelay settings: how a private node gets a relay back.
package main

import (
	"testing"
	"time"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/test"
	autorelay "github.com/libp2p/go-libp2p/p2p/host/autorelay"
	ma "github.com/multiformats/go-multiaddr"
)

// The interval at which AutoRelay asks for its static relays again (the library default,
// which unailib leaves alone).
const autoRelayPollInterval = 30 * time.Second

// relayTestClock adapts the mock clock of the library to the interface AutoRelay takes.
type relayTestClock struct{ *test.MockClock }

func (c relayTestClock) InstantTimer(when time.Time) autorelay.InstantTimer {
	return c.MockClock.InstantTimer(when)
}

// newTestRelay starts a relay on the loopback interface. AutoRelay builds circuit addresses
// from the public addresses of a relay only, so the relay also announces one that the
// library takes for public (a reserved block nobody routes); it is never dialled, because
// the loopback address answers first.
func newTestRelay(t *testing.T) host.Host {
	t.Helper()
	h, err := libp2p.New(
		libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"),
		libp2p.DisableRelay(),
		libp2p.EnableRelayService(),
		libp2p.ForceReachabilityPublic(),
		libp2p.AddrsFactory(func(addrs []ma.Multiaddr) []ma.Multiaddr {
			return append(addrs, ma.StringCast("/ip4/192.0.0.200/tcp/4001"))
		}),
	)
	if err != nil {
		t.Fatalf("relay: %v", err)
	}
	t.Cleanup(func() { h.Close() })
	return h
}

// relayInUse returns the relay behind the circuit addresses of the host, or "" when the
// host has none.
func relayInUse(h host.Host) peer.ID {
	for _, addr := range h.Addrs() {
		if _, err := addr.ValueForProtocol(ma.P_CIRCUIT); err != nil {
			continue
		}
		if id, err := addr.ValueForProtocol(ma.P_P2P); err == nil {
			if relay, err := peer.Decode(id); err == nil {
				return relay
			}
		}
	}
	return ""
}

// waitForRelayState polls until the relay in use satisfies the condition, and returns it.
func waitForRelayState(t *testing.T, h host.Host, what string, ok func(peer.ID) bool) peer.ID {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if relay := relayInUse(h); ok(relay) {
			return relay
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s (relay in use: %q)", what, relayInUse(h))
	return ""
}

// A node that loses its relay twice in a row has used both static relays, and AutoRelay
// leaves a relay alone for the backoff after every attempt, the successful ones too. With
// the backoff of the library, one hour, the node stayed without a relay for the rest of
// the hour. It must have one again as soon as AutoRelayBackoff and one poll have passed.
func TestAutoRelayGetsARelayBackAfterTwoLosses(t *testing.T) {
	clock := relayTestClock{test.NewMockClock()}
	var static []peer.AddrInfo
	for i := 0; i < 2; i++ {
		relay := newTestRelay(t)
		static = append(static, peer.AddrInfo{ID: relay.ID(), Addrs: relay.Network().ListenAddresses()})
	}

	node, err := libp2p.New(
		libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"),
		libp2p.ForceReachabilityPrivate(),
		libp2p.EnableAutoRelayWithStaticRelays(static, append(autoRelayOptions(), autorelay.WithClock(clock))...),
	)
	if err != nil {
		t.Fatalf("node: %v", err)
	}
	t.Cleanup(func() { node.Close() })

	some := func(relay peer.ID) bool { return relay != "" }
	none := func(relay peer.ID) bool { return relay == "" }

	first := waitForRelayState(t, node, "the first relay", some)
	// Past the boot delay of autoRelayOptions, during which a node takes a relay only
	// when both are candidates.
	clock.AdvanceBy(15 * time.Second)
	node.Network().ClosePeer(first)
	second := waitForRelayState(t, node, "the other relay", func(relay peer.ID) bool {
		return relay != "" && relay != first
	})

	// Both relays were tried a moment ago: neither is a candidate now.
	node.Network().ClosePeer(second)
	waitForRelayState(t, node, "no relay", none)

	clock.AdvanceBy(AutoRelayBackoff + autoRelayPollInterval)
	waitForRelayState(t, node, "a relay after the backoff", some)
}
