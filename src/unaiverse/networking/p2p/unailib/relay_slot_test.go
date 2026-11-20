// Tests of the relay slot: the reservation of an instance on its relay is asked again when
// a connection to the relay opens, and not after the relay was given up.
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	libp2pproto "github.com/libp2p/go-libp2p/core/protocol"
	"github.com/libp2p/go-libp2p/p2p/protocol/circuitv2/client"
	relayv2 "github.com/libp2p/go-libp2p/p2p/protocol/circuitv2/relay"
	ma "github.com/multiformats/go-multiaddr"
)

// newSlotWorld starts a relay with no limits that also takes Session frames, as a world host.
func newSlotWorld(t *testing.T) host.Host {
	t.Helper()
	world := newChatRelay(t, false)
	world.SetStreamHandler(libp2pproto.ID(SessionControlProtocol), func(s network.Stream) {
		_, _ = io.Copy(io.Discard, s)
		_ = s.Close()
	})
	return world
}

// newSlotMember returns an instance with the connection notifiers of a node.
func newSlotMember(t *testing.T) *NodeInstance {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	ni := newSendingInstance(t)
	ni.ctx = ctx
	ni.disconnectionTimers = make(map[peer.ID]context.CancelFunc)
	ni.webrtcConnections = make(map[peer.ID]*WebRTCConn)
	ni.persistentChatStreams = make(map[peer.ID]*persistentStream)
	setupNotifiers(ni)
	return ni
}

// joinAndReserve connects the member to the world, exchanges a Session frame with it (which
// makes the world a known peer of the member, as its world Session does) and reserves.
func joinAndReserve(t *testing.T, member *NodeInstance, world host.Host) {
	t.Helper()
	connectNodes(t, member.host, world)
	sendFrame(t, member, world.ID())
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	worldInfo := peer.AddrInfo{ID: world.ID(), Addrs: world.Addrs()}
	if _, err := reserveRelaySlot(ctx, member, worldInfo, false); err != nil {
		t.Fatalf("reservation: %v", err)
	}
}

func sendFrame(t *testing.T, member *NodeInstance, to peer.ID) {
	t.Helper()
	if err := sendSessionFrame(member, to, SessionControlProtocol, []byte("frame")); err != nil {
		t.Fatalf("Session frame: %v", err)
	}
}

// dialThroughRelay dials the target on its circuit address from a host that knows nothing
// else of it. A new host at every call: one that was refused answers from its dial backoff.
func dialThroughRelay(t *testing.T, relay host.Host, target peer.ID) error {
	t.Helper()
	dialler := newLoopbackHost(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	circuit := ma.StringCast(fmt.Sprintf("%s/p2p/%s/p2p-circuit", relay.Addrs()[0], relay.ID()))
	return dialler.Connect(ctx, peer.AddrInfo{ID: target, Addrs: []ma.Multiaddr{circuit}})
}

func hasNoReservation(err error) bool {
	return err != nil && strings.Contains(err.Error(), "NO_RESERVATION")
}

// closedByTheWorld closes the connections to the member on the side of the world and waits
// until the member has seen them go: for the member, a loss it did not ask for.
func closedByTheWorld(t *testing.T, world host.Host, member *NodeInstance) {
	t.Helper()
	if err := world.Network().ClosePeer(member.host.ID()); err != nil {
		t.Fatalf("closing on the world: %v", err)
	}
	waitUntil(t, "the member to see the connection closed", func() bool {
		return len(member.host.Network().ConnsToPeer(world.ID())) == 0
	})
}

func rememberedRelay(ni *NodeInstance) peer.ID {
	ni.relaySlotMutex.Lock()
	defer ni.relaySlotMutex.Unlock()
	return ni.relaySlotPeer
}

// The loss nothing above the library sees: the connection to the world is closed and a
// Session frame opens it again within the grace of a known peer, which stays known all
// along. The frame is the only traffic, and the member is reachable through the world again.
func TestSlotIsAskedAgainWhenTheConnectionToTheRelayComesBack(t *testing.T) {
	world := newSlotWorld(t)
	member := newSlotMember(t)
	joinAndReserve(t, member, world)
	if err := dialThroughRelay(t, world, member.host.ID()); err != nil {
		t.Fatalf("dial through the world with the slot just reserved: %v", err)
	}

	// Twice: a loss after the first one is healed as the first was.
	for _, loss := range []string{"first", "second"} {
		closedByTheWorld(t, world, member)
		if err := dialThroughRelay(t, world, member.host.ID()); !hasNoReservation(err) {
			t.Fatalf("%s loss: the world should have dropped the slot with the connection, the dial gave: %v", loss, err)
		}

		sendFrame(t, member, world.ID())
		if _, known := isFriendly(member, world.ID()); !known {
			t.Fatalf("%s loss: the world left the known peers of the member, the loss was not a hidden one", loss)
		}
		// The request runs beside the frame that opened the connection, a moment after it.
		waitUntil(t, "the member to be reachable through the world again after the "+loss+" loss", func() bool {
			return dialThroughRelay(t, world, member.host.ID()) == nil
		})
	}
}

// A member that gave its relay up (the node disconnects from a world it leaves) does not
// reserve on it again when a connection to that peer opens.
func TestRelayGivenUpIsNotAskedAgain(t *testing.T) {
	world := newSlotWorld(t)
	member := newSlotMember(t)
	joinAndReserve(t, member, world)

	forgetRelaySlot(member, world.ID())
	closedByTheWorld(t, world, member)
	sendFrame(t, member, world.ID())

	for _, conn := range member.host.Network().ConnsToPeer(world.ID()) {
		if wantsRelaySlotAgain(member, conn) {
			t.Errorf("a connection to a relay that was given up asks for the slot again")
		}
	}
	// Time for a request that should not exist to reach the world.
	time.Sleep(300 * time.Millisecond)
	if err := dialThroughRelay(t, world, member.host.ID()); !hasNoReservation(err) {
		t.Errorf("the member holds a slot on the world it gave up, the dial gave: %v", err)
	}
}

func TestWhichConnectionsAskForTheSlotAgain(t *testing.T) {
	world := newSlotWorld(t)
	other := newLoopbackHost(t)
	member := newSlotMember(t)
	connectNodes(t, member.host, other)
	connectNodes(t, member.host, world)
	toWorld := member.host.Network().ConnsToPeer(world.ID())[0]
	toOther := member.host.Network().ConnsToPeer(other.ID())[0]
	asks := func(conn network.Conn) bool {
		t.Helper()
		asked := wantsRelaySlotAgain(member, conn)
		if asked && wantsRelaySlotAgain(member, conn) {
			t.Errorf("a second request was booked while the first had not started")
		}
		member.relaySlotMutex.Lock()
		member.relaySlotAsking = false
		member.relaySlotMutex.Unlock()
		return asked
	}

	if asks(toWorld) {
		t.Errorf("a connection asks for a slot the instance never reserved")
	}

	joinAndReserve(t, member, world)
	if rememberedRelay(member) != world.ID() {
		t.Fatalf("the relay of the slot is not remembered")
	}
	if !asks(toWorld) {
		t.Errorf("a connection to the relay of the slot does not ask again")
	}
	if asks(toOther) {
		t.Errorf("a connection to another peer asks for the slot")
	}

	// Disconnecting from another peer leaves the slot alone.
	forgetRelaySlot(member, other.ID())
	if !asks(toWorld) {
		t.Errorf("the slot was forgotten with a peer that is not its relay")
	}

	// A slot that ran out was not renewed by the node: asking again is not for the library.
	member.relaySlotMutex.Lock()
	member.relaySlotExpiry = time.Now().Add(-time.Second)
	member.relaySlotMutex.Unlock()
	if asks(toWorld) {
		t.Errorf("a connection asks again for a slot that ran out")
	}
}

// A relay reached through another relay, on a limited connection, cannot be asked on it.
func TestLimitedConnectionToTheRelayDoesNotAsk(t *testing.T) {
	limitedRelay := newChatRelay(t, true)
	slotRelay := newLoopbackHost(t)
	member := newSlotMember(t)
	relayedLink(t, limitedRelay, member.host, slotRelay)

	member.relaySlotMutex.Lock()
	member.relaySlotPeer = slotRelay.ID()
	member.relaySlotExpiry = time.Now().Add(time.Hour)
	member.relaySlotMutex.Unlock()

	conns := member.host.Network().ConnsToPeer(slotRelay.ID())
	if len(conns) != 1 || !conns[0].Stat().Limited {
		t.Fatalf("expected one limited connection to the relay of the slot, got %v", conns)
	}
	if wantsRelaySlotAgain(member, conns[0]) {
		t.Errorf("a limited connection to the relay asks for the slot")
	}
}

// A request started by a new connection can end after the node gave the relay up: its
// answer must not bring the relay back.
func TestRequestAfterANewConnectionDoesNotBringBackARelayGivenUp(t *testing.T) {
	world := newSlotWorld(t)
	member := newSlotMember(t)
	joinAndReserve(t, member, world)
	forgetRelaySlot(member, world.ID())

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if _, err := reserveRelaySlot(ctx, member, peer.AddrInfo{ID: world.ID()}, true); err != nil {
		t.Fatalf("reservation: %v", err)
	}
	if relay := rememberedRelay(member); relay != "" {
		t.Errorf("the relay given up is remembered again: %s", relay)
	}

	// A request of the node itself is what remembers a relay.
	if _, err := reserveRelaySlot(ctx, member, peer.AddrInfo{ID: world.ID()}, false); err != nil {
		t.Fatalf("reservation: %v", err)
	}
	if rememberedRelay(member) != world.ID() {
		t.Errorf("the relay of a slot the node asked for is not remembered")
	}
}

// The circuit addresses of the instance are those of its slot: they go when the relay is
// given up, and stay when the node disconnects from any other peer.
func TestCircuitAddressesGoWithTheSlot(t *testing.T) {
	world := newSlotWorld(t)
	other := newLoopbackHost(t)
	member := newSlotMember(t)
	joinAndReserve(t, member, world)
	// What ReserveOnRelay leaves in the instance after the reservation.
	circuit := ma.StringCast(fmt.Sprintf("%s/p2p/%s/p2p-circuit", world.Addrs()[0], world.ID()))
	member.addrMutex.Lock()
	member.privateRelayAddrs = []ma.Multiaddr{circuit}
	member.addrMutex.Unlock()
	circuitAddrs := func() []ma.Multiaddr {
		member.addrMutex.RLock()
		defer member.addrMutex.RUnlock()
		return member.privateRelayAddrs
	}

	forgetRelaySlot(member, other.ID())
	if len(circuitAddrs()) != 1 {
		t.Fatalf("the circuit addresses went with a peer that is not the relay: %v", circuitAddrs())
	}
	forgetRelaySlot(member, world.ID())
	if len(circuitAddrs()) != 0 {
		t.Errorf("the circuit addresses outlived the slot: %v", circuitAddrs())
	}
}

func circuitListeners(h host.Host) int {
	count := 0
	for _, addr := range h.Network().ListenAddresses() {
		if _, err := addr.ValueForProtocol(ma.P_CIRCUIT); err == nil {
			count++
		}
	}
	return count
}

// ReserveOnRelay listens on nothing, because the library does: a host with the circuit
// transport listens for circuits since it started, through whatever relay they come. A
// version of the library that stopped doing so would leave a member unreachable.
func TestHostListensForCircuitsWithNoListenOfOurs(t *testing.T) {
	world := newSlotWorld(t)
	member := newSlotMember(t)
	if n := circuitListeners(member.host); n != 1 {
		t.Fatalf("expected the one circuit listener of the library on a new host, found %d", n)
	}

	joinAndReserve(t, member, world)
	if err := dialThroughRelay(t, world, member.host.ID()); err != nil {
		t.Fatalf("dial through the world: %v", err)
	}
	if n := circuitListeners(member.host); n != 1 {
		t.Errorf("a reservation changed the circuit listeners of the host: %d", n)
	}
}

// newCircuitOnlyWorld starts a world host that the member reaches only through another
// relay, on a limited circuit: a member that joined while it had no direct path.
func newCircuitOnlyWorld(t *testing.T, member *NodeInstance) host.Host {
	t.Helper()
	world := newWorldBehind(t, member, true)
	conns := member.host.Network().ConnsToPeer(world.ID())
	if len(conns) != 1 || !conns[0].Stat().Limited {
		t.Fatalf("expected one limited connection to the world, got %v", conns)
	}
	return world
}

// newWorldBehind starts a world host that the member reaches only through another relay,
// one with the limits of the library or one with none.
func newWorldBehind(t *testing.T, member *NodeInstance, limited bool) host.Host {
	t.Helper()
	world, err := libp2p.New(
		libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"),
		libp2p.ForceReachabilityPublic(),
		libp2p.EnableRelayService(relayv2.WithInfiniteLimits()),
	)
	if err != nil {
		t.Fatalf("world: %v", err)
	}
	t.Cleanup(func() { world.Close() })
	relayedLink(t, newChatRelay(t, limited), member.host, world)
	return world
}

// What the library does with a request for a slot on a relay that is reached only through
// another relay: it waits for a direct connection to appear, and asks nothing until then.
// With nothing that makes one (no hole punching, no reversal) it waits for the whole time
// of the request, which was a minute.
func TestReservationThroughACircuitWaitsForADirectConnection(t *testing.T) {
	member := newSlotMember(t)
	world := newCircuitOnlyWorld(t, member)

	const budget = 2 * time.Second
	ctx, cancel := context.WithTimeout(context.Background(), budget)
	defer cancel()
	start := time.Now()
	_, err := client.Reserve(ctx, member.host, peer.AddrInfo{ID: world.ID()})
	took := time.Since(start)
	if err == nil {
		t.Fatalf("a slot was granted through a circuit")
	}
	if took < budget || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("the request ended after %v with %v: it did not wait for its time", took, err)
	}
	if conns := member.host.Network().ConnsToPeer(world.ID()); len(conns) != 1 || !conns[0].Stat().Limited {
		t.Errorf("the connections to the world changed: %v", conns)
	}
}

// A slot is not asked on a relay the member reaches only through another relay: the request
// is refused at once and says why, and no relay is remembered.
func TestReservationThroughACircuitIsRefusedAtOnce(t *testing.T) {
	for name, limited := range map[string]bool{"a limited circuit": true, "a circuit with no limits": false} {
		t.Run(name, func(t *testing.T) {
			member := newSlotMember(t)
			world := newWorldBehind(t, member, limited)

			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			start := time.Now()
			_, err := reserveRelaySlot(ctx, member, peer.AddrInfo{ID: world.ID()}, false)
			if took := time.Since(start); took > 500*time.Millisecond {
				t.Errorf("the refusal took %v", took)
			}
			if !errors.Is(err, errRelayOnlyThroughACircuit) {
				t.Errorf("a request through a circuit ended with %v", err)
			}
			if relay := rememberedRelay(member); relay != "" {
				t.Errorf("a relay was remembered after a refusal: %s", relay)
			}
		})
	}
}
