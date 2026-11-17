// The relay slot of an instance: the reservation a member holds on its world host.
//
// A relay drops the reservation of a peer the moment no connection to that peer is left
// (circuitv2/relay, disconnected). A connection that is lost and opens again leaves the
// instance with circuit addresses that lead nowhere, and nothing above may have seen the
// loss: the last connection to a known peer is hidden for DisconnectionGracePeriod, and a
// Session frame sent meanwhile dials again by itself. So the instance remembers the relay
// it reserved on and asks again whenever a connection to it opens.
package main

import (
	"context"
	"time"

	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/p2p/protocol/circuitv2/client"
)

// reserveRelaySlot asks the relay for a slot and remembers the relay with the expiry of the
// slot. Requests run one at a time, the wait for the turn included in the time of ctx. A
// request made after a new connection (again) only refreshes what is remembered: it does
// not bring back a relay that was given up while it ran.
func reserveRelaySlot(ctx context.Context, ni *NodeInstance, relayInfo peer.AddrInfo, again bool) (*client.Reservation, error) {
	ni.relayAskMutex.Lock()
	defer ni.relayAskMutex.Unlock()

	reservation, err := client.Reserve(ctx, ni.host, relayInfo)
	if err != nil || reservation == nil {
		return reservation, err
	}

	ni.relaySlotMutex.Lock()
	if !again || ni.relaySlotPeer == relayInfo.ID {
		ni.relaySlotPeer = relayInfo.ID
		ni.relaySlotExpiry = reservation.Expiration
	}
	ni.relaySlotMutex.Unlock()
	return reservation, nil
}

// forgetRelaySlot drops what is remembered of the slot on pid, or of any slot when pid is
// empty, and with it the circuit addresses of the instance through that relay: nobody
// reaches the instance on them any more. The relay itself drops the slot when the
// connection closes.
func forgetRelaySlot(ni *NodeInstance, pid peer.ID) {
	ni.relaySlotMutex.Lock()
	forgotten := ni.relaySlotPeer != "" && (pid == "" || ni.relaySlotPeer == pid)
	if forgotten {
		ni.relaySlotPeer = ""
		ni.relaySlotExpiry = time.Time{}
	}
	ni.relaySlotMutex.Unlock()
	if !forgotten {
		return
	}
	ni.addrMutex.Lock()
	ni.privateRelayAddrs = nil
	ni.addrMutex.Unlock()
}

// wantsRelaySlotAgain reports whether the connection that just opened calls for a new
// request, and books it. It does when the peer is the relay of a slot that has not run out
// (asking for one that did is a decision of the node), the connection is a direct one, and
// no request is already waiting to start.
func wantsRelaySlotAgain(ni *NodeInstance, conn network.Conn) bool {
	if conn.Stat().Limited {
		return false
	}
	ni.relaySlotMutex.Lock()
	defer ni.relaySlotMutex.Unlock()
	if ni.relaySlotPeer != conn.RemotePeer() || ni.relaySlotAsking || !time.Now().Before(ni.relaySlotExpiry) {
		return false
	}
	ni.relaySlotAsking = true
	return true
}

// askRelaySlotAgain is called for every connection that opens. Every connection to the
// relay asks, not only the first after a loss: the two ends do not see the same order of
// closings and openings, and a request with the slot still there only refreshes it.
func askRelaySlotAgain(ni *NodeInstance, conn network.Conn) {
	if !wantsRelaySlotAgain(ni, conn) {
		return
	}
	relay := conn.RemotePeer()
	go func() {
		defer recoverGoroutine("askRelaySlotAgain")
		// Cleared before the request, so that a connection that opens while it runs asks
		// once more: the slot of this request may be gone with the connection it ran on.
		ni.relaySlotMutex.Lock()
		ni.relaySlotAsking = false
		ni.relaySlotMutex.Unlock()

		ctx, cancel := context.WithTimeout(ni.ctx, RelayReservationTimeout)
		defer cancel()
		reservation, err := reserveRelaySlot(ctx, ni, peer.AddrInfo{ID: relay}, true)
		if err != nil || reservation == nil {
			logger.Warnf("[GO] ⚠️ Instance %d: Slot on relay %s not obtained again after a new connection: %v\n", ni.instanceIndex, relay, err)
			return
		}
		logger.Infof("[GO] ✅ Instance %d: Slot on relay %s asked again after a new connection, expires at %s\n", ni.instanceIndex, relay, reservation.Expiration.UTC().Format(time.RFC3339))
	}()
}
