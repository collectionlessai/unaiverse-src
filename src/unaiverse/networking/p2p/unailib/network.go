// Here we define the core logic for the P2P library.
package main

import (
	"os"
	"fmt"
	"time"
	"context"
	"strings"
	"crypto/rand"
	"container/list"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/event"
	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/network"
	ma "github.com/multiformats/go-multiaddr"
	pubsub "github.com/libp2p/go-libp2p-pubsub"
	p2pforge "github.com/ipshipyard/p2p-forge/client"
)

func loadOrCreateIdentity(keyPath string) (crypto.PrivKey, error) {
	// Check if key file already exists.
	if _, err := os.Stat(keyPath); err == nil {
		// Key file exists, read and unmarshal it.
		bytes, err := os.ReadFile(keyPath)
		if err != nil {
			return nil, fmt.Errorf("failed to read existing key file: %w", err)
		}
		// load the key
		privKey, err := crypto.UnmarshalPrivateKey(bytes)
		if err != nil {
			return nil, fmt.Errorf("failed to unmarshal corrupt private key: %w", err)
		}
		return privKey, nil

	} else if os.IsNotExist(err) {
		// Key file does not exist, generate a new one.
		logger.Infof("[GO] 💎 Generating new persistent peer identity in %s\n", keyPath)
		privKey, _, err := crypto.GenerateEd25519Key(rand.Reader)
		if err != nil {
			return nil, fmt.Errorf("failed to generate new key: %w", err)
		}

		// Marshal the new key to bytes.
		bytes, err := crypto.MarshalPrivateKey(privKey)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal new private key: %w", err)
		}

		// Write the new key to a file.
		if err := os.WriteFile(keyPath, bytes, 0400); err != nil {
			return nil, fmt.Errorf("failed to write new key file: %w", err)
		}
		return privKey, nil

	} else {
		// Another error occurred (e.g., permissions).
		return nil, fmt.Errorf("failed to stat key file: %w", err)
	}
}

func getListenAddrs(ips []string, tcpPort int, tlsMode string) ([]ma.Multiaddr, error) {
	if len(ips) == 0 {
        ips = []string{"0.0.0.0"}
    }

	var listenAddrs []ma.Multiaddr
	quicPort := 0
	webtransPort := 0
	webrtcPort := 0
	if tcpPort != 0 {
		quicPort = tcpPort + 1
		webtransPort = tcpPort + 2
		webrtcPort = tcpPort +3
	}

	// --- Create Multiaddrs for both protocols from the single IP list ---
	for _, ip := range ips {
		// TCP
		tcpMaddr, _ := ma.NewMultiaddr(fmt.Sprintf("/ip4/%s/tcp/%d", ip, tcpPort))
		// QUIC
		quicMaddr, _ := ma.NewMultiaddr(fmt.Sprintf("/ip4/%s/udp/%d/quic-v1", ip, quicPort))
		// WebTransport
		webtransMaddr, _ := ma.NewMultiaddr(fmt.Sprintf("/ip4/%s/udp/%d/quic-v1/webtransport", ip, webtransPort))
		// WebRTC Direct
		webrtcMaddr, _ := ma.NewMultiaddr(fmt.Sprintf("/ip4/%s/udp/%d/webrtc-direct", ip, webrtcPort))

		listenAddrs = append(listenAddrs, tcpMaddr, quicMaddr, webtransMaddr, webrtcMaddr)

		switch tlsMode {
		case "autotls":
			// This is the special multiaddr that triggers AutoTLS
			wssMaddr, _ := ma.NewMultiaddr(fmt.Sprintf("/ip4/%s/tcp/%d/tls/sni/*.%s/ws", ip, tcpPort, p2pforge.DefaultForgeDomain))
			listenAddrs = append(listenAddrs, wssMaddr)
		case "domain":
			// This is the standard secure WebSocket address with provided domain
			wssMaddr, _ := ma.NewMultiaddr(fmt.Sprintf("/ip4/%s/tcp/%d/tls/ws", ip, tcpPort))
			listenAddrs = append(listenAddrs, wssMaddr)
		default:
			// Fallback to a standard, non-secure WebSocket address
			wsMaddr, _ := ma.NewMultiaddr(fmt.Sprintf("/ip4/%s/tcp/%d/ws", ip, tcpPort))
			listenAddrs = append(listenAddrs, wsMaddr)
		}
	}

	logger.Debugf("[GO] 🔧 Prepared Listen Addresses: %v\n", listenAddrs)

	return listenAddrs, nil
}

func setupPubSub(ni *NodeInstance) error {
	psOptions := []pubsub.Option{
		// pubsub.WithFloodPublish(true),
		pubsub.WithMaxMessageSize(int(MaxMessageSize)),
	}
	ps, err := pubsub.NewGossipSub(ni.ctx, ni.host, psOptions...)
	if err != nil {
		return err
	}
	ni.pubsub = ps // Set the pubsub field on the instance
	return nil
}

func setupNotifiers(ni *NodeInstance) {
	ni.host.Network().Notify(&network.NotifyBundle{
		ConnectedF: func(_ network.Network, conn network.Conn) {
			remotePeerID := conn.RemotePeer()
			logger.Debugf("[GO] 🔔 Instance %d: Event - Connected to %s (Direction: %s)\n", ni.instanceIndex, remotePeerID, conn.Stat().Direction)
			// --- Abort Graceful Disconnect if active ---
            ni.disconnectionMutex.Lock()
            if cancelTimer, exists := ni.disconnectionTimers[remotePeerID]; exists {
                cancelTimer() // Stop the cleanup timer
                delete(ni.disconnectionTimers, remotePeerID)
                logger.Debugf("[GO] ♻️ Instance %d: Peer %s reconnected within grace period. Cleanup aborted.\n", ni.instanceIndex, remotePeerID)
            }
            ni.disconnectionMutex.Unlock()
		},
		DisconnectedF: func(_ network.Network, conn network.Conn) {
			remotePeerID := conn.RemotePeer()
			logger.Debugf("[GO] 🔔 Instance %d: Event - Disconnected from %s\n", ni.instanceIndex, remotePeerID)

			// Get the host for this instance to query its network state.
			if ni.host == nil {
				// This shouldn't happen if the notifier is active, but a safe check.
				logger.Warnf("[GO] ⚠️ Instance %d: DisconnectedF: Host is nil, cannot perform connection check.\n", ni.instanceIndex)
				return
			}

			// Check if this is the LAST connection to this peer
			if len(ni.host.Network().ConnsToPeer(remotePeerID)) == 0 {
				// If it's a friendlyPeer, wait for the grace period, otherwise close immediately
				ni.peersMutex.RLock()
				_, isFriendly := ni.friendlyPeers[remotePeerID]
				ni.peersMutex.RUnlock()

				if isFriendly {
					logger.Debugf("[GO] ⏳ Instance %d: Last connection to %s closed. Starting %v grace period timer...\n", ni.instanceIndex, remotePeerID, DisconnectionGracePeriod)

					// We create a context that we can cancel if they reconnect
					ctx, cancelTimer := context.WithCancel(context.Background())
					
					ni.disconnectionMutex.Lock()
					// If a timer already exists (rare race condition), cancel the old one first
					if oldCancel, exists := ni.disconnectionTimers[remotePeerID]; exists {
						oldCancel()
					}
					ni.disconnectionTimers[remotePeerID] = cancelTimer
					ni.disconnectionMutex.Unlock()

					// Run cleanup in a goroutine
					go func() {
						select {
						case <-time.After(DisconnectionGracePeriod):
							// Timer expired! Proceed to cleanup.
						case <-ctx.Done():
							// Context cancelled (user reconnected). Stop here.
							return
						case <-ni.ctx.Done():
							// Node is shutting down. Stop here.
							return
						}

						// --- Timer Expired: Execute Cleanup ---
						// Remove from timer map
						ni.disconnectionMutex.Lock()
						// Double-check: did we get cancelled while waiting for lock?
						if ctx.Err() != nil {
							ni.disconnectionMutex.Unlock()
							return
						}
						delete(ni.disconnectionTimers, remotePeerID)
						ni.disconnectionMutex.Unlock()

						// Final Safety Check: Are they actually connected now?
						// (Handles race where they reconnect exactly when timer fires)
						if len(ni.host.Network().ConnsToPeer(remotePeerID)) > 0 {
							logger.Debugf("[GO] ⚠️ Instance %d: Grace period expired for %s, but peer is connected again. Skipping cleanup.\n", ni.instanceIndex, remotePeerID)
							return
						}

						logger.Debugf("[GO] 🗑️ Instance %d: Grace period ended for %s. Removing peer data.\n", ni.instanceIndex, remotePeerID)

						// 3. Clean up friendlyPeers
						ni.peersMutex.Lock()
						if _, exists := ni.friendlyPeers[remotePeerID]; exists {
							delete(ni.friendlyPeers, remotePeerID)
							logger.Debugf("[GO]   Instance %d: Removed %s from friendlyPeers.\n", ni.instanceIndex, remotePeerID)
						}
						ni.peersMutex.Unlock()

						// 4. Clean up persistent streams
						ni.streamsMutex.Lock()
						if stream, ok := ni.persistentChatStreams[remotePeerID]; ok {
							logger.Debugf("[GO]   Instance %d: Cleaning up persistent stream for %s.\n", ni.instanceIndex, remotePeerID)
							_ = stream.Close() 
							delete(ni.persistentChatStreams, remotePeerID)
						}
						ni.streamsMutex.Unlock()

					}()
				} else {
					logger.Debugf("[GO]   Instance %d: Last connection to %s closed. Removing from tracked peers.\n", ni.instanceIndex, remotePeerID)

					// Also clean up persistent stream if one existed for this peer
					ni.streamsMutex.Lock()
					if stream, ok := ni.persistentChatStreams[remotePeerID]; ok {
						logger.Debugf("[GO]   Instance %d: Cleaning up persistent stream for disconnected peer %s via DisconnectedF notifier.\n", ni.instanceIndex, remotePeerID)
						_ = stream.Close() // Attempt graceful close
						delete(ni.persistentChatStreams, remotePeerID)
					}
					ni.streamsMutex.Unlock()
					}
			} else {
				logger.Debugf("[GO]   Instance %d: DisconnectedF: Still have %d active connections to %s, not removing.\n", ni.instanceIndex, len(ni.host.Network().ConnsToPeer(remotePeerID)), remotePeerID)
			}
		},
	})
}

// enforceProtocolCompliance ensures that any connected peer supports the required chat protocol.
// If a peer finishes identification but lacks the protocol, they are immediately disconnected.
func enforceProtocolCompliance(ni *NodeInstance) {
	// 1. Subscribe to the identification completed event
	sub, err := ni.host.EventBus().Subscribe(new(event.EvtPeerIdentificationCompleted))
	if err != nil {
		logger.Errorf("[GO] ❌ Instance %d: Failed to subscribe to identification events: %v", ni.instanceIndex, err)
		return
	}

	logger.Infof("[GO] 🛡️ Instance %d: Strict Isolation ENABLED. Monitoring for non-compliant peers.", ni.instanceIndex)

	go func() {
		defer sub.Close()
		for {
			select {
			case <-ni.ctx.Done():
				return
			case evt, ok := <-sub.Out():
				if !ok {
					return
				}
				idEvt := evt.(event.EvtPeerIdentificationCompleted)

				// Skip check for self
				if idEvt.Peer == ni.host.ID() {
					continue
				}

				isCompliant := false
				for _, proto := range idEvt.Protocols {
					if string(proto) == UnaiverseChatProtocol {
						isCompliant = true
						break
					}
				}

				// 4. Action: Disconnect if not compliant
				if !isCompliant {
					logger.Warnf("[GO] 🚫 Instance %d: Kicking peer %s. (Reason: Protocol Mismatch).", ni.instanceIndex, idEvt.Peer)
					// Disconnect
					ni.host.Network().ClosePeer(idEvt.Peer)
					// Optional: Clean from peerstore to free memory immediately
					ni.host.Peerstore().RemovePeer(idEvt.Peer)
				} else {
					logger.Debugf("[GO] ✅ Instance %d: Peer %s verified compliant.", ni.instanceIndex, idEvt.Peer)
				}
			}
		}
	}()
}

// handleAddressUpdateEvents listens for libp2p address changes and updates the local cache.
func handleAddressUpdateEvents(ni *NodeInstance, sub event.Subscription) {
	defer sub.Close()

	// Initialize cache immediately with current state to avoid race conditions at startup
	ni.addrMutex.Lock()
	ni.localAddrs = ni.host.Addrs()
	ni.addrMutex.Unlock()

	for {
		select {
		case <-ni.ctx.Done():
			return
		case _, ok := <-sub.Out():
			if !ok {
				return
			}
			// We only use the event as a trigger but we take the addresses from the Host
			allAddresses := ni.host.Addrs()
			ni.addrMutex.Lock()
        	ni.localAddrs = allAddresses
            ni.addrMutex.Unlock()
            
            // Log addresses to verify
            addrsStr := make([]string, len(allAddresses))
            for i, a := range allAddresses {
                addrsStr[i] = a.String()
            }
			logger.Infof("[GO] 🔄 Instance %d: Updated local addresses (updating cache). Addrs: %v", ni.instanceIndex, addrsStr)
		}
	}
}

// Helper to filter peers for PeerSource
func (ni *NodeInstance) isSuitableForPeerSource(pid peer.ID) bool {
	ps := ni.host.Peerstore()

	// 1. Check for Relay Hop Protocol (NEEDED)
	protocols, err := ps.GetProtocols(pid)
	if err != nil {
		return false
	}
	isRelay := false
	for _, proto := range protocols {
		if proto == "/libp2p/circuit/relay/0.2.0/hop" {
			isRelay = true
			break
		}
	}
	if !isRelay {
		return false
	}

	// Only accept wss-enabled nodes as relay
	addrs := ps.Addrs(pid)
	isSuitable := false
	for _, addr := range addrs {
		_, err = addr.ValueForProtocol(ma.P_WS) 
		if err == nil {
			_, err := addr.ValueForProtocol(ma.P_TLS) 
			if err == nil {
				isSuitable = true
			}
		}
	}

	return isSuitable
}

// PeerSource acts as the peer discovery backend for AutoRelay.
// It combines a local cache lookup (fast/free) with a DHT random walk (slow/expensive).
func (ni *NodeInstance) PeerSource(ctx context.Context, numPeers int) <-chan peer.AddrInfo {
	out := make(chan peer.AddrInfo)

	go func() {
		defer close(out)
		
		// Safety checks: Ensure host and DHT are fully initialized
		if ni.host == nil || ni.dht == nil {
			return
		}

		// Keep track of peers we've already sent in this batch
		sentPeers := make(map[peer.ID]struct{})
		peersFound := 0

		// --- PHASE 1: Scavenge Local Peerstore ---
		localPeers := ni.host.Peerstore().Peers()
		for _, pid := range localPeers {
			if peersFound >= numPeers {
				return
			}
			if pid == ni.host.ID() {
				continue
			}

			// Add it if it meets our criteria
			if ni.isSuitableForPeerSource(pid) {
				info := ni.host.Peerstore().PeerInfo(pid)
				if len(info.Addrs) == 0 {
					continue
				}

				select {
				case out <- info:
					sentPeers[pid] = struct{}{}
					peersFound++
				case <-ctx.Done():
					return
				}
			}
		}

		// --- PHASE 2: DHT Random Walk ---
		if peersFound < numPeers {
			logger.Debugf("[GO] ⚠️ Instance %d: Local peerstore insufficient (%d/%d). Starting DHT walk...",
				ni.instanceIndex, peersFound, numPeers)

			for peersFound < numPeers {
				randomKey := make([]byte, 32)
				rand.Read(randomKey)
				randomKeyStr := string(randomKey)

				candidatePIDs, err := ni.dht.GetClosestPeers(ctx, randomKeyStr)
				if err != nil {
					select {
					case <-ctx.Done():
						return
					case <-time.After(2 * time.Second):
						continue
					}
				}

				for _, pid := range candidatePIDs {
					if peersFound >= numPeers {
						return
					}
					if pid == ni.host.ID() {
						continue
					}
					if _, alreadySent := sentPeers[pid]; alreadySent {
						continue
					}

					info := ni.host.Peerstore().PeerInfo(pid)
					if len(info.Addrs) > 0 {
						select {
						case out <- info:
							sentPeers[pid] = struct{}{}
							peersFound++
						case <-ctx.Done():
							return
						}
					}
				}
			}
		}
	}()

	return out
}

// goGetNodeAddresses is the internal Go function that performs the core logic
// of fetching and formatting node addresses.
// It takes a pointer to a NodeInstance and a targetPID. If targetPID is empty (peer.ID("")),
// it fetches addresses for the local node of the given instance.
// It returns a slice of fully ma.Multiaddr.
func goGetNodeAddresses(
    ni *NodeInstance,
    targetPID peer.ID,
) ([]ma.Multiaddr, error) {
	if ni.host == nil {
		errMsg := fmt.Sprintf("Instance %d: Host not initialized", ni.instanceIndex)
		logger.Errorf("[GO] ❌ goGetNodeAddresses: %s\n", errMsg)
		return nil, fmt.Errorf("%s", errMsg)
	}

	// Determine the actual Peer ID to resolve addresses for.
	resolvedPID := targetPID
	isThisNode := false
	if targetPID == "" || targetPID == ni.host.ID() {
		resolvedPID = ni.host.ID()
		isThisNode = true
	}

	// --- 1. Gather all candidate addresses from the host and peerstore ---
	var candidateAddrs []ma.Multiaddr
	if isThisNode {
		ni.addrMutex.RLock()
		candidateAddrs = append(candidateAddrs, ni.localAddrs...)
		candidateAddrs = append(candidateAddrs, ni.privateRelayAddrs...)
		ni.addrMutex.RUnlock()
	} else {
		// --- Remote Peer Addresses ---
		ni.peersMutex.RLock()
		if epi, exists := ni.friendlyPeers[resolvedPID]; exists {
			candidateAddrs = append(candidateAddrs, epi.Addrs...)
		}
		ni.peersMutex.RUnlock()
		candidateAddrs = append(candidateAddrs, ni.host.Peerstore().Addrs(resolvedPID)...)
	}

	// --- 2. Process and filter candidate addresses ---
	addrSet := make(map[string]ma.Multiaddr)
	for _, addr := range candidateAddrs {
		// if addr == nil || manet.IsIPLoopback(addr) || manet.IsIPUnspecified(addr) {
		// 	continue
		// }

		// Use the idiomatic `peer.SplitAddr` to check if the address already includes a Peer ID.
		var finalAddr ma.Multiaddr
		transportAddr, idInAddr := peer.SplitAddr(addr)
		if transportAddr == nil {
			continue
		}

		// handle cases for different transport protocols
		if strings.HasPrefix(transportAddr.String(), "/p2p-circuit/") {
			continue
		}
		if strings.Contains(transportAddr.String(), "*") {
			continue
		}

		// handle cases based on presence and correctness of Peer ID in the address
        if idInAddr == resolvedPID {
            finalAddr = addr
        } else if idInAddr == "" {
            p2pComponent, _ := ma.NewMultiaddr(fmt.Sprintf("/p2p/%s", resolvedPID.String()))
            finalAddr = addr.Encapsulate(p2pComponent)
        } else {
            logger.Warnf("[GO] ⚠️ Instance %d: Discarding stale address for peer %s: %s\n", ni.instanceIndex, resolvedPID, addr)
            continue
        }
		addrSet[finalAddr.String()] = finalAddr
	}

	// --- 4. Convert the final set of unique addresses to a slice for returning. ---
	result := make([]ma.Multiaddr, 0, len(addrSet))
	for _, addr := range addrSet {
        result = append(result, addr)
    }

	if len(result) == 0 {
		logger.Warnf("[GO] ⚠️ goGetNodeAddresses: No suitable addresses found for peer %s.", resolvedPID)
	}

	return result, nil
}

// Close gracefully shuts down all components of this node instance.
// This REPLACES the old `closeSingleInstance` function.
func (ni *NodeInstance) Close() error {
	logger.Infof("[GO] 🛑 Instance %d: Closing node...", ni.instanceIndex)

	// --- Stop Cert Manager FIRST ---
	if ni.certManager != nil {
		logger.Debugf("[GO]   - Instance %d: Stopping AutoTLS cert manager...\n", ni.instanceIndex)
		ni.certManager.Stop()
	}

	// --- Cancel Main Context ---
	if ni.cancel != nil {
		logger.Debugf("[GO]   - Instance %d: Cancelling main context...\n", ni.instanceIndex)
		ni.cancel()
	}

	// Give goroutines time to react to context cancellation
	time.Sleep(200 * time.Millisecond)

	// --- Close DHT Client ---
	if ni.dht != nil {
		logger.Debugf("[GO]   - Instance %d: Closing DHT...\n", ni.instanceIndex)
		if err := ni.dht.Close(); err != nil {
			logger.Warnf("[GO] ⚠️ Instance %d: Error closing DHT: %v\n", ni.instanceIndex, err)
		}
		ni.dht = nil
	}

	// --- Close Persistent Outgoing Streams ---
	ni.streamsMutex.Lock()
	if len(ni.persistentChatStreams) > 0 {
		logger.Debugf("[GO]   - Instance %d: Closing %d persistent outgoing streams...\n", ni.instanceIndex, len(ni.persistentChatStreams))
		for pid, stream := range ni.persistentChatStreams {
			logger.Debugf("[GO]     - Instance %d: Closing stream to %s\n", ni.instanceIndex, pid)
			_ = stream.Close() // Attempt graceful close
		}
	}
	ni.persistentChatStreams = make(map[peer.ID]network.Stream) // Clear the map
	ni.streamsMutex.Unlock()

	// --- Clean Up PubSub State ---
	ni.pubsubMutex.Lock()
	if len(ni.subscriptions) > 0 {
		logger.Debugf("[GO]   - Instance %d: Ensuring PubSub subscriptions (%d) are cancelled...\n", ni.instanceIndex, len(ni.subscriptions))
		for channel, sub := range ni.subscriptions {
			logger.Debugf("[GO]     - Instance %d: Cancelling subscription to topic: %s\n", ni.instanceIndex, channel)
			sub.Cancel()
		}
	}
	ni.subscriptions = make(map[string]*pubsub.Subscription) // Clear the map
	ni.topics = make(map[string]*pubsub.Topic)               // Clear the map
	ni.pubsubMutex.Unlock()

	// --- Close Host Instance ---
	var hostErr error
	if ni.host != nil {
		logger.Debugf("[GO]   - Instance %d: Closing host instance...\n", ni.instanceIndex)
		hostErr = ni.host.Close()
		if hostErr != nil {
			logger.Warnf("[GO] ⚠️ %s (proceeding with cleanup)\n", hostErr)
		} else {
			logger.Debugf("[GO]   - Instance %d: Host closed successfully.\n", ni.instanceIndex)
		}
	}

	// --- Clear Remaining State for this instance ---
	ni.peersMutex.Lock()
	ni.friendlyPeers = make(map[peer.ID]ExtendedPeerInfo) // Clear the map
	ni.peersMutex.Unlock()

	// Clear also the addresses
	ni.addrMutex.Lock()
	ni.localAddrs = nil
	ni.privateRelayAddrs = nil
	ni.addrMutex.Unlock()

	// Clear the MessageStore for this instance
	if ni.messageStore != nil {
		ni.messageStore.mu.Lock()
		ni.messageStore.messagesByChannel = make(map[string]*list.List) // Clear the message store
		ni.messageStore.mu.Unlock()
	}
	logger.Debugf("[GO]   - Instance %d: Cleared connected peers map and message buffer.\n", ni.instanceIndex)

	// Clear the rendezvous state for this instance
	ni.rendezvousMutex.Lock()
	ni.rendezvousState = nil // Clear the state
	ni.rendezvousMutex.Unlock()

	// Explicitly cancel all running grace period timers so goroutines exit immediately.
	ni.disconnectionMutex.Lock()
	if len(ni.disconnectionTimers) > 0 {
		logger.Debugf("[GO]   - Instance %d: Cancelling %d active disconnection timers...\n", ni.instanceIndex, len(ni.disconnectionTimers))
		for _, cancelTimer := range ni.disconnectionTimers {
			cancelTimer()
		}
	}
	ni.disconnectionTimers = nil // Clear the map
	ni.disconnectionMutex.Unlock()

	// Nil out components to signify the instance is fully closed
	ni.host = nil
	ni.pubsub = nil
	ni.ctx = nil
	ni.cancel = nil
	ni.certManager = nil
	ni.messageStore = nil

	if hostErr != nil {
		return hostErr
	}

	logger.Infof("[GO] ✅ Instance %d: Node closed successfully.\n", ni.instanceIndex)
	return nil
}
