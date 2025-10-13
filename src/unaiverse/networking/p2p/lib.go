// lib.go
// This Go program compiles into a C shared library (.so file on Linux/macOS, .dll on Windows)
// exposing libp2p functionalities (host creation, peer connection, pubsub, direct messaging)
// for use by other languages, primarily Python via CFFI or ctypes.
package main

/*
#include <stdlib.h>
*/
import "C" // Enables CGo features, allowing Go to call C code and vice-versa.

import (
	// Standard Go libraries
	"bytes"           // For byte buffer manipulations (e.g., encoding/decoding, separators)
	"container/list"  // For an efficient ordered list (doubly-linked list for queues)
	"context"         // For managing cancellation signals and deadlines across API boundaries and goroutines
	"crypto/rand"     // For generating identity keys
	"crypto/tls"      // For TLS configuration and certificates
	"encoding/base64" // For encoding binary message data into JSON-safe strings
	"encoding/binary" // For encoding/decoding length prefixes in stream communication
	"encoding/json"   // For marshalling/unmarshalling data structures to/from JSON (used for C API communication)
	"fmt"             // For formatted string creation and printing
	"io"              // For input/output operations (e.g., reading from streams)
	"log"             // For logging information, warnings, and errors
	"net"             // For network-related errors and interfaces
	"os"              // For interacting with the operating system (e.g., Stdout)
	"path/filepath"   // For file path manipulations (e.g., saving/loading identity keys)
	"strings"         // For string manipulations (e.g., trimming, splitting)
	"sync"            // For synchronization primitives like Mutexes and RWMutexes to protect shared data
	"time"            // For time-related functions (e.g., timeouts, timestamps)
	"unsafe"          // For using Go pointers with C code (specifically C.free)

	// Core libp2p libraries
	libp2p "github.com/libp2p/go-libp2p"                          // Main libp2p package for creating a host
	dht "github.com/libp2p/go-libp2p-kad-dht"                     // Kademlia DHT implementation for peer discovery and routing
	"github.com/libp2p/go-libp2p/core/crypto"                     // Defines cryptographic primitives (keys, signatures)
	"github.com/libp2p/go-libp2p/core/event"                      // Event bus for subscribing to libp2p events (connections, reachability changes)
	"github.com/libp2p/go-libp2p/core/host"                       // Defines the main Host interface, representing a libp2p node
	"github.com/libp2p/go-libp2p/core/network"                    // Defines network interfaces like Stream and Connection
	"github.com/libp2p/go-libp2p/core/peer"                       // Defines Peer ID and AddrInfo types
	"github.com/libp2p/go-libp2p/core/peerstore"                  // Defines the Peerstore interface for storing peer metadata (addresses, keys)
	"github.com/libp2p/go-libp2p/core/routing"                    // Defines the Routing interface for peer routing (e.g., DHT)
	rcmgr "github.com/libp2p/go-libp2p/p2p/host/resource-manager" // For managing resources (bandwidth, memory) for libp2p hosts
	"github.com/libp2p/go-libp2p/p2p/protocol/circuitv2/client"   // For establishing outbound relayed connections (acting as a client)
	rc "github.com/libp2p/go-libp2p/p2p/protocol/circuitv2/relay" // Import for relay service options

	// transport protocols for libp2p
	quic "github.com/libp2p/go-libp2p/p2p/transport/quic"                 // QUIC transport for peer-to-peer connections (e.g., for mobile devices)
	"github.com/libp2p/go-libp2p/p2p/transport/tcp"                       // TCP transport for peer-to-peer connections (most common)
	webrtc "github.com/libp2p/go-libp2p/p2p/transport/webrtc"             // WebRTC transport for peer-to-peer connections (e.g., for browsers or mobile devices)
	ws "github.com/libp2p/go-libp2p/p2p/transport/websocket"              // WebSocket transport for peer-to-peer connections (e.g., for browsers)
	webtransport "github.com/libp2p/go-libp2p/p2p/transport/webtransport" // WebTransport transport for peer-to-peer connections (e.g., for browsers)

	// --- AutoTLS Imports ---
	"github.com/caddyserver/certmagic"                // Automatic TLS certificate management (used by p2p-forge)
	golog "github.com/ipfs/go-log/v2"                 // IPFS logging library for structured logging
	p2pforge "github.com/ipshipyard/p2p-forge/client" // p2p-forge library for automatic TLS and domain management

	// protobuf
	pg "unaiverse/networking/p2p/lib/proto-go" // Generated Protobuf code for our message formats

	"google.golang.org/protobuf/proto" // Core Protobuf library for marshalling/unmarshalling messages

	// PubSub library
	pubsub "github.com/libp2p/go-libp2p-pubsub" // GossipSub implementation for publish/subscribe messaging

	// Multiaddr libraries (libp2p's addressing format)
	ma "github.com/multiformats/go-multiaddr"        // Core multiaddr parsing and manipulation
	manet "github.com/multiformats/go-multiaddr/net" // Utilities for working with multiaddrs and net interfaces (checking loopback, etc.)
)

// ChatProtocol defines the protocol ID string used for direct peer-to-peer messaging streams.
// This ensures that both peers understand how to interpret the data on the stream.
// const UnaiverseChatProtocol = "/unaiverse-chat-protocol/1.0.0"
const UnaiverseChatProtocol = "/chat/1.0.0"
const UnaiverseUserAgent = "go-libp2p/example/autotls"

// ExtendedPeerInfo holds information about a connected peer.
type ExtendedPeerInfo struct {
	ID          peer.ID        `json:"id"`           // the Peer ID of the connected peer.
	Addrs       []ma.Multiaddr `json:"addrs"`        // the Multiaddr(s) associated with the peer.
	ConnectedAt time.Time      `json:"connected_at"` // Timestamp when the connection was established.
	Direction   string         `json:"direction"`    // Direction of the connection: "inbound" or "outbound".
	Misc        int            `json:"misc"`         // Misc information (integer), custom usage
}

// RendezvousState holds the discovered peers from a rendezvous topic,
// along with metadata about the freshness of the data.
type RendezvousState struct {
	Peers       map[peer.ID]ExtendedPeerInfo `json:"peers"`
	UpdateCount int64                        `json:"update_count"`
}

// QueuedMessage represents a message received either directly or via PubSub.
//
// This lightweight version stores the binary payload in the `Data` field,
// while the `From` field contains the Peer ID of the sender for security reasons.
// It has to match with the 'sender' field in the ProtoBuf payload of the message.
type QueuedMessage struct {
	From peer.ID `json:"from"` // The VERIFIED peer ID of the sender from the network layer.
	Data []byte  `json:"-"`    // The raw data payload (Protobuf encoded).
}

// MessageStore holds the QueuedMessages for each channel in separate FIFO queues.
// It has a maximum number of channels and a maximum queue length per channel.
type MessageStore struct {
	mu                sync.Mutex            // protects the message store from concurrent access.
	messagesByChannel map[string]*list.List // stores a FIFO queue of messages for each channel
}

// CreateNodeResponse defines the structure of our success message.
type CreateNodeResponse struct {
	Addresses []string `json:"addresses"`
	IsPublic  bool     `json:"isPublic"`
}

// --- Create a package-level logger ---
var logger = golog.Logger("p2p-library")

// --- Multi-Instance State Management ---
var (
	// Set the libp2p configuration parameters.
	maxInstances       int
	maxChannelQueueLen int
	maxUniqueChannels  int
	MaxMessageSize     uint32

	// Slices to hold state for each instance. Using arrays for fixed size.
	hostInstances                      []host.Host
	pubsubInstances                    []*pubsub.PubSub
	contexts                           []context.Context
	cancelContexts                     []context.CancelFunc
	topicsInstances                    []map[string]*pubsub.Topic
	subscriptionsInstances             []map[string]*pubsub.Subscription
	connectedPeersInstances            []map[peer.ID]ExtendedPeerInfo
	rendezvousDiscoveredPeersInstances []*RendezvousState
	persistentChatStreamsInstances     []map[peer.ID]network.Stream
	messageStoreInstances              []*MessageStore
	certManagerInstances               []*p2pforge.P2PForgeCertMgr

	// Mutexes for protecting concurrent access to instance-specific data.
	connectedPeersMutexes            []sync.RWMutex
	persistentChatStreamsMutexes     []sync.Mutex
	pubsubMutexes                    []sync.RWMutex // Protects topicsInstances and subscriptionsInstances
	rendezvousDiscoveredPeersMutexes []sync.RWMutex // Mutexes for protecting concurrent access to rendezvousDiscoveredPeersInstances.

	// Global mutex to protect access to the instance state slices themselves
	// (e.g., during initialization or checking if an instance exists).
	// Use sparingly to avoid contention.
	instanceStateMutex sync.RWMutex

	// Flag to track if a specific instance index has been initialized
	isInitialized []bool
)

// --- Helper Functions ---
// jsonErrorResponse creates a JSON string representing an error state.
// It takes a base message and an optional error, formats them, escapes the message
// for JSON embedding, and returns a C string pointer (`*C.char`).
// The caller (usually C/Python) is responsible for freeing this C string using FreeString.
func jsonErrorResponse(
	message string,
	err error,
) *C.char {

	errMsg := message
	if err != nil {
		errMsg = fmt.Sprintf("%s: %s", message, err.Error())
	}
	logger.Errorf("[GO] ❌ Error: %s", errMsg)
	// Ensure error messages are escaped properly for JSON embedding
	escapedErrMsg := escapeStringForJSON(errMsg)
	// Format into a standard {"state": "Error", "message": "..."} JSON structure.
	jsonError := fmt.Sprintf(`{"state":"Error","message":"%s"}`, escapedErrMsg)
	// Convert the Go string to a C string (allocates memory in C heap).
	return C.CString(jsonError)
}

// jsonSuccessResponse creates a JSON string representing a success state.
// It takes an arbitrary Go object (`message`), marshals it into JSON, wraps it
// in a standard {"state": "Success", "message": {...}} structure, and returns
// a C string pointer (`*C.char`).
// The caller (usually C/Python) is responsible for freeing this C string using FreeString.
func jsonSuccessResponse(
	message interface{},
) *C.char {

	// Marshal the provided Go data structure into JSON bytes.
	jsonData, err := json.Marshal(message)
	if err != nil {
		// If marshalling fails, return a JSON error response instead.
		return jsonErrorResponse("Failed to marshal success response", err)
	}
	// Format into the standard success structure.
	jsonSuccess := fmt.Sprintf(`{"state":"Success","message":%s}`, string(jsonData))
	// Convert the Go string to a C string (allocates memory in C heap).
	return C.CString(jsonSuccess)
}

// escapeStringForJSON performs basic escaping of characters (like double quotes and backslashes)
// within a string to ensure it's safe to embed within a JSON string value.
// It uses Go's standard JSON encoder for robust escaping.
func escapeStringForJSON(
	s string,
) string {

	var buf bytes.Buffer
	// Encode the string using Go's JSON encoder, which handles escaping.
	json.NewEncoder(&buf).Encode(s)
	// The encoder adds surrounding quotes and a trailing newline, which we remove.
	res := buf.String()
	// Check bounds before slicing to avoid panic.
	if len(res) > 2 && res[0] == '"' && res[len(res)-2] == '"' {
		return res[1 : len(res)-2] // Trim surrounding quotes and newline
	}
	// Fallback if encoding behaves unexpectedly (e.g., empty string).
	return s
}

// newMessageStore initializes a new MessageStore.
func newMessageStore() *MessageStore {
	return &MessageStore{
		messagesByChannel: make(map[string]*list.List),
	}
}

// checkInstanceIndex performs bounds checking on the provided instance index.
func checkInstanceIndex(
	instanceIndex int,
) error {

	if instanceIndex < 0 || instanceIndex >= maxInstances {
		return fmt.Errorf("invalid instance index: %d. Must be between 0 and %d", instanceIndex, maxInstances-1)
	}
	return nil
}

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

func cleanupFailedCreate(instanceIndex int) {
	logger.Infof("[GO] 🧹 Instance %d: Cleaning up after failed creation...", instanceIndex)
	if certManagerInstances[instanceIndex] != nil {
		certManagerInstances[instanceIndex].Stop()
		certManagerInstances[instanceIndex] = nil
	}
	if hostInstances[instanceIndex] != nil { // Attempt cleanup before returning.
		hostInstances[instanceIndex].Close()
		hostInstances[instanceIndex] = nil
	}
	if cancelContexts[instanceIndex] != nil {
		cancelContexts[instanceIndex]()
	}
	// Set all instance state to nil
	pubsubInstances[instanceIndex] = nil
	contexts[instanceIndex] = nil
	cancelContexts[instanceIndex] = nil
	topicsInstances[instanceIndex] = nil
	subscriptionsInstances[instanceIndex] = nil
	connectedPeersInstances[instanceIndex] = nil
	rendezvousDiscoveredPeersInstances[instanceIndex] = nil
	persistentChatStreamsInstances[instanceIndex] = nil
	messageStoreInstances[instanceIndex] = nil
	// Clear the mutexes for this instance
	connectedPeersMutexes[instanceIndex] = sync.RWMutex{}            // Reset to a new mutex
	persistentChatStreamsMutexes[instanceIndex] = sync.Mutex{}       // Reset to a new mutex
	pubsubMutexes[instanceIndex] = sync.RWMutex{}                    // Reset to a new mutex
	rendezvousDiscoveredPeersMutexes[instanceIndex] = sync.RWMutex{} // Reset to a new mutex
	// Reset the isInitialized flag for this instance
	instanceStateMutex.Lock()
	isInitialized[instanceIndex] = false // Mark as uninitialized again
	instanceStateMutex.Unlock()
}

func getListenAddrs(ipsJSON string, tcpPort int, tlsMode string) ([]ma.Multiaddr, error) {
	var ips []string
	// --- Parse IPs from JSON ---
	if ipsJSON == "" || ipsJSON == "[]" {
		ips = []string{"0.0.0.0"} // Default if empty or not provided
	} else {
		if err := json.Unmarshal([]byte(ipsJSON), &ips); err != nil {
			return nil, fmt.Errorf("failed to parse IPs JSON: %w", err)
		}
		if len(ips) == 0 { // Handle case of valid but empty JSON array "[]"
			ips = []string{"0.0.0.0"}
		}
	}

	var listenAddrs []ma.Multiaddr
	quicPort := 0
	webrtcPort := 0
	webtransPort := 0
	if tcpPort != 0 {
		quicPort = tcpPort + 1
		webrtcPort = tcpPort + 2
		webtransPort = tcpPort + 3
	}

	// --- Create Multiaddrs for both protocols from the single IP list ---
	for _, ip := range ips {
		// TCP
		tcpMaddr, _ := ma.NewMultiaddr(fmt.Sprintf("/ip4/%s/tcp/%d", ip, tcpPort))
		// QUIC
		quicMaddr, _ := ma.NewMultiaddr(fmt.Sprintf("/ip4/%s/udp/%d/quic-v1", ip, quicPort))
		// WebTransport
		webtransMaddr, _ := ma.NewMultiaddr(fmt.Sprintf("/ip4/%s/udp/%d/quic-v1/webtransport", ip, webtransPort))
		// WebRTC
		webrtcMaddr, _ := ma.NewMultiaddr(fmt.Sprintf("/ip4/%s/udp/%d/webrtc-direct", ip, webrtcPort))

		listenAddrs = append(listenAddrs, tcpMaddr, quicMaddr, webrtcMaddr, webtransMaddr)

		if tlsMode == "autotls" {
			// This is the special multiaddr that triggers AutoTLS
			wssMaddr, _ := ma.NewMultiaddr(fmt.Sprintf("/ip4/%s/tcp/%d/tls/sni/*.%s/ws", ip, tcpPort, p2pforge.DefaultForgeDomain))
			listenAddrs = append(listenAddrs, wssMaddr)
		} else if tlsMode == "domain" {
			// This is the standard secure WebSocket address with provided domain
			wssMaddr, _ := ma.NewMultiaddr(fmt.Sprintf("/ip4/%s/tcp/%d/tls/ws", ip, tcpPort))
			listenAddrs = append(listenAddrs, wssMaddr)
		} else {
			// Fallback to a standard, non-secure WebSocket address
			wsMaddr, _ := ma.NewMultiaddr(fmt.Sprintf("/ip4/%s/tcp/%d/ws", ip, tcpPort))
			listenAddrs = append(listenAddrs, wsMaddr)
		}
	}

	logger.Debugf("[GO] 🔧 Prepared Listen Addresses: %v\n", listenAddrs)

	return listenAddrs, nil
}

func createResourceManager(maxConnections int) (network.ResourceManager, error) {
	limits := rcmgr.DefaultLimits
	libp2p.SetDefaultServiceLimits(&limits)
	myLimits := rcmgr.PartialLimitConfig{
		System: rcmgr.ResourceLimits{Conns: rcmgr.LimitVal(maxConnections)},
	}
	concreteLimits := myLimits.Build(limits.AutoScale())
	return rcmgr.NewResourceManager(rcmgr.NewFixedLimiter(concreteLimits))
}

func setupPubSub(instanceIndex int) error {
	instanceCtx := contexts[instanceIndex]
	instanceHost := hostInstances[instanceIndex]
	psOptions := []pubsub.Option{
		// pubsub.WithFloodPublish(true),
		pubsub.WithMaxMessageSize(int(MaxMessageSize)),
	}
	ps, err := pubsub.NewGossipSub(instanceCtx, instanceHost, psOptions...)
	if err != nil {
		return err
	}
	pubsubInstances[instanceIndex] = ps
	return nil
}

// setupNotifiers would contain the existing NotifierBundle logic
func setupNotifiers(instanceIndex int) {
	hostInstances[instanceIndex].Network().Notify(&network.NotifyBundle{
		ConnectedF: func(_ network.Network, conn network.Conn) {
			logger.Debugf("[GO] 🔔 Instance %d: Event - Connected to %s (Direction: %s)\n", instanceIndex, conn.RemotePeer(), conn.Stat().Direction)

			remotePeerID := conn.RemotePeer()
			instanceHost := hostInstances[instanceIndex] // Available in CreateNode's scope

			// --- 1. Gather all candidate addresses into a single slice ---
			candidateAddrs := make([]ma.Multiaddr, 0)
			candidateAddrs = append(candidateAddrs, conn.RemoteMultiaddr())
			candidateAddrs = append(candidateAddrs, instanceHost.Peerstore().Addrs(remotePeerID)...)

			// --- 2. Filter, format, and deduplicate in a single pass ---
			finalPeerAddrs := make([]ma.Multiaddr, 0)
			uniqueAddrStrings := make(map[string]struct{}) // Using an empty struct is more memory-efficient for a "set"

			for _, addr := range candidateAddrs {
				if addr == nil || manet.IsIPLoopback(addr) || manet.IsIPUnspecified(addr) {
					continue
				}

				// Ensure the address is fully qualified with the peer's ID
				var fullAddrStr string
				if _, idInAddr := peer.SplitAddr(addr); idInAddr == "" {
					fullAddrStr = fmt.Sprintf("%s/p2p/%s", addr.String(), remotePeerID.String())
				} else {
					fullAddrStr = addr.String()
				}

				// If we haven't seen this exact address string before, add it.
				if _, exists := uniqueAddrStrings[fullAddrStr]; !exists {
					maddr, err := ma.NewMultiaddr(fullAddrStr)
					if err == nil {
						finalPeerAddrs = append(finalPeerAddrs, maddr)
						uniqueAddrStrings[fullAddrStr] = struct{}{}
					}
				}
			}

			if len(finalPeerAddrs) == 0 {
				logger.Debugf("[GO]   Instance %d: ConnectedF: Could not find any non-local addresses for %s immediately.\n", instanceIndex, remotePeerID)
			}

			// --- 3. Determine the direction ---
			var directionString string
			switch conn.Stat().Direction {
			case network.DirInbound:
				directionString = "incoming"
			case network.DirOutbound:
				directionString = "outgoing"
			default:
				directionString = "unknown"
			}

			// --- 4. Update the connected peers list ---
			instanceConnectedPeersMutex := &connectedPeersMutexes[instanceIndex]
			instanceConnectedPeers := connectedPeersInstances[instanceIndex]

			instanceConnectedPeersMutex.Lock()
			// It's possible this peer was already in the map if ConnectTo ran first,
			// or if there were multiple connection events. Update generously.
			if epi, exists := instanceConnectedPeers[remotePeerID]; exists {
				epi.Addrs = finalPeerAddrs // Update with the new comprehensive list
				// epi.Direction = directionString
				instanceConnectedPeers[remotePeerID] = epi
			} else {
				instanceConnectedPeers[remotePeerID] = ExtendedPeerInfo{
					ID:          remotePeerID,
					Addrs:       finalPeerAddrs,
					ConnectedAt: time.Now(),
					Direction:   directionString,
					Misc:        0,
				}
			}
			instanceConnectedPeersMutex.Unlock()

			logger.Debugf("[GO]   Instance %d: Updated ConnectedPeers for %s via ConnectedF. Total addresses: %d. List: %v\n", instanceIndex, remotePeerID, len(finalPeerAddrs), finalPeerAddrs)
		},
		DisconnectedF: func(_ network.Network, conn network.Conn) {
			logger.Debugf("[GO] 🔔 Instance %d: Event - Disconnected from %s\n", instanceIndex, conn.RemotePeer())
			remotePeerID := conn.RemotePeer()

			// Get the host for this instance to query its network state.
			instanceHost := hostInstances[instanceIndex]
			if instanceHost == nil {
				// This shouldn't happen if the notifier is active, but a safe check.
				logger.Warnf("[GO] ⚠️ Instance %d: DisconnectedF: Host is nil, cannot perform connection check.\n", instanceIndex)
				return
			}

			// --- Check if this is the LAST connection to this peer ---
			// libp2p can have multiple connections to a single peer (e.g., TCP, QUIC).
			// We only want to consider the peer fully disconnected when ALL connections are gone.
			if len(instanceHost.Network().ConnsToPeer(remotePeerID)) == 0 {
				logger.Debugf("[GO]   Instance %d: Last connection to %s closed. Removing from tracked peers.\n", instanceIndex, remotePeerID)

				// Handle disconnection for ConnectedPeers
				instanceConnectedPeersMutex := &connectedPeersMutexes[instanceIndex]
				instanceConnectedPeersMutex.Lock()
				if _, exists := connectedPeersInstances[instanceIndex][remotePeerID]; exists {
					delete(connectedPeersInstances[instanceIndex], remotePeerID)
					logger.Debugf("[GO]   Instance %d: Removed %s from ConnectedPeers via DisconnectedF notifier.\n", instanceIndex, remotePeerID)
					//peerRemoved = true
				}
				instanceConnectedPeersMutex.Unlock()

				// Also clean up persistent stream if one existed for this peer
				persistentChatStreamsMutexes[instanceIndex].Lock()
				if stream, ok := persistentChatStreamsInstances[instanceIndex][remotePeerID]; ok {
					logger.Debugf("[GO]   Instance %d: Cleaning up persistent stream for disconnected peer %s via DisconnectedF notifier.\n", instanceIndex, remotePeerID)
					_ = stream.Close() // Attempt graceful close
					delete(persistentChatStreamsInstances[instanceIndex], remotePeerID)
				}
				persistentChatStreamsMutexes[instanceIndex].Unlock()
			} else {
				logger.Debugf("[GO]   Instance %d: DisconnectedF: Still have %d active connections to %s, not removing from tracked peers.\n", instanceIndex, len(instanceHost.Network().ConnsToPeer(remotePeerID)), remotePeerID)
			}
		},
	})
}

// waitForPublicReachability subscribes to the host's event bus and waits for the
// node to confirm its public reachability. It includes a timeout to prevent
// the startup from hanging indefinitely.
func waitForPublicReachability(h host.Host, timeout time.Duration) bool {
	// 1. Subscribe to the reachability event.
	sub, err := h.EventBus().Subscribe(new(event.EvtLocalReachabilityChanged))
	if err != nil {
		logger.Errorf("[GO] ❌ Failed to subscribe to reachability events: %v", err)
		return false
	}
	defer sub.Close() // Clean up the subscription when we're done.

	logger.Debugf("[GO] ⏳ Waiting for public reachability confirmation (timeout: %s)...", timeout)

	// 2. Wait for the event in a select loop with a timeout.
	timeoutCh := time.After(timeout)
	for {
		select {
		case evt := <-sub.Out():
			// We received an event. Cast it to the correct type.
			reachabilityEvent, ok := evt.(event.EvtLocalReachabilityChanged)
			if !ok {
				continue // Should not happen, but good practice to check.
			}

			logger.Infof("[GO] 💡 Reachability status changed to: %s", reachabilityEvent.Reachability)

			// Check if the new status is what we're waiting for.
			if reachabilityEvent.Reachability == network.ReachabilityPublic {
				logger.Debugf("[GO] ✅ Confirmed Public reachability via event.")
				return true // Success! Return true.
			} else if reachabilityEvent.Reachability == network.ReachabilityPrivate {
				logger.Warnf("[GO] ⚠️ Node is behind a NAT or firewall (Private reachability).")
				return false // Node is not publicly reachable.
			}
		case <-timeoutCh:
			logger.Warnf("[GO] ⚠️ Timed out waiting for public reachability.")
			return false // Timeout. Return false.
		}
	}
}

// --- Core Logic Functions ---

// storeReceivedMessage processes a raw message received either from a direct stream
// or a PubSub topic. The sender peerID and the channel to store are retrieved in handleStream and readFromSubscription
func storeReceivedMessage(
	instanceIndex int,
	from peer.ID,
	channel string,
	data []byte,
) {
	// Check instance index validity
	if err := checkInstanceIndex(instanceIndex); err != nil {
		logger.Errorf("[GO] ❌ storeReceivedMessage: %v\n", err)
		return // Cannot process message for invalid instance
	}

	// Get the message store for this instance
	store := messageStoreInstances[instanceIndex]
	if store == nil {
		logger.Errorf("[GO] ❌ storeReceivedMessage: Message store not initialized for instance %d\n", instanceIndex)
		return // Cannot process message if store is nil
	}

	// Create the minimal message envelope.
	newMessage := &QueuedMessage{
		From: from,
		Data: data,
	}

	// Lock the store mutex before accessing the shared maps.
	store.mu.Lock()
	defer store.mu.Unlock()

	// Check if this channel already has a message list.
	messageList, channelExists := store.messagesByChannel[channel]
	if !channelExists {
		// If the channel does not exist, check if we can create a new message queue.
		if len(store.messagesByChannel) >= maxUniqueChannels {
			logger.Warnf("[GO] 🗑️ Instance %d: Message store full. Discarding message for new channel '%s'.\n", instanceIndex, channel)
			return
		}
		messageList = list.New()
		store.messagesByChannel[channel] = messageList
		logger.Debugf("[GO] ✨ Instance %d: Created new channel queue '%s'. Total channels: %d\n", instanceIndex, channel, len(store.messagesByChannel))
	}

	// If the channel already has a message list, check its length.
	if messageList.Len() >= maxChannelQueueLen {
		logger.Warnf("[GO] 🗑️ Instance %d: Queue for channel '%s' full. Discarding message.\n", instanceIndex, channel)
		return
	}

	messageList.PushBack(newMessage)
	logger.Debugf("[GO] 📥 Instance %d: Queued message on channel '%s' from %s. New queue length: %d\n", instanceIndex, channel, from, messageList.Len())
}

// readFromSubscription runs as a dedicated goroutine for each active PubSub subscription for a specific instance.
// It continuously waits for new messages on the subscription's channel (`sub.Next(ctx)`),
// routes them to `storeReceivedMessage`, and handles errors and context cancellation gracefully.
// You need to provide the full Channel to uniquely identify the subscription.
func readFromSubscription(
	instanceIndex int,
	sub *pubsub.Subscription,
) {

	// Check instance index validity (should be done before launching goroutine, but defensive check)
	if err := checkInstanceIndex(instanceIndex); err != nil {
		logger.Errorf("[GO] ❌ readFromSubscription: %v. Exiting goroutine.\n", err)
		return
	}

	// Get the topic string directly from the subscription object.
	topic := sub.Topic()
	instanceCtx := contexts[instanceIndex]
	instanceHost := hostInstances[instanceIndex]

	if instanceCtx == nil || instanceHost == nil {
		logger.Errorf("[GO] ❌ readFromSubscription: Context or Host not initialized for instance %d. Exiting goroutine.\n", instanceIndex)
		return
	}

	logger.Infof("[GO] 👂 Instance %d: Started listener goroutine for topic: %s\n", instanceIndex, topic)
	defer logger.Infof("[GO] 👂 Instance %d: Exiting listener goroutine for topic: %s\n", instanceIndex, topic) // Log when goroutine exits

	for {
		// Check if the main context has been cancelled (e.g., during node shutdown).
		if instanceCtx.Err() != nil {
			logger.Debugf("[GO] 👂 Instance %d: Context cancelled, stopping listener goroutine for topic: %s\n", instanceIndex, topic)
			return // Exit the goroutine.
		}

		// Wait for the next message from the subscription. This blocks until a message
		// arrives, the context is cancelled, or an error occurs.
		msg, err := sub.Next(instanceCtx)
		if err != nil {
			// Check for expected errors during shutdown or cancellation.
			if err == context.Canceled || err == context.DeadlineExceeded || err == pubsub.ErrSubscriptionCancelled || instanceCtx.Err() != nil {
				logger.Debugf("[GO] 👂 Instance %d: Subscription listener for topic '%s' stopping gracefully: %v\n", instanceIndex, topic, err)
				return // Exit goroutine cleanly.
			}
			// Handle EOF, which can sometimes occur. Treat it as a reason to stop.
			if err == io.EOF {
				logger.Debugf("[GO] 👂 Instance %d: Subscription listener for topic '%s' encountered EOF, stopping: %v\n", instanceIndex, topic, err)
				return // Exit goroutine.
			}
			// Log other errors but attempt to continue (they might be transient).
			logger.Errorf("[GO] ❌ Instance %d: Error reading from subscription '%s': %v. Continuing...\n", instanceIndex, topic, err)
			// Pause briefly to avoid busy-looping on persistent errors.
			time.Sleep(1 * time.Second)
			continue // Continue the loop to try reading again.
		}

		logger.Infof("[GO] 📬 Instance %d (id: %s): Received new PubSub message on topic '%s' from %s\n", instanceIndex, instanceHost.ID().String(), topic, msg.GetFrom())

		// Ignore messages published by the local node itself.
		if msg.GetFrom() == instanceHost.ID() {
			continue // Skip processing self-sent messages.
		}

		// Handle Rendezvous or Standard Messages
		if strings.HasSuffix(topic, ":rv") {
			// This is a rendezvous update.
			// 1. First, unmarshal the outer Protobuf message.
			var protoMsg pg.Message
			if err := proto.Unmarshal(msg.Data, &protoMsg); err != nil {
				logger.Warnf("⚠️ Instance %d: Could not decode Protobuf message on topic '%s': %v\n", instanceIndex, topic, err)
				continue
			}

			// 2. The actual payload is a JSON string within the 'json_content' field.
			jsonPayload := protoMsg.GetJsonContent()
			if jsonPayload == "" {
				logger.Warnf("⚠️ Instance %d: Rendezvous message on topic '%s' has empty JSON content.\n", instanceIndex, topic)
				continue
			}

			// 3. Now, unmarshal the inner JSON payload.
			var updatePayload struct {
				Peers       []ExtendedPeerInfo `json:"peers"`
				UpdateCount int64              `json:"update_count"`
			}
			if err := json.Unmarshal([]byte(jsonPayload), &updatePayload); err != nil {
				logger.Warnf("[GO] ⚠️ Instance %d: Could not decode rendezvous update payload on topic '%s': %v\n", instanceIndex, topic, err)
				continue // Skip this malformed message.
			}

			// 4. Create a new map from the decoded peer list.
			newPeerMap := make(map[peer.ID]ExtendedPeerInfo)
			for _, peerInfo := range updatePayload.Peers {
				newPeerMap[peerInfo.ID] = peerInfo
			}

			// 5. Safely replace the old map with the new one.
			rendezvousDiscoveredPeersMutexes[instanceIndex].Lock()
			// If this is the first update for this instance, initialize the state struct.
			if rendezvousDiscoveredPeersInstances[instanceIndex] == nil {
				rendezvousDiscoveredPeersInstances[instanceIndex] = &RendezvousState{}
			}
			rendezvousState := rendezvousDiscoveredPeersInstances[instanceIndex]
			rendezvousState.Peers = newPeerMap
			rendezvousState.UpdateCount = updatePayload.UpdateCount
			rendezvousDiscoveredPeersMutexes[instanceIndex].Unlock()

			logger.Debugf("[GO] ✅ Instance %d: Updated rendezvous peers from topic '%s'. Found %d peers. Update count: %d.\n", instanceIndex, topic, len(newPeerMap), updatePayload.UpdateCount)
		} else {
			// This is a standard message. Queue it as before.
			logger.Debugf("[GO] 📝 Instance %d: Storing new pubsub message from topic '%s'.\n", instanceIndex, topic)
			storeReceivedMessage(instanceIndex, msg.GetFrom(), topic, msg.Data)
		}
	}
}

// handleStream reads from a direct message stream using the new framing protocol.
// It expects the stream to start with a 4-byte length prefix, followed by a 1-byte channel name length,
// the channel name itself, and finally the Protobuf-encoded payload.
func handleStream(instanceIndex int, s network.Stream) {
	senderPeerID := s.Conn().RemotePeer()
	logger.Debugf("[GO] 📥 Instance %d: Accepted new INCOMING stream from %s, storing for duplex communication.\n", instanceIndex, senderPeerID)

	// This defer block ensures cleanup happens when the stream is closed by either side.
	defer func() {
		logger.Debugf("[GO] 🧹 Instance %d: Inbound stream from %s closed. Removing from persistent map.\n", instanceIndex, senderPeerID)
		persistentChatStreamsMutexes[instanceIndex].Lock()
		delete(persistentChatStreamsInstances[instanceIndex], senderPeerID)
		persistentChatStreamsMutexes[instanceIndex].Unlock()
		s.Close() // Ensure the stream is fully closed.
	}()

	// Store the newly accepted stream so we can use it to send messages back to this peer.
	persistentChatStreamsMutexes[instanceIndex].Lock()
	persistentChatStreamsInstances[instanceIndex][senderPeerID] = s
	persistentChatStreamsMutexes[instanceIndex].Unlock()

	for {
		// --- REFACTORED: New Framing Protocol ---
		// 1. Read the 4-byte total length prefix.
		var totalLen uint32
		if err := binary.Read(s, binary.BigEndian, &totalLen); err != nil {
			if err == io.EOF {
				logger.Debugf("[GO] 🔌 Instance %d: Direct stream with peer %s closed (EOF).\n", instanceIndex, senderPeerID)
			} else if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
				logger.Warnf("[GO] ⏳ Instance %d: Timeout reading length from direct stream with %s: %v\n", instanceIndex, senderPeerID, err)
			} else {
				logger.Errorf("[GO] ❌ Instance %d: Unexpected error reading length from direct stream with %s: %v\n", instanceIndex, senderPeerID, err)
			}
			return // Exit handler for any read error on length.
		}

		// --- Check the message size ---
		if totalLen > MaxMessageSize {
			logger.Errorf("[GO] ❌ Instance %d: Received message length %d exceeds limit (%d) from %s. Resetting stream.\n", instanceIndex, totalLen, MaxMessageSize, senderPeerID)
			s.Reset() // Forcefully close the stream due to protocol violation.
			return
		}
		if totalLen == 0 {
			logger.Warnf("[GO] ⚠️ Instance %d: Received zero length message frame from %s, continuing loop.\n", instanceIndex, senderPeerID)
			continue
		}

		// 2. Read the 1-byte channel name length.
		var channelLen uint8
		if err := binary.Read(s, binary.BigEndian, &channelLen); err != nil {
			if err == io.EOF {
				logger.Debugf("[GO] 🔌 Instance %d: Direct stream with peer %s closed (EOF).\n", instanceIndex, senderPeerID)
			} else if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
				logger.Warnf("[GO] ⏳ Instance %d: Timeout reading channel-length from direct stream with %s: %v\n", instanceIndex, senderPeerID, err)
			} else {
				logger.Errorf("[GO] ❌ Instance %d: Unexpected error reading channel-length from direct stream with %s: %v\n", instanceIndex, senderPeerID, err)
			}
			return // Exit handler for any read error on length.
		}

		// 3. Read the channel name string.
		channelBytes := make([]byte, channelLen)
		if _, err := io.ReadFull(s, channelBytes); err != nil {
			if err == io.EOF {
				logger.Debugf("[GO] 🔌 Instance %d: Direct stream with peer %s closed (EOF).\n", instanceIndex, senderPeerID)
			} else if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
				logger.Warnf("[GO] ⏳ Instance %d: Timeout reading channel from direct stream with %s: %v\n", instanceIndex, senderPeerID, err)
			} else {
				logger.Errorf("[GO] ❌ Instance %d: Unexpected error reading channel from direct stream with %s: %v\n", instanceIndex, senderPeerID, err)
			}
			return // Exit handler for any read error on length.
		}
		channel := string(channelBytes)

		// 4. Read the Protobuf payload.
		payloadLen := totalLen - uint32(channelLen) - 1 // Subtract channel len byte and channel string
		payload := make([]byte, payloadLen)
		if _, err := io.ReadFull(s, payload); err != nil {
			if err == io.EOF {
				logger.Debugf("[GO] 🔌 Instance %d: Direct stream with peer %s closed (EOF).\n", instanceIndex, senderPeerID)
			} else if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
				logger.Warnf("[GO] ⏳ Instance %d: Timeout reading payload from direct stream with %s: %v\n", instanceIndex, senderPeerID, err)
			} else {
				logger.Errorf("[GO] ❌ Instance %d: Unexpected error reading payload from direct stream with %s: %v\n", instanceIndex, senderPeerID, err)
			}
			return // Exit handler for any read error on length.
		}

		// 5. Store the message.
		logger.Infof("[GO] 📨 Instance %d: Received direct message on channel '%s' from %s, storing.\n", instanceIndex, channel, senderPeerID)
		storeReceivedMessage(instanceIndex, senderPeerID, channel, payload)
	}
}

// setupDirectMessageHandler configures the libp2p host for a specific instance
// to listen for incoming streams using the custom ChatProtocol.
// When a peer opens a stream with this protocol ID, the provided handler function
// is invoked to manage communication on that stream.
func setupDirectMessageHandler(
	instanceIndex int,
) {

	// Check instance index validity
	if err := checkInstanceIndex(instanceIndex); err != nil {
		logger.Errorf("[GO] ❌ setupDirectMessageHandler: %v\n", err)
		return // Cannot setup handler for invalid instance
	}

	instanceHost := hostInstances[instanceIndex]

	if instanceHost == nil {
		logger.Errorf("[GO] ❌ Instance %d: Cannot setup direct message handler: Host not initialized\n", instanceIndex)
		return
	}

	// Set a handler function for the UnaiverseChatProtocol. This function will be called
	// automatically by libp2p whenever a new incoming stream for this protocol is accepted.
	// Use a closure to capture the instanceIndex.
	instanceHost.SetStreamHandler(UnaiverseChatProtocol, func(s network.Stream) {
		handleStream(instanceIndex, s)
	})
}

// This function constructs and writes a message using our new framing protocol for direct messages.
// It takes a writer (e.g., a network stream), the channel name, and the payload data.
// The message format is:
// - 4-byte total length (including all the following parts)
// - 1-byte channel name length
// - channel name (as a UTF-8 string)
// - payload (Protobuf-encoded data).
func writeDirectMessageFrame(w io.Writer, channel string, payload []byte) error {
	channelBytes := []byte(channel)
	channelLen := uint8(len(channelBytes))

	// Check if channel name is too long for our 1-byte length prefix.
	if len(channelBytes) > 255 {
		return fmt.Errorf("channel name exceeds 255 bytes limit: %s", channel)
	}

	// Total length = 1 (for channel len) + len(channel) + len(payload)
	totalLength := uint32(1 + len(channelBytes) + len(payload))

	// --- Add size check before writing ---
	if totalLength > MaxMessageSize {
		return fmt.Errorf("outgoing message size (%d) exceeds limit (%d)", totalLength, MaxMessageSize)
	}

	buf := new(bytes.Buffer)

	// Write total length (4 bytes)
	if err := binary.Write(buf, binary.BigEndian, totalLength); err != nil {
		return fmt.Errorf("failed to write total length: %w", err)
	}
	// Write channel length (1 byte)
	if err := binary.Write(buf, binary.BigEndian, channelLen); err != nil {
		return fmt.Errorf("failed to write channel length: %w", err)
	}
	// Write channel name
	if _, err := buf.Write(channelBytes); err != nil {
		return fmt.Errorf("failed to write channel name: %w", err)
	}
	// Write payload
	if _, err := buf.Write(payload); err != nil {
		return fmt.Errorf("failed to write payload: %w", err)
	}

	// Write the entire frame to the stream.
	if _, err := w.Write(buf.Bytes()); err != nil {
		return fmt.Errorf("failed to write framed message to stream: %w", err)
	}
	return nil
}

// goGetNodeAddresses is the internal Go function that performs the core logic
// of fetching and formatting node addresses.
// It takes an instanceIndex and a targetPID. If targetPID is empty (peer.ID("")),
// it fetches addresses for the local node of the given instance.
// It returns a slice of fully formatted multiaddress strings and an error if one occurs.
func goGetNodeAddresses(
	instanceIndex int,
	targetPID peer.ID, // Changed from targetPeerIDStr string
) ([]string, error) {
	instanceHost := hostInstances[instanceIndex]
	if instanceHost == nil {
		errMsg := fmt.Sprintf("Instance %d: Host not initialized", instanceIndex)
		logger.Errorf("[GO] ❌ goGetNodeAddresses: %s\n", errMsg)
		return nil, fmt.Errorf("%s", errMsg)
	}

	// Determine the actual Peer ID to resolve addresses for.
	resolvedPID := targetPID
	isThisNode := false
	if targetPID == "" || targetPID == instanceHost.ID() {
		resolvedPID = instanceHost.ID()
		isThisNode = true
	}

	// Use a map to automatically handle duplicate addresses.
	addrSet := make(map[string]struct{})

	// --- 1. Gather all candidate addresses from the host and peerstore ---
	var candidateAddrs []ma.Multiaddr
	candidateAddrs = append(candidateAddrs, instanceHost.Peerstore().Addrs(resolvedPID)...)
	if isThisNode {
		if interfaceAddrs, err := instanceHost.Network().InterfaceListenAddresses(); err == nil {
			candidateAddrs = append(candidateAddrs, interfaceAddrs...)
		}
		candidateAddrs = append(candidateAddrs, instanceHost.Network().ListenAddresses()...)
		candidateAddrs = append(candidateAddrs, instanceHost.Addrs()...)
	} else {
		// --- Remote Peer Addresses ---
		connectedPeersMutexes[instanceIndex].RLock()
		instanceConnectedPeers := connectedPeersInstances[instanceIndex]
		if epi, exists := instanceConnectedPeers[resolvedPID]; exists { // Use resolvedPID
			candidateAddrs = append(candidateAddrs, epi.Addrs...)
		} else {
		}
		connectedPeersMutexes[instanceIndex].RUnlock()
	}

	// --- 2. Process and filter candidate addresses ---
	for _, addr := range candidateAddrs {
		if addr == nil || manet.IsIPLoopback(addr) || manet.IsIPUnspecified(addr) {
			continue // Skip nil, loopback, and unspecified addresses
		}

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
		switch {
		case idInAddr == resolvedPID:
			// Case A: The address is already perfect and has the correct Peer ID. Use it as is.
			finalAddr = addr

		case idInAddr == "":
			// Case B: The address is missing a Peer ID. This is common for addresses from the
			// peerstore and for relayed addresses like `/p2p/RELAY_ID/p2p-circuit`. We must append ours.
			p2pComponent, _ := ma.NewMultiaddr(fmt.Sprintf("/p2p/%s", resolvedPID.String()))
			finalAddr = addr.Encapsulate(p2pComponent)

		case idInAddr != resolvedPID:
			// Case C: The address has the WRONG Peer ID. This is stale or incorrect data. Discard it.
			logger.Warnf("[GO] ⚠️ Instance %d: Discarding stale address for peer %s: %s\n", instanceIndex, resolvedPID, addr)
			continue
		}
		addrSet[finalAddr.String()] = struct{}{}
	}

	// --- 4. Convert the final set of unique addresses to a slice for returning. ---
	result := make([]string, 0, len(addrSet))
	for addr := range addrSet {
		result = append(result, addr)
	}

	if len(result) == 0 {
		logger.Warnf("[GO] ⚠️ goGetNodeAddresses: No suitable addresses found for peer %s.", resolvedPID)
	}

	return result, nil
}

// closeSingleInstance performs the cleanup for a specific node instance.
// It is called by CloseNode for either a single index or as part of the "close all" loop.
// It returns a *C.char JSON string indicating success or failure for that specific instance.
// This function assumes instanceIndex is already validated by the caller (CloseNode).
func closeSingleInstance(
	instanceIndex int,
) *C.char {

	// Acquire global lock briefly to check if instance exists before proceeding
	instanceStateMutex.RLock()
	hostExists := hostInstances[instanceIndex] != nil
	cancelExists := cancelContexts[instanceIndex] != nil
	isInstInitialized := isInitialized[instanceIndex] // Check initialization flag
	instanceStateMutex.RUnlock()

	if !isInstInitialized {
		// Should not happen if called from CloseNode after checking, but defensive
		logger.Debugf("[GO] ℹ️ Instance %d: Node was not initialized (internal close call).\n", instanceIndex)
		return jsonSuccessResponse(fmt.Sprintf("Instance %d: Node was not initialized", instanceIndex))
	}

	if !hostExists && !cancelExists {
		logger.Debugf("[GO] ℹ️ Instance %d: Node was already closed (internal close call).\n", instanceIndex)
		return jsonSuccessResponse(fmt.Sprintf("Instance %d: Node was already closed", instanceIndex))
	}

	// --- Stop Cert Manager FIRST ---
	if certManagerInstances[instanceIndex] != nil {
		logger.Debugf("[GO]   - Instance %d: Stopping AutoTLS cert manager...\n", instanceIndex)
		certManagerInstances[instanceIndex].Stop()
		certManagerInstances[instanceIndex] = nil
	}

	// --- Cancel Main Context ---
	// Acquire global lock to safely access/modify cancelContexts
	instanceStateMutex.Lock()
	if cancelContexts[instanceIndex] != nil {
		logger.Debugf("[GO]   - Instance %d: Cancelling main context...\n", instanceIndex)
		cancelContexts[instanceIndex]()
		// Do NOT set to nil here yet, wait until host is closed
	} else {
		logger.Debugf("[GO]   - Instance %d: Context was already nil.\n", instanceIndex)
	}
	instanceStateMutex.Unlock() // Release global lock

	// Give goroutines time to react to context cancellation (e.g., stream handlers, pubsub listeners)
	time.Sleep(200 * time.Millisecond)

	// --- Close Persistent Outgoing Streams ---
	// Acquire instance-specific mutex
	persistentChatStreamsMutexes[instanceIndex].Lock()
	instancePersistentChatStreams := persistentChatStreamsInstances[instanceIndex]
	if len(instancePersistentChatStreams) > 0 {
		logger.Debugf("[GO]   - Instance %d: Closing %d persistent outgoing streams...\n", instanceIndex, len(instancePersistentChatStreams))
		for pid, stream := range instancePersistentChatStreams {
			logger.Debugf("[GO]     - Instance %d: Closing stream to %s\n", instanceIndex, pid)
			_ = stream.Close() // Attempt graceful close
		}
		persistentChatStreamsInstances[instanceIndex] = make(map[peer.ID]network.Stream) // Clear the map
	} else {
		logger.Debugf("[GO]   - Instance %d: No persistent outgoing streams to close.\n", instanceIndex)
	}
	persistentChatStreamsMutexes[instanceIndex].Unlock() // Release instance-specific mutex

	// --- Clean Up PubSub State ---
	// Acquire instance-specific mutex
	pubsubMutexes[instanceIndex].Lock()
	instanceSubscriptions := subscriptionsInstances[instanceIndex]

	if len(instanceSubscriptions) > 0 {
		logger.Debugf("[GO]   - Instance %d: Ensuring PubSub subscriptions (%d) are cancelled...\n", instanceIndex, len(instanceSubscriptions))
		for channel, sub := range instanceSubscriptions {
			logger.Debugf("[GO]     - Instance %d: Cancelling subscription to topic: %s\n", instanceIndex, channel)
			sub.Cancel()
		}
	}
	subscriptionsInstances[instanceIndex] = make(map[string]*pubsub.Subscription) // Clear the map
	topicsInstances[instanceIndex] = make(map[string]*pubsub.Topic)               // Clear the map
	pubsubMutexes[instanceIndex].Unlock()                                         // Release instance-specific mutex
	pubsubInstances[instanceIndex] = nil                                          // Set instance PubSub to nil

	// --- Close Host Instance ---
	hostErrStr := ""
	// Acquire global lock to safely access/modify hostInstances and cancelContexts
	instanceStateMutex.Lock()
	if hostInstances[instanceIndex] != nil {
		logger.Debugf("[GO]   - Instance %d: Closing host instance...\n", instanceIndex)
		err := hostInstances[instanceIndex].Close()
		hostInstances[instanceIndex] = nil // Set instance host to nil
		// Now that host is closed, it's safe to set cancel context to nil
		cancelContexts[instanceIndex] = nil
		if err != nil {
			hostErrStr = fmt.Sprintf("Instance %d: Error closing host: %v", instanceIndex, err)
			logger.Warnf("[GO] ⚠️ %s (proceeding with cleanup)\n", hostErrStr)
			// Continue cleanup even if host close fails
		} else {
			logger.Debugf("[GO]   - Instance %d: Host closed successfully.\n", instanceIndex)
		}
	} else {
		logger.Debugf("[GO]   - Instance %d: Host instance was already nil.\n", instanceIndex)
		// If host was nil, ensure cancel context is also nil
		cancelContexts[instanceIndex] = nil
	}
	instanceStateMutex.Unlock() // Release global lock

	// --- Clear Remaining State for this instance ---
	// Acquire instance-specific mutex
	connectedPeersMutexes[instanceIndex].Lock()
	connectedPeersInstances[instanceIndex] = make(map[peer.ID]ExtendedPeerInfo) // Clear the map
	connectedPeersMutexes[instanceIndex].Unlock()                               // Release instance-specific mutex

	// Clear the MessageStore for this instance
	if messageStoreInstances[instanceIndex] != nil {
		messageStoreInstances[instanceIndex].mu.Lock()
		messageStoreInstances[instanceIndex].messagesByChannel = make(map[string]*list.List) // Clear the message store
		messageStoreInstances[instanceIndex].mu.Unlock()
		messageStoreInstances[instanceIndex] = nil // Set instance store to nil
	}
	logger.Debugf("[GO]   - Instance %d: Cleared connected peers map and message buffer.\n", instanceIndex)

	// Clear the rendezvous state for this instance
	rendezvousDiscoveredPeersMutexes[instanceIndex].Lock()
	rendezvousDiscoveredPeersInstances[instanceIndex] = nil  // Clear the map
	rendezvousDiscoveredPeersMutexes[instanceIndex].Unlock() // Release instance-specific mutex

	// Mark instance as uninitialized
	instanceStateMutex.Lock()
	isInitialized[instanceIndex] = false
	instanceStateMutex.Unlock()

	// --- Return Result ---
	if hostErrStr != "" {
		// Return error if host closing failed, but mention cleanup attempt
		return jsonErrorResponse(
			fmt.Sprintf("Instance %d: Cleanup attempted, but encountered error during host closure", instanceIndex),
			fmt.Errorf("%s", hostErrStr),
		)
	}

	logger.Infof("[GO] ✅ Instance %d: Node closed successfully.\n", instanceIndex)
	return jsonSuccessResponse(fmt.Sprintf("Instance %d: Node closed successfully", instanceIndex))
}

// --- Exported C Functions ---
// These functions are callable from C (and thus Python). They act as the API boundary.

// This function MUST be called once from Python before any other library function.
//
//export InitializeLibrary
func InitializeLibrary(
	maxInstancesC C.int,
	maxUniqueChannelsC C.int,
	maxChannelQueueLenC C.int,
	maxMessageSizeC C.int,
	enableLoggingC C.int,
) {
	// --- Configure Logging FIRST ---
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
	if int(enableLoggingC) == 1 {
		golog.SetAllLoggers(golog.LevelInfo) // Set a default level
		golog.SetLogLevel("p2p-library", "info")
		// --- Add Specific Log Levels from the Example ---
		// These are crucial for debugging AutoTLS and connectivity.
		golog.SetLogLevel("autotls", "info")
		golog.SetLogLevel("p2p-forge", "info")
		golog.SetLogLevel("nat", "info")
		golog.SetLogLevel("basichost", "info")
		golog.SetLogLevel("p2p-circuit", "info") // Core circuit-v2 protocol logic
		golog.SetLogLevel("relay", "info")
	} else {
		golog.SetAllLoggers(golog.LevelError)
		golog.SetLogLevel("*", "FATAL")
	}

	maxInstances = int(maxInstancesC)
	maxUniqueChannels = int(maxUniqueChannelsC)
	maxChannelQueueLen = int(maxChannelQueueLenC)
	MaxMessageSize = uint32(maxMessageSizeC)

	// Now, initialize all the state slices with the correct size
	hostInstances = make([]host.Host, maxInstances)
	pubsubInstances = make([]*pubsub.PubSub, maxInstances)
	contexts = make([]context.Context, maxInstances)
	cancelContexts = make([]context.CancelFunc, maxInstances)
	topicsInstances = make([]map[string]*pubsub.Topic, maxInstances)
	subscriptionsInstances = make([]map[string]*pubsub.Subscription, maxInstances)
	connectedPeersInstances = make([]map[peer.ID]ExtendedPeerInfo, maxInstances)
	rendezvousDiscoveredPeersInstances = make([]*RendezvousState, maxInstances)
	persistentChatStreamsInstances = make([]map[peer.ID]network.Stream, maxInstances)
	messageStoreInstances = make([]*MessageStore, maxInstances)
	certManagerInstances = make([]*p2pforge.P2PForgeCertMgr, maxInstances)

	// Mutexes for protecting concurrent access to instance-specific data.
	connectedPeersMutexes = make([]sync.RWMutex, maxInstances)
	persistentChatStreamsMutexes = make([]sync.Mutex, maxInstances)
	pubsubMutexes = make([]sync.RWMutex, maxInstances)
	rendezvousDiscoveredPeersMutexes = make([]sync.RWMutex, maxInstances)

	// Flag to track if a specific instance index has been initialized
	isInitialized = make([]bool, maxInstances)
	logger.Infof("[GO] ✅ Go library initialized with MaxInstances=%d, MaxUniqueChannels=%d and MaxChannelQueueLen=%d\n", maxInstances, maxUniqueChannels, maxChannelQueueLen)
}

// CreateNode initializes and starts a new libp2p host (node) for a specific instance.
// It configures the node based on the provided parameters (port, relay capabilities, UPnP).
// Parameters:
//   - instanceIndexC (C.int): The index for this node instance (0 to maxInstances-1).
//   - predefinedPortC (C.int): The TCP port to listen on (0 for random).
//   - enableRelayClientC (C.int): 1 if this node should enable relay communications (client mode)
//   - enableRelayServiceC (C.int): 1 to set this node as a relay service (server mode),
//   - knowsIsPublicC (C.int): 1 to assume public reachability, 0 otherwise (-> tries to assess it in any possible way).
//   - maxConnectionsC (C.int): The maximum number of connections this node can maintain.
//
// Returns:
//   - *C.char: A JSON string indicating success (with node addresses) or failure (with an error message).
//     The structure is `{"state":"Success", "message": ["/ip4/.../p2p/...", ...]}` or `{"state":"Error", "message":"..."}`.
//   - IMPORTANT: The caller (C/Python) MUST free the returned C string using the `FreeString` function
//     exported by this library to avoid memory leaks. Returns NULL only on catastrophic failure before JSON creation.
//
//export CreateNode
func CreateNode(
	instanceIndexC C.int,
	identityDirC *C.char,
	predefinedPortC C.int,
	ipsJSONC *C.char,
	enableRelayClientC C.int,
	enableRelayServiceC C.int,
	knowsIsPublicC C.int,
	maxConnectionsC C.int,
	enableTLSC C.int,
	domainNameC *C.char,
	tlsCertPathC *C.char,
	tlsKeyPathC *C.char,
) *C.char {

	instanceIndex := int(instanceIndexC)

	// --- Basic Instance Index Check ---
	if err := checkInstanceIndex(instanceIndex); err != nil {
		return jsonErrorResponse("Invalid instance index", err)
	}

	// --- Instance-Specific State Initialization (if not already initialized) ---
	instanceStateMutex.Lock()
	if isInitialized[instanceIndex] {
		instanceStateMutex.Unlock()
		msg := fmt.Sprintf("Instance %d is already initialized. Please call CloseNode first.", instanceIndex)
		return jsonErrorResponse(msg, nil)
	}
	isInitialized[instanceIndex] = true
	instanceStateMutex.Unlock()
	logger.Infof("[GO] 🚀 Instance %d: Starting CreateNode...", instanceIndex)

	// Initialize state maps and context for this instance
	contexts[instanceIndex], cancelContexts[instanceIndex] = context.WithCancel(context.Background())
	connectedPeersInstances[instanceIndex] = make(map[peer.ID]ExtendedPeerInfo)
	persistentChatStreamsInstances[instanceIndex] = make(map[peer.ID]network.Stream)
	topicsInstances[instanceIndex] = make(map[string]*pubsub.Topic)
	subscriptionsInstances[instanceIndex] = make(map[string]*pubsub.Subscription)
	messageStoreInstances[instanceIndex] = newMessageStore()
	rendezvousDiscoveredPeersInstances[instanceIndex] = nil

	// --- Configuration ---
	// Convert C integer parameters to Go types.
	identityDir := C.GoString(identityDirC)
	predefinedPort := int(predefinedPortC)
	ipsJSON := C.GoString(ipsJSONC)
	enableRelayClient := int(enableRelayClientC) == 1
	enableRelayService := int(enableRelayServiceC) == 1
	knowsIsPublic := int(knowsIsPublicC) == 1
	maxConnections := int(maxConnectionsC)
	enableTLS := int(enableTLSC) == 1
	domainName := C.GoString(domainNameC)
	tlsCertPath := C.GoString(tlsCertPathC)
	tlsKeyPath := C.GoString(tlsKeyPathC)
	// We already checked on the python side that the following three are all provided (if they are needed)
	useAutoTLS := enableTLS && tlsCertPath == ""
	useCustomTLS := enableTLS && tlsCertPath != ""

	// --- Load or Create Persistent Identity ---
	keyPath := filepath.Join(identityDir, "identity.key")
	privKey, err := loadOrCreateIdentity(keyPath)
	if err != nil {
		cleanupFailedCreate(instanceIndex)
		return jsonErrorResponse(fmt.Sprintf("Instance %d: Failed to prepare identity", instanceIndex), err)
	}

	// --- AutoTLS Cert Manager Setup (if enabled) ---
	var certManager *p2pforge.P2PForgeCertMgr
	if useAutoTLS {
		logger.Debugf("[GO]   - Instance %d: AutoTLS is ENABLED. Setting up certificate manager...\n", instanceIndex)
		certManager, err = p2pforge.NewP2PForgeCertMgr(
			p2pforge.WithCAEndpoint(p2pforge.DefaultCAEndpoint),
			p2pforge.WithCertificateStorage(&certmagic.FileStorage{Path: filepath.Join(identityDir, "p2p-forge-certs")}),
			p2pforge.WithUserAgent(UnaiverseUserAgent),
			p2pforge.WithRegistrationDelay(10*time.Second),
		)
		if err != nil {
			cleanupFailedCreate(instanceIndex)
			return jsonErrorResponse(fmt.Sprintf("Instance %d: Failed to create AutoTLS cert manager", instanceIndex), err)
		}
		certManager.Start()
		certManagerInstances[instanceIndex] = certManager // Store for cleanup
	}

	// --- 4. Libp2p Options Assembly ---
	tlsMode := "none"
	if useCustomTLS {
		tlsMode = "domain"
	} else if useAutoTLS {
		tlsMode = "autotls"
	}
	listenAddrs, err := getListenAddrs(ipsJSON, predefinedPort, tlsMode)
	if err != nil {
		cleanupFailedCreate(instanceIndex)
		return jsonErrorResponse(fmt.Sprintf("Instance %d: Failed to create multiaddrs", instanceIndex), err)
	}

	// Setup Resource Manager
	limiter, err := createResourceManager(maxConnections)
	if err != nil {
		cleanupFailedCreate(instanceIndex)
		return jsonErrorResponse(fmt.Sprintf("Instance %d: Failed to create resource manager", instanceIndex), err)
	}

	options := []libp2p.Option{
		libp2p.Identity(privKey),
		libp2p.ListenAddrs(listenAddrs...),
		libp2p.DefaultSecurity,
		libp2p.DefaultMuxers,
		libp2p.Transport(tcp.NewTCPTransport),
		libp2p.ShareTCPListener(),
		libp2p.Transport(quic.NewTransport),
		libp2p.Transport(webtransport.New),
		libp2p.Transport(webrtc.New),
		libp2p.ResourceManager(limiter),
		libp2p.UserAgent(UnaiverseUserAgent),
	}

	// Add WebSocket transport, with or without TLS based on cert availability
	if useCustomTLS {
		// We already have certificates, use them
		logger.Debugf("[GO]   - Instance %d: Certificates provided, setting up secure WebSocket (WSS).\n", instanceIndex)
		cert, err := tls.LoadX509KeyPair(tlsCertPath, tlsKeyPath)
		if err != nil {
			cleanupFailedCreate(instanceIndex)
			return jsonErrorResponse(fmt.Sprintf("Instance %d: Failed to load Custom TLS certificate and key", instanceIndex), err)
		}
		tlsConfig := &tls.Config{Certificates: []tls.Certificate{cert}}
		// let's also create a custom address factory to ensure we always advertise the correct domain name
		domainAddressFactory := func(addrs []ma.Multiaddr) []ma.Multiaddr {
			// Replace the IP part of the WSS address with our domain.
			result := make([]ma.Multiaddr, 0, len(addrs))
			for _, addr := range addrs {
				if strings.Contains(addr.String(), "/tls/ws") || strings.Contains(addr.String(), "/wss") {
					// This is our WSS listener. Create the public /dns4 version.
					dnsAddr, _ := ma.NewMultiaddr(fmt.Sprintf("/dns4/%s/tcp/%d/tls/ws", domainName, predefinedPort))
					result = append(result, dnsAddr)
				} else {
					// Keep other addresses (like QUIC) as they are.
					result = append(result, addr)
				}
			}
			return result
		}
		options = append(options,
			libp2p.Transport(ws.New, ws.WithTLSConfig(tlsConfig)),
			libp2p.AddrsFactory(domainAddressFactory),
		)
		logger.Debugf("[GO]   - Instance %d: Loaded custom TLS certificate and key for WSS.\n", instanceIndex)
	} else if useAutoTLS {
		// No certificates, create them automatically
		options = append(options,
			libp2p.Transport(ws.New, ws.WithTLSConfig(certManager.TLSConfig())),
			libp2p.AddrsFactory(certManager.AddressFactory()),
		)
	} else {
		// No certificates, use plain WS
		logger.Debugf("[GO]   - Instance %d: No certificates found, setting up non-secure WebSocket.\n", instanceIndex)
		options = append(options, libp2p.Transport(ws.New))
	}

	// Configure Relay Service (ability to *be* a relay)
	if enableRelayService {
		// limit := rc.DefaultLimit()         // open this to see the default limits
		resources := rc.DefaultResources() // open this to see the default resource limits
		// Set the duration for relayed connections. 0 means infinite.
		ttl := 2 * time.Hour // reduced to 2 hours, it will be the node's duty to refresh the reservation if needed.
		// limit.Duration = ttl
		// resources.Limit = limit
		resources.Limit = nil // same as setting rc.WithInfiniteLimits()
		resources.ReservationTTL = ttl

		// This single option enables the node to act as a relay for others, including hopping,
		// with our custom resource limits.
		options = append(options, libp2p.EnableRelayService(rc.WithResources(resources)), libp2p.EnableNATService())
		logger.Debugf("[GO]   - Instance %d: Relay service is ENABLED with custom resource configuration.\n", instanceIndex)
	}

	// EnableRelay (the ability to *use* relays) is default, we can explicitly disable it if needed.
	if !enableRelayClient {
		options = append(options, libp2p.DisableRelay()) // Explicitly disable using relays.
		logger.Debugf("[GO]   - Instance %d: Relay client is DISABLED.\n", instanceIndex)
	}

	// Prepare discovering the bootstrap peers
	var idht *dht.IpfsDHT
	isPublic := false
	if !knowsIsPublic || enableTLS {
		// Add any possible option to be publicly reachable
		options = append(
			options,
			libp2p.NATPortMap(),
			libp2p.EnableHolePunching(),
			libp2p.EnableAutoNATv2(),
			libp2p.Routing(func(h host.Host) (routing.PeerRouting, error) {
				bootstrapAddrInfos := dht.GetDefaultBootstrapPeerAddrInfos()
				// Define the DHT options for a "lazy" client
				dhtOptions := []dht.Option{
					dht.Mode(dht.ModeClient),
					dht.BootstrapPeers(bootstrapAddrInfos...),
				}
				var err error
				idht, err = dht.New(contexts[instanceIndex], h, dhtOptions...)
				return idht, err
			}))
		logger.Debugf("[GO]   - Instance %d: Trying to be publicly reachable.\n", instanceIndex)
	} else {
		options = append(options, libp2p.ForceReachabilityPublic())
		isPublic = true
	}

	// Create the libp2p Host instance with the configured options for this instance.
	instanceHost, err := libp2p.New(options...)
	if err != nil {
		cleanupFailedCreate(instanceIndex)
		return jsonErrorResponse(fmt.Sprintf("Instance %d: Failed to create host", instanceIndex), err)
	}
	hostInstances[instanceIndex] = instanceHost
	logger.Infof("[GO] ✅ Instance %d: Host created with ID: %s\n", instanceIndex, instanceHost.ID())

	// --- Link Host to Cert Manager ---
	if useAutoTLS {
		certManager.ProvideHost(instanceHost)
		logger.Debugf("[GO]   - Instance %d: Provided host to AutoTLS cert manager.\n", instanceIndex)
	}

	// --- PubSub Initialization ---
	if err := setupPubSub(instanceIndex); err != nil {
		cleanupFailedCreate(instanceIndex)
		return jsonErrorResponse(fmt.Sprintf("Instance %d: Failed to create PubSub", instanceIndex), err)
	}
	logger.Debugf("[GO] ✅ Instance %d: PubSub (GossipSub) initialized.\n", instanceIndex)

	// --- Setup Notifiers and Handlers ---
	setupNotifiers(instanceIndex)
	logger.Debugf("[GO] 🔔 Instance %d: Registered network event notifier.\n", instanceIndex)

	setupDirectMessageHandler(instanceIndex)
	logger.Debugf("[GO] ✅ Instance %d: Direct message handler set up.\n", instanceIndex)

	// --- Address Reporting ---
	// Give discovery mechanisms a moment to find the public address.
	logger.Debugf("[GO] ⏳ Instance %d: Waiting for address discovery and NAT to settle...\n", instanceIndex)

	if enableTLS {
		// --- Wait for the EvtLocalAddressesUpdated containing the resolved WSS address ---
		logger.Debugf("[GO] ⏳ Instance %d: Waiting for the final public WSS address to be generated...", instanceIndex)

		// 1. Subscribe to the event that fires *after* the AddrsFactory has run.
		sub, err := instanceHost.EventBus().Subscribe(new(event.EvtLocalAddressesUpdated))
		if err != nil {
			cleanupFailedCreate(instanceIndex)
			return jsonErrorResponse("Failed to subscribe to address update events", err)
		}
		defer sub.Close()

		// 2. Loop and inspect events until our condition is met or we time out.
	WAIT_FOR_WSS_ADDR:
		for {
			select {
			case evt := <-sub.Out():
				// We received an address update event.
				logger.Debugf("[GO] 🔔 Instance %d: Received address update event, checking for WSS address...\n", instanceIndex)
				addrsEvent, ok := evt.(event.EvtLocalAddressesUpdated)
				if !ok {
					continue // Should not happen, but good practice.
				}

				// 3. Check the addresses in the event to see if our WSS address is ready.
				for _, updatedAddr := range addrsEvent.Current {
					addr := updatedAddr.Address
					addrStr := addr.String()
					logger.Debugf("[GO]     - Instance %d: Found address: %s\n", instanceIndex, addrStr)

					// The condition for readiness: the address is public AND is a secure websocket address.
					isPublicWSS := manet.IsPublicAddr(addr) && (strings.Contains(addrStr, "/tls/") && !strings.Contains(addrStr, "*"))

					if isPublicWSS {
						logger.Infof("[GO] ✅ Instance %d: Confirmed final public WSS address: %s", instanceIndex, addrStr)
						isPublic = true         // If we have a public WSS address, the node is public.
						break WAIT_FOR_WSS_ADDR // Exit the loop, we are done.
					}
				}
				// If we checked all addresses in this event and didn't find the WSS one, the loop continues, waiting for the next event.

			case <-time.After(30 * time.Second):
				cleanupFailedCreate(instanceIndex)
				return jsonErrorResponse("Timed out after 90s waiting for the final WSS address", nil)

			case <-contexts[instanceIndex].Done():
				cleanupFailedCreate(instanceIndex)
				return jsonErrorResponse("Node creation cancelled while waiting for final WSS address", nil)
			}
		}
		idht.Close() // DHT job is done.

	} else if !knowsIsPublic {
		// --- Fallback to old reachability check ONLY if AutoTLS is disabled ---
		isPublic = waitForPublicReachability(instanceHost, 30*time.Second)
		if !isPublic {
			logger.Warnf("[GO] ⚠️ Instance %d: The node may not be directly dialable.", instanceIndex)
		}
		idht.Close()
	}

	// --- Cleanup Bootstrap Peers ---
	// In case thhe connectedPeers map was flooded with bootstrap peers during DHT setup, let's do this anyway.
	logger.Debugf("[GO] 🧹 Instance %d: Cleaning up bootstrap peer connections from the tracked list...\n", instanceIndex)
	connectedPeersMutexes[instanceIndex].Lock()
	connectedPeersInstances[instanceIndex] = make(map[peer.ID]ExtendedPeerInfo) // Clear the map
	connectedPeersMutexes[instanceIndex].Unlock()

	// --- Get Final Addresses ---
	nodeAddresses, err := goGetNodeAddresses(instanceIndex, "")
	if err != nil {
		// This is a more critical failure if we can't even get local addresses.
		cleanupFailedCreate(instanceIndex)
		return jsonErrorResponse(
			fmt.Sprintf("Instance %d: Failed to obtain node addresses after waiting for reachability", instanceIndex),
			err,
		)
	}

	// --- Build and return the new structured response ---
	response := CreateNodeResponse{
		Addresses: nodeAddresses,
		IsPublic:  isPublic,
	}

	logger.Infof("[GO] 🌐 Instance %d: Node addresses: %v\n", instanceIndex, nodeAddresses)
	logger.Infof("[GO] 🎉 Instance %d: Node creation complete.\n", instanceIndex)
	return jsonSuccessResponse(response)
}

// ConnectTo attempts to establish a connection with a remote peer given its multiaddress for a specific instance.
// Parameters:
//   - instanceIndexC (C.int): The index of the node instance.
//   - addrsJSONC (*C.char): Pointer to a JSON string containing the list of addresses that can be dialed.
//
// Returns:
//   - *C.char: A JSON string indicating success (with peer AddrInfo of the winning connection) or failure (with an error message).
//     Structure: `{"state":"Success", "message": {"ID": "...", "Addrs": ["...", ...]}}` or `{"state":"Error", "message":"..."}`.
//   - IMPORTANT: The caller MUST free the returned C string using `FreeString`.
//
//export ConnectTo
func ConnectTo(
	instanceIndexC C.int,
	addrsJSONC *C.char,
) *C.char {

	instanceIndex := int(instanceIndexC)
	goAddrsJSON := C.GoString(addrsJSONC)
	logger.Debugf("[GO] 📞 Instance %d: Attempting to connect to peer with addresses: %s\n", instanceIndex, goAddrsJSON)

	// Check instance index validity
	if err := checkInstanceIndex(instanceIndex); err != nil {
		return jsonErrorResponse("Invalid instance index", err) // Caller frees.
	}

	// Get instance-specific state
	instanceHost := hostInstances[instanceIndex]
	instanceCtx := contexts[instanceIndex]

	// Check if the host is initialized for this instance.
	if instanceHost == nil {
		return jsonErrorResponse(
			fmt.Sprintf("Instance %d: Host not initialized, cannot connect", instanceIndex),
			nil,
		) // Caller frees.
	}
	if instanceCtx == nil {
		return jsonErrorResponse(
			fmt.Sprintf("Instance %d: Context not initialized, cannot connect", instanceIndex),
			nil,
		) // Caller frees.
	}

	// --- Unmarshal Address List from JSON ---
	var addrStrings []string
	if err := json.Unmarshal([]byte(goAddrsJSON), &addrStrings); err != nil {
		return jsonErrorResponse("Failed to parse addresses JSON", err)
	}
	if len(addrStrings) == 0 {
		return jsonErrorResponse("Address list is empty", nil)
	}

	// --- Create AddrInfo from the list ---
	addrInfo, err := peer.AddrInfoFromString(addrStrings[0])
	if err != nil {
		return jsonErrorResponse("Invalid first multiaddress in list", err)
	}

	// Add the rest of the addresses to the AddrInfo struct
	for i := 1; i < len(addrStrings); i++ {
		maddr, err := ma.NewMultiaddr(addrStrings[i])
		if err != nil {
			logger.Warnf("[GO] ⚠️ Instance %d: Skipping invalid multiaddress '%s' in list: %v\n", instanceIndex, addrStrings[i], err)
			continue
		}
		// You might want to add a check here to ensure subsequent addresses are for the same peer ID
		addrInfo.Addrs = append(addrInfo.Addrs, maddr)
	}

	// Check if attempting to connect to the local node itself.
	if addrInfo.ID == instanceHost.ID() {
		logger.Debugf("[GO] ℹ️ Instance %d: Attempting to connect to self (%s), skipping explicit connection.\n", instanceIndex, addrInfo.ID)
		// Connecting to self is usually not necessary or meaningful in libp2p.
		// Return success, indicating the "connection" is implicitly present.
		return jsonSuccessResponse(addrInfo) // Caller frees.
	}

	// --- 1. ESTABLISH CONNECTION ---
	// Use a context with a timeout for the connection attempt to prevent blocking indefinitely.
	connCtx, cancel := context.WithTimeout(instanceCtx, 30*time.Second) // 30-second timeout.
	defer cancel()                                                      // Ensure context is cancelled eventually.

	// Add the peer's address(es) to the local peerstore for this instance. This helps libp2p find the peer.
	// ConnectedAddrTTL suggests the address is likely valid for a short time after connection.
	// Use PermanentAddrTTL if the address is known to be stable.
	instanceHost.Peerstore().AddAddrs(addrInfo.ID, addrInfo.Addrs, peerstore.ConnectedAddrTTL)

	// Initiate the connection attempt. libp2p will handle dialing and negotiation.
	logger.Debugf("[GO]   - Instance %d: Attempting host.Connect to %s...\n", instanceIndex, addrInfo.ID)
	if err := instanceHost.Connect(connCtx, *addrInfo); err != nil {
		// Check if the error was due to the connection timeout.
		if connCtx.Err() == context.DeadlineExceeded {
			errMsg := fmt.Sprintf("Instance %d: Connection attempt to %s timed out after 30s", instanceIndex, addrInfo.ID)
			logger.Errorf("[GO] ❌ %s\n", errMsg)
			return jsonErrorResponse(errMsg, nil) // Return specific timeout error (caller frees).
		}
		// Handle other connection errors.
		errMsg := fmt.Sprintf("Instance %d: Failed to connect to peer %s", instanceIndex, addrInfo.ID)
		// Example: Check for specific common errors if needed
		// if strings.Contains(err.Error(), "no route to host") { ... }
		return jsonErrorResponse(errMsg, err) // Return generic connection error (caller frees).
	}

	// --- 2. FIND THE WINNING ADDRESS ---
	// After a successful connection, query the host's network for active connections to the peer.
	// This is where you find the 'winning' address.
	conns := instanceHost.Network().ConnsToPeer(addrInfo.ID)
	var winningAddr string
	if len(conns) > 0 {
		winningAddr = fmt.Sprintf("%s/p2p/%s", conns[0].RemoteMultiaddr().String(), addrInfo.ID.String())
		logger.Debugf("[GO] ✅ Instance %d: Successfully connected to peer %s via: %s\n", instanceIndex, addrInfo.ID, winningAddr)
	} else {
		logger.Warnf("[GO] ⚠️ Instance %d: Connect succeeded for %s, but no active connection found immediately. It may be pending.\n", instanceIndex, addrInfo.ID)
	}

	// Success: log the successful connection and return the response.
	logger.Infof("[GO] ✅ Instance %d: Successfully initiated connection to multiaddress: %s\n", instanceIndex, winningAddr)
	winningAddrInfo, err := peer.AddrInfoFromString(winningAddr)
	if err != nil {
		return jsonErrorResponse("Invalid winner multiaddress.", err)
	}
	return jsonSuccessResponse(winningAddrInfo) // Caller frees.
}

// ReserveOnRelay attempts to reserve a slot on a specified relay node for a specific instance.
// This allows the local node to be reachable via that relay, even if behind NAT/firewall.
// The first connection with the relay node should be done in advance using ConnectTo.
// Parameters:
//   - instanceIndexC (C.int): The index of the node instance.
//   - relayPeerIDC (*C.char): The peerID of the relay node.
//
// Returns:
//   - *C.char: A JSON string indicating success or failure.
//     On success, the `message` contains the expiration date of the reservation (ISO 8601).
//     Structure (Success): `{"state":"Success", "message": "2024-12-31T23:59:59Z"}`
//     Structure (Error): `{"state":"Error", "message":"..."}`
//   - IMPORTANT: The caller MUST free the returned C string using `FreeString`.
//
//export ReserveOnRelay
func ReserveOnRelay(
	instanceIndexC C.int,
	relayPeerIDC *C.char,
) *C.char {

	instanceIndex := int(instanceIndexC)
	// Convert C string input to Go string.
	goRelayPeerID := C.GoString(relayPeerIDC)
	logger.Debugf("[GO] 🅿️ Instance %d: Attempting to reserve slot on relay with Peer ID: %s\n", instanceIndex, goRelayPeerID)

	// Check instance index validity
	if err := checkInstanceIndex(instanceIndex); err != nil {
		return jsonErrorResponse("Invalid instance index", err) // Caller frees.
	}

	// Get instance-specific state
	instanceHost := hostInstances[instanceIndex]
	instanceCtx := contexts[instanceIndex]

	// Check if the host is initialized for this instance.
	if instanceHost == nil {
		return jsonErrorResponse(
			fmt.Sprintf("Instance %d: Host not initialized, cannot reserve", instanceIndex), nil,
		) // Caller frees.
	}
	if instanceCtx == nil {
		return jsonErrorResponse(
			fmt.Sprintf("Instance %d: Context not initialized, cannot reserve", instanceIndex), nil,
		) // Caller frees.
	}

	// --- Decode Peer ID and build AddrInfo from Peerstore ---
	relayPID, err := peer.Decode(goRelayPeerID)
	if err != nil {
		return jsonErrorResponse("Failed to decode relay Peer ID string", err)
	}

	// Construct the AddrInfo using the ID and the addresses we know from the peerstore.
	relayInfo := peer.AddrInfo{
		ID:    relayPID,
		Addrs: instanceHost.Peerstore().Addrs(relayPID),
	}

	// Ensure the node is not trying to reserve a slot on itself.
	if relayInfo.ID == instanceHost.ID() {
		return jsonErrorResponse(
			fmt.Sprintf("Instance %d: Cannot reserve slot on self", instanceIndex), nil,
		) // Caller frees.
	}

	// --- VERIFY CONNECTION TO RELAY ---
	if len(instanceHost.Network().ConnsToPeer(relayInfo.ID)) == 0 {
		errMsg := fmt.Sprintf("Instance %d: Not connected to relay %s. Must connect before reserving.", instanceIndex, relayInfo.ID)
		return jsonErrorResponse(errMsg, nil)
	}
	logger.Debugf("[GO]   - Instance %d: Verified connection to relay: %s\n", instanceIndex, relayInfo.ID)

	// --- Attempt Reservation ---
	// Use a separate context with potentially longer timeout for the reservation itself.
	resCtx, resCancel := context.WithTimeout(instanceCtx, 60*time.Second) // 60-second timeout for reservation.
	defer resCancel()
	// Call the circuitv2 client function to request a reservation.
	// This performs the RPC communication with the relay.
	reservation, err := client.Reserve(resCtx, instanceHost, relayInfo)
	if err != nil {
		errMsg := fmt.Sprintf("Instance %d: Failed to reserve slot on relay %s", instanceIndex, relayInfo.ID)
		// Handle reservation timeout specifically.
		if resCtx.Err() == context.DeadlineExceeded {
			errMsg = fmt.Sprintf("Instance %d: Reservation attempt on relay %s timed out", instanceIndex, relayInfo.ID)
			return jsonErrorResponse(errMsg, nil) // Caller frees.
		}
		return jsonErrorResponse(errMsg, err) // Caller frees.
	}

	// Although Reserve usually errors out if it fails, double-check if the reservation object is nil.
	if reservation == nil {
		errMsg := fmt.Sprintf("Instance %d: Reservation on relay %s returned nil voucher, but no error", instanceIndex, relayInfo.ID)
		return jsonErrorResponse(errMsg, nil) // Caller frees.
	}

	// --- Construct Relayed Addresses and Update Local Peerstore ---
	// We construct a relayed address for each public address of the relay to maximize reachability.
	var constructedAddrs []ma.Multiaddr
	for _, relayAddr := range relayInfo.Addrs {
		// We only want to use public, usable addresses for the circuit
		if manet.IsIPLoopback(relayAddr) || manet.IsIPUnspecified(relayAddr) {
			continue
		}

		// Ensure the relay's address in the peerstore includes its own Peer ID
		baseRelayAddrStr := relayAddr.String()
		if _, idInAddr := peer.SplitAddr(relayAddr); idInAddr == "" {
			baseRelayAddrStr = fmt.Sprintf("%s/p2p/%s", relayAddr.String(), relayInfo.ID.String())
		}

		constructedAddrStr := fmt.Sprintf("%s/p2p-circuit/p2p/%s", baseRelayAddrStr, instanceHost.ID().String())
		constructedAddr, err := ma.NewMultiaddr(constructedAddrStr)
		if err == nil {
			constructedAddrs = append(constructedAddrs, constructedAddr)
		}
	}

	if len(constructedAddrs) == 0 {
		return jsonErrorResponse("Reservation succeeded but failed to construct any valid relayed multiaddr", nil)
	}

	logger.Debugf("[GO]   - Instance %d: Adding %d constructed relayed address(es) to local peerstore (ID: %s) expiring at: %s\n", instanceIndex, len(constructedAddrs), instanceHost.ID(), reservation.Expiration.Format(time.RFC3339))
	instanceHost.Peerstore().AddAddrs(instanceHost.ID(), constructedAddrs, peerstore.PermanentAddrTTL)

	logger.Infof("[GO] ✅ Instance %d: Reservation successful on relay: %s.\n", instanceIndex, relayInfo.ID)

	// Return the expiration time of the reservation as confirmation.
	return jsonSuccessResponse(reservation.Expiration)
}

// DisconnectFrom attempts to close any active connections to a specified peer
// and removes the peer from the internally tracked list for a specific instance.
// Parameters:
//   - instanceIndexC (C.int): The index of the node instance.
//   - peerIDC (*C.char): The Peer ID string of the peer to disconnect from.
//
// Returns:
//   - *C.char: A JSON string indicating success or failure.
//     Structure: `{"state":"Success", "message":"Disconnected from peer ..."}` or `{"state":"Error", "message":"..."}`.
//   - IMPORTANT: The caller MUST free the returned C string using `FreeString`.
//
//export DisconnectFrom
func DisconnectFrom(
	instanceIndexC C.int,
	peerIDC *C.char,
) *C.char {

	instanceIndex := int(instanceIndexC)
	goPeerID := C.GoString(peerIDC)
	logger.Debugf("[GO] 🔌 Instance %d: Attempting to disconnect from peer: %s\n", instanceIndex, goPeerID)

	// Check instance index validity
	if err := checkInstanceIndex(instanceIndex); err != nil {
		return jsonErrorResponse("Invalid instance index", err)
	}

	// Get instance-specific state
	instanceHost := hostInstances[instanceIndex]
	instanceConnectedPeers := connectedPeersInstances[instanceIndex]
	instanceConnectedPeersMutex := &connectedPeersMutexes[instanceIndex]
	instancePersistentChatStreams := persistentChatStreamsInstances[instanceIndex]
	instancePersistentChatStreamsMutex := &persistentChatStreamsMutexes[instanceIndex]

	if instanceHost == nil {
		return jsonErrorResponse(
			fmt.Sprintf("Instance %d: Host not initialized, cannot disconnect", instanceIndex), nil,
		)
	}

	pid, err := peer.Decode(goPeerID)
	if err != nil {
		return jsonErrorResponse(
			fmt.Sprintf("Instance %d: Failed to decode peer ID", instanceIndex), err,
		)
	}

	if pid == instanceHost.ID() {
		logger.Debugf("[GO] ℹ️ Instance %d: Attempting to disconnect from self (%s), skipping.\n", instanceIndex, pid)
		return jsonSuccessResponse("Cannot disconnect from self")
	}

	// --- Close Persistent Outgoing Stream (if exists) for this instance ---
	instancePersistentChatStreamsMutex.Lock()
	stream, exists := instancePersistentChatStreams[pid]
	if exists {
		logger.Debugf("[GO]   ↳ Instance %d: Closing persistent outgoing stream to %s\n", instanceIndex, pid)
		_ = stream.Close() // Attempt graceful close
		delete(instancePersistentChatStreams, pid)
	}
	instancePersistentChatStreamsMutex.Unlock() // Unlock before potentially blocking network call

	// --- Close Network Connections ---
	conns := instanceHost.Network().ConnsToPeer(pid)
	closedNetworkConn := false
	if len(conns) > 0 {
		logger.Debugf("[GO]   - Instance %d: Closing %d active network connection(s) to peer %s...\n", instanceIndex, len(conns), pid)
		err = instanceHost.Network().ClosePeer(pid) // This closes the underlying connection(s)
		if err != nil {
			logger.Warnf("[GO] ⚠️ Instance %d: Error closing network connection(s) to peer %s: %v (proceeding with cleanup)\n", instanceIndex, pid, err)
		} else {
			logger.Debugf("[GO]   - Instance %d: Closed network connection(s) to peer: %s\n", instanceIndex, pid)
			closedNetworkConn = true
		}
	} else {
		logger.Debugf("[GO] ℹ️ Instance %d: No active network connections found to peer %s.\n", instanceIndex, pid)
	}

	// --- Remove from Tracking Map for this instance ---
	instanceConnectedPeersMutex.Lock()
	delete(instanceConnectedPeers, pid)
	instanceConnectedPeersMutex.Unlock()

	logMsg := fmt.Sprintf("Instance %d: Disconnected from peer %s", instanceIndex, goPeerID)
	if !exists && !closedNetworkConn && len(conns) == 0 {
		logMsg = fmt.Sprintf("Instance %d: Disconnected from peer %s (not connected or tracked)", instanceIndex, goPeerID)
	}
	logger.Infof("[GO] 🔌 %s\n", logMsg)

	return jsonSuccessResponse(logMsg)
}

// GetConnectedPeers returns a list of peers currently tracked as connected for a specific instance.
// Note: This relies on the internal `connectedPeersInstances` map which is updated during
// connect/disconnect operations and incoming streams. It may optionally perform
// a liveness check.
// Parameters:
//   - instanceIndexC (C.int): The index of the node instance.
//
// Returns:
//   - *C.char: A JSON string containing a list of connected peers' information.
//     Structure: `{"state":"Success", "message": [ExtendedPeerInfo, ...]}` or `{"state":"Error", "message":"..."}`.
//     Each `ExtendedPeerInfo` object has `addr_info` (ID, Addrs), `connected_at`, `direction`, and `misc`.
//   - IMPORTANT: The caller MUST free the returned C string using `FreeString`.
//
//export GetConnectedPeers
func GetConnectedPeers(
	instanceIndexC C.int,
) *C.char {

	instanceIndex := int(instanceIndexC)

	// Check instance index validity
	if err := checkInstanceIndex(instanceIndex); err != nil {
		return jsonErrorResponse("Invalid instance index", err) // Caller frees.
	}

	// Get instance-specific state and mutex
	instanceConnectedPeers := connectedPeersInstances[instanceIndex]
	instanceConnectedPeersMutex := &connectedPeersMutexes[instanceIndex]
	instanceHost := hostInstances[instanceIndex]

	// Use a Write Lock for the entire critical section to avoid mixing RLock and Lock.
	instanceConnectedPeersMutex.RLock()
	defer instanceConnectedPeersMutex.RUnlock() // Ensure lock is released.

	// Create a slice to hold the results directly from the map.
	peersList := make([]ExtendedPeerInfo, 0, len(instanceConnectedPeers))
	// Prior check if the host is initialized for this instance.
	if instanceHost != nil && instanceHost.Network() != nil {
		// Check if the connectedPeers map itself is initialized for this instance.
		// This map should be initialized in CreateNode.
		if instanceConnectedPeers == nil {
			logger.Warnf("[GO] ⚠️ Instance %d: GetConnectedPeers: connectedPeersInstances map is nil. Returning empty list.\n", instanceIndex)
			// Return success with an empty list.
			return jsonSuccessResponse([]ExtendedPeerInfo{})
		}

		for _, peerInfo := range instanceConnectedPeers {
			peersList = append(peersList, peerInfo)
		}
	} else {
		// If host is not ready, return the current state of the map (which should be empty if CreateNode was called correctly).
		logger.Warnf("[GO] ⚠️ Instance %d: GetConnectedPeers called but host is not fully initialized. Returning potentially empty list based on map.\n", instanceIndex)
		for _, peerInfo := range instanceConnectedPeers {
			peersList = append(peersList, peerInfo)
		}
	}

	logger.Debugf("[GO] ℹ️ Instance %d: Reporting %d currently tracked and active peers.\n", instanceIndex, len(peersList))

	// Return the list of active peers as a JSON success response.
	return jsonSuccessResponse(peersList) // Caller frees.
}

// GetRendezvousPeers returns a list of peers currently tracked as part of the world for a specific instance.
// Note: This relies on the internal `rendezvousDiscoveredPeersInstances` map which is updated by pubsub
// Parameters:
//   - instanceIndexC (C.int): The index of the node instance.
//
// Returns:
//   - *C.char: A JSON string containing a list of connected peers' information.
//     Structure: `{"state":"Success", "message": [ExtendedPeerInfo, ...]}` or `{"state":"Error", "message":"..."}`.
//     Each `ExtendedPeerInfo` object has `addr_info` (ID, Addrs), `connected_at`, `direction`, and `misc`.
//   - IMPORTANT: The caller MUST free the returned C string using `FreeString`.
//
//export GetRendezvousPeers
func GetRendezvousPeers(
	instanceIndexC C.int,
) *C.char {

	instanceIndex := int(instanceIndexC)

	// Check instance index validity
	if err := checkInstanceIndex(instanceIndex); err != nil {
		return jsonErrorResponse("Invalid instance index", err) // Caller frees.
	}

	rendezvousDiscoveredPeersMutexes[instanceIndex].RLock()
	// Get the pointer to the state struct.
	rendezvousState := rendezvousDiscoveredPeersInstances[instanceIndex]
	rendezvousDiscoveredPeersMutexes[instanceIndex].RUnlock()

	// If the state pointer is nil, it means we haven't received the first update yet.
	if rendezvousState == nil {
		return C.CString(`{"state":"Empty"}`)
	}

	// Extract the list of extendedPeerInfo to return it
	peersList := make([]ExtendedPeerInfo, 0, len(rendezvousState.Peers))
	for _, peerInfo := range rendezvousState.Peers {
		peersList = append(peersList, peerInfo)
	}

	// This struct will be marshaled to JSON with exactly the fields you want.
	responsePayload := struct {
		Peers       []ExtendedPeerInfo `json:"peers"`
		UpdateCount int64              `json:"update_count"`
	}{
		Peers:       peersList,
		UpdateCount: rendezvousState.UpdateCount,
	}

	// The state exists, so return the whole struct.
	logger.Debugf("[GO] ℹ️ Instance %d: Reporting %d rendezvous peers (UpdateCount: %d).\n", instanceIndex, len(rendezvousState.Peers), rendezvousState.UpdateCount)
	return jsonSuccessResponse(responsePayload) // Caller frees.
}

// GetNodeAddresses is the C-exported wrapper for goGetNodeAddresses.
// It handles C-Go type conversions and JSON marshaling.
//
//export GetNodeAddresses
func GetNodeAddresses(
	instanceIndexC C.int,
	peerIDC *C.char,
) *C.char {
	instanceIndex := int(instanceIndexC)
	peerIDStr := C.GoString(peerIDC) // Raw string from C

	if err := checkInstanceIndex(instanceIndex); err != nil {
		return jsonErrorResponse("Invalid instance index", err)
	}

	// instanceHost is needed here to compare against for "local" case,
	// or if goGetNodeAddresses itself didn't handle nil host for some reason.
	instanceHost := hostInstances[instanceIndex]
	if instanceHost == nil {
		// This check should ideally also be inside goGetNodeAddresses if it can be called
		// before host is fully up, but for the wrapper it's good.
		return jsonErrorResponse(fmt.Sprintf("Instance %d: Host not initialized", instanceIndex), nil)
	}

	var pidForInternalCall peer.ID // This will be peer.ID("") for local
	var err error

	if peerIDStr == "" || peerIDStr == instanceHost.ID().String() {
		// Convention: Empty peer.ID ("") passed to goGetNodeAddresses means "local node".
		pidForInternalCall = "" // This is peer.ID("")
	} else {
		pidForInternalCall, err = peer.Decode(peerIDStr)
		if err != nil {
			errMsg := fmt.Sprintf("Instance %d: Failed to decode peer ID '%s'", instanceIndex, peerIDStr)
			return jsonErrorResponse(errMsg, err)
		}
	}

	// Call the internal Go function with the resolved peer.ID or empty peer.ID for local
	addresses, err := goGetNodeAddresses(instanceIndex, pidForInternalCall)
	if err != nil {
		return jsonErrorResponse(err.Error(), nil)
	}

	return jsonSuccessResponse(addresses)
}

// SendMessageToPeer sends a message either directly to a specific peer or broadcasts it via PubSub for a specific instance.
// Parameters:
//   - instanceIndexC (C.int): The index of the node instance.
//   - channelC (*C.char): Use the unique channel as defined above in the Message struct.
//   - dataC (*C.char): A pointer to the raw byte data of the message payload.
//   - lengthC (C.int): The length of the data buffer pointed to by `data`.
//
// Returns:
//   - *C.char: A JSON string with {"state": "Success/Error", "message": "..."}.
//   - IMPORTANT: The caller MUST free this string using FreeString.
//
//export SendMessageToPeer
func SendMessageToPeer(
	instanceIndexC C.int,
	channelC *C.char,
	dataC *C.char,
	lengthC C.int,
) *C.char {

	instanceIndex := int(instanceIndexC)
	// Convert C inputs
	goChannel := C.GoString(channelC)
	goData := C.GoBytes(unsafe.Pointer(dataC), C.int(lengthC))

	if err := checkInstanceIndex(instanceIndex); err != nil {
		// Invalid instance index, return error code.
		return jsonErrorResponse("Invalid instance index", err)
	}

	// Get instance-specific state and mutexes
	instanceHost := hostInstances[instanceIndex]
	instanceCtx := contexts[instanceIndex]

	if instanceHost == nil || instanceCtx == nil {
		// Host or context not initialized for this instance.
		return jsonErrorResponse("Host or Context not initialized for this instance", nil)
	}

	// --- Branch: Broadcast or Direct Send ---
	if strings.Contains(goChannel, "::ps:") {
		// --- Broadcast via specific PubSub Topic ---
		instancePubsub := pubsubInstances[instanceIndex] // Get from instance state
		instanceCtx := contexts[instanceIndex]           // Get from instance state

		if instancePubsub == nil {
			// PubSub not initialized, cannot broadcast
			return jsonErrorResponse("PubSub not initialized, cannot broadcast", nil)
		}

		pubsubMutexes[instanceIndex].Lock() // Changed to full Lock since we might Join
		topic, exists := topicsInstances[instanceIndex][goChannel]
		if !exists {
			var err error
			logger.Debugf("[GO]   - Instance %d: Joining PubSub topic '%s' for sending.\n", instanceIndex, goChannel)
			topic, err = instancePubsub.Join(goChannel) // ps is instancePubsub
			if err != nil {
				pubsubMutexes[instanceIndex].Unlock()
				// Failed to join PubSub topic
				return jsonErrorResponse(fmt.Sprintf("Failed to join PubSub topic '%s'", goChannel), err)
			}
			topicsInstances[instanceIndex][goChannel] = topic // Store the new topic
			logger.Debugf("[GO] ✅ Instance %d: Joined PubSub topic: %s for publishing.\n", instanceIndex, goChannel)
		}
		pubsubMutexes[instanceIndex].Unlock() // Unlock after potentially joining

		// Directly publish the raw Protobuf payload.
		if err := topic.Publish(instanceCtx, goData); err != nil {
			// Failed to publish to topic
			return jsonErrorResponse(fmt.Sprintf("Failed to publish to topic '%s'", goChannel), err)
		}
		logger.Infof("[GO] 🌍 Instance %d: Broadcast to topic '%s' (%d bytes)\n", instanceIndex, goChannel, len(goData))
		return jsonSuccessResponse(fmt.Sprintf("Message broadcast to topic %s", goChannel))

	} else if strings.Contains(goChannel, "::dm:") {
		// --- Direct Peer-to-Peer Message Sending (Persistent Stream Logic) ---
		receiverChannelIDStr := strings.Split(goChannel, "::dm:")[1] // Extract the receiver's channel ID from the format "dm:<peerID>-<channelSpecifier>"
		peerIDStr := strings.Split(receiverChannelIDStr, "-")[0]
		pid, err := peer.Decode(peerIDStr)
		if err != nil {
			// Invalid peer ID format
			return jsonErrorResponse("Invalid peer ID format in channel string", err)
		}

		if pid == instanceHost.ID() {
			// Attempt to send direct message to self
			return jsonErrorResponse("Attempt to send direct message to self is invalid", nil)
		}

		instancePersistentChatStreams := persistentChatStreamsInstances[instanceIndex]
		instancePersistentChatStreamsMutex := &persistentChatStreamsMutexes[instanceIndex]

		// Acquire lock to access the persistent stream map for this instance
		instancePersistentChatStreamsMutex.Lock()
		stream, exists := instancePersistentChatStreams[pid]

		// If stream exists, try writing to it
		if exists {
			logger.Debugf("[GO]   ↳ Instance %d: Reusing existing stream to %s\n", instanceIndex, pid)
			err = writeDirectMessageFrame(stream, goChannel, goData)
			if err == nil {
				// Success writing to existing stream
				instancePersistentChatStreamsMutex.Unlock() // Unlock before returning
				logger.Infof("[GO] 📤 Instance %d: Sent direct message to %s (on existing stream)\n", instanceIndex, pid)
				return jsonSuccessResponse(fmt.Sprintf("Direct message sent to %s (reused stream).", pid))
			}
			// Write failed on existing stream - assume it's broken
			logger.Warnf("[GO] ⚠️ Instance %d: Failed to write to existing stream for %s: %v. Closing and removing stream.", instanceIndex, pid, err)
			// Close the stream (Reset is more abrupt, Close attempts graceful)
			_ = stream.Close() // Ignore error during close, as we're removing it anyway
			// Remove from map
			delete(instancePersistentChatStreams, pid)
			// Unlock and return specific error
			instancePersistentChatStreamsMutex.Unlock()
			return jsonErrorResponse(fmt.Sprintf("Failed to write to existing stream for %s. Closing and removing stream.", pid), err)
		} else {
			// Stream does not exist, need to create a new one
			instancePersistentChatStreamsMutex.Unlock()

			logger.Debugf("[GO]   ↳ Instance %d: No existing stream to %s, creating new one...\n", instanceIndex, pid)
			streamCtx, cancel := context.WithTimeout(instanceCtx, 20*time.Second)
			defer cancel()

			newStream, err := instanceHost.NewStream(
				network.WithAllowLimitedConn(streamCtx, UnaiverseChatProtocol),
				pid,
				UnaiverseChatProtocol,
			)

			// Re-acquire lock *after* NewStream finishes or errors
			instancePersistentChatStreamsMutex.Lock()
			defer instancePersistentChatStreamsMutex.Unlock()

			if err != nil {
				// Failed to open a *new* stream
				if streamCtx.Err() == context.DeadlineExceeded || err == context.DeadlineExceeded {
					return jsonErrorResponse(fmt.Sprintf("Failed to open new stream to %s: Timeout", pid), err)
				}
				return jsonErrorResponse(fmt.Sprintf("Failed to open new stream to %s.", pid), err)
			}

			// --- RACE CONDITION HANDLING ---
			// Double-check if another goroutine created a stream while we were unlocked
			existingStream, existsNow := instancePersistentChatStreams[pid]
			if existsNow {
				logger.Warnf("[GO] ⚠️ Instance %d: Race condition: Another stream to %s was created. Using existing one and closing the new one.", instanceIndex, pid)
				_ = newStream.Close() // Close the redundant stream we just created.
				stream = existingStream
			} else {
				logger.Debugf("[GO] ✅ Instance %d: Opened and stored new persistent stream to %s\n", instanceIndex, pid)
				instancePersistentChatStreams[pid] = newStream
				stream = newStream
				go handleStream(instanceIndex, newStream)
			}

			// --- Write message to the determined stream ---
			err = writeDirectMessageFrame(stream, goChannel, goData)
			if err != nil {
				logger.Errorf("[GO] ❌ Instance %d: Failed to write initial message to stream for %s: %v. Closing and removing.", instanceIndex, pid, err)
				_ = stream.Close()
				if currentStream, ok := instancePersistentChatStreams[pid]; ok && currentStream == stream {
					delete(instancePersistentChatStreams, pid)
				}
				return jsonErrorResponse(fmt.Sprintf("Failed to write to new stream to '%s' (needs reconnect).", pid), err)
			}

			logger.Infof("[GO] 📤 Instance %d: Sent direct message to %s (on NEW stream)\n", instanceIndex, pid)
			return jsonSuccessResponse(fmt.Sprintf("Direct message sent to %s (new stream).", pid))
		}
	} else {
		// Invalid channel format
		return jsonErrorResponse(fmt.Sprintf("Invalid channel format '%s'", goChannel), nil)
	}
}

// SubscribeToTopic joins a PubSub topic and starts listening for messages for a specific instance.
// Parameters:
//   - instanceIndexC (C.int): The index of the node instance.
//   - channelC (*C.char): The Channel associated to the topic to subscribe to.
//
// Returns:
//   - *C.char: A JSON string indicating success or failure.
//     Structure: `{"state":"Success", "message":"Subscribed to topic ..."}` or `{"state":"Error", "message":"..."}`.
//   - IMPORTANT: The caller MUST free the returned C string using `FreeString`.
//
//export SubscribeToTopic
func SubscribeToTopic(
	instanceIndexC C.int,
	channelC *C.char,
) *C.char {

	instanceIndex := int(instanceIndexC)
	// Convert C string input to Go string.
	channel := C.GoString(channelC)
	logger.Debugf("[GO] <sub> Instance %d: Attempting to subscribe to topic: %s\n", instanceIndex, channel)

	// Check instance index validity
	if err := checkInstanceIndex(instanceIndex); err != nil {
		return jsonErrorResponse("Invalid instance index", err) // Caller frees.
	}

	// Get instance-specific state and mutex
	instanceHost := hostInstances[instanceIndex]
	instancePubsub := pubsubInstances[instanceIndex]
	instancePubsubMutex := &pubsubMutexes[instanceIndex]
	instanceTopics := topicsInstances[instanceIndex]
	instanceSubscriptions := subscriptionsInstances[instanceIndex]

	// Check if host and PubSub instances are ready for this instance.
	if instanceHost == nil || instancePubsub == nil {
		return jsonErrorResponse(
			fmt.Sprintf("Instance %d: Host or PubSub not initialized", instanceIndex), nil,
		) // Caller frees.
	}

	// Lock the mutex for safe access to the shared topics and subscriptions maps for this instance.
	instancePubsubMutex.Lock()
	defer instancePubsubMutex.Unlock() // Ensure mutex is unlocked when function returns.

	// Check if already subscribed to this topic for this instance.
	if _, exists := instanceSubscriptions[channel]; exists {
		logger.Debugf("[GO] <sub> Instance %d: Already subscribed to topic: %s\n", instanceIndex, channel)
		// Return success, indicating the desired state is already met.
		return jsonSuccessResponse(
			fmt.Sprintf("Instance %d: Already subscribed to topic %s", instanceIndex, channel),
		) // Caller frees.
	}

	// If the channel ends with ":rv", it indicates a rendezvous topic, so we remove other ones
	// from the instanceTopics and instanceSubscriptions list, and we clean the rendezvousDiscoveredPeersInstances.
	if strings.HasSuffix(channel, ":rv") {
		logger.Debugf("  - Instance %d: Joining rendezvous topic '%s'. Cleaning up previous rendezvous state.\n", instanceIndex, channel)
		// Remove all existing rendezvous topics and subscriptions for this instance.
		for existingChannel := range instanceTopics {
			if strings.HasSuffix(existingChannel, ":rv") {
				logger.Debugf("  - Instance %d: Removing existing rendezvous topic '%s' from instance state.\n", instanceIndex, existingChannel)

				// Close the topic handle if it exists.
				if topic, exists := instanceTopics[existingChannel]; exists {
					if err := topic.Close(); err != nil {
						logger.Warnf("⚠️ Instance %d: Error closing topic handle for '%s': %v (proceeding with map cleanup)\n", instanceIndex, existingChannel, err)
					}
					delete(instanceTopics, existingChannel)
				}

				// Remove the subscription if it exists.
				if sub, exists := instanceSubscriptions[existingChannel]; exists {
					sub.Cancel()                                   // Cancel the subscription
					delete(instanceSubscriptions, existingChannel) // Remove from map
				}

				// Also clean up rendezvous discovered peers for this instance.
				logger.Debugf("  - Instance %d: Resetting rendezvous state for new topic '%s'.\n", instanceIndex, channel)
				rendezvousDiscoveredPeersMutexes[instanceIndex].Lock()
				rendezvousDiscoveredPeersInstances[instanceIndex] = nil
				rendezvousDiscoveredPeersMutexes[instanceIndex].Unlock()
			}
		}
		logger.Debugf("  - Instance %d: Cleaned up previous rendezvous state.\n", instanceIndex)
	}

	// --- Join the Topic ---
	// Get a handle for the topic. `Join` creates the topic if it doesn't exist locally
	// and returns a handle. It's safe to call Join multiple times; it's idempotent.
	// We store the handle primarily for potential future publishing from this node.
	topic, err := instancePubsub.Join(channel)
	if err != nil {
		errMsg := fmt.Sprintf("Instance %d: Failed to join topic '%s'", instanceIndex, channel)
		return jsonErrorResponse(errMsg, err) // Caller frees.
	}
	// Store the topic handle in the map for this instance.
	instanceTopics[channel] = topic
	logger.Debugf("[GO]   - Instance %d: Obtained topic handle for: %s\n", instanceIndex, channel)

	// --- Subscribe to the Topic ---
	// Create an actual subscription to receive messages from the topic.
	sub, err := topic.Subscribe()
	if err != nil {
		// Close the newly created topic handle.
		err := topic.Close()
		if err != nil {
			// Log error but proceed with cleanup.
			logger.Warnf("[GO] ⚠️ Instance %d: Error closing topic handle for '%s': %v (proceeding with map cleanup)\n", instanceIndex, channel, err)
		}
		// Remove the topic handle from our local map for this instance.
		delete(instanceTopics, channel)
		errMsg := fmt.Sprintf("Instance %d: Failed to subscribe to topic '%s' after joining", instanceIndex, channel)
		return jsonErrorResponse(errMsg, err) // Caller frees.
	}
	// Store the subscription object in the map for this instance.
	instanceSubscriptions[channel] = sub
	logger.Debugf("[GO]   - Instance %d: Created subscription object for: %s\n", instanceIndex, channel)

	// --- Start Listener Goroutine ---
	// Launch a background goroutine that will continuously read messages
	// from this new subscription and add them to the message buffer for this instance.
	// Pass the instance index, subscription object, and topic name (for logging).
	go readFromSubscription(instanceIndex, sub)

	logger.Debugf("[GO] ✅ Instance %d: Subscribed successfully to topic: %s and started listener.\n", instanceIndex, channel)
	return jsonSuccessResponse(
		fmt.Sprintf("Instance %d: Subscribed to topic %s", instanceIndex, channel),
	) // Caller frees.
}

// UnsubscribeFromTopic cancels an active PubSub subscription and cleans up related resources for a specific instance.
// Parameters:
//   - instanceIndexC (C.int): The index of the node instance.
//   - channelC (*C.char): The Channel associated to the topic to unsubscribe from.
//
// Returns:
//   - *C.char: A JSON string indicating success or failure.
//     Structure: `{"state":"Success", "message":"Unsubscribed from topic ..."}` or `{"state":"Error", "message":"..."}`.
//   - IMPORTANT: The caller MUST free the returned C string using `FreeString`.
//
//export UnsubscribeFromTopic
func UnsubscribeFromTopic(
	instanceIndexC C.int,
	channelC *C.char,
) *C.char {

	instanceIndex := int(instanceIndexC)
	// Convert C string input to Go string.
	channel := C.GoString(channelC)
	logger.Debugf("[GO] </sub> Instance %d: Attempting to unsubscribe from topic: %s\n", instanceIndex, channel)

	// Check instance index validity
	if err := checkInstanceIndex(instanceIndex); err != nil {
		return jsonErrorResponse("Invalid instance index", err) // Caller frees.
	}

	// Get instance-specific state and mutex
	instancePubsubMutex := &pubsubMutexes[instanceIndex]
	instanceTopics := topicsInstances[instanceIndex]
	instanceSubscriptions := subscriptionsInstances[instanceIndex]

	// Check if host and PubSub are initialized. This is mostly for cleaning local state maps
	// if called after CloseNode, but Cancel/Close calls below require the instances.
	if hostInstances[instanceIndex] == nil || pubsubInstances[instanceIndex] == nil {
		logger.Debugf("[GO]  Instance %d: Host/PubSub not initialized during Unsubscribe. Cleaning up local subscription state only.\n", instanceIndex)
		// Allow local map cleanup even if instances are gone.
	}

	// Lock the mutex for write access to shared maps for this instance.
	instancePubsubMutex.Lock()
	defer instancePubsubMutex.Unlock() // Ensure unlock on exit.

	// --- Cancel the Subscription ---
	// Find the subscription object in the map for this instance.
	sub, subExists := instanceSubscriptions[channel]
	if !subExists {
		logger.Warnf("[GO] </sub> Instance %d: Not currently subscribed to topic: %s (or already unsubscribed)\n", instanceIndex, channel)
		// Also remove potential stale topic handle if subscription is gone.
		delete(instanceTopics, channel)
		return jsonSuccessResponse(
			fmt.Sprintf("Instance %d: Not currently subscribed to topic %s", instanceIndex, channel),
		) // Caller frees.
	}

	// Cancel the subscription. This signals the associated `readFromSubscription` goroutine
	// (waiting on `sub.Next()`) to stop by causing `sub.Next()` to return an error (usually `ErrSubscriptionCancelled`).
	// It also cleans up internal PubSub resources related to this subscription.
	sub.Cancel()
	// Remove the subscription entry from our local map for this instance.
	delete(instanceSubscriptions, channel)
	logger.Debugf("[GO]   - Instance %d: Cancelled subscription object for topic: %s\n", instanceIndex, channel)

	// --- Close the Topic Handle ---
	// Find the corresponding topic handle for this instance. It's good practice to close this as well,
	// although PubSub might manage its lifecycle internally based on subscriptions.
	// Explicit closing ensures resources related to the *handle* (like internal routing state) are released.
	topic, topicExists := instanceTopics[channel]
	if topicExists {
		logger.Debugf("[GO]   - Instance %d: Closing topic handle for: %s\n", instanceIndex, channel)
		// Close the topic handle.
		err := topic.Close()
		if err != nil {
			// Log error but proceed with cleanup.
			logger.Warnf("[GO] ⚠️ Instance %d: Error closing topic handle for '%s': %v (proceeding with map cleanup)\n", instanceIndex, channel, err)
		}
		// Remove the topic handle from our local map for this instance.
		delete(instanceTopics, channel)
		logger.Debugf("[GO]   - Instance %d: Removed topic handle from local map for topic: %s\n", instanceIndex, channel)
	} else {
		logger.Debugf("[GO]   - Instance %d: No topic handle found in local map for '%s' to close (already removed or possibly never stored?).\n", instanceIndex, channel)
		// Ensure removal from map even if handle wasn't found (e.g., inconsistent state).
		delete(instanceTopics, channel)
	}

	// If the channel ends with ":rv", it indicates a rendezvous topic, so we have closed the topic and the sub
	// but we also need to clean the rendezvousDiscoveredPeersInstances.
	if strings.HasSuffix(channel, ":rv") {
		logger.Debugf("  - Instance %d: Unsubscribing from rendezvous topic. Clearing state.\n", instanceIndex)
		rendezvousDiscoveredPeersMutexes[instanceIndex].Lock()
		rendezvousDiscoveredPeersInstances[instanceIndex] = nil
		rendezvousDiscoveredPeersMutexes[instanceIndex].Unlock()
	}
	logger.Debugf("[GO]   - Instance %d: Cleaned up previous rendezvous state.\n", instanceIndex)

	logger.Infof("[GO] ✅ Instance %d: Unsubscribed successfully from topic: %s\n", instanceIndex, channel)
	return jsonSuccessResponse(
		fmt.Sprintf("Instance %d: Unsubscribed from topic %s", instanceIndex, channel),
	) // Caller frees.
}

// MessageQueueLength returns the total number of messages waiting across all channel queues for a specific instance.
// Parameters:
//   - instanceIndexC (C.int): The index of the node instance.
//
// Returns:
//   - C.int: The total number of messages. Returns -1 if instance index is invalid.
//
//export MessageQueueLength
func MessageQueueLength(
	instanceIndexC C.int,
) C.int {

	instanceIndex := int(instanceIndexC)

	// Check instance index validity
	if err := checkInstanceIndex(instanceIndex); err != nil {
		logger.Errorf("[GO] ❌ MessageQueueLength: %v\n", err)
		return -1 // Indicate invalid instance index
	}

	// Get the message store for this instance
	store := messageStoreInstances[instanceIndex]
	if store == nil {
		logger.Errorf("[GO] ❌ Instance %d: Message store not initialized.\n", instanceIndex)
		return 0 // Return 0 if store is nil (effectively empty)
	}

	store.mu.Lock()
	defer store.mu.Unlock()

	totalLength := 0
	// TODO: this makes sense but not for the check we are doing from python, think about it
	for _, messageList := range store.messagesByChannel {
		totalLength += messageList.Len()
	}

	return C.int(totalLength)
}

// PopMessages retrieves the oldest message from each channel's queue for a specific instance.
// This function always pops one message per channel that has messages.
// Parameters:
//   - instanceIndexC (C.int): The index of the node instance.
//
// Returns:
//   - *C.char: A JSON string representing a list of the popped messages.
//     Returns `{"state":"Empty"}` if no messages were available in any queue.
//     Returns `{"state":"Error", "message":"..."}` on failure.
//   - IMPORTANT: The caller MUST free the returned C string using `FreeString`.
//
//export PopMessages
func PopMessages(
	instanceIndexC C.int,
) *C.char {
	instanceIndex := int(instanceIndexC)

	// Check instance index validity
	if err := checkInstanceIndex(instanceIndex); err != nil {
		return jsonErrorResponse("Invalid instance index", err)
	}

	// Get the message store for this instance
	store := messageStoreInstances[instanceIndex]
	if store == nil {
		logger.Errorf("[GO] ❌ Instance %d: PopMessages: Message store not initialized.\n", instanceIndex)
		return jsonErrorResponse(fmt.Sprintf("Instance %d: Message store not initialized", instanceIndex), nil)
	}

	store.mu.Lock() // Lock for the entire operation
	defer store.mu.Unlock()

	if len(store.messagesByChannel) == 0 {
		return C.CString(`{"state":"Empty"}`)
	}

	// Create a slice to hold the popped messages. Capacity is the number of channels.
	var poppedMessages []*QueuedMessage
	for _, messageList := range store.messagesByChannel {
		if messageList.Len() > 0 {
			element := messageList.Front()
			msg := element.Value.(*QueuedMessage)
			poppedMessages = append(poppedMessages, msg)
			messageList.Remove(element)
		}
	}

	// After iterating, check if we actually popped anything
	if len(poppedMessages) == 0 {
		return C.CString(`{"state":"Empty"}`)
	}

	// Marshal the slice of popped messages into a JSON array.
	// We create a temporary structure for JSON marshalling to include the base64-encoded data.
	payloads := make([]map[string]interface{}, len(poppedMessages))
	for i, msg := range poppedMessages {
		payloads[i] = map[string]interface{}{
			"from": msg.From,
			"data": base64.StdEncoding.EncodeToString(msg.Data),
		}
	}

	jsonBytes, err := json.Marshal(payloads)
	if err != nil {
		logger.Errorf("[GO] ❌ Instance %d: PopMessages: Failed to marshal messages to JSON: %v\n", instanceIndex, err)
		// Messages have already been popped from the queue at this point.
		// Returning an error is the best we can do.
		return jsonErrorResponse(
			fmt.Sprintf("Instance %d: Failed to marshal popped messages", instanceIndex), err,
		)
	}

	return C.CString(string(jsonBytes))
}

// CloseNode gracefully shuts down the libp2p host, cancels subscriptions, closes connections,
// and cleans up all associated resources.
// Parameters:
//   - instanceIndexC (C.int): The index of the node instance. If -1, closes all initialized instances.
//
// Returns:
//   - *C.char: A JSON string indicating the result of the closure attempt.
//     Structure: `{"state":"Success", "message":"Node closed successfully"}` or `{"state":"Error", "message":"Error closing host: ..."}`.
//     If closing all, the message will summarize the results.
//   - IMPORTANT: The caller MUST free the returned C string using `FreeString`.
//
//export CloseNode
func CloseNode(
	instanceIndexC C.int,
) *C.char {

	instanceIndex := int(instanceIndexC)

	if instanceIndex == -1 {
		logger.Debugf("[GO] 🛑 Closing all initialized instances of this node...")
		successCount := 0
		errorCount := 0
		var errorMessages []string

		// Iterate through all possible instance indices
		for i := 0; i < maxInstances; i++ {
			// Acquire global lock briefly to check if instance is initialized
			instanceStateMutex.RLock()
			isInstInitialized := isInitialized[i]
			instanceStateMutex.RUnlock()

			if isInstInitialized {
				logger.Debugf("[GO] 🛑 Attempting to close instance %d...\n", i)
				// Call the single instance close logic internally
				// This internal call will handle its own instance-specific locks
				resultPtr := closeSingleInstance(i)
				resultJSON := C.GoString(resultPtr)
				C.free(unsafe.Pointer(resultPtr)) // Free the C string from the internal call

				var result struct {
					State   string `json:"state"`
					Message string `json:"message"`
				}
				if err := json.Unmarshal([]byte(resultJSON), &result); err != nil {
					errorCount++
					errorMessages = append(errorMessages, fmt.Sprintf("Instance %d: Failed to parse close result: %v", i, err))
					logger.Errorf("[GO] ❌ Instance %d: Failed to parse close result: %v\n", i, err)
				} else if result.State == "Error" {
					errorCount++
					errorMessages = append(errorMessages, fmt.Sprintf("Instance %d: %s", i, result.Message))
					logger.Errorf("[GO] ❌ Instance %d: Close failed: %s\n", i, result.Message)
				} else {
					successCount++
					logger.Debugf("[GO] ✅ Instance %d: Closed successfully.\n", i)
				}
			}
		}

		summaryMsg := fmt.Sprintf("Closed %d nodes successfully, %d failed.", successCount, errorCount)
		if errorCount > 0 {
			logger.Errorf("[GO] ❌ Errors encountered during batch close:\n")
			for _, msg := range errorMessages {
				logger.Errorf(msg)
			}
			return jsonErrorResponse(summaryMsg, fmt.Errorf("details: %v", errorMessages))
		}

		logger.Infof("[GO] 🛑 All initialized nodes closed.")
		return jsonSuccessResponse(summaryMsg)

	} else {
		// --- Close a single specific instance ---
		logger.Infof("[GO] 🛑 Closing single node instance %d...\n", instanceIndex)
		// Check instance index validity for a single close
		if err := checkInstanceIndex(instanceIndex); err != nil {
			return jsonErrorResponse("Invalid instance index for single close", err) // Caller frees.
		}

		// Call the internal single instance close logic
		return closeSingleInstance(instanceIndex) // Caller frees the returned C string
	}
}

// FreeString is called from the C/Python side to release the memory allocated by Go
// when returning a `*C.char` (via `C.CString`).
// Parameters:
//   - s (*C.char): The pointer to the C string previously returned by an exported Go function.
//
//export FreeString
func FreeString(
	s *C.char,
) {

	// Check for NULL pointer before attempting to free.
	if s != nil {
		C.free(unsafe.Pointer(s)) // Use C.free via unsafe.Pointer to release the memory.
	}
}

// FreeInt is provided for completeness but is generally **NOT** needed if Go functions
// only return `C.int` (by value). It would only be necessary if a Go function manually
// allocated memory for a C integer (`*C.int`) and returned the pointer, which is uncommon.
// Parameters:
//   - i (*C.int): The pointer to the C integer previously allocated and returned by Go.
//
//export FreeInt
func FreeInt(
	i *C.int,
) {

	// Check for NULL pointer.
	if i != nil {
		logger.Warnf("[GO] ⚠️ FreeInt called - Ensure a *C.int pointer was actually allocated and returned from Go (this is unusual).")
		C.free(unsafe.Pointer(i)) // Free the memory if it was indeed allocated.
	}
}

// main is the entry point for a Go executable. However, when building a C shared library
// (`-buildmode=c-shared`), this function is not the primary entry point. The exported
// functions (`//export FunctionName`) serve as the entry points callable from C.
// Including a main function is still required by the Go compiler for the package `main`,
// but its content doesn't run when the code is used as a library. It can be useful
// for standalone testing of the package if needed.
func main() {
	// This message will typically only be seen if you run `go run lib.go`
	// or build and run as a standard executable, NOT when used as a shared library.
	logger.Debugf("[GO] libp2p Go library main function (not executed in c-shared library mode)")
}
