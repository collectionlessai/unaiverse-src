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
	"errors"          // For handling some types of errors
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
	rc "github.com/libp2p/go-libp2p/p2p/protocol/circuitv2/relay" // Import for relay service options
	autorelay "github.com/libp2p/go-libp2p/p2p/host/autorelay"	   // AutoRelay for automatic relay selection and usage
	rcmgr "github.com/libp2p/go-libp2p/p2p/host/resource-manager"			  // Resource manager for controlling resource usage (connections, streams)

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
	pg "unailib/proto-go" // Generated Protobuf code for our message formats

	"google.golang.org/protobuf/proto" // Core Protobuf library for marshalling/unmarshalling messages

	// PubSub library
	pubsub "github.com/libp2p/go-libp2p-pubsub" // GossipSub implementation for publish/subscribe messaging

	// Multiaddr libraries (libp2p's addressing format)
	ma "github.com/multiformats/go-multiaddr"        // Core multiaddr parsing and manipulation
)

// ChatProtocol defines the protocol ID string used for direct peer-to-peer messaging streams.
// This ensures that both peers understand how to interpret the data on the stream.
// const UnaiverseChatProtocol = "/unaiverse-chat-protocol/1.0.0"
const UnaiverseChatProtocol = "/unaiverse/chat/1.0.0"
const UnaiverseUserAgent = "go-libp2p/example/autotls"
const DisconnectionGracePeriod = 10 * time.Second

// ExtendedPeerInfo holds information about a connected peer.
type ExtendedPeerInfo struct {
	ID          peer.ID        `json:"id"`           // the Peer ID of the connected peer.
	Addrs       []ma.Multiaddr `json:"addrs"`        // the Multiaddr(s) associated with the peer.
	ConnectedAt time.Time      `json:"connected_at"` // Timestamp when the connection was established.
	Direction   string         `json:"direction"`    // Direction of the connection: "inbound" or "outbound".
	Misc        int            `json:"misc"`         // Misc information (integer), custom usage
	Relayed     bool           `json:"relayed"`      // Currently unused (but used in JS)
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

// NodeConfig contains the parameters to initialize a node
type NodeConfig struct {
    IdentityDir     string   `json:"identity_dir"`
    PredefinedPort  int      `json:"predefined_port"`
    ListenIPs       []string `json:"listen_ips"`
    
    // Group Relay Logic
    Relay struct {
        EnableClient  bool `json:"enable_client"`
        EnableService bool `json:"enable_service"`
		WithBroadLimits bool `json:"with_broad_limits"`
    } `json:"relay"`

    // Group TLS Logic (Mutually exclusive logic becomes clear here)
    TLS struct {
        AutoTLS     bool   `json:"auto_tls"`
        Domain      string `json:"domain"`
        CertPath    string `json:"cert_path"`
        KeyPath     string `json:"key_path"`
    } `json:"tls"`

    // explicit configuration for network environment
    Network struct {
        Isolated bool `json:"isolated"` // only allows connections with friendly peers
		ForcePublic bool `json:"force_public"` // Replaces knowsIsPublic
    } `json:"network"`

	// Group DHT logic
	DHT struct {
		Enabled bool `json:"enabled"`
		Keep bool	`json:"keep"` // to keep it running after init
    } `json:"dht"`
}

// CreateNodeResponse defines the structure of our success message.
type CreateNodeResponse struct {
	Addresses []string `json:"addresses"`
	IsPublic  bool     `json:"isPublic"`
}

// NodeInstance holds ALL state for a single libp2p node.
type NodeInstance struct {
	// Core Components
	host         host.Host
	pubsub       *pubsub.PubSub
	dht          *dht.IpfsDHT
	ctx          context.Context
	cancel       context.CancelFunc
	certManager  *p2pforge.P2PForgeCertMgr
	messageStore *MessageStore

	// Address Cache
    addrMutex  sync.RWMutex
    localAddrs []ma.Multiaddr

	// Static relay
	privateRelay *autorelay.AutoRelay
	// privateRelayAddrs []ma.Multiaddr

	// PubSub State
	pubsubMutex   sync.RWMutex
	topics        map[string]*pubsub.Topic
	subscriptions map[string]*pubsub.Subscription

	// Peer State
	peersMutex     sync.RWMutex
	friendlyPeers map[peer.ID]ExtendedPeerInfo

	// Stream State
	streamsMutex          sync.Mutex
	persistentChatStreams map[peer.ID]network.Stream

	// Disconnection Grace Period State
    disconnectionMutex  sync.Mutex
    disconnectionTimers map[peer.ID]context.CancelFunc

	// Rendezvous State
	rendezvousMutex sync.RWMutex
	rendezvousState *RendezvousState

	// a copy of its own index for logging
	instanceIndex int
}

// --- Create a package-level logger ---
var logger = golog.Logger("unailib")

// --- Multi-Instance State Management ---
var (
	// Set the libp2p configuration parameters.
	maxInstances       int
	maxChannelQueueLen int
	maxUniqueChannels  int
	MaxMessageSize     uint32

	// A single slice to hold all our instances.
	allInstances []*NodeInstance
	// A SINGLE mutex to protect the allInstances slice itself (during create/close).
	globalInstanceMutex sync.RWMutex
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

// getInstance is a new helper to safely retrieve a node instance.
// It handles bounds checking, nil checks, and locking.
func getInstance(instanceIndex int) (*NodeInstance, error) {
	if instanceIndex < 0 || instanceIndex >= maxInstances {
		return nil, fmt.Errorf("invalid instance index: %d. Must be between 0 and %d", instanceIndex, maxInstances-1)
	}

	// Use a Read Lock, which is fast and allows concurrent reads.
	globalInstanceMutex.RLock()
	instance := allInstances[instanceIndex]
	globalInstanceMutex.RUnlock()

	if instance == nil {
		return nil, fmt.Errorf("instance %d is not initialized or has been closed", instanceIndex)
	}

	// Check host as a proxy for *full* initialization,
	// as it's set late in CreateNode
	if instance.host == nil {
		return nil, fmt.Errorf("instance %d is not fully initialized (host is nil)", instanceIndex)
	}

	return instance, nil
}

// newMessageStore initializes a new MessageStore.
func newMessageStore() *MessageStore {
	return &MessageStore{
		messagesByChannel: make(map[string]*list.List),
	}
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

// --- Core Logic Functions ---

// storeReceivedMessage processes a raw message received either from a direct stream
// or a PubSub topic. The sender peerID and the channel to store are retrieved in handleStream and readFromSubscription
func storeReceivedMessage(
	ni *NodeInstance,
	from peer.ID,
	channel string,
	data []byte,
) {
	// Get the message store for this instance
	store := ni.messageStore
	if store == nil {
		logger.Errorf("[GO] ❌ storeReceivedMessage: Message store not initialized for instance %d\n", ni.instanceIndex)
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
			logger.Warnf("[GO] 🗑️ Instance %d: Message store full. Discarding message for new channel '%s'.\n", ni.instanceIndex, channel)
			return
		}
		messageList = list.New()
		store.messagesByChannel[channel] = messageList
		logger.Debugf("[GO] ✨ Instance %d: Created new channel queue '%s'. Total channels: %d\n", ni.instanceIndex, channel, len(store.messagesByChannel))
	}

	// If the channel already has a message list, check its length.
	if messageList.Len() >= maxChannelQueueLen {
		logger.Warnf("[GO] 🗑️ Instance %d: Queue for channel '%s' full. Discarding message.\n", ni.instanceIndex, channel)
		return
	}

	messageList.PushBack(newMessage)
	logger.Debugf("[GO] 📥 Instance %d: Queued message on channel '%s' from %s. New queue length: %d\n", ni.instanceIndex, channel, from, messageList.Len())
}

// readFromSubscription runs as a dedicated goroutine for each active PubSub subscription for a specific instance.
// It continuously waits for new messages on the subscription's channel (`sub.Next(ctx)`),
// routes them to `storeReceivedMessage`, and handles errors and context cancellation gracefully.
// You need to provide the full Channel to uniquely identify the subscription.
func readFromSubscription(
	ni *NodeInstance,
	sub *pubsub.Subscription,
) {
	// Get the topic string directly from the subscription object.
	topic := sub.Topic()

	if ni.ctx == nil || ni.host == nil {
		logger.Errorf("[GO] ❌ readFromSubscription: Context or Host not initialized for instance %d. Exiting goroutine.\n", ni.instanceIndex)
		return
	}

	logger.Infof("[GO] 👂 Instance %d: Started listener goroutine for topic: %s\n", ni.instanceIndex, topic)
	defer logger.Infof("[GO] 👂 Instance %d: Exiting listener goroutine for topic: %s\n", ni.instanceIndex, topic) // Log when goroutine exits

	for {
		// Check if the main context has been cancelled (e.g., during node shutdown).
		if ni.ctx.Err() != nil {
			logger.Debugf("[GO] 👂 Instance %d: Context cancelled, stopping listener goroutine for topic: %s\n", ni.instanceIndex, topic)
			return // Exit the goroutine.
		}

		// Wait for the next message from the subscription. This blocks until a message
		// arrives, the context is cancelled, or an error occurs.
		msg, err := sub.Next(ni.ctx)
		if err != nil {
			// Check for expected errors during shutdown or cancellation.
			if err == context.Canceled || err == context.DeadlineExceeded || err == pubsub.ErrSubscriptionCancelled || ni.ctx.Err() != nil {
				logger.Debugf("[GO] 👂 Instance %d: Subscription listener for topic '%s' stopping gracefully: %v\n", ni.instanceIndex, topic, err)
				return // Exit goroutine cleanly.
			}
			// Handle EOF, which can sometimes occur. Treat it as a reason to stop.
			if err == io.EOF {
				logger.Debugf("[GO] 👂 Instance %d: Subscription listener for topic '%s' encountered EOF, stopping: %v\n", ni.instanceIndex, topic, err)
				return // Exit goroutine.
			}
			// Log other errors but attempt to continue (they might be transient).
			logger.Errorf("[GO] ❌ Instance %d: Error reading from subscription '%s': %v. Continuing...\n", ni.instanceIndex, topic, err)
			// Pause briefly to avoid busy-looping on persistent errors.
			time.Sleep(1 * time.Second)
			continue // Continue the loop to try reading again.
		}

		logger.Infof("[GO] 📬 Instance %d (id: %s): Received new PubSub message on topic '%s' from %s\n", ni.instanceIndex, ni.host.ID().String(), topic, msg.GetFrom())

		// Ignore messages published by the local node itself.
		if msg.GetFrom() == ni.host.ID() {
			continue // Skip processing self-sent messages.
		}

		// Handle Rendezvous or Standard Messages
		if strings.HasSuffix(topic, ":rv") {
			// This is a rendezvous update.
			// 1. First, unmarshal the outer Protobuf message.
			var protoMsg pg.Message
			if err := proto.Unmarshal(msg.Data, &protoMsg); err != nil {
				logger.Warnf("⚠️ Instance %d: Could not decode Protobuf message on topic '%s': %v\n", ni.instanceIndex, topic, err)
				continue
			}

			// 2. The actual payload is a JSON string within the 'json_content' field.
			jsonPayload := protoMsg.GetJsonContent()
			if jsonPayload == "" {
				logger.Warnf("⚠️ Instance %d: Rendezvous message on topic '%s' has empty JSON content.\n", ni.instanceIndex, topic)
				continue
			}

			// 3. Now, unmarshal the inner JSON payload.
			var updatePayload struct {
				Peers       []ExtendedPeerInfo `json:"peers"`
				UpdateCount int64              `json:"update_count"`
			}
			if err := json.Unmarshal([]byte(jsonPayload), &updatePayload); err != nil {
				logger.Warnf("[GO] ⚠️ Instance %d: Could not decode rendezvous update payload on topic '%s': %v\n", ni.instanceIndex, topic, err)
				continue // Skip this malformed message.
			}

			// 4. Create a new map from the decoded peer list.
			newPeerMap := make(map[peer.ID]ExtendedPeerInfo)
			for _, peerInfo := range updatePayload.Peers {
				newPeerMap[peerInfo.ID] = peerInfo
			}

			// 5. Safely replace the old map with the new one.
			ni.rendezvousMutex.Lock()
			// If this is the first update for this instance, initialize the state struct.
			if ni.rendezvousState == nil {
				ni.rendezvousState = &RendezvousState{}
			}
			rendezvousState := ni.rendezvousState
			rendezvousState.Peers = newPeerMap
			rendezvousState.UpdateCount = updatePayload.UpdateCount
			ni.rendezvousMutex.Unlock()

			logger.Debugf("[GO] ✅ Instance %d: Updated rendezvous peers from topic '%s'. Found %d peers. Update count: %d.\n", ni.instanceIndex, topic, len(newPeerMap), updatePayload.UpdateCount)
		} else {
			// This is a standard message. Queue it as before.
			logger.Debugf("[GO] 📝 Instance %d: Storing new pubsub message from topic '%s'.\n", ni.instanceIndex, topic)
			storeReceivedMessage(ni, msg.GetFrom(), topic, msg.Data)
		}
	}
}

// handleStream reads from a direct message stream using the new framing protocol.
// It expects the stream to start with a 4-byte length prefix, followed by a 1-byte channel name length,
// the channel name itself, and finally the Protobuf-encoded payload.
func handleStream(ni *NodeInstance, s network.Stream) {
	senderPeerID := s.Conn().RemotePeer()
	streamID := s.ID()
	ni.peersMutex.Lock()
	existingPeer, peerExists := ni.friendlyPeers[senderPeerID]

	// 1. Gather fresh info (Addresses & Direction)
	direction := "incoming"
	if s.Stat().Direction == network.DirOutbound {
		direction = "outgoing"
	}
	knownAddrs := ni.host.Peerstore().Addrs(senderPeerID)
	if len(knownAddrs) == 0 {
		knownAddrs = []ma.Multiaddr{s.Conn().RemoteMultiaddr()}
	}

	if !peerExists {
		// CASE A: New Application Peer
		ni.friendlyPeers[senderPeerID] = ExtendedPeerInfo{
			ID:          senderPeerID,
			Addrs:       knownAddrs,
			ConnectedAt: time.Now(),
			Direction:   direction,
			Relayed:     false,
		}
		logger.Infof("[GO] ➕ Instance %d: Peer %s promoted to App Peer via Stream %s (Incoming).", ni.instanceIndex, senderPeerID, streamID)
	} else {
		// CASE B: Existing Peer - Update Addresses
		// We keep ConnectedAt and Direction from the original session start.
		existingPeer.Addrs = knownAddrs
		ni.friendlyPeers[senderPeerID] = existingPeer
		logger.Debugf("[GO] 🔄 Instance %d: Refreshed addresses for Peer %s via Stream %s.", ni.instanceIndex, senderPeerID, streamID)
	}
	ni.peersMutex.Unlock()
	logger.Debugf("[GO] 📥 Instance %d: Accepted INCOMING stream %s from %s. Storing for duplex use.\n", ni.instanceIndex, streamID, senderPeerID)

	// Store the newly accepted stream so we can use it to send messages back to this peer.
	ni.streamsMutex.Lock()
	ni.persistentChatStreams[senderPeerID] = s
	ni.streamsMutex.Unlock()

	// This defer block ensures cleanup happens when the stream is closed by either side.
	defer func() {
		logger.Debugf("[GO] 🧹 Instance %d: Stream %s with %s closed. Removing from map.\n", ni.instanceIndex, streamID, senderPeerID)
		ni.streamsMutex.Lock()
		if current, ok := ni.persistentChatStreams[senderPeerID]; ok && current == s {
			delete(ni.persistentChatStreams, senderPeerID)
		}
		ni.streamsMutex.Unlock()
		s.Close() // Ensure the stream is fully closed.
	}()

	for {
		// Read 4-byte total length
		var totalLen uint32
		if err := binary.Read(s, binary.BigEndian, &totalLen); err != nil {
			if err == io.EOF {
				logger.Debugf("[GO] 🔌 Instance %d: Stream %s with %s closed (EOF).\n", ni.instanceIndex, streamID, senderPeerID)
			} else if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
				logger.Warnf("[GO] ⏳ Instance %d: Timeout reading length from Stream %s (%s): %v\n", ni.instanceIndex, streamID, senderPeerID, err)
			} else if errors.Is(err, network.ErrReset) {
				logger.Warnf("[GO] ⚙️ Instance %d: Stream %s with %s reset.\n", ni.instanceIndex, streamID, senderPeerID)
			} else {
				logger.Errorf("[GO] ❌ Instance %d: Error reading length from Stream %s (%s): %v\n", ni.instanceIndex, streamID, senderPeerID, err)
			}
			return
		}

		if totalLen > MaxMessageSize {
			logger.Errorf("[GO] ❌ Instance %d: Message len %d exceeds limit on Stream %s. Resetting.\n", ni.instanceIndex, totalLen, streamID)
			s.Reset()
			return
		}

		// Read Channel Length
		var channelLen uint8
		if err := binary.Read(s, binary.BigEndian, &channelLen); err != nil {
			logger.Errorf("[GO] ❌ Instance %d: Error reading channel len from Stream %s: %v\n", ni.instanceIndex, streamID, err)
			return
		}

		// Read Channel Name
		channelBytes := make([]byte, channelLen)
		if _, err := io.ReadFull(s, channelBytes); err != nil {
			logger.Errorf("[GO] ❌ Instance %d: Error reading channel from Stream %s: %v\n", ni.instanceIndex, streamID, err)
			return
		}
		channel := string(channelBytes)

		// Read Payload
		payloadLen := totalLen - uint32(channelLen) - 1
		payload := make([]byte, payloadLen)
		if _, err := io.ReadFull(s, payload); err != nil {
			logger.Errorf("[GO] ❌ Instance %d: Error reading payload from Stream %s: %v\n", ni.instanceIndex, streamID, err)
			return
		}

		logger.Infof("[GO] 📨 Instance %d: Received msg on channel '%s' via Stream %s from %s.\n", ni.instanceIndex, channel, streamID, senderPeerID)
		storeReceivedMessage(ni, senderPeerID, channel, payload)
	}
}

// setupDirectMessageHandler configures the libp2p host for a specific instance
// to listen for incoming streams using the custom ChatProtocol.
// When a peer opens a stream with this protocol ID, the provided handler function
// is invoked to manage communication on that stream.
func setupDirectMessageHandler(
	ni *NodeInstance,
) {
	if ni.host == nil {
		logger.Errorf("[GO] ❌ Instance %d: Cannot setup direct message handler: Host not initialized\n", ni.instanceIndex)
		return
	}

	// Set a handler function for the UnaiverseChatProtocol. This function will be called
	// automatically by libp2p whenever a new incoming stream for this protocol is accepted.
	// Use a closure to capture the NodeInstance pointer.
	ni.host.SetStreamHandler(UnaiverseChatProtocol, func(s network.Stream) {
		handleStream(ni, s)
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
// It takes a pointer to a NodeInstance and a targetPID. If targetPID is empty (peer.ID("")),
// it fetches addresses for the local node of the given instance.
// It returns a slice of fully formatted multiaddress strings and an error if one occurs.
func goGetNodeAddresses(
	ni *NodeInstance,
	targetPID peer.ID,
) ([]string, error) {
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
		// candidateAddrs = append(candidateAddrs, ni.privateRelayAddrs...)
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
	addrSet := make(map[string]struct{})
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
			logger.Warnf("[GO] ⚠️ Instance %d: Discarding stale address for peer %s: %s\n", ni.instanceIndex, resolvedPID, addr)
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

	// --- Close AutoRelay ---
	if ni.privateRelay != nil {
		logger.Debugf("[GO]   - Instance %d: Closing AutoRelay service...\n", ni.instanceIndex)
		if err := ni.privateRelay.Close(); err != nil { //
			logger.Warnf("[GO] ⚠️ Instance %d: Error closing AutoRelay: %v\n", ni.instanceIndex, err)
		}
		ni.privateRelay = nil
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
	// ni.privateRelayAddrs = nil
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
	logConfigJSONC *C.char,
) {
	// --- Configure Logging FIRST ---
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
	configStr := C.GoString(logConfigJSONC)
	golog.SetAllLoggers(golog.LevelFatal)
	if configStr != "" {
		var logLevels map[string]string
		if err := json.Unmarshal([]byte(configStr), &logLevels); err != nil {
			log.Printf("[GO] ⚠️ Invalid log config JSON: %v. Using defaults.\n", err)
		} else {
			for logger, levelStr := range logLevels {
				if err := golog.SetLogLevel(logger, levelStr); err != nil {
					log.Printf("[GO] ⚠️ Failed to set log level for '%s': %v\n", logger, err)
				}
			}
		}
	}
	
	maxInstances = int(maxInstancesC)
	maxUniqueChannels = int(maxUniqueChannelsC)
	maxChannelQueueLen = int(maxChannelQueueLenC)
	MaxMessageSize = uint32(maxMessageSizeC)

	// Initialize the *single* global slice.
	allInstances = make([]*NodeInstance, maxInstances)
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
	configJSONC *C.char,
) (ret *C.char) {

	instanceIndex := int(instanceIndexC)

	if instanceIndex < 0 || instanceIndex >= maxInstances {
		errMsg := fmt.Errorf("invalid instance index: %d. Must be between 0 and %d", instanceIndex, maxInstances-1)
		return jsonErrorResponse("Invalid instance index", errMsg)
	}

	// --- Instance Creation and State Check ---
	globalInstanceMutex.Lock()
	if allInstances[instanceIndex] != nil {
		globalInstanceMutex.Unlock()
		msg := fmt.Sprintf("Instance %d is already initialized. Please call CloseNode first.", instanceIndex)
		return jsonErrorResponse(msg, nil)
	}

	// --- Create the new instance object ---
	ni := &NodeInstance{
		instanceIndex:         instanceIndex,
		topics:                make(map[string]*pubsub.Topic),
		subscriptions:         make(map[string]*pubsub.Subscription),
		friendlyPeers:         make(map[peer.ID]ExtendedPeerInfo),
		persistentChatStreams: make(map[peer.ID]network.Stream),
		disconnectionTimers:   make(map[peer.ID]context.CancelFunc),
		messageStore:          newMessageStore(),
	}
	ni.ctx, ni.cancel = context.WithCancel(context.Background())
	isPublic := false

	// Store it in the global slice
	allInstances[instanceIndex] = ni
	globalInstanceMutex.Unlock()

	logger.Infof("[GO] 🚀 Instance %d: Starting CreateNode...", instanceIndex)
	// --- Centralized Cleanup on Failure ---
	var success bool = false
	defer func() {
		if !success {
			// If `success` is still false when CreateNode exits, an error
			// must have occurred. We call Close() and remove the instance.
			logger.Warnf("[GO] ⚠️ Instance %d: CreateNode failed, cleaning up...", instanceIndex)
			ni.Close() // Call the new method!
			globalInstanceMutex.Lock()
			allInstances[instanceIndex] = nil // Remove it from the global list
			globalInstanceMutex.Unlock()
		}
	}()

	// 1. Parse Configuration
	configJSON := C.GoString(configJSONC)
	var cfg NodeConfig
	if err := json.Unmarshal([]byte(configJSON), &cfg); err != nil {
		return jsonErrorResponse("Invalid Configuration JSON", err)
	}

	// --- Sanity checks on the config ---
	// If one of the three parameters for custom certificates is specified, all three are required.
	if cfg.TLS.Domain != "" || cfg.TLS.CertPath != "" || cfg.TLS.KeyPath != "" {
		if cfg.TLS.Domain == "" || cfg.TLS.CertPath == "" || cfg.TLS.KeyPath == "" {
			return jsonErrorResponse(fmt.Sprintf("Instance %d: Missing at least one of 'Domain', 'CertPath' or 'KeyPath'.", instanceIndex), nil)
		}
	} // in the following, cfg.TLS.Domain != "" will be used as flag for useCustomTLS

	// Having both customTLS and autoTLS is not allowed
	if cfg.TLS.Domain != "" && cfg.TLS.AutoTLS {
		return jsonErrorResponse(fmt.Sprintf("Instance %d: Cannot specify both a 'Domain' and 'AutoTLS'.", instanceIndex), nil)
	}

	// If we use AutoTLS we need the DHT on
	if cfg.TLS.AutoTLS && !cfg.DHT.Enabled {
		return jsonErrorResponse(fmt.Sprintf("Instance %d: Using TLS requires DHT 'Enabled'.", instanceIndex), nil)
	}

	// If we want RelayService we must be public (either forced or via AutoNat)
	if cfg.Relay.EnableService {
		if !cfg.Relay.EnableClient {
			return jsonErrorResponse(fmt.Sprintf("Instance %d: Cannot set libp2p.DisableRelay() if we want to offer relay services.", instanceIndex), nil)
		}
		if !(cfg.DHT.Enabled || cfg.Network.ForcePublic) {
			return jsonErrorResponse(fmt.Sprintf("Instance %d: A relay needs to be publicly reachable (forced or discovered).", instanceIndex), nil)
		}
	}

	// If we want to keep dht it needs to be enabled
	if cfg.DHT.Keep && !cfg.DHT.Enabled {
		return jsonErrorResponse(fmt.Sprintf("Instance %d: Cannot set 'DHT.Keep' if DHT is not 'Enabled'.", instanceIndex), nil)
	}

	// --- Load or Create Persistent Identity ---
	keyPath := filepath.Join(cfg.IdentityDir, "identity.key")
	privKey, err := loadOrCreateIdentity(keyPath)
	if err != nil {
		return jsonErrorResponse(fmt.Sprintf("Instance %d: Failed to prepare identity", instanceIndex), err)
	}

	// --- AutoTLS Cert Manager Setup (if enabled) ---
	var certManager *p2pforge.P2PForgeCertMgr
	if cfg.TLS.AutoTLS {
		logger.Debugf("[GO]   - Instance %d: AutoTLS is ENABLED. Setting up certificate manager...\n", instanceIndex)
		certManager, err = p2pforge.NewP2PForgeCertMgr(
			p2pforge.WithCAEndpoint(p2pforge.DefaultCAEndpoint),
			p2pforge.WithCertificateStorage(&certmagic.FileStorage{Path: filepath.Join(cfg.IdentityDir, "p2p-forge-certs")}),
			p2pforge.WithUserAgent(UnaiverseUserAgent),
			p2pforge.WithRegistrationDelay(10*time.Second),
		)
		if err != nil {
			return jsonErrorResponse(fmt.Sprintf("Instance %d: Failed to create AutoTLS cert manager", instanceIndex), err)
		}
		certManager.Start()
		ni.certManager = certManager
	}

	// --- 4. Libp2p Options Assembly ---
	tlsMode := "none"
	if cfg.TLS.AutoTLS {
		tlsMode = "autotls"
	} else if cfg.TLS.Domain != "" {
		tlsMode = "domain"
	}
	listenAddrs, err := getListenAddrs(cfg.ListenIPs, cfg.PredefinedPort, tlsMode)
	if err != nil {
		return jsonErrorResponse(fmt.Sprintf("Instance %d: Failed to create multiaddrs", instanceIndex), err)
	}

    // --- Configure Custom Resource Manager ---
    scalingLimits := rcmgr.DefaultLimits
    libp2p.SetDefaultServiceLimits(&scalingLimits)

    // These apply per unique Peer ID.
    scalingLimits.PeerBaseLimit.Conns = 64
    scalingLimits.PeerBaseLimit.ConnsInbound = 64
    scalingLimits.PeerBaseLimit.ConnsOutbound = 64

    // Tweak System Limits
    scalingLimits.SystemBaseLimit.Conns = 256
    scalingLimits.SystemBaseLimit.ConnsInbound = 128
    scalingLimits.SystemBaseLimit.ConnsOutbound = 128

    // Compute the concrete limits
    scaledLimits := scalingLimits.AutoScale()

	// Raise the per-IP limits
	customIP4Limits := []rcmgr.ConnLimitPerSubnet{
        {
            PrefixLength: 32,   // /32 means "one specific IP address"
            ConnCount:    1024, // Allow 1024 conns from the same IP
        },
    }
	customIP6Limits := []rcmgr.ConnLimitPerSubnet{
        {
            PrefixLength: 56,
            ConnCount:    1024,
        },
    }

    // Create the limiter and manager
    limiter := rcmgr.NewFixedLimiter(scaledLimits)
    rm, err := rcmgr.NewResourceManager(
		limiter,
		rcmgr.WithLimitPerSubnet(customIP4Limits, customIP6Limits),
	)
    if err != nil {
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
		libp2p.ResourceManager(rm),
		libp2p.UserAgent(UnaiverseUserAgent),
		libp2p.NATPortMap(),
		libp2p.EnableHolePunching(),
	}

	// Add WebSocket transport, with or without TLS based on cert availability
	if cfg.TLS.Domain != "" {
		// We already have certificates, use them
		logger.Debugf("[GO]   - Instance %d: Certificates provided, setting up secure WebSocket (WSS).\n", instanceIndex)
		cert, err := tls.LoadX509KeyPair(cfg.TLS.CertPath, cfg.TLS.KeyPath)
		if err != nil {
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
					portStr, err := addr.ValueForProtocol(ma.P_TCP)
					if err != nil {
						// Should not happen for a TCP/WS address, but safe fallback
						result = append(result, addr)
						continue
					}
					dnsAddr, _ := ma.NewMultiaddr(fmt.Sprintf("/dns4/%s/tcp/%s/tls/ws", cfg.TLS.Domain, portStr))
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
	} else if cfg.TLS.AutoTLS {
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

	// Prepare discovering the bootstrap peers
	if cfg.DHT.Enabled {
		// Add any possible option to be publicly reachable
		discoveryOpts := []libp2p.Option{
			libp2p.EnableAutoNATv2(),
			libp2p.Routing(func(h host.Host) (routing.PeerRouting, error) {
				bootstrapAddrInfos := dht.GetDefaultBootstrapPeerAddrInfos()
				dhtOptions := []dht.Option{
					dht.Mode(dht.ModeClient),
					dht.BootstrapPeers(bootstrapAddrInfos...),
				}
				var err error
				ni.dht, err = dht.New(ni.ctx, h, dhtOptions...)
				return ni.dht, err
			}),
		}
		options = append(options, discoveryOpts...)
		logger.Debugf("[GO]   - Instance %d: Trying to be publicly reachable.\n", instanceIndex)
	}

	// EnableRelay (the ability to *use* relays) is default, we can explicitly disable it if needed.
	if !cfg.Relay.EnableClient {
		// In this case we don't want to use the circuit-relay protocol.
		options = append(options, libp2p.DisableRelay()) // Explicitly disable using relays.
		logger.Debugf("[GO]   - Instance %d: Relay client is DISABLED.\n", instanceIndex)
	} else {
		// Configure Relay Service (ability to *be* a relay)
		if cfg.Relay.EnableService {
			resources := rc.DefaultResources() // open this to see the default resource limits
			resources.MaxReservations = 1024			// default is 128
			resources.MaxCircuits = 32					// default is 16
			resources.BufferSize = 4096					// default is 2048
			resources.MaxReservationsPerIP = 1024		// default is 8
			resources.MaxReservationsPerASN = 1024		// default is 32
			if cfg.Relay.WithBroadLimits {
				// Enrich default limits
				resources.Limit = nil // same as setting rc.WithInfiniteLimits()
				logger.Debugf("[GO]   - Instance %d: Relay service is ENABLED with custom resource configuration (WithBroadLimits).\n", instanceIndex)
			} else {
				logger.Debugf("[GO]   - Instance %d: Relay service is ENABLED with default resource configuration.\n", instanceIndex)
			}
			// This single option enables the node to act as a relay for others.
			options = append(options, libp2p.EnableRelayService(rc.WithResources(resources)), libp2p.EnableNATService())
		} else {
			// In this case we want to use relays but not offer the service to others.
			// If we are exploiting the DHT we can start an AutoRelay with PeerSource
			if cfg.DHT.Keep {
				// Enable AutoRelay. This uses the services above (DHT, AutoNAT)
				// to find relays and bind to one if we are private.
				options = append(options, libp2p.EnableAutoRelayWithPeerSource(ni.PeerSource, autorelay.WithBootDelay(time.Second*10)))
				logger.Debugf("[GO]   - Instance %d: AutoRelay client ENABLED.\n", instanceIndex)
			}
		}
	}

	if cfg.Network.ForcePublic {
		// Force public reachability to test local relays
		options = append(options, libp2p.ForceReachabilityPublic())
	}

	// Create the libp2p Host instance with the configured options for this instance.
	host, err := libp2p.New(options...)
	if err != nil {
		return jsonErrorResponse(fmt.Sprintf("Instance %d: Failed to create host", instanceIndex), err)
	}
	ni.host = host
	logger.Infof("[GO] ✅ Instance %d: Host created with ID: %s\n", instanceIndex, ni.host.ID())

	if cfg.Network.Isolated {
        // Turn on the "Protocol Police"
        enforceProtocolCompliance(ni)
    }

	// --- Link Host to Cert Manager ---
	if cfg.TLS.AutoTLS {
		certManager.ProvideHost(ni.host)
		logger.Debugf("[GO]   - Instance %d: Provided host to AutoTLS cert manager.\n", instanceIndex)
	}

	// --- Start Address Reporting & Caching ---
	cacheSub, err := ni.host.EventBus().Subscribe(new(event.EvtLocalAddressesUpdated))
	if err != nil {
		return jsonErrorResponse(fmt.Sprintf("Instance %d: Failed to create address cache subscription", instanceIndex), err)
	}
	go handleAddressUpdateEvents(ni, cacheSub)
	logger.Debugf("[GO] 🧠 Instance %d: Address cache background listener started.", instanceIndex)

	if cfg.Network.ForcePublic {
		isPublic = true
		logger.Debugf("[GO] ⏳ Instance %d: ForcePublic is ON. Waiting for addresses to settle...", instanceIndex)
		waitCtx, waitCancel := context.WithTimeout(ni.ctx, 5*time.Second)
		defer waitCancel()

		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()

		AddressWaitLoop:
		for {
			select {
			case <-waitCtx.Done():
				logger.Warnf("[GO] ⚠️ Instance %d: Timed out waiting for addresses (proceeding anyway).", instanceIndex)
				break AddressWaitLoop
			case <-ticker.C:
				// Check if the host has reported addresses yet
				if len(ni.host.Addrs()) > 0 {
					logger.Debugf("[GO] ✅ Instance %d: Addresses populated.", instanceIndex)
					break AddressWaitLoop
				}
			}
		}
	} else {
		// --- Wait for Reachability Update ---
		// Subscribe to reachability events
		reachSub, err := ni.host.EventBus().Subscribe(new(event.EvtLocalReachabilityChanged))
		if err != nil {
			return jsonErrorResponse("Failed to subscribe to reachability events", err)
		}
		defer reachSub.Close()

		timeoutCtx, timeoutCancel := context.WithTimeout(ni.ctx, 30*time.Second)
		defer timeoutCancel()
		logger.Debugf("[GO] ⏳ Instance %d: Waiting for reachability update.", instanceIndex)
		
		WAIT_LOOP:
		for {		
			select {
			case evt := <-reachSub.Out():
				rEvt := evt.(event.EvtLocalReachabilityChanged)
				if rEvt.Reachability == network.ReachabilityPublic {
					logger.Debugf("[GO] 📶 Instance %d: Reachability -> PUBLIC", instanceIndex)
					isPublic = true
				} else {
					isPublic = false
				}
				break WAIT_LOOP
		
			case <-timeoutCtx.Done():
				logger.Warnf("[GO] ⚠️ Instance %d: Timeout. Proceeding with best effort. (Public: %t)", instanceIndex, isPublic)
				break WAIT_LOOP

			// 4. Node Shutdown
			case <-ni.ctx.Done():
				return jsonErrorResponse("Context cancelled during init", nil)
			}
		}
	}

	// --- PubSub Initialization ---
	if err := setupPubSub(ni); err != nil {
		return jsonErrorResponse(fmt.Sprintf("Instance %d: Failed to create PubSub", instanceIndex), err)
	}
	logger.Debugf("[GO] ✅ Instance %d: PubSub (GossipSub) initialized.\n", instanceIndex)

	// --- Setup Notifiers and Handlers ---
	setupNotifiers(ni)
	logger.Debugf("[GO] 🔔 Instance %d: Registered network event notifier.\n", instanceIndex)

	setupDirectMessageHandler(ni)
	logger.Debugf("[GO] ✅ Instance %d: Direct message handler set up.\n", instanceIndex)

	// --- Close DHT if needed ---
	if !cfg.DHT.Keep {
		if ni.dht != nil {
			logger.Debugf("[GO]   - Instance %d: Closing DHT client...\n", instanceIndex)
			if err := ni.dht.Close(); err != nil {
				logger.Warnf("[GO] ⚠️ Instance %d: Error closing DHT: %v\n", instanceIndex, err)
			}
			ni.dht = nil
		}
	}

	// --- Get Final Addresses ---
	nodeAddresses, err := goGetNodeAddresses(ni, "")
	if err != nil {
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
	reachabilityStatus := map[bool]string{true: "Public", false: "Private"}[isPublic]
	logger.Infof("[GO] 🎉 Instance %d: Node creation complete. Reachability status: %s\n", instanceIndex, reachabilityStatus)
	success = true // Mark success to avoid cleanup in defer.
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

	ni, err := getInstance(int(instanceIndexC))
	if err != nil {
		return jsonErrorResponse("Invalid instance", err)
	}
	
	goAddrsJSON := C.GoString(addrsJSONC)
	logger.Debugf("[GO] 📞 Instance %d: Attempting to connect to peer with addresses: %s\n", ni.instanceIndex, goAddrsJSON)

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
			logger.Warnf("[GO] ⚠️ Instance %d: Skipping invalid multiaddress '%s' in list: %v\n", ni.instanceIndex, addrStrings[i], err)
			continue
		}
		// You might want to add a check here to ensure subsequent addresses are for the same peer ID
		addrInfo.Addrs = append(addrInfo.Addrs, maddr)
	}

	// Check if attempting to connect to the local node itself.
	if addrInfo.ID == ni.host.ID() {
		logger.Debugf("[GO] ℹ️ Instance %d: Attempting to connect to self (%s), skipping explicit connection.\n", ni.instanceIndex, addrInfo.ID)
		// Connecting to self is usually not necessary or meaningful in libp2p.
		// Return success, indicating the "connection" is implicitly present.
		return jsonSuccessResponse(addrInfo) // Caller frees.
	}

	// --- 1. ESTABLISH CONNECTION ---
	// Use a context with a timeout for the connection attempt to prevent blocking indefinitely.
	connCtx, cancel := context.WithTimeout(ni.ctx, 30*time.Second) // 30-second timeout.
	defer cancel()                                                      // Ensure context is cancelled eventually.

	// Add the peer's address(es) to the local peerstore for this instance. This helps libp2p find the peer.
	// ConnectedAddrTTL suggests the address is likely valid for a short time after connection.
	// Use PermanentAddrTTL if the address is known to be stable.
	ni.host.Peerstore().AddAddrs(addrInfo.ID, addrInfo.Addrs, peerstore.ConnectedAddrTTL)

	// Initiate the connection attempt. libp2p will handle dialing and negotiation.
	logger.Debugf("[GO]   - Instance %d: Attempting host.Connect to %s...\n", ni.instanceIndex, addrInfo.ID)
	if err := ni.host.Connect(connCtx, *addrInfo); err != nil {
		// Check if the error was due to the connection timeout.
		if connCtx.Err() == context.DeadlineExceeded {
			errMsg := fmt.Sprintf("Instance %d: Connection attempt to %s timed out after 30s", ni.instanceIndex, addrInfo.ID)
			logger.Errorf("[GO] ❌ %s\n", errMsg)
			return jsonErrorResponse(errMsg, nil) // Return specific timeout error (caller frees).
		}
		// Handle other connection errors.
		errMsg := fmt.Sprintf("Instance %d: Failed to connect to peer %s", ni.instanceIndex, addrInfo.ID)
		// Example: Check for specific common errors if needed
		// if strings.Contains(err.Error(), "no route to host") { ... }
		return jsonErrorResponse(errMsg, err) // Return generic connection error (caller frees).
	}

	// --- 2. FIND THE WINNING ADDRESS ---
	// After a successful connection, query the host's network for active connections to the peer.
	// This is where you find the 'winning' address.
	conns := ni.host.Network().ConnsToPeer(addrInfo.ID)
	var winningAddr string
	if len(conns) > 0 {
		winningAddr = fmt.Sprintf("%s/p2p/%s", conns[0].RemoteMultiaddr().String(), addrInfo.ID.String())
		logger.Debugf("[GO] ✅ Instance %d: Successfully connected to peer %s via: %s\n", ni.instanceIndex, addrInfo.ID, winningAddr)
	} else {
		logger.Warnf("[GO] ⚠️ Instance %d: Connect succeeded for %s, but no active connection found immediately. It may be pending.\n", ni.instanceIndex, addrInfo.ID)
	}

	// Success: log the successful connection and return the response.
	logger.Infof("[GO] ✅ Instance %d: Successfully initiated connection to multiaddress: %s\n", ni.instanceIndex, winningAddr)
	winningAddrInfo, err := peer.AddrInfoFromString(winningAddr)
	if err != nil {
		return jsonErrorResponse("Invalid winner multiaddress.", err)
	}
	return jsonSuccessResponse(winningAddrInfo) // Caller frees.
}

// StartStaticRelay configures and starts the AutoRelay service using a specific
// static relay (e.g., the subnetwork owner). This replaces manual reservation logic.
//
// Parameters:
//   - instanceIndexC: The node instance.
//   - relayAddrInfoJSONC: JSON string of the relay's AddrInfo (id + addrs).
//
//export StartStaticRelay
func StartStaticRelay(
	instanceIndexC C.int,
	relayAddrInfoJSONC *C.char,
) *C.char {

	ni, err := getInstance(int(instanceIndexC))
	if err != nil {
		return jsonErrorResponse("Invalid instance", err)
	}

	// --- 1. Handle Switching Subnetworks ---
	// If an AutoRelay service is already running, close it first.
	if ni.privateRelay != nil {
		logger.Debugf("[GO] 🔄 Instance %d: Switching Relay. Closing existing AutoRelay service...", ni.instanceIndex)
		if err := ni.privateRelay.Close(); err != nil { //
			logger.Warnf("[GO] ⚠️ Instance %d: Error closing old AutoRelay: %v", ni.instanceIndex, err)
		}
		ni.privateRelay = nil
		// // Also clean up any existing relayed addresses
		// ni.addrMutex.Lock()
		// ni.privateRelayAddrs = nil
		// ni.addrMutex.Unlock()
		logger.Infof("[GO] 🔄 Instance %d: Previous AutoRelay service closed.", ni.instanceIndex)
	}

	// --- 2. Parse the New Relay's AddrInfo ---
	relayInfoJSON := C.GoString(relayAddrInfoJSONC)
	var relayInfo peer.AddrInfo
	if err := json.Unmarshal([]byte(relayInfoJSON), &relayInfo); err != nil {
		return jsonErrorResponse("Failed to parse relay AddrInfo JSON", err)
	}

	logger.Debugf("[GO] 🔗 Instance %d: Configuring Static AutoRelay with peer %s", ni.instanceIndex, relayInfo.ID)

	// --- 3. Configure AutoRelay Options ---
	opts := []autorelay.Option{
		autorelay.WithStaticRelays([]peer.AddrInfo{relayInfo}),
		autorelay.WithNumRelays(1), 
		autorelay.WithBootDelay(0), 
	}

	// --- 4. Create the AutoRelay Service ---
	// This initializes the service but might not start the background workers yet.
	ar, err := autorelay.NewAutoRelay(ni.host, opts...) //
	if err != nil {
		return jsonErrorResponse("Failed to create AutoRelay service", err)
	}

	// --- 5. Start the Service ---
	// This kicks off the background goroutines to connect and reserve slots.
    // It returns immediately.
	ar.Start() 

	// Store the reference so we can close it later.
	ni.privateRelay = ar

	logger.Infof("[GO] ✅ Instance %d: Static AutoRelay service started. Target: %s", ni.instanceIndex, relayInfo.ID)

	return jsonSuccessResponse("Static AutoRelay enabled")
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

	ni, err := getInstance(int(instanceIndexC))
	if err != nil {
		return jsonErrorResponse("Invalid instance", err)
	}

	goPeerID := C.GoString(peerIDC)
	logger.Debugf("[GO] 🔌 Instance %d: Attempting to disconnect from peer: %s\n", ni.instanceIndex, goPeerID)

	pid, err := peer.Decode(goPeerID)
	if err != nil {
		return jsonErrorResponse(
			fmt.Sprintf("Instance %d: Failed to decode peer ID", ni.instanceIndex), err,
		)
	}

	if pid == ni.host.ID() {
		logger.Debugf("[GO] ℹ️ Instance %d: Attempting to disconnect from self (%s), skipping.\n", ni.instanceIndex, pid)
		return jsonSuccessResponse("Cannot disconnect from self")
	}

	// --- Close Persistent Outgoing Stream (if exists) for this instance ---
	ni.streamsMutex.Lock()
	stream, exists := ni.persistentChatStreams[pid]
	if exists {
		logger.Debugf("[GO]   ↳ Instance %d: Closing persistent outgoing stream to %s\n", ni.instanceIndex, pid)
		_ = stream.Close() // Attempt graceful close
		delete(ni.persistentChatStreams, pid)
	}
	ni.streamsMutex.Unlock() // Unlock before potentially blocking network call

	// --- Close Network Connections ---
	conns := ni.host.Network().ConnsToPeer(pid)
	closedNetworkConn := false
	if len(conns) > 0 {
		logger.Debugf("[GO]   - Instance %d: Closing %d active network connection(s) to peer %s...\n", ni.instanceIndex, len(conns), pid)
		err = ni.host.Network().ClosePeer(pid) // This closes the underlying connection(s)
		if err != nil {
			logger.Warnf("[GO] ⚠️ Instance %d: Error closing network connection(s) to peer %s: %v (proceeding with cleanup)\n", ni.instanceIndex, pid, err)
		} else {
			logger.Debugf("[GO]   - Instance %d: Closed network connection(s) to peer: %s\n", ni.instanceIndex, pid)
			closedNetworkConn = true
		}
	} else {
		logger.Debugf("[GO] ℹ️ Instance %d: No active network connections found to peer %s.\n", ni.instanceIndex, pid)
	}

	// --- Remove from Tracking Map for this instance ---
	ni.peersMutex.Lock()
	delete(ni.friendlyPeers, pid)
	ni.peersMutex.Unlock()

	logMsg := fmt.Sprintf("Instance %d: Disconnected from peer %s", ni.instanceIndex, goPeerID)
	if !exists && !closedNetworkConn && len(conns) == 0 {
		logMsg = fmt.Sprintf("Instance %d: Disconnected from peer %s (not connected or tracked)", ni.instanceIndex, goPeerID)
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

	ni, err := getInstance(int(instanceIndexC))
	if err != nil {
		// If getInstance errors, it means the host isn't ready.
		// Return success with an empty list, as it's a query, not an operation.
		logger.Warnf("[GO] ⚠️ Instance %d: GetConnectedPeers called but instance is not ready: %v\n", ni.instanceIndex, err)
		return jsonSuccessResponse([]ExtendedPeerInfo{})
	}

	// Use a Write Lock for the entire critical section to avoid mixing RLock and Lock.
	ni.peersMutex.RLock()
	defer ni.peersMutex.RUnlock() // Ensure lock is released.

	// Create a slice to hold the results directly from the map.
	peersList := make([]ExtendedPeerInfo, 0, len(ni.friendlyPeers))
	
	for _, peerInfo := range ni.friendlyPeers {
			peersList = append(peersList, peerInfo)
		}

	logger.Debugf("[GO] ℹ️ Instance %d: Reporting %d currently tracked and active peers.\n", ni.instanceIndex, len(peersList))

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

	ni, err := getInstance(int(instanceIndexC))
	if err != nil {
		// If instance isn't ready, we definitely don't have rendezvous peers.
		return C.CString(`{"state":"Empty"}`)
	}

	ni.rendezvousMutex.RLock()
	rendezvousState := ni.rendezvousState
	ni.rendezvousMutex.RUnlock()

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
	logger.Debugf("[GO] ℹ️ Instance %d: Reporting %d rendezvous peers (UpdateCount: %d).\n", ni.instanceIndex, len(rendezvousState.Peers), rendezvousState.UpdateCount)
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

	ni, err := getInstance(int(instanceIndexC))
	if err != nil {
		return jsonErrorResponse("Invalid instance", err)
	}
	peerIDStr := C.GoString(peerIDC) // Raw string from C
	
	var pidForInternalCall peer.ID // This will be peer.ID("") for local

	if peerIDStr == "" || peerIDStr == ni.host.ID().String() {
		// Convention: Empty peer.ID ("") passed to goGetNodeAddresses means "local node".
		pidForInternalCall = "" // This is peer.ID("")
	} else {
		pidForInternalCall, err = peer.Decode(peerIDStr)
		if err != nil {
			errMsg := fmt.Sprintf("Instance %d: Failed to decode peer ID '%s'", ni.instanceIndex, peerIDStr)
			return jsonErrorResponse(errMsg, err)
		}
	}

	// Call the internal Go function with the resolved peer.ID or empty peer.ID for local
	addresses, err := goGetNodeAddresses(ni, pidForInternalCall)
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

	ni, err := getInstance(int(instanceIndexC))
	if err != nil {
		return jsonErrorResponse("Invalid instance", err)
	}

	// Convert C inputs
	goChannel := C.GoString(channelC)
	goData := C.GoBytes(unsafe.Pointer(dataC), C.int(lengthC))
	
	// --- Branch: Broadcast or Direct Send ---
	if strings.Contains(goChannel, "::ps:") {
		// --- Broadcast via specific PubSub Topic ---
		instancePubsub := ni.pubsub
		if instancePubsub == nil {
			// PubSub not initialized, cannot broadcast
			return jsonErrorResponse("PubSub not initialized, cannot broadcast", nil)
		}

		ni.pubsubMutex.Lock()
		topic, exists := ni.topics[goChannel]
		if !exists {
			var err error
			logger.Debugf("[GO]   - Instance %d: Joining PubSub topic '%s' for sending.\n", ni.instanceIndex, goChannel)
			topic, err = instancePubsub.Join(goChannel) // ps is instancePubsub
			if err != nil {
				ni.pubsubMutex.Unlock()
				// Failed to join PubSub topic
				return jsonErrorResponse(fmt.Sprintf("Failed to join PubSub topic '%s'", goChannel), err)
			}
			ni.topics[goChannel] = topic
			logger.Debugf("[GO] ✅ Instance %d: Joined PubSub topic: %s for publishing.\n", ni.instanceIndex, goChannel)
		}
		ni.pubsubMutex.Unlock()

		// Directly publish the raw Protobuf payload.
		if err := topic.Publish(ni.ctx, goData); err != nil {
			// Failed to publish to topic
			return jsonErrorResponse(fmt.Sprintf("Failed to publish to topic '%s'", goChannel), err)
		}
		logger.Infof("[GO] 🌍 Instance %d: Broadcast to topic '%s' (%d bytes)\n", ni.instanceIndex, goChannel, len(goData))
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

		if pid == ni.host.ID() {
			// Attempt to send direct message to self
			return jsonErrorResponse("Attempt to send direct message to self is invalid", nil)
		}

		ni.streamsMutex.Lock()
		stream, streamExists := ni.persistentChatStreams[pid]
		ni.streamsMutex.Unlock()

		// If stream exists, try writing to it
		if streamExists {
			logger.Debugf("[GO]   ↳ Instance %d: Reusing stream %s to %s\n", ni.instanceIndex, stream.ID(), pid)
			err = writeDirectMessageFrame(stream, goChannel, goData)
			if err == nil {
				logger.Infof("[GO] 📤 Instance %d: Sent to %s via Stream %s (Reused)\n", ni.instanceIndex, pid, stream.ID())
				return jsonSuccessResponse(fmt.Sprintf("Direct message sent to %s (reused stream).", pid))
			}
			
			// Write failed? Now we lock to remove the broken stream.
			logger.Warnf("[GO] ⚠️ Instance %d: Write failed on Stream %s to %s: %v. Removing.\n", ni.instanceIndex, stream.ID(), pid, err)
			ni.streamsMutex.Lock()
			// Check if the stream in the map is still the broken one before deleting
			if s, ok := ni.persistentChatStreams[pid]; ok && s == stream {
				delete(ni.persistentChatStreams, pid)
			}
			ni.streamsMutex.Unlock()
			_ = stream.Close() // Close the broken stream
			return jsonErrorResponse(fmt.Sprintf("Failed to write to stream %s (closed).", pid), err)
		} else {
			// Stream does not exist, need to create a new one
			logger.Debugf("[GO]   ↳ Instance %d: Creating NEW stream to %s...\n", ni.instanceIndex, pid)
			streamCtx, cancel := context.WithTimeout(ni.ctx, 20*time.Second)
			defer cancel()

			newStream, err := ni.host.NewStream(
				network.WithAllowLimitedConn(streamCtx, UnaiverseChatProtocol),
				pid,
				UnaiverseChatProtocol,
			)

			if err != nil {
				return jsonErrorResponse(fmt.Sprintf("Failed to open new stream to %s.", pid), err)
			}

			// --- RACE CONDITION HANDLING ---
			// Double-check if another goroutine created a stream while we were unlocked
			ni.streamsMutex.Lock()
			existingStream, existsNow := ni.persistentChatStreams[pid]
			if existsNow {
				logger.Warnf("[GO] ⚠️ Instance %d: Race detected. Using existing stream %s, closing our new %s.\n", ni.instanceIndex, existingStream.ID(), newStream.ID())
				_ = newStream.Close() // Close the redundant stream we just created.
				stream = existingStream
			} else {
				logger.Debugf("[GO] ✅ Instance %d: Opened and stored new persistent stream %s to %s\n", ni.instanceIndex, newStream.ID(), pid)
				ni.persistentChatStreams[pid] = newStream
				stream = newStream
				go handleStream(ni, newStream)
			}
			ni.streamsMutex.Unlock()

			// --- Write message to the determined stream ---
			err = writeDirectMessageFrame(stream, goChannel, goData)
			if err != nil {
				logger.Errorf("[GO] ❌ Instance %d: Write failed on NEW stream %s to %s: %v.\n", ni.instanceIndex, stream.ID(), pid, err)
				_ = stream.Close()
				ni.streamsMutex.Lock()
				if s, ok := ni.persistentChatStreams[pid]; ok && s == stream {
					delete(ni.persistentChatStreams, pid)
				}
				ni.streamsMutex.Unlock()
				return jsonErrorResponse(fmt.Sprintf("Failed to write to new stream to '%s' (needs reconnect).", pid), err)
			}

			logger.Infof("[GO] 📤 Instance %d: Sent to %s via Stream %s (New)\n", ni.instanceIndex, pid, stream.ID())
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

	ni, err := getInstance(int(instanceIndexC))
	if err != nil {
		return jsonErrorResponse("Invalid instance", err)
	}

	// Convert C string input to Go string.
	channel := C.GoString(channelC)
	logger.Debugf("[GO] <sub> Instance %d: Attempting to subscribe to topic: %s\n", ni.instanceIndex, channel)
	
	// Get instance-specific state and mutex
	instancePubsub := ni.pubsub
	if ni.host == nil || instancePubsub == nil {
		return jsonErrorResponse(
			fmt.Sprintf("Instance %d: Host or PubSub not initialized", ni.instanceIndex), nil,
		)
	}

	// Lock the mutex for safe access to the shared topics and subscriptions maps for this instance.
	ni.pubsubMutex.Lock()
	defer ni.pubsubMutex.Unlock() // Ensure mutex is unlocked when function returns.

	// Check if already subscribed to this topic for this instance.
	if _, exists := ni.subscriptions[channel]; exists {
		logger.Debugf("[GO] <sub> Instance %d: Already subscribed to topic: %s\n", ni.instanceIndex, channel)
		// Return success, indicating the desired state is already met.
		return jsonSuccessResponse(
			fmt.Sprintf("Instance %d: Already subscribed to topic %s", ni.instanceIndex, channel),
		) // Caller frees.
	}

	// If the channel ends with ":rv", it indicates a rendezvous topic, so we remove other ones
	// from the instanceTopics and instanceSubscriptions list, and we clean the rendezvousDiscoveredPeersInstances.
	if strings.HasSuffix(channel, ":rv") {
		logger.Debugf("  - Instance %d: Joining rendezvous topic '%s'. Cleaning up previous rendezvous state.\n", ni.instanceIndex, channel)
		// Remove all existing rendezvous topics and subscriptions for this instance.
		for existingChannel := range ni.topics {
			if strings.HasSuffix(existingChannel, ":rv") {
				logger.Debugf("  - Instance %d: Removing existing rendezvous topic '%s' from instance state.\n", ni.instanceIndex, existingChannel)

				// Close the topic handle if it exists.
				if topic, exists := ni.topics[existingChannel]; exists {
					if err := topic.Close(); err != nil {
						logger.Warnf("⚠️ Instance %d: Error closing topic handle for '%s': %v (proceeding with map cleanup)\n", ni.instanceIndex, existingChannel, err)
					}
					delete(ni.topics, existingChannel)
				}

				// Remove the subscription if it exists.
				if sub, exists := ni.subscriptions[existingChannel]; exists {
					sub.Cancel()                                   // Cancel the subscription
					delete(ni.subscriptions, existingChannel) // Remove from map
				}

				// Also clean up rendezvous discovered peers for this instance.
				logger.Debugf("  - Instance %d: Resetting rendezvous state for new topic '%s'.\n", ni.instanceIndex, channel)
				ni.rendezvousMutex.Lock()
				ni.rendezvousState = nil
				ni.rendezvousMutex.Unlock()
			}
		}
		logger.Debugf("  - Instance %d: Cleaned up previous rendezvous state.\n", ni.instanceIndex)
	}

	// --- Join the Topic ---
	// Get a handle for the topic. `Join` creates the topic if it doesn't exist locally
	// and returns a handle. It's safe to call Join multiple times; it's idempotent.
	// We store the handle primarily for potential future publishing from this node.
	topic, err := instancePubsub.Join(channel)
	if err != nil {
		errMsg := fmt.Sprintf("Instance %d: Failed to join topic '%s'", ni.instanceIndex, channel)
		return jsonErrorResponse(errMsg, err) // Caller frees.
	}
	// Store the topic handle in the map for this instance.
	ni.topics[channel] = topic
	logger.Debugf("[GO]   - Instance %d: Obtained topic handle for: %s\n", ni.instanceIndex, channel)

	// --- Subscribe to the Topic ---
	// Create an actual subscription to receive messages from the topic.
	sub, err := topic.Subscribe()
	if err != nil {
		// Close the newly created topic handle.
		err := topic.Close()
		if err != nil {
			// Log error but proceed with cleanup.
			logger.Warnf("[GO] ⚠️ Instance %d: Error closing topic handle for '%s': %v (proceeding with map cleanup)\n", ni.instanceIndex, channel, err)
		}
		// Remove the topic handle from our local map for this instance.
		delete(ni.topics, channel)
		errMsg := fmt.Sprintf("Instance %d: Failed to subscribe to topic '%s' after joining", ni.instanceIndex, channel)
		return jsonErrorResponse(errMsg, err) // Caller frees.
	}
	// Store the subscription object in the map for this instance.
	ni.subscriptions[channel] = sub
	logger.Debugf("[GO]   - Instance %d: Created subscription object for: %s\n", ni.instanceIndex, channel)

	// --- Start Listener Goroutine ---
	// Launch a background goroutine that will continuously read messages
	// from this new subscription and add them to the message buffer for this instance.
	// Pass the instance index, subscription object, and topic name (for logging).
	go readFromSubscription(ni, sub)

	logger.Debugf("[GO] ✅ Instance %d: Subscribed successfully to topic: %s and started listener.\n", ni.instanceIndex, channel)
	return jsonSuccessResponse(
		fmt.Sprintf("Instance %d: Subscribed to topic %s", ni.instanceIndex, channel),
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

	ni, err := getInstance(int(instanceIndexC))
	if err != nil {
		// If instance is already gone, we can consider it "unsubscribed"
		logger.Warnf("[GO] ⚠️ Instance %d: Unsubscribe called but instance is not ready: %v\n", ni.instanceIndex, err)
		return jsonSuccessResponse(fmt.Sprintf("Instance %d: Not subscribed (instance not running)", ni.instanceIndex))
	}

	// Convert C string input to Go string.
	channel := C.GoString(channelC)
	logger.Debugf("[GO] </sub> Instance %d: Attempting to unsubscribe from topic: %s\n", ni.instanceIndex, channel)
	
	// Lock the mutex for write access to shared maps for this instance.
	ni.pubsubMutex.Lock()
	defer ni.pubsubMutex.Unlock()

	// --- Cancel the Subscription ---
	// Find the subscription object in the map for this instance.
	sub, subExists := ni.subscriptions[channel]
	if !subExists {
		logger.Warnf("[GO] </sub> Instance %d: Not currently subscribed to topic: %s (or already unsubscribed)\n", ni.instanceIndex, channel)
		// Also remove potential stale topic handle if subscription is gone.
		delete(ni.topics, channel)
		return jsonSuccessResponse(
			fmt.Sprintf("Instance %d: Not currently subscribed to topic %s", ni.instanceIndex, channel),
		) // Caller frees.
	}

	// Cancel the subscription. This signals the associated `readFromSubscription` goroutine
	// (waiting on `sub.Next()`) to stop by causing `sub.Next()` to return an error (usually `ErrSubscriptionCancelled`).
	// It also cleans up internal PubSub resources related to this subscription.
	sub.Cancel()
	// Remove the subscription entry from our local map for this instance.
	delete(ni.subscriptions, channel)
	logger.Debugf("[GO]   - Instance %d: Cancelled subscription object for topic: %s\n", ni.instanceIndex, channel)

	// --- Close the Topic Handle ---
	// Find the corresponding topic handle for this instance. It's good practice to close this as well,
	// although PubSub might manage its lifecycle internally based on subscriptions.
	// Explicit closing ensures resources related to the *handle* (like internal routing state) are released.
	topic, topicExists := ni.topics[channel]
	if topicExists {
		logger.Debugf("[GO]   - Instance %d: Closing topic handle for: %s\n", ni.instanceIndex, channel)
		// Close the topic handle.
		err := topic.Close()
		if err != nil {
			// Log error but proceed with cleanup.
			logger.Warnf("[GO] ⚠️ Instance %d: Error closing topic handle for '%s': %v (proceeding with map cleanup)\n", ni.instanceIndex, channel, err)
		}
		// Remove the topic handle from our local map for this instance.
		delete(ni.topics, channel)
		logger.Debugf("[GO]   - Instance %d: Removed topic handle from local map for topic: %s\n", ni.instanceIndex, channel)
	} else {
		logger.Debugf("[GO]   - Instance %d: No topic handle found in local map for '%s' to close (already removed or possibly never stored?).\n", ni.instanceIndex, channel)
		// Ensure removal from map even if handle wasn't found (e.g., inconsistent state).
		delete(ni.topics, channel)
	}

	// If the channel ends with ":rv", it indicates a rendezvous topic, so we have closed the topic and the sub
	// but we also need to clean the rendezvousDiscoveredPeersInstances.
	if strings.HasSuffix(channel, ":rv") {
		logger.Debugf("  - Instance %d: Unsubscribing from rendezvous topic. Clearing state.\n", ni.instanceIndex)
		ni.rendezvousMutex.Lock()
		ni.rendezvousState = nil
		ni.rendezvousMutex.Unlock()
	}
	logger.Debugf("[GO]   - Instance %d: Cleaned up previous rendezvous state.\n", ni.instanceIndex)

	logger.Infof("[GO] ✅ Instance %d: Unsubscribed successfully from topic: %s\n", ni.instanceIndex, channel)
	return jsonSuccessResponse(
		fmt.Sprintf("Instance %d: Unsubscribed from topic %s", ni.instanceIndex, channel),
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

	ni, err := getInstance(int(instanceIndexC))
	if err != nil {
		logger.Errorf("[GO] ❌ MessageQueueLength: %v\n", err)
		return -1 // Return -1 if instance isn't valid
	}

	// Get the message store for this instance
	store := ni.messageStore
	if store == nil {
		logger.Errorf("[GO] ❌ Instance %d: Message store not initialized.\n", ni.instanceIndex)
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

	ni, err := getInstance(int(instanceIndexC))
	if err != nil {
		return jsonErrorResponse("Invalid instance", err)
	}

	// Get the message store for this instance
	store := ni.messageStore
	if store == nil {
		logger.Errorf("[GO] ❌ Instance %d: PopMessages: Message store not initialized.\n", ni.instanceIndex)
		return jsonErrorResponse(fmt.Sprintf("Instance %d: Message store not initialized", ni.instanceIndex), nil)
	}

	store.mu.Lock() // Lock for the entire operation
	defer store.mu.Unlock()

	if len(store.messagesByChannel) == 0 {
		return C.CString(`{"state":"Empty"}`)
	}

	// Create a slice to hold the popped messages. Capacity is the number of channels.
	var poppedMessages []*QueuedMessage
	for channel, messageList := range store.messagesByChannel {
		if messageList.Len() > 0 {
			element := messageList.Front()
			msg := element.Value.(*QueuedMessage)
			poppedMessages = append(poppedMessages, msg)
			messageList.Remove(element)
		}
		// if the queue is now empty, we can delete it from the map to save space
		if messageList.Len() == 0 {
			delete(store.messagesByChannel, channel)
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
		logger.Errorf("[GO] ❌ Instance %d: PopMessages: Failed to marshal messages to JSON: %v\n", ni.instanceIndex, err)
		// Messages have already been popped from the queue at this point.
		// Returning an error is the best we can do.
		return jsonErrorResponse(
			fmt.Sprintf("Instance %d: Failed to marshal popped messages", ni.instanceIndex), err,
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

		// acquire the global lock
		globalInstanceMutex.Lock()
		defer globalInstanceMutex.Unlock()

		for i, ni := range allInstances {
			if ni != nil {
				logger.Debugf("[GO] 🛑 Attempting to close instance %d...\n", i)
				
				err := ni.Close() // Call the new method
				allInstances[i] = nil // Remove from slice
				
				if err != nil {
					errorCount++
					errorMessages = append(errorMessages, fmt.Sprintf("Instance %d: %v", i, err))
					logger.Errorf("[GO] ❌ Instance %d: Close failed: %v\n", i, err)
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
		if instanceIndex < 0 || instanceIndex >= maxInstances {
			err := fmt.Errorf("invalid instance index: %d. Must be between 0 and %d", instanceIndex, maxInstances-1)
			return jsonErrorResponse("Invalid instance index for single close", err) // Caller frees.
		}

		globalInstanceMutex.Lock()
		defer globalInstanceMutex.Unlock()

		instance := allInstances[instanceIndex]
		if instance == nil {
			logger.Debugf("[GO] ℹ️ Instance %d: Node was already closed.\n", instanceIndex)
			return jsonSuccessResponse(fmt.Sprintf("Instance %d: Node was already closed", instanceIndex))
		}

		err := instance.Close()
		allInstances[instanceIndex] = nil

		if err != nil {
			return jsonErrorResponse(fmt.Sprintf("Instance %d: Error closing host", instanceIndex), err)
		}

		logger.Infof("[GO] 🛑 Instance %d: Node closed successfully.\n", instanceIndex)
		return jsonSuccessResponse(fmt.Sprintf("Instance %d: Node closed successfully", instanceIndex))
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

// main is the entry point for a Go executable.
func main() {
	// This message will typically only be seen if you run `go run lib.go`
	// or build and run as a standard executable, NOT when used as a shared library.
	logger.Debugf("[GO] libp2p Go library main function (not executed in c-shared library mode)")
}
