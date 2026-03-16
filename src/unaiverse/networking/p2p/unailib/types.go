// Here we define types, structs and constants for use in the P2P library.
package main

import (
	"sync"
	"time"
	"context"
	"container/list"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	ma "github.com/multiformats/go-multiaddr"
	dht "github.com/libp2p/go-libp2p-kad-dht"
	golog "github.com/ipfs/go-log/v2"
	pubsub "github.com/libp2p/go-libp2p-pubsub"
	p2pforge "github.com/ipshipyard/p2p-forge/client"
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
	privateRelayAddrs []ma.Multiaddr

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
