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
	pwebrtc "github.com/pion/webrtc/v4"
	p2pforge "github.com/ipshipyard/p2p-forge/client"
)

// ChatProtocol defines the protocol ID string used for direct peer-to-peer messaging streams.
// This ensures that both peers understand how to interpret the data on the stream.
// const UnaiverseChatProtocol = "/unaiverse-chat-protocol/1.0.0"
const UnaiverseChatProtocol = "/unaiverse/chat/1.0.0"
const UnaiverseUserAgent = "go-libp2p/example/autotls"
const DisconnectionGracePeriod = 10 * time.Second

// WebRTC signaling constants
const UnaiverseWebRTCSignalProtocol = "/unaiverse/webrtc-signal/1.0.0"
const WebRTCDataChannelLabel = "unaiverse-data"
// Total budget for the entire signaling handshake (offer → answer → DC open).
const WebRTCSignalingTimeout = 60 * time.Second

// WebRTCDCUtRWaitGoPeer is how long a Go node waits for native libp2p DCUtR/hole-punching
// before falling back to our custom WebRTC signaling, when the remote is also a Go peer
// (advertises only /webrtc-direct). Go↔Go DCUtR can succeed, so we give it time.
const WebRTCDCUtRWaitGoPeer = 20 * time.Second

// WebRTCDCUtRWaitJSPeer is the short wait used when the remote is a JS/browser peer
// (advertises /webrtc multiaddrs). Go↔JS DCUtR can never succeed (incompatible transports),
// so we skip straight to our custom signaling after a minimal delay.
const WebRTCDCUtRWaitJSPeer = 3 * time.Second

// PeerConnectionTimeout is the timeout for a single outbound connection attempt to a peer.
const PeerConnectionTimeout = 30 * time.Second

// RelayReservationTimeout is the timeout for reserving a slot on a relay node.
const RelayReservationTimeout = 60 * time.Second

// StreamCreationTimeout is the timeout for opening a new libp2p stream to a peer.
const StreamCreationTimeout = 20 * time.Second

// WebRTC DataChannel chunking constants.
// Browser SCTP implementations cap individual DataChannel messages at ~64KB.
// We split large frames into chunks of at most WebRTCMaxChunkPayload bytes
// and prefix each chunk with a 1-byte flags header.
//
// Chunk wire format: [1-byte flags][payload bytes]
//
// Flag bits:
//
//	Bit 7 (0x80): START – this is the first chunk of a message
//	Bit 6 (0x40): END   – this is the last  chunk of a message
//
// Valid combinations:
//
//	0xC0  START+END  – complete message in a single chunk (fast path)
//	0x80  START      – first chunk of a multi-chunk message
//	0x00  (none)     – middle continuation chunk
//	0x40  END        – final chunk of a multi-chunk message
const (
	webRTCChunkFlagStart    byte = 0x80
	webRTCChunkFlagEnd      byte = 0x40
	webRTCChunkFlagStartEnd byte = 0xC0 // START | END

	// WebRTCMaxChunkSize is the maximum size of a single dc.Send() call
	// (header byte included). Stays well under the 64 KB browser SCTP limit.
	WebRTCMaxChunkSize = 60 * 1024

	// webRTCMaxChunkPayload is the maximum payload bytes per chunk
	// (total chunk size minus the 1-byte flags header).
	webRTCMaxChunkPayload = WebRTCMaxChunkSize - 1
)

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

	// WebRTC signaling (application-level, not a libp2p transport)
	WebRTC struct {
		// Enabled activates the /unaiverse/webrtc-signal/1.0.0 protocol handler.
		Enabled bool `json:"enabled"`
		// ICEConfig overrides the default STUN/TURN server list. If nil, Google
		// public STUN servers are used.
		ICEConfig *ICEConfig `json:"ice_config,omitempty"`
	} `json:"webrtc"`
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

	// WebRTC DataChannel connections (application-level, keyed by remote peer ID)
	webrtcMutex       sync.RWMutex
	webrtcConnections map[peer.ID]*WebRTCConn
	iceConfig         *ICEConfig // nil → use defaults (Google STUN)

	// a copy of its own index for logging
	instanceIndex int
}

// ICEConfig holds STUN/TURN server configuration for WebRTC.
type ICEConfig struct {
	STUNServers []string     `json:"stun_servers"`
	TURNServers []TURNServer `json:"turn_servers"`
}

// TURNServer holds credentials for a TURN relay server.
type TURNServer struct {
	URLs       []string `json:"urls"`
	Username   string   `json:"username"`
	Credential string   `json:"credential"`
}

// SignalMessage is a single JSON-encoded signaling message carried over the
// "/unaiverse/webrtc-signal/1.0.0" stream.
//
// Wire format (per message):
//
//	[4-byte big-endian uint32: JSON length][JSON payload]
type SignalMessage struct {
	Type    string `json:"type"`              // "offer" | "answer" | "error"
	SDP     string `json:"sdp,omitempty"`     // full SDP (offer/answer; includes all ICE candidates)
	Message string `json:"message,omitempty"` // human-readable error detail
}

// WebRTCConn holds the live state of an established WebRTC DataChannel connection
// to a single remote peer.
type WebRTCConn struct {
	pc         *pwebrtc.PeerConnection
	dc         *pwebrtc.DataChannel
	remotePeer peer.ID

	// sendMu serializes the chunked-send loop so that concurrent callers cannot
	// interleave their chunks on the wire (which would corrupt reassembly).
	sendMu sync.Mutex

	// reassemblyBuf accumulates inbound chunks between a START and an END chunk.
	// Pion delivers OnMessage callbacks sequentially for a single DataChannel,
	// so no mutex is needed here.
	reassemblyBuf []byte
}

// Define the list of default STUN servers to use if none are provided in the config.
var defaultSTUNServers = []string{
	"stun:stun.l.google.com:19302",
	"stun:global.stun.twilio.com:3478",
	"stun:stun.cloudflare.com:3478",
	"stun:stun.services.mozilla.com:3478",
}

// Define a list of curated public relay nodes to use as static relays for all nodes in the network.
var defaultRelays = []string{
	// Relay 1
	"/ip4/193.205.7.181/tcp/30060/p2p/12D3KooWHT6ZzSeT7ZT85xRZo5ZMC6gcz7j3QuCQwEPHRqHJWVQK",
	"/ip4/193.205.7.181/udp/30061/quic-v1/p2p/12D3KooWHT6ZzSeT7ZT85xRZo5ZMC6gcz7j3QuCQwEPHRqHJWVQK",
	"/ip4/193.205.7.181/udp/30062/quic-v1/webtransport/certhash/uEiC5qpw6oFVU3bdo8zKpA6qs_l-poteC3DYULmiIT8ByFA/certhash/uEiAsazIfQKQXPyBwK-HaIzmjLv8McEUJ2FZrOp3hqU3MOw/p2p/12D3KooWHT6ZzSeT7ZT85xRZo5ZMC6gcz7j3QuCQwEPHRqHJWVQK",
	"/dns4/multaiverse.diism.unisi.it/tcp/30060/tls/ws/p2p/12D3KooWHT6ZzSeT7ZT85xRZo5ZMC6gcz7j3QuCQwEPHRqHJWVQK",
	"/ip4/193.205.7.181/udp/30063/webrtc-direct/certhash/uEiALPmyX41SN9z78dtVFDgO0JkfT-hFgZmTqp-DaFVbkiQ/p2p/12D3KooWHT6ZzSeT7ZT85xRZo5ZMC6gcz7j3QuCQwEPHRqHJWVQK",
	// Relay 2
	"/ip4/193.205.7.181/udp/30071/quic-v1/p2p/12D3KooWGatNmkeBMaKEiGcYoX6eBxYvpjQm7EWJWPPusbzieC1v",
	"/ip4/193.205.7.181/udp/30072/quic-v1/webtransport/certhash/uEiBZMYxlAyA2I0AbkhCVdiq4nRpeG7v5FKQbdJQOFOtjCw/certhash/uEiAYsv1fcD6CM5ZU1EYRTMlnv-Rt_tpFiPK9_DoPnsgSUA/p2p/12D3KooWGatNmkeBMaKEiGcYoX6eBxYvpjQm7EWJWPPusbzieC1v",
	"/ip4/193.205.7.181/tcp/30070/p2p/12D3KooWGatNmkeBMaKEiGcYoX6eBxYvpjQm7EWJWPPusbzieC1v",
	"/ip4/193.205.7.181/udp/30073/webrtc-direct/certhash/uEiDsrIXLsgRTnum45qJgM2cEnRjaLnKE9rqayk4YlPx88w/p2p/12D3KooWGatNmkeBMaKEiGcYoX6eBxYvpjQm7EWJWPPusbzieC1v",
	"/dns4/multaiverse.diism.unisi.it/tcp/30070/tls/ws/p2p/12D3KooWGatNmkeBMaKEiGcYoX6eBxYvpjQm7EWJWPPusbzieC1v",
}
