// Here we define types, structs and constants for use in the P2P library.
package main

import (
	"sync"
	"time"
	"context"
	"sync/atomic"
	"container/list"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	ma "github.com/multiformats/go-multiaddr"
	golog "github.com/ipfs/go-log/v2"
	pubsub "github.com/libp2p/go-libp2p-pubsub"
	pwebrtc "github.com/pion/webrtc/v4"
	p2pforge "github.com/ipshipyard/p2p-forge/client"
	ping "github.com/libp2p/go-libp2p/p2p/protocol/ping"
)

// ChatProtocol defines the protocol ID string used for direct peer-to-peer messaging streams.
// This ensures that both peers understand how to interpret the data on the stream.
// const UnaiverseChatProtocol = "/unaiverse-chat-protocol/1.0.0"
const UnaiverseChatProtocol = "/unaiverse/chat/1.0.0"

// UnaiverseUserAgent is how a node names itself to the service that issues the AutoTLS
// certificates, and the family name of what it tells its peers.
const UnaiverseUserAgent = "unaiverse"

// UnaiverseIdentifyAgent is how a node names itself to its peers (identify): the family
// name and the transport stack that runs it, which here is this library. The browser
// transport of the SDK names itself "unaiverse/js". A peer learns from it what kind of
// node it is talking to without reading it off the addresses the node publishes.
const UnaiverseIdentifyAgent = UnaiverseUserAgent + "/go"

// P2PForgeLoggerName is the logger of the AutoTLS certificate manager, as p2p-forge names it.
const P2PForgeLoggerName = "p2p-forge/client"
const DisconnectionGracePeriod = 10 * time.Second

// Session abstraction protocol IDs. Each stage of the Session handshake gets its own
// libp2p protocol so timeouts / cancellation / version negotiation are independent.
// One protobuf message per stream (no per-stream channel header), length-prefixed by
// a single uint32. See session/session.proto for the message types and session/session.py
// for the FSM that drives them. Keep these strings in sync with SESSION_*_PROTOCOL
// in session/session.py.
const (
	SessionHelloProtocol   = "/unaiverse/session/hello/1.0.0"
	SessionProfileProtocol = "/unaiverse/session/profile/1.0.0"
	SessionWorldProtocol   = "/unaiverse/session/world/1.0.0"
	SessionControlProtocol = "/unaiverse/session/control/1.0.0"
)

// WebRTC signaling constants
const UnaiverseWebRTCSignalProtocol = "/unaiverse/webrtc-signal/1.0.0"
const WebRTCDataChannelLabel = "unaiverse-data"
// Total budget for one signaling handshake (offer → answer → DC open): of the answerer,
// and of the offerer in the TURN stage.
const WebRTCSignalingTimeout = 60 * time.Second

// The stages of a WebRTC offer, as the browser names them in the "stage" field of its
// signalling messages. An offer of the direct stage is made with STUN alone and asks the
// answerer for the same; one of the TURN stage, like an offer that names no stage, has
// the whole ICE list on both sides. With every server in one attempt ICE takes the first
// pair that answers, and a TURN allocation is often ready before a hole is punched: a
// link that could be direct ended on TURN by a few hundred milliseconds.
const (
	WebRTCStageDirect = "direct"
	WebRTCStageTURN   = "turn"
)

// WebRTCDirectStageTimeout is the budget of the direct stage, signalling and opening of
// the DataChannel together: what the browser gives its own. A direct stage that finds a
// path does it fast, and one that finds none says so only by its time running out, with
// the peer waiting on a relay circuit: in the live round of 2026-11-18 every direct stage
// that succeeded took at most 3.3 s (40 of them) and every one that failed took the whole
// budget, 15 s then.
const WebRTCDirectStageTimeout = 5 * time.Second

// WebRTCStagePause is the pause between the end of a direct stage that failed and the
// offer of the TURN stage. A browser takes one signalling exchange per peer at a time
// and closes the stream of an offer that finds the previous one still running: the
// pause gives it the time to read our abort and let that exchange go.
const WebRTCStagePause = 750 * time.Millisecond

// WebRTCStageRetryDelay is how long the TURN stage waits before its one retry, made
// when the answerer closed the stream with no answer. When our abort did not arrive,
// a browser holds the previous exchange for 3 s after the end of its stream.
const WebRTCStageRetryDelay = 3 * time.Second

// WebRTCAbortWriteTimeout bounds the write of the abort that ends a stage: the stream
// may be gone, and the next stage waits for this write.
const WebRTCAbortWriteTimeout = time.Second

// An offer or an answer leaves when its local description holds the candidates its stage
// needs, not when the gathering ends: the SDP is not trickled, and pion gives a server
// that does not answer 5 s. WebRTCGatherQuiet is the silence after a candidate of the
// needed kind that ends the wait, the other two bound it in the direct stage and in the
// stages with the whole list. The browser has the same rule. Variables, so that a test
// can shorten them.
var (
	WebRTCGatherQuiet     = 500 * time.Millisecond
	WebRTCDirectGatherMax = 3 * time.Second
	WebRTCTURNGatherMax   = 8 * time.Second
)

// WebRTCUpgradeWait is how long the node that dialled a relay circuit waits, once the peer
// is identified, before it offers a WebRTC DataChannel: the head start of the paths that
// need no offer. Toward a Go peer that is the hole punching of the library (DCUtR), which
// connects within a second or two when it can. Toward a browser it is the direct legs the
// browser dials when it identifies us, bounded to 2 s. The wait is the same for both and
// short on purpose: the direct stage of the offer has no TURN server, so a path that
// comes after the offer was made is one more path and never a worse one. The 20 s once
// given to DCUtR between two Go nodes passed in full every time they were measured.
const WebRTCUpgradeWait = 3 * time.Second

// PeerConnectionTimeout is the timeout for a single outbound connection attempt to a peer.
const PeerConnectionTimeout = 30 * time.Second

// ProbeDialTimeout is the timeout for one boot-time dial to a reachability probe peer.
// Probe peers are project infrastructure on good links, so a dial that takes longer
// than this is effectively down and the verdict should not wait for it.
const ProbeDialTimeout = 10 * time.Second

// ProbeMaintainInterval is how often maintainProbePeers checks that at least one
// probe peer is still connected, so AutoNAT never runs out of dial-back servers.
const ProbeMaintainInterval = 2 * time.Minute

// RelayReservationTimeout is the timeout for reserving a slot on a relay node.
const RelayReservationTimeout = 60 * time.Second

// AutoRelayBackoff is how long AutoRelay leaves a relay alone after an attempt to reserve
// on it. The library counts the successful attempts too, and its default is one hour: with
// two static relays a node that lost its relay twice within the hour stayed without one
// until the hour was over. The relays are ours and static, so the backoff only spaces the
// retries toward the same relay.
const AutoRelayBackoff = time.Minute

// SessionSendTimeout bounds the send of one Session frame: the dial when there is no
// connection to the peer, the opening of the stream and the write. The Python node waits
// for every send inside its single loop, so this is how long a peer that stopped answering
// holds the whole node, at each frame sent to it. A dial through a relay takes a few tenths
// of a second.
const SessionSendTimeout = 3 * time.Second

// StreamCreationTimeout is the timeout for opening a new libp2p stream to a
// peer. It is also reused as the per-stream READ deadline for inbound session
// streams (see handleSessionStream), i.e. a 20s budget to receive one frame.
const StreamCreationTimeout = 20 * time.Second

// ChatStreamIdleTimeout bounds how long a persistent chat stream may stay silent
// before we reset it. The chat stream is long-lived and bidirectional, so we do
// not set a hard one-shot deadline (unlike session streams); instead we renew an
// idle deadline on every frame. A peer that opens the stream and then goes quiet
// (slow-loris) gets pinned no longer than this. The window is a per-frame budget,
// so it must comfortably exceed the time to transfer one MaxMessageSize frame on
// a slow link.
const ChatStreamIdleTimeout = 5 * time.Minute

// The ranks of a connection as the place of the chat stream of a peer. It is the order in
// which go-libp2p picks the connection of a new stream (swarm.isBetterConn, without its
// tie-breakers): a circuit with no limits is the one through a world host.
const (
	connRankLimited = iota
	connRankRelayed
	connRankDirect
)

// ChatStreamMoveTimeout bounds the opening of the chat stream on a better connection than
// the one it sits on. That connection is already there, so nothing is dialled, and the send
// that makes the move waits for it.
const ChatStreamMoveTimeout = 2 * time.Second

// ChatStreamMoveRetry is how long a chat stream is left where it is after a move that
// failed. Without it every send toward a peer whose better connection stopped answering
// would wait ChatStreamMoveTimeout again.
const ChatStreamMoveRetry = 30 * time.Second

// ChatStreamDrainTimeout is how long, from the last write on a chat stream that was
// replaced, the first write on its replacement waits for the old one to be read to its
// end. The two streams are read by two goroutines of the peer, so a frame written on the
// new one at once could be queued before the last ones of the old. A Go node closes a
// stream when it has read it to the end, and the wait ends there; a peer that does not
// (a browser) costs the whole of it.
const ChatStreamDrainTimeout = time.Second

// WebRTCReassemblyTimeout bounds how long a multi-chunk DataChannel message may
// take between its START and END chunks. Past this, the partial reassembly buffer
// is discarded so a peer cannot pin up to MaxMessageSize of memory by sending a
// START and then dribbling (or never finishing). Checked lazily on each incoming
// chunk; a peer that goes fully silent is reclaimed via OnConnectionStateChange.
const WebRTCReassemblyTimeout = 120 * time.Second

// WebRTCFreshnessWindow is how recent the last sign of life of a DataChannel must be for
// the channel to be called fresh: something arrived on it within the window, or it opened
// within the window. The states of the channel and of its peer connection were seen to
// say open for a minute after the peer had dropped them, and what is sent then is lost
// with no error. Session frames go on a DataChannel only while it is fresh, and our own
// offer is skipped only over a fresh one. A live channel stays fresh: each side pings at
// least every 20 s and answers every ping.
const WebRTCFreshnessWindow = 30 * time.Second

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

// popBinaryMagic is the first byte of the binary buffer returned by PopMessages
// when there is at least one message. It lets the Python side tell the binary
// message buffer apart from the JSON status responses (Empty/Error), which always
// start with '{' (0x7B). The binary layout avoids the base64 (+33%) and the extra
// string copies that JSON marshalling imposed on large image/tensor payloads:
//
//	[1B magic=0xB1][uint32 totalBytes][uint32 count]
//	then per message: [uint32 fromLen][from][uint32 protoLen][proto][uint32 dataLen][data]
//
// totalBytes is the size of the whole buffer (magic + header + all frames). All
// integers are big-endian. `from` is the textual peer-id, `data` is raw bytes.
const popBinaryMagic byte = 0xB1

// --- Create a package-level logger ---
var logger = golog.Logger("unailib")

// p2p-forge creates its logger when the certificate manager is built, after
// InitializeLibrary has set the levels: naming it here lets the log configuration reach it.
var _ = golog.Logger(P2PForgeLoggerName)

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
//
// Protocol is set to the libp2p protocol ID for session-protocol messages (one of the
// Session*Protocol constants) and is empty for chat / pubsub messages — that's how
// Python's connpool decoder decides whether to feed a message through Msg.from_bytes
// (chat) or hand it raw to SessionManager.deliver (session).
type QueuedMessage struct {
	From     peer.ID `json:"from"`     // The VERIFIED peer ID of the sender from the network layer.
	Data     []byte  `json:"-"`        // The raw data payload (Protobuf encoded).
	Protocol string  `json:"protocol"` // libp2p protocol ID or "" for chat/pubsub.
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
		FloodSub bool `json:"flood_sub"` // whether to use floodsub instead of gossipsub
    } `json:"network"`

	// Reachability probing (AutoNAT dial-back servers). ProbePeers is the list of
	// multiaddrs (with peer IDs) dialed at boot so the AutoNAT client has someone to
	// ask for dial-backs. nil/absent falls back to defaultRelays; an explicit empty
	// list disables probing (no reachability verdict unless ForcePublic).
	Reachability struct {
		ProbePeers []string `json:"probe_peers"`
	} `json:"reachability"`

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
	ctx          context.Context
	cancel       context.CancelFunc
	certManager  *p2pforge.P2PForgeCertMgr
	messageStore *MessageStore

	// Reachability probe peers (self-filtered), the AutoNAT dial-back supply.
	probePeers []peer.AddrInfo

	// Address Cache
    addrMutex  sync.RWMutex
    localAddrs []ma.Multiaddr

	// Static relay
	privateRelayAddrs []ma.Multiaddr

	// The relay this instance holds a slot on (see relay_slot.go): empty when it holds none.
	relaySlotMutex  sync.Mutex
	relaySlotPeer   peer.ID
	relaySlotExpiry time.Time
	// A request after a new connection to the relay is waiting to start.
	relaySlotAsking bool
	// One request to the relay at a time.
	relayAskMutex sync.Mutex

	// Reachability State (AutoNAT). Seeded in CreateNode; for non-ForcePublic nodes it is
	// then kept live by handleReachabilityEvents, which is the only writer from that point.
	reachMutex sync.RWMutex
	isPublic   bool

	// PubSub State
	pubsubMutex   sync.RWMutex
	topics        map[string]*pubsub.Topic
	subscriptions map[string]*pubsub.Subscription

	// Peer State
	peersMutex     sync.RWMutex
	friendlyPeers map[peer.ID]ExtendedPeerInfo

	// Stream State
	streamsMutex          sync.Mutex
	persistentChatStreams map[peer.ID]*persistentStream

	// Disconnection Grace Period State
    disconnectionMutex  sync.Mutex
    disconnectionTimers map[peer.ID]context.CancelFunc

	// Rendezvous State
	rendezvousMutex sync.RWMutex
	rendezvousState *RendezvousState

	// WebRTC DataChannel connections (application-level, keyed by remote peer ID)
	webrtcMutex       sync.RWMutex
	webrtcConnections map[peer.ID]*WebRTCConn
	// The peers an offer of ours is being made to (see initiateWebRTCConnection): one
	// run of the stages per peer at a time.
	webrtcOffering sync.Map
	iceConfigMu       sync.RWMutex  // protects iceConfig swaps
	iceConfig         *ICEConfig // nil → use defaults (Google STUN)
	// clock is what the freshness of a DataChannel and the retry of a chat stream move
	// are measured with: nil → time.Now.
	clock func() time.Time

	// libp2p ping service for Session liveness keep-alive (called from Python).
	pingService *ping.PingService

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
	Type    string `json:"type"`              // "offer" | "answer" | "error" | "abort"
	// Stage is the stage of an offer (WebRTCStage*), repeated in the abort that gives the
	// offer up. A side that does not stage its offers leaves it out, and one that does
	// not know the field ignores it.
	Stage   string `json:"stage,omitempty"`
	SDP     string `json:"sdp,omitempty"`     // full SDP (offer/answer; includes all ICE candidates)
	Message string `json:"message,omitempty"` // human-readable error detail
	// SessionFrames is set, in an offer and in an answer, by a side that accepts Session
	// frames on the DataChannel the exchange creates. A side that does not know the
	// field ignores it and never sets it.
	SessionFrames bool `json:"session_frames,omitempty"`
}

// persistentStream wraps a libp2p direct-chat stream with a per-stream write
// mutex. libp2p streams are not safe for concurrent writes; sendMu serializes
// writeDirectMessageFrame calls so concurrent senders cannot interleave frame
// bytes on the wire (mirrors WebRTCConn.sendMu for the libp2p path).
//
// Lock ordering: sendMu comes before streamsMutex. The move of the stream to a
// better connection takes streamsMutex while it holds sendMu, of the old stream
// and of the new one; nothing takes a sendMu while it holds streamsMutex.
type persistentStream struct {
	s      network.Stream
	sendMu sync.Mutex

	// rank is the rank of the connection the stream sits on (connRank*).
	rank int
	// done is closed when the reader of the stream ends: the peer closed it, or it broke.
	done chan struct{}

	// The fields below are guarded by sendMu. retired says that the stream was replaced
	// by one on a better connection and is closed for writing. lastWrite is when a frame
	// was last written on it. nextMoveAt is when a move may be tried again after one
	// that failed.
	retired    bool
	lastWrite  time.Time
	nextMoveAt time.Time
}

// WebRTCConn holds the live state of an established WebRTC DataChannel connection
// to a single remote peer.
type WebRTCConn struct {
	pc         *pwebrtc.PeerConnection
	dc         *pwebrtc.DataChannel
	remotePeer peer.ID

	// peerSessionFrames says that the peer declared, in the offer or in the answer that
	// created this DataChannel, that it accepts Session frames on it. It belongs to the
	// channel, not to the peer: a later exchange decides again.
	peerSessionFrames bool

	// stage is the stage of the offer that created this DataChannel, ours or the peer's:
	// empty for the offer of a peer that names none.
	stage string

	// The libp2p connection the signalling ran on, as a peer first seen through a Session
	// frame of this channel is surfaced to the application pools with: its direction, and
	// its remote address for a peer the peerstore knows nothing of.
	direction  string
	remoteAddr ma.Multiaddr

	// openedAt and lastInbound are what the channel is fresh by (WebRTCFreshnessWindow):
	// when it was registered and when something last arrived on it, in unix nanoseconds
	// (zero: nothing arrived yet). lastInbound is written by the callback of pion and
	// read by the senders. Wall clock on purpose: a machine that slept must find its
	// channels stale, and the monotonic clock of Go does not count the time of a suspend.
	openedAt    int64
	lastInbound atomic.Int64

	// sendMu serializes the chunked-send loop so that concurrent callers cannot
	// interleave their chunks on the wire (which would corrupt reassembly).
	sendMu sync.Mutex

	// reassemblyBuf accumulates inbound chunks between a START and an END chunk.
	// Pion delivers OnMessage callbacks sequentially for a single DataChannel,
	// so no mutex is needed here.
	reassemblyBuf []byte

	// reassemblyStart is when the current multi-chunk reassembly began (set on the
	// START chunk). Used to enforce WebRTCReassemblyTimeout. Touched only inside the
	// sequential OnMessage callback, like reassemblyBuf, so it needs no mutex.
	reassemblyStart time.Time
}

// The public STUN servers of a node whose config names none: all of its servers when it has
// no config at all, and next to its TURN server in the direct stage of a WebRTC offer.
var defaultSTUNServers = []string{
	"stun:stun.l.google.com:19302",
	"stun:global.stun.twilio.com:3478",
	"stun:stun.cloudflare.com:3478",
}

// Define a list of curated public relay nodes to use as static relays for all nodes in the network.
var defaultRelays = []string{
	// Relay 1
	"/ip4/159.195.197.171/tcp/30060/p2p/12D3KooWEXse6zf6g6rsFckJqXA6SWWrENNB3oN8tG5qPEPWV5do",
	"/dns4/misc.unaiverse.io/tcp/30060/tls/ws/p2p/12D3KooWEXse6zf6g6rsFckJqXA6SWWrENNB3oN8tG5qPEPWV5do",
	"/ip4/159.195.197.171/udp/30061/quic-v1/p2p/12D3KooWEXse6zf6g6rsFckJqXA6SWWrENNB3oN8tG5qPEPWV5do",
	"/ip4/159.195.197.171/udp/30062/quic-v1/webtransport/certhash/uEiBnNxE1gxYn1gm0QazN_XFKZyJgXER26XTh5zj8tNeJqw/certhash/uEiDvDT_0OZ0Zxo5gmvbNqXy7Gmg18M4e51cVWSEDKNRJWQ/p2p/12D3KooWEXse6zf6g6rsFckJqXA6SWWrENNB3oN8tG5qPEPWV5do",
	"/ip4/159.195.197.171/udp/30063/webrtc-direct/certhash/uEiD1gsqDNT0MGDAzjprI3It_WQddXuXJKK9OYjya1VyMHw/p2p/12D3KooWEXse6zf6g6rsFckJqXA6SWWrENNB3oN8tG5qPEPWV5do",
	// Relay 2
    "/ip4/159.195.197.171/tcp/30070/p2p/12D3KooWKSQXVsqfSDJ1MrAD3b4FMCbtAaqKusKKsUXY2QNn2yr6",
    "/dns4/misc.unaiverse.io/tcp/30070/tls/ws/p2p/12D3KooWKSQXVsqfSDJ1MrAD3b4FMCbtAaqKusKKsUXY2QNn2yr6",
    "/ip4/159.195.197.171/udp/30071/quic-v1/p2p/12D3KooWKSQXVsqfSDJ1MrAD3b4FMCbtAaqKusKKsUXY2QNn2yr6",
    "/ip4/159.195.197.171/udp/30072/quic-v1/webtransport/certhash/uEiBYOiBfG-AuF7uRx2J1VX1JohOzxW7iOklOvS8Vc80BVw/certhash/uEiDKjXZwOIj-mdibz9s6xmLwrBBULizyZ3CR-LlQtFwJnQ/p2p/12D3KooWKSQXVsqfSDJ1MrAD3b4FMCbtAaqKusKKsUXY2QNn2yr6",
    "/ip4/159.195.197.171/udp/30073/webrtc-direct/certhash/uEiB-LPy8tHhmDpcs2JWQAXw81QvK6MIWddfo8Qnf9m-R0A/p2p/12D3KooWKSQXVsqfSDJ1MrAD3b4FMCbtAaqKusKKsUXY2QNn2yr6",
}
