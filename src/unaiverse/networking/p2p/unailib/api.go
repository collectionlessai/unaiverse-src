// Here we define the C-exported functions that form the API of the P2P library.
package main

/*
#include <stdlib.h>
*/
import "C" // Enables CGo features, allowing Go to call C code and vice-versa.

import (
	"fmt"
	"log"
	"time"
	"unsafe"
	"context"
	"strings"
	"crypto/tls"
	"encoding/json"
	"path/filepath"
	"encoding/base64"
	"github.com/caddyserver/certmagic"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/event"
	"github.com/libp2p/go-libp2p/core/routing"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peerstore"
	"github.com/libp2p/go-libp2p/p2p/transport/tcp"
	"github.com/libp2p/go-libp2p/p2p/protocol/circuitv2/client"
	ma "github.com/multiformats/go-multiaddr"
	rc "github.com/libp2p/go-libp2p/p2p/protocol/circuitv2/relay"
	ws "github.com/libp2p/go-libp2p/p2p/transport/websocket"
	dht "github.com/libp2p/go-libp2p-kad-dht"
	quic "github.com/libp2p/go-libp2p/p2p/transport/quic"
	golog "github.com/ipfs/go-log/v2"
	manet "github.com/multiformats/go-multiaddr/net"
	rcmgr "github.com/libp2p/go-libp2p/p2p/host/resource-manager"
	libp2p "github.com/libp2p/go-libp2p"                         
	pubsub "github.com/libp2p/go-libp2p-pubsub"
	webrtc "github.com/libp2p/go-libp2p/p2p/transport/webrtc"
	pwebrtc "github.com/pion/webrtc/v4"
	p2pforge "github.com/ipshipyard/p2p-forge/client"
	autorelay "github.com/libp2p/go-libp2p/p2p/host/autorelay"
	webtransport "github.com/libp2p/go-libp2p/p2p/transport/webtransport"
)

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
		return C.CString(jsonErrorResponse("Invalid instance index", errMsg))
	}

	// --- Instance Creation and State Check ---
	globalInstanceMutex.Lock()
	if allInstances[instanceIndex] != nil {
		globalInstanceMutex.Unlock()
		msg := fmt.Sprintf("Instance %d is already initialized. Please call CloseNode first.", instanceIndex)
		return C.CString(jsonErrorResponse(msg, nil))
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
		webrtcConnections:     make(map[peer.ID]*WebRTCConn),
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
		return C.CString(jsonErrorResponse("Invalid Configuration JSON", err))
	}

	// --- Sanity checks on the config ---
	// If one of the three parameters for custom certificates is specified, all three are required.
	if cfg.TLS.Domain != "" || cfg.TLS.CertPath != "" || cfg.TLS.KeyPath != "" {
		if cfg.TLS.Domain == "" || cfg.TLS.CertPath == "" || cfg.TLS.KeyPath == "" {
			return C.CString(jsonErrorResponse(fmt.Sprintf("Instance %d: Missing at least one of 'Domain', 'CertPath' or 'KeyPath'.", instanceIndex), nil))
		}
	} // in the following, cfg.TLS.Domain != "" will be used as flag for useCustomTLS

	// Having both customTLS and autoTLS is not allowed
	if cfg.TLS.Domain != "" && cfg.TLS.AutoTLS {
		return C.CString(jsonErrorResponse(fmt.Sprintf("Instance %d: Cannot specify both a 'Domain' and 'AutoTLS'.", instanceIndex), nil))
	}

	// If we use AutoTLS we need the DHT on
	if cfg.TLS.AutoTLS && !cfg.DHT.Enabled {
		return C.CString(jsonErrorResponse(fmt.Sprintf("Instance %d: Using TLS requires DHT 'Enabled'.", instanceIndex), nil))
	}

	// If we want RelayService we must be public (either forced or via AutoNat)
	if cfg.Relay.EnableService {
		if !cfg.Relay.EnableClient {
			return C.CString(jsonErrorResponse(fmt.Sprintf("Instance %d: Cannot set libp2p.DisableRelay() if we want to offer relay services.", instanceIndex), nil))
		}
		if !(cfg.DHT.Enabled || cfg.Network.ForcePublic) {
			return C.CString(jsonErrorResponse(fmt.Sprintf("Instance %d: A relay needs to be publicly reachable (forced or discovered).", instanceIndex), nil))
		}
	}

	// If we want to keep dht it needs to be enabled
	if cfg.DHT.Keep && !cfg.DHT.Enabled {
		return C.CString(jsonErrorResponse(fmt.Sprintf("Instance %d: Cannot set 'DHT.Keep' if DHT is not 'Enabled'.", instanceIndex), nil))
	}

	// --- Load or Create Persistent Identity ---
	keyPath := filepath.Join(cfg.IdentityDir, "identity.key")
	privKey, err := loadOrCreateIdentity(keyPath)
	if err != nil {
		return C.CString(jsonErrorResponse(fmt.Sprintf("Instance %d: Failed to prepare identity", instanceIndex), err))
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
			return C.CString(jsonErrorResponse(fmt.Sprintf("Instance %d: Failed to create AutoTLS cert manager", instanceIndex), err))
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
		return C.CString(jsonErrorResponse(fmt.Sprintf("Instance %d: Failed to create multiaddrs", instanceIndex), err))
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
        return C.CString(jsonErrorResponse(fmt.Sprintf("Instance %d: Failed to create resource manager", instanceIndex), err))
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
			return C.CString(jsonErrorResponse(fmt.Sprintf("Instance %d: Failed to load Custom TLS certificate and key", instanceIndex), err))
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
			// resources.ReservationTTL = time.Hour		// default is 1h
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
				options = append(options, libp2p.EnableAutoRelayWithPeerSource(
					ni.PeerSource, 
					autorelay.WithBootDelay(time.Second*10),
					autorelay.WithNumRelays(1),            // Stop looking for a 2nd relay
					autorelay.WithMinCandidates(1),        // Start with only 1 candidate found
					autorelay.WithBackoff(time.Second*5),  // Retry every 5s, not every hour
				))
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
		return C.CString(jsonErrorResponse(fmt.Sprintf("Instance %d: Failed to create host", instanceIndex), err))
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
		return C.CString(jsonErrorResponse(fmt.Sprintf("Instance %d: Failed to create address cache subscription", instanceIndex), err))
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
			return C.CString(jsonErrorResponse("Failed to subscribe to reachability events", err))
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
				return C.CString(jsonErrorResponse("Context cancelled during init", nil))
			}
		}
	}

	// --- PubSub Initialization ---
	if err := setupPubSub(ni); err != nil {
		return C.CString(jsonErrorResponse(fmt.Sprintf("Instance %d: Failed to create PubSub", instanceIndex), err))
	}
	logger.Debugf("[GO] ✅ Instance %d: PubSub (GossipSub) initialized.\n", instanceIndex)

	// --- Setup Notifiers and Handlers ---
	setupNotifiers(ni)
	logger.Debugf("[GO] 🔔 Instance %d: Registered network event notifier.\n", instanceIndex)

	setupDirectMessageHandler(ni)
	logger.Debugf("[GO] ✅ Instance %d: Direct message handler set up.\n", instanceIndex)

	// --- WebRTC Signaling Handler ---
	if cfg.WebRTC.Enabled {
		if cfg.WebRTC.ICEConfig != nil {
			ni.iceConfig = cfg.WebRTC.ICEConfig
		}
		setupWebRTCSignalHandler(ni)
		logger.Debugf("[GO] ✅ Instance %d: WebRTC signaling handler registered.\n", instanceIndex)
	}

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
	multiaddrs, err := goGetNodeAddresses(ni, "")
	if err != nil {
		return C.CString(jsonErrorResponse(
			fmt.Sprintf("Instance %d: Failed to obtain node addresses after waiting for reachability", instanceIndex),
			err,
		))
	}

	// We must translate the multiaddrs to strings for the Python boundary
	var nodeAddresses []string
	for _, addr := range multiaddrs {
		nodeAddresses = append(nodeAddresses, addr.String())
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
	return C.CString(jsonSuccessResponse(response))
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
		return C.CString(jsonErrorResponse("Invalid instance", err))
	}
	
	goAddrsJSON := C.GoString(addrsJSONC)
	logger.Debugf("[GO] 📞 Instance %d: Attempting to connect to peer with addresses: %s\n", ni.instanceIndex, goAddrsJSON)

	// --- Unmarshal Address List from JSON ---
	var addrStrings []string
	if err := json.Unmarshal([]byte(goAddrsJSON), &addrStrings); err != nil {
		return C.CString(jsonErrorResponse("Failed to parse addresses JSON", err))
	}
	if len(addrStrings) == 0 {
		return C.CString(jsonErrorResponse("Address list is empty", nil))
	}

	// --- Create AddrInfo from the list ---
	addrInfo, err := peer.AddrInfoFromString(addrStrings[0])
	if err != nil {
		return C.CString(jsonErrorResponse("Invalid first multiaddress in list", err))
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
		return C.CString(jsonSuccessResponse(addrInfo)) // Caller frees.
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
			return C.CString(jsonErrorResponse(errMsg, nil)) // Return specific timeout error (caller frees).
		}
		// Handle other connection errors.
		errMsg := fmt.Sprintf("Instance %d: Failed to connect to peer %s", ni.instanceIndex, addrInfo.ID)
		// Example: Check for specific common errors if needed
		// if strings.Contains(err.Error(), "no route to host") { ... }
		return C.CString(jsonErrorResponse(errMsg, err)) // Return generic connection error (caller frees).
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
		return C.CString(jsonErrorResponse("Invalid winner multiaddress.", err))
	}
	return C.CString(jsonSuccessResponse(winningAddrInfo)) // Caller frees.
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

	ni, err := getInstance(int(instanceIndexC))
	if err != nil {
		return C.CString(jsonErrorResponse("Invalid instance", err))
	}

	// Convert C string input to Go string.
	goRelayPeerID := C.GoString(relayPeerIDC)
	logger.Debugf("[GO] 🅿️ Instance %d: Attempting to reserve slot on relay with Peer ID: %s\n", ni.instanceIndex, goRelayPeerID)

	// --- Decode Peer ID and build AddrInfo from Peerstore ---
	relayPID, err := peer.Decode(goRelayPeerID)
	if err != nil {
		return C.CString(jsonErrorResponse("Failed to decode relay Peer ID string", err))
	}

	// Retrieve the relay's addresses from the peerstore to construct the AddrInfo.
	relayAddrs, err := goGetNodeAddresses(ni, relayPID)
	if err != nil {
		return C.CString(jsonErrorResponse("Failed to retrieve relay addresses", err))
	}

	// Construct the AddrInfo using the ID and the addresses we know from the peerstore.
	relayInfo := peer.AddrInfo{
		ID:    relayPID,
		Addrs: relayAddrs,
	}

	// Ensure the node is not trying to reserve a slot on itself.
	if relayInfo.ID == ni.host.ID() {
		return C.CString(jsonErrorResponse(
			fmt.Sprintf("Instance %d: Cannot reserve slot on self", ni.instanceIndex), nil,
		)) // Caller frees.
	}

	// --- VERIFY CONNECTION TO RELAY ---
	if len(ni.host.Network().ConnsToPeer(relayInfo.ID)) == 0 {
		errMsg := fmt.Sprintf("Instance %d: Not connected to relay %s. Must connect before reserving.", ni.instanceIndex, relayInfo.ID)
		return C.CString(jsonErrorResponse(errMsg, nil))
	}
	logger.Debugf("[GO]   - Instance %d: Verified connection to relay: %s\n", ni.instanceIndex, relayInfo.ID)

	// --- Attempt Reservation ---
	// Use a separate context with potentially longer timeout for the reservation itself.
	resCtx, resCancel := context.WithTimeout(ni.ctx, 60*time.Second) // 60-second timeout for reservation.
	defer resCancel()
	// Call the circuitv2 client function to request a reservation.
	// This performs the RPC communication with the relay.
	reservation, err := client.Reserve(resCtx, ni.host, relayInfo)
	if err != nil {
		errMsg := fmt.Sprintf("Instance %d: Failed to reserve slot on relay %s", ni.instanceIndex, relayInfo.ID)
		// Handle reservation timeout specifically.
		if resCtx.Err() == context.DeadlineExceeded {
			errMsg = fmt.Sprintf("Instance %d: Reservation attempt on relay %s timed out", ni.instanceIndex, relayInfo.ID)
			return C.CString(jsonErrorResponse(errMsg, nil)) // Caller frees.
		}
		return C.CString(jsonErrorResponse(errMsg, err)) // Caller frees.
	}

	// Although Reserve usually errors out if it fails, double-check if the reservation object is nil.
	if reservation == nil {
		errMsg := fmt.Sprintf("Instance %d: Reservation on relay %s returned nil voucher, but no error", ni.instanceIndex, relayInfo.ID)
		return C.CString(jsonErrorResponse(errMsg, nil)) // Caller frees.
	} else {
		logger.Debugf("[GO] ✅ Instance %d: Successfully reserved slot on relay %s. Reservation expires at: %s\nRelay addresses: %v", ni.instanceIndex, relayInfo.ID, reservation.Expiration.Format(time.RFC3339), reservation.Addrs)
	}

	// --- Construct Relayed Addresses and Update Local Peerstore ---
	// We construct a relayed address for each public address of the relay to maximize reachability.
	var constructedAddrs []ma.Multiaddr
	for _, relayAddr := range relayAddrs {
		// We only want to use public, usable addresses for the circuit
		if manet.IsIPLoopback(relayAddr) || manet.IsIPUnspecified(relayAddr) {
			continue
		}

		// Ensure the relay's address in the peerstore includes its own Peer ID
		baseRelayAddrStr := relayAddr.String()
		if _, idInAddr := peer.SplitAddr(relayAddr); idInAddr == "" {
			baseRelayAddrStr = fmt.Sprintf("%s/p2p/%s", relayAddr.String(), relayInfo.ID.String())
		}
		
		constructedAddr, err := ma.NewMultiaddr(fmt.Sprintf("%s/p2p-circuit", baseRelayAddrStr))
		if err == nil {
			constructedAddrs = append(constructedAddrs, constructedAddr)
		}
	}

	if len(constructedAddrs) == 0 {
		return C.CString(jsonErrorResponse("Reservation succeeded but failed to construct any valid relayed multiaddr", nil))
	}

	logger.Debugf("[GO] 🔗 Instance %d: Constructed %d Relayed Addresses:", ni.instanceIndex, len(constructedAddrs))
	for _, addr := range constructedAddrs {
		logger.Debugf("[GO]      -> %s", addr.String())
	}
	// Tell the Network to start listening on these addresses.
    // This activates the circuit transport for this specific relay, updates host.Addrs(),
    // and triggers the EvtLocalAddressesUpdated event automatically.
    if err := ni.host.Network().Listen(constructedAddrs...); err != nil {
		// If listening fails, it's not fatal for the reservation (we still have the slot),
		// but it implies we might not be reachable via those specific paths.
		logger.Warnf("[GO] ⚠️ Instance %d: Failed to start listener on some relayed addresses: %v", ni.instanceIndex, err)
	} else {
		logger.Debugf("[GO] ✅ Instance %d: Successfully started listening on relayed addresses.", ni.instanceIndex)
	}
	// Also add them directly to the istance
	ni.addrMutex.Lock()
    ni.privateRelayAddrs = constructedAddrs 
    ni.addrMutex.Unlock()

	logger.Infof("[GO] ✅ Instance %d: Reservation successful on relay: %s.\n", ni.instanceIndex, relayInfo.ID)

	// Return the expiration time of the reservation as confirmation.
	return C.CString(jsonSuccessResponse(reservation.Expiration))
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
		return C.CString(jsonErrorResponse("Invalid instance", err))
	}

	goPeerID := C.GoString(peerIDC)
	logger.Debugf("[GO] 🔌 Instance %d: Attempting to disconnect from peer: %s\n", ni.instanceIndex, goPeerID)

	pid, err := peer.Decode(goPeerID)
	if err != nil {
		return C.CString(jsonErrorResponse(
			fmt.Sprintf("Instance %d: Failed to decode peer ID", ni.instanceIndex), err,
		))
	}

	if pid == ni.host.ID() {
		logger.Debugf("[GO] ℹ️ Instance %d: Attempting to disconnect from self (%s), skipping.\n", ni.instanceIndex, pid)
		return C.CString(jsonSuccessResponse("Cannot disconnect from self"))
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

	return C.CString(jsonSuccessResponse(logMsg))
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
		return C.CString(jsonSuccessResponse([]ExtendedPeerInfo{}))
	}

	// Use a Write Lock for the entire critical section to avoid mixing RLock and Lock.
	ni.peersMutex.RLock()
	defer ni.peersMutex.RUnlock() // Ensure lock is released.

	// Create a slice to hold the results directly from the map.
	peersList := make([]ExtendedPeerInfo, 0, len(ni.friendlyPeers))
	
	for _, peerInfo := range ni.friendlyPeers {
			peersList = append(peersList, peerInfo)
		}

	// logger.Debugf("[GO] ℹ️ Instance %d: Reporting %d currently tracked and active peers.\n", ni.instanceIndex, len(peersList))

	// Return the list of active peers as a JSON success response.
	return C.CString(jsonSuccessResponse(peersList)) // Caller frees.
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
	return C.CString(jsonSuccessResponse(responsePayload)) // Caller frees.
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
		return C.CString(jsonErrorResponse("Invalid instance", err))
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
			return C.CString(jsonErrorResponse(errMsg, err))
		}
	}

	// Call the internal Go function with the resolved peer.ID or empty peer.ID for local
	multiaddrs, err := goGetNodeAddresses(ni, pidForInternalCall)
	if err != nil {
		return C.CString(jsonErrorResponse(err.Error(), nil))
	}

	// We must translate the multiaddrs to strings for the Python boundary
	var addresses []string
	for _, addr := range multiaddrs {
		addresses = append(addresses, addr.String())
	}

	return C.CString(jsonSuccessResponse(addresses))
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
		return C.CString(jsonErrorResponse("Invalid instance", err))
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
			return C.CString(jsonErrorResponse("PubSub not initialized, cannot broadcast", nil))
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
				return C.CString(jsonErrorResponse(fmt.Sprintf("Failed to join PubSub topic '%s'", goChannel), err))
			}
			ni.topics[goChannel] = topic
			logger.Debugf("[GO] ✅ Instance %d: Joined PubSub topic: %s for publishing.\n", ni.instanceIndex, goChannel)
		}
		ni.pubsubMutex.Unlock()

		// Directly publish the raw Protobuf payload.
		if err := topic.Publish(ni.ctx, goData); err != nil {
			// Failed to publish to topic
			return C.CString(jsonErrorResponse(fmt.Sprintf("Failed to publish to topic '%s'", goChannel), err))
		}
		logger.Infof("[GO] 🌍 Instance %d: Broadcast to topic '%s' (%d bytes)\n", ni.instanceIndex, goChannel, len(goData))
		return C.CString(jsonSuccessResponse(fmt.Sprintf("Message broadcast to topic %s", goChannel)))

	} else if strings.Contains(goChannel, "::dm:") {
		// --- Direct Peer-to-Peer Message Sending (Persistent Stream Logic) ---
		receiverChannelIDStr := strings.Split(goChannel, "::dm:")[1] // Extract the receiver's channel ID from the format "dm:<peerID>-<channelSpecifier>"
		peerIDStr := strings.Split(receiverChannelIDStr, "-")[0]
		pid, err := peer.Decode(peerIDStr)
		if err != nil {
			// Invalid peer ID format
			return C.CString(jsonErrorResponse("Invalid peer ID format in channel string", err))
		}

		if pid == ni.host.ID() {
			// Attempt to send direct message to self
			return C.CString(jsonErrorResponse("Attempt to send direct message to self is invalid", nil))
		}

		// --- WebRTC DataChannel path (preferred: direct, NAT-traversed) ---
		ni.webrtcMutex.RLock()
		wconn, hasWebRTC := ni.webrtcConnections[pid]
		ni.webrtcMutex.RUnlock()
		if hasWebRTC && wconn.dc != nil && wconn.dc.ReadyState() == pwebrtc.DataChannelStateOpen {
			if err = writeToWebRTCDataChannel(wconn, goChannel, goData); err == nil {
				logger.Infof("[GO] 📤 Instance %d: Sent to %s via WebRTC DataChannel\n", ni.instanceIndex, pid)
				return C.CString(jsonSuccessResponse(fmt.Sprintf("Direct message sent to %s (WebRTC).", pid)))
			}
			logger.Warnf("[GO] ⚠️ Instance %d: WebRTC send to %s failed (%v), falling back to stream\n", ni.instanceIndex, pid, err)
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
				return C.CString(jsonSuccessResponse(fmt.Sprintf("Direct message sent to %s (reused stream).", pid)))
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
			return C.CString(jsonErrorResponse(fmt.Sprintf("Failed to write to stream %s (closed).", pid), err))
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
				return C.CString(jsonErrorResponse(fmt.Sprintf("Failed to open new stream to %s.", pid), err))
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
				return C.CString(jsonErrorResponse(fmt.Sprintf("Failed to write to new stream to '%s' (needs reconnect).", pid), err))
			}

			logger.Infof("[GO] 📤 Instance %d: Sent to %s via Stream %s (New)\n", ni.instanceIndex, pid, stream.ID())
			return C.CString(jsonSuccessResponse(fmt.Sprintf("Direct message sent to %s (new stream).", pid)))
		}
	} else {
		// Invalid channel format
		return C.CString(jsonErrorResponse(fmt.Sprintf("Invalid channel format '%s'", goChannel), nil))
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
		return C.CString(jsonErrorResponse("Invalid instance", err))
	}

	// Convert C string input to Go string.
	channel := C.GoString(channelC)
	logger.Debugf("[GO] <sub> Instance %d: Attempting to subscribe to topic: %s\n", ni.instanceIndex, channel)
	
	// Get instance-specific state and mutex
	instancePubsub := ni.pubsub
	if ni.host == nil || instancePubsub == nil {
		return C.CString(jsonErrorResponse(
			fmt.Sprintf("Instance %d: Host or PubSub not initialized", ni.instanceIndex), nil,
		))
	}

	// Lock the mutex for safe access to the shared topics and subscriptions maps for this instance.
	ni.pubsubMutex.Lock()
	defer ni.pubsubMutex.Unlock() // Ensure mutex is unlocked when function returns.

	// Check if already subscribed to this topic for this instance.
	if _, exists := ni.subscriptions[channel]; exists {
		logger.Debugf("[GO] <sub> Instance %d: Already subscribed to topic: %s\n", ni.instanceIndex, channel)
		// Return success, indicating the desired state is already met.
		return C.CString(jsonSuccessResponse(
			fmt.Sprintf("Instance %d: Already subscribed to topic %s", ni.instanceIndex, channel),
		)) // Caller frees.
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
		return C.CString(jsonErrorResponse(errMsg, err)) // Caller frees.
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
		return C.CString(jsonErrorResponse(errMsg, err)) // Caller frees.
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
	return C.CString(jsonSuccessResponse(
		fmt.Sprintf("Instance %d: Subscribed to topic %s", ni.instanceIndex, channel),
	)) // Caller frees.
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
		return C.CString(jsonSuccessResponse(fmt.Sprintf("Instance %d: Not subscribed (instance not running)", ni.instanceIndex)))
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
		return C.CString(jsonSuccessResponse(
			fmt.Sprintf("Instance %d: Not currently subscribed to topic %s", ni.instanceIndex, channel),
		)) // Caller frees.
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
	return C.CString(jsonSuccessResponse(
		fmt.Sprintf("Instance %d: Unsubscribed from topic %s", ni.instanceIndex, channel),
	)) // Caller frees.
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
		return C.CString(jsonErrorResponse("Invalid instance", err))
	}

	// Get the message store for this instance
	store := ni.messageStore
	if store == nil {
		logger.Errorf("[GO] ❌ Instance %d: PopMessages: Message store not initialized.\n", ni.instanceIndex)
		return C.CString(jsonErrorResponse(fmt.Sprintf("Instance %d: Message store not initialized", ni.instanceIndex), nil))
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
		return C.CString(jsonErrorResponse(
			fmt.Sprintf("Instance %d: Failed to marshal popped messages", ni.instanceIndex), err,
		))
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
			return C.CString(jsonErrorResponse(summaryMsg, fmt.Errorf("details: %v", errorMessages)))
		}

		logger.Infof("[GO] 🛑 All initialized nodes closed.")
		return C.CString(jsonSuccessResponse(summaryMsg))

	} else {
		if instanceIndex < 0 || instanceIndex >= maxInstances {
			err := fmt.Errorf("invalid instance index: %d. Must be between 0 and %d", instanceIndex, maxInstances-1)
			return C.CString(jsonErrorResponse("Invalid instance index for single close", err)) // Caller frees.
		}

		globalInstanceMutex.Lock()
		defer globalInstanceMutex.Unlock()

		instance := allInstances[instanceIndex]
		if instance == nil {
			logger.Debugf("[GO] ℹ️ Instance %d: Node was already closed.\n", instanceIndex)
			return C.CString(jsonSuccessResponse(fmt.Sprintf("Instance %d: Node was already closed", instanceIndex)))
		}

		err := instance.Close()
		allInstances[instanceIndex] = nil

		if err != nil {
			return C.CString(jsonErrorResponse(fmt.Sprintf("Instance %d: Error closing host", instanceIndex), err))
		}

		logger.Infof("[GO] 🛑 Instance %d: Node closed successfully.\n", instanceIndex)
		return C.CString(jsonSuccessResponse(fmt.Sprintf("Instance %d: Node closed successfully", instanceIndex)))
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

// InitiateWebRTCConnection opens a /unaiverse/webrtc-signal/1.0.0 stream to
// remotePeer (which must already be reachable, e.g. via relay) and performs
// the SDP offer/answer + ICE handshake to establish a direct WebRTC DataChannel.
//
// Returns JSON {"state":"Success",...} or {"state":"Error",...}.
// The caller MUST free the returned string with FreeString.
//
//export InitiateWebRTCConnection
func InitiateWebRTCConnection(
	instanceIndexC C.int,
	peerIDC *C.char,
) *C.char {
	ni, err := getInstance(int(instanceIndexC))
	if err != nil {
		return C.CString(jsonErrorResponse("Invalid instance", err))
	}

	peerIDStr := C.GoString(peerIDC)
	pid, err := peer.Decode(peerIDStr)
	if err != nil {
		return C.CString(jsonErrorResponse("Invalid peer ID", err))
	}

	if err := initiateWebRTCConnection(ni, pid); err != nil {
		return C.CString(jsonErrorResponse(fmt.Sprintf("WebRTC connection to %s failed", pid), err))
	}
	return C.CString(jsonSuccessResponse(fmt.Sprintf("WebRTC DataChannel established with %s", pid)))
}

// GetWebRTCConnections returns a JSON array of all peers that currently have
// an active WebRTC DataChannel with this node instance.
//
// Each element: {"peer_id": "...", "state": "open"|"other"}
// The caller MUST free the returned string with FreeString.
//
//export GetWebRTCConnections
func GetWebRTCConnections(instanceIndexC C.int) *C.char {
	ni, err := getInstance(int(instanceIndexC))
	if err != nil {
		return C.CString(jsonErrorResponse("Invalid instance", err))
	}

	type entry struct {
		PeerID string `json:"peer_id"`
		State  string `json:"state"`
	}

	ni.webrtcMutex.RLock()
	result := make([]entry, 0, len(ni.webrtcConnections))
	for pid, conn := range ni.webrtcConnections {
		stateStr := "other"
		if conn.dc != nil && conn.dc.ReadyState() == pwebrtc.DataChannelStateOpen {
			stateStr = "open"
		}
		result = append(result, entry{PeerID: pid.String(), State: stateStr})
	}
	ni.webrtcMutex.RUnlock()

	return C.CString(jsonSuccessResponse(result))
}

// CloseWebRTCConnection tears down the WebRTC PeerConnection with a specific peer.
//
// Returns JSON {"state":"Success",...} or {"state":"Error",...}.
// The caller MUST free the returned string with FreeString.
//
//export CloseWebRTCConnection
func CloseWebRTCConnection(
	instanceIndexC C.int,
	peerIDC *C.char,
) *C.char {
	ni, err := getInstance(int(instanceIndexC))
	if err != nil {
		return C.CString(jsonErrorResponse("Invalid instance", err))
	}

	peerIDStr := C.GoString(peerIDC)
	pid, err := peer.Decode(peerIDStr)
	if err != nil {
		return C.CString(jsonErrorResponse("Invalid peer ID", err))
	}

	ni.webrtcMutex.Lock()
	conn, exists := ni.webrtcConnections[pid]
	if exists {
		delete(ni.webrtcConnections, pid)
	}
	ni.webrtcMutex.Unlock()

	if !exists {
		return C.CString(jsonErrorResponse(fmt.Sprintf("No WebRTC connection to %s", pid), nil))
	}

	if conn.dc != nil {
		conn.dc.Close()
	}
	if conn.pc != nil {
		conn.pc.Close()
	}
	logger.Infof("[GO] 🗑️ Instance %d: WebRTC connection to %s closed by request.", ni.instanceIndex, pid)
	return C.CString(jsonSuccessResponse(fmt.Sprintf("WebRTC connection to %s closed.", pid)))
}

// main is the entry point for a Go executable.
func main() {
	// This message will typically only be seen if you run `go run lib.go`
	// or build and run as a standard executable, NOT when used as a shared library.
	logger.Debugf("[GO] libp2p Go library main function (not executed in c-shared library mode)")
}
