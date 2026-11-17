// webrtc_signal.go
// Application-level WebRTC signaling protocol for Go-to-browser NAT traversal.
//
// Both peers must already share a relayed libp2p connection. Over that relay
// they open a "/unaiverse/webrtc-signal/1.0.0" stream to exchange SDP
// offer/answer (using vanilla/batch ICE – full gathering before send).
// Once the DataChannel opens, messages use the same framing as the chat
// protocol and feed into the normal message queue (PopMessages).
package main

import (
	"io"
	"fmt"
	"time"
	"bytes"
	"errors"
	"slices"
	"context"
	"strings"
	"encoding/json"
	"encoding/binary"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/network"
	pwebrtc "github.com/pion/webrtc/v4"
)

// // getWebRTCAPI creates a custom Pion API that enables ICE-TCP
// // to mimic browser fallback behavior on strict networks.
// func getWebRTCAPI() *pwebrtc.API {
// 	sEngine := pwebrtc.SettingEngine{}
	
// 	sEngine.SetNetworkTypes([]pwebrtc.NetworkType{
// 		pwebrtc.NetworkTypeUDP4,
// 		pwebrtc.NetworkTypeTCP4,
// 	})

// 	// 1. Aggressive Timeouts (Fail fast like a browser)
// 	sEngine.SetICETimeouts(
// 		2*time.Second,  // disconnected timeout
// 		5*time.Second,  // failed timeout
// 		2*time.Second,  // keep-alive
// 	)

// 	// 2. Restrict UDP ports to a range less likely to be blanket-blocked
// 	err := sEngine.SetEphemeralUDPPortRange(10000, 10500)
// 	if err != nil {
// 		logger.Warnf("Failed to set ephemeral UDP port range: %v", err)
// 	}

// 	// 3. Setup single-port UDP Mux
// 	udpListener, err := net.ListenUDP("udp", &net.UDPAddr{
// 		IP:   net.IP{0, 0, 0, 0},
// 		Port: 0, 
// 	})
// 	if err != nil {
// 		logger.Warnf("Failed to setup UDP listener for WebRTC: %v", err)
// 	} else {
// 		udpMux := pwebrtc.NewICEUDPMux(nil, udpListener)
// 		sEngine.SetICEUDPMux(udpMux)
// 	}

// 	// 4. Setup TCP Mux as fallback
// 	tcpListener, err := net.ListenTCP("tcp", &net.TCPAddr{
// 		IP:   net.IP{0, 0, 0, 0},
// 		Port: 0,
// 	})
// 	if err != nil {
// 		logger.Warnf("Failed to setup TCP listener for WebRTC: %v", err)
// 	} else {
// 		tcpMux := pwebrtc.NewICETCPMux(nil, tcpListener, 8)
// 		sEngine.SetICETCPMux(tcpMux)
// 	}

// 	return pwebrtc.NewAPI(pwebrtc.WithSettingEngine(sEngine))
// }

// getWebRTCConfig builds a pion WebRTC configuration from the node's stored ICE config.
// Falls back to the public STUN servers when no config is present. The direct stage has
// no TURN server; every other stage, and an offer that names none, has the whole list.
func getWebRTCConfig(ni *NodeInstance, stage string) pwebrtc.Configuration {
    ni.iceConfigMu.RLock()
    cfg := ni.iceConfig
    ni.iceConfigMu.RUnlock()
	if cfg != nil && (len(cfg.STUNServers) > 0 || len(cfg.TURNServers) > 0) {
		if stage == WebRTCStageDirect {
			urls := stunURLs(cfg)
			// The root server gives a node its TURN server and no STUN server. The
			// reflexive address must not hang on that one server answering, so the
			// public ones are asked too.
			if len(cfg.STUNServers) == 0 {
				for _, url := range defaultSTUNServers {
					if !slices.Contains(urls, url) {
						urls = append(urls, url)
					}
				}
			}
			return pwebrtc.Configuration{ICEServers: []pwebrtc.ICEServer{{URLs: urls}}}
		}
		var STUNServers []pwebrtc.ICEServer
		if len(cfg.STUNServers) > 0 {
			STUNServers = append(STUNServers, pwebrtc.ICEServer{URLs: cfg.STUNServers})
		}
		for _, t := range cfg.TURNServers {
			STUNServers = append(STUNServers, pwebrtc.ICEServer{
				URLs:           t.URLs,
				Username:       t.Username,
				Credential:     t.Credential,
				CredentialType: pwebrtc.ICECredentialTypePassword,
			})
		}
		return pwebrtc.Configuration{ICEServers: STUNServers}
	}
	return pwebrtc.Configuration{
		ICEServers: []pwebrtc.ICEServer{
			{URLs: defaultSTUNServers},
		},
	}
}

// stunURLs returns the servers of an ICE config as STUN servers alone: its STUN URLs, and
// each TURN URL turned into the STUN URL of the same server. A TURN server answers STUN
// requests too, and with a TURN URL in the list pion already asks it for the reflexive
// address: a node whose config holds its TURN credentials and nothing else (what the root
// server gives) keeps that address in the direct stage, and gets no relay candidate.
func stunURLs(cfg *ICEConfig) []string {
	urls := append([]string(nil), cfg.STUNServers...)
	for _, t := range cfg.TURNServers {
		for _, url := range t.URLs {
			scheme, rest, _ := strings.Cut(url, ":")
			if strings.EqualFold(scheme, "turn") || strings.EqualFold(scheme, "turns") {
				server, _, _ := strings.Cut(rest, "?")
				url = "stun:" + server
			}
			if !slices.Contains(urls, url) {
				urls = append(urls, url)
			}
		}
	}
	return urls
}

// watchICECandidates has to be called before SetLocalDescription, which starts the
// gathering. The function it returns waits until the local description holds the
// candidates the stage needs, and says what ended the wait: "complete", the gathering is
// over; "quiet", a candidate of the needed kind was seen (a relay one where the list has
// TURN, any one in the direct stage) and none followed for WebRTCGatherQuiet; "limit",
// the bound of the stage passed. What was gathered by then is what the peer gets. It
// returns false when ctx ends first.
func watchICECandidates(pc *pwebrtc.PeerConnection, stage string) func(ctx context.Context) (string, bool) {
	candidates := make(chan pwebrtc.ICECandidateType, 64)
	complete := make(chan struct{}, 1)
	pc.OnICECandidate(func(c *pwebrtc.ICECandidate) {
		// Neither send may hold a goroutine of pion: a candidate that finds no room is
		// not counted, and the end of the gathering is said once.
		if c == nil {
			select {
			case complete <- struct{}{}:
			default:
			}
			return
		}
		select {
		case candidates <- c.Typ:
		default:
		}
	})
	return func(ctx context.Context) (string, bool) {
		limit := WebRTCTURNGatherMax
		if stage == WebRTCStageDirect {
			limit = WebRTCDirectGatherMax
		}
		limitTimer := time.NewTimer(limit)
		defer limitTimer.Stop()
		// Nil until a candidate of the needed kind is seen, then restarted at each candidate.
		var quiet <-chan time.Time
		for {
			select {
			case <-complete:
				return "complete", true
			case typ := <-candidates:
				if quiet != nil || stage == WebRTCStageDirect || typ == pwebrtc.ICECandidateTypeRelay {
					quiet = time.After(WebRTCGatherQuiet)
				}
			case <-quiet:
				return "quiet", true
			case <-limitTimer.C:
				return "limit", true
			case <-ctx.Done():
				return "", false
			}
		}
	}
}

// writeSignalMessage encodes msg as JSON and writes it length-prefixed to w.
func writeSignalMessage(w io.Writer, msg SignalMessage) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("marshal signal message: %w", err)
	}
	buf := new(bytes.Buffer)
	if err := binary.Write(buf, binary.BigEndian, uint32(len(data))); err != nil {
		return err
	}
	buf.Write(data)
	_, err = w.Write(buf.Bytes())
	return err
}

// readSignalMessage reads one length-prefixed JSON SignalMessage from r.
func readSignalMessage(r io.Reader) (SignalMessage, error) {
	var length uint32
	if err := binary.Read(r, binary.BigEndian, &length); err != nil {
		return SignalMessage{}, fmt.Errorf("read signal length: %w", err)
	}
	if length == 0 || length > 1<<20 { // sanity cap: 1 MiB
		return SignalMessage{}, fmt.Errorf("invalid signal message length: %d", length)
	}
	data := make([]byte, length)
	if _, err := io.ReadFull(r, data); err != nil {
		return SignalMessage{}, fmt.Errorf("read signal body: %w", err)
	}
	var msg SignalMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		return SignalMessage{}, fmt.Errorf("unmarshal signal message: %w", err)
	}
	return msg, nil
}

// buildDirectMessageFrame constructs a framed message as a byte slice.
// Each DataChannel.Send call carries exactly one complete frame.
//
// Frame layout (identical to the libp2p chat stream protocol):
//
//	[4-byte big-endian total length][1-byte channel name length][channel name][payload]
//
// "total length" counts bytes from channel-name-length field onward (not including
// the 4-byte total-length field itself).
func buildDirectMessageFrame(channel string, payload []byte) ([]byte, error) {
	channelBytes := []byte(channel)
	if len(channelBytes) > 255 {
		return nil, fmt.Errorf("channel name exceeds 255 bytes: %s", channel)
	}
	channelLen := uint8(len(channelBytes))
	totalLength := uint32(1 + len(channelBytes) + len(payload))
	if totalLength > MaxMessageSize {
		return nil, fmt.Errorf("outgoing message size (%d) exceeds limit (%d)", totalLength, MaxMessageSize)
	}
	buf := new(bytes.Buffer)
	binary.Write(buf, binary.BigEndian, totalLength)
	binary.Write(buf, binary.BigEndian, channelLen)
	buf.Write(channelBytes)
	buf.Write(payload)
	return buf.Bytes(), nil
}

// parseDirectMessageFrame extracts the channel name and payload from a DataChannel frame.
func parseDirectMessageFrame(data []byte) (channel string, payload []byte, err error) {
	if len(data) < 5 {
		return "", nil, fmt.Errorf("frame too short (%d bytes)", len(data))
	}
	totalLength := binary.BigEndian.Uint32(data[0:4])
	if uint32(len(data)) != totalLength + 4 {
		return "", nil, fmt.Errorf("frame length mismatch: header=%d actual=%d", totalLength, uint32(len(data)) - 4)
	}
	channelLen := int(data[4])
	if len(data) < channelLen + 5 {
		return "", nil, fmt.Errorf("frame truncated at channel name")
	}
	return string(data[5 : channelLen + 5]), data[channelLen + 5:], nil
}

// now is the clock the freshness of a DataChannel is measured with.
func (ni *NodeInstance) now() time.Time {
	if ni.clock != nil {
		return ni.clock()
	}
	return time.Now()
}

// isOpen is the test on the state of the channel alone, the one the chat send, the list
// handed to Python and the cleanup of a peer with no libp2p connection left go by.
func (c *WebRTCConn) isOpen() bool {
	return c.dc != nil && c.dc.ReadyState() == pwebrtc.DataChannelStateOpen
}

// isFresh reports whether the channel gave a sign of life within WebRTCFreshnessWindow:
// something arrived on it, or it opened.
func (c *WebRTCConn) isFresh(now time.Time) bool {
	at, window := now.UnixNano(), int64(WebRTCFreshnessWindow)
	if at-c.openedAt < window {
		return true
	}
	last := c.lastInbound.Load()
	return last != 0 && at-last < window
}

// takesSessionFrames reports whether a Session frame may go on the channel now: the peer
// declared that it accepts them there, the channel is open, its peer connection is
// connected, and the channel is fresh.
func (c *WebRTCConn) takesSessionFrames(now time.Time) bool {
	return c.peerSessionFrames && c.isOpen() && c.pc != nil &&
		c.pc.ConnectionState() == pwebrtc.PeerConnectionStateConnected && c.isFresh(now)
}

// sendSessionFrameOnDataChannel sends a Session frame on the DataChannel to pid when the
// channel takes it, in the frame the chat uses there: the channel name is the Session
// protocol, the payload is what a Session stream carries after its length prefix. It
// reports whether the frame left: when it did not, the caller sends it on a stream.
func sendSessionFrameOnDataChannel(ni *NodeInstance, pid peer.ID, protocolID string, payload []byte) bool {
	ni.webrtcMutex.RLock()
	conn := ni.webrtcConnections[pid]
	ni.webrtcMutex.RUnlock()
	if conn == nil || !conn.takesSessionFrames(ni.now()) {
		return false
	}
	if err := writeToWebRTCDataChannel(conn, protocolID, payload); err != nil {
		logger.Warnf("[GO] ⚠️ Instance %d: Session frame on %s to %s failed on the WebRTC DataChannel (%v), falling back to a stream",
			ni.instanceIndex, protocolID, pid, err)
		return false
	}
	// As on a stream: the sending side of Session traffic surfaces the peer to the pools.
	promoteFriendlyPeer(ni, pid, conn.direction, conn.remoteAddr)
	logger.Infof("[GO] 📤 Instance %d: Sent %d bytes on %s to %s (WebRTC DataChannel)\n", ni.instanceIndex, len(payload), protocolID, pid)
	return true
}

// wireWebRTCDataChannel installs the handlers of a DataChannel: delivery of what arrives
// into the shared message queue, and the log of its end. It is called before the channel
// can open. A peer may send as soon as its own end is open, and pion drops without a word
// a message that finds no handler, so the handlers cannot wait for the registration.
// The caller fills conn with the two pion objects, the peer and the libp2p connection the
// signalling runs on.
func wireWebRTCDataChannel(ni *NodeInstance, conn *WebRTCConn) {
	remotePeer, dc := conn.remotePeer, conn.dc

	// deliverFrame parses and stores a complete reassembled DataChannel frame.
	deliverFrame := func(frame []byte) {
		ch, payload, err := parseDirectMessageFrame(frame)
		if err != nil {
			logger.Warnf("[GO] ⚠️ Instance %d: Bad DataChannel frame from %s: %v",
				ni.instanceIndex, remotePeer, err)
			return
		}
		if isSessionProtocol(ch) {
			// A Session frame, treated as one read from a Session stream: the peer is
			// surfaced to the pools and the frame is queued under its protocol, with the
			// protocol as its tag. A frame so named never reaches the chat path. Every
			// offer and answer of this node declares that it accepts them.
			if len(payload) == 0 {
				logger.Warnf("[GO] ⚠️ Instance %d: Empty Session frame on %s from %s on the WebRTC DataChannel, ignoring",
					ni.instanceIndex, ch, remotePeer)
				return
			}
			promoteFriendlyPeer(ni, remotePeer, conn.direction, conn.remoteAddr)
			logger.Infof("[GO] 📨 Instance %d: Session %s from %s (%d bytes, WebRTC DataChannel)\n", ni.instanceIndex, ch, remotePeer, len(payload))
			storeReceivedMessage(ni, remotePeer, ch, ch, payload)
			return
		}
		storeReceivedMessage(ni, remotePeer, ch, UnaiverseWebRTCSignalProtocol, payload)
	}

	// Deliver incoming DataChannel messages into the shared message queue.
	// Chunks are reassembled transparently using the 1-byte flags header.
	// Pion calls OnMessage sequentially for a single DataChannel, so conn.reassemblyBuf
	// is accessed without a mutex.
	dc.OnMessage(func(msg pwebrtc.DataChannelMessage) {
		if len(msg.Data) < 1 {
			logger.Warnf("[GO] ⚠️ Instance %d: Empty DataChannel message from %s, ignoring",
				ni.instanceIndex, remotePeer)
			return
		}
		// Whatever arrives, a chunk included, says that the channel is alive.
		conn.lastInbound.Store(ni.now().UnixNano())

		flags := msg.Data[0]
		chunkPayload := msg.Data[1:]

		switch flags {
		case webRTCChunkFlagStartEnd:
			// 0xC0: complete message in a single chunk — fast path.
			conn.reassemblyBuf = nil
			deliverFrame(chunkPayload)

		case webRTCChunkFlagStart:
			// 0x80: first chunk of a multi-chunk message.
			if conn.reassemblyBuf != nil {
				logger.Warnf("[GO] ⚠️ Instance %d: New START chunk from %s while a reassembly was in progress; discarding %d partial bytes.",
					ni.instanceIndex, remotePeer, len(conn.reassemblyBuf))
			}
			conn.reassemblyBuf = make([]byte, 0, len(chunkPayload)*2)
			conn.reassemblyBuf = append(conn.reassemblyBuf, chunkPayload...)
			conn.reassemblyStart = time.Now()

		case webRTCChunkFlagEnd:
			// 0x40: final chunk — append, check size, deliver.
			if conn.reassemblyBuf == nil {
				logger.Warnf("[GO] ⚠️ Instance %d: Received END chunk without preceding START from %s, discarding",
					ni.instanceIndex, remotePeer)
				return
			}
			if time.Since(conn.reassemblyStart) > WebRTCReassemblyTimeout {
				logger.Warnf("[GO] ⚠️ Instance %d: Reassembly from %s exceeded %v before END; discarding partial message.",
					ni.instanceIndex, remotePeer, WebRTCReassemblyTimeout)
				conn.reassemblyBuf = nil
				return
			}
			conn.reassemblyBuf = append(conn.reassemblyBuf, chunkPayload...)
			if uint32(len(conn.reassemblyBuf)) > MaxMessageSize {
				logger.Errorf("[GO] ❌ Instance %d: Reassembled message from %s exceeds MaxMessageSize (%d), discarding",
					ni.instanceIndex, remotePeer, MaxMessageSize)
				conn.reassemblyBuf = nil
				return
			}
			frame := conn.reassemblyBuf
			conn.reassemblyBuf = nil
			deliverFrame(frame)

		default:
			// flags == 0x00: ambiguous — either a middle continuation chunk or a
			// legacy unchunked frame sent before chunking was introduced.
			//
			// Disambiguate via reassembly state: if we are mid-reassembly the byte
			// is a continuation chunk; otherwise it is a legacy frame whose first
			// byte happens to be 0x00 (MSB of a big-endian uint32 length < 16 MB).
			if conn.reassemblyBuf != nil {
				if time.Since(conn.reassemblyStart) > WebRTCReassemblyTimeout {
					logger.Warnf("[GO] ⚠️ Instance %d: Reassembly from %s exceeded %v mid-stream; discarding partial message.",
						ni.instanceIndex, remotePeer, WebRTCReassemblyTimeout)
					conn.reassemblyBuf = nil
					return
				}
				conn.reassemblyBuf = append(conn.reassemblyBuf, chunkPayload...)
				if uint32(len(conn.reassemblyBuf)) > MaxMessageSize {
					logger.Errorf("[GO] ❌ Instance %d: In-progress reassembly from %s exceeds MaxMessageSize (%d), discarding",
						ni.instanceIndex, remotePeer, MaxMessageSize)
					conn.reassemblyBuf = nil
				}
			} else {
				// Legacy unchunked frame — deliver msg.Data as-is (the 0x00 byte is
				// the MSB of the 4-byte length field, not a header we introduced).
				deliverFrame(msg.Data)
			}
		}
	})

	// The readers of the table look at the state of the DataChannel, the cleanup of the
	// entry at the state of the peer connection: the log has to show both.
	dc.OnClose(func() {
		logger.Infof("[GO] 🔒 Instance %d: WebRTC DataChannel with %s closed", ni.instanceIndex, remotePeer)
	})
	dc.OnError(func(err error) {
		logger.Warnf("[GO] ⚠️ Instance %d: WebRTC DataChannel with %s reported an error: %v", ni.instanceIndex, remotePeer, err)
	})
}

// registerWebRTCDataChannel stores a newly-opened DataChannel connection, already wired,
// and monitors the state of its peer connection. conn also carries what the signalling
// exchange said. The table has one entry per peer: a DataChannel already there is replaced
// and its peer connection is closed, since a new exchange was made only because one of
// the two sides no longer trusted it.
func registerWebRTCDataChannel(ni *NodeInstance, conn *WebRTCConn) {
	remotePeer, dc, pc := conn.remotePeer, conn.dc, conn.pc
	conn.openedAt = ni.now().UnixNano()

	ni.webrtcMutex.Lock()
	replaced := ni.webrtcConnections[remotePeer]
	ni.webrtcConnections[remotePeer] = conn
	ni.webrtcMutex.Unlock()

	logger.Infof("[GO] 🔗 Instance %d: WebRTC DataChannel open with %s (label=%s, session frames: %v)",
		ni.instanceIndex, remotePeer, dc.Label(), conn.peerSessionFrames)
	stage := conn.stage
	if stage == "" {
		stage = "none"
	}
	logger.Infof("[GO] 🧭 Instance %d: WebRTC path with %s: %s, stage of the offer: %s",
		ni.instanceIndex, remotePeer, webRTCPath(pc), stage)
	if replaced != nil {
		logger.Infof("[GO] ♻️ Instance %d: It replaces a WebRTC DataChannel with %s, whose connection is closed (stage of the offer that replaces it: %s)",
			ni.instanceIndex, remotePeer, stage)
		if replaced.dc != nil {
			replaced.dc.Close()
		}
		if replaced.pc != nil {
			replaced.pc.Close()
		}
	}

	// Clean up when the peer connection dies.
	pc.OnConnectionStateChange(func(state pwebrtc.PeerConnectionState) {
		logger.Infof("[GO] 🔄 Instance %d: WebRTC state with %s → %s",
			ni.instanceIndex, remotePeer, state)
		switch state {
		case pwebrtc.PeerConnectionStateFailed,
			pwebrtc.PeerConnectionStateClosed:
			// The table has one entry per peer, and a newer DataChannel to the same peer
			// may have taken it, or a disconnect may have removed it already: a
			// connection that ends removes the entry only if it is its own, and leaves
			// the peer alone otherwise.
			ni.webrtcMutex.Lock()
			owned := ni.webrtcConnections[remotePeer] == conn
			if owned {
				delete(ni.webrtcConnections, remotePeer)
			}
			ni.webrtcMutex.Unlock()
			if !owned {
				logger.Infof("[GO] 🗑️ Instance %d: A WebRTC connection with %s that no longer held the entry of the peer ended (%s)",
					ni.instanceIndex, remotePeer, state)
				return
			}
			logger.Infof("[GO] 🗑️ Instance %d: WebRTC connection with %s removed (%s)",
				ni.instanceIndex, remotePeer, state)

			// If the underlying libp2p connection also died, we can prune this peer entirely (we can assume it's a friendly peer).
			if ni.host != nil {
				if len(ni.host.Network().ConnsToPeer(remotePeer)) == 0 {
					logger.Infof("[GO] 🧹 Instance %d: WebRTC died and no libp2p connections remain for %s. Pruning peer.", ni.instanceIndex, remotePeer)
					ni.peersMutex.Lock()
					delete(ni.friendlyPeers, remotePeer)
					ni.peersMutex.Unlock()

					ni.streamsMutex.Lock()
					if ps, ok := ni.persistentChatStreams[remotePeer]; ok {
						_ = ps.s.Close()
						delete(ni.persistentChatStreams, remotePeer)
					}
					ni.streamsMutex.Unlock()
				}
			}
		case pwebrtc.PeerConnectionStateDisconnected:
			logger.Warnf("[GO] ⚠️ Instance %d: WebRTC connection with %s is temporarily disconnected. Waiting for recovery...",
				ni.instanceIndex, remotePeer)
		}
	})
}

// webRTCPath says, for the log, what the candidate pair of a peer connection runs on:
// "direct" or "turn", with the types of the two candidates.
func webRTCPath(pc *pwebrtc.PeerConnection) string {
	sctp := pc.SCTP()
	if sctp == nil || sctp.Transport() == nil || sctp.Transport().ICETransport() == nil {
		return "unknown"
	}
	pair, err := sctp.Transport().ICETransport().GetSelectedCandidatePair()
	if err != nil || pair == nil {
		return "unknown"
	}
	path := "direct"
	if pair.Local.Typ == pwebrtc.ICECandidateTypeRelay || pair.Remote.Typ == pwebrtc.ICECandidateTypeRelay {
		path = "turn"
	}
	return fmt.Sprintf("%s (local %s, remote %s)", path, pair.Local.Typ, pair.Remote.Typ)
}

// writeToWebRTCDataChannel sends a framed message over an open DataChannel.
// Large frames are transparently split into chunks of at most WebRTCMaxChunkSize
// bytes (including the 1-byte flags header) so that every dc.Send() call stays
// within the browser SCTP message-size limit (~64 KB).
// The mutex ensures that concurrent callers cannot interleave their chunks.
func writeToWebRTCDataChannel(conn *WebRTCConn, channel string, payload []byte) error {
	frame, err := buildDirectMessageFrame(channel, payload)
	if err != nil {
		return err
	}

	conn.sendMu.Lock()
	defer conn.sendMu.Unlock()

	// Fast path: the whole frame fits in a single chunk.
	if len(frame) <= webRTCMaxChunkPayload {
		chunk := make([]byte, 1+len(frame))
		chunk[0] = webRTCChunkFlagStartEnd
		copy(chunk[1:], frame)
		return conn.dc.Send(chunk)
	}

	// Slow path: split the frame across multiple chunks.
	for offset := 0; offset < len(frame); {
		end := offset + webRTCMaxChunkPayload
		if end > len(frame) {
			end = len(frame)
		}

		var flags byte
		if offset == 0 {
			flags |= webRTCChunkFlagStart
		}
		if end == len(frame) {
			flags |= webRTCChunkFlagEnd
		}
		// else middle chunk uses flag 0x00 to indicate "continuation"

		chunk := make([]byte, 1+(end-offset))
		chunk[0] = flags
		copy(chunk[1:], frame[offset:end])

		if err := conn.dc.Send(chunk); err != nil {
			return fmt.Errorf("chunk send failed at offset %d: %w", offset, err)
		}
		offset = end
	}
	return nil
}

// setupWebRTCSignalHandler registers the answerer-side stream handler for
// /unaiverse/webrtc-signal/1.0.0. Call this once after the libp2p host is created.
func setupWebRTCSignalHandler(ni *NodeInstance) {
	ni.host.SetStreamHandler(UnaiverseWebRTCSignalProtocol, func(s network.Stream) {
		go handleSignalingStream(ni, s)
	})
	logger.Infof("[GO] ✅ Instance %d: WebRTC signaling handler registered.", ni.instanceIndex)
}

// handleSignalingStream is the answerer path. It is invoked when a remote peer
// opens a /unaiverse/webrtc-signal/1.0.0 stream to us.
//
// Sequence (vanilla / batch ICE – full gathering before transmission):
//
//  1. Read SDP offer from the stream.
//  2. Create PeerConnection, set remote description.
//  3. Create SDP answer, set local description, wait for ICE gathering.
//  4. Send complete SDP answer (with all ICE candidates embedded).
//  5. Wait for the offerer's DataChannel to open, then register it.
func handleSignalingStream(ni *NodeInstance, s network.Stream) {
	defer recoverGoroutine("handleSignalingStream")
	defer s.Close()

	remotePeer := s.Conn().RemotePeer()
	ni.webrtcMutex.RLock()
	existing := ni.webrtcConnections[remotePeer]
	ni.webrtcMutex.RUnlock()

	// An offer is never refused because we hold a DataChannel to the peer. A side offers
	// only when it holds none that it trusts, so the offer says that ours is dead on the
	// other side, whatever its state says here. The new channel takes the entry when it
	// opens (registerWebRTCDataChannel); until then the old one stays.
	if existing != nil && existing.isOpen() {
		logger.Infof("[GO] 📶 Instance %d: Incoming WebRTC signaling from %s over an open DataChannel, which it will replace", ni.instanceIndex, remotePeer)
	} else {
		logger.Infof("[GO] 📶 Instance %d: Incoming WebRTC signaling from %s", ni.instanceIndex, remotePeer)
	}

	ctx, cancel := context.WithTimeout(ni.ctx, WebRTCSignalingTimeout)
	defer cancel()

	_ = s.SetDeadline(time.Now().Add(WebRTCSignalingTimeout))

	// Read the SDP offer
	offerMsg, err := readSignalMessage(s)
	if err != nil {
		logger.Errorf("[GO] ❌ Instance %d: Reading offer from %s: %v", ni.instanceIndex, remotePeer, err)
		return
	}
	if offerMsg.Type == "error" {
		logger.Errorf("[GO] ❌ Instance %d: Remote signaling error from %s: %s",
			ni.instanceIndex, remotePeer, offerMsg.Message)
		return
	}
	if offerMsg.Type != "offer" {
		logger.Errorf("[GO] ❌ Instance %d: Expected offer from %s, got %q",
			ni.instanceIndex, remotePeer, offerMsg.Type)
		writeSignalMessage(s, SignalMessage{Type: "error", Message: "expected offer"})
		return
	}
	logger.Debugf("[GO] 📥 Instance %d: Successfully received SDP offer from %s (length: %d bytes)", ni.instanceIndex, remotePeer, len(offerMsg.SDP))

	// Create PeerConnection & set remote description using custom API. The offerer chooses
	// the stage: for a direct-stage offer this side leaves TURN out as well, since a relay
	// pair needs an allocation on one side only.
	pc, err := pwebrtc.NewPeerConnection(getWebRTCConfig(ni, offerMsg.Stage))
	if err != nil {
		logger.Errorf("[GO] ❌ Instance %d: NewPeerConnection: %v", ni.instanceIndex, err)
		writeSignalMessage(s, SignalMessage{Type: "error", Message: err.Error()})
		return
	}

	// Capture the DataChannel that the offerer will create. Its handlers are installed
	// here: pion waits for this callback before it opens the channel.
	direction, remoteAddr := connDirection(s), s.Conn().RemoteMultiaddr()
	dcReady := make(chan *WebRTCConn, 1)
	pc.OnDataChannel(func(dc *pwebrtc.DataChannel) {
		logger.Debugf("[GO] 🔌 Instance %d: Remote peer %s created DataChannel '%s'", ni.instanceIndex, remotePeer, dc.Label())
		if dc.Label() == WebRTCDataChannelLabel {
			conn := &WebRTCConn{
				pc: pc, dc: dc, remotePeer: remotePeer,
				peerSessionFrames: offerMsg.SessionFrames,
				stage:             offerMsg.Stage,
				direction:         direction,
				remoteAddr:        remoteAddr,
			}
			wireWebRTCDataChannel(ni, conn)
			dc.OnOpen(func() {
				logger.Debugf("[GO] 🟢 Instance %d: DataChannel '%s' with %s is now OPEN", ni.instanceIndex, dc.Label(), remotePeer)
				select {
				case dcReady <- conn:
				default:
				}
			})
		}
	})

	offer := pwebrtc.SessionDescription{Type: pwebrtc.SDPTypeOffer, SDP: offerMsg.SDP}
	if err := pc.SetRemoteDescription(offer); err != nil {
		logger.Errorf("[GO] ❌ Instance %d: SetRemoteDescription: %v", ni.instanceIndex, err)
		writeSignalMessage(s, SignalMessage{Type: "error", Message: err.Error()})
		pc.Close()
		return
	}

	// Create answer, gather ICE
	answer, err := pc.CreateAnswer(nil)
	if err != nil {
		logger.Errorf("[GO] ❌ Instance %d: CreateAnswer: %v", ni.instanceIndex, err)
		writeSignalMessage(s, SignalMessage{Type: "error", Message: err.Error()})
		pc.Close()
		return
	}

	// The candidates are watched from BEFORE SetLocalDescription.
	gatherStart := time.Now()
	candidatesReady := watchICECandidates(pc, offerMsg.Stage)

	if err := pc.SetLocalDescription(answer); err != nil {
		logger.Errorf("[GO] ❌ Instance %d: SetLocalDescription: %v", ni.instanceIndex, err)
		writeSignalMessage(s, SignalMessage{Type: "error", Message: err.Error()})
		pc.Close()
		return
	}

	logger.Debugf("[GO] ⏳ Instance %d: Waiting for ICE candidates (answerer)...", ni.instanceIndex)
	reason, ok := candidatesReady(ctx)
	if !ok {
		logger.Errorf("[GO] ❌ Instance %d: Timeout during ICE gathering (answerer)", ni.instanceIndex)
		writeSignalMessage(s, SignalMessage{Type: "error", Message: "ICE gathering timeout"})
		pc.Close()
		return
	}
	logger.Infof("[GO] 🧊 Instance %d: ICE candidates of the answer to %s ready after %s (%s)",
		ni.instanceIndex, remotePeer, time.Since(gatherStart).Round(time.Millisecond), reason)

	// Send the SDP answer, with the candidates gathered so far
	answerMsg := SignalMessage{Type: "answer", SDP: pc.LocalDescription().SDP, SessionFrames: true}
	if err := writeSignalMessage(s, answerMsg); err != nil {
		logger.Errorf("[GO] ❌ Instance %d: Sending answer to %s: %v", ni.instanceIndex, remotePeer, err)
		pc.Close()
		return
	}
	logger.Debugf("[GO] ✅ Instance %d: Sent SDP answer to %s.", ni.instanceIndex, remotePeer)

	// An offerer that stages its offers keeps the stream open until its channel is open,
	// and says so on it when it gives the offer up, before it makes the next one. Without
	// this the connection of a direct stage that failed would wait here for its whole time.
	// The end of the stream says nothing: an offerer may close it as soon as it has the answer.
	aborted := make(chan SignalMessage, 1)
	go func() {
		defer recoverGoroutine("handleSignalingStream.abort")
		if msg, err := readSignalMessage(s); err == nil && (msg.Type == "abort" || msg.Type == "error") {
			aborted <- msg
		}
	}()

	// Wait for DataChannel to open
	select {
	case conn := <-dcReady:
		registerWebRTCDataChannel(ni, conn)
	case msg := <-aborted:
		logger.Infof("[GO] ↪️ Instance %d: %s gave up its WebRTC offer (stage: %s): %s",
			ni.instanceIndex, remotePeer, msg.Stage, msg.Message)
		pc.Close()
	case <-ctx.Done():
		logger.Errorf("[GO] ❌ Instance %d: Timeout waiting for DataChannel from %s",
			ni.instanceIndex, remotePeer)
		pc.Close()
	}
}

// errSignalingStreamEnded says that the answerer closed the signalling stream with no
// answer: what a browser does to an offer that finds its previous exchange still running.
var errSignalingStreamEnded = errors.New("the answerer closed the signaling stream with no answer")

// trustsWebRTCDataChannel reports whether we hold a DataChannel to the peer that makes an
// offer of ours pointless: open and, for a peer that takes Session frames on it, fresh.
// With such a peer a channel that is alive is never silent for long. For a peer that takes
// none, nothing is due on an idle channel and silence says nothing: the state decides.
func trustsWebRTCDataChannel(ni *NodeInstance, remotePeer peer.ID) bool {
	ni.webrtcMutex.RLock()
	existing := ni.webrtcConnections[remotePeer]
	ni.webrtcMutex.RUnlock()
	return existing != nil && existing.isOpen() && (!existing.peerSessionFrames || existing.isFresh(ni.now()))
}

// initiateWebRTCConnection is the offerer path. Call this when we want to
// establish a direct WebRTC DataChannel to a peer we are already connected to
// (typically via a relay).
//
// The offer is made in two stages, as the browser makes its own. The direct stage has
// STUN alone on both sides and WebRTCDirectStageTimeout in all. Only when it fails, and
// the peer has gained no direct path meanwhile, the TURN stage offers again with the
// whole ICE list.
func initiateWebRTCConnection(ni *NodeInstance, remotePeer peer.ID) error {
	logger.Infof("[GO] 🚀 Instance %d: Initiating WebRTC connection to %s", ni.instanceIndex, remotePeer)

	// Skip only over a DataChannel that we trust. One that is open and not trusted is
	// offered over, and the new channel replaces it when it opens.
	if trustsWebRTCDataChannel(ni, remotePeer) {
		return fmt.Errorf("already have an open WebRTC DataChannel with %s", remotePeer)
	}
	ni.webrtcMutex.RLock()
	existing := ni.webrtcConnections[remotePeer]
	ni.webrtcMutex.RUnlock()
	if existing != nil && existing.isOpen() {
		logger.Infof("[GO] ♻️ Instance %d: The WebRTC DataChannel with %s is open and gave no sign of life for %v: offering a new one",
			ni.instanceIndex, remotePeer, WebRTCFreshnessWindow)
	}

	directErr := offerWebRTC(ni, remotePeer, WebRTCStageDirect, WebRTCDirectStageTimeout)
	if directErr == nil {
		return nil
	}
	logger.Infof("[GO] ↪️ Instance %d: The direct stage of the WebRTC offer to %s failed: %v", ni.instanceIndex, remotePeer, directErr)

	for attempt, pause := 1, WebRTCStagePause; ; attempt, pause = attempt+1, WebRTCStageRetryDelay {
		select {
		case <-time.After(pause):
		case <-ni.ctx.Done():
			return ni.ctx.Err()
		}
		// The TURN stage is an offer like any other: it is not made when a path that is
		// not relayed appeared meanwhile, a direct libp2p connection or a DataChannel we
		// trust (one the peer offered, or the old one that gave a sign of life). Over an
		// entry that is still open and not trusted it is made, and its channel replaces
		// the entry although it may run through TURN.
		if bestConnRank(ni, remotePeer) == connRankDirect || trustsWebRTCDataChannel(ni, remotePeer) {
			logger.Infof("[GO] ✅ Instance %d: %s has a path that is not relayed after the direct stage: no TURN stage", ni.instanceIndex, remotePeer)
			return nil
		}
		logger.Infof("[GO] ↪️ Instance %d: TURN stage of the WebRTC offer to %s (attempt %d)", ni.instanceIndex, remotePeer, attempt)
		err := offerWebRTC(ni, remotePeer, WebRTCStageTURN, WebRTCSignalingTimeout)
		if err == nil {
			return nil
		}
		if attempt == 2 || !errors.Is(err, errSignalingStreamEnded) {
			return fmt.Errorf("TURN stage: %w (direct stage: %v)", err, directErr)
		}
	}
}

// offerWebRTC makes one offer to the peer, of the given stage and within the given time.
//
// Sequence (vanilla / batch ICE):
//
//  1. Open /unaiverse/webrtc-signal/1.0.0 stream to remotePeer.
//  2. Create PeerConnection + DataChannel, create SDP offer, wait for ICE gathering.
//  3. Send complete SDP offer (with all ICE candidates embedded).
//  4. Read SDP answer, set remote description.
//  5. Wait for DataChannel to open, then register it.
func offerWebRTC(ni *NodeInstance, remotePeer peer.ID, stage string, budget time.Duration) (err error) {
	ctx, cancel := context.WithTimeout(ni.ctx, budget)
	defer cancel()

	// Open signaling stream over existing (relayed) connection
	s, err := ni.host.NewStream(
		network.WithAllowLimitedConn(ctx, UnaiverseWebRTCSignalProtocol),
		remotePeer,
		UnaiverseWebRTCSignalProtocol,
	)
	if err != nil {
		return fmt.Errorf("open signaling stream to %s: %w", remotePeer, err)
	}
	defer s.Close()
	_ = s.SetDeadline(time.Now().Add(budget))

	// Create PeerConnection & set remote description using custom API
	pc, err := pwebrtc.NewPeerConnection(getWebRTCConfig(ni, stage))
	if err != nil {
		return fmt.Errorf("NewPeerConnection: %w", err)
	}
	// An offer that is given up is told to the answerer, which is waiting for our
	// DataChannel: a browser refuses the offer of the next stage until it knows.
	offerSent := false
	defer func() {
		if err == nil {
			return
		}
		if offerSent {
			_ = s.SetWriteDeadline(time.Now().Add(WebRTCAbortWriteTimeout))
			_ = writeSignalMessage(s, SignalMessage{Type: "abort", Stage: stage, Message: err.Error()})
		}
		pc.Close()
	}()

	ordered := true
	dc, err := pc.CreateDataChannel(WebRTCDataChannelLabel, &pwebrtc.DataChannelInit{Ordered: &ordered})
	if err != nil {
		return fmt.Errorf("CreateDataChannel: %w", err)
	}

	// The handlers go in before the channel can open (see wireWebRTCDataChannel).
	conn := &WebRTCConn{
		pc: pc, dc: dc, remotePeer: remotePeer,
		stage:      stage,
		direction:  connDirection(s),
		remoteAddr: s.Conn().RemoteMultiaddr(),
	}
	wireWebRTCDataChannel(ni, conn)

	// Channel closed when DC transitions to open.
	dcOpen := make(chan struct{}, 1)
	dc.OnOpen(func() {
		logger.Debugf("[GO] 🟢 Instance %d: Local DataChannel '%s' with %s is now OPEN", ni.instanceIndex, WebRTCDataChannelLabel, remotePeer)
		
		select {
		case dcOpen <- struct{}{}:
		default:
		}
	})

	// The candidates are watched from BEFORE SetLocalDescription.
	gatherStart := time.Now()
	candidatesReady := watchICECandidates(pc, stage)

	offer, err := pc.CreateOffer(nil)
	if err != nil {
		return fmt.Errorf("CreateOffer: %w", err)
	}
	if err := pc.SetLocalDescription(offer); err != nil {
		return fmt.Errorf("SetLocalDescription: %w", err)
	}

	logger.Debugf("[GO] ⏳ Instance %d: Waiting for ICE candidates (offerer)...", ni.instanceIndex)
	reason, ok := candidatesReady(ctx)
	if !ok {
		return fmt.Errorf("ICE gathering timeout for %s", remotePeer)
	}
	logger.Infof("[GO] 🧊 Instance %d: ICE candidates of the offer to %s ready after %s (%s)",
		ni.instanceIndex, remotePeer, time.Since(gatherStart).Round(time.Millisecond), reason)

	// Send the SDP offer, with the candidates gathered so far
	offerMsg := SignalMessage{Type: "offer", Stage: stage, SDP: pc.LocalDescription().SDP, SessionFrames: true}
	if err := writeSignalMessage(s, offerMsg); err != nil {
		return fmt.Errorf("send offer to %s: %w", remotePeer, signalingStreamError(err))
	}
	offerSent = true
	logger.Debugf("[GO] ✅ Instance %d: Sent SDP offer to %s (stage: %s).", ni.instanceIndex, remotePeer, stage)

	// Read SDP answer
	answerMsg, err := readSignalMessage(s)
	if err != nil {
		return fmt.Errorf("read answer from %s: %w", remotePeer, signalingStreamError(err))
	}
	if answerMsg.Type == "error" {
		return fmt.Errorf("remote signaling error from %s: %s", remotePeer, answerMsg.Message)
	}
	if answerMsg.Type != "answer" {
		return fmt.Errorf("expected answer from %s, got %q", remotePeer, answerMsg.Type)
	}
	logger.Debugf("[GO] 📥 Instance %d: Successfully received SDP answer from %s (length: %d bytes)", ni.instanceIndex, remotePeer, len(answerMsg.SDP))

	conn.peerSessionFrames = answerMsg.SessionFrames
	answer := pwebrtc.SessionDescription{Type: pwebrtc.SDPTypeAnswer, SDP: answerMsg.SDP}
	if err := pc.SetRemoteDescription(answer); err != nil {
		return fmt.Errorf("SetRemoteDescription from %s: %w", remotePeer, err)
	}

	// Wait for DataChannel to open
	select {
	case <-dcOpen:
		registerWebRTCDataChannel(ni, conn)
		logger.Infof("[GO] 🎉 Instance %d: WebRTC DataChannel established with %s",
			ni.instanceIndex, remotePeer)
		return nil
	case <-ctx.Done():
		return fmt.Errorf("timeout waiting for DataChannel to open with %s", remotePeer)
	}
}

// signalingStreamError marks the error of a signalling stream that the answerer ended.
func signalingStreamError(err error) error {
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, network.ErrReset) {
		return fmt.Errorf("%w (%v)", errSignalingStreamEnded, err)
	}
	return err
}
