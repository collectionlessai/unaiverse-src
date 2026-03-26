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
	"context"
	"encoding/json"
	"encoding/binary"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/network"
	pwebrtc "github.com/pion/webrtc/v4"
)

// getWebRTCConfig builds a pion WebRTC configuration from the node's stored ICE config.
// Falls back to Google's public STUN servers when no config is present.
func getWebRTCConfig(ni *NodeInstance) pwebrtc.Configuration {
	if ni.iceConfig != nil && (len(ni.iceConfig.STUNServers) > 0 || len(ni.iceConfig.TURNServers) > 0) {
		var STUNServers []pwebrtc.ICEServer
		if len(ni.iceConfig.STUNServers) > 0 {
			STUNServers = append(STUNServers, pwebrtc.ICEServer{URLs: ni.iceConfig.STUNServers})
		}
		for _, t := range ni.iceConfig.TURNServers {
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

// registerWebRTCDataChannel stores a newly-opened DataChannel connection and
// wires up message delivery + connection state monitoring.
func registerWebRTCDataChannel(ni *NodeInstance, remotePeer peer.ID, dc *pwebrtc.DataChannel, pc *pwebrtc.PeerConnection) {
	conn := &WebRTCConn{pc: pc, dc: dc, remotePeer: remotePeer}

	ni.webrtcMutex.Lock()
	ni.webrtcConnections[remotePeer] = conn
	ni.webrtcMutex.Unlock()

	logger.Infof("[GO] 🔗 Instance %d: WebRTC DataChannel open with %s (label=%s)",
		ni.instanceIndex, remotePeer, dc.Label())

	// Deliver incoming DataChannel messages into the shared message queue.
	dc.OnMessage(func(msg pwebrtc.DataChannelMessage) {
		ch, payload, err := parseDirectMessageFrame(msg.Data)
		if err != nil {
			logger.Warnf("[GO] ⚠️ Instance %d: Bad DataChannel frame from %s: %v",
				ni.instanceIndex, remotePeer, err)
			return
		}
		storeReceivedMessage(ni, remotePeer, ch, payload)
	})

	// Clean up when the peer connection dies.
	pc.OnConnectionStateChange(func(state pwebrtc.PeerConnectionState) {
		logger.Debugf("[GO] 🔄 Instance %d: WebRTC state with %s → %s",
			ni.instanceIndex, remotePeer, state)
		switch state {
		case pwebrtc.PeerConnectionStateFailed,
			pwebrtc.PeerConnectionStateClosed,
			pwebrtc.PeerConnectionStateDisconnected:
			ni.webrtcMutex.Lock()
			delete(ni.webrtcConnections, remotePeer)
			ni.webrtcMutex.Unlock()
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
					if stream, ok := ni.persistentChatStreams[remotePeer]; ok {
						_ = stream.Close()
						delete(ni.persistentChatStreams, remotePeer)
					}
					ni.streamsMutex.Unlock()
				}
			}
		}
	})
}

// writeToWebRTCDataChannel sends a framed message over an open DataChannel.
func writeToWebRTCDataChannel(conn *WebRTCConn, channel string, payload []byte) error {
	frame, err := buildDirectMessageFrame(channel, payload)
	if err != nil {
		return err
	}
	return conn.dc.Send(frame)
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
	defer s.Close()

	remotePeer := s.Conn().RemotePeer()
	ni.webrtcMutex.RLock()
	existing, exists := ni.webrtcConnections[remotePeer]
	ni.webrtcMutex.RUnlock()

	if exists {
		if existing.dc != nil {
			if existing.dc.ReadyState() == pwebrtc.DataChannelStateOpen {
				logger.Warnf("[GO] ⚠️ Instance %d: Rejecting redundant WebRTC signaling from %s (already connected)", ni.instanceIndex, remotePeer)
				return
			}
		}
	}
	logger.Infof("[GO] 📶 Instance %d: Incoming WebRTC signaling from %s", ni.instanceIndex, remotePeer)

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

	// Create PeerConnection & set remote description
	pc, err := pwebrtc.NewPeerConnection(getWebRTCConfig(ni))
	if err != nil {
		logger.Errorf("[GO] ❌ Instance %d: NewPeerConnection: %v", ni.instanceIndex, err)
		writeSignalMessage(s, SignalMessage{Type: "error", Message: err.Error()})
		return
	}

	// Capture the DataChannel that the offerer will create.
	dcReady := make(chan *pwebrtc.DataChannel, 1)
	pc.OnDataChannel(func(dc *pwebrtc.DataChannel) {
		if dc.Label() == WebRTCDataChannelLabel {
			dc.OnOpen(func() {
				select {
				case dcReady <- dc:
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

	// Register gathering-complete promise BEFORE SetLocalDescription.
	gatherDone := pwebrtc.GatheringCompletePromise(pc)

	if err := pc.SetLocalDescription(answer); err != nil {
		logger.Errorf("[GO] ❌ Instance %d: SetLocalDescription: %v", ni.instanceIndex, err)
		writeSignalMessage(s, SignalMessage{Type: "error", Message: err.Error()})
		pc.Close()
		return
	}

	logger.Debugf("[GO] ⏳ Instance %d: Waiting for ICE gathering (answerer)...", ni.instanceIndex)
	select {
	case <-gatherDone:
		logger.Debugf("[GO] ✅ Instance %d: ICE gathering complete (answerer).", ni.instanceIndex)
	case <-ctx.Done():
		logger.Errorf("[GO] ❌ Instance %d: Timeout during ICE gathering (answerer)", ni.instanceIndex)
		writeSignalMessage(s, SignalMessage{Type: "error", Message: "ICE gathering timeout"})
		pc.Close()
		return
	}

	// Send complete SDP answer
	if err := writeSignalMessage(s, SignalMessage{Type: "answer", SDP: pc.LocalDescription().SDP}); err != nil {
		logger.Errorf("[GO] ❌ Instance %d: Sending answer to %s: %v", ni.instanceIndex, remotePeer, err)
		pc.Close()
		return
	}
	logger.Debugf("[GO] ✅ Instance %d: Sent SDP answer to %s.", ni.instanceIndex, remotePeer)

	// Wait for DataChannel to open
	select {
	case dc := <-dcReady:
		registerWebRTCDataChannel(ni, remotePeer, dc, pc)
	case <-ctx.Done():
		logger.Errorf("[GO] ❌ Instance %d: Timeout waiting for DataChannel from %s",
			ni.instanceIndex, remotePeer)
		pc.Close()
	}
}

// initiateWebRTCConnection is the offerer path. Call this when we want to
// establish a direct WebRTC DataChannel to a peer we are already connected to
// (typically via a relay).
//
// Sequence (vanilla / batch ICE):
//
//  1. Open /unaiverse/webrtc-signal/1.0.0 stream to remotePeer.
//  2. Create PeerConnection + DataChannel, create SDP offer, wait for ICE gathering.
//  3. Send complete SDP offer (with all ICE candidates embedded).
//  4. Read SDP answer, set remote description.
//  5. Wait for DataChannel to open, then register it.
func initiateWebRTCConnection(ni *NodeInstance, remotePeer peer.ID) error {
	logger.Infof("[GO] 🚀 Instance %d: Initiating WebRTC connection to %s", ni.instanceIndex, remotePeer)

	// Skip if a healthy DataChannel already exists.
	ni.webrtcMutex.RLock()
	existing, exists := ni.webrtcConnections[remotePeer]
	ni.webrtcMutex.RUnlock()
	if exists && existing.dc != nil && existing.dc.ReadyState() == pwebrtc.DataChannelStateOpen {
		return fmt.Errorf("already have an open WebRTC DataChannel with %s", remotePeer)
	}

	ctx, cancel := context.WithTimeout(ni.ctx, WebRTCSignalingTimeout)
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
	_ = s.SetDeadline(time.Now().Add(WebRTCSignalingTimeout))

	// Create PeerConnection + DataChannel
	pc, err := pwebrtc.NewPeerConnection(getWebRTCConfig(ni))
	if err != nil {
		return fmt.Errorf("NewPeerConnection: %w", err)
	}

	ordered := true
	dc, err := pc.CreateDataChannel(WebRTCDataChannelLabel, &pwebrtc.DataChannelInit{Ordered: &ordered})
	if err != nil {
		pc.Close()
		return fmt.Errorf("CreateDataChannel: %w", err)
	}

	// Channel closed when DC transitions to open.
	dcOpen := make(chan struct{}, 1)
	dc.OnOpen(func() {
		select {
		case dcOpen <- struct{}{}:
		default:
		}
	})

	// Register gathering-complete promise BEFORE SetLocalDescription.
	gatherDone := pwebrtc.GatheringCompletePromise(pc)

	offer, err := pc.CreateOffer(nil)
	if err != nil {
		pc.Close()
		return fmt.Errorf("CreateOffer: %w", err)
	}
	if err := pc.SetLocalDescription(offer); err != nil {
		pc.Close()
		return fmt.Errorf("SetLocalDescription: %w", err)
	}

	logger.Debugf("[GO] ⏳ Instance %d: Waiting for ICE gathering (offerer)...", ni.instanceIndex)
	select {
	case <-gatherDone:
		logger.Debugf("[GO] ✅ Instance %d: ICE gathering complete (offerer).", ni.instanceIndex)
	case <-ctx.Done():
		pc.Close()
		return fmt.Errorf("ICE gathering timeout for %s", remotePeer)
	}

	// Send complete SDP offer
	if err := writeSignalMessage(s, SignalMessage{Type: "offer", SDP: pc.LocalDescription().SDP}); err != nil {
		pc.Close()
		return fmt.Errorf("send offer to %s: %w", remotePeer, err)
	}
	logger.Debugf("[GO] ✅ Instance %d: Sent SDP offer to %s.", ni.instanceIndex, remotePeer)

	// Read SDP answer
	answerMsg, err := readSignalMessage(s)
	if err != nil {
		pc.Close()
		return fmt.Errorf("read answer from %s: %w", remotePeer, err)
	}
	if answerMsg.Type == "error" {
		pc.Close()
		return fmt.Errorf("remote signaling error from %s: %s", remotePeer, answerMsg.Message)
	}
	if answerMsg.Type != "answer" {
		pc.Close()
		return fmt.Errorf("expected answer from %s, got %q", remotePeer, answerMsg.Type)
	}

	answer := pwebrtc.SessionDescription{Type: pwebrtc.SDPTypeAnswer, SDP: answerMsg.SDP}
	if err := pc.SetRemoteDescription(answer); err != nil {
		pc.Close()
		return fmt.Errorf("SetRemoteDescription from %s: %w", remotePeer, err)
	}

	// Wait for DataChannel to open
	select {
	case <-dcOpen:
		registerWebRTCDataChannel(ni, remotePeer, dc, pc)
		logger.Infof("[GO] 🎉 Instance %d: WebRTC DataChannel established with %s",
			ni.instanceIndex, remotePeer)
		return nil
	case <-ctx.Done():
		pc.Close()
		return fmt.Errorf("timeout waiting for DataChannel to open with %s", remotePeer)
	}
}
