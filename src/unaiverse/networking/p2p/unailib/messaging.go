// Here we define stream handling and queue managing logic for the P2P library.
package main

import (
	"io"
	"fmt"
	"net"
	"time"
	"bytes"
	"errors"
	"context"
	"strings"
	"encoding/json"
	"container/list"
	"encoding/binary"
	"google.golang.org/protobuf/proto"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/network"
	libp2pproto "github.com/libp2p/go-libp2p/core/protocol"
	ma "github.com/multiformats/go-multiaddr"
	pg "unailib/proto-go"
	pubsub "github.com/libp2p/go-libp2p-pubsub"
)

// storeReceivedMessage processes a raw message received either from a direct stream
// or a PubSub topic. The libp2p protocol ID is preserved on the queued message so
// Python can route by protocol instead of by content_type. Pass an empty string
// for protocolID if not available.
func storeReceivedMessage(
    ni *NodeInstance,
    from peer.ID,
    channel string,
    protocolID string,
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
		From:     from,
		Data:     data,
		Protocol: protocolID,
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
	defer recoverGoroutine("readFromSubscription")
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
			storeReceivedMessage(ni, msg.GetFrom(), topic, "", msg.Data)  // Empty protocolID for PubSub messages, since they are identified by topic instead.
		}
	}
}

// promoteFriendlyPeer records (or address-refreshes) a peer in friendlyPeers,
// the map GetConnectedPeers reports from. Chat streams have always done this on
// their first message (see handleStream); Session-protocol traffic must do it
// too — on both the sending and the receiving side — otherwise a peer whose
// only traffic is the Session handshake/keep-alive never surfaces to the
// application connection pools.
func promoteFriendlyPeer(ni *NodeInstance, peerID peer.ID, direction string, fallbackAddr ma.Multiaddr) {
	ni.peersMutex.Lock()
	defer ni.peersMutex.Unlock()
	knownAddrs := ni.host.Peerstore().Addrs(peerID)
	if len(knownAddrs) == 0 && fallbackAddr != nil {
		knownAddrs = []ma.Multiaddr{fallbackAddr}
	}
	if existing, ok := ni.friendlyPeers[peerID]; ok {
		// Keep ConnectedAt and Direction from the original first contact.
		existing.Addrs = knownAddrs
		ni.friendlyPeers[peerID] = existing
		return
	}
	ni.friendlyPeers[peerID] = ExtendedPeerInfo{
		ID:          peerID,
		Addrs:       knownAddrs,
		ConnectedAt: time.Now(),
		Direction:   direction,
		Relayed:     false,
	}
	logger.Infof("[GO] ➕ Instance %d: Peer %s promoted to App Peer (%s).", ni.instanceIndex, peerID, direction)
}

// connDirection maps a stream's underlying connection direction to the
// "inbound"/"outbound" labels the Python pool routing expects.
func connDirection(s network.Stream) string {
	if s.Conn().Stat().Direction == network.DirOutbound {
		return "outbound"
	}
	return "inbound"
}

// handleStream reads from a direct message stream on the UnaiverseChatProtocol using the new framing protocol.
// It expects the stream to start with a 4-byte length prefix, followed by a 1-byte channel name length,
// the channel name itself, and finally the Protobuf-encoded payload.
func handleStream(ni *NodeInstance, s network.Stream) {
	defer recoverGoroutine("handleStream")
	senderPeerID := s.Conn().RemotePeer()
	streamID := s.ID()
	// Delegate peer promotion and address updates to the helper function.
	// This safely handles the mutex locking and state management for us.
	promoteFriendlyPeer(ni, senderPeerID, connDirection(s), s.Conn().RemoteMultiaddr())
	logger.Debugf("[GO] 📥 Instance %d: Accepted INCOMING stream %s from %s. Storing for duplex use.\n", ni.instanceIndex, streamID, senderPeerID)

	// Store the newly accepted stream so we can use it to send messages back to this peer.
	incomingPS := &persistentStream{s: s}
	ni.streamsMutex.Lock()
	ni.persistentChatStreams[senderPeerID] = incomingPS
	ni.streamsMutex.Unlock()

	// This defer block ensures cleanup happens when the stream is closed by either side.
	defer func() {
		logger.Debugf("[GO] 🧹 Instance %d: Stream %s with %s closed. Removing from map.\n", ni.instanceIndex, streamID, senderPeerID)
		ni.streamsMutex.Lock()
		if current, ok := ni.persistentChatStreams[senderPeerID]; ok && current == incomingPS {
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

		if totalLen <= uint32(channelLen) + 1 {
			logger.Errorf("[GO] ❌ Instance %d: Total length %d too short for channel header on Stream %s, preventing underflow. Resetting.\n", ni.instanceIndex, totalLen, streamID)
			s.Reset()
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
		storeReceivedMessage(ni, senderPeerID, channel, UnaiverseChatProtocol, payload)
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

	// Per-stage Session protocol handlers — one protobuf message per stream, no channel
	// header in the framing (unlike chat which multiplexes channels over a persistent
	// stream). See handleSessionStream below.
	for _, proto := range []string{
		SessionHelloProtocol,
		SessionProfileProtocol,
		SessionWorldProtocol,
		SessionControlProtocol,
	} {
		protoCopy := proto // capture loop variable
		ni.host.SetStreamHandler(libp2pproto.ID(protoCopy), func(s network.Stream) {
			handleSessionStream(ni, s, protoCopy)
		})
	}
}

// handleSessionStream reads a single length-prefixed protobuf payload from a Session
// stream and enqueues it for Python with the protocol ID attached. Unlike handleStream
// (chat), each Session stream carries exactly one message and is closed after read —
// no persistent stream, no per-stream channel header. Framing: 4-byte big-endian
// uint32 length, then the protobuf payload.
func handleSessionStream(ni *NodeInstance, s network.Stream, protocolID string) {
	defer recoverGoroutine("handleSessionStream")
	senderPeerID := s.Conn().RemotePeer()
	streamID := s.ID()
	defer s.Close()

	// Session traffic is application traffic: surface the peer to the pools.
	promoteFriendlyPeer(ni, senderPeerID, connDirection(s), s.Conn().RemoteMultiaddr())

	// Per-stream read budget — keeps a slow / malicious peer from holding the stream
	// open indefinitely. The Session FSM enforces its own per-stage timeouts on the
	// Python side; this is just a transport-level safety net.
	if err := s.SetReadDeadline(time.Now().Add(StreamCreationTimeout)); err != nil {
		logger.Warnf("[GO] ⚠️ Instance %d: SetReadDeadline failed on session stream %s: %v\n", ni.instanceIndex, streamID, err)
	}

	var payloadLen uint32
	if err := binary.Read(s, binary.BigEndian, &payloadLen); err != nil {
		if err != io.EOF {
			logger.Warnf("[GO] ⚠️ Instance %d: session stream %s (%s) read-length failed: %v\n", ni.instanceIndex, streamID, protocolID, err)
		}
		return
	}
	if payloadLen == 0 || payloadLen > MaxMessageSize {
		logger.Errorf("[GO] ❌ Instance %d: session stream %s (%s) bad payload length %d\n", ni.instanceIndex, streamID, protocolID, payloadLen)
		s.Reset()
		return
	}
	payload := make([]byte, payloadLen)
	if _, err := io.ReadFull(s, payload); err != nil {
		logger.Warnf("[GO] ⚠️ Instance %d: session stream %s (%s) read-payload failed: %v\n", ni.instanceIndex, streamID, protocolID, err)
		return
	}
	logger.Infof("[GO] 📨 Instance %d: Session %s from %s (%d bytes)\n", ni.instanceIndex, protocolID, senderPeerID, payloadLen)

	// Use the protocol ID as the channel key so messages are queued per-protocol.
	// Python's connpool decoder treats any channel matching a session protocol ID
	// as a session message and routes it to the SessionManager.
	storeReceivedMessage(ni, senderPeerID, protocolID, protocolID, payload)
}

// writeSessionFrame writes a single length-prefixed session payload onto a stream:
// 4-byte big-endian uint32 length, then the protobuf bytes. Returns the number of
// bytes actually written (header + payload) on success.
func writeSessionFrame(w io.Writer, payload []byte) error {
	if uint32(len(payload)) > MaxMessageSize {
		return fmt.Errorf("session payload size (%d) exceeds limit (%d)", len(payload), MaxMessageSize)
	}
	if err := binary.Write(w, binary.BigEndian, uint32(len(payload))); err != nil {
		return fmt.Errorf("failed to write session length prefix: %w", err)
	}
	if _, err := w.Write(payload); err != nil {
		return fmt.Errorf("failed to write session payload: %w", err)
	}
	return nil
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
