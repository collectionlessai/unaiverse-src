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
	ma "github.com/multiformats/go-multiaddr"
	pg "unailib/proto-go"
	pubsub "github.com/libp2p/go-libp2p-pubsub"
)

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
