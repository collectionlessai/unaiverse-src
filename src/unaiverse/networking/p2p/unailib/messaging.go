// Here we define stream handling and queue managing logic for the P2P library.
package main

import (
	"container/list"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"strings"
	"time"
	pg "unailib/proto-go"

	pubsub "github.com/libp2p/go-libp2p-pubsub"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	libp2pproto "github.com/libp2p/go-libp2p/core/protocol"
	ma "github.com/multiformats/go-multiaddr"
	"google.golang.org/protobuf/proto"
)

// isSessionProtocol reports whether protocolID is one of the per-stage Session
// handshake protocols. Session traffic is privileged in the message store so a
// flood of attacker-chosen chat channels cannot starve the handshake.
func isSessionProtocol(protocolID string) bool {
	switch protocolID {
	case SessionHelloProtocol, SessionProfileProtocol, SessionWorldProtocol, SessionControlProtocol:
		return true
	default:
		return false
	}
}

// isStreamChannel reports whether a chat/pubsub channel carries stream samples: a
// PubSub topic ("<owner>::ps:<group>"; the rendezvous topic never reaches the store),
// or a direct-message channel of content type stream_sample
// ("<sender>::dm:<receiver>-stream_sample~<group>"). The node pops these once per tick
// of its behavior (PopStreamMessages) and every other channel at every network pass
// (PopNetworkMessages), so that an agent consumes a stream at its own pace while
// everything else is read at once.
func isStreamChannel(channel string) bool {
	if strings.Contains(channel, "::ps:") {
		return true
	}
	_, rest, isDM := strings.Cut(channel, "::dm:")
	if !isDM {
		return false
	}
	_, contentType, hasType := strings.Cut(rest, "-")
	if !hasType {
		return false
	}
	contentType, _, _ = strings.Cut(contentType, "~")
	return contentType == "stream_sample"
}

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
	storeReceivedMessageWithFiles(ni, from, channel, protocolID, data, nil)
}

// storeReceivedMessageWithFiles is storeReceivedMessage for a message that may have files on
// disk (file.go): a message the store does not keep has its files removed.
func storeReceivedMessageWithFiles(
	ni *NodeInstance,
	from peer.ID,
	channel string,
	protocolID string,
	data []byte,
	files []ReceivedFile,
) {
	// Get the message store for this instance
	store := ni.messageStore
	if store == nil {
		logger.Errorf("[GO] ❌ storeReceivedMessage: Message store not initialized for instance %d\n", ni.instanceIndex)
		removeFiles(receivedFilePaths(files))
		return // Cannot process message if store is nil
	}

	// Create the minimal message envelope.
	newMessage := &QueuedMessage{
		From:     from,
		Data:     data,
		Protocol: protocolID,
		Files:    files,
	}

	// Lock the store mutex before accessing the shared maps.
	store.mu.Lock()
	defer store.mu.Unlock()

	// Check if this channel already has a message list.
	messageList, channelExists := store.messagesByChannel[channel]
	if !channelExists {
		// Session-protocol channels are privileged and bypass the channel-count
		// cap: their key is the protocol ID itself (a small fixed set), so they
		// cannot cause unbounded growth. The cap exists to contain attacker-chosen
		// chat/pubsub channel names; a flood of junk chat channels must never be
		// able to evict (by exhausting maxUniqueChannels) an incoming handshake.
		if !isSessionProtocol(protocolID) && len(store.messagesByChannel) >= maxUniqueChannels {
			logger.Warnf("[GO] 🗑️ Instance %d: Message store full. Discarding message for new channel '%s'.\n", ni.instanceIndex, channel)
			removeFiles(receivedFilePaths(files))
			return
		}
		messageList = list.New()
		store.messagesByChannel[channel] = messageList
		logger.Debugf("[GO] ✨ Instance %d: Created new channel queue '%s'. Total channels: %d\n", ni.instanceIndex, channel, len(store.messagesByChannel))
	}

	// If the channel already has a message list, check its length. A full chat/pubsub
	// queue drops its oldest message, so that what is kept is the most recent (a stream
	// that waited while an action ran resumes from fresh samples). A Session queue is
	// shared by all the peers and drained in full at every pop, so it keeps refusing the
	// newest frame instead: dropping the oldest could drop another peer's handshake.
	if messageList.Len() >= maxChannelQueueLen {
		if isSessionProtocol(channel) {
			logger.Warnf("[GO] 🗑️ Instance %d: Queue for channel '%s' full. Discarding message.\n", ni.instanceIndex, channel)
			return
		}
		if oldest := messageList.Remove(messageList.Front()).(*QueuedMessage); len(oldest.Files) > 0 {
			removeFiles(receivedFilePaths(oldest.Files))
		}
		logger.Warnf("[GO] 🗑️ Instance %d: Queue for channel '%s' full. Discarding its oldest message.\n", ni.instanceIndex, channel)
	}

	messageList.PushBack(newMessage)
	logger.Debugf("[GO] 📥 Instance %d: Queued message on channel '%s' from %s. New queue length: %d\n", ni.instanceIndex, channel, from, messageList.Len())
}

// popBatch removes and returns what one PopMessages call hands to the caller: the
// oldest message of every chat/pubsub channel, and every queued message of the
// Session-protocol channels, oldest first. The caller must hold store.mu.
//
// A Session-protocol queue is keyed by the protocol ID, so all the peers of the node
// share it. Popping one frame per call, as for the other channels, made them wait in
// line behind each other at one frame per cycle of the caller: a GOAWAY or a PONG was
// read as many cycles late as there were frames ahead of it, whoever had sent them,
// and a node with enough peers could not keep up with their keep-alives at all.
func (store *MessageStore) popBatch() []*QueuedMessage {
	var popped []*QueuedMessage
	var bytesOut int
	for channel, messageList := range store.messagesByChannel {
		drain := isSessionProtocol(channel)
		for messageList.Len() > 0 {
			element := messageList.Front()
			if len(popped) > 0 && bytesOut+len(element.Value.(*QueuedMessage).Data) > popBatchMaxBytes {
				return popped // The rest comes with the next pop (the reply has 32-bit lengths)
			}
			bytesOut += len(element.Value.(*QueuedMessage).Data)
			popped = append(popped, element.Value.(*QueuedMessage))
			messageList.Remove(element)
			if !drain {
				break
			}
		}
		// if the queue is now empty, we can delete it from the map to save space
		if messageList.Len() == 0 {
			delete(store.messagesByChannel, channel)
		}
	}
	return popped
}

// popNetworkBatch removes and returns what one PopNetworkMessages call hands to the
// caller: every queued message of every channel that is not a stream channel (the
// Session protocols included), oldest first within each channel. The caller must
// hold store.mu.
func (store *MessageStore) popNetworkBatch() []*QueuedMessage {
	var popped []*QueuedMessage
	var bytesOut int
	for channel, messageList := range store.messagesByChannel {
		if isStreamChannel(channel) {
			continue
		}
		for messageList.Len() > 0 {
			element := messageList.Front()
			if len(popped) > 0 && bytesOut+len(element.Value.(*QueuedMessage).Data) > popBatchMaxBytes {
				return popped // The rest comes with the next pop (the reply has 32-bit lengths)
			}
			bytesOut += len(element.Value.(*QueuedMessage).Data)
			popped = append(popped, element.Value.(*QueuedMessage))
			messageList.Remove(element)
		}
		delete(store.messagesByChannel, channel)
	}
	return popped
}

// popStreamBatch removes and returns what one PopStreamMessages call hands to the
// caller: the oldest message of every stream channel. The caller must hold store.mu.
func (store *MessageStore) popStreamBatch() []*QueuedMessage {
	var popped []*QueuedMessage
	var bytesOut int
	for channel, messageList := range store.messagesByChannel {
		if !isStreamChannel(channel) {
			continue
		}
		if element := messageList.Front(); element != nil {
			if len(popped) > 0 && bytesOut+len(element.Value.(*QueuedMessage).Data) > popBatchMaxBytes {
				return popped // The rest comes with the next pop (the reply has 32-bit lengths)
			}
			bytesOut += len(element.Value.(*QueuedMessage).Data)
			popped = append(popped, element.Value.(*QueuedMessage))
			messageList.Remove(element)
		}
		if messageList.Len() == 0 {
			delete(store.messagesByChannel, channel)
		}
	}
	return popped
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
	logger.Debugf("[GO] 📥 Instance %d: Accepted INCOMING stream %s from %s. Storing for duplex use.\n", ni.instanceIndex, s.ID(), senderPeerID)

	// Store the newly accepted stream so we can use it to send messages back to this peer.
	incomingPS := newPersistentStream(s)
	ni.streamsMutex.Lock()
	ni.persistentChatStreams[senderPeerID] = incomingPS
	ni.streamsMutex.Unlock()

	readChatStream(ni, incomingPS)
}

// readChatStream reads the frames of a chat stream, of either direction, until the stream
// ends, and queues them. A stream that was replaced as the one to write on is still read
// here to its end.
func readChatStream(ni *NodeInstance, ps *persistentStream) {
	defer recoverGoroutine("readChatStream")
	s := ps.s
	senderPeerID := s.Conn().RemotePeer()
	streamID := s.ID()
	// Delegate peer promotion and address updates to the helper function.
	// This safely handles the mutex locking and state management for us.
	promoteFriendlyPeer(ni, senderPeerID, connDirection(s), s.Conn().RemoteMultiaddr())

	// This defer block ensures cleanup happens when the stream is closed by either side.
	defer func() {
		logger.Debugf("[GO] 🧹 Instance %d: Stream %s with %s closed. Removing from map.\n", ni.instanceIndex, streamID, senderPeerID)
		ni.streamsMutex.Lock()
		if current, ok := ni.persistentChatStreams[senderPeerID]; ok && current == ps {
			delete(ni.persistentChatStreams, senderPeerID)
		}
		ni.streamsMutex.Unlock()
		s.Close() // Ensure the stream is fully closed.
		close(ps.done)
	}()

	for {
		// Renew the idle deadline on each frame: a peer that opens the stream and
		// then stops sending must not pin this goroutine forever (slow-loris). The
		// existing net.Error/Timeout branch below turns the expiry into a clean
		// return with stream cleanup via the defer above.
		if err := s.SetReadDeadline(time.Now().Add(ChatStreamIdleTimeout)); err != nil {
			logger.Warnf("[GO] ⚠️ Instance %d: SetReadDeadline failed on chat stream %s: %v\n", ni.instanceIndex, streamID, err)
		}

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

		// A chunk frame of a message larger than a chunk (chunked.go)
		if totalLen&chunkFrameFlag != 0 {
			if !readChunkFrame(ni, s, senderPeerID, totalLen&^chunkFrameFlag) {
				s.Reset()
				return
			}
			continue
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

		if totalLen <= uint32(channelLen)+1 {
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

		// The channel name is chosen by the sender. One named like a Session protocol
		// would land in that protocol's queue, which is reserved to Session frames (it
		// bypasses the channel cap and is drained in full): no honest peer does that.
		if isSessionProtocol(channel) {
			logger.Warnf("[GO] 🗑️ Instance %d: Chat message on the reserved channel '%s' via Stream %s from %s. Resetting.\n", ni.instanceIndex, channel, streamID, senderPeerID)
			s.Reset()
			return
		}

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

func newPersistentStream(s network.Stream) *persistentStream {
	return &persistentStream{s: s, rank: connRank(s.Conn()), done: make(chan struct{})}
}

// connRank ranks a connection as the place of the chat stream of a peer.
func connRank(c network.Conn) int {
	if c.Stat().Limited {
		return connRankLimited
	}
	if _, err := c.RemoteMultiaddr().ValueForProtocol(ma.P_CIRCUIT); err == nil {
		return connRankRelayed
	}
	return connRankDirect
}

func connRankName(rank int) string {
	switch rank {
	case connRankDirect:
		return "direct"
	case connRankRelayed:
		return "relayed"
	default:
		return "relayed, limited"
	}
}

// bestConnRank returns the rank of the best open connection to the peer, -1 with none.
func bestConnRank(ni *NodeInstance, pid peer.ID) int {
	best := -1
	for _, c := range ni.host.Network().ConnsToPeer(pid) {
		if c.IsClosed() {
			continue
		}
		if rank := connRank(c); rank > best {
			best = rank
		}
	}
	return best
}

// sendOnChatStream sends one chat frame to the peer on its persistent stream. It opens the
// stream when there is none, and moves it first when a better connection to the peer than
// the one it sits on has appeared. The text it returns is the one of the reply to the
// caller, for both outcomes.
func sendOnChatStream(ni *NodeInstance, pid peer.ID, channel string, data []byte) (string, error) {
	frame, err := buildDirectMessageFrame(channel, data)
	if err != nil {
		return fmt.Sprintf("Failed to build the frame for %s.", pid), err
	}
	return sendFrameOnChatStream(ni, pid, frame)
}

// sendFrameOnChatStream is sendOnChatStream for a frame that is already built (see
// buildDirectMessageFrame): what the writer of a send queue writes.
func sendFrameOnChatStream(ni *NodeInstance, pid peer.ID, frame []byte) (string, error) {
	for {
		ni.streamsMutex.Lock()
		ps, streamExists := ni.persistentChatStreams[pid]
		ni.streamsMutex.Unlock()

		how := "reused stream"
		if !streamExists {
			opened, err := openChatStream(ni, pid)
			if err != nil {
				return fmt.Sprintf("Failed to open new stream to %s.", pid), err
			}
			if opened == nil {
				continue
			}
			ps, how = opened, "new stream"
		}

		ps.sendMu.Lock()
		if ps.retired {
			// Another sender moved the stream while this one was getting here.
			ps.sendMu.Unlock()
			continue
		}
		if streamExists {
			logger.Debugf("[GO]   ↳ Instance %d: Reusing stream %s to %s\n", ni.instanceIndex, ps.s.ID(), pid)
			if next := moveChatStream(ni, pid, ps); next != nil {
				lastWrite := ps.lastWrite
				ps.sendMu.Unlock()
				waitChatStreamDrained(ni, ps, lastWrite)
				ps, how = next, "moved stream"
			}
		}
		err := writeChatFrame(ni, ps.s, frame)
		ps.lastWrite = time.Now()
		ps.sendMu.Unlock()
		if err == nil {
			logger.Infof("[GO] 📤 Instance %d: Sent to %s via Stream %s (%s)\n", ni.instanceIndex, pid, ps.s.ID(), how)
			return fmt.Sprintf("Direct message sent to %s (%s).", pid, how), nil
		}

		// Write failed: remove the broken entry, so that the next send opens a new stream.
		// The stream must not be written again: a deadline can fire in the middle of a
		// frame, and the next frame would then desync the length-prefixed framing of the
		// receiver.
		logger.Warnf("[GO] ⚠️ Instance %d: Write failed on Stream %s to %s (%s): %v. Removing.\n", ni.instanceIndex, ps.s.ID(), pid, how, err)
		ni.streamsMutex.Lock()
		if cur, ok := ni.persistentChatStreams[pid]; ok && cur == ps {
			delete(ni.persistentChatStreams, pid)
		}
		ni.streamsMutex.Unlock()
		_ = ps.s.Close()
		if streamExists {
			return fmt.Sprintf("Failed to write to stream %s (closed).", pid), err
		}
		return fmt.Sprintf("Failed to write to new stream to '%s' (needs reconnect).", pid), err
	}
}

// openChatStream opens the chat stream of a peer that has none and makes it the entry of
// the peer. It returns nil when an entry appeared meanwhile.
func openChatStream(ni *NodeInstance, pid peer.ID) (*persistentStream, error) {
	logger.Debugf("[GO]   ↳ Instance %d: Creating NEW stream to %s...\n", ni.instanceIndex, pid)
	streamCtx, cancel := context.WithTimeout(ni.ctx, StreamCreationTimeout)
	defer cancel()

	newStream, err := ni.host.NewStream(
		network.WithAllowLimitedConn(streamCtx, UnaiverseChatProtocol),
		pid,
		UnaiverseChatProtocol,
	)
	if err != nil {
		return nil, err
	}

	// Double-check if a stream of the peer arrived, or another sender opened one, while
	// ours was opening.
	ni.streamsMutex.Lock()
	defer ni.streamsMutex.Unlock()
	if existingPS, existsNow := ni.persistentChatStreams[pid]; existsNow {
		logger.Warnf("[GO] ⚠️ Instance %d: Race detected. Using existing stream %s, closing our new %s.\n", ni.instanceIndex, existingPS.s.ID(), newStream.ID())
		_ = newStream.Close()
		return nil, nil
	}
	ps := newPersistentStream(newStream)
	logger.Debugf("[GO] ✅ Instance %d: Opened and stored new persistent stream %s to %s\n", ni.instanceIndex, newStream.ID(), pid)
	ni.persistentChatStreams[pid] = ps
	go readChatStream(ni, ps)
	return ps, nil
}

// moveChatStream opens a chat stream on the best connection to the peer when that is a
// better one than the stream of the entry sits on, makes it the entry and closes the old
// one for writing. Read is left open: what the peer still writes on the old stream is
// read to its end by the reader that stream already has. It returns nil when the stream
// stays where it is.
//
// Called with ps.sendMu held. The stream it returns comes with its own sendMu held, taken
// before the stream can be found: nobody writes on it before the caller has waited for
// the old one to drain.
func moveChatStream(ni *NodeInstance, pid peer.ID, ps *persistentStream) *persistentStream {
	if ps.rank >= connRankDirect || bestConnRank(ni, pid) <= ps.rank {
		return nil
	}
	now := ni.now()
	if now.Before(ps.nextMoveAt) {
		return nil
	}

	ctx, cancel := context.WithTimeout(ni.ctx, ChatStreamMoveTimeout)
	defer cancel()
	ctx = network.WithNoDial(ctx, "the chat stream moves to a connection that exists")
	s, err := ni.host.NewStream(network.WithAllowLimitedConn(ctx, UnaiverseChatProtocol), pid, UnaiverseChatProtocol)
	if err != nil {
		ps.nextMoveAt = now.Add(ChatStreamMoveRetry)
		logger.Warnf("[GO] ⚠️ Instance %d: Chat stream %s to %s stays on its %s connection, a stream on a better one did not open: %v\n", ni.instanceIndex, ps.s.ID(), pid, connRankName(ps.rank), err)
		return nil
	}

	next := newPersistentStream(s)
	next.sendMu.Lock()
	ni.streamsMutex.Lock()
	ni.persistentChatStreams[pid] = next
	ni.streamsMutex.Unlock()
	go readChatStream(ni, next)

	ps.retired = true
	_ = ps.s.CloseWrite()
	logger.Infof("[GO] 🔀 Instance %d: Chat stream to %s moved to a %s connection (stream %s), from a %s one (stream %s)\n", ni.instanceIndex, pid, connRankName(next.rank), s.ID(), connRankName(ps.rank), ps.s.ID())
	return next
}

// waitChatStreamDrained waits until the peer has read what was written on a chat stream
// that was replaced: until the peer closes the stream in turn, which ends its reader
// here, or until ChatStreamDrainTimeout after the last write on it.
func waitChatStreamDrained(ni *NodeInstance, ps *persistentStream, lastWrite time.Time) {
	wait := time.Until(lastWrite.Add(ChatStreamDrainTimeout))
	if wait <= 0 {
		return
	}
	timer := time.NewTimer(wait)
	defer timer.Stop()
	select {
	case <-ps.done:
	case <-timer.C:
		logger.Debugf("[GO] ⏳ Instance %d: Chat stream %s was not closed by its peer within %v of the last write on it.\n", ni.instanceIndex, ps.s.ID(), ChatStreamDrainTimeout)
	case <-ni.ctx.Done():
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
			// A rendezvous roster. Only the host that owns the topic may write it: the
			// topic embeds the host peer id and gossipsub authenticates the sender, so
			// drop an update whose sender is not that host.
			expectedHostStr, hasSuffix := strings.CutSuffix(topic, "::ps:rv")
			if !hasSuffix {
				logger.Warnf("[GO] ⚠️ Instance %d: rendezvous topic '%s' is not in '<host>::ps:rv' form; dropping update.\n", ni.instanceIndex, topic)
				continue
			}
			expectedHost, decErr := peer.Decode(expectedHostStr)
			if decErr != nil {
				logger.Warnf("[GO] ⚠️ Instance %d: rendezvous topic '%s' has an undecodable host id: %v; dropping update.\n", ni.instanceIndex, topic, decErr)
				continue
			}
			if msg.GetFrom() != expectedHost {
				logger.Warnf("[GO] 🚫 Instance %d: rendezvous update on '%s' from %s is not the world host %s; dropping forged roster.\n", ni.instanceIndex, topic, msg.GetFrom(), expectedHost)
				continue
			}

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
			first := ni.rendezvousState == nil
			if first {
				ni.rendezvousState = &RendezvousState{}
			}
			rendezvousState := ni.rendezvousState
			// Drop a roster that does not advance the counter: a replayed host-signed
			// envelope must not reinstall a stale one. State resets on (un)subscribe.
			if !first && updatePayload.UpdateCount <= rendezvousState.UpdateCount {
				ni.rendezvousMutex.Unlock()
				logger.Debugf("[GO] ⏮️ Instance %d: rendezvous update on '%s' tag %d does not advance past %d; ignoring stale/replayed roster.\n", ni.instanceIndex, topic, updatePayload.UpdateCount, rendezvousState.UpdateCount)
				continue
			}
			rendezvousState.Peers = newPeerMap
			rendezvousState.UpdateCount = updatePayload.UpdateCount
			ni.rendezvousMutex.Unlock()

			logger.Debugf("[GO] ✅ Instance %d: Updated rendezvous peers from topic '%s'. Found %d peers. Update count: %d.\n", ni.instanceIndex, topic, len(newPeerMap), updatePayload.UpdateCount)
		} else {
			// This is a standard message. Queue it as before.
			logger.Debugf("[GO] 📝 Instance %d: Storing new pubsub message from topic '%s'.\n", ni.instanceIndex, topic)
			storeReceivedMessage(ni, msg.GetFrom(), topic, "", msg.Data) // Empty protocolID for PubSub messages, since they are identified by topic instead.
		}
	}
}

// writeSessionFrame writes a single length-prefixed session payload onto a stream:
// 4-byte big-endian uint32 length, then the protobuf bytes. Returns nil on success
// (an error otherwise); a nil return means "written to the muxer", not delivered.
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
// A writer with a write deadline (a stream) gets the one of writeChatFrame.
func writeDirectMessageFrame(w io.Writer, channel string, payload []byte) error {
	frame, err := buildDirectMessageFrame(channel, payload)
	if err != nil {
		return err
	}
	return writeChatFrame(nil, w, frame)
}

// chatWriteTimeout is the write deadline of a chat frame of the given size: the base, plus
// the time the frame takes at the slowest rate we accept. A small control message gets
// about the base, a large tensor or file the time it needs on a slow link (the same
// reasoning as ChatStreamIdleTimeout on the read side). Zero when ConfigureSendQueues did
// not set the two values: no deadline.
func chatWriteTimeout(frameLen int) time.Duration {
	if chatWriteBaseTimeout <= 0 || chatWriteMinBytesPerSec <= 0 {
		return 0
	}
	return chatWriteBaseTimeout + time.Duration(float64(frameLen)/float64(chatWriteMinBytesPerSec)*float64(time.Second))
}

// writeChatFrame writes one built frame on a chat stream, under the write deadline of its
// size when the writer has one: a peer whose link stopped draining (a stopped process, a
// stalled relay) holds the write no longer than that. Without it the write waits on the
// stream window of the muxer with no end. A deadline that cannot be set is logged and the
// frame is written all the same: sending with no limit beats refusing to send. The caller
// must close the stream on an error, since the deadline can fire in the middle of a frame.
// ni is only for the logs, and may be nil.
func writeChatFrame(ni *NodeInstance, w io.Writer, frame []byte) error {
	if timeout := chatWriteTimeout(len(frame)); timeout > 0 {
		if d, ok := w.(interface{ SetWriteDeadline(time.Time) error }); ok {
			if err := d.SetWriteDeadline(time.Now().Add(timeout)); err != nil {
				instanceIndex := -1
				if ni != nil {
					instanceIndex = ni.instanceIndex
				}
				logger.Warnf("[GO] ⚠️ Instance %d: SetWriteDeadline failed on a chat stream: %v\n", instanceIndex, err)
			} else {
				// A stream that stays open must not keep the deadline of this frame.
				defer d.SetWriteDeadline(time.Time{})
			}
		}
	}

	// Write the entire frame to the stream.
	if _, err := w.Write(frame); err != nil {
		return fmt.Errorf("failed to write framed message to stream: %w", err)
	}
	return nil
}
