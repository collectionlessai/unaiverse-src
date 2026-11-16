// Tests of the Session frames on the WebRTC DataChannel: the capability of the signalling,
// when a frame goes on the channel and when on a stream, how one that arrives is queued,
// and which offers are made and accepted over an entry of the table.
package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"io"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	libp2pproto "github.com/libp2p/go-libp2p/core/protocol"
	pwebrtc "github.com/pion/webrtc/v4"
)

// fakeClock is the clock of an instance under test: it moves only when the test says so.
type fakeClock struct {
	mu sync.Mutex
	t  time.Time
}

func (c *fakeClock) now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *fakeClock) advance(d time.Duration) {
	c.mu.Lock()
	c.t = c.t.Add(d)
	c.mu.Unlock()
}

// sessionNode is an instance with what the Session frames need on both paths: a host on
// the loopback with the Session and signalling handlers, the tables, the store, and a clock
// of its own. streamFrames counts the Session frames that reached it on a libp2p stream.
type sessionNode struct {
	ni           *NodeInstance
	clock        *fakeClock
	streamFrames chan string
}

// useSessionTestSettings sets what is package-level and normally comes from InitializeLibrary
// (the size limit, the limits of the store) and removes the ICE servers, so that the peer
// connections of a test meet on the host candidates. To be called before any node exists:
// the goroutines of the nodes read these variables.
func useSessionTestSettings(t *testing.T, maxMessageSize uint32) {
	t.Helper()
	prevMax, prevQueueLen, prevChannels, prevSTUN := MaxMessageSize, maxChannelQueueLen, maxUniqueChannels, defaultSTUNServers
	MaxMessageSize, maxChannelQueueLen, maxUniqueChannels, defaultSTUNServers = maxMessageSize, 64, 64, nil
	t.Cleanup(func() {
		MaxMessageSize, maxChannelQueueLen, maxUniqueChannels, defaultSTUNServers = prevMax, prevQueueLen, prevChannels, prevSTUN
	})
}

func newSessionNode(t *testing.T) *sessionNode {
	t.Helper()
	clock := &fakeClock{t: time.Unix(1_800_000_000, 0)}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	node := &sessionNode{
		ni: &NodeInstance{
			host:                  newLoopbackHost(t),
			ctx:                   ctx,
			messageStore:          newMessageStore(),
			friendlyPeers:         make(map[peer.ID]ExtendedPeerInfo),
			persistentChatStreams: make(map[peer.ID]*persistentStream),
			webrtcConnections:     make(map[peer.ID]*WebRTCConn),
			clock:                 clock.now,
		},
		clock:        clock,
		streamFrames: make(chan string, 16),
	}
	for _, protocolID := range []string{SessionHelloProtocol, SessionControlProtocol} {
		node.ni.host.SetStreamHandler(libp2pproto.ID(protocolID), func(s network.Stream) {
			node.streamFrames <- protocolID
			handleSessionStream(node.ni, s, protocolID)
		})
	}
	setupWebRTCSignalHandler(node.ni)
	// The connections of pion outlive a test that fails half-way unless they are closed.
	t.Cleanup(func() {
		node.ni.webrtcMutex.Lock()
		defer node.ni.webrtcMutex.Unlock()
		for _, conn := range node.ni.webrtcConnections {
			conn.pc.Close()
		}
	})
	return node
}

func (n *sessionNode) id() peer.ID { return n.ni.host.ID() }

func connectNodes(t *testing.T, from, to host.Host) {
	t.Helper()
	if err := from.Connect(context.Background(), peer.AddrInfo{ID: to.ID(), Addrs: to.Addrs()}); err != nil {
		t.Fatalf("connect: %v", err)
	}
}

// linkNodes gives two nodes a libp2p connection and a DataChannel, registered on both
// ends as a signalling exchange would have, with each side taking Session frames.
func linkNodes(t *testing.T, local, remote *sessionNode) (localConn, remoteConn *WebRTCConn) {
	t.Helper()
	connectNodes(t, local.ni.host, remote.ni.host)
	pair := newWebRTCPair(t)
	localConn = &WebRTCConn{pc: pair.localPC, dc: pair.localDC, remotePeer: remote.id(),
		peerSessionFrames: true, direction: "inbound"}
	remoteConn = &WebRTCConn{pc: pair.remotePC, dc: pair.remoteDC, remotePeer: local.id(),
		peerSessionFrames: true, direction: "outbound"}
	for node, conn := range map[*sessionNode]*WebRTCConn{local: localConn, remote: remoteConn} {
		wireWebRTCDataChannel(node.ni, conn)
		registerWebRTCDataChannel(node.ni, conn)
	}
	return localConn, remoteConn
}

// popped waits for n messages to be queued on the node and returns them with the channel
// key each was queued under.
func (n *sessionNode) popped(t *testing.T, count int) map[string][]*QueuedMessage {
	t.Helper()
	got := make(map[string][]*QueuedMessage)
	total := 0
	waitUntil(t, "the queued messages", func() bool {
		store := n.ni.messageStore
		store.mu.Lock()
		defer store.mu.Unlock()
		for channel, messages := range store.messagesByChannel {
			for messages.Len() > 0 {
				front := messages.Front()
				got[channel] = append(got[channel], front.Value.(*QueuedMessage))
				messages.Remove(front)
				total++
			}
			delete(store.messagesByChannel, channel)
		}
		return total >= count
	})
	return got
}

// arrivedOnAStream reports whether a Session frame reached the node on a libp2p stream
// since the last call. The stream handler counts a frame before it queues it, so once a
// frame has been popped the answer is false only if it came on the DataChannel.
func (n *sessionNode) arrivedOnAStream() bool {
	select {
	case <-n.streamFrames:
		return true
	default:
		return false
	}
}

func isFriendly(ni *NodeInstance, pid peer.ID) (ExtendedPeerInfo, bool) {
	ni.peersMutex.RLock()
	defer ni.peersMutex.RUnlock()
	info, ok := ni.friendlyPeers[pid]
	return info, ok
}

// The capability is one boolean of the signalling message, under the name the two sides
// agreed on. A message of a side that does not know it decodes as "not declared", and a
// field this side does not know (the stage of the browser's offer) is ignored.
func TestSignalMessageCarriesTheSessionFramesCapability(t *testing.T) {
	data, err := json.Marshal(SignalMessage{Type: "offer", SDP: "v=0", SessionFrames: true})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(data), `"session_frames":true`) {
		t.Fatalf("the offer does not declare the capability on the wire: %s", data)
	}

	var declared, silent SignalMessage
	if err := json.Unmarshal([]byte(`{"type":"offer","sdp":"v=0","stage":"direct","session_frames":true}`), &declared); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !declared.SessionFrames || declared.SDP != "v=0" {
		t.Fatalf("declared capability not read: %+v", declared)
	}
	if err := json.Unmarshal([]byte(`{"type":"answer","sdp":"v=0"}`), &silent); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if silent.SessionFrames {
		t.Fatal("a message without the field was read as declaring the capability")
	}
	// Only a JSON boolean is a declaration: anything else is a malformed message.
	var malformed SignalMessage
	if err := json.Unmarshal([]byte(`{"type":"offer","sdp":"v=0","session_frames":"yes"}`), &malformed); err == nil {
		t.Fatal("a capability that is not a boolean was accepted")
	}
}

// The four conditions under which a Session frame may go on a DataChannel, one at a time:
// the peer declared the capability for this channel, the channel is open, its peer
// connection is connected, and the channel is fresh (something arrived in the last 30 s,
// or it opened less than 30 s ago).
func TestSessionFramesNeedCapabilityOpenChannelConnectedPeerAndFreshness(t *testing.T) {
	useSessionTestSettings(t, 8<<20)
	local, remote := newSessionNode(t), newSessionNode(t)
	conn, remoteConn := linkNodes(t, local, remote)
	opened := local.clock.now()

	if !conn.takesSessionFrames(opened) {
		t.Fatal("a channel that has just opened, to a peer that declared the capability, takes no Session frame")
	}

	undeclared := &WebRTCConn{pc: conn.pc, dc: conn.dc, openedAt: conn.openedAt}
	if undeclared.takesSessionFrames(opened) {
		t.Fatal("the peer did not declare the capability and the channel takes Session frames")
	}

	other := newWebRTCPair(t)
	other.localPC.Close()
	ended := &WebRTCConn{pc: other.localPC, dc: conn.dc, peerSessionFrames: true, openedAt: conn.openedAt}
	if ended.takesSessionFrames(opened) {
		t.Fatal("the peer connection is not connected and the channel takes Session frames")
	}

	// Freshness. Nothing arrives: the channel is fresh for the window after its opening.
	window := WebRTCFreshnessWindow
	if !conn.takesSessionFrames(opened.Add(window - time.Second)) {
		t.Fatal("the channel is not fresh just before the end of the window that follows its opening")
	}
	if conn.takesSessionFrames(opened.Add(window + time.Second)) {
		t.Fatal("the channel is fresh after a whole window with nothing arriving on it")
	}
	// A frame of any kind arrives, here a chat one: the window restarts from it.
	local.clock.advance(window + 10*time.Second)
	if err := writeToWebRTCDataChannel(remoteConn, "a-chat-channel", []byte("hi")); err != nil {
		t.Fatalf("chat frame: %v", err)
	}
	local.popped(t, 1)
	arrived := local.clock.now()
	if !conn.takesSessionFrames(arrived.Add(window - time.Second)) {
		t.Fatal("a frame arrived less than a window ago and the channel is not fresh")
	}
	if conn.takesSessionFrames(arrived.Add(window + time.Second)) {
		t.Fatal("the channel is fresh a whole window after the last frame that arrived")
	}

	// The peer closes its end of the channel: the peer connection stays up, the channel does not.
	if err := remoteConn.dc.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	waitUntil(t, "the channel to close", func() bool { return !conn.isOpen() })
	if conn.pc.ConnectionState() != pwebrtc.PeerConnectionStateConnected {
		t.Fatalf("the peer connection is %s, the test needs it connected", conn.pc.ConnectionState())
	}
	if conn.takesSessionFrames(arrived) {
		t.Fatal("the channel is closed and takes Session frames")
	}
}

// A Session frame sent while the channel can be trusted travels on it, in the frame the
// chat uses, chunks included, and the other node queues it as it queues one read from a
// Session stream: under the protocol, with the protocol as its tag, from the peer the
// channel belongs to, and with that peer surfaced to the application pools. A chat frame
// on the same channel keeps the tag of the DataChannel, and a Session frame with no payload
// is dropped, as a Session stream that announces a zero length is.
func TestSessionFrameCrossesTheDataChannelAndIsQueuedAsFromAStream(t *testing.T) {
	useSessionTestSettings(t, 8<<20)
	local, remote := newSessionNode(t), newSessionNode(t)
	_, remoteConn := linkNodes(t, local, remote)

	small := []byte("ping")
	large := bytes.Repeat([]byte("profile-"), 25_000) // 200 kB: four chunks
	if err := sendSessionFrame(local.ni, remote.id(), SessionControlProtocol, small); err != nil {
		t.Fatalf("send: %v", err)
	}
	if err := sendSessionFrame(local.ni, remote.id(), SessionHelloProtocol, large); err != nil {
		t.Fatalf("send of a chunked frame: %v", err)
	}
	// The other way: a Session frame with no payload, which is no frame, then a chat one.
	if err := writeToWebRTCDataChannel(remoteConn, SessionControlProtocol, nil); err != nil {
		t.Fatalf("empty frame: %v", err)
	}
	if err := writeToWebRTCDataChannel(remoteConn, "a-chat-channel", []byte("hi")); err != nil {
		t.Fatalf("chat frame: %v", err)
	}

	got := remote.popped(t, 2)
	if remote.arrivedOnAStream() {
		t.Fatal("a Session frame went on a stream while the DataChannel could take it")
	}
	control, hello := got[SessionControlProtocol], got[SessionHelloProtocol]
	if len(control) != 1 || len(hello) != 1 {
		t.Fatalf("queued under the Session protocols: %d control, %d hello, want one each", len(control), len(hello))
	}
	if control[0].Protocol != SessionControlProtocol || control[0].From != local.id() || !bytes.Equal(control[0].Data, small) {
		t.Fatalf("control frame queued as protocol %q from %s with %q", control[0].Protocol, control[0].From, control[0].Data)
	}
	if hello[0].Protocol != SessionHelloProtocol || !bytes.Equal(hello[0].Data, large) {
		t.Fatalf("chunked frame queued as protocol %q with %d bytes, want %d", hello[0].Protocol, len(hello[0].Data), len(large))
	}
	if info, ok := isFriendly(remote.ni, local.id()); !ok || info.Direction != "outbound" {
		t.Fatalf("the sender was not surfaced to the pools of the receiver with the direction of the signalling: %+v", info)
	}
	if info, ok := isFriendly(local.ni, remote.id()); !ok || info.Direction != "inbound" {
		t.Fatalf("the receiver was not surfaced to the pools of the sender: %+v", info)
	}

	back := local.popped(t, 1)
	if chat := back["a-chat-channel"]; len(chat) != 1 || chat[0].Protocol != UnaiverseWebRTCSignalProtocol {
		t.Fatalf("the chat frame was not queued as a DataChannel chat frame: %+v", chat)
	}
	if empty := back[SessionControlProtocol]; len(empty) != 0 {
		t.Fatalf("a Session frame with no payload was queued: %+v", empty)
	}
}

// When one of the conditions fails the frame goes on a libp2p stream, as it always did:
// the peer did not declare the capability, or the channel is no longer fresh. And it comes
// back to the channel when a frame of any kind arrives on it.
func TestSessionFrameGoesOnAStreamWhenTheDataChannelCannotBeTrusted(t *testing.T) {
	useSessionTestSettings(t, 8<<20)
	local, remote := newSessionNode(t), newSessionNode(t)
	conn, remoteConn := linkNodes(t, local, remote)
	send := func(payload string) {
		t.Helper()
		if err := sendSessionFrame(local.ni, remote.id(), SessionControlProtocol, []byte(payload)); err != nil {
			t.Fatalf("send of %q: %v", payload, err)
		}
		got := remote.popped(t, 1)[SessionControlProtocol]
		if len(got) != 1 || string(got[0].Data) != payload || got[0].Protocol != SessionControlProtocol {
			t.Fatalf("%q did not arrive as a Session frame: %+v", payload, got)
		}
	}

	send("fresh")
	if remote.arrivedOnAStream() {
		t.Fatal("the frame went on a stream while the channel was fresh")
	}

	local.clock.advance(WebRTCFreshnessWindow + time.Second)
	send("stale")
	if !remote.arrivedOnAStream() {
		t.Fatal("the frame went on a channel that gave no sign of life for a whole window")
	}

	if err := writeToWebRTCDataChannel(remoteConn, "a-chat-channel", []byte("hi")); err != nil {
		t.Fatalf("chat frame: %v", err)
	}
	local.popped(t, 1)
	send("fresh again")
	if remote.arrivedOnAStream() {
		t.Fatal("a frame arrived on the channel and the next Session frame still went on a stream")
	}

	conn.peerSessionFrames = false
	send("undeclared")
	if !remote.arrivedOnAStream() {
		t.Fatal("the frame went on the channel of a peer that did not declare the capability")
	}
}

// A send that fails on the channel is repeated on a stream. Here the frame fits the limit
// as a Session frame and exceeds it once the DataChannel frame adds the channel name.
func TestSessionFrameFallsBackToAStreamWhenTheDataChannelSendFails(t *testing.T) {
	payload := bytes.Repeat([]byte("x"), 100)
	useSessionTestSettings(t, uint32(len(payload))+10)
	local, remote := newSessionNode(t), newSessionNode(t)
	linkNodes(t, local, remote)

	if err := sendSessionFrame(local.ni, remote.id(), SessionControlProtocol, payload); err != nil {
		t.Fatalf("send: %v", err)
	}
	got := remote.popped(t, 1)[SessionControlProtocol]
	if len(got) != 1 || !bytes.Equal(got[0].Data, payload) {
		t.Fatalf("the frame did not arrive: %+v", got)
	}
	if !remote.arrivedOnAStream() {
		t.Fatal("the frame arrived, and not on a stream")
	}
}

// Both roles of a signalling exchange declare the capability and record what the peer
// declared: the offerer reads it in the answer, the answerer in the offer.
func TestBothRolesDeclareAndRecordTheSessionFramesCapability(t *testing.T) {
	useSessionTestSettings(t, 8<<20)
	offerer, answerer := newSessionNode(t), newSessionNode(t)
	connectNodes(t, offerer.ni.host, answerer.ni.host)

	if err := initiateWebRTCConnection(offerer.ni, answerer.id()); err != nil {
		t.Fatalf("offer: %v", err)
	}
	waitUntil(t, "the entry of the answerer", func() bool { return tableEntry(answerer.ni, offerer.id()) != nil })
	onOfferer, onAnswerer := tableEntry(offerer.ni, answerer.id()), tableEntry(answerer.ni, offerer.id())
	if !onOfferer.peerSessionFrames {
		t.Fatal("the offerer did not record the capability declared in the answer")
	}
	if !onAnswerer.peerSessionFrames {
		t.Fatal("the answerer did not record the capability declared in the offer")
	}
	if onOfferer.direction != "outbound" || onAnswerer.direction != "inbound" {
		t.Fatalf("directions of the signalling connection: offerer %q, answerer %q", onOfferer.direction, onAnswerer.direction)
	}

	if err := sendSessionFrame(offerer.ni, answerer.id(), SessionControlProtocol, []byte("ping")); err != nil {
		t.Fatalf("send: %v", err)
	}
	if got := answerer.popped(t, 1)[SessionControlProtocol]; len(got) != 1 || string(got[0].Data) != "ping" {
		t.Fatalf("the frame did not arrive: %+v", got)
	}
	if answerer.arrivedOnAStream() {
		t.Fatal("the frame went on a stream between two nodes that declared the capability")
	}
}

// legacyPeer plays a node that knows nothing of the capability, with pion and a libp2p
// host: it answers an offer, or makes one, with signalling messages that lack the field,
// and keeps what the other side sent.
type legacyPeer struct {
	host         host.Host
	received     chan SignalMessage
	streamFrames chan []byte
	// sentAtOpen, when set, is sent on the DataChannel from the handler of its opening.
	sentAtOpen []byte
}

// atOpen makes the legacy peer send a Session frame the moment its end of the channel opens.
func (legacy *legacyPeer) atOpen(t *testing.T, payload string) {
	t.Helper()
	frame, err := buildDirectMessageFrame(SessionControlProtocol, []byte(payload))
	if err != nil {
		t.Fatalf("frame: %v", err)
	}
	legacy.sentAtOpen = append([]byte{webRTCChunkFlagStartEnd}, frame...)
}

func (legacy *legacyPeer) sendAtOpen(dc *pwebrtc.DataChannel) {
	if legacy.sentAtOpen != nil {
		dc.OnOpen(func() { dc.Send(legacy.sentAtOpen) })
	}
}

func newLegacyPeer(t *testing.T) *legacyPeer {
	t.Helper()
	legacy := &legacyPeer{
		host:         newLoopbackHost(t),
		received:     make(chan SignalMessage, 4),
		streamFrames: make(chan []byte, 4),
	}
	legacy.host.SetStreamHandler(SessionControlProtocol, func(s network.Stream) {
		defer s.Close()
		var length uint32
		if err := binary.Read(s, binary.BigEndian, &length); err != nil {
			return
		}
		frame := make([]byte, length)
		if _, err := io.ReadFull(s, frame); err == nil {
			legacy.streamFrames <- frame
		}
	})
	legacy.host.SetStreamHandler(UnaiverseWebRTCSignalProtocol, func(s network.Stream) {
		defer s.Close()
		offer, err := readSignalMessage(s)
		if err != nil {
			return
		}
		legacy.received <- offer
		pc, err := pwebrtc.NewPeerConnection(pwebrtc.Configuration{})
		if err != nil {
			return
		}
		t.Cleanup(func() { pc.Close() })
		pc.OnDataChannel(legacy.sendAtOpen)
		if pc.SetRemoteDescription(pwebrtc.SessionDescription{Type: pwebrtc.SDPTypeOffer, SDP: offer.SDP}) != nil {
			return
		}
		answer, err := pc.CreateAnswer(nil)
		if err != nil {
			return
		}
		gathered := pwebrtc.GatheringCompletePromise(pc)
		if pc.SetLocalDescription(answer) != nil {
			return
		}
		<-gathered
		writeSignalMessage(s, SignalMessage{Type: "answer", SDP: pc.LocalDescription().SDP})
	})
	return legacy
}

// offerTo makes the legacy peer offer a DataChannel to a node and returns the answer.
func (legacy *legacyPeer) offerTo(t *testing.T, to peer.ID) SignalMessage {
	t.Helper()
	must := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatalf("legacy offer: %v", err)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	s, err := legacy.host.NewStream(ctx, to, UnaiverseWebRTCSignalProtocol)
	must(err)
	defer s.Close()
	pc, err := pwebrtc.NewPeerConnection(pwebrtc.Configuration{})
	must(err)
	t.Cleanup(func() { pc.Close() })
	ordered := true
	dc, err := pc.CreateDataChannel(WebRTCDataChannelLabel, &pwebrtc.DataChannelInit{Ordered: &ordered})
	must(err)
	legacy.sendAtOpen(dc)
	offer, err := pc.CreateOffer(nil)
	must(err)
	gathered := pwebrtc.GatheringCompletePromise(pc)
	must(pc.SetLocalDescription(offer))
	<-gathered
	must(writeSignalMessage(s, SignalMessage{Type: "offer", SDP: pc.LocalDescription().SDP}))
	answer, err := readSignalMessage(s)
	must(err)
	must(pc.SetRemoteDescription(pwebrtc.SessionDescription{Type: pwebrtc.SDPTypeAnswer, SDP: answer.SDP}))
	return answer
}

// Against a peer that does not know the capability everything stays as it was. Our offer
// and our answer declare it, the peer ignores it and declares nothing, and the Session
// frames to that peer go on streams, whatever the age of the channel.
func TestPeerWithoutTheCapabilityKeepsItsSessionFramesOnStreams(t *testing.T) {
	t.Run("as answerer", func(t *testing.T) {
		useSessionTestSettings(t, 8<<20)
		node, legacy := newSessionNode(t), newLegacyPeer(t)
		connectNodes(t, node.ni.host, legacy.host)
		if err := initiateWebRTCConnection(node.ni, legacy.host.ID()); err != nil {
			t.Fatalf("offer: %v", err)
		}
		select {
		case offer := <-legacy.received:
			if !offer.SessionFrames {
				t.Fatal("our offer does not declare the capability")
			}
		case <-time.After(5 * time.Second):
			t.Fatal("the legacy peer saw no offer")
		}
		entry := tableEntry(node.ni, legacy.host.ID())
		if entry == nil || !entry.isOpen() {
			t.Fatal("no open DataChannel with the legacy peer")
		}
		if entry.peerSessionFrames {
			t.Fatal("a capability the peer never declared was recorded")
		}

		if err := sendSessionFrame(node.ni, legacy.host.ID(), SessionControlProtocol, []byte("ping")); err != nil {
			t.Fatalf("send: %v", err)
		}
		select {
		case frame := <-legacy.streamFrames:
			if string(frame) != "ping" {
				t.Fatalf("received %q on the stream", frame)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("the Session frame did not arrive on a stream")
		}

		// Nothing is due on an idle channel of such a peer: our own offer is skipped on
		// the state of the channel, however long it stayed silent.
		node.clock.advance(10 * WebRTCFreshnessWindow)
		if err := initiateWebRTCConnection(node.ni, legacy.host.ID()); err == nil {
			t.Fatal("a second offer was made over an open channel of a peer without the capability")
		}
		if tableEntry(node.ni, legacy.host.ID()) != entry {
			t.Fatal("the entry of the peer changed")
		}
	})

	t.Run("as offerer", func(t *testing.T) {
		useSessionTestSettings(t, 8<<20)
		node, legacy := newSessionNode(t), newLegacyPeer(t)
		connectNodes(t, legacy.host, node.ni.host)
		if answer := legacy.offerTo(t, node.id()); !answer.SessionFrames {
			t.Fatal("our answer does not declare the capability")
		}
		waitUntil(t, "the entry of the legacy peer", func() bool { return tableEntry(node.ni, legacy.host.ID()) != nil })
		if tableEntry(node.ni, legacy.host.ID()).peerSessionFrames {
			t.Fatal("a capability the peer never declared was recorded")
		}
	})
}

// A peer may send the moment its end of the channel opens, which can be before this side
// has registered the channel: the frame must be delivered all the same. pion drops a
// message that finds no handler, so the handlers are in place before the channel opens.
func TestFrameSentWhenTheChannelOpensIsNotLost(t *testing.T) {
	arrives := func(t *testing.T, node *sessionNode, legacy *legacyPeer) {
		t.Helper()
		got := node.popped(t, 1)[SessionControlProtocol]
		if len(got) != 1 || string(got[0].Data) != "at-open" || got[0].From != legacy.host.ID() {
			t.Fatalf("the frame sent at the opening did not arrive as a Session frame: %+v", got)
		}
	}
	t.Run("we offer", func(t *testing.T) {
		useSessionTestSettings(t, 8<<20)
		node, legacy := newSessionNode(t), newLegacyPeer(t)
		legacy.atOpen(t, "at-open")
		connectNodes(t, node.ni.host, legacy.host)
		if err := initiateWebRTCConnection(node.ni, legacy.host.ID()); err != nil {
			t.Fatalf("offer: %v", err)
		}
		arrives(t, node, legacy)
	})
	t.Run("we answer", func(t *testing.T) {
		useSessionTestSettings(t, 8<<20)
		node, legacy := newSessionNode(t), newLegacyPeer(t)
		legacy.atOpen(t, "at-open")
		connectNodes(t, legacy.host, node.ni.host)
		legacy.offerTo(t, node.id())
		arrives(t, node, legacy)
	})
}

// Our own offer is skipped while the entry of the peer is open and fresh. Over one that is
// open and gave no sign of life for a window it is made, and the peer accepts it although
// its own entry looks healthy: a side offers only when it holds no channel it trusts. On
// both sides the new DataChannel takes the entry and the peer connection of the replaced
// one is closed, without taking the new entry with it.
func TestOfferOverAnEntryThatIsNotFreshReplacesItOnBothSides(t *testing.T) {
	useSessionTestSettings(t, 8<<20)
	offerer, answerer := newSessionNode(t), newSessionNode(t)
	connectNodes(t, offerer.ni.host, answerer.ni.host)
	if err := initiateWebRTCConnection(offerer.ni, answerer.id()); err != nil {
		t.Fatalf("first offer: %v", err)
	}
	waitUntil(t, "the entry of the answerer", func() bool { return tableEntry(answerer.ni, offerer.id()) != nil })
	firstOnOfferer, firstOnAnswerer := tableEntry(offerer.ni, answerer.id()), tableEntry(answerer.ni, offerer.id())

	if err := initiateWebRTCConnection(offerer.ni, answerer.id()); err == nil {
		t.Fatal("an offer was made over an entry that is open and fresh")
	}
	if tableEntry(offerer.ni, answerer.id()) != firstOnOfferer || tableEntry(answerer.ni, offerer.id()) != firstOnAnswerer {
		t.Fatal("a skipped offer changed an entry")
	}

	// Only the clock of the offerer moves: for the answerer its entry is open and fresh.
	offerer.clock.advance(WebRTCFreshnessWindow + time.Second)
	if !firstOnAnswerer.takesSessionFrames(answerer.clock.now()) {
		t.Fatal("the test needs the entry of the answerer healthy")
	}
	if err := initiateWebRTCConnection(offerer.ni, answerer.id()); err != nil {
		t.Fatalf("offer over an entry that is not fresh: %v", err)
	}
	waitUntil(t, "the new entry of the answerer", func() bool {
		return tableEntry(answerer.ni, offerer.id()) != firstOnAnswerer
	})
	secondOnOfferer, secondOnAnswerer := tableEntry(offerer.ni, answerer.id()), tableEntry(answerer.ni, offerer.id())
	if secondOnOfferer == nil || secondOnOfferer == firstOnOfferer || !secondOnOfferer.isOpen() {
		t.Fatal("the new DataChannel did not take the entry of the offerer")
	}
	if secondOnAnswerer == nil || !secondOnAnswerer.isOpen() {
		t.Fatal("the new DataChannel did not take the entry of the answerer")
	}
	for side, replaced := range map[string]*WebRTCConn{"offerer": firstOnOfferer, "answerer": firstOnAnswerer} {
		waitUntil(t, "the replaced peer connection of the "+side+" to close", func() bool {
			return replaced.pc.ConnectionState() == pwebrtc.PeerConnectionStateClosed
		})
	}
	// The cleanup of a replaced connection runs after its state change: give it time.
	time.Sleep(300 * time.Millisecond)
	if tableEntry(offerer.ni, answerer.id()) != secondOnOfferer || tableEntry(answerer.ni, offerer.id()) != secondOnAnswerer {
		t.Fatal("the end of a replaced connection removed the entry of the new one")
	}

	if err := sendSessionFrame(offerer.ni, answerer.id(), SessionControlProtocol, []byte("ping")); err != nil {
		t.Fatalf("send: %v", err)
	}
	if got := answerer.popped(t, 1)[SessionControlProtocol]; len(got) != 1 || string(got[0].Data) != "ping" {
		t.Fatalf("the frame did not arrive on the new channel: %+v", got)
	}
	if answerer.arrivedOnAStream() {
		t.Fatal("the frame went on a stream with a new DataChannel in place")
	}
}
