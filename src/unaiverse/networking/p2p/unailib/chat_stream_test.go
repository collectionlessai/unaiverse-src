// Tests of the chat stream: on which connection it sits, when it moves to a better one,
// and what happens to the frames around the move.
package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	libp2pproto "github.com/libp2p/go-libp2p/core/protocol"
	rcmgr "github.com/libp2p/go-libp2p/p2p/host/resource-manager"
	"github.com/libp2p/go-libp2p/p2p/protocol/circuitv2/client"
	relayv2 "github.com/libp2p/go-libp2p/p2p/protocol/circuitv2/relay"
	ma "github.com/multiformats/go-multiaddr"
)

const chatTestChannel = "sender::dm:receiver-test"

// newChatNode returns an instance with what the chat stream needs: a host with the chat
// handler, the table of the streams and the store. With no option the host listens on the
// loopback.
func newChatNode(t *testing.T, opts ...libp2p.Option) *NodeInstance {
	t.Helper()
	if len(opts) == 0 {
		opts = []libp2p.Option{libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0")}
	}
	h, err := libp2p.New(opts...)
	if err != nil {
		t.Fatalf("host: %v", err)
	}
	t.Cleanup(func() { h.Close() })
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	ni := &NodeInstance{
		host:                  h,
		ctx:                   ctx,
		messageStore:          newMessageStore(),
		friendlyPeers:         make(map[peer.ID]ExtendedPeerInfo),
		persistentChatStreams: make(map[peer.ID]*persistentStream),
	}
	setupDirectMessageHandler(ni)
	return ni
}

// newChatRelay starts a relay on the loopback: one with the limits of the library, as an
// infrastructure relay, or one with none, as a world host.
func newChatRelay(t *testing.T, limited bool) host.Host {
	t.Helper()
	service := libp2p.EnableRelayService(relayv2.WithInfiniteLimits())
	if limited {
		service = libp2p.EnableRelayService()
	}
	h, err := libp2p.New(
		libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"),
		libp2p.DisableRelay(),
		libp2p.ForceReachabilityPublic(),
		service,
	)
	if err != nil {
		t.Fatalf("relay: %v", err)
	}
	t.Cleanup(func() { h.Close() })
	return h
}

// relayedLink connects from to the target through the relay, on which the target reserves
// first. Whatever from knew of the target is forgotten before, so that the circuit is the
// only address it can dial.
func relayedLink(t *testing.T, relay, from, target host.Host) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	relayInfo := peer.AddrInfo{ID: relay.ID(), Addrs: relay.Addrs()}
	if err := target.Connect(ctx, relayInfo); err != nil {
		t.Fatalf("target to relay: %v", err)
	}
	if _, err := client.Reserve(ctx, target, relayInfo); err != nil {
		t.Fatalf("reservation: %v", err)
	}
	circuit := ma.StringCast(fmt.Sprintf("%s/p2p/%s/p2p-circuit", relay.Addrs()[0], relay.ID()))
	from.Peerstore().ClearAddrs(target.ID())
	err := from.Connect(ctx, peer.AddrInfo{ID: target.ID(), Addrs: []ma.Multiaddr{circuit}})
	if err != nil {
		t.Fatalf("connect through the relay: %v", err)
	}
}

// directLink adds a direct connection between two hosts that may already have a relayed one.
func directLink(t *testing.T, from, target host.Host) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	from.Peerstore().AddAddrs(target.ID(), target.Addrs(), time.Hour)
	if _, err := from.Network().DialPeer(network.WithForceDirectDial(ctx, "test"), target.ID()); err != nil {
		t.Fatalf("direct connection: %v", err)
	}
	// The connection is reported on the two sides at different moments.
	waitUntil(t, "the direct connection on the target", func() bool {
		for _, c := range target.Network().ConnsToPeer(from.ID()) {
			if connRank(c) == connRankDirect {
				return true
			}
		}
		return false
	})
}

func chatEntry(ni *NodeInstance, pid peer.ID) *persistentStream {
	ni.streamsMutex.Lock()
	defer ni.streamsMutex.Unlock()
	return ni.persistentChatStreams[pid]
}

// sendChat sends one frame on the chat stream and fails the test on an error. It returns
// the text of the reply, which says whether the stream was new, reused or moved.
func sendChat(t *testing.T, ni *NodeInstance, pid peer.ID, channel, text string) string {
	t.Helper()
	reply, err := sendOnChatStream(ni, pid, channel, []byte(text))
	if err != nil {
		t.Fatalf("send of %q: %s: %v", text, reply, err)
	}
	return reply
}

// queuedOn returns the payloads queued on a channel of the node, oldest first.
func queuedOn(ni *NodeInstance, channel string) []string {
	store := ni.messageStore
	store.mu.Lock()
	defer store.mu.Unlock()
	var out []string
	if messages, ok := store.messagesByChannel[channel]; ok {
		for e := messages.Front(); e != nil; e = e.Next() {
			out = append(out, string(e.Value.(*QueuedMessage).Data))
		}
	}
	return out
}

func waitQueued(t *testing.T, ni *NodeInstance, channel string, want ...string) {
	t.Helper()
	waitUntil(t, fmt.Sprintf("%d frames on %s", len(want), channel), func() bool {
		return len(queuedOn(ni, channel)) >= len(want)
	})
	if got := queuedOn(ni, channel); !sameStrings(got, want) {
		t.Fatalf("frames on %s: got %v, want %v", channel, got, want)
	}
}

func waitClosed(t *testing.T, what string, done <-chan struct{}) {
	t.Helper()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatalf("%s: still open", what)
	}
}

// scriptedPeer is a peer that is not a node: a host whose chat handler does what the test
// tells it, to stand for a slow peer and for one that is not written in Go.
type scriptedPeer struct {
	host host.Host
	// release is closed by the test: the first stream is read only from then on.
	release chan struct{}
	// closeAtEnd says whether the peer closes a stream it has read to the end, as a Go node
	// does, or leaves it open.
	closeAtEnd bool
	// lastWords is written on the first stream after its end was read.
	lastWords string

	mu      sync.Mutex
	streams int
	frames  []string
}

func newScriptedPeer(t *testing.T, closeAtEnd bool) *scriptedPeer {
	t.Helper()
	p := &scriptedPeer{host: newLoopbackHost(t), release: make(chan struct{}), closeAtEnd: closeAtEnd}
	p.host.SetStreamHandler(UnaiverseChatProtocol, p.handle)
	return p
}

func (p *scriptedPeer) handle(s network.Stream) {
	p.mu.Lock()
	first := p.streams == 0
	p.streams++
	p.mu.Unlock()
	if first {
		<-p.release
	}
	for {
		payload, err := readChatFrame(s)
		if err != nil {
			break
		}
		p.mu.Lock()
		p.frames = append(p.frames, payload)
		p.mu.Unlock()
	}
	if first && p.lastWords != "" {
		_ = writeDirectMessageFrame(s, chatTestChannel, []byte(p.lastWords))
	}
	if p.closeAtEnd {
		s.Close()
	}
}

// seen returns how many chat streams reached the peer and the frames it read, in the order
// it read them, whatever the stream. A stream reaches the peer when the node starts to read
// it, before anything is written on it.
func (p *scriptedPeer) seen() (int, []string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.streams, append([]string(nil), p.frames...)
}

func (p *scriptedPeer) waitFrames(t *testing.T, want ...string) {
	t.Helper()
	waitUntil(t, fmt.Sprintf("%d frames at the peer", len(want)), func() bool {
		_, frames := p.seen()
		return len(frames) >= len(want)
	})
	if _, got := p.seen(); !sameStrings(got, want) {
		t.Fatalf("frames at the peer: got %v, want %v", got, want)
	}
}

func readChatFrame(r io.Reader) (string, error) {
	var total uint32
	if err := binary.Read(r, binary.BigEndian, &total); err != nil {
		return "", err
	}
	body := make([]byte, total)
	if _, err := io.ReadFull(r, body); err != nil {
		return "", err
	}
	return string(body[1+int(body[0]):]), nil
}

// A chat stream opened on the circuit of a world host stays there for as long as it lives.
// When a direct connection to the peer appears, the next send moves the stream to it, the
// frames arrive in the order they were sent, and the old stream is closed on both sides.
func TestChatStreamMovesToADirectConnection(t *testing.T) {
	useSessionTestSettings(t, 8<<20)
	relay := newChatRelay(t, false)
	alice, bob := newChatNode(t), newChatNode(t)
	relayedLink(t, relay, alice.host, bob.host)

	if reply := sendChat(t, alice, bob.host.ID(), chatTestChannel, "m1"); !strings.Contains(reply, "new stream") {
		t.Errorf("first send: %s", reply)
	}
	old := chatEntry(alice, bob.host.ID())
	if old.rank != connRankRelayed {
		t.Fatalf("the stream should sit on the circuit, rank %d", old.rank)
	}
	if reply := sendChat(t, alice, bob.host.ID(), chatTestChannel, "m2"); !strings.Contains(reply, "reused stream") {
		t.Errorf("send with the circuit alone: %s", reply)
	}
	if chatEntry(alice, bob.host.ID()) != old {
		t.Fatalf("with the circuit alone the stream must stay")
	}

	directLink(t, alice.host, bob.host)
	start := time.Now()
	if reply := sendChat(t, alice, bob.host.ID(), chatTestChannel, "m3"); !strings.Contains(reply, "moved stream") {
		t.Errorf("send after the direct connection: %s", reply)
	}
	// The send waited for bob to close the old stream, not for the time a peer that
	// leaves it open is given.
	if took := time.Since(start); took > ChatStreamDrainTimeout/2 {
		t.Errorf("the send that moved the stream took %v", took)
	}
	moved := chatEntry(alice, bob.host.ID())
	if moved == old || moved.rank != connRankDirect {
		t.Fatalf("the stream should have moved to the direct connection, rank %d", moved.rank)
	}
	if reply := sendChat(t, alice, bob.host.ID(), chatTestChannel, "m4"); !strings.Contains(reply, "reused stream") {
		t.Errorf("send after the move: %s", reply)
	}
	if chatEntry(alice, bob.host.ID()) != moved {
		t.Errorf("the stream on the direct connection must stay")
	}
	waitQueued(t, bob, chatTestChannel, "m1", "m2", "m3", "m4")

	// Bob closed the old stream when he had read it to the end, which ends its reader here.
	waitClosed(t, "the old stream", old.done)

	// Bob answers on the stream that reached him last, the one on the direct connection.
	waitUntil(t, "the stream of bob on the direct connection", func() bool {
		entry := chatEntry(bob, alice.host.ID())
		return entry != nil && entry.rank == connRankDirect
	})
	sendChat(t, bob, alice.host.ID(), "bob::dm:alice-test", "r1")
	waitQueued(t, alice, "bob::dm:alice-test", "r1")
	if chatEntry(alice, bob.host.ID()) != moved {
		t.Errorf("the answer of bob should have come on the stream alice opened")
	}
}

// The rank of a connection is read the same way on its two ends: a direct one is the best,
// then the circuit of a relay with no limits (a world host), then the one of a relay with
// limits (an infrastructure relay).
func TestConnectionsAreRankedDirectThenCircuitThenLimitedCircuit(t *testing.T) {
	useSessionTestSettings(t, 8<<20)
	rankOnBothEnds := func(from, target host.Host) [2]int {
		t.Helper()
		var ranks [2]int
		for i, end := range [][2]host.Host{{from, target}, {target, from}} {
			var conns []network.Conn
			waitUntil(t, "the connection on both ends", func() bool {
				conns = end[0].Network().ConnsToPeer(end[1].ID())
				return len(conns) == 1
			})
			ranks[i] = connRank(conns[0])
		}
		return ranks
	}

	alice, bob := newLoopbackHost(t), newLoopbackHost(t)
	relayedLink(t, newChatRelay(t, true), alice, bob)
	if ranks := rankOnBothEnds(alice, bob); ranks != [2]int{connRankLimited, connRankLimited} {
		t.Errorf("circuit of a relay with limits: ranks %v", ranks)
	}

	carol, dave := newLoopbackHost(t), newLoopbackHost(t)
	relayedLink(t, newChatRelay(t, false), carol, dave)
	if ranks := rankOnBothEnds(carol, dave); ranks != [2]int{connRankRelayed, connRankRelayed} {
		t.Errorf("circuit of a relay with no limits: ranks %v", ranks)
	}

	erin, frank := newLoopbackHost(t), newLoopbackHost(t)
	connectNodes(t, erin, frank)
	if ranks := rankOnBothEnds(erin, frank); ranks != [2]int{connRankDirect, connRankDirect} {
		t.Errorf("direct connection: ranks %v", ranks)
	}
	if !(connRankDirect > connRankRelayed && connRankRelayed > connRankLimited) {
		t.Errorf("the ranks are out of order")
	}
}

// A stream opened when the direct connection is already there sits on it from the start.
func TestChatStreamOpensOnTheBestConnection(t *testing.T) {
	useSessionTestSettings(t, 8<<20)
	relay := newChatRelay(t, false)
	alice, bob := newChatNode(t), newChatNode(t)
	relayedLink(t, relay, alice.host, bob.host)
	directLink(t, alice.host, bob.host)

	if reply := sendChat(t, alice, bob.host.ID(), chatTestChannel, "m1"); !strings.Contains(reply, "new stream") {
		t.Errorf("first send: %s", reply)
	}
	first := chatEntry(alice, bob.host.ID())
	if first.rank != connRankDirect {
		t.Fatalf("the stream should sit on the direct connection, rank %d", first.rank)
	}
	if reply := sendChat(t, alice, bob.host.ID(), chatTestChannel, "m2"); !strings.Contains(reply, "reused stream") {
		t.Errorf("second send: %s", reply)
	}
	if chatEntry(alice, bob.host.ID()) != first {
		t.Errorf("the stream must stay where it is")
	}
	waitQueued(t, bob, chatTestChannel, "m1", "m2")
}

// The two streams are read by two goroutines of the peer. The first frame on the new one
// is written only when the peer has read the old one to its end, so that it cannot be
// queued before a frame that was sent earlier. And the old stream is closed for writing
// only: what the peer writes on it after the move is still read.
func TestNothingIsWrittenOnTheNewStreamBeforeTheOldOneIsReadToItsEnd(t *testing.T) {
	useSessionTestSettings(t, 8<<20)
	relay := newChatRelay(t, false)
	alice := newChatNode(t)
	slow := newScriptedPeer(t, true)
	slow.lastWords = "r1"
	relayedLink(t, relay, alice.host, slow.host)

	// The frame is written, and the peer has not read it yet.
	sendChat(t, alice, slow.host.ID(), chatTestChannel, "m1")
	directLink(t, alice.host, slow.host)

	sent := make(chan error, 1)
	go func() {
		_, err := sendOnChatStream(alice, slow.host.ID(), chatTestChannel, []byte("m2"))
		sent <- err
	}()
	waitUntil(t, "the move of the stream", func() bool {
		return chatEntry(alice, slow.host.ID()).rank == connRankDirect
	})
	time.Sleep(150 * time.Millisecond)
	if _, frames := slow.seen(); len(frames) != 0 {
		t.Fatalf("a frame went on the new stream while the old one was not read: %v", frames)
	}
	select {
	case err := <-sent:
		t.Fatalf("the send returned while the old stream was not read (error: %v)", err)
	default:
	}

	close(slow.release)
	select {
	case err := <-sent:
		if err != nil {
			t.Fatalf("send across the move: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatalf("the send is still waiting after the peer closed the old stream")
	}
	slow.waitFrames(t, "m1", "m2")
	waitQueued(t, alice, chatTestChannel, "r1")
}

// A peer that is not a Go node may leave the old stream open when it has read it. The
// first frame on the new stream then waits ChatStreamDrainTimeout from the last write on
// the old one, and no longer.
func TestMoveWaitsABoundedTimeForAPeerThatLeavesTheOldStreamOpen(t *testing.T) {
	useSessionTestSettings(t, 8<<20)
	relay := newChatRelay(t, false)
	alice := newChatNode(t)
	browser := newScriptedPeer(t, false)
	close(browser.release)
	relayedLink(t, relay, alice.host, browser.host)

	lastWrite := time.Now()
	sendChat(t, alice, browser.host.ID(), chatTestChannel, "m1")
	directLink(t, alice.host, browser.host)
	sendChat(t, alice, browser.host.ID(), chatTestChannel, "m2")
	if waited := time.Since(lastWrite); waited < ChatStreamDrainTimeout {
		t.Errorf("the frame went on the new stream %v after the last write on the old one, before the %v it has to wait", waited, ChatStreamDrainTimeout)
	} else if waited > ChatStreamDrainTimeout+2*time.Second {
		t.Errorf("the send waited %v for a stream the peer never closes", waited)
	}
	browser.waitFrames(t, "m1", "m2")
}

// The wait is counted from the last write on the old stream: one that was quiet for longer
// is left at once, whatever the peer does with it.
func TestAStreamThatWasQuietIsLeftAtOnce(t *testing.T) {
	useSessionTestSettings(t, 8<<20)
	relay := newChatRelay(t, false)
	alice := newChatNode(t)
	browser := newScriptedPeer(t, false)
	close(browser.release)
	relayedLink(t, relay, alice.host, browser.host)

	sendChat(t, alice, browser.host.ID(), chatTestChannel, "m1")
	time.Sleep(ChatStreamDrainTimeout + 100*time.Millisecond)
	directLink(t, alice.host, browser.host)
	start := time.Now()
	sendChat(t, alice, browser.host.ID(), chatTestChannel, "m2")
	if took := time.Since(start); took > ChatStreamDrainTimeout/2 {
		t.Errorf("the send waited %v on a stream that had been quiet for more than %v", took, ChatStreamDrainTimeout)
	}
	browser.waitFrames(t, "m1", "m2")
}

// A stream on a better connection that does not open leaves the old one in use, and the
// move is not tried again at every send: a peer whose better connection stopped answering
// would cost each of them the time of the attempt.
func TestMoveThatFailsIsNotTriedAgainAtOnce(t *testing.T) {
	useSessionTestSettings(t, 8<<20)
	// Alice may hold two chat streams of her own at a time: the attempt to open a third fails.
	limits := rcmgr.PartialLimitConfig{
		Protocol: map[libp2pproto.ID]rcmgr.ResourceLimits{UnaiverseChatProtocol: {StreamsOutbound: 2}},
	}.Build(rcmgr.InfiniteLimits)
	manager, err := rcmgr.NewResourceManager(rcmgr.NewFixedLimiter(limits))
	if err != nil {
		t.Fatalf("resource manager: %v", err)
	}
	relay := newChatRelay(t, false)
	alice := newChatNode(t, libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"), libp2p.ResourceManager(manager))
	clock := &fakeClock{t: time.Unix(1_800_000_000, 0)}
	alice.clock = clock.now
	bob := newChatNode(t)
	relayedLink(t, relay, alice.host, bob.host)

	sendChat(t, alice, bob.host.ID(), chatTestChannel, "m1")
	old := chatEntry(alice, bob.host.ID())
	blocker, err := alice.host.NewStream(context.Background(), bob.host.ID(), UnaiverseChatProtocol)
	if err != nil {
		t.Fatalf("second stream: %v", err)
	}
	directLink(t, alice.host, bob.host)

	if reply := sendChat(t, alice, bob.host.ID(), chatTestChannel, "m2"); !strings.Contains(reply, "reused stream") {
		t.Errorf("send whose move failed: %s", reply)
	}
	if chatEntry(alice, bob.host.ID()) != old {
		t.Fatalf("the move could not open its stream, the old one must stay")
	}

	// The stream could open now, and the move is not due yet.
	blocker.Reset()
	clock.advance(ChatStreamMoveRetry - time.Second)
	sendChat(t, alice, bob.host.ID(), chatTestChannel, "m3")
	if chatEntry(alice, bob.host.ID()) != old {
		t.Fatalf("the move was tried again %v after it failed", ChatStreamMoveRetry-time.Second)
	}

	clock.advance(time.Second)
	if reply := sendChat(t, alice, bob.host.ID(), chatTestChannel, "m4"); !strings.Contains(reply, "moved stream") {
		t.Errorf("send after the retry time: %s", reply)
	}
	if rank := chatEntry(alice, bob.host.ID()).rank; rank != connRankDirect {
		t.Fatalf("the stream should have moved, rank %d", rank)
	}
	waitQueued(t, bob, chatTestChannel, "m1", "m2", "m3", "m4")
}

// A sender that was waiting for its turn on a stream finds it retired when the turn comes:
// it takes the stream that replaced it, and does not move the old one a second time.
func TestASenderThatFindsItsStreamRetiredTakesTheNewOne(t *testing.T) {
	useSessionTestSettings(t, 8<<20)
	relay := newChatRelay(t, false)
	alice, bob := newChatNode(t), newChatNode(t)
	relayedLink(t, relay, alice.host, bob.host)
	sendChat(t, alice, bob.host.ID(), chatTestChannel, "m1")
	old := chatEntry(alice, bob.host.ID())
	directLink(t, alice.host, bob.host)

	// The test takes the place of a first sender: it holds the stream and moves it, while
	// a second sender is waiting for the same stream.
	old.sendMu.Lock()
	sent := make(chan error, 1)
	go func() {
		_, err := sendOnChatStream(alice, bob.host.ID(), chatTestChannel, []byte("m2"))
		sent <- err
	}()
	time.Sleep(100 * time.Millisecond)
	next := moveChatStream(alice, bob.host.ID(), old)
	if next == nil {
		t.Fatalf("the stream did not move")
	}
	next.sendMu.Unlock()
	old.sendMu.Unlock()

	select {
	case err := <-sent:
		if err != nil {
			t.Fatalf("the second sender wrote on the retired stream: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatalf("the second sender is still waiting")
	}
	if chatEntry(alice, bob.host.ID()) != next {
		t.Errorf("the second sender moved the stream once more instead of taking the one that replaced it")
	}
	waitQueued(t, bob, chatTestChannel, "m1", "m2")
}

// Several senders write to the same peer while its stream moves: every frame arrives, and
// those of each sender in the order it sent them.
func TestNoFrameIsLostWhenTheStreamMovesUnderSeveralSenders(t *testing.T) {
	useSessionTestSettings(t, 8<<20)
	relay := newChatRelay(t, false)
	alice, bob := newChatNode(t), newChatNode(t)
	relayedLink(t, relay, alice.host, bob.host)

	const senders, frames = 4, 40
	var wg sync.WaitGroup
	failures := make(chan error, senders*frames)
	for s := 0; s < senders; s++ {
		wg.Add(1)
		go func(channel string) {
			defer wg.Done()
			for i := 0; i < frames; i++ {
				if _, err := sendOnChatStream(alice, bob.host.ID(), channel, []byte(fmt.Sprintf("%02d", i))); err != nil {
					failures <- err
				}
				time.Sleep(5 * time.Millisecond)
			}
		}(fmt.Sprintf("sender%d::dm:receiver-test", s))
	}
	time.Sleep(50 * time.Millisecond)
	directLink(t, alice.host, bob.host)
	wg.Wait()
	close(failures)
	for err := range failures {
		t.Errorf("a send failed: %v", err)
	}

	if rank := chatEntry(alice, bob.host.ID()).rank; rank != connRankDirect {
		t.Errorf("the stream should have moved, rank %d", rank)
	}
	var want []string
	for i := 0; i < frames; i++ {
		want = append(want, fmt.Sprintf("%02d", i))
	}
	for s := 0; s < senders; s++ {
		waitQueued(t, bob, fmt.Sprintf("sender%d::dm:receiver-test", s), want...)
	}
}
