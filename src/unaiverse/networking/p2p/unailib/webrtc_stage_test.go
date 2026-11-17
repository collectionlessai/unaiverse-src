// Tests of the two stages of a WebRTC offer: which ICE servers each stage has, when the
// TURN stage is made, what the answerer does with the stage of an offer, and what the
// TURN stage does over an entry of the table.
package main

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"net"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	pwebrtc "github.com/pion/webrtc/v4"
)

// newTestSTUNServer answers STUN binding requests on a UDP port of the loopback and
// returns the port. Nothing listens on the same port over TCP, so a TURN URL with
// transport=tcp on it gives the reflexive address at once and no relay candidate: an ICE
// list with a TURN server whose gathering takes no time.
func newTestSTUNServer(t *testing.T) int {
	t.Helper()
	conn, err := net.ListenUDP("udp4", &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1)})
	if err != nil {
		t.Fatalf("stun server: %v", err)
	}
	t.Cleanup(func() { conn.Close() })
	go func() {
		request := make([]byte, 1500)
		for {
			n, from, err := conn.ReadFromUDP(request)
			if err != nil {
				return
			}
			if n < 20 || binary.BigEndian.Uint16(request[0:2]) != 0x0001 {
				continue
			}
			// A binding success with the address of the sender, XOR-mapped.
			reply := make([]byte, 32)
			binary.BigEndian.PutUint16(reply[0:2], 0x0101)
			binary.BigEndian.PutUint16(reply[2:4], 12)
			copy(reply[4:20], request[4:20])
			binary.BigEndian.PutUint16(reply[20:22], 0x0020)
			binary.BigEndian.PutUint16(reply[22:24], 8)
			reply[25] = 0x01
			binary.BigEndian.PutUint16(reply[26:28], uint16(from.Port)^0x2112)
			for i, b := range from.IP.To4() {
				reply[28+i] = b ^ request[4+i]
			}
			conn.WriteToUDP(reply, from)
		}
	}()
	return conn.LocalAddr().(*net.UDPAddr).Port
}

// withTURN gives a node the ICE config of a node of ours: the TURN credentials of the
// root server and nothing else.
func withTURN(node *sessionNode, port int) string {
	url := fmt.Sprintf("turn:127.0.0.1:%d?transport=tcp", port)
	node.ni.iceConfig = &ICEConfig{TURNServers: []TURNServer{{URLs: []string{url}, Username: "user", Credential: "secret"}}}
	return url
}

// iceURLs returns the URLs of the ICE servers a peer connection was built with, and
// whether any of them came with a credential.
func iceURLs(pc *pwebrtc.PeerConnection) (urls []string, credentials bool) {
	for _, server := range pc.GetConfiguration().ICEServers {
		urls = append(urls, server.URLs...)
		credentials = credentials || server.Username != ""
	}
	return urls, credentials
}

func hasTURN(urls []string) bool {
	for _, url := range urls {
		if strings.HasPrefix(url, "turn") {
			return true
		}
	}
	return false
}

// What a scripted answerer does with a signalling stream.
const (
	// stepRefuse reads the offer, answers with an error and reads what follows.
	stepRefuse = iota
	// stepSilent reads the offer, answers nothing and reads what follows.
	stepSilent
	// stepClose closes the stream at once, as a browser does to an offer that finds its
	// previous exchange with the peer still running.
	stepClose
	// stepAnswerOnly reads the offer, answers it from a peer connection that is closed at
	// once, so that nothing ever connects, and reads what follows.
	stepAnswerOnly
)

// scriptedAnswerer stands in front of the signalling handler of a node. The streams that
// arrive take the steps of the script in order; with the script used up they go to the
// production handler.
type scriptedAnswerer struct {
	node *sessionNode
	// afterRefusal runs once a refusal has been written.
	afterRefusal func()

	mu      sync.Mutex
	steps   []int
	streams int
	offers  []SignalMessage
	aborts  []SignalMessage
}

func scriptAnswerer(node *sessionNode, steps ...int) *scriptedAnswerer {
	a := &scriptedAnswerer{node: node, steps: steps}
	node.ni.host.SetStreamHandler(UnaiverseWebRTCSignalProtocol, a.handle)
	return a
}

func (a *scriptedAnswerer) handle(s network.Stream) {
	a.mu.Lock()
	a.streams++
	step := -1
	if len(a.steps) > 0 {
		step, a.steps = a.steps[0], a.steps[1:]
	}
	a.mu.Unlock()

	switch step {
	case stepClose:
		s.Close()
	case stepRefuse, stepSilent, stepAnswerOnly:
		defer s.Close()
		offer, err := readSignalMessage(s)
		if err != nil {
			return
		}
		a.mu.Lock()
		a.offers = append(a.offers, offer)
		a.mu.Unlock()
		if step == stepRefuse {
			writeSignalMessage(s, SignalMessage{Type: "error", Message: "refused by the test"})
			if a.afterRefusal != nil {
				a.afterRefusal()
			}
		}
		if step == stepAnswerOnly {
			writeSignalMessage(s, SignalMessage{Type: "answer", SDP: deadAnswer(offer.SDP)})
		}
		if next, err := readSignalMessage(s); err == nil {
			a.mu.Lock()
			a.aborts = append(a.aborts, next)
			a.mu.Unlock()
		}
	default:
		handleSignalingStream(a.node.ni, s)
	}
}

// deadAnswer returns a valid answer to an offer from a peer connection that no longer
// exists: the offerer finds nobody behind its candidates.
func deadAnswer(offerSDP string) string {
	pc, err := pwebrtc.NewPeerConnection(pwebrtc.Configuration{})
	if err != nil {
		return ""
	}
	defer pc.Close()
	if pc.SetRemoteDescription(pwebrtc.SessionDescription{Type: pwebrtc.SDPTypeOffer, SDP: offerSDP}) != nil {
		return ""
	}
	answer, err := pc.CreateAnswer(nil)
	if err != nil {
		return ""
	}
	gathered := pwebrtc.GatheringCompletePromise(pc)
	if pc.SetLocalDescription(answer) != nil {
		return ""
	}
	<-gathered
	return pc.LocalDescription().SDP
}

// seen returns how many signalling streams reached the node, and the offers and the
// messages after them that the script read.
func (a *scriptedAnswerer) seen() (streams int, offers, aborts []SignalMessage) {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.streams, append([]SignalMessage(nil), a.offers...), append([]SignalMessage(nil), a.aborts...)
}

// relayedNodes returns two nodes that reach each other through a relay alone, as the two
// ends of a custom offer do: the TURN stage is not made toward a peer with a direct
// libp2p connection.
func relayedNodes(t *testing.T) (offerer, answerer *sessionNode) {
	t.Helper()
	offerer, answerer = newSessionNode(t), newSessionNode(t)
	relayedLink(t, newChatRelay(t, false), offerer.ni.host, answerer.ni.host)
	return offerer, answerer
}

// rawOffer makes a host that is not a node offer a DataChannel to a node, with the given
// stage in the offer ("" for a peer that names none). It returns the stream, left open,
// and the answer. The answer is applied only when connect is true.
func rawOffer(t *testing.T, from host.Host, to peer.ID, stage string, connect bool) (network.Stream, SignalMessage) {
	t.Helper()
	must := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatalf("raw offer: %v", err)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	s, err := from.NewStream(ctx, to, UnaiverseWebRTCSignalProtocol)
	must(err)
	t.Cleanup(func() { s.Close() })
	pc, err := pwebrtc.NewPeerConnection(pwebrtc.Configuration{})
	must(err)
	t.Cleanup(func() { pc.Close() })
	ordered := true
	_, err = pc.CreateDataChannel(WebRTCDataChannelLabel, &pwebrtc.DataChannelInit{Ordered: &ordered})
	must(err)
	offer, err := pc.CreateOffer(nil)
	must(err)
	gathered := pwebrtc.GatheringCompletePromise(pc)
	must(pc.SetLocalDescription(offer))
	<-gathered
	must(writeSignalMessage(s, SignalMessage{Type: "offer", Stage: stage, SDP: pc.LocalDescription().SDP}))
	answer, err := readSignalMessage(s)
	must(err)
	if connect {
		must(pc.SetRemoteDescription(pwebrtc.SessionDescription{Type: pwebrtc.SDPTypeAnswer, SDP: answer.SDP}))
	}
	return s, answer
}

// The direct stage has STUN alone: the STUN servers of the config, and each TURN server
// as the STUN server it also is, with no credential. A config that names no STUN server
// gets the public ones next to its TURN server, each once. Every other stage, and the
// offer of a peer that names none, has the list as it was before the stages.
func TestICEListOfEachStage(t *testing.T) {
	prev := defaultSTUNServers
	defaultSTUNServers = []string{"stun:default.example:3478"}
	t.Cleanup(func() { defaultSTUNServers = prev })
	urlsOf := func(cfg *ICEConfig, stage string) ([]string, bool) {
		var urls []string
		credentials := false
		for _, server := range getWebRTCConfig(&NodeInstance{iceConfig: cfg}, stage).ICEServers {
			urls = append(urls, server.URLs...)
			credentials = credentials || server.Username != "" || server.Credential != nil
		}
		return urls, credentials
	}
	turn := TURNServer{
		URLs:     []string{"turn:turn.example:3478", "turn:turn.example:3478?transport=tcp", "turns:turn.example:5349", "stun:turn.example:3478"},
		Username: "user", Credential: "secret",
	}
	both := &ICEConfig{STUNServers: []string{"stun:stun.example:3478"}, TURNServers: []TURNServer{turn}}
	turnOnly := &ICEConfig{TURNServers: []TURNServer{turn}}

	if urls, credentials := urlsOf(both, WebRTCStageDirect); credentials ||
		!sameStrings(urls, []string{"stun:stun.example:3478", "stun:turn.example:3478", "stun:turn.example:5349"}) {
		t.Errorf("direct stage: %v (credentials: %v)", urls, credentials)
	}
	if urls, credentials := urlsOf(turnOnly, WebRTCStageDirect); credentials ||
		!sameStrings(urls, []string{"stun:turn.example:3478", "stun:turn.example:5349", "stun:default.example:3478"}) {
		t.Errorf("direct stage of a config with TURN alone: %v (credentials: %v)", urls, credentials)
	}
	// A public server that the config already reaches through its TURN entry is not listed twice.
	sameServer := &ICEConfig{TURNServers: []TURNServer{{URLs: []string{"turn:default.example:3478"}, Username: "user", Credential: "secret"}}}
	if urls, _ := urlsOf(sameServer, WebRTCStageDirect); !sameStrings(urls, []string{"stun:default.example:3478"}) {
		t.Errorf("direct stage of a config whose TURN server is a public STUN one: %v", urls)
	}
	// In the stages with the whole list nothing is added.
	if urls, _ := urlsOf(turnOnly, WebRTCStageTURN); !sameStrings(urls, turn.URLs) {
		t.Errorf("TURN stage of a config with TURN alone: %v", urls)
	}
	for _, stage := range []string{WebRTCStageTURN, ""} {
		if urls, credentials := urlsOf(both, stage); !credentials ||
			!sameStrings(urls, append([]string{"stun:stun.example:3478"}, turn.URLs...)) {
			t.Errorf("stage %q: %v (credentials: %v)", stage, urls, credentials)
		}
	}
	for _, stage := range []string{WebRTCStageDirect, WebRTCStageTURN, ""} {
		if urls, _ := urlsOf(nil, stage); !sameStrings(urls, defaultSTUNServers) {
			t.Errorf("stage %q with no config: %v", stage, urls)
		}
	}
}

// An offer or an answer does not wait for the end of the gathering. With a server that
// never answers, the direct stage is ready after the silence that follows its last
// candidate, with the gathering still running; where the list has TURN, candidates that
// are not relay ones end nothing and the wait goes to its limit.
func TestCandidatesAreReadyBeforeTheGatheringEnds(t *testing.T) {
	prevQuiet, prevDirect, prevTURN := WebRTCGatherQuiet, WebRTCDirectGatherMax, WebRTCTURNGatherMax
	WebRTCGatherQuiet, WebRTCDirectGatherMax, WebRTCTURNGatherMax = 200*time.Millisecond, 3*time.Second, 700*time.Millisecond
	t.Cleanup(func() { WebRTCGatherQuiet, WebRTCDirectGatherMax, WebRTCTURNGatherMax = prevQuiet, prevDirect, prevTURN })

	silent, err := net.ListenUDP("udp4", &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1)})
	if err != nil {
		t.Fatalf("udp port: %v", err)
	}
	t.Cleanup(func() { silent.Close() })
	silentSTUN := fmt.Sprintf("stun:127.0.0.1:%d", silent.LocalAddr().(*net.UDPAddr).Port)

	// gather starts the gathering of a new peer connection and waits as an offer does.
	gather := func(stage string, timeout time.Duration, urls ...string) (*pwebrtc.PeerConnection, string, bool, time.Duration) {
		t.Helper()
		var cfg pwebrtc.Configuration
		if len(urls) > 0 {
			cfg.ICEServers = []pwebrtc.ICEServer{{URLs: urls}}
		}
		pc, err := pwebrtc.NewPeerConnection(cfg)
		if err != nil {
			t.Fatalf("peer connection: %v", err)
		}
		t.Cleanup(func() { pc.Close() })
		if _, err := pc.CreateDataChannel(WebRTCDataChannelLabel, nil); err != nil {
			t.Fatalf("data channel: %v", err)
		}
		ready := watchICECandidates(pc, stage)
		offer, err := pc.CreateOffer(nil)
		if err != nil {
			t.Fatalf("offer: %v", err)
		}
		if err := pc.SetLocalDescription(offer); err != nil {
			t.Fatalf("local description: %v", err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		start := time.Now()
		reason, ok := ready(ctx)
		return pc, reason, ok, time.Since(start)
	}

	pc, reason, ok, took := gather(WebRTCStageDirect, 10*time.Second, silentSTUN)
	if !ok || reason != "quiet" || took < WebRTCGatherQuiet || took > 2*time.Second {
		t.Errorf("direct stage with a server that never answers: %q, %v, after %v", reason, ok, took)
	}
	if state := pc.ICEGatheringState(); state == pwebrtc.ICEGatheringStateComplete {
		t.Errorf("the gathering was over (%v): the wait proved nothing", state)
	}
	if sdp := pc.LocalDescription().SDP; !strings.Contains(sdp, "typ host") {
		t.Errorf("the description that would be sent has no host candidate:\n%s", sdp)
	}

	if _, reason, ok, took := gather(WebRTCStageDirect, 10*time.Second); !ok || reason != "complete" || took >= WebRTCGatherQuiet {
		t.Errorf("direct stage with no server: %q, %v, after %v", reason, ok, took)
	}

	for _, stage := range []string{WebRTCStageTURN, ""} {
		if _, reason, ok, took := gather(stage, 10*time.Second, silentSTUN); !ok || reason != "limit" ||
			took < WebRTCTURNGatherMax || took > 2*time.Second {
			t.Errorf("stage %q with no relay candidate: %q, %v, after %v", stage, reason, ok, took)
		}
	}

	if _, reason, ok, took := gather(WebRTCStageTURN, 100*time.Millisecond, silentSTUN); ok || took > 500*time.Millisecond {
		t.Errorf("a wait whose context ended first: %q, %v, after %v", reason, ok, took)
	}

	// The direct stage has a limit of its own, reached when the silence never comes.
	WebRTCGatherQuiet, WebRTCDirectGatherMax = 5*time.Second, 300*time.Millisecond
	if _, reason, ok, took := gather(WebRTCStageDirect, 10*time.Second, silentSTUN); !ok || reason != "limit" ||
		took < WebRTCDirectGatherMax || took >= WebRTCTURNGatherMax {
		t.Errorf("direct stage with a silence longer than its limit: %q, %v, after %v", reason, ok, took)
	}
}

// Our offer is made in the direct stage first: it names the stage, neither side has a
// TURN server in its peer connection although both have one in their config, and when
// the channel opens no second offer follows.
func TestOfferIsMadeInTheDirectStageWithNoTURNOnEitherSide(t *testing.T) {
	useSessionTestSettings(t, 8<<20)
	offerer, answerer := relayedNodes(t)
	port := newTestSTUNServer(t)
	withTURN(offerer, port)
	withTURN(answerer, port)
	script := scriptAnswerer(answerer)

	if err := initiateWebRTCConnection(offerer.ni, answerer.id()); err != nil {
		t.Fatalf("offer: %v", err)
	}
	waitUntil(t, "the entry of the answerer", func() bool { return tableEntry(answerer.ni, offerer.id()) != nil })
	for side, entry := range map[string]*WebRTCConn{"offerer": tableEntry(offerer.ni, answerer.id()), "answerer": tableEntry(answerer.ni, offerer.id())} {
		if entry.stage != WebRTCStageDirect {
			t.Errorf("%s: the channel is recorded with stage %q", side, entry.stage)
		}
		if urls, credentials := iceURLs(entry.pc); hasTURN(urls) || credentials || len(urls) != 1 {
			t.Errorf("%s: ICE servers of the direct stage: %v (credentials: %v)", side, urls, credentials)
		}
		if path := webRTCPath(entry.pc); !strings.HasPrefix(path, "direct") {
			t.Errorf("%s: path %q", side, path)
		}
	}
	if streams, _, _ := script.seen(); streams != 1 {
		t.Errorf("%d offers reached the answerer, one was due", streams)
	}
}

// When the direct stage fails, the answerer is told that the offer is given up, and after
// a pause the offer is made again in the TURN stage, with the whole ICE list on both sides.
func TestTURNStageFollowsADirectStageThatFails(t *testing.T) {
	useSessionTestSettings(t, 8<<20)
	offerer, answerer := relayedNodes(t)
	port := newTestSTUNServer(t)
	turnURL := withTURN(offerer, port)
	withTURN(answerer, port)
	script := scriptAnswerer(answerer, stepRefuse)

	start := time.Now()
	if err := initiateWebRTCConnection(offerer.ni, answerer.id()); err != nil {
		t.Fatalf("offer: %v", err)
	}
	if took := time.Since(start); took < WebRTCStagePause {
		t.Errorf("the TURN stage followed the direct one after %v, the pause is %v", took, WebRTCStagePause)
	}
	waitUntil(t, "the entry of the answerer", func() bool { return tableEntry(answerer.ni, offerer.id()) != nil })
	streams, offers, aborts := script.seen()
	if streams != 2 || len(offers) != 1 || offers[0].Stage != WebRTCStageDirect {
		t.Fatalf("%d offers reached the answerer, the first with stage %+v", streams, offers)
	}
	if len(aborts) != 1 || aborts[0].Type != "abort" || aborts[0].Stage != WebRTCStageDirect {
		t.Errorf("the direct stage was not given up with an abort: %+v", aborts)
	}
	for side, entry := range map[string]*WebRTCConn{"offerer": tableEntry(offerer.ni, answerer.id()), "answerer": tableEntry(answerer.ni, offerer.id())} {
		if entry.stage != WebRTCStageTURN {
			t.Errorf("%s: the channel is recorded with stage %q", side, entry.stage)
		}
		if urls, credentials := iceURLs(entry.pc); !sameStrings(urls, []string{turnURL}) || !credentials {
			t.Errorf("%s: ICE servers of the TURN stage: %v (credentials: %v)", side, urls, credentials)
		}
	}
}

// An offer has the time of its stage for everything, the opening of the channel included.
// One that gets no answer, or an answer and no connection, ends there, and the abort is
// written although the time of the stream is over.
func TestOfferOfAStageEndsAtItsBudgetAndSaysSo(t *testing.T) {
	for name, step := range map[string]int{"no answer": stepSilent, "an answer and no connection": stepAnswerOnly} {
		t.Run(name, func(t *testing.T) {
			useSessionTestSettings(t, 8<<20)
			offerer, answerer := relayedNodes(t)
			script := scriptAnswerer(answerer, step)

			const budget = time.Second
			start := time.Now()
			err := offerWebRTC(offerer.ni, answerer.id(), WebRTCStageDirect, budget)
			took := time.Since(start)
			if err == nil {
				t.Fatal("an offer that could not open its channel succeeded")
			}
			if took < budget || took > budget+WebRTCAbortWriteTimeout+time.Second {
				t.Errorf("the offer ended after %v, its budget is %v", took, budget)
			}
			waitUntil(t, "the abort at the answerer", func() bool {
				_, _, aborts := script.seen()
				return len(aborts) == 1
			})
			if _, _, aborts := script.seen(); aborts[0].Type != "abort" || aborts[0].Stage != WebRTCStageDirect {
				t.Errorf("message after the offer: %+v", aborts[0])
			}
			if tableEntry(offerer.ni, answerer.id()) != nil {
				t.Errorf("an offer that failed left an entry")
			}
		})
	}
}

// An offer that is given up before it was sent writes nothing: the answerer has nothing
// to give up, and an abort would reach it as a first message that is not an offer.
func TestOfferGivenUpBeforeItWasSentWritesNothing(t *testing.T) {
	useSessionTestSettings(t, 8<<20)
	offerer, answerer := relayedNodes(t)
	script := scriptAnswerer(answerer, stepSilent)
	// A STUN server that never answers: the gathering outlasts the budget of the offer.
	silent, err := net.ListenUDP("udp4", &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1)})
	if err != nil {
		t.Fatalf("udp port: %v", err)
	}
	t.Cleanup(func() { silent.Close() })
	offerer.ni.iceConfig = &ICEConfig{STUNServers: []string{fmt.Sprintf("stun:127.0.0.1:%d", silent.LocalAddr().(*net.UDPAddr).Port)}}

	if err := offerWebRTC(offerer.ni, answerer.id(), WebRTCStageDirect, 300*time.Millisecond); err == nil {
		t.Fatal("an offer that never left succeeded")
	}
	time.Sleep(300 * time.Millisecond)
	if _, offers, _ := script.seen(); len(offers) != 0 {
		t.Errorf("a message reached the answerer for an offer that was never sent: %+v", offers)
	}
}

// A browser closes the stream of an offer that finds its previous exchange still running.
// The TURN stage is then tried once more after a while, and only once.
func TestTURNStageIsRetriedOnceWhenTheAnswererClosesItsStream(t *testing.T) {
	t.Run("the second attempt is answered", func(t *testing.T) {
		useSessionTestSettings(t, 8<<20)
		offerer, answerer := relayedNodes(t)
		script := scriptAnswerer(answerer, stepRefuse, stepClose)
		start := time.Now()
		if err := initiateWebRTCConnection(offerer.ni, answerer.id()); err != nil {
			t.Fatalf("offer: %v", err)
		}
		if took := time.Since(start); took < WebRTCStagePause+WebRTCStageRetryDelay {
			t.Errorf("the retry came after %v", took)
		}
		if streams, _, _ := script.seen(); streams != 3 {
			t.Errorf("%d offers reached the answerer, three were due", streams)
		}
		if entry := tableEntry(offerer.ni, answerer.id()); entry == nil || entry.stage != WebRTCStageTURN {
			t.Errorf("no channel of the TURN stage: %+v", entry)
		}
	})
	t.Run("an answer that refuses is not asked again", func(t *testing.T) {
		useSessionTestSettings(t, 8<<20)
		offerer, answerer := relayedNodes(t)
		script := scriptAnswerer(answerer, stepRefuse, stepRefuse)
		err := initiateWebRTCConnection(offerer.ni, answerer.id())
		if err == nil || errors.Is(err, errSignalingStreamEnded) {
			t.Fatalf("error of an offer refused in both stages: %v", err)
		}
		if streams, offers, _ := script.seen(); streams != 2 || len(offers) != 2 || offers[1].Stage != WebRTCStageTURN {
			t.Errorf("%d offers reached the answerer: %+v", streams, offers)
		}
	})
	t.Run("the second attempt is closed too", func(t *testing.T) {
		useSessionTestSettings(t, 8<<20)
		offerer, answerer := relayedNodes(t)
		script := scriptAnswerer(answerer, stepRefuse, stepClose, stepClose)
		err := initiateWebRTCConnection(offerer.ni, answerer.id())
		if !errors.Is(err, errSignalingStreamEnded) {
			t.Fatalf("error of an offer whose stream was closed twice: %v", err)
		}
		if streams, _, _ := script.seen(); streams != 3 {
			t.Errorf("%d offers reached the answerer, three were due", streams)
		}
	})
}

// The TURN stage is an offer like any other: it is not made when the peer gained a path
// that is not relayed while the direct stage was failing.
func TestTURNStageIsNotMadeWhenAPathAppearedMeanwhile(t *testing.T) {
	t.Run("a direct libp2p connection", func(t *testing.T) {
		useSessionTestSettings(t, 8<<20)
		offerer, answerer := relayedNodes(t)
		script := scriptAnswerer(answerer, stepRefuse)
		// Hole punching would do this in the field. The handler of the script is not the
		// goroutine of the test, so a failure is only reported here.
		script.afterRefusal = func() {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			from, target := offerer.ni.host, answerer.ni.host
			from.Peerstore().AddAddrs(target.ID(), target.Addrs(), time.Hour)
			if _, err := from.Network().DialPeer(network.WithForceDirectDial(ctx, "test"), target.ID()); err != nil {
				t.Errorf("direct connection: %v", err)
			}
		}
		if err := initiateWebRTCConnection(offerer.ni, answerer.id()); err != nil {
			t.Fatalf("offer: %v", err)
		}
		if streams, _, _ := script.seen(); streams != 1 {
			t.Errorf("%d offers reached the answerer, the TURN stage was not due", streams)
		}
		if tableEntry(offerer.ni, answerer.id()) != nil {
			t.Errorf("a DataChannel was made")
		}
	})

	t.Run("the old DataChannel gives a sign of life", func(t *testing.T) {
		useSessionTestSettings(t, 8<<20)
		offerer, answerer := relayedNodes(t)
		if err := initiateWebRTCConnection(offerer.ni, answerer.id()); err != nil {
			t.Fatalf("first offer: %v", err)
		}
		waitUntil(t, "the entry of the answerer", func() bool { return tableEntry(answerer.ni, offerer.id()) != nil })
		first, firstOnAnswerer := tableEntry(offerer.ni, answerer.id()), tableEntry(answerer.ni, offerer.id())
		offerer.clock.advance(WebRTCFreshnessWindow + time.Second)

		script := scriptAnswerer(answerer, stepRefuse)
		script.afterRefusal = func() {
			if err := writeToWebRTCDataChannel(firstOnAnswerer, chatTestChannel, []byte("alive")); err != nil {
				t.Errorf("frame on the old channel: %v", err)
			}
		}
		if err := initiateWebRTCConnection(offerer.ni, answerer.id()); err != nil {
			t.Fatalf("offer over an entry that is not fresh: %v", err)
		}
		if streams, _, _ := script.seen(); streams != 1 {
			t.Errorf("%d offers reached the answerer, the TURN stage was not due", streams)
		}
		if tableEntry(offerer.ni, answerer.id()) != first {
			t.Errorf("the entry changed")
		}
	})
}

// Over an entry that is open and not fresh, and still so after a direct stage that
// failed, the TURN stage is made, and its channel takes the entry on both sides with the
// replaced connections closed: a channel that may run through TURN replaces one that may
// have been direct and alive.
func TestTURNStageOverAnEntryThatIsNotFreshReplacesIt(t *testing.T) {
	useSessionTestSettings(t, 8<<20)
	offerer, answerer := relayedNodes(t)
	if err := initiateWebRTCConnection(offerer.ni, answerer.id()); err != nil {
		t.Fatalf("first offer: %v", err)
	}
	waitUntil(t, "the entry of the answerer", func() bool { return tableEntry(answerer.ni, offerer.id()) != nil })
	first, firstOnAnswerer := tableEntry(offerer.ni, answerer.id()), tableEntry(answerer.ni, offerer.id())
	offerer.clock.advance(WebRTCFreshnessWindow + time.Second)

	script := scriptAnswerer(answerer, stepRefuse)
	if err := initiateWebRTCConnection(offerer.ni, answerer.id()); err != nil {
		t.Fatalf("offer over an entry that is not fresh: %v", err)
	}
	waitUntil(t, "the new entry of the answerer", func() bool { return tableEntry(answerer.ni, offerer.id()) != firstOnAnswerer })
	if streams, _, _ := script.seen(); streams != 2 {
		t.Errorf("%d offers reached the answerer, two were due", streams)
	}
	for side, entries := range map[string][2]*WebRTCConn{
		"offerer":  {first, tableEntry(offerer.ni, answerer.id())},
		"answerer": {firstOnAnswerer, tableEntry(answerer.ni, offerer.id())},
	} {
		replaced, current := entries[0], entries[1]
		if current == nil || current == replaced || !current.isOpen() || current.stage != WebRTCStageTURN {
			t.Fatalf("%s: the channel of the TURN stage did not take the entry", side)
		}
		waitUntil(t, "the replaced peer connection of the "+side+" to close", func() bool {
			return replaced.pc.ConnectionState() == pwebrtc.PeerConnectionStateClosed
		})
	}
}

// As answerer the stage of the offer decides the ICE list of this side: STUN alone for a
// direct-stage offer, the whole list for a TURN-stage one and for the offer of a peer
// that names no stage.
func TestAnswererHonoursTheStageOfTheOffer(t *testing.T) {
	for _, stage := range []string{WebRTCStageDirect, WebRTCStageTURN, ""} {
		t.Run("stage "+stage, func(t *testing.T) {
			useSessionTestSettings(t, 8<<20)
			node, offerer := newSessionNode(t), newLoopbackHost(t)
			turnURL := withTURN(node, newTestSTUNServer(t))
			connectNodes(t, offerer, node.ni.host)
			rawOffer(t, offerer, node.id(), stage, true)
			waitUntil(t, "the entry of the offerer", func() bool { return tableEntry(node.ni, offerer.ID()) != nil })
			entry := tableEntry(node.ni, offerer.ID())
			if entry.stage != stage {
				t.Errorf("the channel is recorded with stage %q", entry.stage)
			}
			urls, credentials := iceURLs(entry.pc)
			if stage == WebRTCStageDirect {
				if hasTURN(urls) || credentials || len(urls) != 1 {
					t.Errorf("ICE servers for a direct-stage offer: %v (credentials: %v)", urls, credentials)
				}
			} else if !sameStrings(urls, []string{turnURL}) || !credentials {
				t.Errorf("ICE servers: %v (credentials: %v)", urls, credentials)
			}
		})
	}
}

// An offerer that stages its offers says so when it gives one up. The answerer stops
// waiting for that DataChannel at once: it closes the connection it built and its end of
// the stream, where it would have waited for the whole time of the signalling.
func TestAnswererStopsWaitingWhenTheOffererGivesUp(t *testing.T) {
	useSessionTestSettings(t, 8<<20)
	node, offerer := newSessionNode(t), newLoopbackHost(t)
	connectNodes(t, offerer, node.ni.host)
	s, _ := rawOffer(t, offerer, node.id(), WebRTCStageDirect, false)

	if err := writeSignalMessage(s, SignalMessage{Type: "abort", Stage: WebRTCStageDirect, Message: "given up by the test"}); err != nil {
		t.Fatalf("abort: %v", err)
	}
	// The end of the stream is what the answerer leaves when it lets the exchange go.
	_ = s.SetReadDeadline(time.Now().Add(3 * time.Second))
	_, err := readSignalMessage(s)
	var timeout net.Error
	if err == nil || (errors.As(err, &timeout) && timeout.Timeout()) {
		t.Fatalf("the answerer is still holding the exchange after the abort: %v", err)
	}
	if tableEntry(node.ni, offerer.ID()) != nil {
		t.Errorf("an offer that was given up left an entry")
	}
}
