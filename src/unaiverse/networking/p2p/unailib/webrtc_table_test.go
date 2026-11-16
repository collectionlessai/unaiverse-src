// Tests of the table of WebRTC DataChannels: which connection owns the entry of a peer.
package main

import (
	"context"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/test"
	pwebrtc "github.com/pion/webrtc/v4"
)

// webrtcPair is a DataChannel between two peer connections of this process: `local` is the
// end a node registers, `remote` plays the peer.
type webrtcPair struct {
	localPC, remotePC *pwebrtc.PeerConnection
	localDC, remoteDC *pwebrtc.DataChannel
}

// newWebRTCPair connects two peer connections over the loopback candidates (no ICE server)
// and returns once the DataChannel is open on both ends. The remote end is the offerer.
func newWebRTCPair(t *testing.T) *webrtcPair {
	t.Helper()
	must := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatalf("webrtc pair: %v", err)
		}
	}
	remotePC, err := pwebrtc.NewPeerConnection(pwebrtc.Configuration{})
	must(err)
	localPC, err := pwebrtc.NewPeerConnection(pwebrtc.Configuration{})
	must(err)
	t.Cleanup(func() {
		remotePC.Close()
		localPC.Close()
	})

	ordered := true
	remoteDC, err := remotePC.CreateDataChannel(WebRTCDataChannelLabel, &pwebrtc.DataChannelInit{Ordered: &ordered})
	must(err)
	remoteOpen := make(chan struct{})
	remoteDC.OnOpen(func() { close(remoteOpen) })
	localOpen := make(chan *pwebrtc.DataChannel, 1)
	localPC.OnDataChannel(func(dc *pwebrtc.DataChannel) {
		dc.OnOpen(func() { localOpen <- dc })
	})

	offer, err := remotePC.CreateOffer(nil)
	must(err)
	gathered := pwebrtc.GatheringCompletePromise(remotePC)
	must(remotePC.SetLocalDescription(offer))
	<-gathered
	must(localPC.SetRemoteDescription(*remotePC.LocalDescription()))
	answer, err := localPC.CreateAnswer(nil)
	must(err)
	gathered = pwebrtc.GatheringCompletePromise(localPC)
	must(localPC.SetLocalDescription(answer))
	<-gathered
	must(remotePC.SetRemoteDescription(*localPC.LocalDescription()))

	pair := &webrtcPair{localPC: localPC, remotePC: remotePC, remoteDC: remoteDC}
	select {
	case pair.localDC = <-localOpen:
	case <-time.After(15 * time.Second):
		t.Fatal("webrtc pair: the DataChannel did not open")
	}
	select {
	case <-remoteOpen:
	case <-time.After(15 * time.Second):
		t.Fatal("webrtc pair: the DataChannel did not open on the offerer")
	}
	return pair
}

// newWebRTCInstance returns an instance that can register DataChannels (no libp2p host).
func newWebRTCInstance(t *testing.T) *NodeInstance {
	t.Helper()
	return &NodeInstance{
		ctx:               context.Background(),
		messageStore:      newMessageStore(),
		webrtcConnections: make(map[peer.ID]*WebRTCConn),
	}
}

func tableEntry(ni *NodeInstance, pid peer.ID) *WebRTCConn {
	ni.webrtcMutex.RLock()
	defer ni.webrtcMutex.RUnlock()
	return ni.webrtcConnections[pid]
}

func waitUntil(t *testing.T, what string, ok func() bool) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if ok() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

// A peer can close its DataChannel and open a new one while the peer connection of the old
// one is still up on our side. The new channel takes the entry of the peer; when the old
// connection ends, later, it must not take the entry of the new one with it.
func TestReplacedWebRTCConnectionLeavesTheEntryOfItsSuccessor(t *testing.T) {
	ni := newWebRTCInstance(t)
	pid := test.RandPeerIDFatal(t)

	older := newWebRTCPair(t)
	registerWebRTCDataChannel(ni, pid, older.localDC, older.localPC)
	newer := newWebRTCPair(t)
	registerWebRTCDataChannel(ni, pid, newer.localDC, newer.localPC)
	current := tableEntry(ni, pid)
	if current == nil || current.dc != newer.localDC {
		t.Fatal("the newer DataChannel did not take the entry of the peer")
	}

	older.remotePC.Close()
	waitUntil(t, "the end of the older connection", func() bool {
		return older.localPC.ConnectionState() == pwebrtc.PeerConnectionStateClosed
	})
	// The cleanup runs in a goroutine of pion after the state change: give it time.
	time.Sleep(300 * time.Millisecond)
	if tableEntry(ni, pid) != current {
		t.Fatal("the end of the older connection removed the entry of the newer one")
	}

	newer.remotePC.Close()
	waitUntil(t, "the removal of the entry", func() bool { return tableEntry(ni, pid) == nil })
}
