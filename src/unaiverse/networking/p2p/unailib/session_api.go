// FFI exports backing the Python Session abstraction.
//
// These exports — SendOnProtocol (+ host-key signing/verification) — are what
// Python needs on top of the existing chat/pubsub surface to drive per-stage Session
// handshakes (see session/session.proto and session/session.py). Liveness is a
// session-level PING/PONG over the control protocol, not /ipfs/ping.
package main

/*
#include <stdlib.h>
*/
import "C"

import (
	"context"
	"encoding/base64"
	"fmt"
	"time"
	"unsafe"

	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	libp2pproto "github.com/libp2p/go-libp2p/core/protocol"
)

// SendOnProtocol opens a fresh outbound libp2p stream to peerID on the given
// protocol, writes a single length-prefixed protobuf payload, and closes the
// stream. The Session response (if any) comes back asynchronously as a
// PopMessages entry whose `protocol` field equals protocolIDC.
//
// Returns:
//
//	`{"state":"Success","message":"..."}` on send success;
//	`{"state":"Error","message":"..."}`   on failure.
//
// Caller MUST FreeString the returned *C.char.
//
//export SendOnProtocol
func SendOnProtocol(
	instanceIndexC C.int,
	peerIDC *C.char,
	protocolIDC *C.char,
	payloadC *C.char,
	payloadLenC C.int,
) (ret *C.char) {
	defer recoverToCString(&ret, "SendOnProtocol")
	ni, err := getInstance(int(instanceIndexC))
	if err != nil {
		return C.CString(jsonErrorResponse(fmt.Sprintf("Invalid instance index: %d", int(instanceIndexC)), err))
	}

	peerIDStr := C.GoString(peerIDC)
	protocolIDStr := C.GoString(protocolIDC)
	payload := C.GoBytes(unsafe.Pointer(payloadC), payloadLenC)

	pid, err := peer.Decode(peerIDStr)
	if err != nil {
		return C.CString(jsonErrorResponse(fmt.Sprintf("SendOnProtocol: bad peerID '%s'", peerIDStr), err))
	}
	if pid == ni.host.ID() {
		return C.CString(jsonErrorResponse("SendOnProtocol: refusing to dial self", nil))
	}

	if err := sendSessionFrame(ni, pid, protocolIDStr, payload); err != nil {
		return C.CString(jsonErrorResponse("SendOnProtocol", err))
	}
	return C.CString(jsonSuccessResponse(fmt.Sprintf("Sent %d bytes on %s to %s", len(payload), protocolIDStr, pid)))
}

// sendSessionFrame sends one Session frame to pid. It goes on the WebRTC DataChannel to the
// peer when there is one that takes it (see takesSessionFrames): where that channel is the
// only direct path, a stream would run on a relay circuit that is cut every two minutes.
// In every other case, and when that send fails, it opens a fresh stream on the Session
// protocol, writes the frame and closes the stream: the dial (when there is no connection),
// the opening of the stream and the write share the budget of SessionSendTimeout.
func sendSessionFrame(ni *NodeInstance, pid peer.ID, protocolID string, payload []byte) error {
	if sendSessionFrameOnDataChannel(ni, pid, protocolID, payload) {
		return nil
	}
	ctx, cancel := context.WithTimeout(ni.ctx, SessionSendTimeout)
	defer cancel()
	s, err := ni.host.NewStream(
		network.WithAllowLimitedConn(ctx, protocolID),
		pid,
		libp2pproto.ID(protocolID),
	)
	if err != nil {
		return fmt.Errorf("NewStream(%s, %s) failed: %w", pid, protocolID, err)
	}
	// The sending side of Session traffic must surface the peer to the pools
	// too (the receiving side does it in handleSessionStream). ConnectTo stays
	// promotion-free on purpose: it also dials relays, which must never enter
	// the application pools.
	promoteFriendlyPeer(ni, pid, connDirection(s), s.Conn().RemoteMultiaddr())
	// The write can wait as well: on the stream window, when the frame is larger than it
	// and the peer does not read, or behind a connection that stopped draining.
	if deadline, ok := ctx.Deadline(); ok {
		_ = s.SetWriteDeadline(deadline)
	}
	// One-shot stream: write, then close BOTH sides. CloseWrite is graceful (FIN
	// ordered after the data, delivery still guaranteed while the connection
	// lives); responses always arrive on a fresh inbound stream, so the read side
	// is closed too. NOTE: success below means "accepted by the local muxer", not
	// "delivered" — an abortive DisconnectFrom (stream Reset) issued right after
	// destroys the frame, which is why the Python teardown defers the disconnect.
	if err := writeSessionFrame(s, payload); err != nil {
		// Not inline: behind a connection that stopped draining the reset has to queue
		// like any other frame, and no deadline applies to it.
		go s.Reset()
		return fmt.Errorf("write to %s failed: %w", pid, err)
	}
	// The closing runs on its own: the caller does not wait for it. The frame needs neither
	// the FIN nor the close to be read on the other side, and both can wait on the peer. On
	// yamux the FIN queues like a frame, with no deadline. The stream was negotiated lazily,
	// and its Close first reads the peer's answer to the protocol proposal (go-multistream's
	// lazyClientConn), which a peer that stopped answering never gives: the read deadline
	// lets that read end, and the stream be given back, even then.
	go func() {
		if err := s.CloseWrite(); err != nil {
			logger.Warnf("[GO] ⚠️ Instance %d: SendOnProtocol CloseWrite to %s on %s failed: %v\n", ni.instanceIndex, pid, protocolID, err)
		}
		// Close, not only CloseRead: CloseWrite and CloseRead do not give the stream back to the
		// resource manager, only Close or Reset do. A stream that stays counted adds up, frame
		// after frame, until the limit of streams per protocol and peer (128 or so): no Session
		// frame can then be sent to that peer, and its Session is lost (about 20 minutes of
		// keep-alives). Close after CloseWrite does not reset: the frame still goes through.
		_ = s.SetReadDeadline(time.Now().Add(SessionSendTimeout))
		_ = s.Close()
	}()

	logger.Infof("[GO] 📤 Instance %d: Sent %d bytes on %s to %s\n", ni.instanceIndex, len(payload), protocolID, pid)
	return nil
}

// SignWithHostKey signs `data` with the host's libp2p private key (Ed25519 by default).
// Returns the signature base64-encoded in the success message. Used by Session to
// produce sig_b / sig_a over (nonce || session_id) — Stage-1/Stage-2 replay defence.
//
// Caller MUST FreeString the result.
//
//export SignWithHostKey
func SignWithHostKey(
	instanceIndexC C.int,
	dataC *C.char,
	dataLenC C.int,
) (ret *C.char) {
	defer recoverToCString(&ret, "SignWithHostKey")
	ni, err := getInstance(int(instanceIndexC))
	if err != nil {
		return C.CString(jsonErrorResponse(fmt.Sprintf("Invalid instance index: %d", int(instanceIndexC)), err))
	}
	if ni.host == nil {
		return C.CString(jsonErrorResponse("SignWithHostKey: host not initialised", nil))
	}
	priv := ni.host.Peerstore().PrivKey(ni.host.ID())
	if priv == nil {
		return C.CString(jsonErrorResponse("SignWithHostKey: host private key not in peerstore", nil))
	}
	data := C.GoBytes(unsafe.Pointer(dataC), dataLenC)
	sig, err := priv.Sign(data)
	if err != nil {
		return C.CString(jsonErrorResponse("SignWithHostKey: Sign failed", err))
	}
	return C.CString(jsonSuccessResponse(base64.StdEncoding.EncodeToString(sig)))
}

// VerifyPeerSignature verifies that `sig` is a valid signature of `data` under the
// public key derivable from peerID. Returns Success / Error. For Ed25519 keys
// (libp2p default) the public key is embedded in the peer-id ("identity" multihash);
// for RSA peer-ids this will fail because the pubkey is not recoverable from the id.
//
// Caller MUST FreeString the result.
//
//export VerifyPeerSignature
func VerifyPeerSignature(
	peerIDC *C.char,
	dataC *C.char, dataLenC C.int,
	sigB64C *C.char,
) (ret *C.char) {
	defer recoverToCString(&ret, "VerifyPeerSignature")
	pid, err := peer.Decode(C.GoString(peerIDC))
	if err != nil {
		return C.CString(jsonErrorResponse("VerifyPeerSignature: bad peerID", err))
	}
	pub, err := pid.ExtractPublicKey()
	if err != nil {
		return C.CString(jsonErrorResponse(fmt.Sprintf("VerifyPeerSignature: ExtractPublicKey(%s) failed", pid), err))
	}
	if pub == nil {
		return C.CString(jsonErrorResponse(fmt.Sprintf("VerifyPeerSignature: no public key recoverable from peerID %s", pid), nil))
	}
	sig, err := base64.StdEncoding.DecodeString(C.GoString(sigB64C))
	if err != nil {
		return C.CString(jsonErrorResponse("VerifyPeerSignature: bad signature base64", err))
	}
	data := C.GoBytes(unsafe.Pointer(dataC), dataLenC)
	ok, err := pub.Verify(data, sig)
	if err != nil {
		return C.CString(jsonErrorResponse("VerifyPeerSignature: Verify failed", err))
	}
	if !ok {
		return C.CString(jsonErrorResponse("VerifyPeerSignature: signature does not verify", nil))
	}
	return C.CString(jsonSuccessResponse("verified"))
}
