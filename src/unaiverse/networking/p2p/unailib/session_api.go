// FFI exports backing the Python Session abstraction.
//
// These exports — SendOnProtocol, GenerateSignedPeerRecord,
// VerifySignedPeerRecord (+ host-key signing/verification) — are what Python
// needs on top of the existing chat/pubsub surface to drive per-stage Session
// handshakes (see session/session.proto and session/session.py). Liveness is a
// session-level PING/PONG over the control protocol, not /ipfs/ping.
package main

/*
#include <stdlib.h>
*/
import "C"

import (
	"context"
	"fmt"
	"time"
	"unsafe"
	"encoding/base64"

	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/record"
	libp2pproto "github.com/libp2p/go-libp2p/core/protocol"
)

// sessionOpenStreamTimeout bounds how long we wait for a new outbound stream on
// a Session protocol. Sessions enforce their own per-stage timeouts (5s for HELLO,
// 15s for PROFILE, etc.); this is the safety net at the libp2p layer.
const sessionOpenStreamTimeout = 10 * time.Second

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

	ctx, cancel := context.WithTimeout(ni.ctx, sessionOpenStreamTimeout)
	defer cancel()
	s, err := ni.host.NewStream(
		network.WithAllowLimitedConn(ctx, protocolIDStr),
		pid,
		libp2pproto.ID(protocolIDStr),
	)
	if err != nil {
		return C.CString(jsonErrorResponse(fmt.Sprintf("SendOnProtocol: NewStream(%s, %s) failed", pid, protocolIDStr), err))
	}
	// The sending side of Session traffic must surface the peer to the pools
	// too (the receiving side does it in handleSessionStream). ConnectTo stays
	// promotion-free on purpose: it also dials relays, which must never enter
	// the application pools.
	promoteFriendlyPeer(ni, pid, connDirection(s), s.Conn().RemoteMultiaddr())
	// Best-effort: keep the read side open so a responder reply can come back on
	// THIS stream if the peer ever decides to write before we close — for now
	// responses arrive on a *fresh* inbound stream, so we just half-close write.
	if err := writeSessionFrame(s, payload); err != nil {
		_ = s.Reset()
		return C.CString(jsonErrorResponse(fmt.Sprintf("SendOnProtocol: write to %s failed", pid), err))
	}
	if err := s.CloseWrite(); err != nil {
		logger.Warnf("[GO] ⚠️ Instance %d: SendOnProtocol CloseWrite to %s on %s failed: %v\n", ni.instanceIndex, pid, protocolIDStr, err)
	}
	// CloseRead too — we don't expect inbound on this stream.
	_ = s.CloseRead()

	logger.Infof("[GO] 📤 Instance %d: Sent %d bytes on %s to %s\n", ni.instanceIndex, len(payload), protocolIDStr, pid)
	return C.CString(jsonSuccessResponse(fmt.Sprintf("Sent %d bytes on %s to %s", len(payload), protocolIDStr, pid)))
}

// GenerateSignedPeerRecord builds a libp2p PeerRecord over the host's currently
// known listen addresses and wraps it in a SignedEnvelope sealed with the host
// private key. The envelope binds the host's peer-id to its addresses with a
// monotonic-timestamp sequence number.
//
// Returns:
//
//	`{"state":"Success","message":"<base64-envelope>"}` on success.
//
// Caller MUST FreeString the result.
//
//export GenerateSignedPeerRecord
func GenerateSignedPeerRecord(
	instanceIndexC C.int,
) (ret *C.char) {
	defer recoverToCString(&ret, "GenerateSignedPeerRecord")
	ni, err := getInstance(int(instanceIndexC))
	if err != nil {
		return C.CString(jsonErrorResponse(fmt.Sprintf("Invalid instance index: %d", int(instanceIndexC)), err))
	}
	if ni.host == nil {
		return C.CString(jsonErrorResponse("GenerateSignedPeerRecord: host not initialised", nil))
	}

	info := peer.AddrInfo{ID: ni.host.ID(), Addrs: ni.host.Addrs()}
	rec := peer.PeerRecordFromAddrInfo(info)

	priv := ni.host.Peerstore().PrivKey(ni.host.ID())
	if priv == nil {
		return C.CString(jsonErrorResponse("GenerateSignedPeerRecord: host private key not in peerstore", nil))
	}
	env, err := record.Seal(rec, priv)
	if err != nil {
		return C.CString(jsonErrorResponse("GenerateSignedPeerRecord: Seal failed", err))
	}
	envBytes, err := env.Marshal()
	if err != nil {
		return C.CString(jsonErrorResponse("GenerateSignedPeerRecord: Marshal failed", err))
	}
	return C.CString(jsonSuccessResponse(base64.StdEncoding.EncodeToString(envBytes)))
}

// VerifySignedPeerRecord parses and verifies a base64-encoded SignedEnvelope, and
// asserts that the inner PeerRecord's PeerID matches expectedPeerID. This is how
// the Session responder confirms that a SignedPeerRecord received in PROFILE_OFFER
// — sealed under the joiner's OTHER peer-id (p2p_world key when the handshake runs
// on p2p_public) — actually belongs to that joiner.
//
// Returns:
//
//	`{"state":"Success","message":"verified"}` on success;
//	`{"state":"Error","message":"..."}` on signature / domain / id mismatch.
//
// Caller MUST FreeString the result.
//
//export VerifySignedPeerRecord
func VerifySignedPeerRecord(
	envelopeB64C *C.char,
	expectedPeerIDC *C.char,
) (ret *C.char) {
	defer recoverToCString(&ret, "VerifySignedPeerRecord")
	envB64 := C.GoString(envelopeB64C)
	expectedStr := C.GoString(expectedPeerIDC)

	expectedPID, err := peer.Decode(expectedStr)
	if err != nil {
		return C.CString(jsonErrorResponse(fmt.Sprintf("VerifySignedPeerRecord: bad expected peerID '%s'", expectedStr), err))
	}

	envBytes, err := base64.StdEncoding.DecodeString(envB64)
	if err != nil {
		return C.CString(jsonErrorResponse("VerifySignedPeerRecord: base64 decode failed", err))
	}

	// ConsumeEnvelope verifies the signature against the inner public key and
	// returns the inner record (typed); the domain is checked against the value
	// declared in the envelope.
	_, rec, err := record.ConsumeEnvelope(envBytes, peer.PeerRecordEnvelopeDomain)
	if err != nil {
		return C.CString(jsonErrorResponse("VerifySignedPeerRecord: ConsumeEnvelope failed", err))
	}
	prec, ok := rec.(*peer.PeerRecord)
	if !ok {
		return C.CString(jsonErrorResponse("VerifySignedPeerRecord: inner record is not a PeerRecord", nil))
	}
	if prec.PeerID != expectedPID {
		return C.CString(jsonErrorResponse(
			fmt.Sprintf("VerifySignedPeerRecord: PeerID mismatch, want %s got %s", expectedPID, prec.PeerID),
			nil,
		))
	}
	return C.CString(jsonSuccessResponse("verified"))
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
