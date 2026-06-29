"""Pure-FSM unit tests for ``Session``. No real network, no Go.

These tests stub the conn/p2p surface so we can drive Session through every
state transition and confirm the events / sends it emits. They protect against
regressions in:

  - Stage-1/2/3 happy-path transitions
  - Replay defence (sig_a / sig_b verification gating)
  - Timeout firing in every blocking state
  - Disconnect → grace → resume / lost
  - Intentional close (GOAWAY / GOAWAY_ACK)
  - Keep-alive ping success / failure threshold

The Session FSM is protobuf-free: it speaks ``SessMsg`` value objects and the
FakeConn records the SessMsg sent on each protocol (entry[2] is a SessMsg).
"""

import time
import asyncio
import json
import pytest

from unaiverse.networking.session import (
    Session, SessionState, SessionRole, SessionEventType, SessMsg,
    SESSION_HELLO_PROTOCOL, SESSION_PROFILE_PROTOCOL,
    SESSION_WORLD_PROTOCOL, SESSION_CONTROL_PROTOCOL,
)
from unaiverse.networking.session.sess_msg import (
    OK, VERSION_UNSUPPORTED, NOT_AUTHORIZED, GRACEFUL_SHUTDOWN, WORLD_DISENGAGEMENT,
    SESSION_UNKNOWN, TOKEN_EXPIRED,
)
from unaiverse.networking.node.connpool import NodeConn

pytestmark = pytest.mark.unit


# ── Helpers ─────────────────────────────────────────────────────────────────

def _last_sent(conn, protocol_id):
    """Return the last (peer, proto, SessMsg) sent on protocol_id from the
    FakeConn, or None. Session sends only through ``conn.send_session(...)``."""
    for entry in reversed(conn.sent):
        if entry[1] == protocol_id:
            return entry
    return None


def _run(coro):
    return asyncio.run(coro)


def _mk(peer_id, role, p2p, local_peer_id, local_node_id="local-node",
        local_profile_factory=lambda: {}, addresses=None,
        token_verifier=None, token_factory=None, remote_node_id_hint=None):
    return Session(
        peer_id=peer_id, role=role, p2p=p2p,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=local_peer_id,
        local_node_id=local_node_id, local_profile_factory=local_profile_factory,
        addresses=addresses,
        token_verifier=token_verifier, token_factory=token_factory,
        remote_node_id_hint=remote_node_id_hint,
    )


# ── Initiator happy path ────────────────────────────────────────────────────

def test_initiator_start_sends_hello(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """start() must dial, then emit HelloRequest on the hello protocol and
    park in HELLO_SENT awaiting HELLO_ACK."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               local_profile_factory=lambda: {"x": 1},
               addresses=["/ip4/127.0.0.1/tcp/0/p2p/" + bob_peer_id])
    events = _run(sess.start())
    assert events == []
    assert sess.state == SessionState.HELLO_SENT
    assert alice_conn.connect_calls == [["/ip4/127.0.0.1/tcp/0/p2p/" + bob_peer_id]]
    last = _last_sent(alice_conn, SESSION_HELLO_PROTOCOL)
    assert last is not None, "HelloRequest must be sent on the hello protocol"
    hello = last[2]
    assert hello.kind == SessMsg.KIND_HELLO_REQ
    assert hello.fields["session_id"] == sess.session_id
    assert len(hello.fields["nonce_a"]) == 32
    assert "1.0" in list(hello.fields["protocol_versions"])


def test_initiator_dial_failure_fails_after_max_attempts(alice_conn, alice_peer_id, bob_peer_id, monkeypatch):
    """All 4 dial attempts return None → terminal FAILED event."""
    alice_conn.connect_returns = (None, False)
    async def _no_sleep(_s):
        return None
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               addresses=["/ip4/1.2.3.4/tcp/0"])
    events = _run(sess.start())
    assert len(events) == 1 and events[0].type == SessionEventType.FAILED
    assert events[0].data["reason"] == "dial_failed"
    assert sess.state == SessionState.FAILED


# ── Responder happy path ────────────────────────────────────────────────────

def test_responder_on_hello_sends_hello_ack(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """Inbound HelloRequest in INIT → HELLO_ACK_SENT, HelloResponse on wire."""
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id, local_node_id="bob-node")
    hello = SessMsg(SessMsg.KIND_HELLO_REQ, {
        "session_id": b"\x01" * 16, "nonce_a": b"\xaa" * 32,
        "protocol_versions": ["1.0"], "capabilities": 0, "client_id": "test/1.0",
    })
    events = _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, hello))
    assert events == []
    assert sess.state == SessionState.HELLO_ACK_SENT
    assert sess.session_id == b"\x01" * 16
    assert sess.nonce_a == b"\xaa" * 32
    last = _last_sent(bob_conn, SESSION_HELLO_PROTOCOL)
    assert last is not None
    resp = last[2]
    assert resp.kind == SessMsg.KIND_HELLO_RESP
    assert resp.fields["reason_code"] == OK
    assert resp.fields["chosen_version"] == "1.0"
    assert resp.fields["sig_b"] == b"sig-bytes"   # what our fake signer returned


def test_responder_rejects_unsupported_version(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """HelloRequest with no overlap on protocol_versions → HelloResponse(VERSION_UNSUPPORTED) and FAILED."""
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id, local_node_id="bob-node")
    hello = SessMsg(SessMsg.KIND_HELLO_REQ, {
        "session_id": b"\x01" * 16, "nonce_a": b"\xaa" * 32,
        "protocol_versions": ["99.0"], "client_id": "test/99",
    })
    events = _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, hello))
    assert len(events) == 1 and events[0].type == SessionEventType.FAILED
    assert events[0].data["reason"] == "version_unsupported"
    last = _last_sent(bob_conn, SESSION_HELLO_PROTOCOL)
    assert last is not None
    assert last[2].fields["reason_code"] == VERSION_UNSUPPORTED


# ── Stage-1 replay defence (sig_b verification) ─────────────────────────────

def test_initiator_rejects_bad_sig_b(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """If the responder's sig_b doesn't verify, the initiator MUST fail before
    sending PROFILE_OFFER. This is the replay defence."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               addresses=["/ip4/127.0.0.1/tcp/0"])
    _run(sess.start())
    assert sess.state == SessionState.HELLO_SENT
    alice_p2p.verify_peer_sig_returns = False
    resp = SessMsg(SessMsg.KIND_HELLO_RESP, {
        "chosen_version": "1.0", "capabilities": 0, "nonce_b": b"\xbb" * 32,
        "sig_b": b"bogus", "reason_code": OK,
    })
    events = _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, resp))
    assert len(events) == 1 and events[0].type == SessionEventType.FAILED
    assert events[0].data["reason"] == "hello_ack_bad_sig_b"
    assert _last_sent(alice_conn, SESSION_PROFILE_PROTOCOL) is None


# ── Stage-2 + full initiator happy path ─────────────────────────────────────

def test_initiator_completes_through_profile_decision(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """Drive the initiator from start through ESTABLISHED."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               local_profile_factory=lambda: {"node_id": "alice-node", "k": "v"},
               addresses=["/ip4/127.0.0.1/tcp/0"])
    _run(sess.start())
    ack = SessMsg(SessMsg.KIND_HELLO_RESP, {
        "chosen_version": "1.0", "nonce_b": b"\xbb" * 32, "sig_b": b"sig", "reason_code": OK,
    })
    _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, ack))
    assert sess.state == SessionState.PROFILE_SENT
    last = _last_sent(alice_conn, SESSION_PROFILE_PROTOCOL)
    assert last is not None
    offer = last[2]
    assert offer.kind == SessMsg.KIND_PROFILE_OFFER
    assert offer.fields["sig_a"] == b"sig-bytes"
    # The offer no longer carries a signed_peer_record (binding via p2p_peers).
    assert "signed_peer_record" not in offer.fields

    decision = SessMsg(SessMsg.KIND_PROFILE_DECISION, {
        "accepted": True, "profile_b": json.dumps({"node_id": "bob-node"}).encode(),
        "reason_code": OK, "session_id": sess.session_id,
    })
    events = _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, decision))
    assert len(events) == 1 and events[0].type == SessionEventType.ESTABLISHED
    assert sess.state == SessionState.ESTABLISHED
    assert sess.remote_profile == {"node_id": "bob-node"}


def test_initiator_rejected_profile_fails(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """ProfileDecision(accepted=false) → FAILED with the responder's reason_code."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               addresses=["/ip4/127.0.0.1/tcp/0"])
    _run(sess.start())
    _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, SessMsg(SessMsg.KIND_HELLO_RESP, {
        "chosen_version": "1.0", "nonce_b": b"\xbb" * 32, "sig_b": b"sig", "reason_code": OK,
    })))
    decision = SessMsg(SessMsg.KIND_PROFILE_DECISION, {"accepted": False, "reason_code": NOT_AUTHORIZED})
    events = _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, decision))
    assert len(events) == 1 and events[0].type == SessionEventType.FAILED
    assert "NOT_AUTHORIZED" in events[0].data["reason"]


# ── Stage-2 responder: bad sig_a ────────────────────────────────────────────

def test_responder_rejects_bad_sig_a(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """ProfileOffer with sig_a that fails verification → reject + FAILED."""
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id, local_node_id="bob-node")
    _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, SessMsg(SessMsg.KIND_HELLO_REQ, {
        "session_id": b"\x01" * 16, "nonce_a": b"\xaa" * 32, "protocol_versions": ["1.0"],
    })))
    assert sess.state == SessionState.HELLO_ACK_SENT
    bob_p2p.verify_peer_sig_returns = False
    offer = SessMsg(SessMsg.KIND_PROFILE_OFFER, {
        "profile": json.dumps({}).encode(), "sig_a": b"bad",
        "session_id": sess.session_id,
    })
    events = _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, offer))
    assert len(events) == 1 and events[0].type == SessionEventType.FAILED
    last = _last_sent(bob_conn, SESSION_PROFILE_PROTOCOL)
    assert last is not None
    decision = last[2]
    assert decision.kind == SessMsg.KIND_PROFILE_DECISION
    assert decision.fields["accepted"] is False
    assert decision.fields["reason_code"] == NOT_AUTHORIZED


# ── Stage-2 responder: RT token consumption ─────────────────────────────────

def _drive_responder_to_offer_wait(sess):
    """Push a HelloRequest so the responder parks in HELLO_ACK_SENT, ready for an offer."""
    _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, SessMsg(SessMsg.KIND_HELLO_REQ, {
        "session_id": b"\x01" * 16, "nonce_a": b"\xaa" * 32, "protocol_versions": ["1.0"],
    })))
    assert sess.state == SessionState.HELLO_ACK_SENT


def test_initiator_offer_carries_token(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """With a token_factory injected, the PROFILE_OFFER carries the node token."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               addresses=["/ip4/127.0.0.1/tcp/0"], token_factory=lambda: "tok-abc")
    _run(sess.start())
    _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, SessMsg(SessMsg.KIND_HELLO_RESP, {
        "chosen_version": "1.0", "nonce_b": b"\xbb" * 32, "sig_b": b"sig", "reason_code": OK,
    })))
    offer = _last_sent(alice_conn, SESSION_PROFILE_PROTOCOL)[2]
    assert offer.kind == SessMsg.KIND_PROFILE_OFFER
    assert offer.fields["token"] == "tok-abc"


def test_initiator_offer_token_empty_without_factory(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """No token_factory → the offer carries an empty token (the v1 / no-RT path)."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               addresses=["/ip4/127.0.0.1/tcp/0"])
    _run(sess.start())
    _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, SessMsg(SessMsg.KIND_HELLO_RESP, {
        "chosen_version": "1.0", "nonce_b": b"\xbb" * 32, "sig_b": b"sig", "reason_code": OK,
    })))
    assert _last_sent(alice_conn, SESSION_PROFILE_PROTOCOL)[2].fields["token"] == ""


def test_responder_rejects_invalid_token(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """A configured verifier that rejects the token → reject(NOT_AUTHORIZED) + FAILED,
    before the application decision is ever reached."""
    async def _verifier(token, peer_id):
        return (None, None, (), 0)   # verification failure
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id,
               local_node_id="bob-node", token_verifier=_verifier)
    _drive_responder_to_offer_wait(sess)
    offer = SessMsg(SessMsg.KIND_PROFILE_OFFER, {
        "profile": json.dumps({"node_id": "claimed"}).encode(), "sig_a": b"ok",
        "session_id": sess.session_id, "token": "bad-tok",
    })
    events = _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, offer))
    assert len(events) == 1 and events[0].type == SessionEventType.FAILED
    assert "bad_token" in events[0].data["reason"]
    decision = _last_sent(bob_conn, SESSION_PROFILE_PROTOCOL)[2]
    assert decision.fields["accepted"] is False
    assert decision.fields["reason_code"] == NOT_AUTHORIZED


def test_responder_valid_token_overrides_profile_node_id(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """A valid token is verified against the transport peer-id, and the root-attested
    node_id wins over the self-asserted profile node_id; the handshake then reaches
    ESTABLISHED (agent host, no decision hook)."""
    seen = {}
    async def _verifier(token, peer_id):
        seen["token"], seen["peer_id"] = token, peer_id
        return ("root-attested-node", "cv-hash", (peer_id,), 9999999999)
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id,
               local_node_id="bob-node", token_verifier=_verifier)
    _drive_responder_to_offer_wait(sess)
    offer = SessMsg(SessMsg.KIND_PROFILE_OFFER, {
        "profile": json.dumps({"node_id": "self-claimed"}).encode(), "sig_a": b"ok",
        "session_id": sess.session_id, "token": "good-tok",
    })
    events = _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, offer))
    assert seen == {"token": "good-tok", "peer_id": alice_peer_id}
    assert sess.remote_node_id_hint == "root-attested-node"
    # The peer's attested token expiry / cv_hash are captured on the Session.
    assert sess.remote_token_exp == 9999999999
    assert sess.remote_cv_hash == "cv-hash"
    assert len(events) == 1 and events[0].type == SessionEventType.ESTABLISHED


# ── Stage-2: RT mutual auth (responder token verified by the initiator) ──────

def test_responder_decision_carries_token(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """RT mutual: the responder's PROFILE_DECISION carries its own token."""
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id,
               local_node_id="bob-node", token_factory=lambda: "bob-tok")
    _drive_responder_to_offer_wait(sess)
    offer = SessMsg(SessMsg.KIND_PROFILE_OFFER, {
        "profile": json.dumps({"node_id": "alice"}).encode(), "sig_a": b"ok",
        "session_id": sess.session_id, "token": "",
    })
    events = _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, offer))
    assert len(events) == 1 and events[0].type == SessionEventType.ESTABLISHED
    dec = _last_sent(bob_conn, SESSION_PROFILE_PROTOCOL)[2]
    assert dec.kind == SessMsg.KIND_PROFILE_DECISION
    assert dec.fields["token"] == "bob-tok"


def _drive_initiator_to_decision(sess, p2p):
    """Start the initiator and feed a valid HELLO_ACK so it parks in PROFILE_SENT."""
    _run(sess.start())
    _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, SessMsg(SessMsg.KIND_HELLO_RESP, {
        "chosen_version": "1.0", "nonce_b": b"\xbb" * 32, "sig_b": b"sig", "reason_code": OK,
    })))


def test_initiator_verifies_responder_token(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """RT mutual: the initiator verifies the responder's token from PROFILE_DECISION
    against the responder's peer-id, adopts the attested node-id, and completes."""
    seen = {}
    async def _verifier(token, peer_id):
        seen["token"], seen["peer_id"] = token, peer_id
        return ("bob-attested", "cv", (peer_id,), 9999999999)
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               addresses=["/ip4/127.0.0.1/tcp/0"], token_verifier=_verifier)
    _drive_initiator_to_decision(sess, alice_p2p)
    decision = SessMsg(SessMsg.KIND_PROFILE_DECISION, {
        "accepted": True, "profile_b": json.dumps({"node_id": "bob"}).encode(),
        "reason_code": OK, "session_id": sess.session_id, "token": "bob-tok",
    })
    events = _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, decision))
    assert seen == {"token": "bob-tok", "peer_id": bob_peer_id}
    assert sess.remote_node_id_hint == "bob-attested"
    assert len(events) == 1 and events[0].type == SessionEventType.ESTABLISHED


def test_initiator_rejects_bad_responder_token(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """RT mutual: an invalid responder token fails the initiator's handshake."""
    async def _verifier(token, peer_id):
        return (None, None, (), 0)
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               addresses=["/ip4/127.0.0.1/tcp/0"], token_verifier=_verifier)
    _drive_initiator_to_decision(sess, alice_p2p)
    decision = SessMsg(SessMsg.KIND_PROFILE_DECISION, {
        "accepted": True, "profile_b": json.dumps({"node_id": "bob"}).encode(),
        "reason_code": OK, "session_id": sess.session_id, "token": "bad",
    })
    events = _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, decision))
    assert len(events) == 1 and events[0].type == SessionEventType.FAILED
    assert "bad_token" in events[0].data["reason"]


def test_initiator_rejects_responder_node_id_mismatch(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """RT mutual: when the initiator expected a specific node-id at this peer and the
    responder's token attests a different one, the handshake fails."""
    async def _verifier(token, peer_id):
        return ("someone-else", "cv", (peer_id,), 9999999999)
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               addresses=["/ip4/127.0.0.1/tcp/0"], token_verifier=_verifier,
               remote_node_id_hint="expected-bob")
    _drive_initiator_to_decision(sess, alice_p2p)
    decision = SessMsg(SessMsg.KIND_PROFILE_DECISION, {
        "accepted": True, "profile_b": json.dumps({"node_id": "bob"}).encode(),
        "reason_code": OK, "session_id": sess.session_id, "token": "tok",
    })
    events = _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, decision))
    assert len(events) == 1 and events[0].type == SessionEventType.FAILED
    assert "token_node_id_mismatch" in events[0].data["reason"]


# ── Timeouts ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("state,timeout_field", [
    (SessionState.CONNECTING,     "CONNECT"),
    (SessionState.HELLO_SENT,     "HELLO"),
    (SessionState.PROFILE_SENT,   "PROFILE"),
    (SessionState.HELLO_ACK_SENT, "PROFILE"),
])
def test_check_timeout_fires_failed_for_handshake_states(alice_conn, alice_peer_id, bob_peer_id, state, timeout_field):
    """Every blocking handshake state must FAIL on timeout."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(state, 0.001)
    time.sleep(0.005)
    ev = sess.check_timeout()
    assert ev is not None
    assert ev.type == SessionEventType.FAILED
    assert sess.state == SessionState.FAILED


def test_check_timeout_returns_none_before_expiry(alice_conn, alice_peer_id, bob_peer_id):
    """check_timeout() must not fire while the state deadline is still in the future."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(SessionState.HELLO_SENT, 10.0)
    assert sess.check_timeout() is None


# ── Disconnect / grace / resume ─────────────────────────────────────────────

def test_disconnect_from_established_enters_reconnecting(alice_conn, alice_peer_id, bob_peer_id):
    """A transport disconnect from ESTABLISHED emits RECONNECTING and moves to RECONNECTING."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(SessionState.ESTABLISHED, None)
    ev = sess.on_transport_disconnected()
    assert ev is not None and ev.type == SessionEventType.RECONNECTING
    assert sess.state == SessionState.RECONNECTING


def test_transport_reconnect_does_not_blindly_resume(alice_conn, alice_peer_id, bob_peer_id):
    """A transport-diff reconnect must NOT flip straight to ESTABLISHED (that is the
    /ipfs/ping false-resume bug). It only arms the RESUME handshake; the session
    stays RECONNECTING until a valid RESUME_ACK arrives."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(SessionState.RECONNECTING, 10.0)
    ev = sess.on_transport_reconnected()
    assert ev is None
    assert sess.state == SessionState.RECONNECTING
    assert sess._reconnect_substep == "redial"


def test_grace_expiry_emits_lost(alice_conn, alice_peer_id, bob_peer_id):
    """The grace timer expiring while RECONNECTING emits LOST and lands in FAILED."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(SessionState.RECONNECTING, 0.001)
    time.sleep(0.005)
    ev = sess.check_timeout()
    assert ev is not None and ev.type == SessionEventType.LOST
    assert sess.state == SessionState.FAILED


# ── Intentional close ───────────────────────────────────────────────────────

def test_close_sends_goaway_and_enters_closing(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """close() from ESTABLISHED sends a GOAWAY on the control protocol carrying the given
    reason and moves the session to CLOSING."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(SessionState.ESTABLISHED, None)
    _run(sess.close(WORLD_DISENGAGEMENT))
    assert sess.state == SessionState.CLOSING
    last = _last_sent(alice_conn, SESSION_CONTROL_PROTOCOL)
    assert last is not None
    ga = last[2]
    assert ga.kind == SessMsg.KIND_GOAWAY
    assert ga.fields["reason"] == WORLD_DISENGAGEMENT


def test_goaway_ack_finalizes_close(alice_conn, alice_peer_id, bob_peer_id):
    """Receiving a GOAWAY_ACK while CLOSING finalizes the close: CLOSED event, state CLOSED."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(SessionState.CLOSING, 10.0)
    ack = SessMsg(SessMsg.KIND_GOAWAY_ACK, {"received": True})
    events = _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, ack))
    assert len(events) == 1 and events[0].type == SessionEventType.CLOSED
    assert sess.state == SessionState.CLOSED


def test_remote_goaway_sends_ack_and_closes(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """Receiving a GOAWAY → send GOAWAY_ACK, emit CLOSED."""
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id, local_node_id="bob-node")
    sess._set_state(SessionState.ESTABLISHED, None)
    sess.session_id = b"\x07" * 16
    ga = SessMsg(SessMsg.KIND_GOAWAY, {"reason": GRACEFUL_SHUTDOWN, "session_id": sess.session_id})
    events = _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, ga))
    assert len(events) == 1 and events[0].type == SessionEventType.CLOSED
    last = _last_sent(bob_conn, SESSION_CONTROL_PROTOCOL)
    assert last is not None
    assert last[2].kind == SessMsg.KIND_GOAWAY_ACK
    assert last[2].fields["received"] is True


# ── Session-level keep-alive (PING/PONG over the control protocol) ───────────

def test_tick_keepalive_sends_session_ping_when_due(alice_conn, alice_peer_id, bob_peer_id):
    """When ESTABLISHED and the keep-alive period has elapsed, tick must send a
    session-level PING on the CONTROL protocol (NOT /ipfs/ping) bound to session_id."""
    async def _runner():
        sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
        sess._set_state(SessionState.ESTABLISHED, None)
        sess.session_id = b"\x01" * 16
        sess._last_ping_at = time.monotonic() - 30.0   # well past 15s threshold
        ev = await sess.tick_keepalive()
        assert ev is None
        assert sess._ping_in_flight is True
        # The PING is sent inline by the tick: no extra loop yield needed.
        return sess
    sess = asyncio.run(_runner())
    last = _last_sent(alice_conn, SESSION_CONTROL_PROTOCOL)
    assert last is not None and last[2].kind == SessMsg.KIND_PING
    assert last[2].fields["session_id"] == sess.session_id
    # No /ipfs/ping was used.
    assert alice_conn.p2p_name_to_p2p[NodeConn.P2P_PUBLIC].pinged == []


def test_pong_ok_clears_failures(alice_conn, alice_peer_id, bob_peer_id):
    """A matching PONG(ok=true) clears the in-flight ping and resets the consecutive-failure
    counter to zero, leaving the session ESTABLISHED with no event."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(SessionState.ESTABLISHED, None)
    sess.session_id = b"\x02" * 16
    sess._ping_in_flight = True
    sess._ping_nonce = b"nonce-123"
    sess._consecutive_ping_failures = 1
    pong = SessMsg(SessMsg.KIND_PONG, {"session_id": sess.session_id, "nonce": b"nonce-123", "ok": True})
    events = _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, pong))
    assert events == []
    assert sess._ping_in_flight is False
    assert sess._consecutive_ping_failures == 0
    assert sess.state == SessionState.ESTABLISHED


def test_pong_not_ok_starts_reconnect(alice_conn, alice_peer_id, bob_peer_id):
    """A PONG ok=false (peer holds no Session) is a hard liveness failure → RECONNECTING."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(SessionState.ESTABLISHED, None)
    sess.session_id = b"\x02" * 16
    sess._ping_in_flight = True
    sess._ping_nonce = b"nonce-123"
    pong = SessMsg(SessMsg.KIND_PONG, {"session_id": sess.session_id, "nonce": b"nonce-123", "ok": False})
    events = _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, pong))
    assert len(events) == 1 and events[0].type == SessionEventType.RECONNECTING
    assert sess.state == SessionState.RECONNECTING


def test_keepalive_two_misses_trigger_reconnecting(alice_conn, alice_peer_id, bob_peer_id):
    """Two consecutive unanswered pings while ESTABLISHED → RECONNECTING."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(SessionState.ESTABLISHED, None)
    now = time.monotonic()
    # First miss: a ping is in flight whose RTT deadline has passed.
    sess._ping_in_flight = True
    sess._ping_deadline = now - 1.0
    sess._last_ping_at = now            # not due to send a fresh one this tick
    ev = _run(sess.tick_keepalive(now))
    assert ev is None and sess._consecutive_ping_failures == 1
    # Second miss.
    sess._ping_in_flight = True
    sess._ping_deadline = now - 1.0
    ev = _run(sess.tick_keepalive(now))
    assert ev is not None and ev.type == SessionEventType.RECONNECTING
    assert sess.state == SessionState.RECONNECTING


def test_on_ping_answers_pong_ok_when_session_matches(bob_conn, bob_peer_id, alice_peer_id):
    """An inbound PING whose session_id matches the held session is answered with a
    PONG(ok=true) echoing the ping nonce, with no event emitted."""
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id)
    sess._set_state(SessionState.ESTABLISHED, None)
    sess.session_id = b"\x07" * 16
    ping = SessMsg(SessMsg.KIND_PING, {"session_id": sess.session_id, "nonce": b"abc"})
    events = _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, ping))
    assert events == []
    last = _last_sent(bob_conn, SESSION_CONTROL_PROTOCOL)
    assert last is not None and last[2].kind == SessMsg.KIND_PONG
    assert last[2].fields["ok"] is True and last[2].fields["nonce"] == b"abc"


# ── RESUME (fast-path recovery) ──────────────────────────────────────────────

def test_resume_inbound_while_reconnecting_resumes(bob_conn, bob_peer_id, alice_peer_id):
    """Peer RESUME for a session we still hold (and we are RECONNECTING) → ACK + RESUMED."""
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id)
    sess._set_state(SessionState.RECONNECTING, 10.0)
    sess.session_id = b"\x08" * 16
    resume = SessMsg(SessMsg.KIND_RESUME, {"session_id": sess.session_id, "nonce": b"r1"})
    events = _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, resume))
    assert len(events) == 1 and events[0].type == SessionEventType.RESUMED
    assert sess.state == SessionState.ESTABLISHED
    last = _last_sent(bob_conn, SESSION_CONTROL_PROTOCOL)
    assert last[2].kind == SessMsg.KIND_RESUME_ACK and last[2].fields["accepted"] is True


def test_resume_inbound_unknown_session_naks(bob_conn, bob_peer_id, alice_peer_id):
    """RESUME for a session_id we don't hold → NACK with SESSION_UNKNOWN."""
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id)
    sess._set_state(SessionState.ESTABLISHED, None)
    sess.session_id = b"\x08" * 16
    resume = SessMsg(SessMsg.KIND_RESUME, {"session_id": b"\x99" * 16, "nonce": b"r1"})
    events = _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, resume))
    assert events == []
    last = _last_sent(bob_conn, SESSION_CONTROL_PROTOCOL)
    assert last[2].kind == SessMsg.KIND_RESUME_ACK
    assert last[2].fields["accepted"] is False
    assert last[2].fields["reason_code"] == SESSION_UNKNOWN


def test_resume_ack_accepted_resumes(alice_conn, alice_peer_id, bob_peer_id):
    """An accepted RESUME_ACK with the matching nonce resumes a RECONNECTING initiator
    straight back to ESTABLISHED via a RESUMED event."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(SessionState.RECONNECTING, 10.0)
    sess._reconnect_substep = "resume_sent"
    sess._resume_nonce = b"rn"
    sess.session_id = b"\x08" * 16
    ack = SessMsg(SessMsg.KIND_RESUME_ACK, {"session_id": sess.session_id, "nonce": b"rn",
                                            "accepted": True, "reason_code": OK})
    events = _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, ack))
    assert len(events) == 1 and events[0].type == SessionEventType.RESUMED
    assert sess.state == SessionState.ESTABLISHED


def test_resume_ack_nack_replays_handshake_for_plain_session(alice_conn, alice_peer_id, bob_peer_id):
    """A NACK on a plain initiator session re-runs the full handshake (re-HELLO),
    preserving peer-id."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(SessionState.RECONNECTING, 10.0)
    sess._reconnect_substep = "resume_sent"
    sess._resume_nonce = b"rn"
    old_sid = sess.session_id
    ack = SessMsg(SessMsg.KIND_RESUME_ACK, {"session_id": b"\x00" * 16, "nonce": b"rn",
                                            "accepted": False, "reason_code": SESSION_UNKNOWN})
    events = _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, ack))
    assert events == []
    assert sess.state == SessionState.HELLO_SENT
    assert sess.session_id != old_sid     # identity rotated for the fresh handshake
    last = _last_sent(alice_conn, SESSION_HELLO_PROTOCOL)
    assert last is not None and last[2].kind == SessMsg.KIND_HELLO_REQ


def test_resume_ack_nack_world_session_is_lost(alice_conn, alice_peer_id, bob_peer_id):
    """A world-join session cannot transparently re-handshake (double-join risk) → LOST."""
    sess = Session(
        peer_id=bob_peer_id, role=SessionRole.INITIATOR, p2p=alice_conn,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=alice_peer_id,
        local_node_id="ln", local_profile_factory=lambda: {}, is_world_join=True,
    )
    sess._set_state(SessionState.RECONNECTING, 10.0)
    sess._reconnect_substep = "resume_sent"
    sess._resume_nonce = b"rn"
    ack = SessMsg(SessMsg.KIND_RESUME_ACK, {"session_id": b"\x00" * 16, "nonce": b"rn",
                                            "accepted": False, "reason_code": SESSION_UNKNOWN})
    events = _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, ack))
    assert len(events) == 1 and events[0].type == SessionEventType.LOST
    assert sess.state == SessionState.FAILED


def test_tick_reconnect_responder_does_not_redial(bob_conn, bob_peer_id, alice_peer_id):
    """A responder (no addresses) cannot re-dial; it rides the grace timer to LOST."""
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id)
    sess._set_state(SessionState.RECONNECTING, 10.0)
    assert _run(sess.tick_reconnect()) is None
    assert bob_conn.connect_calls == []


def test_tick_reconnect_initiator_redials_and_sends_resume(alice_conn, alice_peer_id, bob_peer_id):
    """An initiator with addresses re-dials and, on success, sends RESUME."""
    async def _runner():
        sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
                   addresses=["/ip4/127.0.0.1/tcp/0/p2p/" + bob_peer_id])
        alice_conn.connect_returns = (bob_peer_id, False)
        sess._set_state(SessionState.RECONNECTING, 45.0)
        sess._reconnect_substep = "redial"
        sess._next_redial_at = 0.0
        # The re-dial + RESUME runs inline within the tick: no loop yield needed.
        await sess.tick_reconnect()
        return sess
    sess = asyncio.run(_runner())
    assert alice_conn.connect_calls          # a re-dial happened
    assert sess._reconnect_substep == "resume_sent"
    last = _last_sent(alice_conn, SESSION_CONTROL_PROTOCOL)
    assert last is not None and last[2].kind == SessMsg.KIND_RESUME


# ── Out-of-order / duplicate messages ───────────────────────────────────────

def test_profile_offer_in_hello_sent_is_ignored(alice_conn, alice_peer_id, bob_peer_id):
    """A PROFILE_OFFER arriving while the initiator still waits for HELLO_ACK
    (replay or broken peer) must be dropped: no state change, no decision sent."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               addresses=["/ip4/127.0.0.1/tcp/0"])
    _run(sess.start())
    assert sess.state == SessionState.HELLO_SENT
    offer = SessMsg(SessMsg.KIND_PROFILE_OFFER, {
        "profile": json.dumps({}).encode(), "sig_a": b"sig",
        "session_id": sess.session_id,
    })
    events = _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, offer))
    assert events == []
    assert sess.state == SessionState.HELLO_SENT
    assert _last_sent(alice_conn, SESSION_PROFILE_PROTOCOL) is None


def test_duplicate_hello_req_ignored_after_hello_ack_sent(bob_conn, bob_peer_id, alice_peer_id):
    """A retransmitted HELLO_REQ on a responder already past INIT must not
    restart the handshake: exactly one HELLO_RESP total, same session_id."""
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id)
    hello = SessMsg(SessMsg.KIND_HELLO_REQ, {
        "session_id": b"\x01" * 16, "nonce_a": b"\xaa" * 32,
        "protocol_versions": ["1.0"], "capabilities": 0, "client_id": "test/1.0",
    })
    _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, hello))
    assert sess.state == SessionState.HELLO_ACK_SENT
    events = _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, hello))
    assert events == []
    assert sess.state == SessionState.HELLO_ACK_SENT
    assert sess.session_id == b"\x01" * 16
    hello_resps = [e for e in bob_conn.sent
                   if e[1] == SESSION_HELLO_PROTOCOL and e[2].kind == SessMsg.KIND_HELLO_RESP]
    assert len(hello_resps) == 1


def test_profile_decision_in_established_is_ignored(alice_conn, alice_peer_id, bob_peer_id):
    """A duplicate PROFILE_DECISION after ESTABLISHED must not emit a second
    ESTABLISHED event nor overwrite the stored remote profile."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(SessionState.ESTABLISHED, None)
    sess.remote_profile = {"node_id": "bob-node"}
    dup = SessMsg(SessMsg.KIND_PROFILE_DECISION, {
        "accepted": True, "profile_b": json.dumps({"node_id": "evil"}).encode(),
        "reason_code": OK, "session_id": sess.session_id,
    })
    events = _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, dup))
    assert events == []
    assert sess.state == SessionState.ESTABLISHED
    assert sess.remote_profile == {"node_id": "bob-node"}


def test_world_grant_to_plain_initiator_is_ignored(alice_conn, alice_peer_id, bob_peer_id):
    """A WORLD_GRANT sent to an initiator that did not ask for a world join
    (is_world_join=False) must be dropped without state change."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               addresses=["/ip4/127.0.0.1/tcp/0"])
    _run(sess.start())
    _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, SessMsg(SessMsg.KIND_HELLO_RESP, {
        "chosen_version": "1.0", "nonce_b": b"\xbb" * 32, "sig_b": b"sig", "reason_code": OK,
    })))
    assert sess.state == SessionState.PROFILE_SENT
    grant = SessMsg(SessMsg.KIND_WORLD_GRANT, {
        "grant_id": b"\x05" * 32,
        "expiry_unix_ms": int(time.time() * 1000) + 30000,
        "world_profile": b"{}",
        "role": 1, "code_bundle": b"", "initial_stats": b"", "rendezvous_tag": 0,
    })
    events = _run(sess.process_inbound(SESSION_WORLD_PROTOCOL, grant))
    assert events == []
    assert sess.state == SessionState.PROFILE_SENT


# ── Silent abort (tie-breaker support) ──────────────────────────────────────

def test_abort_silently_fails_without_event(alice_conn, alice_peer_id, bob_peer_id):
    """abort_silently() drops the session to FAILED without emitting any event, as the
    losing side of a simultaneous-open tie-break needs."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(SessionState.HELLO_SENT, 5.0)
    sess.abort_silently("simultaneous_open_yielded")
    assert sess.state == SessionState.FAILED


def test_world_grant_arriving_before_decision_is_buffered(alice_conn, alice_peer_id, bob_peer_id):
    """WORLD and PROFILE ride different libp2p protocols, so the host's
    back-to-back DECISION+GRANT can be popped grant-first. The early grant must
    be buffered and consumed when the decision lands (it used to be dropped,
    stranding the joiner in awaiting_world_grant until timeout)."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = Session(
        peer_id=bob_peer_id, role=SessionRole.INITIATOR, p2p=alice_conn,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=alice_peer_id,
        local_node_id="local-node", local_profile_factory=lambda: {},
        addresses=["/ip4/127.0.0.1/tcp/0"], is_world_join=True,
    )
    _run(sess.start())
    _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, SessMsg(SessMsg.KIND_HELLO_RESP, {
        "chosen_version": "1.0", "nonce_b": b"\xbb" * 32, "sig_b": b"sig", "reason_code": OK,
    })))
    assert sess.state == SessionState.PROFILE_SENT

    world_profile = json.dumps({"dynamic": {"private_peer_id": "12D3WorldPriv"}}).encode()
    grant = SessMsg(SessMsg.KIND_WORLD_GRANT, {
        "grant_id": b"\x05" * 32,
        "expiry_unix_ms": int(time.time() * 1000) + 30000,
        "world_profile": world_profile,
        "role": 3, "code_bundle": b"", "initial_stats": b"", "rendezvous_tag": 0,
    })
    events = _run(sess.process_inbound(SESSION_WORLD_PROTOCOL, grant))
    assert events == []
    assert sess.state == SessionState.PROFILE_SENT  # buffered, not applied yet

    events = _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, SessMsg(SessMsg.KIND_PROFILE_DECISION, {
        "accepted": True, "profile_b": b"{}", "reason_code": OK, "session_id": sess.session_id,
    })))
    assert sess.state == SessionState.ESTABLISHED
    established = [e for e in events if e.type == SessionEventType.ESTABLISHED]
    assert established and established[0].data.get("world_grant") is not None


def test_initiator_rejects_expired_world_grant(alice_conn, alice_peer_id, bob_peer_id):
    """The world joiner re-checks the grant expiry when the WORLD_GRANT lands on its
    public Session: a stale grant fails the handshake (the only expiry check is here,
    since the PRIVATE_JOIN redemption round-trip is gone)."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = Session(
        peer_id=bob_peer_id, role=SessionRole.INITIATOR, p2p=alice_conn,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=alice_peer_id,
        local_node_id="local-node", local_profile_factory=lambda: {},
        addresses=["/ip4/127.0.0.1/tcp/0"], is_world_join=True,
    )
    _run(sess.start())
    _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, SessMsg(SessMsg.KIND_HELLO_RESP, {
        "chosen_version": "1.0", "nonce_b": b"\xbb" * 32, "sig_b": b"sig", "reason_code": OK,
    })))
    _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, SessMsg(SessMsg.KIND_PROFILE_DECISION, {
        "accepted": True, "profile_b": b"{}", "reason_code": OK, "session_id": sess.session_id,
    })))
    assert sess.state == SessionState.AWAITING_WORLD_GRANT
    expired = SessMsg(SessMsg.KIND_WORLD_GRANT, {
        "grant_id": b"\x05" * 32, "expiry_unix_ms": 1,  # long past
        "world_profile": json.dumps({"dynamic": {"private_peer_id": "12D3WorldPriv"}}).encode(),
        "role": 3, "code_bundle": b"", "initial_stats": b"", "rendezvous_tag": 0,
    })
    events = _run(sess.process_inbound(SESSION_WORLD_PROTOCOL, expired))
    assert len(events) == 1 and events[0].type == SessionEventType.FAILED
    assert "world_grant_expired" in events[0].data["reason"]


# ── In-session token refresh (TOKEN_UPDATE) + expiry ─────────────────────────

def _established_responder(bob_conn, bob_peer_id, alice_peer_id, verifier, node_hint="node-X"):
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id,
               token_verifier=verifier, remote_node_id_hint=node_hint)
    sess._set_state(SessionState.ESTABLISHED, None)
    return sess


def test_token_update_advances_expiry_monotonically(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """An ESTABLISHED peer's TOKEN_UPDATE advances the cached exp; a replayed older
    token cannot shrink it (monotonic). The fake verifier reads exp from the token."""
    async def _verifier(token, peer_id):
        return ("node-X", "cv2", (peer_id,), int(token))
    sess = _established_responder(bob_conn, bob_peer_id, alice_peer_id, _verifier)
    sess.remote_token_exp = 1000
    _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, SessMsg(SessMsg.KIND_TOKEN_UPDATE, {
        "session_id": sess.session_id, "token": "2000"})))
    assert sess.remote_token_exp == 2000 and sess.remote_cv_hash == "cv2"
    _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, SessMsg(SessMsg.KIND_TOKEN_UPDATE, {
        "session_id": sess.session_id, "token": "1500"})))
    assert sess.remote_token_exp == 2000   # not moved backwards


def test_token_update_rejects_node_id_mismatch(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """A TOKEN_UPDATE whose token attests a different node-id is ignored — the update
    is dropped, the session is not."""
    async def _verifier(token, peer_id):
        return ("intruder", "cv", (peer_id,), 9999999999)
    sess = _established_responder(bob_conn, bob_peer_id, alice_peer_id, _verifier)
    sess.remote_token_exp = 1000
    _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, SessMsg(SessMsg.KIND_TOKEN_UPDATE, {
        "session_id": sess.session_id, "token": "x"})))
    assert sess.remote_token_exp == 1000
    assert sess.state == SessionState.ESTABLISHED


def test_token_update_buffered_until_established(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """A TOKEN_UPDATE arriving before ESTABLISHED is buffered and applied on the next
    tick once established."""
    async def _verifier(token, peer_id):
        return ("node-X", "cv", (peer_id,), 5000)
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id,
               token_verifier=_verifier, remote_node_id_hint="node-X")
    sess._set_state(SessionState.RECONNECTING, 10.0)
    _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, SessMsg(SessMsg.KIND_TOKEN_UPDATE, {
        "session_id": sess.session_id, "token": "x"})))
    assert sess.remote_token_exp == 0 and sess._buffered_token_update is not None
    sess._set_state(SessionState.ESTABLISHED, None)
    _run(sess.tick_keepalive())
    assert sess.remote_token_exp == 5000 and sess._buffered_token_update is None


def test_tick_goaways_on_expired_token(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """An ESTABLISHED session whose peer token lapsed past the grace window is closed
    with a GOAWAY(TOKEN_EXPIRED)."""
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id)
    sess._set_state(SessionState.ESTABLISHED, None)
    sess.remote_token_exp = 1   # 1970 — far past + grace
    _run(sess.tick_keepalive())
    assert sess.state == SessionState.CLOSING
    last = _last_sent(bob_conn, SESSION_CONTROL_PROTOCOL)
    assert last is not None and last[2].kind == SessMsg.KIND_GOAWAY
    assert last[2].fields["reason"] == TOKEN_EXPIRED


def test_session_manager_pushes_token_update_on_token_change(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """SessionManager.tick pushes a TOKEN_UPDATE to ESTABLISHED sessions when its
    token_factory() (self.node_token) changes; the seeded value is not re-broadcast."""
    from unaiverse.networking.session import SessionManager
    tok = ["tok-1"]
    sm = SessionManager(p2p=bob_conn, p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=alice_peer_id,
                        local_node_id="alice", local_profile_factory=lambda: {},
                        token_factory=lambda: tok[0])
    sess = sm._spawn_responder(bob_peer_id)
    sess._set_state(SessionState.ESTABLISHED, None)

    def _updates():
        return [e[2] for e in bob_conn.sent if e[1] == SESSION_CONTROL_PROTOCOL
                and e[2].kind == SessMsg.KIND_TOKEN_UPDATE]

    _run(sm.tick())
    assert _updates() == []          # no change yet → no push
    tok[0] = "tok-2"
    _run(sm.tick())
    assert _updates() and _updates()[-1].fields["token"] == "tok-2"
