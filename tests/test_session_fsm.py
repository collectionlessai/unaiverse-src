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
    SESSION_UNKNOWN,
)
from unaiverse.networking.node.connpool import NodeConn


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


def _mk(peer_id, role, conn, local_peer_id, local_node_id="local-node",
        local_profile_factory=lambda: {}, addresses=None):
    return Session(
        peer_id=peer_id, role=role, conn=conn,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=local_peer_id,
        local_node_id=local_node_id, local_profile_factory=local_profile_factory,
        addresses=addresses,
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
    assert alice_conn.connect_calls == [(["/ip4/127.0.0.1/tcp/0/p2p/" + bob_peer_id], NodeConn.P2P_PUBLIC)]
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
    # signed_peer_record fetched lazily from the (fake) other instance.
    assert offer.fields["signed_peer_record"] == b"spr-bytes"
    assert offer.fields["sig_a"] == b"sig-bytes"

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
        "signed_peer_record": b"spr", "session_id": sess.session_id,
    })
    events = _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, offer))
    assert len(events) == 1 and events[0].type == SessionEventType.FAILED
    last = _last_sent(bob_conn, SESSION_PROFILE_PROTOCOL)
    assert last is not None
    decision = last[2]
    assert decision.kind == SessMsg.KIND_PROFILE_DECISION
    assert decision.fields["accepted"] is False
    assert decision.fields["reason_code"] == NOT_AUTHORIZED


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
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(SessionState.HELLO_SENT, 10.0)
    assert sess.check_timeout() is None


# ── Disconnect / grace / resume ─────────────────────────────────────────────

def test_disconnect_from_established_enters_reconnecting(alice_conn, alice_peer_id, bob_peer_id):
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
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(SessionState.RECONNECTING, 0.001)
    time.sleep(0.005)
    ev = sess.check_timeout()
    assert ev is not None and ev.type == SessionEventType.LOST
    assert sess.state == SessionState.FAILED


# ── Intentional close ───────────────────────────────────────────────────────

def test_close_sends_goaway_and_enters_closing(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
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
        ev = sess.tick_keepalive()
        assert ev is None
        assert sess._ping_in_flight is True
        await asyncio.sleep(0.05)   # let the fire-and-forget send run
        return sess
    sess = asyncio.run(_runner())
    last = _last_sent(alice_conn, SESSION_CONTROL_PROTOCOL)
    assert last is not None and last[2].kind == SessMsg.KIND_PING
    assert last[2].fields["session_id"] == sess.session_id
    # No /ipfs/ping was used.
    assert alice_conn.p2p_name_to_p2p[NodeConn.P2P_PUBLIC].pinged == []


def test_pong_ok_clears_failures(alice_conn, alice_peer_id, bob_peer_id):
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
    ev = sess.tick_keepalive(now)
    assert ev is None and sess._consecutive_ping_failures == 1
    # Second miss.
    sess._ping_in_flight = True
    sess._ping_deadline = now - 1.0
    ev = sess.tick_keepalive(now)
    assert ev is not None and ev.type == SessionEventType.RECONNECTING
    assert sess.state == SessionState.RECONNECTING


def test_on_ping_answers_pong_ok_when_session_matches(bob_conn, bob_peer_id, alice_peer_id):
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
        peer_id=bob_peer_id, role=SessionRole.INITIATOR, conn=alice_conn,
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
    assert sess.tick_reconnect() is None
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
        sess.tick_reconnect()
        await asyncio.sleep(0.05)   # let the fire-and-forget re-dial+resume run
        return sess
    sess = asyncio.run(_runner())
    assert alice_conn.connect_calls          # a re-dial happened
    assert sess._reconnect_substep == "resume_sent"
    last = _last_sent(alice_conn, SESSION_CONTROL_PROTOCOL)
    assert last is not None and last[2].kind == SessMsg.KIND_RESUME


# ── Silent abort (tie-breaker support) ──────────────────────────────────────

def test_abort_silently_fails_without_event(alice_conn, alice_peer_id, bob_peer_id):
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(SessionState.HELLO_SENT, 5.0)
    sess.abort_silently("simultaneous_open_yielded")
    assert sess.state == SessionState.FAILED
