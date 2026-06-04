"""Unit tests for ``SessionManager``: tie-breaker, quarantine, dispatch.

These tests don't exercise full handshakes (covered by the in-process
integration test) — they isolate SessionManager's own logic: how it tracks
sessions per peer, how the tie-breaker resolves simultaneous open, and how
the anti-flap quarantine gates re-dials after a Lost event.
"""

import asyncio
import time

import pytest

from unaiverse.networking.session import (
    SessionManager, Session, SessionRole, SessionState, SessionEventType, SessMsg,
    SESSION_HELLO_PROTOCOL, SESSION_PROFILE_PROTOCOL, SESSION_CONTROL_PROTOCOL,
)
from unaiverse.networking.session.sess_msg import SIMULTANEOUS_OPEN_LOSS, WORLD_DISENGAGEMENT
from unaiverse.networking.node.connpool import NodeConn


def _run(coro):
    return asyncio.run(coro)


def _make_sm(conn, p2p_name, local_peer_id, profile_factory=None):
    return SessionManager(
        conn=conn,
        p2p_name=p2p_name,
        local_peer_id=local_peer_id,
        local_node_id="local-node-id",
        local_profile_factory=profile_factory or (lambda: {"node_id": "local"}),
    )


def _mk_session(conn, peer_id, local_peer_id, role=SessionRole.INITIATOR):
    return Session(
        peer_id=peer_id, role=role, conn=conn,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=local_peer_id,
        local_node_id="alice", local_profile_factory=lambda: {},
    )


# ── initiate() ──────────────────────────────────────────────────────────────

def test_initiate_idempotent_when_already_healthy(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """Calling initiate twice for the same healthy peer must be a no-op the second time."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    events_1 = _run(sm.initiate(bob_peer_id, ["/ip4/127.0.0.1/tcp/0"]))
    assert events_1 == []
    events_2 = _run(sm.initiate(bob_peer_id, ["/ip4/127.0.0.1/tcp/0"]))
    assert events_2 == []
    assert len(alice_conn.connect_calls) == 1


def test_initiate_during_quarantine_returns_failed(alice_conn, alice_peer_id, bob_peer_id):
    """A Lost event must put the peer in quarantine; the next dial returns
    [FAILED] immediately without touching the wire."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sm._quarantine[bob_peer_id] = time.monotonic() + 30.0
    events = _run(sm.initiate(bob_peer_id, ["/ip4/127.0.0.1/tcp/0"]))
    assert len(events) == 1 and events[0].type == SessionEventType.FAILED
    assert events[0].data["reason"] == "quarantined"
    assert alice_conn.connect_calls == []


# ── deliver() — basics ──────────────────────────────────────────────────────

def test_deliver_unknown_protocol_drops(alice_conn, alice_peer_id, bob_peer_id):
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    events = _run(sm.deliver(bob_peer_id, "/something/else/1.0.0", SessMsg(SessMsg.KIND_HELLO_REQ, {})))
    assert events == []


def test_deliver_unknown_peer_non_hello_drops(alice_conn, alice_peer_id, bob_peer_id):
    """A PROFILE message from a peer we have no session with → silently dropped."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    events = _run(sm.deliver(bob_peer_id, SESSION_PROFILE_PROTOCOL,
                             SessMsg(SessMsg.KIND_PROFILE_OFFER, {})))
    assert events == []


def test_deliver_inbound_hello_spawns_responder(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """Unknown peer sends HelloRequest → SessionManager spawns a responder Session
    and runs the message through it."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    hello = SessMsg(SessMsg.KIND_HELLO_REQ, {
        "session_id": b"\x02" * 16, "nonce_a": b"\xaa" * 32,
        "protocol_versions": ["1.0"], "client_id": "test",
    })
    events = _run(sm.deliver(bob_peer_id, SESSION_HELLO_PROTOCOL, hello))
    assert events == []
    sess = sm.get_session(bob_peer_id)
    assert sess is not None
    assert sess.role == SessionRole.RESPONDER
    assert sess.state == SessionState.HELLO_ACK_SENT


# ── Tie-breaker on simultaneous open ────────────────────────────────────────

def test_tiebreak_smaller_peer_keeps_initiator(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """When local_peer_id < remote_peer_id (alice < bob), Alice keeps her
    initiator and rejects Bob's inbound HELLO with SIMULTANEOUS_OPEN_LOSS."""
    assert alice_peer_id < bob_peer_id, "fixture invariant"
    alice_conn.connect_returns = (bob_peer_id, False)
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    _run(sm.initiate(bob_peer_id, ["/ip4/1.2.3.4/tcp/0"]))
    sess_before = sm.get_session(bob_peer_id)
    assert sess_before.role == SessionRole.INITIATOR
    hello = SessMsg(SessMsg.KIND_HELLO_REQ, {
        "session_id": b"\x09" * 16, "nonce_a": b"\xaa" * 32, "protocol_versions": ["1.0"],
    })
    events = _run(sm.deliver(bob_peer_id, SESSION_HELLO_PROTOCOL, hello))
    assert events == []
    sess_after = sm.get_session(bob_peer_id)
    assert sess_after is sess_before
    assert sess_after.role == SessionRole.INITIATOR
    sent_helloresponses = [s for s in alice_conn.sent if s[1] == SESSION_HELLO_PROTOCOL]
    last = sent_helloresponses[-1]
    assert last[2].kind == SessMsg.KIND_HELLO_RESP
    assert last[2].fields["reason_code"] == SIMULTANEOUS_OPEN_LOSS


def test_tiebreak_larger_peer_yields_to_responder(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """When local_peer_id > remote_peer_id (bob > alice), Bob aborts his
    initiator and accepts Alice's inbound HELLO as a responder."""
    assert bob_peer_id > alice_peer_id, "fixture invariant"
    bob_conn.connect_returns = (alice_peer_id, False)
    sm = _make_sm(bob_conn, NodeConn.P2P_PUBLIC, bob_peer_id)
    _run(sm.initiate(alice_peer_id, ["/ip4/1.2.3.4/tcp/0"]))
    sess_before = sm.get_session(alice_peer_id)
    assert sess_before.role == SessionRole.INITIATOR
    initiator_session_id_before = sess_before.session_id
    hello = SessMsg(SessMsg.KIND_HELLO_REQ, {
        "session_id": b"\x42" * 16, "nonce_a": b"\xab" * 32, "protocol_versions": ["1.0"],
    })
    _run(sm.deliver(alice_peer_id, SESSION_HELLO_PROTOCOL, hello))
    sess_after = sm.get_session(alice_peer_id)
    assert sess_after is not sess_before
    assert sess_after.role == SessionRole.RESPONDER
    assert sess_after.session_id == b"\x42" * 16
    assert sess_after.session_id != initiator_session_id_before


# ── tick() ──────────────────────────────────────────────────────────────────

def test_tick_drains_timeouts_and_cleans_terminal_sessions(alice_conn, alice_peer_id, bob_peer_id):
    """Sessions in a state past their timeout → FAILED event; terminal
    sessions are removed from the registry on the same tick."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sess = _mk_session(alice_conn, bob_peer_id, alice_peer_id)
    sm._sessions[bob_peer_id] = sess
    sess._set_state(SessionState.HELLO_SENT, 0.001)
    time.sleep(0.005)
    events = _run(sm.tick())
    assert any(e.type == SessionEventType.FAILED for e in events)
    assert sm.get_session(bob_peer_id) is None


def test_lost_event_enters_quarantine(alice_conn, alice_peer_id, bob_peer_id):
    """A Lost event from grace expiry must put the peer in quarantine."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sess = _mk_session(alice_conn, bob_peer_id, alice_peer_id)
    sm._sessions[bob_peer_id] = sess
    sess._set_state(SessionState.RECONNECTING, 0.001)
    time.sleep(0.005)
    events = _run(sm.tick())
    assert any(e.type == SessionEventType.LOST for e in events)
    ban_until = sm._quarantine.get(bob_peer_id)
    assert ban_until is not None
    assert ban_until > time.monotonic() + 25.0


# ── on_peer_disconnected() ──────────────────────────────────────────────────

def test_disconnect_during_handshake_fails_immediately(alice_conn, alice_peer_id, bob_peer_id):
    """Connection drop while still mid-handshake → FAILED event right away."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sess = _mk_session(alice_conn, bob_peer_id, alice_peer_id)
    sm._sessions[bob_peer_id] = sess
    sess._set_state(SessionState.HELLO_SENT, 5.0)
    events = _run(sm.on_peer_disconnected(bob_peer_id))
    assert len(events) == 1 and events[0].type == SessionEventType.FAILED
    assert sm.get_session(bob_peer_id) is None


def test_disconnect_from_established_emits_reconnecting(alice_conn, alice_peer_id, bob_peer_id):
    """Disconnect from ESTABLISHED → RECONNECTING (grace), session still live."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sess = _mk_session(alice_conn, bob_peer_id, alice_peer_id)
    sm._sessions[bob_peer_id] = sess
    sess._set_state(SessionState.ESTABLISHED, None)
    events = _run(sm.on_peer_disconnected(bob_peer_id))
    assert len(events) == 1 and events[0].type == SessionEventType.RECONNECTING
    assert sm.get_session(bob_peer_id) is sess
    assert sess.state == SessionState.RECONNECTING


def test_disconnect_unknown_peer_is_noop(alice_conn, alice_peer_id, bob_peer_id):
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    events = _run(sm.on_peer_disconnected(bob_peer_id))
    assert events == []


# ── close_session() ─────────────────────────────────────────────────────────

def test_close_session_sends_goaway(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sess = _mk_session(alice_conn, bob_peer_id, alice_peer_id)
    sm._sessions[bob_peer_id] = sess
    sess._set_state(SessionState.ESTABLISHED, None)
    _run(sm.close_session(bob_peer_id, WORLD_DISENGAGEMENT))
    assert sess.state == SessionState.CLOSING
    sent = [s for s in alice_conn.sent if s[1] == SESSION_CONTROL_PROTOCOL]
    assert len(sent) == 1
    assert sent[0][2].kind == SessMsg.KIND_GOAWAY
    assert sent[0][2].fields["reason"] == WORLD_DISENGAGEMENT


# ── Unknown-session liveness reply (restarted-peer detection) ────────────────

def test_deliver_unknown_session_ping_replies_pong_not_ok(alice_conn, alice_peer_id, bob_peer_id):
    """A PING for a peer we hold no Session for (we are the restarted peer) must
    get a PONG ok=false so the initiator stops waiting and re-handshakes."""
    from unaiverse.networking.session.sess_msg import SESSION_UNKNOWN
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    ping = SessMsg(SessMsg.KIND_PING, {"session_id": b"\x01" * 16, "nonce": b"n1"})
    events = _run(sm.deliver(bob_peer_id, SESSION_CONTROL_PROTOCOL, ping))
    assert events == []
    assert sm.get_session(bob_peer_id) is None       # no Session spawned
    sent = [s for s in alice_conn.sent if s[1] == SESSION_CONTROL_PROTOCOL]
    assert sent and sent[-1][2].kind == SessMsg.KIND_PONG
    assert sent[-1][2].fields["ok"] is False
    assert sent[-1][2].fields["nonce"] == b"n1"


def test_deliver_unknown_session_resume_naks(alice_conn, alice_peer_id, bob_peer_id):
    from unaiverse.networking.session.sess_msg import SESSION_UNKNOWN
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    resume = SessMsg(SessMsg.KIND_RESUME, {"session_id": b"\x01" * 16, "nonce": b"r1"})
    events = _run(sm.deliver(bob_peer_id, SESSION_CONTROL_PROTOCOL, resume))
    assert events == []
    sent = [s for s in alice_conn.sent if s[1] == SESSION_CONTROL_PROTOCOL]
    assert sent and sent[-1][2].kind == SessMsg.KIND_RESUME_ACK
    assert sent[-1][2].fields["accepted"] is False
    assert sent[-1][2].fields["reason_code"] == SESSION_UNKNOWN


# ── close_all() ──────────────────────────────────────────────────────────────

def test_close_all_goaways_established_sessions(alice_conn, alice_peer_id, bob_peer_id):
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sess = _mk_session(alice_conn, bob_peer_id, alice_peer_id)
    sess._set_state(SessionState.ESTABLISHED, None)
    sm._sessions[bob_peer_id] = sess
    _run(sm.close_all(WORLD_DISENGAGEMENT))
    assert sess.state == SessionState.CLOSING
    sent = [s for s in alice_conn.sent if s[1] == SESSION_CONTROL_PROTOCOL]
    assert sent and sent[-1][2].kind == SessMsg.KIND_GOAWAY
    assert sent[-1][2].fields["reason"] == WORLD_DISENGAGEMENT


def test_close_all_skips_already_closing(alice_conn, alice_peer_id, bob_peer_id):
    """A Session already CLOSING (e.g. intentional world-handoff GOAWAY) must not
    be re-GOAWAY'd by close_all."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sess = _mk_session(alice_conn, bob_peer_id, alice_peer_id)
    sess._set_state(SessionState.CLOSING, 10.0)
    sm._sessions[bob_peer_id] = sess
    _run(sm.close_all())
    sent = [s for s in alice_conn.sent if s[1] == SESSION_CONTROL_PROTOCOL]
    assert sent == []   # Session.close is a no-op outside ESTABLISHED/RECONNECTING
