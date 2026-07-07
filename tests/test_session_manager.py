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
from unaiverse.networking.session.sess_msg import (
    OK, NOT_AUTHORIZED, SIMULTANEOUS_OPEN_LOSS, WORLD_DISENGAGEMENT,
)
from unaiverse.networking.node.connpool import NodeConn

pytestmark = pytest.mark.unit


def _run(coro):
    return asyncio.run(coro)


def _make_sm(p2p, p2p_name, local_peer_id, profile_factory=None):
    return SessionManager(
        p2p=p2p,
        p2p_name=p2p_name,
        local_peer_id=local_peer_id,
        local_node_id="local-node-id",
        local_profile_factory=profile_factory or (lambda: {"node_id": "local"}),
    )


def _mk_session(p2p, peer_id, local_peer_id, role=SessionRole.INITIATOR):
    return Session(
        peer_id=peer_id, role=role, p2p=p2p,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=local_peer_id,
        local_node_id="alice", local_profile_factory=lambda: {},
    )


# initiate()

def test_initiate_idempotent_when_already_healthy(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """Calling initiate twice for the same healthy peer must be a no-op the second time."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    events_1 = _run(sm.initiate(bob_peer_id, ["/ip4/127.0.0.1/tcp/0"]))
    assert events_1 == []
    events_2 = _run(sm.initiate(bob_peer_id, ["/ip4/127.0.0.1/tcp/0"]))
    assert events_2 == []
    assert len(alice_conn.connect_calls) == 1


def test_initiate_replaces_on_divergent_world_join_intent(alice_conn, alice_peer_id, bob_peer_id):
    """A non-terminal Session whose world-join intent DIFFERS from the request is replaced
    (dropped + re-dialed), not silently reused: otherwise a world join would be lost to a
    leftover plain Session and the caller would see success while nothing (re)starts (B8)."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    _run(sm.initiate(bob_peer_id, ["/ip4/127.0.0.1/tcp/0"], is_world_join=False))
    assert sm.get_session(bob_peer_id).is_world_join is False
    # Same peer, now a WORLD-join intent → the plain Session must be replaced, not no-opped.
    _run(sm.initiate(bob_peer_id, ["/ip4/127.0.0.1/tcp/0"], is_world_join=True))
    assert sm.get_session(bob_peer_id).is_world_join is True
    assert len(alice_conn.connect_calls) == 2            # re-dialed
    assert bob_peer_id in alice_conn.disconnect_calls    # the stale one was dropped


def test_initiate_replace_stale_forces_replacement_at_matching_intent(alice_conn, alice_peer_id, bob_peer_id):
    """replace_stale (the Coordinator's spent-carrier flag) replaces a non-terminal Session
    even when the world-join intent matches, so a spent carrier does not no-op forever (B8)."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    _run(sm.initiate(bob_peer_id, ["/ip4/127.0.0.1/tcp/0"], is_world_join=True))
    # Matching intent, no flag → no-op (idempotent).
    _run(sm.initiate(bob_peer_id, ["/ip4/127.0.0.1/tcp/0"], is_world_join=True))
    assert len(alice_conn.connect_calls) == 1
    # Matching intent, but replace_stale set → replace and re-dial.
    _run(sm.initiate(bob_peer_id, ["/ip4/127.0.0.1/tcp/0"], is_world_join=True, replace_stale=True))
    assert len(alice_conn.connect_calls) == 2


def test_has_established_world_join_carrier(alice_conn, alice_peer_id, bob_peer_id):
    """The spent-carrier detector is True only for an ESTABLISHED world-join Session: an
    in-flight handshake (not yet ESTABLISHED) and a plain Session are never flagged, so a
    live join is never mistaken for a spent carrier."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    assert sm.has_established_world_join_carrier(bob_peer_id) is False   # no Session
    s = _mk_session(alice_conn, bob_peer_id, alice_peer_id)
    s.is_world_join = True
    sm._sessions[bob_peer_id] = s
    s._set_state(SessionState.HELLO_SENT, None)
    assert sm.has_established_world_join_carrier(bob_peer_id) is False   # in-flight, not spent
    s._set_state(SessionState.ESTABLISHED, None)
    assert sm.has_established_world_join_carrier(bob_peer_id) is True    # established carrier
    s.is_world_join = False
    assert sm.has_established_world_join_carrier(bob_peer_id) is False   # plain session


def test_initiate_during_quarantine_returns_failed(alice_conn, alice_peer_id, bob_peer_id):
    """A Lost event must put the peer in quarantine; the next dial returns
    [FAILED] immediately without touching the wire."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sm._quarantine[bob_peer_id] = time.monotonic() + 30.0
    events = _run(sm.initiate(bob_peer_id, ["/ip4/127.0.0.1/tcp/0"]))
    assert len(events) == 1 and events[0].type == SessionEventType.FAILED
    assert events[0].data["reason"] == "quarantined"
    assert alice_conn.connect_calls == []


# deliver() — basics

def test_deliver_unknown_protocol_drops(alice_conn, alice_peer_id, bob_peer_id):
    """A message on a protocol id the manager doesn't route is dropped silently,
    yielding no events and spawning no Session."""
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


@pytest.mark.xfail(reason="bug: deliver's terminal-evict (session_manager.py:445) only covers "
                          "FAILED/CLOSED, so a stale CLOSING/RECONNECTING responder shadows a "
                          "fresh inbound HELLO (the HELLO_SENT re-dial failure)",
                   strict=True)
@pytest.mark.parametrize("stale_state, stale_timeout", [
    (SessionState.RECONNECTING, 45.0),   # link dropped, riding the grace window
    (SessionState.CLOSING, 10.0),        # our GOAWAY still in flight
])
def test_deliver_fresh_hello_supersedes_stale_responder(
    alice_conn, alice_peer_id, bob_peer_id, stale_state, stale_timeout
):
    """A fresh inbound HELLO from a peer we still hold a leftover responder Session
    for must be answered, not shadowed by the dead handshake.

    deliver() only evicts FAILED/CLOSED sessions before routing an inbound HELLO.
    A responder Session stuck in a non-terminal, non-INIT state (RECONNECTING after
    a link drop, or CLOSING while a GOAWAY is in flight) survives that eviction, so
    the fresh HELLO reaches _on_hello_protocol, which ignores it because the state
    is not INIT. No HELLO_ACK goes out and the remote initiator rides its 5s
    HELLO_SENT timeout to FAILED. This is the "re-dial into a just-closed session"
    HELLO_SENT failure.
    """
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)

    # A leftover responder Session for bob, from the session that just went away.
    stale = _mk_session(alice_conn, bob_peer_id, alice_peer_id, role=SessionRole.RESPONDER)
    stale.session_id = b"\x11" * 16
    stale._set_state(stale_state, stale_timeout)
    sm._set_active(bob_peer_id, stale)

    # bob re-dials and opens a brand-new handshake (a different session_id).
    hello = SessMsg(SessMsg.KIND_HELLO_REQ, {
        "session_id": b"\x02" * 16, "nonce_a": b"\xaa" * 32,
        "protocol_versions": ["1.0"], "client_id": "test",
    })
    events = _run(sm.deliver(bob_peer_id, SESSION_HELLO_PROTOCOL, hello))
    assert events == []

    # The fresh HELLO must be answered with a HELLO_ACK (reason OK)...
    acks = [s for s in alice_conn.sent
            if s[1] == SESSION_HELLO_PROTOCOL
            and s[2].kind == SessMsg.KIND_HELLO_RESP
            and s[2].fields.get("reason_code") == OK]
    assert acks, "stale responder Session shadowed the fresh HELLO: no HELLO_ACK sent"

    # ...backed by a brand-new responder Session bound to the new session_id.
    sess_after = sm.get_session(bob_peer_id)
    assert sess_after is not stale
    assert sess_after.role == SessionRole.RESPONDER
    assert sess_after.state == SessionState.HELLO_ACK_SENT
    assert sess_after.session_id == b"\x02" * 16


# Tie-breaker on simultaneous open

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


# tick()

def test_tick_drains_timeouts_and_cleans_terminal_sessions(alice_conn, alice_peer_id, bob_peer_id):
    """Sessions in a state past their timeout → FAILED event; terminal
    sessions are removed from the registry on the same tick."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sess = _mk_session(alice_conn, bob_peer_id, alice_peer_id)
    sm._set_active(bob_peer_id, sess)
    sess._set_state(SessionState.HELLO_SENT, 0.001)
    time.sleep(0.005)
    events = _run(sm.tick())
    assert any(e.type == SessionEventType.FAILED for e in events)
    assert sm.get_session(bob_peer_id) is None


def test_lost_event_enters_quarantine(alice_conn, alice_peer_id, bob_peer_id):
    """A Lost event from grace expiry must put the peer in quarantine."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sess = _mk_session(alice_conn, bob_peer_id, alice_peer_id)
    sm._set_active(bob_peer_id, sess)
    sess._set_state(SessionState.RECONNECTING, 0.001)
    time.sleep(0.005)
    events = _run(sm.tick())
    assert any(e.type == SessionEventType.LOST for e in events)
    ban_until = sm._quarantine.get(bob_peer_id)
    assert ban_until is not None
    assert ban_until > time.monotonic() + 20.0


# on_peer_disconnected()

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
    """A disconnect for a peer the manager holds no Session for is a no-op: no events."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    events = _run(sm.on_peer_disconnected(bob_peer_id))
    assert events == []


# close_session()

def test_close_session_sends_goaway(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """close_session on an ESTABLISHED session moves it to CLOSING and emits a single
    GOAWAY on the control protocol carrying the given reason code."""
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


# Unknown-session liveness reply (restarted-peer detection)

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
    """A RESUME for a peer we hold no Session for is NAK'd: a RESUME_ACK with
    accepted=false and reason_code SESSION_UNKNOWN, and no Session is spawned."""
    from unaiverse.networking.session.sess_msg import SESSION_UNKNOWN
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    resume = SessMsg(SessMsg.KIND_RESUME, {"session_id": b"\x01" * 16, "nonce": b"r1"})
    events = _run(sm.deliver(bob_peer_id, SESSION_CONTROL_PROTOCOL, resume))
    assert events == []
    sent = [s for s in alice_conn.sent if s[1] == SESSION_CONTROL_PROTOCOL]
    assert sent and sent[-1][2].kind == SessMsg.KIND_RESUME_ACK
    assert sent[-1][2].fields["accepted"] is False
    assert sent[-1][2].fields["reason_code"] == SESSION_UNKNOWN


# close_all()

def test_close_all_goaways_established_sessions(alice_conn, alice_peer_id, bob_peer_id):
    """close_all moves each ESTABLISHED session to CLOSING and sends it a GOAWAY on the
    control protocol carrying the given reason code."""
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


def test_session_manager_injects_token_capabilities_into_sessions(alice_p2p, alice_peer_id, bob_peer_id):
    """RT plumbing: the manager forwards token_verifier and token_factory to every
    Session it builds, on both the initiator path (ensure) and the responder path
    (spawn). Not consumed yet — they must just reach the Session. With none injected
    the Sessions carry None (no silent default)."""
    verifier = object()            # sentinel; identity-checked, not called here
    factory = lambda: "tok-123"    # noqa: E731
    sm = SessionManager(
        p2p=alice_p2p, p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=alice_peer_id,
        local_node_id="alice", local_profile_factory=lambda: {},
        token_verifier=verifier, token_factory=factory,
    )
    initiator = sm.ensure(bob_peer_id, ["/ip4/127.0.0.1/tcp/0"])
    responder = sm._spawn_responder(bob_peer_id)
    for sess in (initiator, responder):
        assert sess.token_verifier is verifier
        assert sess.token_factory is factory
        assert sess.token_factory() == "tok-123"

    sm_bare = SessionManager(
        p2p=alice_p2p, p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=alice_peer_id,
        local_node_id="alice", local_profile_factory=lambda: {},
    )
    bare = sm_bare._spawn_responder(bob_peer_id)
    assert bare.token_verifier is None and bare.token_factory is None


# KNOWN state and ensure()/activate() split

def test_ensure_creates_known_session_without_dialing(alice_conn, alice_peer_id, bob_peer_id):
    """ensure() registers an initiator Session in KNOWN without dialing; activate() is
    what starts it."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sess = sm.ensure(bob_peer_id, ["/ip4/127.0.0.1/tcp/0"])
    assert sess.state == SessionState.KNOWN
    assert alice_conn.connect_calls == []          # no dial yet
    assert sm.get_session(bob_peer_id) is sess     # registered


def test_ensure_is_idempotent_for_healthy_session(alice_conn, alice_peer_id, bob_peer_id):
    """A second ensure() returns the same (non-terminal) Session, not a fresh one."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    s1 = sm.ensure(bob_peer_id, ["/ip4/127.0.0.1/tcp/0"])
    s2 = sm.ensure(bob_peer_id, ["/ip4/127.0.0.1/tcp/0"])
    assert s1 is s2


def test_initiate_is_ensure_plus_activate(alice_conn, alice_peer_id, bob_peer_id):
    """initiate() = ensure() + activate(): the Session is dialed and moves past KNOWN,
    unchanged from before the split."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    _run(sm.initiate(bob_peer_id, ["/ip4/127.0.0.1/tcp/0"]))
    assert len(alice_conn.connect_calls) == 1
    sess = sm.get_session(bob_peer_id)
    assert sess is not None and sess.state != SessionState.KNOWN


def test_spawn_responder_starts_in_init(alice_conn, alice_peer_id, bob_peer_id):
    """A responder is engaged from the start (awaiting HELLO), so it lives in INIT,
    not KNOWN."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sess = sm._spawn_responder(bob_peer_id)
    assert sess.state == SessionState.INIT


# Double view: tick-active vs known

def test_ensure_keeps_known_out_of_active_view(alice_conn, alice_peer_id, bob_peer_id):
    """A KNOWN Session lives in the full registry but not in the tick-active view, so
    the per-cycle tick skips it in O(1)."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sess = sm.ensure(bob_peer_id, ["/ip4/127.0.0.1/tcp/0"])
    assert sess.state == SessionState.KNOWN
    assert bob_peer_id in sm._sessions
    assert bob_peer_id not in sm._active_sessions


def test_activate_promotes_known_into_active_view(alice_conn, alice_peer_id, bob_peer_id):
    """activate() moves a started initiator out of KNOWN into the tick-active view."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sm.ensure(bob_peer_id, ["/ip4/127.0.0.1/tcp/0"])
    assert bob_peer_id not in sm._active_sessions
    _run(sm.activate(bob_peer_id))
    assert bob_peer_id in sm._active_sessions
    assert sm._active_sessions[bob_peer_id] is sm.get_session(bob_peer_id)


def test_spawn_responder_is_tick_active(alice_conn, alice_peer_id, bob_peer_id):
    """A responder is engaged from the start, so it enters the tick-active view."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sm._spawn_responder(bob_peer_id)
    assert bob_peer_id in sm._active_sessions


def test_tick_does_not_sweep_known_sessions(alice_conn, alice_peer_id, bob_peer_id):
    """The tick iterates only the active view: a KNOWN Session's tick hooks are never
    invoked (here they would raise), so a large KNOWN roster costs nothing per cycle."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sess = sm.ensure(bob_peer_id, ["/ip4/127.0.0.1/tcp/0"])
    assert sess.state == SessionState.KNOWN

    def _boom(*a, **k):
        raise AssertionError("tick touched a KNOWN session")
    sess.check_timeout = _boom
    sess.tick_dial = _boom
    sess.tick_keepalive = _boom

    events = _run(sm.tick())
    assert events == []
    assert sess.state == SessionState.KNOWN


def test_terminal_session_leaves_both_views(alice_conn, alice_peer_id, bob_peer_id):
    """A Session that fails (here via a transport drop mid-handshake) is removed from
    both the full registry and the tick-active view, so neither leaks a dead FSM."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    _run(sm.initiate(bob_peer_id, ["/ip4/127.0.0.1/tcp/0"]))
    assert bob_peer_id in sm._active_sessions
    _run(sm.on_peer_disconnected(bob_peer_id))
    assert bob_peer_id not in sm._sessions
    assert bob_peer_id not in sm._active_sessions


# KNOWN is not an in-flight handshake on inbound

def test_known_session_yields_to_responder_on_inbound_hello(alice_conn, alice_peer_id, bob_peer_id):
    """A KNOWN initiator (registered, not yet dialed) that receives an inbound HELLO
    yields: the peer dialed us first, so the KNOWN intent is dropped and a responder
    takes over the inbound session — no simultaneous-open tie-breaker, nothing was in
    flight."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    known = sm.ensure(bob_peer_id, ["/ip4/127.0.0.1/tcp/0"])
    assert known.state == SessionState.KNOWN
    hello = SessMsg(SessMsg.KIND_HELLO_REQ, {
        "session_id": b"\x42" * 16, "nonce_a": b"\xaa" * 32, "protocol_versions": ["1.0"],
    })
    _run(sm.deliver(bob_peer_id, SESSION_HELLO_PROTOCOL, hello))
    sess = sm.get_session(bob_peer_id)
    assert sess is not None and sess is not known          # KNOWN replaced
    assert sess.role == SessionRole.RESPONDER
    assert sess.session_id == b"\x42" * 16                 # adopted the inbound session
    assert bob_peer_id in sm._active_sessions              # responder is tick-active


def test_known_session_survives_inbound_ping(alice_conn, alice_peer_id, bob_peer_id):
    """A liveness PING for a session we never established does not destroy a standing
    KNOWN intent: we reply 'unknown session' and keep the KNOWN (which a later
    activate() can still dial)."""
    from unaiverse.networking.session.sess_msg import SESSION_UNKNOWN
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    known = sm.ensure(bob_peer_id, ["/ip4/127.0.0.1/tcp/0"])
    ping = SessMsg(SessMsg.KIND_PING, {"session_id": b"\x02" * 16, "nonce": b"\xbb"})
    events = _run(sm.deliver(bob_peer_id, SESSION_CONTROL_PROTOCOL, ping))
    assert events == []
    assert sm.get_session(bob_peer_id) is known            # KNOWN intact
    assert known.state == SessionState.KNOWN
    assert bob_peer_id not in sm._active_sessions          # still inert
    sent = [s for s in alice_conn.sent if s[1] == SESSION_CONTROL_PROTOCOL]
    assert sent[-1][2].kind == SessMsg.KIND_PONG
    assert sent[-1][2].fields["ok"] is False


# Known roster sync from rendezvous

def test_sync_known_peers_registers_inert_known(alice_conn, alice_peer_id, bob_peer_id):
    """sync_known_peers registers a KNOWN Session (inert: not dialed, not tick-active)
    for each known peer that has no Session yet, carrying its rendezvous addresses and
    applicative role."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sm.sync_known_peers({bob_peer_id: (["/ip4/1.1.1.1/tcp/1"], 1)})
    sess = sm.get_session(bob_peer_id)
    assert sess is not None and sess.state == SessionState.KNOWN
    assert sess.addresses == ["/ip4/1.1.1.1/tcp/1"]
    assert sess.app_role == 1
    assert bob_peer_id not in sm._active_sessions
    assert alice_conn.connect_calls == []


def test_sync_known_peers_is_idempotent(alice_conn, alice_peer_id, bob_peer_id):
    """A second sync with the same peer keeps the existing KNOWN Session, not a fresh one."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sm.sync_known_peers({bob_peer_id: (["/ip4/1.1.1.1/tcp/1"], 1)})
    s1 = sm.get_session(bob_peer_id)
    sm.sync_known_peers({bob_peer_id: (["/ip4/1.1.1.1/tcp/1"], 1)})
    assert sm.get_session(bob_peer_id) is s1


def test_sync_known_peers_refreshes_role_and_addrs_on_existing(alice_conn, alice_peer_id, bob_peer_id):
    """A later sync refreshes addresses and app_role on the existing Session without
    replacing it or touching its FSM state."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sm.sync_known_peers({bob_peer_id: (["/ip4/1.1.1.1/tcp/1"], 1)})
    sess = sm.get_session(bob_peer_id)
    sm.sync_known_peers({bob_peer_id: (["/ip4/2.2.2.2/tcp/2"], 3)})   # promoted to master
    assert sm.get_session(bob_peer_id) is sess        # same Session, not replaced
    assert sess.app_role == 3
    assert sess.addresses == ["/ip4/2.2.2.2/tcp/2"]
    assert sess.state == SessionState.KNOWN           # FSM untouched


def test_sync_known_peers_refreshes_active_session_metadata(alice_conn, alice_peer_id, bob_peer_id):
    """The metadata refresh reaches an active (non-KNOWN) Session too: its app_role and
    addresses update while its FSM state is left untouched."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sess = _mk_session(alice_conn, bob_peer_id, alice_peer_id)
    sm._set_active(bob_peer_id, sess)
    sess._set_state(SessionState.ESTABLISHED, None)
    sm.sync_known_peers({bob_peer_id: (["/ip4/3.3.3.3/tcp/3"], 3)})
    assert sm.get_session(bob_peer_id) is sess
    assert sess.app_role == 3
    assert sess.addresses == ["/ip4/3.3.3.3/tcp/3"]
    assert sess.state == SessionState.ESTABLISHED


def test_sync_known_peers_prunes_vanished_known(alice_conn, alice_peer_id, bob_peer_id):
    """A KNOWN Session whose peer is no longer advertised is pruned on the next sync."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sm.sync_known_peers({bob_peer_id: (["/ip4/1.1.1.1/tcp/1"], 1)})
    assert sm.get_session(bob_peer_id) is not None
    sm.sync_known_peers({})
    assert sm.get_session(bob_peer_id) is None


def test_sync_known_peers_does_not_prune_active_session(alice_conn, alice_peer_id, bob_peer_id):
    """sync never prunes a non-KNOWN Session: an ESTABLISHED peer that drops out of the
    rendezvous keeps its live Session (the transport, not the roster, ends it)."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sess = _mk_session(alice_conn, bob_peer_id, alice_peer_id)
    sm._set_active(bob_peer_id, sess)
    sess._set_state(SessionState.ESTABLISHED, None)
    sm.sync_known_peers({})
    assert sm.get_session(bob_peer_id) is sess


def test_initiate_activates_existing_known_placeholder(alice_conn, alice_peer_id, bob_peer_id):
    """initiate() on a peer that already has a KNOWN placeholder dials it instead of
    no-op'ing: the placeholder is replaced by the caller's parameters and started."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sm.sync_known_peers({bob_peer_id: (["/ip4/1.1.1.1/tcp/1"], 1)})
    assert sm.get_session(bob_peer_id).state == SessionState.KNOWN
    _run(sm.initiate(bob_peer_id, ["/ip4/2.2.2.2/tcp/2"]))
    assert len(alice_conn.connect_calls) == 1
    sess = sm.get_session(bob_peer_id)
    assert sess is not None and sess.state != SessionState.KNOWN
    assert sess.app_role == 1                          # rendezvous role carried across the replacement
    assert bob_peer_id in sm._active_sessions


# Quarantine gate in activate(), FAILED-quarantines

def test_activate_refused_during_quarantine(alice_conn, alice_peer_id, bob_peer_id):
    """activate() is the single dial gate: a quarantined peer is refused immediately
    with FAILED(quarantined), without dialing, and the inert KNOWN stays in place for a
    later activate after the window."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sm.sync_known_peers({bob_peer_id: (["/ip4/1.1.1.1/tcp/1"], 1)})   # KNOWN exists
    sm._quarantine[bob_peer_id] = time.monotonic() + 30.0
    events = _run(sm.activate(bob_peer_id))
    assert len(events) == 1 and events[0].type == SessionEventType.FAILED
    assert events[0].data["reason"] == "quarantined"
    assert alice_conn.connect_calls == []
    assert sm.get_session(bob_peer_id).state == SessionState.KNOWN   # still inert, not dialed


def test_failed_event_enters_quarantine(alice_conn, alice_peer_id, bob_peer_id):
    """A FAILED event (here a handshake timeout) puts the peer in quarantine like a LOST,
    so a just-failed peer is not immediately re-dialed."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sess = _mk_session(alice_conn, bob_peer_id, alice_peer_id)
    sm._set_active(bob_peer_id, sess)
    sess._set_state(SessionState.HELLO_SENT, 0.001)
    time.sleep(0.005)
    events = _run(sm.tick())
    assert any(e.type == SessionEventType.FAILED for e in events)
    assert sm._quarantine.get(bob_peer_id, 0.0) > time.monotonic() + 20.0


def test_failed_then_known_recreated_then_gated_then_dials(alice_conn, alice_peer_id, bob_peer_id):
    """Full cycle: a FAILED quarantines and drops the Session; the rendezvous still
    advertises the peer so the next sync recreates an inert KNOWN; activating it is
    refused until the window expires, then dials."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sess = _mk_session(alice_conn, bob_peer_id, alice_peer_id)
    sm._set_active(bob_peer_id, sess)
    sess._set_state(SessionState.HELLO_SENT, 0.001)
    time.sleep(0.005)
    _run(sm.tick())
    assert sm.get_session(bob_peer_id) is None                       # dropped on FAILED
    assert sm._quarantine.get(bob_peer_id, 0.0) > time.monotonic()   # quarantined

    sm.sync_known_peers({bob_peer_id: (["/ip4/1.1.1.1/tcp/1"], 1)})  # rendezvous still has bob
    assert sm.get_session(bob_peer_id).state == SessionState.KNOWN

    events = _run(sm.activate(bob_peer_id))                          # gated
    assert events[0].type == SessionEventType.FAILED and events[0].data["reason"] == "quarantined"
    assert alice_conn.connect_calls == []

    sm._quarantine[bob_peer_id] = time.monotonic() - 1.0            # window expired
    alice_conn.connect_returns = (bob_peer_id, False)
    _run(sm.activate(bob_peer_id))
    assert len(alice_conn.connect_calls) == 1                        # dials now


# Session-first teardown: settlement cuts + state-dispatched drop
#
# The transport follows the Session's terminal state: drop with a reason only
# GOAWAYs (the CLOSING machinery finally gets to complete via ACK), the deferred
# settlement cut severs the transport after a linger, and only the coercive paths
# (mid-handshake purge, sessionless/KNOWN, perimeter) disconnect inline.

def _established(sm, conn, peer_id, local_peer_id):
    """Plant a real Session in ESTABLISHED into the manager's active view."""
    sess = _mk_session(conn, peer_id, local_peer_id)
    sess._set_state(SessionState.ESTABLISHED, None)
    sm._set_active(peer_id, sess)
    return sess


def test_closing_completes_via_ack_with_transport_alive(alice_conn, alice_peer_id, bob_peer_id):
    """drop(reason) GOAWAYs and leaves the transport up, so the GOAWAY_ACK can come
    back and complete CLOSING via the ACK path (not the 10s timeout); the transport
    is severed only by the settlement cut once the linger expires."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sess = _established(sm, alice_conn, bob_peer_id, alice_peer_id)

    events = _run(sm.drop(bob_peer_id, WORLD_DISENGAGEMENT))
    assert events == []
    goaways = [s for s in alice_conn.sent if s[2].kind == SessMsg.KIND_GOAWAY]
    assert len(goaways) == 1
    assert alice_conn.disconnect_calls == []              # no cut under the GOAWAY
    assert sess.state == SessionState.CLOSING

    # The ACK lands on the still-alive transport -> CLOSED via ACK.
    events = _run(sm.deliver(bob_peer_id, SESSION_CONTROL_PROTOCOL,
                             SessMsg(SessMsg.KIND_GOAWAY_ACK, {"received": True})))
    assert [e.type for e in events] == [SessionEventType.CLOSED]
    assert "local_goaway_acked" in events[0].data["reason"]
    assert sm.get_session(bob_peer_id) is None            # settled: forgotten...
    assert bob_peer_id in sm._pending_cuts                # ...cut scheduled...
    assert alice_conn.disconnect_calls == []              # ...but not executed yet

    # The linger expires -> the executor severs the transport.
    _run(sm.tick(now=time.monotonic() + 3.0))
    assert alice_conn.disconnect_calls == [bob_peer_id]
    assert bob_peer_id not in sm._pending_cuts


def test_settlement_precedes_dispatch_on_remote_goaway(alice_conn, alice_peer_id, bob_peer_id):
    """By the time deliver returns the CLOSED of a remote GOAWAY, the SM has already
    forgotten the Session and scheduled the deferred cut: the purge coming back down
    from Node (drop with no reason) lands on the tearing-down branch and is a no-op,
    so it cannot abort the GOAWAY_ACK written moments earlier."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    _established(sm, alice_conn, bob_peer_id, alice_peer_id)

    events = _run(sm.deliver(bob_peer_id, SESSION_CONTROL_PROTOCOL,
                             SessMsg(SessMsg.KIND_GOAWAY,
                                     {"reason": WORLD_DISENGAGEMENT, "drain_window_ms": 0})))
    assert [e.type for e in events] == [SessionEventType.CLOSED]
    acks = [s for s in alice_conn.sent if s[2].kind == SessMsg.KIND_GOAWAY_ACK]
    assert len(acks) == 1
    # Settled BEFORE the events were returned.
    assert sm.get_session(bob_peer_id) is None
    assert bob_peer_id in sm._pending_cuts

    # Node's purge lands back here reasonless: must NOT disconnect under the ACK.
    _run(sm.drop(bob_peer_id))
    assert alice_conn.disconnect_calls == []


def test_pending_cut_cancelled_by_new_live_session(alice_conn, alice_peer_id, bob_peer_id):
    """The lazy guard: a due settlement cut is cancelled when a live (non-KNOWN)
    Session for the peer exists again — the re-handshake reuses the transport."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sm._pending_cuts[bob_peer_id] = time.monotonic() - 0.1   # already due
    hello = SessMsg(SessMsg.KIND_HELLO_REQ, {
        "session_id": b"\x02" * 16, "nonce_a": b"\xaa" * 32,
        "protocol_versions": ["1.0"], "client_id": "test",
    })
    _run(sm.deliver(bob_peer_id, SESSION_HELLO_PROTOCOL, hello))
    assert sm.get_session(bob_peer_id).state == SessionState.HELLO_ACK_SENT

    _run(sm.tick())
    assert alice_conn.disconnect_calls == []                 # cancelled, not executed
    assert bob_peer_id not in sm._pending_cuts


def test_pending_cut_not_cancelled_by_known_placeholder(alice_conn, alice_peer_id, bob_peer_id):
    """A KNOWN placeholder (rendezvous metadata, recreated every sync) is not a live
    relation: it must NOT cancel a due settlement cut, or a peer that stays on the
    rendezvous roster would keep its half-open transport forever."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sm._pending_cuts[bob_peer_id] = time.monotonic() - 0.1   # already due
    sm.ensure(bob_peer_id, ["/ip4/127.0.0.1/tcp/1"])         # KNOWN placeholder

    _run(sm.tick())
    assert alice_conn.disconnect_calls == [bob_peer_id]      # executed anyway
    assert bob_peer_id not in sm._pending_cuts


def test_drop_on_known_forgets_and_severs_without_goaway(alice_conn, alice_peer_id, bob_peer_id):
    """drop on a KNOWN placeholder sends no GOAWAY (nothing was ever handshaked) but
    severs any half-open transport inline: this is the only cure for a chat-only
    peer that never went past KNOWN."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sm.ensure(bob_peer_id, ["/ip4/127.0.0.1/tcp/1"])

    _run(sm.drop(bob_peer_id, WORLD_DISENGAGEMENT))
    assert alice_conn.sent == []                             # no GOAWAY
    assert sm.get_session(bob_peer_id) is None
    assert alice_conn.disconnect_calls == [bob_peer_id]


def test_drop_none_on_live_session_aborts_and_cuts(alice_conn, alice_peer_id, bob_peer_id):
    """A reasonless drop on a live Session is the silent coercive teardown: abort
    (no event), forget, disconnect inline — no zombie left to RESUME a relation the
    node already dismantled."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sess = _established(sm, alice_conn, bob_peer_id, alice_peer_id)

    events = _run(sm.drop(bob_peer_id))
    assert events == []
    assert sess.state == SessionState.FAILED                 # aborted silently
    assert sm.get_session(bob_peer_id) is None
    assert alice_conn.disconnect_calls == [bob_peer_id]
    goaways = [s for s in alice_conn.sent if s[2].kind == SessMsg.KIND_GOAWAY]
    assert goaways == []


def test_drop_mid_handshake_aborts_and_cuts(alice_conn, alice_peer_id, bob_peer_id):
    """A reasoned drop on a mid-handshake Session cannot GOAWAY (close() only covers
    ESTABLISHED/RECONNECTING): it must abort the handshake and cut, not leave it
    running to completion behind the purge."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sess = _mk_session(alice_conn, bob_peer_id, alice_peer_id)
    sess._set_state(SessionState.HELLO_SENT, 5.0)
    sm._set_active(bob_peer_id, sess)

    _run(sm.drop(bob_peer_id, WORLD_DISENGAGEMENT))
    assert sess.state == SessionState.FAILED
    assert sm.get_session(bob_peer_id) is None
    assert alice_conn.disconnect_calls == [bob_peer_id]


def test_failed_from_deliver_quarantines_and_schedules_cut(alice_conn, alice_peer_id, bob_peer_id):
    """Settlement is path-uniform: a FAILED produced on the deliver path (a rejected
    HELLO_ACK) quarantines and schedules the cut exactly like a tick-path FAILED.
    (The old cleanup exempted deliver-path FAILED from quarantine — deliberate
    change.) The deferred cut also closes the FAILED half-open leak: transport up,
    pools populated, no Session."""
    sm = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id)
    sess = _mk_session(alice_conn, bob_peer_id, alice_peer_id)
    sess._set_state(SessionState.HELLO_SENT, 5.0)
    sm._set_active(bob_peer_id, sess)

    events = _run(sm.deliver(bob_peer_id, SESSION_HELLO_PROTOCOL,
                             SessMsg(SessMsg.KIND_HELLO_RESP, {"reason_code": NOT_AUTHORIZED})))
    assert [e.type for e in events] == [SessionEventType.FAILED]
    assert bob_peer_id in sm._quarantine
    assert bob_peer_id in sm._pending_cuts
    assert sm.get_session(bob_peer_id) is None

    _run(sm.tick(now=time.monotonic() + 3.0))
    assert alice_conn.disconnect_calls == [bob_peer_id]
