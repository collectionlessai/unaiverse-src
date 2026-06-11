"""End-to-end reconnection and keep-alive tests — two SessionManagers wired
through the fake bidirectional transport, driven through transport drops,
re-dials, RESUME exchanges and session-level PING/PONG.

These cover the recovery paths that the unit FSM tests only exercise one side
at a time: RESUME round-trips, the RESUME-NACK → full re-handshake fallback,
grace expiry → LOST + quarantine, and the manager-level SESSION_UNKNOWN
replies a restarted peer sends on behalf of sessions it no longer holds.

Every test runs as a single coroutine under one asyncio.run so the
fire-and-forget tasks spawned by tick_reconnect / tick_keepalive can complete
(the one-run-per-call pump pattern would tear the loop down under them).
"""

import time
import asyncio

import pytest

from unaiverse.networking.session import (
    SessionManager, SessionState, SessionEventType,
)
from unaiverse.networking.session.sess_msg import WORLD_DISENGAGEMENT
from unaiverse.networking.node.connpool import NodeConn

from conftest import apump, establish_pair


def _make_sm(conn, local_peer_id, node_id):
    return SessionManager(
        conn=conn,
        p2p_name=NodeConn.P2P_PUBLIC,
        local_peer_id=local_peer_id,
        local_node_id=node_id,
        local_profile_factory=lambda: {"node_id": node_id, "ver": "test"},
    )


def test_transport_drop_then_resume(bidirectional_fakeconns, alice_peer_id, bob_peer_id):
    """After a transport drop on both sides, the initiator re-dials, sends
    RESUME, the responder (still holding the session) ACKs, and both sides are
    back to ESTABLISHED with the same session_id."""
    alice_conn, bob_conn, alice_inbox, bob_inbox = bidirectional_fakeconns
    alice_conn.connect_returns = (bob_peer_id, False)

    async def _scenario():
        sm_alice = _make_sm(alice_conn, alice_peer_id, "alice")
        sm_bob = _make_sm(bob_conn, bob_peer_id, "bob")
        await establish_pair(sm_alice, sm_bob, alice_inbox, bob_inbox, bob_peer_id)
        old_sid = sm_alice.get_session(bob_peer_id).session_id

        ev_a = await sm_alice.on_peer_disconnected(bob_peer_id)
        ev_b = await sm_bob.on_peer_disconnected(alice_peer_id)
        assert [e.type for e in ev_a] == [SessionEventType.RECONNECTING]
        assert [e.type for e in ev_b] == [SessionEventType.RECONNECTING]

        # Alice's tick triggers the fire-and-forget re-dial + RESUME.
        await sm_alice.tick()
        await asyncio.sleep(0.05)
        assert sm_alice.get_session(bob_peer_id)._reconnect_substep == "resume_sent"

        # Bob holds the session (RECONNECTING) → ACK + RESUMED on his side.
        bob_events = await apump(bob_inbox, sm_bob)
        assert [e.type for e in bob_events] == [SessionEventType.RESUMED]
        # Alice processes the ACK → RESUMED on her side.
        alice_events = await apump(alice_inbox, sm_alice)
        assert [e.type for e in alice_events] == [SessionEventType.RESUMED]

        a_sess = sm_alice.get_session(bob_peer_id)
        b_sess = sm_bob.get_session(alice_peer_id)
        assert a_sess.state == SessionState.ESTABLISHED
        assert b_sess.state == SessionState.ESTABLISHED
        assert a_sess.session_id == old_sid == b_sess.session_id

    asyncio.run(_scenario())


def test_resume_nack_triggers_full_rehandshake(bidirectional_fakeconns,
                                               alice_peer_id, bob_peer_id):
    """If the peer restarted (fresh SessionManager, no session), the RESUME is
    NACKed with SESSION_UNKNOWN and the initiator re-runs the full handshake
    with a rotated session_id, ending ESTABLISHED on both sides."""
    alice_conn, bob_conn, alice_inbox, bob_inbox = bidirectional_fakeconns
    alice_conn.connect_returns = (bob_peer_id, False)

    async def _scenario():
        sm_alice = _make_sm(alice_conn, alice_peer_id, "alice")
        sm_bob = _make_sm(bob_conn, bob_peer_id, "bob")
        await establish_pair(sm_alice, sm_bob, alice_inbox, bob_inbox, bob_peer_id)
        old_sid = sm_alice.get_session(bob_peer_id).session_id

        # Bob "restarts": a fresh manager on the same conn, empty registry.
        sm_bob = _make_sm(bob_conn, bob_peer_id, "bob")

        await sm_alice.on_peer_disconnected(bob_peer_id)
        await sm_alice.tick()
        await asyncio.sleep(0.05)

        # Bob's manager holds no session → RESUME_ACK(SESSION_UNKNOWN).
        assert await apump(bob_inbox, sm_bob) == []
        # Alice falls back to a fresh handshake (re-HELLO, rotated session_id).
        assert await apump(alice_inbox, sm_alice) == []
        a_sess = sm_alice.get_session(bob_peer_id)
        assert a_sess.state == SessionState.HELLO_SENT
        assert a_sess.session_id != old_sid

        # Complete the new handshake.
        await apump(bob_inbox, sm_bob)
        await apump(alice_inbox, sm_alice)
        bob_events = await apump(bob_inbox, sm_bob)
        alice_events = await apump(alice_inbox, sm_alice)
        assert [e.type for e in bob_events] == [SessionEventType.ESTABLISHED]
        assert [e.type for e in alice_events] == [SessionEventType.ESTABLISHED]
        assert sm_alice.get_session(bob_peer_id).session_id == \
               sm_bob.get_session(alice_peer_id).session_id
        assert sm_alice.get_session(bob_peer_id).remote_profile == {"node_id": "bob", "ver": "test"}

    asyncio.run(_scenario())


def test_redial_failure_backoff_then_grace_expiry_lost_and_quarantine(
        bidirectional_fakeconns, alice_peer_id, bob_peer_id):
    """While the peer stays unreachable, re-dials back off behind the
    _next_redial_at gate; when the grace window expires the session is LOST,
    evicted from the registry, and the peer is quarantined so an immediate
    re-initiate fails without dialing."""
    alice_conn, bob_conn, alice_inbox, bob_inbox = bidirectional_fakeconns
    alice_conn.connect_returns = (bob_peer_id, False)

    async def _scenario():
        sm_alice = _make_sm(alice_conn, alice_peer_id, "alice")
        sm_bob = _make_sm(bob_conn, bob_peer_id, "bob")
        await establish_pair(sm_alice, sm_bob, alice_inbox, bob_inbox, bob_peer_id)
        dials_after_handshake = len(alice_conn.connect_calls)

        alice_conn.connect_returns = (None, False)
        await sm_alice.on_peer_disconnected(bob_peer_id)
        sess = sm_alice.get_session(bob_peer_id)

        # First tick: a re-dial fires and fails → backoff gate armed.
        await sm_alice.tick()
        await asyncio.sleep(0.05)
        assert len(alice_conn.connect_calls) == dials_after_handshake + 1
        assert sess._reconnect_attempts == 1
        assert sess._next_redial_at > time.monotonic()

        # Second tick right away: gated by _next_redial_at, no new dial.
        await sm_alice.tick()
        await asyncio.sleep(0.05)
        assert len(alice_conn.connect_calls) == dials_after_handshake + 1

        # Grace expiry: check_timeout fires LOST before tick_reconnect can run,
        # the session is evicted and the peer is quarantined.
        events = await sm_alice.tick(time.monotonic() + 46.0)
        assert [e.type for e in events] == [SessionEventType.LOST]
        assert sm_alice.get_session(bob_peer_id) is None
        assert bob_peer_id in sm_alice._quarantine

        # A re-initiate during quarantine fails immediately, without dialing.
        events = await sm_alice.initiate(bob_peer_id, ["/ip4/127.0.0.1/tcp/0"])
        assert [e.type for e in events] == [SessionEventType.FAILED]
        assert events[0].data["reason"] == "quarantined"
        assert len(alice_conn.connect_calls) == dials_after_handshake + 1

    asyncio.run(_scenario())


def test_keepalive_ping_pong_roundtrip(bidirectional_fakeconns, alice_peer_id, bob_peer_id):
    """A due keep-alive PING reaches the peer, the peer PONGs ok=true, and the
    initiator clears its in-flight state without leaving ESTABLISHED."""
    alice_conn, bob_conn, alice_inbox, bob_inbox = bidirectional_fakeconns
    alice_conn.connect_returns = (bob_peer_id, False)

    async def _scenario():
        sm_alice = _make_sm(alice_conn, alice_peer_id, "alice")
        sm_bob = _make_sm(bob_conn, bob_peer_id, "bob")
        await establish_pair(sm_alice, sm_bob, alice_inbox, bob_inbox, bob_peer_id)

        a_sess = sm_alice.get_session(bob_peer_id)
        a_sess._last_ping_at -= 16.0   # past the 15s keep-alive period
        await sm_alice.tick()
        await asyncio.sleep(0.05)      # let the fire-and-forget PING send run
        assert a_sess._ping_in_flight is True

        assert await apump(bob_inbox, sm_bob) == []     # PING → PONG ok=true
        assert await apump(alice_inbox, sm_alice) == []  # PONG absorbed
        assert a_sess._ping_in_flight is False
        assert a_sess._consecutive_ping_failures == 0
        assert a_sess.state == SessionState.ESTABLISHED

    asyncio.run(_scenario())


def test_keepalive_pong_not_ok_from_restarted_peer(bidirectional_fakeconns,
                                                   alice_peer_id, bob_peer_id):
    """A restarted peer answers our PING with the manager-level PONG ok=false
    (it holds no session) — that is a hard liveness failure → RECONNECTING."""
    alice_conn, bob_conn, alice_inbox, bob_inbox = bidirectional_fakeconns
    alice_conn.connect_returns = (bob_peer_id, False)

    async def _scenario():
        sm_alice = _make_sm(alice_conn, alice_peer_id, "alice")
        sm_bob = _make_sm(bob_conn, bob_peer_id, "bob")
        await establish_pair(sm_alice, sm_bob, alice_inbox, bob_inbox, bob_peer_id)

        sm_bob = _make_sm(bob_conn, bob_peer_id, "bob")  # Bob restarts

        a_sess = sm_alice.get_session(bob_peer_id)
        a_sess._last_ping_at -= 16.0
        await sm_alice.tick()
        await asyncio.sleep(0.05)

        assert await apump(bob_inbox, sm_bob) == []      # manager replies ok=false
        alice_events = await apump(alice_inbox, sm_alice)
        assert [e.type for e in alice_events] == [SessionEventType.RECONNECTING]
        assert a_sess.state == SessionState.RECONNECTING

    asyncio.run(_scenario())


def test_terminal_session_is_evicted_and_inbound_rehandshake_works(
        bidirectional_fakeconns, alice_peer_id, bob_peer_id):
    """After an intentional close, the responder side must evict the CLOSED
    session so a later inbound HELLO from the same peer can re-handshake."""
    alice_conn, bob_conn, alice_inbox, bob_inbox = bidirectional_fakeconns
    alice_conn.connect_returns = (bob_peer_id, False)

    async def _scenario():
        sm_alice = _make_sm(alice_conn, alice_peer_id, "alice")
        sm_bob = _make_sm(bob_conn, bob_peer_id, "bob")
        await establish_pair(sm_alice, sm_bob, alice_inbox, bob_inbox, bob_peer_id)

        # Alice closes intentionally; Bob sees CLOSED via deliver.
        await sm_alice.close_session(bob_peer_id, WORLD_DISENGAGEMENT)
        bob_events = await apump(bob_inbox, sm_bob)
        assert [e.type for e in bob_events] == [SessionEventType.CLOSED]
        alice_events = await apump(alice_inbox, sm_alice)   # GOAWAY_ACK → CLOSED
        assert [e.type for e in alice_events] == [SessionEventType.CLOSED]

        # The terminal session must not linger in Bob's registry.
        assert sm_bob.get_session(alice_peer_id) is None

        # Alice re-initiates: Bob must answer the fresh HELLO and the pair must
        # reach ESTABLISHED again.
        init_events, resp_events = await establish_pair(
            sm_alice, sm_bob, alice_inbox, bob_inbox, bob_peer_id)
        assert [e.type for e in init_events] == [SessionEventType.ESTABLISHED]
        assert [e.type for e in resp_events] == [SessionEventType.ESTABLISHED]

    asyncio.run(_scenario())


def test_has_active_sessions_tracks_lifecycle(bidirectional_fakeconns,
                                              alice_peer_id, bob_peer_id):
    """has_active_sessions is the Node's "am I idle?" probe (it gates the clock
    slowdown): True from ESTABLISHED through RECONNECTING, False once closed."""
    alice_conn, bob_conn, alice_inbox, bob_inbox = bidirectional_fakeconns
    alice_conn.connect_returns = (bob_peer_id, False)

    async def _scenario():
        sm_alice = _make_sm(alice_conn, alice_peer_id, "alice")
        sm_bob = _make_sm(bob_conn, bob_peer_id, "bob")
        assert sm_alice.has_active_sessions() is False

        await establish_pair(sm_alice, sm_bob, alice_inbox, bob_inbox, bob_peer_id)
        assert sm_alice.has_active_sessions() is True
        assert sm_bob.has_active_sessions() is True

        # A reconnect grace window still counts as active (keep-alive must keep
        # running at full pump speed to drive the RESUME).
        await sm_alice.on_peer_disconnected(bob_peer_id)
        assert sm_alice.get_session(bob_peer_id).state == SessionState.RECONNECTING
        assert sm_alice.has_active_sessions() is True

        # An intentional close on the other pair member drops it to inactive.
        await sm_bob.close_session(alice_peer_id, WORLD_DISENGAGEMENT)
        assert sm_bob.has_active_sessions() is False

    asyncio.run(_scenario())
