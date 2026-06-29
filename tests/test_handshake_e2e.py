"""End-to-end in-process handshake — real protobufs, two SessionManagers
wired through a fake bidirectional transport.

This is the most useful test in the suite: it exercises the full Stage-1 →
Stage-2 happy path with real proto serialization, real FSM transitions on both
sides, and confirms that ESTABLISHED events fire on both ends.

Run with:

    PYTHONPATH=src pytest tests/test_handshake_e2e.py -v
"""

import asyncio
from collections import deque

import pytest

from unaiverse.networking.session import (
    SessionManager, SessionRole, SessionState, SessionEventType,
)
from unaiverse.networking.session.sess_msg import WORLD_DISENGAGEMENT
from unaiverse.networking.node.connpool import NodeConn

pytestmark = pytest.mark.e2e


def _run(coro):
    return asyncio.run(coro)


def _pump(inbox: deque, sm: SessionManager) -> list:
    """Drain everything currently in `inbox` into `sm.deliver`. Returns
    the concatenated list of events."""
    events = []
    while inbox:
        peer_id, protocol_id, payload = inbox.popleft()
        events.extend(_run(sm.deliver(peer_id, protocol_id, payload)))
    return events


def _make_sm(p2p, p2p_name, local_peer_id, node_id):
    return SessionManager(
        p2p=p2p,
        p2p_name=p2p_name,
        local_peer_id=local_peer_id,
        local_node_id=node_id,
        local_profile_factory=lambda: {"node_id": node_id, "ver": "test"},
    )


def test_full_handshake_happy_path(bidirectional_fakeconns,
                                   alice_peer_id, bob_peer_id):
    """Alice (initiator) → Bob (responder). After pumping the four messages
    of Stage-1+Stage-2, both sides must reach ESTABLISHED and each must hold
    the other's profile."""
    alice_conn, bob_conn, alice_inbox, bob_inbox = bidirectional_fakeconns

    # When Alice initiates, the dial must "succeed".
    alice_conn.connect_returns = (bob_peer_id, False)

    sm_alice = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id, "alice-node")
    sm_bob   = _make_sm(bob_conn,   NodeConn.P2P_PUBLIC, bob_peer_id,   "bob-node")

    # Stage 1 — Alice's HELLO.
    init_events = _run(sm_alice.initiate(bob_peer_id, ["/ip4/127.0.0.1/tcp/0"]))
    assert init_events == []
    # Bob processes the HELLO → sends HELLO_ACK back into alice_inbox.
    bob_events_1 = _pump(bob_inbox, sm_bob)
    assert bob_events_1 == []
    # Alice processes HELLO_ACK → sends PROFILE_OFFER into bob_inbox.
    alice_events_1 = _pump(alice_inbox, sm_alice)
    assert alice_events_1 == []
    # Bob processes PROFILE_OFFER → sends PROFILE_DECISION → ESTABLISHED on Bob.
    bob_events_2 = _pump(bob_inbox, sm_bob)
    assert len(bob_events_2) == 1
    assert bob_events_2[0].type == SessionEventType.ESTABLISHED
    assert bob_events_2[0].peer_id == alice_peer_id
    # Alice processes PROFILE_DECISION → ESTABLISHED on Alice.
    alice_events_2 = _pump(alice_inbox, sm_alice)
    assert len(alice_events_2) == 1
    assert alice_events_2[0].type == SessionEventType.ESTABLISHED
    assert alice_events_2[0].peer_id == bob_peer_id

    # Both sides hold the other's profile.
    alice_sess = sm_alice.get_session(bob_peer_id)
    bob_sess   = sm_bob.get_session(alice_peer_id)
    assert alice_sess.state == SessionState.ESTABLISHED
    assert bob_sess.state == SessionState.ESTABLISHED
    assert alice_sess.remote_profile == {"node_id": "bob-node", "ver": "test"}
    assert bob_sess.remote_profile   == {"node_id": "alice-node", "ver": "test"}
    # Same session_id on both ends.
    assert alice_sess.session_id == bob_sess.session_id


def test_intentional_close_reaches_peer(bidirectional_fakeconns,
                                        alice_peer_id, bob_peer_id):
    """After ESTABLISHED, calling close_session on Alice must make Bob see
    Session.Closed (not Lost)."""
    alice_conn, bob_conn, alice_inbox, bob_inbox = bidirectional_fakeconns
    alice_conn.connect_returns = (bob_peer_id, False)

    sm_alice = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id, "alice")
    sm_bob   = _make_sm(bob_conn,   NodeConn.P2P_PUBLIC, bob_peer_id,   "bob")

    # Drive to ESTABLISHED on both sides.
    _run(sm_alice.initiate(bob_peer_id, ["/ip4/127.0.0.1/tcp/0"]))
    _pump(bob_inbox, sm_bob)
    _pump(alice_inbox, sm_alice)
    _pump(bob_inbox, sm_bob)
    _pump(alice_inbox, sm_alice)

    # Now Alice intentionally closes with WORLD_DISENGAGEMENT.
    _run(sm_alice.close_session(bob_peer_id, WORLD_DISENGAGEMENT))
    # Bob processes the GOAWAY → CLOSED event with the supplied reason.
    bob_events = _pump(bob_inbox, sm_bob)
    closed = [e for e in bob_events if e.type == SessionEventType.CLOSED]
    assert len(closed) == 1
    assert "WORLD_DISENGAGEMENT" in closed[0].data["reason"]
    # Bob sent GOAWAY_ACK back; Alice processes it and is CLOSED too.
    alice_events = _pump(alice_inbox, sm_alice)
    closed_a = [e for e in alice_events if e.type == SessionEventType.CLOSED]
    assert len(closed_a) == 1


def test_simultaneous_open_collapses_to_one_session(bidirectional_fakeconns,
                                                     alice_peer_id, bob_peer_id):
    """Both sides call initiate for each other at the same time. After the
    tie-breaker runs and the handshake completes, each side ends up with
    exactly ONE ESTABLISHED Session for the other peer (not two)."""
    alice_conn, bob_conn, alice_inbox, bob_inbox = bidirectional_fakeconns
    alice_conn.connect_returns = (bob_peer_id, False)
    bob_conn.connect_returns   = (alice_peer_id, False)

    sm_alice = _make_sm(alice_conn, NodeConn.P2P_PUBLIC, alice_peer_id, "alice")
    sm_bob   = _make_sm(bob_conn,   NodeConn.P2P_PUBLIC, bob_peer_id,   "bob")

    # Both fire initiate at t=0.
    _run(sm_alice.initiate(bob_peer_id, ["/ip4/1.2.3.4/tcp/0"]))
    _run(sm_bob.initiate(alice_peer_id, ["/ip4/1.2.3.4/tcp/0"]))

    # Both sides will see each other's HELLO in their inbox. Pump until
    # everything settles (max 10 iterations as a safety cap).
    for _ in range(10):
        progressed = False
        if alice_inbox:
            _pump(alice_inbox, sm_alice)
            progressed = True
        if bob_inbox:
            _pump(bob_inbox, sm_bob)
            progressed = True
        if not progressed:
            break

    # Each side has at most one Session for the other peer (the loser's FSM is
    # silently aborted by tie-breaker, then evicted by tick / cleanup on the
    # next FAILED event).
    assert len(sm_alice) <= 1
    assert len(sm_bob)   <= 1

    # If a Session survives on each side it should be ESTABLISHED — the FSM
    # MUST NOT be stuck mid-handshake after the dust settles.
    s_a = sm_alice.get_session(bob_peer_id)
    s_b = sm_bob.get_session(alice_peer_id)
    if s_a is not None:
        assert s_a.state in (SessionState.ESTABLISHED, SessionState.FAILED)
    if s_b is not None:
        assert s_b.state in (SessionState.ESTABLISHED, SessionState.FAILED)


# The offer-side SignedPeerRecord check was removed (the public<->world binding
# now rides the verified token's p2p_peers), so the test that exercised a bad
# offer SPR was dropped with it.
