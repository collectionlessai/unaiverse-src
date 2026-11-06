"""Real-libp2p handshake + intentional-close tests (opt-in: ``-m p2p``).

These mirror the go-mocked ``tests/test_handshake_e2e.py`` happy-path and
intentional-close cases, but run over TWO real ``P2P`` instances in one process
talking over loopback/LAN TCP — exercising the real wire, real protobuf framing,
and real Ed25519 signing/verification that the mocked suite stubs out.

Run with:  ``uv run pytest -m p2p -v``  (needs a built ``unailib.*.so``).

Out of scope here (covered by the go-mocked suite — real transport can't
reproduce them cheaply/deterministically): simultaneous-open tie-break,
bad-signed-peer-record rejection, forced timeouts, liveness LOST/RESUMED.
"""

import asyncio

import pytest

from unaiverse.networking.session import SessionState, SessionEventType
from unaiverse.networking.session.sess_msg import WORLD_DISENGAGEMENT
from unaiverse.networking.node.connpool import NodeConn

from realnet_helpers import pick_dialable_addr, pump_until

pytestmark = pytest.mark.p2p


def test_real_handshake_happy_path(real_nodes):
    """alice (initiator) dials bob (responder) over real libp2p. Both must reach
    ESTABLISHED, hold each other's profile, and agree on the session_id."""
    a_p2p, a_conn, a_sm = real_nodes.node("alice")
    b_p2p, b_conn, b_sm = real_nodes.node("bob")
    bob_addr = pick_dialable_addr(b_p2p)

    async def _scenario():
        await asyncio.wait_for(a_sm.initiate(b_p2p.peer_id, [bob_addr]), timeout=10.0)
        await pump_until(
            [(a_conn, a_sm, NodeConn.P2P_PUBLIC), (b_conn, b_sm, NodeConn.P2P_PUBLIC)],
            predicate=lambda: a_sm.is_established(b_p2p.peer_id)
            and b_sm.is_established(a_p2p.peer_id),
            timeout=20.0,
        )

    asyncio.run(_scenario())

    a_sess = a_sm.get_session(b_p2p.peer_id)
    b_sess = b_sm.get_session(a_p2p.peer_id)
    assert a_sess.state == SessionState.ESTABLISHED
    assert b_sess.state == SessionState.ESTABLISHED
    # Same session_id on both ends, negotiated over the real wire.
    assert a_sess.session_id == b_sess.session_id
    # Each side holds the other's profile.
    assert a_sess.remote_profile["node_id"] == b_p2p.peer_id
    assert b_sess.remote_profile["node_id"] == a_p2p.peer_id


def test_real_intentional_close_reaches_peer(real_nodes):
    """After ESTABLISHED, alice closes with WORLD_DISENGAGEMENT. Bob must see a
    CLOSED event carrying that reason — and NOT a LOST event (a clean GOAWAY,
    not a dropped link)."""
    a_p2p, a_conn, a_sm = real_nodes.node("alice")
    b_p2p, b_conn, b_sm = real_nodes.node("bob")
    bob_addr = pick_dialable_addr(b_p2p)

    targets = [(a_conn, a_sm, NodeConn.P2P_PUBLIC), (b_conn, b_sm, NodeConn.P2P_PUBLIC)]
    captured = {}

    async def _scenario():
        nonlocal captured
        await asyncio.wait_for(a_sm.initiate(b_p2p.peer_id, [bob_addr]), timeout=10.0)
        await pump_until(targets,
                         predicate=lambda: a_sm.is_established(b_p2p.peer_id)
                         and b_sm.is_established(a_p2p.peer_id),
                         timeout=20.0)
        # Intentional close — alice emits a GOAWAY carrying the reason.
        await a_sm.close_session(b_p2p.peer_id, WORLD_DISENGAGEMENT)
        # Drain until bob's session leaves ESTABLISHED (it processed the GOAWAY).
        # pump_until returns every event it observed this run — the CLOSED event
        # is emitted in the same drain round, so it lands in the returned dict.
        captured = await pump_until(
            targets,
            predicate=lambda: not b_sm.is_established(a_p2p.peer_id),
            timeout=20.0)

    asyncio.run(_scenario())

    bob_events = captured.get(b_sm, [])
    closed = [e for e in bob_events if e.type == SessionEventType.CLOSED]
    assert len(closed) >= 1, f"bob never saw CLOSED; events={[e.type.value for e in bob_events]}"
    assert "WORLD_DISENGAGEMENT" in closed[0].data.get("reason", "")
    # Clean close, not a dropped link.
    assert not any(e.type == SessionEventType.LOST for e in bob_events)
