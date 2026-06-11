"""Real-libp2p reconnection tests (opt-in: ``-m p2p``).

These cover the two recovery paths the go-mocked suite can only simulate:

  * a real transport drop followed by a real re-dial + RESUME (session
    survives, same session_id);
  * a real peer restart (same identity/port, fresh SessionManager) followed by
    RESUME → SESSION_UNKNOWN → full re-handshake (new session_id).

Run with:  ``uv run pytest -m p2p -v``  (needs a built ``unailib.*.so``).
"""

import socket
import asyncio

import pytest

from unaiverse.networking.session import SessionState, SessionEventType
from unaiverse.networking.node.connpool import ConnectionPools, NodeConn

from realnet_helpers import pick_dialable_addr, pump_until

pytestmark = pytest.mark.p2p


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        return s.getsockname()[1]


def test_real_transport_drop_then_resume(real_nodes):
    """alice and bob establish, the transport is dropped for real, and the
    initiator-driven re-dial + RESUME brings the SAME session back."""
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
        old_sid = a_sm.get_session(b_p2p.peer_id).session_id

        # Drop the link for real, then feed both sides the disconnect signal the
        # Node loop would derive from its pool diff.
        await ConnectionPools.disconnect(a_p2p, b_p2p.peer_id)
        await a_sm.on_peer_disconnected(b_p2p.peer_id)
        await b_sm.on_peer_disconnected(a_p2p.peer_id)
        assert a_sm.get_session(b_p2p.peer_id).state == SessionState.RECONNECTING

        # The tick inside pump_until drives the re-dial + RESUME over the wire.
        captured = await pump_until(
            targets,
            predicate=lambda: a_sm.is_established(b_p2p.peer_id)
            and b_sm.is_established(a_p2p.peer_id),
            timeout=20.0)
        return old_sid

    old_sid = asyncio.run(_scenario())

    a_events = captured.get(a_sm, [])
    assert any(e.type == SessionEventType.RESUMED for e in a_events), \
        f"alice never saw RESUMED; events={[e.type.value for e in a_events]}"
    # Same session survived — no re-handshake happened.
    assert a_sm.get_session(b_p2p.peer_id).session_id == old_sid
    assert b_sm.get_session(a_p2p.peer_id).session_id == old_sid


def test_real_peer_restart_resume_nack_rehandshake(real_nodes):
    """bob restarts (same identity and port, fresh SessionManager): alice's
    RESUME gets SESSION_UNKNOWN and she re-runs the full handshake, ending
    ESTABLISHED with a NEW session_id."""
    bob_port = _free_tcp_port()
    a_p2p, a_conn, a_sm = real_nodes.node("alice")
    b_p2p, b_conn, b_sm = real_nodes.node("bob", port=bob_port)
    bob_addr = pick_dialable_addr(b_p2p)

    async def _establish():
        await asyncio.wait_for(a_sm.initiate(b_p2p.peer_id, [bob_addr]), timeout=10.0)
        await pump_until(
            [(a_conn, a_sm, NodeConn.P2P_PUBLIC), (b_conn, b_sm, NodeConn.P2P_PUBLIC)],
            predicate=lambda: a_sm.is_established(b_p2p.peer_id)
            and b_sm.is_established(a_p2p.peer_id),
            timeout=20.0)

    asyncio.run(_establish())
    old_sid = a_sm.get_session(b_p2p.peer_id).session_id
    old_bob_peer_id = b_p2p.peer_id

    # Restart bob: close the instance, rebuild on the same identity dir + port.
    b_p2p.close()
    b_p2p, b_conn, b_sm = real_nodes.node("bob", port=bob_port)
    assert b_p2p.peer_id == old_bob_peer_id, "restart must preserve the peer identity"

    async def _recover():
        await a_sm.on_peer_disconnected(old_bob_peer_id)
        await pump_until(
            [(a_conn, a_sm, NodeConn.P2P_PUBLIC), (b_conn, b_sm, NodeConn.P2P_PUBLIC)],
            predicate=lambda: a_sm.is_established(old_bob_peer_id)
            and b_sm.is_established(a_p2p.peer_id),
            timeout=30.0)

    asyncio.run(_recover())

    a_sess = a_sm.get_session(old_bob_peer_id)
    b_sess = b_sm.get_session(a_p2p.peer_id)
    assert a_sess.state == SessionState.ESTABLISHED
    assert b_sess.state == SessionState.ESTABLISHED
    # The restarted peer forgot the session: identity must have rotated.
    assert a_sess.session_id != old_sid
    assert a_sess.session_id == b_sess.session_id
