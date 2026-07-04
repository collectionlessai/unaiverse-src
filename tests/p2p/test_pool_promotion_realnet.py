"""Real-libp2p pool promotion for Session-only peers (opt-in: ``-m p2p``).

A peer whose only traffic is the Session handshake/keep-alive must still
surface in ``get_connected_peers_info`` (Go friendlyPeers) on BOTH sides, so
the Python pools — and everything keyed on them: ``conn.send``, pool diffs,
``is_connected``, the transport-disconnect signals to the Session layer — work
without any legacy chat bootstrap message.

Run with:  ``uv run pytest -m p2p -k pool_promotion -v``
"""

import asyncio

import pytest

from unaiverse.networking.p2p.messages import Msg
from unaiverse.networking.session.sess_msg import GRACEFUL_SHUTDOWN
from unaiverse.networking.node.connpool import NodeConn

from realnet_helpers import pick_dialable_addr, pump_until

pytestmark = pytest.mark.p2p


def test_session_only_peers_enter_pools_and_pool_diffs_fire(real_nodes):
    """Session handshake traffic alone promotes both peers into friendlyPeers
    with the correct connection direction (dialer outbound, responder inbound),
    conn.update() routes them into OUT_PUBLIC/IN_PUBLIC so conn.send finds the
    peer, and after closing the session a transport disconnect surfaces as a
    removed pool diff instantly on the disconnecting side and within the Go
    grace window on the other.
    """
    # Two-instance nodes: conn.update() iterates both p2p instances like a real
    # Node does (the single-instance test builder aliases public and world onto
    # one P2P, which update() does not support).
    a_p2p, _, a_conn, a_sm, _ = real_nodes.world_node("alice")
    b_p2p, _, b_conn, b_sm, _ = real_nodes.world_node("bob")
    bob_addr = pick_dialable_addr(b_p2p)
    targets = [(a_conn, a_sm, NodeConn.P2P_PUBLIC), (b_conn, b_sm, NodeConn.P2P_PUBLIC)]

    async def _scenario():
        await asyncio.wait_for(a_sm.initiate(b_p2p.peer_id, [bob_addr]), timeout=10.0)
        await pump_until(targets,
                         predicate=lambda: a_sm.is_established(b_p2p.peer_id)
                         and b_sm.is_established(a_p2p.peer_id),
                         timeout=20.0)

        # Session traffic alone must have promoted both sides to friendlyPeers,
        # with the connection direction (not the per-message stream direction).
        a_view = {c["id"]: c for c in a_p2p.get_connected_peers_info()}
        b_view = {c["id"]: c for c in b_p2p.get_connected_peers_info()}
        assert b_p2p.peer_id in a_view, "dialer side never saw the peer (send path)"
        assert a_p2p.peer_id in b_view, "responder side never saw the peer (recv path)"
        assert a_view[b_p2p.peer_id]["direction"] == "outbound"
        assert b_view[a_p2p.peer_id]["direction"] == "inbound"

        # The pool update routes them, after which the chat send path works.
        added_a, _ = await a_conn.update()
        added_b, _ = await b_conn.update()
        assert b_p2p.peer_id in added_a.get(NodeConn.OUT_PUBLIC, set())
        assert a_p2p.peer_id in added_b.get(NodeConn.IN_PUBLIC, set())
        assert a_conn.is_connected(b_p2p.peer_id)
        assert b_conn.is_connected(a_p2p.peer_id)
        assert await a_conn.send(b_p2p.peer_id, None, Msg.MISC, {"hello": "pool"}), \
            "conn.send must find the peer's p2p once it is in the pools"

        # Tear the sessions down first: a live session's keep-alive would re-dial
        # the peer within the Go grace window and (correctly) cancel the cleanup,
        # masking the pool diff this test is about.
        await a_sm.close_session(b_p2p.peer_id, GRACEFUL_SHUTDOWN)
        await pump_until(targets,
                         predicate=lambda: a_sm.get_session(b_p2p.peer_id) is None
                         and b_sm.get_session(a_p2p.peer_id) is None,
                         timeout=10.0)

        # A real transport drop must surface as a "removed" pool diff: instantly
        # on the side that called disconnect, within the Go grace period (10s)
        # on the other side.
        await a_sm.drop(b_p2p.peer_id)
        _, removed_a = await a_conn.update()
        assert b_p2p.peer_id in removed_a.get(NodeConn.OUT_PUBLIC, set())

        removed_seen = False
        for _ in range(120):
            _, removed_b = await b_conn.update()
            if a_p2p.peer_id in removed_b.get(NodeConn.IN_PUBLIC, set()):
                removed_seen = True
                break
            await asyncio.sleep(0.25)
        assert removed_seen, (
            "peer drop never surfaced in the other side's pool diff; "
            f"friendly={[c['id'][:12] for c in (b_p2p.get_connected_peers_info() or [])]}")

    asyncio.run(_scenario())
