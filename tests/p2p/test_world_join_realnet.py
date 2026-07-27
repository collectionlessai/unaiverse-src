"""Real-libp2p world-join handoff (opt-in: ``-m p2p``).

Real-transport port of the go-mocked ``tests/test_world_join_e2e.py``. Topology
— FOUR real ``P2P`` instances in one process (two per node):

    joiner.public  ─┐   ┌─  host.public   (world host: issues WorldGrant)
                    │   │
    joiner.world   ─┘   └─  host.world     (world host: redeems the grant)

Flow:
  1. joiner.public dials host.public (is_world_join) → real HELLO/PROFILE.
  2. host.public decision hook mints a WorldGrant; the grant is signed with the
     host's **world** key (cross-instance, real Ed25519) → joiner.public
     ESTABLISHED carries the grant blob.
  3. orchestrator (mirrors node._dispatch_session_event) opens joiner.world with
     the grant → real HELLO/PROFILE → both world sides ESTABLISHED (no PRIVATE_JOIN).
  4. orchestrator GOAWAYs joiner.public (WORLD_HANDOFF_COMPLETE) → host.public CLOSED.

This is the highest-value real-net check: the WorldGrant is signed/verified with
the world-side key across two P2P instances, with *real* crypto — validating the
``instanceIndex``-keyed FFI that the mocked suite can only stub.

The world-side dial address is handed to the orchestrator via closure (the test
owns both nodes). The grant's SignedPeerRecord was dropped, so the binding to
the world identity now comes from the verified token; the full real PROFILE
handshake still runs.

Run with:  ``uv run pytest -m p2p tests/p2p/test_world_join_realnet.py -v``
"""

import json
import asyncio

import pytest

from unaiverse.networking.session import (
    SessionState, SessionEventType, ProfileDecisionResult, WorldGrantSpec,
)
from unaiverse.networking.session.sess_msg import WORLD_HANDOFF_COMPLETE
from unaiverse.networking.node.connpool import NodeConn

from realnet_helpers import pick_dialable_addr, pump_until

pytestmark = pytest.mark.p2p


def _host_decision_hook(peer_id, remote_node_id, remote_profile, p2p_name, verified_claims=None):
    """Host accepts on both instances; mints a WorldGrant on the public side and
    accepts on the world side. (Same shape as the go-mocked test.)"""
    async def _hook():
        if p2p_name == NodeConn.P2P_WORLD:
            return ProfileDecisionResult(accepted=True)
        spec = WorldGrantSpec(
            role=3,
            world_profile_bytes=json.dumps({"node_id": "world-node"}).encode(),
            rendezvous_tag=42,
            code_bundle=json.dumps({"agent_files": None}).encode(),
            initial_stats=b"",
        )
        return ProfileDecisionResult(accepted=True, world_grant=spec)
    return _hook()


def test_real_world_join_full_handoff(real_nodes):
    """Over four real P2P instances, a public world-join mints a WorldGrant that the
    joiner redeems on the world side: both world Sessions reach ESTABLISHED, the public
    session is handed off (joiner closes it, host sees CLOSED carrying WORLD_HANDOFF_COMPLETE),
    and the joiner's world Session establishes as initiator via the normal PROFILE handshake
    with no PRIVATE_JOIN."""
    j_pub_p2p, j_world_p2p, j_conn, j_pub, j_world = real_nodes.world_node("joiner")
    h_pub_p2p, h_world_p2p, h_conn, h_pub, h_world = real_nodes.world_node(
        "host", decision_hook=_host_decision_hook)

    host_pub_addr = pick_dialable_addr(h_pub_p2p)
    host_world_addr = pick_dialable_addr(h_world_p2p)
    host_pub_id = h_pub_p2p.peer_id
    host_world_id = h_world_p2p.peer_id
    joiner_world_id = j_world_p2p.peer_id

    targets = [
        (j_conn, j_pub, NodeConn.P2P_PUBLIC), (j_conn, j_world, NodeConn.P2P_WORLD),
        (h_conn, h_pub, NodeConn.P2P_PUBLIC), (h_conn, h_world, NodeConn.P2P_WORLD),
    ]
    established = []
    closed = []

    async def orchestrate(ev):
        # Mirrors node._dispatch_session_event handoff logic, async.
        if ev.type == SessionEventType.ESTABLISHED:
            established.append(ev)
            grant = ev.data.get("world_grant")
            if grant is not None:
                # joiner.public got the grant → open joiner.world to redeem it.
                blob = dict(grant)
                evs = await j_world.initiate(
                    host_world_id, [host_world_addr],
                    grant=blob)
                for e in evs:
                    await orchestrate(e)
            elif ev.p2p_name == NodeConn.P2P_WORLD and ev.role.value == "initiator":
                # joiner.world established → GOAWAY the public session.
                    evs = await j_pub.close_session(host_pub_id, WORLD_HANDOFF_COMPLETE)
                    for e in evs:
                        await orchestrate(e)
        elif ev.type == SessionEventType.CLOSED:
            closed.append(ev)

    async def _scenario():
        evs = await j_pub.initiate(host_pub_id, [host_pub_addr], is_world_join=True)
        for e in evs:
            await orchestrate(e)
        await pump_until(
            targets,
            predicate=lambda: (
                j_world.is_established(host_world_id)
                and h_world.is_established(joiner_world_id)
                and any(e.p2p_name == NodeConn.P2P_PUBLIC for e in closed)),
            on_event=orchestrate,
            timeout=30.0,
        )

    asyncio.run(_scenario())

    # Both world sides ESTABLISHED over the real wire.
    assert j_world.is_established(host_world_id), "joiner world session not established"
    assert h_world.is_established(joiner_world_id), "host world session not established"

    # The public session handed off: joiner closed it; host saw CLOSED w/ the reason.
    j_pub_sess = j_pub.get_session(host_pub_id)
    assert j_pub_sess is None or j_pub_sess.state in (SessionState.CLOSING, SessionState.CLOSED)
    public_closed = [e for e in closed if e.p2p_name == NodeConn.P2P_PUBLIC]
    assert any("WORLD_HANDOFF_COMPLETE" in e.data.get("reason", "") for e in public_closed), \
        f"no public CLOSED with WORLD_HANDOFF_COMPLETE; closed={[(e.p2p_name, e.data.get('reason')) for e in closed]}"

    # The joiner's world Session established as initiator. The join completes on the normal
    # PROFILE handshake (no PRIVATE_JOIN, no rendezvous_topic in the event): the rendezvous_tag
    # travels inside the world_grant on the PUBLIC ESTABLISHED, not in the world one. Kept in
    # lockstep with test_world_join_e2e.
    assert any(e.p2p_name == NodeConn.P2P_WORLD and e.role.value == "initiator"
               for e in established)


# The world-grant signature was removed: the role is no longer carried/trusted in
# the grant (the host reads it from its own pre-admission), so a tampered-role grant
# simply has no effect and there is nothing to reject. The former
# test_real_world_join_rejects_tampered_role was dropped with that behaviour.
