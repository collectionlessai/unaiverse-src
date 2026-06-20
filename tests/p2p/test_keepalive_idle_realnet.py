"""Real-libp2p idle keep-alive tests (opt-in: ``-m p2p``).

Reproduce the field failure where an ESTABLISHED session with no application
traffic loses its PING/PONG exchange after the first keep-alive period and
collapses into RECONNECTING → FAILED on both sides. A healthy stack must keep
an idle session ESTABLISHED across multiple keep-alive cycles. Covered both on
a plain public link (TCP and QUIC) and on the world link right after a full
world-join handoff (the exact field topology).

Run with:  ``uv run pytest -m p2p -k idle_keepalive -v``  (takes ~3 minutes).
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

# Two full keep-alive periods (15s each) plus RTT/tick margin: enough for two
# PING/PONG round-trips per side, and enough for the failure mode to surface
# (miss threshold 2 → RECONNECTING at ~35s when pings are lost).
IDLE_S = 42.0


def _dialable_addr(p2p, kind: str) -> str | None:
    """Non-loopback dialable multiaddr of the requested transport, or None. The
    chat example dials the world over the profile addresses, where
    NodeConn.__connect gives UDP/QUIC priority — so QUIC must be exercised too,
    not just TCP."""
    cands = [a for a in p2p.addresses if "127.0.0.1" not in a and "/p2p/" in a]
    if kind == "tcp":
        cands = [a for a in cands if "/tcp/" in a and "/ws/" not in a]
    else:
        cands = [a for a in cands if "/quic-v1" in a and "webtransport" not in a]
    return cands[0] if cands else None


@pytest.mark.parametrize("transport", ["tcp", "quic"])
def test_real_idle_keepalive_stays_established(real_nodes, transport):
    a_p2p, a_conn, a_sm = real_nodes.node("alice")
    b_p2p, b_conn, b_sm = real_nodes.node("bob")
    bob_addr = _dialable_addr(b_p2p, transport)
    if bob_addr is None:
        pytest.skip(f"no non-loopback {transport} address available")
    targets = [(a_conn, a_sm, NodeConn.P2P_PUBLIC), (b_conn, b_sm, NodeConn.P2P_PUBLIC)]

    # Per-side keep-alive observations sampled while pumping: distinct ping
    # nonces seen in flight, and the worst consecutive-miss count.
    seen = {a_sm: {"nonces": set(), "max_failures": 0},
            b_sm: {"nonces": set(), "max_failures": 0}}
    captured = {}

    async def _scenario():
        nonlocal captured
        await asyncio.wait_for(a_sm.initiate(b_p2p.peer_id, [bob_addr]), timeout=10.0)
        await pump_until(targets,
                         predicate=lambda: a_sm.is_established(b_p2p.peer_id)
                         and b_sm.is_established(a_p2p.peer_id),
                         timeout=20.0)
        sessions = {a_sm: a_sm.get_session(b_p2p.peer_id),
                    b_sm: b_sm.get_session(a_p2p.peer_id)}

        loop = asyncio.get_event_loop()
        deadline = loop.time() + IDLE_S

        def _sample_until_deadline():
            for sm, sess in sessions.items():
                if sess._ping_in_flight:
                    seen[sm]["nonces"].add(bytes(sess._ping_nonce))
                seen[sm]["max_failures"] = max(seen[sm]["max_failures"],
                                               sess._consecutive_ping_failures)
            return loop.time() >= deadline

        captured = await pump_until(targets, predicate=_sample_until_deadline,
                                    timeout=IDLE_S + 15.0)

    asyncio.run(_scenario())

    bad = [(sm.p2p_name, ev.type.value)
           for sm, evs in captured.items() for ev in evs
           if ev.type in (SessionEventType.RECONNECTING, SessionEventType.LOST,
                          SessionEventType.FAILED)]
    assert not bad, f"idle session degraded: {bad}"

    a_sess = a_sm.get_session(b_p2p.peer_id)
    b_sess = b_sm.get_session(a_p2p.peer_id)
    assert a_sess.state == SessionState.ESTABLISHED
    assert b_sess.state == SessionState.ESTABLISHED
    for sm, label in ((a_sm, "alice"), (b_sm, "bob")):
        assert len(seen[sm]["nonces"]) >= 2, \
            f"{label} sent fewer than 2 keep-alive pings in {IDLE_S}s"
        assert seen[sm]["max_failures"] == 0, \
            f"{label} recorded {seen[sm]['max_failures']} ping miss(es) while idle"


def _host_decision_hook(peer_id, remote_node_id, remote_profile, p2p_name, verified_claims=None):
    async def _hook():
        if p2p_name == NodeConn.P2P_WORLD:
            return ProfileDecisionResult(accepted=True, await_private_join=True)
        spec = WorldGrantSpec(
            role=3,
            world_profile_bytes=json.dumps({"node_id": "world-node"}).encode(),
            rendezvous_tag=42,
            code_bundle=json.dumps({"agent_files": None, "stats_code": None}).encode(),
            initial_stats=b"",
        )
        return ProfileDecisionResult(accepted=True, world_grant=spec)
    return _hook()


def test_real_world_link_idle_keepalive_after_handoff(real_nodes):
    """Field topology: full world-join handoff (grant on public, PRIVATE_JOIN on
    world, GOAWAY of the public session), then 42s of idle on the world link.
    The world dial uses QUIC when available, like NodeConn.__connect does with
    profile addresses."""
    j_pub_p2p, j_world_p2p, j_conn, j_pub, j_world = real_nodes.world_node("joiner")
    h_pub_p2p, h_world_p2p, h_conn, h_pub, h_world = real_nodes.world_node(
        "host", decision_hook=_host_decision_hook)

    host_pub_addr = pick_dialable_addr(h_pub_p2p)
    host_world_addr = _dialable_addr(h_world_p2p, "quic") or pick_dialable_addr(h_world_p2p)
    host_pub_id = h_pub_p2p.peer_id
    host_world_id = h_world_p2p.peer_id
    joiner_world_id = j_world_p2p.peer_id

    targets = [
        (j_conn, j_pub, NodeConn.P2P_PUBLIC), (j_conn, j_world, NodeConn.P2P_WORLD),
        (h_conn, h_pub, NodeConn.P2P_PUBLIC), (h_conn, h_world, NodeConn.P2P_WORLD),
    ]
    closed = []
    seen = {j_world: {"nonces": set(), "max_failures": 0},
            h_world: {"nonces": set(), "max_failures": 0}}
    captured = {}

    async def orchestrate(ev):
        if ev.type == SessionEventType.ESTABLISHED:
            grant = ev.data.get("world_grant")
            if grant is not None:
                blob = dict(grant)
                blob["grant_id_bytes"] = bytes.fromhex(grant["grant_id"])
                for e in await j_world.initiate(
                        host_world_id, [host_world_addr],
                        remote_other_peer_id_hint=host_world_id, grant=blob):
                    await orchestrate(e)
            elif ev.p2p_name == NodeConn.P2P_WORLD and ev.role.value == "initiator":
                for e in await j_pub.close_session(host_pub_id, WORLD_HANDOFF_COMPLETE):
                    await orchestrate(e)
        elif ev.type == SessionEventType.CLOSED:
            closed.append(ev)

    async def _scenario():
        nonlocal captured
        for e in await j_pub.initiate(host_pub_id, [host_pub_addr], is_world_join=True,
                                      remote_other_peer_id_hint=host_world_id):
            await orchestrate(e)
        await pump_until(
            targets,
            predicate=lambda: (
                j_world.is_established(host_world_id)
                and h_world.is_established(joiner_world_id)
                and any(e.p2p_name == NodeConn.P2P_PUBLIC for e in closed)),
            on_event=orchestrate,
            timeout=60.0)

        sessions = {j_world: j_world.get_session(host_world_id),
                    h_world: h_world.get_session(joiner_world_id)}
        loop = asyncio.get_event_loop()
        deadline = loop.time() + IDLE_S

        def _sample_until_deadline():
            for sm, sess in sessions.items():
                if sess._ping_in_flight:
                    seen[sm]["nonces"].add(bytes(sess._ping_nonce))
                seen[sm]["max_failures"] = max(seen[sm]["max_failures"],
                                               sess._consecutive_ping_failures)
            return loop.time() >= deadline

        captured = await pump_until(targets, predicate=_sample_until_deadline,
                                    timeout=IDLE_S + 15.0)

    asyncio.run(_scenario())

    bad = [(sm.p2p_name, ev.type.value)
           for sm, evs in captured.items() for ev in evs
           if ev.type in (SessionEventType.RECONNECTING, SessionEventType.LOST,
                          SessionEventType.FAILED)]
    assert not bad, f"world link degraded while idle: {bad}"
    assert j_world.get_session(host_world_id).state == SessionState.ESTABLISHED
    assert h_world.get_session(joiner_world_id).state == SessionState.ESTABLISHED
    for sm, label in ((j_world, "joiner"), (h_world, "host")):
        assert len(seen[sm]["nonces"]) >= 2, \
            f"{label} sent fewer than 2 keep-alive pings in {IDLE_S}s"
        assert seen[sm]["max_failures"] == 0, \
            f"{label} recorded {seen[sm]['max_failures']} ping miss(es) while idle"
