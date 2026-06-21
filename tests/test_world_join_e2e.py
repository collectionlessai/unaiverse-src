"""In-process world-join handshake — validates the full Stage-3 machinery
across two p2p instances per node, wired through a small p2p-aware router.

Topology (4 SessionManagers):

    joiner.public  ─┐   ┌─  host.public   (world host: issues WorldGrant)
                    │   │
    joiner.world   ─┘   └─  host.world     (world host: redeems the grant)

Flow:
  1. joiner.public dials host.public (is_world_join) → HELLO/PROFILE.
  2. host.public decision hook mints a WorldGrant → joiner.public ESTABLISHED
     carries the grant blob.
  3. orchestrator (mirrors node._dispatch_session_event) opens joiner.world with
     the grant → HELLO/PROFILE → both world sides ESTABLISHED (RT.3c: no PRIVATE_JOIN).
  4. orchestrator GOAWAYs joiner.public (WORLD_HANDOFF_COMPLETE) → host.public CLOSED.
"""

import json
import asyncio
from collections import deque

import pytest

from unaiverse.networking.session import (
    SessionManager, SessionState, SessionEventType, ProfileDecisionResult, WorldGrantSpec,
)
from unaiverse.networking.session.sess_msg import WORLD_HANDOFF_COMPLETE
from unaiverse.networking.node.connpool import NodeConn
from conftest import FakeP2P


def _run(coro):
    return asyncio.run(coro)


class RouterP2P:
    """P2P-per-instance injected into one SessionManager. send_session routes into
    the peer node's same-instance inbox; crypto is delegated to a FakeP2P. The
    cross-instance ``twin`` is the other network of the *same* node."""

    def __init__(self, p2p_name, peer_id, log_sub):
        self.p2p_name = p2p_name
        self.peer_id = peer_id
        self._fake = FakeP2P(peer_id=peer_id, log_sub=log_sub)
        self.connect_returns = (None, False)
        # inbox = deque of (sender_peer_id, protocol_id, sess_msg) for this instance.
        self.inbox = deque()
        self.peer_p2p = None   # the same-network RouterP2P of the peer node

    async def connect(self, addresses):
        return self.connect_returns

    async def send_session(self, peer_id, protocol_id, sess_msg):
        # Deliver into the peer node's inbox for the same instance; the sender as
        # seen by the receiver is *our* peer-id on that instance.
        self.peer_p2p.inbox.append((self.peer_id, protocol_id, sess_msg))
        return True

    async def sign_with_host_key(self, data):
        return await self._fake.sign_with_host_key(data)

    async def verify_peer_signature(self, peer_id, data, signature):
        return await self._fake.verify_peer_signature(peer_id, data, signature)


class RouterNode:
    """A node spanning both p2p instances. Holds a RouterP2P per network and
    cross-links them as twins; ``wire_to`` connects the two nodes per network."""

    def __init__(self, pub_peer_id, world_peer_id):
        self.public = RouterP2P(NodeConn.P2P_PUBLIC, pub_peer_id, log_sub="pub")
        self.world = RouterP2P(NodeConn.P2P_WORLD, world_peer_id, log_sub="prv")

    def by_name(self, p2p_name):
        return self.public if p2p_name == NodeConn.P2P_PUBLIC else self.world


    def wire_to(self, other: "RouterNode"):
        self.public.peer_p2p = other.public
        self.world.peer_p2p = other.world


def _host_decision_hook(peer_id, remote_node_id, remote_profile, p2p_name, verified_claims=None):
    async def _hook():
        if p2p_name == NodeConn.P2P_WORLD:
            return ProfileDecisionResult(accepted=True)
        spec = WorldGrantSpec(
            role=3,
            world_profile_bytes=json.dumps({"node_id": "world-node"}).encode(),
            rendezvous_tag=42,
            code_bundle=json.dumps({"agent_files": None, "stats_code": None}).encode(),
            initial_stats=b"",
        )
        return ProfileDecisionResult(accepted=True, world_grant=spec)
    return _hook()


def test_world_join_full_handoff():
    joiner_pub_id, joiner_world_id = "12D3Joiner_PUB", "12D3Joiner_WORLD"
    host_pub_id, host_world_id = "12D3Host_PUB", "12D3Host_WORLD"

    joiner = RouterNode(joiner_pub_id, joiner_world_id)
    host = RouterNode(host_pub_id, host_world_id)
    joiner.wire_to(host)
    host.wire_to(joiner)
    joiner.public.connect_returns = (host_pub_id, False)
    joiner.world.connect_returns = (host_world_id, False)

    # Profiles carry the world-side identity like real ones do: the host binds
    # the grant signature to the joiner's dynamic.private_peer_id.
    joiner_profile = {"node_id": "joiner-node", "dynamic": {"private_peer_id": joiner_world_id}}
    host_profile = {"node_id": "host-node", "dynamic": {"private_peer_id": host_world_id}}
    j_pub = SessionManager(joiner.public, NodeConn.P2P_PUBLIC, joiner_pub_id, "joiner-node",
                           local_profile_factory=lambda: joiner_profile)
    j_world = SessionManager(joiner.world, NodeConn.P2P_WORLD, joiner_world_id, "joiner-node",
                             local_profile_factory=lambda: joiner_profile)
    h_pub = SessionManager(host.public, NodeConn.P2P_PUBLIC, host_pub_id, "host-node",
                           local_profile_factory=lambda: host_profile,
                           decision_hook=_host_decision_hook)
    h_world = SessionManager(host.world, NodeConn.P2P_WORLD, host_world_id, "host-node",
                             local_profile_factory=lambda: host_profile,
                             decision_hook=_host_decision_hook)

    managers = {
        (joiner, NodeConn.P2P_PUBLIC): j_pub, (joiner, NodeConn.P2P_WORLD): j_world,
        (host, NodeConn.P2P_PUBLIC): h_pub,   (host, NodeConn.P2P_WORLD): h_world,
    }

    established = []
    closed = []

    def _orchestrate(ev):
        # Mirrors node._dispatch_session_event handoff logic.
        if ev.type == SessionEventType.ESTABLISHED:
            established.append(ev)
            grant = ev.data.get("world_grant")
            if grant is not None:
                # joiner.public got the grant → open joiner.world to redeem it.
                blob = dict(grant)
                blob["grant_id_bytes"] = bytes.fromhex(grant["grant_id"])
                evs = _run(j_world.initiate(
                    host_world_id, ["/world/addr"],
                    grant=blob))
                for e in evs:
                    _orchestrate(e)
            elif ev.p2p_name == NodeConn.P2P_WORLD and ev.role.value == "initiator":
                # joiner.world established → GOAWAY the public session.
                _run(j_pub.close_session(host_pub_id, WORLD_HANDOFF_COMPLETE))
        elif ev.type == SessionEventType.CLOSED:
            closed.append(ev)

    def pump():
        progressed = True
        guard = 0
        while progressed and guard < 50:
            progressed = False
            guard += 1
            for conn in (joiner, host):
                for p2p_name in (NodeConn.P2P_PUBLIC, NodeConn.P2P_WORLD):
                    inbox = conn.by_name(p2p_name).inbox
                    if inbox:
                        progressed = True
                        sender, proto, msg = inbox.popleft()
                        sm = managers[(conn, p2p_name)]
                        for ev in _run(sm.deliver(sender, proto, msg)):
                            _orchestrate(ev)

    # 1. Joiner starts the public world-join handshake.
    for ev in _run(j_pub.initiate(host_pub_id, ["/pub/addr"], is_world_join=True)):
        _orchestrate(ev)
    pump()

    # Both world sides must be ESTABLISHED.
    assert j_world.is_established(host_world_id), "joiner world session not established"
    assert h_world.is_established(joiner_world_id), "host world session not established"

    # The public sessions handed off: joiner closed it; host saw CLOSED.
    j_pub_sess = j_pub.get_session(host_pub_id)
    assert j_pub_sess is None or j_pub_sess.state in (SessionState.CLOSING, SessionState.CLOSED)
    host_closed = [e for e in closed if e.p2p_name == NodeConn.P2P_PUBLIC]
    assert any("WORLD_HANDOFF_COMPLETE" in e.data.get("reason", "") for e in host_closed)

    # The joiner's world Session established as initiator. RT.3c: the join completes on
    # the normal PROFILE handshake (no PRIVATE_JOIN, no rendezvous_topic in the event).
    assert any(e.p2p_name == NodeConn.P2P_WORLD and e.role.value == "initiator"
               for e in established)


# RT.3c removed PRIVATE_JOIN: the grant expiry is no longer re-checked at
# redemption (there is no redemption round-trip). The joiner still rejects an
# expired WORLD_GRANT on its public Session — see test_session_fsm
# test_initiator_rejects_expired_world_grant. The former
# test_private_join_with_expired_grant_is_rejected was dropped with that path.
