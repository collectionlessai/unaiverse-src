"""In-process world-join handshake — validates the full Stage-3 machinery
across two p2p instances per node, wired through a small p2p-aware router.

Topology (4 SessionManagers):

    joiner.public  ─┐   ┌─  host.public   (world host: issues WorldGrant)
                    │   │
    joiner.world   ─┘   └─  host.world     (world host: redeems via PRIVATE_JOIN)

Flow:
  1. joiner.public dials host.public (is_world_join) → HELLO/PROFILE.
  2. host.public decision hook mints a WorldGrant → joiner.public ESTABLISHED
     carries the grant blob.
  3. orchestrator (mirrors node._dispatch_session_event) opens joiner.world with
     the grant → HELLO/PROFILE then PRIVATE_JOIN → both world sides ESTABLISHED.
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
    return asyncio.get_event_loop().run_until_complete(coro)


class RouterConn:
    """A conn spanning both p2p instances of one node. send_session routes by
    p2p_name into the peer node's per-instance inbox; crypto is delegated to a
    per-instance FakeP2P."""

    def __init__(self, pub_peer_id, world_peer_id):
        self.p2p_name_to_p2p = {
            NodeConn.P2P_PUBLIC: FakeP2P(peer_id=pub_peer_id, log_sub="pub"),
            NodeConn.P2P_WORLD:  FakeP2P(peer_id=world_peer_id, log_sub="prv"),
        }
        self.peer_on = {NodeConn.P2P_PUBLIC: pub_peer_id, NodeConn.P2P_WORLD: world_peer_id}
        self.connect_returns_by_p2p = {}
        # inbox[p2p_name] = deque of (sender_peer_id, protocol_id, sess_msg)
        self.inbox = {NodeConn.P2P_PUBLIC: deque(), NodeConn.P2P_WORLD: deque()}
        self.peer_conn = None  # set to the other RouterConn

    def _p2p(self, p2p_name):
        return self.p2p_name_to_p2p[p2p_name]

    async def connect(self, addresses, p2p_name):
        return self.connect_returns_by_p2p.get(p2p_name, (None, False))

    async def send_session(self, peer_id, protocol_id, sess_msg, p2p_name):
        # Deliver into the peer node's inbox for the same instance; the sender as
        # seen by the receiver is *our* peer-id on that instance.
        self.peer_conn.inbox[p2p_name].append((self.peer_on[p2p_name], protocol_id, sess_msg))
        return True

    async def sign_with_host_key(self, p2p_name, data):
        return self._p2p(p2p_name).sign_with_host_key(data)

    async def verify_peer_signature(self, p2p_name, peer_id, data, signature):
        return self._p2p(p2p_name).verify_peer_signature(peer_id, data, signature)

    async def verify_signed_peer_record(self, p2p_name, envelope, expected_peer_id):
        return self._p2p(p2p_name).verify_signed_peer_record(envelope, expected_peer_id)

    async def ping_peer(self, p2p_name, peer_id, timeout_ms=5000):
        return self._p2p(p2p_name).ping_peer(peer_id, timeout_ms)

    async def generate_signed_peer_record(self, p2p_name):
        return self._p2p(p2p_name).generate_signed_peer_record()


def _host_decision_hook(peer_id, remote_node_id, remote_profile, p2p_name):
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


def test_world_join_full_handoff():
    joiner_pub_id, joiner_world_id = "12D3Joiner_PUB", "12D3Joiner_WORLD"
    host_pub_id, host_world_id = "12D3Host_PUB", "12D3Host_WORLD"

    joiner = RouterConn(joiner_pub_id, joiner_world_id)
    host = RouterConn(host_pub_id, host_world_id)
    joiner.peer_conn = host
    host.peer_conn = joiner
    joiner.connect_returns_by_p2p[NodeConn.P2P_PUBLIC] = (host_pub_id, False)
    joiner.connect_returns_by_p2p[NodeConn.P2P_WORLD] = (host_world_id, False)

    j_pub = SessionManager(joiner, NodeConn.P2P_PUBLIC, joiner_pub_id, "joiner-node",
                           local_profile_factory=lambda: {"node_id": "joiner-node"})
    j_world = SessionManager(joiner, NodeConn.P2P_WORLD, joiner_world_id, "joiner-node",
                             local_profile_factory=lambda: {"node_id": "joiner-node"})
    h_pub = SessionManager(host, NodeConn.P2P_PUBLIC, host_pub_id, "host-node",
                           local_profile_factory=lambda: {"node_id": "host-node"},
                           decision_hook=_host_decision_hook)
    h_world = SessionManager(host, NodeConn.P2P_WORLD, host_world_id, "host-node",
                             local_profile_factory=lambda: {"node_id": "host-node"},
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
                    remote_other_peer_id_hint=host_world_id, grant=blob))
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
                    inbox = conn.inbox[p2p_name]
                    if inbox:
                        progressed = True
                        sender, proto, msg = inbox.popleft()
                        sm = managers[(conn, p2p_name)]
                        for ev in _run(sm.deliver(sender, proto, msg)):
                            _orchestrate(ev)

    # 1. Joiner starts the public world-join handshake.
    for ev in _run(j_pub.initiate(host_pub_id, ["/pub/addr"], is_world_join=True,
                                  remote_other_peer_id_hint=host_world_id)):
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

    # The joiner world session carries the rendezvous topic from the host.
    world_established = [e for e in established
                        if e.p2p_name == NodeConn.P2P_WORLD and e.role.value == "initiator"]
    assert world_established and world_established[0].data.get("rendezvous_topic")
