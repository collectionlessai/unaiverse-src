"""A joiner and a member of the world: the joiner dials the member only once a roster
that lists it has been published.

A member cuts, as an unknown peer, whoever connects to it on the world instance before
the roster that names that peer reached it. A joiner that dials the members right after
its join, from the members its grant listed, races the publication of the roster that
names it, and loses it when the member reads the rendezvous first.

The scenarios run a world host on the REAL ``Node`` loop over a real ``NodeConn``, and a
member and a joiner on real ``NodeConn`` instances driven through the calls ``node.py``
makes to the Coordinator (the public leg, the world handoff, the join body's lists, the
rendezvous read, the reconcile and the pump). Only the transports are fake: they connect
both ends of a dial, and every member reads the last roster the host published.
"""

import json
import time
import asyncio

import pytest

from unaiverse.clock import clock
from unaiverse.networking.node.connpool import NodeConn
from unaiverse.networking.node.profile import NodeProfile
from unaiverse.networking.p2p.messages import Msg
from unaiverse.networking.session import SessionEventType
from unaiverse.networking.session.sess_msg import ProfileDecisionResult

from node_harness import (
    build_node_shell_on_real_conn, mint_node_token, full_profile_dict, FakeWorld, WireP2P,
)

pytestmark = pytest.mark.e2e

HOST_PEER = "12D3KooWhost1worldhosttttttttttttttttttttttttttttttt"


@pytest.fixture(autouse=True)
def _offline_node_profile(monkeypatch):
    """NodeProfile fills missing dynamic specs on construction, which includes a
    public-IP lookup over HTTP: keep the suite offline."""
    monkeypatch.setattr(NodeProfile, "_get_public_ip_address",
                        staticmethod(lambda: "203.0.113.1"))


class _Net:
    """The fake network: every transport by peer-id. A dial connects both ends, a
    disconnect drops both, and each member reads the last roster the host published."""

    def __init__(self):
        self.wires: dict[str, WireP2P] = {}
        self.host_world: WireP2P | None = None

    def add(self, wire: WireP2P) -> None:
        for other in self.wires.values():
            if other.log_sub == wire.log_sub:
                other.routes[wire.peer_id] = wire.inbox
                wire.routes[other.peer_id] = other.inbox
        self.wires[wire.peer_id] = wire

        async def _connect(addresses):
            target = addresses[0].rsplit("/p2p/", 1)[-1]
            wire.link(target, "outbound")
            self.wires[target].link(wire.peer_id, "inbound")
            return target, False

        async def _disconnect(peer_id):
            wire.disconnected.append(peer_id)
            wire.unlink(peer_id)
            if peer_id in self.wires:
                self.wires[peer_id].unlink(wire.peer_id)
            return True

        wire.connect = _connect
        wire.disconnect = _disconnect
        if wire.log_sub == "prv" and wire is not self.host_world:
            wire.get_rendezvous_peers_info = self.last_roster

    def last_roster(self):
        rosters = [msg.content for _, msg in self.host_world.broadcasts
                   if msg.content_type == Msg.WORLD_AGENTS_LIST]
        return rosters[-1] if rosters else None


class _Host:
    """The world host, on the real Node loop."""

    def __init__(self, net: _Net, root_private_key, root_public_key):
        profile = full_profile_dict("world-node", peer_id=HOST_PEER, private_peer_id=HOST_PEER + "-w")
        token = mint_node_token(root_private_key, profile, [HOST_PEER, HOST_PEER + "-w"])
        self.node, self.conn, self.p2p_public, self.p2p_world = build_node_shell_on_real_conn(
            HOST_PEER, world=FakeWorld(), node_id="world-node", public_key=root_public_key,
            token_factory=lambda: token)
        net.host_world = self.p2p_world
        net.add(self.p2p_public)
        net.add(self.p2p_world)

    async def cycle(self) -> None:
        clock.next_cycle()
        await self.node._Node__handle_network_connections()
        await self.node._Node__handle_network_messages()


async def _accept(*args):
    return ProfileDecisionResult(accepted=True)


class _Peer:
    """A node that joins the world, on a real NodeConn, driven as node.py drives it."""

    def __init__(self, name: str, net: _Net, root_private_key, root_public_key):
        self.public_peer_id = f"12D3KooW{name}0public"
        self.world_peer_id = f"12D3KooW{name}0world"
        self.p2p_public = WireP2P(self.public_peer_id, "pub")
        self.p2p_world = WireP2P(self.world_peer_id, "prv")
        node_id = f"{name}-node"
        profile = full_profile_dict(node_id, peer_id=self.public_peer_id,
                                    private_peer_id=self.world_peer_id,
                                    private_peer_addresses=list(self.p2p_world.addresses))
        token = mint_node_token(root_private_key, profile, [self.public_peer_id, self.world_peer_id])
        self.conn = NodeConn(max_connections=32, is_world_node=False, public_key=root_public_key,
                             token="peer-token", local_node_id=node_id)
        self.conn.attach_session_policy(p2p_u=self.p2p_public, p2p_w=self.p2p_world,
                                        local_profile_factory=lambda: profile,
                                        decision_hook=_accept, token_factory=lambda: token)
        net.add(self.p2p_public)
        net.add(self.p2p_world)
        self.events = []

    async def cycle(self) -> list:
        """One cycle in the order of the main loop: the rendezvous, the reconcile (where
        the perimeter cuts), then the Session pump."""
        await self.conn.set_world_agents_and_world_masters_lists_from_rendezvous()
        await self.conn.update()
        _, public_inbound = await self.conn.get_messages(p2p_name=NodeConn.P2P_PUBLIC)
        _, world_inbound = await self.conn.get_messages(p2p_name=NodeConn.P2P_WORLD)
        events = await self.conn.pump_sessions({NodeConn.P2P_PUBLIC: public_inbound,
                                                NodeConn.P2P_WORLD: world_inbound}, now=time.monotonic())
        self.events.extend(events)
        await asyncio.sleep(0)           # let a dial task run
        return events


def _established(events, peer_id: str):
    return [e for e in events if e.type == SessionEventType.ESTABLISHED and e.peer_id == peer_id]


async def _join(host: _Host, peer: _Peer) -> dict:
    """The whole join, as node.py runs it: the public leg up to the grant, the world
    handoff up to the world Session, then the join body's lists and the completion of
    the handoff. It stops as soon as the joiner is in: the host has registered it, and
    has not published the roster that names it yet."""
    await peer.conn.initiate_session(NodeConn.P2P_PUBLIC, host.p2p_public.peer_id,
                                     host.p2p_public.addresses, is_world_join=True)
    grant = None
    for _ in range(10):
        established = _established(await peer.cycle(), host.p2p_public.peer_id)
        if established:
            grant = established[0].data["world_grant"]
            break
        await host.cycle()
    assert grant is not None, "no grant"

    await peer.conn.start_world_handoff(host.p2p_public.peer_id, host.p2p_world.peer_id,
                                        host.p2p_world.addresses, grant, False)
    for _ in range(10):
        if _established(await peer.cycle(), host.p2p_world.peer_id):
            break
        await host.cycle()
    else:
        raise AssertionError("no world Session")

    # The join body: the members of the grant, and the rendezvous gate seeded behind it
    world_profile = json.loads(grant["world_profile_bytes"].decode("utf-8"))
    summary = world_profile["dynamic"]["world_summary"]
    peer.conn.rendezvous_tag = grant["rendezvous_tag"] - 1
    peer.conn.set_world_masters_list(summary["world_masters"] or [])
    peer.conn.set_world_agents_list(summary["world_agents"] or [])
    await peer.conn.complete_world_handoff(host.p2p_world.peer_id)
    return grant


def test_the_joiner_dials_a_member_only_after_the_roster_that_lists_it(stepped_clock, root_keys):
    """The race of T9: the joiner dials the member right after its join, before the host
    has published the roster that names it. The member reads the rendezvous first and
    cuts the joiner as an unknown peer. Holding the dial until the joiner itself reads a
    roster that lists it lets the member read the same roster before the dial lands."""
    async def _scenario():
        private_pem, public_pem = root_keys
        net = _Net()
        host = _Host(net, private_pem, public_pem)
        member = _Peer("member", net, private_pem, public_pem)
        joiner = _Peer("joiner", net, private_pem, public_pem)

        await _join(host, member)
        await host.cycle()                               # the roster that names the member
        await member.cycle()
        await _join(host, joiner)
        assert joiner.conn.get_addrs(member.world_peer_id)   # the grant listed the member

        # The role of the joiner dials the member at once, as connect_by_role does
        await joiner.conn.initiate_session(NodeConn.P2P_WORLD, member.world_peer_id,
                                           joiner.conn.get_addrs(member.world_peer_id))
        for _ in range(6):
            await joiner.cycle()
            await member.cycle()
            await host.cycle()

        assert joiner.world_peer_id not in member.p2p_world.disconnected
        assert joiner.conn._sm_world.is_established(member.world_peer_id)
        assert member.conn._sm_world.is_established(joiner.world_peer_id)

    asyncio.run(_scenario())


def test_a_member_that_dials_the_joiner_first_is_answered_and_not_dialled_again(
        stepped_clock, root_keys):
    """The member reads the roster that names the joiner and dials it while the joiner's
    own dial is still held: the joiner answers as the responder, and the held dial is
    not started on top of that Session."""
    async def _scenario():
        private_pem, public_pem = root_keys
        net = _Net()
        host = _Host(net, private_pem, public_pem)
        member = _Peer("member", net, private_pem, public_pem)
        joiner = _Peer("joiner", net, private_pem, public_pem)
        await _join(host, member)
        await host.cycle()
        await member.cycle()
        await _join(host, joiner)
        await joiner.conn.initiate_session(NodeConn.P2P_WORLD, member.world_peer_id,
                                           joiner.conn.get_addrs(member.world_peer_id))

        await host.cycle()                               # the roster that names the joiner
        await member.cycle()
        await member.conn.initiate_session(NodeConn.P2P_WORLD, joiner.world_peer_id,
                                           member.conn.get_addrs(joiner.world_peer_id))
        await member.cycle()                             # the dial connects
        await member.cycle()                             # the HELLO goes out
        for _ in range(6):
            await joiner.cycle()
            await member.cycle()

        sess = joiner.conn._sm_world.get_session(member.world_peer_id)
        assert sess.role.value == "responder"
        assert joiner.conn._sm_world.is_established(member.world_peer_id)
        assert member.conn._sm_world.is_established(joiner.world_peer_id)
        assert joiner.world_peer_id not in member.p2p_world.disconnected

    asyncio.run(_scenario())
