"""A member that left us is not asked for again from a roster that still lists it.

A member that leaves its world closes its world Sessions with WORLD_DISENGAGEMENT. Only the
host writes the membership, so the roster of the others keeps listing it until the host
publishes again; while the host publishes nothing (frozen, or gone), every
``connect_by_role`` of a role found it there and asked to get in touch, and each ask was
cut by the perimeter of the member. The others now leave it out until a roster newer than
the one they held at the goodbye, a Session with it established again, or their own leave.

The node is an agent on the REAL ``Node`` loop methods over a REAL ``NodeConn``, in a world,
and the member is a remote SessionManager; only the transports are fake.
"""

import time
import asyncio

import pytest

from unaiverse.agent import Agent
from unaiverse.agent_basics import AgentBasics
from unaiverse.clock import clock
from unaiverse.networking.node.connpool import NodeConn
from unaiverse.networking.node.profile import NodeProfile
from unaiverse.networking.session import SessionEventType
from unaiverse.networking.session.sess_msg import GRACEFUL_SHUTDOWN, WORLD_DISENGAGEMENT

from conftest import apump
from node_harness import (
    build_node_shell_on_real_conn, wire_remote_peer, mint_node_token, full_profile_dict, FakeWorld,
)

pytestmark = pytest.mark.e2e

NODE_PEER = "12D3KooWaaa1nodeshelllllllllllllllllllllllllllllllll"
HOST_WORLD = "12D3KooWhost1worldhosttttttttttttttttttttttttttttttt"
MEMBER = "12D3KooWzzz9broadcasterrrrrrrrrrrrrrrrrrrrrrrrrrrrrr"
MEMBER_ADDR = f"/ip4/10.0.0.9/tcp/4001/p2p/{MEMBER}"
TAG = 50


@pytest.fixture(autouse=True)
def _offline_node_profile(monkeypatch):
    """NodeProfile looks its public address up over HTTP when it is built."""
    monkeypatch.setattr(NodeProfile, "_get_public_ip_address", staticmethod(lambda: "203.0.113.1"))


@pytest.fixture
def broadcaster_role():
    """The role of the chat world's broadcaster, registered in the class-level role tables
    and taken out at the end; yields the bits of a world agent with that role."""
    saved = (dict(AgentBasics.ROLE_BITS_TO_STR), dict(AgentBasics.ROLE_STR_TO_BITS), list(AgentBasics.CUSTOM_ROLES))
    Agent(proc=None).augment_roles(*AgentBasics.build_augmented_roles_dictionaries(["broadcaster"]))
    try:
        yield AgentBasics.ROLE_STR_TO_BITS["world_agent~broadcaster"]
    finally:
        AgentBasics.ROLE_BITS_TO_STR.clear()
        AgentBasics.ROLE_BITS_TO_STR.update(saved[0])
        AgentBasics.ROLE_STR_TO_BITS.clear()
        AgentBasics.ROLE_STR_TO_BITS.update(saved[1])
        AgentBasics.CUSTOM_ROLES = saved[2]


class _World:
    """The node in a world whose roster lists the member, the member's SessionManager, and
    a role of the node that asks to get in touch with broadcasters."""

    def __init__(self, root_keys, role_bits: int):
        private_pem, public_pem = root_keys
        self.node, self.conn, _, self.p2p_world = build_node_shell_on_real_conn(NODE_PEER, public_key=public_pem)
        self.conn.set_world(HOST_WORLD)
        self.role_bits = role_bits
        self.roster = None
        self.p2p_world.get_rendezvous_peers_info = lambda: self.roster

        profile = full_profile_dict("member-node", peer_id=MEMBER, private_peer_id=MEMBER)
        token = mint_node_token(private_pem, profile, [MEMBER])
        self.member_sm, self.member_inbox = wire_remote_peer(self.p2p_world, MEMBER, profile, token,
                                                             p2p_name=NodeConn.P2P_WORLD)
        self.agent = Agent(proc=None)
        self.agent.node_conn = self.conn
        self.asked = []

        async def _ask(addresses=None, public=True):
            self.asked.append(addresses)
            return None
        self.agent.node_ask_to_get_in_touch_fcn = _ask

    def publish(self, tag: int) -> None:
        """The roster of the host, listing the member."""
        self.roster = {"update_count": tag, "peers": [{"id": MEMBER, "addrs": [MEMBER_ADDR], "misc": self.role_bits}]}

    async def cycle(self) -> list:
        """One cycle of the node in the order of the main loop, returning its Session events."""
        clock.next_cycle()
        await self.node._Node__handle_network_connections()
        await self.node._Node__handle_network_messages()
        return await apump(self.member_inbox, self.member_sm)

    async def member_connects(self) -> None:
        """The member gets in touch with the node and the handshake completes."""
        self.p2p_world.link(MEMBER)
        await self.member_sm.initiate(self.p2p_world.peer_id, ["/ip4/10.0.0.1/tcp/4001"])
        await asyncio.sleep(0)
        await self.member_sm.tick()
        for _ in range(3):
            events = await self.cycle()
            if any(e.type == SessionEventType.ESTABLISHED for e in events):
                break
        assert self.conn._sm_world.is_established(MEMBER)

    async def member_leaves(self, reason: int = WORLD_DISENGAGEMENT) -> None:
        """The member closes its Session with the node, as a leave of its world does."""
        await self.member_sm.close_session(self.p2p_world.peer_id, reason)
        await self.cycle()
        await self.cycle()
        assert not self.conn._sm_world.is_established(MEMBER)

    async def asks_for_the_member(self) -> bool:
        """The role's connect_by_role("broadcaster"): whether it asked to get in touch with the member."""
        self.asked.clear()
        await self.agent.connect_by_role("broadcaster")
        return [MEMBER_ADDR] in self.asked


async def _member_left(root_keys, role_bits) -> _World:
    world = _World(root_keys, role_bits)
    world.publish(TAG)
    await world.cycle()
    await world.member_connects()
    await world.member_leaves()
    return world


def test_a_member_that_left_is_not_asked_for_until_a_newer_roster(stepped_clock, root_keys, broadcaster_role):
    """Path B of T6.4: the host is frozen and publishes nothing, the roster still lists the
    member that left, and the role asks for a broadcaster every few seconds."""
    async def _scenario():
        world = await _member_left(root_keys, broadcaster_role)
        for _ in range(5):
            await world.cycle()
            assert not await world.asks_for_the_member()

        world.publish(TAG + 1)
        await world.cycle()
        assert await world.asks_for_the_member()

    asyncio.run(_scenario())


def test_a_roster_of_the_tag_held_at_the_goodbye_does_not_bring_it_back(stepped_clock, root_keys, broadcaster_role):
    async def _scenario():
        world = await _member_left(root_keys, broadcaster_role)
        world.publish(TAG)
        await world.cycle()
        assert not await world.asks_for_the_member()

    asyncio.run(_scenario())


def test_a_member_that_gets_in_touch_again_is_taken_back(stepped_clock, root_keys, broadcaster_role):
    """The mark stops our dials only: the member's own Session is taken as any other, and
    once it is established the member is no longer left out."""
    async def _scenario():
        world = await _member_left(root_keys, broadcaster_role)
        await world.member_connects()
        await world.member_leaves(GRACEFUL_SHUTDOWN)
        assert await world.asks_for_the_member()

    asyncio.run(_scenario())


def test_leaving_the_world_takes_every_member_back(stepped_clock, root_keys, broadcaster_role):
    """A join of the same world again reads its members from the grant, whose roster is not
    newer than the one held at the goodbye: the leave must have forgotten the mark."""
    async def _scenario():
        world = await _member_left(root_keys, broadcaster_role)
        world.conn.set_world(None)

        world.conn.set_world(HOST_WORLD)
        world.conn.rendezvous_tag = TAG - 1
        world.conn.set_world_agents_list([{"id": MEMBER, "addrs": [MEMBER_ADDR], "misc": broadcaster_role}])
        assert await world.asks_for_the_member()

    asyncio.run(_scenario())


def test_a_world_host_keeps_no_mark_for_a_member_that_left(stepped_clock, root_keys):
    """The host writes the membership and unlists a member that leaves; it reads no roster
    that would take a mark away, so every member that ever left would stay marked."""
    async def _scenario():
        private_pem, public_pem = root_keys
        world = FakeWorld()
        node, conn, _, p2p_world = build_node_shell_on_real_conn(NODE_PEER, world=world, public_key=public_pem)
        role = FakeWorld.ROLE_STR_TO_BITS["player"] | Agent.ROLE_WORLD_AGENT
        conn.add_to_world_agents_list(MEMBER, [MEMBER_ADDR], role)
        profile = full_profile_dict("member-node", peer_id=MEMBER, private_peer_id=MEMBER)
        member_sm, member_inbox = wire_remote_peer(p2p_world, MEMBER, profile,
                                                   mint_node_token(private_pem, profile, [MEMBER]),
                                                   p2p_name=NodeConn.P2P_WORLD)
        p2p_world.link(MEMBER)
        await member_sm.initiate(p2p_world.peer_id, ["/ip4/10.0.0.1/tcp/4001"],
                                 grant={"role": role, "expiry_unix_ms": int(time.time() * 1000) + 30_000})
        await asyncio.sleep(0)
        await member_sm.tick()
        for _ in range(3):
            clock.next_cycle()
            await node._Node__handle_network_connections()
            await node._Node__handle_network_messages()
            await apump(member_inbox, member_sm)
        assert conn._sm_world.is_established(MEMBER)

        await member_sm.close_session(p2p_world.peer_id, WORLD_DISENGAGEMENT)
        clock.next_cycle()
        await node._Node__handle_network_connections()
        await node._Node__handle_network_messages()
        assert not conn._sm_world.is_established(MEMBER)
        assert not conn.is_departed(MEMBER)

    asyncio.run(_scenario())
