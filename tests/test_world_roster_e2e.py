"""World host: the roster a world advertises about its members.

A world host describes its members in two places: the world summary of its own
profile, which rides every grant (it is the member list a joiner starts from) and
is posted to the root server, and the rendezvous publish, which is how the members
learn about each other. Both must list the confirmed members only, and both must
follow the roster as soon as it changes, whatever path changed it: a join and a
clean leave are driven by Session events and never show up in the pool diff.

The scenarios run the REAL ``Node`` loop methods of a world host, in the order of
the main loop, over a REAL ``NodeConn`` (pools, perimeter, pre-admissions) and a
real ``NodeProfile``; only the transports are fake. The joiners are bare
SessionManagers carrying root-minted tokens, so the handshake runs its real token
and binding checks.
"""

import json
import time
import asyncio

import pytest

from unaiverse.clock import clock
from unaiverse.custom import Custom
from unaiverse.agent import Agent
from unaiverse.networking.node.connpool import NodeConn
from unaiverse.networking.node.profile import NodeProfile
from unaiverse.networking.p2p.messages import Msg
from unaiverse.networking.session import SessionEventType
from unaiverse.networking.session.sess_msg import WORLD_HANDOFF_COMPLETE, WORLD_DISENGAGEMENT

from conftest import apump
from node_harness import (
    build_node_shell_on_real_conn, wire_remote_peer, mint_node_token, full_profile_dict,
    FakeWorld,
)

pytestmark = pytest.mark.e2e

HOST_PEER = "12D3KooWhost1worldhosttttttttttttttttttttttttttttttt"
MASTER_NODE_ID = "master-node"


@pytest.fixture(autouse=True)
def _offline_node_profile(monkeypatch):
    """NodeProfile fills missing dynamic specs on construction, which includes a
    public-IP lookup over HTTP: keep the suite offline."""
    monkeypatch.setattr(NodeProfile, "_get_public_ip_address",
                        staticmethod(lambda: "203.0.113.1"))


class _Host:
    """The world host under test, with the accessors the assertions read."""

    def __init__(self, root_public_key):
        self.world = FakeWorld()
        self.node, self.conn, self.p2p_public, self.p2p_world = build_node_shell_on_real_conn(
            HOST_PEER, world=self.world, node_id="world-node", public_key=root_public_key)
        self.node.world_masters_node_ids = {MASTER_NODE_ID}
        self.public_peer_id = self.p2p_public.peer_id
        self.world_peer_id = self.p2p_world.peer_id

    async def cycle(self) -> int:
        """One cycle in the order of the main loop. Returns the cycle number."""
        clock.next_cycle()
        await self.node._Node__handle_network_connections()
        await self.node._Node__handle_network_messages()
        return clock.get_cycle()

    @property
    def summary(self) -> dict:
        return self.node.profile.get_dynamic_profile()["world_summary"]

    def rendezvous(self) -> list:
        """The rendezvous publishes so far, as (update_count, [member peer-ids])."""
        return [(msg.content["update_count"], [p["id"] for p in msg.content["peers"]])
                for _, msg in self.p2p_world.broadcasts
                if msg.content_type == Msg.WORLD_AGENTS_LIST]


class _Joiner:
    """A remote node joining the world: one bare SessionManager per network, both
    presenting a root-minted token that binds its two peer-ids. Each method is one
    step of the join, so a test can interleave two joiners cycle by cycle."""

    def __init__(self, name: str, host: _Host, root_private_key):
        self.host = host
        self.node_id = f"{name}-node"
        self.public_peer_id = f"12D3KooW{name}0public"
        self.world_peer_id = f"12D3KooW{name}0world"
        self.world_addrs = [f"/ip4/10.0.0.9/tcp/4001/p2p/{self.world_peer_id}"]
        profile = full_profile_dict(self.node_id, peer_id=self.public_peer_id,
                                    private_peer_id=self.world_peer_id,
                                    private_peer_addresses=self.world_addrs)
        token = mint_node_token(root_private_key, profile,
                                [self.public_peer_id, self.world_peer_id])
        self.sm_public, self.inbox_public = wire_remote_peer(
            host.p2p_public, self.public_peer_id, profile, token)
        self.sm_world, self.inbox_world = wire_remote_peer(
            host.p2p_world, self.world_peer_id, profile, token, p2p_name=NodeConn.P2P_WORLD)

    async def dial_public(self) -> None:
        """Open the public world-join Session: the HELLO lands in the host's inbox."""
        await self.sm_public.initiate(self.host.public_peer_id, ["/ip4/10.0.0.1/tcp/4001"],
                                      is_world_join=True)
        await asyncio.sleep(0)           # let the dial task run...
        await self.sm_public.tick()      # ...and reap it

    async def offer_public(self) -> None:
        """HELLO_ACK -> PROFILE_OFFER (the host decides on it at its next cycle)."""
        await apump(self.inbox_public, self.sm_public)

    async def take_grant(self) -> dict:
        """DECISION + WORLD_GRANT -> the grant the public Session established with."""
        events = await apump(self.inbox_public, self.sm_public)
        established = [e for e in events if e.type == SessionEventType.ESTABLISHED]
        assert established, f"no grant: {[e.type.value for e in events]}"
        return established[0].data["world_grant"]

    async def dial_world(self, grant: dict) -> None:
        """Connect to the world instance and open the Session redeeming the grant."""
        self.host.p2p_world.link(self.world_peer_id)
        await self.sm_world.initiate(self.host.world_peer_id, ["/ip4/10.0.0.1/tcp/4002"],
                                     grant=dict(grant))
        await asyncio.sleep(0)
        await self.sm_world.tick()

    async def offer_world(self) -> None:
        """HELLO_ACK -> PROFILE_OFFER (the host registers the member on it)."""
        await apump(self.inbox_world, self.sm_world)

    async def finish_join(self) -> None:
        """DECISION -> world Session established, then retire the public carrier."""
        events = await apump(self.inbox_world, self.sm_world)
        assert any(e.type == SessionEventType.ESTABLISHED for e in events), \
            f"world Session not established: {[e.type.value for e in events]}"
        await self.sm_public.close_session(self.host.public_peer_id, WORLD_HANDOFF_COMPLETE)

    async def leave(self) -> None:
        """Leave the world cleanly: a GOAWAY on the world Session. The transport stays
        up, as it does in the field while the farewell round-trip completes."""
        await self.sm_world.close_session(self.host.world_peer_id, WORLD_DISENGAGEMENT)


async def _get_grant(host: _Host, joiner: _Joiner) -> dict:
    """The whole public leg. The host decides in the second of its two cycles."""
    await joiner.dial_public()
    await host.cycle()                   # HELLO -> HELLO_ACK
    await joiner.offer_public()
    await host.cycle()                   # OFFER -> decision, pre-admission, grant
    return await joiner.take_grant()


async def _join(host: _Host, joiner: _Joiner) -> int:
    """The whole join. Returns the host cycle in which the member was registered."""
    grant = await _get_grant(host, joiner)
    await joiner.dial_world(grant)
    await host.cycle()                   # HELLO -> HELLO_ACK
    await joiner.offer_world()
    registered_at = await host.cycle()   # OFFER -> ESTABLISHED, registration
    await joiner.finish_join()
    return registered_at


def _summary_of(grant: dict) -> dict:
    """The world summary a joiner reads from its grant, parsed as the join does."""
    world_profile = NodeProfile.from_dict(json.loads(grant["world_profile_bytes"].decode("utf-8")))
    return world_profile.get_dynamic_profile()["world_summary"]


def _ids(peer_infos) -> list | None:
    # None is what a summary that was never refreshed lists: keep it visible.
    return None if peer_infos is None else [p["id"] for p in peer_infos]


def test_grant_lists_the_master_that_joined_first(stepped_clock, root_keys):
    """The regression: a master joins, then a second node asks to join. The profile
    in its grant is the only member list it has before the first rendezvous (and the
    only one ever, for a node that gets no pubsub), so it must list the master, with
    the address and role the joiner needs to dial it."""
    async def _scenario():
        private_pem, public_pem = root_keys
        host = _Host(public_pem)
        master = _Joiner("master", host, private_pem)
        agent = _Joiner("agent", host, private_pem)

        await _join(host, master)
        grant = await _get_grant(host, agent)

        summary = _summary_of(grant)
        assert _ids(summary["world_masters"]) == [master.world_peer_id]
        assert summary["world_agents"] == []          # not the joiner itself: it is not a member yet
        assert (summary["world_masters_count"], summary["world_agents_count"],
                summary["total_agents"]) == (1, 0, 1)

        entry = summary["world_masters"][0]
        assert entry["addrs"] == master.world_addrs
        assert entry["misc"] & Agent.ROLE_WORLD_MASTER == Agent.ROLE_WORLD_MASTER

    asyncio.run(_scenario())


def test_member_is_advertised_in_the_cycle_after_its_registration(stepped_clock, root_keys):
    """A member becomes part of the advertised roster when its registration confirms
    it, not while its handshake is still running, and the cycle right after the
    registration both publishes the rendezvous and mints grants that list it. The two
    must go together: a joiner dials the members of its grant at once, and a member
    that has not heard of the joiner yet cuts it and quarantines it."""
    async def _scenario():
        private_pem, public_pem = root_keys
        host = _Host(public_pem)
        master = _Joiner("master", host, private_pem)
        agent = _Joiner("agent", host, private_pem)

        await host.cycle()
        assert host.rendezvous() == [(0, [])]         # a just-started world: empty roster
        assert host.summary["world_masters"] == [] and host.summary["total_agents"] == 0

        grant = await _get_grant(host, master)
        await master.dial_world(grant)
        await agent.dial_public()
        await host.cycle()                            # both HELLOs -> HELLO_ACKs
        await master.offer_world()

        # Mid-handshake: connected and in the world pools (the perimeter needs it
        # there), yet not a member, so nothing was refreshed nor published.
        assert host.conn.get_pool_of(master.world_peer_id) == NodeConn.IN_WORLD_MASTERS
        assert master.world_peer_id in host.conn.world_preadmissions
        assert host.summary["world_masters"] == []
        assert len(host.rendezvous()) == 1

        registered_at = await host.cycle()            # master OFFER -> registration
        assert master.world_peer_id not in host.conn.world_preadmissions
        assert len(host.rendezvous()) == 1            # the registration came after this cycle's publish slot

        # The agent's offer reaches the host in the very next cycle.
        await agent.offer_public()
        await master.finish_join()
        assert await host.cycle() == registered_at + 1

        assert host.rendezvous()[1:] == [(registered_at + 1, [master.world_peer_id])]
        assert _ids(host.summary["world_masters"]) == [master.world_peer_id]
        assert _ids(_summary_of(await agent.take_grant())["world_masters"]) == [master.world_peer_id]

    asyncio.run(_scenario())


def test_world_summary_counts_follow_a_join_and_a_clean_leave(stepped_clock, root_keys):
    """The counts of the world profile (what the root server shows) after two joins
    and after a clean leave. The leave is a GOAWAY: the host evicts the member on the
    Session event while its transport is still up, so no pool diff ever reports it."""
    async def _scenario():
        private_pem, public_pem = root_keys
        host = _Host(public_pem)
        master = _Joiner("master", host, private_pem)
        agent = _Joiner("agent", host, private_pem)

        await _join(host, master)
        await _join(host, agent)
        await host.cycle()
        assert _ids(host.summary["world_masters"]) == [master.world_peer_id]
        assert _ids(host.summary["world_agents"]) == [agent.world_peer_id]
        assert (host.summary["world_masters_count"], host.summary["world_agents_count"],
                host.summary["total_agents"]) == (1, 1, 2)
        assert sorted(host.rendezvous()[-1][1]) == sorted([master.world_peer_id, agent.world_peer_id])

        host.node.profile.unmark_change_in_connections()
        published = len(host.rendezvous())
        await agent.leave()
        left_at = await host.cycle()                  # GOAWAY -> CLOSED, unlisted and purged
        assert agent.world_peer_id in host.p2p_world.links    # the transport did not drop
        await host.cycle()

        assert host.summary["world_agents"] == []
        assert _ids(host.summary["world_masters"]) == [master.world_peer_id]
        assert (host.summary["world_masters_count"], host.summary["world_agents_count"],
                host.summary["total_agents"]) == (1, 0, 1)
        assert host.rendezvous()[published:] == [(left_at + 1, [master.world_peer_id])]
        assert host.node.profile.connections_changed()        # the new counts go to the root

    asyncio.run(_scenario())


def test_unredeemed_preadmission_never_reaches_the_roster(stepped_clock, root_keys):
    """A joiner that got a grant but never completed the world handshake was never a
    member: neither its connection nor the expiry of its pre-admission is a roster
    change, so the summary stays put and nothing is published for it."""
    async def _scenario():
        private_pem, public_pem = root_keys
        host = _Host(public_pem)
        ghost = _Joiner("ghost", host, private_pem)

        await host.cycle()
        grant = await _get_grant(host, ghost)
        await ghost.dial_world(grant)
        await host.cycle()                            # connected, HELLO answered, then silence
        assert host.conn.get_pool_of(ghost.world_peer_id) == NodeConn.IN_WORLD_AGENTS

        host.conn.world_preadmissions[ghost.world_peer_id] = time.monotonic() - 1.   # the TTL ran out
        await host.cycle()                            # swept, then cut by the perimeter
        await host.cycle()

        assert ghost.world_peer_id in host.p2p_world.disconnected
        assert host.conn.get_pool_of(ghost.world_peer_id) is None
        assert host.summary["world_agents"] == [] and host.summary["total_agents"] == 0
        assert host.rendezvous() == [(0, [])]

    asyncio.run(_scenario())


def test_idle_world_publishes_on_the_periodic_slot_only(stepped_clock, root_keys):
    """With a steady roster the rendezvous keeps its PUBLISH_RENDEZVOUS_EVERY cadence:
    no publish per cycle, one when the slot comes due."""
    async def _scenario():
        private_pem, public_pem = root_keys
        host = _Host(public_pem)
        master = _Joiner("master", host, private_pem)

        await _join(host, master)
        await host.cycle()
        published = len(host.rendezvous())

        for _ in range(5):
            await host.cycle()
        assert len(host.rendezvous()) == published

        stepped_clock.advance(Custom.PUBLISH_RENDEZVOUS_EVERY)
        due_at = await host.cycle()
        assert host.rendezvous()[published:] == [(due_at, [master.world_peer_id])]

    asyncio.run(_scenario())


def test_failed_roster_publish_is_retried_at_the_next_cycle(stepped_clock, root_keys):
    """A roster change whose publish fails must not wait for the periodic slot: the
    request stays pending and the next cycle publishes it."""
    async def _scenario():
        private_pem, public_pem = root_keys
        host = _Host(public_pem)
        master = _Joiner("master", host, private_pem)

        registered_at = await _join(host, master)
        published = len(host.rendezvous())
        host.p2p_world.failing_broadcasts = 1

        await host.cycle()                            # the publish of the join fails
        assert len(host.rendezvous()) == published
        assert _ids(host.summary["world_masters"]) == [master.world_peer_id]   # the summary does not wait

        await host.cycle()
        assert host.rendezvous()[published:] == [(registered_at + 2, [master.world_peer_id])]

    asyncio.run(_scenario())
