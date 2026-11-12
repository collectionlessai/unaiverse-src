"""What a node reports about its connections: ``dynamic.connections`` of its profile.

The block lists the peers of the pools (public agents, world agents, world masters,
the world host) and is what the root server shows for the node. It must follow the
pools whatever path changed them: a peer whose Session ends cleanly is evicted on
the Session event, while its transport is still up, so the pool diff of ``update()``
never reports it.

Like ``test_world_roster_e2e.py``, the scenarios run the REAL ``Node`` loop methods
over a REAL ``NodeConn`` and ``NodeProfile``, with fake transports only.
"""

import time
import asyncio

import pytest

from unaiverse.clock import clock
from unaiverse.agent import Agent
from unaiverse.networking.node.connpool import NodeConn
from unaiverse.networking.node.profile import NodeProfile
from unaiverse.networking.session import SessionEventType
from unaiverse.networking.session.sess_msg import GRACEFUL_SHUTDOWN, WORLD_DISENGAGEMENT

from conftest import apump
from node_harness import (
    build_node_shell_on_real_conn, wire_remote_peer, mint_node_token, full_profile_dict,
    FakeWorld,
)

pytestmark = pytest.mark.e2e

NODE_PEER = "12D3KooWaaa1nodeshelllllllllllllllllllllllllllllllll"
REMOTE_PEER = "12D3KooWzzz9remoteeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"


@pytest.fixture(autouse=True)
def _offline_node_profile(monkeypatch):
    """NodeProfile fills missing dynamic specs on construction, which includes a
    public-IP lookup over HTTP: keep the suite offline."""
    monkeypatch.setattr(NodeProfile, "_get_public_ip_address",
                        staticmethod(lambda: "203.0.113.1"))


async def _cycle(node) -> None:
    """One cycle in the order of the main loop."""
    clock.next_cycle()
    await node._Node__handle_network_connections()
    await node._Node__handle_network_messages()


async def _connect(node, p2p, root_private_key, p2p_name=NodeConn.P2P_PUBLIC, **initiate_kwargs):
    """A remote peer connects to the node on the instance served by ``p2p`` and
    completes the handshake. Returns its SessionManager, to make it leave later."""
    profile = full_profile_dict("remote-node", peer_id=REMOTE_PEER, private_peer_id=REMOTE_PEER)
    token = mint_node_token(root_private_key, profile, [REMOTE_PEER])
    remote_sm, remote_inbox = wire_remote_peer(p2p, REMOTE_PEER, profile, token, p2p_name=p2p_name)

    p2p.link(REMOTE_PEER)
    await remote_sm.initiate(p2p.peer_id, ["/ip4/10.0.0.1/tcp/4001"], **initiate_kwargs)
    await asyncio.sleep(0)                # let the dial task run...
    await remote_sm.tick()                # ...and reap it: HELLO on the wire
    await _cycle(node)                    # HELLO -> HELLO_ACK
    await apump(remote_inbox, remote_sm)  # HELLO_ACK -> PROFILE_OFFER
    await _cycle(node)                    # OFFER -> ESTABLISHED
    events = await apump(remote_inbox, remote_sm)
    assert any(e.type == SessionEventType.ESTABLISHED for e in events), \
        f"not established: {[e.type.value for e in events]}"
    return remote_sm


def _ids(peer_infos) -> list | None:
    return None if peer_infos is None else [p["id"] for p in peer_infos]


def test_agent_connections_drop_a_peer_that_left_with_a_goaway(stepped_clock, root_keys):
    """A public peer connects to an agent, then leaves cleanly. The agent evicts it
    on the CLOSED of its Session, with the transport still up, and the connections it
    reports must drop it and flag the change for the root server."""
    async def _scenario():
        private_pem, public_pem = root_keys
        node, conn, p2p_public, _ = build_node_shell_on_real_conn(NODE_PEER, public_key=public_pem)
        connections = node.profile.get_dynamic_profile()["connections"]

        remote_sm = await _connect(node, p2p_public, private_pem)
        assert _ids(connections["public_agents"]) == [REMOTE_PEER]

        node.profile.unmark_change_in_connections()
        await remote_sm.close_session(p2p_public.peer_id, GRACEFUL_SHUTDOWN)
        await _cycle(node)                            # GOAWAY -> CLOSED, purged
        assert conn.get_pool_of(REMOTE_PEER) is None
        assert REMOTE_PEER in p2p_public.links        # the transport did not drop
        await _cycle(node)

        assert connections["public_agents"] == []
        assert node.profile.connections_changed()

    asyncio.run(_scenario())


def test_world_host_connections_drop_a_member_that_left_with_a_goaway(stepped_clock, root_keys):
    """Same on a world host, for the world pools: a member that leaves the world with
    a GOAWAY must leave the connections of the host too."""
    async def _scenario():
        private_pem, public_pem = root_keys
        world = FakeWorld()
        node, conn, _, p2p_world = build_node_shell_on_real_conn(
            NODE_PEER, world=world, public_key=public_pem)
        connections = node.profile.get_dynamic_profile()["connections"]

        # The pre-admission the public decision makes for a joiner, then its world leg.
        role = FakeWorld.ROLE_STR_TO_BITS["player"] | Agent.ROLE_WORLD_AGENT
        conn.add_to_world_agents_list(REMOTE_PEER, ["/ip4/10.0.0.9/tcp/4001"], role)
        remote_sm = await _connect(
            node, p2p_world, private_pem, p2p_name=NodeConn.P2P_WORLD,
            grant={"role": role, "expiry_unix_ms": int(time.time() * 1000) + 30_000})
        assert world.added and world.added[0][0] == REMOTE_PEER
        assert _ids(connections["world_agents"]) == [REMOTE_PEER]

        await remote_sm.close_session(p2p_world.peer_id, WORLD_DISENGAGEMENT)
        await _cycle(node)                            # GOAWAY -> CLOSED, unlisted and purged
        assert REMOTE_PEER in p2p_world.links
        await _cycle(node)

        assert connections["world_agents"] == []

    asyncio.run(_scenario())


def test_steady_node_does_not_flag_its_profile(stepped_clock, root_keys):
    """The connections are compared with the pools at every cycle, so a node whose
    pools do not change must write nothing: a flag raised by mistake would post the
    profile to the root server at every slot. An idle node keeps the None its profile
    starts with (None and the empty list are the same thing here)."""
    async def _scenario():
        private_pem, public_pem = root_keys
        node, conn, p2p_public, _ = build_node_shell_on_real_conn(NODE_PEER, public_key=public_pem)
        connections = node.profile.get_dynamic_profile()["connections"]

        for _ in range(3):
            await _cycle(node)
        assert connections["public_agents"] is None and connections["world_agents"] is None
        assert not node.profile.connections_changed()

        await _connect(node, p2p_public, private_pem)
        await _cycle(node)
        node.profile.unmark_change_in_connections()
        for _ in range(3):
            await _cycle(node)
        assert _ids(connections["public_agents"]) == [REMOTE_PEER]
        assert not node.profile.connections_changed()

    asyncio.run(_scenario())
