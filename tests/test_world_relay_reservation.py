"""The relay slot a member holds on its world host, as ``Node`` keeps it.

A member that is not publicly reachable reserves a slot on the world host at the join
(``Node.__join_world``) and renews it before it expires. The slot belongs to that world:
it is forgotten by every way out of it, and it is asked again when the world Session
resumes after a loss, because a relay drops the reservation of a peer the moment no
connection to that peer is left (seen with real libp2p in
``p2p/test_world_relay_realnet.py``).

The node is the shell of ``node_harness.py``: real ``Node`` methods and real
SessionManagers over a fake conn, whose ``reserve`` records whom it was asked to.
"""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from unaiverse.networking.node.connpool import NodeConn
from unaiverse.networking.node.profile import NodeProfile
from unaiverse.networking.session import SessionEventType, SessionRole, SessionState
from unaiverse.networking.session.session import SessionEvent
from unaiverse.networking.session.sess_msg import WORLD_DISENGAGEMENT
from unaiverse.utils.misc import pack_py_files

from conftest import apump
from node_harness import build_agent_node_shell, make_remote_peer, full_profile_dict

pytestmark = pytest.mark.e2e

NODE_PEER = "12D3KooWaaa1nodeshelllllllllllllllllllllllllllllllll"
WORLD_HOST = "12D3KooWzzz9worldhosttttttttttttttttttttttttttttttt"
OTHER_HOST = "12D3KooWyyy8otherhosttttttttttttttttttttttttttttttt"


@pytest.fixture(autouse=True)
def _offline_node_profile(monkeypatch):
    """NodeProfile looks its public address up over HTTP when it is built."""
    monkeypatch.setattr(NodeProfile, "_get_public_ip_address", staticmethod(lambda: "203.0.113.1"))


def _expiry_in(minutes: float) -> str:
    """An expiry as the Go library returns it."""
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


async def _member_in_world():
    """A node shell with an established world Session it opened toward the world host,
    in the world as a completed join leaves it."""
    node, conn = build_agent_node_shell(NODE_PEER)
    host_sm, _, host_inbox = make_remote_peer(WORLD_HOST, "world-host", conn.inbox_by_p2p[NodeConn.P2P_WORLD],
                                              p2p_name=NodeConn.P2P_WORLD)
    conn.send_inbox = host_inbox
    conn.connect_returns = (WORLD_HOST, False)

    assert await node.connect_via_session(peer_id=WORLD_HOST, addresses=["/ip4/127.0.0.1/tcp/0"],
                                          p2p_name=NodeConn.P2P_WORLD)
    await asyncio.sleep(0)
    await node._Node__handle_network_messages()
    await apump(host_inbox, host_sm)
    await node._Node__handle_network_messages()
    await apump(host_inbox, host_sm)
    await node._Node__handle_network_messages()
    assert conn._sm_world.is_established(WORLD_HOST)

    node.profile.get_dynamic_profile()['connections']['world_peer_id'] = WORLD_HOST
    conn.set_world(WORLD_HOST)
    node.hosted._in_world = True
    return node, conn, host_sm, host_inbox


async def _hold_slot(node, conn, minutes: float = 60):
    """The slot as the join records it, expiring in the given minutes."""
    conn.reserve_returns = _expiry_in(minutes)
    assert await node._reserve_on_world_host(WORLD_HOST)
    conn.reserve_calls.clear()


def _join_body(node, conn, world_is_relay: bool, subscribes: bool = True):
    """The real ``__join_world`` of a member that is not publicly reachable, with the
    steps after the reservation stubbed."""
    node.world_agent_files = None
    node.world_role_fsms = None
    conn.world_is_public = False
    conn.world_relay_enabled = True
    conn.rendezvous_tag = 0

    async def _subscribe(peer_id, channel=None):
        return subscribes
    conn.subscribe = _subscribe

    async def _add_streams(*a, **k):
        return True
    node.agent.add_compatible_streams = _add_streams
    node._Node__replace_agent_instance_by_role = lambda *a, **k: None

    return node._Node__join_world(
        profile=NodeProfile.from_dict(full_profile_dict("world-node-id", is_relay=world_is_relay)), role=1,
        packed_agent_files=pack_py_files({'player.py': "X = 1\n"}), roles_fsm={'player': '{"role": "player"}'},
        rendezvous_tag=0, initial_stats=None, peer_id=WORLD_HOST)


@pytest.mark.parametrize("world_is_relay", [True, False])
def test_join_records_the_slot_with_the_host_that_granted_it(world_is_relay):
    async def _scenario():
        node, conn = build_agent_node_shell(NODE_PEER)
        conn.reserve_returns = _expiry_in(60)

        assert await _join_body(node, conn, world_is_relay) is True

        if world_is_relay:
            assert conn.reserve_calls == [(WORLD_HOST, NodeConn.P2P_WORLD)]
            assert node.relay_reservation_peer_id == WORLD_HOST
            assert node.relay_reservation_expiry - datetime.now(timezone.utc) > timedelta(minutes=59)
        else:
            assert conn.reserve_calls == []
            assert node.relay_reservation_expiry is None and node.relay_reservation_peer_id is None

    asyncio.run(_scenario())


def test_a_join_that_fails_after_the_reservation_forgets_it():
    """The join body reserves first, and the steps after it can fail: the node is then out
    of that world with no leave, since it never got in."""
    async def _scenario():
        node, conn = build_agent_node_shell(NODE_PEER)
        node.joining_world_info = {"world_public_peer_id": "host-pub", "options": {}}
        conn._handoff.start("host-pub", WORLD_HOST)
        conn.set_world(WORLD_HOST)
        conn.reserve_returns = _expiry_in(60)

        async def _join(world_peer_id, grant):
            return await _join_body(node, conn, world_is_relay=True, subscribes=False)
        node._join_world_after_session = _join

        await node._dispatch_session_event(SessionEvent(
            type=SessionEventType.ESTABLISHED, session_id="00" * 8, peer_id=WORLD_HOST,
            role=SessionRole.INITIATOR, p2p_name=NodeConn.P2P_WORLD,
            data={"profile_b": {}, "grant": {"role": 1}}))

        assert conn.reserve_calls == [(WORLD_HOST, NodeConn.P2P_WORLD)]
        assert node.relay_reservation_expiry is None and node.relay_reservation_peer_id is None

    asyncio.run(_scenario())


@pytest.mark.parametrize("way_out", ["leave_world", "goaway_from_the_host", "world_session_lost"])
def test_leaving_the_world_forgets_the_slot_and_the_renewal_finds_nothing(way_out):
    """A node that left its world and stayed up reached the renewal with no world host,
    and the assertion there stopped the main loop."""
    async def _scenario():
        node, conn, host_sm, _ = await _member_in_world()
        await _hold_slot(node, conn, minutes=10)

        if way_out == "leave_world":
            await node.leave_world()
        elif way_out == "goaway_from_the_host":
            await host_sm.close_session(NODE_PEER + "-w", WORLD_DISENGAGEMENT)
            await node._Node__handle_network_messages()
        else:
            await node._dispatch_session_event(SessionEvent(
                type=SessionEventType.LOST, session_id="00" * 8, peer_id=WORLD_HOST,
                role=SessionRole.INITIATOR, p2p_name=NodeConn.P2P_WORLD, data={"reason": "liveness_timeout"}))

        assert not node.hosted.in_world()
        assert node.relay_reservation_expiry is None and node.relay_reservation_peer_id is None
        await node._tick_relay_reservation()
        assert conn.reserve_calls == []

    asyncio.run(_scenario())


@pytest.mark.parametrize("current_host", [None, OTHER_HOST])
def test_renewal_never_asks_a_host_other_than_the_one_that_granted_the_slot(current_host):
    """The last line of defence, for a way out that did not forget the slot: with no
    world the renewal has nothing to do, and with another world it must not reserve
    there, whatever that world is."""
    async def _scenario():
        node, conn, _, _ = await _member_in_world()
        await _hold_slot(node, conn, minutes=10)
        conn.set_world(current_host)

        await node._tick_relay_reservation()

        assert conn.reserve_calls == []
        assert node.relay_reservation_expiry is None and node.relay_reservation_peer_id is None

    asyncio.run(_scenario())


def test_renewal_asks_the_host_when_the_slot_nears_its_expiry():
    async def _scenario():
        node, conn, _, _ = await _member_in_world()

        await _hold_slot(node, conn, minutes=20)
        await node._tick_relay_reservation()
        assert conn.reserve_calls == []

        await _hold_slot(node, conn, minutes=14)
        conn.reserve_returns = _expiry_in(60)
        await node._tick_relay_reservation()
        assert conn.reserve_calls == [(WORLD_HOST, NodeConn.P2P_WORLD)]
        assert node.relay_reservation_expiry - datetime.now(timezone.utc) > timedelta(minutes=59)
        assert node.relay_reservation_peer_id == WORLD_HOST

        # A renewal that is refused gives the slot up, as it did
        await _hold_slot(node, conn, minutes=14)
        conn.reserve_returns = None
        await node._tick_relay_reservation()
        assert conn.reserve_calls == [(WORLD_HOST, NodeConn.P2P_WORLD)]
        assert node.relay_reservation_expiry is None and node.relay_reservation_peer_id is None

    asyncio.run(_scenario())


async def _lose_and_resume_the_world_session(node, conn, host_sm, host_inbox):
    """The transport to the world host drops on both sides, the node dials again and its
    RESUME is accepted."""
    await conn._sm_world.on_peer_disconnected(WORLD_HOST)
    await host_sm.on_peer_disconnected(NODE_PEER + "-w")
    assert conn._sm_world.get_session(WORLD_HOST).state == SessionState.RECONNECTING

    await node._Node__handle_network_messages()   # launches the dial
    await asyncio.sleep(0.05)
    await node._Node__handle_network_messages()   # reaps it, RESUME on the wire
    await apump(host_inbox, host_sm)
    await node._Node__handle_network_messages()   # the ACK, RESUMED dispatched
    assert conn._sm_world.is_established(WORLD_HOST)


def test_resumed_world_session_asks_for_the_slot_again():
    async def _scenario():
        node, conn, host_sm, host_inbox = await _member_in_world()
        await _hold_slot(node, conn, minutes=50)
        before = node.relay_reservation_expiry
        conn.reserve_returns = _expiry_in(60)

        await _lose_and_resume_the_world_session(node, conn, host_sm, host_inbox)

        assert conn.reserve_calls == [(WORLD_HOST, NodeConn.P2P_WORLD)]
        assert node.relay_reservation_expiry > before
        assert node.hosted.in_world()

    asyncio.run(_scenario())


def test_resumed_world_session_of_a_member_with_no_slot_asks_for_nothing():
    """A member that is publicly reachable, or whose world is no relay, holds no slot."""
    async def _scenario():
        node, conn, host_sm, host_inbox = await _member_in_world()
        conn.reserve_returns = _expiry_in(60)

        await _lose_and_resume_the_world_session(node, conn, host_sm, host_inbox)

        assert conn.reserve_calls == []
        assert node.relay_reservation_expiry is None

    asyncio.run(_scenario())


def test_resumed_session_of_another_peer_asks_for_nothing():
    async def _scenario():
        node, conn, _, _ = await _member_in_world()
        await _hold_slot(node, conn, minutes=50)

        await node._dispatch_session_event(SessionEvent(
            type=SessionEventType.RESUMED, session_id="00" * 8, peer_id=OTHER_HOST,
            role=SessionRole.INITIATOR, p2p_name=NodeConn.P2P_WORLD, data={"profile": None}))

        assert conn.reserve_calls == []

    asyncio.run(_scenario())


def test_slot_refused_after_a_resume_is_left_to_the_renewal():
    async def _scenario():
        node, conn, host_sm, host_inbox = await _member_in_world()
        await _hold_slot(node, conn, minutes=50)
        before = node.relay_reservation_expiry
        conn.reserve_returns = None

        await _lose_and_resume_the_world_session(node, conn, host_sm, host_inbox)

        assert conn.reserve_calls == [(WORLD_HOST, NodeConn.P2P_WORLD)]
        assert node.relay_reservation_expiry == before
        assert node.relay_reservation_peer_id == WORLD_HOST

    asyncio.run(_scenario())
