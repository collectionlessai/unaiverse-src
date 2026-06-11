"""Integration tests for Node + ConnPool + Session — the glue layer.

These exercise the real Node methods (``__handle_network_messages``,
``__handle_network_connections``, ``_dispatch_session_event``,
``_session_decision_hook``) over real SessionManagers, with the conn / hosted
/ profile surfaces faked (see ``node_harness.py``). They pin the contract
between the Session layer and the application side-effects: add_agent on
ESTABLISHED, purge on LOST/CLOSED, the allow-list gate, the world-handoff
bookkeeping, and the transport-reconnect signal from the pool diff.

The strict-xfail tests document confirmed integration bugs: they assert the
*correct* behaviour and start passing (drop the marker) once fixed.
"""

import json
import time
import asyncio

import pytest

from unaiverse.networking.node.connpool import NodeConn
from unaiverse.networking.session import SessionState, SessionEventType, SessionRole
from unaiverse.networking.session.session import SessionEvent

from conftest import apump
from node_harness import (
    build_agent_node_shell, make_remote_peer, full_profile_dict, FakeWorld,
)

NODE_PEER = "12D3KooWaaa1nodeshelllllllllllllllllllllllllllllllll"
REMOTE_PEER = "12D3KooWzzz9remoteeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"


@pytest.fixture(autouse=True)
def _offline_node_profile(monkeypatch):
    """NodeProfile fills missing dynamic specs on construction, which includes a
    public-IP lookup over HTTP — keep the suite offline."""
    from unaiverse.networking.node.profile import NodeProfile
    monkeypatch.setattr(NodeProfile, "_get_public_ip_address",
                        staticmethod(lambda: "203.0.113.1"))


async def _establish_inbound(node, conn, remote_sm, remote_inbox):
    """Drive a remote-initiated handshake through the node's message pump until
    both sides are ESTABLISHED. Returns the remote's final events."""
    await remote_sm.initiate(NODE_PEER, ["/ip4/127.0.0.1/tcp/0"])
    await node._Node__handle_network_messages()   # HELLO → HELLO_ACK
    await apump(remote_inbox, remote_sm)          # HELLO_ACK → PROFILE_OFFER
    await node._Node__handle_network_messages()   # OFFER → decision (+ESTABLISHED)
    return await apump(remote_inbox, remote_sm)   # DECISION (+ESTABLISHED)


def _agent_pair():
    """Node shell + a remote peer wired into its public inbox."""
    node, conn = build_agent_node_shell(NODE_PEER)
    remote_sm, remote_conn, remote_inbox = make_remote_peer(
        REMOTE_PEER, "remote-node", conn.inbox_by_p2p[NodeConn.P2P_PUBLIC])
    conn.send_inbox = remote_inbox
    remote_conn.connect_returns = (NODE_PEER, False)
    return node, conn, remote_sm, remote_inbox


def test_inbound_handshake_through_node_pump_calls_add_agent():
    """An inbound handshake pumped through __handle_network_messages must end
    with hosted.add_agent called with the joiner's parsed profile."""
    async def _scenario():
        node, conn, remote_sm, remote_inbox = _agent_pair()
        remote_events = await _establish_inbound(node, conn, remote_sm, remote_inbox)
        assert [e.type for e in remote_events] == [SessionEventType.ESTABLISHED]
        assert node._sm_public.is_established(REMOTE_PEER)
        assert len(node.hosted.added) == 1
        peer_id, profile = node.hosted.added[0]
        assert peer_id == REMOTE_PEER
        assert profile.get_static_profile()["node_id"] == "remote-node"

    asyncio.run(_scenario())


def test_decision_hook_rejects_node_not_in_allow_list():
    """With an allow-list set, a non-listed node-id must be rejected with
    NOT_IN_ALLOW_LIST and never reach add_agent."""
    async def _scenario():
        node, conn, remote_sm, remote_inbox = _agent_pair()
        node.allowed_node_ids = {"someone-else"}
        remote_events = await _establish_inbound(node, conn, remote_sm, remote_inbox)
        assert [e.type for e in remote_events] == [SessionEventType.FAILED]
        assert "NOT_IN_ALLOW_LIST" in remote_events[0].data["reason"]
        assert node.hosted.added == []
        sess = node._sm_public.get_session(REMOTE_PEER)
        assert sess is None or sess.state == SessionState.FAILED

    asyncio.run(_scenario())


def test_decision_hook_rejects_malformed_profile():
    """A profile that NodeProfile.from_dict cannot parse must be rejected via
    the hook's exception path (HOSTED_REJECTED), not crash the pump."""
    async def _scenario():
        node, conn = build_agent_node_shell(NODE_PEER)
        remote_sm, remote_conn, remote_inbox = make_remote_peer(
            REMOTE_PEER, "remote-node", conn.inbox_by_p2p[NodeConn.P2P_PUBLIC],
            profile_factory=lambda: {"node_id": "remote-node"})  # no static/dynamic/cv
        conn.send_inbox = remote_inbox
        remote_conn.connect_returns = (NODE_PEER, False)
        remote_events = await _establish_inbound(node, conn, remote_sm, remote_inbox)
        assert [e.type for e in remote_events] == [SessionEventType.FAILED]
        assert "HOSTED_REJECTED" in remote_events[0].data["reason"]
        assert node.hosted.added == []

    asyncio.run(_scenario())


def test_pool_removal_while_established_reconnects_session_and_purges():
    """Contract of __handle_network_connections on a pool drop: the session
    moves to RECONNECTING (grace) but the peer is purged from hosted and pools
    immediately. The RESUMED dispatch is what re-enrols the agent afterwards
    (see test_resumed_session_readds_purged_agent)."""
    async def _scenario():
        node, conn, remote_sm, remote_inbox = _agent_pair()
        await _establish_inbound(node, conn, remote_sm, remote_inbox)

        conn.next_removed = {NodeConn.OUT_PUBLIC: {REMOTE_PEER}}
        await node._Node__handle_network_connections()

        sess = node._sm_public.get_session(REMOTE_PEER)
        assert sess is not None and sess.state == SessionState.RECONNECTING
        assert REMOTE_PEER in node.hosted.removed
        assert REMOTE_PEER in conn.removed_calls

    asyncio.run(_scenario())


def test_resumed_session_readds_purged_agent():
    """After the pool-drop purge, a successful RESUME must re-enrol the agent:
    the remote initiator re-dials and RESUMEs, the node (responder, still in
    grace) ACKs, and the RESUMED dispatch re-adds the agent from the profile
    the session kept."""
    async def _scenario():
        node, conn, remote_sm, remote_inbox = _agent_pair()
        await _establish_inbound(node, conn, remote_sm, remote_inbox)
        assert REMOTE_PEER in node.hosted.all_agents

        # Transport drop seen by both sides; the node purges immediately.
        conn.next_removed = {NodeConn.OUT_PUBLIC: {REMOTE_PEER}}
        await node._Node__handle_network_connections()
        await remote_sm.on_peer_disconnected(NODE_PEER)
        assert REMOTE_PEER not in node.hosted.all_agents
        assert node._sm_public.get_session(REMOTE_PEER).state == SessionState.RECONNECTING

        # The remote initiator re-dials and RESUMEs; the node pump processes it.
        await remote_sm.tick()
        await asyncio.sleep(0.05)   # let the fire-and-forget re-dial + RESUME run
        await node._Node__handle_network_messages()   # RESUME → ACK + RESUMED dispatch
        await apump(remote_inbox, remote_sm)           # remote absorbs the ACK

        assert node._sm_public.is_established(REMOTE_PEER)
        assert remote_sm.is_established(NODE_PEER)
        # The agent was re-enrolled by the RESUMED dispatch.
        assert REMOTE_PEER in node.hosted.all_agents
        assert len(node.hosted.added) == 2
        assert node.hosted.added[1][1].get_static_profile()["node_id"] == "remote-node"

    asyncio.run(_scenario())


def test_session_lost_while_peer_still_in_pool_purges_and_quarantines():
    """A session that rides its grace period to LOST (peer never removed from
    the pool) must purge the peer and quarantine it on the next pump."""
    async def _scenario():
        node, conn, remote_sm, remote_inbox = _agent_pair()
        await _establish_inbound(node, conn, remote_sm, remote_inbox)

        await node._sm_public.on_peer_disconnected(REMOTE_PEER)
        sess = node._sm_public.get_session(REMOTE_PEER)
        assert sess.state == SessionState.RECONNECTING
        sess._state_entered_at -= 60.0   # blow past the 45s grace window

        await node._Node__handle_network_messages()   # tick → LOST → dispatch
        assert node._sm_public.get_session(REMOTE_PEER) is None
        assert REMOTE_PEER in node._sm_public._quarantine
        assert REMOTE_PEER in node.hosted.removed
        assert REMOTE_PEER in conn.removed_calls

    asyncio.run(_scenario())


def test_added_diff_kicks_resume_for_reconnecting_session():
    """A peer that reappears in the pool diff while its session is RECONNECTING
    must get the on_peer_reconnected signal (re-arming the immediate RESUME)."""
    async def _scenario():
        node, conn, remote_sm, remote_inbox = _agent_pair()
        await _establish_inbound(node, conn, remote_sm, remote_inbox)

        await node._sm_public.on_peer_disconnected(REMOTE_PEER)
        sess = node._sm_public.get_session(REMOTE_PEER)
        assert sess.state == SessionState.RECONNECTING
        # Dirty the recovery knobs so the reconnect kick is observable.
        sess._reconnect_substep = ""
        sess._next_redial_at = time.monotonic() + 999.0

        conn.peer_id_to_pool[REMOTE_PEER] = NodeConn.IN_PUBLIC
        conn.next_added = {NodeConn.IN_PUBLIC: {REMOTE_PEER}}
        await node._Node__handle_network_connections()

        assert sess._reconnect_substep == "redial"
        assert sess._next_redial_at == 0.0

    asyncio.run(_scenario())


def _world_grant_blob():
    world_profile = full_profile_dict("world-node-id")
    return {
        "grant_id": "ab" * 32,
        "grant_sig": "cd" * 8,
        "expiry_unix_ms": int(time.time() * 1000) + 30000,
        "world_profile_bytes": json.dumps(world_profile).encode("utf-8"),
        "role": 1,
        "rendezvous_tag": 0,
        "world_private_peer_record": b"",
        "code_bundle": b"",
        "initial_stats": b"",
    }


@pytest.mark.parametrize("failure_mode", ["dial_failed", "quarantined"])
def test_world_handoff_failure_cleans_pending_maps(failure_mode, monkeypatch):
    """When the world-side initiate fails (unreachable world / quarantined
    peer), the handoff bookkeeping must be rolled back."""
    async def _scenario():
        node, conn = build_agent_node_shell(NODE_PEER)
        world_world_peer = "world-node-id-world-peer"
        if failure_mode == "dial_failed":
            conn.connect_returns = (None, False)

            async def _no_sleep(_s):
                return None
            monkeypatch.setattr("asyncio.sleep", _no_sleep)
        else:
            node._sm_world._quarantine[world_world_peer] = time.monotonic() + 30.0

        await node._start_world_handoff("public-world-peer", _world_grant_blob())

        assert node._sm_world.is_established(world_world_peer) is False
        assert node._pending_world_handoff == {}
        assert node._world_to_public == {}

    asyncio.run(_scenario())


def test_closed_with_world_handoff_complete_does_not_purge():
    """The intentional GOAWAY that ends a world handoff must not purge the
    (just joined) node; any other CLOSED reason must."""
    async def _scenario():
        node, conn = build_agent_node_shell(NODE_PEER)

        def _closed(reason):
            return SessionEvent(
                type=SessionEventType.CLOSED, session_id="ab" * 8,
                peer_id=REMOTE_PEER, role=SessionRole.INITIATOR,
                p2p_name=NodeConn.P2P_PUBLIC, data={"reason": reason})

        await node._dispatch_session_event(_closed("remote_goaway:WORLD_HANDOFF_COMPLETE"))
        assert node.hosted.removed == []

        await node._dispatch_session_event(_closed("remote_goaway:GRACEFUL_SHUTDOWN"))
        assert node.hosted.removed == [REMOTE_PEER]

    asyncio.run(_scenario())


def test_world_host_grant_preregisters_joiner_world_peer():
    """When the world-host decision hook mints a grant, the joiner's private
    peer-id (from its offered profile) must already be admitted on the world
    instance, or the routing pass will disconnect its upcoming dial."""
    async def _scenario():
        world = FakeWorld()
        node, conn = build_agent_node_shell(NODE_PEER, world=world)

        decision = await node._session_decision_hook(
            REMOTE_PEER, "joiner-node", full_profile_dict("joiner-node"),
            NodeConn.P2P_PUBLIC)

        assert decision.accepted and decision.world_grant is not None
        # The grant itself carries the assigned role (it gets signed into the
        # WORLD_GRANT): custom role bits plus the world-agent bit.
        from unaiverse.agent import Agent
        assert decision.world_grant.role == (FakeWorld.ROLE_STR_TO_BITS["player"] |
                                             Agent.ROLE_WORLD_AGENT)
        assert "joiner-node-world-peer" in conn.world_agents_set

    asyncio.run(_scenario())


def test_world_host_detects_master_from_profile_static_node_id():
    """The joiner's node-id reaches the decision hook via the Session, which
    must dig it out of the profile's ``static`` block (get_all_profile() has no
    top-level node_id). A joiner listed in world_masters_node_ids must get the
    master bits in its grant and land in the masters list — this is exactly the
    field failure where a declared master was silently granted a plain role."""
    async def _scenario():
        world = FakeWorld()
        node, conn = build_agent_node_shell(NODE_PEER, world=world)
        node.world_masters_node_ids = {"joiner-node"}
        remote_sm, remote_conn, remote_inbox = make_remote_peer(
            REMOTE_PEER, "joiner-node", conn.inbox_by_p2p[NodeConn.P2P_PUBLIC])
        conn.send_inbox = remote_inbox
        remote_conn.connect_returns = (NODE_PEER, False)

        await remote_sm.initiate(NODE_PEER, ["/ip4/127.0.0.1/tcp/0"],
                                 is_world_join=True)
        await node._Node__handle_network_messages()   # HELLO → HELLO_ACK
        await apump(remote_inbox, remote_sm)          # HELLO_ACK → PROFILE_OFFER
        await node._Node__handle_network_messages()   # OFFER → decision + GRANT
        remote_events = await apump(remote_inbox, remote_sm)

        established = [e for e in remote_events
                       if e.type == SessionEventType.ESTABLISHED]
        assert established, f"no ESTABLISHED: {[e.type.value for e in remote_events]}"
        grant = established[0].data["world_grant"]
        from unaiverse.agent import Agent
        assert grant["role"] & Agent.ROLE_WORLD_MASTER == Agent.ROLE_WORLD_MASTER
        assert (world.assigned and world.assigned[0][1] is True), \
            "assign_role must be called with is_world_master=True"
        assert "joiner-node-world-peer" in conn.world_masters_set

    asyncio.run(_scenario())


def test_world_host_registers_joiner_with_grant_role():
    """At PRIVATE_JOIN the world host must register the joiner with the role
    carried by the (signature-verified) grant echo — no server-side stash, no
    silent fallback. Master bits land the joiner in the masters list."""
    async def _scenario():
        world = FakeWorld()
        node, conn = build_agent_node_shell(NODE_PEER, world=world)
        node_world_peer = NODE_PEER + "-w"
        remote_sm, remote_conn, remote_inbox = make_remote_peer(
            REMOTE_PEER, "joiner-node", conn.inbox_by_p2p[NodeConn.P2P_WORLD],
            p2p_name=NodeConn.P2P_WORLD)
        conn.send_inbox = remote_inbox
        remote_conn.connect_returns = (node_world_peer, False)

        from unaiverse.agent import Agent
        role = (FakeWorld.ROLE_STR_TO_BITS["player"] | Agent.ROLE_WORLD_MASTER)
        blob = {"grant_id_bytes": b"g" * 32, "role": role,
                "expiry_unix_ms": int(time.time() * 1000) + 30_000,
                "grant_sig": b"sig-bytes".hex()}

        await remote_sm.initiate(node_world_peer, ["/ip4/127.0.0.1/tcp/0"],
                                 remote_other_peer_id_hint=node_world_peer,
                                 grant=blob)
        await node._Node__handle_network_messages()   # HELLO → HELLO_ACK
        await apump(remote_inbox, remote_sm)          # HELLO_ACK → PROFILE_OFFER
        await node._Node__handle_network_messages()   # OFFER → await_private_join
        await apump(remote_inbox, remote_sm)          # DECISION → PRIVATE_JOIN
        await node._Node__handle_network_messages()   # JOIN → ACK + register
        remote_events = await apump(remote_inbox, remote_sm)

        assert any(e.type == SessionEventType.ESTABLISHED for e in remote_events), \
            f"joiner never established: {[e.type.value for e in remote_events]}"
        assert node._sm_world.is_established(REMOTE_PEER)
        assert REMOTE_PEER in conn.world_masters_set
        assert conn.world_roles[REMOTE_PEER] == role
        assert world.added and world.added[0][0] == REMOTE_PEER

    asyncio.run(_scenario())


def test_world_host_unlists_member_when_world_session_ends():
    """When a member's world Session terminates (remote GOAWAY here), the world
    host must drop it from the world agents/masters lists too, or routing and
    the rendezvous publish keep advertising a ghost."""
    async def _scenario():
        world = FakeWorld()
        node, conn = build_agent_node_shell(NODE_PEER, world=world)
        node_world_peer = NODE_PEER + "-w"
        remote_sm, remote_conn, remote_inbox = make_remote_peer(
            REMOTE_PEER, "joiner-node", conn.inbox_by_p2p[NodeConn.P2P_WORLD],
            p2p_name=NodeConn.P2P_WORLD)
        conn.send_inbox = remote_inbox
        remote_conn.connect_returns = (node_world_peer, False)

        from unaiverse.agent import Agent
        role = FakeWorld.ROLE_STR_TO_BITS["player"] | Agent.ROLE_WORLD_AGENT
        blob = {"grant_id_bytes": b"g" * 32, "role": role,
                "expiry_unix_ms": int(time.time() * 1000) + 30_000,
                "grant_sig": b"sig-bytes".hex()}
        await remote_sm.initiate(node_world_peer, ["/ip4/127.0.0.1/tcp/0"],
                                 remote_other_peer_id_hint=node_world_peer,
                                 grant=blob)
        await node._Node__handle_network_messages()
        await apump(remote_inbox, remote_sm)
        await node._Node__handle_network_messages()
        await apump(remote_inbox, remote_sm)
        await node._Node__handle_network_messages()
        await apump(remote_inbox, remote_sm)
        assert REMOTE_PEER in conn.world_agents_set

        # The member leaves: its GOAWAY reaches the host's pump as CLOSED.
        from unaiverse.networking.session.sess_msg import WORLD_DISENGAGEMENT
        await remote_sm.close_session(node_world_peer, WORLD_DISENGAGEMENT)
        await node._Node__handle_network_messages()

        assert REMOTE_PEER not in conn.world_agents_set
        assert REMOTE_PEER not in conn.world_roles
        assert REMOTE_PEER in node.hosted.removed or REMOTE_PEER in conn.removed_calls

    asyncio.run(_scenario())


def test_local_closed_event_carries_goaway_reason():
    """The CLOSING side's CLOSED event must carry the GOAWAY reason it sent, so
    dispatch guards (e.g. WORLD_HANDOFF_COMPLETE) behave the same on both sides."""
    async def _scenario():
        node, conn, remote_sm, remote_inbox = _agent_pair()
        await _establish_inbound(node, conn, remote_sm, remote_inbox)

        from unaiverse.networking.session.sess_msg import WORLD_DISENGAGEMENT
        events = await remote_sm.close_session(NODE_PEER, WORLD_DISENGAGEMENT)
        assert events == []  # CLOSED arrives with the ack
        await node._Node__handle_network_messages()   # GOAWAY → ACK + node CLOSED
        remote_events = await apump(remote_inbox, remote_sm)
        closed = [e for e in remote_events if e.type == SessionEventType.CLOSED]
        assert closed and "WORLD_DISENGAGEMENT" in closed[0].data["reason"], \
            f"local CLOSED reason: {[e.data.get('reason') for e in closed]}"

    asyncio.run(_scenario())
