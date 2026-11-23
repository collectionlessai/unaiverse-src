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
from unaiverse.networking.node.profile import NodeProfile
from unaiverse.networking.session import (SessionState, SessionEventType, SessionRole,
                                          VerifiedTokenClaims)
from unaiverse.networking.session.session import SessionEvent
from unaiverse.networking.session.sess_msg import (CV_HASH_MISMATCH, NOT_AUTHORIZED, GRACEFUL_SHUTDOWN,
                                                   ProfileDecisionResult)
from unaiverse.clock import clock
from unaiverse.custom import Custom

from conftest import apump
from node_harness import (
    build_agent_node_shell, make_remote_peer, full_profile_dict, FakeWorld,
)

pytestmark = pytest.mark.e2e

NODE_PEER = "12D3KooWaaa1nodeshelllllllllllllllllllllllllllllllll"
REMOTE_PEER = "12D3KooWzzz9remoteeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"


@pytest.fixture(autouse=True)
def _offline_node_profile(monkeypatch):
    """NodeProfile fills missing dynamic specs on construction, which includes a
    public-IP lookup over HTTP — keep the suite offline."""
    from unaiverse.networking.node.profile import NodeProfile
    monkeypatch.setattr(NodeProfile, "_get_public_ip_address",
                        staticmethod(lambda: "203.0.113.1"))


async def _reap_remote_dial(remote_sm):
    """Design A: initiate() only launches the dial as a background task. Let it run one loop
    turn, then tick to reap it so the initiator's HELLO reaches the wire."""
    await asyncio.sleep(0)
    await remote_sm.tick()


async def _reap_node_dial(node):
    """Same reap for a node-initiated Session: the node's own pump tick reaps its dial and
    puts the HELLO on the wire."""
    await asyncio.sleep(0)
    await node._Node__handle_network_messages()


async def _establish_inbound(node, conn, remote_sm, remote_inbox):
    """Drive a remote-initiated handshake through the node's message pump until
    both sides are ESTABLISHED. Returns the remote's final events."""
    await remote_sm.initiate(NODE_PEER, ["/ip4/127.0.0.1/tcp/0"])
    await _reap_remote_dial(remote_sm)            # reap the dial → remote HELLO on the wire
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
        assert conn._sm_public.is_established(REMOTE_PEER)
        assert len(node.hosted.added) == 1
        peer_id, profile = node.hosted.added[0]
        assert peer_id == REMOTE_PEER
        assert profile.get_static_profile()["node_id"] == "remote-node"

    asyncio.run(_scenario())


def test_inworld_inbound_agent_connection_calls_add_agent():
    """An agent-to-agent handshake on the WORLD instance (a world member
    connecting to us) must enrol the peer exactly like the public path: the
    responder branch is not reserved to world hosts."""
    async def _scenario():
        node, conn = build_agent_node_shell(NODE_PEER)
        remote_sm, remote_conn, remote_inbox = make_remote_peer(
            REMOTE_PEER, "remote-node", conn.inbox_by_p2p[NodeConn.P2P_WORLD],
            p2p_name=NodeConn.P2P_WORLD)
        conn.send_inbox = remote_inbox
        remote_conn.connect_returns = (NODE_PEER + "-w", False)

        await remote_sm.initiate(NODE_PEER + "-w", ["/ip4/127.0.0.1/tcp/0"])
        await _reap_remote_dial(remote_sm)            # reap the dial → remote HELLO on the wire
        await node._Node__handle_network_messages()   # HELLO → HELLO_ACK
        await apump(remote_inbox, remote_sm)          # HELLO_ACK → PROFILE_OFFER
        await node._Node__handle_network_messages()   # OFFER → decision (+ESTABLISHED)
        remote_events = await apump(remote_inbox, remote_sm)

        assert [e.type for e in remote_events] == [SessionEventType.ESTABLISHED]
        assert conn._sm_world.is_established(REMOTE_PEER)
        assert len(node.hosted.added) == 1
        assert node.hosted.added[0][0] == REMOTE_PEER

    asyncio.run(_scenario())


def test_inworld_outbound_agent_connection_joins_agent():
    """An initiator Session on the WORLD instance without a grant is an in-world
    agent-to-agent connection (connect_to/connect_by_role): ESTABLISHED must run
    __join_agent, not the world-join body."""
    async def _scenario():
        node, conn = build_agent_node_shell(NODE_PEER)
        remote_sm, remote_conn, remote_inbox = make_remote_peer(
            REMOTE_PEER, "remote-node", conn.inbox_by_p2p[NodeConn.P2P_WORLD],
            p2p_name=NodeConn.P2P_WORLD)
        conn.send_inbox = remote_inbox
        conn.connect_returns = (REMOTE_PEER, False)

        ok = await node.connect_via_session(peer_id=REMOTE_PEER,
                                            addresses=["/ip4/127.0.0.1/tcp/0"],
                                            p2p_name=NodeConn.P2P_WORLD)
        assert ok
        await _reap_node_dial(node)                   # reap node dial → HELLO on the wire
        await apump(remote_inbox, remote_sm)          # HELLO → HELLO_ACK
        await node._Node__handle_network_messages()   # HELLO_ACK → PROFILE_OFFER
        await apump(remote_inbox, remote_sm)          # OFFER → DECISION
        await node._Node__handle_network_messages()   # DECISION → ESTABLISHED dispatch

        assert conn._sm_world.is_established(REMOTE_PEER)
        assert len(node.hosted.added) == 1
        assert node.hosted.added[0][0] == REMOTE_PEER
        assert node.conn._pending_world_handoff == {} and node.conn._world_to_public == {}

    asyncio.run(_scenario())


def test_inbound_handshake_arms_interactive_lone_wolf_latch():
    """Interactive agent waiting for a lone wolf: the first inbound peer that
    completes the handshake is latched as lone_wolf_peer_id and arms
    ready_to_interact (restores the pre-Session PROFILE-handler flip)."""
    async def _scenario():
        node, conn, remote_sm, remote_inbox = _agent_pair()
        node._interact_mode_opts = {"ready_to_interact": False, "lone_wolf_peer_id": None}

        await _establish_inbound(node, conn, remote_sm, remote_inbox)

        assert conn._sm_public.is_established(REMOTE_PEER)
        assert node._interact_mode_opts["lone_wolf_peer_id"] == REMOTE_PEER
        assert node._interact_mode_opts["ready_to_interact"] is True

    asyncio.run(_scenario())


def test_outbound_public_handshake_arms_interactive_ready():
    """Interactive agent that dialed a specific lone wolf: when the public
    Session to that peer establishes, ready_to_interact is armed (restores the
    pre-Session AGENT_APPROVAL-handler flip)."""
    async def _scenario():
        node, conn = build_agent_node_shell(NODE_PEER)
        remote_sm, remote_conn, remote_inbox = make_remote_peer(
            REMOTE_PEER, "remote-node", conn.inbox_by_p2p[NodeConn.P2P_PUBLIC])
        conn.send_inbox = remote_inbox
        conn.connect_returns = (REMOTE_PEER, False)
        node._interact_mode_opts = {"ready_to_interact": False, "lone_wolf_peer_id": REMOTE_PEER}

        ok = await node.connect_via_session(peer_id=REMOTE_PEER,
                                            addresses=["/ip4/127.0.0.1/tcp/0"],
                                            p2p_name=NodeConn.P2P_PUBLIC)
        assert ok
        await _reap_node_dial(node)                   # reap node dial → HELLO on the wire
        await apump(remote_inbox, remote_sm)          # HELLO → HELLO_ACK
        await node._Node__handle_network_messages()   # HELLO_ACK → PROFILE_OFFER
        await apump(remote_inbox, remote_sm)          # OFFER → DECISION
        await node._Node__handle_network_messages()   # DECISION → ESTABLISHED dispatch

        assert conn._sm_public.is_established(REMOTE_PEER)
        assert node._interact_mode_opts["ready_to_interact"] is True

    asyncio.run(_scenario())


def test_world_session_goaway_triggers_full_world_leave():
    """A terminal event on the Session toward the CURRENT world host must run
    the full leave_world cleanup (profile, world lists, role restore), not a
    plain purge: otherwise in_world() stays True and the agent keeps acting
    against a dead world (stats sender, behaviors)."""
    async def _scenario():
        from unaiverse.agent import Agent
        from unaiverse.networking.session.sess_msg import WORLD_DISENGAGEMENT

        node, conn = build_agent_node_shell(NODE_PEER)
        remote_sm, remote_conn, remote_inbox = make_remote_peer(
            REMOTE_PEER, "world-host", conn.inbox_by_p2p[NodeConn.P2P_WORLD],
            p2p_name=NodeConn.P2P_WORLD)
        conn.send_inbox = remote_inbox
        conn.connect_returns = (REMOTE_PEER, False)

        ok = await node.connect_via_session(peer_id=REMOTE_PEER,
                                            addresses=["/ip4/127.0.0.1/tcp/0"],
                                            p2p_name=NodeConn.P2P_WORLD)
        assert ok
        await _reap_node_dial(node)                   # reap node dial → HELLO on the wire
        await apump(remote_inbox, remote_sm)
        await node._Node__handle_network_messages()
        await apump(remote_inbox, remote_sm)
        await node._Node__handle_network_messages()
        assert conn._sm_world.is_established(REMOTE_PEER)

        # Mark the remote as our current world host and mark ourselves in-world (what
        # __join_world records: the host on conn + the world role bit / in_world()).
        dyn = node.profile.get_dynamic_profile()
        dyn['connections']['world_peer_id'] = REMOTE_PEER
        conn.set_world(REMOTE_PEER)
        node.hosted._in_world = True

        # The world host tears the session down gracefully (GOAWAY).
        await remote_sm.close_session(NODE_PEER + "-w", WORLD_DISENGAGEMENT)
        await node._Node__handle_network_messages()

        assert dyn['connections']['world_peer_id'] is None
        assert conn.set_world_calls[-1] is None
        assert getattr(conn, "removed_all_world_agents", 0) == 1
        assert node.hosted.world_data_clears == 1
        assert node.hosted.accepted_roles[-1] == Agent.ROLE_PUBLIC
        assert node.hosted.behav.actionable is node.hosted
        assert node.hosted.im.agent is node.hosted

        # A later terminal event for the same peer must NOT re-run the leave
        # (world_peer_id is gone): it falls back to the plain purge path.
        late = SessionEvent(type=SessionEventType.LOST, session_id="00" * 8,
                            peer_id=REMOTE_PEER, role=SessionRole.INITIATOR,
                            p2p_name=NodeConn.P2P_WORLD, data={"reason": "liveness_timeout"})
        await node._dispatch_session_event(late)
        assert node.hosted.world_data_clears == 1

    asyncio.run(_scenario())


def test_involuntary_world_link_end_arms_rejoin_but_clean_close_does_not():
    """keep_rejoining: an INVOLUNTARY end of the Session toward the current world host
    re-arms the rejoin timer, and does so BEFORE leave_world runs. LOST (host vanished)
    and FAILED (e.g. an inbound-dispatch error on the live world Session) re-arm; a clean
    CLOSED (a GOAWAY from the host: kick, role-revoke, graceful shutdown, or our own
    token-expiry close) does NOT — re-joining would just be bounced by whatever closed us."""
    async def _scenario():
        for ev_type, should_arm in [
            (SessionEventType.LOST, True),
            (SessionEventType.FAILED, True),
            (SessionEventType.CLOSED, False),
        ]:
            node, conn = build_agent_node_shell(NODE_PEER)
            node.keep_rejoining = True
            node.rejoining_time = -1.
            conn.set_world(REMOTE_PEER)       # REMOTE is our current world host
            node.hosted._in_world = True      # the join completed (role bit set)

            ev = SessionEvent(type=ev_type, session_id="00" * 8, peer_id=REMOTE_PEER,
                              role=SessionRole.INITIATOR, p2p_name=NodeConn.P2P_WORLD,
                              data={"reason": "x"})
            left = await node._maybe_leave_world_on_session_end(ev)

            assert left is True, f"{ev_type.value}: should have left the world"
            armed = node.rejoining_time > 0.
            assert armed is should_arm, \
                f"{ev_type.value}: rejoin armed={armed}, expected {should_arm}"

    asyncio.run(_scenario())


def test_rejoin_watchdog_reschedules_retries_and_is_not_gated_on_in_world():
    """The rejoin watchdog: a due, keep_rejoining timer RESCHEDULES after firing (a dial is
    not yet a completed join); keep_rejoining=False disarms; a not-due timer is a no-op; a
    raising dial is caught and still reschedules (no node death); and it is NOT gated on
    in_world() — a leave_world that raised and left the role bit set must still let the
    watchdog re-fire and recover, not cancel the rejoin."""
    async def _scenario():
        node, conn = build_agent_node_shell(NODE_PEER)
        node.rejoining_kwargs = {}
        calls = []

        async def _fake_join(**kw):
            calls.append(kw)
            return "peer"
        node.ask_to_join_world = _fake_join

        # not due -> no-op
        node.keep_rejoining = True
        node.rejoining_time = -1.
        await node._tick_rejoin_watchdog()
        assert node.rejoining_time == -1. and calls == []

        # due + keep_rejoining False -> disarm, no dial
        node.keep_rejoining = False
        node.rejoining_time = clock.get_time()
        await node._tick_rejoin_watchdog()
        assert node.rejoining_time == -1. and calls == []

        # due + keep_rejoining True -> dial + reschedule (not disarm)
        node.keep_rejoining = True
        node.rejoining_time = clock.get_time()
        await node._tick_rejoin_watchdog()
        assert len(calls) == 1
        assert node.rejoining_time > clock.get_time()

        # due + in_world() True (leave_world partially failed) -> STILL dials, does NOT disarm
        node.hosted._in_world = True
        node.rejoining_time = clock.get_time()
        await node._tick_rejoin_watchdog()
        assert len(calls) == 2                        # fired despite in_world (the wedge fix)
        assert node.rejoining_time > clock.get_time()

        # the dial raises -> caught, still rescheduled (no run_async death)
        async def _boom(**kw):
            raise RuntimeError("boom")
        node.ask_to_join_world = _boom
        node.rejoining_time = clock.get_time()
        await node._tick_rejoin_watchdog()            # must not raise
        assert node.rejoining_time > clock.get_time()

    asyncio.run(_scenario())


def test_quarantine_refusal_pulls_rejoin_retry_in():
    """A quarantined world-join refusal carries retry_after_s, and the rearm PULLS IN an
    already-armed rejoin to just past the ban: without it the next watchdog fire bounces
    on the still-closed gate and the cadence doubles toward a stalled host."""
    async def _scenario():
        node, conn = build_agent_node_shell(NODE_PEER)
        node.keep_rejoining = True
        node.joining_world_info = {"world_public_peer_id": REMOTE_PEER,
                                   "options": {"node_name": "w"}}
        armed_at = clock.get_time() + 30.0
        node.rejoining_time = armed_at

        ev = SessionEvent(type=SessionEventType.FAILED, session_id="",
                          peer_id=REMOTE_PEER, role=SessionRole.INITIATOR,
                          p2p_name=NodeConn.P2P_PUBLIC,
                          data={"reason": "quarantined:4.0s", "retry_after_s": 4.0})
        await node._dispatch_session_event(ev)

        assert node.rejoining_time < armed_at            # pulled in, not left at +30
        remaining = node.rejoining_time - clock.get_time()
        assert 0.0 < remaining <= 5.5                    # ~ban + margin
        assert node.rejoining_kwargs                     # options captured for the retry

        # A LATER target must never push out a sooner armed one.
        node.rejoining_time = clock.get_time() + 2.0
        await node._dispatch_session_event(ev)
        assert node.rejoining_time - clock.get_time() <= 2.5

    asyncio.run(_scenario())


def test_rejoin_watchdog_keeps_sooner_retry_armed_by_attempt():
    """The watchdog disarms across the attempt and re-anchors at the default wait ONLY if
    the attempt's own failure dispatch did not arm a sooner retry (the quarantine refusal
    path does): the sooner target must survive the re-anchor."""
    async def _scenario():
        node, conn = build_agent_node_shell(NODE_PEER)
        node.keep_rejoining = True
        node.rejoining_kwargs = {"node_name": "w"}

        async def _refused_join(**kw):
            node.rejoining_time = clock.get_time() + 5.0   # what the refusal dispatch arms
            return None
        node.ask_to_join_world = _refused_join

        node.rejoining_time = clock.get_time()
        await node._tick_rejoin_watchdog()
        remaining = node.rejoining_time - clock.get_time()
        assert 0.0 < remaining <= 5.5      # kept the sooner retry, not re-anchored to +30

    asyncio.run(_scenario())


def test_the_join_after_a_failed_world_dial_can_redeem_its_grant():
    """T6.4-holly of 2026-11-21: the world dial that redeemed a grant failed, the join
    was retried, and its new grant was refused at once because the failed dial had put
    the world host's world peer in quarantine: a grant spent for nothing, and one more
    join. The failed redemption leaves no ban, so the next one dials."""
    async def _scenario():
        node, conn = build_agent_node_shell(NODE_PEER)
        host_public, host_world = "12D3KooWhost0public", "12D3KooWhost0world"
        node.keep_rejoining = True
        node.joining_world_info = {"world_public_peer_id": host_public, "options": {"node_name": "w"}}
        events = []
        dispatch = node._dispatch_session_event

        async def _spy(ev):
            events.append(ev)
            await dispatch(ev)
        node._dispatch_session_event = _spy

        def _grant():
            return {"grant_id": "aa", "rendezvous_tag": 1,
                    "expiry_unix_ms": int(time.time() * 1000) + 30000}

        for ev in await conn.start_world_handoff(host_public, host_world, ["/ip4/10.0.0.1/tcp/4002"],
                                                 _grant(), False):
            await node._dispatch_session_event(ev)
        for _ in range(5):
            await asyncio.sleep(0)
            await node._Node__handle_network_messages()
        assert [e.data.get("reason") for e in events if e.type == SessionEventType.FAILED] == ["dial_failed"]
        assert node.rejoining_time > 0.                  # the rejoin is armed, as before

        conn.connect_returns = (host_world, False)       # the host takes connections again
        retry = await conn.start_world_handoff(host_public, host_world, ["/ip4/10.0.0.1/tcp/4002"],
                                               _grant(), False)
        await asyncio.sleep(0)
        assert retry == []
        assert len(conn.connect_calls) == 2

    asyncio.run(_scenario())


# A lone wolf started with get_in_touch and keep_in_touch

THE_PEER = {"node_name": "the-peer"}


async def _lone_wolf_in_touch(keep_in_touch: bool):
    """A lone wolf whose public Session with REMOTE_PEER is established, with the state
    run(get_in_touch="the-peer", keep_in_touch=...) leaves; its getting in touch again is
    recorded instead of dialled."""
    node, conn = build_agent_node_shell(NODE_PEER)
    remote_sm, remote_conn, remote_inbox = make_remote_peer(
        REMOTE_PEER, "remote-node", conn.inbox_by_p2p[NodeConn.P2P_PUBLIC])
    conn.send_inbox = remote_inbox
    conn.connect_returns = (REMOTE_PEER, False)
    assert await node.connect_via_session(peer_id=REMOTE_PEER, addresses=["/ip4/127.0.0.1/tcp/0"],
                                          p2p_name=NodeConn.P2P_PUBLIC)
    await _reap_node_dial(node)                   # HELLO
    await apump(remote_inbox, remote_sm)          # HELLO_ACK
    await node._Node__handle_network_messages()   # PROFILE_OFFER
    await apump(remote_inbox, remote_sm)          # DECISION
    await node._Node__handle_network_messages()   # ESTABLISHED
    assert conn._sm_public.is_established(REMOTE_PEER)

    node.keep_in_touch = keep_in_touch
    if keep_in_touch:
        node.in_touch_kwargs = dict(THE_PEER)
        node.in_touch_peer_id = REMOTE_PEER
    dials = []

    async def _get_in_touch(**kwargs):
        dials.append(kwargs)
        return REMOTE_PEER
    node.ask_to_get_in_touch = _get_in_touch
    return node, conn, remote_sm, remote_inbox, dials


@pytest.mark.parametrize("keep_in_touch", [True, False])
def test_a_lone_wolf_kept_in_touch_dials_its_peer_again_after_a_lost(stepped_clock, keep_in_touch):
    """T11 part 2 of 2026-11-17: the dialler's Session ran out its reconnect window while
    its peer was frozen, went LOST, and the node stayed idle with the peer alive again.
    With keep_in_touch the watchdog of the rejoin gets in touch again once its wait is
    over; without it the node stays idle, as before."""
    async def _scenario():
        node, conn, _, _, dials = await _lone_wolf_in_touch(keep_in_touch)
        events = []
        dispatch = node._dispatch_session_event

        async def _spy(ev):
            events.append(ev.type)
            await dispatch(ev)
        node._dispatch_session_event = _spy
        conn.connect_returns = (None, False)      # the redials find nobody
        for ev in await conn.on_peer_disconnected(REMOTE_PEER):
            await node._dispatch_session_event(ev)
        conn._sm_public.get_session(REMOTE_PEER)._state_entered_at -= 3600.
        await node._Node__handle_network_messages()   # the window ends
        assert events == [SessionEventType.RECONNECTING, SessionEventType.LOST]

        await node._tick_rejoin_watchdog()
        assert dials == []                        # not before its wait
        stepped_clock.advance(Custom.REJOINING_WAIT)
        await node._tick_rejoin_watchdog()
        assert dials == ([THE_PEER] if keep_in_touch else [])

    asyncio.run(_scenario())


def test_a_peer_that_says_goodbye_is_not_dialled_again(stepped_clock):
    """A CLOSED is the peer's goodbye, not a LOST: no getting in touch again."""
    async def _scenario():
        node, conn, remote_sm, remote_inbox, dials = await _lone_wolf_in_touch(True)
        await remote_sm.close_session(NODE_PEER, GRACEFUL_SHUTDOWN)
        events = []
        dispatch = node._dispatch_session_event

        async def _spy(ev):
            events.append(ev.type)
            await dispatch(ev)
        node._dispatch_session_event = _spy
        await node._Node__handle_network_messages()
        assert SessionEventType.CLOSED in events

        stepped_clock.advance(Custom.REJOINING_WAIT)
        await node._tick_rejoin_watchdog()
        assert dials == []

    asyncio.run(_scenario())


def test_a_failed_attempt_is_retried_and_a_quarantine_refusal_pulls_the_retry_in(stepped_clock):
    async def _scenario():
        node, conn = build_agent_node_shell(NODE_PEER)
        node.keep_in_touch = True
        node.in_touch_kwargs = dict(THE_PEER)
        node.in_touch_peer_id = REMOTE_PEER

        def _failed(peer_id, data):
            return SessionEvent(type=SessionEventType.FAILED, session_id="", peer_id=peer_id,
                                role=SessionRole.INITIATOR, p2p_name=NodeConn.P2P_PUBLIC, data=data)

        await node._dispatch_session_event(_failed("12D3KooWsomeoneelse", {"reason": "dial_failed"}))
        assert node.in_touch_time == -1.          # another peer

        await node._dispatch_session_event(_failed(REMOTE_PEER, {"reason": "profile_rejected:NOT_AUTHORIZED"}))
        assert node.in_touch_time == clock.get_time() + Custom.REJOINING_WAIT

        await node._dispatch_session_event(_failed(REMOTE_PEER, {"reason": "quarantined:4.0s",
                                                                 "retry_after_s": 4.0}))
        assert 0. < node.in_touch_time - clock.get_time() <= 5.5

    asyncio.run(_scenario())


def test_the_watchdog_follows_the_peer_and_its_established_link_disarms_it(stepped_clock):
    """The peer may come back under another peer-id: the attempt follows it, and the
    ESTABLISHED Session with it disarms the watchdog. Until then each attempt is followed
    by the next one after the default wait."""
    async def _scenario():
        node, conn = build_agent_node_shell(NODE_PEER)
        node.keep_in_touch = True
        node.in_touch_kwargs = dict(THE_PEER)
        node.in_touch_peer_id = REMOTE_PEER
        node.in_touch_time = clock.get_time()
        dials = []

        async def _get_in_touch(**kwargs):
            dials.append(kwargs)
            return "12D3KooWnewpeer"
        node.ask_to_get_in_touch = _get_in_touch

        await node._tick_rejoin_watchdog()
        assert dials == [THE_PEER]
        assert node.in_touch_peer_id == "12D3KooWnewpeer"
        assert node.in_touch_time == clock.get_time() + Custom.REJOINING_WAIT

        await node._dispatch_session_event(SessionEvent(
            type=SessionEventType.ESTABLISHED, session_id="00" * 8, peer_id="12D3KooWnewpeer",
            role=SessionRole.INITIATOR, p2p_name=NodeConn.P2P_PUBLIC, data={}))
        assert node.in_touch_time == -1.

    asyncio.run(_scenario())


def test_run_turns_keep_in_touch_on_only_for_a_node_started_with_get_in_touch():
    node, conn = build_agent_node_shell(NODE_PEER)

    async def _nothing(*args, **kwargs):
        return None
    node.run_async = _nothing
    node.aclose = _nothing
    node.run(get_in_touch="the-peer", keep_in_touch=True)
    assert node.keep_in_touch is True
    node.run(keep_in_touch=True)
    assert node.keep_in_touch is False
    node.run(get_in_touch="the-peer")
    assert node.keep_in_touch is False


def test_run_async_records_what_to_dial_again_for_a_node_kept_in_touch():
    """The start of run_async records the target of get_in_touch, with the options it was
    given, and the peer it reached. A budget of no cycle cuts the loop itself short."""
    async def _scenario():
        node, conn = build_agent_node_shell(NODE_PEER)

        async def _nothing(*args, **kwargs):
            return None

        async def _get_in_touch(**kwargs):
            return REMOTE_PEER
        node.hosted.subscribe_to_pubsub_owned_streams = _nothing
        node.aclose = _nothing
        node.ask_to_get_in_touch = _get_in_touch
        node.keep_in_touch = True
        try:
            await node.run_async(cycles=0, get_in_touch="the-peer", public=True)
        except Exception:
            pass                                  # log.critical on the budget, where it raises
        assert node.in_touch_kwargs == {"node_name": "the-peer", "public": True}
        assert node.in_touch_peer_id == REMOTE_PEER

    asyncio.run(_scenario())


def test_involuntary_end_arms_rejoin_even_if_leave_world_raises():
    """The rejoin is armed BEFORE leave_world (best-effort, may raise): a LOST/FAILED must
    leave the rejoin armed even when leave_world blows up, so the watchdog can still recover
    the node. Validates the arm-before-leave ordering the docstring claims."""
    async def _scenario():
        node, conn = build_agent_node_shell(NODE_PEER)
        node.keep_rejoining = True
        node.rejoining_time = -1.
        conn.set_world(REMOTE_PEER)
        node.hosted._in_world = True

        async def _boom():
            raise RuntimeError("leave_world blew up")
        node.leave_world = _boom

        ev = SessionEvent(type=SessionEventType.LOST, session_id="00" * 8, peer_id=REMOTE_PEER,
                          role=SessionRole.INITIATOR, p2p_name=NodeConn.P2P_WORLD, data={"reason": "x"})
        left = await node._maybe_leave_world_on_session_end(ev)
        assert left is True
        assert node.rejoining_time > 0.               # armed despite leave_world raising

    asyncio.run(_scenario())


def test_world_handoff_failure_before_join_does_not_leave_world():
    """A world-session FAILED during the join handoff — before the join completes, so
    in_world() is still False — must NOT run the full leave_world cleanup: it is a
    failed join, not a departure. conn holds the host from handoff-initiate onward, so
    the guard keys off in_world() (the role bit), not merely the host being set. And it
    must NOT re-arm the rejoin here: a failed JOIN is retried by the rejoin watchdog
    (which reschedules on every dial), not by this departure path."""
    async def _scenario():
        from unaiverse.agent import Agent

        node, conn = build_agent_node_shell(NODE_PEER)
        node.keep_rejoining = True
        node.rejoining_time = -1.
        # Handoff initiated: conn knows the world host, but the join never completed
        # (no role bit set → in_world() stays False).
        conn.set_world(REMOTE_PEER)
        assert node.hosted.in_world() is False

        ev = SessionEvent(type=SessionEventType.FAILED, session_id="00" * 8,
                          peer_id=REMOTE_PEER, role=SessionRole.INITIATOR,
                          p2p_name=NodeConn.P2P_WORLD, data={"reason": "dial_failed"})
        left = await node._maybe_leave_world_on_session_end(ev)

        # Not treated as a departure, and leave_world's side effects never ran.
        assert left is False
        assert node.hosted.world_data_clears == 0
        assert Agent.ROLE_PUBLIC not in node.hosted.accepted_roles
        assert None not in conn.set_world_calls   # set_world(None) never called
        assert node.rejoining_time == -1.         # not re-armed here (the watchdog handles it)

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
        sess = conn._sm_public.get_session(REMOTE_PEER)
        assert sess is None or sess.state == SessionState.FAILED

    asyncio.run(_scenario())


def test_set_decision_policy_propagates_at_runtime():
    """R3: set_decision_policy() re-routes the decision for Sessions already wired. The hook
    the managers hold is a late-bound lambda that re-reads node._decision_policy on every
    call, not the frozen built-in, so swapping the policy at runtime takes effect on the
    same injected hook without rebuilding any Session."""
    async def _scenario():
        node, conn = build_agent_node_shell(NODE_PEER)
        # The default policy is the built-in, and the injected hook is the indirection
        # lambda (not the bound method itself).
        assert node._decision_policy == node._session_decision_hook
        injected = conn._sm_public._decision_hook
        assert injected is not node._session_decision_hook

        marker_reject = ProfileDecisionResult(accepted=False)
        marker_accept = ProfileDecisionResult(accepted=True)

        async def policy_reject(*_a):
            return marker_reject

        async def policy_accept(*_a):
            return marker_accept

        node.set_decision_policy(policy_reject)
        res = await injected(REMOTE_PEER, "remote-node", {}, NodeConn.P2P_PUBLIC, None)
        assert res is marker_reject

        # Swap at runtime: the SAME injected hook now yields the new policy's result.
        node.set_decision_policy(policy_accept)
        res = await injected(REMOTE_PEER, "remote-node", {}, NodeConn.P2P_PUBLIC, None)
        assert res is marker_accept

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


def test_decision_hook_rejects_cv_hash_mismatch():
    """RT: with verified claims present, a profile whose CV does not hash to the
    attested cv_hash is forged → reject CV_HASH_MISMATCH, before any accept/grant."""
    async def _scenario():
        node, conn = build_agent_node_shell(NODE_PEER)
        profile = full_profile_dict("joiner-node")
        claims = VerifiedTokenClaims(node_id="joiner-node", cv_hash="not-the-real-hash",
                                     p2p_peers=("joiner-node-world-peer",))
        decision = await node._session_decision_hook(
            REMOTE_PEER, "joiner-node", profile, NodeConn.P2P_PUBLIC, claims)
        assert decision.accepted is False
        assert decision.reason_code == CV_HASH_MISMATCH

    asyncio.run(_scenario())


def test_decision_hook_accepts_matching_cv_hash():
    """RT: a profile whose CV matches the attested cv_hash passes the integrity gate
    (agent host → accept)."""
    async def _scenario():
        node, conn = build_agent_node_shell(NODE_PEER)
        profile = full_profile_dict("joiner-node")
        _, (_, real_hash) = NodeProfile.from_dict(profile).verify_cv_hash("")
        claims = VerifiedTokenClaims(node_id="joiner-node", cv_hash=real_hash,
                                     p2p_peers=("joiner-node-world-peer",))
        decision = await node._session_decision_hook(
            REMOTE_PEER, "joiner-node", profile, NodeConn.P2P_PUBLIC, claims)
        assert decision.accepted is True

    asyncio.run(_scenario())


def test_world_host_rejects_unbound_world_peer():
    """RT binding: the world host rejects a joiner whose offered private (world-side)
    peer-id is NOT in the root-attested p2p_peers (NOT_AUTHORIZED), so it never
    preregisters a self-claimed world identity."""
    async def _scenario():
        world = FakeWorld()
        node, conn = build_agent_node_shell(NODE_PEER, world=world)
        profile = full_profile_dict("joiner-node")  # private_peer_id = joiner-node-world-peer
        _, (_, real_hash) = NodeProfile.from_dict(profile).verify_cv_hash("")
        claims = VerifiedTokenClaims(node_id="joiner-node", cv_hash=real_hash,
                                     p2p_peers=("joiner-node-pub-peer", "some-other-world-peer"))
        decision = await node._session_decision_hook(
            REMOTE_PEER, "joiner-node", profile, NodeConn.P2P_PUBLIC, claims)
        assert decision.accepted is False
        assert decision.reason_code == NOT_AUTHORIZED
        assert "joiner-node-world-peer" not in conn.world_agents_set

    asyncio.run(_scenario())


def test_world_host_grants_when_world_peer_bound():
    """RT binding: when the offered world-side peer-id IS in the attested p2p_peers,
    the binding passes and the host mints the grant + preregisters the joiner."""
    async def _scenario():
        world = FakeWorld()
        node, conn = build_agent_node_shell(NODE_PEER, world=world)
        profile = full_profile_dict("joiner-node")
        _, (_, real_hash) = NodeProfile.from_dict(profile).verify_cv_hash("")
        claims = VerifiedTokenClaims(node_id="joiner-node", cv_hash=real_hash,
                                     p2p_peers=("joiner-node-pub-peer", "joiner-node-world-peer"))
        decision = await node._session_decision_hook(
            REMOTE_PEER, "joiner-node", profile, NodeConn.P2P_PUBLIC, claims)
        assert decision.accepted and decision.world_grant is not None
        assert "joiner-node-world-peer" in conn.world_agents_set

        # The minted bundle is the world-definition single channel: packed files
        # + role FSMs, and no legacy stats_code field.
        bundle = json.loads(decision.world_grant.code_bundle.decode("utf-8"))
        assert bundle == {"agent_files": world.packed_agent_files,
                          "roles_fsm": world.role_to_behav}

    asyncio.run(_scenario())


def test_world_host_with_no_verified_claims_removes_no_listed_entry():
    """With no verified claims nothing attests who a joiner is: a joiner whose profile
    names the node of a listed member is admitted beside it, and the member stays."""
    async def _scenario():
        node, conn = build_agent_node_shell(NODE_PEER, world=FakeWorld())
        profile = full_profile_dict("joiner-node")
        _, (_, real_hash) = NodeProfile.from_dict(profile).verify_cv_hash("")
        claims = VerifiedTokenClaims(node_id="joiner-node", cv_hash=real_hash,
                                     p2p_peers=("joiner-node-pub-peer", "joiner-node-world-peer"))
        assert (await node._session_decision_hook(
            REMOTE_PEER, "joiner-node", profile, NodeConn.P2P_PUBLIC, claims)).accepted

        other = full_profile_dict("joiner-node", private_peer_id="other-world-peer",
                                  private_peer_addresses=["/ip4/10.0.0.9/tcp/4001/p2p/other-world-peer"])
        decision = await node._session_decision_hook(
            REMOTE_PEER, "joiner-node", other, NodeConn.P2P_PUBLIC, None)
        assert decision.accepted
        assert {"joiner-node-world-peer", "other-world-peer"} <= conn.world_agents_set
        assert "joiner-node-world-peer" not in conn.removed_calls

    asyncio.run(_scenario())


def test_world_host_decision_rejects_joiner_without_private_addr():
    """B5: a world-join whose profile carries no private addresses is rejected at the
    public decision (NOT_AUTHORIZED) instead of asserting, and is never pre-admitted to
    the world lists (which under python -O would leak a None-addressed member)."""
    async def _scenario():
        world = FakeWorld()
        node, conn = build_agent_node_shell(NODE_PEER, world=world)
        profile = full_profile_dict("joiner-node", private_peer_addresses=[])
        _, (_, real_hash) = NodeProfile.from_dict(profile).verify_cv_hash("")
        claims = VerifiedTokenClaims(node_id="joiner-node", cv_hash=real_hash,
                                     p2p_peers=("joiner-node-pub-peer", "joiner-node-world-peer"))
        decision = await node._session_decision_hook(
            REMOTE_PEER, "joiner-node", profile, NodeConn.P2P_PUBLIC, claims)
        assert decision.accepted is False
        assert decision.reason_code == NOT_AUTHORIZED
        assert "joiner-node-world-peer" not in conn.world_agents_set
        assert "joiner-node-world-peer" not in conn.world_masters_set

    asyncio.run(_scenario())


def test_world_host_register_rejects_joiner_without_private_addr():
    """B5: a joiner whose world Session establishes but whose profile has no private
    addresses is rejected at registration instead of asserting (which tore the world
    host's loop down, and under python -O listed a None-addressed member). It is
    unlisted (the pre-admission removed) and purged, never enrolled."""
    async def _scenario():
        from unaiverse.agent import Agent
        world = FakeWorld()
        node, conn = build_agent_node_shell(NODE_PEER, world=world)
        role = FakeWorld.ROLE_STR_TO_BITS["player"] | Agent.ROLE_WORLD_AGENT
        # Pre-admission the public decision would have done for this joiner.
        conn.add_to_world_agents_list(REMOTE_PEER, [], role)
        profile_b = full_profile_dict("joiner-node", private_peer_addresses=[])
        ev = SessionEvent(type=SessionEventType.ESTABLISHED, session_id="00" * 8,
                          peer_id=REMOTE_PEER, role=SessionRole.RESPONDER,
                          p2p_name=NodeConn.P2P_WORLD, data={})

        await node._world_host_register_agent(ev, profile_b)  # must not raise

        assert REMOTE_PEER not in conn.world_agents_set
        assert REMOTE_PEER not in conn.world_masters_set
        assert REMOTE_PEER in conn.removed_calls
        assert not world.added

    asyncio.run(_scenario())


def test_world_host_preadmission_provisional_until_registration():
    """The grant-time pre-admission is provisional (it carries the TTL deadline: a
    joiner that never redeems must not ghost the roster) and the world-side
    registration confirms it by dropping the deadline."""
    async def _scenario():
        world = FakeWorld()
        node, conn = build_agent_node_shell(NODE_PEER, world=world)
        profile = full_profile_dict("joiner-node")
        _, (_, real_hash) = NodeProfile.from_dict(profile).verify_cv_hash("")
        claims = VerifiedTokenClaims(node_id="joiner-node", cv_hash=real_hash,
                                     p2p_peers=("joiner-node-pub-peer", "joiner-node-world-peer"))
        decision = await node._session_decision_hook(
            REMOTE_PEER, "joiner-node", profile, NodeConn.P2P_PUBLIC, claims)
        assert decision.accepted
        assert "joiner-node-world-peer" in conn.world_preadmissions   # provisional

        ev = SessionEvent(type=SessionEventType.ESTABLISHED, session_id="00" * 8,
                          peer_id="joiner-node-world-peer", role=SessionRole.RESPONDER,
                          p2p_name=NodeConn.P2P_WORLD, data={})
        await node._world_host_register_agent(ev, profile)
        assert "joiner-node-world-peer" in conn.world_agents_set
        assert "joiner-node-world-peer" not in conn.world_preadmissions   # confirmed

    asyncio.run(_scenario())


def test_pool_removal_while_established_keeps_enrolment_during_grace():
    """Contract of __handle_network_connections on a pool drop: the session moves
    to RECONNECTING (grace) and the peer is KEPT in hosted (all_agents), so an
    interaction arriving mid-grace is not rejected as coming from an unknown node.
    The transport-level pool state is cleared by on_peer_disconnected (involuntary
    pool-sync, internal to the Coordinator), NOT by a bare-remove from Node; the
    hosted purge is driven by the terminal LOST/CLOSED dispatch (see
    test_session_lost_while_peer_still_in_pool_purges_and_quarantines)."""
    async def _scenario():
        node, conn, remote_sm, remote_inbox = _agent_pair()
        await _establish_inbound(node, conn, remote_sm, remote_inbox)

        conn.next_removed = {NodeConn.OUT_PUBLIC: {REMOTE_PEER}}
        await node._Node__handle_network_connections()

        sess = conn._sm_public.get_session(REMOTE_PEER)
        assert sess is not None and sess.state == SessionState.RECONNECTING
        # Enrolment survives the grace window...
        assert REMOTE_PEER in node.hosted.all_agents
        assert REMOTE_PEER not in node.hosted.removed
        # ...the pool state is cleared by on_peer_disconnected (not a Node bare-remove).
        assert REMOTE_PEER in conn.evicted_on_disconnect
        assert REMOTE_PEER not in conn.removed_calls

    asyncio.run(_scenario())


def test_resumed_session_keeps_enrolment_without_readd():
    """With the grace-aware drop the agent is NOT purged when the link blips: it
    stays enrolled through RECONNECTING, so a successful RESUME finds it already
    present and does not re-add it (no double enrolment). This is the happy
    reconnect path; purge-then-readd only happens on a terminal LOST followed by a
    fresh handshake."""
    async def _scenario():
        node, conn, remote_sm, remote_inbox = _agent_pair()
        await _establish_inbound(node, conn, remote_sm, remote_inbox)
        assert REMOTE_PEER in node.hosted.all_agents

        # Transport drop seen by both sides: enrolment survives the grace window.
        conn.next_removed = {NodeConn.OUT_PUBLIC: {REMOTE_PEER}}
        await node._Node__handle_network_connections()
        await remote_sm.on_peer_disconnected(NODE_PEER)
        assert REMOTE_PEER in node.hosted.all_agents
        assert REMOTE_PEER not in node.hosted.removed
        assert conn._sm_public.get_session(REMOTE_PEER).state == SessionState.RECONNECTING

        # The remote initiator re-dials and RESUMEs; the node pump processes it. Design A:
        # the first tick launches the re-dial, a second (after the task runs) sends RESUME.
        await remote_sm.tick()
        await asyncio.sleep(0.05)
        await remote_sm.tick()
        await node._Node__handle_network_messages()   # RESUME → ACK + RESUMED dispatch
        await apump(remote_inbox, remote_sm)           # remote absorbs the ACK

        assert conn._sm_public.is_established(REMOTE_PEER)
        assert remote_sm.is_established(NODE_PEER)
        # Still enrolled, and NOT re-added (it was never purged): one add total.
        assert REMOTE_PEER in node.hosted.all_agents
        assert len(node.hosted.added) == 1

    asyncio.run(_scenario())


def test_session_lost_while_peer_still_in_pool_purges_and_quarantines():
    """A session that rides its grace period to LOST (peer never removed from
    the pool) must purge the peer and quarantine it on the next pump."""
    async def _scenario():
        node, conn, remote_sm, remote_inbox = _agent_pair()
        await _establish_inbound(node, conn, remote_sm, remote_inbox)

        await conn._sm_public.on_peer_disconnected(REMOTE_PEER)
        sess = conn._sm_public.get_session(REMOTE_PEER)
        assert sess.state == SessionState.RECONNECTING
        sess._state_entered_at -= 60.0   # blow past the 45s grace window

        await node._Node__handle_network_messages()   # tick → LOST → dispatch
        assert conn._sm_public.get_session(REMOTE_PEER) is None
        assert REMOTE_PEER in conn._sm_public._quarantine
        assert REMOTE_PEER in node.hosted.removed
        assert REMOTE_PEER in conn.removed_calls

    asyncio.run(_scenario())


def test_added_diff_kicks_resume_for_reconnecting_session():
    """A peer that reappears in the pool diff while its session is RECONNECTING
    must get the on_peer_reconnected signal, which starts the way back at once: the
    Session here is a responder, so its probe becomes due."""
    async def _scenario():
        node, conn, remote_sm, remote_inbox = _agent_pair()
        await _establish_inbound(node, conn, remote_sm, remote_inbox)

        await conn._sm_public.on_peer_disconnected(REMOTE_PEER)
        sess = conn._sm_public.get_session(REMOTE_PEER)
        assert sess.state == SessionState.RECONNECTING and sess.role == SessionRole.RESPONDER
        # As after a probe that left a moment ago: the next one is a period away.
        sess._last_ping_at = time.monotonic()

        conn.peer_id_to_pool[REMOTE_PEER] = NodeConn.IN_PUBLIC
        conn.next_added = {NodeConn.IN_PUBLIC: {REMOTE_PEER}}
        await node._Node__handle_network_connections()

        assert sess._last_ping_at == float("-inf")

    asyncio.run(_scenario())


def _world_grant_blob():
    world_profile = full_profile_dict("world-node-id")
    return {
        "grant_id": "ab" * 32,
        "expiry_unix_ms": int(time.time() * 1000) + 30000,
        "world_profile_bytes": json.dumps(world_profile).encode("utf-8"),
        "role": 1,
        "rendezvous_tag": 0,
        "code_bundle": b"",
        "initial_stats": b"",
    }


@pytest.mark.parametrize("failure_mode", ["dial_failed", "quarantined"])
def test_world_handoff_failure_cleans_pending_maps(failure_mode):
    """When the world-side initiate fails (unreachable world / quarantined
    peer), the handoff bookkeeping must be rolled back."""
    async def _scenario():
        node, conn = build_agent_node_shell(NODE_PEER)
        # The join in progress, as ask_to_join_world records it: its grant is redeemed.
        node.joining_world_info = {"world_public_peer_id": "public-world-peer", "options": {}}
        world_world_peer = "world-node-id-world-peer"
        if failure_mode == "dial_failed":
            conn.connect_returns = (None, False)
        else:
            conn._sm_world._quarantine[world_world_peer] = time.monotonic() + 30.0

        await node._start_world_handoff("public-world-peer", _world_grant_blob())

        if failure_mode == "dial_failed":
            # Design A: the dial now fails asynchronously over several attempts behind a
            # time-based backoff gate. Drive the world SM's ticks (advancing `now` past each
            # gate) until a terminal FAILED, dispatching its events so the rollback runs.
            now = time.monotonic()
            for _ in range(20):
                await asyncio.sleep(0)
                evs = await conn._sm_world.tick(now)
                for ev in evs:
                    await node._dispatch_session_event(ev)
                if any(e.type == SessionEventType.FAILED for e in evs):
                    break
                now += 31.0

        assert conn._sm_world.is_established(world_world_peer) is False
        assert node.conn._pending_world_handoff == {}
        assert node.conn._world_to_public == {}

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
        await _reap_remote_dial(remote_sm)            # reap the dial → remote HELLO on the wire
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
    """When a joiner's world Session establishes, the world host must register it with the role from
    its own pre-admission record (the grant is unsigned, so the role is not
    echoed/trusted from the join). Master bits land the joiner in the masters list."""
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
        # The authoritative role comes from the host pre-admission (the public
        # decision), not the join echo. Simulate that pre-admission here.
        conn.add_to_world_masters_list(REMOTE_PEER, [], role)
        blob = {"role": role,
                "expiry_unix_ms": int(time.time() * 1000) + 30_000}

        await remote_sm.initiate(node_world_peer, ["/ip4/127.0.0.1/tcp/0"],
                                 grant=blob)
        await _reap_remote_dial(remote_sm)            # reap the dial → remote HELLO on the wire
        await node._Node__handle_network_messages()   # HELLO → HELLO_ACK
        await apump(remote_inbox, remote_sm)          # HELLO_ACK → PROFILE_OFFER
        await node._Node__handle_network_messages()   # OFFER → host ESTABLISHED + register, DECISION sent
        remote_events = await apump(remote_inbox, remote_sm)  # DECISION → joiner ESTABLISHED (no PRIVATE_JOIN)

        assert any(e.type == SessionEventType.ESTABLISHED for e in remote_events), \
            f"joiner never established: {[e.type.value for e in remote_events]}"
        assert conn._sm_world.is_established(REMOTE_PEER)
        assert REMOTE_PEER in conn.world_masters_set
        assert conn.world_roles[REMOTE_PEER] == role
        assert world.added and world.added[0][0] == REMOTE_PEER

    asyncio.run(_scenario())


def test_world_host_purges_non_preadmitted_joiner():
    """A joiner reaching PRIVATE_JOIN without a host pre-admission reads role 0
    (no world bit) and is purged, never registered. The unsigned grant carries no
    trust of its own; admission comes only from the public-decision pre-admission."""
    async def _scenario():
        world = FakeWorld()
        node, conn = build_agent_node_shell(NODE_PEER, world=world)
        node_world_peer = NODE_PEER + "-w"
        remote_sm, remote_conn, remote_inbox = make_remote_peer(
            REMOTE_PEER, "joiner-node", conn.inbox_by_p2p[NodeConn.P2P_WORLD],
            p2p_name=NodeConn.P2P_WORLD)
        conn.send_inbox = remote_inbox
        remote_conn.connect_returns = (node_world_peer, False)

        # NO pre-admission: conn.get_role(REMOTE_PEER) == 0. The joiner even claims a
        # role in the blob, but the host ignores it.
        blob = {"role": FakeWorld.ROLE_STR_TO_BITS["player"],
                "expiry_unix_ms": int(time.time() * 1000) + 30_000}
        await remote_sm.initiate(node_world_peer, ["/ip4/127.0.0.1/tcp/0"],
                                 grant=blob)
        await _reap_remote_dial(remote_sm)            # reap the dial → remote HELLO on the wire
        await node._Node__handle_network_messages()
        await apump(remote_inbox, remote_sm)
        await node._Node__handle_network_messages()
        await apump(remote_inbox, remote_sm)
        await node._Node__handle_network_messages()
        await apump(remote_inbox, remote_sm)

        assert REMOTE_PEER not in conn.world_agents_set
        assert REMOTE_PEER not in conn.world_masters_set
        assert not world.added

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
        # Role is authoritative from the host pre-admission, not the join echo.
        conn.add_to_world_agents_list(REMOTE_PEER, [], role)
        blob = {"role": role,
                "expiry_unix_ms": int(time.time() * 1000) + 30_000}
        await remote_sm.initiate(node_world_peer, ["/ip4/127.0.0.1/tcp/0"],
                                 grant=blob)
        await _reap_remote_dial(remote_sm)            # reap the dial → remote HELLO on the wire
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


def test_s18_node_holds_no_session_manager_handle():
    """Enforce the S18 contract on the Node side: Node caches no handle to the
    SessionManagers. The managers are owned by the Coordinator (conn._sm_public /
    conn._sm_world); Node drives every Session verb through conn facades. This fails if
    Node ever regrows a node._sm_* attribute."""
    node, conn = build_agent_node_shell(NODE_PEER)
    assert not hasattr(node, "_sm_public")
    assert not hasattr(node, "_sm_world")
    # The managers live on the Coordinator instead.
    assert conn._sm_public is not None and conn._sm_world is not None


def test_failed_join_body_tears_down_world_session_and_rearms():
    """When the grant is redeemed but the join body fails (bad code bundle, subscribe
    failure...), the dispatcher must not stop at retiring the carrier: it tears the
    half-join down — GOAWAY the world Session (the host unlists us on its CLOSED),
    un-point world routing — and re-arms the rejoin. Before, the node was left with
    an orphaned ESTABLISHED world Session, a stale set_world and no retry."""
    async def _scenario():
        node, conn = build_agent_node_shell(NODE_PEER)
        node.keep_rejoining = True
        node.rejoining_time = -1.
        node.rejoining_kwargs = {}
        node.joining_world_info = {"world_public_peer_id": "host-pub",
                                   "options": {"world_name": "w"}}
        # What start_world_handoff left behind: the correlation with the carrier of the
        # join in progress, and the routing pointed at the world host.
        conn._handoff.start("host-pub", REMOTE_PEER)
        conn.set_world(REMOTE_PEER)

        async def _join_fails(world_peer_id, grant):
            return False
        node._join_world_after_session = _join_fails

        ev = SessionEvent(type=SessionEventType.ESTABLISHED, session_id="00" * 8,
                          peer_id=REMOTE_PEER, role=SessionRole.INITIATOR,
                          p2p_name=NodeConn.P2P_WORLD,
                          data={"profile_b": {}, "grant": {"role": 1}})
        await node._dispatch_session_event(ev)

        assert REMOTE_PEER in conn.removed_calls      # world session purged (reasoned)
        assert None in conn.set_world_calls           # routing un-pointed
        assert node.rejoining_time > 0.               # retry armed (involuntary end)

    asyncio.run(_scenario())


def _world_established(world_peer_id, grant):
    """The ESTABLISHED of a world Session opened by a handoff, as the joiner sees it."""
    return SessionEvent(type=SessionEventType.ESTABLISHED, session_id="00" * 8,
                        peer_id=world_peer_id, role=SessionRole.INITIATOR,
                        p2p_name=NodeConn.P2P_WORLD, data={"profile_b": {}, "grant": grant})


def _record_join_body(node):
    """Replace the join body with a recorder that reports a successful join."""
    calls = []

    async def _join(world_peer_id, grant):
        calls.append((world_peer_id, grant))
        return True
    node._join_world_after_session = _join
    return calls


@pytest.mark.parametrize("intent_carrier", ["host-pub", None])
def test_world_session_of_the_join_in_progress_runs_the_join_body(intent_carrier):
    """The world Session opened by the grant of the join in progress runs the join body
    and retires its carrier. A join that has not resolved its carrier yet (None) is the
    same request, so it takes the Session as its own."""
    async def _scenario():
        node, conn = build_agent_node_shell(NODE_PEER)
        node.joining_world_info = {"world_public_peer_id": intent_carrier, "options": {}}
        conn._handoff.start("host-pub", REMOTE_PEER)
        conn.set_world(REMOTE_PEER)
        calls = _record_join_body(node)

        grant = {"role": 1}
        await node._dispatch_session_event(_world_established(REMOTE_PEER, grant))

        assert calls == [(REMOTE_PEER, grant)]
        assert conn._world_to_public == {}                 # carrier retired, correlation dropped
        assert REMOTE_PEER not in conn.removed_calls       # the world Session stays
        assert conn.get_world_node_peer_id() == REMOTE_PEER

    asyncio.run(_scenario())


@pytest.mark.parametrize("newer_join", ["other_world", "given_up"])
def test_world_session_of_a_replaced_join_does_not_run_the_join_body(newer_join):
    """A grant was redeemed, then the join was replaced by one to another world (or given
    up) while its world Session was still coming up. When that Session establishes the
    node must not enter the world: no join body, carrier retired, world Session closed
    (the host registered us on its side), routing un-pointed, and no rejoin armed."""
    async def _scenario():
        node, conn = build_agent_node_shell(NODE_PEER)
        node.keep_rejoining = True
        node.rejoining_time = -1.
        conn._handoff.start("old-host-pub", REMOTE_PEER)   # the handoff of the first join
        conn.set_world(REMOTE_PEER)
        node.joining_world_info = ({"world_public_peer_id": "new-host-pub", "options": {}}
                                   if newer_join == "other_world" else None)
        calls = _record_join_body(node)

        await node._dispatch_session_event(_world_established(REMOTE_PEER, {"role": 1}))

        assert calls == []
        assert node.hosted.accepted_roles == []            # never entered the world
        assert conn._world_to_public == {} and conn._pending_world_handoff == {}
        assert REMOTE_PEER in conn.removed_calls           # world Session closed
        assert conn.get_world_node_peer_id() is None       # it pointed at the old world
        assert node.rejoining_time == -1.

    asyncio.run(_scenario())


def test_closing_an_unwanted_world_session_keeps_the_routing_of_the_newer_join():
    """By the time the old world Session establishes, the newer join may already have
    pointed the routing at its own world: closing the old one must not clear it."""
    async def _scenario():
        node, conn = build_agent_node_shell(NODE_PEER)
        conn._handoff.start("old-host-pub", REMOTE_PEER)
        conn._handoff.start("new-host-pub", "new-world-peer")
        conn.set_world("new-world-peer")
        node.joining_world_info = {"world_public_peer_id": "new-host-pub", "options": {}}
        calls = _record_join_body(node)

        await node._dispatch_session_event(_world_established(REMOTE_PEER, {"role": 1}))

        assert calls == []
        assert conn.get_world_node_peer_id() == "new-world-peer"
        assert conn._world_to_public == {"new-world-peer": "new-host-pub"}   # the newer handoff is intact

    asyncio.run(_scenario())


@pytest.mark.parametrize("intent, redeemed", [
    ({"world_public_peer_id": "public-world-peer", "options": {}}, True),
    ({"world_public_peer_id": None, "options": {}}, True),
    ({"world_public_peer_id": "another-world-pub", "options": {}}, False),
    (None, False),
])
def test_only_the_grant_of_the_join_in_progress_is_redeemed(intent, redeemed):
    """A grant opens its world Session only when it comes over the carrier of the join in
    progress. One that arrives after the join was replaced (or given up) is not redeemed:
    no correlation, no routing change, no world dial, and its carrier is closed."""
    async def _scenario():
        node, conn = build_agent_node_shell(NODE_PEER)
        node.joining_world_info = intent
        conn.connect_returns = ("world-node-id-world-peer", False)

        await node._start_world_handoff("public-world-peer", _world_grant_blob())

        if redeemed:
            assert conn._pending_world_handoff == {"public-world-peer": "world-node-id-world-peer"}
            assert conn.get_world_node_peer_id() == "world-node-id-world-peer"
            assert conn._sm_world.get_session("world-node-id-world-peer") is not None
            assert "public-world-peer" not in conn.removed_calls
        else:
            assert conn._pending_world_handoff == {}
            assert conn.set_world_calls == []
            assert conn._sm_world.get_session("world-node-id-world-peer") is None
            assert "public-world-peer" in conn.removed_calls   # carrier closed

    asyncio.run(_scenario())
