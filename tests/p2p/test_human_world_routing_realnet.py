"""Real-libp2p mesh-perimeter PUNISHMENT primitive (opt-in: ``-m p2p``).

Scoped real-wire slice of bug-2 (a human-in-world reply must reach the
broadcaster, not the world node). The full reproduction is a *topology* problem
that needs a full ``Node`` WORLD (token issuance, world roles, stdin-driven human
turns), so it cannot run on the bare realnet helpers. What CAN run on the real
wire, and is exercised here, is the punishment PRIMITIVE the world node fires when
it detects misrouted application content: ``Node.__purge(peer_id,
reason=LOCAL_ERROR)``, whose GOAWAY wire action is the close component of
``conn.drop_peer(peer_id, LOCAL_ERROR)``. That close emits a GOAWAY carrying
``LOCAL_ERROR``; the punished sender's Session must reach the terminal ``CLOSED``
state over the real transport, observing the reason — and must NOT slip into LOST
(the bare-drop symptom). We isolate the GOAWAY here (the owning manager's
``close_session``, the graceful half of drop_peer) so the clean-close property is
asserted without the transport-drop half racing it.

In scope here (real-wire truth, no mocks):
  * a world-instance ``close_session(sender, LOCAL_ERROR)`` -> GOAWAY -> the
    sender's world Session reaches ``CLOSED`` with reason ``LOCAL_ERROR``, with
    the host holding an ESTABLISHED session as the positive precondition.

Explicitly left to the layered tests above this one:
  * ``tests/test_human_routing.py`` (Level 1, xfail): the node-dispatch DECISION
    to purge — that misrouted application content at the world node triggers the
    ``__purge(..., reason=LOCAL_ERROR)`` in the first place.
  * ``tests/test_human_world_routing_e2e.py`` (Level 2, e2e): the full topology —
    the human-in-world reply must reach the broadcaster, not the world node.

Run with:  ``uv run pytest -m p2p tests/p2p/test_human_world_routing_realnet.py -q -rs``
"""

import json
import asyncio

import pytest

from unaiverse.networking.session import (
    SessionState, SessionEventType, ProfileDecisionResult, WorldGrantSpec,
)
from unaiverse.networking.session.sess_msg import LOCAL_ERROR
from unaiverse.networking.node.connpool import NodeConn

from realnet_helpers import pick_dialable_addr, pump_until

pytestmark = pytest.mark.p2p


def _host_decision_hook(peer_id, remote_node_id, remote_profile, p2p_name, verified_claims=None):
    """Host accepts on both instances (copied verbatim from
    test_world_join_realnet). The world side just accepts; the public side would
    mint a WorldGrant but is unused here (we open a PLAIN, grant-less world
    session). Both branches return ``accepted=True``."""
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


def test_world_instance_punishes_misrouted_sender_with_goaway_local_error(real_nodes):
    """A world-instance host GOAWAY-closes the sender with LOCAL_ERROR (the exact
    wire action of ``__purge(reason=LOCAL_ERROR)``). The punished sender's world
    Session must reach the terminal CLOSED state over the real transport, observe
    a CLOSED event carrying ``LOCAL_ERROR``, and NOT see LOST."""
    s_pub_p2p, s_world_p2p, s_conn, s_pub, s_world = real_nodes.world_node("sender")
    h_pub_p2p, h_world_p2p, h_conn, h_pub, h_world = real_nodes.world_node(
        "host", decision_hook=_host_decision_hook)

    host_world_addr = pick_dialable_addr(h_world_p2p)
    host_world_id = h_world_p2p.peer_id
    sender_world_id = s_world_p2p.peer_id

    # Each node contributes its WORLD triple (the public instances are idle here:
    # the session under test is a plain, grant-less world session).
    targets = [
        (s_conn, s_world, NodeConn.P2P_WORLD),
        (h_conn, h_world, NodeConn.P2P_WORLD),
    ]

    # Retained handles to the real Session FSMs. A terminal CLOSED is evicted from
    # the SM registry (_post_dispatch_cleanup drops FAILED/CLOSED), so get_session
    # returns None after the drain. We grab the sender's FSM while ESTABLISHED and
    # assert on it directly: _set_state(CLOSED) mutates this same object in place.
    s_sess = {}
    captured = {}

    async def _scenario():
        nonlocal captured
        # Plain, grant-less world session: the sender (initiator) opens directly on
        # the world instance; the host (responder) accepts via its decision_hook.
        await asyncio.wait_for(
            s_world.initiate(host_world_id, [host_world_addr]), timeout=10.0)
        await pump_until(
            targets,
            predicate=lambda: s_world.is_established(host_world_id)
            and h_world.is_established(sender_world_id),
            timeout=30.0)

        # Positive precondition: the host world SM holds an ESTABLISHED session with
        # the sender right before the punishment is fired.
        s_sess["fsm"] = s_world.get_session(host_world_id)   # capture while ESTABLISHED
        s_sess["host_state"] = h_world.get_session(sender_world_id).state

        # The punishment: the host world instance GOAWAY-closes the sender with
        # LOCAL_ERROR. This is the GOAWAY half of conn.drop_peer(peer_id, reason), the
        # wire action of Node.__purge(reason=LOCAL_ERROR); we call the owning manager's
        # close_session directly (resolved via _sm_of: the world instance is the only
        # one holding the sender) to isolate the graceful close from the transport-drop
        # half. The CLOSED arrives back on the sender via the drain.
        purge_events = await h_conn._sm_of(sender_world_id).close_session(sender_world_id, LOCAL_ERROR)
        # close_session itself returns [] (CLOSED lands later via tick/inbound).
        assert purge_events == [], f"expected no immediate events, got {purge_events}"

        # Drain until the sender leaves ESTABLISHED: it processed the GOAWAY and
        # finalized its close in the same round, so the CLOSED event lands in the dict.
        captured = await pump_until(
            targets,
            predicate=lambda: not s_world.is_established(host_world_id),
            timeout=30.0)

    asyncio.run(_scenario())

    # Precondition held: the host saw an ESTABLISHED session with the sender.
    assert s_sess["host_state"] == SessionState.ESTABLISHED, \
        f"host world session not ESTABLISHED before purge, got {s_sess['host_state'].value}"

    # The punished sender reaches the terminal CLOSED state — a clean GOAWAY close.
    assert s_sess["fsm"].state == SessionState.CLOSED, \
        f"sender expected CLOSED, got {s_sess['fsm'].state.value}"
    # And the terminal session is evicted from / left CLOSED in the registry.
    leftover = s_world.get_session(host_world_id)
    assert leftover is None or leftover.state == SessionState.CLOSED, \
        f"sender world session should be gone or CLOSED, got {leftover.state.value}"

    sender_events = captured.get(s_world, [])
    closed = [e for e in sender_events if e.type == SessionEventType.CLOSED]
    assert len(closed) >= 1, \
        f"sender never saw CLOSED; events={[e.type.value for e in sender_events]}"
    assert "LOCAL_ERROR" in closed[0].data.get("reason", ""), \
        f"sender's CLOSED reason missing LOCAL_ERROR; reason={closed[0].data.get('reason')!r}"
    # A graceful punishment GOAWAY must NOT look like a dropped link to the sender.
    assert not any(e.type == SessionEventType.LOST for e in sender_events), \
        f"sender wrongly saw LOST on a graceful punishment; events={[e.type.value for e in sender_events]}"
