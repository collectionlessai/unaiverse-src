"""Tests for the disengage / GOAWAY teardown path (BUG 3, fixed).

Area: ``DisengageGoawayBug``.

When an agent honours a disengagement request *with* a disconnect
(``Agent.disengage(disconnect_too=True)``), the teardown is graceful: the
disengaged peer must receive a GOAWAY (Session.Closed) instead of being left to
time out (RECONNECTING -> FAILED via liveness). The reference is ``Node.leave``
(node.py:825), which calls ``conn.close_session(peer_id, WORLD_DISENGAGEMENT)``
*before* tearing down.

The fix (agent.py:689): ``disengage`` calls
``self.node_purge_fcn(_requester, reason=WORLD_DISENGAGEMENT)``. ``node_purge_fcn``
is wired to ``Node.__purge``; with a non-None reason the guard at node.py:2685
(``if reason is not None and not keep_connection``) is True, so the
``close_session``/GOAWAY branch runs. The ``reason=None`` path stays the
intentional "silent purge" sentinel for internal drops.

Everything here is pure in-process object manipulation: no libp2p, no
SessionManager, no transport. The raw coroutines are driven via
``Agent.disengage.__wrapped__`` (the @action wrapper swallows exceptions and
resolves agent refs, both unwanted here) and via the name-mangled
``Node._Node__purge`` bound to a fake node.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from unaiverse.agent import Agent
from unaiverse.networking.node.node import Node
from unaiverse.networking.session.sess_msg import WORLD_DISENGAGEMENT

pytestmark = pytest.mark.unit


PEER = "peer-engaged-id"
OTHER = "peer-other-id"

# Raw coroutine bodies, bypassing the @action wrapper / name mangling.
_disengage = Agent.disengage.__wrapped__
_purge = getattr(Node, "_Node__purge")


def _run(coro):
    return asyncio.run(coro)


def _fake_self(*, world_agents=None, world_masters=None, engaged=None):
    """A minimal stand-in carrying exactly the attributes ``disengage`` touches."""
    return SimpleNamespace(
        world_agents={PEER: object()} if world_agents is None else world_agents,
        world_masters={} if world_masters is None else world_masters,
        _engaged_agents=set() if engaged is None else set(engaged),
        node_purge_fcn=AsyncMock(),
        _available=True,
    )


def _fake_interaction():
    """``disengage`` only reads ``.requester`` and writes ``.send_status``."""
    return SimpleNamespace(requester=PEER, send_status=True)


def _propagated_reason(spy_call):
    """Extract the reason argument the purge call carried (None if absent).

    Accepts either the (intended) keyword form ``purge(peer, reason=...)`` or a
    positional ``purge(peer, reason)``; returns None when only the peer-id was
    passed, which is the current buggy state.
    """
    if spy_call is None:
        return None
    if "reason" in spy_call.kwargs:
        return spy_call.kwargs["reason"]
    if len(spy_call.args) >= 2:
        return spy_call.args[1]
    return None


# ── Guard preconditions (early returns) ──────────────────────────────────────

def test_disengage_unknown_requester_returns_false_and_skips_purge():
    """A requester in neither world_agents nor world_masters is rejected before any teardown."""
    me = _fake_self(world_agents={}, world_masters={}, engaged={PEER})
    inter = _fake_interaction()

    ret = _run(_disengage(me, disconnect_too=True, interaction=inter))

    assert ret is False
    me.node_purge_fcn.assert_not_awaited()
    # The engaged set and the ack flag are left untouched on the unknown-agent path.
    assert me._engaged_agents == {PEER}
    assert inter.send_status is True


def test_disengage_not_previously_engaged_returns_false_and_skips_purge():
    """A known agent that was never engaged is rejected (no discard, no purge)."""
    me = _fake_self(world_agents={PEER: object()}, engaged=set())
    inter = _fake_interaction()

    ret = _run(_disengage(me, disconnect_too=True, interaction=inter))

    assert ret is False
    me.node_purge_fcn.assert_not_awaited()
    assert me._engaged_agents == set()
    assert inter.send_status is True


def test_disengage_none_interaction_returns_false():
    """With interaction=None the requester resolves to None, which is an unknown agent."""
    me = _fake_self(engaged={PEER})

    ret = _run(_disengage(me, disconnect_too=True, interaction=None))

    assert ret is False
    me.node_purge_fcn.assert_not_awaited()
    assert me._engaged_agents == {PEER}


# ── Plain disengage (no disconnect) ──────────────────────────────────────────

def test_disengage_discards_requester_and_marks_available():
    """A bare disengage drops the requester, never purges, and frees the agent."""
    me = _fake_self(engaged={PEER})
    inter = _fake_interaction()

    ret = _run(_disengage(me, disconnect_too=False, interaction=inter))

    assert ret is True
    assert me._engaged_agents == set()
    assert me._available is True
    me.node_purge_fcn.assert_not_awaited()
    # No disconnect requested -> the ack is NOT suppressed.
    assert inter.send_status is True


def test_disengage_via_world_master_membership():
    """Membership in world_masters (not world_agents) also authorizes the disengage."""
    me = _fake_self(world_agents={}, world_masters={PEER: object()}, engaged={PEER})
    inter = _fake_interaction()

    ret = _run(_disengage(me, disconnect_too=False, interaction=inter))

    assert ret is True
    assert me._engaged_agents == set()


def test_disengage_stays_unavailable_while_other_agents_remain():
    """``_available`` only flips True once the engaged set is empty."""
    me = _fake_self(engaged={PEER, OTHER})
    inter = _fake_interaction()

    ret = _run(_disengage(me, disconnect_too=False, interaction=inter))

    assert ret is True
    assert me._engaged_agents == {OTHER}
    assert me._available is False


# ── disconnect_too=True: current behaviour (documents the bug) ───────────────

def test_disengage_disconnect_too_suppresses_ack_and_purges_once():
    """disconnect_too suppresses the ack and purges the requester exactly once.

    State-side companion to the reason assertion below: the engaged set is
    cleared, the agent is freed, and the confirmation is suppressed because the
    link is about to be torn down.
    """
    me = _fake_self(engaged={PEER})
    inter = _fake_interaction()

    ret = _run(_disengage(me, disconnect_too=True, interaction=inter))

    assert ret is True
    assert me._engaged_agents == set()
    assert me._available is True
    # Ack suppressed because we are about to tear the link down.
    assert inter.send_status is False
    # The purge is awaited exactly once (the ReasonCode is asserted below).
    me.node_purge_fcn.assert_awaited_once()


def test_disengage_disconnect_too_without_interaction_still_purges():
    """disconnect_too with a None interaction cannot reach purge (requester is None)."""
    # interaction=None means _requester is None -> unknown agent -> early False.
    me = _fake_self(engaged={PEER})
    ret = _run(_disengage(me, disconnect_too=True, interaction=None))
    assert ret is False
    me.node_purge_fcn.assert_not_awaited()


# ── disconnect_too=True: CORRECT behaviour (the bug, xfail strict) ───────────

def test_disengage_disconnect_too_propagates_world_disengagement_reason():
    """The purge carries WORLD_DISENGAGEMENT so a GOAWAY goes out on a graceful disengage.

    Mirrors Node.leave (node.py:825), which closes the Session with
    WORLD_DISENGAGEMENT before teardown.
    """
    me = _fake_self(engaged={PEER})
    inter = _fake_interaction()

    ret = _run(_disengage(me, disconnect_too=True, interaction=inter))

    assert ret is True
    me.node_purge_fcn.assert_awaited_once()
    reason = _propagated_reason(me.node_purge_fcn.call_args)
    assert reason == WORLD_DISENGAGEMENT


# ── Node.__purge guard (the downstream consequence, green) ───────────────────
# These pin the guard at node.py:2683 that turns a reason-less purge into a
# silent drop: with reason=None the GOAWAY (close_session) branch is skipped.

def _fake_node():
    return SimpleNamespace(
        conn=SimpleNamespace(close_session=AsyncMock(return_value=[]), remove=AsyncMock()),
        hosted=SimpleNamespace(remove_agent=AsyncMock()),
        _dispatch_session_event=AsyncMock(),
    )


def test_purge_with_reason_none_skips_close_session_but_tears_down():
    """reason=None (an internal silent purge): no GOAWAY, yet the peer is still removed."""
    node = _fake_node()

    _run(_purge(node, PEER))  # reason defaults to None

    node.conn.close_session.assert_not_awaited()
    node._dispatch_session_event.assert_not_awaited()
    node.hosted.remove_agent.assert_awaited_once_with(PEER)
    node.conn.remove.assert_awaited_once_with(PEER)


def test_purge_with_reason_sends_goaway_via_close_session():
    """A non-None reason routes through conn.close_session with that exact reason."""
    node = _fake_node()

    _run(_purge(node, PEER, reason=WORLD_DISENGAGEMENT))

    node.conn.close_session.assert_awaited_once_with(PEER, WORLD_DISENGAGEMENT)
    node.hosted.remove_agent.assert_awaited_once_with(PEER)
    node.conn.remove.assert_awaited_once_with(PEER)


def test_purge_dispatches_each_session_event_from_close_session():
    """Events returned by close_session are dispatched one by one before teardown."""
    node = _fake_node()
    node.conn.close_session = AsyncMock(return_value=["ev-closed"])

    _run(_purge(node, PEER, reason=WORLD_DISENGAGEMENT))

    node._dispatch_session_event.assert_awaited_once_with("ev-closed")


def test_purge_keep_connection_skips_close_session_and_remove():
    """keep_connection=True suppresses both the GOAWAY and the connection removal."""
    node = _fake_node()

    _run(_purge(node, PEER, keep_connection=True, reason=WORLD_DISENGAGEMENT))

    node.conn.close_session.assert_not_awaited()
    node.conn.remove.assert_not_awaited()
    node.hosted.remove_agent.assert_awaited_once_with(PEER)
