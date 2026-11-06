"""Real-libp2p disengagement / GOAWAY test (opt-in: ``-m p2p``).

Pins the bug-3 symptom on the real wire: how a close reaches the *remote* peer
depends entirely on whether the close is graceful or a bare transport drop.

  * A GRACEFUL close (``close_session`` with a reason) sends a GOAWAY carrying
    that reason. The remote processes it and lands in a terminal ``CLOSED``
    state, observing a ``CLOSED`` event with the reason string — a clean
    disengagement.
  * A bare transport drop (no GOAWAY) only surfaces to the remote as a transport
    diff. The remote can't tell intent from accident, so it enters
    ``RECONNECTING`` and rides the 45s grace window toward ``FAILED``.

``disengage(disconnect_too=True)`` takes the bare-drop path (it purges the
Session without emitting a GOAWAY), which is why a disengaged peer falls into the
RECONNECTING->FAILED loop instead of seeing a GOAWAY. The two tests below pin the
two ends of that fork over two real ``P2P`` instances, so the contrast is wire
truth, not a mock.

Run with:  ``uv run pytest -m p2p -v``  (needs a built ``unailib.*.so``).
"""

import asyncio

import pytest

from unaiverse.networking.session import SessionState, SessionEventType
from unaiverse.networking.session.sess_msg import WORLD_DISENGAGEMENT
from unaiverse.networking.node.connpool import NodeConn

from realnet_helpers import pick_dialable_addr, pump_until

pytestmark = pytest.mark.p2p


def test_graceful_goaway_drives_remote_to_closed(real_nodes):
    """alice closes the session GRACEFULLY (WORLD_DISENGAGEMENT). The GOAWAY
    reaches bob over the real wire: bob's Session must reach the terminal CLOSED
    state and emit a CLOSED event carrying the reason — and NOT slip into
    RECONNECTING / LOST (those are the bare-drop symptoms this test rules out)."""
    a_p2p, a_conn, a_sm = real_nodes.node("alice")
    b_p2p, b_conn, b_sm = real_nodes.node("bob")
    bob_addr = pick_dialable_addr(b_p2p)

    targets = [(a_conn, a_sm, NodeConn.P2P_PUBLIC), (b_conn, b_sm, NodeConn.P2P_PUBLIC)]
    captured = {}
    # Retained handle to bob's real Session FSM. The SM evicts a terminal session
    # from its registry the moment it reaches CLOSED (_post_dispatch_cleanup drops
    # only FAILED/CLOSED), so get_session() returns None after the drain. We grab
    # the object while ESTABLISHED and assert on it directly: _set_state(CLOSED)
    # mutates this same object in place, so the named SessionState.CLOSED is still
    # observable on the wire FSM after it leaves the registry.
    b_sess = {}

    async def _scenario():
        nonlocal captured
        await asyncio.wait_for(a_sm.initiate(b_p2p.peer_id, [bob_addr]), timeout=10.0)
        await pump_until(targets,
                         predicate=lambda: a_sm.is_established(b_p2p.peer_id)
                         and b_sm.is_established(a_p2p.peer_id),
                         timeout=20.0)
        b_sess["fsm"] = b_sm.get_session(a_p2p.peer_id)  # capture while ESTABLISHED
        # Graceful close — alice emits a GOAWAY carrying the disengagement reason.
        await a_sm.close_session(b_p2p.peer_id, WORLD_DISENGAGEMENT)
        # Drain until bob's session leaves ESTABLISHED: it processed the GOAWAY and
        # finalized its close in the same drain round, so the CLOSED event lands in
        # the returned dict.
        captured = await pump_until(
            targets,
            predicate=lambda: not b_sm.is_established(a_p2p.peer_id),
            timeout=20.0)

    asyncio.run(_scenario())

    # The remote (bob) reaches the terminal CLOSED state — a clean disengagement.
    assert b_sess["fsm"].state == SessionState.CLOSED, \
        f"bob expected CLOSED, got {b_sess['fsm'].state.value}"
    # A terminal CLOSED is evicted from the registry: corroborates terminal disposition.
    assert b_sm.get_session(a_p2p.peer_id) is None, \
        "bob's CLOSED session should be evicted from the registry"

    bob_events = captured.get(b_sm, [])
    closed = [e for e in bob_events if e.type == SessionEventType.CLOSED]
    assert len(closed) >= 1, \
        f"bob never saw CLOSED; events={[e.type.value for e in bob_events]}"
    assert "WORLD_DISENGAGEMENT" in closed[0].data.get("reason", ""), \
        f"bob's CLOSED reason missing WORLD_DISENGAGEMENT; reason={closed[0].data.get('reason')!r}"
    # A graceful GOAWAY must NOT look like a dropped link to the remote.
    assert not any(e.type == SessionEventType.RECONNECTING for e in bob_events), \
        f"bob wrongly entered RECONNECTING on a graceful close; events={[e.type.value for e in bob_events]}"
    assert not any(e.type == SessionEventType.LOST for e in bob_events), \
        f"bob wrongly saw LOST on a graceful close; events={[e.type.value for e in bob_events]}"


def test_bare_transport_drop_drives_remote_to_reconnecting(real_nodes):
    """Same establish, but no GOAWAY: only the transport-diff disconnect signal
    reaches bob (the bare-drop path that disengage(disconnect_too=True) takes).
    bob can't distinguish intent from accident, so his Session must enter
    RECONNECTING (not CLOSED), arming the 45s reconnect window.

    CRITICAL: we do NOT pump_until after feeding the disconnect signal. The real
    transport between the two in-process P2P instances is still up, so the
    RECONNECTING loop would re-dial and RESUME, healing the session and
    falsifying the assert. We assert synchronously on the events returned by the
    on_peer_disconnected call itself (mirrors test_reconnect_realnet.py:52-55,
    already green)."""
    a_p2p, a_conn, a_sm = real_nodes.node("alice")
    b_p2p, b_conn, b_sm = real_nodes.node("bob")
    bob_addr = pick_dialable_addr(b_p2p)

    targets = [(a_conn, a_sm, NodeConn.P2P_PUBLIC), (b_conn, b_sm, NodeConn.P2P_PUBLIC)]
    b_events = []

    async def _scenario():
        nonlocal b_events
        await asyncio.wait_for(a_sm.initiate(b_p2p.peer_id, [bob_addr]), timeout=10.0)
        await pump_until(targets,
                         predicate=lambda: a_sm.is_established(b_p2p.peer_id)
                         and b_sm.is_established(a_p2p.peer_id),
                         timeout=20.0)
        # Feed bob the bare-drop signal the Node loop derives from its pool diff —
        # the bare-drop path, no GOAWAY. Capture the events synchronously; do NOT
        # pump afterwards (see the docstring) or the live transport would RESUME.
        b_events = await b_sm.on_peer_disconnected(a_p2p.peer_id)

    asyncio.run(_scenario())

    # The remote (bob) enters RECONNECTING, NOT a terminal close.
    b_sess = b_sm.get_session(a_p2p.peer_id)
    assert b_sess.state == SessionState.RECONNECTING, \
        f"bob expected RECONNECTING, got {b_sess.state.value}"

    reconnecting = [e for e in b_events if e.type == SessionEventType.RECONNECTING]
    assert len(reconnecting) >= 1, \
        f"bob never saw RECONNECTING; events={[e.type.value for e in b_events]}"
    assert reconnecting[0].data.get("cause") == "transport_disconnected", \
        f"bob's RECONNECTING cause wrong; cause={reconnecting[0].data.get('cause')!r}"
    # A bare drop carries no GOAWAY, so the remote must NOT reach CLOSED here.
    assert not any(e.type == SessionEventType.CLOSED for e in b_events), \
        f"bob wrongly saw CLOSED on a bare transport drop; events={[e.type.value for e in b_events]}"
