"""
Phase 0.1 of the concurrency plan: what a stalled main loop costs a node today.

The session keep-alive (PING/PONG) is answered only when the peer's main loop pumps its Sessions, and the pump runs
once per cycle, next to behave(). A node whose behave() blocks (a long action) stops answering, and its peers declare
the Session lost. These tests drive two SessionManagers over the fake transport with a simulated clock: in the first
the responder never pumps for two minutes (it is stuck in an action), in the second it pumps every second. They
characterize the current behavior and are the acceptance reference for the loop split (Phase 2), where the pump
keeps running while the HSM is busy.
"""
import types
import pytest
import asyncio
from conftest import apump, establish_pair
from unaiverse.networking.node.connpool import NodeConn
from unaiverse.networking.session import session as session_mod
from unaiverse.networking.session import session_manager as session_manager_mod
from unaiverse.networking.session import SessionManager, SessionEventType, SessionState

pytestmark = pytest.mark.e2e

STEP_S = 1.0  # Simulated seconds between two cycles of the initiator
HORIZON_S = 120.0  # Simulated length of each scenario


class _FakeClock:
    """A monotonic clock that only moves when told to."""

    def __init__(self, start: float = 1000.0) -> None:
        """Initializes the clock.

        Args:
            start: The initial reading (default: 1000.0).
        """
        self.now = start

    def monotonic(self) -> float:
        """Reads the clock.

        Returns:
            The current simulated time.
        """
        return self.now


def _make_sm(p2p: object, local_peer_id: str, node_id: str) -> SessionManager:
    """Builds a SessionManager on the public instance of a fake transport.

    Args:
        p2p: The fake transport.
        local_peer_id: The peer id of this side.
        node_id: The node id of this side.

    Returns:
        The SessionManager.
    """
    return SessionManager(p2p=p2p, p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=local_peer_id,
                          local_node_id=node_id, local_profile_factory=lambda: {"node_id": node_id, "ver": "test"})


def _install_clock(monkeypatch: pytest.MonkeyPatch, clock: _FakeClock) -> None:
    """Makes the Session layer read the simulated clock (wall-clock reads are left untouched).

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        clock: The simulated clock.
    """
    fake_time = types.SimpleNamespace(monotonic=clock.monotonic, time=session_mod.time.time)
    monkeypatch.setattr(session_mod, "time", fake_time)
    monkeypatch.setattr(session_manager_mod, "time", fake_time)


async def _run(sm_alice: SessionManager, sm_bob: SessionManager, alice_inbox: object, bob_inbox: object,
               clock: _FakeClock, bob_pumps: bool) -> list[tuple[float, SessionEventType]]:
    """Runs the scenario: Alice cycles every STEP_S; Bob pumps and ticks too, or stays stuck (async).

    Args:
        sm_alice: The initiator's manager.
        sm_bob: The responder's manager.
        alice_inbox: What reaches Alice.
        bob_inbox: What reaches Bob.
        clock: The simulated clock.
        bob_pumps: False to keep Bob stuck (no pump, no tick) for the whole horizon.

    Returns:
        Alice's events with the simulated second they happened at.
    """
    start = clock.now
    events = []
    while clock.now - start < HORIZON_S:
        clock.now += STEP_S
        for ev in await apump(alice_inbox, sm_alice) + await sm_alice.tick(clock.now):
            events.append((clock.now - start, ev.type))
        await asyncio.sleep(0)  # Lets a re-dial task run
        if bob_pumps:
            await apump(bob_inbox, sm_bob)
            await sm_bob.tick(clock.now)
        else:
            bob_inbox.clear()  # Frames that Bob's stuck loop never reads
    return events


def test_stuck_peer_is_declared_lost(bidirectional_fakeconns: tuple, alice_peer_id: str, bob_peer_id: str,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """A peer whose loop is stuck for two minutes goes RECONNECTING after ~2 missed PINGs, then LOST."""
    alice_conn, bob_conn, alice_inbox, bob_inbox = bidirectional_fakeconns
    alice_conn.connect_returns = (bob_peer_id, False)
    clock = _FakeClock()
    _install_clock(monkeypatch, clock)

    async def _scenario() -> list[tuple[float, SessionEventType]]:
        """Establishes the pair, then runs with Bob stuck (async).

        Returns:
            Alice's timed events.
        """
        sm_alice = _make_sm(alice_conn, alice_peer_id, "alice")
        sm_bob = _make_sm(bob_conn, bob_peer_id, "bob")
        await establish_pair(sm_alice, sm_bob, alice_inbox, bob_inbox, bob_peer_id)
        return await _run(sm_alice, sm_bob, alice_inbox, bob_inbox, clock, bob_pumps=False)

    events = asyncio.run(_scenario())
    times = {t: s for s, t in reversed(events)}
    assert SessionEventType.RECONNECTING in times and SessionEventType.LOST in times, events

    # Two missed PINGs: the first PING goes out at the keep-alive period, each miss is seen after the RTT window
    period, rtt = session_mod._KEEPALIVE_PERIOD_S, session_mod._PING_RTT_TIMEOUT_S
    assert period + rtt <= times[SessionEventType.RECONNECTING] <= 2 * period + rtt + 2 * STEP_S, events

    # Then the reconnect window runs out (the stuck peer never answers the RESUME)
    assert times[SessionEventType.LOST] >= times[SessionEventType.RECONNECTING] + session_mod._TIMEOUT_RECONNECT_S - \
        STEP_S, events
    assert times[SessionEventType.LOST] <= HORIZON_S, events


def test_responsive_peer_stays_established(bidirectional_fakeconns: tuple, alice_peer_id: str, bob_peer_id: str,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """A peer that pumps every second keeps the Session ESTABLISHED for the whole two minutes."""
    alice_conn, bob_conn, alice_inbox, bob_inbox = bidirectional_fakeconns
    alice_conn.connect_returns = (bob_peer_id, False)
    clock = _FakeClock()
    _install_clock(monkeypatch, clock)

    async def _scenario() -> tuple[list[tuple[float, SessionEventType]], SessionState]:
        """Establishes the pair, then runs with Bob pumping (async).

        Returns:
            Alice's timed events and the final state of her Session.
        """
        sm_alice = _make_sm(alice_conn, alice_peer_id, "alice")
        sm_bob = _make_sm(bob_conn, bob_peer_id, "bob")
        await establish_pair(sm_alice, sm_bob, alice_inbox, bob_inbox, bob_peer_id)
        events = await _run(sm_alice, sm_bob, alice_inbox, bob_inbox, clock, bob_pumps=True)
        return events, sm_alice.get_session(bob_peer_id).state

    events, state = asyncio.run(_scenario())
    assert events == [] and state == SessionState.ESTABLISHED, events
