"""A joiner holds its dials toward the other members until it reads a roster that lists
it, newer than its grant, with a limit of 5 s from the end of the join body and of
30 s from the start of the handoff.

A real ``NodeConn`` on two fake transports, driven through the calls ``node.py`` makes
to the Coordinator; the transports record the dials and serve a roster set by the test.
The monotonic clock is the test's, so the limits are crossed by moving it."""

import time
import asyncio

import pytest

from unaiverse.networking.node.connpool import NodeConn
from unaiverse.networking.session import SessionState
from unaiverse.networking.session.sess_msg import ProfileDecisionResult

from node_harness import WireP2P

HOST_PUBLIC = "12D3KooWhost0public"
HOST_WORLD = "12D3KooWhost0world"
JOINER_WORLD = "12D3KooWjoiner0world"
MEMBER = "12D3KooWmember0world"
OTHER = "12D3KooWother0world"
GRANT_TAG = 40


def _addr(peer_id: str) -> str:
    return f"/ip4/10.0.0.9/tcp/4001/p2p/{peer_id}"


def _info(peer_id: str) -> dict:
    return {"id": peer_id, "addrs": [_addr(peer_id)], "misc": 1}


class _Monotonic:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def mono(monkeypatch):
    clock = _Monotonic()
    monkeypatch.setattr(time, "monotonic", clock)
    return clock


async def _accept(*args):
    return ProfileDecisionResult(accepted=True)


class _Joiner:
    def __init__(self):
        self.p2p_public = WireP2P("12D3KooWjoiner0public", "pub")
        self.p2p_world = WireP2P(JOINER_WORLD, "prv")
        self.dials = []
        self.roster = None
        for wire in (self.p2p_public, self.p2p_world):
            wire.connect = self._connect
        self.p2p_world.get_rendezvous_peers_info = lambda: self.roster
        self.conn = NodeConn(max_connections=32, is_world_node=False, public_key=None,
                             token="joiner-token", local_node_id="joiner-node")
        self.conn.attach_session_policy(p2p_u=self.p2p_public, p2p_w=self.p2p_world,
                                        local_profile_factory=lambda: {}, decision_hook=_accept)

    async def _connect(self, addresses):
        self.dials.append(addresses[0].rsplit("/p2p/", 1)[-1])
        return None, False

    async def start_handoff(self) -> None:
        grant = {"rendezvous_tag": GRANT_TAG, "expiry_unix_ms": int(time.time() * 1000) + 30000}
        await self.conn.start_world_handoff(HOST_PUBLIC, HOST_WORLD, [_addr(HOST_WORLD)], grant, False)
        await asyncio.sleep(0)

    async def join(self) -> None:
        """The handoff, the members of the grant, and the end of the join body."""
        await self.start_handoff()
        self.conn.rendezvous_tag = GRANT_TAG - 1
        self.conn.set_world_agents_list([_info(MEMBER), _info(OTHER)])
        await self.conn.complete_world_handoff(HOST_WORLD)

    async def dial(self, peer_id: str, p2p_name: str = NodeConn.P2P_WORLD) -> None:
        await self.conn.initiate_session(p2p_name, peer_id, [_addr(peer_id)])
        await asyncio.sleep(0)

    def publish(self, tag: int, peer_ids: list) -> None:
        self.roster = {"update_count": tag, "peers": [_info(p) for p in peer_ids]}

    async def cycle(self) -> None:
        """The rendezvous, the reconcile and the pump, in the order of the main loop."""
        await self.conn.set_world_agents_and_world_masters_lists_from_rendezvous()
        await self.conn.update()
        await self.conn.pump_sessions({}, now=time.monotonic())
        await asyncio.sleep(0)

    def state_of(self, peer_id: str):
        sess = self.conn._sm_world.get_session(peer_id)
        return None if sess is None else sess.state


def test_a_dial_to_a_member_waits_for_a_roster_newer_than_the_grant_that_lists_us(mono):
    async def _scenario():
        joiner = _Joiner()
        await joiner.join()
        await joiner.dial(MEMBER)
        await joiner.cycle()
        await joiner.cycle()
        assert MEMBER not in joiner.dials
        assert joiner.state_of(MEMBER) == SessionState.KNOWN

        joiner.publish(GRANT_TAG + 1, [MEMBER, OTHER, JOINER_WORLD])
        await joiner.cycle()
        assert joiner.dials.count(MEMBER) == 1
        assert joiner.state_of(MEMBER) == SessionState.CONNECTING

    asyncio.run(_scenario())


def test_a_roster_of_the_grant_cycle_or_one_without_us_does_not_end_the_wait(mono):
    """The roster published in the cycle of the grant was made before the decision, and
    can list the old entry of a node that joins again with the same peer-id."""
    async def _scenario():
        joiner = _Joiner()
        await joiner.join()
        await joiner.dial(MEMBER)

        joiner.publish(GRANT_TAG, [MEMBER, OTHER, JOINER_WORLD])
        await joiner.cycle()
        joiner.publish(GRANT_TAG + 1, [MEMBER, OTHER])
        await joiner.cycle()
        assert MEMBER not in joiner.dials

        joiner.publish(GRANT_TAG + 2, [MEMBER, OTHER, JOINER_WORLD])
        await joiner.cycle()
        assert MEMBER in joiner.dials

    asyncio.run(_scenario())


def test_with_no_roster_the_dial_goes_out_five_seconds_after_the_join(mono):
    async def _scenario():
        joiner = _Joiner()
        await joiner.join()
        await joiner.dial(MEMBER)
        mono.now += 4.9
        await joiner.cycle()
        assert MEMBER not in joiner.dials

        mono.now += 0.2
        await joiner.cycle()
        assert MEMBER in joiner.dials
        await joiner.dial(OTHER)                         # the wait is over
        assert OTHER in joiner.dials

    asyncio.run(_scenario())


def test_with_no_end_of_the_join_the_wait_ends_thirty_seconds_after_the_handoff_began(mono):
    """complete_world_handoff starts the 5 s; if it never comes, the wait still ends."""
    async def _scenario():
        joiner = _Joiner()
        await joiner.start_handoff()
        joiner.conn.set_world_agents_list([_info(MEMBER)])
        await joiner.dial(MEMBER)
        mono.now += 29.9
        await joiner.cycle()
        assert MEMBER not in joiner.dials

        mono.now += 0.2
        await joiner.cycle()
        assert MEMBER in joiner.dials

    asyncio.run(_scenario())


def test_the_world_host_and_the_public_instance_are_never_held(mono):
    async def _scenario():
        joiner = _Joiner()
        await joiner.join()
        assert HOST_WORLD in joiner.dials                # the redemption of the grant
        await joiner.conn._sm_world.drop(HOST_WORLD)
        joiner.dials.clear()
        await joiner.dial(HOST_WORLD)                    # the world host dialled again
        assert HOST_WORLD in joiner.dials
        await joiner.dial("12D3KooWsomeone0public", p2p_name=NodeConn.P2P_PUBLIC)
        assert "12D3KooWsomeone0public" in joiner.dials

    asyncio.run(_scenario())


def test_a_held_peer_that_left_the_roster_is_not_dialled(mono):
    async def _scenario():
        joiner = _Joiner()
        await joiner.join()
        await joiner.dial(MEMBER)
        joiner.publish(GRANT_TAG + 1, [OTHER, JOINER_WORLD])
        await joiner.cycle()
        assert MEMBER not in joiner.dials
        assert joiner.state_of(MEMBER) is None

    asyncio.run(_scenario())


def test_leaving_the_world_drops_the_wait(mono):
    async def _scenario():
        joiner = _Joiner()
        await joiner.join()
        await joiner.dial(MEMBER)
        joiner.conn.set_world(None)
        await joiner.dial(OTHER)
        assert OTHER in joiner.dials                     # at once, not at the deadline
        mono.now += 60.
        await joiner.cycle()
        assert MEMBER not in joiner.dials

    asyncio.run(_scenario())
