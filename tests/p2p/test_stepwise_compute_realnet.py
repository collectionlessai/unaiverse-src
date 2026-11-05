"""Real-libp2p stepwise-compute e2e (opt-in: ``-m p2p``).

The property under test is the whole point of the cooperative slicing: with a
heavy computation spread one chunk per tick by ``AgentBasics.stepwise``, the
SINGLE task that alternates behave and pump (the node main-loop shape) keeps
pumping, so a real-wire handshake lands DURING the computation instead of
starving. The full ``Node`` cannot boot in CI (NTP, rootserver), so the
main-loop shape is reproduced faithfully in-test: one action poll (= one chunk)
then one pump round per iteration, in one task, over two REAL ``P2P``
instances. Node B dials node A mid-computation; ESTABLISHED must land on both
sides while chunks are still left to run, within a couple of seconds of the
dial (the starved baseline was 4-13s of frozen pump per evaluation).

Run with:  ``uv run pytest -m p2p tests/p2p/test_stepwise_compute_realnet.py -q``
"""

import time
import asyncio

import pytest

from unaiverse.agent import Agent, action
from unaiverse.hsm.action import Action
from unaiverse.interaction import Interaction
from unaiverse.networking.node.connpool import NodeConn

from realnet_helpers import pick_dialable_addr

pytestmark = pytest.mark.p2p

TOTAL_CHUNKS = 200
CHUNK_S = 0.015  # ~3s of "compute", far longer than a LAN handshake needs


class ChunkyAgent(Agent):
    """Offline agent whose single action steps a compute stand-in: TOTAL_CHUNKS
    chunks of CHUNK_S busy-wait each (sleep blocks the task exactly like compute)."""

    def __init__(self):
        super().__init__(proc=None)
        self.chunks_run = 0
        self.applied = None

    def _make_gen(self):
        def _gen():
            for _ in range(TOTAL_CHUNKS):
                time.sleep(CHUNK_S)
                self.chunks_run += 1
                yield
            return 0.42

        return _gen()

    @action
    async def heavy(self, interaction: Interaction | None = None) -> bool:
        done, result = self.stepwise(interaction, self._make_gen)
        if not done:
            return False
        self.applied = result
        return True


def test_handshake_completes_during_stepwise_compute(real_nodes):
    a_p2p, a_conn, a_sm = real_nodes.node("stepwise-a")
    b_p2p, b_conn, b_sm = real_nodes.node("stepwise-b")
    a_addr = pick_dialable_addr(a_p2p)
    targets = [(a_conn, a_sm, NodeConn.P2P_PUBLIC), (b_conn, b_sm, NodeConn.P2P_PUBLIC)]

    agent = ChunkyAgent()
    inter = Interaction(action_name="heavy", requester="R", target=["T"])
    act = Action("heavy", {}, agent)
    inter.set_action_ref(act)
    result = {}

    async def _scenario():
        loop = asyncio.get_event_loop()

        # Start the computation on A: the first call runs exactly one chunk.
        assert await act(inter) == 1
        assert agent.chunks_run == 1

        # B dials A DURING the computation.
        await asyncio.wait_for(b_sm.initiate(a_p2p.peer_id, [a_addr]), timeout=10.0)
        dialed_at = loop.time()

        # Single-task driver, node main-loop shape: one action poll (= one chunk)
        # then one pump round per iteration. If the computation monopolized the
        # task (the pre-stepwise shape), the handshake could only complete after
        # ALL chunks, and the timing assertion below would fail.
        established_at = None
        established_at_chunk = None
        status = 1
        deadline = loop.time() + 20.0
        while loop.time() < deadline:
            status = await act(inter)
            if status != 1:
                break  # the generator returned: computation complete
            for conn, sm, p2p_name in targets:
                _, session = await conn.get_messages(p2p_name=p2p_name)
                for inb in session:
                    await sm.deliver(inb.peer_id, inb.protocol_id, inb.sess_msg)
                await sm.tick()
            if established_at is None and \
                    a_sm.is_established(b_p2p.peer_id) and b_sm.is_established(a_p2p.peer_id):
                established_at = loop.time()
                established_at_chunk = agent.chunks_run
            await asyncio.sleep(0.005)

        result["established_after_s"] = None if established_at is None \
            else established_at - dialed_at
        result["established_at_chunk"] = established_at_chunk
        result["final_status"] = status

    asyncio.run(_scenario())

    assert result["established_after_s"] is not None, "handshake never completed"
    assert result["established_after_s"] < 2.0  # pump alive during the compute
    # The handshake landed strictly MID-computation, not after it drained.
    assert 0 < result["established_at_chunk"] < TOTAL_CHUNKS
    assert result["final_status"] == 0
    assert agent.applied == 0.42
    assert agent.chunks_run == TOTAL_CHUNKS
