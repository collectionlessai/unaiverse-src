"""Unit tests for the node shutdown boundary (area: Lifecycle).

``run_async`` returns both when the node is done for good and when the caller's cycle/time
budget simply ran out. Only the first is a shutdown. ``NodeSynchronizer`` drives every node
with ``run_async(cycles=1)`` once per synchronized cycle, so replaying the teardown on every
return meant, after each single cycle: telling the root the node is gone, leaving the world,
saying goodbye to every connected peer and closing the Go hosts. The observable damage was a
world whose member roster fell back to zero right after filling up, agents advertising ports
nobody listened on, and the next cycle dying on "instance N is not initialized or has been
closed".

So the teardown lives in ``Node.aclose()``, runs once, and is called by the two owners of a
whole node lifetime: ``Node.run()`` and the synchronizer at the end of its loop.
"""

import asyncio

import pytest

from unaiverse.networking.node.node import Node, NodeSynchronizer

pytestmark = pytest.mark.unit


class _RecordingConn:
    """A NodeConn stand-in recording the session-teardown steps, in order."""

    def __init__(self, fail: bool = False):
        self.closes = 0
        self.fail = fail
        self.steps: list[str] = []

    async def close_all_sessions(self):
        self.steps.append("goaway")

    async def drain_teardown(self, budget_s: float = 3.0):
        self.steps.append("drain")

    def close_transports(self):
        self.steps.append("close")
        self.closes += 1
        if self.fail:
            raise RuntimeError("go host refused to close")


class _Hosted:
    all_agents: dict = {}

    @staticmethod
    def save():
        pass


def _node_shell(fail: bool = False) -> Node:
    """A Node reduced to what aclose touches (no __init__: it builds real P2P hosts)."""
    node = Node.__new__(Node)
    node.conn = _RecordingConn(fail=fail)
    node.node_type = Node.WORLD
    node.world = None
    node.agent = None
    node.hosted = _Hosted()
    node.send_alive = lambda alive: None
    return node


def test_aclose_closes_the_transports():
    node = _node_shell()
    asyncio.run(node.aclose())
    assert node.conn.closes == 1


def test_aclose_drains_the_goodbyes_before_killing_the_transports():
    """The whole point of the ordering: GOAWAY, then drain the acks, THEN close.
    Closing first leaves the peers with nothing but a liveness timeout."""
    node = _node_shell()
    asyncio.run(node.aclose())
    assert node.conn.steps == ["goaway", "drain", "close"]


def test_aclose_runs_once():
    """The owners call it unconditionally and run_async may have called it already:
    a second pass must not re-send the goodbyes nor re-close the hosts."""
    node = _node_shell()
    asyncio.run(node.aclose())
    asyncio.run(node.aclose())
    assert node.conn.closes == 1


def test_aclose_reaches_the_transport_close_after_a_failing_step():
    """Each step is guarded on its own: a goodbye that raises must not cost the close."""
    node = _node_shell()
    node.send_alive = lambda alive: (_ for _ in ()).throw(RuntimeError("root unreachable"))
    asyncio.run(node.aclose())
    assert node.conn.closes == 1


def test_aclose_swallows_a_failing_transport_close():
    """Shutdown must not raise out of the caller's finally: the other nodes of a
    synchronized run still have to be closed."""
    node = _node_shell(fail=True)
    asyncio.run(node.aclose())  # must not raise
    assert node.conn.closes == 1


class _FakeNode:
    """The surface NodeSynchronizer.run touches, plus a call log."""

    def __init__(self, node_type, name="a", raise_on_cycle=None):
        self.node_type = node_type
        self.node_id = name
        self.calls: list[str] = []
        self._raise_on_cycle = raise_on_cycle
        self._cycles = 0

        class _Behav:
            @staticmethod
            def get_state_changed():
                return False

        class _Agent:
            behav = _Behav()

            @staticmethod
            def get_name():
                return name

        self.agent = _Agent() if node_type == Node.AGENT else None

    async def run_async(self, cycles=None, join_world=None):
        self.calls.append("run")
        self._cycles += 1
        if self._raise_on_cycle is not None and self._cycles == self._raise_on_cycle:
            raise KeyboardInterrupt

    async def aclose(self):
        self.calls.append("close")


def _synchronizer(nodes) -> NodeSynchronizer:
    """A synchronizer over ready-made fake nodes (add_node wants a real Node)."""
    s = NodeSynchronizer()
    s.nodes = list(nodes)
    s.world = object()  # otherwise run() reports a missing world and raises
    s.gap = 0.0  # no per-node sleep: this test is about ordering, not timing
    return s


def test_synchronizer_reuses_the_nodes_across_cycles_and_closes_them_once_at_the_end():
    """The regression: three synchronized cycles must be three runs per node and a
    single shutdown each, at the very end."""
    world = _FakeNode(Node.WORLD, "w")
    agent = _FakeNode(Node.AGENT, "a")
    s = _synchronizer([world, agent])

    asyncio.run(s.run(addresses=None, synch_cycles=3))

    assert agent.calls == ["run", "run", "run", "close"]
    assert world.calls == ["run", "run", "run", "close"]


def test_synchronizer_closes_the_nodes_on_interrupt():
    """A Ctrl+C in the middle of a cycle still has to release the Go hosts."""
    world = _FakeNode(Node.WORLD, "w")
    agent = _FakeNode(Node.AGENT, "a", raise_on_cycle=2)
    s = _synchronizer([world, agent])

    asyncio.run(s.run(addresses=None, synch_cycles=None))

    assert agent.calls.count("close") == 1
    assert world.calls.count("close") == 1


def test_synchronizer_closes_the_world_after_the_agents():
    """Closing the world first would drop the transports the agents say goodbye on."""
    world = _FakeNode(Node.WORLD, "w")
    agent1 = _FakeNode(Node.AGENT, "a1")
    agent2 = _FakeNode(Node.AGENT, "a2")
    order: list[str] = []

    def _closer(name):
        async def _close():
            order.append(name)
        return _close

    for n in (world, agent1, agent2):
        n.aclose = _closer(n.node_id)

    s = _synchronizer([world, agent1, agent2])  # add_node order: world first
    asyncio.run(s.run(addresses=None, synch_cycles=1))

    assert order[-1] == "w"
    assert set(order) == {"w", "a1", "a2"}
