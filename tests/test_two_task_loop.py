"""
Phases 2 and 3 of the concurrency plan: the two tasks of a running node and the deferral of the agent changes.

- _agent_op: with a tick running (the tick task holds the agent lock) an operation from the network side is queued
  and applied in order at the boundary; with no tick it runs at once (after what is queued); nested operations and
  operations of the tick task itself run inline; a swap of the hosted entity asked during behave() waits;
- run_async without a cycle budget runs two tasks: while behave() awaits for seconds (a long action), the network
  side keeps pumping the Sessions and reading messages, the clock counts HSM ticks only, an INTERACTION that arrives
  during the action is injected after it, and a budget in time ends both tasks cleanly.
"""
import time
import asyncio
from unaiverse.clock import clock
from unaiverse.networking.p2p.messages import Msg
from node_harness import FakeHosted, build_agent_node_shell

NODE_PEER = "12D3KooWTwoTaskNodeShell"
REMOTE_PEER = "12D3KooWTwoTaskRemotePeer"


def _shell() -> object:
    """Builds a node shell with the state of the two tasks (as Node.__init__ sets it).

    Returns:
        The node.
    """
    node, _ = build_agent_node_shell(NODE_PEER)
    node._agent_lock = asyncio.Lock()
    node._deferred_agent_ops = []
    return node


def test_ops_wait_for_the_tick_boundary_and_keep_their_order() -> None:
    """A tick holds the lock: network-side ops queue up; the boundary drains them in order; then they run at once."""
    async def scenario() -> list[str]:
        """Runs a fake tick and three network-side operations around it (async).

        Returns:
            The order in which things happened.
        """
        node = _shell()
        done = []
        release = asyncio.Event()

        async def tick() -> None:
            """Holds the agent lock like a tick, then drains (async)."""
            async with node._agent_lock:
                node._agent_lock_owner = asyncio.current_task()
                await release.wait()
                done.append("tick end")
                await node._drain_deferred_agent_ops()
                node._agent_lock_owner = None

        def op(name: str) -> object:
            """Makes an operation that records its name.

            Args:
                name: The name.

            Returns:
                The operation.
            """
            async def _op() -> None:
                """Records the name (async)."""
                done.append(name)
            return _op

        node._tick_task = asyncio.create_task(tick())
        await asyncio.sleep(0)
        await node._agent_op(op("a"))
        await node._agent_op(op("b"))
        assert done == []
        release.set()
        await node._tick_task
        await node._agent_op(op("c"))
        return done

    assert asyncio.run(scenario()) == ["tick end", "a", "b", "c"]


def test_ops_of_the_tick_task_run_inline_except_swaps_during_behave() -> None:
    """From inside the tick (a purge from an action) an op runs inline; a swap during behave() waits for the end."""
    async def scenario() -> list[str]:
        """Runs a fake tick that asks for two operations from inside behave() (async).

        Returns:
            The order in which things happened.
        """
        node = _shell()
        done = []

        async def record(name: str) -> None:
            """Records a name (async).

            Args:
                name: The name.
            """
            done.append(name)

        async def tick() -> None:
            """A tick whose behave() asks for a purge and a swap (async)."""
            async with node._agent_lock:
                node._agent_lock_owner = asyncio.current_task()
                node._in_behave = True
                await node._agent_op(lambda: record("purge"))
                await node._agent_op(lambda: record("swap"), swap=True)
                done.append("behave end")
                node._in_behave = False
                await node._drain_deferred_agent_ops()
                node._agent_lock_owner = None

        node._tick_task = asyncio.create_task(tick())
        await node._tick_task
        return done

    assert asyncio.run(scenario()) == ["purge", "behave end", "swap"]


def test_nested_ops_run_inline_and_a_failing_op_does_not_stop_the_others() -> None:
    """An op applied by the network side that asks for another one runs it inline; a raising queued op is skipped."""
    async def scenario() -> list[str]:
        """Runs nested and failing operations (async).

        Returns:
            The order in which things happened.
        """
        node = _shell()
        done = []

        async def inner() -> None:
            """Records (async)."""
            done.append("inner")

        async def outer() -> None:
            """Asks for the inner operation (async)."""
            done.append("outer")
            await node._agent_op(inner)
            done.append("outer end")

        async def boom() -> None:
            """Raises (async)."""
            raise RuntimeError("bad message")

        node._tick_task = asyncio.create_task(asyncio.sleep(10))
        await node._agent_op(outer)
        node._deferred_agent_ops.extend([boom, inner])
        await node._drain_deferred_agent_ops()
        node._tick_task.cancel()
        return done

    assert asyncio.run(scenario()) == ["outer", "inner", "outer end", "inner"]


class _BusyHosted(FakeHosted):
    """A hosted agent whose behave() is a long action awaiting `action_s` seconds, counting the ticks."""

    def __init__(self, action_s: float) -> None:
        """Initializes the agent.

        Args:
            action_s: Seconds each behave() awaits.
        """
        super().__init__()
        self.action_s = action_s
        self.behaves = []  # (start, end) of each behave()
        self.injected = []  # (time, sender) of each injected interaction
        self.proc_inputs = []
        self.im = _FakeIM()

    async def behave(self) -> None:
        """A long action (async)."""
        t0 = time.monotonic()
        await asyncio.sleep(self.action_s)
        self.behaves.append((t0, time.monotonic()))

    def inject_received_interaction(self, sender: str, content: object) -> None:
        """Records an injected interaction.

        Args:
            sender: The sender.
            content: The content.
        """
        self.injected.append((time.monotonic(), sender))

    async def send_stream_samples(self) -> None:
        """No streams (async)."""

    async def subscribe_to_pubsub_owned_streams(self) -> None:
        """No pubsub (async)."""

    def streams_to_str(self) -> str:
        """No streams.

        Returns:
            An empty string.
        """
        return ""

    def get_name(self) -> str:
        """The name of the agent.

        Returns:
            The name.
        """
        return "busy"


class _FakeIM:
    """The interaction manager surface the loop reads."""

    def count_interactions(self) -> int:
        """No interactions.

        Returns:
            Zero.
        """
        return 0

    def room_for_registration(self) -> bool:
        """Always room.

        Returns:
            True.
        """
        return True


def _run_busy_node(action_s: float, max_time: float, interaction_at: float | None = None) -> tuple:
    """Runs a shell node with two tasks for max_time seconds, its behave() awaiting action_s each time.

    Args:
        action_s: Seconds of each behave().
        max_time: The time budget of the run.
        interaction_at: Seconds after the start at which an INTERACTION from a known peer arrives (default: None).

    Returns:
        The hosted agent, the times of the network passes, the start time, the cycles counted.
    """
    node, conn = build_agent_node_shell(NODE_PEER, hosted=_BusyHosted(action_s))
    node._agent_lock = asyncio.Lock()
    node._deferred_agent_ops = []
    node.hosted.all_agents[REMOTE_PEER] = object()
    node.run_hook = None
    node.last_alive_time = clock.get_time()  # No alive message to the root during the run
    node.get_public_addresses = lambda: [f"/ip4/10.0.0.1/tcp/4001/p2p/{NODE_PEER}"]
    node.get_world_addresses = lambda: [f"/ip4/10.0.0.1/tcp/4002/p2p/{NODE_PEER}-w"]
    node.get_public_peer_id = lambda: NODE_PEER
    node.get_world_peer_id = lambda: NODE_PEER + "-w"
    conn.passed_time_since_last_communication = lambda: 0.
    conn.requires_fast_clock = lambda: False
    passes = []
    start = time.monotonic()
    real_get_messages = conn.get_messages

    async def get_messages(p2p_name: str, kind: str = "all") -> tuple:
        """Records the pass and plants the INTERACTION when due (async).

        Args:
            p2p_name: The instance.

        Returns:
            What the fake conn returns, plus the planted message.
        """
        now = time.monotonic()
        msgs, inbound = await real_get_messages(p2p_name=p2p_name, kind=kind)
        if kind == "streams":
            return msgs, inbound
        if p2p_name == conn.P2P_PUBLIC:
            passes.append(now)
            if interaction_at is not None and now - start >= interaction_at and not getattr(conn, "_planted", False):
                conn._planted = True
                msgs = list(msgs) + [Msg(sender=REMOTE_PEER, content_type=Msg.INTERACTION, content={"x": 1},
                                         channel=f"{REMOTE_PEER}::dm:{NODE_PEER}-{Msg.INTERACTION}",
                                         piggyback="tok0")]
        return msgs, inbound
    conn.get_messages = get_messages
    cycle0 = clock.get_cycle()
    asyncio.run(node.run_async(max_time=max_time))
    return node.hosted, passes, start, clock.get_cycle() - cycle0


def test_network_keeps_running_during_a_long_action() -> None:
    """While behave() awaits 2 s, the network passes keep coming (every ~50 ms), and the cycle counts ticks only."""
    hosted, passes, start, cycles = _run_busy_node(action_s=2.0, max_time=4.5)
    assert len(hosted.behaves) >= 2
    b0, b1 = hosted.behaves[0]
    during = [t for t in passes if b0 <= t <= b1]
    assert len(during) >= 10, f"{len(during)} network passes during a 2 s action"
    gaps = [b - a for a, b in zip(during, during[1:])]
    assert max(gaps) < 1.0, f"the network waited {max(gaps):.2f}s during the action"
    assert cycles <= len(hosted.behaves) + 1, (cycles, len(hosted.behaves))


def test_an_interaction_during_an_action_is_injected_after_it() -> None:
    """An INTERACTION read in the middle of behave() reaches the agent only once that behave() ended."""
    hosted, passes, start, _ = _run_busy_node(action_s=2.0, max_time=4.5, interaction_at=0.8)
    assert hosted.injected, "the interaction was never injected"
    t_inject = hosted.injected[0][0]
    running = [(a, b) for a, b in hosted.behaves if a <= start + 0.8 <= b]
    assert running, hosted.behaves
    assert t_inject >= running[0][1], "injected while behave() was running"


def test_a_stop_request_ends_both_tasks() -> None:
    """Ctrl+C (stop_requested) during an action: the action ends, then both tasks stop and the node shuts down."""
    node, conn = build_agent_node_shell(NODE_PEER, hosted=_BusyHosted(1.0))
    node._agent_lock = asyncio.Lock()
    node._deferred_agent_ops = []
    node.run_hook = None
    node.last_alive_time = clock.get_time()
    node.get_public_addresses = lambda: [f"/ip4/10.0.0.1/tcp/4001/p2p/{NODE_PEER}"]
    node.get_world_addresses = lambda: [f"/ip4/10.0.0.1/tcp/4002/p2p/{NODE_PEER}-w"]
    node.get_public_peer_id = lambda: NODE_PEER
    node.get_world_peer_id = lambda: NODE_PEER + "-w"
    conn.passed_time_since_last_communication = lambda: 0.
    conn.requires_fast_clock = lambda: False
    closed = []

    async def aclose() -> None:
        """Records the shutdown (async)."""
        closed.append(True)
    node.aclose = aclose

    async def scenario() -> float:
        """Runs the node without a budget and asks it to stop after 1.5 s (async).

        Returns:
            How long the run lasted.
        """
        asyncio.get_running_loop().call_later(1.5, lambda: setattr(node, "stop_requested", True))
        t0 = time.monotonic()
        await asyncio.wait_for(node.run_async(), timeout=10.0)
        return time.monotonic() - t0

    elapsed = asyncio.run(scenario())
    assert 1.5 <= elapsed < 4.0, elapsed
    assert closed == [True] and node._tick_task is None


def test_messages_waiting_for_a_tick_are_bounded_and_interactions_rejected_past_the_bound(monkeypatch: object) \
        -> None:
    """Past the per-sender bound an INTERACTION is rejected (its sender told) and another message dropped; the room
    comes back once the queued operations are applied."""
    from unaiverse.custom import Custom
    monkeypatch.setattr(Custom, "DEFERRED_MSGS_MAX_PER_PEER", 2)

    async def scenario() -> tuple:
        """Holds the lock like a tick, offers four messages, then drains (async).

        Returns:
            What was applied, what was rejected, and the counters after the drain.
        """
        node = _shell()
        node._deferred_msgs_by_peer, node._deferred_msg_bytes = {}, 0
        applied, rejected = [], []

        async def reject(sender: str, content: dict) -> None:
            """Records the rejection (async).

            Args:
                sender: The sender.
                content: The interaction.
            """
            rejected.append(content["uuid"])
        node.agent.reject_received_interaction = reject
        release = asyncio.Event()

        async def tick() -> None:
            """Holds the agent lock like a tick, then drains (async)."""
            async with node._agent_lock:
                node._agent_lock_owner = asyncio.current_task()
                await release.wait()
                await node._drain_deferred_agent_ops()
                node._agent_lock_owner = None

        node._tick_task = asyncio.create_task(tick())
        await asyncio.sleep(0)
        for i, content_type in enumerate([Msg.INTERACTION, Msg.INTERACTION_STATUS, Msg.INTERACTION, Msg.HSM]):
            msg = Msg(sender=REMOTE_PEER, content_type=content_type, content={"uuid": f"u{i}"},
                      channel=f"{REMOTE_PEER}::dm:{NODE_PEER}-{content_type}", piggyback="tok0")
            msg.wire_size = 100

            async def op(_i: int = i) -> None:
                """Records the application (async)."""
                applied.append(f"u{_i}")
            await node._agent_op_for_message(msg, op)
        release.set()
        await node._tick_task
        return applied, rejected, dict(node._deferred_msgs_by_peer), node._deferred_msg_bytes

    applied, rejected, by_peer, total = asyncio.run(scenario())
    assert applied == ["u0", "u1"] and rejected == ["u2"], (applied, rejected)
    assert by_peer == {} and total == 0


class _TicketHosted(_BusyHosted):
    """A hosted agent whose action sends a tracked message and awaits its ticket."""

    def __init__(self) -> None:
        """Initializes the agent."""
        super().__init__(action_s=0.)
        self.conn = None
        self.result = None  # (what wait() gave, seconds it took)

    async def behave(self) -> None:
        """The action, once (async)."""
        if self.result is None and self.conn is not None:
            ticket = await self.conn.send_tracked(REMOTE_PEER, None, Msg.MISC, {"hello": 1})
            t0 = time.monotonic()
            ok = await ticket.wait(5.0)
            self.result = (ok, time.monotonic() - t0)


def test_an_action_awaiting_a_ticket_is_woken_by_the_network_task() -> None:
    """With the two tasks, a ticket awaited inside behave() is resolved by the network task as the event comes."""
    from node_harness import build_node_shell_on_real_conn
    node, conn, p2p_public, _ = build_node_shell_on_real_conn(NODE_PEER)
    hosted = _TicketHosted()
    hosted.conn = conn
    node.hosted = node.agent = hosted
    node.run_hook = None
    node.last_alive_time = clock.get_time()
    node.get_public_addresses = lambda: [f"/ip4/10.0.0.1/tcp/4001/p2p/{NODE_PEER}"]
    node.get_world_addresses = lambda: [f"/ip4/10.0.0.1/tcp/4002/p2p/{NODE_PEER}-w"]
    sm = conn._sm_public
    conn._sm_of = lambda peer_id: sm
    events = []

    async def send(channel: str, msg: object) -> int:
        """Queues: the message gets number 1 (async).

        Args:
            channel: The channel.
            msg: The message.

        Returns:
            The sequence number.
        """
        asyncio.get_running_loop().call_later(0.5, lambda: events.append(
            {"type": "written", "peer_id": REMOTE_PEER, "seq": 1}))
        return 1

    def pop_send_events() -> list:
        """What the queue reported since the last pop.

        Returns:
            The events.
        """
        out = list(events)
        events.clear()
        return out

    p2p_public.send = send
    p2p_public.pop_send_events = pop_send_events
    p2p_public._send_queues_enabled = True

    async def no_close() -> None:
        """No root to say goodbye to (async)."""
    node.aclose = no_close
    asyncio.run(node.run_async(max_time=3.0))
    ok, elapsed = hosted.result
    assert ok is True and 0.4 <= elapsed < 1.5, hosted.result
