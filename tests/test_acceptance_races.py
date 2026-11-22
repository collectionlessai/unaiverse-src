"""
Phase 5, items 6, 9 and 10 of the concurrency plan: what the network side does to the hosted entity while an action
runs waits for the end of the tick, and never shows the running action a half-changed agent.

A node shell (node_harness) with the state of the two tasks; a fake tick holds the agent lock with behave() "running"
until released. Meanwhile the network side: purges a peer (6), receives a role change from the world host (6), leaves
the world (6), completes a world join (9: the handoff goes out at once, the agent part waits), and receives an
INTERACTION_STATUS (10). Each one: the network/Session part now, nothing on the hosted entity during the action, all
of it right after, in order.
"""
import asyncio
from unaiverse.networking.p2p.messages import Msg
from node_harness import FakeHosted, build_agent_node_shell

NODE_PEER = "12D3KooWRaceNodeShell"
PEER = "12D3KooWRacePeer"
WORLD_HOST = "12D3KooWRaceWorldHost"


class _RecordingIM:
    """The interaction manager surface: records the status updates."""

    def __init__(self) -> None:
        """Initializes the manager."""
        self.updates = []

    def count_interactions(self) -> int:
        """None in progress.

        Returns:
            Zero.
        """
        return 0

    async def update_sent_status(self, status: dict) -> None:
        """Records the update (async).

        Args:
            status: The status dict.
        """
        self.updates.append(status["uuid"])


def _node() -> tuple:
    """A node shell with the two-task state, and a fake tick that holds the lock until released.

    Returns:
        The node, its conn, the release event, and an async starter for the tick.
    """
    node, conn = build_agent_node_shell(NODE_PEER, hosted=FakeHosted())
    node._agent_lock = asyncio.Lock()
    node._deferred_agent_ops = []
    node._deferred_msgs_by_peer, node._deferred_msg_bytes = {}, 0
    node.hosted.all_agents[PEER] = object()
    release = asyncio.Event()
    behave_running = []

    async def tick() -> None:
        """Holds the lock like a tick in behave(), then drains at the boundary (async)."""
        async with node._agent_lock:
            node._agent_lock_owner = asyncio.current_task()
            node._in_behave = True
            behave_running.append(True)
            await release.wait()
            node._in_behave = False
            behave_running.append(False)
            await node._drain_deferred_agent_ops()
            node._agent_lock_owner = None

    async def start() -> None:
        """Starts the tick and lets it take the lock (async)."""
        node._tick_task = asyncio.create_task(tick())
        await asyncio.sleep(0)

    return node, conn, release, start


def test_a_purge_during_an_action_drops_the_link_now_and_the_agent_entry_at_the_boundary() -> None:
    """The conn part of __purge runs at once; remove_agent waits for the end of the action."""
    async def scenario() -> tuple:
        """Purges during the fake action (async).

        Returns:
            What the conn and the hosted entity saw, during and after.
        """
        node, conn, release, start = _node()
        await start()
        await node._Node__purge(PEER, reason=None)
        during = (list(conn.removed_calls), list(node.hosted.removed), PEER in node.hosted.all_agents)
        release.set()
        await node._tick_task
        return during, list(node.hosted.removed)

    (dropped, removed_during, still_there), removed_after = asyncio.run(scenario())
    assert dropped == [PEER] and removed_during == [] and still_there
    assert removed_after == [PEER]


def _plant(conn: object, msg: Msg) -> None:
    """Makes the next network pass read one chat message on the public instance.

    Args:
        conn: The fake conn.
        msg: The message.
    """
    real = conn.get_messages
    planted = [msg]

    async def get_messages(p2p_name: str, allowed_not_connected_peers: object = None, kind: str = "all") -> tuple:
        """The planted message once, then whatever the fake conn has (async).

        Args:
            p2p_name: The instance.
            allowed_not_connected_peers: Unused.
            kind: The pop kind.

        Returns:
            Chat messages and Session frames.
        """
        chat, sess = await real(p2p_name, kind=kind)
        if p2p_name == conn.P2P_PUBLIC and planted and kind != "streams":
            chat = chat + [planted.pop()]
        return chat, sess
    conn.get_messages = get_messages


def test_a_role_change_during_an_action_swaps_the_agent_only_at_the_boundary() -> None:
    """A ROLE_SUGGESTION from the world host is read at once but the new agent instance comes after the action."""
    async def scenario() -> tuple:
        """Receives the role change during the fake action (async).

        Returns:
            The swaps seen during and after.
        """
        node, conn, release, start = _node()
        conn.set_world(WORLD_HOST)
        node.get_world_peer_id = lambda: NODE_PEER + "-w"
        swaps = []
        node._Node__replace_agent_instance_by_role = lambda role, **kw: swaps.append(role)
        _plant(conn, Msg(sender=WORLD_HOST, content_type=Msg.ROLE_SUGGESTION,
                         content={"peer_id": NODE_PEER + "-w", "role": 5},
                         channel=f"{WORLD_HOST}::dm:{NODE_PEER}-{Msg.ROLE_SUGGESTION}", piggyback="nodeid0"))
        await start()
        await node._Node__handle_network_messages(kind="network")
        during = list(swaps)
        release.set()
        await node._tick_task
        return during, swaps

    during, after = asyncio.run(scenario())
    assert during == [] and after == [5]


def test_leaving_the_world_during_an_action_unpoints_routing_now_and_touches_the_agent_after() -> None:
    """leave_world: the world host is purged and routing cleared at once; the agent part waits for the boundary."""
    async def scenario() -> tuple:
        """Leaves the world during the fake action (async).

        Returns:
            Routing and agent state seen during and after.
        """
        node, conn, release, start = _node()
        conn.set_world(WORLD_HOST)
        node.hosted._in_world = True
        await start()
        await node.leave_world()
        during = (conn.get_world_node_peer_id(), node.hosted.world_data_clears, list(node.hosted.accepted_roles))
        release.set()
        await node._tick_task
        return during, (node.hosted.world_data_clears, list(node.hosted.accepted_roles))

    (world_during, clears_during, roles_during), (clears_after, roles_after) = asyncio.run(scenario())
    assert world_during is None and clears_during == 0 and roles_during == []
    assert clears_after == 1 and len(roles_after) == 1


def test_a_world_join_during_an_action_hands_off_at_once_and_swaps_at_the_boundary() -> None:
    """The network half of the join and the handoff GOAWAY run now; the agent half after the action."""
    async def scenario() -> tuple:
        """Completes a join during the fake action (async).

        Returns:
            The order of the steps.
        """
        node, conn, release, start = _node()
        steps = []

        async def network_part(profile: object, role: int, peer_id: str) -> bool:
            """Records the network half (async).

            Args:
                profile: The world profile.
                role: The role.
                peer_id: The world Session.

            Returns:
                True.
            """
            steps.append("network")
            return True

        async def agent_part(**kwargs: object) -> bool:
            """Records the agent half (async).

            Args:
                **kwargs: The join arguments.

            Returns:
                True.
            """
            steps.append("agent")
            return True

        async def handoff(world_peer_id: str) -> None:
            """Records the handoff GOAWAY (async).

            Args:
                world_peer_id: The world Session.
            """
            steps.append("handoff")

        node._Node__join_world_network = network_part
        node._Node__join_world_agent = agent_part
        node._world_join_kwargs = lambda peer_id, grant: {"profile": None, "role": 5, "peer_id": peer_id}
        conn.complete_world_handoff = handoff
        await start()
        ev = type("Ev", (), {"peer_id": WORLD_HOST, "p2p_name": conn.P2P_WORLD})()
        await node._join_world_split(ev, {"grant": True})
        during = list(steps)
        release.set()
        await node._tick_task
        return during, steps

    during, after = asyncio.run(scenario())
    assert during == ["network", "handoff"] and after == ["network", "handoff", "agent"]


def test_an_interaction_status_during_an_action_is_applied_at_the_boundary() -> None:
    """An INTERACTION_STATUS (it may run a callback and toggle the HSMs) waits for the end of the action."""
    async def scenario() -> tuple:
        """Receives a status during the fake action (async).

        Returns:
            The updates seen during and after.
        """
        node, conn, release, start = _node()
        node.hosted.im = _RecordingIM()
        _plant(conn, Msg(sender=PEER, content_type=Msg.INTERACTION_STATUS,
                         content={"uuid": "u9", "status": "completed"},
                         channel=f"{PEER}::dm:{NODE_PEER}-{Msg.INTERACTION_STATUS}", piggyback="nodeid0"))
        await start()
        await node._Node__handle_network_messages(kind="network")
        during = list(node.hosted.im.updates)
        release.set()
        await node._tick_task
        return during, node.hosted.im.updates

    during, after = asyncio.run(scenario())
    assert during == [] and after == ["u9"]
