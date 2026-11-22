"""
Phase 5, item 1 of the concurrency plan: a node whose HSM runs one long action (an offloaded processor that works for
60 s, or 3 minutes) keeps its Sessions alive and keeps receiving messages (opt-in: ``-m long``, real time).

A real NodeConn with real SessionManagers on the fake wire (node_harness) and a remote peer that completes the public
handshake, then pumps and ticks its manager every 200 ms, as a healthy node does. The node runs run_async with its two
tasks; once the Session is ESTABLISHED, its next behave() awaits a ModuleWrapper whose module sleeps in the worker.
During that action: both Sessions stay ESTABLISHED with no RECONNECTING or LOST (the keep-alive PINGs of both sides are
answered: period 15 s, RTT window 10 s, two misses allowed, so a stalled node would be dropped after 25-40 s), and an
INTERACTION sent by the remote is read by the network task and injected right after the action, against a stable
state. The result of the action comes back.
"""
import time
import torch
import pytest
import asyncio
import threading
from conftest import apump
from unaiverse.clock import clock
from unaiverse.networking.p2p.messages import Msg
from unaiverse.modules.utils import ModuleWrapper
from unaiverse.streams.dataprops import StreamType
from unaiverse.networking.node.profile import NodeProfile
from unaiverse.networking.session import SessionEventType, SessionState
from node_harness import FakeHosted, build_node_shell_on_real_conn, wire_remote_peer, mint_node_token, full_profile_dict

pytestmark = pytest.mark.long

NODE_PEER = "12D3KooWaaa1nodeshelllllllllllllllllllllllllllllllll"
REMOTE_PEER = "12D3KooWrrr5remotepeerrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr"
REMOTE_NODE_ID = "remote-node"


@pytest.fixture(autouse=True)
def _offline_node_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keeps NodeProfile from looking up a public IP over HTTP.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
    """
    monkeypatch.setattr(NodeProfile, "_get_public_ip_address", staticmethod(lambda: "203.0.113.1"))


class _SlowModule(torch.nn.Module):
    """A text module that works for a long time (in the worker thread of its wrapper)."""

    def __init__(self, seconds: float) -> None:
        """Initializes the module.

        Args:
            seconds: How long each call takes.
        """
        super().__init__()
        self.seconds = seconds
        self.thread = None

    def forward(self, text: str) -> str:
        """Works, then answers.

        Args:
            text: The input.

        Returns:
            The answer.
        """
        self.thread = threading.current_thread().name
        time.sleep(self.seconds)
        return f"done: {text}"


class _LongActionHosted(FakeHosted):
    """A hosted agent whose behave() is quick until told to run one long action through an offloaded processor."""

    def __init__(self, seconds: float) -> None:
        """Initializes the agent.

        Args:
            seconds: How long the long action takes.
        """
        super().__init__()
        self.proc = ModuleWrapper(module=_SlowModule(seconds), proc_inputs=[StreamType("text")],
                                  proc_outputs=[StreamType("text")])
        self.proc_inputs = []
        self.im = _NoInteractions()
        self.go_long = False
        self.action = None  # (start, end, result) of the long action
        self.injected = []  # (time, sender) of each injected interaction

    async def behave(self) -> None:
        """Quick, or the long action once asked (async)."""
        if self.go_long and self.action is None:
            start = time.monotonic()
            result = await self.proc.aforward("the task")
            self.action = (start, time.monotonic(), result)

    def inject_received_interaction(self, sender: str, content: object) -> None:
        """Records an injected interaction.

        Args:
            sender: The sender.
            content: The interaction.
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
        return "long-action"


class _NoInteractions:
    """The interaction-manager surface the loop reads."""

    def count_interactions(self) -> int:
        """None in progress.

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


def _run(action_s: float, root_keys: tuple) -> dict:
    """Runs the scenario with an action of the given length and gathers what happened.

    Args:
        action_s: Seconds of the long action.
        root_keys: The (private, public) PEM keys of the test root.

    Returns:
        What the test checks.
    """
    private_pem, public_pem = root_keys
    node, conn, p2p_public, _ = build_node_shell_on_real_conn(NODE_PEER, public_key=public_pem)
    hosted = _LongActionHosted(action_s)
    node.hosted = node.agent = hosted
    node.run_hook = None
    node.last_alive_time = clock.get_time()  # No alive message to the root
    node.get_public_addresses = lambda: [f"/ip4/10.0.0.1/tcp/4001/p2p/{NODE_PEER}"]
    node.get_world_addresses = lambda: [f"/ip4/10.0.0.1/tcp/4002/p2p/{NODE_PEER}-w"]
    profile = full_profile_dict(REMOTE_NODE_ID, peer_id=REMOTE_PEER, private_peer_id=REMOTE_PEER)
    token = mint_node_token(private_pem, profile, [REMOTE_PEER])
    remote_sm, remote_inbox = wire_remote_peer(p2p_public, REMOTE_PEER, profile, token)
    remote_events = []
    sent_at = None
    remote_stop = threading.Event()

    async def remote_peer() -> None:
        """The remote node: dials, then pumps and ticks its Sessions every 200 ms, until told to stop (async)."""
        p2p_public.link(REMOTE_PEER)
        await remote_sm.initiate(p2p_public.peer_id, ["/ip4/10.0.0.1/tcp/4001"])
        while not remote_stop.is_set():
            await asyncio.sleep(0.2)
            remote_events.extend((time.monotonic(), e.type) for e in await apump(remote_inbox, remote_sm))
            remote_events.extend((time.monotonic(), e.type) for e in await remote_sm.tick())

    # The remote is another node: it runs on its own event loop (its own thread here), so a node whose loop stalls is
    # seen stalling, as a real peer would see it. The two only share the deques of the fake wire
    remote_thread = threading.Thread(target=lambda: asyncio.run(remote_peer()), daemon=True)

    async def scenario() -> None:
        """Runs the node, starts the long action once ESTABLISHED, sends an INTERACTION in the middle (async)."""
        nonlocal sent_at
        node_task = asyncio.create_task(node.run_async(max_time=action_s + 40.))
        remote_thread.start()
        deadline = time.monotonic() + 20.
        while not (remote_sm.is_established(p2p_public.peer_id) and REMOTE_PEER in hosted.all_agents):
            assert time.monotonic() < deadline, f"no Session: {remote_events}"
            await asyncio.sleep(0.1)
        hosted.go_long = True
        while hosted.action is None and hosted.proc.module.thread is None:
            await asyncio.sleep(0.05)
        await asyncio.sleep(min(5., action_s / 4))
        sent_at = time.monotonic()
        p2p_public.chat_inbox.append(Msg(sender=REMOTE_PEER, content_type=Msg.INTERACTION,
                                         content={"uuid": "u1", "action_name": "noop"},
                                         channel=f"{REMOTE_PEER}::dm:{NODE_PEER}-{Msg.INTERACTION}",
                                         piggyback=token + "0"))
        while hosted.action is None:
            await asyncio.sleep(0.2)
        node.stop_requested = True
        await asyncio.wait_for(node_task, timeout=30.)
        remote_stop.set()
        remote_thread.join(timeout=5.)

    async def no_close() -> None:
        """The shell has no root to say goodbye to (async)."""
    node.aclose = no_close
    asyncio.run(scenario())
    node_sess = conn._sm_public.get_session(REMOTE_PEER)
    return {"action": hosted.action, "thread": hosted.proc.module.thread, "injected": hosted.injected,
            "sent_at": sent_at, "remote_events": remote_events,
            "remote_state": remote_sm.get_session(p2p_public.peer_id).state,
            "node_state": node_sess.state if node_sess is not None else None}


def _check(r: dict, action_s: float) -> None:
    """The acceptance checks shared by both lengths.

    Args:
        r: What the scenario gathered.
        action_s: Seconds of the long action.
    """
    start, end, result = r["action"]
    bad = [(round(t - start, 1), e.value) for t, e in r["remote_events"]
           if start <= t <= end and e in (SessionEventType.RECONNECTING, SessionEventType.LOST)]
    assert not bad, f"the remote lost the node during the action: {bad}"
    assert result == ("done: the task",) and end - start >= action_s * 0.95
    assert r["thread"].startswith("unaiverse-proc")
    assert r["remote_state"] == SessionState.ESTABLISHED and r["node_state"] == SessionState.ESTABLISHED
    assert start < r["sent_at"] < end
    assert r["injected"] and r["injected"][0][0] >= end, "the INTERACTION was injected during the action"


def test_sessions_and_messages_survive_a_60_s_action(root_keys: tuple) -> None:
    """A 60 s offloaded action: Sessions alive, the INTERACTION read meanwhile and injected after, result back."""
    _check(_run(60., root_keys), 60.)


def test_sessions_and_messages_survive_a_3_min_action(root_keys: tuple) -> None:
    """The same with a 3 minute action (several keep-alive periods and reconnect windows)."""
    _check(_run(180., root_keys), 180.)
