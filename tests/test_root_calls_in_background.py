"""The calls to the root server that the network loop makes run in the background: a root server that does not answer
must not stop the loop (no Session pings, no frames). One call per kind at a time; the badges assigned while a call
is on its way are kept, and the ones sent go back to the world when the call fails.
"""
import asyncio
import threading
import pytest

from unaiverse.world import World
from unaiverse.networking.node.node import Node

pytestmark = pytest.mark.unit


class _Profile:
    """The part of the profile that send_dynamic_profile reads."""

    def __init__(self):
        self.dynamic = {"peer_addresses": ["/ip4/1.2.3.4/tcp/1"], "hidden": None}

    def get_dynamic_profile(self) -> dict:
        return self.dynamic


class _Conn:
    """The part of the connection manager that the badges notify through."""

    def __init__(self):
        self.sent = []

    async def send(self, peer_id, channel_trail=None, content=None, content_type=None) -> bool:
        self.sent.append(peer_id)
        return True


class _Root:
    """A root server whose answers are held until released (a server that does not answer)."""

    def __init__(self, fail: bool = False):
        self.release = threading.Event()
        self.calls = []
        self.fail = fail

    def __call__(self, api: str, payload: dict) -> dict:
        self.calls.append((api, payload))
        self.release.wait(10.0)
        if self.fail:
            raise ConnectionError("the root server is down")
        if api.endswith("/badge/assign"):
            return {"badge_token": "next", "badges_states": [{"state": {"code": "ok", "message": ""}}
                                                             for _ in payload["badges"]]}
        if api.endswith("/alive"):
            return {"was_alive": False}
        return {}


def _node(root: _Root) -> Node:
    """A world node with only what the calls to the root server use."""
    node = object.__new__(Node)
    node._Node__root_jobs = {}
    node._Node__root = root
    node.node_id = "node-1"
    node.unaiverse_key = "key"
    node.node_type = Node.WORLD
    node.profile = _Profile()
    node.conn = _Conn()
    node.badge_token = "first"
    world = object.__new__(World)
    world.agent_badges = {}
    node.world = world
    return node


async def _stays_responsive(seconds: float) -> int:
    """Counts the turns the event loop gives in a while (a stopped loop gives none) (async)."""
    turns, end = 0, asyncio.get_running_loop().time() + seconds
    while asyncio.get_running_loop().time() < end:
        await asyncio.sleep(0.01)
        turns += 1
    return turns


def test_a_root_that_does_not_answer_does_not_stop_the_caller():
    async def scenario() -> None:
        root = _Root()
        node = _node(root)
        profile_job = node.send_dynamic_profile(wait=False)
        alive_job = node.send_alive(wait=False)
        assert not profile_job.done() and not alive_job.done()
        assert await _stays_responsive(0.3) > 10

        # One call per kind: while it is on its way, the same task comes back
        assert node.send_dynamic_profile(wait=False) is profile_job
        root.release.set()
        await asyncio.gather(profile_job, alive_job)
        assert [api for api, _ in root.calls].count("/account/node/profile/dynamic/post") == 1
        assert await alive_job is False
        assert node.send_dynamic_profile(wait=False) is not profile_job

    asyncio.run(scenario())


def test_the_profile_sent_is_the_one_of_when_the_call_started():
    async def scenario() -> None:
        root = _Root()
        node = _node(root)
        job = node.send_dynamic_profile(wait=False)
        node.profile.dynamic["peer_addresses"].append("/ip4/5.6.7.8/tcp/2")  # The node goes on meanwhile
        root.release.set()
        await job
        _, payload = root.calls[0]
        assert payload["profile"]["peer_addresses"] == ["/ip4/1.2.3.4/tcp/1"]
        assert "hidden" not in payload["profile"]

    asyncio.run(scenario())


def test_badges_assigned_during_the_call_are_kept():
    async def scenario() -> None:
        root = _Root()
        node = _node(root)
        node.world.agent_badges = {"peer-a": [{"badge": 1}]}
        job = node.send_badges()
        await asyncio.sleep(0.05)
        node.world.agent_badges.setdefault("peer-b", []).append({"badge": 2})  # Assigned while the call is on its way
        root.release.set()
        await job
        assert node.world.agent_badges == {"peer-b": [{"badge": 2}]}
        assert node.badge_token == "next" and node.conn.sent == ["peer-a"]

    asyncio.run(scenario())


def test_badges_that_could_not_be_sent_go_back_to_the_world():
    async def scenario() -> None:
        root = _Root(fail=True)
        root.release.set()
        node = _node(root)
        node.world.agent_badges = {"peer-a": [{"badge": 1}]}
        job = node.send_badges()
        node.world.agent_badges.setdefault("peer-a", []).append({"badge": 2})  # Assigned meanwhile
        await job
        assert node.world.agent_badges == {"peer-a": [{"badge": 1}, {"badge": 2}]}
        assert node.badge_token == "first"

    asyncio.run(scenario())


def test_a_token_refresh_in_the_background_tells_whether_it_worked(monkeypatch):
    async def scenario() -> None:
        node = _node(_Root())
        applied = []
        monkeypatch.setattr(node, "_request_node_token", lambda peer_ids: {"token": "t"}, raising=False)
        monkeypatch.setattr(node, "_apply_node_token", lambda response: applied.append(response), raising=False)
        assert await node.get_node_token(["a", "b"], wait=False) is True
        assert applied == [{"token": "t"}]

        def fails(peer_ids):
            raise ConnectionError("the root server is down")

        monkeypatch.setattr(node, "_request_node_token", fails, raising=False)
        assert await node.get_node_token(["a", "b"], wait=False) is False
        assert applied == [{"token": "t"}]

    asyncio.run(scenario())
