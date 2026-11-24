"""The calls to the root server that the network loop makes run in the background: a root server that does not answer
must not stop the loop (no Session pings, no frames). One call per kind at a time; the badges assigned while a call
is on its way are kept, and the ones sent go back to the world when the call fails.
"""
import json
import time
import hashlib
import asyncio
import threading
import jwt
import pytest

from unaiverse.world import World
from unaiverse.networking.node.node import Node
from unaiverse.networking.p2p.messages import Msg
from node_harness import build_agent_node_shell, make_root_keys

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
    node.node_token = None
    node.public_key = None
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




ROOT_PRIVATE, ROOT_PUBLIC = make_root_keys()


def _canonical(cv: list[dict]) -> list[dict]:
    """The CV in the order the root server hashes it: entries by last edit, keys sorted."""
    return [{k: entry[k] for k in sorted(entry)} for entry in sorted(cv, key=lambda entry: entry["last_edit_utc"])]


def _hash(cv: list[dict]) -> str:
    """The hash of a CV that the root server attests in a token."""
    return hashlib.blake2b(json.dumps(_canonical(cv)).encode("utf-8"), digest_size=16).hexdigest()


def _cv(*badges: str) -> list[dict]:
    """A CV with the given badges, its keys in no particular order, as the root server may send it."""
    return [{"last_edit_utc": i, "badge": badge} for i, badge in enumerate(badges)]


class _OrderedRoot:
    """A root server that holds each answer until the test releases it, by the position of the call. It answers with
    the state it had when the call came: its CV, and a token that attests that CV (or token_cv_hash, when set)."""

    def __init__(self, cv: list[dict], hold: bool = True):
        self.calls = []
        self.cv = cv
        self.token_cv_hash = None
        self.token_fails = False
        self.hold = hold

    def token(self) -> str:
        """A token as the root mints it now."""
        return jwt.encode({"node_id": "shell-node", "p2p_peers": json.dumps(["agent-pub", "agent-pub-w"]),
                           "cv_hash": self.token_cv_hash or _hash(self.cv),
                           "exp": int(time.time()) + 3600}, ROOT_PRIVATE, algorithm="RS256")

    def __call__(self, api: str, payload: dict):
        if api.endswith("/cv/get"):
            answer = [dict(entry) for entry in self.cv]
        elif api.endswith("/get/addresses"):
            answer = {"addresses": []}
        elif self.token_fails:
            answer = ConnectionError("the root server is down")
        else:
            answer = {"token": self.token(), "public_key": ROOT_PUBLIC}
        released = threading.Event()
        self.calls.append((api, released))
        if self.hold:
            released.wait(3.0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def apis(self) -> list[str]:
        return [api.rsplit("/", 1)[-1] for api, _ in self.calls]

    def release(self, i: int) -> None:
        self.calls[i][1].set()


async def _until(condition, seconds: float = 2.0) -> None:
    """Waits, on the event loop, until the condition holds (async)."""
    end = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < end, "the condition never held"
        await asyncio.sleep(0.01)


async def _answer_all(root: _OrderedRoot, calls: int) -> None:
    """Releases the calls one by one as they come, up to the given number (async)."""
    for i in range(calls):
        await _until(lambda: len(root.calls) > i)
        root.release(i)


def _agent(root: _OrderedRoot):
    """An agent shell that holds the CV of the root server and a token that attests it, and asks both of that root."""
    node, conn = build_agent_node_shell("agent-pub")
    node._Node__root_jobs = {}
    node._Node__cv_stale, node._Node__pending_cv, node._Node__cv_mismatch_logged = False, None, False
    node._Node__root = root
    node.unaiverse_key = "key"
    node.get_public_peer_id = lambda: "agent-pub"
    node.get_world_peer_id = lambda: "agent-pub-w"
    node._request_node_token = lambda peer_ids: root("/account/node/token/get", {})
    conn.set_token = lambda token: None
    held = {"cv": _canonical(root.cv)}
    node.profile.get_cv = lambda: held["cv"]
    node.profile.update_cv = lambda cv: held.update(cv=_canonical(cv))
    node.node_token, node.public_key, node.turn_info = root.token(), ROOT_PUBLIC, None
    return node, conn


def _held(node) -> tuple[str, str]:
    """The hash of the CV in the node's profile, and the one its token attests."""
    token_hash = jwt.decode(node.node_token, ROOT_PUBLIC, algorithms=["RS256"])["cv_hash"]
    return _hash(node.profile.get_cv()), token_hash


def test_a_cv_notification_does_not_stop_the_loop_and_the_cv_goes_in_with_its_token():
    """The world host tells an agent to download its CV again after a badge: the handler leaves both calls to the
    background, and the new CV goes into the profile only together with a token that attests it."""
    async def scenario() -> None:
        root = _OrderedRoot(_cv("first"))
        node, conn = _agent(root)
        root.cv = _cv("first", "badge")
        notification = Msg(sender="world-pub", content_type=Msg.GET_CV_FROM_ROOT, piggyback="world-node0")
        delivered = []

        async def get_messages(p2p_name, allowed_not_connected_peers=None, kind="all"):
            if kind == "streams" or delivered or p2p_name != conn.P2P_PUBLIC:
                return [], []
            delivered.append(notification)
            return [notification], []
        conn.get_messages = get_messages

        started = time.monotonic()
        await node._Node__handle_network_messages()
        assert time.monotonic() - started < 1.0
        assert await _stays_responsive(0.2) > 5
        assert root.apis() == ["get"]

        root.release(0)
        await _until(lambda: len(root.calls) == 2)
        assert root.apis() == ["get", "get"] and root.calls[1][0].endswith("/token/get")
        assert _held(node)[0] == _hash(_cv("first"))   # Not before its token
        root.release(1)
        await _until(lambda: _held(node)[0] == _hash(root.cv))
        assert _held(node)[1] == _hash(root.cv)

    asyncio.run(scenario())


def test_a_token_refresh_on_its_way_when_the_cv_comes_is_followed_by_a_new_one():
    """A refresh of the token asked before the badge attests the old CV: it is let end, then a new one is asked, and
    the CV goes in with that one."""
    async def scenario() -> None:
        root = _OrderedRoot(_cv("first"))
        node, _ = _agent(root)
        old_refresh = node.get_node_token(["agent-pub", "agent-pub-w"], wait=False)
        await _until(lambda: len(root.calls) == 1)
        root.cv = _cv("first", "badge")
        job = node.refresh_cv()
        await _until(lambda: len(root.calls) == 2)

        root.release(1)
        assert await _stays_responsive(0.1) and len(root.calls) == 2   # Nothing asked before the old one ends
        root.release(0)
        await old_refresh
        await _until(lambda: len(root.calls) == 3)
        root.release(2)
        await job
        assert [api for api, _ in root.calls][1].endswith("/cv/get") and root.calls[2][0].endswith("/token/get")
        assert _held(node) == (_hash(root.cv), _hash(root.cv))

    asyncio.run(scenario())


def test_a_notification_during_a_download_downloads_the_cv_again():
    """The download on its way may have been asked before the change a second notification announces."""
    async def scenario() -> None:
        root = _OrderedRoot(_cv("first"))
        node, _ = _agent(root)
        root.cv = _cv("first", "badge")
        job = node.refresh_cv()
        await _until(lambda: len(root.calls) == 1)
        root.cv = _cv("first", "badge", "another")
        assert node.refresh_cv() is job

        await _answer_all(root, 3)
        await job
        assert [api.rsplit("/", 2)[-2] for api, _ in root.calls] == ["cv", "token", "cv"]
        assert _held(node) == (_hash(root.cv), _hash(root.cv))

    asyncio.run(scenario())


def test_looking_up_the_addresses_of_a_node_by_name_does_not_stop_the_loop():
    async def scenario() -> None:
        root = _OrderedRoot(_cv("first"))
        node, _ = _agent(root)
        lookup = asyncio.create_task(node.ask_to_get_in_touch(node_name="someone/agent"))
        await _until(lambda: root.calls)
        assert await _stays_responsive(0.2) > 5 and not lookup.done()   # The loop runs while the root is silent
        root.release(0)
        assert await lookup is None   # The root server knows no addresses for it
        assert root.calls[0][0] == "/account/node/get/addresses"

    asyncio.run(scenario())


def test_a_token_that_fails_after_the_cv_leaves_the_old_cv_and_token_together():
    """The CV came, the token that should follow it failed: the profile keeps the old CV, which the token held still
    attests, rather than a new one that every Python node would reject in our handshakes."""
    async def scenario() -> None:
        root = _OrderedRoot(_cv("first"))
        node, _ = _agent(root)
        root.cv = _cv("first", "badge")
        root.token_fails = True
        job = node.refresh_cv()
        await _answer_all(root, 2)
        await job
        assert _held(node) == (_hash(_cv("first")), _hash(_cv("first")))

    asyncio.run(scenario())


def test_a_periodic_token_that_attests_a_newer_cv_makes_the_node_download_it():
    """A refresh of the token that falls between the badge and its notification attests the new CV, while the
    profile still has the old one: the node downloads the CV, which the token held then attests, and asks no token."""
    async def scenario() -> None:
        root = _OrderedRoot(_cv("first"))
        node, _ = _agent(root)
        root.cv = _cv("first", "badge")
        refresh = node.get_node_token(["agent-pub", "agent-pub-w"], wait=False)
        await _answer_all(root, 2)
        assert await refresh is True
        await _until(lambda: _held(node)[0] == _hash(root.cv))
        assert [api.rsplit("/", 2)[-2] for api, _ in root.calls] == ["token", "cv"]
        assert _held(node)[1] == _hash(root.cv)

    asyncio.run(scenario())


def test_a_root_that_keeps_answering_a_mismatched_pair_gets_one_round_per_refresh(monkeypatch):
    """When the root's token never attests the CV it gives, the node tries one round (the CV, then a token), says
    so once, and waits for the next refresh of the token, which tries one round again without saying it again."""
    errors = []
    monkeypatch.setattr("unaiverse.networking.node.node.log.error",
                        lambda msg, *a, **k: errors.append(msg) if "do not match" in msg else None)

    async def scenario() -> None:
        root = _OrderedRoot(_cv("first"), hold=False)
        node, _ = _agent(root)
        root.token_cv_hash = "0" * 32
        for refreshes in (1, 2):
            await node.get_node_token(["agent-pub", "agent-pub-w"], wait=False)
            await _until(lambda: len(root.calls) == 3 * refreshes)
            await _until(lambda: node._Node__root_jobs["cv"].done())
            assert await _stays_responsive(0.1) and len(root.calls) == 3 * refreshes
        assert [api.rsplit("/", 2)[-2] for api, _ in root.calls] == ["token", "cv", "token"] * 2
        assert len(errors) == 1

    asyncio.run(scenario())


def test_the_boot_makes_the_token_and_the_cv_attest_the_same_hash():
    """At boot the token is asked before the CV. A badge in between leaves a token older than the CV: a token is asked
    again. A token newer than the CV: the CV is asked again."""
    root = _OrderedRoot(_cv("first"), hold=False)
    node, _ = _agent(root)
    root.cv = _cv("first", "badge")
    node.profile.update_cv(root.cv)                         # The CV came after the badge, the token before it
    node._Node__match_token_and_cv_at_boot()
    assert [api.rsplit("/", 2)[-2] for api, _ in root.calls] == ["token"]
    assert _held(node) == (_hash(root.cv), _hash(root.cv))

    root.calls.clear()
    root.token_cv_hash = _hash(_cv("first", "badge", "another"))
    node.node_token = root.token()                          # A token newer than the CV
    root.cv, root.token_cv_hash = _cv("first", "badge", "another"), None
    node._Node__match_token_and_cv_at_boot()
    assert [api.rsplit("/", 2)[-2] for api, _ in root.calls] == ["token", "cv"]
    assert _held(node) == (_hash(root.cv), _hash(root.cv))
