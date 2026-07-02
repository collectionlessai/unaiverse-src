"""Characterization tests for the ``WorldRoles`` area.

These pin the *current* behavior of the world-side role machinery:

  * ``World.assign_role`` (base/default role assignment),
  * ``AgentBasics.policy_default`` as inherited by ``World`` (action-selection
    policy over a list of actions),
  * the CUSTOM_ROLES folder-discovery contract in
    ``World.load_and_refactor_action_file_and_behav_files`` (the light, pre-load
    error/discovery paths only; the happy path is a heavy in-memory import + HSM
    load and is intentionally not exercised here).

Everything here is a pure unit: a ``World`` is built directly on a temp folder
with a non-None dummy ``stats`` so the default ``Stats`` DB is never created, and
no network is touched. The autouse logger-silencing fixture from ``conftest``
keeps ``log.*`` calls quiet.

One cross-layer bug (the single-master demotion being undone at the node layer)
is documented with an ``xfail`` test that reproduces the exact integer arithmetic
of the ``node.py`` call site; see ``suspected_bugs`` in the run report.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from unaiverse.world import World
from unaiverse.custom import GenException
from unaiverse.networking.session.sess_msg import LOCAL_ERROR

pytestmark = pytest.mark.unit


# ── Test doubles ─────────────────────────────────────────────────────────────

class _DummyStats:
    """A non-None sentinel handed to ``World(stats=...)`` so the default
    folder-backed ``Stats`` DB is never created. ``World.__init__`` only asserts
    ``self.stats is not None``; nothing in these tests touches it."""


class FakeProfile:
    """The slice of ``NodeProfile`` that ``assign_role`` touches: just
    ``get_dynamic_profile()`` returning a plain dict."""

    def __init__(self, dynamic: dict | None = None):
        self._dynamic = dynamic if dynamic is not None else {}

    def get_dynamic_profile(self) -> dict:
        return self._dynamic


class FakeInteractionList:
    """A list-of-interactions stand-in: ``policy_default`` only calls ``len(...)``
    and ``get_oldest_interaction()`` on it."""

    def __init__(self, length: int = 0, oldest=None):
        self._length = length
        self._oldest = oldest

    def __len__(self) -> int:
        return self._length

    def get_oldest_interaction(self):
        return self._oldest


class FakeAction:
    """An ``Action`` stand-in exposing exactly the surface ``policy_default``
    reads: ``is_high_priority``, ``get_list_of_interactions()`` and
    ``is_ready(consider_interactions=...)``."""

    def __init__(self, is_high_priority: bool = False, length: int = 0,
                 oldest=None, ready: bool = False):
        self.is_high_priority = is_high_priority
        self._list = FakeInteractionList(length=length, oldest=oldest)
        self._ready = ready
        self.ready_calls: list[bool] = []

    def get_list_of_interactions(self) -> FakeInteractionList:
        return self._list

    def is_ready(self, consider_interactions: bool = False) -> bool:
        self.ready_calls.append(consider_interactions)
        return self._ready


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def make_world(tmp_path):
    """Factory building a real ``World`` on a temp folder with dummy stats.

    Pass an explicit ``folder`` (a ``pathlib.Path``) for the discovery tests that
    need to populate it first; otherwise a fresh empty subfolder is used."""
    counter = {"n": 0}

    def _make(folder=None) -> World:
        if folder is None:
            counter["n"] += 1
            folder = tmp_path / f"world{counter['n']}"
            folder.mkdir()
        return World(world_folder=str(folder), stats=_DummyStats())

    return _make


@pytest.fixture
def world(make_world) -> World:
    return make_world()


# ── World.assign_role ────────────────────────────────────────────────────────

def test_assign_role_first_master_returns_world_master(world):
    """The first master (no masters registered yet) is assigned 'world_master'."""
    assert len(world.world_masters) == 0
    role = world.assign_role(FakeProfile(), is_world_master=True)
    assert role == "world_master"
    assert isinstance(role, str)


def test_assign_role_non_master_returns_world_agent(world):
    """A non-master joiner is always assigned 'world_agent'."""
    assert world.assign_role(FakeProfile(), is_world_master=False) == "world_agent"


def test_assign_role_second_master_stays_world_master(world):
    """With a master already present, a second would-be master still gets
    'world_master': the base policy places no single-master cap (who may be a master
    is governed upstream by the world_masters_node_ids allow-list, not here)."""
    world.world_masters["already-a-master"] = object()
    assert world.assign_role(FakeProfile(), is_world_master=True) == "world_master"


def test_assign_role_returns_string_not_bits(world):
    """assign_role returns a role STRING, never the integer role bits."""
    role = world.assign_role(FakeProfile(), is_world_master=True)
    assert role == "world_master"
    assert not isinstance(role, int)
    # And the string is a valid key of the base role vocabulary.
    assert role in World.ROLE_STR_TO_BITS


def test_assign_role_never_returns_none_across_all_combinations(make_world):
    """None of the four (is_world_master, masters-present) combinations returns
    None; the base World always yields a concrete role string."""
    # No masters registered.
    w0 = make_world()
    assert w0.assign_role(FakeProfile(), is_world_master=True) == "world_master"
    assert w0.assign_role(FakeProfile(), is_world_master=False) == "world_agent"
    # One master already registered: the base policy has no single-master cap.
    w1 = make_world()
    w1.world_masters["m"] = object()
    assert w1.assign_role(FakeProfile(), is_world_master=True) == "world_master"
    assert w1.assign_role(FakeProfile(), is_world_master=False) == "world_agent"


def test_assign_role_asserts_is_world(world):
    """assign_role asserts it is being called on a world; flipping the flag off
    trips the assertion."""
    world.is_world = False
    with pytest.raises(AssertionError):
        world.assign_role(FakeProfile(), is_world_master=False)


def test_assign_role_guessed_location_branch_is_a_noop(world):
    """The 'guessed_location == Some Dummy Location...' branch is a dead ``pass``:
    a profile carrying that exact value still resolves to the ordinary role."""
    profile = FakeProfile({"guessed_location": "Some Dummy Location, Just An Example Here"})
    assert world.assign_role(profile, is_world_master=False) == "world_agent"
    assert world.assign_role(profile, is_world_master=True) == "world_master"


# ── Cross-layer role bits (node layer) ───────────────────────────────────────
# node.py:2278-2282 (public world host) computes the joiner's role like this:
#     role_str = self.world.assign_role(profile=profile, is_world_master=...)
#     role = self.world.ROLE_STR_TO_BITS[role_str]
#     role = role | (Agent.ROLE_WORLD_MASTER if is_world_master else Agent.ROLE_WORLD_AGENT)
# We reproduce that integer arithmetic here (the real call site needs the whole
# Node shell, out of scope for this pure unit) to characterize what the node
# layer does with assign_role's output.

def _node_layer_role_bits(world: World, profile, is_world_master: bool) -> int:
    """Exact reproduction of node.py:2278-2282 for a base World."""
    role_str = world.assign_role(profile=profile, is_world_master=is_world_master)
    role = world.ROLE_STR_TO_BITS[role_str]
    role = role | (World.ROLE_WORLD_MASTER if is_world_master else World.ROLE_WORLD_AGENT)
    return role


def test_node_layer_first_master_role_bits(world):
    """For the first master the node-layer role bits are ROLE_WORLD_MASTER."""
    bits = _node_layer_role_bits(world, FakeProfile(), is_world_master=True)
    assert bits == World.ROLE_WORLD_MASTER == 3


def test_node_layer_non_master_role_bits(world):
    """For a non-master the node-layer role bits are ROLE_WORLD_AGENT."""
    bits = _node_layer_role_bits(world, FakeProfile(), is_world_master=False)
    assert bits == World.ROLE_WORLD_AGENT == 1


def test_node_layer_second_master_is_world_master(world):
    """With a master already present, a second would-be master resolves to
    world_master bits at the node layer too: assign_role returns 'world_master'
    (no single-master cap) and node.py OR-s ROLE_WORLD_MASTER in, so bits == 3."""
    world.world_masters["already-a-master"] = object()
    bits = _node_layer_role_bits(world, FakeProfile(), is_world_master=True)
    assert bits == World.ROLE_WORLD_MASTER == 3


# ── policy_default (inherited by World) ──────────────────────────────────────

def test_policy_default_empty_list_returns_miss(world):
    """An empty action list selects nothing: (-1, None)."""
    assert world.policy_default([]) == (-1, None)


def test_policy_default_all_passes_miss_returns_minus_one(world):
    """An action that is not high-priority, has no pending interactions and is not
    ready falls through all three passes to (-1, None)."""
    action = FakeAction(is_high_priority=False, length=0, ready=False)
    assert world.policy_default([action]) == (-1, None)


def test_policy_default_high_priority_with_empty_interactions(world):
    """A high-priority action with an empty request list returns its index and a
    None interaction (pass 1, len == 0 branch)."""
    action = FakeAction(is_high_priority=True, length=0)
    assert world.policy_default([action]) == (0, None)


def test_policy_default_high_priority_with_interactions_returns_oldest(world):
    """A high-priority action with a non-empty request list returns its index and
    the oldest interaction (pass 1, len > 0 branch)."""
    sentinel = object()
    action = FakeAction(is_high_priority=True, length=2, oldest=sentinel)
    idx, inter = world.policy_default([action])
    assert idx == 0
    assert inter is sentinel


def test_policy_default_high_priority_beats_pending_and_ready_regardless_of_index(world):
    """Pass 1 scans the whole list for the first high-priority action before any
    pending/ready consideration, so a high-priority action wins even when it sits
    after a pending one and a ready one."""
    sentinel = object()
    pending = FakeAction(is_high_priority=False, length=1, oldest=object())  # idx 0
    ready = FakeAction(is_high_priority=False, ready=True)                    # idx 1
    high = FakeAction(is_high_priority=True, length=3, oldest=sentinel)       # idx 2
    idx, inter = world.policy_default([pending, ready, high])
    assert idx == 2
    assert inter is sentinel


def test_policy_default_first_high_priority_wins(world):
    """When several actions are high-priority, the first one (lowest index) wins."""
    first = FakeAction(is_high_priority=True, length=0)   # idx 0 -> None interaction
    second = FakeAction(is_high_priority=True, length=5, oldest=object())  # idx 1
    idx, inter = world.policy_default([first, second])
    assert idx == 0
    assert inter is None


def test_policy_default_pending_interactions_second_pass(world):
    """With no high-priority action, the first action whose request list is
    non-empty is selected with its oldest interaction (pass 2)."""
    sentinel = object()
    empty = FakeAction(is_high_priority=False, length=0)               # idx 0, skipped
    pending = FakeAction(is_high_priority=False, length=1, oldest=sentinel)  # idx 1
    idx, inter = world.policy_default([empty, pending])
    assert idx == 1
    assert inter is sentinel


def test_policy_default_pending_precedes_ready(world):
    """Pass 2 (pending interactions) runs before pass 3 (ready), so a pending
    action at a higher index beats a ready-but-empty action at a lower index."""
    sentinel = object()
    ready = FakeAction(is_high_priority=False, length=0, ready=True)         # idx 0
    pending = FakeAction(is_high_priority=False, length=1, oldest=sentinel)  # idx 1
    idx, inter = world.policy_default([ready, pending])
    assert idx == 1
    assert inter is sentinel


def test_policy_default_ready_third_pass_returns_none_interaction(world):
    """When nothing is high-priority and nothing has pending interactions, the
    first ready action is selected with a None interaction (pass 3)."""
    not_ready = FakeAction(is_high_priority=False, length=0, ready=False)  # idx 0
    ready = FakeAction(is_high_priority=False, length=0, ready=True)       # idx 1
    idx, inter = world.policy_default([not_ready, ready])
    assert idx == 1
    assert inter is None
    # Pass 3 probes readiness without considering interactions.
    assert ready.ready_calls == [False]


# ── CUSTOM_ROLES discovery (light error / discovery paths only) ──────────────

def test_discovery_empty_folder_raises_no_role_files(world):
    """An empty world folder raises before any heavy loading: no *.json roles."""
    with pytest.raises(GenException, match="No world-role files"):
        world.load_and_refactor_action_file_and_behav_files()


def test_discovery_only_non_json_files_raises_no_role_files(make_world, tmp_path):
    """A folder with files but no top-level *.json still raises 'No world-role
    files' (discovery is .json-driven)."""
    folder = tmp_path / "noroles"
    folder.mkdir()
    (folder / "readme.txt").write_text("not a role")
    (folder / "agent.py").write_text("# code, but no json role")
    world = make_world(folder)
    with pytest.raises(GenException, match="No world-role files"):
        world.load_and_refactor_action_file_and_behav_files()


def test_discovery_json_without_py_raises_mismatch(make_world, tmp_path):
    """A single <role>.json with no matching .py and no agent.py raises the
    role/.py mismatch error before any in-memory import."""
    folder = tmp_path / "nopy"
    folder.mkdir()
    (folder / "student.json").write_text("{}")
    world = make_world(folder)
    with pytest.raises(GenException, match="Mismatching JSON-file roles and .py files"):
        world.load_and_refactor_action_file_and_behav_files()
    # CUSTOM_ROLES was discovered before the mismatch was raised.
    assert world.CUSTOM_ROLES == ["student"]


def test_discovery_custom_roles_top_level_case_insensitive(make_world, tmp_path):
    """CUSTOM_ROLES discovery is top-level only (os.listdir, not recursive),
    accepts .json case-insensitively, keeps the basename's original case, and
    ignores non-.json files. Order is filesystem-dependent, so assert as a set."""
    folder = tmp_path / "roles"
    folder.mkdir()
    (folder / "Student.JSON").write_text("{}")   # uppercase extension, mixed-case stem
    (folder / "teacher.json").write_text("{}")
    (folder / "notes.txt").write_text("ignored")  # not a role
    nested = folder / "sub"
    nested.mkdir()
    (nested / "hidden.json").write_text("{}")     # nested -> must NOT be discovered
    world = make_world(folder)

    # Two roles with no .py -> mismatch is raised after CUSTOM_ROLES is set.
    with pytest.raises(GenException, match="Mismatching JSON-file roles and .py files"):
        world.load_and_refactor_action_file_and_behav_files()

    assert set(world.CUSTOM_ROLES) == {"Student", "teacher"}
    assert "hidden" not in world.CUSTOM_ROLES  # nested .json was not picked up


def test_discovery_is_skipped_when_not_a_world(world):
    """The discovery body is gated by ``self.is_world``: with the flag cleared the
    method is a no-op and returns None even on an empty folder (no raise)."""
    world.is_world = False
    assert world.load_and_refactor_action_file_and_behav_files() is None


# ── set_role send-failure teardown (BUG 3) ───────────────────────────────────

def _set_role_fake_self(send_ok):
    """The slice of ``World`` that ``set_role`` touches, with a controllable send
    outcome. Roles are kept fake (a single custom bit) so the test does not depend
    on the real role-bit vocabulary."""
    return SimpleNamespace(
        is_world=True,
        node_conn=SimpleNamespace(
            get_role=lambda peer_id: 1,                 # current role: world_agent
            set_role=lambda peer_id, role: None,        # spy no-op
            send=AsyncMock(return_value=send_ok),
        ),
        ROLE_BITS_TO_STR={4: "custom_role"},
        role_to_behav={"custom_role": "<behav>"},
        node_purge_fcn=AsyncMock(),
        role_changed_by_world=False,
    )


def test_set_role_send_failure_purges_with_local_error():
    """When the ROLE_SUGGESTION send fails, set_role purges the peer carrying
    LOCAL_ERROR, so __purge emits a GOAWAY instead of a silent drop (BUG 3)."""
    me = _set_role_fake_self(send_ok=False)

    # role=5 -> new_role_without_base_int=4 (!= the current role 1) -> the change is
    # broadcast, the send fails, and the peer is purged.
    asyncio.run(World.set_role(me, "peerX", 5))

    me.node_purge_fcn.assert_awaited_once_with("peerX", reason=LOCAL_ERROR)
    # The success branch (role_changed_by_world=True) never ran.
    assert me.role_changed_by_world is False


def test_set_role_send_success_does_not_purge():
    """Positive control: when the ROLE_SUGGESTION send succeeds, no purge happens
    and the world records the role change."""
    me = _set_role_fake_self(send_ok=True)

    asyncio.run(World.set_role(me, "peerX", 5))

    me.node_purge_fcn.assert_not_awaited()
    assert me.role_changed_by_world is True
