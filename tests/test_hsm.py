"""Characterization tests for ``unaiverse.hsm.hsm.HybridStateMachine``.

These pin the *current* behaviour of the synchronous, network-free surface of
the HSM: construction, ``add_state``/``add_transit``, the wildcard machinery,
``to_dict``/``load`` (round-trips), the default action-selection policy,
``request_action`` and ``set_actionable``. They are plain unit tests: no clock,
no sockets, no async. The ``act*`` coroutines and the graphviz helpers are out
of scope (they dereference ``agent.im`` or need the graphviz binary).

The HSM resolves an action name to a bound method on the supplied ``agent`` via
``getattr`` + ``inspect.signature`` only (no call happens in these tests), so a
plain object whose methods declare the *exact* keyword parameters used by the
action_kwargs is enough. A bare ``**kwargs`` would NOT satisfy
``Action.check_provided_args`` (it validates each kwarg against the explicit
signature), so every fixture below names its parameters.

One genuine asymmetry between the build path and the load path is pinned and
flagged as a suspected bug (xfail, strict): ``add_global_teleport`` builds a
self-loop X->X teleport, which ``to_dict`` silently drops, so the
build->serialize->load round-trip is not stable. (A second one, the welcome
wildcard left unsubstituted on load, has been fixed and its test is green.)

The conftest autouse fixture swaps the logger's sink for a no-op, but
``log.critical`` still ``raise``s ``GenException`` (the raise sits after the sink
call), so the critical paths below really do raise.
"""
import json
from pathlib import Path

import pytest

from unaiverse.custom import Custom, GenException
from unaiverse.hsm.hsm import HybridStateMachine
from unaiverse.interaction import Interaction

pytestmark = pytest.mark.unit


# Logger contract
# The suite-wide conftest fixture swaps the logger sink for a no-op object, which
# also neuters ``log.critical`` (it becomes a silent lambda). But the HSM relies
# on ``log.critical`` actually raising ``GenException`` to abort on invalid input
# (see logger.py:599-607, "Preserve the raising contract"). To characterize the
# production behaviour of the HSM, this module restores the raising logger: with
# the inner instance set to None, ``log.critical`` raises while every other
# ``log.*`` call degrades to a print (kwargs swallowed), so the error paths that
# only log still return their value without raising.
@pytest.fixture(autouse=True)
def _raising_critical_logger():
    from unaiverse.utils.logger import log as _log
    saved = getattr(_log, "_Logger__instance", None)
    setattr(_log, "_Logger__instance", None)
    yield
    setattr(_log, "_Logger__instance", saved)


# Fixtures (light mocks; methods are never called, only introspected)

class DummyAgent:
    """Plain actionable: each method exposes the EXACT keyword params used by the
    action_kwargs of the transitions/states built in these tests."""

    def nop(self):
        pass

    def engage(self, acceptable_role=None):
        pass

    def connect_by_role(self, role=None, filter_fcn=None):
        pass

    def process(self):
        pass


class StudentAgent:
    """Actionable matching every action referenced by the social_learning
    ``student.json`` template (used as a real-world load oracle)."""

    def clear_pending_requests(self, preserve=None):
        pass

    def disconnected(self):
        pass

    def disengage(self):
        pass

    def engage(self, acceptable_role=None):
        pass

    def learn(self):
        pass

    def learn_from_student(self):
        pass

    def nop(self):
        pass

    def process(self):
        pass

    def subscribe(self):
        pass

    def teach(self):
        pass


# Anchored to this file, not the CWD: the tests must pass whether pytest runs from the
# repo dir or from the uv root one level up. The examples repo is a sibling checkout, so
# it may be absent; the two tests below skip cleanly when it is.
STUDENT_TEMPLATE = str(
    Path(__file__).resolve().parents[2] / "unaiverse-examples/worlds/social_learning/src/student.json"
)
_requires_student_template = pytest.mark.skipif(
    not Path(STUDENT_TEMPLATE).exists(),
    reason="unaiverse-examples checkout not present",
)


def _new() -> HybridStateMachine:
    return HybridStateMachine(DummyAgent())


# Construction

def test_init_seeds_default_wildcards_and_empty_machine():
    """A fresh HSM has no states/transitions, no current state, and exactly the four default wildcards."""
    h = _new()
    assert h.wildcards == {
        Custom.WORLD_WILDCARD: Custom.WORLD_WILDCARD,
        Custom.AGENT_WILDCARD: Custom.AGENT_WILDCARD,
        Custom.PARTNER_WILDCARD: Custom.PARTNER_WILDCARD,
        Custom.ROLE_WILDCARD: Custom.ROLE_WILDCARD,
    }
    assert h.states == {}
    assert h.transitions == {}
    assert h.state is None
    assert h.initial_state is None
    assert h.role is None
    assert h.enabled is True


def test_init_default_policy_is_builtin_first_requested_or_first_ready():
    """With policy=None the HSM installs its private __policy_first_requested_or_first_ready."""
    h = _new()
    assert h.policy == h._HybridStateMachine__policy_first_requested_or_first_ready


def test_init_none_agent_raises_genexception():
    """Passing agent=None blows up at construction: set_actionable(None) -> log.critical -> GenException."""
    with pytest.raises(GenException):
        HybridStateMachine(None)


# add_state

def test_add_state_first_state_becomes_current_and_initial():
    """The very first state added is auto-selected as both current and initial; later states are not."""
    h = _new()
    h.add_state("s0", action="nop")
    assert h.state == "s0"
    assert h.initial_state == "s0"
    h.add_state("s1")
    # Adding a second state does not move the current/initial pointers.
    assert h.state == "s0"
    assert h.initial_state == "s0"
    assert h.get_states() == ["s0", "s1"]


def test_add_state_blocking_and_waiting_defaults_are_overridable():
    """blocking defaults to True and waiting_time to 0., but both are honoured when given."""
    h = _new()
    h.add_state("d")  # defaults
    assert h.states["d"].blocking is True
    assert h.states["d"].waiting_time == 0.
    h.add_state("x", action="nop", blocking=False, waiting_time=1.5)
    assert h.states["x"].blocking is False
    assert h.states["x"].waiting_time == 1.5


def test_add_state_readd_reuses_id_swaps_action_and_leaves_old_action_stale():
    """Re-adding an existing state keeps its id, swaps in the new Action, and grows __id_to_action (old stays)."""
    h = _new()
    h.add_state("s", action="nop")
    sid = h.states["s"].id
    assert len(h.get_all_actions()) == 1
    h.add_state("s", action="engage")
    assert h.states["s"].id == sid  # same state id reused
    assert h.states["s"].action.name == "engage"  # new action wins
    assert len(h.get_states()) == 1  # still a single state
    # The re-add appended a brand-new Action; the previous one is left stale in the id->action list.
    assert len(h.get_all_actions()) == 2
    assert h.get_all_actions()[0].name == "nop"
    assert h.get_all_actions()[1].name == "engage"


def test_add_state_forbidden_name_all_raises():
    """The reserved state name 'all' is rejected by State.__init__ via log.critical -> GenException."""
    h = _new()
    with pytest.raises(GenException):
        h.add_state(Custom.ALL_STATES_NAME, action="nop")


# add_transit

def test_add_transit_auto_creates_missing_endpoints():
    """add_transit auto-creates both endpoint states (action=None) and the transitions dict entries."""
    h = _new()
    h.add_transit("a", "b", action="engage", args={"acceptable_role": "x"})
    assert set(h.get_states()) == {"a", "b"}
    # 'a' is the first state created, so it becomes current/initial.
    assert h.state == "a"
    assert h.initial_state == "a"
    assert "a" in h.transitions and "b" in h.transitions
    assert list(h.transitions["a"].keys()) == ["b"]
    act = h.transitions["a"]["b"][0]
    assert act.name == "engage"
    assert act.args == {"acceptable_role": "x"}


def test_add_transit_duplicate_raises_valueerror():
    """A second transition with the same name+args between the same states raises ValueError."""
    h = _new()
    h.add_transit("a", "b", action="nop")
    with pytest.raises(ValueError):
        h.add_transit("a", "b", action="nop")


def test_add_transit_same_name_different_args_coexist():
    """Two transitions sharing the action name but differing in args are NOT duplicates and coexist."""
    h = _new()
    h.add_transit("a", "b", action="engage", args={"acceptable_role": "teacher"})
    h.add_transit("a", "b", action="engage", args={"acceptable_role": "student"})
    acts = h.transitions["a"]["b"]
    assert len(acts) == 2
    assert {a.args["acceptable_role"] for a in acts} == {"teacher", "student"}


def test_add_transit_meta_times_are_carried_onto_the_action():
    """total_time/timeout/delay land on the Action and surface under the meta keys in to_dict."""
    h = _new()
    h.add_transit("a", "b", action="nop", total_time=5.0, timeout=3.0, delay=2.0)
    on = h.to_dict()["transitions"]["a"][0]["on"]
    assert on[Custom.SECONDS_ARG_NAMES[0]] == 5.0
    assert on[Custom.TIMEOUT_ARG_NAMES[0]] == 3.0
    assert on[Custom.DELAY_ARG_NAMES[0]] == 2.0


def test_add_transit_json_suffix_missing_file_raises_filenotfound():
    """A to_state ending in '.json' that does not exist raises FileNotFoundError."""
    h = _new()
    with pytest.raises(FileNotFoundError):
        h.add_transit("a", "does_not_exist.json", action="nop")


def test_add_transit_json_suffix_plugs_in_sub_hsm(tmp_path):
    """A '.json' destination loads the referenced HSM, includes its states and wires
    from_state -> sub.initial_state (the entry action is the one passed to add_transit)."""
    agent = DummyAgent()
    sub = HybridStateMachine(agent)
    sub.add_transit("subinit", "subdone", action="nop")
    sub_path = tmp_path / "sub.json"
    sub.save(str(sub_path))

    main = HybridStateMachine(agent)
    main.add_state("main_start", action="nop")
    main.add_transit("main_start", str(sub_path), action="engage", args={"acceptable_role": "x"})

    assert {"main_start", "subinit", "subdone"} <= set(main.states.keys())
    assert "subinit" in main.transitions["main_start"]
    assert main.transitions["main_start"]["subinit"][0].name == "engage"
    # main_start was the first state created, so it stays initial/current after the plug-in.
    assert main.initial_state == "main_start"


# Wildcards

def test_apply_wildcards_substitutes_nested_action_args():
    """apply_wildcards pushes a string wildcard into the (nested) action args of every state/transition."""
    h = _new()
    h.add_wildcards({Custom.ROLE_WILDCARD: "hero"})
    h.add_state("s", action="connect_by_role", args={"role": Custom.ROLE_WILDCARD})
    h.apply_wildcards()
    assert h.states["s"].action.args == {"role": "hero"}


def test_apply_wildcards_non_string_value_replaces_whole_value():
    """A non-string wildcard value replaces a whole-value match, so '<count>' becomes the int 5."""
    h = _new()
    h.add_state("s", action="connect_by_role", args={"role": "<count>"})
    h.add_wildcards({"<count>": 5})
    h.apply_wildcards()
    assert h.states["s"].action.args == {"role": 5}
    assert isinstance(h.states["s"].action.args["role"], int)


def test_set_wildcards_completely_replaces_and_drops_defaults():
    """set_wildcards swaps the whole dict, so the default <world>/<agent>/<partner> keys are dropped."""
    h = _new()
    h.set_wildcards({Custom.ROLE_WILDCARD: "x"})
    assert h.wildcards == {Custom.ROLE_WILDCARD: "x"}
    assert Custom.WORLD_WILDCARD not in h.wildcards
    assert Custom.AGENT_WILDCARD not in h.wildcards


def test_add_wildcards_preserves_defaults_and_overwrites_collisions():
    """add_wildcards is a dict.update: it keeps the defaults and overwrites only colliding keys."""
    h = _new()
    h.add_wildcards({Custom.ROLE_WILDCARD: "boss", "<extra>": "ok"})
    assert h.wildcards[Custom.ROLE_WILDCARD] == "boss"
    assert h.wildcards["<extra>"] == "ok"
    # Untouched defaults survive.
    assert h.wildcards[Custom.WORLD_WILDCARD] == Custom.WORLD_WILDCARD


def test_update_wildcard_unknown_key_logs_and_does_not_change():
    """update_wildcard on an unknown key logs an error (no raise) and leaves the dict untouched."""
    h = _new()
    before = dict(h.wildcards)
    h.update_wildcard("<unknown>", "v")  # log.error, no GenException
    assert h.wildcards == before


def test_set_role_updates_role_wildcard_and_applies():
    """set_role stores the role and substitutes <role> everywhere via apply_wildcards."""
    h = _new()
    h.add_state("s", action="connect_by_role", args={"role": Custom.ROLE_WILDCARD})
    h.set_role("captain")
    assert h.role == "captain"
    assert h.wildcards[Custom.ROLE_WILDCARD] == "captain"
    assert h.states["s"].action.args == {"role": "captain"}


# to_dict (serialization shape)

def test_to_dict_top_level_shape_and_machine_block():
    """to_dict emits machine/states/transitions/teleports with the expected machine sub-keys."""
    h = _new()
    h.set_role("worker")
    h.add_state("a", action="nop")
    h.add_transit("a", "b", action="engage", args={"acceptable_role": "x"})
    d = h.to_dict()

    assert set(d.keys()) == {"machine", "states", "transitions", "teleports"}
    assert d["machine"]["role"] == "worker"
    assert d["machine"]["initial_state"] == "a"
    assert d["machine"]["msg"] is None
    assert d["machine"]["options"] == {
        "highlight_blocking_states_in_messages": False,
        "show_action_ticks_after_messages": False,
        "show_action_request_after_messages": False,
    }
    # State.to_dict shape.
    assert d["states"]["a"] == {
        "action": "nop",
        "action_kwargs": {},
        "msg": None,
        "blocking": True,
        Custom.TIME_TO_WAIT_BEFORE_ACTING_ARG_NAMES[0]: 0.0,
    }
    # A non-teleport transition is listed under 'transitions', and 'teleports' is empty.
    on = d["transitions"]["a"][0]
    assert on["goto"] == "b"
    assert on["on"]["action"] == "engage"
    assert on["on"]["action_kwargs"] == {"acceptable_role": "x"}
    assert d["teleports"] == {}


def test_to_dict_xmlcharref_encodes_non_ascii_state_msg():
    """Non-ASCII characters in a state message are xmlcharref-encoded in to_dict."""
    h = _new()
    h.add_state("a", action="nop", msg="café")
    assert h.to_dict()["states"]["a"]["msg"] == "caf&#233;"


def test_to_dict_per_source_teleport_stays_under_source_key():
    """A teleport from a single source (not covering all-but-destination) is NOT collapsed under 'all'."""
    h = _new()
    h.add_transit("a", "b", action="engage", args={"acceptable_role": "x"})
    h.add_transit("b", "c", action="process")
    h.add_teleport(from_state="a", to_state="c", action="nop")
    tp = h.to_dict()["teleports"]
    assert Custom.ALL_STATES_NAME not in tp
    assert list(tp.keys()) == ["a"]
    assert tp["a"][0]["goto"] == "c"
    assert tp["a"][0]["on"]["action"] == "nop"


# to_dict / load round-trip

def _build_roundtrip_machine(agent) -> HybridStateMachine:
    h = HybridStateMachine(agent)
    h.set_role("worker")
    h.add_state("a", action="nop", msg="hi")
    h.add_transit("a", "b", action="engage", args={"acceptable_role": "x"}, timeout=3.0)
    h.add_transit("b", "a", action="nop", delay=2.0)
    return h


def test_to_dict_load_roundtrip_is_stable_without_teleports():
    """A teleport-free machine survives to_dict -> json -> load -> to_dict byte-for-byte."""
    h = _build_roundtrip_machine(DummyAgent())
    d1 = h.to_dict()
    h2 = HybridStateMachine(DummyAgent()).load(json.dumps(d1))
    d2 = h2.to_dict()
    assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)
    # Spot-check a few rebuilt fields.
    assert h2.role == "worker"
    assert h2.initial_state == "a"
    assert h2.state == "a"
    assert set(h2.get_states()) == {"a", "b"}


def test_load_missing_states_key_raises_genexception():
    """A 'machine'-bearing payload without the 'states' key hits an unguarded log.critical -> GenException."""
    payload = {
        "machine": {
            "role": "r",
            "initial_state": "a",
            "msg": None,
            "options": {},
        },
        # 'states' deliberately omitted.
        "transitions": {},
    }
    with pytest.raises(GenException):
        HybridStateMachine(DummyAgent()).load(json.dumps(payload))


# Default policy ordering

def _two_actions(ready_first=True):
    """Return (hsm, ready_action, gated_action) where the first is ready and the
    second is initially not inner-ready (built with ready=False)."""
    h = _new()
    h.add_transit("a", "b", action="nop", ready=ready_first)
    h.add_transit("a", "c", action="engage", args={"acceptable_role": "t"}, ready=False)
    return h, h.transitions["a"]["b"][0], h.transitions["a"]["c"][0]


def test_policy_picks_first_ready_action_with_no_interaction():
    """With no requests, the policy returns the first inner-ready action and a None interaction."""
    h, ready_act, gated = _two_actions()
    idx, inter = h.policy([ready_act, gated])
    assert idx == 0
    assert inter is None


def test_policy_high_priority_beats_first_ready():
    """A high-priority action is selected regardless of position; with no queue it yields a None interaction."""
    h, ready_act, gated = _two_actions()
    gated.set_high_priority()
    idx, inter = h.policy([ready_act, gated])
    assert idx == 1
    assert inter is None


def test_policy_requested_interaction_beats_first_ready():
    """An action carrying a queued interaction outranks an earlier merely-ready action and returns that interaction."""
    h = _new()
    h.add_transit("a", "b", action="nop", ready=True)
    h.add_transit("a", "c", action="engage", args={"acceptable_role": "t"})
    ready_act = h.transitions["a"]["b"][0]
    requested = h.transitions["a"]["c"][0]
    ok = h.request_action(Interaction(
        action_name="engage", action_kwargs={"acceptable_role": "t"},
        requester="peerA", from_state="a", to_state="c", id="req1"))
    assert ok is True
    idx, inter = h.policy([ready_act, requested])
    assert idx == 1
    assert inter is not None
    assert inter.uuid == "req1_peerA"


def test_policy_high_priority_returns_oldest_queued_interaction():
    """A high-priority action that also has a queued interaction returns that interaction (not None)."""
    h = _new()
    h.add_transit("a", "b", action="engage", args={"acceptable_role": "t"})
    act = h.transitions["a"]["b"][0]
    act.set_high_priority()
    h.request_action(Interaction(
        action_name="engage", action_kwargs={"acceptable_role": "t"},
        requester="peerA", from_state="a", to_state="b", id="hp1"))
    idx, inter = h.policy([act])
    assert idx == 0
    assert inter is not None
    assert inter.uuid == "hp1_peerA"


def test_policy_no_candidate_returns_minus_one_none():
    """When nothing is ready, high-priority or requested, the policy returns (-1, None)."""
    h, ready_act, gated = _two_actions()
    ready_act.set_as_not_ready()  # now neither action is inner-ready, no queues, no high-priority
    idx, inter = h.policy([ready_act, gated])
    assert idx == -1
    assert inter is None


# request_action (accept + every reject path)

def _request_machine() -> HybridStateMachine:
    h = _new()
    h.add_state("a", action="nop")  # 'a' is the initial/current state
    h.add_transit("a", "b", action="engage", args={"acceptable_role": "teacher"})
    return h


def test_request_action_accepts_and_queues_matching_interaction():
    """A well-formed request matching an outer-allowing transition is queued and returns True."""
    h = _request_machine()
    inter = Interaction(action_name="engage", action_kwargs={"acceptable_role": "teacher"},
                        requester="peerA", from_state="a", to_state="b", id="r1")
    assert h.request_action(inter) is True
    queued = h.transitions["a"]["b"][0].get_list_of_interactions()
    assert len(queued) == 1
    assert queued.get_oldest_interaction().uuid == "r1_peerA"


def test_request_action_scans_all_destinations_when_to_state_is_none():
    """With to_state=None the request is matched against every destination of from_state."""
    h = _request_machine()
    inter = Interaction(action_name="engage", action_kwargs={"acceptable_role": "teacher"},
                        requester="peerA", from_state="a", id="r2")  # no to_state
    assert h.request_action(inter) is True


def test_request_action_no_action_name_returns_false():
    """A nameless interaction is rejected (log.error) and returns False, without raising."""
    h = _request_machine()
    inter = Interaction(action_name=None, requester="peerA", from_state="a", to_state="b")
    assert h.request_action(inter) is False


def test_request_action_invalid_from_state_returns_false():
    """A from_state absent from the transitions map is rejected and returns False."""
    h = _request_machine()
    inter = Interaction(action_name="engage", requester="peerA", from_state="ZZZ", to_state="b")
    assert h.request_action(inter) is False


def test_request_action_invalid_to_state_returns_false():
    """A to_state that is not a valid destination from from_state is rejected and returns False."""
    h = _request_machine()
    inter = Interaction(action_name="engage", requester="peerA", from_state="a", to_state="ZZZ")
    assert h.request_action(inter) is False


def test_request_action_unknown_action_name_returns_false():
    """A request whose action_name matches no transition is rejected and returns False."""
    h = _request_machine()
    inter = Interaction(action_name="nope", requester="peerA", from_state="a", to_state="b")
    assert h.request_action(inter) is False


def test_request_action_disallowed_outer_interactions_returns_false():
    """If the target action forbids outer interactions (set_as_ready -> outer False) the request returns False."""
    h = _request_machine()
    h.transitions["a"]["b"][0].set_as_ready()  # outer becomes False
    inter = Interaction(action_name="engage", action_kwargs={"acceptable_role": "teacher"},
                        requester="peerA", from_state="a", to_state="b")
    assert h.request_action(inter) is False


# set_actionable

def test_set_actionable_rebinds_agent_on_states_and_actions():
    """set_actionable rebinds self.agent and the .agent of every state action and every Action."""
    h = _request_machine()
    new_agent = DummyAgent()
    h.set_actionable(new_agent)
    assert h.agent is new_agent
    assert h.states["a"].action.agent is new_agent
    assert all(act.agent is new_agent for act in h.get_all_actions())


def test_set_actionable_none_raises_genexception():
    """set_actionable(None) calls log.critical, which raises GenException (the trailing 'return' is dead code)."""
    h = _request_machine()
    with pytest.raises(GenException):
        h.set_actionable(None)


# Real-world template oracle (social_learning student.json)

@_requires_student_template
def test_load_real_student_template_rebuilds_machine():
    """The shipped student.json loads into the expected role, states and a known transition."""
    h = HybridStateMachine(StudentAgent()).load(STUDENT_TEMPLATE)
    assert h.role == "student"
    assert h.initial_state == "init"
    assert h.state == "init"
    assert set(h.get_states()) == {
        "init", "teacher_engaged", "finished_learning", "listening_to_best_student",
    }
    assert h.welcome_msg is not None
    # init --engage(acceptable_role=teacher, ready=False)--> teacher_engaged
    init_engage = h.transitions["init"]["teacher_engaged"][0]
    assert init_engage.name == "engage"
    assert init_engage.args == {"acceptable_role": "teacher"}
    assert init_engage.ready is False


@_requires_student_template
def test_load_real_student_template_roundtrips_via_to_dict():
    """Reserializing the loaded student.json and reloading it is byte-for-byte stable."""
    h = HybridStateMachine(StudentAgent()).load(STUDENT_TEMPLATE)
    d1 = h.to_dict()
    h2 = HybridStateMachine(StudentAgent()).load(json.dumps(d1))
    assert json.dumps(d1, sort_keys=True) == json.dumps(h2.to_dict(), sort_keys=True)


# Suspected bugs (characterized; xfail strict on the intended behaviour)

def test_add_global_teleport_creates_self_loop_on_destination_actual():
    """ACTUAL behaviour: add_global_teleport(to_state=X) wrongly builds an X->X self-loop teleport.

    The 'all' branch of add_transit skips only states named 'all' (a forbidden, never-matching name)
    instead of skipping the destination, so the destination gets a teleport back to itself.
    """
    h = _new()
    h.set_role("w")
    h.add_state("a", action="nop")
    h.add_state("b")
    h.add_global_teleport(to_state="b", action="nop")
    # a -> b is wanted; b -> b is the spurious self-loop.
    assert "b" in h.transitions["a"]
    assert "b" in h.transitions["b"]  # <-- the self-loop the fix should remove
    assert h.transitions["b"]["b"][0].is_teleport() is True


@pytest.mark.xfail(strict=True,
                   reason="add_global_teleport builds an X->X self-loop that to_dict drops, "
                          "so the build->serialize->load round-trip is not stable "
                          "(hsm.py:704-714 vs load at hsm.py:1483).")
def test_add_global_teleport_roundtrip_is_stable_intended():
    """INTENDED behaviour (currently failing): a machine built with add_global_teleport should survive
    to_dict -> load -> to_dict unchanged. It does not, because the spurious self-loop is dropped on save."""
    h = HybridStateMachine(DummyAgent())
    h.set_role("w")
    h.add_state("a", action="nop")
    h.add_state("b")
    h.add_global_teleport(to_state="b", action="nop")
    d1 = h.to_dict()
    h2 = HybridStateMachine(DummyAgent()).load(json.dumps(d1))
    assert json.dumps(d1, sort_keys=True) == json.dumps(h2.to_dict(), sort_keys=True)


def test_load_applies_welcome_wildcard():
    """load() installs the welcome message and only then set_role, so a freshly
    loaded machine has the role wildcard already resolved, matching what the
    build path produces."""
    h = HybridStateMachine(DummyAgent())
    h.set_welcome_message("Hi <role> ok")
    h.set_role("boss")  # build path: role set AFTER welcome -> substituted now
    assert h.welcome_msg == "Hi boss ok"

    h2 = HybridStateMachine(DummyAgent()).load(json.dumps(h.to_dict()))
    assert h2.role == "boss"
    assert h2.welcome_msg == "Hi boss ok"
