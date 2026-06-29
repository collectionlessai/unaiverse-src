"""Characterization tests for the *pure* action surface of ``Agent``
(area: AgentActionsPure).

These pin the CURRENT behaviour of the agent's scope/playlist/engagement
coroutines, none of which touch the real network, the clock or the Go layer:

  * the ``@action`` decorator                       (agent.py:34)
  * ``engage``                                      (agent.py:593)
  * ``disengage_all``                               (agent.py:695)
  * ``find_agents`` / ``get_agents_by_role``        (agent.py:1148 / agent_basics.py:673)
  * ``next_pref_stream`` / ``first_pref_stream``    (agent.py:1170 / 1188)
  * ``check_pref_stream``                           (agent.py:1204)
  * ``set_pref_streams``                            (agent.py:1244)
  * ``evaluate``                                    (agent.py:1273)
  * ``compare_eval``                                (agent.py:1329)
  * ``nop`` / ``all_sent_completed`` /
    ``all_engagements_completed``                   (agent.py:825 / 407 / 869)

A bare ``Agent(proc=None)`` builds fully OFFLINE (verified): no node_conn, no
sockets, no NTP. Every scope state variable (``_available``, ``_engaged_agents``,
``_found_agents``, ``_valid_cmp_agents``, ``_eval_results``,
``_preferred_streams``, ``_cur_preferred_stream``, ``_repeat``) is initialised in
``Agent.__init__``. Each action is a coroutine, driven here with
``asyncio.run`` (the suite has no pytest-asyncio).

Two cross-cutting facts of the ``@action`` decorator are baked into the
assertions below: it swallows EVERY exception (AssertionError, KeyError,
IndexError, ZeroDivisionError) and returns ``False``, and it logs (but does not
raise) when the wrapped fn returns ``None``. So every "edge" case here surfaces
as a returned ``False``, never as a raised exception.

The conftest autouse ``_silence_unaiverse_logger`` fixture swallows the heavy
``log.*`` calls and inserts ``../src`` on ``sys.path`` so the import resolves
offline.

Four defects surfaced while writing these tests are reported under
``suspected_bugs``; the two with an unambiguous "correct" behaviour are pinned
with ``xfail(strict=True)`` so the suite stays green today and flags (via XPASS)
the moment either is fixed. The other two are documented with passing
characterization tests that assert the actual (buggy) behaviour, commented as
such.
"""

import asyncio

import pytest

from unaiverse.agent import Agent
from unaiverse.agent_basics import AgentBasics

pytestmark = pytest.mark.unit


# ── Helpers ─────────────────────────────────────────────────────────────────

def _run(coro):
    return asyncio.run(coro)


class FakeInteraction:
    """Minimal stand-in for ``Interaction``: ``engage`` only reads ``.requester``."""

    def __init__(self, requester):
        self.requester = requester


class FakeConn:
    """Stand-in for ``node_conn`` exposing only ``find_addrs_by_role`` (the single
    node-layer seam touched by this whole area, via ``get_agents_by_role``).

    ``mapping`` is role_int -> list of peer_ids; everything else is empty.
    """

    def __init__(self, mapping=None):
        self.mapping = dict(mapping or {})
        self.calls = []

    def find_addrs_by_role(self, role_int, return_peer_ids_too=False):
        self.calls.append(role_int)
        peers = list(self.mapping.get(role_int, []))
        return (["/addr"] * len(peers), peers)


class FakeProfile:
    """Stand-in for ``NodeProfile``: ``compare_eval`` reads only these two keys."""

    def __init__(self, email, node_name):
        self._static = {"email": email, "node_name": node_name}

    def get_static_profile(self):
        return self._static


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def agent():
    """A bare, fully-offline Agent with default scope state."""
    return Agent(proc=None)


@pytest.fixture
def roled_agent():
    """An Agent with the custom roles ('teacher', 'student') registered.

    ``build_augmented_roles_dictionaries`` / ``augment_roles`` MUTATE the
    class-level ``AgentBasics.ROLE_STR_TO_BITS`` / ``ROLE_BITS_TO_STR`` dicts, so
    we snapshot and restore them around the test to keep role-dict state isolated
    between tests. With these roles registered the relevant ints are:
        teacher=4, student=8,
        world_agent~teacher=5, world_master~teacher=7,
        world_agent~student=9, world_master~student=11.
    """
    saved_bits_to_str = dict(AgentBasics.ROLE_BITS_TO_STR)
    saved_str_to_bits = dict(AgentBasics.ROLE_STR_TO_BITS)
    saved_custom = list(AgentBasics.CUSTOM_ROLES)

    a = Agent(proc=None)
    a.CUSTOM_ROLES = ["teacher", "student"]
    a.augment_roles(*AgentBasics.build_augmented_roles_dictionaries(["teacher", "student"]))
    try:
        yield a
    finally:
        AgentBasics.ROLE_BITS_TO_STR.clear()
        AgentBasics.ROLE_BITS_TO_STR.update(saved_bits_to_str)
        AgentBasics.ROLE_STR_TO_BITS.clear()
        AgentBasics.ROLE_STR_TO_BITS.update(saved_str_to_bits)
        AgentBasics.CUSTOM_ROLES = saved_custom


# ── Construction / initial state ─────────────────────────────────────────────

def test_bare_agent_builds_offline_with_default_scope_state(agent):
    """``Agent(proc=None)`` constructs offline and seeds every scope variable."""
    assert agent._available is True
    assert agent._engaged_agents == set()
    assert agent._found_agents == set()
    assert agent._valid_cmp_agents == set()
    assert agent._eval_results == {}
    assert agent._preferred_streams == []
    assert agent._cur_preferred_stream == 0
    assert agent._repeat == 1
    # The default buffering sentinel is the STRING 'none' (relevant for evaluate).
    assert agent.buffer_generated_by_others == "none"


def test_role_augmentation_is_class_level_and_restored(roled_agent):
    """The custom roles map to the documented bit patterns once augmented."""
    assert roled_agent.ROLE_STR_TO_BITS["teacher"] == 4
    assert roled_agent.ROLE_STR_TO_BITS["student"] == 8
    assert roled_agent.ROLE_STR_TO_BITS["world_agent~teacher"] == 5
    assert roled_agent.ROLE_STR_TO_BITS["world_master~student"] == 11


def test_role_dict_isolation_restored_after_fixture():
    """Outside the ``roled_agent`` fixture the class-level role dict is back to
    the three default roles only (proves the snapshot/restore works)."""
    assert "teacher" not in AgentBasics.ROLE_STR_TO_BITS
    assert set(AgentBasics.ROLE_STR_TO_BITS) == {"public_agent", "world_master", "world_agent"}


# ── nop / disengage_all ──────────────────────────────────────────────────────

def test_nop_returns_true_and_does_not_mutate_state(agent):
    """``nop`` returns True with or without a ``message`` and leaves the engaged
    set and availability untouched."""
    agent._engaged_agents = {"x"}
    agent._available = False
    assert _run(agent.nop()) is True
    assert _run(agent.nop(message="hello")) is True
    # No state change.
    assert agent._engaged_agents == {"x"}
    assert agent._available is False


def test_disengage_all_resets_engaged_and_availability(agent):
    """``disengage_all`` clears ``_engaged_agents`` and flips ``_available`` back
    to True."""
    agent._engaged_agents = {"a", "b"}
    agent._available = False
    assert _run(agent.disengage_all()) is True
    assert agent._engaged_agents == set()
    assert agent._available is True


# ── all_sent_completed / all_engagements_completed ───────────────────────────

class _FakeSent:
    def __init__(self, action_name, completed):
        self.action_name = action_name
        self._completed = completed

    def is_completed(self):
        return self._completed


def test_all_sent_completed_true_on_empty_sent(agent):
    """An empty ``im.sent`` means nothing is pending -> True."""
    assert _run(agent.all_sent_completed()) is True


def test_all_sent_completed_false_when_any_pending(agent):
    """A single not-yet-completed entry in ``im.sent`` makes
    ``all_sent_completed`` return False."""
    agent.im.sent = {"u1": _FakeSent("teach", True), "u2": _FakeSent("eval", False)}
    assert _run(agent.all_sent_completed()) is False


def test_all_sent_completed_filters_by_action_name(agent):
    """With ``action_name`` set, only entries of that action are checked: 'teach'
    (all complete) is True while 'eval' (still pending) is False."""
    agent.im.sent = {"u1": _FakeSent("teach", True), "u2": _FakeSent("eval", False)}
    # Only 'teach' entries are complete; only 'eval' is pending.
    assert _run(agent.all_sent_completed(action_name="teach")) is True
    assert _run(agent.all_sent_completed(action_name="eval")) is False


def test_all_engagements_completed_tracks_found_agents(agent):
    """``all_engagements_completed`` is True while ``_found_agents`` is empty and
    False once it holds any agent."""
    assert _run(agent.all_engagements_completed()) is True
    agent._found_agents = {"s1"}
    assert _run(agent.all_engagements_completed()) is False


# ── engage ───────────────────────────────────────────────────────────────────

def test_engage_match_adds_requester_and_flips_availability(roled_agent):
    """acceptable_role='teacher'(=4) vs sender 'world_agent~teacher'(=5): the
    sender int is masked to (5>>2)<<2 == 4, so it matches. The requester is
    added to ``_engaged_agents`` and the agent becomes unavailable."""
    roled_agent.world_agents = {"peerW": object()}
    ok = _run(roled_agent.engage(acceptable_role="teacher", sender_role="world_agent~teacher",
                                 interaction=FakeInteraction("peerW")))
    assert ok is True
    assert roled_agent._engaged_agents == {"peerW"}
    assert roled_agent._available is False


def test_engage_world_master_requester_is_also_accepted(roled_agent):
    """The unknown-agent guard passes if the requester is a world MASTER too."""
    roled_agent.world_masters = {"peerM": object()}
    ok = _run(roled_agent.engage(acceptable_role="teacher", sender_role="world_master~teacher",
                                 interaction=FakeInteraction("peerM")))
    # world_master~teacher == 7; masked (7>>2)<<2 == 4 == teacher -> match.
    assert ok is True
    assert roled_agent._engaged_agents == {"peerM"}


def test_engage_rejects_when_not_available(roled_agent):
    """A second engagement is refused while the agent is already engaged."""
    roled_agent.world_agents = {"peerW": object()}
    assert _run(roled_agent.engage(acceptable_role="teacher", sender_role="world_agent~teacher",
                                   interaction=FakeInteraction("peerW"))) is True
    # Now unavailable: the next engagement returns False and leaves state intact.
    again = _run(roled_agent.engage(acceptable_role="teacher", sender_role="world_agent~teacher",
                                    interaction=FakeInteraction("peerW")))
    assert again is False
    assert roled_agent._available is False


def test_engage_unknown_requester_returns_false(roled_agent):
    """A requester in neither world_agents nor world_masters is rejected."""
    roled_agent.world_agents = {"peerW": object()}
    assert _run(roled_agent.engage(acceptable_role="teacher", sender_role="world_agent~teacher",
                                   interaction=FakeInteraction("stranger"))) is False
    assert roled_agent._engaged_agents == set()
    assert roled_agent._available is True


def test_engage_none_sender_role_returns_false(roled_agent):
    """A ``sender_role`` of None has no bits to mask and match, so engage returns
    False."""
    roled_agent.world_agents = {"peerW": object()}
    assert _run(roled_agent.engage(acceptable_role="teacher", sender_role=None,
                                   interaction=FakeInteraction("peerW"))) is False


def test_engage_none_acceptable_role_returns_false(roled_agent):
    """An ``acceptable_role`` of None gives nothing to match against, so engage
    returns False."""
    roled_agent.world_agents = {"peerW": object()}
    assert _run(roled_agent.engage(acceptable_role=None, sender_role="world_agent~teacher",
                                   interaction=FakeInteraction("peerW"))) is False


def test_engage_plain_base_role_never_matches(roled_agent):
    """Design note pinned as behaviour: a plain base role like 'world_agent' as
    acceptable_role can NEVER match, because the sender int is masked with
    (bits>>2)<<2 which drops exactly the base-role bits the acceptable side keeps
    unmasked (acceptable_role_int=1 vs sender masked to 4)."""
    roled_agent.world_agents = {"peerW": object()}
    assert _run(roled_agent.engage(acceptable_role="world_agent", sender_role="world_agent~teacher",
                                   interaction=FakeInteraction("peerW"))) is False


def test_engage_role_mismatch_returns_false(roled_agent):
    """teacher(=4) vs a student sender(masked to 8) does not match."""
    roled_agent.world_agents = {"peerW": object()}
    assert _run(roled_agent.engage(acceptable_role="teacher", sender_role="world_agent~student",
                                   interaction=FakeInteraction("peerW"))) is False


def test_engage_unregistered_role_keyerror_swallowed_to_false(roled_agent):
    """A role string absent from ROLE_STR_TO_BITS raises KeyError inside engage,
    which @action swallows into a returned False (never a raise)."""
    roled_agent.world_agents = {"peerW": object()}
    result = _run(roled_agent.engage(acceptable_role="nonexistent_role",
                                     sender_role="world_agent~teacher",
                                     interaction=FakeInteraction("peerW")))
    assert result is False


def test_engage_none_interaction_means_none_requester_rejected(roled_agent):
    """With ``interaction=None`` the requester is None, which is in no world dict,
    so the unknown-agent guard rejects it."""
    roled_agent.world_agents = {"peerW": object()}
    assert _run(roled_agent.engage(acceptable_role="teacher", sender_role="world_agent~teacher",
                                   interaction=None)) is False


# ── find_agents / get_agents_by_role ─────────────────────────────────────────

def test_find_agents_searches_both_world_role_ints(roled_agent):
    """For 'student' (=8, no '~') the search ORs in ROLE_WORLD_AGENT(1) and
    ROLE_WORLD_MASTER(3): role-ints 9 and 11. handshake_completed defaults False
    so every returned peer is kept."""
    conn = FakeConn({9: ["s1"], 11: ["s2"]})
    roled_agent.node_conn = conn
    found = _run(roled_agent.find_agents(role="student"))
    assert found is True
    assert roled_agent._found_agents == {"s1", "s2"}
    assert sorted(conn.calls) == [9, 11]
    # engage defaults False: the engaged set is untouched.
    assert roled_agent._engaged_agents == set()


def test_find_agents_returns_false_when_none_found(roled_agent):
    """When the node layer yields no peers for the role, find_agents returns
    False and leaves ``_found_agents`` empty."""
    roled_agent.node_conn = FakeConn({})
    assert _run(roled_agent.find_agents(role="student")) is False
    assert roled_agent._found_agents == set()


def test_find_agents_engage_true_copies_into_engaged(roled_agent):
    """``engage=True`` copies the found peers into ``_engaged_agents``, after
    which engagements are no longer reported complete."""
    roled_agent.node_conn = FakeConn({9: ["s1"], 11: ["s2"]})
    assert _run(roled_agent.find_agents(role="student", engage=True)) is True
    assert roled_agent._engaged_agents == {"s1", "s2"}
    # After find_agents populated _found_agents, engagements are NOT complete.
    assert _run(roled_agent.all_engagements_completed()) is False


def test_find_agents_handshake_completed_filters_by_all_agents(roled_agent):
    """With handshake_completed=True only peers present in ``all_agents`` survive."""
    roled_agent.node_conn = FakeConn({9: ["s1"], 11: ["s2"]})
    roled_agent.all_agents = {"s1": object()}
    assert _run(roled_agent.find_agents(role="student", handshake_completed=True)) is True
    assert roled_agent._found_agents == {"s1"}


def test_find_agents_clears_previous_found(roled_agent):
    """Each call clears ``_found_agents`` before re-populating it."""
    roled_agent.node_conn = FakeConn({9: ["s1"]})
    roled_agent._found_agents = {"stale"}
    _run(roled_agent.find_agents(role="student"))
    assert roled_agent._found_agents == {"s1"}


# ── next_pref_stream / first_pref_stream ─────────────────────────────────────

def test_next_pref_stream_wraps_around(agent):
    """``next_pref_stream`` advances the pointer one step per call and wraps from
    the last index back to 0."""
    agent._preferred_streams = ["x", "y", "z"]
    agent._cur_preferred_stream = 0
    assert _run(agent.next_pref_stream()) is True
    assert agent._cur_preferred_stream == 1
    assert _run(agent.next_pref_stream()) is True
    assert agent._cur_preferred_stream == 2
    # Wraps to 0 at the end.
    assert _run(agent.next_pref_stream()) is True
    assert agent._cur_preferred_stream == 0


def test_next_pref_stream_empty_returns_false(agent):
    """With an empty playlist there is nothing to advance, so
    ``next_pref_stream`` returns False."""
    assert _run(agent.next_pref_stream()) is False


def test_first_pref_stream_resets_pointer(agent):
    """``first_pref_stream`` resets ``_cur_preferred_stream`` back to 0."""
    agent._preferred_streams = ["x", "y", "z"]
    agent._cur_preferred_stream = 2
    assert _run(agent.first_pref_stream()) is True
    assert agent._cur_preferred_stream == 0


def test_first_pref_stream_empty_returns_false(agent):
    """With an empty playlist ``first_pref_stream`` has no pointer to reset and
    returns False."""
    assert _run(agent.first_pref_stream()) is False


# ── check_pref_stream ────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "cur,first,last,last_round,last_song",
    [
        # repeat=2, songs=3 -> playlist of 6; song-length = 6//2 = 3.
        (0, True, False, False, False),
        (1, False, False, False, False),
        (2, False, False, False, True),
        (3, False, False, True, False),
        (4, False, False, True, False),
        (5, False, True, True, True),
    ],
)
def test_check_pref_stream_table_repeat2_songs3(agent, cur, first, last, last_round, last_song):
    """Full truth table for a repeat=2, 3-song (length-6) playlist: at every
    pointer position it pins first/last/last_round/last_song (and each negation)
    of ``check_pref_stream``."""
    agent._preferred_streams = list("abcdef")
    agent._repeat = 2
    agent._cur_preferred_stream = cur
    assert _run(agent.check_pref_stream(what="first")) is first
    assert _run(agent.check_pref_stream(what="last")) is last
    assert _run(agent.check_pref_stream(what="not_first")) is (not first)
    assert _run(agent.check_pref_stream(what="not_last")) is (not last)
    assert _run(agent.check_pref_stream(what="last_round")) is last_round
    assert _run(agent.check_pref_stream(what="not_last_round")) is (not last_round)
    assert _run(agent.check_pref_stream(what="last_song")) is last_song
    assert _run(agent.check_pref_stream(what="not_last_song")) is (not last_song)


def test_check_pref_stream_last_round_always_true_when_repeat_one(agent):
    """With repeat=1, len//repeat == len, so cur + len >= len is always True."""
    agent._preferred_streams = list("abc")
    agent._repeat = 1
    for cur in range(3):
        agent._cur_preferred_stream = cur
        assert _run(agent.check_pref_stream(what="last_round")) is True


def test_check_pref_stream_empty_first_and_last(agent):
    """On an empty playlist 'first' is True (cur==0) and 'last' is False
    (cur != -1)."""
    agent._preferred_streams = []
    assert _run(agent.check_pref_stream(what="first")) is True
    assert _run(agent.check_pref_stream(what="last")) is False


def test_check_pref_stream_invalid_what_assert_swallowed_to_false(agent):
    """An out-of-set ``what`` fails the assert, which @action swallows to False."""
    agent._preferred_streams = ["a"]
    assert _run(agent.check_pref_stream(what="banana")) is False


def test_check_pref_stream_empty_playlist_last_song_zerodiv_swallowed(agent):
    """SUSPECTED BUG (agent.py:1235): 'last_song' computes
    ``len(_preferred_streams) // _repeat`` then ``(cur+1) % that``. On an empty
    playlist the divisor is 0, raising ZeroDivisionError, which @action swallows
    into a silent False rather than a meaningful value. We pin the actual (buggy)
    behaviour: it returns exactly False, not a raise."""
    agent._preferred_streams = []
    agent._repeat = 1
    assert _run(agent.check_pref_stream(what="last_song")) is False


def test_check_pref_stream_repeat_zero_zerodiv_swallowed(agent):
    """SUSPECTED BUG (agent.py:1235/1244 family): with _repeat==0 the 'last_round'
    and 'last_song' divisions blow up and are swallowed to False."""
    agent._preferred_streams = list("abc")
    agent._repeat = 0
    assert _run(agent.check_pref_stream(what="last_round")) is False
    assert _run(agent.check_pref_stream(what="last_song")) is False


# ── set_pref_streams ─────────────────────────────────────────────────────────

def test_set_pref_streams_builds_repeat_major_playlist(agent):
    """Order is repeat-major: each round re-lists all net_hashes. With a fake
    find_streams that resolves one net_hash per ``peer_id:name`` the playlist is
    [round0..., round1..., round2...]."""
    agent.find_streams = lambda peer_id, name_or_group=None: {f"{peer_id}::ps:{name_or_group}": {}}
    ok = _run(agent.set_pref_streams(net_hashes=["p1:teach", "p2:eval"], repeat=3))
    assert ok is True
    assert agent._preferred_streams == [
        "p1::ps:teach", "p2::ps:eval",
        "p1::ps:teach", "p2::ps:eval",
        "p1::ps:teach", "p2::ps:eval",
    ]
    assert agent._cur_preferred_stream == 0
    assert agent._repeat == 3


def test_set_pref_streams_double_colon_form_uses_last_component(agent):
    """A 'peer::ps:group' hash is split on ':' and the LAST component is the
    name_or_group passed to find_streams (peer_id is the first)."""
    seen = []

    def fake_find(peer_id, name_or_group=None):
        seen.append((peer_id, name_or_group))
        return {f"{peer_id}::ps:{name_or_group}": {}}

    agent.find_streams = fake_find
    _run(agent.set_pref_streams(net_hashes=["p1::ps:grp"], repeat=1))
    assert seen == [("p1", "grp")]
    assert agent._preferred_streams == ["p1::ps:grp"]


def test_set_pref_streams_no_streams_leaves_empty_playlist_but_returns_true(agent):
    """When find_streams resolves nothing the playlist stays EMPTY despite
    repeat, yet set_pref_streams still returns True (always-True contract)."""
    agent.find_streams = lambda peer_id, name_or_group=None: {}
    assert _run(agent.set_pref_streams(net_hashes=["p1:teach"], repeat=2)) is True
    assert agent._preferred_streams == []
    assert agent._repeat == 2


# ── compare_eval ─────────────────────────────────────────────────────────────

def _seed_eval(agent, results, profiles=None):
    """Seed ``_eval_results`` (peer_id -> score) and matching profiles."""
    agent._eval_results = dict(results)
    if profiles is None:
        profiles = {pid: FakeProfile(f"{pid}@x", pid) for pid in results}
    agent.all_agents = dict(profiles)


def test_compare_eval_threshold_less_than(agent):
    """'<' 0.2 keeps only the agent below the threshold."""
    _seed_eval(agent, {"A": 0.1, "B": 0.5})
    assert _run(agent.compare_eval(cmp="<", thres=0.2)) is True
    assert agent._valid_cmp_agents == {"A"}


def test_compare_eval_good_if_true_false_inverts(agent):
    """good_if_true=False inverts pass/fail: '<' 0.2 then keeps the NON-passing
    agent instead."""
    _seed_eval(agent, {"A": 0.1, "B": 0.5})
    assert _run(agent.compare_eval(cmp="<", thres=0.2, good_if_true=False)) is True
    assert agent._valid_cmp_agents == {"B"}


def test_compare_eval_min_keeps_single_best_low(agent):
    """min with thres=1.0 keeps the single lowest-scoring agent."""
    _seed_eval(agent, {"A": 0.1, "B": 0.5})
    assert _run(agent.compare_eval(cmp="min", thres=1.0)) is True
    assert agent._valid_cmp_agents == {"A"}


def test_compare_eval_max_thres_negative_disables_gate(agent):
    """max with thres=-1 disables the threshold gate and keeps the single
    highest-scoring agent."""
    _seed_eval(agent, {"A": 0.1, "B": 0.5})
    assert _run(agent.compare_eval(cmp="max", thres=-1)) is True
    assert agent._valid_cmp_agents == {"B"}


def test_compare_eval_empty_results_returns_false(agent):
    """With no ``_eval_results`` to compare, compare_eval returns False and leaves
    ``_valid_cmp_agents`` empty."""
    _seed_eval(agent, {})
    assert _run(agent.compare_eval(cmp="<", thres=0.2)) is False
    assert agent._valid_cmp_agents == set()


def test_compare_eval_negative_score_returns_false(agent):
    """A score < 0 short-circuits the whole comparison to False."""
    _seed_eval(agent, {"A": -0.5})
    assert _run(agent.compare_eval(cmp="<", thres=0.2)) is False
    assert agent._valid_cmp_agents == set()


def test_compare_eval_invalid_cmp_assert_swallowed_to_false(agent):
    """An unsupported ``cmp`` such as '==' fails the assert inside compare_eval,
    which @action swallows into a returned False."""
    _seed_eval(agent, {"A": 0.1})
    assert _run(agent.compare_eval(cmp="==", thres=0.2)) is False


def test_compare_eval_negative_thres_with_relational_cmp_swallowed(agent):
    """thres < 0 is only allowed for min/max; with '<' it fails the assert and
    @action swallows to False."""
    _seed_eval(agent, {"A": 0.1})
    assert _run(agent.compare_eval(cmp="<", thres=-1.0)) is False


def test_compare_eval_missing_profile_keyerror_swallowed_to_false(agent):
    """A score keyed by a peer absent from ``all_agents`` raises KeyError on the
    profile lookup, swallowed by @action to False."""
    agent._eval_results = {"Z": 0.1}
    agent.all_agents = {}  # no profile for 'Z'
    assert _run(agent.compare_eval(cmp="<", thres=0.2)) is False


def test_compare_eval_min_keeps_best_even_when_iterated_first(agent):
    """In min/max the best agent iterated FIRST is kept: the return value and
    ``_valid_cmp_agents`` stay correct, and the user-facing ``msgs`` are no longer
    overwritten to 'No best agent found' once a best exists. We pin the correct
    return/state (the log text itself is not asserted)."""
    # dict order: best (lowest) 'A' first, then worse 'B'.
    _seed_eval(agent, {"A": 0.1, "B": 0.9})
    assert _run(agent.compare_eval(cmp="min", thres=-1)) is True
    assert agent._valid_cmp_agents == {"A"}


# ── evaluate ─────────────────────────────────────────────────────────────────

def _patch_compare(agent, score=0.42, ret=True, record=None):
    """Monkeypatch the name-mangled ``__compare_streams`` so evaluate needs no
    real BufferedStream pairs."""
    def fake(net_hash_a, net_hash_b, how, steps, re_offset):
        if record is not None:
            record.append((net_hash_a, net_hash_b, how, steps, re_offset))
        return (score, ret)

    setattr(agent, "_Agent__compare_streams", fake)


def test_evaluate_happy_path_via_playlist(agent):
    """With buffering proceeding, '<playlist>' resolves the current preferred
    stream as the reference, compares against the buffered stream and stores the
    score keyed by the buffered stream's owner."""
    calls = []
    _patch_compare(agent, score=0.42, ret=True, record=calls)
    agent._preferred_streams = ["ref::ps:x"]
    agent._cur_preferred_stream = 0
    agent._engaged_agents = {"pX"}
    agent.last_buffered_peer_id_to_info = {"pX": {"net_hash": "pX::ps:gen"}}
    assert _run(agent.evaluate(stream_hash="<playlist>", how="max", steps=5)) is True
    assert agent._eval_results == {"pX": 0.42}
    # Reference is the playlist item; buffered is the received net_hash.
    assert calls == [("ref::ps:x", "pX::ps:gen", "max", 5, False)]


def test_evaluate_skips_agents_without_buffered_stream(agent):
    """An engaged agent with no entry in ``last_buffered_peer_id_to_info`` is
    skipped (continue), not failed."""
    _patch_compare(agent)
    agent._preferred_streams = ["ref::ps:x"]
    agent._engaged_agents = {"pX", "pY"}
    agent.last_buffered_peer_id_to_info = {"pX": {"net_hash": "pX::ps:gen"}}
    assert _run(agent.evaluate(stream_hash="<playlist>", how="max")) is True
    assert agent._eval_results == {"pX": 0.42}


def test_evaluate_returns_false_when_compare_returns_ret_false(agent):
    """If ``__compare_streams`` reports ret=False, evaluate returns False at
    once."""
    _patch_compare(agent, score=0.0, ret=False)
    agent._preferred_streams = ["ref::ps:x"]
    agent._engaged_agents = {"pX"}
    agent.last_buffered_peer_id_to_info = {"pX": {"net_hash": "pX::ps:gen"}}
    assert _run(agent.evaluate(stream_hash="<playlist>", how="max")) is False


def test_evaluate_empty_playlist_indexerror_swallowed_to_false(agent):
    """'<playlist>' indexes ``_preferred_streams[_cur_preferred_stream]``; on an
    empty playlist that IndexError is swallowed by @action to False."""
    agent._preferred_streams = []
    agent._engaged_agents = {"pX"}
    assert _run(agent.evaluate(stream_hash="<playlist>", how="max")) is False


def test_evaluate_str_agents_to_evaluate_is_wrapped_to_list(agent):
    """A string ``agents_to_evaluate`` is wrapped into a 1-list and used in place
    of the engaged set."""
    _patch_compare(agent)
    agent._preferred_streams = ["ref::ps:x"]
    agent.last_buffered_peer_id_to_info = {"pZ": {"net_hash": "pZ::ps:gen"}}
    assert _run(agent.evaluate(stream_hash="<playlist>", how="max", agents_to_evaluate="pZ")) is True
    assert agent._eval_results == {"pZ": 0.42}


# ── Suspected bugs pinned as xfail (desired behaviour) ───────────────────────

@pytest.mark.xfail(
    strict=True,
    reason="BUG agent.py:1290 - dead guard. 'if not self.buffer_generated_by_others' is "
           "checked against the STRING sentinel 'none'; `not 'none'` is False, so the guard "
           "never fires and evaluate proceeds even when buffering of others' data is OFF. "
           "Desired: with the default 'none' (no buffering) evaluate refuses (returns False).",
)
def test_evaluate_dead_guard_should_block_when_not_buffering():
    """The default ``buffer_generated_by_others == 'none'`` means no buffering of
    others' data, so evaluate SHOULD refuse. It currently proceeds and returns
    True, so this strict-xfail flags the bug (and will XPASS-fail when fixed)."""
    agent = Agent(proc=None)
    assert agent.buffer_generated_by_others == "none"  # no buffering of others
    _patch_compare(agent, score=0.42, ret=True)
    agent._preferred_streams = ["ref::ps:x"]
    agent._engaged_agents = {"pX"}
    agent.last_buffered_peer_id_to_info = {"pX": {"net_hash": "pX::ps:gen"}}
    result = _run(agent.evaluate(stream_hash="<playlist>", how="max"))
    # DESIRED behaviour: refuse to evaluate when not buffering others' data.
    assert result is False


@pytest.mark.xfail(
    strict=True,
    reason="BUG agent.py:1323 - evaluate keys _eval_results by the buffered stream's OWNER "
           "(peer_id_from_net_hash(received)) instead of the agents_to_evaluate key used to "
           "fetch it. When the buffered net_hash is owned by a different peer than the "
           "engaged key, compare_eval's later all_agents[peer_id] lookup keys on the wrong id.",
)
def test_evaluate_eval_results_should_be_keyed_by_engaged_key():
    """Engaged key 'engagedKey' whose buffered net_hash is owned by 'OWNER':
    DESIRED is to key ``_eval_results`` by 'engagedKey'. The code keys it by
    'OWNER', so this strict-xfail pins the mis-keying."""
    agent = Agent(proc=None)
    _patch_compare(agent, score=0.7, ret=True)
    agent._preferred_streams = ["ref::ps:x"]
    agent._engaged_agents = {"engagedKey"}
    agent.last_buffered_peer_id_to_info = {"engagedKey": {"net_hash": "OWNER::ps:gen"}}
    _run(agent.evaluate(stream_hash="<playlist>", how="max"))
    # DESIRED: keyed by the engaged-agent key, not the buffered owner.
    assert set(agent._eval_results) == {"engagedKey"}
