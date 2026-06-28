"""Characterization tests for the AgentResolution surface of ``Agent`` /
``AgentBasics``.

These pin the *current* behaviour of the in-memory reference resolvers that map
user-facing names/wildcards onto concrete peer ids and stream hashes:

  * ``resolve_agent_ref``                 (agent_basics.py:1274)
  * ``resolve_stream_ref`` (base)         (agent_basics.py:1304)
  * ``resolve_stream_ref`` (Agent <playlist> override)  (agent.py:1428)
  * ``user_stream_hash_to_net_hash``      (agent_basics.py:1369)
  * ``get_stream`` / ``get_streams``      (agent_basics.py:1144 / 1166)
  * ``find_streams``                      (agent_basics.py:1179)
  * ``__involved_agents``                 (agent.py:1691, name-mangled)

None of these touch the network, the clock or the Go layer: they only read
in-memory dicts/sets and call pure static string parsers
(``Stream.is_net_hash`` / ``is_user_hash`` / ``name_from_user_hash`` /
``name_or_group_from_net_hash`` and ``DataProps.build_net_hash`` /
``build_user_hash``). We therefore drive them on a light ``Agent`` shell built
with ``Agent.__new__`` and manual attribute injection, bypassing the heavy
``AgentBasics.__init__``.

Hash conventions used throughout (see ``DataProps``):
  * net hash:  ``peer_id::ps:name`` (pubsub) or ``peer_id::dm:name`` (direct)
  * user hash: ``peer_id:name``

The conftest's autouse ``_silence_unaiverse_logger`` fixture swallows the
``log.error`` / ``log.misc`` calls these methods make, and inserts ``../src``
on ``sys.path`` so the import below resolves offline.

Four known defects are pinned with ``xfail(strict=True)`` so the suite stays
green today while flagging (via XPASS) the moment any of them is fixed; each is
also reported under ``suspected_bugs``.
"""

import pytest

from unaiverse.agent import Agent


# ── Lightweight fakes ───────────────────────────────────────────────────────

class FakeProfile:
    """Stand-in for ``NodeProfile``: exposes only the two profile accessors the
    resolvers read."""

    def __init__(self, static=None, dynamic=None):
        self._static = static if static is not None else {}
        self._dynamic = dynamic if dynamic is not None else {}

    def get_static_profile(self):
        return self._static

    def get_dynamic_profile(self):
        return self._dynamic


class _NonDictProfile:
    """A profile whose ``get_static_profile()`` returns a non-dict, to exercise
    the ``isinstance(static_profile, dict)`` skip in ``resolve_agent_ref``."""

    def __init__(self, static):
        self._static = static

    def get_static_profile(self):
        return self._static


class FakeBehav:
    """Drives ``behaving_in_world()`` -> ``get_peer_id()``. Disabled means the
    agent is *not* behaving in a world, so ``get_peer_id`` returns the public
    peer id from the dynamic profile."""

    def __init__(self, enabled: bool):
        self._enabled = enabled

    def is_enabled(self) -> bool:
        return self._enabled


class FakeStream:
    """Only the surface the resolvers touch: ``is_public()`` (public filter of
    ``resolve_stream_ref``) and ``.props.data_type`` (``get_stream`` filter)."""

    def __init__(self, public: bool = True, data_type: str | None = None):
        self._public = public
        self.props = type("FakeProps", (), {"data_type": data_type})()

    def is_public(self) -> bool:
        return self._public


def make_agent(email: str = "me@x.org", peer_id: str = "PIDme") -> Agent:
    """Build a bare ``Agent`` shell with just the attributes the resolution
    methods read. ``behav`` is disabled, so ``get_peer_id()`` == ``peer_id``."""
    a = Agent.__new__(Agent)
    a.node_profile = FakeProfile(
        static={"email": email, "node_name": "meself"},
        dynamic={"peer_id": peer_id, "private_peer_id": peer_id + "_priv"},
    )
    a.behav = FakeBehav(enabled=False)
    a.all_agents = {}
    a.known_streams = {}
    a.owned_streams = {}
    a.known_streams_by_user_hash = {}
    a.owned_streams_by_user_hash = {}
    a._engaged_agents = set()
    a._found_agents = set()
    a._valid_cmp_agents = set()
    a._preferred_streams = []
    a._cur_preferred_stream = 0
    return a


@pytest.fixture
def agent() -> Agent:
    return make_agent()


# ── resolve_agent_ref ───────────────────────────────────────────────────────

def test_resolve_agent_ref_known_peer_id_passthrough(agent):
    """A ref that is already a key in ``all_agents`` is returned verbatim, before
    any name resolution."""
    agent.all_agents = {"PIDbob": FakeProfile(static={"email": "bob@x.org", "node_name": "bob"})}
    assert agent.resolve_agent_ref("PIDbob") == "PIDbob"


def test_resolve_agent_ref_full_email_at_name(agent):
    """The full ``email@node_name`` form resolves to the matching peer id even
    when it belongs to another user."""
    agent.all_agents = {"PIDbob": FakeProfile(static={"email": "bob@x.org", "node_name": "bob"})}
    # 'email' + '@' + 'node_name' == 'bob@x.org' + '@' + 'bob' == the ref.
    assert agent.resolve_agent_ref("bob@x.org@bob") == "PIDbob"


def test_resolve_agent_ref_bare_name_resolves_only_own_agents(agent):
    """A bare node_name is prefixed with the CALLER's own email, so it resolves
    only agents that share the caller's email."""
    # Caller email is 'me@x.org'; 'myagent' belongs to the same email -> resolves.
    agent.all_agents = {
        "PIDbob": FakeProfile(static={"email": "bob@x.org", "node_name": "bob"}),
        "PIDmine": FakeProfile(static={"email": "me@x.org", "node_name": "myagent"}),
    }
    assert agent.resolve_agent_ref("myagent") == "PIDmine"


def test_resolve_agent_ref_bare_name_of_other_user_is_silent_none(agent):
    """FOOTGUN (agent_basics.py:1292): a bare node_name of *another* user's agent
    silently returns None, because it is prefixed with the caller's own email
    ('me@x.org@bob' never matches bob's 'bob@x.org@bob'). The full email@name
    form is required to reach another user's agent."""
    agent.all_agents = {"PIDbob": FakeProfile(static={"email": "bob@x.org", "node_name": "bob"})}
    assert agent.resolve_agent_ref("bob") is None


def test_resolve_agent_ref_total_miss_returns_none(agent):
    """An ``email@name`` form that matches nobody returns None."""
    agent.all_agents = {"PIDbob": FakeProfile(static={"email": "bob@x.org", "node_name": "bob"})}
    assert agent.resolve_agent_ref("ghost@y.org@nobody") is None


def test_resolve_agent_ref_skips_non_dict_static_profile(agent):
    """Per-profile guard: a stored profile whose ``get_static_profile()`` returns
    a non-dict is skipped by the ``isinstance(static_profile, dict)`` check, so
    resolution falls through to None instead of crashing.

    NOTE: a profile that lacks ``get_static_profile`` entirely is coerced to the
    empty dict ``{}``, which is still a dict and therefore reaches the buggy
    ``'' + '@' + None`` concatenation (see
    test_resolve_agent_ref_profile_missing_node_name_crashes); only a genuinely
    non-dict static profile is skipped cleanly."""
    agent.all_agents = {"PIDnone": _NonDictProfile(None),
                        "PIDlist": _NonDictProfile(["not", "a", "dict"])}
    assert agent.resolve_agent_ref("ghost@y.org@nobody") is None


@pytest.mark.xfail(reason="BUG agent_basics.py:1299 - missing 'node_name' key -> "
                          "str + None TypeError (no default, asymmetric with 'email')",
                   raises=TypeError, strict=True)
def test_resolve_agent_ref_profile_missing_node_name_crashes(agent):
    """A stored profile missing the 'node_name' key should be treated as a
    non-match (ideally -> None). Today the unguarded
    ``email + '@' + None`` concatenation raises TypeError instead."""
    agent.all_agents = {"PIDbad": FakeProfile(static={"email": "x@y.org"})}
    # ref has '@', so no email prefixing: the loop reaches the bad profile.
    assert agent.resolve_agent_ref("anything@here") is None


# ── resolve_stream_ref (base, AgentBasics) ──────────────────────────────────

def test_resolve_stream_ref_known_user_hash_passthrough(agent):
    """A ref already present as a key in ``known_streams_by_user_hash`` is
    returned verbatim."""
    agent.known_streams_by_user_hash = {"PIDx:foo": FakeStream()}
    assert agent.resolve_stream_ref("PIDx:foo", public=False) == "PIDx:foo"


def test_resolve_stream_ref_known_net_hash_passthrough(agent):
    """A ref already present as a key in ``known_streams`` (net hash) is returned
    verbatim."""
    agent.known_streams = {"PIDx::ps:grp": {"s": FakeStream()}}
    assert agent.resolve_stream_ref("PIDx::ps:grp", public=False) == "PIDx::ps:grp"


def test_resolve_stream_ref_absent_net_formatted_is_none(agent):
    """A net-formatted ref (contains '::') that is not a known key short-circuits
    to None without any name search."""
    assert agent.resolve_stream_ref("X::ps:bar", public=False) is None


def test_resolve_stream_ref_plain_owned_name(agent):
    """A plain name resolves against owned streams first, returning the owned
    user hash."""
    agent.owned_streams_by_user_hash = {"PIDme:foo": FakeStream(public=True)}
    assert agent.resolve_stream_ref("foo", public=False) == "PIDme:foo"


def test_resolve_stream_ref_public_filter_skips_private_owned(agent):
    """With ``public=True`` a non-public owned stream is skipped; the same ref
    with ``public=False`` resolves it."""
    agent.owned_streams_by_user_hash = {"PIDme:foo": FakeStream(public=False)}
    assert agent.resolve_stream_ref("foo", public=True) is None
    assert agent.resolve_stream_ref("foo", public=False) == "PIDme:foo"


def test_resolve_stream_ref_owned_net_hash_by_group_name(agent):
    """A plain name also matches an owned stream group via its net hash group
    name."""
    agent.owned_streams = {"PIDme::ps:g2": {"x": FakeStream(public=True)}}
    assert agent.resolve_stream_ref("g2", public=False) == "PIDme::ps:g2"


def test_resolve_stream_ref_unknown_name_is_none(agent):
    """A plain name matching nothing returns None."""
    assert agent.resolve_stream_ref("nope", public=False) is None


def test_resolve_stream_ref_user_hash_other_peer_searches_other_streams(agent):
    """A user-hash ref whose peer_id is NOT us searches the *other* streams: the
    group name is extracted and matched against ``known_streams`` net hashes."""
    agent.known_streams = {"PIDother::ps:bar": {"s": FakeStream()}}
    assert agent.resolve_stream_ref("PIDother:bar", public=False) == "PIDother::ps:bar"


def test_resolve_stream_ref_user_hash_own_peer_searches_owned(agent):
    """A user-hash ref whose peer_id IS us (matches ``get_peer_id()``) searches
    owned streams only."""
    agent.owned_streams_by_user_hash = {"PIDme:foo": FakeStream()}
    assert agent.resolve_stream_ref("PIDme:foo", public=False) == "PIDme:foo"


@pytest.mark.xfail(reason="BUG agent_basics.py:1349/1362 - next(iter(stream_dict.values())) "
                          "raises StopIteration on an EMPTY inner dict when public=True",
                   raises=StopIteration, strict=True)
def test_resolve_stream_ref_empty_group_public_crashes(agent):
    """An owned net-hash group that maps to an EMPTY inner dict should be skipped
    when ``public=True``. Today ``next(iter({}.values()))`` raises StopIteration."""
    agent.owned_streams = {"PIDme::ps:grp": {}}
    assert agent.resolve_stream_ref("grp", public=True) is None


# ── resolve_stream_ref (Agent override, <playlist>) ─────────────────────────

def test_resolve_stream_ref_playlist_returns_current_preferred(agent):
    """The Agent override maps the ``<playlist>`` wildcard to the current entry
    of ``_preferred_streams``."""
    agent._preferred_streams = ["PIDme::ps:pl0", "PIDme::ps:pl1"]
    agent._cur_preferred_stream = 1
    assert agent.resolve_stream_ref("<playlist>", public=False) == "PIDme::ps:pl1"


def test_resolve_stream_ref_non_playlist_delegates_to_super(agent):
    """Any non-``<playlist>`` ref delegates to the base resolver."""
    agent.owned_streams_by_user_hash = {"PIDme:foo": FakeStream()}
    assert agent.resolve_stream_ref("foo", public=False) == "PIDme:foo"


@pytest.mark.xfail(reason="BUG agent.py:1441 - '<playlist>' indexes _preferred_streams "
                          "with no empty guard -> IndexError on an empty playlist",
                   raises=IndexError, strict=True)
def test_resolve_stream_ref_empty_playlist_crashes(agent):
    """An empty playlist should resolve to None (or be guarded). Today indexing
    ``_preferred_streams[0]`` on an empty list raises IndexError."""
    agent._preferred_streams = []
    agent._cur_preferred_stream = 0
    assert agent.resolve_stream_ref("<playlist>", public=False) is None


# ── user_stream_hash_to_net_hash ────────────────────────────────────────────

def test_user_stream_hash_to_net_hash_pubsub_group(agent):
    """A user hash ``peer_id:group`` resolves to the matching pubsub net hash."""
    agent.known_streams = {"PIDme::ps:grp": {"inner": FakeStream()}}
    assert agent.user_stream_hash_to_net_hash("PIDme:grp") == "PIDme::ps:grp"


def test_user_stream_hash_to_net_hash_inner_stream_name(agent):
    """A user hash naming an *inner* stream (not the group) still resolves to the
    owning net hash via the inner-name scan."""
    agent.known_streams = {"PIDme::ps:grp": {"inner": FakeStream()}}
    assert agent.user_stream_hash_to_net_hash("PIDme:inner") == "PIDme::ps:grp"


def test_user_stream_hash_to_net_hash_already_net_unchanged(agent):
    """An input that already contains '::' is returned unchanged."""
    assert agent.user_stream_hash_to_net_hash("PIDme::ps:grp") == "PIDme::ps:grp"


def test_user_stream_hash_to_net_hash_none_returns_none(agent):
    """A None input maps to None."""
    assert agent.user_stream_hash_to_net_hash(None) is None


def test_user_stream_hash_to_net_hash_miss_returns_none(agent):
    """A user hash matching no known stream returns None."""
    agent.known_streams = {"PIDme::ps:grp": {"inner": FakeStream()}}
    assert agent.user_stream_hash_to_net_hash("PIDme:nope") is None


# ── get_stream ──────────────────────────────────────────────────────────────

def test_get_stream_direct_user_hash_hit(agent):
    """A stream present in ``known_streams_by_user_hash`` is returned directly."""
    s = FakeStream()
    agent.known_streams_by_user_hash = {"PIDme:foo": s}
    assert agent.get_stream("foo", peer_id="PIDme") is s


def test_get_stream_default_peer_id_uses_get_peer_id(agent):
    """When ``peer_id`` is omitted it defaults to ``get_peer_id()`` (== 'PIDme'
    for this disabled-behav shell)."""
    s = FakeStream()
    agent.known_streams_by_user_hash = {"PIDme:foo": s}
    assert agent.get_stream("foo") is s


def test_get_stream_group_of_two_is_ambiguous_none(agent):
    """A group of two streams with no ``data_type`` filter is ambiguous -> None."""
    sa, sb = FakeStream(data_type="img"), FakeStream(data_type="text")
    agent.known_streams = {"PIDme::ps:grp": {"a": sa, "b": sb}}
    assert agent.get_stream("grp", peer_id="PIDme") is None


def test_get_stream_data_type_filter_picks_match(agent):
    """With a ``data_type`` filter the first stream whose ``props.data_type``
    matches is returned."""
    sa, sb = FakeStream(data_type="img"), FakeStream(data_type="text")
    agent.known_streams = {"PIDme::ps:grp": {"a": sa, "b": sb}}
    assert agent.get_stream("grp", peer_id="PIDme", data_type="text") is sb


def test_get_stream_data_type_no_match_returns_none(agent):
    """A ``data_type`` filter that matches none of the group's streams -> None."""
    sa, sb = FakeStream(data_type="img"), FakeStream(data_type="text")
    agent.known_streams = {"PIDme::ps:grp": {"a": sa, "b": sb}}
    assert agent.get_stream("grp", peer_id="PIDme", data_type="audio") is None


def test_get_stream_single_member_group_returned(agent):
    """A group with exactly one member and no filter returns that member."""
    solo = FakeStream()
    agent.known_streams = {"PIDme::ps:solo": {"x": solo}}
    assert agent.get_stream("solo", peer_id="PIDme") is solo


def test_get_stream_miss_returns_none(agent):
    """No matching user hash and no group -> None."""
    assert agent.get_stream("absent", peer_id="PIDme") is None


# ── get_streams ─────────────────────────────────────────────────────────────

def test_get_streams_pubsub_group_returns_list(agent):
    """The pubsub net hash is tried first; a group of two yields a list of two."""
    sa, sb = FakeStream(), FakeStream()
    agent.known_streams = {"PIDme::ps:grp": {"a": sa, "b": sb}}
    result = agent.get_streams("grp", peer_id="PIDme")
    assert isinstance(result, list) and len(result) == 2
    assert set(result) == {sa, sb}


def test_get_streams_dm_fallback(agent):
    """When the pubsub net hash is absent, the direct-message net hash is tried."""
    sa, sb = FakeStream(), FakeStream()
    agent.known_streams = {"PIDme::dm:grp": {"a": sa, "b": sb}}
    result = agent.get_streams("grp", peer_id="PIDme")
    assert isinstance(result, list) and len(result) == 2


def test_get_streams_unknown_returns_none(agent):
    """Neither pubsub nor dm net hash present -> None."""
    assert agent.get_streams("zzz", peer_id="PIDme") is None


# ── find_streams ────────────────────────────────────────────────────────────

def test_find_streams_matching_peer_returns_whole_group(agent):
    """With no ``name_or_group`` filter, every net hash owned by the peer yields
    its whole inner dict under the net-hash key."""
    sa, sb = FakeStream(), FakeStream()
    agent.known_streams = {"PIDbob::ps:grp": {"a": sa, "b": sb}}
    result = agent.find_streams("PIDbob")
    assert set(result.keys()) == {"PIDbob::ps:grp"}
    assert result["PIDbob::ps:grp"] == {"a": sa, "b": sb}


def test_find_streams_group_name_filter_matches_group(agent):
    """A ``name_or_group`` equal to the hash's group name adds the whole inner
    dict."""
    sa, sb = FakeStream(), FakeStream()
    agent.known_streams = {"PIDbob::ps:grp": {"a": sa, "b": sb}}
    result = agent.find_streams("PIDbob", name_or_group="grp")
    assert set(result.keys()) == {"PIDbob::ps:grp"}
    assert result["PIDbob::ps:grp"] == {"a": sa, "b": sb}


def test_find_streams_individual_name_filter(agent):
    """A ``name_or_group`` that does not match the group name adds only the inner
    streams whose name matches."""
    sa, sb = FakeStream(), FakeStream()
    agent.known_streams = {"PIDbob::ps:grp": {"a": sa, "b": sb}}
    result = agent.find_streams("PIDbob", name_or_group="a")
    assert result == {"PIDbob::ps:grp": {"a": sa}}


def test_find_streams_wrong_peer_returns_empty_dict(agent):
    """No match returns an EMPTY dict (not None)."""
    agent.known_streams = {"PIDbob::ps:grp": {"a": FakeStream()}}
    result = agent.find_streams("PIDnobody")
    assert result == {}
    assert isinstance(result, dict)


def test_find_streams_discard_owned_skips_owned_net_hashes(agent):
    """With ``discard_owned=True`` a net hash also present in ``owned_streams`` is
    skipped, yielding {}."""
    sa = FakeStream()
    agent.known_streams = {"PIDbob::ps:grp": {"a": sa}}
    agent.owned_streams = {"PIDbob::ps:grp": {"a": sa}}
    assert agent.find_streams("PIDbob", discard_owned=True) == {}
    # Without the flag the same call returns the group.
    assert set(agent.find_streams("PIDbob", discard_owned=False).keys()) == {"PIDbob::ps:grp"}


# ── __involved_agents (name-mangled: _Agent__involved_agents) ────────────────

def test_involved_agents_concrete_peer_id(agent):
    """A concrete peer_id resolves to a single-element list."""
    assert agent._Agent__involved_agents("PIDx") == ["PIDx"]


def test_involved_agents_set_is_listified(agent):
    """A set argument is converted to a list (order-independent)."""
    result = agent._Agent__involved_agents({"p3", "p4"})
    assert isinstance(result, list)
    assert sorted(result) == ["p3", "p4"]


def test_involved_agents_valid_cmp_wildcard(agent):
    """The '<valid_cmp>' wildcard expands to ``_valid_cmp_agents``."""
    agent._valid_cmp_agents = {"PIDc1", "PIDc2"}
    result = agent._Agent__involved_agents("<valid_cmp>")
    assert sorted(result) == ["PIDc1", "PIDc2"]


def test_involved_agents_none_prefers_engaged(agent):
    """``None`` falls back to engaged agents when non-empty."""
    agent._engaged_agents = {"PIDe1"}
    agent._found_agents = {"PIDf1"}
    assert agent._Agent__involved_agents(None) == ["PIDe1"]


def test_involved_agents_none_falls_back_to_found(agent):
    """``None`` falls back to found agents when there are no engaged agents."""
    agent._engaged_agents = set()
    agent._found_agents = {"PIDf1"}
    assert agent._Agent__involved_agents(None) == ["PIDf1"]


def test_involved_agents_none_empty_returns_empty_list(agent):
    """``None`` with neither engaged nor found agents returns [] (and logs an
    error, swallowed by the silencing fixture)."""
    assert agent._Agent__involved_agents(None) == []


@pytest.mark.xfail(reason="BUG agent.py:1703 - the list branch returns the caller's "
                          "argument by reference (return agent), not list(agent), "
                          "unlike every other branch",
                   strict=True)
def test_involved_agents_list_is_copied(agent):
    """A list argument should be returned as a fresh copy (like the set / peer_id
    / None branches). Today it is returned by reference, aliasing the caller's
    input."""
    lst = ["p1", "p2"]
    result = agent._Agent__involved_agents(lst)
    assert result == ["p1", "p2"]
    assert result is not lst  # fails today: same object
