"""Characterization tests for ``unaiverse.interaction.InteractionManager``.

These pin the *current* behavior of the registration/bookkeeping surface of the
manager: the capacity accountancy (``room_for_registration`` /
``count_interactions``), ``unregister`` and the ``is_*`` predicates, the stream
expansion/matching helpers (``expand_and_normalize_streams`` / ``match_streams``)
and the three registration entry points (``register_sent`` / ``register_received``
/ ``register_lazy``) on no-stream interactions.

The manager is exercised as a light-mock unit: a hand-rolled ``FakeAgent`` (not the
real ``AgentBasics``) stands in for the back-reference, exposing only the attributes
and methods the tested code paths actually touch. No real network / Go / libp2p.

Tests assert *actual* behavior, warts included, with a comment whenever that behavior
looks questionable. The broken-callback section restores the raising logger locally,
because the production rejection contract (log.critical raising GenException) is
neutered by the suite-wide no-op logger fixture.
"""
import pytest

from unaiverse.custom import Custom, GenException
from unaiverse.interaction import (
    Interaction,
    InteractionManager,
    InteractionStatus,
    InteractionType,
)

pytestmark = pytest.mark.unit


# Fakes

class FakeProps:
    """Minimal stand-in for ``DataProps`` exposing only the accessors the manager
    reads on a stream's ``.props`` during expansion/matching."""

    def __init__(self, name="images", group="none", public=True, pubsub=False):
        self._name = name
        self._group = group
        self._public = public
        self._pubsub = pubsub

    def get_name(self):
        return self._name

    def get_group(self):
        return self._group

    def name_or_group(self):
        return self._group if self._group != "none" else self._name

    def is_public(self):
        return self._public

    def is_pubsub(self):
        return self._pubsub


class FakeStream:
    """Lightweight stream stand-in. Only ``props`` (for net-hash expansion) and the
    interaction bookkeeping methods are exercised by the tested paths; the stdext-only
    ``match_streams`` path stores the object verbatim without calling any method on it."""

    def __init__(self, name="images", **props_kw):
        self.props = FakeProps(name=name, **props_kw)
        self.added = []  # interactions registered via add_interaction

    def add_interaction(self, interaction):
        self.added.append(interaction)

    def has_interaction(self, uuid):
        return any(i.uuid == uuid for i in self.added)

    def remove_interaction(self, interaction):
        self.added = [i for i in self.added if i is not interaction]

    def remove_data(self, uuid):
        pass

    def is_pubsub(self):
        return self.props.is_pubsub()


class FakeAgent:
    """Hand-rolled back-reference for InteractionManager. Defaults are empty/identity so
    the no-stream happy paths succeed; individual tests poke the knobs they need."""

    def __init__(self):
        self.known_streams_by_user_hash = {}
        self.owned_streams_by_user_hash = {}
        self.proc_streams_by_user_hash = {}
        self.known_streams = {}            # net_hash -> {name: stream}
        self.all_agents = set()
        self.public_agents = set()
        # Processor slots: left None on purpose. They are only asserted-not-None inside
        # match_streams when a 'stdin'/'stdunk' stream forces the processor path, which
        # none of the light tests trigger.
        self.proc_inputs = None
        self.proc_outputs = None
        self.proc_optional_inputs = None
        self.compat_in_streams = None
        self.compat_out_streams = None
        # Knobs.
        self.unresolvable_targets = set()  # targets that resolve_agent_ref maps to None
        self.find_streams_returns = {}     # what find_streams hands back
        self.stream_refs = {}              # ref -> resolved hash for resolve_stream_ref

    def resolve_agent_ref(self, ref):
        if ref in self.unresolvable_targets:
            return None
        return ref

    def resolve_stream_ref(self, ref, public):
        return self.stream_refs.get(ref, ref)

    def find_streams(self, peer_id, name_or_group=None, discard_owned=False):
        return dict(self.find_streams_returns)

    def set_default_stream_binding(self, public=None):
        pass


# Helpers

def _manager(agent=None, max_interactions=-1):
    return InteractionManager(agent=agent, max_interactions=max_interactions)


def _inter(action_name="act", requester="peerA", target=None, id="random", **kw):
    return Interaction(action_name=action_name, requester=requester, target=target, id=id, **kw)


# __init__ / max_interactions clamping

def test_init_defaults_and_empty_containers():
    """A fresh manager starts with six empty containers, no current/last_registered."""
    im = _manager()
    assert im.sent == {} and im.received == {} and im.lazy == {}
    assert im.sent_recently_completed == set()
    assert im.received_recently_completed == set()
    assert im.lazy_recently_completed == set()
    assert im.current is None
    assert im.last_registered is None


def test_max_interactions_falls_back_to_custom_when_nonpositive():
    """max_interactions <= 0 is replaced by Custom.MAX_INTERACTIONS; a positive value is kept."""
    assert _manager(max_interactions=-1).max_interactions == Custom.MAX_INTERACTIONS == 100
    assert _manager(max_interactions=0).max_interactions == 100
    assert _manager(max_interactions=7).max_interactions == 7


# room_for_registration / count_interactions

def test_room_for_registration_counts_only_active_dicts():
    """Capacity is enforced on the three ACTIVE dicts only; full when their total hits the limit."""
    im = _manager(max_interactions=2)
    assert im.room_for_registration() is True

    a = _inter(id="a")
    b = _inter(id="b")
    im.sent[a.uuid] = a
    assert im.room_for_registration() is True  # 1 < 2
    im.received[b.uuid] = b
    assert im.room_for_registration() is False  # 2 == 2, no longer < 2


def test_count_interactions_sums_all_six_containers():
    """count_interactions adds the three active dicts AND the three recently-completed sets."""
    im = _manager(max_interactions=100)
    a, b, c, d = _inter(id="a"), _inter(id="b"), _inter(id="c"), _inter(id="d")
    im.sent[a.uuid] = a
    im.received[b.uuid] = b
    im.lazy[c.uuid] = c
    im.sent_recently_completed.add(d)
    assert im.count_interactions() == 4


def test_room_ignores_recently_completed_while_count_includes_them():
    """BUG (interaction.py:897 vs 901): room_for_registration ignores the recently-completed
    sets that count_interactions includes, so count can exceed max while room still says full
    only on the active dicts. Here count exceeds max yet room is governed solely by actives."""
    im = _manager(max_interactions=2)
    c1, c2, c3 = _inter(id="c1"), _inter(id="c2"), _inter(id="c3")
    im.sent_recently_completed.update({c1, c2, c3})  # 3 completed, 0 active

    assert im.room_for_registration() is True            # actives 0 < 2 -> room
    assert im.count_interactions() == 3                  # but the total is 3
    assert im.count_interactions() > im.max_interactions  # which already exceeds the cap


# is_sent / is_received / is_lazy / is_known

def test_is_sent_tracks_active_and_recently_completed():
    """is_sent is True for an interaction in the active sent dict OR the recently-completed set."""
    im = _manager()
    active = _inter(id="s1")
    done = _inter(id="s2")
    im.sent[active.uuid] = active
    im.sent_recently_completed.add(done)
    assert im.is_sent(active) is True
    assert im.is_sent(done) is True
    assert im.is_sent(_inter(id="s3")) is False


def test_is_received_and_is_lazy_track_their_containers():
    """is_received / is_lazy follow their own active dict and recently-completed set."""
    im = _manager()
    r = _inter(id="r1")
    lz = _inter(id="l1")
    im.received[r.uuid] = r
    im.lazy[lz.uuid] = lz
    assert im.is_received(r) is True
    assert im.is_lazy(lz) is True
    assert im.is_received(lz) is False
    assert im.is_lazy(r) is False


def test_is_known_is_union_of_the_three_predicates():
    """is_known is True iff the interaction is sent, received, or lazy (active or completed)."""
    im = _manager()
    unknown = _inter(id="u1")
    assert im.is_known(unknown) is False
    im.lazy[unknown.uuid] = unknown
    assert im.is_known(unknown) is True


# unregister

def test_unregister_running_returns_false_without_side_effects():
    """A RUNNING interaction is never deregistered: returns False and stays in its dict."""
    im = _manager(FakeAgent())
    inter = _inter(id="run")
    inter.status = InteractionStatus.RUNNING
    im.sent[inter.uuid] = inter

    assert im.unregister(inter) is False
    assert im.sent[inter.uuid] is inter  # untouched


def test_unregister_removes_by_identity_and_returns_true():
    """A non-running, tracked interaction is removed from its dict (identity match) -> True."""
    im = _manager(FakeAgent())
    inter = _inter(id="ok")
    im.sent[inter.uuid] = inter

    assert im.unregister(inter) is True
    assert inter.uuid not in im.sent


def test_unregister_unknown_interaction_returns_false():
    """An interaction present in none of the dicts returns False (found in zero dicts)."""
    im = _manager(FakeAgent())
    assert im.unregister(_inter(id="ghost")) is False


def test_unregister_requires_object_identity_not_just_uuid():
    """Deletion uses an ``is`` check: a different object holding the same uuid is not removed."""
    im = _manager(FakeAgent())
    stored = _inter(id="dup", requester="peerA")
    im.sent[stored.uuid] = stored

    other = _inter(id="dup", requester="peerA")  # same uuid, different object
    assert other.uuid == stored.uuid
    assert im.unregister(other) is False
    assert im.sent[stored.uuid] is stored  # original kept


def test_unregister_from_multiple_dicts_returns_true():
    """BUG (interaction.py:962-965): the same uuid can legitimately live in two dicts; unregister
    then sweeps the same object out of every dict it is identical in, returning True."""
    im = _manager(FakeAgent())
    inter = _inter(id="multi")
    im.sent[inter.uuid] = inter
    im.lazy[inter.uuid] = inter

    assert im.unregister(inter) is True
    assert inter.uuid not in im.sent
    assert inter.uuid not in im.lazy


# expand_and_normalize_streams

def test_expand_empty_streams_returns_two_empty_bucket_dicts():
    """A no-stream interaction expands into two dicts of empty buckets (never None)."""
    im = _manager(FakeAgent())
    inter = _inter(id="ns")
    expanded, owned = im.expand_and_normalize_streams(inter)

    empty = {"stdin": [], "stdtar": [], "stdext": [], "stdunk": []}
    assert expanded == empty
    assert owned == empty


def test_expand_uses_stdunk_key_and_never_returns_none():
    """BUG (interaction.py:1189-1232 / 1245-1248): the buckets are keyed by the literal 'stdunk'
    (the docstring/annotation say None), and the method has no code path returning (None, None)
    despite the docstring promise and the dead ``is None`` guards in its two callers."""
    im = _manager(FakeAgent())
    expanded, owned = im.expand_and_normalize_streams(_inter(id="ns"))
    assert "stdunk" in expanded and None not in expanded
    assert expanded is not None and owned is not None


def test_expand_user_hash_appends_verbatim_and_marks_owned():
    """A user-hash stream is appended verbatim to its bucket and mirrored into the owned dict
    only when it is among the agent's owned streams."""
    agent = FakeAgent()
    agent.owned_streams_by_user_hash = {"peerB:images": FakeStream("images")}
    im = _manager(agent)

    inter = _inter(id="uh", streams={"stdext": ["peerB:images"], "stdin": ["peerC:audio"]})
    expanded, owned = im.expand_and_normalize_streams(inter)

    assert expanded["stdext"] == ["peerB:images"]
    assert expanded["stdin"] == ["peerC:audio"]
    # Only the owned hash is mirrored; the not-owned one is absent.
    assert owned["stdext"] == ["peerB:images"]
    assert owned["stdin"] == []


def test_expand_net_hash_explodes_into_per_name_user_hashes():
    """A net-hash stream is expanded over agent.known_streams into one user hash per stream name."""
    agent = FakeAgent()
    net_hash = "peerB::dm:images"  # contains '::' -> net hash
    agent.known_streams = {net_hash: {"images": FakeStream("images")}}
    im = _manager(agent)

    inter = _inter(id="nh", streams={"stdin": [net_hash]})
    expanded, owned = im.expand_and_normalize_streams(inter)

    # user_hash_from_net_hash("peerB::dm:images", "images") -> "peerB:images"
    assert expanded["stdin"] == ["peerB:images"]
    assert owned["stdin"] == []  # not owned by this agent


# match_streams

def test_match_streams_all_empty_returns_true_five_empty_dicts():
    """BUG (interaction.py:1234/1245-1248): match_streams returns a 5-tuple (the docstring claims a
    4-tuple, omitting owned_streams). With all-empty buckets it short-circuits to (True, {}*4)."""
    im = _manager(FakeAgent())
    result = im.match_streams({"stdin": [], "stdtar": [], "stdext": [], "stdunk": []})
    assert len(result) == 5
    assert result == (True, {}, {}, {}, {})


def test_match_streams_stdext_only_places_streams_without_processor():
    """A stdext-only bucket places each known stream into stdext_streams; nothing else is touched."""
    stream = FakeStream("images")
    agent = FakeAgent()
    agent.known_streams_by_user_hash = {"peerB:images": stream}
    im = _manager(agent)

    valid, stdin, stdtar, stdext, owned = im.match_streams(
        {"stdin": [], "stdtar": [], "stdext": ["peerB:images"], "stdunk": []}
    )
    assert valid is True
    assert stdin == {} and stdtar == {}
    assert stdext == {"peerB:images": stream}
    assert owned == {}  # not in owned_streams_by_user_hash


def test_match_streams_stdext_owned_stream_is_mirrored_into_owned():
    """An stdext stream that is owned is also surfaced in the owned dict."""
    stream = FakeStream("images")
    agent = FakeAgent()
    agent.known_streams_by_user_hash = {"peerB:images": stream}
    agent.owned_streams_by_user_hash = {"peerB:images": stream}
    im = _manager(agent)

    valid, _stdin, _stdtar, stdext, owned = im.match_streams(
        {"stdin": [], "stdtar": [], "stdext": ["peerB:images"], "stdunk": []}
    )
    assert valid is True
    assert stdext == {"peerB:images": stream}
    assert owned == {"peerB:images": stream}


def test_match_streams_unknown_stdext_hash_returns_false_tuple():
    """An stdext hash absent from known_streams_by_user_hash invalidates the whole match."""
    im = _manager(FakeAgent())  # empty known streams
    result = im.match_streams({"stdin": [], "stdtar": [], "stdext": ["peerB:ghost"], "stdunk": []})
    assert result == (False, {}, {}, {}, {})


# register_sent (no-stream happy path + rejections)

def test_register_sent_no_stream_happy_path():
    """A no-stream sent interaction registers: lands in sent, status REQUESTED, type SENT."""
    agent = FakeAgent()
    im = _manager(agent)
    inter = _inter(id="snt", requester="peerA", target="peerB")

    assert im.register_sent(inter, public=True) is True
    assert im.sent[inter.uuid] is inter
    assert inter.status == InteractionStatus.REQUESTED
    assert inter.type == InteractionType.SENT
    assert im.last_registered is inter
    assert im.count_interactions() == 1
    # Target was resolved in place through the identity resolver.
    assert inter.target == ["peerB"]


def test_register_sent_uuid_collision_in_sent_is_rejected():
    """A different object already holding the uuid in self.sent blocks registration (returns False)."""
    agent = FakeAgent()
    im = _manager(agent)
    first = _inter(id="dup", requester="peerA", target="peerB")
    assert im.register_sent(first, public=True) is True

    clash = _inter(id="dup", requester="peerA", target="peerB")  # same uuid, different object
    assert clash.uuid == first.uuid
    assert im.register_sent(clash, public=True) is False
    assert im.sent[first.uuid] is first  # incumbent kept


def test_register_sent_full_capacity_rejected():
    """When the active dicts are full, register_sent refuses and does not mutate sent."""
    agent = FakeAgent()
    im = _manager(agent, max_interactions=1)
    im.received["filler_x"] = _inter(id="filler")  # occupy the single slot
    inter = _inter(id="snt", requester="peerA", target="peerB")

    assert im.register_sent(inter, public=True) is False
    assert inter.uuid not in im.sent


def test_register_sent_unknown_target_rejected_via_resolve():
    """An unresolvable target makes resolve() fail, so register_sent returns False and skips sent."""
    agent = FakeAgent()
    agent.unresolvable_targets = {"peerGhost"}
    im = _manager(agent)
    inter = _inter(id="snt", requester="peerA", target="peerGhost")

    assert im.register_sent(inter, public=True) is False
    assert inter.uuid not in im.sent


def test_register_sent_reserved_kwarg_rejected():
    """A reserved action kwarg (e.g. 'interaction') fails the consistency check -> False."""
    agent = FakeAgent()
    im = _manager(agent)
    inter = _inter(id="snt", requester="peerA", target="peerB", action_kwargs={"interaction": 1})

    assert im.register_sent(inter, public=True) is False
    assert inter.uuid not in im.sent


def test_register_sent_echoed_inconsistent_field_rejected():
    """An interaction-field kwarg echoed with a value different from the actual field is rejected."""
    agent = FakeAgent()
    im = _manager(agent)
    # num_steps is an INTERACTION_FIELD_NAME; the interaction's real num_steps is -1, not 999.
    inter = _inter(id="snt", requester="peerA", target="peerB", action_kwargs={"num_steps": 999})
    assert inter.num_steps == -1

    assert im.register_sent(inter, public=True) is False
    assert inter.uuid not in im.sent


# register_received / register_lazy (no-stream)

def test_register_received_no_stream_happy_path():
    """A received interaction from a known requester registers: status/type RECEIVED."""
    agent = FakeAgent()
    agent.all_agents = {"peerA"}
    im = _manager(agent)
    inter = _inter(id="rcv", requester="peerA", target="peerSelf")

    assert im.register_received(inter, public=True) is True
    assert im.received[inter.uuid] is inter
    assert inter.status == InteractionStatus.RECEIVED
    assert inter.type == InteractionType.RECEIVED
    assert im.last_registered is inter


def test_register_received_unknown_requester_rejected():
    """An interaction whose requester is neither a known agent nor the system label is refused."""
    agent = FakeAgent()  # all_agents empty
    im = _manager(agent)
    inter = _inter(id="rcv", requester="peerStranger", target="peerSelf")

    assert im.register_received(inter, public=True) is False
    assert inter.uuid not in im.received


def test_register_received_uuid_collision_in_received_rejected():
    """A different object holding the uuid in self.received blocks the registration."""
    agent = FakeAgent()
    agent.all_agents = {"peerA"}
    im = _manager(agent)
    first = _inter(id="dup", requester="peerA", target="peerSelf")
    assert im.register_received(first, public=True) is True

    clash = _inter(id="dup", requester="peerA", target="peerSelf")
    assert im.register_received(clash, public=True) is False
    assert im.received[first.uuid] is first


def test_register_lazy_system_requester_happy_path():
    """A lazy system interaction (requester == system label) registers: status/type LAZY,
    even with an empty all_agents (the system label bypasses the membership check)."""
    agent = FakeAgent()  # all_agents empty on purpose
    im = _manager(agent)
    inter = _inter(id="lz", requester=Custom.SYSTEM_INTERACTION_LABEL)

    assert im.register_lazy(inter, public=True) is True
    assert im.lazy[inter.uuid] is inter
    assert inter.status == InteractionStatus.LAZY
    assert inter.type == InteractionType.LAZY


def test_register_received_failure_does_not_mutate_dict():
    """On the rejection path the received dict is left untouched (no partial registration)."""
    agent = FakeAgent()  # unknown requester -> rejection inside __register_received_or_lazy
    im = _manager(agent)
    inter = _inter(id="rcv", requester="peerStranger", target="peerSelf")

    im.register_received(inter, public=True)
    assert im.received == {}


# Cross-dict uuid invariant (suspected bug)

def test_same_uuid_can_live_in_sent_and_received_simultaneously():
    """BUG (interaction.py:962-965/1092): the uuid-collision guard inspects only the *own* dict,
    so a uuid already present in sent does not block register_received; the same uuid then ends
    up in two dicts at once."""
    agent = FakeAgent()
    agent.all_agents = {"peerA"}
    im = _manager(agent)

    incumbent = _inter(id="shared", requester="peerA", target="peerB")
    im.sent[incumbent.uuid] = incumbent  # placed as if sent

    twin = _inter(id="shared", requester="peerA", target="peerSelf")  # same uuid, different object
    assert twin.uuid == incumbent.uuid
    assert im.register_received(twin, public=True) is True

    # The invariant "a uuid lives in at most one dict" is violated.
    assert im.sent[incumbent.uuid] is incumbent
    assert im.received[twin.uuid] is twin


# Broken-callback contract
# In production a callback naming a non-existent agent method is rejected FAIL-FAST:
# __check_callback_method runs at the top of every register_* and its log.critical
# raises GenException (logger.py, "Preserve the raising contract") BEFORE any dict is
# mutated. The suite-wide conftest swaps the logger for a no-op that also neuters the
# raise, so these tests restore the raising logger locally (same trick as test_hsm.py)
# to characterize the production behaviour.

@pytest.fixture
def raising_critical_logger():
    from unaiverse.utils.logger import log as _log
    saved = getattr(_log, "_Logger__instance", None)
    setattr(_log, "_Logger__instance", None)
    yield
    setattr(_log, "_Logger__instance", saved)


def test_register_sent_broken_callback_raises_before_registering(raising_critical_logger):
    """A sent interaction whose callback names a non-existent agent method is refused
    at registration time by GenException, with no partial state left behind."""
    agent = FakeAgent()
    im = _manager(agent)
    inter = _inter(id="cb", requester="peerA", target="peerB", callback="no_such_callback")

    with pytest.raises(GenException):
        im.register_sent(inter, public=True)
    assert im.sent == {}


def test_register_received_broken_callback_raises_before_registering(raising_critical_logger):
    """Same fail-fast contract on the received path (only reachable with a locally
    built interaction: the callback does not survive the wire, see the round-trip
    test below)."""
    agent = FakeAgent()
    agent.all_agents = {"peerA"}
    im = _manager(agent)
    inter = _inter(id="cb", requester="peerA", target="peerSelf", callback="no_such_callback")

    with pytest.raises(GenException):
        im.register_received(inter, public=True)
    assert im.received == {}


def test_register_lazy_broken_callback_raises_before_registering(raising_critical_logger):
    """Same fail-fast contract on the lazy path (system interactions built by the HSM)."""
    agent = FakeAgent()
    im = _manager(agent)
    inter = _inter(id="cb", requester=Custom.SYSTEM_INTERACTION_LABEL, callback="no_such_callback")

    with pytest.raises(GenException):
        im.register_lazy(inter, public=True)
    assert im.lazy == {}


def test_callback_does_not_survive_the_wire():
    """to_dict/from_dict deliberately drop the callback, so an interaction received
    from the network always carries callback=None: the broken-callback check can only
    ever fire on locally built interactions (developer error), never remotely."""
    inter = _inter(id="cb", requester="peerA", target="peerB", callback="no_such_callback")
    back = Interaction.from_dict(inter.to_dict())
    assert back.callback is None
