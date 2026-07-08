"""Characterization tests for the human chat-routing cache.

These pin the behaviour of the two halves of the human interaction cache:

  * WRITE side  ``Agent.__rebind_stdin_if_human``  (src/unaiverse/agent.py)
  * READ  side  ``AgentBasics.prepare_stdin_if_human`` (src/unaiverse/agent_basics.py)

Outside a world the cache is keyed by the requester (the broadcaster agent).
Inside a world (``public == False`` / ``in_world == True``) BOTH methods
deliberately replace that key with ``get_world_node_peer_id()`` (the REMOTE
world host): the world acts as a single well-known slot for the pending
interaction. This is by design, not a routing bug: the read side is driven by
the interact-mode keyboard loop, which only knows the local world peer id and
could never name the broadcaster, so the well-known world key is what lets the
store and the lookup meet (see the "no matter who sent the interaction" and
"Zero-step" comments at the two call sites). The typed reply's network
destination is never derived from this cache key: recipients come from
``interaction.requester`` via the interaction manager, so the reply reaches
the broadcaster regardless of how the cache is keyed.

The accepted limit of the design is pinned below: the world slot is single,
so concurrent requesters overwrite each other and the last one wins.

The methods only touch ``self`` via plain attribute/method access, so they are
driven as UNBOUND functions over hand-built stubs, with no real Agent/Node and
no network or Go shared library.
"""
import pytest

from unaiverse.agent import Agent
from unaiverse.custom import Custom

pytestmark = pytest.mark.unit


# Distinct, non-equal sentinels so the world-key override is observable: a
# lookup that swaps BROADCASTER -> WORLD_NODE changes the result.
BROADCASTER = "BROADCASTER"
WORLD_NODE = "WORLD_NODE"


# Fakes

class FakeInteraction:
    """Minimal stand-in exposing only the four members both methods read:
    ``requester``, ``uuid``, ``is_system()`` and ``is_completed()``. Avoids the
    heavy real ``Interaction`` construction."""

    def __init__(self, requester=BROADCASTER, uuid="int-uuid-123",
                 is_system=False, is_completed=False):
        self.requester = requester
        self.uuid = uuid
        self._is_system = is_system
        self._is_completed = is_completed

    def is_system(self) -> bool:
        return self._is_system

    def is_completed(self) -> bool:
        return self._is_completed


class FakeConnPool:
    """Connection-pool stand-in: only ``get_world_node_peer_id`` is consulted,
    returning the REMOTE world host peer-id."""

    def __init__(self, world_node_peer_id=WORLD_NODE):
        self._world = world_node_peer_id

    def get_world_node_peer_id(self):
        return self._world


class FakeBehav:
    """Behaviour stand-in exposing ``get_wildcards`` only. Default ``{}`` keeps
    PARTNER_WILDCARD absent so the system-interaction blocking branch is never
    taken and the store branch is reached."""

    def __init__(self, wildcards=None):
        self._wildcards = wildcards if wildcards is not None else {}

    def get_wildcards(self):
        return self._wildcards


class RebindStub:
    """``self`` for ``Agent.__rebind_stdin_if_human``. is_human/behaving_in_world
    are stubbed straight to the desired booleans; the cache is a real dict so the
    test can read back the key that was used."""

    def __init__(self, in_world: bool, world_node_peer_id=WORLD_NODE, wildcards=None):
        self._in_world = in_world
        self._conn = FakeConnPool(world_node_peer_id)
        self.proc_human_peer_id_to_interaction = {}
        self.behav = FakeBehav(wildcards)
        self.behav_lone_wolf = FakeBehav(wildcards)
        self.binding_calls = 0

    def is_human(self) -> bool:
        return True

    def behaving_in_world(self) -> bool:
        return self._in_world

    def get_connection_pool_manager(self):
        return self._conn

    def set_default_stdin_binding(self, *args, **kwargs):
        self.binding_calls += 1


class NonHumanRebindStub(RebindStub):
    def is_human(self) -> bool:
        return False


class PrepareStub:
    """``self`` for ``AgentBasics.prepare_stdin_if_human``. Only ``node_conn``,
    ``proc_human_peer_id_to_interaction`` and ``set_default_stdin_binding`` are
    touched."""

    def __init__(self, cache=None, world_node_peer_id=WORLD_NODE):
        self.node_conn = FakeConnPool(world_node_peer_id)
        self.proc_human_peer_id_to_interaction = cache if cache is not None else {}
        self.binding_public = []

    def set_default_stdin_binding(self, public=None):
        self.binding_public.append(public)


# Convenience aliases for the unbound functions under test.
_rebind = Agent._Agent__rebind_stdin_if_human
_prepare = Agent.prepare_stdin_if_human


# READ side: AgentBasics.prepare_stdin_if_human

def test_prepare_public_returns_broadcaster_interaction_uuid():
    """PUBLIC path (lone wolf): the passed-in broadcaster peer_id is NOT swapped,
    so a cached entry under the broadcaster is found and its uuid returned."""
    inter = FakeInteraction(requester=BROADCASTER, uuid="int-123")
    stub = PrepareStub(cache={BROADCASTER: inter})

    result = _prepare(stub, public=True, peer_id=BROADCASTER)

    assert result == "int-123"
    # The binding is reset with the same `public` flag that was passed in.
    assert stub.binding_public == [True]


def test_prepare_world_ignores_requester_keyed_entries():
    """World path: the lookup only consults the single world slot, so an entry
    keyed under a specific requester is invisible and the read falls back to
    the SYSTEM interaction uuid. (The write side never produces such an entry
    in a world; this pins the single-slot contract from the read side.)"""
    inter = FakeInteraction(requester=BROADCASTER, uuid="int-123")
    stub = PrepareStub(cache={BROADCASTER: inter}, world_node_peer_id=WORLD_NODE)

    result = _prepare(stub, public=False, peer_id=BROADCASTER)

    assert result == Custom.SYSTEM_INTERACTION_UUID == "sys_system"
    assert stub.binding_public == [False]


def test_prepare_world_finds_entry_keyed_under_world_node():
    """In the world path the lookup is performed under the world-node id, so an
    entry stored under WORLD_NODE *is* found (the symmetric proof that the key
    used is the world-node id, not the passed-in broadcaster)."""
    inter = FakeInteraction(requester=WORLD_NODE, uuid="int-world")
    stub = PrepareStub(cache={WORLD_NODE: inter}, world_node_peer_id=WORLD_NODE)

    result = _prepare(stub, public=False, peer_id=BROADCASTER)

    assert result == "int-world"


def test_prepare_purges_completed_entries_before_lookup():
    """The residual-purge loop drops only COMPLETED interactions; the live
    broadcaster entry survives and drives the returned uuid (public path)."""
    live = FakeInteraction(requester=BROADCASTER, uuid="int-live", is_completed=False)
    done = FakeInteraction(requester="OLD", uuid="int-done", is_completed=True)
    stub = PrepareStub(cache={"OLD": done, BROADCASTER: live})

    result = _prepare(stub, public=True, peer_id=BROADCASTER)

    assert result == "int-live"
    # The completed entry was removed; the live one kept.
    assert "OLD" not in stub.proc_human_peer_id_to_interaction
    assert BROADCASTER in stub.proc_human_peer_id_to_interaction


def test_prepare_miss_returns_system_uuid():
    """A genuine miss (empty cache) returns the SYSTEM interaction uuid."""
    stub = PrepareStub(cache={})
    assert _prepare(stub, public=True, peer_id=BROADCASTER) == Custom.SYSTEM_INTERACTION_UUID


# WRITE side: Agent.__rebind_stdin_if_human

def test_rebind_not_human_short_circuits_true():
    """Non-human self early-returns True without touching the cache."""
    stub = NonHumanRebindStub(in_world=True)
    inter = FakeInteraction()
    assert _rebind(stub, inter) is True
    assert stub.proc_human_peer_id_to_interaction == {}
    assert stub.binding_calls == 0


def test_rebind_none_interaction_short_circuits_true():
    """A None interaction early-returns True (nothing stored, no binding reset)."""
    stub = RebindStub(in_world=True)
    assert _rebind(stub, None) is True
    assert stub.proc_human_peer_id_to_interaction == {}
    assert stub.binding_calls == 0


def test_rebind_lone_wolf_stores_under_requester():
    """NON-world path (lone wolf): a non-system interaction is correctly stored
    under interaction.requester (the broadcaster), and the binding is reset."""
    stub = RebindStub(in_world=False)
    inter = FakeInteraction(requester=BROADCASTER, uuid="int-1", is_system=False)

    assert _rebind(stub, inter) is True
    assert stub.proc_human_peer_id_to_interaction == {BROADCASTER: inter}
    assert WORLD_NODE not in stub.proc_human_peer_id_to_interaction
    assert stub.binding_calls == 1


def test_rebind_world_stores_under_world_node():
    """World path: the non-system interaction is stored under the world-node id
    rather than the requester, so it lands in the single well-known slot that
    the read side (which only knows the world id) will look up."""
    stub = RebindStub(in_world=True)
    inter = FakeInteraction(requester=BROADCASTER, uuid="int-1", is_system=False)

    assert _rebind(stub, inter) is True
    assert WORLD_NODE in stub.proc_human_peer_id_to_interaction
    assert stub.proc_human_peer_id_to_interaction[WORLD_NODE] is inter
    assert BROADCASTER not in stub.proc_human_peer_id_to_interaction


def test_rebind_world_last_requester_wins_the_slot():
    """World path, accepted design limit: the world slot is single, so a second
    pending interaction from a DIFFERENT requester overwrites the first one and
    the human can only answer the most recent request."""
    stub = RebindStub(in_world=True)
    first = FakeInteraction(requester=BROADCASTER, uuid="int-1", is_system=False)
    second = FakeInteraction(requester="OTHER_REQUESTER", uuid="int-2", is_system=False)

    assert _rebind(stub, first) is True
    assert _rebind(stub, second) is True
    assert stub.proc_human_peer_id_to_interaction == {WORLD_NODE: second}


def test_rebind_system_blocked_when_partner_already_stored():
    """A system interaction is blocked (returns False) when a PARTNER_WILDCARD
    partner is already present in the cache (the dashed-interaction guard)."""
    stub = RebindStub(in_world=True, wildcards={Custom.PARTNER_WILDCARD: BROADCASTER})
    stub.proc_human_peer_id_to_interaction[BROADCASTER] = FakeInteraction(uuid="prev")
    sys_inter = FakeInteraction(requester=Custom.SYSTEM_INTERACTION_LABEL,
                                uuid="sys", is_system=True)

    assert _rebind(stub, sys_inter) is False
    # Blocked before any store / binding reset.
    assert stub.binding_calls == 0


def test_rebind_system_not_stored_when_no_partner():
    """A system interaction with no partner wildcard is not stored (the store
    branch is gated on ``not is_system``) but still resets the binding."""
    stub = RebindStub(in_world=True)
    sys_inter = FakeInteraction(requester=Custom.SYSTEM_INTERACTION_LABEL,
                                uuid="sys", is_system=True)

    assert _rebind(stub, sys_inter) is True
    assert stub.proc_human_peer_id_to_interaction == {}
    assert stub.binding_calls == 1
