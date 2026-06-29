"""Characterization tests for the human chat-routing cache (BUG 2).

These pin the *current* behaviour of the two halves of the per-broadcaster
human interaction cache and isolate the routing bug that misroutes a human's
reply toward the remote world-node peer-id instead of the broadcaster:

  * WRITE side  ``Agent.__rebind_stdin_if_human``  (src/unaiverse/agent.py:1740)
  * READ  side  ``AgentBasics.prepare_stdin_if_human`` (src/unaiverse/agent_basics.py:435)

When the human behaves inside a world (``public == False`` / ``in_world ==
True``) both methods replace the broadcaster peer-id with
``get_world_node_peer_id()`` (the REMOTE world host). The store-side lands the
interaction under the world-node id (agent.py:1783) and the read-side looks it
up under the same world-node id (agent_basics.py:439), so a per-broadcaster
entry is never found and the read falls back to ``Custom.SYSTEM_INTERACTION_UUID``.
The downstream consequence (node.py:1865-1867) is that the world node receives
an agent-to-agent INTERACTION it never expects and purges the human.

The two correct-behaviour assertions are marked ``xfail(strict=True)``: they
MUST fail against today's source and only go green once both world-node-id
overrides are replaced by the broadcaster/requester id. The remaining tests pin
the current (buggy and non-buggy) behaviour and stay green.

The methods only touch ``self`` via plain attribute/method access, so they are
driven as UNBOUND functions over hand-built stubs, with no real Agent/Node and
no network or Go shared library.
"""
import pytest

from unaiverse.agent import Agent
from unaiverse.custom import Custom

pytestmark = pytest.mark.unit


# Distinct, non-equal sentinels so the buggy override is observable: a lookup
# that swaps BROADCASTER -> WORLD_NODE changes the result.
BROADCASTER = "BROADCASTER"
WORLD_NODE = "WORLD_NODE"


# ── Fakes ────────────────────────────────────────────────────────────────────

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


# ── READ side: AgentBasics.prepare_stdin_if_human ────────────────────────────

def test_prepare_public_returns_broadcaster_interaction_uuid():
    """PUBLIC path (lone wolf): the passed-in broadcaster peer_id is NOT swapped,
    so a cached entry under the broadcaster is found and its uuid returned."""
    inter = FakeInteraction(requester=BROADCASTER, uuid="int-123")
    stub = PrepareStub(cache={BROADCASTER: inter})

    result = _prepare(stub, public=True, peer_id=BROADCASTER)

    assert result == "int-123"
    # The binding is reset with the same `public` flag that was passed in.
    assert stub.binding_public == [True]


def test_prepare_world_currently_falls_back_to_system_uuid():
    """CURRENT (buggy) world path: with the broadcaster keyed in the cache,
    ``public=False`` swaps the key to the world-node id (agent_basics.py:439),
    misses, and returns the SYSTEM interaction uuid instead of the real one."""
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


@pytest.mark.xfail(reason="bug: agent_basics.py:439 swaps broadcaster->world-node id on read",
                   strict=True)
def test_prepare_world_should_return_broadcaster_interaction_uuid():
    """CORRECT behaviour (read side): inside a world, a reply typed for a
    broadcaster must resolve to that broadcaster's interaction uuid. The cache is
    keyed under the broadcaster and ``prepare_stdin_if_human(public=False,
    peer_id=BROADCASTER)`` must return its uuid, NOT the SYSTEM fallback. It fails
    today because line 439 overrides peer_id with the world-node id."""
    inter = FakeInteraction(requester=BROADCASTER, uuid="int-123")
    stub = PrepareStub(cache={BROADCASTER: inter}, world_node_peer_id=WORLD_NODE)

    assert _prepare(stub, public=False, peer_id=BROADCASTER) == "int-123"


# ── WRITE side: Agent.__rebind_stdin_if_human ────────────────────────────────

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


def test_rebind_world_currently_stores_under_world_node():
    """CURRENT (buggy) world path: the non-system interaction is stored under the
    world-node id (agent.py:1783) rather than the broadcaster."""
    stub = RebindStub(in_world=True)
    inter = FakeInteraction(requester=BROADCASTER, uuid="int-1", is_system=False)

    assert _rebind(stub, inter) is True
    assert WORLD_NODE in stub.proc_human_peer_id_to_interaction
    assert stub.proc_human_peer_id_to_interaction[WORLD_NODE] is inter
    assert BROADCASTER not in stub.proc_human_peer_id_to_interaction


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


@pytest.mark.xfail(reason="bug: agent.py:1783 stores under world-node id instead of requester",
                   strict=True)
def test_rebind_world_should_store_under_requester():
    """CORRECT behaviour (write side): inside a world, a non-system chat from a
    broadcaster must be stored under interaction.requester (the broadcaster), and
    the world-node id must NOT be used as the key. It fails today because line
    1783 keys the entry on the world-node id."""
    stub = RebindStub(in_world=True)
    inter = FakeInteraction(requester=BROADCASTER, uuid="int-1", is_system=False)

    assert _rebind(stub, inter) is True
    assert BROADCASTER in stub.proc_human_peer_id_to_interaction
    assert stub.proc_human_peer_id_to_interaction[BROADCASTER] is inter
    assert WORLD_NODE not in stub.proc_human_peer_id_to_interaction
