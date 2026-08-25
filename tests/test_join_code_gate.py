"""Node-level tests for the world-join code-integrity gate and its wiring.

The gate lives in ``Node.__join_world``: when the world's verified token attests
a code hash, the received bundle must fold to exactly that hash, otherwise the
join is blocked; without a claim the check is skipped (transitional fail-open).
The wiring lives in ``_join_world_after_session``: the claim annotated on the
grant blob by the public Session, the bundle roles_fsm (with the legacy raw-dict
fallback) and the legacy stats_code channel must all reach ``__join_world``.
"""

import asyncio
import json

import pytest

from node_harness import build_agent_node_shell, full_profile_dict
from unaiverse.networking.node.profile import NodeProfile
from unaiverse.utils.misc import (canonical_world_hash, pack_py_files,
                                  world_definition_members)

pytestmark = pytest.mark.unit


ROLE_FILES = {'player.py': "X = 1\n", 'stats.py': "class WStats:\n    pass\n"}
ROLES_FSM = {'player': '{"role": "player"}'}


def _join_ready_node():
    """Agent shell stubbed just enough for __join_world to reach the gate and,
    on success, return True without the heavy role-replacement machinery."""
    node, conn = build_agent_node_shell("joiner-peer")
    node.world_agent_files = None  # Node.__init__ sets these; the shell bypasses it
    node.world_role_fsms = None
    conn.world_is_public = True  # Skips the relay-reservation branch

    async def _subscribe(peer_id, channel=None):
        return True
    conn.subscribe = _subscribe
    conn.rendezvous_tag = 0

    async def _add_streams(*a, **k):
        return True
    node.agent.add_compatible_streams = _add_streams
    node._Node__replace_agent_instance_by_role = lambda *a, **k: None
    return node, conn


def _join(node, attested_code_hash):
    return node._Node__join_world(
        profile=NodeProfile.from_dict(full_profile_dict("world-node-id")), role=1,
        packed_agent_files=pack_py_files(ROLE_FILES), roles_fsm=dict(ROLES_FSM),
        rendezvous_tag=0, initial_stats=None, peer_id="world-priv-peer",
        attested_code_hash=attested_code_hash)


@pytest.mark.xfail(reason="the join-side world-code hash gate is parked ('disabled code hasing')", strict=True)
def test_join_blocked_on_code_hash_mismatch():
    async def _scenario():
        node, _ = _join_ready_node()
        assert await _join(node, "00" * 32) is False
        assert node.world_agent_files is None  # Blocked before caching anything
        assert node.world_role_fsms is None
    asyncio.run(_scenario())


def test_join_passes_when_bundle_folds_to_the_attested_hash():
    async def _scenario():
        node, _ = _join_ready_node()
        attested = canonical_world_hash(world_definition_members(ROLE_FILES, ROLES_FSM))
        assert await _join(node, attested) is True
        assert node.world_agent_files == ROLE_FILES
        assert node.world_role_fsms == ROLES_FSM
    asyncio.run(_scenario())


def test_join_skips_check_when_no_claim_attested():
    async def _scenario():
        node, _ = _join_ready_node()
        assert await _join(node, None) is True
    asyncio.run(_scenario())


# Wiring: _join_world_after_session -> __join_world kwargs


def _wired_node():
    node, conn = build_agent_node_shell("joiner-peer")
    rec = {}

    async def _fake_join(**kwargs):
        rec.update(kwargs)
        return True
    node._Node__join_world = _fake_join
    return node, rec


def _grant(raw_profile, bundle, claim=None):
    grant = {
        "world_profile_bytes": json.dumps(raw_profile).encode("utf-8"),
        "role": 1, "rendezvous_tag": 7,
        "code_bundle": json.dumps(bundle).encode("utf-8"),
        "initial_stats": b"",
    }
    if claim is not None:
        grant["world_code_hash_claim"] = claim
    return grant


def test_wiring_passes_claim_and_bundle_fsms():
    async def _scenario():
        node, rec = _wired_node()
        grant = _grant(full_profile_dict("world-node-id"),
                       {"agent_files": "packed", "roles_fsm": {"player": "{}"}},
                       claim="aa" * 32)
        assert await node._join_world_after_session("world-priv-peer", grant) is True
        assert rec["attested_code_hash"] == "aa" * 32
        assert rec["roles_fsm"] == {"player": "{}"}
        assert rec["legacy_stats_code"] is None
        assert rec["packed_agent_files"] == "packed"
    asyncio.run(_scenario())


def test_wiring_legacy_fallbacks_from_raw_profile_and_stats_code():
    """A legacy world: no roles_fsm in the bundle (they sit in the raw profile
    dynamic dict, which the current NodeProfile template drops at parse time),
    stats shipped via the dedicated stats_code field, no hash claim."""
    async def _scenario():
        node, rec = _wired_node()
        raw = full_profile_dict("world-node-id")
        raw["dynamic"]["world_roles_fsm"] = {"legacy_role": "{}"}
        grant = _grant(raw, {"agent_files": "packed",
                             "stats_code": "class WStats:\n    pass\n"})
        assert await node._join_world_after_session("world-priv-peer", grant) is True
        assert rec["attested_code_hash"] is None
        assert rec["roles_fsm"] == {"legacy_role": "{}"}
        assert rec["legacy_stats_code"] == "class WStats:\n    pass\n"
    asyncio.run(_scenario())
