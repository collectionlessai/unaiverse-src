"""Tests for the canonical world-definition members and hash.

The invariant under test: the joiner-side hash (``canonical_world_hash`` over
the members rebuilt from the grant bundle) equals the hash the root server
computes over the bundle read from a world's declared repository (members
sorted by name, folded as ``<name>\\x00<content>`` into BLAKE2b-256).
"""

import pytest

from unaiverse.utils.misc import canonical_world_hash, world_definition_members

pytestmark = pytest.mark.unit


AGENT_FILES = {
    'guest.py': "from .config import Config\n",
    'config.py': "X = 1\n",
    'stats.py': "class WStats:\n    pass\n",
}
ROLES_FSM = {
    'guest': '{"role": "guest", "states": {}}',
    'hotel_manager': '{"role": "hotel_manager", "states": {}}',
}


def test_members_layout_code_and_fsm_prefixes():
    members = world_definition_members(AGENT_FILES, ROLES_FSM)
    assert set(members) == {
        'code/guest.py', 'code/config.py', 'code/stats.py',
        'fsm/guest.json', 'fsm/hotel_manager.json',
    }
    assert members['code/config.py'] == b"X = 1\n"
    assert members['fsm/guest.json'] == ROLES_FSM['guest'].encode('utf-8')


def test_canonical_hash_matches_root_server_fold_golden():
    """Golden digest, produced by fold_members in the root server's
    app/services/world_bundle.py (commit 6dd42f6) over exactly these members.

    The whole attestation feature hinges on the two folds staying byte-identical:
    a divergence of one byte fails every join to every attested world, so any
    change that trips this constant must be rejected, not accommodated.
    """
    members = world_definition_members(AGENT_FILES, ROLES_FSM)
    assert canonical_world_hash(members) == \
        "db1d6379e835876e7294844a4bc2a2c59dc4c6ce1a6484fae858630b4ea17187"


def test_canonical_hash_is_order_independent_and_deterministic():
    members = world_definition_members(AGENT_FILES, ROLES_FSM)
    reordered = dict(reversed(list(members.items())))
    assert canonical_world_hash(members) == canonical_world_hash(reordered)
    assert len(canonical_world_hash(members)) == 64


def test_canonical_hash_is_content_and_name_sensitive():
    members = world_definition_members(AGENT_FILES, ROLES_FSM)
    changed = dict(members)
    changed['code/config.py'] = b"X = 2\n"
    assert canonical_world_hash(changed) != canonical_world_hash(members)
    renamed = {('code/renamed.py' if k == 'code/config.py' else k): v
               for k, v in members.items()}
    assert canonical_world_hash(renamed) != canonical_world_hash(members)
