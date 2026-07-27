"""Tests for the canonical world-definition members, hash and archive.

The invariant under test: the joiner-side hash (``canonical_world_hash`` over
the members rebuilt from the grant bundle) equals the hash the root server
computes over the uploaded ``.tar.gz`` (members sorted by name, folded as
``<name>\\x00<content>`` into BLAKE2b-256).
"""

import gzip
import hashlib
import io
import tarfile

import pytest

from unaiverse.utils.misc import (build_world_archive, canonical_world_hash,
                                  world_definition_members)

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


def _server_side_hash(gzip_bytes: bytes) -> str:
    """Replica of the root server's canonical_hash tar path (code_storage.py)."""
    with tarfile.open(fileobj=io.BytesIO(gzip_bytes), mode='r:gz') as tar:
        h = hashlib.blake2b(digest_size=32)
        members = sorted((m for m in tar.getmembers() if m.isfile()),
                         key=lambda m: m.name)
        for m in members:
            extracted = tar.extractfile(m)
            content = extracted.read() if extracted is not None else b''
            h.update(m.name.encode('utf-8'))
            h.update(b'\x00')
            h.update(content)
        return h.hexdigest()


def test_members_layout_code_and_fsm_prefixes():
    members = world_definition_members(AGENT_FILES, ROLES_FSM)
    assert set(members) == {
        'code/guest.py', 'code/config.py', 'code/stats.py',
        'fsm/guest.json', 'fsm/hotel_manager.json',
    }
    assert members['code/config.py'] == b"X = 1\n"
    assert members['fsm/guest.json'] == ROLES_FSM['guest'].encode('utf-8')


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


def test_archive_bytes_are_deterministic():
    members = world_definition_members(AGENT_FILES, ROLES_FSM)
    assert build_world_archive(members) == build_world_archive(dict(reversed(list(members.items()))))


def test_archive_hash_matches_server_side_fold():
    members = world_definition_members(AGENT_FILES, ROLES_FSM)
    archive = build_world_archive(members)
    assert _server_side_hash(archive) == canonical_world_hash(members)


def test_archive_is_a_valid_gzip_tar_with_all_members():
    members = world_definition_members(AGENT_FILES, ROLES_FSM)
    archive = build_world_archive(members)
    gzip.decompress(archive)  # Valid gzip stream
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tar:
        names = [m.name for m in tar.getmembers()]
        assert names == sorted(members)
        for m in tar.getmembers():
            assert m.isfile()
            assert tar.extractfile(m).read() == members[m.name]
