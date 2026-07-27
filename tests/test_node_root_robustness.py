"""Root-channel robustness (Tranche C, commit 1).

The periodic root calls in the main loop are capped (off the event loop, under a
timeout) and non-fatal, and ``get_node_token`` is split into a thread-safe fetch
(``_request_node_token``, no self-mutation) and a main-task apply
(``_apply_node_token``), so a fetch thread that outlives its timeout cannot mutate
the token / push it into the transport concurrently with the main loop.

The cap and the asymmetric refresh backoff live at the loop call-site (inside
``run_async``) and are not exercised here; these unit tests pin the fetch/apply
split that makes running the fetch off the event loop safe.
"""

import pytest

from unaiverse.networking.node.node import Node


def _mini_node(**attrs):
    node = Node.__new__(Node)
    node.node_id = "n1"
    node.unaiverse_key = "acct-key"
    node.node_token = "old-token"
    node.public_key = "old-pk"
    for key, value in attrs.items():
        setattr(node, key, value)
    return node


class _RecConn:
    """Records the token/turn pushes _apply_node_token fans into the transport."""

    def __init__(self):
        self.tokens = []
        self.turn = []

    def set_token(self, token):
        self.tokens.append(token)

    def set_turn_credentials(self, ice_turn_servers):
        self.turn.append(ice_turn_servers)


def test_request_node_token_does_not_mutate_self():
    node = _mini_node()
    calls = []

    def fake_root(api, payload):
        calls.append((api, payload))
        return {"token": "new-token", "public_key": "new-pk"}

    node._Node__root = fake_root
    resp = node._request_node_token(["p-pub", "p-world"])

    assert resp == {"token": "new-token", "public_key": "new-pk"}
    # The fetch may run off the event loop in a thread: it must not touch self.
    assert node.node_token == "old-token"
    assert node.public_key == "old-pk"
    assert len(calls) == 1


def test_request_node_token_raises_after_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)  # skip the real backoff wait
    node = _mini_node()

    def fake_root(api, payload):
        raise RuntimeError("root unreachable")

    node._Node__root = fake_root
    with pytest.raises(Exception):
        node._request_node_token(["p-pub"])
    # No self-mutation on failure either.
    assert node.node_token == "old-token"


def test_apply_node_token_mutates_self_and_pushes_to_transport():
    conn = _RecConn()
    node = _mini_node(conn=conn)

    node._apply_node_token({
        "token": "new-token", "public_key": "new-pk",
        "turn_info": {"urls": "turn:host", "username": "u", "credential": "c"},
    })

    assert node.node_token == "new-token"
    assert node.public_key == "new-pk"
    assert conn.tokens == ["new-token"]
    assert conn.turn and conn.turn[0][0]["urls"] == "turn:host"


def test_get_node_token_fetches_then_applies():
    conn = _RecConn()
    node = _mini_node(conn=conn)

    def fake_root(api, payload):
        return {"token": "fresh", "public_key": "fresh-pk"}

    node._Node__root = fake_root
    node.get_node_token(["p-pub", "p-world"])

    assert node.node_token == "fresh"
    assert conn.tokens == ["fresh"]


# World-definition upload (one-shot at world boot)


def _world_ns(files, fsms):
    from types import SimpleNamespace
    from unaiverse.utils.misc import pack_py_files
    return SimpleNamespace(packed_agent_files=pack_py_files(files), role_to_behav=fsms)


def test_send_world_definition_posts_canonical_archive_and_fsms():
    """The uploaded archive folds (server-side algorithm) to the canonical hash of
    exactly the members that ride the grant bundle, and the payload field names
    match the root API (code_b64, world_roles_fsm)."""
    import base64

    from test_world_definition import _server_side_hash
    from unaiverse.utils.misc import canonical_world_hash, world_definition_members

    files = {'player.py': "X = 1\n", 'stats.py': "class WStats:\n    pass\n"}
    fsms = {'player': '{"role": "player"}'}
    node = _mini_node(world=_world_ns(files, fsms))
    calls = []

    def fake_root(api, payload):
        calls.append((api, payload))
        if api.endswith("/code/post"):
            return {"hash": canonical_world_hash(world_definition_members(files, fsms)),
                    "size_bytes": 1}
        return None

    node._Node__root = fake_root
    node.send_world_definition()

    assert [a for a, _ in calls] == ["/account/node/code/post", "/account/node/fsm/post"]
    code_payload = calls[0][1]
    assert code_payload["node_id"] == "n1"
    archive = base64.b64decode(code_payload["code_b64"])
    assert _server_side_hash(archive) == canonical_world_hash(
        world_definition_members(files, fsms))
    assert calls[1][1]["world_roles_fsm"] == fsms


def test_send_world_definition_gives_up_without_raising(monkeypatch):
    """Three failed attempts must be logged, never fatal: the world still boots
    (joiners simply have no stored hash to check against)."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    node = _mini_node(world=_world_ns({'a.py': "x = 1\n"}, {}))
    attempts = []

    def fake_root(api, payload):
        attempts.append(api)
        raise RuntimeError("root down")

    node._Node__root = fake_root
    node.send_world_definition()  # Must not raise

    assert attempts == ["/account/node/code/post"] * 3
