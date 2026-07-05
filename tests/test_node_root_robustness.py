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
