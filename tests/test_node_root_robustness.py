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

from unaiverse.custom import GenException, RootServerError
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

    attempts = []

    def fake_root(api, payload):
        attempts.append(api)
        raise RootServerError("root unreachable", api=api)

    node._Node__root = fake_root
    with pytest.raises(RootServerError):
        node._request_node_token(["p-pub"])
    # Three attempts for a transport failure, and no self-mutation on failure either.
    assert len(attempts) == 3
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


# World-source declaration (one-shot at world boot)


FILES = {'player.py': "X = 1\n", 'stats.py': "class WStats:\n    pass\n"}
FSMS = {'player': '{"role": "player"}'}


def _world_ns(files, fsms):
    from types import SimpleNamespace
    from unaiverse.utils.misc import pack_py_files
    return SimpleNamespace(packed_agent_files=pack_py_files(files), role_to_behav=fsms)


def _local_hash(files, fsms):
    from unaiverse.utils.misc import canonical_world_hash, world_definition_members
    return canonical_world_hash(world_definition_members(files, fsms))


def _world_node(files=FILES, fsms=FSMS, **attrs):
    return _mini_node(world=_world_ns(files, fsms),
                      code_repo=attrs.pop("code_repo", "owner/repo"),
                      code_commit=attrs.pop("code_commit", "c" * 40),
                      **attrs)


def test_declare_world_source_posts_fsms_and_declares():
    """The declaration carries the caller_* auth pair (the endpoint's decorator does
    not read the node_token that __root injects) and the repo/commit pointer; the
    role FSMs are posted for the website, first and independently."""
    node = _world_node()
    calls = []

    def fake_root(api, payload):
        calls.append((api, payload))
        if api.endswith("/code/source/set"):
            return {"hash": _local_hash(FILES, FSMS)}
        return None

    node._Node__root = fake_root
    node.declare_world_source()

    assert [a for a, _ in calls] == ["/account/node/fsm/post", "/account/node/code/source/set"]
    assert calls[0][1]["world_roles_fsm"] == FSMS
    declared = calls[1][1]
    assert declared["node_id"] == "n1"
    assert declared["caller_node_id"] == "n1"
    assert declared["caller_node_token"] == "old-token"
    assert declared["repo"] == "owner/repo"
    assert declared["commit_sha"] == "c" * 40


@pytest.mark.xfail(reason="the code/source/clear call is parked ('remove wrong call to repo-cleaning API')",
                   strict=True)
def test_declare_world_source_unconfigured_world_stays_unpublished():
    """With no code_repo/code_commit the world declares nothing: the FSMs are still
    posted, no source/set call is made, and any declaration left on the server by a
    previous run is cleared (or a fresh token would still mint its stale digest)."""
    node = _world_node(code_repo=None, code_commit=None)
    calls = []

    def fake_root(api, payload):
        calls.append(api)
        return None

    node._Node__root = fake_root
    node.declare_world_source()  # Must not raise

    assert calls == ["/account/node/fsm/post", "/account/node/code/source/clear"]


def test_declare_world_source_clear_failure_is_not_fatal():
    """The clear is defensive (a no-op for a world that never declared): its failure
    is loudly logged but must not stop an unpublished world from booting."""
    node = _world_node(code_repo=None, code_commit=None)

    def fake_root(api, payload):
        if api.endswith("/source/clear"):
            raise RootServerError("down", api=api)
        return None

    node._Node__root = fake_root
    node.declare_world_source()  # Must not raise


@pytest.mark.parametrize("attrs", [{"code_repo": None}, {"code_commit": None}])
def test_declare_world_source_half_configured_is_fatal(attrs):
    """Exactly one of code_repo/code_commit set is an author mistake: a declaration
    was requested, so silently running unpublished would defeat the intent."""
    node = _world_node(**attrs)
    node._Node__root = lambda api, payload: None

    with pytest.raises(GenException, match="missing"):
        node.declare_world_source()


def test_declare_world_source_repo_timeout_is_retried(monkeypatch):
    """repo_timeout is the one transient refusal (the server's git read timed out):
    it is retried like a transport failure instead of aborting the boot."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    node = _world_node()
    attempts = []

    def fake_root(api, payload):
        if api.endswith("/fsm/post"):
            return None
        attempts.append(api)
        if len(attempts) < 3:
            raise RootServerError("timeout", api=api, api_rejected=True,
                                  data_code="repo_timeout")
        return {"hash": _local_hash(FILES, FSMS)}

    node._Node__root = fake_root
    node.declare_world_source()  # Third attempt succeeds

    assert len(attempts) == 3


def test_declare_world_source_roleless_world_skips_fsm_post():
    node = _world_node(fsms={})
    calls = []

    def fake_root(api, payload):
        calls.append(api)
        return {"hash": _local_hash(FILES, {})}

    node._Node__root = fake_root
    node.declare_world_source()

    assert calls == ["/account/node/code/source/set"]


@pytest.mark.xfail(reason="the declare-side world-code attestation is parked ('avoided python to call "
                          "repo-related stuff if not specified')", strict=True)
def test_declare_world_source_hash_mismatch_refuses_to_start():
    """The root folded the repository, the world hands out the grant bundle: on a
    divergence every join would fail at the fail-closed joiner gate, so the world
    must refuse to start instead."""
    node = _world_node()

    def fake_root(api, payload):
        return {"hash": "0" * 64} if api.endswith("/code/source/set") else None

    node._Node__root = fake_root
    with pytest.raises(GenException, match="commit and push"):
        node.declare_world_source()


def test_declare_world_source_refusal_fails_fast():
    """A refusal is deterministic: one attempt, then the typed error propagates
    (fatal at boot, so a stale attested digest can never fail every join silently)."""
    node = _world_node()
    attempts = []

    def fake_root(api, payload):
        if api.endswith("/fsm/post"):
            return None
        attempts.append(api)
        raise RootServerError("commit unreachable", api=api, api_rejected=True,
                              data_code="commit_unreachable")

    node._Node__root = fake_root
    with pytest.raises(RootServerError):
        node.declare_world_source()
    assert len(attempts) == 1


def test_declare_world_source_transport_failure_raises_after_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    node = _world_node()
    attempts = []

    def fake_root(api, payload):
        if api.endswith("/fsm/post"):
            return None
        attempts.append(api)
        raise RootServerError("root down", api=api)

    node._Node__root = fake_root
    with pytest.raises(RootServerError):
        node.declare_world_source()
    assert len(attempts) == 3


def test_declare_world_source_fsm_post_failure_is_not_fatal():
    """The FSM post feeds the website only: its failure must not stop the boot nor
    the declaration."""
    node = _world_node()
    calls = []

    def fake_root(api, payload):
        calls.append(api)
        if api.endswith("/fsm/post"):
            raise RootServerError("fsm refused", api=api, api_rejected=True)
        return {"hash": _local_hash(FILES, FSMS)}

    node._Node__root = fake_root
    node.declare_world_source()  # Must not raise

    assert calls == ["/account/node/fsm/post", "/account/node/code/source/set"]
