"""Tests for the connpool ``get_messages`` protocol partition.

Confirms that:
  * Session-protocol messages skip ``Msg.from_bytes`` and JWT verification.
  * Non-session messages still go through the legacy chat path.
  * ``send_session`` refuses non-session protocol IDs.
"""

import base64
import asyncio
from unittest.mock import MagicMock

import pytest

from unaiverse.networking.node.connpool import (
    ConnectionPools, NodeConn, SessionInbound, SESSION_PROTOCOL_IDS,
)


class _FakeP2P:
    """Minimal stand-in: only ``pop_messages`` is called by ``get_messages``."""
    log_sub = "fake"

    def __init__(self, batch):
        self._batch = batch

    def pop_messages(self):
        return self._batch


@pytest.fixture
def pools_with_one_p2p():
    """A ConnectionPools wrapper that returns whatever messages we plant."""
    fake_p2p = _FakeP2P([])
    pools = ConnectionPools(
        max_connections=8,
        pool_name_to_p2p_name_and_ratio={NodeConn.P2P_PUBLIC: [NodeConn.P2P_PUBLIC, 1.0]},
        p2p_name_to_p2p={NodeConn.P2P_PUBLIC: fake_p2p},
    )
    return pools, fake_p2p


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Routing ─────────────────────────────────────────────────────────────────

def test_session_protocol_messages_are_partitioned(pools_with_one_p2p):
    """A batch containing one session message and zero chat messages → the
    chat list is empty, the session list has exactly one SessionInbound with
    raw bytes intact (no Msg.from_bytes)."""
    pools, fake = pools_with_one_p2p
    fake._batch = [{
        "from": "12D3KooWRemote",
        "data": _b64(b"\xde\xad\xbe\xef-session-payload-not-a-Msg"),
        "protocol": "/unaiverse/session/hello/1.0.0",
    }]
    chat, session = _run(pools.get_messages(p2p_name=NodeConn.P2P_PUBLIC))
    assert chat == []
    assert len(session) == 1
    assert isinstance(session[0], SessionInbound)
    assert session[0].peer_id == "12D3KooWRemote"
    assert session[0].protocol_id == "/unaiverse/session/hello/1.0.0"
    assert session[0].data == b"\xde\xad\xbe\xef-session-payload-not-a-Msg"


def test_chat_message_still_uses_legacy_path(pools_with_one_p2p, monkeypatch):
    """A batch with an empty ``protocol`` field must go through the chat path
    (Msg.from_bytes + JWT verify). We mock both to confirm they get called and
    the message lands in the chat list."""
    from unaiverse.networking.p2p import messages as msg_mod
    pools, fake = pools_with_one_p2p

    # Mock Msg.from_bytes to return an object that looks like a Msg with a
    # matching sender and a piggyback we can pretend-verify.
    fake_msg = MagicMock()
    fake_msg.content_type = "misc"
    fake_msg.channel = "ch"
    fake_msg.sender = "12D3KooWChat"
    fake_msg.piggyback = "fake-jwt0"
    monkeypatch.setattr(msg_mod.Msg, "from_bytes", staticmethod(lambda _b: fake_msg))

    # Pretend the sender is in our pool so it passes the gate.
    pools.peer_id_to_pool_name["12D3KooWChat"] = NodeConn.P2P_PUBLIC

    # Mock verify_token to accept the JWT.
    async def _verify(token, sender):
        return ("node-xyz", None)
    monkeypatch.setattr(pools, "verify_token", _verify)

    fake._batch = [{
        "from": "12D3KooWChat",
        "data": _b64(b"any-bytes"),
        "protocol": "",   # chat path
    }]
    chat, session = _run(pools.get_messages(p2p_name=NodeConn.P2P_PUBLIC))
    assert session == []
    assert len(chat) == 1
    assert chat[0] is fake_msg
    # Piggyback rewritten to node_id + inspector_bit, as today.
    assert chat[0].piggyback.startswith("node-xyz")


def test_mixed_batch_split_correctly(pools_with_one_p2p, monkeypatch):
    """Batch with both chat and session messages → each lands in its list."""
    from unaiverse.networking.p2p import messages as msg_mod
    pools, fake = pools_with_one_p2p

    fake_msg = MagicMock()
    fake_msg.content_type = "misc"
    fake_msg.channel = "ch"
    fake_msg.sender = "12D3KooWChat"
    fake_msg.piggyback = "fake-jwt0"
    monkeypatch.setattr(msg_mod.Msg, "from_bytes", staticmethod(lambda _b: fake_msg))
    pools.peer_id_to_pool_name["12D3KooWChat"] = NodeConn.P2P_PUBLIC

    async def _verify(token, sender):
        return ("node-xyz", None)
    monkeypatch.setattr(pools, "verify_token", _verify)

    fake._batch = [
        {"from": "12D3KooWChat", "data": _b64(b"x"), "protocol": ""},
        {"from": "12D3KooWRemote", "data": _b64(b"sess"),
         "protocol": "/unaiverse/session/profile/1.0.0"},
        {"from": "12D3KooWRemote", "data": _b64(b"goaway"),
         "protocol": "/unaiverse/session/control/1.0.0"},
    ]
    chat, session = _run(pools.get_messages(p2p_name=NodeConn.P2P_PUBLIC))
    assert len(chat) == 1
    assert len(session) == 2
    assert {s.protocol_id for s in session} == {
        "/unaiverse/session/profile/1.0.0",
        "/unaiverse/session/control/1.0.0",
    }


# ── send_session() ──────────────────────────────────────────────────────────

def test_send_session_rejects_non_session_protocol(pools_with_one_p2p):
    pools, _ = pools_with_one_p2p
    p2p = pools.p2p_name_to_p2p[NodeConn.P2P_PUBLIC]
    with pytest.raises(ValueError, match="not a Session protocol"):
        _run(pools.send_session(
            peer_id="12D3KooWRemote",
            protocol_id="/unaiverse/chat/1.0.0",   # not a session protocol
            payload=b"x", p2p=p2p,
        ))


def test_session_protocol_ids_match_constants():
    """connpool's SESSION_PROTOCOL_IDS must match session.py's."""
    from unaiverse.networking.session import SESSION_PROTOCOL_IDS as session_ids
    assert SESSION_PROTOCOL_IDS == session_ids


def test_session_inbound_namedtuple_shape():
    si = SessionInbound(peer_id="P", protocol_id="/x", data=b"d")
    assert si.peer_id == "P"
    assert si.protocol_id == "/x"
    assert si.data == b"d"
    # Positional access (it's a namedtuple).
    assert si[0] == "P" and si[1] == "/x" and si[2] == b"d"
