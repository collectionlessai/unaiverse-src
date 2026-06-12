"""Tests for the connpool ``get_messages`` protocol partition + SessMsg codec.

Confirms that:
  * Session-protocol messages skip ``Msg.from_bytes`` and JWT verification and are
    decoded into ``SessMsg`` value objects by the codec.
  * Non-session messages still go through the legacy chat path.
  * ``send_session`` refuses non-session protocol IDs.
  * The SessMsg ↔ protobuf codec round-trips.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from unaiverse.networking.node.connpool import (
    ConnectionPools, NodeConn, SessionInbound, SESSION_PROTOCOL_IDS,
)
from unaiverse.networking.session import SessMsg


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


def _run(coro):
    return asyncio.run(coro)


# ── Codec round-trip ─────────────────────────────────────────────────────────

def test_codec_round_trips_hello_request():
    msg = SessMsg(SessMsg.KIND_HELLO_REQ, {
        "session_id": b"\x01" * 16, "nonce_a": b"\xaa" * 32,
        "protocol_versions": ["1.0"], "capabilities": 0, "client_id": "py/1.0",
    })
    wire = SessMsg.to_bytes(msg)
    back = SessMsg.from_bytes(wire, "/unaiverse/session/hello/1.0.0")
    assert back.kind == SessMsg.KIND_HELLO_REQ
    assert back.fields["session_id"] == b"\x01" * 16
    assert back.fields["nonce_a"] == b"\xaa" * 32
    assert list(back.fields["protocol_versions"]) == ["1.0"]
    assert back.fields["client_id"] == "py/1.0"


def test_codec_round_trips_world_grant():
    msg = SessMsg(SessMsg.KIND_WORLD_GRANT, {
        "grant_id": b"\x05" * 32, "grant_sig": b"sig", "expiry_unix_ms": 1234567890123,
        "world_profile": b'{"a":1}', "world_private_peer_record": b"spr",
        "role": 3, "code_bundle": b"", "initial_stats": b"", "rendezvous_tag": 7,
    })
    back = SessMsg.from_bytes(SessMsg.to_bytes(msg), "/unaiverse/session/world/1.0.0")
    assert back.kind == SessMsg.KIND_WORLD_GRANT
    assert back.fields["grant_id"] == b"\x05" * 32
    assert back.fields["expiry_unix_ms"] == 1234567890123
    assert back.fields["role"] == 3
    assert back.fields["rendezvous_tag"] == 7


# ── Routing ─────────────────────────────────────────────────────────────────

def test_session_protocol_messages_are_partitioned(pools_with_one_p2p):
    """A batch containing one session message and zero chat messages → the
    chat list is empty, the session list has exactly one SessionInbound carrying
    a decoded SessMsg (no Msg.from_bytes / JWT verify)."""
    pools, fake = pools_with_one_p2p
    wire = SessMsg.to_bytes(SessMsg(SessMsg.KIND_HELLO_REQ, {
        "session_id": b"\x01" * 16, "nonce_a": b"\xaa" * 32, "protocol_versions": ["1.0"],
    }))
    fake._batch = [{
        "from": "12D3KooWRemote",
        "data": wire,
        "protocol": "/unaiverse/session/hello/1.0.0",
    }]
    chat, session = _run(pools.get_messages(p2p_name=NodeConn.P2P_PUBLIC))
    assert chat == []
    assert len(session) == 1
    assert isinstance(session[0], SessionInbound)
    assert session[0].peer_id == "12D3KooWRemote"
    assert session[0].protocol_id == "/unaiverse/session/hello/1.0.0"
    assert session[0].sess_msg.kind == SessMsg.KIND_HELLO_REQ
    assert session[0].sess_msg.fields["session_id"] == b"\x01" * 16


def test_chat_message_still_uses_legacy_path(pools_with_one_p2p, monkeypatch):
    """A batch with an empty ``protocol`` field must go through the chat path
    (Msg.from_bytes + JWT verify)."""
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

    fake._batch = [{
        "from": "12D3KooWChat",
        "data": b"any-bytes",
        "protocol": "",   # chat path
    }]
    chat, session = _run(pools.get_messages(p2p_name=NodeConn.P2P_PUBLIC))
    assert session == []
    assert len(chat) == 1
    assert chat[0] is fake_msg
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

    offer_wire = SessMsg.to_bytes(SessMsg(SessMsg.KIND_PROFILE_OFFER, {"sig_a": b"s"}))
    goaway_wire = SessMsg.to_bytes(SessMsg(SessMsg.KIND_GOAWAY, {"reason": 9}))
    fake._batch = [
        {"from": "12D3KooWChat", "data": b"x", "protocol": ""},
        {"from": "12D3KooWRemote", "data": offer_wire,
         "protocol": "/unaiverse/session/profile/1.0.0"},
        {"from": "12D3KooWRemote", "data": goaway_wire,
         "protocol": "/unaiverse/session/control/1.0.0"},
    ]
    chat, session = _run(pools.get_messages(p2p_name=NodeConn.P2P_PUBLIC))
    assert len(chat) == 1
    assert len(session) == 2
    assert {s.protocol_id for s in session} == {
        "/unaiverse/session/profile/1.0.0",
        "/unaiverse/session/control/1.0.0",
    }
    assert {s.sess_msg.kind for s in session} == {
        SessMsg.KIND_PROFILE_OFFER, SessMsg.KIND_GOAWAY,
    }


# ── send_session() ──────────────────────────────────────────────────────────

def test_send_session_rejects_non_session_protocol(pools_with_one_p2p):
    pools, _ = pools_with_one_p2p
    with pytest.raises(ValueError, match="not a Session protocol"):
        _run(pools.send_session(
            peer_id="12D3KooWRemote",
            protocol_id="/unaiverse/chat/1.0.0",   # not a session protocol
            sess_msg=SessMsg(SessMsg.KIND_GOAWAY, {}),
            p2p_name=NodeConn.P2P_PUBLIC,
        ))


def test_session_protocol_ids_match_constants():
    """connpool's SESSION_PROTOCOL_IDS must match session.py's."""
    from unaiverse.networking.session import SESSION_PROTOCOL_IDS as session_ids
    assert SESSION_PROTOCOL_IDS == session_ids


def test_session_inbound_namedtuple_shape():
    si = SessionInbound(peer_id="P", protocol_id="/x", sess_msg="d")
    assert si.peer_id == "P"
    assert si.protocol_id == "/x"
    assert si.sess_msg == "d"
    assert si[0] == "P" and si[1] == "/x" and si[2] == "d"


def test_codec_roundtrips_liveness_control_messages():
    """The JSON-envelope codec for PING/PONG/RESUME/RESUME_ACK must round-trip,
    restoring bytes fields to bytes and preserving bools/ints."""
    from unaiverse.networking.session.sess_msg import SESSION_UNKNOWN
    cases = [
        SessMsg(SessMsg.KIND_PING, {"session_id": b"\x01" * 16, "nonce": b"\xaa\xbb"}),
        SessMsg(SessMsg.KIND_PONG, {"session_id": b"\x02" * 16, "nonce": b"\xcc", "ok": True}),
        SessMsg(SessMsg.KIND_PONG, {"session_id": b"\x02" * 16, "nonce": b"\xcc", "ok": False}),
        SessMsg(SessMsg.KIND_RESUME, {"session_id": b"\x03" * 16, "nonce": b"\xdd"}),
        SessMsg(SessMsg.KIND_RESUME_ACK, {"session_id": b"\x04" * 16, "nonce": b"\xee",
                                          "accepted": False, "reason_code": SESSION_UNKNOWN}),
    ]
    for msg in cases:
        wire = SessMsg.to_bytes(msg)
        back = SessMsg.from_bytes(wire, "/unaiverse/session/control/1.0.0")
        assert back.kind == msg.kind
        for k, v in msg.fields.items():
            assert back.fields[k] == v, (msg.kind, k, back.fields[k], v)
        # bytes fields must come back as bytes, not hex strings
        assert isinstance(back.fields["session_id"], bytes)


def test_is_peer_session_pending():
    """NodeConn.is_peer_session_pending is True only for a Session that exists and
    is neither ESTABLISHED nor terminal."""
    from unaiverse.networking.session import SessionState

    class _FakeSess:
        def __init__(self, state):
            self.state = state

    class _FakeSM:
        def __init__(self, sessions):
            self._s = sessions

        def get_session(self, pid):
            return self._s.get(pid)

    conn = NodeConn.__new__(NodeConn)  # bypass the heavy P2P-bound __init__
    conn.set_session_managers(_FakeSM({
        "pending": _FakeSess(SessionState.HELLO_SENT),
        "reconnecting": _FakeSess(SessionState.RECONNECTING),
        "established": _FakeSess(SessionState.ESTABLISHED),
        "failed": _FakeSess(SessionState.FAILED),
        "closed": _FakeSess(SessionState.CLOSED),
    }))
    assert conn.is_peer_session_pending("pending") is True
    # RECONNECTING was already established once → not "waiting to be managed".
    assert conn.is_peer_session_pending("reconnecting") is False
    assert conn.is_peer_session_pending("established") is False
    assert conn.is_peer_session_pending("failed") is False
    assert conn.is_peer_session_pending("closed") is False
    assert conn.is_peer_session_pending("unknown") is False
