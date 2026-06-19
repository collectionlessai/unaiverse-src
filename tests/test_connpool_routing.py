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


# ── Membership snapshot (perimeter seam) ─────────────────────────────────────

def test_membership_snapshot_is_immutable():
    """membership_snapshot freezes the live sets so a snapshot already handed to the
    routing cannot change if the membership mutates afterwards."""
    from unaiverse.networking.node.connpool import WorldMembership

    conn = NodeConn.__new__(NodeConn)  # bypass the heavy P2P-bound __init__
    conn.world_agents_set = {"a"}
    conn.world_masters_set = {"m"}
    conn.world_node_peer_id = "wn"
    conn.inspector_peer_id = "insp"
    snap = conn.membership_snapshot()
    assert isinstance(snap, WorldMembership)
    assert snap.world_agents == frozenset({"a"})
    assert snap.world_masters == frozenset({"m"})
    assert snap.world_node_peer_id == "wn" and snap.inspector_peer_id == "insp"
    # Mutating the live set afterwards must not touch the already-taken snapshot.
    conn.world_agents_set.add("b")
    assert "b" not in snap.world_agents


def test_conn_routing_classifies_against_passed_snapshot():
    """conn_routing_fcn classifies world peers against the *passed* membership
    snapshot, not the Coordinator's live sets. With empty live sets, a peer the
    snapshot calls a world agent is still routed (proving the seam is the snapshot)."""
    from unaiverse.networking.node.connpool import WorldMembership

    class _WP2P:
        log_sub = "world"
    world_p2p = _WP2P()

    conn = NodeConn.__new__(NodeConn)  # bypass the heavy P2P-bound __init__
    conn.p2p_public = object()
    conn.p2p_to_pool_names = {world_p2p: [
        NodeConn.IN_WORLD_AGENTS, NodeConn.OUT_WORLD_AGENTS,
        NodeConn.IN_WORLD_NODE, NodeConn.OUT_WORLD_NODE,
        NodeConn.IN_WORLD_MASTERS, NodeConn.OUT_WORLD_MASTERS,
    ]}
    # Live sets are EMPTY: reading self.* would make the peer unclassifiable.
    conn.world_agents_set = set()
    conn.world_masters_set = set()
    conn.world_node_peer_id = None
    conn.inspector_peer_id = None

    membership = WorldMembership(world_agents=frozenset({"agentX"}), world_masters=frozenset(),
                                world_node_peer_id=None, inspector_peer_id=None)
    infos = [{"id": "agentX", "direction": "inbound", "addrs": [], "connected_at": 0}]
    routed = _run(conn.conn_routing_fcn(infos, world_p2p, membership))
    assert "agentX" in routed[NodeConn.IN_WORLD_AGENTS]


# ── Characterization oracle for update()/conn_routing_fcn (U6 baseline) ───────
#
# These freeze the exact contract of the per-cycle diff and the world perimeter so
# the U6 refactor (per-instance reconcile in the SessionManagers) can be checked
# bit-for-bit against them. They drive the real ConnectionPools.update() through fake
# P2Ps (the NodeConn constructor never calls P2P methods, so fakes are enough).

class _UpdP2P:
    """Fake P2P for driving update(): serves planted connected-peer infos and records
    perimeter disconnects (conn_routing_fcn -> ConnectionPools.disconnect)."""
    def __init__(self, log_sub):
        self.log_sub = log_sub
        self.peer_id = log_sub + "-self"
        self._infos = []
        self.disconnected = []

    def get_connected_peers_info(self):
        return list(self._infos)

    def disconnect_from(self, peer_id):
        self.disconnected.append(peer_id)
        return True


def _info(peer_id, direction):
    return {"id": peer_id, "direction": direction, "addrs": [], "connected_at": 0}


def _make_node_conn():
    pub, world = _UpdP2P("pub"), _UpdP2P("world")
    conn = NodeConn(max_connections=16, p2p_u=pub, p2p_w=world,
                    is_world_node=False, public_key=None, token=None, local_node_id="n")
    return conn, pub, world


def test_char_public_peer_routes_to_in_out_public():
    """A public inbound peer lands in IN_PUBLIC, an outbound one in OUT_PUBLIC."""
    conn, pub, world = _make_node_conn()
    pub._infos = [_info("pin", "inbound"), _info("pout", "outbound")]
    added, removed = _run(conn.update())
    assert added[NodeConn.IN_PUBLIC] == {"pin"}
    assert added[NodeConn.OUT_PUBLIC] == {"pout"}


def test_char_world_classification_precedence_agent_over_node():
    """A world peer that is at once agent, node and inspector is classified as agent
    (the ternary precedence agent > node > master): inbound -> IN_WORLD_AGENTS."""
    conn, pub, world = _make_node_conn()
    conn.world_agents_set = {"w"}
    conn.world_node_peer_id = "w"
    conn.inspector_peer_id = "w"
    world._infos = [_info("w", "inbound")]
    added, removed = _run(conn.update())
    assert added.get(NodeConn.IN_WORLD_AGENTS) == {"w"}
    assert "w" not in added.get(NodeConn.IN_WORLD_NODE, set())


def test_char_world_perimeter_cuts_unclassifiable():
    """An unknown world peer (not in the membership) is disconnected and never enters
    a pool nor the diff. This is the security perimeter."""
    conn, pub, world = _make_node_conn()
    world._infos = [_info("ghost", "inbound")]
    added, removed = _run(conn.update())
    assert "ghost" in world.disconnected
    assert all("ghost" not in s for s in added.values())
    assert "ghost" not in conn.peer_id_to_pool_name


def test_char_full_pool_truncates_diff_added():
    """When more peers are offered than the pool's max size, only max-size of them are
    admitted: the overflow is rejected by add() and never enters the diff added."""
    conn, pub, world = _make_node_conn()
    cap = conn.pool_name_to_pool_triple[NodeConn.IN_WORLD_AGENTS][1]
    names = [f"a{i}" for i in range(cap + 2)]
    conn.world_agents_set = set(names)
    world._infos = [_info(n, "inbound") for n in names]
    added, removed = _run(conn.update())
    assert len(added[NodeConn.IN_WORLD_AGENTS]) == cap


def test_char_same_peer_both_networks_enters_once():
    """The same peer-id seen on both networks enters exactly one pool: public is
    processed first, and the world add() is refused by the cross-pool guard."""
    conn, pub, world = _make_node_conn()
    conn.world_agents_set = {"dual"}
    pub._infos = [_info("dual", "inbound")]
    world._infos = [_info("dual", "inbound")]
    added, removed = _run(conn.update())
    assert added[NodeConn.IN_PUBLIC] == {"dual"}
    assert "dual" not in added.get(NodeConn.IN_WORLD_AGENTS, set())


def test_char_unchanged_pool_absent_from_diff():
    """A pool with no change in a cycle is not even a key of the diff (setdefault only
    creates the key on a successful add/remove)."""
    conn, pub, world = _make_node_conn()
    pub._infos = [_info("pin", "inbound")]
    _run(conn.update())
    added2, removed2 = _run(conn.update())
    assert NodeConn.IN_PUBLIC not in added2
    assert NodeConn.IN_PUBLIC not in removed2
