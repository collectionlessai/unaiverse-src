"""Tests for the connpool ``get_messages`` protocol partition + SessMsg codec.

Confirms that:
  * Session-protocol messages skip ``Msg.from_bytes`` and JWT verification and are
    decoded into ``SessMsg`` value objects by the codec.
  * Non-session messages still go through the legacy chat path.
  * The SessMsg ↔ protobuf codec round-trips.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from unaiverse.networking.node.connpool import (
    NodeConn, SessionInbound,
)
from unaiverse.networking.session import SessMsg
from unaiverse.networking.session.sess_msg import PERIMETER_CUT
from unaiverse.networking.session.session import SessionState
from unaiverse.networking.p2p.p2p import P2P

pytestmark = pytest.mark.unit


class _FakeP2P:
    """Minimal stand-in: fakes pop_messages and reuses the real P2P.get_messages, so the
    session/chat partition + SessMsg codec exercised here is the production one."""
    log_sub = "fake"

    def __init__(self, batch):
        self._batch = batch

    def pop_messages(self):
        return self._batch

    get_messages = P2P.get_messages


@pytest.fixture
def pools_with_one_p2p():
    """A NodeConn whose public P2P returns whatever messages we plant. The world P2P is
    a second fake we never feed: get_messages is only exercised on the public one. The
    constructor never calls P2P methods, so the bare fakes are enough. get_messages resolves
    the instance through the SM (S18), so we register minimal SM stubs that just carry the
    p2p_name and passthrough get_messages/log_sub to their P2P handle (R1a: the Coordinator
    reaches the transport only via the SM, never directly)."""
    from types import SimpleNamespace
    fake_p2p = _FakeP2P([])
    world_p2p = _FakeP2P([])
    pools = NodeConn(
        max_connections=16,
        is_world_node=False,
        public_key=None,
        token=None,
    )
    pools.set_session_managers(
        SimpleNamespace(p2p=fake_p2p, p2p_name=NodeConn.P2P_PUBLIC,
                        get_messages=fake_p2p.get_messages, log_sub=fake_p2p.log_sub),
        SimpleNamespace(p2p=world_p2p, p2p_name=NodeConn.P2P_WORLD,
                        get_messages=world_p2p.get_messages, log_sub=world_p2p.log_sub),
    )
    return pools, fake_p2p


def _run(coro):
    return asyncio.run(coro)


# ── Codec round-trip ─────────────────────────────────────────────────────────

def test_codec_round_trips_hello_request():
    """A HELLO_REQ SessMsg survives to_bytes/from_bytes over the hello protocol id,
    preserving kind, the bytes fields session_id and nonce_a, the protocol_versions
    list and the client_id string."""
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
    """A WORLD_GRANT SessMsg survives to_bytes/from_bytes over the world protocol id,
    preserving kind, the grant_id bytes, the large expiry_unix_ms integer, and the
    role and rendezvous_tag integers."""
    msg = SessMsg(SessMsg.KIND_WORLD_GRANT, {
        "grant_id": b"\x05" * 32, "expiry_unix_ms": 1234567890123,
        "world_profile": b'{"a":1}',
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


def test_session_inbound_namedtuple_shape():
    """SessionInbound is a namedtuple whose peer_id, protocol_id and sess_msg fields are
    reachable both by name and by positional index 0/1/2."""
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
    is neither ESTABLISHED nor terminal. The Coordinator now aggregates the per-state
    classification from SessionManager.is_handshake_in_flight, so the fake manager exposes
    that predicate (mirroring the real SessionManager) instead of a bare get_session."""
    from unaiverse.networking.session import SessionState

    class _FakeSess:
        def __init__(self, state):
            self.state = state

    class _FakeSM:
        def __init__(self, sessions):
            self._s = sessions

        def get_session(self, pid):
            return self._s.get(pid)

        def is_handshake_in_flight(self, pid):
            sess = self._s.get(pid)
            if sess is None:
                return False
            return sess.state not in (SessionState.KNOWN, SessionState.ESTABLISHED,
                                      SessionState.RECONNECTING, SessionState.FAILED,
                                      SessionState.CLOSED)

    conn = NodeConn.__new__(NodeConn)  # bypass the heavy P2P-bound __init__
    conn.set_session_managers(_FakeSM({
        "pending": _FakeSess(SessionState.HELLO_SENT),
        "known": _FakeSess(SessionState.KNOWN),
        "reconnecting": _FakeSess(SessionState.RECONNECTING),
        "established": _FakeSess(SessionState.ESTABLISHED),
        "failed": _FakeSess(SessionState.FAILED),
        "closed": _FakeSess(SessionState.CLOSED),
    }))
    assert conn.is_peer_session_pending("pending") is True
    # KNOWN is a known-but-not-dialed peer: no handshake in flight → not pending.
    assert conn.is_peer_session_pending("known") is False
    # RECONNECTING was already established once → not "waiting to be managed".
    assert conn.is_peer_session_pending("reconnecting") is False
    assert conn.is_peer_session_pending("established") is False
    assert conn.is_peer_session_pending("failed") is False
    assert conn.is_peer_session_pending("closed") is False
    assert conn.is_peer_session_pending("unknown") is False


def test_requires_fast_clock_aggregates_managers():
    """NodeConn.requires_fast_clock is True iff any registered SessionManager reports a
    fast-clock Session; the Coordinator just ORs the per-instance predicate."""
    class _FakeSM:
        def __init__(self, fast):
            self._fast = fast

        def requires_fast_clock(self):
            return self._fast

    conn = NodeConn.__new__(NodeConn)  # bypass the heavy P2P-bound __init__
    conn.set_session_managers(_FakeSM(False), _FakeSM(False))
    assert conn.requires_fast_clock() is False
    conn.set_session_managers(_FakeSM(False), _FakeSM(True))
    assert conn.requires_fast_clock() is True


# ── Membership snapshot (perimeter seam) ─────────────────────────────────────

def test_membership_snapshot_is_immutable():
    """membership_snapshot freezes sm_world's live sets so a snapshot already handed to
    the routing cannot change if the membership mutates afterwards. The pure membership
    lives in sm_world now; inspector_peer_id stays the Coordinator's own field (D-8a)."""
    from types import SimpleNamespace
    from unaiverse.networking.node.connpool import WorldMembership

    conn = NodeConn.__new__(NodeConn)  # bypass the heavy P2P-bound __init__
    conn._sm_world = SimpleNamespace(
        world_agents_set={"a"}, world_masters_set={"m"}, world_node_peer_id="wn")
    conn.inspector_peer_id = "insp"
    snap = conn.membership_snapshot()
    assert isinstance(snap, WorldMembership)
    assert snap.world_agents == frozenset({"a"})
    assert snap.world_masters == frozenset({"m"})
    assert snap.world_node_peer_id == "wn" and snap.inspector_peer_id == "insp"
    # Mutating the live set afterwards must not touch the already-taken snapshot.
    conn._sm_world.world_agents_set.add("b")
    assert "b" not in snap.world_agents


def test_route_world_classifies_against_passed_snapshot():
    """The world routing classifier keys purely off the membership snapshot it is given:
    the Coordinator holds no live roster (it lives in sm_world), so a peer the snapshot
    calls a world agent routes to IN_WORLD_AGENTS, and a peer the snapshot does not know is
    cut (None). This is the classifier the production path runs through sm.reconcile; the
    perimeter cut and the agent>node>master precedence are exercised end-to-end by the
    char-tests via update()."""
    from unaiverse.networking.node.connpool import WorldMembership

    membership = WorldMembership(world_agents=frozenset({"agentX"}), world_masters=frozenset(),
                                world_node_peer_id=None, inspector_peer_id=None)
    agent_in = {"id": "agentX", "direction": "inbound", "addrs": [], "connected_at": 0}
    ghost_in = {"id": "ghost", "direction": "inbound", "addrs": [], "connected_at": 0}
    assert NodeConn._route_world(agent_in, membership) == NodeConn.IN_WORLD_AGENTS
    assert NodeConn._route_world(ghost_in, membership) is None


# ── Characterization oracle for update() ──────────────────────────────────────
#
# These freeze the exact contract of the per-cycle diff and the world perimeter so
# the per-instance reconcile in the SessionManagers can be checked bit-for-bit
# against them. They drive the real NodeConn.update() through fake P2Ps
# (the NodeConn constructor never calls P2P methods, so fakes are enough).

class _UpdP2P:
    """Fake P2P for driving update(): serves planted connected-peer infos and records
    perimeter disconnects (the world reconcile -> sm.drop -> p2p.disconnect)."""
    def __init__(self, log_sub):
        self.log_sub = log_sub
        self.peer_id = log_sub + "-self"
        self._infos = []
        self._topic_peers = []  # pubsub topic subscribers; plant to drive the cut-sweep
        self.disconnected = []

    def get_connected_peers_info(self):
        # None models an FFI error (transport state unknown); a list is the real state.
        return None if self._infos is None else list(self._infos)

    def get_topic_peers(self):
        return list(self._topic_peers)

    async def disconnect(self, peer_id):
        self.disconnected.append(peer_id)
        return True


def _info(peer_id, direction):
    return {"id": peer_id, "direction": direction, "addrs": [], "connected_at": 0}


def _make_node_conn():
    pub, world = _UpdP2P("pub"), _UpdP2P("world")
    conn = NodeConn(max_connections=16, is_world_node=False, public_key=None, token=None,
                    local_node_id="n")
    # Build the per-instance SessionManagers (update() routes through sm.reconcile); the
    # transports are handed to attach_session_policy now, not to the constructor.
    conn.attach_session_policy(p2p_u=pub, p2p_w=world, local_profile_factory=lambda: {}, decision_hook=None)
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
    conn._sm_world.world_agents_set = {"w"}
    conn._sm_world.world_node_peer_id = "w"
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


def test_char_world_perimeter_cuts_pubsub_only_nonmember():
    """A peer whose only presence is a world topic subscription (it never opened a stream, so
    it is absent from connected_peers_info and the stream loop is blind to it) is still cut if
    it is not a world member. A subscribed member is kept. This closes the exfiltration channel
    the private instance exists to protect."""
    conn, pub, world = _make_node_conn()
    conn._sm_world.world_agents_set = {"member"}
    world._infos = []                             # no stream: invisible to the stream-based loop
    world._topic_peers = ["intruder", "member"]   # both subscribe to a world topic
    _run(conn.update())
    assert "intruder" in world.disconnected       # non-member subscriber is cut
    assert "member" not in world.disconnected     # a subscribed member is kept


class _CutStubSession:
    """An ESTABLISHED-enough Session that records close(): planted into the world SM's
    registry so the perimeter cut's GOAWAY can be asserted without a full FSM. The state
    keeps sync_known_peers from pruning it as a stale KNOWN entry."""
    def __init__(self):
        self.closed_with = None
        self.state = SessionState.ESTABLISHED

    async def close(self, reason_code):
        self.closed_with = reason_code


def test_char_world_perimeter_cut_closes_session_not_just_transport():
    """A world peer that falls out of membership but still holds a Session is not merely
    disconnected: its Session is GOAWAY-closed with PERIMETER_CUT so it will never re-dial
    (closing the cut -> redial -> RESUME -> cut loop), on top of the transport drop."""
    conn, pub, world = _make_node_conn()
    conn._sm_world.world_agents_set = {"member"}   # 'ghost' is NOT a member
    stub = _CutStubSession()
    conn._sm_world._sessions["ghost"] = stub
    world._infos = [_info("ghost", "inbound")]
    _run(conn.update())
    assert stub.closed_with == PERIMETER_CUT       # GOAWAY closed, not left ESTABLISHED
    assert "ghost" in world.disconnected           # transport severed too
    assert "ghost" not in conn.peer_id_to_pool_name


def test_char_grace_companion_slips_leave_sweep_but_is_cut_on_reconcile():
    """A companion riding its reconnect grace at leave-time was already evicted from the
    pool, so remove_all_world_agents (which sweeps the WORLD pools) cannot see it and does
    NOT close it — the known leak. The backstop is the perimeter cut: once membership is
    cleared, the next reconcile classifies the still-connected companion as a non-member
    and GOAWAY-closes it. This is why B2 (leave closes companions) and B6 (perimeter closes
    on reconcile) must ship in the same commit."""
    conn, pub, world = _make_node_conn()
    stub = _CutStubSession()
    conn._sm_world._sessions["companion"] = stub  # live Session, but NOT in any pool (grace-evicted)

    # The leave sweep iterates the WORLD pools, so it never reaches the pool-less companion.
    _run(conn.remove_all_world_agents())
    assert stub.closed_with is None, "expected the grace companion to slip the leave sweep"

    # Backstop: membership cleared and the companion still connected -> reconcile cuts it.
    conn._sm_world.world_agents_set = set()
    world._infos = [_info("companion", "inbound")]
    _run(conn.update())
    assert stub.closed_with == PERIMETER_CUT, "the perimeter cut did not close the leaked companion"
    assert "companion" in world.disconnected


def test_char_world_perimeter_topic_cut_ignores_self():
    """The instance's own peer-id, should it ever surface among topic peers, is never cut."""
    conn, pub, world = _make_node_conn()
    world._topic_peers = [world.peer_id]
    _run(conn.update())
    assert world.peer_id not in world.disconnected


def test_char_world_perimeter_topic_cut_skips_stream_peers():
    """A peer already handled by the stream loop (present in connected_peers_info) is not
    re-processed by the topic sweep: a member with a stream is admitted, not double-cut."""
    conn, pub, world = _make_node_conn()
    conn._sm_world.world_agents_set = {"member"}
    world._infos = [_info("member", "inbound")]   # member with a real stream
    world._topic_peers = ["member"]               # also a topic subscriber
    _run(conn.update())
    assert "member" not in world.disconnected
    assert conn.is_connected("member")


def test_char_public_perimeter_never_cuts_topic_peers():
    """The public instance has no membership and must never cut topic peers (it is the open
    network); only the world instance enforces the perimeter."""
    conn, pub, world = _make_node_conn()
    pub._topic_peers = ["anyone"]
    _run(conn.update())
    assert "anyone" not in pub.disconnected


def test_char_ffi_error_none_skips_diff_and_keeps_pools():
    """When the transport reports None (an FFI error: transport state UNKNOWN, not "zero
    peers"), the reconcile skips the diff and leaves the pools intact. A transient error
    must NOT empty every pool — for an agent that would fire a spurious leave_world (B28)."""
    conn, pub, world = _make_node_conn()
    conn._sm_world.world_agents_set = {"wa"}
    world._infos = [_info("wa", "inbound")]
    added, _ = _run(conn.update())
    assert added[NodeConn.IN_WORLD_AGENTS] == {"wa"}     # admitted

    # Next cycle the FFI errors (None), which must NOT read as "zero peers".
    world._infos = None
    _, removed = _run(conn.update())
    assert "wa" not in removed.get(NodeConn.IN_WORLD_AGENTS, set())   # not spuriously removed
    assert conn.is_connected("wa")                                    # still pooled

    # A genuine empty list (zero peers), by contrast, DOES remove it.
    world._infos = []
    _, removed2 = _run(conn.update())
    assert "wa" in removed2.get(NodeConn.IN_WORLD_AGENTS, set())


def test_char_full_pool_truncates_diff_added():
    """When more peers are offered than the pool's max size, only max-size of them are
    admitted: the overflow is rejected by add() and never enters the diff added."""
    conn, pub, world = _make_node_conn()
    cap = conn.pool_name_to_pool_triple[NodeConn.IN_WORLD_AGENTS][1]
    names = [f"a{i}" for i in range(cap + 2)]
    conn._sm_world.world_agents_set = set(names)
    world._infos = [_info(n, "inbound") for n in names]
    added, removed = _run(conn.update())
    assert len(added[NodeConn.IN_WORLD_AGENTS]) == cap


def test_char_same_peer_both_networks_enters_once():
    """Defense-in-depth: the cross-pool guard keeps a peer-id out of a second pool.

    The same peer-id appearing on BOTH networks cannot happen in production: a node has
    two DISTINCT peer-ids (public from p2p_u, world from p2p_w), bound by the token's
    p2p_peers, so a remote never classifies into both a public and a world pool. This is
    an anomalous scenario the guard defends against, not a normal path. We simulate it by
    planting "dual" on both fake instances: public is processed first (registration
    order), and the world add() is refused by the cross-pool guard, so the peer ends up
    in exactly one pool."""
    conn, pub, world = _make_node_conn()
    conn._sm_world.world_agents_set = {"dual"}
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


def test_char_removal_from_pool_absent_from_classified():
    """A peer that leaves the network is reported in `removed` even though its pool is then
    absent from the classified dict: _reconcile_instance iterates the instance's OWN pools, so
    a pool that drops to empty still yields its lost peers. update() only REPORTS the loss (it
    does not remove); the actual remove is Node's reaction to the diff. This must keep holding
    once each SM owns its pools and produces its own partial diff (S10)."""
    conn, pub, world = _make_node_conn()
    conn._sm_world.world_agents_set = {"wa"}
    world._infos = [_info("wa", "inbound")]
    added, _ = _run(conn.update())
    assert added[NodeConn.IN_WORLD_AGENTS] == {"wa"}
    # Next cycle: the peer is gone from the transport; classified no longer has IN_WORLD_AGENTS.
    world._infos = []
    _, removed = _run(conn.update())
    assert removed[NodeConn.IN_WORLD_AGENTS] == {"wa"}


def test_char_drop_peer_keeps_misc():
    """drop_peer() drops the peer from its pool and the connection indices but deliberately KEEPS
    peer_id_to_misc (the role taxonomy), so the role survives a disconnect/re-add. This is part
    of I-2 and must hold after the pool state descends into the SM while the role taxonomy stays
    Coordinator-owned (D-2)."""
    conn, pub, world = _make_node_conn()
    # add_to_world_agents_list records the role in peer_id_to_misc (via set_role) and the roster.
    conn.add_to_world_agents_list("wa", ["/ip4/1.1.1.1/tcp/1"], role=ROLE_WORLD_AGENT)
    world._infos = [_info("wa", "inbound")]
    _run(conn.update())
    assert conn.is_connected("wa")
    assert conn.get_role("wa") == ROLE_WORLD_AGENT
    _run(conn.drop_peer("wa"))
    assert "wa" in world.disconnected
    assert not conn.is_connected("wa")
    # The pool / connection indices dropped it, but the role taxonomy is intentionally kept.
    assert conn.get_role("wa") == ROLE_WORLD_AGENT


def test_sm_of_prefers_session_then_index():
    """_sm_of resolves the peer's instance (the SessionManager, the SM is the instance)
    from the manager that holds its Session (public before world), falling back to the
    connection index for peers that are connected but have no Session yet, and None when
    nothing knows the peer."""
    conn, pub, world = _make_node_conn()
    # A Session on the world manager resolves to the world instance.
    conn._sm_world._sessions["w1"] = object()
    assert conn._sm_of("w1") is conn._sm_world
    # A Session on the public manager resolves to the public instance.
    conn._sm_public._sessions["p1"] = object()
    assert conn._sm_of("p1") is conn._sm_public
    # No Session: fall back to the pool a connected peer lives in (no Session yet).
    conn.peer_id_to_pool_name["idx"] = NodeConn.IN_WORLD_AGENTS
    assert conn._sm_of("idx") is conn._sm_world
    # Unknown to both Sessions and pools.
    assert conn._sm_of("nope") is None


def test_cycle_snapshot_frozen_until_next_update():
    """update() freezes the world membership for the whole cycle. A roster mutation
    written after update() (a mid-cycle pre-admission) is invisible to _cycle_membership
    until the next update() takes a fresh snapshot, so the perimeter and the get_messages
    gate stay in lockstep (I-1)."""
    conn, pub, world = _make_node_conn()
    _run(conn.update())
    assert conn._cycle_membership.world_agents == frozenset()

    # Mid-cycle pre-admission mutates the live roster (now in sm_world)...
    conn.add_to_world_agents_list("late", ["/ip4/9.9.9.9/tcp/9"], role=ROLE_WORLD_AGENT)
    assert "late" in conn._sm_world.world_agents_set
    # ...but the per-cycle snapshot stays frozen until the next update().
    assert "late" not in conn._cycle_membership.world_agents

    _run(conn.update())
    assert "late" in conn._cycle_membership.world_agents


def test_get_messages_gate_reads_cycle_snapshot(pools_with_one_p2p, monkeypatch):
    """get_messages admits a not-yet-connected chat sender from the per-cycle snapshot,
    not the live mutable roster: a sender present only in the live set (pre-admitted
    mid-cycle) is dropped until the next update() refreshes the snapshot."""
    from unaiverse.networking.node.connpool import WorldMembership
    from unaiverse.networking.p2p import messages as msg_mod
    pools, fake = pools_with_one_p2p

    fake_msg = MagicMock()
    fake_msg.content_type = "misc"
    fake_msg.channel = "ch"
    fake_msg.sender = "lateagent"
    fake_msg.piggyback = "jwt0"
    monkeypatch.setattr(msg_mod.Msg, "from_bytes", staticmethod(lambda _b: fake_msg))

    async def _verify(token, sender):
        return ("node-late", None)
    monkeypatch.setattr(pools, "verify_token", _verify)

    # This cycle's frozen snapshot does not include "lateagent", so the gate drops the
    # message. The live roster (now in sm_world) is irrelevant here: since S4 the gate
    # reads only the per-cycle snapshot, never the live set.
    pools._cycle_membership = WorldMembership(frozenset(), frozenset(), None, None)
    fake._batch = [{"from": "lateagent", "data": b"x", "protocol": ""}]
    chat, _ = _run(pools.get_messages(p2p_name=NodeConn.P2P_PUBLIC))
    assert chat == []

    # Once the next snapshot includes it, the same sender is admitted.
    pools._cycle_membership = WorldMembership(frozenset({"lateagent"}), frozenset(), None, None)
    fake._batch = [{"from": "lateagent", "data": b"x", "protocol": ""}]
    chat, _ = _run(pools.get_messages(p2p_name=NodeConn.P2P_PUBLIC))
    assert len(chat) == 1


# ── Characterization oracle for the facade contract ────────────────────────────
#
# These freeze the role/addrs/is_connected facade (KNOWN Session per rendezvous
# peer, role/addrs sourced from the Sessions) bit-for-bit. They drive the real
# rendezvous writers (set_world_*_list / add_to_world_*_list / set_role) and read
# back through the public facade, so they stay valid across the data-source
# migration: writer and reader move together. The load-bearing invariant is that
# "known" is not "connected": a peer known from the rendezvous exposes role/addrs
# but stays out of is_connected and the connected-peer infos until the transport
# actually admits it to a pool.

ROLE_WORLD_AGENT = 1    # bit0=1 (world member), bit1=0 (agent)
ROLE_WORLD_MASTER = 3   # bit0=1 (world member), bit1=1 (master)


def test_char_known_peer_has_role_and_addrs_but_is_not_connected():
    """A peer known from the rendezvous exposes role and addrs through the facade,
    yet is_connected stays False and it never appears in get_all_connected_peer_infos.
    This is the invariant that KNOWN Sessions must not break."""
    conn, pub, world = _make_node_conn()
    conn.set_world_agents_list([{"id": "wa", "addrs": ["/ip4/1.1.1.1/tcp/1"], "misc": ROLE_WORLD_AGENT}])

    assert conn.get_role("wa") == ROLE_WORLD_AGENT
    assert conn.get_addrs("wa") == ["/ip4/1.1.1.1/tcp/1"]
    assert conn.count_by_role(ROLE_WORLD_AGENT) == 1
    assert conn.find_addrs_by_role(ROLE_WORLD_AGENT) == [["/ip4/1.1.1.1/tcp/1"]]
    # Known is not connected.
    assert conn.is_connected("wa") is False
    assert conn.get_all_connected_peer_infos(NodeConn.WORLD) == []


def test_char_connected_peer_is_connected_and_in_infos():
    """Mirror of the 'known is not connected' test: a peer actually admitted to a pool
    by update() reports is_connected True and shows up in get_all_connected_peer_infos.
    Freezes that the connected view stays transport/pool-driven, not session-driven."""
    conn, pub, world = _make_node_conn()
    conn._sm_world.world_agents_set = {"wc"}
    world._infos = [_info("wc", "inbound")]
    _run(conn.update())
    assert conn.is_connected("wc") is True
    infos = conn.get_all_connected_peer_infos(NodeConn.WORLD)
    assert any(i["id"] == "wc" for i in infos)


def test_char_unknown_peer_facade_defaults():
    """The facade's unknown-peer contract: role 0 (public), addrs None, empty role
    query (both arities), count 0, not connected."""
    conn, pub, world = _make_node_conn()
    assert conn.get_role("ghost") == 0
    assert conn.get_addrs("ghost") is None
    assert conn.find_addrs_by_role(ROLE_WORLD_AGENT) == []
    assert conn.find_addrs_by_role(ROLE_WORLD_AGENT, return_peer_ids_too=True) == ([], [])
    assert conn.count_by_role(ROLE_WORLD_AGENT) == 0
    assert conn.is_connected("ghost") is False


def test_char_find_addrs_by_role_returns_peer_ids_aligned():
    """find_addrs_by_role(return_peer_ids_too=True) returns addrs-lists and peer-ids
    positionally aligned, over all peers of that role (connected or not). Order follows
    set iteration, so assert as an aligned mapping, not a fixed sequence."""
    conn, pub, world = _make_node_conn()
    conn.set_world_masters_list([
        {"id": "m1", "addrs": ["/ip4/10.0.0.1/tcp/1"], "misc": ROLE_WORLD_MASTER},
        {"id": "m2", "addrs": ["/ip4/10.0.0.2/tcp/1"], "misc": ROLE_WORLD_MASTER},
    ])
    addrs, peer_ids = conn.find_addrs_by_role(ROLE_WORLD_MASTER, return_peer_ids_too=True)
    assert dict(zip(peer_ids, addrs)) == {
        "m1": ["/ip4/10.0.0.1/tcp/1"], "m2": ["/ip4/10.0.0.2/tcp/1"],
    }
    assert conn.count_by_role(ROLE_WORLD_MASTER) == 2


def test_char_find_addrs_by_role_skips_peers_without_addrs():
    """A peer that has a role but no stored addresses is skipped by find_addrs_by_role
    (the addrs-None filter), though count_by_role still counts it."""
    conn, pub, world = _make_node_conn()
    conn.set_role("noaddr", ROLE_WORLD_AGENT)  # role set, but no peer_id_to_addrs entry
    assert conn.count_by_role(ROLE_WORLD_AGENT) == 1
    assert conn.find_addrs_by_role(ROLE_WORLD_AGENT) == []
    assert conn.find_addrs_by_role(ROLE_WORLD_AGENT, return_peer_ids_too=True) == ([], [])


def test_char_world_host_writer_feeds_same_facade():
    """The world-host writer add_to_world_agents_list feeds role/addrs into the same
    facade as the rendezvous writer (the two-writers invariant): a peer added this way
    is visible through get_role/get_addrs/find_addrs_by_role but still not connected."""
    conn, pub, world = _make_node_conn()
    conn.add_to_world_agents_list("wh", ["/ip4/2.2.2.2/tcp/9"], role=ROLE_WORLD_AGENT)
    assert conn.get_role("wh") == ROLE_WORLD_AGENT
    assert conn.get_addrs("wh") == ["/ip4/2.2.2.2/tcp/9"]
    assert conn.find_addrs_by_role(ROLE_WORLD_AGENT) == [["/ip4/2.2.2.2/tcp/9"]]
    assert conn.is_connected("wh") is False


# ── KNOWN roster from the rendezvous via update() ────────────────────────────

def test_char_update_registers_known_world_sessions_off_diff():
    """update() registers a KNOWN Session in the world SessionManager for each peer the
    rendezvous advertises (here via set_world_agents_list), without that peer entering a
    pool, the added/removed diff, or is_connected. KNOWN is a registry, not a connection."""
    from unaiverse.networking.session import SessionState
    conn, pub, world = _make_node_conn()
    conn.set_world_agents_list([{"id": "wa", "addrs": ["/ip4/1.1.1.1/tcp/1"], "misc": ROLE_WORLD_AGENT}])
    added, removed = _run(conn.update())
    sess = conn._sm_world.get_session("wa")
    assert sess is not None and sess.state == SessionState.KNOWN
    assert sess.app_role == ROLE_WORLD_AGENT          # role carried through from the rendezvous
    assert sess.addresses == ["/ip4/1.1.1.1/tcp/1"]
    assert conn.is_connected("wa") is False
    assert all("wa" not in s for s in added.values())
    assert all("wa" not in s for s in removed.values())
    assert "wa" not in conn.peer_id_to_pool_name


def test_char_update_excludes_self_from_known_sessions():
    """The world instance never registers a KNOWN Session toward our own peer-id."""
    conn, pub, world = _make_node_conn()
    self_id = conn.world_peer_id
    conn.set_world_agents_list([{"id": self_id, "addrs": ["/ip4/9.9.9.9/tcp/9"], "misc": ROLE_WORLD_AGENT}])
    _run(conn.update())
    assert conn._sm_world.get_session(self_id) is None


def test_char_update_prunes_known_when_peer_leaves_rendezvous():
    """A KNOWN world Session is pruned once the rendezvous stops advertising the peer."""
    conn, pub, world = _make_node_conn()
    conn.set_world_agents_list([{"id": "wa", "addrs": ["/ip4/1.1.1.1/tcp/1"], "misc": ROLE_WORLD_AGENT}])
    _run(conn.update())
    assert conn._sm_world.get_session("wa") is not None
    conn.set_world_agents_list([])
    _run(conn.update())
    assert conn._sm_world.get_session("wa") is None


# ── S18 layering enforcement ──────────────────────────────────────────────────

def test_s18_coordinator_indexes_the_instance_by_sm_not_p2p():
    """Enforce the S18 contract on the Coordinator side: the per-instance indices are keyed
    on the SessionManager (the instance), not the raw P2P handle, and the raw handle is no
    longer exposed as a call/read channel. The P2P objects survive purely as custody."""
    from unaiverse.networking.session import SessionManager
    conn, pub, world = _make_node_conn()

    # _sm_by_name resolves to the SM, not the P2P. The raw-P2P index is gone, replaced by
    # the SM-keyed one.
    assert isinstance(conn._sm_by_name(NodeConn.P2P_PUBLIC), SessionManager)
    assert conn._sm_by_name(NodeConn.P2P_PUBLIC) is conn._sm_public
    assert conn._sm_by_name(NodeConn.P2P_WORLD) is conn._sm_world
    assert hasattr(conn, "_p2p_name_to_sm") and not hasattr(conn, "p2p_name_to_p2p")

    # Each pool triple's instance slot is the owning SM, never a P2P.
    assert conn.pool_name_to_pool_triple[NodeConn.IN_PUBLIC][2] is conn._sm_public
    assert conn.pool_name_to_pool_triple[NodeConn.IN_WORLD_AGENTS][2] is conn._sm_world
    for _, _, instance in conn.pool_name_to_pool_triple.values():
        assert isinstance(instance, SessionManager)

    # The Coordinator keeps no separate transport handle: each P2P lives inside its owning
    # SM, and the Coordinator reaches it (and the instance identity) through the SM.
    assert conn._sm_public.p2p is pub and conn._sm_world.p2p is world
    assert not hasattr(conn, "p2p_public") and not hasattr(conn, "p2p_world")
