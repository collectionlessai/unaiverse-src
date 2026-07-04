"""Teardown verbs of the Coordinator (NodeConn).

The Coordinator exposes a single VOLUNTARY per-peer teardown, ``drop_peer``, and
reacts to an INVOLUNTARY transport drop via ``on_peer_disconnected``. Both are the
consequence of a Session transition, never a bare "close a connection":

  * ``drop_peer(peer, reason)`` — we close the peer's Session (GOAWAY on the owning
    instance) and then drop the transport + pool bookkeeping.
  * ``drop_peer(peer, None)`` — the Session already ended (remote CLOSED/LOST or a
    timeout): no GOAWAY, only the transport + pool drop.
  * ``on_peer_disconnected(peer)`` — the link is already gone: sync the pool
    bookkeeping (``_evict_from_pools``) and signal the SessionManagers, WITHOUT a
    transport disconnect (there is nothing to disconnect).

``SM.drop`` (GOAWAY-close + transport disconnect on the owning instance) and
``_evict_from_pools`` are the primitives drop_peer composes; the only external
teardown surface is ``drop_peer`` (voluntary) and ``on_peer_disconnected``
(involuntary). The NodeConn constructor never calls P2P methods, so a bare fake
transport is enough.
"""

import asyncio

import pytest

from unaiverse.networking.node.connpool import NodeConn
from unaiverse.networking.session.sess_msg import WORLD_DISENGAGEMENT
from unaiverse.networking.session.session import SessionState

pytestmark = pytest.mark.unit

ROLE_WORLD_AGENT = 1


class _FakeP2P:
    """Minimal transport: no connected peers, records disconnect calls."""

    def __init__(self, log_sub):
        self.log_sub = log_sub
        self.peer_id = log_sub + "-self"
        self.disconnected = []

    def get_connected_peers_info(self):
        return []

    async def disconnect(self, peer_id):
        self.disconnected.append(peer_id)
        return True


class _StubSession:
    """A live-enough Session that records close(); planted directly into an SM's
    registry so drop_peer's GOAWAY branch has something to close without driving a
    full handshake."""

    def __init__(self):
        self.closed_with = None

    async def close(self, reason_code):
        self.closed_with = reason_code


def _run(coro):
    return asyncio.run(coro)


def _conn():
    pub, world = _FakeP2P("pub"), _FakeP2P("world")
    conn = NodeConn(max_connections=16, is_world_node=False, public_key=None, token=None,
                    local_node_id="n")
    conn.attach_session_policy(p2p_u=pub, p2p_w=world,
                              local_profile_factory=lambda: {}, decision_hook=None)
    return conn, pub, world


def _plant_pool_member(conn, peer_id, pool_name):
    """Put a connected peer into a pool by hand (bypasses update())."""
    pool, _, _ = conn.pool_name_to_pool_triple[pool_name]
    pool.add(peer_id)
    conn.peer_id_to_pool_name[peer_id] = pool_name
    conn.pool_name_to_peer_infos[pool_name][peer_id] = {"id": peer_id}
    conn.peer_id_to_token[peer_id] = "tok"


# ── drop_peer: voluntary teardown ────────────────────────────────────────────

def test_drop_peer_with_reason_goaways_then_drops():
    """drop_peer(peer, reason) GOAWAYs the Session (close with that exact reason on
    the owning instance), then disconnects the transport and evicts the pool."""
    conn, pub, world = _conn()
    peer = "w1"
    _plant_pool_member(conn, peer, NodeConn.IN_WORLD_AGENTS)
    stub = _StubSession()
    conn._sm_world._sessions[peer] = stub   # a live Session on the world instance

    events = _run(conn.drop_peer(peer, WORLD_DISENGAGEMENT))

    # (a) GOAWAY: the Session was closed with the reason.
    assert stub.closed_with == WORLD_DISENGAGEMENT
    # (b) transport disconnected on the owning instance...
    assert peer in world.disconnected
    # (c) ...and the pool bookkeeping evicted.
    assert peer not in conn.peer_id_to_pool_name
    assert peer not in conn.pool_name_to_peer_infos[NodeConn.IN_WORLD_AGENTS]
    assert peer not in conn.peer_id_to_token
    # close returns [] (the CLOSED event arrives later via tick/goaway-ack).
    assert events == []


def test_drop_peer_without_reason_drops_without_goaway():
    """drop_peer(peer, None) (the Session already ended): no GOAWAY, but the
    transport is still disconnected and the pool evicted."""
    conn, pub, world = _conn()
    peer = "w1"
    _plant_pool_member(conn, peer, NodeConn.IN_WORLD_AGENTS)
    stub = _StubSession()
    conn._sm_world._sessions[peer] = stub

    _run(conn.drop_peer(peer))   # reason defaults to None

    assert stub.closed_with is None          # no GOAWAY
    assert peer in world.disconnected        # transport still dropped
    assert peer not in conn.peer_id_to_pool_name


def test_drop_peer_keeps_role_taxonomy():
    """drop_peer evicts the pool/connection indices but KEEPS peer_id_to_misc (the
    role), so a role survives a disconnect/re-add (I-2)."""
    conn, pub, world = _conn()
    peer = "w1"
    conn.add_to_world_agents_list(peer, ["/ip4/1.1.1.1/tcp/1"], role=ROLE_WORLD_AGENT)
    _plant_pool_member(conn, peer, NodeConn.IN_WORLD_AGENTS)

    _run(conn.drop_peer(peer))

    assert peer not in conn.peer_id_to_pool_name
    assert conn.get_role(peer) == ROLE_WORLD_AGENT


def test_remove_all_world_agents_goaways_ex_companions():
    """Leaving a world GOAWAY-closes every in-world agent (WORLD_DISENGAGEMENT) so its
    Session reaches CLOSED and never re-dials us back into the abandoned world — not a
    bare transport drop that leaves the Session ESTABLISHED to reconnect and re-enrol."""
    conn, pub, world = _conn()
    peer = "wa1"
    _plant_pool_member(conn, peer, NodeConn.IN_WORLD_AGENTS)
    stub = _StubSession()
    conn._sm_world._sessions[peer] = stub

    _run(conn.remove_all_world_agents())

    assert stub.closed_with == WORLD_DISENGAGEMENT   # GOAWAY, not a bare drop
    assert peer in world.disconnected                # transport severed
    assert peer not in conn.peer_id_to_pool_name     # pool evicted


# ── initiate_session: spent-carrier detection (B8 × B12) ─────────────────────

def test_initiate_session_flags_spent_carrier_but_protects_pending_handoff():
    """A world-join dial to a peer holding an ESTABLISHED world-join carrier is flagged
    replace_stale ONLY when there is no pending handoff. A spent carrier (handoff failed and
    rolled back) is replaced so the join restarts (B8); an in-flight handoff is protected
    (no replace), so B12's watchdog cannot tear down a live join."""
    conn, pub, world = _conn()
    peer = "host-pub"
    carrier = _StubSession()
    carrier.is_world_join = True
    carrier.state = SessionState.ESTABLISHED
    conn._sm_public._sessions[peer] = carrier

    # Capture the flag the Coordinator passes down, without driving a real dial.
    captured = {}

    async def _spy_initiate(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return []
    conn._sm_public.initiate = _spy_initiate

    # No pending handoff → spent carrier → replace.
    _run(conn.initiate_session(NodeConn.P2P_PUBLIC, peer, ["/ip4/1.1.1.1/tcp/1"], is_world_join=True))
    assert captured["replace_stale"] is True

    # A pending handoff for the same peer → live join → protected (no replace).
    conn._pending_world_handoff[peer] = peer + "-w"
    _run(conn.initiate_session(NodeConn.P2P_PUBLIC, peer, ["/ip4/1.1.1.1/tcp/1"], is_world_join=True))
    assert captured["replace_stale"] is False


# ── on_peer_disconnected: involuntary pool-sync ──────────────────────────────

def test_on_peer_disconnected_evicts_pool_without_transport_disconnect():
    """on_peer_disconnected reacts to a link that is ALREADY gone: it evicts the
    pool bookkeeping (so the projection reflects reality) but does NOT call
    disconnect_from (there is nothing to disconnect). This is the involuntary
    pool-sync that used to be a Node bare-remove in the grace branch."""
    conn, pub, world = _conn()
    peer = "w1"
    _plant_pool_member(conn, peer, NodeConn.IN_WORLD_AGENTS)

    # No Session for the peer: on_peer_disconnected still syncs the pool.
    events = _run(conn.on_peer_disconnected(peer))

    assert peer not in conn.peer_id_to_pool_name
    assert peer not in conn.pool_name_to_peer_infos[NodeConn.IN_WORLD_AGENTS]
    # The transport is NOT disconnected (the link is already gone).
    assert peer not in world.disconnected
    assert peer not in pub.disconnected
    # No Session → no events (the SM signalling is covered by the SM/integration tests).
    assert events == []
