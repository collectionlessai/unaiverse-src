"""Teardown verbs of the Coordinator (NodeConn).

The Coordinator exposes a single VOLUNTARY per-peer teardown, ``drop_peer``, and
reacts to an INVOLUNTARY transport drop via ``on_peer_disconnected``. Both are the
consequence of a Session transition, never a bare "close a connection":

  * ``drop_peer(peer, reason)`` on a live established Session — GOAWAY-close only
    (the transport must outlive the farewell; the settlement cut severs it once
    the CLOSED lands) + pool evict.
  * ``drop_peer(peer, None)`` on a live Session — silent coercive teardown: abort,
    forget, disconnect inline + pool evict.
  * ``on_peer_disconnected(peer)`` — the link is already gone: sync the pool
    bookkeeping (``_evict_from_pools``) and signal the SessionManagers, WITHOUT a
    transport disconnect (there is nothing to disconnect).

``SM.drop`` (the per-state teardown dispatch) and ``_evict_from_pools`` are the
primitives drop_peer composes; the only external teardown surface is ``drop_peer``
(voluntary), ``on_peer_disconnected`` (involuntary) and the handoff facades
(``complete_world_handoff`` / ``rollback_handoff``). The NodeConn constructor
never calls P2P methods, so a bare fake transport is enough.
"""

import asyncio

import pytest

from unaiverse.networking.node.connpool import NodeConn
from unaiverse.networking.session.sess_msg import WORLD_DISENGAGEMENT, SessionInbound
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
    """A live-enough Session that records close()/abort_silently(); planted directly
    into an SM's registry so drop's state dispatch has something to act on without
    driving a full handshake. Defaults to ESTABLISHED (the cooperative-close branch)."""

    def __init__(self, state=SessionState.ESTABLISHED):
        self.closed_with = None
        self.aborted_with = None
        self.state = state

    async def close(self, reason_code):
        self.closed_with = reason_code

    def abort_silently(self, reason):
        self.aborted_with = reason


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


# drop_peer: voluntary teardown

def test_drop_peer_with_reason_goaways_and_defers_transport_cut():
    """drop_peer(peer, reason) GOAWAYs the Session (close with that exact reason on
    the owning instance) and evicts the pool, but does NOT disconnect the transport:
    P2P.disconnect is abortive (it resets the stream under the just-written GOAWAY),
    so the cut is deferred to the settlement of the CLOSED terminal (ACK or the
    CLOSING timeout). The transport outliving the Session by a beat is what makes
    the farewell actually reach the peer on a real network."""
    conn, pub, world = _conn()
    peer = "w1"
    _plant_pool_member(conn, peer, NodeConn.IN_WORLD_AGENTS)
    stub = _StubSession()
    conn._sm_world._sessions[peer] = stub   # a live Session on the world instance

    events = _run(conn.drop_peer(peer, WORLD_DISENGAGEMENT))

    # (a) GOAWAY: the Session was closed with the reason.
    assert stub.closed_with == WORLD_DISENGAGEMENT
    # (b) the transport is NOT cut in the same stack as the GOAWAY write...
    assert peer not in world.disconnected
    # (c) ...but the pool bookkeeping is evicted right away.
    assert peer not in conn.peer_id_to_pool_name
    assert peer not in conn.pool_name_to_peer_infos[NodeConn.IN_WORLD_AGENTS]
    assert peer not in conn.peer_id_to_token
    # close returns [] (the CLOSED event arrives later via tick/goaway-ack).
    assert events == []


def test_drop_peer_without_reason_aborts_live_session_and_drops():
    """drop_peer(peer, None) on a live Session is the silent coercive teardown: the
    Session is aborted (no GOAWAY, no terminal event), forgotten, and the transport
    disconnected inline — the observable semantics of the old reasonless drop,
    without leaving a zombie Session in the registry that could later RESUME a
    relation the node already dismantled."""
    conn, pub, world = _conn()
    peer = "w1"
    _plant_pool_member(conn, peer, NodeConn.IN_WORLD_AGENTS)
    stub = _StubSession()
    conn._sm_world._sessions[peer] = stub

    _run(conn.drop_peer(peer))   # reason defaults to None

    assert stub.closed_with is None          # no GOAWAY
    assert stub.aborted_with is not None     # silently aborted, no event
    assert conn._sm_world.get_session(peer) is None   # forgotten
    assert peer in world.disconnected        # transport dropped inline
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
    bare transport drop that leaves the Session ESTABLISHED to reconnect and re-enrol.
    The transport is NOT severed in the same stack (that would abort the GOAWAY on
    the wire): the settlement cut of the CLOSED terminal severs it a beat later."""
    conn, pub, world = _conn()
    peer = "wa1"
    _plant_pool_member(conn, peer, NodeConn.IN_WORLD_AGENTS)
    stub = _StubSession()
    conn._sm_world._sessions[peer] = stub

    _run(conn.remove_all_world_agents())

    assert stub.closed_with == WORLD_DISENGAGEMENT   # GOAWAY, not a bare drop
    assert peer not in world.disconnected            # cut deferred to settlement
    assert peer not in conn.peer_id_to_pool_name     # pool evicted


# initiate_session: spent-carrier detection (B8 × B12)

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


# on_peer_disconnected: involuntary pool-sync

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


# World-handoff carrier retirement

def test_complete_world_handoff_retires_carrier_without_touching_transport():
    """complete_world_handoff GOAWAYs the correlated public carrier on its still-live
    transport and evicts its pool entry itself (the dispatch early-returns on
    WORLD_HANDOFF_COMPLETE, so no purge will do it), leaving the transport to the
    settlement cut. Disconnecting the carrier before/inside the retirement used to
    destroy the very GOAWAY the host was parked in AWAITING_HANDOFF waiting for."""
    conn, pub, world = _conn()
    pub_peer, world_peer = "host-pub", "host-world"
    _plant_pool_member(conn, pub_peer, NodeConn.OUT_PUBLIC)
    carrier = _StubSession(state=SessionState.AWAITING_HANDOFF)
    conn._sm_public._sessions[pub_peer] = carrier
    conn._pending_world_handoff[pub_peer] = world_peer
    conn._world_to_public[world_peer] = pub_peer

    ret = _run(conn.complete_world_handoff(world_peer))

    assert ret is None                                   # contract: nothing to dispatch
    from unaiverse.networking.session.sess_msg import WORLD_HANDOFF_COMPLETE
    assert carrier.closed_with == WORLD_HANDOFF_COMPLETE  # GOAWAY out...
    assert pub_peer not in pub.disconnected               # ...transport untouched
    assert pub_peer not in conn.peer_id_to_pool_name      # pool evicted here
    assert not conn.is_pending_handoff(world_peer)        # correlation dropped


def test_rollback_handoff_unpoints_world_routing():
    """rollback_handoff on a failed handoff also resets set_world when the rolled-back
    world peer is the current world host: start_world_handoff pointed routing at it
    optimistically, and a world never joined must not stay classified as ours."""
    conn, pub, world = _conn()
    pub_peer, world_peer = "host-pub", "host-world"
    conn.set_world(world_peer)
    conn._pending_world_handoff[pub_peer] = world_peer
    conn._world_to_public[world_peer] = pub_peer

    _run(conn.rollback_handoff(pub_peer))   # the public carrier FAILED

    assert conn.get_world_node_peer_id() is None
    assert not conn.is_pending_handoff(world_peer)


def test_rollback_handoff_aborts_inflight_world_session():
    """A public carrier dying mid-handoff takes the still-in-flight world redemption
    down with it: left running, the dial/handshake would complete into a headless
    join (correlation gone, set_world None) that ends in the node coercively cutting
    its own world host with no event — a permanent wedge."""
    conn, pub, world = _conn()
    pub_peer, world_peer = "host-pub", "host-world"
    conn.set_world(world_peer)
    conn._pending_world_handoff[pub_peer] = world_peer
    conn._world_to_public[world_peer] = pub_peer
    redemption = _StubSession(state=SessionState.HELLO_SENT)   # world dial in flight
    conn._sm_world._sessions[world_peer] = redemption

    _run(conn.rollback_handoff(pub_peer))

    assert redemption.aborted_with is not None            # redemption aborted...
    assert conn._sm_world.get_session(world_peer) is None  # ...and forgotten
    assert world_peer in world.disconnected               # transport of the dial severed
    assert not conn.is_pending_handoff(world_peer)
    assert conn.get_world_node_peer_id() is None


def test_rollback_handoff_noop_when_world_session_established():
    """If the correlated world Session already ESTABLISHED, the carrier's death no
    longer matters: the join is completing and complete_world_handoff owns the
    cleanup. Rolling back here would strand the completing join half-rolled-back
    (in-world with no world host = the perimeter cuts our own host, wedge)."""
    conn, pub, world = _conn()
    pub_peer, world_peer = "host-pub", "host-world"
    conn.set_world(world_peer)
    conn._pending_world_handoff[pub_peer] = world_peer
    conn._world_to_public[world_peer] = pub_peer
    redemption = _StubSession(state=SessionState.ESTABLISHED)
    conn._sm_world._sessions[world_peer] = redemption

    _run(conn.rollback_handoff(pub_peer))

    assert redemption.aborted_with is None                 # untouched
    assert conn.is_pending_handoff(world_peer)             # correlation kept
    assert conn.get_world_node_peer_id() == world_peer     # routing kept


# Shutdown drain

def _fp2p_with_inbox(fp2p):
    """Give a _FakeP2P the session surface the drain pumps: send_session records,
    get_messages pops a preloadable inbox of SessionInbound-shaped items."""
    fp2p.sent = []
    fp2p.inbox = []

    async def send_session(peer_id, protocol_id, sess_msg):
        fp2p.sent.append((peer_id, protocol_id, sess_msg))
        return True

    def get_messages():
        batch, fp2p.inbox = fp2p.inbox, []
        return [], batch

    fp2p.send_session = send_session
    fp2p.get_messages = get_messages
    return fp2p


def _real_established(conn, sm_attr, fp2p, peer_id):
    from unaiverse.networking.session import Session, SessionRole
    sm = getattr(conn, sm_attr)
    sess = Session(peer_id=peer_id, role=SessionRole.INITIATOR, p2p=fp2p,
                   p2p_name=sm.p2p_name, local_peer_id=sm.local_peer_id,
                   local_node_id="n", local_profile_factory=lambda: {})
    sess._set_state(SessionState.ESTABLISHED, None)
    sm._set_active(peer_id, sess)
    return sess


def test_drain_teardown_settles_ack_then_cut(monkeypatch):
    """The drain pumps until the farewell settles end-to-end: GOAWAY out (close_all),
    the ACK delivered by the pump completes CLOSING -> CLOSED, the settlement cut
    executes after its linger, and only then the drain returns — well within budget."""
    import unaiverse.networking.session.session_manager as sm_module
    from unaiverse.networking.session import SessMsg, SESSION_CONTROL_PROTOCOL
    monkeypatch.setattr(sm_module, "_CUT_LINGER_S", 0.05)   # keep the test fast

    conn, pub, world = _conn()
    _fp2p_with_inbox(world)
    peer = "w1"
    sess = _real_established(conn, "_sm_world", world, peer)

    _run(conn.close_all_sessions())
    assert sess.state == SessionState.CLOSING
    # The peer's GOAWAY_ACK arrives while draining.
    world.inbox.append(SessionInbound(peer_id=peer, protocol_id=SESSION_CONTROL_PROTOCOL,
                                      sess_msg=SessMsg(SessMsg.KIND_GOAWAY_ACK, {"received": True})))

    import time as _t
    t0 = _t.monotonic()
    _run(conn.drain_teardown(budget_s=3.0))
    elapsed = _t.monotonic() - t0

    assert conn._sm_world.get_session(peer) is None       # CLOSED + settled
    assert peer in world.disconnected                     # settlement cut executed
    assert not conn._sm_world.has_pending_teardown()
    assert elapsed < 1.0                                  # settled, did not eat the budget


def test_drain_exits_with_inflight_handshake_aborted():
    """A handshake still in flight at shutdown is aborted by close_all (no farewell
    owed), so the drain has nothing to wait on and returns immediately instead of
    burning its whole budget on an FSM nobody will ever answer."""
    conn, pub, world = _conn()
    _fp2p_with_inbox(world)
    peer = "w1"
    sess = _real_established(conn, "_sm_world", world, peer)
    sess._set_state(SessionState.HELLO_SENT, 5.0)         # in-flight, not closable

    _run(conn.close_all_sessions())
    assert conn._sm_world.get_session(peer) is None       # aborted + forgotten

    import time as _t
    t0 = _t.monotonic()
    _run(conn.drain_teardown(budget_s=3.0))
    assert _t.monotonic() - t0 < 0.5


def test_drain_gives_up_at_budget_when_ack_never_comes():
    """No ACK and no timeout within the budget: the drain is bounded — it returns at
    the deadline with the Session still CLOSING (the GOAWAY had its chance on the
    wire; the transport teardown that follows is the best we can do)."""
    conn, pub, world = _conn()
    _fp2p_with_inbox(world)
    peer = "w1"
    sess = _real_established(conn, "_sm_world", world, peer)

    _run(conn.close_all_sessions())
    import time as _t
    t0 = _t.monotonic()
    _run(conn.drain_teardown(budget_s=0.3))
    elapsed = _t.monotonic() - t0
    assert 0.25 <= elapsed < 1.5
    assert sess.state == SessionState.CLOSING


def test_settled_drop_peer_never_touches_the_session():
    """The reckoning purge of an already-settled end (terminal dispatch, polled
    transport drop) evicts pools only: a SUCCESSOR Session born in the same pump
    batch — live where the settled one is gone — must survive it untouched. This is
    what keeps the old relation's consuntivo from coercively killing the fresh
    relation that replaced it (and from destroying its just-queued HELLO_RESP)."""
    conn, pub, world = _conn()
    peer = "w1"
    _plant_pool_member(conn, peer, NodeConn.IN_WORLD_AGENTS)
    successor = _StubSession(state=SessionState.HELLO_ACK_SENT)   # fresh, mid-handshake
    conn._sm_world._sessions[peer] = successor

    _run(conn.drop_peer(peer, settled=True))

    assert successor.aborted_with is None            # untouched...
    assert successor.closed_with is None
    assert conn._sm_world.get_session(peer) is successor   # ...and still registered
    assert peer not in world.disconnected            # transport untouched
    assert peer not in conn.peer_id_to_pool_name     # pools evicted (the reckoning's job)
