"""
SessionManager — owns all Sessions for one P2P instance.

One SessionManager is created per p2p instance (p2p_public / p2p_world). Node
holds these and calls:

  - ``await sm.initiate(peer_id, addresses, ...)`` when it wants to open a new
    Session as initiator.
  - ``await sm.deliver(peer_id, protocol_id, payload_bytes)`` for every inbound
    session-protocol message returned by ``conn.get_messages``.
  - ``await sm.tick(now)`` once per polling-loop iteration to fire timeouts and
    keep-alive pings.
  - ``await sm.on_peer_disconnected(peer_id)`` / ``sm.on_peer_reconnected(...)``
    when the libp2p connection state changes (as observed by the existing
    polling diff in ``__handle_network_connections``).
  - ``await sm.close_session(peer_id, reason)`` for graceful GOAWAY.

Every public method returns ``list[SessionEvent]`` — node.py is the single
dispatch site. No internal pending-event queue, no on_event callback.

Tie-breaker on simultaneous open
--------------------------------
libp2p collapses *connections* automatically; we still need to collapse the two
*FSMs* (initiator we created, responder spawned from inbound HELLO) when both
exist for the same peer-id. Rule mirrors libp2p's standard:
the peer with the lexicographically *smaller* peer-id keeps its initiator. The
other side drops its initiator and accepts the inbound HELLO.

Anti-flap quarantine
--------------------
After a Lost event, the peer-id is held in quarantine for ``_QUARANTINE_S``
seconds — re-dials during that window return immediately with
``[SessionEvent(FAILED, reason="quarantined")]``. Replaces the legacy
``keep_connection=True`` soft-purge path.
"""

from __future__ import annotations

import time
import logging
from typing import Dict, List, Optional, Callable, TYPE_CHECKING

from unaiverse.networking.session.session import (
    Session, SessionState, SessionRole, SessionEvent, SessionEventType,
    SESSION_HELLO_PROTOCOL, SESSION_PROFILE_PROTOCOL,
    SESSION_WORLD_PROTOCOL, SESSION_CONTROL_PROTOCOL,
    SESSION_PROTOCOL_IDS,
)
from unaiverse.networking.p2p import session_pb2 as spb

if TYPE_CHECKING:
    from unaiverse.networking.node.connpool import ConnectionPools

log = logging.getLogger("SessionManager")

_QUARANTINE_S = 30.0


def _p2p_label(name: str) -> str:
    return "pub" if "public" in name else "prv" if "world" in name else name


def _looks_like_hello_request(payload: bytes) -> bool:
    """Heuristically distinguish a HelloRequest from a HelloResponse on the
    shared /hello/1.0.0 protocol.

    HelloRequest invariants: session_id is exactly 16 bytes, nonce_a is
    exactly 32 bytes. Both messages use the same protocol_id and both decode
    permissively into either schema; this check holds because HelloResponse
    field 1 is a small protocol-version string and field 2 is a varint, which
    won't match the byte-length constraints above when (mis-)read as
    HelloRequest fields.
    """
    try:
        req = spb.HelloRequest()
        req.ParseFromString(payload)
        return len(req.session_id) == 16 and len(req.nonce_a) == 32
    except Exception:
        return False


class SessionManager:
    """Per-p2p-instance Session registry and dispatcher."""

    def __init__(
        self,
        conn: "ConnectionPools",
        p2p_name: str,
        local_peer_id: str,
        local_node_id: str,
        local_profile_factory: Callable[[], dict],
        local_spr_other_factory: Optional[Callable[[], Optional[bytes]]] = None,
    ):
        """
        Args:
            conn: the ConnectionPools instance.
            p2p_name: the P2P instance this manager owns (e.g. "p2p_public").
            local_peer_id: our libp2p peer-id on `p2p_name` (used for tie-breaker).
            local_node_id: our application node-id (passed into each Session).
            local_profile_factory: () -> NodeProfile dict snapshot.
            local_spr_other_factory: () -> SignedPeerRecord bytes for the *other*
                peer-id (the p2p_world key when this manager runs on p2p_public).
                None disables SignedPeerRecord inclusion for this instance.
        """
        self.conn = conn
        self.p2p_name = p2p_name
        self.local_peer_id = local_peer_id
        self.local_node_id = local_node_id
        self._local_profile_factory = local_profile_factory
        self._local_spr_other_factory = local_spr_other_factory or (lambda: None)
        self._label = _p2p_label(p2p_name)

        # peer_id → Session (at most one Session per peer in this p2p instance).
        self._sessions: Dict[str, Session] = {}

        # peer_id → monotonic timestamp until which dials are refused.
        self._quarantine: Dict[str, float] = {}

    # ── Initiator API ─────────────────────────────────────────────────────────

    async def initiate(
        self,
        peer_id: str,
        addresses: List[str],
        is_world_join: bool = False,
        remote_other_peer_id_hint: Optional[str] = None,
        remote_node_id_hint: Optional[str] = None,
    ) -> List[SessionEvent]:
        """Open a new initiator Session to `peer_id`. Idempotent if the peer
        already has a healthy Session — returns []. If a stale FAILED/CLOSED
        Session exists, it is replaced."""
        now = time.monotonic()
        # Quarantine check.
        ban_until = self._quarantine.get(peer_id, 0.0)
        if ban_until > now:
            log.info(f"[{self._label}] initiate({peer_id[:12]}…) refused: quarantined for {ban_until - now:.1f}s")
            ev = SessionEvent(
                type=SessionEventType.FAILED, session_id="",
                peer_id=peer_id, role=SessionRole.INITIATOR,
                p2p_name=self.p2p_name, data={"reason": "quarantined"},
            )
            return [ev]

        existing = self._sessions.get(peer_id)
        if existing is not None and existing.state not in (SessionState.FAILED, SessionState.CLOSED):
            log.debug(f"[{self._label}] initiate({peer_id[:12]}…): "
                      f"already have Session in state {existing.state.value} — no-op")
            return []

        sess = Session(
            peer_id=peer_id,
            role=SessionRole.INITIATOR,
            conn=self.conn,
            p2p_name=self.p2p_name,
            local_peer_id=self.local_peer_id,
            local_node_id=self.local_node_id,
            local_profile_factory=self._local_profile_factory,
            local_spr_other=self._local_spr_other_factory(),
            remote_node_id_hint=remote_node_id_hint,
            remote_other_peer_id_hint=remote_other_peer_id_hint,
            addresses=addresses,
            is_world_join=is_world_join,
        )
        self._sessions[peer_id] = sess
        log.info(f"[{self._label}] initiating Session to {peer_id[:12]}…")
        return await sess.start()

    # ── Inbound dispatch ──────────────────────────────────────────────────────

    async def deliver(self, peer_id: str, protocol_id: str, payload: bytes) -> List[SessionEvent]:
        """Route a session-protocol message at this manager's p2p instance.
        Handles tie-breaker on inbound HELLO from an unknown / contending peer."""
        if protocol_id not in SESSION_PROTOCOL_IDS:
            log.warning(f"[{self._label}] deliver: unknown protocol {protocol_id}, dropping")
            return []

        sess = self._sessions.get(peer_id)

        if sess is None:
            if protocol_id == SESSION_HELLO_PROTOCOL:
                # Inbound HELLO from an unknown peer → spawn a responder Session
                # (no tie-breaker possible: we have no in-flight initiator for them).
                sess = self._spawn_responder(peer_id, remote_other_peer_id_hint=None)
            else:
                log.debug(f"[{self._label}] {protocol_id} from unknown peer {peer_id[:12]}… — dropped")
                return []

        # Tie-breaker — but only when the inbound is actually a competing
        # HelloRequest, not the HelloResponse coming back to our own dial.
        # Both messages travel on the same protocol_id, so we disambiguate by
        # parsing and checking the HelloRequest-specific invariants
        # (16-byte session_id + 32-byte nonce_a). HelloResponse field numbers
        # have different wire types or different value-length expectations and
        # won't pass both checks.
        elif (protocol_id == SESSION_HELLO_PROTOCOL and
              sess.role == SessionRole.INITIATOR and
              sess.state in (SessionState.CONNECTING, SessionState.HELLO_SENT) and
              _looks_like_hello_request(payload)):
            events = await self._handle_simultaneous_open(peer_id, payload)
            if events == "_proceed_with_existing_":   # type: ignore[comparison-overlap]
                # We kept the existing initiator; the inbound HELLO is dropped after
                # we wrote a soft-reject to the peer (handled inside the helper).
                return []
            elif events == "_replaced_with_responder_":   # type: ignore[comparison-overlap]
                sess = self._sessions[peer_id]
                # Fall through and process the inbound HELLO against the new responder.
            else:
                return events  # type: ignore[return-value]

        return await sess.process_inbound(protocol_id, payload)

    def _spawn_responder(self, peer_id: str, remote_other_peer_id_hint: Optional[str]) -> Session:
        sess = Session(
            peer_id=peer_id,
            role=SessionRole.RESPONDER,
            conn=self.conn,
            p2p_name=self.p2p_name,
            local_peer_id=self.local_peer_id,
            local_node_id=self.local_node_id,
            local_profile_factory=self._local_profile_factory,
            local_spr_other=self._local_spr_other_factory(),
            remote_other_peer_id_hint=remote_other_peer_id_hint,
        )
        self._sessions[peer_id] = sess
        log.info(f"[{self._label}] spawned responder Session for {peer_id[:12]}…")
        return sess

    async def _handle_simultaneous_open(self, peer_id: str, _payload: bytes):
        """Deterministic tie-breaker: smaller peer-id keeps initiator, larger
        yields to the responder. Returns one of:

          * "_proceed_with_existing_"   — kept initiator; caller drops inbound HELLO
          * "_replaced_with_responder_" — dropped initiator, spawned responder;
                                          caller MUST process the inbound HELLO
          * list[SessionEvent]          — on any error (e.g. reject-send failed)
        """
        existing = self._sessions[peer_id]
        we_keep_initiator = self.local_peer_id < peer_id
        if we_keep_initiator:
            # Politely tell the remote responder that we are also dialing and they
            # should drop their initiator attempt. We can't trivially send on a
            # never-opened stream, so the cleanest thing is to send a HelloResponse
            # rejection on the hello protocol from our side as the "responder we
            # would have been" — the remote initiator's HELLO_SENT FSM will see it
            # and fail with SIMULTANEOUS_OPEN_LOSS. Use the existing initiator's
            # session_id (cheap, non-conflicting) for the rejection.
            try:
                reject = spb.HelloResponse(
                    chosen_version="",
                    capabilities=0,
                    nonce_b=b"",
                    sig_b=b"",
                    reason_code=spb.SIMULTANEOUS_OPEN_LOSS,
                )
                await self.conn.send_session(
                    peer_id=peer_id,
                    protocol_id=SESSION_HELLO_PROTOCOL,
                    payload=reject.SerializeToString(),
                    p2p=self.conn.p2p_name_to_p2p[self.p2p_name],
                )
            except Exception as e:
                log.warning(f"[{self._label}] simultaneous-open reject to {peer_id[:12]}… failed: {e}")
            log.info(f"[{self._label}] simultaneous-open with {peer_id[:12]}…: kept initiator")
            return "_proceed_with_existing_"

        # We yield: drop our initiator, spawn a fresh responder for the inbound HELLO.
        log.info(f"[{self._label}] simultaneous-open with {peer_id[:12]}…: yielding to responder")
        existing.abort_silently("simultaneous_open_yielded")
        del self._sessions[peer_id]
        self._spawn_responder(peer_id, remote_other_peer_id_hint=existing.remote_other_peer_id_hint)
        return "_replaced_with_responder_"

    # ── Per-cycle tick (timeouts + keep-alive) ────────────────────────────────

    async def tick(self, now: Optional[float] = None) -> List[SessionEvent]:
        """Sweep all sessions: timeout firings + /ipfs/ping keep-alive. Single
        dispatch path; the returned list is what node.py acts on."""
        if now is None:
            now = time.monotonic()
        events: List[SessionEvent] = []
        # Snapshot of values; mutation during iteration is safe because we operate
        # on the snapshot only (per-session state changes happen synchronously and
        # don't add new sessions inside tick).
        for sess in list(self._sessions.values()):
            t_event = sess.check_timeout(now)
            if t_event is not None:
                events.append(t_event)
            k_event = sess.tick_keepalive(now)
            if k_event is not None:
                events.append(k_event)

        # Reap quarantine entries that have expired (cheap; bounded by peer count).
        if self._quarantine:
            expired = [pid for pid, t in self._quarantine.items() if t <= now]
            for pid in expired:
                self._quarantine.pop(pid, None)

        # Quarantine on Lost; remove session entries after terminal events.
        self._post_dispatch_cleanup(events, now)
        return events

    def _post_dispatch_cleanup(self, events: List[SessionEvent], now: float) -> None:
        for ev in events:
            if ev.type == SessionEventType.LOST:
                self._quarantine[ev.peer_id] = now + _QUARANTINE_S
            if ev.type in (SessionEventType.LOST,
                           SessionEventType.CLOSED,
                           SessionEventType.FAILED):
                # Only drop from registry if the session is genuinely terminal.
                sess = self._sessions.get(ev.peer_id)
                if sess is not None and sess.state in (SessionState.FAILED, SessionState.CLOSED):
                    self._sessions.pop(ev.peer_id, None)

    # ── Transport events ──────────────────────────────────────────────────────

    async def on_peer_disconnected(self, peer_id: str) -> List[SessionEvent]:
        """Called when the polled libp2p diff reports a peer dropping."""
        sess = self._sessions.get(peer_id)
        if sess is None:
            return []
        if sess.state in (SessionState.CONNECTING,
                          SessionState.HELLO_SENT,
                          SessionState.HELLO_ACK_SENT,
                          SessionState.PROFILE_SENT,
                          SessionState.AWAITING_WORLD_GRANT,
                          SessionState.AWAITING_PRIVATE_JOIN):
            # Handshake can't continue without a live connection — fail fast.
            ev = sess._fail("connection_dropped_during_handshake")
            self._post_dispatch_cleanup([ev], time.monotonic())
            return [ev]

        ev = sess.on_transport_disconnected()
        return [ev] if ev is not None else []

    async def on_peer_reconnected(self, peer_id: str) -> List[SessionEvent]:
        sess = self._sessions.get(peer_id)
        if sess is None:
            return []
        ev = sess.on_transport_reconnected()
        return [ev] if ev is not None else []

    # ── Intentional close ─────────────────────────────────────────────────────

    async def close_session(
        self,
        peer_id: str,
        reason_code: int = spb.GRACEFUL_SHUTDOWN,
    ) -> List[SessionEvent]:
        sess = self._sessions.get(peer_id)
        if sess is None:
            log.debug(f"[{self._label}] close_session({peer_id[:12]}…): no Session — no-op")
            return []
        await sess.close(reason_code)
        return []  # CLOSED event will arrive via process_inbound(GoawayAck) or tick

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_session(self, peer_id: str) -> Optional[Session]:
        return self._sessions.get(peer_id)

    def is_established(self, peer_id: str) -> bool:
        s = self._sessions.get(peer_id)
        return s is not None and s.state == SessionState.ESTABLISHED

    def all_established_peer_ids(self) -> List[str]:
        return [pid for pid, s in self._sessions.items()
                if s.state == SessionState.ESTABLISHED]

    def __len__(self) -> int:
        return len(self._sessions)

    def __repr__(self) -> str:
        by_state: Dict[str, int] = {}
        for s in self._sessions.values():
            by_state[s.state.value] = by_state.get(s.state.value, 0) + 1
        return f"SessionManager(p2p={self.p2p_name} sessions={by_state} quarantined={len(self._quarantine)})"
