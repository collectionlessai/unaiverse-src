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
from typing import Dict, List, Optional, Callable, TYPE_CHECKING

from unaiverse.networking.session.session import (
    Session, SessionState, SessionRole, SessionEvent, SessionEventType, DecisionHook,
    SESSION_HELLO_PROTOCOL, SESSION_PROFILE_PROTOCOL,
    SESSION_WORLD_PROTOCOL, SESSION_CONTROL_PROTOCOL,
    SESSION_PROTOCOL_IDS,
)
from unaiverse.networking.session.sess_msg import (
    SessMsg, SIMULTANEOUS_OPEN_LOSS, GRACEFUL_SHUTDOWN, SESSION_UNKNOWN,
)
from unaiverse.utils.logger import log

if TYPE_CHECKING:
    from unaiverse.networking.node.connpool import ConnectionPools

_QUARANTINE_S = 30.0


def _p2p_label(name: str) -> str:
    return "pub" if "public" in name else "prv" if "world" in name else name


class SessionManager:
    """Per-p2p-instance Session registry and dispatcher."""

    def __init__(
        self,
        conn: "ConnectionPools",
        p2p_name: str,
        local_peer_id: str,
        local_node_id: str,
        local_profile_factory: Callable[[], dict],
        decision_hook: Optional[DecisionHook] = None,
    ):
        """
        Args:
            conn: the ConnectionPools instance (or a JS NodeConn mirror in Pyodide).
            p2p_name: the P2P instance this manager owns (e.g. "p2p_public").
            local_peer_id: our libp2p peer-id on `p2p_name` (used for tie-breaker).
            local_node_id: our application node-id (passed into each Session).
            local_profile_factory: () -> NodeProfile dict snapshot (JSON-serializable).
            decision_hook: responder side — async callback to accept/reject an inbound
                PROFILE_OFFER and (for a world host) produce the WorldGrant spec.
                None ⇒ Sessions accept by default (v1 agent happy-path).
        """
        self.conn = conn
        self.p2p_name = p2p_name
        self.local_peer_id = local_peer_id
        self.local_node_id = local_node_id
        self._local_profile_factory = local_profile_factory
        self._decision_hook = decision_hook
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
        grant: Optional[dict] = None,
        reject_relay: bool = False,
    ) -> List[SessionEvent]:
        """Open a new initiator Session to `peer_id`. Idempotent if the peer
        already has a healthy Session — returns []. If a stale FAILED/CLOSED
        Session exists, it is replaced."""
        now = time.monotonic()
        # Quarantine check.
        ban_until = self._quarantine.get(peer_id, 0.0)
        if ban_until > now:
            log.session(f"initiate(...{peer_id[-5:]}) refused: quarantined for {ban_until - now:.1f}s", sub=self._label)
            ev = SessionEvent(
                type=SessionEventType.FAILED, session_id="",
                peer_id=peer_id, role=SessionRole.INITIATOR,
                p2p_name=self.p2p_name, data={"reason": "quarantined"},
            )
            return [ev]

        existing = self._sessions.get(peer_id)
        if existing is not None and existing.state not in (SessionState.FAILED, SessionState.CLOSED):
            log.session(f"initiate(...{peer_id[-5:]}): "
                      f"already have Session in state {existing.state.value} — no-op", sub=self._label)
            return []

        sess = Session(
            peer_id=peer_id,
            role=SessionRole.INITIATOR,
            conn=self.conn,
            p2p_name=self.p2p_name,
            local_peer_id=self.local_peer_id,
            local_node_id=self.local_node_id,
            local_profile_factory=self._local_profile_factory,
            remote_node_id_hint=remote_node_id_hint,
            remote_other_peer_id_hint=remote_other_peer_id_hint,
            addresses=addresses,
            is_world_join=is_world_join,
            decision_hook=self._decision_hook,
            grant=grant,
            reject_relay=reject_relay,
        )
        self._sessions[peer_id] = sess
        log.session(f"initiating Session to ...{peer_id[-5:]}", sub=self._label)
        events = await sess.start()
        self._evict_if_terminal(peer_id, sess)
        return events

    # ── Inbound dispatch ──────────────────────────────────────────────────────

    async def deliver(self, peer_id: str, protocol_id: str, sess_msg: SessMsg) -> List[SessionEvent]:
        """Route a session message at this manager's p2p instance. Handles the
        tie-breaker on inbound HELLO from an unknown / contending peer."""
        if protocol_id not in SESSION_PROTOCOL_IDS:
            log.error(f"deliver: unknown protocol {protocol_id}, dropping", sub=self._label)
            return []

        sess = self._sessions.get(peer_id)

        # A terminal Session must not shadow new traffic from the peer (an
        # inbound HELLO routed to a dead FSM is silently ignored, blocking any
        # re-handshake initiated by the remote): evict it and fall through to
        # the no-session paths.
        if sess is not None and sess.state in (SessionState.FAILED, SessionState.CLOSED):
            self._sessions.pop(peer_id, None)
            sess = None

        if sess is None:
            if protocol_id == SESSION_HELLO_PROTOCOL and sess_msg.kind == SessMsg.KIND_HELLO_REQ:
                # Inbound HELLO from an unknown peer → spawn a responder Session
                # (no tie-breaker possible: we have no in-flight initiator for them).
                sess = self._spawn_responder(peer_id, remote_other_peer_id_hint=None)
            elif (protocol_id == SESSION_CONTROL_PROTOCOL and
                  sess_msg.kind in (SessMsg.KIND_PING, SessMsg.KIND_RESUME)):
                # We are (most likely) the restarted peer: we hold no Session for
                # this session_id. Tell the initiator so it stops waiting and
                # re-handshakes, instead of dropping silently (which would make it
                # ride its grace timer to LOST). Only the manager knows "no Session
                # exists for this peer", so the reply lives here.
                await self._send_unknown_session_reply(peer_id, sess_msg)
                return []
            else:
                log.session(f"{protocol_id}/{sess_msg.kind} from unknown peer "
                          f"...{peer_id[-5:]} — dropped", sub=self._label)
                return []

        # Tie-breaker — only when the inbound is a competing HelloRequest (its
        # explicit kind tag distinguishes it from the HelloResponse coming back to
        # our own dial; both travel on the same protocol_id).
        elif (protocol_id == SESSION_HELLO_PROTOCOL and
              sess_msg.kind == SessMsg.KIND_HELLO_REQ and
              sess.role == SessionRole.INITIATOR and
              sess.state in (SessionState.CONNECTING, SessionState.HELLO_SENT,
                             SessionState.RECONNECTING)):
            events = await self._handle_simultaneous_open(peer_id)
            if events == "_proceed_with_existing_":   # type: ignore[comparison-overlap]
                # We kept the existing initiator; the inbound HELLO is dropped after
                # we wrote a soft-reject to the peer (handled inside the helper).
                return []
            elif events == "_replaced_with_responder_":   # type: ignore[comparison-overlap]
                sess = self._sessions[peer_id]
                # Fall through and process the inbound HELLO against the new responder.
            else:
                return events  # type: ignore[return-value]

        events = await sess.process_inbound(protocol_id, sess_msg)
        self._evict_if_terminal(peer_id, sess)
        return events

    def _evict_if_terminal(self, peer_id: str, sess: Session) -> None:
        """Drop a Session that just reached FAILED/CLOSED outside the tick path,
        so the registry never holds a dead FSM that would shadow later inbound
        traffic from the same peer. Quarantine stays a tick-path concern."""
        if (sess.state in (SessionState.FAILED, SessionState.CLOSED) and
                self._sessions.get(peer_id) is sess):
            self._sessions.pop(peer_id, None)

    def _spawn_responder(self, peer_id: str, remote_other_peer_id_hint: Optional[str]) -> Session:
        sess = Session(
            peer_id=peer_id,
            role=SessionRole.RESPONDER,
            conn=self.conn,
            p2p_name=self.p2p_name,
            local_peer_id=self.local_peer_id,
            local_node_id=self.local_node_id,
            local_profile_factory=self._local_profile_factory,
            remote_other_peer_id_hint=remote_other_peer_id_hint,
            decision_hook=self._decision_hook,
        )
        self._sessions[peer_id] = sess
        log.session(f"spawned responder Session for ...{peer_id[-5:]}", sub=self._label)
        return sess

    async def _send_unknown_session_reply(self, peer_id: str, sess_msg: SessMsg) -> None:
        """Reply to a liveness PING/RESUME for which we hold no Session (we are the
        restarted/forgetful peer): PONG ok=false / RESUME_ACK accepted=false with
        SESSION_UNKNOWN, echoing session_id + nonce. Mirrors the simultaneous-open
        reject pattern — a one-shot send on the control protocol, no Session object."""
        if sess_msg.kind == SessMsg.KIND_PING:
            reply = SessMsg(SessMsg.KIND_PONG, {
                "session_id": bytes(sess_msg.get("session_id") or b""),
                "nonce": bytes(sess_msg.get("nonce") or b""),
                "ok": False,
            })
        else:  # KIND_RESUME
            reply = SessMsg(SessMsg.KIND_RESUME_ACK, {
                "session_id": bytes(sess_msg.get("session_id") or b""),
                "nonce": bytes(sess_msg.get("nonce") or b""),
                "accepted": False, "reason_code": SESSION_UNKNOWN,
            })
        try:
            await self.conn.send_session(
                peer_id=peer_id, protocol_id=SESSION_CONTROL_PROTOCOL,
                sess_msg=reply, p2p_name=self.p2p_name,
            )
        except Exception as e:
            log.error(f"unknown-session reply to ...{peer_id[-5:]} failed: {e}", sub=self._label)

    async def _handle_simultaneous_open(self, peer_id: str):
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
                reject = SessMsg(SessMsg.KIND_HELLO_RESP, {
                    "chosen_version": "", "capabilities": 0, "nonce_b": b"",
                    "sig_b": b"", "reason_code": SIMULTANEOUS_OPEN_LOSS,
                })
                await self.conn.send_session(
                    peer_id=peer_id,
                    protocol_id=SESSION_HELLO_PROTOCOL,
                    sess_msg=reject,
                    p2p_name=self.p2p_name,
                )
            except Exception as e:
                log.error(f"simultaneous-open reject to ...{peer_id[-5:]} failed: {e}", sub=self._label)
            log.session(f"simultaneous-open with ...{peer_id[-5:]}: kept initiator", sub=self._label)
            return "_proceed_with_existing_"

        # We yield: drop our initiator, spawn a fresh responder for the inbound HELLO.
        log.session(f"simultaneous-open with ...{peer_id[-5:]}: yielding to responder", sub=self._label)
        existing.abort_silently("simultaneous_open_yielded")
        del self._sessions[peer_id]
        self._spawn_responder(peer_id, remote_other_peer_id_hint=existing.remote_other_peer_id_hint)
        return "_replaced_with_responder_"

    # ── Per-cycle tick (timeouts + keep-alive) ────────────────────────────────

    async def tick(self, now: Optional[float] = None) -> List[SessionEvent]:
        """Sweep all sessions: timeout firings + active re-dial + session-level
        keep-alive. Single dispatch path; the returned list is what node.py acts on."""
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
            r_event = await sess.tick_reconnect(now)
            if r_event is not None:
                events.append(r_event)
            k_event = await sess.tick_keepalive(now)
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
        reason_code: int = GRACEFUL_SHUTDOWN,
    ) -> List[SessionEvent]:
        sess = self._sessions.get(peer_id)
        if sess is None:
            log.session(f"close_session(...{peer_id[-5:]}): no Session — no-op", sub=self._label)
            return []
        await sess.close(reason_code)
        return []  # CLOSED event will arrive via process_inbound(GoawayAck) or tick

    async def close_all(self, reason_code: int = GRACEFUL_SHUTDOWN) -> List[SessionEvent]:
        """GOAWAY every Session that can be gracefully closed (ESTABLISHED /
        RECONNECTING). Used on node/runner shutdown so peers see a clean CLOSED
        instead of falling through liveness-timeout. Sessions already CLOSING/CLOSED
        (e.g. an intentional world-handoff GOAWAY) are skipped by Session.close."""
        for sess in list(self._sessions.values()):
            try:
                await sess.close(reason_code)
            except Exception as e:
                log.error(f"close_all: close(...{sess.peer_id[-5:]}) failed: {e}", sub=self._label)
        return []

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_session(self, peer_id: str) -> Optional[Session]:
        return self._sessions.get(peer_id)

    def is_established(self, peer_id: str) -> bool:
        s = self._sessions.get(peer_id)
        return s is not None and s.state == SessionState.ESTABLISHED

    def all_established_peer_ids(self) -> List[str]:
        return [pid for pid, s in self._sessions.items()
                if s.state == SessionState.ESTABLISHED]

    def has_active_sessions(self) -> bool:
        """True while any Session is ESTABLISHED or riding its reconnect grace.
        The Node uses this to keep the main loop at full speed: the keep-alive
        RTT window assumes a responsive pump, so a node with live sessions must
        never be treated as idle."""
        return any(s.state in (SessionState.ESTABLISHED, SessionState.RECONNECTING)
                   for s in self._sessions.values())

    def __len__(self) -> int:
        return len(self._sessions)

    def __repr__(self) -> str:
        by_state: Dict[str, int] = {}
        for s in self._sessions.values():
            by_state[s.state.value] = by_state.get(s.state.value, 0) + 1
        return f"SessionManager(p2p={self.p2p_name} sessions={by_state} quarantined={len(self._quarantine)})"
