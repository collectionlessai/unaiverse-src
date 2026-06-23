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
    from unaiverse.networking.p2p.p2p import P2P

# Anti-flap quarantine window after a LOST/FAILED. Kept below the world rejoin delay
# (Custom.REJOINING_WAIT, 30s) so a scheduled rejoin lands after the window has expired
# instead of racing it and bouncing on the gate.
_QUARANTINE_S = 25.0


def _p2p_label(name: str) -> str:
    return "pub" if "public" in name else "prv" if "world" in name else name


class SessionManager:
    """Per-p2p-instance Session registry and dispatcher."""

    def __init__(
        self,
        p2p: "P2P",
        p2p_name: str,
        local_peer_id: str,
        local_node_id: str,
        local_profile_factory: Callable[[], dict],
        decision_hook: Optional[DecisionHook] = None,
        token_verifier: Optional[Callable] = None,
        token_factory: Optional[Callable[[], str]] = None,
        routing_classifier=None,
    ):
        """
        Args:
            p2p: the P2P instance this manager owns and injects into each Session
                (or a JS OP2P mirror in Pyodide). The manager no longer reaches the
                transport through the conn layer, it speaks to its P2P directly.
            p2p_name: the label of the P2P instance this manager owns (e.g.
                "p2p_public"), used for SessionEvent tagging.
            local_peer_id: our libp2p peer-id on `p2p_name` (used for tie-breaker).
            local_node_id: our application node-id (passed into each Session).
            local_profile_factory: () -> NodeProfile dict snapshot (JSON-serializable).
            decision_hook: responder side — async callback to accept/reject an inbound
                PROFILE_OFFER and (for a world host) produce the WorldGrant spec.
                None ⇒ Sessions accept by default (v1 agent happy-path).
        """
        self.p2p = p2p
        self.p2p_name = p2p_name
        self.local_peer_id = local_peer_id
        self.local_node_id = local_node_id
        self._local_profile_factory = local_profile_factory
        self._decision_hook = decision_hook
        # Token capabilities (RT plumbing), forwarded into every Session this manager
        # builds. token_verifier(token, p2p_peer) -> (node_id, cv_hash); token_factory()
        # -> our current node token (re-sampled per handshake so periodic refresh shows).
        self._token_verifier = token_verifier
        self._token_factory = token_factory
        # Last node token broadcast to peers; tick() pushes a TOKEN_UPDATE when
        # token_factory() (i.e. self.node_token) changes. Seed with the current value
        # so the initial token is not re-broadcast.
        self._last_pushed_token = token_factory() if token_factory is not None else None
        # Per-instance perimeter routing, injected by the Coordinator (which owns the
        # pool taxonomy): a pure classifier (peer_info, membership) -> pool_name | None.
        # The unclassifiable-world-peer cut is no longer an injected callable: the manager
        # owns its P2P, so it disconnects the peer itself (self.disconnect), keeping the
        # Node -> NodeConn -> SM -> P2P chain intact.
        self._routing_classifier = routing_classifier
        self._label = _p2p_label(p2p_name)

        # peer_id → Session (at most one Session per peer in this p2p instance).
        # Two views: `_sessions` is the full registry (including inert KNOWN
        # Sessions for known-but-not-dialed peers); `_active_sessions` mirrors only
        # the ones the per-cycle tick must sweep (anything past KNOWN), so a large
        # roster of KNOWN peers costs O(1) per cycle instead of O(N).
        self._sessions: Dict[str, Session] = {}
        self._active_sessions: Dict[str, Session] = {}

        # peer_id → monotonic timestamp until which dials are refused.
        self._quarantine: Dict[str, float] = {}

        # Pure world membership, owned here so it sits next to the Sessions it gates.
        # Meaningful only on the world instance, where the Coordinator populates these
        # from the rendezvous and the world-host writers; inert (empty) on the public
        # instance. The Coordinator reads them back through membership_snapshot() to build
        # the per-cycle WorldMembership the perimeter classifies against. The role/addrs
        # indices and inspector_peer_id stay Coordinator-owned (D-2 / D-8a): only the two
        # sets and the world node's peer-id live here.
        self.world_agents_set: set = set()
        self.world_masters_set: set = set()
        self.world_node_peer_id: Optional[str] = None

    # ── Session registry (keep the two views consistent) ──────────────────────
    #
    # Every mutation of the session store goes through these helpers so the
    # tick-active view never drifts from the full registry.

    def _set_known(self, peer_id: str, sess: "Session") -> None:
        """Register/replace an inert KNOWN Session: in the full registry, out of the
        tick-active view."""
        self._sessions[peer_id] = sess
        self._active_sessions.pop(peer_id, None)

    def _set_active(self, peer_id: str, sess: "Session") -> None:
        """Register a Session the tick must sweep (a responder, or a started initiator)."""
        self._sessions[peer_id] = sess
        self._active_sessions[peer_id] = sess

    def _promote(self, peer_id: str, sess: "Session") -> None:
        """Move a just-started initiator out of KNOWN into the tick-active view."""
        if self._sessions.get(peer_id) is sess and sess.state != SessionState.KNOWN:
            self._active_sessions[peer_id] = sess

    def _drop_session(self, peer_id: str) -> None:
        """Remove a Session from both views."""
        self._sessions.pop(peer_id, None)
        self._active_sessions.pop(peer_id, None)

    # ── Per-instance reconcile (routing + perimeter) ──────────────────────────

    async def reconcile(self, connected_peer_infos: list, membership) -> dict:
        """Classify this instance's connected peers into pools, returning a dict
        ``pool_name -> {peer_id -> peer_info}``.

        Uses the injected per-instance classifier. When a membership snapshot is given
        (the world instance) an unclassifiable peer is cut from the perimeter
        (disconnected) and left out; the public instance has no membership and never
        cuts (an undefined-direction peer is simply skipped). Empty pools are absent
        from the result: the Coordinator iterates its own pools and treats a missing
        pool as empty, so removals are still detected.
        """
        classified: dict = {}
        if connected_peer_infos is None or self._routing_classifier is None:
            return classified
        for c in connected_peer_infos:
            pool_name = self._routing_classifier(c, membership)
            if pool_name is None:
                if membership is not None:
                    log.session(f"perimeter: cutting unclassifiable peer ...{c['id'][-5:]}", sub=self._label)
                    await self.disconnect(c['id'])
                continue
            classified.setdefault(pool_name, {})[c['id']] = c
        return classified

    async def disconnect(self, peer_id: str) -> bool:
        """Disconnect a peer on THIS instance's P2P (best-effort): the manager owns its
        P2P handle, so the perimeter cut and the Coordinator's remove() route through here
        instead of reaching the transport from the conn layer (Node -> NodeConn -> SM ->
        P2P). Returns False if the transport refused, True otherwise. Best-effort: any
        transport error means 'gone', and the manager stays free of the P2P error type
        (portability)."""
        try:
            self.p2p.disconnect_from(peer_id)
        except Exception:
            return False
        return True

    # ── Raw transport passthroughs (S14b) ─────────────────────────────────────
    #
    # The manager owns its P2P, so the Coordinator's chat/relay facades call these
    # instead of reaching the transport directly (Node -> NodeConn -> SM -> P2P). They
    # are thin sync wrappers over the P2P handle: the Msg encoding and the piggyback
    # token stay above on the Coordinator (the manager stays protobuf-free, I-3), and
    # the P2P error type is not caught here (the caller handles it).

    def send_message_to_peer(self, channel: str, msg_bytes: bytes) -> None:
        self.p2p.send_message_to_peer(channel, msg_bytes=msg_bytes)

    def broadcast_message(self, channel: str, msg_bytes: bytes) -> None:
        self.p2p.broadcast_message(channel, msg_bytes=msg_bytes)

    def subscribe_to_topic(self, channel: str) -> None:
        self.p2p.subscribe_to_topic(channel)

    def unsubscribe_from_topic(self, channel: str) -> None:
        self.p2p.unsubscribe_from_topic(channel)

    def reserve_on_relay(self, peer_id: str):
        return self.p2p.reserve_on_relay(peer_id)

    def sync_known_peers(self, known: Dict[str, tuple]) -> None:
        """Reconcile the inert KNOWN roster against the peers currently advertised by
        the rendezvous (``peer_id -> (addresses, app_role)``). Registers a KNOWN Session
        for every known peer that has no Session yet, refreshes the rendezvous-sourced
        metadata (addresses + applicative role) on the Sessions that already exist, and
        prunes KNOWN Sessions whose peer has vanished.

        The metadata refresh touches a Session's addresses/app_role only, never its FSM
        state, so an active (established / in-transit) Session keeps running while its
        role/addrs stay current. Pruning is still KNOWN-only: an ESTABLISHED peer that
        drops out of the rendezvous keeps its live Session until the transport tears it
        down, and a FAILED / CLOSED one is left to the normal evict path. KNOWN Sessions
        live solely in the registry, never in a pool, so this cannot perturb the
        per-cycle diff or the perimeter."""
        for peer_id, (addrs, app_role) in known.items():
            addrs = list(addrs) if addrs else []
            sess = self._sessions.get(peer_id)
            if sess is None:
                self.ensure(peer_id, addrs, app_role=app_role)
            else:
                sess.addresses = addrs
                sess.app_role = app_role
        stale = [pid for pid, s in self._sessions.items()
                 if s.state == SessionState.KNOWN and pid not in known]
        for peer_id in stale:
            self._drop_session(peer_id)

    # ── Initiator API ─────────────────────────────────────────────────────────

    def ensure(
        self,
        peer_id: str,
        addresses: List[str],
        is_world_join: bool = False,
        remote_node_id_hint: Optional[str] = None,
        grant: Optional[dict] = None,
        reject_relay: bool = False,
        app_role: int = 0,
    ) -> Session:
        """Register an initiator Session to `peer_id` in the KNOWN state WITHOUT
        starting it (no dial yet). Idempotent: a non-terminal existing Session is
        returned unchanged; a FAILED/CLOSED one is replaced. ``activate`` dials it
        later. This lets a caller know a peer without committing to a connection."""
        existing = self._sessions.get(peer_id)
        if existing is not None and existing.state not in (SessionState.FAILED, SessionState.CLOSED):
            return existing
        sess = Session(
            peer_id=peer_id,
            role=SessionRole.INITIATOR,
            p2p=self.p2p,
            p2p_name=self.p2p_name,
            local_peer_id=self.local_peer_id,
            local_node_id=self.local_node_id,
            local_profile_factory=self._local_profile_factory,
            remote_node_id_hint=remote_node_id_hint,
            addresses=addresses,
            is_world_join=is_world_join,
            decision_hook=self._decision_hook,
            grant=grant,
            reject_relay=reject_relay,
            app_role=app_role,
            token_verifier=self._token_verifier,
            token_factory=self._token_factory,
        )
        self._set_known(peer_id, sess)
        return sess

    async def activate(self, peer_id: str) -> List[SessionEvent]:
        """Start the (initiator) Session for `peer_id`: dial + HELLO, then evict it if
        the dial fails terminally. Returns the events from start (typically [] until
        ESTABLISHED arrives). No-op ([]) if there is no Session for the peer."""
        sess = self._sessions.get(peer_id)
        if sess is None:
            return []
        # Quarantine gate (single dial door): a peer that recently went LOST/FAILED is
        # refused immediately, without dialing, until its window expires. Non-blocking:
        # we return FAILED(quarantined) at once and leave the (inert KNOWN) Session in
        # place, so a later activate after the window can dial it.
        now = time.monotonic()
        ban_until = self._quarantine.get(peer_id, 0.0)
        if ban_until > now:
            log.session(f"activate(...{peer_id[-5:]}) refused: quarantined for {ban_until - now:.1f}s", sub=self._label)
            return [SessionEvent(
                type=SessionEventType.FAILED, session_id="",
                peer_id=peer_id, role=SessionRole.INITIATOR,
                p2p_name=self.p2p_name, data={"reason": "quarantined"},
            )]
        log.session(f"activating Session to ...{peer_id[-5:]}", sub=self._label)
        events = await sess.start()
        self._promote(peer_id, sess)
        self._evict_if_terminal(peer_id, sess)
        return events

    async def initiate(
        self,
        peer_id: str,
        addresses: List[str],
        is_world_join: bool = False,
        remote_node_id_hint: Optional[str] = None,
        grant: Optional[dict] = None,
        reject_relay: bool = False,
    ) -> List[SessionEvent]:
        """Open a new initiator Session to `peer_id` and start it. Idempotent if the
        peer already has a healthy Session — returns []. A stale FAILED/CLOSED Session
        is replaced. initiate() = ensure() + activate()."""
        # The quarantine gate lives in activate() (the single dial door), so a
        # quarantined peer is refused there, even on a direct activate.
        existing = self._sessions.get(peer_id)
        app_role = 0
        if existing is not None and existing.state == SessionState.KNOWN:
            # A KNOWN is a rendezvous placeholder, not a dial in flight: drop it so the
            # caller's addresses / grant / is_world_join / hints take effect below. Carry
            # its rendezvous role across the replacement so the dialed Session is not
            # blank on app_role until the next sync refresh.
            app_role = existing.app_role
            self._drop_session(peer_id)
        elif existing is not None and existing.state not in (SessionState.FAILED, SessionState.CLOSED):
            log.session(f"initiate(...{peer_id[-5:]}): "
                      f"already have Session in state {existing.state.value} — no-op", sub=self._label)
            return []

        self.ensure(peer_id, addresses, is_world_join=is_world_join,
                    remote_node_id_hint=remote_node_id_hint, grant=grant, reject_relay=reject_relay,
                    app_role=app_role)
        log.session(f"initiating Session to ...{peer_id[-5:]}", sub=self._label)
        return await self.activate(peer_id)

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
            self._drop_session(peer_id)
            sess = None

        # A KNOWN Session could be an initiator that has not dialed yet: it is not an
        # in-flight handshake, so route inbound traffic as if there were no Session.
        # The peer dialed us first, so an inbound HELLO makes us the responder
        # (spawn_responder overwrites the KNOWN registry entry); a KNOWN never reaches
        # the tie-breaker below because it didn't dial yet, and on PING/RESUME
        # the no-session path replies "unknown session" while the KNOWN intent stays put for a later dial.
        if sess is None or sess.state == SessionState.KNOWN:
            if protocol_id == SESSION_HELLO_PROTOCOL and sess_msg.kind == SessMsg.KIND_HELLO_REQ:
                # Unknown peer, or a KNOWN intent that never dialed → spawn a responder
                # Session (no tie-breaker possible: no in-flight initiator for them).
                sess = self._spawn_responder(peer_id)
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
        # our own dial; both travel on the same protocol_id). A KNOWN Session is also
        # role=INITIATOR, but it is not an in-flight dial: the state guard keeps it out
        # of the tie-breaker (and the no-session branch above has already handled it).
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
            self._drop_session(peer_id)

    def _spawn_responder(self, peer_id: str) -> Session:
        sess = Session(
            peer_id=peer_id,
            role=SessionRole.RESPONDER,
            p2p=self.p2p,
            p2p_name=self.p2p_name,
            local_peer_id=self.local_peer_id,
            local_node_id=self.local_node_id,
            local_profile_factory=self._local_profile_factory,
            decision_hook=self._decision_hook,
            token_verifier=self._token_verifier,
            token_factory=self._token_factory,
        )
        self._set_active(peer_id, sess)
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
            await self.p2p.send_session(
                peer_id=peer_id, protocol_id=SESSION_CONTROL_PROTOCOL,
                sess_msg=reply,
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
                await self.p2p.send_session(
                    peer_id=peer_id,
                    protocol_id=SESSION_HELLO_PROTOCOL,
                    sess_msg=reject,
                )
            except Exception as e:
                log.error(f"simultaneous-open reject to ...{peer_id[-5:]} failed: {e}", sub=self._label)
            log.session(f"simultaneous-open with ...{peer_id[-5:]}: kept initiator", sub=self._label)
            return "_proceed_with_existing_"

        # We yield: drop our initiator, spawn a fresh responder for the inbound HELLO.
        log.session(f"simultaneous-open with ...{peer_id[-5:]}: yielding to responder", sub=self._label)
        existing.abort_silently("simultaneous_open_yielded")
        self._drop_session(peer_id)
        self._spawn_responder(peer_id)
        return "_replaced_with_responder_"

    # ── Per-cycle tick (timeouts + keep-alive) ────────────────────────────────

    async def tick(self, now: Optional[float] = None) -> List[SessionEvent]:
        """Sweep all sessions: timeout firings + active re-dial + session-level
        keep-alive. Single dispatch path; the returned list is what node.py acts on."""
        if now is None:
            now = time.monotonic()
        events: List[SessionEvent] = []

        # Our root token refreshed (token_factory reflects self.node_token): push it
        # in-band to every ESTABLISHED peer so their cached expiry advances without a
        # re-handshake. Detected by comparing against the last pushed value.
        if self._token_factory is not None:
            tok = self._token_factory()
            if tok != self._last_pushed_token:
                self._last_pushed_token = tok
                for sess in list(self._active_sessions.values()):
                    await sess.send_token_update(tok)

        # Snapshot of values; mutation during iteration is safe because we operate
        # on the snapshot only (per-session state changes happen synchronously and
        # don't add new sessions inside tick).
        for sess in list(self._active_sessions.values()):
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
            # Quarantine a peer that just failed (handshake/timeout) or was lost
            # (reconnect grace expired), so we don't immediately re-dial it. An
            # intentional CLOSED is not a failure and is not quarantined.
            if ev.type in (SessionEventType.LOST, SessionEventType.FAILED):
                self._quarantine[ev.peer_id] = now + _QUARANTINE_S
            if ev.type in (SessionEventType.LOST,
                           SessionEventType.CLOSED,
                           SessionEventType.FAILED):
                # Only drop from registry if the session is genuinely terminal.
                sess = self._sessions.get(ev.peer_id)
                if sess is not None and sess.state in (SessionState.FAILED, SessionState.CLOSED):
                    self._drop_session(ev.peer_id)

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
                          SessionState.AWAITING_HANDOFF):
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

    def is_in_grace(self, peer_id: str) -> bool:
        """True if this peer's Session is riding its reconnect grace window
        (RECONNECTING): disconnected at the transport level but not yet gone. The
        Coordinator aggregates this across its instances so the Node can choose
        pool-remove (keep the enrolment across the blip) vs full purge without reaching
        into Session state itself."""
        sess = self._sessions.get(peer_id)
        return sess is not None and sess.state == SessionState.RECONNECTING

    def is_handshake_in_flight(self, peer_id: str) -> bool:
        """True if this peer has a Session whose FIRST handshake is still in flight: it
        exists but has never reached ESTABLISHED, and is neither a rendezvous-only KNOWN
        nor RECONNECTING (already established once) nor terminal (FAILED/CLOSED). This is
        the session-era 'connected but not yet managed' notion the Coordinator aggregates
        for Agent.agents_are_waiting. The classification lives here, next to SessionState,
        so the Coordinator stays free of the session enum."""
        sess = self._sessions.get(peer_id)
        if sess is None:
            return False
        return sess.state not in (SessionState.KNOWN, SessionState.ESTABLISHED,
                                  SessionState.RECONNECTING, SessionState.FAILED,
                                  SessionState.CLOSED)

    def has_active_sessions(self) -> bool:
        """True while any Session is ESTABLISHED or riding its reconnect grace.
        The Node uses this to keep the main loop at full speed: the keep-alive
        RTT window assumes a responsive pump, so a node with live sessions must
        never be treated as idle."""
        return any(s.state in (SessionState.ESTABLISHED, SessionState.RECONNECTING)
                   for s in self._active_sessions.values())

    def requires_fast_clock(self) -> bool:
        """True while any Session needs the main loop at full speed: every
        tick-active state except the terminals FAILED and CLOSED (KNOWN Sessions
        are never in the tick-active view, so they are excluded for free). Unlike
        has_active_sessions, which counts only ESTABLISHED/RECONNECTING, this also
        covers handshakes in flight (CONNECTING/HELLO_*/PROFILE_SENT/AWAITING_*)
        and CLOSING, whose RTT-sized timers would be starved by a slowed clock."""
        return any(s.state not in (SessionState.FAILED, SessionState.CLOSED)
                   for s in self._active_sessions.values())

    def __len__(self) -> int:
        return len(self._sessions)

    def __repr__(self) -> str:
        by_state: Dict[str, int] = {}
        for s in self._sessions.values():
            by_state[s.state.value] = by_state.get(s.state.value, 0) + 1
        return f"SessionManager(p2p={self.p2p_name} sessions={by_state} quarantined={len(self._quarantine)})"
