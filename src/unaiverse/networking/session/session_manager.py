"""
       █████  █████ ██████   █████           █████ █████   █████ ██████████ ███████████    █████████  ██████████
      ░░███  ░░███ ░░██████ ░░███           ░░███ ░░███   ░░███ ░░███░░░░░█░░███░░░░░███  ███░░░░░███░░███░░░░░█
       ░███   ░███  ░███░███ ░███   ██████   ░███  ░███    ░███  ░███  █ ░  ░███    ░███ ░███    ░░░  ░███  █ ░ 
       ░███   ░███  ░███░░███░███  ░░░░░███  ░███  ░███    ░███  ░██████    ░██████████  ░░█████████  ░██████   
       ░███   ░███  ░███ ░░██████   ███████  ░███  ░░███   ███   ░███░░█    ░███░░░░░███  ░░░░░░░░███ ░███░░█   
       ░███   ░███  ░███  ░░█████  ███░░███  ░███   ░░░█████░    ░███ ░   █ ░███    ░███  ███    ░███ ░███ ░   █
       ░░████████   █████  ░░█████░░████████ █████    ░░███      ██████████ █████   █████░░█████████  ██████████
        ░░░░░░░░   ░░░░░    ░░░░░  ░░░░░░░░ ░░░░░      ░░░      ░░░░░░░░░░ ░░░░░   ░░░░░  ░░░░░░░░░  ░░░░░░░░░░ 
                 A Collectionless AI Project (https://collectionless.ai)
                 Registration/Login: https://unaiverse.io
                 Code Repositories:  https://github.com/collectionlessai/
                 Main Developers:    Stefano Melacci (Project Leader), Christian Di Maio, Tommaso Guidi
"""
import time
from unaiverse.utils.logger import log
from typing import Any, Callable, TYPE_CHECKING
from unaiverse.networking.session.sess_msg import SessMsg, reason_name
from unaiverse.networking.session.session import SessionEvent, SessionEventType, DecisionHook
from unaiverse.networking.session.session import SESSION_HELLO_PROTOCOL, SESSION_CONTROL_PROTOCOL
from unaiverse.networking.session.session import SESSION_PROTOCOL_IDS, Session, SessionState, SessionRole
from unaiverse.networking.session.sess_msg import SIMULTANEOUS_OPEN_LOSS, GRACEFUL_SHUTDOWN, SESSION_UNKNOWN

# This import ONLY happens during IDE/Type inspection
if TYPE_CHECKING:
    from unaiverse.networking.p2p.p2p import P2P

# Anti-flap quarantine window after a LOST/FAILED. Kept below the world rejoin delay
# (Custom.REJOINING_WAIT, 30s) so a scheduled rejoin lands after the window has expired
# instead of racing it and bouncing on the gate.
_QUARANTINE_S = 25.0

# Short variant for handshake-stall failures (timeout:hello_sent /
# timeout:hello_ack_sent): the peer's pump missed one handshake window, which is
# typically slowness under load, not flapping, and the full window would make any
# retry scheduled under it bounce off the dial gate. Must stay above _CUT_LINGER_S
# so the settlement cut of the failed handshake still precedes the next dial.
_QUARANTINE_STALL_S = 5.0

# Settlement linger before the deferred transport cut of a peer whose Session just
# reached a terminal state. The transport must outlive the Session by a beat because
# the last farewell frame (a GOAWAY_ACK, a reject decision, an unknown-session NACK)
# has only been written to the local muxer when the terminal event fires, and
# P2P.disconnect is abortive (it resets the stream, destroying undelivered frames).
# Bounds: > one WAN round-trip + multistream negotiation, relay path included;
# < the CLOSING window (10s) and < the quarantine (25s).
_CUT_LINGER_S = 2.0

# Linger before the transport cut of a peer that was answered on behalf of no Session
# (PONG ok=false, refused RESUME, GOAWAY_ACK). No Session of ours will ever settle that
# transport, so with no cut the peer stays connected, and in the pools, for as long as
# the connection lives. The reply also tells the peer to start over: an initiator answers
# a negative PONG with RESUME and the refusal of that with a fresh HELLO, which cancels
# the cut and keeps the transport. The linger is the time given to that HELLO, the same
# a responder told that its Session is gone waits for it (_TIMEOUT_PEER_LOST_SESSION_S in
# session.py), sized like it for a slow peer pump and not for the wire.
_UNKNOWN_REPLY_LINGER_S = 10.0


def _p2p_label(name: str) -> str:
    """Gets the short log sub-label of a p2p instance name.

    Args:
        name: The name of the p2p instance (e.g. "p2p_public", "p2p_world").

    Returns:
        "pub" for the public instance, "prv" for the world instance, otherwise the name itself.
    """
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
        decision_hook: DecisionHook | None = None,
        token_verifier: Callable | None = None,
        token_factory: Callable[[], str] | None = None,
        routing_classifier: Callable[[dict, Any], str | None] | None = None,
        address_lookup: Callable | None = None,
    ) -> None:
        """Initializes the Session registry and dispatcher of one p2p instance.

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
            token_verifier: async callable (token, p2p_peer) -> (node_id, cv_hash, p2p_peers, exp, code_hash),
                forwarded into every Session; None ⇒ no token check (v1 path).
            token_factory: () -> our current node token, forwarded into every Session; None ⇒ no token sent.
            routing_classifier: pure classifier (peer_info, membership) -> pool_name | None, injected by the
                Coordinator (which owns the pool taxonomy); None ⇒ no pool classification.
            address_lookup: initiator side — async node_id -> addresses of that node,
                handed to every initiator Session (see Session); None ⇒ no lookup.
        """
        self.p2p = p2p
        self.p2p_name = p2p_name
        self.local_peer_id = local_peer_id
        self.local_node_id = local_node_id
        self._local_profile_factory = local_profile_factory
        self._decision_hook = decision_hook

        # Token capabilities (RT plumbing), forwarded into every Session this manager
        # builds. token_verifier(token, p2p_peer) -> (node_id, cv_hash, p2p_peers, exp,
        # code_hash); token_factory() -> our current node token (re-sampled per
        # handshake so periodic refresh shows).
        self._token_verifier = token_verifier
        self._token_factory = token_factory
        self._address_lookup = address_lookup

        # Last node token broadcast to peers; tick() pushes a TOKEN_UPDATE when
        # token_factory() (i.e. self.node_token) changes. Seed with the current value
        # so the initial token is not re-broadcast.
        self._last_pushed_token = token_factory() if token_factory is not None else None

        # Per-instance perimeter routing, injected by the Coordinator (which owns the
        # pool taxonomy): a pure classifier (peer_info, membership) -> pool_name | None.
        # The unclassifiable-world-peer cut is no longer an injected callable: the manager
        # owns its P2P, so it drops the peer itself (self.drop), keeping the
        # Node -> NodeConn -> SM -> P2P chain intact.
        self._routing_classifier = routing_classifier
        self._label = _p2p_label(p2p_name)

        # peer_id → Session (at most one Session per peer in this p2p instance).
        # Two views: `_sessions` is the full registry (including inert KNOWN
        # Sessions for known-but-not-dialed peers); `_active_sessions` mirrors only
        # the ones the per-cycle tick must sweep (anything past KNOWN), so a large
        # roster of KNOWN peers costs O(1) per cycle instead of O(N).
        self._sessions: dict[str, Session] = {}
        self._active_sessions: dict[str, Session] = {}

        # peer_id → monotonic timestamp until which dials are refused.
        self._quarantine: dict[str, float] = {}

        # peer_id → monotonic deadline of a deferred transport cut (the settlement
        # cut), scheduled by _on_terminal when a Session ends, or after a reply sent
        # on behalf of no Session, and executed by tick once the linger expires. Only
        # settlement cuts live here: the coercive perimeter cut never goes through the
        # ledger (it must not be cancellable).
        # A live non-KNOWN Session found at execution time cancels the entry (lazy
        # guard), so a re-handshake that reused the transport keeps it.
        self._pending_cuts: dict[str, float] = {}

        # The membership snapshot of the last reconcile, None on an instance with no
        # perimeter. Read by the one cut that is scheduled on a frame of the peer and
        # not on the end of a Session, see _send_unknown_session_reply.
        self._perimeter_membership = None

        # Pure world membership, owned here so it sits next to the Sessions it gates.
        # Meaningful only on the world instance, where the Coordinator populates these
        # from the rendezvous and the world-host writers; inert (empty) on the public
        # instance. The Coordinator reads them back through membership_snapshot() to build
        # the per-cycle WorldMembership the perimeter classifies against. The role/addrs
        # indices and inspector_peer_id stay Coordinator-owned: only the two
        # sets and the world node's peer-id live here.
        self.world_agents_set: set = set()
        self.world_masters_set: set = set()
        self.world_node_peer_id: str | None = None

    async def reconcile(self, connected_peer_infos: list | None, membership: Any) -> dict:
        """Classifies this instance's connected peers into pools (async).

        Uses the injected per-instance classifier. When a membership snapshot is given (the world instance) an
        unclassifiable peer is cut from the perimeter with the COERCIVE cut: the reconcile cannot tell a revoked
        ex-member from an intruder, so the mesh isolation (I-1) wins over a graceful goodbye and the transport is
        severed at once. The public instance has no membership and never cuts (an undefined-direction peer is simply
        skipped) — the coercive path is de facto world-instance only. Peers already tearing down (CLOSING, or holding
        a pending settlement cut) are skipped BEFORE classification: they must not be re-admitted to the pools during
        their linger, nor cut a second time; the pool diff the Coordinator computes stays identical to the
        pre-settlement one. Empty pools are absent from the result: the Coordinator iterates its own pools and treats
        a missing pool as empty, so removals are still detected.

        Args:
            connected_peer_infos: The peer infos of the connected peers (None when unknown).
            membership: The world membership snapshot, or None on an instance with no perimeter.

        Returns:
            The dictionary ``pool_name -> {peer_id -> peer_info}``.
        """
        classified: dict = {}
        self._perimeter_membership = membership
        if connected_peer_infos is None or self._routing_classifier is None:
            return classified
        seen: set = set()
        for c in connected_peer_infos:
            seen.add(c['id'])
            if self.is_tearing_down(c['id']):
                continue
            pool_name = self._routing_classifier(c, membership)
            if pool_name is None:
                if membership is not None:
                    log.session(f"perimeter: cutting unclassifiable peer ...{c['id'][-5:]}", sub=self._label)
                    await self._coercive_cut(c['id'])
                continue
            classified.setdefault(pool_name, {})[c['id']] = c

        # connected_peer_infos only holds peers that opened a stream, so a peer whose only
        # traffic is a topic subscription is absent above yet still receives every broadcast.
        # On the world instance, also cut the non-member subscribers the stream loop did not
        # handle. These inform only the cut, never pool admission (which stays stream-driven),
        # so we pass bare ids and build no pool entries. Direction does not affect the member
        # test, so a synthetic value reuses the classifier. Relays never subscribe.
        if membership is not None:
            for peer_id in self.topic_peers():
                if peer_id in seen or peer_id == self.local_peer_id or self.is_tearing_down(peer_id):
                    continue  # already classified/cut by the stream loop, ourselves, or leaving
                if self._routing_classifier({'id': peer_id, 'direction': 'inbound'}, membership) is None:
                    log.session(f"perimeter: cutting pubsub-only non-member ...{peer_id[-5:]}", sub=self._label)
                    await self._coercive_cut(peer_id)

        return classified

    # Transport passthroughs
    #
    # The manager owns its P2P, so the Coordinator's chat/relay facades call these
    # instead of reaching the transport directly (Node -> NodeConn -> SM -> P2P). The chat
    # egress passes a Msg object (not bytes): the codec (Msg.to_bytes) and the thread
    # offload of the blocking Go call both live in P2P, so the manager stays protobuf-free
    # and free of the async boundary. The piggyback token stays above on the Coordinator,
    # and the P2P error type is not caught here (the caller handles it).
    #
    # The async boundary is uniform and lives only in P2P: the operations that block on a
    # round-trip (send, broadcast, reserve, disconnect) are async and offload there. The
    # instantaneous Go FFI calls (subscribe/unsubscribe, the peer-info getters,
    # get_messages) stay sync, they run in the hot polling loop and a thread hop would
    # only add latency.
    async def send(self, channel: str, msg: Any) -> int | None:
        """Sends a direct message through this instance's transport (async).

        The chat egress passes a Msg object (not bytes): the codec and the thread offload live in P2P.

        Args:
            channel: The direct-message channel.
            msg: The message object.

        Returns:
            What the transport returns: the sequence number of the queued message, or None (no send queues, or the
            browser transport).
        """
        return await self.p2p.send(channel, msg)

    def drop_send_queue(self, peer_id: str) -> int:
        """Empties the send queue of a peer on this instance's transport (a no-op where there is none).

        Args:
            peer_id: The peer.

        Returns:
            How many queued messages were dropped.
        """
        drop = getattr(self.p2p, "drop_send_queue", None)
        return drop(peer_id) if drop is not None else 0

    def pop_send_events(self) -> list[dict]:
        """Pops the events of the send queues of this instance's transport (none where there are no queues).

        Returns:
            The list of the events, oldest first.
        """
        pop = getattr(self.p2p, "pop_send_events", None)
        return pop() if pop is not None else []

    def set_peer_capabilities(self, peer_id: str, capabilities: int, max_message_size: int | None,
                              max_file_size: int | None = None) -> None:
        """Tells this instance's transport what a peer can receive (a no-op on a transport that does not care).

        Args:
            peer_id: The peer.
            capabilities: The CAP_* bits of its HELLO / HELLO_ACK.
            max_message_size: The largest message it takes, from its profile (0: no limit; None or absent: legacy).
            max_file_size: The largest file it takes, from its profile (0 or None: no limit).
        """
        setter = getattr(self.p2p, "set_peer_capabilities", None)
        if setter is not None:
            setter(peer_id, capabilities, max_message_size, max_file_size)

    def request_keepalive_probe(self, peer_id: str) -> None:
        """Asks the Session of a peer for its next keep-alive PING now, instead of at the end of the period.

        For a send that failed: a dead peer is then confirmed within the RTT window instead of up to a period later.

        Args:
            peer_id: The peer.
        """
        sess = self._sessions.get(peer_id)
        if sess is not None:
            sess.request_early_ping()

    async def broadcast(self, channel: str, msg: Any) -> None:
        """Publishes a message on a topic through this instance's transport (async).

        Args:
            channel: The pubsub channel.
            msg: The message object.
        """
        await self.p2p.broadcast(channel, msg)

    def subscribe_to_topic(self, channel: str) -> None:
        """Subscribes this instance to a topic (passthrough to the transport).

        Args:
            channel: The pubsub channel.
        """
        self.p2p.subscribe_to_topic(channel)

    def unsubscribe_from_topic(self, channel: str) -> None:
        """Unsubscribes this instance from a topic (passthrough to the transport).

        Args:
            channel: The pubsub channel.
        """
        self.p2p.unsubscribe_from_topic(channel)

    async def reserve(self, peer_id: str) -> str | None:
        """Reserves a slot on a relay through this instance's transport (async).

        Args:
            peer_id: The relay peer.

        Returns:
            The expiry of the reservation (ISO string), or None when no slot was granted.
        """
        return await self.p2p.reserve(peer_id)

    def set_turn_credentials(self, ice_turn_servers: list) -> None:
        """Updates this instance's TURN credentials (passthrough).

        Sync: it is a local ICE config update for future WebRTC connections, not a network round-trip.

        Args:
            ice_turn_servers: The TURN server descriptions.
        """
        self.p2p.set_turn_credentials(ice_turn_servers)

    def connected_peers_info(self) -> list | None:
        """Gets the transport-level connected peers on this instance (passthrough).

        The Coordinator reads peer info through the manager instead of reaching into the P2P handle. None on an FFI
        error (transport state unknown), so a transient failure is not read as "zero peers": the reconcile skips the
        diff on None instead of emptying the pools.

        Returns:
            The list of the connected peer infos, or None when the transport state is unknown.
        """
        return self.p2p.get_connected_peers_info()

    def topic_peers(self) -> list:
        """Gets the peer IDs subscribed to this instance's topics (passthrough).

        They are the directly-connected peers that receive its broadcasts. The world perimeter uses them to see the
        pubsub-only peers that open no stream and so never appear in connected_peers_info. Empty on transport error,
        so a transient FFI failure never causes a spurious cut.

        Returns:
            The list of the peer IDs.
        """
        return self.p2p.get_topic_peers()

    def rendezvous_peers(self) -> dict | None:
        """Gets the rendezvous advertisement snapshot on this instance (passthrough).

        Meaningful only on the world instance, where the rendezvous is joined; inert (None) on the public one. The
        Coordinator parses and stores the result, the manager only fetches it.

        Returns:
            The rendezvous snapshot, or None.
        """
        return self.p2p.get_rendezvous_peers_info()

    def get_messages(self, kind: str = "all") -> tuple:
        """Pops and decodes this instance's inbound batch (passthrough to the transport).

        P2P returns (chat_messages, session_inbounds): chat_messages are decoded, sender-verified Msg,
        session_inbounds are SessionInbound. The wire codec lives entirely in P2P (the manager stays protobuf-free);
        the Coordinator's chat path only gates membership and verifies the piggyback token on top of these. A
        transport without split pops (the browser one) serves kind "network" as "all" and "streams" with nothing.

        Args:
            kind: "all" (default), "network" (everything but the streams) or "streams" (one per stream channel).

        Returns:
            The tuple (chat_messages, session_inbounds).
        """
        if kind == "all" or not getattr(self.p2p, "supports_split_pop", False):
            return ([], []) if kind == "streams" else self.p2p.get_messages()
        return self.p2p.get_messages(kind)

    @property
    def log_sub(self) -> str:
        """Gets the transport's log sub-label ('pub'/'wor').

        Exposed so the Coordinator can tag its chat-path logs without reaching into the P2P handle.

        Returns:
            The sub-label.
        """
        return self.p2p.log_sub

    # Per-instance identity / posture facade. peer_id/addresses/is_public are properties of
    # THIS instance (the manager is the instance), so the Coordinator reads its own network
    # identity through the owning manager, never through a raw P2P handle. The peer-id is
    # already self.local_peer_id (set at construction, used by the tie-breaker).
    @property
    def addresses(self) -> list[str]:
        """Gets this instance's live multiaddresses (passthrough to the transport).

        Returns:
            The list of the multiaddresses.
        """
        return self.p2p.addresses

    @property
    def is_public(self) -> "bool | None":
        """Tells whether this instance is publicly reachable (passthrough to the transport).

        Returns:
            True or False, or None when not yet known.
        """
        return self.p2p.is_public

    @property
    def relay_enabled(self) -> bool:
        """Tells whether the relay client is enabled on this instance (passthrough).

        Returns:
            True if the relay client is enabled.
        """
        return self.p2p.relay_is_enabled

    def sync_known_peers(self, known: dict[str, tuple]) -> None:
        """Reconciles the inert KNOWN roster against the peers currently advertised by the rendezvous.

        Registers a KNOWN Session for every known peer that has no Session yet, refreshes the rendezvous-sourced
        metadata (addresses + applicative role) on the Sessions that already exist, and prunes KNOWN Sessions whose
        peer has vanished.

        The metadata refresh touches a Session's addresses/app_role only, never its FSM state, so an active
        (established / in-transit) Session keeps running while its role/addrs stay current. Pruning is still
        KNOWN-only: an ESTABLISHED peer that drops out of the rendezvous keeps its live Session until the transport
        tears it down, and a FAILED / CLOSED one is left to the normal evict path. KNOWN Sessions live solely in the
        registry, never in a pool, so this cannot perturb the per-cycle diff or the perimeter.

        Args:
            known: The advertised peers, ``peer_id -> (addresses, app_role)``.
        """
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
            self._forget(peer_id)

    def ensure(
        self,
        peer_id: str,
        addresses: list[str],
        is_world_join: bool = False,
        remote_node_id_hint: str | None = None,
        grant: dict | None = None,
        reject_relay: bool = False,
        app_role: int = 0,
        dial_attempts: int | None = None,
        dial_budget_s: float | None = None,
    ) -> Session:
        """Registers an initiator Session to `peer_id` in the KNOWN state WITHOUT starting it (no dial yet).

        Idempotent: a non-terminal existing Session is returned unchanged; a FAILED/CLOSED one is replaced.
        ``activate`` dials it later. This lets a caller know a peer without committing to a connection.

        Args:
            peer_id: The peer.
            addresses: The multiaddrs of the peer.
            is_world_join: True for the public Session of a world join (default: False).
            remote_node_id_hint: The node-id expected at the peer, when known (default: None).
            grant: The grant to redeem, for the world Session of a join (default: None).
            reject_relay: True to fail a dial that resolves through a relay (default: False).
            app_role: The applicative world role of the peer (default: 0).
            dial_attempts: Override of the dial-attempt cap (default: None, module default).
            dial_budget_s: Override of the CONNECTING budget in seconds (default: None, module default).

        Returns:
            The registered (or existing) Session.
        """
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
            dial_attempts=dial_attempts,
            dial_budget_s=dial_budget_s,
            address_lookup=self._address_lookup,
        )
        self._set_known(peer_id, sess)
        return sess

    async def activate(self, peer_id: str) -> list[SessionEvent]:
        """Starts the initiator Session for `peer_id`: enters CONNECTING and launches the dial off-task (async).

        The outcome of the dial — HELLO / retry / FAILED — is reaped by tick_dial. A quarantined peer is refused
        without dialing. No-op if there is no Session for the peer.

        Args:
            peer_id: The peer.

        Returns:
            The events from start (always [] today), or a synthetic FAILED event for a quarantined peer.
        """
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

            # Surface the refusal to the operator: a silent wall of instant failures
            # reads as "it never connects". The USER line names the wait; the reason
            # keeps its "quarantined" prefix (substring/startswith stays robust) but
            # is human-oriented — the machine-readable remainder rides
            # data["retry_after_s"], so a scheduler (the rejoin watchdog) can retry
            # right past the ban instead of bouncing on a blind cadence.
            remaining = ban_until - now
            log.user(f"dial to ...{peer_id[-5:]} refused: peer quarantined for {remaining:.1f}s more",
                     sub=self._label)
            return [SessionEvent(
                type=SessionEventType.FAILED, session_id="",
                peer_id=peer_id, role=SessionRole.INITIATOR,
                p2p_name=self.p2p_name,
                data={"reason": f"quarantined:{remaining:.1f}s", "retry_after_s": remaining},
            )]
        log.session(f"activating Session to ...{peer_id[-5:]}", sub=self._label)
        events = sess.start()   # non-blocking: the dial runs off-task, reaped by tick_dial
        self._promote(peer_id, sess)
        self._on_terminal(events, now)
        return events

    async def initiate(
        self,
        peer_id: str,
        addresses: list[str],
        is_world_join: bool = False,
        remote_node_id_hint: str | None = None,
        grant: dict | None = None,
        reject_relay: bool = False,
        replace_stale: bool = False,
        dial_attempts: int | None = None,
        dial_budget_s: float | None = None,
        hold: bool = False,
    ) -> list[SessionEvent]:
        """Opens a new initiator Session to `peer_id` and starts it (async).

        A stale FAILED/CLOSED Session is replaced. initiate() = ensure() + activate(). With ``hold`` the Session is
        registered with this intent and left KNOWN, for a later ``activate``.

        Idempotent on a matching-intent non-terminal Session (returns []), BUT a Session whose intent DIVERGES from the
        request is replaced rather than silently reused: the world-join flag must match, and the Coordinator sets
        ``replace_stale`` for a spent world-join carrier (an established carrier whose handoff already failed and was
        rolled back). Otherwise a leftover Session would shadow the dial and the join would never (re)start — the
        caller sees success while nothing happens.

        Args:
            peer_id: The peer.
            addresses: The multiaddrs of the peer.
            is_world_join: True for the public Session of a world join (default: False).
            remote_node_id_hint: The node-id expected at the peer, when known (default: None).
            grant: The grant to redeem, for the world Session of a join (default: None).
            reject_relay: True to fail a dial that resolves through a relay (default: False).
            replace_stale: True to replace a spent world-join carrier (default: False).
            dial_attempts: Override of the dial-attempt cap (default: None, module default).
            dial_budget_s: Override of the CONNECTING budget in seconds (default: None, module default).
            hold: True to register the Session without dialing yet (default: False).

        Returns:
            The events of the start (possibly empty).
        """

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
            self._forget(peer_id)
        elif existing is not None and existing.state == SessionState.CLOSING:

            # A CLOSING Session is not a healthy one to no-op on: its GOAWAY is out
            # and the relation is ending, so a new dial intent inside its window (up
            # to 10s) must start a fresh Session — not be swallowed while the caller
            # is told the dial started. This is the outbound mirror of deliver's
            # inbound supersede. No farewell is owed (already sent): forget and dial
            # fresh over the still-live transport. The forgotten FSM emits no CLOSED,
            # so no settlement cut is scheduled — right, the re-dial owns it now.
            log.session(f"initiate(...{peer_id[-5:]}): superseding CLOSING Session with a "
                        f"fresh dial", sub=self._label)
            self._forget(peer_id)
        elif existing is not None and existing.state not in (SessionState.FAILED, SessionState.CLOSED):
            if not replace_stale and existing.is_world_join == is_world_join:
                log.session(f"initiate(...{peer_id[-5:]}): "
                          f"already have Session in state {existing.state.value} — no-op", sub=self._label)
                return []

            # Intent diverged (world-join vs plain), or a spent carrier: the stale Session
            # would shadow the new dial, and a bare forget leaves the remote ESTABLISHED,
            # ignoring our HELLO. GOAWAY-close it (so the remote drops its Session and
            # answers the new HELLO) and forget it now — without waiting for the CLOSED,
            # or ensure() below would return the CLOSING — then dial fresh. The transport
            # is deliberately NOT touched: the live connection is exactly what the re-dial
            # wants, and a disconnect here would abort the GOAWAY we just queued (the
            # forgotten Session emits no CLOSED, so no settlement cut is scheduled).
            log.session(f"initiate(...{peer_id[-5:]}): replacing {existing.state.value} Session "
                        f"(intent diverged or stale carrier)", sub=self._label)
            await self.close_session(peer_id, GRACEFUL_SHUTDOWN)
            self._forget(peer_id)

        self.ensure(peer_id, addresses, is_world_join=is_world_join,
                    remote_node_id_hint=remote_node_id_hint, grant=grant, reject_relay=reject_relay,
                    app_role=app_role, dial_attempts=dial_attempts, dial_budget_s=dial_budget_s)
        if hold:
            log.session(f"Session to ...{peer_id[-5:]} registered, its dial held", sub=self._label)
            return []
        log.session(f"initiating Session to ...{peer_id[-5:]}", sub=self._label)
        return await self.activate(peer_id)

    def has_established_world_join_carrier(self, peer_id: str) -> bool:
        """Tells whether peer_id holds an ESTABLISHED initiator Session opened for a world join.

        Paired with the Coordinator's pending-handoff view: an established world-join carrier with NO pending handoff
        is a spent carrier (its private handoff failed and was rolled back, leaving the carrier ESTABLISHED), which the
        next join attempt must replace rather than reuse. Restricted to ESTABLISHED so an in-flight handshake is never
        torn down.

        Args:
            peer_id: The peer.

        Returns:
            True if the peer holds an established world-join carrier.
        """
        s = self._sessions.get(peer_id)
        return s is not None and s.is_world_join and s.state == SessionState.ESTABLISHED

    async def deliver(self, peer_id: str, protocol_id: str, sess_msg: SessMsg) -> list[SessionEvent]:
        """Routes a Session message at this manager's p2p instance (async).

        Handles the tie-breaker on inbound HELLO from an unknown / contending peer, and replies on behalf of no Session
        when we hold none for the peer.

        Args:
            peer_id: The sender peer.
            protocol_id: The Session protocol the message arrived on.
            sess_msg: The decoded message.

        Returns:
            The events generated (already settled, see _on_terminal).
        """
        if protocol_id not in SESSION_PROTOCOL_IDS:
            log.error(f"deliver: unknown protocol {protocol_id}, dropping", sub=self._label)
            return []

        sess = self._sessions.get(peer_id)

        # A terminal Session must not shadow new traffic from the peer (an
        # inbound HELLO routed to a dead FSM is silently ignored, blocking any
        # re-handshake initiated by the remote): evict it and fall through to
        # the no-session paths.
        if sess is not None and sess.state in (SessionState.FAILED, SessionState.CLOSED):
            self._forget(peer_id)
            sess = None

        # A terminal-ADJACENT Session must not shadow a fresh handshake either: a new
        # inbound HELLO_REQ while we are CLOSING (our GOAWAY still in flight, any
        # role) or while a RESPONDER rides its reconnect grace is the peer's
        # declaration that the old session is dead on its side — it re-dialed.
        # Supersede silently: abort the old FSM (no terminal event; the hosted/pool
        # state of the old relation survives until the new ESTABLISHED re-enrols it)
        # and let the HELLO spawn a fresh responder below. These are already-dying
        # sessions, so no session_id guard is needed.
        if (sess is not None and protocol_id == SESSION_HELLO_PROTOCOL and
                sess_msg.kind == SessMsg.KIND_HELLO_REQ and
                (sess.state == SessionState.CLOSING or
                 (sess.state == SessionState.RECONNECTING and
                  sess.role == SessionRole.RESPONDER))):
            log.session(f"fresh inbound HELLO supersedes stale {sess.state.value} Session "
                        f"for ...{peer_id[-5:]}", sub=self._label)
            sess.abort_silently("superseded_by_new_hello")
            self._forget(peer_id)
            sess = None

        # Reborn-peer supersede: a HELLO_REQ carrying a NEW session_id while we hold a
        # session the peer is supposed to still own — ESTABLISHED (either role), or
        # RECONNECTING as INITIATOR — proves the peer abandoned that session and came
        # back fresh (a live peer would RESUME the shared id, not send a new HELLO).
        # Without this the ESTABLISHED FSM silently ignores the HELLO and the initiator
        # goes to the tie-breaker and may reject the reborn peer; either way recovery
        # waits ~55s for our own liveness path. The session_id must be NON-EMPTY and
        # DIFFERENT: a replay of the establishing HELLO carries the same id (falls
        # through, ignored) and a malformed empty id must never tear down a live session.
        # abort_silently keeps the transport and the hosted/pool state, re-enrolled when
        # the fresh responder reaches ESTABLISHED. NB: a genuine simultaneous open lives
        # in CONNECTING/HELLO_SENT (kept on the tie-breaker), not here.
        inbound_sid = bytes(sess_msg.get("session_id") or b"")
        if (sess is not None and protocol_id == SESSION_HELLO_PROTOCOL and
                sess_msg.kind == SessMsg.KIND_HELLO_REQ and
                inbound_sid and inbound_sid != sess.session_id and
                (sess.state == SessionState.ESTABLISHED or
                 (sess.state == SessionState.RECONNECTING and
                  sess.role == SessionRole.INITIATOR))):
            log.session(f"fresh inbound HELLO (new session_id) supersedes stale "
                        f"{sess.state.value} Session for ...{peer_id[-5:]} (reborn peer)",
                        sub=self._label)
            sess.abort_silently("superseded_by_reborn_hello")
            self._forget(peer_id)
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
                  sess_msg.kind in (SessMsg.KIND_PING, SessMsg.KIND_RESUME, SessMsg.KIND_GOAWAY)):

                # We are (most likely) the restarted/forgetful peer: we hold no
                # Session for this session_id. Tell the remote so it stops waiting
                # instead of dropping silently: PING/RESUME get "unknown session" so
                # the initiator re-handshakes rather than riding its grace to LOST;
                # a GOAWAY gets a GOAWAY_ACK so the remote's CLOSING ends in one RTT
                # instead of its full timeout (it already forgot us, no Session to tear
                # down here). Only the manager knows "no Session for this peer", so it
                # also settles the transport, which no Session will.
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
        # Only a genuine simultaneous open (both sides first-dialing) reaches here:
        # RECONNECTING is not eligible — an initiator in reconnect re-dials with RESUME,
        # not a HELLO, so an inbound HELLO there is a reborn peer handled just above.
        elif (protocol_id == SESSION_HELLO_PROTOCOL and
              sess_msg.kind == SessMsg.KIND_HELLO_REQ and
              sess.role == SessionRole.INITIATOR and
              sess.state in (SessionState.CONNECTING, SessionState.HELLO_SENT)):
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
        self._on_terminal(events)
        return events

    async def tick(self, now: float | None = None) -> list[SessionEvent]:
        """Sweeps all Sessions: timeout firings + active re-dial + session-level keep-alive (async).

        It then settles this sweep's terminal events and executes the settlement cuts whose linger expired. Single
        dispatch path; the returned list is what node.py acts on (already settled, see _on_terminal).

        Args:
            now: The current monotonic time, or None to read it here (default: None).

        Returns:
            The events generated by the sweep.
        """
        if now is None:
            now = time.monotonic()
        events: list[SessionEvent] = []

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
            events.extend(await sess.tick_dial(now))
            k_event = await sess.tick_keepalive(now)
            if k_event is not None:
                events.append(k_event)

        # Reap quarantine entries that have expired (cheap; bounded by peer count).
        if self._quarantine:
            expired = [pid for pid, t in self._quarantine.items() if t <= now]
            for pid in expired:
                self._quarantine.pop(pid, None)

        # Settle the terminal events of this sweep (quarantine + forget + cut ledger).
        self._on_terminal(events, now)

        # Execute the settlement cuts whose linger expired: the deferred transport
        # teardown of peers whose Session already ended. A live non-KNOWN Session
        # found now cancels the cut (lazy guard): a re-handshake (re-dial, supersede,
        # replace-stale) reuses the transport and owns it again. A KNOWN placeholder
        # does NOT cancel — it is rendezvous metadata, not a live relation, and
        # keeping the half-open transport would defeat the cut.
        if self._pending_cuts:
            due = [pid for pid, deadline in self._pending_cuts.items() if deadline <= now]
            for pid in due:
                del self._pending_cuts[pid]
                sess = self._sessions.get(pid)
                if sess is not None and sess.state not in (SessionState.KNOWN,
                                                           SessionState.FAILED,
                                                           SessionState.CLOSED):
                    log.session(f"settlement cut for ...{pid[-5:]} cancelled: live Session "
                                f"({sess.state.value})", sub=self._label)
                    continue
                await self.p2p.disconnect(pid)
                log.session(f"settlement cut executed for ...{pid[-5:]}", sub=self._label)
        return events

    async def on_peer_disconnected(self, peer_id: str) -> list[SessionEvent]:
        """Handles a peer dropping, as reported by the polled libp2p diff (async).

        Args:
            peer_id: The peer.

        Returns:
            The events generated (a FAILED event for a handshake in flight, RECONNECTING, or none).
        """
        sess = self._sessions.get(peer_id)
        if sess is None:
            return []

        # CONNECTING is not here: it has no handshake on any connection yet, its own dial
        # decides, and a dial that failed is tried again. A connection that its dial was
        # handed and that carried no HELLO is such a failed dial, and its drop, reported
        # a moment later, must not fail and quarantine the attempt that follows.
        if sess.state in (SessionState.HELLO_SENT,
                          SessionState.HELLO_ACK_SENT,
                          SessionState.PROFILE_SENT,
                          SessionState.AWAITING_WORLD_GRANT):

            # Handshake can't continue without a live connection — fail fast.
            # AWAITING_HANDOFF is deliberately NOT here: the parked grant carrier
            # tolerates transport blips (the joiner's re-dial + RESUME is accepted
            # in that state, and failing it here would quarantine the joiner and
            # collapse the handoff on any blip the poll can see); its 30s timeout
            # is the backstop. on_transport_disconnected below no-ops for it.
            ev = sess._fail("connection_dropped_during_handshake")
            self._on_terminal([ev])
            return [ev]

        ev = sess.on_transport_disconnected()
        return [ev] if ev is not None else []

    async def on_peer_reconnected(self, peer_id: str) -> list[SessionEvent]:
        """Handles a peer reappearing in the polled libp2p diff (async).

        Args:
            peer_id: The peer.

        Returns:
            The events generated (possibly empty).
        """
        sess = self._sessions.get(peer_id)
        if sess is None:
            return []
        ev = sess.on_transport_reconnected()
        return [ev] if ev is not None else []

    async def close_session(
        self,
        peer_id: str,
        reason_code: int = GRACEFUL_SHUTDOWN,
    ) -> list[SessionEvent]:
        """Closes the Session of a peer with a GOAWAY (async).

        Args:
            peer_id: The peer.
            reason_code: The reason code carried by the GOAWAY (default: GRACEFUL_SHUTDOWN).

        Returns:
            An empty list: the CLOSED event arrives later via process_inbound(GoawayAck) or tick.
        """
        sess = self._sessions.get(peer_id)
        if sess is None:
            log.session(f"close_session(...{peer_id[-5:]}): no Session — no-op", sub=self._label)
            return []
        await sess.close(reason_code)
        return []  # CLOSED event will arrive via process_inbound(GoawayAck) or tick

    async def drop(self, peer_id: str, reason_code: int | None = None) -> list[SessionEvent]:
        """Tears the relation down on THIS instance, dispatching on the Session state (async).

        The transport follows the Session's terminal state — never a disconnect in the same stack as a farewell write
        (P2P.disconnect is abortive and would destroy the GOAWAY/ACK just queued):

          * already tearing down (CLOSING, or a settlement cut pending — which
            implies nothing live behind it: a live Session cancels the entry
            eagerly at birth) → no-op: makes the frequent double-purge (a CLOSED
            dispatched after an earlier purge) idempotent WITHOUT aborting the
            farewell in flight;
          * ESTABLISHED/RECONNECTING/AWAITING_HANDOFF with a reason → GOAWAY-close
            only; the deferred settlement cut severs the transport once the CLOSED
            lands (ACK or the CLOSING timeout);
          * KNOWN → forget the placeholder and sever any half-open transport now
            (there is no GOAWAY owed to a peer we never handshaked; the disconnect
            is a Go no-op when nothing is connected, and it is the only cure for a
            chat-only/half-open peer whose Session never went past KNOWN);
          * any other live Session (mid-handshake, or a live Session dropped with no
            reason) → silent coercive teardown: abort, forget, disconnect now — a
            purge must abort a handshake, not leave it running to completion;
          * no Session (or a stale terminal one) → disconnect now: nothing will ever
            settle this transport, and this is the only cure for a half-open peer.

        The manager owns its P2P handle, so the conn layer never reaches the transport directly
        (Node -> NodeConn -> SM -> P2P); the disconnect is best-effort (an error means 'gone').

        Args:
            peer_id: The peer.
            reason_code: The reason code of the GOAWAY, or None for a teardown with no farewell (default: None).

        Returns:
            The close events (always []: the CLOSED arrives later via tick/ACK).
        """
        sess = self._sessions.get(peer_id)

        if sess is not None and sess.state == SessionState.CLOSING:
            log.session(f"drop(...{peer_id[-5:]}): already tearing down — no-op", sub=self._label)
            return []

        if peer_id in self._pending_cuts:

            # A pending settlement cut implies nothing live past a KNOWN placeholder
            # behind it (a live Session cancels the entry eagerly at birth, see
            # _set_active/_promote): this is the frequent double-purge, a no-op that
            # must not abort the farewell in flight.
            log.session(f"drop(...{peer_id[-5:]}): already tearing down — no-op", sub=self._label)
            return []

        if sess is None or sess.state in (SessionState.FAILED, SessionState.CLOSED):
            if sess is not None:
                self._forget(peer_id)
            await self.p2p.disconnect(peer_id)
            return []

        if sess.state == SessionState.KNOWN:
            self._forget(peer_id)
            await self.p2p.disconnect(peer_id)
            return []

        if reason_code is not None and sess.state in (SessionState.ESTABLISHED,
                                                      SessionState.RECONNECTING,
                                                      SessionState.AWAITING_HANDOFF):
            return await self.close_session(peer_id, reason_code)

        sess.abort_silently(
            f"dropped:{reason_name(reason_code) if reason_code is not None else 'no_reason'}")
        self._forget(peer_id)
        await self.p2p.disconnect(peer_id)
        return []

    async def close_all(self, reason_code: int = GRACEFUL_SHUTDOWN) -> list[SessionEvent]:
        """Sends a GOAWAY to every Session that can be gracefully closed, aborting the others (async).

        Used on node/runner shutdown so peers see a clean CLOSED instead of falling through liveness-timeout. Sessions
        already CLOSING/CLOSED (e.g. an intentional world-handoff GOAWAY) are left alone; every other non-terminal
        state (a handshake in flight, a KNOWN placeholder) is aborted silently and forgotten — no farewell is owed
        mid-handshake, and the shutdown drain must be able to reach 'nothing closing, nothing pending' instead of
        waiting on FSMs nobody will ever pump again.

        Args:
            reason_code: The reason code carried by the GOAWAYs (default: GRACEFUL_SHUTDOWN).

        Returns:
            An empty list: the CLOSED events arrive later.
        """
        for peer_id, sess in list(self._sessions.items()):
            if sess.state in (SessionState.CLOSING, SessionState.CLOSED, SessionState.FAILED):
                continue
            if sess.state in (SessionState.ESTABLISHED, SessionState.RECONNECTING,
                              SessionState.AWAITING_HANDOFF):
                try:
                    await sess.close(reason_code)
                except Exception as e:
                    log.error(f"close_all: close(...{sess.peer_id[-5:]}) failed: {e}", sub=self._label)
            else:
                sess.abort_silently("shutdown")
                self._forget(peer_id)
        return []

    def get_session(self, peer_id: str) -> Session | None:
        """Gets the Session of a peer.

        Args:
            peer_id: The peer.

        Returns:
            The Session, or None if there is none.
        """
        return self._sessions.get(peer_id)

    def is_established(self, peer_id: str) -> bool:
        """Tells whether the Session of a peer is ESTABLISHED.

        Args:
            peer_id: The peer.

        Returns:
            True if the peer has an ESTABLISHED Session.
        """
        s = self._sessions.get(peer_id)
        return s is not None and s.state == SessionState.ESTABLISHED

    def all_established_peer_ids(self) -> list[str]:
        """Gets the peers whose Session is ESTABLISHED.

        Returns:
            The list of their peer IDs.
        """
        return [pid for pid, s in self._sessions.items()
                if s.state == SessionState.ESTABLISHED]

    def has_pending_teardown(self) -> bool:
        """Tells whether a farewell is still in flight on this instance.

        That is a CLOSING Session waiting its GOAWAY_ACK, or a settlement cut not yet executed. The shutdown drain
        polls this to know when the goodbyes are done.

        Returns:
            True while a teardown is in flight.
        """
        return bool(self._pending_cuts) or any(
            s.state == SessionState.CLOSING for s in self._active_sessions.values())

    def close_transport(self) -> None:
        """Shuts down this instance's transport (the Go host), freeing its slot.

        Terminal: only the shutdown path calls it, after the drain.
        """
        self.p2p.close()

    def is_tearing_down(self, peer_id: str) -> bool:
        """Tells whether this peer's teardown is in flight.

        Its Session is CLOSING (a GOAWAY awaiting its ACK) or a settlement cut is pending in the ledger. reconcile
        skips such peers (no pool re-admission during the linger, no second cut) and the shutdown drain waits on them.

        Args:
            peer_id: The peer.

        Returns:
            True while the peer's teardown is in flight.
        """
        sess = self._sessions.get(peer_id)
        return (peer_id in self._pending_cuts or
                (sess is not None and sess.state == SessionState.CLOSING))

    def quarantine_left(self, peer_id: str, now: float | None = None) -> float:
        """Seconds left of this peer's quarantine, 0 when it has none: the time before a dial
        toward it is let through."""
        if now is None:
            now = time.monotonic()
        return max(0.0, self._quarantine.get(peer_id, 0.0) - now)

    def is_in_grace(self, peer_id: str) -> bool:
        """Tells whether this peer's Session is riding its reconnect grace window (RECONNECTING).

        Disconnected at the transport level but not yet gone. The Coordinator aggregates this across its instances so
        the Node can choose pool-remove (keep the enrolment across the blip) vs full purge without reaching into
        Session state itself.

        Args:
            peer_id: The peer.

        Returns:
            True if the peer's Session is RECONNECTING.
        """
        sess = self._sessions.get(peer_id)
        return sess is not None and sess.state == SessionState.RECONNECTING

    def is_handshake_in_flight(self, peer_id: str) -> bool:
        """Tells whether this peer has a Session whose FIRST handshake is still in flight.

        It exists but has never reached ESTABLISHED, and is neither a rendezvous-only KNOWN nor RECONNECTING (already
        established once) nor terminal (FAILED/CLOSED). This is the session-era 'connected but not yet managed' notion
        the Coordinator aggregates for Agent.agents_are_waiting. The classification lives here, next to SessionState,
        so the Coordinator stays free of the session enum.

        Args:
            peer_id: The peer.

        Returns:
            True while the first handshake of the peer is in flight.
        """
        sess = self._sessions.get(peer_id)
        if sess is None:
            return False
        return sess.state not in (SessionState.KNOWN, SessionState.ESTABLISHED,
                                  SessionState.RECONNECTING, SessionState.FAILED,
                                  SessionState.CLOSED)

    def has_active_sessions(self) -> bool:
        """Tells whether any Session is ESTABLISHED or riding its reconnect grace.

        The Node uses this to keep the main loop at full speed: the keep-alive RTT window assumes a responsive pump,
        so a node with live sessions must never be treated as idle.

        Returns:
            True if there is at least one live Session.
        """
        return any(s.state in (SessionState.ESTABLISHED, SessionState.RECONNECTING)
                   for s in self._active_sessions.values())

    def requires_fast_clock(self) -> bool:
        """Tells whether any Session needs the main loop at full speed.

        Every tick-active state except the terminals FAILED and CLOSED (KNOWN Sessions are never in the tick-active
        view, so they are excluded for free). Unlike has_active_sessions, which counts only ESTABLISHED/RECONNECTING,
        this also covers handshakes in flight (CONNECTING/HELLO_*/PROFILE_SENT/AWAITING_*) and CLOSING, whose
        RTT-sized timers would be starved by a slowed clock.

        Returns:
            True if some Session needs the fast clock.
        """
        return any(s.state not in (SessionState.FAILED, SessionState.CLOSED)
                   for s in self._active_sessions.values())

    def __len__(self) -> int:
        """Gets the number of Sessions in the registry (KNOWN ones included).

        Returns:
            The number of Sessions.
        """
        return len(self._sessions)

    def __repr__(self) -> str:
        """Returns a short description of the manager (instance, Sessions by state, quarantined peers), used in logs.

        Returns:
            The description string.
        """
        by_state: dict[str, int] = {}
        for s in self._sessions.values():
            by_state[s.state.value] = by_state.get(s.state.value, 0) + 1
        return f"SessionManager(p2p={self.p2p_name} sessions={by_state} quarantined={len(self._quarantine)})"

    # Session registry (keep the two views consistent)
    #
    # Every mutation of the session store goes through these helpers so the
    # tick-active view never drifts from the full registry.
    def _set_known(self, peer_id: str, sess: "Session") -> None:
        """Registers or replaces an inert KNOWN Session: in the full registry, out of the tick-active view.

        Args:
            peer_id: The peer.
            sess: The Session.
        """
        self._sessions[peer_id] = sess
        self._active_sessions.pop(peer_id, None)

    def _set_active(self, peer_id: str, sess: "Session") -> None:
        """Registers a Session the tick must sweep (a responder, or a started initiator).

        A live Session owns the transport again: any settlement cut left over from the previous relation is cancelled
        eagerly here (it must not fire under the new relation, nor keep the peer flagged as tearing-down for its
        residual linger). A KNOWN placeholder deliberately does NOT cancel (see _set_known).

        Args:
            peer_id: The peer.
            sess: The Session.
        """
        self._sessions[peer_id] = sess
        self._active_sessions[peer_id] = sess
        self._pending_cuts.pop(peer_id, None)

    def _promote(self, peer_id: str, sess: "Session") -> None:
        """Moves a just-started initiator out of KNOWN into the tick-active view.

        Same eager cut-cancel as _set_active: the dial reuses a still-open connection when there is one, and a
        leftover cut firing mid-handshake would sever it.

        Args:
            peer_id: The peer.
            sess: The Session.
        """
        if self._sessions.get(peer_id) is sess and sess.state != SessionState.KNOWN:
            self._active_sessions[peer_id] = sess
            self._pending_cuts.pop(peer_id, None)

    def _forget(self, peer_id: str) -> None:
        """Removes a Session from both views.

        Args:
            peer_id: The peer.
        """
        self._sessions.pop(peer_id, None)
        self._active_sessions.pop(peer_id, None)

    async def _coercive_cut(self, peer_id: str) -> None:
        """Applies the perimeter's security cut to a peer (async).

        It severs the transport NOW, with no GOAWAY and no settlement ledger (it must never be cancellable), and
        quarantines the peer so our own side does not immediately re-dial it. The GOAWAY would be advisory at best (a
        hostile peer ignores it) and the mesh isolation cannot wait for an ACK round-trip; the cut peer converges on
        its own via re-dial -> RESUME -> NACK(SESSION_UNKNOWN) -> LOST and its own quarantine. Any live Session is
        aborted silently first so no terminal event (and no settlement cut) fires.

        Args:
            peer_id: The peer.
        """
        sess = self._sessions.get(peer_id)
        if sess is not None:
            sess.abort_silently("perimeter_cut")
            self._forget(peer_id)
        self._quarantine[peer_id] = time.monotonic() + _QUARANTINE_S
        await self.p2p.disconnect(peer_id)

    def _on_terminal(self, events: list[SessionEvent], now: float | None = None) -> None:
        """Settles terminal SessionEvents: the single reaction to them, whichever path produced them.

        The paths are deliver, tick, or the transport fail-fast. It quarantines the peer on LOST/FAILED (deliberately
        including the deliver-path FAILED — rejections and dispatch errors — which the old cleanup exempted), forgets
        the dead FSM so it never shadows later inbound traffic, and schedules the deferred transport cut in the
        _pending_cuts ledger.

        ORDER INVARIANT: this runs inside deliver/tick, i.e. BEFORE the events are returned to the layers above. By the
        time Node dispatches a terminal event the Session is already forgotten and the cut already scheduled, so the
        purge coming back down (drop with no reason) lands on the tearing-down branch and is a no-op — it cannot abort
        the farewell frame (GOAWAY_ACK, reject) written moments earlier.

        Args:
            events: The events to settle.
            now: The current monotonic time, or None to read it here (default: None).
        """
        if now is None:
            now = time.monotonic()
        for ev in events:
            reason = str(ev.data.get("reason", ""))

            # A failed redemption of a world grant is not quarantined: the world host's world
            # peer is dialled only to redeem a grant, the next grant comes with the next join,
            # which the rejoin paces, and a ban would only refuse its redemption
            sess = self._sessions.get(ev.peer_id)
            redemption = ev.type is SessionEventType.FAILED and sess is not None and sess.redeems_grant
            if ev.type in (SessionEventType.LOST, SessionEventType.FAILED) and not redemption:
                stall = (ev.type is SessionEventType.FAILED and
                         reason.startswith(("timeout:hello_sent", "timeout:hello_ack_sent")))
                self._quarantine[ev.peer_id] = now + (_QUARANTINE_STALL_S if stall
                                                      else _QUARANTINE_S)
            if ev.type in (SessionEventType.LOST,
                           SessionEventType.CLOSED,
                           SessionEventType.FAILED):

                # Only settle if the session is genuinely terminal: a synthetic
                # event (e.g. the quarantine refusal) has no dead FSM behind it.
                sess = self._sessions.get(ev.peer_id)
                if sess is not None and sess.state in (SessionState.FAILED, SessionState.CLOSED):
                    self._forget(ev.peer_id)
                    if "SIMULTANEOUS_OPEN_LOSS" in reason:

                        # The tie-break loser shares its transport with the WINNER's
                        # handshake still in flight: no settlement cut, or a slow
                        # winner dial (e.g. via relay) gets its connection severed
                        # under it. The quarantine above still gates our own dials.
                        continue

                    # Overwrite, not setdefault: the LAST farewell owns the linger.
                    # Inheriting a nearly-due deadline from a previous relation's
                    # entry would cut the transport right under this one's ACK.
                    # A CLOSED completed by the received GOAWAY_ACK needs no linger
                    # at all: the ACK is the end-to-end proof that nothing of ours
                    # is still in flight — cut at the next tick (faster shutdowns).
                    linger = 0.0 if reason.startswith("local_goaway_acked") else _CUT_LINGER_S
                    self._pending_cuts[ev.peer_id] = now + linger

    def _spawn_responder(self, peer_id: str) -> Session:
        """Creates and registers a responder Session for a peer that sent us a HELLO.

        Args:
            peer_id: The peer.

        Returns:
            The new responder Session.
        """
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
        """Replies to a liveness PING/RESUME or a farewell GOAWAY for which we hold no Session (async).

        We are the restarted/forgetful peer: PONG ok=false / RESUME_ACK accepted=false with SESSION_UNKNOWN /
        GOAWAY_ACK received=true, echoing session_id + nonce. Mirrors the simultaneous-open reject pattern — a one-shot
        send on the control protocol, no Session object. The transport the frame came on, or the one our reply
        dialled, belongs to no Session: a settlement cut is scheduled for it, which the HELLO of a peer that starts
        over cancels.

        Args:
            peer_id: The peer.
            sess_msg: The message we hold no Session for.
        """
        if sess_msg.kind == SessMsg.KIND_PING:
            reply = SessMsg(SessMsg.KIND_PONG, {
                "session_id": bytes(sess_msg.get("session_id") or b""),
                "nonce": bytes(sess_msg.get("nonce") or b""),
                "ok": False,
            })
        elif sess_msg.kind == SessMsg.KIND_GOAWAY:

            # The remote is closing a session we already forgot: ack it so its
            # CLOSING completes now (local_goaway_acked) instead of timing out.
            # GOAWAY_ACK carries only 'received' (routed by peer_id, not session_id).
            reply = SessMsg(SessMsg.KIND_GOAWAY_ACK, {"received": True})
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

        # Inside a perimeter, a peer that is not a member gets no entry: the ledger makes
        # the reconcile skip a peer, and anybody can send the frame that brought us here,
        # as often as needed to keep an entry alive. The coercive cut deals with it at
        # the next reconcile, as it does without this reply.
        if (self._perimeter_membership is not None and self._routing_classifier is not None and
                self._routing_classifier({'id': peer_id, 'direction': 'inbound'},
                                         self._perimeter_membership) is None):
            return

        # Overwrite, as in _on_terminal: each reply restarts the wait for the peer's HELLO.
        self._pending_cuts[peer_id] = time.monotonic() + _UNKNOWN_REPLY_LINGER_S
        log.session(f"no Session for ...{peer_id[-5:]}: settlement cut in "
                    f"{_UNKNOWN_REPLY_LINGER_S}s unless it starts over", sub=self._label)

    async def _handle_simultaneous_open(self, peer_id: str) -> str | list[SessionEvent]:
        """Breaks a simultaneous open deterministically: smaller peer-id keeps initiator, larger yields (async).

        Args:
            peer_id: The peer that sent a competing HELLO while we were dialing it.

        Returns:
            One of:

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
        self._forget(peer_id)
        self._spawn_responder(peer_id)
        return "_replaced_with_responder_"
