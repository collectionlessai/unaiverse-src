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
import json
import time
import random
import asyncio
import secrets
from enum import Enum
from unaiverse.utils.logger import log
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, TYPE_CHECKING
from unaiverse.networking.session.sess_msg import CAP_CHUNKED_FRAMES, CAP_FILE_STREAM
from unaiverse.networking.session.sess_msg import GRACEFUL_SHUTDOWN, SESSION_UNKNOWN, TOKEN_EXPIRED
from unaiverse.networking.session.sess_msg import OK, VERSION_UNSUPPORTED, NOT_AUTHORIZED, LOCAL_ERROR
from unaiverse.networking.session.sess_msg import SessMsg, reason_name, ProfileDecisionResult, WorldGrantSpec
from unaiverse.networking.session.sess_msg import VerifiedTokenClaims, SESSION_HELLO_PROTOCOL, SESSION_PROFILE_PROTOCOL
from unaiverse.networking.session.sess_msg import SESSION_WORLD_PROTOCOL, SESSION_CONTROL_PROTOCOL, SESSION_PROTOCOL_IDS

# This import ONLY happens during IDE/Type inspection
if TYPE_CHECKING:
    from unaiverse.networking.p2p.p2p import P2P

# Per-state timeouts (seconds)
# _TIMEOUT_CONNECT_S is the total CONNECTING budget: a backstop for a wedged dial task. The per-attempt bound is Go's
# PeerConnectionTimeout (~30s, x2 on the tcp fallback); tick-driven retries up to _MAX_DIAL_ATTEMPTS (which FAILs
# first) fit inside this window without starving other peers.
_TIMEOUT_CONNECT_S = 300.0

# Sized for slow peer pumps, not for the wire: a peer whose main loop is busy
# (heavy compute in an action, a loaded machine, Pyodide) can drain and answer a
# HELLO several seconds late, and the observed field delays reach ~10s. 15s gives
# margin over that and aligns stage 1 with stage 2, removing the initiator-5s /
# responder-15s asymmetry that produced spurious timeout:hello_sent. A late ACK
# is already accepted at any point in HELLO_SENT, so this is just accepting it.
_TIMEOUT_HELLO_S = 15.0  # Stage 1: HELLO → HELLO_ACK
_TIMEOUT_PROFILE_S = 15.0  # Stage 2: PROFILE_OFFER → PROFILE_DECISION
_TIMEOUT_WORLD_S = 30.0  # Stage 3: WORLD_GRANT expiry

# Session-level reconnect ceiling before Session.Lost. Longer than the Go transport grace (DisconnectionGracePeriod)
# so several backoff'd re-dial + RESUME cycles fit inside the window (see tick_dial).
_TIMEOUT_RECONNECT_S = 45.0

# A responder whose peer answered a PING with ok=false: the peer holds no Session for our id.
# A responder cannot dial, and no RESUME comes from a peer that has just said so, hence
# nothing can end this wait but its timeout. It is kept short and not zero for what may
# still be on its way: a fresh HELLO, which supersedes this Session and spares the pool
# state, or a GOAWAY that the PONG overtook, which closes it cleanly. The PING is the
# keep-alive of an ESTABLISHED Session or the probe of one that is reconnecting.
_TIMEOUT_PEER_LOST_SESSION_S = 10.0
_TIMEOUT_CLOSING_S = 10.0  # GOAWAY sent; wait for GOAWAY_ACK

# GOAWAY write failed (link dead): no ACK can ever come, finalize fast so the shutdown drain does not burn its budget
# waiting.
_TIMEOUT_CLOSING_FAILED_SEND_S = 1.0

# How long past the peer's attested token exp we keep the session before a
# GOAWAY(TOKEN_EXPIRED). Generous on purpose — PING/PONG liveness is the primary
# signal and a refresh (TOKEN_UPDATE) normally lands ~30min before exp; this only
# fires when refreshes are missed AND no re-handshake happened.
_TOKEN_EXP_GRACE_S = 3600.0

# Retry / backoff
_MAX_DIAL_ATTEMPTS = 4
_RETRY_BASE_S = 1.0
_RETRY_CAP_S = 30.0
_RESUME_ACK_TIMEOUT_S = 3.0  # RESUME sent; if un-acked within this, re-dial + re-RESUME

# An initiator in HELLO_SENT with no HELLO_ACK sends the same HELLO again after this long, at
# most this many times, within the timeout of the stage: a HELLO or a HELLO_ACK lost on a
# connection that closed under it (a relay circuit closed by the native upgrade) is not
# known to anybody, and only the timeout of the stage would end the wait.
_HELLO_REPEAT_AFTER_S = 3.0
_MAX_HELLO_REPEATS = 2

# How many repeats of its HELLO a responder answers on one Session, each a signature with
# the host key and a send. An initiator sends one HELLO per dial attempt: _MAX_DIAL_ATTEMPTS
# here, 6 in the browser (maxDialAttempts of MOBILE_SESSION_TIMEOUTS in the SDK), and the last
# one again _MAX_HELLO_REPEATS times, so a legitimate one repeats it 7 times at most.
_MAX_REPEATED_HELLOS_ANSWERED = 7

# How often a reconnecting initiator asks the layer above for fresh addresses of its peer.
# A peer that moved posts its new addresses with its profile within 10 s
# (SEND_DYNAMIC_PROFILE_EVERY), so the second lookup finds them if the first did not.
_ADDRESS_LOOKUP_INTERVAL_S = 15.0

# Session-level keep-alive (PING/PONG over the control protocol)
# Bound to session_id on purpose: /ipfs/ping is a transport probe that go-libp2p
# answers by dial-on-demand, so it cannot distinguish a restarted peer (no Session)
# from a live one. A PONG with ok=false (peer holds no matching Session) is treated
# as a hard liveness failure → reconnect → RESUME → re-handshake.
_KEEPALIVE_PERIOD_S = 15.0

# The RTT window must absorb the full polling chain on BOTH sides (our send is a
# fire-and-forget task dispatched on the next loop turn; the peer answers on its
# next pump; we read the PONG on ours), so it is sized for slow main loops
# (clock slowed to 2s cycles plus blocking work per cycle), not for network RTT.
# Must stay below _KEEPALIVE_PERIOD_S (one ping in flight at a time).
_PING_RTT_TIMEOUT_S = 10.0
_KEEPALIVE_FAIL_THRESHOLD = 2

# Misc
_SESSION_ID_BYTES = 16
_NONCE_BYTES = 32
_GRANT_ID_BYTES = 32


# State / Role / Event enums
class SessionState(str, Enum):
    """The states of the per-peer Session FSM."""
    KNOWN = "known"  # Initiator: peer known, not yet started (no dial)
    INIT = "init"  # Responder: spawned, awaiting inbound HELLO
    CONNECTING = "connecting"  # Initiator: dialing
    HELLO_SENT = "hello_sent"  # Initiator: HELLO out, awaiting HELLO_ACK
    HELLO_ACK_SENT = "hello_ack_sent"  # Responder: HELLO_ACK out, awaiting PROFILE_OFFER
    PROFILE_SENT = "profile_sent"  # Initiator: PROFILE_OFFER out, awaiting PROFILE_DECISION
    AWAITING_WORLD_GRANT = "awaiting_world_grant"  # Initiator: world joiner expecting WORLD_GRANT
    AWAITING_HANDOFF = "awaiting_handoff"  # Responder public: WORLD_GRANT out, awaiting the joiner's GOAWAY
    ESTABLISHED = "established"
    RECONNECTING = "reconnecting"  # Link dropped, within grace period
    CLOSING = "closing"  # Sent GOAWAY, awaiting GOAWAY_ACK
    CLOSED = "closed"
    FAILED = "failed"


class SessionRole(str, Enum):
    """The direction of a Session: who dialed (fixed for the whole life of the Session)."""
    INITIATOR = "initiator"
    RESPONDER = "responder"


class SessionEventType(str, Enum):
    """The application-visible events a Session emits."""
    ESTABLISHED = "established"
    FAILED = "failed"
    RECONNECTING = "reconnecting"  # Transport dropped, within grace
    RESUMED = "resumed"
    LOST = "lost"
    CLOSED = "closed"


@dataclass
class SessionEvent:
    """Emitted by Session when the application-visible state changes."""
    type: SessionEventType
    session_id: str  # Hex string
    peer_id: str
    role: SessionRole
    p2p_name: str
    data: dict[str, Any] = field(default_factory=dict)


# Type alias for the responder decision hook (see SessionManager). The trailing
# argument carries the root-attested claims (RT) when a token-verifier vouched for
# the peer, or None on the v1 / no-verifier path.
DecisionHook = Callable[[str, str | None, dict, str, VerifiedTokenClaims | None],
                        Awaitable[ProfileDecisionResult]]


# Helpers
def _sign_payload(nonce: bytes, session_id: bytes) -> bytes:
    """Builds the canonical sign-payload for sig_a / sig_b: ``nonce || session_id``.

    Args:
        nonce: The nonce of the other side.
        session_id: The session identifier.

    Returns:
        The bytes to sign or to verify.
    """
    return bytes(nonce) + bytes(session_id)


# Session
class Session:
    """Per-peer Session FSM: handshake (HELLO, PROFILE, WORLD), keep-alive, reconnection and close."""

    def __init__(
        self,
        peer_id: str,
        role: SessionRole,
        p2p: "P2P",
        p2p_name: str,
        local_peer_id: str,
        local_node_id: str,
        local_profile_factory: Callable[[], dict],
        remote_node_id_hint: str | None = None,
        addresses: list[str] | None = None,
        is_world_join: bool = False,
        decision_hook: DecisionHook | None = None,
        grant: dict | None = None,
        reject_relay: bool = False,
        app_role: int = 0,
        token_verifier: Callable | None = None,
        token_factory: Callable[[], str] | None = None,
        dial_attempts: int | None = None,
        dial_budget_s: float | None = None,
        address_lookup: Callable[[str], Awaitable[list[str] | None]] | None = None,
    ) -> None:
        """Initializes a Session with a peer, as initiator (state KNOWN) or responder (state INIT).

        Args:
            peer_id: libp2p peer-id of the remote (on this p2p instance).
            role: INITIATOR or RESPONDER.
            p2p: the P2P instance this Session runs on (or a JS OP2P mirror in
                Pyodide). The Session reaches the transport and crypto through it.
            p2p_name: the label of the P2P instance this Session lives on (used for
                events).
            local_peer_id: our libp2p peer-id on `p2p_name`.
            local_node_id: our application-level node-id (the JWT-bound identity).
            local_profile_factory: zero-arg callable returning the current NodeProfile
                dict (must be plain JSON-serializable — sampled fresh at offer time).
            remote_node_id_hint: optional; the remote's application node-id when known.
            addresses: initiator only — peer multiaddrs from root server / cache.
            is_world_join: initiator only — if True on a public Session, expect a
                WORLD_GRANT (Stage-3) after PROFILE_DECISION.
            decision_hook: responder only — async callback to accept/reject an offer
                and (for a world host) produce the WorldGrant spec.
            grant: initiator only, world Session — the grant blob obtained on the
                public Session; when present this Session is a world-join redeemer, so
                node.py runs the world-join body once it reaches ESTABLISHED after the
                normal HELLO+PROFILE handshake on p2p_world.
            app_role: Applicative world role (the rendezvous `misc` bitmask: 1=agent, 3=master),
                refreshed by SessionManager.sync_known_peers and used as the source for the
                role/addrs facade and the world perimeter. Distinct from self.role
                (SessionRole, which is directional). 0 means public / unknown.
            dial_attempts: initiator only — override of the CONNECTING dial-attempt
                cap (None = module default).
            dial_budget_s: initiator only — override of the total CONNECTING budget
                in seconds (None = module default). The world-join redeemer uses both
                to ride a single attempt bounded by its grant window.
            address_lookup: initiator only — async callable, node_id -> the addresses of
                that node as the root server has them (None when not found or on any
                error), given by the Node layer, which alone reaches the root server. A
                reconnecting Session asks it after a failed attempt (see tick_dial).
        """
        self.peer_id = peer_id
        self.role = role
        self.p2p = p2p
        self.p2p_name = p2p_name
        self._sub = "pub" if p2p_name == "p2p_public" else "prv"
        self.local_peer_id = local_peer_id
        self.local_node_id = local_node_id
        self._local_profile_factory = local_profile_factory
        self.remote_node_id_hint = remote_node_id_hint
        self.addresses = list(addresses or [])
        self.app_role = app_role
        self._address_lookup = address_lookup
        self._lookup_task: asyncio.Task | None = None
        self._lookup_due = False   # a reconnect attempt failed since the last lookup
        self._next_lookup_at = 0.0

        # Token capabilities (RT), injected by the layer that owns them
        # (Coordinator/Node). token_verifier(token, p2p_peer) -> (node_id, cv_hash,
        # p2p_peers, exp, code_hash) backed by the root TokenVerifier; token_factory() -> our current
        # node token. Both directions present and verify: the initiator's token rides
        # the PROFILE_OFFER and the responder's the PROFILE_DECISION, each verified by
        # the other against the transport peer-id (see _verify_peer_token). The
        # responder additionally feeds the attested cv_hash / p2p_peers to its
        # decision hook.
        self.token_verifier = token_verifier
        self.token_factory = token_factory

        # Per-Session view of the peer's verified token, captured at handshake and
        # refreshed in-session by TOKEN_UPDATE. exp 0 = no verifier / not yet seen.
        self.remote_token_exp: int = 0
        self.remote_cv_hash: str | None = None
        self.remote_code_hash: str | None = None

        # A TOKEN_UPDATE that arrived before ESTABLISHED is buffered and applied on tick.
        self._buffered_token_update: SessMsg | None = None
        self.is_world_join = is_world_join
        self._decision_hook = decision_hook
        self._grant = grant  # initiator world Session: redeem this after PROFILE
        self._reject_relay = reject_relay  # fail the dial if it resolves through a relay

        # Session identity. Initiator chooses; responder adopts from HELLO.
        self.session_id: bytes = secrets.token_bytes(_SESSION_ID_BYTES)
        self.session_id_hex: str = self.session_id.hex()
        self.nonce_a: bytes = b""
        self.nonce_b: bytes = b""
        self._repeated_hellos_answered: int = 0   # responder: see _responder_on_repeated_hello

        # Initiator in HELLO_SENT: the HELLO sent, when it last went out, and how many times
        # it went out again (see _repeat_hello_if_due)
        self._hello_msg: SessMsg | None = None
        self._hello_sent_at = 0.0
        self._hello_repeats = 0

        # FSM state, derived from the role: an initiator starts KNOWN (registered, not
        # yet dialed); a responder starts INIT (already engaged, awaiting the inbound
        # HELLO). Role is the direction (who dials) and is fixed for life; state is the
        # progress. Hence a KNOWN Session is always an INITIATOR (a dial we intend to
        # make), while a responder is never registered ahead of time and so is never
        # KNOWN. Consumers that read role==INITIATOR therefore pair it with a state
        # guard to exclude the not-yet-dialed KNOWN (see the deliver() tie-breaker).
        self.state: SessionState = (SessionState.KNOWN if role == SessionRole.INITIATOR
                                    else SessionState.INIT)
        self._state_entered_at: float = time.monotonic()
        self._timeout_s: float | None = None

        # Dial retry counter (initiator) + the in-flight dial task. Design A: the dial runs
        # off the main loop as a task and is reaped by tick_dial, so a slow dial cannot starve
        # the other Sessions serviced from the same single main-loop task.
        self._dial_attempts = 0
        self._dial_task: asyncio.Task | None = None
        self._next_dial_at: float = 0.0   # monotonic gate for the CONNECTING dial backoff
        self._max_dial_attempts = dial_attempts if dial_attempts is not None else _MAX_DIAL_ATTEMPTS
        self._dial_budget_s = dial_budget_s if dial_budget_s is not None else _TIMEOUT_CONNECT_S

        # Profile received from the remote peer (set on ESTABLISHED).
        self.remote_profile: dict | None = None
        self.peer_capabilities = 0  # The capability bits of the peer's HELLO / HELLO_ACK (CAP_*)

        # Session-level keep-alive state (PING/PONG over the control protocol,
        # bound to session_id — not /ipfs/ping). One ping in flight at a time; its
        # PONG arrives via process_inbound and clears _ping_in_flight.
        self._last_ping_at: float = 0.0
        self._consecutive_ping_failures: int = 0
        self._ping_in_flight: bool = False
        self._ping_nonce: bytes = b""
        self._ping_deadline: float = 0.0

        # The send of a probe in flight (responder, while RECONNECTING), see _start_probe.
        self._probe_task: asyncio.Task | None = None

        # Reason string of the GOAWAY we sent (close()), echoed in the local
        # CLOSED event so both sides observe the same close reason.
        self._local_goaway_reason: str = ""

        # WORLD_GRANT that overtook the PROFILE_DECISION (no cross-protocol
        # ordering); consumed as soon as the decision lands.
        self._early_world_grant: SessMsg | None = None

        # Active re-dial + RESUME state (initiator drives recovery while RECONNECTING).
        self._reconnect_substep: str = ""   # "" | "redial" | "resume_sent"

        # True while RECONNECTING because the peer said it lost the Session (responder only).
        self._peer_lost_session: bool = False

        # When RECONNECTING began and its window, which bound the restarts of that window.
        self._reconnecting_since: float | None = None
        self._reconnect_window_s: float = _TIMEOUT_RECONNECT_S
        self._reconnect_attempts: int = 0
        self._next_redial_at: float = 0.0   # monotonic gate, set/advanced by tick_dial
        self._resume_nonce: bytes = b""

        # When the RESUME in flight is given up, and whether one is owed on a transport
        # that came back without a dial of ours (see _transport_is_back).
        self._resume_deadline: float = 0.0
        self._resume_due: bool = False

        # World-join specifics.
        self.grant_id: bytes = b""
        self.grant_expiry_unix_ms: int = 0

    def __repr__(self) -> str:
        """Returns a short description of the Session (peer, role, state, p2p instance), used in logs.

        Returns:
            The description string.
        """
        return (f"Session(...{self.peer_id[-5:]} role={self.role.value} "
                f"state={self.state.value} p2p={self.p2p_name})")

    @property
    def redeems_grant(self) -> bool:
        """Tells whether this is an initiator Session opened to redeem a world grant.

        Returns:
            True for an initiator Session that carries a world grant.
        """
        return self.role == SessionRole.INITIATOR and self._grant is not None

    def start(self) -> list[SessionEvent]:
        """Starts an initiator Session: enters CONNECTING and launches the dial as a background task.

        The dial runs off the main loop (Design A) so it does not starve the other Sessions serviced from the same
        task; its outcome (HELLO on success, retry/FAILED on failure) is reaped on the next tick_dial, so this returns
        immediately with no events.

        Returns:
            An empty list of events.

        Raises:
            RuntimeError: If the Session is not an initiator.
        """
        if self.role != SessionRole.INITIATOR:
            raise RuntimeError("Session.start() is for initiators only")
        self._set_state(SessionState.CONNECTING, self._dial_budget_s)
        self._next_dial_at = 0.0
        self._start_dial()
        return []

    async def tick_dial(self, now: float | None = None) -> list[SessionEvent]:
        """Reaps a finished dial (→ HELLO / RESUME / retry / FAILED), then launches the next one when due (async).

        Replaces _dial_then_hello, tick_reconnect and _redial_and_resume: the dial is a task, so no other Session is
        starved while it runs. While RECONNECTING it first sends the RESUME owed to a transport that came back by itself
        (see _transport_is_back), which waits for no dial.

        Args:
            now: The current monotonic time, or None to read it here (default: None).

        Returns:
            The events generated by the reaped dial (possibly empty).
        """
        if now is None:
            now = time.monotonic()

        # The transport came back without a dial of ours: the RESUME goes on it now,
        # whatever the dial in flight is doing, and that dial is left to end by itself.
        if self._resume_due:
            self._resume_due = False
            if self.state == SessionState.RECONNECTING:
                if await self._send_resume(now):

                    # Should no answer come, the dial is tried as soon as the wait is over.
                    self._next_redial_at = min(self._next_redial_at, self._resume_deadline)
                    log.session(f"{self!r} RESUME sent on the transport that came back", sub=self._sub)
                else:
                    self._reconnect_substep = "redial"
                    self._next_redial_at = now
                    log.session(f"{self!r} RESUME send failed, re-dialling", sub=self._sub)

        if self.state == SessionState.RECONNECTING:
            self._tick_address_lookup(now)

        # (a) Reap a completed dial.
        task = self._dial_task
        if task is not None and task.done():
            self._dial_task = None
            if not task.cancelled():
                peer_id, through_relay = task.result()   # _run_dial never raises
                if self.state == SessionState.CONNECTING:
                    return await self._on_dial_done_connecting(peer_id, through_relay, now)
                if self.state == SessionState.RECONNECTING:
                    return await self._on_dial_done_reconnecting(peer_id, through_relay, now)

                # State advanced (a terminal event fired) while the dial was in flight →
                # discard the outcome here. This state guard is why a terminal transition
                # needs no explicit dial detach: _run_dial holds no Session reference, so a
                # late task is harmless. The connection is normally severed by the
                # settlement cut scheduled at the terminal transition; a dial that lands
                # only AFTER that cut executed leaves a stray sessionless connection that
                # the next drop/perimeter pass (or the remote's own teardown) reaps.
            return []
        if self._dial_task is not None:
            return []   # a dial is still in flight

        # No HELLO_ACK yet: the same HELLO again, when due
        if self.state == SessionState.HELLO_SENT:
            await self._repeat_hello_if_due(now)
            return []

        # (b) Launch the next dial once its gate opens.
        if self.role != SessionRole.INITIATOR or not self.addresses:
            return []
        if self.state == SessionState.CONNECTING and now >= self._next_dial_at:
            self._start_dial()
        elif self.state == SessionState.RECONNECTING:

            # A RESUME that wasn't acked within its window → fall back to another re-dial.
            # It counts as an attempt: the transport it went on delivered nothing, as the
            # connection to a peer that stopped does while it is still listed.
            if self._reconnect_substep == "resume_sent" and now >= self._resume_deadline:
                self._reconnect_substep = "redial"
                self._reconnect_attempts += 1
                self._lookup_due = True
                log.session(f"{self!r} RESUME got no answer within {_RESUME_ACK_TIMEOUT_S}s, "
                            f"counted as reconnect attempt {self._reconnect_attempts}", sub=self._sub)
            if self._reconnect_substep == "redial" and now >= self._next_redial_at:
                self._start_dial()
        return []

    async def process_inbound(self, protocol_id: str, sess_msg: SessMsg) -> list[SessionEvent]:
        """Routes an inbound Session message through the FSM (async).

        Called by SessionManager.deliver for every SessionInbound.

        Args:
            protocol_id: The Session protocol the message arrived on.
            sess_msg: The decoded message.

        Returns:
            The events generated by the message (possibly empty).
        """
        events: list[SessionEvent] = []
        try:
            if protocol_id == SESSION_HELLO_PROTOCOL:
                events += await self._on_hello_protocol(sess_msg)
            elif protocol_id == SESSION_PROFILE_PROTOCOL:
                events += await self._on_profile_protocol(sess_msg)
            elif protocol_id == SESSION_WORLD_PROTOCOL:
                events += await self._on_world_protocol(sess_msg)
            elif protocol_id == SESSION_CONTROL_PROTOCOL:
                events += await self._on_control_protocol(sess_msg)
            else:
                log.session(f"{self!r} unknown protocol {protocol_id}", sub=self._sub)
        except Exception as e:
            log.error(f"{self!r} process_inbound({protocol_id}) raised: {e}", sub=self._sub)
            events.append(self._fail(f"inbound_dispatch_error:{e}"))
        return events

    async def send_token_update(self, token: str) -> None:
        """Pushes our refreshed root token to this peer, ESTABLISHED Sessions only (async).

        The peer re-verifies it and advances its cached expiry for us, so the Session does not have to re-handshake
        just because our JWT rolled over.

        Args:
            token: Our current node token.
        """
        if self.state != SessionState.ESTABLISHED:
            return
        await self._send_payload(SESSION_CONTROL_PROTOCOL, SessMsg(SessMsg.KIND_TOKEN_UPDATE, {
            "session_id": self.session_id, "token": token or "",
        }))

    async def close(self, reason_code: int = GRACEFUL_SHUTDOWN) -> None:
        """Sends GOAWAY to the remote peer and enters CLOSING (async).

        The FSM completes via process_inbound(GoawayAck) or check_timeout. Closable states are the two live ones plus
        AWAITING_HANDOFF: a world host shutting down must be able to GOAWAY a parked grant carrier too, instead of
        leaving it to die of timeout after the process is gone.

        Args:
            reason_code: The reason code carried by the GOAWAY (default: GRACEFUL_SHUTDOWN).
        """
        if self.state not in (SessionState.ESTABLISHED, SessionState.RECONNECTING,
                              SessionState.AWAITING_HANDOFF):
            log.session(f"{self!r} close({reason_code}) called in state {self.state.value} — ignored", sub=self._sub)
            return
        sent = await self._send_payload(SESSION_CONTROL_PROTOCOL, SessMsg(SessMsg.KIND_GOAWAY, {
            "reason": reason_code, "drain_window_ms": 0, "session_id": self.session_id,
        }))

        # The reason NAME is embedded in both sides' CLOSED reason strings
        # (local_goaway_acked:<name> / goaway_ack_timeout:<name> here,
        # remote_goaway:<name> there): the application dispatch matches it by
        # substring, e.g. WORLD_HANDOFF_COMPLETE must not purge on either side.
        self._local_goaway_reason = reason_name(reason_code)
        if not sent:

            # The local write failed: the link is already gone and no ACK can ever
            # come. Enter CLOSING anyway (the terminal contract is unchanged) but
            # with the short fallback, so a shutdown drain is not held hostage.
            log.session(f"{self!r} GOAWAY send failed (closing via short timeout)", sub=self._sub)
            self._set_state(SessionState.CLOSING, _TIMEOUT_CLOSING_FAILED_SEND_S)
            return
        self._set_state(SessionState.CLOSING, _TIMEOUT_CLOSING_S)
        log.session(f"{self!r} GOAWAY sent (reason={self._local_goaway_reason})", sub=self._sub)

    def check_timeout(self, now: float | None = None) -> SessionEvent | None:
        """Checks whether the timeout of the current state fired, purely synchronous (no sends).

        Args:
            now: The current monotonic time, or None to read it here (default: None).

        Returns:
            A single SessionEvent when the timeout fired and the state reacts with an event, otherwise None.
        """
        if self._timeout_s is None:
            return None
        if now is None:
            now = time.monotonic()
        elapsed = now - self._state_entered_at
        if elapsed < self._timeout_s:
            return None

        log.error(f"{self!r} TIMEOUT after {elapsed:.1f}s in state {self.state.value}", sub=self._sub)

        if self.state in (SessionState.CONNECTING,
                          SessionState.HELLO_SENT,
                          SessionState.PROFILE_SENT,
                          SessionState.HELLO_ACK_SENT,
                          SessionState.AWAITING_WORLD_GRANT,
                          SessionState.AWAITING_HANDOFF):
            return self._fail(f"timeout:{self.state.value}")

        if self.state == SessionState.RECONNECTING:
            return self._lose("peer_lost_session" if self._peer_lost_session else "grace_period_expired")

        if self.state == SessionState.CLOSING:
            return self._finalize_close(
                f"goaway_ack_timeout:{getattr(self, '_local_goaway_reason', '')}")

        return None

    def on_transport_disconnected(self) -> SessionEvent | None:
        """Handles a dropped link (transport diff) or a session-level liveness failure.

        ESTABLISHED → RECONNECTING, arming the active re-dial + RESUME loop (tick_dial). Already reconnecting: the
        reconnect window starts over, never past twice its length.

        Returns:
            The RECONNECTING event when the Session leaves ESTABLISHED, otherwise None.
        """
        if self.state == SessionState.ESTABLISHED:
            return self._enter_reconnecting(_TIMEOUT_RECONNECT_S)

        # The window of a peer that lost the Session is not restarted: a transport loss
        # adds nothing to wait for.
        # The whole time in RECONNECTING never passes twice the window: a link that keeps
        # coming and going would otherwise keep the Session there with no way back.
        if self.state == SessionState.RECONNECTING and not self._peer_lost_session:
            now = time.monotonic()
            ran = now - self._state_entered_at
            self._state_entered_at = now
            window = self._reconnect_window_s if self._reconnecting_since is not None else self._timeout_s
            if self._reconnecting_since is not None:
                self._timeout_s = max(0.0, min(window, self._reconnecting_since + 2 * window - now))
            if self._timeout_s < window:
                log.session(f"{self!r} transport lost again {ran:.1f}s into the reconnect window, "
                            f"which starts over cut short to {self._timeout_s:.1f}s: "
                            f"{2 * window}s in RECONNECTING at most", sub=self._sub)
            else:
                log.session(f"{self!r} transport lost again {ran:.1f}s into the reconnect window, "
                            f"which starts over ({window}s)", sub=self._sub)
        return None

    def on_transport_reconnected(self) -> SessionEvent | None:
        """Handles the transport-diff reconnect signal.

        Do NOT blindly resume — a reconnect can be to a *restarted* peer that holds no Session (the /ipfs/ping
        false-resume bug). Just kick the RESUME handshake; only a valid RESUME_ACK promotes us back to ESTABLISHED
        (see _on_resume_ack / _resume_established). A responder has no RESUME to send: it probes at once on the
        connection that came back, and the answer decides (_on_pong).

        Returns:
            Always None (the outcome arrives later, with the RESUME_ACK or the PONG).
        """
        if self.state == SessionState.RECONNECTING:
            self._transport_is_back("transport reconnect signal")
        return None

    async def tick_keepalive(self, now: float | None = None) -> SessionEvent | None:
        """Times out an unanswered ping, then sends a new one if due (async).

        The PONG itself is handled inbound (_on_pong). The keep-alive PING is sent inline: a fire-and-forget task would
        only start when the loop yields, which the mostly-synchronous main loop cannot guarantee. The probe of a
        responder that is reconnecting is the exception, see _start_probe.

        Args:
            now: The current monotonic time, or None to read it here (default: None).

        Returns:
            At most one SessionEvent (RECONNECTING on the miss threshold), otherwise None.
        """
        if now is None:
            now = time.monotonic()

        # Reap the send of the last probe. Its outcome changes nothing, it is only logged.
        task = self._probe_task
        if task is not None and task.done():
            self._probe_task = None
            if not task.cancelled() and not task.result():   # _run_probe never raises
                log.session(f"{self!r} probe PING send failed", sub=self._sub)

                # That PING never left, so no answer can come: the slot is free for the
                # probe that a transport coming back asks for. The period is untouched.
                if self._is_probing():
                    self._ping_in_flight = False

        # Apply a TOKEN_UPDATE that arrived before we were ESTABLISHED.
        if self._buffered_token_update is not None and self.state == SessionState.ESTABLISHED:
            buf, self._buffered_token_update = self._buffered_token_update, None
            await self._apply_token_update(buf)

        # GOAWAY a session whose peer token has lapsed past the grace window.
        # Uses wall-clock (the JWT exp is unix seconds), unlike the monotonic ping
        # timers. The generous grace + PING/PONG liveness mean this rarely fires.
        if (self.state == SessionState.ESTABLISHED and self.remote_token_exp > 0 and
                time.time() > self.remote_token_exp + _TOKEN_EXP_GRACE_S):
            log.session(f"{self!r} peer token expired past grace → GOAWAY(TOKEN_EXPIRED)", sub=self._sub)
            await self.close(TOKEN_EXPIRED)
            return None

        # An in-flight ping that has not been PONGed within the RTT window is a miss. A probe
        # is not counted: the Session is already reconnecting, silence adds nothing.
        probing = self._is_probing()
        if self._ping_in_flight and now >= self._ping_deadline:
            self._ping_in_flight = False
            if probing:
                log.session(f"{self!r} probe unanswered", sub=self._sub)
            else:
                self._consecutive_ping_failures += 1
                log.session(f"{self!r} ping miss ({self._consecutive_ping_failures}/{_KEEPALIVE_FAIL_THRESHOLD})",
                            sub=self._sub)
                if (self._consecutive_ping_failures >= _KEEPALIVE_FAIL_THRESHOLD and
                        self.state == SessionState.ESTABLISHED):
                    self._consecutive_ping_failures = 0
                    return self.on_transport_disconnected()

        # Schedule a new ping when due, one in flight at a time: the keep-alive of an
        # ESTABLISHED Session, or the probe of a responder that is reconnecting.
        if ((self.state == SessionState.ESTABLISHED or probing) and
                not self._ping_in_flight and
                now - self._last_ping_at >= _KEEPALIVE_PERIOD_S):
            self._last_ping_at = now
            self._ping_in_flight = True
            self._ping_nonce = secrets.token_bytes(8)
            self._ping_deadline = now + _PING_RTT_TIMEOUT_S
            if probing:
                self._start_probe()
            else:
                await self._send_ping()
        return None

    def request_early_ping(self) -> None:
        """Makes the next keep-alive PING due at once (on the next tick_keepalive), if none is in flight.

        Only for an ESTABLISHED Session; the miss counting is unchanged, so this only shortens the time to detect a
        dead peer after a sign of trouble (a failed send).
        """
        if self.state == SessionState.ESTABLISHED and not self._ping_in_flight:
            self._last_ping_at = float("-inf")

    def abort_silently(self, reason: str) -> None:
        """Drops this Session without emitting an event.

        Used by the SessionManager simultaneous-open tie-breaker on the losing initiator.

        Args:
            reason: Why the Session is dropped (logged).
        """
        self._set_state(SessionState.FAILED, None)
        log.session(f"{self!r} aborted silently: {reason}", sub=self._sub)

    def _start_dial(self) -> None:
        """Launches the dial as a background task (single-flight: no-op if one is already in flight).

        Non-blocking; only _dial_task is touched here — the FSM mutates only when tick_dial reaps the outcome, on the
        main task, so the single-writer invariant holds.
        """
        if self._dial_task is not None and not self._dial_task.done():
            return
        if self.state == SessionState.RECONNECTING:
            log.session(f"{self!r} reconnect dial (attempt {self._reconnect_attempts + 1})", sub=self._sub)
        else:
            self._dial_attempts += 1
            log.session(f"{self!r} dial attempt {self._dial_attempts}/{self._max_dial_attempts}",
                        sub=self._sub)
        self._dial_task = asyncio.create_task(self._run_dial(self.p2p, self.addresses))

    @staticmethod
    async def _run_dial(p2p: 'P2P', addresses: list[str]) -> tuple[str | None, bool]:
        """Runs the dial, as a background task (async).

        It reads only its arguments (no Session state), so the only new interleaving it introduces is "a result becomes
        available". It never raises.

        Args:
            p2p: The P2P instance (or its JS mirror) to dial with.
            addresses: The multiaddrs of the peer.

        Returns:
            The tuple (peer_id or None, through_relay).
        """
        try:
            return await p2p.connect(addresses)
        except Exception as e:
            log.error(f"dial raised: {e}")
            return None, False

    def _tick_address_lookup(self, now: float) -> None:
        """Takes the addresses a finished lookup brought and starts a new lookup when due, while RECONNECTING.

        A lookup starts when an attempt failed since the last one, at most once every _ADDRESS_LOOKUP_INTERVAL_S. The
        peer may have moved, and the addresses this Session started with are all it dials. The lookup is a task, as the
        dial is: the tick that starts it does not wait for it.

        Args:
            now: The current monotonic time.
        """
        task = self._lookup_task
        if task is not None and task.done():
            self._lookup_task = None
            if not task.cancelled():
                self._take_looked_up_addresses(task.result(), now)
        if (self._address_lookup is not None and self.role == SessionRole.INITIATOR
                and self.remote_node_id_hint and self._lookup_due and self._lookup_task is None
                and now >= self._next_lookup_at):
            self._lookup_due = False
            self._next_lookup_at = now + _ADDRESS_LOOKUP_INTERVAL_S
            self._lookup_task = asyncio.create_task(
                self._run_lookup(self._address_lookup, self.remote_node_id_hint))

    @staticmethod
    async def _run_lookup(address_lookup: Callable[[str], Awaitable[list[str] | None]], node_id: str) \
            -> list[str] | None:
        """Runs the address lookup, as a background task (async).

        It reads only its arguments and never raises.

        Args:
            address_lookup: The async callable, node_id -> addresses, given by the Node layer.
            node_id: The node whose addresses are looked up.

        Returns:
            The addresses found, or None when there are none or the lookup failed.
        """
        try:
            found = await address_lookup(node_id)
            return [str(a) for a in found] if found else None
        except Exception as e:
            log.error(f"address lookup of node {node_id} failed: {e}")
            return None

    def _take_looked_up_addresses(self, found: list[str] | None, now: float) -> None:
        """Adopts the looked-up addresses of this peer when they differ from the ones held.

        Only the addresses of this peer id are kept (another peer id is another process, which a RESUME cannot reach); a
        list that differs from the one held replaces it, and the next dial is made at once unless a dial or a RESUME is
        in flight.

        Args:
            found: The addresses returned by the lookup, or None.
            now: The current monotonic time.
        """
        mine = [a for a in (found or []) if a.rsplit("/p2p/", 1)[-1].split("/")[0] == self.peer_id]
        if not mine or set(mine) == set(self.addresses):
            log.session(f"{self!r} address lookup: nothing new for this peer", sub=self._sub)
            return
        fresh = [a for a in mine if a not in self.addresses]
        self.addresses = mine
        log.session(f"{self!r} addresses of the peer refreshed from the root server: "
                    f"{len(mine)}, {len(fresh)} new", sub=self._sub)
        if self._dial_task is None and self._reconnect_substep == "redial":
            self._next_redial_at = now

    async def _on_dial_done_connecting(self, peer_id: str | None, through_relay: bool, now: float) \
            -> list[SessionEvent]:
        """Handles the outcome of a dial while CONNECTING: sends the HELLO, or schedules a retry, or fails (async).

        Args:
            peer_id: The peer id the dial reached, or None if it failed.
            through_relay: Whether the connection goes through a relay circuit.
            now: The current monotonic time.

        Returns:
            The events generated (possibly empty).
        """
        if peer_id is not None and through_relay and self._reject_relay:
            log.error(f"{self!r} dial resolved through a relay circuit but relay nodes are "
                      f"disallowed (talk_to_relay_based_nodes=False)", sub=self._sub)
            return [self._fail("relay_rejected")]
        if peer_id is None:
            return self._dial_attempt_failed(now, "dial failed", "dial_failed")

        # If the dial returned a different peer-id than we expected (e.g. relayed multiaddr
        # without the /p2p hop), update it before the HELLO.
        if peer_id != self.peer_id:
            log.session(f"{self!r} peer_id updated from ...{self.peer_id[-5:]} → ...{peer_id[-5:]}", sub=self._sub)
            self.peer_id = peer_id
        return await self._send_hello(now)

    def _dial_attempt_failed(self, now: float, what: str, reason: str) -> list[SessionEvent]:
        """Handles a failed dial attempt of CONNECTING: the next one after a jittered backoff, or FAILED.

        Args:
            now: The current monotonic time.
            what: What failed, for the log.
            reason: The reason of the FAILED event once the attempts are used up.

        Returns:
            An empty list while attempts are left, otherwise the FAILED event.
        """
        if self._dial_attempts < self._max_dial_attempts:
            delay = min(
                random.uniform(0, _RETRY_BASE_S * (2 ** (self._dial_attempts - 1))),
                _RETRY_CAP_S,
            )
            self._next_dial_at = now + delay
            log.session(f"{self!r} {what}, retrying in {delay:.1f}s", sub=self._sub)
            return []
        log.error(f"{self!r} {what} after {self._max_dial_attempts} attempts", sub=self._sub)
        return [self._fail(reason)]

    async def _on_dial_done_reconnecting(self, peer_id: str | None, through_relay: bool, now: float) \
            -> list[SessionEvent]:
        """Handles the outcome of a dial while RECONNECTING: sends the RESUME, or schedules the next dial (async).

        Args:
            peer_id: The peer id the dial reached, or None if it failed.
            through_relay: Whether the connection goes through a relay circuit.
            now: The current monotonic time.

        Returns:
            An empty list (the outcome arrives with the RESUME_ACK).
        """
        if peer_id is not None and through_relay and self._reject_relay:
            peer_id = None   # relay-only reconnect disallowed; treat it as a failed dial
        if peer_id is None:

            # The substep is left alone: a RESUME may be in flight on a transport that came
            # back while this dial ran, and its answer is still to be taken.
            self._reconnect_attempts += 1
            self._lookup_due = True
            delay = min(_RETRY_BASE_S * (2 ** (self._reconnect_attempts - 1)), _RETRY_CAP_S)
            self._next_redial_at = now + delay
            log.session(f"{self!r} reconnect dial failed, next try in {delay:.1f}s", sub=self._sub)
            return []
        if peer_id != self.peer_id:
            self.peer_id = peer_id
        if self._reconnect_substep == "resume_sent" and now < self._resume_deadline:

            # That RESUME left while this dial ran. Its answer decides, and the dial is
            # tried again when its wait is over.
            self._next_redial_at = self._resume_deadline
            return []

        # Transport back — try to RESUME the existing session before re-handshaking.
        await self._send_resume(now)
        self._next_redial_at = self._resume_deadline
        log.session(f"{self!r} reconnect dial ok, RESUME sent", sub=self._sub)
        return []

    async def _send_resume(self, now: float) -> bool:
        """Sends RESUME on whatever transport there is and opens the wait for its answer (async).

        One is in flight at a time: sent again inside that wait it keeps its nonce, so the answer to either copy is
        taken.

        Args:
            now: The current monotonic time.

        Returns:
            Whether the transport accepted the RESUME.
        """
        if self._reconnect_substep != "resume_sent":
            self._resume_nonce = secrets.token_bytes(8)
        self._reconnect_substep = "resume_sent"
        self._resume_deadline = now + _RESUME_ACK_TIMEOUT_S
        sent = await self._send_payload(SESSION_CONTROL_PROTOCOL, SessMsg(SessMsg.KIND_RESUME, {
            "session_id": self.session_id, "nonce": self._resume_nonce,
        }))
        if not sent:
            self._lookup_due = True
        return sent

    async def _send_hello(self, now: float | None = None) -> list[SessionEvent]:
        """Sends the HELLO of an initiator and enters HELLO_SENT (async).

        Args:
            now: The current monotonic time, or None to read it when needed (default: None).

        Returns:
            The events generated (empty on success; a retry or a FAILED event when the send fails).
        """
        self.nonce_a = secrets.token_bytes(_NONCE_BYTES)
        msg = SessMsg(SessMsg.KIND_HELLO_REQ, {
            "session_id": self.session_id,
            "nonce_a": self.nonce_a,
            "protocol_versions": ["1.0"],
            "capabilities": self._local_capabilities(),
            "client_id": "python/1.0",
        })
        if not await self._send_payload(SESSION_HELLO_PROTOCOL, msg):

            # Right after a dial, the dial resolved on a connection that carries nothing (one
            # that was closing, or whose peer had just cut it): a dial that failed. Should
            # the HELLO have arrived all the same, the responder answers it again.
            if self.state == SessionState.CONNECTING:
                return self._dial_attempt_failed(time.monotonic() if now is None else now,
                                                 "HELLO could not be sent on the connection of the dial",
                                                 "hello_send_failed")
            return [self._fail("hello_send_failed")]
        self._set_state(SessionState.HELLO_SENT, _TIMEOUT_HELLO_S)
        self._hello_msg = msg
        self._hello_sent_at = time.monotonic() if now is None else now
        self._hello_repeats = 0
        log.session(f"{self!r} HELLO sent", sub=self._sub)
        return []

    async def _repeat_hello_if_due(self, now: float) -> None:
        """In HELLO_SENT with no answer: sends the same HELLO again, same session_id and nonce_a, so that an answer to
        either verifies (async). The responder answers a repeated HELLO (see _responder_on_repeated_hello), and the
        timeout of the stage is not restarted. A send that fails here fails nothing: the stage still waits for its
        answer.

        Args:
            now: The current monotonic time (seconds).
        """
        if (self._hello_msg is None or self._hello_repeats >= _MAX_HELLO_REPEATS or
                now < self._hello_sent_at + _HELLO_REPEAT_AFTER_S):
            return
        self._hello_repeats += 1
        self._hello_sent_at = now
        sent = await self._send_payload(SESSION_HELLO_PROTOCOL, self._hello_msg)
        log.session(f"{self!r} no HELLO_ACK within {_HELLO_REPEAT_AFTER_S:.0f}s, HELLO "
                    f"{'sent again' if sent else 'could not be sent again'} "
                    f"({self._hello_repeats}/{_MAX_HELLO_REPEATS})", sub=self._sub)

    async def _on_hello_protocol(self, msg: SessMsg) -> list[SessionEvent]:
        """Dispatches a message of the HELLO protocol according to role and state (async).

        Args:
            msg: The message.

        Returns:
            The events generated (possibly empty).
        """
        if (msg.kind == SessMsg.KIND_HELLO_REQ and
                self.role == SessionRole.RESPONDER and self.state == SessionState.INIT):
            return await self._responder_on_hello(msg)

        if (msg.kind == SessMsg.KIND_HELLO_REQ and
                self.role == SessionRole.RESPONDER and self.state == SessionState.HELLO_ACK_SENT and
                bytes(msg.get("session_id") or b"") == self.session_id):
            return await self._responder_on_repeated_hello(msg)

        if (msg.kind == SessMsg.KIND_HELLO_RESP and
                self.role == SessionRole.INITIATOR and self.state == SessionState.HELLO_SENT):
            return await self._initiator_on_hello_ack(msg)

        log.session(f"{self!r} ignored HELLO-protocol {msg.kind} in state {self.state.value}", sub=self._sub)
        return []

    async def _responder_on_hello(self, req: SessMsg) -> list[SessionEvent]:
        """Answers the HELLO of an initiator (version negotiation, signed HELLO_ACK) and enters HELLO_ACK_SENT (async).

        Args:
            req: The HELLO request.

        Returns:
            The events generated (empty on success, a FAILED event otherwise).
        """
        session_id = req.get("session_id") or b""
        if not session_id:
            return [self._fail("hello_missing_session_id")]
        self.session_id = bytes(session_id)
        self.session_id_hex = self.session_id.hex()
        self.nonce_a = bytes(req.get("nonce_a") or b"")
        self.peer_capabilities = int(req.get("capabilities") or 0)

        # Version negotiation — single supported version for v1.
        chosen = "1.0" if "1.0" in list(req.get("protocol_versions") or []) else ""
        if not chosen:
            await self._send_payload(SESSION_HELLO_PROTOCOL, SessMsg(SessMsg.KIND_HELLO_RESP, {
                "chosen_version": "", "capabilities": 0, "nonce_b": b"",
                "sig_b": b"", "reason_code": VERSION_UNSUPPORTED,
            }))
            return [self._fail("version_unsupported")]

        self.nonce_b = secrets.token_bytes(_NONCE_BYTES)
        sig_b = await self.p2p.sign_with_host_key(_sign_payload(self.nonce_a, self.session_id))
        if sig_b is None:
            return [self._fail("hello_sign_failed")]

        resp = SessMsg(SessMsg.KIND_HELLO_RESP, {
            "chosen_version": chosen, "capabilities": self._local_capabilities(), "nonce_b": self.nonce_b,
            "sig_b": sig_b, "reason_code": OK,
        })
        if not await self._send_payload(SESSION_HELLO_PROTOCOL, resp):
            return [self._fail("hello_ack_send_failed")]
        self._set_state(SessionState.HELLO_ACK_SENT, _TIMEOUT_PROFILE_S)
        log.session(f"{self!r} HELLO_ACK sent", sub=self._sub)
        return []

    async def _responder_on_repeated_hello(self, req: SessMsg) -> list[SessionEvent]:
        """Answers again the HELLO of this Session (async).

        Its initiator was told that the send of the first one failed, although that frame arrived, and sent it again
        after a new dial. It is answered over its own nonce_a, the only answer that initiator still takes, with the
        nonce_b already given, so that the PROFILE_OFFER verifies whichever answer it took. The wait for that offer is
        not restarted, and past a ceiling a repeat is ignored.

        Args:
            req: The repeated HELLO request.

        Returns:
            The events generated (empty on success, a FAILED event otherwise).
        """
        if self._repeated_hellos_answered >= _MAX_REPEATED_HELLOS_ANSWERED:
            log.session(f"{self!r} repeated HELLO ignored, {self._repeated_hellos_answered} already "
                        f"answered", sub=self._sub)
            return []
        self._repeated_hellos_answered += 1
        sig_b = await self.p2p.sign_with_host_key(
            _sign_payload(bytes(req.get("nonce_a") or b""), self.session_id))
        if sig_b is None:
            return [self._fail("hello_sign_failed")]
        resp = SessMsg(SessMsg.KIND_HELLO_RESP, {
            "chosen_version": "1.0", "capabilities": self._local_capabilities(), "nonce_b": self.nonce_b,
            "sig_b": sig_b, "reason_code": OK,
        })
        if not await self._send_payload(SESSION_HELLO_PROTOCOL, resp):
            return [self._fail("hello_ack_send_failed")]
        log.session(f"{self!r} HELLO_ACK sent again, to a repeated HELLO", sub=self._sub)
        return []

    async def _initiator_on_hello_ack(self, resp: SessMsg) -> list[SessionEvent]:
        """Handles the HELLO_ACK of the responder: verifies it and sends the PROFILE_OFFER (async).

        Args:
            resp: The HELLO response.

        Returns:
            The events generated (possibly empty).
        """
        reason_code = resp.get("reason_code", OK)
        if reason_code != OK:
            return [self._fail(f"hello_ack_rejected:{reason_name(reason_code)}")]
        if resp.get("chosen_version") != "1.0":
            return [self._fail(f"hello_ack_bad_version:{resp.get('chosen_version')}")]
        self.peer_capabilities = int(resp.get("capabilities") or 0)

        # Verify sig_b under the responder's peer-id (libp2p host key). A failed
        # verify is IGNORED, not failed: HELLO_RESP carries no session_id, so an
        # ACK of a previous generation (e.g. a responder draining a backlog answers
        # a stale HELLO and the current one back to back) is indistinguishable from
        # a forgery, and failing here would kill the handshake the good ACK — often
        # in the same batch — is about to complete. Staying in HELLO_SENT keeps the
        # wait bounded by the stage timeout.
        if not await self.p2p.verify_peer_signature(
            self.peer_id,
            _sign_payload(bytes(self.nonce_a), self.session_id), bytes(resp.get("sig_b") or b""),
        ):
            log.session(f"{self!r} HELLO_ACK with invalid sig_b ignored (stale generation "
                        f"or forgery)", sub=self._sub)
            return []

        self.nonce_b = bytes(resp.get("nonce_b") or b"")
        return await self._initiator_send_profile_offer()

    async def _initiator_send_profile_offer(self) -> list[SessionEvent]:
        """Sends the signed PROFILE_OFFER (with our profile and token) and enters PROFILE_SENT (async).

        Returns:
            The events generated (empty on success, a FAILED event otherwise).
        """
        sig_a = await self.p2p.sign_with_host_key(_sign_payload(self.nonce_b, self.session_id))
        if sig_a is None:
            return [self._fail("profile_offer_sign_failed")]
        profile_bytes = json.dumps(self._local_profile_factory()).encode("utf-8")
        offer = SessMsg(SessMsg.KIND_PROFILE_OFFER, {
            "profile": profile_bytes,
            "sig_a": sig_a,
            "session_id": self.session_id,

            # RT: present our root-issued node token in-band. The responder verifies
            # it against our transport peer-id and trusts the attested node_id. Empty
            # when no token_factory is injected (the v1 / FSM-test path). The public
            # <-> world binding the SignedPeerRecord used to carry now rides the
            # token's p2p_peers, so the offer no longer ships a signed_peer_record.
            "token": self.token_factory() if self.token_factory is not None else "",
        })
        if not await self._send_payload(SESSION_PROFILE_PROTOCOL, offer):
            return [self._fail("profile_offer_send_failed")]
        self._set_state(SessionState.PROFILE_SENT, _TIMEOUT_PROFILE_S)
        log.session(f"{self!r} PROFILE_OFFER sent", sub=self._sub)
        return []

    async def _on_profile_protocol(self, msg: SessMsg) -> list[SessionEvent]:
        """Dispatches a message of the PROFILE protocol according to role and state (async).

        Args:
            msg: The message.

        Returns:
            The events generated (possibly empty).
        """
        if (msg.kind == SessMsg.KIND_PROFILE_OFFER and
                self.role == SessionRole.RESPONDER and self.state == SessionState.HELLO_ACK_SENT):
            return await self._responder_on_profile_offer(msg)

        if (msg.kind == SessMsg.KIND_PROFILE_DECISION and
                self.role == SessionRole.INITIATOR and self.state == SessionState.PROFILE_SENT):
            return await self._initiator_on_profile_decision(msg)

        log.session(f"{self!r} ignored PROFILE-protocol {msg.kind} in state {self.state.value}", sub=self._sub)
        return []

    async def _verify_peer_token(self, token: str) -> tuple[VerifiedTokenClaims | None, str | None]:
        """Verifies an in-band peer token (RT) against the transport peer-id (async).

        Shared by both handshake directions (responder at PROFILE_OFFER, initiator at PROFILE_DECISION). On success the
        root-attested node-id is adopted into ``remote_node_id_hint``; when we already expected a specific node-id at
        this peer, a mismatch is rejected (we reached the right peer-id but the wrong node). On success the peer's
        attested token expiry / cv_hash are captured on the Session (``remote_token_exp`` / ``remote_cv_hash``) for the
        in-session refresh, together with the attested world-code hash (``remote_code_hash``, None when the token
        carries no such claim).

        Args:
            token: The token carried by the message.

        Returns:
            ``(claims, None)`` on success, ``(None, reason)`` on a verification or cross-check failure, and
            ``(None, None)`` when no verifier is configured (the v1 path).
        """
        if self.token_verifier is None:
            return None, None
        node_id, cv_hash, p2p_peers, exp, code_hash = await self.token_verifier(token or "", self.peer_id)
        if not node_id:
            return None, "bad_token"
        if self.remote_node_id_hint and self.remote_node_id_hint != node_id:
            return None, "token_node_id_mismatch"
        self.remote_node_id_hint = node_id
        self.remote_token_exp = exp
        self.remote_cv_hash = cv_hash
        self.remote_code_hash = code_hash
        return VerifiedTokenClaims(node_id=node_id, cv_hash=cv_hash,
                                   p2p_peers=p2p_peers, exp=exp, code_hash=code_hash), None

    async def _responder_on_profile_offer(self, offer: SessMsg) -> list[SessionEvent]:

        # Verify sig_a under the initiator's peer-id.
        """Handles the PROFILE_OFFER of an initiator: verifies it, asks the decision hook, answers (async).

        On acceptance it sends the PROFILE_DECISION and either issues a WORLD_GRANT (world host on its public instance)
        or enters ESTABLISHED.

        Args:
            offer: The profile offer.

        Returns:
            The events generated (ESTABLISHED, FAILED, or none while a grant waits to be redeemed).
        """
        if not await self.p2p.verify_peer_signature(
            self.peer_id,
            _sign_payload(bytes(self.nonce_b), self.session_id), bytes(offer.get("sig_a") or b""),
        ):
            return await self._reject_profile(NOT_AUTHORIZED, "bad_sig_a")

        # RT credential check. When a root token-verifier is configured, the offer
        # must carry a token the root vouches for, bound to this transport peer-id
        # (already proven by sig_a). The root-attested node_id then supersedes the
        # self-asserted profile node_id for the allow-list / grant binding below.
        # No verifier (the v1 / FSM-test path) → skip, keeping the profile-based path.
        verified_claims, terr = await self._verify_peer_token(offer.get("token") or "")
        if terr:
            return await self._reject_profile(NOT_AUTHORIZED, terr)

        # The initiator's public<->world binding is consumed from the verified token's
        # p2p_peers (see the decision hook), so the offer no longer carries a
        # SignedPeerRecord to verify here.
        try:
            self.remote_profile = json.loads(bytes(offer.get("profile") or b"").decode("utf-8"))
        except Exception as e:
            return await self._reject_profile(LOCAL_ERROR, f"bad_profile_json:{e}")

        # Learn the peer's application node-id from its profile when we weren't
        # given a hint at spawn time (an inbound responder never is). The world
        # host needs it both for the allow-list decision below and to bind the
        # WorldGrant signature to the joiner — the joiner verifies that signature
        # against its own local_node_id (see _responder_issue_world_grant /
        # _initiator_on_world_grant), so the two must agree on this value.
        # NodeProfile.get_all_profile() nests it under "static"; bare test
        # profiles may carry it top-level.
        if not self.remote_node_id_hint:
            nid = (self.remote_profile.get("node_id") or
                   (self.remote_profile.get("static") or {}).get("node_id"))
            if isinstance(nid, str) and nid:
                self.remote_node_id_hint = nid

        # Application decision: accept/reject + (world host) the grant spec. When no
        # hook is injected we accept (preserves the v1 agent happy-path default).
        if self._decision_hook is not None:
            try:
                decision = await self._decision_hook(
                    self.peer_id, self.remote_node_id_hint, self.remote_profile, self.p2p_name,
                    verified_claims)
            except Exception as e:
                log.error(f"{self!r} decision_hook raised: {e}", sub=self._sub)
                return await self._reject_profile(LOCAL_ERROR, f"decision_hook_error:{e}")
        else:
            decision = ProfileDecisionResult(accepted=True)

        if not decision.accepted:
            return await self._reject_profile(decision.reason_code, "decision_rejected")

        # Send the (positive) decision carrying our profile and, for RT mutual auth,
        # our own token so the initiator can verify us against our peer-id in turn.
        dec_msg = SessMsg(SessMsg.KIND_PROFILE_DECISION, {
            "accepted": True,
            "profile_b": json.dumps(self._local_profile_factory()).encode("utf-8"),
            "reason_code": OK,
            "session_id": self.session_id,
            "token": self.token_factory() if self.token_factory is not None else "",
        })
        if not await self._send_payload(SESSION_PROFILE_PROTOCOL, dec_msg):
            return [self._fail("profile_decision_send_failed")]

        # World host on its PUBLIC instance: issue a WORLD_GRANT. The joiner will
        # redeem it on a *separate* world Session; this public Session stays as the
        # grant carrier (no ESTABLISHED) until the joiner GOAWAYs it post-handoff.
        if decision.world_grant is not None:
            return await self._responder_issue_world_grant(decision.world_grant)

        # Agent host, or world host on its WORLD instance (the joiner redeeming a
        # grant): established immediately. The world host registers the agent on this
        # ESTABLISHED event (role from its own pre-admission); the extra PRIVATE_JOIN
        # round-trip was dropped since the grant is unsigned and the world Session is
        # already mutually token-authenticated.
        self._set_state(SessionState.ESTABLISHED, None)
        log.session(f"{self!r} ESTABLISHED (responder)", sub=self._sub)
        self._last_ping_at = time.monotonic()
        return [self._emit(SessionEventType.ESTABLISHED, {"profile_b": self.remote_profile})]

    async def _reject_profile(self, reason_code: int, log_reason: str) -> list[SessionEvent]:
        """Sends a negative PROFILE_DECISION and fails the Session (async).

        Args:
            reason_code: The reason code sent to the peer.
            log_reason: The reason recorded in the FAILED event.

        Returns:
            A list with the FAILED event.
        """
        await self._send_payload(SESSION_PROFILE_PROTOCOL, SessMsg(SessMsg.KIND_PROFILE_DECISION, {
            "accepted": False, "reason_code": reason_code, "session_id": self.session_id,
        }))
        return [self._fail(f"rejected_offer:{log_reason}")]

    async def _initiator_on_profile_decision(self, decision: SessMsg) -> list[SessionEvent]:
        """Handles the PROFILE_DECISION of the responder: verifies its token and profile, then goes on (async).

        A world joiner on the public Session waits for the WORLD_GRANT; any other initiator enters ESTABLISHED.

        Args:
            decision: The profile decision.

        Returns:
            The events generated (possibly empty).
        """
        if not decision.get("accepted"):
            reason = reason_name(decision.get("reason_code", 0))
            return [self._fail(f"profile_rejected:{reason}")]

        # RT mutual auth: verify the responder's in-band token against its peer-id
        # (it verified ours at PROFILE_OFFER). A bad token, or one whose attested
        # node-id is not the node we meant to reach, fails the handshake. No verifier
        # (the v1 path) → skip.
        _claims, terr = await self._verify_peer_token(decision.get("token") or "")
        if terr:
            return [self._fail(f"profile_decision_{terr}")]
        try:
            self.remote_profile = json.loads(bytes(decision.get("profile_b") or b"").decode("utf-8"))
        except Exception as e:
            return [self._fail(f"profile_decision_bad_profile_b:{e}")]

        # World joiner on the PUBLIC Session: expect a WORLD_GRANT next. The host
        # sends DECISION and GRANT back-to-back on different protocols, so the
        # grant can be popped first — consume a buffered one right away.
        if self.is_world_join:
            self._set_state(SessionState.AWAITING_WORLD_GRANT, _TIMEOUT_WORLD_S)
            log.session(f"{self!r} PROFILE accepted, awaiting WORLD_GRANT", sub=self._sub)
            if self._early_world_grant is not None:
                grant, self._early_world_grant = self._early_world_grant, None
                log.session(f"{self!r} consuming WORLD_GRANT buffered ahead of the decision",
                            sub=self._sub)
                return await self._initiator_on_world_grant(grant)
            return []

        # Plain agent connection, or a world joiner on the WORLD Session redeeming a
        # grant: established immediately. node.py branches on the grant carried by the
        # ESTABLISHED event (ev.data["grant"], set below) to run the world-join body and
        # retire the public Session, without reaching into the private Session._grant.
        # The PRIVATE_JOIN round-trip the world joiner used to send here was dropped.
        self._set_state(SessionState.ESTABLISHED, None)
        log.session(f"{self!r} ESTABLISHED (initiator)", sub=self._sub)

        # A world-join Session was opened (start_world_handoff) carrying the grant to
        # redeem; carry it on the event so Node drives the join body without reaching into
        # the private Session._grant. None for a plain agent connection.
        self._last_ping_at = time.monotonic()
        return [self._emit(SessionEventType.ESTABLISHED,
                           {"profile_b": self.remote_profile, "grant": self._grant})]

    async def _responder_issue_world_grant(self, spec: WorldGrantSpec) -> list[SessionEvent]:
        """Mints and sends a WORLD_GRANT, as a world host on its PUBLIC Session, then waits for the handoff (async).

        The joiner redeems it on the world Session. The grant is no longer signed (RT): the joiner trusts it because
        the public Session is mutually token-authenticated, and the world Session re-verifies the joiner's token while
        the host pre-admission (conn_routing) gates the binding.

        Args:
            spec: What the decision hook returned for the grant.

        Returns:
            The events generated (empty on success, a FAILED event otherwise).
        """
        self.grant_id = secrets.token_bytes(_GRANT_ID_BYTES)
        self.grant_expiry_unix_ms = int(time.time() * 1000) + int(spec.grant_ttl_ms)
        grant = SessMsg(SessMsg.KIND_WORLD_GRANT, {
            "grant_id": self.grant_id,
            "expiry_unix_ms": self.grant_expiry_unix_ms,
            "world_profile": spec.world_profile_bytes,
            "role": spec.role,
            "code_bundle": spec.code_bundle,
            "initial_stats": spec.initial_stats,
            "rendezvous_tag": spec.rendezvous_tag,
        })
        if not await self._send_payload(SESSION_WORLD_PROTOCOL, grant):
            return [self._fail("world_grant_send_failed")]

        # Park as the grant carrier: the joiner redeems on a separate world Session and
        # then GOAWAYs this public one (WORLD_HANDOFF_COMPLETE). No PRIVATE_JOIN here.
        self._set_state(SessionState.AWAITING_HANDOFF, _TIMEOUT_WORLD_S)
        log.session(f"{self!r} WORLD_GRANT sent, awaiting handoff GOAWAY", sub=self._sub)
        return []

    async def _on_world_protocol(self, msg: SessMsg) -> list[SessionEvent]:
        """Dispatches a message of the WORLD protocol according to role and state (async).

        A grant that arrives before the profile decision is buffered (no cross-protocol ordering).

        Args:
            msg: The message.

        Returns:
            The events generated (possibly empty).
        """
        if (msg.kind == SessMsg.KIND_WORLD_GRANT and
                self.role == SessionRole.INITIATOR and self.state == SessionState.AWAITING_WORLD_GRANT):
            return await self._initiator_on_world_grant(msg)

        # WORLD and PROFILE ride different libp2p protocols, so the pump gives no
        # cross-protocol ordering: the grant routinely lands BEFORE the decision.
        # Buffer it; _initiator_on_profile_decision consumes it.
        if (msg.kind == SessMsg.KIND_WORLD_GRANT and self.role == SessionRole.INITIATOR and
                self.is_world_join and self.state == SessionState.PROFILE_SENT):
            self._early_world_grant = msg
            log.session(f"{self!r} buffered WORLD_GRANT that arrived before the profile decision",
                        sub=self._sub)
            return []

        log.session(f"{self!r} ignored WORLD-protocol {msg.kind} in state {self.state.value}", sub=self._sub)
        return []

    async def _initiator_on_world_grant(self, grant: SessMsg) -> list[SessionEvent]:
        """Handles the WORLD_GRANT received by a joiner: checks it and enters ESTABLISHED carrying the grant (async).

        Args:
            grant: The world grant.

        Returns:
            The events generated (the ESTABLISHED event with the grant, or a FAILED event).
        """
        grant_id = grant.get("grant_id") or b""
        if not grant_id:
            return [self._fail("world_grant_missing_id")]
        expiry = int(grant.get("expiry_unix_ms", 0))
        if expiry < int(time.time() * 1000):
            return [self._fail("world_grant_expired")]

        # The grant is unsigned. The public Session is already mutually
        # token-authenticated, so the joiner trusts the grant without verifying a
        # host signature; the world Session re-verifies tokens on its own handshake.
        self.grant_id = bytes(grant_id)
        self.grant_expiry_unix_ms = expiry

        # No WORLD_GRANT_ACK is sent: nothing ever consumed it (the host parks in
        # AWAITING_HANDOFF waiting the joiner's GOAWAY, and a stray ack would just hit
        # its "ignored WORLD-protocol" log). The kind stays decodable for old joiners.
        # The public Session reaches ESTABLISHED carrying the grant blob; node.py
        # reacts by opening the world Session and, once it is ESTABLISHED, GOAWAYs
        # this public Session (WORLD_HANDOFF_COMPLETE).
        self._set_state(SessionState.ESTABLISHED, None)
        self._last_ping_at = time.monotonic()
        log.session(f"{self!r} WORLD_GRANT received, public session ESTABLISHED (handoff pending)", sub=self._sub)
        return [self._emit(SessionEventType.ESTABLISHED, {
            "profile_b": self.remote_profile,
            "world_grant": {
                "grant_id": self.grant_id.hex(),
                "expiry_unix_ms": self.grant_expiry_unix_ms,
                "world_profile_bytes": bytes(grant.get("world_profile") or b""),
                "role": int(grant.get("role", 0)),
                "rendezvous_tag": int(grant.get("rendezvous_tag", 0)),
                "code_bundle": bytes(grant.get("code_bundle") or b""),
                "initial_stats": bytes(grant.get("initial_stats") or b""),

                # Joiner-side annotation from the VERIFIED token claims of this
                # Session (never from the grant message): the root-attested hash
                # of the world's stored code, checked against the bundle at join.
                "world_code_hash_claim": self.remote_code_hash,
            },
        })]

    async def _on_control_protocol(self, msg: SessMsg) -> list[SessionEvent]:
        """Dispatches a message of the CONTROL protocol (GOAWAY, PING/PONG, RESUME, TOKEN_UPDATE) (async).

        Args:
            msg: The message.

        Returns:
            The events generated (possibly empty).
        """
        if msg.kind == SessMsg.KIND_GOAWAY:
            return await self._on_remote_goaway(msg)
        if msg.kind == SessMsg.KIND_GOAWAY_ACK and self.state == SessionState.CLOSING:
            if msg.get("received"):
                return [self._finalize_close(
                    f"local_goaway_acked:{getattr(self, '_local_goaway_reason', '')}")]
            return []
        if msg.kind == SessMsg.KIND_PING:
            return await self._on_ping(msg)
        if msg.kind == SessMsg.KIND_PONG:
            ev = self._on_pong(msg)
            return [ev] if ev is not None else []
        if msg.kind == SessMsg.KIND_RESUME:
            return await self._on_resume(msg)
        if msg.kind == SessMsg.KIND_RESUME_ACK:
            return await self._on_resume_ack(msg)
        if msg.kind == SessMsg.KIND_TOKEN_UPDATE:
            return await self._on_token_update(msg)
        log.session(f"{self!r} unrecognised CONTROL {msg.kind} in state {self.state.value}", sub=self._sub)
        return []

    async def _on_token_update(self, msg: SessMsg) -> list[SessionEvent]:
        """Handles an inbound refreshed peer token: buffered until ESTABLISHED, otherwise applied now (async).

        Args:
            msg: The TOKEN_UPDATE message.

        Returns:
            An empty list of events.
        """
        if self.token_verifier is None:
            return []
        if self.state != SessionState.ESTABLISHED:
            self._buffered_token_update = msg
            return []
        await self._apply_token_update(msg)
        return []

    async def _apply_token_update(self, msg: SessMsg) -> None:
        """Applies a refreshed peer token (async).

        It re-verifies the pushed token against the same transport peer-id, requires the attested node-id to be
        unchanged (a divergent one rejects the update, not the Session), and advances the cached expiry MONOTONICALLY
        so a replayed older token cannot shrink the validity window.

        Args:
            msg: The TOKEN_UPDATE message.
        """
        node_id, cv_hash, _p2p_peers, exp, code_hash = await self.token_verifier(msg.get("token") or "",
                                                                                 self.peer_id)
        if not node_id:
            log.session(f"{self!r} TOKEN_UPDATE failed verification — ignored", sub=self._sub)
            return
        if self.remote_node_id_hint and node_id != self.remote_node_id_hint:
            log.session(f"{self!r} TOKEN_UPDATE node_id mismatch — ignored", sub=self._sub)
            return
        if int(exp) > self.remote_token_exp:
            self.remote_token_exp = int(exp)
        if cv_hash:
            self.remote_cv_hash = cv_hash
        if code_hash:
            self.remote_code_hash = code_hash
        log.session(f"{self!r} TOKEN_UPDATE applied (exp={self.remote_token_exp})", sub=self._sub)

    async def _on_ping(self, msg: SessMsg) -> list[SessionEvent]:
        """Answers a peer's liveness PING (async).

        ok=True only if we hold this exact Session and it is usable: ESTABLISHED, mid-reconnect, or a grant carrier
        parked in AWAITING_HANDOFF — the joiner's carrier keepalives while its world dial runs, and an ok=False there
        would collapse the handoff at the first PING (~15s budget). A restarted peer with no Session never reaches
        here — SessionManager.deliver answers ok=False on its behalf.

        Args:
            msg: The PING message.

        Returns:
            An empty list of events.
        """
        ok = (bytes(msg.get("session_id") or b"") == self.session_id and
              self.state in (SessionState.ESTABLISHED, SessionState.RECONNECTING,
                             SessionState.AWAITING_HANDOFF))
        if not await self._send_payload(SESSION_CONTROL_PROTOCOL, SessMsg(SessMsg.KIND_PONG, {
            "session_id": self.session_id,
            "nonce": bytes(msg.get("nonce") or b""),
            "ok": ok,
        })):
            log.session(f"{self!r} PONG send failed", sub=self._sub)

        # A PING that names this Session, read while reconnecting: the peer still holds
        # it and the transport has just carried a frame of it.
        if ok and self.state == SessionState.RECONNECTING:
            self._transport_is_back("PING of the peer")
        return []

    def _on_pong(self, msg: SessMsg) -> SessionEvent | None:
        """Handles a PONG to our in-flight ping, purely synchronous (no sends).

        Args:
            msg: The PONG message.

        Returns:
            The event generated (RESUMED after a probe answered ok=true, RECONNECTING on ok=false), otherwise None.
        """
        if not self._ping_in_flight or bytes(msg.get("nonce") or b"") != self._ping_nonce:
            return None  # stale / unmatched
        self._ping_in_flight = False
        if msg.get("ok"):
            self._consecutive_ping_failures = 0

            # The answer to a probe: the peer still holds the Session, so the responder is
            # back. The proof is the one an accepted RESUME gives: the peer names our
            # session id, echoes the nonce of the probe in flight and answers from a usable
            # state (_on_ping), over a transport that has just worked in both directions.
            # A restarted peer cannot give it: its manager answers ok=false. This is the
            # way back when only this side saw the loss, since an initiator that is not
            # reconnecting sends no RESUME.
            if self._is_probing() and bytes(msg.get("session_id") or b"") == self.session_id:
                log.session(f"{self!r} PONG ok=true to the probe (peer holds our session)", sub=self._sub)
                return self._resume_established()
            return None

        # Peer holds no Session for our session_id (restarted / forgot us): a hard liveness
        # failure. An initiator begins the reconnect + RESUME path immediately. A responder
        # has no such path and waits its own short window, not the reconnect one.
        self._consecutive_ping_failures = 0
        if self._is_probing():

            # The answer to a probe. The layers above have already seen RECONNECTING: only
            # the window changes, counted from this answer.
            log.session(f"{self!r} PONG ok=false to the probe (peer lost our session), "
                        f"{_TIMEOUT_PEER_LOST_SESSION_S}s left", sub=self._sub)
            self._peer_lost_session = True
            self._state_entered_at = time.monotonic()
            self._timeout_s = _TIMEOUT_PEER_LOST_SESSION_S
            return None
        log.session(f"{self!r} PONG ok=false (peer lost our session) → reconnecting", sub=self._sub)
        if self.role == SessionRole.RESPONDER and self.state == SessionState.ESTABLISHED:
            return self._enter_reconnecting(_TIMEOUT_PEER_LOST_SESSION_S, peer_lost_session=True)
        return self.on_transport_disconnected()

    async def _on_resume(self, msg: SessMsg) -> list[SessionEvent]:
        """Answers the RESUME of a peer (initiator) resuming after a link blip (async).

        ACK if we still hold the Session and it is usable (AWAITING_HANDOFF included: a public blip during the joiner's
        world dial must not kill the parked carrier, whose RESUME would otherwise be NACKed into a LOST); NACK
        (SESSION_UNKNOWN) otherwise so the peer re-handshakes.

        Args:
            msg: The RESUME message.

        Returns:
            The RESUMED event when this side was reconnecting, otherwise an empty list.
        """
        if (bytes(msg.get("session_id") or b"") == self.session_id and
                self.state in (SessionState.ESTABLISHED, SessionState.RECONNECTING,
                               SessionState.AWAITING_HANDOFF)):
            await self._send_payload(SESSION_CONTROL_PROTOCOL, SessMsg(SessMsg.KIND_RESUME_ACK, {
                "session_id": self.session_id, "nonce": bytes(msg.get("nonce") or b""),
                "accepted": True, "reason_code": OK,
            }))
            if self.state == SessionState.RECONNECTING:
                return [self._resume_established()]
            return []
        await self._send_payload(SESSION_CONTROL_PROTOCOL, SessMsg(SessMsg.KIND_RESUME_ACK, {
            "session_id": bytes(msg.get("session_id") or b""), "nonce": bytes(msg.get("nonce") or b""),
            "accepted": False, "reason_code": SESSION_UNKNOWN,
        }))
        return []

    async def _on_resume_ack(self, msg: SessMsg) -> list[SessionEvent]:
        """Handles the answer to our RESUME: accepted → RESUMED; NACK → re-handshake (async).

        Args:
            msg: The RESUME_ACK message.

        Returns:
            The events generated (possibly empty).
        """
        if self.state != SessionState.RECONNECTING or self._reconnect_substep != "resume_sent":
            return []
        if bytes(msg.get("nonce") or b"") != self._resume_nonce:
            return []  # stale ack for a previous attempt
        if msg.get("accepted"):
            return [self._resume_established()]
        return await self._restart_handshake_or_fail(reason_name(msg.get("reason_code", SESSION_UNKNOWN)))

    def _resume_established(self) -> SessionEvent:
        """Goes back from RECONNECTING to ESTABLISHED, resetting liveness.

        Called after a confirmed RESUME, or after a probe of the responder that the peer answered with ok=true.

        Returns:
            The RESUMED event (carrying the stored peer profile).
        """
        self._set_state(SessionState.ESTABLISHED, None)
        self._last_ping_at = time.monotonic()
        self._consecutive_ping_failures = 0
        self._ping_in_flight = False
        self._reconnect_substep = ""
        self._reconnect_attempts = 0
        self._resume_due = False
        log.session(f"{self!r} session resumed → ESTABLISHED", sub=self._sub)

        # Carry the stored peer profile on the event so Node can re-enrol from RESUMED
        # without reaching into Session.remote_profile (a fast RESUME keeps it; only a
        # full re-handshake clears it). The profile is a plain dict, no protobuf.
        return self._emit(SessionEventType.RESUMED, {"profile": self.remote_profile})

    async def _restart_handshake_or_fail(self, reason: str) -> list[SessionEvent]:
        """Reacts to a RESUME NACK: the peer does not recognise our Session (async).

        A plain initiator Session re-runs the full handshake over the (already re-dialed) transport, preserving the
        application-visible peer-id/node-id. World Sessions can't be transparently re-joined (double-join risk) → LOST
        so the node re-drives the join.

        Args:
            reason: The reason name of the NACK.

        Returns:
            The events generated (LOST, or those of the new HELLO).
        """
        if self.role != SessionRole.INITIATOR or self.is_world_join or self._grant is not None:
            self._set_state(SessionState.FAILED, None)
            log.session(f"{self!r} resume NACK ({reason}); not re-handshakeable → LOST", sub=self._sub)
            return [self._emit(SessionEventType.LOST, {"reason": f"resume_nack:{reason}"})]
        log.session(f"{self!r} resume NACK ({reason}) → re-running handshake", sub=self._sub)
        self.session_id = secrets.token_bytes(_SESSION_ID_BYTES)
        self.session_id_hex = self.session_id.hex()
        self.nonce_a = b""
        self.nonce_b = b""
        self.remote_profile = None
        self._reconnect_substep = ""
        self._reconnect_attempts = 0
        self._resume_due = False
        self._consecutive_ping_failures = 0
        self._ping_in_flight = False
        return await self._send_hello()

    async def _on_remote_goaway(self, ga: SessMsg) -> list[SessionEvent]:

        # A GOAWAY from a PREVIOUS incarnation (cross-protocol reordering: the
        # peer's replace-stale sends GOAWAY(old) and HELLO(new) back to back on
        # different one-shot streams) must not kill this fresh Session. Courtesy-ACK
        # it so the closer's CLOSING completes, but stay alive. An empty session_id
        # is tolerated (older senders); PING/RESUME have the same id guard.
        """Handles a GOAWAY from the peer: acknowledges it and closes the Session, unless it names another one (async).

        A GOAWAY from a PREVIOUS incarnation (cross-protocol reordering) is acknowledged but does not close this one.

        Args:
            ga: The GOAWAY message.

        Returns:
            The CLOSED event, or an empty list for a stale GOAWAY.
        """
        sid = bytes(ga.get("session_id") or b"")
        if sid and sid != self.session_id:
            await self._send_payload(SESSION_CONTROL_PROTOCOL, SessMsg(SessMsg.KIND_GOAWAY_ACK,
                                                                       {"received": True}))
            log.session(f"{self!r} stale GOAWAY for another incarnation — acked, ignored",
                        sub=self._sub)
            return []
        await self._send_payload(SESSION_CONTROL_PROTOCOL, SessMsg(SessMsg.KIND_GOAWAY_ACK,
                                                                   {"received": True}))
        reason = reason_name(ga.get("reason", 0))
        return [self._finalize_close(f"remote_goaway:{reason}")]

    def _enter_reconnecting(self, window_s: float, peer_lost_session: bool = False) -> SessionEvent:
        """Goes from ESTABLISHED to RECONNECTING with the given window.

        The event is the same whatever the cause, so the layers above see one RECONNECTING and then its outcome.

        Args:
            window_s: The length of the reconnect window, in seconds.
            peer_lost_session: True when the peer said it lost the Session (responder only) (default: False).

        Returns:
            The RECONNECTING event.
        """
        self._set_state(SessionState.RECONNECTING, window_s)
        self._reconnecting_since = self._state_entered_at
        self._reconnect_window_s = window_s
        self._lookup_due = False
        self._next_lookup_at = 0.0
        self._peer_lost_session = peer_lost_session
        self._reconnect_substep = "redial"
        self._reconnect_attempts = 0
        self._next_redial_at = 0.0   # re-dial on the next tick_dial
        self._resume_due = False
        self._ping_in_flight = False
        self._last_ping_at = float("-inf")   # a responder probes on the next tick_keepalive
        log.session(f"{self!r} link lost, starting {window_s}s reconnect window", sub=self._sub)
        return self._emit(SessionEventType.RECONNECTING, {"cause": "transport_disconnected"})

    def _transport_is_back(self, why: str) -> None:
        """Starts the way back after the transport toward the peer was seen working again, while RECONNECTING.

        The pools saw the peer connect, or a PING of this Session arrived. Whoever opened the connection, the way back
        starts now and on that transport. A responder probes (see _is_probing). An initiator owes a RESUME, sent by the
        next tick_dial with no dial before it: the dial goes to the addresses the Session started with, and a peer that
        moved meanwhile is reached only by the connection it opened itself. Sync, as the transport events are.

        Args:
            why: What showed the transport is back, for the log.
        """
        if self._is_probing():

            # Due on the next tick_keepalive, not at the next period. One probe in
            # flight at a time: if one is, this one waits for its answer or for the
            # end of its RTT window.
            self._last_ping_at = float("-inf")
            log.session(f"{self!r} {why} → probing", sub=self._sub)
        elif self.role == SessionRole.INITIATOR:
            if not self._reject_relay:
                self._resume_due = True
                log.session(f"{self!r} {why} → attempting RESUME", sub=self._sub)
            elif self._reconnect_substep != "resume_sent":

                # What came back may be a relay circuit, which this Session refuses, and
                # only the dial says which path it is: the dial is made at once. With a
                # RESUME in flight it has just been made.
                self._next_redial_at = 0.0
                log.session(f"{self!r} {why} → re-dialling", sub=self._sub)

    async def _send_ping(self) -> None:
        """Sends the keep-alive PING, logging a failed send (async).

        A send failure would otherwise be invisible until it surfaces as a ping miss.
        """
        if not await self._send_payload(SESSION_CONTROL_PROTOCOL, SessMsg(SessMsg.KIND_PING, {
            "session_id": self.session_id, "nonce": self._ping_nonce,
        })):
            log.session(f"{self!r} keepalive PING send failed", sub=self._sub)

    def _is_probing(self) -> bool:
        """Tells whether this is a responder that is reconnecting after a transport loss (and so probes).

        It cannot dial and would learn nothing until its window expires, so it asks: a PING on entering the state, at
        the keep-alive period and when the transport comes back, whose send re-opens a connection when it can. The
        answer decides (_on_pong): ok=false ends the wait early, ok=true brings the Session back. An initiator has its
        own path.

        Returns:
            True for a probing responder, False otherwise.
        """
        return (self.role == SessionRole.RESPONDER and
                self.state == SessionState.RECONNECTING and not self._peer_lost_session)

    def _start_probe(self) -> None:
        """Sends the probe PING as a background task, as the dial is and for the same reason.

        With the transport gone the send dials first, for as long as the transport gives a Session frame when the peer
        does not answer, and a node has one such send per reconnecting peer. The tick serves every Session and must not
        wait for them. Nothing is timed on this send (an unanswered probe is not a miss), so a late start costs nothing.
        The task reads only its arguments, see _run_probe.
        """
        if self._probe_task is not None and not self._probe_task.done():
            return
        ping = SessMsg(SessMsg.KIND_PING, {"session_id": self.session_id, "nonce": self._ping_nonce})
        log.session(f"{self!r} probe PING", sub=self._sub)
        self._probe_task = asyncio.create_task(self._run_probe(self.p2p, self.peer_id, ping))

    @staticmethod
    async def _run_probe(p2p: 'P2P', peer_id: str, ping: SessMsg) -> bool:
        """Runs the probe send, as a background task: no Session state is read or written (async).

        It never raises.

        Args:
            p2p: The P2P instance (or its JS mirror) to send with.
            peer_id: The peer to probe.
            ping: The PING message.

        Returns:
            Whether the PING was accepted by the transport.
        """
        try:
            return await p2p.send_session(peer_id=peer_id, protocol_id=SESSION_CONTROL_PROTOCOL,
                                          sess_msg=ping)
        except Exception as e:
            log.error(f"probe PING raised: {e}")
            return False

    def _set_state(self, state: SessionState, timeout_s: float | None) -> None:
        """Moves the FSM to a new state, restarting its timer.

        Args:
            state: The new state.
            timeout_s: The timeout of the new state in seconds, or None for no timeout.
        """
        prev = self.state
        self.state = state
        self._state_entered_at = time.monotonic()
        self._timeout_s = timeout_s
        log.session(f"{self!r} {prev.value} → {state.value} (timeout={timeout_s}s)", sub=self._sub)

    def _fail(self, reason: str) -> SessionEvent:
        """Moves the Session to FAILED.

        Args:
            reason: Why it failed.

        Returns:
            The FAILED event.
        """
        self._set_state(SessionState.FAILED, None)
        log.error(f"{self!r} FAILED: {reason}", sub=self._sub)
        return self._emit(SessionEventType.FAILED, {"reason": reason})

    def _lose(self, reason: str) -> SessionEvent:
        """Moves the Session to FAILED and reports it as LOST (it was established, then could not be recovered).

        Args:
            reason: Why it was lost.

        Returns:
            The LOST event.
        """
        self._set_state(SessionState.FAILED, None)
        log.error(f"{self!r} LOST: {reason}", sub=self._sub)
        return self._emit(SessionEventType.LOST, {"reason": reason})

    def _finalize_close(self, reason: str) -> SessionEvent:
        """Moves the Session to CLOSED.

        Args:
            reason: Why it closed.

        Returns:
            The CLOSED event.
        """
        self._set_state(SessionState.CLOSED, None)
        log.session(f"{self!r} CLOSED: {reason}", sub=self._sub)
        return self._emit(SessionEventType.CLOSED, {"reason": reason})

    def _emit(self, event_type: SessionEventType, data: dict) -> SessionEvent:
        """Builds an event of this Session.

        Args:
            event_type: The type of the event.
            data: The data carried by the event.

        Returns:
            The event.
        """
        if event_type in (SessionEventType.ESTABLISHED, SessionEventType.RESUMED):
            data = {**data, "peer_capabilities": self.peer_capabilities}
        return SessionEvent(
            type=event_type,
            session_id=self.session_id_hex,
            peer_id=self.peer_id,
            role=self.role,
            p2p_name=self.p2p_name,
            data=data,
        )

    def _local_capabilities(self) -> int:
        """The capability bits this end advertises: what its transport can do, not what this code knows.

        The browser runs this same code on a transport without chunked frames or file streaming: it advertises 0.
        CAP_FILE_STREAM says that this end takes file messages (it can, and Custom.ACCEPT_FILES says it does).

        Returns:
            The CAP_* bits.
        """
        caps = CAP_CHUNKED_FRAMES if getattr(self.p2p, "supports_chunked_frames", False) else 0
        return caps | (CAP_FILE_STREAM if getattr(self.p2p, "accepts_file_stream", False) else 0)

    async def _send_payload(self, protocol_id: str, msg: SessMsg) -> bool:
        """Sends a Session message to the peer on the given protocol (async).

        Args:
            protocol_id: The Session protocol to send on.
            msg: The message.

        Returns:
            Whether the transport accepted the message.
        """
        return await self.p2p.send_session(
            peer_id=self.peer_id,
            protocol_id=protocol_id,
            sess_msg=msg,
        )
