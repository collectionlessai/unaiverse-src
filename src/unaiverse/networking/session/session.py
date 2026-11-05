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
from __future__ import annotations

import json
import secrets
import asyncio
import time
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Callable, Awaitable, TYPE_CHECKING

from unaiverse.networking.session.sess_msg import (
    SessMsg, reason_name, ProfileDecisionResult, WorldGrantSpec, VerifiedTokenClaims,
    OK, VERSION_UNSUPPORTED, NOT_AUTHORIZED, LOCAL_ERROR, GRACEFUL_SHUTDOWN,
    SESSION_UNKNOWN, TOKEN_EXPIRED,
    SESSION_HELLO_PROTOCOL, SESSION_PROFILE_PROTOCOL,
    SESSION_WORLD_PROTOCOL, SESSION_CONTROL_PROTOCOL, SESSION_PROTOCOL_IDS,
)
from unaiverse.utils.logger import log

if TYPE_CHECKING:
    from unaiverse.networking.p2p.p2p import P2P

# Per-state timeouts (seconds)
_TIMEOUT_CONNECT_S = 300.0  # total CONNECTING budget: a backstop for a wedged dial task. The
                            # per-attempt bound is Go's PeerConnectionTimeout (~30s, x2 on the
                            # tcp fallback); tick-driven retries up to _MAX_DIAL_ATTEMPTS (which
                            # FAILs first) fit inside this window without starving other peers.
# Sized for slow peer pumps, not for the wire: a peer whose main loop is busy
# (heavy compute in an action, a loaded machine, Pyodide) can drain and answer a
# HELLO several seconds late, and the observed field delays reach ~10s. 15s gives
# margin over that and aligns stage 1 with stage 2, removing the initiator-5s /
# responder-15s asymmetry that produced spurious timeout:hello_sent. A late ACK
# is already accepted at any point in HELLO_SENT, so this is just accepting it.
_TIMEOUT_HELLO_S = 15.0     # Stage 1: HELLO → HELLO_ACK
_TIMEOUT_PROFILE_S = 15.0   # Stage 2: PROFILE_OFFER → PROFILE_DECISION
_TIMEOUT_WORLD_S = 30.0     # Stage 3: WORLD_GRANT expiry
_TIMEOUT_RECONNECT_S = 45.0 # Session-level reconnect ceiling before Session.Lost. Longer than the
                            # Go transport grace (DisconnectionGracePeriod) so several backoff'd
                            # re-dial + RESUME cycles fit inside the window (see tick_dial).
_TIMEOUT_CLOSING_S = 10.0   # GOAWAY sent; wait for GOAWAY_ACK
_TIMEOUT_CLOSING_FAILED_SEND_S = 1.0  # GOAWAY write failed (link dead): no ACK can
                                      # ever come, finalize fast so the shutdown
                                      # drain does not burn its budget waiting.
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
    KNOWN = "known"                     # initiator: peer known, not yet started (no dial)
    INIT = "init"                       # responder: spawned, awaiting inbound HELLO
    CONNECTING = "connecting"           # initiator: dialing
    HELLO_SENT = "hello_sent"           # initiator: HELLO out, awaiting HELLO_ACK
    HELLO_ACK_SENT = "hello_ack_sent"   # responder: HELLO_ACK out, awaiting PROFILE_OFFER
    PROFILE_SENT = "profile_sent"       # initiator: PROFILE_OFFER out, awaiting PROFILE_DECISION
    AWAITING_WORLD_GRANT = "awaiting_world_grant"  # initiator: world joiner expecting WORLD_GRANT
    AWAITING_HANDOFF = "awaiting_handoff"  # responder public: WORLD_GRANT out (grant carrier), awaiting the joiner's GOAWAY
    ESTABLISHED = "established"
    RECONNECTING = "reconnecting"       # link dropped, within grace period
    CLOSING = "closing"                 # sent GOAWAY, awaiting GOAWAY_ACK
    CLOSED = "closed"
    FAILED = "failed"


class SessionRole(str, Enum):
    INITIATOR = "initiator"
    RESPONDER = "responder"


class SessionEventType(str, Enum):
    ESTABLISHED = "established"
    FAILED = "failed"
    RECONNECTING = "reconnecting"   # transport dropped, within grace
    RESUMED = "resumed"
    LOST = "lost"
    CLOSED = "closed"


@dataclass
class SessionEvent:
    """Emitted by Session when the application-visible state changes."""
    type: SessionEventType
    session_id: str   # hex string
    peer_id: str
    role: SessionRole
    p2p_name: str
    data: Dict[str, Any] = field(default_factory=dict)


# Type alias for the responder decision hook (see SessionManager). The trailing
# argument carries the root-attested claims (RT) when a token-verifier vouched for
# the peer, or None on the v1 / no-verifier path.
DecisionHook = Callable[[str, Optional[str], dict, str, Optional[VerifiedTokenClaims]],
                        Awaitable[ProfileDecisionResult]]


# Helpers

def _sign_payload(nonce: bytes, session_id: bytes) -> bytes:
    """Canonical sign-payload for sig_a / sig_b: ``nonce || session_id``."""
    return bytes(nonce) + bytes(session_id)


# Session

class Session:
    """Per-peer session FSM. See module docstring for lifecycle."""

    def __init__(
        self,
        peer_id: str,
        role: SessionRole,
        p2p: "P2P",
        p2p_name: str,
        local_peer_id: str,
        local_node_id: str,
        local_profile_factory: Callable[[], dict],
        remote_node_id_hint: Optional[str] = None,
        addresses: Optional[List[str]] = None,
        is_world_join: bool = False,
        decision_hook: Optional[DecisionHook] = None,
        grant: Optional[dict] = None,
        reject_relay: bool = False,
        app_role: int = 0,
        token_verifier: Optional[Callable] = None,
        token_factory: Optional[Callable[[], str]] = None,
        dial_attempts: Optional[int] = None,
        dial_budget_s: Optional[float] = None,
    ):
        """
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
        self.remote_cv_hash: Optional[str] = None
        self.remote_code_hash: Optional[str] = None
        # A TOKEN_UPDATE that arrived before ESTABLISHED is buffered and applied on tick.
        self._buffered_token_update: Optional[SessMsg] = None
        self.is_world_join = is_world_join
        self._decision_hook = decision_hook
        self._grant = grant  # initiator world Session: redeem this after PROFILE
        self._reject_relay = reject_relay  # fail the dial if it resolves through a relay

        # Session identity. Initiator chooses; responder adopts from HELLO.
        self.session_id: bytes = secrets.token_bytes(_SESSION_ID_BYTES)
        self.session_id_hex: str = self.session_id.hex()
        self.nonce_a: bytes = b""
        self.nonce_b: bytes = b""

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
        self._timeout_s: Optional[float] = None

        # Dial retry counter (initiator) + the in-flight dial task. Design A: the dial runs
        # off the main loop as a task and is reaped by tick_dial, so a slow dial cannot starve
        # the other Sessions serviced from the same single main-loop task.
        self._dial_attempts = 0
        self._dial_task: Optional[asyncio.Task] = None
        self._next_dial_at: float = 0.0   # monotonic gate for the CONNECTING dial backoff
        self._max_dial_attempts = dial_attempts if dial_attempts is not None else _MAX_DIAL_ATTEMPTS
        self._dial_budget_s = dial_budget_s if dial_budget_s is not None else _TIMEOUT_CONNECT_S

        # Profile received from the remote peer (set on ESTABLISHED).
        self.remote_profile: Optional[dict] = None

        # Session-level keep-alive state (PING/PONG over the control protocol,
        # bound to session_id — not /ipfs/ping). One ping in flight at a time; its
        # PONG arrives via process_inbound and clears _ping_in_flight.
        self._last_ping_at: float = 0.0
        self._consecutive_ping_failures: int = 0
        self._ping_in_flight: bool = False
        self._ping_nonce: bytes = b""
        self._ping_deadline: float = 0.0
        # Reason string of the GOAWAY we sent (close()), echoed in the local
        # CLOSED event so both sides observe the same close reason.
        self._local_goaway_reason: str = ""
        # WORLD_GRANT that overtook the PROFILE_DECISION (no cross-protocol
        # ordering); consumed as soon as the decision lands.
        self._early_world_grant: Optional[SessMsg] = None

        # Active re-dial + RESUME state (initiator drives recovery while RECONNECTING).
        self._reconnect_substep: str = ""   # "" | "redial" | "resume_sent"
        self._reconnect_attempts: int = 0
        self._next_redial_at: float = 0.0   # monotonic gate, set/advanced by tick_dial
        self._resume_nonce: bytes = b""

        # World-join specifics.
        self.grant_id: bytes = b""
        self.grant_expiry_unix_ms: int = 0

    # repr

    def __repr__(self) -> str:
        return (f"Session(...{self.peer_id[-5:]} role={self.role.value} "
                f"state={self.state.value} p2p={self.p2p_name})")

    # Initiator entry point

    def start(self) -> List[SessionEvent]:
        """Initiator: enter CONNECTING and launch the dial as a background task. The dial
        runs off the main loop (Design A) so it does not starve the other Sessions serviced
        from the same task; its outcome (HELLO on success, retry/FAILED on failure) is reaped
        on the next tick_dial, so this returns immediately with no events."""
        if self.role != SessionRole.INITIATOR:
            raise RuntimeError("Session.start() is for initiators only")
        self._set_state(SessionState.CONNECTING, self._dial_budget_s)
        self._next_dial_at = 0.0
        self._start_dial()
        return []

    def _start_dial(self) -> None:
        """Launch the dial as a background task (single-flight: no-op if one is already in
        flight). Non-blocking; only _dial_task is touched here — the FSM mutates only when
        tick_dial reaps the outcome, on the main task, so the single-writer invariant holds."""
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
    async def _run_dial(p2p, addresses) -> tuple:
        """The dial coroutine, run as a background task: reads only its arguments (no Session
        state), so the only new interleaving it introduces is 'a result becomes available'.
        Returns (peer_id or None, through_relay); never raises."""
        try:
            return await p2p.connect(addresses)
        except Exception as e:
            log.error(f"dial raised: {e}")
            return None, False

    async def tick_dial(self, now: Optional[float] = None) -> List[SessionEvent]:
        """Reap a finished dial (→ HELLO / RESUME / retry / FAILED) and launch the next one
        when the backoff gate opens. Replaces _dial_then_hello, tick_reconnect and
        _redial_and_resume: the dial is a task, so no other Session is starved while it runs."""
        if now is None:
            now = time.monotonic()

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

        # (b) Launch the next dial once its gate opens.
        if self.role != SessionRole.INITIATOR or not self.addresses:
            return []
        if self.state == SessionState.CONNECTING and now >= self._next_dial_at:
            self._start_dial()
        elif self.state == SessionState.RECONNECTING:
            # A RESUME that wasn't acked within its window → fall back to another re-dial.
            if self._reconnect_substep == "resume_sent" and now >= self._next_redial_at:
                self._reconnect_substep = "redial"
            if self._reconnect_substep == "redial" and now >= self._next_redial_at:
                self._start_dial()
        return []

    async def _on_dial_done_connecting(self, peer_id, through_relay, now) -> List[SessionEvent]:
        if peer_id is not None and through_relay and self._reject_relay:
            log.error(f"{self!r} dial resolved through a relay circuit but relay nodes are "
                      f"disallowed (talk_to_relay_based_nodes=False)", sub=self._sub)
            return [self._fail("relay_rejected")]
        if peer_id is None:
            if self._dial_attempts < self._max_dial_attempts:
                delay = min(
                    random.uniform(0, _RETRY_BASE_S * (2 ** (self._dial_attempts - 1))),
                    _RETRY_CAP_S,
                )
                self._next_dial_at = now + delay
                log.session(f"{self!r} dial failed, retrying in {delay:.1f}s", sub=self._sub)
                return []
            log.error(f"{self!r} dial failed after {self._max_dial_attempts} attempts", sub=self._sub)
            return [self._fail("dial_failed")]
        # If the dial returned a different peer-id than we expected (e.g. relayed multiaddr
        # without the /p2p hop), update it before the HELLO.
        if peer_id != self.peer_id:
            log.session(f"{self!r} peer_id updated from ...{self.peer_id[-5:]} → ...{peer_id[-5:]}", sub=self._sub)
            self.peer_id = peer_id
        return await self._send_hello()

    async def _on_dial_done_reconnecting(self, peer_id, through_relay, now) -> List[SessionEvent]:
        if peer_id is not None and through_relay and self._reject_relay:
            peer_id = None   # relay-only reconnect disallowed; treat it as a failed dial
        if peer_id is None:
            self._reconnect_attempts += 1
            delay = min(_RETRY_BASE_S * (2 ** (self._reconnect_attempts - 1)), _RETRY_CAP_S)
            self._next_redial_at = now + delay
            self._reconnect_substep = "redial"
            log.session(f"{self!r} reconnect dial failed, next try in {delay:.1f}s", sub=self._sub)
            return []
        if peer_id != self.peer_id:
            self.peer_id = peer_id
        # Transport back — try to RESUME the existing session before re-handshaking.
        self._resume_nonce = secrets.token_bytes(8)
        self._reconnect_substep = "resume_sent"
        self._next_redial_at = now + _RESUME_ACK_TIMEOUT_S
        await self._send_payload(SESSION_CONTROL_PROTOCOL, SessMsg(SessMsg.KIND_RESUME, {
            "session_id": self.session_id, "nonce": self._resume_nonce,
        }))
        log.session(f"{self!r} reconnect dial ok, RESUME sent", sub=self._sub)
        return []

    async def _send_hello(self) -> List[SessionEvent]:
        self.nonce_a = secrets.token_bytes(_NONCE_BYTES)
        msg = SessMsg(SessMsg.KIND_HELLO_REQ, {
            "session_id": self.session_id,
            "nonce_a": self.nonce_a,
            "protocol_versions": ["1.0"],
            "capabilities": 0,
            "client_id": "python/1.0",
        })
        if not await self._send_payload(SESSION_HELLO_PROTOCOL, msg):
            return [self._fail("hello_send_failed")]
        self._set_state(SessionState.HELLO_SENT, _TIMEOUT_HELLO_S)
        log.session(f"{self!r} HELLO sent", sub=self._sub)
        return []

    # Inbound dispatch

    async def process_inbound(self, protocol_id: str, sess_msg: SessMsg) -> List[SessionEvent]:
        """Route an inbound session message through the FSM. Called by
        SessionManager.deliver for every SessionInbound."""
        events: List[SessionEvent] = []
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

    # Stage-1: HELLO

    async def _on_hello_protocol(self, msg: SessMsg) -> List[SessionEvent]:
        if (msg.kind == SessMsg.KIND_HELLO_REQ and
                self.role == SessionRole.RESPONDER and self.state == SessionState.INIT):
            return await self._responder_on_hello(msg)

        if (msg.kind == SessMsg.KIND_HELLO_RESP and
                self.role == SessionRole.INITIATOR and self.state == SessionState.HELLO_SENT):
            return await self._initiator_on_hello_ack(msg)

        log.session(f"{self!r} ignored HELLO-protocol {msg.kind} in state {self.state.value}", sub=self._sub)
        return []

    async def _responder_on_hello(self, req: SessMsg) -> List[SessionEvent]:
        session_id = req.get("session_id") or b""
        if not session_id:
            return [self._fail("hello_missing_session_id")]
        self.session_id = bytes(session_id)
        self.session_id_hex = self.session_id.hex()
        self.nonce_a = bytes(req.get("nonce_a") or b"")

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
            "chosen_version": chosen, "capabilities": 0, "nonce_b": self.nonce_b,
            "sig_b": sig_b, "reason_code": OK,
        })
        if not await self._send_payload(SESSION_HELLO_PROTOCOL, resp):
            return [self._fail("hello_ack_send_failed")]
        self._set_state(SessionState.HELLO_ACK_SENT, _TIMEOUT_PROFILE_S)
        log.session(f"{self!r} HELLO_ACK sent", sub=self._sub)
        return []

    async def _initiator_on_hello_ack(self, resp: SessMsg) -> List[SessionEvent]:
        reason_code = resp.get("reason_code", OK)
        if reason_code != OK:
            return [self._fail(f"hello_ack_rejected:{reason_name(reason_code)}")]
        if resp.get("chosen_version") != "1.0":
            return [self._fail(f"hello_ack_bad_version:{resp.get('chosen_version')}")]

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

    # Stage-2: PROFILE

    async def _initiator_send_profile_offer(self) -> List[SessionEvent]:
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

    async def _on_profile_protocol(self, msg: SessMsg) -> List[SessionEvent]:
        if (msg.kind == SessMsg.KIND_PROFILE_OFFER and
                self.role == SessionRole.RESPONDER and self.state == SessionState.HELLO_ACK_SENT):
            return await self._responder_on_profile_offer(msg)

        if (msg.kind == SessMsg.KIND_PROFILE_DECISION and
                self.role == SessionRole.INITIATOR and self.state == SessionState.PROFILE_SENT):
            return await self._initiator_on_profile_decision(msg)

        log.session(f"{self!r} ignored PROFILE-protocol {msg.kind} in state {self.state.value}", sub=self._sub)
        return []

    async def _verify_peer_token(self, token: str):
        """Verify an in-band peer token (RT) against the transport peer-id, shared by
        both handshake directions (responder at PROFILE_OFFER, initiator at
        PROFILE_DECISION). Returns ``(claims, None)`` on success, ``(None, reason)``
        on a verification or cross-check failure, and ``(None, None)`` when no verifier
        is configured (the v1 path). On success the root-attested node-id is adopted
        into ``remote_node_id_hint``; when we already expected a specific node-id at
        this peer, a mismatch is rejected (we reached the right peer-id but the wrong
        node). On success the peer's attested token expiry / cv_hash are captured on
        the Session (``remote_token_exp`` / ``remote_cv_hash``) for the in-session
        refresh, together with the attested world-code hash (``remote_code_hash``,
        None when the token carries no such claim)."""
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

    async def _responder_on_profile_offer(self, offer: SessMsg) -> List[SessionEvent]:
        # Verify sig_a under the initiator's peer-id.
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

    async def _reject_profile(self, reason_code: int, log_reason: str) -> List[SessionEvent]:
        await self._send_payload(SESSION_PROFILE_PROTOCOL, SessMsg(SessMsg.KIND_PROFILE_DECISION, {
            "accepted": False, "reason_code": reason_code, "session_id": self.session_id,
        }))
        return [self._fail(f"rejected_offer:{log_reason}")]

    async def _initiator_on_profile_decision(self, decision: SessMsg) -> List[SessionEvent]:
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

    # Stage-3: WORLD

    async def _responder_issue_world_grant(self, spec: WorldGrantSpec) -> List[SessionEvent]:
        """World host (on its PUBLIC Session): mint and send a WORLD_GRANT, then
        wait for the joiner to redeem it on the world Session. The grant is no longer
        signed (RT): the joiner trusts it because the public Session is mutually
        token-authenticated, and the world Session re-verifies the joiner's token
        while the host pre-admission (conn_routing) gates the binding."""
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

    async def _on_world_protocol(self, msg: SessMsg) -> List[SessionEvent]:
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

    async def _initiator_on_world_grant(self, grant: SessMsg) -> List[SessionEvent]:
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

    # Control: GOAWAY

    async def _on_control_protocol(self, msg: SessMsg) -> List[SessionEvent]:
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

    # Control: token refresh

    async def send_token_update(self, token: str) -> None:
        """Push our refreshed root token to this peer (ESTABLISHED only). The peer
        re-verifies it and advances its cached expiry for us, so the session does not
        have to re-handshake just because our JWT rolled over."""
        if self.state != SessionState.ESTABLISHED:
            return
        await self._send_payload(SESSION_CONTROL_PROTOCOL, SessMsg(SessMsg.KIND_TOKEN_UPDATE, {
            "session_id": self.session_id, "token": token or "",
        }))

    async def _on_token_update(self, msg: SessMsg) -> List[SessionEvent]:
        """Inbound refreshed peer token. Buffer it until ESTABLISHED, else apply now."""
        if self.token_verifier is None:
            return []
        if self.state != SessionState.ESTABLISHED:
            self._buffered_token_update = msg
            return []
        await self._apply_token_update(msg)
        return []

    async def _apply_token_update(self, msg: SessMsg) -> None:
        """Re-verify the pushed token against the same transport peer-id, require the
        attested node-id to be unchanged (a divergent one rejects the update, not the
        session), and advance the cached expiry MONOTONICALLY so a replayed older
        token cannot shrink the validity window."""
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

    # Control: session-level liveness (PING/PONG)

    async def _on_ping(self, msg: SessMsg) -> List[SessionEvent]:
        """Answer a peer's liveness PING. ok=True only if we hold this exact
        Session and it is usable: ESTABLISHED, mid-reconnect, or a grant carrier
        parked in AWAITING_HANDOFF — the joiner's carrier keepalives while its
        world dial runs, and an ok=False there would collapse the handoff at the
        first PING (~15s budget). A restarted peer with no Session never reaches
        here — SessionManager.deliver answers ok=False on its behalf."""
        ok = (bytes(msg.get("session_id") or b"") == self.session_id and
              self.state in (SessionState.ESTABLISHED, SessionState.RECONNECTING,
                             SessionState.AWAITING_HANDOFF))
        if not await self._send_payload(SESSION_CONTROL_PROTOCOL, SessMsg(SessMsg.KIND_PONG, {
            "session_id": self.session_id,
            "nonce": bytes(msg.get("nonce") or b""),
            "ok": ok,
        })):
            log.session(f"{self!r} PONG send failed", sub=self._sub)
        return []

    def _on_pong(self, msg: SessMsg) -> Optional[SessionEvent]:
        """Handle a PONG to our in-flight ping. Pure sync (no sends)."""
        if not self._ping_in_flight or bytes(msg.get("nonce") or b"") != self._ping_nonce:
            return None  # stale / unmatched
        self._ping_in_flight = False
        if msg.get("ok"):
            self._consecutive_ping_failures = 0
            return None
        # Peer holds no Session for our session_id (restarted / forgot us): treat as
        # a hard liveness failure and begin the reconnect + RESUME path immediately.
        self._consecutive_ping_failures = 0
        log.session(f"{self!r} PONG ok=false (peer lost our session) → reconnecting", sub=self._sub)
        return self.on_transport_disconnected()

    # Control: RESUME (fast-path recovery before re-handshake)

    async def _on_resume(self, msg: SessMsg) -> List[SessionEvent]:
        """Peer (initiator) is resuming after a link blip. ACK if we still hold the
        Session and it is usable (AWAITING_HANDOFF included: a public blip during
        the joiner's world dial must not kill the parked carrier, whose RESUME
        would otherwise be NACKed into a LOST); NACK (SESSION_UNKNOWN) otherwise
        so the peer re-handshakes."""
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

    async def _on_resume_ack(self, msg: SessMsg) -> List[SessionEvent]:
        """Our RESUME was answered. accepted → RESUMED; NACK → re-handshake."""
        if self.state != SessionState.RECONNECTING or self._reconnect_substep != "resume_sent":
            return []
        if bytes(msg.get("nonce") or b"") != self._resume_nonce:
            return []  # stale ack for a previous attempt
        if msg.get("accepted"):
            return [self._resume_established()]
        return await self._restart_handshake_or_fail(reason_name(msg.get("reason_code", SESSION_UNKNOWN)))

    def _resume_established(self) -> SessionEvent:
        """RECONNECTING → ESTABLISHED after a confirmed RESUME. Resets liveness."""
        self._set_state(SessionState.ESTABLISHED, None)
        self._last_ping_at = time.monotonic()
        self._consecutive_ping_failures = 0
        self._ping_in_flight = False
        self._reconnect_substep = ""
        self._reconnect_attempts = 0
        log.session(f"{self!r} session resumed → ESTABLISHED", sub=self._sub)
        # Carry the stored peer profile on the event so Node can re-enrol from RESUMED
        # without reaching into Session.remote_profile (a fast RESUME keeps it; only a
        # full re-handshake clears it). The profile is a plain dict, no protobuf.
        return self._emit(SessionEventType.RESUMED, {"profile": self.remote_profile})

    async def _restart_handshake_or_fail(self, reason: str) -> List[SessionEvent]:
        """RESUME NACK: the peer doesn't recognise our Session. A plain initiator
        Session re-runs the full handshake over the (already re-dialed) transport,
        preserving the application-visible peer-id/node-id. World sessions can't be
        transparently re-joined (double-join risk) → LOST so node re-drives the join."""
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
        self._consecutive_ping_failures = 0
        self._ping_in_flight = False
        return await self._send_hello()

    async def close(self, reason_code: int = GRACEFUL_SHUTDOWN) -> None:
        """Send GOAWAY to the remote peer and enter CLOSING. The FSM completes
        via process_inbound(GoawayAck) or check_timeout. Closable states are the
        two live ones plus AWAITING_HANDOFF: a world host shutting down must be
        able to GOAWAY a parked grant carrier too, instead of leaving it to die
        of timeout after the process is gone."""
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

    async def _on_remote_goaway(self, ga: SessMsg) -> List[SessionEvent]:
        # A GOAWAY from a PREVIOUS incarnation (cross-protocol reordering: the
        # peer's replace-stale sends GOAWAY(old) and HELLO(new) back to back on
        # different one-shot streams) must not kill this fresh Session. Courtesy-ACK
        # it so the closer's CLOSING completes, but stay alive. An empty session_id
        # is tolerated (older senders); PING/RESUME have the same id guard.
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

    # Timeout sweep

    def check_timeout(self, now: Optional[float] = None) -> Optional[SessionEvent]:
        """Pure synchronous — no sends. Returns a single SessionEvent on timeout
        firing, or None."""
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
            return self._lose("grace_period_expired")

        if self.state == SessionState.CLOSING:
            return self._finalize_close(
                f"goaway_ack_timeout:{getattr(self, '_local_goaway_reason', '')}")

        return None

    # Transport events (sync)

    def on_transport_disconnected(self) -> Optional[SessionEvent]:
        """Called when the link drops (transport diff) or when session-level
        liveness fails. ESTABLISHED → RECONNECTING, arming the active re-dial +
        RESUME loop (tick_dial). Already reconnecting: reset the grace timer."""
        if self.state == SessionState.ESTABLISHED:
            self._set_state(SessionState.RECONNECTING, _TIMEOUT_RECONNECT_S)
            self._reconnect_substep = "redial"
            self._reconnect_attempts = 0
            self._next_redial_at = 0.0   # re-dial on the next tick_dial
            self._ping_in_flight = False
            log.session(f"{self!r} link lost, starting {_TIMEOUT_RECONNECT_S}s reconnect window", sub=self._sub)
            return self._emit(SessionEventType.RECONNECTING, {"cause": "transport_disconnected"})
        if self.state == SessionState.RECONNECTING:
            self._state_entered_at = time.monotonic()
        return None

    def on_transport_reconnected(self) -> Optional[SessionEvent]:
        """Transport-diff reconnect signal. Do NOT blindly resume — a reconnect can
        be to a *restarted* peer that holds no Session (the /ipfs/ping false-resume
        bug). Just kick the RESUME handshake; only a valid RESUME_ACK promotes us
        back to ESTABLISHED (see _on_resume_ack / _resume_established)."""
        if self.state == SessionState.RECONNECTING:
            self._reconnect_substep = "redial"
            self._next_redial_at = 0.0   # send RESUME on the next tick_dial
            log.session(f"{self!r} transport reconnect signal → attempting RESUME", sub=self._sub)
        return None

    # Session-level keep-alive (PING/PONG over the control protocol)

    async def tick_keepalive(self, now: Optional[float] = None) -> Optional[SessionEvent]:
        """Time out an unanswered ping, then send a new one if due. The PONG
        itself is handled inbound (_on_pong). Returns at most one SessionEvent
        (RECONNECTING on the miss threshold). The PING is sent inline: a
        fire-and-forget task would only start when the loop yields, which the
        mostly-synchronous main loop cannot guarantee."""
        if now is None:
            now = time.monotonic()

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

        # An in-flight ping that has not been PONGed within the RTT window is a miss.
        if self._ping_in_flight and now >= self._ping_deadline:
            self._ping_in_flight = False
            self._consecutive_ping_failures += 1
            log.session(f"{self!r} ping miss ({self._consecutive_ping_failures}/{_KEEPALIVE_FAIL_THRESHOLD})", sub=self._sub)
            if (self._consecutive_ping_failures >= _KEEPALIVE_FAIL_THRESHOLD and
                    self.state == SessionState.ESTABLISHED):
                self._consecutive_ping_failures = 0
                return self.on_transport_disconnected()

        # Schedule a new ping when due (ESTABLISHED only, one in flight at a time).
        if (self.state == SessionState.ESTABLISHED and
                not self._ping_in_flight and
                now - self._last_ping_at >= _KEEPALIVE_PERIOD_S):
            self._last_ping_at = now
            self._ping_in_flight = True
            self._ping_nonce = secrets.token_bytes(8)
            self._ping_deadline = now + _PING_RTT_TIMEOUT_S
            await self._send_ping()
        return None

    async def _send_ping(self) -> None:
        """Keep-alive PING; a send failure would otherwise be invisible until
        it surfaces as a ping miss, so log it here."""
        if not await self._send_payload(SESSION_CONTROL_PROTOCOL, SessMsg(SessMsg.KIND_PING, {
            "session_id": self.session_id, "nonce": self._ping_nonce,
        })):
            log.session(f"{self!r} keepalive PING send failed", sub=self._sub)

    # Internal FSM helpers

    def _set_state(self, state: SessionState, timeout_s: Optional[float]) -> None:
        prev = self.state
        self.state = state
        self._state_entered_at = time.monotonic()
        self._timeout_s = timeout_s
        log.session(f"{self!r} {prev.value} → {state.value} (timeout={timeout_s}s)", sub=self._sub)

    def _fail(self, reason: str) -> SessionEvent:
        self._set_state(SessionState.FAILED, None)
        log.error(f"{self!r} FAILED: {reason}", sub=self._sub)
        return self._emit(SessionEventType.FAILED, {"reason": reason})
    
    def _lose(self, reason: str) -> SessionEvent:
        self._set_state(SessionState.FAILED, None)
        log.error(f"{self!r} LOST: {reason}", sub=self._sub)
        return self._emit(SessionEventType.LOST, {"reason": reason})

    def _finalize_close(self, reason: str) -> SessionEvent:
        self._set_state(SessionState.CLOSED, None)
        log.session(f"{self!r} CLOSED: {reason}", sub=self._sub)
        return self._emit(SessionEventType.CLOSED, {"reason": reason})

    def _emit(self, event_type: SessionEventType, data: dict) -> SessionEvent:
        return SessionEvent(
            type=event_type,
            session_id=self.session_id_hex,
            peer_id=self.peer_id,
            role=self.role,
            p2p_name=self.p2p_name,
            data=data,
        )

    async def _send_payload(self, protocol_id: str, msg: SessMsg) -> bool:
        return await self.p2p.send_session(
            peer_id=self.peer_id,
            protocol_id=protocol_id,
            sess_msg=msg,
        )

    # Silent abort (used by SessionManager tie-breaker)

    def abort_silently(self, reason: str) -> None:
        """Drop this Session without emitting an event. Used by the
        SessionManager simultaneous-open tie-breaker on the losing initiator."""
        self._set_state(SessionState.FAILED, None)
        log.session(f"{self!r} aborted silently: {reason}", sub=self._sub)
