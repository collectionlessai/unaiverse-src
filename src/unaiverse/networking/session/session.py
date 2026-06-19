"""
Session — per-peer connection lifecycle FSM.

Each Session owns the entire lifecycle of a connection to one remote peer:
  dial (via p2p.connect) → 4-stage handshake → established → session-level
  PING/PONG keep-alive → active re-dial + RESUME on loss → intentional GOAWAY
  close. Liveness/recovery are session-aware (bound to session_id), so a peer
  that restarts and forgets the session is detected and re-handshaked rather
  than masked by a transport that transparently re-dials.

Design (v1 — polling mode)
--------------------------
The FSM is *synchronous* with async send helpers: ``process_inbound()`` and
``check_timeout()`` are called from the polling loop (Node's
``__handle_network_messages`` / main loop) and advance state immediately.
``process_inbound()`` is async only because some transitions need to
``await p2p.send_session()`` / ``await p2p.<crypto>()``.

No asyncio.Queue, no background Task at all — the polling loop drives
everything, and all I/O (sends, keep-alive PING, re-dial + RESUME) happens
inline at the polling points (``deliver`` / ``tick``). Fire-and-forget tasks
are deliberately avoided: they only start when the event loop regains control,
which a mostly-synchronous main loop cannot guarantee.

Portability (CPython + Pyodide)
-------------------------------
This module is **protobuf-free**: it speaks ``SessMsg`` value objects and reaches
the transport/crypto only through the high-level async ``P2P`` surface injected as
``self.p2p`` (``connect`` / ``send_session`` / ``get_local_peer_id`` /
``sign_with_host_key`` / ``verify_peer_signature`` / ``verify_signed_peer_record`` /
``generate_signed_peer_record``). Cross-instance operations (the SignedPeerRecord
and grant signature for the agent's *other* peer-id) go through ``self.xcrypto``,
the capability injected by the layer that owns both P2P instances.
That is exactly the surface a JS ``OP2P`` mirror exposes, so this file runs
unchanged in Pyodide. The wire (de)serialization lives inside ``P2P.send_session``
(via ``SessMsg.to_bytes``) in CPython and in ``session_pb.js`` in the browser.

Each stage runs on its own libp2p protocol:
  /unaiverse/session/hello/1.0.0     -- HelloRequest / HelloResponse
  /unaiverse/session/profile/1.0.0   -- ProfileOffer / ProfileDecision
  /unaiverse/session/world/1.0.0     -- WorldGrant / WorldGrantAck,
                                        PrivateJoin / PrivateJoinAck
  /unaiverse/session/control/1.0.0   -- Goaway / GoawayAck

Identity is bound by ``sig_b`` (Stage-1, responder over ``nonce_a||session_id``),
``sig_a`` (Stage-2, initiator over ``nonce_b||session_id``), and the libp2p
SignedPeerRecord (Stage-2, binding the agent's *other* peer-id). No per-message
JWT piggyback for session protocols.
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
    SessMsg, reason_name, ProfileDecisionResult, WorldGrantSpec,
    OK, VERSION_UNSUPPORTED, NOT_AUTHORIZED, LOCAL_ERROR, GRACEFUL_SHUTDOWN,
    SIMULTANEOUS_OPEN_LOSS, SESSION_UNKNOWN, GRANT_EXPIRED, P2P_PUBLIC,
)
from unaiverse.utils.logger import log

if TYPE_CHECKING:
    from unaiverse.networking.p2p.p2p import P2P, CrossInstanceCrypto

# ── Protocol IDs (keep in sync with unailib/types.go + connpool SESSION_PROTOCOL_IDS) ─
SESSION_HELLO_PROTOCOL   = "/unaiverse/session/hello/1.0.0"
SESSION_PROFILE_PROTOCOL = "/unaiverse/session/profile/1.0.0"
SESSION_WORLD_PROTOCOL   = "/unaiverse/session/world/1.0.0"
SESSION_CONTROL_PROTOCOL = "/unaiverse/session/control/1.0.0"

SESSION_PROTOCOL_IDS = frozenset({
    SESSION_HELLO_PROTOCOL,
    SESSION_PROFILE_PROTOCOL,
    SESSION_WORLD_PROTOCOL,
    SESSION_CONTROL_PROTOCOL,
})

# ── Per-state timeouts (seconds) ──────────────────────────────────────────────
_TIMEOUT_CONNECT_S = 35.0   # dial: PeerConnectionTimeout (30s Go) + 5s buffer
_TIMEOUT_HELLO_S = 5.0      # Stage 1: HELLO → HELLO_ACK
_TIMEOUT_PROFILE_S = 15.0   # Stage 2: PROFILE_OFFER → PROFILE_DECISION
_TIMEOUT_WORLD_S = 30.0     # Stage 3: WORLD_GRANT expiry
_TIMEOUT_RECONNECT_S = 45.0 # Session-level reconnect ceiling before Session.Lost. Longer than the
                            # Go transport grace (DisconnectionGracePeriod) so several backoff'd
                            # re-dial + RESUME cycles fit inside the window (see tick_reconnect).
_TIMEOUT_CLOSING_S = 10.0   # GOAWAY sent; wait for GOAWAY_ACK

# ── Retry / backoff ──────────────────────────────────────────────────────────
_MAX_DIAL_ATTEMPTS = 4
_RETRY_BASE_S = 1.0
_RETRY_CAP_S = 30.0
_RESUME_ACK_TIMEOUT_S = 3.0  # RESUME sent; if un-acked within this, re-dial + re-RESUME

# ── Session-level keep-alive (PING/PONG over the control protocol) ────────────
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

# ── Misc ─────────────────────────────────────────────────────────────────────
_SESSION_ID_BYTES = 16
_NONCE_BYTES = 32
_GRANT_ID_BYTES = 32


# ── State / Role / Event enums ────────────────────────────────────────────────

class SessionState(str, Enum):
    INIT = "init"
    CONNECTING = "connecting"           # initiator: dialing
    HELLO_SENT = "hello_sent"           # initiator: HELLO out, awaiting HELLO_ACK
    HELLO_ACK_SENT = "hello_ack_sent"   # responder: HELLO_ACK out, awaiting PROFILE_OFFER
    PROFILE_SENT = "profile_sent"       # initiator: PROFILE_OFFER out, awaiting PROFILE_DECISION
    AWAITING_WORLD_GRANT = "awaiting_world_grant"  # initiator: world joiner expecting WORLD_GRANT
    AWAITING_PRIVATE_JOIN = "awaiting_private_join"  # responder world: WORLD_GRANT out, awaiting PRIVATE_JOIN
    PRIVATE_JOIN_SENT = "private_join_sent"  # initiator on p2p_world: PRIVATE_JOIN out, awaiting ACK
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


# Type alias for the responder decision hook (see SessionManager).
DecisionHook = Callable[[str, Optional[str], dict, str], Awaitable[ProfileDecisionResult]]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sign_payload(nonce: bytes, session_id: bytes) -> bytes:
    """Canonical sign-payload for sig_a / sig_b: ``nonce || session_id``."""
    return bytes(nonce) + bytes(session_id)


def _grant_sign_payload(grant_id: bytes, joiner_world_peer_id: str, role: int,
                        expiry_unix_ms: int) -> bytes:
    """Canonical sign-payload for grant_sig:
    ``grant_id || joiner_world_peer_id || role_be || expiry_be``.

    Binding the joiner's WORLD-side peer-id (instead of its application node-id)
    makes the grant single-redeemer at the transport level: the world host
    verifies its own signature at PRIVATE_JOIN against the peer that actually
    dialed, so a grant lifted from another node cannot be redeemed. Binding the
    role lets the host trust the role echoed back in PRIVATE_JOIN with no
    server-side stash."""
    return (bytes(grant_id) + joiner_world_peer_id.encode("utf-8") +
            int(role).to_bytes(8, "big") + int(expiry_unix_ms).to_bytes(8, "big"))


# ── Session ───────────────────────────────────────────────────────────────────

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
        remote_other_peer_id_hint: Optional[str] = None,
        addresses: Optional[List[str]] = None,
        is_world_join: bool = False,
        decision_hook: Optional[DecisionHook] = None,
        grant: Optional[dict] = None,
        reject_relay: bool = False,
        xcrypto: Optional["CrossInstanceCrypto"] = None,
    ):
        """
        Args:
            peer_id: libp2p peer-id of the remote (on this p2p instance).
            role: INITIATOR or RESPONDER.
            p2p: the P2P instance this Session runs on (or a JS OP2P mirror in
                Pyodide). The Session reaches the transport and crypto through it,
                and the agent's *other* network through the injected ``xcrypto``.
            p2p_name: the label of the P2P instance this Session lives on (used for
                events).
            local_peer_id: our libp2p peer-id on `p2p_name`.
            local_node_id: our application-level node-id (the JWT-bound identity).
            local_profile_factory: zero-arg callable returning the current NodeProfile
                dict (must be plain JSON-serializable — sampled fresh at offer time).
            remote_node_id_hint: optional; the remote's application node-id when known.
            remote_other_peer_id_hint: optional; the remote's *other* peer-id, needed
                to verify the inbound SignedPeerRecord / a world PRIVATE_JOIN sig.
            addresses: initiator only — peer multiaddrs from root server / cache.
            is_world_join: initiator only — if True on a public Session, expect a
                WORLD_GRANT (Stage-3) after PROFILE_DECISION.
            decision_hook: responder only — async callback to accept/reject an offer
                and (for a world host) produce the WorldGrant spec.
            grant: initiator only, world Session — the grant blob obtained on the
                public Session; when present this Session redeems it via PRIVATE_JOIN
                after the normal HELLO+PROFILE handshake on p2p_world.
            xcrypto: crypto capability for the agent's *other* P2P instance (the
                cross-instance SignedPeerRecord / world-grant signer, sealed under
                the other host key). Injected from the layer that owns both
                instances (the Coordinator / Node).
        """
        self.peer_id = peer_id
        self.role = role
        self.p2p = p2p
        self.p2p_name = p2p_name
        # Cross-instance crypto capability (other-network host key), injected by the
        # layer that owns both P2P instances. The cross-instance call-sites use it
        # for the SignedPeerRecord and the world-grant signature.
        self.xcrypto = xcrypto
        self._sub = "pub" if p2p_name == P2P_PUBLIC else "prv"
        self.local_peer_id = local_peer_id
        self.local_node_id = local_node_id
        self._local_profile_factory = local_profile_factory
        self.remote_node_id_hint = remote_node_id_hint
        self.remote_other_peer_id_hint = remote_other_peer_id_hint
        self.addresses = list(addresses or [])
        self.is_world_join = is_world_join
        self._decision_hook = decision_hook
        self._grant = grant  # initiator world Session: redeem this after PROFILE
        self._reject_relay = reject_relay  # fail the dial if it resolves through a relay

        # Session identity. Initiator chooses; responder adopts from HELLO.
        self.session_id: bytes = secrets.token_bytes(_SESSION_ID_BYTES)
        self.session_id_hex: str = self.session_id.hex()
        self.nonce_a: bytes = b""
        self.nonce_b: bytes = b""

        # FSM state.
        self.state: SessionState = SessionState.INIT
        self._state_entered_at: float = time.monotonic()
        self._timeout_s: Optional[float] = None

        # Dial retry counter (initiator).
        self._dial_attempts = 0

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
        self._next_redial_at: float = 0.0   # monotonic gate, set/advanced by tick_reconnect
        self._resume_nonce: bytes = b""

        # World-join specifics.
        self.grant_id: bytes = b""
        self.grant_sig: bytes = b""
        self.grant_expiry_unix_ms: int = 0

    # ── repr ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (f"Session(...{self.peer_id[-5:]} role={self.role.value} "
                f"state={self.state.value} p2p={self.p2p_name})")

    # ── Initiator entry point ─────────────────────────────────────────────────

    async def start(self) -> List[SessionEvent]:
        """Initiator: dial the peer and send HELLO. Returns events accumulated
        during start (typically empty until ESTABLISHED arrives). On terminal
        dial failure returns ``[SessionEvent(FAILED, ...)]``."""
        if self.role != SessionRole.INITIATOR:
            raise RuntimeError("Session.start() is for initiators only")
        return await self._dial_then_hello()

    async def _dial_then_hello(self) -> List[SessionEvent]:
        self._dial_attempts += 1
        self._set_state(SessionState.CONNECTING, _TIMEOUT_CONNECT_S)
        log.session(f"{self!r} dial attempt {self._dial_attempts}/{_MAX_DIAL_ATTEMPTS}", sub=self._sub)

        try:
            peer_id, through_relay = await self.p2p.connect(self.addresses)
        except Exception as e:
            log.error(f"{self!r} p2p.connect raised: {e}", sub=self._sub)
            peer_id, through_relay = None, False

        if peer_id is not None and through_relay and self._reject_relay:
            log.error(f"{self!r} dial resolved through a relay circuit but relay nodes are "
                      f"disallowed (talk_to_relay_based_nodes=False)", sub=self._sub)
            return [self._fail("relay_rejected")]

        if peer_id is None:
            if self._dial_attempts < _MAX_DIAL_ATTEMPTS:
                delay = min(
                    random.uniform(0, _RETRY_BASE_S * (2 ** (self._dial_attempts - 1))),
                    _RETRY_CAP_S,
                )
                log.session(f"{self!r} dial failed, retrying in {delay:.1f}s", sub=self._sub)
                await asyncio.sleep(delay)
                return await self._dial_then_hello()
            log.error(f"{self!r} dial failed after {_MAX_DIAL_ATTEMPTS} attempts", sub=self._sub)
            return [self._fail("dial_failed")]

        # If the dial returned a different peer-id than we expected (e.g., relayed
        # multiaddr without /p2p hop), update it.
        if peer_id != self.peer_id:
            log.session(f"{self!r} peer_id updated from ...{self.peer_id[-5:]} → ...{peer_id[-5:]}", sub=self._sub)
            self.peer_id = peer_id

        return await self._send_hello()

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

    # ── Inbound dispatch ──────────────────────────────────────────────────────

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

    # ── Stage-1: HELLO ────────────────────────────────────────────────────────

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

        # Verify sig_b under the responder's peer-id (libp2p host key).
        if not await self.p2p.verify_peer_signature(
            self.peer_id,
            _sign_payload(bytes(self.nonce_a), self.session_id), bytes(resp.get("sig_b") or b""),
        ):
            return [self._fail("hello_ack_bad_sig_b")]

        self.nonce_b = bytes(resp.get("nonce_b") or b"")
        return await self._initiator_send_profile_offer()

    # ── Stage-2: PROFILE ──────────────────────────────────────────────────────

    async def _initiator_send_profile_offer(self) -> List[SessionEvent]:
        sig_a = await self.p2p.sign_with_host_key(_sign_payload(self.nonce_b, self.session_id))
        if sig_a is None:
            return [self._fail("profile_offer_sign_failed")]
        # SignedPeerRecord for our *other* peer-id, fetched lazily/async at offer
        # time (sealed over current addresses; must not be precomputed sync). Sealed
        # sealed under the *other* network's host key — hence xcrypto, not p2p.
        spr_other = await self.xcrypto.generate_signed_peer_record() or b""
        profile_bytes = json.dumps(self._local_profile_factory()).encode("utf-8")
        offer = SessMsg(SessMsg.KIND_PROFILE_OFFER, {
            "profile": profile_bytes,
            "sig_a": sig_a,
            "signed_peer_record": spr_other,
            "session_id": self.session_id,
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

    async def _responder_on_profile_offer(self, offer: SessMsg) -> List[SessionEvent]:
        # Verify sig_a under the initiator's peer-id.
        if not await self.p2p.verify_peer_signature(
            self.peer_id,
            _sign_payload(bytes(self.nonce_b), self.session_id), bytes(offer.get("sig_a") or b""),
        ):
            return await self._reject_profile(NOT_AUTHORIZED, "bad_sig_a")

        # If we know the initiator's other peer-id, verify the signed peer record
        # binds those bytes to that id. If we don't have a hint, skip — the other
        # side may not be a dual-key agent.
        spr = offer.get("signed_peer_record") or b""
        if self.remote_other_peer_id_hint and spr:
            if not await self.p2p.verify_signed_peer_record(
                bytes(spr), self.remote_other_peer_id_hint,
            ):
                return await self._reject_profile(NOT_AUTHORIZED, "bad_signed_peer_record")

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
                    self.peer_id, self.remote_node_id_hint, self.remote_profile, self.p2p_name)
            except Exception as e:
                log.error(f"{self!r} decision_hook raised: {e}", sub=self._sub)
                return await self._reject_profile(LOCAL_ERROR, f"decision_hook_error:{e}")
        else:
            decision = ProfileDecisionResult(accepted=True)

        if not decision.accepted:
            return await self._reject_profile(decision.reason_code, "decision_rejected")

        # Send the (positive) decision carrying our profile.
        dec_msg = SessMsg(SessMsg.KIND_PROFILE_DECISION, {
            "accepted": True,
            "profile_b": json.dumps(self._local_profile_factory()).encode("utf-8"),
            "reason_code": OK,
            "session_id": self.session_id,
        })
        if not await self._send_payload(SESSION_PROFILE_PROTOCOL, dec_msg):
            return [self._fail("profile_decision_send_failed")]

        # World host on its PUBLIC instance: issue a WORLD_GRANT. The joiner will
        # redeem it on a *separate* world Session; this public Session stays as the
        # grant carrier (no ESTABLISHED) until the joiner GOAWAYs it post-handoff.
        if decision.world_grant is not None:
            return await self._responder_issue_world_grant(decision.world_grant)

        # World host on its WORLD instance: the peer is redeeming a grant. Do not
        # ESTABLISH yet — wait for the PRIVATE_JOIN that follows on this Session.
        if decision.await_private_join:
            self._set_state(SessionState.AWAITING_PRIVATE_JOIN, _TIMEOUT_WORLD_S)
            log.session(f"{self!r} PROFILE accepted, awaiting PRIVATE_JOIN (world responder)", sub=self._sub)
            return []

        # Agent host: established immediately.
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

        # World joiner on the WORLD Session (carries a grant): redeem it via
        # PRIVATE_JOIN instead of going ESTABLISHED now.
        if self._grant is not None:
            return await self._initiator_send_private_join()

        self._set_state(SessionState.ESTABLISHED, None)
        log.session(f"{self!r} ESTABLISHED (initiator)", sub=self._sub)
        self._last_ping_at = time.monotonic()
        return [self._emit(SessionEventType.ESTABLISHED, {"profile_b": self.remote_profile})]

    # ── Stage-3: WORLD ────────────────────────────────────────────────────────

    async def _responder_issue_world_grant(self, spec: WorldGrantSpec) -> List[SessionEvent]:
        """World host (on its PUBLIC Session): mint and send a WORLD_GRANT, then
        wait for the joiner to redeem it on the world Session. The grant is signed
        with the WORLD-side key so the joiner can bind it to the private identity
        it will actually connect to."""
        self.grant_id = secrets.token_bytes(_GRANT_ID_BYTES)
        self.grant_expiry_unix_ms = int(time.time() * 1000) + int(spec.grant_ttl_ms)
        # The grant is bound to the world-side identity the joiner will redeem it
        # from, taken from its offered profile.
        joiner_world_peer_id = (self.remote_profile.get("dynamic") or {}).get("private_peer_id") or ""
        if not joiner_world_peer_id:
            return [self._fail("world_grant_missing_joiner_world_peer_id")]
        # Signed with the WORLD-side (other-network) host key so the joiner can bind
        # it to the private identity it will dial — hence xcrypto.
        grant_sig = await self.xcrypto.sign_with_host_key(
            _grant_sign_payload(self.grant_id, joiner_world_peer_id, spec.role,
                                self.grant_expiry_unix_ms),
        )
        if grant_sig is None:
            return [self._fail("world_grant_sign_failed")]
        world_spr = await self.xcrypto.generate_signed_peer_record() or b""

        grant = SessMsg(SessMsg.KIND_WORLD_GRANT, {
            "grant_id": self.grant_id,
            "grant_sig": grant_sig,
            "expiry_unix_ms": self.grant_expiry_unix_ms,
            "world_profile": spec.world_profile_bytes,
            "world_private_peer_record": world_spr,
            "role": spec.role,
            "code_bundle": spec.code_bundle,
            "initial_stats": spec.initial_stats,
            "rendezvous_tag": spec.rendezvous_tag,
        })
        if not await self._send_payload(SESSION_WORLD_PROTOCOL, grant):
            return [self._fail("world_grant_send_failed")]
        self._set_state(SessionState.AWAITING_PRIVATE_JOIN, _TIMEOUT_WORLD_S)
        log.session(f"{self!r} WORLD_GRANT sent, awaiting PRIVATE_JOIN", sub=self._sub)
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

        if (msg.kind == SessMsg.KIND_PRIVATE_JOIN and
                self.role == SessionRole.RESPONDER and self.state == SessionState.AWAITING_PRIVATE_JOIN):
            return await self._responder_on_private_join(msg)

        if (msg.kind == SessMsg.KIND_PRIVATE_JOIN_ACK and
                self.role == SessionRole.INITIATOR and self.state == SessionState.PRIVATE_JOIN_SENT):
            return await self._initiator_on_private_join_ack(msg)

        log.session(f"{self!r} ignored WORLD-protocol {msg.kind} in state {self.state.value}", sub=self._sub)
        return []

    async def _initiator_on_world_grant(self, grant: SessMsg) -> List[SessionEvent]:
        grant_id = grant.get("grant_id") or b""
        if not grant_id:
            return [self._fail("world_grant_missing_id")]
        expiry = int(grant.get("expiry_unix_ms", 0))
        if expiry < int(time.time() * 1000):
            return [self._fail("world_grant_expired")]

        world_spr = bytes(grant.get("world_private_peer_record") or b"")
        # Verify grant_sig against the world's WORLD-side identity: the spawn-time
        # hint when given, otherwise the private peer-id advertised in the grant's
        # world profile — the exact identity the handoff will dial, so a forged
        # role/expiry would need that identity's key. Verification is mandatory:
        # an unverifiable grant fails the handshake instead of riding through.
        expected_world_peer_id = self.remote_other_peer_id_hint
        if not expected_world_peer_id:
            try:
                wp = json.loads(bytes(grant.get("world_profile") or b"").decode("utf-8"))
                expected_world_peer_id = (wp.get("dynamic") or {}).get("private_peer_id") or ""
            except Exception:
                expected_world_peer_id = ""
        if not expected_world_peer_id:
            return [self._fail("world_grant_unverifiable_identity")]
        if world_spr and not await self.p2p.verify_signed_peer_record(
            world_spr, expected_world_peer_id,
        ):
            return [self._fail("world_grant_bad_spr")]
        grant_sig = bytes(grant.get("grant_sig") or b"")
        # The grant binds our WORLD-side peer-id (the one we will dial from), so the
        # local-peer-id used in the sign-payload comes from xcrypto.
        if not await self.p2p.verify_peer_signature(
            expected_world_peer_id,
            _grant_sign_payload(grant_id, await self.xcrypto.get_local_peer_id(),
                                int(grant.get("role", 0)), expiry),
            grant_sig,
        ):
            return [self._fail("world_grant_bad_sig")]

        self.grant_id = bytes(grant_id)
        self.grant_sig = bytes(grant.get("grant_sig") or b"")
        self.grant_expiry_unix_ms = expiry

        await self._send_payload(SESSION_WORLD_PROTOCOL, SessMsg(SessMsg.KIND_WORLD_GRANT_ACK,
                                                                 {"received": True}))

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
                "grant_sig": self.grant_sig.hex(),
                "expiry_unix_ms": self.grant_expiry_unix_ms,
                "world_profile_bytes": bytes(grant.get("world_profile") or b""),
                "role": int(grant.get("role", 0)),
                "rendezvous_tag": int(grant.get("rendezvous_tag", 0)),
                "world_private_peer_record": world_spr,
                "code_bundle": bytes(grant.get("code_bundle") or b""),
                "initial_stats": bytes(grant.get("initial_stats") or b""),
            },
        })]

    async def _initiator_send_private_join(self) -> List[SessionEvent]:
        """World Session initiator: after PROFILE_DECISION, redeem the grant. The
        signed grant fields (role/expiry/grant_sig) are echoed back so the host
        can verify its own signature statelessly."""
        grant = self._grant or {}
        grant_id = grant.get("grant_id_bytes") or b""
        if not grant_id:
            return [self._fail("private_join_missing_grant")]
        sig_init = await self.p2p.sign_with_host_key(bytes(grant_id))
        if sig_init is None:
            return [self._fail("private_join_sign_failed")]
        try:
            grant_sig = bytes.fromhex(grant.get("grant_sig") or "")
        except (ValueError, TypeError):
            grant_sig = b""
        join = SessMsg(SessMsg.KIND_PRIVATE_JOIN, {
            "grant_id": bytes(grant_id),
            "sig_init": sig_init,
            "joiner_node_id": self.local_node_id,
            "role": int(grant.get("role", 0)),
            "expiry_unix_ms": int(grant.get("expiry_unix_ms", 0)),
            "grant_sig": grant_sig,
        })
        if not await self._send_payload(SESSION_WORLD_PROTOCOL, join):
            return [self._fail("private_join_send_failed")]
        self._set_state(SessionState.PRIVATE_JOIN_SENT, _TIMEOUT_WORLD_S)
        log.session(f"{self!r} PRIVATE_JOIN sent", sub=self._sub)
        return []

    async def _initiator_on_private_join_ack(self, ack: SessMsg) -> List[SessionEvent]:
        if not ack.get("accepted"):
            return [self._fail(f"private_join_rejected:{reason_name(ack.get('reason_code', 0))}")]
        self._set_state(SessionState.ESTABLISHED, None)
        self._last_ping_at = time.monotonic()
        log.session(f"{self!r} ESTABLISHED (world initiator)", sub=self._sub)
        return [self._emit(SessionEventType.ESTABLISHED, {
            "profile_b": self.remote_profile,
            "rendezvous_topic": ack.get("rendezvous_topic", ""),
        })]

    async def _responder_on_private_join(self, join: SessMsg) -> List[SessionEvent]:
        grant_id = bytes(join.get("grant_id") or b"")
        role = int(join.get("role") or 0)
        expiry = int(join.get("expiry_unix_ms") or 0)

        async def _nack(reason_code: int, fail_reason: str) -> List[SessionEvent]:
            await self._send_payload(SESSION_WORLD_PROTOCOL, SessMsg(SessMsg.KIND_PRIVATE_JOIN_ACK, {
                "accepted": False, "reason_code": reason_code, "rendezvous_topic": "",
            }))
            return [self._fail(fail_reason)]

        if expiry < int(time.time() * 1000):
            return await _nack(GRANT_EXPIRED, "private_join_grant_expired")

        # Verify OUR OWN grant signature over (grant_id, the peer that actually
        # dialed, role, expiry): proves we minted this grant for this exact
        # world-side identity with this role — no server-side stash needed, and a
        # grant lifted from another joiner cannot be redeemed.
        if not await self.p2p.verify_peer_signature(
            self.local_peer_id,
            _grant_sign_payload(grant_id, self.peer_id, role, expiry),
            bytes(join.get("grant_sig") or b""),
        ):
            return await _nack(NOT_AUTHORIZED, "private_join_bad_grant_sig")

        # Verify sig_init over grant_id under the joiner's world-side peer-id
        # (the transport identity of this very Session): proof of key possession.
        if not await self.p2p.verify_peer_signature(
            self.peer_id, grant_id, bytes(join.get("sig_init") or b""),
        ):
            return await _nack(NOT_AUTHORIZED, "private_join_bad_sig")

        rendezvous_topic = f"world::{self.peer_id}::ps:rv"
        ack = SessMsg(SessMsg.KIND_PRIVATE_JOIN_ACK, {
            "accepted": True, "reason_code": OK, "rendezvous_topic": rendezvous_topic,
        })
        if not await self._send_payload(SESSION_WORLD_PROTOCOL, ack):
            return [self._fail("private_join_ack_send_failed")]

        self._set_state(SessionState.ESTABLISHED, None)
        self._last_ping_at = time.monotonic()
        log.session(f"{self!r} ESTABLISHED (world responder)", sub=self._sub)
        return [self._emit(SessionEventType.ESTABLISHED, {
            "profile_b": self.remote_profile,
            "rendezvous_topic": rendezvous_topic,
            "private_join": {"grant_id": grant_id.hex(),
                             "joiner_node_id": join.get("joiner_node_id", ""),
                             "role": role},
        })]

    # ── Control: GOAWAY ──────────────────────────────────────────────────────

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
        log.session(f"{self!r} unrecognised CONTROL {msg.kind} in state {self.state.value}", sub=self._sub)
        return []

    # ── Control: session-level liveness (PING/PONG) ──────────────────────────

    async def _on_ping(self, msg: SessMsg) -> List[SessionEvent]:
        """Answer a peer's liveness PING. ok=True only if we hold this exact
        Session and are usable (ESTABLISHED or mid-reconnect). A restarted peer
        with no Session never reaches here — SessionManager.deliver answers ok=False
        on its behalf."""
        ok = (bytes(msg.get("session_id") or b"") == self.session_id and
              self.state in (SessionState.ESTABLISHED, SessionState.RECONNECTING))
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

    # ── Control: RESUME (fast-path recovery before re-handshake) ─────────────

    async def _on_resume(self, msg: SessMsg) -> List[SessionEvent]:
        """Peer (initiator) is resuming after a link blip. ACK if we still hold the
        Session; NACK (SESSION_UNKNOWN) otherwise so the peer re-handshakes."""
        if (bytes(msg.get("session_id") or b"") == self.session_id and
                self.state in (SessionState.ESTABLISHED, SessionState.RECONNECTING)):
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
        return self._emit(SessionEventType.RESUMED, {})

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
        via process_inbound(GoawayAck) or check_timeout."""
        if self.state not in (SessionState.ESTABLISHED, SessionState.RECONNECTING):
            log.session(f"{self!r} close({reason_code}) called in state {self.state.value} — ignored", sub=self._sub)
            return
        await self._send_payload(SESSION_CONTROL_PROTOCOL, SessMsg(SessMsg.KIND_GOAWAY, {
            "reason": reason_code, "drain_window_ms": 0, "session_id": self.session_id,
        }))
        # Remembered so the local CLOSED event carries the same reason the remote
        # sees — the application dispatch branches on it symmetrically (e.g.
        # WORLD_HANDOFF_COMPLETE must not purge on either side).
        self._local_goaway_reason = reason_name(reason_code)
        self._set_state(SessionState.CLOSING, _TIMEOUT_CLOSING_S)
        log.session(f"{self!r} GOAWAY sent (reason={self._local_goaway_reason})", sub=self._sub)

    async def _on_remote_goaway(self, ga: SessMsg) -> List[SessionEvent]:
        await self._send_payload(SESSION_CONTROL_PROTOCOL, SessMsg(SessMsg.KIND_GOAWAY_ACK,
                                                                   {"received": True}))
        reason = reason_name(ga.get("reason", 0))
        return [self._finalize_close(f"remote_goaway:{reason}")]

    # ── Timeout sweep ─────────────────────────────────────────────────────────

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
                          SessionState.AWAITING_PRIVATE_JOIN,
                          SessionState.PRIVATE_JOIN_SENT):
            return self._fail(f"timeout:{self.state.value}")

        if self.state == SessionState.RECONNECTING:
            self._set_state(SessionState.FAILED, None)
            return self._emit(SessionEventType.LOST, {"reason": "grace_period_expired"})

        if self.state == SessionState.CLOSING:
            return self._finalize_close(
                f"goaway_ack_timeout:{getattr(self, '_local_goaway_reason', '')}")

        return None

    # ── Transport events (sync) ───────────────────────────────────────────────

    def on_transport_disconnected(self) -> Optional[SessionEvent]:
        """Called when the link drops (transport diff) or when session-level
        liveness fails. ESTABLISHED → RECONNECTING, arming the active re-dial +
        RESUME loop (tick_reconnect). Already reconnecting: reset the grace timer."""
        if self.state == SessionState.ESTABLISHED:
            self._set_state(SessionState.RECONNECTING, _TIMEOUT_RECONNECT_S)
            self._reconnect_substep = "redial"
            self._reconnect_attempts = 0
            self._next_redial_at = 0.0   # re-dial on the next tick_reconnect
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
            self._next_redial_at = 0.0   # send RESUME on the next tick_reconnect
            log.session(f"{self!r} transport reconnect signal → attempting RESUME", sub=self._sub)
        return None

    # ── Session-level keep-alive (PING/PONG over the control protocol) ───────

    async def tick_keepalive(self, now: Optional[float] = None) -> Optional[SessionEvent]:
        """Time out an unanswered ping, then send a new one if due. The PONG
        itself is handled inbound (_on_pong). Returns at most one SessionEvent
        (RECONNECTING on the miss threshold). The PING is sent inline: a
        fire-and-forget task would only start when the loop yields, which the
        mostly-synchronous main loop cannot guarantee."""
        if now is None:
            now = time.monotonic()

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

    # ── Active re-dial + RESUME (initiator-driven recovery) ──────────────────

    async def tick_reconnect(self, now: Optional[float] = None) -> Optional[SessionEvent]:
        """Drive active re-dial + RESUME while RECONNECTING (at most one attempt
        per backoff window, run inline — see tick_keepalive for why no task).
        Only an initiator with known addresses can re-dial; a responder rides the
        grace timer to LOST and is re-handshaked by the peer. Always returns
        None — the RESUMED/LOST/FAILED events surface via _on_resume_ack /
        check_timeout."""
        if now is None:
            now = time.monotonic()
        if self.state != SessionState.RECONNECTING:
            return None
        if self.role != SessionRole.INITIATOR or not self.addresses:
            return None
        # A RESUME that wasn't acked within its window → fall back to another re-dial.
        if self._reconnect_substep == "resume_sent" and now >= self._next_redial_at:
            self._reconnect_substep = "redial"
        if self._reconnect_substep == "redial" and now >= self._next_redial_at:
            await self._redial_and_resume()
        return None

    async def _redial_and_resume(self) -> None:
        """Re-dial the peer; on success send a RESUME bound to the current
        session_id. Outcome is observed on the next tick / inbound RESUME_ACK.
        Backoff is stored in _next_redial_at, never slept."""
        try:
            peer_id, through_relay = await self.p2p.connect(self.addresses)
        except Exception as e:
            log.error(f"{self!r} reconnect dial raised: {e}", sub=self._sub)
            peer_id, through_relay = None, False
        if self.state != SessionState.RECONNECTING:
            return  # grace expired (→ LOST) or otherwise advanced while awaiting
        if peer_id is not None and through_relay and self._reject_relay:
            peer_id = None  # relay-only reconnect disallowed; treat as a failed dial
        if peer_id is None:
            self._reconnect_attempts += 1
            delay = min(_RETRY_BASE_S * (2 ** (self._reconnect_attempts - 1)), _RETRY_CAP_S)
            self._next_redial_at = time.monotonic() + delay
            log.session(f"{self!r} reconnect dial failed, next try in {delay:.1f}s", sub=self._sub)
            return
        if peer_id != self.peer_id:
            self.peer_id = peer_id
        # Transport back — try to resume the existing session before re-handshaking.
        self._resume_nonce = secrets.token_bytes(8)
        self._reconnect_substep = "resume_sent"
        self._next_redial_at = time.monotonic() + _RESUME_ACK_TIMEOUT_S
        await self._send_payload(SESSION_CONTROL_PROTOCOL, SessMsg(SessMsg.KIND_RESUME, {
            "session_id": self.session_id, "nonce": self._resume_nonce,
        }))
        log.session(f"{self!r} reconnect dial ok, RESUME sent", sub=self._sub)

    # ── Internal FSM helpers ──────────────────────────────────────────────────

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

    # ── Silent abort (used by SessionManager tie-breaker) ────────────────────

    def abort_silently(self, reason: str) -> None:
        """Drop this Session without emitting an event. Used by the
        SessionManager simultaneous-open tie-breaker on the losing initiator."""
        self._set_state(SessionState.FAILED, None)
        log.session(f"{self!r} aborted silently: {reason}", sub=self._sub)
