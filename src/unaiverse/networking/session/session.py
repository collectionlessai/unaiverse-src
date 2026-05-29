"""
Session — per-peer connection lifecycle FSM.

Each Session owns the entire lifecycle of a connection to one remote peer:
  dial (via conn.connect) → 4-stage handshake → established → /ipfs/ping
  keep-alive → reconnect grace → intentional GOAWAY close.

Design (v1 — polling mode)
--------------------------
The FSM is *synchronous* with async send helpers: ``process_inbound()`` and
``check_timeout()`` are called from the polling loop (Node's
``__handle_network_messages`` / main loop) and advance state immediately.
``process_inbound()`` is async only because some transitions need to
``await conn.send_session()``.

No asyncio.Queue, no background Task for the FSM — the polling loop drives it.
The single exception is the keep-alive ping, which runs in a fire-and-forget
``asyncio.create_task`` and posts its result back into a tiny single-writer
queue drained on the next ``tick_keepalive()`` call.

Each stage runs on its own libp2p protocol:
  /unaiverse/session/hello/1.0.0     -- HelloRequest / HelloResponse
  /unaiverse/session/profile/1.0.0   -- ProfileOffer / ProfileDecision
  /unaiverse/session/world/1.0.0     -- WorldGrant / WorldGrantAck,
                                        PrivateJoin / PrivateJoinAck
  /unaiverse/session/control/1.0.0   -- Goaway / GoawayAck

Payloads are protobuf messages from p2p/session.proto. Identity is bound by
``sig_b`` (Stage-1, responder over ``nonce_a||session_id``), ``sig_a`` (Stage-2,
initiator over ``nonce_b||session_id``), and the libp2p SignedPeerRecord
(Stage-2, binding the agent's *other* peer-id to its addresses). No per-message
JWT piggyback for session protocols.
"""

from __future__ import annotations

import json
import secrets
import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, TYPE_CHECKING

from unaiverse.networking.p2p import session_pb2 as spb

if TYPE_CHECKING:
    from unaiverse.networking.node.connpool import ConnectionPools
    from unaiverse.networking.p2p.p2p import P2P

log = logging.getLogger("Session")

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
_TIMEOUT_RECONNECT_S = 10.0 # Grace period before Session.Lost (matches Go DisconnectionGracePeriod)
_TIMEOUT_CLOSING_S = 10.0   # GOAWAY sent; wait for GOAWAY_ACK

# ── Retry / backoff ──────────────────────────────────────────────────────────
_MAX_DIAL_ATTEMPTS = 4
_RETRY_BASE_S = 1.0
_RETRY_CAP_S = 30.0

# ── Keep-alive (/ipfs/ping) ──────────────────────────────────────────────────
_KEEPALIVE_PERIOD_S = 15.0
_KEEPALIVE_TIMEOUT_MS = 5000
_KEEPALIVE_FAIL_THRESHOLD = 2

# ── Misc ─────────────────────────────────────────────────────────────────────
_SESSION_ID_BYTES = 16
_NONCE_BYTES = 32


# ── State / Role / Event enums ────────────────────────────────────────────────

class SessionState(str, Enum):
    INIT = "init"
    CONNECTING = "connecting"           # initiator: dialing
    HELLO_SENT = "hello_sent"           # initiator: HELLO out, awaiting HELLO_ACK
    HELLO_ACK_SENT = "hello_ack_sent"   # responder: HELLO_ACK out, awaiting PROFILE_OFFER
    PROFILE_SENT = "profile_sent"       # initiator: PROFILE_OFFER out, awaiting PROFILE_DECISION
    AWAITING_WORLD_GRANT = "awaiting_world_grant"  # initiator: world joiner expecting WORLD_GRANT
    AWAITING_PRIVATE_JOIN = "awaiting_private_join"  # responder world: WORLD_GRANT out, awaiting PRIVATE_JOIN
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sign_payload(nonce: bytes, session_id: bytes) -> bytes:
    """Canonical sign-payload for sig_a / sig_b: ``nonce || session_id``."""
    return bytes(nonce) + bytes(session_id)


# ── Session ───────────────────────────────────────────────────────────────────

class Session:
    """Per-peer session FSM. See module docstring for lifecycle."""

    def __init__(
        self,
        peer_id: str,
        role: SessionRole,
        conn: "ConnectionPools",
        p2p_name: str,
        local_peer_id: str,
        local_node_id: str,
        local_profile_factory,
        local_spr_other: Optional[bytes],
        remote_node_id_hint: Optional[str] = None,
        remote_other_peer_id_hint: Optional[str] = None,
        addresses: Optional[List[str]] = None,
        is_world_join: bool = False,
    ):
        """
        Args:
            peer_id: libp2p peer-id of the remote (on this p2p instance).
            role: INITIATOR or RESPONDER.
            conn: the ConnectionPools instance.
            p2p_name: the P2P instance this Session lives on.
            local_peer_id: our libp2p peer-id on `p2p_name`.
            local_node_id: our application-level node-id (the JWT-bound identity).
            local_profile_factory: zero-arg callable that returns the current NodeProfile dict.
            local_spr_other: serialized SignedPeerRecord for our *other* peer-id
                (the p2p_world key when this Session runs on p2p_public). May be
                None if we don't have a second peer-id yet — Stage-2 then ships
                an empty signed_peer_record and responders that require it reject.
            remote_node_id_hint: optional; if known, used to bind the JWT identity.
            remote_other_peer_id_hint: optional; the remote's *other* peer-id,
                needed to verify the inbound SignedPeerRecord.
            addresses: initiator only — peer multiaddrs from root server / cache.
            is_world_join: initiator only — if True, also drive Stage-3 (world handoff).
        """
        self.peer_id = peer_id
        self.role = role
        self.conn = conn
        self.p2p_name = p2p_name
        self.local_peer_id = local_peer_id
        self.local_node_id = local_node_id
        self._local_profile_factory = local_profile_factory
        self._local_spr_other = local_spr_other or b""
        self.remote_node_id_hint = remote_node_id_hint
        self.remote_other_peer_id_hint = remote_other_peer_id_hint
        self.addresses = list(addresses or [])
        self.is_world_join = is_world_join

        # Resolved once; avoids repeated dict look-up in hot path.
        self._p2p: "P2P" = conn.p2p_name_to_p2p[p2p_name]

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

        # Keep-alive state.
        self._last_ping_at: float = 0.0
        self._consecutive_ping_failures: int = 0
        self._keepalive_in_flight: bool = False
        # Single-producer, drained on tick_keepalive.
        self._pending_keepalive_results: List[Optional[int]] = []

        # World-join specifics (initiator only).
        self.grant_id: bytes = b""
        self.grant_sig: bytes = b""
        self.grant_expiry_unix_ms: int = 0

    # ── repr ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (f"Session({self.peer_id[:12]}… role={self.role.value} "
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
        log.info(f"{self!r} dial attempt {self._dial_attempts}/{_MAX_DIAL_ATTEMPTS}")

        try:
            peer_id, _through_relay = await self.conn.connect(self.addresses, p2p_name=self.p2p_name)
        except Exception as e:
            log.warning(f"{self!r} conn.connect raised: {e}")
            peer_id = None

        if peer_id is None:
            if self._dial_attempts < _MAX_DIAL_ATTEMPTS:
                import random
                delay = min(
                    random.uniform(0, _RETRY_BASE_S * (2 ** (self._dial_attempts - 1))),
                    _RETRY_CAP_S,
                )
                log.info(f"{self!r} dial failed, retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
                return await self._dial_then_hello()
            log.error(f"{self!r} dial failed after {_MAX_DIAL_ATTEMPTS} attempts")
            return [self._fail("dial_failed")]

        # If the dial returned a different peer-id than we expected (e.g., relayed
        # multiaddr without /p2p hop), update it.
        if peer_id != self.peer_id:
            log.info(f"{self!r} peer_id updated from {self.peer_id[:12]}… → {peer_id[:12]}…")
            self.peer_id = peer_id

        return await self._send_hello()

    async def send_hello(self):
        self.conn._send_hello()

    async def _send_hello(self) -> List[SessionEvent]:
        self.nonce_a = secrets.token_bytes(_NONCE_BYTES)
        msg = spb.HelloRequest(
            session_id=self.session_id,
            nonce_a=self.nonce_a,
            protocol_versions=["1.0"],
            capabilities=0,
            client_id="python/1.0",
        )
        ok = await self.conn.send_session(
            peer_id=self.peer_id,
            protocol_id=SESSION_HELLO_PROTOCOL,
            payload=msg.SerializeToString(),
            p2p=self._p2p,
        )
        if not ok:
            return [self._fail("hello_send_failed")]
        self._set_state(SessionState.HELLO_SENT, _TIMEOUT_HELLO_S)
        log.info(f"{self!r} HELLO sent")
        return []

    # ── Inbound dispatch ──────────────────────────────────────────────────────

    async def process_inbound(self, protocol_id: str, payload: bytes) -> List[SessionEvent]:
        """Route an inbound session-protocol payload through the FSM. Called by
        SessionManager.deliver for every SessionInbound."""
        events: List[SessionEvent] = []
        try:
            if protocol_id == SESSION_HELLO_PROTOCOL:
                events += await self._on_hello_protocol(payload)
            elif protocol_id == SESSION_PROFILE_PROTOCOL:
                events += await self._on_profile_protocol(payload)
            elif protocol_id == SESSION_WORLD_PROTOCOL:
                events += await self._on_world_protocol(payload)
            elif protocol_id == SESSION_CONTROL_PROTOCOL:
                events += await self._on_control_protocol(payload)
            else:
                log.debug(f"{self!r} unknown protocol {protocol_id}")
        except Exception as e:
            log.error(f"{self!r} process_inbound({protocol_id}) raised: {e}", exc_info=True)
            events.append(self._fail(f"inbound_dispatch_error:{e}"))
        return events

    # ── Stage-1: HELLO ────────────────────────────────────────────────────────

    async def _on_hello_protocol(self, payload: bytes) -> List[SessionEvent]:
        # Two distinguishable wire messages on the hello protocol:
        # responder receives HelloRequest; initiator receives HelloResponse.
        if self.role == SessionRole.RESPONDER and self.state == SessionState.INIT:
            req = spb.HelloRequest()
            req.ParseFromString(payload)
            return await self._responder_on_hello(req)

        if self.role == SessionRole.INITIATOR and self.state == SessionState.HELLO_SENT:
            resp = spb.HelloResponse()
            resp.ParseFromString(payload)
            return await self._initiator_on_hello_ack(resp)

        log.debug(f"{self!r} ignored HELLO-protocol payload in state {self.state.value}")
        return []

    async def _responder_on_hello(self, req: "spb.HelloRequest") -> List[SessionEvent]:
        if not req.session_id:
            return [self._fail("hello_missing_session_id")]
        self.session_id = bytes(req.session_id)
        self.session_id_hex = self.session_id.hex()
        self.nonce_a = bytes(req.nonce_a)

        # Version negotiation — single supported version for v1.
        chosen = "1.0" if "1.0" in list(req.protocol_versions) else ""
        if not chosen:
            resp = spb.HelloResponse(
                chosen_version="",
                capabilities=0,
                nonce_b=b"",
                sig_b=b"",
                reason_code=spb.VERSION_UNSUPPORTED,
            )
            await self._send_payload(SESSION_HELLO_PROTOCOL, resp)
            return [self._fail("version_unsupported")]

        self.nonce_b = secrets.token_bytes(_NONCE_BYTES)
        sig_b = self._p2p.sign_with_host_key(_sign_payload(self.nonce_a, self.session_id))
        if sig_b is None:
            return [self._fail("hello_sign_failed")]

        resp = spb.HelloResponse(
            chosen_version=chosen,
            capabilities=0,
            nonce_b=self.nonce_b,
            sig_b=sig_b,
            reason_code=spb.OK,
        )
        if not await self._send_payload(SESSION_HELLO_PROTOCOL, resp):
            return [self._fail("hello_ack_send_failed")]
        self._set_state(SessionState.HELLO_ACK_SENT, _TIMEOUT_PROFILE_S)
        log.info(f"{self!r} HELLO_ACK sent")
        return []

    async def _initiator_on_hello_ack(self, resp: "spb.HelloResponse") -> List[SessionEvent]:
        if resp.reason_code != spb.OK:
            return [self._fail(f"hello_ack_rejected:{spb.ReasonCode.Name(resp.reason_code)}")]
        if resp.chosen_version != "1.0":
            return [self._fail(f"hello_ack_bad_version:{resp.chosen_version}")]

        # Verify sig_b under the responder's peer-id (libp2p host key).
        if not self._p2p.verify_peer_signature(
            self.peer_id,
            _sign_payload(bytes(self.nonce_a), self.session_id),
            bytes(resp.sig_b),
        ):
            return [self._fail("hello_ack_bad_sig_b")]

        self.nonce_b = bytes(resp.nonce_b)
        return await self._initiator_send_profile_offer()

    # ── Stage-2: PROFILE ──────────────────────────────────────────────────────

    async def _initiator_send_profile_offer(self) -> List[SessionEvent]:
        sig_a = self._p2p.sign_with_host_key(_sign_payload(self.nonce_b, self.session_id))
        if sig_a is None:
            return [self._fail("profile_offer_sign_failed")]
        profile_bytes = json.dumps(self._local_profile_factory()).encode("utf-8")
        offer = spb.ProfileOffer(
            profile=profile_bytes,
            sig_a=sig_a,
            signed_peer_record=self._local_spr_other,
            session_id=self.session_id,
        )
        if not await self._send_payload(SESSION_PROFILE_PROTOCOL, offer):
            return [self._fail("profile_offer_send_failed")]
        self._set_state(SessionState.PROFILE_SENT, _TIMEOUT_PROFILE_S)
        log.info(f"{self!r} PROFILE_OFFER sent")
        return []

    async def _on_profile_protocol(self, payload: bytes) -> List[SessionEvent]:
        if self.role == SessionRole.RESPONDER and self.state == SessionState.HELLO_ACK_SENT:
            offer = spb.ProfileOffer()
            offer.ParseFromString(payload)
            return await self._responder_on_profile_offer(offer)

        if self.role == SessionRole.INITIATOR and self.state == SessionState.PROFILE_SENT:
            decision = spb.ProfileDecision()
            decision.ParseFromString(payload)
            return self._initiator_on_profile_decision(decision)

        log.debug(f"{self!r} ignored PROFILE-protocol payload in state {self.state.value}")
        return []

    async def _responder_on_profile_offer(self, offer: "spb.ProfileOffer") -> List[SessionEvent]:
        # Verify sig_a under the initiator's peer-id.
        if not self._p2p.verify_peer_signature(
            self.peer_id,
            _sign_payload(bytes(self.nonce_b), self.session_id),
            bytes(offer.sig_a),
        ):
            return await self._reject_profile(spb.NOT_AUTHORIZED, "bad_sig_a")

        # If we know the initiator's other peer-id, verify the signed peer record
        # binds those bytes to that id. If we don't have a hint, we skip — the
        # other side may not be a dual-key agent.
        if self.remote_other_peer_id_hint and offer.signed_peer_record:
            if not self._p2p.verify_signed_peer_record(
                bytes(offer.signed_peer_record),
                self.remote_other_peer_id_hint,
            ):
                return await self._reject_profile(spb.NOT_AUTHORIZED, "bad_signed_peer_record")

        try:
            self.remote_profile = json.loads(bytes(offer.profile).decode("utf-8"))
        except Exception as e:
            return await self._reject_profile(spb.LOCAL_ERROR, f"bad_profile_json:{e}")

        # v1: always accept. Future: call hosted validation hook here (allow-list,
        # cv_hash, role compatibility, rate-limit).
        decision = spb.ProfileDecision(
            accepted=True,
            profile_b=json.dumps(self._local_profile_factory()).encode("utf-8"),
            reason_code=spb.OK,
            session_id=self.session_id,
        )
        if not await self._send_payload(SESSION_PROFILE_PROTOCOL, decision):
            return [self._fail("profile_decision_send_failed")]

        self._set_state(SessionState.ESTABLISHED, None)
        log.info(f"{self!r} ESTABLISHED (responder)")
        self._last_ping_at = time.monotonic()
        return [self._emit(SessionEventType.ESTABLISHED, {"profile_b": self.remote_profile})]

    async def _reject_profile(self, reason_code: int, log_reason: str) -> List[SessionEvent]:
        decision = spb.ProfileDecision(
            accepted=False,
            reason_code=reason_code,
            session_id=self.session_id,
        )
        await self._send_payload(SESSION_PROFILE_PROTOCOL, decision)
        return [self._fail(f"rejected_offer:{log_reason}")]

    def _initiator_on_profile_decision(self, decision: "spb.ProfileDecision") -> List[SessionEvent]:
        if not decision.accepted:
            reason = spb.ReasonCode.Name(decision.reason_code) if decision.reason_code else "unspecified"
            return [self._fail(f"profile_rejected:{reason}")]
        try:
            self.remote_profile = json.loads(bytes(decision.profile_b).decode("utf-8"))
        except Exception as e:
            return [self._fail(f"profile_decision_bad_profile_b:{e}")]

        if self.is_world_join:
            self._set_state(SessionState.AWAITING_WORLD_GRANT, _TIMEOUT_WORLD_S)
            log.info(f"{self!r} PROFILE accepted, awaiting WORLD_GRANT")
            return []  # Stage-3 inbound triggers ESTABLISHED for the world Session

        self._set_state(SessionState.ESTABLISHED, None)
        log.info(f"{self!r} ESTABLISHED (initiator)")
        self._last_ping_at = time.monotonic()
        return [self._emit(SessionEventType.ESTABLISHED, {"profile_b": self.remote_profile})]

    # ── Stage-3: WORLD (initiator side only for v1 — responder/world hosting
    # uses the same protocol but its driving logic lives in node.py's world
    # role handler, which calls Session._send_world_grant directly) ──────────

    async def _on_world_protocol(self, payload: bytes) -> List[SessionEvent]:
        if (self.role == SessionRole.INITIATOR and
                self.state == SessionState.AWAITING_WORLD_GRANT):
            grant = spb.WorldGrant()
            grant.ParseFromString(payload)
            return await self._initiator_on_world_grant(grant)

        if (self.role == SessionRole.RESPONDER and
                self.state == SessionState.AWAITING_PRIVATE_JOIN):
            join = spb.PrivateJoin()
            join.ParseFromString(payload)
            return await self._responder_on_private_join(join)

        log.debug(f"{self!r} ignored WORLD-protocol payload in state {self.state.value}")
        return []

    async def _initiator_on_world_grant(self, grant: "spb.WorldGrant") -> List[SessionEvent]:
        if not grant.grant_id:
            return [self._fail("world_grant_missing_id")]
        if grant.expiry_unix_ms < int(time.time() * 1000):
            return [self._fail("world_grant_expired")]

        self.grant_id = bytes(grant.grant_id)
        self.grant_sig = bytes(grant.grant_sig)
        self.grant_expiry_unix_ms = grant.expiry_unix_ms

        ack = spb.WorldGrantAck(received=True)
        await self._send_payload(SESSION_WORLD_PROTOCOL, ack)

        # The public Session stays ESTABLISHED until the *world* Session reports
        # ESTABLISHED — node.py owns that orchestration and will then call
        # Session.close(WORLD_HANDOFF_COMPLETE) on this Session.
        self._set_state(SessionState.ESTABLISHED, None)
        self._last_ping_at = time.monotonic()
        log.info(f"{self!r} WORLD_GRANT received, public session ESTABLISHED (world handoff pending)")
        return [self._emit(SessionEventType.ESTABLISHED, {
            "profile_b": self.remote_profile,
            "world_grant": {
                "grant_id": self.grant_id.hex(),
                "grant_sig": self.grant_sig.hex(),
                "expiry_unix_ms": self.grant_expiry_unix_ms,
                "world_profile_bytes": bytes(grant.world_profile),
                "role": grant.role,
                "rendezvous_tag": grant.rendezvous_tag,
                "world_private_peer_record": bytes(grant.world_private_peer_record),
            },
        })]

    async def _responder_on_private_join(self, join: "spb.PrivateJoin") -> List[SessionEvent]:
        # Verify sig_init under the joiner's world-side peer-id (this responder
        # IS the world, running on p2p_world; the joiner's hint was captured by
        # SessionManager when it scaffolded this Session).
        if self.remote_other_peer_id_hint and not self._p2p.verify_peer_signature(
            self.remote_other_peer_id_hint,
            bytes(join.grant_id),
            bytes(join.sig_init),
        ):
            ack = spb.PrivateJoinAck(accepted=False, reason_code=spb.NOT_AUTHORIZED, rendezvous_topic="")
            await self._send_payload(SESSION_WORLD_PROTOCOL, ack)
            return [self._fail("private_join_bad_sig")]

        rendezvous_topic = f"world::{self.peer_id}::ps:rv"
        ack = spb.PrivateJoinAck(accepted=True, reason_code=spb.OK, rendezvous_topic=rendezvous_topic)
        if not await self._send_payload(SESSION_WORLD_PROTOCOL, ack):
            return [self._fail("private_join_ack_send_failed")]

        self._set_state(SessionState.ESTABLISHED, None)
        self._last_ping_at = time.monotonic()
        log.info(f"{self!r} ESTABLISHED (world responder)")
        return [self._emit(SessionEventType.ESTABLISHED, {
            "profile_b": self.remote_profile,
            "rendezvous_topic": rendezvous_topic,
        })]

    # ── Control: GOAWAY ──────────────────────────────────────────────────────

    async def _on_control_protocol(self, payload: bytes) -> List[SessionEvent]:
        # Try parsing as Goaway first; if that fails, GoawayAck.
        try:
            ga = spb.Goaway()
            ga.ParseFromString(payload)
            if ga.session_id:
                return await self._on_remote_goaway(ga)
        except Exception:
            pass

        if self.state == SessionState.CLOSING:
            try:
                ack = spb.GoawayAck()
                ack.ParseFromString(payload)
                if ack.received:
                    return [self._finalize_close("local_goaway_acked")]
            except Exception:
                pass
        log.debug(f"{self!r} unrecognised CONTROL payload in state {self.state.value}")
        return []

    async def close(self, reason_code: int = spb.GRACEFUL_SHUTDOWN) -> None:
        """Send GOAWAY to the remote peer and enter CLOSING. The FSM completes
        via process_inbound(GoawayAck) or check_timeout."""
        if self.state not in (SessionState.ESTABLISHED, SessionState.RECONNECTING):
            log.debug(f"{self!r} close({reason_code}) called in state {self.state.value} — ignored")
            return
        msg = spb.Goaway(reason=reason_code, drain_window_ms=0, session_id=self.session_id)
        await self._send_payload(SESSION_CONTROL_PROTOCOL, msg)
        self._set_state(SessionState.CLOSING, _TIMEOUT_CLOSING_S)
        log.info(f"{self!r} GOAWAY sent (reason={spb.ReasonCode.Name(reason_code)})")

    async def _on_remote_goaway(self, ga: "spb.Goaway") -> List[SessionEvent]:
        ack = spb.GoawayAck(received=True)
        await self._send_payload(SESSION_CONTROL_PROTOCOL, ack)
        reason = spb.ReasonCode.Name(ga.reason) if ga.reason else "unspecified"
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

        log.warning(f"{self!r} TIMEOUT after {elapsed:.1f}s in state {self.state.value}")

        if self.state in (SessionState.CONNECTING,
                          SessionState.HELLO_SENT,
                          SessionState.PROFILE_SENT,
                          SessionState.HELLO_ACK_SENT,
                          SessionState.AWAITING_WORLD_GRANT,
                          SessionState.AWAITING_PRIVATE_JOIN):
            return self._fail(f"timeout:{self.state.value}")

        if self.state == SessionState.RECONNECTING:
            self._set_state(SessionState.FAILED, None)
            return self._emit(SessionEventType.LOST, {"reason": "grace_period_expired"})

        if self.state == SessionState.CLOSING:
            return self._finalize_close("goaway_ack_timeout")

        return None

    # ── Transport events (sync) ───────────────────────────────────────────────

    def on_transport_disconnected(self) -> Optional[SessionEvent]:
        """Called when the underlying libp2p connection drops unexpectedly.
        ESTABLISHED → RECONNECTING (grace); already reconnecting: reset timer."""
        if self.state == SessionState.ESTABLISHED:
            self._set_state(SessionState.RECONNECTING, _TIMEOUT_RECONNECT_S)
            log.info(f"{self!r} transport dropped, starting {_TIMEOUT_RECONNECT_S}s grace")
            return self._emit(SessionEventType.RECONNECTING, {"cause": "transport_disconnected"})
        if self.state == SessionState.RECONNECTING:
            self._state_entered_at = time.monotonic()
        return None

    def on_transport_reconnected(self) -> Optional[SessionEvent]:
        if self.state == SessionState.RECONNECTING:
            self._set_state(SessionState.ESTABLISHED, None)
            self._last_ping_at = time.monotonic()
            self._consecutive_ping_failures = 0
            log.info(f"{self!r} transport restored → ESTABLISHED (resumed)")
            return self._emit(SessionEventType.RESUMED, {})
        return None

    # ── /ipfs/ping keep-alive ────────────────────────────────────────────────

    def tick_keepalive(self, now: Optional[float] = None) -> Optional[SessionEvent]:
        """Drain any pending ping results, then schedule a new ping if due.
        Returns at most one SessionEvent (RECONNECTING on threshold hit)."""
        if now is None:
            now = time.monotonic()

        # Drain results from any prior fire-and-forget pings.
        for result_ms in self._pending_keepalive_results:
            if result_ms is None:
                self._consecutive_ping_failures += 1
                log.info(f"{self!r} ping miss ({self._consecutive_ping_failures}/{_KEEPALIVE_FAIL_THRESHOLD})")
            else:
                self._consecutive_ping_failures = 0
                log.debug(f"{self!r} ping OK ({result_ms}ms)")
        self._pending_keepalive_results.clear()

        if (self._consecutive_ping_failures >= _KEEPALIVE_FAIL_THRESHOLD and
                self.state == SessionState.ESTABLISHED):
            self._consecutive_ping_failures = 0
            return self.on_transport_disconnected()

        if (self.state == SessionState.ESTABLISHED and
                not self._keepalive_in_flight and
                now - self._last_ping_at >= _KEEPALIVE_PERIOD_S):
            self._last_ping_at = now
            self._keepalive_in_flight = True
            asyncio.create_task(self._do_ping())
        return None

    async def _do_ping(self) -> None:
        try:
            rtt = await asyncio.to_thread(self._p2p.ping_peer, self.peer_id, _KEEPALIVE_TIMEOUT_MS)
        except Exception as e:
            log.error(f"{self!r} ping_peer raised: {e}")
            rtt = None
        finally:
            self._pending_keepalive_results.append(rtt)
            self._keepalive_in_flight = False

    # ── Internal FSM helpers ──────────────────────────────────────────────────

    def _set_state(self, state: SessionState, timeout_s: Optional[float]) -> None:
        prev = self.state
        self.state = state
        self._state_entered_at = time.monotonic()
        self._timeout_s = timeout_s
        log.debug(f"{self!r} {prev.value} → {state.value} (timeout={timeout_s}s)")

    def _fail(self, reason: str) -> SessionEvent:
        self._set_state(SessionState.FAILED, None)
        log.error(f"{self!r} FAILED: {reason}")
        return self._emit(SessionEventType.FAILED, {"reason": reason})

    def _finalize_close(self, reason: str) -> SessionEvent:
        self._set_state(SessionState.CLOSED, None)
        log.info(f"{self!r} CLOSED: {reason}")
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

    async def _send_payload(self, protocol_id: str, msg) -> bool:
        try:
            payload = msg.SerializeToString()
        except Exception as e:
            log.error(f"{self!r} serialize {protocol_id} failed: {e}")
            return False
        return await self.conn.send_session(
            peer_id=self.peer_id,
            protocol_id=protocol_id,
            payload=payload,
            p2p=self._p2p,
        )

    # ── Silent abort (used by SessionManager tie-breaker) ────────────────────

    def abort_silently(self, reason: str) -> None:
        """Drop this Session without emitting an event. Used by the
        SessionManager simultaneous-open tie-breaker on the losing initiator."""
        self._set_state(SessionState.FAILED, None)
        log.info(f"{self!r} aborted silently: {reason}")
