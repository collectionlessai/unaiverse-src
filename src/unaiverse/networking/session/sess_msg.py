"""
SessMsg — protobuf-free value object for Session handshake messages.

Why this module exists
----------------------
``session.py`` / ``session_manager.py`` must run **unchanged in Pyodide** (Python
in the browser, driving a JS-implemented connection layer). Pyodide mocks the
``google.protobuf`` runtime, so the FSM cannot touch ``session_pb2`` directly.

The split is therefore:

  * Session/SessionManager speak only in :class:`SessMsg` — a plain ``kind`` tag
    plus a ``dict`` of fields (field names mirror ``p2p/session.proto`` exactly).
    Pure stdlib; works identically in CPython and Pyodide.
  * The **conn layer** owns wire (de)serialization: in CPython
    ``connpool.py`` maps ``SessMsg`` ↔ ``session_pb2`` bytes; in the browser the
    JS ``NodeConn`` maps ``SessMsg`` ↔ ``session_pb.js`` bytes.

This module must NOT import ``google.protobuf`` (mocked in Pyodide) nor anything
that does (e.g. ``p2p/messages.py``). Stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ── ReasonCode (verbatim from p2p/session.proto enum ReasonCode) ──────────────
OK                     = 0
VERSION_UNSUPPORTED    = 1
OVERLOADED             = 2
NOT_AUTHORIZED         = 3
NOT_IN_ALLOW_LIST      = 4
CV_HASH_MISMATCH       = 5
ROLE_INCOMPATIBLE      = 6
WORLD_BUSY             = 7
HOSTED_REJECTED        = 8
GRACEFUL_SHUTDOWN      = 9
WORLD_DISENGAGEMENT    = 10
WORLD_HANDOFF_COMPLETE = 11
ROLE_REVOKED           = 12
TOKEN_REVOKED          = 13
KICKED_BY_HOST         = 14
LOCAL_ERROR            = 15
GRANT_EXPIRED          = 16
SIMULTANEOUS_OPEN_LOSS = 17
SESSION_UNKNOWN        = 18  # liveness: peer holds no Session for this session_id

REASON_NAME: Dict[int, str] = {
    OK: "OK",
    VERSION_UNSUPPORTED: "VERSION_UNSUPPORTED",
    OVERLOADED: "OVERLOADED",
    NOT_AUTHORIZED: "NOT_AUTHORIZED",
    NOT_IN_ALLOW_LIST: "NOT_IN_ALLOW_LIST",
    CV_HASH_MISMATCH: "CV_HASH_MISMATCH",
    ROLE_INCOMPATIBLE: "ROLE_INCOMPATIBLE",
    WORLD_BUSY: "WORLD_BUSY",
    HOSTED_REJECTED: "HOSTED_REJECTED",
    GRACEFUL_SHUTDOWN: "GRACEFUL_SHUTDOWN",
    WORLD_DISENGAGEMENT: "WORLD_DISENGAGEMENT",
    WORLD_HANDOFF_COMPLETE: "WORLD_HANDOFF_COMPLETE",
    ROLE_REVOKED: "ROLE_REVOKED",
    TOKEN_REVOKED: "TOKEN_REVOKED",
    KICKED_BY_HOST: "KICKED_BY_HOST",
    LOCAL_ERROR: "LOCAL_ERROR",
    GRANT_EXPIRED: "GRANT_EXPIRED",
    SIMULTANEOUS_OPEN_LOSS: "SIMULTANEOUS_OPEN_LOSS",
    SESSION_UNKNOWN: "SESSION_UNKNOWN",
}


def reason_name(code: int) -> str:
    """``spb.ReasonCode.Name`` replacement (returns the int as str if unknown)."""
    return REASON_NAME.get(int(code), str(code))


# ── SessMsg ───────────────────────────────────────────────────────────────────

@dataclass
class SessMsg:
    """A single Session handshake message as a portable value object.

    ``kind`` is one of the ``KIND_*`` constants below; ``fields`` holds the
    message fields keyed by the exact ``session.proto`` field names. ``bytes``
    fields stay ``bytes`` and ints stay ints — the conn-layer codec handles the
    wire encoding (protobuf in CPython, protobuf-js in the browser).
    """
    kind: str
    fields: Dict[str, Any] = field(default_factory=dict)

    # Message kinds — one per session.proto message type.
    KIND_HELLO_REQ        = "hello_request"
    KIND_HELLO_RESP       = "hello_response"
    KIND_PROFILE_OFFER    = "profile_offer"
    KIND_PROFILE_DECISION = "profile_decision"
    KIND_WORLD_GRANT      = "world_grant"
    KIND_WORLD_GRANT_ACK  = "world_grant_ack"
    KIND_PRIVATE_JOIN     = "private_join"
    KIND_PRIVATE_JOIN_ACK = "private_join_ack"
    KIND_GOAWAY           = "goaway"
    KIND_GOAWAY_ACK       = "goaway_ack"
    # Liveness / resume control messages (post-date session.proto; the conn-layer
    # codec encodes these as a small JSON control envelope rather than protobuf —
    # see connpool.py. They ride SESSION_CONTROL_PROTOCOL like GOAWAY).
    KIND_PING             = "ping"          # {session_id: bytes, nonce: bytes}
    KIND_PONG             = "pong"          # {session_id: bytes, nonce: bytes, ok: bool}
    KIND_RESUME           = "resume"        # {session_id: bytes, nonce: bytes}
    KIND_RESUME_ACK       = "resume_ack"    # {session_id, nonce, accepted: bool, reason_code: int}

    def get(self, key: str, default: Any = None) -> Any:
        return self.fields.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.fields[key]


# ── Application decision types (returned by the responder decision hook) ──────

@dataclass
class WorldGrantSpec:
    """What a world host returns from its decision hook when it accepts a joiner
    on its PUBLIC Session. The Session uses this to build a ``WorldGrant`` message;
    ``*_bytes`` fields are already serialized payloads. ``code_bundle`` carries the
    application's agent code + stats code (the host packs them however it likes —
    e.g. a JSON blob — and the joiner unpacks it); the proto has no separate
    agent-stats field, so anything beyond the named fields rides in ``code_bundle``.
    """
    role: int
    world_profile_bytes: bytes
    rendezvous_tag: int
    code_bundle: bytes = b""
    initial_stats: bytes = b""
    grant_ttl_ms: int = 30000


@dataclass
class ProfileDecisionResult:
    """Return value of the responder decision hook (see SessionManager).

    * ``world_grant`` set ⇒ responder is a world host on its PUBLIC instance: mint
      and send a WorldGrant, then await the joiner's PRIVATE_JOIN on the world
      instance.
    * ``await_private_join`` ⇒ responder is a world host on its WORLD instance: the
      peer is redeeming a grant, so do not ESTABLISH on PROFILE — wait for the
      PRIVATE_JOIN that follows.
    """
    accepted: bool
    reason_code: int = OK
    world_grant: Optional[WorldGrantSpec] = None
    await_private_join: bool = False
