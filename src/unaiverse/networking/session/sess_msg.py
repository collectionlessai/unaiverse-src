"""
SessMsg — self-contained value object + wire codec for Session handshake messages.

Why this module exists
----------------------
``session.py`` / ``session_manager.py`` must run **unchanged in Pyodide** (Python
in the browser, driving a JS-implemented connection layer). Pyodide mocks the
``google.protobuf`` runtime, so the FSM cannot touch ``session_pb2`` directly.

The contract that keeps the FSM portable:

  * Session/SessionManager speak only in :class:`SessMsg` — a plain ``kind`` tag
    plus a ``dict`` of fields (field names mirror ``session.proto`` exactly). They
    *construct* and *read* SessMsg, but never serialize: the conn layer does that.
  * :class:`SessMsg` is self-contained like ``p2p/messages.py``'s ``Msg``: it owns
    its own wire (de)serialization via :meth:`SessMsg.to_bytes` /
    :meth:`SessMsg.from_bytes`. The conn layer (``connpool.py`` in CPython, the JS
    ``NodeConn`` in the browser) just calls those — it no longer holds the codec.

The module **top** stays stdlib-only so importing it is safe under Pyodide. The
``session_pb2`` import lives **inside** ``to_bytes``/``from_bytes`` (a lazy import):
those methods run only in the CPython conn layer, never in the FSM, so protobuf
never enters the Pyodide import graph. Do not hoist that import to module top, and
do not import anything that pulls protobuf at module level (e.g. ``p2p/messages.py``).
The JS twin is a single ``sess_msg.js`` carrying both the value object and the codec.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ── ReasonCode (verbatim from session.proto enum ReasonCode) ──────────────────
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
TOKEN_EXPIRED          = 19  # peer's root token lapsed past the grace window

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
    TOKEN_EXPIRED: "TOKEN_EXPIRED",
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
    KIND_GOAWAY           = "goaway"
    KIND_GOAWAY_ACK       = "goaway_ack"
    # Liveness / resume control messages (post-date session.proto; the conn-layer
    # codec encodes these as a small JSON control envelope rather than protobuf —
    # see connpool.py. They ride SESSION_CONTROL_PROTOCOL like GOAWAY).
    KIND_PING             = "ping"          # {session_id: bytes, nonce: bytes}
    KIND_PONG             = "pong"          # {session_id: bytes, nonce: bytes, ok: bool}
    KIND_RESUME           = "resume"        # {session_id: bytes, nonce: bytes}
    KIND_RESUME_ACK       = "resume_ack"    # {session_id, nonce, accepted: bool, reason_code: int}
    KIND_TOKEN_UPDATE     = "token_update"  # {session_id: bytes, token: str} refreshed root token

    def get(self, key: str, default: Any = None) -> Any:
        return self.fields.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.fields[key]

    # ── Wire (de)serialization ───────────────────────────────────────────────
    # SessMsg is self-contained: it owns its own codec, like p2p/messages.py's Msg.
    # The protobuf runtime is imported lazily inside these two methods (never at
    # module top) so importing SessMsg stays protobuf-free — the FSM constructs and
    # reads SessMsg but never serializes, so these run only in the CPython conn layer
    # and protobuf never reaches the Pyodide import graph.

    def to_bytes(self) -> bytes:
        """Serialize to ``[1-byte kind tag] || payload`` wire bytes: protobuf for the
        handshake messages, a small JSON envelope for the liveness/resume control
        messages (which carry no protobuf at all)."""
        json_tag = _JSON_KIND_TO_TAG.get(self.kind)
        if json_tag is not None:
            spec = _JSON_KIND_FIELDS[self.kind]
            obj = {}
            for fn, is_bytes in spec.items():
                val = self.fields.get(fn)
                if val is None:
                    continue
                obj[fn] = bytes(val).hex() if is_bytes else val
            return bytes([json_tag]) + json.dumps(obj).encode("utf-8")

        spec = _KIND_SPEC.get(self.kind)
        if spec is None:
            raise ValueError(f"SessMsg.to_bytes: unknown SessMsg kind '{self.kind}'")
        from . import session_pb2 as _spb  # lazy: CPython conn layer only, never under Pyodide
        tag, cls_name, field_names = spec
        msg = getattr(_spb, cls_name)()
        for fn in field_names:
            if fn not in self.fields or self.fields[fn] is None:
                continue
            val = self.fields[fn]
            if fn in _REPEATED_FIELDS:
                getattr(msg, fn).extend(list(val))
            else:
                setattr(msg, fn, val)
        return bytes([tag]) + msg.SerializeToString()

    @classmethod
    def from_bytes(cls, payload: bytes, protocol_id: str | None = None) -> "SessMsg":
        """Decode ``[1-byte kind tag] || payload`` wire bytes into a SessMsg (inverse of
        :meth:`to_bytes`). ``protocol_id`` is optional and only used in error text."""
        if not payload:
            raise ValueError("SessMsg.from_bytes: empty payload")
        json_kind = _JSON_TAG_TO_KIND.get(payload[0])
        if json_kind is not None:
            spec = _JSON_KIND_FIELDS[json_kind]
            obj = json.loads(payload[1:].decode("utf-8")) if len(payload) > 1 else {}
            fields = {}
            for fn, is_bytes in spec.items():
                if fn not in obj:
                    fields[fn] = b"" if is_bytes else _JSON_FIELD_DEFAULTS.get(fn, 0)
                    continue
                fields[fn] = bytes.fromhex(obj[fn]) if is_bytes else obj[fn]
            return cls(kind=json_kind, fields=fields)

        kind = _TAG_TO_KIND.get(payload[0])
        if kind is None:
            raise ValueError(f"SessMsg.from_bytes: unknown kind tag {payload[0]} on {protocol_id}")
        from . import session_pb2 as _spb  # lazy: CPython conn layer only, never under Pyodide
        _tag, cls_name, field_names = _KIND_SPEC[kind]
        msg = getattr(_spb, cls_name)()
        msg.ParseFromString(payload[1:])
        fields = {}
        for fn in field_names:
            val = getattr(msg, fn)
            fields[fn] = list(val) if fn in _REPEATED_FIELDS else val
        return cls(kind=kind, fields=fields)


# ── SessMsg ↔ wire codec tables ───────────────────────────────────────────────
#
# Wire bytes are ``[1-byte kind tag] || payload``. Handshake messages encode with
# protobuf (session_pb2); the liveness/resume control messages ride a small JSON
# envelope instead, to avoid a protoc regen for their handful of fields. The Go
# layer is a pure pipe (it never parses these bytes), so the wire format is owned
# entirely by these tables + SessMsg.to_bytes/from_bytes and their JS mirror. The
# explicit kind tag means no direction/field heuristic is needed to tell which
# message a stream carries.

# kind -> (tag_byte, pb_class_name, [field names in proto order])
_KIND_SPEC = {
    SessMsg.KIND_HELLO_REQ:        (0x01, "HelloRequest",
                                    ["session_id", "nonce_a", "protocol_versions", "capabilities", "client_id"]),
    SessMsg.KIND_HELLO_RESP:       (0x02, "HelloResponse",
                                    ["chosen_version", "capabilities", "nonce_b", "sig_b", "reason_code"]),
    SessMsg.KIND_PROFILE_OFFER:    (0x03, "ProfileOffer",
                                    ["profile", "sig_a", "session_id", "token"]),
    SessMsg.KIND_PROFILE_DECISION: (0x04, "ProfileDecision",
                                    ["accepted", "profile_b", "reason_code", "hint_addrs", "retry_after_ms",
                                     "session_id", "token"]),
    SessMsg.KIND_WORLD_GRANT:      (0x05, "WorldGrant",
                                    ["grant_id", "expiry_unix_ms", "world_profile",
                                     "role", "code_bundle", "initial_stats",
                                     "rendezvous_tag"]),
    SessMsg.KIND_WORLD_GRANT_ACK:  (0x06, "WorldGrantAck", ["received"]),
    SessMsg.KIND_GOAWAY:           (0x09, "Goaway", ["reason", "drain_window_ms", "session_id"]),
    SessMsg.KIND_GOAWAY_ACK:       (0x0A, "GoawayAck", ["received"]),
}
_TAG_TO_KIND = {tag: kind for kind, (tag, _cls, _f) in _KIND_SPEC.items()}
_REPEATED_FIELDS = frozenset({"protocol_versions", "hint_addrs"})

# JSON-envelope codec (protobuf-free): the liveness/resume control messages ride a
# small JSON envelope rather than protobuf. `bytes` fields are hex-encoded so they
# survive JSON; the codec restores them to `bytes` on decode so the FSM sees the same
# value types as for protobuf messages.
_JSON_KIND_TO_TAG = {
    SessMsg.KIND_PING:         0x0B,
    SessMsg.KIND_PONG:         0x0C,
    SessMsg.KIND_RESUME:       0x0D,
    SessMsg.KIND_RESUME_ACK:   0x0E,
    SessMsg.KIND_TOKEN_UPDATE: 0x0F,
}
_JSON_TAG_TO_KIND = {tag: kind for kind, tag in _JSON_KIND_TO_TAG.items()}
# Field name -> True if it is a `bytes` field (hex-encoded in the JSON envelope).
_JSON_KIND_FIELDS = {
    SessMsg.KIND_PING:         {"session_id": True, "nonce": True},
    SessMsg.KIND_PONG:         {"session_id": True, "nonce": True, "ok": False},
    SessMsg.KIND_RESUME:       {"session_id": True, "nonce": True},
    SessMsg.KIND_RESUME_ACK:   {"session_id": True, "nonce": True,
                                "accepted": False, "reason_code": False},
    SessMsg.KIND_TOKEN_UPDATE: {"session_id": True, "token": False},
}
# Decode defaults for missing non-bytes JSON fields (bytes default to b"").
_JSON_FIELD_DEFAULTS = {"ok": False, "accepted": False, "joiner_node_id": "", "token": ""}


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
      and send a WorldGrant; the joiner redeems it on a separate world Session.
    On the world instance the world host just accepts and ESTABLISHes (the extra
    PRIVATE_JOIN round-trip was removed); the agent is registered on that ESTABLISHED.
    """
    accepted: bool
    reason_code: int = OK
    world_grant: Optional[WorldGrantSpec] = None


@dataclass(frozen=True)
class VerifiedTokenClaims:
    """Root-attested claims from a verified node token (RT). The responder builds
    this after verifying an inbound token against the transport peer-id, then passes
    it to the decision hook, which trusts these over the self-asserted profile:
    ``node_id`` for the allow-list, ``cv_hash`` for the profile-CV integrity check,
    ``p2p_peers`` (the root-attested [public, world] peer-ids) for the public<->world
    binding, and ``exp`` (JWT expiry, unix seconds) for the in-session refresh."""
    node_id: str
    cv_hash: str
    p2p_peers: tuple = ()
    exp: int = 0
