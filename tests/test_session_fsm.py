"""Pure-FSM unit tests for ``Session``. No real network, no Go.

These tests stub the conn/p2p surface so we can drive Session through every
state transition and confirm the events / sends it emits. They protect against
regressions in:

  - Stage-1/2/3 happy-path transitions
  - Replay defence (sig_a / sig_b verification gating)
  - Timeout firing in every blocking state
  - Disconnect → grace → resume / lost
  - Intentional close (GOAWAY / GOAWAY_ACK)
  - Keep-alive ping success / failure threshold
"""

import time
import asyncio
import pytest

from unaiverse.networking.p2p import session_pb2 as spb
from unaiverse.networking.session import (
    Session, SessionState, SessionRole, SessionEventType,
    SESSION_HELLO_PROTOCOL, SESSION_PROFILE_PROTOCOL,
    SESSION_WORLD_PROTOCOL, SESSION_CONTROL_PROTOCOL,
)
from unaiverse.networking.node.connpool import NodeConn


# ── Helpers ─────────────────────────────────────────────────────────────────

def _last_sent(conn_or_p2p, protocol_id):
    """Return the last (peer, proto, payload) sent on protocol_id from the
    given FakeConn or FakeP2P, or None.

    Session goes through ``conn.send_session(...)``, which is the FakeConn —
    so for Session-level assertions pass the FakeConn. (FakeP2P.sent only
    records direct ``send_on_protocol`` calls, which Session doesn't make.)"""
    for entry in reversed(conn_or_p2p.sent):
        if entry[1] == protocol_id:
            return entry
    return None


def _parse(msg_class, payload):
    m = msg_class()
    m.ParseFromString(payload)
    return m


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Initiator happy path ────────────────────────────────────────────────────

def test_initiator_start_sends_hello(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """start() must dial, then emit HelloRequest on the hello protocol and
    park in HELLO_SENT awaiting HELLO_ACK."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = Session(
        peer_id=bob_peer_id, role=SessionRole.INITIATOR, conn=alice_conn,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=alice_peer_id,
        local_node_id="alice-node", local_profile_factory=lambda: {"x": 1},
        local_spr_other=b"alice-spr",
        addresses=["/ip4/127.0.0.1/tcp/0/p2p/" + bob_peer_id],
    )
    events = _run(sess.start())
    assert events == []
    assert sess.state == SessionState.HELLO_SENT
    assert alice_conn.connect_calls == [(["/ip4/127.0.0.1/tcp/0/p2p/" + bob_peer_id], NodeConn.P2P_PUBLIC)]
    last = _last_sent(alice_conn, SESSION_HELLO_PROTOCOL)
    assert last is not None, "HelloRequest must be sent on the hello protocol"
    hello = _parse(spb.HelloRequest, last[2])
    assert hello.session_id == sess.session_id
    assert len(hello.nonce_a) == 32
    assert "1.0" in list(hello.protocol_versions)


def test_initiator_dial_failure_fails_after_max_attempts(alice_conn, alice_peer_id, bob_peer_id, monkeypatch):
    """All 4 dial attempts return None → terminal FAILED event."""
    alice_conn.connect_returns = (None, False)
    # Skip the actual backoff sleep so the test runs in milliseconds.
    async def _no_sleep(_s):
        return None
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    sess = Session(
        peer_id=bob_peer_id, role=SessionRole.INITIATOR, conn=alice_conn,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=alice_peer_id,
        local_node_id="alice-node", local_profile_factory=lambda: {},
        local_spr_other=b"", addresses=["/ip4/1.2.3.4/tcp/0"],
    )
    events = _run(sess.start())
    assert len(events) == 1 and events[0].type == SessionEventType.FAILED
    assert events[0].data["reason"] == "dial_failed"
    assert sess.state == SessionState.FAILED


# ── Responder happy path ────────────────────────────────────────────────────

def test_responder_on_hello_sends_hello_ack(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """Inbound HelloRequest in INIT → HELLO_ACK_SENT, HelloResponse on wire."""
    sess = Session(
        peer_id=alice_peer_id, role=SessionRole.RESPONDER, conn=bob_conn,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=bob_peer_id,
        local_node_id="bob-node", local_profile_factory=lambda: {},
        local_spr_other=b"",
    )
    hello = spb.HelloRequest(
        session_id=b"\x01" * 16,
        nonce_a=b"\xaa" * 32,
        protocol_versions=["1.0"],
        capabilities=0,
        client_id="test/1.0",
    )
    events = _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, hello.SerializeToString()))
    assert events == []
    assert sess.state == SessionState.HELLO_ACK_SENT
    assert sess.session_id == b"\x01" * 16
    assert sess.nonce_a == b"\xaa" * 32
    last = _last_sent(bob_conn, SESSION_HELLO_PROTOCOL)
    assert last is not None
    resp = _parse(spb.HelloResponse, last[2])
    assert resp.reason_code == spb.OK
    assert resp.chosen_version == "1.0"
    assert resp.sig_b == b"sig-bytes"   # what our fake signer returned


def test_responder_rejects_unsupported_version(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """HelloRequest with no overlap on protocol_versions → HelloResponse(VERSION_UNSUPPORTED) and FAILED."""
    sess = Session(
        peer_id=alice_peer_id, role=SessionRole.RESPONDER, conn=bob_conn,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=bob_peer_id,
        local_node_id="bob-node", local_profile_factory=lambda: {},
        local_spr_other=b"",
    )
    hello = spb.HelloRequest(
        session_id=b"\x01" * 16, nonce_a=b"\xaa" * 32,
        protocol_versions=["99.0"], client_id="test/99",
    )
    events = _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, hello.SerializeToString()))
    assert len(events) == 1 and events[0].type == SessionEventType.FAILED
    assert events[0].data["reason"] == "version_unsupported"
    last = _last_sent(bob_conn, SESSION_HELLO_PROTOCOL)
    assert last is not None
    resp = _parse(spb.HelloResponse, last[2])
    assert resp.reason_code == spb.VERSION_UNSUPPORTED


# ── Stage-1 replay defence (sig_b verification) ─────────────────────────────

def test_initiator_rejects_bad_sig_b(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """If the responder's sig_b doesn't verify, the initiator MUST fail before
    sending PROFILE_OFFER. This is the replay defence."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = Session(
        peer_id=bob_peer_id, role=SessionRole.INITIATOR, conn=alice_conn,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=alice_peer_id,
        local_node_id="alice-node", local_profile_factory=lambda: {},
        local_spr_other=b"", addresses=["/ip4/127.0.0.1/tcp/0"],
    )
    _run(sess.start())
    assert sess.state == SessionState.HELLO_SENT
    # Now inject a HelloResponse with sig_b that our fake verifier will reject.
    alice_p2p.verify_peer_sig_returns = False
    resp = spb.HelloResponse(
        chosen_version="1.0", capabilities=0, nonce_b=b"\xbb" * 32,
        sig_b=b"bogus", reason_code=spb.OK,
    )
    events = _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, resp.SerializeToString()))
    assert len(events) == 1 and events[0].type == SessionEventType.FAILED
    assert events[0].data["reason"] == "hello_ack_bad_sig_b"
    # PROFILE_OFFER MUST NOT have been sent.
    assert _last_sent(alice_conn, SESSION_PROFILE_PROTOCOL) is None


# ── Stage-2 + full initiator happy path ─────────────────────────────────────

def test_initiator_completes_through_profile_decision(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """Drive the initiator from start through ESTABLISHED."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = Session(
        peer_id=bob_peer_id, role=SessionRole.INITIATOR, conn=alice_conn,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=alice_peer_id,
        local_node_id="alice-node",
        local_profile_factory=lambda: {"node_id": "alice-node", "k": "v"},
        local_spr_other=b"alice-spr", addresses=["/ip4/127.0.0.1/tcp/0"],
    )
    _run(sess.start())
    # Stage-1 reply.
    ack = spb.HelloResponse(chosen_version="1.0", nonce_b=b"\xbb" * 32,
                            sig_b=b"sig", reason_code=spb.OK)
    _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, ack.SerializeToString()))
    assert sess.state == SessionState.PROFILE_SENT
    # Verify a PROFILE_OFFER got sent with our signed_peer_record attached.
    last = _last_sent(alice_conn, SESSION_PROFILE_PROTOCOL)
    assert last is not None
    offer = _parse(spb.ProfileOffer, last[2])
    assert offer.signed_peer_record == b"alice-spr"
    assert offer.sig_a == b"sig-bytes"

    # Stage-2 decision: accepted.
    import json
    decision = spb.ProfileDecision(
        accepted=True, profile_b=json.dumps({"node_id": "bob-node"}).encode(),
        reason_code=spb.OK, session_id=sess.session_id,
    )
    events = _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, decision.SerializeToString()))
    assert len(events) == 1 and events[0].type == SessionEventType.ESTABLISHED
    assert sess.state == SessionState.ESTABLISHED
    assert sess.remote_profile == {"node_id": "bob-node"}


def test_initiator_rejected_profile_fails(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """ProfileDecision(accepted=false) → FAILED with the responder's reason_code."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = Session(
        peer_id=bob_peer_id, role=SessionRole.INITIATOR, conn=alice_conn,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=alice_peer_id,
        local_node_id="alice-node", local_profile_factory=lambda: {},
        local_spr_other=b"", addresses=["/ip4/127.0.0.1/tcp/0"],
    )
    _run(sess.start())
    _run(sess.process_inbound(SESSION_HELLO_PROTOCOL,
        spb.HelloResponse(chosen_version="1.0", nonce_b=b"\xbb" * 32,
                          sig_b=b"sig", reason_code=spb.OK).SerializeToString()))
    decision = spb.ProfileDecision(accepted=False, reason_code=spb.NOT_AUTHORIZED)
    events = _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, decision.SerializeToString()))
    assert len(events) == 1 and events[0].type == SessionEventType.FAILED
    assert "NOT_AUTHORIZED" in events[0].data["reason"]


# ── Stage-2 responder: bad sig_a ────────────────────────────────────────────

def test_responder_rejects_bad_sig_a(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """ProfileOffer with sig_a that fails verification → reject + FAILED."""
    sess = Session(
        peer_id=alice_peer_id, role=SessionRole.RESPONDER, conn=bob_conn,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=bob_peer_id,
        local_node_id="bob-node", local_profile_factory=lambda: {},
        local_spr_other=b"",
    )
    _run(sess.process_inbound(SESSION_HELLO_PROTOCOL,
        spb.HelloRequest(session_id=b"\x01" * 16, nonce_a=b"\xaa" * 32,
                         protocol_versions=["1.0"]).SerializeToString()))
    assert sess.state == SessionState.HELLO_ACK_SENT
    # Now the offer arrives but sig_a doesn't verify.
    bob_p2p.verify_peer_sig_returns = False
    import json
    offer = spb.ProfileOffer(
        profile=json.dumps({}).encode(), sig_a=b"bad",
        signed_peer_record=b"spr", session_id=sess.session_id,
    )
    events = _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, offer.SerializeToString()))
    assert len(events) == 1 and events[0].type == SessionEventType.FAILED
    # We should have sent a PROFILE_DECISION(accepted=false) before failing.
    last = _last_sent(bob_conn, SESSION_PROFILE_PROTOCOL)
    assert last is not None
    decision = _parse(spb.ProfileDecision, last[2])
    assert decision.accepted is False
    assert decision.reason_code == spb.NOT_AUTHORIZED


# ── Timeouts ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("state,timeout_field", [
    (SessionState.CONNECTING,     "CONNECT"),
    (SessionState.HELLO_SENT,     "HELLO"),
    (SessionState.PROFILE_SENT,   "PROFILE"),
    (SessionState.HELLO_ACK_SENT, "PROFILE"),
])
def test_check_timeout_fires_failed_for_handshake_states(alice_conn, alice_peer_id, bob_peer_id, state, timeout_field):
    """Every blocking handshake state must FAIL on timeout."""
    sess = Session(
        peer_id=bob_peer_id, role=SessionRole.INITIATOR, conn=alice_conn,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=alice_peer_id,
        local_node_id="alice-node", local_profile_factory=lambda: {},
        local_spr_other=b"",
    )
    sess._set_state(state, 0.001)
    time.sleep(0.005)
    ev = sess.check_timeout()
    assert ev is not None
    assert ev.type == SessionEventType.FAILED
    assert sess.state == SessionState.FAILED


def test_check_timeout_returns_none_before_expiry(alice_conn, alice_peer_id, bob_peer_id):
    sess = Session(
        peer_id=bob_peer_id, role=SessionRole.INITIATOR, conn=alice_conn,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=alice_peer_id,
        local_node_id="alice-node", local_profile_factory=lambda: {},
        local_spr_other=b"",
    )
    sess._set_state(SessionState.HELLO_SENT, 10.0)
    assert sess.check_timeout() is None


# ── Disconnect / grace / resume ─────────────────────────────────────────────

def test_disconnect_from_established_enters_reconnecting(alice_conn, alice_peer_id, bob_peer_id):
    sess = Session(
        peer_id=bob_peer_id, role=SessionRole.INITIATOR, conn=alice_conn,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=alice_peer_id,
        local_node_id="alice-node", local_profile_factory=lambda: {},
        local_spr_other=b"",
    )
    sess._set_state(SessionState.ESTABLISHED, None)
    ev = sess.on_transport_disconnected()
    assert ev is not None and ev.type == SessionEventType.RECONNECTING
    assert sess.state == SessionState.RECONNECTING


def test_reconnect_within_grace_resumes(alice_conn, alice_peer_id, bob_peer_id):
    sess = Session(
        peer_id=bob_peer_id, role=SessionRole.INITIATOR, conn=alice_conn,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=alice_peer_id,
        local_node_id="alice-node", local_profile_factory=lambda: {},
        local_spr_other=b"",
    )
    sess._set_state(SessionState.RECONNECTING, 10.0)
    ev = sess.on_transport_reconnected()
    assert ev is not None and ev.type == SessionEventType.RESUMED
    assert sess.state == SessionState.ESTABLISHED


def test_grace_expiry_emits_lost(alice_conn, alice_peer_id, bob_peer_id):
    sess = Session(
        peer_id=bob_peer_id, role=SessionRole.INITIATOR, conn=alice_conn,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=alice_peer_id,
        local_node_id="alice-node", local_profile_factory=lambda: {},
        local_spr_other=b"",
    )
    sess._set_state(SessionState.RECONNECTING, 0.001)
    time.sleep(0.005)
    ev = sess.check_timeout()
    assert ev is not None and ev.type == SessionEventType.LOST
    assert sess.state == SessionState.FAILED


# ── Intentional close ───────────────────────────────────────────────────────

def test_close_sends_goaway_and_enters_closing(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    sess = Session(
        peer_id=bob_peer_id, role=SessionRole.INITIATOR, conn=alice_conn,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=alice_peer_id,
        local_node_id="alice-node", local_profile_factory=lambda: {},
        local_spr_other=b"",
    )
    sess._set_state(SessionState.ESTABLISHED, None)
    _run(sess.close(spb.WORLD_DISENGAGEMENT))
    assert sess.state == SessionState.CLOSING
    last = _last_sent(alice_conn, SESSION_CONTROL_PROTOCOL)
    assert last is not None
    ga = _parse(spb.Goaway, last[2])
    assert ga.reason == spb.WORLD_DISENGAGEMENT


def test_goaway_ack_finalizes_close(alice_conn, alice_peer_id, bob_peer_id):
    sess = Session(
        peer_id=bob_peer_id, role=SessionRole.INITIATOR, conn=alice_conn,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=alice_peer_id,
        local_node_id="alice-node", local_profile_factory=lambda: {},
        local_spr_other=b"",
    )
    sess._set_state(SessionState.CLOSING, 10.0)
    ack = spb.GoawayAck(received=True)
    events = _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, ack.SerializeToString()))
    assert len(events) == 1 and events[0].type == SessionEventType.CLOSED
    assert sess.state == SessionState.CLOSED


def test_remote_goaway_sends_ack_and_closes(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """Receiving a GOAWAY → send GOAWAY_ACK, emit CLOSED."""
    sess = Session(
        peer_id=alice_peer_id, role=SessionRole.RESPONDER, conn=bob_conn,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=bob_peer_id,
        local_node_id="bob-node", local_profile_factory=lambda: {},
        local_spr_other=b"",
    )
    sess._set_state(SessionState.ESTABLISHED, None)
    sess.session_id = b"\x07" * 16
    ga = spb.Goaway(reason=spb.GRACEFUL_SHUTDOWN, session_id=sess.session_id)
    events = _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, ga.SerializeToString()))
    assert len(events) == 1 and events[0].type == SessionEventType.CLOSED
    last = _last_sent(bob_conn, SESSION_CONTROL_PROTOCOL)
    assert last is not None
    ack = _parse(spb.GoawayAck, last[2])
    assert ack.received is True


# ── Keep-alive ──────────────────────────────────────────────────────────────

def test_tick_keepalive_schedules_ping_when_due(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """When ESTABLISHED and the keep-alive period has elapsed, tick must
    spawn a ping task that ends up calling p2p.ping_peer."""
    async def _runner():
        sess = Session(
            peer_id=bob_peer_id, role=SessionRole.INITIATOR, conn=alice_conn,
            p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=alice_peer_id,
            local_node_id="alice-node", local_profile_factory=lambda: {},
            local_spr_other=b"",
        )
        sess._set_state(SessionState.ESTABLISHED, None)
        sess._last_ping_at = time.monotonic() - 30.0   # well past 15s threshold
        ev = sess.tick_keepalive()
        assert ev is None
        # Let the create_task'd ping coroutine + the asyncio.to_thread it uses run.
        await asyncio.sleep(0.05)
    asyncio.get_event_loop().run_until_complete(_runner())
    assert alice_p2p.pinged == [(bob_peer_id, 5000)]


def test_keepalive_threshold_triggers_reconnecting(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """Two consecutive ping failures while ESTABLISHED → RECONNECTING."""
    sess = Session(
        peer_id=bob_peer_id, role=SessionRole.INITIATOR, conn=alice_conn,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=alice_peer_id,
        local_node_id="alice-node", local_profile_factory=lambda: {},
        local_spr_other=b"",
    )
    sess._set_state(SessionState.ESTABLISHED, None)
    sess._last_ping_at = time.monotonic()   # not due
    # Simulate two ping-fail results landing in the queue:
    sess._pending_keepalive_results.extend([None, None])
    ev = sess.tick_keepalive()
    assert ev is not None and ev.type == SessionEventType.RECONNECTING
    assert sess.state == SessionState.RECONNECTING


# ── Silent abort (tie-breaker support) ──────────────────────────────────────

def test_abort_silently_fails_without_event(alice_conn, alice_peer_id, bob_peer_id):
    sess = Session(
        peer_id=bob_peer_id, role=SessionRole.INITIATOR, conn=alice_conn,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=alice_peer_id,
        local_node_id="alice-node", local_profile_factory=lambda: {},
        local_spr_other=b"",
    )
    sess._set_state(SessionState.HELLO_SENT, 5.0)
    sess.abort_silently("simultaneous_open_yielded")
    assert sess.state == SessionState.FAILED
    # No event has been emitted — abort_silently returns None and produces nothing.
