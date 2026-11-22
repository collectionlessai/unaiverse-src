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

The Session FSM is protobuf-free: it speaks ``SessMsg`` value objects and the
FakeConn records the SessMsg sent on each protocol (entry[2] is a SessMsg).
"""

import time
import asyncio
import json
import pytest

from unaiverse.networking.session import (
    Session, SessionState, SessionRole, SessionEventType, SessMsg,
    SESSION_HELLO_PROTOCOL, SESSION_PROFILE_PROTOCOL,
    SESSION_WORLD_PROTOCOL, SESSION_CONTROL_PROTOCOL,
)
from unaiverse.networking.session import session as session_mod
from unaiverse.networking.session.sess_msg import (
    OK, VERSION_UNSUPPORTED, NOT_AUTHORIZED, GRACEFUL_SHUTDOWN, WORLD_DISENGAGEMENT,
    SESSION_UNKNOWN, TOKEN_EXPIRED,
)
from unaiverse.networking.node.connpool import NodeConn

pytestmark = pytest.mark.unit


# Helpers

def _last_sent(conn, protocol_id):
    """Return the last (peer, proto, SessMsg) sent on protocol_id from the
    FakeConn, or None. Session sends only through ``conn.send_session(...)``."""
    for entry in reversed(conn.sent):
        if entry[1] == protocol_id:
            return entry
    return None


def _run(coro):
    return asyncio.run(coro)


def _start(sess):
    """Design A drive for an initiator: Session.start() only launches the dial as a
    background task, so let it run one loop turn and reap it with tick_dial (HELLO on
    success) — all inside a single event loop. Returns the reaped events (usually [])."""
    async def _go():
        sess.start()
        await asyncio.sleep(0)
        return await sess.tick_dial()
    return _run(_go())


def _mk(peer_id, role, p2p, local_peer_id, local_node_id="local-node",
        local_profile_factory=lambda: {}, addresses=None,
        token_verifier=None, token_factory=None, remote_node_id_hint=None,
        dial_attempts=None, dial_budget_s=None):
    return Session(
        peer_id=peer_id, role=role, p2p=p2p,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=local_peer_id,
        local_node_id=local_node_id, local_profile_factory=local_profile_factory,
        addresses=addresses,
        token_verifier=token_verifier, token_factory=token_factory,
        remote_node_id_hint=remote_node_id_hint,
        dial_attempts=dial_attempts, dial_budget_s=dial_budget_s,
    )


async def _cancel_dial(sess):
    """Drop the background dial task started by _start_dial (we only assert the
    counter it bumps synchronously, not the dial outcome)."""
    if sess._dial_task is not None:
        sess._dial_task.cancel()
        try:
            await sess._dial_task
        except BaseException:
            pass
        sess._dial_task = None


def test_reconnect_redial_does_not_inflate_dial_attempts(alice_conn, alice_peer_id, bob_peer_id):
    """The first-handshake dial cap (_dial_attempts / _MAX_DIAL_ATTEMPTS) must not be
    bumped by reconnect re-dials, which are bounded by the grace window and counted on
    _reconnect_attempts. Bumping it here made the log read 'attempt 5/4, 6/4, ...'."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               addresses=["/ip4/127.0.0.1/tcp/0/p2p/" + bob_peer_id])

    async def _go():
        sess._set_state(SessionState.CONNECTING, 300.0)
        sess._start_dial()
        connecting_count = sess._dial_attempts   # bumped synchronously
        await _cancel_dial(sess)

        sess._set_state(SessionState.RECONNECTING, 45.0)
        sess._start_dial()
        reconnect_count = sess._dial_attempts
        await _cancel_dial(sess)
        return connecting_count, reconnect_count

    connecting_count, reconnect_count = _run(_go())
    assert connecting_count == 1     # the CONNECTING dial bumps the counter
    assert reconnect_count == 1      # the RECONNECTING re-dial leaves it alone


# Initiator happy path

def test_initiator_start_sends_hello(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """start() must dial, then emit HelloRequest on the hello protocol and
    park in HELLO_SENT awaiting HELLO_ACK."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               local_profile_factory=lambda: {"x": 1},
               addresses=["/ip4/127.0.0.1/tcp/0/p2p/" + bob_peer_id])
    events = _start(sess)
    assert events == []
    assert sess.state == SessionState.HELLO_SENT
    assert alice_conn.connect_calls == [["/ip4/127.0.0.1/tcp/0/p2p/" + bob_peer_id]]
    last = _last_sent(alice_conn, SESSION_HELLO_PROTOCOL)
    assert last is not None, "HelloRequest must be sent on the hello protocol"
    hello = last[2]
    assert hello.kind == SessMsg.KIND_HELLO_REQ
    assert hello.fields["session_id"] == sess.session_id
    assert len(hello.fields["nonce_a"]) == 32
    assert "1.0" in list(hello.fields["protocol_versions"])


def test_initiator_dial_failure_fails_after_max_attempts(alice_conn, alice_peer_id, bob_peer_id):
    """All 4 dial attempts return None → terminal FAILED event. Each failed attempt arms a
    time-based backoff gate, so we advance `now` past each gate to reach the next attempt."""
    alice_conn.connect_returns = (None, False)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               addresses=["/ip4/1.2.3.4/tcp/0"])

    async def _go():
        sess.start()                       # attempt 1 launched
        now = time.monotonic()
        for _ in range(20):
            await asyncio.sleep(0)          # let the dial task run
            evs = await sess.tick_dial(now)
            if evs:                         # [FAILED dial_failed] once attempts are exhausted
                return evs
            now += 60.0                     # jump past the backoff gate for the next attempt
        return []

    events = _run(_go())
    assert len(events) == 1 and events[0].type == SessionEventType.FAILED
    assert events[0].data["reason"] == "dial_failed"
    assert sess.state == SessionState.FAILED


def test_single_attempt_dial_fails_without_retry(alice_conn, alice_peer_id, bob_peer_id):
    """dial_attempts=1 (the world-join redeemer's policy): the first failed dial is
    terminal, no backoff retries — retrying belongs to the caller (the rejoin
    watchdog, which re-arms with a fresh grant)."""
    alice_conn.connect_returns = (None, False)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               addresses=["/ip4/1.2.3.4/tcp/0"], dial_attempts=1)

    async def _go():
        sess.start()
        now = time.monotonic()
        for _ in range(5):
            await asyncio.sleep(0)          # let the dial task run
            evs = await sess.tick_dial(now)
            if evs:
                return evs
            now += 60.0
        return []

    events = _run(_go())
    assert len(events) == 1 and events[0].type == SessionEventType.FAILED
    assert events[0].data["reason"] == "dial_failed"
    assert len(alice_conn.connect_calls) == 1   # exactly one dial, no retry


def test_dial_budget_bounds_connecting(alice_conn, alice_peer_id, bob_peer_id):
    """dial_budget_s overrides the CONNECTING backstop: a wedged dial times out at
    the per-Session budget (the grant window for a world redeemer), not at the
    module default."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               addresses=["/ip4/1.2.3.4/tcp/0"], dial_budget_s=0.001)

    async def _go():
        sess.start()
        await _cancel_dial(sess)   # leave CONNECTING with no dial outcome to reap
    _run(_go())

    assert sess.state == SessionState.CONNECTING
    time.sleep(0.005)
    ev = sess.check_timeout()
    assert ev is not None and ev.type == SessionEventType.FAILED
    assert ev.data["reason"] == "timeout:connecting"


# Responder happy path

def test_responder_on_hello_sends_hello_ack(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """Inbound HelloRequest in INIT → HELLO_ACK_SENT, HelloResponse on wire."""
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id, local_node_id="bob-node")
    hello = SessMsg(SessMsg.KIND_HELLO_REQ, {
        "session_id": b"\x01" * 16, "nonce_a": b"\xaa" * 32,
        "protocol_versions": ["1.0"], "capabilities": 0, "client_id": "test/1.0",
    })
    events = _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, hello))
    assert events == []
    assert sess.state == SessionState.HELLO_ACK_SENT
    assert sess.session_id == b"\x01" * 16
    assert sess.nonce_a == b"\xaa" * 32
    last = _last_sent(bob_conn, SESSION_HELLO_PROTOCOL)
    assert last is not None
    resp = last[2]
    assert resp.kind == SessMsg.KIND_HELLO_RESP
    assert resp.fields["reason_code"] == OK
    assert resp.fields["chosen_version"] == "1.0"
    assert resp.fields["sig_b"] == b"sig-bytes"   # what our fake signer returned


def test_responder_rejects_unsupported_version(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """HelloRequest with no overlap on protocol_versions → HelloResponse(VERSION_UNSUPPORTED) and FAILED."""
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id, local_node_id="bob-node")
    hello = SessMsg(SessMsg.KIND_HELLO_REQ, {
        "session_id": b"\x01" * 16, "nonce_a": b"\xaa" * 32,
        "protocol_versions": ["99.0"], "client_id": "test/99",
    })
    events = _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, hello))
    assert len(events) == 1 and events[0].type == SessionEventType.FAILED
    assert events[0].data["reason"] == "version_unsupported"
    last = _last_sent(bob_conn, SESSION_HELLO_PROTOCOL)
    assert last is not None
    assert last[2].fields["reason_code"] == VERSION_UNSUPPORTED


# Stage-1 replay defence (sig_b verification)

def test_initiator_ignores_bad_sig_b_then_accepts_good_ack(alice_conn, alice_p2p,
                                                           alice_peer_id, bob_peer_id):
    """An ACK whose sig_b doesn't verify is ignored (no PROFILE_OFFER, no terminal):
    HELLO_RESP has no session_id, so a stale-generation ACK must not kill the
    handshake the good ACK — possibly in the same drained batch — completes."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               addresses=["/ip4/127.0.0.1/tcp/0"])
    _start(sess)
    assert sess.state == SessionState.HELLO_SENT
    alice_p2p.verify_peer_sig_returns = False
    bogus = SessMsg(SessMsg.KIND_HELLO_RESP, {
        "chosen_version": "1.0", "capabilities": 0, "nonce_b": b"\xbb" * 32,
        "sig_b": b"bogus", "reason_code": OK,
    })
    events = _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, bogus))
    assert events == []
    assert sess.state == SessionState.HELLO_SENT
    assert _last_sent(alice_conn, SESSION_PROFILE_PROTOCOL) is None
    # The good ACK (verifying sig) still completes stage 1.
    alice_p2p.verify_peer_sig_returns = True
    good = SessMsg(SessMsg.KIND_HELLO_RESP, {
        "chosen_version": "1.0", "capabilities": 0, "nonce_b": b"\xcc" * 32,
        "sig_b": b"sig", "reason_code": OK,
    })
    events = _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, good))
    assert events == []
    assert sess.state == SessionState.PROFILE_SENT
    assert _last_sent(alice_conn, SESSION_PROFILE_PROTOCOL) is not None


# Stage-2 + full initiator happy path

def test_initiator_completes_through_profile_decision(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """Drive the initiator from start through ESTABLISHED."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               local_profile_factory=lambda: {"node_id": "alice-node", "k": "v"},
               addresses=["/ip4/127.0.0.1/tcp/0"])
    _start(sess)
    ack = SessMsg(SessMsg.KIND_HELLO_RESP, {
        "chosen_version": "1.0", "nonce_b": b"\xbb" * 32, "sig_b": b"sig", "reason_code": OK,
    })
    _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, ack))
    assert sess.state == SessionState.PROFILE_SENT
    last = _last_sent(alice_conn, SESSION_PROFILE_PROTOCOL)
    assert last is not None
    offer = last[2]
    assert offer.kind == SessMsg.KIND_PROFILE_OFFER
    assert offer.fields["sig_a"] == b"sig-bytes"
    # The offer no longer carries a signed_peer_record (binding via p2p_peers).
    assert "signed_peer_record" not in offer.fields

    decision = SessMsg(SessMsg.KIND_PROFILE_DECISION, {
        "accepted": True, "profile_b": json.dumps({"node_id": "bob-node"}).encode(),
        "reason_code": OK, "session_id": sess.session_id,
    })
    events = _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, decision))
    assert len(events) == 1 and events[0].type == SessionEventType.ESTABLISHED
    assert sess.state == SessionState.ESTABLISHED
    assert sess.remote_profile == {"node_id": "bob-node"}


def test_initiator_rejected_profile_fails(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """ProfileDecision(accepted=false) → FAILED with the responder's reason_code."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               addresses=["/ip4/127.0.0.1/tcp/0"])
    _start(sess)
    _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, SessMsg(SessMsg.KIND_HELLO_RESP, {
        "chosen_version": "1.0", "nonce_b": b"\xbb" * 32, "sig_b": b"sig", "reason_code": OK,
    })))
    decision = SessMsg(SessMsg.KIND_PROFILE_DECISION, {"accepted": False, "reason_code": NOT_AUTHORIZED})
    events = _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, decision))
    assert len(events) == 1 and events[0].type == SessionEventType.FAILED
    assert "NOT_AUTHORIZED" in events[0].data["reason"]


# Stage-2 responder: bad sig_a

def test_responder_rejects_bad_sig_a(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """ProfileOffer with sig_a that fails verification → reject + FAILED."""
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id, local_node_id="bob-node")
    _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, SessMsg(SessMsg.KIND_HELLO_REQ, {
        "session_id": b"\x01" * 16, "nonce_a": b"\xaa" * 32, "protocol_versions": ["1.0"],
    })))
    assert sess.state == SessionState.HELLO_ACK_SENT
    bob_p2p.verify_peer_sig_returns = False
    offer = SessMsg(SessMsg.KIND_PROFILE_OFFER, {
        "profile": json.dumps({}).encode(), "sig_a": b"bad",
        "session_id": sess.session_id,
    })
    events = _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, offer))
    assert len(events) == 1 and events[0].type == SessionEventType.FAILED
    last = _last_sent(bob_conn, SESSION_PROFILE_PROTOCOL)
    assert last is not None
    decision = last[2]
    assert decision.kind == SessMsg.KIND_PROFILE_DECISION
    assert decision.fields["accepted"] is False
    assert decision.fields["reason_code"] == NOT_AUTHORIZED


# Stage-2 responder: RT token consumption

def _drive_responder_to_offer_wait(sess):
    """Push a HelloRequest so the responder parks in HELLO_ACK_SENT, ready for an offer."""
    _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, SessMsg(SessMsg.KIND_HELLO_REQ, {
        "session_id": b"\x01" * 16, "nonce_a": b"\xaa" * 32, "protocol_versions": ["1.0"],
    })))
    assert sess.state == SessionState.HELLO_ACK_SENT


def test_initiator_offer_carries_token(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """With a token_factory injected, the PROFILE_OFFER carries the node token."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               addresses=["/ip4/127.0.0.1/tcp/0"], token_factory=lambda: "tok-abc")
    _start(sess)
    _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, SessMsg(SessMsg.KIND_HELLO_RESP, {
        "chosen_version": "1.0", "nonce_b": b"\xbb" * 32, "sig_b": b"sig", "reason_code": OK,
    })))
    offer = _last_sent(alice_conn, SESSION_PROFILE_PROTOCOL)[2]
    assert offer.kind == SessMsg.KIND_PROFILE_OFFER
    assert offer.fields["token"] == "tok-abc"


def test_initiator_offer_token_empty_without_factory(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """No token_factory → the offer carries an empty token (the v1 / no-RT path)."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               addresses=["/ip4/127.0.0.1/tcp/0"])
    _start(sess)
    _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, SessMsg(SessMsg.KIND_HELLO_RESP, {
        "chosen_version": "1.0", "nonce_b": b"\xbb" * 32, "sig_b": b"sig", "reason_code": OK,
    })))
    assert _last_sent(alice_conn, SESSION_PROFILE_PROTOCOL)[2].fields["token"] == ""


def test_responder_rejects_invalid_token(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """A configured verifier that rejects the token → reject(NOT_AUTHORIZED) + FAILED,
    before the application decision is ever reached."""
    async def _verifier(token, peer_id):
        return (None, None, (), 0, None)   # verification failure
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id,
               local_node_id="bob-node", token_verifier=_verifier)
    _drive_responder_to_offer_wait(sess)
    offer = SessMsg(SessMsg.KIND_PROFILE_OFFER, {
        "profile": json.dumps({"node_id": "claimed"}).encode(), "sig_a": b"ok",
        "session_id": sess.session_id, "token": "bad-tok",
    })
    events = _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, offer))
    assert len(events) == 1 and events[0].type == SessionEventType.FAILED
    assert "bad_token" in events[0].data["reason"]
    decision = _last_sent(bob_conn, SESSION_PROFILE_PROTOCOL)[2]
    assert decision.fields["accepted"] is False
    assert decision.fields["reason_code"] == NOT_AUTHORIZED


def test_responder_valid_token_overrides_profile_node_id(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """A valid token is verified against the transport peer-id, and the root-attested
    node_id wins over the self-asserted profile node_id; the handshake then reaches
    ESTABLISHED (agent host, no decision hook)."""
    seen = {}
    async def _verifier(token, peer_id):
        seen["token"], seen["peer_id"] = token, peer_id
        return ("root-attested-node", "cv-hash", (peer_id,), 9999999999, None)
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id,
               local_node_id="bob-node", token_verifier=_verifier)
    _drive_responder_to_offer_wait(sess)
    offer = SessMsg(SessMsg.KIND_PROFILE_OFFER, {
        "profile": json.dumps({"node_id": "self-claimed"}).encode(), "sig_a": b"ok",
        "session_id": sess.session_id, "token": "good-tok",
    })
    events = _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, offer))
    assert seen == {"token": "good-tok", "peer_id": alice_peer_id}
    assert sess.remote_node_id_hint == "root-attested-node"
    # The peer's attested token expiry / cv_hash are captured on the Session.
    assert sess.remote_token_exp == 9999999999
    assert sess.remote_cv_hash == "cv-hash"
    assert len(events) == 1 and events[0].type == SessionEventType.ESTABLISHED


# Stage-2: RT mutual auth (responder token verified by the initiator)

def test_responder_decision_carries_token(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """RT mutual: the responder's PROFILE_DECISION carries its own token."""
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id,
               local_node_id="bob-node", token_factory=lambda: "bob-tok")
    _drive_responder_to_offer_wait(sess)
    offer = SessMsg(SessMsg.KIND_PROFILE_OFFER, {
        "profile": json.dumps({"node_id": "alice"}).encode(), "sig_a": b"ok",
        "session_id": sess.session_id, "token": "",
    })
    events = _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, offer))
    assert len(events) == 1 and events[0].type == SessionEventType.ESTABLISHED
    dec = _last_sent(bob_conn, SESSION_PROFILE_PROTOCOL)[2]
    assert dec.kind == SessMsg.KIND_PROFILE_DECISION
    assert dec.fields["token"] == "bob-tok"


def _drive_initiator_to_decision(sess, p2p):
    """Start the initiator and feed a valid HELLO_ACK so it parks in PROFILE_SENT."""
    _start(sess)
    _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, SessMsg(SessMsg.KIND_HELLO_RESP, {
        "chosen_version": "1.0", "nonce_b": b"\xbb" * 32, "sig_b": b"sig", "reason_code": OK,
    })))


def test_initiator_verifies_responder_token(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """RT mutual: the initiator verifies the responder's token from PROFILE_DECISION
    against the responder's peer-id, adopts the attested node-id, and completes."""
    seen = {}
    async def _verifier(token, peer_id):
        seen["token"], seen["peer_id"] = token, peer_id
        return ("bob-attested", "cv", (peer_id,), 9999999999, None)
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               addresses=["/ip4/127.0.0.1/tcp/0"], token_verifier=_verifier)
    _drive_initiator_to_decision(sess, alice_p2p)
    decision = SessMsg(SessMsg.KIND_PROFILE_DECISION, {
        "accepted": True, "profile_b": json.dumps({"node_id": "bob"}).encode(),
        "reason_code": OK, "session_id": sess.session_id, "token": "bob-tok",
    })
    events = _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, decision))
    assert seen == {"token": "bob-tok", "peer_id": bob_peer_id}
    assert sess.remote_node_id_hint == "bob-attested"
    assert len(events) == 1 and events[0].type == SessionEventType.ESTABLISHED


def test_initiator_rejects_bad_responder_token(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """RT mutual: an invalid responder token fails the initiator's handshake."""
    async def _verifier(token, peer_id):
        return (None, None, (), 0, None)
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               addresses=["/ip4/127.0.0.1/tcp/0"], token_verifier=_verifier)
    _drive_initiator_to_decision(sess, alice_p2p)
    decision = SessMsg(SessMsg.KIND_PROFILE_DECISION, {
        "accepted": True, "profile_b": json.dumps({"node_id": "bob"}).encode(),
        "reason_code": OK, "session_id": sess.session_id, "token": "bad",
    })
    events = _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, decision))
    assert len(events) == 1 and events[0].type == SessionEventType.FAILED
    assert "bad_token" in events[0].data["reason"]


def test_initiator_rejects_responder_node_id_mismatch(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """RT mutual: when the initiator expected a specific node-id at this peer and the
    responder's token attests a different one, the handshake fails."""
    async def _verifier(token, peer_id):
        return ("someone-else", "cv", (peer_id,), 9999999999, None)
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               addresses=["/ip4/127.0.0.1/tcp/0"], token_verifier=_verifier,
               remote_node_id_hint="expected-bob")
    _drive_initiator_to_decision(sess, alice_p2p)
    decision = SessMsg(SessMsg.KIND_PROFILE_DECISION, {
        "accepted": True, "profile_b": json.dumps({"node_id": "bob"}).encode(),
        "reason_code": OK, "session_id": sess.session_id, "token": "tok",
    })
    events = _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, decision))
    assert len(events) == 1 and events[0].type == SessionEventType.FAILED
    assert "token_node_id_mismatch" in events[0].data["reason"]


# Timeouts

@pytest.mark.parametrize("state,timeout_field", [
    (SessionState.CONNECTING,     "CONNECT"),
    (SessionState.HELLO_SENT,     "HELLO"),
    (SessionState.PROFILE_SENT,   "PROFILE"),
    (SessionState.HELLO_ACK_SENT, "PROFILE"),
])
def test_check_timeout_fires_failed_for_handshake_states(alice_conn, alice_peer_id, bob_peer_id, state, timeout_field):
    """Every blocking handshake state must FAIL on timeout."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(state, 0.001)
    time.sleep(0.005)
    ev = sess.check_timeout()
    assert ev is not None
    assert ev.type == SessionEventType.FAILED
    assert sess.state == SessionState.FAILED


def test_check_timeout_returns_none_before_expiry(alice_conn, alice_peer_id, bob_peer_id):
    """check_timeout() must not fire while the state deadline is still in the future."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(SessionState.HELLO_SENT, 10.0)
    assert sess.check_timeout() is None


# Disconnect / grace / resume

def test_disconnect_from_established_enters_reconnecting(alice_conn, alice_peer_id, bob_peer_id):
    """A transport disconnect from ESTABLISHED emits RECONNECTING and moves to RECONNECTING."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(SessionState.ESTABLISHED, None)
    ev = sess.on_transport_disconnected()
    assert ev is not None and ev.type == SessionEventType.RECONNECTING
    assert sess.state == SessionState.RECONNECTING


def test_transport_reconnect_does_not_blindly_resume(alice_conn, alice_peer_id, bob_peer_id):
    """A transport-diff reconnect must NOT flip straight to ESTABLISHED (that is the
    /ipfs/ping false-resume bug). It only arms the RESUME handshake; the session
    stays RECONNECTING until a valid RESUME_ACK arrives."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(SessionState.RECONNECTING, 10.0)
    ev = sess.on_transport_reconnected()
    assert ev is None
    assert sess.state == SessionState.RECONNECTING
    assert sess._resume_due is True


def test_grace_expiry_emits_lost(alice_conn, alice_peer_id, bob_peer_id):
    """The grace timer expiring while RECONNECTING emits LOST and lands in FAILED."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(SessionState.RECONNECTING, 0.001)
    time.sleep(0.005)
    ev = sess.check_timeout()
    assert ev is not None and ev.type == SessionEventType.LOST
    assert sess.state == SessionState.FAILED


# Intentional close

def test_close_sends_goaway_and_enters_closing(alice_conn, alice_p2p, alice_peer_id, bob_peer_id):
    """close() from ESTABLISHED sends a GOAWAY on the control protocol carrying the given
    reason and moves the session to CLOSING."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(SessionState.ESTABLISHED, None)
    _run(sess.close(WORLD_DISENGAGEMENT))
    assert sess.state == SessionState.CLOSING
    last = _last_sent(alice_conn, SESSION_CONTROL_PROTOCOL)
    assert last is not None
    ga = last[2]
    assert ga.kind == SessMsg.KIND_GOAWAY
    assert ga.fields["reason"] == WORLD_DISENGAGEMENT


def test_goaway_ack_finalizes_close(alice_conn, alice_peer_id, bob_peer_id):
    """Receiving a GOAWAY_ACK while CLOSING finalizes the close: CLOSED event, state CLOSED."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(SessionState.CLOSING, 10.0)
    ack = SessMsg(SessMsg.KIND_GOAWAY_ACK, {"received": True})
    events = _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, ack))
    assert len(events) == 1 and events[0].type == SessionEventType.CLOSED
    assert sess.state == SessionState.CLOSED


def test_remote_goaway_sends_ack_and_closes(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """Receiving a GOAWAY → send GOAWAY_ACK, emit CLOSED."""
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id, local_node_id="bob-node")
    sess._set_state(SessionState.ESTABLISHED, None)
    sess.session_id = b"\x07" * 16
    ga = SessMsg(SessMsg.KIND_GOAWAY, {"reason": GRACEFUL_SHUTDOWN, "session_id": sess.session_id})
    events = _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, ga))
    assert len(events) == 1 and events[0].type == SessionEventType.CLOSED
    last = _last_sent(bob_conn, SESSION_CONTROL_PROTOCOL)
    assert last is not None
    assert last[2].kind == SessMsg.KIND_GOAWAY_ACK
    assert last[2].fields["received"] is True


# Session-level keep-alive (PING/PONG over the control protocol)

def test_tick_keepalive_sends_session_ping_when_due(alice_conn, alice_peer_id, bob_peer_id):
    """When ESTABLISHED and the keep-alive period has elapsed, tick must send a
    session-level PING on the CONTROL protocol (NOT /ipfs/ping) bound to session_id."""
    async def _runner():
        sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
        sess._set_state(SessionState.ESTABLISHED, None)
        sess.session_id = b"\x01" * 16
        sess._last_ping_at = time.monotonic() - 30.0   # well past 15s threshold
        ev = await sess.tick_keepalive()
        assert ev is None
        assert sess._ping_in_flight is True
        # The PING is sent inline by the tick: no extra loop yield needed.
        return sess
    sess = asyncio.run(_runner())
    last = _last_sent(alice_conn, SESSION_CONTROL_PROTOCOL)
    assert last is not None and last[2].kind == SessMsg.KIND_PING
    assert last[2].fields["session_id"] == sess.session_id
    # No /ipfs/ping was used.
    assert alice_conn.p2p_name_to_p2p[NodeConn.P2P_PUBLIC].pinged == []


def test_pong_ok_clears_failures(alice_conn, alice_peer_id, bob_peer_id):
    """A matching PONG(ok=true) clears the in-flight ping and resets the consecutive-failure
    counter to zero, leaving the session ESTABLISHED with no event."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(SessionState.ESTABLISHED, None)
    sess.session_id = b"\x02" * 16
    sess._ping_in_flight = True
    sess._ping_nonce = b"nonce-123"
    sess._consecutive_ping_failures = 1
    pong = SessMsg(SessMsg.KIND_PONG, {"session_id": sess.session_id, "nonce": b"nonce-123", "ok": True})
    events = _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, pong))
    assert events == []
    assert sess._ping_in_flight is False
    assert sess._consecutive_ping_failures == 0
    assert sess.state == SessionState.ESTABLISHED


def test_pong_not_ok_starts_reconnect(alice_conn, alice_peer_id, bob_peer_id):
    """A PONG ok=false (peer holds no Session) is a hard liveness failure → RECONNECTING."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(SessionState.ESTABLISHED, None)
    sess.session_id = b"\x02" * 16
    sess._ping_in_flight = True
    sess._ping_nonce = b"nonce-123"
    pong = SessMsg(SessMsg.KIND_PONG, {"session_id": sess.session_id, "nonce": b"nonce-123", "ok": False})
    events = _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, pong))
    assert len(events) == 1 and events[0].type == SessionEventType.RECONNECTING
    assert sess.state == SessionState.RECONNECTING
    # An initiator has the RESUME path ahead: it keeps the whole reconnect window.
    assert sess._timeout_s == session_mod._TIMEOUT_RECONNECT_S
    assert sess._reconnect_substep == "redial"


def _responder_told_its_session_is_gone(bob_conn, bob_peer_id, alice_peer_id):
    """An ESTABLISHED responder that has just received PONG ok=false for its ping.
    Returns the Session and the events of that PONG."""
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id)
    sess._set_state(SessionState.ESTABLISHED, None)
    sess.session_id = b"\x02" * 16
    sess._ping_in_flight = True
    sess._ping_nonce = b"nonce-123"
    pong = SessMsg(SessMsg.KIND_PONG, {"session_id": sess.session_id, "nonce": b"nonce-123", "ok": False})
    events = _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, pong))
    return sess, events


def test_pong_not_ok_on_responder_waits_a_short_window_then_is_lost(bob_conn, bob_peer_id, alice_peer_id):
    """A responder cannot dial, and no RESUME comes from a peer that has just said it holds
    no Session: it must not ride the whole reconnect window. It goes RECONNECTING like
    any other loss (same event), waits a short window of its own for a fresh HELLO or a
    late GOAWAY, and is then LOST with a reason that says what happened."""
    sess, events = _responder_told_its_session_is_gone(bob_conn, bob_peer_id, alice_peer_id)
    assert len(events) == 1 and events[0].type == SessionEventType.RECONNECTING
    assert events[0].data == {"cause": "transport_disconnected"}
    assert sess.state == SessionState.RECONNECTING

    window = session_mod._TIMEOUT_PEER_LOST_SESSION_S
    assert window < session_mod._TIMEOUT_RECONNECT_S
    entered = sess._state_entered_at
    assert sess.check_timeout(entered + window - 0.1) is None
    ev = sess.check_timeout(entered + window + 0.1)
    assert ev is not None and ev.type == SessionEventType.LOST
    assert ev.data["reason"] == "peer_lost_session"
    assert sess.state == SessionState.FAILED


def test_transport_loss_does_not_restart_the_window_of_a_lost_session(bob_conn, bob_peer_id, alice_peer_id):
    """The transport usually drops right after the peer said the Session is gone. That
    adds nothing to wait for: the short window keeps its deadline (an ordinary reconnect
    window is restarted by a new transport loss)."""
    sess, _ = _responder_told_its_session_is_gone(bob_conn, bob_peer_id, alice_peer_id)
    window = session_mod._TIMEOUT_PEER_LOST_SESSION_S
    entered = sess._state_entered_at

    assert sess.on_transport_disconnected() is None
    assert sess._state_entered_at == entered
    assert sess._timeout_s == window
    ev = sess.check_timeout(entered + window + 0.1)
    assert ev is not None and ev.data["reason"] == "peer_lost_session"


def _reconnecting_for(sess, seconds):
    """Lose the link of an ESTABLISHED Session and move the time it entered RECONNECTING,
    and the start of its current window, `seconds` into the past."""
    sess._set_state(SessionState.ESTABLISHED, None)
    assert sess.on_transport_disconnected().type == SessionEventType.RECONNECTING
    sess._reconnecting_since -= seconds
    sess._state_entered_at -= seconds


@pytest.mark.parametrize("window", [45.0, 120.0])
def test_reconnect_window_starts_over_within_twice_the_window(alice_conn, alice_peer_id, bob_peer_id,
                                                              monkeypatch, window):
    """A transport lost again starts the window over, but the whole time in RECONNECTING
    never passes twice the window: 90 s with the window of Python, 240 s with the 120 s
    the browser sets. The log says when the ceiling cut the window short."""
    lines = []
    monkeypatch.setattr(session_mod.log, "session", lambda msg, *a, **k: lines.append(msg))
    monkeypatch.setattr(session_mod, "_TIMEOUT_RECONNECT_S", window)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)

    _reconnecting_for(sess, 0.25 * window)
    assert sess.on_transport_disconnected() is None
    assert sess._timeout_s == window                 # far from the ceiling: a whole window
    assert "which starts over" in lines[-1] and "cut short" not in lines[-1]

    _reconnecting_for(sess, 1.75 * window)
    assert sess.on_transport_disconnected() is None
    restarted = sess._state_entered_at
    assert sess._timeout_s == pytest.approx(0.25 * window, abs=0.5)
    assert "cut short" in lines[-1] and f"{2 * window}s in RECONNECTING at most" in lines[-1]
    assert sess.check_timeout(restarted + 0.25 * window - 1.0) is None
    ev = sess.check_timeout(restarted + 0.25 * window + 1.0)
    assert ev is not None and ev.type == SessionEventType.LOST
    assert ev.data["reason"] == "grace_period_expired"


def test_reconnect_window_past_its_ceiling_does_not_start_over(alice_conn, alice_peer_id, bob_peer_id):
    """A loss that comes when the ceiling has passed leaves nothing to wait for."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    window = session_mod._TIMEOUT_RECONNECT_S
    _reconnecting_for(sess, 2 * window + 5.0)
    sess.on_transport_disconnected()
    assert sess._timeout_s == 0.0
    ev = sess.check_timeout(sess._state_entered_at + 0.1)
    assert ev is not None and ev.type == SessionEventType.LOST


def test_each_reconnect_has_a_ceiling_of_its_own(alice_conn, alice_peer_id, bob_peer_id):
    """A Session that came back and lost its link again counts from the new loss."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    window = session_mod._TIMEOUT_RECONNECT_S
    _reconnecting_for(sess, 1.9 * window)
    sess._resume_established()
    _reconnecting_for(sess, 0.0)
    sess.on_transport_disconnected()
    assert sess._timeout_s == window


def test_goaway_inside_the_window_of_a_lost_session_closes(bob_conn, bob_peer_id, alice_peer_id):
    """The negative PONG may have overtaken the GOAWAY of a peer that is closing: the
    GOAWAY that follows must still end the Session as CLOSED, not leave it to LOST."""
    sess, _ = _responder_told_its_session_is_gone(bob_conn, bob_peer_id, alice_peer_id)
    ga = SessMsg(SessMsg.KIND_GOAWAY, {"reason": GRACEFUL_SHUTDOWN, "session_id": sess.session_id})
    events = _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, ga))
    assert len(events) == 1 and events[0].type == SessionEventType.CLOSED
    assert events[0].data["reason"] == "remote_goaway:GRACEFUL_SHUTDOWN"
    assert sess.state == SessionState.CLOSED
    assert _last_sent(bob_conn, SESSION_CONTROL_PROTOCOL)[2].kind == SessMsg.KIND_GOAWAY_ACK


def test_resume_inside_the_window_of_a_lost_session_resumes(bob_conn, bob_peer_id, alice_peer_id):
    """A RESUME for the id we hold is still honoured inside the short window, and the
    Session it brings back is an ordinary one: its next transport loss gets the whole
    reconnect window and the usual reason."""
    sess, _ = _responder_told_its_session_is_gone(bob_conn, bob_peer_id, alice_peer_id)
    resume = SessMsg(SessMsg.KIND_RESUME, {"session_id": sess.session_id, "nonce": b"r1"})
    events = _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, resume))
    assert len(events) == 1 and events[0].type == SessionEventType.RESUMED
    assert sess.state == SessionState.ESTABLISHED
    last = _last_sent(bob_conn, SESSION_CONTROL_PROTOCOL)
    assert last[2].kind == SessMsg.KIND_RESUME_ACK and last[2].fields["accepted"] is True

    ev = sess.on_transport_disconnected()
    assert ev is not None and ev.type == SessionEventType.RECONNECTING
    assert sess._timeout_s == session_mod._TIMEOUT_RECONNECT_S
    ev = sess.check_timeout(sess._state_entered_at + session_mod._TIMEOUT_RECONNECT_S + 0.1)
    assert ev is not None and ev.data["reason"] == "grace_period_expired"


# The probe of a responder that is reconnecting

def _reconnecting_responder(bob_conn, bob_peer_id, alice_peer_id):
    """A responder that has just lost its transport while ESTABLISHED."""
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id)
    sess._set_state(SessionState.ESTABLISHED, None)
    sess.session_id = b"\x03" * 16
    sess._last_ping_at = time.monotonic()   # its keep-alive ping is not due
    ev = sess.on_transport_disconnected()
    assert ev is not None and ev.type == SessionEventType.RECONNECTING
    return sess


def _pings(conn):
    return [e[2] for e in conn.sent
            if e[1] == SESSION_CONTROL_PROTOCOL and e[2].kind == SessMsg.KIND_PING]


async def _tick_and_let_the_probe_run(sess, now):
    """One keep-alive tick, then one loop turn for the probe task it may have started."""
    ev = await sess.tick_keepalive(now)
    await asyncio.sleep(0)
    return ev


def _answer_the_probe(sess, ok):
    pong = SessMsg(SessMsg.KIND_PONG, {"session_id": sess.session_id,
                                       "nonce": sess._ping_nonce, "ok": ok})
    return _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, pong))


def test_reconnecting_responder_probes_at_once_and_at_the_keepalive_period(bob_conn, bob_peer_id,
                                                                          alice_peer_id):
    """A responder cannot dial, so while it is reconnecting it asks the peer: a PING on its
    first tick in the state, bound to the session id, then one per keep-alive period. A
    probe that gets no answer is not a ping miss and changes nothing."""
    sess = _reconnecting_responder(bob_conn, bob_peer_id, alice_peer_id)
    entered, window = sess._state_entered_at, sess._timeout_s
    period, rtt = session_mod._KEEPALIVE_PERIOD_S, session_mod._PING_RTT_TIMEOUT_S
    now = time.monotonic()

    assert _run(_tick_and_let_the_probe_run(sess, now)) is None
    assert len(_pings(bob_conn)) == 1
    assert _pings(bob_conn)[0].fields["session_id"] == sess.session_id
    first_nonce = sess._ping_nonce

    assert _run(_tick_and_let_the_probe_run(sess, now + rtt + 0.1)) is None
    assert sess._ping_in_flight is False and sess._consecutive_ping_failures == 0
    assert _run(_tick_and_let_the_probe_run(sess, now + period - 0.1)) is None
    assert len(_pings(bob_conn)) == 1

    # Past the period and not on it: (now + period) - now can come out a hair below period.
    assert _run(_tick_and_let_the_probe_run(sess, now + period + 0.1)) is None
    assert len(_pings(bob_conn)) == 2
    assert _pings(bob_conn)[1].fields["nonce"] != first_nonce

    assert sess.state == SessionState.RECONNECTING
    assert (sess._state_entered_at, sess._timeout_s) == (entered, window)


def test_probe_answered_not_ok_starts_the_short_window(bob_conn, bob_peer_id, alice_peer_id):
    """PONG ok=false to the probe: the peer holds no Session for our id, so the wait is
    cut to the short window, counted from the answer, and ends LOST with the reason of a
    peer that lost the Session. The layers above have already seen RECONNECTING, so no
    event comes with the answer, and no further probe is sent."""
    sess = _reconnecting_responder(bob_conn, bob_peer_id, alice_peer_id)
    now = time.monotonic()
    _run(_tick_and_let_the_probe_run(sess, now))

    assert _answer_the_probe(sess, ok=False) == []
    assert sess.state == SessionState.RECONNECTING
    window = session_mod._TIMEOUT_PEER_LOST_SESSION_S
    assert sess._timeout_s == window

    _run(_tick_and_let_the_probe_run(sess, now + session_mod._KEEPALIVE_PERIOD_S))
    assert len(_pings(bob_conn)) == 1

    entered = sess._state_entered_at
    assert sess.check_timeout(entered + window - 0.1) is None
    ev = sess.check_timeout(entered + window + 0.1)
    assert ev is not None and ev.type == SessionEventType.LOST
    assert ev.data["reason"] == "peer_lost_session"


def test_probe_answered_ok_resumes_the_responder(bob_conn, bob_peer_id, alice_peer_id):
    """PONG ok=true to the probe: the peer still holds the Session, names our session id
    and echoes the nonce of the probe, over a transport that has just worked both ways.
    That is what an accepted RESUME proves, so the responder is back the same way: RESUMED
    with the profile it kept, liveness reset. It is the only way back when the initiator
    never saw the loss, since it then sends no RESUME. The Session that comes back is an
    ordinary one: its next loss gets the whole window."""
    sess = _reconnecting_responder(bob_conn, bob_peer_id, alice_peer_id)
    sess.remote_profile = {"node_id": "alice"}
    _run(_tick_and_let_the_probe_run(sess, time.monotonic()))

    events = _answer_the_probe(sess, ok=True)
    assert len(events) == 1 and events[0].type == SessionEventType.RESUMED
    assert events[0].data == {"profile": {"node_id": "alice"}, "peer_capabilities": 0}
    assert sess.state == SessionState.ESTABLISHED and sess._timeout_s is None
    assert sess._ping_in_flight is False and sess._consecutive_ping_failures == 0
    assert sess.check_timeout(time.monotonic() + session_mod._TIMEOUT_RECONNECT_S + 1.0) is None

    ev = sess.on_transport_disconnected()
    assert ev is not None and ev.type == SessionEventType.RECONNECTING
    assert sess._timeout_s == session_mod._TIMEOUT_RECONNECT_S


def test_probe_answer_that_does_not_match_the_probe_in_flight_does_not_resume(bob_conn, bob_peer_id,
                                                                             alice_peer_id):
    """Only the answer to the probe in flight counts: one with another nonce (the answer
    to an older ping), one that names another Session, and one that arrives when no probe
    is in flight leave the responder where it is."""
    sess = _reconnecting_responder(bob_conn, bob_peer_id, alice_peer_id)
    _run(_tick_and_let_the_probe_run(sess, time.monotonic()))

    def answer(session_id, nonce):
        pong = SessMsg(SessMsg.KIND_PONG, {"session_id": session_id, "nonce": nonce, "ok": True})
        return _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, pong))

    assert answer(sess.session_id, b"another-nonce") == []
    assert sess.state == SessionState.RECONNECTING and sess._ping_in_flight is True
    assert answer(b"\x99" * 16, sess._ping_nonce) == []
    assert sess.state == SessionState.RECONNECTING
    # That answer used the probe up: the right one, now, answers nothing in flight.
    assert sess._ping_in_flight is False
    assert answer(sess.session_id, sess._ping_nonce) == []
    assert sess.state == SessionState.RECONNECTING
    assert sess._timeout_s == session_mod._TIMEOUT_RECONNECT_S


def test_positive_pong_resumes_only_a_responder_that_is_probing(alice_conn, bob_conn, alice_peer_id,
                                                               bob_peer_id):
    """The way back by a PONG belongs to the probe. An initiator that is reconnecting comes
    back by its RESUME alone, and a responder told that its Session is gone stays in its
    short window: neither sends a ping in that state, and should one ever be in flight
    there, its answer must not resume them."""
    initiator = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    initiator._set_state(SessionState.ESTABLISHED, None)
    initiator.on_transport_disconnected()
    told, _ = _responder_told_its_session_is_gone(bob_conn, bob_peer_id, alice_peer_id)
    for sess in (initiator, told):
        window = sess._timeout_s
        sess._ping_in_flight, sess._ping_nonce = True, b"forced"
        assert _answer_the_probe(sess, ok=True) == []
        assert sess.state == SessionState.RECONNECTING and sess._timeout_s == window


def test_transport_reported_back_makes_a_probing_responder_ask_at_once(bob_conn, bob_peer_id,
                                                                      alice_peer_id):
    """A connection that comes back, made by the initiator or by the dial of an earlier
    probe, is what a probe needs: the responder asks on its next tick and does not wait
    for the keep-alive period. One probe at a time: with one in flight the new one waits
    for its answer or for the end of its RTT window, and one signal asks for one probe."""
    sess = _reconnecting_responder(bob_conn, bob_peer_id, alice_peer_id)
    period, rtt = session_mod._KEEPALIVE_PERIOD_S, session_mod._PING_RTT_TIMEOUT_S
    now = time.monotonic()
    _run(_tick_and_let_the_probe_run(sess, now))
    assert len(_pings(bob_conn)) == 1

    # The first probe is in flight: the signal does not send a second one over it.
    assert sess.on_transport_reconnected() is None
    _run(_tick_and_let_the_probe_run(sess, now + 1.0))
    assert len(_pings(bob_conn)) == 1
    # Its RTT window ends with no answer, well before the period: the probe asked for by
    # the signal leaves then, and no other follows it.
    assert rtt + 0.2 < period
    _run(_tick_and_let_the_probe_run(sess, now + rtt + 0.1))
    assert len(_pings(bob_conn)) == 2
    _run(_tick_and_let_the_probe_run(sess, now + rtt + 0.2))
    assert len(_pings(bob_conn)) == 2

    events = _answer_the_probe(sess, ok=True)
    assert len(events) == 1 and events[0].type == SessionEventType.RESUMED


def test_probe_whose_send_failed_leaves_room_for_the_probe_of_a_transport_that_comes_back(
        bob_conn, bob_peer_id, alice_peer_id):
    """The probe sent on entering the state usually fails: there is no transport, and the
    responder may be unable to dial. That PING never left, so no answer can come and it
    must not hold the slot for its whole RTT window: when the initiator's dial brings the
    transport back, the responder asks at once."""
    accepted = [False]

    async def _send(peer_id, protocol_id, sess_msg):
        bob_conn.sent.append((peer_id, protocol_id, sess_msg))
        return accepted[0]
    bob_conn.send_session = _send
    sess = _reconnecting_responder(bob_conn, bob_peer_id, alice_peer_id)
    now = time.monotonic()
    _run(_tick_and_let_the_probe_run(sess, now))
    # The failed send is reaped, and nothing else leaves before the period.
    _run(_tick_and_let_the_probe_run(sess, now + 1.0))
    assert len(_pings(bob_conn)) == 1

    accepted[0] = True
    sess.on_transport_reconnected()
    _run(_tick_and_let_the_probe_run(sess, now + 2.0))
    assert len(_pings(bob_conn)) == 2
    events = _answer_the_probe(sess, ok=True)
    assert len(events) == 1 and events[0].type == SessionEventType.RESUMED


def test_resume_after_a_probe_brought_the_session_back_is_acked_with_no_event(bob_conn, bob_peer_id,
                                                                             alice_peer_id):
    """An initiator that is reconnecting too answers the probe with ok=true, and its own
    RESUME follows. By then the responder is ESTABLISHED: the RESUME is acked, and no
    second RESUMED goes to the layers above."""
    sess = _reconnecting_responder(bob_conn, bob_peer_id, alice_peer_id)
    _run(_tick_and_let_the_probe_run(sess, time.monotonic()))
    assert [e.type for e in _answer_the_probe(sess, ok=True)] == [SessionEventType.RESUMED]

    resume = SessMsg(SessMsg.KIND_RESUME, {"session_id": sess.session_id, "nonce": b"r1"})
    assert _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, resume)) == []
    assert sess.state == SessionState.ESTABLISHED
    last = _last_sent(bob_conn, SESSION_CONTROL_PROTOCOL)
    assert last[2].kind == SessMsg.KIND_RESUME_ACK and last[2].fields["accepted"] is True


def test_probe_whose_send_fails_changes_nothing(bob_conn, bob_peer_id, alice_peer_id):
    """The probe dials a peer that may be gone for good. A send that fails leaves the
    Session where it was, and the next probe leaves at the keep-alive period."""
    async def _refused(peer_id, protocol_id, sess_msg):
        bob_conn.sent.append((peer_id, protocol_id, sess_msg))
        return False
    bob_conn.send_session = _refused
    sess = _reconnecting_responder(bob_conn, bob_peer_id, alice_peer_id)
    entered, window = sess._state_entered_at, sess._timeout_s
    period = session_mod._KEEPALIVE_PERIOD_S
    now = time.monotonic()

    assert _run(_tick_and_let_the_probe_run(sess, now)) is None
    assert _run(_tick_and_let_the_probe_run(sess, now + period + 0.1)) is None
    assert len(_pings(bob_conn)) == 2
    assert sess.state == SessionState.RECONNECTING
    assert (sess._state_entered_at, sess._timeout_s) == (entered, window)


def test_probe_send_does_not_hold_the_tick(bob_conn, bob_peer_id, alice_peer_id):
    """With the transport gone the send of the probe dials first, for seconds when the peer
    does not answer. The tick serves every Session of the node from one task, so it must
    come back at once and leave the send behind."""
    async def _runner():
        release = asyncio.Event()

        async def _slow(peer_id, protocol_id, sess_msg):
            await release.wait()
            return False
        bob_conn.send_session = _slow
        sess = _reconnecting_responder(bob_conn, bob_peer_id, alice_peer_id)
        await asyncio.wait_for(sess.tick_keepalive(time.monotonic()), timeout=0.5)
        await asyncio.sleep(0)
        held = sess._probe_task is not None and not sess._probe_task.done()
        release.set()
        await sess._probe_task
        return held
    assert asyncio.run(_runner()) is True


def test_resume_drops_a_probe_in_flight(bob_conn, bob_peer_id, alice_peer_id):
    """A RESUME that lands while a probe is unanswered brings the Session back, and the
    answer to that probe, whatever it says, no longer counts."""
    sess = _reconnecting_responder(bob_conn, bob_peer_id, alice_peer_id)
    _run(_tick_and_let_the_probe_run(sess, time.monotonic()))
    resume = SessMsg(SessMsg.KIND_RESUME, {"session_id": sess.session_id, "nonce": b"r1"})
    events = _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, resume))
    assert len(events) == 1 and events[0].type == SessionEventType.RESUMED

    assert _answer_the_probe(sess, ok=False) == []
    assert _answer_the_probe(sess, ok=True) == []
    assert sess.state == SessionState.ESTABLISHED


def test_only_a_responder_that_lost_its_transport_probes(alice_conn, bob_conn, alice_peer_id,
                                                         bob_peer_id):
    """An initiator that is reconnecting has its own path (re-dial and RESUME), and a
    responder already told that its Session is gone has nothing left to ask."""
    initiator = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    initiator._set_state(SessionState.ESTABLISHED, None)
    initiator._last_ping_at = time.monotonic()
    initiator.on_transport_disconnected()
    _run(_tick_and_let_the_probe_run(initiator, time.monotonic()))
    assert _pings(alice_conn) == []

    told, _ = _responder_told_its_session_is_gone(bob_conn, bob_peer_id, alice_peer_id)
    _run(_tick_and_let_the_probe_run(told, time.monotonic()))
    told.on_transport_reconnected()
    _run(_tick_and_let_the_probe_run(told, time.monotonic()))
    assert _pings(bob_conn) == []


def test_keepalive_two_misses_trigger_reconnecting(alice_conn, alice_peer_id, bob_peer_id):
    """Two consecutive unanswered pings while ESTABLISHED → RECONNECTING."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(SessionState.ESTABLISHED, None)
    now = time.monotonic()
    # First miss: a ping is in flight whose RTT deadline has passed.
    sess._ping_in_flight = True
    sess._ping_deadline = now - 1.0
    sess._last_ping_at = now            # not due to send a fresh one this tick
    ev = _run(sess.tick_keepalive(now))
    assert ev is None and sess._consecutive_ping_failures == 1
    # Second miss.
    sess._ping_in_flight = True
    sess._ping_deadline = now - 1.0
    ev = _run(sess.tick_keepalive(now))
    assert ev is not None and ev.type == SessionEventType.RECONNECTING
    assert sess.state == SessionState.RECONNECTING


def test_on_ping_answers_pong_ok_when_session_matches(bob_conn, bob_peer_id, alice_peer_id):
    """An inbound PING whose session_id matches the held session is answered with a
    PONG(ok=true) echoing the ping nonce, with no event emitted."""
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id)
    sess._set_state(SessionState.ESTABLISHED, None)
    sess.session_id = b"\x07" * 16
    ping = SessMsg(SessMsg.KIND_PING, {"session_id": sess.session_id, "nonce": b"abc"})
    events = _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, ping))
    assert events == []
    last = _last_sent(bob_conn, SESSION_CONTROL_PROTOCOL)
    assert last is not None and last[2].kind == SessMsg.KIND_PONG
    assert last[2].fields["ok"] is True and last[2].fields["nonce"] == b"abc"


# RESUME (fast-path recovery)

def test_resume_inbound_while_reconnecting_resumes(bob_conn, bob_peer_id, alice_peer_id):
    """Peer RESUME for a session we still hold (and we are RECONNECTING) → ACK + RESUMED."""
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id)
    sess._set_state(SessionState.RECONNECTING, 10.0)
    sess.session_id = b"\x08" * 16
    resume = SessMsg(SessMsg.KIND_RESUME, {"session_id": sess.session_id, "nonce": b"r1"})
    events = _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, resume))
    assert len(events) == 1 and events[0].type == SessionEventType.RESUMED
    assert sess.state == SessionState.ESTABLISHED
    last = _last_sent(bob_conn, SESSION_CONTROL_PROTOCOL)
    assert last[2].kind == SessMsg.KIND_RESUME_ACK and last[2].fields["accepted"] is True


def test_resume_inbound_unknown_session_naks(bob_conn, bob_peer_id, alice_peer_id):
    """RESUME for a session_id we don't hold → NACK with SESSION_UNKNOWN."""
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id)
    sess._set_state(SessionState.ESTABLISHED, None)
    sess.session_id = b"\x08" * 16
    resume = SessMsg(SessMsg.KIND_RESUME, {"session_id": b"\x99" * 16, "nonce": b"r1"})
    events = _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, resume))
    assert events == []
    last = _last_sent(bob_conn, SESSION_CONTROL_PROTOCOL)
    assert last[2].kind == SessMsg.KIND_RESUME_ACK
    assert last[2].fields["accepted"] is False
    assert last[2].fields["reason_code"] == SESSION_UNKNOWN


def test_resume_ack_accepted_resumes(alice_conn, alice_peer_id, bob_peer_id):
    """An accepted RESUME_ACK with the matching nonce resumes a RECONNECTING initiator
    straight back to ESTABLISHED via a RESUMED event."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(SessionState.RECONNECTING, 10.0)
    sess._reconnect_substep = "resume_sent"
    sess._resume_nonce = b"rn"
    sess.session_id = b"\x08" * 16
    ack = SessMsg(SessMsg.KIND_RESUME_ACK, {"session_id": sess.session_id, "nonce": b"rn",
                                            "accepted": True, "reason_code": OK})
    events = _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, ack))
    assert len(events) == 1 and events[0].type == SessionEventType.RESUMED
    assert sess.state == SessionState.ESTABLISHED


def test_resume_ack_nack_replays_handshake_for_plain_session(alice_conn, alice_peer_id, bob_peer_id):
    """A NACK on a plain initiator session re-runs the full handshake (re-HELLO),
    preserving peer-id."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(SessionState.RECONNECTING, 10.0)
    sess._reconnect_substep = "resume_sent"
    sess._resume_nonce = b"rn"
    old_sid = sess.session_id
    ack = SessMsg(SessMsg.KIND_RESUME_ACK, {"session_id": b"\x00" * 16, "nonce": b"rn",
                                            "accepted": False, "reason_code": SESSION_UNKNOWN})
    events = _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, ack))
    assert events == []
    assert sess.state == SessionState.HELLO_SENT
    assert sess.session_id != old_sid     # identity rotated for the fresh handshake
    last = _last_sent(alice_conn, SESSION_HELLO_PROTOCOL)
    assert last is not None and last[2].kind == SessMsg.KIND_HELLO_REQ


def test_resume_ack_nack_world_session_is_lost(alice_conn, alice_peer_id, bob_peer_id):
    """A world-join session cannot transparently re-handshake (double-join risk) → LOST."""
    sess = Session(
        peer_id=bob_peer_id, role=SessionRole.INITIATOR, p2p=alice_conn,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=alice_peer_id,
        local_node_id="ln", local_profile_factory=lambda: {}, is_world_join=True,
    )
    sess._set_state(SessionState.RECONNECTING, 10.0)
    sess._reconnect_substep = "resume_sent"
    sess._resume_nonce = b"rn"
    ack = SessMsg(SessMsg.KIND_RESUME_ACK, {"session_id": b"\x00" * 16, "nonce": b"rn",
                                            "accepted": False, "reason_code": SESSION_UNKNOWN})
    events = _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, ack))
    assert len(events) == 1 and events[0].type == SessionEventType.LOST
    assert sess.state == SessionState.FAILED


def test_tick_dial_responder_does_not_redial(bob_conn, bob_peer_id, alice_peer_id):
    """A responder (no addresses) cannot re-dial; it rides the grace timer to LOST."""
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id)
    sess._set_state(SessionState.RECONNECTING, 10.0)
    assert _run(sess.tick_dial()) == []
    assert bob_conn.connect_calls == []


def test_tick_dial_initiator_redials_and_sends_resume(alice_conn, alice_peer_id, bob_peer_id):
    """An initiator with addresses re-dials and, on success, sends RESUME. Design A: the
    first tick_dial launches the redial task, a second (after the task runs) reaps it and
    sends RESUME."""
    async def _runner():
        sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
                   addresses=["/ip4/127.0.0.1/tcp/0/p2p/" + bob_peer_id])
        alice_conn.connect_returns = (bob_peer_id, False)
        sess._set_state(SessionState.RECONNECTING, 45.0)
        sess._reconnect_substep = "redial"
        sess._next_redial_at = 0.0
        now = time.monotonic()
        await sess.tick_dial(now)     # launch the redial
        await asyncio.sleep(0)        # let _run_dial run
        await sess.tick_dial(now)     # reap → RESUME
        return sess
    sess = asyncio.run(_runner())
    assert alice_conn.connect_calls          # a re-dial happened
    assert sess._reconnect_substep == "resume_sent"
    last = _last_sent(alice_conn, SESSION_CONTROL_PROTOCOL)
    assert last is not None and last[2].kind == SessMsg.KIND_RESUME


# The way back on a transport that the peer re-opened

_BOB_ADDR = "/ip4/127.0.0.1/tcp/0/p2p/"


def _reconnecting_initiator(alice_conn, alice_peer_id, bob_peer_id, reject_relay=False):
    """An initiator that lost its transport and whose own dial reaches nobody: the
    addresses it holds are the ones the Session started with."""
    sess = Session(
        peer_id=bob_peer_id, role=SessionRole.INITIATOR, p2p=alice_conn,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=alice_peer_id,
        local_node_id="local-node", local_profile_factory=lambda: {},
        addresses=[_BOB_ADDR + bob_peer_id], reject_relay=reject_relay,
    )
    sess._set_state(SessionState.ESTABLISHED, None)
    sess.session_id = b"\x05" * 16
    alice_conn.connect_returns = (None, False)
    assert sess.on_transport_disconnected().type == SessionEventType.RECONNECTING
    return sess


def _resumes(conn):
    return [e[2] for e in conn.sent
            if e[1] == SESSION_CONTROL_PROTOCOL and e[2].kind == SessMsg.KIND_RESUME]


def _resume_ack(sess, nonce):
    return SessMsg(SessMsg.KIND_RESUME_ACK, {"session_id": sess.session_id, "nonce": nonce,
                                             "accepted": True, "reason_code": OK})


async def _fail_the_dial_in_flight(sess, now):
    """Run the redial that was launched and reap its failure."""
    await asyncio.sleep(0)
    await sess.tick_dial(now)


def test_transport_back_sends_resume_with_no_dial_before_it(alice_conn, alice_peer_id, bob_peer_id):
    """The peer re-opened the connection itself. The initiator sends RESUME on it at
    the next tick: a dial of its own would go to the addresses it started with."""
    async def _runner():
        sess = _reconnecting_initiator(alice_conn, alice_peer_id, bob_peer_id)
        now = time.monotonic()
        await sess.tick_dial(now)                    # the redial of the loss
        await _fail_the_dial_in_flight(sess, now)    # ...which reaches nobody
        dials = len(alice_conn.connect_calls)
        assert _resumes(alice_conn) == []

        sess.on_transport_reconnected()
        await sess.tick_dial(now + 0.1)
        return sess, dials
    sess, dials = asyncio.run(_runner())
    assert len(_resumes(alice_conn)) == 1
    assert len(alice_conn.connect_calls) == dials    # no dial was made for it
    assert sess._reconnect_substep == "resume_sent"
    assert sess.state == SessionState.RECONNECTING   # the answer decides, not the signal


def test_resume_on_a_transport_that_came_back_does_not_wait_for_the_dial_in_flight(
        alice_conn, alice_peer_id, bob_peer_id):
    """A redial that is still running holds nothing: the RESUME leaves at once, and
    its answer brings the Session back while the dial is still in flight."""
    async def _runner():
        gate = asyncio.Event()

        async def _slow_connect(addresses):
            alice_conn.connect_calls.append(list(addresses))
            await gate.wait()
            return None, False
        alice_conn.connect = _slow_connect

        sess = _reconnecting_initiator(alice_conn, alice_peer_id, bob_peer_id)
        now = time.monotonic()
        await sess.tick_dial(now)
        await asyncio.sleep(0)
        assert sess._dial_task is not None and not sess._dial_task.done()

        sess.on_transport_reconnected()
        await sess.tick_dial(now + 0.1)
        sent = _resumes(alice_conn)
        assert len(sent) == 1
        events = await sess.process_inbound(SESSION_CONTROL_PROTOCOL,
                                            _resume_ack(sess, sent[0].fields["nonce"]))
        gate.set()
        await asyncio.sleep(0)
        late = await sess.tick_dial(now + 0.2)       # the dial ends after the resume
        return sess, events, late
    sess, events, late = asyncio.run(_runner())
    assert [e.type for e in events] == [SessionEventType.RESUMED]
    assert late == [] and sess.state == SessionState.ESTABLISHED


def test_dial_that_fails_does_not_cancel_a_resume_in_flight(alice_conn, alice_peer_id, bob_peer_id):
    """The RESUME left on the connection the peer opened, and the redial that was
    running fails a moment later. The answer to that RESUME must still be taken."""
    async def _runner():
        gate = asyncio.Event()

        async def _slow_connect(addresses):
            alice_conn.connect_calls.append(list(addresses))
            await gate.wait()
            return None, False
        alice_conn.connect = _slow_connect

        sess = _reconnecting_initiator(alice_conn, alice_peer_id, bob_peer_id)
        now = time.monotonic()
        await sess.tick_dial(now)
        await asyncio.sleep(0)
        sess.on_transport_reconnected()
        await sess.tick_dial(now + 0.1)
        nonce = _resumes(alice_conn)[0].fields["nonce"]

        gate.set()
        await asyncio.sleep(0)
        await sess.tick_dial(now + 0.2)              # reaps the failure of the dial
        assert sess._reconnect_substep == "resume_sent"
        return sess, await sess.process_inbound(SESSION_CONTROL_PROTOCOL, _resume_ack(sess, nonce))
    sess, events = asyncio.run(_runner())
    assert [e.type for e in events] == [SessionEventType.RESUMED]
    assert sess.state == SessionState.ESTABLISHED


def test_dial_that_succeeds_while_a_resume_is_in_flight_sends_no_second_one(
        alice_conn, alice_peer_id, bob_peer_id):
    async def _runner():
        gate = asyncio.Event()

        async def _slow_connect(addresses):
            alice_conn.connect_calls.append(list(addresses))
            await gate.wait()
            return bob_peer_id, False
        alice_conn.connect = _slow_connect

        sess = _reconnecting_initiator(alice_conn, alice_peer_id, bob_peer_id)
        now = time.monotonic()
        await sess.tick_dial(now)
        await asyncio.sleep(0)
        sess.on_transport_reconnected()
        await sess.tick_dial(now + 0.1)
        gate.set()
        await asyncio.sleep(0)
        await sess.tick_dial(now + 0.2)
        return sess
    sess = asyncio.run(_runner())
    assert len(_resumes(alice_conn)) == 1
    assert sess._reconnect_substep == "resume_sent"


def test_unanswered_resume_on_a_transport_that_came_back_is_followed_by_a_dial(
        alice_conn, alice_peer_id, bob_peer_id):
    """No answer within the wait: the initiator is back to its dials, at once and not
    at the end of the backoff of the dial that failed before."""
    wait = session_mod._RESUME_ACK_TIMEOUT_S

    async def _runner():
        sess = _reconnecting_initiator(alice_conn, alice_peer_id, bob_peer_id)
        failed_at = time.monotonic()
        for _ in range(4):                           # failures that push the backoff out
            failed_at = max(failed_at, sess._next_redial_at)
            await sess.tick_dial(failed_at)
            await _fail_the_dial_in_flight(sess, failed_at)
        gate_of_the_backoff = sess._next_redial_at
        dials = len(alice_conn.connect_calls)

        back_at = failed_at + 1.0
        assert back_at + wait + 0.1 < gate_of_the_backoff
        sess.on_transport_reconnected()
        await sess.tick_dial(back_at)
        await asyncio.sleep(0)                       # a dial launched here would run now
        await sess.tick_dial(back_at + wait - 0.1)
        await asyncio.sleep(0)
        inside_the_wait = len(alice_conn.connect_calls)
        await sess.tick_dial(back_at + wait + 0.1)
        await asyncio.sleep(0)
        return sess, dials, inside_the_wait
    sess, dials, inside_the_wait = asyncio.run(_runner())
    assert inside_the_wait == dials
    assert len(alice_conn.connect_calls) == dials + 1
    assert sess._reconnect_substep == "redial"


def test_resume_with_no_answer_counts_as_a_reconnect_attempt(alice_conn, alice_peer_id, bob_peer_id,
                                                             monkeypatch):
    """The dial returns at once on the connection to a peer that stopped, still listed, and
    the RESUME sent on it gets no answer. That is an attempt, said in the log: the next dial
    is the second one, it still leaves at once, and when it fails its backoff is the one of
    a second failure."""
    lines = []
    monkeypatch.setattr(session_mod.log, "session", lambda msg, *a, **k: lines.append(msg))
    wait = session_mod._RESUME_ACK_TIMEOUT_S

    async def _runner():
        sess = _reconnecting_initiator(alice_conn, alice_peer_id, bob_peer_id)
        alice_conn.connect_returns = (bob_peer_id, False)
        now = time.monotonic()
        await sess.tick_dial(now)                    # the dial on the dead connection...
        await asyncio.sleep(0)
        await sess.tick_dial(now)                    # ...is "ok", and the RESUME leaves
        assert len(_resumes(alice_conn)) == 1
        dials = len(alice_conn.connect_calls)

        alice_conn.connect_returns = (None, False)
        given_up_at = now + wait + 0.1
        await sess.tick_dial(given_up_at)
        await _fail_the_dial_in_flight(sess, given_up_at)
        return sess, len(alice_conn.connect_calls) - dials, given_up_at
    sess, launched, given_up_at = asyncio.run(_runner())
    assert launched == 1
    assert sess._reconnect_attempts == 2
    assert sess._next_redial_at == pytest.approx(given_up_at + 2 * session_mod._RETRY_BASE_S)
    assert any("RESUME got no answer" in line and "attempt 1" in line for line in lines), lines
    assert any("reconnect dial (attempt 2)" in line for line in lines), lines


# Fresh addresses for the redials

_MOVED = "/ip4/10.13.210.250/udp/37102/webrtc-direct/p2p/"


def _initiator_with_lookup(alice_conn, alice_peer_id, bob_peer_id, answer):
    """A reconnecting initiator that knows the node id of its peer and can ask the layer
    above for its addresses. `answer` is what each lookup returns."""
    calls = []

    async def _lookup(node_id):
        calls.append(node_id)
        if isinstance(answer, Exception):
            raise answer
        return answer
    sess = _reconnecting_initiator(alice_conn, alice_peer_id, bob_peer_id)
    sess._address_lookup = _lookup
    sess.remote_node_id_hint = "node-b"
    return sess, calls


async def _reap_lookup(sess, now):
    """Let the lookup that was started run, and take its answer."""
    await asyncio.sleep(0)
    await sess.tick_dial(now)


def test_addresses_are_looked_up_after_a_failed_dial_and_the_new_ones_dialled_at_once(
        alice_conn, alice_peer_id, bob_peer_id):
    """The peer moved: the dial with the addresses the Session started with fails, the
    lookup brings the new ones, and the next dial leaves at once with them, not at the
    end of its backoff. Addresses of another peer in the answer are left out."""
    other = "/ip4/10.0.0.9/tcp/4001/p2p/12D3KooWothereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
    sess, calls = _initiator_with_lookup(alice_conn, alice_peer_id, bob_peer_id,
                                         [_MOVED + bob_peer_id, other])

    async def _runner():
        now = time.monotonic()
        await sess.tick_dial(now)                    # the first redial, with the old addresses
        assert calls == []                           # nothing failed yet
        await _fail_the_dial_in_flight(sess, now)
        backoff_gate = sess._next_redial_at
        await sess.tick_dial(now + 0.1)              # the lookup starts...
        await _reap_lookup(sess, now + 0.2)          # ...and its answer is taken
        await asyncio.sleep(0)
        return now, backoff_gate
    now, backoff_gate = asyncio.run(_runner())
    assert calls == ["node-b"]
    assert sess.addresses == [_MOVED + bob_peer_id]
    assert backoff_gate > now + 0.2
    assert alice_conn.connect_calls[-1] == [_MOVED + bob_peer_id]


@pytest.mark.parametrize("failure", ["resume_send_failed", "resume_not_answered"])
def test_a_resume_that_failed_also_asks_for_addresses(alice_conn, alice_peer_id, bob_peer_id, failure):
    """A RESUME whose send failed, as in the first seconds of a change of network, and a
    RESUME that got no answer are failed attempts for the lookup as much as a dial is."""
    sess, calls = _initiator_with_lookup(alice_conn, alice_peer_id, bob_peer_id, None)
    if failure == "resume_send_failed":
        async def _refused(peer_id, protocol_id, sess_msg):
            return False
        alice_conn.send_session = _refused
    else:
        alice_conn.connect_returns = (bob_peer_id, False)

    async def _runner():
        now = time.monotonic()
        if failure == "resume_send_failed":
            sess.on_transport_reconnected()
            await sess.tick_dial(now)                # the RESUME cannot be sent
        else:
            await sess.tick_dial(now)                # the dial "succeeds"...
            await asyncio.sleep(0)
            await sess.tick_dial(now)                # ...and the RESUME leaves
            now += session_mod._RESUME_ACK_TIMEOUT_S + 0.1
            await sess.tick_dial(now)                # no answer
        await sess.tick_dial(now + 0.1)
        await _reap_lookup(sess, now + 0.2)
    asyncio.run(_runner())
    assert calls == ["node-b"]


def test_addresses_are_looked_up_at_most_once_every_interval(alice_conn, alice_peer_id, bob_peer_id):
    sess, calls = _initiator_with_lookup(alice_conn, alice_peer_id, bob_peer_id, None)
    interval = session_mod._ADDRESS_LOOKUP_INTERVAL_S

    async def _runner():
        now = time.monotonic()
        for t in (0.0, 2.0, interval - 1.0, interval + 0.5):
            at = max(now + t, sess._next_redial_at)
            await sess.tick_dial(at)
            await _fail_the_dial_in_flight(sess, at)
            await sess.tick_dial(at)
            await _reap_lookup(sess, at)
            if t == interval - 1.0:
                assert len(calls) == 1, calls
    asyncio.run(_runner())
    assert len(calls) == 2


def test_lookup_with_nothing_new_changes_nothing(alice_conn, alice_peer_id, bob_peer_id):
    """The peer kept its addresses (the common case: we moved, it did not), or the lookup
    failed: the addresses stay, and the dial keeps its backoff."""
    for answer in ([_BOB_ADDR + bob_peer_id], None, RuntimeError("root server down")):
        alice_conn.connect_calls.clear()
        sess, calls = _initiator_with_lookup(alice_conn, alice_peer_id, bob_peer_id, answer)

        async def _runner():
            now = time.monotonic()
            await sess.tick_dial(now)
            await _fail_the_dial_in_flight(sess, now)
            gate = sess._next_redial_at
            await sess.tick_dial(now + 0.1)
            await _reap_lookup(sess, now + 0.2)
            return gate
        gate = asyncio.run(_runner())
        assert calls == ["node-b"], answer
        assert sess.addresses == [_BOB_ADDR + bob_peer_id], answer
        assert sess._next_redial_at == gate and len(alice_conn.connect_calls) == 1, answer


def test_no_lookup_without_the_node_id_of_the_peer_or_a_lookup(alice_conn, alice_peer_id, bob_peer_id):
    sess, calls = _initiator_with_lookup(alice_conn, alice_peer_id, bob_peer_id, [_MOVED + bob_peer_id])
    sess.remote_node_id_hint = None
    plain = _reconnecting_initiator(alice_conn, alice_peer_id, bob_peer_id)

    async def _runner():
        now = time.monotonic()
        for s_ in (sess, plain):
            await s_.tick_dial(now)
            await _fail_the_dial_in_flight(s_, now)
            await s_.tick_dial(now + 0.1)
            await _reap_lookup(s_, now + 0.2)
    asyncio.run(_runner())
    assert calls == []
    assert sess.addresses == [_BOB_ADDR + bob_peer_id]


def test_the_tick_does_not_wait_for_the_lookup(alice_conn, alice_peer_id, bob_peer_id):
    """The root server may take its time: the tick that starts the lookup returns, and the
    redials go on meanwhile."""
    gate = None
    calls = []

    async def _runner():
        nonlocal gate
        gate = asyncio.Event()

        async def _slow_lookup(node_id):
            calls.append(node_id)
            await gate.wait()
            return [_MOVED + bob_peer_id]
        sess = _reconnecting_initiator(alice_conn, alice_peer_id, bob_peer_id)
        sess._address_lookup = _slow_lookup
        sess.remote_node_id_hint = "node-b"
        now = time.monotonic()
        await sess.tick_dial(now)
        await _fail_the_dial_in_flight(sess, now)
        await asyncio.wait_for(sess.tick_dial(now + 0.1), timeout=1.0)
        await asyncio.sleep(0)
        assert calls == ["node-b"] and sess._lookup_task is not None
        later = sess._next_redial_at
        await sess.tick_dial(later)                  # a redial while the lookup runs
        await asyncio.sleep(0)
        dials = len(alice_conn.connect_calls)
        gate.set()
        return sess, dials
    sess, dials = asyncio.run(_runner())
    assert dials == 2


def test_resume_sent_again_inside_its_wait_keeps_its_nonce(alice_conn, alice_peer_id, bob_peer_id):
    """Two signs that the transport is back, close to each other: the second RESUME is
    the first one again, so that the answer to either copy is taken."""
    async def _runner():
        sess = _reconnecting_initiator(alice_conn, alice_peer_id, bob_peer_id)
        now = time.monotonic()
        sess.on_transport_reconnected()
        await sess.tick_dial(now)
        sess.on_transport_reconnected()
        await sess.tick_dial(now + 0.5)
        sent = _resumes(alice_conn)
        events = await sess.process_inbound(SESSION_CONTROL_PROTOCOL,
                                            _resume_ack(sess, sent[0].fields["nonce"]))
        return sent, events
    sent, events = asyncio.run(_runner())
    assert len(sent) == 2 and sent[0].fields["nonce"] == sent[1].fields["nonce"]
    assert [e.type for e in events] == [SessionEventType.RESUMED]


def test_resume_whose_send_fails_falls_back_to_the_dial(alice_conn, alice_peer_id, bob_peer_id):
    async def _runner():
        sess = _reconnecting_initiator(alice_conn, alice_peer_id, bob_peer_id)
        now = time.monotonic()
        await sess.tick_dial(now)
        await _fail_the_dial_in_flight(sess, now)
        dials = len(alice_conn.connect_calls)

        async def _refused(peer_id, protocol_id, sess_msg):
            return False
        alice_conn.send_session = _refused
        sess.on_transport_reconnected()
        await sess.tick_dial(now + 0.1)
        await asyncio.sleep(0)
        return sess, dials
    sess, dials = asyncio.run(_runner())
    assert sess._reconnect_substep == "redial"
    assert len(alice_conn.connect_calls) == dials + 1


def test_ping_of_the_peer_makes_a_reconnecting_initiator_resume(alice_conn, alice_peer_id, bob_peer_id):
    """The peer still holds the Session and pings on it, over a connection the pools
    never reported. The PING is answered, and it is also the sign that the transport
    works: the RESUME follows at the next tick."""
    async def _runner():
        sess = _reconnecting_initiator(alice_conn, alice_peer_id, bob_peer_id)
        now = time.monotonic()
        await sess.tick_dial(now)
        await _fail_the_dial_in_flight(sess, now)
        ping = SessMsg(SessMsg.KIND_PING, {"session_id": sess.session_id, "nonce": b"p1"})
        events = await sess.process_inbound(SESSION_CONTROL_PROTOCOL, ping)
        pong = _last_sent(alice_conn, SESSION_CONTROL_PROTOCOL)[2]
        await sess.tick_dial(now + 0.1)
        return sess, events, pong
    sess, events, pong = asyncio.run(_runner())
    assert events == [] and sess.state == SessionState.RECONNECTING
    assert pong.kind == SessMsg.KIND_PONG and pong.fields["ok"] is True
    assert len(_resumes(alice_conn)) == 1


def test_ping_for_another_session_starts_no_resume(alice_conn, alice_peer_id, bob_peer_id):
    async def _runner():
        sess = _reconnecting_initiator(alice_conn, alice_peer_id, bob_peer_id)
        now = time.monotonic()
        await sess.tick_dial(now)
        await _fail_the_dial_in_flight(sess, now)
        ping = SessMsg(SessMsg.KIND_PING, {"session_id": b"\x99" * 16, "nonce": b"p1"})
        await sess.process_inbound(SESSION_CONTROL_PROTOCOL, ping)
        pong = _last_sent(alice_conn, SESSION_CONTROL_PROTOCOL)[2]
        await sess.tick_dial(now + 0.1)
        return pong
    pong = asyncio.run(_runner())
    assert pong.fields["ok"] is False
    assert _resumes(alice_conn) == []


def test_ping_of_the_peer_makes_a_reconnecting_responder_probe_at_once(bob_conn, bob_peer_id,
                                                                       alice_peer_id):
    """The same sign on the other role: a responder has no RESUME to send, it asks."""
    rtt = session_mod._PING_RTT_TIMEOUT_S

    async def _runner():
        sess = _reconnecting_responder(bob_conn, bob_peer_id, alice_peer_id)
        now = time.monotonic()
        await _tick_and_let_the_probe_run(sess, now)             # the probe of the loss
        await _tick_and_let_the_probe_run(sess, now + rtt + 0.1)   # it got no answer
        asked = len(_pings(bob_conn))
        await _tick_and_let_the_probe_run(sess, now + rtt + 0.2)
        assert len(_pings(bob_conn)) == asked        # the next one is not due before its period
        ping = SessMsg(SessMsg.KIND_PING, {"session_id": sess.session_id, "nonce": b"p1"})
        await sess.process_inbound(SESSION_CONTROL_PROTOCOL, ping)
        await _tick_and_let_the_probe_run(sess, now + rtt + 0.3)
        return asked
    asked = asyncio.run(_runner())
    assert len(_pings(bob_conn)) == asked + 1


def test_initiator_that_refuses_relays_redials_when_the_transport_comes_back(
        alice_conn, alice_peer_id, bob_peer_id):
    """What came back may be a relay circuit, and only a dial says which path it is: a
    Session that refuses relays keeps to its dial, started at once."""
    async def _runner():
        sess = _reconnecting_initiator(alice_conn, alice_peer_id, bob_peer_id, reject_relay=True)
        now = time.monotonic()
        for _ in range(3):
            await sess.tick_dial(now)
            await _fail_the_dial_in_flight(sess, now)
            now = sess._next_redial_at
        dials = len(alice_conn.connect_calls)
        now -= 1.0                                   # before the gate of the backoff
        await sess.tick_dial(now)
        await asyncio.sleep(0)
        assert len(alice_conn.connect_calls) == dials
        sess.on_transport_reconnected()
        await sess.tick_dial(now)
        await asyncio.sleep(0)
        return dials
    dials = asyncio.run(_runner())
    assert _resumes(alice_conn) == []
    assert len(alice_conn.connect_calls) == dials + 1


def test_sign_that_follows_the_dial_of_an_initiator_that_refuses_relays_costs_no_second_dial(
        alice_conn, alice_peer_id, bob_peer_id):
    """Its dial succeeded and its RESUME is in flight when the pools report the peer that
    dial has just reached. The RESUME is left to its answer: the sign used to send the
    Session back to its dial, and the answer on its way was then not taken."""
    async def _runner():
        sess = _reconnecting_initiator(alice_conn, alice_peer_id, bob_peer_id, reject_relay=True)
        alice_conn.connect_returns = (bob_peer_id, False)
        now = time.monotonic()
        await sess.tick_dial(now)
        await asyncio.sleep(0)
        await sess.tick_dial(now)                    # the dial is reaped: RESUME
        sent = _resumes(alice_conn)
        dials = len(alice_conn.connect_calls)
        sess.on_transport_reconnected()
        await sess.tick_dial(now + 0.1)
        await asyncio.sleep(0)
        events = await sess.process_inbound(SESSION_CONTROL_PROTOCOL,
                                            _resume_ack(sess, sent[0].fields["nonce"]))
        return sent, dials, events
    sent, dials, events = asyncio.run(_runner())
    assert len(sent) == 1 and len(alice_conn.connect_calls) == dials
    assert [e.type for e in events] == [SessionEventType.RESUMED]


# Out-of-order / duplicate messages

def test_profile_offer_in_hello_sent_is_ignored(alice_conn, alice_peer_id, bob_peer_id):
    """A PROFILE_OFFER arriving while the initiator still waits for HELLO_ACK
    (replay or broken peer) must be dropped: no state change, no decision sent."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               addresses=["/ip4/127.0.0.1/tcp/0"])
    _start(sess)
    assert sess.state == SessionState.HELLO_SENT
    offer = SessMsg(SessMsg.KIND_PROFILE_OFFER, {
        "profile": json.dumps({}).encode(), "sig_a": b"sig",
        "session_id": sess.session_id,
    })
    events = _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, offer))
    assert events == []
    assert sess.state == SessionState.HELLO_SENT
    assert _last_sent(alice_conn, SESSION_PROFILE_PROTOCOL) is None


# A HELLO that cannot be sent is a dial that failed

def _hello_send_fails(conn, times, delivered=False):
    """The next `times` HELLOs sent through `conn` are reported as not sent. With
    `delivered` they reach the peer all the same: the frame left before the send failed."""
    original = conn.send_session
    left = [times]

    async def _send(peer_id, protocol_id, sess_msg):
        if (protocol_id == SESSION_HELLO_PROTOCOL and sess_msg.kind == SessMsg.KIND_HELLO_REQ
                and left[0] > 0):
            left[0] -= 1
            if delivered:
                await original(peer_id, protocol_id, sess_msg)
            return False
        return await original(peer_id, protocol_id, sess_msg)
    conn.send_session = _send


def _hellos(conn):
    return [e[2] for e in conn.sent
            if e[1] == SESSION_HELLO_PROTOCOL and e[2].kind == SessMsg.KIND_HELLO_REQ]


def test_hello_that_cannot_be_sent_is_a_dial_that_failed(alice_conn, alice_peer_id, bob_peer_id,
                                                         monkeypatch):
    """The dial resolved on a connection that carried nothing. With attempts left the
    Session stays in CONNECTING, says so, and dials again after the backoff of a failed
    dial; the HELLO of that dial leaves."""
    lines = []
    monkeypatch.setattr(session_mod.log, "session", lambda msg, *a, **k: lines.append(msg))
    alice_conn.connect_returns = (bob_peer_id, False)
    _hello_send_fails(alice_conn, 1)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               addresses=["/ip4/127.0.0.1/tcp/0/p2p/" + bob_peer_id])

    async def _runner():
        sess.start()
        await asyncio.sleep(0)
        now = time.monotonic()
        events = await sess.tick_dial(now)
        state, gate = sess.state, sess._next_dial_at
        await sess.tick_dial(gate)                   # the second dial...
        await asyncio.sleep(0)
        await sess.tick_dial(gate)                   # ...and its HELLO
        return events, state, now, gate
    events, state, now, gate = _run(_runner())
    assert events == [] and state == SessionState.CONNECTING
    assert now <= gate <= now + session_mod._RETRY_BASE_S
    assert any("HELLO could not be sent" in line and "retrying in" in line for line in lines), lines
    assert len(alice_conn.connect_calls) == 2 and sess._dial_attempts == 2
    assert sess.state == SessionState.HELLO_SENT
    assert len(_hellos(alice_conn)) == 1


def test_hello_that_cannot_be_sent_on_the_last_attempt_fails_the_session(alice_conn, alice_peer_id,
                                                                         bob_peer_id):
    alice_conn.connect_returns = (bob_peer_id, False)
    _hello_send_fails(alice_conn, 1)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               addresses=["/ip4/127.0.0.1/tcp/0/p2p/" + bob_peer_id], dial_attempts=1)
    events = _start(sess)
    assert [e.type for e in events] == [SessionEventType.FAILED]
    assert events[0].data["reason"] == "hello_send_failed"
    assert sess.state == SessionState.FAILED


def test_repeated_hello_is_answered_over_its_own_nonce(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """The initiator takes only an answer signed over the nonce_a of its last HELLO."""
    signed = []
    bob_p2p._sign_with_host_key_sync = lambda data: signed.append(bytes(data)) or b"sig-bytes"
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id)
    first = SessMsg(SessMsg.KIND_HELLO_REQ, {
        "session_id": b"\x01" * 16, "nonce_a": b"\xaa" * 32,
        "protocol_versions": ["1.0"], "capabilities": 0, "client_id": "test/1.0",
    })
    again = SessMsg(SessMsg.KIND_HELLO_REQ, dict(first.fields, nonce_a=b"\xbb" * 32))
    _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, first))
    _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, again))
    assert signed == [session_mod._sign_payload(b"\xaa" * 32, b"\x01" * 16),
                      session_mod._sign_payload(b"\xbb" * 32, b"\x01" * 16)]


@pytest.mark.parametrize("first_answer", ["while_connecting", "after_the_second_hello"])
def test_hello_that_arrived_although_its_send_failed_ends_established(
        bidirectional_fakeconns, alice_peer_id, bob_peer_id, first_answer):
    """The send of the first HELLO failed after its frame left: the responder answered it,
    the initiator dialled again and sent it again, and the responder answers once more.
    The answer to the first HELLO reaches the initiator in CONNECTING, where it is not
    taken, or after the second HELLO, where it is stale. Both sides end ESTABLISHED, and
    nobody was quarantined."""
    from conftest import apump
    from unaiverse.networking.session import SessionManager
    alice_conn, bob_conn, alice_inbox, bob_inbox = bidirectional_fakeconns
    alice_conn.connect_returns = (bob_peer_id, False)
    _hello_send_fails(alice_conn, 1, delivered=True)

    def _sm(conn, peer_id, node_id):
        return SessionManager(p2p=conn, p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=peer_id,
                              local_node_id=node_id, local_profile_factory=lambda: {"node_id": node_id})

    async def _runner():
        sm_a, sm_b = _sm(alice_conn, alice_peer_id, "alice"), _sm(bob_conn, bob_peer_id, "bob")
        await sm_a.initiate(bob_peer_id, ["/ip4/127.0.0.1/tcp/0/p2p/" + bob_peer_id])
        await asyncio.sleep(0)
        await sm_a.tick()                            # the dial, and the HELLO that "failed"
        assert sm_a.get_session(bob_peer_id).state == SessionState.CONNECTING
        await apump(bob_inbox, sm_b)                 # the responder answers it
        if first_answer == "while_connecting":
            await apump(alice_inbox, sm_a)
        gate = sm_a.get_session(bob_peer_id)._next_dial_at
        await sm_a.tick(gate)
        await asyncio.sleep(0)
        await sm_a.tick(gate)                        # the second dial and HELLO
        events = []
        for _ in range(4):
            events += await apump(bob_inbox, sm_b)
            events += await apump(alice_inbox, sm_a)
        return sm_a, sm_b, events
    sm_a, sm_b, events = _run(_runner())
    assert sm_a.get_session(bob_peer_id).state == SessionState.ESTABLISHED
    assert sm_b.get_session(alice_peer_id).state == SessionState.ESTABLISHED
    assert [e.type for e in events] == [SessionEventType.ESTABLISHED] * 2
    assert sm_a._quarantine == {} and sm_b._quarantine == {}


def test_repeated_hellos_past_the_ceiling_are_not_answered(bob_conn, bob_peer_id, alice_peer_id,
                                                           monkeypatch):
    """Each answer to a repeated HELLO is a signature with the host key and a send. Up to
    the ceiling they are answered, the one past it is ignored, says so, and changes
    nothing. The ceiling covers the most a legitimate initiator repeats its HELLO: one per
    dial attempt after the first (the browser makes up to 6 attempts), and the repeats of
    the last one when its answer does not come."""
    lines = []
    monkeypatch.setattr(session_mod.log, "session", lambda msg, *a, **k: lines.append(msg))
    ceiling = session_mod._MAX_REPEATED_HELLOS_ANSWERED
    assert ceiling >= 6 - 1 + session_mod._MAX_HELLO_REPEATS
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id)
    hello = SessMsg(SessMsg.KIND_HELLO_REQ, {
        "session_id": b"\x01" * 16, "nonce_a": b"\xaa" * 32,
        "protocol_versions": ["1.0"], "capabilities": 0, "client_id": "test/1.0",
    })

    def _answers():
        return [e for e in bob_conn.sent
                if e[1] == SESSION_HELLO_PROTOCOL and e[2].kind == SessMsg.KIND_HELLO_RESP]
    _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, hello))
    nonce_b, entered_at = sess.nonce_b, sess._state_entered_at
    for _ in range(ceiling):
        _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, hello))
    assert len(_answers()) == 1 + ceiling
    events = _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, hello))
    assert events == [] and len(_answers()) == 1 + ceiling
    assert sess.state == SessionState.HELLO_ACK_SENT and sess.session_id == b"\x01" * 16
    assert sess.nonce_b == nonce_b and sess._state_entered_at == entered_at
    assert any("repeated HELLO ignored" in line for line in lines), lines


# The HELLO sent again when no HELLO_ACK came

def _lose(conn, kind, times):
    """The next `times` frames of `kind` sent through `conn` are reported as sent and never
    reach the peer: a connection closed under them after the write returned."""
    original = conn.send_session
    left = [times]

    async def _send(peer_id, protocol_id, sess_msg):
        if sess_msg.kind == kind and left[0] > 0:
            left[0] -= 1
            conn.sent.append((peer_id, protocol_id, sess_msg))
            return True
        return await original(peer_id, protocol_id, sess_msg)
    conn.send_session = _send


def _pair(alice_conn, bob_conn, alice_peer_id, bob_peer_id):
    from unaiverse.networking.session import SessionManager

    def _sm(conn, peer_id, node_id):
        return SessionManager(p2p=conn, p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=peer_id,
                              local_node_id=node_id, local_profile_factory=lambda: {"node_id": node_id})
    alice_conn.connect_returns = (bob_peer_id, False)
    return _sm(alice_conn, alice_peer_id, "alice"), _sm(bob_conn, bob_peer_id, "bob")


@pytest.mark.parametrize("lost", [SessMsg.KIND_HELLO_REQ, SessMsg.KIND_HELLO_RESP])
def test_a_lost_hello_or_hello_ack_is_made_good_by_the_hello_sent_again(
        bidirectional_fakeconns, alice_peer_id, bob_peer_id, lost, monkeypatch):
    """T5a-local of 2026-11-21 and X5 of 2026-11-18: the circuit the HELLO, or its
    HELLO_ACK, was written on closed before the frame was read, and nobody knew. The
    initiator waited the whole stage and failed timeout:hello_sent. Three seconds with no
    answer, and it sends the same HELLO again: the handshake completes."""
    from conftest import apump
    lines = []
    monkeypatch.setattr(session_mod.log, "session", lambda msg, *a, **k: lines.append(msg))
    alice_conn, bob_conn, alice_inbox, bob_inbox = bidirectional_fakeconns
    _lose(alice_conn if lost == SessMsg.KIND_HELLO_REQ else bob_conn, lost, 1)

    async def _runner():
        sm_a, sm_b = _pair(alice_conn, bob_conn, alice_peer_id, bob_peer_id)
        await sm_a.initiate(bob_peer_id, ["/ip4/127.0.0.1/tcp/0/p2p/" + bob_peer_id])
        await asyncio.sleep(0)
        await sm_a.tick()                            # the dial, and the HELLO
        sess = sm_a.get_session(bob_peer_id)
        entered_at = sess._state_entered_at
        events = await apump(bob_inbox, sm_b)
        events += await apump(alice_inbox, sm_a)
        assert sess.state == SessionState.HELLO_SENT
        for at in (2.9, 3.1, 15.1):
            events += await sm_a.tick(entered_at + at)
            for _ in range(4):
                events += await apump(bob_inbox, sm_b)
                events += await apump(alice_inbox, sm_a)
        return sm_a, sm_b, events
    sm_a, sm_b, events = _run(_runner())
    assert [(e.type, e.data.get("reason")) for e in events] == [(SessionEventType.ESTABLISHED, None)] * 2
    assert sm_a.get_session(bob_peer_id).state == SessionState.ESTABLISHED
    assert sm_b.get_session(alice_peer_id).state == SessionState.ESTABLISHED
    assert sum("HELLO sent again (1/2)" in line for line in lines) == 1


def test_an_answer_that_comes_after_the_repeat_breaks_nothing(bidirectional_fakeconns, alice_peer_id,
                                                              bob_peer_id):
    """The answer was only slow: it reaches the initiator after its repeat left. The
    initiator goes on with it, the responder answers the repeat as well, and the second
    answer is ignored by an initiator already past HELLO_SENT."""
    from conftest import apump
    alice_conn, bob_conn, alice_inbox, bob_inbox = bidirectional_fakeconns

    async def _runner():
        sm_a, sm_b = _pair(alice_conn, bob_conn, alice_peer_id, bob_peer_id)
        await sm_a.initiate(bob_peer_id, ["/ip4/127.0.0.1/tcp/0/p2p/" + bob_peer_id])
        await asyncio.sleep(0)
        await sm_a.tick()
        entered_at = sm_a.get_session(bob_peer_id)._state_entered_at
        events = await apump(bob_inbox, sm_b)        # the answer, on its way
        events += await sm_a.tick(entered_at + 3.1)   # the repeat leaves before it arrives
        for _ in range(4):
            events += await apump(alice_inbox, sm_a)
            events += await apump(bob_inbox, sm_b)
        return sm_a, sm_b, events
    sm_a, sm_b, events = _run(_runner())
    assert [e.type for e in events] == [SessionEventType.ESTABLISHED] * 2
    assert sm_a.get_session(bob_peer_id).state == SessionState.ESTABLISHED
    assert sm_b.get_session(alice_peer_id).state == SessionState.ESTABLISHED
    answers = [e for e in bob_conn.sent if e[2].kind == SessMsg.KIND_HELLO_RESP]
    offers = [e for e in alice_conn.sent if e[2].kind == SessMsg.KIND_PROFILE_OFFER]
    assert len(answers) == 2 and len(offers) == 1


def test_the_hello_goes_out_again_twice_at_most_within_the_unchanged_stage(alice_conn, alice_peer_id,
                                                                          bob_peer_id, monkeypatch):
    """No answer at all: the same HELLO (same session and nonce) again at 3 s and at 6 s
    from the last one, one line each, then nothing more; the stage still ends at 15 s."""
    lines = []
    monkeypatch.setattr(session_mod.log, "session", lambda msg, *a, **k: lines.append(msg))
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               addresses=["/ip4/127.0.0.1/tcp/0/p2p/" + bob_peer_id])

    async def _runner():
        sess.start()
        await asyncio.sleep(0)
        await sess.tick_dial()
        entered_at = sess._state_entered_at
        counts = []
        for at in (2.9, 3.1, 4.0, 6.0, 6.2, 9.5, 14.9):
            await sess.tick_dial(entered_at + at)
            counts.append(len(_hellos(alice_conn)))
        assert sess._state_entered_at == entered_at   # the stage was not started again
        return counts, sess.check_timeout(entered_at + 14.9), sess.check_timeout(entered_at + 15.1)
    counts, early, late = _run(_runner())
    assert counts == [1, 2, 2, 2, 3, 3, 3]
    hellos = _hellos(alice_conn)
    assert all(h.fields == hellos[0].fields for h in hellos)
    assert early is None
    assert late.type == SessionEventType.FAILED and late.data["reason"] == "timeout:hello_sent"
    assert [line for line in lines if "HELLO sent again" in line] == [
        line for line in lines if "(1/2)" in line or "(2/2)" in line]
    assert sum("HELLO sent again" in line for line in lines) == 2


def test_a_repeat_that_cannot_be_sent_fails_nothing(alice_conn, alice_peer_id, bob_peer_id, monkeypatch):
    """The send of a repeat fails: the stage still waits for its answer, and says so."""
    lines = []
    monkeypatch.setattr(session_mod.log, "session", lambda msg, *a, **k: lines.append(msg))
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               addresses=["/ip4/127.0.0.1/tcp/0/p2p/" + bob_peer_id])

    async def _runner():
        sess.start()
        await asyncio.sleep(0)
        await sess.tick_dial()
        _hello_send_fails(alice_conn, 1)
        return await sess.tick_dial(sess._state_entered_at + 3.1)
    events = _run(_runner())
    assert events == [] and sess.state == SessionState.HELLO_SENT
    assert any("HELLO could not be sent again (1/2)" in line for line in lines), lines


def test_duplicate_hello_req_ignored_after_hello_ack_sent(bob_conn, bob_peer_id, alice_peer_id):
    """A retransmitted HELLO_REQ on a responder already past INIT must not
    restart the handshake: same state, same session_id, same nonce_b, and the wait
    for the PROFILE_OFFER is not restarted. It is answered once more (its initiator
    may have been told that its first send failed), and a HELLO_REQ of another
    session_id is not answered at all."""
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id)
    hello = SessMsg(SessMsg.KIND_HELLO_REQ, {
        "session_id": b"\x01" * 16, "nonce_a": b"\xaa" * 32,
        "protocol_versions": ["1.0"], "capabilities": 0, "client_id": "test/1.0",
    })
    _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, hello))
    assert sess.state == SessionState.HELLO_ACK_SENT
    nonce_b, entered_at = sess.nonce_b, sess._state_entered_at
    events = _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, hello))
    assert events == []
    assert sess.state == SessionState.HELLO_ACK_SENT
    assert sess.session_id == b"\x01" * 16
    assert sess.nonce_b == nonce_b and sess._state_entered_at == entered_at
    other = SessMsg(SessMsg.KIND_HELLO_REQ, dict(hello.fields, session_id=b"\x02" * 16))
    assert _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, other)) == []
    assert sess.session_id == b"\x01" * 16
    hello_resps = [e for e in bob_conn.sent
                   if e[1] == SESSION_HELLO_PROTOCOL and e[2].kind == SessMsg.KIND_HELLO_RESP]
    assert len(hello_resps) == 2
    assert all(e[2].fields["nonce_b"] == nonce_b for e in hello_resps)


def test_profile_decision_in_established_is_ignored(alice_conn, alice_peer_id, bob_peer_id):
    """A duplicate PROFILE_DECISION after ESTABLISHED must not emit a second
    ESTABLISHED event nor overwrite the stored remote profile."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(SessionState.ESTABLISHED, None)
    sess.remote_profile = {"node_id": "bob-node"}
    dup = SessMsg(SessMsg.KIND_PROFILE_DECISION, {
        "accepted": True, "profile_b": json.dumps({"node_id": "evil"}).encode(),
        "reason_code": OK, "session_id": sess.session_id,
    })
    events = _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, dup))
    assert events == []
    assert sess.state == SessionState.ESTABLISHED
    assert sess.remote_profile == {"node_id": "bob-node"}


def test_world_grant_to_plain_initiator_is_ignored(alice_conn, alice_peer_id, bob_peer_id):
    """A WORLD_GRANT sent to an initiator that did not ask for a world join
    (is_world_join=False) must be dropped without state change."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id,
               addresses=["/ip4/127.0.0.1/tcp/0"])
    _start(sess)
    _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, SessMsg(SessMsg.KIND_HELLO_RESP, {
        "chosen_version": "1.0", "nonce_b": b"\xbb" * 32, "sig_b": b"sig", "reason_code": OK,
    })))
    assert sess.state == SessionState.PROFILE_SENT
    grant = SessMsg(SessMsg.KIND_WORLD_GRANT, {
        "grant_id": b"\x05" * 32,
        "expiry_unix_ms": int(time.time() * 1000) + 30000,
        "world_profile": b"{}",
        "role": 1, "code_bundle": b"", "initial_stats": b"", "rendezvous_tag": 0,
    })
    events = _run(sess.process_inbound(SESSION_WORLD_PROTOCOL, grant))
    assert events == []
    assert sess.state == SessionState.PROFILE_SENT


# Silent abort (tie-breaker support)

def test_abort_silently_fails_without_event(alice_conn, alice_peer_id, bob_peer_id):
    """abort_silently() drops the session to FAILED without emitting any event, as the
    losing side of a simultaneous-open tie-break needs."""
    sess = _mk(bob_peer_id, SessionRole.INITIATOR, alice_conn, alice_peer_id)
    sess._set_state(SessionState.HELLO_SENT, 5.0)
    sess.abort_silently("simultaneous_open_yielded")
    assert sess.state == SessionState.FAILED


def test_world_grant_arriving_before_decision_is_buffered(alice_conn, alice_peer_id, bob_peer_id):
    """WORLD and PROFILE ride different libp2p protocols, so the host's
    back-to-back DECISION+GRANT can be popped grant-first. The early grant must
    be buffered and consumed when the decision lands (it used to be dropped,
    stranding the joiner in awaiting_world_grant until timeout)."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = Session(
        peer_id=bob_peer_id, role=SessionRole.INITIATOR, p2p=alice_conn,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=alice_peer_id,
        local_node_id="local-node", local_profile_factory=lambda: {},
        addresses=["/ip4/127.0.0.1/tcp/0"], is_world_join=True,
    )
    _start(sess)
    _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, SessMsg(SessMsg.KIND_HELLO_RESP, {
        "chosen_version": "1.0", "nonce_b": b"\xbb" * 32, "sig_b": b"sig", "reason_code": OK,
    })))
    assert sess.state == SessionState.PROFILE_SENT

    world_profile = json.dumps({"dynamic": {"private_peer_id": "12D3WorldPriv"}}).encode()
    grant = SessMsg(SessMsg.KIND_WORLD_GRANT, {
        "grant_id": b"\x05" * 32,
        "expiry_unix_ms": int(time.time() * 1000) + 30000,
        "world_profile": world_profile,
        "role": 3, "code_bundle": b"", "initial_stats": b"", "rendezvous_tag": 0,
    })
    events = _run(sess.process_inbound(SESSION_WORLD_PROTOCOL, grant))
    assert events == []
    assert sess.state == SessionState.PROFILE_SENT  # buffered, not applied yet

    events = _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, SessMsg(SessMsg.KIND_PROFILE_DECISION, {
        "accepted": True, "profile_b": b"{}", "reason_code": OK, "session_id": sess.session_id,
    })))
    assert sess.state == SessionState.ESTABLISHED
    established = [e for e in events if e.type == SessionEventType.ESTABLISHED]
    assert established and established[0].data.get("world_grant") is not None


def test_world_grant_event_carries_verified_code_hash_claim(alice_conn, alice_peer_id, bob_peer_id):
    """The world_grant event is annotated with the code hash from the VERIFIED
    token claims of the public Session (never from the grant message itself), so
    the join body can check the received bundle against the root-attested hash."""
    async def _verifier(token, peer_id):
        return ("world-node", "cv", (peer_id,), 9999999999, "aa" * 32)
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = Session(
        peer_id=bob_peer_id, role=SessionRole.INITIATOR, p2p=alice_conn,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=alice_peer_id,
        local_node_id="local-node", local_profile_factory=lambda: {},
        addresses=["/ip4/127.0.0.1/tcp/0"], is_world_join=True,
        token_verifier=_verifier,
    )
    _start(sess)
    _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, SessMsg(SessMsg.KIND_HELLO_RESP, {
        "chosen_version": "1.0", "nonce_b": b"\xbb" * 32, "sig_b": b"sig", "reason_code": OK,
    })))
    _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, SessMsg(SessMsg.KIND_PROFILE_DECISION, {
        "accepted": True, "profile_b": b"{}", "reason_code": OK, "session_id": sess.session_id,
        "token": "world-tok",
    })))
    assert sess.state == SessionState.AWAITING_WORLD_GRANT
    grant = SessMsg(SessMsg.KIND_WORLD_GRANT, {
        "grant_id": b"\x05" * 32,
        "expiry_unix_ms": int(time.time() * 1000) + 30000,
        "world_profile": b"{}",
        "role": 3, "code_bundle": b"", "initial_stats": b"", "rendezvous_tag": 0,
    })
    events = _run(sess.process_inbound(SESSION_WORLD_PROTOCOL, grant))
    established = [e for e in events if e.type == SessionEventType.ESTABLISHED]
    assert established
    assert established[0].data["world_grant"]["world_code_hash_claim"] == "aa" * 32


def test_initiator_rejects_expired_world_grant(alice_conn, alice_peer_id, bob_peer_id):
    """The world joiner re-checks the grant expiry when the WORLD_GRANT lands on its
    public Session: a stale grant fails the handshake (the only expiry check is here,
    since the PRIVATE_JOIN redemption round-trip is gone)."""
    alice_conn.connect_returns = (bob_peer_id, False)
    sess = Session(
        peer_id=bob_peer_id, role=SessionRole.INITIATOR, p2p=alice_conn,
        p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=alice_peer_id,
        local_node_id="local-node", local_profile_factory=lambda: {},
        addresses=["/ip4/127.0.0.1/tcp/0"], is_world_join=True,
    )
    _start(sess)
    _run(sess.process_inbound(SESSION_HELLO_PROTOCOL, SessMsg(SessMsg.KIND_HELLO_RESP, {
        "chosen_version": "1.0", "nonce_b": b"\xbb" * 32, "sig_b": b"sig", "reason_code": OK,
    })))
    _run(sess.process_inbound(SESSION_PROFILE_PROTOCOL, SessMsg(SessMsg.KIND_PROFILE_DECISION, {
        "accepted": True, "profile_b": b"{}", "reason_code": OK, "session_id": sess.session_id,
    })))
    assert sess.state == SessionState.AWAITING_WORLD_GRANT
    expired = SessMsg(SessMsg.KIND_WORLD_GRANT, {
        "grant_id": b"\x05" * 32, "expiry_unix_ms": 1,  # long past
        "world_profile": json.dumps({"dynamic": {"private_peer_id": "12D3WorldPriv"}}).encode(),
        "role": 3, "code_bundle": b"", "initial_stats": b"", "rendezvous_tag": 0,
    })
    events = _run(sess.process_inbound(SESSION_WORLD_PROTOCOL, expired))
    assert len(events) == 1 and events[0].type == SessionEventType.FAILED
    assert "world_grant_expired" in events[0].data["reason"]


# In-session token refresh (TOKEN_UPDATE) + expiry

def _established_responder(bob_conn, bob_peer_id, alice_peer_id, verifier, node_hint="node-X"):
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id,
               token_verifier=verifier, remote_node_id_hint=node_hint)
    sess._set_state(SessionState.ESTABLISHED, None)
    return sess


def test_token_update_advances_expiry_monotonically(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """An ESTABLISHED peer's TOKEN_UPDATE advances the cached exp; a replayed older
    token cannot shrink it (monotonic). The fake verifier reads exp from the token."""
    async def _verifier(token, peer_id):
        return ("node-X", "cv2", (peer_id,), int(token), None)
    sess = _established_responder(bob_conn, bob_peer_id, alice_peer_id, _verifier)
    sess.remote_token_exp = 1000
    _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, SessMsg(SessMsg.KIND_TOKEN_UPDATE, {
        "session_id": sess.session_id, "token": "2000"})))
    assert sess.remote_token_exp == 2000 and sess.remote_cv_hash == "cv2"
    _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, SessMsg(SessMsg.KIND_TOKEN_UPDATE, {
        "session_id": sess.session_id, "token": "1500"})))
    assert sess.remote_token_exp == 2000   # not moved backwards


def test_token_update_rejects_node_id_mismatch(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """A TOKEN_UPDATE whose token attests a different node-id is ignored — the update
    is dropped, the session is not."""
    async def _verifier(token, peer_id):
        return ("intruder", "cv", (peer_id,), 9999999999, None)
    sess = _established_responder(bob_conn, bob_peer_id, alice_peer_id, _verifier)
    sess.remote_token_exp = 1000
    _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, SessMsg(SessMsg.KIND_TOKEN_UPDATE, {
        "session_id": sess.session_id, "token": "x"})))
    assert sess.remote_token_exp == 1000
    assert sess.state == SessionState.ESTABLISHED


def test_token_update_buffered_until_established(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """A TOKEN_UPDATE arriving before ESTABLISHED is buffered and applied on the next
    tick once established."""
    async def _verifier(token, peer_id):
        return ("node-X", "cv", (peer_id,), 5000, None)
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id,
               token_verifier=_verifier, remote_node_id_hint="node-X")
    sess._set_state(SessionState.RECONNECTING, 10.0)
    _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, SessMsg(SessMsg.KIND_TOKEN_UPDATE, {
        "session_id": sess.session_id, "token": "x"})))
    assert sess.remote_token_exp == 0 and sess._buffered_token_update is not None
    sess._set_state(SessionState.ESTABLISHED, None)
    _run(sess.tick_keepalive())
    assert sess.remote_token_exp == 5000 and sess._buffered_token_update is None


def test_tick_goaways_on_expired_token(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """An ESTABLISHED session whose peer token lapsed past the grace window is closed
    with a GOAWAY(TOKEN_EXPIRED)."""
    sess = _mk(alice_peer_id, SessionRole.RESPONDER, bob_conn, bob_peer_id)
    sess._set_state(SessionState.ESTABLISHED, None)
    sess.remote_token_exp = 1   # 1970 — far past + grace
    _run(sess.tick_keepalive())
    assert sess.state == SessionState.CLOSING
    last = _last_sent(bob_conn, SESSION_CONTROL_PROTOCOL)
    assert last is not None and last[2].kind == SessMsg.KIND_GOAWAY
    assert last[2].fields["reason"] == TOKEN_EXPIRED


def test_session_manager_pushes_token_update_on_token_change(bob_conn, bob_p2p, bob_peer_id, alice_peer_id):
    """SessionManager.tick pushes a TOKEN_UPDATE to ESTABLISHED sessions when its
    token_factory() (self.node_token) changes; the seeded value is not re-broadcast."""
    from unaiverse.networking.session import SessionManager
    tok = ["tok-1"]
    sm = SessionManager(p2p=bob_conn, p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=alice_peer_id,
                        local_node_id="alice", local_profile_factory=lambda: {},
                        token_factory=lambda: tok[0])
    sess = sm._spawn_responder(bob_peer_id)
    sess._set_state(SessionState.ESTABLISHED, None)

    def _updates():
        return [e[2] for e in bob_conn.sent if e[1] == SESSION_CONTROL_PROTOCOL
                and e[2].kind == SessMsg.KIND_TOKEN_UPDATE]

    _run(sm.tick())
    assert _updates() == []          # no change yet → no push
    tok[0] = "tok-2"
    _run(sm.tick())
    assert _updates() and _updates()[-1].fields["token"] == "tok-2"
