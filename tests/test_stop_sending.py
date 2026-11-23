"""The receiver of a large message stops it, and tells the sender (STOP_SENDING), so the rest does not travel.

STOP_SENDING rides the CONTROL protocol in the small JSON envelope of the liveness messages (no protoc regen). It is
sent only to a peer whose HELLO said CAP_STOP_SENDING, and the sender's transport stops the message it names
(P2P.cancel_send: Go's CancelSend).
"""
import asyncio
from unaiverse.networking.node.connpool import NodeConn
from unaiverse.networking.session.session import Session, SessionRole, SessionState
from unaiverse.networking.session.sess_msg import SessMsg, SESSION_CONTROL_PROTOCOL, CAP_STOP_SENDING, CAP_FILE_STREAM


def _run(coro):
    """Runs a coroutine to its end, in a new event loop."""
    return asyncio.run(coro)


def _established(p2p, peer_id: str, local_peer_id: str, peer_capabilities: int = 0) -> Session:
    """An ESTABLISHED Session with the peer, on the given transport, as if its handshake had just ended."""
    sess = Session(peer_id=peer_id, role=SessionRole.RESPONDER, p2p=p2p, p2p_name=NodeConn.P2P_PUBLIC,
                   local_peer_id=local_peer_id, local_node_id="local-node", local_profile_factory=lambda: {})
    sess._set_state(SessionState.ESTABLISHED, None)
    sess.peer_capabilities = peer_capabilities
    return sess


def _sent_on_control(conn) -> list[SessMsg]:
    """The Session messages sent on the CONTROL protocol, oldest first."""
    return [entry[2] for entry in conn.sent if entry[1] == SESSION_CONTROL_PROTOCOL]


def test_stop_sending_round_trips_on_the_wire_as_json():
    """The message survives the codec: its kind tag, its session id (bytes) and its seq (a number)."""
    msg = SessMsg(SessMsg.KIND_STOP_SENDING, {"session_id": b"\x01" * 16, "seq": 42})
    wire = msg.to_bytes()
    assert wire[0] == 0x10
    back = SessMsg.from_bytes(wire, SESSION_CONTROL_PROTOCOL)
    assert back.kind == SessMsg.KIND_STOP_SENDING
    assert back["session_id"] == b"\x01" * 16 and back["seq"] == 42


def test_stop_sending_is_sent_only_to_a_peer_that_takes_it(bob_conn, bob_peer_id, alice_peer_id):
    """A peer whose HELLO did not say CAP_STOP_SENDING (an older one) is not asked: it could not read the message."""
    old = _established(bob_conn, alice_peer_id, bob_peer_id, peer_capabilities=CAP_FILE_STREAM)
    assert _run(old.send_stop_sending(7)) is False
    assert _sent_on_control(bob_conn) == []

    new = _established(bob_conn, alice_peer_id, bob_peer_id, peer_capabilities=CAP_FILE_STREAM | CAP_STOP_SENDING)
    _run(new.send_stop_sending(7))
    sent = _sent_on_control(bob_conn)
    assert [m.kind for m in sent] == [SessMsg.KIND_STOP_SENDING]
    assert sent[0]["seq"] == 7 and sent[0]["session_id"] == new.session_id
    assert _run(new.send_stop_sending(0)) is False, "seq 0 names no message"


def test_stop_sending_is_not_sent_outside_an_established_session(bob_conn, bob_peer_id, alice_peer_id):
    """A Session that is reconnecting (or closing) has no sender to tell."""
    sess = _established(bob_conn, alice_peer_id, bob_peer_id, peer_capabilities=CAP_STOP_SENDING)
    sess._set_state(SessionState.RECONNECTING, 10.0)
    assert _run(sess.send_stop_sending(7)) is False
    assert _sent_on_control(bob_conn) == []


def test_a_received_stop_sending_stops_the_message_on_the_transport(bob_conn, bob_peer_id, alice_peer_id):
    """The sender's Session hands the seq to its transport (cancel_send), for this peer."""
    stopped = []
    bob_conn.cancel_send = lambda peer_id, seq: stopped.append((peer_id, seq)) or True
    sess = _established(bob_conn, alice_peer_id, bob_peer_id)
    events = _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, SessMsg(SessMsg.KIND_STOP_SENDING, {
        "session_id": sess.session_id, "seq": 9})))
    assert events == []
    assert stopped == [(alice_peer_id, 9)]
    assert sess.state == SessionState.ESTABLISHED


def test_a_stop_sending_of_another_session_or_without_a_transport_that_can_stop_changes_nothing(
        bob_conn, bob_peer_id, alice_peer_id):
    """Another session id is not this one's to obey; a transport without cancel_send just goes on sending."""
    stopped = []
    bob_conn.cancel_send = lambda peer_id, seq: stopped.append((peer_id, seq)) or True
    sess = _established(bob_conn, alice_peer_id, bob_peer_id)
    _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, SessMsg(SessMsg.KIND_STOP_SENDING, {
        "session_id": b"\xff" * 16, "seq": 9})))
    assert stopped == []

    del bob_conn.cancel_send
    _run(sess.process_inbound(SESSION_CONTROL_PROTOCOL, SessMsg(SessMsg.KIND_STOP_SENDING, {
        "session_id": sess.session_id, "seq": 9})))
    assert sess.state == SessionState.ESTABLISHED


def test_the_capability_is_advertised_only_by_a_transport_that_can_stop_a_message(
        bob_conn, bob_peer_id, alice_peer_id):
    """CAP_STOP_SENDING comes from the transport (Go with CancelSend, or the browser), not from this code."""
    sess = _established(bob_conn, alice_peer_id, bob_peer_id)
    assert not sess._local_capabilities() & CAP_STOP_SENDING
    bob_conn.supports_stop_sending = True
    assert sess._local_capabilities() & CAP_STOP_SENDING
