"""
Phase 1 of the concurrency plan, Python side of the send queues (no real network: see p2p/test_send_queue_realnet.py).

What is checked here:
- P2P: the replies of Go are mapped (a sequence number, P2PQueueFull with queue_full, other errors), and drop / pop
  are no-ops while the queues are off;
- NodeConn: a refusal is False with a rate-limited log, pop_send_events tags the instance, a transport without
  queues (the browser) gives no events, drop_peer empties the queue unless a GOAWAY is in flight;
- AgentBasics: send_status follows the events and the on_send_event hook runs, a raising hook does no harm;
- the call sites that used to purge on a False keep the peer when the False is a refusal;
- Session: an early keep-alive probe makes the next PING due at once.
"""
import time
import types
import asyncio
from node_harness import WireP2P
from unaiverse.world import World
from unittest.mock import AsyncMock
from unaiverse.agent_basics import AgentBasics
from unaiverse.networking.p2p.messages import Msg
from unaiverse.networking.node.connpool import NodeConn
from unaiverse.networking.p2p.p2p import P2P, P2PError, P2PQueueFull
from unaiverse.networking.session.session import Session, SessionState

PEER = "12D3KooWPeerOfTheSendQueueTests"


class _FakeTypeInterface:
    """The conversions of TypeInterface, as identities; from_go_ptr_to_json returns what the fake lib returned."""

    def to_go_int(self, v: int) -> int:
        """Passes an int through.

        Args:
            v: The value.

        Returns:
            The same value.
        """
        return v

    def to_go_string(self, v: str) -> str:
        """Passes a string through.

        Args:
            v: The value.

        Returns:
            The same value.
        """
        return v

    def to_go_bytes(self, v: bytes) -> bytes:
        """Passes bytes through.

        Args:
            v: The value.

        Returns:
            The same value.
        """
        return v

    def from_go_ptr_to_json(self, v: dict) -> dict:
        """Passes the reply through (the fake lib returns dicts, not pointers).

        Args:
            v: The reply.

        Returns:
            The same reply.
        """
        return v


def _p2p_with_replies(monkeypatch: object, replies: dict, enabled: bool = True) -> P2P:
    """Builds a P2P whose Go calls return canned replies (no library, no __init__).

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        replies: Export name -> the reply dict it returns.
        enabled: The value of P2P._send_queues_enabled (default: True).

    Returns:
        The P2P instance.
    """
    lib = types.SimpleNamespace(**{name: (lambda *a, _r=r: _r) for name, r in replies.items()})
    monkeypatch.setattr(P2P, "libp2p", lib, raising=False)
    monkeypatch.setattr(P2P, "_type_interface", _FakeTypeInterface(), raising=False)
    monkeypatch.setattr(P2P, "_send_queues_enabled", enabled)
    p2p = object.__new__(P2P)
    p2p._instance = 0
    return p2p


def test_queued_send_returns_its_sequence_number(monkeypatch: object) -> None:
    """A queued message gives back the sequence number Go assigned; a synchronous send gives None."""
    p2p = _p2p_with_replies(monkeypatch, {"SendMessageToPeer": {"state": "Success", "message": "queued", "seq": 7}})
    assert p2p.send_message_to_peer(f"me::dm:{PEER}-x", b"data") == 7
    p2p = _p2p_with_replies(monkeypatch, {"SendMessageToPeer": {"state": "Success", "message": "sent"}})
    assert p2p.send_message_to_peer(f"me::dm:{PEER}-x", b"data") is None


def test_full_queue_raises_queue_full_and_other_errors_do_not(monkeypatch: object) -> None:
    """queue_full becomes P2PQueueFull (a P2PError with queue_full and the peer); any other error a plain P2PError."""
    p2p = _p2p_with_replies(monkeypatch, {"SendMessageToPeer": {"state": "Error", "code": "queue_full",
                                                                "message": "send queue full (peer_msgs)"}})
    try:
        p2p.send_message_to_peer(f"me::dm:{PEER}-x", b"data")
        raise AssertionError("no exception")
    except P2PQueueFull as e:
        assert e.queue_full and e.peer_id == PEER and isinstance(e, P2PError)
    p2p = _p2p_with_replies(monkeypatch, {"SendMessageToPeer": {"state": "Error", "code": "too_large",
                                                                "message": "too large"}})
    try:
        p2p.send_message_to_peer(f"me::dm:{PEER}-x", b"data")
        raise AssertionError("no exception")
    except P2PError as e:
        assert not getattr(e, "queue_full", False)


def test_drop_and_pop_follow_the_switch(monkeypatch: object) -> None:
    """With the queues off, drop and pop do not call Go; with them on, the replies are decoded."""
    replies = {"DropSendQueue": {"state": "Success", "message": 3},
               "PopSendEvents": {"state": "Success", "message": [{"type": "written", "peer_id": PEER, "seq": 2}]}}
    p2p = _p2p_with_replies(monkeypatch, {}, enabled=False)
    assert p2p.drop_send_queue(PEER) == 0 and p2p.pop_send_events() == []
    p2p = _p2p_with_replies(monkeypatch, replies)
    assert p2p.drop_send_queue(PEER) == 3
    assert p2p.pop_send_events() == [{"type": "written", "peer_id": PEER, "seq": 2}]
    p2p = _p2p_with_replies(monkeypatch, {"PopSendEvents": {"state": "Empty"}})
    assert p2p.pop_send_events() == []


class _QueuedP2P(WireP2P):
    """A WireP2P with a fake send queue: send can be told to refuse, and drop / pop are recorded."""

    _send_queues_enabled = True

    def __init__(self, peer_id: str, log_sub: str) -> None:
        """Initializes the transport.

        Args:
            peer_id: The peer ID of this instance.
            log_sub: The log sub-channel.
        """
        super().__init__(peer_id, log_sub)
        self.refuse = False
        self.fail = None  # An error message to raise instead of queuing
        self.next_seq = 0
        self.events = []
        self.dropped = []

    async def send(self, channel: str, msg: object) -> int:
        """Queues (or refuses) a direct message (async).

        Args:
            channel: The channel.
            msg: The message.

        Returns:
            A sequence number.
        """
        if self.refuse:
            raise P2PQueueFull("send queue full (peer_msgs)", PEER)
        if self.fail is not None:
            raise P2PError(self.fail)
        self.next_seq += 1
        return self.next_seq

    def drop_send_queue(self, peer_id: str) -> int:
        """Records the drop.

        Args:
            peer_id: The peer.

        Returns:
            One dropped message.
        """
        self.dropped.append(peer_id)
        return 1

    def pop_send_events(self) -> list[dict]:
        """Pops the planted events.

        Returns:
            The events.
        """
        events, self.events = self.events, []
        return events


def _conn(queued: bool = True) -> tuple[NodeConn, WireP2P, WireP2P]:
    """Builds a real NodeConn on two transports, with or without send queues.

    Args:
        queued: True for transports with send queues, False for plain ones (like the browser).

    Returns:
        The conn and its public and world transports.
    """
    cls = _QueuedP2P if queued else WireP2P
    p2p_u, p2p_w = cls("12D3KooWSelfPublic", "pub"), cls("12D3KooWSelfWorld", "prv")
    conn = NodeConn(max_connections=8, is_world_node=False, public_key=None, token="t", local_node_id="me")
    conn.attach_session_policy(p2p_u=p2p_u, p2p_w=p2p_w, local_profile_factory=lambda: {},
                               decision_hook=lambda *a: True)
    return conn, p2p_u, p2p_w


def test_refused_send_is_false_and_logged_at_a_limited_rate(monkeypatch: object) -> None:
    """A full queue makes send return False; the refusals are logged once per interval with their count."""
    conn, p2p_u, _ = _conn()
    sm = conn._sm_public
    monkeypatch.setattr(conn, "_sm_of", lambda peer_id: sm)
    logged = []
    monkeypatch.setattr("unaiverse.networking.node.connpool.log",
                        types.SimpleNamespace(network=lambda m, **kw: logged.append(m),
                                              error=lambda m, **kw: logged.append("ERROR " + m)))
    assert asyncio.run(conn.send(PEER, None, "misc", {})) is True
    p2p_u.refuse = True
    results = [asyncio.run(conn.send(PEER, None, "misc", {})) for _ in range(5)]
    assert results == [False] * 5
    refusals = [m for m in logged if "full" in m]
    assert len(refusals) == 1 and not any(m.startswith("ERROR") for m in logged), logged
    assert conn.sends_are_queued


def test_pop_send_events_tags_the_instance_and_skips_plain_transports() -> None:
    """Events come tagged with their instance; transports without queues (the browser) give none."""
    conn, p2p_u, p2p_w = _conn()
    p2p_u.events = [{"type": "written", "peer_id": PEER, "seq": 1}]
    p2p_w.events = [{"type": "refused", "peer_id": PEER, "count": 2}]
    events = conn.pop_send_events()
    assert [(e["type"], e["p2p_name"]) for e in events] == [("written", NodeConn.P2P_PUBLIC),
                                                            ("refused", NodeConn.P2P_WORLD)]
    conn, _, _ = _conn(queued=False)
    assert conn.pop_send_events() == [] and not conn.sends_are_queued


def test_drop_peer_empties_the_queue_unless_a_goaway_is_in_flight(monkeypatch: object) -> None:
    """A purge drops the queue of the peer, except while its GOAWAY is in flight; the settled purge drops it then."""
    conn, p2p_u, _ = _conn()
    sm = conn._sm_public
    monkeypatch.setattr(conn, "_sm_of", lambda peer_id: sm)
    monkeypatch.setattr(sm, "drop", AsyncMock(return_value=[]))

    # A peer with no farewell in flight: dropped at once
    monkeypatch.setattr(sm, "is_tearing_down", lambda peer_id: False)
    asyncio.run(conn.drop_peer(PEER, reason=None))
    assert p2p_u.dropped == [PEER]

    # A GOAWAY in flight: the queue keeps draining on the live link...
    monkeypatch.setattr(sm, "is_tearing_down", lambda peer_id: True)
    asyncio.run(conn.drop_peer(PEER, reason=1))
    assert p2p_u.dropped == [PEER]

    # ...until the settled purge that follows the CLOSED
    asyncio.run(conn.drop_peer(PEER, settled=True))
    assert p2p_u.dropped == [PEER, PEER]


def _agent() -> AgentBasics:
    """Builds the slice of AgentBasics that the send events touch.

    Returns:
        An AgentBasics with only send_status set.
    """
    agent = object.__new__(AgentBasics)
    agent.send_status = {}
    return agent


def test_send_status_follows_the_events_and_the_hook_runs() -> None:
    """Each event type moves send_status as documented, and on_send_event sees every event of a peer."""
    agent = _agent()
    seen = []

    async def hook(peer_id: str, event: dict) -> None:
        """Records the event.

        Args:
            peer_id: The peer.
            event: The event.
        """
        seen.append((peer_id, event["type"]))

    agent.on_send_event = hook
    events = [{"type": "written", "peer_id": PEER, "seq": 3, "time": 1.0},
              {"type": "write_failed", "peer_id": PEER, "seq": 4, "attempt": 1, "error": "reset", "time": 2.0},
              {"type": "recovered", "peer_id": PEER, "seq": 4, "count": 1, "time": 3.0},
              {"type": "refused", "peer_id": PEER, "count": 5, "time": 4.0},
              {"type": "queue_dropped", "peer_id": PEER, "count": 7, "reason": "max_age", "error": "eof", "time": 5.0},
              {"type": "events_lost", "count": 9, "time": 6.0}]
    states = []
    for ev in events:
        asyncio.run(agent._handle_send_event(ev))
        states.append(agent.send_status[PEER]["state"])
    st = agent.send_status[PEER]
    assert states == ["ok", "failing", "ok", "ok", "dropped", "dropped"]
    assert st["last_written_seq"] == 3 and st["refused"] == 5 and st["dropped"] == 7
    assert st["last_error"] == "eof" and st["last_change"] == 5.0
    assert seen == [(PEER, t) for t in ("written", "write_failed", "recovered", "refused", "queue_dropped")]

    # After a drop the failures start from zero: the next write is the recovery
    asyncio.run(agent._handle_send_event({"type": "written", "peer_id": PEER, "seq": 12, "time": 7.0}))
    assert agent.send_status[PEER]["state"] == "ok"


def test_a_raising_hook_does_not_break_the_bookkeeping() -> None:
    """An on_send_event that raises is caught: send_status is updated all the same."""
    agent = _agent()

    async def hook(peer_id: str, event: dict) -> None:
        """Raises.

        Args:
            peer_id: The peer.
            event: The event.
        """
        raise RuntimeError("bug in the world code")

    agent.on_send_event = hook
    asyncio.run(agent._handle_send_event({"type": "refused", "peer_id": PEER, "count": 2, "time": 1.0}))
    assert agent.send_status[PEER]["refused"] == 2


def test_profile_resend_keeps_a_peer_whose_queue_is_full() -> None:
    """send_profile_to_all removes a peer on a failed write, but not on the refusal of a full queue."""
    for queued, removed in ((True, []), (False, [PEER])):
        agent = object.__new__(AgentBasics)
        agent.all_agents = {PEER: None}
        agent.node_profile = types.SimpleNamespace(get_all_profile=lambda: {})
        agent.node_conn = types.SimpleNamespace(send=AsyncMock(return_value=False), sends_are_queued=queued)
        gone = []
        agent.remove_agent = AsyncMock(side_effect=lambda pid: gone.append(pid))
        asyncio.run(AgentBasics.send_profile_to_all(agent))
        assert gone == removed


def test_role_change_refused_keeps_the_agent() -> None:
    """set_role purges the agent on a failed write, but not on the refusal of a full queue."""
    for queued, purged in ((True, False), (False, True)):
        me = types.SimpleNamespace(is_world=True, ROLE_BITS_TO_STR={4: "r"}, role_to_behav={"r": "b"},
                                   node_purge_fcn=AsyncMock(), role_changed_by_world=False,
                                   node_conn=types.SimpleNamespace(get_role=lambda pid: 1,
                                                                   set_role=lambda pid, role: None,
                                                                   send=AsyncMock(return_value=False),
                                                                   sends_are_queued=queued))
        asyncio.run(World.set_role(me, PEER, 5))
        assert me.node_purge_fcn.await_count == (1 if purged else 0)


def test_early_probe_makes_the_next_ping_due() -> None:
    """request_early_ping makes an ESTABLISHED Session PING on its next tick; it does nothing in other states."""
    sess = object.__new__(Session)
    sess.state, sess._ping_in_flight, sess._last_ping_at = SessionState.ESTABLISHED, False, 1e12
    sess.request_early_ping()
    assert sess._last_ping_at == float("-inf")
    sess.state, sess._last_ping_at = SessionState.RECONNECTING, 5.0
    sess.request_early_ping()
    assert sess._last_ping_at == 5.0


def test_pops_follow_the_kind_and_fall_back_without_split_pops(monkeypatch: object) -> None:
    """The network pass and the tick call their own export; a library without them gets PopMessages and no streams."""
    calls = []

    def lib_call(name: str) -> object:
        """Makes a fake export that records its name.

        Args:
            name: The export name.

        Returns:
            The fake export.
        """
        return lambda *a: calls.append(name) or 0

    lib = types.SimpleNamespace(PopMessages=lib_call("all"), PopNetworkMessages=lib_call("network"),
                                PopStreamMessages=lib_call("streams"))
    monkeypatch.setattr(P2P, "libp2p", lib, raising=False)
    tif = _FakeTypeInterface()
    tif.from_go_pop_messages = lambda ptr: {"state": "Empty"}
    monkeypatch.setattr(P2P, "_type_interface", tif, raising=False)
    p2p = object.__new__(P2P)
    p2p._instance = 0
    monkeypatch.setattr(P2P, "supports_split_pop", True)
    for kind in ("all", "network", "streams"):
        assert p2p.pop_messages(kind) == []
    assert calls == ["all", "network", "streams"]
    calls.clear()
    monkeypatch.setattr(P2P, "supports_split_pop", False)
    assert p2p.pop_messages("network") == [] and p2p.pop_messages("streams") == []
    assert calls == ["all"]


def test_a_transport_without_split_pops_gets_everything_with_the_network_pass() -> None:
    """The SessionManager serves "network" as everything and "streams" as nothing on a browser-like transport."""
    from unaiverse.networking.session.session_manager import SessionManager
    sm = object.__new__(SessionManager)
    sm.p2p = types.SimpleNamespace(get_messages=lambda *a: (["chat"], ["sess"]))
    assert sm.get_messages("network") == (["chat"], ["sess"])
    assert sm.get_messages("streams") == ([], [])
    sm.p2p = types.SimpleNamespace(supports_split_pop=True, get_messages=lambda kind="all": ([kind], []))
    assert sm.get_messages("streams") == (["streams"], [])


def test_a_rejected_interaction_completes_at_its_sender_with_rejected() -> None:
    """reject_received_interaction sends a COMPLETED status with reason REJECTED for the interaction's uuid."""
    sent = []

    async def send(peer_id: str, channel_trail: object, content: dict, content_type: str) -> bool:
        """Records the send (async).

        Args:
            peer_id: The peer.
            channel_trail: The trail.
            content: The content.
            content_type: The content type.

        Returns:
            True.
        """
        sent.append((peer_id, content_type, content))
        return True

    agent = object.__new__(AgentBasics)
    agent.public_agents = {PEER: None}
    agent.get_peer_ids = lambda: ("me-public", "me-private")
    agent.node_conn = types.SimpleNamespace(send=send)
    asyncio.run(agent.reject_received_interaction(PEER, {"uuid": "u-7", "action_name": "learn"}))
    (peer_id, content_type, content), = sent
    assert peer_id == PEER and content_type == Msg.INTERACTION_STATUS
    assert content["uuid"] == "u-7" and content["status_generated_by"] == "me-public"
    assert content["status"] == "completed" and content["completion_reason"] == "rejected"


class _SyncP2P(WireP2P):
    """A transport without send queues (as the browser one): its send returns once written, with no number."""

    async def send(self, channel: str, msg: object) -> None:
        """Writes at once (async).

        Args:
            channel: The channel.
            msg: The message.
        """


def _tracked_conn(monkeypatch: object, queued: bool = True) -> tuple:
    """A conn whose public instance serves PEER.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        queued: True for a transport with send queues.

    Returns:
        The conn and its public transport.
    """
    if queued:
        conn, p2p_u, _ = _conn()
    else:
        p2p_u, p2p_w = _SyncP2P("12D3KooWSelfPublic", "pub"), _SyncP2P("12D3KooWSelfWorld", "prv")
        conn = NodeConn(max_connections=8, is_world_node=False, public_key=None, token="t", local_node_id="me")
        conn.attach_session_policy(p2p_u=p2p_u, p2p_w=p2p_w, local_profile_factory=lambda: {},
                                   decision_hook=lambda *a: True)
    sm = conn._sm_public
    monkeypatch.setattr(conn, "_sm_of", lambda peer_id: sm)
    return conn, p2p_u


def test_a_tracked_send_is_written_when_the_queue_says_so(monkeypatch: object) -> None:
    """Queued tickets resolve as written by a "written up to" event, all those up to that number at once."""
    conn, p2p_u = _tracked_conn(monkeypatch)

    async def scenario() -> tuple:
        """Sends three, then the queue reports the first two written (async).

        Returns:
            The statuses after the event, and what wait() gives.
        """
        tickets = [await conn.send_tracked(PEER, None, "misc", {"i": i}) for i in range(3)]
        assert [t.seq for t in tickets] == [1, 2, 3] and all(t.status == "queued" for t in tickets)
        p2p_u.events = [{"type": "written", "peer_id": PEER, "seq": 2}]
        conn.pop_send_events()
        return [t.status for t in tickets], await tickets[1].wait(1.0), await tickets[2].wait(0.05)

    statuses, second, third = asyncio.run(scenario())
    assert statuses == ["written", "written", "queued"] and second is True and third is False


def test_a_dropped_queue_drops_its_tickets_with_the_reason(monkeypatch: object) -> None:
    """A queue_dropped event resolves the tickets of its range as dropped, with the reason; wait() gives False."""
    conn, p2p_u = _tracked_conn(monkeypatch)

    async def scenario() -> tuple:
        """Sends two, then the queue is dropped (async).

        Returns:
            Statuses, reasons and what wait() gives.
        """
        tickets = [await conn.send_tracked(PEER, None, "misc", {}) for _ in range(2)]
        p2p_u.events = [{"type": "queue_dropped", "peer_id": PEER, "from_seq": 1, "to_seq": 2, "count": 2,
                         "reason": "max_attempts"}]
        conn.pop_send_events()
        return [(t.status, t.reason) for t in tickets], await tickets[0].wait(1.0)

    states, waited = asyncio.run(scenario())
    assert states == [("dropped", "max_attempts")] * 2 and waited is False
    assert conn._tickets == {}


def test_events_popped_before_the_ticket_exists_still_resolve_it(monkeypatch: object) -> None:
    """While a send is still returning its number, the network task may pop its events: the ticket resolves at once."""
    conn, p2p_u = _tracked_conn(monkeypatch)

    async def scenario() -> tuple:
        """Pops the events of the next two messages before sending them (async).

        Returns:
            The two tickets.
        """
        p2p_u.events = [{"type": "written", "peer_id": PEER, "seq": 1},
                        {"type": "queue_dropped", "peer_id": PEER, "from_seq": 2, "to_seq": 2, "reason": "max_age"}]
        conn.pop_send_events()
        return await conn.send_tracked(PEER, None, "misc", {}), await conn.send_tracked(PEER, None, "misc", {})

    first, second = asyncio.run(scenario())
    assert first.status == "written" and (second.status, second.reason) == ("dropped", "max_age")


def test_refused_failed_and_unqueued_sends(monkeypatch: object) -> None:
    """None when refused; dropped with the error when the send failed; written at once without send queues."""
    conn, p2p_u = _tracked_conn(monkeypatch)
    p2p_u.refuse = True
    assert asyncio.run(conn.send_tracked(PEER, None, "misc", {})) is None
    p2p_u.refuse, p2p_u.fail = False, "boom"
    failed = asyncio.run(conn.send_tracked(PEER, None, "misc", {}))
    assert failed.status == "dropped" and "boom" in failed.reason
    conn2, _ = _tracked_conn(monkeypatch, queued=False)
    written = asyncio.run(conn2.send_tracked(PEER, None, "misc", {}))
    assert written.status == "written" and written.seq is None and asyncio.run(written.wait(0.01)) is True


def test_an_action_awaiting_a_ticket_wakes_up_when_the_network_side_pops_its_event(monkeypatch: object) -> None:
    """An action awaits ticket.wait() while another task (the network one) pops the events: it wakes up at once."""
    conn, p2p_u = _tracked_conn(monkeypatch)

    async def scenario() -> tuple:
        """Runs the waiting action next to the popping network task (async).

        Returns:
            What wait() gave and how long it took.
        """
        ticket = await conn.send_tracked(PEER, None, "misc", {})

        async def network() -> None:
            """Pops the events after a while, as the network passes do (async)."""
            await asyncio.sleep(0.3)
            p2p_u.events = [{"type": "written", "peer_id": PEER, "seq": ticket.seq}]
            conn.pop_send_events()
        net = asyncio.create_task(network())
        t0 = time.monotonic()
        ok = await ticket.wait(5.0)
        await net
        return ok, time.monotonic() - t0

    ok, elapsed = asyncio.run(scenario())
    assert ok is True and 0.25 <= elapsed < 1.0


def test_a_started_session_tells_the_transport_what_its_peer_can_receive() -> None:
    """ESTABLISHED / RESUMED pass the peer's capability bits and the maximum of its profile to its transport."""
    from unaiverse.networking.session import SessionEvent, SessionEventType, SessionRole
    conn, p2p_u, _ = _conn()
    told = []
    p2p_u.set_peer_capabilities = lambda peer_id, caps, max_size, max_file: told.append((peer_id, caps, max_size,
                                                                                          max_file))

    def ev(kind: object, data: dict) -> SessionEvent:
        """An event of the public instance.

        Args:
            kind: The event type.
            data: Its data.

        Returns:
            The event.
        """
        return SessionEvent(type=kind, session_id="00", peer_id=PEER, role=SessionRole.INITIATOR,
                            p2p_name=NodeConn.P2P_PUBLIC, data=data)

    conn._tell_transports_about_peers([
        ev(SessionEventType.ESTABLISHED, {"profile_b": {"dynamic": {"max_message_size": 123, "max_file_size": 7}},
                                          "peer_capabilities": 3}),
        ev(SessionEventType.RESUMED, {"profile": {"dynamic": {}}, "peer_capabilities": 0}),
        ev(SessionEventType.LOST, {"peer_capabilities": 1}),
    ])
    assert told == [(PEER, 3, 123, 7), (PEER, 0, None, None)]
