"""
Phase 1 of the concurrency plan over the real libp2p library (opt-in: ``-m p2p``): the per-peer send queues.

The receiver is another process (send_queue_peer.py), so that it can be stopped with SIGSTOP (its link stops
draining while the connection stays up) or killed. The sender is a real P2P in this process, with the queues on
(P2P.setup_library passes the limits of Custom to Go).

- a stopped receiver: every send returns at once, the queue refuses at its cap (and keeps refusing until it drains
  to the low-water mark), and after SIGCONT everything accepted arrives, complete and in order;
- a killed receiver: the writer retries, then drops the whole queue (max_attempts), and says so in the events;
- DropSendQueue empties the queue at once (explicit);
- a library whose queues were never configured sends synchronously, as before (what the Tauri plugin relies on).
"""
import os
import sys
import json
import time
import signal
import pytest
import asyncio
import tempfile
import threading
import subprocess
from pathlib import Path
from unaiverse.custom import Custom
from unaiverse.networking.p2p.messages import Msg
from unaiverse.networking.p2p.p2p import P2P, P2PQueueFull

pytestmark = pytest.mark.p2p

PEER_SCRIPT = str(Path(__file__).with_name("send_queue_peer.py"))
MSG_BYTES = 1 << 20  # 1 MiB: well above the stream window, so a stopped receiver blocks the writer at once


class _Receiver:
    """The receiver process: its address, and the indices it reports, collected by a thread."""

    def __init__(self) -> None:
        """Starts the process and reads its address."""
        self.proc = subprocess.Popen([sys.executable, PEER_SCRIPT, "receiver"], stdout=subprocess.PIPE, text=True,
                                     stderr=subprocess.DEVNULL)
        info = json.loads(self.proc.stdout.readline())
        self.peer_id, self.addresses = info["peer_id"], info["addresses"]
        self.got = []
        self.times = []  # Wall-clock time of each arrival, as the receiver saw it
        self.lengths = []  # Bytes of each message, as the receiver got it
        threading.Thread(target=self._read, daemon=True).start()

    def wait_for(self, n: int, timeout: float) -> bool:
        """Waits until n messages were reported.

        Args:
            n: How many.
            timeout: Seconds.

        Returns:
            True if they all arrived in time.
        """
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if len(self.got) >= n:
                return True
            time.sleep(0.05)
        return False

    def close(self) -> None:
        """Kills the process (resuming it first if it was stopped)."""
        try:
            os.kill(self.proc.pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
        self.proc.kill()
        self.proc.wait(timeout=10)

    def _read(self) -> None:
        """Collects the reported indices until the process ends."""
        for line in self.proc.stdout:
            item = json.loads(line)
            self.got.append(item["got"])
            self.times.append(item.get("t"))
            self.lengths.append(item.get("len"))


@pytest.fixture
def sender(_libp2p_setup: object, tmp_path: Path) -> P2P:
    """A real P2P instance with the send queues on.

    Args:
        _libp2p_setup: The session fixture that set up the library.
        tmp_path: The pytest temporary folder.

    Returns:
        The instance (closed at teardown).
    """
    assert P2P._send_queues_enabled, "the library was not configured with send queues (old library?)"
    p2p = P2P(identity_dir=str(tmp_path / "sender"), port=0, ips=["0.0.0.0"], enable_relay_client=False,
              enable_relay_service=False, use_broad_limits=False, is_isolated=False, knows_is_public=True,
              enable_tls=False, probe_peers=[], webrtc_enabled=False, flood_sub=False, log_sub="pub")
    yield p2p
    p2p.close()


@pytest.fixture
def receiver() -> _Receiver:
    """The receiver process.

    Returns:
        The receiver (killed at teardown).
    """
    r = _Receiver()
    yield r
    r.close()


def _connect(p2p: P2P, rx: _Receiver) -> str:
    """Connects the sender to the receiver.

    Args:
        p2p: The sender.
        rx: The receiver.

    Returns:
        The direct-message channel to the receiver.
    """
    peer_id, _ = asyncio.run(p2p.connect(rx.addresses))
    assert peer_id == rx.peer_id
    return f"{p2p.peer_id}::dm:{rx.peer_id}-misc"


def _msg(i: int, size: int = 64) -> bytes:
    """A message carrying its index.

    Args:
        i: The index.
        size: Total bytes (default: 64).

    Returns:
        The bytes.
    """
    return i.to_bytes(8, "big") + b"x" * (size - 8)


def _events_until(p2p: P2P, predicate: object, timeout: float) -> list[dict]:
    """Pops the send events until one of them satisfies the predicate.

    Args:
        p2p: The sender.
        predicate: Called on each event.
        timeout: Seconds.

    Returns:
        All the events popped.
    """
    events, end = [], time.monotonic() + timeout
    while time.monotonic() < end:
        new = p2p.pop_send_events()
        events.extend(new)
        if any(predicate(e) for e in new):
            return events
        time.sleep(0.1)
    return events


def test_stopped_receiver_never_blocks_the_sender_and_gets_everything_in_order(sender: P2P,
                                                                              receiver: _Receiver) -> None:
    """With the receiver stopped, sends return at once until the cap refuses; after SIGCONT all accepted arrive."""
    channel = _connect(sender, receiver)
    for i in range(3):
        assert isinstance(sender.send_message_to_peer(channel, _msg(i)), int)
    assert receiver.wait_for(3, 10.0), receiver.got

    os.kill(receiver.proc.pid, signal.SIGSTOP)
    accepted, slowest, refused_at = [], 0.0, None
    for i in range(3, 3 + 2 * Custom.SEND_QUEUE_MAX_MSGS_PER_PEER):
        t0 = time.perf_counter()
        try:
            sender.send_message_to_peer(channel, _msg(i, MSG_BYTES))
            accepted.append(i)
        except P2PQueueFull:
            refused_at = i
            break
        finally:
            slowest = max(slowest, time.perf_counter() - t0)
    assert refused_at is not None, "the queue never refused"
    assert slowest < 0.5, f"a send to a stopped peer took {slowest:.3f}s"
    assert len(accepted) >= Custom.SEND_QUEUE_MAX_MSGS_PER_PEER - 1, len(accepted)

    # Refusing until it drained to the low-water mark: a small message is refused too (no gap in the sequence)
    with pytest.raises(P2PQueueFull):
        sender.send_message_to_peer(channel, _msg(refused_at))

    os.kill(receiver.proc.pid, signal.SIGCONT)
    assert receiver.wait_for(3 + len(accepted), 60.0), (len(receiver.got), 3 + len(accepted))
    assert receiver.got == list(range(3)) + accepted

    events = _events_until(sender, lambda e: e["type"] == "written" and e["seq"] == 3 + len(accepted), 10.0)
    types = {e["type"] for e in events}
    assert "refused" in types and "queue_dropped" not in types, events


def test_killed_receiver_drops_the_whole_queue_after_the_retries(sender: P2P, receiver: _Receiver) -> None:
    """A dead peer: the writer retries the head, then drops the queue (max_attempts) and reports it."""
    channel = _connect(sender, receiver)
    sender.send_message_to_peer(channel, _msg(0))
    assert receiver.wait_for(1, 10.0)
    receiver.close()
    for i in range(1, 4):
        sender.send_message_to_peer(channel, _msg(i, MSG_BYTES))
    budget = sum(Custom.SEND_RETRY_BACKOFF) + Custom.SEND_MAX_DIAL_ATTEMPTS * 25.0
    events = _events_until(sender, lambda e: e["type"] == "queue_dropped", budget)
    dropped = [e for e in events if e["type"] == "queue_dropped"]
    assert dropped and dropped[0]["reason"] in ("max_attempts", "max_age"), events
    assert any(e["type"] == "write_failed" for e in events), events

    # The next message is accepted again (a new writer, a new dial)
    assert isinstance(sender.send_message_to_peer(channel, _msg(9)), int)


def test_drop_send_queue_empties_it_at_once(sender: P2P, receiver: _Receiver) -> None:
    """DropSendQueue empties the queue of a stopped peer and reports it as explicit."""
    channel = _connect(sender, receiver)
    os.kill(receiver.proc.pid, signal.SIGSTOP)
    for i in range(5):
        sender.send_message_to_peer(channel, _msg(i, MSG_BYTES))
    time.sleep(0.5)
    assert sender.drop_send_queue(receiver.peer_id) >= 4
    events = _events_until(sender, lambda e: e["type"] == "queue_dropped", 5.0)
    assert [e["reason"] for e in events if e["type"] == "queue_dropped"] == ["explicit"], events
    assert sender.drop_send_queue(receiver.peer_id) == 0


def test_unconfigured_library_sends_synchronously(sender: P2P, receiver: _Receiver) -> None:
    """A process whose library never got valid queue limits sends as before: written on return, no sequence."""
    with tempfile.TemporaryDirectory():
        out = subprocess.run([sys.executable, PEER_SCRIPT, "legacy-sender", receiver.addresses[0]],
                             capture_output=True, text=True, timeout=120)
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
    assert lines, out.stderr[-2000:]
    reply = json.loads(lines[-1])
    assert reply == {"seq": None, "queues": False}, reply
    assert receiver.wait_for(1, 10.0) and receiver.got == [0]


def test_a_receiver_stopped_past_the_write_deadline_is_recovered_in_order(sender: P2P, receiver: _Receiver) -> None:
    """Stopped longer than the deadline of a frame: the write fails and is recorded; after SIGCONT the writer retries
    by itself and everything arrives, in order, the failed frame first, with nothing dropped."""
    channel = _connect(sender, receiver)
    sender.send_message_to_peer(channel, _msg(0))
    assert receiver.wait_for(1, 10.0)
    sender.pop_send_events()
    big = 4 << 20  # Deadline: base + 4 MB at the slowest rate allowed
    deadline = Custom.SEND_WRITE_BASE_TIMEOUT + big / Custom.SEND_WRITE_MIN_BYTES_PER_SEC
    os.kill(receiver.proc.pid, signal.SIGSTOP)
    for i in range(1, 4):
        sender.send_message_to_peer(channel, _msg(i, big))
    events = _events_until(sender, lambda e: e["type"] == "write_failed", deadline + 10.0)
    failed = [e for e in events if e["type"] == "write_failed"]
    assert failed and failed[0]["seq"] == 2, events  # seq 1 was the small one, written before the stop
    time.sleep(2.0)
    os.kill(receiver.proc.pid, signal.SIGCONT)
    assert receiver.wait_for(4, 60.0), receiver.got
    assert receiver.got == [0, 1, 2, 3]
    events += _events_until(sender, lambda e: e["type"] == "recovered", 10.0)
    types = [e["type"] for e in events]
    assert "recovered" in types and "queue_dropped" not in types, events


def test_go_keeps_delivering_while_the_sender_holds_the_gil(sender: P2P, receiver: _Receiver) -> None:
    """Phase 0.3 after Phase 1: messages queued just before a pure-Python busy loop of the sender arrive during it
    (the writer goroutines need no Python)."""
    channel = _connect(sender, receiver)
    n, busy_s = 30, 3.0
    for i in range(n):
        assert isinstance(sender.send_message_to_peer(channel, _msg(i, MSG_BYTES)), int)
    t0 = time.time()
    x = 0
    while time.time() - t0 < busy_s:
        x += 1  # Holds the GIL: no await, no sleep, no FFI call
    t1 = time.time()
    assert receiver.wait_for(n, 30.0)
    assert receiver.got == list(range(n))

    # By the end of the loop, not within it: on a fast machine some arrive while the sends are still being queued,
    # before the loop starts, and those were not waiting for Python either
    by_end = [t for t in receiver.times if t <= t1]
    assert len(by_end) >= n * 0.9, f"{len(by_end)} of {n} arrived by the end of the busy loop"


def test_tickets_over_real_libp2p(real_nodes: object) -> None:
    """Two real nodes with a Session: a tracked send ends written; messages still queued when the peer is purged end
    dropped, reason explicit; wait() gives True and False accordingly."""
    from realnet_helpers import pick_dialable_addr, pump_until
    from unaiverse.networking.node.connpool import NodeConn
    a_p2p, a_conn, a_sm = real_nodes.node("alice")
    b_p2p, b_conn, b_sm = real_nodes.node("bob")
    a_conn.set_token("alice-token")
    targets = [(a_conn, a_sm, NodeConn.P2P_PUBLIC), (b_conn, b_sm, NodeConn.P2P_PUBLIC)]

    async def scenario() -> tuple:
        """Establishes the Session, then sends tracked messages (async).

        Returns:
            The ticket of the first send, what its wait gave, and the tickets of the purged ones.
        """
        await asyncio.wait_for(a_sm.initiate(b_p2p.peer_id, [pick_dialable_addr(b_p2p)]), timeout=10.0)
        await pump_until(targets, predicate=lambda: a_sm.is_established(b_p2p.peer_id))
        first = await a_conn.send_tracked(b_p2p.peer_id, None, Msg.MISC, {"hello": 1})
        assert first is not None and first.status == "queued"

        async def network() -> None:
            """Pops the send events as the network passes do (async)."""
            while True:
                a_conn.pop_send_events()
                await asyncio.sleep(0.01)
        net = asyncio.create_task(network())
        written = await first.wait(10.0)

        # Big messages, then the purge right away: what is still queued is dropped
        big = [await a_conn.send_tracked(b_p2p.peer_id, None, Msg.MISC, {"blob": "x" * (8 << 20)}) for _ in range(4)]
        await a_conn.drop_peer(b_p2p.peer_id, reason=None)
        results = [await t.wait(10.0) for t in big if t is not None]
        net.cancel()
        return first, written, big, results

    first, written, big, results = asyncio.run(scenario())
    assert written is True and first.status == "written"
    statuses = [t.status for t in big if t is not None]
    assert "dropped" in statuses and all(s in ("written", "dropped") for s in statuses), statuses
    assert all(t.reason == "explicit" for t in big if t is not None and t.status == "dropped")
    assert results == [t.status == "written" for t in big if t is not None]
