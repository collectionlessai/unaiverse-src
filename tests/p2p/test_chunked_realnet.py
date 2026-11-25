"""
Phase 6 of the concurrency plan over the real libp2p library (opt-in: ``-m p2p``): messages larger than a chunk.

Two real nodes in this process, with a Session: each learns CAP_CHUNKED_FRAMES from the other's HELLO. The transfers
are raw payloads at the P2P level (the transport, not the protobuf codec of 100 MB), after telling Go what the peer
can take as NodeConn does: 4 MB goes as one frame, 10 MB and 100 MB as chunk frames, whole and in order, a small
message waiting behind a large one; a legacy peer (no capability) gets up to the legacy size as one frame and a
refusal above; a message above the maximum of the peer is refused; and a receiver stopped in the middle of a large
message (another process) gets it once and whole after the writer starts it over from its first chunk.
"""
import os
import time
import signal
import pytest
import asyncio
from unaiverse.custom import Custom
from unaiverse.networking.p2p.p2p import P2P, P2PError
from unaiverse.networking.node.connpool import NodeConn
from realnet_helpers import pick_dialable_addr, pump_until
from unaiverse.networking.session.sess_msg import CAP_CHUNKED_FRAMES

pytestmark = pytest.mark.p2p

MB = 1024 * 1024


def _payload(i: int, size: int) -> bytes:
    """A payload carrying its index and its size, then a pattern.

    Args:
        i: The index.
        size: Total bytes.

    Returns:
        The bytes.
    """
    return i.to_bytes(8, "big") + size.to_bytes(8, "big") + bytes(range(256)) * ((size - 16) // 256) + \
        b"x" * ((size - 16) % 256)


def _pair(real_nodes: object) -> tuple:
    """Two real nodes with an established Session.

    Args:
        real_nodes: The factory fixture.

    Returns:
        (alice p2p, alice manager, bob p2p, bob manager).
    """
    a_p2p, a_conn, a_sm = real_nodes.node("alice")
    b_p2p, b_conn, b_sm = real_nodes.node("bob")
    targets = [(a_conn, a_sm, NodeConn.P2P_PUBLIC), (b_conn, b_sm, NodeConn.P2P_PUBLIC)]

    async def establish() -> None:
        """Dials and pumps until both sides are ESTABLISHED (async)."""
        await asyncio.wait_for(a_sm.initiate(b_p2p.peer_id, [pick_dialable_addr(b_p2p)]), timeout=10.0)
        await pump_until(targets, predicate=lambda: a_sm.is_established(b_p2p.peer_id) and
                         b_sm.is_established(a_p2p.peer_id))
    asyncio.run(establish())
    return a_p2p, a_sm, b_p2p, b_sm


def _receive(p2p: P2P, n: int, timeout: float) -> list[bytes]:
    """Pops raw messages until n arrived.

    Args:
        p2p: The receiver.
        n: How many.
        timeout: Seconds.

    Returns:
        Their payloads, in the order they came.
    """
    got, end = [], time.monotonic() + timeout
    while len(got) < n and time.monotonic() < end:
        got += [bytes(m["data"]) for m in p2p.pop_messages()]
        time.sleep(0.01)
    return got


def test_both_ends_learn_the_capability_from_the_hello(real_nodes: object) -> None:
    """The Go-backed transport advertises CAP_CHUNKED_FRAMES, and each Session records what the other said."""
    assert P2P.supports_chunked_frames
    a_p2p, a_sm, b_p2p, b_sm = _pair(real_nodes)
    assert a_sm.get_session(b_p2p.peer_id).peer_capabilities & CAP_CHUNKED_FRAMES
    assert b_sm.get_session(a_p2p.peer_id).peer_capabilities & CAP_CHUNKED_FRAMES


def test_large_messages_arrive_whole_and_in_order(real_nodes: object) -> None:
    """4 MB as one frame, 10 MB and 100 MB as chunks, then a small message: all whole, in order."""
    a_p2p, a_sm, b_p2p, b_sm = _pair(real_nodes)
    a_p2p.set_peer_capabilities(b_p2p.peer_id, CAP_CHUNKED_FRAMES, None)
    channel = f"{a_p2p.peer_id}::dm:{b_p2p.peer_id}-misc"
    sizes = [4 * MB, 10 * MB, 100 * MB, 1000]
    sent = [_payload(i, size) for i, size in enumerate(sizes)]
    t0 = time.monotonic()
    for p in sent:
        assert isinstance(a_p2p.send_message_to_peer(channel, p), int)
    got = _receive(b_p2p, len(sent), 120.0)
    assert [len(g) for g in got] == sizes, [len(g) for g in got]
    assert got == sent
    print(f"\n{sum(sizes) / MB:.0f} MB delivered in {time.monotonic() - t0:.1f} s")


def test_a_legacy_peer_gets_single_frames_and_refusals_above_the_legacy_size(real_nodes: object) -> None:
    """With no capability the peer gets one frame up to the legacy size; a larger message is refused at once."""
    a_p2p, a_sm, b_p2p, b_sm = _pair(real_nodes)
    a_p2p.set_peer_capabilities(b_p2p.peer_id, 0, None)
    channel = f"{a_p2p.peer_id}::dm:{b_p2p.peer_id}-misc"
    small = _payload(0, 10 * MB)
    assert isinstance(a_p2p.send_message_to_peer(channel, small), int)
    with pytest.raises(P2PError):
        a_p2p.send_message_to_peer(channel, _payload(1, Custom.LEGACY_MAX_MESSAGE_SIZE + MB))
    assert _receive(b_p2p, 1, 30.0) == [small]


def test_a_message_above_the_maximum_of_the_peer_is_refused(real_nodes: object) -> None:
    """The peer's profile says 20 MB: 30 MB is refused at once instead of sent and reset."""
    a_p2p, a_sm, b_p2p, b_sm = _pair(real_nodes)
    a_p2p.set_peer_capabilities(b_p2p.peer_id, CAP_CHUNKED_FRAMES, 20 * MB)
    channel = f"{a_p2p.peer_id}::dm:{b_p2p.peer_id}-misc"
    with pytest.raises(P2PError):
        a_p2p.send_message_to_peer(channel, _payload(0, 30 * MB))
    ok = _payload(1, 15 * MB)
    assert isinstance(a_p2p.send_message_to_peer(channel, ok), int)
    assert _receive(b_p2p, 1, 30.0) == [ok]


def test_a_stream_broken_in_the_middle_of_a_large_message(_libp2p_setup: object, tmp_path: object) -> None:
    """The receiver (another process) stops in the middle of a 60 MB message, past the deadline of a chunk: the write
    fails, and after SIGCONT the writer sends the message again from its first chunk; it arrives once, whole."""
    from test_send_queue_realnet import _Receiver, _connect
    sender = P2P(identity_dir=str(tmp_path / "sender"), port=0, ips=["0.0.0.0"], enable_relay_client=False,
                 enable_relay_service=False, use_broad_limits=False, is_isolated=False, knows_is_public=True,
                 enable_tls=False, probe_peers=[], webrtc_enabled=False, flood_sub=False, log_sub="pub")
    receiver = _Receiver()
    try:
        channel = _connect(sender, receiver)
        sender.set_peer_capabilities(receiver.peer_id, CAP_CHUNKED_FRAMES, None)
        big = _payload(0, 60 * MB)
        sender.pop_send_events()

        # Stopped before the send, so that the message cannot be through before the stop, however fast the machine:
        # the buffers of the link take the first chunks, and the write of a later one waits until its deadline
        os.kill(receiver.proc.pid, signal.SIGSTOP)
        sender.send_message_to_peer(channel, big)
        chunk_deadline = Custom.SEND_WRITE_STALL_TIMEOUT  # A write fails once the link took nothing for this long
        end, failed = time.monotonic() + chunk_deadline + 15.0, False
        while time.monotonic() < end and not failed:
            failed = any(e["type"] == "write_failed" for e in sender.pop_send_events())
            time.sleep(0.2)
        assert failed, "the write of a chunk never failed while the receiver was stopped"
        os.kill(receiver.proc.pid, signal.SIGCONT)
        assert receiver.wait_for(1, 90.0)
        time.sleep(2.0)
        assert receiver.got == [0] and receiver.lengths == [len(big)], (receiver.got, receiver.lengths)
    finally:
        receiver.close()
        sender.close()
