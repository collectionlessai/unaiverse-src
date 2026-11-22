"""
Phase 6.5 of the concurrency plan over the real libp2p library (opt-in: ``-m p2p``): file messages.

Two real nodes in this process, with a Session: each learns CAP_FILE_STREAM from the other's HELLO. A stream_sample
message with a path-backed FILE sample goes through P2P.send / P2P.get_messages as a node sends and receives it: a
512 MB file arrives whole (hash checked) in the downloads folder, with the memory of the process (both ends) staying
far below the size of the file; two files of the same name do not overwrite each other, and a small message sent
after them arrives after them; to a peer without the capability the file goes in memory, inside the message.
"""
import os
import time
import hashlib
import asyncio
import threading
import subprocess
import pytest
from unaiverse.custom import Custom
from unaiverse.networking.p2p import files
from unaiverse.networking.p2p.p2p import P2P
from unaiverse.networking.p2p.messages import Msg
from unaiverse.streams.dataprops import FileContainer
from test_chunked_realnet import _pair, MB
from unaiverse.networking.session.sess_msg import CAP_CHUNKED_FRAMES, CAP_FILE_STREAM

pytestmark = pytest.mark.p2p


def _big_file(path: str, size: int) -> str:
    """Writes a file of random blocks and returns its SHA-256.

    Args:
        path: Where.
        size: Bytes (a multiple of 8 MB, or less than it).

    Returns:
        The hex digest.
    """
    block, h = os.urandom(min(size, 8 * MB)), hashlib.sha256()
    with open(path, "wb") as f:
        for _ in range(max(1, size // len(block))):
            f.write(block)
            h.update(block)
    return h.hexdigest()


def _sha256(path: str) -> str:
    """The SHA-256 of a file, read in blocks.

    Args:
        path: The file.

    Returns:
        The hex digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * MB), b""):
            h.update(block)
    return h.hexdigest()


def _file_msg(sender: str, receiver: str, container: FileContainer, tag: int = 0) -> Msg:
    """A stream_sample message with one FILE sample.

    Args:
        sender: The sending peer.
        receiver: The receiving peer.
        container: The file.
        tag: The data tag.

    Returns:
        The message.
    """
    return Msg(sender=sender, content_type=Msg.STREAM_SAMPLE, channel=f"{sender}::dm:{receiver}-stream",
               content={"doc": {"data": container, "data_tag": tag, "data_uuid": None}})


def _receive(p2p: P2P, n: int, timeout: float) -> list[Msg]:
    """Pops decoded chat messages until n arrived.

    Args:
        p2p: The receiver.
        n: How many.
        timeout: Seconds.

    Returns:
        The messages, in the order they came.
    """
    got, end = [], time.monotonic() + timeout
    while len(got) < n and time.monotonic() < end:
        got += p2p.get_messages()[0]
        time.sleep(0.01)
    return got


class _RssPeak:
    """The peak resident memory of this process while a block runs, sampled with ps every 50 ms."""

    def __enter__(self) -> '_RssPeak':
        """Starts the sampling.

        Returns:
            Itself.
        """
        self.baseline = self.peak = self._rss()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        """Stops the sampling.

        Args:
            *exc: The exception, if any.
        """
        self._stop.set()
        self._thread.join()

    def _sample(self) -> None:
        """Samples until stopped."""
        while not self._stop.wait(0.05):
            self.peak = max(self.peak, self._rss())

    @staticmethod
    def _rss() -> int:
        """The resident memory of this process now.

        Returns:
            Bytes.
        """
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout
        return int(out.strip() or 0) * 1024


def test_both_ends_learn_that_the_other_takes_file_messages(real_nodes: object) -> None:
    """The Go-backed transport advertises CAP_FILE_STREAM, and each Session records what the other said."""
    assert P2P.supports_file_stream and P2P.accepts_file_stream
    a_p2p, a_sm, b_p2p, b_sm = _pair(real_nodes)
    assert a_sm.get_session(b_p2p.peer_id).peer_capabilities & CAP_FILE_STREAM
    assert b_sm.get_session(a_p2p.peer_id).peer_capabilities & CAP_FILE_STREAM


def test_a_large_file_arrives_whole_with_constant_memory(real_nodes: object, tmp_path: object) -> None:
    """512 MB from disk to disk: the hash matches, the file is in Downloads/unknown/<sender>/ (no profile known), the
    incoming folder is empty again, and the process never holds more than a fraction of the file."""
    a_p2p, a_sm, b_p2p, b_sm = _pair(real_nodes)
    a_p2p.set_peer_capabilities(b_p2p.peer_id, CAP_CHUNKED_FRAMES | CAP_FILE_STREAM, None, 0)
    size = 512 * MB
    path = str(tmp_path / "big.bin")
    digest = _big_file(path, size)
    msg = _file_msg(a_p2p.peer_id, b_p2p.peer_id, FileContainer.from_path(path))

    t0, sent_progress, received_progress = time.monotonic(), [], []
    with _RssPeak() as rss:
        seq = asyncio.run(a_p2p.send(msg.channel, msg))
        assert isinstance(seq, int)
        got, end = [], time.monotonic() + 300.0
        while not got and time.monotonic() < end:
            got += b_p2p.get_messages()[0]
            sent_progress += [e for e in a_p2p.pop_send_events() if e.get("type") == "progress"]
            received_progress += [e for e in b_p2p.pop_send_events() if e.get("type") == "receive_progress"]
            time.sleep(0.01)
    took = time.monotonic() - t0
    assert len(got) == 1
    sent_progress += [e for e in a_p2p.pop_send_events() if e.get("type") == "progress"]

    # Both ends report percentages that only grow, up to 100%, with the name of the file
    for events in (sent_progress, received_progress):
        done = [e["bytes"] for e in events]
        assert len(events) >= 2 and done == sorted(done) and done[-1] == events[-1]["total"], done
        assert all(e["seq"] == seq and e["kind"] == "file" and e["files"] == ["big.bin"] for e in events)
    assert sent_progress[-1]["total"] == received_progress[-1]["total"] > size
    data = got[0].content["doc"]["data"]
    assert data.path == os.path.join(files.downloads_dir(), "unknown", a_p2p.peer_id, "big.bin")
    assert not data.has_content and data.size == size and data.filename == "big.bin"
    assert _sha256(data.path) == digest
    assert got[0].content["doc"]["data_tag"] == 0
    assert files.streamed_files(Msg.STREAM_SAMPLE, got[0].content)  # It can be sent on from disk in turn
    incoming = os.path.join(files.incoming_dir(), b_p2p.peer_id)
    assert not os.path.isdir(incoming) or os.listdir(incoming) == []
    growth = rss.peak - rss.baseline
    print(f"\n{size / MB:.0f} MB file in {took:.1f} s, memory peak +{growth / MB:.0f} MB, progress events: "
          f"{[round(100 * e['bytes'] / e['total']) for e in sent_progress]}% sent, "
          f"{[round(100 * e['bytes'] / e['total']) for e in received_progress]}% received")
    assert growth < 160 * MB, f"memory grew by {growth / MB:.0f} MB for a {size / MB:.0f} MB file"
    os.remove(data.path)


def test_files_of_the_same_name_do_not_overwrite_and_order_holds(real_nodes: object, tmp_path: object) -> None:
    """Two files named r.txt (different contents), then a small message: r.txt, r (1).txt, then the message."""
    a_p2p, a_sm, b_p2p, b_sm = _pair(real_nodes)
    a_p2p.set_peer_capabilities(b_p2p.peer_id, CAP_CHUNKED_FRAMES | CAP_FILE_STREAM, None, 0)
    sent = []
    for i, body in enumerate([b"first" * 1000, b"second" * 300000]):
        folder = tmp_path / str(i)
        folder.mkdir()
        (folder / "r.txt").write_bytes(body)
        sent.append(body)
        msg = _file_msg(a_p2p.peer_id, b_p2p.peer_id, FileContainer.from_path(str(folder / "r.txt")), tag=i)
        asyncio.run(a_p2p.send(msg.channel, msg))
    note = Msg(sender=a_p2p.peer_id, content_type=Msg.MISC, channel=f"{a_p2p.peer_id}::dm:{b_p2p.peer_id}-stream",
               content={"after": True})
    asyncio.run(a_p2p.send(note.channel, note))

    got = _receive(b_p2p, 3, 60.0)
    assert [m.content_type for m in got] == [Msg.STREAM_SAMPLE, Msg.STREAM_SAMPLE, Msg.MISC]
    paths = [m.content["doc"]["data"].path for m in got[:2]]
    folder = os.path.join(files.downloads_dir(), "unknown", a_p2p.peer_id)
    assert paths == [os.path.join(folder, "r.txt"), os.path.join(folder, "r (1).txt")]
    assert [open(p, "rb").read() for p in paths] == sent
    for p in paths:
        os.remove(p)


def test_a_peer_without_file_messages_gets_the_file_in_memory(real_nodes: object, tmp_path: object) -> None:
    """The peer said CAP_CHUNKED_FRAMES only: the 3 MB file is read into the message, and arrives as bytes."""
    a_p2p, a_sm, b_p2p, b_sm = _pair(real_nodes)
    a_p2p.set_peer_capabilities(b_p2p.peer_id, CAP_CHUNKED_FRAMES, None, 0)
    path = str(tmp_path / "small.bin")
    _big_file(path, 3 * MB)
    msg = _file_msg(a_p2p.peer_id, b_p2p.peer_id, FileContainer.from_path(path))
    asyncio.run(a_p2p.send(msg.channel, msg))
    got = _receive(b_p2p, 1, 60.0)
    data = got[0].content["doc"]["data"]
    assert data.path is None and data.has_content and data.content == open(path, "rb").read()
    assert Custom.ACCEPT_FILES  # Nothing in this test turned files off: the fallback is the capability alone
