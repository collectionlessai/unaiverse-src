"""
Phase 6.5 of the concurrency plan, without the network: the Python side of the file messages.

A path-backed FileContainer does not read its file until asked; a message leaves the content of its streamed FILE
samples out and gets their files back on arrival; P2P.send streams the path-backed files to a peer with
CAP_FILE_STREAM and reads them into the message for any other; a received file goes to
Downloads/<handle>/<node name>/ under a safe name, never over an existing file, and a message that cannot be
delivered leaves no file behind; the pop reply carries the files of a message.
"""
import os
import json
import struct
import pytest
import asyncio
from unaiverse.custom import Custom
from unaiverse.networking.p2p import files
from unaiverse.networking.p2p.messages import Msg
from unaiverse.streams.dataprops import FileContainer
from unaiverse.networking.p2p.lib_types import TypeInterface
from unaiverse.networking.p2p.p2p import P2P, P2PNoFileStream
from unaiverse.networking.session.sess_msg import CAP_FILE_STREAM

PEER = "12D3KooWEXse6zf6g6rsFckJqXA6SWWrENNB3oN8tG5qPEPWV5do"


@pytest.fixture
def downloads(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> str:
    """A downloads folder of its own for a test.

    Args:
        tmp_path: The folder of the test.
        monkeypatch: To set Custom.DOWNLOADS_DIR.

    Returns:
        The folder.
    """
    folder = os.path.join(str(tmp_path), "Downloads")
    monkeypatch.setattr(Custom, "DOWNLOADS_DIR", folder)
    return folder


def _file(tmp_path: object, name: str, data: bytes) -> str:
    """Writes a file for a test.

    Args:
        tmp_path: The folder of the test.
        name: Its name.
        data: Its bytes.

    Returns:
        Its path.
    """
    path = os.path.join(str(tmp_path), name)
    with open(path, "wb") as f:
        f.write(data)
    return path


def _sample_msg(container: FileContainer) -> Msg:
    """A stream_sample message with one FILE sample, "doc".

    Args:
        container: The file.

    Returns:
        The message.
    """
    return Msg(sender=PEER, content_type=Msg.STREAM_SAMPLE, channel=f"me::dm:{PEER}-s",
               content={"doc": {"data": container, "data_tag": 3, "data_uuid": None}})


def test_a_path_backed_container_reads_its_file_only_when_asked(tmp_path: object) -> None:
    """from_path keeps the path; size needs no read; content reads the file once."""
    path = _file(tmp_path, "report.pdf", b"%PDF-1.7 data")
    c = FileContainer.from_path(path)
    assert c.path == os.path.abspath(path) and not c.has_content and c.size == 13
    assert c.filename == "report.pdf" and c.mime_type == "application/pdf"
    assert c.content == b"%PDF-1.7 data" and c.has_content
    assert c == FileContainer(content=b"%PDF-1.7 data", filename="report.pdf", mime_type="application/pdf")
    with pytest.raises(FileNotFoundError):
        FileContainer.from_path(os.path.join(str(tmp_path), "missing"))
    with pytest.raises(ValueError):
        FileContainer(filename="x")


def test_only_path_backed_file_samples_are_streamed(tmp_path: object) -> None:
    """A FILE sample with a path is streamed; one with its bytes, or another kind of message, is not."""
    path = _file(tmp_path, "a.bin", b"abc")
    content = {"doc": {"data": FileContainer.from_path(path)}, "mem": {"data": FileContainer(content=b"x")},
               "txt": {"data": "hello"}}
    assert files.streamed_files(Msg.STREAM_SAMPLE, content) == [{"key": "doc", "path": os.path.abspath(path),
                                                                  "name": "a.bin"}]
    assert files.streamed_files(Msg.MISC, {"doc": {"data": FileContainer.from_path(path)}}) == []


def test_a_streamed_sample_travels_without_content_and_gets_its_file_back(tmp_path: object) -> None:
    """to_bytes leaves the content out; attach_received_files gives the sample a path-backed container with the name
    the sender gave."""
    path = _file(tmp_path, "a.bin", b"x" * 1000)
    msg = _sample_msg(FileContainer.from_path(path))
    full, light = msg.to_bytes(), msg.to_bytes(streamed_keys={"doc"})
    assert len(light) < len(full) - 900
    received = Msg.from_bytes(light)
    placed = _file(tmp_path, "a (1).bin", b"x" * 1000)
    received.attach_received_files({"doc": placed})
    data = received.content["doc"]["data"]
    assert data.path == placed and data.filename == "a.bin" and data.content == b"x" * 1000
    assert received.content["doc"]["data_tag"] == 3


@pytest.mark.parametrize("name,expected", [
    ("report.pdf", "report.pdf"), ("../../etc/passwd", "_.._etc_passwd"), ("/etc/x", "_etc_x"),
    ("", "file"), ("..", "file"), (".hidden", "hidden"), ("a\x00b\nc", "a_b_c"), ("C:\\x.txt", "C__x.txt"),
])
def test_names_from_the_peer_are_made_safe(name: str, expected: str) -> None:
    """No separators, no "..", no control characters, no hidden files, never empty."""
    assert files.safe_name(name) == expected


def test_a_received_file_goes_to_the_folder_of_its_sender_and_never_over_another(tmp_path: object,
                                                                                 downloads: str) -> None:
    """Downloads/<handle>/<node name>/<name>, then "<name> (1)"; an unknown sender gets unknown/<peer ID>; a hostile
    name stays inside Downloads."""
    first = files.place_received_file(_file(tmp_path, "p1", b"1"), ("alice", "agent"), PEER, "r.txt")
    second = files.place_received_file(_file(tmp_path, "p2", b"2"), ("alice", "agent"), PEER, "r.txt")
    assert first == os.path.join(downloads, "alice", "agent", "r.txt")
    assert second == os.path.join(downloads, "alice", "agent", "r (1).txt")
    assert open(first, "rb").read() == b"1" and open(second, "rb").read() == b"2"
    hostile = files.place_received_file(_file(tmp_path, "p3", b"3"), ("../..", ".."), PEER, "../../x")
    assert os.path.realpath(hostile).startswith(os.path.realpath(downloads) + os.sep)
    unknown = files.place_received_file(_file(tmp_path, "p4", b"4"), None, PEER, "r.txt")
    assert unknown == os.path.join(downloads, "unknown", PEER, "r.txt")


def _bare_p2p(monkeypatch: pytest.MonkeyPatch) -> P2P:
    """A P2P object with no Go instance behind it, that streams files.

    Args:
        monkeypatch: To turn file streaming on for the class.

    Returns:
        The object.
    """
    p2p = object.__new__(P2P)
    p2p._peer_caps, p2p.file_owner_of, p2p._instance = {}, None, 0
    monkeypatch.setattr(P2P, "supports_file_stream", True)
    return p2p


def test_send_streams_the_files_to_a_peer_that_takes_them_and_reads_them_for_any_other(
        tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """CAP_FILE_STREAM: a file message without the content; no capability, or a refusal of Go: the ordinary message,
    with the content read from disk."""
    p2p = _bare_p2p(monkeypatch)
    calls = []
    p2p.send_message_to_peer = lambda channel, msg_bytes: calls.append(("plain", len(msg_bytes))) or 1
    p2p.send_file_message_to_peer = lambda channel, msg_bytes, fl: calls.append(("file", [f["key"] for f in fl])) or 2
    msg = _sample_msg(FileContainer.from_path(_file(tmp_path, "a.bin", b"y" * 5000)))

    assert asyncio.run(p2p.send(msg.channel, msg)) == 1  # The peer said nothing: in memory
    p2p._peer_caps[PEER] = CAP_FILE_STREAM
    assert asyncio.run(p2p.send(msg.channel, msg)) == 2
    assert calls[0][0] == "plain" and calls[0][1] > 5000 and calls[1] == ("file", ["doc"])

    def refuse(channel: str, msg_bytes: bytes, fl: list) -> int:
        """Go says the peer does not take file messages.

        Args:
            channel: The channel.
            msg_bytes: The message.
            fl: The files.

        Raises:
            P2PNoFileStream: Always.
        """
        raise P2PNoFileStream("no")
    p2p.send_file_message_to_peer = refuse
    assert asyncio.run(p2p.send(msg.channel, msg)) == 1 and calls[-1][0] == "plain"


def test_a_message_whose_files_cannot_be_placed_leaves_nothing(tmp_path: object, downloads: str,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """A file for a sample the message does not have drops the message and removes the file."""
    p2p = _bare_p2p(monkeypatch)
    p2p.file_owner_of = lambda peer_id: ("alice", "agent")
    msg = Msg.from_bytes(_sample_msg(FileContainer(content=b"", filename="a.bin")).to_bytes(streamed_keys={"doc"}))
    good = _file(tmp_path, "g.part", b"g")
    assert p2p._place_received_files(msg, PEER, [{"key": "doc", "path": good, "size": 1}])
    assert msg.content["doc"]["data"].path == os.path.join(downloads, "alice", "agent", "a.bin")
    stray = _file(tmp_path, "s.part", b"s")
    msg = Msg.from_bytes(_sample_msg(FileContainer(content=b"", filename="a.bin")).to_bytes(streamed_keys={"doc"}))
    assert not p2p._place_received_files(msg, PEER, [{"key": "other", "path": stray, "size": 1}])
    assert not os.path.exists(stray)


def test_the_pop_reply_carries_the_files_of_a_message() -> None:
    """The per-message files field: absent when empty, the JSON list otherwise."""
    def entry(frm: bytes, proto: bytes, data: bytes, fl: bytes) -> bytes:
        """One message of the reply.

        Args:
            frm: The sender.
            proto: The protocol.
            data: The data.
            fl: The files JSON.

        Returns:
            The bytes.
        """
        return b"".join(struct.pack(">I", len(x)) + x for x in (frm, proto, data, fl))
    listed = json.dumps([{"key": "doc", "path": "/tmp/x.part", "size": 3}]).encode()
    body = struct.pack(">I", 2) + entry(b"A", b"p", b"one", b"") + entry(b"B", b"p", b"two", listed)
    blob = bytes([0xB1]) + struct.pack(">I", 5 + len(body)) + body
    got = TypeInterface._parse_pop_binary(blob)
    assert got[0] == {"from": "A", "data": b"one", "protocol": "p"}
    assert got[1]["files"] == [{"key": "doc", "path": "/tmp/x.part", "size": 3}] and got[1]["data"] == b"two"


def test_a_node_advertises_file_messages_only_when_it_takes_them() -> None:
    """CAP_FILE_STREAM follows accepts_file_stream (streaming possible and Custom.ACCEPT_FILES), not the code."""
    from unaiverse.networking.session.session import Session

    class Transport:
        """A transport that streams files but does not take them."""
        supports_chunked_frames, supports_file_stream, accepts_file_stream = True, True, False
    s = object.__new__(Session)
    s.p2p = Transport()
    assert s._local_capabilities() == 1
    Transport.accepts_file_stream = True
    assert s._local_capabilities() == 1 | CAP_FILE_STREAM


def test_progress_reaches_the_ticket_and_the_transfer_hook() -> None:
    """A progress event sets the progress of its ticket; progress and receive events keep transfers current (a
    message leaves it at 100% or dropped) and reach on_transfer_progress."""
    from unaiverse.agent_basics import AgentBasics
    from unaiverse.networking.node.connpool import NodeConn, SendTicket
    conn = object.__new__(NodeConn)
    ticket = SendTicket(PEER, 4)
    conn._tickets, conn._written_upto, conn._recent_drops = {("pub", PEER): {4: ticket}}, {}, {}
    conn._resolve_tickets({"type": "progress", "peer_id": PEER, "p2p_name": "pub", "seq": 4, "bytes": 10,
                           "total": 40})
    assert ticket.progress == (10, 40) and ticket.status == "queued"

    calls = []

    class Agent(AgentBasics):
        """An agent that records its progress calls."""

        async def on_transfer_progress(self, *args: object) -> None:
            """Records the call (async).

            Args:
                *args: What the framework passed.
            """
            calls.append(args)
    agent = object.__new__(Agent)
    agent.transfers = {}
    for ev in [{"type": "receive_progress", "peer_id": PEER, "seq": 2, "bytes": 5, "total": 20, "kind": "file",
                "files": ["a.bin"]},
               {"type": "progress", "peer_id": PEER, "seq": 4, "bytes": 10, "total": 40, "kind": "chunked"}]:
        asyncio.run(agent._handle_transfer_event(ev))
    assert set(agent.transfers) == {(PEER, "receive", 2), (PEER, "send", 4)}
    assert agent.transfers[(PEER, "receive", 2)]["files"] == ["a.bin"]
    asyncio.run(agent._handle_transfer_event({"type": "receive_dropped", "peer_id": PEER, "seq": 2, "bytes": 7,
                                              "total": 20, "reason": "timeout"}))
    asyncio.run(agent._handle_transfer_event({"type": "progress", "peer_id": PEER, "seq": 4, "bytes": 40,
                                              "total": 40}))
    assert agent.transfers == {}
    assert calls == [(PEER, "receive", 2, 5, 20, ["a.bin"], None), (PEER, "send", 4, 10, 40, [], None),
                     (PEER, "receive", 2, 7, 20, [], "timeout"), (PEER, "send", 4, 40, 40, [], None)]
