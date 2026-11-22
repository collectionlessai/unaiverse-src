"""
Phase 7 of the concurrency plan: fewer copies and gzip level 1 in the codec of the stream samples.

Tensors of every dtype and shape (the empty and the 0-d ones included, bfloat16 and float8, which numpy lacks) come
back identical; what a new node sends
(level 1) an old node decodes, and what an old node sends (level 9, through GzipFile) a new node decodes; protobuf
parses a message from a memoryview of the pop reply (which is what the reply now gives); the bytes for Go are passed
as they are.
"""
import io
import gzip
import torch
import pytest
from unaiverse.networking.p2p.messages import Msg
from unaiverse.networking.p2p.lib_types import TypeInterface

DTYPES = [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int8, torch.uint8, torch.int16, torch.int32,
          torch.int64, torch.bool, torch.complex64] + [getattr(torch, name) for name in ("float8_e4m3fn", "float8_e5m2")
                                                       if hasattr(torch, name)]
SHAPES = [(), (0,), (3, 0, 2), (7,), (4, 5), (2, 3, 4)]


def _msg(t: torch.Tensor) -> Msg:
    """A stream_sample message with one tensor.

    Args:
        t: The tensor.

    Returns:
        The message.
    """
    return Msg(sender="A", content_type=Msg.STREAM_SAMPLE, channel="A::dm:B-s",
               content={"x": {"data": t, "data_tag": 1, "data_uuid": None}})


def _tensor(dtype: torch.dtype, shape: tuple) -> torch.Tensor:
    """A tensor whose values differ, of the given dtype and shape.

    Args:
        dtype: The dtype.
        shape: The shape.

    Returns:
        The tensor.
    """
    n = 1
    for d in shape:
        n *= d
    return (torch.arange(n, dtype=torch.float64) * 1.37 - 3).to(dtype).reshape(shape)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether two tensors are identical, bit for bit (torch.equal has no kernel for float8).

    Args:
        a: A tensor.
        b: Another.

    Returns:
        True if dtype, shape and bytes are the same.
    """
    as_bytes = lambda t: t.contiguous().reshape(-1).view(torch.uint8)
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(as_bytes(a), as_bytes(b))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_tensors_come_back_identical(dtype: torch.dtype, shape: tuple) -> None:
    """Every dtype and shape, the empty and 0-d ones included, bfloat16 and float8 too, round trips exactly."""
    t = _tensor(dtype, shape)
    back = Msg.from_bytes(_msg(t).to_bytes()).content["x"]["data"]
    assert _same(back, t)


def test_a_non_contiguous_tensor_and_one_that_requires_grad() -> None:
    """The codec makes them contiguous and detached, without changing the values."""
    t = torch.arange(24, dtype=torch.float32).reshape(4, 6).t()
    assert not t.is_contiguous()
    assert torch.equal(Msg.from_bytes(_msg(t).to_bytes()).content["x"]["data"], t)
    g = torch.ones(3, requires_grad=True) * 2
    assert torch.equal(Msg.from_bytes(_msg(g).to_bytes()).content["x"]["data"], g.detach())


def test_what_a_new_node_sends_an_old_node_decodes() -> None:
    """Level 1 is plain gzip: the decoder of the previous release (GzipFile read, then frombuffer) reads it."""
    from unaiverse.networking.p2p import message_pb2 as pb
    t = _tensor(torch.float32, (64, 64))
    proto = pb.Message()
    proto.ParseFromString(_msg(t).to_bytes())
    td = proto.stream_sample.samples["x"].data.tensor_data
    with gzip.GzipFile(fileobj=io.BytesIO(td.data), mode='rb') as f:
        raw = f.read()
    old = torch.frombuffer(bytearray(raw), dtype=getattr(torch, td.dtype)).reshape(list(td.shape))
    assert torch.equal(old, t)


def test_what_an_old_node_sends_a_new_node_decodes() -> None:
    """The encoder of the previous release (tobytes, GzipFile at level 9) gives a message the new decoder reads."""
    from unaiverse.networking.p2p import message_pb2 as pb
    t = _tensor(torch.int32, (10, 20))
    proto = pb.Message()
    proto.sender, proto.content_type, proto.channel = "A", Msg.STREAM_SAMPLE, "A::dm:B-s"
    sample = proto.stream_sample.samples["x"]
    sample.data_tag = 1
    with io.BytesIO() as buffer:
        with gzip.GzipFile(fileobj=buffer, mode='wb') as f:
            f.write(t.numpy().tobytes())
        sample.data.tensor_data.data = buffer.getvalue()
    sample.data.tensor_data.dtype = "int32"
    sample.data.tensor_data.shape.extend([10, 20])
    assert torch.equal(Msg.from_bytes(proto.SerializeToString()).content["x"]["data"], t)


def test_tensor_data_of_the_wrong_length_is_refused() -> None:
    """A shape that does not match the data is an error, not a tensor of garbage."""
    from unaiverse.networking.p2p import message_pb2 as pb
    proto = pb.Message()
    proto.ParseFromString(_msg(_tensor(torch.float32, (10,))).to_bytes())
    proto.stream_sample.samples["x"].data.tensor_data.shape[0] = 11
    with pytest.raises(ValueError):
        Msg.from_bytes(proto.SerializeToString())


def test_a_message_parses_from_a_memoryview_of_the_pop_reply() -> None:
    """The pop reply gives each payload as a view of the buffer; the message decodes from it."""
    import struct
    payload = _msg(_tensor(torch.float32, (8, 8))).to_bytes()
    fields = [b"A", b"p", payload, b""]
    body = struct.pack(">I", 1) + b"".join(struct.pack(">I", len(x)) + x for x in fields)
    blob = bytes([0xB1]) + struct.pack(">I", 5 + len(body)) + body
    data = TypeInterface._parse_pop_binary(blob)[0]["data"]
    assert isinstance(data, memoryview) and data == payload
    assert torch.equal(Msg.from_bytes(data).content["x"]["data"], _tensor(torch.float32, (8, 8)))


def test_bytes_for_go_are_passed_as_they_are() -> None:
    """No copy into a ctypes buffer: the same bytes object (a bytearray becomes bytes; None, empty bytes)."""
    ti = object.__new__(TypeInterface)
    b = b"\x00payload\x00"
    assert ti.to_go_bytes(b) is b
    assert ti.to_go_bytes(bytearray(b"ab")) == b"ab" and ti.to_go_bytes(None) == b""


def test_session_frames_decode_from_the_views_of_the_pop_reply() -> None:
    """get_messages hands the Session codec bytes, not the view the reply gives (it decodes them with .decode)."""
    from unaiverse.networking.p2p.p2p import P2P
    from unaiverse.networking.session.sess_msg import SessMsg, SESSION_CONTROL_PROTOCOL
    frame = SessMsg(SessMsg.KIND_PING, {"session_id": b"\x00" * 8, "nonce": b"n1"}).to_bytes()
    p2p = object.__new__(P2P)
    p2p.pop_messages = lambda *a: [{"from": "B", "data": memoryview(b"x" + frame)[1:],
                                    "protocol": SESSION_CONTROL_PROTOCOL}]
    _, inbound = p2p.get_messages()
    assert len(inbound) == 1 and inbound[0].sess_msg.kind == SessMsg.KIND_PING
    assert bytes(inbound[0].sess_msg.get("nonce")) == b"n1"


def test_an_old_node_decodes_bfloat16_from_a_new_one() -> None:
    """The decoder of the previous release already took bfloat16 (by name, with frombuffer): only the encoder lacked."""
    from unaiverse.networking.p2p import message_pb2 as pb
    t = _tensor(torch.bfloat16, (5, 3))
    proto = pb.Message()
    proto.ParseFromString(_msg(t).to_bytes())
    td = proto.stream_sample.samples["x"].data.tensor_data
    with gzip.GzipFile(fileobj=io.BytesIO(td.data), mode='rb') as f:
        raw = f.read()
    assert td.dtype == "bfloat16"
    assert _same(torch.frombuffer(bytearray(raw), dtype=getattr(torch, td.dtype)).reshape(list(td.shape)), t)
