"""Characterization tests for p2p/messages.py Msg.

These pin the CURRENT behaviour of Msg before the R5 rewrite (Msg as a
protobuf-free value object). They guard two contracts that the rewrite must not
break:

  * the WIRE: to_bytes() must keep producing the same protobuf bytes, so a
    rewritten CPython node stays interoperable with the existing fleet. Pinned
    via an INDEPENDENT protobuf oracle (the expected pb.Message is built by
    hand), only for the deterministic content kinds (json_content, stats_update).
    stream_sample is NOT byte-pinned: it gzips tensors, and gzip writes a
    non-deterministic mtime in its header, so its wire is checked by round-trip.

  * the OBSERVABLE SHAPE: the scalar properties and the structure of `content`
    per content_type. After R5 this shape is also the contract the JS twin reads
    over the Pyodide proxy, so it must stay identical.

The module top of messages.py is allowed to import protobuf today; the import
check that the rewrite makes the top stdlib-only lives elsewhere (Pyodide-like
import test), not here.
"""

import io
import json
import re

import torch
from PIL import Image

from unaiverse.networking.p2p.messages import Msg
from unaiverse.networking.p2p import message_pb2 as pb
from unaiverse.streams.dataprops import FileContainer

FIXED_TS = "2026-01-02 03:04:05.678901"


# ── scalar fields + observable shape ─────────────────────────────────────────

def test_scalar_fields_preserved_on_construction_and_roundtrip():
    msg = Msg(sender="alice", content={"k": 1}, timestamp_net=FIXED_TS,
              channel="ch-1", content_type=Msg.MISC, piggyback="tok0")
    assert msg.sender == "alice"
    assert msg.content_type == Msg.MISC
    assert msg.channel == "ch-1"
    assert msg.piggyback == "tok0"
    assert msg.timestamp_net == FIXED_TS

    back = Msg.from_bytes(msg.to_bytes())
    assert back.sender == "alice"
    assert back.content_type == Msg.MISC
    assert back.channel == "ch-1"
    assert back.piggyback == "tok0"
    assert back.timestamp_net == FIXED_TS
    assert back.content == {"k": 1}


def test_default_timestamp_when_absent():
    """A new Msg with no timestamp_net stamps now() in the fixed UTC format."""
    msg = Msg(sender="a", content_type=Msg.MISC)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}", msg.timestamp_net)


def test_empty_content_is_sentinel():
    """content None or '<empty>' leaves the oneof unset; reading content gives '<empty>'."""
    for empty in (None, "<empty>"):
        msg = Msg(sender="a", content=empty, content_type=Msg.MISC, timestamp_net=FIXED_TS)
        assert msg.content == "<empty>"
        back = Msg.from_bytes(msg.to_bytes())
        assert back.content == "<empty>"


def test_invalid_content_type_rejected():
    try:
        Msg(sender="a", content_type="not_a_real_type")
        assert False, "expected an assertion on invalid content_type"
    except AssertionError as e:
        assert "Invalid content type" in str(e)


# ── json_content: wire pinned via independent protobuf oracle ────────────────

def test_json_content_wire_matches_protobuf_oracle():
    content = {"hello": "world", "n": 3, "nested": {"a": [1, 2]}}
    msg = Msg(sender="s", content=content, timestamp_net=FIXED_TS,
              channel="c", content_type=Msg.INTERACTION, piggyback="p")

    expected = pb.Message()
    expected.sender = "s"
    expected.timestamp_net = FIXED_TS
    expected.content_type = Msg.INTERACTION
    expected.channel = "c"
    expected.piggyback = "p"
    expected.json_content = json.dumps(content)

    assert msg.to_bytes() == expected.SerializeToString()


def test_json_content_roundtrip_shape():
    content = {"x": 1, "y": ["a", "b"]}
    msg = Msg(sender="s", content=content, content_type=Msg.WORLD_AGENTS_LIST,
              timestamp_net=FIXED_TS)
    back = Msg.from_bytes(msg.to_bytes())
    assert back.content == content


# ── stream_sample: round-trip (gzip mtime makes the wire non-deterministic) ──

def _sample(data, tag=7, uuid="uuid-1"):
    return {"data": data, "data_tag": tag, "data_uuid": uuid}


def test_stream_sample_tensor_roundtrip():
    t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    msg = Msg(sender="s", content={"feat": _sample(t)},
              content_type=Msg.STREAM_SAMPLE, timestamp_net=FIXED_TS)
    back = Msg.from_bytes(msg.to_bytes())
    sample = back.content["feat"]
    assert sample["data_tag"] == 7
    assert sample["data_uuid"] == "uuid-1"
    assert torch.equal(sample["data"], t)
    assert sample["data"].dtype == torch.float32


def test_stream_sample_image_roundtrip():
    img = Image.new("RGB", (4, 3), color=(10, 20, 30))
    img.putpixel((1, 1), (200, 100, 50))
    msg = Msg(sender="s", content={"img": _sample(img, uuid=None)},
              content_type=Msg.STREAM_SAMPLE, timestamp_net=FIXED_TS)
    back = Msg.from_bytes(msg.to_bytes())
    sample = back.content["img"]
    assert sample["data_uuid"] is None
    out = sample["data"].convert("RGB")
    assert out.size == img.size
    assert out.tobytes() == img.tobytes()


def test_stream_sample_text_roundtrip():
    msg = Msg(sender="s", content={"t": _sample("hello there")},
              content_type=Msg.STREAM_SAMPLE, timestamp_net=FIXED_TS)
    back = Msg.from_bytes(msg.to_bytes())
    assert back.content["t"]["data"] == "hello there"


def test_stream_sample_file_roundtrip():
    fc = FileContainer(content=b"\x00\x01rawbytes", filename="f.bin", mime_type="application/octet-stream")
    msg = Msg(sender="s", content={"f": _sample(fc)},
              content_type=Msg.STREAM_SAMPLE, timestamp_net=FIXED_TS)
    back = Msg.from_bytes(msg.to_bytes())
    out = back.content["f"]["data"]
    assert isinstance(out, FileContainer)
    assert out.content == b"\x00\x01rawbytes"
    assert out.filename == "f.bin"
    assert out.mime_type == "application/octet-stream"


# ── stats_update: round-trip, including the existing int->float coercion ─────

def test_stats_update_roundtrip():
    payload = [
        {"group_key": "peerA", "stat_name": "loss", "timestamp": 100, "value": 1.5},
        {"group_key": "peerB", "stat_name": "tags", "timestamp": 200, "value": ["x", "y"]},
        {"group_key": "peerC", "stat_name": "meta", "timestamp": 300, "value": {"a": 1}},
    ]
    msg = Msg(sender="s", content=payload, content_type=Msg.STATS_UPDATE, timestamp_net=FIXED_TS)
    back = Msg.from_bytes(msg.to_bytes())
    out = back.content

    assert [u["group_key"] for u in out] == ["peerA", "peerB", "peerC"]
    assert [u["stat_name"] for u in out] == ["loss", "tags", "meta"]
    assert [u["timestamp"] for u in out] == [100, 200, 300]
    assert out[0]["value"] == 1.5
    assert out[1]["value"] == ["x", "y"]
    assert out[2]["value"] == {"a": 1}


def test_stats_update_int_value_becomes_float():
    """Existing behaviour: a Python int value rides protobuf Value.number_value
    and comes back as float. Pinned so the rewrite preserves it (not 'fixes' it)."""
    payload = [{"group_key": "p", "stat_name": "n", "timestamp": 1, "value": 5}]
    msg = Msg(sender="s", content=payload, content_type=Msg.STATS_UPDATE, timestamp_net=FIXED_TS)
    back = Msg.from_bytes(msg.to_bytes())
    assert back.content[0]["value"] == 5.0
    assert isinstance(back.content[0]["value"], float)
