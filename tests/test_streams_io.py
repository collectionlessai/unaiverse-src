"""Characterization tests for the data semantics of ``Stream`` and ``BufferedStream``.

These tests pin down the *current* behaviour of the in-memory data store of a
stream: keying by uuid, the per-requester "deliver once" rule of ``get``, tags
and timestamps, ``enable``/``disable`` gating and the three buffer modes
(static, queue, cyclic). They use a deterministic fake clock injected into the
global ``clock`` singleton so no wall-clock or network time is ever touched.
"""

import pytest

from unaiverse.clock import clock
from unaiverse.streams.streams import Stream, BufferedStream
from unaiverse.streams.dataprops import DataProps, StreamType

pytestmark = pytest.mark.unit


# Deterministic clock

class FakeClock:
    """A fully controllable stand-in for ``unaiverse.clock._Clock``.

    Only the surface the streams touch is implemented: ``get_cycle`` (current
    cycle index), ``get_cycle_time`` (timestamp bound 1:1 to the cycle so that
    distinct cycles always yield distinct timestamps), ``get_time`` (wall-clock
    proxy, irrelevant when ``delta <= 0``) and ``next_cycle`` to advance.
    """

    def __init__(self):
        self.cycle = -1
        self._time = 1000.0

    def get_cycle(self) -> int:
        return self.cycle

    def get_cycle_time(self) -> float:
        # Timestamp tied to the cycle: each cycle is a unique timestamp, which is
        # exactly what the per-requester dedup logic of ``get`` keys on.
        return float(self.cycle)

    def get_time(self) -> float:
        return self._time

    def next_cycle(self) -> bool:
        self.cycle += 1
        self._time += 1.0
        return True


@pytest.fixture
def fake_clock():
    """Inject a deterministic clock into the global singleton for the test."""
    saved = getattr(clock, "_Clock__instance", None)
    fc = FakeClock()
    clock.set(fc)
    yield fc
    setattr(clock, "_Clock__instance", saved)


# Helpers

def text_props() -> DataProps:
    """A minimal text DataProps (no transforms, real-time, delta<=0)."""
    return DataProps(name="txt", data_type="text")


# Stream: set/get keyed by uuid

def test_set_get_keyed_by_uuid(fake_clock):
    """Data is stored and retrieved per uuid; an unknown uuid yields None."""
    s = Stream(props=text_props())
    s.set("hello", uuid="A")

    assert s.get(uuid="A") == "hello"
    assert s.get(uuid="B") is None
    assert s.get(uuid=None) is None  # default uuid slot was never written
    assert s.has_data("A")
    assert not s.has_data("B")
    assert set(s.get_data_uuids()) == {"A"}


def test_get_without_requester_is_repeatable(fake_clock):
    """Without ``requested_by`` the same sample is returned on every get."""
    s = Stream(props=text_props())
    s.set("payload", uuid="A")
    assert s.get(uuid="A") == "payload"
    assert s.get(uuid="A") == "payload"
    assert s.get(uuid="A") == "payload"


# Stream: per-requester "deliver once" semantics

def test_get_requested_by_delivers_once_per_requester(fake_clock):
    """A given sample is handed to each requester at most once, but a different
    requester still receives it."""
    s = Stream(props=text_props())
    s.set("payload", uuid="A")

    # Same requester: first call gets it, second is None.
    assert s.get(requested_by="r1", uuid="A") == "payload"
    assert s.get(requested_by="r1", uuid="A") is None

    # A different requester still receives the same sample once.
    assert s.get(requested_by="r2", uuid="A") == "payload"
    assert s.get(requested_by="r2", uuid="A") is None


def test_new_set_rearms_requester(fake_clock):
    """A fresh ``set`` (new timestamp) re-arms delivery for a requester that had
    already consumed the previous sample."""
    s = Stream(props=text_props())

    fake_clock.next_cycle()  # cycle 0 -> timestamp 0.0
    s.set("v1", uuid="A")
    assert s.get(requested_by="r1", uuid="A") == "v1"
    assert s.get(requested_by="r1", uuid="A") is None

    fake_clock.next_cycle()  # cycle 1 -> timestamp 1.0 (distinct)
    s.set("v2", uuid="A")
    assert s.get(requested_by="r1", uuid="A") == "v2"
    assert s.get(requested_by="r1", uuid="A") is None


def test_resetting_same_cycle_does_not_rearm(fake_clock):
    """Two sets in the same cycle share a timestamp, so a requester that already
    consumed the first does NOT get re-armed (current timestamp-based dedup)."""
    s = Stream(props=text_props())
    fake_clock.next_cycle()  # cycle 0

    s.set("v1", uuid="A")
    assert s.get(requested_by="r1", uuid="A") == "v1"
    # Same cycle -> same get_cycle_time() -> same data_timestamp.
    s.set("v2", uuid="A")
    assert s.get(requested_by="r1", uuid="A") is None
    # A brand-new requester sees the latest value though.
    assert s.get(requested_by="r2", uuid="A") == "v2"


# Stream: tag and timestamp accessors

def test_get_tag_and_timestamp(fake_clock):
    """``get_tag``/``get_timestamp`` expose the stored tag and cycle timestamp;
    unknown uuids return None."""
    fake_clock.next_cycle()  # cycle 0
    s = Stream(props=text_props())
    s.set("x", data_tag=7, uuid="A")

    assert s.get_tag("A") == 7
    assert s.get_timestamp("A") == 0.0  # get_cycle_time() at cycle 0
    assert s.get_tag("missing") is None
    assert s.get_timestamp("missing") is None


def test_timestamp_tracks_cycle(fake_clock):
    """Re-setting at a later cycle updates the stored timestamp."""
    s = Stream(props=text_props())
    fake_clock.next_cycle()  # cycle 0
    s.set("a", uuid="A")
    assert s.get_timestamp("A") == 0.0
    fake_clock.next_cycle()  # cycle 1
    s.set("b", uuid="A")
    assert s.get_timestamp("A") == 1.0


# Stream: data_tag ordering (auto vs explicit, stale tags)

def test_data_tag_explicit_vs_autoincrement(fake_clock):
    """An explicit tag>=0 is stored verbatim; a tag of -1 auto-fills with the
    current cycle. An older explicit tag is accepted as-is (no rejection here)."""
    s = Stream(props=text_props())

    fake_clock.cycle = 5
    s.set("a", uuid="A")              # auto -> current cycle
    assert s.get_tag("A") == 5

    fake_clock.cycle = 9
    s.set("b", data_tag=2, uuid="A")  # explicit, older than 5: still accepted
    assert s.get_tag("A") == 2

    s.set("c", uuid="A")              # auto -> current cycle 9
    assert s.get_tag("A") == 9


def test_keep_existing_tag(fake_clock):
    """``keep_existing_tag`` retains the prior tag when the new tag is -1; with
    the flag off the tag is refreshed to the current cycle."""
    s = Stream(props=text_props())

    fake_clock.cycle = 3
    s.set("a", data_tag=7, uuid="A")
    assert s.get_tag("A") == 7

    fake_clock.cycle = 8
    s.set("b", data_tag=-1, keep_existing_tag=True, uuid="A")
    assert s.get_tag("A") == 7  # kept

    s.set("c", data_tag=-1, keep_existing_tag=False, uuid="A")
    assert s.get_tag("A") == 8  # refreshed to current cycle


def test_set_tag_overwrites_and_missing_is_noop(fake_clock):
    """``set_tag`` overwrites the tag in place; on an unknown uuid it is a no-op."""
    s = Stream(props=text_props())
    s.set("a", data_tag=1, uuid="A")
    s.set_tag(42, "A")
    assert s.get_tag("A") == 42
    s.set_tag(5, "missing")  # must not raise
    assert s.get_tag("missing") is None


# Stream: enable/disable gating

def test_disable_blocks_set_but_get_still_reads(fake_clock):
    """``disable`` blocks ``set`` (returns None, data unchanged) but ``get`` keeps
    returning the last accepted value; ``force`` bypasses the gate."""
    s = Stream(props=text_props())
    s.set("a", uuid="A")

    s.disable()
    assert s.set("b", uuid="A") is None      # blocked
    assert s.get(uuid="A") == "a"            # get is not gated, old value stays

    assert s.set("c", uuid="A", force=True) is not None  # force bypasses
    assert s.get(uuid="A") == "c"

    s.enable()
    assert s.set("d", uuid="A") is not None
    assert s.get(uuid="A") == "d"


def test_disable_blocks_first_set(fake_clock):
    """While disabled, even the very first set for a uuid is rejected and stores
    nothing."""
    s = Stream(props=text_props())
    s.disable()
    assert s.set("a", uuid="Z") is None
    assert not s.has_data("Z")
    assert s.get(uuid="Z") is None


# Stream: all_uuids retrieval

def test_get_all_uuids_returns_tuples(fake_clock):
    """``get(all_uuids=True)`` returns a (sample, tag, timestamp) tuple for every
    uuid that currently holds data."""
    fake_clock.next_cycle()  # cycle 0
    s = Stream(props=text_props())
    s.set("a", data_tag=1, uuid="A")
    s.set("b", data_tag=2, uuid="B")

    result = s.get(all_uuids=True)
    assert isinstance(result, list) and len(result) == 2

    by_sample = {sample: (tag, ts) for sample, tag, ts in result}
    assert set(by_sample) == {"a", "b"}
    assert by_sample["a"][0] == 1 and by_sample["b"][0] == 2
    assert by_sample["a"][1] == 0.0 and by_sample["b"][1] == 0.0  # cycle-0 ts


def test_get_all_uuids_honours_requester_dedup(fake_clock):
    """``all_uuids`` routes through the per-requester rule: a second pass for the
    same requester yields an empty list."""
    fake_clock.next_cycle()
    s = Stream(props=text_props())
    s.set("a", uuid="A")

    first = s.get(requested_by="r", all_uuids=True)
    assert [sample for sample, _, _ in first] == ["a"]

    second = s.get(requested_by="r", all_uuids=True)
    assert second == []


# DataProps built through StreamType (per the documented build path)

def test_dataprops_from_streamtype_roundtrip(fake_clock):
    """A DataProps reconstructed from ``StreamType(...).to_list_of_dicts()`` drives
    a working Stream."""
    dicts = StreamType("text").to_list_of_dicts()
    assert len(dicts) == 2  # private + public variants

    props = DataProps.from_dict(dicts[0])
    assert props.is_text()

    s = Stream(props=props)
    s.set("hi", uuid="A")
    assert s.get(uuid="A") == "hi"


# BufferedStream: static mode

def test_buffered_static_returns_same_sample(fake_clock):
    """A static buffer keeps a single sample and yields it on every get, across
    cycles."""
    fake_clock.next_cycle()  # cycle 0
    bs = BufferedStream(text_props(), is_static=True)
    bs.set("only")

    assert bs.get() == "only"
    assert bs.get() == "only"
    fake_clock.next_cycle()  # cycle 1
    assert bs.get() == "only"
    fake_clock.next_cycle()  # cycle 2
    assert bs.get() == "only"

    # Static buffer never grows beyond one element.
    assert len(bs.data_buffer_by_uuid[None]) == 1


def test_buffered_static_extra_set_does_not_grow_buffer(fake_clock):
    """A second set on a static buffer does not append; the buffer stays at one."""
    fake_clock.next_cycle()
    bs = BufferedStream(text_props(), is_static=True)
    bs.set("first")
    bs.set("second")
    assert len(bs.data_buffer_by_uuid[None]) == 1
    fake_clock.next_cycle()
    # A fresh cycle re-reads the (single) buffered sample, i.e. the original one.
    assert bs.get() == "first"


# BufferedStream: queue mode (FIFO)

def test_buffered_queue_is_fifo(fake_clock):
    """A queue buffer delivers samples in FIFO order, one per cycle, popping each
    as it is consumed."""
    bs = BufferedStream(text_props(), is_queue=True)
    bs.set("a", data_tag=0, uuid="U")
    bs.set("b", data_tag=1, uuid="U")
    bs.set("c", data_tag=2, uuid="U")
    assert len(bs.data_buffer_by_uuid["U"]) == 3

    fake_clock.next_cycle()  # cycle 0
    assert bs.get(uuid="U") == "a"
    fake_clock.next_cycle()  # cycle 1
    assert bs.get(uuid="U") == "b"
    fake_clock.next_cycle()  # cycle 2
    assert bs.get(uuid="U") == "c"

    # The buffer has been drained by the FIFO reads.
    assert len(bs.data_buffer_by_uuid["U"]) == 0

    # Once exhausted, a further get yields the last delivered sample (no new
    # data is available, so the cached Data struct is left untouched).
    fake_clock.next_cycle()
    assert bs.get(uuid="U") == "c"


def test_buffered_queue_no_advance_within_same_cycle(fake_clock):
    """Within one cycle the queue does not advance: repeated gets give the same
    head sample."""
    bs = BufferedStream(text_props(), is_queue=True)
    bs.set("a", data_tag=0, uuid="U")
    bs.set("b", data_tag=1, uuid="U")

    fake_clock.next_cycle()  # cycle 0
    assert bs.get(uuid="U") == "a"
    assert bs.get(uuid="U") == "a"  # same cycle -> no pop
    assert len(bs.data_buffer_by_uuid["U"]) == 1  # only one element popped


# BufferedStream: cyclic (non-static, non-queue) mode

def test_buffered_cycles_through_buffer(fake_clock):
    """A plain buffer walks its samples by index, one per cycle, without consuming
    the buffer."""
    bs = BufferedStream(text_props())
    bs.set("a", data_tag=0, uuid="U")
    bs.set("b", data_tag=1, uuid="U")
    bs.set("c", data_tag=2, uuid="U")

    fake_clock.next_cycle()  # cycle 0
    assert bs.get(uuid="U") == "a"
    assert bs.get_tag("U") == 0
    fake_clock.next_cycle()  # cycle 1
    assert bs.get(uuid="U") == "b"
    assert bs.get_tag("U") == 1
    fake_clock.next_cycle()  # cycle 2
    assert bs.get(uuid="U") == "c"
    assert bs.get_tag("U") == 2

    # Non-queue buffers are not consumed by reads.
    assert len(bs.data_buffer_by_uuid["U"]) == 3


def test_buffered_no_advance_within_same_cycle(fake_clock):
    """A plain buffer does not advance the read index within a single cycle."""
    bs = BufferedStream(text_props())
    bs.set("a", data_tag=0, uuid="U")
    bs.set("b", data_tag=1, uuid="U")

    fake_clock.next_cycle()  # cycle 0
    assert bs.get(uuid="U") == "a"
    assert bs.get(uuid="U") == "a"  # same cycle
    fake_clock.next_cycle()  # cycle 1
    assert bs.get(uuid="U") == "b"


def test_buffered_past_end_keeps_last_sample(fake_clock):
    """Reading past the end of a plain (non-circular) buffer leaves the last
    delivered sample in place rather than returning None."""
    bs = BufferedStream(text_props())
    bs.set("a", data_tag=0, uuid="U")
    bs.set("b", data_tag=1, uuid="U")

    fake_clock.next_cycle()  # cycle 0 -> "a"
    assert bs.get(uuid="U") == "a"
    fake_clock.next_cycle()  # cycle 1 -> "b"
    assert bs.get(uuid="U") == "b"
    fake_clock.next_cycle()  # cycle 2 -> index past end, stale "b"
    assert bs.get(uuid="U") == "b"


def test_buffered_get_all_uuids(fake_clock):
    """``BufferedStream.get(all_uuids=True)`` returns one tuple per uuid that has a
    materialised sample (the bootstrap None slot, being empty, is skipped)."""
    bs = BufferedStream(text_props())
    bs.set("a", data_tag=0, uuid="U")

    fake_clock.next_cycle()  # cycle 0
    result = bs.get(all_uuids=True)

    assert isinstance(result, list) and len(result) == 1
    sample, tag, ts = result[0]
    assert sample == "a"
    assert tag == 0
    assert ts == 0.0
