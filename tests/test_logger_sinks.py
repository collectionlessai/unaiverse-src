"""Unit tests for the pluggable record sinks on the logger (area: LoggerSinks).

These pin the contract of ``_Logger.add_record_sink`` / ``remove_record_sink``: a
sink is invoked with ``(channel_name, record)`` for every emitted record, only for
channels that pass the ``_active`` filter (same gate as the file/screen output). A
sink that raises must never break logging nor starve the other sinks, and removal
must stop delivery. All tests run the logger with ``file_enabled=False`` so nothing
touches the filesystem.
"""

import pytest

from unaiverse.utils.logger import _Logger, Ch

pytestmark = pytest.mark.unit


def _make_logger(active):
    return _Logger(name="test", log_dir="/tmp", active=set(active), screen=set(),
                   no_color=True, file_enabled=False)


def test_sink_receives_channel_and_record():
    log = _make_logger({Ch.MISC})
    seen = []
    log.add_record_sink(lambda ch, rec: seen.append((ch, rec)))

    log.misc("hello", foo=1)

    assert len(seen) == 1
    ch, rec = seen[0]
    assert ch == Ch.MISC.value
    assert rec["ch"] == Ch.MISC.value
    assert rec["msg"] == "hello"
    assert rec["info"] == {"foo": 1}


def test_sink_not_called_for_inactive_channel():
    log = _make_logger({Ch.MISC})  # DEBUG is not active here
    seen = []
    log.add_record_sink(lambda ch, rec: seen.append(ch))

    log.debug("filtered out")  # inactive channel: dropped before the sink
    log.misc("delivered")

    assert seen == [Ch.MISC.value]


def test_raising_sink_does_not_break_logging_nor_other_sinks():
    log = _make_logger({Ch.MISC})
    good = []

    def boom(ch, rec):
        raise RuntimeError("sink blew up")

    log.add_record_sink(boom)
    log.add_record_sink(lambda ch, rec: good.append(rec["msg"]))

    # Must not raise, and the healthy sink must still fire.
    log.misc("still here")

    assert good == ["still here"]


def test_remove_record_sink_stops_delivery():
    log = _make_logger({Ch.MISC})
    seen = []
    sink = lambda ch, rec: seen.append(rec["msg"])

    log.add_record_sink(sink)
    log.misc("first")
    log.remove_record_sink(sink)
    log.misc("second")

    assert seen == ["first"]


def test_remove_unregistered_sink_is_noop():
    log = _make_logger({Ch.MISC})
    log.remove_record_sink(lambda ch, rec: None)  # never added: must not raise


def test_add_record_sink_is_idempotent():
    log = _make_logger({Ch.MISC})
    seen = []
    sink = lambda ch, rec: seen.append(rec["msg"])

    log.add_record_sink(sink)
    log.add_record_sink(sink)  # second add must not double-register
    log.misc("once")

    assert seen == ["once"]


def test_always_on_channel_reaches_sink():
    log = _make_logger(set())  # only ALWAYS_ON channels active
    seen = []
    log.add_record_sink(lambda ch, rec: seen.append(ch))

    log.user("to the user")

    assert seen == [Ch.USER.value]
