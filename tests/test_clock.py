"""Characterization tests for ``unaiverse.clock``.

These exercise only the deterministic, offline-safe surface of the clock. The
constructor of ``_Clock`` reaches out to NTP/HTTP time servers *only* when no
``current_time`` is supplied, so every clock built here is given a fixed
``current_time`` to keep construction purely local. ``1700000000.0`` corresponds
to ``2023-11-14T22:13:20`` UTC, which lets us assert on the rendered date.

What is intentionally NOT covered, because it is inherently network/wall-clock
dependent and cannot be made deterministic:
  - ``_Clock.__get_time_from_server`` (NTP then HTTP fallback);
  - the network branch of the constructor (``current_time <= 0``);
  - the absolute wall-clock value returned by ``get_time`` (we only assert
    structural/relational properties of it).
"""

import re

import pytest

from unaiverse.clock import _Clock, Clock, clock
from unaiverse.custom import Custom

pytestmark = pytest.mark.unit


# A fixed wall-clock anchor: 1700000000.0 == 2023-11-14T22:13:20 UTC.
FIXED_T = 1700000000.0


def _local_clock(min_delta: float = -1.0) -> _Clock:
    """Build a fully offline ``_Clock`` (fixed ``current_time`` skips the NTP/HTTP sync)."""
    return _Clock(min_delta=min_delta, current_time=FIXED_T)


class _ManualClock(_Clock):
    """A ``_Clock`` whose virtual time is advanced explicitly via ``advance``.

    ``get_time`` is stable between calls, so cycle bookkeeping is deterministic and
    never depends on the wall-clock resolution. This is what makes the cycle tests
    below reproducible: with the real ``get_time`` two fast ``next_cycle`` calls can
    read an identical instant and ``__add_timestamp`` then raises on the duplicate
    timestamp (the real-time-mode footgun pinned by
    ``test_real_time_next_cycle_without_advance_raises``).
    """

    def __init__(self, min_delta: float = -1.0, current_time: float = FIXED_T):
        super().__init__(min_delta=min_delta, current_time=current_time)
        self._anchor = current_time
        self._virtual_now = current_time

    def advance(self, dt: float = 1.0) -> None:
        self._virtual_now += dt

    def get_time(self, passed: bool = False) -> float:
        return (self._virtual_now - self._anchor) if passed else self._virtual_now


def _manual_clock(min_delta: float = -1.0) -> _ManualClock:
    """Build an offline ``_Clock`` with an explicitly advanced virtual time."""
    return _ManualClock(min_delta=min_delta)


# cycle2time (pure arithmetic with explicit delta)

def test_cycle2time_with_delta_is_linear():
    """With a positive delta, cycle2time maps cycle k to k*delta exactly."""
    c = _local_clock()
    assert c.cycle2time(0, delta=2.0) == 0.0
    assert c.cycle2time(5, delta=2.0) == 10.0
    assert c.cycle2time(7, delta=2.5) == 17.5


def test_cycle2time_negative_cycle_without_delta_returns_minus_one():
    """Without a stored timestamp, a negative cycle index maps to -1.0."""
    c = _local_clock()
    assert c.cycle2time(-1) == -1.0


def test_cycle2time_ignores_non_positive_delta():
    """A delta <= 0 falls back to the timestamp-table branch (here: cycle -1 -> -1.0)."""
    c = _local_clock()
    assert c.cycle2time(-1, delta=0.0) == -1.0
    assert c.cycle2time(-1, delta=-5.0) == -1.0


# get_cycle / next_cycle (real-time, unconstrained)

def test_initial_cycle_is_minus_one():
    """A freshly created clock starts at cycle index -1."""
    c = _local_clock()
    assert c.get_cycle() == -1


def test_next_cycle_increments_when_unconstrained():
    """With min_delta <= 0 the clock is real-time: next_cycle always advances."""
    c = _manual_clock(min_delta=-1.0)
    assert c.next_cycle() is True
    assert c.get_cycle() == 0
    c.advance()
    assert c.next_cycle() is True
    assert c.get_cycle() == 1


def test_timestamps_are_strictly_increasing_across_cycles():
    """Each advanced cycle records a strictly later timestamp."""
    c = _manual_clock(min_delta=-1.0)
    c.next_cycle()
    c.advance()
    c.next_cycle()
    t0 = c.cycle2time(0)
    t1 = c.cycle2time(1)
    assert t1 > t0


def test_get_cycle_time_matches_current_cycle_timestamp():
    """get_cycle_time is just cycle2time of the current cycle index."""
    c = _manual_clock(min_delta=-1.0)
    c.next_cycle()
    c.advance()
    c.next_cycle()
    assert c.get_cycle_time() == c.cycle2time(c.get_cycle())


# min_delta gating of next_cycle

def test_min_delta_blocks_second_immediate_advance():
    """With a large min_delta the first advance succeeds, the immediate second does not."""
    c = _local_clock(min_delta=1000.0)
    assert c.next_cycle() is True   # cycle -1 -> 0 is never gated
    assert c.get_cycle() == 0
    assert c.next_cycle() is False  # not enough wall-time passed since last timestamp
    assert c.get_cycle() == 0


def test_real_time_next_cycle_without_advance_raises():
    """SUSPECTED BUG (clock.py:222): in real-time mode (min_delta <= 0) next_cycle does no
    gating, so two advances at the very same instant record a duplicate timestamp and
    __add_timestamp raises instead of degrading gracefully (e.g. returning False). Pinned
    here as the current behaviour, deterministically (the virtual clock is not advanced
    between the two calls), which is what the real wall-clock can also produce when two
    next_cycle calls land within the timer resolution. See the TODO(review) at the site."""
    c = _manual_clock(min_delta=-1.0)
    assert c.next_cycle() is True            # cycle -1 -> 0 records the current virtual instant
    with pytest.raises(ValueError):
        c.next_cycle()                       # same virtual instant -> duplicate timestamp -> raise


# time2cycle without delta (timestamp-table search)

def test_time2cycle_finds_exact_recorded_timestamps():
    """An exact recorded timestamp resolves to its cycle index."""
    c = _manual_clock(min_delta=-1.0)
    c.next_cycle()
    c.advance()
    c.next_cycle()
    t0 = c.cycle2time(0)
    t1 = c.cycle2time(1)
    assert c.time2cycle(t0) == 0
    assert c.time2cycle(t1) == 1


def test_time2cycle_future_timestamp_returns_length():
    """A timestamp newer than every recorded one yields the table length."""
    c = _manual_clock(min_delta=-1.0)
    c.next_cycle()
    c.advance()
    c.next_cycle()
    t1 = c.cycle2time(1)
    assert c.time2cycle(t1 + 1000.0) == 2  # two timestamps recorded -> length 2


def test_time2cycle_unrecorded_intermediate_returns_minus_one():
    """A timestamp strictly between two recorded ones is reported as not found (-1)."""
    c = _manual_clock(min_delta=-1.0)
    c.next_cycle()
    c.advance()
    c.next_cycle()
    t0 = c.cycle2time(0)
    t1 = c.cycle2time(1)
    assert c.time2cycle((t0 + t1) / 2.0) == -1


# time2cycle with explicit delta (constant-cadence model)

def test_time2cycle_with_delta_returns_current_cycle_for_now():
    """Converting ~now under a huge delta gives back the current cycle."""
    c = _manual_clock(min_delta=-1.0)
    c.next_cycle()
    c.advance()
    c.next_cycle()
    assert c.time2cycle(c.get_time(), delta=1e9) == c.get_cycle()


def test_time2cycle_with_delta_counts_back_whole_deltas():
    """A timestamp 1.5 deltas in the past is one whole cycle behind; 0.5 deltas is none."""
    c = _manual_clock(min_delta=-1.0)
    c.next_cycle()
    c.advance()
    c.next_cycle()
    base = c.get_cycle()
    assert c.time2cycle(c.get_time() - 15.0, delta=10.0) == base - 1
    assert c.time2cycle(c.get_time() - 5.0, delta=10.0) == base


# get_time / get_time_ms

def test_get_time_uses_fixed_anchor():
    """get_time is the fixed anchor plus a tiny non-negative local elapsed term."""
    c = _local_clock()
    now = c.get_time()
    assert now >= FIXED_T
    assert now < FIXED_T + 60.0  # local elapsed since construction is sub-second


def test_get_time_passed_flag_returns_elapsed_only():
    """With passed=True only the (non-negative) local elapsed time is returned."""
    c = _local_clock()
    elapsed = c.get_time(passed=True)
    assert elapsed >= 0.0
    assert elapsed < 60.0


def test_get_time_ms_is_integer_and_non_decreasing():
    """Plain get_time_ms returns an int and never goes backwards."""
    c = _local_clock()
    a = c.get_time_ms()
    b = c.get_time_ms()
    assert isinstance(a, int)
    assert isinstance(b, int)
    assert b >= a


def test_get_time_ms_monotonic_is_strictly_increasing():
    """The monotonic flag guarantees each returned millisecond strictly exceeds the last."""
    c = _local_clock()
    vals = [c.get_time_ms(monotonic=True) for _ in range(6)]
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))


# get_time_as_string

def test_get_time_as_string_iso_millisecond_utc_format():
    """The string is ISO-8601 UTC with millisecond precision and a +00:00 offset."""
    c = _local_clock()
    s = c.get_time_as_string()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}\+00:00", s)
    # The fixed anchor pins the date down to the second.
    assert s.startswith("2023-11-14T22:13:20")


# slow-mode toggling

def test_run_slower_then_natural_speed_round_trip():
    """run_slower raises min_delta to the slow constant; run_natural_speed restores it."""
    c = _local_clock(min_delta=-1.0)
    assert c.is_slowed_down() is False
    c.run_slower()
    assert c.is_slowed_down() is True
    assert c.min_delta == Custom.SLOW_CLOCK_DELTA
    c.run_natural_speed()
    assert c.is_slowed_down() is False
    assert c.min_delta == -1.0


def test_run_slower_keeps_larger_existing_delta():
    """run_slower never lowers an already-larger min_delta (it takes the max)."""
    big = Custom.SLOW_CLOCK_DELTA + 10.0
    c = _local_clock(min_delta=big)
    c.run_slower()
    assert c.min_delta == big


# _search static helper (pure)

def test_search_empty_list_returns_minus_one():
    """Searching an empty timestamp table yields -1."""
    assert _Clock._search([], 5.0, 0) == -1


def test_search_target_beyond_last_returns_length():
    """A target greater than the last element returns the list length."""
    assert _Clock._search([1.0, 2.0, 3.0], 4.0, 0) == 3


def test_search_exact_match_via_bisect():
    """An exact match not adjacent to the cache hint is found by bisection."""
    assert _Clock._search([10.0, 20.0, 30.0], 30.0, 0) == 2


def test_search_missing_intermediate_returns_minus_one():
    """A value between two entries is reported as not found."""
    assert _Clock._search([10.0, 20.0, 30.0], 25.0, 0) == -1


def test_search_uses_cache_hint_fast_paths():
    """The cache hint short-circuits to the hinted position or the next one."""
    assert _Clock._search([10.0, 20.0, 30.0], 20.0, 1) == 1
    assert _Clock._search([10.0, 20.0, 30.0], 30.0, 1) == 2


# Clock proxy wrapper

def test_clock_proxy_set_delegates_attributes():
    """Clock.set wires the proxy onto an existing _Clock and forwards attribute access."""
    inner = _local_clock(min_delta=-1.0)
    inner.next_cycle()
    proxy = Clock()
    proxy.set(inner)
    assert proxy.get_cycle() == inner.get_cycle()
    assert proxy.cycle2time(3, delta=2.0) == 6.0


def test_clock_proxy_lazy_create_is_offline():
    """A bare Clock lazily builds its _Clock on first access using the local wall clock."""
    proxy = Clock()
    # First attribute access triggers create(current_time=now); no network involved.
    assert proxy.get_cycle() == -1


def test_module_singleton_is_usable_offline():
    """The exported ``clock`` singleton answers without any network round-trip."""
    assert isinstance(clock.get_cycle(), int)
    assert clock.cycle2time(4, delta=0.5) == 2.0
