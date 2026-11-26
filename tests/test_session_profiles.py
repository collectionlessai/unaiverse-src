"""The Session values as named profiles: what each gives, how it is applied, and the bounds between values.

The browser used to put its values over the module globals by attribute name, and a renamed global was skipped with
no error. A profile is applied by one function that refuses a name it does not know and a set of values that breaks a
bound, and the values it gives are the ones the browser patched until now.
"""

import pytest

from unaiverse.networking.node import session_profiles
from unaiverse.networking.node.session_profiles import (apply_session_profile, session_profile, session_profile_names,
                                                        session_profile_problems, session_values)
from unaiverse.networking.session import session, session_manager

pytestmark = pytest.mark.unit

# The values the SDK put over the module globals (MOBILE_SESSION_TIMEOUTS in Pyodide/Py-Js/JNodeConn.js at bead6a5),
# by the names of the profiles (connectS is connect_s, and so on), but for the keep-alive and the reconnect window
SDK_MOBILE_VALUES = {
    "connect_s": 60, "hello_s": 15, "profile_s": 30, "world_s": 45, "closing_s": 15,
    "max_dial_attempts": 6, "retry_base_s": 1, "retry_cap_s": 30, "resume_ack_timeout_s": 8, "quarantine_s": 20,
}

# Those every platform shares
SHARED_VALUES = {"reconnect_s": 90, "keepalive_period_s": 15, "ping_rtt_timeout_s": 10, "keepalive_fail_threshold": 2}


@pytest.fixture(autouse=True)
def _default_values_after_each_test():
    """Every test leaves the module globals as the default profile has them."""
    yield
    apply_session_profile("default")


def test_the_default_profile_is_what_a_python_node_runs():
    assert session_profile("default") == session_values()
    assert session_values()["reconnect_s"] == session._TIMEOUT_RECONNECT_S
    assert session_values()["quarantine_s"] == session_manager._QUARANTINE_S


def test_the_mobile_profile_gives_the_values_the_browser_patched_and_the_default_for_the_rest():
    mobile, default = session_profile("mobile"), session_profile("default")
    assert {name: mobile[name] for name in SDK_MOBILE_VALUES} == SDK_MOBILE_VALUES
    assert {name: mobile[name] for name in SHARED_VALUES} == {name: default[name] for name in SHARED_VALUES}
    assert {name: value for name, value in mobile.items() if name not in SDK_MOBILE_VALUES} == \
        {name: value for name, value in default.items() if name not in SDK_MOBILE_VALUES}


def test_applying_a_profile_sets_the_module_globals_and_the_default_takes_them_back():
    applied = apply_session_profile("mobile")
    assert applied == session_values() == session_profile("mobile")
    assert session._TIMEOUT_PROFILE_S == 30 and session._MAX_DIAL_ATTEMPTS == 6
    assert session_manager._QUARANTINE_S == 20

    apply_session_profile("default")
    assert session._TIMEOUT_PROFILE_S == session_profile("default")["profile_s"]
    assert session_manager._QUARANTINE_S == session_profile("default")["quarantine_s"]


def test_one_keep_alive_and_one_reconnect_window_for_every_platform():
    """The two ends of a Session do not tell each other their window: one value on every platform."""
    for name in session_profile_names():
        assert {key: session_profile(name)[key] for key in SHARED_VALUES} == SHARED_VALUES


def test_an_override_by_name_goes_over_the_profile():
    applied = apply_session_profile("mobile", {"connect_s": 75, "closing_s": 12.5})
    assert applied["connect_s"] == session._TIMEOUT_CONNECT_S == 75
    assert session._TIMEOUT_CLOSING_S == 12.5
    assert session._TIMEOUT_PROFILE_S == 30


@pytest.mark.parametrize("name, overrides, says", [
    ("phone", None, "unknown Session profile 'phone'"),
    ("mobile", {"reconnectS": 90}, "unknown Session values ['reconnectS']"),
    ("mobile", {"closing_s": "15"}, "not a number"),
    ("mobile", {"max_dial_attempts": 2.5}, "not a count"),
    ("mobile", {"ping_rtt_timeout_s": 25}, "the wait for a PONG must stay under the keep-alive period"),
])
def test_a_wrong_name_or_value_is_refused_and_nothing_is_applied(name, overrides, says):
    """A renamed global was skipped in silence by the patches of the browser: here it fails, before any change."""
    before = session_values()
    with pytest.raises(ValueError, match=says.replace("[", r"\[").replace("]", r"\]")):
        apply_session_profile(name, overrides)
    assert session_values() == before


@pytest.mark.parametrize("name", ["default", "mobile"])
def test_every_profile_keeps_the_bounds_between_its_values(name):
    assert session_profile_problems(session_profile(name)) == []


@pytest.mark.parametrize("overrides, says", [
    ({"ping_rtt_timeout_s": 15}, "the wait for a PONG"),
    ({"quarantine_s": 30}, "under the rejoin wait"),
    ({"cut_linger_s": 5}, "under the stall quarantine"),
    ({"cut_linger_s": 4, "closing_s": 3}, "under the closing window"),
    ({"hello_repeat_after_s": 8}, "fit in the stage of the HELLO"),
    ({"address_lookup_interval_s": 5}, "more often than a peer posts them"),
    ({"profile_s": 45}, "must outlast the redemption of a grant"),
    ({"max_repeated_hellos_answered": 6}, "cover those of an initiator of any profile"),
    ({"keepalive_period_s": 30, "ping_rtt_timeout_s": 10}, "the freshness of a DataChannel"),
    ({"reconnect_s": 25}, "must outlast the grace of the transport"),
])
def test_each_bound_is_checked(overrides, says):
    problems = session_profile_problems(session_profile("default", overrides))
    assert any(says in problem for problem in problems), problems


def test_a_responder_answers_the_hellos_of_the_initiator_with_most_attempts():
    """A Python node meets browsers: what it answers covers the dial attempts of the mobile profile too."""
    default = session_profile("default")
    assert default["max_dial_attempts"] < session_profile("mobile")["max_dial_attempts"]
    assert session_profile_problems(default) == []
    with pytest.raises(ValueError, match="cover those of an initiator"):
        apply_session_profile("default", {"max_repeated_hellos_answered": default["max_dial_attempts"] + 1})


def test_the_names_of_the_profiles():
    assert session_profile_names() == ["default", "mobile"]


def test_every_value_names_a_module_global_that_exists():
    """A global renamed in the Session files must fail here, not leave a profile that sets a new attribute."""
    for name, (module, attr) in session_profiles._VALUES.items():
        assert hasattr(module, attr), (name, attr)
