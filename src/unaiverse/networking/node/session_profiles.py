"""The values of the Session (timeouts, retries, keep-alive, quarantine) as named profiles.

The values stay module globals of ``session.py`` and ``session_manager.py``, read at call time, and the ``default``
profile is what those two files declare: a Python node runs it without applying anything. A profile differs from it
by the values it names, and applying one sets the globals before any Session runs. The browser applies ``mobile``.

The bounds between values, which the comments of the two files give as sentences, are checked by
``session_profile_problems``: the tests run it on every profile, and ``apply_session_profile`` refuses what breaks one.
"""

import json
import numbers
from pathlib import Path
from types import ModuleType

from unaiverse.custom import Custom
from unaiverse.utils.logger import log
from unaiverse.networking.node import connpool
from unaiverse.networking.session import session, session_manager

# Each value of a profile: its name, the module that holds it and the global it is held in
_VALUES: dict[str, tuple[ModuleType, str]] = {
    "connect_s": (session, "_TIMEOUT_CONNECT_S"),
    "hello_s": (session, "_TIMEOUT_HELLO_S"),
    "profile_s": (session, "_TIMEOUT_PROFILE_S"),
    "world_s": (session, "_TIMEOUT_WORLD_S"),
    "reconnect_s": (session, "_TIMEOUT_RECONNECT_S"),
    "peer_lost_session_s": (session, "_TIMEOUT_PEER_LOST_SESSION_S"),
    "closing_s": (session, "_TIMEOUT_CLOSING_S"),
    "closing_failed_send_s": (session, "_TIMEOUT_CLOSING_FAILED_SEND_S"),
    "token_exp_grace_s": (session, "_TOKEN_EXP_GRACE_S"),
    "max_dial_attempts": (session, "_MAX_DIAL_ATTEMPTS"),
    "retry_base_s": (session, "_RETRY_BASE_S"),
    "retry_cap_s": (session, "_RETRY_CAP_S"),
    "resume_ack_timeout_s": (session, "_RESUME_ACK_TIMEOUT_S"),
    "hello_repeat_after_s": (session, "_HELLO_REPEAT_AFTER_S"),
    "max_hello_repeats": (session, "_MAX_HELLO_REPEATS"),
    "max_repeated_hellos_answered": (session, "_MAX_REPEATED_HELLOS_ANSWERED"),
    "address_lookup_interval_s": (session, "_ADDRESS_LOOKUP_INTERVAL_S"),
    "keepalive_period_s": (session, "_KEEPALIVE_PERIOD_S"),
    "ping_rtt_timeout_s": (session, "_PING_RTT_TIMEOUT_S"),
    "keepalive_fail_threshold": (session, "_KEEPALIVE_FAIL_THRESHOLD"),
    "quarantine_s": (session_manager, "_QUARANTINE_S"),
    "quarantine_stall_s": (session_manager, "_QUARANTINE_STALL_S"),
    "cut_linger_s": (session_manager, "_CUT_LINGER_S"),
    "unknown_reply_linger_s": (session_manager, "_UNKNOWN_REPLY_LINGER_S"),
}

# The values that count something: whole numbers, at least one
_COUNTS = {"max_dial_attempts", "max_hello_repeats", "max_repeated_hellos_answered", "keepalive_fail_threshold"}

# What the two files declare, read before any profile is applied
_DEFAULT = {name: getattr(module, attr) for name, (module, attr) in _VALUES.items()}

# Each profile by the values in which it differs from the default
_PROFILES: dict[str, dict[str, float]] = {
    "default": {},

    # A browser, also on a phone over a slow or intermittent link (handoffs, tunnels) and in a background tab. Its dial
    # attempts are bounded by the transport of the browser (5 s each), so its whole budget is far shorter than the
    # backstop of a Python node, and a peer it lost may be dialled again sooner. The keep-alive and the reconnect
    # window are those of every platform: the two ends of a Session do not tell each other theirs.
    "mobile": {
        "connect_s": 60.0,
        "profile_s": 30.0,  # The profile bytes over a slow link
        "world_s": 45.0,
        "closing_s": 15.0,
        "max_dial_attempts": 6,
        "resume_ack_timeout_s": 8.0,
        "quarantine_s": 20.0,
    },
}

# The table of the values the transports share (see shared_transport_values.json): read here for the bounds only
_SHARED_TRANSPORT_VALUES = Path(__file__).resolve().parent.parent / "p2p" / "shared_transport_values.json"

# How many re-dials, each with its backoff and its wait for the RESUME_ACK, the reconnect window must leave room for
# after the grace of the transport
_REDIALS_IN_A_WINDOW = 3


def apply_session_profile(name: str = "default", overrides: dict[str, float] | None = None) -> dict[str, float]:
    """Applies a profile, with optional overrides by name, before any Session runs.

    Nothing is applied when the name of the profile or of an override is unknown, a value is not a number, or the
    values break a bound between them.

    Args:
        name: The name of the profile (default: "default").
        overrides: Values by name that replace those of the profile (default: None).

    Returns:
        The values in force after it, by name.

    Raises:
        ValueError: On an unknown name, a value that is not a number, or a broken bound (all of them are named).
    """
    values = session_profile(name, overrides)
    problems = session_profile_problems(values)
    if problems:
        raise ValueError(f"Session profile {name!r} not applied: " + "; ".join(problems))
    for value_name, value in values.items():
        module, attr = _VALUES[value_name]
        setattr(module, attr, value)
    changed = {k: v for k, v in values.items() if v != _DEFAULT[k]}
    log.session(f"Session profile {name!r} applied" + (f", differing from the default in {changed}" if changed else ""))
    return dict(values)


def session_profile(name: str = "default", overrides: dict[str, float] | None = None) -> dict[str, float]:
    """The values a profile gives, with optional overrides by name, without applying them.

    Args:
        name: The name of the profile (default: "default").
        overrides: Values by name that replace those of the profile (default: None).

    Returns:
        Every value of the Session, by name.

    Raises:
        ValueError: On an unknown name of profile or of value, or on a value that is not a number.
    """
    if name not in _PROFILES:
        raise ValueError(f"unknown Session profile {name!r} (known: {', '.join(sorted(_PROFILES))})")
    overrides = dict(overrides or {})
    unknown = sorted(set(overrides) - set(_VALUES))
    if unknown:
        raise ValueError(f"unknown Session values {unknown} (known: {', '.join(sorted(_VALUES))})")
    values = {**_DEFAULT, **_PROFILES[name], **overrides}
    for value_name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or value < 0:
            raise ValueError(f"the Session value {value_name!r} is {value!r}, not a number of zero or more")
        if value_name in _COUNTS and (value != int(value) or value < 1):
            raise ValueError(f"the Session value {value_name!r} is {value!r}, not a count of one or more")
    return values


def session_profile_names() -> list[str]:
    """The names of the profiles.

    Returns:
        The names, sorted.
    """
    return sorted(_PROFILES)


def session_values() -> dict[str, float]:
    """The values of the Session in force, by name: what the modules hold now.

    Returns:
        Every value of the Session, by name.
    """
    return {name: getattr(module, attr) for name, (module, attr) in _VALUES.items()}


def session_profile_problems(values: dict[str, float]) -> list[str]:
    """The bounds between values that a set of values breaks.

    The bounds are those the comments of session.py and session_manager.py give, plus those with the world (the
    pre-admission of a granted peer, the rejoin) and with the transports (shared_transport_values.json). A responder
    meets initiators of every profile, so what it answers is checked against all of them.

    Args:
        values: Every value of the Session, by name.

    Returns:
        One sentence per broken bound, empty when none is broken.
    """
    v, problems = values, []
    shared = _shared_transport_values_ms()

    def check(holds: bool, sentence: str) -> None:
        """Records a bound that does not hold."""
        if not holds:
            problems.append(sentence)

    check(v["ping_rtt_timeout_s"] < v["keepalive_period_s"],
          "the wait for a PONG must stay under the keep-alive period (one PING in flight at a time)")
    check(v["quarantine_s"] < Custom.REJOINING_WAIT,
          "the quarantine must stay under the rejoin wait, so that a rejoin lands after it")
    check(v["cut_linger_s"] < v["quarantine_stall_s"],
          "the linger of the cut must stay under the stall quarantine, so that the cut precedes the next dial")
    check(v["cut_linger_s"] < v["closing_s"] and v["cut_linger_s"] < v["quarantine_s"],
          "the linger of the cut must stay under the closing window and the quarantine")
    check(v["hello_repeat_after_s"] * v["max_hello_repeats"] < v["hello_s"],
          "the repeats of a HELLO must fit in the stage of the HELLO")
    check(v["address_lookup_interval_s"] >= Custom.SEND_DYNAMIC_PROFILE_EVERY,
          "the lookup of fresh addresses must not come more often than a peer posts them")
    longest_dial = connpool._WORLD_DIAL_MAX_BUDGET_S
    redemption = max(longest_dial + p["hello_s"] + p["profile_s"] for p in _every_profile(values))
    check(Custom.WORLD_PREADMISSION_TTL > redemption,
          f"the pre-admission ({Custom.WORLD_PREADMISSION_TTL} s) must outlast the redemption of a grant "
          f"({redemption} s: the longest world dial and the two handshake stages)")
    repeats = max(p["max_dial_attempts"] - 1 + p["max_hello_repeats"] for p in _every_profile(values))
    check(v["max_repeated_hellos_answered"] >= repeats,
          f"the repeated HELLOs a responder answers must cover those of an initiator of any profile ({repeats})")
    check(shared["webrtc_freshness"] / 1000 > v["keepalive_period_s"],
          "the freshness of a DataChannel must exceed the keep-alive period, so that a live channel stays fresh")
    redials = sum(min(v["retry_base_s"] * 2 ** n, v["retry_cap_s"]) + v["resume_ack_timeout_s"]
                  for n in range(_REDIALS_IN_A_WINDOW))
    check(v["reconnect_s"] > shared["disconnection_grace"] / 1000 + redials,
          f"the reconnect window must outlast the grace of the transport by {_REDIALS_IN_A_WINDOW} re-dials "
          f"with their backoff and their wait for the RESUME_ACK")
    return problems


def _every_profile(values: dict[str, float]) -> list[dict[str, float]]:
    """The values of every profile, and the ones given.

    Args:
        values: Every value of the Session, by name.

    Returns:
        A list of sets of values.
    """
    return [{**_DEFAULT, **overrides} for overrides in _PROFILES.values()] + [values]


def _shared_transport_values_ms() -> dict[str, float]:
    """The values of the table the transports share, in milliseconds.

    Returns:
        The values by name.
    """
    table = json.loads(_SHARED_TRANSPORT_VALUES.read_text(encoding="utf-8"))
    to_ms = {"ms": 1, "s": 1000}
    return {entry["name"]: entry["value"] * to_ms[entry["unit"]] for entry in table["values"]}
