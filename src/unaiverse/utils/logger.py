"""
      █████  █████ ██████   █████           █████ █████   █████ ██████████ ███████████    █████████  ██████████
     ░░███  ░░███ ░░██████ ░░███           ░░███ ░░███   ░░███ ░░███░░░░░█░░███░░░░░███  ███░░░░░███░░███░░░░░█
      ░███   ░███  ░███░███ ░███   ██████   ░███  ░███    ░███  ░███  █ ░  ░███    ░███ ░███    ░░░  ░███  █ ░
      ░███   ░███  ░███░░███░███  ░░░░░███  ░███  ░███    ░███  ░██████    ░██████████  ░░█████████  ░██████
      ░███   ░███  ░███ ░░██████   ███████  ░███  ░░███   ███   ░███░░█    ░███░░░░░███  ░░░░░░░░███ ░███░░█
      ░███   ░███  ░███  ░░█████  ███░░███  ░███   ░░░█████░    ░███ ░   █ ░███    ░███  ███    ░███ ░███ ░   █
      ░░████████   █████  ░░█████░░████████ █████    ░░███      ██████████ █████   █████░░█████████  ██████████
       ░░░░░░░░   ░░░░░    ░░░░░  ░░░░░░░░ ░░░░░      ░░░      ░░░░░░░░░░ ░░░░░   ░░░░░  ░░░░░░░░░  ░░░░░░░░░░
                A Collectionless AI Project (https://collectionless.ai)
                Registration/Login: https://unaiverse.io
                Code Repositories:  https://github.com/collectionlessai/
                Main Developers:    Stefano Melacci (Project Leader), Christian Di Maio, Tommaso Guidi
"""
import re
import sys
import json
import html
import threading
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from datetime import datetime, timezone
from unaiverse.custom import Custom, GenException


class Ch(str, Enum):
    """Log channels. Independent, non-hierarchical output categories.

    Each channel can be individually enabled or disabled for screen output
    and file persistence.  CRITICAL is the sole exception: it is always on
    and cannot be suppressed by any means.

    CRITICAL - Fatal conditions. Always written to file and screen.
    ERROR    - Non-fatal errors and warnings.
    USER     - Output intended for the end user (human-readable app output).
    STATEM   - Per-clock-cycle full app state snapshots.
    NETWORK  - Networking events: connections, disconnections, message routing.
    STREAMS  - Stream-level data-flow events: tokens, buffers, lifecycle.
    INTER    - Inter-node / cross-instance communication events (RPCs, coordination).
    MISC     - Miscellaneous / uncategorized internal output.
    DEBUG    - Fine-grained developer diagnostics; off by default.
    CPOOL    - Connection pool lifecycle events (slots, eviction, limits).
    P2P      - Raw output captured from the underlying Go libp2p library (stdout/stderr).

    Subdomain tag (``sub=`` kwarg, available on every channel)
    ------------------------------------------------------------
    Every channel supports an optional ``sub=`` tag to identify which
    subdomain the record belongs to.  The three standard values are:

    ``"pub"``  — public network side
    ``"prv"``  — private network side
    ``"gen"``  — generic / not subdomain specific (default when omitted)

    Pass ``sub=`` explicitly to override::

        log.network("peer connected",  sub="pub", peer=pid)
        log.network("relay heartbeat", sub="prv", relay=rid)
        log.streams("token received",  sub="pub", tokens=12)
        log.statem("state snapshot",   sub="prv", mode="IDLE")
        log.debug("internal trace",    sub="prv")
        log.misc("startup",            sub="gen")   # same as omitting sub=

    The tag is stored as a top-level ``"sub"`` field in every JSON record
    (parallel to ``"ch"``, not inside ``"info"``).  When ``sub=`` is omitted
    the sticky value set by ``set_sub()`` is used; if that is also unset the
    record gets ``"sub": "gen"`` automatically.
    """
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    USER = "USER"
    STATEM = "STATEM"
    NETWORK = "NETWORK"
    STREAMS = "STREAMS"
    INTER = "INTER"
    MISC = "MISC"
    DEBUG = "DEBUG"
    CPOOL = "CPOOL"
    P2P = "P2P"


# Channels that are unconditionally always active (cannot be disabled)
ALWAYS_ON_CHANNELS: frozenset[Ch] = frozenset({Ch.CRITICAL, Ch.ERROR, Ch.USER})

# All channels
EXTRA_CHANNELS: frozenset[Ch] = frozenset({Ch.NETWORK, Ch.STREAMS, Ch.INTER,
                                           Ch.MISC, Ch.DEBUG, Ch.STATEM, Ch.CPOOL, Ch.P2P})

# All channels
ALL_CHANNELS: frozenset[Ch] = frozenset(ALWAYS_ON_CHANNELS | EXTRA_CHANNELS)

# ANSI color codes per channel
_COLORS: dict[Ch, str] = {
    Ch.CRITICAL: "\033[30;41m",  # Black text on red background
    Ch.ERROR: "\033[91m",  # Bright red
    Ch.USER: "\033[0m",  # Default/white
    Ch.STATEM: "\033[94m",  # Bright blue
    Ch.NETWORK: "\033[93m",  # Yellow
    Ch.STREAMS: "\033[96m",  # Cyan
    Ch.INTER: "\033[38;5;214m",  # Orange (256-colour)
    Ch.MISC: "\033[90m",   # Dark gray
    Ch.DEBUG: "\033[2;37m",   # Dim white
    Ch.CPOOL: "\033[38;5;74m",  # Steel blue
    Ch.P2P: "\033[95m",    # Magenta
}

# Misc
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"  # Used for the info bracket on screen

SUB_PRV = "prv"
SUB_PUB = "pub"
SUB_GEN = "gen"

# Valid subdomain tags.  "gen" is the default (generic / not subdomain specific).
_SUB_DEFAULT: str = SUB_GEN

_ALL_SUBS: frozenset[str] = frozenset({SUB_PRV, SUB_PUB, SUB_GEN})


class _Logger:
    """Structured, channel-based logger for long-running applications.

    Channels are independent output categories — not a severity hierarchy.
    Each channel can be enabled or disabled independently for screen output
    and file persistence.  CRITICAL is always on and cannot be suppressed.

    All records are written to a single append log file as JSONL (one JSON
    object per line), making them trivially parseable by any external tool.

    **Record schema**::

        {
            "ts":    "2026-03-14T12:00:00.123+00:00",  # ISO-8601 UTC
            "ch":    "NETWORK",                         # channel name
            "sub":   "pub",                             # optional subdomain tag (pub | prv)
            "cycle": 42,                                # clock-cycle index (-1 = pre-cycle)
            "ctx":   {"node": "n1", "mode": "ACTIVE"}, # sticky context (set via set_ctx)
            "msg":   "Human-readable message",
            "info":  {"peer": "10.0.0.1", "latency_ms": 4}  # optional per-call extra info
        }

    The ``"sub"`` field is only present on NETWORK / STREAMS / STATEM records
    when ``sub=`` is passed to the log method.  All other channels omit it.

    Screen format::

        [CHANNEL ] HH:MM:SS.mmm  c=<cycle> | ctx_key=val ...  Message text  [key=val  key=val]
        [NETWORK ] HH:MM:SS.mmm  c=<cycle> | ...  {pub}  Message text  [key=val]

    **Thread safety**: all public methods are protected by a single "R-Lock".

    Args:
        name:     Base name for the log file (no extension).
        log_dir:  Directory for the log file; created if absent.
        active:   Set of channels to enable.  CRITICAL is always on.
                  Defaults to ``{ERROR, USER}`` if not given.
        screen:   Subset of active channels that also print to stdout.
                  Defaults to all active channels if not given.
        no_color:       Disable ANSI color codes (auto-detected for non-TTY).
        verbose_screen: When ``False`` (default), USER / ERROR / CRITICAL print
                        only the bare message to stdout — no timestamp, no channel
                        badge, no info bracket.  All other channels always print
                        the full record.  Set to ``True`` to get the full record
                        for every channel (useful while debugging).
        file_enabled:   When ``True`` (default), records are written to the JSONL
                        log file.  Set to ``False`` to suppress all file output
                        (e.g. during unit tests or when only screen output is
                        wanted).  Can be toggled at runtime via
                        ``set_file_enabled()``.
    """
    def __init__(
            self,
            name: str = "app",
            log_dir: str | Path = "logs",
            active: set[Ch] | None = None,
            screen: set[Ch] | None = None,
            no_color: bool = False,
            verbose_screen: bool = False,
            file_enabled: bool = True,
    ):
        self._lock = threading.RLock()
        self._name = name
        self._no_color = no_color or not sys.stdout.isatty()
        self._verbose_screen = verbose_screen
        self._force_plain_msg_only = False
        self._file_enabled = file_enabled
        self._cycle_idx: int = -1
        self._ctx: dict[str, Any] = {}
        self._sub: str = ""  # sticky subdomain tag (set via set_sub)
        self._clock = None
        self._print_fcn: dict[str, dict[str, Callable[..., Any]]] = {ks: {k: print for k in ALL_CHANNELS}
                                                                     for ks in _ALL_SUBS}
        self._print_fcn_supports_html = {ks: {k: False for k in ALL_CHANNELS} for ks in _ALL_SUBS}

        # Resolve active channels
        if active is None:
            active = ALWAYS_ON_CHANNELS
        self._active: frozenset[Ch] = frozenset(active) | ALWAYS_ON_CHANNELS

        # Resolve screen channels (default: all active)
        if screen is None:
            self._screen: frozenset[Ch] = self._active
        else:
            self._screen = (frozenset(screen) | ALWAYS_ON_CHANNELS) & self._active

        # File setup — single append log
        log_dir = Path(log_dir).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = log_dir / f"{name}.jsonl"
        self._log_file = None
        if self._file_enabled and self._log_file is None:
            self._log_file = open(self._log_path, "w", encoding="utf-8", buffering=1)

        # Inspector only
        self.inspector_activated = False
        self._output_messages = [""] * 20
        self._output_messages_ids = [-1] * 20
        self._output_messages_count = 0
        self._output_messages_last_pos = -1

        # Avoid repetitions
        self.__last_printed_msg = None
        self.__last_printed_tick = None

    def set_name(self, name: str) -> None:
        self._name = name

    def enable(self, *channels: Ch) -> None:
        """Enable one or more channels at runtime."""
        with self._lock:
            self._active |= frozenset(channels)

    def disable(self, *channels: Ch) -> None:
        """Disable one or more channels at runtime. Some channels cannot be disabled."""
        with self._lock:
            self._active = (self._active - frozenset(channels)) | ALWAYS_ON_CHANNELS

    def enable_screen(self, *channels: Ch) -> None:
        """Enable screen output for one or more channels."""
        with self._lock:
            self._screen |= frozenset(channels)

    def disable_screen(self, *channels: Ch) -> None:
        """Suppress screen output for one or more channels. Some channels cannot be suppressed."""
        with self._lock:
            self._screen = (self._screen - frozenset(channels)) | ALWAYS_ON_CHANNELS

    def disable_all_screen(self) -> None:
        """Suppress screen output for all channels. Some channels cannot be suppressed."""
        with self._lock:
            self._screen = ALWAYS_ON_CHANNELS

    def inspector_enabled(self, true_or_false: bool) -> None:
        self.inspector_activated = true_or_false

    def set_plain_msg_only(self, yes_or_no):
        self._force_plain_msg_only = yes_or_no

    def set_verbose_screen(self, verbose: bool) -> None:
        """Toggle verbose screen output at runtime.

        When ``verbose=False`` (default at construction), USER / ERROR / CRITICAL
        print only the bare message.  When ``verbose=True``, every channel prints
        the full record (channel badge, timestamp, cycle, context, info bracket).
        """
        with self._lock:
            self._verbose_screen = verbose

    def set_file_enabled(self, enabled: bool) -> None:
        """Enable or disable writing records to the JSONL log file at runtime.

        Screen output is unaffected.  Useful for temporarily suppressing disk
        writes (e.g. during startup noise, unit tests, or replay sessions).
        """
        with self._lock:
            if enabled and self._log_file is None:
                self._log_file = open(self._log_path, "w", encoding="utf-8", buffering=1)
            self._file_enabled = enabled

    def set_clock(self, clock) -> None:
        """Update the current clock object."""
        with self._lock:
            self._clock = clock

    def set_sub(self, sub: str) -> None:
        """Set a sticky subdomain tag applied automatically to NETWORK / STREAMS / STATEM records.

        Call this once at the entry point of a code section that deals exclusively
        with one subdomain, so every subsequent ``network()`` / ``streams()`` /
        ``statem()`` call carries the tag without having to pass ``sub=`` each time.
        An explicit ``sub=`` argument at the call site always overrides the sticky value.

        Args:
            sub: ``"pub"`` for the public network, ``"prv"`` for the private network,
                 or ``""`` to clear the sticky tag (equivalent to ``clear_sub()``).

        Example::

            log.set_sub("pub")
            log.network("peer connected", peer=pid)   # → sub="pub" automatically
            log.streams("token received", tokens=12)  # → sub="pub" automatically

            log.set_sub("prv")
            log.network("relay heartbeat", relay=rid) # → sub="prv" automatically

            log.clear_sub()                           # back to no sticky tag
        """
        with self._lock:
            self._sub = sub

    def clear_sub(self) -> None:
        """Clear the sticky subdomain tag.  Subsequent records carry no ``sub`` field
        unless ``sub=`` is passed explicitly at the call site."""
        with self._lock:
            self._sub = ""

    def set_ctx(self, **kwargs: Any) -> None:
        """Merge key-value pairs into the sticky context.

        The context dict is included verbatim in every record under ``"ctx"``.
        Call this on mode/phase transitions rather than every call.
        """
        with self._lock:
            self._ctx.update(kwargs)

    def clear_ctx(self, *keys: str) -> None:
        """Remove specific keys from the sticky context, or clear all if none given."""
        with self._lock:
            if keys:
                for k in keys:
                    self._ctx.pop(k, None)
            else:
                self._ctx.clear()

    def critical(self, msg: str, sub: str = "", **info: Any) -> None:
        """Log a fatal condition. Always shown and persisted; cannot be suppressed.

        Args:
            msg:    Human-readable message.
            sub:    Sub-domain tag — ``"pub"``, ``"prv"``, or ``"gen"`` (default).
            **info: Per-call key-value details appended as ``[key=val ...]`` on
                    screen and stored under ``"info"`` in the JSON record.
        """
        self._log(Ch.CRITICAL, msg, info, sub=sub)
        raise GenException(msg)

    def error(self, msg: str, sub: str = "", **info: Any) -> None:
        """Log a non-fatal error or warning."""
        self._log(Ch.ERROR, msg, info, sub=sub)

    def user(self, msg: str, sub: str = "", **info: Any) -> None:
        """Log a user-facing application message."""
        self._log(Ch.USER, msg, info, sub=sub)

    def statem(self, msg: str, sub: str = "", **info: Any) -> None:
        """Log an app state message (e.g. a per-cycle snapshot summary).

        Args:
            msg: Human-readable message.
            sub: Sub-domain tag — ``"pub"``, ``"prv"``, or ``"gen"`` (default).
            **info: Per-call key-value details.
        """
        self._log(Ch.STATEM, msg, info, sub=sub)

    def network(self, msg: str, sub: str = "", **info: Any) -> None:
        """Log a networking event (connections, routing, peer messages).

        Args:
            msg: Human-readable message.
            sub: Sub-domain tag — ``"pub"``, ``"prv"``, or ``"gen"`` (default).
            **info: Per-call key-value details.
        """
        self._log(Ch.NETWORK, msg, info, sub=sub)

    def streams(self, msg: str, sub: str = "", **info: Any) -> None:
        """Log a stream-level data-flow event (tokens, buffers, lifecycle).

        Args:
            msg: Human-readable message.
            sub: Sub-domain tag — ``"pub"``, ``"prv"``, or ``"gen"`` (default).
            **info: Per-call key-value details.
        """
        self._log(Ch.STREAMS, msg, info, sub=sub)

    def inter(self, msg: str, sub: str = "", **info: Any) -> None:
        """Log an internode / cross-instance communication event.

        Use this for RPCs, coordination messages, and any traffic that crosses
        the boundary between application instances (as opposed to lower-level
        libp2p transport events, which belong to NETWORK or P2P).

        Args:
            msg:    Human-readable message.  Unicode and emoji are fully supported.
            sub:    Sub-domain tag — ``"pub"``, ``"prv"``, or ``"gen"`` (default).
            **info: Per-call key-value details.
        """
        self._log(Ch.INTER, msg, info, sub=sub)

    def misc(self, msg: str, sub: str = "", **info: Any) -> None:
        """Log a miscellaneous / uncategorized internal message."""
        self._log(Ch.MISC, msg, info, sub=sub)

    def debug(self, msg: str, sub: str = "", **info: Any) -> None:
        """Log a fine-grained developer diagnostic.

        This channel is **off by default** — it must be explicitly included in
        ``active`` at construction or enabled at runtime via ``enable(Ch.DEBUG)``.
        Use it for high-frequency or very detailed traces that would be too noisy
        for normal operation.
        """
        self._log(Ch.DEBUG, msg, info, sub=sub)

    def cpool(self, msg: str, sub: str = "", **info: Any) -> None:
        """Log a connection pool lifecycle event (slot open/close, eviction, limit hit)."""
        self._log(Ch.CPOOL, msg, info, sub=sub)

    def p2p(self, msg: str, sub: str = "", **info: Any) -> None:
        """Log a raw line captured from the Go libp2p library (stdout/stderr)."""
        self._log(Ch.P2P, msg[44:], info, sub=sub)  # Skipping timestamp and initial line info coming from GO

    def _log(self, ch: Ch, msg: str, info: dict[str, Any], sub: str = "") -> None:
        with self._lock:
            if ch not in self._active:
                return

            if self._clock is None:
                ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
                cycle = 0
            else:
                ts = self._clock.get_time_as_string()
                cycle = self._clock.get_cycle()

            record: dict[str, Any] = {
                "ts": ts,
                "ch": ch.value,
                "cy": cycle,
                "ctx": dict(self._ctx),
                "msg": msg,
            }
            # call-site wins → sticky fallback → default "gen"
            effective_sub = sub or self._sub or _SUB_DEFAULT
            record["sub"] = effective_sub
            if info:
                record["info"] = info

            if self._file_enabled:
                assert self._log_file is not None
                self._log_file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

            if ch in self._screen:
                self._print_record(ch, ts, record)

            if ch in ALWAYS_ON_CHANNELS and self.inspector_activated:
                last_id = self._output_messages_ids[self._output_messages_last_pos]
                self._output_messages_last_pos = (self._output_messages_last_pos + 1) % len(self._output_messages)
                self._output_messages_count = min(self._output_messages_count + 1, len(self._output_messages))
                self._output_messages_ids[self._output_messages_last_pos] = last_id + 1
                self._output_messages[self._output_messages_last_pos] = html.escape(str(msg), quote=True)

    def _print_record(self, ch: Ch, ts: str, record: dict[str, Any]) -> None:
        color = "" if self._no_color else _COLORS[ch]
        reset = "" if self._no_color else _RESET
        bold = "" if self._no_color else _BOLD
        dim = "" if self._no_color else _DIM

        msg = record['msg']
        if msg is not None:
            must_print = False

            if "info" in record and "rep" in record["info"] and record["info"]["rep"] is True:
                must_print = True
            else:
                if msg in Custom.ACTION_TICKS_PER_STATUS:
                    if msg != self.__last_printed_tick:
                        self.__last_printed_tick = msg
                        must_print = True
                elif msg != self.__last_printed_msg:
                    self.__last_printed_msg = msg
                    self.__last_printed_tick = None
                    must_print = True

            if must_print:
                if not self._print_fcn_supports_html:
                    # Handle a bit of HTML (<br/>, <a href=...>...</a>, <strong>...</strong>)
                    msg = (msg.replace('<br/>', '\n').replace('<strong>', '')
                           .replace('</strong>', ''))
                    pattern = r'<a\s+href=[\'"](.*?)[\'"][^>]*>(.*?)</a>'
                    msg = re.sub(pattern, r'\2 (\1)', msg)

                # Replacing
                record['msg'] = msg

                sub = record.get("sub", "")

                if self._force_plain_msg_only:
                    self._print_fcn[sub][ch](f"{record['msg']}", file=sys.stdout, flush=True)
                    return

                # USER / ERROR / CRITICAL in non-verbose mode: just the message, no preamble.
                # Full record for everything else, or when verbose_screen=True.
                if not self._verbose_screen and ch in ALWAYS_ON_CHANNELS:
                    self._print_fcn[sub][ch](f"{color}{record['msg']}{reset}", file=sys.stdout, flush=True)
                    return

                time_part = ts[11:23]  # HH:MM:SS.mmm
                cycle_str = f"cy={record['cy']}"

                ctx = record.get("ctx", {})
                ctx_str = (" | " + "  ".join(f"{k}={v}" for k, v in ctx.items())) if ctx else ""

                # Subdomain tag: rendered as {pub} or {prv} in dim braces before the message
                sub_str = f"  {dim}{{{sub}}}{reset}{color}" if sub else ""

                info = record.get("info", {})
                info_str = (f"  {dim}[" + "  ".join(f"{k}={v}" for k, v in info.items()) + f"]{reset}{color}") \
                    if info else ""

                prefix = f"{color}{bold}[{ch.value:^8}]{reset}{color} {time_part}  {cycle_str}{ctx_str}"
                self._print_fcn[sub][ch](f"{prefix}{sub_str}  {record['msg']}{info_str}{reset}",
                                         file=sys.stdout, flush=True)

    def __call__(self, msg: str, sub: str = "", **info: Any) -> None:
        """Log to MISC by calling the logger instance directly.

        Allows the logger to be used as a drop-in callable anywhere a simple
        print-like function is expected::

            log("initializing subsystem", component="foo")
            log("pub path shortcut", sub="pub")
        """
        self._log(Ch.MISC, msg, info, sub=sub)

    def set_print_fcn(self, print_fcn, ch, sub, supports_html):
        if ch is None:
            chs = ALL_CHANNELS
        else:
            chs = [ch]
        if sub is None:
            subs = _ALL_SUBS
        else:
            subs = [sub]

        for _sub in subs:
            for _ch in chs:
                self._print_fcn[_sub][_ch] = print_fcn
                self._print_fcn_supports_html[_sub][_ch] = supports_html

    def close(self) -> None:
        """Flush and close the log file."""
        with self._lock:
            assert self._log_file is not None
            self._log_file.flush()
            self._log_file.close()

    def __enter__(self) -> "_Logger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def get_inspector_console(self):
        f = self._output_messages_last_pos - self._output_messages_count + 1  # Included
        t = self._output_messages_last_pos  # Included
        ff = -1
        tt = -1
        if t >= 0 > f:  # If there is something, and we incurred in the circular organization (t: valid; f: negative)
            ff = len(self._output_messages) + f  # Included
            tt = len(self._output_messages) - 1  # Included
            f = 0
        elif t < 0:  # If there are no messages at all (t: -1; f: 0 - due to the way we initialized class attributes)
            f = -1
            t = -1
        console = {'output_messages': self._output_messages[ff:tt+1] + self._output_messages[f:t+1]}
        return console


class Logger:
    def __init__(self):

        # Default instance, expected to be overwritten by calling create(...) when the application has enough info
        # to decide how to set up the logger
        self.__instance = None  # _Logger(name="logger", file_enabled=False)

    def create(self, *args, **kwargs):
        self.__instance = _Logger(*args, **kwargs)

    def set_print_fcn(self, *args, **kwargs):
        self.__instance.set_print_fcn(*args, **kwargs)

    def __getattr__(self, name):
        if self.__instance is None:
            if name == "critical":
                def _crit(msg, *a, **k):
                    print(msg)
                    raise GenException(str(msg))     # Preserve the raising contract
                return _crit
            return lambda *args, **kwargs: print(*args)   # Swallow logger-specific kwargs
        return getattr(self.__instance, name)


# The basic shared object (laze initialization by the first class which imports it)
# The actual thing to import
log = Logger()
