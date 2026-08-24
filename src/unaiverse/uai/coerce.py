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
# Coercion of a raw answer to a field's canonical value, and re-check of the declared constraints
# (INTERACT-PROTOCOL.md, sections 5.1 and 5.2). Two entry points: coerce_text for a string written by a human
# or by a model, check_canonical for a value that arrived already typed inside a reply block. Both end in
# check_constraints, so a value out of range is caught the same way whichever door it came through.
# Every function returns a Coerced tuple with a reason from the closed issue list. Nothing raises.
import re
from typing import NamedTuple
from .templates import template_for
from .validate import is_iso_date, is_int, is_num, normalize_label
from .constants import (ISSUE_TYPE, ISSUE_BELOW_MIN, ISSUE_ABOVE_MAX, ISSUE_TOO_LONG, ISSUE_FORMAT,
                        ISSUE_UNKNOWN_OPTION)

EMAIL_RE = re.compile(r"[^\s@]+@[^\s@]+\.[^\s@]+")
URL_RE = re.compile(r"https?://[^\s/$.?#].[^\s]*", re.IGNORECASE)
TEL_RE = re.compile(r"[+()0-9\s-]+")
DIGIT_RE = re.compile(r"[0-9]")
DECIMAL_COMMA_RE = re.compile(r"-?[0-9]+,[0-9]+")
NUMBER_RE = re.compile(r"-?([0-9]+\.?[0-9]*|\.[0-9]+)")


class Coerced(NamedTuple):
    """Outcome of a coercion: either a canonical value, or a reason from the closed issue list."""
    ok: bool
    value: object = None
    reason: str | None = None


def _ok(value) -> Coerced:
    return Coerced(True, value, None)


def _fail(reason: str) -> Coerced:
    return Coerced(False, None, reason)


def coerce_text(f: dict, raw, lang=None, allow_decimal_comma: bool = True) -> Coerced:
    """Coerces a string to the canonical value of a field.

    Args:
        f: A validated field spec.
        raw: What the human or the model wrote for that field.
        lang: Language of the form, which selects the accepted words and date shapes.
        allow_decimal_comma: False in the positional csv rung, where a comma is a separator.
    """
    s = ("" if raw is None else str(raw)).strip()
    _, t = template_for(lang)

    if f["type"] in ("text", "textarea"):
        return check_constraints(f, s)
    if f["type"] in ("number", "integer"):
        n = parse_number(s, f.get("unit"), allow_decimal_comma)
        if n is None:
            return _fail(ISSUE_TYPE)
        if f["type"] == "integer" and not is_int(n):
            return _fail(ISSUE_TYPE)
        return check_constraints(f, n)
    if f["type"] == "bool":
        w = s.lower()
        if w in t["true_words"]:
            return _ok(True)
        if w in t["false_words"]:
            return _ok(False)
        return _fail(ISSUE_TYPE)
    if f["type"] == "date":
        iso = parse_date(s, t)
        if iso is None:
            return _fail(ISSUE_TYPE)
        return check_constraints(f, iso)
    if f["type"] == "select":
        if f["max"] == 1:
            v = resolve_option(f, s)
            if v is None:
                return _fail(ISSUE_UNKNOWN_OPTION)
            return check_constraints(f, v)
        values = []
        for item in [x.strip() for x in s.split(",")]:
            if not item:
                continue
            v = resolve_option(f, item)
            if v is None:
                return _fail(ISSUE_UNKNOWN_OPTION)
            if v not in values:
                values.append(v)
        return check_constraints(f, values)
    return check_constraints(f, s)


def check_canonical(f: dict, v) -> Coerced:
    """Type-checks a value that arrived already canonical, in a reply block.

    It is never re-parsed: the string "180" for a number field is a type issue, not the number 180.
    """
    if f["type"] in ("text", "textarea"):
        return check_constraints(f, v) if isinstance(v, str) else _fail(ISSUE_TYPE)
    if f["type"] == "number":
        return check_constraints(f, v) if is_num(v) else _fail(ISSUE_TYPE)
    if f["type"] == "integer":
        return check_constraints(f, v) if is_int(v) else _fail(ISSUE_TYPE)
    if f["type"] == "bool":
        return _ok(v) if isinstance(v, bool) else _fail(ISSUE_TYPE)
    if f["type"] == "date":
        return check_constraints(f, v) if is_iso_date(v) else _fail(ISSUE_TYPE)
    if f["type"] == "select":
        known = {o["value"] for o in f["options"]}
        if f["max"] == 1:
            if not isinstance(v, str):
                return _fail(ISSUE_TYPE)
            return check_constraints(f, v) if v in known else _fail(ISSUE_UNKNOWN_OPTION)
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            return _fail(ISSUE_TYPE)
        if not all(x in known for x in v):
            return _fail(ISSUE_UNKNOWN_OPTION)
        return check_constraints(f, list(dict.fromkeys(v)))
    return check_constraints(f, v) if isinstance(v, str) else _fail(ISSUE_TYPE)


def check_constraints(f: dict, v) -> Coerced:
    """Re-checks the constraints declared by a field on an already canonical value."""
    if f["type"] == "text":
        if "maxLength" in f and len(v) > f["maxLength"]:
            return _fail(ISSUE_TOO_LONG)
        fmt = f.get("format")
        if fmt == "email" and EMAIL_RE.fullmatch(v) is None:
            return _fail(ISSUE_FORMAT)
        if fmt == "url" and URL_RE.fullmatch(v) is None:
            return _fail(ISSUE_FORMAT)
        if fmt == "tel" and (TEL_RE.fullmatch(v) is None or len(DIGIT_RE.findall(v)) < 5):
            return _fail(ISSUE_FORMAT)
        return _ok(v)
    if f["type"] == "textarea":
        if "maxLength" in f and len(v) > f["maxLength"]:
            return _fail(ISSUE_TOO_LONG)
        return _ok(v)
    if f["type"] in ("number", "integer", "date"):
        if "min" in f and v < f["min"]:
            return _fail(ISSUE_BELOW_MIN)
        if "max" in f and v > f["max"]:
            return _fail(ISSUE_ABOVE_MAX)
        return _ok(v)
    if f["type"] == "select":
        if f["max"] == 1:
            return _ok(v)
        if "min" in f and len(v) < f["min"]:
            return _fail(ISSUE_BELOW_MIN)
        if len(v) > f["max"]:
            return _fail(ISSUE_ABOVE_MAX)
        return _ok(v)
    return _ok(v)


def parse_number(s: str, unit=None, allow_decimal_comma: bool = True):
    """Reads a number from text, stripping the declared unit and, where allowed, the decimal comma."""
    x = s.strip()
    if unit:
        u = unit.lower()
        if x.lower().endswith(u):
            x = x[:len(x) - len(u)].strip()
    if allow_decimal_comma and DECIMAL_COMMA_RE.fullmatch(x) is not None:
        x = x.replace(",", ".")
    if NUMBER_RE.fullmatch(x) is None:
        return None
    try:
        n = float(x)
    except ValueError:
        return None
    return n if n == n and n not in (float("inf"), float("-inf")) else None


def parse_date(s: str, t: dict):
    """Reads a date from text: ISO first, then the shapes of the language."""
    if is_iso_date(s):
        return s
    for pattern, order in t["date_shapes"]:
        m = pattern.fullmatch(s)
        if m is None:
            continue
        g = {order[0]: m.group(1), order[1]: m.group(2), order[2]: m.group(3)}
        iso = f"{g['y']}-{g['m'].zfill(2)}-{g['d'].zfill(2)}"
        if is_iso_date(iso):
            return iso
    return None


def resolve_option(f: dict, s: str):
    """Maps a written option (value or label, exact or normalised) to its canonical value."""
    for o in f["options"]:
        if o["value"] == s:
            return o["value"]
    for o in f["options"]:
        if o["label"] == s:
            return o["value"]
    n = normalize_label(s)
    for o in f["options"]:
        if normalize_label(o["value"]) == n:
            return o["value"]
    for o in f["options"]:
        if normalize_label(o["label"]) == n:
            return o["value"]
    return None
