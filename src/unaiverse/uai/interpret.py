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
# The one interpreter of INTERACT-PROTOCOL.md section 5: five rungs, tried in order, one normalised event out,
# whoever wrote the answer. It is deliberately not "for the model": a human typing by hand instead of using the
# widget goes through exactly the same code.
#
# Inputs are the message text and the pending form specs of the conversation, oldest first. No registry is
# needed for the first rung: a canonical reply block carries its form id and its values.
import re
from dataclasses import field, asdict, dataclass
from .parse import parse_message, find_reply
from .coerce import coerce_text, check_canonical
from .validate import interactive_fields, field_label, normalize_label
from .constants import (VIA_BLOCK, VIA_LABELED, VIA_BARE, VIA_CSV, VIA_FREETEXT, ISSUE_TYPE,
                        ISSUE_UNKNOWN_OPTION)

# A line shaped like "word:" is a key we do not know: it ends the value of the previous one
NEXT_KEY_RE = re.compile(r"\n[ \t]*[^\n:]{1,64}:")
TRAILING_RE = re.compile(r"[,;\s]+\Z")


@dataclass(frozen=True)
class ReplyEvent:
    """The normalised answer event of the contract, section 5.3.

    It is frozen because an event is consumed, not modified, and the world reads it as ``event.values["x"]``
    and ``"x" in event.missing``.
    """
    to: str | None = None
    name: str | None = None
    values: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)
    issues: dict = field(default_factory=dict)
    raw: str = ""
    via: str = VIA_FREETEXT
    duplicate: bool = False
    kind: str = "reply"

    def to_dict(self) -> dict:
        """Returns the event as a plain dict, in the field order of the contract."""
        d = asdict(self)
        return {"kind": d.pop("kind"), **d}


def interpret_reply(text, pending=None, answered=None, lang=None) -> ReplyEvent:
    """Interprets an incoming text as an answer to one of the pending forms.

    Args:
        text: The raw message, as it arrived.
        pending: Validated form specs, oldest first (the ones this peer has been asked).
        answered: Form ids already answered, used to flag a duplicate.
        lang: Fallback language, used when a spec declares none.

    Returns:
        A ReplyEvent. An unrecognisable text is never dropped: it comes back as a freetext event.
    """
    raw = text if isinstance(text, str) else ""
    pending = [s for s in (pending or []) if s]
    answered = set(answered or [])
    newest_first = list(reversed(pending))
    parts = parse_message(raw)
    prose = "\n".join(p["text"] for p in parts if p["type"] == "text").strip()

    # Rung 1: canonical block. When the block carries the words it was read from, the LAST of them is the
    # raw of the event (the author's most recent words), not the wire text around the block; a block
    # without them keeps the message text as before (validation only ever stores a non-empty list)
    reply = find_reply(parts)
    if reply is not None:
        event_raw = reply["raw"][-1] if "raw" in reply else raw
        spec = next((s for s in pending if s.get("id") == reply["to"]), None)
        if spec is None:
            return ReplyEvent(to=reply["to"], name=None, values=dict(reply["values"]), raw=event_raw,
                              via=VIA_BLOCK, duplicate=reply["to"] in answered)
        values, issues = apply_canonical(spec, reply["values"])
        return _finish(spec, values, issues, event_raw, VIA_BLOCK, answered)

    if len(newest_first) == 0 or prose == "":
        return _freetext(raw)

    # Rung 2: labeled lines, the pending spec with the most recognised keys wins
    best = None
    for spec in newest_first:
        found = scan_labeled(prose, spec)
        if len(found) == 0:
            continue
        if best is None or len(found) > len(best[1]):
            best = (spec, found)
    if best is not None:
        spec, found = best
        spec_lang = spec.get("lang") or lang
        values, issues = {}, {}
        for f in interactive_fields(spec):
            written = found.get(f["name"])
            if written is None or written == "":
                continue
            result = coerce_text(f, written, lang=spec_lang, allow_decimal_comma=True)
            if result.ok:
                values[f["name"]] = result.value
            else:
                issues[f["name"]] = result.reason
        return _finish(spec, values, issues, raw, VIA_LABELED, answered)

    # Rungs 3 and 4 only look at the newest pending form
    spec = newest_first[0]
    fields = interactive_fields(spec)
    spec_lang = spec.get("lang") or lang

    # Rung 3: single-field form, the whole text is the value. A type mismatch is not a match, because a line
    # of chat is not an answer to a number; a constraint failure is a match, and it is reported.
    if len(fields) == 1:
        f = fields[0]
        result = coerce_text(f, prose, lang=spec_lang, allow_decimal_comma=True)
        if result.ok:
            return _finish(spec, {f["name"]: result.value}, {}, raw, VIA_BARE, answered)
        if result.reason not in (ISSUE_TYPE, ISSUE_UNKNOWN_OPTION):
            return _finish(spec, {}, {f["name"]: result.reason}, raw, VIA_BARE, answered)

    # Rung 4: positional csv, at least two items, at most one per field, every item type-coercing
    if "\n" not in prose:
        items = [x.strip() for x in prose.split(",")]
        if 2 <= len(items) <= len(fields) and all(items):
            values, issues, typed = {}, {}, True
            for i, written in enumerate(items):
                f = fields[i]
                result = coerce_text(f, written, lang=spec_lang, allow_decimal_comma=False)
                if result.ok:
                    values[f["name"]] = result.value
                elif result.reason in (ISSUE_TYPE, ISSUE_UNKNOWN_OPTION):
                    typed = False
                else:
                    issues[f["name"]] = result.reason
            if typed:
                return _finish(spec, values, issues, raw, VIA_CSV, answered)

    # Rung 5
    return _freetext(raw)


def scan_labeled(text: str, spec: dict) -> dict:
    """Finds the "key: value" pairs of one spec in a text.

    A key (field name or label, whitespace-insensitive and case-insensitive) separates only at the start of the
    text, after a newline, or after a comma or semicolon, and only when followed by a colon.

    Returns:
        A dict mapping field name to the raw value string that was written for it.
    """
    fields = interactive_fields(spec)
    keys = []
    for f in fields:
        keys.append((f["name"], f))
        label = field_label(f)
        if normalize_label(label) != normalize_label(f["name"]):
            keys.append((label, f))

    # Longer keys first, so that "Nome completo" beats "Nome" at the same position
    keys.sort(key=lambda item: len(item[0]), reverse=True)

    matches = []
    for key, f in keys:
        pattern = r"\s+".join(re.escape(word) for word in key.strip().split())
        if not pattern:
            continue
        key_re = re.compile(r"(^|\n|[,;])[ \t]*(" + pattern + r")[ \t]*:", re.IGNORECASE)
        for m in key_re.finditer(text):
            matches.append({"start": m.start() + len(m.group(1)), "end": m.end(), "field": f, "len": len(key)})

    matches.sort(key=lambda m: (m["start"], -m["len"]))
    chosen = []
    last_end = -1
    for m in matches:
        if m["start"] < last_end:
            continue
        chosen.append(m)
        last_end = m["end"]

    found = {}
    for i, m in enumerate(chosen):
        stop = chosen[i + 1]["start"] if i + 1 < len(chosen) else len(text)
        value = text[m["end"]:stop]
        next_key = NEXT_KEY_RE.search(value)
        if next_key is not None:
            value = value[:next_key.start()]
        value = TRAILING_RE.sub("", value.strip())
        if m["field"]["name"] not in found:
            found[m["field"]["name"]] = value
    return found


def apply_canonical(spec: dict, incoming: dict) -> tuple[dict, dict]:
    """Type-checks the canonical values of a reply block against a known spec, dropping the unknown keys."""
    values, issues = {}, {}
    for f in interactive_fields(spec):
        if f["name"] not in incoming:
            continue
        result = check_canonical(f, incoming[f["name"]])
        if result.ok:
            values[f["name"]] = result.value
        else:
            issues[f["name"]] = result.reason
    return values, issues


def check_reply(spec: dict, values: dict) -> tuple[dict, list, dict]:
    """Re-verifies the values of a reply against the form that asked for them.

    This is what a world calls on values that came from a peer: they are attacker-controlled JSON until they
    have been through here. Keys the form does not declare are dropped (contract, section 3.6).

    Returns:
        A tuple ``(values, missing, issues)``: the accepted canonical values, the required fields that were not
        answered, and the fields whose value violates the spec, by reason.
    """
    checked, issues = apply_canonical(spec, values if isinstance(values, dict) else {})
    missing = [f["name"] for f in interactive_fields(spec)
               if f.get("required") and f["name"] not in checked and f["name"] not in issues]
    return checked, missing, issues


def parse_reply(text, spec: dict | None = None) -> ReplyEvent | None:
    """Reads a canonical reply block out of a text, the answer a widget produces.

    Args:
        text: The raw message.
        spec: The form spec this text should answer, when the world still has it. Given one, values are
            type-checked and constraint-checked, and required fields that are absent are reported as missing.

    Returns:
        A ReplyEvent, or None when the text carries no reply block (a freetext answer, which the world handles
        with its own fallback).
    """
    if not isinstance(text, str) or find_reply(parse_message(text)) is None:
        return None
    return interpret_reply(text, pending=[spec] if spec else None)


def _finish(spec: dict, values: dict, issues: dict, raw: str, via: str, answered: set) -> ReplyEvent:
    missing = [f["name"] for f in interactive_fields(spec)
               if f.get("required") and f["name"] not in values and f["name"] not in issues]
    return ReplyEvent(to=spec["id"], name=spec["name"], values=values, missing=missing, issues=issues,
                      raw=raw, via=via, duplicate=spec["id"] in answered)


def _freetext(raw: str) -> ReplyEvent:
    return ReplyEvent(to=None, name=None, values={}, missing=[], issues={}, raw=raw, via=VIA_FREETEXT,
                      duplicate=False)
