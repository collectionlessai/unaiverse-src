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
# Canonical serialisation of a validated block: fixed key order per block type (INTERACT-PROTOCOL.md, sections
# 3.1 to 3.6), no whitespace outside strings, wrapped in a uai fence. Canonical so that the Python and the
# JavaScript implementations produce byte-identical output for the same spec, which is what lets the shared
# golden vectors compare on the string.
import json
import math
from .fence import wrap_fence

BLOCK_ORDER = {
    "media": ("v", "type", "src", "mime", "title", "poster", "alt"),
    "data": ("v", "type", "chart", "series", "alt"),
    "form": ("v", "type", "id", "name", "lang", "fields", "progress", "aiHint", "alt"),
    "reply": ("v", "kind", "to", "values", "alt"),
    "x": ("v", "type", "alt"),
}
FIELD_ORDER = ("name", "type", "label", "required", "help", "default",
               "placeholder", "maxLength", "format", "min", "max", "unit", "options", "ui")
OPTION_ORDER = ("value", "label", "help", "media")
SERIES_ORDER = ("label", "points")


def num_to_text(x) -> str:
    """Renders a number as JavaScript does, so that generated text matches byte for byte.

    The difference that matters is the integral float: JavaScript prints 180, Python would print 180.0.
    """
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return str(int(x)) if math.isfinite(x) and x.is_integer() else repr(x)
    return str(x)


def _json_ready(value):
    """Rewrites integral floats as integers, recursively, so that json.dumps matches JSON.stringify."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() else value
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _ordered(obj: dict, order) -> dict:
    """Rebuilds a dict with the given keys first, then any leftover key in alphabetical order."""
    out = {}
    for k in order:
        if k in obj:
            out[k] = obj[k]
    for k in sorted(obj):
        if k not in out:
            out[k] = obj[k]
    return out


def canonical_json(spec: dict) -> str:
    """Returns the canonical JSON body (no fence) of a validated spec."""
    if spec.get("kind") == "reply":
        kind = "reply"
    elif isinstance(spec.get("type"), str) and spec["type"].startswith("x-"):
        kind = "x"
    else:
        kind = spec.get("type")
    order = BLOCK_ORDER.get(kind)
    if order is None:
        raise ValueError(f"Interact: cannot serialize block of type {spec.get('type', spec.get('kind'))}")

    out = _ordered(spec, order)
    if kind == "form":
        fields = []
        for f in spec["fields"]:
            if f.get("type") == "section":
                fields.append({"type": "section", "label": f["label"]})
                continue
            field = _ordered(f, FIELD_ORDER)
            if "options" in field:
                field["options"] = [_ordered(op, OPTION_ORDER) for op in field["options"]]
            fields.append(field)
        out["fields"] = fields
    if kind == "data":
        out["series"] = [_ordered(s, SERIES_ORDER) for s in spec["series"]]
    if kind == "reply":

        # The caller's insertion order is the contract for reply values
        out["values"] = dict(spec["values"])
    return json.dumps(_json_ready(out), ensure_ascii=False, separators=(",", ":"))


def serialize_block(spec: dict) -> str:
    """Returns a validated spec as a fenced block, ready to sit inside a message."""
    return wrap_fence(canonical_json(spec))
