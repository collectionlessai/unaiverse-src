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
# From a message string to an ordered list of typed parts: the rendering path of INTERACT-PROTOCOL.md sections
# 2 and 13, minus any drawing. Every failure degrades, nothing raises on input, and one bad block never
# affects the rest of the message.
import json
from .fence import extract_fences
from .validate import validate_block


def parse_message(text) -> list[dict]:
    """Parses a message into ordered parts.

    Args:
        text: The message, as it arrived on the wire.

    Returns:
        A list of parts. Prose is ``{"type": "text", "text"}``; a valid content block is
        ``{"type": "media" | "data" | "form" | "x", "spec", "alt"}``; a valid reply block is
        ``{"type": "reply", "spec": {"to", "values"}}``; anything invalid degrades to
        ``{"type": "text", "text", "degraded": True, "errors": [...]}``.
    """
    parts = []
    for seg in extract_fences(text):
        if seg["kind"] == "prose":
            parts.append({"type": "text", "text": seg["text"]})
        else:
            parts.append(_block_to_part(seg))
    return parts


def _block_to_part(seg: dict) -> dict:
    """Turns one raw fence body into a part, applying the degradation table of the contract."""
    if seg["too_large"]:
        return {"type": "text", "text": seg["raw"], "degraded": True, "errors": ["too_large"]}
    try:
        obj = json.loads(seg["raw"])
    except (ValueError, RecursionError):
        return {"type": "text", "text": seg["raw"], "degraded": True, "errors": ["json_invalid"]}

    result = validate_block(obj)
    if not result["ok"]:

        # Degrade to the alt when there is one worth showing, to the raw body otherwise
        alt = obj.get("alt") if isinstance(obj, dict) else None
        alt = alt if isinstance(alt, str) and alt.strip() else None
        return {"type": "text", "text": alt if alt is not None else seg["raw"],
                "degraded": True, "errors": result["errors"]}

    spec = result["spec"]
    if result["kind"] == "reply":

        # The alt of a reply is its projection: a part always carries one, empty when the block had none
        return {"type": "reply", "spec": {"to": spec["to"], "values": spec["values"]}, "alt": spec.get("alt", "")}
    if spec["type"].startswith("x-"):
        return {"type": "x", "spec": spec, "alt": spec["alt"]}
    return {"type": spec["type"], "spec": spec, "alt": spec["alt"]}


def has_interactive_parts(parts: list[dict]) -> bool:
    """Tells whether a parsed message contains at least one non-text part."""
    return any(p["type"] != "text" for p in parts)


def find_form(parts: list[dict]) -> dict | None:
    """Returns the first form spec of a parsed message, if any."""
    return next((p["spec"] for p in parts if p["type"] == "form"), None)


def newest_form(parts: list[dict]) -> dict | None:
    """Returns the last form spec of a parsed message, which is the one an answer refers to."""
    return next((p["spec"] for p in reversed(parts) if p["type"] == "form"), None)


def find_reply(parts: list[dict]) -> dict | None:
    """Returns the first reply spec of a parsed message, if any."""
    return next((p["spec"] for p in parts if p["type"] == "reply"), None)
