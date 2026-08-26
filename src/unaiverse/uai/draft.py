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
# The answer side of a form (INTERACT-PROTOCOL.md, section 7): the human-readable projection of some values,
# and the one encoding that turns them into a wire message. The widget of the web application produces exactly
# this string, so a Python actor answering a form programmatically produces the same thing.
from .templates import template_for
from .fence import cut_to_bytes
from .coerce import check_canonical
from .serialize import canonical_json, serialize_block, num_to_text
from .constants import PROTOCOL_VERSION, MAX_ALT_CHARS, MAX_BLOCK_BYTES
from .validate import interactive_fields, field_label, non_empty


def project_value(f: dict, v, t: dict) -> str:
    """Renders one canonical value the way a human reads it."""
    if f["type"] == "select":
        labels = {o["value"]: o["label"] for o in f["options"]}
        if isinstance(v, list):
            return ", ".join(labels.get(x, str(x)) for x in v)
        return labels.get(v, str(v))
    if f["type"] == "bool":
        return t["yes"] if v else t["no"]
    if f["type"] in ("number", "integer"):
        return f"{num_to_text(v)} {f['unit']}" if f.get("unit") else num_to_text(v)
    if f["type"] == "date":
        year, month, day = str(v).split("-")
        return t["date"](year, month, day)
    return str(v)


def _is_filled(v) -> bool:
    return not (v is None or v == "" or (isinstance(v, list) and len(v) == 0))


def project_draft(spec: dict, values: dict) -> str:
    """Returns the label-only line that shows what has been filled: form order, filled fields only."""
    _, t = template_for(spec.get("lang"))
    out = []
    for f in interactive_fields(spec):
        v = (values or {}).get(f["name"])
        if not _is_filled(v):
            continue
        out.append(f"{field_label(f)}: {project_value(f, v, t)}")
    return "; ".join(out)


def encode_reply(spec: dict, values: dict, raw: str | list | None = None) -> str:
    """Turns some values into the message that answers a form.

    Only the fields the form declares travel, and a value that fails its own type or constraint check is left
    out, so that the receiver reports it as missing rather than acting on it. A partial answer encodes fine:
    skipping is the way out of a form, not an error.

    What comes out is the shape of the contract (section 7): one block, carrying its own readable projection
    as ``alt``, which is the human rendering of a reply. A projection over the alt cap is cut to it, with no
    marker, so that every implementation produces the same bytes. When ``raw`` is given, the block also
    carries the words the values were read from, verbatim, as a list of texts, oldest first: an answer can
    be built across corrective rounds, and the list is the provenance of the values. The field is
    transparent (every receiver ignores it unless it asks for it), and it is what lets an answer that fell
    short still travel as one block, values and words together, instead of words beside a block.

    Args:
        spec: The validated form being answered.
        values: What was filled in, by field name.
        raw: The words the values were read from: the list of the contributing texts, oldest first (a bare
            string counts as a one-text list).

    Returns:
        The message to send.
    """
    accepted = {}
    for f in interactive_fields(spec):
        v = (values or {}).get(f["name"])
        if not _is_filled(v):
            continue
        result = check_canonical(f, v)
        if result.ok:
            accepted[f["name"]] = result.value
    projection = project_draft(spec, accepted)[:MAX_ALT_CHARS]
    block = {"v": PROTOCOL_VERSION, "kind": "reply", "to": spec["id"], "values": accepted}
    if projection:
        block["alt"] = projection
    raw_texts = [raw] if isinstance(raw, str) else list(raw) if isinstance(raw, (list, tuple)) else []
    raw_texts = [t for t in raw_texts if non_empty(t)]
    if raw_texts:
        block["raw"] = raw_texts

        # The words must never cost the values: a raw that pushes the fenced body past the block cap every
        # receiver enforces is trimmed to fit, oldest texts first, then by cutting the last one (dropped
        # entirely when even a stub cannot fit). A block whose VALUES alone exceed the cap is past raw's
        # reach and degrades on arrival as it always did
        while "raw" in block and len(canonical_json(block).encode("utf-8")) > MAX_BLOCK_BYTES:
            texts = block["raw"]
            if len(texts) > 1:
                block["raw"] = texts[1:]
                continue
            budget = len(texts[0].encode("utf-8")) // 2
            if budget < 16:
                del block["raw"]
            else:
                block["raw"] = [cut_to_bytes(texts[0], budget)]
    return serialize_block(block)
