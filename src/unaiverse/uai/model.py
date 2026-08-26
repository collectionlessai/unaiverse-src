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
# The model-facing rendering of a whole message (INTERACT-PROTOCOL.md, section 4): every block replaced by its
# alt, prose untouched, order kept. A reply block has no alt, so its model rendering is a "name: value" listing
# of the values, the same shape the model is asked to write.
#
# The two caps are a Python-side addition, not protocol: the contract bounds a single block but not a whole
# message, so without them a peer could make a model read an unbounded amount of text.
import json
from .fence import cut_to_bytes
from .parse import parse_message
from .serialize import num_to_text
from .constants import MAX_BLOCKS_MODEL_VIEW, MAX_MODEL_VIEW_BYTES


def to_model_text(text, max_blocks: int = MAX_BLOCKS_MODEL_VIEW, max_bytes: int = MAX_MODEL_VIEW_BYTES) -> str:
    """Returns what a model should read in place of a message that may carry protocol blocks.

    Args:
        text: The raw message.
        max_blocks: How many blocks are rendered before the rest is summarised in one line.
        max_bytes: Upper bound, in UTF-8 bytes, of the returned text.

    Returns:
        The message with every block replaced by its model rendering, prose and order untouched.
    """
    return parts_to_model_text(parse_message(text), max_blocks=max_blocks, max_bytes=max_bytes)


def parts_to_model_text(parts: list, max_blocks: int = MAX_BLOCKS_MODEL_VIEW,
                        max_bytes: int = MAX_MODEL_VIEW_BYTES) -> str:
    """Same as to_model_text, for a caller that has already parsed the message."""
    out = []
    blocks = 0
    omitted = 0
    for p in parts:
        if p["type"] == "text":
            out.append(p["text"])
            continue
        blocks += 1
        if blocks > max_blocks:
            omitted += 1
            continue
        if p["type"] == "reply":

            # A reply reads as its values and, when the block carries them, as the latest words they were
            # read from: a partial answer keeps the phrasing (and whatever the values missed) visible,
            # and a failed one reads as its words alone, never as an empty line
            lines = "\n".join(f"{k}: {_value_to_text(v)}" for k, v in p["spec"]["values"].items())
            raw_texts = p["spec"].get("raw")
            raw = raw_texts[-1] if raw_texts else ""
            out.append(f"{lines}\n{raw}" if lines and raw else (lines or raw))
        else:
            out.append(p["alt"])
    if omitted > 0:
        out.append(f"[... {omitted} more blocks omitted]")

    rendered = "\n".join(out)
    if len(rendered.encode("utf-8")) > max_bytes:
        rendered = cut_to_bytes(rendered, max_bytes) + "\n[... truncated]"
    return rendered


def _value_to_text(v) -> str:
    """Renders one reply value for a model, the same way the JavaScript mirror does."""
    if isinstance(v, list):
        return ", ".join(_value_to_text(x) for x in v)
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    if isinstance(v, float):
        return num_to_text(v)
    return str(v)
