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
# Extraction of ```uai fenced blocks from a message, and their re-insertion. This is the only place that knows
# the fence grammar (INTERACT-PROTOCOL.md, section 1). No JSON is parsed here: the caller gets raw bodies plus
# the surrounding prose, in order, with the byte limit already applied to each body.
import re
from .constants import FENCE_TAG, MAX_BLOCK_BYTES

# Opening fence: up to 3 spaces, 3+ backticks, optional spaces, the tag, optional spaces
OPEN_RE = re.compile(r" {0,3}(`{3,})[ \t]*" + FENCE_TAG + r"[ \t]*")

# Cheap pre-check, used as a fast path before any real work is done on a message
HAS_FENCE_RE = re.compile(r"```[ \t]*" + FENCE_TAG + r"[ \t]*(\r?\n|$)")

# Fence tests ignore one trailing carriage return, so that CRLF input behaves like LF input
CR_RE = re.compile(r"\r\Z")


def has_fence(text) -> bool:
    """Tells whether a text contains at least one candidate opening fence."""
    return isinstance(text, str) and HAS_FENCE_RE.search(text) is not None


def wrap_fence(body: str) -> str:
    """Wraps a canonical JSON body in a uai fence."""
    return "```" + FENCE_TAG + "\n" + body + "\n```"


def cut_to_bytes(s: str, max_bytes: int) -> str:
    """Cuts a string so that its UTF-8 encoding fits in max_bytes, on a character boundary."""
    out = []
    used = 0
    for ch in s:
        n = len(ch.encode("utf-8"))
        if used + n > max_bytes:
            break
        out.append(ch)
        used += n
    return "".join(out)


def extract_fences(text) -> list[dict]:
    """Splits a message into an ordered list of prose and raw block segments.

    Args:
        text: The message to split.

    Returns:
        A list of dicts. Prose segments are ``{"kind": "prose", "text", "start", "end"}`` and are never empty;
        block segments are ``{"kind": "block", "raw", "bytes", "too_large", "start", "end"}``, where "start" and
        "end" are character offsets of the whole fence in the original text. A block whose raw body exceeds the
        byte limit comes back with ``too_large`` and its "raw" already cut: it must never be parsed.
    """
    segments: list[dict] = []
    if not isinstance(text, str) or len(text) == 0:
        return segments

    lines = text.split("\n")

    # Character offset of the beginning of each line, plus one past the end of the text
    offsets = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1
    offsets.append(pos)

    prose: list[str] = []
    prose_start = 0
    i = 0

    def flush_prose(end_offset: int) -> None:
        if len(prose) == 0:
            return
        # Blank lines that only separate prose from a fence are not content
        joined = re.sub(r"\A\n+|\n+\Z", "", "\n".join(prose))
        if joined.strip() != "":
            segments.append({"kind": "prose", "text": joined, "start": prose_start, "end": end_offset})
        prose.clear()

    while i < len(lines):
        open_match = OPEN_RE.fullmatch(CR_RE.sub("", lines[i]))
        if open_match is None:
            if len(prose) == 0:
                prose_start = offsets[i]
            prose.append(lines[i])
            i += 1
            continue

        # Find the closing fence: at least as many backticks, nothing else
        ticks = len(open_match.group(1))
        close_re = re.compile(r" {0,3}`{" + str(ticks) + r",}[ \t]*")
        j = i + 1
        while j < len(lines) and close_re.fullmatch(CR_RE.sub("", lines[j])) is None:
            j += 1
        if j >= len(lines):
            # Unterminated: the fence is prose, as the contract prescribes
            if len(prose) == 0:
                prose_start = offsets[i]
            prose.append(lines[i])
            i += 1
            continue

        flush_prose(offsets[i])
        raw_full = "\n".join(lines[i + 1:j])
        n_bytes = len(raw_full.encode("utf-8"))
        too_large = n_bytes > MAX_BLOCK_BYTES
        segments.append({"kind": "block",
                         "raw": cut_to_bytes(raw_full, MAX_BLOCK_BYTES) if too_large else raw_full,
                         "bytes": n_bytes, "too_large": too_large,
                         "start": offsets[i], "end": min(offsets[j] + len(lines[j]), len(text))})
        i = j + 1

    flush_prose(len(text))
    return segments


def fence_spans(text) -> list[tuple[int, int]]:
    """Returns the character spans of the fenced blocks of a message, in order."""
    return [(s["start"], s["end"]) for s in extract_fences(text) if s["kind"] == "block"]
