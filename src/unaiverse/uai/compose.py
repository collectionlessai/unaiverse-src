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
# Authoring of protocol blocks on the sending side. The rule is "emit valid or emit nothing": a block that
# fails validation is rendered by the receiver as its alt (no widget at all), and a block without an alt is
# rendered as raw JSON in a human's chat bubble, so composition raises instead of shipping either.
#
# Blocks are plain dicts here, as they are on the wire. There is no element-constructor sugar yet: a world
# writes the fields it wants and gets back a string ready to be embedded in any message.
import re
import random
from .alt import generate_form_alt
from .fence import fence_spans
from .validate import interactive_fields, validate_block
from .serialize import serialize_block, canonical_json
from .constants import PROTOCOL_VERSION, MAX_ALT_CHARS, MAX_BLOCK_BYTES

# Ids must survive the receiver's own check, [A-Za-z0-9_-] up to 32 characters
_ID_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
_ID_STRIP_RE = re.compile(r"[^A-Za-z0-9_-]+")
_MIME_BY_EXTENSION = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "gif": "image/gif",
                      "webp": "image/webp", "svg": "image/svg+xml", "mp4": "video/mp4", "webm": "video/webm",
                      "mov": "video/quicktime", "mp3": "audio/mpeg", "wav": "audio/wav", "ogg": "audio/ogg",
                      "m4a": "audio/mp4", "pdf": "application/pdf"}

# Own generator, so that composing an id never consumes the global random stream a world may have seeded
_RNG = random.Random()


class UaiError(Exception):
    """Raised when a block cannot be composed. Carries which element failed and why, both optional."""

    def __init__(self, message: str, element: int | None = None, reason: str | None = None) -> None:
        super().__init__(message)
        self.element = element
        self.reason = reason


def gen_id(suffix: str | None = None) -> str:
    """Returns a fresh form id, optionally carrying a caller suffix (sanitised and truncated to fit)."""
    head = "".join(_RNG.choice(_ID_ALPHABET) for _ in range(6))
    if not suffix:
        return head
    tail = _ID_STRIP_RE.sub("", str(suffix))
    return f"{head}-{tail}"[:32] if tail else head


def guess_mime(src: str) -> str | None:
    """Guesses a MIME type from the extension of a URL, or None when the URL does not say.

    None rather than a generic binary type on purpose: a media block whose MIME says nothing is valid for
    the protocol and undrawable for a renderer, which is the one outcome composing here refuses to produce.
    """
    if not isinstance(src, str):
        return None
    ext = src.rsplit("?", 1)[0].rsplit("#", 1)[0].rsplit(".", 1)[-1].lower() if "." in src else ""
    return _MIME_BY_EXTENSION.get(ext)


def _validated(spec: dict, what: str) -> dict:
    result = validate_block(spec)
    if not result["ok"]:
        reason = result["errors"][0] if result["errors"] else None
        raise UaiError(f"uai: invalid {what} block ({', '.join(result['errors'])})", reason=reason)
    return result["spec"]


def _check_size(spec: dict, what: str) -> None:
    body = canonical_json(spec)
    if len(body.encode("utf-8")) > MAX_BLOCK_BYTES:
        raise UaiError(f"uai: {what} block is too large ({len(body.encode('utf-8'))} bytes, "
                            f"the limit is {MAX_BLOCK_BYTES}); split it over more messages",
                            reason="block_too_large")


def build_form(name: str, fields: list, form_id: str | None = None, lang: str | None = None,
               alt: str | None = None, ai_hint: str | None = None, progress=None) -> dict:
    """Builds and validates a form spec, generating the model instruction when the author gives none.

    Args:
        name: The form name, shown to a human and used to refer to the form.
        fields: The field dicts, in the wire vocabulary (name, type, label, required, options, ...).
        form_id: The id an answer refers to. Generated when omitted; a form sent to several recipients must
            use a different id per recipient, or one answer marks every copy answered.
        lang: Language tag, which selects the wording of the generated instruction and of the parsing.
        alt: The model instruction. Generated from the spec when omitted, which is the normal case.
        ai_hint: Replaces the whole generated instruction, for a form that needs a bespoke one.
        progress: A pair (step, total), when the form is one page of a sequence.

    Returns:
        The normalised, validated spec, to be kept by the world so that it can check the answer later.

    Raises:
        UaiError: If the spec is not valid, if the generated instruction is too long, or if the block
            exceeds the size a receiver accepts.
    """
    spec = {"v": PROTOCOL_VERSION, "type": "form", "id": form_id or gen_id(), "name": name}
    if lang is not None:
        spec["lang"] = lang
    spec["fields"] = list(fields or [])
    if progress is not None:
        spec["progress"] = list(progress)
    if ai_hint is not None:
        spec["aiHint"] = ai_hint

    # Validation needs an alt to be there; the generated one is built from the validated spec and put in after,
    # exactly as the JavaScript mirror does.
    spec["alt"] = alt if alt is not None else "?"
    validated = _validated(spec, "form")
    if alt is None:
        generated = generate_form_alt(validated)
        if len(generated) > MAX_ALT_CHARS:
            raise UaiError(f"uai: the generated instruction of form '{validated['id']}' is "
                                f"{len(generated)} characters, the limit is {MAX_ALT_CHARS}; shorten the "
                                f"labels, split the form, or pass your own alt", reason="alt_too_long")
        validated["alt"] = generated
    _check_size(validated, "form")
    return validated


def compose_form(name: str, fields: list, **kwargs) -> str:
    """Builds a form and returns it as a fenced block. See build_form for the arguments."""
    return serialize_block(build_form(name, fields, **kwargs))


def build_media(src: str, alt: str, mime: str | None = None, title: str | None = None,
                poster: str | None = None) -> dict:
    """Builds and validates a media spec. The source must be an https URL, and alt is what a model reads."""
    guessed = mime or guess_mime(src)
    if guessed is None:
        raise UaiError(f"uai: cannot tell the MIME type of '{src}' from its URL; pass mime= "
                            f"(a media block that does not say what it carries cannot be drawn)",
                            reason="mime_unknown")
    spec = {"v": PROTOCOL_VERSION, "type": "media", "src": src, "mime": guessed}
    if title is not None:
        spec["title"] = title
    if poster is not None:
        spec["poster"] = poster
    spec["alt"] = alt
    validated = _validated(spec, "media")
    _check_size(validated, "media")
    return validated


def compose_media(src: str, alt: str, **kwargs) -> str:
    """Builds a media block and returns it as a fenced block. See build_media for the arguments."""
    return serialize_block(build_media(src, alt, **kwargs))


def build_chart(chart: str, series: list, alt: str) -> dict:
    """Builds and validates a chart spec. A caption belongs in the prose beside it: the block carries none."""
    spec = {"v": PROTOCOL_VERSION, "type": "data", "chart": chart, "series": list(series or []), "alt": alt}
    validated = _validated(spec, "data")
    _check_size(validated, "data")
    return validated


def compose_chart(chart: str, series: list, alt: str) -> str:
    """Builds a chart and returns it as a fenced block. See build_chart for the arguments."""
    return serialize_block(build_chart(chart, series, alt))


def build_re_ask(spec: dict, only: list | None = None, form_id: str | None = None, lang: str | None = None,
                 progress=None) -> dict:
    """Asks again for a form that was not satisfied, keeping only the fields still worth asking about.

    This is what a world does with what it learns from an answer: the fields nobody said anything about and
    the ones whose value it could not accept are exactly ``event.missing + list(event.issues)``. A form left
    with a single field is also the easiest one to answer, since a bare value is enough for it.

    Args:
        spec: The validated form that was asked the first time.
        only: The field names to ask again, or None for all of them. Section rows are not carried over.
        form_id: The id of the new question. Generated when omitted, and it must differ from the first, or
            an answer to one would look like an answer to the other.
        lang: Language of the new question, defaulting to the one of the original.
        progress: A pair (step, total), when the re-ask is one page of a sequence.

    Returns:
        The validated spec of the new form.

    Raises:
        UaiError: If nothing is left to ask.
    """
    fields = [dict(f) for f in interactive_fields(spec) if only is None or f["name"] in only]
    if len(fields) == 0:
        raise UaiError(f"uai: nothing left to ask of form '{spec.get('id')}'", reason="no_fields")
    return build_form(spec.get("name", "form"), fields, form_id=form_id or gen_id(),
                      lang=lang if lang is not None else spec.get("lang"), progress=progress)


def re_ask(spec: dict, only: list | None = None, **kwargs) -> str:
    """Asks again, as a fenced block ready to sit in a message. See build_re_ask for the arguments."""
    return serialize_block(build_re_ask(spec, only=only, **kwargs))


def truncate_outside_fences(text, max_chars: int) -> str:
    """Shortens a message without ever cutting inside a fence.

    A partial fence is not a block any more: the receiver shows the leftover JSON as prose. So when the cut
    would fall inside a block, the whole block goes, together with what follows it.
    """
    if not isinstance(text, str) or max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = max_chars
    for start, end in fence_spans(text):
        if start < cut < end:
            cut = start
            break
    return text[:cut].rstrip()
