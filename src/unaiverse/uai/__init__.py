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
# Interactive messages: the Python side of the protocol a world uses to put forms, media and charts inside a
# chat message that stays one string on the wire (INTERACT-PROTOCOL.md, shared with the JavaScript SDK in
# unaiverse-js, src/unaiverse/Interact/). This package turns strings into typed data and back, and nothing
# else: no torch, no networking, no dependency on the rest of the library, so that a world without a
# processor can use it from its actions and the same code runs in the browser.
#
# The public API is this module: import from unaiverse.uai, never from its submodules, so that the
# internal layout can follow the JavaScript one without touching any caller.
from .fill import (AnswerOutcome, AnswerWithheld, describe_answer, encode_partial, fill_slots, read_answer,
                   reprompt_person, retry_prompt)
from .model import parts_to_model_text, to_model_text
from .templates import TEMPLATES, template_for
from .alt import describe_field, generate_form_alt
from .draft import encode_reply, project_draft, project_value
from .serialize import canonical_json, serialize_block, num_to_text
from .fence import extract_fences, fence_spans, has_fence, wrap_fence
from .parse import find_form, find_reply, has_interactive_parts, newest_form, parse_message
from .compose import (UaiError, build_chart, build_form, build_media, build_re_ask, compose_chart,
                      compose_form, compose_media, gen_id, guess_mime, re_ask,
                      truncate_outside_fences)
from .coerce import Coerced, check_canonical, check_constraints, coerce_text, parse_date, parse_number
from .interpret import ReplyEvent, apply_canonical, check_reply, interpret_reply, parse_reply, scan_labeled
from .validate import (field_label, interactive_fields, is_https_url, is_iso_date, normalize_label,
                       validate_block)
from .constants import (BLOCK_TYPES, CHART_KINDS, DEFAULT_LANG, FENCE_TAG, FIELD_TYPES, ISSUE_ABOVE_MAX,
                        ISSUE_BELOW_MIN, ISSUE_FORMAT, ISSUE_TOO_LONG, ISSUE_TYPE, ISSUE_UNKNOWN_OPTION,
                        LANGS, MAX_ALT_CHARS, MAX_BLOCK_BYTES, MAX_BLOCKS_MODEL_VIEW, MAX_INBOX_PEERS,
                        MAX_MODEL_VIEW_BYTES,
                        MAX_OUTPUT_BYTES, PROTOCOL_VERSION, TEXT_FORMATS, VIA_BARE, VIA_BLOCK, VIA_CSV,
                        VIA_FREETEXT, VIA_LABELED)

__all__ = [
    # Reading what arrived
    "parse_message", "to_model_text", "parts_to_model_text", "parse_reply", "check_reply",
    "interpret_reply", "ReplyEvent", "find_form", "newest_form", "find_reply", "has_interactive_parts",
    "fill_slots", "read_answer", "describe_answer", "retry_prompt", "reprompt_person", "encode_partial",
    "AnswerOutcome", "AnswerWithheld",
    # Writing what leaves
    "build_form", "compose_form", "build_media", "compose_media", "build_chart", "compose_chart",
    "encode_reply", "project_draft", "gen_id", "guess_mime", "truncate_outside_fences", "UaiError",
    "re_ask", "build_re_ask",
    # Lower level, for a world that needs the details
    "validate_block", "serialize_block", "canonical_json", "generate_form_alt", "describe_field",
    "interactive_fields", "field_label", "normalize_label", "is_iso_date", "is_https_url",
    "extract_fences", "fence_spans", "has_fence", "wrap_fence", "scan_labeled", "apply_canonical",
    "coerce_text", "check_canonical", "check_constraints", "parse_number", "parse_date", "project_value",
    "num_to_text", "Coerced", "TEMPLATES", "template_for",
    # Contract terms
    "FENCE_TAG", "PROTOCOL_VERSION", "BLOCK_TYPES", "FIELD_TYPES", "CHART_KINDS", "TEXT_FORMATS",
    "LANGS", "DEFAULT_LANG", "MAX_BLOCK_BYTES", "MAX_ALT_CHARS", "MAX_BLOCKS_MODEL_VIEW",
    "MAX_INBOX_PEERS",
    "MAX_MODEL_VIEW_BYTES", "MAX_OUTPUT_BYTES", "VIA_BLOCK", "VIA_LABELED", "VIA_BARE", "VIA_CSV",
    "VIA_FREETEXT", "ISSUE_TYPE", "ISSUE_BELOW_MIN", "ISSUE_ABOVE_MAX", "ISSUE_TOO_LONG", "ISSUE_FORMAT",
    "ISSUE_UNKNOWN_OPTION",
]
