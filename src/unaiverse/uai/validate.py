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
# Structural validation of protocol blocks (INTERACT-PROTOCOL.md, section 3). Pure: it takes a parsed JSON
# object and returns {"ok", "spec", "errors", "kind"}. On success "spec" is a normalised copy (defaults filled,
# unknown keys dropped, section rows kept in place). It never raises on bad input: a malformed object is just
# not ok, with reasons, so that a hostile peer can at most make a block degrade.
import re
import math
import calendar
import urllib.parse
from .constants import (PROTOCOL_VERSION, BLOCK_TYPES, FIELD_TYPES, CHART_KINDS, TEXT_FORMATS, MAX_ALT_CHARS,
                        MAX_FIELDS_PER_FORM, MAX_FIELD_ITEMS, MAX_OPTIONS_PER_SELECT, MAX_LABEL_CHARS,
                        MAX_HELP_CHARS, MAX_UNIT_CHARS, MAX_SERIES_PER_CHART, MAX_POINTS_PER_SERIES)

# Character classes are spelled out because Python, unlike JavaScript, would match non-ASCII digits with \d
ID_RE = re.compile(r"[A-Za-z0-9_-]{1,32}")
NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
ISO_DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
MIME_RE = re.compile(r"[a-z0-9.+*-]+/[a-z0-9.+*-]+", re.IGNORECASE)
LANG_RE = re.compile(r"[a-z]{2,3}(-[A-Za-z0-9]{2,8})*")


def is_str(v) -> bool:
    return isinstance(v, str)


def is_int(v) -> bool:
    """Mirrors JavaScript's Number.isInteger: a float with no fractional part is an integer, a bool is not."""
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True
    return isinstance(v, float) and math.isfinite(v) and v.is_integer()


def is_num(v) -> bool:
    return not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)


def is_bool(v) -> bool:
    return isinstance(v, bool)


def is_obj(v) -> bool:
    return isinstance(v, dict)


def non_empty(v) -> bool:
    return isinstance(v, str) and len(v.strip()) > 0


def is_https_url(s) -> bool:
    """Accepts https and nothing else, so that a hostile javascript: or data: URL never survives."""
    if not isinstance(s, str):
        return False
    try:
        parts = urllib.parse.urlsplit(s)
    except ValueError:
        return False
    return parts.scheme == "https" and len(parts.netloc) > 0


def normalize_label(s) -> str:
    """Normalises a label for uniqueness and for tolerant matching: trim, collapse whitespace runs, lower case.

    This is deliberately the whole rule: no diacritics folding (the JavaScript mirror has none either, and
    stripping accents would reject as duplicates two option labels that the receiver accepts) and no casefold
    (which would map some characters differently from JavaScript's toLowerCase).
    """
    return re.sub(r"\s+", " ", ("" if s is None else str(s)).strip()).lower()


def is_iso_date(s) -> bool:
    """Tells whether a string is a valid ISO calendar date (YYYY-MM-DD, existing day)."""
    if not isinstance(s, str) or ISO_DATE_RE.fullmatch(s) is None:
        return False
    year, month, day = (int(x) for x in s.split("-"))
    if month < 1 or month > 12 or day < 1 or year < 1:
        return False
    return day <= calendar.monthrange(year, month)[1]


def interactive_fields(spec) -> list[dict]:
    """Returns the interactive fields of a validated form spec, skipping the section rows."""
    fields = (spec or {}).get("fields") or []
    return [f for f in fields if f.get("type") != "section"]


def field_label(f: dict) -> str:
    """Returns the human label of a field, falling back to its name when the author gave none."""
    return f["label"] if "label" in f else f["name"]


def validate_block(obj) -> dict:
    """Validates one parsed block object.

    Args:
        obj: The result of json.loads on a fence body.

    Returns:
        A dict ``{"ok": bool, "spec": dict | None, "errors": list[str], "kind": "content" | "reply" | None}``.
    """
    errors: list[str] = []
    if not is_obj(obj):
        return {"ok": False, "spec": None, "errors": ["not_an_object"], "kind": None}

    if not is_int(obj.get("v")):
        errors.append("v_missing")
    elif obj["v"] > PROTOCOL_VERSION:
        errors.append("v_unsupported")

    has_type = is_str(obj.get("type"))
    has_kind = is_str(obj.get("kind"))
    if has_type == has_kind:
        errors.append("type_and_kind" if has_type else "type_or_kind_missing")
        return {"ok": False, "spec": None, "errors": errors, "kind": None}

    if has_kind:
        if obj["kind"] != "reply":
            errors.append("kind_unknown")
        if not is_str(obj.get("to")) or ID_RE.fullmatch(obj["to"]) is None:
            errors.append("to_invalid")
        if not is_obj(obj.get("values")):
            errors.append("values_invalid")
        if errors:
            return {"ok": False, "spec": None, "errors": errors, "kind": "reply"}
        spec = {"v": obj["v"], "kind": "reply", "to": obj["to"], "values": dict(obj["values"])}

        # The readable projection of the answer travels inside the block (contract, section 3.6), and it is
        # the human rendering of a reply. It is never fatal: one that is missing, empty, not a string or
        # over the cap is dropped, and the block stands on its own.
        alt = obj.get("alt")
        if isinstance(alt, str) and alt.strip() and len(alt) <= MAX_ALT_CHARS:
            spec["alt"] = alt
        return {"ok": True, "spec": spec, "errors": errors, "kind": "reply"}

    # Content blocks: alt is mandatory before anything else is looked at
    if not non_empty(obj.get("alt")):
        errors.append("alt_missing")
    elif len(obj["alt"]) > MAX_ALT_CHARS:
        errors.append("alt_too_long")

    block_type = obj["type"]
    if block_type.startswith("x-"):
        if errors:
            return {"ok": False, "spec": None, "errors": errors, "kind": "content"}
        return {"ok": True, "spec": {"v": obj["v"], "type": block_type, "alt": obj["alt"]},
                "errors": errors, "kind": "content"}
    if block_type not in BLOCK_TYPES:
        errors.append("type_unknown")
        return {"ok": False, "spec": None, "errors": errors, "kind": "content"}
    if errors:
        return {"ok": False, "spec": None, "errors": errors, "kind": "content"}

    spec = None
    if block_type == "media":
        spec = _validate_media(obj, errors)
    elif block_type == "data":
        spec = _validate_data(obj, errors)
    elif block_type == "form":
        spec = _validate_form(obj, errors)

    if errors:
        return {"ok": False, "spec": None, "errors": errors, "kind": "content"}
    return {"ok": True, "spec": spec, "errors": errors, "kind": "content"}


def _validate_media(o: dict, errors: list) -> dict | None:
    if not is_https_url(o.get("src")):
        errors.append("src_invalid")
    if not non_empty(o.get("mime")) or MIME_RE.fullmatch(o["mime"]) is None:
        errors.append("mime_invalid")
    if "title" in o and (not is_str(o["title"]) or len(o["title"]) > MAX_LABEL_CHARS):
        errors.append("title_invalid")
    if "poster" in o and not is_https_url(o["poster"]):
        errors.append("poster_invalid")
    if errors:
        return None
    spec = {"v": o["v"], "type": "media", "src": o["src"], "mime": o["mime"].lower()}
    if "title" in o:
        spec["title"] = o["title"]
    if "poster" in o:
        spec["poster"] = o["poster"]
    spec["alt"] = o["alt"]
    return spec


def _validate_data(o: dict, errors: list) -> dict | None:
    if o.get("chart") not in CHART_KINDS:
        errors.append("chart_invalid")
    raw_series = o.get("series")
    if not isinstance(raw_series, list) or len(raw_series) < 1 or len(raw_series) > MAX_SERIES_PER_CHART:
        errors.append("series_invalid")
        return None
    series = []
    for s in raw_series:
        if (not is_obj(s) or not is_str(s.get("label")) or not isinstance(s.get("points"), list)
                or len(s["points"]) > MAX_POINTS_PER_SERIES):
            errors.append("series_item_invalid")
            return None
        points = []
        for p in s["points"]:
            if (not isinstance(p, list) or len(p) != 2 or not (is_str(p[0]) or is_num(p[0])) or not is_num(p[1])):
                errors.append("point_invalid")
                return None
            points.append([p[0], p[1]])
        series.append({"label": s["label"], "points": points})
    if errors:
        return None

    # Any other key (a chart library configuration, a callback) is dropped here, never read
    return {"v": o["v"], "type": "data", "chart": o["chart"], "series": series, "alt": o["alt"]}


def _validate_form(o: dict, errors: list) -> dict | None:
    if not is_str(o.get("id")) or ID_RE.fullmatch(o["id"]) is None:
        errors.append("id_invalid")
    if not non_empty(o.get("name")) or len(o["name"]) > MAX_LABEL_CHARS:
        errors.append("name_invalid")
    if "lang" in o and (not is_str(o["lang"]) or LANG_RE.fullmatch(o["lang"]) is None):
        errors.append("lang_invalid")
    if "aiHint" in o and (not is_str(o["aiHint"]) or len(o["aiHint"]) > MAX_ALT_CHARS):
        errors.append("aiHint_invalid")
    if "progress" in o:
        p = o["progress"]
        if (not isinstance(p, list) or len(p) != 2 or not is_int(p[0]) or not is_int(p[1])
                or p[0] < 1 or p[1] < 1 or p[0] > p[1]):
            errors.append("progress_invalid")
    raw_fields = o.get("fields")
    if not isinstance(raw_fields, list) or len(raw_fields) < 1 or len(raw_fields) > MAX_FIELD_ITEMS:
        errors.append("fields_invalid")
        return None

    fields = []
    names = set()
    interactive = 0
    for f in raw_fields:
        if not is_obj(f):
            errors.append("field_invalid")
            return None
        if f.get("type") == "section":
            if not non_empty(f.get("label")) or len(f["label"]) > MAX_LABEL_CHARS:
                errors.append("section_invalid")
                return None
            fields.append({"type": "section", "label": f["label"]})
            continue
        interactive += 1
        spec = _validate_field(f, errors)
        if spec is None:
            return None
        if spec["name"] in names:
            errors.append(f"name_duplicate:{spec['name']}")
            return None
        names.add(spec["name"])
        fields.append(spec)
    if interactive < 1:
        errors.append("no_interactive_field")
    if interactive > MAX_FIELDS_PER_FORM:
        errors.append("too_many_fields")
    if errors:
        return None

    spec = {"v": o["v"], "type": "form", "id": o["id"], "name": o["name"]}
    if "lang" in o:
        spec["lang"] = o["lang"]
    spec["fields"] = fields
    if "progress" in o:
        spec["progress"] = [o["progress"][0], o["progress"][1]]
    if "aiHint" in o:
        spec["aiHint"] = o["aiHint"]
    spec["alt"] = o["alt"]
    return spec


def _validate_field(f: dict, errors: list) -> dict | None:
    """Validates one interactive field. An unknown type degrades to "text", keeping only the common keys."""
    if not is_str(f.get("name")) or NAME_RE.fullmatch(f["name"]) is None:
        errors.append("field_name_invalid")
        return None
    field_type = f["type"] if f.get("type") in FIELD_TYPES else "text"
    degraded = field_type != f.get("type")
    spec = {"name": f["name"], "type": field_type}
    if "label" in f:
        if not is_str(f["label"]) or len(f["label"]) > MAX_LABEL_CHARS:
            errors.append(f"label_invalid:{f['name']}")
            return None
        spec["label"] = f["label"]
    if "required" in f:
        if not is_bool(f["required"]):
            errors.append(f"required_invalid:{f['name']}")
            return None
        spec["required"] = f["required"]
    if "help" in f:
        if not is_str(f["help"]) or len(f["help"]) > MAX_HELP_CHARS:
            errors.append(f"help_invalid:{f['name']}")
            return None
        spec["help"] = f["help"]
    if degraded:
        # A field type from the future: text input, common keys only, default kept if it is a string
        if is_str(f.get("default")):
            spec["default"] = f["default"]
        return spec

    if field_type in ("text", "textarea"):
        if not _validate_text_field(f, spec, field_type, errors):
            return None
    elif field_type in ("number", "integer"):
        if not _validate_number_field(f, spec, field_type, errors):
            return None
    elif field_type == "bool":
        if "default" in f:
            if not is_bool(f["default"]):
                errors.append(f"default_invalid:{f['name']}")
                return None
            spec["default"] = f["default"]
    elif field_type == "date":
        if not _validate_date_field(f, spec, errors):
            return None
    elif field_type == "select":
        if not _validate_select_field(f, spec, errors):
            return None

    # Unknown hints are carried, renderers ignore what they do not know
    if "ui" in f and is_str(f["ui"]):
        spec["ui"] = f["ui"]
    return spec


def _validate_text_field(f: dict, spec: dict, field_type: str, errors: list) -> bool:
    if "placeholder" in f:
        if not is_str(f["placeholder"]) or len(f["placeholder"]) > MAX_HELP_CHARS:
            errors.append(f"placeholder_invalid:{f['name']}")
            return False
        spec["placeholder"] = f["placeholder"]
    if "maxLength" in f:
        if not is_int(f["maxLength"]) or f["maxLength"] < 1:
            errors.append(f"maxLength_invalid:{f['name']}")
            return False
        spec["maxLength"] = f["maxLength"]
    if field_type == "text" and "format" in f:
        if f["format"] not in TEXT_FORMATS:
            errors.append(f"format_invalid:{f['name']}")
            return False
        spec["format"] = f["format"]
    if "default" in f:
        if not is_str(f["default"]):
            errors.append(f"default_invalid:{f['name']}")
            return False
        spec["default"] = f["default"]
    return True


def _validate_number_field(f: dict, spec: dict, field_type: str, errors: list) -> bool:
    ok = is_int if field_type == "integer" else is_num
    for key in ("min", "max"):
        if key in f:
            if not ok(f[key]):
                errors.append(f"{key}_invalid:{f['name']}")
                return False
            spec[key] = f[key]
    if "min" in spec and "max" in spec and spec["min"] > spec["max"]:
        errors.append(f"range_invalid:{f['name']}")
        return False
    if "unit" in f:
        if not non_empty(f["unit"]) or len(f["unit"]) > MAX_UNIT_CHARS:
            errors.append(f"unit_invalid:{f['name']}")
            return False
        spec["unit"] = f["unit"]
    if "default" in f:
        if not ok(f["default"]):
            errors.append(f"default_invalid:{f['name']}")
            return False
        spec["default"] = f["default"]
    return True


def _validate_date_field(f: dict, spec: dict, errors: list) -> bool:
    for key in ("min", "max"):
        if key in f:
            if not is_iso_date(f[key]):
                errors.append(f"{key}_invalid:{f['name']}")
                return False
            spec[key] = f[key]
    if "min" in spec and "max" in spec and spec["min"] > spec["max"]:
        errors.append(f"range_invalid:{f['name']}")
        return False
    if "default" in f:
        if not is_iso_date(f["default"]):
            errors.append(f"default_invalid:{f['name']}")
            return False
        spec["default"] = f["default"]
    return True


def _validate_select_field(f: dict, spec: dict, errors: list) -> bool:
    raw_options = f.get("options")
    if not isinstance(raw_options, list) or len(raw_options) < 1 or len(raw_options) > MAX_OPTIONS_PER_SELECT:
        errors.append(f"options_invalid:{f['name']}")
        return False
    options = []
    seen_labels = set()
    seen_values = set()
    for op in raw_options:
        if (not is_obj(op) or not is_str(op.get("value")) or not non_empty(op.get("label"))
                or len(op["label"]) > MAX_LABEL_CHARS):
            errors.append(f"option_invalid:{f['name']}")
            return False
        if "," in op["label"]:
            errors.append(f"option_label_comma:{f['name']}")
            return False
        normalized = normalize_label(op["label"])
        if normalized in seen_labels:
            errors.append(f"option_label_duplicate:{f['name']}")
            return False
        if op["value"] in seen_values:
            errors.append(f"option_value_duplicate:{f['name']}")
            return False
        seen_labels.add(normalized)
        seen_values.add(op["value"])
        option = {"value": op["value"], "label": op["label"]}
        if "help" in op:
            if not is_str(op["help"]) or len(op["help"]) > MAX_HELP_CHARS:
                errors.append(f"option_help_invalid:{f['name']}")
                return False
            option["help"] = op["help"]
        if "media" in op:
            if not is_https_url(op["media"]):
                errors.append(f"option_media_invalid:{f['name']}")
                return False
            option["media"] = op["media"]
        options.append(option)
    spec["options"] = options

    max_selected = 1 if "max" not in f else f["max"]
    if not is_int(max_selected) or max_selected < 1:
        errors.append(f"max_invalid:{f['name']}")
        return False
    spec["max"] = max_selected
    if "min" in f:
        if not is_int(f["min"]) or f["min"] < 0 or f["min"] > max_selected:
            errors.append(f"min_invalid:{f['name']}")
            return False
        spec["min"] = f["min"]
    if "default" in f:
        values = {o["value"] for o in options}
        if max_selected == 1:
            if not is_str(f["default"]) or f["default"] not in values:
                errors.append(f"default_invalid:{f['name']}")
                return False
        elif (not isinstance(f["default"], list)
                or not all(is_str(v) and v in values for v in f["default"])):
            errors.append(f"default_invalid:{f['name']}")
            return False
        spec["default"] = f["default"]
    return True
