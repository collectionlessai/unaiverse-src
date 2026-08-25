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
# The model-facing rendering of a form: a deterministic instruction built from the spec (INTERACT-PROTOCOL.md,
# section 4). Select options are listed by LABEL, because a model writes words far more readily than
# identifiers, and the interpreter maps them back. The author's aiHint replaces the whole instruction.
from .templates import template_for
from .validate import interactive_fields, field_label


def describe_field(f: dict, t: dict) -> str:
    """Describes one field for the instruction, in the descriptor order fixed by the contract."""
    d = []
    if f["type"] == "select":
        labels = [o["label"] for o in f["options"]]
        if f["max"] == 1:
            d.append(t["select_one"](labels))
        elif f.get("min") is not None and f["min"] > 0:
            d.append(t["select_many"](f["min"], f["max"], labels))
        elif f["max"] < len(f["options"]):
            d.append(t["select_many"](f.get("min", 0), f["max"], labels))
        else:
            d.append(t["select_some"](labels))
    else:
        d.append(t["types"][f["type"]])
    if f.get("unit"):
        d.append(t["unit"](f["unit"]))
    if f["type"] in ("number", "integer", "date"):
        if "min" in f and "max" in f:
            d.append(t["between"](f["min"], f["max"]))
        elif "min" in f:
            d.append(t["at_least"](f["min"]))
        elif "max" in f:
            d.append(t["at_most"](f["max"]))
    if "maxLength" in f:
        d.append(t["max_length"](f["maxLength"]))
    if f.get("format"):
        d.append(t["format"][f["format"]])
    d.append(t["required"] if f.get("required") else t["optional"])
    return f"{field_label(f)} ({', '.join(d)})"


def generate_form_alt(spec: dict) -> str:
    """Returns the generated instruction for a validated form spec, honouring aiHint."""
    hint = spec.get("aiHint")
    if isinstance(hint, str) and hint.strip():
        return hint
    _, t = template_for(spec.get("lang"))
    head = t["progress"](spec["progress"][0], spec["progress"][1]) if "progress" in spec else ""
    fields = [describe_field(f, t) for f in interactive_fields(spec)]
    return f"{head}{t['intro']}{', '.join(fields)}."
