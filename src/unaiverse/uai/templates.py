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
# Per-language wording for the generated model instruction (INTERACT-PROTOCOL.md, section 4) and for the human
# projection (section 7), plus the words the tolerant parser accepts. One table feeds all three, so the
# renderings and the parsing cannot drift apart. These strings are contract: the golden vectors pin them.
# Adding a language is adding an entry here, nothing else changes.
import re
from .serialize import num_to_text

_IT_DATE_RE = re.compile(r"([0-9]{1,2})[/-]([0-9]{1,2})[/-]([0-9]{4})")
_EN_DATE_RE = re.compile(r"([0-9]{1,2})[/-]([0-9]{1,2})[/-]([0-9]{4})")

TEMPLATES = {
    "it": {
        "progress": lambda step, total: f"Passo {num_to_text(step)} di {num_to_text(total)}. ",
        "intro": "Rispondi con righe 'campo: valore'. Campi: ",
        "types": {"text": "testo", "textarea": "testo lungo", "number": "numero", "integer": "numero intero",
                  "bool": "sì o no", "date": "data AAAA-MM-GG"},
        "select_one": lambda labels: "una tra: " + ", ".join(labels),
        "select_many": lambda lo, hi, labels: f"da {num_to_text(lo)} a {num_to_text(hi)} tra: " + ", ".join(labels),
        "select_some": lambda labels: "una o più tra: " + ", ".join(labels),
        "unit": lambda u: f"in {u}",
        "between": lambda a, b: f"tra {num_to_text(a)} e {num_to_text(b)}",
        "at_least": lambda a: f"almeno {num_to_text(a)}",
        "at_most": lambda b: f"al massimo {num_to_text(b)}",
        "max_length": lambda n: f"massimo {num_to_text(n)} caratteri",
        "format": {"email": "formato email", "url": "formato url", "tel": "formato telefono"},
        "required": "obbligatorio",
        "optional": "facoltativo",
        "yes": "Sì",
        "no": "No",

        # Human projection of dates: day first
        "date": lambda y, m, d: f"{d}/{m}/{y}",

        # Words the tolerant parser accepts for booleans, lower case
        "true_words": ("sì", "si", "vero", "yes", "true", "1"),
        "false_words": ("no", "falso", "false", "0"),

        # Date shapes the tolerant parser accepts after ISO, as (pattern, group meaning)
        "date_shapes": ((_IT_DATE_RE, ("d", "m", "y")),),
    },
    "en": {
        "progress": lambda step, total: f"Step {num_to_text(step)} of {num_to_text(total)}. ",
        "intro": "Reply with lines 'field: value'. Fields: ",
        "types": {"text": "text", "textarea": "long text", "number": "number", "integer": "integer",
                  "bool": "yes or no", "date": "date YYYY-MM-DD"},
        "select_one": lambda labels: "one of: " + ", ".join(labels),
        "select_many": lambda lo, hi, labels: f"{num_to_text(lo)} to {num_to_text(hi)} of: " + ", ".join(labels),
        "select_some": lambda labels: "one or more of: " + ", ".join(labels),
        "unit": lambda u: f"in {u}",
        "between": lambda a, b: f"between {num_to_text(a)} and {num_to_text(b)}",
        "at_least": lambda a: f"at least {num_to_text(a)}",
        "at_most": lambda b: f"at most {num_to_text(b)}",
        "max_length": lambda n: f"at most {num_to_text(n)} characters",
        "format": {"email": "format email", "url": "format url", "tel": "format phone"},
        "required": "required",
        "optional": "optional",
        "yes": "Yes",
        "no": "No",
        "date": lambda y, m, d: f"{m}/{d}/{y}",
        "true_words": ("yes", "true", "1", "y"),
        "false_words": ("no", "false", "0", "n"),
        "date_shapes": ((_EN_DATE_RE, ("m", "d", "y")),),
    },
}


def template_for(lang) -> tuple[str, dict]:
    """Resolves a language tag (or anything else) to a template table, falling back to English."""
    primary = str(lang or "").lower().split("-")[0]
    if primary and primary in TEMPLATES:
        return primary, TEMPLATES[primary]
    return "en", TEMPLATES["en"]
