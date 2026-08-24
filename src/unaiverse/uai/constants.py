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
# Numbers and closed vocabularies of the interactive-messages protocol (INTERACT-PROTOCOL.md, sections 3, 5, 6, 9).
# Every value here is a contract term: change it in the contract first, then in the JavaScript SDK
# (unaiverse-js, src/unaiverse/Interact/constants.js), then here.

# The info string of the fenced code block that carries a protocol block
FENCE_TAG = "uai"

# Protocol version implemented here: a block declaring a higher one degrades to its alt
PROTOCOL_VERSION = 1

# Content block types (the ones carrying "type" and requiring "alt")
BLOCK_TYPES = ("media", "data", "form")

# Interactive field types of v1; an unknown type degrades to "text"
FIELD_TYPES = ("text", "textarea", "number", "integer", "bool", "date", "select")

# Chart kinds accepted in a "data" block
CHART_KINDS = ("bar", "line", "pie")

# Closed list of "format" values for text fields: never a regular expression on the wire
TEXT_FORMATS = ("email", "url", "tel")

# Presentation hints known on "select"; unknown ones travel untouched and renderers ignore them
SELECT_UI = ("buttons",)

# Which rung of the tolerance ladder produced a reply event (contract, section 5.3)
VIA_BLOCK = "block"
VIA_LABELED = "labeled"
VIA_BARE = "bare"
VIA_CSV = "csv"
VIA_FREETEXT = "freetext"

# Closed list of constraint-violation reasons (contract, section 5.2)
ISSUE_TYPE = "type"
ISSUE_BELOW_MIN = "below_min"
ISSUE_ABOVE_MAX = "above_max"
ISSUE_TOO_LONG = "too_long"
ISSUE_FORMAT = "format"
ISSUE_UNKNOWN_OPTION = "unknown_option"

# Hard limits, checked on raw input before parsing (contract, section 9)
MAX_BLOCK_BYTES = 8192
MAX_FIELDS_PER_FORM = 20
MAX_FIELD_ITEMS = 40
MAX_OPTIONS_PER_SELECT = 50
MAX_LABEL_CHARS = 120
MAX_NAME_CHARS = 64
MAX_HELP_CHARS = 240
MAX_ALT_CHARS = 2000
MAX_UNIT_CHARS = 16
MAX_SERIES_PER_CHART = 10
MAX_POINTS_PER_SERIES = 500

# Languages with an instruction template; anything else falls back to English
LANGS = ("it", "en")
DEFAULT_LANG = "en"

# Python-side extensions, not protocol. The contract caps a single block but not a whole message, while the
# network layer accepts very large ones: these bound what a model can be made to read and what the slot
# filler may append, so a hostile or buggy peer cannot stuff a prompt or bust a world's own size limits.
MAX_BLOCKS_MODEL_VIEW = 16
MAX_MODEL_VIEW_BYTES = 65536
MAX_OUTPUT_BYTES = 16384
MAX_INBOX_PEERS = 32
