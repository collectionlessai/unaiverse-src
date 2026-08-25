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
# Runs the shared golden vectors of the interactive-messages protocol (INTERACT-PROTOCOL.md, section 12)
# against the Python implementation. The very same JSON files are run by the JavaScript suite
# (unaiverse-js, src/unaiverse/Interact/test/run-vectors.mjs), so a divergence between the two
# implementations of the one grammar fails here, on one side or on the other.
#
# The files under vectors/ are a copy of unaiverse-js, src/unaiverse/Interact/test/vectors/ taken at commit
# f5f5834 (126 cases). Re-syncing them is a copy; a case discovered in the field is added there first, then here.
#
# Two operations are not implemented on this side yet, and are skipped explicitly rather than silently:
# "compose", which needs the element-constructor API this delivery does not ship, and "registry", which needs
# the pending-request registry. Everything else is expected to pass.
import json
import pathlib
import pytest
from unaiverse.uai import (encode_reply, generate_form_alt, interpret_reply, parse_message, project_draft,
                                serialize_block, to_model_text, validate_block)

VECTORS = pathlib.Path(__file__).parent / "vectors"
NOT_IMPLEMENTED_OPS = {"compose", "registry"}


def _op_parse(inp: dict) -> dict:
    return {"parts": parse_message(inp["text"])}


def _op_model_text(inp: dict) -> dict:
    return {"text": to_model_text(inp["text"])}


def _op_alt(inp: dict) -> dict:
    result = validate_block(inp["spec"])
    if not result["ok"]:
        return {"error": ",".join(result["errors"])}
    return {"alt": generate_form_alt(result["spec"])}


def _op_serialize(inp: dict) -> dict:
    result = validate_block(inp["spec"])
    if not result["ok"]:
        return {"error": ",".join(result["errors"])}
    return {"text": serialize_block(result["spec"])}


def _op_interpret(inp: dict) -> dict:
    pending = []
    for spec in inp.get("pending", []):
        result = validate_block(spec)
        assert result["ok"], f"pending spec invalid: {result['errors']}"
        pending.append(result["spec"])
    event = interpret_reply(inp["text"], pending=pending, answered=inp.get("answered", []), lang=inp.get("lang"))
    return event.to_dict()


def _op_project(inp: dict) -> dict:
    spec = validate_block(inp["spec"])["spec"]
    return {"text": project_draft(spec, inp.get("draft", {}).get("values", {}))}


def _op_encode(inp: dict) -> dict:
    spec = validate_block(inp["spec"])["spec"]
    return {"text": encode_reply(spec, inp.get("draft", {}).get("values", {}))}


OPS = {"parse": _op_parse, "modelText": _op_model_text, "alt": _op_alt, "serialize": _op_serialize,
       "interpret": _op_interpret, "project": _op_project, "encode": _op_encode}


def _matches(actual, expect, path: str = "") -> str | None:
    """Compares as the JavaScript runner does: every key of expect must be equal, arrays in full, and an
    expected error only has to be contained in the actual message. Keys absent from expect are not checked."""
    if isinstance(expect, list):
        if not isinstance(actual, list) or len(actual) != len(expect):
            return f"{path}: expected an array of {len(expect)}, got {actual!r}"
        for i, (a, e) in enumerate(zip(actual, expect)):
            diff = _matches(a, e, f"{path}[{i}]")
            if diff is not None:
                return diff
        return None
    if isinstance(expect, dict):
        if not isinstance(actual, dict):
            return f"{path}: expected an object, got {actual!r}"
        for k, e in expect.items():
            here = f"{path}.{k}" if path else k
            if k not in actual:
                return f"{here}: missing from the result"
            diff = _matches(actual[k], e, here)
            if diff is not None:
                return diff
        return None
    if isinstance(expect, str) and isinstance(actual, str) and path.endswith("error"):
        return None if expect in actual else f"{path}: expected an error containing {expect!r}, got {actual!r}"
    if isinstance(expect, bool) != isinstance(actual, bool):
        return f"{path}: expected {expect!r}, got {actual!r}"
    if actual == expect:
        return None
    return f"{path}: expected {expect!r}, got {actual!r}"


def _load_cases():
    cases = []
    for path in sorted(VECTORS.glob("*.json")):
        suite = json.loads(path.read_text(encoding="utf-8"))
        for case in suite["cases"]:
            cases.append(pytest.param(suite["suite"], case, id=f"{suite['suite']}/{case['id']}"))
    return cases


@pytest.mark.parametrize("suite, case", _load_cases())
def test_vector(suite: str, case: dict):
    if case["op"] in NOT_IMPLEMENTED_OPS:
        pytest.skip(f"operation '{case['op']}' is not part of this delivery")
    actual = OPS[case["op"]](case["input"])
    diff = _matches(actual, case["expect"])
    assert diff is None, f"{suite}/{case['id']}: {diff}\n  actual: {json.dumps(actual, ensure_ascii=False)[:600]}"
