#!/usr/bin/env python3
"""
Compares two API surfaces written by pyodide_smoke.mjs (plan, Phase 0.7): the baseline wheel the browser runs today
and a candidate wheel.

    python tests/tools/browser_compat/api_surface_diff.py <baseline.json> <candidate.json> [--js-src <dir>]

Removed modules, names and class members, and incompatible call shapes are reported (annotations are ignored; new
trailing parameters with a default are compatible, any other change is not). One whose name appears in the JS SDK
sources (--js-src, default the SDK next to unaiverse-src) is an ERROR: the browser may call it. The others are
warnings. Additions are only counted. Exit code 1 on any ERROR or on a module that failed to import in the candidate.
"""
import re
import sys
import json
import argparse
from pathlib import Path

DEFAULT_JS_SRC = Path(__file__).resolve().parents[4] / "unaiverse-js" / "unaiverse-js-src" / "src"
ADDRESS = re.compile(r" at 0x[0-9a-f]+")  # Default values printed as "<function f at 0x...>" differ at every run


def js_identifiers(js_src: Path) -> set[str]:
    """Collects every identifier-like word in the JS SDK sources (tests excluded).

    Args:
        js_src: The SDK source folder.

    Returns:
        The set of words.
    """
    words = set()
    for f in js_src.rglob("*"):
        if f.suffix in (".js", ".mjs", ".ts") and "test" not in f.parts and f.is_file():
            words.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", f.read_text(encoding="utf-8", errors="ignore")))
    return words


def same(a: str, b: str) -> bool:
    """Tells if two call shapes are the same, ignoring the memory addresses in the printed defaults.

    Args:
        a: The first call shape.
        b: The second call shape.

    Returns:
        True if they are the same.
    """
    return ADDRESS.sub("", a) == ADDRESS.sub("", b)


def _params(sig: str) -> tuple[bool, list[str]] | None:
    """Splits a printed call shape into its async-ness and its parameters (brackets and quotes kept whole).

    Args:
        sig: The call shape, e.g. "async (self, a, b=None)".

    Returns:
        (is async, parameters), or None when it cannot be read.
    """
    sig = ADDRESS.sub("", sig).strip()
    is_async = sig.startswith("async ")
    sig = sig.removeprefix("async ").strip()
    if not (sig.startswith("(") and sig.endswith(")")):
        return None
    params, depth, cur, quote = [], 0, "", None
    for ch in sig[1:-1]:
        if quote:
            quote = None if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            params.append(cur.strip())
            cur = ""
            continue
        cur += ch
    if cur.strip():
        params.append(cur.strip())
    return is_async, params


def compatible(old: str, new: str) -> bool:
    """Tells if a caller of the old call shape still works with the new one.

    That is: the same async-ness, the old parameters unchanged and in the same places, and the new ones only at the
    end, each with a default (or *args / **kwargs, or keyword-only after them).

    Args:
        old: The old call shape.
        new: The new call shape.

    Returns:
        True if compatible.
    """
    a, b = _params(old), _params(new)
    if a is None or b is None or a[0] != b[0] or b[1][:len(a[1])] != a[1]:
        return False
    return all("=" in p or p.startswith("*") for p in b[1][len(a[1]):])


def diff(base: dict, cand: dict) -> tuple[list[tuple[str, str]], int]:
    """Lists the incompatible differences between two surfaces.

    Args:
        base: The baseline surface.
        cand: The candidate surface.

    Returns:
        The (name the JS would use, description) of each difference, and the number of additions.
    """
    found, added = [], 0
    for mod, names in base["modules"].items():
        if mod not in cand["modules"]:
            found.append((mod.rsplit(".", 1)[-1], f"module {mod} removed or not importable"))
            continue
        cnames = cand["modules"][mod]
        added += len(set(cnames) - set(names))
        for name, info in names.items():
            if name not in cnames:
                found.append((name, f"{mod}.{name} removed"))
                continue
            cinfo = cnames[name]
            if info["kind"] != cinfo["kind"]:
                found.append((name, f"{mod}.{name} changed from {info['kind']} to {cinfo['kind']}"))
            elif info["kind"] == "function" and not compatible(info["sig"], cinfo["sig"]):
                found.append((name, f"{mod}.{name}{info['sig']} -> {cinfo['sig']}"))
            elif info["kind"] == "class":
                added += len(set(cinfo["members"]) - set(info["members"]))
                for member, sig in info["members"].items():
                    if member not in cinfo["members"]:
                        found.append((member, f"{mod}.{name}.{member} removed (was {sig})"))
                    elif sig not in ("property", "?") and not compatible(sig, cinfo["members"][member]):
                        found.append((member, f"{mod}.{name}.{member}{sig} -> {cinfo['members'][member]}"))
                    elif sig in ("property", "?") and not same(sig, cinfo["members"][member]):
                        found.append((member, f"{mod}.{name}.{member}{sig} -> {cinfo['members'][member]}"))
    return found, added


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--js-src", type=Path, default=DEFAULT_JS_SRC)
    args = parser.parse_args()
    base = json.loads(Path(args.baseline).read_text())
    cand = json.loads(Path(args.candidate).read_text())
    words = js_identifiers(args.js_src)
    found, added = diff(base, cand)
    errors = [d for n, d in found if n in words]
    warnings = [d for n, d in found if n not in words]
    for m, tb in cand.get("errors", {}).items():
        errors.append(f"module {m} fails to import in Pyodide: {tb.strip().splitlines()[-1]}")
    for d in errors:
        print(f"ERROR    {d}")
    for d in warnings:
        print(f"warning  {d}")
    print(f"{len(errors)} errors (names the JS SDK sources mention), {len(warnings)} warnings, {added} additions")
    sys.exit(1 if errors else 0)
