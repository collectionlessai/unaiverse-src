"""
Checks that the Python sources follow the UNaIVERSE code style.

The rules are the ones of the core files: lines of at most 120 characters (comments and docstrings included), a
single block of imports sorted by increasing line length (the first line, for a long import wrapped in parentheses)
with no empty lines, no `from __future__` imports, types on
every parameter and return value, a docstring on every class, function and method ("(async)" at the end of the
summary of an async function), an empty line before an own-line comment and no empty line right after it, public
methods before private ones. Only the files listed in STYLED_FILES are checked: a file joins the list once it has
been brought to the style, so the list grows until it covers the whole package.
"""
import ast
import pytest
from pathlib import Path

pytestmark = pytest.mark.unit

SRC = Path(__file__).resolve().parent.parent / "src" / "unaiverse"
MAX_LINE = 120
BANNED_TYPING = {"Optional", "List", "Dict", "Tuple", "Set", "Union"}

# The files already brought to the style (paths relative to src/unaiverse)
STYLED_FILES = [
    "agent.py",
    "agent_basics.py",
    "custom.py",
    "interaction.py",
    "modules/utils.py",
    "networking/node/connpool.py",
    "networking/node/node.py",
    "networking/p2p/messages.py",
    "networking/p2p/p2p.py",
    "networking/session/__init__.py",
    "networking/session/sess_msg.py",
    "networking/session/session.py",
    "networking/session/session_manager.py",
]


def _banner_end(lines: list[str]) -> int:
    """Returns the index of the first line after the module docstring (the ASCII-art banner), or 0 if missing.

    Args:
        lines: The lines of the file.

    Returns:
        The index of the first line after the closing quotes of the module docstring.
    """
    if not lines or not lines[0].startswith('"""'):
        return 0
    for i in range(1, len(lines)):
        if lines[i].rstrip().endswith('"""'):
            return i + 1
    return 0


def _is_private(name: str) -> bool:
    """Tells whether a method name is private (one or two leading underscores, not a dunder).

    Args:
        name: The method name.

    Returns:
        True for `_name` and `__name`, False for public names and for dunders such as `__init__`.
    """
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def check_file(path: Path) -> list[str]:
    """Checks one source file against the style rules.

    Args:
        path: The path of the Python file.

    Returns:
        The list of the violations found, each one as "line: message" (empty when the file is fine).
    """
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    tree = ast.parse(src)
    issues = []

    # Line length (the banner docstring is exempt, its box characters are wide)
    start = _banner_end(lines)
    for i, line in enumerate(lines[start:], start=start + 1):
        if len(line) > MAX_LINE:
            issues.append(f"{i}: line longer than {MAX_LINE} characters ({len(line)})")

    # Imports: one block at the top, sorted by increasing length of the first line, no empty lines, no __future__
    top = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and not top:
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            top.append(node)
        else:
            break
    for node in top:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            issues.append(f"{node.lineno}: `from __future__` import")
    for a, b in zip(top, top[1:]):
        if len(lines[b.lineno - 1]) < len(lines[a.lineno - 1]):
            issues.append(f"{b.lineno}: import shorter than the one before it")
        if b.lineno - a.end_lineno > 1:
            issues.append(f"{b.lineno}: empty line between imports")

    # Old-style typing generics (Optional[...], List[...], ..., also as typing.Optional[...])
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            base = node.value
            name = base.id if isinstance(base, ast.Name) else base.attr if isinstance(base, ast.Attribute) else None
            if name in BANNED_TYPING:
                issues.append(f"{node.lineno}: `{name}[...]` typing, use `X | None`, `list[...]`, `dict[...]`")

    # Functions and methods (module level and class level; nested helpers are not checked)
    def check_function(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Checks the annotations and the docstring of one function.

        Args:
            fn: The function node.
        """
        args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
        for a in args:
            if a.arg not in ("self", "cls") and a.annotation is None:
                issues.append(f"{fn.lineno}: parameter `{a.arg}` of `{fn.name}` has no type")
        for a in (fn.args.vararg, fn.args.kwarg):
            if a is not None and a.annotation is None:
                issues.append(f"{fn.lineno}: parameter `{a.arg}` of `{fn.name}` has no type")
        if fn.returns is None:
            issues.append(f"{fn.lineno}: `{fn.name}` has no return type")
        doc = ast.get_docstring(fn)
        if doc is None:
            issues.append(f"{fn.lineno}: `{fn.name}` has no docstring")
        elif isinstance(fn, ast.AsyncFunctionDef) and "(async)" not in doc.split("\n\n")[0]:
            issues.append(f"{fn.lineno}: async `{fn.name}` without \"(async)\" in the docstring summary")

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            check_function(node)
        elif isinstance(node, ast.ClassDef):
            if ast.get_docstring(node) is None:
                issues.append(f"{node.lineno}: class `{node.name}` has no docstring")
            seen_private = None
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    check_function(item)
                    if _is_private(item.name):
                        seen_private = seen_private or item.name
                    elif seen_private is not None:
                        issues.append(f"{item.lineno}: public `{item.name}` after private `{seen_private}`")

    # Own-line comments: an empty line before the first comment line, none right after the last one
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("#") or i < start:
            continue
        prev = lines[i - 1].strip() if i > 0 else ""
        if prev.startswith("#"):
            continue
        if prev != "":
            issues.append(f"{i + 1}: own-line comment without an empty line before it")
        j = i + 1
        while j < len(lines) and lines[j].strip().startswith("#"):
            j += 1
        if j < len(lines) and lines[j].strip() == "":
            issues.append(f"{j}: empty line right after an own-line comment")
    return issues


@pytest.mark.parametrize("rel_path", STYLED_FILES)
def test_file_follows_the_code_style(rel_path: str) -> None:
    """Every styled file has no violation.

    Args:
        rel_path: The path of the file, relative to src/unaiverse.
    """
    issues = check_file(SRC / rel_path)
    assert not issues, f"{rel_path}:\n" + "\n".join(issues)
