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

# Identity of the Go shared library: its file name and the hash of the sources it is built from.
# setup.py embeds the hash into the library at link time, and the package loader compares it with the
# sources it finds next to the library. setup.py loads this file by path, without importing the package,
# so it must depend on the standard library only.
import re
import hashlib
from pathlib import Path

LIB_STEM = "_unailib"
LIB_SUFFIXES = {"Linux": ".so", "Darwin": ".dylib", "Windows": ".dll"}


def lib_filename(system: str) -> str:
    """File name of the shared library on the given OS (as returned by platform.system())."""
    try:
        return LIB_STEM + LIB_SUFFIXES[system]
    except KeyError:
        raise RuntimeError(f"Unsupported OS for the UNaIVERSE p2p library: {system}") from None


def go_version(go_dir: str | Path) -> str:
    """Go toolchain version declared in go.mod: its toolchain directive if present, otherwise the go one."""
    text = (Path(go_dir) / "go.mod").read_text()
    match = re.search(r"^toolchain go(\S+)", text, re.MULTILINE) or re.search(r"^go (\S+)", text, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"No go directive in {Path(go_dir) / 'go.mod'}")
    version = match.group(1)

    # Since Go 1.21 a release is always named with its patch number (go1.24.0, not go1.24)
    return version if version.count(".") >= 2 else version + ".0"


def source_files(go_dir: str | Path) -> list[Path]:
    """Every file the library is built from: the Go sources at any depth, go.mod and go.sum."""
    root = Path(go_dir)
    files = set(root.rglob("*.go"))
    files.update(p for p in (root / "go.mod", root / "go.sum") if p.is_file())
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def source_hash(go_dir: str | Path) -> str | None:
    """SHA-256 over the relative paths and contents of the library sources, None if go_dir is missing."""
    root = Path(go_dir)
    if not root.is_dir():
        return None
    digest = hashlib.sha256()
    for path in source_files(root):

        # The path is part of the digest, so that renaming or moving a file changes it too
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()
