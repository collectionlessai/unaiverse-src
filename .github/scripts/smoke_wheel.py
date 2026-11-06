"""Smoke test of a built wheel, run on the build runner without installing it.

It checks the wheel layout (one py3-none wheel with exactly one Go library and no Go sources), then loads
the library through ctypes and reads the source hash embedded at build time. Nothing of the package is
imported, so none of its dependencies is needed.
"""
import sys
import ctypes
import zipfile
import tempfile
from pathlib import Path


def fail(message: str) -> None:
    sys.exit(f"::error::{message}")


def main(paths: list[str]) -> None:
    if len(paths) != 1:
        fail(f"expected exactly one wheel for this platform, found {len(paths)}: {paths}")
    wheel = Path(paths[0])
    if "-py3-none-" not in wheel.name:
        fail(f"{wheel.name} is not a py3-none wheel")

    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
        libs = [n for n in names if Path(n).stem == "_unailib" and not n.endswith(".py")]
        if len(libs) != 1:
            fail(f"expected exactly one _unailib library in {wheel.name}, found {libs}")
        sources = [n for n in names if "/p2p/unailib/" in n]
        if sources:
            fail(f"{wheel.name} carries the Go sources: {sources[:5]}")

        # Not a TemporaryDirectory: on Windows a loaded DLL cannot be deleted, and the runner is thrown away
        lib_path = zf.extract(libs[0], tempfile.mkdtemp())

    lib = ctypes.CDLL(lib_path)
    lib.UnailibBuildHash.argtypes = []
    lib.UnailibBuildHash.restype = ctypes.c_void_p
    lib.FreeString.argtypes = [ctypes.c_void_p]
    lib.FreeString.restype = None
    ptr = lib.UnailibBuildHash()
    build_hash = ctypes.string_at(ptr).decode()
    lib.FreeString(ptr)
    if len(build_hash) != 64:
        fail(f"the library carries no source hash ({build_hash!r}): it was not built by setup.py")

    print(f"{wheel.name}: {libs[0]} loads, built from sources {build_hash}")


if __name__ == "__main__":
    main(sys.argv[1:])
