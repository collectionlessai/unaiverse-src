# setup.py
import os
import platform
import subprocess
import importlib.util
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.bdist_wheel import bdist_wheel

HERE = Path(__file__).parent.resolve()
P2P_DIR = HERE / "src" / "unaiverse" / "networking" / "p2p"
GO_DIR = P2P_DIR / "unailib"
BUILD_PURE_WHEEL = os.environ.get("UNAI_BUILD_PURE", "0") == "1"

# Artifacts of the previous build layout: the loader no longer reads them, so they are removed to keep a
# source checkout from carrying several stale copies of the library
LEGACY_ARTIFACTS = ("unailib*.so", "unailib*.dylib", "unailib*.dll", "unailib*.pyd", "unailib*.h",
                    "unailib*.sha256")


def _load_unailib_source():
    # Loaded by path: importing the package would pull in its dependencies and the library being built
    spec = importlib.util.spec_from_file_location("_unailib_source", P2P_DIR / "_unailib_source.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


unailib_source = _load_unailib_source()


class GoBuildExt(build_ext):
    """Builds the Go shared library in place of a C extension."""

    def get_ext_filename(self, fullname):
        # A plain shared library loaded through ctypes: its name carries no Python ABI tag
        *package, _ = fullname.split(".")
        return os.path.join(*package, unailib_source.lib_filename(platform.system()))

    def run(self):
        for path in (p for d in (P2P_DIR, GO_DIR) for pattern in LEGACY_ARTIFACTS for p in d.glob(pattern)):
            path.unlink(missing_ok=True)

        for ext in self.extensions:
            dest = Path(self.get_ext_fullpath(ext.name)).resolve()
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Always handed to the Go toolchain, whose build cache makes an unchanged rebuild cheap; the source
            # hash is embedded so that the loader can verify the library it loads
            ldflags = f"-s -w -X main.buildSourceHash={unailib_source.source_hash(GO_DIR)}"

            # Exactly the toolchain go.mod declares (Go fetches it if the local one differs): with the default
            # GOTOOLCHAIN=auto a newer local Go would win, and Go 1.26 builds a library whose quic-go v0.54
            # panics at the first QUIC handshake
            toolchain = f"go{unailib_source.go_version(GO_DIR)}"
            print(f"--- Building the Go library into {dest} with {toolchain} ---")
            subprocess.run(['go', 'build', '-buildmode=c-shared', '-ldflags', ldflags, '-o', str(dest), '.'],
                           check=True, cwd=GO_DIR, env=dict(os.environ, GOTOOLCHAIN=toolchain))

            # The C header that c-shared writes next to the library is not used: ctypes declares the prototypes
            dest.with_suffix(".h").unlink(missing_ok=True)


class PlatformWheel(bdist_wheel):
    """One wheel per platform: the library is loaded through ctypes and does not depend on the Python ABI."""

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, plat = super().get_tag()
        return "py3", "none", plat


if BUILD_PURE_WHEEL:

    # Pure Python wheel for Pyodide, without the Go library
    ext_modules = []
    cmdclass = {}
    print("--- Building Pure Python Wheel (No Go Extensions) ---")
else:
    ext_modules = [Extension("unaiverse.networking.p2p._unailib", sources=[])]
    cmdclass = {'build_ext': GoBuildExt, 'bdist_wheel': PlatformWheel}

setup(
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    zip_safe=False,
)
