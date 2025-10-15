# setup.py
import os
import platform
import subprocess
import hashlib
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


GO_SOURCE_NAME = 'lib.go'
HASH_FILE_SUFFIX = '.sha256'

def get_ext_filename_with_path():
    system = platform.system()
    if system == 'Linux':
        lib_name = 'unailib.so'
    elif system == 'Darwin':
        lib_name = 'unailib.dylib'
    elif system == 'Windows':
        lib_name = 'unailib.dll'
    else:
        raise RuntimeError(f"Unsupported OS: {system}")
    return os.path.join('src', 'unaiverse', 'networking', 'p2p', lib_name)

def get_go_source_dir():
    return os.path.join('src', 'unaiverse', 'networking', 'p2p')

def get_file_hash(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

class GoBuildExtCommand(build_ext):
    """Custom build_ext that compiles Go lib if needed and marks wheel as native."""
    def run(self):
        go_dir = get_go_source_dir()
        go_path = os.path.join(go_dir, GO_SOURCE_NAME)
        out_path = get_ext_filename_with_path()
        hash_path = go_path + HASH_FILE_SUFFIX
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        current_hash = get_file_hash(go_path)
        stored_hash = None
        if os.path.exists(hash_path):
            with open(hash_path, 'r') as f:
                stored_hash = f.read().strip()

        # Only rebuild if Go source changed or lib missing
        if current_hash != stored_hash or not os.path.exists(out_path):
            print(f"--- Go source changed, building {out_path} ---")
            subprocess.run(
                ['go', 'build', '-buildmode=c-shared', '-ldflags', '-s -w',
                 '-o', os.path.basename(out_path), GO_SOURCE_NAME],
                check=True, cwd=go_dir
            )
            with open(hash_path, 'w') as f:
                f.write(current_hash)
        else:
            print("--- Go source unchanged; skipping build. ---")

        # Trick: make setuptools think there’s a compiled ext
        for ext in self.extensions:
            ext.sources = []  # prevents clang call
        super().run()


# Fake extension only to mark wheel as platform-dependent
go_extension = Extension(
    "unaiverse.networking.p2p.unailib",
    sources=["src/unaiverse/networking/p2p/lib.go"],  # fake input
)

setup(
    cmdclass={'build_ext': GoBuildExtCommand},
    ext_modules=[go_extension],
    package_data={
        'unaiverse.networking.p2p': [
            os.path.basename(get_ext_filename_with_path()),
            GO_SOURCE_NAME + HASH_FILE_SUFFIX,
        ],
    },
    zip_safe=False,
)
