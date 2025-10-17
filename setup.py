# setup.py
import os
import shutil
import hashlib
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


GO_SOURCE_NAME = 'lib.go'
HASH_FILE_SUFFIX = '.sha256'

def get_file_hash(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

class GoBuildExtCommand(build_ext):
    """Custom build_ext that builds the Go library directly to the final location."""
    def run(self):
        # 1. Get the one true destination path from setuptools.
        # This will be the correctly named file inside the build/lib... directory.
        dest_path = self.get_ext_fullpath(self.extensions[0].name)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        go_dir = os.path.join('src', 'unaiverse', 'networking', 'p2p')
        go_path = os.path.join(go_dir, GO_SOURCE_NAME)
        hash_path = go_path + HASH_FILE_SUFFIX

        current_hash = get_file_hash(go_path)
        stored_hash = None
        if os.path.exists(hash_path):
            with open(hash_path, 'r') as f:
                stored_hash = f.read().strip()

        # Check if the FINAL destination file needs to be built.
        if current_hash != stored_hash or not os.path.exists(dest_path):
            print(f"--- Go source changed, building directly to {dest_path} ---")
            
            # Build DIRECTLY to the final destination path.
            subprocess.run(
                ['go', 'build', '-buildmode=c-shared', '-ldflags', '-s -w',
                 '-o', dest_path, GO_SOURCE_NAME],
                check=True, cwd=go_dir
            )
            with open(hash_path, 'w') as f:
                f.write(current_hash)
        else:
            print("--- Go source unchanged; skipping build. ---")
        
        print(f"--- Build complete. Artifact is at {dest_path} ---")

    def get_outputs(self):
        return [self.get_ext_fullpath(self.extensions[0].name)]


# A "marker" extension to trigger the build_ext command
go_extension = Extension(
    "unaiverse.networking.p2p.unailib",
    sources=[],
)

setup(
    cmdclass={'build_ext': GoBuildExtCommand},
    ext_modules=[go_extension],
    zip_safe=False,
)
