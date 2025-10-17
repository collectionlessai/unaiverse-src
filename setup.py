# setup.py
import os
import hashlib
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


# Define the path to your Go source file
GO_SOURCE_PATH = os.path.join('src', 'unaiverse', 'networking', 'p2p', 'lib.go')
HASH_FILE_SUFFIX = '.sha256'

def get_file_hash(filepath):
    """Computes the SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

class GoBuildExtCommand(build_ext):
    """
    Custom build_ext command to build the Go shared library.
    This overrides build_extension to integrate with setuptools' build process.
    """
    def build_extension(self, ext):
        """Override the default C/C++ extension build process for our Go extension."""
        
        # Get the path for the final compiled library that setuptools expects
        dest_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Use the source file defined in the Extension object
        go_path = ext.sources[0]
        go_dir = os.path.dirname(go_path)
        go_source_file = os.path.basename(go_path)
        
        # Check hash to see if a rebuild is needed (this is good practice)
        hash_path = go_path + HASH_FILE_SUFFIX
        current_hash = get_file_hash(go_path)
        stored_hash = None
        if os.path.exists(hash_path):
            with open(hash_path, 'r') as f:
                stored_hash = f.read().strip()

        # Rebuild if source has changed or the final library is missing
        if current_hash != stored_hash or not os.path.exists(dest_path):
            print(f"--- Building Go source '{go_path}' to '{dest_path}' ---")
            subprocess.run(
                ['go', 'build', '-buildmode=c-shared', '-ldflags', '-s -w',
                 '-o', dest_path, go_source_file],
                check=True, cwd=go_dir
            )
            # Store the new hash
            with open(hash_path, 'w') as f:
                f.write(current_hash)
        else:
            print(f"--- Go source '{go_path}' unchanged; skipping build. ---")
        
        print(f"--- Build complete for {dest_path} ---")


# The Extension object now correctly lists the Go file as a source.
go_extension = Extension(
    "unaiverse.networking.p2p.unailib",
    sources=[GO_SOURCE_PATH],
)

setup(
    cmdclass={'build_ext': GoBuildExtCommand},
    ext_modules=[go_extension],
    zip_safe=False,
)
