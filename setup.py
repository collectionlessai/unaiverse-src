# setup.py
import os
import platform
import subprocess
import hashlib
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# --- Configuration ---
GO_SOURCE_NAME = 'lib.go'
HASH_FILE_SUFFIX = '.sha256'

def get_ext_filename_with_path():
    """Gets the platform-specific path for the shared library."""
    system = platform.system()
    lib_name = ""
    if system == 'Linux':
        lib_name = 'unailib.so'
    elif system == 'Darwin':
        lib_name = 'unailib.dylib'
    elif system == 'Windows':
        lib_name = 'unailib.dll'
    else:
        raise RuntimeError(f"Unsupported operating system: {system}")
    
    return os.path.join('src', 'unaiverse', 'networking', 'p2p', lib_name)

def get_go_source_dir():
    """Gets the directory of the Go source files."""
    return os.path.join('src', 'unaiverse', 'networking', 'p2p')

def get_file_hash(filepath):
    """Calculates the SHA256 hash of a file."""
    if not os.path.exists(filepath):
        return None
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

class GoBuildExtCommand(build_ext):
    """Custom command to build the Go extension."""
    def run(self):
        go_source_dir = get_go_source_dir()
        go_source_path = os.path.join(go_source_dir, GO_SOURCE_NAME)
        output_lib_path = get_ext_filename_with_path()
        os.makedirs(os.path.dirname(output_lib_path), exist_ok=True)
        
        hash_path = go_source_path + HASH_FILE_SUFFIX

        current_hash = get_file_hash(go_source_path)
        stored_hash = None
        if os.path.exists(hash_path):
            with open(hash_path, 'r') as f:
                stored_hash = f.read().strip()

        if current_hash == stored_hash and os.path.exists(output_lib_path):
            print("--- Go source unchanged and library exists, skipping compilation. ---")
            super().run()
            return

        print("--- Go source changed, hash or library not found, starting build. ---")
        try:
            print("--- Running go mod tidy ---")
            subprocess.run(['go', 'mod', 'tidy'], check=True, cwd=go_source_dir)

            print("--- Running go build ---")
            build_command = [
                "go", "build",
                "-buildmode=c-shared",
                "-ldflags", "-s -w",
                "-o", os.path.basename(output_lib_path),
                GO_SOURCE_NAME
            ]
            print(f"Executing command: {' '.join(build_command)}")
            subprocess.run(
                build_command,
                check=True,
                cwd=go_source_dir,
                capture_output=True, text=True
            )
            print(f"Go build successful! Output at {output_lib_path}")

            with open(hash_path, 'w') as f:
                f.write(current_hash)

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"--- Go build failed! --- \n{e}")
            raise RuntimeError("Go build failed.")

        super().run()

go_extension = Extension(
    "unaiverse.networking.p2p.unailib",
    sources=[]
)

setup(
    cmdclass={
        'build_ext': GoBuildExtCommand,
    },
    package_data={
        'unaiverse.networking.p2p': [
            os.path.basename(get_ext_filename_with_path()),
            GO_SOURCE_NAME + HASH_FILE_SUFFIX,
        ],
    },
    ext_modules=[go_extension],
    zip_safe=False,
)
