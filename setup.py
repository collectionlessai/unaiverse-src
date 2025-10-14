# setup.py
import os
import platform
import subprocess
import hashlib
from setuptools import setup
from setuptools.command.build_py import build_py


def get_ext_filename():
    """Gets the platform-specific name for the shared library."""
    system = platform.system()
    if system == 'Linux':
        return 'unailib.so'
    elif system == 'Darwin':
        return 'unailib.dylib'
    elif system == 'Windows':
        return 'unailib.dll'
    raise RuntimeError(f"Unsupported operating system: {system}")

def get_file_hash(filepath):
    """Calculates the SHA256 hash of a file."""
    if not os.path.exists(filepath):
        return None
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# --- Configuration ---
GO_SOURCE_DIR = os.path.join('src', 'unaiverse', 'networking', 'p2p')
GO_SOURCE_NAME = 'lib.go'
COMPILED_LIB_NAME = get_ext_filename()
HASH_FILE_SUFFIX = '.sha256'

class GoBuildCommand(build_py):
    """Custom build command to compile Go source conditionally."""
    def run(self):
        go_source_path = os.path.join(GO_SOURCE_DIR, GO_SOURCE_NAME)
        output_lib_path = os.path.join(GO_SOURCE_DIR, COMPILED_LIB_NAME)
        hash_path = go_source_path + HASH_FILE_SUFFIX

        # --- Smart Compilation Check for Local Development ---
        # In a clean CI environment, this check will always fail (correctly),
        # forcing a build. For local dev, it avoids recompiling unchanged code.
        current_hash = get_file_hash(go_source_path)
        
        # Read the stored hash directly from the file's content
        if os.path.exists(hash_path):
            with open(hash_path, 'r') as f:
                stored_hash = f.read().strip()

            if current_hash == stored_hash and os.path.exists(output_lib_path):
                print("--- Go source unchanged and library exists, skipping compilation. ---")
                super().run()
                return

        print("--- Go source changed, hash or library not found, starting build. ---")

        # 2. Run Go commands
        try:
            # First, check if we need to initialize the Go module
            if not (os.path.exists(os.path.join(GO_SOURCE_DIR, 'go.mod)')) and 
                    os.path.exists(os.path.join(GO_SOURCE_DIR, 'go.sum'))):
                print("--- Initializing Go module ---")
                subprocess.run(
                    ['go', 'mod', 'init', 'unaiverse/networking/p2p/lib'],
                    check=True,
                    cwd=GO_SOURCE_DIR,
                    capture_output=True, text=True
                )
            
            # Then, ensure dependencies are tidy. Run this from the Go source directory.
            print("--- Running go mod tidy ---")
            subprocess.run(
                ['go', 'mod', 'tidy'],
                check=True,
                cwd=GO_SOURCE_DIR,
                capture_output=True, text=True
            )

            # Finally, build the shared library
            print("--- Running go build ---")
            build_command = [
                "go", "build", "-buildmode=c-shared", "-ldflags", "-s -w", "-o", COMPILED_LIB_NAME, GO_SOURCE_NAME
            ]
            print(f"Executing command: {' '.join(build_command)}")
            subprocess.run(
                build_command,
                check=True,
                cwd=GO_SOURCE_DIR,
                capture_output=True, text=True
            )
            print(f"Go build successful! Output at {output_lib_path}")

            # 3. Store the new hash after a successful build
            with open(hash_path, 'w') as f:
                f.write(current_hash)
            print(f"Stored new source hash at {hash_path}")

        except subprocess.CalledProcessError as e:
            print("--- Go build failed! ---")
            print(f"STDERR:\n{e.stderr}")
            print(f"STDOUT:\n{e.stdout}")
            raise RuntimeError("Go build failed.")
        except FileNotFoundError:
            print("--- Go command not found ---")
            print("Could not find 'go' executable. Please install Go and ensure it is in your PATH.")
            raise

        # Finally, run the original build_py command
        super().run()

# The setup() call now also includes the hash file in the package data.
setup(
    cmdclass={
        'build_py': GoBuildCommand,
    },
    package_data={
        'unaiverse.networking.p2p': [
            get_ext_filename(),
            GO_SOURCE_NAME + HASH_FILE_SUFFIX, # Include the hash file
        ],
    },
    zip_safe=False,
)
