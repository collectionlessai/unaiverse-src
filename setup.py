# setup.py
import os
import glob
import shutil
import hashlib
import platform
import subprocess
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

def get_file_hash(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

class GoBuildExtCommand(build_ext):
    """Custom build_ext that builds the Go library."""
    def _cleanup_old_artifacts(self, target_dir):
        print("--- Cleaning old build artifacts ---")
        patterns = [
            os.path.join(target_dir, "unailib*.so"),
            os.path.join(target_dir, "unailib*.dll"),
            os.path.join(target_dir, "unailib*.pyd"),
            os.path.join(target_dir, "unailib*.h"),
            os.path.join(target_dir, "go-build*"),
        ]
        for pattern in patterns:
            for path in glob.glob(pattern):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                        print(f"Removed dir {path}")
                    else:
                        os.remove(path)
                        print(f"Removed file {path}")
                except Exception as e:
                    print(f"Could not remove {path}: {e}")

    def run(self):
        go_dir = os.path.join('src', 'unaiverse', 'networking', 'p2p')
        go_path = os.path.join(go_dir, GO_SOURCE_NAME)
        out_path = get_ext_filename_with_path()
        hash_path = go_path + HASH_FILE_SUFFIX
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        self._cleanup_old_artifacts(go_dir)

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
        
        # We get the final destination path for the extension and copy our pre-built library there.
        dest_path = self.get_ext_fullpath(self.extensions[0].name)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        print(f"--- Copying {out_path} to {dest_path} ---")
        shutil.copyfile(out_path, dest_path)
        
        # Also copy the generated hash file to the final package directory
        dest_dir = os.path.dirname(dest_path)
        if os.path.abspath(os.path.dirname(hash_path)) != os.path.abspath(dest_dir):
            print(f"--- Copying {hash_path} to {dest_dir} ---")
            shutil.copy(hash_path, dest_dir)
        else:
            print(f"--- Hash file is already in the source directory (editable install); skipping copy. ---")
        
        subprocess.run(['go', 'clean', '-cache', '-modcache', '-testcache', '-fuzzcache'],
                       cwd=go_dir, check=False)


# Fake extension only to mark wheel as platform-dependent
go_extension = Extension(
    "unaiverse.networking.p2p.unailib",
    sources=[],  # dummy
)

setup(
    cmdclass={'build_ext': GoBuildExtCommand},
    ext_modules=[go_extension],
    package_data={},
    zip_safe=False,
)
