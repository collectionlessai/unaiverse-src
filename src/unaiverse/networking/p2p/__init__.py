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
from . import messages
from . import p2p
from . import golibp2p
from . import lib_types
import os
import sys
import json
import ctypes
import platform
import requests
import subprocess
from typing import cast
from .messages import Msg
from .p2p import P2P, P2PError
from .golibp2p import GoLibP2P  # Your stub interface definition
from .lib_types import TypeInterface  # Assuming TypeInterface handles the void* results


# --- Setup and Pre-build Checks ---

# Determine the correct library file extension and URL based on the OS
os_to_ext = {
    "Windows": ".dll",
    "Darwin": ".dylib",
    "Linux": ".so"
}
os_to_url = {
    "Windows": "https://raw.githubusercontent.com/collectionlessai/unaiverse-misc/main/precompiled/lib.dll",
    "Darwin": "https://raw.githubusercontent.com/collectionlessai/unaiverse-misc/main/precompiled/lib.dylib",
    "Linux": "https://raw.githubusercontent.com/collectionlessai/unaiverse-misc/main/precompiled/lib.so"
}

current_os = platform.system()
lib_ext = os_to_ext.get(current_os, ".so")
lib_url = os_to_url.get(current_os, os_to_url["Linux"])

# --- Configuration & Paths ---
lib_dir = os.path.dirname(os.path.abspath(__file__))
go_source_file = os.path.join(lib_dir, "lib.go")
lib_filename = f"lib{lib_ext}"
lib_path = os.path.join(lib_dir, lib_filename)
go_mod_file = os.path.join(lib_dir, "go.mod")
version_file = os.path.join(lib_dir, "lib.version.json")

# --- Helper Functions ---

def load_shared_library(path):
    """Attempt to load the shared library and return the handle."""
    try:
        print(f"INFO: Attempting to load library from '{path}'...")
        return ctypes.CDLL(path)
    except OSError as e:
        print(f"INFO: Failed to load library: {e}")
        return None

def get_remote_version_info(url):
    """Performs a HEAD request to get ETag from the remote file."""
    try:
        print(f"INFO: Checking remote version at '{url}'...")
        response = requests.head(url, allow_redirects=True, timeout=10)
        response.raise_for_status()
        etag = response.headers.get("ETag")
        return etag
    except Exception as e:
        print(f"INFO: Failed to get remote version info: {e}")
        return None

def download_library(url, path, etag):
    """Downloads the shared library and saves version info."""
    print(f"INFO: Downloading new library from '{url}'...")
    try:
        headers = {"User-Agent": "python-requests/2.31.0"}
        response = requests.get(url, headers=headers, allow_redirects=True, timeout=30)
        response.raise_for_status()
        with open(path, "wb") as f:
            f.write(response.content)
            
        print("INFO: Download complete.")
        return True
    except Exception as e:
        print(f"INFO: Failed to download library: {e}")
        return False

def build_go_library():
    """Build the Go shared library and saves version info."""
    print("INFO: Building library from source...")
    if not os.path.exists(go_mod_file):
        print(f"INFO: 'go.mod' not found. Initializing Go module...")
        module_path = "unaiverse/networking/p2p/lib"
        try:
            subprocess.run(["go", "mod", "init", module_path], cwd=lib_dir, check=True, capture_output=True, text=True)
            print("INFO: Go module initialized. Running 'go mod tidy'...")
            subprocess.run(["go", "mod", "tidy"], cwd=lib_dir, check=True, capture_output=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"FATAL: Failed to initialize Go module. Is Go installed? Error: {e}", file=sys.stderr)
            raise e

    try:
        build_command = ["go", "build", "-buildmode=c-shared", "-ldflags", "-s -w", "-o", lib_filename, "lib.go"]
        subprocess.run(build_command, cwd=lib_dir, check=True, capture_output=True, text=True)
            
        print(f"INFO: Successfully built '{lib_filename}'.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"FATAL: Failed to build library. Is Go installed? Error: {e}", file=sys.stderr)
        return False

# --- Main Logic Flow ---
_shared_lib = None
should_update = False
go_source_time = os.path.getmtime(go_source_file)

# Step 1: Check for existing local binary and version info
if os.path.exists(lib_path):
    print(f"INFO: Found existing library '{lib_filename}'.")
    if os.path.exists(version_file):
        try:
            with open(version_file, "r") as f:
                version_info = json.load(f)
            
            if version_info.get("source") == "remote":
                remote_etag = get_remote_version_info(lib_url)
                if remote_etag and remote_etag != version_info.get("etag"):
                    print("INFO: Remote binary is newer (different ETag). An update is required.")
                    should_update = True
                else:
                    print("INFO: Local binary is up-to-date with remote.")
            elif version_info.get("source") == "local":
                if go_source_time > version_info.get("timestamp", 0):
                    print("INFO: Go source file is newer than local binary. Re-compilation is required.")
                    should_update = True
                else:
                    print("INFO: Local binary is up-to-date with source.")
            
        except (IOError, json.JSONDecodeError):
            print("INFO: Could not read version file. Assuming outdated.")
            should_update = True
    else:
        print("INFO: No version file found. Assuming outdated.")
        should_update = True
    
    if should_update:
        if os.path.exists(lib_path):
            os.remove(lib_path)
    else:
        # Load if it exists and is up-to-date
        _shared_lib = load_shared_library(lib_path)

# Step 2: Try to get a library if none is loaded
if _shared_lib is None:  # same as should_update being True
    # Attempt to download the latest binary
    remote_etag = get_remote_version_info(lib_url)
    if remote_etag and download_library(lib_url, lib_path, remote_etag):
        _shared_lib = load_shared_library(lib_path)
        if _shared_lib is not None:
            with open(version_file, "w") as f:
                json.dump({"source": "remote", "etag": remote_etag}, f)
    
    # Step 3: Fallback to local build if download failed or the file is invalid
    if _shared_lib is None:  # meaning that the download and load failed
        print("INFO: Download failed or produced an invalid library. Building from source...")
        if build_go_library():
            _shared_lib = load_shared_library(lib_path)
            if _shared_lib is not None:
                with open(version_file, "w") as f:
                    json.dump({"source": "local", "timestamp": go_source_time}, f)

# Final check
if _shared_lib is None:
    print("FATAL: Critical failure. Could not obtain or load the shared library.", file=sys.stderr)
    sys.exit(1)
else:
    print("SUCCESS: Library is ready to use.")

# --- Function Prototypes (argtypes and restype) ---
# Using void* for returned C strings, requiring TypeInterface for conversion/freeing.

# Define argtypes for the Go init function here
_shared_lib.InitializeLibrary.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_shared_lib.InitializeLibrary.restype = None

# Node Lifecycle & Info
_shared_lib.CreateNode.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int]
_shared_lib.CreateNode.restype = ctypes.c_void_p  # Treat returned *C.char as opaque pointer

_shared_lib.CloseNode.argtypes = [ctypes.c_int]
_shared_lib.CloseNode.restype = ctypes.c_void_p  # Treat returned *C.char as opaque pointer

_shared_lib.GetNodeAddresses.argtypes = [ctypes.c_int, ctypes.c_char_p]  # Input is still a Python string -> C string
_shared_lib.GetNodeAddresses.restype = ctypes.c_void_p  # Treat returned *C.char as opaque pointer

_shared_lib.GetConnectedPeers.argtypes = [ctypes.c_int]
_shared_lib.GetConnectedPeers.restype = ctypes.c_void_p  # Treat returned *C.char as opaque pointer

_shared_lib.GetRendezvousPeers.argtypes = [ctypes.c_int]
_shared_lib.GetRendezvousPeers.restype = ctypes.c_void_p  # Treat returned *C.char as opaque pointer

# Peer Connection
_shared_lib.ConnectTo.argtypes = [ctypes.c_int, ctypes.c_char_p]
_shared_lib.ConnectTo.restype = ctypes.c_void_p  # Treat returned *C.char as opaque pointer

_shared_lib.DisconnectFrom.argtypes = [ctypes.c_int, ctypes.c_char_p]
_shared_lib.DisconnectFrom.restype = ctypes.c_void_p  # Treat returned *C.char as opaque pointer

# Direct Messaging
_shared_lib.SendMessageToPeer.argtypes = [
    ctypes.c_int,  # Instance
    ctypes.c_char_p,  # Channel
    ctypes.c_char_p,  # Data buffer
    ctypes.c_int,  # Data length
]
_shared_lib.SendMessageToPeer.restype = ctypes.c_void_p  # Returns status code, not pointer

# Message Queue
_shared_lib.MessageQueueLength.argtypes = [ctypes.c_int]
_shared_lib.MessageQueueLength.restype = ctypes.c_int  # Returns length, not pointer

_shared_lib.PopMessages.argtypes = [ctypes.c_int]
_shared_lib.PopMessages.restype = ctypes.c_void_p  # Treat returned *C.char as opaque pointer

# PubSub
_shared_lib.SubscribeToTopic.argtypes = [ctypes.c_int, ctypes.c_char_p]
_shared_lib.SubscribeToTopic.restype = ctypes.c_void_p  # Treat returned *C.char as opaque pointer

_shared_lib.UnsubscribeFromTopic.argtypes = [ctypes.c_int, ctypes.c_char_p]
_shared_lib.UnsubscribeFromTopic.restype = ctypes.c_void_p  # Treat returned *C.char as opaque pointer

# Relay Client
_shared_lib.ReserveOnRelay.argtypes = [ctypes.c_int, ctypes.c_char_p]
_shared_lib.ReserveOnRelay.restype = ctypes.c_void_p  # Treat returned *C.char as opaque pointer

# Memory Management
# FreeString now accepts the opaque pointer directly
_shared_lib.FreeString.argtypes = [ctypes.c_void_p]
_shared_lib.FreeString.restype = None  # Void return

_shared_lib.FreeInt.argtypes = [ctypes.POINTER(ctypes.c_int)]  # Still expects a pointer to int
_shared_lib.FreeInt.restype = None  # Void return

# --- Python Interface Setup ---

# Import necessary components
# IMPORTANT: TypeInterface (or equivalent logic) MUST now handle converting
# the c_char_p results back to strings/JSON before freeing.
# Ensure TypeInterface methods like from_go_string_to_json are adapted for this.

# Import the stub type for type checking
try:
    from .golibp2p import GoLibP2P  # Your stub interface definition
except ImportError:
    print("Warning: GoLibP2P stub not found. Type checking will be limited.", file=sys.stderr)
    GoLibP2P = ctypes.CDLL

# Cast the loaded library object to the stub type
_shared_lib_typed = cast(GoLibP2P, _shared_lib)

# Attach the typed shared library object to the P2P class
P2P.libp2p = _shared_lib_typed
TypeInterface.libp2p = _shared_lib_typed  # Attach to TypeInterface if needed

# Attach the typed shared library object to the P2PError class

# Define the public API of this package
__all__ = [
    "P2P",
    "P2PError",
    "TypeInterface"  # Expose TypeInterface if users need its conversion helpers directly
]
