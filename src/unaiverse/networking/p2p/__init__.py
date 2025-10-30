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
import glob
import ctypes
import hashlib
import warnings
import itertools
from typing import cast
from .messages import Msg
from .p2p import P2P, P2PError
from .golibp2p import GoLibP2P  # Your stub interface definition
from .lib_types import TypeInterface  # Assuming TypeInterface handles the void* results


def _get_file_hash(filepath):
    """Calculates the SHA256 hash of a file, returning None if it doesn't exist."""
    if not os.path.exists(filepath):
        return None
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def _developer_source_check():
    """
    If source files are present (i.e., in a dev environment), check if the
    compiled library is in sync with the source code.
    """
    go_source_file = os.path.join(_lib_dir, 'lib.go')
    hash_file = go_source_file + '.sha256'
    
    # This check only runs if all dev files are present. For users who
    # installed from a wheel, these files won't exist, and this is skipped.
    if os.path.exists(go_source_file) and os.path.exists(hash_file):
        current_source_hash = _get_file_hash(go_source_file)
        
        stored_build_hash = None
        with open(hash_file, 'r') as f:
            stored_build_hash = f.read().strip()
            
        if current_source_hash != stored_build_hash:
            # Use warnings.warn for a standard, non-intrusive developer warning.
            warnings.warn(
                "\033[93m" + "\n" + "="*80 +
                "\nWARNING: The Go source file (lib.go) has been modified since the shared\n"
                "library was last compiled. Your running code may not reflect recent changes.\n\n"
                "To fix this, run: pip install -e .\n" +
                "="*80 + "\033[0m",
                UserWarning
            )

# --- Main Library Loading ---
# The shared library is guaranteed by the build process to be in this directory.
_lib_dir = os.path.dirname(os.path.abspath(__file__))
_shared_lib = None

try:
    _lib_path = _lib_dir
    patterns = ["*.so", "*.pyd", "*.dll", "*.dylib"]
    _results = list(itertools.chain.from_iterable(
        glob.glob(os.path.join(_lib_dir, f"unailib{ext}")) for ext in patterns
    ))
    _lib_path = os.path.join(_lib_dir, _results[0])
    _shared_lib = ctypes.CDLL(_lib_path)
except (IndexError, OSError) as e:
    print(
        f"FATAL: Could not load the required p2p shared library from {_lib_path}.\n"
        "This indicates a corrupted or missing installation. "
        "Please try reinstalling the 'unaiverse' package.\n"
        f"Underlying error: {e}",
        file=sys.stderr
    )
    raise ImportError("Failed to load the UNaIVERSE p2p shared library.") from e

# Run the check after successfully loading the library.
if _shared_lib is not None:
    _developer_source_check()
    print(f"UNaIVERSE: Successfully loaded p2p library from {_lib_path}")

# --- Function Prototypes (argtypes and restype) ---
# Using void* for returned C strings, requiring TypeInterface for conversion/freeing.

# Define argtypes for the Go init function here
_shared_lib.InitializeLibrary.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_shared_lib.InitializeLibrary.restype = None

# Node Lifecycle & Info
_shared_lib.CreateNode.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
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
_shared_lib.SendMessageToPeer.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
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
