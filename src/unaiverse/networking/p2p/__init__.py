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
import ctypes
import platform
import warnings
from typing import cast
from .messages import Msg
from .p2p import P2P, P2PError
from .golibp2p import GoLibP2P  # Your stub interface definition
from .lib_types import TypeInterface  # Assuming TypeInterface handles the void* results
from ._unailib_source import lib_filename, source_hash


def _developer_source_check(lib: ctypes.CDLL) -> None:
    """In a source checkout, warns when the loaded library was not built from the Go sources next to it.

    Installed wheels carry no Go sources, so there is nothing to compare and the check is skipped.
    """
    current_source_hash = source_hash(os.path.join(_lib_dir, 'unailib'))
    if current_source_hash is None:
        return

    # A library built before the hash was embedded has no such symbol, and counts as out of date
    built_source_hash = None
    if hasattr(lib, 'UnailibBuildHash'):
        lib.UnailibBuildHash.argtypes = []
        lib.UnailibBuildHash.restype = ctypes.c_void_p
        ptr = lib.UnailibBuildHash()
        if ptr:
            built_source_hash = ctypes.string_at(ptr).decode()
            lib.FreeString(ptr)

    if built_source_hash != current_source_hash:
        warnings.warn(
            "\033[93m" + "\n" + "="*80 +
            "\nWARNING: The Go source files in 'unailib/' differ from the ones the loaded shared library\n"
            "was built from. Your running code may not reflect recent changes.\n\n"
            "To fix this, reinstall the package from the repository root (e.g. `pip install -e .`)\n" +
            "="*80 + "\033[0m",
            UserWarning
        )


# --- Main Library Loading ---
# The shared library is guaranteed by the build process to be in this directory, under a fixed name.
_lib_dir = os.path.dirname(os.path.abspath(__file__))
_lib_path = _lib_dir

try:
    _lib_path = os.path.join(_lib_dir, lib_filename(platform.system()))
    _shared_lib = ctypes.CDLL(_lib_path)
except (RuntimeError, OSError) as e:
    print(
        f"FATAL: Could not load the required p2p shared library from {_lib_path}.\n"
        "This indicates a corrupted or missing installation. "
        "Please try reinstalling the 'unaiverse' package.\n"
        f"Underlying error: {e}",
        file=sys.stderr
    )
    raise ImportError("Failed to load the UNaIVERSE p2p shared library.") from e

# --- Function Prototypes (argtypes and restype) ---
# Using void* for returned C strings, requiring TypeInterface for conversion/freeing.

# Define argtypes for the Go init function here
_shared_lib.InitializeLibrary.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
_shared_lib.InitializeLibrary.restype = None

# Node Lifecycle & Info
_shared_lib.CreateNode.argtypes = [ctypes.c_int, ctypes.c_char_p]
_shared_lib.CreateNode.restype = ctypes.c_void_p  # Treat returned *C.char as opaque pointer

_shared_lib.CloseNode.argtypes = [ctypes.c_int]
_shared_lib.CloseNode.restype = ctypes.c_void_p  # Treat returned *C.char as opaque pointer

_shared_lib.GetNodeAddresses.argtypes = [ctypes.c_int, ctypes.c_char_p]  # Input is still a Python string -> C string
_shared_lib.GetNodeAddresses.restype = ctypes.c_void_p  # Treat returned *C.char as opaque pointer

_shared_lib.GetConnectedPeers.argtypes = [ctypes.c_int]
_shared_lib.GetConnectedPeers.restype = ctypes.c_void_p  # Treat returned *C.char as opaque pointer

_shared_lib.GetTopicPeers.argtypes = [ctypes.c_int]
_shared_lib.GetTopicPeers.restype = ctypes.c_void_p  # Treat returned *C.char as opaque pointer

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

# WebRTC Signaling
_shared_lib.GetWebRTCConnections.argtypes = [ctypes.c_int]
_shared_lib.GetWebRTCConnections.restype = ctypes.c_void_p  # Treat returned *C.char as opaque pointer

# Reachability (AutoNAT)
_shared_lib.GetReachability.argtypes = [ctypes.c_int]
_shared_lib.GetReachability.restype = ctypes.c_void_p  # Treat returned *C.char as opaque pointer

# Session abstraction (per-stage handshake protocols + signed peer record).
# See unailib/session_api.go. (Liveness is session-level PING/PONG, not /ipfs/ping.)
_shared_lib.SendOnProtocol.argtypes = [
    ctypes.c_int,      # instanceIndex
    ctypes.c_char_p,   # peerID
    ctypes.c_char_p,   # protocolID
    ctypes.c_char_p,   # payload
    ctypes.c_int,      # payloadLen
]
_shared_lib.SendOnProtocol.restype = ctypes.c_void_p

_shared_lib.SignWithHostKey.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
_shared_lib.SignWithHostKey.restype = ctypes.c_void_p

_shared_lib.VerifyPeerSignature.argtypes = [
    ctypes.c_char_p,   # peerID
    ctypes.c_char_p,   # data
    ctypes.c_int,      # dataLen
    ctypes.c_char_p,   # sigB64
]
_shared_lib.VerifyPeerSignature.restype = ctypes.c_void_p

# Memory Management
# FreeString now accepts the opaque pointer directly
_shared_lib.FreeString.argtypes = [ctypes.c_void_p]
_shared_lib.FreeString.restype = None  # Void return

_shared_lib.FreeInt.argtypes = [ctypes.POINTER(ctypes.c_int)]  # Still expects a pointer to int
_shared_lib.FreeInt.restype = None  # Void return

# Run the check once FreeString is declared, since it frees the hash returned by the library
_developer_source_check(_shared_lib)

# TURN related
_shared_lib.UpdateICEConfig.argtypes = [ctypes.c_int, ctypes.c_char_p]
_shared_lib.UpdateICEConfig.restype = ctypes.c_void_p

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
