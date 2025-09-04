# __init__.py
import ctypes
import os
import platform
import sys
import subprocess
from typing import cast


# --- Setup and Pre-build Checks ---

# Define paths and library names
lib_dir = os.path.dirname(__file__)
go_mod_file = os.path.join(lib_dir, "go.mod")
go_source_file = os.path.join(lib_dir, "lib.go")
lib_name = "lib"

# Determine the correct library file extension based on the OS
if platform.system() == "Windows":
    lib_ext = ".dll"
elif platform.system() == "Darwin": # macOS
    lib_ext = ".dylib"
else: # Linux and other Unix-like
    lib_ext = ".so"

lib_filename = f"{lib_name}{lib_ext}"
lib_path = os.path.join(lib_dir, lib_filename)

# --- Automatically initialize Go module if needed ---
if not os.path.exists(go_mod_file):
    print(f"INFO: 'go.mod' not found. Initializing Go module in '{lib_dir}'...")
    try:
        # Define a module path. This can be anything, but a path-like name is conventional.
        module_path = "unaiverse/networking/p2p/lib"
        # Run 'go mod init'
        subprocess.run(
            ["go", "mod", "init", module_path],
            cwd=lib_dir,  # Run the command in the directory containing lib.go
            check=True,   # Raise an exception if the command fails
            capture_output=True, # Capture stdout/stderr
            text=True
        )
        # Run 'go mod tidy' to find dependencies and create go.sum
        print("INFO: Go module initialized. Running 'go mod tidy'...")
        subprocess.run(
            ["go", "mod", "tidy"],
            cwd=lib_dir,
            check=True,
            capture_output=True,
            text=True
        )
        print("INFO: 'go.mod' and 'go.sum' created successfully.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("FATAL: Failed to initialize Go module.", file=sys.stderr)
        print("Please ensure Go is installed and in your system's PATH.", file=sys.stderr)
        # If 'go mod' failed, print its output for debugging
        if isinstance(e, subprocess.CalledProcessError):
            print(f"Go command stderr:\n{e.stderr}", file=sys.stderr)
        raise e

# --- Automatically build the shared library if it's missing or outdated ---
rebuild_needed = False
reason = ""

if not os.path.exists(lib_path):
    rebuild_needed = True
    reason = f"the shared library '{lib_filename}' was not found."
elif os.path.getmtime(go_source_file) > os.path.getmtime(lib_path):
    rebuild_needed = True
    reason = f"the last modification to '{go_source_file}' is more recent than the '{lib_filename}' last build."

if rebuild_needed:
    print(f"INFO: Rebuilding shared library because {reason}")
    try:
        build_command = ["go", "build", "-buildmode=c-shared", "-o", lib_filename, "lib.go"]
        print(f"Running command: {' '.join(build_command)}")
        result = subprocess.run(
            build_command,
            cwd=lib_dir,
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(f"Go build stdout:\n{result.stdout}")
        print(f"INFO: Successfully built '{lib_filename}'.")

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"FATAL: Failed to build Go shared library.", file=sys.stderr)
        print("Please ensure Go is installed and in your system's PATH.", file=sys.stderr)
        if isinstance(e, subprocess.CalledProcessError):
            print(f"Go compiler stderr:\n{e.stderr}", file=sys.stderr)
        raise e

# --- Library Loading ---
try:
    _shared_lib = ctypes.CDLL(lib_path)
    # print(f"Successfully loaded Go library: {lib_path}")
except OSError as e:
    print(f"Error loading shared library at {lib_path}: {e}", file=sys.stderr)
    raise

# --- Function Prototypes (argtypes and restype) ---
# Using void* for returned C strings, requiring TypeInterface for conversion/freeing.

# Define argtypes for the Go init function here
_shared_lib.InitializeLibrary.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_shared_lib.InitializeLibrary.restype = None

# Node Lifecycle & Info
_shared_lib.CreateNode.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_shared_lib.CreateNode.restype = ctypes.c_void_p # Treat returned *C.char as opaque pointer

_shared_lib.CloseNode.argtypes = [ctypes.c_int]
_shared_lib.CloseNode.restype = ctypes.c_void_p # Treat returned *C.char as opaque pointer

_shared_lib.GetNodeAddresses.argtypes = [ctypes.c_int, ctypes.c_char_p] # Input is still a Python string -> C string
_shared_lib.GetNodeAddresses.restype = ctypes.c_void_p # Treat returned *C.char as opaque pointer

_shared_lib.GetConnectedPeers.argtypes = [ctypes.c_int]
_shared_lib.GetConnectedPeers.restype = ctypes.c_void_p # Treat returned *C.char as opaque pointer

_shared_lib.GetRendezvousPeers.argtypes = [ctypes.c_int]
_shared_lib.GetRendezvousPeers.restype = ctypes.c_void_p # Treat returned *C.char as opaque pointer

# Peer Connection
_shared_lib.ConnectTo.argtypes = [ctypes.c_int, ctypes.c_char_p]
_shared_lib.ConnectTo.restype = ctypes.c_void_p # Treat returned *C.char as opaque pointer

_shared_lib.DisconnectFrom.argtypes = [ctypes.c_int, ctypes.c_char_p]
_shared_lib.DisconnectFrom.restype = ctypes.c_void_p # Treat returned *C.char as opaque pointer

# Direct Messaging
_shared_lib.SendMessageToPeer.argtypes = [
    ctypes.c_int,     # instance
    ctypes.c_char_p,  # channel
    ctypes.c_char_p,  # data buffer
    ctypes.c_int,     # data length
]
_shared_lib.SendMessageToPeer.restype = ctypes.c_void_p # Returns status code, not pointer

# Message Queue
_shared_lib.MessageQueueLength.argtypes = [ctypes.c_int]
_shared_lib.MessageQueueLength.restype = ctypes.c_int # Returns length, not pointer

_shared_lib.PopMessages.argtypes = [ctypes.c_int]
_shared_lib.PopMessages.restype = ctypes.c_void_p # Treat returned *C.char as opaque pointer

# PubSub
_shared_lib.SubscribeToTopic.argtypes = [ctypes.c_int, ctypes.c_char_p]
_shared_lib.SubscribeToTopic.restype = ctypes.c_void_p # Treat returned *C.char as opaque pointer

_shared_lib.UnsubscribeFromTopic.argtypes = [ctypes.c_int, ctypes.c_char_p]
_shared_lib.UnsubscribeFromTopic.restype = ctypes.c_void_p # Treat returned *C.char as opaque pointer

# Relay Client
_shared_lib.ReserveOnRelay.argtypes = [ctypes.c_int, ctypes.c_char_p]
_shared_lib.ReserveOnRelay.restype = ctypes.c_void_p # Treat returned *C.char as opaque pointer

# Memory Management
# FreeString now accepts the opaque pointer directly
_shared_lib.FreeString.argtypes = [ctypes.c_void_p]
_shared_lib.FreeString.restype = None # void return

_shared_lib.FreeInt.argtypes = [ctypes.POINTER(ctypes.c_int)] # Still expects a pointer to int
_shared_lib.FreeInt.restype = None # void return

# --- Python Interface Setup ---

# Import necessary components
from .p2p import P2P, P2PError
from .messages import Msg
# IMPORTANT: TypeInterface (or equivalent logic) MUST now handle converting
# the c_char_p results back to strings/JSON before freeing.
# Ensure TypeInterface methods like from_go_string_to_json are adapted for this.
from .lib_types import TypeInterface # Assuming TypeInterface handles the void* results

# Import the stub type for type checking
try:
    from .golibp2p import GoLibP2P # Your stub interface definition
except ImportError:
    print("Warning: GoLibP2P stub not found. Type checking will be limited.", file=sys.stderr)
    GoLibP2P = ctypes.CDLL

# Cast the loaded library object to the stub type
_shared_lib_typed = cast(GoLibP2P, _shared_lib)


# Attach the typed shared library object to the P2P class
P2P.libp2p = _shared_lib_typed
TypeInterface.libp2p = _shared_lib_typed # Attach to TypeInterface if needed

# Attach the typed shared library object to the P2PError class

# Define the public API of this package
__all__ = [
    "P2P",
    "P2PError",
    "MessageType",
    "TypeInterface", # Expose TypeInterface if users need its conversion helpers directly
]
