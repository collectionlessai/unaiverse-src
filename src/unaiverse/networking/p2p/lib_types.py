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
import json
import ctypes
import logging
from typing import List, Any

from .golibp2p import GoLibP2P  # Assuming this class loads the library

logger = logging.getLogger('LIB-TYPES')
logger.setLevel(logging.WARNING)


class TypeInterface:
    """
    Helper class for converting between Python types and Go types using ctypes.
    """
    def __init__(self, libp2p_instance: GoLibP2P):
        self.libp2p: GoLibP2P = libp2p_instance  # Store the shared library object instance

    def to_go_string(self, s: str) -> bytes:
        """
        Converts a Python string to a UTF-8 encoded Python 'bytes' object.

        This 'bytes' object is suitable for direct use with ctypes when passing
        to a C function expecting a 'char*' (ctypes.c_char_p), as ctypes
        will automatically pass a pointer to the byte string's data.

        Args:
            s: The Python string.

        Returns:
            A Python 'bytes' object containing the UTF-8 encoded string.
        """
        if s is None:
            s = ""
        return s.encode("utf-8")

    def from_go_string(self, cstr: bytes) -> str:
        """
        Converts a C char pointer (Go string) to a Python string.

        Args:
            cstr: The C char pointer.

        Returns:
            The decoded Python string.
        """
        if not cstr:
            return ""
        return cstr.decode("utf-8")

    def to_go_int(self, i: int) -> ctypes.c_int:
        """
        Converts a Python integer to a Go-compatible ctypes integer.

        Args:
            i: The Python integer.

        Returns:
            A ctypes.c_int equivalent.
        """
        return ctypes.c_int(i)

    def from_go_int(self, val: ctypes.c_int) -> int:
        """
        Converts a ctypes.c_int from Go to a Python integer.

        Args:
            val: The ctypes.c_int value.

        Returns:
            The corresponding Python integer.
        """
        return int(val)

    def to_go_float(self, f: float) -> ctypes.c_float:
        """
        Converts a Python float to a Go-compatible ctypes float.

        Args:
            f: The Python float.

        Returns:
            A ctypes.c_float equivalent.
        """
        return ctypes.c_float(f)

    def from_go_float(self, val: ctypes.c_float) -> float:
        """
        Converts a ctypes.c_float from Go to a Python float.

        Args:
            val: The ctypes.c_float value.

        Returns:
            The corresponding Python float.
        """
        return float(val)

    def to_go_bool(self, b: bool) -> ctypes.c_int:
        """
        Converts a Python boolean to a Go-compatible integer (1 if True, 0 if False).

        Args:
            b: The Python boolean.

        Returns:
            A ctypes.c_int (1 or 0).
        """
        return ctypes.c_int(1 if b else 0)

    def from_go_bool(self, val: ctypes.c_int) -> bool:
        """
        Converts a Go-compatible integer (ctypes.c_int) to a Python boolean.

        Args:
            val: The ctypes.c_int value.

        Returns:
            True if the value equals 1, False otherwise.
        """
        return val == 1

    def to_go_bytes(self, b: bytes) -> ctypes.c_char_p:
        """
        Converts a Python bytes object to a Go-compatible C char pointer.

        Args:
            b: The Python bytes.

        Returns:
            A ctypes.c_char_p pointing to the byte data.
        """
        if b is None:
            b = b""
        buf = ctypes.create_string_buffer(b, len(b))
        return ctypes.cast(buf, ctypes.c_char_p)

    def from_go_bytes(self, cptr: ctypes.c_char_p, length: int) -> bytes:
        """
        Converts a Go pointer representing a byte array to a Python bytes object.

        Args:
            cptr: The C pointer to the byte array.
            length: The number of bytes to read.

        Returns:
            A Python bytes object containing the read data.
        """
        if not cptr or length <= 0:
            return bytes()
        return ctypes.string_at(cptr, length)

    def from_go_ptr_to_json(self, c_void_ptr_val: int) -> Any:
        """
        Converts a C void* pointer (returned by Go as int) pointing to a
        null-terminated C string containing JSON into a Python object.

        It reads the string, parses it as JSON, and crucially frees the C memory
        using the provided FreeString function from the Go library.

        Args:
            c_void_ptr_val: The integer value representing the C pointer address.

        Returns:
            The parsed Python object from the JSON string.

        Raises:
            GoLibError: If the pointer is NULL, reading/decoding fails, or
                        JSON parsing fails.
            TypeError: When go_lib is not a valid ctypes library object.
        """

        if not c_void_ptr_val:  # NULL (0): nothing to read or free.
            raise Exception("Received a NULL pointer from Go function")

        try:
            # --- Cast void* to c_char_p and read the string ---
            try:
                c_char_ptr_for_read = ctypes.cast(c_void_ptr_val, ctypes.c_char_p)
                raw_bytes = ctypes.string_at(c_char_ptr_for_read)
                json_string = raw_bytes.decode('utf-8')
            except (ctypes.ArgumentError, ValueError, UnicodeDecodeError) as read_err:
                logger.error(f"Failed to read/decode string from pointer {hex(c_void_ptr_val)}: "
                             f"{read_err}", exc_info=False)
                raise Exception(f"Failed to read string from pointer "
                                f"{hex(c_void_ptr_val)}: {read_err}") from read_err

            # --- Parse JSON ---
            try:
                return json.loads(json_string)
            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to decode JSON from pointer {hex(c_void_ptr_val)}: {json_err}", exc_info=False)
                raise Exception(f"Failed to decode JSON from pointer "
                                f"{hex(c_void_ptr_val)}: {json_err}") from json_err

        finally:

            # --- Free C memory allocated by Go (C.CString) ---
            # Each pointer crosses this boundary exactly once, so no double-free
            # tracking is needed; FreeString -> C.free is thread-safe for distinct
            # pointers. We never raise from here, to preserve the original error.
            try:
                self.libp2p.FreeString(c_void_ptr_val)
            except Exception as free_err:
                logger.critical(f"🚨 FAILED TO FREE C MEMORY for pointer "
                                f"{hex(c_void_ptr_val)} via FreeString: {free_err}", exc_info=True)

    def to_go_json(self, data: Any) -> bytes:
        """
    Encodes a Python object to a JSON string, returning a UTF-8 encoded
    Python 'bytes' object.

    This 'bytes' object is suitable for direct use with ctypes when passing
    to a C function expecting a 'char*' (ctypes.c_char_p).

    Args:
        data: The Python object (e.g., dict, list) to encode.

    Returns:
        A Python 'bytes' object containing the JSON string, UTF-8 encoded.
    """
        json_str = json.dumps(data)
        return self.to_go_string(json_str)

    def from_go_string_to_list(self, cstr: ctypes.c_char_p) -> List[Any]:
        """
        Decodes a JSON-encoded list from a Go C char pointer into a Python list.

        Args:
            cstr: The Go string (C char pointer) containing a JSON list.

        Returns:
            A Python list representing the JSON data.
        """
        s = self.from_go_string(cstr)

        return json.loads(s)
