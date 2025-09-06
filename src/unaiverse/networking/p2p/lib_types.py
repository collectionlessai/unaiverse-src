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
from threading import Lock
from typing import Optional, List, Any, TYPE_CHECKING

from .golibp2p import GoLibP2P  # Assuming this class loads the library

logger = logging.getLogger('LIB-TYPES')


class TypeInterface:
    """
    Helper class for converting between Python types and Go types using ctypes.
    """
    def __init__(self, libp2p_instance: GoLibP2P):
        self.__freed_pointers: set[int] = set()  # Track freed pointers to prevent double-free errors
        self.__freed_pointers_lock: Any = Lock()  # A threading lock
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

        json_string: Optional[str] = None  # To store the string for error reporting

        if not c_void_ptr_val:  # Check if the address is NULL (0)
            raise print("Received a NULL pointer from Go function")

        try:

            # --- Double-Free Check (Before Reading/Casting) ---
            self.__freed_pointers_lock.acquire()  # Acquire lock if using threading
            if c_void_ptr_val in self.__freed_pointers:

                # This indicates a serious logic error elsewhere - the pointer
                # was already freed but somehow passed here again.
                logger.warning(f"🔥🔥🔥 ATTEMPT TO PROCESS ALREADY FREED POINTER {hex(c_void_ptr_val)}! 🔥🔥🔥")

                # Raising an error is safer than trying to read potentially invalid memory.
                logger.error(f"Attempt to process pointer {hex(c_void_ptr_val)} which was already freed",
                             pointer_val=c_void_ptr_val)
                raise Exception(f"Attempt to process pointer {hex(c_void_ptr_val)} which was already freed",
                                pointer_val=c_void_ptr_val)
            self.__freed_pointers_lock.release()  # Release lock if using threading

            # --- Cast void* to c_char_p and Read String ---
            try:

                # Perform the cast only when needed for reading
                c_char_ptr_for_read = ctypes.cast(c_void_ptr_val, ctypes.c_char_p)
                raw_bytes = ctypes.string_at(c_char_ptr_for_read)
                json_string = raw_bytes.decode('utf-8')

                # Logger.debug(f"Read string (len={len(json_string)}) from pointer {hex(c_void_ptr_val)}: %.100s...", json_string)
            except (ctypes.ArgumentError, ValueError, UnicodeDecodeError) as read_err:
                logger.error(f"Failed to read/decode string from pointer {hex(c_void_ptr_val)}: {read_err}", exc_info=False)

                # Even if reading fails, the pointer itself *might* still be valid C memory
                # that Go expects us to free. We will proceed to free it in finally.
                raise Exception(f"Failed to read string from pointer {hex(c_void_ptr_val)}: {read_err}",
                                pointer_val=c_void_ptr_val) from read_err
            except Exception as unexpected_read_err:  # Catch other potential ctypes issues
                logger.error(f"Unexpected error reading C string from pointer {hex(c_void_ptr_val)}: {unexpected_read_err}", exc_info=True)
                raise Exception(f"Unexpected error reading C string from pointer {hex(c_void_ptr_val)}: {unexpected_read_err}",
                                pointer_val=c_void_ptr_val) from unexpected_read_err

            # --- Check for Empty String ---

            # --- Parse JSON ---
            try:

                # Now that we have the string, parse it
                logger.debug(f"Parsing JSON from string: {json_string}")
                parsed_data = json.loads(json_string)
                logger.debug(f"Parsed JSON data: {parsed_data}")

                # Logger.debug(f"Successfully parsed JSON from pointer {hex(c_void_ptr_val)}")
                return parsed_data  # Return the parsed Python object

            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to decode JSON from pointer {hex(c_void_ptr_val)}: {json_err}", exc_info=False)

                # Again, the pointer is likely valid C memory, but the content is bad.
                # Let the block handle freeing.
                raise Exception(f"Failed to decode JSON from pointer {hex(c_void_ptr_val)}: {json_err}",
                                pointer_val=c_void_ptr_val) from json_err

        finally:

            # --- CRITICAL: Free C Memory ---
            # This block executes even if errors occurred during read/parse,
            # ensuring we attempt to free any non-NULL pointer received from Go.
            with self.__freed_pointers_lock:
                if c_void_ptr_val:
                    logger.info(f"🐍 FINALLY: Freeing pointer {hex(c_void_ptr_val)}...")
                    if c_void_ptr_val in self.__freed_pointers:

                        # This check is technically redundant if the initial check worked,
                        # but provides an extra safety layer in case of concurrency issues
                        # (if freed_pointers is shared without locks - which it shouldn't be).
                        logger.warning(f"🔥🔥🔥 DOUBLE FREE DETECTED in finally block for {hex(c_void_ptr_val)}! Skipping FreeString call again. 🔥🔥🔥")
                    else:

                        # Add before calling free
                        try:
                            self.libp2p.FreeString(c_void_ptr_val)  # Pass the original void* value
                            logger.info(f"✅ FINALLY: FreeString successful for {hex(c_void_ptr_val)}.")
                        except Exception as free_err:

                            # Log if FreeString fails, but don't raise from finally
                            # as it might hide the original error.
                            logger.critical(f"🚨 FAILED TO FREE C MEMORY for pointer {hex(c_void_ptr_val)} via FreeString: {free_err}", exc_info=True)

                            # Consider removing from freed_pointers if free failed?
                            # freed_pointers.discard(c_void_ptr_val) # Maybe, to allow retry? Risky.
                            # But if FreeString fails, the pointer is likely invalid anyway.

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
