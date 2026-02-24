"""Type conversion utilities for Python-Go - UNaIVERSE Networking.


This module provides type conversion utilities between Python and Go (libp2p) using ctypes.
It handles string encoding, memory management, JSON serialization, and primitive type conversions
to ensure safe communication between Python code and the Go-libp2p shared library.

A Collectionless AI Project (https://collectionless.ai) / UNaIVERSE SRL (https://unaiverse.ai)

- Registration/Login: https://unaiverse.io
- Code Repositories: https://github.com/collectionlessai/
- Main Developers: Stefano Melacci (Project Leader), Christian Di Maio, Tommaso Guidi
"""

# Standard library imports
import json
import ctypes
from typing_extensions import Self
import logging
from threading import Lock
from typing import Any, Set
from typing import List, Optional, Generic, TypeVar, Union
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import base64

# 3rd party imports
from pydantic import BaseModel, Field, ConfigDict, model_validator

# 1st party imports
from .golibp2p import GoLibP2P


logger = logging.getLogger(__name__)


class TypeInterfaceError(Exception):
    """Base exception for type conversion errors."""

    pass


class PointerError(TypeInterfaceError):
    """Exception raised for pointer-related errors (null, double-free, etc.)."""

    pass


class ConversionError(TypeInterfaceError):
    """Exception raised when type conversion fails."""

    pass



# ==========================================
class TypeInterface(BaseModel):
    """Helper class for converting between Python and Go types using ctypes.

    This class provides bidirectional type conversion utilities for interacting with
    Go-libp2p shared library. It handles:

    - String encoding/decoding (UTF-8)
    - Primitive types (int, float, bool)
    - Binary data (bytes)
    - JSON serialization/deserialization
    - Memory management and double-free prevention

    Attributes:
        model_config: Pydantic model configuration to allow arbitrary types (e.g., GoLibP2P).
        libp2p (GoLibP2P): The GoLibP2P shared library instance
        _freed_pointers (Set[int]): A set to track freed pointers and prevent double-free errors
        _freed_pointers_lock (Lock): A lock to synchronize access to the _freed_pointers set
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    libp2p: GoLibP2P = Field(
        ...,
        description="The GoLibP2P shared library instance for memory management operations",
    )

    _freed_pointers: Set[int] = Field(
        default_factory=set,
        description="A set to track freed pointers and prevent double-free errors. Stores integer addresses of freed pointers.",
    )

    _freed_pointers_lock: Lock = Field(
        default_factory=Lock,
        description="Lock to synchronize access to the _freed_pointers set for thread safety",
    )

    @model_validator(mode="after")
    def init_lock_and_set(self):
        """Initialize the _freed_pointers set and _freed_pointers_lock after model validation."""
        if not hasattr(self, "_freed_pointers") or self._freed_pointers is None:
            object.__setattr__(self, "_freed_pointers", set())
        if (
            not hasattr(self, "_freed_pointers_lock")
            or self._freed_pointers_lock is None
        ):
            object.__setattr__(self, "_freed_pointers_lock", Lock())
        return self

    def to_go_string(self, s: str | None) -> bytes:
        """Convert a Python string to UTF-8 encoded bytes for Go.

        This method encodes a Python string into UTF-8 bytes suitable for passing
        to Go functions expecting a C char* pointer via ctypes.c_char_p.

        Args:
            s: The Python string to convert. If None, converts empty string.

        Returns:
            UTF-8 encoded bytes object.

        """
        if s is None:
            s = ""
        return s.encode("utf-8")

    def from_go_string(self, cstr: bytes | None) -> str:
        """Convert a C char pointer (Go string) to a Python string.

        Args:
            cstr: The C char pointer as bytes. If None, returns empty string.

        Returns:
            Decoded Python string.

        Raises:
            ConversionError: If UTF-8 decoding fails.

        """
        if not cstr:
            return ""

        try:
            return cstr.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ConversionError(f"Failed to decode Go string: {e}") from e

    def to_go_int(self, i: int) -> ctypes.c_int:
        """Convert a Python integer to a ctypes.c_int for Go.

        Args:
            i: The Python integer.

        Returns:
            A ctypes.c_int value.

        """
        return ctypes.c_int(i)

    def from_go_int(self, val: ctypes.c_int) -> int:
        """Convert a ctypes.c_int from Go to a Python integer.

        Args:
            val: The ctypes.c_int value from Go.

        Returns:
            Python integer.

        """
        return int(val)

    def to_go_float(self, f: float) -> ctypes.c_float:
        """Convert a Python float to a ctypes.c_float for Go.

        Args:
            f: The Python float.

        Returns:
            A ctypes.c_float value.

        """
        return ctypes.c_float(f)

    def from_go_float(self, val: ctypes.c_float) -> float:
        """Convert a ctypes.c_float from Go to a Python float.

        Args:
            val: The ctypes.c_float value from Go.

        Returns:
            Python float.

        """
        return float(val)

    def to_go_bool(self, b: bool) -> ctypes.c_int:
        """Convert a Python boolean to a Go-compatible integer.

        Go represents booleans as integers: 1 for True, 0 for False.

        Args:
            b: The Python boolean.

        Returns:
            ctypes.c_int with value 1 (True) or 0 (False).
        """
        return ctypes.c_int(1 if b else 0)

    def from_go_bool(self, val: ctypes.c_int) -> bool:
        """Convert a Go boolean (integer) to a Python boolean.

        Args:
            val: The ctypes.c_int value (1 or 0) from Go.

        Returns:
            True if val equals 1, False otherwise.

        """
        return val == 1

    def to_go_bytes(self, b: bytes | None) -> ctypes.c_char_p:
        """Convert Python bytes to a C char pointer for Go.

        Args:
            b: The Python bytes object. If None, converts empty bytes.

        Returns:
            A ctypes.c_char_p pointer to the byte data.

        """
        if b is None:
            b = b""
        buf = ctypes.create_string_buffer(b, len(b))
        return ctypes.cast(buf, ctypes.c_char_p)

    def from_go_bytes(self, cptr: ctypes.c_char_p, length: int) -> bytes:
        """Convert a Go byte array pointer to Python bytes.

        Args:
            cptr: The C pointer to the byte array.
            length: The number of bytes to read.

        Returns:
            Python bytes object containing the data.
        """
        if not cptr or length <= 0:
            return bytes()
        return ctypes.string_at(cptr, length)

    def to_go_json(self, data: Any) -> bytes:
        """Encode a Python object to JSON bytes for Go.

        Serializes a Python object (dict, list, etc.) to JSON and returns
        UTF-8 encoded bytes suitable for passing to Go via ctypes.c_char_p.

        Args:
            data: The Python object to encode as JSON.

        Returns:
            UTF-8 encoded bytes containing the JSON string.

        Raises:
            ConversionError: If JSON serialization fails.

        Example:
            ```python
            json_bytes = type_interface.to_go_json({"key": "value"})
            ```
        """
        try:
            json_str = json.dumps(data)
            return self.to_go_string(json_str)
        except (TypeError, ValueError) as e:
            raise ConversionError(f"Failed to encode object to JSON: {e}") from e
    def from_go_ptr_to_json(
        self, c_void_ptr_val: ctypes.c_int
    ) -> dict[str, Any] | list[Any] | Any | None:
        """Convert a C void* pointer to JSON data, with automatic memory cleanup.
    
        This method handles the complete lifecycle of a Go-allocated string pointer:

        1. Validates the pointer is not NULL
        2. Checks for double-free attempts
        3. Reads the null-terminated C string
        4. Decodes UTF-8 and parses JSON
        5. Frees the C memory via GoLibP2P.FreeString

        This is the primary method for receiving complex data structures from Go.

        Args:
            c_void_ptr_val: Integer address of the C void* pointer containing JSON.

        Returns:
            Parsed Python object (dict, list, etc.) from the JSON string.

        Raises:
            PointerError: If pointer is NULL or already freed.
            ConversionError: If string reading, decoding, or JSON parsing fails.

        Warning:
            This method automatically frees the C memory. Do not call FreeString
            manually on this pointer after calling this method.
        """
        if not c_void_ptr_val:
            raise PointerError("Received NULL pointer from Go function")

        try:
            # --- Double-Free Check (Before Reading/Casting) ---
            with self._freed_pointers_lock:  # Acquire lock if using threading
                if c_void_ptr_val in self._freed_pointers:
                    # This indicates a serious logic error elsewhere - the pointer
                    # was already freed but somehow passed here again.
                    logger.warning(
                        f"🔥🔥🔥 ATTEMPT TO PROCESS ALREADY FREED POINTER {hex(int(c_void_ptr_val))}! 🔥🔥🔥"
                    )

                    # Raising an error is safer than trying to read potentially invalid memory.
                    logger.error(
                        f"Attempt to process pointer {hex(int(c_void_ptr_val))} which was already freed"
                    )
                    raise Exception(
                        f"Attempt to process pointer {hex(int(c_void_ptr_val))} which was already freed"
                    )

            # --- Cast void* to c_char_p and Read String ---
            try:
                # Perform the cast only when needed for reading
                c_char_ptr_for_read = ctypes.cast(c_void_ptr_val, ctypes.c_char_p)
                raw_bytes = ctypes.string_at(c_char_ptr_for_read)
                json_string = raw_bytes.decode("utf-8")

                # Logger.debug(f"Read string (len={len(json_string)})
                # from pointer {hex(c_void_ptr_val)}: %.100s...", json_string)
            except (ctypes.ArgumentError, ValueError, UnicodeDecodeError) as read_err:
                logger.error(
                    f"Failed to read/decode string from pointer {hex(int(c_void_ptr_val))}: "
                    f"{read_err}",
                    exc_info=False,
                )

                # Even if reading fails, the pointer itself *might* still be valid C memory
                # that Go expects us to free. We will proceed to free it in finally.
                raise Exception(
                    f"Failed to read string from pointer {hex(int(c_void_ptr_val))}: {read_err}"
                ) from read_err
            except (
                Exception
            ) as unexpected_read_err:  # Catch other potential ctypes issues
                logger.error(
                    f"Unexpected error reading C string from pointer {hex(int(c_void_ptr_val))}: "
                    f"{unexpected_read_err}",
                    exc_info=True,
                )
                raise Exception(
                    f"Unexpected error reading C string from pointer {hex(int(c_void_ptr_val))}: "
                    f"{unexpected_read_err}"
                ) from unexpected_read_err

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
                logger.error(
                    f"Failed to decode JSON from pointer {hex(int(c_void_ptr_val))}: {json_err}",
                    exc_info=False,
                )

                # Again, the pointer is likely valid C memory, but the content is bad.
                # Let the block handle freeing.
                raise Exception(
                    f"Failed to decode JSON from pointer {hex(int(c_void_ptr_val))}: {json_err}"
                ) from json_err

        finally:
            # --- CRITICAL: Free C Memory ---
            # This block executes even if errors occurred during read/parse,
            # ensuring we attempt to free any non-NULL pointer received from Go.
            with self._freed_pointers_lock:
                if c_void_ptr_val:
                    logger.info(
                        f"🐍 FINALLY: Freeing pointer {hex(int(c_void_ptr_val))}..."
                    )
                    if c_void_ptr_val in self._freed_pointers:
                        # This check is technically redundant if the initial check worked,
                        # but provides an extra safety layer in case of concurrency issues
                        # (if freed_pointers is shared without locks - which it shouldn't be).
                        logger.warning(
                            f"🔥🔥🔥 DOUBLE FREE DETECTED in finally block for "
                            f"{hex(int(c_void_ptr_val))}! Skipping FreeString call again. 🔥🔥🔥"
                        )
                    else:
                        # Add before calling free
                        try:
                            self.libp2p.FreeString(
                                c_void_ptr_val
                            )  # Pass the original void* value
                            logger.info(
                                f"✅ FINALLY: FreeString successful for {hex(int(c_void_ptr_val))}."
                            )
                        except Exception as free_err:
                            # Log if FreeString fails, but don't raise from finally
                            # as it might hide the original error.
                            logger.critical(
                                f"🚨 FAILED TO FREE C MEMORY for pointer "
                                f"{hex(int(c_void_ptr_val))} via FreeString: {free_err}",
                                exc_info=True,
                            )

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

    def from_go_string_to_list(self, cstr: ctypes.c_char_p) -> list[Any]:
        """Decode a JSON-encoded list from a Go C char pointer.

        Args:
            cstr: The C char pointer containing a JSON array string.

        Returns:
            Python list parsed from the JSON data.

        Raises:
            ConversionError: If decoding or JSON parsing fails.
        """
        try:
            json_str = self.from_go_string(cstr)
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ConversionError(f"Failed to parse JSON list: {e}") from e
