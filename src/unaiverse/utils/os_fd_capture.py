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
import os
import sys
import threading
from typing import Callable

__all__ = ["FdCapture"]


class FdCapture:
    """OS-level file-descriptor redirector for capturing C/Go library output.

    The Go libp2p library writes its log lines directly to the OS stdout/stderr
    file descriptors, bypassing Python's ``sys.stdout``/``sys.stderr`` entirely.
    A simple ``sys.stdout`` reassignment therefore has no effect.

    ``FdCapture`` solves this by operating at the OS level:

    1. An ``os.pipe()`` is created — a pair of (read_fd, write_fd).
    2. The original target file descriptor (e.g. fd 1 for stdout) is saved
       via ``dup()``.
    3. The write end of the pipe is ``dup2``'d over the target fd so that
       anything any library writes to fd 1 now flows into the pipe instead.
    4. A background daemon thread reads from the pipe line-by-line and calls
       the user-supplied ``callback(line: str)`` for each line.
    5. On ``stop()``, the original fd is restored and the pipe is closed.

    Both ``stdout`` (fd 1) and ``stderr`` (fd 2) can be captured simultaneously
    by creating two ``FdCapture`` instances.

    Thread safety: ``start()`` and ``stop()`` are idempotent and protected by
    an internal lock.

    Args:
        fd:        The OS file descriptor to capture (1 = stdout, 2 = stderr).
        callback:  Called for every complete line read from the captured fd.
                   Receives the line as a ``str`` with the trailing newline stripped.
        encoding:  Byte encoding used to decode captured bytes (default ``utf-8``).
        errors:    Error handler for decode failures (default ``replace``).

    Example::

        from unaiverse.utils.os_fd_capture import FdCapture
        from unaiverse.utils.logger import Logger, Ch

        log = Logger("app", active={Ch.P2P, Ch.USER})

        cap_out = FdCapture(1, lambda line: log.p2p(line, src="stdout"))
        cap_err = FdCapture(2, lambda line: log.p2p(line, src="stderr"))
        cap_out.start()
        cap_err.start()

        # ... run Go library ...

        cap_out.stop()
        cap_err.stop()
    """

    def __init__(
            self,
            fd: int,
            callback: Callable[[str], None],
            encoding: str = "utf-8",
            errors: str = "replace",
    ) -> None:
        if fd not in (1, 2):
            raise ValueError(f"FdCapture only supports fd 1 (stdout) or fd 2 (stderr), got {fd}")
        self._target_fd = fd
        self._callback = callback
        self._encoding = encoding
        self._errors = errors
        self._lock = threading.Lock()
        self._active = False
        self._py_stream = None

        # Set when start() is called; cleared on stop()
        self._saved_fd: int | None = None  # dup of original fd
        self._read_fd: int | None = None  # read end of the pipe
        self._write_fd: int | None = None  # write end (dup2'd over target)
        self._thread: threading.Thread | None = None

    @property
    def real_fd(self) -> int | None:
        """The duped copy of the original fd, usable for writing screen output
        without feeding back into the capture pipe. ``None`` if not active."""
        return self._saved_fd

    def start(self) -> None:
        """Begin capturing. No-op if already active."""
        with self._lock:
            if self._active:
                return

            # 1. Flush the Python-level stream before touching the fd
            if self._target_fd == 1:
                sys.stdout.flush()
            else:
                sys.stderr.flush()

            # 2. Save a copy of the real FD, so we can restore it (and write to it)
            self._saved_fd = os.dup(self._target_fd)

            # 3. Create pipe
            self._read_fd, self._write_fd = os.pipe()

            # 4. Redirect target fd → write end of pipe.
            #    Python's sys.stdout/stderr still point to their original file
            #    objects, which internally hold the original fd number.  After
            #    dup2, those file objects now also write into the pipe.  We
            #    therefore reassign sys.stdout/stderr to write through saved_fd
            #    so that Logger screen output bypasses the pipe.
            os.dup2(self._write_fd, self._target_fd)

            if self._target_fd == 1:
                self._py_stream = sys.stdout
                sys.stdout = open(self._saved_fd, "w",
                                  encoding=self._encoding, errors=self._errors,
                                  closefd=False)  # closefd=False: we own saved_fd
            else:
                self._py_stream = sys.stderr
                sys.stderr = open(self._saved_fd, "w",
                                  encoding=self._encoding, errors=self._errors,
                                  closefd=False)

            # 5. Start reader thread
            self._thread = threading.Thread(
                target=self._reader_loop,
                name=f"FdCapture-fd{self._target_fd}",
                daemon=True,
            )
            self._active = True
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop capturing and restore the original fd.  No-op if not active.

        Args:
            timeout: Seconds to wait for the reader thread to drain and exit.
        """
        with self._lock:
            if not self._active:
                return

            # 1. Restore Python-level stream
            if self._target_fd == 1:
                try:
                    sys.stdout.flush()
                except Exception:
                    pass
                sys.stdout = self._py_stream
            else:
                try:
                    sys.stderr.flush()
                except Exception:
                    pass
                sys.stderr = self._py_stream

            # 2. Restore the original OS-level fd
            os.dup2(self._saved_fd, self._target_fd)
            os.close(self._saved_fd)
            self._saved_fd = None

            # 3. Close write end of pipe → reader thread sees EOF
            os.close(self._write_fd)
            self._write_fd = None

            self._active = False

        # 4. Wait for reader thread to drain (outside lock)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

        # 5. Close read end
        if self._read_fd is not None:
            try:
                os.close(self._read_fd)
            except OSError:
                pass
            self._read_fd = None

    def __enter__(self) -> "FdCapture":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    def _reader_loop(self) -> None:
        """Read lines from the pipe and dispatch to the callback."""
        buf = b""

        # Keep a local reference — self._read_fd may be cleared by stop()
        # but we captured it before the thread started.
        read_fd = self._read_fd
        try:
            while True:
                try:
                    chunk = os.read(read_fd, 4096)
                except OSError:
                    break  # fd closed — we're done
                if not chunk:
                    break  # EOF

                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    text = line.decode(self._encoding, errors=self._errors).rstrip("\r")
                    if text:  # skip blank lines
                        try:
                            self._callback(text)
                        except Exception:
                            pass  # never let a bad callback kill the thread

            # Flush any partial line left in the buffer after EOF
            if buf:
                text = buf.decode(self._encoding, errors=self._errors).rstrip("\r\n")
                if text:
                    try:
                        self._callback(text)
                    except Exception:
                        pass
        finally:
            pass  # read_fd is closed by stop(); nothing to clean up here
