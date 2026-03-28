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
import ast
import sys
import time
import json
import math
import gzip
import random
import base64
import threading
import importlib
import importlib.abc
import uuid as _uuid
from tqdm import tqdm
from typing import Any
from pathlib import Path
import importlib.machinery
from datetime import datetime
from collections import deque
from unaiverse.custom import GenException


def save_node_addresses_to_file(node: object, dir_path: str, public: bool,
                                filename: str = "addresses.txt", append: bool = False) -> None:
    """Writes the hosted node's name and its public or world addresses to a file.

    Args:
        node: The node object whose addresses are to be saved; expected to expose
            ``hosted.get_name()``, ``get_public_addresses()``, and ``get_world_addresses()``.
        dir_path: Directory in which the output file is created.
        public: If True, saves the public addresses; otherwise saves the world addresses.
        filename: Name of the output file (default: ``"addresses.txt"``).
        append: If True, appends to an existing file instead of overwriting it (default: False).
    """
    address_file = os.path.join(dir_path, filename)
    with open(address_file, "w" if not append else "a") as file:
        file.write(node.hosted.get_name() + ";" +
                   str(node.get_public_addresses() if public else node.get_world_addresses()) + "\n")
        file.flush()


def get_node_addresses_from_file(dir_path: str, filename: str = "addresses.txt") -> dict[str, list[str]]:
    """Reads node names and their address lists from a file.

    Supports both the legacy single-address-per-line format (starting with ``"/"``) and the
    current ``name;[addr, ...]`` semicolon-separated format.

    Args:
        dir_path: Directory containing the file.
        filename: Name of the file to read (default: ``"addresses.txt"``).

    Returns:
        A dictionary mapping node names to lists of address strings.  Legacy files are
        returned under the key ``"unk"``.
    """
    ret = {}
    with open(os.path.join(dir_path, filename)) as file:
        lines = file.readlines()

        # Old file format
        if lines[0].strip() == "/":
            addresses = []
            for line in lines:
                _line = line.strip()
                if len(_line) > 0:
                    addresses.append(_line)
            ret["unk"] = addresses
            return ret

        # New file format
        for line in lines:
            if line.strip().startswith("***"):  # Header marker
                continue
            comma_separated_values = [v.strip() for v in line.split(';')]
            node_name, addresses_str = comma_separated_values
            ret[node_name] = ast.literal_eval(addresses_str)  # Name appearing multiple times? the last entry is kept

    return ret


class Silent:
    """Context manager that suppresses stdout by redirecting it to /dev/null."""

    def __init__(self, ignore: bool = False) -> None:
        """Initialises the Silent context manager.

        Args:
            ignore: If True, stdout is not suppressed (context manager is a no-op; default: False).
        """
        self.ignore = ignore

    def __enter__(self) -> None:
        """Redirects stdout to /dev/null on entry (unless ignore is True)."""
        if not self.ignore:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None,
                 exc_tb: object | None) -> None:
        """Restores the original stdout on exit (unless ignore is True)."""
        if not self.ignore:
            sys.stdout.close()
            sys.stdout = self._original_stdout


# The countdown function
def countdown_start(seconds: int, msg: str) -> threading.Thread:
    """Starts a tqdm progress-bar countdown in a background thread.

    Args:
        seconds: Number of seconds to count down.
        msg: Description label displayed next to the progress bar.

    Returns:
        The background ``threading.Thread`` running the countdown.
    """
    class TqdmPrintRedirector:
        def __init__(self, tqdm_instance: object) -> None:
            self.tqdm_instance = tqdm_instance
            self.original_stdout = sys.__stdout__

        def write(self, s: str) -> None:
            if s.strip():  # Ignore empty lines (needed for the way tqdm works)
                self.tqdm_instance.write(s, file=self.original_stdout)

        def flush(self) -> None:
            pass  # Tqdm handles flushing

    def drawing(secs: int, message: str) -> None:
        with tqdm(total=secs, desc=message, file=sys.__stdout__) as t:
            sys.stdout = TqdmPrintRedirector(t)  # Redirect prints to tqdm.write
            for i in range(secs):
                time.sleep(1)
                t.update(1.)
            sys.stdout = sys.__stdout__  # Restore original stdout

    sys.stdout.flush()
    handle = threading.Thread(target=drawing, args=(seconds, msg))
    handle.start()
    return handle


def countdown_wait(handle: threading.Thread) -> None:
    """Blocks until the countdown thread created by ``countdown_start`` finishes.

    Args:
        handle: The thread handle returned by ``countdown_start``.
    """
    handle.join()


def check_json_start(file: str, msg: str, delete_existing: bool = False) -> threading.Thread:
    """Starts a background daemon thread that monitors a JSON file for changes and pretty-prints it.

    Args:
        file: Path to the JSON file to monitor.
        msg: A header message printed once when monitoring begins.
        delete_existing: If True, removes the file before starting the monitor (default: False).

    Returns:
        The daemon ``threading.Thread`` running the file-watcher loop.
    """
    from rich.json import JSON
    from rich.console import Console
    cons = Console(file=sys.__stdout__)

    if delete_existing:
        if os.path.exists(file):
            os.remove(file)

    def checking(file_path: str, console: Console):
        print(msg)
        prev_dict = {}
        while True:
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding='utf-8') as f:
                        json_dict = json.load(f)
                        if json_dict != prev_dict:
                            now = datetime.now()
                            console.print("─" * 80)
                            console.print("Printing updated file "
                                          "(print time: " + now.strftime("%Y-%m-%d %H:%M:%S") + ")")
                            console.print("─" * 80)
                            console.print(JSON.from_data(json_dict))
                        prev_dict = json_dict
                except KeyboardInterrupt:
                    break
                except Exception:
                    pass
            time.sleep(1)

    handle = threading.Thread(target=checking, args=(file, cons), daemon=True)
    handle.start()
    return handle


def check_json_start_wait(handle: threading.Thread) -> None:
    """Blocks until the JSON-watcher thread created by ``check_json_start`` finishes.

    Args:
        handle: The thread handle returned by ``check_json_start``.
    """
    handle.join()


def show_images_grid(image_paths: list[str], max_cols: int = 3) -> None:
    """Displays a grid of images from the given file paths using matplotlib.

    Args:
        image_paths: A list of file-path strings pointing to the images to display.
        max_cols: Maximum number of columns in the grid (default: 3).
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    n = len(image_paths)
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)

    # Load images
    images = [mpimg.imread(p) for p in image_paths]

    # Determine figure size based on image sizes
    widths, heights = zip(*[(img.shape[1], img.shape[0]) for img in images])

    # Use average width/height for scaling
    avg_width = sum(widths) / len(widths)
    avg_height = sum(heights) / len(heights)

    fig_width = cols * avg_width / 100
    fig_height = rows * avg_height / 100

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes = axes.flatten() if n > 1 else [axes]

    fig.canvas.manager.set_window_title("Image Grid")

    # Hide unused axes
    for ax in axes[n:]:
        ax.axis('off')

    for idx, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(str(idx), fontsize=12, fontweight='bold')

    plt.subplots_adjust(wspace=0, hspace=0)

    # Turn on interactive mode
    plt.ion()
    plt.show()

    fig.canvas.draw()
    plt.pause(0.1)


class FileTracker:
    """Monitors a folder for file creation, modification, and deletion events."""

    def __init__(self, folder: str, ext: str = ".json",
                 prefix: str | None = None, skip: str | None = None) -> None:
        """Initialises the FileTracker and takes an initial snapshot of the folder.

        Args:
            folder: Path to the folder to monitor.
            ext: File extension to watch (default: ``".json"``).
            prefix: If set, only files whose names start with this prefix are tracked (default: None).
            skip: If set, files with this exact name are ignored (default: None).
        """
        self.folder = Path(folder)
        self.ext = ext.lower()
        self.skip = skip
        self.prefix = prefix
        self.last_state = self.__scan_files()

    def __scan_files(self) -> dict[str, float]:
        """Scans the monitored folder and returns a snapshot of matching files and their mtimes.

        Returns:
            A dictionary mapping file names to their last-modification timestamps.
        """
        state = {}
        for file in self.folder.iterdir():
            try:
                if ((file.is_file() and file.suffix.lower() == self.ext and
                        (self.skip is None or file.name != self.skip)) and
                        (self.prefix is None or file.name.startswith(self.prefix))):
                    # state[file.name] = os.path.getmtime(file) # this is less stable than what you see below
                    state[file.name] = file.stat().st_mtime_ns
            except Exception:
                pass
        return state

    def something_changed(self) -> bool:
        """Checks whether any tracked file has been created, modified, or deleted since the last call.

        Updates the internal snapshot on every call.

        Returns:
            True if at least one file changed, False otherwise.
        """
        new_state = self.__scan_files()

        created = [f for f in new_state if f not in self.last_state]
        modified = [f for f in new_state if f in self.last_state and new_state[f] != self.last_state[f]]
        deleted = [f for f in self.last_state if f not in new_state]  # Track deletions

        has_changed = bool(created or modified or deleted)
        self.last_state = new_state
        return has_changed


def pack_py_files(py_files: dict[str, str]) -> str:
    """Serialize, gzip-compress and base64-encode a py-files dict.

    Args:
        py_files: Dict mapping relative paths to Python source text.

    Returns:
        A plain ASCII string suitable for embedding in any text-based message.
    """
    raw = json.dumps(py_files).encode('utf-8')
    return base64.b64encode(gzip.compress(raw)).decode('ascii')


def collect_py_files(folder: str, exclude_dirs: set[str] | None = None) -> dict[str, str]:
    """Collect all .py sources under *folder*, skipping named subdirectories.

    Args:
        folder: Root folder to walk.
        exclude_dirs: Sub-directory names to skip entirely (default: None).

    Returns:
        A dict mapping each file's path relative to *folder* (forward slashes,
        e.g. ``'role1.py'``, ``'utils/helper.py'``) to its source text.
    """
    if exclude_dirs is None:
        exclude_dirs = set()
    root = Path(folder)
    result: dict[str, str] = {}
    for py_file in root.rglob('*.py'):
        rel = py_file.relative_to(root)
        if any(part in exclude_dirs for part in rel.parts[:-1]):
            continue
        file_name_with_relative_path = str(rel).replace('\\', '/')
        result[file_name_with_relative_path] = py_file.read_text(encoding='utf-8')
    return result


def unpack_py_files(packed: str) -> dict[str, str]:
    """Base64-decode, decompress and deserialize a packed py-files string.

    Args:
        packed: A string produced by ``pack_py_files``.

    Returns:
        The original dict mapping relative paths to Python source text.
    """
    return json.loads(gzip.decompress(base64.b64decode(packed)).decode('utf-8'))


class _InMemoryLoader(importlib.abc.SourceLoader):
    """Serves Python source from a plain string."""

    def __init__(self, source: str, virtual_path: str) -> None:
        self._source = source
        self._path = virtual_path

    def get_data(self, path: str) -> bytes:           # called by SourceLoader machinery
        return self._source.encode('utf-8')

    def get_filename(self, fullname: str) -> str:
        return self._path


class InMemoryFinder(importlib.abc.MetaPathFinder):
    """MetaPathFinder that resolves imports from an in-memory source dict.

    Each instance owns a unique namespace (e.g. ``_dyn_abc12345``) so that
    concurrent worlds loaded by different agents never collide in sys.modules.

    Args:
        py_files: Dict mapping relative paths (forward-slash, e.g.
            ``'role1.py'``, ``'utils/helper.py'``) to Python source text.
        namespace: Unique string used as the top-level package name.
    """

    def __init__(self, py_files: dict[str, str], namespace: str) -> None:
        self._ns = namespace

        # fullname -> (source, virtual_path, is_package)
        self._mods: dict[str, tuple[str, str, bool]] = {}

        # Discover all directories that contain at least one .py file,
        # so we can auto-generate implicit namespace packages for them.
        dirs: set[str] = set()
        for rel in py_files:
            parts = rel.split('/')
            for depth in range(1, len(parts)):
                dirs.add('/'.join(parts[:depth]))

        for rel, source in py_files.items():
            parts = rel.split('/')
            filename = parts[-1]
            sub_dirs = parts[:-1]

            if filename == '__init__.py':
                mod_name = namespace + ('.' + '.'.join(sub_dirs) if sub_dirs else '')
                is_pkg = True
            else:
                stem = filename[:-3]  # strip .py
                mod_name = namespace + '.' + '.'.join(sub_dirs + [stem])
                is_pkg = False

            v_path = f'<world:{namespace}>/{rel}'
            self._mods[mod_name] = (source, v_path, is_pkg)

        # Auto-create empty namespace packages for directories with no __init__.py
        for d in dirs:
            pkg_name = namespace + '.' + d.replace('/', '.')
            if pkg_name not in self._mods:
                self._mods[pkg_name] = ('', f'<world:{namespace}>/{d}/__init__.py', True)

    def find_spec(self, fullname: str, path: object, target: object = None) -> importlib.machinery.ModuleSpec | None:

        # Root namespace package
        if fullname == self._ns:
            spec = importlib.machinery.ModuleSpec(fullname, loader=None, origin=f'<world:{self._ns}>', is_package=True)
            spec.submodule_search_locations = []
            return spec

        if fullname not in self._mods:
            return None

        source, v_path, is_pkg = self._mods[fullname]
        loader = _InMemoryLoader(source, v_path)
        spec = importlib.machinery.ModuleSpec(fullname, loader, origin=v_path, is_package=is_pkg)
        if is_pkg:
            spec.submodule_search_locations = []
        return spec

    def cleanup(self) -> None:
        """Remove all owned modules from "sys.modules" and unregister this finder.

        Call this when the agent that was built from these sources is destroyed.
        """
        for name in list(self._mods):
            sys.modules.pop(name, None)
        sys.modules.pop(self._ns, None)
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass


def load_agent_in_memory(py_files: dict[str, str], role: str, **init_kwargs: Any) -> tuple[Any, InMemoryFinder]:
    """Instantiate a ``WAgent`` from an in-memory Python source tree.

    The file ``{role}.py`` (at the root of *py_files*) is imported as the
    agent's main module.  Any ``import`` statements inside it are resolved
    against the other entries in *py_files* through a temporary
    ``InMemoryFinder`` registered on ``sys.meta_path``.

    Args:
        py_files: Dict mapping relative paths to source text (as produced by
            ``collect_py_files``).
        role: Role name; ``{role}.py`` is the entry-point module.
        **init_kwargs: Keyword arguments forwarded to ``WAgent.__init__``
            (e.g. ``proc=None``).

    Returns:
        A ``(agent_instance, finder)`` tuple.  The caller **must** keep
        *finder* alive as long as the agent is in use, and call
        ``finder.cleanup()`` when the agent is destroyed.

    Raises:
        KeyError: If ``{role}.py`` is not present in *py_files*.
        Exception: Any exception raised during module execution or
            ``WAgent.__init__``.
    """
    if f'{role}.py' not in py_files:
        raise GenException(f"No source file '{role}.py' found in the received module set "
                           f"(available: {list(py_files)})")

    namespace = f'_dyn_{_uuid.uuid4().hex[:8]}'
    finder = InMemoryFinder(py_files, namespace)
    sys.meta_path.insert(0, finder)  # insert first so it wins over real filesystem

    try:
        mod = importlib.import_module(f'{namespace}.{role}')
        agent = mod.WAgent(**init_kwargs)
    except Exception:
        finder.cleanup()
        raise

    return agent, finder


def prepare_app_dir(app_name: str = "unaiverse") -> str:
    """Resolves and creates the platform-appropriate application data directory.

    Args:
        app_name: Name of the application subdirectory (default: ``"unaiverse"``).

    Returns:
        The absolute path to the created directory.
    """
    app_name = app_name.lower()
    if os.name == "nt":  # Windows
        if os.getenv("APPDATA") is not None:
            key_dir = os.path.join(os.getenv("APPDATA"), "Local", app_name)  # Expected
        else:
            key_dir = os.path.join(str(Path.home()), f".{app_name}")  # Fallback
    else:  # Linux/macOS
        key_dir = os.path.join(str(Path.home()), f".{app_name}")
    os.makedirs(key_dir, exist_ok=True)
    return key_dir


def get_key_considering_multiple_sources(key_variable: str | None) -> str:
    """Retrieves the UNaIVERSE authentication key from one of several sources.

    Checks, in priority order: (1) the ``key_variable`` argument, (2) the ``NODE_KEY``
    environment variable, and (3) a cached key file.  If multiple sources are found,
    a warning is printed.  If no source is found, the user is prompted interactively and
    the entered key is saved to the cache file for future use.

    Args:
        key_variable: A key string passed directly from code, or None (and placeholder
            strings enclosed in ``<...>`` are also treated as None).

    Returns:
        The resolved authentication key string.
    """

    # Creating folder (if needed) to store the key
    try:
        key_dir = prepare_app_dir(app_name="UNaIVERSE")
    except Exception:
        raise GenException("Cannot create folder to store the key file")
    key_file = os.path.join(key_dir, "key")

    # Getting from an existing file
    key_from_file = None
    if os.path.exists(key_file):
        with open(key_file, "r") as f:
            key_from_file = f.read().strip()

    # Getting from env variable
    key_from_env = os.getenv("NODE_KEY", None)

    # Getting from code-specified option
    if key_variable is not None and len(key_variable.strip()) > 0:
        key_from_var = key_variable.strip()
        if key_from_var.startswith("<") and key_from_var.endswith(">"):  # Something like <UNAIVERSE_KEY_GOES_HERE>
            key_from_var = None
    else:
        key_from_var = None

    # Finding valid sources and checking if multiple keys were provided
    _keys = [key_from_var, key_from_env, key_from_file]
    _source_names = ["your code", "env variable 'NODE_KEY'", f"cache file {key_file}"]
    source_names = []
    mismatching = False
    multiple_source = False
    first_key = None
    first_source = None
    _prev_key = None
    for i, (_key, _source_name) in enumerate(zip(_keys, _source_names)):
        if _key is not None:
            source_names.append(_source_name)
            if _prev_key is not None:
                if _key != _prev_key:
                    mismatching = True
                multiple_source = True
                _prev_key = _key
            else:
                _prev_key = _key
                first_key = _key
                first_source = _source_name

    if len(source_names) > 0:
        if multiple_source and not mismatching:
            msg = "UNaIVERSE key (the exact same key) present in multiple locations: " + ", ".join(source_names)
            print(msg)
        if multiple_source and mismatching:
            msg = "UNaIVERSE keys (different keys) present in multiple locations: " + ", ".join(source_names)
            msg += "\nLoaded the one stored in " + first_source
            print(msg)
        # if not multiple_source:
        #    msg = f"UNaIVERSE key loaded from {first_source}"
        #    print(msg)
        return first_key
    else:

        # If no key present, ask user and save to file
        print("UNaIVERSE key not present in " + ", ".join(_source_names))
        print("If you did not already do it, go to https://unaiverse.io, login, and generate a key")
        key = input("Enter your UNaIVERSE key, that will be saved to the cache file: ").strip()
        with open(key_file, "w") as f:
            f.write(key)
        return key


class PolicyFilterSelfGen:
    """Policy filter that delays self-generation or learning actions by a configurable wait time."""

    def __init__(self, wait: float, add_random_up_to: float = 0.) -> None:
        """Initialises the PolicyFilterSelfGen.

        Args:
            wait: Minimum number of seconds to wait before allowing a ``do_gen`` or
                ``do_learn`` action to proceed.
            add_random_up_to: Upper bound of an additional random delay in seconds
                added to ``wait`` (default: 0.0; negative values are clamped to 0.0).

        Raises:
            GenException: If ``wait`` is not greater than zero.
        """
        self.wait = wait
        self.add_random_up_to = max(add_random_up_to, 0.)
        if wait <= 0.:
            raise GenException("Invalid number of seconds ('wait' must be > 0)")

    def __call__(self, action_id: int, request: object, all_actions: object,
                 policy_filter_opts: dict) -> tuple[int, object]:
        """Applies the timing-based filter to the selected policy action.

        For ``do_gen`` and ``do_learn`` actions, defers execution until the configured
        wait time has elapsed since the action was first selected.

        Args:
            action_id: Index of the action selected by the policy.
            request: The request object associated with the action, or None.
            all_actions: The collection of all available actions (indexed by ``action_id``).
            policy_filter_opts: A mutable dictionary carrying per-agent filter state;
                must contain an ``"agent"`` key.

        Returns:
            A tuple ``(action_id, request)`` with the (possibly deferred) decision.
            Returns ``(-1, None)`` to signal that the action should be skipped this cycle.
        """

        # Getting basic info from the policy options (reference ot agent, and to the last time do_gen was approved)
        if 'first_t' not in policy_filter_opts:
            policy_filter_opts['first_t'] = -1

        # This is also available:
        # _agent, _first_t = policy_filter_opts['agent']

        _first_t = policy_filter_opts['first_t']

        # If the agent lives in the TuringHotel world...
        action = all_actions[action_id]
        action_name = action.name

        # We want to handle as an exception the case of "do_gen" with "u_hashes=[...processor_in]" (self-generation)
        if action_name == "do_gen" or action_name == "do_learn":

            # Saving the time when the action we were looking for was actually selected by the policy
            if _first_t < 0:
                _first_t = time.monotonic()
                policy_filter_opts['first_t'] = _first_t

            # Don't generate, don't do anything, if it passed less than 5 seconds from when we decided to generate
            if time.monotonic() - _first_t < (self.wait + random.uniform(0, self.add_random_up_to)):
                return -1, None
            else:
                policy_filter_opts['first_t'] = -1  # Clearing

        # Returning the revised policy decision
        return action_id, request


class PolicyFilterHuman:
    """Policy filter that redirects generation/learning actions to the human processor input stream."""

    def __init__(self) -> None:
        """Initialises the PolicyFilterHuman."""
        pass

    def __call__(self, action_id: int, request: object, all_actions: object,
                 policy_filter_opts: dict) -> tuple[int, object]:
        """Applies the human-input filter to the selected policy action.

        Disables the agent's processor input stream and, for ``do_gen`` / ``do_learn``
        actions with a dashed request, redirects the input hashes to the human processor's
        input stream.

        Args:
            action_id: Index of the action selected by the policy.
            request: The request object associated with the action, or None.
            all_actions: The collection of all available actions (indexed by ``action_id``).
            policy_filter_opts: A mutable dictionary carrying per-agent filter state;
                must contain ``"agent"`` and ``"public"`` keys.

        Returns:
            A tuple ``(action_id, request)`` with the (possibly altered) decision.
        """

        # Getting basic info from the policy options (reference to agent)
        agent = policy_filter_opts['agent']
        public = policy_filter_opts['public']

        # Ensuring the input stream is disabled (important)
        agent.disable_proc_input(public=public)

        # If the agent lives in the TuringHotel world...
        action = all_actions[action_id]
        action_name = action.name

        # We want to handle as an exception the case of "do_gen"
        if action_name == "do_gen" or action_name == "do_learn":

            # Checking the type of action (dashed or solid)
            if request is not None:
                is_dashed = True
                mark = request.get_mark()
                already_altered_request = False
                if mark is not None and mark == "altered_by_policy_filter":
                    already_altered_request = True
            else:
                is_dashed = False
                already_altered_request = False  # Unused (dashed only)

            # We alter the original request, forcing the input hashes to be the processor input
            if is_dashed:
                proc_input_net_hash = agent.get_proc_input_net_hash(public=public)

                if not already_altered_request:

                    # Getting the original u_hashes of the request
                    u_hashes = request.get_arg("u_hashes")

                    # Moving original u_hashes of the request to extra_hashes
                    if u_hashes is not None:
                        extra_hashes = request.get_arg("extra_hashes")
                        if extra_hashes is not None:

                            if not set(u_hashes).issubset(set(extra_hashes)):

                                # Arg 'extra_hashes' could have been already there for some world-specific reasons
                                request.alter_arg("extra_hashes", extra_hashes + u_hashes)
                        else:

                            # If arg 'extra_hashes' was not there
                            request.set_arg("extra_hashes", u_hashes)

                        request.alter_arg("u_hashes", [proc_input_net_hash])
                        request.set_mark("altered_by_policy_filter")  # Marking to avoid doing this again

                # Out of the blue: checking if 'extra_hashes' is part of the request
                extra_hashes = request.get_arg("extra_hashes")
                data_tag_from_extra_hashes = None

                if extra_hashes is not None and extra_hashes[0] not in agent.known_streams:

                    # Fallback: when the other agent disconnects and the stream in extra_hashes is not known anymore
                    agent.set_uuid(proc_input_net_hash, None, expected=False)
                    agent.set_uuid(proc_input_net_hash, None, expected=True)
                    agent.set_tag(proc_input_net_hash, -1)

                else:
                    if extra_hashes is not None:
                        extra_hashes_0 = extra_hashes[0]  # Assuming the first extra hash dictates the tag

                        # Guessing the (max) data tag of the whole stream (heuristic)
                        data_tag_from_extra_hashes = agent.get_tag(extra_hashes_0)

                    # Preparing the input stream with the request UUID
                    agent.set_uuid(proc_input_net_hash, request.uuid, expected=False)
                    agent.set_uuid(proc_input_net_hash, request.uuid, expected=True)

                    # We also force the data tag that was/is in 'extra_hashes', if 'extra_hashes' is present
                    if data_tag_from_extra_hashes is not None:
                        agent.set_tag(proc_input_net_hash, data_tag_from_extra_hashes)

        # Returning the revised policy decision
        return action_id, request


class MultiPartLimitedDict(dict):
    """A dict subclass that keeps at most a configurable number of entries per named partition."""

    def __init__(self, part_name_to_limit: dict) -> None:
        """Initialises the MultiPartLimitedDict.

        Args:
            part_name_to_limit: A dictionary mapping partition names to their maximum entry counts.
        """
        super().__init__()
        self._limits = part_name_to_limit
        self._trackers = {k: deque() for k in part_name_to_limit}
        self._current_part = next(iter(part_name_to_limit.keys()))  # Default part

    def set_part(self, part_label: str) -> None:
        """Switches the active partition so that subsequent ``__setitem__`` calls are attributed to it.

        Args:
            part_label: The name of the partition to activate.  Unknown labels are silently ignored.
        """
        if part_label in self._limits:
            self._current_part = part_label

    def __setitem__(self, key: object, value: object) -> None:
        """Inserts or updates a key/value pair in the active partition, evicting the oldest entry if full.

        Args:
            key: The dictionary key to set.
            value: The value to associate with the key.
        """
        part = self._current_part
        limit = self._limits[part]
        tracker = self._trackers[part]

        # 1. If key is already in this specific tracker, move it to the end (most recent)
        if key in tracker:
            tracker.remove(key)

        # 2. Add to the tracker
        tracker.append(key)

        # 3. If we exceeded the limit for this part, remove the oldest key
        if len(tracker) > limit:
            oldest_key = tracker.popleft()
            # Only delete from the actual dict if it's not tracked by the OTHER part
            # (Optional logic: prevents Part A from deleting Part B's keys)
            if oldest_key in self:
                super().__delitem__(oldest_key)

        # 4. Set the actual value
        super().__setitem__(key, value)
