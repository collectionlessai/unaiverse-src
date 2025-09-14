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
import threading
from tqdm import tqdm
from pathlib import Path
from datetime import datetime


class GenException(Exception):
    """Base exception for this application (a simple wrapper around a generic Exception)."""
    pass


def save_node_addresses_to_file(node, dir_path: str, public: bool,
                                filename: str = "addresses.txt", append: bool = False):
    address_file = os.path.join(dir_path, filename)
    with open(address_file, "w" if not append else "a") as file:
        file.write(node.hosted.get_name() + ";" +
                   str(node.get_public_addresses() if public else node.get_world_addresses()) + "\n")
        file.flush()


def get_node_addresses_from_file(dir_path: str, filename: str = "addresses.txt") -> dict[str, list[str]]:
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
    def __init__(self, ignore: bool = False):
        self.ignore = ignore

    def __enter__(self):
        if not self.ignore:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.ignore:
            sys.stdout.close()
            sys.stdout = self._original_stdout


# The countdown function
def countdown_start(seconds: int, msg: str):
    class TqdmPrintRedirector:
        def __init__(self, tqdm_instance):
            self.tqdm_instance = tqdm_instance
            self.original_stdout = sys.__stdout__

        def write(self, s):
            if s.strip():  # Ignore empty lines (needed for the way tqdm works)
                self.tqdm_instance.write(s, file=self.original_stdout)

        def flush(self):
            pass  # Tqdm handles flushing

    def drawing(secs: int, message: str):
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


def countdown_wait(handle):
    handle.join()


def check_json_start(file: str, msg: str, delete_existing: bool = False):
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
                    with open(file_path, "r") as f:
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


def check_json_start_wait(handle):
    handle.join()


def show_images_grid(image_paths, max_cols=3):
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

    # Display images
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)

    # Turn on interactive mode
    plt.ion()
    plt.show()

    fig.canvas.draw()
    plt.pause(0.1)


class FileTracker:
    def __init__(self, folder, ext=".json"):
        self.folder = Path(folder)
        self.ext = ext.lower()
        self.last_state = self._scan_files()

    def _scan_files(self):
        state = {}
        for file in self.folder.iterdir():
            if file.is_file() and file.suffix.lower() == self.ext:
                state[file.name] = os.path.getmtime(file)
        return state

    def something_changed(self):
        new_state = self._scan_files()
        created = [f for f in new_state if f not in self.last_state]
        modified = [f for f in new_state
                    if f in self.last_state and new_state[f] != self.last_state[f]]
        self.last_state = new_state
        return created or modified


def prepare_key_dir(app_name):
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

    # Creating folder (if needed) to store the key
    try:
        key_dir = prepare_key_dir(app_name="UNaIVERSE")
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
            else:
                _prev_key = _key
                first_key = _key
                first_source = _source_name

    if len(source_names) > 0:
        msg = ""
        if multiple_source and not mismatching:
            msg = "UNaIVERSE key (the exact same key) present in multiple locations: " + ", ".join(source_names)
        if multiple_source and mismatching:
            msg = "UNaIVERSE keys (different keys) present in multiple locations: " + ", ".join(source_names)
            msg += "\nLoaded the one stored in " + first_source
        if not multiple_source:
            msg = f"UNaIVERSE key loaded from {first_source}"
        print(msg)
        return first_key
    else:

        # If no key present, ask user and save to file
        print("UNaIVERSE key not present in " + ", ".join(_source_names))
        print("If you did not already do it, go to https://unaiverse.io, login, and generate a key")
        key = input("Enter your UNaIVERSE key, that will be saved to the cache file: ").strip()
        with open(key_file, "w") as f:
            f.write(key)
        return key
