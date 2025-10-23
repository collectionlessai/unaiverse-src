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
import shutil
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
    def __init__(self, folder, ext=".json", prefix=None, skip=None):
        self.folder = Path(folder)
        self.ext = ext.lower()
        self.skip = skip
        self.prefix = prefix
        self.last_state = self.__scan_files()

    def __scan_files(self):
        state = {}
        for file in self.folder.iterdir():
            if ((file.is_file() and file.suffix.lower() == self.ext and
                    (self.skip is None or file.name != self.skip)) and
                    (self.prefix is None or file.name.startswith(self.prefix))):
                state[file.name] = os.path.getmtime(file)
        return state

    def something_changed(self):
        new_state = self.__scan_files()
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


class StatLoadedSaver:

    def __init__(self, base_filename: str = "stats", save_dir: str = "./", max_size_mb: int = 5,
                 dynamic_stats: set | list | tuple | None = None, static_stats: set | list | tuple | None = None,
                 group_indexed_stats: set | list | tuple | None = None, group_key: str | None = None):
        self.base_filename = base_filename
        self.save_dir = save_dir
        self.max_size_bytes = max_size_mb * 1024 * 1024

        self.time_indexed_stats = dynamic_stats
        self.static_stats = static_stats
        self.group_indexed_stats = group_indexed_stats
        self.group_key = group_key

        self.changed_stats = set()
        self.last_saved = {}  # (group_id, stat_name) -> last_saved_timestamp

        if not os.path.exists(self.save_dir) or not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

        self.__ensure_current_file()

        assert group_indexed_stats is None or len(group_indexed_stats) == 0 or (group_key != None), \
            "Specify the group key (if you have group indexed stats)"

    def mark_stat_as_changed(self, stat_name):
        self.changed_stats.add(stat_name)

    def load_existing_data(self):
        """Load all existing CSV files and rebuild last_saved timestamps."""
        self.last_saved = {}  # Reset

        # Find all files that match the pattern, to get the time indexed data
        files = []
        prefix = self.base_filename + "_"
        for f_name in os.listdir(self.save_dir):
            if f_name.startswith(prefix) and f_name.endswith(".csv"):
                try:
                    idx = int(f_name.split("_")[-1].split(".")[0])
                    files.append((idx, f_name))
                except ValueError:
                    continue

        # Sort by index to read in order
        files.sort(reverse=True)  # From the oldest to the newest
        stats = {}

        for _, f_name in files:
            path = os.path.join(self.save_dir, f_name)
            with open(path, "r") as f:
                lines = f.readlines()
                for row in lines:
                    row_tokens = row.split(',')
                    group = row_tokens[0]
                    if group == "group":  # Header row
                        continue
                    stat_name = row_tokens[1]
                    ts = row_tokens[2]
                    val = float(row_tokens[3])
                    last_ts = self.last_saved.get((group, stat_name), float("-1.0"))
                    if ts > last_ts:
                        self.last_saved[(group, stat_name)] = ts
                    stats[self.group_key][group][stat_name][ts] = val

        # Set file_index to one past the highest existing index
        self.__ensure_current_file()

    def save_incremental(self, stats):
        """Save every static not-grouped stats to its own JSON file; save static grouped stats in a single, shared CSV;
        save dynamic stats (grouped and not) to a single, shared CSV => only new data points since the last call."""

        # Static (and not group indexed) => <base_filename>_<stat_name>.json
        for stat_name in self.static_stats:
            if stat_name not in self.group_indexed_stats:
                if stat_name not in self.changed_stats:
                    data = stats.get(stat_name, {})
                    with open(os.path.join(self.save_dir, f"{self.base_filename}_{stat_name}.json"), "w") as f:
                        json.dump(data, f)

        # Static and group indexed => <base_filename>_static.csv
        shared_static_stats_changed = False
        for stat_name in self.static_stats:
            if stat_name in self.group_indexed_stats:
                if stat_name not in self.changed_stats:
                    shared_static_stats_changed = True
        stats_list = [s for s in self.static_stats if s in self.group_indexed_stats]
        if shared_static_stats_changed and len(stats_list) > 0:
            header = ["group"] + stats_list
            with open(os.path.join(self.save_dir, f"{self.base_filename}_static.json"), "w") as f:
                f.write(",".join(header) + "\n")

                group_to_group_stats = stats[self.group_key]
                for group_name, group_stats in group_to_group_stats.items():
                    row = [group_name]
                    for stat_name in self.static_stats:
                        if stat_name in self.group_indexed_stats and stat_name in group_stats:
                            row.append(group_stats[stat_name])
                        f.write(",".join(row) + "\n")

        # Dynamic (both group indexed and not group indexed) => <base_filename>_1.csv, <base_filename>_2.csv, ...
        filename = self.__current_filename()
        self.__ensure_current_file()

        with open(filename, "a") as f:

            # Dynamic and not group indexed (introducing a fake group to handle all of them the same way)
            group_to_group_stats = {}
            fake_group_for_not_grouped_stats = "<ungrouped>"
            for stat_name in self.time_indexed_stats:
                if stat_name not in self.group_indexed_stats and stat_name in stats:
                    if fake_group_for_not_grouped_stats not in group_to_group_stats:
                        group_to_group_stats[fake_group_for_not_grouped_stats] = {}
                    group_to_group_stats[fake_group_for_not_grouped_stats][stat_name] = stats[stat_name]

            # Dynamic and group indexed
            if self.group_key in stats:
                group_to_group_stats.update(stats[self.group_key])

            # Dynamic (not they are all group indexed, thanks to the introduction of the fake group)
            for group_name, group_stats in group_to_group_stats.items():
                for stat_name in self.time_indexed_stats:
                    if stat_name in self.group_indexed_stats and stat_name in group_stats:
                        timestamps = group_stats[stat_name].keys()
                        last_ts = self.last_saved.get((group_name, stat_name), float("-1.0"))

                        for ts in timestamps:
                            if ts > last_ts:
                                value = group_stats[stat_name][ts]
                                row = [group_name, stat_name, ts, value]
                                f.write(",".join(row) + "\n")
                                self.last_saved[(group_name, stat_name)] = ts

        # Clearing markers
        self.changed_stats = set()

    def __current_filename(self):
        """Always return the newest (index 1) file."""
        return os.path.join(self.save_dir, f"{self.base_filename}_{1:06d}.csv")

    def __ensure_current_file(self):
        """Ensure the current newest file is <base_filename>_000001.csv. If rotation is needed, shift existing files."""
        filename = self.__current_filename()  # This will return the file with suffix '_1'
        stats_list = [s for s in self.time_indexed_stats if s in self.group_indexed_stats]

        if len(stats_list) > 0:

            # If current file exists but is too large, rotate all existing ones upward
            if os.path.exists(filename) and os.path.getsize(filename) >= self.max_size_bytes:
                self.__rotate_files_up()

                # Create a new fresh file as _1
                with open(filename, "w") as f:
                    header = ["group"] + stats_list
                    f.write(",".join(header) + "\n")
            elif not os.path.exists(filename):

                # Create _1 if it does not exist
                with open(filename, "w") as f:
                    header = ["group"] + stats_list
                    f.write(",".join(header) + "\n")

    def __rotate_files_up(self):
        """Shift existing files upward by 1 index (e.g. _1 -> _2; _2 -> _3, etc.)."""
        prefix = self.base_filename + "_"
        files = []
        for f_name in os.listdir(self.save_dir):
            if f_name.startswith(prefix) and f_name.endswith(".csv"):
                try:
                    idx = int(f_name.split("_")[-1].split(".")[0])
                    files.append((idx, f_name))
                except ValueError:
                    continue

        # Sort descending so renaming does not overwrite
        files.sort(reverse=True)

        for idx, f_name in files:
            src = os.path.join(self.save_dir, f_name)
            dst = os.path.join(self.save_dir, f"{self.base_filename}_{idx+1:06d}.csv")
            shutil.move(src, dst)
