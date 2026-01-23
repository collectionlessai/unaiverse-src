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
import random
import threading
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from unaiverse.modules.utils import HumanModule


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
        modified = [f for f in new_state if f in self.last_state and new_state[f] != self.last_state[f]]
        deleted = [f for f in self.last_state if f not in new_state]  # Track deletions

        has_changed = bool(created or modified or deleted)
        self.last_state = new_state
        return has_changed


def prepare_app_dir(app_name: str = "unaiverse"):
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


class PolicyFilterSelfGen:
    def __init__(self, wait: float, add_random_up_to: float = 0.):
        self.wait = wait
        self.add_random_up_to = max(add_random_up_to, 0.)
        if wait <= 0.:
            raise GenException("Invalid number of seconds ('wait' must be > 0)")

    def __call__(self, action_id, request, all_actions, policy_filter_opts):
        """Run the policy filter."""

        # Getting basic info from the policy options (reference ot agent, and to the last time do_gen was approved)
        if 'first_t' not in policy_filter_opts:
            policy_filter_opts['first_t'] = -1
        _agent, _first_t = policy_filter_opts['agent'], policy_filter_opts['first_t']

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
    def __init__(self):
        pass

    def __call__(self, action_id, request, all_actions, policy_filter_opts):
        """Run the policy filter."""

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


def has_human_processor(agent):
    return agent.proc is not None and isinstance(agent.proc.module, HumanModule)
