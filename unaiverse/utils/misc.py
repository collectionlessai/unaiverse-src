import os
import ast
import sys
import time
import json
import math
import threading
from tqdm import tqdm
from rich.json import JSON
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from rich.console import Console


def save_node_addresses_to_file(node, dir_path: str, public: bool):
    address_file = os.path.join(dir_path, "addresses.txt")
    with open(address_file, "w") as file:
        file.write(str(node.get_public_addresses() if public else node.get_world_addresses()))


def get_node_addresses_from_file(dir_path: str):
    while not os.path.exists(os.path.join(dir_path, "addresses.txt")):
        time.sleep(1.)
    with open(os.path.join(dir_path, "addresses.txt")) as file:
        lines = file.readlines()
        if lines[0].strip()[0] == "[":
            addresses = ast.literal_eval(lines[0].strip())
        elif lines[0].strip()[0] == "/":
            addresses = []
            for line in lines:
                _line = line.strip()
                if len(_line) > 0:
                    addresses.append(_line)
        else:
            raise ValueError("Invalid format of file address.txt")
    return addresses


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
            if s.strip():  # ignore empty lines (needed for the way tqdm works)
                self.tqdm_instance.write(s, file=self.original_stdout)

        def flush(self):
            pass  # tqdm handles flushing

    def drawing(secs: int, message: str):
        with tqdm(total=secs, desc=message, file=sys.__stdout__) as t:
            sys.stdout = TqdmPrintRedirector(t)  # redirect prints to tqdm.write
            for i in range(secs):
                time.sleep(1)
                t.update(1.)
            sys.stdout = sys.__stdout__  # restore original stdout

    sys.stdout.flush()
    handle = threading.Thread(target=drawing, args=(seconds, msg))
    handle.start()
    return handle


def countdown_wait(handle):
    handle.join()


def check_json_start(file: str, msg: str, delete_existing: bool = False):
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
    n = len(image_paths)
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)

    # load images
    images = [mpimg.imread(p) for p in image_paths]

    # determine figure size based on image sizes
    widths, heights = zip(*[(img.shape[1], img.shape[0]) for img in images])

    # use average width/height for scaling
    avg_width = sum(widths) / len(widths)
    avg_height = sum(heights) / len(heights)

    fig_width = cols * avg_width / 100
    fig_height = rows * avg_height / 100

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes = axes.flatten() if n > 1 else [axes]

    fig.canvas.manager.set_window_title("Image Grid")

    # hide unused axes
    for ax in axes[n:]:
        ax.axis('off')

    for idx, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(str(idx), fontsize=12, fontweight='bold')

    # display images
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)

    # turn on interactive mode
    plt.ion()
    plt.show()

    fig.canvas.draw()
    plt.pause(0.1)
