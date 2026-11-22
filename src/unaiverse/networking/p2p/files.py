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
import shutil
import unicodedata
from typing import Any
from unaiverse.custom import Custom

# The files of the file messages (Phase 6.5): where the received ones go, and which samples of a message to send are
# streamed from disk instead of being put in the message. Stdlib only, as the rest of the transport tier.

INCOMING_FOLDER = ".incoming"  # Under the downloads folder: the .part files Go writes, one folder per instance
FALLBACK_NAME = "file"  # The name of a file whose name is nothing once made safe


def downloads_dir() -> str:
    """The folder of the received files: Custom.DOWNLOADS_DIR, or <app folder>/Downloads (the folder of the key).

    Returns:
        The absolute path of the folder (not created here).
    """
    if Custom.DOWNLOADS_DIR:
        return os.path.abspath(os.path.expanduser(Custom.DOWNLOADS_DIR))
    from unaiverse.utils.misc import prepare_app_dir
    return os.path.join(prepare_app_dir("unaiverse"), "Downloads")


def incoming_dir() -> str:
    """The folder where Go writes the files while they arrive (each instance in a subfolder of its peer ID).

    Returns:
        The absolute path of the folder.
    """
    return os.path.join(downloads_dir(), INCOMING_FOLDER)


def clear_incoming(peer_id: str) -> None:
    """Removes what an instance left in its incoming folder (files of messages that never completed or that were
    never popped). Only its own folder: the instances of a machine can share the downloads folder.

    Args:
        peer_id: The peer ID of the instance.
    """
    folder = os.path.join(incoming_dir(), safe_name(peer_id))
    if os.path.isdir(folder):
        shutil.rmtree(folder, ignore_errors=True)


def safe_name(name: str | None) -> str:
    """A name that the peer gave, made safe as one path component: no separators, no "..", no control characters,
    no leading dots or spaces. A name that becomes empty is FALLBACK_NAME.

    Args:
        name: The name.

    Returns:
        The safe name.
    """
    name = unicodedata.normalize("NFC", str(name or ""))
    name = "".join("_" if (c in "/\\:" or unicodedata.category(c).startswith("C")) else c for c in name)
    name = name.strip().lstrip(".").strip()
    if name in ("", ".", ".."):
        return FALLBACK_NAME
    return name[:200]


def unique_path(folder: str, filename: str) -> str:
    """A path in a folder for a file of the given name that does not exist yet: "report.pdf", else
    "report (1).pdf", "report (2).pdf", ...

    Args:
        folder: The folder.
        filename: The safe name.

    Returns:
        The path.
    """
    stem, ext = os.path.splitext(filename)
    path, n = os.path.join(folder, filename), 0
    while os.path.exists(path):
        n += 1
        path = os.path.join(folder, f"{stem} ({n}){ext}")
    return path


def place_received_file(incoming_path: str, owner: tuple[str, str] | None, sender_peer_id: str,
                        filename: str | None) -> str:
    """Moves a received file from the incoming folder to Downloads/<handle>/<node_name>/<filename> (a rename on the
    same filesystem), never overwriting a file already there. A sender whose profile is unknown gets
    Downloads/unknown/<peer ID>/.

    Args:
        incoming_path: Where Go wrote it.
        owner: The handle and node name of the sender, or None.
        sender_peer_id: The peer ID of the sender.
        filename: The name of the file, as the sender gave it.

    Returns:
        The path of the file in its place.
    """
    parts = (owner[0], owner[1]) if owner is not None else ("unknown", sender_peer_id)
    folder = os.path.join(downloads_dir(), *(safe_name(p) for p in parts))
    os.makedirs(folder, exist_ok=True)

    # Exclusive creation of the name closes the race with another node of the same machine placing the same name
    while True:
        path = unique_path(folder, safe_name(filename))
        try:
            os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
            break
        except FileExistsError:
            continue
    os.replace(incoming_path, path)
    return path


def streamed_files(content_type: str, content: Any) -> list[dict]:
    """The samples of a message whose files are sent from disk, next to the message: the FILE samples of a
    stream_sample message whose container has a path (with the name the receiver shows in its progress).

    Args:
        content_type: The content type of the message.
        content: Its content.

    Returns:
        A list of {"key": sample name, "path": file, "name": its name}, empty when there is nothing to stream.
    """
    if content_type != "stream_sample" or not isinstance(content, dict):
        return []
    out = []
    for name, sample in content.items():
        data = sample.get("data") if isinstance(sample, dict) else None
        path = getattr(data, "path", None)
        if path is not None:
            label = str(getattr(data, "filename", "")).encode("utf-8")[:255].decode("utf-8", errors="ignore")
            out.append({"key": name, "path": os.path.abspath(path), "name": label})
    return out


def remove_quietly(paths: list[str]) -> None:
    """Removes files nobody will receive, ignoring the ones already gone.

    Args:
        paths: The files.
    """
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass
