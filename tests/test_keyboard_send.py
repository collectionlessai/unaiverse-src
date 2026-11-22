"""The keyboard of interactive mode (node.py): what the person types or sends with /send goes in the input of its type
of the human processor (text, img, file, all), and /send sends an image as image data, anything else as FILE data."""
import types
import pytest
from PIL import Image
from unaiverse.networking.node.node import Node
from unaiverse.streams.dataprops import DataProps, FileContainer


class _Stdin:
    """The default stdin of a fake agent: it keeps what was set."""

    def __init__(self) -> None:
        self.values = None

    def __len__(self) -> int:
        return 1

    def set(self, values: list, uuid: str | None = None, data_tag: int | None = None, force: bool = False) -> None:
        self.values = values


def _node(input_types: list[str]) -> types.SimpleNamespace:
    """A stand-in for the node: only the agent the keyboard methods use."""
    agent = types.SimpleNamespace(
        stdin=_Stdin(),
        proc_inputs=[DataProps(data_type=t) for t in input_types],
        prepare_stdin_if_human=lambda public, peer_id=None: ("uuid", 0),
    )
    return types.SimpleNamespace(agent=agent)


def _to_stdin(node: types.SimpleNamespace, **kwargs: object) -> None:
    Node._Node__keyboard_to_stdin(node, True, "peer", **kwargs)


def _send(node: types.SimpleNamespace, path: str) -> None:
    node._Node__keyboard_to_stdin = lambda *a, **k: Node._Node__keyboard_to_stdin(node, *a, **k)
    Node._Node__send_file_from_keyboard(node, path, True, "peer")


def test_each_piece_goes_in_the_input_of_its_type():
    node = _node(["text", "img", "file", "all"])
    _to_stdin(node, text="hi", whatever=42)
    assert node.agent.stdin.values == ["hi", None, None, 42]


def test_one_input_takes_the_text():
    node = _node(["text"])
    _to_stdin(node, text="hi")
    assert node.agent.stdin.values == ["hi"]


def test_send_an_image_as_image_data(tmp_path):
    path = tmp_path / "pic.png"
    Image.new("RGB", (3, 2), "red").save(path)
    node = _node(["text", "img", "file", "all"])
    _send(node, str(path))
    text, img, file, _ = node.agent.stdin.values
    assert text is None and file is None
    assert isinstance(img, Image.Image) and img.size == (3, 2)


@pytest.mark.parametrize("name, content", [("notes.txt", b"some notes"), ("fake.png", b"not a png")])
def test_send_anything_else_as_file_data(tmp_path, name, content):
    path = tmp_path / name
    path.write_bytes(content)
    node = _node(["text", "img", "file", "all"])
    _send(node, str(path))
    _, img, file, _ = node.agent.stdin.values
    assert img is None
    assert isinstance(file, FileContainer) and file.filename == name and file.path == str(path)


def test_send_without_a_file_input_or_a_file_sends_nothing(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes(b"x")
    node = _node(["text", "img", "all"])
    _send(node, str(path))
    _send(node, str(tmp_path / "missing.txt"))
    assert node.agent.stdin.values is None
