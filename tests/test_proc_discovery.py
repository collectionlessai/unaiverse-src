"""
Automatic discovery of the inputs and outputs of a processor (AgentProcessorChecker), when they are not given.

Each case checks what is discovered and that the agent can then run real data through what was discovered (the
streams check the number of dimensions, the batch one included): text and image modules found from the annotations
of their forward, Linear / Conv1d / Conv2d / Embedding modules from their first layer, a plain annotated function,
a module with nothing to read (a clear error), and a stateful module whose state the probes leave as it was.
"""
import torch
import pytest
import asyncio
from PIL import Image
from unaiverse.agent import Agent
from unaiverse.custom import GenException


class _Echo(torch.nn.Module):
    """A text module with no layer: found from the annotation of its forward."""

    def forward(self, text: str) -> str:
        """Echoes.

        Args:
            text: The input.

        Returns:
            The input with a mark.
        """
        return text + "!"


class _TextToTextAndTensor(torch.nn.Module):
    """A text module with two outputs, a text and a tensor."""

    def forward(self, text: str) -> tuple:
        """Answers with a text and a tensor.

        Args:
            text: The input.

        Returns:
            The text and a (1, 4) tensor.
        """
        return text, torch.zeros(1, 4)


class _TokenPool(torch.nn.Module):
    """Token ids in, a vector out: an embedding averaged over the sequence, then a linear layer."""

    def __init__(self) -> None:
        """Initializes the module."""
        super().__init__()
        self.emb = torch.nn.Embedding(10, 4)
        self.out = torch.nn.Linear(4, 3)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embeds, averages, projects.

        Args:
            tokens: The token ids, (batch, sequence).

        Returns:
            A (batch, 3) tensor.
        """
        return self.out(self.emb(tokens).mean(dim=1))


class _NoHint(torch.nn.Module):
    """A module with no layer and an input that says neither text nor image: nothing to discover from."""

    def forward(self, x: object) -> object:
        """Echoes.

        Args:
            x: The input.

        Returns:
            The input.
        """
        return x


class _Stateful(torch.nn.Module):
    """A Linear model with a hidden state that each call changes (as an RNN would)."""

    def __init__(self) -> None:
        """Initializes the module."""
        super().__init__()
        self.lin = torch.nn.Linear(3, 3)
        self.register_buffer("h", torch.zeros(1, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Updates the state.

        Args:
            x: The input.

        Returns:
            The new state.
        """
        self.h = torch.tanh(self.lin(x) + self.h)
        return self.h


def _discover(mod: object) -> Agent:
    """Builds an agent with nothing described about its processor.

    Args:
        mod: The processor.

    Returns:
        The agent.
    """
    return Agent(proc=mod, proc_inputs=None, proc_outputs=None)


def _shapes(descs: list) -> list:
    """The data type and shape of each descriptor.

    Args:
        descs: The descriptors.

    Returns:
        The list of (data type, tensor shape).
    """
    return [(d.data_type, tuple(d.tensor_shape) if d.tensor_shape is not None else None) for d in descs]


def test_text_modules_are_found_from_their_annotations() -> None:
    """A text module with no layer gets a text input; its outputs (text, then a tensor) are found by a probe."""
    a = _discover(_Echo())
    assert _shapes(a.proc.proc_inputs) == [("text", None)] and _shapes(a.proc.proc_outputs) == [("text", None)]
    assert asyncio.run(a.proc.aforward("hi")) == ("hi!",)
    b = _discover(_TextToTextAndTensor())
    assert _shapes(b.proc.proc_outputs) == [("text", None), ("tensor", (None, 4))]
    text, tensor = asyncio.run(b.proc.aforward("x"))
    assert text == "x" and tuple(tensor.shape) == (1, 4)


def test_a_plain_annotated_function_is_found_too() -> None:
    """Not a torch module: a function whose first argument is declared as text."""

    def shout(text: str) -> str:
        """Shouts.

        Args:
            text: The input.

        Returns:
            The input, upper case.
        """
        return text.upper()

    a = _discover(shout)
    assert _shapes(a.proc.proc_inputs) == [("text", None)] and _shapes(a.proc.proc_outputs) == [("text", None)]
    assert asyncio.run(a.proc.aforward("hey")) == ("HEY",)


def test_layer_modules_get_shapes_with_the_batch_dimension_and_accept_batched_data() -> None:
    """Linear, Conv1d and an image CNN: shapes batch first (None), and real batched data goes through."""
    lin = _discover(torch.nn.Linear(3, 2))
    assert _shapes(lin.proc.proc_inputs) == [("tensor", (None, 3))]
    assert _shapes(lin.proc.proc_outputs) == [("tensor", (None, 2))]
    assert tuple(asyncio.run(lin.proc.aforward(torch.ones(5, 3)))[0].shape) == (5, 2)

    conv1d = _discover(torch.nn.Sequential(torch.nn.Conv1d(2, 4, 3), torch.nn.AdaptiveAvgPool1d(1),
                                           torch.nn.Flatten()))
    assert _shapes(conv1d.proc.proc_inputs) == [("tensor", (None, 2, None))]
    assert tuple(asyncio.run(conv1d.proc.aforward(torch.ones(1, 2, 16)))[0].shape) == (1, 4)

    cnn = _discover(torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.AdaptiveAvgPool2d(1),
                                        torch.nn.Flatten(), torch.nn.Linear(4, 5)))
    assert _shapes(cnn.proc.proc_inputs) == [("img", None)]
    assert _shapes(cnn.proc.proc_outputs) == [("tensor", (None, 5))]
    assert tuple(asyncio.run(cnn.proc.aforward(Image.new("RGB", (32, 32))))[0].shape) == (1, 5)


def test_token_modules_get_integer_dummies() -> None:
    """An Embedding first: token ids (long) as input, probed with integer tokens."""
    emb = _discover(_TokenPool())
    assert _shapes(emb.proc.proc_inputs) == [("tensor", (None, None))]
    assert emb.proc.proc_inputs[0].tensor_dtype == torch.long
    assert _shapes(emb.proc.proc_outputs) == [("tensor", (None, 3))]
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    assert tuple(asyncio.run(emb.proc.aforward(tokens))[0].shape) == (1, 3)


def test_a_module_with_nothing_to_read_is_a_clear_error() -> None:
    """No layer and no annotation: the agent says the inputs must be described."""
    with pytest.raises((ValueError, GenException)):
        _discover(_NoHint())


def test_the_probes_leave_the_state_of_a_module_as_it_was() -> None:
    """A stateful module: after the discovery, its state is the one it had before."""
    mod = _Stateful()
    before = mod.h.clone()
    a = _discover(mod)
    assert _shapes(a.proc.proc_outputs) == [("tensor", (None, 3))]
    assert torch.equal(mod.h, before)
