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
import json
import time
import torch
import random
import asyncio
import inspect
import logging
import tempfile
import functools
import numpy as np
from PIL import Image
from typing import Callable
from unaiverse.utils.logger import log
from torch.utils.data import DataLoader
from datetime import datetime, timezone
from unaiverse.custom import GenException
from unaiverse.uai import has_fence, AnswerWithheld
from torchvision import datasets, transforms
from unaiverse.streams.dataprops import StreamType


def has_human_processor(agent: object) -> bool:
    """Checks whether the given agent has a human processor module.

    Args:
        agent: The agent object to inspect; expected to have a ``proc`` attribute.

    Returns:
        True if the agent's processor module is a HumanModule, False otherwise.
    """
    assert hasattr(agent, "proc")
    return getattr(agent, "proc") is not None and isinstance(getattr(agent, "proc").module, HumanModule)


def transforms_factory(trans_type: str, add_batch_dim: bool = True,
                       return_inverse: bool = False) -> transforms.Compose | None:
    """Builds and returns a torchvision transform pipeline for the given type.

    Args:
        trans_type: A string identifying the desired transform.  Supported values are
            ``"rgb<N>"`` / ``"gray<N>"`` (resize + center-crop to ``<N>`` pixels, with normalization).
            ``"rgb-no_norm<N>"`` / ``"gray-no_norm<N>"`` (resize + center-crop, no normalization).
            ``"rgb"`` / ``"gray"`` (full-image with normalization).
            ``"rgb-no_norm"`` / ``"gray-no_norm"`` (full-image, no normalization).
            ``"gray_mnist"`` (MNIST-specific 28×28 grayscale pipeline).
        add_batch_dim: If True, appends an ``unsqueeze(0)`` step to the forward transform and
            inserts a ``squeeze(0)`` step at the start of the inverse transform (default: True).
        return_inverse: If True, returns the inverse (de-normalization) transform instead of the
            forward transform (default: False).

    Returns:
        The requested ``transforms.Compose`` pipeline (or None, if the type is unknown).
    """
    supported_types = {"rgb*",          "gray*",
                       "rgb-no_norm*",  "gray-no_norm*",
                       "rgb",           "gray",
                       "rgb-no_norm",   "gray-no_norm",
                                        "gray_mnist"}

    found = False
    num = -1
    for _type in supported_types:
        if _type.endswith("*"):
            has_star = True
            __type = _type[0:-1]
        else:
            has_star = False
            __type = _type

        if has_star and trans_type.startswith(__type) and len(trans_type) > len(__type):
            try:
                num = int(trans_type[len(__type):])
                trans_type = _type
                found = True
                break
            except ValueError:
                pass
        elif trans_type == _type:
            found = True
            break

    if not found:
        raise ValueError(f"Invalid transformation type '{trans_type}': must be one of {supported_types}, "
                         f"where * is an integer number")

    trans = None
    inverse_trans = None

    if trans_type == "rgb*":
        trans = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),  # Ensure 3 chans
            transforms.Resize(num),
            transforms.CenterCrop(num),
            transforms.ToTensor(),  # Convert PIL to tensor (3, H, W), float [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        inverse_trans = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.],
                                 std=[1. / 0.229, 1. / 0.224, 1. / 0.225]),
            transforms.Lambda(lambda x: torch.clamp(x + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1),
                                                    min=0.0, max=1.0)),
            transforms.ToPILImage()
        ])
    elif trans_type == "gray*":
        trans = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("L") if img.mode != "L" else img),  # Ensure 1 channel
            transforms.Resize(num),
            transforms.CenterCrop(num),
            transforms.ToTensor(),  # Convert PIL to tensor (1, H, W), float [0,1]
            transforms.Normalize(mean=[0.45],
                                 std=[0.225])
        ])
        inverse_trans = transforms.Compose([
            transforms.Normalize(mean=[0.],
                                 std=[1. / 0.225]),
            transforms.Lambda(lambda x: torch.clamp(x + 0.45, min=0.0, max=1.0)),
            transforms.ToPILImage()
        ])
    elif trans_type == "rgb-no_norm*":
        trans = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),  # Ensure 3 channels
            transforms.Resize(num),
            transforms.CenterCrop(num),
            transforms.PILToTensor(),  # Convert PIL to tensor (3, H, W), uint [0,255]
        ])
        inverse_trans = transforms.Compose([transforms.ToPILImage()])
    elif trans_type == "gray-no_norm*":
        trans = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("L") if img.mode != "L" else img),  # Ensure 1 channel
            transforms.Resize(num),
            transforms.CenterCrop(num),
            transforms.PILToTensor(),  # Convert PIL to tensor (1, H, W), uint [0,255]
        ])
        inverse_trans = transforms.Compose([transforms.ToPILImage()])
    elif trans_type == "rgb":
        trans = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),  # Ensure 3 channels
            transforms.ToTensor(),  # Convert PIL to tensor (3, H, W), float [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        inverse_trans = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.],
                                 std=[1. / 0.229, 1. / 0.224, 1. / 0.225]),
            transforms.Lambda(lambda x: torch.clamp(x + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1),
                                                    min=0.0, max=1.0)),
            transforms.ToPILImage()
        ])
    elif trans_type == "gray":
        trans = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("L") if img.mode != "L" else img),  # Ensure 1 channel
            transforms.ToTensor(),  # Convert PIL to tensor (1, H, W), float [0,1]
            transforms.Normalize(mean=[0.45],
                                 std=[0.225])
        ])
        inverse_trans = transforms.Compose([
            transforms.Normalize(mean=[0.],
                                 std=[1. / 0.225]),
            transforms.Lambda(lambda x: torch.clamp(x + 0.45, min=0.0, max=1.0)),
            transforms.ToPILImage()
        ])
    elif trans_type == "rgb-no_norm":
        trans = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),  # Ensure 3 channels
            transforms.PILToTensor(),  # Convert PIL to tensor (3, H, W), uint [0,255]
        ])
        inverse_trans = transforms.Compose([transforms.ToPILImage()])
    elif trans_type == "gray-no_norm":
        trans = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("L") if img.mode != "L" else img),  # Ensure 1 channel
            transforms.PILToTensor(),  # Convert PIL to tensor (1, H, W), uint [0,255]
        ])
        inverse_trans = transforms.Compose([transforms.ToPILImage()])
    elif trans_type == "gray_mnist":
        trans = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("L") if img.mode != "L" else img),  # Ensure 1 channel
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor(),  # Convert PIL to tensor (1, H, W), float [0,1]
            transforms.Normalize(mean=[0.1307],  # MNIST
                                 std=[0.3081])  # MNIST
        ])
        inverse_trans = transforms.Compose([
            transforms.Normalize(mean=[0.],
                                 std=[1. / 0.3081]),
            transforms.Lambda(lambda x: torch.clamp(x + 0.1307, min=0.0, max=1.0)),
            transforms.ToPILImage()
        ])

    if add_batch_dim and (trans is not None and inverse_trans is not None):
        trans.transforms.append(transforms.Lambda(lambda x: x.unsqueeze(0)))
        inverse_trans.transforms.insert(0, transforms.Lambda(lambda x: x.squeeze(0)))

    return trans if not return_inverse else inverse_trans


def hard_tanh(x: torch.Tensor) -> torch.Tensor:
    """Clamps tensor values to the range [-1, 1] (hard tanh activation).

    Args:
        x: Input tensor.

    Returns:
        Tensor with all values clamped to ``[-1, 1]``.
    """
    return torch.clamp(x, min=-1., max=1.)


def target_shape_fixed_cross_entropy(output: torch.Tensor, target: torch.Tensor,
                                     *args, **kwargs) -> torch.Tensor:
    """Computes cross-entropy loss, squeezing a leading batch dimension from the target if present.

    Args:
        output: The model output logits tensor.
        target: The target class-index tensor; if it has more than one dimension the first
            dimension is squeezed before calling ``cross_entropy``.
        *args: Additional positional arguments forwarded to ``torch.nn.functional.cross_entropy``.
        **kwargs: Additional keyword arguments forwarded to ``torch.nn.functional.cross_entropy``.

    Returns:
        The scalar cross-entropy loss tensor.
    """
    if len(target.shape) > 1:
        target = target.squeeze(0)
    return torch.nn.functional.cross_entropy(output, target, *args, **kwargs)


def set_seed(seed: int) -> None:
    """Seeds all relevant random-number generators for reproducibility.

    Args:
        seed: The integer seed value.  If negative, no seeding is performed.
    """
    if seed >= 0:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)


def get_proc_inputs_and_proc_outputs_for_rnn(u_shape: torch.Size | tuple, du_dim: int,
                                             y_dim: int) -> tuple[list, list]:
    """Creates Stream descriptors for the inputs and output of an RNN-style processor.

    Args:
        u_shape: Shape of the main input tensor (excluding the batch dimension).
        du_dim: Dimensionality of the secondary (delta-u) input.
        y_dim: Dimensionality of the output.

    Returns:
        A tuple ``(proc_inputs, proc_outputs)`` where each element is a list of
        ``Stream`` objects describing the processor's I/O signature.
    """
    if isinstance(u_shape, torch.Size):
        u_shape = tuple(u_shape)
    proc_inputs = [
        StreamType(data_type="tensor", tensor_shape=(None,) + u_shape, tensor_dtype=torch.float32,
                   pubsub=False, private_only=False),
        StreamType(data_type="tensor", tensor_shape=(None, du_dim,), tensor_dtype=torch.float32,
                   pubsub=False, private_only=False)
    ]
    proc_outputs = [
        StreamType(data_type="tensor", tensor_shape=(None, y_dim), tensor_dtype=torch.float32,
                   pubsub=False, private_only=False)
    ]
    return proc_inputs, proc_outputs


def get_proc_inputs_and_proc_outputs_for_image_classification(y_dim: int = -1, trans_forms=None) \
        -> tuple[list[StreamType], list[StreamType]]:
    """Creates Stream descriptors for the inputs and output of an image-classification processor.

    Args:
        y_dim: Number of output classes.  Pass ``-1`` to default to 1000 (ImageNet).
        trans_forms: Stream to processor transforms, if any (Default: None).

    Returns:
        A tuple ``(proc_inputs, proc_outputs)`` where each element is a list of
        ``Stream`` objects describing the processor's I/O signature.
    """
    if y_dim == -1:
        y_dim = 1000  # Assuming ImageNet-trained models
    proc_inputs = [StreamType(data_type="img", pubsub=False, private_only=False, stream_to_proc_transforms=trans_forms)]
    proc_outputs = [StreamType(data_type="tensor", tensor_shape=(None, y_dim), tensor_dtype=torch.float32,
                               pubsub=False, private_only=False)]
    return proc_inputs, proc_outputs


def isinstance_fcn(obj: object, class_to_check: type | tuple) -> bool:
    """Thin wrapper around the built-in ``isinstance`` function.

    Args:
        obj: The object to test.
        class_to_check: A type or tuple of types to check against.

    Returns:
        True if ``obj`` is an instance of ``class_to_check``, False otherwise.
    """
    return isinstance(obj, class_to_check)


def error_rate_mnist_test_set(network: torch.nn.Module, mnist_data_save_path: str) -> float:
    """Evaluates a network's classification error rate on the MNIST test set, in one
    blocking call (drains the stepwise generator above).

    Args:
        network: The PyTorch module to evaluate.
        mnist_data_save_path: Path where the MNIST dataset will be downloaded and cached.

    Returns:
        The fraction of misclassified test samples (in [0, 1]).
    """
    gen = error_rate_mnist_test_set_steps(network, mnist_data_save_path)
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            return stop.value


def error_rate_mnist_test_set_steps(network: torch.nn.Module, mnist_data_save_path: str):
    """Generator variant of error_rate_mnist_test_set, one chunk of work per yield: the
    dataset setup is a chunk of its own (it may touch the disk), then one test batch per
    chunk. Built for AgentBasics.stepwise, so the node loop keeps pumping during the
    evaluation. torch.no_grad() is entered and exited within each chunk (grad mode is
    thread-global: it must never stay flipped across a yield), and the train-mode
    restore lives in a finally block, which also runs if an abandoned generator is
    garbage-collected. Returns the error rate like the plain function.

    Args:
        network: The PyTorch module to evaluate.
        mnist_data_save_path: Path where the MNIST dataset will be downloaded and cached.

    Returns:
        The fraction of misclassified test samples (in [0, 1]).
    """

    # Getting MNIST test set
    mnist_test = datasets.MNIST(root=mnist_data_save_path,
                                train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))
    mnist_test = DataLoader(mnist_test, batch_size=200, shuffle=False)
    yield

    # Checking error rate
    error_rate = 0.
    n = 0
    training_flag_backup = network.training
    network.eval()
    try:
        device = next(network.parameters()).device
        for x, y in mnist_test:
            with torch.no_grad():
                x = x.to(device)
                y = y.to(device)
                o = network(x)
                c = torch.argmax(o, dim=1)
                error_rate += float(torch.sum(c != y).item())
                n += x.shape[0]
            yield
    finally:
        network.train(training_flag_backup)

    return error_rate / n


class MultiIdentity(torch.nn.Module):
    """Identity module that passes one or more inputs through unchanged."""

    # A relay gives back what it was given: rewriting its input would corrupt the message it forwards
    UAI_ECHOES_INPUT = True

    def __init__(self) -> None:
        super().__init__()

    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def forward(self, *args: object, **kwargs: object):
        """Returns the single input unchanged, or all positional inputs as a tuple.

        Args:
            *args: One or more input tensors / objects.
            **kwargs: Unused.

        Returns:
            The single input if exactly one positional argument is given, otherwise the
            full ``args`` tuple.
        """
        if len(args) == 1:
            return args[0]
        return args


class HumanModule(torch.nn.Module):
    """Dummy human-in-the-loop module that echoes text and image inputs unchanged."""

    # A person's processor gives back what the person wrote: that text is already wire text
    UAI_ECHOES_INPUT = True

    def __init__(self) -> None:
        super().__init__()

    # noinspection PyMethodMayBeStatic
    # noinspection PyUnusedLocal
    def forward(self, text: str | None = None, img: Image.Image | None = None,
                whatever: object | None = None) -> tuple[str | None, Image.Image | None]:
        """Returns the text and image inputs as-is (placeholder for human interaction).

        Args:
            text: Optional text input.
            img: Optional PIL image input.
            whatever: Optional additional input (ignored).

        Returns:
            A tuple ``(text, img)`` with the unchanged inputs.
        """
        return text, img


class LoggerModule(torch.nn.Module):
    """A module that logs inputs and outputs to a file and cycles through a vocabulary of dummy responses."""

    def __init__(self, log_file: str | None = None) -> None:
        """Initializes the LoggerModule.

        Args:
            log_file: Path to the log file that records inputs and outputs (default: ``"app_log.txt"``).
        """
        super().__init__()
        self.log_file = log_file
        self._initialized = False
        self._logger = logging.getLogger("CallableLogger")
        self._logger.setLevel(logging.INFO)
        self.__handler = None
        self._idx = 0
        self._objects = ["telescope", "hammer", "compass", "anchor", "lantern", "keyboard", "cat", "dog", "tiger",
                         "zebra", "batman", "superman", "candy", "table", "chair", "balloon", "kitchen", "sofa", "lamp",
                         "arrow", "green", "red", "blue", "yellow", "magenta", "brown", "pink", "orange", "white",
                         "paris", "rome", "boston", "york", "berlin", "singapore", "taiwan", "japan", "china",
                         "turkey", "italy", "france", "germany", "spain", "madrid", "barcelona", "portugal",
                         "norway", "sweden", "belgium", "romania", "sunny", "snowy", "rainy"]
        random.shuffle(self._objects)

    def __setup_logger(self) -> None:
        """Creates the output directory and Initializes the file-based logger handler."""

        if self.log_file is not None:
            os.makedirs(os.path.abspath(os.path.dirname(self.log_file)), exist_ok=True)
            self.__handler = logging.FileHandler(self.log_file, mode='w')  # 'w' mode overwrites the file
            formatter = logging.Formatter('%(message)s')
            self.__handler.setFormatter(formatter)
            self._logger.addHandler(self.__handler)
        self._initialized = True

    def forward(self, text: str, img: Image.Image | None = None) -> tuple[str | None, Image.Image | None]:
        """Logs the received inputs and returns a cycling dummy text response with no image output.

        Args:
            text: The text input to log.
            img: An optional PIL image input to log (not forwarded to the output).

        Returns:
            A tuple ``(response_text, None)`` where ``response_text`` is the next item from
            the internal word cycle.
        """
        if self.log_file is not None:
            if not self._initialized:
                self.__setup_logger()
            self._logger.info("-------------------------------------------------------------------------------")
            self._logger.info(f"[INPUT] text={text if text is not None else None}, "
                              f"img={img.size if img is not None else None}")
        else:
            log.user(f"[INPUT] text={text if text is not None else None}, "
                     f"img={img.size if img is not None else None}")

        text = f"{self._idx}_{self._objects[self._idx]}"
        self._idx = (self._idx + 1) % len(self._objects)
        img = None

        if self.log_file is not None:
            self._logger.info(f"[OUTPUT] text={text}, img={None}")
            self._logger.info("-------------------------------------------------------------------------------")
            self.__handler.flush()
        else:
            log.user(f"[OUTPUT] text={text}, img={None}")
        return text, img


class ModuleWrapper(torch.nn.Module):
    """Wraps a torch.nn.Module with Stream-based input/output preprocessing and optional learning support."""

    # Backstop on how many times the module can be called again in one turn by the answer gate, whatever the
    # agent's policy says: a runaway override must not turn one step into an endless loop
    UAI_HARD_CAP = 8

    def __init__(self,
                 module: torch.nn.Module | Callable,
                 proc_inputs: list[StreamType] | None = None,
                 proc_outputs: list[StreamType] | None = None,
                 proc_opts: dict | None = None,
                 seed: int = -1,
                 agent: object = None,
                 device=None) -> None:
        """Initializes the ModuleWrapper.

        Args:
            module: The torch.nn.Module or callable object to wrap (default: None).
            proc_inputs: List of Stream objects describing the input types (default: None).
            proc_outputs: List of Stream objects describing the output types (default: None).
            proc_opts: Dictionary with keys ``"optimizer"`` and ``"losses"`` (default: None).
            seed: Random seed to fix before instantiation; negative values are ignored (default: -1).
            agent: The agent who owns this processor.
            device: The device on which to run the processor (default: None).
        """
        super(ModuleWrapper, self).__init__()
        self.device = None
        self.module: Callable | None = None
        self.proc_inputs = None  # The list of Stream objects describing the input types of the module
        self.proc_outputs = None  # The list of Stream objects describing the output types of the module
        self.proc_opts = None
        self.proc_optional_inputs = []
        self.agent = None
        self._has_first = False
        self._has_last = False
        self._sig_defaults = []
        self._last_raw_outputs = None

        # Working
        set_seed(seed)

        self.guess_device(device)
        self.set_module(module)
        self.set_proc_inputs_and_outputs(proc_inputs, proc_outputs)
        self.set_proc_opts(proc_opts)
        self.set_agent(agent)
    
    def guess_device(self, device):
        device_env = os.getenv("PROC_DEVICE", None)
        self.device = torch.device("cpu") if device_env is None else torch.device(device_env)

        # Override if it was given
        if device is not None:
            self.device = torch.device(device) if isinstance(device, str) else device

    def set_agent(self, agent):
        self.agent = agent

    def set_proc_inputs_and_outputs(self, proc_inputs, proc_outputs):
        if proc_inputs is not None:
            self.proc_inputs = proc_inputs
            self.__refresh_proc_optional_inputs()  # rebuilds, truncating or padding as needed
        if proc_outputs is not None:
            self.proc_outputs = proc_outputs

    def set_proc_opts(self, proc_opts):
        if proc_opts is not None:
            self.proc_opts = proc_opts

    def set_module(self, mod: Callable | torch.nn.Module):

        # Case 1: torch.nn.Module / object with .forward
        if hasattr(mod, "forward") and callable(getattr(mod, "forward")):
            sig = inspect.signature(mod.forward)
        elif callable(mod):
            sig = inspect.signature(mod)
        else:
            raise GenException("Invalid processor module, cannot guess the signature of the callable method, if any")

        self.module = mod.to(self.device) if isinstance(mod, torch.nn.Module) else mod
        self._has_first = "first" in sig.parameters  # B8: rename below
        self._has_last = "last" in sig.parameters

        # Cache the signature-derived defaults ONCE, excluding framework-handled kwargs and *args/**kwargs
        # catch-alls. This is the canonical per-positional-data-arg default list of the wrapped module's forward.
        framework_kwargs = {"first", "last"}
        self._sig_defaults: list[dict] = []
        for name, param in sig.parameters.items():
            if name in framework_kwargs:
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if param.default is not inspect.Parameter.empty:
                self._sig_defaults.append({"has_default": True, "default_value": param.default})
            else:
                self._sig_defaults.append({"has_default": False, "default_value": None})

        # Align proc_optional_inputs to current proc_inputs (if already known)
        self.__refresh_proc_optional_inputs()

    def __refresh_proc_optional_inputs(self) -> None:
        """Rebuild `proc_optional_inputs` to be exactly parallel to `proc_inputs` (same length, same ordering).
        Slices the cached signature defaults; pads with no-default if `proc_inputs` is longer than the
        signature has data args."""
        if self.proc_inputs is None:
            # Until proc_inputs is set, expose the full signature defaults so __guess_proc_inputs/_outputs
            # can still inspect them.
            self.proc_optional_inputs = list(self._sig_defaults)
            return
        n = len(self.proc_inputs)
        self.proc_optional_inputs = list(self._sig_defaults[:n])
        while len(self.proc_optional_inputs) < n:
            self.proc_optional_inputs.append({"has_default": False, "default_value": None})

    def forward(self, *args, **kwargs) -> tuple:
        """Preprocesses inputs, runs the wrapped module, and post-processes the outputs.

        Args:
            *args: Positional inputs; each element is preprocessed via the corresponding ``proc_inputs`` Stream.
            **kwargs: Extra keyword arguments forwarded to the module.  The ``first`` and ``last``
                keys are silently removed before the call.

        Returns:
            A tuple of post-processed outputs, one per entry in ``proc_outputs``.
        """
        assert self.module is not None
        assert self.proc_inputs is not None
        assert self.proc_outputs is not None

        # The forward signature expected by who calls this method is:
        # forward(self, *args, first: bool, last: bool, **kwargs)
        # so we have to discard 'first' and 'last' that are not used by an external module not designed for this library
        if 'first' in kwargs and not self._has_first:
            del kwargs['first']
        if 'last' in kwargs and not self._has_last:
            del kwargs['last']
        self._last_raw_outputs = None

        # A model reads the alternative text of a protocol block, never its JSON (INTERACT-PROTOCOL.md,
        # section 4). A module that echoes its input is not a model: what it gives back is wire text and
        # must leave exactly as it is, or a person's own answer would be rewritten on its way out. The
        # stream is untouched either way: what is replaced here is only what the module is handed.
        uai_spec, uai_arg = None, None
        if self.agent is not None and not getattr(self.module, 'UAI_ECHOES_INPUT', False):
            args = list(args)
            for i in range(0, min(len(args), len(self.proc_inputs))):
                if not (isinstance(args[i], str) and has_fence(args[i])):
                    continue
                try:
                    args[i], spec = self.agent.uai_preprocess(args[i])
                    if spec is not None:
                        uai_spec, uai_arg = spec, i
                except Exception as e:
                    log.error(f"Error rendering an interactive message for the module, keeping it as it is: {e}")
        rendered = list(args)

        outputs = self.__run_module(args, kwargs)

        # A block is a message with a contract on its format, and an answer that does not honour it does not
        # leave as it is: the agent decides, and may ask the module again with a corrective prompt (a model
        # that was handed the form in this very call, or that was shown it earlier), or withhold the answer
        # altogether (a person, who is told to write again). The form may come from this call or from an
        # earlier message, which is why this runs for echoing modules too: there it is a no-op on an answer
        # that is already canonical. A failure of the gate itself never costs the answer.
        if self.agent is not None:
            outputs = self.__uai_answer(outputs, rendered, kwargs, uai_spec, uai_arg)

        return tuple(outputs)

    def __run_module(self, args: list, kwargs: dict) -> list:
        """Preprocesses the inputs, calls the module once, and post-processes what it returned."""

        # Preprocessing data
        args = [self.proc_inputs[i].check_and_preprocess(args[i], device=self.device)
                for i in range(0, len(self.proc_inputs))]  # Don't try to build a tuple here, keep a list!

        # Calling the module
        outputs = self.module(*args, **kwargs)  # Forward

        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self._last_raw_outputs = outputs

        # Postprocessing data (this does not affect the raw outputs, that might be used while learning)
        return [self.proc_outputs[i].check_and_postprocess(outputs[i])
                for i in range(0, len(self.proc_outputs))]  # Don't try to build a tuple here, keep a list!

    def __uai_answer(self, outputs: list, rendered: list, kwargs: dict,
                          spec: dict | None, arg_index: int | None) -> list:
        """Runs the agent's policy on the answer to a form, asking the module again as long as it says so.

        Args:
            outputs: The post-processed outputs of the first call.
            rendered: The inputs as the module read them (blocks already replaced by their alternative text).
            kwargs: The keyword arguments of the module call.
            spec: The form handed to the module in this call, if any.
            arg_index: Which input carried it.

        Returns:
            The outputs that should travel.

        Raises:
            AnswerWithheld: When the agent decided that nothing should travel.
        """
        out_index = next((i for i in range(0, len(outputs)) if isinstance(outputs[i], str)), None)
        if out_index is None:
            return outputs
        asked_now = spec is not None
        model_view = rendered[arg_index] if arg_index is not None else None
        outcome = None
        try:
            if spec is None:
                spec = self.agent.uai_pending_form()
            if spec is None:
                return outputs
            attempt = 0
            while True:
                outcome = self.agent.uai_postprocess(outputs[out_index], spec, attempt=attempt,
                                                          asked_now=asked_now, model_view=model_view)
                if outcome.retry is None or attempt >= self.UAI_HARD_CAP:
                    break

                # The corrective prompt takes the place of the message that carried the form (or of the first
                # text input, for a form remembered from an earlier message) and goes through the very same
                # path the first input took
                if arg_index is None:
                    arg_index = next((i for i in range(0, min(len(rendered), len(self.proc_inputs)))
                                      if self.proc_inputs[i].is_text()), None)
                    if arg_index is None:
                        break
                rendered[arg_index] = outcome.retry
                outputs = self.__run_module(list(rendered), kwargs)
                attempt += 1
        except Exception as e:
            log.error(f"Error deciding about the answer to an interactive message, keeping it: {e}")
            return outputs
        if outcome is None:
            return outputs
        if outcome.withhold:
            raise AnswerWithheld(f"The answer to form '{spec.get('id')}' was withheld")
        if outcome.text is not None:
            outputs[out_index] = outcome.text
        return outputs

    def learn_backward(self, targets: list | None = None) -> list:
        """Runs a supervised or unsupervised backward pass and optimizer step.

        Args:
            targets: Optional list of target tensors, one per processor output.  Individual
                entries may be ``None`` (treated as unsupervised for that slot).

        Returns:
            A list of per-output loss values if a backward pass was performed.
        """
        if (self.proc_opts is None or len(self.proc_opts) == 0 or
                ('losses' not in self.proc_opts and 'optimizer' not in self.proc_opts) or self.proc_outputs is None):
            return []

        loss_functions: list = self.proc_opts['losses']
        optimizer: torch.optim.optimizer.Optimizer = self.proc_opts['optimizer']

        # Evaluating loss function(s), one for each processor output slot (they are set to 0. if no targets are there)
        if targets is not None and len(targets) > 0 and any(x is not None for x in targets):

            # Preprocessing targets
            targets = [self.proc_outputs[i].check_and_preprocess(targets[i], device=self.device,
                                                                 allow_class_ids=True, targets=True)
                       for i in range(0, len(self.proc_outputs))]

            # Supervised or partly supervised learning
            loss_values = [loss_fcn(self._last_raw_outputs[i], targets[i]) if targets[i] is not None else
                           torch.tensor(0., device=self.device)
                           for i, loss_fcn in enumerate(loss_functions)]
            loss = torch.stack(loss_values).sum()  # Sum of losses
        else:

            # Unsupervised learning
            loss_values = [loss_fcn(self._last_raw_outputs[i]) for i, loss_fcn in enumerate(loss_functions)]
            loss = torch.stack(loss_values).sum()  # Sum of losses

        # Learning step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Teaching (for autoregressive models, expected to have attribute "y")
        if targets is not None and hasattr(self.module, 'y'):
            self.module.y = targets[0]

        return loss_values


class AgentProcessorChecker:
    """Validates and normalizes the processor-related attributes of an agent-like container object."""

    def __init__(self, agent: object) -> None:
        """Validates and normalizes processor attributes on the given container.

        Checks that ``processor_container`` exposes the required processor attributes, performs
        type validation, auto-wraps plain ``torch.nn.Module`` objects in ``ModuleWrapper``,
        and infers missing ``proc_inputs``, ``proc_outputs``, and ``proc_opts`` heuristically.
        The validated and completed values are written back onto ``processor_container``.

        Args:
            agent: Any object that exposes ``proc``, ``proc_inputs``,
                ``proc_outputs``, ``proc_opts``, and ``proc_optional_inputs`` attributes.
        """

        # Right now, the processor has its own attributes, but they are also directly referenced from the agent object.
        # In the future, we might drop the ones on the agent side.
        assert hasattr(agent, 'proc'), "Invalid processor container (agent) object"
        assert hasattr(agent, 'proc_inputs'), "Invalid processor container (agent) object"
        assert hasattr(agent, 'proc_outputs'), "Invalid processor container (agent) object"
        assert hasattr(agent, 'proc_opts'), "Invalid processor container (agent) object"
        assert hasattr(agent, 'proc_optional_inputs'), "Invalid processor container (agent) object"

        # Getting processor-related info from the main object which collects processor and its properties
        proc: torch.nn.Module | ModuleWrapper | Callable | None = agent.proc
        proc_inputs: list[StreamType] | list[str] | None = agent.proc_inputs
        proc_outputs: list[StreamType] | list[str] | None = agent.proc_outputs
        proc_opts: dict | None = agent.proc_opts

        # The agent itself could be a processor
        if proc is None and callable(agent):
            proc = agent

        if not (proc is None or isinstance_fcn(proc, torch.nn.Module) or isinstance_fcn(proc, ModuleWrapper) or
                callable(proc) or (isinstance(proc, str) and proc == "human")):
            raise GenException("Processor (proc) must be either None or a torch.nn.Module, "
                               "or a ModuleWrapper, or a callable object")
        if not ((proc_inputs is None or (
                isinstance_fcn(proc_inputs, list) and (len(proc_inputs) == 0 or
                                                       (len(proc_inputs) > 0 and
                                                        isinstance_fcn(proc_inputs[0],
                                                                       (StreamType, str))))))):
            raise GenException("Invalid proc_inputs: it must be None or a list of StreamType/str")
        if not ((proc_outputs is None or (
                isinstance_fcn(proc_outputs, list) and (len(proc_outputs) == 0 or
                                                        (len(proc_outputs) > 0 and
                                                         isinstance_fcn(proc_outputs[0],
                                                                        (StreamType, str))))))):
            raise GenException("Invalid proc_outputs: it must be None or a list of StreamType/str")
        if not (proc_opts is None or isinstance_fcn(proc_opts, dict)):
            raise GenException("Invalid proc_opts: it must be None or a dictionary")

        # From strings to StreamType (proc_inputs)
        if proc_inputs is not None:
            proc_inputs_copy = proc_inputs.copy()
            for i, p in enumerate(proc_inputs_copy):
                if isinstance(p, str):
                    try:
                        proc_inputs_copy[i] = StreamType(data_type=p, pubsub=False, private_only=False)
                    except Exception as e:
                        raise GenException(f"Invalid stream type {p}: {e}")
            proc_inputs = proc_inputs_copy

        # From strings to StreamType (proc_outputs)
        if proc_outputs is not None:
            proc_outputs_copy = proc_outputs.copy()
            for i, p in enumerate(proc_outputs_copy):
                if isinstance(p, str):
                    try:
                        proc_outputs_copy[i] = StreamType(data_type=p, pubsub=False, private_only=False)
                    except Exception as e:
                        raise GenException(f"Invalid stream type {p}: {e}")
            proc_outputs = proc_outputs_copy

        # From None to empty dictionary (proc_opts)
        if proc_opts is None:
            proc_opts = {}

        # Type checking
        if proc_inputs is not None:
            proc_inputs: list[StreamType]
        if proc_outputs is not None:
            proc_outputs: list[StreamType]

        # Dummy processor (if no processor was provided)
        if proc is None:
            if proc_inputs is None:
                proc_inputs: list[StreamType] = [StreamType(data_type="all", pubsub=False, private_only=False)]
            if proc_outputs is None:
                proc_outputs: list[StreamType] = [StreamType(data_type="all", pubsub=False, private_only=False)]
            assert proc_outputs is not None
            proc_opts = {'optimizer': None, 'losses': [None] * len(proc_outputs)}
            proc = ModuleWrapper(module=MultiIdentity(), proc_inputs=proc_inputs,
                                 proc_outputs=proc_outputs, proc_opts=proc_opts,
                                 agent=agent, device=torch.device("cpu"))
        else:

            if isinstance(proc, str) and proc.lower().strip() == "human":
                proc_inputs: list[StreamType] = [StreamType(data_type="text", pubsub=False, private_only=False),
                                                 StreamType(data_type="img", pubsub=False, private_only=False)]
                proc_outputs: list[StreamType] = [StreamType(data_type="text", pubsub=False, private_only=False),
                                                  StreamType(data_type="img", pubsub=False, private_only=False)]
                proc = ModuleWrapper(module=HumanModule(), proc_inputs=proc_inputs,
                                     proc_outputs=proc_outputs, agent=agent, device=torch.device("cpu"))

            # Creating a ModuleWrapper with the given processor
            elif not isinstance(proc, ModuleWrapper):
                proc: torch.Module | Callable
                proc = ModuleWrapper(module=proc, proc_opts=proc_opts, proc_inputs=proc_inputs,
                                     proc_outputs=proc_outputs, agent=agent, device=torch.device("cpu"))
            else:

                # If it is already a ModuleWrapper, we force the reference to the agent and the references to the
                # types of inputs and outputs: actually, the inputs/outpus/opts might have been already defined by the
                # wrapper creator, but we give priority to the user options
                proc.set_proc_inputs_and_outputs(proc_inputs, proc_outputs)
                proc.set_proc_opts(proc_opts)
                proc.set_agent(agent)

        # From here after, the proc is not None, and it is a ModuleWrapper.
        # However, proc_inputs, proc_outputs, and proc_opts might still not be specified
        assert proc is not None
        assert isinstance(proc, ModuleWrapper)

        # Guessing inputs, fixing attributes
        AgentProcessorChecker.__guess_proc_inputs(proc)
        if proc.proc_inputs is None:
            raise GenException("The inputs of the processor were not described and cannot be automatically guessed")

        # Fixing naming conventions
        AgentProcessorChecker.__fix_proc_inputs(proc)

        # Guessing outputs, fixing attributes
        AgentProcessorChecker.__guess_proc_outputs(proc)
        if proc.proc_outputs is None:
            raise GenException("The outputs of the processor were not described and cannot be automatically guessed")

        # Fixing naming conventions (should not be needed)
        AgentProcessorChecker.__fix_proc_outputs(proc)

        # Guessing optimization-related options and stuff, fixing attributes
        AgentProcessorChecker.__guess_proc_opts(proc)

        # Fixing attribute-presence conventions
        AgentProcessorChecker.__fix_proc_opts(proc)

        # Ensuring all is OK
        assert proc.proc_inputs is not None
        assert proc.proc_outputs is not None
        assert proc.proc_opts is not None
        assert proc.proc_optional_inputs is not None

        # Updating processor container object (agent)
        agent.proc = proc
        agent.proc_inputs = proc.proc_inputs
        agent.proc_outputs = proc.proc_outputs
        agent.proc_opts = proc.proc_opts
        agent.proc_optional_inputs = proc.proc_optional_inputs

    @staticmethod
    def __guess_proc_inputs(proc: ModuleWrapper):
        """Heuristically infers ``proc_inputs`` from the first layer of the wrapped module."""
        assert proc is not None
        assert proc.module is not None
        if proc.proc_inputs is not None:
            return

        if not isinstance(proc.module, torch.nn.Module):
            return

        first_layer = None

        # Traverse modules to find the first real layer (skip containers like Sequential)
        for layer in proc.module.modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.Conv1d, torch.nn.Embedding)):
                first_layer = layer
                break

        if first_layer is None:
            raise ValueError("Cannot automatically guess the shape of the input data, "
                             "please explicitly provide it (proc_input)")

        # Infer input properties
        data_desc = "automatically guessed"
        tensor_shape = None
        tensor_labels = None
        tensor_dtype = None
        stream_to_proc_transforms = None
        proc_to_stream_transforms = None

        import copy as _copy

        def probe(_proc: ModuleWrapper, call: Callable[[], object]) -> bool:
            """Run `call()` against the wrapped module and return True iff it didn't raise.
            Snapshots the module's state_dict around the call so probes don't corrupt stateful modules
            (RNN/CTE/CTB hidden state, etc.)."""
            snap = None
            if isinstance(_proc.module, torch.nn.Module):
                try:
                    snap = _copy.deepcopy(_proc.module.state_dict())
                except Exception:
                    snap = None
            try:
                call()
                return True
            except Exception:
                return False
            finally:
                if snap is not None:
                    try:
                        assert isinstance(_proc.module, torch.nn.Module)
                        _proc.module.load_state_dict(snap)
                    except Exception:
                        pass

        if isinstance(first_layer, torch.nn.Conv2d):

            if first_layer.in_channels == 3 or first_layer.in_channels == 1:
                data_type = "img"

                # Creating dummy PIL images
                rgb_input_img = Image.new('RGB', (224, 224))
                pixels = rgb_input_img.load()
                assert pixels is not None
                for x in range(28):
                    for y in range(28):
                        pixels[x, y] = (random.randint(0, 255),
                                        random.randint(0, 255),
                                        random.randint(0, 255))
                gray_input_img = rgb_input_img.convert('L')

                # Checking if the model supports PIL images as input
                can_handle_rgb_img = probe(proc,
                                           lambda: proc.module(rgb_input_img)
                                           if proc.module is not None else None)

                can_handle_gray_img = probe(proc,
                                            lambda: proc.module(gray_input_img)
                                            if proc.module is not None else None)

                if can_handle_gray_img and can_handle_rgb_img:
                    stream_to_proc_transforms = None
                elif can_handle_rgb_img:
                    stream_to_proc_transforms = transforms.Grayscale(num_output_channels=3)
                elif can_handle_gray_img:
                    stream_to_proc_transforms = transforms.Grayscale()
                else:
                    if first_layer.in_channels == 1:
                        stream_to_proc_transforms = transforms_factory("gray-no_norm")
                    else:
                        stream_to_proc_transforms = transforms_factory("rgb-no_norm")
            else:

                # If the number of input channels is not 1 and not 3...
                data_type = "tensor"
                tensor_shape = (first_layer.in_channels, None, None)
                tensor_dtype = torch.float32

        elif isinstance(first_layer, torch.nn.Conv1d):
            data_type = "tensor"
            tensor_shape = (first_layer.in_channels, None)
            tensor_dtype = torch.float32
        elif isinstance(first_layer, torch.nn.Linear):
            data_type = "tensor"
            tensor_dtype = torch.float32
            tensor_shape = (first_layer.in_features,)
        elif isinstance(first_layer, torch.nn.Embedding):

            can_handle_text = probe(proc,
                                    lambda: proc.module("testing if tokenizer is present")
                                    if proc.module is not None else None)
            if can_handle_text:
                can_handle_more_than_one_token = True
            else:
                device = torch.device("cpu")
                for param in proc.module.parameters():
                    device = param.device
                    break
                input_tokens = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=device)
                can_handle_more_than_one_token = probe(proc,
                                                       lambda: proc.module(input_tokens)
                                                       if proc.module is not None else None)

            if can_handle_text:
                data_type = "text"
                stream_to_proc_transforms = None
            else:
                data_type = "tensor"
                if can_handle_more_than_one_token:
                    tensor_shape = (None,)
                else:
                    tensor_shape = (1,)
                tensor_dtype = torch.long
                tensor_labels = ["token" + str(i) for i in range(0, first_layer.num_embeddings)]
        else:
            raise ValueError("Cannot automatically guess the shape of the input data, "
                             "please explicitly provide it (proc_input)")

        # Setting the input attribute
        proc.set_proc_inputs_and_outputs([StreamType(name="proc_input_0",
                                                     data_type=data_type,
                                                     data_desc=data_desc,
                                                     tensor_shape=tensor_shape,
                                                     tensor_labels=tensor_labels,
                                                     tensor_dtype=tensor_dtype,
                                                     stream_to_proc_transforms=stream_to_proc_transforms,
                                                     proc_to_stream_transforms=proc_to_stream_transforms,
                                                     pubsub=False,
                                                     private_only=False)], None)

    @staticmethod
    def __fix_proc_inputs(proc: ModuleWrapper):
        # Fixing naming conventions
        for j in range(len(proc.proc_inputs)):
            if proc.proc_inputs[j].get_name() == "unk":
                proc.proc_inputs[j].set_name("proc_input_" + str(j))

    @staticmethod
    def __guess_proc_outputs(proc: ModuleWrapper):
        """Heuristically infers ``proc_outputs`` by running a dummy forward pass."""
        if proc.proc_outputs is not None:
            return

        device = proc.device
        inputs = []

        assert proc.proc_inputs is not None
        for i, proc_input in enumerate(proc.proc_inputs):
            if proc_input.is_tensor():
                inputs.append(proc_input.check_and_preprocess(
                    torch.randn([1] + list(proc_input.tensor_shape),  # Adding batch size here
                                dtype=proc_input.tensor_dtype).to(device)))
            elif proc_input.is_img():
                rgb_input_img = Image.new('RGB', (224, 224))
                pixels = rgb_input_img.load()
                assert pixels is not None
                for x in range(224):
                    for y in range(224):
                        pixels[x, y] = (random.randint(0, 255),
                                        random.randint(0, 255),
                                        random.randint(0, 255))
                inputs.append(proc_input.check_and_preprocess(rgb_input_img))
            elif proc_input.is_text():
                inputs.append(proc_input.check_and_preprocess("test text as input"))

        # Forward
        with torch.no_grad():
            outputs = proc(*inputs)
        if not isinstance(outputs, tuple | list):
            outputs = [outputs]
        if isinstance(outputs, tuple):
            outputs = list(outputs)

        # This will be filled below
        proc_outputs = []
        assert proc_outputs is not None

        for j, output in enumerate(outputs):

            # Infer output properties
            data_desc = "automatically guessed"
            tensor_shape = None
            tensor_labels = None
            tensor_dtype = None
            stream_to_proc_transforms = None
            proc_to_stream_transforms = None

            if isinstance(output, Image.Image):  # PIL Image
                data_type = "img"
            elif isinstance(output, torch.Tensor):  # Tensor
                output_shape = list(output.shape[1:])  # Removing batch size here
                if len(output_shape) == 3 and (output_shape[0] == 3 or output_shape[0] == 1):
                    data_type = "img"
                    if output_shape[0] == 3:
                        proc_to_stream_transforms = transforms_factory("rgb", return_inverse=True)
                    else:
                        proc_to_stream_transforms = transforms_factory("gray", return_inverse=True)
                else:
                    data_type = "tensor"
                    tensor_dtype = str(output.dtype)
                    tensor_shape = output_shape
                    tensor_labels = None
            elif isinstance(output, str):
                data_type = "text"
            else:
                raise ValueError(f"Unsupported output type {type(output)}")

            # Setting the output attribute
            proc_outputs.append(StreamType(name="proc_output_" + str(j),
                                           data_type=data_type,
                                           data_desc=data_desc,
                                           tensor_shape=tensor_shape,
                                           tensor_labels=tensor_labels,
                                           tensor_dtype=tensor_dtype,
                                           stream_to_proc_transforms=stream_to_proc_transforms,
                                           proc_to_stream_transforms=proc_to_stream_transforms,
                                           pubsub=False,
                                           private_only=False))
        proc.set_proc_inputs_and_outputs(None, proc_outputs)

    @staticmethod
    def __fix_proc_outputs(proc: ModuleWrapper):
        # Fixing naming conventions
        for j in range(len(proc.proc_outputs)):
            if proc.proc_outputs[j].get_name() == "unk":
                proc.proc_outputs[j].set_name("proc_output_" + str(j))

    @staticmethod
    def __guess_proc_opts(proc: ModuleWrapper) -> None:
        """Fills in missing optimizer and loss entries in ``proc_opts``."""
        assert proc.proc_outputs is not None and proc is not None
        proc_opts = {"optimizer": None,
                     "losses": [None] * len(proc.proc_outputs)}
        if proc.proc_opts is not None and len(proc.proc_opts) > 0:
            proc_opts = dict(proc.proc_opts)  # shallow copy — never mutate caller's dict
            if "optimizer" not in proc_opts:
                proc_opts["optimizer"] = None
            if "losses" not in proc_opts:
                proc_opts["losses"] = [None] * len(proc.proc_outputs)
        proc.set_proc_opts(proc_opts)

    @staticmethod
    def __fix_proc_opts(proc: ModuleWrapper) -> None:
        """normalizes ``proc_opts`` to always contain the canonical ``"optimizer"`` and ``"losses"`` keys."""
        opts = {}
        found_optimizer = False
        found_loss = False
        cannot_fix = False

        assert proc.proc_opts is not None
        if "optimizer" in proc.proc_opts:
            found_optimizer = True
        if "losses" in proc.proc_opts:
            found_loss = True

        if not found_loss:
            assert proc.proc_outputs is not None
            opts['losses'] = [torch.nn.functional.mse_loss] * len(proc.proc_outputs)

        assert proc.proc_opts is not None
        for k, v in proc.proc_opts.items():
            if isinstance(v, torch.optim.Optimizer):
                if k == "optimizer":
                    opts["optimizer"] = v
                    continue
                else:
                    if not found_optimizer:
                        opts["optimizer"] = v
                        found_optimizer = True
                    else:
                        cannot_fix = True
                        break
            elif k == "losses" and (isinstance(v, list) or isinstance(v, tuple)):
                opts["losses"] = v
                continue
            elif (v == torch.nn.functional.mse_loss or isinstance(v, torch.nn.MSELoss)
                  or v == torch.nn.functional.binary_cross_entropy or isinstance(v, torch.nn.BCELoss)
                  or isinstance(v, torch.nn.CrossEntropyLoss) or v == torch.nn.functional.cross_entropy):
                if not found_loss:
                    opts["losses"] = [v]
                    found_loss = True
                else:
                    cannot_fix = True
                    break
            else:
                opts[k] = v

        if not found_optimizer:
            if 'lr' in opts:
                assert proc.module is not None and isinstance(proc.module, torch.nn.Module)
                opts['optimizer'] = torch.optim.SGD(proc.module.parameters(), lr=opts['lr'])

        assert not cannot_fix, \
            "About proc_opts: cannot find required keys ('optimizer', 'losses') and/or cannot automatically guess them"

        # Removing batch dim from targets in case of cross-entropy
        fixed_list = []
        for _loss_fcn in opts['losses']:
            if _loss_fcn == torch.nn.functional.cross_entropy or isinstance(_loss_fcn, torch.nn.CrossEntropyLoss):
                fixed_list.append(target_shape_fixed_cross_entropy)
            else:
                fixed_list.append(_loss_fcn)
        opts['losses'] = fixed_list

        # Updating
        proc.set_proc_opts(opts)


# Process-wide Featherless clients, one per API key, created once on first use (see _get_featherless_client). Reused
# for every call so HTTP keep-alive avoids a fresh TLS handshake per query; this does NOT change how many requests hit
# Featherless in parallel, which is governed solely by the scheduler's unit budget.
_featherless_clients: dict = {}


def _get_featherless_client(api_key: str | None = None):
    """Return the shared Featherless ``AsyncOpenAI`` client for the given API key, creating it lazily on first use.

    It must be created inside the running event loop (the server's), so it binds to the right loop. Idle keep-alive
    connections expire on their own; the client object stays valid for the whole process lifetime and transparently
    reopens a connection on the next call.

    Args:
        api_key: The Featherless API key to authenticate with (None falls back to FEATHERLESS_API_KEY).

    Returns:
        The process-wide ``AsyncOpenAI`` client instance associated to the key.
    """
    key = api_key or os.environ.get("FEATHERLESS_API_KEY", "KEY")
    if key == "KEY":
        log.critical("No API key: it was not in the request and FEATHERLESS_API_KEY was not set!")
    if key not in _featherless_clients:
        from openai import AsyncOpenAI
        _featherless_clients[key] = AsyncOpenAI(base_url="https://api.featherless.ai/v1", api_key=key)
    return _featherless_clients[key]


# Sampling parameters that the OpenAI chat-completions API accepts as top-level fields. Everything else in a sampler
# dict (e.g. top_k, repetition_penalty, min_p) is a vLLM/Featherless-only extension and must be forwarded via
# extra_body, otherwise the OpenAI SDK rejects it as an unexpected argument.
OPENAI_NATIVE_SAMPLER_PARAMS: frozenset[str] = frozenset({
    "max_tokens", "temperature", "top_p", "frequency_penalty", "presence_penalty",
    "stop", "seed", "n", "logit_bias", "logprobs", "top_logprobs", "response_format",
})


async def featherless_call(sys_prompt: str, prompt: str, sampler: dict | None = None,
                           model: str | None = None, api_key: str | None = None) -> str | None:
    """Perform the actual call to the Featherless API and return the generated text.

    Args:
        sys_prompt: The system prompt, if any (can be empty).
        prompt: The prompt string to send to the model.
        sampler: Sampling parameters to send (e.g. ``temperature``, ``max_tokens``, ``top_p``, ``top_k``,
            ``frequency_penalty``, ``presence_penalty``, ``repetition_penalty``, ``min_p``). OpenAI-native keys are
            passed as top-level fields; any other key is forwarded via ``extra_body``. None/empty uses API defaults.
        model: The model identifier to use; if None, falls back to the MODEL_ID environment variable.
        api_key: The Featherless API key to authenticate with (None falls back to FEATHERLESS_API_KEY).

    Returns:
        The text generated by the model.
    """
    client = _get_featherless_client(api_key)
    model = model or os.environ.get("MODEL_ID", "MOD")

    if model == "MOD":
        log.critical("Calling Featherless API: the model argument was not set; "
                     "the MODEL_ID environment variable was not set!")

    # Split the requested sampler params: OpenAI-native ones go top-level, the rest ride along in extra_body
    sampler = sampler or {}
    extra_body = {k: v for k, v in sampler.items() if k not in OPENAI_NATIVE_SAMPLER_PARAMS}
    native = {k: v for k, v in sampler.items() if k in OPENAI_NATIVE_SAMPLER_PARAMS}

    def _account_log(row_to_add: str) -> None:
        """Append a row to the per-account usage log, if the flag file enables it (crash-durable append)."""
        flag_file = os.path.expanduser("~/DO_LOG_FEATHERLESS_AI")
        if os.path.exists(flag_file):
            file_path = os.path.join(os.path.expanduser("~"), f"featherless_ai_account_{(api_key or 'env')[-6:]}.log")
            with open(file_path, "a") as f:
                f.write(row_to_add + " \n")
                f.flush()  # Flushes Python's internal buffer to OS buffer
                os.fsync(f.fileno())  # Forces OS buffer to write to physical storage

    # Retry with exponential backoff, keeping the last error so a final failure reports its cause
    last_err = None

    for attempt in range(3):
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        t0 = time.perf_counter()
        try:
            # Always use chat completions; include the system message only when a system prompt is given
            msgs = [{"role": "system", "content": sys_prompt}] if sys_prompt else []
            msgs.append({"role": "user", "content": prompt})

            # The dict is typed dict[str, object] so PyCharm accepts the mixed value types (str / list / int / float)
            kwargs: dict[str, object] = {"model": model, "messages": msgs, **native}
            if extra_body:
                kwargs["extra_body"] = extra_body

            resp = await client.chat.completions.create(**kwargs)  # type: ignore
            elapsed = time.perf_counter() - t0

            # Diagnostics: wall time, throughput, completion length + stop reason. A large completion_tokens count
            # suggests thinking is still on; a low tokens/sec on a warm model points at a slow/cold API replica.
            usage = getattr(resp, "usage", None)
            c_toks = getattr(usage, "completion_tokens", 0)
            finish_reason = resp.choices[0].finish_reason if resp.choices else "n/a"
            tps = f"{c_toks / elapsed:.1f}" if (c_toks and elapsed > 0) else "n/a"

            _account_log(f"{current_time} model={model} finish_reason={finish_reason} "
                         f"completion_tokens={c_toks if c_toks is not None else 'n/a'} "
                         f"elapsed={elapsed:.2f}s tokens/sec={tps}")

            # Return the first choice that carries non-empty text (content may be None)
            for choice in resp.choices:
                content = choice.message.content
                if content is not None and content.strip():
                    return content

            # Empty/blank content: treat it as a transient failure so the loop retries instead of returning ""
            finish = resp.choices[0].finish_reason if resp.choices else "n/a"
            last_err = RuntimeError(f"Empty content from model (finish_reason={finish})")
        except Exception as e:
            last_err = e
            elapsed = time.perf_counter() - t0
            _account_log(f"{current_time} model={model} FAILED attempt={attempt + 1} "
                         f"elapsed={elapsed:.2f}s error={e!r}")

        # Retry on either an exception or empty content
        log.error(f"Empty response or exception! (prompt={prompt}, attempt={attempt + 1}, last_err={last_err!r}), "
                  f"waiting {2 ** attempt} seconds before trying again...")
        await asyncio.sleep(2 ** attempt)

    log.critical(f"Featherless AI call failed after retries: {last_err!r}")  # This will raise an exception
    return ""  # Never reached


class APIGatewayServer:
    """The shared gateway server: owns the connection to Featherless and runs the Scheduler.

    The server speaks a newline-delimited JSON protocol over TCP (localhost only):
        client -> server: {"op": "hello"}                                              (on the persistent connection)
        client -> server: {"op": "generate", "process_id": .., "prompt": .., "cost": .., "model": ..,
                           "api_key": .., "key_units": ..}
        server -> client: {"ok": true, "result": ..} | {"ok": false, "error": ..}

    Each API key is a distinct Featherless account with its own total unit budget ("key_units"): requests are routed
    to a per-key scheduler (own queues, own budget, own dispatcher), so accounts never stall each other.

    Each client keeps ONE persistent "registration" connection open for its whole life; that open socket is its
    liveness token. When a client dies for any reason, the OS closes the socket and the server notices the drop. The
    server tracks the set of live registrations and, once it empties, shuts itself down.
    """
    HOST: str = os.environ.get("GW_HOST", "127.0.0.1")
    PORT: int = int(os.environ.get("GW_PORT", "11321"))
    TOTAL_UNITS = int(os.environ.get("GW_UNITS", "8"))
    VALID_COSTS = tuple(int(x) for x in os.environ.get("GW_VALID_COSTS", "1,2,4").split(","))
    LOCKFILE: str = os.environ.get("GW_LOCKFILE", os.path.join(tempfile.gettempdir(), "llm_api_gateway.lock"))

    def __init__(self, call: Callable = featherless_call):
        """Create a Server.

        Args:
            call: The asynchronous function performing the actual model call (Default: featherless_call).
        """
        self.scheds: dict[str, APIGatewayScheduler] = {}  # API key -> scheduler (own queues/budget/dispatcher)
        self.call: Callable = call
        self.live: set = set()  # Set of registration writers (one per live client)
        self.shutdown: asyncio.Event = asyncio.Event()

    def _sched_for(self, api_key: str, key_units: int) -> 'APIGatewayScheduler':
        """Return the scheduler for the given API key, creating it (and its dispatcher task) on first use.

        Each API key is a distinct Featherless account with its own unit budget, so each key gets its own scheduler
        (own queues, own budget, own dispatcher coroutine): a saturated account never stalls the others. The budget
        is sized by the FIRST request that mentions the key; later requests declaring a different total are served
        under the original budget (a mismatch is logged and ignored).

        Args:
            api_key: The Featherless API key identifying the account.
            key_units: The total unit budget supported by that key.

        Returns:
            The ``APIGatewayScheduler`` associated to the key.
        """
        if api_key not in self.scheds:
            sched = APIGatewayScheduler(total_units=key_units)
            self.scheds[api_key] = sched
            asyncio.create_task(sched.dispatcher(functools.partial(self.call, api_key=api_key)))
        else:
            sched = self.scheds[api_key]
            if sched.total_units != key_units:
                log.error(f"Key ...{api_key[-4:]} was registered with {sched.total_units} total units, "
                          f"ignoring the new value {key_units}")
        return self.scheds[api_key]

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Serve a single client connection until it closes.

        A connection that sends ``{"op": "hello"}`` is treated as that client's liveness registration; a connection
        that sends ``{"op": "generate", ...}`` has its query scheduled and the result written back.

        Args:
            reader: The stream reader for the client connection.
            writer: The stream writer for the client connection.
        """
        registered = False
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break  # Connection closed by the client (or it died)
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                op = msg.get("op")
                if op == "hello":

                    # This connection is the client's liveness registration
                    registered = True
                    self.live.add(writer)
                elif op == "generate":
                    pid = msg["process_id"]
                    sys_prompt = msg.get("sys_prompt", "")
                    prompt = msg["prompt"]
                    cost = int(msg.get("cost", 1))
                    model = msg.get("model")
                    sampler = msg.get("sampler") or {}
                    api_key = msg.get("api_key") or os.environ.get("FEATHERLESS_API_KEY", "")
                    key_units = int(msg.get("key_units", APIGatewayServer.TOTAL_UNITS))
                    if cost not in APIGatewayServer.VALID_COSTS:
                        out = {"ok": False, "error": f"bad cost {cost}"}
                    elif not api_key:
                        out = {"ok": False, "error": "no API key: not in the request, and no FEATHERLESS_API_KEY"}
                    else:
                        try:
                            sched = self._sched_for(api_key, key_units)
                            result = await sched.submit(pid, sys_prompt, prompt, cost, model, sampler)
                            out = {"ok": True, "result": result}
                        except Exception as e:
                            out = {"ok": False, "error": str(e)}
                    writer.write((json.dumps(out) + "\n").encode())
                    await writer.drain()
        except (ConnectionResetError, asyncio.IncompleteReadError):
            pass
        finally:
            if registered:
                self.live.discard(writer)

                # When the last interested client disconnects, shut down
                if not self.live:
                    self.shutdown.set()
            try:
                writer.close()
            except Exception:
                pass

    async def run(self) -> None:
        """Bind the listening socket, start the dispatcher, and run until shutdown is requested.

        Binding also arbitrates the startup race: if the port is already taken, ``start_server`` raises and this process
        is not the server. A grace period guards against an orphan server whose starter died before registering.
        """

        # Bind first (this is also the startup-race arbiter): if the port is taken, bind() raises and we are NOT
        # the server. Dispatchers are started lazily, one per API key, on the first request mentioning each key
        server = await asyncio.start_server(self.handle, APIGatewayServer.HOST, APIGatewayServer.PORT)

        # Grace period: if no client registers shortly after boot, exit, so a server spawned by a starter that
        # immediately died does not linger
        grace_secs = float(os.environ.get("GW_GRACE", "10"))

        async def grace():
            await asyncio.sleep(grace_secs)
            if not self.live:
                self.shutdown.set()
        asyncio.create_task(grace())

        async with server:
            await self.shutdown.wait()


class APIGatewayScheduler:
    """Strict round-robin scheduler with a shared unit-budget admission, used by the gateway server.

    The scheduler serves one query per process per turn (round-robin), with equal importance regardless of how many
    units a query costs: the cost only affects the budget admission check, never the turn order. Within a process,
    queries are served in submission order (FIFO).

    Two design decisions are worth highlighting:
    - Strict hold, never skip. When it is a process's turn but its next query does not fit the remaining budget, the
        dispatcher STALLS and holds the free units until that query can run; it does not skip ahead to another process.
        This is intentional (chosen over skip-to-fill), accepting the throughput cost.
    - At most one held "head" per process. The dispatcher pops a single query and keeps it in the ``held`` dict instead
        of peeking the private ``asyncio.Queue`` deque (which caused an IndexError race across ``await`` boundaries).
    A single ``asyncio.Condition`` drives every wakeup (new work submitted, or units freed by a finished job).
    """

    def __init__(self, total_units: int = -1):
        """Create a Scheduler.

        Args:
            total_units: The total unit budget this scheduler admits (negative falls back to GW_UNITS) (Default: -1).
        """
        self.total_units: int = total_units if total_units > 0 else APIGatewayServer.TOTAL_UNITS
        self.available: int = self.total_units
        self.cond: asyncio.Condition = asyncio.Condition()
        self.queues: dict[str, asyncio.Queue] = {}  # Process ID -> queue of (prompt, cost, model, fut) - ORDERED
        self.rr: int = 0  # Round-robin cursor into the (insertion-ordered) list of process IDs

    def _queue(self, pid: str) -> asyncio.Queue:
        """Return the per-process queue for the given process ID, creating it on first use.

        Args:
            pid: The process ID whose queue is requested.

        Returns:
            The ``asyncio.Queue`` associated to the process ID.
        """
        if pid not in self.queues:
            self.queues[pid] = asyncio.Queue()
        return self.queues[pid]

    async def submit(self, pid: str, sys_prompt: str, prompt: str, cost: int, model: str | None,
                     sampler: dict | None = None) -> str:
        """Enqueue a query for a process and wait until the dispatcher runs it.

        Args:
            pid: The process ID the query belongs to (round-robin fairness is per process).
            sys_prompt: The system prompt string (can be "").
            prompt: The prompt string to send to the model.
            cost: The query's unit cost (one of VALID_COSTS).
            model: The model identifier to use (None lets the call fall back to its own default).
            sampler: Sampling parameters to forward to the model call (None means use API defaults).

        Returns:
            The model's textual result.
        """
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._queue(pid).put((sys_prompt, prompt, cost, model, sampler, fut))
        async with self.cond:
            self.cond.notify_all()  # Wake the dispatcher: there is new work
        return await fut

    async def _run_job(self, sys_prompt: str, prompt: str,
                       cost: int, model: str | None, sampler: dict | None,
                       fut: asyncio.Future, call: Callable) -> None:
        """Run a single query to completion, then return its units to the budget.

        Args:
            sys_prompt: The system prompt to send to the model (can be empty, "").
            prompt: The prompt string to send to the model.
            cost: The query's unit cost, returned to the budget once the job finishes.
            model: The model identifier to use.
            sampler: Sampling parameters to forward to the model call (None means use API defaults).
            fut: The future awaited by the submitter, resolved with the result or the exception.
            call: The asynchronous function performing the actual model call.
        """
        try:
            result = await call(sys_prompt, prompt, sampler, model)
            fut.set_result(result)
        except Exception as e:
            fut.set_exception(e)
        finally:
            async with self.cond:
                self.available += cost  # Give the units back and wake the dispatcher
                self.cond.notify_all()

    async def dispatcher(self, call: Callable) -> None:
        """Forever serve queued queries in strict round-robin order under the unit budget.

        Args:
            call: The asynchronous function performing the actual model call (passed on to each job).
        """

        # held[pid] holds a (prompt, cost, model, fut) we popped for a process but could not launch yet (its turn came
        # but the budget did not fit). Strict hold keeps it here until it can run; never put it back nor skip past it.
        held: dict = {}
        while True:
            async with self.cond:
                pids = list(self.queues.keys())
                if not pids:
                    await self._wait_locked()
                    continue

                pid = pids[self.rr % len(pids)]
                q = self.queues[pid]

                # Determine this process's current head: either the one we are holding, or the next one off its queue
                if pid in held:
                    sys_prompt, prompt, cost, model, sampler, fut = held[pid]
                elif not q.empty():
                    sys_prompt, prompt, cost, model, sampler, fut = q.get_nowait()
                else:

                    # The current process has nothing: advance the turn. If nobody has work at all, wait; otherwise
                    # keep rotating until we reach a process that does
                    self.rr = (self.rr + 1) % len(pids)
                    if all(self.queues[p].empty() for p in pids) and not held:
                        await self._wait_locked()
                    continue

                if self.available >= cost:
                    self.available -= cost
                    held.pop(pid, None)
                    asyncio.create_task(
                        self._run_job(sys_prompt, prompt, cost, model, sampler, fut, call))
                    self.rr = (self.rr + 1) % len(pids)
                    continue
                else:

                    # Strict hold: it is this process's turn and its head does not fit. Hold the head and wait for units
                    # to free up. Do NOT advance the cursor and do NOT serve anyone else.
                    held[pid] = (sys_prompt, prompt, cost, model, sampler, fut)
                    await self._wait_locked()
                    continue

    async def _wait_locked(self) -> None:
        """Release the condition lock and block until notified (new work or freed units), then reacquire it.

        Must be called while holding ``self.cond``.
        """
        await self.cond.wait()


def serve_api_gateway() -> None:
    """Process entrypoint for the detached gateway server (see the APIGatewayServer class).

    Spawned by the client via
    ``python -c "from unaiverse.modules.utils import serve_api_gateway; serve_api_gateway()"``. We deliberately do
    NOT use ``python -m unaiverse.modules.utils`` here: importing the parent package ``unaiverse.modules`` already
    imports this module, so running it again via ``runpy`` would execute a second copy of it under the name
    ``__main__`` (RuntimeWarning, and two distinct APIGatewayServer classes). Calling this function avoids that.
    """

    async def _main():

        # Construct the server INSIDE the running loop: the scheduler builds an asyncio.Condition, which binds to
        # the current loop at creation. Building it before asyncio.run() would bind it to the wrong loop.
        await APIGatewayServer(call=featherless_call).run()

    try:
        asyncio.run(_main())
    except OSError:
        pass  # Port already bound: another process beat us to it, just exit
