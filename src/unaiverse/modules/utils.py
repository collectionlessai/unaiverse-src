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
import torch
import random
import inspect
import logging
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from unaiverse.streams.dataprops import Stream


class GenException(Exception):
    """Base exception for this application (a simple wrapper around a generic Exception)."""
    pass


def has_human_processor(agent: object) -> bool:
    """Checks whether the given agent has a human processor module.

    Args:
        agent: The agent object to inspect; expected to have a ``proc`` attribute.

    Returns:
        True if the agent's processor module is a HumanModule, False otherwise.
    """
    return agent.proc is not None and isinstance(agent.proc.module, HumanModule)


def transforms_factory(trans_type: str, add_batch_dim: bool = True,
                       return_inverse: bool = False) -> transforms.Compose:
    """Builds and returns a torchvision transform pipeline for the given type.

    Args:
        trans_type: A string identifying the desired transform.  Supported values are
            ``"rgb<N>"`` / ``"gray<N>"`` (resize + center-crop to ``<N>`` pixels, with normalisation),
            ``"rgb-no_norm<N>"`` / ``"gray-no_norm<N>"`` (resize + center-crop, no normalisation),
            ``"rgb"`` / ``"gray"`` (full-image with normalisation),
            ``"rgb-no_norm"`` / ``"gray-no_norm"`` (full-image, no normalisation), and
            ``"gray_mnist"`` (MNIST-specific 28×28 grayscale pipeline).
        add_batch_dim: If True, appends an ``unsqueeze(0)`` step to the forward transform and
            inserts a ``squeeze(0)`` step at the start of the inverse transform (default: True).
        return_inverse: If True, returns the inverse (de-normalisation) transform instead of the
            forward transform (default: False).

    Returns:
        The requested ``transforms.Compose`` pipeline.
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
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),  # Ensure 3 channels
            transforms.Resize(num),
            transforms.CenterCrop(num),
            transforms.ToTensor(),  # Convert PIL to tensor (3, H, W), float [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        inverse_trans = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.],
                                 std=[1. / 0.229, 1. / 0.224, 1. / 0.225]),
            transforms.Lambda(lambda x: x + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)),
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
            transforms.Lambda(lambda x: x + 0.45),
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
        inverse_trans = transforms.ToPILImage()
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
            transforms.Lambda(lambda x: x + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)),
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
            transforms.Lambda(lambda x: x + 0.45),
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
            transforms.Lambda(lambda x: x + 0.1307),
            transforms.ToPILImage()
        ])

    if add_batch_dim:
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
        np.random.seed(0)


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
        Stream(data_type="tensor", tensor_shape=(None,) + u_shape, tensor_dtype=torch.float32,
               pubsub=False, private_only=True),
        Stream(data_type="tensor", tensor_shape=(None, du_dim,), tensor_dtype=torch.float32,
               pubsub=False, private_only=True)
    ]
    proc_outputs = [
        Stream(data_type="tensor", tensor_shape=(None, y_dim), tensor_dtype=torch.float32,
               pubsub=False, private_only=True)
    ]
    return proc_inputs, proc_outputs


def get_proc_inputs_and_proc_outputs_for_image_classification(y_dim: int) -> tuple[list, list]:
    """Creates Stream descriptors for the inputs and output of an image-classification processor.

    Args:
        y_dim: Number of output classes.  Pass ``-1`` to default to 1000 (ImageNet).

    Returns:
        A tuple ``(proc_inputs, proc_outputs)`` where each element is a list of
        ``Stream`` objects describing the processor's I/O signature.
    """
    if y_dim == -1:
        y_dim = 1000  # Assuming ImageNet-trained models
    proc_inputs = [Stream(data_type="img", pubsub=False, private_only=True)]
    proc_outputs = [Stream(data_type="tensor", tensor_shape=(None, y_dim), tensor_dtype=torch.float32,
                           pubsub=False, private_only=True)]
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
    """Evaluates a network's classification error rate on the MNIST test set.

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

    # Checking error rate
    error_rate = 0.
    n = 0
    training_flag_backup = network.training
    network.eval()
    device = next(network.parameters()).device
    for x, y in mnist_test:
        x = x.to(device)
        y = y.to(device)
        o = network(x)
        c = torch.argmax(o, dim=1)
        error_rate += float(torch.sum(c != y).item())
        n += x.shape[0]
    error_rate /= n
    network.training = training_flag_backup

    return error_rate


class MultiIdentity(torch.nn.Module):
    """Identity module that passes one or more inputs through unchanged."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args, **kwargs) -> object | tuple:
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

    def __init__(self) -> None:
        super().__init__()

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

    def __init__(self, log_file: str = "app_log.txt") -> None:
        """Initialises the LoggerModule.

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
        """Creates the output directory and initialises the file-based logger handler."""
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
        if not self._initialized:
            self.__setup_logger()
        self._logger.info("-------------------------------------------------------------------------------")
        self._logger.info(f"[INPUT] text={text if text is not None else None}, "
                          f"img={img.size if img is not None else None}")
        # text = random.choice(objects)
        text = f"{self._idx}_{self._objects[self._idx]}"
        self._idx = (self._idx + 1) % len(self._objects)
        img = None
        self._logger.info(f"[OUTPUT] text={text}, img={None}")
        self._logger.info("-------------------------------------------------------------------------------")
        self.__handler.flush()
        return text, img


class ModuleWrapper(torch.nn.Module):
    """Wraps a torch.nn.Module with Stream-based input/output preprocessing and optional learning support."""

    def __init__(self,
                 module: torch.nn.Module | None = None,
                 proc_inputs: list[Stream] | None = None,
                 proc_outputs: list[Stream] | None = None,
                 proc_opts: dict | None = None,
                 seed: int = -1) -> None:
        """Initialises the ModuleWrapper.

        Args:
            module: The torch.nn.Module to wrap (default: None).
            proc_inputs: List of Stream objects describing the input types (default: None).
            proc_outputs: List of Stream objects describing the output types (default: None).
            proc_opts: Dictionary with keys ``"optimizer"`` and ``"losses"`` (default: None).
            seed: Random seed to fix before instantiation; negative values are ignored (default: -1).
        """
        super(ModuleWrapper, self).__init__()
        self.device = None  # The device which is supposed to host the module
        self.module = None  # The module itself
        self.proc_inputs = proc_inputs  # The list of Stream objects describing the input types of the module
        self.proc_outputs = proc_outputs  # The list of Stream objects describing the output types of the module
        self.proc_opts = proc_opts

        # Working
        set_seed(seed)
        device_env = os.getenv("PROC_DEVICE", None)
        self.device = torch.device("cpu") if device_env is None else torch.device(device_env)
        self.module = module.to(self.device) if module is not None else None
        self.__last_raw_outputs = None

    def forward(self, *args, **kwargs) -> tuple:
        """Preprocesses inputs, runs the wrapped module, and post-processes the outputs.

        Args:
            *args: Positional inputs; each element is preprocessed via the corresponding ``proc_inputs`` Stream.
            **kwargs: Extra keyword arguments forwarded to the module.  The ``first`` and ``last``
                keys are silently removed before the call.

        Returns:
            A tuple of post-processed outputs, one per entry in ``proc_outputs``.
        """

        # The forward signature expected by who calls this method is:
        # forward(self, *args, first: bool, last: bool, **kwargs)
        # so we have to discard 'first' and 'last' that are not used by an external module not designed for this library
        if 'first' in kwargs:
            del kwargs['first']
        if 'last' in kwargs:
            del kwargs['last']
        self.__last_raw_outputs = None

        # Preprocessing data
        args = [self.proc_inputs[i].check_and_preprocess(args[i], device=self.device)
                for i in range(0, len(self.proc_inputs))]  # Don't try to build a tuple here, keep a list!

        # Calling the module
        outputs = self.module.forward(*args, **kwargs)
        self.__last_raw_outputs = outputs

        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        # Postprocessing data (this does not affect the raw outputs, that might be used while learning)
        outputs = [self.proc_outputs[i].check_and_postprocess(outputs[i])
                   for i in range(0, len(self.proc_outputs))]  # Don't try to build a tuple here, keep a list!

        return tuple(outputs)

    def learn_backward(self, targets: list | None = None) -> list | bool:
        """Runs a supervised or unsupervised backward pass and optimiser step.

        Args:
            targets: Optional list of target tensors, one per processor output.  Individual
                entries may be ``None`` (treated as unsupervised for that slot).

        Returns:
            A list of per-output loss values if a backward pass was performed, or ``False``
            if ``proc_opts`` contains no optimiser or losses.
        """
        if (self.proc_opts is None or len(self.proc_opts) == 0 or
                ('losses' not in self.proc_opts and 'optimizer' not in self.proc_opts)):
            return False

        loss_functions: list = self.proc_opts['losses']
        optimizer: torch.optim.optimizer.Optimizer | None = self.proc_opts['optimizer']

        # Evaluating loss function(s), one for each processor output slot (they are set to 0. if no targets are there)
        if targets is not None and len(targets) > 0 and any(x is not None for x in targets):

            # Preprocessing targets
            targets = [self.proc_outputs[i].check_and_preprocess(targets[i], device=self.device,
                                                                 allow_class_ids=True, targets=True)
                       for i in range(0, len(self.proc_outputs))]

            # Supervised or partly supervised learning
            loss_values = [loss_fcn(self.__last_raw_outputs[i], targets[i]) if targets[i] is not None else
                           torch.tensor(0., device=self.device)
                           for i, loss_fcn in enumerate(loss_functions)]
            loss = torch.stack(loss_values).sum()  # Sum of losses
        else:

            # Unsupervised learning
            loss_values = [loss_fcn(self.__last_raw_outputs[i]) for i, loss_fcn in enumerate(loss_functions)]
            loss = torch.stack(loss_values).sum()  # Sum of losses

        # Learning step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Teaching (for autoregressive models, expected to have attribute "y")
        if hasattr(self.module, 'y'):
            self.module.y = targets[0]

        return loss_values


class AgentProcessorChecker:
    """Validates and normalises the processor-related attributes of an agent-like container object."""

    def __init__(self, processor_container: object) -> None:
        """Validates and normalises processor attributes on the given container.

        Checks that ``processor_container`` exposes the required processor attributes, performs
        type validation, auto-wraps plain ``torch.nn.Module`` objects in ``ModuleWrapper``,
        and infers missing ``proc_inputs``, ``proc_outputs``, and ``proc_opts`` heuristically.
        The validated and completed values are written back onto ``processor_container``.

        Args:
            processor_container: Any object that exposes ``proc``, ``proc_inputs``,
                ``proc_outputs``, ``proc_opts``, and ``proc_optional_inputs`` attributes.
        """
        assert hasattr(processor_container, 'proc'), "Invalid processor container object"
        assert hasattr(processor_container, 'proc_inputs'), "Invalid processor container object"
        assert hasattr(processor_container, 'proc_outputs'), "Invalid processor container object"
        assert hasattr(processor_container, 'proc_opts'), "Invalid processor container object"
        assert hasattr(processor_container, 'proc_optional_inputs'), "Invalid processor container object"

        # Getting processor-related info from the main object which collects processor and its properties
        proc: torch.nn.Module = processor_container.proc
        proc_inputs: list[Stream] | None = processor_container.proc_inputs
        proc_outputs: list[Stream] | None = processor_container.proc_outputs
        proc_opts: dict | None = processor_container.proc_opts
        proc_optional_inputs: list | None = processor_container.proc_optional_inputs

        # Auto-wrap
        if 'forward' in vars(processor_container) and isinstance(processor_container, torch.nn.Module):
            proc = processor_container

        if not (proc is None or isinstance(proc, torch.nn.Module)):
            raise GenException("Processor (proc) must be a torch.nn.Module")
        if not ((proc_inputs is None or (
                isinstance_fcn(proc_inputs, list) and (len(proc_inputs) == 0 or
                                                       (len(proc_inputs) > 0 and
                                                        isinstance_fcn(proc_inputs[0],
                                                                       (Stream, str))))))):
            raise GenException("Invalid proc_inputs: it must be None or a list of Stream/str")
        if not ((proc_outputs is None or (
                isinstance_fcn(proc_outputs, list) and (len(proc_outputs) == 0 or
                                                        (len(proc_outputs) > 0 and
                                                         isinstance_fcn(proc_outputs[0],
                                                                        (Stream, str))))))):
            raise GenException("Invalid proc_inputs: it must be None or a list of Stream/str")
        if not (proc_opts is None or isinstance_fcn(proc_opts, dict)):
            raise GenException("Invalid proc_opts: it must be None or a dictionary")

        if proc_inputs is not None:
            proc_inputs_copy = proc_inputs.copy()
            for i, p in enumerate(proc_inputs_copy):
                if isinstance(p, str):
                    try:
                        proc_inputs_copy[i] = Stream(data_type=p, pubsub=False, private_only=False)
                    except Exception as e:
                        raise GenException(f"Invalid Stream type {p}: {e}")
            proc_inputs = proc_inputs_copy

        # Saving as attributes
        self.proc = proc
        self.proc_inputs = proc_inputs
        self.proc_outputs = proc_outputs
        self.proc_opts = proc_opts
        self.proc_optional_inputs = proc_optional_inputs

        # Dummy processor (if no processor was provided)
        if self.proc is None:
            if self.proc_inputs is None:
                self.proc_inputs = [Stream(data_type="all", pubsub=False, private_only=False)]
            if self.proc_outputs is None:
                self.proc_outputs = [Stream(data_type="all", pubsub=False, private_only=False)]
            self.proc_opts = {'optimizer': None, 'losses': [None] * len(self.proc_outputs)}
            self.proc = ModuleWrapper(module=MultiIdentity(), proc_inputs=self.proc_inputs,
                                      proc_outputs=self.proc_outputs, proc_opts=self.proc_opts)
            self.proc.device = torch.device("cpu")
        else:

            # String telling it is a human
            if isinstance(self.proc, str) and self.proc.lower() == "human":
                self.proc_inputs = [Stream(data_type="text", pubsub=False, private_only=False),
                                    Stream(data_type="img", pubsub=False, private_only=False)]
                self.proc_outputs = [Stream(data_type="text", pubsub=False, private_only=False),
                                     Stream(data_type="img", pubsub=False, private_only=False)]
                self.proc = ModuleWrapper(module=HumanModule(), proc_inputs=self.proc_inputs,
                                          proc_outputs=self.proc_outputs)
                self.proc.device = torch.device("cpu")

            # Wrapping to have the basic attributes (device)
            elif not isinstance(self.proc, ModuleWrapper):
                self.proc = ModuleWrapper(module=self.proc, proc_opts=self.proc_opts, proc_inputs=self.proc_inputs,
                                          proc_outputs=self.proc_outputs)
                self.proc.device = torch.device("cpu")

        # Guessing inputs, fixing attributes
        if self.proc_inputs is None:
            self.__guess_proc_inputs()

        for j in range(len(self.proc_inputs)):
            if self.proc_inputs[j].get_name() == "unk":
                self.proc_inputs[j].set_name("proc_input_" + str(j))

        # Guessing outputs, fixing attributes
        if self.proc_outputs is None:
            self.__guess_proc_outputs()

        for j in range(len(self.proc_outputs)):
            if self.proc_outputs[j].get_name() == "unk":
                self.proc_outputs[j].set_name("proc_output_" + str(j))

        # Guessing optimization-related options and stuff, fixing attributes
        if (self.proc_opts is None or len(self.proc_opts) == 0 or
                'optimizer' not in self.proc_opts or 'losses' not in self.proc_opts):
            self.__guess_proc_opts()
        self.__fix_proc_opts()

        # Ensuring all is OK
        if self.proc is not None:
            assert "optimizer" in self.proc_opts, "Missing 'optimizer' key in proc_opts (required)"
            assert "losses" in self.proc_opts, "Missing 'losses' key in proc_opts (required)"

        # Checking inputs with default values
        if self.proc_optional_inputs is None:
            self.__guess_proc_optional_inputs()

        # Updating processor container object
        processor_container.proc = self.proc
        processor_container.proc_inputs = self.proc_inputs
        processor_container.proc_outputs = self.proc_outputs
        processor_container.proc_opts = self.proc_opts
        processor_container.proc_optional_inputs = self.proc_optional_inputs

    def __guess_proc_inputs(self) -> None:
        """Heuristically infers ``proc_inputs`` from the first layer of the wrapped module."""
        if hasattr(self.proc, "proc_inputs"):
            if self.proc.proc_inputs is not None:
                self.proc_inputs = []
                for p in self.proc.proc_inputs:
                    self.proc_inputs.append(p.clone())
            return

        first_layer = None

        # Traverse modules to find the first real layer (skip containers like Sequential)
        for layer in self.proc.modules():
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

        if isinstance(first_layer, torch.nn.Conv2d):

            if first_layer.in_channels == 3 or first_layer.in_channels == 1:
                data_type = "img"

                # Creating dummy PIL images
                rgb_input_img = Image.new('RGB', (224, 224))
                pixels = rgb_input_img.load()
                for x in range(28):
                    for y in range(28):
                        pixels[x, y] = (random.randint(0, 255),
                                        random.randint(0, 255),
                                        random.randint(0, 255))
                gray_input_img = rgb_input_img.convert('L')

                # Checking if the model supports PIL images as input
                # noinspection PyBroadException
                try:
                    _ = self.proc(rgb_input_img)
                    can_handle_rgb_img = True
                except Exception:
                    can_handle_rgb_img = False

                # Noinspection PyBroadException
                try:
                    _ = self.proc(gray_input_img)
                    can_handle_gray_img = True
                except Exception:
                    can_handle_gray_img = False

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

            # Noinspection PyBroadException
            try:
                input_text = "testing if tokenizer is present"
                _ = self.proc(input_text)
                can_handle_text = True
                can_handle_more_than_one_token = True  # Unused
            except Exception:
                can_handle_text = False

                # Noinspection PyBroadException
                try:
                    device = torch.device("cpu")
                    for param in self.proc.parameters():
                        device = param.device
                        break
                    input_tokens = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=device)
                    _ = self.proc(input_tokens)
                    can_handle_more_than_one_token = True
                except Exception:
                    can_handle_more_than_one_token = False

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
        self.proc_inputs = [Stream(name="proc_input_0",
                                   data_type=data_type,
                                   data_desc=data_desc,
                                   tensor_shape=tensor_shape,
                                   tensor_labels=tensor_labels,
                                   tensor_dtype=tensor_dtype,
                                   stream_to_proc_transforms=stream_to_proc_transforms,
                                   proc_to_stream_transforms=proc_to_stream_transforms,
                                   pubsub=False,
                                   private_only=True)]

    def __guess_proc_outputs(self) -> None:
        """Heuristically infers ``proc_outputs`` by running a dummy forward pass."""
        if hasattr(self.proc, "proc_outputs"):
            if self.proc.proc_outputs is not None:
                self.proc_outputs = []
                for p in self.proc.proc_outputs:
                    self.proc_outputs.append(p.clone())
            return

        proc = self.proc
        device = self.proc.device
        inputs = []

        for i, proc_input in enumerate(self.proc_inputs):
            if proc_input.is_tensor():
                inputs.append(proc_input.check_and_preprocess(
                    torch.randn([1] + list(proc_input.tensor_shape),  # Adding batch size here
                                dtype=proc_input.tensor_dtype).to(device)))
            elif proc_input.is_img():
                rgb_input_img = Image.new('RGB', (224, 224))
                pixels = rgb_input_img.load()
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
        self.proc_outputs = []

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
            self.proc_outputs.append(Stream(name="proc_output_" + str(j),
                                            data_type=data_type,
                                            data_desc=data_desc,
                                            tensor_shape=tensor_shape,
                                            tensor_labels=tensor_labels,
                                            tensor_dtype=tensor_dtype,
                                            stream_to_proc_transforms=stream_to_proc_transforms,
                                            proc_to_stream_transforms=proc_to_stream_transforms,
                                            pubsub=False,
                                            private_only=True))

    def __guess_proc_opts(self) -> None:
        """Fills in missing optimiser and loss entries in ``proc_opts``."""
        if self.proc_opts is None:
            if isinstance(self.proc.module, MultiIdentity) or len(list(self.proc.parameters())) == 0:
                self.proc_opts = {"optimizer": None,
                                  "losses": [None] * len(self.proc_outputs)}
            else:
                self.proc_opts = {"optimizer": torch.optim.SGD(self.proc.parameters(), lr=1e-5),
                                  "losses": [torch.nn.functional.mse_loss] * len(self.proc_outputs)}
        else:
            if "optimizer" not in self.proc_opts:
                self.proc_opts["optimizer"] = None
            if "losses" not in self.proc_opts:
                self.proc_opts["losses"] = [None] * len(self.proc_outputs)

    def __fix_proc_opts(self) -> None:
        """Normalises ``proc_opts`` to always contain the canonical ``"optimizer"`` and ``"losses"`` keys."""
        opts = {}
        found_optimizer = False
        found_loss = False
        cannot_fix = False

        if "optimizer" in self.proc_opts:
            found_optimizer = True
        if "losses" in self.proc_opts:
            found_loss = True

        if not found_loss:
            opts['losses'] = [torch.nn.functional.mse_loss] * len(self.proc_outputs)

        for k, v in self.proc_opts.items():
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
                opts['optimizer'] = torch.optim.SGD(self.proc.parameters(), lr=opts['lr'])

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
        self.proc_opts = opts

    def __guess_proc_optional_inputs(self) -> None:
        """Infers which processor inputs have default values by inspecting the module's ``forward`` signature."""
        self.proc_optional_inputs = []
        if isinstance(self.proc, ModuleWrapper):
            if hasattr(self.proc.module, "forward"):
                sig = inspect.signature(self.proc.module.forward)
            else:
                sig = inspect.signature(self.proc.forward)
        else:
            sig = inspect.signature(self.proc.forward)

        i = 0
        for name, param in sig.parameters.items():
            if i >= len(self.proc_inputs):
                break
            if param.default is not inspect.Parameter.empty:
                self.proc_optional_inputs.append({"has_default": True, "default_value": param.default})
            else:
                self.proc_optional_inputs.append({"has_default": False, "default_value": None})
            i += 1
