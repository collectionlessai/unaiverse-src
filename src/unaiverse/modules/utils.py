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
import threading
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from unaiverse.dataprops import Data4Proc
from torchvision import datasets, transforms


def transforms_factory(trans_type: str, add_batch_dim: bool = True, return_inverse: bool = False):
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
    return torch.clamp(x, min=-1., max=1.)


def target_shape_fixed_cross_entropy(output, target, *args, **kwargs):
    if len(target.shape) > 1:
        target = target.squeeze(0)
    return torch.nn.functional.cross_entropy(output, target, *args, **kwargs)


def set_seed(seed: int) -> None:
    if seed >= 0:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(0)


def get_proc_inputs_and_proc_outputs_for_rnn(u_shape: torch.Size | tuple, du_dim: int, y_dim: int):
    if isinstance(u_shape, torch.Size):
        u_shape = tuple(u_shape)
    proc_inputs = [
        Data4Proc(data_type="tensor", tensor_shape=(None,) + u_shape, tensor_dtype=torch.float32,
                  pubsub=False, private_only=True),
        Data4Proc(data_type="tensor", tensor_shape=(None, du_dim,), tensor_dtype=torch.float32,
                  pubsub=False, private_only=True)
    ]
    proc_outputs = [
        Data4Proc(data_type="tensor", tensor_shape=(None, y_dim), tensor_dtype=torch.float32,
                  pubsub=False, private_only=True)
    ]
    return proc_inputs, proc_outputs


def get_proc_inputs_and_proc_outputs_for_image_classification(y_dim: int):
    if y_dim == -1:
        y_dim = 1000  # Assuming ImageNet-trained models
    proc_inputs = [Data4Proc(data_type="img", pubsub=False, private_only=True)]
    proc_outputs = [Data4Proc(data_type="tensor", tensor_shape=(None, y_dim), tensor_dtype=torch.float32,
                              pubsub=False, private_only=True)]
    return proc_inputs, proc_outputs


def error_rate_mnist_test_set(network: torch.nn.Module, mnist_data_save_path: str):

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
    for x, y in mnist_test:
        o = network(x)
        c = torch.argmax(o, dim=1)
        error_rate += float(torch.sum(c != y).item())
        n += x.shape[0]
    error_rate /= n
    network.training = training_flag_backup

    return error_rate


class MultiIdentity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        if len(args) == 1:
            return args[0]
        return args


class HumanModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.__out_text = None
        self.__out_img = None
        self.__event = threading.Event()

    def forward(self, text: str, img: Image):
        self.__out_text = None
        self.__out_img = None
        self.__event.clear()
        self.__event.wait()  # Blocks until set (Waiting for human inputs)
        return self.__out_text, self.__out_image

    def go_ahead(self, text: str, img: Image):
        self.__out_text = text
        self.__out_img = img
        self.__event.set()  # Unblock forward(...)


class ModuleWrapper(torch.nn.Module):
    def __init__(self,
                 module: torch.nn.Module | None = None,
                 proc_inputs: list[Data4Proc] | None = None,
                 proc_outputs: list[Data4Proc] | None = None,
                 seed: int = -1):
        super(ModuleWrapper, self).__init__()
        self.device = None  # The device which is supposed to host the module
        self.module = None  # The module itself
        self.proc_inputs = proc_inputs  # The list of Data4Proc objects describing the input types of the module
        self.proc_outputs = proc_outputs  # The list of Data4Proc objects describing the output types of the module

        # Working
        set_seed(seed)
        device_env = os.getenv("PROC_DEVICE", None)
        self.device = torch.device("cpu") if device_env is None else torch.device(device_env)
        self.module = module.to(self.device) if module is not None else None

    def forward(self, *args, **kwargs):

        # The forward signature expected by who calls this method is:
        # forward(self, *args, first: bool, last: bool, **kwargs)
        # so we have to discard 'first' and 'last' that are not used by an external module not designed for this library
        del kwargs['first']
        del kwargs['last']

        # Calling the module
        return self.module(*args, **kwargs)


class AgentProcessorChecker:

    def __init__(self, processor_container: object):
        assert hasattr(processor_container, 'proc'), "Invalid processor container object"
        assert hasattr(processor_container, 'proc_inputs'), "Invalid processor container object"
        assert hasattr(processor_container, 'proc_outputs'), "Invalid processor container object"
        assert hasattr(processor_container, 'proc_opts'), "Invalid processor container object"
        assert hasattr(processor_container, 'proc_optional_inputs'), "Invalid processor container object"

        # Getting processor-related info from the main object which collects processor and its properties
        proc: torch.nn.Module = processor_container.proc
        proc_inputs: list[Data4Proc] | None = processor_container.proc_inputs
        proc_outputs: list[Data4Proc] | None = processor_container.proc_outputs
        proc_opts: dict | None = processor_container.proc_opts
        proc_optional_inputs: list | None = processor_container.proc_optional_inputs

        assert proc is None or isinstance(proc, torch.nn.Module), "Processor (proc) must be a torch.nn.Module"
        assert (proc_inputs is None or (
                isinstance(proc_inputs, list) and (len(proc_inputs) == 0 or
                                                   len(proc_inputs) > 0 and isinstance(proc_inputs[0], Data4Proc)))), \
            "Invalid proc_inputs: it must be None or a list of Data4Proc"
        assert (proc_outputs is None or (
                isinstance(proc_inputs, list) and (len(proc_inputs) == 0 or
                                                   len(proc_inputs) > 0 and isinstance(proc_inputs[0], Data4Proc)))), \
            "Invalid proc_inputs: it must be None or a list of Data4Proc"
        assert (proc_opts is None or isinstance(proc_opts, dict)), "Invalid proc_opts: it must be None or a dictionary"

        # Saving as attributes
        self.proc = proc
        self.proc_inputs = proc_inputs
        self.proc_outputs = proc_outputs
        self.proc_opts = proc_opts
        self.proc_optional_inputs = proc_optional_inputs

        # Dummy processor (if no processor was provided)
        if self.proc is None:
            self.proc = ModuleWrapper(module=MultiIdentity())
            self.proc.device = torch.device("cpu")
            if self.proc_inputs is None:
                self.proc_inputs = [Data4Proc(data_type="all", pubsub=False, private_only=False)]
            if self.proc_outputs is None:
                self.proc_outputs = [Data4Proc(data_type="all", pubsub=False, private_only=False)]
            self.proc_opts = {'optimizer': None, 'losses': [None] * len(self.proc_outputs)}
        else:

            # String telling it is a human
            if isinstance(self.proc, str) and self.proc.lower() == "human":
                self.proc = ModuleWrapper(module=HumanModule())
                self.proc.device = torch.device("cpu")
                self.proc_inputs = [Data4Proc(data_type="text", pubsub=False, private_only=False),
                                    Data4Proc(data_type="img", pubsub=False, private_only=False)]
                self.proc_outputs = [Data4Proc(data_type="text", pubsub=False, private_only=False),
                                     Data4Proc(data_type="img", pubsub=False, private_only=False)]

            # Wrapping to have the basic attributes (device)
            elif not isinstance(self.proc, ModuleWrapper):
                self.proc = ModuleWrapper(module=self.proc)
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

    def __guess_proc_inputs(self):
        if hasattr(self.proc, "proc_inputs"):
            if self.proc.proc_inputs is not None:
                self.proc_inputs = []
                for p in self.proc.proc_inputs:
                    self.proc_inputs.append(p.clone())
            return

        first_layer = None

        # Traverse modules to find the first real layer (skip containers like Sequential)
        for layer in self.proc.modules():
            if (not isinstance(layer, (torch.nn.Sequential,
                                       torch.nn.ModuleList,
                                       torch.nn.ModuleDict))
                    and not isinstance(layer, torch.nn.Module)
                    and hasattr(layer, 'weight')):
                continue  # Skip non-leaf layers
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
        self.proc_inputs = [Data4Proc(name="proc_input_0",
                                      data_type=data_type,
                                      data_desc=data_desc,
                                      tensor_shape=tensor_shape,
                                      tensor_labels=tensor_labels,
                                      tensor_dtype=tensor_dtype,
                                      stream_to_proc_transforms=stream_to_proc_transforms,
                                      proc_to_stream_transforms=proc_to_stream_transforms,
                                      pubsub=False,
                                      private_only=True)]

    def __guess_proc_outputs(self):
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
            self.proc_outputs.append(Data4Proc(name="proc_output_" + str(j),
                                               data_type=data_type,
                                               data_desc=data_desc,
                                               tensor_shape=tensor_shape,
                                               tensor_labels=tensor_labels,
                                               tensor_dtype=tensor_dtype,
                                               stream_to_proc_transforms=stream_to_proc_transforms,
                                               proc_to_stream_transforms=proc_to_stream_transforms,
                                               pubsub=False,
                                               private_only=True))

    def __guess_proc_opts(self):
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

    def __fix_proc_opts(self):
        opts = {}
        found_optimizer = False
        found_loss = False
        cannot_fix = False

        if "optimizer" in self.proc_opts:
            found_optimizer = True
        if "losses" in self.proc_opts:
            found_loss = True

        if not found_loss:
            opts['losses'] = [torch.nn.functional.mse_loss] * len(self.proc_opts)

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
            elif k == "losses" and isinstance(v, list) or isinstance(v, tuple):
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

    def __guess_proc_optional_inputs(self):
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
