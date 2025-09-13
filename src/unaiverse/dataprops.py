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
import torch
from PIL import Image
from typing import Callable, Any
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class Data4Proc:
    def __init__(self, *args, private_only: bool = False, public_only: bool = False, **kwargs):
        """Initializes a `Data4Proc` object, which is a container for one or two `DataProps` instances. It creates a
         `DataProps` object for a private stream and, optionally, a public stream.

        Args:
            *args: Variable length argument list to be passed to the `DataProps` constructor.
            private_only: If True, only a private stream's `DataProps` is created.
            public_only: If True, only a public stream's `DataProps` is created.
            **kwargs: Arbitrary keyword arguments passed to the `DataProps` constructor.

        Raises:
            ValueError: If both `private_only` and `public_only` are set to True, or if the 'public' argument is passed
                directly.
        """
        self.props = []
        if public_only and private_only:
            raise ValueError("Cannot set both private_only and public_only to True (it does not make any sense)")
        if 'public' in kwargs:
            raise ValueError("Invalid argument was provided to Data4Proc: 'public' (it is an argument of DataProps)")
        kwargs['public'] = False
        self.props.append(DataProps(*args, **kwargs))
        if not private_only:
            kwargs['public'] = True
            self.props.append(DataProps(*args, **kwargs))

    def to_list_of_dicts(self):
        """Converts the contained `DataProps` objects into a list of dictionaries. Each dictionary represents the
        properties of a single data stream.

        Returns:
            A list of dictionaries, where each dictionary holds the properties of a `DataProps` object.
        """
        return [props.to_dict() for props in self.props]

    def to_dict(self):
        """Raises a `RuntimeError` because this method is intended for a single `DataProps` object, not for the
        container class `Data4Proc` which can hold multiple properties.

        Raises:
            RuntimeError: Always, as this method is not supported.
        """
        raise RuntimeError("This method can only be called on a DataProps object and not on Data4Proc")

    def from_dict(self):
        """Raises a `RuntimeError` because this method is intended for a single `DataProps` object, not for the
        container class `Data4Proc` which can hold multiple properties.

        Raises:
            RuntimeError: Always, as this method is not supported.
        """
        raise RuntimeError("This method can only be called on a DataProps object and not on Data4Proc")

    def clone(self):
        """Creates and returns a deep copy of the `Data4Proc` object.

        Returns:
            A new `Data4Proc` object that is a clone of the original.
        """
        ret = Data4Proc()
        ret.props = []
        for p in self.props:
            ret.props.append(p.clone())
        return ret

    def is_public(self):
        """Raises a `RuntimeError` because this method is intended for a single `DataProps` object, not for the
        container class `Data4Proc` which can hold multiple properties.

        Raises:
            RuntimeError: Always, as this method is not supported.
        """
        raise RuntimeError("This method can only be called on a DataProps object and not on Data4Proc")

    def __str__(self):
        """Provides a formatted string representation of the `Data4Proc` object. It lists the number of `DataProps`
        objects it contains and includes the string representation of each of them.

        Returns:
            A string detailing the contents of the `Data4Proc` object.
        """
        s = f"[Data4Proc] Number of DataProps: {len(self.props)}"
        for p in self.props:
            z = str(p).replace("\n", "\n\t")
            s += "\t" + z
        return s

    def __getattr__(self, method_or_attribute_name):
        """Handles dynamic attribute and method access for the `Data4Proc` class. If the requested method name starts
        with 'set_', it creates a new function that applies the corresponding setter method to all contained
        `DataProps` objects. For any other attribute, it returns the attribute from the first `DataProps` object
        in the list.

        Args:
            method_or_attribute_name: The name of the method or attribute being accessed.

        Returns:
            Either a callable function or the requested attribute from the first `DataProps` object.
        """
        if method_or_attribute_name.startswith('set_'):
            def apply_set_method_to_all_props(*args, **kwargs):
                for prop in self.props:
                    getattr(prop, method_or_attribute_name)(*args, **kwargs)

            return apply_set_method_to_all_props
        else:
            return getattr(self.props[0], method_or_attribute_name)


class DataProps:
    """
    A class for handling the properties and transformations of data, including labels.
    It supports different data types: 'tensor', 'tensor_token_id', 'img', and 'text'.

    Attributes:
        VALID_DATA_TYPES (tuple): Tuple of valid data types ('tensor', 'tensor_token_id', 'img', 'text').
    """
    VALID_DATA_TYPES = ('tensor', 'img', 'text', 'all')

    def __init__(self,
                 name: str = "unk",
                 group: str = "none",
                 data_type: str = "text",  # Do not set tensor as default
                 data_desc: str = "unk",
                 tensor_shape: tuple[int | None, ...] | None = None,
                 tensor_labels: list[str] | str | None = None,
                 tensor_dtype: torch.dtype | str | None = None,
                 tensor_labeling_rule: str = "max",
                 stream_to_proc_transforms: Callable[..., Any] | PreTrainedTokenizerBase | str | dict | tuple[
                     dict | Callable[..., Any] | PreTrainedTokenizerBase | str | None,
                     dict | Callable[..., Any] | PreTrainedTokenizerBase | str | None] | None = None,
                 proc_to_stream_transforms: Callable[..., Any] | PreTrainedTokenizerBase | str | list | None = None,
                 delta: float = -1,
                 pubsub: bool = False,
                 public: bool = False):
        """Initializes a DataProps instance.

        Args:
            name (str): Name of the data (default is "unk").
            group (str): Name of the group to which this DataProps belong (default: "none").
            data_type (str): The type of data ('tensor', 'img', or 'text').
            data_desc (str): Description of the data (default is "unk").
            tensor_shape (tuple[int | None] or None): Shape of the tensor data (e.g., (3, 224, 224) or (3, None, None)
                for tensors that are variable size images). It is None for non-tensor data.
            tensor_labels (list[str] or "AutoTokenizer:<tokenizer_model_id>" or None):
                List of labels for the components (features) of tensor data. It can be a string representing the ID of a
                tokenizer which is valid in AutoTokenizer (with prefix "AutoTokenizer:").
            tensor_dtype (torch.dtype or str or None): The string representing the Pytorch dtype of the tensor data.
            tensor_labeling_rule (str): The labeling rule for tensor data ('max' or 'geqX' where X is a number).
            stream_to_proc_transforms (callable or PreTrainedTokenizerBase or str or dict or a list* or None):
                A callable stream format to tensor format conversion fcn (any callable thing, torchvision transforms, a
                pretrained tokenizer, or the model ID from which it can be downloaded (it must have prefix
                "AutoTokenizer:"), or a vocabulary str->int). It is None for already-tensorial data.
                *If you need to distinguish the transform applied to the inputs and to targets, you can pass a list of
                two elements like the just described ones - one for input, one for targets, respectively.
            proc_to_stream_transforms (callable or PreTrainedTokenizerBase or str or list or None): A callable tensor
                to stream format function (any callable thing, torchvision transforms, a Pretrained tokenizer (HF),
                or the model ID from which it can be downloaded, (it must have prefix "AutoTokenizer:"), or a
                vocabulary int->str). It is None for non text data.
            delta (float): Time interval between consecutive data samples (<= 0 for real-time data).
            pubsub (bool): If the stream is supposed to be sent to/received from a Pub/Sub topic.
            public (bool): If the stream is supposed to be accessed through the public net or through the private one.

        Returns:
            None
        """

        # Checking data type
        assert data_type in DataProps.VALID_DATA_TYPES, "Invalid data type"
        assert isinstance(data_desc, str), "Invalid data description"

        # Checking transformations
        assert (stream_to_proc_transforms is None or
                isinstance(stream_to_proc_transforms, str) or
                isinstance(stream_to_proc_transforms, PreTrainedTokenizerBase) or
                callable(stream_to_proc_transforms) or
                isinstance(stream_to_proc_transforms, dict) or
                isinstance(stream_to_proc_transforms, tuple) or
                isinstance(stream_to_proc_transforms, list)), \
            "Invalid stream to processor transforms"

        if stream_to_proc_transforms is not None:
            if not isinstance(stream_to_proc_transforms, list) and not isinstance(stream_to_proc_transforms, tuple):
                self.stream_to_proc_transforms = [stream_to_proc_transforms, stream_to_proc_transforms]
            else:
                assert len(stream_to_proc_transforms) == 2, \
                    "Expected a list with two sets of transforms (input, target)"
                self.stream_to_proc_transforms = stream_to_proc_transforms
            self.__original_stream_to_proc_transforms = stream_to_proc_transforms
        else:
            self.stream_to_proc_transforms = None
            self.__original_stream_to_proc_transforms = None

        assert (proc_to_stream_transforms is None or
                isinstance(proc_to_stream_transforms, str) or
                isinstance(proc_to_stream_transforms, PreTrainedTokenizerBase) or
                callable(proc_to_stream_transforms) or
                isinstance(proc_to_stream_transforms, list)), \
            "Invalid stream to processor transforms"

        self.proc_to_stream_transforms = proc_to_stream_transforms
        self.__original_proc_to_stream_transforms = proc_to_stream_transforms

        # Setting data type and description
        self.data_type = data_type
        self.data_desc = data_desc

        # Setting empty attributes
        self.tensor_shape = None
        self.tensor_dtype = None
        self.tensor_labels = None

        # Checking data in function of its type
        if self.is_tensor():

            # Checking shape
            assert (tensor_shape is not None and
                    isinstance(tensor_shape, (tuple, list))), f"Invalid shape for DataProps: {tensor_shape}"
            assert all(x is None or isinstance(x, int) for x in tensor_shape), \
                f"Invalid shape for DataProps: {tensor_shape}"

            # Setting shape
            self.tensor_shape = tuple(tensor_shape)  # Forcing (important)

            # Checking dtype
            assert (tensor_dtype is not None and
                    (isinstance(tensor_dtype, torch.dtype) or isinstance(tensor_dtype, str)
                     and tensor_dtype.startswith("torch."))), \
                f"Invalid tensor type: {tensor_dtype}"

            # Setting dtype
            self.tensor_dtype = tensor_dtype if isinstance(tensor_dtype, torch.dtype) else eval(tensor_dtype)

            # Checking labels
            assert tensor_labels is None or (isinstance(tensor_labels, list) or
                                             (isinstance(tensor_shape, str) and
                                              tensor_labels.startswith("AutoTokenizer:"))), \
                f"Invalid tensor labels: {tensor_labels}"

            # Setting labels
            if tensor_labels is not None:
                if not (isinstance(tensor_labels, str) and tensor_labels.startswith("AutoTokenizer:")):
                    self.tensor_labels = TensorLabels(self, labels=tensor_labels, labeling_rule=tensor_labeling_rule)
                else:
                    self.set_tensor_labels_from_auto_tokenizer(tensor_labels.split[:][1])

        elif self.is_img():

            # Ensuring other type-related tools are not set
            assert tensor_shape is None and tensor_labels is None and tensor_dtype is None, \
                f"Tensor-related arguments must be None when using a DataProps of type {data_type}"
            assert (self.stream_to_proc_transforms is None or (not isinstance(self.stream_to_proc_transforms, str)
                                                               and not isinstance(self.stream_to_proc_transforms,
                                                                                  PreTrainedTokenizerBase))), \
                "Non-image-related transforms were selected"
            assert (self.proc_to_stream_transforms is None or (not isinstance(self.proc_to_stream_transforms, str)
                                                               and not isinstance(self.proc_to_stream_transforms,
                                                                                  PreTrainedTokenizerBase)
                                                               and not isinstance(self.proc_to_stream_transforms,
                                                                                  list))), \
                "Non-image-related transforms were selected"

        elif self.is_text():

            # Ensuring other type-related tools are not set
            assert tensor_shape is None and tensor_labels is None and tensor_dtype is None, \
                f"Tensor/image-related arguments must be None when using a DataProps of type {data_type}"

            # Setting text to tensor transform (tokenizer in encode mode) (if given)
            if self.stream_to_proc_transforms is not None:
                for j, _tttt in enumerate(self.stream_to_proc_transforms):
                    assert ((isinstance(_tttt, str) and _tttt.startswith("AutoTokenizer:")) or
                            isinstance(_tttt, PreTrainedTokenizerBase) or
                            isinstance(_tttt, dict) or
                            callable(_tttt)), \
                        ("Invalid text tokenizer: expected object of type PreTrainedTokenizerBase or a "
                         "string starting with 'AutoTokenizer:' or a callable object or a dictionary "
                         "(vocabulary str->int)")
                    if isinstance(_tttt, str) and _tttt.startswith("AutoTokenizer:"):
                        self.stream_to_proc_transforms[j] = AutoTokenizer.from_pretrained(_tttt.split(":")[1])

            # Setting tensor to text transform (tokenizer in decode mode OR a given vocabulary int->str) (if given)
            if self.proc_to_stream_transforms is not None:
                assert ((isinstance(self.proc_to_stream_transforms, str) and
                         self.proc_to_stream_transforms.startswith("AutoTokenizer:")) or
                        isinstance(self.proc_to_stream_transforms, PreTrainedTokenizerBase) or
                        isinstance(self.proc_to_stream_transforms, list) or
                        callable(self.proc_to_stream_transforms)), \
                    ("Invalid text tokenizer: expected object of type PreTrainedTokenizerBase or a "
                     "string starting with 'AutoTokenizer:' or a callable object or a dictionary "
                     "(vocabulary int->str)")
                if (isinstance(self.proc_to_stream_transforms, str) and
                        self.proc_to_stream_transforms.startswith("AutoTokenizer:")):
                    self.proc_to_stream_transforms = (
                        AutoTokenizer.from_pretrained(self.proc_to_stream_transforms.split(":")[1]))

        # Checking name and group
        assert "~" not in name, "Invalid chars in stream name"
        assert "~" not in group, "Invalid chars in group name"

        # Initialize properties
        self.name = name
        self.group = group
        self.delta = delta
        self.pubsub = pubsub
        self.public = public

    def to_dict(self):
        """Serializes the `DataProps` object into a dictionary, making it suitable for transmission or storage.
        It converts complex types like `torch.dtype` and `TensorLabels` into simple, serializable formats.

        Returns:
            A dictionary representation of the object's properties.
        """
        return {
            'name': self.name,
            'group': self.group,
            'data_type': self.data_type,
            'data_desc': self.data_desc,
            'tensor_shape': self.tensor_shape,
            'tensor_dtype': str(self.tensor_dtype) if self.tensor_dtype is not None else None,
            'tensor_labels': self.tensor_labels.to_dict() if self.tensor_labels is not None else None,
            'delta': self.delta,
            'pubsub': self.pubsub,
            'public': self.public
        }

    @staticmethod
    def from_dict(d_props):
        """A static method that deserializes a dictionary into a `DataProps` object.

        Args:
            d_props: The dictionary containing the object's properties.

        Returns:
            A new `DataProps` object.
        """
        d_labels = d_props['tensor_labels']
        return DataProps(name=d_props['name'],
                         group=d_props['group'],
                         data_type=d_props['data_type'],
                         data_desc=d_props['data_desc'],
                         tensor_shape=d_props['tensor_shape'],
                         tensor_dtype=d_props['tensor_dtype'],
                         tensor_labels=d_labels['labels'] if d_labels is not None else None,
                         tensor_labeling_rule=d_labels['labeling_rule'] if d_labels is not None else "max",
                         delta=d_props['delta'],
                         pubsub=d_props['pubsub'],
                         public=d_props['public'])

    def clone(self):
        """Creates and returns a deep copy of the `DataProps` instance.
        It preserves the original transformation objects rather than re-evaluating them.

        Returns:
            A new `DataProps` object that is a clone of the original.
        """
        return DataProps(name=self.name,
                         group=self.group,
                         data_type=self.data_type,
                         data_desc=self.data_desc,
                         tensor_shape=self.tensor_shape,
                         tensor_dtype=self.tensor_dtype,
                         tensor_labels=self.tensor_labels.labels if self.tensor_labels is not None else None,
                         tensor_labeling_rule=self.tensor_labels.original_labeling_rule
                         if self.tensor_labels is not None else "max",
                         stream_to_proc_transforms=self.__original_stream_to_proc_transforms,
                         proc_to_stream_transforms=self.__original_proc_to_stream_transforms,
                         delta=self.delta,
                         pubsub=self.pubsub,
                         public=self.public)

    def get_name(self):
        """Retrieves the name of the data stream.

        Returns:
            A string representing the stream's name.
        """

        return self.name

    def get_group(self):
        """Retrieves the name of the group to which the data stream belongs.

        Returns:
            A string representing the group name.
        """

        return self.group

    def get_description(self):
        """Retrieves the description of the data.

        Returns:
            A string with the data description.
        """
        return self.data_desc

    def get_tensor_labels(self):
        """Retrieves the list of tensor labels, if they exist.

        Returns:
            A list of strings or None.
        """
        return self.tensor_labels.labels if self.tensor_labels is not None else None

    def set_name(self, name: str):
        """Sets a new name for the stream, with a check for invalid characters.

        Args:
            name: The new name as a string.
        """
        assert "~" not in name, "Invalid chars in stream name"
        self.name = name

    def set_group(self, group: str):
        """Sets a new group name for the stream, with a check for invalid characters.

        Args:
            group: The new group name as a string.
        """
        assert "~" not in group, "Invalid chars in group name"
        self.group = group

    def set_description(self, desc: str):
        """Sets a new description for the data.

        Args:
            desc: The new description as a string.
        """
        self.data_desc = desc

    def set_public(self, public: bool):
        """Sets whether the stream is public or not.

        Args:
            public: A boolean value.
        """
        self.public = public

    def set_pubsub(self, pubsub: bool):
        """Sets whether the stream uses Pub/Sub or direct messaging.

        Args:
            pubsub: A boolean value.
        """
        self.pubsub = pubsub

    def is_tensor(self):
        """Checks if the data type is 'tensor'.

        Returns:
            True if the type is 'tensor', False otherwise.
        """
        return self.data_type == "tensor"

    def is_img(self):
        """Checks if the data type is 'img'.

        Returns:
            True if the type is 'img', False otherwise.
        """
        return self.data_type == "img"

    def is_text(self):
        """Checks if the data type is 'text'.

        Returns:
            True if the type is 'text', False otherwise.
        """
        return self.data_type == "text"

    def is_tensor_long(self):
        """Checks if the tensor's data type is `torch.long`.

        Returns:
            True if the dtype is `torch.long`, False otherwise.
        """
        return self.tensor_dtype == torch.long if self.tensor_dtype is not None else False

    def is_tensor_float(self):
        """Checks if the tensor's data type is a float type (e.g., `torch.float32`).

        Returns:
            True if the dtype is a float type, False otherwise.
        """
        return str(self.tensor_dtype).startswith("torch.float") if self.tensor_dtype is not None else False

    def is_tensor_img(self):
        """Checks if the tensor's shape corresponds to a typical image format (4D, with 1 or 3 channels).

        Returns:
            True if the shape matches, False otherwise.
        """
        return len(self.tensor_shape) == 4 and (self.tensor_shape[1] == 1 or self.tensor_shape[1] == 3) \
            if self.tensor_shape is not None else False

    def is_tensor_token_ids(self):
        """Checks if the tensor represents token IDs, which is indicated by `torch.long` data type and a 2D shape
        suitable for sequences.

        Returns:
            True if it matches, False otherwise.
        """
        return (self.tensor_dtype == torch.long and
                len(self.tensor_shape) == 2 and (self.tensor_shape[1] >= 1 or self.tensor_shape[1] is None)) \
            if self.tensor_shape is not None else False

    def is_tensor_target_id(self):
        """Checks if the tensor represents a single target ID, indicated by `torch.long` data type and a 1D shape.

        Returns:
            True if it matches, False otherwise.
        """
        return (self.tensor_dtype == torch.long and
                len(self.tensor_shape) == 1) \
            if self.tensor_shape is not None else False

    def is_all(self):
        """Checks if the data type is 'all', which is a wildcard type.

        Returns:
            True if the type is 'all', False otherwise.
        """
        return self.data_type == "all"

    def net_hash(self, prefix: str):
        """Generates a unique network hash for the stream using a provided prefix, Pub/Sub status, and name/group.

        Args:
            prefix: The prefix, typically the peer ID.

        Returns:
            A string representing the network hash.
        """
        return DataProps.build_net_hash(prefix, self.pubsub, self.name_or_group())

    @staticmethod
    def peer_id_from_net_hash(net_hash):
        """A static method to extract the peer ID from a network hash.

        Args:
            net_hash: The network hash string.

        Returns:
            A string representing the peer ID.
        """
        return net_hash.split("::")[0]

    @staticmethod
    def name_or_group_from_net_hash(net_hash):
        """A static method to extract the name or group from a network hash.

        Args:
            net_hash: The network hash string.

        Returns:
            A string representing the name or group.
        """
        return net_hash.split("::ps:")[1] if DataProps.is_pubsub_from_net_hash(net_hash) else net_hash.split("::dm:")[1]

    @staticmethod
    def is_pubsub_from_net_hash(net_hash):
        """A static method to check if a network hash belongs to a Pub/Sub stream.

        Args:
            net_hash: The network hash string.

        Returns:
            True if the hash is for a Pub/Sub stream, False otherwise.
        """
        return "::ps:" in net_hash

    def name_or_group(self):
        """Retrieves the group name if it's set, otherwise defaults to the stream name.

        Returns:
            A string representing the name or group.
        """
        group = self.get_group()
        return group if group != 'none' else self.get_name()

    @staticmethod
    def build_net_hash(prefix: str, pubsub: bool, name_or_group: str):
        """A static method to construct a complete network hash from a prefix, Pub/Sub status, and name/group.

        Args:
            prefix: The peer ID prefix.
            pubsub: The Pub/Sub status.
            name_or_group: The name or group of the stream.

        Returns:
            The constructed network hash string.
        """
        if pubsub:
            return f"{prefix}::ps:{name_or_group}"
        else:
            return f"{prefix}::dm:{name_or_group}"

    @staticmethod
    def normalize_net_hash(not_normalized_net_hash: str):
        """A static method that cleans up or normalizes a network hash string to a canonical format, particularly
        for direct messages.

        Args:
            not_normalized_net_hash: The network hash to normalize.

        Returns:
            The normalized network hash string.
        """
        if not DataProps.is_pubsub_from_net_hash(not_normalized_net_hash):
            if "~" in not_normalized_net_hash:
                return not_normalized_net_hash.split("::dm:")[0] + "::dm:" + not_normalized_net_hash.split("~")[1]
            else:
                parts = not_normalized_net_hash.split("::dm:")
                return parts[0] + "::dm:" + parts[1].split("-")[1]
        else:
            return not_normalized_net_hash

    def is_pubsub(self):
        """Checks if the stream is set to use Pub/Sub.

        Returns:
            True if it's a Pub/Sub stream, False otherwise.
        """
        return self.pubsub

    def is_public(self):
        """Checks if the stream is set to be public.

        Returns:
            True if it's a public stream, False otherwise.
        """
        return self.public

    def set_tensor_labels_from_auto_tokenizer(self, model_id):
        """Initializes and sets the tensor labels by fetching the vocabulary from a Hugging Face `AutoTokenizer`
        model ID.

        Args:
            model_id: The ID of the tokenizer model.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        vocab_size = len(tokenizer.vocab)
        reverse_vocab_list: list[str | None] = [None] * vocab_size
        for i in range(vocab_size):
            reverse_vocab_list[i] = tokenizer.convert_ids_to_tokens(i)
        self.set_tensor_labels(reverse_vocab_list)

    def set_tensor_labels(self, labels: list[str] | None, labeling_rule: str = "max"):
        """Sets the labels for the data.

        Args:
            labels (list[str] or None): List of labels to associate with the data.
            labeling_rule (str): The labeling rule for the labels.

        Returns:
            None
        """
        self.tensor_labels = TensorLabels(self, labels=labels, labeling_rule=labeling_rule)

    def adapt_tensor_to_tensor_labels(self, data: torch.Tensor) -> torch.Tensor:
        """Interleaves data in function of its corresponding labels and the current super-set labels.

        Args:
            data (torch.Tensor): The data tensor to interleave.

        Returns:
            torch.Tensor: The interleaved data tensor.
        """
        if self.is_tensor():
            num_labels = len(self.tensor_labels) if self.tensor_labels is not None else 0
            if num_labels > 0 and data.shape[1] < num_labels and self.tensor_labels.indices is not None:
                data_padded = torch.zeros((data.shape[0], num_labels), device=data.device, dtype=data.dtype)
                data_padded[:, self.tensor_labels.indices] = data
                return data_padded
            else:
                return data  # Do nothing
        else:
            return data  # Do nothing

    def clear_label_adaptation(self, data: torch.Tensor):
        """Removes the padding and returns the original data from an adapted tensor.

        Args:
            data: The adapted tensor.

        Returns:
            The original, un-padded tensor.
        """
        return data[:, self.tensor_labels.indices] if self.tensor_labels.indices is not None else data

    def is_flat_tensor_with_labels(self):
        """Checks if the tensor is a 2D array and has labels, which is a common structure for general feature data.

        Returns:
            True if it is, False otherwise.
        """
        return self.is_tensor() and len(self.tensor_shape) == 2 and self.has_tensor_labels()

    def has_tensor_labels(self):
        """Checks if any tensor labels are associated with the stream.

        Returns:
            True if labels exist, False otherwise.
        """
        return self.tensor_labels is not None and len(self.tensor_labels) > 0

    def to_text(self, data: torch.Tensor | str):
        """Converts the tensor data into a text-based representation exploiting the given labels and the labeling rule.

        Args:
            data (torch.Tensor or str): The data tensor to convert into text (if a string, then pass-through only).

        Returns:
            str or None: The corresponding text representation of the data.

        Raises:
            ValueError: If the data type is not supported for conversion.
        """
        if isinstance(data, str):
            return data
        elif not isinstance(data, torch.Tensor):
            return None
        elif len(data.shape) > 2:  # Can only print 1d data (recall that 1d data has 2 dimensions, due to batch size)
            return None

        if data.shape[0] != 1:
            return None  # "Code designed for a batch of only 1 element

        if self.is_tensor():
            if not self.has_tensor_labels():
                return None

            if self.is_tensor_token_ids():

                # This is the case in which we assume to have a vector of token IDs
                text = ""
                for i in range(0, data.shape[1]):
                    if i > 0:
                        text += " "
                    text += self.tensor_labels[data[0][i].item()]
                return text

            elif self.is_tensor_float():

                # This is the generic case of a 1d tensor
                if self.tensor_labels.labeling_rule == "max":
                    j = torch.argmax(data, dim=1)
                    return self.tensor_labels[j.item()]
                elif self.tensor_labels.labeling_rule == "geq":

                    # Warning: does not work for mini-batches
                    jj = torch.where(data >= self.tensor_labels.labeling_rule_thres)[1]
                    return ", ".join(self.tensor_labels[j] for j in jj.tolist())
                else:
                    return None

        elif self.is_text():
            if self.proc_to_stream_transforms is None:
                return None
            if isinstance(self.proc_to_stream_transforms, PreTrainedTokenizerBase):
                return self.proc_to_stream_transforms.decode(data[0])
            elif isinstance(self.proc_to_stream_transforms, dict):
                if data.dtype != torch.long:

                    # This is the case of probabilities
                    j = torch.argmax(data, dim=1)  # Warning: does not work for mini-batches
                    return self.proc_to_stream_transforms[j.item()]
                else:

                    # This is the case in which we assume to have a vector of token IDs
                    text = ""
                    for i in range(0, data.shape[1]):
                        if i > 0:
                            text += " "
                        text += self.proc_to_stream_transforms[data[0][i].item()]
                    return text
            else:
                return self.proc_to_stream_transforms(data)
        else:
            return None

    def check_and_preprocess(self, data: str | Image.Image | torch.Tensor,
                             allow_class_ids: bool = False, targets: bool = False,
                             device: torch.device = torch.device("cpu")):
        """Prepares incoming data for a processor by validating its type and applying necessary transformations.
        It handles different data types, including tensors, text (strings), and images, raising `ValueError` if
        the data type is unexpected or incompatible with the stream's properties. For text and images, it can apply a
        pre-configured transformation (like a tokenizer or a standard image transform) to convert the data into a
        tensor format suitable for processing. For tensors, it performs validation on shape and data type.

        Args:
            data: The data sample to check and preprocess.
            allow_class_ids: A boolean to allow single-element long tensors, typically for class IDs.
            targets: A boolean to indicate if the data is a target (used to select the correct transformation in a
                dual-transform setup).
            device: The PyTorch device (e.g., 'cpu' or 'cuda') to which the tensor should be moved.

        Returns:
            The preprocessed data, typically a tensor on the specified device.
        """
        if self.is_tensor():
            if isinstance(data, torch.Tensor):

                # Skipping all checks, it is enough to know it is a tensor
                if allow_class_ids and data.dtype == torch.long and len(data.shape) == 1:
                    return data.to(device)

                # Checking dtype
                if self.tensor_dtype != data.dtype:
                    raise ValueError(f"Expected data of type {self.tensor_dtype}, got {data.dtype} ("
                                     f"shape {data.shape})")

                # Checking shape
                if len(self.tensor_shape) != len(data.shape):
                    raise ValueError(f"Expected data with shape {self.tensor_shape}, got {data.shape}")
                for i, s in enumerate(self.tensor_shape):
                    if s is not None:
                        if s != data.shape[i]:
                            raise ValueError(f"Expected data with shape {self.tensor_shape}, got {data.shape}")

                # Checking labels
                if self.has_tensor_labels():
                    if data.ndim != 2:
                        raise ValueError("Only 2d tensors are expected for "
                                         "labeled attributes (1st dimension is batch dim)")
                    if not (self.is_tensor_token_ids() or data.shape[1] == self.tensor_labels.num_labels):
                        raise ValueError(f"Expected data with {self.tensor_labels.num_labels} "
                                         f"components (ignoring the 1st dimension), "
                                         f"got {data[0].numel()}")

                return data.to(device)
            else:
                raise ValueError(f"Expecting tensor data, got {type(data)}")
        elif self.is_text():
            if isinstance(data, str):
                if self.stream_to_proc_transforms is not None:
                    text_to_tensor_transform = self.stream_to_proc_transforms[int(targets)]
                    if text_to_tensor_transform is not None:
                        if isinstance(text_to_tensor_transform, PreTrainedTokenizerBase):
                            return text_to_tensor_transform(data, return_tensors='pt')['input_ids'].to(device)  # Tok
                        elif isinstance(text_to_tensor_transform, dict):
                            return torch.tensor(text_to_tensor_transform[data]
                                                if data in text_to_tensor_transform else len(text_to_tensor_transform),
                                                dtype=torch.long, device=device).view(1, -1)  # Warning batch size 1
                        else:
                            return text_to_tensor_transform(data).to(device)  # Custom callable function
                    else:
                        return data
                else:
                    return data
            else:
                raise ValueError(f"Expecting text (string) data, got {type(data)}")
        elif self.is_img():
            if isinstance(data, Image.Image):
                if self.stream_to_proc_transforms is not None:
                    img_to_tensor_transform = self.stream_to_proc_transforms[int(targets)]
                    if img_to_tensor_transform is not None:
                        return img_to_tensor_transform(data).to(device)
                    else:
                        return data
                else:
                    return data
            else:
                raise ValueError(f"Expecting image (PIL.Image) data, got {type(data)}")
        elif self.is_all():
            return data
        else:
            raise ValueError(f"Unexpected data type, {self.data_type}")

    def check_and_postprocess(self, data: str | Image.Image | torch.Tensor):
        """Takes a processor's output and validates it before converting it back into a stream-compatible format.
        It handles `torch.Tensor` data, applying a `proc_to_stream_transform` (if one exists) to convert the tensor
        into an appropriate format for the stream, such as a string for text or a PIL `Image` for images. It performs
        a final check on the data's format (shape, dtype, etc.) to ensure consistency with the stream's properties.

        Args:
            data: The output from the processor, typically a tensor.

        Returns:
            The post-processed data, in a stream-compatible format (e.g., a string, image, or CPU tensor).
        """
        if self.is_tensor():
            if isinstance(data, torch.Tensor):
                if self.proc_to_stream_transforms is not None:
                    data = self.proc_to_stream_transforms(data)
                data = data.cpu()

                # Checking dtype
                if self.tensor_dtype != data.dtype:
                    raise ValueError(f"Expected data of type {self.tensor_dtype}, got {data.dtype}")

                # Checking shape
                if len(self.tensor_shape) != len(data.shape):
                    raise ValueError(f"Expected data with shape {self.tensor_shape}, got {data.shape}")
                for i, s in enumerate(self.tensor_shape):
                    if s is not None:
                        if s != data.shape[i]:
                            raise ValueError(f"Expected data with shape {self.tensor_shape}, got {data.shape}")

                # Checking labels
                if self.has_tensor_labels():
                    if data.ndim != 2:
                        raise ValueError("Only 2d tensors are expected for "
                                         "labeled attributes (1st dimension is batch dim)")
                    if not (self.is_tensor_token_ids() or data.shape[1] == self.tensor_labels.num_labels):
                        raise ValueError(f"Expected data with {self.tensor_labels.num_labels} "
                                         f"components (ignoring the 1st dimension), "
                                         f"got {data[0].numel()}")

                return data
            else:
                raise ValueError(f"Expecting tensor data, got {type(data)}")
        elif self.is_text():
            if isinstance(data, str):
                return data
            elif isinstance(data, torch.Tensor):
                data = data.cpu()
                if self.proc_to_stream_transforms is not None:
                    assert data.shape[0] == 1, f"Code designed for a batch of only 1 element, got {data.shape[0]}"
                    if isinstance(self.proc_to_stream_transforms, PreTrainedTokenizerBase):
                        return self.proc_to_stream_transforms.decode(data[0])  # Tokenizer
                    elif isinstance(self.proc_to_stream_transforms, list):
                        if data.dtype != torch.long:

                            # This is the case of probabilities
                            j = torch.argmax(data, dim=1)  # Warning: does not work for mini-batches
                            return self.proc_to_stream_transforms[j.item()]
                        else:

                            # This is the case in which we assume to have a vector of token IDs
                            text = ""
                            for i in range(0, data.shape[1]):
                                if i > 0:
                                    text += " "
                                text += self.proc_to_stream_transforms[data[0][i].item()]
                            return text
                    else:
                        return self.proc_to_stream_transforms(data)  # Custom callable function
                else:
                    raise ValueError(f"Cannot decode torch.Tensor to text, since text_to_tensor_inv_transform is None")
            else:
                raise ValueError(f"Expecting text (string) or tensor data, got {type(data)}")
        elif self.is_img():
            if isinstance(data, Image.Image):
                return data
            elif isinstance(data, torch.Tensor):
                data = data.cpu()
                if self.proc_to_stream_transforms is not None:
                    return self.proc_to_stream_transforms(data)
                else:
                    raise ValueError(f"Cannot convert a tensor to PIL.Image, since img_to_tensor_inv_transform is None")
            else:
                raise ValueError(f"Expecting image (PIL.Image) data or torch.Tensor, got {type(data)}")
        elif self.is_all():
            return data
        else:
            raise ValueError(f"Unexpected data type, {self.data_type}")

    def is_compatible(self, props_to_compare: 'DataProps') -> bool:
        """Checks if the current DataProps instance is compatible with another DataProps instance.
        Checks include data type, shape, and labels.

        Args:
            props_to_compare (DataProps): The DataProps instance to check compatibility with.

        Returns:
            bool: True if compatible, False otherwise.
        """

        # Checking data type
        if self.data_type != props_to_compare.data_type and self.data_type != "all":
            return False

        # In the case of tensors...
        if self.is_tensor():

            # Checking shape
            if len(self.tensor_shape) == len(props_to_compare.tensor_shape):
                for s, p in zip(self.tensor_shape, props_to_compare.tensor_shape):
                    if s is not None and p is not None and s != p:
                        return False
            else:
                return False

            # Checking labels (if possible)
            if (not self.has_tensor_labels()) or (not props_to_compare.has_tensor_labels()):
                return True
            else:
                return self.tensor_labels == props_to_compare.tensor_labels
        else:
            return True

    def __str__(self):
        """Provides a string representation of the DataProps instance.

        Returns:
            str: The string representation of the instance.
        """
        return f"[DataProps]\n{self.to_dict()}"


class TensorLabels:
    """
    A class to manage labels associated with data and perform operations on them.

    Attributes:
        VALID_LABELING_RULES (tuple): Tuple of valid labeling rules ('max', 'geq').
    """

    VALID_LABELING_RULES = ('max', 'geq')

    def __init__(self, data_props: DataProps, labels: list[str] | None, labeling_rule: str = "max"):
        """Initializes the TensorLabels instance.

        Args:
            data_props (DataProps): The DataProps instance that owns these labels.
            labels (list[str] or None): List of labels.
            labeling_rule (str): The rule for labeling (either 'max' or 'geqX', where X is a number).

        Returns:
            None

        Raises:
            AssertionError: If the labels or labeling_rule are invalid.
        """
        assert data_props.is_tensor(), "Tensor labels can only be attached to tensor data properties"
        num_labels = len(labels) if labels is not None else 0
        assert num_labels == 0 or (data_props.is_tensor() and len(data_props.tensor_shape) == 2), \
            "Data attribute labels can only be specified for 2d arrays (batch size + data features)"
        assert len(labeling_rule) >= 3 and labeling_rule[0:3] in TensorLabels.VALID_LABELING_RULES, \
            "Invalid labeling rule"
        try:
            original_labeling_rule = labeling_rule
            if len(labeling_rule) > 3:
                labeling_rule_thres = float(labeling_rule[3:])
                labeling_rule = labeling_rule[0:3]
            else:
                labeling_rule_thres = None
        except ValueError:
            assert False, "Invalid labeling rule"

        # Basic attributes
        self.data_props = data_props
        self.labels = labels
        self.labeling_rule = labeling_rule
        self.labeling_rule_thres = labeling_rule_thres
        self.original_labeling_rule = original_labeling_rule

        # These are mostly operational stuff, similar to private info (but it could be useful to expose them)
        self.num_labels = num_labels
        self.indices = None

    def to_dict(self):
        """Serializes the `TensorLabels` instance into a dictionary, which includes the list of labels and the original
        labeling rule.

        Returns:
            A dictionary containing the labels and the original labeling rule.
        """
        return {
            'labels': self.labels,
            'labeling_rule': self.original_labeling_rule
        }

    def clear_indices(self):
        """Resets the internal `indices` attribute to `None`. This effectively clears any previous label adaptation
        that was performed and allows the object to revert to its original, non-interleaved state.
        """
        self.indices = None

    def __getitem__(self, idx):
        """Retrieves the label at the specified index.

        Args:
            idx (int): The index of the label to retrieve.

        Returns:
            str: The label at the specified index.

        Raises:
            ValueError: If the index is out of bounds or labels are not defined.
        """
        if self.labels is None:
            raise ValueError(f"Cannot retrieve any labels, since they are not there at all (None)")
        if idx < 0 or idx >= self.num_labels:
            raise ValueError(f"Invalid index {idx} for attribute labels of size {self.num_labels}")
        return self.labels[idx]

    def __len__(self):
        """Returns the number of labels.

        Returns:
            int: The number of labels.
        """
        return self.num_labels

    def __iter__(self):
        """Iterates over the labels.

        Returns:
            iterator: An iterator over the labels.
        """
        return iter(self.labels) if self.labels is not None else iter([])

    def __str__(self):
        """Provides a string representation of the DataLabels instance.

        Returns:
            str: The string representation of the instance.
        """
        return (f"[TensorLabels] "
                f"labels: {self.labels}, labeling_rule: {self.labeling_rule}, "
                f"indices_in_superset: {self.indices})")

    def __eq__(self, other):
        """Defines how two `TensorLabels` instances are compared for equality using the `==` operator. Two instances
        are considered equal if they have the same number of labels and the labels themselves match in order.

        Args:
            other: The other object to compare with.

        Returns:
            True if the instances are equal, False otherwise.
        """
        if not isinstance(other, TensorLabels):
            return ValueError("Cannot compare a TensorLabels instance and something else")

        if self.num_labels == other.num_labels:
            if self.num_labels > 0:
                for i, j in zip(self.labels, other.labels):
                    if i != j:
                        return False
                return True
            else:
                return True
        else:
            return False

    def interleave_with(self, superset_labels: list[str]):
        """Interleaves the current labels with a super-set of labels, determining how to index them.

        Args:
            superset_labels (list[str]): The super-set of labels to interleave with.

        Returns:
            None

        Raises:
            AssertionError: If the super-set of labels is not compatible.
        """
        assert superset_labels is not None and self.labels is not None, \
            f"Can only interleave non-empty sets of attribute labels"
        assert len(superset_labels) >= len(self), f"You must provide a super-set of attribute labels"

        # Ensuring it is a super-set of the current labels and finding its position
        if self.indices is not None:
            labels = []
            indices_list = self.indices.tolist()
            for i in indices_list:
                labels.append(self.labels[i])
        else:
            labels = self.labels

        indices = []
        for label in labels:
            assert label in superset_labels, \
                f"Cannot find attribute label {label} in (expected) super-set {superset_labels}"
            indices.append(superset_labels.index(label))

        if len(indices) == len(superset_labels):
            same_labels_and_order = True
            for j, i in enumerate(indices):
                if j != i:
                    same_labels_and_order = False
                    break
        else:
            same_labels_and_order = False

        if not same_labels_and_order:
            self.labels = superset_labels
            self.num_labels = len(self.labels)
            self.indices = torch.tensor(indices, dtype=torch.long)

            # Altering shape
            self.data_props.tensor_shape = (self.data_props.tensor_shape[0], self.num_labels)
        else:
            self.indices = None
