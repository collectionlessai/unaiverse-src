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
import csv
import math
import torch
import random
import pathlib
from PIL import Image
from .clock import Clock
from .dataprops import DataProps
from unaiverse.utils.misc import show_images_grid


class DataStream:
    """
    Base class for handling a generic data stream.
    """

    def __init__(self,
                 props: DataProps,
                 clock: Clock = Clock()) -> None:
        """Initialize a DataStream.

        Args:
            props (DataProps): Properties of the stream.
            clock (Clock): Clock object for time management (usually provided from outside).
        """

        # A stream can be turned off
        self.props = props
        self.clock = clock
        self.data = None
        self.data_timestamp = self.clock.get_time()
        self.data_timestamp_when_got_by = {}
        self.data_tag = -1
        self.data_uuid = None
        self.data_uuid_expected = None
        self.data_uuid_clearable = False
        self.enabled = True

    @staticmethod
    def create(stream: 'DataStream', name: str | None = None, group: str = 'none',
               public: bool = True, pubsub: bool = True):
        """Create and set the name for a given stream, also updates data stream labels.

        Args:
            stream (DataStream): The stream object to modify.
            name (str): The name of the stream.
            group (str): The name of the group to which the stream belongs.
            public (bool): If the stream is going to be served in the public net or the private one.
            pubsub (bool): If the stream is going to be served by broadcasting (PubSub) or not.

        Returns:
            Stream: The modified stream with updated group name.
        """
        assert name is not None or group != 'none', "Must provide either name or group name."
        stream.props.set_group(group)
        if name is not None:
            stream.props.set_name(name)
        else:
            stream.props.set_name(stream.props.get_name() + "@" + group)  # If name is None, then a group was provided
        stream.props.set_public(public)
        stream.props.set_pubsub(pubsub)
        if (stream.props.is_flat_tensor_with_labels() and
                len(stream.props.tensor_labels) == 1 and stream.props.tensor_labels[0] == 'unk'):
            stream.props.tensor_labels[0] = group if group != 'none' else name
        return stream

    def enable(self):
        """Enable the stream, allowing data to be retrieved.

        Returns:
            None
        """
        self.enabled = True

    def disable(self):
        """Disable the stream, preventing data from being retrieved.

        Returns:
            None
        """
        self.enabled = False

    def get_props(self) -> DataProps:
        return self.props

    def net_hash(self, prefix: str):
        return self.get_props().net_hash(prefix)

    @staticmethod
    def peer_id_from_net_hash(net_hash):
        return DataProps.peer_id_from_net_hash(net_hash)

    @staticmethod
    def name_or_group_from_net_hash(net_hash):
        return DataProps.name_or_group_from_net_hash(net_hash)

    @staticmethod
    def is_pubsub_from_net_hash(net_hash):
        return DataProps.is_pubsub_from_net_hash(net_hash)

    def is_pubsub(self):
        return self.get_props().is_pubsub()

    def is_public(self):
        return self.get_props().is_public()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | None, int]:
        """Get item for a specific clock cycle. Not implemented for base class: it will be implemented in buffered data
        streams or stream that can generate data on-the-fly.

        Args:
            idx (int): Index of the data to retrieve.

        Raises:
            ValueError: Always, since this method should be overridden.
        """
        raise ValueError("Not implemented (expected to be only present in data streams that are buffered or "
                         "that can generated on the fly)")

    def __str__(self) -> str:
        """String representation of the object.

        Returns:
            str: String representation of the object.
        """
        return f"[DataStream] enabled: {self.enabled}\n\tprops: {self.props}"

    def __len__(self):
        """Get the length of the stream.

        Returns:
            int: Infinity (as this is a lifelong stream).
        """
        return math.inf

    def set(self, data: torch.Tensor | Image.Image | str, data_tag: int = -1) -> bool:
        """Set a new data sample into the stream, that will be provided when calling "get()".

        Args:
            data (torch.Tensor): Data sample to set.
            data_tag (int): Custom data time tag >= 0 (Default: -1, meaning no tags).

        Returns:
            bool: True if data was accepted based on time constraints, else False.
        """
        if self.props.delta <= 0. or self.props.delta <= (self.clock.get_time() - self.data_timestamp):
            self.data = self.props.adapt_tensor_to_tensor_labels(data) if data is not None else None
            self.data_timestamp = self.clock.get_cycle_time()
            self.data_tag = data_tag if data_tag >= 0 else self.clock.get_cycle()
            return True
        else:
            return False

    def get(self, requested_by: str | None = None) -> torch.Tensor | None:
        """Get the most recent data sample from the stream (i.e., the last one that was "set").

        Returns:
            torch.Tensor | None: Adapted data sample if available.
        """
        if self.enabled:
            if requested_by is not None:
                if (requested_by not in self.data_timestamp_when_got_by or
                        self.data_timestamp_when_got_by[requested_by] != self.data_timestamp):
                    self.data_timestamp_when_got_by[requested_by] = self.data_timestamp
                    return self.data
                else:
                    return None
            else:
                return self.data
        else:
            return None

    def get_timestamp(self) -> float:
        return self.data_timestamp

    def get_tag(self) -> int:
        return self.data_tag

    def get_uuid(self, expected: bool = False) -> str | None:
        return self.data_uuid if not expected else self.data_uuid_expected

    def set_uuid(self, ref_uuid: str | None, expected: bool = False):
        if not expected:
            self.data_uuid = ref_uuid
        else:
            self.data_uuid_expected = ref_uuid

        # When we "set" a UUID, even if None, we unmark the "clear UUID" flag, to be sure that nobody is going to
        # accidentally clear this information
        self.data_uuid_clearable = False

    def mark_uuid_as_clearable(self):
        self.data_uuid_clearable = True

    def clear_uuid_if_marked_as_clearable(self):
        if self.data_uuid_clearable:
            self.data_uuid = None
            self.data_uuid_expected = None
            self.data_uuid_clearable = False

    def set_props(self, data_stream: 'DataStream'):
        """Set (edit) the data properties picking up the ones of another DataStream.

        Args:
            data_stream (DataStream): the source DataStream from which DataProp is taken.

        Returns:
            None
        """
        self.props = data_stream.props


class BufferedDataStream(DataStream):
    """
    Data stream with buffer support to store historical data.
    """

    def __init__(self, props: DataProps, clock: Clock = Clock(), is_static: bool = False):
        """Initialize a BufferedDataStream.

        Args:
            is_static (bool): If True, the buffer stores only one item that is reused.
        """
        super().__init__(props=props, clock=clock)

        # We store the data samples, and we cache their text representation (for speed)
        self.data_buffer = []
        self.text_buffer = []

        self.is_static = is_static  # A static stream store only one sample and always yields it

        # We need to remember the fist cycle in which we started buffering and the last one we buffered
        self.first_cycle = -1
        self.last_cycle = -1
        self.last_get_cycle = -2  # Keep it to -2 (since -1 is the starting value for cycles)
        self.buffered_data_index = -1

        self.restart_before_next_get = set()

    def get(self, requested_by: str | None = None) -> tuple[torch.Tensor | None, float]:
        """Get the current data sample based on cycle and buffer.

        Returns:
            torch.Tensor | None: Current buffered sample.
        """
        if requested_by is not None and requested_by in self.restart_before_next_get:
            self.restart_before_next_get.remove(requested_by)
            self.restart()

        cycle = self.clock.get_cycle() - self.first_cycle  # This ensures that first get clock = first sample

        # These two lines might make you think "hey, call super().set(self[cycle]), it is the same!"
        # however, it is not like that, since "set" will also call "adapt_to_labels", that is not needed for
        # buffered streams
        if (self.last_get_cycle != cycle and
                (self.props.delta <= 0. or self.props.delta <= (self.clock.get_time() - self.data_timestamp))):
            self.last_get_cycle = cycle
            self.buffered_data_index += 1
            new_data, new_tag = self[self.buffered_data_index]
            self.data = new_data
            self.data_timestamp = self.clock.get_cycle_time()
            self.data_tag = new_tag
        return super().get(requested_by)

    def set(self, data: torch.Tensor, data_tag: int = -1):
        """Store a new data sample into the buffer.

        Args:
            data (torch.Tensor): Data to store.
            data_tag (int): Custom data time tag >= 0 (Default: -1, meaning no tags).

        Returns:
            bool: True if the data was buffered.
        """
        ret = super().set(data, data_tag)

        if ret:
            if not self.is_static or len(self.data_buffer) == 0:
                self.data_buffer.append((self.props.adapt_tensor_to_tensor_labels(data), self.get_tag()))
                if self.props.is_flat_tensor_with_labels():
                    self.text_buffer.append(self.props.to_text(data))
                elif self.props.is_text():
                    self.text_buffer.append(data)

                # Boilerplate
                if self.first_cycle < 0:
                    self.first_cycle = self.clock.get_cycle()
                    self.last_cycle = self.first_cycle
                else:

                    # Filling gaps with "None"
                    cycle = self.clock.get_cycle()
                    if cycle > self.last_cycle + 1:
                        for cycle in range(cycle, self.last_cycle + 1):
                            self.data_buffer.append((None, -1))
                        self.last_cycle = cycle - 1

                    self.last_cycle += 1
        return ret

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | None, int]:
        """Retrieve a sample from the buffer based on the given clock cycle.

        Args:
            idx (int): Index (>=0) of the sample.

        Returns:
            torch.Tensor | None: The sample, if available.
        """
        if not self.is_static:
            if idx >= self.__len__() or idx < 0:
                return None, -1
            data, data_tag = self.data_buffer[idx]
        else:
            data, data_tag = self.data_buffer[0]
        return data, data_tag if data_tag >= 0 else self.clock.get_cycle() - self.first_cycle

    def __len__(self):
        """Get number of samples in the buffer.

        Returns:
            int: Number of buffered samples.
        """
        return len(self.data_buffer)

    def set_first_cycle(self, cycle):
        """Manually set the first cycle for the buffer.

        Args:
            cycle (int): Global cycle to start from.
        """
        self.first_cycle = cycle
        self.last_cycle = cycle + len(self)

    def get_first_cycle(self):
        """Get the first cycle of the stream.

        Returns:
            int: First cycle value.
        """
        return self.first_cycle

    def restart(self):
        """Restart the buffer using the current clock cycle.
        """
        self.set_first_cycle(max(self.clock.get_cycle(), 0))
        self.buffered_data_index = -1
        self.data_timestamp_when_got_by = {}

    def plan_restart_before_next_get(self, requested_by: str):
        self.restart_before_next_get.add(requested_by)

    def clear_buffer(self):
        """Clear the data buffer
        """
        self.data_buffer = []
        self.text_buffer = []

        self.first_cycle = -1
        self.last_cycle = -1
        self.last_get_cycle = -2  # Keep it to -2 (since -1 is the starting value for cycles)
        self.buffered_data_index = -1

        self.restart_before_next_get = set()

    def shuffle_buffer(self, seed: int = -1):
        old_buffer = self.data_buffer
        indices = list(range(len(old_buffer)))

        state = random.getstate()
        if seed >= 0:
            random.seed(seed)
        random.shuffle(indices)
        if seed >= 0:
            random.setstate(state)

        self.data_buffer = []
        k = 0
        for i in indices:
            self.data_buffer.append((old_buffer[i][0], old_buffer[k][1]))
            k += 1

        if self.text_buffer is not None and len(self.text_buffer) == len(self.data_buffer):
            old_text_buffer = self.text_buffer
            self.text_buffer = []
            for i in indices:
                self.text_buffer.append(old_text_buffer[i])

    def to_text_snippet(self, length: int | None = None):
        """Convert buffered text samples to a single long string.

        Args:
            length (int | None): Optional length of the resulting text snippet.

        Returns:
            str | None: Human-readable text sequence.
        """
        if self.text_buffer is not None and len(self.text_buffer) > 0:
            if length is not None:
                le = max(length // 2, 1)
                text = " ".join(self.text_buffer[0:min(le, len(self.text_buffer))])
                text += (" ... " + (" ".join(self.text_buffer[max(le, len(self.text_buffer) - le):]))) \
                    if len(self.text_buffer) > le else ""
            else:
                text = " ".join(self.text_buffer)
        else:
            text = None
        return text

    def get_since_timestamp(self, since_what_timestamp: float, stride: int = 1) -> (
            tuple[list[int] | None, list[torch.Tensor | None] | None, int, DataProps]):
        """Retrieve all samples starting from a given timestamp.

        Args:
            since_what_timestamp (float): Timestamp in seconds.
            stride (int): Sampling stride.

        Returns:
            Tuple containing list of cycles, data, current cycle, and data properties.
        """
        since_what_cycle = self.clock.time2cycle(since_what_timestamp)
        return self.get_since_cycle(since_what_cycle, stride)

    def get_since_cycle(self, since_what_cycle: int, stride: int = 1) -> (
            tuple[list[int] | None, list[torch.Tensor | None] | None, int, DataProps]):
        """Retrieve all samples starting from a given clock cycle.

        Args:
            since_what_cycle (int): Cycle number.
            stride (int): Stride to skip cycles.

        Returns:
            Tuple with cycles, data, current cycle, and properties.
        """
        assert stride >= 1 and isinstance(stride, int), f"Invalid stride: {stride}"

        # Notice: this whole routed never calls ".get()", on purpose! it must be as it is
        global_cycle = self.clock.get_cycle()
        if global_cycle < 0:
            return None, None, -1, self.props

        # Fist check: ensure we do not go beyond the first clock and counting the resulting number of steps
        since_what_cycle = max(since_what_cycle, 0)
        num_steps = global_cycle - since_what_cycle + 1

        # Second check: now we compute the index we should pass to get item
        since_what_idx_in_getitem = since_what_cycle - self.first_cycle

        ret_cycles = []
        ret_data = []

        for k in range(0, num_steps, stride):
            _idx = since_what_idx_in_getitem + k
            _data, _ = self[_idx]

            if _data is not None:
                ret_cycles.append(since_what_cycle + k)
                ret_data.append(_data)

        return ret_cycles, ret_data, global_cycle, self.props


class System(DataStream):
    def __init__(self):
        super().__init__(props=DataProps(name=System.__name__, data_type="text", data_desc="System stream",
                                         pubsub=False))
        self.set("ping")

    def get(self, requested_by: str | None = None):
        return super().get()


class Dataset(BufferedDataStream):
    """
    A buffered dataset that streams data from a PyTorch dataset and simulates data-streams for input/output.
    """

    def __init__(self, tensor_dataset: torch.utils.data.Dataset, shape: tuple, index: int = 0, batch_size: int = 1):
        """Initialize a Dataset instance, which wraps around a PyTorch Dataset.

        Args:
            tensor_dataset (torch.utils.data.Dataset): The PyTorch Dataset to wrap.
            shape (tuple): The shape of each sample from the data stream.
            index (int): The index of the element returned by __getitem__ to pick up.
        """
        sample = tensor_dataset[0][index]
        if isinstance(sample, torch.Tensor):
            dtype = sample.dtype
        elif isinstance(sample, int):
            dtype = torch.long
        elif isinstance(sample, float):
            dtype = torch.float32
        else:
            raise ValueError("Expected tensor data or a scalar")

        super().__init__(props=DataProps(name=Dataset.__name__,
                                         data_type="tensor",
                                         data_desc="dataset",
                                         tensor_shape=shape,
                                         tensor_dtype=dtype,
                                         pubsub=True))

        n = len(tensor_dataset)
        b = batch_size
        nb = math.ceil(float(n) / float(b))
        r = n - b * (nb - 1)

        for i in range(0, nb):
            batch = []
            if i == (nb - 1):
                b = r

            for j in range(0, b):
                sample = tensor_dataset[i * b + j][index]
                if isinstance(sample, (int, float)):
                    sample = torch.tensor(sample, dtype=dtype)
                batch.append(sample)

            self.data_buffer.append((torch.stack(batch), -1))

        # It was buffered previously than every other thing
        self.restart()


class ImageFileStream(BufferedDataStream):
    """
    A buffered dataset for image data.
    """

    def __init__(self, image_dir: str, list_of_image_files: str,
                 device: torch.device = None, circular: bool = True, show_images: bool = False):
        """Initialize an ImageFileStream instance for streaming image data.

        Args:
            image_dir (str): The directory containing image files.
            list_of_image_files (str): Path to the file with list of file names of the images.
            device (torch.device): The device to store the tensors on. Default is CPU.
            circular (bool): Whether to loop the dataset or not. Default is True.
        """
        self.image_dir = image_dir
        self.device = device if device is not None else torch.device("cpu")
        self.circular = circular

        # Reading the image file
        # (assume a file with one filename per line or a CSV format with lines such as: cat.jpg,cat,mammal,animal)
        self.image_paths = []

        # Calling the constructor
        super().__init__(props=DataProps(name=ImageFileStream.__name__,
                                         data_type="img",
                                         pubsub=True))

        with open(list_of_image_files, 'r') as f:
            for line in f:
                parts = line.strip().split(',')  # Tolerates if it is a CVS and the first field is the image file name
                image_name = parts[0]
                self.image_paths.append(os.path.join(image_dir, image_name))

        # It was buffered previously than every other thing
        self.last_cycle = -1
        self.first_cycle = self.last_cycle - len(self.image_paths) + 1

        # Possibly print to screen the "clickable" list of images
        if show_images:
            show_images_grid(self.image_paths)
            for i, image_path in enumerate(self.image_paths):
                abs_path = os.path.abspath(image_path)
                file_url = pathlib.Path(abs_path).as_uri()
                basename = os.path.basename(abs_path)  # 'photo.jpg'
                parent = os.path.basename(os.path.dirname(abs_path))  # 'images'
                label = os.path.join(parent, basename) if parent else basename
                clickable_label = f"\033]8;;{file_url}\033\\[{label}]\033]8;;\033\\"
                print(str(i) + " => " + clickable_label)

    def __len__(self):
        """Return the number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | None, int]:
        """Get the image and label for the specified cycle number.

        Args:
            idx (int): The cycle number to retrieve data for.

        Returns:
            tuple: A tuple of tensors (image, label) for the specified cycle.
        """
        if self.circular:
            idx %= self.__len__()
        else:
            if idx >= self.__len__() or idx < 0:
                return None, -1

        image = Image.open(self.image_paths[idx])
        return image, self.clock.get_cycle() - self.first_cycle


class LabelStream(BufferedDataStream):
    """
    A buffered stream for single and multi-label annotations.
    """

    def __init__(self, label_dir: str, label_file_csv: str,
                 device: torch.device = None, circular: bool = True, single_class: bool = False,
                 line_header: bool = False):
        """Initialize an LabelStream instance for streaming labels.

        Args:
            label_dir (str): The directory containing image files.
            label_file_csv (str): Path to the CSV file with labels for the images.
            device (torch.device): The device to store the tensors on. Default is CPU.
            circular (bool): Whether to loop the dataset or not. Default is True.
            single_class (bool): Whether to only consider a single class for labeling. Default is False.
        """
        self.label_dir = label_dir
        self.device = device if device is not None else torch.device("cpu")
        self.circular = circular

        # Reading the label file
        # (assume a file with a labeled element per line or a CSV format with lines such as: cat.jpg,cat,mammal,animal)
        self.labels = []

        class_names = {}
        with open(label_file_csv, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                label = parts[1:]
                for lab in label:
                    class_names[lab] = True
        class_name_to_index = {}
        class_names = list(class_names.keys())

        # Call the constructor
        super().__init__(props=DataProps(name=LabelStream.__name__,
                                         data_type="tensor",
                                         data_desc="label stream",
                                         tensor_shape=(1, len(class_names)),
                                         tensor_dtype=str(torch.float),
                                         tensor_labels=class_names,
                                         tensor_labeling_rule="geq0.5" if not single_class else "max",
                                         pubsub=True))

        for idx, class_name in enumerate(class_names):
            class_name_to_index[class_name] = idx

        with open(label_file_csv, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                label = parts if not line_header else parts[1:]
                target_vector = torch.zeros((1, len(class_names)), dtype=torch.float32)
                for lab in label:
                    idx = class_name_to_index[lab]
                    target_vector[0, idx] = 1.
                self.labels.append(target_vector)

        # It was buffered previously than every other thing
        self.last_cycle = -1
        self.first_cycle = self.last_cycle - len(self.labels) + 1

    def __len__(self):
        """Return the number of labels in the dataset.

        Returns:
            int: Number of labels in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> torch.Tensor | None:
        """Get the image and label for the specified cycle number.

        Args:
            idx (int): The cycle number to retrieve data for.

        Returns:
            tuple: A tuple of tensors (image, label) for the specified cycle.
        """
        if self.circular:
            idx %= self.__len__()
        else:
            if idx >= self.__len__() or idx < 0:
                return None, -1

        label = self.labels[idx].unsqueeze(0).to(self.device)  # Multi-label vector for the image
        return self.props.adapt_tensor_to_tensor_labels(label), self.clock.get_cycle() - self.first_cycle


class TokensStream(BufferedDataStream):
    """
    A buffered dataset for tokenized text, where each token is paired with its corresponding labels.
    """

    def __init__(self, tokens_file_csv: str, circular: bool = True, max_tokens: int = -1):
        """Initialize a Tokens instance for streaming tokenized data and associated labels.

        Args:
            tokens_file_csv (str): Path to the CSV file containing token data.
            circular (bool): Whether to loop the dataset or not. Default is True.
            max_tokens (int): Whether to cut the stream to a maximum number of tokens. Default is -1 (no cut).
        """
        self.circular = circular

        # Reading the data file (assume a token per line or a CSV format with lines such as:
        # token,category_label1,category_label2,etc.)
        tokens = []
        with open(tokens_file_csv, 'r') as f:
            for line in f:
                parts = next(csv.reader([line], quotechar='"', delimiter=','))
                tokens.append(parts[0])
                if 0 < max_tokens <= len(tokens):
                    break

        # Vocabulary
        idx = 0
        word2id = {}
        sorted_stream_of_tokens = sorted(tokens)
        for token in sorted_stream_of_tokens:
            if token not in word2id:
                word2id[token] = idx
                idx += 1
        id2word = [""] * len(word2id)
        for _word, _id in word2id.items():
            id2word[_id] = _word

        # Calling the constructor
        super().__init__(props=DataProps(name=TokensStream.__name__,
                                         data_type="text",
                                         data_desc="stream of words",
                                         stream_to_proc_transforms=word2id,
                                         proc_to_stream_transforms=id2word,
                                         pubsub=True))

        # Tokenized text
        for i, token in enumerate(tokens):
            data = token
            self.data_buffer.append((data, -1))

        # It was buffered previously than every other thing
        self.restart()

    def __getitem__(self, idx: int) -> torch.Tensor | None:
        """Get the image and label for the specified cycle number.

        Args:
            idx (int): The index to retrieve data for.

        Returns:
            tuple: A tuple of tensors (image, label) for the specified cycle.
        """
        if self.circular:
            idx %= self.__len__()
        return super().__getitem__(idx)
