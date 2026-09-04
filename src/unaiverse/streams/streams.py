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
import io
import os
import csv
import PIL
import math
import json
import torch
import base64
import random
from torch import Tensor
from PIL.Image import Image
from typing import TYPE_CHECKING
from unaiverse.clock import clock
from .dataprops import DataProps, TensorLabels
from unaiverse.utils.misc import show_images_grid
from unaiverse.custom import Custom, GenException

# This import ONLY happens during IDE/Type inspection
if TYPE_CHECKING:
    from unaiverse.interaction import Interaction


class Data:
    def __init__(self, copy_me: 'Data | None' = None, uuid: str | None = None) -> None:
        """Initialize a Data instance, optionally as a shallow copy of another Data object.

        Args:
            copy_me: Another Data instance to copy from (shallow copy). If None, creates a fresh instance.
            uuid: UUID to assign to the new instance (ignored when copy_me is provided).
        """
        if not copy_me:
            self.data = None
            self.data_timestamp = 0.
            self.data_timestamp_when_got_by = {}
            self.data_tag = -1
            self.uuid = uuid
        else:
            self.data = copy_me.data  # Shallow
            self.data_timestamp = copy_me.data_timestamp
            self.data_timestamp_when_got_by = {k: v for k, v in copy_me.data_timestamp_when_got_by.items()}
            self.data_tag = copy_me.data_tag
            self.uuid = copy_me.uuid

    def to_code_str(self) -> str:
        """Return a compact string encoding the data availability and tag for debugging.

        Returns:
            A short string of the form ``dt:<ok|no>|<tag>``.
        """
        return "dt:" + ("no" if self.data is None else "ok") + "|" + str(self.data_tag)

    def __str__(self) -> str:
        """Return a compact string encoding the data availability and tag for debugging."""
        return self.to_code_str()


class Stream:
    """
    Base class for handling a generic data stream.

    Can be constructed in two ways:

    1. **Legacy** (backward compat): ``Stream(props=DataProps(...))``
    2. **Keyword-based**: ``Stream(name="x", data_type="tensor", ...)``
    """

    # Class-level type constants for the new creation pattern
    text = "text"
    tensor = "tensor"
    img = "img"
    file = "file"

    def __init__(self,
                 props: DataProps | None = None,
                 name: str | None = None,
                 group: str = "none",
                 data_type: str | None = None,
                 data_desc: str = "unk",
                 tensor_shape: tuple | None = None,
                 tensor_labels: list | str | None = None,
                 tensor_dtype: torch.dtype | str | None = None,
                 tensor_labeling_rule: str = "max",
                 stream_to_proc_transforms=None,
                 proc_to_stream_transforms=None,
                 delta: float = -1,
                 pubsub: bool = False,
                 public: bool = False) -> None:
        """Initialize a Stream.

        Args:
            props: Legacy DataProps object (backward compat). If provided, other DataProps-level params are ignored.
            name: Name of the stream (used when props is None).
            group: Group name (used when props is None).
            data_type: Data type string (used when props is None).
            data_desc: Description (used when props is None).
            tensor_shape: Shape tuple (used when props is None).
            tensor_labels: Labels (used when props is None).
            tensor_dtype: Dtype (used when props is None).
            tensor_labeling_rule: Labeling rule (used when props is None).
            stream_to_proc_transforms: Transforms (used when props is None).
            proc_to_stream_transforms: Transforms (used when props is None).
            delta: Minimum time between samples (used when props is None).
            pubsub: Whether to broadcast via PubSub (used when props is None).
            public: Whether the stream is public (used when props is None).
        """

        # Build props from keyword args if not provided
        if props is None:
            props = DataProps(
                name=name if name is not None else "unk",
                group=group,
                data_type=data_type if data_type is not None else "text",
                data_desc=data_desc,
                tensor_shape=tensor_shape,
                tensor_labels=tensor_labels,
                tensor_dtype=tensor_dtype,
                tensor_labeling_rule=tensor_labeling_rule,
                stream_to_proc_transforms=stream_to_proc_transforms,
                proc_to_stream_transforms=proc_to_stream_transforms,
                delta=delta,
                pubsub=pubsub,
                public=public)

        self.props = props
        self.data_by_uuid = {}
        self.interactions_by_uuid = {}  # Interactions tracking
        self.enabled = True

    @staticmethod
    def create(stream: 'Stream', name: str | None = None, group: str = 'none',
               public: bool = True, pubsub: bool = True, delta: float = -1.):
        """Create and set the name for a given stream, also updates data stream labels.

        Args:
            stream (Stream): The stream object to modify.
            name (str): The name of the stream.
            group (str): The name of the group to which the stream belongs.
            public (bool): If the stream is going to be served in the public net or the private one.
            pubsub (bool): If the stream is going to be served by broadcasting (PubSub) or not.
            delta (float): The minimum interval between two consecutive 'set' operations on the stream.

        Returns:
            Stream: The modified stream with updated group name.
        """
        assert name is not None or group != 'none', "Must provide either name or group name."
        stream.props.set_group(group)
        if name is not None:
            if group != 'none':
                name = name + "@" + group  # This will alter the original name, needed to keep backward compatibility
        else:
            name = stream.props.get_name() + "@" + group  # If name is None, then a group was provided
        stream.props.set_name(name)
        stream.props.set_public(public)
        stream.props.set_pubsub(pubsub)
        stream.props.set_delta(delta)
        assert stream.props is not None
        if stream.props.is_flat_tensor_with_labels():
            tensor_labels = stream.props.tensor_labels
            assert tensor_labels is not None
            tensor_labels: TensorLabels
            if len(tensor_labels) == 1 and tensor_labels[0] == 'unk':
                tensor_labels[0] = group if group != 'none' else name
        return stream

    def enable(self) -> None:
        """Enable the stream, allowing data to be set."""
        self.enabled = True

    def disable(self) -> None:
        """Disable the stream, preventing data from being set."""
        self.enabled = False

    def get_props(self) -> DataProps:
        """Return the DataProps object associated with this stream.

        Returns:
            The DataProps instance describing this stream's properties.
        """
        return self.props

    def net_hash(self, prefix: str) -> str:
        """Return the network hash for this stream.

        Args:
            prefix: The peer ID prefix.

        Returns:
            The network hash string.
        """
        return self.get_props().net_hash(prefix)

    def user_hash(self, prefix: str) -> str:
        """Return the user hash for this stream.

        Args:
            prefix: The peer ID prefix.

        Returns:
            The user hash string.
        """
        return self.get_props().user_hash(prefix)

    @staticmethod
    def peer_id_from_net_hash(net_hash: str) -> str:
        """Extract the peer ID from a network hash.

        Args:
            net_hash: The network hash string.

        Returns:
            The peer ID portion of the hash.
        """
        return DataProps.peer_id_from_net_hash(net_hash)

    @staticmethod
    def peer_id_from_user_hash(user_hash: str) -> str:
        """Extract the peer ID from a user hash.

        Args:
            user_hash: The user hash string.

        Returns:
            The peer ID portion of the hash.
        """
        return DataProps.peer_id_from_user_hash(user_hash)

    @staticmethod
    def name_from_user_hash(user_hash: str) -> str:
        """Extract the stream name from a user hash.

        Args:
            user_hash: The user hash string.

        Returns:
            The stream name portion of the hash.
        """
        return DataProps.name_from_user_hash(user_hash)

    @staticmethod
    def peer_id_from_hash(some_hash: str) -> str | None:
        """Extract the peer ID from either a net hash or a user hash.

        Args:
            some_hash: A net hash or user hash string.

        Returns:
            The peer ID string, or None if the hash format is unrecognized.
        """
        if Stream.is_net_hash(some_hash):
            return DataProps.peer_id_from_net_hash(some_hash)
        elif Stream.is_user_hash(some_hash):
            return DataProps.peer_id_from_user_hash(some_hash)
        return None

    @staticmethod
    def name_or_group_from_net_hash(net_hash: str) -> str:
        """Extract the name or group from a network hash.

        Args:
            net_hash: The network hash string.

        Returns:
            The name or group portion of the hash.
        """
        return DataProps.name_or_group_from_net_hash(net_hash)

    @staticmethod
    def is_pubsub_from_net_hash(net_hash: str) -> bool:
        """Check if a network hash belongs to a Pub/Sub stream.

        Args:
            net_hash: The network hash string.

        Returns:
            True if the hash encodes a Pub/Sub stream, False otherwise.
        """
        return DataProps.is_pubsub_from_net_hash(net_hash)

    @staticmethod
    def is_net_hash(some_hash: str | None) -> bool:
        """Check if the given hash is a network hash (contains '::').

        Args:
            some_hash: The hash string to test.

        Returns:
            True if it is a net hash, False otherwise.
        """
        return some_hash is not None and '::' in some_hash

    @staticmethod
    def is_user_hash(some_hash: str | None) -> bool:
        """Check if the given hash is a user hash (contains ':' but not '::').

        Args:
            some_hash: The hash string to test.

        Returns:
            True if it is a user hash, False otherwise.
        """
        return some_hash is not None and ':' in some_hash and not Stream.is_net_hash(some_hash)

    def is_pubsub(self) -> bool:
        """Check whether this stream uses Pub/Sub.

        Returns:
            True if the stream is a Pub/Sub stream, False otherwise.
        """
        return self.get_props().is_pubsub()

    def is_public(self) -> bool:
        """Check whether this stream is public.

        Returns:
            True if the stream is public, False otherwise.
        """
        return self.get_props().is_public()

    def to_code_str(self) -> str:
        """Return a compact multi-line debug string describing the stream's current state.

        Returns:
            A string encoding the stream name, group, type, and per-UUID interaction/data status.
        """
        s = (f"ngti:{self.props.name}|{self.props.group}|{self.props.data_type}" +
             ("_pubsub" if self.props.pubsub else "") + "|")
        uuids = self.interactions_by_uuid.keys() | self.get_data_uuids()  # Call get_data_uuids due to buffered streams
        if len(uuids) > 0:
            s += "\n" + "\n".join([f"   {uuid} => " +
                                   (self.interactions_by_uuid[uuid].to_code_str()
                                    if uuid in self.interactions_by_uuid else "no-int") + "; " +
                                   (self.data_by_uuid[uuid].to_code_str()
                                    if uuid in self.data_by_uuid else "no-data")
                                   for uuid in uuids])
        else:
            s += "no-int,no-data"
        return s

    def set(self,
            data: torch.Tensor | Image | str | None, data_tag: int = -1, keep_existing_tag: bool = False,
            uuid: str | None = None, force: bool = False) -> Data | None:
        """Set a new data sample into the stream, that will be provided when calling "get()".

        Args:
            data: Data sample to set (tensor, PIL image, or string).
            data_tag: Custom data time tag >= 0 (Default: -1, meaning no tags).
            keep_existing_tag: Keep the data tag that is already in the stream, if the provided data_tag arg is -1.
            uuid: UUID of the interaction storing this data (can be None, meaning "generic UUID", default None).
            force: Boolean flag to force the set operation also if the stream is disabled.

        Returns:
            The updated Data struct if data was accepted based on time constraints, else None.
        """
        if not self.enabled and not force:
            return None

        first = False
        if uuid not in self.data_by_uuid:
            first = True
            self.limit_data_without_interactions()
            self.data_by_uuid[uuid] = Data(uuid=uuid)

        # This is the data sample (with tag and so on)
        data_struct = self.data_by_uuid[uuid]

        if first or self.props.delta <= 0. or self.props.delta <= (clock.get_time() - data_struct.data_timestamp):
            data_struct.data = self.props.adapt_tensor_to_tensor_labels(data) if data is not None else None
            data_struct.data_timestamp = clock.get_cycle_time()
            data_struct.data_tag = data_tag if data_tag >= 0 else (
                clock.get_cycle()) if data_struct.data_tag < 0 or not keep_existing_tag else data_struct.data_tag
            return data_struct
        else:
            return None

    def has_interaction(self, uuid: str | None) -> bool:
        """Check whether an interaction is registered for the given UUID.

        Args:
            uuid: The UUID to look up.

        Returns:
            True if an interaction exists for that UUID, False otherwise.
        """
        return uuid in self.interactions_by_uuid

    def has_data(self, uuid: str | None) -> bool:
        """Check whether data is stored for the given UUID.

        Args:
            uuid: The UUID to look up.

        Returns:
            True if data exists for that UUID, False otherwise.
        """
        return uuid in self.data_by_uuid

    def has_data_and_interaction(self, uuid: str | None) -> bool:
        """Check whether both an interaction and data exist for the given UUID.

        Args:
            uuid: The UUID to look up.

        Returns:
            True if both an interaction and data exist for that UUID, False otherwise.
        """
        return self.has_interaction(uuid) and self.has_data(uuid)

    def limit_data_without_interactions(self) -> None:
        """Evict the oldest data entries that have no associated interaction, keeping at most
        ``Custom.MAX_STREAM_DATA_WITHOUT_INTERACTIONS`` such entries.
        """
        c, oldest_uuid_without_interaction = self.count_data_without_interactions()
        while c >= Custom.MAX_STREAM_DATA_WITHOUT_INTERACTIONS:
            self.remove_data(oldest_uuid_without_interaction)  # Compatible with buffered streams
            if c == Custom.MAX_STREAM_DATA_WITHOUT_INTERACTIONS:
                break
            c, oldest_uuid_without_interaction = self.count_data_without_interactions()

    def count_data_without_interactions(self) -> tuple[int, str | None]:
        """Count data entries that have no associated interaction, and return the oldest such UUID.

        Returns:
            A tuple ``(count, oldest_uuid)`` where ``count`` is the number of data entries without an interaction
            and ``oldest_uuid`` is the UUID of the oldest such entry (or None if count is zero).
        """
        c = 0
        oldest_uuid = None
        for uuid in self.get_data_uuids():  # Call get_data_uuids to be compatible with buffered streams
            if uuid is None:  # Preserve the None (generic) UUID, important for buffered streams, excluded from counts
                continue
            if uuid not in self.interactions_by_uuid:
                if c == 0:
                    oldest_uuid = uuid
                c += 1
        return c, oldest_uuid

    def get_data_uuids(self):
        """Return the set of UUIDs for which data has been stored.

        Returns:
            A view of the keys in the internal data-by-UUID mapping.
        """
        return self.data_by_uuid.keys()

    def get_interaction_uuids(self):
        """Return the set of UUIDs for which an interaction has been registered.

        Returns:
            A view of the keys in the internal interactions-by-UUID mapping.
        """
        return self.interactions_by_uuid.keys()

    def get(self, requested_by: str | None = None, uuid: str | None = None, all_uuids: bool = False) \
            -> str | Image | torch.Tensor | None | list[tuple[str | Image | torch.Tensor | None, int, float]]:
        """Get the most recent data sample from the stream (i.e., the last one that was "set").

        Args:
            requested_by: Identifier of the caller; when provided, each caller receives each sample at most once.
            uuid: UUID of the interaction whose data to retrieve (it can be None in pre-built streams).
            all_uuids: Whether to return a list with all the data samples for all the existing UUIDs (default: False).
                In this case, the uuid argument is ignored.

        Returns:
            The most recent data_sample if available and not already returned to this caller, else None.
            If all_uuids is True, then it returns a list of tuples (data_sample, tag, timestamp),
                where each data_sample has the just mentioned properties.
        """
        # previously stored data (tests/test_streams_io.py::test_disable_blocks_set_but_get_still_reads).
        if all_uuids:
            samples = []
            for _uuid in self.get_data_uuids():
                sample = self.get(requested_by, _uuid)
                if sample is None:
                    continue
                tag = self.get_tag(_uuid)
                timestamp = self.get_timestamp(_uuid)
                samples.append((sample, tag, timestamp))
            return samples

        if uuid not in self.data_by_uuid:
            return None

        data_struct = self.data_by_uuid[uuid]

        if requested_by is not None:
            if (requested_by not in data_struct.data_timestamp_when_got_by or
                    data_struct.data_timestamp_when_got_by[requested_by] != data_struct.data_timestamp):
                data_struct.data_timestamp_when_got_by[requested_by] = data_struct.data_timestamp
                return data_struct.data
            else:
                return None
        else:
            return data_struct.data

    def get_(self, requested_by: str | None = None, uuid: str | None = None):
        """Wrapper of 'get'."""
        return self.get(requested_by, uuid)

    def get_timestamp(self, uuid: str | None = None) -> float | None:
        """Return the timestamp at which data was last set for the given UUID.

        Args:
            uuid: The UUID to query (defaults to the last set UUID).

        Returns:
            The data timestamp as a float, or None if no data exists for that UUID.
        """
        if uuid not in self.data_by_uuid:
            return None

        return self.data_by_uuid[uuid].data_timestamp

    def get_tag(self, uuid: str | None = None) -> int | None:
        """Return the data tag associated with the latest sample for the given UUID.

        Args:
            uuid: The UUID to query (defaults to the last set UUID).

        Returns:
            The integer data tag, or None if no data exists for that UUID.
        """
        if uuid not in self.data_by_uuid:
            return None

        return self.data_by_uuid[uuid].data_tag

    def set_tag(self, data_tag: int, uuid: str | None = None) -> None:
        """Overwrite the data tag for the given UUID without changing the data itself.

        Args:
            data_tag: The new integer tag value.
            uuid: The UUID to update (defaults to the last set UUID).
        """
        if uuid not in self.data_by_uuid:
            return

        self.data_by_uuid[uuid].data_tag = data_tag

    def clear_data(self, uuid: str | None = None) -> None:
        """Remove the data entry for the given UUID, if present.

        Args:
            uuid: The UUID whose data should be removed.
        """
        if uuid in self.data_by_uuid:
            del self.data_by_uuid[uuid]

    def clear_expired_data(self, timeout: float) -> list[str]:
        """Remove all data entries whose timestamp is older than the given retry_timeout.

        Args:
            timeout: Maximum age in seconds; entries older than this are deleted. No-op if <= 0.

        Returns:
            A list of UUIDs that were removed.
        """
        if timeout <= 0.:
            return []

        to_remove = []
        cur_time = clock.get_time()
        for uuid, data_struct in self.data_by_uuid.items():
            if (cur_time - data_struct.data_timestamp) >= timeout:
                to_remove.append(uuid)

        for uuid in to_remove:
            self.remove_data(uuid)  # Overwritten in BufferedStream
            if uuid in self.interactions_by_uuid:
                del self.interactions_by_uuid[uuid]
        return to_remove

    def add_interaction(self, interaction: "Interaction") -> None:
        """Register an interaction that involves this stream.

        Args:
            interaction: The Interaction object.
        """
        if interaction is None:
            return
        uuid = interaction.uuid
        self.interactions_by_uuid[uuid] = interaction

    def edit_uuid_in_data(self, current_uuid: str, new_uuid: str) -> None:
        """Rename a UUID in the data store (used when a temporary UUID is replaced by a permanent one).

        Args:
            current_uuid: The existing UUID to rename.
            new_uuid: The replacement UUID.
        """
        if self.has_data(current_uuid):
            data = self.data_by_uuid[current_uuid]
            data.uuid = new_uuid
            del self.data_by_uuid[current_uuid]  # Do not call remove_data here, since it will break buffered streams
            self.data_by_uuid[new_uuid] = data

    def remove_interaction(self, interaction: "Interaction") -> None:
        """Remove an interaction from this stream.

        Args:
            interaction: The Interaction object to remove.
        """
        uuid = interaction.uuid
        if self.interactions_by_uuid.get(uuid) is interaction:
            del self.interactions_by_uuid[uuid]
            self.limit_data_without_interactions()

    def remove_data(self, uuid: str | None) -> None:
        """Remove data from this stream.

        Args:
            uuid: The UUID of the object to remove.
        """
        if uuid in self.data_by_uuid:
            del self.data_by_uuid[uuid]

    def clear_all_interactions(self) -> None:
        """Remove all registered interactions from this stream."""
        self.interactions_by_uuid = {}
        self.limit_data_without_interactions()

    def clear_all_data(self) -> None:
        """Remove all stored data from this stream."""
        self.data_by_uuid = {}

    def get_interaction(self, uuid: str | None = None) -> "Interaction | None":
        """Get the interaction registered for the given UUID.

        Args:
            uuid: The UUID to look up.

        Returns:
            The Interaction object for that UUID, or None if not found.
        """
        if uuid not in self.interactions_by_uuid:
            return None
        else:
            return self.interactions_by_uuid[uuid]

    def get_last_added_uuid_in_interactions(self) -> object | None:
        """Return the most recently registered interaction.

        Returns:
            The last-added Interaction object, or None if no interactions are registered.
        """
        if len(self.interactions_by_uuid) == 0:
            return None
        else:
            uuid = next(reversed(self.interactions_by_uuid))
            return self.interactions_by_uuid[uuid]

    def get_last_added_uuid_in_data(self) -> object | None:
        """Return the most recently added Data struct.

        Returns:
            The last-added Data object, or None if no data has been stored.
        """
        if len(self.data_by_uuid) == 0:
            return None
        else:
            uuid = next(reversed(self.data_by_uuid))
            return self.data_by_uuid[uuid]

    def get_interactions(self) -> list:
        """Get the list of interactions involving this stream.

        Returns:
            List of Interaction objects.
        """
        return list(self.interactions_by_uuid.values())

    def set_props(self, data_stream: 'Stream'):
        """Set (edit) the data properties picking up the ones of another Stream.

        Args:
            data_stream (Stream): the source Stream from which DataProp is taken.

        Returns:
            None
        """
        self.props = data_stream.props

    def __len__(self):
        """Get the length of the stream.

        Returns:
            int: Infinity (as this is a lifelong stream).
        """
        return math.inf

    def __getitem__(self, idx_and_uuid: tuple[int, str | None]) -> tuple[torch.Tensor | Image | str | None, int]:
        """Get item for a specific clock cycle. Not implemented for base class: it will be implemented in buffered data
        streams or stream that can generate data on-the-fly.

        Args:
            idx_and_uuid (tuple[int, str | None]): Index of the data to retrieve and UUID (tuple).

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
        return self.to_code_str()


class BufferedStream(Stream):
    """
    Data stream with buffer support to store historical data.
    """

    def __init__(self, props: DataProps,
                 is_static: bool = False, is_queue: bool = False) -> None:
        """Initialize a BufferedStream.

        Args:
            props: The DataProps object describing the stream's data type and properties.
            is_static: If True, the buffer stores only one item that is reused.
            is_queue: If True, the buffer acts as a data queue.
        """
        super().__init__(props=props)

        # Properties
        self.is_static = is_static  # A static stream store only one sample and always yields it
        self.is_queue = is_queue
        self.is_read_only = False

        # We store the data samples, and we cache their text representation (for speed)
        self.data_buffer_by_uuid: dict[None | str, list] = {None: []}
        self.text_buffer_by_uuid: dict[None | str, list] = {None: []}

        # We need to remember the fist cycle in which we started buffering and the last one we buffered
        self.first_cycle_by_uuid: dict[None | str, int] = {None: -1}  # -1
        self.last_cycle_by_uuid: dict[None | str, int] = {None: -1}  # -1
        self.last_get_cycle_by_uuid: dict[None | str, int] = {None: -2}  # Keep it to -2 (-1 is starting cycle value)
        self.buffered_data_index_by_uuid: dict[None | str, int] = {None: -1}  # -1

        self.restart_before_next_get_by_uuid: dict[None | str, set] = {None: set()}  # set()

    def has_data(self, uuid: str | None) -> bool:
        """Check whether data is stored for the given UUID.

        Args:
            uuid: The UUID to look up.

        Returns:
            True if data exists for that UUID, False otherwise.
        """
        return uuid in self.data_buffer_by_uuid

    def get_data_uuids(self):
        """Return the set of UUIDs for which buffered data exists.

        Returns:
            A view of the keys in the internal data-buffer-by-UUID mapping.
        """
        return self.data_buffer_by_uuid.keys()

    def add_interaction(self, interaction: "Interaction") -> None:
        """Register an interaction and initialize per-UUID buffer structures if needed.

        Args:
            interaction: The Interaction object to register.
        """
        super().add_interaction(interaction)

        # What happens: buffered streams always have some data to provide.
        # However, they need to know what is the index of the data sample you are asking for, and this depends on the
        # UUID. What if you ask (get()) data for a never seen before UUID?
        # One might be tempted to say "ok, whenever a new UUID is asked in get(), we set its index to 0" and provide
        # the data.
        # This is not fine here.
        # In fact, we want get() to only return data that was already deposited by some processes in the past.
        # The past process which enables the data deposit is the presence of an interaction.
        # Hence, we eagerly anticipate the creation of the following data structures here.
        uuid = interaction.uuid
        if uuid not in self.data_buffer_by_uuid:
            self.data_buffer_by_uuid[uuid] = []
        if uuid not in self.text_buffer_by_uuid:
            self.text_buffer_by_uuid[uuid] = []
        if uuid not in self.first_cycle_by_uuid:
            self.first_cycle_by_uuid[uuid] = -1
        if uuid not in self.last_cycle_by_uuid:
            self.last_cycle_by_uuid[uuid] = -1
        if uuid not in self.last_get_cycle_by_uuid:
            self.last_get_cycle_by_uuid[uuid] = -2  # Keep it to -2 (since -1 is the starting value for cycles)
        if uuid not in self.buffered_data_index_by_uuid:
            self.buffered_data_index_by_uuid[uuid] = -1  # Keep it to -1, it will be incremented then used

    def get(self, requested_by: str | None = None, uuid: str | None = None, all_uuids: bool = False) \
            -> str | Image | Tensor | None | list[tuple[str | Image | Tensor | None, int, float]]:
        """Get the current data sample based on cycle and buffer.

        Args:
            requested_by: Identifier of the caller for per-caller deduplication.
            uuid: UUID of the interaction to retrieve data for (defaults to the last set UUID).
            all_uuids: Whether to return a list with all the data samples for all the existing UUIDs (default: False).
                In this case, the uuid argument is ignored.

        Returns:
            The current buffered sample, or None if no new data is available for this caller.
            If all_uuids is True, then it returns a list of tuples (data_sample, tag, timestamp),
                where each data_sample has the just mentioned properties.
        """
        if all_uuids:
            samples = []
            for _uuid in list(self.get_data_uuids()):  # Keep "list(...)" to avoid a rare but possible issue
                sample = self.get(requested_by, _uuid)
                if sample is None:
                    continue
                tag = self.get_tag(_uuid)
                timestamp = self.get_timestamp(_uuid)
                samples.append((sample, tag, timestamp))
            return samples

        if (requested_by is not None and uuid in self.restart_before_next_get_by_uuid and
                requested_by in self.restart_before_next_get_by_uuid[uuid]):
            self.restart_before_next_get_by_uuid[uuid].remove(requested_by)
            if len(self.restart_before_next_get_by_uuid[uuid]) == 0:
                del self.restart_before_next_get_by_uuid[uuid]
            self.restart(uuid)

        # Auto-anchor `first_cycle_by_uuid[uuid]` on the very first get for any non-None uuid, so subclass
        # __getitem__ tag formulas (e.g. ImageFileStream's `clock.get_cycle() - first_cycle_by_uuid[uuid]`)
        # emit tag=0 at that first read and walk 0, 1, 2, … from there — independent of when the uuid was
        # registered relative to first publish. Replaces the explicit `plan_restart_before_next_get` pattern.
        # Detected via `last_get_cycle_by_uuid[uuid] == -2` (the sentinel set by `add_interaction`, only ever
        # overwritten by a successful get below).
        if uuid is not None and self.last_get_cycle_by_uuid.get(uuid, -2) == -2:
            self.restart(uuid)

        cycle = clock.get_cycle()

        if uuid in self.data_by_uuid:
            first = False
            data_struct = self.data_by_uuid[uuid]
        else:
            first = True
            data_struct = None

        # These two lines might make you think "hey, call super().set(self[cycle]), it is the same!"
        # however, it is not like that, since "set" will also call "adapt_to_labels", that is not needed for
        # buffered streams
        if (first or (self.last_get_cycle_by_uuid[uuid] != cycle and
                      (self.props.delta <= 0. or
                       self.props.delta <=
                       (clock.get_time() - data_struct.data_timestamp)))):  # type: ignore[union-attr]
            self.last_get_cycle_by_uuid[uuid] = cycle

            if not self.is_queue:
                if uuid in self.buffered_data_index_by_uuid:
                    self.buffered_data_index_by_uuid[uuid] += 1
                    idx = self.buffered_data_index_by_uuid[uuid]
                else:
                    idx = -1
                new_data, new_tag = self[(idx, uuid)]
            else:
                new_data, new_tag = self[(0, uuid)]
                if new_data is not None:
                    self.data_buffer_by_uuid[uuid].pop(0)
                    self.text_buffer_by_uuid[uuid].pop(0)

            if new_tag >= 0:  # When tag == -1, the data was not present at all
                if first:
                    data_struct = Data(uuid=uuid)
                    self.limit_data_without_interactions()
                    self.data_by_uuid[uuid] = data_struct
                data_struct.data = new_data
                data_struct.data_timestamp = clock.get_cycle_time()
                data_struct.data_tag = new_tag
        return super().get(requested_by, uuid)

    def set(self, data: torch.Tensor | Image | str,
            data_tag: int = -1, keep_existing_tag: bool = False, uuid: str | None = None, force: bool = False) \
            -> Data | None:
        """Store a new data sample into the buffer.

        Args:
            data: Data to store (tensor, PIL image, or string).
            data_tag: Custom data time tag >= 0 (Default: -1, meaning no tags).
            keep_existing_tag: Keep the data tag that is already in the stream, if the provided data_tag arg is -1.
            uuid: UUID of the data.
            force: Boolean flag to force the set operation also if the stream is disabled.

        Returns:
            The updated Data struct if the data was buffered, else None.
        """
        if not self.enabled and not force:
            return None

        if self.is_read_only:
            return None

        if self.is_static:
            uuid = None

        if self.is_queue and data is None:
            return None

        data_struct = super().set(data, data_tag, keep_existing_tag, uuid, force)

        if data_struct:
            if uuid not in self.data_buffer_by_uuid:
                self.data_buffer_by_uuid[uuid] = []
                self.text_buffer_by_uuid[uuid] = []
                self.first_cycle_by_uuid[uuid] = -1
                self.last_cycle_by_uuid[uuid] = -1
                self.last_get_cycle_by_uuid[uuid] = -2  # Keep it to -2 (since -1 is the starting value for cycles)
                self.buffered_data_index_by_uuid[uuid] = -1

            if not self.is_static or len(self.data_buffer_by_uuid[uuid]) == 0:
                self.data_buffer_by_uuid[uuid].append(Data(data_struct))
                if self.props.is_flat_tensor_with_labels():
                    self.text_buffer_by_uuid[uuid].append(self.props.to_text(data))
                elif self.props.is_text():
                    self.text_buffer_by_uuid[uuid].append(data)
                else:
                    self.text_buffer_by_uuid[uuid].append("")

                # Boilerplate
                if self.first_cycle_by_uuid[uuid] < 0:
                    self.first_cycle_by_uuid[uuid] = clock.get_cycle()
                self.last_cycle_by_uuid[uuid] = clock.get_cycle()
        return data_struct

    def __getitem__(self, idx_and_uuid: tuple[int, str | None]) -> tuple[torch.Tensor | None, int]:
        """Retrieve a sample from the buffer based on the given clock cycle.

        Args:
            idx_and_uuid (tuple composed of an int and a string): Index (>=0) of the sample and UUID (tuple).

        Returns:
            torch.Tensor | None: The sample, if available.
        """
        idx, uuid = idx_and_uuid
        if self.is_read_only or self.is_static:
            uuid = None
        if not self.is_static:
            if uuid not in self.data_buffer_by_uuid or idx >= len(self.data_buffer_by_uuid[uuid]) or idx < 0:
                return None, -1
            data_struct = self.data_buffer_by_uuid[uuid][idx]
            data = data_struct.data if data_struct is not None else None
            data_tag = data_struct.data_tag if data_struct is not None else -1
        else:
            data_struct = self.data_buffer_by_uuid[uuid][0]
            data = data_struct.data if data_struct is not None else None
            
            # Static streams have a single sample with a fixed data tag, and that is not a valid tag to use multiple
            # times, since the repeated tag will trigger a rejection on the receiver's side.
            # We have to rely on the clock.
            # Hence, the line below, data_tag = -1, will trigger the fall-through on the "return" line,
            # actually setting data_tag to clock.get_cycle() - self.first_cycle_by_uuid[uuid]
            data_tag = -1
        return data, data_tag if data_tag >= 0 else clock.get_cycle() - self.first_cycle_by_uuid[uuid]

    def __len__(self):
        """Get number of samples in the buffer (max among UUIDs).

        Returns:
            int: The max number of buffered samples (across UUIDs).
        """
        m = 0
        for z in self.data_buffer_by_uuid.values():
            m = max(m, len(z))
        return m

    def set_first_cycle(self, cycle: int, uuid: str | None = None) -> None:
        """Manually set the first cycle for the buffer and adjust the last-cycle counter.

        Args:
            cycle: Global cycle number to start from.
            uuid: UUID of the data (defaults to the global/no-UUID slot).
        """
        self.first_cycle_by_uuid[uuid] = cycle
        self.last_cycle_by_uuid[uuid] = cycle + len(self.data_buffer_by_uuid[uuid])

    def get_first_cycle(self, uuid: str | None = None) -> int:
        """Return the first buffered cycle for the given UUID.

        Args:
            uuid: UUID of the data (defaults to the global/no-UUID slot).

        Returns:
            The first cycle value (-1 if not yet set).
        """
        return self.first_cycle_by_uuid[uuid]

    def restart(self, uuid: str | None = None) -> None:
        """Restart the buffer read-position to the current clock cycle.

        Args:
            uuid: UUID of the data to restart (defaults to the global/no-UUID slot).
        """
        if uuid in self.buffered_data_index_by_uuid:
            self.set_first_cycle(max(clock.get_cycle(), 0), uuid)
            self.buffered_data_index_by_uuid[uuid] = -1
            self.last_get_cycle_by_uuid[uuid] = -2
            if uuid in self.data_by_uuid:
                del self.data_by_uuid[uuid]

    def edit_uuid_in_data(self, current_uuid: str, new_uuid: str) -> None:
        """Rename a UUID across all buffer structures (used when a temporary UUID is replaced).

        Args:
            current_uuid: The existing UUID to rename.
            new_uuid: The replacement UUID.
        """
        super().edit_uuid_in_data(current_uuid, new_uuid)

        edit_what = [self.data_buffer_by_uuid, self.text_buffer_by_uuid, self.first_cycle_by_uuid,
                     self.last_cycle_by_uuid, self.last_get_cycle_by_uuid, self.buffered_data_index_by_uuid,
                     self.restart_before_next_get_by_uuid]

        for edit_this in edit_what:
            if current_uuid in edit_this:
                edit_this[new_uuid] = edit_this[current_uuid]
                del edit_this[current_uuid]

        if new_uuid in self.data_buffer_by_uuid:
            for data in self.data_buffer_by_uuid[new_uuid]:
                data.uuid = new_uuid

    def remove_data(self, uuid: str | None) -> None:
        """Remove data from this stream.

        Args:
            uuid: The UUID of the object to remove.
        """
        super().remove_data(uuid)

        remove_from_what = [self.data_buffer_by_uuid, self.text_buffer_by_uuid, self.first_cycle_by_uuid,
                            self.last_cycle_by_uuid, self.last_get_cycle_by_uuid, self.buffered_data_index_by_uuid,
                            self.restart_before_next_get_by_uuid]

        for remove_from_this in remove_from_what:
            if uuid in remove_from_this:
                del remove_from_this[uuid]

    def plan_restart_before_next_get(self, requested_by: str, uuid: str | None = None) -> None:
        """Schedule a buffer restart to happen just before the next ``get()`` call by the given caller.

        Args:
            requested_by: Identifier of the caller that will trigger the restart.
            uuid: UUID of the buffer to restart (defaults to the global/no-UUID slot).
        """
        if uuid not in self.restart_before_next_get_by_uuid:
            self.restart_before_next_get_by_uuid[uuid] = set()
        self.restart_before_next_get_by_uuid[uuid].add(requested_by)

    def clear_buffer(self):
        """Clear the data buffer
        """

        # We store the data samples, and we cache their text representation (for speed)
        self.data_buffer_by_uuid = {None: []}
        self.text_buffer_by_uuid = {None: []}

        # We need to remember the fist cycle in which we started buffering and the last one we buffered
        self.first_cycle_by_uuid = {None: -1}  # -1
        self.last_cycle_by_uuid = {None: -1}  # -1
        self.last_get_cycle_by_uuid = {None: -2}  # -2  # Keep it to -2 (since -1 is the starting value for cycles)
        self.buffered_data_index_by_uuid = {None: -1}  # -1

        self.restart_before_next_get_by_uuid = {None: set()}  # set()

    def shuffle_buffer(self, seed: int = -1) -> None:
        """Shuffle the data (and text) buffer in-place for each UUID.

        Args:
            seed: Random seed for reproducibility. Use -1 (default) for a non-seeded shuffle.
        """
        for uuid in self.data_buffer_by_uuid.keys():
            old_buffer = self.data_buffer_by_uuid[uuid]
            indices = list(range(len(old_buffer)))

            state = random.getstate()
            if seed >= 0:
                random.seed(seed)
            random.shuffle(indices)
            if seed >= 0:
                random.setstate(state)

            self.data_buffer_by_uuid[uuid] = []
            k = 0
            for i in indices:
                data_struct = Data(old_buffer[i])
                data_struct.data_tag = old_buffer[k].data_tag
                self.data_buffer_by_uuid[uuid].append(data_struct)
                k += 1

            if (self.text_buffer_by_uuid is not None and
                    len(self.text_buffer_by_uuid[uuid]) == len(self.data_buffer_by_uuid[uuid])):
                old_text_buffer = self.text_buffer_by_uuid[uuid]
                self.text_buffer_by_uuid[uuid] = []
                for i in indices:
                    self.text_buffer_by_uuid[uuid].append(old_text_buffer[i])

    def to_text_snippet(self, length: int | None = None, uuid: str | None = None) -> str | None:
        """Convert buffered text samples to a single long string.

        Args:
            length: Optional maximum character length of the resulting text snippet.
            uuid: UUID of the data.

        Returns:
            Human-readable text sequence, or None if no text data is available.
        """
        if (self.text_buffer_by_uuid is not None and uuid in self.text_buffer_by_uuid and
                len(self.text_buffer_by_uuid[uuid]) > 0):
            text_buffer = self.text_buffer_by_uuid[uuid]
            if length is not None:
                le = max(length // 2, 1)
                text = " ".join(text_buffer[0:min(le, len(text_buffer))])
                text += (" ... " + (" ".join(text_buffer[max(le, len(text_buffer) - le):]))) \
                    if len(text_buffer) > le else ""
            else:
                text = " ".join(text_buffer)
        else:
            text = None
        return text

    def get_since_timestamp(self, since_what_timestamp: float, stride: int = 1, uuid: str | None = None) -> (
            tuple[list[int] | None, list[torch.Tensor | None] | None, int, DataProps]):
        """Retrieve all samples starting from a given timestamp.

        Args:
            since_what_timestamp (float): Timestamp in seconds.
            stride (int): Sampling stride.
            uuid: UUID of the data.

        Returns:
            Tuple containing list of cycles, data, current cycle, and data properties.
        """
        since_what_cycle = clock.time2cycle(since_what_timestamp)
        return self.get_since_cycle(since_what_cycle, stride, uuid)

    def get_since_cycle(self, since_what_cycle: int, stride: int = 1, uuid: str | None = None) -> (
            tuple[list[int] | None, list[torch.Tensor | None] | None, int, DataProps]):
        """Retrieve all samples whose data_tag (cycle when written) is >= since_what_cycle.

        Args:
            since_what_cycle (int): Cycle number.
            stride (int): Minimum cycle gap between consecutive returned samples.
            uuid: UUID of the data.

        Returns:
            Tuple with cycles, data, current cycle, and properties.
        """
        assert stride >= 1 and isinstance(stride, int), f"Invalid stride: {stride}"

        global_cycle = clock.get_cycle()
        if global_cycle < 0:
            return None, None, -1, self.props

        since_what_cycle = max(since_what_cycle, 0)
        ret_cycles: list[int] = []
        ret_data: list = []

        if uuid in self.data_buffer_by_uuid:
            last_kept = since_what_cycle - stride  # ensure the first eligible sample is kept
            for d in self.data_buffer_by_uuid[uuid]:
                if d.data is None or d.data_tag < since_what_cycle:
                    continue
                if d.data_tag - last_kept < stride:
                    continue
                ret_cycles.append(d.data_tag)
                ret_data.append(d.data)
                last_kept = d.data_tag

        return ret_cycles, ret_data, global_cycle, self.props


class Dataset(BufferedStream):
    """
    A buffered dataset that streams data from a PyTorch dataset and simulates data-streams for input/output.
    """

    def __init__(self, tensor_dataset, shape: tuple, index: int = 0, batch_size: int = 1) -> None:
        """Initialize a Dataset instance, which wraps around a PyTorch Dataset.

        Args:
            tensor_dataset: The PyTorch Dataset to wrap (must support ``len()`` and indexed access).
            shape: The shape of each sample from the data stream.
            index: The index of the element returned by ``__getitem__`` to pick up (default: 0).
            batch_size: Number of samples to group into each batch (default: 1).
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
            base = i * b
            if i == (nb - 1):
                b = r

            for j in range(0, b):
                sample = tensor_dataset[base + j][index]
                if isinstance(sample, (int, float)):
                    sample = torch.tensor(sample, dtype=dtype)
                batch.append(sample)

            super().set(torch.stack(batch), data_tag=i)

        # It was buffered previously than every other thing
        self.is_read_only = True
        self.restart()


class ImageFileStream(BufferedStream):
    """
    A buffered dataset for image data.
    """

    def __init__(self, image_dir: str, list_of_image_files: str | list[str],
                 device: torch.device | None = None, circular: bool = True, show_images: bool = False):
        """Initialize an ImageFileStream instance for streaming image data.

        Args:
            image_dir: The directory containing image files.
            list_of_image_files: Path to the file with list of file names of the images (one per line);
                it can also be the path to a list of file names.
            device: The device to store the tensors on. Default is CPU.
            circular: Whether to loop the dataset or not. Default is True.
            show_images: If True, display a clickable grid of the images. Default is False.
        """
        self.image_dir = image_dir
        # noinspection PyCallingNonCallable
        self.device = device if device is not None else torch.device("cpu")
        self.circular = circular

        # Reading the image file
        # (assume a file with one filename per line or a CSV format with lines such as: cat.jpg,cat,mammal,animal)
        self.image_paths = []

        # Calling the constructor
        super().__init__(props=DataProps(name=ImageFileStream.__name__,
                                         data_type="img",
                                         pubsub=True))
        self.is_read_only = True

        if isinstance(list_of_image_files, str):
            with open(list_of_image_files, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')  # Tolerates if it is a CVS and the first field is the file name
                    image_name = parts[0]
                    self.image_paths.append(os.path.join(image_dir, image_name))
        else:
            for image_name in list_of_image_files:
                self.image_paths.append(os.path.join(image_dir, image_name))

        # It was buffered previously than every other thing (not strictly required due to the fact that
        # BufferedStream's now force restart at the first get
        self.last_cycle_by_uuid[None] = -1
        self.first_cycle_by_uuid[None] = self.last_cycle_by_uuid[None] - len(self.image_paths) + 1

        # Possibly print to screen the "clickable" list of images
        if show_images:
            show_images_grid(self.image_paths)

    def add(self, image_file_name: str):
        """Adds a new image to the set."""
        self.image_paths.append(image_file_name)

    def __len__(self):
        """Return the number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx_and_uuid: tuple[int, str | None]) -> tuple[torch.Tensor | Image | str | None, int]:
        """Get an image.

        Args:
            idx_and_uuid (tuple[int, str | None]): The index to retrieve data for and its UUID (tuple).

        Returns:
            tuple: A tuple (image, idx).
        """
        idx, uuid = idx_and_uuid
        if self.circular:
            idx %= self.__len__()
        else:
            if idx >= self.__len__() or idx < 0:
                return None, -1

        image = PIL.Image.open(self.image_paths[idx])
        return image, clock.get_cycle() - self.first_cycle_by_uuid[uuid]


class LabelStream(BufferedStream):
    """
    A buffered stream for single and multi-label annotations.
    """

    def __init__(self, label_dir: str, label_file_csv: str,
                 device: torch.device | None = None, circular: bool = True, single_class: bool = False,
                 line_header: bool = False):
        """Initialize a LabelStream instance for streaming labels.

        Args:
            label_dir: The directory containing label files.
            label_file_csv: Path to the CSV file with labels (format: ``filename,label1,label2,...``).
            device: The device to store the tensors on. Default is CPU.
            circular: Whether to loop the dataset or not. Default is True.
            single_class: Whether to only consider a single class for labeling. Default is False.
            line_header: If True, treat the first field as a header to skip. Default is False.
        """
        self.label_dir = label_dir
        # noinspection PyCallingNonCallable
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
        self.is_read_only = True

        for idx, class_name in enumerate(class_names):
            class_name_to_index[class_name] = idx

        with open(label_file_csv, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                label = parts[0:] if not line_header else parts[1:]
                target_vector = torch.zeros((1, len(class_names)), dtype=torch.float32)
                for lab in label:
                    idx = class_name_to_index[lab]
                    target_vector[0, idx] = 1.
                self.labels.append(target_vector)

        # It was buffered previously than every other thing
        self.last_cycle_by_uuid[None] = -1
        self.first_cycle_by_uuid[None] = self.last_cycle_by_uuid[None] - len(self.labels) + 1

    def __len__(self):
        """Return the number of labels in the dataset.

        Returns:
            int: Number of labels in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx_and_uuid: tuple[int, str | None]) -> tuple[torch.Tensor | Image | str | None, int]:
        """Get item for a specific clock cycle. Not implemented for base class: it will be implemented in buffered data
        streams or stream that can generate data on-the-fly.

        Args:
            idx_and_uuid (tuple[int, str | None]): Index of the data to retrieve and UUID (tuple).

        Returns:
            tuple: A tuple (image, idx) for the specified cycle.
        """
        idx, uuid = idx_and_uuid
        if self.circular:
            idx %= self.__len__()
        else:
            if idx >= self.__len__() or idx < 0:
                return None, -1

        label = self.labels[idx].to(self.device)  # Multi-label vector for the label
        return self.props.adapt_tensor_to_tensor_labels(label), clock.get_cycle() - self.first_cycle_by_uuid[uuid]


class StringStream(ImageFileStream):
    """
    A buffered dataset for strings.
    """
    def __init__(self, strings: list[str], circular: bool = True):
        super().__init__(image_dir="./", list_of_image_files=strings, circular=circular)

    def __getitem__(self, idx_and_uuid: tuple[int, str | None]) -> tuple[str | None, int]:
        """Get a string.

        Args:
            idx_and_uuid (tuple[int, str | None]): The index to retrieve data for and its UUID (tuple).

        Returns:
            tuple: A tuple (str, idx).
        """
        idx, uuid = idx_and_uuid
        if self.circular:
            idx %= self.__len__()
        else:
            if idx >= self.__len__() or idx < 0:
                return None, -1

        return self.image_paths[idx], clock.get_cycle() - self.first_cycle_by_uuid[uuid]


class TokensStream(BufferedStream):
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
            super().set(token, data_tag=i)

        # It was buffered previously than every other thing
        self.is_read_only = True
        self.restart()


def serialize_payload(data_list: list) -> str:
    """Serialize a list of tensors, PIL images, and/or strings to a JSON string.

    Args:
        data_list: List of values to serialize; each element must be a ``torch.Tensor``,
            a PIL ``Image``, or a ``str``.

    Returns:
        A JSON-encoded string representation of the payload.

    Raises:
        GenException: If any element in ``data_list`` is of an unsupported type.
    """
    payload = []

    for value in data_list:

        # Handle PyTorch Tensors
        if isinstance(value, torch.Tensor):
            buffer = io.BytesIO()
            torch.save(value, buffer)
            b64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            payload.append({"type": Stream.tensor, "data": b64_data})

        # Handle Pillow Images
        elif isinstance(value, Image):
            buffer = io.BytesIO()
            value.save(buffer, format="PNG")  # Use PNG for lossless
            b64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            payload.append({"type": Stream.img, "data": b64_data})

        # Handle standard strings
        elif isinstance(value, str):
            payload.append({"type": Stream.text, "data": value})

        else:
            raise GenException(f"Cannot serialize object of type {type(value)}")

    return json.dumps(payload)


def deserialize_payload(json_str: str) -> list:
    """Deserialize a JSON payload string back into a list of tensors, PIL images, and/or strings.

    Args:
        json_str: A JSON string previously produced by :func:`serialize_payload`.

    Returns:
        A list of deserialized objects (``torch.Tensor``, PIL ``Image``, or ``str``).

    Raises:
        GenException: If an item declares an unrecognized type.
    """
    payload = json.loads(json_str)
    output = []

    for item in payload:
        data = item["data"]
        i_type = item["type"]

        if i_type == Stream.tensor:
            buffer = io.BytesIO(base64.b64decode(data))
            output.append(torch.load(buffer, weights_only=True))
        elif i_type == Stream.img:
            buffer = io.BytesIO(base64.b64decode(data))
            output.append(PIL.Image.open(buffer).copy())
        elif i_type == Stream.text:
            output.append(data)
        else:
            raise GenException(f"Cannot deserialize object declared to be of type {i_type}")

    return output
