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
from itertools import islice
from unaiverse.streams.streams import Stream
from unaiverse.utils.logger import log
from unaiverse.utils.misc import GenException
from unaiverse.streams.dataprops import DataProps


class StreamProxy:
    """Proxy that wraps a set of streams for unified stdin/stdout access.

    Provides a consistent interface for getting/setting data on streams
    by name, index, or implicitly (when there is only one stream).
    """

    def __init__(self, streams: dict[str, Stream | object] | None = None):
        """Create a StreamIO proxy.

        Args:
            streams: Optional dict mapping stream name to stream object.
        """
        self._streams: dict[str, Stream | object] = streams if streams is not None else {}
        self._stream_list: list = list(self._streams.values())
        self._streams_by_name_only: dict[str, Stream | object] = {}
        for stream_hash, stream_obj in self._streams.items():
            name_only = DataProps.name_from_user_hash(stream_hash)
            if name_only not in self._streams_by_name_only:  # In case of collision, the first name wins
                self._streams_by_name_only[DataProps.name_from_user_hash(stream_hash)] = stream_obj
        self._uuid: str | None = None

    def bind(self, streams: dict[str, Stream | object], uuid: str | None = None):
        """Rebind this proxy to a different set of streams (different from the ones used when building it).

        Args:
            streams: Dict mapping stream name to stream object.
            uuid: The default UUID to look for when doing get/set.
        """
        self._streams = streams.copy()  # Shallow copy
        self._stream_list = list(streams.values())
        self._streams_by_name_only = {}
        for stream_hash, stream_obj in self._streams.items():

            # Warning: 'stream_hash' could be user hash or <default_pos...>
            name_only = DataProps.name_from_user_hash(stream_hash) if Stream.is_user_hash(stream_hash) else stream_hash
            if name_only not in self._streams_by_name_only:  # In case of collision, the first name wins
                self._streams_by_name_only[name_only] = stream_obj
        if uuid is not None:
            self._uuid = uuid

    def add_new_bind(self, stream_hash: str, stream: Stream) -> None:
        """Register an additional stream under the given hash, if not already bound.

        Args:
            stream_hash: The key (typically a user hash) to associate with the stream.
            stream: The Stream object to add.
        """
        if stream_hash not in self._streams:
            self._streams[stream_hash] = stream
            self._stream_list.append(stream)
            name_only = DataProps.name_from_user_hash(stream_hash)
            if name_only not in self._streams_by_name_only:  # In case of collision, the first name wins
                self._streams_by_name_only[name_only] = stream

    def clear_data(self, uuid: str | None):
        """Remove all the currently existing data from the bind streams (not the interactions)."""

        for stream_obj in self._stream_list:
            stream_obj.clear_data(uuid=uuid)

    def clear_all_data(self):
        """Remove all the currently existing data from the bind streams (not the interactions)."""

        for stream_obj in self._stream_list:
            stream_obj.clear_all_data()

    def enable(self) -> None:
        """Enable all streams in this proxy stream, allowing data to be set."""
        for stream_obj in self._stream_list:
            stream_obj.enable()

    def disable(self) -> None:
        """Disable all streams in this proxy, preventing data from being set."""
        for stream_obj in self._stream_list:
            stream_obj.disable()

    def get(self, key: str | int | None = None, requested_by: str | None = None, uuid: str | None = None,
            all_uuids: bool = False, data_type: str | None = None):
        """Get data from a stream.

        Args:
            key: Stream name (str), index (int), or None to retrieve from all streams.
            requested_by: The identifier of who requests access to the stream data (default: None).
            uuid: UUID of the interaction to retrieve data for (default: None, that will use the default-bind UUID).
            all_uuids: If True, data from all the existing UUIDs (no matter what) are returned.
                It is valid if and only if paired with the selection of a specific data_type or of key.
            data_type: The string name of the data type of interest. If multiple streams with the same data_type
                are bind, the first found one is returned.

        Returns:
            The data from the requested stream, or None if no new data is available.
            If all_uuids is True, returns a list of tuples, (data, data_tag).
        """
        if len(self._stream_list) == 0:
            return None

        if uuid is None:
            uuid = self._uuid

        if all_uuids and (not data_type and not key):
            raise GenException("You can only ask for all UUIDs if you also also specify a "
                               "stream name (key) or a data type")

        if key and data_type:
            raise GenException("You can only specify a stream name (key) or a data type, not both")

        if key is None:
            found_at_least_one = False
            ret = []
            for s in self._stream_list:
                if isinstance(s, Stream):
                    if data_type and s.props.data_type != data_type:
                        continue
                    data = s.get(requested_by, uuid, all_uuids)  # This might will be None if requested multiple times
                    if data is not None:
                        found_at_least_one = True
                    ret.append(data)
                else:
                    ret.append(s)  # This will always be not-None (well, unless None is the actual value)
            return ret if found_at_least_one else None
        elif isinstance(key, int):
            if isinstance(self._stream_list[key], Stream):
                return self._stream_list[key].get(requested_by, uuid, all_uuids)
            else:
                return self._stream_list[key]  # Default value
        elif key in self._streams:
            if self._streams[key] is not None:
                return self._streams[key].get(requested_by, uuid, all_uuids)
            else:
                return self._streams[key]  # Default value
        elif key in self._streams_by_name_only:
            if self._streams_by_name_only[key] is not None:
                return self._streams_by_name_only[key].get(requested_by, uuid, all_uuids)
            else:
                return self._streams_by_name_only[key]  # Default value
        else:
            raise GenException(f"Unknown stream/key: {key}"
                               f"\n{self._stream_list}\n{self._streams}\n{self._streams_by_name_only}")  # TODO remove

    def add_interaction(self, interaction) -> None:
        """Register an interaction on all bound streams.

        Args:
            interaction: The Interaction object to register.
        """
        for s in self._stream_list:
            s.add_interaction(interaction)

    def has_interaction(self, uuid: str | None) -> bool:
        """Check whether all bound streams have an interaction for the given UUID.

        Args:
            uuid: The UUID to look up.

        Returns:
            True if every bound stream has an interaction for that UUID, False otherwise.
        """
        for s in self._stream_list:
            if not isinstance(s, Stream):
                continue
            if not s.has_interaction(uuid):
                return False
        return True

    def get_interaction(self, uuid: str | None) -> object | None:
        """Retrieve the interaction for the given UUID from the first stream that has one.

        Args:
            uuid: The UUID of the interaction to retrieve.

        Returns:
            The Interaction object if found, else None.
        """
        if not self.has_interaction(uuid):
            return None

        for s in self._stream_list:
            if not isinstance(s, Stream):
                continue
            interaction = s.get_interaction(uuid)
            if interaction is not None:
                return interaction  # This will happen at the 1st iteration for the way "has_interaction" is implemented
        return None  # This will never happen, since we already ensured that it "has_interaction"

    def set(self, key_or_data, data=None, data_tag: int = -1, uuid: str | None = None, force: bool = False):
        """Set data on a stream.

        Usage (example for stdin - it could be other StreamIO):
            - ``self.stdin.set(data)`` — when there is only one stream
            - ``self.stdin.set(stream_name, data)`` — by name
            - ``self.stdin.set(stream_index, data)`` — by index

        Args:
            key_or_data: Stream name/index if ``data`` is also provided, the data itself
                for the single-stream case, or a list of values to set on all streams at once.
            data: The data to set (when ``key_or_data`` is a name/index).
            data_tag: Custom integer tag for the sample (default: -1, meaning auto-tag).
            uuid: UUID of the interaction for which data is being set (default: None, that will use the
                default-bind UUID)
            force: Boolean flag to force the set operation also on disabled streams
        """
        if len(self._stream_list) == 0:
            raise GenException("No streams bound to this StreamIO")

        if uuid is None:
            uuid = self._uuid

        if isinstance(key_or_data, (list, tuple)):
            if len(key_or_data) != len(self._stream_list):
                raise GenException(f"The list of stream values to set must have the same length of the stream list "
                                   f"({len(key_or_data)} != {len(self._stream_list)})")
            for i, data in enumerate(key_or_data):
                s = self._stream_list[i]
                if isinstance(s, Stream):
                    s.set(data, data_tag, uuid=uuid, force=force)
                else:
                    self._stream_list[i] = data
                    key_at_index = next(islice(self._streams, i, None))
                    self._streams[key_at_index] = data
        elif isinstance(key_or_data, int):
            s = self._stream_list[key_or_data]
            if isinstance(s, Stream):
                s.set(data, uuid=uuid, force=force)
            else:
                self._stream_list[key_or_data] = data
        elif key_or_data in self._streams:
            s = self._streams[key_or_data]
            if isinstance(s, Stream):
                s.set(data, data_tag, uuid=uuid, force=force)
            else:
                self._streams[key_or_data] = data
        else:
            raise GenException(f"Unknown stream: {key_or_data}")

    def get_tag(self, key: str | int | None = None, uuid: str | None = None) -> int:
        """Return the data tag for the given stream key and UUID.

        Args:
            key: Stream name (str), index (int), or None to return the max tag across all streams.
            uuid: UUID of the interaction to query (default: None, that will use the default-bind UUID).

        Returns:
            The integer data tag, or -1 if not applicable.

        Raises:
            GenException: If no streams are bound or the key is not found.
        """
        if len(self._stream_list) == 0:
            raise GenException("No streams bound to this StreamIO")

        if uuid is None:
            uuid = self._uuid

        if key is None:
            ret = []
            for s in self._stream_list:
                if isinstance(s, Stream):
                    data_tag = s.get_tag(uuid)
                    if data_tag is None:
                        return -1
                    ret.append(data_tag)
                else:
                    ret.append(-1)
            if len(ret) == 0:
                return -1
            return max(ret)
        elif isinstance(key, int):
            if isinstance(self._stream_list[key], Stream):
                return self._stream_list[key].get_tag(uuid)
            else:
                return -1  # Default value
        elif key in self._streams:
            if self._streams[key] is not None:
                return self._streams[key].get_tag(uuid)
            else:
                return -1  # Default value
        else:
            raise GenException(f"Unknown stream: {key}")

    def clear(self) -> None:
        """Reset all bound streams by setting their data to None."""
        self.set([None] * len(self))

    def __getitem__(self, key: str | int):
        """Retrieve data from the stream identified by ``key``.

        Args:
            key: Stream name or index.
        """
        return self.get(key)

    def __len__(self) -> int:
        """Return the number of bound streams."""
        return len(self._stream_list)

    def __iter__(self):
        """Iterate over the bound stream objects.

        Returns:
            An iterator over the stream list.
        """
        return iter(self._stream_list)

    def __contains__(self, key: object) -> bool:
        """Check whether a stream name is bound to this proxy.

        Args:
            key: A stream name string to look up.

        Returns:
            True if ``key`` is a bound stream name, False otherwise.
        """
        if isinstance(key, str):
            return key in self._streams or key in self._streams_by_name_only
        return False

    def items(self):
        """Iterate over ``(stream_name, stream)`` pairs for all bound streams.

        Returns:
            An iterator of ``(key, value)`` pairs.
        """
        for key, value in self._streams.items():
            yield key, value

    @property
    def names(self) -> list[str]:
        """Return the names (user hashes) of all bound streams."""
        return list(self._streams.keys())

    def __str__(self) -> str:
        """Return a compact human-readable summary of the bound streams, grouped by peer.

        Returns:
            A comma-separated string of ``peer:stream1,stream2`` entries, or ``"no-streams"`` if empty.
        """
        grouped_by_user = {}
        for user_hash in self._streams.keys():
            if user_hash.startswith("<default"):
                user = "owner"
                name = user_hash
            else:
                user = DataProps.peer_id_from_user_hash(user_hash)
                name = DataProps.name_from_user_hash(user_hash)
            if user not in grouped_by_user:
                grouped_by_user[user] = []
            grouped_by_user[user].append(name)
        if len(grouped_by_user) > 0:
            return str(", ".join([(user + ":" + ",".join(stream_names))
                                  for user, stream_names in grouped_by_user.items()]))
        else:
            return "no-streams"


class StreamsProxyWithDefaults(StreamProxy):
    def __init__(self, *args, default_values=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_values = default_values

    def get(self, *args, **kwargs):
        data_list = super().get(*args, **kwargs)
        if data_list is not None and self.default_values is not None:
            for i, data in enumerate(data_list):
                if data is None:
                    data_list[i] = self.default_values[i]
        return data_list
