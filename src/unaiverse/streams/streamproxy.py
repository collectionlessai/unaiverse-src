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

    def bind(self, streams: dict[str, Stream | object]):
        """Rebind this proxy to a different set of streams (different from the ones used when building it).

        Args:
            streams: Dict mapping stream name to stream object.
        """
        self._streams = streams.copy()  # Shallow copy
        self._stream_list = list(streams.values())

    def add_new_bind(self, stream_hash: str, stream: Stream):
        if stream_hash not in self._streams:
            self._streams[stream_hash] = stream
            self._stream_list.append(stream)

    def get(self, key: str | int | None = None, requested_by: str | None = None, uuid: str | None = None):
        """Get data from a stream.

        Args:
            key: Stream name (str), index (int), or None for single-stream case.
            requested_by (str | None): The identifier of "who" requests access to the stream data (default: None).

        Returns:
            The data from the requested stream.
        """
        if len(self._stream_list) == 0:
            raise GenException("No streams bound to this StreamIO")

        if key is None:
            found_at_least_one = False
            ret = []
            for s in self._stream_list:
                if isinstance(s, Stream):
                    data = s.get(requested_by, uuid)  # This might will be None if requested multiple times
                    if data is not None:
                        found_at_least_one = True
                    ret.append(data)
                else:
                    ret.append(s)  # This will always be not-None (well, unless None is the actual value)
            return ret if found_at_least_one else None
        elif isinstance(key, int):
            if isinstance(self._stream_list[key], Stream):
                return self._stream_list[key].get(requested_by, uuid)
            else:
                return self._stream_list[key]  # Default value
        elif key in self._streams:
            if self._streams[key] is not None:
                return self._streams[key].get(requested_by, uuid)
            else:
                return self._streams[key]  # Default value
        else:
            raise GenException(f"Unknown stream: {key}")

    def add_interaction(self, interaction: 'Interaction'):
        for s in self._stream_list:
            s.add_interacton(interaction)

    def has_interaction(self, uuid: str | None):
        for s in self._stream_list:
            if not isinstance(s, Stream):
                continue
            if not s.has_interaction(uuid):
                return False
        return True

    def get_interaction(self, uuid: str | None):
        if not self.has_interaction(uuid):
            return None

        for s in self._stream_list:
            if not isinstance(s, Stream):
                continue
            interaction = s.get_interaction(uuid)
            if interaction is not None:
                return interaction  # This will happen at the 1st iteration for the way "has_interaction" is implemented
        return None  # This will never happen, since we already ensured that it "has_interaction"

    def set(self, key_or_data, data=None, data_tag: int = -1, uuid: str | None = None):
        """Set data on a stream.

        Usage (example for stdin - it could be other StreamIO):
            - ``self.stdin.set(data)`` — when there is only one stream
            - ``self.stdin.set(stream_name, data)`` — by name
            - ``self.stdin.set(stream_index, data)`` — by index

        Args:
            key_or_data: Stream name/index if data is also provided, or the data itself
                         for the single-stream case.
            data: The data to set (when key_or_data is a name/index).
        """
        if len(self._stream_list) == 0:
            raise GenException("No streams bound to this StreamIO")

        if isinstance(key_or_data, list):
            if len(key_or_data) != len(self._stream_list):
                raise GenException(f"The list of stream values to set must have the same length of the stream list "
                                   f"({len(key_or_data)} != {len(self._stream_list)})")
            for i, data in enumerate(key_or_data):
                s = self._stream_list[i]
                if isinstance(s, Stream):
                    s.set(data, data_tag, uuid=uuid)
                else:
                    self._stream_list[i] = data
                    key_at_index = next(islice(self._streams, i, None))
                    self._streams[key_at_index] = data
        elif isinstance(key_or_data, int):
            s = self._stream_list[key_or_data]
            if isinstance(s, Stream):
                s.set(data, uuid=uuid)
            else:
                self._stream_list[key_or_data] = data
        elif key_or_data in self._streams:
            s = self._streams[key_or_data]
            if isinstance(s, Stream):
                s.set(data, data_tag, uuid=uuid)
            else:
                self._streams[key_or_data] = data
        else:
            raise GenException(f"Unknown stream: {key_or_data}")

    def get_tag(self, key: str | int | None = None, uuid: str | None = None):
        if len(self._stream_list) == 0:
            raise GenException("No streams bound to this StreamIO")

        if key is None:
            ret = []
            for s in self._stream_list:
                if isinstance(s, Stream):
                    data_tag = s.get_tag(uuid)
                    ret.append(data_tag)
                else:
                    ret.append(-1)
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

    def clear(self):
        self.set([None] * len(self))

    def __getitem__(self, key):
        self.get(key)

    def __len__(self):
        return len(self._stream_list)

    def __iter__(self):
        return iter(self._stream_list)

    def __contains__(self, key):
        if isinstance(key, str):
            return key in self._streams
        return False

    def items(self):
        for key, value in self._streams.items():
            yield key, value

    @property
    def names(self) -> list[str]:
        """Return the names (user hashes) of all bound streams."""
        return list(self._streams.keys())

    def __str__(self):
        grouped_by_user = {}
        for user_hash in self._streams.keys():
            user = DataProps.peer_id_from_user_hash(user_hash)
            if user not in grouped_by_user:
                grouped_by_user[user] = []
            grouped_by_user[user].append(DataProps.name_from_user_hash(user_hash))
        if len(grouped_by_user) > 0:
            return str(", ".join([(user + ":" + ",".join(stream_names))
                                  for user, stream_names in grouped_by_user.items()]))
        else:
            return "no-streams"
