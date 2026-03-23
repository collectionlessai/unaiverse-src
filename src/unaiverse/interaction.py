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
import copy
import time
import json
import uuid as _uuid
from enum import Enum
from unaiverse.clock import clock
from unaiverse.custom import Custom
from collections.abc import Callable
from unaiverse.utils.logger import log
from unaiverse.utils.misc import GenException
from unaiverse.streams.streamproxy import StreamProxy
from unaiverse.streams.dataprops import DataProps
from unaiverse.streams.streams import Stream, BufferedStream, serialize_payload, deserialize_payload


class Interaction:
    """Represents a single interaction between agents.

    An Interaction encapsulates everything needed for inter-agent communication:
    the requested action, its arguments, sample specifications, timing, status,
    and the data access interface (get/set).
    """

    def __init__(self,
                 action_name: str | None = None,
                 action_kwargs: dict | None = None,
                 streams: tuple[list[str], int] | list[str] | None = None,
                 data_samples: list | None = None,
                 num_steps: int = -1,
                 requester: str | None = None,
                 target: str | list[str] | None = None,
                 from_state: str | None = None,
                 to_state: str | None = None,
                 timeout: float = -1.,
                 uuid: str | None = "random"):
        """Create a new Interaction.

        Args:
            action_name: The name of the action being requested (e.g. "process", "learn").
            action_kwargs: Action arguments (dict, no streams).
            streams: List of (stream_name - a.k.a. stream user hash, num_samples) tuples specifying expected data.
            data_samples: List of actual data samples (alternative to samples specification).
            requester: Peer ID of the agent originating this interaction.
            target: Peer ID of the target agent or list of peer IDs.
            from_state: Name of the state from which the interaction is expected to happen.
            to_state: Name of the state where the interaction is supposed to yield.
            timeout: Maximum timeout until which the interaction is valid (-1 means no limit: the InteractionManager
            will override this with its internal timeout valid for all actions).
        """
        # Identity
        self.uuid: str = _uuid.uuid4().hex[0:8] if (uuid is not None and uuid == 'random') else uuid

        # Action requested
        self.action_name: str = action_name
        self.action_kwargs: dict = action_kwargs if action_kwargs is not None else {}

        # Timing information
        self.timestamp_created: float = -1.
        self.timestamp_started: float = -1.
        self.timestamp_completed: float = -1.
        self.timeout: float = timeout
        self.cycle_created: int = -1
        self.cycle_started: int = -1
        self.cycle_completed: int = -1

        # States
        self.from_state: str | None = from_state
        self.to_state: str | None = to_state

        # Samples specification: list of (stream_name, num_samples)
        self.streams: list = []
        self.data_samples: list = []
        self.num_steps: int = num_steps  # Generic not-data-based interactions have -1 here

        # Stream-based specification
        if streams is not None and len(streams) > 0:
            self.streams = Interaction.__parse_streams(streams)
            if self.num_steps == -1:
                self.num_steps = max([stream_dict['num_samples'] for stream_dict in self.streams])

        # Actual data (alternative to samples above)
        if streams is None and data_samples is not None and len(data_samples) > 0:
            if isinstance(data_samples[0], dict):  # If it is a list of dicts, each is type_name->base64...
                self.data_samples: list = deserialize_payload(data_samples)  # List of PIL.Image, torch.Tensor, str, ...
            else:
                self.data_samples = data_samples  # List of PIL.Image, torch.Tensor, str, ...
            self.num_steps = 1

        # Status
        self.status: InteractionStatus = InteractionStatus.CREATED
        self.data_sent_after_completion = False
        self.completion_reason: CompletionReason | None = None
        self.buffered_stream_restarted = False
        self.send_status = True

        # Pointers
        self.action_ref: Callable[Interaction] | None = None  # Reference to the actual Action object
        self.received_data_tags: dict[str, list[int]] = {}  # Streams could have different data tags
        self.received_timestamps: dict[str, list[float]] = {}  # Different treams could yield different data tags

        # Participants
        self.requester: str | None = requester
        self.target: list[str] | None = target if isinstance(target, list) else [target]

        # Destination state reached, if the action was completed (for status messages)
        self.destination_state: str | None = None

        # Params that depends on the way it is stored in the action
        self.by_insertion_order_id = -1
        self.by_requester_insertion_order_id = -1

        # Status of the running action
        self.__step_idx = -1  # The first done step will have index 0, while -1 means "no steps done so far"
        self.__starting_time = 0.
        self.__timeout_starting_time = 0.
        self.__mark = None
        self.stdin_streams = {}  # User hash to stream object
        self.stdtar_streams = {}  # User hash to stream object
        self.stdext_streams = {}  # User hash to stream object
        self.owned_streams = {}  # User hash to stream object
        self.lazy_streams = {}  # User hash to stream object
        self.iostreams = StreamProxy()  # Virtual stream IO associated to all streams above (not the lazy ones)

        # Back-reference to the InteractionManager (set when registered)
        self.__im: 'InteractionManager | None' = None

    @property
    def created(self) -> bool:
        """True when the interaction has finished."""
        return self.status == InteractionStatus.CREATED

    @property
    def completed(self) -> bool:
        """True when the interaction has finished."""
        return self.status == InteractionStatus.COMPLETED

    @property
    def running(self) -> bool:
        """True when the interaction is currently running."""
        return self.status == InteractionStatus.RUNNING

    def set_manager(self,
                    im: 'InteractionManager',
                    stdin_streams: dict[str, Stream | object],
                    stdtar_streams: dict[str, Stream | object],
                    stdext_streams: dict[str, Stream | object],
                    owned_streams: dict[str, Stream | object]):
        self.__im = im
        self.timestamp_created = clock.get_time()
        self.cycle_created = clock.get_cycle()
        self.stdin_streams = stdin_streams
        self.stdtar_streams = stdtar_streams
        self.stdext_streams = stdext_streams
        self.owned_streams = owned_streams
        data_samples_dict = {"<data_sample_" + str(i) + ">": self.data_samples[i]
                             for i in range(0, len(self.data_samples))}
        self.iostreams.bind(self.stdin_streams | self.stdtar_streams | self.stdext_streams | self.owned_streams |
                            data_samples_dict)

    def reset_state(self):
        """Resets the state, including the step counter and timing metrics, allowing it to be re-run from the
        beginning.
        """
        self.__step_idx = -1
        self.__starting_time = 0.
        self.__timeout_starting_time = 0.

    def set_mark(self, mark: object):
        self.__mark = mark

    def get_mark(self):
        return self.__mark

    def get_step_idx(self):
        return self.__step_idx

    def set_step_idx(self, steps: int):
        self.__step_idx = steps

    def inc_step_idx(self):
        self.__step_idx += 1

    def dec_step_idx(self):
        self.__step_idx -= 1

    def set_starting_time(self, t: float):
        self.__starting_time = t

    def set_timeout_starting_time(self, t: float):
        self.__timeout_starting_time = t

    def get_total_steps(self):
        """Retrieves the total number of steps configured for the action.

        Returns:
            An integer representing the total steps.
        """
        return self.num_steps

    def get_starting_time(self):
        """Retrieves the timestamp when the action's current execution started.

        Returns:
            A float representing the starting time.
        """
        return self.__starting_time

    def get_timeout_starting_time(self):
        """Retrieves the timestamp when the action's current execution started.

        Returns:
            A float representing the timeout starting time.
        """
        return self.__timeout_starting_time

    def is_multi_steps(self):
        """Determines if the action is configured to be a multistep action (i.e., not a single-step action).

        Returns:
            A boolean indicating if the action is multistep.
        """
        return self.num_steps > 1

    def is_single_step(self):
        return self.num_steps == 1 or self.num_steps < 0  # Action with a single data sample or no data samples at all

    def is_valid(self):
        return self.by_insertion_order_id >= 0 and self.by_requester_insertion_order_id >= 0

    def is_completed(self):
        return self.status == InteractionStatus.COMPLETED

    def was_data_sent_after_completion(self):
        return self.data_sent_after_completion

    def has_dummy_requester(self):
        return self.requester is None

    def set_arg(self, arg_name: str, arg_value: object):
        self.action_kwargs[arg_name] = arg_value

    def get_arg(self, arg_name):
        return self.action_kwargs[arg_name] if arg_name in self.action_kwargs else None

    def was_at_least_one_step_done(self):
        return self.__step_idx >= 0

    def was_last_step_done(self):
        """Determines if the action has reached its completion criteria, either by reaching the total number of steps
        or by exceeding the maximum allowed execution time.

        Returns:
            True if the action is completed, False otherwise.
        """
        return ((self.num_steps < 0 and self.__step_idx == 0) or  # Action with no data (no steps)
                (self.num_steps > 0 and self.__step_idx == self.num_steps - 1))  # Action with one or more steps

    def is_delayed(self, starting_time: float):
        """Checks if the action is currently in a delayed state and cannot be executed yet, based on a defined delay
        period.

        Args:
            starting_time: The time the delay period began.

        Returns:
            True if the action is delayed, False otherwise.
        """
        return self.action_ref.get_delay() > 0 and (time.perf_counter() - starting_time) <= self.action_ref.get_delay()

    def is_timed_out(self):
        """Checks if the action has exceeded its configured timeout period since the last successful execution attempt.

        Returns:
            True if the action has timed out, False otherwise.
        """

        # If the action was never started (even a failed attempt), this method has no sense
        if self.__starting_time <= 0.:
            return False

        # Checking global timeout: if too much time passed, no matter if the action started or not, it's timeout!
        if self.action_ref.get_total_time() > 0:
            if self.action_ref.get_total_time() <= (time.perf_counter() - self.__starting_time):
                log.inter(f"Timeout for {self.action_name}! "
                          f"({(time.perf_counter() - self.__starting_time)}/"
                          f"{self.action_ref.get_total_time()})!")
                return True
            else:
                log.debug(f"Running timeout for {self.action_name}! "
                          f"({(time.perf_counter() - self.__starting_time)}/"
                          f"{self.action_ref.get_total_time()})!")

        # Checking next-step timeout
        if self.__timeout_starting_time > 0. and self.action_ref.get_timeout() > 0:
            if self.action_ref.get_timeout() <= (time.perf_counter() - self.__timeout_starting_time):
                log.inter(
                    f"Hot timeout for {self.action_name}! "
                    f"({(time.perf_counter() - self.__timeout_starting_time)}/"
                    f"{self.action_ref.get_timeout()})!")
                return True
            else:
                log.debug(f"Running hot timeout for {self.action_name} "
                          f"({(time.perf_counter() - self.__timeout_starting_time)}/"
                          f"{self.action_ref.get_timeout()})!")
                return False
        else:
            return False

    def set_action_ref(self, action_ref: object):
        self.action_ref = action_ref

        # Augmenting argument list, by fusing with the arguments defined in the HSM
        action_kwargs = self.action_kwargs if self.action_kwargs is not None else {}
        self.action_ref.check_provided_args(action_kwargs, exception=True)

        # Force a default timeout on multistep actions, to avoid infinite trials
        if self.is_multi_steps() and self.action_ref.get_timeout() <= 0:
            self.action_ref.set_default_timeout()

    def has_stream(self, user_hash):
        return user_hash in self.iostreams or user_hash in self.lazy_streams

    def add_lazy_stream(self, user_hash, stream_obj, is_owned: bool = False):
        if not self.has_stream(user_hash):
            self.lazy_streams[user_hash] = stream_obj
            if is_owned:
                self.owned_streams[user_hash] = stream_obj
            self.iostreams.add_new_bind(user_hash, stream_obj)

    def mark_running(self):
        """Mark this interaction as currently running."""
        self.status = InteractionStatus.RUNNING
        self.timestamp_started = clock.get_time()
        self.cycle_started = clock.get_cycle()

    def mark_completed(self, reason: 'CompletionReason', dest_state: str | None = None):
        """Mark this interaction as completed.

        Args:
            reason: The reason for completion.
            dest_state: The destination state reached by completing this action.
        """
        self.status = InteractionStatus.COMPLETED
        self.destination_state = dest_state
        self.completion_reason = reason
        self.timestamp_completed = clock.get_time()
        self.cycle_completed = clock.get_cycle()

    def clear_from_streams_and_action(self):

        # Clearing interaction from all the streams involved as input, extra, or targets
        for stream in self.iostreams:

            # Skipping data samples and default input values
            if not isinstance(stream, Stream):
                continue

            stream.remove_interaction(self)

        # Clearing interaction from action list
        if self.action_ref is not None:
            self.action_ref.get_list_of_interactions().remove(self)

    def is_expired(self, timeout_secs: float | None = None) -> bool:
        """Check if this interaction has expired based on an external timeout.

        Args:
            timeout_secs: Maximum higher-priority timeout in seconds (default: None).

        Returns:
            True if the interaction is older than timeout_secs.
        """

        # Deciding the real value of timeout_specs, also in function of the interaction-specific timeout (if given)
        if timeout_secs is not None and timeout_secs > 0. and self.timeout is not None and self.timeout > 0.:
            timeout_secs = min(timeout_secs, self.timeout)
        elif timeout_secs is None or timeout_secs <= 0.:
            timeout_secs = self.timeout  # It could still be None or < 0.

        if timeout_secs is None or timeout_secs <= 0.:  # Perpetual interaction
            return False
        else:
            return (clock.get_time() - self.timestamp_created) >= timeout_secs

    def record_data_tags(self, data_tags: dict[str, int], timestamps: dict[str, float] | None = None):
        """Record that data with the given tag was received for this interaction.

        Args:
            data_tags: The data tags/sequence numbers.
            timestamps: Optional timestamps (defaults to current time).
        """
        for stream_name, data_tag in data_tags.items():
            self.received_data_tags[stream_name].append(data_tag)
            self.received_timestamps[stream_name].append(
                timestamps[stream_name] if timestamps is not None and stream_name in timestamps
                else clock.get_time())

    def check_if_doable(self):
        if self.__im is not None:
            return self.__im.check_if_doable(self)
        else:
            return not self.completed  # System interactions does not have an interaction manager

    def alter_arg(self, arg_name: str, arg_value: object):
        if arg_name in self.action_kwargs:
            self.action_kwargs[arg_name] = arg_value
            return True
        else:
            return False

    def to_dict(self) -> dict:
        """Serialize this interaction for network transmission.

        Returns:
            A dictionary representation of this interaction.
        """
        return {
            'uuid': self.uuid,
            'action_name': self.action_name,
            'requester': self.requester,
            'target': self.target,
            'action_kwargs': self.action_kwargs,
            'streams': self.streams,
            'data_samples': serialize_payload(self.data_samples),
            'num_steps': self.num_steps,
            'completion_reason': self.completion_reason.value if self.completion_reason else None,
            'destination_state': self.destination_state,
            'from_state': self.from_state,
            'to_state': self.to_state,
            'timeout': self.timeout,
            'status': self.status.value
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Interaction':
        """Deserialize an Interaction from a dictionary.

        Args:
            d: Dictionary from to_dict().

        Returns:
            A new Interaction instance.
        """
        interaction = cls(
            action_name=d['action_name'],
            action_kwargs=d.get('action_kwargs', {}),
            streams=d.get('streams', []),
            data_samples=d.get('data_samples', None),
            num_steps=d.get('num_steps'),
            requester=d.get('requester'),
            target=d.get('target'),
            from_state=d.get('from_state'),
            to_state=d.get('to_state'),
            timeout=d.get('timeout', -1.),
        )
        interaction.uuid = d['uuid']
        interaction.status = InteractionStatus(d['status'])
        if d.get('completion_reason'):
            interaction.completion_reason = CompletionReason(d['completion_reason'])
        interaction.destination_state = d.get('destination_state')
        return interaction

    def to_status_dict(self) -> dict:
        """Build a minimal status dict for status update messages.

        Returns:
            A dictionary with uuid, status, completion_reason, destination_state, and data_tag.
        """
        return {
            'uuid': self.uuid,
            'status': self.status.value,
            'completion_reason': self.completion_reason.value if self.completion_reason else None,
            'destination_state': self.destination_state,
            'data_tags': {k: self.received_data_tags[k][-1] for k in self.received_data_tags.keys()}
            if self.received_data_tags else None,
        }

    def to_code_str(self, include_uuid: bool = False):
        s = ""
        if include_uuid:
            s = f"{self.uuid} => "
        return s + (f"artss:{self.action_name}|{self.requester}|{self.target}|"
                    f"{self.iostreams}|{self.status.value[0:3]}" +
                    (("_" + self.completion_reason.value[0:3]) if self.completion_reason is not None else ""))

    def to_str(self):
        return json.dumps([self.requester, self.action_kwargs, self.timestamp_created, self.uuid])

    @staticmethod
    def __parse_streams(streams: list):
        """Parse the list of streams provided by the user, standardizing its format.

        The user can specify streams in these 4 ways:
            (1) [stream_hash_1 (str), ... , stream_hash_N (str)]: 1 sample per stream
            (2) [stream_hash_1 (str), ... , stream_hash_N (str), Z (int)]: Z samples each stream
            (3) [stream_hash_1 (str), Z_1 (int), ... , stream_hash_N (str), Z_N (int)]: Z_i samples for the i-th stream
            (4) [... , {'stream_hash': stream_hash_i (str),
                        'num_samples' (optional): Z_i (int),
                        'redirect' (optional): 'stdin' or 'stdext' (str)}, ...]

            The last one (4) means Z_i samples for the i-th stream (or 1 if not provided), and it also optionally
            specifies if the stream is expected to become an input of the processor ('stdin') or not ('stdext').
            Notice that this choice could be re-arranged by the Interaction Manager, in function of the actual
            capabilities of the processor.

        Args:
            streams (list): The list of streams (see the example above).

        Returns:
            The standardized list of streams, i.e., a list of dictionaries in this format:
                {
                    'stream_hash': stream_hash,  # the hash that was provided (whatever that is)
                    'num_samples': num_samples,  # the number of samples for this stream (auto-computed if not provided)
                    'redirect': None or 'stdin' or 'stdext'  # computed by the Interaction Manager if None
                    'user_hash': None,  # it will be computed by the Interaction Manager
                    'net_hash': None,  # it will be computed by the Interaction Manager
                    'name': None,  # it will be computed by the Interaction Manager
                    'group': None,  # it will be computed by the Interaction Manager
                }
        """

        # Possible ways of providing streams
        format_types = {'hashes_only', 'hashes_and_global_num_samples', 'hashes_and_num_samples_per_stream', 'detailed'}

        # Detecting format type
        format_type = set(format_types)
        for i, s in enumerate(streams):
            if not isinstance(s, str) and not isinstance(s, int) and not isinstance(s, dict):
                format_type.clear()
                break
            if not isinstance(s, str):
                format_type.discard('hashes_only')
            if not isinstance(s, dict):
                format_type.discard('detailed')
            if isinstance(s, int) and i != len(streams) - 1:
                format_type.discard('hashes_only')
                format_type.discard('hashes_and_global_num_samples')
                format_type.discard('detailed')
            if isinstance(s, dict):
                format_type.discard('hashes_only')
                format_type.discard('hashes_and_global_num_samples')
                format_type.discard('hashes_and_num_samples_per_stream')
                for k in s.keys():
                    if k not in {'stream_hash', 'num_samples', 'redirect', 'user_hash', 'net_hash', 'name', 'group'}:
                        format_type.discard('detailed')
                    if (k == 'stream_hash' and not isinstance(s[k], int)
                            and not (Stream.is_user_hash(s[k]) or Stream.is_net_hash(s[k]))):
                        format_type.discard('detailed')
                        break
                    if k == 'num_samples' and (not isinstance(s[k], int) or s[k] <= 0):
                        format_type.discard('detailed')
                        break
                    if (k == 'redirect' and s[k] is not None and
                            (not isinstance(s, str) or s[k] not in {'stdin', 'stdext', 'stdtar'})):
                        format_type.discard('detailed')
                        break
        if (len(format_type) == 3 and
                'hashes_and_global_num_samples' in format_type and 'hashes_and_num_samples_per_stream' in format_type):
            for i, s in enumerate(streams):
                if len(streams) < 2 or i % 2 == 0 and not isinstance(s, str) or i % 2 == 1 and not isinstance(s, int):
                    format_type.discard('hashes_and_num_samples_per_stream')
                if i < len(streams) - 1 and not isinstance(s, str) or i == len(streams) - 1 and not isinstance(s, int):
                    format_type.discard('hashes_and_global_num_samples')
        if len(format_type) > 1:
            raise GenException(f'Invalid syntax for streams involved in an interaction: {streams} '
                               f'(unknown format in {format_type})')
        if len(format_type) == 0:
            raise GenException(f'Invalid syntax for streams involved in an interaction: {streams} '
                               f'(no valid format detected)')

        # Converting all not-dict-based format to 'hashes_and_num_samples_per_stream'
        format_type = next(iter(format_type))
        if format_type == 'hashes_only':
            _streams = []
            for s in streams:
                _streams.append(s)
                _streams.append(1)
            streams = _streams
            format_type = 'hashes_and_num_samples_per_stream'
        elif format_type == 'hashes_and_global_num_samples':
            _streams = []
            for i in range(0, len(streams) - 1):
                _streams.append(streams[i])
                _streams.append(streams[-1])
            streams = _streams
            format_type = 'hashes_and_num_samples_per_stream'

        # Creating the detailed representation, from the not-dict-based one 'hashes_and_num_samples_per_stream'
        _streams = []
        if format_type == 'hashes_and_num_samples_per_stream':
            for i in range(0, len(streams), 2):
                stream_hash = streams[i]
                num_samples = streams[i + 1]
                _streams.append({
                    'stream_hash': stream_hash,  # the hash that was provided (whatever that is)
                    'num_samples': num_samples,
                    'redirect': None,  # computed by the Interaction Manager
                    'user_hash': None,  # computed by the Interaction Manager
                    'net_hash': None,  # computed by the Interaction Manager
                    'name': None,  # computed by the Interaction Manager
                    'group': None,  # computed by the Interaction Manager
                })
        elif format_type == 'detailed':
            for i in range(0, len(streams)):
                stream_dict = streams[i]
                _streams.append({
                    'stream_hash': stream_dict.get('stream_hash'),  # the hash that was provided (whatever that is)
                    'num_samples': stream_dict.get('num_samples', 1),
                    'redirect': stream_dict.get('redirect', None),  # computed by the Interaction Manager, if not given
                    'user_hash': None,  # computed by the Interaction Manager
                    'net_hash': None,  # computed by the Interaction Manager
                    'name': None,  # computed by the Interaction Manager
                    'group': None,  # computed by the Interaction Manager
                })
        else:
            raise GenException(f'Unexpected format type: {format_type}')
        return _streams

    def __str__(self):
        """Provides a string representation of the `Interaction` instance.

        Returns:
            A string containing a formatted summary of the instance.
        """
        return f"{self.to_code_str()}"


class InteractionManager:
    """Manages all interactions for an agent.

    Responsibilities:
    - Maintains lists of sent and received interactions
    - Sets stdin/stdout to the streams of the current interaction
    - Sets the recipients of the generation
    - Clears expired interactions at every clock cycle
    - Sends back interaction status
    - Waits for all involved streams to have sent data before marking action ready
    """

    def __init__(self, agent: object, max_interactions: int = Custom.MAX_INTERACTIONS):
        """Create an InteractionManager.

        Args:
            agent: Back-reference to the owning agent (AgentBasics instance).
        """
        self.agent = agent
        self.max_interactions = max_interactions
        self.sent: dict[str, Interaction] = {}       # uuid -> Interaction (sent by this agent)
        self.received: dict[str, Interaction] = {}   # uuid -> Interaction (received from others)
        self.lazy: dict[str, Interaction] = {}   # uuid -> Interaction (added by you)
        self.current: Interaction | None = None      # Currently executing interaction
        self.last_registered: Interaction | None = None
        self.sent_recently_completed: set[Interaction] = set()  # Completed interactions
        self.received_recently_completed: set[Interaction] = set()  # Completed interactions
        self.lazy_recently_completed: set[Interaction] = set()  # Completed interactions

    def room_for_registration(self):
        return len(self.sent) + len(self.received) + len(self.lazy) < self.max_interactions

    def count_interactions(self):
        return (len(self.sent) + len(self.received) + len(self.lazy) +
                len(self.sent_recently_completed) + len(self.received_recently_completed) +
                len(self.lazy_recently_completed))

    def clear_from_all_owned_streams(self, interaction: Interaction):
        for stream_obj in self.agent.owned_streams_by_user_hash.values():
            if stream_obj.has_interaction(interaction.uuid):
                stream_obj.remove_interaction(interaction)

    def unregister(self, interaction: Interaction):
        # if len(interaction.iostreams) == 0:
        #     self.clear_from_all_owned_streams(interaction)
        interaction.clear_from_streams_and_action()
        found = False
        if interaction.uuid in self.sent and self.sent[interaction.uuid].requester == interaction.requester:
            del self.sent[interaction.uuid]
            found = True
        if interaction.uuid in self.received and self.received[interaction.uuid].requester == interaction.requester:
            del self.received[interaction.uuid]
            found = True
        if interaction.uuid in self.lazy and self.lazy[interaction.uuid].requester == interaction.requester:
            del self.lazy[interaction.uuid]
            found = True
        return found

    def register_sent(self, interaction: Interaction) -> bool:
        """Register an interaction that this agent has sent.

        Args:
            interaction: The Interaction to register.
        """
        if not self.room_for_registration():
            log.error(f"No more room for interactions (limit: {self.max_interactions})")
            return False

        #  Ensuring all the streams mentioned in the interaction are known, and normalizing them
        _, expanded_owned_streams = self.expand_and_normalize_streams(interaction)
        if expanded_owned_streams is None:
            log.error(f"Invalid stream in interaction: {interaction}")
            return False

        # Converting format for owned streams (no matter what the matching routine decided)
        owned_user_hashes_to_stream_objs = {_stream_dict['user_hash']: _stream_dict['obj']
                                            for _stream_dicts in expanded_owned_streams.values()
                                            for _stream_dict in _stream_dicts}

        for stream_obj in owned_user_hashes_to_stream_objs.values():
            stream_obj.add_interaction(interaction)

        interaction.set_manager(self, stdin_streams={}, stdtar_streams={}, stdext_streams={},
                                owned_streams=owned_user_hashes_to_stream_objs)

        # No matter what the interaction does: the processor output streams of the target be aware of the possibility
        # that this interaction might yield new data
        for target_agent in interaction.target:
            streams_of_target_agent = self.agent.find_streams(target_agent, 'processor', discard_owned=True)
            for stream_dict in streams_of_target_agent.values():
                for stream in stream_dict.values():
                    stream.add_interaction(interaction)

        # Registering
        self.sent[interaction.uuid] = interaction

        interaction.status = InteractionStatus.REQUESTED
        self.last_registered = interaction
        return True

    def register_received(self, interaction: Interaction) -> bool:
        """Register an interaction received from another agent.

        Args:
            interaction: The Interaction to register.
        """
        if not self.room_for_registration():
            log.error(f"No more room for interactions (limit: {self.max_interactions})")
            return False

        #  Ensuring all the streams mentioned in the interaction are known, and normalizing them
        expanded_streams, expanded_owned_streams = self.expand_and_normalize_streams(interaction)
        if expanded_streams is None:
            return False

        # 1. Filling the missing field of the different stream dictionaries, and assigning streams to stdin or stdext,
        # following the suggestions provided in 'redirect', when possible
        # 2. Returning False if there were no ways to fill the processor input arguments.
        (valid, stdin_user_hashes_to_stream_objs, stdtar_user_hashes_to_stream_objs,
         stdext_user_hashes_to_stream_objs) = self.match_streams(expanded_streams)
        if not valid:
            return False

        # Converting format for owned streams (no matter what the matching routine decided)
        owned_user_hashes_to_stream_objs = {_stream_dict['user_hash']: _stream_dict['obj']
                                            for _stream_dicts in expanded_owned_streams.values()
                                            for _stream_dict in _stream_dicts}

        # Updating the interaction object with the decisions from the manager
        interaction.set_manager(self,
                                stdin_streams=stdin_user_hashes_to_stream_objs,
                                stdtar_streams=stdtar_user_hashes_to_stream_objs,
                                stdext_streams=stdext_user_hashes_to_stream_objs,
                                owned_streams=owned_user_hashes_to_stream_objs)

        # Registering the interaction in the involved streams (do this AFTER calling set_manager)
        for stream in interaction.iostreams:

            # Skipping data samples and default input values
            if not isinstance(stream, Stream):
                continue

            stream.add_interaction(interaction)

        # No matter what the interaction does: the processor output streams must be aware of the possibility that this
        # interaction might yield new data, that will be sent (if any) to the author of this interaction
        for stream in self.agent.proc_streams_by_user_hash.values():
            if interaction.requester in self.agent.public_agents:
                if not stream.props.is_public():
                    continue
            elif interaction.requester in self.agent.all_agents:
                if stream.props.is_public():
                    continue
            else:
                log.critical(f"Unexpected interaction from an unknown agent: {interaction.requester}")
            stream.add_interaction(interaction)

        # Registering
        self.received[interaction.uuid] = interaction

        interaction.status = InteractionStatus.RECEIVED
        self.last_registered = interaction
        return True

    def register_lazy(self, interaction: Interaction) -> bool:
        """Register an interaction that you manually generated within this agent.

        Args:
            interaction: The Interaction to register.
        """
        if not self.room_for_registration():
            log.error(f"No more room for interactions (limit: {self.max_interactions})")
            return False

        #  Ensuring all the streams mentioned in the interaction are known, and normalizing them
        _, expanded_owned_streams = self.expand_and_normalize_streams(interaction)
        if expanded_owned_streams is None:
            log.error(f"Invalid stream in interaction: {interaction}")
            return False

        # Converting format for owned streams (no matter what the matching routine decided)
        owned_user_hashes_to_stream_objs = {_stream_dict['user_hash']: _stream_dict['obj']
                                            for _stream_dicts in expanded_owned_streams.values()
                                            for _stream_dict in _stream_dicts}

        for stream_obj in owned_user_hashes_to_stream_objs.values():
            stream_obj.add_interaction(interaction)

        interaction.set_manager(self, stdin_streams={}, stdtar_streams={}, stdext_streams={},
                                owned_streams=owned_user_hashes_to_stream_objs)
        self.lazy[interaction.uuid] = interaction

        interaction.status = InteractionStatus.LAZY
        self.last_registered = interaction
        return True

    def expand_and_normalize_streams(self, interaction: Interaction) -> (
            tuple[dict[str | None, list], dict[str | None, list]] | tuple[None, None]):

        # Guessing net hash and specific name af each stream: if the name was not provided, then all the streams of the
        # group are considered. Generating a full list of streams, distinguishing them in function of their suggested
        # redirection, i.e., 'stdin', 'stdtar', 'stdext', None (meaning 'no suggestions').
        expanded_streams = {'stdin': [], 'stdtar': [], 'stdext': [], None: []}
        expanded_owned_streams = {'stdin': [], 'stdtar': [], 'stdext': [], None: []}
        for stream_dict in interaction.streams:
            stream_hash = stream_dict['stream_hash']
            if Stream.is_user_hash(stream_hash):
                peer_id = Stream.peer_id_from_user_hash(stream_hash)
                name = Stream.name_from_user_hash(stream_hash)
                net_hash_to_streams = self.agent.find_streams(peer_id=peer_id, name_or_group=name)
                if net_hash_to_streams is None or len(net_hash_to_streams) == 0:
                    return None, None

                stream_dict['user_hash'] = stream_hash
                stream_dict['net_hash'] = next(iter(net_hash_to_streams.keys()))
                stream_dict['name'] = name
                stream_dict['group'] = DataProps.name_or_group_from_net_hash(stream_dict['net_hash'])
                stream_dict['obj'] = self.agent.known_streams_by_user_hash(stream_dict['user_hash'])
                expanded_streams[stream_dict['redirect']].append(stream_dict)

                if stream_dict['user_hash'] in self.agent.owned_streams_by_user_hash:
                    expanded_owned_streams[stream_dict['redirect']].append(stream_dict)
            elif Stream.is_net_hash(stream_hash):
                if stream_hash not in self.agent.known_streams:
                    return None, None

                streams = self.agent.known_streams[stream_hash]
                stream_dict['net_hash'] = stream_hash
                stream_dict['group'] = DataProps.name_or_group_from_net_hash(stream_dict['net_hash'])
                for stream_obj in streams.values():
                    _stream_dict = copy.deepcopy(stream_dict)
                    _stream_dict['name'] = stream_obj.props.name
                    _stream_dict['user_hash'] = DataProps.user_hash_from_net_hash(_stream_dict['net_hash'],
                                                                                  _stream_dict['name'])
                    _stream_dict['obj'] = stream_obj
                    expanded_streams[stream_dict['redirect']].append(_stream_dict)
                    if _stream_dict['user_hash'] in self.agent.owned_streams_by_user_hash:
                        expanded_owned_streams[_stream_dict['redirect']].append(_stream_dict)
        return expanded_streams, expanded_owned_streams

    def match_streams(self, expanded_streams: dict) -> tuple[bool, dict, dict, dict]:

        # Assigning streams to 'stdin' or 'stdext', following the given suggestions, when possible, and being a bit
        # heuristic for all the not-well-defined cases
        processor_will_be_used = len(expanded_streams['stdin']) > 0 or len(expanded_streams[None]) > 0
        stdin_streams = {}
        stdtar_streams = {}
        stdext_streams = {
            stream_dict_inter['user_hash']: self.agent.known_streams_by_user_hash[stream_dict_inter['user_hash']] for
            stream_dict_inter in expanded_streams['stdext']}

        # We try to match the 'stdin' suggestions with the different input arguments of the processors.
        # We also consider the streams with no specific suggestions (lower priority).
        if processor_will_be_used:
            sources = ['stdin', None]
            pos_stdin_streams = [None] * len(self.agent.proc_inputs)
            for i in range(len(self.agent.proc_inputs)):
                found_match = -1
                for source in sources:
                    for j, stream_dict_inter in enumerate(expanded_streams[source]):
                        net_hash = stream_dict_inter['net_hash']
                        name = stream_dict_inter['name']

                        # If the current input stream is compatible with the i-th input slot...
                        if (net_hash, name) in self.agent.compat_in_streams[i]:
                            pos_stdin_streams[i] = stream_dict_inter['user_hash']
                            found_match = j
                            break

                    if found_match >= 0:
                        del expanded_streams[source][found_match]  # Removing the stream from the suggestions
                        break

            # We ensured that the not-filled-arguments of the processor have a default value, otherwise no ways
            for i in range(len(self.agent.proc_inputs)):
                if pos_stdin_streams[i] is None:
                    if not self.agent.proc_optional_inputs[i]["has_default"]:
                        return False, {}, {}, {}
                    else:
                        stdin_streams["<default_input_pos_" + str(i) + ">"] = (
                            self.agent.self.agent.proc_optional_inputs)[i]["default_value"]
                else:
                    stdin_streams[pos_stdin_streams[i]] = self.agent.known_streams_by_user_hash[pos_stdin_streams[i]]

            # Matching targets
            sources = ['stdtar', None]
            pos_stdtar_streams = [None] * len(self.agent.proc_outputs)
            for i in range(len(self.agent.proc_outputs)):
                found_match = -1
                for source in sources:
                    for j, stream_dict_inter in enumerate(expanded_streams[source]):
                        net_hash = stream_dict_inter['net_hash']
                        name = stream_dict_inter['name']

                        # If the current input stream is compatible with the i-th input slot...
                        if (net_hash, name) in self.agent.compat_out_streams[i]:
                            pos_stdtar_streams[i] = stream_dict_inter['user_hash']
                            found_match = j
                            break

                    if found_match >= 0:
                        del expanded_streams[source][found_match]  # Removing the stream from the suggestions
                        break

            # We ensured that the not-filled-arguments of the processor have a default value, otherwise no ways
            for i in range(len(self.agent.proc_outputs)):
                if pos_stdtar_streams[i] is not None:
                    stdin_streams[pos_stdtar_streams[i]] = self.agent.known_streams_by_user_hash[pos_stdtar_streams[i]]

        # We add to 'stdext' all the streams that did not fit the processor (both coming from suggestions in 'stdin' or
        # not suggested at all)
        sources = ['stdin', 'stdtar', None]
        for source in sources:
            for stream_dict_inter in expanded_streams[source]:
                stdext_streams[stream_dict_inter['user_hash']] = (
                    self.agent.known_streams_by_user_hash)[stream_dict_inter['user_hash']]
        return True, stdin_streams, stdtar_streams, stdext_streams

    def check_if_doable(self, interaction: Interaction) -> bool:
        requester_is_known = interaction.requester in self.agent.all_agents
        run_deprecated_completion_step = ((interaction.is_timed_out() and
                                           interaction.was_at_least_one_step_done()) or
                                          interaction.was_last_step_done())
        return (not interaction.completed and requester_is_known and
                (self.check_stream_readiness(interaction) or run_deprecated_completion_step))

    def get_current(self):
        return self.current

    def get_last_registered(self):
        return self.last_registered

    def set_current(self, interaction: Interaction | None):
        """Set the given interaction as the currently executing one.

        This marks the interaction as running and configures the agent's
        stdin/stdout to point to the correct streams for this interaction.

        Args:
            interaction: The Interaction to set as current.
        """
        self.current = interaction
        if interaction is not None:
            interaction.mark_running()

            # Restart buffered streams and activate them if they were off
            for stream in interaction.iostreams:
                if isinstance(stream, BufferedStream):
                    stream.restart(interaction.uuid)

            self.agent.stdin.bind(interaction.stdin_streams)
            self.agent.stdtar.bind(interaction.stdtar_streams)
            self.agent.stdext.bind(interaction.stdext_streams)
        else:
            self.agent.stdin.bind(self.agent.proc_in_streams_by_user_hash)
            self.agent.stdtar.bind({})
            self.agent.stdext.bind(self.agent.env_streams_by_user_hash)

    def has_data(self, interaction: 'Interaction'):
        for stream_obj in self.agent.owned_streams_by_user_hash.values():
            if stream_obj.has_data(interaction):
                return True
        return False

    def get_recipients(self, interaction: 'Interaction'):
        if self.is_known(interaction):
            log.streams("IS KNOWN")
            if self.is_sent(interaction):
                log.streams("IS SENT")
                recipients = interaction.target  # This is always a list, even when with 1 element only
            elif self.is_received(interaction):
                log.streams("IS RECEIVED")
                recipients = [interaction.requester]  # This is always 1 element, we make it a list
            elif self.is_lazy(interaction):
                log.streams("IS LAZY")
                recipients = interaction.target  # This is always a list, even when with 1 element only
            else:
                raise GenException("Unexpected case of a known interaction that is both not sent or received")
        else:
            log.streams("IS NOT-KNOWN")

            # This is the case of a not-registered interaction.
            # It is exploited for backward compatibility, for those interactions whose only purpose
            # is to provide a recipient over a stream
            recipients = interaction.target  # This is always a list, even when with 1 element only
        return [x for x in recipients if x is not None]

    def complete(self, interaction: 'Interaction', reason: 'CompletionReason', dest_state: str | None = None):
        if interaction is not None:
            interaction.mark_completed(reason, dest_state=dest_state)
            if (interaction.uuid in self.sent and
                    interaction.requester == self.sent[interaction.uuid].requester):  # Distinguish chained
                self.sent_recently_completed.add(interaction)
                del self.sent[interaction.uuid]
            if (interaction.uuid in self.received and
                    interaction.requester == self.received[interaction.uuid].requester):  # Distinguish chained
                self.received_recently_completed.add(interaction)
                del self.received[interaction.uuid]
            if (interaction.uuid in self.lazy and
                    interaction.requester == self.lazy[interaction.uuid].requester):  # Distinguish chained
                self.lazy_recently_completed.add(interaction)
                del self.lazy[interaction.uuid]

    def complete_current(self, dest_state: str, reason: 'CompletionReason'):
        """Mark the current interaction as completed.

        Args:
            dest_state: The destination state reached by completing this action.
            reason: The reason for completion.
        """
        self.complete(self.current, dest_state=dest_state, reason=reason)
        self.current = None

    def drain_completed(self) -> list[Interaction]:
        """Return and clear the list of recently completed interactions.

        Used by the agent's ``behave()`` loop to send status notifications.

        Returns:
            List of Interaction objects that were recently completed.
        """
        cur_time = clock.get_time()
        cur_clock_cycle = clock.get_cycle()
        drained = []

        for recently_completed in [self.sent_recently_completed,
                                   self.received_recently_completed,
                                   self.lazy_recently_completed]:
            to_remove = []
            for i, interaction in enumerate(recently_completed):

                # Wait AT LEAST 1 clock cycle, to allow sending samples generated during the interaction
                # Of course, "disconnected"-agent-related interactions are immediately drained
                if (interaction.completion_reason == CompletionReason.DISCONNECTED or
                        (interaction.cycle_completed < cur_clock_cycle and
                         (cur_time - interaction.timestamp_completed) > Custom.DRAIN_TIMEOUT)):
                    log.inter(
                        f"Draining {interaction.to_code_str(True)} "
                        f"(cycle_completed={interaction.cycle_completed})")
                    to_remove.append(interaction)
                    drained.append(interaction)
            for interaction in to_remove:
                interaction.clear_from_streams_and_action()

                # Clearing from all the owned streams, that might have been used for output purposes
                # In principle, only the processor streams should be involved, since it is the only one in which we plug
                # this interaction in this class.
                # However, the user might have added the interaction to other owned streams.
                self.clear_from_all_owned_streams(interaction)

                self.sent_recently_completed.discard(interaction)
                self.received_recently_completed.discard(interaction)
                self.lazy_recently_completed.discard(interaction)

        return drained

    def complete_expired(self):
        """Remove expired interactions and return them for notification.

        Checks all received and sent interactions against the timeout.
        Expired interactions are marked as COMPLETED with TIMEOUT reason
        and removed from the tracking dicts.
        """
        for interaction in list(self.sent.values()) + list(self.received.values()) + list(self.lazy.values()):
            if interaction.status == InteractionStatus.COMPLETED:
                continue
            if interaction.is_expired(Custom.DEFAULT_INTER_TIMEOUT):
                self.complete(interaction, reason=CompletionReason.TIMEOUT)
            else:
                if self.is_received(interaction) and interaction.requester not in self.agent.all_agents:
                    self.complete(interaction, reason=CompletionReason.DISCONNECTED)
                if (self.is_sent(interaction) and
                        not any(target in self.agent.all_agents for target in interaction.target)):
                    self.complete(interaction, reason=CompletionReason.DISCONNECTED)

    def clear_expired_stream_data(self):
        all_interactions = list(self.received.values()) + list(self.sent.values()) + list(self.lazy.values())
        for interaction in all_interactions:
            for stream in interaction.iostreams:

                # In case of data samples or default input values, skip
                if not isinstance(stream, Stream):
                    continue

                # We only consider streams whose interactions were either removed, or are already completed, or
                # were just created (artificial interaction, for example the ones created just to send a sample)
                if (not stream.has_interaction(interaction.uuid)
                        or stream.get_interaction(interaction.uuid).completed
                        or stream.get_interaction(interaction.uuid).created):

                    # This will also clear the associated (completed) interaction, if still there
                    stream.clear_expired_data(Custom.DEFAULT_INTER_TIMEOUT)

    @staticmethod
    def check_stream_readiness(interaction: Interaction) -> bool:
        """Check if all streams involved in this interaction have fresh data.

        Args:
            interaction: The interaction to check.

        Returns:
            True if all involved streams have data not yet consumed by this interaction.
        """
        if interaction.received_data_tags is not None:
            for stream_user_hash, stream_obj in interaction.iostreams.items():

                # Skipping data samples and default input values
                if not isinstance(stream_obj, Stream):
                    continue

                if (stream_user_hash in interaction.received_data_tags and
                        len(interaction.received_data_tags[stream_user_hash]) > 0):
                    last_received_data_tag = interaction.received_data_tags[stream_user_hash][-1]
                else:
                    last_received_data_tag = None

                # Check if stream has data with a tag not yet consumed by this interaction
                current_tag = stream_obj.get_tag() if hasattr(stream_obj, 'get_tag') else (
                    getattr(stream_obj, 'data_tag', -1))
                if last_received_data_tag is not None and current_tag != last_received_data_tag:
                    return False  # Already consumed this sample
        return True

    def update_sent_status(self, status_dict: dict):
        """Update a previously sent interaction's status from a received status message.

        Args:
            status_dict: Dict from Interaction.to_status_dict().
        """
        uuid = status_dict.get('uuid')
        if uuid and uuid in self.sent:
            interaction = self.sent[uuid]
            interaction_status = InteractionStatus(status_dict['status'])
            if status_dict.get('completion_reason'):
                completion_reason = CompletionReason(status_dict['completion_reason'])
            else:
                completion_reason = CompletionReason.OK
            interaction_destination_state = status_dict.get('destination_state')
            data_tags: dict[str, int] | None = status_dict.get('data_tags', None)
            if data_tags is not None:
                interaction.record_data_tags(data_tags)
            # If completed, remove from sent tracking
            if interaction_status == InteractionStatus.COMPLETED:
                self.complete(interaction, dest_state=interaction_destination_state, reason=completion_reason)

    def is_received(self, interaction: Interaction) -> bool:
        return interaction.uuid in self.received or interaction in self.received_recently_completed

    def is_sent(self, interaction: Interaction) -> bool:
        return interaction.uuid in self.sent or interaction in self.sent_recently_completed

    def is_lazy(self, interaction: Interaction) -> bool:
        return interaction.uuid in self.lazy or interaction in self.lazy_recently_completed

    def is_known(self, interaction: Interaction):
        return self.is_received(interaction) or self.is_sent(interaction) or self.is_lazy(interaction)

    def get_interaction(self, uuid: str | None, consider_completed_too: bool = False):
        if uuid in self.received:
            return self.received[uuid]
        elif uuid in self.sent:
            return self.sent[uuid]
        elif uuid in self.lazy:
            return self.lazy[uuid]
        else:
            if consider_completed_too:
                found_so_far = None
                for inter in self.received_recently_completed:
                    if inter.uuid == uuid:
                        if found_so_far is None or found_so_far.timestamp_completed < inter.timestamp_completed:
                            found_so_far = inter
                for inter in self.sent_recently_completed:
                    if inter.uuid == uuid:
                        if found_so_far is None or found_so_far.timestamp_completed < inter.timestamp_completed:
                            found_so_far = inter
                for inter in self.lazy_recently_completed:
                    if inter.uuid == uuid:
                        if found_so_far is None or found_so_far.timestamp_completed < inter.timestamp_completed:
                            found_so_far = inter
                return found_so_far  # It returns the last completed, there could be more than one with the same UUID
            else:
                return None

    def add_lazy_stream_to_interaction(self, stream_hash: str, interaction: Interaction):
        user_hashes = self.__normalize_to_user_hashes(stream_hash)
        log.debug(f"[add_lazy_stream_to_interaction] stream_hash={stream_hash}, user_hashes={user_hashes}, "
                  f"interaction={interaction}")
        for user_hash in user_hashes:
            stream_obj = self.agent.known_streams_by_user_hash[user_hash]
            stream_obj.add_interaction(interaction)
            interaction.add_lazy_stream(user_hash, stream_obj,
                                        is_owned=user_hash in self.agent.owned_streams_by_user_hash)

    def remove_interactions_of_agent(self, agent):
        interaction_dicts = [self.sent, self.received, self.lazy]
        to_remove = []
        for interaction_dict in interaction_dicts:
            for uuid, inter in interaction_dict.items():
                if inter.requester == agent:
                    to_remove.append(inter)
                if agent in inter.target:
                    if len(inter.target) > 1:
                        inter.target.remove(agent)
                    else:
                        to_remove.append(inter)
        for inter in to_remove:
            self.complete(inter, reason=CompletionReason.DISCONNECTED)

    def __normalize_to_user_hashes(self, stream_hash):
        user_hashes = []
        if Stream.is_user_hash(stream_hash):
            user_hashes.append(stream_hash)
        elif Stream.is_net_hash(stream_hash):
            net_hash = stream_hash
            if net_hash in self.agent.known_streams:
                streams = self.agent.known_streams[net_hash]
                for stream_obj in streams.values():
                    name = stream_obj.props.name
                    user_hash = DataProps.user_hash_from_net_hash(net_hash, name)
                    user_hashes.append(user_hash)
        return user_hashes

    def __str__(self):
        s1 = ""
        s2 = ""
        s3 = ""
        if len(self.received) > 0 or len(self.received_recently_completed):
            s1 = "Received interactions:\n"
            s1 += "\n".join(([("   " + inter.to_code_str(True)) for inter in self.received.values()] +
                             [("   *" + inter.to_code_str(True)) for inter in self.received_recently_completed]))
        if len(self.sent) > 0 or len(self.sent_recently_completed):
            s2 = "Sent interactions:\n"
            s2 += "\n".join(([("   " + inter.to_code_str(True)) for inter in self.sent.values()] +
                             [("   *" + inter.to_code_str(True)) for inter in self.sent_recently_completed]))
        if len(self.lazy) > 0 or len(self.lazy_recently_completed) > 0:
            s3 = "Lazy interactions:\n"
            s3 += "\n".join(([("   " + inter.to_code_str(True)) for inter in self.lazy.values()] +
                            [("   *" + inter.to_code_str(True)) for inter in self.lazy_recently_completed]))
        return "\n".join([z for z in [s1, s2, s3] if len(z) > 0])


class InteractionStatus(Enum):
    CREATED = "created"
    REQUESTED = "requested"
    LAZY = "lazy"
    RECEIVED = "received"
    RUNNING = "running"
    COMPLETED = "completed"


class CompletionReason(Enum):
    OK = "ok"  # Correctly completed: triggered a transition
    TIMEOUT = "timeout"  # The interaction was waiting in the queue for too long, it's time to remove it
    REJECTED = "rejected"  # The interaction was not accepted since the very beginning
    DISCONNECTED = "disconnected"
    ERROR = "error"  # An error occurred
