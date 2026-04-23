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
import time
import json
import inspect
import uuid as _uuid
from enum import Enum
from unaiverse.clock import clock
from collections.abc import Callable
from unaiverse.utils.logger import log
from unaiverse.streams.dataprops import DataProps
from unaiverse.custom import Custom, GenException
from unaiverse.streams.streamproxy import StreamProxy
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
                 streams: dict[str, str] | list[str] | None = None,
                 data_samples: list | dict | None = None,
                 num_steps: int = -1,
                 requester: str | None = None,
                 target: str | list[str] | None = None,
                 from_state: str | None = None,
                 to_state: str | None = None,
                 timeout: float = -1.,
                 callback: str | None = None,
                 volatile: bool = False,
                 forced_uuid: str | None = "do_not_force",
                 id: str | None = "random"):
        """Create a new Interaction.

        Args:
            action_name: The name of the action being requested (e.g. "process", "learn").
            action_kwargs: Action arguments (dict, no streams).
            streams: List of (stream_name - a.k.a. stream user hash, num_samples) tuples specifying expected data.
            data_samples: List of actual data samples (alternative to samples specification). It can also be a dict
                stream user hash -> data sample, that will load samples on the indicated stream hashes.
            num_steps: Total number of steps for this interaction (-1 means no data-based steps).
            requester: Peer ID of the agent originating this interaction.
            target: Peer ID of the target agent or list of peer IDs.
            from_state: Name of the state from which the interaction is expected to happen.
            to_state: Name of the state where the interaction is supposed to yield.
            timeout: Maximum timeout until which the interaction is valid (-1 means no limit: the InteractionManager
                will override this with its internal timeout valid for all actions).
            callback: Name of the method to call on the agent (the InteractionManager will do it) when this
                interaction finishes (it must be a method with a single argument, which is the interaction object).
            volatile: If True, it is marked so that the recipient is asked to not
                send back any status about its completion.
            forced_uuid: A very custom option to force a specific UUID (None is a valid UUID here), that is by default
                a mixture of the id (below) and the target fields. To be used only in very special cases.
                Defaults to "do_not_force".
            id: Identifier string (might not be unique); ``"random"`` generates one automatically (default).
                When set to None, it defaults to the UUID of a system interaction. The requester will be appended to
                it (separated by "_") to get the final UUID, that is the "unique" identification of the interaction.
        """
        # Participants
        self.requester: str | None = requester
        self.target: list[str | None] = target if isinstance(target, list) else [target]

        # Identity
        self.id: str = _uuid.uuid4().hex[0:8] if (id is not None and id == 'random') else id \
            if id is not None else Custom.SYSTEM_INTERACTION_UUID

        # Indexing string (the actual unique identifier of the interaction, that also involves the requester)
        self.uuid: str = Interaction.build_uuid(self.id, str(self.requester))

        # Forcing a specific UUID (if needed)
        if forced_uuid != "do_not_force":
            self.id = None
            self.uuid = forced_uuid

        # Action requested
        self.action_name: str = action_name
        self.action_kwargs: dict = action_kwargs if action_kwargs is not None else {}

        # Timing information
        self.timestamp_created: float = -1.
        self.timestamp_started: float = -1.
        self.timeout: float = timeout
        self.cycle_created: int = -1
        self.cycle_started: int = -1
        self.timestamp_completed: float = -1.
        self.cycle_completed: int = -1

        # States
        self.from_state: str | None = from_state
        self.to_state: str | None = to_state

        # Samples specification: list of (stream_name, num_samples)
        self.streams: dict = {}
        self.data_samples: list = []
        self.num_steps: int = num_steps  # Generic not-data-based interactions have -1 here

        # Stream-based specification
        if streams is not None and len(streams) > 0:
            self.parse_streams(streams)  # This will create a specific stream structure

        # Actual data (alternative to samples above)
        if streams is None and data_samples is not None:
            if isinstance(data_samples, str):  # If it is JSON string of a list of dicts, each is type_name->base64...
                self.data_samples: list = deserialize_payload(data_samples)  # List of PIL.Image, torch.Tensor, str, ...
            else:
                self.data_samples = data_samples  # List of PIL.Image, torch.Tensor, str, ...
            self.num_steps = 1

        # Status
        self.status: InteractionStatus = InteractionStatus.CREATED
        self.data_sent_after_completion = False
        self.buffered_stream_restarted = False
        self.volatile = volatile

        # Pointers
        self.action_ref: Callable[Interaction] | None = None  # Reference to the actual Action object

        # Data tags
        self.data_tags: dict[str, list[int]] = {}  # Streams could have different data tags

        # Completion reason
        self.completion_reason: CompletionReason | None = None

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
        self.stream_proxy = StreamProxy()  # Virtual stream IO associated to all streams above (not the lazy ones)

        # Target-related info
        self.target_data_tags: list[dict[str, list[int]]] = [{} for _ in range(len(self.target))]
        self.target_cycle_completed: list[int] = [-1] * len(self.target)
        self.target_timestamp_completed: list[float] = [-1.] * len(self.target)
        self.target_completion_reason: list[CompletionReason | None] = [None] * len(self.target)
        self.target_destination_state: list[str | None] = [None] * len(self.target)

        # Back-reference to the InteractionManager (set when registered)
        self.__im: InteractionManager | None = None

        # Interaction type (from Interaction Manager)
        self.type: InteractionType | None = None

        # Callback method name
        self.callback = callback

        # Checking
        if self.callback is not None and self.volatile:
            raise GenException("You cannot set a callback on a volatile interaction")

    @staticmethod
    def build_uuid(id: str | None, requester: str | None = None) -> str:
        if id is None:
            id = Custom.SYSTEM_INTERACTION_ID
        if requester is None:
            requester = Custom.SYSTEM_INTERACTION_LABEL
        return id + "_" + requester

    @staticmethod
    def get_id_from_uuid(uuid: str) -> str:
        return uuid.split("_")[0]

    @property
    def created(self) -> bool:
        """True when the interaction has just been created and not yet requested or received."""
        return self.status == InteractionStatus.CREATED

    @property
    def completed(self) -> bool:
        """True when the interaction has finished."""
        return self.status == InteractionStatus.COMPLETED

    @property
    def running(self) -> bool:
        """True when the interaction is currently running."""
        return self.status == InteractionStatus.RUNNING

    @property
    def paused(self) -> bool:
        """True when the interaction is currently paused."""
        return self.status == InteractionStatus.PAUSED

    @property
    def registered_as_sent(self) -> bool:
        """True when the interaction was registered as 'sent'."""
        return self.type == InteractionType.SENT

    @property
    def registered_as_received(self) -> bool:
        """True when the interaction was registered as 'received'."""
        return self.type == InteractionType.RECEIVED

    def registered_as_lazy(self) -> bool:
        """True when the interaction was registered as 'lazy'."""
        return self.type == InteractionType.LAZY

    def update_target(self, target: str | list[str]):
        if isinstance(target, str):
            target = [target]
        self.target = target
        self.target_completion_reason = [None] * len(self.target)
        self.target_data_tags = [{} for _ in range(len(self.target))]
        self.target_timestamp_completed = [-1.] * len(self.target)
        self.target_destination_state = [None] * len(self.target)
        self.target_cycle_completed = [-1] * len(self.target)

    def set_manager(self,
                    im: 'InteractionManager',
                    stdin_streams: dict[str, Stream | object],
                    stdtar_streams: dict[str, Stream | object],
                    stdext_streams: dict[str, Stream | object],
                    owned_streams: dict[str, Stream | object]) -> None:
        """Bind this interaction to its owning InteractionManager and initialize its stream references.

        Args:
            im: The InteractionManager that owns this interaction.
            stdin_streams: Mapping from user-hash to stream objects used as processor inputs.
            stdtar_streams: Mapping from user-hash to stream objects used as processor targets.
            stdext_streams: Mapping from user-hash to stream objects used as extra (non-input) streams.
            owned_streams: Mapping from user-hash to stream objects owned by this agent.
        """
        self.__im = im
        self.timestamp_created = clock.get_time()
        self.cycle_created = clock.get_cycle()

        self.stdin_streams = stdin_streams
        self.stdtar_streams = stdtar_streams
        self.stdext_streams = stdext_streams
        self.owned_streams = owned_streams
        data_samples_dict = {"<data_sample_" + str(i) + ">": self.data_samples[i]
                             for i in range(0, len(self.data_samples))}
        self.stream_proxy.bind(self.stdin_streams | self.stdtar_streams | self.stdext_streams | self.owned_streams |
                               data_samples_dict, uuid=self.uuid)

    def reset_state(self):
        """Resets the state, including the step counter and timing metrics, allowing it to be re-run from the
        beginning.
        """
        self.__step_idx = -1
        self.__starting_time = 0.
        self.__timeout_starting_time = 0.

    def set_mark(self, mark: object) -> None:
        """Store an arbitrary marker object on the interaction."""
        self.__mark = mark

    def get_mark(self) -> object:
        """Return the marker object previously set by set_mark."""
        return self.__mark

    def clear_mark(self) -> None:
        """Clear the arbitrary marker object on the interaction."""
        self.__mark = None

    def get_step_idx(self) -> int:
        """Return the index of the last completed step (-1 if no steps done yet)."""
        return self.__step_idx

    def set_step_idx(self, steps: int) -> None:
        """Directly assign the current step index.

        Args:
            steps: The step index to set.
        """
        self.__step_idx = steps

    def inc_step_idx(self) -> None:
        """Increment the step index by one."""
        self.__step_idx += 1

    def dec_step_idx(self) -> None:
        """Decrement the step index by one."""
        self.__step_idx -= 1

    def set_starting_time(self, t: float) -> None:
        """Record the wall-clock time at which execution started.

        Args:
            t: Starting timestamp (from ``time.perf_counter()``).
        """
        self.__starting_time = t

    def set_timeout_starting_time(self, t: float) -> None:
        """Record the wall-clock time used as the baseline for next-step timeout checks.

        Args:
            t: Timeout baseline timestamp (from ``time.perf_counter()``).
        """
        self.__timeout_starting_time = t

    def get_total_steps(self) -> int:
        """Retrieves the total number of steps configured for the action.

        Returns:
            An integer representing the total steps.
        """
        return self.num_steps

    def get_starting_time(self) -> float:
        """Retrieves the timestamp when the action's current execution started.

        Returns:
            A float representing the starting time.
        """
        return self.__starting_time

    def get_timeout_starting_time(self) -> float:
        """Retrieves the timestamp used as the baseline for next-step timeout checks.

        Returns:
            A float representing the timeout starting time.
        """
        return self.__timeout_starting_time

    def is_multi_steps(self) -> bool:
        """Determines if the action is configured to be a multistep action (i.e., not a single-step action).

        Returns:
            A boolean indicating if the action is multistep.
        """
        return self.num_steps > 1

    def is_single_step(self) -> bool:
        """Return True if this is a single-step action (one data sample or no data at all)."""
        return self.num_steps == 1 or self.num_steps < 0  # Action with a single data sample or no data samples at all

    def is_valid(self) -> bool:
        """Return True if the interaction has been assigned insertion-order IDs by the action."""
        return self.by_insertion_order_id >= 0 and self.by_requester_insertion_order_id >= 0

    def is_system(self) -> bool:
        """Return True if the interaction is a system-level interaction."""
        return self.requester == Custom.SYSTEM_INTERACTION_LABEL

    def is_completed(self) -> bool:
        """Return True if the interaction status is COMPLETED."""
        return self.status == InteractionStatus.COMPLETED

    def was_data_sent_after_completion(self) -> bool:
        """Return True if data was generated and sent after this interaction completed."""
        return self.data_sent_after_completion

    def has_dummy_requester(self) -> bool:
        """Return True if no requester peer ID is set (system-generated interaction)."""
        return self.requester is None

    def set_arg(self, arg_name: str, arg_value: object) -> None:
        """Set or overwrite an action keyword argument.

        Args:
            arg_name: The argument name.
            arg_value: The value to assign.
        """
        self.action_kwargs[arg_name] = arg_value

    def get_arg(self, arg_name: str) -> object:
        """Retrieve an action keyword argument by name.

        Args:
            arg_name: The argument name to look up.

        Returns:
            The argument value, or ``None`` if the argument is not present.
        """
        return self.action_kwargs[arg_name] if arg_name in self.action_kwargs else None

    def was_at_least_one_step_done(self) -> bool:
        """Return True if at least one execution step has been completed."""
        return self.__step_idx >= 0

    def was_last_step_done(self) -> bool:
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

    def set_action_ref(self, action_ref: object) -> None:
        """Bind the interaction to its Action object and validate/augment its argument list.

        Args:
            action_ref: The Action instance that will execute this interaction.
        """
        self.action_ref = action_ref

        # Augmenting argument list, by fusing with the arguments defined in the HSM
        action_kwargs = self.action_kwargs if self.action_kwargs is not None else {}
        self.action_ref.check_provided_args(action_kwargs, exception=True)

        # Force a default timeout on multistep actions, to avoid infinite trials
        if self.is_multi_steps() and self.action_ref.get_timeout() <= 0:
            self.action_ref.set_default_timeout()

    def has_stream(self, user_hash: str) -> bool:
        """Return True if a stream with the given user hash is already registered.

        Args:
            user_hash: The stream's user-level hash identifier.
        """
        return user_hash in self.stream_proxy or user_hash in self.lazy_streams

    def add_lazy_stream(self, user_hash: str, stream_obj: Stream, is_owned: bool = False) -> None:
        """Lazily attach an additional stream to this interaction after it was created.

        Args:
            user_hash: The stream's user-level hash identifier.
            stream_obj: The stream object to attach.
            is_owned: Whether the stream is owned by the current agent.
        """
        if not self.has_stream(user_hash):
            self.lazy_streams[user_hash] = stream_obj
            if is_owned:
                self.owned_streams[user_hash] = stream_obj
            self.stream_proxy.add_new_bind(user_hash, stream_obj)

    def mark_running(self) -> None:
        """Mark this interaction as currently running."""
        self.status = InteractionStatus.RUNNING
        if self.timestamp_started <= 0:
            self.timestamp_started = clock.get_time()
            self.cycle_started = clock.get_cycle()

    def mark_paused(self) -> None:
        """Mark this interaction as currently paused."""
        self.status = InteractionStatus.PAUSED

    def mark_completed(self, reason: 'CompletionReason', dest_state: str | None = None,
                       target: str | None = None) -> None:
        """Mark this interaction as completed.

        Args:
            reason: The reason for completion.
            dest_state: The destination state reached by completing this action.
            target: The target agent who completed this interaction (Default: None).
        """
        if target is None:
            self.destination_state = dest_state
            self.completion_reason = reason
            self.timestamp_completed = clock.get_time()
            self.cycle_completed = clock.get_cycle()
            self.status = InteractionStatus.COMPLETED

            _time = clock.get_time()
            _cycle = clock.get_cycle()
            for i, tar in enumerate(self.target):
                if self.target_completion_reason[i] is None:
                    self.target_completion_reason[i] = reason  # The completion reason of the whole interaction
                    self.target_timestamp_completed[i] = _time
                    self.target_cycle_completed[i] = _cycle
        else:
            num_completed = 0
            _time = clock.get_time()
            _cycle = clock.get_cycle()
            for i, tar in enumerate(self.target):
                if tar == target:
                    self.target_destination_state[i] = dest_state
                    self.target_completion_reason[i] = reason
                    self.target_timestamp_completed[i] = _time
                    self.target_cycle_completed[i] = _cycle
                if self.target_completion_reason[i] is not None:
                    num_completed += 1
            if num_completed == len(self.target):
                self.mark_completed(reason)  # The completion reason of the last target who completed

    def clear_from_action(self) -> None:
        """Deregister this interaction from its action's interaction list."""

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

    def get_new_stream_data_tags(self, all_fresh_or_fail: bool = False) -> dict[str, list[int]] | None:
        """Checks all streams involved in this interaction, and if they have fresh data it returns their tags.

        Args:
            all_fresh_or_fail: If True, return None if there is at least one stream with no fresh data (Default: False).

        Returns:
            A dictionary "stream user hash to data tag", only involving streams with fresh data. It can return None if
                the all_fresh_or_fail option was turned on.
        """
        data_tags = {}
        for stream_user_hash, stream_obj in self.stream_proxy.items():

            # Skipping data samples and default input values
            if not isinstance(stream_obj, Stream):
                continue

            if (stream_user_hash in self.data_tags and
                    len(self.data_tags[stream_user_hash]) > 0):
                last_handled_data_tag = self.data_tags[stream_user_hash][-1]
            else:
                last_handled_data_tag = None

            # Check if stream has data with a tag not yet consumed by this interaction
            current_tag = stream_obj.get_tag(self.uuid)
            if last_handled_data_tag is not None and current_tag == last_handled_data_tag:
                if all_fresh_or_fail:
                    return None
                else:
                    continue

            # Storing
            data_tags[stream_user_hash] = [current_tag]
        return data_tags

    def record_data_tags(self, data_tags: dict[str, list[int]] | None = None, target: str | None = None):
        """Record that data with the given tag was received for this interaction.

        Args:
            data_tags: The data tags/sequence numbers.
            target: The agent who handled the data tags (if None, then it's the current agent).
        """
        if data_tags is None:
            data_tags: dict[str, list[int]] = self.get_new_stream_data_tags()

        if len(data_tags) == 0:
            return

        if target is None:
            handled_data_tags = self.data_tags
        else:
            if target in self.target:
                i = self.target.index(target)
                handled_data_tags = self.target_data_tags[i]
            else:
                log.error(f"Invalid target {target} specified to record data tags: "
                          f"it does not exist in the list of targets ({self.target}) of "
                          f"interaction with UUID {self.uuid}")
                return

        _time = clock.get_time()
        for stream_name, list_of_data_tag in data_tags.items():
            if stream_name not in handled_data_tags:
                handled_data_tags[stream_name] = []
            handled_data_tags[stream_name] += list_of_data_tag

    def check_if_doable(self) -> bool:
        """Return True if the interaction can currently be executed.

        Delegates to the owning InteractionManager when present; for system interactions without a manager,
        returns True as long as the interaction is not yet completed.
        """
        if self.__im is not None:
            return self.__im.check_if_doable(self)
        else:
            return not self.completed  # System interactions does not have an interaction manager

    def alter_arg(self, arg_name: str, arg_value: object) -> bool:
        """Overwrite an existing action keyword argument, leaving the dict unchanged if the key is absent.

        Args:
            arg_name: The argument name to update.
            arg_value: The new value to assign.

        Returns:
            True if the argument existed and was updated; False otherwise.
        """
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
            'id': self.id,
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
            'volatile': self.volatile,
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
            id=d['id'],
            action_name=d['action_name'],
            action_kwargs=d.get('action_kwargs', {}),
            streams=d.get('streams', {}),
            data_samples=d.get('data_samples', None),
            num_steps=d.get('num_steps'),
            requester=d.get('requester'),
            target=d.get('target'),
            from_state=d.get('from_state'),
            to_state=d.get('to_state'),
            volatile=d.get('volatile'),
            timeout=d.get('timeout', -1.),
        )
        interaction.uuid = d['uuid']
        interaction.status = InteractionStatus(d['status'])
        if d.get('completion_reason'):
            interaction.completion_reason = CompletionReason(d['completion_reason'])
        interaction.destination_state = d.get('destination_state')
        return interaction

    def to_status_dict(self, status_generated_by: str) -> dict:
        """Build a minimal status dict for status update messages.

        Args:
            status_generated_by: The agent who generate this dict.

        Returns:
            A dictionary with status_generated_by, uuid, status, completion_reason, destination_state, and data_tag.
        """
        return {
            'status_generated_by': status_generated_by,
            'uuid': self.uuid,
            'status': self.status.value,
            'completion_reason': self.completion_reason.value if self.completion_reason else None,
            'destination_state': self.destination_state,
            'data_tags': self.data_tags if len(self.data_tags) > 0 else None,
        }

    def to_code_str(self, include_id: bool = False, include_received_tags: bool = False) -> str:
        """Return a compact single-line representation for logs and debugging.

        Args:
            include_id: Prepend the interaction UUID to the output when True.
            include_received_tags: Append the received data tags when True.

        Returns:
            A compact string describing the interaction.
        """
        s = ""
        t: list[str] = self.target

        if include_id:
            s = f"{self.uuid} => "
        if include_received_tags:
            t: list[str] = []
            for i, tar in enumerate(self.target):
                all_tags = [str(num) for sublist in self.target_data_tags[i].values() for num in sublist]
                if all_tags:
                    t.append(f"{tar}_(" + ",".join(all_tags) + ")")
                else:
                    t.append(f"{tar}")

        return s + (f"artss:{self.action_name}|{self.requester}|{t}|"
                    f"{self.stream_proxy}|{self.status.value[0:3]}" +
                    (("_" + self.completion_reason.value[0:3]) if self.completion_reason is not None else ""))

    def to_str(self) -> str:
        """Return a JSON-encoded string with the core interaction fields for logging.

        Returns:
            A JSON string containing requester, action_kwargs, timestamp_created, and uuid.
        """
        return json.dumps([self.requester, self.action_kwargs, self.timestamp_created, self.uuid])

    def parse_streams(self, streams: list | dict):
        """Parse the list of streams provided by the user, standardizing its format and saving it.

        The user can specify streams in these two ways, where "stream_hash_j" can be just the stream name, the stream
        group, the net hash, or the user hash.
            (1) [..., stream_hash_j (str), ...]
            (2) {
                    "stdin": [..., stream_hash_j (str), ...],
                    "stdtar": [..., stream_hash_k (str), ...],
                    "stdext": [..., stream_hash_z (str), ...],
                    "stdunk": [..., stream_hash_z (str), ...],
                }

            The last one (2) optionally specifies if the stream is expected to become an input of the processor
            ('stdin'), a target ('stdtar'), and extra stream ('stdext'), or no preferences (None).
            Notice that this choice could be re-arranged by the Interaction Manager, in function of the actual
            capabilities of the processor.

        Args:
            streams (list | dict): The list or dict of streams (see the example above).
        """
        invalid_msg = f'Invalid syntax for streams involved in an interaction: {streams}'
        self.streams = {"stdin": [], "stdtar": [], "stdext": [], "stdunk": []}

        if isinstance(streams, list):
            for stream in streams:
                if not isinstance(stream, str):
                    raise GenException(invalid_msg)
                self.streams["stdunk"].append(stream)
        elif isinstance(streams, dict):
            for redirect, streams_list in streams.items():
                if redirect not in self.streams:
                    raise GenException(invalid_msg)
                for stream in streams_list:
                    if not isinstance(stream, str):
                        raise GenException(invalid_msg)
                self.streams[redirect] = streams_list
        else:
            raise GenException(invalid_msg)

    def __str__(self) -> str:
        """Return a compact string representation of this Interaction."""
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

    def __init__(self, agent: object, max_interactions: int = -1):
        """Create an InteractionManager.

        Args:
            agent: Back-reference to the owning agent (AgentBasics instance).
            max_interactions: Maximum number of simultaneously tracked interactions (sent + received + lazy).
        """
        self.agent = agent
        self.max_interactions = max_interactions if max_interactions > 0 else Custom.MAX_INTERACTIONS
        self.sent: dict[str, Interaction] = {}       # uuid -> Interaction (sent by this agent)
        self.received: dict[str, Interaction] = {}   # uuid -> Interaction (received from others)
        self.lazy: dict[str, Interaction] = {}   # uuid -> Interaction (added by you)
        self.current: Interaction | None = None      # Currently executing interaction
        self.last_registered: Interaction | None = None
        self.sent_recently_completed: set[Interaction] = set()  # Completed interactions
        self.received_recently_completed: set[Interaction] = set()  # Completed interactions
        self.lazy_recently_completed: set[Interaction] = set()  # Completed interactions

    def room_for_registration(self) -> bool:
        """Return True if there is capacity to register another interaction."""
        return len(self.sent) + len(self.received) + len(self.lazy) < self.max_interactions

    def count_interactions(self) -> int:
        """Return the total number of tracked interactions, including recently completed ones."""
        return (len(self.sent) + len(self.received) + len(self.lazy) +
                len(self.sent_recently_completed) + len(self.received_recently_completed) +
                len(self.lazy_recently_completed))

    def clear_from_all_streams(self, interaction: Interaction) -> None:
        """Remove the given interaction from every stream that references it.

        Args:
            interaction: The interaction to deregister from all the known streams.
        """
        for stream_obj in self.agent.known_streams_by_user_hash.values():
            if stream_obj.has_interaction(interaction.uuid):
                stream_obj.remove_interaction(interaction)

    def unregister_all(self):
        """Deregister all current interactions"""
        all_interactions = (set(self.sent.values()) | set(self.received.values()) | set(self.lazy.values()))
        for interaction in all_interactions:
            self.unregister(interaction)

    def unregister(self, interaction: Interaction) -> bool:
        """Deregister an interaction from all tracking dicts and dissociate it from streams and actions.

        Args:
            interaction: The interaction to remove.

        Returns:
            True if the interaction was found and removed from at least one tracking dict.
        """
        if interaction.status == InteractionStatus.RUNNING:  # Avoid de-registering a currently running interaction
            return False

        interaction.clear_from_action()
        self.clear_from_all_streams(interaction)
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

        Returns:
            True if registration succeeded; False if there is no room or the streams are invalid.
        """
        self.__check_callback_method(interaction)
        if not self.room_for_registration():
            log.error(f"No more room for interactions (limit: {self.max_interactions})")
            return False

        # This will resolve target names, data samples, stream names
        if not self.resolve(interaction):
            return False

        #  Ensuring all the streams mentioned in the interaction are known, and normalizing them
        _, expanded_owned_streams = self.expand_and_normalize_streams(interaction)
        if expanded_owned_streams is None:
            log.error(f"Invalid stream field in interaction: {interaction}")
            return False

        # Converting format for owned streams (no matter what the matching routine decided)
        owned_user_hashes_to_stream_objs = {_stream_user_hash: self.agent.known_streams_by_user_hash[_stream_user_hash]
                                            for _streams_list in expanded_owned_streams.values()
                                            for _stream_user_hash in _streams_list}

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
        interaction.type = InteractionType.SENT
        self.last_registered = interaction
        return True

    def __register_received_or_lazy(self, interaction: Interaction) -> bool:
        """Register an interaction received from another agent or auto-created (lazy).

        Args:
            interaction: The Interaction to register.

        Returns:
            True if registration succeeded; False if there is no room, streams are invalid, or matching failed.
        """
        self.__check_callback_method(interaction)
        if not self.room_for_registration():
            log.error(f"No more room for interactions (limit: {self.max_interactions})")
            return False

        if interaction.requester not in self.agent.all_agents:
            log.error(f"Unexpected interaction from an unknown agent: {interaction.requester}. "
                      f"Interaction is: {interaction}")
            return False

        # This will resolve target names, data samples, stream names
        if not self.resolve(interaction, resolve_target=False):  # You are the target, no need to further resolve it
            return False

        # Ensuring all the streams mentioned in the interaction are known, and normalizing them
        expanded_streams, expanded_owned_streams = self.expand_and_normalize_streams(interaction)
        if expanded_streams is None:
            log.error(f"Invalid stream field in interaction: {interaction}")
            return False

        # 1. Filling the missing field of the different stream dictionaries, and assigning streams to stdin or stdext,
        # following the suggestions provided in 'redirect', when possible
        # 2. Returning False if there were no ways to fill the processor input arguments.
        (valid, stdin_user_hashes_to_stream_objs, stdtar_user_hashes_to_stream_objs,
         stdext_user_hashes_to_stream_objs, owned_user_hashes_to_stream_objs) = self.match_streams(expanded_streams)
        if not valid:
            return False

        # Updating the interaction object with the decisions from the manager
        interaction.set_manager(self,
                                stdin_streams=stdin_user_hashes_to_stream_objs,
                                stdtar_streams=stdtar_user_hashes_to_stream_objs,
                                stdext_streams=stdext_user_hashes_to_stream_objs,
                                owned_streams=owned_user_hashes_to_stream_objs)

        # Registering the interaction in the involved streams (do this AFTER calling set_manager)
        for stream in interaction.stream_proxy:

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
            stream.add_interaction(interaction)

        # Registering
        self.last_registered = interaction
        return True

    def register_received(self, interaction: Interaction) -> bool:
        """Register an interaction received from another agent.

        Args:
            interaction: The Interaction to register.

        Returns:
            True if registration succeeded; False if there is no room, streams are invalid, or matching failed.
        """
        if self.__register_received_or_lazy(interaction):
            self.received[interaction.uuid] = interaction

            interaction.status = InteractionStatus.RECEIVED
            interaction.type = InteractionType.RECEIVED
            return True
        else:
            return False

    def register_lazy(self, interaction: Interaction) -> bool:
        """Register an interaction that you manually generated within this agent.

        Args:
            interaction: The Interaction to register.

        Returns:
            True if registration succeeded; False if there is no room or the streams are invalid.
        """
        if self.__register_received_or_lazy(interaction):
            self.lazy[interaction.uuid] = interaction

            interaction.status = InteractionStatus.LAZY
            interaction.type = InteractionType.LAZY
            return True
        else:
            return False

    def resolve(self, interaction: Interaction, resolve_target: bool = True) -> bool:

        # A. Resolving targets
        if resolve_target:
            target = interaction.target
            if target is not None:
                for i, _target in enumerate(target):
                    _target = self.agent.resolve_agent_ref(_target)
                    if _target is not None:
                        target[i] = _target
                    else:
                        log.error(f"Unknown target specified in an interaction ({_target})")
                        return False

        # B. Moving data samples to streams, if needed (when data_samples is a dictionary)
        # In this case, "streams" must be None (if not None, then the data_samples field is ignored)
        if isinstance(interaction.data_samples, dict):

            # This is already an empty list (otherwise the data_samples field would not be there), but better
            # be extra safe
            streams = []
            for stream_user_hash, data_sample in interaction.data_samples.items():
                stream_user_hash = self.agent.resolve_stream_ref(stream_user_hash)
                if stream_user_hash is None:
                    log.error(f"Unknown stream hash specified in an interaction ({stream_user_hash}")
                    return False
                stream: Stream = self.agent.known_streams_by_user_hash[stream_user_hash]
                stream.set(data_sample, uuid=interaction.uuid)

                # Appending the stream to the interaction, that now looks like a stream-based interaction
                streams.append(stream_user_hash)

            # Purging
            interaction.data_samples.clear()
            interaction.parse_streams(streams)  # Rebuilding the stream dictionaries

        # C. Resolving streams
        streams = interaction.streams
        if streams is not None:
            for redirect, streams_list in streams.items():
                for j, stream in enumerate(streams_list):
                    stream_user_hash = self.agent.resolve_stream_ref(stream)
                    if stream_user_hash is None:
                        log.error(f"Unknown stream hash specified in an interaction ({stream_user_hash}")
                        return False
                    streams_list[j] = stream_user_hash
        return True

    def expand_and_normalize_streams(self, interaction: Interaction) -> (
            tuple)[dict[str | None, list[str]], dict[str | None, list[str]]]:
        """Resolve all stream references in an interaction to fully-populated stream dicts.

        Translates user-level and net-level hashes into detailed dicts containing ``user_hash``,
        ``net_hash``, ``name``, ``group``, and the stream object itself, grouped by their suggested
        redirection (``'stdin'``, ``'stdtar'``, ``'stdext'``, or ``None``).

        Args:
            interaction: The interaction whose stream list should be expanded.

        Returns:
            A 2-tuple ``(expanded_streams, expanded_owned_streams)`` where each element is a dict
            keyed by redirection hint (``'stdin'``, ``'stdtar'``, ``'stdext'``, ``None``) mapping to
            lists of fully-populated stream dicts.  Returns ``(None, None)`` if any stream hash cannot
            be resolved.
        """

        # Guessing net hash and specific name af each stream: if the name was not provided, then all the streams of the
        # group are considered. Generating a full list of streams, distinguishing them in function of their suggested
        # redirection, i.e., 'stdin', 'stdtar', 'stdext', None (meaning 'no suggestions').
        expanded_streams = {'stdin': [], 'stdtar': [], 'stdext': [], "stdunk": []}
        expanded_owned_streams = {'stdin': [], 'stdtar': [], 'stdext': [], "stdunk": []}

        # Checking
        if len(interaction.streams) == 0:
            return expanded_streams, expanded_owned_streams

        for redirect, streams_list in interaction.streams.items():
            for stream_hash in streams_list:

                if Stream.is_user_hash(stream_hash):
                    expanded_streams[redirect].append(stream_hash)
                    if stream_hash in self.agent.owned_streams_by_user_hash:
                        expanded_owned_streams[redirect].append(stream_hash)

                elif Stream.is_net_hash(stream_hash):
                    streams = self.agent.known_streams[stream_hash]
                    for stream_obj in streams.values():
                        _stream_hash = DataProps.user_hash_from_net_hash(stream_hash, stream_obj.props.get_name())
                        expanded_streams[redirect].append(_stream_hash)
                        if _stream_hash in self.agent.owned_streams_by_user_hash:
                            expanded_owned_streams[redirect].append(_stream_hash)
        return expanded_streams, expanded_owned_streams

    def match_streams(self, expanded_streams: dict) -> tuple[bool, dict, dict, dict, dict]:
        """Assign expanded stream dicts to stdin, stdtar, or stdext based on processor compatibility.

        Attempts to match streams to the processor's input and output argument slots, following any
        redirection hints.  Streams that do not fit a processor slot are placed in stdext.

        Args:
            expanded_streams: The output of ``expand_and_normalize_streams``; a dict keyed by
                redirection hint (``'stdin'``, ``'stdtar'``, ``'stdext'``, ``None``) containing lists
                of fully-populated stream dicts.

        Returns:
            A 4-tuple ``(valid, stdin_streams, stdtar_streams, stdext_streams)`` where ``valid`` is
            False if required processor inputs have no matching stream and no default value, and the
            remaining three elements are ``{user_hash: stream_obj}`` dicts for each category.
        """

        # Assigning streams to 'stdin' or 'stdext', following the given suggestions, when possible, and being a bit
        # heuristic for all the not-well-defined cases
        processor_will_be_used = len(expanded_streams['stdin']) > 0 or len(expanded_streams["stdunk"]) > 0
        stdin_streams = {}
        stdtar_streams = {}
        stdext_streams = {}
        owned_streams = {}

        for user_hash in expanded_streams['stdext']:
            if user_hash not in self.agent.known_streams_by_user_hash:
                return False, {}, {}, {}, {}
            stdext_streams[user_hash] = self.agent.known_streams_by_user_hash[user_hash]
            if user_hash in self.agent.owned_streams_by_user_hash:
                owned_streams[user_hash] = stdext_streams[user_hash]

        # We try to match the 'stdin' suggestions with the different input arguments of the processors.
        # We also consider the streams with no specific suggestions (lower priority).
        if processor_will_be_used:
            sources = ['stdin', 'stdunk']
            pos_stdin_streams = [None] * len(self.agent.proc_inputs)
            for i in range(len(self.agent.proc_inputs)):
                found_match = -1
                for source in sources:
                    for j, user_hash in enumerate(expanded_streams[source]):
                        if user_hash not in self.agent.known_streams_by_user_hash:
                            return False, {}, {}, {}, {}
                        stream_obj = self.agent.known_streams_by_user_hash[user_hash]
                        net_hash = DataProps.build_net_hash(DataProps.peer_id_from_user_hash(user_hash),
                                                            stream_obj.is_pubsub(),
                                                            stream_obj.props.name_or_group())

                        name = stream_obj.props.get_name()

                        # If the current input stream is compatible with the i-th input slot...
                        if (net_hash, name) in self.agent.compat_in_streams[i]:
                            pos_stdin_streams[i] = user_hash
                            found_match = j
                            break

                    if found_match >= 0:
                        del expanded_streams[source][found_match]  # Removing the stream from the suggestions
                        break

            # We ensured that the not-filled-arguments of the processor have a default value, otherwise no ways
            for i in range(len(self.agent.proc_inputs)):
                if pos_stdin_streams[i] is None:
                    if not self.agent.proc_optional_inputs[i]["has_default"]:
                        return False, {}, {}, {}, {}
                    else:
                        stdin_streams["<default_input_pos_" + str(i) + ">"] = (
                            self.agent.proc_optional_inputs)[i]["default_value"]
                else:
                    stdin_streams[pos_stdin_streams[i]] = self.agent.known_streams_by_user_hash[pos_stdin_streams[i]]
                    if pos_stdin_streams[i] in self.agent.owned_streams_by_user_hash:
                        owned_streams[pos_stdin_streams[i]] = stdin_streams[pos_stdin_streams[i]]

            # Matching targets
            sources = ['stdtar', 'stdunk']
            pos_stdtar_streams = [None] * len(self.agent.proc_outputs)
            for i in range(len(self.agent.proc_outputs)):
                found_match = -1
                for source in sources:
                    for j, user_hash in enumerate(expanded_streams[source]):
                        if user_hash not in self.agent.known_streams_by_user_hash:
                            return False, {}, {}, {}, {}
                        stream_obj = self.agent.known_streams_by_user_hash[user_hash]
                        net_hash = DataProps.build_net_hash(DataProps.peer_id_from_user_hash(user_hash),
                                                            stream_obj.is_pubsub(),
                                                            stream_obj.props.get_group())

                        name = stream_obj.props.get_name()

                        # If the current input stream is compatible with the i-th input slot...
                        if (net_hash, name) in self.agent.compat_out_streams[i]:
                            pos_stdtar_streams[i] = user_hash
                            found_match = j
                            break

                    if found_match >= 0:
                        del expanded_streams[source][found_match]  # Removing the stream from the suggestions
                        break

            # We ensured that the not-filled-arguments of the processor have a default value, otherwise no ways
            for i in range(len(self.agent.proc_outputs)):
                if pos_stdtar_streams[i] is not None:
                    stdtar_streams[pos_stdtar_streams[i]] = self.agent.known_streams_by_user_hash[pos_stdtar_streams[i]]
                    if pos_stdtar_streams[i] in self.agent.owned_streams_by_user_hash:
                        owned_streams[pos_stdtar_streams[i]] = stdtar_streams[pos_stdtar_streams[i]]

        # We add to 'stdext' all the streams that did not fit the processor (both coming from suggestions in 'stdin' or
        # not suggested at all)
        sources = ['stdin', 'stdtar', 'stdunk']
        for source in sources:
            for user_hash in expanded_streams[source]:
                if user_hash not in self.agent.known_streams_by_user_hash:
                    return False, {}, {}, {}, {}
                stdext_streams[user_hash] = self.agent.known_streams_by_user_hash[user_hash]
                if user_hash in self.agent.owned_streams_by_user_hash:
                    owned_streams[user_hash] = stdext_streams[user_hash]

        return True, stdin_streams, stdtar_streams, stdext_streams, owned_streams

    def check_if_doable(self, interaction: Interaction) -> bool:
        """Return True if the given interaction can be executed right now.

        An interaction is doable when it is not yet completed, the requester is a known agent, and
        either all involved streams have fresh data or a deprecated completion step is pending.

        Args:
            interaction: The interaction to evaluate.
        """
        requester_is_known = interaction.requester in self.agent.all_agents
        run_deprecated_completion_step = ((interaction.is_timed_out() and
                                           interaction.was_at_least_one_step_done()) or
                                          interaction.was_last_step_done())
        return (not interaction.completed and requester_is_known and
                (interaction.get_new_stream_data_tags(all_fresh_or_fail=True) is not None or
                 run_deprecated_completion_step))

    def get_current(self) -> Interaction | None:
        """Return the currently executing interaction, or None if no interaction is running."""
        return self.current

    def get_last_registered(self) -> Interaction | None:
        """Return the most recently registered interaction, or None if none have been registered."""
        return self.last_registered

    def set_current_as_paused(self) -> None:
        """Set the current interaction as something that was started but now must be paused because some other
        interactions are handled."""
        if self.current is not None:
            self.current.mark_paused()

    def set_current(self, interaction: Interaction | None) -> None:
        """Set the given interaction as the currently executing one.

        This marks the interaction as running and configures the agent's
        stdin/stdout to point to the correct streams for this interaction.

        Args:
            interaction: The Interaction to set as current.
        """
        self.current = interaction

        # Default stream bindings (this also automatically switches from private to public and vice-versa)
        self.agent.set_default_stream_binding()

        if interaction is not None:
            interaction.mark_running()

            if len(interaction.stream_proxy) > 0:

                # Restart buffered streams and activate them if they were off
                for stream in interaction.stream_proxy:
                    if isinstance(stream, BufferedStream):
                        stream.restart(interaction.uuid)

                # Every interaction with some-streams-specified forces the interaction-described bindings
                self.agent.stdin.bind(interaction.stdin_streams, uuid=interaction.uuid)
                self.agent.stdtar.bind(interaction.stdtar_streams, uuid=interaction.uuid)
                self.agent.stdext.bind(interaction.stdext_streams, uuid=interaction.uuid)

    def has_data(self, interaction: 'Interaction') -> bool:
        """Return True if any owned stream has data available for the given interaction.

        Args:
            interaction: The interaction to check data availability for.
        """
        for stream_obj in self.agent.owned_streams_by_user_hash.values():
            if stream_obj.has_data(interaction.uuid):
                return True
        return False

    def get_recipients(self, interaction: 'Interaction') -> list[str]:
        """Return the list of peer IDs that should receive data generated by the interaction.

        For sent interactions the recipients are the interaction targets; for received interactions it
        is the original requester; for lazy interactions it is the interaction targets.  Unknown
        (unregistered) interactions fall back to the interaction's target list for backward
        compatibility.

        Args:
            interaction: The interaction whose recipients should be determined.

        Returns:
            A list of peer ID strings, filtering out any ``None`` entries.
        """
        if self.is_known(interaction):
            if self.is_sent(interaction):
                recipients = interaction.target  # This is always a list, even when with 1 element only
            elif self.is_received(interaction):
                recipients = [interaction.requester]  # This is always 1 element, we make it a list
            elif self.is_lazy(interaction):
                recipients = interaction.target  # This is always a list, even when with 1 element only
            else:
                raise GenException("Unexpected case of a known interaction that is both not sent or received")
        else:

            # This is the case of a not-registered interaction.
            # It is exploited for backward compatibility, for those interactions whose only purpose
            # is to provide a recipient over a stream
            recipients = interaction.target  # This is always a list, even when with 1 element only
        return [x for x in recipients if x is not None]

    async def complete(self, interaction: 'Interaction', reason: 'CompletionReason',
                       dest_state: str | None = None, target: str | None = None) -> None:
        """Mark an interaction as completed and move it to the recently-completed set, removing it from its action
        (async).

        Args:
            interaction: The interaction to complete.
            reason: The reason for completion.
            dest_state: The destination state reached, if any.
            target: The target agent who completed the interaction (Default: None).
        """
        if interaction is not None:
            interaction.mark_completed(reason, dest_state=dest_state, target=target)
            interaction.clear_from_action()  # We do not remove it from streams yet - it might be needed for sending
            if interaction.completed:
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

                    # Clearing the human module related case, if needed
                    self.agent.clear_stdin_backlog_if_human(interaction.requester)

                # Running callback method, if any
                if interaction.callback is not None:
                    callback_method = getattr(self.agent, interaction.callback)

                    self.agent.behav_lone_wolf.enable(False)
                    self.agent.behav.enable(False)
                    if interaction.requester in self.agent.public_agents:
                        self.agent.behav_lone_wolf.enable(True)
                    else:
                        if self.agent.in_world():
                            self.agent.behav.enable(True)

                            await callback_method(interaction=interaction)  # "Calling callback!"

                    self.agent.behav_lone_wolf.enable(False)
                    self.agent.behav.enable(False)

    async def complete_current(self, dest_state: str, reason: 'CompletionReason') -> None:
        """Mark the current interaction as completed (async).

        Args:
            dest_state: The destination state reached by completing this action.
            reason: The reason for completion.
        """
        await self.complete(self.current, dest_state=dest_state, reason=reason)
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
                    log.error(
                        f"Draining {interaction.to_code_str(True)} "
                        f"(cycle_completed={interaction.cycle_completed})")
                    to_remove.append(interaction)
                    drained.append(interaction)
            for interaction in to_remove:
                self.sent_recently_completed.discard(interaction)
                self.received_recently_completed.discard(interaction)
                self.lazy_recently_completed.discard(interaction)

                # Clearing from all the owned and not-owned streams (all), that might have been used for output purposes
                # by the current agent or by others.
                # In principle, only the processor streams should be involved, since it is the only one in which we plug
                # this interaction in this class.
                # However, the user might have added the interaction to other owned streams.
                if not self.is_known(interaction):
                    # If, meanwhile, another interaction with the same UUID (e.g., None) was received,
                    # do not clear it from the streams
                    self.clear_from_all_streams(interaction)

        return drained

    async def complete_expired(self) -> None:
        """Remove expired (or no-action-based) interactions and return them for notification (async).

        Checks all received and sent interactions against the timeout.
        Expired interactions are marked as COMPLETED with TIMEOUT reason
        and removed from the tracking dicts.
        """
        for interaction in list(self.sent.values()) + list(self.received.values()) + list(self.lazy.values()):
            if interaction.status == InteractionStatus.COMPLETED:
                continue
            if (interaction.is_expired(Custom.DEFAULT_INTER_TIMEOUT) or interaction.action_name is None or
                    interaction.volatile):
                await self.complete(interaction, reason=CompletionReason.TIMEOUT)
            else:
                if self.is_received(interaction) and interaction.requester not in self.agent.all_agents:
                    await self.complete(interaction, reason=CompletionReason.DISCONNECTED)
                if self.is_sent(interaction):
                    for target in interaction.target:
                        if target not in self.agent.all_agents:
                            await self.complete(interaction, reason=CompletionReason.DISCONNECTED, target=target)

    def clear_expired_stream_data(self) -> None:
        """Purge stale buffered data from streams whose associated interaction is done or gone."""
        all_interactions = list(self.received.values()) + list(self.sent.values()) + list(self.lazy.values())
        for interaction in all_interactions:
            for stream in interaction.stream_proxy:

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

    async def update_sent_status(self, status_dict: dict) -> None:
        """Update a previously sent interaction's status from a received status message (async).

        Args:
            status_dict: Dict from Interaction.to_status_dict().
        """
        sender = status_dict.get('status_generated_by')
        uuid = status_dict.get('uuid')  # UUID can be None (meaning generic UUID, still valid!), recall this
        if uuid in self.sent:

            # Collecting data
            interaction = self.sent[uuid]
            interaction_status = InteractionStatus(status_dict['status'])
            if status_dict.get('completion_reason'):
                completion_reason = CompletionReason(status_dict['completion_reason'])
            else:
                completion_reason = CompletionReason.OK
            interaction_destination_state = status_dict.get('destination_state')

            # Recording data tags
            data_tags: dict[str, list[int]] | None = status_dict.get('data_tags', None)
            if data_tags is not None:
                interaction.record_data_tags(data_tags,
                                             target=sender)

            #  Mark the completion from the sender
            if interaction_status == InteractionStatus.COMPLETED:
                await self.complete(interaction, dest_state=interaction_destination_state, reason=completion_reason,
                                    target=sender)

    def is_received(self, interaction: Interaction) -> bool:
        """Return True if the interaction was received from another agent (active or recently completed).

        Args:
            interaction: The interaction to check.
        """
        return interaction.uuid in self.received or interaction in self.received_recently_completed

    def is_sent(self, interaction: Interaction) -> bool:
        """Return True if the interaction was sent by this agent (active or recently completed).

        Args:
            interaction: The interaction to check.
        """
        return interaction.uuid in self.sent or interaction in self.sent_recently_completed

    def is_lazy(self, interaction: Interaction) -> bool:
        """Return True if the interaction was lazily generated within this agent (active or recently completed).

        Args:
            interaction: The interaction to check.
        """
        return interaction.uuid in self.lazy or interaction in self.lazy_recently_completed

    def is_known(self, interaction: Interaction) -> bool:
        """Return True if the interaction is tracked in any of sent, received, or lazy dicts.

        Args:
            interaction: The interaction to check.
        """
        return self.is_received(interaction) or self.is_sent(interaction) or self.is_lazy(interaction)

    def get_interaction(self, uuid: str | None, consider_completed_too: bool = False) -> Interaction | None:
        """Look up an interaction by UUID across the active tracking dicts.

        Args:
            uuid: The UUID string to look up.
            consider_completed_too: When True, also search the recently-completed sets and return the
                most recently completed match (there may be more than one with the same UUID).

        Returns:
            The matching Interaction, or None if not found.
        """
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

    def add_lazy_stream_to_interaction(self, stream_hash: str, interaction: Interaction) -> None:
        """Attach a stream (identified by hash) to an already-registered interaction as a lazy stream.

        Args:
            stream_hash: A user-level or net-level stream hash to resolve and attach.
            interaction: The interaction to attach the stream to.
        """
        user_hashes = self.__normalize_to_user_hashes(stream_hash)
        log.debug(f"[add_lazy_stream_to_interaction] stream_hash={stream_hash}, user_hashes={user_hashes}, "
                  f"interaction={interaction}")
        for user_hash in user_hashes:
            stream_obj = self.agent.known_streams_by_user_hash[user_hash]
            stream_obj.add_interaction(interaction)
            interaction.add_lazy_stream(user_hash, stream_obj,
                                        is_owned=user_hash in self.agent.owned_streams_by_user_hash)

    async def remove_interactions_of_agent(self, agent: str) -> None:
        """Complete and deregister all interactions that involve the given agent as requester or target (async).

        Args:
            agent: Peer ID of the agent being removed.
        """
        interaction_dicts = [self.sent, self.received, self.lazy]
        for interaction_dict in interaction_dicts:
            for uuid, inter in list(interaction_dict.items()):  # Do not remove list(...)
                if inter.requester == agent:
                    await self.complete(inter, reason=CompletionReason.DISCONNECTED)
                if agent in inter.target:
                    await self.complete(inter, reason=CompletionReason.DISCONNECTED, target=agent)

    def __check_callback_method(self, interaction: Interaction) -> None:
        if interaction.callback is not None:
            callback_method = getattr(self.agent, interaction.callback, None)
            if callable(callback_method):
                sig = inspect.signature(callback_method)
                all_have_defaults_with_the_exception_of_interaction_if_needed = all(
                    p.default is not inspect.Parameter.empty
                    for name, p in sig.parameters.items()
                    if name not in Custom.INTERACTION_ARG_NAMES
                )
                if not all_have_defaults_with_the_exception_of_interaction_if_needed:
                    log.critical(f"Invalid callback method {interaction.callback}: it must have default values "
                                 f"for all parameters (the interaction argument might not have a default, if needed)")
                found_args = [name for name in sig.parameters if name in Custom.INTERACTION_ARG_NAMES]
                if len(found_args) != 1:
                    log.critical(f"Invalid callback method {interaction.callback}: it must have an argument "
                                 f"which is the interaction object (a single one)")
            else:
                log.critical(f"Invalid callback method {interaction.callback}: it must be a method of the agent class")

    def __normalize_to_user_hashes(self, stream_hash: str) -> list[str]:
        """Expand a stream hash (user-level or net-level) into a list of user-level hashes.

        Args:
            stream_hash: A user-level hash (returns it directly) or a net-level hash (expanded to all
                matching user-level hashes known to this agent).

        Returns:
            A list of user-level hash strings.
        """
        user_hashes = []
        if Stream.is_user_hash(stream_hash):
            user_hashes.append(stream_hash)
        elif Stream.is_net_hash(stream_hash):
            net_hash = stream_hash
            if net_hash in self.agent.known_streams:
                streams = self.agent.known_streams[net_hash]
                for stream_obj in streams.values():
                    name = stream_obj.props.get_name()
                    user_hash = DataProps.user_hash_from_net_hash(net_hash, name)
                    user_hashes.append(user_hash)
        return user_hashes

    def __str__(self) -> str:
        """Return a multi-line human-readable summary of all tracked interactions."""
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
    PAUSED = "paused"
    COMPLETED = "completed"


class CompletionReason(Enum):
    OK = "ok"  # Correctly completed: triggered a transition
    TIMEOUT = "timeout"  # The interaction was waiting in the queue for too long, it's time to remove it
    REJECTED = "rejected"  # The interaction was not accepted since the very beginning
    DISCONNECTED = "disconnected"
    ERROR = "error"  # An error occurred
    DISCARDED = "discarded"  # Discarded by the agent (on purpose)


class InteractionType(Enum):
    SENT = "sent"
    RECEIVED = "received"
    LAZY = "lazy"
