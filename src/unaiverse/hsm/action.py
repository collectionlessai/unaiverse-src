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
import json
import copy
import html
import time
import inspect
from typing import Any
from unaiverse.custom import Custom
from collections.abc import Iterator
from unaiverse.utils.logger import log
from unaiverse.interaction import Interaction, CompletionReason


class Action:

    def __init__(self, name: str, args: dict, actionable: object,
                 idx: int = -1,
                 ready: bool = True,
                 msg: str | None = None,
                 avoid_changing_ready: bool = False,
                 teleport: bool = False,
                 high_priority: bool = False,
                 max_duration: float | str = 0.,
                 retry_timeout: float | str = 0,
                 delay: float | str = 0):
        """Initializes an `Action` object, which encapsulates a method to be executed on a given object (`actionable`)
        with specified arguments. It sets up various properties for managing multistep actions, including
        `total_steps`, `total_time`, and `retry_timeout`. It also handles wildcard argument replacement and checks for the
        existence of required parameters. It identifies if the action is a 'not ready' type (e.g., `do_`, `get_`) and
        sets its initial status accordingly.

        Args:
            name: The name of the method to call.
            args: A dictionary of arguments for the method.
            actionable: The object on which the method will be executed.
            idx: A unique ID for the Custom.
            ready: A boolean indicating if the action is ready to be executed.
            msg: An optional human-readable message.
            avoid_changing_ready: A boolean indicating that the selected ready state should not be changed by
                internal rules.
            teleport: A boolean indicating that this action must be hidden when drawing the state machine ('teleport').
            high_priority: A boolean indicating that this action must be preferred over others by the policy.
            max_duration: The number of seconds representing the max duration of this action (from when it starts).
                When <= 0., then no limits. It can be a wildcard (str).
            retry_timeout: The number of seconds representing the max time we keep try running this action before
                giving up. When <= 0., then no timeouts at all. It can be a wildcard (str).
            delay: The number of seconds that must pass when joining a state before considering this action.
                When <= 0., then no delays at all. It can be a wildcard (str).
        """
        # Basic properties
        self.name = name  # Name of the action (name of the corresponding method)
        self.args = args.copy()  # Dictionary of arguments to pass to the action (shallow copy, we will remove some)
        self.actionable = actionable  # Object on which the method whose name is self.name is searched
        self.interactions = ActionInteractionList()  # List of interactions to make this action ready to be executed
        self.id = idx  # Unique ID of the action (-1 if not needed)
        self.msg = msg  # Human-readable message associated to this instance of action
        self.inner = ready
        self.outer = True
        self.state_machine = None
        self.teleport = teleport
        self.high_priority = high_priority
        self.__mark = None

        # Fix UNICODE chars
        if self.msg is not None:
            self.msg = html.unescape(self.msg)

        # Reference elements
        self.__fcn = self.__action_name_to_callable(name)  # The real method to be called
        self.__sig = inspect.signature(self.__fcn)  # Signature of the method for argument inspection

        # Parameter names and default values
        self.param_list = []  # Full list of the parameters that the action supports
        self.param_to_default_value = {}  # From parameter to its default value, if any
        self.__get_action_params()  # This will fill the two attributes above

        # Time-based metrics
        self.__action_max_duration = max_duration  # A total time <= 0 means "no total time at all"
        self.__action_max_duration_with_wildcard = max_duration if isinstance(max_duration, str) else None
        self.__guess_total_time(self.args)  # This will "guess" the value of self.__action_max_duration from args dict

        # Time-based metrics
        self.__action_retry_timeout = retry_timeout  # A retry_timeout <= 0 means "no total time at all"
        self.__action_retry_timeout_with_wildcard = retry_timeout if isinstance(retry_timeout, str) else None
        self.__guess_timeout(self.args)  # This will "guess" the value of self.__action_retry_timeout from the args dict

        # Time-based metrics
        self.__delay = delay
        self.__delay_with_wildcard = delay if isinstance(delay, str) else None
        self.__guess_delay(self.args)  # This will "guess" the value of self.__delay from the args dict

        # Checking arguments (also removing 'retry_timeout', 'delay', etc...)
        self.check_provided_args(self.args, exception=True, remove_special_arguments=True)

        # Argument values replaced by wildcards (commonly assumed to be in the format <value>)
        self.wildcards = {}  # Value-to-value (es: <playlist> to this:and:this)
        self.args_with_wildcards = copy.deepcopy(self.args)  # Backup of the originally provided arguments
        self.msg_with_wildcards = self.msg

        # Fixing (forcing NOT-ready on some actions)
        if not avoid_changing_ready:
            for p in self.param_list:
                if p in Custom.INTERACTION_INJECT_NAMES:
                    self.inner = False
                    self.outer = True
                    break

        # Whether for this Action we've already register_lazy a system_interaction,
        # and whether `copy_sys` was requested for the current round.
        self._lazy_registered = False
        self._must_lazy_register = False
        self._copy_sys_on_register = False

        # Build the initial system_interaction (lightweight dummy or interaction-shaped). MUST stay
        # at the very end of __init__.
        self.__build_system_interaction()

    async def __on_lazy_done(self) -> None:
        """Complete the current lazy system_interaction and rebuild a fresh one so the next round
        starts with a brand-new uuid. No-op for dummy (non-registered) system_interactions; expired
        lazy interactions are also caught by `InteractionManager.complete_expired()` every cycle."""
        if not self._lazy_registered:
            return
        await self.actionable.im.complete(self.system_interaction, CompletionReason.OK)
        self.__build_system_interaction()

    @property
    def ready(self) -> bool:
        """Returns the inner readiness flag of the Custom."""
        return self.inner

    @property
    def is_high_priority(self) -> bool:
        """Returns whether the action is a high priority one."""
        return self.high_priority

    def set_high_priority(self) -> None:
        self.high_priority = True

    def set_state_machine(self, hsm: object) -> None:
        """Registers the parent state machine that owns this action."""
        self.state_machine = hsm
        self.set_wildcards(hsm.get_wildcards())

    def get_name(self) -> str:
        return self.name

    def set_mark(self, mark: object) -> None:
        """Store an arbitrary marker object on the action."""
        self.__mark = mark

    def get_mark(self) -> object:
        """Return the marker object previously set by set_mark."""
        return self.__mark

    def clear_mark(self) -> None:
        """Clear the arbitrary marker object on the action."""
        self.__mark = None

    async def __call__(self, interaction: Interaction | None = None) -> int:
        """Executes the action's associated method. This is the main entry point for running an Action. It handles
        multistep logic by updating the step counter and checking for completion based on steps, time, or retry_timeout.
        It also injects dynamic arguments like the `requester`, `request_time`, and `request_uuid` into the method's
        arguments before execution. If the action is a multistep action and has a completion step, it handles that
        callback as well (async).

        Args:
            interaction: The Interaction object or None, if the action was not requested by other agents.

        Returns:
            An integer status code: ``0`` for success, ``1`` to retry, ``2`` to skip/move to next, ``-1`` for
            unexpected cases.
        """
        if interaction is None:
            interaction = self.system_interaction

        if self._must_lazy_register and not self._lazy_registered:
            # If `system_interaction` is interaction-shaped (the HSM transit passed `streams` /
            # `num_steps` / etc.), register it lazily on first dispatch. The action body may or may
            # not read from `interaction` — the framework's use of the metadata (readiness gate,
            # mult-step counter, recipient resolution) is independent.
            # It is the same things that happen in "send".
            public = not self.actionable.behaving_in_world()
            if self.actionable.im.register_lazy(self.system_interaction, public=public):
                self._lazy_registered = True
                if self._copy_sys_on_register:
                    for stream in self.system_interaction.owned_streams.values():
                        last_data = stream.get(uuid=Custom.SYSTEM_INTERACTION_UUID)
                        if last_data is not None:
                            stream.set(last_data, uuid=self.system_interaction.uuid)
            else:
                log.error(f"Calling action {self.name}: register_lazy failed, falling back to dummy")

        interaction_args = self.__apply_wildcards_to_args(interaction.action_kwargs, in_place=False)
        actual_args = self.get_actual_params(additional_args=interaction_args)

        if self.msg is not None and interaction.get_step_idx() < 0:
            if self.state_machine.show_action_request_info and interaction is not None:
                msg = self.msg + (f" (requester: {interaction.requester}, interaction uuid: {interaction.uuid}, "
                                  f"#interactions: {len(self.get_list_of_interactions())})")
            else:
                msg = self.msg
            log.user(msg)

        # Storing the starting time of this action (only at the very first run attempt)
        if interaction.get_starting_time() <= 0:
            interaction.set_starting_time(time.perf_counter())

        if interaction.is_timed_out():
            interaction.clear_mark()
            if interaction.is_single_step():
                return 2
            if interaction.is_multi_steps():
                if interaction.was_at_least_one_step_done():
                    return 0
                else:
                    return 2

        if interaction.is_multi_steps() and interaction.was_at_least_one_step_done():
            if interaction.get_new_stream_data_tags(all_fresh_or_fail=True) is None:
                return 1

        for p in self.param_list:
            if p in Custom.INTERACTION_INJECT_NAMES:
                actual_args[p] = interaction

        # If the action continues (multistep), increasing the step index
        # This is a step index, so interaction.__step == 0 means "doing/done 1 step"
        # We start with  interaction.__step = -1, that here will become 0 - it will be run in the following code
        interaction.inc_step_idx()

        # Calling the method here
        ret = await self.__fcn(**actual_args)

        # If action failed, be sure to reduce the step counter (only if it was actually incremented)
        if not ret:
            interaction.dec_step_idx()

            if interaction.is_single_step():
                if self.is_pedantic():
                    return 1
                else:
                    interaction.clear_mark()
                    return 2
            if interaction.is_multi_steps():
                if interaction.was_at_least_one_step_done():
                    if self.is_pedantic():
                        return 1
                    interaction.clear_mark()
                    await self.__on_lazy_done()
                    return 0
                else:
                    interaction.clear_mark()
                    return 2

        # If it went OK, we reset the time counter that is related to the retry_timeout
        else:
            interaction.set_timeout_starting_time(time.perf_counter())

            if interaction.is_single_step():
                interaction.clear_mark()
                await self.__on_lazy_done()
                return 0
            if interaction.is_multi_steps():
                if interaction.was_last_step_done():
                    interaction.clear_mark()
                    await self.__on_lazy_done()
                    return 0
                else:
                    return 1

        # Unexpected
        return -1

    def __str__(self) -> str:
        """Provides a string representation of the `Action` instance.

        Returns:
            A string containing a formatted summary of the instance.
        """
        return (f"[Action: {self.name}] id: {self.id}, args: {self.args}, param_list: {self.param_list}, "
                f"{Custom.SECONDS_ARG_NAMES[0]}: {self.__action_max_duration}, "
                f"{Custom.TIMEOUT_ARG_NAMES[0]}: {self.__action_retry_timeout}, "
                f"{Custom.DELAY_ARG_NAMES[0]}: {self.__delay}, "
                f"ready (inner): {self.inner}, outer: {self.outer}, interactions: {str(self.interactions)}, "
                f"msg: {str(self.msg)}]")

    def to_code_str(self) -> str:
        """Returns a compact code-style string representation of the action (for logging/debugging)."""
        s = f"aai:{self.name}|{self.args}|"
        if len(self.interactions) > 0:
            return s + "\n" + "\n".join(f"   {inter.to_code_str(True)}" for inter in self.interactions)
        else:
            return s + "no-int"

    def set_as_ready(self) -> None:
        """Sets the action's ready flag to `True`, indicating it can now be executed."""
        self.outer = False  # For the moment, I prefer to avoid "solid" actions to also take requests from the outside
        self.inner = True

    def set_as_not_ready(self) -> None:
        """Sets the action's ready flag to `False`, preventing it from being executed
            (unless requested from the outside)."""
        self.inner = False
        self.outer = True

    def set_msg(self, msg: str | None) -> None:
        """Sets the message associated to this Custom."""

        if msg is not None:
            self.msg = html.unescape(msg)
            self.msg_with_wildcards = self.msg
        else:
            self.msg = None
            self.msg_with_wildcards = None

    def is_ready(self, consider_interactions: bool = True, delay_starting_time: float = -1.) -> bool:
        """Checks if the action is ready to be executed. It returns `True` if the `ready` flag is set or if there are
        any pending interactions.

        Args:
            consider_interactions: A boolean flag to include pending interactions in the readiness check.
            delay_starting_time: The starting time from which the delay is calculated (in seconds).

        Returns:
            A boolean indicating the action's readiness.
        """
        is_delayed = (delay_starting_time > 0 and self.__delay > 0 and
                      (time.perf_counter() - delay_starting_time) <= self.__delay)
        return not is_delayed and (self.inner or (consider_interactions and self.outer and len(
            self.interactions.get_interactions(doable_only=True)) > 0))

    def is_teleport(self):
        """Returns whether this action is a 'teleport' action, hidden from the shown state machine."""
        return self.teleport

    def allows_outer_interactions(self) -> bool:
        """Returns whether this action can be triggered by external (outer) interactions."""
        return self.outer

    def allows_inner_interactions(self) -> bool:
        """Returns whether this action can be triggered internally (inner readiness flag)."""
        return self.inner

    def set_timeout(self, retry_timeout: float) -> None:
        """Sets the retry_timeout to a custom value."""
        self.__action_retry_timeout = retry_timeout

    def set_default_timeout(self) -> None:
        """Sets the retry_timeout to the class-level default value (``Custom.DEFAULT_TIMEOUT``)."""
        self.__action_retry_timeout = Custom.DEFAULT_TIMEOUT

    def is_pedantic(self) -> bool:
        """Checks if a retry_timeout has been configured for the Custom.

        Returns:
            A boolean indicating if a retry_timeout is set.
        """
        return self.__action_retry_timeout > 0

    def get_total_time(self) -> float | str:
        """Returns the total execution time configured for this action (0 means no limit)."""
        return self.__action_max_duration

    def get_timeout(self) -> float | str:
        """Returns the retry_timeout configured for this action (0 means no retry_timeout)."""
        return self.__action_retry_timeout

    def get_delay(self) -> float | str:
        """Returns the delay configured for this action (0 means no delay)."""
        return self.__delay

    def to_list(self) -> list:
        """Converts the action's properties into a list for easy serialization and comparisons.

        Returns:
            A list containing the action's properties.
        """
        total_time = 0.
        retry_timeout = 0.
        delay = 0.
        if isinstance(self.__action_max_duration, str) or self.__action_max_duration > 0:
            total_time = self.__action_max_duration
        if isinstance(self.__action_retry_timeout, str) or self.__action_retry_timeout > 0.:
            retry_timeout = self.__action_retry_timeout
        if isinstance(self.__delay, str) or self.__delay > 0.:
            delay = self.__delay
        return [self.name, self.args, self.inner, self.outer, total_time, retry_timeout, delay, self.msg]

    def to_dict(self) -> dict:
        """Converts the action's properties into a dict for easy serialization.

        Returns:
            A dict containing the action's properties.
        """
        return {
            "action": self.name,
            "action_kwargs": self.args,
            "msg": self.msg.encode("ascii",
                                   "xmlcharrefreplace").decode("ascii") if self.msg is not None else None,
            "ready": self.inner,
            "high_priority": self.is_high_priority,
            Custom.SECONDS_ARG_NAMES[0]: self.__action_max_duration,
            Custom.TIMEOUT_ARG_NAMES[0]: self.__action_retry_timeout,
            Custom.DELAY_ARG_NAMES[0]: self.__delay
        }

    def same_as(self, name: str, args: dict | None) -> bool:
        """Compares the current action to a target action by name and arguments. It returns `True` if they are
        considered the same, ignoring specific arguments like time or retry_timeout.

        Args:
            name: The name of the target Custom.
            args: The arguments of the target Custom.

        Returns:
            A boolean indicating if the actions are a match.
        """
        if args is None:
            args = {}

        # The current action is the same of another action called with some arguments "args" if:
        # 1) it has the same name of the other action
        # 2) the name of the arguments in "args" are known and valid
        # 3) the values of the arguments in "args" matches the ones of the current action, being them default or not
        # the values of those arguments that are not in "args" are assumed to the equivalent to the ones in the current
        # action, so:
        # - if the current action is act(a=3, b=4), then it is the same_as(name='act', args={'a': 3})
        # - if the current action is act(a=3, b=4), then it is the same_as(name='act', args={'a': 3, 'b': 4, 'c': 5})
        args_to_exclude = Custom.RESERVED_IN_ACTION_KWARGS
        return (name == self.name and
                self.check_provided_args(args) and
                all(k in args_to_exclude or k not in self.args or self.args[k] == v for k, v in args.items()))

    def check_provided_args(self, args: dict, exception: bool = False, remove_special_arguments: bool = False) -> bool:
        """A helper method that validates that all provided arguments for an action exist in the action's
        parameter list. It can either raise a `ValueError` or return a boolean.

        Args:
            args: The dictionary of arguments to check.
            exception: If `True`, a `ValueError` is raised on failure.
            remove_special_arguments: If `True`, removes from the args dictionary those special arguments that are
                internally handled in a specific manner (such as time-related arguments).

        Returns:
            True if all arguments are valid, False otherwise (if `exception` is `False`).
        """
        if args is not None:
            for k in list(args.keys()):
                if k in Custom.HSM_TRANSIT_META_NAMES:
                    if remove_special_arguments:
                        del args[k]
                    elif exception:
                        raise ValueError(f"Parameter {k} is not a valid param for action"
                                         f" {self.name} (it is a private HSM transit meta name)")
                    else:
                        return False
                elif k in Custom.WIRE_SENTINEL_NAMES or k in Custom.INTERACTION_INJECT_NAMES:
                    if exception:
                        raise ValueError(f"Parameter {k} is not a valid param for action"
                                         f" {self.name} (it is reserved)")
                    else:
                        return False
                elif k in Custom.INTERACTION_FIELD_NAMES:
                    continue  # Skipping
                elif k not in self.param_list:
                    if exception:
                        raise ValueError(f"Parameter {k} is not a valid param for action"
                                         f" {self.name} (it is not part of the action signature)")
                    else:
                        return False
        return True

    def set_wildcards(self, wildcards: dict[str, str | float | int] | None) -> None:
        """Replaces wildcard values with new ones.

        Args:
            wildcards: A dictionary mapping wildcard placeholders to their concrete values.
        """
        self.wildcards = wildcards if wildcards is not None else {}

    def add_interaction(self, interaction: Interaction) -> bool:
        """Adds a new interaction to the action's internal list.
        This is used to track pending requests that might make the action ready to be executed.

        Args:
            interaction: Interaction object.
        """

        # Let's augment interaction object
        if not self.allows_outer_interactions():
            return False
        interaction.set_action_ref(self)
        self.interactions.add(interaction)
        return True

    def clear_interactions(self, requester: str | None = None) -> None:
        """Clears all pending requests from the action's list."""
        if requester is None:
            self.interactions = ActionInteractionList()
        else:
            if self.interactions.is_requester_known(requester):
                requests = self.interactions.get_interactions(requester)
                for req in requests:
                    self.interactions.remove(req)

    def clear_interaction(self, requester: str, req_id: int) -> None:
        """Removes a single specific interaction identified by requester and insertion-order ID.

        Args:
            requester: The peer ID of the requester who owns the interaction.
            req_id: The insertion-order ID of the interaction to remove.
        """
        req = self.interactions.get_interaction(req_id, requester)
        if req is not None:
            self.interactions.remove(req)

    def get_list_of_interactions(self) -> 'ActionInteractionList':
        """Retrieves the list of pending interactions (ActionInteractionList object).

        Returns:
            The list of pending interactions, i.e., an object of type ActionInteractionList.
        """
        return self.interactions

    def get_actual_params(self, additional_args: dict | None) -> dict | None:
        """A helper method that resolves all parameters for an action's execution. It combines the action's
        default arguments, initial arguments, and any additional arguments provided during the call, ensuring all
        necessary parameters have a value.

        Args:
            additional_args: A dictionary of arguments to be combined with the action's defaults.

        Returns:
            A dictionary of all resolved arguments, or `None` if a required parameter is missing.
        """
        actual_params = {}
        params = self.param_list
        defaults = self.param_to_default_value
        for param_name in params:
            if param_name in self.args:
                actual_params[param_name] = self.args[param_name]
            elif additional_args is not None and param_name in additional_args:
                actual_params[param_name] = additional_args[param_name]
            elif param_name in defaults:
                actual_params[param_name] = defaults[param_name]
            else:
                log.error(f"Getting actual params for {self.name}; missing param: {param_name}")
                return None
        return actual_params

    def __build_system_interaction(self) -> None:
        """Build (or rebuild) self.system_interaction reflecting current self.args.

        If self.args contains any interaction-shaped keys (`Custom.INTERACTION_FIELD_NAMES`), the new
        interaction is populated with those fields and will be register_lazy-ed on first dispatch
        (see `__call__`). The action body may or may not read from `interaction` — that's
        independent; the framework uses the interaction's metadata regardless."""
        interaction_kwargs = {k: self.args[k] for k in Custom.INTERACTION_FIELD_NAMES
                              if k in self.args and k not in self.param_list}
        self._copy_sys_on_register = bool(interaction_kwargs.pop("copy_sys", False))

        action_kwargs = {k: v for k, v in self.args.items() if k not in interaction_kwargs}

        if "id" not in interaction_kwargs and "forced_uuid" not in interaction_kwargs:
            interaction_kwargs["id"] = Custom.SYSTEM_INTERACTION_ID
        if "target" not in interaction_kwargs:
            interaction_kwargs["target"] = Custom.SYSTEM_INTERACTION_LABEL

        self.system_interaction = Interaction(
            requester=Custom.SYSTEM_INTERACTION_LABEL,
            action_name=self.name,
            action_kwargs=action_kwargs,
            **interaction_kwargs,
        )
        self.system_interaction.set_action_ref(self)
        self._lazy_registered = False
        self._must_lazy_register = (len(interaction_kwargs) > 0 and
                                    (self.system_interaction.streams or self.system_interaction.num_steps > 0))

    def __action_name_to_callable(self, action_name: str) -> Any | None:
        """A private helper method that resolves a string action name into a callable method on the `actionable`
        object. It raises a `ValueError` if the method is not found.

        Args:
            action_name: The name of the method to retrieve.

        Returns:
            A callable function or method.
        """
        if self.actionable is not None:
            action_fcn = getattr(self.actionable, action_name)
            if action_fcn is None:
                raise ValueError("Cannot find function/method: " + str(action_name))
            return action_fcn
        else:
            return None

    def __get_action_params(self) -> None:
        """A private helper method that inspects the signature of the action's method to populate the list of
        supported parameters and their default values.
        """
        self.param_list = [param_name for param_name in self.__sig.parameters.keys()]
        self.param_to_default_value = {param.name: param.default for param in self.__sig.parameters.values() if
                                       param.default is not inspect.Parameter.empty}

        # Ensuring those *specially handled* arguments are not present in the method signature (they can only be
        # present in the kwargs used to call the function)
        # Exceptions:
        # - Some actions are special, and some arguments must be allowed to them (e.g., send(action_kwargs)).
        # - The INTERACTION_INJECT_NAMES must be allowed here, since it is natural for an action to have the
        # interaction argument somewhere.
        error_msg = ""
        for p in self.param_list:
            if p in Custom.HSM_TRANSIT_META_NAMES:
                error_msg = f"{p} is HSM-transit metadata, cannot be a body param"
            elif p in Custom.WIRE_SENTINEL_NAMES and self.name not in Custom.SPECIAL_ACTION_NAMES:
                error_msg = f"{p} is a framework-private name, only {list(Custom.SPECIAL_ACTION_NAMES)} may use it"
        if len(error_msg) > 0:
            log.critical(f"Action {self.name} includes and argument that is problematic: {error_msg}")

    def __apply_wildcards_to_args(self, args: dict | None = None, in_place: bool = True):
        """Replace placeholder values (wildcards) in action arguments with their concrete values.
        Handles single values, lists, and nested dicts (so an arg like
        `{"streams": {"stdin": ["<world>:all"]}}` is fully expanded)."""
        if not in_place:
            args = copy.deepcopy(args)

        def _replace(value):
            for wildcard_from, wildcard_to in self.wildcards.items():
                if isinstance(wildcard_to, str):
                    if isinstance(value, str) and wildcard_from in value:
                        value = value.replace(wildcard_from, wildcard_to)
                else:
                    if value == wildcard_from:
                        return wildcard_to
            return value

        def _walk(node):
            if isinstance(node, dict):
                for k in list(node.keys()):
                    node[k] = _walk(node[k])
                return node
            if isinstance(node, list):
                for i in range(len(node)):
                    node[i] = _walk(node[i])
                return node
            return _replace(node)

        _walk(args)
        return args

    def apply_wildcards(self) -> None:
        """Given the current wildcards, it applies the replacements they suggest to whatever uses them."""

        # Setting up the original wildcard-based arguments and messages
        if self.args_with_wildcards is None:
            self.args_with_wildcards = copy.deepcopy(self.args)  # Backup before applying wildcards (1st time only)
        else:
            self.args = copy.deepcopy(self.args_with_wildcards)  # Restore a backup before applying wildcards
        if self.msg_with_wildcards is not None:
            self.msg = self.msg_with_wildcards

        # Applying wildcard-suggested replacements to arguments
        self.__apply_wildcards_to_args(self.args)

        # Applying wildcard-suggested replacements to message and other stuff
        for wildcard_from, wildcard_to in self.wildcards.items():

            # ... to message
            if self.msg is not None:
                self.msg = self.msg.replace(wildcard_from, str(wildcard_to))

            # ... to the rest
            if (self.__action_max_duration_with_wildcard is not None and
                    wildcard_from in self.__action_max_duration_with_wildcard):
                self.__action_max_duration = (
                    float(self.__action_max_duration_with_wildcard.replace(wildcard_from, str(wildcard_to))))
            if (self.__action_retry_timeout_with_wildcard is not None and
                    wildcard_from in self.__action_retry_timeout_with_wildcard):
                self.__action_retry_timeout = (
                    float(self.__action_retry_timeout_with_wildcard.replace(wildcard_from, str(wildcard_to))))
            if self.__delay_with_wildcard is not None and wildcard_from in self.__delay_with_wildcard:
                self.__delay = float(self.__delay_with_wildcard.replace(wildcard_from, str(wildcard_to)))

        # Now that wildcards have been resolved in self.args, rebuild the system_interaction so any
        # streams / num_steps / etc. reflect the concrete (post-wildcard) values.
        self.__build_system_interaction()

    def __guess_total_time(self, args) -> None:
        """A private helper method that attempts to determine the total execution time for an action by looking for a
        'time' or 'seconds' argument.

        Args:
            args: The dictionary of arguments to inspect.
        """
        for arg_name in Custom.SECONDS_ARG_NAMES:
            if arg_name in args:
                try:
                    self.__action_max_duration = max(float(args[arg_name]), 0.)
                    self.__action_max_duration_with_wildcard = None
                except ValueError:
                    if isinstance(args[arg_name], str):
                        self.__action_max_duration = str(args[arg_name])
                        self.__action_max_duration_with_wildcard = str(args[arg_name])
                    else:
                        self.__action_max_duration = -1.
                    pass
                break

    def __guess_timeout(self, args) -> None:
        """A private helper method that attempts to determine the retry_timeout duration for an action by looking for a
        'retry_timeout' argument.

        Args:
            args: The dictionary of arguments to inspect.
        """
        for arg_name in Custom.TIMEOUT_ARG_NAMES:
            if arg_name in args:
                try:
                    self.__action_retry_timeout = max(float(args[arg_name]), 0.)
                    self.__action_retry_timeout_with_wildcard = None
                except ValueError:
                    if isinstance(args[arg_name], str):
                        self.__action_retry_timeout = str(args[arg_name])
                        self.__action_retry_timeout_with_wildcard = str(args[arg_name])
                    else:
                        self.__action_retry_timeout = -1.
                break

    def __guess_delay(self, args) -> None:
        """A private helper method that attempts to determine a delay duration for an action by looking for a 'delay'
        argument.

        Args:
            args: The dictionary of arguments to inspect.
        """
        for arg_name in Custom.DELAY_ARG_NAMES:
            if arg_name in args:
                try:
                    self.__delay = max(float(args[arg_name]), 0.)
                    self.__delay_with_wildcard = None
                except ValueError:
                    if isinstance(args[arg_name], str):
                        self.__delay = str(args[arg_name])
                        self.__delay_with_wildcard = str(args[arg_name])
                    else:
                        self.__delay = -1.
                    pass
                break


class ActionInteractionList:
    def __init__(self, max_per_requester: int = -1):
        """Initializes an empty interaction list with an optional per-requester cap.

        Args:
            max_per_requester: Maximum number of interactions stored per requester (-1 for unlimited).
        """
        self.by_insertion_order = []
        self.by_requester_and_by_insertion_order = {}
        self.max_per_requester = max_per_requester
        self.by_insertion_order_entering_time = []

    def add(self, interaction: Interaction) -> None:
        """Appends an interaction to the list, updating all internal indices. Interactions with ``uuid=None`` are
        deduplicated per requester. If ``max_per_requester`` is set, the oldest entry for that requester is evicted.

        Args:
            interaction: The Interaction object to add.
        """

        # Searching for already existing interactions with this UUID, FROM THE SAME REQUESTER
        # If already there - do not accumulate multiple requests with same UUID (useful also for system interactions)
        existing_request_same_uuid = self.get_interaction_by_uuid(interaction.requester, interaction.uuid)
        if existing_request_same_uuid:

            # We skip this request if we are already taking care (running) of another one with the same UUID
            if existing_request_same_uuid.running:  # This means the HSM is taking care of it, don't mess up things!
                log.error(f"Tried to add an interaction with the UUID of an already existing one that was in "
                          f"'running' state, skipping (requester: {interaction.requester}")
                return

            # Since we know the requester is the same, we just "refresh" the existing interaction object,
            # without altering its position in the list
            self.by_insertion_order[existing_request_same_uuid.by_insertion_order_id] = interaction
            interaction.by_insertion_order_id = existing_request_same_uuid.by_insertion_order_id

            self.by_requester_and_by_insertion_order[
                interaction.requester][existing_request_same_uuid.by_requester_insertion_order_id] = (
                interaction)
            interaction.by_requester_insertion_order_id = existing_request_same_uuid.by_requester_insertion_order_id
            return  # We stop here in this case, no need to do any other things

        # Updating by-requester index
        if interaction.requester not in self.by_requester_and_by_insertion_order:
            self.by_requester_and_by_insertion_order[interaction.requester] = []

        if 0 < self.max_per_requester <= len(self.by_requester_and_by_insertion_order[interaction.requester]):
            self.remove(self.get_oldest_interaction(interaction.requester))
        by_requester_insertion_order_id = len(self.by_requester_and_by_insertion_order[interaction.requester])
        self.by_requester_and_by_insertion_order[interaction.requester].append(interaction)

        # Updating direct global index
        insertion_order_id = len(self.by_insertion_order)
        self.by_insertion_order.append(interaction)

        # Updating reverse indices
        interaction.by_insertion_order_id = insertion_order_id
        interaction.by_requester_insertion_order_id = by_requester_insertion_order_id

        # Saving joining time
        self.by_insertion_order_entering_time.append(time.perf_counter())

    def remove(self, interaction: Interaction) -> None:
        """Removes an interaction from the list, reindexing all subsequent entries to keep indices consistent.

        Args:
            interaction: The Interaction object to remove.
        """
        if interaction.is_valid():
            if (interaction.by_insertion_order_id < len(self.by_insertion_order) and
                    self.by_insertion_order[interaction.by_insertion_order_id] == interaction):

                for i in range(interaction.by_insertion_order_id + 1, len(self.by_insertion_order)):
                    self.by_insertion_order[i].by_insertion_order_id -= 1
                del self.by_insertion_order[interaction.by_insertion_order_id]
                del self.by_insertion_order_entering_time[interaction.by_insertion_order_id]

                d = self.by_requester_and_by_insertion_order[interaction.requester]
                for i in range(interaction.by_requester_insertion_order_id + 1, len(d)):
                    d[i].by_requester_insertion_order_id -= 1
                del d[interaction.by_requester_insertion_order_id]
                if len(d) == 0:
                    del self.by_requester_and_by_insertion_order[interaction.requester]
                interaction.by_insertion_order_id = -1
                interaction.by_requester_insertion_order_id = -1

    def remove_due_to_timeout(self, timeout_secs: float) -> None:
        """Removes all interactions that have been waiting longer than the specified retry_timeout.

        Args:
            timeout_secs: The maximum age in seconds; older interactions are removed.
        """
        to_remove = []
        for i, req in enumerate(self.by_insertion_order):
            if (time.perf_counter() - self.by_insertion_order_entering_time[i]) >= timeout_secs:
                to_remove.append(req)
        for req in to_remove:
            self.remove(req)

    def remove_completed(self) -> None:
        """Removes all interactions that have been marked as completed."""
        to_remove = []
        for i, req in enumerate(self.by_insertion_order):
            if req.completed:
                to_remove.append(req)
        for req in to_remove:
            self.remove(req)

    def move_interaction_to_back(self, interaction: Interaction) -> None:
        """Moves an interaction to the back of the insertion-order list, preserving its original entering time.

        Args:
            interaction: The Interaction object to reposition.
        """
        if len(self.by_insertion_order) > 1 and interaction.is_valid():
            try:
                entering_time = self.by_insertion_order_entering_time[interaction.by_insertion_order_id]
                self.remove(interaction)
                self.add(interaction)
                self.by_insertion_order_entering_time[interaction.by_insertion_order_id] = entering_time
            except Exception as e:
                raise e

    def move_requester_to_back(self, requester: str) -> None:
        """Moves all interactions belonging to a requester to the back of the list, preserving their entering times.

        Args:
            requester: The peer ID whose interactions should be moved to the back.
        """
        requests = self.get_interactions(requester)
        if requests is not None and len(requests) > 0:
            requests_copy = []
            entering_times = []
            for req in requests:
                if req.is_valid():
                    requests_copy.append(req)
                    entering_times.append(self.by_insertion_order_entering_time[req.by_insertion_order_id])
                    self.remove(req)
            for i, req in enumerate(requests_copy):
                self.add(req)
                self.by_insertion_order_entering_time[req.by_insertion_order_id] = entering_times[i]

    def get_interaction(self, req_order_id: int, requester: str | None = None) -> Interaction | None:
        """Retrieves an interaction by its insertion-order index, optionally scoped to a specific requester.

        Args:
            req_order_id: The insertion-order index of the interaction to retrieve.
            requester: If provided, scopes the lookup to the sub-list of this requester.

        Returns:
            The matching Interaction, or None if not found.
        """
        if req_order_id < 0 and req_order_id != -1:
            return None
        if requester is None:
            return self.by_insertion_order[req_order_id] if req_order_id < len(self.by_insertion_order) else None
        else:
            if requester not in self.by_requester_and_by_insertion_order:
                return None
            return self.by_requester_and_by_insertion_order[requester][req_order_id] \
                if req_order_id < len(self.by_requester_and_by_insertion_order[requester]) else None

    def get_oldest_interaction(self, requester: str | None = None, ignore_completed: bool = False) \
            -> Interaction | None:
        """Returns the oldest (first-added) interaction, optionally scoped to a specific requester.

        Args:
            requester: If provided, scopes the lookup to the sub-list of this requester.
            ignore_completed: If True, completed interactions will be ignored/skipped.

        Returns:
            The oldest Interaction, or None if the list is empty.
        """
        if not ignore_completed:
            return self.get_interaction(0, requester)
        else:
            req = self.get_interaction(0, requester)
            i = 1
            while req.completed:
                if i < len(self):
                    req = self.get_interaction(i, requester)
                    i += 1
                else:
                    return None
            return req

    def get_most_recent_interaction(self, requester: str | None = None, ignore_completed: bool = False) \
            -> Interaction | None:
        """Returns the most recently added interaction, optionally scoped to a specific requester.

        Args:
            requester: If provided, scopes the lookup to the sub-list of this requester.
            ignore_completed: If True, completed interactions will be ignored/skipped.

        Returns:
            The most recent Interaction, or None if the list is empty.
        """
        if not ignore_completed:
            return self.get_interaction(-1, requester)
        else:
            req = self.get_interaction(-1, requester)
            i = len(self) - 2
            while req.completed:
                if i >= 0:
                    req = self.get_interaction(i, requester)
                    i -= 1
                else:
                    return None
            return req

    def get_interaction_by_uuid(self, requester: str, uuid: str | None) -> Interaction | None:
        """Finds the first interaction for a given requester that matches the specified UUID.

        Args:
            requester: The peer ID of the requester.
            uuid: The UUID to search for (None is a valid UUID).

        Returns:
            The matching Interaction, or None if not found.
        """
        requests = self.get_interactions(requester)
        if requests is None or len(requests) == 0:
            return None

        for req in requests:
            if req.uuid == uuid:
                return req

    def keep_only_the_most_recent_interaction(self, ignore_completed: bool = False) -> None:
        """Discards all interactions except the most recently added one, preserving its entering time.

        Args:
            ignore_completed: If True, completed interactions will be ignored/skipped.

        Returns:
            None
        """
        req = self.get_most_recent_interaction(ignore_completed=ignore_completed)
        if req is not None:
            entering_time = self.by_insertion_order_entering_time[req.by_insertion_order_id]
            self.clear()
            self.add(req)
            self.by_insertion_order_entering_time[req.by_insertion_order_id] = entering_time
        else:
            self.clear()

    def get_interactions(self, requester: str | None = None, to_str: bool = False,
                         doable_only: bool = False) -> list[Interaction] | str:
        """Returns interactions, optionally filtered by requester or "do-ability", or as a JSON string.

        Args:
            requester: If provided, returns only interactions from this requester.
            to_str: If True, returns a JSON-encoded string instead of a list.
            doable_only: If True, filters to only interactions that pass ``check_if_doable()``.

        Returns:
            A list of Interaction objects, or a JSON-encoded string if ``to_str`` is True.
        """
        if requester is None:
            reqs = self.by_insertion_order
            if doable_only:
                reqs = [req for req in reqs if req.check_if_doable()]  # This excludes completed too
            if not to_str:
                return reqs
            else:
                return json.dumps([req.to_str() for req in reqs])
        else:
            if requester in self.by_requester_and_by_insertion_order:
                reqs = self.by_requester_and_by_insertion_order[requester]
                if doable_only:
                    reqs = [req for req in reqs if req.check_if_doable()]
                if not to_str:
                    return reqs
                else:
                    return json.dumps([req.to_str() for req in reqs])
            else:
                if not to_str:
                    return []
                else:
                    return json.dumps([])

    def clear(self) -> None:
        """Removes all interactions and resets all internal indices."""
        self.by_insertion_order.clear()
        self.by_requester_and_by_insertion_order.clear()
        self.by_insertion_order_entering_time.clear()

    def is_requester_known(self, requester: str) -> bool:
        """Checks whether any interactions from the given requester are currently stored.

        Args:
            requester: The peer ID to look up.

        Returns:
            True if the requester has at least one interaction in the list, False otherwise.
        """
        return requester in self.by_requester_and_by_insertion_order

    def __len__(self) -> int:
        """Returns the total number of interactions in the list."""
        return len(self.by_insertion_order)

    def __iter__(self) -> Iterator[Interaction]:
        """Iterates over all interactions in insertion order."""
        return iter(self.by_insertion_order)

    def __str__(self) -> str:
        """Provides a string representation of the `ActionInteractionList` instance.

        Returns:
            A string containing a formatted summary of the instance.
        """
        if len(self.by_insertion_order) > 0:
            return "; ".join([str(r.to_code_str(True)) for r in self.by_insertion_order])
        else:
            return "empty"
