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
from unaiverse.interaction import Interaction


class Action:

    def __init__(self, name: str, args: dict, actionable: object,
                 idx: int = -1,
                 ready: bool = True,
                 msg: str | None = None,
                 avoid_changing_ready: bool = False,
                 teleport: bool = False,
                 total_time: float | str = 0.,
                 timeout: float | str = 0,
                 delay: float | str = 0):
        """Initializes an `Action` object, which encapsulates a method to be executed on a given object (`actionable`)
        with specified arguments. It sets up various properties for managing multistep actions, including
        `total_steps`, `total_time`, and `timeout`. It also handles wildcard argument replacement and checks for the
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
            total_time: The number of seconds representing the max duration of this action (from the moment it stars).
                When <= 0., then no limits. It can be a wildcard (str).
            timeout: The number of seconds representing the max time we keep try running this action before giving up.
                When <= 0., then no timeouts at all. It can be a wildcard (str).
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
        self.__total_time = total_time  # A total time <= 0 means "no total time at all"
        self.__total_time_with_wildcard = total_time if isinstance(total_time, str) else None
        self.__guess_total_time(self.args)  # This will "guess" the value of self.__total_time from the args dict

        # Time-based metrics
        self.__timeout = timeout  # A timeout <= 0 means "no total time at all"
        self.__timeout_with_wildcard = timeout if isinstance(timeout, str) else None
        self.__guess_timeout(self.args)  # This will "guess" the value of self.__timeout from the args dict

        # Time-based metrics
        self.__delay = delay
        self.__delay_with_wildcard = delay if isinstance(delay, str) else None
        self.__guess_delay(self.args)  # This will "guess" the value of self.__delay from the args dict

        # Checking arguments (also removing 'timeout', 'delay', etc...)
        self.check_provided_args(self.args, exception=True, remove_special_arguments=True)

        # Argument values replaced by wildcards (commonly assumed to be in the format <value>)
        self.wildcards = {}  # Value-to-value (es: <playlist> to this:and:this)
        self.args_with_wildcards = copy.deepcopy(self.args)  # Backup of the originally provided arguments
        self.msg_with_wildcards = self.msg

        # Checking
        self.deprecated = (any([self.name.startswith(p) for p in Custom.SPECIAL_DEPRECATED_CASES_PREFIXES]) and
                           not any([p in Custom.INTERACTION_ARG_NAMES for p in self.param_list]))
        self.deprecated_has_completion = Custom.DEPRECATED_COMPLETED_ARG in self.param_list

        # Fixing (forcing NOT-ready on some actions)
        if not avoid_changing_ready:
            for p in self.param_list:
                if p in Custom.INTERACTION_ARG_NAMES or p == "_requester":
                    self.inner = False
                    self.outer = True
                    break

        # This must be done at the very end of the constructor!
        # This interaction is NOT registered
        self.system_interaction = Interaction(requester="system", target="system",
                                              action_name=self.name, action_kwargs=self.args)
        self.system_interaction.set_action_ref(self)

    @property
    def ready(self) -> bool:
        """Returns the inner readiness flag of the Custom."""
        return self.inner

    @property
    def has_completion_step(self) -> bool:
        """Returns whether the action has a dedicated completion step (deprecated actions only)."""
        return self.deprecated_has_completion

    def set_state_machine(self, hsm: object) -> None:
        """Registers the parent state machine that owns this action."""
        self.state_machine = hsm
        self.set_wildcards(hsm.get_wildcards())

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
        multistep logic by updating the step counter and checking for completion based on steps, time, or timeout.
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

        interaction_args = self.__apply_wildcards_to_args(interaction.action_kwargs, in_place=False)
        actual_args = self.get_actual_params(additional_args=interaction_args)

        if self.msg is not None:
            if self.state_machine.show_action_request_info and interaction is not None:
                msg = self.msg + (f" (requester: {interaction.requester}, interaction uuid: {interaction.uuid}, "
                                  f"#interactions: {len(self.get_list_of_interactions())})")
            else:
                msg = self.msg
            log.user(msg)

        # Storing the starting time of this action (only at the very first run attempt)
        if interaction.get_starting_time() <= 0:
            interaction.set_starting_time(time.perf_counter())

        # Deprecated (old-style) actions
        run_deprecated_completion_step = False
        timed_out = interaction.is_timed_out() or interaction.is_expired()
        if self.deprecated:
            if timed_out:
                if interaction.is_single_step():
                    return 2

                if interaction.is_multi_steps():
                    if not interaction.was_at_least_one_step_done():
                        return 2

            # Check if we are in front of an action that completed all its steps (even just a single step or an
            # action  with no data at all) or that was timed out, but it did at last a step (of course,
            # multistep actions only): in those cases, the action is considered "completed in a correct way"
            run_deprecated_completion_step = (interaction.was_last_step_done() or
                                              (timed_out and interaction.was_at_least_one_step_done()))

            for p in self.param_list:
                if p == '_requester':
                    actual_args[p] = interaction.requester
                elif p == '_request_time':
                    actual_args[p] = interaction.timestamp_created
                elif p == '_request_uuid':
                    actual_args[p] = interaction.uuid
                elif p == '_completed':
                    actual_args[p] = run_deprecated_completion_step
                elif p == 'samples' or p == 'steps':
                    actual_args[p] = max(actual_args[p], interaction.get_total_steps())  # total number of samples
                elif p == 'time':
                    actual_args[p] = self.get_total_time()
                elif p == 'timeout':
                    actual_args[p] = self.get_timeout()
        else:
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
                if p in Custom.INTERACTION_ARG_NAMES or p == "_requester":  # Second part is for backward compatibility
                    actual_args[p] = interaction

            # If the action continues (multistep), increasing the step index
            # This is a step index, so interaction.__step == 0 means "doing/done 1 step"
            # We start with  interaction.__step = -1, that here will become 0 - it will be run in the following code
        interaction.inc_step_idx()

        log.debug(f"action: {self.name}, deprecated: {self.deprecated}, actual_args: {actual_args}, param_list: {self.param_list}")

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
                if self.deprecated:
                    if interaction.was_at_least_one_step_done():
                        return 1
                    else:
                        return 2
                else:
                    interaction.clear_mark()
                    if interaction.was_at_least_one_step_done():
                        return 0
                    else:
                        return 2

        # If it went OK, we reset the time counter that is related to the timeout
        else:
            interaction.set_timeout_starting_time(time.perf_counter())

            if interaction.is_single_step():
                interaction.clear_mark()
                return 0
            if interaction.is_multi_steps():
                if self.deprecated:
                    if run_deprecated_completion_step:
                        return 0
                    else:
                        return 1
                else:
                    if interaction.was_last_step_done():
                        interaction.clear_mark()
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
                f"total_time: {self.__total_time}, timeout: {self.__timeout}, delay: {self.__delay}, "
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
        self.inner = True

    def set_as_not_ready(self) -> None:
        """Sets the action's ready flag to `False`, preventing it from being executed."""
        self.inner = False

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

    def set_timeout(self, timeout: float) -> None:
        """Sets the timeout to a custom value."""
        self.__timeout = timeout

    def set_default_timeout(self) -> None:
        """Sets the timeout to the class-level default value (``Custom.DEFAULT_TIMEOUT``)."""
        self.__timeout = Custom.DEFAULT_TIMEOUT

    def is_pedantic(self) -> bool:
        """Checks if a timeout has been configured for the Custom.

        Returns:
            A boolean indicating if a timeout is set.
        """
        return self.__timeout > 0

    def get_total_time(self) -> float | str:
        """Returns the total execution time configured for this action (0 means no limit)."""
        return self.__total_time

    def get_timeout(self) -> float | str:
        """Returns the timeout configured for this action (0 means no timeout)."""
        return self.__timeout

    def get_delay(self) -> float | str:
        """Returns the delay configured for this action (0 means no delay)."""
        return self.__delay

    def to_list(self) -> list:
        """Converts the action's properties into a list for easy serialization and comparisons.

        Returns:
            A list containing the action's properties.
        """
        total_time = 0.
        timeout = 0.
        delay = 0.
        if isinstance(self.__total_time, str) or self.__total_time > 0:
            total_time = self.__total_time
        if isinstance(self.__timeout, str) or self.__timeout > 0.:
            timeout = self.__timeout
        if isinstance(self.__delay, str) or self.__delay > 0.:
            delay = self.__delay
        return [self.name, self.args, self.inner, self.outer, total_time, timeout, delay, self.msg]

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
            "max_duration": self.__total_time,
            "retry_timeout": self.__timeout,
            "time_to_wait_before_running": self.__delay
        }

    def same_as(self, name: str, args: dict | None) -> bool:
        """Compares the current action to a target action by name and arguments. It returns `True` if they are
        considered the same, ignoring specific arguments like time or timeout.

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
        args_to_exclude = Custom.NOT_ALLOWED_IN_ACTION_SIGNATURE  # Some deprecated actions might still have them
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
            args_to_remove = []
            deprecated_format = False
            for p in Custom.SPECIAL_DEPRECATED_CASES_PREFIXES:
                if self.name.startswith(p):
                    deprecated_format = True
                    break
            for arg_name in args.keys():
                if arg_name in Custom.NOT_ALLOWED_IN_ACTION_SIGNATURE:
                    # if deprecated_format:
                    #    continue
                    if remove_special_arguments:
                        args_to_remove.append(arg_name)
                    else:
                        if exception:
                            raise ValueError(f"Parameter {arg_name} has a private name and cannot be used in action"
                                             f" {self.name}")
                        else:
                            return False
                elif arg_name not in self.param_list:
                    if exception:
                        raise ValueError(f"Unknown parameter {arg_name} for action {self.name}")
                    else:
                        return False
            for arg_name in args_to_remove:
                del args[arg_name]
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
        """Retrieves the list of pending interactions.

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
                log.statem.error(f"Getting actual params for {self.name}; missing param: {param_name}")
                return None
        return actual_params

    get_list_of_requests = get_list_of_interactions  # Backward compatibility

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
        deprecated_hp = any([self.name.startswith(p) for p in Custom.SPECIAL_DEPRECATED_CASES_PREFIXES])
        for p in self.param_list:
            if p in Custom.NOT_ALLOWED_IN_ACTION_SIGNATURE and not deprecated_hp:
                log.user(f"Action {self.name} includes argument {p} in its signature: "
                         f"this is a special argument that is automatically handled, and cannot be "
                         f"included in the function signature (change its name).")

    def __apply_wildcards_to_args(self, args: dict | None = None, in_place: bool = True) -> dict:
        """A private helper method that replaces placeholder values (wildcards) in given action's arguments with their
        actual, concrete values. It handles both single-value and list-based wildcards.
        """
        if not in_place:
            args = copy.deepcopy(args)

        # Applying wildcard-suggested replacements
        for wildcard_from, wildcard_to in self.wildcards.items():

            # ... to arguments
            for k, v in args.items():
                if not isinstance(wildcard_to, str):
                    if wildcard_from == v:
                        args[k] = wildcard_to
                else:
                    if isinstance(v, list):
                        for i, vv in enumerate(v):
                            if isinstance(vv, str) and wildcard_from in vv:
                                v[i] = vv.replace(wildcard_from, wildcard_to)
                    elif isinstance(v, str):
                        if wildcard_from in v:
                            args[k] = v.replace(wildcard_from, wildcard_to)

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
            if self.__total_time_with_wildcard is not None and wildcard_from in self.__total_time_with_wildcard:
                self.__total_time = float(self.__total_time_with_wildcard.replace(wildcard_from, str(wildcard_to)))
            if self.__timeout_with_wildcard is not None and wildcard_from in self.__timeout_with_wildcard:
                self.__timeout = float(self.__timeout_with_wildcard.replace(wildcard_from, str(wildcard_to)))
            if self.__delay_with_wildcard is not None and wildcard_from in self.__delay_with_wildcard:
                self.__delay = float(self.__delay_with_wildcard.replace(wildcard_from, str(wildcard_to)))

    def __guess_total_time(self, args) -> None:
        """A private helper method that attempts to determine the total execution time for an action by looking for a
        'time' or 'seconds' argument.

        Args:
            args: The dictionary of arguments to inspect.
        """
        for arg_name in Custom.SECONDS_ARG_NAMES:
            if arg_name in args:
                try:
                    self.__total_time = max(float(args[arg_name]), 0.)
                    self.__total_time_with_wildcard = None
                except ValueError:
                    if isinstance(args[arg_name], str):
                        self.__total_time = str(args[arg_name])
                        self.__total_time_with_wildcard = str(args[arg_name])
                    else:
                        self.__total_time = -1.
                    pass
                break

    def __guess_timeout(self, args) -> None:
        """A private helper method that attempts to determine the timeout duration for an action by looking for a
        'timeout' argument.

        Args:
            args: The dictionary of arguments to inspect.
        """
        for arg_name in Custom.TIMEOUT_ARG_NAMES:
            if arg_name in args:
                try:
                    self.__timeout = max(float(args[arg_name]), 0.)
                    self.__timeout_with_wildcard = None
                except ValueError:
                    if isinstance(args[arg_name], str):
                        self.__timeout = str(args[arg_name])
                        self.__timeout_with_wildcard = str(args[arg_name])
                    else:
                        self.__timeout = -1.
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

        # Updating by-requester index
        if interaction.requester not in self.by_requester_and_by_insertion_order:
            self.by_requester_and_by_insertion_order[interaction.requester] = []

        # Searching for UUID = None, if already there - do not accumulate multiple requests with UUID None
        if interaction.uuid is None:
            existing_request_same_uuid = self.get_interaction_by_uuid(interaction.requester, interaction.uuid)
            if existing_request_same_uuid:
                return

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
        """Removes all interactions that have been waiting longer than the specified timeout.

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
        if interaction.is_valid():
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

    def get_oldest_interaction(self, requester: str | None = None) -> Interaction | None:
        """Returns the oldest (first-added) interaction, optionally scoped to a specific requester.

        Args:
            requester: If provided, scopes the lookup to the sub-list of this requester.

        Returns:
            The oldest Interaction, or None if the list is empty.
        """
        return self.get_interaction(0, requester)

    def get_most_recent_interaction(self, requester: str | None = None) -> Interaction | None:
        """Returns the most recently added interaction, optionally scoped to a specific requester.

        Args:
            requester: If provided, scopes the lookup to the sub-list of this requester.

        Returns:
            The most recent Interaction, or None if the list is empty.
        """
        return self.get_interaction(-1, requester)

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

    def keep_only_the_most_recent_interaction(self) -> None:
        """Discards all interactions except the most recently added one, preserving its entering time."""
        req = self.get_most_recent_interaction()
        entering_time = self.by_insertion_order_entering_time[req.by_insertion_order_id]
        self.clear()
        self.add(req)
        self.by_insertion_order_entering_time[req.by_insertion_order_id] = entering_time

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
                reqs = [req for req in reqs if req.check_if_doable()]
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
            return ("Action interactions:\n   " +
                    "\n   ".join([str(r.to_code_str(True)) for r in self.by_insertion_order]))
        else:
            return "Action interaction: none"
