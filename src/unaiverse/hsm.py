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
import json
import copy
import html
import time
import inspect
import graphviz
import importlib.resources
from unaiverse.custom import Custom
from collections.abc import Callable
from unaiverse.utils.logger import log
from unaiverse.interaction import Interaction, CompletionReason


class ActionInteractionList:
    def __init__(self, max_per_requester: int = -1):
        self.by_insertion_order = []
        self.by_requester_and_by_insertion_order = {}
        self.max_per_requester = max_per_requester
        self.by_insertion_order_entering_time = []

    def add(self, interaction: Interaction):

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

    def remove(self, interaction: Interaction):
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

    def remove_due_to_timeout(self, timeout_secs: float):
        to_remove = []
        for i, req in enumerate(self.by_insertion_order):
            if (time.perf_counter() - self.by_insertion_order_entering_time[i]) >= timeout_secs:
                to_remove.append(req)
        for req in to_remove:
            self.remove(req)

    def remove_completed(self):
        to_remove = []
        for i, req in enumerate(self.by_insertion_order):
            if req.completed:
                to_remove.append(req)
        for req in to_remove:
            self.remove(req)

    def move_interaction_to_back(self, interaction: Interaction):
        if interaction.is_valid():
            try:
                entering_time = self.by_insertion_order_entering_time[interaction.by_insertion_order_id]
                self.remove(interaction)
                self.add(interaction)
                self.by_insertion_order_entering_time[interaction.by_insertion_order_id] = entering_time
            except Exception as e:
                raise e

    def move_requester_to_back(self, requester: str):
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

    def get_interaction(self, req_order_id: int, requester: str | None = None):
        if req_order_id < 0 and req_order_id != -1:
            return None
        if requester is None:
            return self.by_insertion_order[req_order_id] if req_order_id < len(self.by_insertion_order) else None
        else:
            if requester not in self.by_requester_and_by_insertion_order:
                return None
            return self.by_requester_and_by_insertion_order[requester][req_order_id] \
                if req_order_id < len(self.by_requester_and_by_insertion_order[requester]) else None

    def get_oldest_interaction(self, requester: str | None = None):
        return self.get_interaction(0, requester)

    def get_most_recent_interaction(self, requester: str | None = None):
        return self.get_interaction(-1, requester)

    def get_interaction_by_uuid(self, requester: str, uuid: str | None) -> None | Interaction:
        requests = self.get_interactions(requester)
        if requests is None or len(requests) == 0:
            return None

        for req in requests:
            if req.uuid == uuid:
                return req

    def keep_only_the_most_recent_interaction(self):
        req = self.get_most_recent_interaction()
        entering_time = self.by_insertion_order_entering_time[req.by_insertion_order_id]
        self.clear()
        self.add(req)
        self.by_insertion_order_entering_time[req.by_insertion_order_id] = entering_time

    def get_interactions(self, requester: str | None = None, to_str: bool = False, doable_only: bool = False):
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

    def clear(self):
        self.by_insertion_order.clear()
        self.by_requester_and_by_insertion_order.clear()
        self.by_insertion_order_entering_time.clear()

    def is_requester_known(self, requester: str):
        return requester in self.by_requester_and_by_insertion_order

    def __len__(self):
        return len(self.by_insertion_order)

    def __iter__(self):
        return iter(self.by_insertion_order)

    def __str__(self):
        """Provides a string representation of the `ActionInteractionList` instance.

        Returns:
            A string containing a formatted summary of the instance.
        """
        if len(self.by_insertion_order) > 0:
            return ("Action interactions:\n   " +
                    "\n   ".join([str(r.to_code_str(True)) for r in self.by_insertion_order]))
        else:
            return "Action interaction: none"


class Action:
    # Candidate argument names (when calling an action) that tells that such an action is multi-steps
    SECONDS_ARG_NAMES = {'time'}
    TIMEOUT_ARG_NAMES = {'timeout'}
    DELAY_ARG_NAMES = {'delay'}
    NOT_ALLOWED_IN_ACTION_SIGNATURE = SECONDS_ARG_NAMES | TIMEOUT_ARG_NAMES | DELAY_ARG_NAMES
    INTERACTION_ARG_NAMES = {'interaction', '_requester'}
    DEFAULT_TIMEOUT = 10.0

    # Deprecated
    SPECIAL_DEPRECATED_CASES_PREFIXES = {'ask_', 'do_', 'done_'}
    NOT_READY_PREFIXES = {'get_', 'got_', 'do_', 'done_'}

    def __init__(self, name: str, args: dict, actionable: object,
                 idx: int = -1,
                 ready: bool = True,
                 msg: str | None = None,
                 avoid_changing_ready: bool = False):
        """Initializes an `Action` object, which encapsulates a method to be executed on a given object (`actionable`)
        with specified arguments. It sets up various properties for managing multistep actions, including
        `total_steps`, `total_time`, and `timeout`. It also handles wildcard argument replacement and checks for the
        existence of required parameters. It identifies if the action is a 'not ready' type (e.g., `do_`, `get_`) and
        sets its initial status accordingly.

        Args:
            name: The name of the method to call.
            args: A dictionary of arguments for the method.
            actionable: The object on which the method will be executed.
            idx: A unique ID for the action.
            ready: A boolean indicating if the action is ready to be executed.
            msg: An optional human-readable message.
            avoid_changing_ready: A boolean indicating that the selected ready state should not be changed by
                internal rules.
        """
        # Basic properties
        self.name = name  # Name of the action (name of the corresponding method)
        self.args = args.copy()  # Dictionary of arguments to pass to the action (shallow copy, we will remove some)
        self.actionable = actionable  # Object on which the method whose name is self.name is searched
        self.interactions = ActionInteractionList()  # List of interactions to make this action ready to be executed
        self.id = idx  # Unique ID of the action (-1 if not needed)
        self.msg = msg  # Human-readable message associated to this instance of action
        self.deprecated = any([self.name.startswith(p) for p in Action.SPECIAL_DEPRECATED_CASES_PREFIXES])
        self.deprecated_has_completion = False
        self.inner = ready
        self.outer = True
        self.state_machine = None

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
        self.__total_time = 0  # A total time <= 0 means "no total time at all"
        self.__total_time_with_wildcard = None
        self.__guess_total_time(self.args)  # This will "guess" the value of self.__total_time from the args dict

        # Time-based metrics
        self.__timeout = 0  # A timeout <= 0 means "no total time at all"
        self.__timeout_with_wildcard = None
        self.__guess_timeout(self.args)  # This will "guess" the value of self.__timeout from the args dict

        # Time-based metrics
        self.__delay = 0
        self.__delay_with_wildcard = None
        self.__guess_delay(self.args)  # This will "guess" the value of self.__delay from the args dict

        # Checking arguments (also removing 'timeout', 'delay', etc...)
        self.check_provided_args(self.args, exception=True, remove_special_arguments=True)

        # Argument values replaced by wildcards (commonly assumed to be in the format <value>)
        self.wildcards = {}  # Value-to-value (es: <playlist> to this:and:this)
        self.args_with_wildcards = copy.deepcopy(self.args)  # Backup of the originally provided arguments
        self.msg_with_wildcards = self.msg

        # Checking
        self.deprecated_has_completion = '_completed' in self.param_list

        # Fixing (forcing NOT-ready on some actions)
        if not avoid_changing_ready:
            for prefix in Action.NOT_READY_PREFIXES:
                if self.name.startswith(prefix):
                    self.inner = False
                    break
            for p in self.param_list:
                if p in Action.INTERACTION_ARG_NAMES:
                    self.outer = True
                    break

        # This must be done at the very end of the constructor!
        # This interaction is NOT registered
        self.system_interaction = Interaction(requester="system", target="system",
                                              action_name=self.name, action_kwargs=self.args)
        self.system_interaction.set_action_ref(self)

    @property
    def ready(self):
        return self.inner

    @property
    def has_completion_step(self) -> bool:
        return self.deprecated_has_completion

    def set_state_machine(self, hsm: 'HybridStateMachine'):
        self.state_machine = hsm

    async def __call__(self, interaction: Interaction | None = None):
        """Executes the action's associated method. This is the main entry point for running an action. It handles
        multistep logic by updating the step counter and checking for completion based on steps, time, or timeout.
        It also injects dynamic arguments like the `requester`, `request_time`, and `request_uuid` into the method's
        arguments before execution. If the action is a multistep action and has a completion step, it handles that
        callback as well (async).

        Args:
            interaction: The Interaction object or None, if the action was not requested by other agents.

        Returns:
            A boolean indicating whether the action was executed successfully.
        """
        if interaction is None:
            interaction = self.system_interaction
        actual_args = self.get_actual_params(additional_args=interaction.action_kwargs)
        actual_args = self.__replace_wildcard_values(actual_args)

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
        if self.deprecated:
            if interaction.is_timed_out():
                if interaction.is_single_step():
                    return 2

                if interaction.is_multi_steps():
                    if not interaction.was_at_least_one_step_done():
                        return 2

            # Check if we are in front of an action that completed all its steps (even just a single step or an
            # action  with no data at all) or that was timed out, but it did at last a step (of course,
            # multistep actions only): in those cases, the action is considered "completed in a correct way"
            run_deprecated_completion_step = (interaction.was_last_step_done() or
                                              (interaction.is_timed_out() and interaction.was_at_least_one_step_done()))

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
                if interaction.is_single_step():
                    return 2
                if interaction.is_multi_steps():
                    if interaction.was_at_least_one_step_done():
                        return 0
                    else:
                        return 2
            if interaction.is_multi_steps() and interaction.was_at_least_one_step_done():
                if not self.actionable.im.check_stream_readiness(interaction):
                    return 1

            for p in self.param_list:
                if p in Action.INTERACTION_ARG_NAMES:
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
                    return 2
            if interaction.is_multi_steps():
                if self.deprecated:
                    if interaction.was_at_least_one_step_done():
                        return 1
                    else:
                        return 2
                else:
                    if interaction.was_at_least_one_step_done():
                        return 0
                    else:
                        return 2

        # If it went OK, we reset the time counter that is related to the timeout
        else:
            interaction.set_timeout_starting_time(time.perf_counter())

            if interaction.is_single_step():
                return 0
            if interaction.is_multi_steps():
                if self.deprecated:
                    if run_deprecated_completion_step:
                        return 0
                    else:
                        return 1
                else:
                    if interaction.was_last_step_done():
                        return 0
                    else:
                        return 1

        # Unexpected
        return -1

    def __str__(self):
        """Provides a string representation of the `Action` instance.

        Returns:
            A string containing a formatted summary of the instance.
        """
        return (f"[Action: {self.name}] id: {self.id}, args: {self.args}, param_list: {self.param_list}, "
                f"total_time: {self.__total_time}, timeout: {self.__timeout}, delay: {self.__delay}, "
                f"ready (inner): {self.inner}, outer: {self.outer}, interactions: {str(self.interactions)}, "
                f"msg: {str(self.msg)}]")

    def to_code_str(self):
        s = f"aati:{self.name}|{self.args}|{[self.__total_time, self.__timeout, self.__delay]}|"
        if len(self.interactions) > 0:
            return s + "\n" + "\n".join(f"   {inter}" for inter in self.interactions)
        else:
            return s + "no-int"

    def set_as_ready(self):
        """Sets the action's ready flag to `True`, indicating it can now be executed."""
        self.inner = True

    def set_as_not_ready(self):
        """Sets the action's ready flag to `False`, preventing it from being executed."""
        self.inner = False

    def set_msg(self, msg):
        """Sets the message associated to this action."""

        if msg is not None:
            self.msg = html.unescape(msg)
            self.msg_with_wildcards = self.msg
        else:
            self.msg = None
            self.msg_with_wildcards = None

    def is_ready(self, consider_interactions: bool = True, delay_starting_time: float = -1.):
        """Checks if the action is ready to be executed. It returns `True` if the `ready` flag is set or if there are
        any pending interactions.

        Args:
            consider_interactions: A boolean flag to include pending interactions in the readiness check.

        Returns:
            A boolean indicating the action's readiness.
        """
        is_delayed = (delay_starting_time > 0 and self.__delay > 0 and
                      (time.perf_counter() - delay_starting_time) <= self.__delay)
        return not is_delayed and (self.inner or (consider_interactions and self.outer and len(
            self.interactions.get_interactions(doable_only=True)) > 0))

    def allows_outer_interactions(self):
        return self.outer

    def allows_inner_interactions(self):
        return self.inner

    def set_default_timeout(self):
        self.__timeout = Action.DEFAULT_TIMEOUT

    def is_pedantic(self):
        """Checks if a timeout has been configured for the action.

        Returns:
            A boolean indicating if a timeout is set.
        """
        return self.__timeout > 0

    def get_total_time(self):
        return self.__total_time

    def get_timeout(self):
        return self.__timeout

    def get_delay(self):
        return self.__delay

    def to_list(self, minimal=False):
        """Converts the action's properties into a list for easy serialization. It can generate either a full or a
        minimal representation.

        Args:
            minimal: A boolean flag to return a minimal list representation.

        Returns:
            A list containing the action's properties.
        """
        special_args = {}
        if isinstance(self.__total_time, str) or self.__total_time > 0:
            special_args[next(iter(Action.SECONDS_ARG_NAMES))] = self.__total_time
        if isinstance(self.__timeout, str) or self.__timeout > 0.:
            special_args[next(iter(Action.TIMEOUT_ARG_NAMES))] = self.__timeout
        if isinstance(self.__delay, str) or self.__delay > 0.:
            special_args[next(iter(Action.DELAY_ARG_NAMES))] = self.__delay
        if not minimal:
            if self.msg is not None:
                msg = self.msg.encode("ascii", "xmlcharrefreplace").decode("ascii")
            else:
                msg = None
            return [self.name, self.args | special_args, self.inner, self.id] + ([msg] if msg is not None else [])
        else:
            return [self.name, self.args | special_args]

    def same_as(self, name: str, args: dict | None):
        """Compares the current action to a target action by name and arguments. It returns `True` if they are
        considered the same, ignoring specific arguments like time or timeout.

        Args:
            name: The name of the target action.
            args: The arguments of the target action.

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
        args_to_exclude = Action.NOT_ALLOWED_IN_ACTION_SIGNATURE  # Some deprecated actions might still have them
        return (name == self.name and
                self.check_provided_args(args) and
                all(k in args_to_exclude or k not in self.args or self.args[k] == v for k, v in args.items()))

    def check_provided_args(self, args: dict, exception: bool = False, remove_special_arguments: bool = False) -> bool:
        """A private helper method to validate that all provided arguments for an action exist in the action's
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
            for p in Action.SPECIAL_DEPRECATED_CASES_PREFIXES:
                if self.name.startswith(p):
                    deprecated_format = True
                    break
            for arg_name in args.keys():
                if arg_name in Action.NOT_ALLOWED_IN_ACTION_SIGNATURE:
                    if deprecated_format:
                        continue
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

    def set_wildcards(self, wildcards: dict[str, str | float | int] | None, permanent: bool = False):
        """Replaces wildcard values in the action's arguments with actual values. This method is used to dynamically
        configure actions with context-specific data.

        Args:
            wildcards: A dictionary mapping wildcard placeholders to their concrete values.
        """
        if not permanent:
            self.wildcards = wildcards if wildcards is not None else {}
            self.__replace_wildcard_values()
        else:
            self.__replace_wildcard_values()
            self.args_with_wildcards = copy.deepcopy(self.args)
            self.__timeout_with_wildcard = None
            self.__total_time_with_wildcard = None
            self.__delay_with_wildcard = None
            self.set_msg(self.msg)

    def add_interaction(self, interaction: Interaction):
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

    def clear_interactions(self, requester: str | None = None):
        """Clears all pending requests from the action's list."""
        if requester is None:
            self.interactions = ActionInteractionList()
        else:
            if self.interactions.is_requester_known(requester):
                requests = self.interactions.get_interactions(requester)
                for req in requests:
                    self.interactions.remove(req)

    def clear_interaction(self, requester: str, req_id: int):
        req = self.interactions.get_interaction(req_id, requester)
        if req is not None:
            self.interactions.remove(req)

    def get_list_of_interactions(self) -> ActionInteractionList:
        """Retrieves the list of pending interactions.

        Returns:
            The list of pending interactions, i.e., an object of type ActionInteractionList.
        """
        return self.interactions

    def get_actual_params(self, additional_args: dict | None):
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

    def __action_name_to_callable(self, action_name: str):
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

    def __get_action_params(self):
        """A private helper method that inspects the signature of the action's method to populate the list of
        supported parameters and their default values.
        """
        self.param_list = [param_name for param_name in self.__sig.parameters.keys()]
        self.param_to_default_value = {param.name: param.default for param in self.__sig.parameters.values() if
                                       param.default is not inspect.Parameter.empty}

        # Ensuring those *specially handled* arguments are not present in the method signature (they can only be
        # present in the kwargs used to call the function)
        for p in self.param_list:
            if p in Action.NOT_ALLOWED_IN_ACTION_SIGNATURE and not self.deprecated:
                log.user(f"Action {self.name} includes argument {p} in its signature: "
                         f"this is a special argument that is automatically handled, and cannot be "
                         f"included in the function signature (change its name).")

    def __replace_wildcard_values(self, args: dict | None = None):
        """A private helper method that replaces placeholder values (wildcards) in the action's arguments with their
        actual, concrete values. It handles both single-value and list-based wildcards.
        """

        updated_static_stuff = True
        if args is not None:
            args = copy.copy(args)
            updated_static_stuff = False

        if updated_static_stuff:
            if self.args_with_wildcards is None:
                self.args_with_wildcards = copy.deepcopy(self.args)  # Backup before applying wildcards (1st time only)
            else:
                self.args = copy.deepcopy(self.args_with_wildcards)  # Restore a backup before applying wildcards
            args = self.args

            if self.msg_with_wildcards is not None:
                self.msg = self.msg_with_wildcards

        for wildcard_from, wildcard_to in self.wildcards.items():
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

            if updated_static_stuff:
                if self.msg is not None:
                    self.msg = self.msg.replace(wildcard_from, str(wildcard_to))
                if self.__total_time_with_wildcard is not None and wildcard_from in self.__total_time_with_wildcard:
                    self.__total_time = float(self.__total_time_with_wildcard.replace(wildcard_from, str(wildcard_to)))
                if self.__timeout_with_wildcard is not None and wildcard_from in self.__timeout_with_wildcard:
                    self.__timeout = float(self.__timeout_with_wildcard.replace(wildcard_from, str(wildcard_to)))
                if self.__delay_with_wildcard is not None and wildcard_from in self.__delay_with_wildcard:
                    self.__delay = float(self.__delay_with_wildcard.replace(wildcard_from, str(wildcard_to)))

        return args

    def __guess_total_time(self, args):
        """A private helper method that attempts to determine the total execution time for an action by looking for a
        'time' or 'seconds' argument.

        Args:
            args: The dictionary of arguments to inspect.
        """
        for arg_name in Action.SECONDS_ARG_NAMES:
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

    def __guess_timeout(self, args):
        """A private helper method that attempts to determine the timeout duration for an action by looking for a
        'timeout' argument.

        Args:
            args: The dictionary of arguments to inspect.
        """
        for arg_name in Action.TIMEOUT_ARG_NAMES:
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

    def __guess_delay(self, args):
        """A private helper method that attempts to determine a delay duration for an action by looking for a 'delay'
        argument.

        Args:
            args: The dictionary of arguments to inspect.
        """
        for arg_name in Action.DELAY_ARG_NAMES:
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


class State:

    def __init__(self, name: str, idx: int = -1, action: Action | None = None, waiting_time: float = 0.,
                 blocking: bool = True, msg: str | None = None):
        """Initializes a `State` object, which is a fundamental component of a Hybrid State Machine. A state can be
        associated with an optional `Action` to be performed, a unique name, and various properties like waiting time
        and blocking behavior. It also stores a human-readable message.

        Args:
            name: The unique name of the state.
            idx: A unique ID for the state.
            action: An optional `Action` object to be executed when the state is entered.
            waiting_time: The number of seconds to wait before the state can transition.
            blocking: A boolean indicating if the state blocks execution until a condition is met.
            msg: An optional message associated with the state.
        """
        self.name = name  # Name of the state (must be unique)
        self.action = action  # Inner state action (it can be None)
        self.id = idx  # Unique ID of the state (-1 if not needed)
        self.waiting_time = waiting_time  # Number of seconds to wait in the current state before acting
        self.starting_time = 0.
        self.blocking = blocking
        self.msg = msg  # Human-readable message associated to this instance of state
        self.state_machine = None

        # Fix UNICODE chars
        if self.msg is not None:
            self.msg = html.unescape(self.msg)

        # Message parts replaced by wildcards (commonly assumed to be in the format <value>)
        self.wildcards = {}  # Value-to-value (es: <playlist> to this:and:this)
        self.msg_with_wildcards = self.msg

    async def __call__(self, *args, **kwargs):
        """Executes the state's logic. If a `waiting_time` is set, it starts a timer. If an `action` is associated with
        the state, it resets the action's step counter and then executes the action by calling it. It returns the
        result of the action's execution (async).

        Args:
            *args: Positional arguments to pass to the action's `__call__` method.
            **kwargs: Keyword arguments to pass to the action's `__call__` method.

        Returns:
            The return value of the action's `__call__` method, or `None` if no action is set.
        """

        # The following condition is true only when we enter this state
        if self.starting_time <= 0.:
            self.starting_time = time.perf_counter()

            # Print the state message only when we enter this state: for example, do not print it if we run an action,
            # it fails, and then the callable state function is executed again.
            if self.msg is not None:
                if self.state_machine is not None and self.state_machine.show_blocking_states:
                    if self.blocking:
                        msg = self.msg + " 🔴"
                    else:
                        msg = self.msg + " 🟢"
                else:
                    msg = self.msg
                log.user(msg)

        # The state action is executed when we enter the state AND whenever we further run the state callable function
        if self.action is not None:
            log.statem(f"Running action {self.action.name} on the current state...", state=self.name)
            self.action.system_interaction.reset_state()
            return await self.action(*args, **kwargs)
        else:
            return None

    def __str__(self):
        """Provides a string representation of the `State` object. This is useful for debugging and logging, as it
        summarizes the state's properties, including its name, ID, waiting time, blocking status, and its associated
        action (if any).

        Returns:
            A string containing a formatted summary of the state's instance.
        """
        return (f"[State: {self.name}] id: {self.id}, waiting_time: {self.waiting_time}, blocking: {self.blocking}, "
                f"action -> {self.action if self.action is not None else 'none'}, msg: {self.msg}")

    def set_state_machine(self, hsm: 'HybridStateMachine'):
        self.state_machine = hsm

    def set_msg(self, msg):
        """Sets the message associated to this state."""

        if msg is not None:
            self.msg = html.unescape(msg)
            self.msg_with_wildcards = self.msg
        else:
            self.msg = None
            self.msg_with_wildcards = None

    def must_wait(self):
        """Checks if the state needs to wait before it can transition. It compares the current elapsed time since
        entering the state with the configured `waiting_time`. If the elapsed time is less than the waiting time,
        it returns `True`, indicating the state is still in a waiting period.

        Returns:
            A boolean indicating whether the state is currently waiting.
        """
        if self.waiting_time > 0.:
            if (time.perf_counter() - self.starting_time) >= self.waiting_time:
                return False
            else:
                log.debug(f"Time passing: {(time.perf_counter() - self.starting_time)} seconds")
                return True
        else:
            return False

    def to_list(self):
        """Converts the state's properties into a list. This method is useful for serialization, allowing the state to
        be easily stored or transmitted. It includes the action's minimal list representation, the state's ID,
        blocking status, waiting time, and message.

        Returns:
            A list containing the state's properties.
        """
        if self.msg is not None:
            msg = self.msg.encode("ascii", "xmlcharrefreplace").decode("ascii")
        else:
            msg = None
        return ((self.action.to_list(minimal=True) if self.action is not None else [None, None]) +
                ([self.id, self.blocking, self.waiting_time] + ([msg] if msg is not None else [])))

    def has_action(self):
        """A simple getter that checks if an action is associated with the state.

        Returns:
            True if an action is set, False otherwise.
        """
        return self.action is not None

    def get_starting_time(self):
        """Retrieves the timestamp when the state's execution began. This is used to calculate the elapsed waiting time.

        Returns:
            A float representing the starting time.
        """
        return self.starting_time

    def reset(self):
        """Resets the state's internal counters. This method is typically called when re-entering a state. It sets the
        `starting_time` to zero and also resets the associated action's step counter if an action exists.
        """
        self.starting_time = 0.
        if self.action is not None:
            self.action.system_interaction.reset_state()

    def set_blocking(self, blocking: bool):
        """Sets the blocking status of the state. A blocking state will prevent the state machine from transitioning to
        the next state until the action is fully completed.

        Args:
            blocking: A boolean value to set the blocking status.
        """
        self.blocking = blocking

    def set_wildcards(self, wildcards: dict[str, str | float | int] | None, permanent: bool = False):
        """Replaces wildcard values in the state messages. This method is used to dynamically
        configure state messages with context-specific data.

        Args:
            wildcards: A dictionary mapping wildcard placeholders to their concrete values.
        """
        self.wildcards = wildcards if wildcards is not None else {}
        self.__replace_wildcard_values()
        if permanent:
            self.set_msg(self.msg)

    def __replace_wildcard_values(self):
        """A private helper method that replaces placeholder values (wildcards) in the state message.
        It handles both single-value and list-based wildcards.
        """
        if self.msg_with_wildcards is None:
            self.msg_with_wildcards = self.msg
        else:
            self.msg = self.msg_with_wildcards

        if self.msg is not None:
            for wildcard_from, wildcard_to in self.wildcards.items():
                self.msg = self.msg.replace(wildcard_from, str(wildcard_to))


class HybridStateMachine:
    DEFAULT_WILDCARDS = {'<world>': '<world>', '<agent>': '<agent>', '<partner>': '<partner>', '<role>': '<role>'}

    def __init__(self, actionable: object, wildcards: dict[str, str | float | int] | None = None,
                 policy: Callable[[list[Action]], tuple[int, Interaction | None]] | None = None):
        """Initializes a `HybridStateMachine` object, which orchestrates states and transitions. It manages a set of
        states and actions, and handles the logic for transitions between states based on conditions and a defined
        policy. It sets up initial and current states, wildcards for dynamic arguments, and references to an
        `actionable` object whose methods are the actions to be called. It also includes debug and output settings.

        Args:
            actionable: The object on which actions (methods) are to be executed.
            wildcards: A dictionary of key-value pairs for dynamic argument substitution.
            policy: An optional callable that determines which action to execute from a list of feasible actions.
        """

        # States are identified by strings, and then handled as State object with possibly and integer ID and action
        self.initial_state: str | None = None  # Initial state of the machine
        self.prev_state: str | None = None  # Previous state
        self.limbo_state: str | None = None  # When an action takes more than a step to complete, we are in "limbo"
        self.state: str | None = None  # Current state
        self.role: str | None = None  # Role of the agent in the state machine (e.g., teacher, student, etc.)
        self.enabled: bool = True
        self.states: dict[str, State] = {}  # State name to State object

        # Actions (transitions) are handled as Action objects in-between state strings
        self.transitions: dict[str, dict[str, list[Action]]] = {}  # Pair-of-states to the actions between them
        self.actionable: object = None  # The object on whose methods are actions that the machine calls
        self.wildcards: dict[str, str | float | int] | None = wildcards \
            if wildcards is not None else {}  # From a wildcards string to a specific value (used in action arguments)
        self.policy = policy if policy is not None else self.__policy_first_requested_or_first_ready
        self.policy_filter = None
        self.policy_filter_opts = {}
        self.welcome_msg = None
        self.welcome_msg_with_wildcards = None
        self.show_blocking_states = False
        self.show_action_completion = False
        self.show_action_request_info = False

        # Running data
        self.__action: Action | None = None  # Action that is being executed (could take more than a step to complete)
        self.__last_completed_action: Action | None = None
        self.__cur_feasible_actions_status: dict | None = None  # Store info of the executed action (for multi-steps)
        self.__id_to_state: list[State] = []  # Map from state ID to State object
        self.__id_to_action: list[Action] = []  # Map from action ID to Action object
        self.__state_changed = False  # Internal flag
        self.__id_to_original_state_msg: list[tuple[str | None, str | None]] = []
        self.__id_to_original_action_msg: list[str | None] = []

        # Forcing default wildcards
        self.add_wildcards(HybridStateMachine.DEFAULT_WILDCARDS)

        # Forcing output function
        self.__debug_messages_active = False

        self.set_actionable(actionable)

    # Backward compatibility
    def set_print_fcn(self, print_fcn, supports_html):
        if log is not None:
            log.set_print_fcn(print_fcn, supports_html)

    def show_ticks_in_action_messages(self, do_it: bool = True):
        self.show_action_completion = do_it

    def show_marks_in_blocking_state_messages(self, do_it: bool = True):
        self.show_blocking_states = do_it

    def show_request_info_in_action_messages(self, do_it: bool = True):
        self.show_action_request_info = do_it

    def set_welcome_message(self, msg):
        """Sets a message that will be printed only once when the initial state is reached."""

        if msg is not None:
            self.welcome_msg = html.unescape(msg)
            self.welcome_msg_with_wildcards = self.welcome_msg
        else:
            self.welcome_msg = None
            self.welcome_msg_with_wildcards = None

    def to_dict(self):
        """Serializes the state machine's current configuration into a dictionary. This includes its states,
        transitions, roles, and the current action being executed. It is useful for saving the state of the machine or
        for logging its status in a structured format.

        Returns:
            A dictionary representation of the state machine's properties.
        """
        return {
            'initial_state': self.initial_state,
            'state': self.state,
            'role': self.role,
            'prev_state': self.prev_state,
            'limbo_state': self.limbo_state,
            'welcome_msg':
                self.welcome_msg_with_wildcards.encode("ascii", "xmlcharrefreplace").decode(
                    "ascii") if self.welcome_msg_with_wildcards is not None else None,
            'highlight_blocking_states_in_messages': self.show_blocking_states,
            'show_action_ticks_after_messages': self.show_action_completion,
            'show_action_request_after_messages': self.show_action_request_info,
            'state_actions': {state.name: state.to_list() for state in self.__id_to_state},
            'transitions': {from_state: {to_state: [act.to_list() for act in action_list] for to_state, action_list in
                                         to_states.items()} for from_state, to_states in self.transitions.items() if
                            len(to_states) > 0},
            'cur_action': self.__action.to_list() if self.__action is not None else None
        }

    def __str__(self):
        """Generates a human-readable string representation of the state machine. It uses the `to_dict` method to get
        the machine's data and then formats it as a compact JSON string, making it easy to inspect for debugging
        purposes.

        Returns:
            A formatted JSON string representing the state machine.
        """
        hsm_data = self.to_dict()

        def custom_serializer(obj):
            if not isinstance(obj, (int, str, float, bool, list, tuple, dict, set)):
                return "_non_basic_type_removed_"
            else:
                return obj

        json_str = json.dumps(hsm_data, indent=4, default=custom_serializer)

        # Compacting lists
        def remove_newlines_in_lists(json_string):
            stack = []
            output = []
            i = 0
            while i < len(json_string):
                char = json_string[i]
                if char == '[':
                    stack.append('[')
                    output.append(char)
                elif char == ']':
                    stack.pop()
                    output.append(char)
                elif char == '\n' and stack:  # Skipping newline
                    i += 1
                    while i < len(json_string) and json_string[i] in ' \t':
                        i += 1
                    if output[-1] == ",":
                        output.append(" ")
                    continue  # Do not output newline or following spaces
                else:
                    output.append(char)
                i += 1
            return ''.join(output)

        return remove_newlines_in_lists(json_str)

    def set_actionable(self, obj: object):
        """Sets the object on which the state machine's actions will be performed. This allows the same state machine
        logic to be applied to different objects. It updates the `actionable` reference for all states and actions
        within the machine.

        Args:
            obj: The object instance to be set as the new `actionable`.
        """
        if obj is None:
            return

        self.actionable = obj

        for state_obj in self.states.values():
            if state_obj.action is not None:
                state_obj.action.actionable = obj

    def set_policy(self, policy_fcn: Callable[[list[Action]], tuple[int, Interaction | None]]):
        """Sets the policy to be used in selecting what action to perform in the current state.

        Args:
            policy_fcn: A function that takes a list of `Action` objects that are candidates for execution, and returns
                the index of the selected action and an ActionRequest object with the action-requester details (object,
                arguments, time, and UUID), or -1 and None if no action is selected.
        """
        self.policy = policy_fcn

    def set_policy_filter(self, filter_fcn: Callable[
        [int, Interaction | None, list[Action], dict], tuple[int, Interaction | None]],
                          filter_fcn_opts: dict):
        """Sets the filter function that will overload the decision of the policy.

        Args:
            filter_fcn: A function that takes the decision of the policy, a list of `Action` objects
                that are candidates for execution, and a customizable dict of options, and returns the index of the
                selected action and an ActionRequest with the requested details (requests, arguments, time, and UUID),
                or -1 and None if no action is selected.
            filter_fcn_opts: A reference to the dictionary of custom options that will be passed to the filter function.
        """
        self.policy_filter = filter_fcn
        self.policy_filter_opts = filter_fcn_opts
        self.policy_filter_opts.clear()

    def set_wildcards(self, wildcards: dict[str, str | float | int] | None, permanent: bool = False):
        """Sets the dictionary of wildcards that are used to dynamically replace placeholder values in action
        arguments. It updates all actions with the new wildcard dictionary.

        Args:
            wildcards: A dictionary containing wildcard key-value pairs.
        """
        wildcards = wildcards if wildcards is not None else {}
        if not permanent:
            self.wildcards = wildcards
        for action in self.__id_to_action:
            action.set_wildcards(wildcards, permanent)
        for state in self.__id_to_state:
            state.set_wildcards(wildcards, permanent)
        if self.welcome_msg is not None:
            self.welcome_msg = self.welcome_msg_with_wildcards  # Restore before updating
            for wildcard_from, wildcard_to in self.wildcards.items():
                self.welcome_msg = self.welcome_msg.replace(wildcard_from, str(wildcard_to))  # Update
            if permanent:
                self.set_welcome_message(self.welcome_msg)

    def set_role(self, role: str):
        """Sets the role of the agent associated with this state machine. This can be used to influence state machine
        behavior based on the agent's role (e.g., 'teacher', 'student').

        Args:
            role: The string representation of the new role.
        """
        self.role = role
        self.update_wildcard("<role>", self.role)

    def get_wildcards(self):
        """Retrieves the dictionary of wildcards currently used by the state machine.

        Returns:
            A dictionary of the wildcards.
        """
        return self.wildcards

    def add_wildcards(self, wildcards: dict[str, str | float | int | list[str]], permanent: bool = False):
        """Adds new key-value pairs to the existing wildcard dictionary. It also triggers an update to all actions with
        the new combined dictionary.

        Args:
            wildcards: A dictionary of new wildcards to add.
        """
        if not permanent:
            self.wildcards.update(wildcards)
            self.set_wildcards(self.wildcards, permanent=False)
        else:
            self.set_wildcards(self.wildcards, permanent=True)

    def replace_wildcards(self, wildcards: dict[str, str | float | int | list[str]]):
        self.add_wildcards(wildcards, permanent=True)

    def update_wildcard(self, wildcard_key: str, wildcard_value: str | float | int):
        """Updates the value of a single existing wildcard. It raises an error if the key does not exist. This method
        is useful for changing a single dynamic value without redefining all wildcards.

        Args:
            wildcard_key: The key of the wildcard to update.
            wildcard_value: The new value for the wildcard.
        """
        assert wildcard_key in self.wildcards, f"{wildcard_key} is not a valid wildcard"
        self.wildcards[wildcard_key] = wildcard_value
        self.set_wildcards(self.wildcards)

    def get_action_step_idx(self):
        """Retrieves the current step index of the action being executed. This is particularly useful for tracking the
        progress of multistep actions.

        Returns:
            An integer representing the current step, or -1 if no action is running.
        """
        return self.__cur_feasible_actions_status['selected_interaction'].get_step_idx() \
            if self.__action is not None else -1

    def is_busy_acting(self):
        """Checks if the state machine is currently executing an action. This is determined by checking if the action
        step index is greater than or equal to 0.

        Returns:
            True if an action is running, False otherwise.
        """
        return self.get_action_step_idx() >= 0

    def add_state(self, state: str, action: str = None, args: dict | None = None, state_id: int | None = None,
                  waiting_time: float | None = None, blocking: bool | None = None,
                  msg: str | None = None, msg_action: str | None = None):
        """Adds a new state to the state machine. This method can create a new state with an optional inner action or
        update an existing state. It assigns a unique ID to the state and its action.

        Args:
            state: The name of the state to add.
            action: The name of the action to associate with the state.
            args: A dictionary of arguments for the action.
            state_id: An optional unique ID for the state.
            waiting_time: A float representing a delay before the state can transition.
            blocking: A boolean indicating if the state is blocking.
            msg: A human-readable message for the state.
            msg_action: A human-readable message for the action running on this state.
        """
        if args is None:
            args = {}
        sta_obj = None
        if state_id is None:
            if state not in self.states:
                state_id = len(self.__id_to_state)
            else:
                sta_obj = self.states[state]
                state_id = sta_obj.id
        if action is None:
            act = sta_obj.action if sta_obj is not None else None
        else:
            act = Action(name=action, args=args, idx=len(self.__id_to_action),
                         actionable=self.actionable, avoid_changing_ready=True,
                         msg=msg_action)
            act.set_wildcards(self.wildcards)
            self.__id_to_action.append(act)
        if waiting_time is None:
            waiting_time = sta_obj.waiting_time if sta_obj is not None else 0.  # Default waiting time
        if blocking is None:
            blocking = sta_obj.blocking if sta_obj is not None else True  # Default blocking
        if msg is None:
            msg = sta_obj.msg_with_wildcards if sta_obj is not None else None

        sta = State(name=state, idx=state_id, action=act, waiting_time=waiting_time, blocking=blocking, msg=msg)
        sta.set_wildcards(self.wildcards)
        if state not in self.states:
            self.__id_to_state.append(sta)
        else:
            self.__id_to_state[state_id] = sta
        self.states[state] = sta

        if len(self.__id_to_state) == 1 and self.state is None:
            self.set_state(sta.name)

        sta.set_state_machine(self)

    def get_state_name(self, consider_limbo: bool = False):
        """Retrieves the name of the current state of the state machine.

        Returns:
            A string with the state's name, or `None` if no state is set.
        """

        if not consider_limbo:
            return self.state
        else:
            if self.state is None and self.limbo_state is not None:
                return self.limbo_state
            else:
                return self.state

    def get_state(self):
        """Retrieves the current `State` object of the state machine.

        Returns:
            A `State` object or `None`.
        """
        return self.states[self.state] if self.state is not None else None

    def get_all_states(self):
        """Retrieves the list of all `State` objects.

        Returns:
            List of `State` objects.
        """
        return self.__id_to_state

    def get_all_actions(self):
        """Retrieves the list of all `Action` objects.

        Returns:
            List of `Action` objects.
        """
        return self.__id_to_action

    def get_action(self):
        """Retrieves the `Action` object that is currently being executed.

        Returns:
            An `Action` object or `None`.
        """
        return self.__action

    def get_action_name(self):
        """Retrieves the name of the action currently being executed.

        Returns:
            A string with the action's name, or `None` if no action is running.
        """
        return self.__action.name if self.__action is not None else None

    def get_last_completed_action_name(self):
        """Retrieves the name of the last action that was correctly executed.

        Returns:
            A string with the action's name, or `None` if no actions were executed before.
        """
        return self.__last_completed_action.name if self.__last_completed_action is not None else None

    def reset_state(self):
        """Resets the state machine to its initial state. This clears the current action, the previous state, and
        the limbo state. It also resets the step counters for all actions within the machine.
        """
        self.state = self.initial_state
        self.limbo_state = None
        self.prev_state = None
        self.__action = None
        for act in self.__id_to_action:
            act.clear_interactions()
            act.system_interaction.reset_state()
        for s in self.__id_to_state:
            if s.action is not None:
                s.action.clear_interactions()
                s.action.system_interaction.reset_state()

    def get_states(self):
        """Returns an iterable of all state names defined in the state machine.

        Returns:
            An iterable of state names.
        """
        return list(set(list(self.transitions.keys()) + self.__id_to_state))

    def set_state(self, state: str):
        """Sets the current state of the state machine to a new, specified state. It also handles the transition logic
        by resetting the current action and updating the previous state. Raises an error if the new state is not known
        to the machine.

        Args:
            state: The name of the state to transition to.
        """
        if state in self.transitions or state in self.states:
            self.prev_state = self.state
            self.state = state
            if self.__action is not None:
                self.__cur_feasible_actions_status['selected_interaction'].reset_status()
                self.__action = None
            if self.initial_state is None:
                self.initial_state = state
        else:
            raise ValueError("Unknown state: " + str(state))

    def are_debug_messages_active(self):
        return self.__debug_messages_active

    def set_debug_messages_active(self, yes: bool):
        self.__debug_messages_active = yes

        if yes:
            self.show_ticks_in_action_messages(True)
            self.show_marks_in_blocking_state_messages(True)
            self.show_request_info_in_action_messages(True)

            # Replace original messages
            self.generate_auto_messages(force=True)
        else:
            self.show_ticks_in_action_messages(False)
            self.show_marks_in_blocking_state_messages(False)
            self.show_request_info_in_action_messages(False)

            # Restore original messages
            if len(self.__id_to_original_state_msg) > 0:
                for i, state in enumerate(self.__id_to_state):
                    state.set_msg(self.__id_to_original_state_msg[i][0])
                    if state.action is not None:
                        state.action.set_msg(self.__id_to_original_state_msg[i][1])
            if len(self.__id_to_original_action_msg) > 0:
                for i, action in enumerate(self.__id_to_action):
                    action.set_msg(self.__id_to_original_action_msg[i])
            self.__id_to_original_state_msg.clear()
            self.__id_to_original_action_msg.clear()

    def generate_auto_messages(self, states: bool = True, actions: bool = True, force: bool = False):
        if states is True and len(self.__id_to_original_state_msg) == 0:
            for state in self.__id_to_state:
                original1 = state.msg_with_wildcards
                original2 = state.action.msg_with_wildcards if state.action is not None else None

                if state.msg_with_wildcards is None:
                    state.set_msg("📍 " + state.name.replace('_', ' ').capitalize())
                elif force is True:
                    state.set_msg("📍 " + state.name.replace('_', ' ').capitalize() +
                                  " [" + state.msg_with_wildcards + "]")
                if state.action is not None:
                    if state.action.msg_with_wildcards is None:
                        state.action.set_msg("📍 " + state.action.name.replace('_', ' ').capitalize())
                    elif force is True:
                        state.action.set_msg("📍 " + state.action.name.replace('_', ' ').capitalize() +
                                             " [" + state.action.msg_with_wildcards + "]")

                self.__id_to_original_state_msg.append((original1, original2))
        if actions is True and len(self.__id_to_original_action_msg) == 0:
            for action in self.__id_to_action:
                original = action.msg_with_wildcards

                if action.msg_with_wildcards is None:
                    action.set_msg("🚀 " + action.name.replace('_', ' ').capitalize())
                elif force:
                    action.set_msg("🚀 " + action.name.replace('_', ' ').capitalize() +
                                   " [" + action.msg_with_wildcards + "]")

                self.__id_to_original_action_msg.append(original)

    def add_transit(self, from_state: str, to_state: str,
                    action: str, args: dict | None = None, ready: bool = True,
                    act_id: int | None = None, msg: str | None = None, avoid_changing_ready: bool = False):
        """Defines a transition between two states with an associated action. This method is central to building the
        state machine's logic. It can also handle loading and integrating a complete state machine from a file,
        resolving any state name clashes.

        Args:
            from_state: The name of the starting state.
            to_state: The name of the destination state (can be a file path to load another HSM).
            action: The name of the action to trigger the transition.
            args: A dictionary of arguments for the action.
            ready: A boolean indicating if the action is ready by default.
            act_id: An optional unique ID for the action.
            msg: An optional human-readable message for the action.
            avoid_changing_ready: A boolean indicating that the selected ready state should not be changed by
                internal rules.
        """

        # Plugging a previously loaded HSM
        if to_state.lower().endswith(".json"):
            if not os.path.exists(to_state):
                raise FileNotFoundError(f"Cannot find {to_state}")

            file_name = to_state
            hsm = HybridStateMachine(self.actionable).load(file_name)

            # First, we avoid name clashes, renaming already-used-state-names in original_name~1 (or ~2, or ~3, ...)
            hsm_states = list(hsm.states.keys())  # Keep the list(...) thing, since we need a copy here (it will change)
            for state in hsm_states:
                renamed_state = state
                i = 1
                while renamed_state in self.states or (i > 1 and renamed_state in hsm.states):
                    renamed_state = state + "." + str(i)
                    i += 1

                if hsm.initial_state == state:
                    hsm.initial_state = renamed_state
                if hsm.prev_state == state:
                    hsm.prev_state = renamed_state
                if hsm.state == state:
                    hsm.state = renamed_state
                if hsm.limbo_state == state:
                    hsm.limbo_state = renamed_state

                hsm.states[renamed_state] = hsm.states[state]
                if renamed_state != state:
                    del hsm.states[state]
                hsm.transitions[renamed_state] = hsm.transitions[state]
                if renamed_state != state:
                    del hsm.transitions[state]

                for to_states in hsm.transitions.values():
                    if state in to_states:
                        to_states[renamed_state] = to_states[state]
                        if renamed_state != state:
                            del to_states[state]

            # Saving
            initial_state_was_set = self.initial_state is not None
            state_was_set = self.state is not None

            # Include actions/states from another HSM
            self.include(hsm)

            # Adding a transition to the initial state of the given HSM
            self.add_transit(from_state=from_state, to_state=hsm.initial_state, action=action, args=args,
                             ready=ready, act_id=None, msg=msg)

            # Restoring
            self.initial_state = from_state if not initial_state_was_set else self.initial_state
            self.state = from_state if not state_was_set else self.state
            return

        # Adding a new transition
        if from_state not in self.transitions:
            if from_state not in self.states:
                self.add_state(from_state, action=None)
            self.transitions[from_state] = {}
        if to_state not in self.transitions:
            if to_state not in self.states:
                self.add_state(to_state, action=None)
            self.transitions[to_state] = {}
        if args is None:
            args = {}
        if act_id is None:
            act_id = len(self.__id_to_action)

        # Clearing
        if to_state not in self.transitions[from_state]:
            self.transitions[from_state][to_state] = []

        # Checking
        existing_action_list = self.transitions[from_state][to_state]
        for existing_action in existing_action_list:
            if existing_action.same_as(name=action, args=args):
                raise ValueError(f"Repeated transition from {from_state} to {to_state}: "
                                 f"{existing_action.to_list()}")

        # Adding the new action
        new_action = Action(name=action, args=args, idx=act_id, actionable=self.actionable, ready=ready, msg=msg,
                            avoid_changing_ready=avoid_changing_ready)
        self.transitions[from_state][to_state].append(new_action)
        self.__id_to_action.append(new_action)

        new_action.set_state_machine(self)

    def include(self, hsm, make_a_copy=False):
        """Integrates the states and transitions of another state machine (`hsm`) into the current one. This is a
        crucial method for composing complex state machines from smaller, reusable components. It copies wildcards,
        states, and transitions, ensuring that all actions and states are properly added and linked. This method also
        handles an optional `make_a_copy` flag to completely replicate the source machine's state (e.g., current state,
        initial state).

        Args:
            hsm: The `HybridStateMachine` object to include.
            make_a_copy: A boolean to indicate whether the current state machine should adopt the state (e.g.,
                current state, initial state) of the included one.
        """

        # Copying wildcards
        self.add_wildcards(hsm.get_wildcards())

        # Adding states before adding transitions, so that we also add inner state actions, if any
        for _state in hsm.states.values():
            self.add_state(state=_state.name,
                           action=_state.action.name if _state.action is not None else None,
                           waiting_time=_state.waiting_time,
                           args=_state.action.args if _state.action is not None else None,
                           state_id=None,
                           blocking=_state.blocking,
                           msg=_state.msg_with_wildcards)

        # Copy all the transitions of the HSM
        for _from_state, _to_states in hsm.transitions.items():
            for _to_state, _action_list in _to_states.items():
                for _action in _action_list:
                    special_args = {}
                    if isinstance(_action.get_total_time(), str) or _action.get_total_time() > 0:
                        special_args[next(iter(Action.SECONDS_ARG_NAMES))] = _action.get_total_time()
                    if isinstance(_action.get_timeout(), str) or _action.get_timeout() > 0.:
                        special_args[next(iter(Action.TIMEOUT_ARG_NAMES))] = _action.get_timeout()
                    if isinstance(_action.get_delay(), str) or _action.get_delay() > 0.:
                        special_args[next(iter(Action.DELAY_ARG_NAMES))] = _action.get_delay()
                    self.add_transit(from_state=_from_state, to_state=_to_state, action=_action.name,
                                     args=_action.args | special_args, ready=_action.ready,
                                     act_id=None, msg=_action.msg_with_wildcards, avoid_changing_ready=True)

        if make_a_copy:
            self.state = hsm.state
            self.prev_state = hsm.state
            self.initial_state = hsm.initial_state
            self.limbo_state = hsm.limbo_state
            self.set_welcome_message(hsm.welcome_msg_with_wildcards)
            self.show_blocking_states = hsm.show_blocking_states
            self.show_action_completion = hsm.show_blocking_states
            self.show_action_request_info = hsm.show_action_request_info

    def must_wait(self):
        """Checks if the current state is in a waiting period before any transitions can occur.

        Returns:
            A boolean indicating if the state machine must wait.
        """
        if self.state is not None:
            return self.states[self.state].must_wait()
        else:
            return False

    def is_enabled(self):
        """A simple getter to check if the state machine is currently enabled to run.

        Returns:
            True if the state machine is enabled, False otherwise.
        """
        return self.enabled

    def enable(self, yes_or_not: bool):
        """Enables or disables the state machine. When disabled, the `act_states` and `act_transitions` methods will
        not perform any actions.

        Args:
            yes_or_not: A boolean to enable (`True`) or disable (`False`) the state machine.
        """
        self.enabled = yes_or_not

    async def act_states(self):
        """Executes the inner action of the current state, if one exists. This method is for actions that occur upon
        entering a state but do not cause an immediate transition. It only runs if the state machine is enabled (async).
        """
        if not self.enabled:
            return

        if self.state is not None:  # When in the middle of an action, the state is Nones
            await self.states[self.state]()  # Run the action (if any)

    async def act_transitions(self, only_the_ones_with_interactions: bool = False):
        """This is the core execution loop for transitions. It finds all feasible actions from the current state and,
        using a policy, selects and executes one. It handles single-step and multistep actions, managing state changes,
        timeouts, and failed executions. It returns an integer status code indicating the outcome (e.g., transition
        done, try again, move to next action) (async).

        Args:
            only_the_ones_with_interactions: A boolean to consider only actions that have pending interactions.

        Returns:
            An integer status code: `0` for a successful transition, `1` to retry the same action, `2` to move to the
            next action, or `-1` if no actions were found.
        """
        if not self.enabled:
            return -1

        # Collecting list of feasible actions, wait flags, etc. (from the current state)
        if self.__cur_feasible_actions_status is None:
            if self.state is None:
                return -1

            actions_list = []
            to_state_list = []
            attempts_to_serve_an_interaction_list = []

            for to_state, action_list in self.transitions[self.state].items():
                for i, action in enumerate(action_list):

                    # Checking is_ready will check if streams are ready and if the interaction was completed meanwhile
                    if (action.is_ready(consider_interactions=True,
                                        delay_starting_time=self.states[self.state].starting_time) and
                            (not only_the_ones_with_interactions or
                             len(action.interactions.get_interactions(doable_only=True)) > 0)):
                        actions_list.append(action)
                        to_state_list.append(to_state)
                        attempts_to_serve_an_interaction_list.append(0)

            if len(actions_list) > 0:
                self.__cur_feasible_actions_status = {
                    'actions_list': actions_list,
                    'to_state_list': to_state_list,
                    'selected_idx': 0,
                    'selected_interaction': None,
                    'attempts_to_serve_an_interaction_list': attempts_to_serve_an_interaction_list
                }
        else:

            # Reloading the already computed set of actions, wait flags, etc. (when in the middle of an action)
            actions_list = self.__cur_feasible_actions_status['actions_list']
            to_state_list = self.__cur_feasible_actions_status['to_state_list']
            attempts_to_serve_an_interaction_list = (
                self.__cur_feasible_actions_status)['attempts_to_serve_an_interaction_list']

            # Pruning interactions that were completed, meanwhile (due to some timeouts), if any
            idx_to_remove = []
            for i, action in enumerate(actions_list):
                if len(action.interactions) > 0:
                    action.interactions.remove_completed()
                    if len(action.interactions) == 0:
                        idx_to_remove.append(i)
            for i in idx_to_remove:
                if self.__action is not None and self.__action == actions_list[i]:
                    self.__action = None
                del actions_list[i]
                del to_state_list[i]
                del attempts_to_serve_an_interaction_list[i]

        # Using the selected policy to decide what action to apply
        while len(actions_list) > 0:

            # It there was an already selected action (for example a multistep action), then continue with it,
            # otherwise, select a new one following a certain policy (actually, first-come first-served)
            if self.__action is None:
                log.statem(f"List of action to choose from:\n   " +
                           "\n   ".join([a.to_code_str().replace("\n", "\n   ")
                                         for a in actions_list]), state=self.get_state_name())

                # Naive policy: take the first action that is ready
                _idx, _interaction = self.policy(actions_list)

                if _idx < 0:
                    log.statem(f"Selected no actions", state=self.get_state_name())

                    # No actions were applied
                    self.__cur_feasible_actions_status = None
                    self.__state_changed = False
                    return -1  # Early stop
                else:
                    if _interaction is not None:
                        log.statem(f"Selected {actions_list[_idx].to_code_str()}, "
                                   f"{_interaction.to_code_str(True)}",
                                   state=self.get_state_name())
                    else:
                        log.statem(f"Selected {actions_list[_idx].to_code_str()}, no-interactions",
                                   state=self.get_state_name())

                # Revisiting decisions due to the policy filter
                if self.policy_filter is not None:
                    try:
                        _idx_f, _interactions_f = self.policy_filter(_idx, _interaction,
                                                                     actions_list, self.policy_filter_opts)
                    except Exception as e:
                        log.statem.error(f"Skipping policy filter due to exception: {e}",
                                         state=self.get_state_name())
                        _idx_f = _idx
                        _interactions_f = _interaction

                    if _idx_f != _idx or _interactions_f != _interaction:
                        _idx = _idx_f
                        _interaction = _interactions_f
                        if _idx < 0:
                            log.statem(f"Filter selected no actions")

                            # No actions were applied
                            self.__cur_feasible_actions_status = None
                            self.__state_changed = False
                            return -1  # Early stop
                        else:
                            if _interaction is not None:
                                log.statem(
                                    f"Filter selected {actions_list[_idx].to_code_str()}, {_interaction.__str__()}",
                                    state=self.get_state_name())
                            else:
                                log.statem(f"Filter selected {actions_list[_idx].to_code_str()}, "
                                           f"no-interactions", state=self.get_state_name())
                    else:
                        log.statem(f"Filter confirmed the selection", state=self.get_state_name())

                # Saving current action
                self.limbo_state = self.state
                self.state = None
                self.__action = actions_list[_idx]
                _interaction.reset_state()  # Resetting
                self.__cur_feasible_actions_status['selected_idx'] = _idx
                self.__cur_feasible_actions_status['selected_interaction'] = _interaction

            # References
            action = self.__action
            idx = self.__cur_feasible_actions_status['selected_idx']
            interaction = self.__cur_feasible_actions_status['selected_interaction']

            # If this action has an associated Interaction, set it as current on the IM
            # (this configures stdin/stdout for the action)
            # If the Interaction is None, the stdin will be set back to the default streams
            if self.actionable.im.current is None or self.actionable.im.current != interaction:
                self.actionable.im.set_current(interaction)

            # Status can be one of these:
            # 0: action fully done;
            # 1: try again this action;
            # if action.name == "do_learn":
            #    print(f"calling={action.name}, uuid={interaction.uuid if interaction is not None else 'no int'}")
            log.statem(f">>> ACTION {self.__action.name}...", state=self.get_state_name(True))
            if len(self.__action.get_list_of_interactions()) > 0:
                log.statem(str(self.__action.get_list_of_interactions()))

            status = await action(interaction=interaction)
            # if action.name == "do_learn":
            #    print(f"returned status={status}")

            if status == 0:
                log.statem(f"+++ ACTION {self.__action.name} correctly completed", state=self.get_state_name(True))
            elif status == 1:
                log.statem(f"~~~ ACTION {self.__action.name} will be run again", state=self.get_state_name(True))
            else:
                log.statem(f"--- ACTION {self.__action.name} failed", state=self.get_state_name(True))

            if action.msg is not None and self.show_action_completion:
                log.user(Custom.ACTION_TICKS_PER_STATUS[status])

            # Post-call operations
            if status == 0:  # Done

                # Clearing request
                if interaction != self.__action.system_interaction:
                    self.__action.get_list_of_interactions().remove(interaction)

                # State transition
                self.prev_state = self.limbo_state
                self.state = to_state_list[idx]
                self.limbo_state = None

                # Complete the associated Interaction (if any) via Interaction Manager (IM)
                if interaction is not None:
                    # The IM on the actionable (agent) will handle the completion, also saving the destination state
                    self.actionable.im.complete_current(self.state, CompletionReason.OK)

                # Update status
                self.__state_changed = self.state != self.prev_state  # Checking if we are on a self-loop or not
                self.__last_completed_action = self.__action  # This will be set also if the state does not change

                # If we moved to another state
                # (this is not true anymore: "clearing all the pending annotations for the next possible actions")
                if self.__state_changed:
                    log.statem(f">>> MOVING TO STATE: {self.state}", state=self.get_state_name())
                    # for to_state, action_list in self.transitions[self.state].items():
                    #    for i, act in enumerate(action_list):
                    #        act.clear_requests()

                    # Propagating (trying to propagate forward the residual requests)
                    list_of_residual_interactions = self.__action.get_list_of_interactions()
                    propagated_requests = []
                    for interaction in list_of_residual_interactions:
                        interaction.from_state = None
                        interaction.to_state = None
                        if self.request_action(interaction):
                            _interaction = Interaction()
                            propagated_requests.append(_interaction.from_dict(interaction.to_dict()))
                    for _interaction in propagated_requests:
                        list_of_residual_interactions.remove(_interaction)  # Clearing propagated requests

                    # if len(propagated_requests) > 0:
                    #    print(f"!!! Reached state {self.state}, "
                    #          f"propagated these requests taken from {self.__action.name}, and
                    #          now starting from here {propagated_requests}")
                    self.states[self.prev_state].reset()  # Reset starting time (only if state changed!)

                interaction.reset_state()
                self.__action = None  # Clearing
                self.__cur_feasible_actions_status = None

                return 0  # Transition done, no need to check other actions!

            elif status == 1:  # Try again the same action (either a new step or an already done-and-failed one)

                # Update status
                self.__state_changed = False
                if self.prev_state is not None:
                    self.states[self.prev_state].reset()  # Reset starting time

                return 1  # Transition not-done: no need to check other actions, the current one will be run again

            elif status == 2:  # Move to the next action (or to the next request of the same action)

                # Clearing request
                if interaction is not None:
                    self.__action.interactions.move_interaction_to_back(interaction)  # Rotating to avoid starvation
                    attempts_to_serve_an_interaction_list[idx] += 1

                # Back to the original state
                self.state = self.limbo_state
                self.limbo_state = None

                # Purging action from the current list
                if interaction is None or attempts_to_serve_an_interaction_list[idx] >= len(self.__action.interactions):
                    del actions_list[idx]
                    del to_state_list[idx]

                # Update status
                self.__state_changed = False
                interaction.reset_state()
                self.__action = None  # Clearing

                continue  # Move to the next action
            else:
                raise ValueError("Unexpected status: " + str(status))

        # No actions were applied
        self.__cur_feasible_actions_status = None
        self.__state_changed = False
        return -1

    async def act(self):
        """A high-level method that combines `act_states` and `act_transitions` to run the state machine. It repeatedly
        processes states and transitions until a blocking state is reached or all feasible actions have been tried,
        thus ensuring a complete processing cycle in one call (async).
        """

        # It keeps processing states and actions, until all the current feasible actions fail
        # (also when a step of a multistep action is executed) or a blocking state is reached
        while True:
            if self.welcome_msg is not None and self.state is not None and self.state == self.initial_state:
                log.user(self.welcome_msg)
                self.set_welcome_message(None)

            await self.act_states()
            ret = await self.act_transitions(self.must_wait())
            if ret != 0 or (self.state is not None and self.states[self.state].blocking):
                break

    def get_state_changed(self):
        """Returns an internal flag that indicates if a state transition has occurred in the last execution cycle.
        This can be used by an external loop to know when to re-evaluate the state machine's context.

        Returns:
            True if the state has changed, False otherwise.
        """
        return self.__state_changed

    def request_action(self, interaction: Interaction | None = None, **kwargs):
        """Allows an external entity to request a specific action. The request is validated by a signature checker
        (if one exists) and then queued on the corresponding action. This method enables dynamic, external triggers for
        state machine transitions.

        Args:
            interaction: The interaction object.

        Returns:
            True if the request was accepted and queued, False otherwise.
        """

        # Backward compatibility
        if interaction is None:
            interaction = Interaction(
                action_name=kwargs.get('action_name', None),
                action_kwargs=kwargs.get('args', None),
                from_state=kwargs.get('from_state', None),
                to_state=kwargs.get('to_state', None),
                requester=kwargs.get('signature', None),
                target="self",
                timeout=-1.,
                uuid=kwargs.get('uuid', "random"))

        log.statem(f"Received an action request with this interaction: {interaction}", state=self.get_state_name())

        # Getting data
        action_name = interaction.action_name
        args = interaction.action_kwargs

        # If state is not provided, the current state is assumed
        from_state = interaction.from_state
        if from_state is None:
            # If the request arrives in the middle of a multistep action, we need to check limbo state
            from_state = self.state if self.state is not None else self.limbo_state
        if from_state not in self.transitions:
            log.statem(f"Request not accepted: not valid source state ({from_state})",
                       state=self.get_state_name())
            return False

        # If the destination state is not provided, all the possible destination from the current state are considered
        to_state = interaction.to_state
        if to_state is not None and to_state not in self.transitions[from_state]:
            log.statem(f"Request not accepted: not valid destination state ({to_state})",
                       state=self.get_state_name())
            return False
        to_states = self.transitions[from_state].keys() if to_state is None else [to_state]

        for to_state in to_states:
            action_list = self.transitions[from_state][to_state]
            for i, action in enumerate(action_list):
                if action.same_as(name=action_name, args=args):
                    log.statem(f"Requested action found in state {from_state}, adding interaction to the queue",
                               state=self.get_state_name())

                    # Action found, let's save the suggestion
                    if action.add_interaction(interaction):
                        return True
                    else:
                        return False  # If the action does not support interactions

        # If the action was not found
        log.statem("Requested action not found", state=self.get_state_name())
        return False

    def wait_for_all_actions_that_start_with(self, prefix):
        """Sets the `ready` flag to `False` for all actions whose name begins with a given prefix. This method is used
        to programmatically disable a group of actions, effectively pausing them.

        Args:
            prefix: The string prefix to match against action names.
        """
        for state, to_states in self.transitions.items():
            for to_state, action_list in to_states.items():
                for i, action in enumerate(action_list):
                    if action.name.startswith(prefix):
                        action.set_as_not_ready()

    def wait_for_all_actions_that_include_an_arg(self, arg_name):
        """Sets the `ready` flag to `False` for all actions that include a specific argument name in their signature.
        This provides another way to programmatically disable actions.

        Args:
            arg_name: The name of the argument to look for.
        """
        for state, to_states in self.transitions.items():
            for to_state, action_list in to_states.items():
                for i, action in enumerate(action_list):
                    if arg_name in action.args:
                        action.set_as_not_ready()

    def save(self, filename: str, only_if_changed: object | None = None):
        """Saves the state machine's current configuration to a JSON file. It can optionally check if the configuration
        has changed before saving to avoid redundant file writes.

        Args:
            filename: The path to the file to save to.
            only_if_changed: An optional object to compare against for changes. If a change is not detected, the file
                is not written.

        Returns:
            True if the file was written, False otherwise.
        """
        if only_if_changed is not None and os.path.exists(filename):
            try:
                existing = HybridStateMachine(actionable=only_if_changed).load(filename)
                if str(existing) == str(self):
                    return False
            except Exception as e:  # If load fails, we assume it changed
                log.error(f"Error while reloading the exising machine from {filename}, "
                          f"assuming it changed: {e}")

        with open(filename, 'w', encoding='utf-8') as file:
            file.write(str(self))
        return True

    def load(self, filename_or_hsm_as_string: str | io.TextIOWrapper):
        """Loads a state machine's configuration from a JSON file or a JSON string. It reconstructs the states,
        actions, and transitions from the serialized data. This method is critical for persistence and for loading
        pre-defined state machine models.

        Args:
            filename_or_hsm_as_string: The path to the JSON file or a JSON string representation of the state machine.

        Returns:
            The loaded `HybridStateMachine` object (self).
        """

        # Loading the whole file
        if (isinstance(filename_or_hsm_as_string, importlib.resources.abc.Traversable) or
                isinstance(filename_or_hsm_as_string, io.TextIOWrapper)):

            # Safe way to load when this file is packed in a pip package
            hsm_data = json.load(filename_or_hsm_as_string)
        else:

            # Ordinary case
            if os.path.exists(filename_or_hsm_as_string) and os.path.isfile(filename_or_hsm_as_string):
                with open(filename_or_hsm_as_string, 'r', encoding="utf-8") as file:
                    hsm_data = json.load(file)
            else:

                # Assuming it is a string
                hsm_data = json.loads(filename_or_hsm_as_string)

        # Getting state info
        self.initial_state = hsm_data['initial_state']
        self.state = hsm_data['state']
        self.prev_state = hsm_data['prev_state']
        self.limbo_state = hsm_data['limbo_state']
        self.set_role(hsm_data.get('role', None))
        self.set_welcome_message(hsm_data.get('welcome_msg', None))
        self.show_blocking_states = hsm_data.get('highlight_blocking_states_in_messages', False)
        self.show_action_completion = hsm_data.get('show_action_ticks_after_messages', False)
        self.show_action_request_info = hsm_data.get('show_action_request_after_messages', False)

        # Getting states
        self.states = {}
        if 'state_actions' in hsm_data:
            for state, state_action_list in hsm_data['state_actions'].items():
                if len(state_action_list) == 3:  # Backward compatibility
                    act_name, act_args, state_id = state_action_list
                    waiting_time = 0.
                    blocking = True
                    msg = None
                elif len(state_action_list) == 4:  # Backward compatibility
                    act_name, act_args, state_id, waiting_time_or_blocking = state_action_list
                    if isinstance(waiting_time_or_blocking, bool):
                        waiting_time = 0.
                        blocking = waiting_time_or_blocking
                    else:
                        waiting_time = waiting_time_or_blocking
                        blocking = True
                    msg = None
                elif len(state_action_list) == 5:  # Backward compatibility
                    act_name, act_args, state_id, blocking, waiting_time = state_action_list
                    msg = None
                else:
                    act_name, act_args, state_id, blocking, waiting_time, msg = state_action_list

                # Recall that state_id can be set to -1 in the original file, meaning "automatically set the state_id"
                self.add_state(state, action=act_name, args=act_args,
                               state_id=state_id if state_id >= 0 else None,
                               waiting_time=waiting_time, blocking=blocking, msg=msg)

        # Getting transitions
        self.transitions = {}
        for from_state, to_states in hsm_data['transitions'].items():
            for to_state, action_list in to_states.items():
                for action_list_tuple in action_list:
                    if len(action_list_tuple) == 4:
                        act_name, act_args, act_ready, act_id = action_list_tuple
                        msg = None
                    else:
                        act_name, act_args, act_ready, act_id, msg = action_list_tuple

                    # Recall that act_id can be set to -1 in the original file, meaning "automatically set the act_id"
                    self.add_transit(from_state, to_state,
                                     action=act_name, args=act_args, ready=act_ready,
                                     act_id=act_id if act_id >= 0 else None, msg=msg,
                                     avoid_changing_ready=True)
        return self

    def to_graphviz(self):
        """Generates a Graphviz `Digraph` object representing the state machine's structure. This method visualizes
        states as nodes and transitions as edges. It includes details such as node shapes (diamond for initial state,
        oval for others), styles (filled for blocking states), and labels for both states and transitions. The labels
        for actions include their names and arguments, formatted to wrap lines for readability.

        Returns:
            A `graphviz.Digraph` object ready for rendering.
        """
        graph = graphviz.Digraph()
        graph.attr('node', fontsize='8')
        for state, state_obj in self.states.items():
            action = state_obj.action
            if action is not None:
                s = "("
                for i, (k, v) in enumerate(action.args.items()):
                    s += str(k) + "=" + (str(v) if not isinstance(v, str) else ("'" + v + "'"))
                    if i < len(action.args) - 1:
                        s += ", "
                s += ")"
                label = action.name + s
                if len(label) > 40:
                    tokens = label.split(" ")
                    z = ""
                    i = 0
                    done = False
                    while i < len(tokens):
                        z += (" " if i > 0 else "") + tokens[i]
                        if not done and i < (len(tokens) - 1) and len(z + tokens[i + 1]) > 40:
                            z += "\n    "
                            done = True
                        i += 1
                    label = z
                suffix = "\n" + label
            else:
                suffix = ""
            if state == self.initial_state:
                graph.attr('node', shape='diamond')
            else:
                graph.attr('node', shape='oval')
            if self.states[state].blocking:
                graph.attr('node', style='filled')
            else:
                graph.attr('node', style='solid')
            graph.node(state, state + suffix, _attributes={'id': "node" + str(state_obj.id)})

        for from_state, to_states in self.transitions.items():
            for to_state, action_list in to_states.items():
                for action in action_list:
                    special_args = {}
                    if action.get_total_time() > 0:
                        special_args[next(iter(Action.SECONDS_ARG_NAMES))] = action.get_total_time()
                    if action.get_timeout() > 0:
                        special_args[next(iter(Action.TIMEOUT_ARG_NAMES))] = action.get_timeout()
                    if action.get_delay() > 0:
                        special_args[next(iter(Action.DELAY_ARG_NAMES))] = action.get_delay()
                    args = action.args | special_args
                    s = "("
                    for i, (k, v) in enumerate(args.items()):
                        s += str(k) + "=" + (str(v) if not isinstance(v, str) else ("'" + str(v) + "'"))
                        if i < len(args) - 1:
                            s += ", "
                    s += ")"
                    label = action.name + s
                    if len(label) > 40:
                        tokens = label.split(" ")
                        z = ""
                        i = 0
                        done = False
                        while i < len(tokens):
                            z += (" " if i > 0 else "") + tokens[i]
                            if not done and i < (len(tokens) - 1) and len(z + tokens[i + 1]) > 40:
                                z += "\n"
                                done = True
                            i += 1
                        label = z
                    graph.edge(from_state, to_state, label=" " + label + " ", fontsize='8',
                               style='dashed' if not action.is_ready() else 'solid',
                               _attributes={'id': "edge" + str(action.id)})
        return graph

    def save_pdf(self, filename: str):
        """Saves the state machine's Graphviz representation as a PDF file. It calls `to_graphviz()` to create the
        graph and then uses the Graphviz library's `render` method to generate the PDF.

        Args:
            filename: The path and name of the PDF file to save.

        Returns:
            True if the file was successfully saved, False otherwise.
        """
        if filename.lower().endswith(".pdf"):
            filename = filename[0:-4]

        try:
            self.to_graphviz().render(filename, format='pdf', cleanup=True)
            return True
        except Exception as e:
            log.error(f"Error while saving to PDF {e}")
            return False

    def print_actions(self, state: str | None = None):
        """Prints a list of all transitions and their associated actions from a given state. If no state is provided,
        it defaults to the current state. This method is useful for quickly inspecting the available transitions from
        a specific point in the state machine's flow.

        Args:
            state: The name of the state from which to print actions. Defaults to the current state.
        """
        state = (self.state if self.state is not None else self.limbo_state) if state is None else state
        for to_state, action_list in self.transitions[state].items():
            if action_list is None or len(action_list) == 0:
                log.user(f"{state}, no actions")
            for action in action_list:
                log.user(f"{state} --> {to_state} {action}")

    # Noinspection PyMethodMayBeStatic
    def __policy_first_requested_or_first_ready(self, actions_list: list[Action]) -> tuple[int, Interaction | None]:
        """This is the default policy for selecting which action to execute from a list of feasible actions.
        It prioritizes actions that have been explicitly requested (i.e., have pending requests) on a first-come,
        first-served basis. If no requested actions are found, it then selects the first action in the list that is
        marked as `ready`.
    
        Args:
            actions_list: A list of `Action` objects that are candidates for execution.

        Returns:
            The index of the selected action and the ActionRequest object with the requester details (object,
                arguments, time, and UUID), or -1 and the None if no action is selected.
        """
        for i, action in enumerate(actions_list):
            _list_of_interactions = action.get_list_of_interactions()
            if len(_list_of_interactions) > 0:
                _selected_action_idx = i
                _selected_interaction = _list_of_interactions.get_oldest_interaction()
                return _selected_action_idx, _selected_interaction
        for i, action in enumerate(actions_list):
            if action.is_ready(consider_interactions=False):
                _selected_action_idx = i
                _selected_interaction = action.system_interaction
                return _selected_action_idx, _selected_interaction
        _selected_action_idx = -1
        _selected_interaction = None
        return _selected_action_idx, _selected_interaction
