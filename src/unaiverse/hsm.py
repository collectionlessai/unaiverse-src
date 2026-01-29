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
import re
import json
import copy
import html
import sys
import time
import inspect
import graphviz
import importlib.resources
from collections.abc import Callable


class ActionRequest:
    def __init__(self, requester: object, action: 'Action',
                 args: dict | None = None, timestamp: float = -1., uuid: str | None = None):
        self.requester = requester
        self.action = action
        self.args = args
        self.timestamp = timestamp
        self.uuid = uuid
        self.by_insertion_order_id = -1
        self.by_requester_insertion_order_id = -1
        self.mark = None
        self.hidden = False

    def set_mark(self, mark: object):
        self.mark = mark

    def get_mark(self):
        return self.mark

    def get_order_id(self, by_requester: bool = False) -> int:
        if not by_requester:
            return self.by_insertion_order_id
        else:
            return self.by_requester_insertion_order_id

    def is_valid(self):
        return self.by_insertion_order_id >= 0 and self.by_requester_insertion_order_id >= 0

    def has_dummy_requester(self):
        return self.requester is None

    def to_str(self):
        return json.dumps([self.requester, self.args, self.timestamp, self.uuid])

    def alter_arg(self, arg_name: str, arg_value: object):
        if arg_name in self.args:
            self.args[arg_name] = arg_value
            return True
        else:
            return False

    def set_arg(self, arg_name: str, arg_value: object):
        self.args[arg_name] = arg_value

    def get_arg(self, arg_name):
        return self.args[arg_name] if arg_name in self.args else None

    def __str__(self):
        """Provides a string representation of the `ActionRequest` instance.

        Returns:
            A string containing a formatted summary of the instance.
        """
        return (f"(action={self.action.name}, args={self.args}, "
                f"requester={self.requester}, timestamp={self.timestamp}, uuid={self.uuid})")


class ActionRequestList:
    def __init__(self, max_per_requester: int = -1):
        self.by_insertion_order = []
        self.by_requester_and_by_insertion_order = {}
        self.max_per_requester = max_per_requester
        self.by_insertion_order_entering_time = []

    def add(self, req: ActionRequest):

        # Updating by-requester index
        if req.requester not in self.by_requester_and_by_insertion_order:
            self.by_requester_and_by_insertion_order[req.requester] = []

        # Searching for UUID = None, if already there - do not accumulate multiple requests with UUID None
        if req.uuid is None:
            existing_request_same_uuid = self.get_request_by_uuid(req.requester, req.uuid)
            if existing_request_same_uuid:
                return

        if 0 < self.max_per_requester <= len(self.by_requester_and_by_insertion_order[req.requester]):
            self.remove(self.get_oldest_request(req.requester))
        by_requester_insertion_order_id = len(self.by_requester_and_by_insertion_order[req.requester])
        self.by_requester_and_by_insertion_order[req.requester].append(req)

        # Updating direct global index
        insertion_order_id = len(self.by_insertion_order)
        self.by_insertion_order.append(req)

        # Updating reverse indices
        req.by_insertion_order_id = insertion_order_id
        req.by_requester_insertion_order_id = by_requester_insertion_order_id

        # Saving joining time
        self.by_insertion_order_entering_time.append(time.perf_counter())

    def remove(self, req: ActionRequest):
        if req.is_valid():
            for i in range(req.by_insertion_order_id + 1, len(self.by_insertion_order)):
                self.by_insertion_order[i].by_insertion_order_id -= 1
            del self.by_insertion_order[req.by_insertion_order_id]
            del self.by_insertion_order_entering_time[req.by_insertion_order_id]

            d = self.by_requester_and_by_insertion_order[req.requester]
            for i in range(req.by_requester_insertion_order_id + 1, len(d)):
                d[i].by_requester_insertion_order_id -= 1
            del d[req.by_requester_insertion_order_id]
            if len(d) == 0:
                del self.by_requester_and_by_insertion_order[req.requester]
            req.by_insertion_order_id = -1
            req.by_requester_insertion_order_id = -1

    def remove_due_to_timeout(self, timeout_secs: float):
        to_remove = []
        for i, req in enumerate(self.by_insertion_order):
            if (time.perf_counter() - self.by_insertion_order_entering_time[i]) >= timeout_secs:
                to_remove.append(req)
        for req in to_remove:
            self.remove(req)

    def move_request_to_back(self, req: ActionRequest):
        if req.is_valid():
            entering_time = self.by_insertion_order_entering_time[req.by_insertion_order_id]
            self.remove(req)
            self.add(req)
            self.by_insertion_order_entering_time[req.by_insertion_order_id] = entering_time

    def move_requester_to_back(self, requester: object):
        requests = self.get_requests(requester)
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

    def get_request(self, req_order_id: int, requester: object | None = None):
        if req_order_id < 0 and req_order_id != -1:
            return None
        if requester is None:
            return self.by_insertion_order[req_order_id] if req_order_id < len(self.by_insertion_order) else None
        else:
            if requester not in self.by_requester_and_by_insertion_order:
                return None
            return self.by_requester_and_by_insertion_order[requester][req_order_id] \
                if req_order_id < len(self.by_requester_and_by_insertion_order[requester]) else None

    def get_oldest_request(self, requester: object | None = None):
        return self.get_request(0, requester)

    def get_most_recent_request(self, requester: object | None = None):
        return self.get_request(-1, requester)

    def get_request_by_uuid(self, requester: object, uuid: str | None) -> None | ActionRequest:
        requests = self.get_requests(requester)
        if requests is None or len(requests) == 0:
            return None

        for req in requests:
            if req.uuid == uuid:
                return req

    def keep_only_the_most_recent_request(self):
        req = self.get_most_recent_request()
        entering_time = self.by_insertion_order_entering_time[req.by_insertion_order_id]
        self.clear()
        self.add(req)
        self.by_insertion_order_entering_time[req.by_insertion_order_id] = entering_time

    def get_requests(self, requester: object | None = None, to_str: bool = False):
        if requester is None:
            if not to_str:
                return self.by_insertion_order
            else:
                return json.dumps([req.to_str() for req in self.by_insertion_order])
        else:
            if requester in self.by_requester_and_by_insertion_order:
                if not to_str:
                    return self.by_requester_and_by_insertion_order[requester]
                else:
                    reqs = self.by_requester_and_by_insertion_order[requester]
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

    def is_requester_known(self, requester: object):
        return requester in self.by_requester_and_by_insertion_order

    def __len__(self):
        return len(self.by_insertion_order)

    def __iter__(self):
        return iter(self.by_insertion_order)

    def __str__(self):
        """Provides a string representation of the `ActionRequestList` instance.

        Returns:
            A string containing a formatted summary of the instance.
        """
        return "{" + ", ".join([str(r) for r in self.by_insertion_order]) + "}"


class Action:

    # Candidate argument names (when calling an action) that tells that such an action is multi-steps
    STEPS_ARG_NAMES = {'steps', 'samples'}
    SECONDS_ARG_NAMES = {'time'}
    TIMEOUT_ARG_NAMES = {'timeout'}
    DELAY_ARG_NAMES = {'delay'}
    COMPLETED_NAMES = {'_completed'}
    REQUESTER_ARG_NAMES = {'_requester'}
    REQUEST_TIME_NAMES = {'_request_time'}
    REQUEST_UUID_NAMES = {'_request_uuid'}
    NOT_READY_PREFIXES = ('get_', 'got_', 'do_', 'done_')
    KNOWN_SINGLE_STEP_ACTION_PREFIXES = ('ask_',)

    # Completion reasons
    MAX_STEPS_REACHED = 0  # Single-step actions always complete due to this reason
    MAX_TIME_REACHED = 1
    MAX_TIMEOUT_DURING_ATTEMPTS_REACHED = 2

    def __init__(self, name: str, args: dict, actionable: object,
                 idx: int = -1,
                 ready: bool = True,
                 msg: str | None = None,
                 avoid_changing_ready: bool = False, out_fcn: Callable = print):
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
            out_fcn: Output print function.
        """
        # Basic properties
        self.name = name  # Name of the action (name of the corresponding method)
        self.args = args  # Dictionary of arguments to pass to the action
        self.actionable = actionable  # Object on which the method whose name is self.name is searched
        self.ready = ready  # Boolean flag telling if the action can considered ready to be executed
        self.requests = ActionRequestList()  # List of requests to make this action ready to be executed (customizable)
        self.id = idx  # Unique ID of the action (-1 if not needed)
        self.msg = msg  # Human-readable message associated to this instance of action
        self.out_fcn = out_fcn

        # Fix UNICODE chars
        if self.msg is not None:
            self.msg = html.unescape(self.msg)

        # Reference elements
        self.args_with_wildcards = copy.deepcopy(self.args)  # Backup of the originally provided arguments
        self.msg_with_wildcards = self.msg
        self.__fcn = self.__action_name_to_callable(name)  # The real method to be called
        self.__sig = inspect.signature(self.__fcn)  # Signature of the method for argument inspection

        # Parameter names and default values
        self.param_list = []  # Full list of the parameters that the action supports
        self.param_to_default_value = {}  # From parameter to its default value, if any
        self.__get_action_params()  # This will fill the two attributes above
        self.__check_if_args_exist(self.args, exception=True)  # Checking arguments

        # Argument values replaced by wildcards (commonly assumed to be in the format <value>)
        self.wildcards = {}  # Value-to-value (es: <playlist> to this:and:this)

        # Number of steps of this function
        self.__step = -1  # Default initial step index (remark: "step INDEX", so when it is 0 it means a step was done)
        self.__total_steps = 1  # Total step of an action (a multi-steps action has != 1 steps)
        self.__guess_total_steps(self.get_actual_params({}))  # This will "guess" the value of self.__total_steps

        # Time-based metrics
        self.__starting_time = 0
        self.__total_time = 0  # A total time <= 0 means "no total time at all"
        self.__guess_total_time(self.get_actual_params({}))  # This will "guess" the value of self.__total_time

        # Time-based metrics
        self.__timeout_starting_time = 0
        self.__timeout = 0  # A timeout <= 0 means "no total time at all"
        self.__guess_timeout(self.get_actual_params({}))  # This will "guess" the value of self.__timeout

        # Time-based metrics
        self.__delay = 0
        self.__guess_delay(self.get_actual_params({}))  # This will "guess" the value of self.__delay

        # Fixing (if no options are specified, assuming a single-step action)
        if self.__total_steps <= 0 and self.__total_time <= 0:
            self.__total_steps = 1

        # Fixing (forcing NOT-ready on some actions)
        if not avoid_changing_ready:
            for prefix in Action.NOT_READY_PREFIXES:
                if self.name.startswith(prefix):
                    self.ready = False

        self.__has_completion_step = False
        for completed_name in Action.COMPLETED_NAMES:
            if completed_name in self.param_list:
                self.__has_completion_step = True
                break

        # Status
        self.__cannot_be_run_anymore = False

    async def __call__(self, request: ActionRequest | None = None):
        """Executes the action's associated method. This is the main entry point for running an action. It handles
        multistep logic by updating the step counter and checking for completion based on steps, time, or timeout.
        It also injects dynamic arguments like the `requester`, `request_time`, and `request_uuid` into the method's
        arguments before execution. If the action is a multistep action and has a completion step, it handles that
        callback as well (async).

        Args:
            request: The ActionRequest object or None, if the action was not requested by other agents.

        Returns:
            A boolean indicating whether the action was executed successfully.
        """
        request_args = request.args if request is not None else {}
        self.__check_if_args_exist(request_args, exception=True)
        actual_args = self.get_actual_params(request_args)  # Getting the actual values of the arguments

        if self.msg is not None:
            self.out_fcn(self.msg, request, self.requests)

        if actual_args is not None:

            # Getting the values for the main involved measures: total steps, total time, timeout
            self.__guess_total_steps(actual_args)
            self.__guess_total_time(actual_args)
            self.__guess_timeout(actual_args)

            # Storing the time index that is related to the timeout (do this before calling self.is_timed_out())
            if self.__timeout_starting_time <= 0:
                self.__timeout_starting_time = time.perf_counter()

            # Storing the starting time (do this before calling self.was_last_step_done())
            if self.__starting_time <= 0:
                self.__starting_time = time.perf_counter()

            # Setting up the flag that tells if the action reached a point in which it cannot be run anymore
            self.__cannot_be_run_anymore = self.is_timed_out() or self.was_last_step_done()

            if HybridStateMachine.DEBUG:
                if self.__cannot_be_run_anymore:
                    print(f"[DEBUG HSM] Cannot-be-run-anymore set to True, "
                          f"due to self.is_timed_out()={self.is_timed_out()} or "
                          f"self.was_last_step_done()={self.was_last_step_done()}")

            if self.__cannot_be_run_anymore and not self.is_multi_steps():
                return False

            # Setting up the information on whether a multistep action is completed
            # (for example, to tell that now it is time for a callback)
            calling_completion_step = False
            for completed_name in Action.COMPLETED_NAMES:
                if completed_name in actual_args:
                    calling_completion_step = self.__cannot_be_run_anymore and self.get_step() >= 0
                    actual_args[completed_name] = calling_completion_step
                    break

            # We are done, no need to call the action again
            if self.__cannot_be_run_anymore and not calling_completion_step:
                return True

            # Setting up the requester
            for req_arg_name in Action.REQUESTER_ARG_NAMES:
                if req_arg_name in actual_args:
                    actual_args[req_arg_name] = request.requester if request is not None else None
                    break

            # Setting up the request time
            for req_time_name in Action.REQUEST_TIME_NAMES:
                if req_time_name in actual_args:
                    actual_args[req_time_name] = request.timestamp if request is not None else -1.
                    break

            # Setting up the request uuid
            for req_uuid_name in Action.REQUEST_UUID_NAMES:
                if req_uuid_name in actual_args:
                    actual_args[req_uuid_name] = request.uuid if request is not None else None
                    break

            # Fixing (if no options are specified, assuming a single-step action)
            if self.__total_steps == 0 and self.__total_time == 0:
                self.__total_steps = 1

            # Fixing the single step case: in this case, time does not matter, so we force it to zero
            if self.__total_steps == 1:
                self.__total_time = 0

            # Increasing the step index
            self.__step += 1  # This is a step index, so self.__step == 0 means "done 1 step"

            if HybridStateMachine.DEBUG:
                if request is None or request.requester is None:
                    requester_str = "nobody"
                else:
                    requester_str = request.requester
                print(f"[DEBUG HSM] Calling function {self.name} (multi_steps: {self.is_multi_steps()}), "
                      f"requested by {requester_str}, with actual params: {actual_args}")

            # Calling the method here
            ret = await self.__fcn(**actual_args)

            if HybridStateMachine.DEBUG:
                print(f"[DEBUG HSM] Returned: {ret}")

            # If action failed, be sure to reduce the step counter (only if it was actually incremented)
            if not ret:
                self.__step -= 1

            # If it went OK, we reset the time counter that is related to the timeout
            else:
                self.__timeout_starting_time = 0

            return ret
        else:
            if HybridStateMachine.DEBUG:
                print(f"[DEBUG HSM] Tried and failed (missing actual param): {self}")
            return False

    def __str__(self):
        """Provides a string representation of the `Action` instance.

        Returns:
            A string containing a formatted summary of the instance.
        """
        return (f"[Action: {self.name}] id: {self.id}, args: {self.args}, param_list: {self.param_list}, "
                f"total_steps: {self.__total_steps}, "
                f"total_time: {self.__total_time}, timeout: {self.__timeout}, "
                f"ready: {self.ready}, requests: {str(self.requests)}, msg: {str(self.msg)}]")

    def set_as_ready(self):
        """Sets the action's ready flag to `True`, indicating it can now be executed."""
        self.ready = True

    def set_as_not_ready(self):
        """Sets the action's ready flag to `False`, preventing it from being executed."""
        self.ready = False

    def set_msg(self, msg):
        """Sets the message associated to this action."""

        if msg is not None:
            self.msg = html.unescape(msg)
            self.msg_with_wildcards = self.msg
        else:
            self.msg = None
            self.msg_with_wildcards = None

    def is_ready(self, consider_requests: bool = True):
        """Checks if the action is ready to be executed. It returns `True` if the `ready` flag is set or if there are
        any pending requests.

        Args:
            consider_requests: A boolean flag to include pending requests in the readiness check.

        Returns:
            A boolean indicating the action's readiness.
        """
        return self.ready or (consider_requests and len(self.requests) > 0)

    def was_last_step_done(self):
        """Determines if the action has reached its completion criteria, either by reaching the total number of steps
        or by exceeding the maximum allowed execution time.

        Returns:
            True if the action is completed, False otherwise.
        """
        return ((self.__total_steps > 0 and self.__step == self.__total_steps - 1) or
                (self.__total_time > 0 and ((time.perf_counter() - self.__starting_time) >= self.__total_time)))

    def cannot_be_run_anymore(self):
        """Checks if the action has reached a state where it cannot be executed further, for instance, due to
        completion or a timeout.

        Returns:
            A boolean indicating if the action can no longer be run.
        """
        return self.__cannot_be_run_anymore

    def has_completion_step(self):
        """Checks if the action is designed to have a completion step, which is a final execution pass after the main
        action logic has finished.

        Returns:
            A boolean indicating the presence of a completion step.
        """
        return self.__has_completion_step

    def is_multi_steps(self):
        """Determines if the action is configured to be a multistep action (i.e., not a single-step action).

        Returns:
            A boolean indicating if the action is multistep.
        """
        return self.__total_steps != 1

    def has_a_timeout(self):
        """Checks if a timeout has been configured for the action.

        Returns:
            A boolean indicating if a timeout is set.
        """
        return self.__timeout > 0

    def is_delayed(self, starting_time: float):
        """Checks if the action is currently in a delayed state and cannot be executed yet, based on a defined delay
        period.

        Args:
            starting_time: The time the delay period began.

        Returns:
            True if the action is delayed, False otherwise.
        """
        return self.__delay > 0 and (time.perf_counter() - starting_time) <= self.__delay

    def is_timed_out(self):
        """Checks if the action has exceeded its configured timeout period since the last successful execution attempt.

        Returns:
            True if the action has timed out, False otherwise.
        """
        if self.__timeout <= 0 or self.__timeout_starting_time <= 0:
            return False
        else:
            if HybridStateMachine.DEBUG:
                print(f"[DEBUG HSM] checking if {self.name} is timed out:"
                      f" {(time.perf_counter() - self.__timeout_starting_time)} >= {self.__timeout}")
            if (time.perf_counter() - self.__timeout_starting_time) >= self.__timeout:
                if HybridStateMachine.DEBUG:
                    print(f"[DEBUG HSM] Timeout for {self.name}!")
                return True
            else:
                return False

    def to_list(self, minimal=False):
        """Converts the action's properties into a list for easy serialization. It can generate either a full or a
        minimal representation.

        Args:
            minimal: A boolean flag to return a minimal list representation.

        Returns:
            A list containing the action's properties.
        """
        if not minimal:
            if self.msg is not None:
                msg = self.msg.encode("ascii", "xmlcharrefreplace").decode("ascii")
            else:
                msg = None
            return [self.name, self.args, self.ready, self.id] + ([msg] if msg is not None else [])
        else:
            return [self.name, self.args]

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
        args_to_exclude = Action.SECONDS_ARG_NAMES | Action.TIMEOUT_ARG_NAMES | Action.DELAY_ARG_NAMES
        return (name == self.name and
                self.__check_if_args_exist(args) and
                all(k in args_to_exclude or k not in self.args or self.args[k] == v for k, v in args.items()))

    def __check_if_args_exist(self, args: dict, exception: bool = False):
        """A private helper method to validate that all provided arguments for an action exist in the action's
        parameter list. It can either raise a `ValueError` or return a boolean.

        Args:
            args: The dictionary of arguments to check.
            exception: If `True`, a `ValueError` is raised on failure.

        Returns:
            True if all arguments are valid, False otherwise (if `exception` is `False`).
        """
        if args is not None:
            for param_name in args.keys():
                if param_name not in self.param_list:
                    if exception:
                        raise ValueError(f"Unknown parameter {param_name} for action {self.name}")
                    else:
                        return False
        return True

    def set_wildcards(self, wildcards: dict[str, str | float | int] | None):
        """Replaces wildcard values in the action's arguments with actual values. This method is used to dynamically
        configure actions with context-specific data.

        Args:
            wildcards: A dictionary mapping wildcard placeholders to their concrete values.
        """
        self.wildcards = wildcards if wildcards is not None else {}
        self.__replace_wildcard_values()

    def add_request(self, signature: object, args: dict, timestamp: float, uuid: str):
        """Adds a new request to the action's internal list. This is used to track pending requests that might make the
        action ready to be executed.

        Args:
            signature: The object making the request.
            args: The arguments associated with the request.
            timestamp: The time the request was made.
            uuid: A unique ID for the request.
        """
        req = ActionRequest(signature, self, args, timestamp, uuid)
        self.requests.add(req)

    def clear_requests(self, signature: object | None = None):
        """Clears all pending requests from the action's list."""
        if signature is None:
            self.requests = ActionRequestList()
        else:
            if self.requests.is_requester_known(signature):
                requests = self.requests.get_requests(signature)
                for req in requests:
                    self.requests.remove(req)

    def clear_request(self, signature: object, req_id: int):
        req = self.requests.get_request(req_id, signature)
        if req is not None:
            self.requests.remove(req)

    def get_list_of_requests(self) -> ActionRequestList:
        """Retrieves the list of pending requests.

        Returns:
            The list of pending requests, i.e., an object of type ActionRequestList.
        """
        return self.requests

    def reset_step(self):
        """Resets the action's state, including the step counter and timing metrics, allowing it to be re-run from the
        beginning.
        """
        self.__step = -1
        self.__starting_time = 0.
        self.__timeout_starting_time = 0.
        self.__cannot_be_run_anymore = False

    def get_step(self):
        """Retrieves the current step index of the multistep action.

        Returns:
            An integer representing the current step index.
        """
        return self.__step

    def get_total_steps(self):
        """Retrieves the total number of steps configured for the action.
        
        Returns:
            An integer representing the total steps.
        """
        return self.__total_steps

    def get_starting_time(self):
        """Retrieves the timestamp when the action's current execution started.

        Returns:
            A float representing the starting time.
        """
        return self.__starting_time

    def get_total_time(self):
        """Retrieves the total time configured for the action's execution.

        Returns:
            A float representing the total time.
        """
        return self.__total_time

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
                if HybridStateMachine.DEBUG:
                    print(f"[DEBUG HSM] Getting actual params for {self.name}; missing param: {param_name}")
                return None
        return actual_params

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

    def __replace_wildcard_values(self):
        """A private helper method that replaces placeholder values (wildcards) in the action's arguments with their
        actual, concrete values. It handles both single-value and list-based wildcards.
        """
        if self.args_with_wildcards is None:
            self.args_with_wildcards = copy.deepcopy(self.args)  # Backup before applying wildcards (first time only)
        else:
            self.args = copy.deepcopy(self.args_with_wildcards)  # Restore a backup before applying wildcards

        if self.msg_with_wildcards is None:
            self.msg_with_wildcards = self.msg
        else:
            self.msg = self.msg_with_wildcards

        for wildcard_from, wildcard_to in self.wildcards.items():
            for k, v in self.args.items():
                if not isinstance(wildcard_to, str):
                    if wildcard_from == v:
                        self.args[k] = wildcard_to
                else:
                    if isinstance(v, list):
                        for i, vv in enumerate(v):
                            if isinstance(vv, str) and wildcard_from in vv:
                                v[i] = vv.replace(wildcard_from, wildcard_to)
                    elif isinstance(v, str):
                        if wildcard_from in v:
                            self.args[k] = v.replace(wildcard_from, wildcard_to)

            if self.msg is not None:
                self.msg = self.msg.replace(wildcard_from, str(wildcard_to))

    def __guess_total_steps(self, args):
        """A private helper method that attempts to determine the total number of steps for a multistep action by
        looking for specific keyword arguments like 'steps' or 'samples'.

        Args:
            args: The dictionary of arguments to inspect.
        """
        for prefix in Action.KNOWN_SINGLE_STEP_ACTION_PREFIXES:
            if self.name.startswith(prefix):
                return
        for arg_name in Action.STEPS_ARG_NAMES:
            if arg_name in args:
                if isinstance(args[arg_name], int):
                    self.__total_steps = max(float(args[arg_name]), 1.)
                break

    def __guess_total_time(self, args):
        """A private helper method that attempts to determine the total execution time for an action by looking for a
        'time' or 'seconds' argument.

        Args:
            args: The dictionary of arguments to inspect.
        """
        for prefix in Action.KNOWN_SINGLE_STEP_ACTION_PREFIXES:
            if self.name.startswith(prefix):
                return
        for arg_name in Action.SECONDS_ARG_NAMES:
            if arg_name in args:
                if isinstance(args[arg_name], int) or isinstance(args[arg_name], float):
                    try:
                        self.__total_time = max(float(args[arg_name]), 0.)
                    except ValueError:
                        self.__total_time = -1.
                        pass
                break

    def __guess_timeout(self, args):
        """A private helper method that attempts to determine the timeout duration for an action by looking for a
        'timeout' argument.

        Args:
            args: The dictionary of arguments to inspect.
        """
        for prefix in Action.KNOWN_SINGLE_STEP_ACTION_PREFIXES:
            if self.name.startswith(prefix):
                return
        for arg_name in Action.TIMEOUT_ARG_NAMES:
            if arg_name in args:
                try:
                    self.__timeout = max(float(args[arg_name]), 0.)
                except ValueError:
                    self.__timeout = -1.
                    pass
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
                except ValueError:
                    self.__delay = -1.
                    pass
                break


class State:

    def __init__(self, name: str, idx: int = -1, action: Action | None = None, waiting_time: float = 0.,
                 blocking: bool = True, msg: str | None = None, out_fcn: Callable = print):
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
            out_fcn: Output print function.
        """
        self.name = name  # Name of the state (must be unique)
        self.action = action  # Inner state action (it can be None)
        self.id = idx  # Unique ID of the state (-1 if not needed)
        self.waiting_time = waiting_time  # Number of seconds to wait in the current state before acting
        self.starting_time = 0.
        self.blocking = blocking
        self.msg = msg  # Human-readable message associated to this instance of state
        self.out_fcn = out_fcn

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
        if self.starting_time <= 0.:
            self.starting_time = time.perf_counter()

        if self.msg is not None:
            self.out_fcn(self.msg)

        if self.action is not None:
            if HybridStateMachine.DEBUG:
                print("[DEBUG HSM] Running action on state: " + self.action.name)
            self.action.reset_step()
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
                if HybridStateMachine.DEBUG:
                    print(f"[DEBUG HSM] Time passing: {(time.perf_counter() - self.starting_time)} seconds")
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
            self.action.reset_step()

    def set_blocking(self, blocking: bool):
        """Sets the blocking status of the state. A blocking state will prevent the state machine from transitioning to
        the next state until the action is fully completed.

        Args:
            blocking: A boolean value to set the blocking status.
        """
        self.blocking = blocking

    def set_wildcards(self, wildcards: dict[str, str | float | int] | None):
        """Replaces wildcard values in the state messages. This method is used to dynamically
        configure state messages with context-specific data.

        Args:
            wildcards: A dictionary mapping wildcard placeholders to their concrete values.
        """
        self.wildcards = wildcards if wildcards is not None else {}
        self.__replace_wildcard_values()

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
    DEBUG = True
    DEFAULT_WILDCARDS = {'<world>': '<world>', '<agent>': '<agent>', '<partner>': '<partner>', '<role>': '<role>'}
    REQUEST_VALIDITY_TIMEOUT = 5 * 60.0  # Seconds
    ACTION_TICKS_PER_STATUS = ["   ✅ ", "   🔄 ", "   ❌ "]  # Keep the final spaces

    def __init__(self, actionable: object, wildcards: dict[str, str | float | int] | None = None,
                 request_signature_checker: Callable[[object], bool] | None = None,
                 policy: Callable[[list[Action]], tuple[int, ActionRequest | None]] | None = None):
        """Initializes a `HybridStateMachine` object, which orchestrates states and transitions. It manages a set of
        states and actions, and handles the logic for transitions between states based on conditions and a defined
        policy. It sets up initial and current states, wildcards for dynamic arguments, and references to an
        `actionable` object whose methods are the actions to be called. It also includes debug and output settings.

        Args:
            actionable: The object on which actions (methods) are to be executed.
            wildcards: A dictionary of key-value pairs for dynamic argument substitution.
            request_signature_checker: An optional callable to validate incoming action requests.
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
        self.actionable: object = actionable  # The object on whose methods are actions that the machine calls
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

        # Actions can be requested from the "outside": each request if checked by this function, if any
        self.request_signature_checker: Callable[[object], bool] | None = request_signature_checker

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
        self.__last_printed_msg = None
        self.__last_printed_tick = None
        self.__debug_messages_active = False
        self.print_stream = sys.stdout
        self.print_start = ""
        self.print_ending = "\n"
        self.print_fcn = print
        self.print_fcn_supports_html = False

        def wrapped_out_fcn(msg: str, is_state: bool,
                            action_request: ActionRequest | None, action_requests: ActionRequestList | None = None):
            if msg is not None:
                must_print = False
                if msg in HybridStateMachine.ACTION_TICKS_PER_STATUS:
                    if msg != self.__last_printed_tick:
                        self.__last_printed_tick = msg
                        must_print = True
                elif msg != self.__last_printed_msg:
                    self.__last_printed_msg = msg
                    self.__last_printed_tick = None
                    must_print = True

                if must_print:
                    if not self.print_fcn_supports_html:

                        # Handle a bit of HTML (<br/>, <a href=...>...</a>, <strong>...</strong>)
                        msg = (msg.replace('<br/>', '\n').replace('<strong>', '')
                               .replace('</strong>', ''))
                        pattern = r'<a\s+href=[\'"](.*?)[\'"][^>]*>(.*?)</a>'
                        msg = re.sub(pattern, r'\2 (\1)', msg)

                    if is_state and self.show_blocking_states:
                        if self.states[self.state].blocking:
                            msg += " 🔴"
                        else:
                            msg += " 🟢"

                    if not is_state and self.show_action_request_info and action_request is not None:
                        msg += (f" (requester: {action_request.requester}, uuid: {action_request.uuid}, "
                                f"#requests: {len(action_requests)})")

                    self.print_fcn(self.print_start + msg, end=self.print_ending, file=self.print_stream)

        def wrapped_state_out_fcn(msg: str):
            wrapped_out_fcn(msg, True, None, None)

        def wrapped_action_out_fcn(msg: str,
                                   request: ActionRequest | None = None, requests: ActionRequestList | None = None):
            wrapped_out_fcn(msg, False, request, requests)

        self.state_out_fcn = wrapped_state_out_fcn
        self.action_out_fcn = wrapped_action_out_fcn

    def set_print_fcn(self, print_fcn, supports_html):
        self.print_fcn = print_fcn
        self.print_fcn_supports_html = supports_html

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
        self.actionable = obj

        for state_obj in self.states.values():
            if state_obj.action is not None:
                state_obj.action.actionable = obj

    def set_policy(self, policy_fcn: Callable[[list[Action]], tuple[int, ActionRequest | None]]):
        """Sets the policy to be used in selecting what action to perform in the current state.

        Args:
            policy_fcn: A function that takes a list of `Action` objects that are candidates for execution, and returns
                the index of the selected action and an ActionRequest object with the action-requester details (object,
                arguments, time, and UUID), or -1 and None if no action is selected.
        """
        self.policy = policy_fcn

    def set_policy_filter(self, filter_fcn: Callable[
                              [int, ActionRequest | None, list[Action], dict], tuple[int, ActionRequest | None]],
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

    def set_wildcards(self, wildcards: dict[str, str | float | int] | None):
        """Sets the dictionary of wildcards that are used to dynamically replace placeholder values in action
        arguments. It updates all actions with the new wildcard dictionary.

        Args:
            wildcards: A dictionary containing wildcard key-value pairs.
        """
        self.wildcards = wildcards if wildcards is not None else {}
        for action in self.__id_to_action:
            action.set_wildcards(self.wildcards)
        for state in self.__id_to_state:
            state.set_wildcards(self.wildcards)
        if self.welcome_msg is not None:
            if self.welcome_msg_with_wildcards is None:
                self.welcome_msg_with_wildcards = self.welcome_msg
            else:
                self.welcome_msg = self.welcome_msg_with_wildcards
            for wildcard_from, wildcard_to in self.wildcards.items():
                self.welcome_msg = self.welcome_msg.replace(wildcard_from, str(wildcard_to))

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

    def add_wildcards(self, wildcards: dict[str, str | float | int | list[str]]):
        """Adds new key-value pairs to the existing wildcard dictionary. It also triggers an update to all actions with
        the new combined dictionary.

        Args:
            wildcards: A dictionary of new wildcards to add.
        """
        self.wildcards.update(wildcards)
        self.set_wildcards(self.wildcards)

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

    def get_action_step(self):
        """Retrieves the current step index of the action being executed. This is particularly useful for tracking the
        progress of multistep actions.

        Returns:
            An integer representing the current step, or -1 if no action is running.
        """
        return self.__action.get_step() if self.__action is not None else -1

    def is_busy_acting(self):
        """Checks if the state machine is currently executing an action. This is determined by checking if the action
        step index is greater than or equal to 0.

        Returns:
            True if an action is running, False otherwise.
        """
        return self.get_action_step() >= 0

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
                         msg=msg_action, out_fcn=self.action_out_fcn)
            act.set_wildcards(self.wildcards)
            self.__id_to_action.append(act)
        if waiting_time is None:
            waiting_time = sta_obj.waiting_time if sta_obj is not None else 0.  # Default waiting time
        if blocking is None:
            blocking = sta_obj.blocking if sta_obj is not None else True  # Default blocking
        if msg is None:
            msg = sta_obj.msg_with_wildcards if sta_obj is not None else None

        sta = State(name=state, idx=state_id, action=act, waiting_time=waiting_time, blocking=blocking, msg=msg,
                    out_fcn=self.state_out_fcn)
        sta.set_wildcards(self.wildcards)
        if state not in self.states:
            self.__id_to_state.append(sta)
        else:
            self.__id_to_state[state_id] = sta
        self.states[state] = sta

        if len(self.__id_to_state) == 1 and self.state is None:
            self.set_state(sta.name)

    def get_state_name(self):
        """Retrieves the name of the current state of the state machine.

        Returns:
            A string with the state's name, or `None` if no state is set.
        """

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
            act.reset_step()
        for s in self.__id_to_state:
            if s.action is not None:
                s.action.reset_step()

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
                self.__action.reset_step()
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
                            avoid_changing_ready=avoid_changing_ready, out_fcn=self.action_out_fcn)
        self.transitions[from_state][to_state].append(new_action)
        self.__id_to_action.append(new_action)

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
                           args=copy.deepcopy(_state.action.args_with_wildcards) if _state.action is not None else None,
                           state_id=None,
                           blocking=_state.blocking,
                           msg=_state.msg_with_wildcards)

        # Copy all the transitions of the HSM
        for _from_state, _to_states in hsm.transitions.items():
            for _to_state, _action_list in _to_states.items():
                for _action in _action_list:
                    self.add_transit(from_state=_from_state, to_state=_to_state, action=_action.name,
                                     args=copy.deepcopy(_action.args_with_wildcards), ready=_action.ready,
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

    async def act_transitions(self, requested_only: bool = False):
        """This is the core execution loop for transitions. It finds all feasible actions from the current state and,
        using a policy, selects and executes one. It handles single-step and multistep actions, managing state changes,
        timeouts, and failed executions. It returns an integer status code indicating the outcome (e.g., transition
        done, try again, move to next action) (async).

        Args:
            requested_only: A boolean to consider only actions that have pending requests.

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
            attempts_to_serve_a_request_list = []

            for to_state, action_list in self.transitions[self.state].items():
                for i, action in enumerate(action_list):

                    # Pruning too old requests
                    action.requests.remove_due_to_timeout(HybridStateMachine.REQUEST_VALIDITY_TIMEOUT)

                    if (action.is_ready() and (not requested_only or len(action.requests) > 0) and
                            not action.is_delayed(self.states[self.state].starting_time)):
                        actions_list.append(action)
                        to_state_list.append(to_state)
                        attempts_to_serve_a_request_list.append(0)

            if len(actions_list) > 0:
                self.__cur_feasible_actions_status = {
                    'actions_list': actions_list,
                    'to_state_list': to_state_list,
                    'selected_idx': 0,
                    'selected_request': None,
                    'attempts_to_serve_a_request_list': attempts_to_serve_a_request_list
                }
        else:

            # Reloading the already computed set of actions, wait flags, etc. (when in the middle of an action)
            actions_list = self.__cur_feasible_actions_status['actions_list']
            to_state_list = self.__cur_feasible_actions_status['to_state_list']
            attempts_to_serve_a_request_list = self.__cur_feasible_actions_status['attempts_to_serve_a_request_list']

            # Pruning too old requests
            idx_to_remove = []
            for i, action in enumerate(actions_list):
                if len(action.requests) > 0:
                    action.requests.remove_due_to_timeout(HybridStateMachine.REQUEST_VALIDITY_TIMEOUT)
                    if len(action.requests) == 0:
                        idx_to_remove.append(i)
            for i in idx_to_remove:
                del actions_list[i]
                del to_state_list[i]
                del attempts_to_serve_a_request_list[i]

        # Using the selected policy to decide what action to apply
        while len(actions_list) > 0:

            # It there was an already selected action (for example a multistep action), then continue with it,
            # otherwise, select a new one following a certain policy (actually, first-come first-served)
            if self.__action is None:

                if HybridStateMachine.DEBUG:
                    print(f"[DEBUG HSM] Considering the following list of actions: "
                          f"{[a.__str__() for a in actions_list]}")

                # Naive policy: take the first action that is ready
                _idx, _request = self.policy(actions_list)

                if _idx < 0:
                    if HybridStateMachine.DEBUG:
                        print(f"[DEBUG HSM, {self.role}] Policy selected no actions")

                    # No actions were applied
                    self.__cur_feasible_actions_status = None
                    self.__state_changed = False
                    return -1  # Early stop
                else:
                    # if HybridStateMachine.DEBUG:  # TODO re-move the following prints inside this IF
                    if _request is not None:
                        print(f"[DEBUG HSM, {self.role}] Policy selected {actions_list[_idx].__str__()}"
                              f" whose request is {_request.__str__()}")
                        print(f"[DEBUG HSM, {self.role}] (request in to_str format: {_request.to_str()})")
                    else:
                        print(f"[DEBUG HSM, {self.role}] Policy selected {actions_list[_idx].__str__()}")

                # Revisiting decisions due to the policy filter
                if self.policy_filter is not None:
                    try:
                        _idx_f, _request_f = self.policy_filter(_idx, _request, actions_list, self.policy_filter_opts)
                    except Exception as e:
                        if HybridStateMachine.DEBUG:
                            print(f"[DEBUG HSM, {self.role}] Skipping policy filter due to exception: {e}")
                        _idx_f = _idx
                        _request_f = _request

                    if _idx_f != _idx or _request_f != _request:
                        _idx = _idx_f
                        _request = _request_f
                        if _idx < 0:
                            if HybridStateMachine.DEBUG:
                                print(f"[DEBUG HSM, {self.role}] After the policy filter, policy selected no-actions")
                                if _request is not None:
                                    print(f"[DEBUG HSM, {self.role}] Request is not None (unexpected) "
                                          f"{_request.__str__()}")
                                    print(f"[DEBUG HSM, {self.role}] (request in to_str format: {_request.to_str()})")
                                else:
                                    print(f"[DEBUG HSM, {self.role}] Request is None (as expected)")

                            # No actions were applied
                            self.__cur_feasible_actions_status = None
                            self.__state_changed = False
                            return -1  # Early stop
                        else:
                            if HybridStateMachine.DEBUG:
                                if _request is not None:
                                    print(f"[DEBUG HSM, {self.role}] After the policy filter, policy selected "
                                          f"{actions_list[_idx].__str__()}"
                                          f" whose request is {_request.__str__()}")
                                    print(f"[DEBUG HSM, {self.role}] (request in to_str format: {_request.to_str()})")
                                else:
                                    print(f"[DEBUG HSM, {self.role}] After the policy filter, policy selected "
                                          f"{actions_list[_idx].__str__()}")
                    else:
                        if HybridStateMachine.DEBUG:
                            print(f"[DEBUG HSM, {self.role}] After the policy filter, the policy decision was not "
                                  f"changed")
                            if _request is not None:
                                print(f"[DEBUG HSM, {self.role}] In particular, after the policy filter, policy "
                                      f"selected "
                                      f"{actions_list[_idx].__str__()}"
                                      f" whose request is {_request.__str__()}")
                                print(f"[DEBUG HSM, {self.role}] (request in to_str format: {_request.to_str()})")
                            else:
                                print(f"[DEBUG HSM, {self.role}] In particular, after the policy filter, policy "
                                      f"selected "
                                      f"{actions_list[_idx].__str__()}")

                # Saving current action
                self.limbo_state = self.state
                self.state = None
                self.__action = actions_list[_idx]
                self.__action.reset_step()  # Resetting
                self.__cur_feasible_actions_status['selected_idx'] = _idx
                self.__cur_feasible_actions_status['selected_request'] = _request

            # References
            action = self.__action
            idx = self.__cur_feasible_actions_status['selected_idx']
            request = self.__cur_feasible_actions_status['selected_request']

            # Call action
            action_call_returned_true = await action(request=request)

            # Status can be one of these:
            # 0: action fully done;
            # 1: try again this action;
            # 2: move to next action.
            if action_call_returned_true:
                if not action.is_multi_steps():

                    # Single-step actions
                    status = 0  # Done
                else:

                    # multistep actions
                    if action.cannot_be_run_anymore():  # Timeout, max time reached, max steps reached
                        if HybridStateMachine.DEBUG:
                            print(f"[DEBUG HSM] multistep action {self.__action.name} returned True and "
                                  f"cannot-be-run-anymore "
                                  f"(step: {action.get_step()}, "
                                  f"has_completion_step: {action.has_completion_step()})")
                        if self.__action.has_completion_step() and action.get_step() == 0:
                            status = 1  # Try again (next step, it will trigger the completion step)
                        else:
                            if action.get_step() >= 0:
                                status = 0  # Done, the action is fully completed
                            else:
                                status = 2  # Move to the next action (or to the next request of the same action)
                    else:
                        if HybridStateMachine.DEBUG:
                            print(f"[DEBUG HSM] multistep action {self.__action.name} can still be run")
                        status = 1  # Try again (next step)
            else:
                if not action.is_multi_steps():

                    # Single-step actions
                    if not action.has_a_timeout() or action.is_timed_out():
                        status = 2  # Move to the next action (or to the next request of the same action)
                    else:
                        status = 1  # Try again (one more time, until timeout is reached)
                else:

                    # multistep actions
                    if action.cannot_be_run_anymore():  # Timeout, max time reached, max steps reached
                        if HybridStateMachine.DEBUG:
                            print(f"[DEBUG HSM] multistep action {self.__action.name} returned False and "
                                  f"cannot-be-run-anymore "
                                  f"(step: {action.get_step()}, "
                                  f"has_completion_step: {self.__action.has_completion_step()})")
                        status = 2  # Move to the next action, since the final communication failed
                    else:
                        status = 1  # Try again (same step)

            if HybridStateMachine.DEBUG:
                print(f"[DEBUG HSM] Action {self.__action.name}, after being called, leaded to status: {status}")

            if action.msg is not None and self.show_action_completion:
                action.out_fcn(HybridStateMachine.ACTION_TICKS_PER_STATUS[status], None, None)

            # Post-call operations
            if status == 0:  # Done

                # Clearing request
                if request is not None:
                    self.__action.get_list_of_requests().remove(request)

                # State transition
                self.prev_state = self.limbo_state
                self.state = to_state_list[idx]
                self.limbo_state = None

                # Update status
                self.__state_changed = self.state != self.prev_state  # Checking if we are on a self-loop or not
                self.__last_completed_action = self.__action  # This will be set also if the state does not change

                # If we moved to another state
                # (this is not true anymore: "clearing all the pending annotations for the next possible actions")
                if self.__state_changed:
                    if HybridStateMachine.DEBUG:
                        print(f"[DEBUG HSM] Moving to state: {self.state}")
                    # for to_state, action_list in self.transitions[self.state].items():
                    #    for i, act in enumerate(action_list):
                    #        act.clear_requests()

                    # Propagating (trying to propagate forward the residual requests)
                    list_of_residual_requests = self.__action.get_list_of_requests()
                    propagated_requests = []
                    for req in list_of_residual_requests:
                        if self.request_action(req.requester, action_name=self.__action.name, args=req.args,
                                               from_state=None, to_state=None, timestamp=req.timestamp,
                                               uuid=req.uuid):
                            propagated_requests.append(req)
                    for req in propagated_requests:
                        list_of_residual_requests.remove(req)  # Clearing propagated requests

                    self.states[self.prev_state].reset()  # Reset starting time (only if state changed!)

                if HybridStateMachine.DEBUG:
                    print(f"[DEBUG HSM] Correctly completed action: {self.__action.name}")

                self.__action.reset_step()
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
                if request is not None:
                    self.__action.requests.move_request_to_back(request)  # Rotating to avoid starvation
                    attempts_to_serve_a_request_list[idx] += 1

                # Back to the original state
                self.state = self.limbo_state
                self.limbo_state = None
                if HybridStateMachine.DEBUG:
                    print(f"[DEBUG HSM] Tried and failed (failed execution): {action.name}")

                # Purging action from the current list
                if request is None or attempts_to_serve_a_request_list[idx] >= len(self.__action.requests):
                    del actions_list[idx]
                    del to_state_list[idx]

                # Update status
                self.__state_changed = False
                self.__action.reset_step()
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
                self.state_out_fcn(self.welcome_msg)
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

    def request_action(self, signature: object, action_name: str, args: dict | None = None,
                       from_state: str | None = None, to_state: str | None = None,
                       timestamp: float | None = None, uuid: str | None = None):
        """Allows an external entity to request a specific action. The request is validated by a signature checker
        (if one exists) and then queued on the corresponding action. This method enables dynamic, external triggers for
        state machine transitions.

        Args:
            signature: An object used for validating the request's origin.
            action_name: The name of the requested action.
            args: Arguments for the requested action.
            from_state: The optional starting state for the requested transition.
            to_state: The optional destination state for the requested transition.
            timestamp: The time the request was made.
            uuid: A unique identifier for the request.

        Returns:
            True if the request was accepted and queued, False otherwise.
        """
        if HybridStateMachine.DEBUG:
            print(f"[DEBUG HSM] Received a request signed as {signature}, "
                  f"asking for action {action_name}, with args: {args}, "
                  f"from_state: {from_state}, to_state: {to_state}, uuid: {uuid}")

        # Discard suggestions if they are not trusted
        if self.request_signature_checker is not None and not self.request_signature_checker(signature):
            if HybridStateMachine.DEBUG:
                print("[DEBUG HSM] Request signature check failed")
            return False

        # If state is not provided, the current state is assumed
        if from_state is None:
            # If the request arrives in the middle of a multistep action, we need to check limbo state
            from_state = self.state if self.state is not None else self.limbo_state
        if from_state not in self.transitions:
            if HybridStateMachine.DEBUG:
                print(f"[DEBUG HSM] Request not accepted: not valid source state ({from_state})")
            return False

        # If the destination state is not provided, all the possible destination from the current state are considered
        if to_state is not None and to_state not in self.transitions[from_state]:
            if HybridStateMachine.DEBUG:
                print(f"[DEBUG HSM] Request not accepted: not valid destination state ({to_state})")
            return False
        to_states = self.transitions[from_state].keys() if to_state is None else [to_state]

        for to_state in to_states:
            action_list = self.transitions[from_state][to_state]
            for i, action in enumerate(action_list):
                if HybridStateMachine.DEBUG:
                    print(f"[DEBUG HSM] Comparing with action: {str(action)}")
                if action.same_as(name=action_name, args=args):
                    if HybridStateMachine.DEBUG:
                        print(f"[DEBUG HSM] Requested action found in state {from_state}, adding request to the queue")

                    # Action found, let's save the suggestion
                    action.add_request(signature, args, timestamp=timestamp, uuid=uuid)
                    return True

        # If the action was not found
        if HybridStateMachine.DEBUG:
            print("[DEBUG HSM] Requested action not found")
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

    def wait_for_actions(self, from_state: str, to_state: str, wait: bool = True):
        """Sets the `ready` flag for a specific action (or group of actions) between two states. This allows for
        fine-grained control over which transitions are active.

        Args:
            from_state: The name of the starting state.
            to_state: The name of the destination state.
            wait: A boolean flag to either set the action as not ready (`True`) or ready (`False`).

        Returns:
            True if the specified action was found, False otherwise.
        """
        if from_state not in self.transitions or to_state not in self.transitions[from_state]:
            return False

        for action in self.transitions[from_state][to_state]:
            if wait:
                action.set_as_not_ready()
            else:
                action.set_as_ready()
        return True

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
            except Exception:   # If load fails, we assume it changed
                if HybridStateMachine.DEBUG:
                    print(f"[DEBUG HSM] Error while reloading the exising machine from {filename}, assuming it changed")

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
                    act_name, act_args, state_id, waiting_time = state_action_list
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
            if HybridStateMachine.DEBUG:
                print(f"[DEBUG HSM] Error while saving to PDF {e}")
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
                print(f"{state}, no actions")
            for action in action_list:
                print(f"{state} --> {to_state} {action}")

    # Noinspection PyMethodMayBeStatic
    def __policy_first_requested_or_first_ready(self, actions_list: list[Action]) -> tuple[int, ActionRequest | None]:
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
            _list_of_requests = action.get_list_of_requests()
            if len(_list_of_requests) > 0:
                _selected_action_idx = i
                _selected_request = _list_of_requests.get_oldest_request()
                return _selected_action_idx, _selected_request
        for i, action in enumerate(actions_list):
            if action.is_ready(consider_requests=False):
                _selected_action_idx = i
                _selected_request = None
                return _selected_action_idx, _selected_request
        _selected_action_idx = -1
        _selected_request = None
        return _selected_action_idx, _selected_request
