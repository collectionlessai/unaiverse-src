"""
 █████  █████ ██████   █████           █████ █████   █████ ██████████ ███████████    █████████  ██████████
░░███  ░░███ ░░██████ ░░███           ░░███ ░░███   ░░███ ░░███░░░░░█░░███░░░░░███  ███░░░░░███░░███░░░░░█
 ░███   ░███  ░███░███ ░███   ██████   ░███  ░███    ░███  ░███  █ ░  ░███    ░███ ░███    ░░░  ░███  █ ░ 
 ░███   ░███  ░███░░███░███  ░░░░░███  ░███  ░███    ░███  ░██████    ░██████████  ░░█████████  ░██████   
 ░███   ░███  ░███ ░░██████   ███████  ░███  ░░███   ███   ░███░░█    ░███░░░░░███  ░░░░░░░░███ ░███░░█   
 ░███   ░███  ░███  ░░█████  ███░░███  ░███   ░░░█████░    ░███ ░   █ ░███    ░███  ███    ░███ ░███ ░   █
 ░░████████   █████  ░░█████░░████████ █████    ░░███      ██████████ █████   █████░░█████████  ██████████
  ░░░░░░░░   ░░░░░    ░░░░░  ░░░░░░░░ ░░░░░      ░░░      ░░░░░░░░░░ ░░░░░   ░░░░░  ░░░░░░░░░  ░░░░░░░░░░ 

~ Registration/Login: https://unaiverse.io
~ Code Repositories:  https://github.com/collectionlessai/
~ Main Developers:    Stefano Melacci (Project Leader), Christian Di Maio, Tommaso Guidi

A Collectionless AI Project (https://collectionless.ai)
"""
import os
import json
import copy
import html
import time
import inspect
import graphviz
from typing_extensions import Self
from collections.abc import Iterable, Callable


class Action:
    # candidate argument names (when calling an action) that tells that such an action is multi-steps
    STEPS_ARG_NAMES = {'steps', 'samples'}
    SECONDS_ARG_NAMES = {'time'}
    TIMEOUT_ARG_NAMES = {'timeout'}
    DELAY_ARG_NAMES = {'delay'}
    COMPLETED_NAMES = {'_completed'}
    REQUESTER_ARG_NAMES = {'_requester'}
    REQUEST_TIME_NAMES = {'_request_time'}
    REQUEST_UUID_NAMES = {'_request_uuid'}
    NOT_READY_PREFIXES = ('get_', 'got_', 'do_', 'done_')
    KNOWN_SINGLE_STEP_ACTION_PREFIXES = ('ask_', )

    # completion reasons
    MAX_STEPS_REACHED = 0  # single-step actions always complete due to this reason
    MAX_TIME_REACHED = 1
    MAX_TIMEOUT_DURING_ATTEMPTS_REACHED = 2

    # output print function
    out_fcn = print

    def __init__(self, name: str, args: dict, actionable: object,
                 idx: int = -1,
                 ready: bool = True,
                 wildcards: dict[str, str | float | int] | None = None,
                 msg: str | None = None):

        # basic properties
        self.name = name  # name of the action (name of the corresponding method)
        self.args = args  # dictionary of arguments to pass to the action
        self.actionable = actionable  # object on which the method whose name is self.name is searched
        self.ready = ready  # boolean flag telling if the action can considered ready to be executed
        self.requests = {}  # list of requests to make this action ready to be executed (customizable)
        self.id = idx  # unique ID of the action (-1 if not needed)
        self.msg = msg  # human-readable message associated to this instance of action

        # fix UNICODE chars
        if self.msg is not None:
            self.msg = html.unescape(self.msg)

        # reference elements
        self.args_with_wildcards = copy.deepcopy(self.args)  # backup of the originally provided arguments
        self.__fcn = self.__action_name_to_callable(name)  # the real method to be called
        self.__sig = inspect.signature(self.__fcn)  # signature of the method for argument inspection

        # parameter names and default values
        self.param_list = []  # full list of the parameters that the action supports
        self.param_to_default_value = {}  # from parameter to its default value, if any
        self.__get_action_params()  # this will fill the two attributes above
        self.__check_if_args_exist(self.args, exception=True)  # checking arguments

        # argument values replaced by wildcards (commonly assumed to be in the format <value>)
        self.wildcards = wildcards if wildcards is not None else {}  # value-to-value (es: <playlist> to this:and:this)
        self.__replace_wildcard_values()  # this will alter self.arg in function of the provided wildcards

        # number of steps of this function
        self.__step = -1  # default initial step index (remark: "step INDEX", so when it is 0 it means a step was done)
        self.__total_steps = 1   # total step of an action (a multi-steps action has != 1 steps)
        self.__guess_total_steps(self.__get_actual_params({}))  # this will "guess" the value of self.__total_steps

        # time-based metrics
        self.__starting_time = 0
        self.__total_time = 0  # a total time <= 0 means "no total time at all"
        self.__guess_total_time(self.__get_actual_params({}))  # this will "guess" the value of self.__total_time

        # time-based metrics
        self.__timeout_starting_time = 0
        self.__timeout = 0  # a timeout <= 0 means "no total time at all"
        self.__guess_timeout(self.__get_actual_params({}))  # this will "guess" the value of self.__timeout

        # time-based metrics
        self.__delay = 0
        self.__guess_delay(self.__get_actual_params({}))  # this will "guess" the value of self.__delay

        # fixing (if no options are specified, assuming a single-step action)
        if self.__total_steps <= 0 and self.__total_time <= 0:
            self.__total_steps = 1

        # fixing (forcing NOT-ready on some actions)
        for prefix in Action.NOT_READY_PREFIXES:
            if self.name.startswith(prefix):
                self.ready = False

        self.__has_completion_step = False
        for completed_name in Action.COMPLETED_NAMES:
            if completed_name in self.param_list:
                self.__has_completion_step = True
                break

        # status
        self.__cannot_be_run_anymore = False

    def __call__(self, requester: object | None = None, requested_args: dict | None = None,
                 request_time: float = -1, request_uuid: str | None = None) -> bool:
        self.__check_if_args_exist(requested_args, exception=True)
        actual_args = self.__get_actual_params(requested_args)  # getting the actual values of the arguments

        if self.msg is not None:
            Action.out_fcn(self.msg)

        if actual_args is not None:

            # getting the values for the main involved measures: total steps, total time, timeout
            self.__guess_total_steps(actual_args)
            self.__guess_total_time(actual_args)
            self.__guess_timeout(actual_args)

            # storing the time index that is related to the timeout (do this before calling self.is_timed_out())
            if self.__timeout_starting_time <= 0:
                self.__timeout_starting_time = time.perf_counter()

            # storing the starting time (do this before calling self.was_last_step_done())
            if self.__starting_time <= 0:
                self.__starting_time = time.perf_counter()

            # setting up the flag that tells if the action reached a point in which it cannot be run anymore
            self.__cannot_be_run_anymore = self.is_timed_out() or self.was_last_step_done()

            if HybridStateMachine.DEBUG:
                if self.__cannot_be_run_anymore:
                    print(f"[DEBUG HSM] Cannot-be-run-anymore set to True, "
                          f"due to self.is_timed_out()={self.is_timed_out()} or "
                          f"self.was_last_step_done()={self.was_last_step_done()}")

            if self.__cannot_be_run_anymore and not self.is_multi_steps():
                return False

            # setting up the information on whether a multistep action is completed
            # (for example, to tell that now it is time for a callback)
            calling_completion_step = False
            for completed_name in Action.COMPLETED_NAMES:
                if completed_name in actual_args:
                    calling_completion_step = self.__cannot_be_run_anymore and self.get_step() >= 0
                    actual_args[completed_name] = calling_completion_step
                    break

            # we are done, no need to call the action again
            if self.__cannot_be_run_anymore and not calling_completion_step:
                return True

            # setting up the requester
            for req_arg_name in Action.REQUESTER_ARG_NAMES:
                if req_arg_name in actual_args:
                    actual_args[req_arg_name] = requester
                    break

            # setting up the request time
            for req_time_name in Action.REQUEST_TIME_NAMES:
                if req_time_name in actual_args:
                    actual_args[req_time_name] = request_time
                    break

            # setting up the request uuid
            for req_uuid_name in Action.REQUEST_UUID_NAMES:
                if req_uuid_name in actual_args:
                    actual_args[req_uuid_name] = request_uuid
                    break

            # fixing (if no options are specified, assuming a single-step action)
            if self.__total_steps == 0 and self.__total_time == 0:
                self.__total_steps = 1

            # fixing the single step case: in this case, time does not matter, so we force it to zero
            if self.__total_steps == 1:
                self.__total_time = 0

            # increasing the step index
            self.__step += 1  # this is a step index, so self.__step == 0 means "done 1 step"

            if HybridStateMachine.DEBUG:
                if requester is None:
                    requester_str = "nobody"
                else:
                    requester_str = requester
                print(f"[DEBUG HSM] Calling function {self.name} (multi_steps: {self.is_multi_steps()}), "
                      f"requested by {requester_str}, with actual params: {actual_args}")

            # calling the method here
            ret = self.__fcn(**actual_args)

            if HybridStateMachine.DEBUG:
                print(f"[DEBUG HSM] Returned: {ret}")

            # if action failed, be sure to reduce the step counter (only if it was actually incremented)
            if not ret:
                self.__step -= 1

            # if it went OK, we reset the time counter that is related to the timeout
            else:
                self.__timeout_starting_time = 0

            return ret
        else:
            if HybridStateMachine.DEBUG:
                print(f"[DEBUG HSM] Tried and failed (missing actual param): {self}")
            return False

    def __str__(self):
        return (f"[Action: {self.name}] id: {self.id}, args: {self.args}, param_list: {self.param_list}, "
                f"total_steps: {self.__total_steps}, "
                f"total_time: {self.__total_time}, timeout: {self.__timeout}, "
                f"ready: {self.ready}, requests: {str(self.requests)}, msg: {str(self.msg)}]")

    def set_as_ready(self):
        self.ready = True

    def set_as_not_ready(self):
        self.ready = False

    def is_ready(self, consider_requests: bool = True) -> bool:
        return self.ready or (consider_requests and len(self.requests) > 0)

    def was_last_step_done(self) -> bool:
        return ((self.__total_steps > 0 and self.__step == self.__total_steps - 1) or
                (self.__total_time > 0 and ((time.perf_counter() - self.__starting_time) >= self.__total_time)))

    def cannot_be_run_anymore(self):
        return self.__cannot_be_run_anymore

    def has_completion_step(self):
        return self.__has_completion_step

    def is_multi_steps(self) -> bool:
        return self.__total_steps != 1

    def has_a_timeout(self):
        return self.__timeout > 0

    def is_delayed(self, starting_time: float):
        return self.__delay > 0 and (time.perf_counter() - starting_time) <= self.__delay

    def is_timed_out(self):
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

    def to_list(self, minimal=False) -> list:
        if not minimal:
            if self.msg is not None:
                msg = self.msg.encode("ascii", "xmlcharrefreplace").decode("ascii")
            else:
                msg = None
            return [self.name, self.args, self.ready, self.id] + ([msg] if msg is not None else [])
        else:
            return [self.name, self.args]

    def same_as(self, name: str, args: dict | None):
        if args is None:
            args = {}

        # the current action is the same of another action called with some arguments "args" if:
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

    def __check_if_args_exist(self, args: dict, exception: bool = False) -> bool:
        if args is not None:
            for param_name in args.keys():
                if param_name not in self.param_list:
                    if exception:
                        raise ValueError(f"Unknown parameter {param_name} for action {self.name}")
                    else:
                        return False
        return True

    def set_wildcards(self, wildcards: dict[str, str | float | int] | None):
        self.wildcards = wildcards if wildcards is not None else {}
        self.__replace_wildcard_values()

    def add_request(self, generic_request_obj: object, args: dict, timestamp: float, uuid: str):
        if generic_request_obj not in self.requests:
            self.requests[generic_request_obj] = (args, timestamp, uuid)

    def clear_requests(self):
        self.requests = {}

    def get_requests(self) -> dict[object, tuple[dict, float, str]]:
        return self.requests

    def reset_step(self):
        self.__step = -1
        self.__starting_time = 0.
        self.__timeout_starting_time = 0.
        self.__cannot_be_run_anymore = False

    def get_step(self):
        return self.__step

    def get_total_steps(self):
        return self.__total_steps

    def get_starting_time(self):
        return self.__starting_time

    def get_total_time(self):
        return self.__total_time

    def __get_actual_params(self, additional_args: dict | None) -> dict | None:
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
        """Get a function from its name."""

        if self.actionable is not None:
            action_fcn = getattr(self.actionable, action_name)
            if action_fcn is None:
                raise ValueError("Cannot find function/method: " + str(action_name))
            return action_fcn
        else:
            return None

    def __get_action_params(self):
        self.param_list = [param_name for param_name in self.__sig.parameters.keys()]
        self.param_to_default_value = {param.name: param.default for param in self.__sig.parameters.values() if
                                       param.default is not inspect.Parameter.empty}

    def __replace_wildcard_values(self):
        if self.args_with_wildcards is None:
            self.args_with_wildcards = copy.deepcopy(self.args)  # backup before applying wildcards (first time only)
        else:
            self.args = copy.deepcopy(self.args_with_wildcards)  # restore a backup before applying wildcards

        for k, v in self.args.items():
            for wildcard_from, wildcard_to in self.wildcards.items():
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

    def __guess_total_steps(self, args):
        for prefix in Action.KNOWN_SINGLE_STEP_ACTION_PREFIXES:
            if self.name.startswith(prefix):
                return
        for arg_name in Action.STEPS_ARG_NAMES:
            if arg_name in args:
                if isinstance(args[arg_name], int):
                    self.__total_steps = max(float(args[arg_name]), 1.)
                break

    def __guess_total_time(self, args):
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
        for arg_name in Action.DELAY_ARG_NAMES:
            if arg_name in args:
                try:
                    self.__delay = max(float(args[arg_name]), 0.)
                except ValueError:
                    self.__delay = -1.
                    pass
                break


class State:

    # output print function
    out_fcn = print

    def __init__(self, name: str, idx: int = -1, action: Action | None = None, waiting_time: float = 0.,
                 blocking: bool = True, msg: str | None = None):
        self.name = name  # name of the state (must be unique)
        self.action = action  # inner state action (it can be None)
        self.id = idx  # unique ID of the state (-1 if not needed)
        self.waiting_time = waiting_time  # number of seconds to wait in the current state before acting
        self.starting_time = 0.
        self.blocking = blocking
        self.msg = msg  # human-readable message associated to this instance of action

        # fix UNICODE chars
        if self.msg is not None:
            self.msg = html.unescape(self.msg)

    def __call__(self, *args, **kwargs) -> bool | None:
        if self.starting_time <= 0.:
            self.starting_time = time.perf_counter()

        if self.msg is not None:
            State.out_fcn(self.msg)

        if self.action is not None:
            if HybridStateMachine.DEBUG:
                print("[DEBUG HSM] Running action on state: " + self.action.name)
            self.action.reset_step()
            return self.action(*args, **kwargs)
        else:
            return None

    def __str__(self):
        return (f"[State: {self.name}] id: {self.id}, waiting_time: {self.waiting_time}, blocking: {self.blocking}, "
                f"action -> {self.action if self.action is not None else 'none'}, msg: {self.msg}")

    def must_wait(self) -> bool:
        if self.waiting_time > 0.:
            if (time.perf_counter() - self.starting_time) >= self.waiting_time:
                if HybridStateMachine.DEBUG:
                    print(f"[DEBUG HSM] Time passing: {(time.perf_counter() - self.starting_time)} seconds")
                return False
            else:
                return True
        else:
            return False

    def to_list(self) -> list:
        if self.msg is not None:
            msg = self.msg.encode("ascii", "xmlcharrefreplace").decode("ascii")
        else:
            msg = None
        return ((self.action.to_list(minimal=True) if self.action is not None else [None, None]) +
                ([self.id, self.blocking, self.waiting_time] + ([msg] if msg is not None else [])))

    def has_action(self):
        return self.action is not None

    def get_starting_time(self):
        return self.starting_time

    def reset(self):
        self.starting_time = 0.
        if self.action is not None:
            self.action.reset_step()

    def set_blocking(self, blocking: bool):
        self.blocking = blocking


class HybridStateMachine:
    DEBUG = True
    DEFAULT_WILDCARDS = {'<world>': '<world>', '<agent>': '<agent>'}

    def __init__(self, actionable: object, wildcards: dict[str, str | float | int] | None = None,
                 request_signature_checker: Callable[[object], bool] | None = None,
                 policy: Callable[[list[Action]], int] | None = None):

        # states are identified by strings, and then handled as State object with possibly and integer ID and action
        self.initial_state: str | None = None  # initial state of the machine
        self.prev_state: str | None = None  # previous state
        self.limbo_state: str | None = None  # when an action takes more than a step to complete, we are in "limbo"
        self.state: str | None = None  # current state
        self.role: str | None = None  # role of the agent in the state machine (e.g., teacher, student, etc.)
        self.enabled: bool = True
        self.states: dict[str, State] = {}  # state name to State object

        # actions (transitions) are handled as Action objects in-between state strings
        self.transitions: dict[str, dict[str, list[Action]]] = {}  # pair-of-states to the actions between them
        self.actionable: object = actionable  # the object on whose methods are actions that the machine calls
        self.wildcards: dict[str, str | float | int] | None = wildcards \
            if wildcards is not None else {}  # from a wildcards string to a specific value (used in action arguments)
        self.policy = policy if policy is not None else self.__policy_first_requested_or_first_ready

        # actions can be requested from the "outside": each request if checked by this function, if any
        self.request_signature_checker: Callable[[object], bool] | None = request_signature_checker

        # running data
        self.__action: Action | None = None  # action that is being executed (could take more than a step to complete)
        self.__last_completed_action: Action | None = None
        self.__cur_feasible_actions_status: dict | None = None  # store info of the executed action (for multi-steps)
        self.__id_to_state: list[State] = []  # map from state ID to State object
        self.__id_to_action: list[Action] = []  # map from action ID to Action object
        self.__state_changed = False  # internal flag

        # forcing default wildcards
        self.add_wildcards(HybridStateMachine.DEFAULT_WILDCARDS)

        # forcing output function
        self.__last_printed_msg = None

        def wrapped_out_fcn(msg: str):
            if msg is not None:
                if msg != self.__last_printed_msg:
                    print(msg)
                    self.__last_printed_msg = msg

        State.out_fcn = wrapped_out_fcn
        Action.out_fcn = wrapped_out_fcn

    def to_dict(self):
        return {
            'initial_state': self.initial_state,
            'state': self.state,
            'role': self.role,
            'prev_state': self.prev_state,
            'limbo_state': self.limbo_state,
            'state_actions': {
                state.name: state.to_list() for state in self.__id_to_state
            },
            'transitions': {
                from_state: {
                    to_state: [act.to_list() for act in action_list] for to_state, action_list in to_states.items()
                }
                for from_state, to_states in self.transitions.items() if len(to_states) > 0
            },
            'cur_action': self.__action.to_list() if self.__action is not None else None
        }

    def __str__(self):
        hsm_data = self.to_dict()

        def custom_serializer(obj):
            if not isinstance(obj, (int, str, float, bool, list, tuple, dict, set)):
                return "_non_basic_type_removed_"
            else:
                return obj

        json_str = json.dumps(hsm_data, indent=4, default=custom_serializer)

        # compacting lists
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
                elif char == '\n' and stack:  # skipping newline
                    i += 1
                    while i < len(json_string) and json_string[i] in ' \t':
                        i += 1
                    if output[-1] == ",":
                        output.append(" ")
                    continue  # do not output newline or following spaces
                else:
                    output.append(char)
                i += 1
            return ''.join(output)

        return remove_newlines_in_lists(json_str)

    def set_actionable(self, obj: object):
        """Set the object where actions should be found (as methods)."""

        self.actionable = obj

        for state_obj in self.states.values():
            if state_obj.action is not None:
                state_obj.action.actionable = obj

    def set_wildcards(self, wildcards: dict[str, str | float | int] | None):
        """Set the dictionary of wildcards used during the loading process."""

        self.wildcards = wildcards if wildcards is not None else {}
        for action in self.__id_to_action:
            action.set_wildcards(self.wildcards)

    def set_role(self, role: str):
        """Set the role."""
        self.role = role

    def get_wildcards(self) -> dict[str, str | float | int]:
        return self.wildcards

    def add_wildcards(self, wildcards: dict[str, str | float | int | list[str]]):
        self.wildcards.update(wildcards)
        self.set_wildcards(self.wildcards)

    def update_wildcard(self, wildcard_key: str, wildcard_value: str | float | int):
        assert wildcard_key in self.wildcards, f"{wildcard_key} is not a valid wildcard"
        self.wildcards[wildcard_key] = wildcard_value
        self.set_wildcards(self.wildcards)

    def get_action_step(self) -> int:
        return self.__action.get_step() if self.__action is not None else -1

    def is_busy_acting(self):
        return self.get_action_step() >= 0

    def add_state(self, state: str, action: str = None, args: dict | None = None, state_id: int | None = None,
                  waiting_time: float | None = None, blocking: bool | None = None, msg: str | None = None):
        """Add a state with its action (inner action) creating it from scratch."""

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
                         actionable=self.actionable, wildcards=self.wildcards)
            self.__id_to_action.append(act)
        if waiting_time is None:
            waiting_time = sta_obj.waiting_time if sta_obj is not None else 0.  # default waiting time
        if blocking is None:
            blocking = sta_obj.blocking if sta_obj is not None else True  # default blocking
        if msg is None:
            msg = sta_obj.msg if sta_obj is not None else None

        sta = State(name=state, idx=state_id, action=act, waiting_time=waiting_time, blocking=blocking, msg=msg)
        if state not in self.states:
            self.__id_to_state.append(sta)
        else:
            self.__id_to_state[state_id] = sta
        self.states[state] = sta

        if len(self.__id_to_state) == 1 and self.state is None:
            self.set_state(sta.name)

    def get_state_name(self) -> str | None:
        """Returns the name of the current state of the HSM."""

        return self.state

    def get_state(self) -> State | None:
        """Returns the current state of the HSM."""

        return self.states[self.state] if self.state is not None else None

    def get_action(self):
        return self.__action

    def get_action_name(self) -> str | None:
        """Returns the name of current action being performed by the HSM, if any."""

        return self.__action.name if self.__action is not None else None

    def reset_state(self):
        """Go back to the initial state of the HSM."""

        self.state = self.initial_state
        self.limbo_state = None
        self.prev_state = None
        self.__action = None
        for act in self.__id_to_action:
            act.reset_step()
        for s in self.__id_to_state:
            if s.action is not None:
                s.action.reset_step()

    def get_states(self) -> Iterable[str]:
        """Get all the states of the HSM."""

        return list(set(list(self.transitions.keys()) + self.__id_to_state))

    def set_state(self, state: str):
        """Set the current state."""

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

    def add_transit(self, from_state: str, to_state: str,
                    action: str, args: dict | None = None, ready: bool = True,
                    act_id: int | None = None, msg: str | None = None):
        """Define a transition between two states with an associated action."""

        # plugging a previously loaded HSM
        if os.path.exists(to_state):
            file_name = to_state
            hsm = HybridStateMachine(self.actionable).load(file_name)

            # first, we avoid name clashes, renaming already-used-state-names in original_name~1 (or ~2, or ~3, ...)
            hsm_states = list(hsm.states.keys())  # keep the list(...) thing, since we need a copy here (it will change)
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

            # saving
            initial_state_was_set = self.initial_state is not None
            state_was_set = self.state is not None

            # include actions/states from another HSM
            self.include(hsm)

            # adding a transition to the initial state of the given HSM
            self.add_transit(from_state=from_state, to_state=hsm.initial_state, action=action, args=args,
                             ready=ready, act_id=None, msg=msg)

            # restoring
            self.initial_state = from_state if not initial_state_was_set else self.initial_state
            self.state = from_state if not state_was_set else self.state
            return

        # adding a new transition
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

        # clearing
        if to_state not in self.transitions[from_state]:
            self.transitions[from_state][to_state] = []

        # checking
        existing_action_list = self.transitions[from_state][to_state]
        for existing_action in existing_action_list:
            if existing_action.same_as(name=action, args=args):
                raise ValueError(f"Repeated transition from {from_state} to {to_state}: "
                                 f"{existing_action.to_list()}")

        # adding the new action
        new_action = Action(name=action, args=args, idx=act_id, actionable=self.actionable, ready=ready, msg=msg)
        self.transitions[from_state][to_state].append(new_action)
        self.__id_to_action.append(new_action)

    def include(self, hsm, make_a_copy=False):

        # copying wildcards
        self.add_wildcards(hsm.get_wildcards())

        # adding states before adding transitions, so that we also add inner state actions, if any
        for _state in hsm.states.values():
            self.add_state(state=_state.name,
                           action=_state.action.name if _state.action is not None else None,
                           waiting_time=_state.waiting_time,
                           args=copy.deepcopy(_state.action.args_with_wildcards) if _state.action is not None else None,
                           state_id=None,
                           blocking=_state.blocking,
                           msg=_state.msg)

        # copy all the transitions of the HSM
        for _from_state, _to_states in hsm.transitions.items():
            for _to_state, _action_list in _to_states.items():
                for _action in _action_list:
                    self.add_transit(from_state=_from_state, to_state=_to_state, action=_action.name,
                                     args=copy.deepcopy(_action.args_with_wildcards), ready=_action.ready,
                                     act_id=None, msg=_action.msg)

        if make_a_copy:
            self.state = hsm.state
            self.prev_state = hsm.state
            self.initial_state = hsm.initial_state
            self.limbo_state = hsm.limbo_state

    def must_wait(self) -> bool:
        if self.state is not None:
            return self.states[self.state].must_wait()
        else:
            return False

    def is_enabled(self):
        return self.enabled

    def enable(self, yes_or_not: bool):
        self.enabled = yes_or_not

    def act_states(self):
        """Apply actions that do not trigger a state transition."""
        if not self.enabled:
            return

        if self.state is not None:  # when in the middle of an action, the state is Nones
            self.states[self.state]()  # run the action (if any)

    def act_transitions(self, requested_only: bool = False):
        if not self.enabled:
            return -1

        # collecting list of feasible actions, wait flags, etc. (from the current state)
        if self.__cur_feasible_actions_status is None:
            if self.state is None:
                return -1

            actions_list = []
            to_state_list = []

            for to_state, action_list in self.transitions[self.state].items():
                for i, action in enumerate(action_list):
                    if (action.is_ready() and (not requested_only or len(action.requests) > 0) and
                            not action.is_delayed(self.states[self.state].starting_time)):
                        actions_list.append(action)
                        to_state_list.append(to_state)

            if len(actions_list) > 0:
                self.__cur_feasible_actions_status = {
                    'actions_list': actions_list,
                    'to_state_list': to_state_list,
                    'selected_idx': 0,
                    'selected_requester': None,
                    'selected_requested_args': {},
                    'selected_request_time': -1.,
                    'selected_request_uuid': None
                }
        else:

            # reloading the already computed set of actions, wait flags, etc. (when in the middle of an action)
            actions_list = self.__cur_feasible_actions_status['actions_list']
            to_state_list = self.__cur_feasible_actions_status['to_state_list']

        # using the selected policy to decide what action to apply
        while len(actions_list) > 0:

            # it there was an already selected action (for example a multistep action), then continue with it,
            # otherwise, select a new one following a certain policy (actually, first-come first-served)
            if self.__action is None:

                # naive policy: take the first action that is ready
                _idx, (_requester, (_requested_args, _request_time, _request_uuid)) = self.policy(actions_list)

                # saving current action
                self.limbo_state = self.state
                self.state = None
                self.__action = actions_list[_idx]
                self.__action.reset_step()  # resetting
                self.__cur_feasible_actions_status['selected_idx'] = _idx
                self.__cur_feasible_actions_status['selected_requester'] = _requester
                self.__cur_feasible_actions_status['selected_requested_args'] = _requested_args
                self.__cur_feasible_actions_status['selected_request_time'] = _request_time
                self.__cur_feasible_actions_status['selected_request_uuid'] = _request_uuid

                if HybridStateMachine.DEBUG:
                    print(f"[DEBUG HSM] Policy selected {self.__action.__str__()} whose requester is {_requester}")

            # references
            action = self.__action
            idx = self.__cur_feasible_actions_status['selected_idx']
            requester = self.__cur_feasible_actions_status['selected_requester']
            requested_args = self.__cur_feasible_actions_status['selected_requested_args']
            request_time = self.__cur_feasible_actions_status['selected_request_time']
            request_uuid = self.__cur_feasible_actions_status['selected_request_uuid']

            # call action
            action_call_returned_true = action(requester=requester,
                                               requested_args=requested_args,
                                               request_time=request_time, request_uuid=request_uuid)

            # status can be one of these:
            # 0: action fully done;
            # 1: try again this action;
            # 2: move to next action.
            if action_call_returned_true:
                if not action.is_multi_steps():
                    # single-step actions
                    status = 0  # done
                else:
                    # multi-step actions
                    if action.cannot_be_run_anymore():  # timeout, max time reached, max steps reached
                        if HybridStateMachine.DEBUG:
                            print(f"[DEBUG HSM] Multi-step action {self.__action.name} returned True and "
                                  f"cannot-be-run-anymore "
                                  f"(step: {action.get_step()}, "
                                  f"has_completion_step: {action.has_completion_step()})")
                        if self.__action.has_completion_step() and action.get_step() == 0:
                            status = 1  # try again (next step, it will trigger the completion step)
                        else:
                            if action.get_step() >= 0:
                                status = 0  # done, the action is fully completed
                            else:
                                status = 2  # move to the next action
                    else:
                        if HybridStateMachine.DEBUG:
                            print(f"[DEBUG HSM] Multi-step action {self.__action.name} can still be run")
                        status = 1  # try again (next step)
            else:
                if not action.is_multi_steps():
                    # single-step actions
                    if not action.has_a_timeout() or action.is_timed_out():
                        status = 2  # move to the next action
                    else:
                        status = 1  # try again (one more time, until timeout is reached)
                else:
                    # multi-step actions
                    if action.cannot_be_run_anymore():  # timeout, max time reached, max steps reached
                        if HybridStateMachine.DEBUG:
                            print(f"[DEBUG HSM] Multi-step action {self.__action.name} returned False and "
                                  f"cannot-be-run-anymore "
                                  f"(step: {action.get_step()}, "
                                  f"has_completion_step: {self.__action.has_completion_step()})")
                        status = 2  # move to the next action, since the final communication failed
                    else:
                        status = 1  # try again (same step)

            if HybridStateMachine.DEBUG:
                print(f"[DEBUG HSM] Action {self.__action.name}, after being called, leaded to status: {status}")

            # post-call operations
            if status == 0:  # done

                # clearing request
                requests = self.__action.get_requests()
                if requester is not None and requester in requests:
                    del requests[requester]

                # state transition
                self.prev_state = self.limbo_state
                self.state = to_state_list[idx]
                self.limbo_state = None

                # update status
                self.__state_changed = self.state != self.prev_state  # checking if we are on a self-loop or not

                # if we moved to another state, clearing all the pending annotations for the next possible actions
                if self.__state_changed:
                    if HybridStateMachine.DEBUG:
                        print(f"[DEBUG HSM] Moving to state: {self.state}")
                    for to_state, action_list in self.transitions[self.state].items():
                        for i, act in enumerate(action_list):
                            act.clear_requests()

                    # propagating (trying to propagate forward the residual requests)
                    residual_requests = self.__action.get_requests()
                    for _requester, (_requested_args, _request_time, _request_uuid) in residual_requests.items():
                        self.request_action(_requester, action_name=self.__action.name, args=_requested_args,
                                            from_state=None, to_state=None, timestamp=_request_time, uuid=_request_uuid)

                if HybridStateMachine.DEBUG:
                    print(f"[DEBUG HSM] Correctly completed action: {self.__action.name}")

                self.states[self.prev_state].reset()  # reset starting time
                self.__action.reset_step()
                self.__action = None  # clearing
                self.__cur_feasible_actions_status = None

                return 0  # transition done, no need to check other actions!

            elif status == 1:  # try again the same action (either a new step or an already done-and-failed one)

                # update status
                self.__state_changed = False
                if self.prev_state is not None:
                    self.states[self.prev_state].reset()  # reset starting time

                return 1  # transition not-done: no need to check other actions, the current one will be run again

            elif status == 2:  # move to the next action

                # clearing request
                requests = self.__action.get_requests()
                if requester is not None and requester in requests:
                    del requests[requester]

                # back to the original state
                self.state = self.limbo_state
                self.limbo_state = None
                if HybridStateMachine.DEBUG:
                    print(f"[DEBUG HSM] Tried and failed (failed execution): {action.name}")

                # purging action from the current list
                del actions_list[idx]
                del to_state_list[idx]

                # update status
                self.__state_changed = False
                self.__action.reset_step()
                self.__action = None  # clearing

                continue  # move to the next action
            else:
                raise ValueError("Unexpected status: " + str(status))

        # no actions were applied
        self.__cur_feasible_actions_status = None
        self.__state_changed = False
        return -1

    def act(self):

        # it keeps processing states and actions, until all the current feasible actions fail
        # (also when a step of a multistep action is executed) or a blocking state is reached
        while True:
            self.act_states()
            ret = self.act_transitions(self.must_wait())
            if ret != 0 or (self.state is not None and self.states[self.state].blocking):
                break

    def get_state_changed(self):
        return self.__state_changed

    def request_action(self, signature: object, action_name: str, args: dict | None = None,
                       from_state: str | None = None, to_state: str | None = None,
                       timestamp: float | None = None, uuid: str | None = None) -> bool:
        """Adds a suggestion to an action, if the action exists and if it is positively checked."""
        if HybridStateMachine.DEBUG:
            print(f"[DEBUG HSM] Received a request signed as {signature}, "
                  f"asking for action {action_name}, with args: {args}, "
                  f"from_state: {from_state}, to_state: {to_state}, uuid: {uuid}")

        # discard suggestions if they are not trusted
        if self.request_signature_checker is not None and not self.request_signature_checker(signature):
            if HybridStateMachine.DEBUG:
                print("[DEBUG HSM] Request signature check failed")
            return False

        # if state is not provided, the current state is assumed
        if from_state is None:
            from_state = self.state
        if from_state not in self.transitions:
            if HybridStateMachine.DEBUG:
                print(f"[DEBUG HSM] Request not accepted: not valid source state ({from_state})")
            return False

        # if the destination state is not provided, all the possible destination from the current state are considered
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
                        print("[DEBUG HSM] Requested action found, adding request to the queue")

                    # action found, let's save the suggestion
                    action.add_request(signature, args, timestamp=timestamp, uuid=uuid)
                    return True

        # if the action was not found
        if HybridStateMachine.DEBUG:
            print("[DEBUG HSM] Requested action not found")
        return False

    def wait_for_all_actions_that_start_with(self, prefix):
        """Forces the ready flag of all actions whose name start with a given prefix."""

        for state, to_states in self.transitions.items():
            for to_state, action_list in to_states.items():
                for i, action in enumerate(action_list):
                    if action.name.startswith(prefix):
                        action.set_as_not_ready()

    def wait_for_all_actions_that_include_an_arg(self, arg_name):
        """Forces the ready flag of all actions whose name start with a given prefix."""

        for state, to_states in self.transitions.items():
            for to_state, action_list in to_states.items():
                for i, action in enumerate(action_list):
                    if arg_name in action.args:
                        action.set_as_not_ready()

    def wait_for_actions(self, from_state: str, to_state: str, wait: bool = True):
        """Forces the ready flag of a specific action."""

        if from_state not in self.transitions or to_state not in self.transitions[from_state]:
            return False

        for action in self.transitions[from_state][to_state]:
            if wait:
                action.set_as_not_ready()
            else:
                action.set_as_ready()
        return True

    def save(self, filename: str, only_if_changed: object | None = None) -> bool:
        """Save the HSM to a JSON file."""

        if only_if_changed is not None and os.path.exists(filename):
            existing = HybridStateMachine(actionable=only_if_changed).load(filename)
            if str(existing) == str(self):
                return False

        with (open(filename, 'w') as file):
            file.write(str(self))
        return True

    def load(self, filename_or_hsm_as_string: str) -> Self:
        """Load the HSM state from a JSON file and resolve actions."""

        # loading the whole file
        if os.path.exists(filename_or_hsm_as_string) and os.path.isfile(filename_or_hsm_as_string):
            with open(filename_or_hsm_as_string, 'r') as file:
                hsm_data = json.load(file)
        else:
            assert not filename_or_hsm_as_string.endswith(".json"), f"File {filename_or_hsm_as_string} does not exist"
            hsm_data = json.loads(filename_or_hsm_as_string)

        # getting state info
        self.initial_state = hsm_data['initial_state']
        self.state = hsm_data['state']
        self.prev_state = hsm_data['prev_state']
        self.limbo_state = hsm_data['limbo_state']
        self.role = hsm_data.get('role', None)

        # getting states
        self.states = {}
        for state, state_action_list in hsm_data['state_actions'].items():
            if len(state_action_list) == 3:  # backward compatibility
                act_name, act_args, state_id = state_action_list
                waiting_time = 0.
                blocking = True
                msg = None
            elif len(state_action_list) == 4:    # backward compatibility
                act_name, act_args, state_id, blocking = state_action_list
                waiting_time = 0.
                msg = None
            elif len(state_action_list) == 5:    # backward compatibility
                act_name, act_args, state_id, blocking, waiting_time = state_action_list
                msg = None
            else:
                act_name, act_args, state_id, blocking, waiting_time, msg = state_action_list
            self.add_state(state, action=act_name, args=act_args, state_id=state_id,
                           waiting_time=waiting_time, blocking=blocking, msg=msg)

        # getting transitions
        self.transitions = {}
        for from_state, to_states in hsm_data['transitions'].items():
            for to_state, action_list in to_states.items():
                for action_list_tuple in action_list:
                    if len(action_list_tuple) == 4:
                        act_name, act_args, act_ready, act_id = action_list_tuple
                        msg = None
                    else:
                        act_name, act_args, act_ready, act_id, msg = action_list_tuple
                    self.add_transit(from_state, to_state,
                                     action=act_name, args=act_args, ready=act_ready, act_id=act_id, msg=msg)

        return self

    def to_graphviz(self):
        """Encode the HSM in GraphViz format."""

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
        """Save the HSM in GraphViz format, drawn on a PDF file."""

        if filename.lower().endswith(".pdf"):
            filename = filename[0:-4]

        try:
            self.to_graphviz().render(filename, format='pdf', cleanup=True)
            return True
        except Exception:
            return False

    def print_actions(self, state: str | None = None):
        state = (self.state if self.state is not None else self.limbo_state) if state is None else state
        for to_state, action_list in self.transitions[state].items():
            if action_list is None or len(action_list) == 0:
                print(f"{state}, no actions")
            for action in action_list:
                print(f"{state} --> {to_state} {action}")

    # noinspection PyMethodMayBeStatic
    def __policy_first_requested_or_first_ready(self, actions_list: list[Action]) \
            -> tuple[int, tuple[object | None, tuple[dict, float, str | None]]]:
        for i, action in enumerate(actions_list):
            if len(action.get_requests()) > 0:
                return i, next(iter(action.get_requests().items()))
        for i, action in enumerate(actions_list):
            if action.is_ready(consider_requests=False):
                return i, (None, ({}, -1., None))
        return -1, (None, ({}, -1., None))


if __name__ == "__main__":
    class Dummy:
        def __init__(self):
            self.dummy = "[Dummy]"

        def method1(self, a: int, b: float = 3.5):
            print(f"{self.dummy} method1: a: {a}, b: {b}")
            return True

        def method2(self, c: str = "default", d: tuple = ("test", "this")):
            print(f"{self.dummy} method2: c: {c}, d: {d}")
            return True

        def method3(self, f: int):
            print(f"{self.dummy} method3: f: {f}")
            return True

        def method4(self, z: int, steps: int = 3):
            print(f"{self.dummy} method4: z: {z}, steps: {steps}")
            return True

    def checker(request: object) -> bool:
        if not isinstance(request, str):
            return False
        if request.startswith("dave") or request.startswith("paul"):
            return True
        else:
            return False

    dummy = Dummy()

    _hsm1 = HybridStateMachine(actionable=dummy, wildcards={"<hi>": "replaced_by_hello"},
                               request_signature_checker=checker)
    _hsm1.add_state("first", action="method3", args={"f": 3})
    _hsm1.add_transit("init", "first", action="method1", args={"a": 3})
    _hsm1.add_transit("init", "second", action="method2", args={"c": "cat", "d": ["pizza", "style"]})
    _hsm1.add_transit("init", "fourth", action="method4", args={"z": 5, "steps": 3})
    _hsm1.add_transit("first", "third", action="method2", args={"c": "dog"})
    _hsm1.add_transit("third", "second", action="method2", args={"c": "furry"})
    _hsm1.add_transit("second", "first", action="method3", args={"f": 10})
    _hsm1.add_transit("fourth", "second", action="method1", args={"a": 2, "b": 7.6})
    _hsm1.save("test1.json")
    _hsm1.save_pdf("test1.pdf")
    _hsm1.to_graphviz()
    _hsm1.print_actions("fourth")
    print(_hsm1)

    _hsm2 = HybridStateMachine(actionable=dummy)
    _hsm2.load("test1.json")
    _hsm2.save("test2.json")
    _hsm2.save_pdf("test2.pdf")
    _hsm2.to_graphviz()
    print(_hsm2)

    _ret = _hsm1.request_action(signature="dave", action_name="method4", args={"z": 5, "steps": 3})
    print("Request action returned: " + str(_ret))
    for _i in range(0, 10):
        _hsm1.act_states()
        _hsm1.act_transitions()
