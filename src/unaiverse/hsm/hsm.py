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
import html
import time

import graphviz
import importlib.resources
from unaiverse.custom import Custom
from collections.abc import Callable
from unaiverse.hsm.state import State
from unaiverse.utils.logger import log
from unaiverse.hsm.action import Action
from unaiverse.interaction import Interaction, CompletionReason


class HybridStateMachine:

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
        self.add_wildcards(Custom.DEFAULT_WILDCARDS)

        # Applying to the welcome message, state, actions, if needed
        self.apply_wildcards()

        # Forcing output function
        self.__debug_messages_active = False

        self.set_actionable(actionable)

    def show_ticks_in_action_messages(self, do_it: bool = True) -> None:
        """Enables or disables tick symbols appended to action log messages upon completion."""
        self.show_action_completion = do_it

    def show_marks_in_blocking_state_messages(self, do_it: bool = True) -> None:
        """Enables or disables colored markers (🔴/🟢) appended to blocking-state log messages."""
        self.show_blocking_states = do_it

    def show_request_info_in_action_messages(self, do_it: bool = True) -> None:
        """Enables or disables requester/UUID info appended to action log messages."""
        self.show_action_request_info = do_it

    def set_welcome_message(self, msg: str | None) -> None:
        """Sets a message that will be printed only once when the initial state is reached."""

        if msg is not None:
            self.welcome_msg = html.unescape(msg)
            self.welcome_msg_with_wildcards = self.welcome_msg
        else:
            self.welcome_msg = None
            self.welcome_msg_with_wildcards = None

    def to_dict(self) -> dict:
        """Serializes the state machine's current configuration into a dictionary. This includes its states,
        transitions, roles, and the current action being executed. It is useful for saving the state of the machine or
        for logging its status in a structured format.

        Returns:
            A dictionary representation of the state machine's properties.
        """

        # Inverting the organization of the transition matrix, and considering teleport actions only
        teleport_inv_transitions = {}
        action_that_are_teleports = []
        for src, dests in self.transitions.items():
            for dest, actions in dests.items():
                if dest == src:
                    continue  # Skipping self loops for teleports (they should not be there at all, if so, we filter)
                for act in actions:
                    if act.is_teleport():
                        action_that_are_teleports.append(act)
                        if dest not in teleport_inv_transitions:
                            teleport_inv_transitions[dest] = {}
                        if src not in teleport_inv_transitions[dest]:
                            teleport_inv_transitions[dest][src] = []
                        teleport_inv_transitions[dest][src].append(act)

        # Parsing destination states: if the same action connects this to all the possible states, then it is an
        # "all"-like teleport
        teleport_transitions = {Custom.ALL_STATES_NAME: {}}  # Keep "all" on top (it is handled like a new state)
        for dest, srcs in teleport_inv_transitions.items():

            # For each source state reaching the current destination...
            for src, teleports in srcs.items():

                # For each teleport linking source to the current destination...
                for tel in teleports:
                    if tel.get_mark() == "considered":  # Skipping already considered ones
                        continue

                    tel.set_mark("considered")  # Marking
                    tel_to_list = tel.to_list()
                    _found_teleports = [(src, tel)]

                    # Let's see if a teleport equivalent to the considered one connects all sources
                    for _src, _teleports in srcs.items():
                        if src == _src:
                            continue  # Already considered case

                        _found = False
                        for _tel in _teleports:
                            if _tel.get_mark() == "considered":  # Skipping already considered ones
                                continue

                            if tel_to_list == _tel.to_list():
                                _tel.set_mark("considered")  # Marking
                                _found_teleports.append((_src, _tel))
                                _found = True
                                break
                        if not _found:
                            break

                    # Distinguishing
                    if len(_found_teleports) == (len(self.states) - 1):
                        if Custom.ALL_STATES_NAME not in teleport_transitions:
                            teleport_transitions[Custom.ALL_STATES_NAME] = {}
                        if dest not in teleport_transitions[Custom.ALL_STATES_NAME]:
                            teleport_transitions[Custom.ALL_STATES_NAME][dest] = []
                        teleport_transitions[Custom.ALL_STATES_NAME][dest].append(tel)  # Append a single one
                    else:
                        for _src, _tel in _found_teleports:
                            if _src not in teleport_transitions:
                                teleport_transitions[_src] = {}
                            if dest not in teleport_transitions[_src]:
                                teleport_transitions[_src][dest] = []
                            teleport_transitions[_src][dest].append(_tel)

        # Clearing mark
        for tel in action_that_are_teleports:
            tel.clear_mark()

        return {
            'machine': {
                'role': self.role,
                'initial_state': self.initial_state,
                'msg': self.welcome_msg_with_wildcards.encode("ascii", "xmlcharrefreplace").decode(
                        "ascii") if self.welcome_msg_with_wildcards is not None else None,
                'options': {
                    'highlight_blocking_states_in_messages': self.show_blocking_states,
                    'show_action_ticks_after_messages': self.show_action_completion,
                    'show_action_request_after_messages': self.show_action_request_info
                }
            },
            'states': {
                state.name: state.to_dict() for state in self.__id_to_state
            },
            'transitions': {
                from_state: [{
                    "on": act.to_dict(),
                    "goto": to_state
                } for to_state, action_list in to_states.items()
                    for act in action_list if not act.is_teleport()
                ] for from_state, to_states in self.transitions.items() if len(to_states) > 0
            },
            'teleports': {
                from_state: [{
                    "on": act.to_dict(),
                    "goto": to_state
                } for to_state, action_list in to_states.items()
                    for act in action_list
                ] for from_state, to_states in teleport_transitions.items() if len(to_states) > 0
            }
        }

    def __str__(self) -> str:
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
        return json_str

    def set_actionable(self, obj: object) -> None:
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

    def set_policy(self, policy_fcn: Callable[[list[Action]], tuple[int, Interaction | None]]) -> None:
        """Sets the policy to be used in selecting what action to perform in the current state.

        Args:
            policy_fcn: A function that takes a list of `Action` objects that are candidates for execution, and returns
                the index of the selected action and an ActionRequest object with the action-requester details (object,
                arguments, time, and UUID), or -1 and None if no action is selected.
        """
        self.policy = policy_fcn

    def set_policy_filter(self, filter_fcn: Callable[
        [int, Interaction | None, list[Action], dict], tuple[int, Interaction | None]],
                          filter_fcn_opts: dict) -> None:
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

    def set_wildcards(self, wildcards: dict[str, str | float | int] | None, apply: bool = False) -> None:
        """Sets the dictionary of wildcards that are used to dynamically replace placeholder values in action
        arguments.

        Args:
            wildcards: A dictionary containing wildcard key-value pairs.
            apply: True if the wildcards must be applied to whatever uses it (defaults to False)
        """
        wildcards = wildcards if wildcards is not None else {}

        # Saving wildcard dictionary, completely replacing the previous one
        self.wildcards = wildcards

        # Propagating the reference to the new dictionary to actions and states.
        for action in self.__id_to_action:
            action.set_wildcards(self.wildcards)
        for state in self.__id_to_state:
            state.set_wildcards(self.wildcards)

        if apply:
            self.apply_wildcards()

    def apply_wildcards(self) -> None:
        """Given the current wildcards, it applies the replacements they suggest to whatever uses them."""

        # Propagating the reference to the new dictionary to actions and states.
        for action in self.__id_to_action:
            action.apply_wildcards()
        for state in self.__id_to_state:
            state.apply_wildcards()

        # If there is a welcome message that includes wildcards...
        if self.welcome_msg is not None:
            self.welcome_msg = self.welcome_msg_with_wildcards  # Restore before updating
            for wildcard_from, wildcard_to in self.wildcards.items():
                self.welcome_msg = self.welcome_msg.replace(wildcard_from, str(wildcard_to))  # Update

    def set_role(self, role: str) -> None:
        """Sets the role of the agent associated with this state machine. This can be used to influence state machine
        behavior based on the agent's role (e.g., 'teacher', 'student').

        Args:
            role: The string representation of the new role.
        """
        self.role = role
        self.update_wildcard(Custom.ROLE_WILDCARD, self.role)
        self.apply_wildcards()

    def get_wildcards(self) -> dict:
        """Retrieves the dictionary of wildcards currently used by the state machine.

        Returns:
            A dictionary of the wildcards.
        """
        return self.wildcards

    def add_wildcards(self, wildcards: dict[str, str | float | int | list[str]], apply: bool = False) -> None:
        """Adds new key-value pairs to the existing wildcard dictionary.

        Args:
            wildcards: A dictionary of new wildcards to add.
            apply: True if the wildcards must be applied to whatever uses it (defaults to False).
        """

        # Update dictionary
        self.wildcards.update(wildcards)
        if apply:
            self.apply_wildcards()

    def update_wildcard(self, wildcard_key: str, wildcard_value: str | float | int, apply: bool = False) -> None:
        """Updates the value of a single existing wildcard. It raises an error if the key does not exist. This method
        is useful for changing a single dynamic value without redefining all wildcards.

        Args:
            wildcard_key: The key of the wildcard to update.
            wildcard_value: The new value for the wildcard.
            apply: True if the wildcards must be applied to whatever uses it (defaults to False).
        """
        if wildcard_key not in self.wildcards:
            log.error(f"{wildcard_key} is not a valid wildcard")
        else:
            self.wildcards[wildcard_key] = wildcard_value
            if apply:
                self.apply_wildcards()

    def get_action_step_idx(self) -> int:
        """Retrieves the current step index of the action being executed. This is particularly useful for tracking the
        progress of multistep actions.

        Returns:
            An integer representing the current step, or -1 if no action is running.
        """
        return self.__cur_feasible_actions_status['selected_interaction'].get_step_idx() \
            if self.__action is not None else -1

    def is_busy_acting(self) -> bool:
        """Checks if the state machine is currently executing an action. This is determined by checking if the action
        step index is greater than or equal to 0.

        Returns:
            True if an action is running, False otherwise.
        """
        return self.get_action_step_idx() >= 0

    def add_state(self, state: str, action: str = None, args: dict | None = None, state_id: int | None = None,
                  waiting_time: float | None = None, blocking: bool | None = None,
                  msg: str | None = None, msg_action: str | None = None) -> None:
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

    def get_state_name(self, consider_limbo: bool = False) -> str | None:
        """Retrieves the name of the current state of the state machine.

        Args:
            consider_limbo: If True and the current state is None (mid-transition), returns the limbo state
                name instead.

        Returns:
            A string with the state's name, or ``None`` if no state is set.
        """

        if not consider_limbo:
            return self.state
        else:
            if self.state is None and self.limbo_state is not None:
                return self.limbo_state
            else:
                return self.state

    def get_state(self) -> 'State | None':
        """Retrieves the current `State` object of the state machine.

        Returns:
            A `State` object or `None`.
        """
        return self.states[self.state] if self.state is not None else None

    def get_all_states(self) -> list['State']:
        """Retrieves the list of all `State` objects.

        Returns:
            List of `State` objects.
        """
        return self.__id_to_state

    def get_all_actions(self) -> list['Action']:
        """Retrieves the list of all `Action` objects.

        Returns:
            List of `Action` objects.
        """
        return self.__id_to_action

    def get_action(self) -> 'Action | None':
        """Retrieves the `Action` object that is currently being executed.

        Returns:
            An `Action` object or `None`.
        """
        return self.__action

    def get_action_name(self) -> str | None:
        """Retrieves the name of the action currently being executed.

        Returns:
            A string with the action's name, or `None` if no action is running.
        """
        return self.__action.name if self.__action is not None else None

    def get_last_completed_action_name(self) -> str | None:
        """Retrieves the name of the last action that was correctly executed.

        Returns:
            A string with the action's name, or `None` if no actions were executed before.
        """
        return self.__last_completed_action.name if self.__last_completed_action is not None else None

    def reset_state(self) -> None:
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

    def get_states(self) -> list:
        """Returns an iterable of all state names defined in the state machine.

        Returns:
            An iterable of state names.
        """
        return [s.name for s in self.__id_to_state]

    def set_state(self, state: str) -> None:
        """Sets the current state of the state machine to a new, specified state. It also handles the transition logic
        by resetting the current action and updating the previous state. Raises an error if the new state is not known
        to the machine.

        Args:
            state: The name of the state to transition to.
        """
        if state in self.transitions or state in self.states:
            self.limbo_state = None
            self.prev_state = self.state
            self.state = state
            if self.__action is not None:
                self.__cur_feasible_actions_status['selected_interaction'].reset_status()
                self.__action = None
            if self.initial_state is None:
                self.initial_state = state
        else:
            raise ValueError("Unknown state: " + str(state))

    def are_debug_messages_active(self) -> bool:
        """Returns whether debug messages (auto-generated labels, ticks, marks) are currently active."""
        return self.__debug_messages_active

    def set_debug_messages_active(self, yes: bool) -> None:
        """Activates or deactivates debug mode, enabling ticks, blocking marks, request info, and auto-labels.

        Args:
            yes: True to enable debug mode, False to restore original messages and disable extra info.
        """
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

            # Apply wildcards to the restored messages
            self.apply_wildcards()

    def generate_auto_messages(self, states: bool = True, actions: bool = True, force: bool = False) -> None:
        """Auto-generates human-readable messages for states and actions that currently have none.

        Args:
            states: If True, generates messages for states (and their inner actions).
            actions: If True, generates messages for transition actions.
            force: If True, overwrites existing messages, appending the original text in brackets.
        """
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

    def add_global_teleport(self, to_state: str,
                            action: str, args: dict | None = None, ready: bool = True,
                            msg: str | None = None,
                            high_priority: bool = False,
                            total_time: float | str = 0., timeout: float | str = 0., delay: float | str = 0., ) -> None:
        self.add_teleport(from_state=Custom.ALL_STATES_NAME, to_state=to_state, action=action, args=args,
                          ready=ready, act_id=None, high_priority=high_priority,
                          msg=msg, total_time=total_time, timeout=timeout, delay=delay)

    def add_teleport(self, from_state: str, to_state: str,
                     action: str, args: dict | None = None, ready: bool = True,
                     act_id: int | None = None, msg: str | None = None, avoid_changing_ready: bool = False,
                     high_priority: bool = False,
                     total_time: float | str = 0., timeout: float | str = 0., delay: float | str = 0., ) -> None:
        self.add_transit(from_state=from_state, to_state=to_state, action=action, args=args,
                         ready=ready, act_id=act_id, avoid_changing_ready=avoid_changing_ready,
                         msg=msg, teleport=True, high_priority=high_priority,
                         total_time=total_time, timeout=timeout, delay=delay)

    def add_transit(self, from_state: str, to_state: str,
                    action: str, args: dict | None = None, ready: bool = True,
                    act_id: int | None = None, msg: str | None = None,
                    avoid_changing_ready: bool = False,
                    teleport: bool = False,
                    high_priority: bool = False,
                    total_time: float | str = 0., timeout: float | str = 0., delay: float | str = 0.,) -> None:
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
            teleport: A boolean indicating that this transition must be hidden when drawing the machine ('teleport').
            high_priority: A boolean indicating that this transition must be preferred by the policy over others.
            total_time: The number of seconds representing the max duration of this transition (from when it stars).
                When <= 0., then no limits.
            timeout: The number of seconds representing the max time we keep try running this transition before
                giving up. When <= 0., then no timeouts at all.
            delay: The number of seconds that must pass when joining a state before considering this transition.
                When <= 0., then no delays at all.
        """

        # Handling special state names
        if from_state == Custom.ALL_STATES_NAME:
            for from_state_obj in self.__id_to_state:
                if from_state_obj.name == from_state:
                    continue

                # Adding a transition to the dest state
                self.add_transit(from_state=from_state_obj.name, to_state=to_state, action=action, args=args,
                                 ready=ready, act_id=None, avoid_changing_ready=avoid_changing_ready,
                                 msg=msg, teleport=teleport, high_priority=high_priority,
                                 total_time=total_time, timeout=timeout, delay=delay)
            return

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
                             ready=ready, act_id=None, msg=msg, teleport=teleport, high_priority=high_priority,
                             total_time=total_time, timeout=timeout, delay=delay)

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
                            avoid_changing_ready=avoid_changing_ready,
                            teleport=teleport, high_priority=high_priority,
                            total_time=total_time, timeout=timeout, delay=delay)
        self.transitions[from_state][to_state].append(new_action)
        self.__id_to_action.append(new_action)

        new_action.set_state_machine(self)  # This will also share the same wildcards dictionary

    def include(self, hsm, make_a_copy=False) -> None:
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
                    total_time = 0.
                    timeout = 0.
                    delay = 0.
                    if isinstance(_action.get_total_time(), str) or _action.get_total_time() > 0:
                        total_time = _action.get_total_time()
                    if isinstance(_action.get_timeout(), str) or _action.get_timeout() > 0.:
                        timeout = _action.get_timeout()
                    if isinstance(_action.get_delay(), str) or _action.get_delay() > 0.:
                        delay = _action.get_delay()
                    self.add_transit(from_state=_from_state, to_state=_to_state, action=_action.name,
                                     args=_action.args, ready=_action.ready, teleport=_action.is_teleport(),
                                     high_priority=_action.is_high_priority,
                                     act_id=None, msg=_action.msg_with_wildcards, avoid_changing_ready=True,
                                     total_time=total_time, timeout=timeout, delay=delay)

        if make_a_copy:
            self.state = hsm.state
            self.prev_state = hsm.state
            self.initial_state = hsm.initial_state
            self.limbo_state = hsm.limbo_state
            self.set_welcome_message(hsm.welcome_msg_with_wildcards)
            self.show_blocking_states = hsm.show_blocking_states
            self.show_action_completion = hsm.show_action_completion
            self.show_action_request_info = hsm.show_action_request_info

    def must_wait(self) -> bool:
        """Checks if the current state is in a waiting period before any transitions can occur.

        Returns:
            A boolean indicating if the state machine must wait.
        """
        if self.state is not None:
            return self.states[self.state].must_wait()
        else:
            return False

    def is_enabled(self) -> bool:
        """A simple getter to check if the state machine is currently enabled to run.

        Returns:
            True if the state machine is enabled, False otherwise.
        """
        return self.enabled

    def enable(self, yes_or_not: bool) -> None:
        """Enables or disables the state machine. When disabled, the `act_states` and `act_transitions` methods will
        not perform any actions.

        Args:
            yes_or_not: A boolean to enable (`True`) or disable (`False`) the state machine.
        """
        self.enabled = yes_or_not

    async def act_states(self) -> None:
        """Executes the inner action of the current state, if one exists. This method is for actions that occur upon
        entering a state but do not cause an immediate transition. It only runs if the state machine is enabled (async).
        """
        if not self.enabled:
            return

        if self.state is not None:  # When in the middle of an action, the state is Nones
            await self.states[self.state]()  # Run the action (if any)

    async def act_ghost_transition(self, to_state: str) -> int:

        # Forcing to go back to current state if in the middle of a multistep action
        from_state = self.state
        if from_state is None:
            from_state = self.limbo_state
        self.set_state(from_state)  # This will also reset the current interaction (if any)

        # Setting up the ghost action (nop) as only possible action from the current state
        original_transitions = self.transitions[from_state]
        self.transitions[from_state] = \
            {to_state: [Action(name="nop", args={}, actionable=self.actionable, ready=True, teleport=True)]}

        # Acting the ghost action (nop)
        ret = await self.act_transitions()

        # Restoring original transitions
        self.transitions[from_state] = original_transitions
        return ret

    def get_time_spent_in_current_state(self):
        state = self.state
        if state is None:
            state = self.limbo_state
        if state is None:
            return -1.
        else:
            return self.states[self.state].get_time_passed()

    async def act_transitions(self, only_the_ones_with_interactions: bool = False) -> int:
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
                        if self.__action is not None and self.__action == action:
                            self.__action = None
            for i in idx_to_remove:
                del actions_list[i]
                del to_state_list[i]
                del attempts_to_serve_an_interaction_list[i]

        # Using the selected policy to decide what action to apply
        while len(actions_list) > 0:

            # It there was an already selected action (for example a multistep action), then continue with it,
            # otherwise, select a new one following a certain policy (actually, first-come first-served)
            if self.__action is None:
                log.statem(f"List of actions to choose from:\n   " +
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

                        # Keep this part of code: needed to stabilize potential object duplicates (JS)
                        if _interactions_f != _interaction and _interactions_f is not None:
                            _interactions_f = self.actionable.im.get_interaction(uuid=_interactions_f.uuid)
                    except Exception as e:
                        log.statem.error(f"Skipping policy filter due to exception: {e}",
                                         state=self.get_state_name())
                        _idx_f = _idx
                        _interactions_f = _interaction

                    if _idx_f != _idx or _interactions_f != _interaction:
                        _idx = _idx_f
                        _interaction = _interactions_f
                        if _idx < 0:
                            log.error(f"Filter selected no actions among {[a.name for a in actions_list]}")  # TODO statem

                            # No actions were applied
                            self.__cur_feasible_actions_status = None
                            self.__state_changed = False
                            return 2  # Move to the next action
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
                log.error(f"Running action {self.__action.name}, chosen among {[a.name for a in actions_list]}")

                # Forcing system interaction if no external interactions were selected
                if _interaction is None:
                    _interaction = self.__action.system_interaction

                _interaction.reset_state()  # Resetting
                self.__cur_feasible_actions_status['selected_idx'] = _idx
                self.__cur_feasible_actions_status['selected_interaction'] = _interaction

            # References
            action = self.__action
            idx = self.__cur_feasible_actions_status['selected_idx']
            interaction = self.__cur_feasible_actions_status['selected_interaction']

            # If this action has an associated Interaction, set it as current on the IM
            # (this configures stdin/stdout for the action)
            # If the Interaction is (1) None, or (2) a system interaction with no streams at all, or
            # (3) a received interaction with no streams, then the stdin/stdout/... will be set back to default streams
            if self.actionable.im.current is None or self.actionable.im.current != interaction:
                self.actionable.im.set_current(interaction)

            log.statem(f">>> ACTION {self.__action.name}...", state=self.get_state_name(True))
            if len(self.__action.get_list_of_interactions()) > 0:
                log.statem(str(self.__action.get_list_of_interactions()))

            # Calling action, getting back status.
            # Status can be one of these:
            # 0: action fully done;
            # 1: try again this action;
            # 2: failed, move to another action.
            status = await action(interaction=interaction)

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

                # Memorizing tags of data used in this action call
                interaction.record_data_tags()

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
                    await self.actionable.im.complete_current(self.state, CompletionReason.OK)

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
                    #    print(f"Reached state {self.state}, "
                    #          f"propagated these requests taken from {self.__action.name}, and
                    #          now starting from here {propagated_requests}")

                self.states[self.prev_state].reset(self.__state_changed)  # Reset starting time
                interaction.reset_state()
                self.__action = None  # Clearing
                self.__cur_feasible_actions_status = None

                return 0  # Transition done, no need to check other actions!

            elif status == 1:  # Try again the same action (either a new step or an already done-and-failed one)

                # Memorizing tags of data used in this action call
                interaction.record_data_tags()

                # Update status
                self.__state_changed = False
                # if self.prev_state is not None:
                #    self.states[self.prev_state].reset()  # Reset starting time

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
                    del attempts_to_serve_an_interaction_list[idx]

                # Update status
                self.__state_changed = False
                interaction.reset_state()
                self.actionable.im.set_current_as_paused()
                self.__action = None  # Clearing

                continue  # Move to the next action
            else:
                raise ValueError("Unexpected status: " + str(status))

        # No actions were applied
        self.__cur_feasible_actions_status = None
        self.__state_changed = False
        return -1

    async def act(self) -> bool:
        """A high-level method that combines `act_states` and `act_transitions` to run the state machine. It repeatedly
        processes states and transitions until a blocking state is reached or all feasible actions have been tried,
        thus ensuring a complete processing cycle in one call (async).

        Returns:
            True if, during this whole 'act', the state changed at least once.
        """

        # It keeps processing states and actions, until all the current feasible actions fail
        # (also when a step of a multistep action is executed) or a blocking state is reached
        changed_state = False
        starting_state = self.state
        while True:
            if self.welcome_msg is not None and self.state is not None and self.state == self.initial_state:
                log.user(self.welcome_msg)
                self.set_welcome_message(None)

            await self.act_states()
            ret = await self.act_transitions(self.must_wait())
            if self.state != starting_state:
                changed_state = True
            if ret != 0 or (self.state is not None and self.states[self.state].blocking):
                break
        return changed_state

    def get_state_changed(self) -> bool:
        """Returns an internal flag that indicates if a state transition has occurred in the last execution cycle.
        This can be used by an external loop to know when to re-evaluate the state machine's context.

        Returns:
            True if the state has changed, False otherwise.
        """
        return self.__state_changed

    def transition_exists(self, name: str, only_ready: bool = True) -> bool:
        state = self.state if self.state is not None else self.limbo_state
        trans = self.transitions[state]
        for _, action_list in trans.items():
            for action in action_list:
                if (not only_ready or action.is_ready()) and action.name == name:
                    return True
        return False

    def request_action(self, interaction: Interaction | None = None, **kwargs) -> bool:
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
                forced_uuid=kwargs.get('forced_uuid', "do_not_force"),
                id=kwargs.get('id', "random"))

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

    def wait_for_all_actions_that_start_with(self, prefix: str) -> None:
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

    def wait_for_all_actions_that_include_an_arg(self, arg_name: str) -> None:
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

    def save(self, filename: str, only_if_changed: object | None = None) -> bool:
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

    def load(self, filename_or_hsm_as_string: str | io.TextIOWrapper) -> 'HybridStateMachine':
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

        # Backward compatibility
        if "machine" not in hsm_data:
            return self.load_backward_compat(hsm_data)

        # Getting state info
        try:
            self.initial_state = hsm_data['machine']['initial_state']
            self.state = self.initial_state
            self.prev_state = None
            self.limbo_state = None
            self.set_role(hsm_data['machine']['role'])
            self.set_welcome_message(hsm_data['machine'].get('msg', None))
            self.show_blocking_states = (
                hsm_data['machine']['options'].get('highlight_blocking_states_in_messages', False))
            self.show_action_completion = (
                hsm_data['machine']['options'].get('show_action_ticks_after_messages', False))
            self.show_action_request_info = (
                hsm_data['machine']['options'].get('show_action_request_after_messages', False))
        except Exception as e:
            log.critical(f"Invalid JSON data format when loading the state machine [{e}]")

        self.states = {}
        self.transitions = {}

        # Getting states
        if 'states' not in hsm_data:
            log.critical(f"Invalid JSON data format when loading the state machine (missing state list)")

        for state, state_action in hsm_data['states'].items():
            act_name = state_action.get("action", None)
            act_args = state_action.get("action_kwargs", {})
            blocking = state_action.get("blocking", True)
            msg = state_action.get("msg", None)
            waiting_time = state_action.get("time_to_wait_for_interactions", 0.)

            self.add_state(state, action=act_name, args=act_args,
                           waiting_time=waiting_time, blocking=blocking, msg=msg)

        # Getting ordinary transitions (before teleports, to ensure higher priority)
        for from_state, list_of_to_state_dicts in hsm_data['transitions'].items():
            for to_state_dict in list_of_to_state_dicts:
                to_state = to_state_dict["goto"]
                action_dict = to_state_dict["on"]

                act_name = action_dict["action"]
                act_args = action_dict.get("action_kwargs", {})
                msg = action_dict.get("msg", None)
                act_ready = action_dict.get("ready", True)
                high_priority = action_dict.get("high_priority", False)
                total_time = action_dict.get("max_duration", 0.)
                timeout = action_dict.get("retry_timeout", 0.)
                delay = action_dict.get("time_to_wait_before_running", 0.)

                self.add_transit(from_state, to_state,
                                 action=act_name, args=act_args, ready=act_ready, msg=msg,
                                 avoid_changing_ready="ready" in action_dict, total_time=total_time,
                                 high_priority=high_priority,
                                 timeout=timeout, delay=delay)

        # Getting teleports (do it after getting ordinary transitions, so teleports will have lower priority)
        if 'teleports' in hsm_data:

            # Replicating the "all" teleport over all source states
            if Custom.ALL_STATES_NAME in hsm_data['teleports']:
                list_of_to_state_dicts = hsm_data['teleports'][Custom.ALL_STATES_NAME]

                for to_state_dict in list_of_to_state_dicts:
                    to_state = to_state_dict["goto"]
                    action_dict = to_state_dict["on"]

                    act_name = action_dict["action"]
                    act_args = action_dict.get("action_kwargs", {})
                    msg = action_dict.get("msg", None)
                    act_ready = action_dict.get("ready", True)
                    high_priority = action_dict.get("high_priority", False)
                    total_time = action_dict.get("max_duration", 0.)
                    timeout = action_dict.get("retry_timeout", 0.)
                    delay = action_dict.get("time_to_wait_before_running", 0.)

                    # Looping over all states
                    for from_state_obj in self.__id_to_state:
                        from_state = from_state_obj.name

                        # Skipping the destination state
                        if from_state == to_state:
                            continue

                        self.add_transit(from_state, to_state,
                                         action=act_name, args=act_args, ready=act_ready, msg=msg,
                                         avoid_changing_ready="ready" in action_dict, teleport=True,
                                         high_priority=high_priority,
                                         total_time=total_time, timeout=timeout, delay=delay)

            # Handling all the ordinary teleports
            for from_state, list_of_to_state_dicts in hsm_data['teleports'].items():

                # Skipping "all"-like teleport (already handled)
                if from_state == Custom.ALL_STATES_NAME:
                    continue

                for to_state_dict in list_of_to_state_dicts:
                    to_state = to_state_dict["goto"]
                    action_dict = to_state_dict["on"]

                    act_name = action_dict["action"]
                    act_args = action_dict.get("action_kwargs", {})
                    msg = action_dict.get("msg", None)
                    act_ready = action_dict.get("ready", True)
                    high_priority = action_dict.get("high_priority", False)
                    total_time = action_dict.get("max_duration", 0.)
                    timeout = action_dict.get("retry_timeout", 0.)
                    delay = action_dict.get("time_to_wait_before_running", 0.)

                    self.add_transit(from_state, to_state,
                                     action=act_name, args=act_args, ready=act_ready, msg=msg,
                                     avoid_changing_ready="ready" in action_dict, teleport=True,
                                     high_priority=high_priority,
                                     total_time=total_time, timeout=timeout, delay=delay)

        return self

    def load_backward_compat(self, hsm_data: dict) -> 'HybridStateMachine':
        """Loads a state machine's configuration from a JSON file or a JSON string. It reconstructs the states,
        actions, and transitions from the serialized data. This method is critical for persistence and for loading
        pre-defined state machine models.

        Args:
            hsm_data: The dict with an HSM loaded from a JSON file.

        Returns:
            The loaded `HybridStateMachine` object (self).
        """

        # Getting state info
        self.initial_state = hsm_data['initial_state']
        self.state = hsm_data['state']
        self.prev_state = hsm_data['prev_state']
        self.limbo_state = hsm_data['limbo_state']
        self.set_role(hsm_data.get('role', 'unknown'))
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

    def to_graphviz(self) -> graphviz.Digraph:
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
            j = 1
            for to_state, action_list in to_states.items():
                for action in action_list:
                    args = action.args
                    s = "("
                    for i, (k, v) in enumerate(args.items()):
                        s += str(k) + "=" + (str(v) if not isinstance(v, str) else ("'" + str(v) + "'"))
                        if i < len(args) - 1:
                            s += ", "
                    s += ")"

                    special_args = {}
                    if isinstance(action.get_total_time(), str) or action.get_total_time() > 0:
                        total_time = action.get_total_time()
                        special_args['max_duration'] = total_time
                    if isinstance(action.get_timeout(), str) or action.get_timeout() > 0.:
                        timeout = action.get_timeout()
                        special_args['retry_timeout'] = timeout
                    if isinstance(action.get_delay(), str) or action.get_delay() > 0.:
                        delay = action.get_delay()
                        special_args['delay'] = delay

                    if len(special_args) > 0:
                        s += "\n["
                        for i, (k, v) in enumerate(special_args.items()):
                            s += str(k) + "=" + (str(v) if not isinstance(v, str) else ("'" + str(v) + "'"))
                            if i < len(special_args) - 1:
                                s += ", "
                        s += "]"

                    label = str(j) + "." + action.name + s
                    j += 1
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
                    edge_color = "#00000050" if action.is_teleport() else "#000000"
                    graph.edge(from_state, to_state, label=" " + label + " ", fontsize='8',
                               style='dashed' if not action.is_ready() else 'solid',
                               color=edge_color,
                               fontcolor=edge_color,
                               _attributes={'id': "edge" + str(action.id)})
        return graph

    def save_pdf(self, filename: str) -> bool:
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

    def print_actions(self, state: str | None = None) -> None:
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
    def __policy_first_requested_or_first_ready(self, actions_list: list[Action]) -> tuple[int, Interaction | None]:  # noqa: E501
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
            if action.is_high_priority:
                _list_of_requests = action.get_list_of_interactions()
                _selected_action_idx = i
                _selected_interaction = _list_of_requests.get_oldest_interaction() \
                    if len(_list_of_requests) > 0 else None
                return _selected_action_idx, _selected_interaction
        for i, action in enumerate(actions_list):
            _list_of_interactions = action.get_list_of_interactions()
            if len(_list_of_interactions) > 0:
                _selected_action_idx = i
                _selected_interaction = _list_of_interactions.get_oldest_interaction()
                return _selected_action_idx, _selected_interaction
        for i, action in enumerate(actions_list):
            if action.is_ready(consider_interactions=False):
                _selected_action_idx = i
                _selected_interaction = None
                return _selected_action_idx, _selected_interaction
        _selected_action_idx = -1
        _selected_interaction = None
        return _selected_action_idx, _selected_interaction
