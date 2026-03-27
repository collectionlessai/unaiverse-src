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
import html
import time
from unaiverse.custom import Custom
from unaiverse.utils.logger import log
from unaiverse.hsm.action import Action


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

        if self.name is Custom.NOT_ALLOWED_STATE_NAMES:
            log.critical(f"Invalid state name (not allowed): {self.name}")

    async def __call__(self, *args, **kwargs) -> int | None:
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

    def __str__(self) -> str:
        """Provides a string representation of the `State` object. This is useful for debugging and logging, as it
        summarizes the state's properties, including its name, ID, waiting time, blocking status, and its associated
        action (if any).

        Returns:
            A string containing a formatted summary of the state's instance.
        """
        return (f"[State: {self.name}] id: {self.id}, waiting_time: {self.waiting_time}, blocking: {self.blocking}, "
                f"action -> {self.action if self.action is not None else 'none'}, msg: {self.msg}")

    def set_state_machine(self, hsm: object) -> None:
        """Registers the parent state machine that owns this state."""
        self.state_machine = hsm
        self.set_wildcards(hsm.get_wildcards())
        if self.action is not None:
            self.action.set_state_machine(hsm)

    def set_msg(self, msg: str | None) -> None:
        """Sets the message associated to this state."""

        if msg is not None:
            self.msg = html.unescape(msg)
            self.msg_with_wildcards = self.msg
        else:
            self.msg = None
            self.msg_with_wildcards = None

    def must_wait(self) -> bool:
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

    def to_list(self) -> list:
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

    def to_dict(self) -> dict:
        """Converts the state's properties into a dict. This method is useful for serialization, allowing the state to
        be easily stored or transmitted.

        Returns:
            A dict containing the state's properties.
        """
        d = {}
        if self.action is not None:
            d["action"] = self.action.name
            d["action_kwargs"] = self.action.args
        if self.msg is not None:
            d["msg"] = self.msg.encode("ascii", "xmlcharrefreplace").decode("ascii")
        d["blocking"] = self.blocking
        d["time_to_wait_for_interactions"] = self.waiting_time
        return d

    def has_action(self) -> bool:
        """A simple getter that checks if an action is associated with the state.

        Returns:
            True if an action is set, False otherwise.
        """
        return self.action is not None

    def get_starting_time(self) -> float:
        """Retrieves the timestamp when the state's execution began. This is used to calculate the elapsed waiting time.

        Returns:
            A float representing the starting time.
        """
        return self.starting_time

    def reset(self) -> None:
        """Resets the state's internal counters. This method is typically called when re-entering a state. It sets the
        `starting_time` to zero and also resets the associated action's step counter if an action exists.
        """
        self.starting_time = 0.
        if self.action is not None:
            self.action.system_interaction.reset_state()

    def set_blocking(self, blocking: bool) -> None:
        """Sets the blocking status of the state. A blocking state will prevent the state machine from transitioning to
        the next state until the action is fully completed.

        Args:
            blocking: A boolean value to set the blocking status.
        """
        self.blocking = blocking

    def set_wildcards(self, wildcards: dict[str, str | float | int] | None) -> None:
        """Replaces wildcard dictionary.

        Args:
            wildcards: A dictionary mapping wildcard placeholders to their concrete values.
        """
        self.wildcards = wildcards if wildcards is not None else {}
        if self.action is not None:
            self.action.set_wildcards(self.wildcards)

    def apply_wildcards(self) -> None:
        """A method that replaces placeholder values (wildcards) in the state message.
        It handles both single-value and list-based wildcards.
        """
        if self.msg_with_wildcards is None:
            self.msg_with_wildcards = self.msg
        else:
            self.msg = self.msg_with_wildcards

        if self.msg is not None:
            for wildcard_from, wildcard_to in self.wildcards.items():
                self.msg = self.msg.replace(wildcard_from, str(wildcard_to))

        if self.action is not None:
            self.action.apply_wildcards()
