"""The message of a transition is printed once per series of attempts, not again at each retry of a failing action.

Seen live: a connection action that kept failing printed its message at every tick (about ten a second in the chat of
the world, in a browser). The message is printed again once the action is done, or when the machine comes back to the
state the action starts from.
"""
import asyncio
import pytest

from unaiverse.hsm.hsm import HybridStateMachine

pytestmark = pytest.mark.unit


class _IM:
    """The few methods of the InteractionManager that the state machine calls."""

    def __init__(self):
        self.current = None

    def set_current(self, interaction) -> None:
        self.current = interaction

    def set_current_as_paused(self) -> None:
        pass

    def get_interaction(self, uuid=None):
        return None

    async def complete_current(self, state, reason) -> None:
        self.current = None


class _Agent:
    """An agent whose `connect` fails until `connected` is set, and whose `leave` always succeeds."""

    def __init__(self):
        self.im = _IM()
        self.connected = False

    async def connect(self) -> bool:
        return self.connected

    async def leave(self) -> bool:
        return True

    async def nop(self) -> bool:
        return True


@pytest.fixture
def printed(monkeypatch):
    from unaiverse.hsm import action as action_module
    lines: list[str] = []
    monkeypatch.setattr(action_module.log, "user", lambda msg, *args, **kwargs: lines.append(msg))
    return lines


def _machine(agent: _Agent) -> HybridStateMachine:
    hsm = HybridStateMachine(agent)
    hsm.add_state("init", blocking=True)
    hsm.add_state("connected", blocking=True)
    hsm.add_transit("init", "connected", action="connect", msg="Connecting...")
    hsm.add_transit("connected", "init", action="leave")
    hsm.set_state("init")
    return hsm


def test_a_failing_action_prints_its_message_once_for_all_its_retries(printed):
    asyncio.run(_test_a_failing_action_prints_its_message_once_for_all_its_retries(printed))


async def _test_a_failing_action_prints_its_message_once_for_all_its_retries(printed):
    agent = _Agent()
    hsm = _machine(agent)
    for _ in range(20):
        await hsm.act()
    assert printed.count("Connecting...") == 1
    assert hsm.get_state_name() == "init"

    agent.connected = True
    await hsm.act()
    assert hsm.get_state_name() == "connected"
    assert printed.count("Connecting...") == 1


def test_the_message_is_printed_again_when_the_machine_comes_back_to_the_state(printed):
    asyncio.run(_test_the_message_is_printed_again_when_the_machine_comes_back_to_the_state(printed))


async def _test_the_message_is_printed_again_when_the_machine_comes_back_to_the_state(printed):
    agent = _Agent()
    hsm = _machine(agent)
    for _ in range(5):
        await hsm.act()
    agent.connected = True
    await hsm.act()  # init -> connected
    agent.connected = False
    await hsm.act()  # connected -> init
    assert hsm.get_state_name() == "init"
    for _ in range(5):
        await hsm.act()
    assert printed.count("Connecting...") == 2


def test_a_self_loop_that_succeeds_prints_its_message_at_every_run(printed):
    asyncio.run(_test_a_self_loop_that_succeeds_prints_its_message_at_every_run(printed))


async def _test_a_self_loop_that_succeeds_prints_its_message_at_every_run(printed):
    agent = _Agent()
    hsm = HybridStateMachine(agent)
    hsm.add_state("loop", blocking=True)
    hsm.add_transit("loop", "loop", action="nop", msg="Tick")
    hsm.set_state("loop")
    for _ in range(3):
        await hsm.act()
    assert printed.count("Tick") == 3
