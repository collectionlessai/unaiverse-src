"""What an action needs from a request before it runs on it (the ``needs`` and ``defaults_have_priority`` options of
the @action decorator).

A request binds streams of the requester in three groups (stdin, stdtar, stdext). By default an action waits for
fresh data on all of them. ``process`` reads only stdin, so it waits only for stdin: a stream of the requester that
matched an output of the processor (a learning target) or nothing does not block it. An agent that gives priority to
the defaults of its processor inputs (a person) also runs it when only some stdin streams are fresh, the inputs with
a default value taking it, as long as one is fresh.
"""
import pytest

from unaiverse.agent import Agent, action
from unaiverse.agent_basics import AgentBasics
from unaiverse.interaction import Interaction, InteractionManager
from unaiverse.streams import Stream
from unaiverse.streams.dataprops import DataProps

pytestmark = pytest.mark.unit


# Helpers

def _stream(name: str, data_type: str = "text") -> Stream:
    """A stream of the requester, as the receiver knows it.

    Args:
        name: Its name.
        data_type: Its data type.

    Returns:
        The stream.
    """
    return Stream(props=DataProps(name=name, data_type=data_type, delta=-1))


def _request(action_name: str, stdin: dict, stdtar: dict | None = None, stdext: dict | None = None,
             im: object = None) -> Interaction:
    """A request received from peerA, with its streams already matched.

    Args:
        action_name: The action asked.
        stdin: The stdin streams, by user hash.
        stdtar: The stdtar streams, by user hash.
        stdext: The stdext streams, by user hash.
        im: The interaction manager (None: the usual rule, all the streams).

    Returns:
        The request.
    """
    request = Interaction(action_name=action_name, requester="peerA", target="peerB")
    request.set_manager(im, stdin, stdtar or {}, stdext or {}, {})
    return request


def _send(stream: Stream, request: Interaction, data: object, tag: int) -> None:
    """A sample of the requester for this request arrives on a stream.

    Args:
        stream: The stream.
        request: The request.
        data: The sample.
        tag: Its data tag.
    """
    stream.set(data, data_tag=tag, uuid=request.uuid, check_delta=False)


class _FakeAgent:
    """The back-reference of the interaction manager: the rule of an action and the inputs of the processor."""

    def __init__(self, rule: tuple, optional_inputs: list[bool]):
        self.rule = rule
        self.proc_optional_inputs = [{"has_default": d, "default_value": None} for d in optional_inputs]
        self.all_agents = {"peerA"}

    def action_freshness_rule(self, action_name: str | None) -> tuple:
        """The rule of every action (the one given to the fake).

        Args:
            action_name: The action.

        Returns:
            The rule.
        """
        return self.rule


# The decorator and the rule of each action

def test_process_needs_stdin_only_and_the_other_actions_need_everything():
    """process says needs=("stdin",); learn and the other actions say nothing (all the streams)."""
    assert Agent.__dict__["process"]._action_needs == ("stdin",)
    assert Agent.__dict__["learn"]._action_needs is None
    assert Agent.__dict__["record"]._action_needs is None


def test_the_decorator_refuses_an_unknown_group():
    """needs takes only stdin, stdtar and stdext."""
    with pytest.raises(ValueError):
        action(needs=("stdout",))(lambda self: True)


def test_an_override_that_says_nothing_keeps_the_rule_of_the_overridden_action():
    """A world that overrides process with a plain @action (the turing guest) keeps needs=("stdin",); a person gives
    priority to the defaults; an override can still say something else."""

    class Base:
        @action(needs=("stdin",))
        async def process(self, interaction=None):
            return True

    class Override(Base):
        @action
        async def process(self, interaction=None):
            return True

    class Strict(Base):
        @action(needs=("stdin", "stdtar"), defaults_have_priority=False)
        async def process(self, interaction=None):
            return True

    agent = Override()
    agent.proc_defaults_have_priority = False
    assert AgentBasics.action_freshness_rule(agent, "process") == (("stdin",), False)
    agent.proc_defaults_have_priority = True
    assert AgentBasics.action_freshness_rule(agent, "process") == (("stdin",), True)
    assert AgentBasics.action_freshness_rule(agent, "learn") == (None, False)

    strict = Strict()
    strict.proc_defaults_have_priority = False
    assert AgentBasics.action_freshness_rule(strict, "process") == (("stdin", "stdtar"), False)


def test_the_priority_of_the_agent_does_not_reach_actions_that_need_everything():
    """The defaults belong to the processor inputs: an action that needs all the streams is not touched by them."""

    class Base:
        @action
        async def record(self, interaction=None):
            return True

    agent = Base()
    agent.proc_defaults_have_priority = True
    assert AgentBasics.action_freshness_rule(agent, "record") == (None, False)


# The freshness of a request

def test_a_request_without_rule_waits_for_all_its_streams():
    """The usual rule: a stream of the request without fresh data blocks it (unchanged)."""
    text, file = _stream("text"), _stream("file", "file")
    request = _request("process", {"peerA:text": text}, stdtar={"peerA:file": file})
    _send(text, request, "hello", 1)
    assert request.has_enough_fresh_data() is False
    _send(file, request, None, 1)
    assert request.has_enough_fresh_data() is True


def test_process_is_not_blocked_by_a_stream_matched_to_an_output():
    """A text-only message to an agent with a file output: the file stream of the requester became a learning target
    (stdtar) and has no sample, but process waits only for stdin."""
    text, file = _stream("text"), _stream("file", "file")
    im = InteractionManager(_FakeAgent(rule=(("stdin",), False), optional_inputs=[False]))
    request = _request("process", {"peerA:text": text}, stdtar={"peerA:file": file}, im=im)
    assert request.has_enough_fresh_data() is False  # Nothing yet
    _send(text, request, "hello", 1)
    assert request.has_enough_fresh_data() is True
    assert im.check_if_doable(request) is True


def test_an_agent_without_priority_waits_for_all_its_stdin_streams():
    """Two inputs, the second with a default value: without priority both must be fresh (no race between messages)."""
    text, file = _stream("text"), _stream("file", "file")
    im = InteractionManager(_FakeAgent(rule=(("stdin",), False), optional_inputs=[False, True]))
    request = _request("process", {"peerA:text": text, "peerA:file": file}, im=im)
    _send(text, request, "hello", 1)
    assert request.has_enough_fresh_data() is False


def test_a_person_runs_with_only_some_of_its_inputs():
    """A person (text, image, file, all, all with a default value) gets a text-only message: it runs; with nothing
    fresh it does not."""
    text, img, file = _stream("text"), _stream("img", "img"), _stream("file", "file")
    im = InteractionManager(_FakeAgent(rule=(("stdin",), True), optional_inputs=[True, True, True, True]))
    stdin = {"peerA:text": text, "peerA:img": img, "peerA:file": file, "<default_input_pos_3>": None}
    request = _request("process", stdin, im=im)
    assert request.has_enough_fresh_data() is False  # Nothing fresh at all
    _send(text, request, "hello", 1)
    assert request.has_enough_fresh_data() is True


def test_an_input_without_a_default_must_be_fresh_even_with_priority():
    """Priority to the defaults does not touch an input that has none."""
    text, file = _stream("text"), _stream("file", "file")
    im = InteractionManager(_FakeAgent(rule=(("stdin",), True), optional_inputs=[True, False]))
    request = _request("process", {"peerA:text": text, "peerA:file": file}, im=im)
    _send(text, request, "hello", 1)
    assert request.has_enough_fresh_data() is False
    _send(file, request, None, 1)
    assert request.has_enough_fresh_data() is True


def test_a_request_with_no_needed_stream_keeps_the_usual_rule():
    """A process request whose streams are all outside stdin waits for them, as before."""
    ext = _stream("ext")
    im = InteractionManager(_FakeAgent(rule=(("stdin",), False), optional_inputs=[False]))
    request = _request("process", {}, stdext={"peerA:ext": ext}, im=im)
    assert request.has_enough_fresh_data() is False
    _send(ext, request, "x", 1)
    assert request.has_enough_fresh_data() is True


def test_a_consumed_sample_is_not_fresh_again():
    """Once a step recorded the tags it saw, the same sample does not make the request ready again."""
    text, file = _stream("text"), _stream("file", "file")
    im = InteractionManager(_FakeAgent(rule=(("stdin",), False), optional_inputs=[False]))
    request = _request("process", {"peerA:text": text}, stdtar={"peerA:file": file}, im=im)
    _send(text, request, "hello", 1)
    request.record_data_tags(request.get_new_stream_data_tags())
    assert request.has_enough_fresh_data() is False
    _send(text, request, "again", 2)
    assert request.has_enough_fresh_data() is True
