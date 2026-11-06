"""Unit tests for the cooperative stepwise compute (area: StepwiseCompute).

These pin the behaviour of ``AgentBasics.stepwise``: a generator-based computation
advanced ONE chunk per HSM retry on the main task, no threads involved. The
machinery mirrors the mark + pedantic dance of ``send(wait_completion=True)``: the
mark carries the in-flight state (generator + deadline), the action is made
pedantic at the first call so a returned False means "retry next tick", and the
pedantic grant is handed back on every terminal exit (completion, chunk exception,
budget give-up). The retry budget travels in the mark as a deadline, so it holds
in every drive mode.

Determinism is structural: chunks advance only when the test calls the action, so
there is no timing dependence except in the budget test (sub-second budget).

Integration items (status codes on completion / exception / give-up) drive the
REAL ``Action.__call__`` on a real ``Interaction`` bound via ``set_action_ref``;
surgical items call ``stepwise`` directly.
"""

import gc
import time
import asyncio

import pytest

from unaiverse.agent import Agent, action
from unaiverse.hsm.action import Action
from unaiverse.custom import GenException
from unaiverse.interaction import Interaction

pytestmark = pytest.mark.unit


def _run(coro):
    return asyncio.run(coro)


class ProbeAgent(Agent):
    """A bare offline agent whose single ``probe`` action steps a controllable
    generator: ``total_chunks`` yields then ``core_return``; ``raise_at_chunk``
    makes chunk i raise instead; counters expose factory calls, chunks actually
    run and finally executions (the cleanup guarantee)."""

    def __init__(self):
        super().__init__(proc=None)
        self.total_chunks = 3
        self.core_return = 0.42
        self.raise_at_chunk = None
        self.probe_retry_timeout = 60.0
        self.factory_calls = 0
        self.chunks_run = 0
        self.finally_runs = 0
        self.applied = None

    def _make_gen(self):
        self.factory_calls += 1

        def _gen():
            try:
                for i in range(self.total_chunks):
                    if self.raise_at_chunk is not None and i == self.raise_at_chunk:
                        raise ValueError("boom in chunk")
                    self.chunks_run += 1
                    yield
                return self.core_return
            finally:
                self.finally_runs += 1

        return _gen()

    @action
    async def probe(self, interaction: Interaction | None = None) -> bool:
        done, result = self.stepwise(interaction, self._make_gen,
                                     retry_timeout=self.probe_retry_timeout)
        if not done:
            return False
        self.applied = result
        return True


def _linked_probe(agent):
    """A real 'probe' Interaction bound to a real Action, as the HSM would leave
    it before the first dispatch."""
    inter = Interaction(action_name="probe", requester="R", target=["T"])
    act = Action("probe", {}, agent)
    inter.set_action_ref(act)
    return inter, act


# 1. First run: builds the generator, advances ONE chunk, action made pedantic

def test_first_run_builds_generator_and_advances_one_chunk():
    agent = ProbeAgent()
    inter, act = _linked_probe(agent)

    assert act.is_pedantic() is False  # a plain single-step action is not pedantic
    status = _run(act(inter))

    assert status == 1  # pedantic False -> retry, not discard
    assert inter.get_mark() is not None  # the run state is parked in the mark
    assert act.is_pedantic() is True
    assert agent.factory_calls == 1
    assert agent.chunks_run == 1  # the first call already does one chunk of work


# 2. Polls: exactly one chunk per call, the factory is never re-invoked

def test_each_poll_advances_exactly_one_chunk():
    agent = ProbeAgent()
    inter, act = _linked_probe(agent)

    assert _run(act(inter)) == 1
    assert _run(act(inter)) == 1
    assert _run(act(inter)) == 1

    assert agent.chunks_run == 3  # one per call, cooperative slicing
    assert agent.factory_calls == 1  # same generator across retries
    assert inter.get_mark() is not None  # still in flight (result not returned yet)


# 3. Completion: (True, result) applied, mark cleared, pedantic restored, cleanup ran

def test_completion_applies_result_and_restores_state():
    agent = ProbeAgent()
    inter, act = _linked_probe(agent)

    for _ in range(agent.total_chunks):
        assert _run(act(inter)) == 1
    status = _run(act(inter))  # the generator returns on this call

    assert status == 0
    assert agent.applied == 0.42
    assert inter.get_mark() is None
    assert act.is_pedantic() is False  # restored to the non-pedantic default
    assert agent.finally_runs == 1  # the generator's own cleanup executed


# 4. A chunk that raises: cleaned up and re-raised -> a single discard

def test_chunk_exception_becomes_single_discard():
    agent = ProbeAgent()
    agent.raise_at_chunk = 1
    inter, act = _linked_probe(agent)

    assert _run(act(inter)) == 1  # chunk 0 fine
    status = _run(act(inter))  # chunk 1 raises -> @action swallows to False

    assert status == 2  # non-pedantic again -> discard, not an endless retry loop
    assert agent.applied is None
    assert inter.get_mark() is None
    assert act.is_pedantic() is False
    assert agent.finally_runs == 1  # finally ran at raise time


# 5. Budget: past the deadline the poll gives up in one discard; a fresh
#    interaction restarts from scratch (the factory is re-invoked)

def test_budget_exhaustion_discards_and_restarts_from_scratch():
    agent = ProbeAgent()
    agent.total_chunks = 10_000
    agent.probe_retry_timeout = 0.1
    inter, act = _linked_probe(agent)

    assert _run(act(inter)) == 1
    time.sleep(0.2)  # exceed the budget between two polls
    status = _run(act(inter))  # deadline hit -> GenException -> single discard

    assert status == 2
    assert inter.get_mark() is None
    # The runtime-granted pedantic is restored on the give-up path too: the next
    # failure of this action must be an instant discard, not a retry loop.
    assert act.is_pedantic() is False

    # The abandoned generator is collected once unreferenced, running its finally.
    gc.collect()
    assert agent.finally_runs == 1

    # A fresh interaction re-runs from scratch with ITS OWN result: the distinct
    # value proves nothing stale is reused.
    agent.total_chunks = 2
    agent.probe_retry_timeout = 60.0
    agent.core_return = 7.7
    inter2, act2 = _linked_probe(agent)
    assert _run(act2(inter2)) == 1
    assert _run(act2(inter2)) == 1
    assert _run(act2(inter2)) == 0
    assert agent.applied == 7.7
    assert agent.factory_calls == 2


# 6. interaction=None: the generator is drained inline, blocking

def test_none_interaction_drains_inline():
    agent = ProbeAgent()

    done, result = agent.stepwise(None, agent._make_gen)

    assert done is True
    assert result == 0.42
    assert agent.chunks_run == agent.total_chunks
    assert agent.finally_runs == 1


# 7. A pedantic-by-default action (configured retry_timeout) is never touched

def test_pedantic_by_default_action_is_not_touched():
    agent = ProbeAgent()
    inter = Interaction(action_name="probe", requester="R", target=["T"])
    act = Action("probe", {}, agent, retry_timeout=30.)
    inter.set_action_ref(act)
    assert act.is_pedantic_by_default() is True

    assert _run(act(inter)) == 1
    assert act.get_timeout() == 30.  # first call must not clobber the configured budget

    for _ in range(agent.total_chunks):
        status = _run(act(inter))
    assert status == 0
    assert act.is_pedantic() is True  # completion must not strip it either
    assert act.get_timeout() == 30.
    assert agent.applied == 0.42


# 8. Multi-step interaction: a healthy completion keeps the framework-forced
#    default. Driven through stepwise directly: the Action-level step machinery
#    cannot host a step-0 stepwise at all (see the companion constraint test).

def test_multi_step_completion_keeps_forced_default():
    agent = ProbeAgent()
    inter = Interaction(action_name="probe", requester="R", target=["T"], num_steps=3)
    act = Action("probe", {}, agent)
    inter.set_action_ref(act)  # forces the DEFAULT_TIMEOUT pedantic on multi-step
    assert act.is_pedantic() is True
    assert act.is_pedantic_by_default() is False
    forced = act.get_timeout()

    done = False
    while not done:
        done, result = agent.stepwise(inter, agent._make_gen)
    assert result == 0.42

    # The forced default survives the completion: the remaining steps of this same
    # interaction keep their retry semantics.
    assert act.is_pedantic() is True
    assert act.get_timeout() == forced


def test_multi_step_step0_stepwise_is_discarded_by_step_semantics():
    """Usage constraint, pinned: a multi-step interaction whose FIRST step returns
    False is discarded by Action.__call__ regardless of pedantic (the zero-steps-done
    branch), so a stepwise started at step 0 comes back as an instant discard, not a
    retry. Use stepwise from step >= 1, or on a single-step action (the flagship
    disengage case)."""
    agent = ProbeAgent()
    inter = Interaction(action_name="probe", requester="R", target=["T"], num_steps=3)
    act = Action("probe", {}, agent)
    inter.set_action_ref(act)

    assert _run(act(inter)) == 2  # the first chunk's False at step 0 -> instant discard
    assert inter.get_mark() is None  # Action cleared it; the generator is abandoned
    gc.collect()
    assert agent.finally_runs == 1  # ...and collected, running its cleanup


# 9. A non-positive retry budget is a programming error, rejected up front

def test_non_positive_retry_timeout_is_rejected():
    agent = ProbeAgent()
    inter, act = _linked_probe(agent)

    with pytest.raises(GenException):
        agent.stepwise(inter, agent._make_gen, retry_timeout=0.)

    assert inter.get_mark() is None  # nothing started
    assert agent.factory_calls == 0
