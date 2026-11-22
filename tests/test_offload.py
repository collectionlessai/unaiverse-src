"""
Phase 4 of the concurrency plan: ModuleWrapper runs its module in a worker thread (aforward / alearn_backward).

- the module calls run in the worker, the agent's answer gate on the caller's thread (first call and re-asks alike);
- the event loop keeps running during a slow forward pass;
- same outputs, same losses, same parameter updates as the sync methods;
- the grad mode of the caller is the one of the worker;
- inline when it must be: modules that echo their input, UAI_OFFLOAD = False, OFFLOAD_ENABLED off, the browser,
  a subclass that overrides forward();
- exceptions come back through the await; a wrapper that ran in its worker can still be pickled (checkpoints).
"""
import time
import copy
import torch
import pickle
import asyncio
import threading
from unaiverse.custom import Custom
from unaiverse.uai.fill import AnswerOutcome
from concurrent.futures import ThreadPoolExecutor
from unaiverse.streams.dataprops import StreamType
from unaiverse.modules.utils import ModuleWrapper, HumanModule


class _Recorder(torch.nn.Module):
    """A text module that records the thread of each call and the grad mode it saw, and can be slow."""

    def __init__(self, delay: float = 0.) -> None:
        """Initializes the module.

        Args:
            delay: Seconds each call sleeps (default: 0).
        """
        super().__init__()
        self.delay = delay
        self.threads = []
        self.grad_modes = []

    def forward(self, text: str) -> str:
        """Records, sleeps, answers.

        Args:
            text: The input.

        Returns:
            The input with a mark.
        """
        self.threads.append(threading.current_thread().name)
        self.grad_modes.append(torch.is_grad_enabled())
        time.sleep(self.delay)
        return f"{text}!"


def _text_wrapper(module: torch.nn.Module) -> ModuleWrapper:
    """Wraps a text module.

    Args:
        module: The module.

    Returns:
        The wrapper.
    """
    return ModuleWrapper(module=module, proc_inputs=[StreamType("text")], proc_outputs=[StreamType("text")])


def test_the_module_runs_in_the_worker_and_the_result_is_the_same() -> None:
    """aforward gives what forward gives, with the module call in the wrapper's worker thread."""
    rec = _Recorder()
    mw = _text_wrapper(rec)
    assert mw.forward("a") == ("a!",)
    assert asyncio.run(mw.aforward("a")) == ("a!",)
    assert rec.threads[0] == threading.current_thread().name
    assert rec.threads[1].startswith("unaiverse-proc")


def test_the_event_loop_keeps_running_during_a_slow_forward() -> None:
    """While a forward pass sleeps 1 s in the worker, another coroutine keeps ticking every 10 ms."""
    mw = _text_wrapper(_Recorder(delay=1.0))

    async def scenario() -> tuple:
        """Runs the slow forward next to a ticker (async).

        Returns:
            The output and how many ticks happened meanwhile.
        """
        ticks = 0
        done = asyncio.Event()

        async def ticker() -> None:
            """Counts 10 ms ticks until the forward is done (async)."""
            nonlocal ticks
            while not done.is_set():
                await asyncio.sleep(0.01)
                ticks += 1
        t = asyncio.create_task(ticker())
        out = await mw.aforward("slow")
        done.set()
        await t
        return out, ticks

    out, ticks = asyncio.run(scenario())
    assert out == ("slow!",) and ticks >= 50, ticks


class _GateAgent:
    """The agent side of the answer gate: asks the module again twice, then lets the answer go; records threads."""

    def __init__(self) -> None:
        """Initializes the agent."""
        self.threads = []
        self.calls = 0

    def uai_preprocess(self, text: str) -> tuple:
        """Renders an interactive block (not used: the input has none).

        Args:
            text: The input.

        Returns:
            The input and no form.
        """
        return text, None

    def uai_pending_form(self) -> dict:
        """A form asked earlier.

        Returns:
            The form.
        """
        self.threads.append(threading.current_thread().name)
        return {"id": "f1"}

    def uai_pending_view(self) -> str:
        """The rendering of the message that carried the form.

        Returns:
            Its text.
        """
        return "the form"

    def uai_postprocess(self, text: str, spec: dict, **kwargs: object) -> AnswerOutcome:
        """Asks again twice, then accepts.

        Args:
            text: The answer.
            spec: The form.
            **kwargs: The other arguments of the gate.

        Returns:
            The decision.
        """
        self.threads.append(threading.current_thread().name)
        self.calls += 1
        return AnswerOutcome(retry=f"again {self.calls}") if self.calls <= 2 else AnswerOutcome()


def test_the_answer_gate_stays_on_the_loop_and_its_re_asks_go_to_the_worker() -> None:
    """A gate that asks twice again: three module calls in the worker, every policy call on the loop's thread."""
    rec = _Recorder()
    mw = _text_wrapper(rec)
    agent = _GateAgent()
    mw.set_agent(agent)
    out = asyncio.run(mw.aforward("q"))
    assert out == ("again 2!",)
    assert len(rec.threads) == 3 and all(t.startswith("unaiverse-proc") for t in rec.threads), rec.threads
    assert agent.threads and all(t == threading.current_thread().name for t in agent.threads), agent.threads


def test_the_worker_follows_the_grad_mode_of_the_caller() -> None:
    """Under no_grad the module sees no grad mode in the worker; otherwise it does."""
    rec = _Recorder()
    mw = _text_wrapper(rec)

    async def scenario() -> None:
        """Calls once with and once without grad (async)."""
        await mw.aforward("g")
        with torch.no_grad():
            await mw.aforward("n")

    asyncio.run(scenario())
    assert rec.grad_modes == [True, False]


def _tensor_wrapper(seed: int) -> ModuleWrapper:
    """A small linear model with its optimizer and loss, deterministic for a seed.

    Args:
        seed: The seed.

    Returns:
        The wrapper.
    """
    torch.manual_seed(seed)
    net = torch.nn.Linear(3, 2)
    mw = ModuleWrapper(module=net,
                       proc_inputs=[StreamType(data_type="tensor", tensor_shape=(None, 3), tensor_dtype=torch.float32)],
                       proc_outputs=[StreamType(data_type="tensor", tensor_shape=(None, 2),
                                                tensor_dtype=torch.float32)],
                       proc_opts={"optimizer": torch.optim.SGD(net.parameters(), lr=0.1),
                                  "losses": [torch.nn.functional.mse_loss]})
    return mw


def test_same_outputs_losses_and_updates_as_the_sync_methods() -> None:
    """Two identical models, one trained through forward/learn_backward, one through aforward/alearn_backward."""
    x, y = torch.ones(4, 3), torch.zeros(4, 2)
    sync_mw, async_mw = _tensor_wrapper(7), _tensor_wrapper(7)
    out_sync = sync_mw.forward(x)
    loss_sync = sync_mw.learn_backward([y])

    async def scenario() -> tuple:
        """The async twin of the two calls (async).

        Returns:
            Outputs and losses.
        """
        out = await async_mw.aforward(x)
        return out, await async_mw.alearn_backward([y])

    out_async, loss_async = asyncio.run(scenario())
    assert torch.allclose(out_sync[0], out_async[0])
    assert [v.item() for v in loss_sync] == [v.item() for v in loss_async]
    for p_sync, p_async in zip(sync_mw.module.parameters(), async_mw.module.parameters()):
        assert torch.equal(p_sync, p_async)


def test_inline_cases(monkeypatch: object) -> None:
    """Echoing modules, UAI_OFFLOAD = False, OFFLOAD_ENABLED off and the browser run on the caller's thread."""
    here = threading.current_thread().name
    human = _text_wrapper(HumanModule())
    assert asyncio.run(human.aforward("hi")) == ("hi",)

    opted_out = _Recorder()
    opted_out.UAI_OFFLOAD = False
    asyncio.run(_text_wrapper(opted_out).aforward("x"))
    assert opted_out.threads == [here]

    rec = _Recorder()
    mw = _text_wrapper(rec)
    monkeypatch.setattr(Custom, "OFFLOAD_ENABLED", False)
    asyncio.run(mw.aforward("x"))
    monkeypatch.setattr(Custom, "OFFLOAD_ENABLED", True)
    monkeypatch.setattr("unaiverse.modules.utils.sys.platform", "emscripten")
    asyncio.run(mw.aforward("x"))
    assert rec.threads == [here, here]


def test_a_subclass_overriding_forward_runs_inline() -> None:
    """A wrapper whose subclass overrides forward() keeps it, called inline by aforward()."""

    class _Custom(ModuleWrapper):
        """A wrapper with its own forward."""

        def forward(self, *args: object, **kwargs: object) -> tuple:
            """Its own forward.

            Args:
                *args: Inputs.
                **kwargs: Keyword arguments.

            Returns:
                The thread it ran in.
            """
            return (threading.current_thread().name,)

    mw = _Custom(module=_Recorder(), proc_inputs=[StreamType("text")], proc_outputs=[StreamType("text")])
    assert asyncio.run(mw.aforward("x")) == (threading.current_thread().name,)


def test_errors_come_back_and_the_worker_is_not_part_of_the_wrapper() -> None:
    """An exception in the worker is raised by the await; the worker is not stored on the wrapper, so what a
    checkpoint saves (the state dict of the module) is unchanged."""

    class _Boom(torch.nn.Module):
        """A module that fails."""

        def forward(self, text: str) -> str:
            """Fails.

            Args:
                text: The input.

            Returns:
                Never.
            """
            raise ValueError("boom")

    mw = _text_wrapper(_Boom())
    try:
        asyncio.run(mw.aforward("x"))
        raise AssertionError("no exception")
    except ValueError as e:
        assert str(e) == "boom"
    used = _tensor_wrapper(3)
    asyncio.run(used.aforward(torch.ones(2, 3)))
    assert not any(isinstance(v, ThreadPoolExecutor) for v in vars(used).values())
    assert set(pickle.loads(pickle.dumps(used.module.state_dict()))) == {"weight", "bias"}


def test_a_wrapper_that_ran_in_its_worker_can_be_pickled_and_copied() -> None:
    """After aforward (its worker exists), a wrapper pickles and deep-copies, and the copies run."""
    used = _tensor_wrapper(5)
    asyncio.run(used.aforward(torch.ones(2, 3)))
    for clone in (pickle.loads(pickle.dumps(used)), copy.deepcopy(used)):
        assert tuple(clone.forward(torch.ones(4, 3))[0].shape) == (4, 2)
        assert tuple(asyncio.run(clone.aforward(torch.ones(1, 3)))[0].shape) == (1, 2)


class _AsyncTextModule(torch.nn.Module):
    """A text module whose forward is async: it waits on the event loop, as a model in a JavaScript worker would."""

    def __init__(self, delay: float = 0.) -> None:
        """Initializes the module.

        Args:
            delay: Seconds each call waits (default: 0).
        """
        super().__init__()
        self.delay = delay
        self.calls = 0

    async def forward(self, text: str) -> str:
        """Waits, answers (async).

        Args:
            text: The input.

        Returns:
            The input with a mark.
        """
        self.calls += 1
        await asyncio.sleep(self.delay)
        return f"{text}?"


def test_an_async_module_is_awaited_and_postprocessed_offloaded_or_not() -> None:
    """aforward waits for what an async module gives back, in the worker thread or inline, and postprocesses it."""
    for offload in (True, False):
        mod = _AsyncTextModule()
        mw = _text_wrapper(mod)
        mw.UAI_OFFLOAD = offload
        assert asyncio.run(mw.aforward("a")) == ("a?",), f"offload {offload}"
        assert mod.calls == 1
        assert mw._last_raw_outputs == ("a?",)


def test_the_event_loop_keeps_running_while_an_async_module_waits() -> None:
    """While an async module waits 0.5 s (no thread involved), another coroutine keeps ticking."""
    mw = _text_wrapper(_AsyncTextModule(delay=0.5))
    mw.UAI_OFFLOAD = False

    async def scenario() -> tuple:
        """Runs the waiting forward next to a ticker (async).

        Returns:
            The output and how many ticks happened meanwhile.
        """
        ticks = 0
        done = asyncio.Event()

        async def ticker() -> None:
            """Counts 10 ms ticks until the forward is done (async)."""
            nonlocal ticks
            while not done.is_set():
                await asyncio.sleep(0.01)
                ticks += 1
        t = asyncio.create_task(ticker())
        out = await mw.aforward("wait")
        done.set()
        await t
        return out, ticks

    out, ticks = asyncio.run(scenario())
    assert out == ("wait?",)
    assert ticks >= 20


def test_forward_refuses_an_async_module_that_waits() -> None:
    """The sync forward() has no event loop: an async module that waits raises (asyncio says no loop is running)."""
    mw = _text_wrapper(_AsyncTextModule(delay=0.01))
    try:
        mw.forward("a")
        assert False, "forward() should refuse a module that waits"
    except RuntimeError:
        pass
