"""Characterization tests for ``unaiverse.modules.utils``.

These pin the CURRENT behavior of the human-in-the-loop module, the generic
``ModuleWrapper`` Stream-based wrapper, and the ``AgentProcessorChecker`` that
normalizes the processor attributes of an agent-like container. Tensors are kept
tiny and no network / time waits happen.
"""

import inspect

import pytest
import torch
from PIL import Image

from unaiverse.modules.utils import (
    HumanModule,
    ModuleWrapper,
    MultiIdentity,
    AgentProcessorChecker,
)
from unaiverse.streams.dataprops import StreamType
from unaiverse.custom import GenException

pytestmark = pytest.mark.unit


# ── Minimal agent-like container ─────────────────────────────────────────────


class FakeAgent:
    """Bare processor container exposing exactly the attributes the checker reads.

    Deliberately NOT callable, so ``AgentProcessorChecker`` does not adopt the
    container itself as the processor when ``proc`` is None.
    """

    def __init__(self, proc=None, proc_inputs=None, proc_outputs=None, proc_opts=None):
        self.proc = proc
        self.proc_inputs = proc_inputs
        self.proc_outputs = proc_outputs
        self.proc_opts = proc_opts
        self.proc_optional_inputs = None


# ── HumanModule.forward ──────────────────────────────────────────────────────


def test_human_module_echoes_text_and_image_dropping_whatever():
    """forward(text, img, whatever) returns just (text, img), ignoring whatever."""
    hm = HumanModule()
    img = Image.new("RGB", (2, 2))
    out = hm.forward("hello", img, "ignored-extra")
    assert out == ("hello", img)
    assert out[1] is img  # The image is echoed unchanged (same object).


def test_human_module_echoes_none_inputs():
    """forward() with all defaults returns the (None, None) pair unchanged."""
    hm = HumanModule()
    assert hm.forward() == (None, None)
    assert hm.forward(text="only-text") == ("only-text", None)


def test_human_module_always_returns_two_tuple():
    """forward always returns a length-2 tuple regardless of inputs."""
    hm = HumanModule()
    out = hm.forward("t", None, None)
    assert isinstance(out, tuple) and len(out) == 2


def test_human_module_is_callable_nn_module():
    """HumanModule is a torch.nn.Module and callable via __call__."""
    hm = HumanModule()
    assert isinstance(hm, torch.nn.Module)
    assert hm("x") == ("x", None)


# ── ModuleWrapper: wrapping behaviors ────────────────────────────────────────


def _text_wrapper(module):
    return ModuleWrapper(
        module=module,
        proc_inputs=[StreamType("text")],
        proc_outputs=[StreamType("text")],
    )


def test_wrapper_human_forward_returns_shaped_tuple():
    """Wrapping a HumanModule with one text in/out yields a length-1 output tuple."""
    mw = _text_wrapper(HumanModule())
    out = mw.forward("hello")
    assert out == ("hello",)
    assert isinstance(out, tuple) and len(out) == len(mw.proc_outputs)


def test_wrapper_forward_echoes_none_input():
    """A None input flows through preprocess/module/postprocess as None."""
    mw = _text_wrapper(HumanModule())
    assert mw.forward(None) == (None,)


def test_wrapper_trivial_callable_forward_returns_shaped_tuple():
    """A plain callable returning a bare value is wrapped into a length-1 tuple."""

    def echo(text):
        return text

    mw = _text_wrapper(echo)
    assert mw.forward("hi") == ("hi",)
    # A non-nn.Module callable is stored verbatim (no .to() move).
    assert mw.module is echo


def test_wrapper_records_last_raw_outputs_before_postprocess():
    """_last_raw_outputs holds the module's raw tuple, here (text, None)."""
    mw = _text_wrapper(HumanModule())
    mw.forward("hello")
    assert mw._last_raw_outputs == ("hello", None)


def test_wrapper_strips_first_and_last_kwargs_when_module_lacks_them():
    """forward silently drops framework 'first'/'last' kwargs HumanModule cannot take."""
    mw = _text_wrapper(HumanModule())
    assert mw._has_first is False and mw._has_last is False
    assert mw.forward("hello", first=True, last=True) == ("hello",)


def test_wrapper_multi_output_shape_matches_proc_outputs():
    """Two text/img in-out slots produce a length-2 tuple, image echoed unchanged."""
    img = Image.new("RGB", (2, 2))
    mw = ModuleWrapper(
        module=HumanModule(),
        proc_inputs=[StreamType("text"), StreamType("img")],
        proc_outputs=[StreamType("text"), StreamType("img")],
    )
    out = mw.forward("hi", img)
    assert len(out) == 2
    assert out[0] == "hi"
    assert out[1] is img


# ── ModuleWrapper: signature / default caching ───────────────────────────────


def test_wrapper_sig_defaults_cached_from_human_forward():
    """_sig_defaults captures one entry per data arg (text, img, whatever), all defaulted."""
    mw = _text_wrapper(HumanModule())
    assert mw._sig_defaults == [
        {"has_default": True, "default_value": None},
        {"has_default": True, "default_value": None},
        {"has_default": True, "default_value": None},
    ]


def test_wrapper_optional_inputs_truncated_to_proc_inputs_length():
    """proc_optional_inputs is sliced to match proc_inputs length (1 here)."""
    mw = _text_wrapper(HumanModule())
    assert mw.proc_optional_inputs == [{"has_default": True, "default_value": None}]


def test_wrapper_optional_inputs_padded_when_proc_inputs_longer():
    """When proc_inputs outnumbers signature data args, the tail is padded no-default."""

    def echo2(text, n=5):
        return text

    mw = ModuleWrapper(
        module=echo2,
        proc_inputs=[StreamType("text")],
        proc_outputs=[StreamType("text")],
    )
    # echo2 has two data args: 'text' (no default) and 'n' (default 5).
    assert mw._sig_defaults == [
        {"has_default": False, "default_value": None},
        {"has_default": True, "default_value": 5},
    ]
    # Grow proc_inputs to three slots: slice gives 2, then pad one no-default entry.
    mw.set_proc_inputs_and_outputs(
        [StreamType("text"), StreamType("text"), StreamType("text")], None
    )
    assert mw.proc_optional_inputs == [
        {"has_default": False, "default_value": None},
        {"has_default": True, "default_value": 5},
        {"has_default": False, "default_value": None},
    ]


def test_wrapper_var_args_callable_has_empty_sig_defaults():
    """A *args/**kwargs module (MultiIdentity) caches no positional defaults."""
    mw = ModuleWrapper(
        module=MultiIdentity(),
        proc_inputs=[StreamType("text")],
        proc_outputs=[StreamType("text")],
    )
    assert mw._sig_defaults == []
    # proc_inputs has length 1, so optional inputs is padded to one no-default entry.
    assert mw.proc_optional_inputs == [{"has_default": False, "default_value": None}]


def test_wrapper_non_callable_module_raises():
    """A module that is neither callable nor has a forward raises GenException."""
    with pytest.raises(GenException):
        ModuleWrapper(
            module=42,
            proc_inputs=[StreamType("text")],
            proc_outputs=[StreamType("text")],
        )


# ── ModuleWrapper: device handling ───────────────────────────────────────────


def test_wrapper_default_device_is_cpu():
    """With no env override and no device arg, the wrapper runs on CPU."""
    mw = _text_wrapper(HumanModule())
    assert mw.device == torch.device("cpu")


def test_wrapper_explicit_string_device():
    """A device passed as a string is normalized into a torch.device."""
    mw = ModuleWrapper(
        module=HumanModule(),
        proc_inputs=[StreamType("text")],
        proc_outputs=[StreamType("text")],
        device="cpu",
    )
    assert mw.device == torch.device("cpu")


def test_guess_device_reads_env_then_override(monkeypatch):
    """guess_device honors PROC_DEVICE, but an explicit device argument wins."""
    mw = _text_wrapper(HumanModule())  # built on CPU
    monkeypatch.setenv("PROC_DEVICE", "cuda")
    mw.guess_device(None)
    assert mw.device.type == "cuda"  # device object only; no tensor is moved
    mw.guess_device("cpu")
    assert mw.device == torch.device("cpu")


# ── AgentProcessorChecker: proc is None → dummy processor ─────────────────────


def test_checker_none_proc_builds_dummy_multiidentity():
    """proc=None yields a ModuleWrapper around MultiIdentity with 'all' I/O streams."""
    agent = FakeAgent(proc=None)
    AgentProcessorChecker(agent)
    assert isinstance(agent.proc, ModuleWrapper)
    assert isinstance(agent.proc.module, MultiIdentity)
    assert agent.proc_inputs[0].is_all() is True
    assert agent.proc_outputs[0].is_all() is True
    assert agent.proc_opts == {"optimizer": None, "losses": [None]}


def test_checker_none_proc_fixes_stream_names():
    """The dummy processor gets canonical proc_input_0 / proc_output_0 names."""
    agent = FakeAgent(proc=None)
    AgentProcessorChecker(agent)
    assert agent.proc_inputs[0].get_name() == "proc_input_0"
    assert agent.proc_outputs[0].get_name() == "proc_output_0"


def test_checker_writes_back_processor_attributes_on_agent():
    """The container's proc_* attributes are made to alias the wrapper's own."""
    agent = FakeAgent(proc=None)
    AgentProcessorChecker(agent)
    assert agent.proc_inputs is agent.proc.proc_inputs
    assert agent.proc_outputs is agent.proc.proc_outputs
    assert agent.proc_opts is agent.proc.proc_opts
    assert agent.proc_optional_inputs is agent.proc.proc_optional_inputs


def test_checker_dummy_processor_runs_on_cpu():
    """The auto-built dummy processor is placed on CPU by the checker."""
    agent = FakeAgent(proc=None)
    AgentProcessorChecker(agent)
    assert agent.proc.device == torch.device("cpu")


# ── AgentProcessorChecker: proc="human" (current behavior) ───────────────────


def test_checker_human_string_currently_raises():
    """proc='human' raises GenException: the string fails the callable validation.

    The HumanModule-wrapping branch (modules/utils.py around lines 767-773) is
    guarded by an earlier 'must be None / nn.Module / ModuleWrapper / callable'
    check that a plain string does not satisfy, so that branch never runs. This
    pins the CURRENT behavior; see the suspected-bug note in the report.
    """
    agent = FakeAgent(proc="human")
    with pytest.raises(GenException):
        AgentProcessorChecker(agent)


def test_checker_human_module_instance_is_wrapped():
    """Passing a HumanModule INSTANCE (an nn.Module) is the working way to get one."""
    agent = FakeAgent(
        proc=HumanModule(),
        proc_inputs=["text", "img"],
        proc_outputs=["text", "img"],
    )
    AgentProcessorChecker(agent)
    assert isinstance(agent.proc, ModuleWrapper)
    assert isinstance(agent.proc.module, HumanModule)


# ── AgentProcessorChecker: string proc_inputs → StreamType ───────────────────


def test_checker_string_proc_inputs_converted_to_streamtype():
    """String entries in proc_inputs/outputs become StreamType with that data_type."""
    agent = FakeAgent(
        proc=HumanModule(),
        proc_inputs=["text", "img"],
        proc_outputs=["text", "img"],
    )
    AgentProcessorChecker(agent)
    assert all(isinstance(p, StreamType) for p in agent.proc_inputs)
    assert all(isinstance(p, StreamType) for p in agent.proc_outputs)
    assert [p.data_type for p in agent.proc_inputs] == ["text", "img"]
    assert [p.data_type for p in agent.proc_outputs] == ["text", "img"]


def test_checker_callable_agent_used_as_processor():
    """When proc is None and the container is callable, the container is the processor."""

    class CallableAgent(FakeAgent):
        def __call__(self, text):
            return text.upper()

    agent = CallableAgent(
        proc=None, proc_inputs=["text"], proc_outputs=["text"]
    )
    AgentProcessorChecker(agent)
    assert isinstance(agent.proc, ModuleWrapper)
    assert agent.proc.module is agent  # The container itself became the module.
    assert agent.proc.forward("ab") == ("AB",)


# ── AgentProcessorChecker: validation errors ─────────────────────────────────


def test_checker_missing_attribute_asserts():
    """A container missing the required processor attributes fails an assertion."""

    class Incomplete:
        proc = None  # but no proc_inputs / proc_outputs / proc_opts / ...

    with pytest.raises(AssertionError):
        AgentProcessorChecker(Incomplete())


def test_checker_invalid_proc_type_raises():
    """A non-callable, non-module proc (an int) is rejected with GenException."""
    agent = FakeAgent(proc=42)
    with pytest.raises(GenException):
        AgentProcessorChecker(agent)


def test_checker_invalid_proc_inputs_not_a_list_raises():
    """proc_inputs that is a bare string (not a list) is rejected with GenException."""
    agent = FakeAgent(proc=HumanModule(), proc_inputs="text")
    with pytest.raises(GenException):
        AgentProcessorChecker(agent)


# ── A couple of cross-checks on the wrapper's parametrization ─────────────────


def test_wrapper_set_proc_inputs_refreshes_optional_inputs_length():
    """set_proc_inputs_and_outputs re-aligns proc_optional_inputs to the new length."""
    mw = _text_wrapper(HumanModule())
    assert len(mw.proc_optional_inputs) == 1
    mw.set_proc_inputs_and_outputs([StreamType("text"), StreamType("img")], None)
    assert len(mw.proc_optional_inputs) == 2
    mw.set_proc_inputs_and_outputs([StreamType("text")], None)
    assert len(mw.proc_optional_inputs) == 1


def test_wrapper_signature_introspection_matches_human_forward():
    """Sanity: HumanModule.forward exposes text/img/whatever, none framework-handled."""
    params = list(inspect.signature(HumanModule().forward).parameters)
    assert params == ["text", "img", "whatever"]
    assert "first" not in params and "last" not in params
