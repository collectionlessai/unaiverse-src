"""Characterization tests for ``unaiverse.interaction.Interaction``.

These exercise the pure lifecycle/value methods of the ``Interaction`` class
(construction, serialization round-trip, status predicates, step/arg/mark
helpers) plus the ``InteractionManager.get_recipients`` routing rule. They
document the *current* behavior of the code and must stay green.
"""
import pytest

from unaiverse.custom import Custom, GenException
from unaiverse.interaction import (
    Interaction,
    InteractionManager,
    InteractionStatus,
    CompletionReason,
    InteractionType,
)


# ── Construction ─────────────────────────────────────────────────────────────

def test_init_sets_basic_participant_fields():
    """A freshly built interaction stores requester, wraps target into a list and starts CREATED."""
    inter = Interaction(action_name="proc", requester="peerA", target="peerB")

    assert inter.action_name == "proc"
    assert inter.requester == "peerA"
    # target is shallow-copied into a list even when a single string is passed.
    assert inter.target == ["peerB"]
    assert inter.action_kwargs == {}
    assert inter.status == InteractionStatus.CREATED
    assert inter.created is True
    assert inter.completed is False
    assert inter.completion_reason is None
    # No streams and no data_samples means a non-data interaction with num_steps == -1.
    assert inter.num_steps == -1
    assert inter.data_samples == []


def test_init_generates_random_id_and_uuid():
    """With the default id='random' the id is an 8-char hex and the uuid fuses id and requester."""
    inter = Interaction(action_name="a", requester="peerA", target="peerB")

    assert isinstance(inter.id, str)
    assert len(inter.id) == 8
    assert inter.uuid == inter.id + "_peerA"


def test_init_uuid_uses_str_of_none_requester():
    """When requester is None the uuid embeds the literal string 'None' (build_uuid via str())."""
    inter = Interaction(action_name="a")
    assert inter.requester is None
    assert inter.uuid == inter.id + "_None"
    assert inter.has_dummy_requester() is True


def test_init_explicit_id_is_kept_verbatim():
    """A non-'random' id string is stored verbatim and drives the uuid."""
    inter = Interaction(action_name="a", requester="peerA", id="fixedid")
    assert inter.id == "fixedid"
    assert inter.uuid == "fixedid_peerA"


def test_init_streams_force_single_step():
    """Passing a plain stream list parses into the std* buckets and bumps num_steps to 1."""
    inter = Interaction(action_name="a", requester="peerA", streams=["sA", "sB"])
    assert inter.streams == {"stdin": [], "stdtar": [], "stdext": [], "stdunk": ["sA", "sB"]}
    assert inter.num_steps == 1


def test_init_callback_on_volatile_raises():
    """A callback on a volatile interaction is rejected at construction time."""
    with pytest.raises(GenException):
        Interaction(action_name="a", requester="peerA", callback="cb", volatile=True)


# ── to_dict / from_dict round-trip ──────────────────────────────────────────

def test_to_from_dict_roundtrip_basic_fields():
    """A non-data interaction survives a to_dict/from_dict round-trip with all core fields intact."""
    inter = Interaction(
        action_name="proc",
        action_kwargs={"x": 1, "y": "z"},
        requester="peerA",
        target="peerB",
        from_state="s1",
        to_state="s2",
        timeout=5.0,
        volatile=True,
        id="fixedid",
    )
    # Decorate the completion-related fields so they participate in the round-trip.
    inter.status = InteractionStatus.COMPLETED
    inter.completion_reason = CompletionReason.OK
    inter.destination_state = "done"

    rebuilt = Interaction.from_dict(inter.to_dict())

    assert rebuilt.id == "fixedid"
    assert rebuilt.uuid == "fixedid_peerA"
    assert rebuilt.action_name == "proc"
    assert rebuilt.action_kwargs == {"x": 1, "y": "z"}
    assert rebuilt.requester == "peerA"
    assert rebuilt.target == ["peerB"]
    assert rebuilt.from_state == "s1"
    assert rebuilt.to_state == "s2"
    assert rebuilt.timeout == 5.0
    assert rebuilt.volatile is True
    assert rebuilt.num_steps == -1
    assert rebuilt.status == InteractionStatus.COMPLETED
    assert rebuilt.completion_reason == CompletionReason.OK
    assert rebuilt.destination_state == "done"


def test_to_from_dict_roundtrip_streams():
    """A stream-based interaction round-trips its std* bucket structure and num_steps."""
    inter = Interaction(action_name="a", requester="peerA", streams=["sA", "sB"], id="sid")
    rebuilt = Interaction.from_dict(inter.to_dict())

    assert rebuilt.streams == {"stdin": [], "stdtar": [], "stdext": [], "stdunk": ["sA", "sB"]}
    assert rebuilt.num_steps == 1
    assert rebuilt.data_samples == []


def test_to_from_dict_roundtrip_data_samples():
    """A data-sample interaction round-trips the payload (encoded then decoded) back to the same list."""
    inter = Interaction(action_name="a", requester="peerA", data_samples=["hello", "world"], id="did")
    assert inter.num_steps == 1

    d = inter.to_dict()
    # data_samples is serialized to a JSON payload string inside the dict.
    assert isinstance(d["data_samples"], str)

    rebuilt = Interaction.from_dict(d)
    assert rebuilt.data_samples == ["hello", "world"]
    assert rebuilt.num_steps == 1


def test_to_dict_renames_timeout_to_retry_timeout():
    """to_dict exposes the timeout field under the 'retry_timeout' key."""
    inter = Interaction(action_name="a", requester="peerA", timeout=3.5)
    d = inter.to_dict()
    assert d["retry_timeout"] == 3.5
    assert "timeout" not in d


# ── Status predicates ───────────────────────────────────────────────────────

def test_is_system_matches_system_label():
    """is_system is True exactly when the requester equals the system label constant."""
    sys_inter = Interaction(action_name="a", requester=Custom.SYSTEM_INTERACTION_LABEL)
    user_inter = Interaction(action_name="a", requester="peerA")
    assert sys_inter.is_system() is True
    assert user_inter.is_system() is False


def test_is_completed_tracks_status():
    """is_completed and the completed property both follow the COMPLETED status."""
    inter = Interaction(action_name="a", requester="peerA")
    assert inter.is_completed() is False
    inter.status = InteractionStatus.COMPLETED
    assert inter.is_completed() is True
    assert inter.completed is True


def test_status_properties_reflect_status():
    """The running/paused/created boolean properties mirror the underlying status enum."""
    inter = Interaction(action_name="a", requester="peerA")
    assert inter.created is True and inter.running is False and inter.paused is False
    inter.status = InteractionStatus.RUNNING
    assert inter.running is True and inter.created is False
    inter.status = InteractionStatus.PAUSED
    assert inter.paused is True and inter.running is False


def test_registered_type_predicates():
    """registered_as_* predicates follow the interaction's type field."""
    inter = Interaction(action_name="a", requester="peerA")
    assert inter.registered_as_sent is False
    assert inter.registered_as_received is False
    assert inter.registered_as_lazy() is False
    inter.type = InteractionType.SENT
    assert inter.registered_as_sent is True
    inter.type = InteractionType.RECEIVED
    assert inter.registered_as_received is True
    inter.type = InteractionType.LAZY
    assert inter.registered_as_lazy() is True


def test_is_valid_requires_insertion_order_ids():
    """is_valid is True only once both insertion-order ids are non-negative."""
    inter = Interaction(action_name="a", requester="peerA")
    assert inter.is_valid() is False
    inter.by_insertion_order_id = 0
    assert inter.is_valid() is False
    inter.by_requester_insertion_order_id = 0
    assert inter.is_valid() is True


def test_single_vs_multi_step():
    """is_single_step/is_multi_steps classify by num_steps (negative or 1 is single, >1 is multi)."""
    no_data = Interaction(action_name="a", requester="peerA")  # num_steps == -1
    assert no_data.is_single_step() is True
    assert no_data.is_multi_steps() is False

    multi = Interaction(action_name="a", requester="peerA", num_steps=3)
    assert multi.is_single_step() is False
    assert multi.is_multi_steps() is True


def test_mark_completed_sets_reason_and_status():
    """mark_completed flips status to COMPLETED and records the completion reason and destination."""
    inter = Interaction(action_name="a", requester="peerA", target="peerB")
    inter.mark_completed(CompletionReason.OK, dest_state="done")
    assert inter.status == InteractionStatus.COMPLETED
    assert inter.is_completed() is True
    assert inter.completion_reason == CompletionReason.OK
    assert inter.destination_state == "done"
    # Per-target bookkeeping is populated too.
    assert inter.target_completion_reason == [CompletionReason.OK]
    assert inter.target_destination_state == ["done"]


# ── to_status_dict ──────────────────────────────────────────────────────────

def test_to_status_dict_required_fields_present():
    """to_status_dict exposes the status-update fields, with data_tags None when empty."""
    inter = Interaction(action_name="a", requester="peerA", target="peerB")
    inter.completion_reason = CompletionReason.OK
    inter.destination_state = "done"

    sd = inter.to_status_dict(status_generated_by="peerSelf")

    assert set(sd.keys()) == {
        "status_generated_by", "uuid", "status",
        "completion_reason", "destination_state", "data_tags",
    }
    assert sd["status_generated_by"] == "peerSelf"
    assert sd["uuid"] == inter.uuid
    assert sd["status"] == inter.status.value
    assert sd["completion_reason"] == CompletionReason.OK.value
    assert sd["destination_state"] == "done"
    # No data tags recorded yet, hence None rather than an empty dict.
    assert sd["data_tags"] is None


def test_to_status_dict_none_completion_reason():
    """A still-running interaction reports a None completion_reason in its status dict."""
    inter = Interaction(action_name="a", requester="peerA")
    sd = inter.to_status_dict(status_generated_by="peerSelf")
    assert sd["completion_reason"] is None


# ── Step / arg / mark helpers (cheap pure operations) ────────────────────────

def test_step_idx_helpers():
    """The step-index helpers increment, decrement, set and reset around the -1 baseline."""
    inter = Interaction(action_name="a", requester="peerA")
    assert inter.get_step_idx() == -1
    assert inter.was_at_least_one_step_done() is False

    inter.inc_step_idx()
    assert inter.get_step_idx() == 0
    assert inter.was_at_least_one_step_done() is True

    inter.set_step_idx(5)
    assert inter.get_step_idx() == 5
    inter.dec_step_idx()
    assert inter.get_step_idx() == 4

    inter.reset_state()
    assert inter.get_step_idx() == -1
    assert inter.get_starting_time() == 0.
    assert inter.get_timeout_starting_time() == 0.


def test_was_last_step_done_no_data():
    """For a no-data interaction (num_steps -1) the last step is done once step_idx reaches 0."""
    inter = Interaction(action_name="a", requester="peerA")
    assert inter.was_last_step_done() is False
    inter.inc_step_idx()  # -> 0
    assert inter.was_last_step_done() is True


def test_arg_helpers_set_get_alter():
    """set_arg/get_arg/alter_arg manage action kwargs; get_arg returns None for unknown keys."""
    inter = Interaction(action_name="a", requester="peerA", action_kwargs={"k": 1})
    assert inter.get_arg("k") == 1
    assert inter.get_arg("missing") is None

    inter.set_arg("new", 7)
    assert inter.get_arg("new") == 7

    assert inter.alter_arg("new", 8) is True
    assert inter.get_arg("new") == 8
    # alter_arg refuses to create absent keys.
    assert inter.alter_arg("absent", 9) is False
    assert inter.get_arg("absent") is None


def test_mark_helpers():
    """set_mark/get_mark/clear_mark store and clear an arbitrary marker object."""
    inter = Interaction(action_name="a", requester="peerA")
    assert inter.get_mark() is None
    sentinel = object()
    inter.set_mark(sentinel)
    assert inter.get_mark() is sentinel
    inter.clear_mark()
    assert inter.get_mark() is None


def test_get_total_steps_and_starting_time_setters():
    """get_total_steps echoes num_steps; the time setters feed the matching getters."""
    inter = Interaction(action_name="a", requester="peerA", num_steps=4)
    assert inter.get_total_steps() == 4
    inter.set_starting_time(12.5)
    inter.set_timeout_starting_time(3.0)
    assert inter.get_starting_time() == 12.5
    assert inter.get_timeout_starting_time() == 3.0


# ── Static helpers ──────────────────────────────────────────────────────────

def test_build_uuid_defaults_to_system():
    """build_uuid fuses id and requester, defaulting to the system id/label when None."""
    assert Interaction.build_uuid("abc", "peerA") == "abc_peerA"
    assert Interaction.build_uuid(None, None) == (
        Custom.SYSTEM_INTERACTION_ID + "_" + Custom.SYSTEM_INTERACTION_LABEL
    )
    assert Interaction.build_uuid("x") == "x_" + Custom.SYSTEM_INTERACTION_LABEL


def test_get_id_from_uuid():
    """get_id_from_uuid returns the id portion before the first underscore."""
    assert Interaction.get_id_from_uuid("abc_peerA") == "abc"
    assert Interaction.get_id_from_uuid("sid_None") == "sid"


def test_update_target_resets_per_target_bookkeeping():
    """update_target normalizes to a list and resizes the per-target tracking arrays."""
    inter = Interaction(action_name="a", requester="peerA", target="peerB")
    inter.update_target(["t1", "t2", "t3"])
    assert inter.target == ["t1", "t2", "t3"]
    assert inter.target_completion_reason == [None, None, None]
    assert inter.target_data_tags == [{}, {}, {}]
    assert inter.target_cycle_completed == [-1, -1, -1]


# ── InteractionManager.get_recipients (routing rule) ────────────────────────
# get_recipients lives on InteractionManager, but its routing only consults the
# manager's own sent/received/lazy dicts, so it is exercised here with a minimal
# manager (agent=None) by registering interactions in those dicts directly.

def _bare_manager() -> InteractionManager:
    return InteractionManager(agent=None)


def test_get_recipients_received_returns_requester():
    """For a received interaction the recipient is the original requester, wrapped in a list."""
    im = _bare_manager()
    inter = Interaction(action_name="a", requester="peerA", target="peerB")
    im.received[inter.uuid] = inter
    assert im.get_recipients(inter) == ["peerA"]


def test_get_recipients_sent_returns_targets():
    """For a sent interaction the recipients are the interaction's targets."""
    im = _bare_manager()
    inter = Interaction(action_name="a", requester="peerA", target=["peerB", "peerC"])
    im.sent[inter.uuid] = inter
    assert im.get_recipients(inter) == ["peerB", "peerC"]


def test_get_recipients_lazy_returns_targets():
    """For a lazy interaction the recipients are the interaction's targets."""
    im = _bare_manager()
    inter = Interaction(action_name="a", requester="peerA", target=["peerB"])
    im.lazy[inter.uuid] = inter
    assert im.get_recipients(inter) == ["peerB"]


def test_get_recipients_unknown_falls_back_to_target_and_filters_none():
    """An unregistered interaction falls back to its targets, dropping None entries."""
    im = _bare_manager()
    inter = Interaction(action_name="a", requester="peerA", target="peerB")
    assert im.get_recipients(inter) == ["peerB"]

    none_target = Interaction(action_name="a", requester="peerA")  # target == [None]
    assert im.get_recipients(none_target) == []
