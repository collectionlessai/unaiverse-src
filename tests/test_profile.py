"""Characterization tests for unaiverse.networking.node.profile.NodeProfile.

These tests document the CURRENT behavior of NodeProfile: construction from
static/dynamic/cv data, the required-key validation, the from_dict round-trip,
the getter shapes and the connection-change flag.

Network access performed by ``_fill_missing_specs`` (public IP lookup) is
neutralized so the tests stay deterministic and never touch the real network.
``psutil`` / ``platform`` calls are local and are exercised for real.
"""
import json
import hashlib

import pytest

from unaiverse.networking.node import profile as profile_module
from unaiverse.networking.node.profile import NodeProfile


# Keys that MUST be present in the static dict or __init__ raises ValueError.
REQUIRED_STATIC_KEYS = [
    "node_id",
    "node_type",
    "node_name",
    "node_description",
    "created_utc",
    "name",
    "surname",
    "title",
    "organization",
    "email",
    "max_nr_connections",
]

# Keys the constructor tolerates as missing (patched / backward-compat).
OPTIONAL_STATIC_KEYS = [
    "allowed_node_ids",
    "world_masters_node_ids",
    "certified",
    "inspector_node_id",
    "location",
    "location_method",
]

# Full set of static keys present in the internal template.
ALL_STATIC_KEYS = set(REQUIRED_STATIC_KEYS) | set(OPTIONAL_STATIC_KEYS)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Make every requests.get inside the profile module fail fast.

    With all HTTP lookups raising, ``_get_public_ip_address`` deterministically
    returns its sentinel string and no real network call is performed.
    """

    def boom(*args, **kwargs):
        raise profile_module.requests.exceptions.RequestException("no network in tests")

    monkeypatch.setattr(profile_module.requests, "get", boom)


def make_static(**overrides):
    """Builds a fresh, minimal-but-valid static profile dict."""
    static = {
        "node_id": "node-1",
        "node_type": "agent",
        "node_name": "TestNode",
        "node_description": "a test node",
        "created_utc": "2024-01-01T00:00:00+00:00",
        "name": "Mario",
        "surname": "Rossi",
        "title": "Dr",
        "organization": "UNISI",
        "email": "mario@example.com",
        "max_nr_connections": 8,
    }
    static.update(overrides)
    return static


def make_profile(static=None, dynamic=None, cv=None):
    """Convenience constructor for a NodeProfile with sane defaults."""
    return NodeProfile(
        static=make_static() if static is None else static,
        dynamic={} if dynamic is None else dynamic,
        cv=[] if cv is None else cv,
    )


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def test_minimal_static_constructs():
    """A minimal valid static dict (plus empty dynamic/cv) builds a profile."""
    profile = make_profile()
    assert isinstance(profile, NodeProfile)


def test_static_template_keys_present():
    """The static profile always exposes the full template key set."""
    profile = make_profile()
    static = profile.get_static_profile()
    assert set(static.keys()) == ALL_STATIC_KEYS


def test_provided_static_values_are_stored():
    """Values passed in the static dict are copied verbatim into the profile."""
    profile = make_profile()
    static = profile.get_static_profile()
    assert static["node_id"] == "node-1"
    assert static["node_type"] == "agent"
    assert static["email"] == "mario@example.com"
    assert static["max_nr_connections"] == 8


def test_extra_static_keys_are_kept():
    """Static keys outside the template are preserved, not dropped."""
    profile = make_profile(static=make_static(custom_field="hello"))
    assert profile.get_static_profile()["custom_field"] == "hello"


def test_empty_static_raises():
    """An empty static dict is rejected before any key check."""
    with pytest.raises(ValueError, match="Missing static profile data"):
        NodeProfile(static={}, dynamic={}, cv=[])


@pytest.mark.parametrize("missing_key", REQUIRED_STATIC_KEYS)
def test_missing_required_static_key_raises(missing_key):
    """Omitting any required static key raises ValueError naming that key."""
    static = make_static()
    del static[missing_key]
    with pytest.raises(ValueError) as exc_info:
        NodeProfile(static=static, dynamic={}, cv=[])
    assert missing_key in str(exc_info.value)


@pytest.mark.parametrize("optional_key", OPTIONAL_STATIC_KEYS)
def test_missing_optional_static_key_is_tolerated(optional_key):
    """Optional/back-compat keys may be absent without raising."""
    static = make_static()
    static.pop(optional_key, None)
    profile = NodeProfile(static=static, dynamic={}, cv=[])
    assert isinstance(profile, NodeProfile)


def test_backward_compat_location_defaults():
    """When absent, location defaults to {} and location_method to 'manual'."""
    profile = make_profile()
    static = profile.get_static_profile()
    assert static["location"] == {}
    assert static["location_method"] == "manual"


def test_untouched_optional_keys_default_to_none():
    """Optional keys not provided remain None in the static profile."""
    profile = make_profile()
    static = profile.get_static_profile()
    assert static["certified"] is None
    assert static["allowed_node_ids"] is None
    assert static["world_masters_node_ids"] is None
    assert static["inspector_node_id"] is None


# --------------------------------------------------------------------------- #
# Dynamic profile filling
# --------------------------------------------------------------------------- #
def test_dynamic_template_structure():
    """The dynamic profile exposes the nested connections/world_summary dicts."""
    dynamic = make_profile().get_dynamic_profile()
    assert isinstance(dynamic["connections"], dict)
    assert set(dynamic["connections"].keys()) == {
        "public_agents",
        "world_agents",
        "world_masters",
        "world_peer_id",
        "role",
    }
    assert isinstance(dynamic["world_summary"], dict)
    assert "world_title" in dynamic["world_summary"]


def test_specs_are_auto_filled():
    """_fill_missing_specs populates the machine-level spec fields."""
    dynamic = make_profile().get_dynamic_profile()
    assert isinstance(dynamic["os"], str) and dynamic["os"]
    assert isinstance(dynamic["timestamp"], str) and dynamic["timestamp"]
    assert isinstance(dynamic["memory_gb"], float)


def test_public_ip_sentinel_when_network_down():
    """With HTTP mocked to fail, the public IP falls back to its sentinel."""
    dynamic = make_profile().get_dynamic_profile()
    assert dynamic["public_ip_address"] == "Public IP not available."


def test_guessed_location_follows_manual_location():
    """With manual location_method, guessed_location mirrors the static location."""
    static = make_static(location={"city": "Siena"}, location_method="manual")
    dynamic = make_profile(static=static).get_dynamic_profile()
    assert dynamic["guessed_location"] == {"city": "Siena"}


def test_provided_dynamic_scalar_is_kept():
    """A provided dynamic scalar overrides the None template default."""
    dynamic = make_profile(dynamic={"peer_id": "QmPeer123"}).get_dynamic_profile()
    assert dynamic["peer_id"] == "QmPeer123"


def test_provided_dynamic_does_not_get_overwritten_by_specs():
    """A provided spec field (os) is not clobbered by _fill_missing_specs."""
    dynamic = make_profile(dynamic={"os": "CustomOS-1.0"}).get_dynamic_profile()
    assert dynamic["os"] == "CustomOS-1.0"


def test_connections_nested_merge():
    """A provided connections sub-key fills the matching None template slot."""
    dynamic = make_profile(
        dynamic={"connections": {"role": "world_master"}}
    ).get_dynamic_profile()
    assert dynamic["connections"]["role"] == "world_master"
    assert dynamic["connections"]["public_agents"] is None


def test_world_summary_nested_merge():
    """A provided world_summary sub-key fills the matching None template slot."""
    dynamic = make_profile(
        dynamic={"world_summary": {"world_title": "W1"}}
    ).get_dynamic_profile()
    assert dynamic["world_summary"]["world_title"] == "W1"


def test_tmp_prefixed_dynamic_key_is_kept():
    """Dynamic keys with a 'tmp_' prefix are stored even if not in the template."""
    dynamic = make_profile(dynamic={"tmp_scratch": 42}).get_dynamic_profile()
    assert dynamic["tmp_scratch"] == 42


def test_unknown_dynamic_key_is_ignored():
    """A non-template, non-tmp_ dynamic key is silently dropped."""
    dynamic = make_profile(dynamic={"totally_unknown": 1}).get_dynamic_profile()
    assert "totally_unknown" not in dynamic


# --------------------------------------------------------------------------- #
# CV handling
# --------------------------------------------------------------------------- #
def test_cv_entries_sorted_by_last_edit_utc():
    """CV entries are reordered ascending by their last_edit_utc."""
    cv = [
        {"last_edit_utc": 5, "role": "a"},
        {"last_edit_utc": 2, "role": "b"},
    ]
    stored = make_profile(cv=cv).get_cv()
    assert [entry["last_edit_utc"] for entry in stored] == [2, 5]


def test_cv_entry_keys_are_sorted():
    """Each CV entry has its keys reordered alphabetically."""
    cv = [{"last_edit_utc": 1, "zeta": "z", "alpha": "a"}]
    stored = make_profile(cv=cv).get_cv()
    assert list(stored[0].keys()) == ["alpha", "last_edit_utc", "zeta"]


def test_empty_cv_round_trips():
    """An empty CV stays an empty list."""
    assert make_profile(cv=[]).get_cv() == []


def test_update_cv_replaces_without_resorting():
    """update_cv overwrites the stored CV verbatim (no re-sort)."""
    profile = make_profile(cv=[{"last_edit_utc": 1, "role": "x"}])
    new_cv = [{"last_edit_utc": 9}, {"last_edit_utc": 3}]
    profile.update_cv(new_cv)
    assert profile.get_cv() == new_cv


# --------------------------------------------------------------------------- #
# from_dict round-trip
# --------------------------------------------------------------------------- #
def test_from_dict_round_trips():
    """from_dict rebuilds a profile and preserves the provided sub-structures."""
    combined = {
        "static": make_static(),
        "dynamic": {"peer_id": "QmAbc", "private_peer_id": "QmPriv"},
        "cv": [{"last_edit_utc": 7, "role": "lead"}],
    }
    profile = NodeProfile.from_dict(combined)

    static = profile.get_static_profile()
    assert static["node_id"] == "node-1"
    assert static["organization"] == "UNISI"

    dynamic = profile.get_dynamic_profile()
    assert dynamic["peer_id"] == "QmAbc"
    assert dynamic["private_peer_id"] == "QmPriv"

    assert profile.get_cv() == [{"last_edit_utc": 7, "role": "lead"}]


def test_get_all_profile_structure():
    """get_all_profile returns the three top-level sections as live references."""
    profile = make_profile()
    allp = profile.get_all_profile()
    assert set(allp.keys()) == {"static", "dynamic", "cv"}
    assert allp["static"] is profile.get_static_profile()
    assert allp["dynamic"] is profile.get_dynamic_profile()
    assert allp["cv"] is profile.get_cv()


def test_from_dict_missing_node_id_raises():
    """from_dict requires a truthy node_id in the static section."""
    static = make_static()
    del static["node_id"]
    combined = {"static": static, "dynamic": {}, "cv": []}
    with pytest.raises(ValueError, match="node_id"):
        NodeProfile.from_dict(combined)


def test_from_dict_empty_node_id_raises():
    """An empty-string node_id is treated as missing by from_dict."""
    combined = {"static": make_static(node_id=""), "dynamic": {}, "cv": []}
    with pytest.raises(ValueError, match="node_id"):
        NodeProfile.from_dict(combined)


def test_from_dict_without_static_section_raises():
    """from_dict asserts the presence of the 'static' section."""
    with pytest.raises(AssertionError):
        NodeProfile.from_dict({"dynamic": {}, "cv": []})


# --------------------------------------------------------------------------- #
# Connection-change flag
# --------------------------------------------------------------------------- #
def test_connections_flag_starts_false():
    """A fresh profile reports no connection change."""
    assert make_profile().connections_changed() is False


def test_mark_change_in_connections_sets_flag():
    """mark_change_in_connections flips the flag to True."""
    profile = make_profile()
    profile.mark_change_in_connections()
    assert profile.connections_changed() is True


def test_unmark_change_in_connections_clears_flag():
    """unmark_change_in_connections resets the flag to False."""
    profile = make_profile()
    profile.mark_change_in_connections()
    profile.unmark_change_in_connections()
    assert profile.connections_changed() is False


# --------------------------------------------------------------------------- #
# CV hashing
# --------------------------------------------------------------------------- #
def test_verify_cv_hash_matches_computed():
    """verify_cv_hash accepts the blake2b hash of the stored CV."""
    profile = make_profile(cv=[{"last_edit_utc": 3, "role": "r"}])
    stored = profile.get_cv()
    expected = hashlib.blake2b(
        json.dumps(stored).encode("utf-8"), digest_size=16
    ).hexdigest()

    match, (provided, computed) = profile.verify_cv_hash(expected)
    assert match is True
    assert provided == expected
    assert computed == expected


def test_verify_cv_hash_rejects_wrong_hash():
    """verify_cv_hash reports a mismatch but still returns the computed hash."""
    profile = make_profile(cv=[{"last_edit_utc": 3, "role": "r"}])
    match, (provided, computed) = profile.verify_cv_hash("deadbeef")
    assert match is False
    assert provided == "deadbeef"
    assert computed != "deadbeef"


# --------------------------------------------------------------------------- #
# check_and_update_specs
# --------------------------------------------------------------------------- #
def test_check_and_update_specs_update_only_returns_false():
    """update_only=True merges specs unconditionally and reports no change."""
    profile = make_profile()
    assert profile.check_and_update_specs(update_only=True) is False
    # The merge keeps the public-IP sentinel coming from the mocked network.
    assert profile.get_dynamic_profile()["public_ip_address"] == "Public IP not available."
