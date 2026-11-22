"""The options of a world join (as role_preference in run(join_world=..., role_preference=...)) travel in the profile
that the joiner offers in the Session handshake, as temporary "tmp_<option>" entries, which the world reads in
assign_role (see the social_learning world, which gives the isolated role this way)."""
import types
from unaiverse.networking.node.node import Node


class _Profile:
    """A profile that gives its live dictionary, as NodeProfile.get_all_profile does."""

    def __init__(self) -> None:
        self.data = {"static": {"node_name": "N"}, "dynamic": {"peer_id": "P"}, "cv": []}

    def get_all_profile(self) -> dict:
        return self.data


def _node(options: dict | None) -> types.SimpleNamespace:
    """A stand-in for the node: its profile and the world join under way, if any."""
    joining = None if options is None else {"world_public_peer_id": None, "options": options}
    return types.SimpleNamespace(profile=_Profile(), joining_world_info=joining)


def test_join_options_become_temporary_profile_entries():
    node = _node({"node_name": "W", "addresses": None, "node_id": None, "role_preference": "student_isolated"})
    offered = Node._offered_profile(node)
    assert offered["dynamic"]["tmp_role_preference"] == "student_isolated"
    assert not any(k in offered["dynamic"] for k in ("tmp_node_name", "tmp_addresses", "tmp_node_id"))
    assert "tmp_role_preference" not in node.profile.data["dynamic"]  # The live profile is left as it is


def test_without_a_join_or_options_the_profile_is_offered_as_it_is():
    for options in (None, {"node_name": "W", "addresses": None, "node_id": None}):
        node = _node(options)
        assert Node._offered_profile(node) is node.profile.data
