"""The role tables after an agent has been in more than one world.

``ROLE_BITS_TO_STR`` and ``ROLE_STR_TO_BITS`` are class attributes of ``AgentBasics``: every agent of the process
reads the same two dicts, and ``augment_roles`` rewrites them in place at every join. When the augmented tables were
built on the live ones, a second world inherited the roles of the first: a role name present in both worlds kept the
bits of the first, and the peers of the second world were shown with the role names of the first (the participant
list of the webapp read them through ``Node.js`` in the browser).
"""
import asyncio

import pytest

from unaiverse.agent_basics import AgentBasics

pytestmark = pytest.mark.unit


@pytest.fixture
def agent():
    """An agent shell that reads the role tables the way a real agent does (class attributes, no copy).

    The shared tables are put back as they were afterwards, so no other test sees the roles joined here.
    """
    saved_bits_to_str = dict(AgentBasics.ROLE_BITS_TO_STR)
    saved_str_to_bits = dict(AgentBasics.ROLE_STR_TO_BITS)
    shell = object.__new__(AgentBasics)
    yield shell
    AgentBasics.ROLE_BITS_TO_STR.clear()
    AgentBasics.ROLE_BITS_TO_STR.update(saved_bits_to_str)
    AgentBasics.ROLE_STR_TO_BITS.clear()
    AgentBasics.ROLE_STR_TO_BITS.update(saved_str_to_bits)


def join(agent, world_roles):
    """What the node does on a join: build the tables for the world's roles and install them on the agent."""
    role_bits_to_str, role_str_to_bits = AgentBasics.build_augmented_roles_dictionaries(world_roles)
    agent.augment_roles(role_bits_to_str, role_str_to_bits)


def test_a_second_world_shows_its_own_role_names(agent):
    join(agent, ["host", "player"])
    join(agent, ["player"])

    world_agent_player = AgentBasics.ROLE_WORLD_AGENT | (1 << 2)
    assert agent.ROLE_BITS_TO_STR[1 << 2] == "player"
    assert agent.ROLE_BITS_TO_STR[world_agent_player] == "world_agent~player"
    assert agent.ROLE_STR_TO_BITS["world_agent~player"] == world_agent_player


def test_no_role_of_the_previous_world_is_left(agent):
    join(agent, ["host", "player"])
    join(agent, ["player"])

    names = set(agent.ROLE_BITS_TO_STR.values()) | set(agent.ROLE_STR_TO_BITS)
    assert not any("host" in name for name in names)
    assert 1 << 3 not in agent.ROLE_BITS_TO_STR


def test_a_world_without_custom_roles_leaves_the_default_roles_only(agent):
    join(agent, ["host", "player"])
    join(agent, [])

    assert agent.ROLE_BITS_TO_STR == AgentBasics.BASE_ROLE_BITS_TO_STR
    assert agent.ROLE_STR_TO_BITS == AgentBasics.BASE_ROLE_STR_TO_BITS


def test_building_the_tables_does_not_touch_the_shared_ones(agent):
    before_bits_to_str = dict(AgentBasics.ROLE_BITS_TO_STR)
    before_str_to_bits = dict(AgentBasics.ROLE_STR_TO_BITS)

    AgentBasics.build_augmented_roles_dictionaries(["host", "player"])

    assert AgentBasics.ROLE_BITS_TO_STR == before_bits_to_str
    assert AgentBasics.ROLE_STR_TO_BITS == before_str_to_bits


class _NodeConnStub:
    """The two calls of the node connection that leaving a world makes."""

    def reset_rendezvous_tag(self):
        pass

    async def unsubscribe(self, peer_id, channel=None):
        pass


async def _nothing():
    pass


def test_leaving_a_world_restores_the_default_roles(agent):
    join(agent, ["host", "player"])
    agent.CUSTOM_ROLES.extend(["host", "player"])

    # Only the role tables are under test: the rest of the cleanup is stubbed out
    agent.reset_agent_status_attrs = lambda: None
    agent._AgentBasics__remove_all_world_private_streams = _nothing
    agent._AgentBasics__remove_all_world_related_agents = _nothing
    agent.proc_human_peer_id_to_interaction = {}
    agent.node_conn = _NodeConnStub()
    saved_custom_roles = list(AgentBasics.CUSTOM_ROLES)
    try:
        asyncio.run(agent.clear_world_related_data("world-peer"))

        assert agent.ROLE_BITS_TO_STR == AgentBasics.BASE_ROLE_BITS_TO_STR
        assert agent.ROLE_STR_TO_BITS == AgentBasics.BASE_ROLE_STR_TO_BITS
        assert agent.CUSTOM_ROLES == []
    finally:
        AgentBasics.CUSTOM_ROLES[:] = saved_custom_roles
