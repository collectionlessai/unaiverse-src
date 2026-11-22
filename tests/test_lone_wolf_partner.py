"""The <partner> of the lone-wolf behavior: where the turns an agent takes on its own are sent.

It is the public peer that got in touch last, whether this agent dialed it or was dialed (add_agent is the one place
both go through), then the public peer whose request it accepted last; when the partner leaves, the public peer that
got in touch last among the others, or the placeholder when none is left (a send to it fails cleanly). The answers to
requests go to whoever asked, whatever the partner. A person's agent is left alone: whoever drives it (the browser,
the interactive mode) sets the partner to the peer the person writes to.
"""

import asyncio
import pytest
from unaiverse.clock import clock
from unaiverse.custom import Custom
from inproc_harness import InProcSwitchboard, InProcNodeConn, HarnessClock, build_real_agent_node, ROLE_PUBLIC


@pytest.fixture
def harness_clock():
    """A deterministic clock for the agents, the real one put back after the test."""
    saved = getattr(clock, "_Clock__instance", None)
    fc = HarnessClock()
    clock.set(fc)
    fc.next_cycle()
    yield fc
    setattr(clock, "_Clock__instance", saved)


def _echo(x: object, first: bool = False, last: bool = False) -> object:
    """Echo processor.

    Args:
        x: The input.
        first: Unused.
        last: Unused.

    Returns:
        The input.
    """
    return x


def _build(sb: InProcSwitchboard, name: str) -> tuple:
    """A public lone-wolf agent on the in-process switchboard.

    Args:
        sb: The switchboard.
        name: The name of the node.

    Returns:
        The node, the agent and its conn.
    """
    conn = InProcNodeConn(f"node-{name}", f"{name}-pub", f"{name}-world", sb)
    node, agent = build_real_agent_node(f"{name}-pub", proc=_echo, proc_inputs=["text"], proc_outputs=["text"],
                                        conn=conn, node_id=f"node-{name}", world_peer_id=f"{name}-world",
                                        role="public_agent")
    agent.behav.enable(False)  # The public peer id, not the world one (see test_agent_chat_e2e.py)
    return node, agent, conn


async def _link(conn: InProcNodeConn, agent: object, other: object, name: str) -> None:
    """Makes the other agent a public peer of this one (the conn tables, then add_agent).

    Args:
        conn: The conn of this agent.
        agent: This agent.
        other: The other agent.
        name: The name of the other node.
    """
    conn.register_peer(f"{name}-pub", node_id=f"node-{name}", role=ROLE_PUBLIC, pool=conn.IN_PUBLIC)
    assert await agent.add_agent(f"{name}-pub", other.node_profile)


def _partner(agent: object) -> str:
    """The <partner> of the lone-wolf behavior of an agent.

    Args:
        agent: The agent.

    Returns:
        Its value.
    """
    return agent.behav_lone_wolf.get_wildcards()[Custom.PARTNER_WILDCARD]


def _send_targets(agent: object) -> list:
    """The targets of the send actions of the lone-wolf behavior, as resolved now.

    Args:
        agent: The agent.

    Returns:
        The targets.
    """
    targets = []
    for by_state in agent.behav_lone_wolf.transitions.values():
        for actions in by_state.values():
            for action in actions:
                if action.name == "send":
                    targets.append(action.args.get("target"))
    return targets


def test_the_partner_is_the_peer_that_got_in_touch_last_and_follows_accepted_requests(harness_clock) -> None:
    """A server dialed by two clients: the last one is its partner, then the one whose request it accepted."""
    sb = InProcSwitchboard()
    node_s, server, conn_s = _build(sb, "S")
    _, client_a, conn_a = _build(sb, "A")
    _, client_b, _ = _build(sb, "B")

    async def _run() -> None:
        assert _partner(server) == Custom.PARTNER_WILDCARD, "nobody yet"
        await _link(conn_s, server, client_a, "A")
        assert _partner(server) == "A-pub"
        await _link(conn_s, server, client_b, "B")
        assert _partner(server) == "B-pub", "the last that got in touch"
        assert _send_targets(server) == ["B-pub"], "the send of its own turns is resolved to it"

        # A asks the server something: once the request is accepted, A is the partner
        await _link(conn_a, client_a, server, "S")
        assert await client_a.send(action_name="process", target="S-pub", data_samples=["hi"])
        await node_s._Node__handle_network_messages()
        assert _partner(server) == "A-pub"
        assert _send_targets(server) == ["A-pub"]

    asyncio.run(_run())


def test_when_the_partner_leaves_the_last_other_peer_takes_its_place_then_the_placeholder(harness_clock) -> None:
    """The partner leaving hands over to the public peer that got in touch last among the others."""
    sb = InProcSwitchboard()
    _, server, conn_s = _build(sb, "S")
    _, client_a, _ = _build(sb, "A")
    _, client_b, _ = _build(sb, "B")

    async def _run() -> None:
        await _link(conn_s, server, client_a, "A")
        await _link(conn_s, server, client_b, "B")
        await server.remove_agent("A-pub")
        assert _partner(server) == "B-pub", "not the partner: nothing changes"
        await server.remove_agent("B-pub")
        assert _partner(server) == Custom.PARTNER_WILDCARD, "nobody left: the placeholder"
        assert _send_targets(server) == [Custom.PARTNER_WILDCARD]
        await _link(conn_s, server, client_a, "A")
        await _link(conn_s, server, client_b, "B")
        await server.remove_agent("B-pub")
        assert _partner(server) == "A-pub", "the partner left: the last other one"

    asyncio.run(_run())


def test_the_partner_of_a_person_is_left_to_whoever_drives_it(harness_clock) -> None:
    """A person's agent (the browser) keeps the partner it was given: peers getting in touch, their requests and their
    leaving change nothing (the browser sets it to the peer the person writes to, at every message)."""
    sb = InProcSwitchboard()
    node_p, person, conn_p = _build(sb, "P")
    _, client_a, conn_a = _build(sb, "A")
    _, client_b, _ = _build(sb, "B")
    person.node_profile.get_static_profile()["node_type"] = person.HUMAN
    assert person.is_human()

    async def _run() -> None:
        await _link(conn_p, person, client_a, "A")
        assert _partner(person) == Custom.PARTNER_WILDCARD, "connecting does not choose for the person"

        # What the browser does before sending what the person typed to A
        person.behav_lone_wolf.update_wildcard(Custom.PARTNER_WILDCARD, "A-pub")
        person.behav_lone_wolf.apply_wildcards()
        await _link(conn_p, person, client_b, "B")
        assert _partner(person) == "A-pub"
        await _link(conn_a, client_a, person, "P")
        assert await client_a.send(action_name="process", target="P-pub", data_samples=["hi"])
        await node_p._Node__handle_network_messages()
        assert _partner(person) == "A-pub"
        await person.remove_agent("A-pub")
        assert _partner(person) == "A-pub", "not even when that peer leaves"

    asyncio.run(_run())
