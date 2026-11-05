"""In-process e2e for the engage -> process round -> disengage lifecycle.

Two REAL world-capable ``Agent`` instances are wired back-to-back on a single
``InProcSwitchboard`` (mirroring ``tests/test_inproc_harness_smoke.py`` but with
``role='world_agent'`` and a teacher/student pair). The engage/disengage handlers
are network-independent ``@action`` coroutines, so they are driven DIRECTLY on the
receive-side (student) agent against REAL ``Interaction`` objects and REAL
``world_agents`` populated by ``cross_register``. A real ``_send`` 'process' round
from the teacher exercises the codec round-trip + ``im.sent`` registration + DM
delivery through the harness conn. A purge closure that mirrors ``Node.__purge``
characterizes what ``disengage(disconnect_too=True)`` does to the conn.

Documented boundary (NOT a fidelity loss for these assertions): full-loop
EXECUTION of engage/disengage as HSM actions needs a real world-role HSM. A
harness-built ``world_agent``'s ``behav`` is the empty one-state placeholder, whose
``request_action`` rejects every action, so an engage delivered through
``Node.__handle_network_messages`` is DROPPED before execution (pinned in
``test_full_loop_engage_is_dropped_by_placeholder_behav``). The lifecycle assertions
therefore drive the engage/disengage handlers directly, which is exactly what
``behav.act()`` would ultimately invoke.

BUG 3 (fixed): ``disengage(disconnect_too=True)`` calls
``node_purge_fcn(_requester, reason=WORLD_DISENGAGEMENT)``, so ``Node.__purge``
forwards the reason to ``conn.drop_peer``, which GOAWAYs the Session (recorded on
``conn.closed_sessions``) before dropping the connection. The Level 1 reason
assertion lives in ``tests/test_disengage_goaway.py``; here we assert the end-to-end
effect on the conn (GOAWAY recorded, peer dropped from the pool + membership).
"""

import asyncio

import pytest

from unaiverse.clock import clock
from unaiverse.agent import Agent
from unaiverse.agent_basics import AgentBasics
from unaiverse.interaction import Interaction
from unaiverse.networking.p2p.messages import Msg
from unaiverse.networking.session.sess_msg import WORLD_DISENGAGEMENT

from inproc_harness import (
    InProcSwitchboard, InProcNodeConn, HarnessClock,
    build_real_agent_node, cross_register, ROLE_WORLD_AGENT,
)

pytestmark = pytest.mark.e2e


# Deterministic clock fixture (verbatim from the smoke test)

@pytest.fixture
def harness_clock():
    saved = getattr(clock, "_Clock__instance", None)
    fc = HarnessClock()
    clock.set(fc)
    fc.next_cycle()  # leave cycle 0 active so timestamps are well-defined
    yield fc
    setattr(clock, "_Clock__instance", saved)


# Custom-role registration fixture (snapshot/restore the class-level dicts)

@pytest.fixture
def roles():
    """Register the 'teacher'/'student' custom roles class-wide (so
    'world_agent~teacher' resolves) and restore the dicts afterwards, exactly like
    ``test_agent_actions_pure.py::roled_agent``. ``augment_roles`` /
    ``build_augmented_roles_dictionaries`` mutate the shared ``AgentBasics`` dicts."""
    saved_bits_to_str = dict(AgentBasics.ROLE_BITS_TO_STR)
    saved_str_to_bits = dict(AgentBasics.ROLE_STR_TO_BITS)
    saved_custom = list(AgentBasics.CUSTOM_ROLES)

    helper = Agent(proc=None)
    helper.CUSTOM_ROLES = ["teacher", "student"]
    helper.augment_roles(*AgentBasics.build_augmented_roles_dictionaries(["teacher", "student"]))
    try:
        yield
    finally:
        AgentBasics.ROLE_BITS_TO_STR.clear()
        AgentBasics.ROLE_BITS_TO_STR.update(saved_bits_to_str)
        AgentBasics.ROLE_STR_TO_BITS.clear()
        AgentBasics.ROLE_STR_TO_BITS.update(saved_str_to_bits)
        AgentBasics.CUSTOM_ROLES = saved_custom


def _echo(x):
    """Trivial echo processor (str in -> str out), codec-stable."""
    return x


def _build_world_pair():
    """Build a teacher + student, both REAL world agents, on one switchboard, and
    cross-register them as world agents (each lands in the other's
    ``world_agents`` + ``all_agents``)."""
    sb = InProcSwitchboard()
    conn_t = InProcNodeConn(node_id="node-teacher", public_peer_id="T-pub",
                            world_peer_id="T-world", switchboard=sb)
    conn_s = InProcNodeConn(node_id="node-student", public_peer_id="S-pub",
                            world_peer_id="S-world", switchboard=sb)
    node_t, agent_t = build_real_agent_node(
        "T-pub", proc=_echo, proc_inputs=["text"], proc_outputs=["text"],
        conn=conn_t, node_id="node-teacher", world_peer_id="T-world", role="world_agent")
    node_s, agent_s = build_real_agent_node(
        "S-pub", proc=_echo, proc_inputs=["text"], proc_outputs=["text"],
        conn=conn_s, node_id="node-student", world_peer_id="S-world", role="world_agent")
    cross_register(node_t, agent_t, node_s, agent_s, role=ROLE_WORLD_AGENT)
    return sb, conn_t, conn_s, node_t, agent_t, node_s, agent_s


def _engage_interaction():
    """A REAL engage Interaction from the teacher (only ``.requester`` is read)."""
    return Interaction(action_name="engage",
                       action_kwargs={"sender_role": "world_agent~teacher"},
                       requester="T-world", target=["S-world"])


def _disengage_interaction():
    """A REAL disengage Interaction from the teacher (only ``.requester`` is read)."""
    return Interaction(action_name="disengage", requester="T-world", target=["S-world"])


# (a) engage establishes the relationship

def test_engage_match_establishes_engaged_relationship(harness_clock, roles):
    """A matching engage from the cross-registered teacher adds T-world to the
    student's ``_engaged_agents`` and flips ``_available`` to False."""
    sb, conn_t, conn_s, node_t, agent_t, node_s, agent_s = _build_world_pair()

    # The cross-registration really did seed the gate the handler reads.
    assert "T-world" in agent_s.world_agents
    assert "T-world" in agent_s.all_agents

    # BEFORE: a fresh student is available and engaged to nobody.
    assert agent_s._engaged_agents == set()
    assert agent_s._available is True

    ok = asyncio.run(agent_s.engage(acceptable_role="teacher",
                                    sender_role="world_agent~teacher",
                                    interaction=_engage_interaction()))

    # AFTER: positive mutation of the engaged set + availability flag.
    assert ok is True
    assert agent_s._engaged_agents == {"T-world"}
    assert agent_s._available is False


# (b) a real process round registers im.sent and delivers an INTERACTION

def test_process_round_registers_sent_and_delivers_interaction(harness_clock, roles):
    """After engaging, the teacher's ``_send(action_name='process')`` round (in the
    world context) registers the interaction in ``im.sent`` and delivers exactly one
    INTERACTION Msg into the student's WORLD inbox through the harness conn + codec."""
    sb, conn_t, conn_s, node_t, agent_t, node_s, agent_s = _build_world_pair()

    # BEFORE: a fresh sender's IM has no sent interactions.
    assert agent_t.im.sent == {}

    async def _run():
        # Student engages the teacher first (the relationship under test).
        assert await agent_s.engage(acceptable_role="teacher",
                                    sender_role="world_agent~teacher",
                                    interaction=_engage_interaction()) is True

        # Drive the 'process' send in the WORLD behaving context so the sender's
        # peer-id (the interaction requester) is the world peer-id 'T-world' the
        # student knows, and im.register_sent records it as a non-public interaction.
        # enable() only flips a flag; it is harmless on the placeholder behav.
        agent_t.behav.enable(True)
        try:
            assert agent_t.behaving_in_world() is True
            inter = await agent_t._send(action_name="process", target=["S-world"],
                                        num_steps=1)
        finally:
            agent_t.behav.enable(False)
        return inter

    inter = asyncio.run(_run())

    # POSITIVE: a real Interaction came back, registered under its uuid in im.sent.
    assert inter is not None
    assert inter.requester == "T-world"
    assert inter.uuid in agent_t.im.sent
    assert agent_t.im.sent[inter.uuid] is inter

    # POSITIVE: exactly one INTERACTION Msg landed in the student's WORLD inbox,
    # stamped with the teacher's world peer-id as the sender.
    inbox = conn_s.chat_inbox[InProcNodeConn.P2P_WORLD]
    assert len(inbox) == 1
    staged = inbox[0]
    assert isinstance(staged, Msg)
    assert staged.content_type == Msg.INTERACTION
    assert staged.sender == "T-world"
    # Codec round-trip carried the interaction payload (a fresh dict, no aliasing).
    assert staged.content["action_name"] == "process"
    assert staged.content["requester"] == "T-world"
    assert staged.content["uuid"] == inter.uuid


# (c) graceful disengage leaves the conn entirely untouched

def test_graceful_disengage_clears_relationship_and_leaves_conn_untouched(harness_clock, roles):
    """``disengage(disconnect_too=False)`` discards the requester from
    ``_engaged_agents`` and restores availability, while leaving the membership
    tables AND the conn untouched (no purge: peer still known + connected, no
    GOAWAY recorded)."""
    sb, conn_t, conn_s, node_t, agent_t, node_s, agent_s = _build_world_pair()

    # Establish the relationship first (positive precondition).
    assert asyncio.run(agent_s.engage(acceptable_role="teacher",
                                      sender_role="world_agent~teacher",
                                      interaction=_engage_interaction())) is True
    assert agent_s._engaged_agents == {"T-world"}
    assert agent_s._available is False

    ok = asyncio.run(agent_s.disengage(disconnect_too=False,
                                       interaction=_disengage_interaction()))

    # POSITIVE: the relationship was torn down and availability restored.
    assert ok is True
    assert agent_s._engaged_agents == set()
    assert agent_s._available is True

    # Graceful disengage touches NO conn/membership state: the peer is still a known
    # world agent and still connected, and no GOAWAY was recorded.
    assert "T-world" in agent_s.world_agents
    assert "T-world" in agent_s.all_agents
    assert conn_s.closed_sessions == []
    assert conn_s.is_connected("T-world") is True
    assert conn_s.get_pool_of("T-world") == InProcNodeConn.IN_WORLD_AGENTS


# (d) disconnect_too: disengage drives a GOAWAY on the conn (BUG 3, fixed)

def test_disconnect_too_purges_peer_and_records_goaway(harness_clock, roles):
    """``disengage(disconnect_too=True)`` runs the node purge with
    WORLD_DISENGAGEMENT, so ``Node.__purge`` forwards the reason to
    ``conn.drop_peer``: a GOAWAY is recorded, AND ``remove_agent`` + the pool
    drop remove the peer from the membership tables and the conn pool.

    This is the e2e expression of the BUG 3 fix (agent.py:669). The Level 1 reason
    assertion lives in
    ``tests/test_disengage_goaway.py::test_disengage_disconnect_too_propagates_world_disengagement_reason``;
    here we verify the GOAWAY reaches the conn end to end."""
    sb, conn_t, conn_s, node_t, agent_t, node_s, agent_s = _build_world_pair()

    # Establish the relationship first (positive precondition).
    assert asyncio.run(agent_s.engage(acceptable_role="teacher",
                                      sender_role="world_agent~teacher",
                                      interaction=_engage_interaction())) is True
    assert agent_s._engaged_agents == {"T-world"}
    assert conn_s.is_connected("T-world") is True

    # Wire a purge closure that MIRRORS Node.__purge exactly (same order), bound to
    # the harness conn + real agent. disengage calls it with
    # reason=WORLD_DISENGAGEMENT, so drop_peer GOAWAYs the Session then drops the conn.
    async def purge(peer_id, reason=None):
        for ev in await conn_s.drop_peer(peer_id, reason):
            await node_s._dispatch_session_event(ev)
        await agent_s.remove_agent(peer_id)

    agent_s.node_purge_fcn = purge

    inter = _disengage_interaction()
    ok = asyncio.run(agent_s.disengage(disconnect_too=True, interaction=inter))

    # The @action swallows exceptions to False, so a True return first proves the
    # purge ran to completion (the bug-3 assertions below would otherwise be masked).
    assert ok is True
    assert agent_s._engaged_agents == set()
    assert agent_s._available is True

    # BUG 3 (fixed): the reasoned purge went through conn.drop_peer, so the GOAWAY
    # was recorded on the conn with WORLD_DISENGAGEMENT.
    assert conn_s.closed_sessions == [("T-world", WORLD_DISENGAGEMENT)]

    # ...yet remove_agent + the pool drop still ran: the peer is gone from the
    # membership tables AND from the conn pool (no longer connected).
    assert "T-world" not in agent_s.all_agents
    assert "T-world" not in agent_s.world_agents
    assert conn_s.is_connected("T-world") is False
    assert conn_s.get_pool_of("T-world") is None


# (e) executable boundary: the full loop drops engage at the placeholder

def test_full_loop_engage_is_dropped_by_placeholder_behav(harness_clock, roles):
    """An engage INTERACTION delivered through the REAL inbound loop
    (``Node.__handle_network_messages``) reaches the INTERACTION dispatcher but is
    NOT executed: a harness world_agent's ``behav`` is the empty one-state
    placeholder whose ``request_action`` rejects the action, so
    ``inject_received_interaction`` never calls ``register_received`` and the engage
    handler never runs.

    The assert-absent (no engagement, nothing received) is made meaningful by
    POSITIVELY proving that every gate UPSTREAM of the HSM action gate passes:
    T-world is an admitted pool peer (the get_messages membership gate) AND a known
    agent (the INTERACTION ``all_agents`` gate), and the placeholder
    ``request_action`` returns False. So the ONLY thing that drops the engage is the
    world-HSM boundary, which is exactly what this test pins (it is why the lifecycle
    tests above drive the handlers directly; see the module docstring)."""
    sb, conn_t, conn_s, node_t, agent_t, node_s, agent_s = _build_world_pair()

    engage_inter = Interaction(action_name="engage",
                               action_kwargs={"sender_role": "world_agent~teacher"},
                               requester="T-world", target=["S-world"])

    async def _run():
        ok = await conn_t.send("S-world", channel_trail=None,
                               content=engage_inter.to_dict(),
                               content_type=Msg.INTERACTION)
        assert ok is True
        # The Msg is really staged in the student's WORLD inbox.
        assert len(conn_s.chat_inbox[InProcNodeConn.P2P_WORLD]) == 1

        # POSITIVE: every gate upstream of the HSM action gate admits this message,
        # so the drop below cannot be a silent membership/all_agents rejection.
        #  - get_messages membership gate: the sender is a known pool peer;
        #  - INTERACTION dispatch all_agents gate (node.py:1855): sender is known;
        #  - room for one more received interaction.
        assert "T-world" in conn_s.peer_id_to_pool_name
        assert "T-world" in agent_s.all_agents
        assert agent_s.im.room_for_registration() is True

        # POSITIVE proof of WHY it is dropped: the placeholder world HSM rejects the
        # action at the boundary gate (inject_received_interaction -> request_action).
        assert agent_s.behav.request_action(engage_inter) is False

        # Drive the REAL inbound loop: it delivers, gates and dispatches the message.
        # (get_messages popleft-drains the inbox unconditionally, so the inbox going
        # empty is NOT evidence of anything; the gates asserted above are.)
        await node_s._Node__handle_network_messages()

    asyncio.run(_run())

    # BOUNDARY: with all upstream gates open and only request_action rejecting, the
    # engage never executed: no engagement was established and, crucially, nothing
    # was registered as received (inject_received_interaction took the error branch,
    # not register_received).
    assert agent_s._engaged_agents == set()
    assert agent_s._available is True
    assert agent_s.im.received == {}


# (f) disconnect_by_role targets only enrolled agents (close-spam fix)

def test_disconnect_by_role_targets_only_enrolled_agents(roles):
    """disconnect_by_role must purge only agents that completed the handshake (in
    all_agents), not every role match in the rendezvous roster. Without the
    handshake_completed filter it fires purge on not-yet-connected peers on every
    behavior tick — a flood of no-op closes plus a synchronous cgo DisconnectFrom
    each. Here a role match that is roster-only must be left alone."""
    enrolled, roster_only = "PIDenrolled", "PIDroster"

    class _Conn:
        def find_addrs_by_role(self, role_int, return_peer_ids_too=False):
            # both peers match the role in the roster; only `enrolled` is connected
            return {}, [enrolled, roster_only]

    a = Agent.__new__(Agent)
    a.node_conn = _Conn()
    a.all_agents = {enrolled: object()}
    a._found_agents = set()
    a._engaged_agents = set()

    purged = []

    async def _purge(peer_id, reason=None):
        purged.append(peer_id)

    a.node_purge_fcn = _purge

    ok = asyncio.run(a.disconnect_by_role("student"))
    assert ok is True
    assert purged == [enrolled]                    # the roster-only peer is NOT purged
