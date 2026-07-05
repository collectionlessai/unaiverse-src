"""In-process e2e of the lone-wolf chat path: A<->B request/response + broadcast.

Two REAL public lone-wolf Agents (role='public_agent') talk over the faithful
InProcNodeConn. The public HSM (public.json) is fully loaded, so the receiver
actually EXECUTES 'process' (a str echo) rather than parking on a placeholder.

The four scenarios POSITIVELY ASSERT mutated state, never mere "no exception":

  1. registration: A.send('process', B, data_samples=['hi']) -> B's REAL receive
     dispatch registers the interaction AND the HSM links the action
     (received_interaction.action_ref is not None), A.im.sent holds the sent one.
     The LIST form is registration-only by design: it crashes the receiver's
     behave() (a real, freshly-found bug pinned here) and never binds to stdin,
     so it cannot complete in-process. The completion round-trip uses the DICT
     form instead (Test 2).
  2. status round-trip (DICT form, the faithful chat path): B completes, emits
     INTERACTION_STATUS, A.im.update_sent_status flips A's sent interaction to
     COMPLETED with destination_state=='generated'; to_status_dict round-trips.
     STRETCH: the processor-output echo lands back at A.known_streams.
  3. broadcast-to-N: A sends to [B, C]; each receiver registers the interaction.
  4. unknown-sender gate: an INTERACTION whose sender is admitted by the conn
     pool but is NOT in B.all_agents is dropped before inject (membership gate).

See the scoping notes in the task for the load-bearing gotchas (behav-enabled-by
-default flips get_peer_id() to the WORLD id, the two membership gates, the
deterministic tick ordering).
"""

import asyncio

import pytest

from unaiverse.clock import clock
from unaiverse.interaction import InteractionStatus

from inproc_harness import (
    InProcSwitchboard, InProcNodeConn, HarnessClock,
    build_real_agent_node, tick, ROLE_PUBLIC,
)

pytestmark = pytest.mark.e2e


# Deterministic clock fixture (verbatim from the smoke test)

@pytest.fixture
def harness_clock():
    saved = getattr(clock, "_Clock__instance", None)
    fc = HarnessClock()
    clock.set(fc)
    fc.next_cycle()  # leave cycle 0 active so stream timestamps are well-defined
    yield fc
    setattr(clock, "_Clock__instance", saved)


def _echo(x, first: bool = False, last: bool = False):
    """Echo processor (str in -> str out). MUST accept (x, first=, last=): the
    'process' action calls proc(*input_data, first=..., last=...); a 1-arg lambda
    would TypeError. A plain str is codec-stable through Msg.to_bytes/from_bytes."""
    return x


# Builders for a public lone-wolf agent and the public cross-register

def _build_agent(pub_id: str, node_id: str, world_id: str, conn: InProcNodeConn):
    return build_real_agent_node(
        pub_id, proc=_echo, proc_inputs=["text"], proc_outputs=["text"],
        conn=conn, node_id=node_id, world_peer_id=world_id, role="public_agent")


async def _public_link(conn_self: InProcNodeConn, agent_self,
                       other_pub: str, other_node_id: str, other_profile) -> None:
    """One-directional public wiring: seed the conn membership/identity/role
    tables for ``other_pub`` (ROLE_PUBLIC, IN_PUBLIC pool), then add_agent so the
    counterpart is classified into public_agents and its processor stream learned."""
    conn_self.register_peer(other_pub, node_id=other_node_id,
                            role=ROLE_PUBLIC, pool=conn_self.IN_PUBLIC)
    await agent_self.add_agent(other_pub, other_profile)


# Test 1: registration via the REAL receive dispatch (LIST form, no completion)

def test_send_registers_interaction_and_links_hsm_action(harness_clock):
    """A.send('process', B, data_samples=['hi']): A.im.sent holds the sent
    interaction; B's REAL Node.__handle_network_messages registers the received
    interaction AND the HSM links it to the 'process' action (action_ref set,
    status RECEIVED). The LIST form is registration-only; the receiver's behave()
    then runs cleanly over it (the old '<data_sample_0>' crash is fixed, see below)."""
    sb = InProcSwitchboard()
    conn_a = InProcNodeConn("node-A", "A-pub", "A-world", sb)
    conn_b = InProcNodeConn("node-B", "B-pub", "B-world", sb)
    node_a, agent_a = _build_agent("A-pub", "node-A", "A-world", conn_a)
    node_b, agent_b = _build_agent("B-pub", "node-B", "B-world", conn_b)

    async def _run():
        await _public_link(conn_a, agent_a, "B-pub", "node-B", agent_b.node_profile)
        await _public_link(conn_b, agent_b, "A-pub", "node-A", agent_a.node_profile)
        # MANDATORY: flip get_peer_id() back to the PUBLIC id (a fresh public agent
        # has the world HSM enabled, so the requester would otherwise be 'A-world').
        agent_a.behav.enable(False)
        agent_b.behav.enable(False)

        ok = await agent_a.send(action_name="process", target="B-pub",
                                data_samples=["hi"])
        assert ok is True
        sent = agent_a.last_sent_interaction
        assert sent is not None
        # A registered the sent interaction with the PUBLIC requester id.
        assert sent.uuid in agent_a.im.sent
        assert agent_a.im.sent[sent.uuid] is sent
        assert sent.requester == "A-pub"

        # Drive ONLY B's real inbound dispatch (do NOT pump full ticks: the list
        # form crashes B.behave(), pinned separately below).
        await node_b._Node__handle_network_messages()

        return sent

    sent = asyncio.run(_run())

    # POSITIVE ASSERTIONS on B's mutated state: the interaction was registered and
    # the HSM linked it to the 'process' action.
    rec = agent_b.im.get_interaction(sent.uuid, consider_completed_too=True)
    assert rec is not None
    assert rec.uuid == sent.uuid
    assert rec.requester == "A-pub"
    assert rec.status == InteractionStatus.RECEIVED
    assert rec.action_ref is not None  # HybridStateMachine.request_action linked it

    # A list-form data_samples=['hi'] used to crash the receiver's behave():
    # Interaction.set_manager binds synthetic '<data_sample_0>' keys into the
    # StreamProxy, and StreamProxy.__str__ only special-cased '<default' keys, so
    # '<data_sample_0>' fell through to DataProps.name_from_user_hash, whose
    # rsplit(':',1)[1] raised IndexError when the HSM logged the queued interaction
    # during act(). __str__ now special-cases '<data_sample' keys, so behave() runs
    # cleanly and the registered interaction survives the pass.
    asyncio.run(agent_b.behave())
    assert agent_b.im.get_interaction(sent.uuid, consider_completed_too=True) is not None


# Test 2: status round-trip (DICT form) + STRETCH processor-output echo

def test_status_round_trip_completes_sent_interaction(harness_clock):
    """DICT form data_samples={'proc_output_0':'hi'} (the faithful chat path):
    B registers + executes 'process' (echo) + completes, emits INTERACTION_STATUS,
    A.im.update_sent_status flips A's sent interaction to COMPLETED with
    destination_state=='generated'. to_status_dict round-trips status/dest_state.
    STRETCH: the echo lands back at A.known_streams['B-pub::dm:processor']."""
    sb = InProcSwitchboard()
    conn_a = InProcNodeConn("node-A", "A-pub", "A-world", sb)
    conn_b = InProcNodeConn("node-B", "B-pub", "B-world", sb)
    node_a, agent_a = _build_agent("A-pub", "node-A", "A-world", conn_a)
    node_b, agent_b = _build_agent("B-pub", "node-B", "B-world", conn_b)

    state = {}

    async def _run():
        await _public_link(conn_a, agent_a, "B-pub", "node-B", agent_b.node_profile)
        await _public_link(conn_b, agent_b, "A-pub", "node-A", agent_a.node_profile)
        agent_a.behav.enable(False)
        agent_b.behav.enable(False)

        ok = await agent_a.send(action_name="process", target="B-pub",
                                data_samples={"proc_output_0": "hi"})
        assert ok is True
        sent = agent_a.last_sent_interaction
        assert sent is not None
        assert sent.uuid in agent_a.im.sent
        state["sent"] = sent

        # Pump several full ticks on BOTH agents in canonical order. The dict-form
        # round-trip needs ~3 iterations (B registers+completes; B emits
        # INTERACTION_STATUS + ships the echo; A consumes both); 8 gives margin.
        b_received = None
        for _ in range(8):
            await tick(node_a)
            await tick(node_b)
            if b_received is None:
                cand = agent_b.im.get_interaction(sent.uuid, consider_completed_too=True)
                if cand is not None:
                    b_received = cand
        state["b_received"] = b_received

    asyncio.run(_run())

    sent = state["sent"]
    # POSITIVE ASSERTION: A's sent interaction transitioned to COMPLETED and the
    # destination_state carried by the status message landed on it.
    assert sent.is_completed()
    assert sent.status == InteractionStatus.COMPLETED
    assert sent.destination_state == "generated"
    # The status dict round-trips the same status/destination_state.
    status_dict = sent.to_status_dict("A-pub")
    assert status_dict["status"] == "completed"
    assert status_dict["destination_state"] == "generated"
    assert status_dict["uuid"] == sent.uuid

    # B really registered the interaction (captured the first tick it appeared).
    b_received = state["b_received"]
    assert b_received is not None
    assert b_received.uuid == sent.uuid
    assert b_received.action_ref is not None

    # STRETCH (deterministic): the processor-output echo came back to A under the
    # interaction uuid on B's known processor stream.
    echo_net_hash = "B-pub::dm:processor"
    assert echo_net_hash in agent_a.known_streams
    echo_stream = agent_a.known_streams[echo_net_hash]["proc_output_0"]
    assert echo_stream.get(uuid=sent.uuid) == "hi"


# Test 3: broadcast-to-N (DICT form) — each target registers the interaction

def test_broadcast_registers_on_every_target(harness_clock):
    """A.send to a list [B, C]: each receiver's REAL dispatch registers the
    interaction (one sent interaction fanned out to N targets)."""
    sb = InProcSwitchboard()
    conn_a = InProcNodeConn("node-A", "A-pub", "A-world", sb)
    conn_b = InProcNodeConn("node-B", "B-pub", "B-world", sb)
    conn_c = InProcNodeConn("node-C", "C-pub", "C-world", sb)
    node_a, agent_a = _build_agent("A-pub", "node-A", "A-world", conn_a)
    node_b, agent_b = _build_agent("B-pub", "node-B", "B-world", conn_b)
    node_c, agent_c = _build_agent("C-pub", "node-C", "C-world", conn_c)

    async def _run():
        # A learns B and C; B and C each learn A (so they admit A's interaction).
        await _public_link(conn_a, agent_a, "B-pub", "node-B", agent_b.node_profile)
        await _public_link(conn_a, agent_a, "C-pub", "node-C", agent_c.node_profile)
        await _public_link(conn_b, agent_b, "A-pub", "node-A", agent_a.node_profile)
        await _public_link(conn_c, agent_c, "A-pub", "node-A", agent_a.node_profile)
        agent_a.behav.enable(False)
        agent_b.behav.enable(False)
        agent_c.behav.enable(False)

        ok = await agent_a.send(action_name="process", target=["B-pub", "C-pub"],
                                data_samples={"proc_output_0": "hi"})
        assert ok is True
        sent = agent_a.last_sent_interaction
        assert sent is not None
        assert sent.uuid in agent_a.im.sent

        # Drive each receiver's REAL inbound dispatch once (registration only).
        await node_b._Node__handle_network_messages()
        await node_c._Node__handle_network_messages()
        return sent

    sent = asyncio.run(_run())

    # POSITIVE ASSERTION: BOTH B and C registered the broadcast interaction.
    rec_b = agent_b.im.get_interaction(sent.uuid, consider_completed_too=True)
    rec_c = agent_c.im.get_interaction(sent.uuid, consider_completed_too=True)
    assert rec_b is not None and rec_b.uuid == sent.uuid
    assert rec_b.action_ref is not None
    assert rec_c is not None and rec_c.uuid == sent.uuid
    assert rec_c.action_ref is not None


# Test 4: unknown-sender gate — INTERACTION from a non-member is dropped

def test_unknown_sender_interaction_is_dropped(harness_clock):
    """X is admitted by B's conn POOL (passes the get_messages gate) but is NOT in
    B.all_agents (no add_agent on B's side). The Node INTERACTION handler drops it
    BEFORE inject ('Unexpected interaction from unknown node'), so nothing is
    registered. This pins the AGENT all_agents membership gate (distinct from the
    conn pool gate already pinned by the smoke test)."""
    sb = InProcSwitchboard()
    conn_b = InProcNodeConn("node-B", "B-pub", "B-world", sb)
    conn_x = InProcNodeConn("node-X", "X-pub", "X-world", sb)
    node_b, agent_b = _build_agent("B-pub", "node-B", "B-world", conn_b)
    node_x, agent_x = _build_agent("X-pub", "node-X", "X-world", conn_x)

    async def _run():
        # X knows B (so X can resolve its target and stamp requester='X-pub')...
        await _public_link(conn_x, agent_x, "B-pub", "node-B", agent_b.node_profile)
        # ...and B admits X in its conn POOL so the get_messages gate passes, but X
        # is deliberately NOT add_agent'd on B -> X stays out of B.all_agents.
        conn_b.register_peer("X-pub", node_id="node-X",
                             role=ROLE_PUBLIC, pool=conn_b.IN_PUBLIC)
        agent_x.behav.enable(False)
        agent_b.behav.enable(False)

        # Sanity: B has a working all_agents gate (X is absent from it).
        assert "X-pub" not in agent_b.all_agents

        ok = await agent_x.send(action_name="process", target="B-pub")
        assert ok is True
        sent = agent_x.last_sent_interaction
        assert sent is not None

        before = set(agent_b.im.received.keys())
        await node_b._Node__handle_network_messages()
        after = set(agent_b.im.received.keys())
        return sent, before, after

    sent, before, after = asyncio.run(_run())

    # POSITIVE control + gate assertion: X really sent it, but B registered NOTHING.
    assert after == before
    assert sent.uuid not in agent_b.im.received
    assert agent_b.im.get_interaction(sent.uuid, consider_completed_too=True) is None
