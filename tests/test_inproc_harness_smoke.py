"""Walking-skeleton smoke test for the in-process application chat-path harness.

This proves the harness is FAITHFUL, not merely non-crashing. Agent A sends a DM
STREAM_SAMPLE to agent B through the real ``conn.send`` channel formula + codec
round-trip; B's REAL ``Node.__handle_network_messages`` is driven; and we
POSITIVELY ASSERT the sample landed in
``B.known_streams[normalized_net_hash]['processor']`` (so the channel was built
right, ``normalize_net_hash`` matched, the membership gate passed, and the
piggyback rewrite + codec round-trip worked).

A second test pins the membership gate: an UNREGISTERED sender's sample is
silently dropped (the harness does not trivially deliver everything).
"""

import asyncio

import pytest

from unaiverse.clock import clock
from unaiverse.networking.p2p.messages import Msg
from unaiverse.streams.dataprops import DataProps

from inproc_harness import (
    InProcSwitchboard, InProcNodeConn, HarnessClock,
    build_real_agent_node, register_known_processor_stream,
)

pytestmark = pytest.mark.e2e


#  Deterministic clock fixture (save/restore the global singleton)

@pytest.fixture
def harness_clock():
    saved = getattr(clock, "_Clock__instance", None)
    fc = HarnessClock()
    clock.set(fc)
    fc.next_cycle()  # leave cycle 0 active so stream timestamps are well-defined
    yield fc
    setattr(clock, "_Clock__instance", saved)


def _echo(x):
    """Trivial echo processor (str in -> str out): keeps STREAM_SAMPLE content a
    plain str, which is codec-stable through Msg.to_bytes/from_bytes."""
    return x


def _build_pair():
    """Build two real agents (A, B) on a shared switchboard, both world agents."""
    sb = InProcSwitchboard()
    conn_a = InProcNodeConn(node_id="node-A", public_peer_id="A-pub",
                            world_peer_id="A-world", switchboard=sb)
    conn_b = InProcNodeConn(node_id="node-B", public_peer_id="B-pub",
                            world_peer_id="B-world", switchboard=sb)
    node_a, agent_a = build_real_agent_node(
        "A-pub", proc=_echo, proc_inputs=["text"], proc_outputs=["text"],
        conn=conn_a, node_id="node-A", world_peer_id="A-world")
    node_b, agent_b = build_real_agent_node(
        "B-pub", proc=_echo, proc_inputs=["text"], proc_outputs=["text"],
        conn=conn_b, node_id="node-B", world_peer_id="B-world")
    return sb, conn_a, conn_b, node_a, agent_a, node_b, agent_b


def test_dm_stream_sample_lands_via_real_node_loop(harness_clock):
    """A->B DM STREAM_SAMPLE: drive B's REAL Node.__handle_network_messages and
    assert the sample landed in B.known_streams[net_hash]['processor']."""
    sb, conn_a, conn_b, node_a, agent_a, node_b, agent_b = _build_pair()

    # B registers a known processor stream keyed by the net_hash that a sample
    # from A (channel_trail='processor', sender=A-world) normalizes to.
    net_hash = register_known_processor_stream(agent_b, sender_world_peer_id="A-world",
                                               name="processor")
    assert net_hash == "A-world::dm:processor"

    # Admit A on B's world net (membership gate) and give it a node-id (piggyback
    # rewrite). Without this the message would be silently dropped.
    conn_b.register_peer("A-world", node_id="node-A")

    sample = {"processor": {"data": "hi", "data_tag": 0, "data_uuid": "u1"}}

    async def _run():
        # A sends through the REAL send formula (channel built from A's world
        # peer-id, codec round-trip applied).
        ok = await conn_a.send("B-world", channel_trail="processor",
                               content_type=Msg.STREAM_SAMPLE, content=sample)
        assert ok is True

        # Sanity: the message reached B's WORLD inbox as a real Msg with the
        # production channel string.
        inbox = conn_b.chat_inbox[InProcNodeConn.P2P_WORLD]
        assert len(inbox) == 1
        staged = inbox[0]
        assert isinstance(staged, Msg)
        assert staged.sender == "A-world"
        assert staged.channel == "A-world::dm:B-world-stream_sample~processor"
        assert DataProps.normalize_net_hash(staged.channel) == net_hash
        # Codec round-trip preserved the plain-str sample (new object, no aliasing).
        assert staged.content == sample
        assert staged.content is not sample

        # Drive B's REAL inbound dispatch loop (mode A: real gate/sort/dispatch).
        await node_b._Node__handle_network_messages()

    asyncio.run(_run())

    # POSITIVE ASSERTION: the sample landed in B's known stream.
    assert net_hash in agent_b.known_streams
    assert "processor" in agent_b.known_streams[net_hash]
    landed = agent_b.known_streams[net_hash]["processor"]
    assert landed.has_data("u1")
    assert landed.get(uuid="u1") == "hi"
    assert landed.get_tag("u1") == 0

    # The inbox was fully drained by the real loop.
    assert len(conn_b.chat_inbox[InProcNodeConn.P2P_WORLD]) == 0


def test_membership_gate_drops_unregistered_sender(harness_clock):
    """If the sender is NOT admitted (not in a pool / cycle membership), the chat
    Msg is silently dropped by get_messages and never reaches the stream. This
    pins that the harness reproduces the gate rather than delivering everything."""
    sb, conn_a, conn_b, node_a, agent_a, node_b, agent_b = _build_pair()

    net_hash = register_known_processor_stream(agent_b, sender_world_peer_id="A-world",
                                               name="processor")
    # NOTE: deliberately DO NOT register A on conn_b.

    sample = {"processor": {"data": "hi", "data_tag": 0, "data_uuid": "u1"}}

    async def _run():
        ok = await conn_a.send("B-world", channel_trail="processor",
                               content_type=Msg.STREAM_SAMPLE, content=sample)
        assert ok is True
        # The Msg is staged in the inbox...
        assert len(conn_b.chat_inbox[InProcNodeConn.P2P_WORLD]) == 1
        # ...but the gate drops it: get_messages returns nothing.
        chat, sess = await conn_b.get_messages(InProcNodeConn.P2P_WORLD)
        assert chat == []
        assert sess == []

    asyncio.run(_run())

    # Nothing landed.
    assert not agent_b.known_streams[net_hash]["processor"].has_data("u1")
