"""In-process e2e for the pubsub stream path.

A publisher agent (A) OWNS a pubsub stream ``{owner}::ps:{group}`` and emits a
sample through the REAL egress code ``AgentBasics.send_stream_samples`` (the
pubsub branch at agent_basics.py:2119-2137 calls ``conn.publish``). A subscriber
agent (B) receives it through the REAL inbound dispatch
``Node.__handle_network_messages`` -> ``get_stream_sample``, and we POSITIVELY
ASSERT the sample landed in ``B.known_streams[net_hash][group]``. Because a pubsub
channel passes through ``normalize_net_hash`` UNCHANGED, ``msg.channel`` equals a
known_streams key verbatim (no DM-style ``~`` collapse).

What is REAL source here: the publisher's ``send_stream_samples`` egress and the
subscriber's ``get_stream_sample`` ingestion (and the pubsub passthrough in
``normalize_net_hash``). What is HARNESS (not real gossipsub): the topic fan-out
that decides WHO receives a publish — the switchboard subscriber-set. So the two
negatives below (non-subscriber C, no self-delivery to A) pin the harness routing
contract, while the real gossipsub routing is covered at Level 3 (``-m p2p``).
"""

import asyncio

import pytest

from unaiverse.clock import clock
from unaiverse.streams.streams import Stream
from unaiverse.streams.dataprops import DataProps

from inproc_harness import (
    InProcSwitchboard, InProcNodeConn, HarnessClock,
    build_real_agent_node,
)

pytestmark = pytest.mark.e2e

GROUP = "best_student_stream"
PUB_NET_HASH = DataProps.build_net_hash("A-world", pubsub=True, name_or_group=GROUP)


# ── Deterministic clock fixture (save/restore the global singleton) ──────────

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


def _build_triple():
    """A publisher (A) + subscriber (B) + non-subscriber (C), all real world
    agents on one switchboard."""
    sb = InProcSwitchboard()
    conn_a = InProcNodeConn(node_id="node-A", public_peer_id="A-pub",
                            world_peer_id="A-world", switchboard=sb)
    conn_b = InProcNodeConn(node_id="node-B", public_peer_id="B-pub",
                            world_peer_id="B-world", switchboard=sb)
    conn_c = InProcNodeConn(node_id="node-C", public_peer_id="C-pub",
                            world_peer_id="C-world", switchboard=sb)
    node_a, agent_a = build_real_agent_node(
        "A-pub", proc=_echo, proc_inputs=["text"], proc_outputs=["text"],
        conn=conn_a, node_id="node-A", world_peer_id="A-world", role="world_agent")
    node_b, agent_b = build_real_agent_node(
        "B-pub", proc=_echo, proc_inputs=["text"], proc_outputs=["text"],
        conn=conn_b, node_id="node-B", world_peer_id="B-world", role="world_agent")
    node_c, agent_c = build_real_agent_node(
        "C-pub", proc=_echo, proc_inputs=["text"], proc_outputs=["text"],
        conn=conn_c, node_id="node-C", world_peer_id="C-world", role="world_agent")
    return (sb, conn_a, conn_b, conn_c,
            node_a, agent_a, node_b, agent_b, node_c, agent_c)


def _give_publisher_owned_pubsub_stream(agent, value="hi", uuid="u1", data_tag=0):
    """Make ``agent`` OWN the pubsub stream ``A-world::ps:{GROUP}`` carrying one
    sample, so the REAL ``send_stream_samples`` pubsub branch publishes it."""
    stream = Stream(name=GROUP, group=GROUP, data_type="text", pubsub=True)
    agent.add_stream(stream, owned=True, net_hash=PUB_NET_HASH)
    stream.set(value, data_tag=data_tag, uuid=uuid)
    return stream


def _register_known_pubsub_stream(agent):
    """Register on ``agent`` a KNOWN (not-owned) pubsub stream under ``PUB_NET_HASH``
    — the home where a landed sample is stored. The stream ``name`` must equal GROUP
    (the content-dict key the publisher emits)."""
    stream = Stream(name=GROUP, group=GROUP, data_type="text", pubsub=True)
    agent.add_stream(stream, owned=False, net_hash=PUB_NET_HASH)


# ── (a) pubsub net_hash passes through normalize_net_hash unchanged ───────────

def test_pubsub_channel_passes_through_normalize_unchanged():
    """The load-bearing property: a pubsub net_hash is recognized as such and is
    NOT collapsed by normalize_net_hash (unlike a DM channel), so the published
    msg.channel equals the subscriber's known_streams key byte-for-byte."""
    assert PUB_NET_HASH == "A-world::ps:best_student_stream"
    assert DataProps.is_pubsub_from_net_hash(PUB_NET_HASH)
    assert DataProps.normalize_net_hash(PUB_NET_HASH) == PUB_NET_HASH


# ── (b) real egress -> real ingestion: the subscriber receives the sample ─────

def test_pubsub_egress_via_send_stream_samples_reaches_subscriber(harness_clock):
    """A publishes through the REAL ``send_stream_samples`` (pubsub branch); B,
    subscribed, receives it through the REAL inbound loop and stores it."""
    (sb, conn_a, conn_b, conn_c,
     node_a, agent_a, node_b, agent_b, node_c, agent_c) = _build_triple()

    _give_publisher_owned_pubsub_stream(agent_a)
    _register_known_pubsub_stream(agent_b)
    # Membership/identity gate: the publisher's WORLD sender id must be admitted on
    # B's gate or get_messages silently drops the Msg.
    conn_b.register_peer("A-world", node_id="node-A")

    async def _run():
        # B subscribes to A's topic on the WORLD network.
        assert await conn_b.subscribe(peer_id="A-world", channel=PUB_NET_HASH,
                                      default_p2p_name=InProcNodeConn.P2P_WORLD) is True
        assert "B-world" in sb.subs[InProcNodeConn.P2P_WORLD][PUB_NET_HASH]

        # REAL EGRESS: the publisher's own send_stream_samples emits the publish.
        await agent_a.send_stream_samples()

        # The publish fanned out exactly one Msg into B's WORLD inbox.
        assert len(conn_b.chat_inbox[InProcNodeConn.P2P_WORLD]) == 1
        staged = conn_b.chat_inbox[InProcNodeConn.P2P_WORLD][0]
        assert staged.sender == "A-world"
        assert staged.channel == PUB_NET_HASH  # verbatim pubsub channel, no collapse

        # REAL INGESTION: B's inbound loop dispatches STREAM_SAMPLE -> get_stream_sample.
        await node_b._Node__handle_network_messages()

    asyncio.run(_run())

    # POSITIVE STATE ASSERTION: the sample landed in B's known pubsub stream.
    assert PUB_NET_HASH in agent_b.known_streams
    assert GROUP in agent_b.known_streams[PUB_NET_HASH]
    landed = agent_b.known_streams[PUB_NET_HASH][GROUP]
    assert landed.has_data("u1")
    assert landed.get(uuid="u1") == "hi"
    assert landed.get_tag("u1") == 0


# ── (c) a non-subscriber never receives the publish ───────────────────────────

def test_pubsub_non_subscriber_not_delivered(harness_clock):
    """C, which did NOT subscribe, is not in the topic's subscriber set, so the
    publish never reaches it (harness fan-out contract; real gossipsub at Level 3).
    Proven meaningful by the positive control: B, subscribed in the same run, DOES
    receive the very same publish."""
    (sb, conn_a, conn_b, conn_c,
     node_a, agent_a, node_b, agent_b, node_c, agent_c) = _build_triple()

    _give_publisher_owned_pubsub_stream(agent_a)
    _register_known_pubsub_stream(agent_b)
    _register_known_pubsub_stream(agent_c)
    conn_b.register_peer("A-world", node_id="node-A")
    conn_c.register_peer("A-world", node_id="node-A")

    async def _run():
        # B subscribes, C deliberately does NOT.
        await conn_b.subscribe(peer_id="A-world", channel=PUB_NET_HASH,
                               default_p2p_name=InProcNodeConn.P2P_WORLD)
        assert "C-world" not in sb.subs[InProcNodeConn.P2P_WORLD].get(PUB_NET_HASH, set())

        await agent_a.send_stream_samples()  # real egress

        # Positive control: the subscriber B got it; the non-subscriber C did not.
        assert len(conn_b.chat_inbox[InProcNodeConn.P2P_WORLD]) == 1
        assert len(conn_c.chat_inbox[InProcNodeConn.P2P_WORLD]) == 0

        # Drive both loops; only B can ingest (C's inbox is empty by routing).
        await node_b._Node__handle_network_messages()
        await node_c._Node__handle_network_messages()

    asyncio.run(_run())

    # POSITIVE on the subscriber, NEGATIVE on the non-subscriber.
    assert agent_b.known_streams[PUB_NET_HASH][GROUP].has_data("u1")
    assert not agent_c.known_streams[PUB_NET_HASH][GROUP].has_data("u1")


# ── (d) the publisher is not self-delivered ───────────────────────────────────

def test_pubsub_publisher_not_self_delivered(harness_clock):
    """A, even when subscribed to its OWN topic, is not delivered its own publish
    (the harness publish skips ``sub_conn is self``, faithful to gossipsub, which
    does not loop a publish back to the publisher)."""
    (sb, conn_a, conn_b, conn_c,
     node_a, agent_a, node_b, agent_b, node_c, agent_c) = _build_triple()

    _give_publisher_owned_pubsub_stream(agent_a)
    _register_known_pubsub_stream(agent_a)  # A keeps a home stream too
    _register_known_pubsub_stream(agent_b)
    conn_a.register_peer("A-world", node_id="node-A")
    conn_b.register_peer("A-world", node_id="node-A")

    async def _run():
        # Both A (its own topic) and B subscribe.
        await conn_a.subscribe(peer_id="A-world", channel=PUB_NET_HASH,
                               default_p2p_name=InProcNodeConn.P2P_WORLD)
        await conn_b.subscribe(peer_id="A-world", channel=PUB_NET_HASH,
                               default_p2p_name=InProcNodeConn.P2P_WORLD)
        assert "A-world" in sb.subs[InProcNodeConn.P2P_WORLD][PUB_NET_HASH]

        await agent_a.send_stream_samples()  # real egress

        # Positive control: B (other subscriber) got it; A (publisher) did NOT.
        assert len(conn_b.chat_inbox[InProcNodeConn.P2P_WORLD]) == 1
        assert len(conn_a.chat_inbox[InProcNodeConn.P2P_WORLD]) == 0

        await node_a._Node__handle_network_messages()  # nothing to ingest

    asyncio.run(_run())

    # The publisher's own home stream never received its own publish.
    assert not agent_a.known_streams[PUB_NET_HASH][GROUP].has_data("u1")
