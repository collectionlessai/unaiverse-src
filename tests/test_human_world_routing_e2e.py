"""World-side purge of application content (the bug-2 punishment mechanism).

A node whose ``node_type`` is ``Node.WORLD`` must never receive application
chat (INTERACTION / STREAM_SAMPLE): those are agent-to-agent content. When one
arrives anyway, the world dispatch branch (node.py:1845-1847 for STREAM_SAMPLE,
node.py:1865-1867 for INTERACTION) calls ``__purge(sender, reason=LOCAL_ERROR)``
(node.py:2664-2692), which (a) GOAWAY-closes the sender's Session with
LOCAL_ERROR, (b) drops the sender from the hosted entity via
``hosted.remove_agent``, and (c) removes it from the connection pool via
``conn.remove``.

This is the *mechanism* that punishes a human whose reply is mis-routed to the
world node instead of to the broadcaster. The mis-routing itself (BUG 2) is a
separate defect, already pinned as strict-xfail UNIT tests in
tests/test_human_routing.py (the broadcaster peer-id is overwritten with the
world-node id on both the store and the read side) and slated for a realnet
test at Level 3. Here we CHARACTERIZE the world-purge that the mis-route
triggers: it currently happens and is the correct world-side behaviour, so it
is asserted green, not xfail.

The purge fires only for a sender that PASSED the get_messages membership gate
(a pool member / this-cycle world member). A truly-unknown sender is dropped at
the gate and never reaches dispatch, so it is never purged. The second test
pins that contrast, which is exactly the faithful bug-2 scenario: the human is a
registered world member, so its mis-routed reply does reach the world and does
get purged.
"""
import asyncio

import pytest

from unaiverse.networking.node.node import Node
from unaiverse.networking.p2p.messages import Msg
from unaiverse.networking.session.sess_msg import LOCAL_ERROR  # == 15

from inproc_harness import InProcSwitchboard, InProcNodeConn
from node_harness import FakeHosted, FakeWorld, FakeNodeProfile


def _build_world_and_sender():
    """Two InProcNodeConn on one switchboard: a WORLD-host Node shell (FakeHosted
    + FakeWorld) and a bare sender conn (no Agent/Node needed, only ``send``).

    The sender 'H' is registered on the world conn so it PASSES the membership
    gate, and pre-seeded into ``hosted.all_agents`` so the purge's drop is
    observable. This mirrors a human/world-agent registered in the world whose
    reply is mis-routed to the world node (bug-2, see tests/test_human_routing.py).
    """
    sb = InProcSwitchboard()
    conn_w = InProcNodeConn(node_id="node-W", public_peer_id="W-pub",
                            world_peer_id="W-world", switchboard=sb)
    conn_h = InProcNodeConn(node_id="node-H", public_peer_id="H-pub",
                            world_peer_id="H-world", switchboard=sb)

    # WORLD-host Node shell: node_type WORLD, a hosted entity exposing
    # remove_agent + all_agents, and a profile exposing inspector_node_id (the
    # piggyback/inspector unpack at node.py:1774 reads it). A real World host with
    # role discovery via assign_role is unnecessary for the purge side-effects.
    node_w = Node.__new__(Node)
    node_w.node_id = "node-W"
    node_w.node_type = Node.WORLD
    node_w.conn = conn_w
    node_w.hosted = FakeHosted()
    node_w.world = FakeWorld()
    node_w.agent = None
    node_w.profile = FakeNodeProfile("node-W", world_peer_id="W-world")
    return sb, conn_w, conn_h, node_w


@pytest.mark.parametrize("content_type,content,trail", [
    (Msg.INTERACTION, {"action_name": "human-reply"}, None),
    (Msg.STREAM_SAMPLE,
     {"processor": {"data": "human-reply", "data_tag": 0, "data_uuid": "u1"}},
     "processor"),
])
def test_world_purges_sender_of_application_content(content_type, content, trail):
    """A registered sender delivers application content (INTERACTION or
    STREAM_SAMPLE) to a WORLD node; the world purges it with LOCAL_ERROR.

    Asserts all three __purge side-effects: close_session recorded
    (sender, LOCAL_ERROR), hosted.remove_agent ran and dropped the sender, and
    conn.remove cleared the pool entry. Cross-ref: the mis-routing that causes a
    human reply to land here is the xfail unit in tests/test_human_routing.py.
    """
    sb, conn_w, conn_h, node_w = _build_world_and_sender()

    # Registered sender: passes the get_messages membership gate + has a node-id
    # for the piggyback rewrite. Without this the content is dropped pre-dispatch.
    conn_w.register_peer("H-world", node_id="node-H")
    # Pre-seed the hosted entity so the remove_agent drop is observable.
    node_w.hosted.all_agents["H-world"] = object()

    async def _run():
        ok = await conn_h.send("W-world", channel_trail=trail,
                               content_type=content_type, content=content)
        assert ok is True
        # World chat lands on the WORLD (private) net; staged as a real Msg.
        assert len(conn_w.chat_inbox[InProcNodeConn.P2P_WORLD]) == 1
        # Drive the REAL inbound dispatch loop: gate + sort + WORLD purge branch.
        await node_w._Node__handle_network_messages()

    asyncio.run(_run())

    # (a) Session GOAWAY-closed with LOCAL_ERROR (close_session is recorded;
    # it returns [] in-process so no _dispatch_session_event is invoked).
    assert ("H-world", LOCAL_ERROR) in conn_w.closed_sessions
    # (b) hosted.remove_agent ran and dropped the sender.
    assert "H-world" in node_w.hosted.removed
    assert "H-world" not in node_w.hosted.all_agents
    # (c) conn.remove dropped the sender from the pool/identity tables.
    assert "H-world" not in conn_w.peer_id_to_pool_name
    # The inbox was fully drained by the real loop.
    assert len(conn_w.chat_inbox[InProcNodeConn.P2P_WORLD]) == 0


def test_world_does_not_purge_unregistered_sender():
    """Contrast: an UNREGISTERED sender's INTERACTION is dropped at the
    membership gate before dispatch, so the world never purges it. This pins
    that the purge fires only for a sender that passed the gate (the faithful
    bug-2 case: the human IS a registered world member).

    This is a valid negative-only assertion because the parametrized test above
    already proves the positive purge path fires for a registered sender; here
    we pin the gate precondition that gates that very mechanism.
    """
    sb, conn_w, conn_h, node_w = _build_world_and_sender()
    # NOTE: deliberately DO NOT register "H-world" on conn_w.

    async def _run():
        ok = await conn_h.send("W-world", channel_trail=None,
                               content_type=Msg.INTERACTION,
                               content={"action_name": "human-reply"})
        assert ok is True
        assert len(conn_w.chat_inbox[InProcNodeConn.P2P_WORLD]) == 1
        await node_w._Node__handle_network_messages()

    asyncio.run(_run())

    # Dropped at the gate: no purge side-effects whatsoever.
    assert conn_w.closed_sessions == []
    assert node_w.hosted.removed == []
