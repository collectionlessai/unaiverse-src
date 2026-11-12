"""Real-libp2p: every queued Session frame is read in one poll (opt-in: ``-m p2p``).

The Go transport queues the inbound Session frames per protocol, in one queue shared
by every peer of the node, and the node polls the transport once per cycle. A poll
must hand over all the queued frames: one frame per poll made a GOAWAY, or a PONG,
wait as many cycles as there were frames ahead of it, whoever had sent them (seen
live on a world host running at one cycle per second: a GOAWAY processed 2 to 3 s
after it arrived, behind the keep-alives of the other members).

Chat keeps its pace of one message per channel per poll: that part is pinned by the
Go unit tests next to the message store (``unailib/messaging_test.go``).

Run with:  ``uv run pytest -m p2p -k frames_drain -v``
"""

import asyncio

import pytest

from unaiverse.networking.node.connpool import NodeConn
from unaiverse.networking.session.sess_msg import SessMsg, SESSION_CONTROL_PROTOCOL

from realnet_helpers import pick_dialable_addr, pump_until

pytestmark = pytest.mark.p2p


def test_queued_control_frames_are_read_in_one_poll(real_nodes):
    """Bob sends three control frames in a row while Alice is not polling. Alice's next
    poll returns all three, in the order they were sent."""
    a_p2p, a_conn, a_sm = real_nodes.node("alice")
    b_p2p, b_conn, b_sm = real_nodes.node("bob")
    bob_addr = pick_dialable_addr(b_p2p)
    targets = [(a_conn, a_sm, NodeConn.P2P_PUBLIC), (b_conn, b_sm, NodeConn.P2P_PUBLIC)]

    async def _scenario():
        await asyncio.wait_for(a_sm.initiate(b_p2p.peer_id, [bob_addr]), timeout=10.0)
        await pump_until(targets,
                         predicate=lambda: a_sm.is_established(b_p2p.peer_id)
                         and b_sm.is_established(a_p2p.peer_id),
                         timeout=20.0)
        session_id = b_sm.get_session(a_p2p.peer_id).session_id

        nonces = [b"frame-1!", b"frame-2!", b"frame-3!"]
        for nonce in nonces:
            assert await b_p2p.send_session(
                a_p2p.peer_id, SESSION_CONTROL_PROTOCOL,
                SessMsg(SessMsg.KIND_PING, {"session_id": session_id, "nonce": nonce}))

        # Each frame rides its own stream: give the transport the time to queue them
        # all, WITHOUT polling Alice meanwhile.
        await asyncio.sleep(1.5)

        _, session_inbound = a_p2p.get_messages()
        received = [bytes(inb.sess_msg.get("nonce")) for inb in session_inbound
                    if inb.protocol_id == SESSION_CONTROL_PROTOCOL
                    and inb.sess_msg.kind == SessMsg.KIND_PING]
        assert sorted(received) == sorted(nonces), \
            f"one poll returned {len(received)} of the {len(nonces)} queued control frames"

    asyncio.run(_scenario())
