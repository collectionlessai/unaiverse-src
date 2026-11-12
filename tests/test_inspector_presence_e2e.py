"""The inspector attached to a node, and how the node notices that it left.

The owner's inspector connects on the public instance and announces itself with an
INSPECT_ON chat message. From then on the node feeds it at every cycle and obeys its
commands, a pause included. The node must notice when the inspector goes away
whatever the way: a peer whose Session ends cleanly is evicted on the Session event,
while its transport is still up, so the pool diff of ``update()`` never reports it.

Like the other suites built on ``build_node_shell_on_real_conn``, the scenarios run
the REAL ``Node`` methods over a REAL ``NodeConn``, with fake transports only.
"""

import asyncio

import pytest

from unaiverse.clock import clock
from unaiverse.networking.node.profile import NodeProfile
from unaiverse.networking.p2p.messages import Msg
from unaiverse.networking.session import SessionEventType
from unaiverse.networking.session.sess_msg import GRACEFUL_SHUTDOWN

from conftest import apump
from node_harness import (
    build_node_shell_on_real_conn, wire_remote_peer, mint_node_token, full_profile_dict,
)

pytestmark = pytest.mark.e2e

NODE_PEER = "12D3KooWaaa1nodeshelllllllllllllllllllllllllllllllll"
INSPECTOR_PEER = "12D3KooWiii5inspectorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr"
INSPECTOR_NODE_ID = "owner-human-node"


@pytest.fixture(autouse=True)
def _offline_node_profile(monkeypatch):
    """NodeProfile fills missing dynamic specs on construction, which includes a
    public-IP lookup over HTTP: keep the suite offline."""
    monkeypatch.setattr(NodeProfile, "_get_public_ip_address",
                        staticmethod(lambda: "203.0.113.1"))


async def _cycle(node) -> None:
    """One cycle in the order of the main loop."""
    clock.next_cycle()
    await node._Node__handle_network_connections()
    await node._Node__handle_network_messages()


class _Inspector:
    """The owner's inspector: a remote peer that completes the public handshake and
    then talks to the node with chat messages carrying the inspector-mode bit."""

    def __init__(self, p2p_public, root_private_key):
        self.p2p = p2p_public
        profile = full_profile_dict(INSPECTOR_NODE_ID, peer_id=INSPECTOR_PEER,
                                    private_peer_id=INSPECTOR_PEER)
        self.token = mint_node_token(root_private_key, profile, [INSPECTOR_PEER])
        self.sm, self.inbox = wire_remote_peer(p2p_public, INSPECTOR_PEER, profile, self.token)

    def say(self, content_type: str, content=None) -> None:
        """Queue a chat message for the node (it reads it at its next poll)."""
        self.p2p.chat_inbox.append(Msg(sender=INSPECTOR_PEER, content_type=content_type,
                                       content=content, channel="inspector",
                                       piggyback=self.token + "1"))

    async def attach(self, node) -> None:
        self.p2p.link(INSPECTOR_PEER)
        await self.sm.initiate(self.p2p.peer_id, ["/ip4/10.0.0.1/tcp/4001"])
        await asyncio.sleep(0)                 # let the dial task run...
        await self.sm.tick()                   # ...and reap it: HELLO on the wire
        await _cycle(node)                     # HELLO -> HELLO_ACK
        await apump(self.inbox, self.sm)       # HELLO_ACK -> PROFILE_OFFER
        await _cycle(node)                     # OFFER -> ESTABLISHED
        events = await apump(self.inbox, self.sm)
        assert any(e.type == SessionEventType.ESTABLISHED for e in events)
        self.say(Msg.INSPECT_ON)
        await _cycle(node)
        assert node.inspector_activated and node.inspector_peer_id == INSPECTOR_PEER

    async def leave(self) -> None:
        """Leave cleanly: a GOAWAY, with the transport still up for the farewell."""
        await self.sm.close_session(self.p2p.peer_id, GRACEFUL_SHUTDOWN)


def _inspected_node(root_keys):
    """An agent node that names INSPECTOR_NODE_ID as its inspector, and that inspector."""
    private_pem, public_pem = root_keys
    node, conn, p2p_public, _ = build_node_shell_on_real_conn(NODE_PEER, public_key=public_pem)
    node.profile.get_static_profile()["inspector_node_id"] = INSPECTOR_NODE_ID
    return node, conn, _Inspector(p2p_public, private_pem)


def test_inspector_that_leaves_with_a_goaway_is_deactivated(stepped_clock, root_keys):
    """The inspector closes its Session cleanly. The node evicts it on the CLOSED, with
    the transport still up, and must stop treating it as attached."""
    async def _scenario():
        node, conn, inspector = _inspected_node(root_keys)
        await inspector.attach(node)

        await inspector.leave()
        await _cycle(node)                            # GOAWAY -> CLOSED, purged
        assert conn.get_pool_of(INSPECTOR_PEER) is None
        assert INSPECTOR_PEER in inspector.p2p.links  # the transport did not drop
        await _cycle(node)

        assert not node.inspector_activated
        assert node.inspector_peer_id is None

    asyncio.run(_scenario())


def test_inspector_whose_transport_drops_is_deactivated(stepped_clock, root_keys):
    """The abrupt way out, which the pool diff does report, keeps working."""
    async def _scenario():
        node, conn, inspector = _inspected_node(root_keys)
        await inspector.attach(node)

        inspector.p2p.unlink(INSPECTOR_PEER)
        await _cycle(node)

        assert not node.inspector_activated
        assert node.inspector_peer_id is None

    asyncio.run(_scenario())


def test_paused_node_resumes_when_the_inspector_leaves(stepped_clock, root_keys):
    """A node paused by its inspector serves only the network. If the inspector leaves
    without saying "play", the node must notice from there and resume by itself."""
    async def _scenario():
        node, conn, inspector = _inspected_node(root_keys)
        await inspector.attach(node)
        inspector.say(Msg.INSPECT_CMD, {"cmd": "pause", "arg": None})
        await _cycle(node)
        assert node._Node__inspector_told_to_pause

        await inspector.leave()
        await asyncio.wait_for(node._wait_while_paused_by_inspector(), timeout=5.)

        assert not node._Node__inspector_told_to_pause
        assert not node.inspector_activated

    asyncio.run(_scenario())


def test_paused_node_waits_for_the_play_of_an_attached_inspector(stepped_clock, root_keys):
    """While the inspector is attached the pause holds, and its "play" ends it: the
    presence check must not drop an inspector that is still there."""
    async def _scenario():
        node, conn, inspector = _inspected_node(root_keys)
        await inspector.attach(node)
        inspector.say(Msg.INSPECT_CMD, {"cmd": "pause", "arg": None})
        await _cycle(node)

        paused = asyncio.ensure_future(node._wait_while_paused_by_inspector())
        await asyncio.sleep(0.35)                     # a few rounds of the pause loop
        assert not paused.done()

        inspector.say(Msg.INSPECT_CMD, {"cmd": "play", "arg": None})
        await asyncio.wait_for(paused, timeout=5.)
        assert node.inspector_activated and node.inspector_peer_id == INSPECTOR_PEER

    asyncio.run(_scenario())
