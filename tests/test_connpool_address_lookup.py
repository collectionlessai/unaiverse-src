"""The lookup of fresh addresses that the Node layer gives the Coordinator.

It reaches the public instance only: the root server holds the addresses of the public
instance of a node, and on the world instance the rendezvous keeps them current.
"""

import pytest

from unaiverse.networking.node.connpool import NodeConn

pytestmark = pytest.mark.unit

PEER = "12D3KooWzzz9peerrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr"


class _FakeP2P:
    def __init__(self, log_sub):
        self.log_sub = log_sub
        self.peer_id = log_sub + "-self"

    def get_connected_peers_info(self):
        return []


def test_lookup_reaches_the_initiators_of_the_public_instance_only():
    async def _lookup(node_id):
        return None
    conn = NodeConn(max_connections=16, is_world_node=False, public_key=None, token=None,
                    local_node_id="n")
    conn.attach_session_policy(p2p_u=_FakeP2P("pub"), p2p_w=_FakeP2P("world"),
                               local_profile_factory=lambda: {}, decision_hook=None,
                               address_lookup=_lookup)
    public = conn._sm_public.ensure(PEER, ["/ip4/127.0.0.1/tcp/1/p2p/" + PEER])
    world = conn._sm_world.ensure(PEER, ["/ip4/127.0.0.1/tcp/2/p2p/" + PEER])
    assert public._address_lookup is _lookup
    assert world._address_lookup is None
