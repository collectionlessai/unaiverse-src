"""The request of a relay slot through the Coordinator (``NodeConn.reserve``).

A refusal of the transport becomes ``None`` for the caller, and its reason goes to the
error log there: in the browser nothing else says why a member holds no slot on its world
host, because the transport of the SDK raises the ``P2PError`` that is caught here.
"""

import asyncio

import pytest

from unaiverse.networking.node.connpool import NodeConn
from unaiverse.networking.node import connpool as connpool_mod
from unaiverse.networking.p2p.p2p import P2PError

pytestmark = pytest.mark.unit

RELAY = "12D3KooWzzz9worldhosttttttttttttttttttttttttttttttt"


class _FakeP2P:
    def __init__(self, log_sub, reserve_result=None, reserve_error=None):
        self.log_sub = log_sub
        self.peer_id = log_sub + "-self"
        self.reserve_result = reserve_result
        self.reserve_error = reserve_error

    def get_connected_peers_info(self):
        return []

    async def disconnect(self, peer_id):
        return True

    async def reserve(self, peer_id):
        if self.reserve_error is not None:
            raise self.reserve_error
        return self.reserve_result


def _conn(world):
    conn = NodeConn(max_connections=16, is_world_node=False, public_key=None, token=None,
                    local_node_id="n")
    conn.attach_session_policy(p2p_u=_FakeP2P("pub"), p2p_w=world,
                               local_profile_factory=lambda: {}, decision_hook=None)
    return conn


def test_refused_slot_is_none_and_its_reason_is_logged(monkeypatch):
    errors = []
    monkeypatch.setattr(connpool_mod.log, "error", lambda msg, *a, **k: errors.append(msg))
    reason = "no connection to the relay that is not relayed"
    conn = _conn(_FakeP2P("world", reserve_error=P2PError(reason)))
    assert asyncio.run(conn.reserve(RELAY, NodeConn.P2P_WORLD)) is None
    assert len(errors) == 1 and reason in errors[0] and RELAY[-5:] in errors[0], errors


def test_granted_slot_logs_no_error(monkeypatch):
    errors = []
    monkeypatch.setattr(connpool_mod.log, "error", lambda msg, *a, **k: errors.append(msg))
    conn = _conn(_FakeP2P("world", reserve_result="2026-11-20T18:00:00Z"))
    assert asyncio.run(conn.reserve(RELAY, NodeConn.P2P_WORLD)) == "2026-11-20T18:00:00Z"
    assert errors == []
