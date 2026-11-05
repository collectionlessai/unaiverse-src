"""Real-libp2p application-message crossing test (opt-in: ``-m p2p``).

Everything else in tests/p2p/ exercises the Session control plane (handshake,
keep-alive, GOAWAY, pool diffs). This one crosses a real *application* ``Msg``
(``Msg.INTERACTION``) over the same real wire, with real protobuf framing and
real Ed25519 sender verification, and asserts it survives the TWO Node-bound
gates of ``NodeConn.get_messages``:

  1. the membership gate: the sender must be in ``peer_id_to_pool_name`` (here
     promoted to ``IN_PUBLIC`` by ``conn.update()`` off the real transport);
  2. the JWT token gate: the piggyback RS256 token must verify against the
     receiver's ``TokenVerifier`` AND bind the sender's peer-id in ``p2p_peers``.

``build_real_node`` hardcodes ``public_key=None`` / ``token=None`` and registers
no ``routing_classifier``, so three things are wired with real values (no mocks):
the sender's token via the public ``set_token``, the receiver's verifier via the
name-mangled ``_NodeConn__token_verifier`` field, and the receiver public SM's
``_routing_classifier`` (without it ``reconcile`` short-circuits and the pool
never fills, so the membership gate would reject the message).

Run with:  ``uv run pytest -m p2p -k interaction_over_session -v``
"""

import asyncio
import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from unaiverse.networking.p2p.messages import Msg
from unaiverse.networking.node.connpool import NodeConn
from unaiverse.networking.node.tokens import TokenVerifier

from realnet_helpers import pick_dialable_addr, pump_until

pytestmark = pytest.mark.p2p


def _keypair():
    # Mirrors tests/test_tokens.py: a real RSA keypair, PEM-encoded, that signs
    # an RS256 JWT the receiver's TokenVerifier can validate end to end.
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv = key.private_bytes(serialization.Encoding.PEM,
                             serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())
    pub = key.public_key().public_bytes(serialization.Encoding.PEM,
                                         serialization.PublicFormat.SubjectPublicKeyInfo)
    return priv, pub


def test_interaction_msg_crosses_session_membership_and_token_gates(real_nodes):
    """alice (initiator) establishes a public Session to bob, then sends a real
    ``Msg.INTERACTION`` over the wire. Bob's ``get_messages`` must admit it past
    BOTH the pool-membership gate and the JWT piggyback gate, surfacing the exact
    Msg fields and rewriting the piggyback to ``node_id + inspector_bit``."""
    a_p2p, a_conn, a_sm = real_nodes.node("alice")
    b_p2p, b_conn, b_sm = real_nodes.node("bob")
    bob_addr = pick_dialable_addr(b_p2p)

    # 1) Real RSA keypair + RS256 token. p2p_peers MUST bind alice's public
    # peer-id: the token gate checks p2p_peer == msg.sender against this list.
    priv, pub = _keypair()
    token = jwt.encode(
        {"node_id": "node-A", "cv_hash": "cv",
         "p2p_peers": json.dumps([a_p2p.peer_id]),
         "exp": int(time.time()) + 3600},
        priv, algorithm="RS256")

    # 2) Sender carries the token in the piggyback (public setter).
    a_conn.set_token(token)
    # 3) Receiver verifies the piggyback against the matching public key.
    b_conn._NodeConn__token_verifier = TokenVerifier(pub)
    # 4) Receiver public pool classifier: without it reconcile() short-circuits to
    # empty and update() never fills peer_id_to_pool_name, so the membership gate
    # would drop the message before the token is even inspected.
    b_sm._routing_classifier = NodeConn._route_public

    captured = {}

    async def _scenario():
        # 5) Establish a public Session (verbatim from test_session_realnet).
        await asyncio.wait_for(a_sm.initiate(b_p2p.peer_id, [bob_addr]), timeout=10.0)
        await pump_until(
            [(a_conn, a_sm, NodeConn.P2P_PUBLIC), (b_conn, b_sm, NodeConn.P2P_PUBLIC)],
            predicate=lambda: a_sm.is_established(b_p2p.peer_id)
            and b_sm.is_established(a_p2p.peer_id),
            timeout=20.0)

        # 6) Promote alice into bob's IN_PUBLIC pool off the real transport, so the
        # membership gate of get_messages will let her chat through.
        added_b, _ = await b_conn.update()
        assert a_p2p.peer_id in added_b.get(NodeConn.IN_PUBLIC, set()), \
            "alice was not admitted into bob's IN_PUBLIC pool by update()"
        assert b_conn.peer_id_to_pool_name.get(a_p2p.peer_id) == NodeConn.IN_PUBLIC

        # 7) Cross a real application Msg over the wire.
        assert await a_conn.send(b_p2p.peer_id, None, Msg.INTERACTION,
                                 {"text": "hello over the real wire"})

        # 8) Bounded custom drain (NOT pump_until, which discards the chat list):
        # poll bob's get_messages, keep delivering any Session frames to keep the
        # link alive, break as soon as a chat message lands.
        loop = asyncio.get_event_loop()
        deadline = loop.time() + 15.0
        chat = []
        while loop.time() < deadline:
            await asyncio.sleep(0.05)
            chat, session = await b_conn.get_messages(NodeConn.P2P_PUBLIC)
            for inb in session:
                await b_sm.deliver(inb.peer_id, inb.protocol_id, inb.sess_msg)
            if chat:
                break
        captured["chat"] = chat

    asyncio.run(_scenario())

    chat = captured.get("chat", [])
    assert chat, "no INTERACTION message survived the real wire + the two gates within 15s"
    m = chat[0]
    # The Msg crossed real protobuf framing and the real sender-mismatch check.
    assert m.sender == a_p2p.peer_id
    assert m.content_type == Msg.INTERACTION
    assert m.content == {"text": "hello over the real wire"}
    assert m.channel == f"{a_p2p.peer_id}::dm:{b_p2p.peer_id}-{Msg.INTERACTION}"
    # On a successful token verification get_messages replaces the raw piggyback
    # with node_id + the inspector-mode bit (the dummy "0" appended by send).
    assert m.piggyback == "node-A" + "0"
    # And it caches the verified bearer token under the sender's peer-id.
    assert b_conn.peer_id_to_token[a_p2p.peer_id] == token
