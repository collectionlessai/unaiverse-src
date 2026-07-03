"""Real-libp2p perimeter-security repros (opt-in: ``-m p2p``).

Two wire-level behaviours the mocked suite cannot reach (the Go transport is
stubbed there), so these exercise the real libp2p stack:

* A rendezvous ``:rv`` roster is authoritative only from the world host named in
  the topic. A roster published by anyone else must be dropped, and a forged high
  ``update_count`` must not wedge later legitimate host updates.
* A peer whose only traffic is a pubsub subscription (never a stream) is invisible
  to ``get_connected_peers_info`` (Go ``friendlyPeers``) yet, as a directly-connected
  topic subscriber, receives every world broadcast. The world perimeter must see it
  among the topic subscribers and cut it.

Run with:  ``uv run pytest -m p2p -k perimeter_security -v``
"""

import asyncio

import pytest

from unaiverse.networking.p2p.messages import Msg
from unaiverse.networking.node.connpool import NodeConn

from realnet_helpers import pick_dialable_addr

pytestmark = pytest.mark.p2p


def _roster_bytes(sender_peer_id: str, member_peer_ids: list[str], update_count: int) -> bytes:
    """A rendezvous roster on the wire: a WORLD_AGENTS_LIST Msg whose json_content is
    ``{"peers": [...], "update_count": N}``, exactly what the world host publishes and
    what Go's readFromSubscription decodes. Peer ids must be real (Go decodes each into a
    peer.ID); addrs is left empty (an invalid multiaddr would make Go reject the decode and
    mask the authorization check under test)."""
    peers = [{"id": pid, "addrs": [], "direction": "inbound", "misc": 0} for pid in member_peer_ids]
    msg = Msg(sender=sender_peer_id, content_type=Msg.WORLD_AGENTS_LIST,
              content={"peers": peers, "update_count": update_count})
    return msg.to_bytes()


async def _wait_until(predicate, timeout: float = 8.0, interval: float = 0.25) -> bool:
    """Poll a zero-arg predicate until true or timeout; returns the final value."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return predicate()


def _rv_peer_ids(p2p) -> set:
    """The peer ids currently in this instance's rendezvous snapshot (empty if none)."""
    state = p2p.get_rendezvous_peers_info()
    if not state:
        return set()
    return {p["id"] for p in state.get("peers", [])}


def test_perimeter_security_b3_forged_rendezvous_rejected(real_nodes):
    """A ``:rv`` roster from a non-host member is dropped; the host's own roster is applied.

    Topology: host H owns the topic ``<H>::ps:rv``; victim V subscribes to it; intruder X
    also joins the topic and connects to V. X publishes a forged roster (listing itself,
    with a huge update_count); H publishes a legitimate one. After the fix V's rendezvous
    snapshot must reflect ONLY H's roster, and its update_count must be H's small value,
    proving the forged high tag never wedged the monotonic gate.
    """
    h_p2p, _, _ = real_nodes.node("b3-host")
    v_p2p, _, _ = real_nodes.node("b3-victim")
    x_p2p, _, _ = real_nodes.node("b3-intruder")

    topic = f"{h_p2p.peer_id}::ps:rv"

    async def _scenario():
        # All three join the topic mesh; V connects to both publishers so gossipsub can
        # deliver either roster to it.
        for p in (h_p2p, v_p2p, x_p2p):
            p.subscribe_to_topic(topic)
        v_p2p.connect_to([pick_dialable_addr(h_p2p)])
        v_p2p.connect_to([pick_dialable_addr(x_p2p)])

        # Let subscriptions propagate and the gossipsub mesh graft.
        await asyncio.sleep(2.0)

        # X forges a roster naming itself a member, with a poisoning-high tag.
        x_p2p.broadcast_message(topic, _roster_bytes(x_p2p.peer_id, [x_p2p.peer_id], 999_999))
        await asyncio.sleep(2.0)

        # The forged roster must NOT have been applied on the victim.
        assert x_p2p.peer_id not in _rv_peer_ids(v_p2p), \
            "forged rendezvous roster from a non-host peer was accepted"

        # The genuine host roster (small tag) must be applied — proving delivery works and
        # the forged high tag did not wedge the monotonic gate.
        h_p2p.broadcast_message(topic, _roster_bytes(h_p2p.peer_id, [h_p2p.peer_id], 7))
        applied = await _wait_until(lambda: h_p2p.peer_id in _rv_peer_ids(v_p2p), timeout=8.0)
        assert applied, "host's own rendezvous roster was not applied on the victim"

        state = v_p2p.get_rendezvous_peers_info()
        assert state is not None and state.get("update_count") == 7, \
            f"host update was rejected/wedged; snapshot={state}"
        assert x_p2p.peer_id not in _rv_peer_ids(v_p2p), \
            "intruder still present after the host roster replaced it"

    asyncio.run(_scenario())


def test_perimeter_security_b4_pubsub_only_intruder_is_cut(real_nodes):
    """A pubsub-only peer on the world instance is invisible to friendlyPeers but present
    among the topic subscribers, and the world perimeter cut-sweep disconnects it.

    Host H runs a world instance subscribed to a world topic; intruder X connects to that
    instance and only subscribes to the topic (no Session, no stream). X therefore never
    enters friendlyPeers (get_connected_peers_info) yet receives world broadcasts. After
    conn.update() runs the world reconcile, X must be cut.
    """
    _, h_world, h_conn, _, _ = real_nodes.world_node("b4-host")
    x_p2p, _, _ = real_nodes.node("b4-intruder")

    topic = f"{h_world.peer_id}::ps:data"

    async def _scenario():
        # Host world instance and intruder share a world topic; intruder dials the world
        # instance and subscribes only (it never opens a Session stream).
        h_world.subscribe_to_topic(topic)
        x_p2p.subscribe_to_topic(topic)
        x_p2p.connect_to([pick_dialable_addr(h_world)])

        # The intruder must surface among the topic subscribers (the exfiltration channel)...
        in_topic = await _wait_until(lambda: x_p2p.peer_id in h_world.get_topic_peers(), timeout=8.0)
        assert in_topic, "intruder never joined the host's world topic (test setup)"

        # ...but must be INVISIBLE to friendlyPeers, since it opened no application stream.
        # This is exactly the blindness that let a pubsub-only peer evade the old perimeter.
        friendly = {c["id"] for c in (h_world.get_connected_peers_info() or [])}
        assert x_p2p.peer_id not in friendly, \
            "intruder unexpectedly in friendlyPeers; it should be pubsub-only"

        # The intruder is not a world member, so the perimeter cut-sweep must disconnect it.
        assert x_p2p.peer_id not in h_conn._sm_world.world_agents_set
        await h_conn.update()

        cut = await _wait_until(lambda: x_p2p.peer_id not in h_world.get_topic_peers(), timeout=8.0)
        assert cut, "pubsub-only non-member was never cut from the world topic"

    asyncio.run(_scenario())
