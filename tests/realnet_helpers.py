"""Pure helpers for the real-libp2p (`-m p2p`) integration tests.

Kept in a uniquely-named module (not ``conftest``) so test files can
``from realnet_helpers import ...`` without colliding with the root
``tests/conftest.py`` under pytest's prepend import mode. ``conftest.py``
imports the node builders from here for its fixtures.

See ``tests/p2p/conftest.py`` for the constraints (listen on 0.0.0.0, dial a
non-loopback address, close every P2P to free instance slots).
"""

import asyncio

import pytest

from unaiverse.networking.node.connpool import NodeConn


def _profile_factory(node_id: str, world_peer_id: str | None = None):
    """Real-shaped enough for the grant flow: the world host binds the grant
    signature to the joiner's ``dynamic.private_peer_id``."""
    profile = {"node_id": node_id, "test": True}
    if world_peer_id is not None:
        profile["dynamic"] = {"private_peer_id": world_peer_id}
    return lambda: profile


def build_real_node(identity_dir: str, port: int = 0, listen_ip: str = "0.0.0.0",
                    log_sub: str = "pub", node_id: str | None = None):
    """A single-instance real node: (p2p, conn, sm_public).

    Mirrors ``session_runner._build_node`` (relay/DHT/WebRTC off,
    ``knows_is_public=True``) but does not touch logging."""
    from unaiverse.networking.p2p.p2p import P2P
    from unaiverse.networking.session import SessionManager

    if not getattr(P2P, "_library_initialized", False):
        P2P.setup_library()

    p2p = P2P(
        identity_dir=identity_dir, port=port, ips=[listen_ip],
        enable_relay_client=False, enable_relay_service=False, use_broad_limits=False,
        is_isolated=False, knows_is_public=True, enable_tls=False,
        dht_enabled=False, dht_keep=False, webrtc_enabled=False, flood_sub=False,
        log_sub=log_sub,
    )
    nid = node_id or ("test-" + p2p.peer_id[-5:])
    # Single P2P for both pools as an economy: this builder returns only sm_public and
    # its callers exercise the public network alone, so the same instance on both sides
    # never produces a same-peer-on-both-networks case (that anomaly, and the cross-pool
    # guard that defends it, is exercised by the mocked char-test instead). World-join
    # tests that need two distinct peer-ids use build_real_world_node.
    conn = NodeConn(max_connections=16, p2p_u=p2p, p2p_w=p2p,
                    is_world_node=False, public_key=None, token=None)
    sm = SessionManager(
        p2p=p2p, p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=p2p.peer_id,
        local_node_id=nid, local_profile_factory=_profile_factory(p2p.peer_id, p2p.peer_id),
    )
    # Register the (single, public) manager on the conn so conn.get_messages resolves the
    # instance through the SM (S18): the SM is the instance, conn.get_messages reads its
    # P2P handle via _sm_by_name. In production attach_session_policy does this; this
    # builder bypasses it, so wire it explicitly like build_real_world_node does.
    conn._sm_public = sm
    conn.set_session_managers(sm)
    return p2p, conn, sm


def build_real_world_node(idir_pub: str, idir_world: str, decision_hook=None,
                          node_id: str = "node"):
    """A two-instance node (public + world) sharing one NodeConn.

    Returns (p2p_pub, p2p_world, conn, sm_public, sm_world). The conn is built
    with ``p2p_u=p2p_pub, p2p_w=p2p_world`` so ``conn.<op>(P2P_WORLD, ...)`` hits
    the world instance — this is what exercises real cross-instance world-key
    signing in the world-join handshake."""
    from unaiverse.networking.p2p.p2p import P2P
    from unaiverse.networking.session import SessionManager

    if not getattr(P2P, "_library_initialized", False):
        P2P.setup_library()

    common = dict(port=0, ips=["0.0.0.0"], enable_relay_client=False,
                  enable_relay_service=False, use_broad_limits=False, is_isolated=False,
                  knows_is_public=True, enable_tls=False, dht_enabled=False,
                  dht_keep=False, webrtc_enabled=False, flood_sub=False)
    p2p_pub = P2P(identity_dir=idir_pub, log_sub="pub", **common)
    p2p_world = P2P(identity_dir=idir_world, log_sub="prv", **common)

    conn = NodeConn(max_connections=16, p2p_u=p2p_pub, p2p_w=p2p_world,
                    is_world_node=False, public_key=None, token=None)
    # The profile must advertise the world-side peer-id: the world host pre-admits
    # the joiner's dynamic.private_peer_id (from the root-attested p2p_peers) and the
    # world Session re-verifies the joiner token on its PROFILE handshake.
    # Mirror attach_session_policy: the public manager routes by direction (no membership,
    # never cuts), the world one classifies against the snapshot and cuts unclassifiable
    # peers on its P2P instance. Without the routing_classifier, sm.reconcile short-circuits
    # to empty and conn.update() produces no pool diff.
    sm_pub = SessionManager(p2p=p2p_pub, p2p_name=NodeConn.P2P_PUBLIC,
                            local_peer_id=p2p_pub.peer_id, local_node_id=node_id,
                            local_profile_factory=_profile_factory(node_id, p2p_world.peer_id),
                            decision_hook=decision_hook,
                            routing_classifier=NodeConn._route_public)
    sm_world = SessionManager(p2p=p2p_world, p2p_name=NodeConn.P2P_WORLD,
                              local_peer_id=p2p_world.peer_id, local_node_id=node_id,
                              local_profile_factory=_profile_factory(node_id, p2p_world.peer_id),
                              decision_hook=decision_hook,
                              routing_classifier=NodeConn._route_world)
    # Wire the SMs into the conn so conn.update() / _reconcile_instance / _sm_of see them.
    # This builder bypasses Node.__init__, which is where attach_session_policy normally does
    # this; without it conn._sm_* stay None and conn.update() trips over sm.reconcile.
    conn._sm_public, conn._sm_world = sm_pub, sm_world
    conn.set_session_managers(sm_pub, sm_world)
    return p2p_pub, p2p_world, conn, sm_pub, sm_world


def pick_dialable_addr(p2p) -> str:
    """First non-loopback dialable multiaddr from ``p2p.addresses``.

    Prefers a plain ``/tcp/`` (non-ws) address, then QUIC, then anything
    non-loopback. ``pytest.skip`` if the host has no usable (LAN) address."""
    cands = [a for a in p2p.addresses if "127.0.0.1" not in a and "/p2p/" in a]
    tcp = [a for a in cands if "/tcp/" in a and "/ws/" not in a]
    quic = [a for a in cands if "/quic-v1" in a and "webtransport" not in a]
    chosen = tcp or quic or cands
    if not chosen:
        pytest.skip("no non-loopback dialable address available (no usable NIC)")
    return chosen[0]


async def pump_until(targets, predicate, on_event=None, timeout: float = 20.0,
                     interval: float = 0.05):
    """Round-robin drain real Session traffic until ``predicate()`` holds.

    Args:
        targets: list of ``(conn, sm, p2p_name)`` triples. Each iteration calls
            ``conn.get_messages(p2p_name)``, delivers every inbound to ``sm``, and
            ticks ``sm``. World-join nodes contribute two triples (one per
            instance).
        predicate: zero-arg callable; loop stops as soon as it returns True.
        on_event: optional awaitable ``async def (event)`` invoked for every
            emitted ``SessionEvent`` (the world-join orchestrator). May itself
            call ``sm.initiate``/``sm.close_session``.
        timeout: hard wall-clock bound; raises ``AssertionError`` on expiry.

    Returns: ``dict[SessionManager, list[SessionEvent]]`` of everything emitted.
    """
    loop = asyncio.get_event_loop()
    collected: dict = {}
    deadline = loop.time() + timeout

    async def _handle(sm, events):
        for ev in events:
            collected.setdefault(sm, []).append(ev)
            if on_event is not None:
                await on_event(ev)

    while loop.time() < deadline:
        for conn, sm, p2p_name in targets:
            _, session = await conn.get_messages(p2p_name=p2p_name)
            for inb in session:
                await _handle(sm, await sm.deliver(inb.peer_id, inb.protocol_id, inb.sess_msg))
            await _handle(sm, await sm.tick())
        if predicate():
            return collected
        await asyncio.sleep(interval)
    raise AssertionError(
        f"pump_until timed out after {timeout}s; collected="
        + "{" + ", ".join(f"{sm.p2p_name}:{[e.type.value for e in evs]}"
                          for sm, evs in collected.items()) + "}")
