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
    nid = node_id or ("test-" + p2p.peer_id[-8:])
    conn = NodeConn(max_connections=16, p2p_u=p2p, p2p_w=p2p,
                    is_world_node=False, public_key=None, token=None)
    sm = SessionManager(
        conn=conn, p2p_name=NodeConn.P2P_PUBLIC, local_peer_id=p2p.peer_id,
        local_node_id=nid, local_profile_factory=_profile_factory(p2p.peer_id, p2p.peer_id),
    )
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
    # The profile must advertise the world-side peer-id: the world host binds the
    # WorldGrant signature to the joiner's dynamic.private_peer_id, and verifies
    # it again at PRIVATE_JOIN against the peer that actually dialed.
    sm_pub = SessionManager(conn=conn, p2p_name=NodeConn.P2P_PUBLIC,
                            local_peer_id=p2p_pub.peer_id, local_node_id=node_id,
                            local_profile_factory=_profile_factory(node_id, p2p_world.peer_id),
                            decision_hook=decision_hook)
    sm_world = SessionManager(conn=conn, p2p_name=NodeConn.P2P_WORLD,
                              local_peer_id=p2p_world.peer_id, local_node_id=node_id,
                              local_profile_factory=_profile_factory(node_id, p2p_world.peer_id),
                              decision_hook=decision_hook)
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
