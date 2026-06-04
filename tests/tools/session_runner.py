#!/usr/bin/env python3
"""Cross-machine smoke harness for the Session abstraction.

Runs against the *real* libp2p shared library. Two modes:

    # On machine B (the responder/listener):
    python tests/tools/session_runner.py listen \
        --identity /tmp/runner-bob --port 30060

    # → prints its peer_id and one or more multiaddrs, then idles.

    # On machine A (the initiator/dialer), pasting B's address:
    python tests/tools/session_runner.py dial \
        --identity /tmp/runner-alice --port 30070 \
        --peer-addr "/ip4/198.51.100.7/tcp/30060/p2p/12D3KooW...bob"

    # → starts a Node, calls connect_via_session against the given multiaddr,
    #   waits for ESTABLISHED, then idles so you can observe RECONNECTING /
    #   RESUMED / LOST / CLOSED events when you disrupt the network or kill
    #   the peer.

Both modes print every SessionEvent emitted by the SessionManager dispatcher
so you can verify behaviour across NATs, networks, OSes.

Notes:
  - This harness depends on a built `unailib.cpython-*.so` and is therefore
    *not* a pytest test. It's intentionally separate from the unit suite.
  - The Node class normally pulls a profile / JWT from the root server. This
    harness uses a *stub* node with the bare minimum surface needed to drive
    Session — it does NOT join a world, run an HSM, or do anything else.
  - Press Ctrl-C to exit. Both sides will log a CLOSED event if the peer
    sends GOAWAY before disconnect.
  - IMPORTANT: dial a NON-loopback multiaddr. ``connpool.__connect`` filters out
    every ``127.0.0.1`` address, so even for a same-machine test you must paste
    one of the listener's LAN multiaddrs (e.g. ``/ip4/192.168.x.y/tcp/30060/p2p/…``),
    not the ``/ip4/127.0.0.1/…`` one — otherwise the dial fails with
    "invalid addresses".
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

# Make the package importable without installing.
_HERE = Path(__file__).resolve().parent
_REPO_SRC = _HERE.parent.parent / "src"
sys.path.insert(0, str(_REPO_SRC))

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("runner")


def _build_node(identity_dir: str, port: int, listen_ip: str = "0.0.0.0"):
    """Build a bare libp2p instance + ConnectionPools + SessionManager.

    Avoids the full ``Node`` constructor (which expects a root-server profile,
    a hosted Agent/World, etc.) — this harness is *only* about exercising the
    Session machinery."""
    from unaiverse.networking.p2p.p2p import P2P
    from unaiverse.networking.node.connpool import NodeConn
    from unaiverse.networking.session import SessionManager

    # Bring the shared library up if it isn't already. ``setup_library`` is a
    # classmethod on ``P2P`` (the package __init__ only loads the .so; it does
    # not call InitializeLibrary).
    if not getattr(P2P, "_library_initialized", False):
        P2P.setup_library()

    # Initialize the unaiverse logger singleton. connpool's dial path calls
    # ``log.cpool(..., sub=...)`` / ``log.error(..., sub=...)``; if the singleton
    # is left uninitialized those fall through to builtin ``print(..., sub=...)``
    # and raise ``TypeError: 'sub' is an invalid keyword argument for print()``,
    # crashing every dial. Mirror what ``Node`` does (node.py), but keep it to the
    # screen only (no stray .jsonl) so connpool diagnostics stay visible here.
    from unaiverse.utils.logger import log as _ulog, ALL_CHANNELS
    if getattr(_ulog, "_Logger__instance", None) is None:
        _ulog.create(name="session_runner", log_dir="/tmp",
                     active=ALL_CHANNELS, screen=ALL_CHANNELS,
                     no_color=False, file_enabled=False)

    Path(identity_dir).mkdir(parents=True, exist_ok=True)
    # No relay / DHT / WebRTC: this harness needs a directly-reachable peer.
    # ``knows_is_public=True`` skips the NAT-discovery dance.
    p2p = P2P(
        identity_dir=identity_dir,
        port=port,
        ips=[listen_ip],
        enable_relay_client=False,
        enable_relay_service=False,
        use_broad_limits=False,
        is_isolated=False,
        knows_is_public=True,
        enable_tls=False,
        dht_enabled=False,
        dht_keep=False,
        webrtc_enabled=False,
        flood_sub=False,
        log_sub="pub",  # must be a valid logger sub tag ("pub"/"prv"/"gen")
    )
    log.info("started libp2p instance peer_id=%s", p2p.peer_id)
    for a in p2p.addresses:
        log.info("  listen multiaddr: %s", a)

    # A minimal NodeConn that only uses ``p2p_public``. Tokens are disabled
    # because we don't have a root-server signature here.
    conn = NodeConn(
        max_connections=16,
        p2p_u=p2p, p2p_w=p2p,                 # use the same instance for both pools
        is_world_node=False,
        public_key=None, token=None,
    )

    sm = SessionManager(
        conn=conn,
        p2p_name=NodeConn.P2P_PUBLIC,
        local_peer_id=p2p.peer_id,
        local_node_id="runner-" + p2p.peer_id[-8:],
        local_profile_factory=lambda: {"node_id": p2p.peer_id, "runner": True},
    )
    return p2p, conn, sm


async def _drain_loop(conn, sm, stop_event: asyncio.Event):
    """Poll inbound messages + tick the SessionManager, printing every event."""
    from unaiverse.networking.node.connpool import NodeConn

    log.info("event loop running — Ctrl-C to exit")
    while not stop_event.is_set():
        try:
            chat, session = await conn.get_messages(p2p_name=NodeConn.P2P_PUBLIC)
        except Exception as e:
            log.error("get_messages failed: %s", e)
            await asyncio.sleep(0.5)
            continue

        for inb in session:
            events = await sm.deliver(inb.peer_id, inb.protocol_id, inb.sess_msg)
            for ev in events:
                _print_event(ev)

        for ev in await sm.tick():
            _print_event(ev)

        await asyncio.sleep(0.05)


def _print_event(ev) -> None:
    log.info("[SessionEvent] %-12s peer=%s… session=%s… role=%-9s data=%s",
             ev.type.value, ev.peer_id[:12], ev.session_id[:8],
             ev.role.value, ev.data)


async def _graceful_close(conn, sm, iters: int = 30) -> None:
    """On shutdown, GOAWAY every Session and keep pumping deliver+tick briefly so
    the GOAWAY goes out and the peer's GOAWAY_ACK is processed (clean CLOSED on
    both sides). Without this the peer only learns we left via liveness timeout."""
    from unaiverse.networking.node.connpool import NodeConn
    log.info("sending GOAWAY to peers (graceful close)…")
    await sm.close_all()
    for _ in range(iters):
        try:
            _chat, session = await conn.get_messages(p2p_name=NodeConn.P2P_PUBLIC)
        except Exception:
            session = []
        for inb in session:
            for ev in await sm.deliver(inb.peer_id, inb.protocol_id, inb.sess_msg):
                _print_event(ev)
        for ev in await sm.tick():
            _print_event(ev)
        await asyncio.sleep(0.05)


# ── Commands ────────────────────────────────────────────────────────────────

async def cmd_listen(args):
    p2p, conn, sm = _build_node(args.identity, args.port, args.listen_ip)
    print("\n=== LISTENER READY ===")
    print(f"peer_id:    {p2p.peer_id}")
    print("multiaddrs:")
    for a in p2p.addresses:
        print("   ", a)
    print("On the dialer machine, run:")
    print(f"  python {sys.argv[0]} dial --peer-addr '<one of the above>'")
    print()
    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await _drain_loop(conn, sm, stop)
    log.info("shutting down")
    await _graceful_close(conn, sm)
    p2p.close()


async def cmd_dial(args):
    from unaiverse.networking.node.connpool import NodeConn

    p2p, conn, sm = _build_node(args.identity, args.port, args.listen_ip)
    target_addr = args.peer_addr
    # Extract the target peer-id from the multiaddr (.../p2p/<peer-id>).
    if "/p2p/" not in target_addr:
        log.critical("--peer-addr must contain /p2p/<peer-id>")
        sys.exit(2)
    target_peer = target_addr.rsplit("/p2p/", 1)[-1]
    log.info("dialing peer=%s via %s", target_peer, target_addr)

    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    # Kick off the dial in a background task; the drain loop services its events.
    async def _do_dial():
        events = await sm.initiate(target_peer, [target_addr])
        for ev in events:
            _print_event(ev)
    asyncio.create_task(_do_dial())

    await _drain_loop(conn, sm, stop)
    log.info("shutting down")
    await _graceful_close(conn, sm)
    p2p.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("listen", "dial"):
        sp = sub.add_parser(name)
        sp.add_argument("--identity", required=True, help="Path to a private identity dir.")
        sp.add_argument("--port", type=int, default=0, help="TCP/UDP listen port (0 = auto).")
        sp.add_argument("--listen-ip", default="0.0.0.0",
                        help="Listen IP (default 0.0.0.0).")
    sub.choices["dial"].add_argument(
        "--peer-addr", required=True,
        help="Full multiaddr of the listener, e.g. /ip4/1.2.3.4/tcp/30060/p2p/12D3KooW...",
    )

    args = ap.parse_args()
    if args.cmd == "listen":
        asyncio.run(cmd_listen(args))
    elif args.cmd == "dial":
        asyncio.run(cmd_dial(args))


if __name__ == "__main__":
    main()
