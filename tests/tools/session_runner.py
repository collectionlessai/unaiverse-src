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
  - Logging honours the same env vars as Node: NODE_LOG=1 writes a JSONL next
    to --log-dir (named session_runner_<mode>.jsonl), NODE_LOG_APPEND=1 keeps
    it across restarts, NODE_LOG_CHANNELS selects channels, NODE_LIBP2PLOG=1
    captures the Go/libp2p output into the same file.
  - --identity can point to an existing ~/.unaiverse/<node_id> dir to reuse a
    known peer identity (never run two processes with the same identity).
  - IMPORTANT: dial a NON-loopback multiaddr. ``connpool.__connect`` filters out
    every ``127.0.0.1`` address, so even for a same-machine test you must paste
    one of the listener's LAN multiaddrs (e.g. ``/ip4/192.168.x.y/tcp/30060/p2p/…``),
    not the ``/ip4/127.0.0.1/…`` one — otherwise the dial fails with
    "invalid addresses".
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

# Make the package importable without installing.
_HERE = Path(__file__).resolve().parent
_REPO_SRC = _HERE.parent.parent / "src"
sys.path.insert(0, str(_REPO_SRC))

# The unaiverse package is importable only after the sys.path tweak above, so
# every unaiverse import in this harness is below this line. Use the project's
# channel-based logger (same one connpool/session speak); the runner's own
# messages ride the SESSION channel.
from unaiverse.utils.logger import log


def _build_node(identity_dir: str, port: int, listen_ip: str = "0.0.0.0",
                log_dir: str = ".", log_name: str = "session_runner"):
    """Build a bare libp2p instance + NodeConn + SessionManager.

    Avoids the full ``Node`` constructor (which expects a root-server profile,
    a hosted Agent/World, etc.) — this harness is *only* about exercising the
    Session machinery."""
    from unaiverse.custom import Custom
    from unaiverse.networking.p2p.p2p import P2P
    from unaiverse.networking.node.connpool import NodeConn
    from unaiverse.networking.session import SessionManager

    # Initialize the unaiverse logger singleton BEFORE the Go library: the
    # fd-capture that routes libp2p stdout/stderr into the JSONL needs the
    # logger to exist. connpool's dial path calls ``log.cpool(..., sub=...)`` /
    # ``log.error(..., sub=...)``; if the singleton is left uninitialized those
    # fall through to builtin ``print(..., sub=...)`` and raise ``TypeError``,
    # crashing every dial. Mirror what ``Node`` does (node.py): honour NODE_LOG,
    # NODE_LOG_APPEND and NODE_LOG_CHANNELS; default to everything on screen
    # (this is a debug harness).
    from unaiverse.utils.logger import ALL_CHANNELS, ALWAYS_ON_CHANNELS
    if getattr(log, "_Logger__instance", None) is None:
        if Custom.LOG_CHANNELS:
            _valid = {c.name: c for c in ALL_CHANNELS}
            _wanted = frozenset(_valid[n] for n in
                                (s.strip().upper() for s in Custom.LOG_CHANNELS.split(",")) if n in _valid)
            active = screen = ALWAYS_ON_CHANNELS | _wanted
        else:
            active = screen = ALL_CHANNELS
        log.create(name=log_name, log_dir=log_dir,
                   active=active, screen=screen, no_color=False,
                   file_enabled=Custom.LOG_TO_FILE, file_append=Custom.LOG_APPEND)

    # Bring the shared library up if it isn't already. ``setup_library`` is a
    # classmethod on ``P2P`` (the package __init__ only loads the .so; it does
    # not call InitializeLibrary). Same call as Node: with NODE_LIBP2PLOG=1 the
    # Go output lands in the JSONL as P2P-channel records.
    if not getattr(P2P, "_library_initialized", False):
        P2P.setup_library(enable_logging=Custom.LIBP2PLOG, unai_logger=log)

    Path(identity_dir).mkdir(parents=True, exist_ok=True)
    # No relay / DHT / WebRTC: this harness needs a directly-reachable peer.
    # ``knows_is_public=True`` skips the NAT-discovery dance.
    p2p = P2P(
        identity_dir=identity_dir,
        port=port,
        enable_relay_client=True,
        enable_relay_service=False,
        use_broad_limits=False,
        is_isolated=False,
        knows_is_public=False,
        enable_tls=False,
        dht_enabled=True,
        dht_keep=True,
        webrtc_enabled=False,
        flood_sub=False,
        log_sub="pub",  # must be a valid logger sub tag ("pub"/"prv"/"gen")
    )
    log.session(f"started libp2p instance peer_id={p2p.peer_id}")
    for a in p2p.addresses:
        log.session(f"  listen multiaddr: {a}")

    # A minimal NodeConn that only uses ``p2p_public``. Tokens are disabled
    # because we don't have a root-server signature here.
    conn = NodeConn(
        max_connections=16,
        is_world_node=False,
        public_key=None, token=None,
    )

    sm = SessionManager(
        p2p=p2p,
        p2p_name=NodeConn.P2P_PUBLIC,
        local_peer_id=p2p.peer_id,
        local_node_id="runner-" + p2p.peer_id[-5:],
        local_profile_factory=lambda: {"node_id": p2p.peer_id, "runner": True},
    )
    return p2p, conn, sm


async def _drain_loop(conn, sm, stop_event: asyncio.Event):
    """Poll inbound messages + tick the SessionManager, printing every event."""
    from unaiverse.networking.node.connpool import NodeConn

    log.session("event loop running — Ctrl-C to exit")
    while not stop_event.is_set():
        try:
            chat, session = await conn.get_messages(p2p_name=NodeConn.P2P_PUBLIC)
        except Exception as e:
            log.error(f"get_messages failed: {e}")
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
    log.session(f"[SessionEvent] {ev.type.value:<12} peer=...{ev.peer_id[-5:]} "
                f"session={ev.session_id[:8]}… role={ev.role.value:<9} data={ev.data}")


async def _graceful_close(conn, sm, iters: int = 30) -> None:
    """On shutdown, GOAWAY every Session and keep pumping deliver+tick briefly so
    the GOAWAY goes out and the peer's GOAWAY_ACK is processed (clean CLOSED on
    both sides). Without this the peer only learns we left via liveness timeout."""
    from unaiverse.networking.node.connpool import NodeConn
    log.session("sending GOAWAY to peers (graceful close)…")
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


# Commands

async def cmd_listen(args):
    p2p, conn, sm = _build_node(args.identity, args.port, args.listen_ip,
                                log_dir=args.log_dir, log_name="session_runner_listen")
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
    log.session("shutting down")
    await _graceful_close(conn, sm)
    p2p.close()


async def cmd_dial(args):
    from unaiverse.networking.node.connpool import NodeConn

    p2p, conn, sm = _build_node(args.identity, args.port, args.listen_ip,
                                log_dir=args.log_dir, log_name="session_runner_dial")
    target_addr = args.peer_addr
    # Extract the target peer-id from the multiaddr (.../p2p/<peer-id>).
    if "/p2p/" not in target_addr:
        # NB: log.critical on the custom logger raises GenException; use error here
        # so the harness exits cleanly via sys.exit instead.
        log.error("--peer-addr must contain /p2p/<peer-id>")
        sys.exit(2)
    target_peer = target_addr.rsplit("/p2p/", 1)[-1]
    log.session(f"dialing peer={target_peer} via {target_addr}")

    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    # Kick off the dial in a background task; the drain loop services its events.
    # With --redial-every N the initiate is re-issued every N seconds: on a live
    # Session it is a documented no-op, after a FAILED/LOST/CLOSED it re-dials.
    # This reproduces an application-level retry loop (quarantine refusals show
    # up as synthetic FAILED events with a "quarantined" reason).
    async def _do_dial():
        while True:
            try:
                events = await sm.initiate(target_peer, [target_addr])
            except Exception as e:
                log.error(f"initiate toward ...{target_peer[-5:]} failed: {e}")
                events = []
            for ev in events:
                _print_event(ev)
            if args.redial_every <= 0 or stop.is_set():
                return
            try:
                await asyncio.wait_for(stop.wait(), timeout=args.redial_every)
                return  # stop was set while waiting
            except asyncio.TimeoutError:
                continue
    dial_task = asyncio.create_task(_do_dial())

    await _drain_loop(conn, sm, stop)
    dial_task.cancel()
    log.session("shutting down")
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
        sp.add_argument("--log-dir", default=".",
                        help="Directory for the JSONL log (with NODE_LOG=1; default: cwd).")
    sub.choices["dial"].add_argument(
        "--peer-addr", required=True,
        help="Full multiaddr of the listener, e.g. /ip4/1.2.3.4/tcp/30060/p2p/12D3KooW...",
    )
    sub.choices["dial"].add_argument(
        "--redial-every", type=float, default=0.0,
        help="Re-issue the initiate every N seconds (0 = dial once). No-op while "
             "the Session is alive; retries after a terminal state.",
    )

    args = ap.parse_args()
    if args.cmd == "listen":
        asyncio.run(cmd_listen(args))
    elif args.cmd == "dial":
        asyncio.run(cmd_dial(args))


if __name__ == "__main__":
    main()
