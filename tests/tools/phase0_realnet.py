#!/usr/bin/env python3
"""
Phase 0 measurements of the concurrency plan, over the real libp2p library and two processes.

    # A stopped receiver (kill -STOP): how long does a direct send block today?
    python tests/tools/phase0_realnet.py stuck-send

    # A receiver busy in pure Python for 60 s: does its Go side keep receiving?
    python tests/tools/phase0_realnet.py go-busy

The parent process is the sender; it starts the receiver as a child process (this same script with --child), reads
its address from the child's first output line, and prints a short report. Raw P2P instances only: no Node, no
Session, no root server. Not a pytest test (it needs the built Go library, a usable NIC and two processes).
"""
import os
import sys
import json
import time
import signal
import asyncio
import argparse
import tempfile
import threading
import subprocess

CHANNEL_SUFFIX = "-misc"
BUSY_S = 60.0  # How long the receiver of go-busy holds the GIL
SEND_TIMEOUT_S = 120.0  # How long the sender waits for one blocked send before giving up on it


def _make_p2p(identity_dir: str) -> object:
    """Creates a minimal real P2P instance (no relay, no WebRTC, forced public).

    Args:
        identity_dir: The folder where the identity of the instance is kept.

    Returns:
        The P2P instance.
    """
    from unaiverse.networking.p2p.p2p import P2P
    if not getattr(P2P, "_library_initialized", False):
        P2P.setup_library()
    return P2P(identity_dir=identity_dir, port=0, ips=["0.0.0.0"], enable_relay_client=False,
               enable_relay_service=False, use_broad_limits=False, is_isolated=False, knows_is_public=True,
               enable_tls=False, probe_peers=[], webrtc_enabled=False, flood_sub=False, log_sub="pub")


def _dialable(p2p: object) -> list[str]:
    """Gets the non-loopback dialable addresses of an instance (TCP first).

    Args:
        p2p: The P2P instance.

    Returns:
        The list of the addresses.
    """
    cands = [a for a in p2p.addresses if "127.0.0.1" not in a and "/p2p/" in a]
    return sorted(cands, key=lambda a: 0 if ("/tcp/" in a and "/ws" not in a) else 1)


def child(mode: str, topic: str) -> None:
    """Runs the receiver: prints its peer id and addresses, then behaves according to the mode.

    Args:
        mode: "stuck-send" (just stays alive, the parent stops it) or "go-busy" (busy loop, then pops).
        topic: The pubsub topic to subscribe to (go-busy).
    """
    with tempfile.TemporaryDirectory() as identity_dir:
        p2p = _make_p2p(identity_dir)
        if topic:
            p2p.subscribe_to_topic(topic)
        print(json.dumps({"peer_id": p2p.peer_id, "addresses": _dialable(p2p)}), flush=True)
        if mode == "go-busy":
            sys.stdin.readline()  # The parent says when to start being busy

            # Pure-Python busy loop that holds the GIL: no await, no sleep, no torch
            t0 = time.perf_counter()
            x = 0
            while time.perf_counter() - t0 < BUSY_S:
                x += 1

            # One pop returns at most one message per channel: drain until empty to count what Go kept
            popped = 0
            while batch := p2p.pop_messages():
                popped += len(batch)
            print(json.dumps({"busy_s": BUSY_S, "popped": popped}), flush=True)
        else:
            while True:
                time.sleep(1.0)  # Stays alive until the parent stops and kills it
        p2p.close()


async def parent_stuck_send() -> None:
    """Measures how long a direct send blocks when the receiver is stopped with SIGSTOP (async)."""
    proc = subprocess.Popen([sys.executable, __file__, "--child", "stuck-send"], stdout=subprocess.PIPE, text=True)
    info = json.loads(proc.stdout.readline())
    with tempfile.TemporaryDirectory() as identity_dir:
        p2p = _make_p2p(identity_dir)
        peer_id, _ = await p2p.connect(info["addresses"])
        channel = f"{p2p.peer_id}::dm:{info['peer_id']}{CHANNEL_SUFFIX}"
        print(f"connected to the receiver: {peer_id is not None}")

        def timed_send(payload: bytes) -> float | None:
            """Sends one message in a thread, waiting at most SEND_TIMEOUT_S for it.

            Args:
                payload: The bytes to send.

            Returns:
                The seconds the send took, or None if it was still blocked at the timeout.
            """
            result = {}

            def run() -> None:
                """Sends and records the elapsed time."""
                t0 = time.perf_counter()
                try:
                    p2p.send_message_to_peer(channel, payload)
                    result["s"] = time.perf_counter() - t0
                except Exception as e:
                    result["s"] = time.perf_counter() - t0
                    result["err"] = str(e)
            t = threading.Thread(target=run, daemon=True)
            t.start()
            t.join(SEND_TIMEOUT_S)
            if "err" in result:
                print(f"    send failed after {result['s']:.1f}s: {result['err'][:100]}")
            return result.get("s")

        print("receiver running: 3 sends of 1 MB")
        for _ in range(3):
            print(f"    {timed_send(os.urandom(1 << 20)):.3f}s")
        os.kill(proc.pid, signal.SIGSTOP)
        print("receiver stopped (SIGSTOP): sends of 4 MB, more than the stream flow-control window")
        for i in range(4):
            s = timed_send(os.urandom(4 << 20))
            print(f"    send {i + 1}: " + (f"{s:.1f}s" if s is not None else f"still blocked after {SEND_TIMEOUT_S}s"))
            if s is None:
                break
        os.kill(proc.pid, signal.SIGCONT)
        proc.kill()
        p2p.close()


async def parent_go_busy() -> None:
    """Checks that the receiver's Go side keeps receiving while its Python is busy for BUSY_S seconds (async)."""
    with tempfile.TemporaryDirectory() as identity_dir:
        p2p = _make_p2p(identity_dir)
        topic = f"{p2p.peer_id}::ps:phase0"
        proc = subprocess.Popen([sys.executable, __file__, "--child", "go-busy", "--topic", topic],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        info = json.loads(proc.stdout.readline())
        peer_id, _ = await p2p.connect(info["addresses"])
        p2p.subscribe_to_topic(topic)
        await asyncio.sleep(3.0)  # Lets the pubsub mesh form
        channel = f"{p2p.peer_id}::dm:{info['peer_id']}{CHANNEL_SUFFIX}"
        print(f"connected to the receiver: {peer_id is not None}; receiver goes busy for {BUSY_S:.0f}s")
        proc.stdin.write("go\n")
        proc.stdin.flush()
        sent_dm = sent_ps = 0
        for _ in range(int(BUSY_S) - 5):
            try:
                await asyncio.to_thread(p2p.send_message_to_peer, channel, b"x" * 1024)
                sent_dm += 1
            except Exception as e:
                print(f"    DM failed: {e}")
            try:
                await asyncio.to_thread(p2p.broadcast_message, topic, b"y" * 1024)
                sent_ps += 1
            except Exception as e:
                print(f"    publish failed: {e}")
            await asyncio.sleep(1.0)
        connected = any(c.get("id") == info["peer_id"] for c in (p2p.get_connected_peers_info() or []))
        result = json.loads(proc.stdout.readline())
        print(f"sent {sent_dm} DMs and {sent_ps} publications while the receiver was busy; "
              f"still connected at the end: {connected}")
        print(f"the receiver popped {result['popped']} messages right after its busy loop "
              f"(the per-channel receive cap may discard some)")
        proc.wait(timeout=10)
        p2p.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["stuck-send", "go-busy"])
    parser.add_argument("--child", action="store_true", help="internal: run as the receiver")
    parser.add_argument("--topic", default="", help="internal: the topic the receiver subscribes to")
    args = parser.parse_args(sys.argv[1:] if sys.argv[1] != "--child" else sys.argv[2:] + ["--child"])
    if args.child:
        child(args.mode, args.topic)
    else:
        asyncio.run(parent_stuck_send() if args.mode == "stuck-send" else parent_go_busy())
