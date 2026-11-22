#!/usr/bin/env python3
"""
The other process of test_send_queue_realnet.py (a peer that the test can stop with SIGSTOP or kill).

    python send_queue_peer.py receiver            # prints {"peer_id", "addresses"}, then {"got": n, "t": time} each
    python send_queue_peer.py legacy-sender ADDR  # with the send queues off, sends one message, prints {"seq": ...}

A receiver pops all the time, so its per-channel receive cap (50) is never what drops a message: the order and the
completeness it reports are those of the sender. A message carries its index in its first 8 bytes (big endian).
"""
import sys
import json
import time
import asyncio
import tempfile


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


def receiver() -> None:
    """Prints the address of a new instance, then the index of every message it receives, forever."""
    with tempfile.TemporaryDirectory() as identity_dir:
        p2p = _make_p2p(identity_dir)
        addrs = [a for a in p2p.addresses if "127.0.0.1" not in a and "/p2p/" in a]
        addrs.sort(key=lambda a: 0 if ("/tcp/" in a and "/ws" not in a) else 1)
        print(json.dumps({"peer_id": p2p.peer_id, "addresses": addrs}), flush=True)
        while True:
            batch = p2p.pop_messages()
            for m in batch:
                data = m["data"] if isinstance(m, dict) else m
                print(json.dumps({"got": int.from_bytes(bytes(data)[:8], "big"), "t": time.time(), "len": len(data)}),
                      flush=True)
            if not batch:
                time.sleep(0.002)


def legacy_sender(addr: str) -> None:
    """Sends one message with the send queues off (limits that Go refuses), and prints what send returned.

    Args:
        addr: The address of the receiver.
    """
    from unaiverse.custom import Custom
    Custom.SEND_QUEUE_MAX_MSGS_PER_PEER = 0  # Not valid: Go keeps the queues off, as for a caller that never asks
    with tempfile.TemporaryDirectory() as identity_dir:
        p2p = _make_p2p(identity_dir)
        peer_id, _ = asyncio.run(p2p.connect([addr]))
        seq = p2p.send_message_to_peer(f"{p2p.peer_id}::dm:{peer_id}-misc", (0).to_bytes(8, "big") + b"legacy")
        from unaiverse.networking.p2p.p2p import P2P
        print(json.dumps({"seq": seq, "queues": P2P._send_queues_enabled}), flush=True)
        time.sleep(1.0)
        p2p.close()


if __name__ == "__main__":
    if sys.argv[1] == "receiver":
        receiver()
    else:
        legacy_sender(sys.argv[2])
