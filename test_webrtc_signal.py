"""
test_webrtc_signal.py
─────────────────────
Smoke test for the WebRTC signaling protocol.

Two P2P nodes are created on localhost (no relay, no NAT).
Node B connects directly to Node A, then initiates WebRTC signaling.
Once the DataChannel is open we send a message from B → A via the normal
send_message_to_peer() path (which now routes through the DataChannel) and
verify it arrives in A's pop_messages() queue.

Run from the repo root:
    python test_webrtc_signal.py
"""

import base64
import sys
import time
import tempfile
import os

# ── locate the package ────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "src")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from unaiverse.networking.p2p import P2P, P2PError

# ── helpers ───────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET}  {msg}")
def fail(msg): print(f"  {RED}✗{RESET}  {msg}"); sys.exit(1)
def info(msg): print(f"  {YELLOW}…{RESET}  {msg}")

# ── test ──────────────────────────────────────────────────────────────────────

def main():
    print("\n══════════════════════════════════════════════")
    print("  WebRTC Signaling Smoke Test")
    print("══════════════════════════════════════════════\n")

    # ── library setup ─────────────────────────────────────────────────────────
    info("Setting up P2P library…")
    P2P.setup_library(max_instances=4, enable_logging=False)
    ok("Library initialised.")

    with tempfile.TemporaryDirectory() as dir_a, \
         tempfile.TemporaryDirectory() as dir_b:

        # ── create two nodes ──────────────────────────────────────────────────
        info("Creating Node A (answerer)…")
        node_a = P2P(
            identity_dir=dir_a,
            port=0,
            knows_is_public=True,   # skip AutoNAT wait on localhost
            webrtc_enabled=True,
        )
        ok(f"Node A  peer_id = {node_a.peer_id}")
        ok(f"Node A  addrs   = {node_a.addresses}")

        info("Creating Node B (offerer)…")
        node_b = P2P(
            identity_dir=dir_b,
            port=0,
            knows_is_public=True,
            webrtc_enabled=True,
        )
        ok(f"Node B  peer_id = {node_b.peer_id}")

        # ── Step 1 : direct libp2p connection B → A ───────────────────────────
        print()
        info("Step 1 – Connect B → A over localhost TCP…")
        addrs_a = node_a.addresses
        if not addrs_a:
            fail("Node A returned no addresses - cannot connect.")
        node_b.connect_to(addrs_a)
        time.sleep(0.5)   # let the connection settle
        ok("B is connected to A.")

        # ── Step 2 : initiate WebRTC signaling ────────────────────────────────
        print()
        info("Step 2 – Initiate WebRTC signaling from B → A…")
        info("(batch ICE: gathering all candidates before exchanging SDP)")
        try:
            result = node_b.initiate_webrtc_connection(node_a.peer_id)
            ok(f"initiate_webrtc_connection returned: {result}")
        except P2PError as e:
            fail(f"initiate_webrtc_connection failed: {e}")

        # give the answerer a moment to fire OnDataChannel / OnOpen
        time.sleep(0.3)

        # ── Step 3 : verify both sides see the DataChannel ────────────────────
        print()
        info("Step 3 – Verify DataChannel visibility…")

        conns_b = node_b.get_webrtc_connections()
        conns_a = node_a.get_webrtc_connections()

        print(f"       B sees: {conns_b}")
        print(f"       A sees: {conns_a}")

        b_open = any(c.get("peer_id") == node_a.peer_id and c.get("state") == "open"
                     for c in conns_b)
        a_open = any(c.get("peer_id") == node_b.peer_id and c.get("state") == "open"
                     for c in conns_a)

        if b_open:
            ok("Node B has an open DataChannel to A.")
        else:
            fail(f"Node B does NOT report an open DataChannel to A.  conns={conns_b}")

        if a_open:
            ok("Node A has an open DataChannel to B.")
        else:
            # Non-fatal: the answerer registers the DC inside the goroutine;
            # it should be there but we give it a second chance.
            info("Node A doesn't see the DC yet – waiting 1 s…")
            time.sleep(1.0)
            conns_a = node_a.get_webrtc_connections()
            a_open = any(c.get("peer_id") == node_b.peer_id and c.get("state") == "open"
                         for c in conns_a)
            if a_open:
                ok("Node A now has an open DataChannel to B.")
            else:
                fail(f"Node A still does NOT report an open DataChannel.  conns={conns_a}")

        # ── Step 4 : send a message B → A via the DataChannel ─────────────────
        print()
        info("Step 4 – Send message B → A (should route via WebRTC DataChannel)…")
        test_payload = b"hello_webrtc_datachannel"
        # Channel format: <anything>::dm:<target_peer_id>-<tag>
        channel = f"test::dm:{node_a.peer_id}-smoke"
        node_b.send_message_to_peer(channel, test_payload)
        ok(f"send_message_to_peer({channel!r}, {test_payload!r}) completed.")

        time.sleep(0.3)   # let the message propagate

        # ── Step 5 : receive the message on A ─────────────────────────────────
        print()
        info("Step 5 – Pop messages on Node A…")
        messages = node_a.pop_messages()
        print(f"       raw pop_messages() result: {messages}")

        if not messages:
            fail("No messages arrived at Node A.")

        # Each entry is {"from": peer_id, "data": base64_bytes}
        received_payloads = []
        for m in messages:
            raw = base64.b64decode(m["data"])
            received_payloads.append(raw)
            print(f"         from={m['from']}  data={raw!r}")

        if test_payload in received_payloads:
            ok(f"Payload {test_payload!r} received correctly at Node A  🎉")
        else:
            fail(f"Expected {test_payload!r} but got {received_payloads}")

        # ── Step 6 : close WebRTC, verify fallback still works ─────────────────
        print()
        info("Step 6 – Close WebRTC connection, verify clean shutdown…")
        node_b.close_webrtc_connection(node_a.peer_id)
        time.sleep(0.2)
        conns_b_after = node_b.get_webrtc_connections()
        if not any(c.get("peer_id") == node_a.peer_id for c in conns_b_after):
            ok("DataChannel removed from B's map after explicit close.")
        else:
            fail(f"DataChannel still in B's map after close: {conns_b_after}")

        # ── teardown ──────────────────────────────────────────────────────────
        print()
        info("Closing nodes…")
        node_a.close()
        node_b.close()
        ok("Both nodes closed.")

    print()
    print(f"{GREEN}══════════════════════════════════════════════{RESET}")
    print(f"{GREEN}  ALL CHECKS PASSED{RESET}")
    print(f"{GREEN}══════════════════════════════════════════════{RESET}\n")


if __name__ == "__main__":
    main()
