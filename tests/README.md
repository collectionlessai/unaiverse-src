# Session tests

Two pieces:

  * **`test_session_fsm.py`, `test_session_manager.py`, `test_connpool_routing.py`,
    `test_handshake_e2e.py`** — pure-Python unit + in-process integration tests.
    Stub the Go shared library; runnable on any machine in ~2 seconds.

  * **`tools/session_runner.py`** — cross-machine smoke harness. Boots the real
    libp2p stack and lets you drive the Session FSM against a remote peer over
    your actual network. Used to verify behaviour on different OSes, behind
    different NATs, with different transports.

## Running the unit suite

```bash
# From the repo root:
PYTHONPATH=src pytest tests/ -v
```

45 tests, ~2 s. Covers:

| Area                              | File                          |
|-----------------------------------|-------------------------------|
| Session FSM (every transition)    | `test_session_fsm.py`         |
| SessionManager (tie-breaker, etc) | `test_session_manager.py`     |
| connpool routing                  | `test_connpool_routing.py`    |
| Full in-process handshake         | `test_handshake_e2e.py`       |

The in-process handshake test (`test_full_handshake_happy_path`) is the most
useful one — it wires two `SessionManager`s through a fake bidirectional
transport and drives the full Stage-1/Stage-2 sequence with real protobufs.
A regression in the FSM, the protocol routing, or the message schemas will
break it.

## Running the cross-machine harness

The harness needs the built `unailib` shared library — make sure the package
is installed (`pip install -e .` from the repo root) or that `unailib*.so` is
present in `src/unaiverse/networking/p2p/`.

### Loopback (single machine, two terminals)

Terminal 1 (listener):

```bash
PYTHONPATH=src python tests/tools/session_runner.py listen \
    --identity /tmp/runner-bob --port 30060
```

This prints something like:

```
=== LISTENER READY ===
peer_id:    12D3KooW...bobpeerid
multiaddrs:
    /ip4/127.0.0.1/tcp/30060/p2p/12D3KooW...bobpeerid
    /ip4/192.168.1.42/tcp/30060/p2p/12D3KooW...bobpeerid
```

Terminal 2 (dialer):

```bash
PYTHONPATH=src python tests/tools/session_runner.py dial \
    --identity /tmp/runner-alice --port 30070 \
    --peer-addr "/ip4/127.0.0.1/tcp/30060/p2p/12D3KooW...bobpeerid"
```

Both processes log every `SessionEvent` they receive. Expected sequence on
each side (within ~2 s of the dial):

```
[SessionEvent] established peer=12D3KooW... session=… role=initiator data={'profile_b': {...}}
```

After that, ping liveness fires every 15 s. Both processes idle indefinitely.

### Two machines on the same LAN

Same as loopback but use the LAN multiaddr from the listener's output
(`/ip4/192.168.x.y/...`) on the dialer.

### Across NATs / over the Internet

The harness disables relay and DCUtR by default to keep it minimal. For NAT
traversal you'll need to either:

1. Run the listener on a publicly-reachable host (bare TCP works on a VPS
   with the port forwarded), or
2. Modify `_build_node` in `session_runner.py` to enable the relay client and
   advertise the resulting `/p2p-circuit` address.

(Option 2 is roughly four lines of config in the `cfg` dict — flip
`relay.enable_client = True` and the listener will reserve on the default
public relays.)

## Things to watch for in the cross-machine harness

| Scenario | Expected events on the dialer |
|---|---|
| Healthy connect | `established` once, then nothing (ping is silent on success) |
| Kill the listener process | `reconnecting` (cause=transport_disconnected), then `lost` ~10 s later |
| Listener disrupts the link, comes back within 10 s | `reconnecting` then `resumed` |
| Listener sends GOAWAY (Ctrl-C with clean shutdown if you wire it) | `closed` (with reason) |
| Dial unreachable peer | 4 dial attempts with backoff, then `failed` (reason=`dial_failed`) |

## Verifying the Session keep-alive works

Block the peer's UDP port mid-session (Linux example):

```bash
sudo iptables -A INPUT -p udp --sport 30060 -j DROP
```

The dialer should emit `reconnecting` within ~35 s (two missed pings at 15 s
intervals + the 5 s ping timeout each), then `lost` 10 s after that (grace
period). Remove the iptables rule and the next dial works normally.

## Notes

  * The unit suite stubs the libp2p shared library entirely. The cross-machine
    harness uses the real `unailib`.
  * Both sides need consistent JWT-public-key configuration if you enable the
    legacy chat path. For Session-protocol-only flows (which the harness
    exercises), no JWT is required — Session identity is bound by `sig_a` /
    `sig_b` + signed peer records inside Stage-1/2.
  * The harness keeps a single libp2p instance and uses it for both
    `p2p_public` and `p2p_world` pools — that's fine for Session smoke
    testing but doesn't exercise the dual-instance world-handoff path.
