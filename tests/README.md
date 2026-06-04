# Session tests

Three layers, fastest first:

  * **Go-mocked unit + in-process integration** (`test_session_fsm.py`,
    `test_session_manager.py`, `test_connpool_routing.py`, `test_handshake_e2e.py`,
    `test_world_join_e2e.py`) — pure Python. The Go shared library is stubbed
    (`FakeP2P`/`FakeConn` in `conftest.py`); runnable on any machine in ~2 s. This
    is the default suite and the per-commit gate.

  * **Real-libp2p integration** (`tests/p2p/`, marked `@pytest.mark.p2p`) — boots
    *real* `P2P` instances over the built `unailib.*.so` and talks over real
    loopback/LAN TCP, in one process. Covers handshake, intentional close, and the
    full world-join handoff (with real cross-instance crypto). **Opt-in** — not run
    by default (needs the `.so` and a usable NIC). Helpers live in
    `realnet_helpers.py`.

  * **`tools/session_runner.py`** — manual cross-machine / liveness harness. Boots
    the real stack and drives a Session against a remote peer. Use it for what the
    automated suites *can't* do: two physical machines, real network disruption,
    and live observation of `RECONNECTING`/`RESUMED`/`LOST`. It does **not** check
    handshake/world-join correctness — the `-m p2p` suite owns that now.

All commands below run from the repository root, inside the project's Python
environment.

## Running the default (go-mocked) suite

```bash
pytest -q
```

48 tests, ~2 s, `p2p` tests excluded automatically (via `addopts = -m 'not p2p'`
in `pyproject.toml`). Covers:

| Area                              | File                          |
|-----------------------------------|-------------------------------|
| Session FSM (every transition)    | `test_session_fsm.py`         |
| SessionManager (tie-breaker, etc) | `test_session_manager.py`     |
| connpool routing + SessMsg codec  | `test_connpool_routing.py`    |
| Full in-process handshake         | `test_handshake_e2e.py`       |
| World-join handoff (mocked)        | `test_world_join_e2e.py`      |

`test_full_handshake_happy_path` is the keystone: it wires two `SessionManager`s
through a fake bidirectional transport and drives the full Stage-1/Stage-2
sequence with real protobufs. A regression in the FSM, the protocol routing, or
the message schemas breaks it.

Note these are go-mocked: crypto/ping/SPR are canned (`FakeP2P` returns fixed
values), so they verify *protocol logic*, not real Ed25519 signing or the wire.
That's what the `-m p2p` suite is for.

## Running the real-libp2p suite (`-m p2p`)

```bash
pytest -m p2p -v
```

3 tests, ~5 s (the CLI `-m p2p` overrides the default exclusion). Requires a
built `unailib.*.so` and a non-loopback (LAN) address — see the gotcha below.

| Test | What it proves over the real wire |
|---|---|
| `test_real_handshake_happy_path` | both sides ESTABLISHED, matching `session_id` |
| `test_real_intentional_close_reaches_peer` | peer sees `CLOSED` (with reason), not `LOST` |
| `test_real_world_join_full_handoff` | 4 instances; WorldGrant signed/verified with the **world** key cross-instance; public session closed via `WORLD_HANDOFF_COMPLETE` |

These build real nodes via the `real_nodes` fixture (`conftest.py`), which closes
every `P2P` on teardown to free instance slots (32-slot global cap). If the host
has no usable LAN address the tests `skip` rather than fail.

> **`127.0.0.1` gotcha.** `connpool.__connect` filters out every `127.0.0.1`
> multiaddr, so a loopback dial reduces to an empty address list. Real nodes
> therefore listen on `0.0.0.0` and dial a **LAN** address — `pick_dialable_addr`
> in `realnet_helpers.py` selects one. This is intended production behaviour, not
> a bug.

## Rebuilding the shared library

The `-m p2p` suite and the runner need a current `unailib.*.so`. If the Go
sources under `p2p/unailib/` change, reinstall the package to rebuild it:

```bash
# From the repo root:
pip install -e .   # setup.py recompiles the Go extension when the source hash changes
```

## Running the cross-machine harness

### Single machine, two terminals

Terminal 1 (listener):

```bash
python tests/tools/session_runner.py listen \
    --identity /tmp/runner-bob --port 30060
```

It prints its peer-id and multiaddrs, e.g.:

```
=== LISTENER READY ===
peer_id:    12D3KooW...bobpeerid
multiaddrs:
    /ip4/127.0.0.1/tcp/30060/p2p/12D3KooW...bobpeerid
    /ip4/192.168.1.42/tcp/30060/p2p/12D3KooW...bobpeerid
```

Terminal 2 (dialer) — **use the LAN multiaddr, not `127.0.0.1`** (see the gotcha
above; a loopback address gets filtered and the dial fails with "invalid
addresses"):

```bash
python tests/tools/session_runner.py dial \
    --identity /tmp/runner-alice --port 30070 \
    --peer-addr "/ip4/192.168.1.42/tcp/30060/p2p/12D3KooW...bobpeerid"
```

Both processes log every `SessionEvent`. Within ~2 s of the dial each side logs:

```
[SessionEvent] established peer=12D3KooW... session=… role=initiator data={'profile_b': {...}}
```

After that, ping liveness fires every 15 s and both processes idle. Tip:
`LOG_LEVEL=WARNING` quiets the Go/`LIB-TYPES` chatter; `LOG_LEVEL=INFO` shows the
`[SessionEvent]` lines.

### Two machines on the same LAN

Same as above, run the listener on one machine and dial its LAN multiaddr from
the other. Open the listener's TCP port in its firewall.

### Across NATs / over the Internet

The harness disables relay, DHT, and WebRTC to stay minimal, so it has **no NAT
traversal** — you need direct reachability. Either:

1. Run the listener on a publicly-reachable host (bare TCP on a VPS with the port
   forwarded), or
2. Enable the relay client in `_build_node` (`session_runner.py`) — flip
   `enable_relay_client=True` in the `P2P(...)` constructor call and the listener
   will reserve on the default public relays, advertising a `/p2p-circuit` address.

## Liveness scenarios (cross-machine harness only)

| Scenario | Expected events on the dialer |
|---|---|
| Healthy connect | `established` once, then nothing (ping is silent on success) |
| Kill the listener process | `reconnecting` (transport_disconnected), then `lost` ~10 s later |
| Link disrupted, peer returns within 10 s | `reconnecting` then `resumed` |
| Listener sends GOAWAY (clean shutdown) | `closed` (with reason) |
| Dial unreachable peer | 4 dial attempts with backoff, then `failed` (`dial_failed`) |

### Verifying keep-alive

Block the peer's UDP port mid-session (Linux):

```bash
sudo iptables -A INPUT -p udp --sport 30060 -j DROP
```

The dialer emits `reconnecting` within ~35 s (two missed 15 s pings + a 5 s ping
timeout each), then `lost` 10 s later (grace period). Remove the rule and the next
dial works normally.

## Notes

  * The go-mocked suite stubs libp2p entirely; the `-m p2p` suite and the runner
    use the real `unailib`.
  * Session-protocol flows need no JWT — identity is bound by `sig_a`/`sig_b` +
    signed peer records inside Stage-1/2. (JWT only matters for the legacy chat
    path.)
  * The runner uses a single libp2p instance for both the `p2p_public` and
    `p2p_world` pools, so it does **not** exercise the dual-instance world handoff
    — `tests/p2p/test_world_join_realnet.py` covers that (four real instances).
```
