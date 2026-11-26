# Tests

Three layers, fastest first:

  * **Default suite** (every `test_*.py` in this folder, plus `uai/`). Pure Python:
    whatever would call into the Go shared library is replaced by fakes (`FakeP2P` and
    `FakeConn` in `conftest.py`, the Node shells of `node_harness.py`, the in-process
    agents of `inproc_harness.py`), so it runs on any machine. This is the per-commit
    gate.

  * **Real-libp2p integration** (`tests/p2p/`, marked `@pytest.mark.p2p`). Boots
    *real* `P2P` instances over the built `unailib.*.so` and talks over real
    loopback/LAN TCP, in one process. **Opt-in**: not run by default (it needs the
    `.so` and a usable NIC). Helpers live in `realnet_helpers.py`.

  * **`tools/session_runner.py`**, the manual cross-machine / liveness harness. Boots
    the real stack and drives a Session against a remote peer. Use it for what the
    automated suites *can't* do: two physical machines, real network disruption,
    and live observation of `RECONNECTING`/`RESUMED`/`LOST`. It does **not** check
    handshake/world-join correctness, the `-m p2p` suite owns that.

All commands below run from the repository root, inside the project's Python
environment.

## Running the default suite

```bash
pytest -q
```

About 950 tests in about 15 s. The `p2p` tests are excluded automatically (via
`addopts = -m 'not p2p'` in `pyproject.toml`). A few tests are skipped or expected
to fail on purpose (parked gates, vector operations not implemented on this side):
`pytest -q -rsx` lists them with their reason.

The modules are marked `unit` or `e2e` (`pytest -m unit`, `pytest -m e2e`), except
`test_node_root_robustness.py` and `uai/`, which carry no marker. By area:

| Area | Files |
|---|---|
| Session FSM and SessionManager | `test_session_fsm.py`, `test_session_manager.py` |
| Handshake, reconnection and world join between SessionManagers | `test_handshake_e2e.py`, `test_session_reconnect_e2e.py`, `test_world_join_e2e.py` |
| Coordinator (`NodeConn`): routing, perimeter, teardown, codecs, tokens | `test_connpool_routing.py`, `test_connpool_teardown.py`, `test_messages.py`, `test_tokens.py` |
| Node over real SessionManagers and a fake conn (`node_harness.py`) | `test_node_integration.py`, `test_join_code_gate.py`, `test_node_lifecycle.py`, `test_node_root_robustness.py` |
| Node over the real `NodeConn` and fake transports (`node_harness.py`) | `test_world_roster_e2e.py`, `test_profile_connections_e2e.py`, `test_inspector_presence_e2e.py` |
| Real agents in process (`inproc_harness.py`): chat, pubsub, engage, world purge | `test_agent_chat_e2e.py`, `test_pubsub_e2e.py`, `test_engage_lifecycle_e2e.py`, `test_human_world_routing_e2e.py`, `test_inproc_harness_smoke.py` |
| Agent surface: actions, resolution, interactions, human routing, disengage, stepwise compute | `test_agent_actions_pure.py`, `test_agent_resolution.py`, `test_interaction.py`, `test_interaction_manager.py`, `test_human_routing.py`, `test_disengage_goaway.py`, `test_stepwise_compute.py` |
| World: roles, definition hash, shipped-code closure | `test_world_roles.py`, `test_world_definition.py`, `test_import_closure.py` |
| HSM, streams, clock, profile, modules, logger | `test_hsm.py`, `test_streams_dataprops.py`, `test_streams_io.py`, `test_clock.py`, `test_profile.py`, `test_modules.py`, `test_logger_sinks.py` |
| Interactive-messages protocol (`unaiverse.uai`), with the golden vectors shared with unaiverse-js | `uai/` |

`test_full_handshake_happy_path` (in `test_handshake_e2e.py`) is the keystone of the
Session layer: it wires two `SessionManager`s through a fake bidirectional transport
and drives the full Stage-1/Stage-2 sequence with real protobufs. A regression in the
FSM, the protocol routing, or the message schemas breaks it.

Note that nothing here touches the wire: crypto/ping/SPR are canned (`FakeP2P` returns
fixed values), so the suite verifies *protocol and application logic*, not real
Ed25519 signing or the transport. That's what the `-m p2p` suite is for.

## Running the real-libp2p suite (`-m p2p`)

```bash
pytest -m p2p -v
```

18 tests in 10 files (the CLI `-m p2p` overrides the default exclusion). Requires a
built `unailib.*.so` and a non-loopback (LAN) address, see the gotcha below.

| File (`tests/p2p/`) | What it covers over the real wire |
|---|---|
| `test_session_realnet.py` | handshake to ESTABLISHED on both sides, intentional close seen as `CLOSED` and not `LOST` |
| `test_world_join_realnet.py` | the full world-join handoff across four real instances |
| `test_reconnect_realnet.py` | transport drop then re-dial and `RESUME`, peer restart and re-handshake, a reborn peer superseding the survivor's Session |
| `test_keepalive_idle_realnet.py` | an idle Session stays ESTABLISHED across keep-alive cycles, on a public link and on the world link after a handoff |
| `test_disengage_goaway_realnet.py` | a graceful GOAWAY drives the remote to `CLOSED`, a bare transport drop to `RECONNECTING` |
| `test_perimeter_security_realnet.py` | forged rendezvous roster rejected, pubsub-only intruder cut, revoked member cut and unresumable |
| `test_pool_promotion_realnet.py` | Session-only peers enter the pools on both sides and the pool diffs fire |
| `test_interaction_over_session_realnet.py` | an application `Msg` crosses the membership and token gates of `get_messages` |
| `test_human_world_routing_realnet.py` | the world's purge primitive: GOAWAY(`LOCAL_ERROR`) to a sender of application content |
| `test_stepwise_compute_realnet.py` | a handshake completes during a heavy stepwise computation |

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

The harness disables the relay service and WebRTC, but keeps the relay client and
the default reachability probing on (`_build_node` in `session_runner.py`): a
private listener gets its AutoNAT verdict from the infrastructure relays and then
reserves on them automatically, advertising a `/p2p-circuit` address. For a
direct (non-relayed) path you still need direct reachability: run the listener on
a publicly-reachable host (bare TCP on a VPS with the port forwarded).

## Liveness scenarios (cross-machine harness only)

| Scenario | Expected events on the dialer |
|---|---|
| Healthy connect | `established` once, then nothing (ping is silent on success) |
| Kill the listener process | `reconnecting` (transport_disconnected), then `lost` when the 90 s reconnect window ends |
| Link disrupted, peer returns within the reconnect window | `reconnecting` then `resumed` |
| Listener sends GOAWAY (clean shutdown) | `closed` (with reason) |
| Dial unreachable peer | 4 dial attempts with backoff, then `failed` (`dial_failed`) |

### Verifying keep-alive

Block the peer's UDP port mid-session (Linux):

```bash
sudo iptables -A INPUT -p udp --sport 30060 -j DROP
```

The dialer emits `reconnecting` within ~35 s (two missed PINGs, 15 s apart, each
waited for 10 s), then `lost` when the 90 s reconnect window ends. Remove the rule
before that and the Session resumes.

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
