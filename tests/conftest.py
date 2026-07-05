"""Shared test fixtures and fakes for the Session test suite.

Tests in this suite cover the *Python* layer (Session FSM, SessionManager,
connpool routing) without touching the Go shared library — every place that
would call into Go is mocked. The exception is the cross-machine harness in
``tools/session_runner.py``, which boots the real ``Node``.
"""

import sys
import os
import time
import asyncio
from collections import deque
from typing import Optional

import pytest

# Allow running tests from the repo root without installing the package.
_HERE = os.path.dirname(__file__)
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


class _NoopLogger:
    """Swallows every attribute lookup and call. Replaces ``Logger.__instance``
    in tests so the codebase's ``log.error(..., sub=...)`` calls don't fall
    through to ``print`` (which rejects the ``sub`` kwarg)."""
    def __getattr__(self, name):
        return lambda *a, **kw: None


@pytest.fixture(autouse=True)
def _silence_unaiverse_logger():
    from unaiverse.utils.logger import log as _log
    # The instance attribute is name-mangled to ``_Logger__instance``.
    saved = getattr(_log, "_Logger__instance", None)
    setattr(_log, "_Logger__instance", _NoopLogger())
    yield
    setattr(_log, "_Logger__instance", saved)


# ── Fake P2P (mocks the ctypes layer) ───────────────────────────────────────

class FakeP2P:
    """Stand-in for ``unaiverse.networking.p2p.p2p.P2P`` in unit tests.

    Records every send for inspection; signature/ping helpers return canned
    values by default and can be overridden per-test. The high-level async surface
    (``connect`` / ``send_session`` / async crypto) is what the
    Session/SessionManager call on their injected instance.
    """

    def __init__(self, peer_id: str = "12D3FakeSelf", log_sub: str = "fake"):
        self.peer_id = peer_id
        self.log_sub = log_sub
        # Behavior knobs — flip to None / False to simulate failures.
        self.sign_returns: Optional[bytes] = b"sig-bytes"
        self.verify_peer_sig_returns: bool = True
        self.ping_returns: Optional[int] = 1  # ms
        self.send_returns: bool = True
        # Capture.
        self.sent: list[tuple[str, str, bytes]] = []  # (peer_id, protocol_id, payload)
        self.pinged: list[tuple[str, int]] = []      # (peer_id, timeout_ms)

    # ── Synchronous Go-FFI primitives (named like the real P2P) ──────────────
    def _sign_with_host_key_sync(self, data: bytes) -> Optional[bytes]:
        return self.sign_returns

    def _verify_peer_signature_sync(self, peer_id: str, data: bytes, signature: bytes) -> bool:
        return self.verify_peer_sig_returns

    def ping_peer(self, peer_id: str, timeout_ms: int = 5000) -> Optional[int]:
        self.pinged.append((peer_id, timeout_ms))
        return self.ping_returns

    def send_on_protocol(self, peer_id: str, protocol_id: str, payload: bytes) -> bool:
        self.sent.append((peer_id, protocol_id, payload))
        return self.send_returns

    # ── High-level async surface (Session/SessionManager call these) ──────────
    async def sign_with_host_key(self, data: bytes) -> Optional[bytes]:
        return self._sign_with_host_key_sync(data)

    async def verify_peer_signature(self, peer_id: str, data: bytes, signature: bytes) -> bool:
        return self._verify_peer_signature_sync(peer_id, data, signature)


# ── Fake conn layer (mocks the connpool surface Session touches) ────────────

class FakeConn:
    """P2P-per-instance stand-in injected into Session/SessionManager as ``p2p=``.

    Each FakeConn models *one* network of a node (the public one in single-
    instance tests). It exposes the high-level P2P surface the Session calls
    (``connect`` / ``send_session`` / async crypto), with no
    ``p2p_name`` argument, and delegates the crypto to its underlying FakeP2P so
    per-test knobs (set on ``alice_p2p`` etc.) still take effect. A queue-based
    ``send_session`` lets two FakeConns be wired bidirectionally in integration
    tests (see ``bidirectional_fakeconns``).

    Args:
        p2p_public: the underlying FakeP2P. Its ``peer_id`` is "this side's
            identity" — when we forward a send into the other side's inbox we
            substitute it as the sender so the receiver sees
            ``(my_peer_id, protocol, payload)`` like a real ``pop_messages`` entry.
        p2p_world: optional second-network FakeP2P, registered as this side's
            cross-instance ``twin`` (the SignedPeerRecord / grant signer).
    """

    def __init__(self, p2p_public: FakeP2P, p2p_world: Optional[FakeP2P] = None):
        from unaiverse.networking.node.connpool import NodeConn
        self._p2p_self = p2p_public
        self.p2p_name_to_p2p = {
            NodeConn.P2P_PUBLIC: p2p_public,
        }
        if p2p_world is not None:
            self.p2p_name_to_p2p[NodeConn.P2P_WORLD] = p2p_world
        # ``my_peer_id`` is the sender identity used when forwarding into the
        # other side's inbox.
        self.my_peer_id: str = p2p_public.peer_id
        # connect() behaviour — override on the instance.
        self.connect_returns: tuple[Optional[str], bool] = (None, False)
        # Last call recording (addresses only — there is no p2p_name anymore).
        self.connect_calls: list[list[str]] = []
        # send_session behaviour — set send_inbox on the *other* FakeConn to
        # forward writes; otherwise send_session just records.
        self.send_inbox: Optional[deque] = None
        self.sent: list[tuple[str, str, object]] = []
        self.disconnect_calls: list[str] = []

    async def connect(self, addresses):
        self.connect_calls.append(list(addresses))
        return self.connect_returns

    async def disconnect(self, peer_id: str) -> bool:
        # The transport-disconnect primitive SessionManager.drop delegates to.
        self.disconnect_calls.append(peer_id)
        return True

    async def send_session(self, peer_id: str, protocol_id: str, sess_msg) -> bool:
        # Record the outbound call exactly as it was made (peer_id == destination).
        self.sent.append((peer_id, protocol_id, sess_msg))
        if self.send_inbox is not None:
            # The other side's pop_messages reports the SENDER, not the
            # destination — our own per-instance peer-id.
            self.send_inbox.append((self.my_peer_id, protocol_id, sess_msg))
        return True

    # ── Crypto interface (delegates to the underlying FakeP2P) ────
    async def sign_with_host_key(self, data: bytes):
        return await self._p2p_self.sign_with_host_key(data)

    async def verify_peer_signature(self, peer_id: str, data: bytes, signature: bytes) -> bool:
        return await self._p2p_self.verify_peer_signature(peer_id, data, signature)

    # ── Per-network view (multi-instance harness) ────────────────────────────
    def as_p2p(self, p2p_name: str) -> "_FakeConnView":
        """Return the P2P-per-network object to inject into the SessionManager for
        `p2p_name`. Sends, dial outcome and crypto knobs stay shared with this
        FakeConn (so tests keep setting ``conn.send_inbox`` / ``conn.connect_returns``),
        but the on-wire sender is this network's own peer-id and ``twin`` points to
        the other network's FakeP2P."""
        return _FakeConnView(self, p2p_name)


class _FakeConnView:
    """A single-network view over a two-network FakeConn. The Session injected with
    it sends/receives on one instance; the cross-instance ``twin`` reaches the other.
    Dial outcome, the outbound inbox and the send log stay shared with the parent
    FakeConn so multi-instance harness tests drive both networks through one handle."""

    def __init__(self, parent: FakeConn, p2p_name: str):
        self._parent = parent
        self._p2p_name = p2p_name
        self._fake = parent.p2p_name_to_p2p[p2p_name]
        self.my_peer_id = self._fake.peer_id

    async def connect(self, addresses):
        self._parent.connect_calls.append(list(addresses))
        return self._parent.connect_returns

    async def disconnect(self, peer_id: str) -> bool:
        self._parent.disconnect_calls.append(peer_id)
        return True

    async def send_session(self, peer_id: str, protocol_id: str, sess_msg) -> bool:
        self._parent.sent.append((peer_id, protocol_id, sess_msg))
        if self._parent.send_inbox is not None:
            self._parent.send_inbox.append((self.my_peer_id, protocol_id, sess_msg))
        return True

    async def sign_with_host_key(self, data: bytes):
        return await self._fake.sign_with_host_key(data)

    async def verify_peer_signature(self, peer_id: str, data: bytes, signature: bytes) -> bool:
        return await self._fake.verify_peer_signature(peer_id, data, signature)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def alice_peer_id() -> str:
    # Lexicographically smaller than bob_peer_id, so tie-breaker says Alice keeps
    # her initiator (mirrors libp2p's simultaneous-open rule).
    return "12D3KooWaaa1aliceeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"


@pytest.fixture
def bob_peer_id() -> str:
    return "12D3KooWzzz9bobbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


@pytest.fixture
def alice_p2p(alice_peer_id) -> FakeP2P:
    return FakeP2P(peer_id=alice_peer_id, log_sub="alice")


@pytest.fixture
def bob_p2p(bob_peer_id) -> FakeP2P:
    return FakeP2P(peer_id=bob_peer_id, log_sub="bob")


@pytest.fixture
def alice_conn(alice_p2p) -> FakeConn:
    return FakeConn(p2p_public=alice_p2p)


@pytest.fixture
def bob_conn(bob_p2p) -> FakeConn:
    return FakeConn(p2p_public=bob_p2p)


@pytest.fixture
def bidirectional_fakeconns(alice_conn, bob_conn):
    """Wire two FakeConns so writes from one land in the other's inbox.

    Returns (alice_conn, bob_conn, alice_inbox, bob_inbox). Tests pump
    messages by popping from one inbox and calling SessionManager.deliver
    on the other side.
    """
    alice_inbox: deque = deque()
    bob_inbox: deque = deque()
    alice_conn.send_inbox = bob_inbox   # what Alice sends → Bob's inbox
    bob_conn.send_inbox = alice_inbox   # and vice versa
    return alice_conn, bob_conn, alice_inbox, bob_inbox


@pytest.fixture
def profile_factory():
    """Returns a function that produces a deterministic small profile."""
    def _f():
        return {"node_id": "fake-node-id", "version": "0.0-test"}
    return _f


# ── Helpers used across tests ───────────────────────────────────────────────

def _make_session(role, p2p, p2p_name, local_peer_id, addresses=None,
                  is_world_join=False, peer_id="12D3KooWRemote"):
    """Construct a Session with sane defaults for testing."""
    from unaiverse.networking.session import Session, SessionRole
    return Session(
        peer_id=peer_id,
        role=role,
        p2p=p2p,
        p2p_name=p2p_name,
        local_peer_id=local_peer_id,
        local_node_id="local-node-id",
        local_profile_factory=lambda: {"node_id": "local-node-id"},
        addresses=list(addresses or []),
        is_world_join=is_world_join,
    )


@pytest.fixture
def make_session():
    return _make_session


async def apump(inbox: deque, sm) -> list:
    """Drain everything currently in `inbox` into `sm.deliver` from within an
    already-running event loop. Unlike `_pump_call` (one asyncio.run per
    message), this keeps the loop alive across pumps, mirroring how the node's
    main loop drives the SessionManagers. Returns the concatenated list of
    SessionEvents."""
    events = []
    while inbox:
        peer_id, protocol_id, payload = inbox.popleft()
        events.extend(await sm.deliver(peer_id, protocol_id, payload))
    return events


async def establish_pair(sm_initiator, sm_responder, initiator_inbox, responder_inbox,
                         responder_peer_id, addresses=None):
    """Drive the standard 4-message HELLO+PROFILE handshake to ESTABLISHED on
    both sides. The initiator's FakeConn must already have `connect_returns`
    set to a successful dial. Returns (initiator_events, responder_events) as
    emitted on the final two pumps (each should contain one ESTABLISHED)."""
    await sm_initiator.initiate(responder_peer_id, addresses or ["/ip4/127.0.0.1/tcp/0"])
    # Design A: initiate() only launches the dial as a background task. Let it run one loop
    # turn, then tick to reap it so the initiator's HELLO reaches the responder's inbox.
    await asyncio.sleep(0)
    await sm_initiator.tick()
    await apump(responder_inbox, sm_responder)                       # HELLO → HELLO_ACK
    await apump(initiator_inbox, sm_initiator)                       # HELLO_ACK → PROFILE_OFFER
    responder_events = await apump(responder_inbox, sm_responder)    # OFFER → DECISION (+ESTABLISHED)
    initiator_events = await apump(initiator_inbox, sm_initiator)    # DECISION (+ESTABLISHED)
    return initiator_events, responder_events


def pump_once(inbox: deque, sm) -> int:
    """Drain *one* message from inbox into the given SessionManager. Returns
    the number of events emitted (caller can inspect events via SM accessors
    or by capturing the return value of deliver — exposed via wrapper)."""
    if not inbox:
        return 0
    peer_id, protocol_id, payload = inbox.popleft()
    return _pump_call(sm, peer_id, protocol_id, payload)


def _pump_call(sm, peer_id, protocol_id, payload):
    """Synchronously run an async deliver call from sync tests via asyncio."""
    import asyncio
    return asyncio.run(sm.deliver(peer_id, protocol_id, payload))


# ── Real-libp2p (`-m p2p`) fixtures ─────────────────────────────────────────
# These live in the *root* conftest (not a second tests/p2p/conftest.py) so the
# bare ``from conftest import ...`` used by the go-mocked tests keeps resolving to
# this module — two conftest.py files with no package collide on the module name
# under pytest's prepend import mode. The pure helpers (build_real_node,
# pump_until, …) live in tests/realnet_helpers.py. Fixtures defined here are
# inherited by tests/p2p/ tests; they are lazy, so the fast suite never starts a
# real P2P node.

@pytest.fixture(scope="session")
def _libp2p_setup():
    """Initialize the Go shared library exactly once for the whole session."""
    from unaiverse.networking.p2p.p2p import P2P
    if not getattr(P2P, "_library_initialized", False):
        P2P.setup_library()
    return P2P


@pytest.fixture
def real_nodes(_libp2p_setup, tmp_path):
    """Factory for real libp2p nodes with unique temp identities; closes every
    P2P on teardown to free instance slots (32-slot global cap). ALL real-node
    creation must go through this. See tests/realnet_helpers.py."""
    from realnet_helpers import build_real_node, build_real_world_node
    created = []

    class _Factory:
        def node(self, name: str, port: int = 0):
            # Same `name` → same identity_dir → same peer-id: rebuilding a node
            # with a fixed port simulates a peer restart (fresh SessionManager,
            # same network identity). Port 0 lets the OS pick.
            p2p, conn, sm = build_real_node(str(tmp_path / f"id-{name}"), port=port, node_id=name)
            created.append(p2p)
            return p2p, conn, sm

        def world_node(self, name: str, decision_hook=None):
            pp, pw, conn, smp, smw = build_real_world_node(
                str(tmp_path / f"id-{name}-pub"), str(tmp_path / f"id-{name}-world"),
                decision_hook=decision_hook, node_id=name)
            created.extend([pp, pw])
            return pp, pw, conn, smp, smw

    yield _Factory()

    for p2p in created:
        try:
            p2p.close()
        except Exception:
            pass
