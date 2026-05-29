"""Shared test fixtures and fakes for the Session test suite.

Tests in this suite cover the *Python* layer (Session FSM, SessionManager,
connpool routing) without touching the Go shared library — every place that
would call into Go is mocked. The exception is the cross-machine harness in
``tools/session_runner.py``, which boots the real ``Node``.
"""

import sys
import os
import time
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

    Records every send for inspection; signatures/ping/SPR helpers return
    canned values by default and can be overridden per-test.
    """

    def __init__(self, peer_id: str = "12D3FakeSelf", log_sub: str = "fake"):
        self.peer_id = peer_id
        self.log_sub = log_sub
        # Behavior knobs — flip to None / False to simulate failures.
        self.sign_returns: Optional[bytes] = b"sig-bytes"
        self.verify_peer_sig_returns: bool = True
        self.verify_spr_returns: bool = True
        self.spr_bytes: Optional[bytes] = b"spr-bytes"
        self.ping_returns: Optional[int] = 1  # ms
        self.send_returns: bool = True
        # Capture.
        self.sent: list[tuple[str, str, bytes]] = []  # (peer_id, protocol_id, payload)
        self.pinged: list[tuple[str, int]] = []      # (peer_id, timeout_ms)

    # The methods Session / SessionManager actually call:
    def sign_with_host_key(self, data: bytes) -> Optional[bytes]:
        return self.sign_returns

    def verify_peer_signature(self, peer_id: str, data: bytes, signature: bytes) -> bool:
        return self.verify_peer_sig_returns

    def generate_signed_peer_record(self) -> Optional[bytes]:
        return self.spr_bytes

    def verify_signed_peer_record(self, envelope_bytes: bytes, expected_peer_id: str) -> bool:
        return self.verify_spr_returns

    def ping_peer(self, peer_id: str, timeout_ms: int = 5000) -> Optional[int]:
        self.pinged.append((peer_id, timeout_ms))
        return self.ping_returns

    def send_on_protocol(self, peer_id: str, protocol_id: str, payload: bytes) -> bool:
        self.sent.append((peer_id, protocol_id, payload))
        return self.send_returns


# ── Fake ConnectionPools (mocks the connpool surface Session touches) ───────

class FakeConn:
    """Stand-in for ``ConnectionPools``.

    Holds one or two FakeP2P instances under ``p2p_name_to_p2p`` and a
    queue-based ``send_session`` so two FakeConns can be wired
    bidirectionally in integration tests (see ``bidirectional_fakeconns``).

    Args:
        p2p_public: the FakeP2P for ``P2P_PUBLIC``. Its ``peer_id`` doubles as
            "this side's identity" — when we forward a send into the other
            side's inbox we substitute *this* peer_id as the sender so the
            receiving side sees ``(my_peer_id, protocol, payload)`` exactly
            like a real ``pop_messages`` entry would.
    """

    def __init__(self, p2p_public: FakeP2P, p2p_world: Optional[FakeP2P] = None):
        from unaiverse.networking.node.connpool import NodeConn
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
        # Last call recording.
        self.connect_calls: list[tuple[list[str], str]] = []
        # send_session behaviour — set send_inbox on the *other* FakeConn to
        # forward writes; otherwise send_session just records.
        self.send_inbox: Optional[deque] = None
        self.sent: list[tuple[str, str, object]] = []

    def _p2p(self, p2p_name: str) -> FakeP2P:
        # Single-instance test conns only hold P2P_PUBLIC; the Session fetches the
        # SPR for the *other* instance (p2p_world) lazily — fall back to public so
        # unit tests don't need to wire a second FakeP2P.
        from unaiverse.networking.node.connpool import NodeConn
        return self.p2p_name_to_p2p.get(p2p_name, self.p2p_name_to_p2p[NodeConn.P2P_PUBLIC])

    async def connect(self, addresses, p2p_name: str):
        self.connect_calls.append((list(addresses), p2p_name))
        return self.connect_returns

    async def send_session(self, peer_id: str, protocol_id: str, sess_msg, p2p_name: str) -> bool:
        # Record the outbound call exactly as it was made (peer_id == destination).
        self.sent.append((peer_id, protocol_id, sess_msg))
        if self.send_inbox is not None:
            # The other side's pop_messages reports the SENDER, not the destination.
            self.send_inbox.append((self.my_peer_id, protocol_id, sess_msg))
        return True

    # ── Async crypto / ping interface (mirrors ConnectionPools) ──────────────
    async def sign_with_host_key(self, p2p_name: str, data: bytes):
        return self._p2p(p2p_name).sign_with_host_key(data)

    async def verify_peer_signature(self, p2p_name: str, peer_id: str, data: bytes, signature: bytes) -> bool:
        return self._p2p(p2p_name).verify_peer_signature(peer_id, data, signature)

    async def verify_signed_peer_record(self, p2p_name: str, envelope: bytes, expected_peer_id: str) -> bool:
        return self._p2p(p2p_name).verify_signed_peer_record(envelope, expected_peer_id)

    async def ping_peer(self, p2p_name: str, peer_id: str, timeout_ms: int = 5000):
        return self._p2p(p2p_name).ping_peer(peer_id, timeout_ms)

    async def generate_signed_peer_record(self, p2p_name: str):
        return self._p2p(p2p_name).generate_signed_peer_record()


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

def _make_session(role, conn, p2p_name, local_peer_id, addresses=None,
                  remote_other_peer_id_hint=None, is_world_join=False,
                  peer_id="12D3KooWRemote"):
    """Construct a Session with sane defaults for testing."""
    from unaiverse.networking.session import Session, SessionRole
    return Session(
        peer_id=peer_id,
        role=role,
        conn=conn,
        p2p_name=p2p_name,
        local_peer_id=local_peer_id,
        local_node_id="local-node-id",
        local_profile_factory=lambda: {"node_id": "local-node-id"},
        remote_other_peer_id_hint=remote_other_peer_id_hint,
        addresses=list(addresses or []),
        is_world_join=is_world_join,
    )


@pytest.fixture
def make_session():
    return _make_session


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
    return asyncio.get_event_loop().run_until_complete(
        sm.deliver(peer_id, protocol_id, payload)
    )
