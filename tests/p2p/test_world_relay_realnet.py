"""Real-libp2p: a member reserves a slot on its world host and is reached through it
(opt-in: ``-m p2p``).

The world instance of a world node offers the relay service, and a member that is not
publicly reachable asks it for a reservation right after the join (``Node.__join_world``).
The reservation is what puts a ``/p2p-circuit`` address through the world host among the
addresses of the member, the one another member dials when no direct path exists.

The three instances here are forced public: a private one would start AutoRelay toward
the infrastructure relays, which a test must not touch. The reservation is asked
explicitly, as the node does, so the path under test is the same.

Run with:  ``uv run pytest -m p2p -k world_relay -v``
"""

import time
from datetime import datetime, timedelta, timezone

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from unaiverse.networking.p2p.p2p import P2PError
from unaiverse.networking.session.sess_msg import SESSION_CONTROL_PROTOCOL

from realnet_helpers import pick_dialable_addr

pytestmark = pytest.mark.p2p


WORLD_DOMAIN = "world.test"


def _instance(P2P, identity_dir, relay_service: bool, **tls):
    return P2P(identity_dir=str(identity_dir), port=0, ips=["0.0.0.0"],
               enable_relay_client=True, enable_relay_service=relay_service,
               use_broad_limits=relay_service, is_isolated=False, knows_is_public=True,
               enable_tls=False, probe_peers=[], webrtc_enabled=False, flood_sub=True,
               log_sub="prv", **tls)


def _self_signed(folder, domain: str) -> dict:
    """A certificate of its own for the WebSocket listener, as a world with a domain has."""
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, domain)])
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5)).not_valid_after(now + timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(domain)]), critical=False)
            .sign(key, hashes.SHA256()))
    folder.mkdir(parents=True, exist_ok=True)
    cert_path, key_path = folder / "cert.pem", folder / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    return {"domain_name": domain, "tls_cert_path": str(cert_path), "tls_key_path": str(key_path)}


def _wait_for(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return predicate()


def test_member_reserves_on_the_world_host_and_is_dialled_through_it(_libp2p_setup, tmp_path):
    P2P = _libp2p_setup
    created = []
    try:
        world = _instance(P2P, tmp_path / "world", relay_service=True)
        created.append(world)
        member = _instance(P2P, tmp_path / "member", relay_service=False)
        created.append(member)
        other = _instance(P2P, tmp_path / "other", relay_service=False)
        created.append(other)

        # The member is connected to the world host when it asks, as after a join
        member.connect_to([pick_dialable_addr(world)])
        assert not [a for a in member.addresses if "/p2p-circuit" in a]

        raw_expiry = member.reserve_on_relay(world.peer_id)
        # The node prints this time as UTC, as it comes
        assert raw_expiry.endswith("Z"), f"the expiry of the reservation is not given in UTC: {raw_expiry}"
        expiry = datetime.fromisoformat(raw_expiry.replace("Z", "+00:00"))
        assert expiry - datetime.now(timezone.utc) > timedelta(minutes=30), \
            f"the reservation expires too early for the renewal of the node, which starts 15 minutes ahead: {expiry}"

        through_world = [a for a in member.addresses
                         if f"/p2p/{world.peer_id}/p2p-circuit/p2p/{member.peer_id}" in a
                         and "127.0.0.1" not in a]
        assert through_world, f"no circuit address through the world host in {member.addresses}"

        # A peer that knows the circuit address alone reaches the member, through the world
        circuit_tcp = [a for a in through_world if "/tcp/" in a and "/ws/" not in a] or through_world
        winner = other.connect_to([circuit_tcp[0]])
        assert winner["ID"] == member.peer_id
        assert any(f"/p2p/{world.peer_id}/p2p-circuit" in a for a in winner["Addrs"]), \
            f"not a connection relayed by the world host: {winner}"
    finally:
        for p2p in created:
            try:
                p2p.close()
            except Exception:
                pass


def test_reservation_is_lost_with_the_connection_to_the_world_and_must_be_asked_again(_libp2p_setup, tmp_path):
    """A relay drops the reservation of a peer the moment no connection to that peer is
    left. A member that disconnected from its world host gave the relay up: connecting
    again does not bring the slot back, and its circuit address leads nowhere until it
    reserves again."""
    P2P = _libp2p_setup
    created = []

    def _dialler(name):
        # A new instance at every check: one that reached the member once would dial its
        # direct addresses, and one that was refused answers from its dial backoff
        created.append(_instance(P2P, tmp_path / name, relay_service=False))
        return created[-1]

    def _no_reservation(dialler, address) -> bool:
        try:
            dialler.connect_to([address])
        except P2PError as e:
            return "NO_RESERVATION" in str(e.__cause__)
        return False

    try:
        world = _instance(P2P, tmp_path / "world", relay_service=True)
        created.append(world)
        member = _instance(P2P, tmp_path / "member", relay_service=False)
        created.append(member)
        world_addr = pick_dialable_addr(world)
        member_through_world = f"{world_addr}/p2p-circuit/p2p/{member.peer_id}"

        member.connect_to([world_addr])
        member.reserve_on_relay(world.peer_id)
        assert _dialler("before").connect_to([member_through_world])["ID"] == member.peer_id

        # The world must see the old connection close before the new one opens
        member.disconnect_from(world.peer_id)
        refused = _dialler("while-lost")
        assert _wait_for(lambda: _no_reservation(refused, member_through_world))

        member.connect_to([world_addr])
        assert any(f"/p2p/{world.peer_id}/p2p-circuit" in a for a in member.addresses), \
            f"the member no longer shows its circuit address: {member.addresses}"
        assert _no_reservation(_dialler("still-lost"), member_through_world), \
            "the world relays to a member that came back and did not reserve again"

        member.reserve_on_relay(world.peer_id)
        winner = _dialler("after").connect_to([member_through_world])
        assert winner["ID"] == member.peer_id
        assert any(f"/p2p/{world.peer_id}/p2p-circuit" in a for a in winner["Addrs"]), \
            f"not a connection relayed by the world host: {winner}"
    finally:
        for p2p in created:
            try:
                p2p.close()
            except Exception:
                pass


def test_member_is_reachable_again_after_a_loss_that_a_frame_healed(_libp2p_setup, tmp_path):
    """The loss nothing above the library sees: the world closes the connection and the
    next Session frame of the member opens it again by itself. The library asks for the
    slot again when the connection to its relay opens, with no request from the node."""
    P2P = _libp2p_setup
    created = []

    def _reached(name) -> bool:
        # A new instance at every attempt, as in the test above
        created.append(_instance(P2P, tmp_path / name, relay_service=False))
        try:
            return created[-1].connect_to([member_through_world])["ID"] == member.peer_id
        except P2PError:
            return False

    try:
        world = _instance(P2P, tmp_path / "world", relay_service=True)
        created.append(world)
        member = _instance(P2P, tmp_path / "member", relay_service=False)
        created.append(member)
        world_addr = pick_dialable_addr(world)
        member_through_world = f"{world_addr}/p2p-circuit/p2p/{member.peer_id}"

        member.connect_to([world_addr])
        member.reserve_on_relay(world.peer_id)
        assert _reached("before")

        world.disconnect_from(member.peer_id)
        assert _wait_for(lambda: not _reached(f"lost-{len(created)}"))

        assert member.send_on_protocol(world.peer_id, SESSION_CONTROL_PROTOCOL, b"\x00")
        assert _wait_for(lambda: _reached(f"after-{len(created)}"), timeout=5.0), \
            "the member came back to the world and is not reachable through it"
    finally:
        for p2p in created:
            try:
                p2p.close()
            except Exception:
                pass


@pytest.mark.parametrize("world_has_tls", [True, False])
def test_circuit_address_a_browser_can_use_needs_a_world_with_tls(_libp2p_setup, tmp_path, world_has_tls):
    """A browser opens secure WebSockets only, so it reaches a member through the world host
    only on a circuit address built on the secure WebSocket address of the world. The member
    has one when the world has TLS, and none when it has not."""
    P2P = _libp2p_setup
    created = []
    try:
        tls = _self_signed(tmp_path / "tls", WORLD_DOMAIN) if world_has_tls else {}
        world = _instance(P2P, tmp_path / "world", relay_service=True, **tls)
        created.append(world)
        member = _instance(P2P, tmp_path / "member", relay_service=False)
        created.append(member)
        assert world.tls_is_enabled is world_has_tls
        assert member.tls_is_enabled is False

        secure_ws = [a for a in world.addresses if a.startswith(f"/dns4/{WORLD_DOMAIN}/") and "/tls/ws/" in a]
        assert bool(secure_ws) is world_has_tls, world.addresses

        member.connect_to([pick_dialable_addr(world)])
        # The circuit addresses are built from what the member knows of the world when it asks:
        # the addresses the world announces reach it with identify, shortly after the connection
        assert _wait_for(lambda: len(member.get_node_addresses(world.peer_id)) > 1)
        if world_has_tls:
            assert _wait_for(lambda: any("/tls/ws" in a for a in member.get_node_addresses(world.peer_id)))
        member.reserve_on_relay(world.peer_id)

        for_a_browser = [a for a in member.addresses
                         if "/tls/ws/" in a and f"/p2p/{world.peer_id}/p2p-circuit/p2p/{member.peer_id}" in a]
        assert bool(for_a_browser) is world_has_tls, member.addresses
        if world_has_tls:
            assert any(a.startswith(f"/dns4/{WORLD_DOMAIN}/tcp/") for a in for_a_browser), for_a_browser
    finally:
        for p2p in created:
            try:
                p2p.close()
            except Exception:
                pass
