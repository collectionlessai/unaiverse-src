"""Unit tests for TokenVerifier.

RT enrichment: `verify_token_claims` exposes the root-attested `p2p_peers`
binding (and cv_hash) so the Session handshake can consume the public<->world
peer-id binding instead of trusting the offered profile. `verify_token` keeps its
original 2-tuple contract for the per-message piggyback path; both share a single
decode/validation path so the security checks never diverge.
"""
import json
import time

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from unaiverse.networking.node.tokens import TokenVerifier

PEERS = ["12D3KooWpublicccccccccccccccccccccccccccccccccccccc",
         "12D3KooWworldddddddddddddddddddddddddddddddddddddd"]


def _keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv = key.private_bytes(serialization.Encoding.PEM,
                             serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())
    pub = key.public_key().public_bytes(serialization.Encoding.PEM,
                                         serialization.PublicFormat.SubjectPublicKeyInfo)
    return priv, pub


def _mint(priv, **payload):
    return jwt.encode(payload, priv, algorithm="RS256")


def _valid_payload(**over):
    # The root mints the token with p2p_peers as a JSON string (json.dumps), not a
    # native JSON array, so reproduce that here: jwt would otherwise round-trip a
    # Python list as a native array and the wire format would diverge from production.
    p = {"node_id": "node-N", "cv_hash": "cv-abc", "p2p_peers": json.dumps(PEERS),
         "exp": int(time.time()) + 3600}
    p.update(over)
    return p


def test_verify_token_claims_returns_node_id_cv_hash_and_p2p_peers():
    priv, pub = _keypair()
    tv = TokenVerifier(pub)
    token = _mint(priv, **_valid_payload())

    # Back-compat: verify_token still returns exactly the 2-tuple.
    assert tv.verify_token(token, p2p_peer=PEERS[0]) == ("node-N", "cv-abc")

    # RT: verify_token_claims additionally returns the binding and the JWT expiry.
    node_id, cv_hash, peers, exp = tv.verify_token_claims(token, p2p_peer=PEERS[1])
    assert node_id == "node-N"
    assert cv_hash == "cv-abc"
    assert peers == tuple(PEERS)
    assert isinstance(exp, int) and exp > 0


def test_verify_token_claims_rejects_peer_not_in_binding():
    priv, pub = _keypair()
    tv = TokenVerifier(pub)
    token = _mint(priv, **_valid_payload())
    # A speaker whose peer-id is not attested by the token fails the binding check.
    assert tv.verify_token_claims(token, p2p_peer="12D3KooWintruder") == (None, None, (), 0)
    assert tv.verify_token(token, p2p_peer="12D3KooWintruder") == (None, None)


def test_verify_token_claims_rejects_expired_token():
    priv, pub = _keypair()
    tv = TokenVerifier(pub)
    token = _mint(priv, **_valid_payload(exp=int(time.time()) - 10))
    assert tv.verify_token_claims(token, p2p_peer=PEERS[0]) == (None, None, (), 0)


def test_verify_token_claims_rejects_wrong_signature():
    priv, _ = _keypair()
    _, other_pub = _keypair()
    tv = TokenVerifier(other_pub)   # verifier holds a DIFFERENT public key
    token = _mint(priv, **_valid_payload())
    assert tv.verify_token_claims(token, p2p_peer=PEERS[0]) == (None, None, (), 0)


def test_verify_token_claims_returns_full_peer_ids_not_chars():
    # Regression for the json.dumps payload: p2p_peers must come back as the tuple
    # of full peer-id strings, never a tuple of single characters (which is what
    # tuple() over the raw JSON string used to produce before the _decode fix).
    priv, pub = _keypair()
    tv = TokenVerifier(pub)
    token = _mint(priv, **_valid_payload())
    _, _, peers, exp = tv.verify_token_claims(token, p2p_peer=PEERS[0])
    assert peers == tuple(PEERS)
    assert all(p in PEERS for p in peers)
    assert isinstance(exp, int)


def test_verify_token_claims_rejects_substring_of_peer():
    # A fragment of a real peer-id must NOT pass the binding check. Before the fix
    # the membership test ran on the raw JSON string, so any substring slipped
    # through; now it runs on the decoded tuple.
    priv, pub = _keypair()
    tv = TokenVerifier(pub)
    token = _mint(priv, **_valid_payload())
    assert tv.verify_token_claims(token, p2p_peer=PEERS[0][:20]) == (None, None, (), 0)
    assert tv.verify_token(token, p2p_peer=PEERS[0][:20]) == (None, None)
