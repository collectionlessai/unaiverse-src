"""
       █████  █████ ██████   █████           █████ █████   █████ ██████████ ███████████    █████████  ██████████
      ░░███  ░░███ ░░██████ ░░███           ░░███ ░░███   ░░███ ░░███░░░░░█░░███░░░░░███  ███░░░░░███░░███░░░░░█
       ░███   ░███  ░███░███ ░███   ██████   ░███  ░███    ░███  ░███  █ ░  ░███    ░███ ░███    ░░░  ░███  █ ░ 
       ░███   ░███  ░███░░███░███  ░░░░░███  ░███  ░███    ░███  ░██████    ░██████████  ░░█████████  ░██████   
       ░███   ░███  ░███ ░░██████   ███████  ░███  ░░███   ███   ░███░░█    ░███░░░░░███  ░░░░░░░░███ ░███░░█   
       ░███   ░███  ░███  ░░█████  ███░░███  ░███   ░░░█████░    ░███ ░   █ ░███    ░███  ███    ░███ ░███ ░   █
       ░░████████   █████  ░░█████░░████████ █████    ░░███      ██████████ █████   █████░░█████████  ██████████
        ░░░░░░░░   ░░░░░    ░░░░░  ░░░░░░░░ ░░░░░      ░░░      ░░░░░░░░░░ ░░░░░   ░░░░░  ░░░░░░░░░  ░░░░░░░░░░ 
                 A Collectionless AI Project (https://collectionless.ai)
                 Registration/Login: https://unaiverse.io
                 Code Repositories:  https://github.com/collectionlessai/
                 Main Developers:    Stefano Melacci (Project Leader), Christian Di Maio, Tommaso Guidi
"""
import json
import jwt


class TokenVerifier:
    def __init__(self, public_key: str | bytes):
        """Initializes the `TokenVerifier` with a public key.

        This key is essential for securely decoding and verifying JSON Web Tokens (JWTs) issued by a corresponding
        private key. The public key can be provided as either a string or a bytes object.

        Args:
            public_key: The public key used for decoding and verification.
        """
        self.public_key = public_key

    def _decode(self, token: str | bytes,
                node_id: str | None = None, ip: str | None = None,
                hostname: str | None = None,
                port: int | None = None,
                p2p_peer: str | None = None):
        """Decode and validate the JWT, returning the payload dict if every check
        passes, otherwise None.

        Single decode/validation path shared by `verify_token` and
        `verify_token_claims` so the security checks (RS256 signature, required
        `exp`, and the optional node_id/ip/hostname/port/p2p_peer matches) never
        diverge between the two accessors.
        """
        # Decoding token using the public key
        try:
            payload = jwt.decode(token, self.public_key, algorithms=["RS256"], options={"require": ["exp"]})
        except jwt.DecodeError:
            return None
        except jwt.ExpiredSignatureError:  # This checks expiration time (required)
            return None
        except jwt.InvalidTokenError:
            return None
        except jwt.InvalidKeyError:
            return None
        except Exception:
            return None

        # The root serializes p2p_peers as a JSON string (json.dumps), so decode it
        # once here, at the token-codec boundary, into a real tuple of peer-ids. The
        # membership check below and both accessors then operate on the tuple rather
        # than the raw string (a bare tuple() over the string would explode into
        # single characters, and ``peer not in <string>`` would be a substring test).
        # exp is normalized to int for the same single-boundary reason.
        peers = payload.get("p2p_peers", ())
        if isinstance(peers, str):
            try:
                peers = json.loads(peers)
            except (ValueError, TypeError):
                return None
        payload["p2p_peers"] = tuple(peers)
        payload["exp"] = int(payload.get("exp", 0))

        # Checking optional information
        if node_id is not None and payload.get("node_id", None) != node_id:
            return None
        if ip is not None and payload.get("ip", None) != ip:
            return None
        if hostname is not None and payload.get("hostname", None) != hostname:
            return None
        if port is not None and payload.get("port", None) != port:
            return None
        if p2p_peer is not None and p2p_peer not in payload.get("p2p_peers", ()):
            return None

        return payload

    def verify_token(self, token: str | bytes,
                     node_id: str | None = None, ip: str | None = None,
                     hostname: str | None = None,
                     port: int | None = None,
                     p2p_peer: str | None = None):
        """Verifies a JSON Web Token (JWT) against a set of criteria.

        The token is decoded with the provided public key (RS256), then optional
        checks ensure the payload matches `node_id`, `ip`, `hostname`, `port`, and
        that `p2p_peer` is present in the token's `p2p_peers` list.

        Args:
            token: The JWT to verify, as a string or bytes object.
            node_id: Optional `node_id` to check against the token's payload.
            ip: Optional IP address to check.
            hostname: Optional hostname to check.
            port: Optional port number to check.
            p2p_peer: Optional peer identifier to check within the `p2p_peers` list.

        Returns:
            A tuple `(node_id, cv_hash)` from the token's payload if all checks pass,
            otherwise `(None, None)`.
        """
        payload = self._decode(token, node_id=node_id, ip=ip, hostname=hostname, port=port, p2p_peer=p2p_peer)
        if payload is None:
            return None, None
        return payload.get("node_id", None), payload.get("cv_hash", None)

    def verify_token_claims(self, token: str | bytes,
                            node_id: str | None = None, ip: str | None = None,
                            hostname: str | None = None,
                            port: int | None = None,
                            p2p_peer: str | None = None):
        """Like `verify_token` but also returns the root-attested `p2p_peers`
        binding, for callers that need the full root-attested claims instead of
        trusting an offered profile.

        Returns:
            A tuple `(node_id, cv_hash, p2p_peers, exp, code_hash)` if all checks pass,
            otherwise `(None, None, (), 0, None)`. `p2p_peers` is a tuple of the node's
            peer-ids, `exp` is the JWT expiry (unix seconds) for callers that track
            token lapse, and `code_hash` is the root-attested digest of the world's declared
            code (None when the token carries no such claim: agent nodes, or a world
            that declared no source repository).
        """
        payload = self._decode(token, node_id=node_id, ip=ip, hostname=hostname, port=port, p2p_peer=p2p_peer)
        if payload is None:
            return None, None, (), 0, None
        return (payload.get("node_id", None), payload.get("cv_hash", None),
                payload.get("p2p_peers", ()), payload.get("exp", 0),
                payload.get("code_hash", None))

    def __str__(self):
        key_str = self.public_key.decode("utf-8") if isinstance(self.public_key, bytes) else self.public_key
        return f"[{self.__class__.__name__}] public_key: {key_str + '...'}"
