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

    def verify_token(self, token: str | bytes,
                     node_id: str | None = None, ip: str | None = None,
                     hostname: str | None = None,
                     port: int | None = None,
                     p2p_peer: str | None = None):
        """Verifies a JSON Web Token (JWT) against a set of criteria.

        The method first attempts to decode the token using the provided public key and the RS256 algorithm,
        handling `DecodeError` and `ExpiredSignatureError`. It then performs optional checks to ensure that
        the token's payload matches specific network identifiers, such as `node_id`, `ip`, `hostname`, and `port`.
        It can also verify if a specific peer is present in the token's list of `p2p_peers`.

        Args:
            token: The JWT to verify, as a string or bytes object.
            node_id: Optional `node_id` to check against the token's payload.
            ip: Optional IP address to check.
            hostname: Optional hostname to check.
            port: Optional port number to check.
            p2p_peer: Optional peer identifier to check within the `p2p_peers` list.

        Returns:
            A tuple containing the `node_id` and `cv_hash` from the token's payload if all checks pass. Otherwise,
            it returns a tuple of `(None, None)`.
        """

        # Decoding token using the public key
        try:
            payload = jwt.decode(token, self.public_key, algorithms=["RS256"])
        except jwt.DecodeError as e:
            return None, None
        except jwt.ExpiredSignatureError as e:  # This checks expiration time (required)
            return None, None

        # Checking optional information
        if node_id is not None and payload["node_id"] != node_id:
            return None, None
        if ip is not None and payload["ip"] != ip:
            return None, None
        if hostname is not None and payload["hostname"] != hostname:
            return None, None
        if port is not None and payload["port"] != port:
            return None, None
        if p2p_peer is not None and p2p_peer not in payload["p2p_peers"]:
            return None, None

        # All ok
        return payload["node_id"], payload["cv_hash"]

    def __str__(self):
        return f"[{self.__class__.__name__}] public_key: {self.public_key[0:50] + b'...'}"
