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
        self.public_key = public_key

    def verify_token(self, token: str | bytes,
                     node_id: str | None = None, ip: str | None = None,
                     hostname: str | None = None,
                     port: int | None = None,
                     p2p_peer: str | None = None) -> tuple[str, str] | tuple[None, None]:

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
