"""Token Manager Module - UNaIVERSE Networking.

This module provides the `TokenVerifier` class, which is responsible for verifying JSON Web Tokens (JWTs) used for node authentication in the UNaIVERSE network.
The `TokenVerifier` class utilizes a public key to decode and validate JWTs, ensuring that they are properly signed and contain the expected claims.

A Collectionless AI Project (https://collectionless.ai) / UNaIVERSE SRL (https://unaiverse.ai)

- Registration/Login: https://unaiverse.io
- Code Repositories: https://github.com/collectionlessai/
- Main Developers: Stefano Melacci (Project Leader), Christian Di Maio, Tommaso Guidi
"""

# Standard library imports
from typing import TypedDict
import jwt

# 3rd party imports
from pydantic import BaseModel, Field, field_validator


class TokenVerifiedData(TypedDict):
    node_id: str  # Unique identifier for the node, a UUID4 string (without dashes)
    cv_hash: str  # The hash of the node's curriculum vitae (CV) of the node


class TokenVerifier(BaseModel):
    """A class responsible for verifying JSON Web Tokens (JWTs) used for node authentication in the UNaIVERSE network.
    The `TokenVerifier` class utilizes a public key to decode and validate JWTs, ensuring that they are properly signed and contain the expected claims.

    Args:
        public_key (bytes | str): The public key used to verify the JWT signatures. This key should correspond to the private key used to sign the tokens.
    """

    public_key: bytes | str = Field(
        ..., description="The public key used to verify JWT signatures."
    )

    @field_validator("public_key", mode="before")
    def validate_public_key(cls, value) -> bytes:
        """Validates and processes the public key input, ensuring it is in bytes format.

        This validator checks if the provided public key is a string and encodes it to bytes if necessary. If the input is already in bytes, it is returned as is.

        Args:
            value: The public key to validate, which should be a string or bytes.
        Returns:
            The public key in bytes format, ready for use in JWT verification.
        Raises:
            ValueError: If the input value is not a string or bytes, indicating an invalid type for the public key.
        """
        if not isinstance(value, (str, bytes)):
            raise ValueError(
                f"Invalid type for public_key: expected str or bytes, got {type(value)}"
            )

        if isinstance(value, str):
            value = value.encode("utf-8")

        return value

    def verify_token(
        self,
        token: str | bytes,
        node_id: str | None = None,
        ip: str | None = None,
        hostname: str | None = None,
        port: int | None = None,
        p2p_peer: str | None = None,
    ) -> TokenVerifiedData | tuple[None, None]:
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
            A `TokenVerifiedData` dictionary containing the `node_id` and `cv_hash` from the token's payload if all checks pass. Otherwise,
            it returns a tuple of `(None, None)` to indicate verification failure.

        Example:
            ```python
                verifier = TokenVerifier(public_key=b"your_public_key_here")
                node_id, cv_hash = verifier.verify_token(
                    token="your_jwt_token_here",
                    node_id="expected_node_id"
                )
                if node_id is not None:
                    print(f"Token is valid for node_id: {node_id} with CV hash: {cv_hash}")
                else:
                    print("Token verification failed.")

                # If you want to check for a specific peer in the token's `p2p_peers` list:

                node_id, cv_hash = verifier.verify_token(
                    token="your_jwt_token_here",
                    p2p_peer="peer_id_to_check"
                )
                if node_id is not None:
                    print(f"Token is valid and contains the specified peer. Node ID: {node_id}, CV hash: {cv_hash}")
                else:
                    print("Token verification failed or specified peer not found in token.")

                # You can also combine multiple checks:

                node_id, cv_hash = verifier.verify_token(
                    token="your_jwt_token_here",
                    node_id="expected_node_id",
                    ip="expected_ip_address",
                    hostname="expected_hostname",
                    port=expected_port_number,
                    p2p_peer="peer_id_to_check"
                )
                if node_id is not None:
                    print(f"Token is valid and matches all specified criteria. Node ID: {node_id}, CV hash: {cv_hash}")
                else:
                    print("Token verification failed or one of the specified criteria did not match the token's payload.")
            ```
        """

        # Decoding token using the public key
        try:
            payload = jwt.decode(token, self.public_key, algorithms=["RS256"])
        except jwt.DecodeError:
            return None, None
        except jwt.ExpiredSignatureError:  # This checks expiration time (required)
            return None, None
        except (
            Exception
        ) as e:  # Catching any other exceptions that may occur during decoding
            raise RuntimeError(
                f"An unexpected error occurred during token verification: {e}"
            )

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
        return f"[{self.__class__.__name__}] public_key: {self.public_key[0:50] + b'...' if isinstance(self.public_key, bytes) else self.public_key[0:50] + '...'}"
