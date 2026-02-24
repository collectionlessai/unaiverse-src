"""P2P Module - UNaIVERSE Networking - P2P.

This module provides the `P2P` class, which serves as a Python wrapper for the Go libp2p shared library.
The `P2P` class allows users to initialize a libp2p node, connect to peers, send and receive messages, manage PubSub subscriptions, and handle relay functionality.
It abstracts the complexities of interacting with the Go library and provides a user-friendly interface for P2P networking in the UNaIVERSE ecosystem.

A Collectionless AI Project (https://collectionless.ai) / UNaIVERSE SRL (https://unaiverse.ai)

- Registration/Login: https://unaiverse.io
- Code Repositories: https://github.com/collectionlessai/
- Main Developers: Stefano Melacci (Project Leader), Christian Di Maio, Tommaso Guidi
"""

# Standard library imports
import base64
import datetime
import os
import socket
import logging
import threading
from narwhals import Enum
from typing_extensions import Self
from typing import (
    Generic,
    Optional,
    List,
    Dict,
    Any,
    TYPE_CHECKING,
    ClassVar,
    TypeVar,
    Annotated,
)

# 3rd party imports
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    model_serializer,
    validate_call,
)

# 1st party imports
from .lib_types import TypeInterface


# Conditional import for type hinting to avoid circular dependencies
if TYPE_CHECKING:
    try:
        from unaiverse.networking.p2p.golibp2p import GoLibP2P
    except ImportError:
        pass


LETS_ENCRYPT_TEST_HOST = (
    "acme-v02.api.letsencrypt.org"  # Used to test connectivity for AutoTLS
)

logger = logging.getLogger("P2P")


class P2PError(Exception):
    """Custom exception class for P2P library errors."""

    pass


class P2PConfig(BaseModel):
    """P2P Node configuration

    Attributes:
        max_instances (int): Maximum number of P2P instances allowed (default: 32).
        max_channels (int): Maximum number of channels per instance (default: 100).
        max_queue_per_channel (int): Maximum number of messages queued per channel (default: 50).
        max_message_size (int): Maximum size of messages in bytes (default: 50 MB).
    """

    max_instances: int = Field(
        default=32, description="Maximum number of P2P instances allowed.", ge=1
    )
    max_channels: int = Field(
        default=100, description="Maximum number of channels per instance.", ge=1
    )
    max_queue_per_channel: int = Field(
        default=50, description="Maximum number of messages queued per channel.", ge=1
    )
    max_message_size: int = Field(
        default=50 * 1024 * 1024,
        description="Maximum size of messages in bytes (default: 50 MB).",
        ge=1,
    )
    enable_logging: bool = Field(
        default=False,
        description="Whether to enable detailed logging from the Go library (default: False).",
    )


class GoLibConfig(BaseModel):
    """Go library configuration

    Attributes:
        identity_dir (str): Directory path to load/store the node's private key and certificates.
        predefined_port (int): The first port to listen on (0 for random).
        listen_ips (Optional[List[str]]): A list of specific IP addresses to listen on. Defaults to ['0.0.0.0'] if None.
        relay_enable_client (bool): Enable listening to relayed connections for this node.
        relay_enable_service (bool): Enable relay service capabilities for this node.
        relay_with_broad_limits (bool): Whether to use broad limits for relay service (if enabled).
        tls_auto_tls (bool): Whether to enable AutoTLS certificate management (requires internet access).
        tls_domain (Optional[str]): Optional domain name for TLS certificate (required if auto_tls is True).
        tls_cert_path (Optional[str]): Optional path to a custom TLS certificate file (PEM format).
        tls_key_path (Optional[str]): Optional path to a custom TLS private key file (PEM format).
        network_isolated (bool): Whether to run the node in isolated mode (no network).
        network_force_public (bool): If you already know that the node is public this forces its public reachability. Otherwise, it tries every possible attempt to make the node publicly reachable (UPnP, HolePunching, AutoNat via DHT...).
        dht_enabled (bool): Whether to enable the DHT for peer discovery.
        dht_keep (bool): Whether to keep DHT records for this node (only relevant if dht_enabled is True).
    """

    identity_dir: str = Field(
        ...,
        description="Directory path to load/store the node's private key and certificates.",
    )
    predefined_port: int = Field(
        default=0, description="The first port to listen on (0 for random)."
    )
    listen_ips: Optional[List[str]] = Field(
        default=None,
        description="A list of specific IP addresses to listen on. Defaults to ['0.0.0.0'] if None.",
    )
    relay_enable_client: bool = Field(
        default=True,
        description="Enable listening to relayed connections for this node.",
    )
    relay_enable_service: bool = Field(
        default=False, description="Enable relay service capabilities for this node."
    )
    relay_with_broad_limits: bool = Field(
        default=False,
        description="Whether to use broad limits for relay service (if enabled).",
    )
    enable_tls: bool = Field(
        default=False,
        description="Whether to enable TLS for the node (either AutoTLS or custom TLS). Requires internet access for AutoTLS.",
    )
    tls_auto_tls: bool = Field(
        default=False,
        description="Whether to enable AutoTLS certificate management (requires internet access).",
    )
    tls_domain_name: Optional[str] = Field(
        default=None,
        description="Optional domain name for TLS certificate (required if auto_tls is True).",
    )
    tls_cert_path: Optional[str] = Field(
        default=None,
        description="Optional path to a custom TLS certificate file (PEM format).",
    )
    tls_key_path: Optional[str] = Field(
        default=None,
        description="Optional path to a custom TLS private key file (PEM format).",
    )
    network_isolated: bool = Field(
        default=False,
        description="Whether to run the node in isolated mode (no network).",
    )
    network_force_public: bool = Field(
        default=False,
        description="If you already know that the node is public this forces its public reachability. Otherwise, it tries every possible attempt to make the node publicly reachable (UPnP, HolePunching, AutoNat via DHT...).",
    )
    dht_enabled: bool = Field(
        default=False, description="Whether to enable the DHT for peer discovery."
    )
    dht_keep: bool = Field(
        default=True,
        description="Whether to keep DHT records for this node (only relevant if dht_enabled is True).",
    )

    @field_validator("tls_domain_name", mode="before")
    @classmethod
    def validate_tls_domain_name(cls, v: Optional[str]) -> Optional[str]:
        """Validates that tls_domain_name is a valid type if provided. Moreover it checks that if a domain name is provided, then TLS must be enabled."""
        if v is not None and not cls.enable_tls:
            raise ValueError("TLS must be enabled if tls_domain_name is provided.")
        return v

    @model_validator(mode="after")
    def validate_object(self) -> Self:
        """Validates the identity directory path and ensures it exists.

        This validator checks if the provided `identity_dir` is a valid string and creates the directory if it does not already exist.

        Raises:
            ValueError: If the `identity_dir` is not a valid string or if the directory cannot be created.
        """
        identity_dir = self.identity_dir
        if not identity_dir.strip():
            raise ValueError("Invalid identity_dir: must be a non-empty string.")

        try:
            os.makedirs(identity_dir, exist_ok=True)
            logger.info(f"Identity directory '{identity_dir}' is ready.")
        except Exception as e:
            raise ValueError(
                f"Failed to create identity directory '{identity_dir}': {e}"
            ) from e

        # Now check auto_tls requirements
        has_custom_tls_args = (
            (self.tls_cert_path is not None)
            or (self.tls_key_path is not None)
            or (self.tls_domain_name is not None)
        )
        if has_custom_tls_args:
            if not all([
                self.tls_domain_name is not None,
                self.tls_cert_path is not None,
                self.tls_key_path is not None,
            ]):
                raise ValueError(
                    "Custom TLS requires 'domain_name', 'tls_cert_path' and 'tls_key_path'."
                )

        self.tls_auto_tls = self.enable_tls and not has_custom_tls_args
        if self.tls_auto_tls:  # If auto TLS is enabled, check connectivity to Let's Encrypt servers (port 443) and log a warning if not reachable
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.settimeout(5)  # 5 second timeout
            try:
                test_socket.connect((LETS_ENCRYPT_TEST_HOST, 443))
                logger.info(
                    "AutoTLS enabled and connectivity to Let's Encrypt servers verified."
                )
            except (socket.timeout, socket.error) as e:
                raise ValueError(
                    f"AutoTLS enabled but cannot connect to Let's Encrypt servers: {e}. "
                    "TLS certificate management may fail without internet access."
                )
        return self

    @model_serializer()
    def serialize_config(self) -> dict[str, Any]:
        """Serializes the P2PConfig instance to a dictionary.

        This method converts the P2PConfig dataclass instance into a dictionary representation, which can be used for configuration purposes when initializing the Go library.

        Returns:
            A dictionary representing the P2PConfig instance.
        """
        flat_dict = self.model_dump()
        structured_dict: dict[str, Any] = {
            "identity_dir": flat_dict.get("identity_dir"),
            "predefined_port": flat_dict.get("predefined_port"),
            "listen_ips": flat_dict.get("listen_ips"),
            "relay": {
                "enable_client": flat_dict.get("relay_enable_client", True),
                "enable_service": flat_dict.get("relay_enable_service", False),
                "with_broad_limits": flat_dict.get("relay_with_broad_limits", False),
            },
            "tls": {
                "auto_tls": flat_dict.get("tls_auto_tls", False),
                "domain": flat_dict.get("tls_domain", ""),
                "cert_path": flat_dict.get("tls_cert_path", ""),
                "key_path": flat_dict.get("tls_key_path", ""),
            },
            "network": {
                "isolated": flat_dict.get("network_isolated", False),
                "force_public": flat_dict.get("network_force_public", False),
            },
            "dht": {
                "enabled": flat_dict.get("dht_enabled", False),
                "keep": flat_dict.get("dht_keep", True)
                and flat_dict.get("dht_enabled", False),
            },
        }
        return structured_dict


class P2PChannel(BaseModel):
    """ It represents the communication channel  

    Attributes:
        source_peer_id (str): The Peer ID of the sender peer.
        destination_peer_id (str): The Peer ID of the receiver peer (empty for PubSub).
        channel_type (str): The type of channel, either 'dm' for direct messages or 'ps' for PubSub.
        topic_name (Optional[str]): The name of the PubSub topic (only relevant for PubSub channels).
        content_type (Optional[str]): The content type of the messages sent on this channel (e.g., 'json', 'protobuf').
        channel_trail (Optional[str]): An optional string to further differentiate channels.

    """
class P2P(BaseModel):
    """
    Python wrapper for the Go libp2p shared library.

    This class initializes a libp2p node, provides methods to interact with the
    p2p network (connect, send/receive messages, pubsub, relay), and manages
    the lifecycle of the underlying Go node.

    Attributes:
        libp2p (LibP2P): Static class attribute holding the loaded Go library instance.
                         Must be set before instantiating P2P. Example: P2P.libp2p = LibP2P()
        _type_interface (TypeInterface): Shared type interface for converting between Python and Go types.
        _MAX_INSTANCES (int): Maximum number of P2P instances allowed (default: 32).
        _MAX_NUM_CAHNNELS (int): Maximum number of channels per instance (default: 100).
        _MAX_QUEUE_PER_CHANNEL (int): Maximum number of messages queued per channel (default: 50).
        _MAX_MESSAGE_SIZE (int): Maximum size of messages in bytes (default: 50 MB).
        _library_initialized (bool): Flag indicating whether the Go library has been initialized.
        _initialize_lock (threading.Lock): Lock to ensure thread-safe library initialization.
        _instance_ids (List[bool]): List to track assigned instance IDs for P2P instances.
        _instance_lock (threading.Lock): Lock to ensure thread-safe instance ID assignment.

    Properties:
        peer_id (str): The Peer ID of the initialized local node.
        addresses (Optional[List[str]]): List of multiaddresses the local node is listening on.
        is_public (bool): Whether the node is publicly reachable.
        peer_map (Dict[str, Any]): A dictionary to potentially store information about connected peers
            (managed manually or by polling thread).
    """

    # --- Class-level state ---
    libp2p: ClassVar["GoLibP2P"]  # Static variable for the loaded Go library
    _type_interface: ClassVar[
        "TypeInterface"
    ]  # Shared type interface for all instances
    _library_initialized: ClassVar[bool] = False
    _initialize_lock: ClassVar[threading.Lock] = threading.Lock()

    _instance_ids: ClassVar[List[bool]] = (
        [
            False,
        ]
        * 32
    )  # Default to 32 instances, can be resized based on config. True means assigned, False means available.

    _instance_lock: ClassVar[threading.Lock] = threading.Lock()

    # --- Config class variables for configuration ---
    p2p_config: P2PConfig = Field(
        default_factory=P2PConfig, description="Configuration for P2P node."
    )
    go_p2p_config: GoLibConfig = Field(
        ..., description="Configuration for Go library initialization."
    )
    _instance: int = Field(
        default=-1, description="Assigned instance ID for this P2P instance."
    )
    _peer_id: Optional[str] = Field(
        default=None, description="Peer ID of the current instance."
    )
    _is_public: bool = Field(
        default=False, description="Whether the P2P instance is publicly reachable."
    )

    @model_validator(mode="before")
    def setup_library(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initializes the underlying Go library. Must be called once. This is called automatically.
        """
        with P2P._initialize_lock:
            if P2P._library_initialized:
                logger.warning("P2P library is already initialized. Skipping setup.")
                return data

            logger.setLevel(logging.CRITICAL)
            __log_config = {}
            # Configure Python logging based on the flag
            if data["p2p_config"]["enable_logging"]:
                logger.setLevel(logging.INFO)
                __log_config = {
                    "net/identify": "debug",
                    "unailib": "debug",
                    # 'autotls': 'debug',
                    # 'p2p-forge': 'debug',
                    "nat": "debug",
                    "basichost": "debug",
                    "p2p-circuit": "debug",
                    "relay": "debug",
                    "p2p-holepunch": "debug",
                    "tcp-tpt": "debug",
                    "connmgr": "debug",
                    "dht": "debug",
                    "autorelay": "debug",
                    "autonat": "debug",
                    # 'rcmgr': 'debug',
                    "swarm2": "debug",
                    "yamux": "debug",
                }

            logger.info(
                "🐍 Setting up and initializing P2P library core with user settings..."
            )
            P2P._type_interface = TypeInterface(libp2p=P2P.libp2p)

            # Update class attributes if they were overridden
            P2P._instance_ids = [
                False,
            ] * data["p2p_config"]["max_instances"]  # Resize the tracking list

            # Call the Go function to set up its internal state
            logger.info("🐍 Initializing Go library core...")

            __max_instances: int = data["p2p_config"]["max_instances"]
            __max_channels: int = data["p2p_config"]["max_channels"]
            __max_queue_per_channel: int = data["p2p_config"]["max_queue_per_channel"]
            __max_message_size: int = data["p2p_config"]["max_message_size"]

            P2P.libp2p.InitializeLibrary(
                P2P._type_interface.to_go_int(__max_instances),
                P2P._type_interface.to_go_int(__max_channels),
                P2P._type_interface.to_go_int(__max_queue_per_channel),
                P2P._type_interface.to_go_int(__max_message_size),
                P2P._type_interface.to_go_json(__log_config),
            )

            P2P._library_initialized = True  # Set the flag to indicate that the library has been initialized, and turns the method into a no-op on subsequent calls
            logger.info("✅ Go library initialized successfully.")
            return data

    @model_validator(mode="after")
    def start_instance(self) -> Self:
        """
        Initializes and starts a new libp2p node.

        Raises:
            P2PError: If the node creation fails in the Go library.
            AttributeError: If P2P.libp2p has not been set before instantiation.
        """

        assigned_instance_id = -1
        with P2P._instance_lock:
            assigned_instance_id = (
                P2P._instance_ids.index(False) if False in P2P._instance_ids else -1
            )
            if assigned_instance_id == -1:
                raise P2PError(
                    f"Cannot create new P2P instance: Maximum number of instances "
                    f"({self.p2p_config.max_instances})."
                )

        self._instance: int = assigned_instance_id
        logger.info(
            f"🚀 Attempting to initialize P2P Node with auto-assigned Instance ID: {self._instance}"
        )

        logger.info(f"🐍 Creating Node (Instance ID: {self._instance})...")
        try:
            # Call the Go function
            result_ptr = P2P.libp2p.CreateNode(
                P2P._type_interface.to_go_int(self._instance),
                P2P._type_interface.to_go_json(self.go_p2p_config.model_dump_json()),
            )
            result = P2P._type_interface.from_go_ptr_to_json(result_ptr)

            if result is None:
                err_msg = "Received null result from Go CreateNode."
                logger.error(f"[Instance {self._instance}] {err_msg}")
                raise P2PError(f"[Instance {self._instance}] {err_msg}")

            result = CreateNodeResponse.model_validate(result)

            if (
                result.state == ResponseState.ERROR
                or result.state == ResponseState.EMPTY
                or result.data is None
            ):
                err_msg = result.error or "Unknown Go error on CreateNode"
                logger.error(f"[Instance {self._instance}] Go error: {err_msg}")
                raise P2PError(
                    f"[Instance {self._instance}] Failed to create node: {err_msg}"
                )

            initial_addresses = result.data.addresses
            self._is_public = result.data.is_public
            
            try:
                self._peer_id = initial_addresses[0].split("/")[-1]
            except IndexError:
                err_msg = "Received empty addresses list from Go CreateNode."
                logger.error(f"[Instance {self._instance}] {err_msg}")
                raise P2PError(f"[Instance {self._instance}] {err_msg}")

            log_message = (
                f"✅ [Instance {self._instance}] Node created with ID: {self._peer_id}\n"
                + f"👂 [Instance {self._instance}] Listening on: {initial_addresses}\n"
                + f"🌐 [Instance {self._instance}] Publicly reachable: {self._is_public}\n"
                + f"🎉 [Instance {self._instance}] Node initialized successfully."
            )

            logger.info(log_message)

        except Exception as e:
            logger.error(f"❌ [Instance {self._instance}] Node creation failed: {e}")

            # Reclaim the instance ID using the _instance_ids list
            if self._instance != -1:  # Check if an ID was actually assigned
                with P2P._instance_lock:
                    P2P._instance_ids[self._instance] = False
                    logger.info(
                        f"[Instance {self._instance}] "
                        f"Reclaimed instance ID {self._instance} due to creation failure."
                    )
            raise  # Re-raise the exception that caused the failure

        logger.info("🎉 Node created successfully and background polling started.")
        return self

    @validate_call
    def connect_to(
        self, multiaddrs: Annotated[List[str], Field(min_length=1)]
    ) -> "PeerAddrInfo":
        """
        Establishes a connection with a remote peer.

        Args:
            multiaddrs: A list of multiaddresses (strings) to connect to. Must contain at least one valid multiaddress.

        Returns:
            A dictionary containing the connected peer's AddrInfo (ID and Addrs).

        Raises:
            P2PError: If the connection fails.
            ValueError: If the multiaddr is invalid.

        Example:
            ```python
            try:
                peer_info = p2p_instance.connect_to([
                    "/ip4/.../udp/4001/quic/p2p/QmPeerID",
                    "/ip4/.../tcp/4001/p2p/QmPeerID"
                ])
            except P2PError as e:
                print(f"Connection failed: {e}")
            except ValueError as e:
                print(f"Invalid multiaddress: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")
            print(peer_info.id)  # Peer ID string
            print(peer_info.addrs)  # List of multiaddress strings
            ```
        """

        try:
            dest_peer_id: str = multiaddrs[0].split("/")[-1]
        except Exception as e:
            logger.error(f"❌ Invalid multiaddress provided: {e}")
            raise ValueError(f"Invalid multiaddress provided: {e}") from e

        logger.info(f"📞 Attempting to connect to: {dest_peer_id}...")
        try:
            result_ptr = P2P.libp2p.ConnectTo(
                P2P._type_interface.to_go_int(self._instance),
                P2P._type_interface.to_go_json(multiaddrs),
            )
            result = ConnectToResponse.model_validate(
                P2P._type_interface.from_go_ptr_to_json(result_ptr)
            )
        except Exception as e:
            logger.error(f"❌ Connection to {dest_peer_id} failed: {e}")
            raise P2PError(f"Connection to {dest_peer_id} failed") from e

        if result.state == ResponseState.ERROR:
            logger.error(f"Failed to connect to peer '{dest_peer_id}': {result.error}")
            raise P2PError(
                f"Failed to connect to peer '{dest_peer_id}': {result.error}"
            )

        if not result.data:
            logger.error("Failed to connect to peer, received empty peer info.")
            raise P2PError("Failed to connect to peer, received empty peer info.")

        logger.info(f"✅ Connection initiated to peer: {result.data.id}")

        return result.data

    @validate_call
    def disconnect_from(self, peer_id: str) -> bool:
        """
        Closes connections to a specific peer and removes tracking.

        Args:
            peer_id: The Peer ID string of the peer to disconnect from.

        Returns:
            True if the disconnection was successful.

        Raises:
            P2PError: If disconnecting fails.
            ValueError: If the peer_id is invalid.

        Example:
            ```python
            try:
                success = p2p_instance.disconnect_from("QmPeerID")
                if success:
                    print("Successfully disconnected.")
            except P2PError as e:
                print(f"Disconnection failed: {e}")
            except ValueError as e:
                print(f"Invalid Peer ID: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")
            ```
        """

        # Basic peer ID format check (Qm... or 12D3...)
        if not (peer_id.startswith("Qm") or peer_id.startswith("12D3")):
            logger.warning(
                f"⚠️ Warning: Peer ID '{peer_id}' does not look like a standard v0 or v1 ID."
            )

        logger.info(f"🔌 Attempting to disconnect from peer: {peer_id}...")
        try:
            result_ptr = P2P.libp2p.DisconnectFrom(
                P2P._type_interface.to_go_int(self._instance),
                P2P._type_interface.to_go_string(peer_id),
            )
            result = StringResponse.model_validate(
                P2P._type_interface.from_go_ptr_to_json(result_ptr)
            )
        except Exception as e:
            logger.error(f"❌ Disconnection from {peer_id} failed: {e}")
            raise P2PError(f"Disconnection from {peer_id} failed") from e

        if result.state == ResponseState.ERROR:
            logger.error(
                f"Failed to disconnect from peer '{peer_id}': {result.error}"
            )
            raise P2PError(
                f"Failed to disconnect from peer '{peer_id}': "
                f"{result.error}"
            )

        logger.info(f"✅ Successfully disconnected from {peer_id}")
        return True

    @validate_call
    def send_message_to_peer(self, channel: Annotated[str, Field(pattern=r".+::dm:.+")], msg_bytes: bytes) -> bool:
        """
        Sends a direct message to a specific peer.

        Args:
            channel: The string identifying the channel for the communication. Must include '::dm:' to be a valid channel.
            msg_bytes: The message to send (bytes).

        Returns:
            True if the message was sent successfully.

        Raises:
            P2PError: If message sending fails (based on return code).
            ValueError: If inputs are invalid.
            TypeError: If data is not bytes.
            
        Example:
            ```python
            try:
                connected_peer_info = p2p_instance.connect_to(["/ip4/.../tcp/4001/p2p/QmPeerID2"])
                success = p2p_instance.send_message_to_peer(
                    "QmPeerID1::dm:QmPeerID2",
                    b"Hello, peer!"
                )
                if success:
                    print("Message sent successfully.")
            except P2PError as e:
                print(f"Message sending failed: {e}")
            except ValueError as e:
                print(f"Invalid input: {e}")
            except TypeError as e:
                print(f"Data type error: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")
            ```
        """

        # Serialize the entire message object to bytes using Protobuf.
        payload_len = len(msg_bytes)
        peer_id = channel.split("::dm:")[1].split("-")[
            0
        ]  # Extract Peer ID from channel format

        # Call the Go function
        try:
            result_ptr = P2P.libp2p.SendMessageToPeer(
                P2P._type_interface.to_go_int(self._instance),
                P2P._type_interface.to_go_string(channel),
                P2P._type_interface.to_go_bytes(msg_bytes),  # Pass bytes directly
                P2P._type_interface.to_go_int(payload_len),
            )
            result = StringResponse.model_validate(
                P2P._type_interface.from_go_ptr_to_json(result_ptr)
            )
        except Exception as e:
            logger.error(f"❌ Sending direct message to {peer_id} failed: {e}")
            raise P2PError(f"Sending direct message to {peer_id} failed") from e  

        if result.state == ResponseState.ERROR:
            logger.error(
                f"Failed to send direct message to '{peer_id}': "
                f"{result.error}"
            )
            raise P2PError(
                f"Failed to send direct message to '{peer_id}': "
                f"{result.error}"
            )

        logger.info(f"✅ Successfully sent direct message to {peer_id[-5:]}.")
        return True
        
    @validate_call
    def broadcast_message(self, channel: str, msg_bytes: bytes) -> bool:
        """
        Broadcasts a message using PubSub to the node's own topic.
        Peers subscribed to this node's Peer ID topic will receive it.

        Args:
            channel: The Channel for this topic (e.g., owner_peer_id::ps:topic_name).
            msg_bytes: The message to send (bytes).

        Raises:
            P2PError: If broadcasting fails.
            ValueError: If inputs are invalid.
            TypeError: If data is not bytes.
        """

        # Serialize the entire message object to bytes using Protobuf.
        payload_len = len(msg_bytes)

        # Call SendMessageToPeer with an empty peer_id string for broadcast
        try:
            result_ptr = P2P.libp2p.SendMessageToPeer(
                P2P._type_interface.to_go_int(self._instance),
                P2P._type_interface.to_go_string(channel),
                P2P._type_interface.to_go_bytes(msg_bytes),
                P2P._type_interface.to_go_int(payload_len),
            )

            result = StringResponse.model_validate(
                P2P._type_interface.from_go_ptr_to_json(result_ptr)
            )
        except Exception as e:
            logger.error(f"❌ Broadcasting to {channel} failed: {e}")
            raise P2PError(f"Broadcasting to {channel} failed") from e

        if result.state == ResponseState.ERROR:
            logger.error(
                f"Failed to broadcast message on channel '{channel}': "
                f"{result.error}"
            )
            raise P2PError(
                f"Failed to broadcast message on channel '{channel}': "
                f"{result.error}"
            )

        logger.info(f"✅ Successfully broadcasted message on channel {channel}.")
        return True

    def pop_messages(self) -> List[bytes]:
        """
        Retrieves and removes the first message from the queue of each channel for this node instance.

        Returns:
            A list of byte arrays (messages). Returns an empty list if no messages were available.

        Raises:
            P2PError: If popping messages failed internally in Go, or if data
                      conversion fails for any message.
        """
        logger.debug(f"[Instance {self._instance}] Popping message(s)...")
        try:
            go_instance_c = P2P._type_interface.to_go_int(self._instance)

            result_ptr = P2P.libp2p.PopMessages(go_instance_c)

            # From_go_ptr_to_json should handle freeing result_ptr
            raw_result = P2P._type_interface.from_go_ptr_to_json(result_ptr)

            if raw_result is None:
                # This indicates an issue with the C call or JSON conversion in TypeInterface
                logger.error(
                    f"[Instance {self._instance}] PopMessages: "
                    f"Received null/invalid result from TypeInterface."
                )
                raise P2PError(
                    f"[Instance {self._instance}] PopMessages: Failed to get valid JSON response."
                )

            # Check for Go-side error or empty states first
            if isinstance(raw_result, dict):
                state = raw_result.get("state")
                if state == "Empty":
                    logger.debug(
                        f"[Instance {self._instance}] PopMessages: Queue is empty."
                    )
                    return []  # No messages available
                if state == "Error":
                    error_message = raw_result.get(
                        "message", "Unknown Go error during PopMessages"
                    )
                    logger.error(
                        f"[Instance {self._instance}] PopMessages: {error_message}"
                    )
                    raise P2PError(
                        f"[Instance {self._instance}] PopMessages: {error_message}"
                    )

                # If it's a dict but not a known state, it's unexpected
                logger.warning(
                    f"[Instance {self._instance}] PopMessages: Unexpected dictionary format: {raw_result}"
                )
                raise P2PError(
                    f"[Instance {self._instance}] PopMessages: Unexpected dictionary response format."
                )

            # Expecting a list of messages if not an error/empty dict
            if not isinstance(raw_result, list):
                # This also covers the case where n=0 and Go returns "[]" which json.loads makes a list
                # If it's not a list at this point, it's an unexpected format.
                logger.error(
                    f"[Instance {self._instance}] PopMessages: Unexpected response format, expected a list or "
                    f"specific state dictionary. Got: {type(raw_result)}"
                )
                raise P2PError(
                    f"[Instance {self._instance}] PopMessages: Unexpected response format."
                )

            return raw_result

        except P2PError:  # Re-raise P2PError directly
            raise
        except Exception as e:
            # Catch potential JSON parsing errors from TypeInterface or other unexpected errors
            logger.error(
                f"[Instance {self._instance}] ❌ Error during pop_message: {e}"
            )
            raise P2PError(
                f"[Instance {self._instance}] Unexpected error during pop_message: {e}"
            ) from e

    # --- PubSub Operations ---

    def subscribe_to_topic(self, channel: str) -> None:
        """
        Subscribes to a PubSub topic to receive messages.

        Args:
            channel: The Channel for this topic (e.g., owner_peer_id::ps:topic_name).

        Raises:
            P2PError: If subscribing fails.
            ValueError: If topic_name is invalid.
        """
        if not channel or not isinstance(channel, str):
            logger.error("Invalid topic name provided.")
            raise ValueError("Invalid topic name provided.")
        logger.info(f"<sub> Subscribing to topic: {channel}...")
        try:
            result_ptr = P2P.libp2p.SubscribeToTopic(
                P2P._type_interface.to_go_int(self._instance),
                P2P._type_interface.to_go_string(channel),
            )
            result = P2P._type_interface.from_go_ptr_to_json(result_ptr)

            if result is None:
                logger.error("Failed to subscribe to topic, received null result.")
                raise P2PError("Failed to subscribe to topic, received null result.")
            if result.get("state") == "Error":
                logger.error(
                    f"Failed to subscribe to topic '{channel}': {result.get('message', 'Unknown Go error')}"
                )
                raise P2PError(
                    f"Failed to subscribe to topic '{channel}': {result.get('message', 'Unknown Go error')}"
                )

            logger.info(f"✅ Successfully subscribed to {channel}")

        except Exception as e:
            logger.error(f"❌ Subscription to {channel} failed: {e}")
            raise P2PError(f"Subscription to {channel} failed") from e

    def unsubscribe_from_topic(self, channel: str) -> None:
        """
        Unsubscribes from a PubSub topic.

        Args:
            channel: The Channel for this topic (e.g., owner_peer_id::ps:topic_name).

        Raises:
            P2PError: If unsubscribing fails.
            ValueError: If topic_name is invalid.
        """
        if not channel or not isinstance(channel, str):
            logger.error("Invalid topic name provided.")
            raise ValueError("Invalid topic name provided.")
        logger.info(f"</sub> Unsubscribing from topic: {channel}...")
        try:
            result_ptr = P2P.libp2p.UnsubscribeFromTopic(
                P2P._type_interface.to_go_int(self._instance),
                P2P._type_interface.to_go_string(channel),
            )
            result = P2P._type_interface.from_go_ptr_to_json(result_ptr)

            if result is None:
                logger.error("Failed to unsubscribe from topic, received null result.")
                raise P2PError(
                    "Failed to unsubscribe from topic, received null result."
                )
            if result.get("state") == "Error":
                logger.error(
                    f"Failed to unsubscribe from topic '{channel}': "
                    f"{result.get('message', 'Unknown Go error')}"
                )
                raise P2PError(
                    f"Failed to unsubscribe from topic '{channel}': "
                    f"{result.get('message', 'Unknown Go error')}"
                )

            logger.info(f"✅ Successfully unsubscribed from {channel}")

        except Exception as e:
            logger.error(f"❌ Unsubscription from {channel} failed: {e}")
            raise P2PError(f"Unsubscription from {channel} failed") from e

    # --- Relay Operations ---
    def start_static_relay(self, relay_peer_id: str, relay_addrs: List[str]) -> None:
        """
        Enables (or switches to) a static AutoRelay service pointing to a specific relay node.
        This handles connection, reservation, and automatic renewal in the background.

        Args:
            relay_peer_id: The Peer ID of the relay node (subnetwork owner).
            relay_addrs: A list of multiaddresses for the relay node.

        Raises:
            P2PError: If the operation fails.
            ValueError: If inputs are invalid.
        """
        if not relay_peer_id or not isinstance(relay_peer_id, str):
            logger.error("Invalid relay Peer ID provided.")
            raise ValueError("Invalid relay Peer ID provided.")

        if not relay_addrs or not isinstance(relay_addrs, list):
            logger.error("Invalid relay addresses provided.")
            raise ValueError("Invalid relay addresses provided.")

        logger.info(f"🔗 Enabling Static AutoRelay via {relay_peer_id}...")

        # Construct the AddrInfo structure expected by Go's json.Unmarshal
        relay_info = {"ID": relay_peer_id, "Addrs": relay_addrs}

        try:
            result_ptr = P2P.libp2p.StartStaticRelay(
                P2P._type_interface.to_go_int(self._instance),
                P2P._type_interface.to_go_json(relay_info),
            )
            result = P2P._type_interface.from_go_ptr_to_json(result_ptr)

            if result is None:
                logger.error("Failed to enable static relay, received null result.")
                raise P2PError("Failed to enable static relay, received null result.")

            if result.get("state") == "Error":
                err_msg = result.get("message", "Unknown Go error")
                logger.error(f"Failed to enable static relay: {err_msg}")
                raise P2PError(f"Failed to enable static relay: {err_msg}")

            logger.info(
                f"✅ Static AutoRelay enabled successfully for {relay_peer_id}."
            )

        except Exception as e:
            logger.error(f"❌ Failed to enable static relay: {e}")
            raise P2PError(f"Failed to enable static relay: {e}") from e

    # --- Node Information ---

    @property
    def peer_id(self) -> Optional[str]:
        """Returns the Peer ID of the local node."""
        return self._peer_id

    @property
    def addresses(self) -> List[str]:
        """
        Returns the LIVE list of multiaddresses from the Go engine.
        Since Go caches this via events, this call is instant O(1).
        """
        try:
            return self.get_node_addresses()
        except P2PError as e:
            logger.warning(f"Failed to fetch addresses: {e}")
            return []

    @property
    def is_public(self) -> Optional[bool]:
        """Returns a boolean stating whether the local node is publicly reachable."""
        return self._is_public

    @property
    def relay_is_enabled(self) -> bool:
        """Returns whether the relay client functionality is enabled for this node."""
        return self._enable_relay_client

    def get_node_addresses(self, peer_id: str = "") -> List[str]:
        """
        Gets the known multiaddresses for the local node or a specific peer.

        Args:
            peer_id: The Peer ID string of the target peer. If empty, gets
                     addresses for the local node.

        Returns:
            A list of multiaddress strings (including the /p2p/PeerID suffix).

        Raises:
            P2PError: If fetching addresses fails.
        """
        target = "local node" if not peer_id else f"peer {peer_id}"
        logger.info(f"ℹ️ Fetching addresses for {target}...")
        try:
            result_ptr = P2P.libp2p.GetNodeAddresses(
                P2P._type_interface.to_go_int(self._instance),
                P2P._type_interface.to_go_string(peer_id),
            )
            result = P2P._type_interface.from_go_ptr_to_json(result_ptr)

            if result is None:
                logger.error("Failed to get node addresses, received null result.")
                raise P2PError("Failed to get node addresses, received null result.")
            if result.get("state") == "Error":
                logger.error(
                    f"Failed to get addresses for '{target}': {result.get('message', 'Unknown Go error')}"
                )
                raise P2PError(
                    f"Failed to get addresses for '{target}': {result.get('message', 'Unknown Go error')}"
                )

            addr_list = result.get("message", [])
            logger.info(f"✅ Found addresses for {target}: {addr_list}")
            return addr_list

        except Exception as e:
            logger.error(f"❌ Failed to get addresses for {target}: {e}")
            raise P2PError(f"Failed to get addresses for {target}") from e

    def get_connected_peers_info(self) -> List[Dict[str, Any]]:
        """
        Gets information about currently connected peers from the Go library.

        Returns:
            A list of dictionaries, each representing a connected peer with
            keys like 'addr_info' (containing 'ID', 'Addrs'), 'connected_at', 'direction', and 'misc'.

        Raises:
            P2PError: If fetching connected peers fails.
        """

        # Logger.info("ℹ️ Fetching connected peers info...") # Can be noisy
        try:
            # GetConnectedPeers takes no arguments in Go
            result_ptr = P2P.libp2p.GetConnectedPeers(
                P2P._type_interface.to_go_int(self._instance)
            )
            result = P2P._type_interface.from_go_ptr_to_json(result_ptr)

            if result is None:
                logger.error("Failed to get connected peers, received null result.")
                raise P2PError("Failed to get connected peers, received null result.")
            if result.get("state") == "Error":
                logger.error(
                    f"Failed to get connected peers: {result.get('message', 'Unknown Go error')}"
                )
                raise P2PError(
                    f"Failed to get connected peers: {result.get('message', 'Unknown Go error')}"
                )

            peers_list = result.get("message", [])

            # Update internal map (optional)
            # logger.info(f"  Connected peers count: {len(peers_list)}") # Can be noisy
            return peers_list

        except Exception as e:
            # Avoid crashing the polling thread, just log the error
            logger.error(f"❌ Error fetching connected peers info: {e}")

            # Optionally raise P2PError(f"Failed to get connected peers info") from e if called directly
            return []  # Return empty list on error during polling

    def get_rendezvous_peers_info(self) -> Dict[str, Any] | List | None:
        """
        Gets the full rendezvous state from the Go library, including peers and metadata.

        Returns:
            - A dictionary representing the RendezvousState (containing 'peers',
              'update_count', 'last_updated') if an update has been received.
            - None if no rendezvous topic is active or no updates have arrived yet.

        Raises:
            P2PError: If fetching the state fails in Go.
        """
        try:
            result_ptr = P2P.libp2p.GetRendezvousPeers(
                P2P._type_interface.to_go_int(self._instance)
            )
            result = P2P._type_interface.from_go_ptr_to_json(result_ptr)

            if result is None:
                logger.error("Failed to get rendezvous peers, received null result.")
                raise P2PError("Failed to get rendezvous peers, received null result.")

            state = result.get("state")
            if state == "Empty":
                logger.debug(
                    f"[Instance {self._instance}] GetRendezvousPeers: No rendezvous messages received yet."
                )
                return None  # Return None for the "empty" state
            elif state == "Error":
                error_msg = result.get("message", "Unknown Go error")
                logger.error(f"Failed to get rendezvous peers: {error_msg}")
                raise P2PError(f"Failed to get rendezvous peers: {error_msg}")
            elif state == "Success":
                # The message payload is the full RendezvousState object
                rendezvous_state = result.get("message", {})
                return rendezvous_state
            else:
                logger.error(
                    f"[Instance {self._instance}] GetRendezvousPeers: Received invalid state '{state}'."
                )
                raise P2PError(
                    f"[Instance {self._instance}] GetRendezvousPeers: Received invalid state."
                )

        except Exception as e:
            # Avoid crashing the polling thread, just log the error
            logger.error(f"❌ Error fetching rendezvous peers info: {e}")

            # Optionally raise P2PError(f"Failed to get rendezvous peers info") from e if called directly
            return []  # Return empty list on error during polling

    def get_message_queue_length(self) -> int:
        """
        Gets the current number of messages in the incoming queue.

        Returns:
            The number of messages waiting.

        Raises:
            P2PError: If querying the length fails (should be rare).
        """
        try:
            # Call Go function, returns C.int directly
            length_cint = P2P.libp2p.MessageQueueLength(
                P2P._type_interface.to_go_int(self._instance)
            )
            length = P2P._type_interface.from_go_int(length_cint)

            # Print(f"  Current Message Queue Len: {length}") # Can be noisy
            return length
        except Exception as e:
            # Avoid crashing polling thread
            logger.error(f"❌ Error fetching message queue length: {e}")
            return -1  # Indicate error

    # --- Lifecycle Management ---

    def close(self, close_all: bool = False) -> None | str:
        """
        Gracefully shuts down the libp2p node and stops background threads.

        Args:
            close_all: If True, closes all instances of the node. Default is False.
        """
        logger.info("🛑 Closing node...")

        # 1. Signal background threads to stop
        logger.info("  - Stopping background threads...")

        # 2. Wait briefly for threads to finish (optional, they are daemons)
        # self._get_connected_peers_thread.join(timeout=2)
        # self._check_message_queue_thread.join(timeout=2)
        # print("  - Background threads signaled.")

        # 3. Call the Go CloseNode function
        try:
            if close_all:
                result_ptr = P2P.libp2p.CloseNode(P2P._type_interface.to_go_int(-1))
            else:
                result_ptr = P2P.libp2p.CloseNode(
                    P2P._type_interface.to_go_int(self._instance)
                )
            result = P2P._type_interface.from_go_ptr_to_json(result_ptr)

            if result is None:
                logger.error("Node closure failed: received null result.")
                raise P2PError("Node closure failed: received null result.")
            if result.get("state") == "Error":
                logger.error(
                    f"Node closure failed: {result.get('message', 'Unknown Go error')}"
                )
                raise P2PError(
                    f"Node closure failed: {result.get('message', 'Unknown Go error')}"
                )

            close_msg = (
                f"Node closed successfully "
                f"({'all instances' if close_all else f'instance {str(self._instance)}'})."
            )
            logger.info(f"✅ {close_msg}")

        except Exception as e:
            logger.error(f"❌ Error closing node: {e}")
            raise P2PError(f"Error closing node: {e}") from e

        # 4. Clear internal state
        self._peer_id = None
        with P2P._instance_lock:
            if close_all:
                # Also apply the lock here and use the corrected logic
                P2P._instance_ids = [False] * P2P._MAX_INSTANCES
                logger.info("🐍 All instance slots have been marked as free.")
            else:
                if self._instance != -1:  # Ensure instance was set
                    P2P._instance_ids[self._instance] = False
                    logger.info(
                        f"🐍 Instance slot {self._instance} has been marked as free."
                    )

        logger.info("🐍 Python P2P object state cleared.")

        return close_msg

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager, ensuring node closure."""
        self.close()


class ResponseState(str, Enum):
    """Represents the 'state' field in the JSON response.

    Attributes:
        SUCCESS: Indicates a successful operation with data in 'message'.
        ERROR: Indicates an error occurred, with error details in 'message'.
        EMPTY: Indicates a successful operation but no data to return (message is None).
    """

    SUCCESS = "Success"
    ERROR = "Error"
    EMPTY = "Empty"


class NodeConfigResult(BaseModel):
    """Payload for CreateNode.

    Attributes:
    addresses: List of multiaddresses the node is listening on.
    is_public: Whether the node is publicly reachable (maps to Go's `isPublic` JSON tag).
    """

    addresses: List[str]
    is_public: bool = Field(..., alias="isPublic")


class PeerAddrInfo(BaseModel):
    """Payload for ConnectTo and general peer info.

    Attributes:
        id: The peer's unique identifier.
        addrs: List of multiaddresses associated with the peer.
    """

    id: str
    addrs: List[str]


class ExtendedPeerInfo(BaseModel):
    """Payload for GetConnectedPeers and Rendezvous.

    Attributes:
        id: The peer's unique identifier.
        addrs: List of multiaddresses associated with the peer.
        connected_at: Timestamp of when the connection was established (maps to Go's `connected_at` JSON tag).
        direction: Connection direction ("inbound" or "outbound").
        misc: Additional metadata or flags as an integer (maps to Go's `misc` JSON tag).
        relayed: Whether the connection is relayed through a relay node (maps to Go's `relayed` JSON tag).
    """

    id: str
    addrs: List[str]
    connected_at: datetime.datetime = Field(..., alias="connected_at")
    direction: str
    misc: int
    relayed: bool

    @field_validator("direction")
    def validate_direction(cls, v):
        if v not in ("inbound", "outbound"):
            raise ValueError("Direction must be 'inbound' or 'outbound'")
        return v

    @field_validator("connected_at", mode="before")
    def parse_connected_at(cls, v):
        if isinstance(v, str):
            try:
                return datetime.datetime.fromisoformat(v)
            except ValueError as e:
                raise ValueError(
                    f"Invalid datetime format for connected_at: {v}"
                ) from e
        elif isinstance(v, (int, float)):
            # If it's a timestamp in seconds, convert to datetime
            return datetime.datetime.fromtimestamp(v)
        elif isinstance(v, datetime.datetime):
            return v
        else:
            raise ValueError(f"Unsupported type for connected_at: {type(v)}")


class RendezvousState(BaseModel):
    """Payload for GetRendezvousPeers.

    Attributes:
        peers: List of extended peer information.
        update_count: Number of updates (maps to Go's `update_count` JSON tag).
    """

    peers: List[ExtendedPeerInfo]
    update_count: int = Field(..., alias="update_count")


class IncomingMessage(BaseModel):
    """Payload for PopMessages items.

    Attributes:
        from_peer: The Peer ID of the sender (maps to Go's `from` JSON tag).
        data_b64: The message data encoded as a base64 string (maps to Go's `data` JSON tag).
    """

    from_peer: str = Field(..., alias="from")
    data_b64: str = Field(..., alias="data")
    data: bytes = Field(default=b"", description="Decoded message data as bytes")

    @model_validator(mode="after")
    def decoded_data(self) -> Self:
        """Helper to get raw bytes from the base64 string."""
        self.data = base64.b64decode(self.data_b64)
        return self


T = TypeVar("T")


class LibP2PResponse(BaseModel, Generic[T]):
    """
    Represents the standard JSON wrapper returned by the GO library.

    Structure: {"state": "...", "message": ...}

    Attributes:
        state: The state of the response, indicating success, error, or empty.
        message: The payload of the response, which can be of type T if successful, a string error message if an error occurred, or None if the response is empty.

    """

    state: ResponseState
    data: T | None = Field(
        ...,
        alias="message",
        description="The payload of the response, which can be of type T if successful, a string error message if an error occurred, or None if the response is empty.",
    )
    error: Optional[str] = Field(
        None, description="The error message if the state is 'Error'."
    )

    @property
    def is_success(self) -> bool:
        return self.state == ResponseState.SUCCESS

    @property
    def is_empty(self) -> bool:
        return self.state == ResponseState.EMPTY

    @model_validator(mode="before")
    def validate_message(cls, values):

        if values is None:
            raise ValueError("P2P response cannot be null!")

        state = values.get("state")
        data = values.get("data")

        if state == ResponseState.SUCCESS:
            if data is None:
                raise ValueError("Success response must include a data payload.")
        elif state == ResponseState.ERROR:
            if not isinstance(data, str):
                raise ValueError("Error response must include a string error message.")
        elif state == ResponseState.EMPTY:
            if data is not None:
                raise ValueError("Empty response must have data set to None.")
        else:
            raise ValueError(f"Invalid state: {state}")
        return values

    @model_validator(mode="after")
    def unwrap(self) -> Self:
        """
        Returns the data if success, raises ValueError if error/empty.
        """
        if self.state == ResponseState.SUCCESS:
            return self
        elif self.state == ResponseState.ERROR:
            if not isinstance(self.data, str):
                raise ValueError("Error response must include a string error message.")
            raise ValueError(f"Error response: {self.error}")
        elif self.state == ResponseState.EMPTY:
            self.data = None
        return self


# CreateNode
# Returns: {"state": "Success", "message": {"addresses": [...], "isPublic": bool}}
CreateNodeResponse = LibP2PResponse[NodeConfigResult]

# ConnectTo
# Returns: {"state": "Success", "message": {"id": "...", "addrs": [...]}}
ConnectToResponse = LibP2PResponse[PeerAddrInfo]

# GetConnectedPeers
# Returns: {"state": "Success", "message": [ExtendedPeerInfo, ...]}
ConnectedPeersResponse = LibP2PResponse[List[ExtendedPeerInfo]]

# GetRendezvousPeers
# Returns: {"state": "Success", "message": {"peers": [...], "update_count": 1}}
# Or: {"state": "Empty"}
RendezvousPeersResponse = LibP2PResponse[RendezvousState]

# GetNodeAddresses
# Returns: {"state": "Success", "message": ["/ip4/...", ...]}
NodeAddressesResponse = LibP2PResponse[List[str]]

# PopMessages
# Returns: {"state": "Success", "message": [{"from": "...", "data": "..."}]}
# Or: {"state": "Empty"}
PopMessagesResponse = LibP2PResponse[List[IncomingMessage]]

# Standard String Responses
# Used by: DisconnectFrom, StartStaticRelay, SendMessageToPeer,
# SubscribeToTopic, UnsubscribeFromTopic, CloseNode
# Returns: {"state": "Success", "message": "Some confirmation string"}
StringResponse = LibP2PResponse[str]
