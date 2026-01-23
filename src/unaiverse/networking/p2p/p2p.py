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
import os
import logging
import threading
from .lib_types import TypeInterface
from typing import Optional, List, Dict, Any, TYPE_CHECKING

# Conditional import for type hinting to avoid circular dependencies
if TYPE_CHECKING:
    try:
        from unaiverse.networking.p2p.golibp2p import GoLibP2P  # Assuming this class loads the library
    except ImportError:
        pass

logger = logging.getLogger('P2P')


class P2PError(Exception):
    """Custom exception class for P2P library errors."""
    pass


class P2P:
    """
    Python wrapper for the Go libp2p shared library.

    This class initializes a libp2p node, provides methods to interact with the
    p2p network (connect, send/receive messages, pubsub, relay), and manages
    the lifecycle of the underlying Go node.

    Attributes:
        libp2p (LibP2P): Static class attribute holding the loaded Go library instance.
                         Must be set before instantiating P2P. Example: P2P.libp2p = LibP2P()
    Properties:
        peer_id (str): The Peer ID of the initialized local node.
        addresses (Optional[List[str]]): List of multiaddresses the local node is listening on.
        is_public (bool): Whether the node is publicly reachable.
        peer_map (Dict[str, Any]): A dictionary to potentially store information about connected peers
            (managed manually or by polling thread).
    """

    # --- Class-level state ---
    libp2p: 'GoLibP2P'  # Static variable for the loaded Go library
    _type_interface: 'TypeInterface'  # Shared type interface for all instances

    # --- Config class variables for configuration ---
    _MAX_INSTANCES = 32
    _MAX_NUM_CAHNNELS = 100
    _MAX_QUEUE_PER_CHANNEL = 50
    _MAX_MESSAGE_SIZE = 50 * 1024 * 1024  # 50 MB

    # --- Class-level tracking ---
    _library_initialized = False
    _initialize_lock = threading.Lock()
    _instance_ids = [False, ] * _MAX_INSTANCES
    _instance_lock = threading.Lock()

    @classmethod
    def setup_library(cls,
                      max_instances: Optional[int] = None,
                      max_channels: Optional[int] = None,
                      max_queue_per_channel: Optional[int] = None,
                      max_message_size: Optional[int] = None,
                      enable_logging: bool = False) -> None:
        """
        Initializes the underlying Go library. Must be called once. This is called automatically.
        """
        with cls._initialize_lock:
            if cls._library_initialized:
                logger.warning("P2P library is already initialized. Skipping setup.")
                return

            if not hasattr(cls, 'libp2p') or cls.libp2p is None:
                raise P2PError("Library not loaded before setup. Check package __init__.py")

            # Configure Python logging based on the flag
            if not enable_logging:
                logger.setLevel(logging.CRITICAL)
                _log_config = {}
            else:
                logger.setLevel(logging.INFO)
                _log_config = {
                    'net/identify': 'debug',
                    'unailib': 'debug',
                    # 'autotls': 'debug',
                    # 'p2p-forge': 'debug',
                    'nat': 'debug',
                    'basichost': 'debug',
                    'p2p-circuit': 'debug',
                    'relay': 'debug',
                    'p2p-holepunch': 'debug',
                    'tcp-tpt': 'debug',
                    'connmgr': 'debug',
                    'dht': 'debug',
                    'autorelay': 'debug',
                    'autonat': 'debug',
                    # 'rcmgr': 'debug',
                    'swarm2': 'debug',
                    'yamux': 'debug'
                }

            logger.info("🐍 Setting up and initializing P2P library core with user settings...")
            cls._type_interface = TypeInterface(cls.libp2p)

            # Use provided arguments or fall back to class defaults
            _max_instances = max_instances if max_instances is not None else cls._MAX_INSTANCES
            _max_channels = max_channels if max_channels is not None else cls._MAX_NUM_CAHNNELS
            _max_queue = max_queue_per_channel if max_queue_per_channel is not None else cls._MAX_QUEUE_PER_CHANNEL
            _max_msg_size = max_message_size if max_message_size is not None else cls._MAX_MESSAGE_SIZE

            # Update class attributes if they were overridden
            cls._MAX_INSTANCES = _max_instances
            cls._instance_ids = [False, ] * _max_instances  # Resize the tracking list

            # Call the Go function to set up its internal state
            logger.info("🐍 Initializing Go library core...")
            cls.libp2p.InitializeLibrary(
                cls._type_interface.to_go_int(_max_instances),
                cls._type_interface.to_go_int(_max_channels),
                cls._type_interface.to_go_int(_max_queue),
                cls._type_interface.to_go_int(_max_msg_size),
                cls._type_interface.to_go_json(_log_config)
            )

            cls._library_initialized = True
            logger.info("✅ Go library initialized successfully.")

    def __init__(self,
                 identity_dir: str,
                 port: int = 0,
                 ips: List[str] = None,
                 enable_relay_client: bool = True,
                 enable_relay_service: bool = False,
                 use_broad_limits: bool = False,
                 is_isolated: bool = False,
                 knows_is_public: bool = False,
                 enable_tls: bool = False,
                 domain_name: Optional[str] = None,
                 tls_cert_path: Optional[str] = None,
                 tls_key_path: Optional[str] = None,
                 dht_enabled: bool = False,
                 dht_keep: bool = True
                 ) -> None:
        """
        Initializes and starts a new libp2p node.

        Args:
            identity_dir: Directory path to load/store the node's private key and certificates.
            port: The (first) TCP port to listen on (0 for random).
            ips: A list of specific IP addresses to listen on. Defaults to ["0.0.0.0"].
            enable_relay_client: Enable listening to relayed connections for this node.
            enable_relay_service: Enable relay service capabilities for this node.
            knows_is_public: If you already know that the node is public this forces its public reachability.
                Otherwise, it tries every possible attempt to make the node publicly reachable (UPnP, HolePunching,
                AutoNat via DHT...).
            enable_tls: Whether to enable AutoTLS certificate management (requires internet access).
            domain_name: Optional domain name for TLS certificate (required if enable_tls is True).
            tls_cert_path: Optional path to a custom TLS certificate file (PEM format).
            tls_key_path: Optional path to a custom TLS private key file (PEM format).

        Raises:
            P2PError: If the node creation fails in the Go library.
            AttributeError: If P2P.libp2p has not been set before instantiation.
        """

        # --- CRITICAL: Check if library is initialized ---
        if not P2P._library_initialized:
            raise P2PError("P2P library not set up. Call P2P.setup_library() before creating an instance.")

        # Assign instance ID
        assigned_instance_id = -1
        with P2P._instance_lock:
            for _instance_id, i in enumerate(self._instance_ids):
                if not i:
                    self._instance_ids[_instance_id] = True
                    assigned_instance_id = _instance_id
                    break
            if assigned_instance_id == -1:
                raise P2PError(
                    f"Cannot create new P2P instance: Maximum number of instances "
                    f"({P2P._MAX_INSTANCES})."
                )

        self._instance: int = assigned_instance_id
        logger.info(f"🚀 Attempting to initialize P2P Node with auto-assigned Instance ID: {self._instance}")
        
        os.makedirs(identity_dir, exist_ok=True)

        self._enable_relay_client = enable_relay_client or enable_relay_service
        self._peer_id: Optional[str] = None

        # TLS Validation Logic
        has_custom_tls_args = (tls_cert_path is not None) or (tls_key_path is not None) or (domain_name is not None)
        if has_custom_tls_args:
            if domain_name is None or tls_cert_path is None or tls_key_path is None:
                raise ValueError("Custom TLS requires 'domain_name', 'tls_cert_path' and 'tls_key_path'.")

        use_auto_tls = enable_tls and not has_custom_tls_args
        
        # --- Build Configuration JSON ---
        node_config = {
            "identity_dir": identity_dir,
            "predefined_port": port,
            "listen_ips": ips,
            "relay": {
                "enable_client": self._enable_relay_client,
                "enable_service": enable_relay_service,
                "with_broad_limits": use_broad_limits,
            },
            "tls": {
                "auto_tls": use_auto_tls,
                "domain": domain_name if domain_name else "",
                "cert_path": tls_cert_path if tls_cert_path else "",
                "key_path": tls_key_path if tls_key_path else "",
            },
            "network": {
                "isolated": is_isolated,
                "force_public": knows_is_public,
            },
            "dht": {
                "enabled": dht_enabled,
                "keep": dht_keep and dht_enabled,
            }
        }

        logger.info(f"🐍 Creating Node (Instance ID: {self._instance})...")
        try:

            # Call the Go function
            result_ptr = P2P.libp2p.CreateNode(
                P2P._type_interface.to_go_int(self._instance),
                P2P._type_interface.to_go_json(node_config),
            )
            result = P2P._type_interface.from_go_ptr_to_json(result_ptr)

            if result is None:
                err_msg = "Received null result from Go CreateNode."
                logger.error(f"[Instance {self._instance}] {err_msg}")
                raise P2PError(f"[Instance {self._instance}] {err_msg}")
            if result.get('state') == "Error":
                err_msg = result.get('message', 'Unknown Go error on CreateNode')
                logger.error(f"[Instance {self._instance}] Go error: {err_msg}")
                raise P2PError(f"[Instance {self._instance}] Failed to create node: {err_msg}")

            message_data = result.get('message')
            initial_addresses = message_data.get("addresses", [])
            self._is_public = message_data.get("isPublic", False)

            # Check the returned data
            if not isinstance(initial_addresses, list):
                err_msg = "Received invalid addresses list from Go CreateNode."
                logger.error(f"[Instance {self._instance}] {err_msg}")
                raise P2PError(f"[Instance {self._instance}] {err_msg}")
            elif not initial_addresses:
                err_msg = "Received empty addresses list from Go CreateNode."
                logger.error(f"[Instance {self._instance}] {err_msg}")

            self._peer_id = initial_addresses[0].split("/")[-1]

            logger.info(f"✅ [Instance {self._instance}] Node created with ID: {self._peer_id}")
            logger.info(f"👂 [Instance {self._instance}] Listening on: {initial_addresses}")
            logger.info(f"🌐 [Instance {self._instance}] Publicly reachable: {self._is_public}")

            logger.info(f"🎉 [Instance {self._instance}] Node initialized successfully.")

        except Exception as e:
            logger.error(f"❌ [Instance {self._instance}] Node creation failed: {e}")

            # Reclaim the instance ID using the _instance_ids list
            if self._instance != -1:  # Check if an ID was actually assigned
                with P2P._instance_lock:
                    P2P._instance_ids[self._instance] = False
                    logger.info(f"[Instance {self._instance}] "
                                f"Reclaimed instance ID {self._instance} due to creation failure.")
            raise  # Re-raise the exception that caused the failure

        logger.info("🎉 Node created successfully and background polling started.")

    # --- Core P2P Operations ---

    def connect_to(self, multiaddrs: list[str]) -> Dict[str, Any]:
        """
        Establishes a connection with a remote peer.

        Args:
            multiaddrs: The list of multiaddress strings of the peer to try to connect to.

        Returns:
            A dictionary containing the connected peer's AddrInfo (ID and Addrs).

        Raises:
            P2PError: If the connection fails.
            ValueError: If the multiaddr is invalid.
        """
        if not multiaddrs or not isinstance(multiaddrs, list):
            logger.error("Invalid multiaddr provided.")
            raise ValueError("Invalid multiaddr provided.")
        dest_peer_id = multiaddrs[0].split('/')[-1]
        logger.info(f"📞 Attempting to connect to: {dest_peer_id}...")
        try:
            result_ptr = P2P.libp2p.ConnectTo(
                P2P._type_interface.to_go_int(self._instance),
                P2P._type_interface.to_go_json(multiaddrs)
                )
            result = P2P._type_interface.from_go_ptr_to_json(result_ptr)

            if result is None:
                logger.error("Failed to connect to peer, received null result.")
                raise P2PError("Failed to connect to peer, received null result.")
            if result.get('state') == "Error":
                logger.error(f"Failed to connect to peer '{dest_peer_id}': {result.get('message', 'Unknown Go error')}")
                raise P2PError(f"Failed to connect to peer '{dest_peer_id}': "
                               f"{result.get('message', 'Unknown Go error')}")

            peer_info = result.get('message', {})
            logger.info(f"✅ Connection initiated to peer: {peer_info.get('ID', dest_peer_id)}")  # Use ID if available

            return peer_info

        except Exception as e:
            logger.error(f"❌ Connection to {dest_peer_id} failed: {e}")
            raise P2PError(f"Connection to {dest_peer_id} failed") from e

    def disconnect_from(self, peer_id: str) -> None:
        """
        Closes connections to a specific peer and removes tracking.

        Args:
            peer_id: The Peer ID string of the peer to disconnect from.

        Raises:
            P2PError: If disconnecting fails.
            ValueError: If the peer_id is invalid.
        """
        if not peer_id or not isinstance(peer_id, str):
            logger.error("Invalid Peer ID provided.")
            raise ValueError("Invalid Peer ID provided.")

        # Basic peer ID format check (Qm... or 12D3...)
        if not (peer_id.startswith("Qm") or peer_id.startswith("12D3")):
            logger.warning(f"⚠️ Warning: Peer ID '{peer_id}' does not look like a standard v0 or v1 ID.")

        logger.info(f"🔌 Attempting to disconnect from peer: {peer_id}...")
        try:
            result_ptr = P2P.libp2p.DisconnectFrom(
                P2P._type_interface.to_go_int(self._instance),
                P2P._type_interface.to_go_string(peer_id)
                )
            result = P2P._type_interface.from_go_ptr_to_json(result_ptr)

            if result is None:
                logger.error("Failed to disconnect from peer, received null result.")
                raise P2PError("Failed to disconnect from peer, received null result.")

            if result.get('state') == "Error":
                logger.error(f"Failed to disconnect from peer '{peer_id}': {result.get('message', 'Unknown Go error')}")
                raise P2PError(f"Failed to disconnect from peer '{peer_id}': "
                               f"{result.get('message', 'Unknown Go error')}")

            logger.info(f"✅ Successfully disconnected from {peer_id}")

        except Exception as e:
            logger.error(f"❌ Disconnection from {peer_id} failed: {e}")
            raise P2PError(f"Disconnection from {peer_id} failed") from e

    def send_message_to_peer(self, channel: str, msg_bytes: bytes) -> None:
        """
        Sends a direct message to a specific peer.

        Args:
            channel: The string identifying the channel for the communication.
            msg_bytes: The message to send (bytes).

        Raises:
            P2PError: If message sending fails (based on return code).
            ValueError: If inputs are invalid.
            TypeError: If data is not bytes.
        """
        if not channel or not isinstance(channel, str):
            logger.error("Invalid channel provided.")
            raise ValueError("Invalid channel provided.")

        # Serialize the entire message object to bytes using Protobuf.
        payload_len = len(msg_bytes)
        peer_id = channel.split("::dm:")[1].split('-')[0]  # Extract Peer ID from channel format

        # Call the Go function
        try:
            result_ptr = P2P.libp2p.SendMessageToPeer(
                P2P._type_interface.to_go_int(self._instance),
                P2P._type_interface.to_go_string(channel),
                P2P._type_interface.to_go_bytes(msg_bytes),  # Pass bytes directly
                P2P._type_interface.to_go_int(payload_len),
            )
            result = P2P._type_interface.from_go_ptr_to_json(result_ptr)

            if result is None:
                logger.error(f"Failed to send direct message to {peer_id}, received null result.")
                raise P2PError(f"Failed to send direct message to {peer_id}, received null result.")

            if result.get('state') == "Error":
                logger.error(f"Failed to send direct message to '{peer_id}': "
                             f"{result.get('message', 'Unknown Go error')}")
                raise P2PError(f"Failed to send direct message to '{peer_id}': "
                               f"{result.get('message', 'Unknown Go error')}")

            logger.info(f"✅ Successfully sent direct message to {peer_id[-5:]}.")

        except Exception as e:
            logger.error(f"❌ Sending direct message to {peer_id} failed: {e}")
            raise P2PError(f"Sending direct message to {peer_id} failed") from e

    def broadcast_message(self, channel: str, msg_bytes: bytes) -> None:
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
        if not channel or not isinstance(channel, str):
            raise ValueError("Invalid channel provided.")

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

            result = P2P._type_interface.from_go_ptr_to_json(result_ptr)

            if result is None:
                logger.error(f"Failed to broadcast message on channel {channel}, received null result.")
                raise P2PError(f"Failed to broadcast message on channel {channel}, received null result.")

            if result.get('state') == "Error":
                logger.error(f"Failed to broadcast message on channel '{channel}': "
                             f"{result.get('message', 'Unknown Go error')}")
                raise P2PError(f"Failed to broadcast message on channel '{channel}': "
                               f"{result.get('message', 'Unknown Go error')}")

        except Exception as e:
            logger.error(f"❌ Broadcasting to {channel} failed: {e}")
            raise P2PError(f"Broadcasting to {channel} failed") from e

        logger.info(f"✅ Successfully broadcasted message on channel {channel}.")

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
                logger.error(f"[Instance {self._instance}] PopMessages: "
                             f"Received null/invalid result from TypeInterface.")
                raise P2PError(f"[Instance {self._instance}] PopMessages: Failed to get valid JSON response.")

            # Check for Go-side error or empty states first
            if isinstance(raw_result, dict):
                state = raw_result.get('state')
                if state == "Empty":
                    logger.debug(f"[Instance {self._instance}] PopMessages: Queue is empty.")
                    return []  # No messages available
                if state == "Error":
                    error_message = raw_result.get('message', 'Unknown Go error during PopMessages')
                    logger.error(f"[Instance {self._instance}] PopMessages: {error_message}")
                    raise P2PError(f"[Instance {self._instance}] PopMessages: {error_message}")

                # If it's a dict but not a known state, it's unexpected
                logger.warning(f"[Instance {self._instance}] PopMessages: Unexpected dictionary format: {raw_result}")
                raise P2PError(f"[Instance {self._instance}] PopMessages: Unexpected dictionary response format.")

            # Expecting a list of messages if not an error/empty dict
            if not isinstance(raw_result, list):
                # This also covers the case where n=0 and Go returns "[]" which json.loads makes a list
                # If it's not a list at this point, it's an unexpected format.
                logger.error(f"[Instance {self._instance}] PopMessages: Unexpected response format, expected a list or "
                             f"specific state dictionary. Got: {type(raw_result)}")
                raise P2PError(f"[Instance {self._instance}] PopMessages: Unexpected response format.")

            return raw_result

        except P2PError:  # Re-raise P2PError directly
            raise
        except Exception as e:

            # Catch potential JSON parsing errors from TypeInterface or other unexpected errors
            logger.error(f"[Instance {self._instance}] ❌ Error during pop_message: {e}")
            raise P2PError(f"[Instance {self._instance}] Unexpected error during pop_message: {e}") from e

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
                P2P._type_interface.to_go_string(channel)
                )
            result = P2P._type_interface.from_go_ptr_to_json(result_ptr)

            if result is None:
                logger.error("Failed to subscribe to topic, received null result.")
                raise P2PError("Failed to subscribe to topic, received null result.")
            if result.get('state') == "Error":
                logger.error(f"Failed to subscribe to topic '{channel}': {result.get('message', 'Unknown Go error')}")
                raise P2PError(f"Failed to subscribe to topic '{channel}': {result.get('message', 'Unknown Go error')}")

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
                P2P._type_interface.to_go_string(channel)
                )
            result = P2P._type_interface.from_go_ptr_to_json(result_ptr)

            if result is None:
                logger.error("Failed to unsubscribe from topic, received null result.")
                raise P2PError("Failed to unsubscribe from topic, received null result.")
            if result.get('state') == "Error":
                logger.error(f"Failed to unsubscribe from topic '{channel}': "
                             f"{result.get('message', 'Unknown Go error')}")
                raise P2PError(f"Failed to unsubscribe from topic '{channel}': "
                               f"{result.get('message', 'Unknown Go error')}")

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
        relay_info = {
            "ID": relay_peer_id,
            "Addrs": relay_addrs
        }

        try:
            result_ptr = P2P.libp2p.StartStaticRelay(
                P2P._type_interface.to_go_int(self._instance),
                P2P._type_interface.to_go_json(relay_info)
            )
            result = P2P._type_interface.from_go_ptr_to_json(result_ptr)

            if result is None:
                logger.error("Failed to enable static relay, received null result.")
                raise P2PError("Failed to enable static relay, received null result.")
            
            if result.get('state') == "Error":
                err_msg = result.get('message', 'Unknown Go error')
                logger.error(f"Failed to enable static relay: {err_msg}")
                raise P2PError(f"Failed to enable static relay: {err_msg}")

            logger.info(f"✅ Static AutoRelay enabled successfully for {relay_peer_id}.")

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
                P2P._type_interface.to_go_string(peer_id)
                )
            result = P2P._type_interface.from_go_ptr_to_json(result_ptr)

            if result is None:
                logger.error("Failed to get node addresses, received null result.")
                raise P2PError("Failed to get node addresses, received null result.")
            if result.get('state') == "Error":
                logger.error(f"Failed to get addresses for '{target}': {result.get('message', 'Unknown Go error')}")
                raise P2PError(f"Failed to get addresses for '{target}': {result.get('message', 'Unknown Go error')}")

            addr_list = result.get('message', [])
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
            result_ptr = P2P.libp2p.GetConnectedPeers(P2P._type_interface.to_go_int(self._instance))
            result = P2P._type_interface.from_go_ptr_to_json(result_ptr)

            if result is None:
                logger.error("Failed to get connected peers, received null result.")
                raise P2PError("Failed to get connected peers, received null result.")
            if result.get('state') == "Error":
                logger.error(f"Failed to get connected peers: {result.get('message', 'Unknown Go error')}")
                raise P2PError(f"Failed to get connected peers: {result.get('message', 'Unknown Go error')}")

            peers_list = result.get('message', [])

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
            result_ptr = P2P.libp2p.GetRendezvousPeers(P2P._type_interface.to_go_int(self._instance))
            result = P2P._type_interface.from_go_ptr_to_json(result_ptr)

            if result is None:
                logger.error("Failed to get rendezvous peers, received null result.")
                raise P2PError("Failed to get rendezvous peers, received null result.")

            state = result.get('state')
            if state == "Empty":
                logger.debug(f"[Instance {self._instance}] GetRendezvousPeers: No rendezvous messages received yet.")
                return None  # Return None for the "empty" state
            elif state == "Error":
                error_msg = result.get('message', 'Unknown Go error')
                logger.error(f"Failed to get rendezvous peers: {error_msg}")
                raise P2PError(f"Failed to get rendezvous peers: {error_msg}")
            elif state == "Success":

                # The message payload is the full RendezvousState object
                rendezvous_state = result.get('message', {})
                return rendezvous_state
            else:
                logger.error(f"[Instance {self._instance}] GetRendezvousPeers: Received invalid state '{state}'.")
                raise P2PError(f"[Instance {self._instance}] GetRendezvousPeers: Received invalid state.")

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
            length_cint = P2P.libp2p.MessageQueueLength(P2P._type_interface.to_go_int(self._instance))
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
                result_ptr = P2P.libp2p.CloseNode(P2P._type_interface.to_go_int(self._instance))
            result = P2P._type_interface.from_go_ptr_to_json(result_ptr)

            if result is None:
                logger.error("Node closure failed: received null result.")
                raise P2PError("Node closure failed: received null result.")
            if result.get('state') == "Error":
                logger.error(f"Node closure failed: {result.get('message', 'Unknown Go error')}")
                raise P2PError(f"Node closure failed: {result.get('message', 'Unknown Go error')}")

            close_msg = (f"Node closed successfully "
                         f"({'all instances' if close_all else f'instance {str(self._instance)}'}).")
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
                    logger.info(f"🐍 Instance slot {self._instance} has been marked as free.")

        logger.info("🐍 Python P2P object state cleared.")

        return close_msg

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager, ensuring node closure."""
        self.close()
