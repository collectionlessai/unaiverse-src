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
import base64
import asyncio
import logging
import threading
from unaiverse.custom import Custom
from .lib_types import TypeInterface
from typing import Any, TYPE_CHECKING
from unaiverse.networking.p2p.messages import Msg
from unaiverse.networking.p2p import files as p2p_files
from unaiverse.networking.session.sess_msg import SessMsg, SESSION_PROTOCOL_IDS, SessionInbound, CAP_FILE_STREAM

# The session-layer wire codec lives in the leaf module sess_msg (no FSM, no cycle back into the conn layer):
# P2P.get_messages decodes inbound Session frames with it (SessMsg, imported above)
# FdCapture and Logger are optional — gracefully degrade if not present
try:
    from unaiverse.utils.os_fd_capture import FdCapture
    _CAPTURE_AVAILABLE = True
except ImportError:
    _CAPTURE_AVAILABLE = False

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


class P2PNoFileStream(P2PError):
    """A file message refused because its peer does not take file messages: the caller sends the files in memory."""


class P2PQueueFull(P2PError):
    """A direct message refused because the send queue of its peer is full (the expected answer to a sender that
    goes faster than the peer drains, not a fault): nothing queued before it was touched.

    Callers tell it apart by its queue_full attribute rather than by its class: the browser runtime mocks this module
    with a bare P2PError, so code that also runs there (connpool.py) must not import this class.
    """

    queue_full = True

    def __init__(self, message: str, peer_id: str) -> None:
        """Initializes the error.

        Args:
            message: The text of the refusal, as Go gives it (it names the cap that was hit).
            peer_id: The peer whose queue is full.
        """
        super().__init__(message)
        self.peer_id = peer_id


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
    _MAX_MESSAGE_SIZE = Custom.LEGACY_MAX_MESSAGE_SIZE  # One frame (pubsub, Session payloads, peers without chunks)

    # --- Class-level tracking ---
    _library_initialized = False
    _send_queues_enabled = False  # True once Go accepted the send queue limits of Custom (otherwise sends are sync)
    supports_split_pop = False  # True when the library pops streams apart from the rest (PopNetworkMessages)
    supports_chunked_frames = False  # True when Go sends/receives messages larger than a chunk (CAP_CHUNKED_FRAMES)
    supports_file_stream = False  # True when Go streams FILE data from disk (file messages; Phase 6.5)
    accepts_file_stream = False  # True when this node also takes them (Custom.ACCEPT_FILES): CAP_FILE_STREAM
    supports_stop_sending = False  # True when Go stops one message its receiver no longer wants: CAP_STOP_SENDING
    _initialize_lock = threading.Lock()
    _instance_ids = [False, ] * _MAX_INSTANCES
    _instance_lock = threading.Lock()

    # --- Go output capture (set by setup_library when logger is provided) ---
    _cap_stdout: object = None   # FdCapture instance for fd 1
    _cap_stderr: object = None   # FdCapture instance for fd 2

    @classmethod
    def setup_library(cls,
                      max_instances: int | None = None,
                      max_channels: int | None = None,
                      max_queue_per_channel: int | None = None,
                      max_message_size: int | None = None,
                      enable_logging: bool = False,
                      unai_logger: Any = None) -> None:
        """
        Initializes the underlying Go library. Must be called once. This is called automatically.
                Initializes the underlying Go library. Must be called once. This is called automatically.

        Args:
            enable_logging: Enable Go-level log output.  When True and ``unai_logger``
                            is supplied, all Go stdout/stderr lines are captured and
                            routed to the ``P2P`` channel of the provided logger
                            instead of being printed raw to the terminal.
            unai_logger:    An instance of ``unaiverse.utils.logger.Logger``.
                            When provided (and ``enable_logging=True``), Go output is
                            intercepted at the OS file-descriptor level and emitted
                            via ``logger.p2p()``.  Ignored when ``enable_logging=False``.
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
                    'net/identify': 'info',
                    'unailib': 'info',

                    # The AutoTLS certificate manager: the wait for the verdict, the registration
                    # and the certificate, or why none was asked
                    'p2p-forge/client': 'info',
                    'nat': 'info',
                    'basichost': 'info',
                    'p2p-circuit': 'info',
                    'relay': 'info',
                    'p2p-holepunch': 'info',
                    'tcp-tpt': 'info',
                    'quic-transport': 'info',
                    'quicreuse': 'info',
                    'webtransport': 'info',
                    'connmgr': 'info',
                    'dht': 'info',

                    # AutoRelay reports its candidates and reservation attempts at debug only
                    'autorelay': 'debug',
                    'autonat': 'info',
                    'autonatv2': 'info',

                    # 'rcmgr': 'info',
                    'swarm2': 'info',
                    'yamux': 'info',
                }

            # --- Start OS-level fd capture BEFORE InitializeLibrary, so we
            #     catch every line the Go runtime emits from the very start ---
            if enable_logging and unai_logger is not None and _CAPTURE_AVAILABLE:
                def _make_cb(src_label: str):
                    def _cb(line: str):
                        unai_logger.p2p(line, src=src_label)
                    return _cb
                cls._cap_stdout = FdCapture(1, _make_cb("stdout"))
                cls._cap_stderr = FdCapture(2, _make_cb("stderr"))
                cls._cap_stdout.start()
                cls._cap_stderr.start()
                logger.info("🐍 Go stdout/stderr redirected to P2P log channel.")

            logger.info("🐍 Setting up and initializing P2P library core with user settings...")
            cls._type_interface = TypeInterface(cls.libp2p)

            # Use provided arguments or fall back to class defaults
            _max_instances = max_instances if max_instances is not None else cls._MAX_INSTANCES
            _max_channels = max_channels if max_channels is not None else cls._MAX_NUM_CAHNNELS
            _max_queue = max_queue_per_channel if max_queue_per_channel is not None else cls._MAX_QUEUE_PER_CHANNEL
            _max_msg_size = max_message_size if max_message_size is not None else cls._MAX_MESSAGE_SIZE

            # Validate the message-size cap: it is passed to Go as a C int and there
            # reinterpreted as uint32. A non-positive or >2^31-1 value would wrap to a
            # negative C int and then to a huge uint32, silently lifting the cap.
            if not (0 < _max_msg_size <= 2 ** 31 - 1):
                raise ValueError(
                    f"max_message_size must be a positive int <= 2^31-1, got {_max_msg_size}."
                )

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

            # The per-peer send queues of the direct messages, with the limits of Custom (Go has no defaults of its
            # own); a library older than them has no such export and keeps sending synchronously
            cls._send_queues_enabled = False
            if hasattr(cls.libp2p, 'ConfigureSendQueues'):
                send_queue_config = {
                    'max_bytes_per_peer': int(Custom.SEND_QUEUE_MAX_BYTES_PER_PEER),
                    'max_msgs_per_peer': int(Custom.SEND_QUEUE_MAX_MSGS_PER_PEER),
                    'max_bytes_total': int(Custom.SEND_QUEUE_MAX_BYTES_TOTAL),
                    'low_water': float(Custom.SEND_QUEUE_LOW_WATER),
                    'max_wait_age_ms': int(Custom.SEND_MAX_WAIT_AGE * 1000),
                    'max_dial_attempts': int(Custom.SEND_MAX_DIAL_ATTEMPTS),
                    'retry_backoff_ms': [int(b * 1000) for b in Custom.SEND_RETRY_BACKOFF],
                    'write_base_timeout_ms': int(Custom.SEND_WRITE_BASE_TIMEOUT * 1000),
                    'write_min_bytes_per_sec': int(Custom.SEND_WRITE_MIN_BYTES_PER_SEC),
                    'max_events': int(Custom.SEND_MAX_EVENTS),
                    'chunk_size': int(Custom.CHUNK_SIZE),
                    'max_message_size': int(Custom.MAX_MESSAGE_SIZE),
                    'reassembly_max_bytes_per_peer': int(Custom.REASSEMBLY_MAX_BYTES_PER_PEER),
                    'reassembly_max_bytes_total': int(Custom.REASSEMBLY_MAX_BYTES_TOTAL),
                    'reassembly_timeout_ms': int(Custom.REASSEMBLY_TIMEOUT * 1000),
                    'file_incoming_dir': p2p_files.incoming_dir(),
                    'accept_files': bool(Custom.ACCEPT_FILES),
                    'max_file_size': int(Custom.MAX_FILE_SIZE),
                }
                result = cls._type_interface.from_go_ptr_to_json(
                    cls.libp2p.ConfigureSendQueues(cls._type_interface.to_go_json(send_queue_config)))
                if result is not None and result.get('state') == "Success":
                    cls._send_queues_enabled = True
                    cls.supports_chunked_frames = (hasattr(cls.libp2p, 'SetPeerCapabilities') and
                                                   int(Custom.CHUNK_SIZE) > 0)
                    cls.supports_file_stream = (cls.supports_chunked_frames and
                                                hasattr(cls.libp2p, 'SendFileMessageToPeer'))
                    cls.accepts_file_stream = cls.supports_file_stream and bool(Custom.ACCEPT_FILES)
                    cls.supports_stop_sending = cls.supports_chunked_frames and hasattr(cls.libp2p, 'CancelSend')
                    logger.info(f"✅ Send queues configured: {send_queue_config}")
                else:
                    logger.error(f"❌ Send queues not configured, sends stay synchronous: {result}")

            cls.supports_split_pop = hasattr(cls.libp2p, 'PopNetworkMessages')
            cls._library_initialized = True
            logger.info("✅ Go library initialized successfully.")

    def __init__(self,
                 identity_dir: str,
                 port: int = 0,
                 ips: list[str] = None,
                 enable_relay_client: bool = True,
                 enable_relay_service: bool = False,
                 use_broad_limits: bool = False,
                 is_isolated: bool = False,
                 knows_is_public: bool = False,
                 enable_tls: bool = False,
                 domain_name: str | None = None,
                 tls_cert_path: str | None = None,
                 tls_key_path: str | None = None,
                 probe_peers: list[str] | None = None,
                 webrtc_enabled: bool = True,
                 flood_sub: bool = False,
                 ice_stun_servers: list[str] | None = None,
                 ice_turn_servers: list[dict[str, Any]] | None = None,
                 log_sub: str = "pub",
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
                AutoNAT dial-backs from the probe peers...).
            enable_tls: Whether to enable AutoTLS certificate management (requires internet access).
            probe_peers: Multiaddrs (with peer IDs) dialed at boot as AutoNAT dial-back servers, so the
                node can discover its own reachability. None uses the built-in infrastructure relays;
                an empty list disables probing entirely (no reachability verdict unless knows_is_public).
            domain_name: Optional domain name for TLS certificate (required if enable_tls is True).
            tls_cert_path: Optional path to a custom TLS certificate file (PEM format).
            tls_key_path: Optional path to a custom TLS private key file (PEM format).
            webrtc_enabled: Enable the /unaiverse/webrtc-signal/1.0.0 protocol for
                Go-to-browser NAT traversal via WebRTC DataChannels. Defaults to True.
            flood_sub: Enable flood sub. Defaults to False.
            ice_stun_servers: List of STUN server URIs (e.g. ["stun:stun.example.com:3478"]).
                Defaults to Google public STUN when None.
            ice_turn_servers: List of TURN server dicts, each with keys "urls", "username",
                "credential". Optional; leave None if no TURN servers are needed.

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
        self._peer_id: str | None = None
        self.log_sub = log_sub
        self._peer_caps: dict[str, int] = {}  # Peer ID -> the CAP_* bits of its Session (set_peer_capabilities)

        # Who sent a file (a callable: peer ID -> (handle, node name), or None when unknown), set by the node that
        # owns this instance: a received file goes to Downloads/<handle>/<node name>/
        self.file_owner_of = None

        # TLS Validation Logic
        has_custom_tls_args = (tls_cert_path is not None) or (tls_key_path is not None) or (domain_name is not None)
        if has_custom_tls_args:
            if domain_name is None or tls_cert_path is None or tls_key_path is None:
                raise ValueError("Custom TLS requires 'domain_name', 'tls_cert_path' and 'tls_key_path'.")

        use_auto_tls = enable_tls and not has_custom_tls_args
        self._enable_tls = use_auto_tls or has_custom_tls_args
        
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
                "flood_sub": flood_sub,
            },
            "reachability": {
                "probe_peers": probe_peers,
            },
            "webrtc": {
                "enabled": webrtc_enabled,
                "ice_config": {
                    "stun_servers": ice_stun_servers,
                    "turn_servers": ice_turn_servers,
                } if webrtc_enabled else None,
            },
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
                raise P2PError(f"[Instance {self._instance}] {err_msg}")

            self._peer_id = initial_addresses[0].split("/")[-1]
            if P2P.supports_file_stream:
                p2p_files.clear_incoming(self._peer_id)  # What a previous run of this identity left half received

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

        logger.info("🎉 Node created successfully.")

    # --- Core P2P Operations ---
    def connect_to(self, multiaddrs: list[str]) -> dict[str, Any]:
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

    def send_message_to_peer(self, channel: str, msg_bytes: bytes) -> int | None:
        """
        Sends a direct message to a specific peer. With the send queues on, the message is only queued (the call
        returns at once, whatever the state of the peer) and a writer of the peer writes it, in order.

        Args:
            channel: The string identifying the channel for the communication.
            msg_bytes: The message to send (bytes).

        Returns:
            The sequence number of the message in the queue of its peer, or None when the send queues are off (the
            message was then written before returning).

        Raises:
            P2PQueueFull: If the send queue of the peer is full (the message is not queued).
            P2PError: If message sending fails (based on return code).
            ValueError: If inputs are invalid.
            TypeError: If data is not bytes.
        """
        if not channel or not isinstance(channel, str):
            logger.error("Invalid channel provided.")
            raise ValueError("Invalid channel provided.")
        if "::dm:" not in channel:

            # Guard the split below: without "::dm:" it would raise a raw IndexError.
            logger.error(f"Invalid direct-message channel format: '{channel}'.")
            raise ValueError(f"Invalid direct-message channel format: '{channel}'.")

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
                if result.get('code') == "queue_full":
                    raise P2PQueueFull(result.get('message', 'Send queue full'), peer_id)
                logger.error(f"Failed to send direct message to '{peer_id}': "
                             f"{result.get('message', 'Unknown Go error')}")
                raise P2PError(f"Failed to send direct message to '{peer_id}': "
                               f"{result.get('message', 'Unknown Go error')}")

            seq = result.get('seq')
            logger.info(f"✅ Successfully {'queued' if seq is not None else 'sent'} direct message to "
                        f"...{peer_id[-5:]}.")
            return seq

        except P2PQueueFull:
            raise
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
        
        if "::ps:" not in channel:
            logger.error(f"Invalid pubsub channel format: '{channel}'.")
            raise ValueError(f"Invalid pubsub channel format: '{channel}'.")

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

    async def send(self, channel: str, msg: Msg) -> int | None:
        """Encodes and sends a direct chat Msg to a peer (async).

        The wire codec (Msg.to_bytes) and the thread offload both live here, at the transport tier (the rewrite
        boundary toward OP2P.js): the Coordinator hands over a Msg object and awaits, it never serializes nor
        offloads. Outbound mirror of send_session. With the send queues on, the Go call only queues, and the thread
        is still worth it for the protobuf + PNG + gzip encoding.

        Args:
            channel: The direct-message channel.
            msg: The message object.

        Returns:
            The sequence number of the message in the queue of its peer, or None when the send queues are off.

        Raises:
            P2PQueueFull: If the send queue of the peer is full.
            P2PError: If the send fails.
        """

        # The FILE samples with a path go next to the message, read from disk while they are sent, to a peer that takes
        # file messages; to any other peer their files are read into the message
        streamed = p2p_files.streamed_files(msg.content_type, msg.content) if P2P.supports_file_stream else []
        if streamed and self._peer_caps.get(self._peer_of_channel(channel), 0) & CAP_FILE_STREAM:
            keys = {f["key"] for f in streamed}
            try:
                return await asyncio.to_thread(
                    lambda: self.send_file_message_to_peer(channel, msg.to_bytes(streamed_keys=keys), streamed))
            except P2PNoFileStream:
                pass

        # The lambda function moves the protobuf+PNG+gzip work (the expensive part) off the event loop
        return await asyncio.to_thread(lambda: self.send_message_to_peer(channel, msg_bytes=msg.to_bytes()))

    def send_file_message_to_peer(self, channel: str, msg_bytes: bytes, files: list[dict]) -> int | None:
        """Queues a file message for a peer: the message, whose FILE samples have no content, and their files, which Go
        reads from disk while it sends them (Phase 6.5).

        Args:
            channel: The direct-message channel.
            msg_bytes: The message.
            files: The files, [{"key": sample name, "path": file}].

        Returns:
            The sequence number of the message in the queue of its peer.

        Raises:
            P2PNoFileStream: If the peer does not take file messages (nothing queued).
            P2PQueueFull: If the send queue of the peer is full.
            P2PError: If the message is refused for another reason (a limit, a missing file).
        """
        import json
        peer_id = self._peer_of_channel(channel)
        result = P2P._type_interface.from_go_ptr_to_json(P2P.libp2p.SendFileMessageToPeer(
            P2P._type_interface.to_go_int(self._instance), P2P._type_interface.to_go_string(channel),
            P2P._type_interface.to_go_bytes(msg_bytes), P2P._type_interface.to_go_int(len(msg_bytes)),
            P2P._type_interface.to_go_string(json.dumps(files))))
        if result is None:
            raise P2PError(f"Failed to send a file message to {peer_id}, received null result.")
        if result.get('state') == "Error":
            if result.get('code') == "no_file_stream":
                raise P2PNoFileStream(result.get('message', 'No file messages'))
            if result.get('code') == "queue_full":
                raise P2PQueueFull(result.get('message', 'Send queue full'), peer_id)
            logger.error(f"Failed to send a file message to '{peer_id}': {result.get('message')}")
            raise P2PError(f"Failed to send a file message to '{peer_id}': {result.get('message')}")
        logger.info(f"✅ Queued a file message ({len(files)} files) to ...{peer_id[-5:]}.")
        return result.get('seq')
    async def broadcast(self, channel: str, msg: Msg) -> None:
        """Encode and broadcast a chat Msg over pubsub. Like send, the codec and the
        thread offload live here; the Coordinator hands over a Msg object and awaits (async)."""
        await asyncio.to_thread(self.broadcast_message, channel, msg.to_bytes())

    def pop_messages(self, kind: str = "all") -> list[bytes]:
        """
        Retrieves and removes the first message from the queue of each chat/pubsub channel for this
        node instance, and every queued frame of the Session protocols (their queues are shared by
        all the peers, so they are drained in full at every call, oldest frame first).

        With kind "network" it takes instead every queued message of every channel that is not a stream channel
        (Session frames included), with kind "streams" the oldest message of every stream channel (stream_sample
        direct messages, PubSub topics): the network pass and the tick of a node with two tasks. A library without
        the split pops serves "network" as "all" and "streams" with nothing.

        Args:
            kind: "all" (default), "network" or "streams".

        Returns:
            A list of byte arrays (messages). Returns an empty list if no messages were available.

        Raises:
            P2PError: If popping messages failed internally in Go, or if data
                      conversion fails for any message.
        """
        logger.debug(f"[Instance {self._instance}] Popping message(s)...")
        try:
            go_instance_c = P2P._type_interface.to_go_int(self._instance)

            if kind == "all" or not P2P.supports_split_pop:
                if kind == "streams":
                    return []
                result_ptr = P2P.libp2p.PopMessages(go_instance_c)
            elif kind == "network":
                result_ptr = P2P.libp2p.PopNetworkMessages(go_instance_c)
            else:
                result_ptr = P2P.libp2p.PopStreamMessages(go_instance_c)

            # Binary message buffer (success) or JSON status dict (Empty/Error);
            # from_go_pop_messages decodes either and frees result_ptr. On the binary
            # path each message's "data" is already raw bytes (no base64).
            raw_result = P2P._type_interface.from_go_pop_messages(result_ptr)

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

    # --- Send queues of the direct messages ---
    def drop_send_queue(self, peer_id: str) -> int:
        """Empties the send queue of a peer at once (Go records a queue_dropped event, reason "explicit").

        For the purge of a peer: what is still queued for it is no longer wanted, and its writer must not dial it
        again. A no-op when the send queues are off.

        Args:
            peer_id: The peer.

        Returns:
            How many queued messages were dropped.
        """
        if not P2P._send_queues_enabled:
            return 0
        try:
            result = P2P._type_interface.from_go_ptr_to_json(P2P.libp2p.DropSendQueue(
                P2P._type_interface.to_go_int(self._instance), P2P._type_interface.to_go_string(peer_id)))
        except Exception as e:
            logger.error(f"[Instance {self._instance}] DropSendQueue({peer_id}) failed: {e}")
            return 0
        if result is None or result.get('state') != "Success":
            logger.error(f"[Instance {self._instance}] DropSendQueue({peer_id}) failed: {result}")
            return 0
        return int(result.get('message') or 0)

    def cancel_send(self, peer_id: str, seq: int) -> bool:
        """Stops one message of the send queue of a peer, because the peer (its receiver) asked so (STOP_SENDING).

        A message still waiting leaves the queue at once; the one being written stops before its next chunk. Go then
        records a queue_dropped event of that seq, reason "receiver_stopped". A no-op when the send queues are off.

        Args:
            peer_id: The peer (the receiver).
            seq: The sequence number of the message in its queue.

        Returns:
            True when the message was in the queue (and is now stopped).
        """
        if not P2P._send_queues_enabled or not P2P.supports_stop_sending:
            return False
        try:
            result = P2P._type_interface.from_go_ptr_to_json(P2P.libp2p.CancelSend(
                P2P._type_interface.to_go_int(self._instance), P2P._type_interface.to_go_string(peer_id), int(seq)))
        except Exception as e:
            logger.error(f"[Instance {self._instance}] CancelSend({peer_id}, {seq}) failed: {e}")
            return False
        if result is None or result.get('state') != "Success":
            logger.error(f"[Instance {self._instance}] CancelSend({peer_id}, {seq}) failed: {result}")
            return False
        return bool(result.get('message'))

    def pop_send_events(self) -> list[dict]:
        """Pops the events of the send queues recorded since the last call, oldest first.

        Each event is a dict with "type" (written, write_failed, recovered, refused, queue_dropped, events_lost),
        "peer_id" (absent for events_lost), "time" (unix seconds) and the fields of its type: "seq", "from_seq",
        "to_seq", "count", "bytes", "attempt", "reason", "error". Consecutive written (or refused) events of a peer
        are merged into one. An empty list when the send queues are off.

        Returns:
            The list of the events.
        """
        if not P2P._send_queues_enabled:
            return []
        try:
            result = P2P._type_interface.from_go_ptr_to_json(
                P2P.libp2p.PopSendEvents(P2P._type_interface.to_go_int(self._instance)))
        except Exception as e:
            logger.error(f"[Instance {self._instance}] PopSendEvents failed: {e}")
            return []
        if result is None or result.get('state') != "Success":
            if result is None or result.get('state') != "Empty":
                logger.error(f"[Instance {self._instance}] PopSendEvents failed: {result}")
            return []
        return list(result.get('message') or [])

    def set_peer_capabilities(self, peer_id: str, capabilities: int, max_message_size: int | None,
                              max_file_size: int | None = None) -> None:
        """Tells Go what a peer can receive: the capability bits of its HELLO / HELLO_ACK, and the largest message and
        file its profile accepts. A message larger than a chunk then goes to it as chunk frames if it has
        CAP_CHUNKED_FRAMES, a message with path-backed files as a file message if it has CAP_FILE_STREAM, and one above
        its maximum is refused instead of sent (it would be reset).

        Args:
            peer_id: The peer.
            capabilities: Its CAP_* bits.
            max_message_size: The largest message it accepts (0: no limit; None: unknown, legacy).
            max_file_size: The largest file it accepts (0 or None: no limit).
        """
        self._peer_caps[peer_id] = int(capabilities)
        if not hasattr(P2P.libp2p, 'SetPeerCapabilities'):
            return
        try:
            P2P._type_interface.from_go_ptr_to_json(P2P.libp2p.SetPeerCapabilities(
                P2P._type_interface.to_go_int(self._instance), P2P._type_interface.to_go_string(peer_id),
                int(capabilities), -1 if max_message_size is None else int(max_message_size),
                int(max_file_size or 0)))
        except Exception as e:
            logger.error(f"[Instance {self._instance}] SetPeerCapabilities({peer_id}) failed: {e}")

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

    # --- Session abstraction (per-stage handshake / liveness / signed peer record) ---
    def send_on_protocol(self, peer_id: str, protocol_id: str, payload: bytes) -> bool:
        """
        Open a fresh outbound stream on `protocol_id` to `peer_id`, write a single
        length-prefixed protobuf payload, and close the stream. Returns True on
        send success, False on transport failure.

        The response (if any) is delivered asynchronously as a pop_messages entry
        whose `protocol` field equals `protocol_id`.
        """
        if not peer_id or not isinstance(peer_id, str):
            raise ValueError("send_on_protocol: peer_id must be a non-empty string")
        if not protocol_id or not isinstance(protocol_id, str):
            raise ValueError("send_on_protocol: protocol_id must be a non-empty string")
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError("send_on_protocol: payload must be bytes")

        try:
            result_ptr = P2P.libp2p.SendOnProtocol(
                P2P._type_interface.to_go_int(self._instance),
                P2P._type_interface.to_go_string(peer_id),
                P2P._type_interface.to_go_string(protocol_id),
                P2P._type_interface.to_go_bytes(bytes(payload)),
                P2P._type_interface.to_go_int(len(payload)),
            )
            result = P2P._type_interface.from_go_ptr_to_json(result_ptr)
            if result is None or result.get('state') == 'Error':
                logger.warning(f"send_on_protocol(...{peer_id[-5:]}, {protocol_id}) failed: "
                               f"{result.get('message') if result else 'null result'}")
                return False
            return True
        except Exception as e:
            logger.error(f"send_on_protocol(...{peer_id[-5:]}, {protocol_id}) exception: {e}")
            return False

    # --- High-level Session transport surface (CPython twin of OP2P.js)
    #
    # These are the methods the Session/SessionManager call directly on their
    # injected P2P instance, with no p2p_name argument (each P2P is already a single
    # instance). They wrap the raw Go primitives above, adding the address
    # sorting/retry/relay logic and the SessMsg serialization that used to live in
    # connpool, plus thread offload for the blocking Go calls. The crypto wrappers
    # are async so session.py `await`s the same way in CPython and in Pyodide/JS.
    async def connect(self, addresses: list[str]) -> tuple[str | None, bool]:
        """Establish a connection to a peer: one ConnectTo with every non-loopback
        address. The libp2p swarm dials them in parallel under a single Go-side
        budget (its dialer already ranks transports; the first success wins), so
        there is no second sequential pass and the worst case is one dial timeout.
        Returns (peer_id, through_relay) (async)."""
        force: str | None = None

        # force: str | None = '/tcp'
        if force is not None:
            _addresses = [a for a in addresses if force in a]
            addresses.clear()
            for a in _addresses:
                addresses.append(a)
        addresses = [a for a in addresses if "127.0.0.1" not in a]
        if len(addresses) == 0:
            logger.error("Connection failed! (not even tried, no usable addresses)")
            return None, False
        logger.info(f"Connecting to {addresses}")
        try:

            # Off the event loop: the dial can block up to its timeout, and other
            # sessions must keep ticking meanwhile (e.g. during a re-dial).
            winning_addr_info_dict = await asyncio.to_thread(self.connect_to, addresses)
            peer_id = winning_addr_info_dict.get('ID')
            addrs = winning_addr_info_dict.get('Addrs')
            assert addrs is not None and isinstance(addrs, (tuple, list))
            connected_addr_str = addrs[0]
            through_relay = '/p2p-circuit/' in connected_addr_str
            logger.info(f"Connected to peer {peer_id} via {connected_addr_str} "
                        f"(through relay: {through_relay})")
            return peer_id, through_relay
        except P2PError:
            logger.error("Connection failed!")
            return None, False

    async def disconnect(self, peer_id: str) -> bool:
        """Abortively tear down the connection to `peer_id`. Go resets every open
        stream before closing the connection, so any frame written but not yet
        delivered (e.g. a GOAWAY queued microseconds earlier) is DESTROYED, not
        flushed: callers that just sent a farewell must not disconnect in the same
        stack — schedule the teardown after the Session reaches a terminal state.
        Returns True on success, False on a transport-level failure. The canonical
        transport-disconnect primitive the SessionManager delegates to; best-effort,
        any error (including a malformed peer_id) means 'gone' rather than
        propagating into the caller. The blocking Go round-trip (WebRTC teardown +
        stream resets + ClosePeer) runs off the event loop, same as `send` (async)."""
        try:
            await asyncio.to_thread(self.disconnect_from, peer_id)
        except (P2PError, ValueError):
            return False
        return True

    async def send_session(self, peer_id: str, protocol_id: str, sess_msg: SessMsg) -> bool:
        """Serialize `sess_msg` via its own `to_bytes()` and open a fresh outbound
        stream on `protocol_id` to `peer_id`, writing one length-prefixed payload and
        closing the stream. Returns True when the payload was ACCEPTED by the local
        muxer (written + half-closed) — NOT when it was delivered: delivery needs the
        connection to stay alive for the transport to flush/retransmit, and an
        abortive disconnect (or process exit) right after a True return destroys the
        frame. End-to-end confirmation is the Session protocol's own ACKs (async).

        The Session FSM passes a `SessMsg` value object (not bytes), so this method
        stays protobuf-free: it just calls `.to_bytes()` on whatever it receives. The
        Go call runs off the event loop, same as `send`.
        """
        try:
            payload = sess_msg.to_bytes()
        except Exception as e:
            logger.error(f"send_session(...{peer_id[-5:]}, {protocol_id}) encode failed: {e}")
            return False
        try:
            return await asyncio.to_thread(self.send_on_protocol, peer_id, protocol_id, payload)
        except Exception as e:
            logger.error(f"send_session(...{peer_id[-5:]}, {protocol_id}) failed: {e}")
            return False

    def get_messages(self, kind: str = "all") -> tuple:
        """Pop this instance's inbound batch (all, the network part or the streams: see pop_messages) and split it.
        The whole wire codec lives here, in the transport tier (this is the rewrite boundary toward OP2P.js):
        Session-protocol frames are decoded into SessionInbound(peer_id, protocol_id, SessMsg) via
        SessMsg.from_bytes, chat frames into Msg objects via Msg.from_bytes. Chat frames are
        also sender-verified here (the sender claimed inside the payload must match the
        cryptographically verified network sender); spoofed or undecodable frames are dropped.
        The Node-bound chat path downstream only gates membership and verifies the JWT
        piggyback, it never touches the codec. Inbound mirror of send_session / send.

        Returns (chat_messages, session_inbounds): chat_messages a list of decoded,
        sender-verified Msg; session_inbounds a list of SessionInbound.
        """
        chat_messages = []
        session_inbounds = []
        for msg_dict in (self.pop_messages() if kind == "all" else self.pop_messages(kind)):
            protocol_id = msg_dict.get("protocol") or ""
            if protocol_id in SESSION_PROTOCOL_IDS:
                try:

                    # The pop reply gives views of its buffer: protobuf parses them as they are, the Session codec
                    # needs bytes (its frames are small)
                    sess_msg = SessMsg.from_bytes(bytes(msg_dict.get("data") or b""), protocol_id)
                except Exception as e:
                    logger.error(f"get_messages: failed to decode session message on "
                                 f"{protocol_id} from {msg_dict.get('from')}: {e}")
                    continue
                session_inbounds.append(SessionInbound(
                    peer_id=msg_dict.get("from"), protocol_id=protocol_id, sess_msg=sess_msg))
            else:

                # Chat path (Node-bound). The binary transport delivers "data" already as
                # raw bytes (Go structure: {"from":"Qm...", "data":<bytes>, "protocol":"..."}).
                verified_sender_id = msg_dict.get("from")
                received_files = msg_dict.get("files") or []
                try:
                    msg_obj = Msg.from_bytes(msg_dict.get("data"))
                except Exception as e:
                    logger.error(f"get_messages: failed to decode chat message from "
                                 f"{verified_sender_id}: {e}")
                    p2p_files.remove_quietly([f["path"] for f in received_files])
                    continue

                # --- CRITICAL SECURITY CHECK ---
                # Verify that the sender claimed inside the message payload matches the
                # cryptographically verified sender from the network layer.
                if msg_obj.sender != verified_sender_id:
                    logger.error(f"SENDER MISMATCH! Network sender '{verified_sender_id}' "
                                 f"does not match payload sender '{msg_obj.sender}'. "
                                 f"Discarding message.")
                    p2p_files.remove_quietly([f["path"] for f in received_files])
                    continue
                if received_files and not self._place_received_files(msg_obj, verified_sender_id, received_files):
                    continue

                msg_obj.wire_size = len(msg_dict.get("data") or b"")  # For the bounds of what waits for a tick
                chat_messages.append(msg_obj)
        return chat_messages, session_inbounds

    # --- Async crypto wrappers (Session awaits these)
    # Thin async wrappers over the synchronous Go FFI primitives above so session.py
    # `await`s identically in CPython and in Pyodide-over-JS.
    async def sign_with_host_key(self, data: bytes) -> bytes | None:
        """Async wrapper over the host-key signer (see sync impl) (async)."""
        return self._sign_with_host_key_sync(data)

    async def verify_peer_signature(self, peer_id: str, data: bytes, signature: bytes) -> bool:
        """Async wrapper over the peer-signature verifier (see sync impl) (async)."""
        return self._verify_peer_signature_sync(peer_id, data, signature)

    # --- Relay Operations ---
    def reserve_on_relay(self, relay_peer_id: str) -> str:
        """
        Attempts to reserve a slot on a specified relay node.

        Args:
            relay_peer_id: The peerID of the relay node

        Returns:
            The UTC expiration timestamp of the reservation as an ISO 8601 string.

        Raises:
            P2PError: If the reservation fails.
            ValueError: If the relay_multiaddr is invalid.
        """
        if not relay_peer_id or not isinstance(relay_peer_id, str):
            logger.error("Invalid relay multiaddr provided.")
            raise ValueError("Invalid relay multiaddr provided.")
        logger.info(f"🅿️ Attempting to reserve slot on relay: {relay_peer_id}...")
        try:
            result_ptr = P2P.libp2p.ReserveOnRelay(
                P2P._type_interface.to_go_int(self._instance),
                P2P._type_interface.to_go_string(relay_peer_id)
                )
            result = P2P._type_interface.from_go_ptr_to_json(result_ptr)

            if result is None:
                logger.error("Failed to reserve on relay, received null result.")
                raise P2PError("Failed to reserve on relay, received null result.")
            if result.get('state') == "Error":
                go_error = result.get('message', 'Unknown Go error')
                logger.error(f"Failed to reserve on relay '{relay_peer_id}': {go_error}")
                raise P2PError(f"Failed to reserve on relay '{relay_peer_id}': {go_error}")

            expiration_utc = result.get('message', {})
            if not isinstance(expiration_utc, str):
                raise P2PError(f"Expected expiration timestamp string, but got {type(expiration_utc)}")

            logger.info(f"✅ Reservation successful. Expires at: {expiration_utc}")
            return expiration_utc

        except Exception as e:
            logger.error(f"❌ Reservation on {relay_peer_id} failed: {e}")
            print(e)
            raise P2PError(f"Reservation on {relay_peer_id} failed") from e

    async def reserve(self, relay_peer_id: str) -> str:
        """Reserve a relay slot, offloading the blocking Go call to a thread. The async
        boundary lives here, at the transport tier (like send / broadcast / send_session),
        so the Coordinator just awaits and never offloads. Returns the expiry timestamp;
        raises P2PError on failure (the Coordinator catches it) (async)."""
        return await asyncio.to_thread(self.reserve_on_relay, relay_peer_id)

    # --- Node Information ---
    @property
    def peer_id(self) -> str | None:
        """Returns the Peer ID of the local node."""
        return self._peer_id

    @property
    def addresses(self) -> list[str]:
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
    def is_public(self) -> bool | None:
        """Returns whether the local node is publicly reachable, reading the live AutoNAT
        verdict from Go on each access (O(1), cached Go-side, like ``addresses``). For a
        ForcePublic node this is a fixed true; otherwise it tracks the latest AutoNAT
        verdict, falling back to the last known value on FFI error."""
        return self.get_reachability()

    @property
    def relay_is_enabled(self) -> bool:
        """Returns whether the relay client functionality is enabled for this node."""
        return self._enable_relay_client

    @property
    def tls_is_enabled(self) -> bool:
        """Returns whether the WebSocket listener of this node is a secure one (its own certificate or AutoTLS)."""
        return self._enable_tls

    def get_node_addresses(self, peer_id: str = "") -> list[str]:
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

    def get_reachability(self) -> bool:
        """Reads the current AutoNAT reachability from the Go engine (O(1), event-cached)
        and updates the cached ``_is_public``. Returns the last known value on FFI error
        rather than raising, since callers treat reachability as best-effort telemetry."""
        try:
            result_ptr = P2P.libp2p.GetReachability(P2P._type_interface.to_go_int(self._instance))
            result = P2P._type_interface.from_go_ptr_to_json(result_ptr)
            if result is None or result.get('state') == "Error":
                msg = result.get('message', 'null result') if result is not None else 'null result'
                logger.warning(f"Failed to read reachability: {msg}. Using last known value.")
                return bool(self._is_public)
            self._is_public = bool(result.get('message', False))
            return self._is_public
        except Exception as e:
            logger.warning(f"Failed to read reachability: {e}. Using last known value.")
            return bool(self._is_public)

    def get_webrtc_connections(self) -> list[dict[str, Any]]:
        """
        Returns the list of peers that currently have an active WebRTC
        DataChannel connection with this node.

        Each entry is a dict with keys ``peer_id`` (str) and ``state`` (str,
        ``"open"`` or ``"other"``).

        Raises:
            P2PError: If the Go call fails.
        """
        try:
            result_ptr = P2P.libp2p.GetWebRTCConnections(
                P2P._type_interface.to_go_int(self._instance),
            )
            result = P2P._type_interface.from_go_ptr_to_json(result_ptr)
            if result is None:
                raise P2PError("Null result from GetWebRTCConnections.")
            if result.get('state') == "Error":
                raise P2PError(f"GetWebRTCConnections failed: "
                               f"{result.get('message', 'Unknown Go error')}")
            return result.get('message', [])
        except Exception as e:
            logger.error(f"❌ GetWebRTCConnections failed: {e}")
            raise P2PError("GetWebRTCConnections failed") from e

    def set_turn_credentials(self, ice_turn_servers: list[dict[str, Any]]) -> None:
        """Updates the TURN servers of the ICE configuration (future WebRTC connections).

        Args:
            ice_turn_servers: The TURN server descriptions.
        """
        self.set_ice_config(ice_turn_servers=ice_turn_servers)

    def set_ice_config(self,
                       ice_stun_servers: list[str] | None = None,
                       ice_turn_servers: list[dict[str, Any]] | None = None) -> None:
        """Update STUN/TURN servers used for *future* WebRTC connections.

        Existing PeerConnections keep their captured ICE config; each newly
        constructed PC reads the updated values.  Safe to call from any thread.

        Pass ``None`` for a field to leave it unchanged on the Go side; pass
        ``[]`` to explicitly clear it.  Typical usage rotates only TURN creds
        while leaving STUN alone.

        Args:
            ice_stun_servers: STUN URIs, or ``None`` to keep existing STUN list.
            ice_turn_servers: TURN dicts (urls/username/credential), or
                ``None`` to keep existing TURN list.
        """
        cfg: dict[str, Any] = {}
        if ice_stun_servers is not None:
            cfg["stun_servers"] = ice_stun_servers
        if ice_turn_servers is not None:
            cfg["turn_servers"] = ice_turn_servers
        if not cfg:
            return  # nothing to update — no round-trip
        result_ptr = P2P.libp2p.UpdateICEConfig(
            P2P._type_interface.to_go_int(self._instance),
            P2P._type_interface.to_go_json(cfg),
        )
        result = P2P._type_interface.from_go_ptr_to_json(result_ptr)
        if result is None or result.get('state') == "Error":
            msg: str = result.get('message', 'unknown error') if result else 'null result'
            raise P2PError(f"[Instance {self._instance}] set_ice_config failed: {msg}")

    def get_connected_peers_info(self) -> list[dict[str, Any]] | None:
        """
        Gets information about currently connected peers from the Go library.

        Returns:
            On success, a (possibly empty) list of dictionaries, each representing a
            connected peer with keys like 'addr_info' (containing 'ID', 'Addrs'),
            'connected_at', 'direction', and 'misc'. A real empty list means "zero peers".

            ``None`` on an FFI-LEVEL failure — a Go panic reported as ``state == "Error"``,
            a NULL/unparseable result pointer, or a marshalling exception — so such a
            transient is not read as "zero peers": the reconcile skips the diff on None
            (leaving the pools intact) instead of tearing every peer down. Note this does
            NOT cover a Go "instance not ready", which Go reports as a successful empty list
            (read here as a genuine zero-peers), not as an error.
        """

        # Logger.info("ℹ️ Fetching connected peers info...") # Can be noisy
        try:

            # GetConnectedPeers takes no arguments in Go
            result_ptr = P2P.libp2p.GetConnectedPeers(P2P._type_interface.to_go_int(self._instance))
            result = P2P._type_interface.from_go_ptr_to_json(result_ptr)

            if result is None:
                logger.error("Failed to get connected peers, received null result.")
                return None
            if result.get('state') == "Error":
                logger.error(f"Failed to get connected peers: {result.get('message', 'Unknown Go error')}")
                return None

            # A successful call returns the real list (empty means genuinely zero peers).
            return result.get('message', [])

        except Exception as e:

            # Avoid crashing the polling thread, just log the error
            logger.error(f"❌ Error fetching connected peers info: {e}")

            # None (not []) so the caller distinguishes "FFI errored" from "zero peers".
            return None

    def get_topic_peers(self) -> list[str]:
        """
        Gets the peer IDs subscribed to any of this instance's topics (the de-duplicated
        union of topic.ListPeers()). Each is a peer we are DIRECTLY connected to that
        announced a subscription: the pubsub router intersects its connected-peer set with the
        topic subscribers, so a peer we are not connected to never appears. These are exactly
        the peers that receive this instance's broadcasts; the world perimeter uses them to cut
        the pubsub-only peers that open no stream and so never surface in
        get_connected_peers_info. A relay does not subscribe, so it never appears here.

        Returns:
            A list of peer-id strings, empty on error so that a transient FFI failure never
            triggers a spurious perimeter cut.
        """
        try:
            result_ptr = P2P.libp2p.GetTopicPeers(P2P._type_interface.to_go_int(self._instance))
            result = P2P._type_interface.from_go_ptr_to_json(result_ptr)

            if result is None:
                logger.error("Failed to get topic peers, received null result.")
                raise P2PError("Failed to get topic peers, received null result.")
            if result.get('state') == "Error":
                logger.error(f"Failed to get topic peers: {result.get('message', 'Unknown Go error')}")
                raise P2PError(f"Failed to get topic peers: {result.get('message', 'Unknown Go error')}")

            return result.get('message', [])

        except Exception as e:

            # Avoid crashing the polling loop; empty means "cut nobody" (fail-safe).
            logger.error(f"❌ Error fetching topic peers: {e}")
            return []

    def get_rendezvous_peers_info(self) -> dict[str, Any] | None:
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

            # Return None (not []) on error: the rest of this method uses None for
            # "no data yet", and callers branch on `is None`.
            return None

    # --- Lifecycle Management ---
    def close(self, close_all: bool = False) -> None | str:
        """
        Shuts down this instance's libp2p host via the Go CloseNode, freeing its
        slot (there are no Python-side background threads to stop). Closing the
        host abortively tears every remaining connection, so callers must drain
        their farewells first (see NodeConn.drain_teardown). Blocking FFI: the
        shutdown path calls it after the main loop has ended.

        Args:
            close_all: If True, closes ALL instances in the process (Go slot -1),
                not just this one. Default is False.
        """
        logger.info("🛑 Closing node...")

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
        finally:

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

    def __enter__(self) -> 'P2P':
        """Enters the context manager.

        Returns:
            This P2P instance.
        """
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        """Exits the context manager, ensuring node closure.

        Args:
            exc_type: The type of the exception raised in the block, if any.
            exc_val: The exception raised in the block, if any.
            exc_tb: The traceback of the exception, if any.
        """
        self.close()

    def _sign_with_host_key_sync(self, data: bytes) -> bytes | None:
        """Sign `data` with the host's libp2p private key (Ed25519 by default).
        Returns the raw signature bytes, or None on error. Used by Session to
        produce sig_a / sig_b over (nonce || session_id) for replay defence."""
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("sign_with_host_key: data must be bytes")
        try:
            result_ptr = P2P.libp2p.SignWithHostKey(
                P2P._type_interface.to_go_int(self._instance),
                P2P._type_interface.to_go_bytes(bytes(data)),
                P2P._type_interface.to_go_int(len(data)),
            )
            result = P2P._type_interface.from_go_ptr_to_json(result_ptr)
            if result is None or result.get('state') == 'Error':
                return None
            b64 = result.get('message', '')
            return base64.b64decode(b64) if b64 else None
        except Exception as e:
            logger.error(f"sign_with_host_key exception: {e}")
            return None

    def _verify_peer_signature_sync(self, peer_id: str, data: bytes, signature: bytes) -> bool:
        """Verify that `signature` over `data` is valid under the public key derivable
        from `peer_id`. Works for Ed25519 peer-ids (libp2p default — pubkey is
        embedded in the id); fails for RSA peer-ids."""
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("verify_peer_signature: data must be bytes")
        if not isinstance(signature, (bytes, bytearray)):
            raise TypeError("verify_peer_signature: signature must be bytes")
        try:
            sig_b64 = base64.b64encode(bytes(signature)).decode("ascii")
            result_ptr = P2P.libp2p.VerifyPeerSignature(
                P2P._type_interface.to_go_string(peer_id),
                P2P._type_interface.to_go_bytes(bytes(data)),
                P2P._type_interface.to_go_int(len(data)),
                P2P._type_interface.to_go_string(sig_b64),
            )
            result = P2P._type_interface.from_go_ptr_to_json(result_ptr)
            return bool(result) and result.get('state') == 'Success'
        except Exception as e:
            logger.error(f"verify_peer_signature exception: {e}")
            return False

    @staticmethod
    def _peer_of_channel(channel: str) -> str:
        """The peer of a direct-message channel ("...::dm:<peer ID>-...").

        Args:
            channel: The channel.

        Returns:
            The peer ID (empty for a channel of another kind).
        """
        return channel.split("::dm:")[1].split('-')[0] if "::dm:" in channel else ""

    def _place_received_files(self, msg_obj: Msg, sender_id: str, received_files: list[dict]) -> bool:
        """Moves the files of a received file message to Downloads/<handle>/<node name>/ and gives its FILE samples
        their paths. On any failure the files are removed and the message is dropped.

        Args:
            msg_obj: The decoded message.
            sender_id: Its verified sender.
            received_files: Its files, as Go wrote them ([{key, path, size}]).

        Returns:
            Whether the message is to be delivered.
        """
        owner = None
        try:
            if callable(self.file_owner_of):
                owner = self.file_owner_of(sender_id)
        except Exception as e:
            logger.warning(f"get_messages: owner of the files of {sender_id} unknown: {e}")
        content = msg_obj.content if isinstance(msg_obj.content, dict) else {}
        placed: dict[str, str] = {}
        try:
            for f in received_files:
                sample = content.get(f["key"])
                data = sample.get("data") if isinstance(sample, dict) else None
                if data is None or not hasattr(data, "filename"):
                    raise ValueError(f"no FILE sample {f['key']!r} in the message")
                placed[f["key"]] = p2p_files.place_received_file(f["path"], owner, sender_id, data.filename)
        except Exception as e:
            logger.error(f"get_messages: the files of a message from {sender_id} could not be placed: {e}")
            p2p_files.remove_quietly([f["path"] for f in received_files] + list(placed.values()))
            return False
        msg_obj.attach_received_files(placed)
        return True
