import os


class Custom:
    """Default options and constants of UNaIVERSE, collected in one place (class attributes, no instances)."""

    # ==================
    # NODE CONFIGURATION
    # ==================
    SEND_DYNAMIC_PROFILE_EVERY = 10.  # Seconds
    ROOT_REQUEST_TIMEOUT = 20.  # Seconds, per attempt on every root-server HTTP call
    GET_NEW_TOKEN_EVERY = 23 * 60. * 60. + 30 * 60.  # Seconds (23 hours and 30 minutes, safer)
    PUBLISH_RENDEZVOUS_EVERY = 10.  # Seconds
    SEND_ALIVE_EVERY = 2.5 * 60.  # Seconds
    SAVE_STATS_EVERY = 10.  # Seconds
    SEND_STATS_EVERY = 30.  # Seconds (warning: modified in Node constructor!)
    SAVE_CHECKPOINT_EVERY = -1.  # When negative, means "do not save" (warning: modified by means in Node constructor!)
    SLOW_DOWN_CLOCK_AFTER = 10.0  # Seconds without networks exchanges and 0 interaction (stats are not considered)
    SLOW_CLOCK_DELTA = 2.0  # Clock delta when running in slow-mode
    REJOINING_WAIT = 30.0  # Time to wait before trying to rejoin a world from which we were disconnected

    # Seconds a grant-time pre-admission may sit unredeemed before the world host unlists it (ghost bound; must
    # exceed the worst-case redemption: grant-bounded dial ~35s + HELLO/PROFILE stages 15s each, plus margin)
    WORLD_PREADMISSION_TTL = 90.0

    # Pace of the network task of a running node (session pump, messages, connections), independent of the HSM clock:
    # both well under the 10 s PING RTT window of the Session keep-alive
    NETWORK_LOOP_INTERVAL = 0.01  # Seconds between two network passes (at most one clock cycle) when not idle
    NETWORK_LOOP_INTERVAL_IDLE = 0.1  # Seconds between two network passes when the node is idle

    # Messages for the hosted entity (interactions, statuses, roles, ...) read while a tick runs wait for its end; when
    # these bounds are reached a new interaction is rejected (its sender is told), any other message is dropped
    DEFERRED_MSGS_MAX_PER_PEER = 64  # Messages waiting for the end of the tick, per sender
    DEFERRED_MSGS_MAX_BYTES = 256 * 1024 * 1024  # Bytes waiting for the end of the tick, over all the senders

    # ===========================================================
    # SEND QUEUES (direct messages, passed to Go at library setup)
    # ===========================================================
    # A send only queues the message; a writer per peer writes it. A full queue refuses the new message (send returns
    # False) and keeps refusing until it drained to the low-water mark; what is queued is never dropped one by one.
    SEND_QUEUE_MAX_MSGS_PER_PEER = 64  # Messages
    SEND_QUEUE_MAX_BYTES_PER_PEER = (1024 + 64) * 1024 * 1024  # Bytes (MAX_MESSAGE_SIZE, 1 GB, plus 64 MB)
    SEND_QUEUE_MAX_BYTES_TOTAL = 4 * 1024 * 1024 * 1024  # Bytes, over the queues of all the peers
    SEND_QUEUE_LOW_WATER = 0.5  # Fraction of the caps a queue that refused must drain to before accepting again
    SEND_MAX_WAIT_AGE = 60.  # Seconds the oldest unsent message may wait with no progress, then the queue is dropped
    SEND_MAX_DIAL_ATTEMPTS = 3  # Failed writes in a row (each with a new stream) before the queue is dropped
    SEND_RETRY_BACKOFF = (1., 2., 4.)  # Seconds before the n-th retry (the last one repeats)
    SEND_WRITE_STALL_TIMEOUT = 30.  # Seconds a write may wait with nothing drained (slow links are waited for)
    SEND_WRITE_BASE_TIMEOUT = 10.  # Seconds, the write deadline of a frame with an older library: this plus its size...
    SEND_WRITE_MIN_BYTES_PER_SEC = 1024 * 1024  # ...at this rate (failed every chunk below about 2.8 Mbit/s)
    SEND_MAX_EVENTS = 1024  # Send events kept in Go until Python pops them (the oldest go first)
    SEND_REFUSED_LOG_EVERY = 5.  # Seconds, the refusals of a full queue are logged at most this often per peer

    # =========================================
    # LARGE MESSAGES (chunked frames) AND FILES
    # =========================================
    # A direct message larger than CHUNK_SIZE goes as chunked frames to a peer that can take them (CAP_CHUNKED_FRAMES);
    # to any other peer (older nodes, browsers) the legacy limit holds, as for pubsub and Session payloads
    CHUNK_SIZE = 5 * 1024 * 1024  # Bytes of message per chunk frame on the wire (the largest one taken)
    SEND_CHUNK_SIZE = 1024 * 1024  # Bytes per chunk frame written (each must arrive within REASSEMBLY_TIMEOUT)
    MAX_MESSAGE_SIZE = 1024 * 1024 * 1024  # Bytes of one in-memory message, sent or received (0: no limit)
    LEGACY_MAX_MESSAGE_SIZE = 50 * 1024 * 1024  # Bytes of one message as a single frame (peers without chunks)
    REASSEMBLY_MAX_BYTES_PER_PEER = MAX_MESSAGE_SIZE  # Bytes under reassembly from one peer (0: MAX_MESSAGE_SIZE)
    REASSEMBLY_MAX_BYTES_TOTAL = 4 * 1024 * 1024 * 1024  # Bytes under reassembly from all the peers
    REASSEMBLY_TIMEOUT = 120.  # Seconds a message under reassembly may wait for its next chunk
    ACCEPT_FILES = True  # File messages (FILE data streamed from disk) are accepted, saved under DOWNLOADS_DIR
    MAX_FILE_SIZE = 0  # Bytes of one file, sent or received (0: no limit)
    DOWNLOADS_DIR = None  # Folder of the received files (None: <app folder>/Downloads, next to the key)
    MAX_IMAGE_EDGE = 1600  # Longest edge, in pixels, of a picture sent as an image (a larger one is shrunk to it)

    # =====================
    # ENVIRONMENT VARIABLES
    # =====================
    PRINT_LEVEL = int(os.getenv("NODE_PRINT", "0"))  # 0, 1
    LOG_TO_FILE = int(os.getenv("NODE_LOG", "0")) == 1  # 0, 1
    LOG_APPEND = int(os.getenv("NODE_LOG_APPEND", "0")) == 1  # 0, 1 (keep the JSONL across restarts)
    LOG_CHANNELS = os.getenv("NODE_LOG_CHANNELS", "").strip()
    PRINT_SCREEN_BASIC_ONLY = int(os.getenv("NODE_SCREEN_BASIC_PRINT", "0")) == 1  # 0, 1
    LOG_SINK_MODULE = os.getenv("NODE_LOG_SINK_MODULE", "").strip()  # import path of an optional log sink plugin
    SKIP_WAS_ALIVE_CHECK = os.getenv("NODE_IGNORE_ALIVE", "0") == "1"
    LIBP2PLOG = os.getenv("NODE_LIBP2PLOG", "0") == "1"
    ENV_IS_ISOLATED = os.getenv("NODE_IS_ISOLATED", "0") == "1"
    ENV_IS_PUBLIC = os.getenv("NODE_IS_PUBLIC", "0") == "1"
    ENV_IS_PUBLIC_RELAY = os.getenv("NODE_IS_PUBLIC_RELAY", "0") == "1"
    ENV_USE_TLS = os.getenv("NODE_USE_TLS", "0") == "1"
    ENV_START_PORT = int(os.getenv("NODE_STARTING_PORT", "0"))
    ENV_DOMAIN = os.getenv("DOMAIN", None)
    ENV_CERT_PATH = os.getenv("TLS_CERT_PATH", None)
    ENV_KEY_PATH = os.getenv("TLS_KEY_PATH", None)
    PATH_TO_APPEND_ADDRESSES = os.getenv("NODE_SAVE_RUNNING_ADDRESSES")

    # The processor (a ModuleWrapper) runs in a worker thread of its own during process/learn, so that the node keeps
    # running during a long forward or backward pass (0 to run it inline, as before)
    OFFLOAD_ENABLED = os.getenv("NODE_OFFLOAD", "1") == "1"

    # In a multistep interaction (samples one per tick) the processor runs inline, so that each step takes its own
    # sample, unless its last call took at least this many seconds: then it is offloaded too, so that the node keeps
    # running (its Sessions answer) during each long step
    OFFLOAD_SLOW_S = float(os.getenv("NODE_OFFLOAD_SLOW", "0.5"))

    # =============
    # STATE MACHINE
    # =============
    ROLE_WILDCARD = '<role>'
    WORLD_WILDCARD = '<world>'
    AGENT_WILDCARD = '<agent>'
    PARTNER_WILDCARD = '<partner>'
    DEFAULT_WILDCARDS = {
        WORLD_WILDCARD: WORLD_WILDCARD,
        AGENT_WILDCARD: AGENT_WILDCARD,
        PARTNER_WILDCARD: PARTNER_WILDCARD,
        ROLE_WILDCARD: ROLE_WILDCARD
    }

    # Logged ticks for action completion (unique markers, used also to avoid repetitions)
    ACTION_TICKS_PER_STATUS = ["   ✅ ", "   🔄 ", "   ❌ "]  # Keep the final spaces

    # =======
    # ACTIONS
    # =======
    # Special action arguments, that will be considered when using the @action decorator
    STREAM_ARG_NAMES = {'stream', 'streams'}  # Last one => INTERACTION_FIELD_NAMES too
    AGENT_ARG_NAMES = {'agent', 'agents', 'partner', 'partners',
                       'targets', 'target'}  # Last one => INTERACTION_FIELD_NAMES too

    # Special action argument names (when calling an action) that tells intrinsic information of the action and
    # are stipped from the argument list
    SECONDS_ARG_NAMES = ['max_duration']  # Keep the preferred name on top
    TIMEOUT_ARG_NAMES = ['retry_timeout']  # Keep the preferred name on top
    DELAY_ARG_NAMES = ['delay']  # Keep the preferred name on top
    TIME_TO_WAIT_BEFORE_ACTING_ARG_NAMES = ['time_to_wait_before_acting']  # State only

    # Special cases of action arguments
    """
    --------------------------------------------------------------------------------------------------------------------
    Role	            Names	                                Semantics
    --------------------------------------------------------------------------------------------------------------------
    INTERACTION_INJECT  {'interaction'}	                        Framework injects the Interaction object at call time.
                                                                Must be declarable in body sig (that's how injection
                                                                opts in). Forbidden in action.args and in wire
                                                                action_kwargs.
    INTERACTION_FIELD   {streams, num_steps, target, timeout,   Promoted from Action.args to a field of the Interaction
                        copy_sys, forced_uuid, id, volatile,    object by __build_system_interaction. Allowed in body
                        data_samples, callback}                 sig (then stays in action_kwargs).
                                                                Allowed in wire action_kwargs iff value equals the
                                                                matching Interaction field.
    HSM_TRANSIT_META	{max_duration, retry_timeout, delay}    Per-transit scalars consumed by __guess_* then stripped
                                                                from Action.args. Forbidden in body sig and in wire
                                                                action_kwargs.
    WIRE_SENTINEL       {'action_kwargs'}	                    Framework-private container name. Forbidden everywhere
                                                                except send's body sig.
    REGULAR	            everything else	                        Validated only against the body sig.
    --------------------------------------------------------------------------------------------------------------------
    """
    INTERACTION_INJECT_NAMES = {'interaction'}
    INTERACTION_FIELD_NAMES = {'streams', 'num_steps', 'target', 'timeout', 'copy_sys', 'forced_uuid', 'id', 'volatile',
                               'data_samples', 'callback', 'num_samples_to_stream'}
    HSM_TRANSIT_META_NAMES = {SECONDS_ARG_NAMES[0], TIMEOUT_ARG_NAMES[0], DELAY_ARG_NAMES[0]}
    WIRE_SENTINEL_NAMES = {'action_kwargs'}
    SPECIAL_ACTION_NAMES = {'send'}

    # Collectors
    RESERVED_IN_BODY_SIGNATURE = HSM_TRANSIT_META_NAMES | WIRE_SENTINEL_NAMES
    RESERVED_IN_ACTION_KWARGS = HSM_TRANSIT_META_NAMES | WIRE_SENTINEL_NAMES | INTERACTION_INJECT_NAMES

    # Others
    DEFAULT_TIMEOUT = 10.0
    SEND_TICKET_TIMEOUT = DEFAULT_TIMEOUT  # Seconds a ticket.wait() of a tracked send waits by default
    ALL_STATES_NAME = 'all'
    NOT_ALLOWED_STATE_NAMES = {ALL_STATES_NAME}

    # ============
    # INTERACTIONS
    # ============
    SYSTEM_INTERACTION_ID = "sys"
    SYSTEM_INTERACTION_LABEL = "system"
    SYSTEM_INTERACTION_UUID = SYSTEM_INTERACTION_ID + "_" + SYSTEM_INTERACTION_LABEL
    MAX_INTERACTIONS = 100
    MAX_STREAM_DATA_WITHOUT_INTERACTIONS = 10
    DEFAULT_INTER_TIMEOUT = 60. * 5.  # Seconds
    DRAIN_TIMEOUT = 0.
    FAKE_INTERACTION_UUID = "fake_system"


class GenException(Exception):
    """Base exception for this application (a simple wrapper around a generic Exception)."""
    pass


class RootServerError(GenException):
    """Failure of a root-server API call.

    api_rejected distinguishes an API-level refusal (the server answered and said no:
    retrying cannot change the outcome) from a transport failure (worth a retry).
    Refusals carry the response envelope: the state code and message, the data.code
    discriminator and the flags, so callers can branch on them.
    """

    def __init__(self, message: str, *, api: str, api_rejected: bool = False,
                 state_code: str | None = None, state_message: str | None = None,
                 data_code: str | None = None, data: dict | list | None = None,
                 flags: dict | None = None, status: int | None = None) -> None:
        """Initializes the error with the details of the failed root-server call.

        Args:
            message: The error message.
            api: The root-server API that was called.
            api_rejected: True if the server answered with a refusal, False for a transport failure (default: False).
            state_code: The state code of the response envelope (default: None).
            state_message: The state message of the response envelope (default: None).
            data_code: The data.code discriminator of the response (default: None).
            data: The data payload of the refusal (default: None).
            flags: The flags of the response (default: None, meaning no flags).
            status: The HTTP status code (default: None).
        """
        super().__init__(message)
        self.api = api
        self.api_rejected = api_rejected
        self.state_code = state_code
        self.state_message = state_message
        self.data_code = data_code
        self.data = data  # The refusal's data payload (e.g. the legal acceptance state)
        self.flags = flags if flags is not None else {}
        self.status = status
