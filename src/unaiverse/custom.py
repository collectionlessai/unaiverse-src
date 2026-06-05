import os


class Custom:

    # ==================
    # NODE CONFIGURATION
    # ==================
    SEND_DYNAMIC_PROFILE_EVERY = 10.  # Seconds
    GET_NEW_TOKEN_EVERY = 23 * 60. * 60. + 30 * 60.  # Seconds (23 hours and 30 minutes, safer)
    PUBLISH_RENDEZVOUS_EVERY = 10.  # Seconds
    INTERVIEW_TIMEOUT = 60.  # Seconds
    CONNECT_WITHOUT_ACK_RETRY_TIMEOUT = 30.  # Seconds
    CONNECT_WITHOUT_ACK_TOTAL_TIMEOUT = 60.  # Seconds
    SEND_ALIVE_EVERY = 2.5 * 60.  # Seconds
    SAVE_STATS_EVERY = 10.  # Seconds
    SEND_STATS_EVERY = 30.  # Seconds (warning: modified in Node constructor!)
    SAVE_CHECKPOINT_EVERY = -1.  # When negative, means "do not save" (warning: modified by means in Node constructor!)
    SLOW_DOWN_CLOCK_AFTER = 10.0  # Seconds without networks exchanges and 0 interaction (stats are not considered)
    SLOW_CLOCK_DELTA = 2.0  # Clock delta when running in slow-mode

    # =====================
    # ENVIRONMENT VARIABLES
    # =====================
    PRINT_LEVEL = int(os.getenv("NODE_PRINT", "0"))  # 0, 1
    LOG_TO_FILE = int(os.getenv("NODE_LOG", "0")) == 1  # 0, 1
    PRINT_SCREEN_BASIC_ONLY = int(os.getenv("NODE_SCREEN_BASIC_PRINT", "0")) == 1  # 0, 1
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
                               'data_samples', 'callback'}
    HSM_TRANSIT_META_NAMES = {SECONDS_ARG_NAMES[0], TIMEOUT_ARG_NAMES[0], DELAY_ARG_NAMES[0]}
    WIRE_SENTINEL_NAMES = {'action_kwargs'}
    SPECIAL_ACTION_NAMES = {'send'}

    # Collectors
    RESERVED_IN_BODY_SIGNATURE = HSM_TRANSIT_META_NAMES | WIRE_SENTINEL_NAMES
    RESERVED_IN_ACTION_KWARGS = HSM_TRANSIT_META_NAMES | WIRE_SENTINEL_NAMES | INTERACTION_INJECT_NAMES

    # Others
    DEFAULT_TIMEOUT = 10.0
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
