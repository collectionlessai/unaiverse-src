import os


class Custom:

    # Custom configurations, default values
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

    # Interactions
    MAX_INTERACTIONS = 100
    MAX_STREAM_DATA_WITHOUT_INTERACTIONS = 2

    # Environment variable
    PRINT_LEVEL = int(os.getenv("NODE_PRINT", "0"))  # 0, 1, 2
    LOG_TO_FILE = int(os.getenv("NODE_LOG", "0")) == 1  # 0, 1
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

    # Misc
    ACTION_TICKS_PER_STATUS = ["   ✅ ", "   🔄 ", "   ❌ "]  # Keep the final spaces

    # Arguments
    STREAM_ARG_NAMES = {'stream', 'streams', 'u_hashes', 'yhat_hashes'}
    AGENT_ARG_NAMES = {'agent', 'agents', 'partner', 'partners'}

    # HSM wildcards
    DEFAULT_WILDCARDS = {'<world>': '<world>', '<agent>': '<agent>', '<partner>': '<partner>', '<role>': '<role>'}

    # Candidate Action argument names (when calling an action) that tells that such an action is multi-steps
    SECONDS_ARG_NAMES = {'time'}
    TIMEOUT_ARG_NAMES = {'timeout'}
    DELAY_ARG_NAMES = {'delay'}
    NOT_ALLOWED_IN_ACTION_SIGNATURE = SECONDS_ARG_NAMES | TIMEOUT_ARG_NAMES | DELAY_ARG_NAMES
    INTERACTION_ARG_NAMES = {'interaction', '_requester'}
    DEFAULT_TIMEOUT = 10.0

    # Deprecated Action argument-related things
    SPECIAL_DEPRECATED_CASES_PREFIXES = {'ask_', 'do_', 'done_'}
    NOT_READY_PREFIXES = {'get_', 'got_', 'do_', 'done_'}

    # Interactions
    DEFAULT_INTER_TIMEOUT = 60. * 5.  # Seconds
    DRAIN_TIMEOUT = 0.
