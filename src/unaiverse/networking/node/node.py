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
import sys
import cv2
import copy
import json
import math
import time
import queue
import types
import signal
import asyncio
import requests
import threading
import traceback
from PIL import Image
from typing import Any
from datetime import timedelta
from unaiverse.clock import clock
from unaiverse.world import World
from unaiverse.agent import Agent
from collections.abc import Callable
from datetime import datetime, timezone
from unaiverse.networking.p2p.messages import Msg
from unaiverse.custom import Custom, GenException, RootServerError
from unaiverse.networking.p2p import P2P, P2PError
from unaiverse.networking.node.connpool import NodeConn
from unaiverse.networking.node.profile import NodeProfile
from unaiverse.networking.session import (ProfileDecisionResult, WorldGrantSpec, VerifiedTokenClaims,
                                          SessionEventType, SessionRole)
from unaiverse.networking.session.sess_msg import (WORLD_DISENGAGEMENT,
                                                   LOCAL_ERROR, NOT_IN_ALLOW_LIST, ROLE_INCOMPATIBLE,
                                                   HOSTED_REJECTED, CV_HASH_MISMATCH, NOT_AUTHORIZED, GRACEFUL_SHUTDOWN)
from unaiverse.streams.streams import DataProps, BufferedStream
from unaiverse.utils.logger import log, ALWAYS_ON_CHANNELS, ALL_CHANNELS
from unaiverse.utils.misc import (get_key_considering_multiple_sources, save_node_addresses_to_file,
                                  prepare_app_dir, load_agent_in_memory, unpack_py_files, analyze_code,
                                  world_definition_members, canonical_world_hash, owner_handle)

# Margin added past a quarantine ban when the rejoin watchdog pulls a retry in:
# just enough to land on the open side of the gate.
_REJOIN_RETRY_MARGIN_S = 1.0


class Node:
    # Each node can host an agent or a world
    AGENT = "agent"  # Artificial agent
    WORLD = "world"  # World agent

    _closed = False  # Set by aclose, which must run once (class-level: no __init__ needed)

    def __init__(self,
                 hosted: Agent | World,
                 unaiverse_key: str | None = None,
                 node_name: str | None = None,
                 node_id: str | None = None,
                 hidden: bool | None = None,
                 code_repo: str | None = None,
                 code_commit: str | None = None,
                 clock_delta: float = 1. / 25.,
                 base_identity_dir: str | None = None,
                 only_certified_agents: bool = False,
                 allowed_node_ids: list[str] | set[str] | None = None,  # Optional: it is loaded from online profile
                 world_masters_node_ids: list[str] | set[str] | None = None,  # Optional: it is loaded from profile
                 world_masters_node_names: list[str] | set[str] | None = None,  # Optional: then converted to node IDs
                 allow_connection_through_relay: bool = True,
                 talk_to_relay_based_nodes: bool = True,
                 run_hook: Callable[['Node'], None] | None = None,
                 send_stats_every: float = 30.,
                 save_checkpoint_every: float = -1.):
        """Initializes a new instance of the Node class.

        Args:
            hosted: The Agent or World entity hosted by this node.
            unaiverse_key: The UNaIVERSE key for authentication (if None, it will be loaded from env var or cache file,
                or you will be asked for it).
            node_name: A human-readable name for the node (using node ID is preferable; use this or node ID, not both).
            node_id: A unique identifier for the node (use this or the node name, not both).
            hidden: A flag to determine if the node is hidden (i.e., only the owner of the account can
                see it). When None (the default) the visibility is not asserted at all, so whatever
                the owner chose (e.g. in the web app) is preserved.
            code_repo: World only: the public GitHub repository holding this world's published code,
                as "owner/repo" or the full https://github.com/owner/repo URL. The root fetches it,
                folds the canonical bundle and attests its digest in the node token. When absent the
                world declares nothing and runs unpublished (joins work exactly as before).
            code_commit: World only: the full 40-character id of the (pushed) commit to declare,
                not a branch and not a tag.
            clock_delta: The minimum time delta for the node's clock.
            base_identity_dir: Base directory for storing node identity files. If None, uses the default app directory.
            only_certified_agents: A flag to allow only certified agents to connect.
            allowed_node_ids: A list or set of allowed node IDs to connect (t is loaded from the online profile).
            world_masters_node_ids: A list or set of world masters' node IDs (it is also loaded from online profile).
            world_masters_node_names: A list or set of world masters' node names (using IDs is preferable).
            allow_connection_through_relay: A flag to allow connections through a relay.
            talk_to_relay_based_nodes: A flag to allow talking to relay-based nodes.
            run_hook: A function taking the Node instance as argument, called every cycle.
            send_stats_every: Send the stats update to the world every N seconds.
            save_checkpoint_every: Time interval in seconds to save the hosted entity's state to disk (< 0 not to save).
        """

        # Creating the basic objects/setting the basic references
        self.hosted = hosted
        self.clock = clock  # Backward compatibility
        try:
            clock.create(min_delta=clock_delta)  # Node clock (synch by NTP servers)
        except ValueError as e:
            go_ahead = False
            while not go_ahead:
                user_choice = input("Proceed with local time (strongly NOT suggested)? (y/n) ")
                if user_choice.strip().lower() == 'y':
                    print("Proceeding with local time.")
                    go_ahead = True
                elif user_choice.strip().lower() == 'n':
                    raise e
            clock.create(min_delta=clock_delta,
                         current_time=datetime.now(timezone.utc).timestamp())  # Node clock (not synced at all!)

        # Logging: we create a new logger and will share it with the agent/world as well
        # "active" is the set of recorded channels, "screen" the subset also printed to stdout: they are
        # independent knobs, so NODE_SCREEN_BASIC_PRINT must be honored even when NODE_LOG_CHANNELS selects
        # a custom recording set.
        if Custom.LOG_CHANNELS:
            _valid = {c.name: c for c in ALL_CHANNELS}
            _wanted = frozenset(_valid[n] for n in
                                (s.strip().upper() for s in Custom.LOG_CHANNELS.split(",")) if n in _valid)
            active = ALWAYS_ON_CHANNELS | _wanted
        else:
            active = ALWAYS_ON_CHANNELS if Custom.PRINT_LEVEL < 1 else ALL_CHANNELS
        screen = ALWAYS_ON_CHANNELS if Custom.PRINT_SCREEN_BASIC_ONLY else active
        log.create(name=os.path.basename(sys.argv[0]), log_dir=os.path.dirname(os.path.abspath(sys.argv[0])),
                   active=active, screen=screen, no_color=False, file_enabled=Custom.LOG_TO_FILE,
                   file_append=Custom.LOG_APPEND)
        log.set_clock(clock)  # Wiring the clock

        # Optional log sink plugin: an out-of-core module (resolved on PYTHONPATH) that observes the log
        # stream, wired in through NODE_LOG_SINK_MODULE. It only needs to expose attach(log). Any failure
        # here is logged and swallowed: an observability add-on must never abort the node boot.
        if Custom.LOG_SINK_MODULE:
            try:
                import importlib
                importlib.import_module(Custom.LOG_SINK_MODULE).attach(log)
            except Exception as e:
                log.error(f"Failed to attach log sink module '{Custom.LOG_SINK_MODULE}': {e}")

        # Checking main arguments
        if not (isinstance(hosted, Agent) or isinstance(hosted, World)):
            log.critical("Invalid hosted entity, must be Agent or World")
        if not (node_id is None or isinstance(node_id, str)):
            log.critical("Invalid node ID")
        if not (node_name is None or isinstance(node_name, str)):
            log.critical("Invalid node name")
        if not (node_name is None or node_id is None):
            log.critical("Cannot specify both node ID and node name")
        if not (node_name is not None or node_id is not None):
            log.critical("You must specify either node ID or node name: both are missing")
        if not (unaiverse_key is None or isinstance(unaiverse_key, str)):
            log.critical("Invalid UNaIVERSE key")

        # Main attributes
        self.node_id = node_id
        self.run_hook = run_hook
        self.unaiverse_key = unaiverse_key
        self.code_repo = code_repo
        self.code_commit = code_commit
        self.node_type = Node.AGENT if (isinstance(hosted, Agent) and not isinstance(hosted, World)) else Node.WORLD
        self.agent: Agent | None = hosted if isinstance(hosted, Agent) else None
        self.world: World | None = hosted if isinstance(hosted, World) else None
        self.conn: NodeConn  # Manages the network operations in the P2P network
        # The two SessionManagers are owned by the Coordinator (conn.attach_session_policy);
        # Node holds no handle to them and drives every Session verb through conn facades.
        # The world-handoff correlation (public<->world peer-id) now lives in the
        # Coordinator (conn.start_world_handoff / rollback_handoff / complete_world_handoff);
        # Node drives the handoff through those facades and holds no maps.
        # Interactive-mode options dict (built in run() when interact_mode is on),
        # surfaced on self so the SessionEvent pump can flip ready_to_interact on
        # handshake completion. None outside interactive mode.
        self._interact_mode_opts: dict | None = None
        self.memory_finder = None
        self.world_agent_files = None
        self.world_role_fsms: dict | None = None  # Role -> FSM (JSON string), from the world's grant bundle
        self.talk_to_relay_based_nodes = talk_to_relay_based_nodes
        self.keep_rejoining = False
        self.rejoining_kwargs: dict = {}
        self.rejoining_time = -1.

        # Expected properties of the nodes that will try to connect to this one
        self.only_certified_agents = only_certified_agents
        self.allowed_node_ids = set(allowed_node_ids) if allowed_node_ids is not None else None
        self.world_masters_node_ids = set(world_masters_node_ids) if world_masters_node_ids is not None else None

        # Profile
        self.profile: NodeProfile

        # Rendezvous
        self.last_rendezvous_time = 0.

        # Automatic address update and relay refresh (if needed)
        self.relay_reservation_expiry: datetime | None = None

        # Interview of newly connected nodes
        self.reconnected = set()

        # Alive messaging
        self.last_alive_time = 0.

        # stats reporting agent -> world
        Custom.SEND_STATS_EVERY = send_stats_every

        # Save agent state
        Custom.SAVE_CHECKPOINT_EVERY = save_checkpoint_every

        # Alive messaging
        self.run_start_time = 0.

        # Root server-related
        self.root_endpoint = 'https://unaiverse.io:8443/api'  # WARNING: EDITING THIS ADDRESS VIOLATES THE LICENSE
        self.node_token = ""
        self.public_key = ""
        self.turn_info = None  # TURN credentials from the node token, see inject_turn_credentials

        # Attributes: handshake-related. The legacy interview / ack / ping-pong
        # structures are gone — the per-peer Session FSM (SessionManager) now owns
        # establishment state, retry, and reconnection. See networking/session/.
        self.joining_world_info = None
        self.first = True
        self.stop_requested = False

        # Inspector related
        self.inspector_activated = False
        self.inspector_peer_id = None
        self.__inspector_cache = {"behav": None, "known_streams_count": 0, "all_agents_count": 0}
        self.__inspector_told_to_pause = False

        # Get key
        self.unaiverse_key = get_key_considering_multiple_sources(self.unaiverse_key)

        # Getting node ID (retrieving by name), if it was not provided (the node is created if not existing)
        if self.node_id is None:
            if node_name is None:
                log.critical("You must specify the name of the node (or provide the node ID, up to you)")
            assert node_name is not None
            node_ids, were_alive = self.get_node_id_by_name([node_name],
                                                            create_if_missing=True)
            self.node_id = node_ids[0]
            if were_alive[0] and not Custom.SKIP_WAS_ALIVE_CHECK:
                log.critical(f"Cannot access node {node_name}, it is already running! "
                             f"(set env variable NODE_IGNORE_ALIVE=1 to ignore this control)")

        # Automatically create a unique data directory for this specific node
        if base_identity_dir is None:
            base_identity_dir = prepare_app_dir(app_name="unaiverse")
        self.node_identity_dir = os.path.join(base_identity_dir, self.node_id)
        p2p_u_identity_dir = os.path.join(self.node_identity_dir, "p2p_public")
        p2p_w_identity_dir = os.path.join(self.node_identity_dir, "p2p_private")

        # Getting node ID of world masters, if needed
        if world_masters_node_names is not None and len(world_masters_node_names) > 0:
            master_node_ids, were_alive = self.get_node_id_by_name(world_masters_node_names,
                                                                   create_if_missing=True, node_type=Node.AGENT)
            for master_node_name, master_node_id in zip(world_masters_node_names, master_node_ids):
                if master_node_id is None:
                    log.critical(f"Cannot find world master node ID given its name: {master_node_name}")
                else:
                    if self.world_masters_node_ids is None:
                        self.world_masters_node_ids = set()
                    self.world_masters_node_ids.add(master_node_id)

        # Here you can set up max_instances, max_channels, enable_logging at libp2p level etc.
        P2P.setup_library(enable_logging=Custom.LIBP2PLOG, unai_logger=log)

        # --- PARALLEL P2P NODE CREATION ---
        # 1. Define configurations for both nodes
        p2p_u_config = {
            "identity_dir": p2p_u_identity_dir,
            "port": Custom.ENV_START_PORT,
            "ips": None,
            "enable_relay_client": allow_connection_through_relay,
            "enable_relay_service": Custom.ENV_IS_PUBLIC_RELAY,
            "use_broad_limits": False,
            "is_isolated": Custom.ENV_IS_ISOLATED,
            "knows_is_public": Custom.ENV_IS_PUBLIC,
            "flood_sub": False,
            "enable_tls": Custom.ENV_USE_TLS,
            "domain_name": Custom.ENV_DOMAIN,
            "tls_cert_path": Custom.ENV_CERT_PATH,
            "tls_key_path": Custom.ENV_KEY_PATH,
            "log_sub": "pub",
            "webrtc_enabled": True,
            "ice_stun_servers": None,
            "ice_turn_servers": None,
        }

        p2p_w_config = {
            "identity_dir": p2p_w_identity_dir,
            "port": (Custom.ENV_START_PORT + 4) if Custom.ENV_START_PORT > 0 else 0,
            "ips": None,
            "enable_relay_client": allow_connection_through_relay,
            "enable_relay_service": self.node_type is Node.WORLD,
            "use_broad_limits": True,
            "is_isolated": Custom.ENV_IS_ISOLATED,
            "knows_is_public": Custom.ENV_IS_PUBLIC,
            "flood_sub": True,
            "enable_tls": Custom.ENV_USE_TLS,
            "domain_name": Custom.ENV_DOMAIN,
            "tls_cert_path": Custom.ENV_CERT_PATH,
            "tls_key_path": Custom.ENV_KEY_PATH,
            "log_sub": "prv",
            "webrtc_enabled": True,
            "ice_stun_servers": None,
            "ice_turn_servers": None,
        }

        # 2. Prepare a dictionary to store results or exceptions
        results: dict[str, P2P | Exception | None] = {
            "p2p_u": None,
            "p2p_w": None
        }

        # 3. Define the worker function for the threads
        def create_p2p_instance(name: str, config: dict):
            try:
                # This is the slow, blocking call
                instance = P2P(**config)
                results[name] = instance
            except Exception as _e:
                # Store the exception if creation fails
                results[name] = _e
            return True

        # 4. Create and start both threads
        thread_u = threading.Thread(target=create_p2p_instance, args=("p2p_u", p2p_u_config))
        thread_w = threading.Thread(target=create_p2p_instance, args=("p2p_w", p2p_w_config))

        thread_u.start()
        thread_w.start()

        # 5. Wait for both threads to complete
        # This BLOCKS the __init__ method until both are done.
        thread_u.join()
        thread_w.join()

        # 6. Retrieve results and check for errors
        p2p_u: P2P | Exception | None = results["p2p_u"]
        p2p_w: P2P | Exception | None = results["p2p_w"]

        if isinstance(p2p_u, Exception):
            # We must re-raise the exception to fail the Node creation
            raise P2PError(f"Failed to initialize public P2P node (p2p_u): {p2p_u}") from p2p_u
        if isinstance(p2p_w, Exception):
            raise P2PError(f"Failed to initialize private P2P node (p2p_w): {p2p_w}") from p2p_w
        if p2p_u is None or p2p_w is None:
            # This should not happen if threads ran, but it's a safe check
            raise P2PError("P2P node creation did not complete, but no exception was caught.")
        if p2p_u.peer_id is None or p2p_w.peer_id is None:
            raise P2PError("P2P node creation did not correctly complete.")

        # Get first node token. This is the first root call that can succeed: it is the
        # only one that bootstraps on the account key (see _request_node_token), while
        # every other route wants a node token we do not have yet. It needs the peer-ids,
        # which is why it waits for the transports.
        public_peer_id = p2p_u.peer_id
        private_peer_id = p2p_w.peer_id
        assert public_peer_id is not None
        assert private_peer_id is not None
        self.get_node_token(peer_ids=[public_peer_id, private_peer_id])  # Passing both the peer IDs

        # Get profile (static), authenticated by the token we just stored
        profile_static = self.__root(api="/account/node/profile/static/get", payload={"node_id": self.node_id})
        if not isinstance(profile_static, dict):
            log.critical(f"Unexpected static-profile payload from the root server: {profile_static}")

        # Getting list of allowed nodes from the static profile,
        # if we did not already specify it when creating the node in the code (the code has higher priority)
        if (self.allowed_node_ids is None and 'allowed_node_ids' in profile_static and
                profile_static['allowed_node_ids'] is not None and len(profile_static['allowed_node_ids']) > 0):
            self.allowed_node_ids = set(profile_static['allowed_node_ids'])

        # Getting list of world master nodes from the static profile,
        # if we did not already specify it when creating the node in the code (the code has higher priority)
        if self.node_type is Node.WORLD:
            if (self.world_masters_node_ids is None and 'world_masters_node_ids' in profile_static and
                    profile_static['world_masters_node_ids'] is not None
                    and len(profile_static['world_masters_node_ids']) > 0):
                self.world_masters_node_ids = set(profile_static['world_masters_node_ids'])
        else:
            self.world_masters_node_ids = None  # Clearing this in case the user specified it for a non-world node

        # Creating the connection manager
        # guessing max number of connections (max number of valid
        # the connection manager will ensure that this limit is fulfilled)
        # however, the actual number of connection attempts handled by libp2p must be higher that
        self.conn = NodeConn(max_connections=profile_static['max_nr_connections'],
                             is_world_node=self.node_type is Node.WORLD,
                             public_key=self.public_key,
                             token=self.node_token,
                             local_node_id=self.node_id)

        # Get first badge token
        if self.node_type is Node.WORLD:
            self.badge_token = self.__root(api="/account/node/cv/badge/token/get", payload={"node_id": self.node_id})
        else:
            self.badge_token = None

        # The Coordinator (NodeConn) owns the two per-instance SessionManagers: it
        # builds them from its own P2P handles and the cross-instance capability, and
        # only needs the Node policy that is born late — the profile factory (self.
        # profile is set further down) and the responder decision hook. The Coordinator
        # owns the two managers; Node keeps no handle and drives every Session verb through
        # conn facades, and conn consults them on behalf of the hosted entity (e.g.
        # is_peer_session_pending).
        # The responder decision policy is late-bound through self._decision_policy so a
        # runtime set_decision_policy() propagates to every Session, existing or future:
        # the hook injected into each Session is a thin lambda that re-reads the current
        # policy on every call, instead of the bound method frozen once at attach time.
        self._decision_policy = self._session_decision_hook
        self.conn.attach_session_policy(
            p2p_u=p2p_u,
            p2p_w=p2p_w,
            local_profile_factory=lambda: (self.profile.get_all_profile() if self.profile is not None else {}),
            decision_hook=lambda *a: self._decision_policy(*a),
            token_factory=lambda: self.node_token,
        )

        # The managers exist now, so the TURN credentials that arrived with the token can
        # finally reach the two transports. This is the one point in the boot where that
        # is possible: earlier there is nothing to inject into, later the instances would
        # already be dialling without them.
        self.inject_turn_credentials()

        # Get CV
        cv = self.get_cv()

        # Creating full node profile putting together static info, dynamic profile, adding P2P node info, CV
        self.profile = NodeProfile(static=profile_static,
                                   dynamic={'peer_id': p2p_u.peer_id,
                                            'peer_addresses': p2p_u.addresses,
                                            'private_peer_id': p2p_w.peer_id,
                                            'private_peer_addresses': p2p_w.addresses,
                                            # A world node is a relay for its members only if its world instance is
                                            # publicly reachable (the relay service is enabled iff node_type is WORLD).
                                            # TODO: The live AutoNAT verdict can change later, but the profile field is
                                            # seeded once here from the boot verdict and never refreshed.
                                            'is_relay': (self.node_type is Node.WORLD) and bool(p2p_w.is_public),
                                            'connections': {
                                                'role': self.hosted.ROLE_BITS_TO_STR[self.hosted.ROLE_PUBLIC]
                                            },
                                            'world_summary': {
                                                'world_title':
                                                    profile_static['node_name']
                                                    if self.node_type is Node.WORLD else None
                                            },
                                            "hidden": hidden  # True/False only when explicitly requested
                                            #  (None means: do not touch the server-side visibility)
                                            },
                                   cv=cv)  # Adding CV here

        # Sharing node-level info with the hosted entity. The trailing dict was the
        # legacy "agents waiting for ack" queue (never read by the hosted entity);
        # the Session FSM owns that state now, so we pass an empty placeholder.
        self.hosted.set_node_info(self.conn, self.profile, self.ask_to_get_in_touch,
                                  self.__purge, self.node_identity_dir)

        # World only: optionally declare where this world's code lives (a GitHub repo
        # and commit) so the root can attest its digest, and post the role FSMs for the
        # website. The token is then re-requested so it is minted AFTER the declaration
        # (the code_hash claim is read from the stored declaration at mint time).
        if self.node_type is Node.WORLD:
            self.declare_world_source()
            self.get_node_token(peer_ids=[self.get_public_peer_id(), self.get_world_peer_id()])

        # Finally, sending dynamic profile to the root server
        # (send AFTER set_node_info, not before, since set_node_info updates the
        # profile with the processor/stream info)
        self.send_dynamic_profile()

        # Save public addresses
        if Custom.PATH_TO_APPEND_ADDRESSES is not None and os.path.exists(Custom.PATH_TO_APPEND_ADDRESSES):
            save_node_addresses_to_file(self, public=True, dir_path=Custom.PATH_TO_APPEND_ADDRESSES,
                                        filename="running.csv", append=True)

        # Update lone-wolf machines to replace default wildcards (like <agent>) - the private one will be handled when
        # joining a world
        if self.node_type is Node.AGENT:
            assert self.agent is not None
            self.agent.behav_lone_wolf.update_wildcard(Custom.AGENT_WILDCARD, f"{self.get_public_peer_id()}")
            self.agent.behav_lone_wolf.apply_wildcards()

    def get_node_id_by_name(self, node_names: list[str] | set[str], create_if_missing: bool = False,
                            node_type: str | None = None) -> tuple[list[str], list[bool]]:
        """Retrieves the node ID by its name from the root server, creating a new node if it's missing and specified.

        Args:
            node_names: The list (or set) with the names of the nodes to retrieve.
            create_if_missing: A flag to create the node if it doesn't exist (only valid for your own nodes).
            node_type: The type of the node to create if missing (when create_if_missing is True) - default: the type of
                the current node.

        Returns:
            The list of node IDs and the list of boolean flags telling if a node was already alive,
            or an exception if an error occurs.
        """
        missing = []
        node_ids = []
        were_alive = []
        node_names = node_names if isinstance(node_names, list) else list(node_names)
        try:
            response = self.__root("/account/node/get/id",
                                   payload={
                                       "node_name": node_names,
                                       "account_token": self.unaiverse_key})
            for i in range(0, len(response["nodes"])):
                if response["nodes"][i] is not None:
                    node_ids.append(response["nodes"][i]["node_id"])
                    were_alive.append(response["nodes"][i]["was_alive"])
                else:
                    node_ids.append(None)
                    were_alive.append(None)
                    missing.append(i)
        except Exception as e:
            log.critical(f"Error while retrieving nodes named {node_names} from server! [{e}]")

        if create_if_missing:
            for i in missing:
                node_name = node_names[i]
                if "/" in node_name or "@" in node_name:  # Cannot create nodes belonging to others
                    continue
                try:
                    response = self.__root("/account/node/fast_register",
                                           payload={"node_name": node_name,
                                                    "node_type": self.node_type if node_type is None else node_type,
                                                    "account_token": self.unaiverse_key})
                    node_ids[i] = response["node_id"]
                    were_alive[i] = False
                except Exception as e:
                    log.critical(f"Error while registering node named {node_name} in server! [{e}]")
        return node_ids, were_alive

    def send_alive(self, alive: bool = True) -> bool:
        """Send an alive message to the root server.

        Args:
            alive (bool): Whether to communicate that the node is alive (i.e., if this is False, it means the
                node is simpy not alive anymore (dead)).

        Returns:
            A boolean flag indicating whether the node was already live before sending this.
        """
        try:
            response = self.__root("/account/node/alive",
                                   payload={"node_id": self.node_id,
                                            "account_token": self.unaiverse_key,
                                            "alive": alive})
            return response["was_alive"]
        except Exception as e:
            log.error(f"Error while sending alive message to server! [{e}]")
            return False

    def get_node_token(self, peer_ids: list[str]) -> None:
        """Generates and retrieves a node token from the root server, then applies it.

        Fatal on failure (the fetch raises): a node with no token cannot run, which is the
        wanted behaviour at startup. In the main loop the fetch and then apply are split so
        the fetch can run off the event loop under a timeout (see the periodic-refresh
        call-site), without a late thread mutating the token/transport concurrently.

        Args:
            peer_ids: A list of public and private peer IDs.
        """
        self._apply_node_token(self._request_node_token(peer_ids))

    @staticmethod
    def _outstanding_legal_documents(data) -> str:
        """Human-readable list of the legal documents still to accept, read from the
        acceptance state the root attaches to a legal_acceptance_required refusal."""
        outstanding = []
        state = data if isinstance(data, dict) else {}
        for kind, label in (("tos", "Terms of Service"), ("privacy", "Privacy Policy")):
            side = state.get(kind)
            if isinstance(side, dict) and not side.get("up_to_date", False):
                version = side.get("current_version")
                outstanding.append(label + (f" {version}" if version else ""))
        return ", ".join(outstanding) if outstanding else "Terms of Service, Privacy Policy"

    def _request_node_token(self, peer_ids: list[str]) -> dict:
        """Blocking HTTP to the root to (re)generate the node token. Reads self but does
        NOT mutate it, so it is safe to run off the event loop via asyncio.to_thread.
        Raises on failure (fatal at startup, caught at the periodic-refresh call-site)."""
        response = None

        for i in range(0, 3):  # It will try 3 times before raising the exception...
            try:
                response = self.__root("/account/node/token/generate",
                                       payload={"node_id": self.node_id,
                                                "account_token": self.unaiverse_key
                                                if self.node_token is None or len(self.node_token) == 0 else None,
                                                "node_token": self.node_token, "peer_ids": json.dumps(peer_ids)})
                break
            except RootServerError as e:
                if e.api_rejected:
                    # The server answered and refused: retrying cannot change the outcome
                    if e.flags.get('legal_acceptance_required'):
                        log.error("The root server refuses to mint a node token until the owner accepts "
                                  f"the current legal documents ({self._outstanding_legal_documents(e.data)}): "
                                  "log in at https://unaiverse.io, accept them, then restart the node")
                    elif e.flags.get('blocked_utc') is not None:
                        log.error("The root server refuses to mint a node token: this node is blocked "
                                  f"(since {e.flags['blocked_utc']})")
                    raise
                if i < 2:
                    log.error("Error while getting token from server, retrying...")
                    time.sleep(1)  # Wait a little bit
                else:
                    log.error(f"Error while getting token from server [{e}]")
                    raise

        if not isinstance(response, dict):
            log.critical(f"Unexpected token payload from the root server: {response}")
        return response

    def _apply_node_token(self, response: dict) -> None:
        """Apply a freshly fetched node token. Main-task only: mutates self and pushes the
        token into the transport via FFI, so it must not run concurrently with the loop.

        Storing the token and injecting the TURN credentials are deliberately two steps.
        The credentials ride the token, but they can only reach a transport through the
        SessionManagers, which are built last (attach_session_policy), while the token
        itself is needed immediately: the static profile is fetched with it, and NodeConn
        is constructed around it. So this stores what it can and remembers the rest for
        inject_turn_credentials(), which the boot sequence calls once the managers exist.
        On the periodic refresh both already exist, and the injection happens right here.
        """
        self.node_token = response["token"]
        self.public_key = response["public_key"]
        self.turn_info = response.get("turn_info", None)

        # Sharing the token with the connection manager (absent this early at boot)
        if getattr(self, 'conn', None) is not None:
            self.conn.set_token(self.node_token)
            self.inject_turn_credentials()

    def inject_turn_credentials(self) -> None:
        """Hand the TURN credentials that came with the node token to both transports.

        Goes through the Coordinator (Node -> NodeConn -> SM -> P2P), which fans them out:
        the raw transport handles stay custody-only behind the SessionManagers, so this
        must not run before attach_session_policy has built them.

        A no-op when the root sent no turn_info, which is the normal case for a deployment
        that runs no TURN server: the node then relies on STUN and the relays alone. That
        silence is the reason the precondition below is an assert and not another quiet
        return: called too early it would look exactly like "no TURN configured", and a
        node running without TURN only shows up much later, as one pair of peers behind
        symmetric NATs failing to connect.
        """
        assert getattr(self, 'conn', None) is not None, \
            "inject_turn_credentials() runs after attach_session_policy: nothing to inject into yet"
        if self.turn_info is None:
            return
        self.conn.set_turn_credentials(ice_turn_servers=[{
            "urls": self.turn_info["urls"],
            "username": self.turn_info["username"],
            "credential": self.turn_info["credential"],
        }])

    def get_cv(self) -> list[dict]:
        """Retrieves the node's CV (Curriculum Vitae) from the root server.

        Returns:
            The node's CV as a list of dictionaries dictionary.
        """
        ret = None
        for i in range(0, 3):  # It will try 3 times before raising the exception...
            try:
                ret = self.__root(api="/account/node/cv/get", payload={"node_id": self.node_id})
                break
            except RootServerError as e:
                if e.api_rejected:
                    raise  # The server answered and refused: retrying cannot change the outcome
                log.error(f"Error while getting CV from server [{e}]")
                if i < 2:
                    log.misc("Retrying...")
                    time.sleep(1)  # Wait a little bit
                else:
                    raise

        if not isinstance(ret, list):
            log.critical(f"Unexpected CV payload from the root server: {ret}")
        return ret

    def declare_world_source(self) -> None:
        """Posts the role FSMs and optionally declares where this world's code lives (world only).

        Attestation only: the code still travels to joiners inside the world grant. The
        root fetches the declared commit from GitHub, folds the canonical bundle
        (code/ + fsm/ at the repository root) and stores its digest, which is minted
        into the node token as the code_hash claim; the joiner-side gate compares the
        grant bundle against that claim. With no code_repo/code_commit configured the
        world declares nothing and runs unpublished, exactly as before.

        Fatal when a declaration was requested and cannot be completed, or does not
        match the code this world is about to hand out: a stale or wrong attested
        digest would fail every join, on machines whose operators cannot see the cause.
        """
        assert self.world is not None

        # Structured role FSMs for the website: independent of the attestation, best-effort
        try:
            if self.world.role_to_behav:
                self.__root(api="/account/node/fsm/post",
                            payload={"node_id": self.node_id,
                                     "world_roles_fsm": self.world.role_to_behav})
            else:
                # The API refuses an empty world_roles_fsm, and an empty dict does
                # not mean "no roles" to it: simply skip the post
                log.debug("No world roles to post (role_to_behav is empty), skipping fsm/post")
        except Exception as e:
            log.error(f"Error while posting the world role FSMs to the root server [{e}]")

        # If there is no code repo and no code commit, we do not touch what is already there
        # (maybe it was set by the GUI).
        if self.code_repo is not None or self.code_commit is not None:
            if self.code_repo is None or self.code_commit is None:
                # Exactly one of the two is set: a declaration was clearly requested, and
                # silently running unpublished would defeat the author's intent
                missing = "code_commit" if self.code_commit is None else "code_repo"
                msg = (f"A world-source declaration was requested but {missing} is missing: "
                       f"pass both code_repo and code_commit, or neither")
                log.error(msg)
                raise GenException(msg)

            # Operator-facing hints for the refusal discriminators of code/source/set:
            # each is a distinct author mistake, and the message should say what to fix
            hints = {
                "commit_invalid": "declare the full 40-character commit id (not a branch, not a tag)",
                "commit_unreachable": "the commit is not reachable on GitHub: push it, and make sure "
                                      "the repository is publicly readable without credentials",
                "repo_no_bundle": "the repository must carry the canonical layout at its root: code/ "
                                  "for the shipped sources and fsm/ for the role FSMs",
                "repo_bad_member": "only plain files are allowed inside code/ and fsm/ "
                                   "(no symlinks, no submodules)",
                "repo_too_large": "the repository exceeds the server limits on the code/ and fsm/ trees",
                "repo_unavailable": "the API host has no git available to read repositories",
                "repo_timeout": "the server timed out reading the repository: try again later",
            }

            ret = None
            for i in range(0, 3):  # Retrying transport failures (and server-side git timeouts) only
                try:
                    # The caller_* pair is what authenticates this call (the node_token that
                    # __root injects is not read by this endpoint's decorator)
                    ret = self.__root(api="/account/node/code/source/set",
                                      payload={"node_id": self.node_id,
                                               "caller_node_id": self.node_id,
                                               "caller_node_token": self.node_token,
                                               "repo": self.code_repo,
                                               "commit_sha": self.code_commit})
                    break
                except RootServerError as e:
                    if e.api_rejected and e.data_code == "repo_timeout" and i < 2:
                        # The one transient refusal: the server's git read timed out
                        log.error("The root server timed out reading the repository, retrying...")
                        time.sleep(1)
                        continue
                    if e.api_rejected:
                        hint = hints.get(e.data_code)
                        log.error(f"The root server refused the world-source declaration of "
                                  f"'{self.code_repo}' @ {self.code_commit}"
                                  + (f": {hint}" if hint else f" [{e}]"))
                        raise
                    log.error(f"Error while declaring the world source to the root server [{e}]")
                    if i < 2:
                        log.misc("Retrying...")
                        time.sleep(1)  # Wait a little bit
                    else:
                        raise

            # The root folded the repository, while this world will hand out the grant
            # bundle: the two must be byte-identical, or the joiner's fail-closed gate
            # refuses every join. Refuse to start instead, here, where the author can fix it.
            # TODO commented since still incomplete right now
            #local_hash = canonical_world_hash(
            #    world_definition_members(unpack_py_files(self.world.packed_agent_files),
            #                             self.world.role_to_behav))
            #stored_hash = ret.get("hash") if isinstance(ret, dict) else None
            #if stored_hash != local_hash:
            #    msg = (f"The declared repository does not match the code this world is about to "
            #           f"hand out (root attested {stored_hash}, the local bundle folds to "
            #           f"{local_hash}): commit and push, then declare that commit")
            #    log.error(msg)
            #    raise GenException(msg)
            #log.misc(f"World source declared and attested by the root server (hash: {local_hash})")

    def send_dynamic_profile(self) -> None:
        """Sends the node's dynamic profile to the root server."""
        try:
            profile = dict(self.profile.get_dynamic_profile())  # Copy: the getter returns the internal dict
            if profile.get('hidden') is None:
                # Visibility never explicitly set this run: omit the key, so the server keeps
                # whatever the owner chose (the API only touches hidden when the key is present)
                profile.pop('hidden', None)
            self.__root(api="/account/node/profile/dynamic/post", payload={"node_id": self.node_id,
                                                                           "profile": profile})
        except Exception as e:
            log.error(f"Error while sending dynamic profile to root server [{e}]")

    async def send_badges(self) -> None:
        """Sends new badges assigned by a world node to the root server and notifies the agents. (async)"""
        if self.node_type is Node.WORLD:
            assert self.world is not None
            peer_id_to_badges = self.world.get_all_badges()
            if len(peer_id_to_badges) > 0:
                log.misc(f"Sending {len(peer_id_to_badges)} badges to root server")
                for i in range(0, 3):  # It will try 3 times before raising the exception...
                    try:
                        badges = [badge for _badges in peer_id_to_badges.values() for badge in _badges]
                        peer_ids = [peer_id for peer_id, _badges in peer_id_to_badges.items() for _ in _badges]

                        response = self.__root(api="/account/node/cv/badge/assign",
                                               payload={"badges": badges,
                                                        "world_node_id": self.node_id,
                                                        "world_badge_token": self.badge_token})

                        # Getting the next badge token
                        self.badge_token = response["badge_token"]
                        badges_states = response["badges_states"]  # List of booleans

                        # Check if posting went well and saving the set of peer IDs to contact
                        peer_ids_to_notify = set()
                        for z in range(0, len(badges_states)):
                            ret = badges_states[z]
                            if 'state' not in ret or 'code' not in ret['state'] or 'message' not in ret['state']:
                                log.error(f"Error while posting a badge assigned to {peer_ids[z]}. "
                                          f"Badge: {badges[z]}. "
                                          f"Error message: invalid response format")
                            else:
                                if ret['state']['code'] != "ok":
                                    log.error(f"Error while posting a badge assigned to {peer_ids[z]}. "
                                              f"Badge: {badges[z]}. "
                                              f"Error message: {ret['state']['message']}")
                                else:
                                    peer_ids_to_notify.add(peer_ids[z])

                        # Notify agents
                        for peer_id in peer_ids_to_notify:
                            if not (await self.conn.send(peer_id, channel_trail=None, content=None,
                                                         content_type=Msg.GET_CV_FROM_ROOT)):
                                log.error(f"Error while sending the request to re-download CV to peer {peer_id}")

                        # Clearing
                        self.world.clear_badges()
                        break
                    except Exception as e:
                        if isinstance(e, RootServerError) and e.api_rejected:
                            log.error(f"Badge assignment refused by the root server [{e}] (stop trying)")
                            break
                        log.error(f"Error while sending badges to server or when notifying peers [{e}]")
                        if i < 2:
                            log.misc("Retrying...")
                            time.sleep(1)  # Wait a little bit
                        else:
                            log.error("Couldn't complete badge sending or notification procedure (stop trying)")

    def get_public_addresses(self) -> list[str]:
        """Returns the public addresses of the P2P node

        Returns:
            The list of public addresses.
        """
        return self.conn.public_addresses

    def get_world_addresses(self) -> list[str]:
        """Returns the world addresses of the P2P node

        Returns:
            The list of world addresses.
        """
        return self.conn.world_addresses

    def get_public_peer_id(self) -> str:
        """Returns the public peer ID of the P2P node

        Returns:
            The public peer ID.
        """
        peer_id = self.conn.public_peer_id
        if peer_id is None:
            log.critical("Unable to retrieve the public peer ID (critical)")
            return ""
        return peer_id

    def get_world_peer_id(self) -> str:
        """Returns the world peer ID of the P2P node

        Returns:
            The world peer ID.
        """
        peer_id = self.conn.world_peer_id
        if peer_id is None:
            log.critical("Unable to retrieve the world/private peer ID (critical)")
            return ""
        return peer_id

    async def ask_to_get_in_touch(self, node_name: str | None = None, addresses: list[str] | None = None,
                                  node_id: str | None = None,
                                  public: bool = True, before_updating_pools_fcn: Callable | None = None,
                                  is_world_join: bool = False) -> str | None:
        """Tries to connect to another agent or world node. (async)

        Args:
            node_name: Name of the node to join (alternative to addresses below).
            addresses: A list of network addresses to connect to (alternative to node_name).
            node_id: The ID of the node to connect to (alternative to node_name and addresses).
            public: A boolean flag indicating whether to use the public or world P2P network.
            before_updating_pools_fcn: A callable invoked with the peer ID before updating the connection pools.
            is_world_join: If True, the public Session expects a WORLD_GRANT and drives
                the world handoff (used by ask_to_join_world).

        Returns:
            The peer ID of the connected node if successful, otherwise None.
        """

        # Checking arguments
        if node_name is None and addresses is None and node_id is None:
            return None
        if sum(x is not None for x in [node_name, addresses, node_id]) > 1:
            log.error(f"Cannot specify more than one of node_name ({node_name}), addresses, or node_id ({node_id}), "
                      f"check your code!")
            return None

        # Getting addresses, if needed
        if addresses is None:
            try:
                payload = {"account_token": self.unaiverse_key}
                if node_name is not None:
                    payload["node_name"] = node_name
                if node_id is not None:
                    payload["node_id"] = node_id
                addresses = self.__root(api="/account/node/get/addresses",
                                        payload=payload)["addresses"]
            except Exception as e:
                log.error(f"Error while retrieving addresses of node named {node_name} [{e}]")
                return None

        if addresses is None or len(addresses) == 0:
            log.error(f"Addresses of {node_name} were not found, cannot connect!")
            return None

        # Resolve the target peer-id from the multiaddrs (every root-server address
        # carries a trailing /p2p/<peer-id>; for a relay circuit it is the final hop,
        # i.e. the target — not the relay). The Session is now the *sole* dialer:
        # the Session runs the dial off-task (Session.start() -> _start_dial -> _run_dial,
        # relaunched on re-dial by tick_dial), so we no longer pre-dial here. The relay gate
        # is enforced when the dial is reaped, via the Session's reject_relay flag.
        peer_id = None
        for a in addresses:
            if "/p2p/" in a:
                cand = a.rsplit("/p2p/", 1)[-1].split("/")[0]
                if cand:
                    peer_id = cand
                    break
        if peer_id is None:
            log.error(f"Could not resolve a peer-id from addresses of {node_name}, cannot connect!")
            return None

        # A world join binds the pending carrier as soon as it is known: the dial below
        # can dispatch a synchronous refusal (quarantine gate), and the rejoin re-arm
        # correlates that FAILED with the in-flight join through this very field.
        if is_world_join and self.joining_world_info is not None:
            self.joining_world_info["world_public_peer_id"] = peer_id

        p2p_name = NodeConn.P2P_PUBLIC if public else NodeConn.P2P_WORLD

        # Legacy pool-routing hook (e.g. conn.set_world so the world private peer is
        # not filtered when pools refresh). Harmless for the agent path (None).
        if before_updating_pools_fcn is not None:
            before_updating_pools_fcn(peer_id)

        # Drive the per-peer Session handshake (the Session owns dial + handshake +
        # re-dial). On success the peer is enrolled via the Session ESTABLISHED
        # side effects (add_agent / join_world). Pool membership refreshes via the
        # main polling loop while we await ESTABLISHED.
        log.misc("Connecting to another agent/world (Session-owned dial + handshake)...")
        ok = await self.connect_via_session(peer_id=peer_id, addresses=addresses, p2p_name=p2p_name,
                                            is_world_join=is_world_join,
                                            reject_relay=not self.talk_to_relay_based_nodes)
        if not ok:
            log.error("Connection / Session handshake failed!")
            return None
        return peer_id

    async def ask_to_join_world(self, node_name: str | None = None, addresses: list[str] | None = None,
                                node_id: str | None = None, **kwargs) -> str | None:
        """Initiates a request to join a world. (async)

        Args:
            node_name: The name of the node hosting the world to join (alternative to addresses below).
            addresses: A list of network addresses of the world node (alternative to node_name).
            node_id: The ID of the node to join (alternative to node_name and addresses).
            **kwargs: Additional options forwarded to ask_to_get_in_touch.

        Returns:
            The public peer ID of the world node if the connection request is successful, otherwise None.
        """
        log.user("Asking to join world...")

        # Leave an already entered world (if any); idempotent, reads the host from conn.
        await self.leave_world()

        # Record the join intent BEFORE dialing, with the target already merged in:
        # a refusal dispatched synchronously inside the dial (the quarantine gate)
        # re-arms the rejoin from these options, which must therefore be complete —
        # merging only on success would stash target-less kwargs into the retry.
        kwargs |= {
            'node_name': node_name,
            'addresses': addresses,
            'node_id': node_id
        }
        self.joining_world_info = {"world_public_peer_id": None, "options": kwargs}

        # Start the public-network Session to the world. Fire-and-forget: the dial runs
        # off-task and the whole world handshake — dial → PROFILE → WORLD_GRANT → private
        # world Session → join_world body — completes asynchronously in the main loop. An
        # unreachable world surfaces later as a FAILED event (which arms the rejoin when
        # keep_rejoining is set), not as a synchronous failure here.
        peer_id = await self.ask_to_get_in_touch(node_name=node_name, addresses=addresses,
                                                 node_id=node_id, public=True, is_world_join=True)
        if peer_id is not None:
            self.joining_world_info["world_public_peer_id"] = peer_id
            log.user("Connected on the public network, waiting for the world handshake to complete...")
        else:
            log.error("Failed to join world!")
            self.joining_world_info = None
        return peer_id

    async def leave(self, peer_id: str):
        await self.__purge(peer_id, reason=GRACEFUL_SHUTDOWN)

    async def leave_world(self) -> None:
        """Leave the world we are currently in, if any. (async)

        The world host is read from the Coordinator (``conn.get_world_node_peer_id``),
        the single runtime source of truth for the current world identity (the profile
        field is a persisted mirror). Idempotent: a no-op when not in a world. This is the
        world-departure domain flow that used to live, inverted, inside the per-peer
        ``leave`` god-method:

          * GOAWAY + drop the world host (``__purge`` with WORLD_DISENGAGEMENT);
          * clear the world roster and point pool routing away from the host (``set_world(None)``);
          * drop every remaining in-world agent (``remove_all_world_agents``);
          * clear the world fields of the profile (the persisted mirror);
          * drop the cached world data and, for a code-hosting WAgent, restore a plain
            public Agent instance carrying over the pre-world attributes.

        Gated on ``in_world()`` (the role bit set by a completed join), NOT merely on conn
        holding a host: conn holds the host from handoff-initiate onward, so gating on the
        host alone would tear down on an in-flight or failed join. This mirrors the
        pre-refactor gate (profile world_peer_id, written only on a successful join).
        """
        if self.agent is None or not self.agent.in_world():
            return
        host = self.conn.get_world_node_peer_id()
        if host is None:
            return

        log.user("Leaving world...")

        # GOAWAY the world host and drop its connection (so the remote sees Session.Closed,
        # not Lost). remove_all_world_agents below GOAWAY-closes the other world agents the
        # same way, so none of them re-dials us back into a world we are leaving.
        await self.__purge(host, reason=WORLD_DISENGAGEMENT)

        # Clearing world-related lists in the connection manager (to avoid world agents
        # reconnecting) and pointing pool routing away from the host.
        self.conn.set_world(None)
        self.conn.set_world_agents_list(None)
        self.conn.set_world_masters_list(None)

        # Disconnecting all the remaining connected world agents (clears their roles too).
        await self.conn.remove_all_world_agents()

        # Clearing the world fields of the profile (the persisted mirror of the runtime state).
        dynamic_profile = self.profile.get_dynamic_profile()
        dynamic_profile['connections']['world_peer_id'] = None
        dynamic_profile['connections']['world_agents'] = None
        dynamic_profile['connections']['world_masters'] = None
        self.profile.mark_change_in_connections()

        # Clearing all joining options + cached world data.
        assert self.agent is not None
        await self.agent.clear_world_related_data(host)
        self.joining_world_info = None

        if self.agent.__class__.__name__ == "WAgent":
            new_agent = Agent(proc=None)

            # Clearing memory
            if self.memory_finder is not None:
                self.memory_finder.cleanup()
            self.memory_finder = None
        else:
            new_agent = self.agent

        # Cloning attributes of the existing agent
        skip = {'stats', 'world_profile', 'world_role_fsms', 'behav'}
        for key, value in self.agent.__dict__.items():
            if hasattr(new_agent, key):  # This will skip ROLE_BITS_TO_STR, CUSTOM_ROLES, etc...
                if key in skip:
                    continue
                else:
                    setattr(new_agent, key, value)

        # Fixing internal references to agent (im/proc were cloned from the
        # in-world instance and still point at it)
        new_agent.im.agent = new_agent
        new_agent.proc.agent = new_agent

        # Telling the FSM that actions are related to this new agent
        new_agent.behav.set_actionable(new_agent)
        new_agent.behav_lone_wolf.set_actionable(new_agent)

        # Inheriting the pre-defined policy filter (if any)
        new_agent.set_policy_filter(self.agent.policy_filter, public=False)
        new_agent.set_policy_filter(self.agent.policy_filter_lone_wolf, public=True)

        # Interaction Manager and processor
        new_agent.im.agent = new_agent
        if new_agent.proc is not None and hasattr(new_agent.proc, 'set_agent'):
            new_agent.proc.set_agent(new_agent)

        # Fixing role in the agent profile. Both world fields must be cleared HERE:
        # on the non-WAgent branch new_agent IS self.agent, so the skip-set above
        # cleans nothing, and accept_new_role reads world_role_fsms first — a stale
        # dict would survive the leave (and get pickled into the saved state).
        new_agent.world_profile = None
        new_agent.world_role_fsms = None
        self.world_role_fsms = None
        new_agent.accept_new_role(Agent.ROLE_PUBLIC)

        # Updating node-level references
        self.agent = new_agent
        self.hosted = new_agent

    @staticmethod
    def _profile_from_search_element(element: dict) -> NodeProfile:
        """Maps one element of /discover/search/query into a NodeProfile.

        The element is a flat spread of the API's public node serialisation (plus
        lat/lon/location): there is no static/dynamic/cv envelope, the owner fields
        live under 'account', the stored dynamic profile under 'profile' and the
        badge list under 'cv'.
        """
        account = element.get('account') if isinstance(element.get('account'), dict) else {}
        static = {
            'node_id': element.get('node_id'),
            'node_type': element.get('node_type'),
            'node_name': element.get('node_name'),
            'node_description': element.get('description'),
            'created_utc': element.get('created_utc'),
            'max_nr_connections': element.get('max_connections'),
            'certified': element.get('certified'),
            'allowed_node_ids': element.get('allowed_node_ids'),
            'world_masters_node_ids': element.get('world_masters_node_ids'),
            'location_method': element.get('location_method'),
            'nickname': account.get('nickname'),
            'name': account.get('name'),
            'surname': account.get('surname'),
            'title': account.get('title'),
            'organization': account.get('organization'),
        }
        dynamic = element.get('profile') if isinstance(element.get('profile'), dict) else {}
        cv = element.get('cv') if isinstance(element.get('cv'), list) else []
        return NodeProfile(static=static, dynamic=dynamic, cv=cv)

    def search(self, query_text: str, nickname: str | None = None, email: str | None = None) -> list[NodeProfile]:
        """Searches the UNaIVERSE platform for nodes matching the given query.

        Args:
            query_text: The text query used to search for nodes (matched against node
                names and descriptions).
            nickname: An optional account nickname to filter results by owner.
            email: Deprecated alias of nickname; addresses no longer identify accounts,
                so filtering by an address matches nothing on current servers.

        Returns:
            A list of NodeProfile objects matching the search query.
        """
        if email is not None and nickname is None:
            log.error("search(email=...) is deprecated: addresses no longer identify accounts, "
                      "pass the owner's nickname instead (forwarding the value as a nickname)")
            nickname = email

        elements = []
        profiles = []

        try:
            elements = self.__root(api="/discover/search/query", payload={
                "message": query_text,
                "nickname": nickname,
                "account_token": self.unaiverse_key
            })
        except Exception as e:
            log.critical(f"Error while searching! Query: {query_text}, nickname: {nickname} [{e}]")

        try:
            for p in elements:
                profiles.append(self._profile_from_search_element(p))
        except Exception as e:
            log.critical(f"Error while converting data returned by 'search'! "
                         f"Query: {query_text}, nickname: {nickname} [{e}]")
        return profiles

    def run(self, keep_rejoining: bool = True, *args, **kwargs) -> None:
        """Starts the main execution loop for the node, calling method run_async(...) by means of asyncio.run.
        See documentation of method run_async."""
        if keep_rejoining and 'join_world' not in kwargs:
            keep_rejoining = False
        self.keep_rejoining = keep_rejoining

        async def _run_then_close():
            try:
                await self.run_async(*args, **kwargs)
            finally:
                await self.aclose()  # This entry point owns the whole node lifetime

        try:
            asyncio.run(_run_then_close())
        except KeyboardInterrupt:
            return  # CTRL+C
        except GenException:
            return  # It should never happen (safety)

    async def run_async(self, cycles: int | None = None,
                        max_time: float | None = None,
                        interact_mode: bool = False,
                        resume_from_checkpoint: bool = False,
                        join_world: str | list[str] | None = None,
                        get_in_touch: str | list[str] | None = None,
                        **kwargs) -> None:
        """Starts the main execution loop for the node. (async)

        Args:
            cycles: The number of clock cycles to run the loop for. If None, runs indefinitely.
            max_time: The maximum time in seconds to run the loop. If None, runs indefinitely.
            interact_mode: A boolean value that turns interactive mode on (still experimental!).
            resume_from_checkpoint: If True, we load the checkpoint saved (if present).
            join_world: The name of the World to join or the list of its addresses.
            get_in_touch: The name of Agent to connect to or the list of its addresses.
            **kwargs: Additional keyword arguments forwarded to ask_to_join_world or ask_to_get_in_touch.
        """
        log.set_sub("gen")

        # First CTRL+C asks for graceful shutdown (the main loop checks
        # `self.stop_requested` between cycles). Second CTRL+C bypasses
        # all cleanup with os._exit(130). We install ONE handler (see below
        # for why not both): the loop-level one on Unix (immune to matplotlib/
        # PyTorch signal hijacks and to GIL-holding C extensions), falling back
        # to signal.signal only where the loop handler is unavailable (Windows /
        # no running loop). Registered here — after user imports but before the
        # main loop — to overwrite any handlers installed at import time by
        # matplotlib/PyTorch/etc.
        self.stop_requested = False
        stop_requested_at = 0.0

        # True only when the loop stops because the caller's cycle/time budget ran out,
        # which is a pause and not a shutdown (see the finally at the end of this method)
        budget_consumed = False

        def _request_stop() -> None:
            nonlocal stop_requested_at
            if self.stop_requested:
                # A single physical Ctrl+C can be DELIVERED here more than once: a
                # terminal signals the whole foreground process group, and a wrapper
                # like `uv run` additionally forwards the SIGINT to its child — so
                # the second delivery used to fire the force-quit escape hatch and
                # os._exit(130) truncated the graceful teardown (GOAWAYs, drain) at
                # a random point. Debounce: within the window it is the same
                # keypress and is ignored; past it, it is a real second Ctrl+C.
                if time.monotonic() - stop_requested_at >= 1.0:
                    os._exit(130)
                return
            stop_requested_at = time.monotonic()
            log.user("\nDetected Ctrl+C! Exiting gracefully... (Ctrl+C again to force-quit)")
            self.stop_requested = True

        # Install ONE handler, not both. asyncio's add_signal_handler already routes through
        # the loop's wakeup fd (immune to matplotlib/PyTorch signal hijacks and GIL-holding C
        # extensions), and arming signal.signal on top of it does NOT disarm that wakeup fd, so
        # a single SIGINT fires _request_stop twice: the second call sees stop_requested already
        # set and os._exit(130)s past the finally, so the GOAWAY teardown never runs. Use
        # signal.signal only as a fallback when the loop-level handler is unavailable (Windows
        # or no running loop). The second real Ctrl+C still force-quits via os._exit.
        loop_handler_installed = False
        try:
            asyncio.get_running_loop().add_signal_handler(signal.SIGINT, _request_stop)  # noqa
            loop_handler_installed = True
        except (NotImplementedError, RuntimeError):
            pass  # Windows or no event loop — fall back to signal.signal below

        if not loop_handler_installed:
            try:
                signal.signal(signal.SIGINT, lambda signum, frame: _request_stop())
            except (ValueError, OSError):
                pass  # Not in main thread, etc.

        # Subscribing/creating our own pubsub
        await self.hosted.subscribe_to_pubsub_owned_streams()

        # Load checkpoint (if exists)
        if resume_from_checkpoint:
            try:
                if not self.hosted.load():
                    log.user("No saved state found. Starting fresh.")
                else:
                    log.user("Successfully loaded previous agent state.")
            except Exception as e:
                log.critical(f"Found a save file but failed to load it: {e}")

        # Asking to join a World or connect to an Agent, if specified
        joined_this_world = None
        got_in_touch_with_this_lone_wolf = None
        waiting_for_lone_wolves = False
        if join_world is not None:
            if isinstance(join_world, str):
                ret = await self.ask_to_join_world(node_name=join_world, **kwargs)
            elif isinstance(join_world, list):
                ret = await self.ask_to_join_world(addresses=join_world, **kwargs)
            else:
                ret = None
                log.critical("Invalid value for the 'join_world' argument")
            if ret is None:
                log.critical(f"Unable to connect to world: {join_world}")
            else:
                joined_this_world = ret  # saving peer ID
        elif self.hosted.world_profile is not None:
            # We resumed from a state in which we were in this world, so we reconnect
            world_name = self.hosted.world_profile.get_static_profile()['node_name']
            owner = owner_handle(self.hosted.world_profile.get_static_profile())
            ret = await self.ask_to_join_world(node_name=f'{owner}/{world_name}', **kwargs)
            if ret is None:
                log.critical(f"Unable to connect to world: {owner}/{world_name}")
            else:
                joined_this_world = ret  # saving peer ID
        elif get_in_touch is not None:
            if isinstance(get_in_touch, str):
                ret = await self.ask_to_get_in_touch(node_name=get_in_touch, **kwargs)
            elif isinstance(get_in_touch, list):
                ret = await self.ask_to_get_in_touch(addresses=get_in_touch, **kwargs)
            else:
                ret = None
                log.critical("Invalid value for the 'get_in_touch' argument")
            if ret is None:
                log.critical(f"Unable to get in touch with agent: {get_in_touch}")
            else:
                got_in_touch_with_this_lone_wolf = ret  # saving peer ID
        else:
            waiting_for_lone_wolves = True

        # Saving for future reuse
        if self.keep_rejoining and self.joining_world_info is not None:
            self.rejoining_kwargs = copy.deepcopy(self.joining_world_info["options"])

        try:
            last_dynamic_profile_time = clock.get_time()
            last_get_token_time = clock.get_time()
            last_stats_send_time = clock.get_time()
            last_stats_save_time = clock.get_time()
            last_state_save_time = clock.get_time()
            if not (cycles is None or cycles > 0):
                log.critical("Invalid number of cycles")

            # Interactive mode (useful when chatting with lone wolves)
            keyboard_queue = None
            keyboard_listener = None
            processor_img_stream = None
            cap = None
            splash_text_shown = False
            interact_mode_opts: dict | None = None

            if interact_mode:
                from prompt_toolkit import prompt
                from prompt_toolkit.patch_stdout import patch_stdout

                if self.agent is None:
                    log.critical("Interactive mode is only valid for agents")
                interact_mode_opts = {
                    "ready_to_interact": False,
                }
                assert isinstance(interact_mode_opts, dict)
                if got_in_touch_with_this_lone_wolf is not None:
                    interact_mode_opts["lone_wolf_peer_id"] = got_in_touch_with_this_lone_wolf
                elif joined_this_world is not None:
                    interact_mode_opts["world_peer_id"] = joined_this_world
                elif waiting_for_lone_wolves:
                    interact_mode_opts["lone_wolf_peer_id"] = None

                # Share the (mutable) dict with the SessionEvent pump so
                # _on_session_established can flip ready_to_interact when the
                # handshake completes (same object, mutations are visible here).
                self._interact_mode_opts = interact_mode_opts

                public_streams = "lone_wolf_peer_id" in interact_mode_opts
                assert self.agent is not None
                proc_streams = self.agent.owned_streams[self.agent.get_proc_output_net_hash(public=public_streams)]
                for stream in proc_streams.values():
                    if processor_img_stream is None and stream.props.is_img():
                        processor_img_stream = stream

                def is_debug():
                    return not (sys.gettrace() is None and
                                not any(env in os.environ for env in ['DEBUGPY_RUNNING', 'PYCHARM_HOSTED']))

                def keyboard_listener(k_queue):
                    prev_keyboard_msg = None
                    with (patch_stdout(raw=True)):  # type: ignore
                        while True:
                            webcam_shot = None
                            keyboard_msg: str | None = None
                            if not is_debug():
                                keyboard_msg = prompt("\n👉 ")  # Get from keyboards
                            else:
                                if keyboard_msg is None or len(keyboard_msg) == 0:
                                    if os.path.exists("human_input.txt"):
                                        with open("human_input.txt", 'r') as file:
                                            keyboard_msg = file.read().strip()
                                            if keyboard_msg == prev_keyboard_msg:
                                                keyboard_msg = None
                                    if keyboard_msg is None or len(keyboard_msg) == 0:
                                        continue
                                    else:
                                        prev_keyboard_msg = keyboard_msg
                            if cap is not None:
                                _ret, got_shot = cap.read()  # Get from webcam
                                if _ret:
                                    target_area = 224 * 224
                                    webcam_shot = Image.fromarray(cv2.cvtColor(got_shot, cv2.COLOR_BGR2RGB))
                                    width, height = webcam_shot.size
                                    current_area = width * height

                                    if current_area > target_area:
                                        scale_factor = math.sqrt(target_area / current_area)
                                        new_width = int(round(width * scale_factor))
                                        new_height = int(round(height * scale_factor))
                                        webcam_shot = webcam_shot.resize((new_width, new_height),
                                                                         Image.Resampling.LANCZOS)

                            if keyboard_msg is not None and len(keyboard_msg) > 0:
                                k_queue.put((keyboard_msg, webcam_shot, "whatever"))  # Store in the asynch queue

                            if (keyboard_msg is not None and
                                    (keyboard_msg.strip() == "exit()" or keyboard_msg.strip() == "quit()")):
                                break

                keyboard_queue = queue.Queue()  # Create a thread-safe queue for communication
                keyboard_listener = threading.Thread(target=keyboard_listener, args=(keyboard_queue,), daemon=True)

            if clock.get_cycle() == -1:

                def format_list(items: list[str], base_indent: int = 21):
                    blocks = []
                    circuit_blocks = []
                    indent = " " * base_indent
                    for s in items:
                        if "127.0.0.1" in s:
                            continue
                        circuit = "/p2p-circuit" in s
                        s = s[1:]
                        s = s[s.find("/") + 1:]
                        s = s[0:s.find("/p2p")]  # Also covers /p2p-circuit
                        if "/certhash/" in s:
                            s = s[0:s.find("/certhash/"):]
                        if circuit:
                            circuit_blocks.append(indent + "@" + s)
                        else:
                            blocks.append(indent + s)
                    blocks.sort()
                    circuit_blocks.sort()
                    blocks += circuit_blocks
                    blocks[0] = blocks[0][base_indent:]
                    return ",\n".join(blocks)

                log.user("\nRunning " + ("agent" if self.agent else "world") + " '" +
                         self.hosted.get_name() + "' ..."
                         + f"\n- Owner:             {owner_handle(self.profile.get_static_profile())}"
                         + f"\n- Node ID:           {self.node_id}"
                         + f"\n- Public peer ID:    {self.get_public_peer_id()}"
                         + f"\n- Private peer ID:   {self.get_world_peer_id()}")
                log.user("- Public addresses:  " + format_list(self.get_public_addresses()))
                log.user("- Private addresses: " + format_list(self.get_world_addresses()))

            # Main loop
            must_quit = False
            self.run_start_time = clock.get_time()
            while not must_quit and not self.stop_requested:
                log.set_sub("gen")

                # Tuning clock speed. A node with a Session that needs a responsive
                # pump is never idle: the keep-alive PING RTT window and a handshake
                # in flight are both sized for a full-speed pump, and a slowed clock
                # (2s cycles) would starve them into false ping misses / handshake
                # timeouts. requires_fast_clock() covers every non-terminal, non-KNOWN
                # Session, so handshakes in flight keep the clock fast too, not just
                # ESTABLISHED/RECONNECTING.
                if (self.conn.passed_time_since_last_communication() >= Custom.SLOW_DOWN_CLOCK_AFTER and
                        self.hosted.im.count_interactions() == 0 and
                        not self.conn.requires_fast_clock()):
                    if not clock.is_slowed_down():
                        log.misc("Reducing clock speed (idle)")
                        clock.run_slower()
                else:
                    if clock.is_slowed_down():
                        log.misc("Restoring original clock speed")
                        clock.run_natural_speed()

                # Sending alive message every "K" seconds
                if ((clock.get_time() - self.last_alive_time >= Custom.SEND_ALIVE_EVERY) and
                        len(self.get_public_addresses()) > 0):
                    try:
                        # Capped and off the event loop: a hung POST to the root must not
                        # freeze the pump, and a transient failure must not kill the node.
                        was_alive = await asyncio.wait_for(asyncio.to_thread(self.send_alive),
                                                           timeout=Custom.DEFAULT_TIMEOUT)
                    except Exception as e:
                        log.error(f"Error while sending alive message to server (going ahead) [{e}]")
                        was_alive = False

                    # Checking only at the first run
                    if self.last_alive_time == 0 and was_alive and not Custom.SKIP_WAS_ALIVE_CHECK:
                        log.critical("The node is already alive, maybe running in a different machine? "
                                     "(set env variable NODE_IGNORE_ALIVE=1 to ignore this control)")
                    self.last_alive_time = clock.get_time()

                # Check inspector
                if self.inspector_activated:
                    if self.__inspector_told_to_pause:
                        log.misc("Paused by the inspector, waiting...")

                        while self.__inspector_told_to_pause:
                            if not self.inspector_activated:  # Disconnected
                                self.__inspector_told_to_pause = False
                                log.misc("Inspector is not active/connected anymore, resuming...")
                                break

                            public_messages, public_session_inbound = await self.conn.get_messages(
                                p2p_name=NodeConn.P2P_PUBLIC)
                            for msg in public_messages:
                                if msg.content_type == Msg.INSPECT_CMD:

                                    # Unpacking piggyback
                                    sender_node_id, sender_inspector_mode_on = (msg.piggyback[0:-1],
                                                                                msg.piggyback[-1] == "1")

                                    # Is message from inspector?
                                    sender_is_inspector = (sender_node_id == self.profile.get_static_profile()[
                                        'inspector_node_id'] and
                                                           sender_inspector_mode_on)

                                    if sender_is_inspector:
                                        await self.__handle_inspector_command(msg.content['cmd'], msg.content['arg'])
                                    else:
                                        log.error(
                                            "Inspector command was not sent by the expected inspector node ID "
                                            "or no inspector connected")
                                        await self.__purge(msg.sender)
                            # get_messages pops the session frames off the transport, so we must
                            # still pump them or peers get no PONG/keepalive and mark us LOST while
                            # paused. Dispatch normally (the pause freezes the agent's behave, not
                            # the network); wrap it so a raising dispatch branch does not kill the
                            # node under the inspector's eyes.
                            _, world_session_inbound = await self.conn.get_messages(
                                p2p_name=NodeConn.P2P_WORLD)
                            try:
                                events = await self.conn.pump_sessions(
                                    {NodeConn.P2P_PUBLIC: public_session_inbound,
                                     NodeConn.P2P_WORLD: world_session_inbound},
                                    now=time.monotonic())
                                for ev in events:
                                    await self._dispatch_session_event(ev)
                            except Exception as e:
                                log.error(f"[pause] session pump/dispatch failed (going ahead): {e}")
                            await asyncio.sleep(0.1)  # Don't block the loop

                # Move to the next cycle. The wait must be an ASYNC sleep: the
                # main loop is mostly synchronous (FFI calls), so this is where
                # asyncio.to_thread completions (conn.send/send_session/connect)
                # and loop-level callbacks (e.g. signal handlers) get their
                # chance to run — a blocking sleep here would starve them.
                while not clock.next_cycle():
                    await asyncio.sleep(0.001)  # Seconds (lowest possible granularity level)

                log.misc(f"=== Starting clock cycle {clock.get_cycle()} ===")

                # Re-fire a scheduled world rejoin when it comes due (see _tick_rejoin_watchdog).
                await self._tick_rejoin_watchdog()

                # Handle new connections or lost connections
                await self.__handle_network_connections()

                # Handle (read, execute) received network data/commands
                await self.__handle_network_messages(interact_mode_opts=interact_mode_opts)

                # Stream live data (generated and environmental)
                log.set_sub("gen")
                if len(self.hosted.all_agents) > 0:
                    if self.node_type is Node.WORLD:
                        if self.first:
                            self.first = False
                            for net_hash, stream_dict in self.hosted.known_streams.items():
                                for stream_obj in stream_dict.values():
                                    if isinstance(stream_obj, BufferedStream):
                                        stream_obj.restart()
                await self.hosted.send_stream_samples()
                if self.hosted.im.count_interactions() > 0:
                    log.inter(str(self.hosted.im))
                log.streams(self.hosted.streams_to_str())

                # Trigger HSM of the agent
                if self.node_type is Node.AGENT:
                    assert self.agent is not None
                    assert self.agent.proc_inputs is not None
                    if (interact_mode and isinstance(interact_mode_opts, dict) and
                            interact_mode_opts['ready_to_interact']):
                        public = "lone_wolf_peer_id" in interact_mode_opts
                        target_peer_id = interact_mode_opts['lone_wolf_peer_id'] \
                            if public else self.get_world_peer_id()
                        try:
                            if not splash_text_shown:
                                log.disable_all_screen()  # Will disable all channels, except the default ones
                                splash_text_shown = True
                                if public:
                                    self.agent.behav_lone_wolf.update_wildcard(Custom.PARTNER_WILDCARD,
                                                                               interact_mode_opts['lone_wolf_peer_id'])
                                    self.agent.behav_lone_wolf.apply_wildcards()
                                    log.user(f"\n*** Connected to agent "
                                             f"{interact_mode_opts['lone_wolf_peer_id']} ***")
                                else:
                                    log.user(f"\n*** Connected to world {interact_mode_opts['world_peer_id']} ***")
                                cap = cv2.VideoCapture(0) if processor_img_stream is not None else None
                                log.user("*** Entering interactive mode [type exit() to quit] ***\n")
                                assert keyboard_listener is not None
                                keyboard_listener.start()
                                time.sleep(1)

                            # Getting message from keyboard
                            assert keyboard_queue is not None
                            msg, image_pil, whatever = keyboard_queue.get_nowait()
                            msg = msg.strip()

                            if msg.lower() == "exit()" or msg.lower() == "quit()":

                                # Quit?
                                must_quit = True
                                if cap is not None:
                                    cap.release()

                                if self.agent.in_world():
                                    await self.leave_world()
                                connected_peer_ids = list(self.agent.all_agents.keys())
                                for peer_id in connected_peer_ids:
                                    await self.__purge(peer_id, reason=WORLD_DISENGAGEMENT)
                            elif msg.lower() == "/debug":
                                self.agent.behav_lone_wolf.set_debug_messages_active(
                                    not self.agent.behav_lone_wolf.are_debug_messages_active())
                                self.agent.behav.set_debug_messages_active(
                                    not self.agent.behav.are_debug_messages_active())
                            else:

                                # Writing message in the agent's default-stdin (that is where the human process will
                                # pick up data, no matter what is the actual stdin from the point of view of the
                                # interaction)
                                uuid = self.agent.prepare_stdin_if_human(public, peer_id=target_peer_id)

                                if len(self.agent.stdin) > 0:
                                    if len(self.agent.proc_inputs) == 1:
                                        self.agent.stdin.set([msg], uuid=uuid, force=True)
                                    else:
                                        self.agent.stdin.set([msg, image_pil, whatever], uuid=uuid, force=True)
                                else:
                                    log.error("Empty stdin!")
                        except queue.Empty:
                            pass  # If nothing has been typed (+ enter)

                    # Ordinary behavior
                    if not must_quit:
                        await self.agent.behave()
                        if self.hosted.im.count_interactions() > 0:
                            log.inter(str(self.hosted.im))
                        log.streams(self.hosted.streams_to_str())

                # Periodic Save
                if Custom.SAVE_CHECKPOINT_EVERY > 0.:
                    if clock.get_time() - last_state_save_time >= Custom.SAVE_CHECKPOINT_EVERY:
                        try:
                            log.user("Auto-saving state...")
                            self.hosted.save()
                            last_state_save_time = clock.get_time()
                        except Exception as e:
                            log.error(f"Auto-save failed: {e}")

                # Send dynamic profile every "N" seconds
                if (clock.get_time() - last_dynamic_profile_time >= Custom.SEND_DYNAMIC_PROFILE_EVERY
                        and self.profile.connections_changed()):
                    try:
                        last_dynamic_profile_time = clock.get_time()
                        self.profile.unmark_change_in_connections()
                        await self.send_badges()  # Sending and clearing badges
                        # Capped and off the event loop so a hung POST cannot freeze the pump.
                        await asyncio.wait_for(asyncio.to_thread(self.send_dynamic_profile),
                                               timeout=Custom.DEFAULT_TIMEOUT)  # Sending
                    except Exception as e:
                        log.error(f"Error while sending the updated dynamic profile (or badges) to the server "
                                  f"(trying to go ahead...) [{e}]")

                # Getting a new token every "N" seconds
                if clock.get_time() - last_get_token_time >= Custom.GET_NEW_TOKEN_EVERY:
                    try:
                        # Fetch off the event loop (capped), apply on the main task: a hung
                        # or failing refresh must neither freeze the pump nor kill the node.
                        response = await asyncio.wait_for(
                            asyncio.to_thread(self._request_node_token,
                                              [self.get_public_peer_id(), self.get_world_peer_id()]),
                            timeout=Custom.DEFAULT_TIMEOUT)
                        self._apply_node_token(response)
                        last_get_token_time = clock.get_time()  # success: next refresh in ~23.5h
                    except Exception as e:
                        log.error(f"Error while refreshing the node token (will retry shortly) [{e}]")
                        # Asymmetric backoff: retry in ~DEFAULT_TIMEOUT, not every iteration
                        # (busy-loop) nor in ~23.5h (the token would expire first).
                        last_get_token_time = (clock.get_time()
                                               - Custom.GET_NEW_TOKEN_EVERY + Custom.DEFAULT_TIMEOUT)

                # Continuously check the addresses of the node for changes
                try:
                    current_public_addrs = self.conn.public_addresses
                    current_private_addrs = self.conn.world_addresses
                    profile_public_addrs = self.profile.get_dynamic_profile().get('peer_addresses', [])
                    profile_private_addrs = self.profile.get_dynamic_profile().get('private_peer_addresses', [])

                    if set(current_public_addrs) != set(profile_public_addrs):
                        log.misc(f"Address change detected for the public instance! "
                                 f"New addresses: {current_public_addrs}")

                        # Update profile in-place
                        address_list = self.profile.get_dynamic_profile()['peer_addresses']
                        address_list.clear()
                        address_list.extend(current_public_addrs)

                        # mark as changed (-> sends the profile to the root)
                        self.profile.mark_change_in_connections()

                    # If private addresses changed, update the profile and notify the world
                    elif set(current_private_addrs) != set(profile_private_addrs):
                        log.misc(f"Address change detected for the private instance! "
                                 f"New addresses: {current_private_addrs}")

                        # Update profile in-place
                        address_list = self.profile.get_dynamic_profile()['private_peer_addresses']
                        address_list.clear()
                        address_list.extend(current_private_addrs)

                        # mark as changed (-> sends the profile to the root)
                        self.profile.mark_change_in_connections()

                        world_peer_id = self.conn.get_world_node_peer_id()
                        if world_peer_id is not None and self.node_type is Node.AGENT:
                            log.misc("Notifying world of address change...")
                            await self.conn.send(
                                world_peer_id, content_type=Msg.ADDRESS_UPDATE, channel_trail=None,
                                content={'addresses': self.profile.get_dynamic_profile()['private_peer_addresses']}
                            )
                    else:
                        log.misc("No address changes detected.")
                except Exception as e:
                    log.error(f"Failed to check for address updates: {e}")

                # Refresh relay reservation if nearing expiration
                if self.relay_reservation_expiry is not None:
                    time_to_expiry = self.relay_reservation_expiry - datetime.now(timezone.utc)
                    if time_to_expiry < timedelta(minutes=15):
                        log.misc("Relay reservation nearing expiration. Attempting to renew...")
                        world_private_peer_id = self.conn.get_world_node_peer_id()
                        assert world_private_peer_id is not None
                        new_expiry_utc = await self.conn.reserve(world_private_peer_id, NodeConn.P2P_WORLD)
                        if new_expiry_utc is not None:
                            self.relay_reservation_expiry = datetime.fromisoformat(
                                new_expiry_utc.replace('Z', '+00:00'))
                            assert self.relay_reservation_expiry is not None
                            log.misc(f"Relay reservation renewed. New expiration: "
                                     f"{self.relay_reservation_expiry.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                        else:
                            log.error("Failed to renew relay reservation. Node may become unreachable.")
                            self.relay_reservation_expiry = None

                # Send stats to the world
                if self.node_type is Node.AGENT:
                    assert self.agent is not None
                    if self.agent.in_world():
                        if clock.get_time() - last_stats_send_time >= Custom.SEND_STATS_EVERY:
                            try:
                                log.misc("Sending stats update to the world...")
                                last_stats_send_time = clock.get_time()
                                await self.agent.send_stats_to_world()
                            except Exception as e:
                                log.error(f"Error while sending stats to the world (trying to go ahead...) [{e}]")

                # Save stats to disk if this is the world node
                if self.node_type is Node.WORLD:
                    assert self.world is not None
                    if clock.get_time() - last_stats_save_time >= Custom.SAVE_STATS_EVERY:
                        try:
                            log.user("Saving stats to disk (world)...")
                            last_stats_save_time = clock.get_time()
                            assert self.world.stats is not None
                            self.world.stats.save_to_disk()
                        except Exception as e:
                            log.error(f"Error while saving stats to disk [{e}]")

                # Taking to the inspector
                if self.inspector_activated:
                    await self.__send_to_inspector()

                # Execute User Callback
                if self.run_hook is not None:
                    try:
                        self.run_hook(self)
                        # if asyncio.iscoroutinefunction(self.run_hook):
                        #     await self.run_hook(self)
                        # else:
                        #     self.run_hook(self)
                    except Exception as e:
                        log.error(f"Error in step_callback: {e}")

                # Stop conditions. Consuming a bounded budget is NOT the end of the node:
                # the caller asked for that many cycles and decides what happens next.
                if cycles is not None and ((clock.get_cycle() + 1) >= cycles):
                    budget_consumed = True
                    break
                if max_time is not None and (clock.get_time() - self.run_start_time) >= max_time:
                    budget_consumed = True
                    break

        except KeyboardInterrupt:
            log.user("\nDetected Ctrl+C! Exiting gracefully...")

        except asyncio.CancelledError:
            log.user("\nDetected process termination! Exiting gracefully...")
            raise

        except Exception as e:
            traceback.print_exc()
            log.critical(f"An error occurred: {e}")

        finally:

            # Any exit other than a consumed budget (the loop ending, Ctrl+C, an error)
            # is the end of this node, so shut it down here as always. When the budget
            # was merely consumed the node has to stay usable: a caller stepping it one
            # cycle at a time (NodeSynchronizer) would otherwise get a node that GOAWAYs
            # every peer, drains its sessions and closes its hosts between one cycle and
            # the next. Those callers own the lifetime and call aclose().
            if not budget_consumed:
                await self.aclose()

    async def aclose(self) -> None:
        """Shuts the node down: this is the point of no return, and it runs once.

        Tells the root the node is gone, saves what has to survive, GOAWAYs every peer
        and every Session, drains the farewell round-trips, closes the Go hosts (freeing
        their instance slots, which a process embedding several Nodes would otherwise
        run out of) and flushes the logger. Every step is guarded on its own: a failing
        goodbye must not cost the drain, and a failing drain must not cost the close.

        The order matters and is the reason this is not a plain sequence of calls: the
        transports must die AFTER the goodbyes, or the peers only ever observe a
        liveness timeout.
        """
        if self._closed:
            return
        self._closed = True

        try:
            self.send_alive(alive=False)
        except Exception as e:
            log.error(f"Error telling the node is not alive anymore: {e}")

        try:
            if Custom.SAVE_CHECKPOINT_EVERY > 0.:
                log.user("Saving hosted agent state to disk...")
                self.hosted.save()
        except Exception as e:
            log.error(f"Error saving hosted agent state: {e}")

        try:
            if self.node_type is Node.WORLD and self.world is not None:
                log.user("Shutting down stats database...")
                assert self.world.stats is not None
                self.world.stats.shutdown()
        except Exception as e:
            log.error(f"Error closing database: {e}")

        try:
            if self.node_type is Node.AGENT:
                assert self.agent is not None
                if self.agent.in_world():
                    await self.leave_world()
        except Exception:
            pass

        # Per-peer try/except so one peer that raises during purge does not skip the
        # GOAWAY for the others, nor the close_all_sessions sweep below.
        try:
            connected_peer_ids = list(self.hosted.all_agents.keys())
        except Exception:
            connected_peer_ids = []
        for peer_id in connected_peer_ids:
            try:
                await self.__purge(peer_id, reason=WORLD_DISENGAGEMENT)
            except Exception as e:
                log.error(f"Error purging a peer on shutdown: {e}")

        try:
            # GOAWAY any Session not covered by the per-peer leave above.
            await self.conn.close_all_sessions()
        except Exception as e:
            log.error(f"Error closing all sessions on shutdown: {e}")

        try:
            # Drain the farewell round-trips: pump the session traffic until every
            # GOAWAY got its ACK (or aborted) and the settlement cuts executed, so
            # the transports die AFTER the goodbyes, not under them. Without this
            # the process exits with the GOAWAYs still queued in the local muxer
            # and the peers only ever observe a liveness timeout.
            await self.conn.drain_teardown(budget_s=3.0)
        except Exception as e:
            log.error(f"Error draining the session teardown on shutdown: {e}")

        try:
            # Close the Go hosts (frees their slots for processes embedding
            # several Nodes); nothing else runs on the transports after the drain.
            self.conn.close_transports()
        except Exception as e:
            log.error(f"Error closing the transports on shutdown: {e}")

        try:
            log.close()  # Closing logger (asserts on a stdout-only logger)
        except Exception:
            pass

    async def __handle_network_connections(self) -> None:
        """Manages new and lost network connections. (async)"""

        # World host: expire the pre-admitted joiners that never redeemed their grant,
        # BEFORE the reconcile/publish below read the roster.
        if self.node_type is Node.WORLD:
            for peer_id in self.conn.sweep_world_preadmissions():
                log.misc(f"[session] pre-admission of ...{peer_id[-5:]} expired unredeemed "
                         f"→ unlisted", sub="prv")

        # Getting fresh lists of existing world agents and world masters (from the rendezvous)
        if self.node_type is Node.AGENT:
            log.misc("Updating list of world agents and world masters by using data from the rendezvous",
                     sub="prv")
            await self.conn.set_world_agents_and_world_masters_lists_from_rendezvous()

        # Updating connection pools, getting back the lists (well, dictionaries) of new agents and lost agents
        new_peer_ids_by_pool, removed_peer_ids_by_pool = await self.conn.update()
        if len(new_peer_ids_by_pool) > 0 or len(removed_peer_ids_by_pool) > 0:
            log.cpool("Current status of the pools, right after the update:\n" + str(self.conn))

        # Checking if some peers were removed
        an_agent_left_the_world = False
        removed_peers = False
        for pool_name, removed_peer_ids in removed_peer_ids_by_pool.items():
            for peer_id in removed_peer_ids:
                removed_peers = True
                log.misc("Removing a not-connected-anymore peer, "
                         "pool_name: " + pool_name + ", peer_id: " + peer_id + "...",
                         sub="prv" if pool_name in self.conn.WORLD or pool_name in self.conn.WORLD_NODE else "pub")
                # Inform the SessionManagers via the Coordinator — ESTABLISHED →
                # RECONNECTING (grace), in-handshake states → FAILED immediately.
                for ev in await self.conn.on_peer_disconnected(peer_id):
                    await self._dispatch_session_event(ev)

                # on_peer_disconnected above already synced the pool bookkeeping (the
                # link is gone). A Session riding its grace period (RECONNECTING) is not
                # gone yet: keep the enrolment (all_agents) alive across the blip so an
                # INTERACTION arriving mid-grace is not rejected as coming from an unknown
                # node. The hosted purge is then driven solely by the terminal LOST/CLOSED
                # Session event (see _dispatch_session_event). When no Session is in grace
                # for this peer, purge the hosted side now (the pool is already clean).
                # Settled: the link is already gone and on_peer_disconnected above already
                # signalled the SMs — this reckoning must not touch a Session again.
                if not self.conn.is_in_grace(peer_id):
                    await self.__purge(peer_id, settled=True)

                # Checking if we removed an agent from this world
                if self.node_type is Node.WORLD and pool_name in self.conn.WORLD:
                    an_agent_left_the_world = True

                # The world host dropping out of the pool is NOT where we leave/rejoin: a
                # transport blip must ride the world Session's reconnect grace (RESUME), and
                # only its terminal LOST/FAILED event (in _maybe_leave_world_on_session_end)
                # leaves the world and arms the rejoin. Deciding here, on the transport diff,
                # would tear the world down on a brief blip and skip the grace entirely.

                # Checking if the inspector disconnected
                if peer_id == self.inspector_peer_id:
                    self.inspector_activated = False
                    log.inspector_enabled(False)
                    self.inspector_peer_id = None
                    self.__inspector_cache = {"behav": None, "known_streams_count": 0, "all_agents_count": 0}
                    log.misc("Inspector disconnected")

        # Handling newly connected peers
        an_agent_joined_the_world = False
        added_peers = False
        # Every peer (re)appearing in the pool diff is a potential transport
        # reconnect for a Session riding its grace period: on_peer_reconnected
        # is a no-op unless the Session is RECONNECTING, so feeding it fresh
        # first-time connections is harmless.
        for new_peer_ids in new_peer_ids_by_pool.values():
            self.reconnected.update(new_peer_ids)
        for r in self.reconnected:
            # Tell the SessionManagers via the Coordinator — RECONNECTING →
            # ESTABLISHED if the grace period hasn't expired yet (else no-op).
            for ev in await self.conn.on_peer_reconnected(r):
                await self._dispatch_session_event(ev)
            pool_name = self.conn.get_pool_of(r)
            if pool_name is None:
                continue
            if pool_name not in new_peer_ids_by_pool:
                new_peer_ids_by_pool[pool_name] = {r}
            else:
                new_peer_ids_by_pool[pool_name].add(r)
        self.reconnected.clear()
        for pool_name, new_peer_ids in new_peer_ids_by_pool.items():
            for peer_id in new_peer_ids:
                added_peers = True
                log.misc("Processing a newly connected peer, "
                         "pool_name: " + pool_name + ", peer_id: " + peer_id + "...",
                         sub="prv" if pool_name in self.conn.WORLD or pool_name in self.conn.WORLD_NODE else "pub")

                # Peer enrolment (interview/profile-request/world.add_agent) is no longer
                # driven off pool diffs — it is driven by the per-peer Session handshake:
                # an inbound HELLO spawns a responder Session (SessionManager.deliver), and
                # the world-host registration / agent add happens in the Session ESTABLISHED
                # dispatch (_world_host_register_agent / add_agent). Nothing to do here.
                continue

        # Updating list of world agents & friends, if needed
        # (it happens only if the node hosts a world, otherwise 'an_agent_joined_the_world' and
        # 'an_agent_left_the_world' are certainly False)
        world_agents_peer_infos = None
        world_masters_peer_infos = None
        if self.node_type is Node.WORLD:
            assert self.world is not None
            enter_left = an_agent_joined_the_world or an_agent_left_the_world
            timeout = (clock.get_time() - self.last_rendezvous_time) >= Custom.PUBLISH_RENDEZVOUS_EVERY

            if enter_left or timeout or self.world.role_changed_by_world or self.world.received_address_update:
                if enter_left or self.world.role_changed_by_world:
                    # Updating world-node profile with the summary of currently connected agents in the world
                    world_agents_peer_infos = self.conn.get_all_connected_peer_infos(NodeConn.WORLD_AGENTS)
                    world_masters_peer_infos = self.conn.get_all_connected_peer_infos(NodeConn.WORLD_MASTERS)

                    dynamic_profile = self.profile.get_dynamic_profile()
                    dynamic_profile['world_summary']['world_agents'] = world_agents_peer_infos
                    dynamic_profile['world_summary']['world_masters'] = world_masters_peer_infos
                    dynamic_profile['world_summary']["world_agents_count"] = len(world_agents_peer_infos)
                    dynamic_profile['world_summary']["world_masters_count"] = len(world_masters_peer_infos)
                    dynamic_profile['world_summary']["total_agents"] = (len(world_agents_peer_infos) +
                                                                        len(world_masters_peer_infos))
                    self.profile.mark_change_in_connections()

                # Publish updated list of (all) world agents (i.e., both agents and masters)
                world_all_peer_infos = self.conn.get_all_connected_peer_infos(NodeConn.WORLD)
                if not (await self.conn.publish(self.get_world_peer_id(),
                                                f"{self.conn.world_peer_id}::ps:rv",
                                                content_type=Msg.WORLD_AGENTS_LIST,
                                                content={"peers": world_all_peer_infos,
                                                         "update_count": clock.get_cycle()})):
                    log.error("Failed to publish the updated list of (all) world agents (ignoring)", sub="prv")
                else:
                    self.last_rendezvous_time = clock.get_time()
                    log.misc(f"Rendezvous messages just published "
                             f"(tag: {clock.get_cycle()}, peers: {len(world_all_peer_infos)})", sub="prv")
                    log.misc(f"Rendezvous message included peer IDs: "
                             f"{[p['id'] for p in world_all_peer_infos]})", sub="prv")

                    # Clearing
                    self.world.role_changed_by_world = False
                    self.world.received_address_update = False

        # Updating list of node connections (being this a world or a plain agent)
        if added_peers or removed_peers:

            # The following could have been already computed in the code above, let's reuse
            if world_agents_peer_infos is None:
                world_agents_peer_infos = self.conn.get_all_connected_peer_infos(NodeConn.WORLD_AGENTS)
            if world_masters_peer_infos is None:
                world_masters_peer_infos = self.conn.get_all_connected_peer_infos(NodeConn.WORLD_MASTERS)
            world_private_peer_id = self.conn.get_all_connected_peer_infos(NodeConn.WORLD_NODE)
            world_private_peer_id = world_private_peer_id[0]['id'] if len(world_private_peer_id) > 0 else None

            # This is only computed here
            public_agents_peer_infos = self.conn.get_all_connected_peer_infos(NodeConn.PUBLIC)

            # Updating node profile with the summary of currently connected peers
            dynamic_profile = self.profile.get_dynamic_profile()
            dynamic_profile['connections']['public_agents'] = public_agents_peer_infos
            dynamic_profile['connections']['world_agents'] = world_agents_peer_infos
            dynamic_profile['connections']['world_masters'] = world_masters_peer_infos
            dynamic_profile['connections']['world_peer_id'] = world_private_peer_id
            self.profile.mark_change_in_connections()

    async def __handle_network_messages(self, interact_mode_opts: dict | None = None) -> None:
        """Handles and processes all incoming network messages. (async)

        Args:
            interact_mode_opts: A dictionary of options for interactive mode, or None if not in interactive mode.
        """
        # Fetching all messages — get_messages now returns (chat, session) tuples.
        public_messages, public_session_inbound = await self.conn.get_messages(p2p_name=NodeConn.P2P_PUBLIC)
        world_messages, world_session_inbound = await self.conn.get_messages(p2p_name=NodeConn.P2P_WORLD)
        interact_mode = interact_mode_opts is not None

        log.misc("Got " + str(len(public_messages)) + " messages from the public net", sub="pub")
        log.misc("Got " + str(len(world_messages)) + " messages from the world/private net", sub="prv")

        # Pump the Sessions (deliver inbound + tick), Coordinator-owned
        # SessionInbound items skip Msg.from_bytes / JWT verify; identity is
        # established cryptographically by sig_a / sig_b inside the handshake. The
        # Coordinator delivers each inbound to its instance's manager and then ticks
        # every manager, returning a flat 1:1 SessionEvent list in generation order
        # (all delivers public-then-world, then all ticks); Node stays the single
        # dispatch site.
        events = await self.conn.pump_sessions(
            {NodeConn.P2P_PUBLIC: public_session_inbound,
             NodeConn.P2P_WORLD: world_session_inbound},
            now=time.monotonic())
        for ev in events:
            await self._dispatch_session_event(ev)

        # Sorting messages
        public_messages = self.__sort_messages_by_priority(public_messages)
        world_messages = self.__sort_messages_by_priority(world_messages)

        # Process all messages
        all_messages = public_messages + world_messages
        log.set_sub("pub")
        is_private_message = False

        for i, msg in enumerate(all_messages):
            if i < len(public_messages):
                log.misc("Processing public message " + str(i + 1) + "/"
                         + str(len(public_messages)) + ": " + str(msg))
            else:
                if not is_private_message:
                    log.set_sub("prv")
                    is_private_message = True
                log.misc("Processing world/private message " + str(i - len(public_messages) + 1)
                         + "/" + str(len(world_messages)) + ": " + str(msg))

            # Checking
            if not isinstance(msg, Msg):
                log.error("Expected message of type Msg, got {} (skipping)".format(type(msg)))
                continue

            # Unpacking piggyback
            sender_node_id, sender_inspector_mode_on = (msg.piggyback[0:-1],
                                                        msg.piggyback[-1] == "1")

            # Is message from inspector?
            sender_is_inspector = (sender_node_id == self.profile.get_static_profile()['inspector_node_id'] and
                                   sender_inspector_mode_on)

            # (A) legacy PROFILE handshake — REMOVED. The per-peer Session now carries all
            # of it (PROFILE_OFFER/DECISION, WORLD_GRANT) and runs the role assignment +
            # add_agent + join_world side effects via _dispatch_session_event. PROFILE is
            # still a valid content-type (send_profile_to_all may emit it), but the
            # Session-era consumer ignores any that arrives. The old WORLD_APPROVAL /
            # AGENT_APPROVAL / PROFILE_REQUEST types are gone: an undecodable frame from a
            # not-yet-migrated peer is already dropped at the codec (Msg.from_bytes).
            if msg.content_type == Msg.PROFILE:
                log.misc(f"Ignoring legacy handshake message {msg.content_type} from {msg.sender} "
                         f"(Session handshake supersedes it)")

            # (E) the world node received an ADDRESS_UPDATE from an agent
            elif msg.content_type == Msg.ADDRESS_UPDATE:
                log.misc("Received an address update from " + msg.sender)

                if self.node_type is Node.WORLD:
                    assert self.world is not None
                    if msg.sender in self.world.all_agents:
                        all_addresses = msg.content.get('addresses')
                        if all_addresses and isinstance(all_addresses, list):
                            # Update the address both in the connection and in the profile
                            self.conn.set_addresses_in_peer_info(msg.sender, all_addresses)
                            self.world.set_addresses_in_profile(msg.sender, all_addresses)
                            log.misc(f"Waiting rendezvous publish after address update from {msg.sender}")

            # (F) got stream data
            elif msg.content_type == Msg.STREAM_SAMPLE:
                log.misc("Received a stream sample...")

                if self.node_type is Node.AGENT:  # Handling the received samples
                    assert self.agent is not None
                    added_data = self.agent.get_stream_sample(net_hash=msg.channel, sample_dict=msg.content)

                    # Printing messages to screen, if needed (useful when chatting with lone wolves)
                    if interact_mode:
                        net_hash = DataProps.normalize_net_hash(msg.channel)
                        if net_hash in self.agent.known_streams:
                            peer_id = DataProps.peer_id_from_net_hash(net_hash)
                            group = DataProps.name_or_group_from_net_hash(net_hash)
                            owner_account = owner_handle(self.agent.all_agents[peer_id].get_static_profile())
                            agent_name = self.agent.all_agents[peer_id].get_static_profile()['node_name']
                            for (user_hash, uuid) in added_data:
                                if user_hash not in self.agent.known_streams_by_user_hash:
                                    continue
                                stream_obj = self.agent.known_streams_by_user_hash[user_hash]
                                data = stream_obj.get(requested_by="print", uuid=uuid)
                                name = stream_obj.props.get_name()
                                if data is None:
                                    continue
                                if stream_obj.props.is_text():
                                    msg = data  # Getting message
                                    msg = "\n   ｜".join([line[i:i + 120] for line in msg.splitlines()
                                                         for i in range(0, len(line), 120)])
                                    log.user(f"\n💬 [{owner_account}/{agent_name}.{group}.{name}]\n   ｜{msg}")
                                elif stream_obj.props.is_img():
                                    img = data  # Getting image
                                    filename = f"{net_hash.replace(':', '_')}.{name}.png"
                                    img.save(filename)
                                    log.user(f"\n🖼️ [{owner_account}/{agent_name}.{group}.{name}]\n   "
                                             f"｜Saved image to {filename})")
                                else:
                                    msg = stream_obj.props.to_text(data)
                                    msg = "\n   ｜".join([line[i:i + 120] for line in msg.splitlines()
                                                         for i in range(0, len(line), 120)])
                                    log.user(f"\n🗂️ [{owner_account}/{agent_name}.{group}.{name}]\n   "
                                             f"｜Got a sample of type {stream_obj.props.data_type}, "
                                             f"tag {stream_obj.get_tag()}\n   ｜{msg}")

                elif self.node_type is Node.WORLD:
                    log.error("Unexpected stream samples received by this world node, sent by: " + msg.sender)
                    await self.__purge(msg.sender, reason=LOCAL_ERROR)

            # (G) got interaction request (new-style, replaces ACTION_REQUEST for new agents)
            elif msg.content_type == Msg.INTERACTION:
                log.misc(f"Received an interaction request...")

                if self.node_type is Node.AGENT:
                    assert self.agent is not None
                    if msg.sender not in self.agent.all_agents:
                        log.error("Unexpected interaction from unknown node: " + msg.sender)
                    else:
                        room_for_registration = True
                        if hasattr(self.agent, 'im'):
                            room_for_registration = self.agent.im.room_for_registration()

                        if room_for_registration:
                            self.agent.inject_received_interaction(msg.sender, msg.content)

                elif self.node_type is Node.WORLD:
                    log.error("Unexpected interaction received by this world node, sent by: " + msg.sender)
                    await self.__purge(msg.sender, reason=LOCAL_ERROR)

            # (G3) got interaction status update
            elif msg.content_type == Msg.INTERACTION_STATUS:
                log.misc("Received an interaction status update...")

                if self.node_type is Node.AGENT:
                    if hasattr(self.agent, 'im'):
                        await self.agent.im.update_sent_status(msg.content)

            # (H) got role suggestion
            elif msg.content_type == Msg.ROLE_SUGGESTION:
                log.misc("Received a role suggestion/new role...")

                if self.node_type is Node.AGENT:
                    if msg.sender == self.conn.get_world_node_peer_id():
                        new_role_indication = msg.content
                        if new_role_indication['peer_id'] == self.get_world_peer_id():
                            self.__replace_agent_instance_by_role(role=new_role_indication['role'],
                                                                  changing_an_existing_role_in_world=True)

                elif self.node_type is Node.WORLD:
                    assert self.world is not None
                    if msg.sender in self.world.world_masters:
                        for role_suggestion in msg.content:
                            await self.world.set_role(peer_id=role_suggestion['peer_id'], role=role_suggestion['role'])

            # (I) got request to alter the HSM
            elif msg.content_type == Msg.HSM:
                log.misc("Received a request to alter the HSM...")

                if self.node_type is Node.AGENT:
                    assert self.agent is not None
                    if msg.sender in self.agent.world_masters:  # This must be coherent with what we do in set_role
                        ret = getattr(self.agent.behav, msg.content['method'])(*msg.content['args'])
                        if not ret:
                            log.error(f"Cannot run HSM action named {msg.content['method']} with args "
                                      f"{msg.content['args']}")
                    else:
                        log.error(
                            "Only world-master can alter HSMs of other agents: " + msg.sender)  # No need to purge

                elif self.node_type is Node.WORLD:
                    log.error(
                        "Unexpected request to alter the HSM received by this world node, sent by: " + msg.sender)
                    await self.__purge(msg.sender, reason=LOCAL_ERROR)

            # (J) misc — the ping-pong *handshake* / reconnection use is REMOVED
            # (liveness is the session-level PING/PONG keep-alive in the Session;
            # reconnection is the Session RECONNECTING/RESUMED/LOST grace machinery).
            # The Msg.MISC content-type is kept for any other miscellaneous use.
            elif msg.content_type == Msg.MISC:
                log.misc(f"Received a misc message from {msg.sender} (no handshake action)")

            # (K) got a request to re-download the CV from the root server
            elif msg.content_type == Msg.GET_CV_FROM_ROOT:
                log.misc("Received a notification to re-download the CV...")

                # Downloading CV
                self.profile.update_cv(self.get_cv())

                # Re-downloading token (it will include the new CV hash)
                self.get_node_token(peer_ids=[self.get_public_peer_id(), self.get_world_peer_id()])

            # (L) got one or more badge suggestions
            elif msg.content_type == Msg.BADGE_SUGGESTIONS:
                log.misc("Received badge suggestions...")

                if self.node_type is Node.WORLD:
                    assert self.world is not None
                    for badge_dict in msg.content:
                        # Right now, we accept all the suggestions
                        self.world.add_badge(**badge_dict)  # Adding to the list of badges
                elif self.node_type is Node.AGENT:
                    log.error("Receiving badge suggestions is not expected for an agent node")

            # (M) got a special connection/presence message for an inspector
            elif msg.content_type == Msg.INSPECT_ON:
                log.misc("Received an inspector-activation message...")

                if sender_is_inspector:
                    self.inspector_activated = True
                    log.inspector_enabled(True)
                    self.inspector_peer_id = msg.sender
                    log.misc("Inspector activated")
                else:
                    log.error("Inspector-activation message was not sent by the expected inspector node ID")
                    await self.__purge(msg.sender, reason=LOCAL_ERROR)

            # (N) got a command from an inspector
            elif msg.content_type == Msg.INSPECT_CMD:
                log.misc("Received a command from the inspector...")

                if sender_is_inspector and self.inspector_activated:
                    await self.__handle_inspector_command(msg.content['cmd'], msg.content['arg'])
                else:
                    log.error("Inspector command was not sent by the expected inspector node ID "
                              "or the inspector was not yet activated (Msg.INSPECT_ON not received yet)")
                    await self.__purge(msg.sender, reason=LOCAL_ERROR)

            # (O) world got stats update from an agent
            elif msg.content_type == Msg.STATS_UPDATE:
                log.misc("Received a stats update from " + msg.sender)
                if self.node_type is Node.WORLD:
                    assert self.world is not None
                    if msg.sender in self.world.all_agents:
                        # This calls the world.add_stats_from_peer method
                        self.world.add_peer_stats(msg.content)
                    else:
                        log.error(f"Received stats update from {msg.sender}, "
                                  f"but they are not a known agent in this world.")
                elif self.node_type is Node.AGENT:
                    log.error("Receiving stats updates is not expected for an agent node.")

            # (P) got a request for stats from an agent
            elif msg.content_type == Msg.STATS_REQUEST:
                log.misc("Received a stats request from " + msg.sender)
                if self.node_type is Node.WORLD:
                    assert self.world is not None
                    assert self.world.stats is not None

                    # Extract filters from content
                    filters = msg.content or {}
                    asker = msg.sender

                    # If the filter is about one or more specific stats, then it is a fine-grain request,
                    # otherwise it is a generic request, so we send back the rendered HTML
                    stat_names = filters.get('stat_names', [])

                    if stat_names is not None and len(stat_names) > 0:

                        # This is a fine-grain request, so we query the DB. CONTRACT: the answer must be
                        # a get_view()-shaped dict ({"world": ..., "peers": ...}) - the asker merges it
                        # into its local stats view (update_stats_view)
                        response_payload = self.world.answer_stats_request(filters, asker)
                    else:

                        # This is a generic request, so we respond generating an HTML
                        time_range = filters.get('time_range', 0)
                        response_payload = self.world.stats.plot(since_timestamp=time_range)

                    # Send back as STATS_RESPONSE
                    await self.conn.send(msg.sender, channel_trail=None,
                                         content_type=Msg.STATS_RESPONSE,
                                         content=response_payload)
                elif self.node_type is Node.AGENT:
                    log.error("Receiving stats request is not expected for an agent node.")

            # (Q) agent got stats response from a world
            elif msg.content_type == Msg.STATS_RESPONSE:
                log.misc("Received a stats response from " + msg.sender)
                if self.node_type is Node.AGENT:
                    assert self.agent is not None
                    if msg.sender == self.conn.get_world_node_peer_id():
                        if isinstance(msg.content, dict) and ('world' in msg.content or 'peers' in msg.content):
                            self.agent.update_stats_view(msg.content, self.agent.overwrite_stats)
                        else:
                            log.misc("Ignoring a stats response that is not view-shaped (e.g., HTML)")
                    else:
                        log.error(f"Received stats response from {msg.sender}, but it is not the world.")
                elif self.node_type is Node.AGENT:
                    log.error("Receiving stats response is not expected for a world node.")

        # (interview-queue cleanup + connected-without-ack retry removed — the
        # Session FSM owns handshake timeouts, retry, and reconnection.)

    async def __join_world(self, profile: NodeProfile, role: int,
                           packed_agent_files: str, roles_fsm: dict[str, str] | None,
                           rendezvous_tag: int, initial_stats: dict[str, Any] | None,
                           peer_id: str | None = None,
                           attested_code_hash: str | None = None,
                           legacy_stats_code: str | None = None) -> bool:
        """Performs the actual operation of joining a world after receiving confirmation. (async)

        Args:
            profile: The profile of the world to join.
            role: The role assigned to the agent in the world (int).
            packed_agent_files: The string encoding agent-code-files, defining the agent's actions
                (including stats.py, when the world ships custom stats).
            roles_fsm: Dict role -> FSM (JSON string) from the grant bundle; None with
                legacy worlds, whose profile still carries the FSMs.
            attested_code_hash: The world-code hash attested by the root in the world's
                verified token; None when the token carries no such claim (check skipped).
            legacy_stats_code: The stats.py source shipped by legacy worlds in the
                dedicated code_bundle field; folded into the bundle when it lacks stats.py.
            rendezvous_tag: The rendezvous tag from the world's profile.
            initial_stats: When joining a world we eventually receive the recent history.
            peer_id: the peer-id of the already-established private world Session
                (set by the world-handoff path, the only caller); the post-connection body
                runs against it.

        Returns:
            True if the join operation is successful, otherwise False.
        """
        log.user(f"\nActually joining world, role ID will be '{role}'")

        # The private world Session already connected during the handoff.
        # The post-connection body runs against the passed-in peer_id.
        assert self.agent is not None

        if peer_id is None:
            return False

        # Relay reservation logic: reserve a slot on the world node only when WE are not
        # publicly reachable AND the world node actually offers a reachable relay (is_relay
        # from its profile), not merely because our relay client is enabled.
        world_is_relay = bool(profile.get_dynamic_profile().get('is_relay'))
        if not self.conn.world_is_public and self.conn.world_relay_enabled and world_is_relay:
            log.user("Node is not publicly reachable. Attempting to reserve a slot on the world's private network.")
            expiry_utc = await self.conn.reserve(peer_id, NodeConn.P2P_WORLD)

            if expiry_utc is not None:
                self.relay_reservation_expiry = \
                    (datetime.fromisoformat(expiry_utc.replace('Z', '+00:00')))
                assert self.relay_reservation_expiry is not None
                log.user(f"Reserved relay slot. Expires at "
                         f"{self.relay_reservation_expiry.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            else:
                log.error("An error occurred during relay reservation.")
        
        # Subscribing to the world rendezvous topic, from which we will get fresh information
        # about the world agents and masters
        log.misc("Subscribing to the world-members topic...")
        if not (await self.conn.subscribe(peer_id, channel=f"{peer_id}::ps:rv")):  # Special rendezvous (ps:rv)
            await self.__purge(peer_id, reason=WORLD_DISENGAGEMENT)  # If subscribing fails, we quit (safer)
            return False

        # The public connection to the world node is NOT torn down here: the dispatcher
        # retires the carrier right after this body returns (complete_world_handoff sends
        # GOAWAY(WORLD_HANDOFF_COMPLETE) and evicts its pools), and the settlement cut
        # severs the transport once the CLOSED lands. Purging here used to disconnect the
        # public transport BEFORE that GOAWAY was even queued, so the host never saw the
        # handoff close and failed its parked carrier on every successful join.

        # Subscribing to all the other world topics, from which we will get fresh information
        # about the streams
        log.misc("Subscribing to the world-streams topics...")
        dynamic_profile = profile.get_dynamic_profile()
        list_of_props = []
        list_of_props += dynamic_profile['streams'] if dynamic_profile['streams'] is not None else []
        list_of_props += dynamic_profile['proc_outputs'] if dynamic_profile['proc_outputs'] is not None else []

        if not (await self.agent.add_compatible_streams(peer_id, list_of_props, buffered=False, public=False)):
            await self.__purge(peer_id, reason=WORLD_DISENGAGEMENT)
            return False

        # Role FSMs, delivered by the grant bundle (the caller already folded in the
        # legacy profile-carried variant when the bundle had none). Cached on self only
        # past the integrity gate, together with the agent files.
        roles_fsm = roles_fsm or {}

        # Getting agent files
        try:
            agent_files = unpack_py_files(packed_agent_files)
        except Exception:

            # Assuming we are in the case of a single agent.py for all roles (still valid)
            agent_files = {f"{k}.py": packed_agent_files for k in roles_fsm}

        # Legacy stats channel: fold it into the bundle when absent there. Harmless
        # for attested worlds (they never ship the field; a tampering attempt would
        # just change the fold below and fail the gate).
        if legacy_stats_code and 'stats.py' not in agent_files:
            agent_files['stats.py'] = legacy_stats_code

        # Code-integrity gate: when the world's root token attests a code hash, the
        # received bundle must fold to exactly that hash ("what you run is what the
        # root attested", same canonical members the root read from the world's
        # declared repository). Tokens without the claim skip this, leaving
        # analyze_code below as the only check.
        # TODO commented since still incomplete right now
        #if attested_code_hash:
        #    local_hash = canonical_world_hash(world_definition_members(agent_files, roles_fsm or {}))
        #    if local_hash != attested_code_hash:
        #        log.error(f"World-code hash mismatch: the token attests {attested_code_hash}, the received "
        #                  f"bundle folds to {local_hash}; blocking the join operation")
        #        return False
        #    log.misc(f"World-code hash verified against the root-attested claim ({local_hash})")
        #else:
            # No claim in the token: the world declared no source repository, a fully
            # supported outcome (it runs unpublished), so there is nothing to compare
            # the grant bundle against and analyze_code below remains the only gate.
        #    log.misc("No world-code hash attested in the world's token, skipping the code-integrity check")

        # Checking files (single gate: stats.py, when shipped, is part of the bundle)
        if not analyze_code(agent_files):
            log.error(
                f"Invalid agent-related code (syntax errors or unsafe code) was provided by "
                f"the world, blocking the join operation")
            return False

        # Load the custom stats class when the bundle ships one. stats.py is executed
        # in a bare module (no package), exactly as before: relative imports cannot
        # resolve there, so it must stay self-contained on the joiner side.
        stats_class = None
        agent_stats_code = agent_files.get('stats.py')
        if agent_stats_code:
            try:
                stats_mod = types.ModuleType("dynamic_stats_module")
                exec(agent_stats_code, stats_mod.__dict__)
                if not hasattr(stats_mod, 'WStats'):
                    log.error("World sent stats.py, but it lacks a 'WStats' class. Using default Stats.")
                else:
                    stats_class = stats_mod.WStats
                    log.misc("Loaded custom WStats class from world.")
            except Exception as e:
                log.error(f"Failed to exec custom stats.py from world: {e}. Using default Stats.")

        # Saving reference files and role FSMs (in memory - needed if switching role)
        self.world_agent_files = agent_files
        self.world_role_fsms = roles_fsm

        # Replacing agent instance with a new one, accordingly to the role
        self.__replace_agent_instance_by_role(role, peer_id, profile, stats_class, initial_stats)

        # Telling the connection manager the info needed to discriminate peers (getting them from the world profile)
        # notice that the world node private ID was already told to the connection manager (see a few lines above)
        log.misc(f"Rendezvous tag received with profile: {rendezvous_tag} "
                 f"(in conn pool: {self.conn.rendezvous_tag})")
        if self.conn.rendezvous_tag < rendezvous_tag:
            # Seed the gate one cycle BEHIND the grant tag: the first post-join publish
            # can carry update_count == grant tag (the host's cycle may not have advanced
            # between the grant and its next publish, e.g. on a slowed clock) and the
            # strictly-greater gate would discard it, costing a whole publish cadence.
            # A same-cycle publish is at worst marginally older than the grant snapshot
            # applied right below, and the next publish repairs it.
            self.conn.rendezvous_tag = rendezvous_tag - 1
            num_world_masters = len(dynamic_profile['world_summary']['world_masters']) \
                if dynamic_profile['world_summary']['world_masters'] is not None else 'none'
            num_world_agents = len(dynamic_profile['world_summary']['world_agents']) \
                if dynamic_profile['world_summary']['world_agents'] is not None else 'none'
            log.misc(f"Rendezvous from profile (tag: {rendezvous_tag}), world masters: {num_world_masters}")
            log.misc(f"Rendezvous from profile (tag: {rendezvous_tag}), world agents: {num_world_agents}")
            self.conn.set_world_masters_list(dynamic_profile['world_summary']['world_masters'])
            self.conn.set_world_agents_list(dynamic_profile['world_summary']['world_agents'])

            masters = dynamic_profile['world_summary']['world_masters']
            if masters is None:
                masters = []
            agents = dynamic_profile['world_summary']['world_agents']
            if agents is None:
                agents = []
            world_all_peer_ids = ([p['id'] for p in masters] + [p['id'] for p in agents])
            log.misc(f"Rendezvous from profile included peer IDs: "
                     f"{world_all_peer_ids})", sub="prv")

        # Updating our profile to set the world we are in
        self.profile.get_dynamic_profile()['connections']['world_peer_id'] = peer_id
        self.profile.mark_change_in_connections()

        # The join truly completed (the role bit is set): disarm the rejoin watchdog so a
        # pending tick does not retry (and tear down) the world we just joined, and refresh
        # the rejoin target to THIS world (a runtime world switch must not rejoin the old one).
        self.rejoining_time = -1.
        if self.keep_rejoining and self.joining_world_info is not None:
            self.rejoining_kwargs = copy.deepcopy(self.joining_world_info["options"])

        log.user("Handshake completed, world joined!")
        return True

    async def __join_agent(self, profile: NodeProfile, peer_id: str) -> bool:
        """Adds a new known agent after receiving an approval message. (async)

        Args:
            profile: The profile of the agent to join.
            peer_id: The peer ID of the agent.

        Returns:
            True if the agent is successfully added, otherwise False.
        """
        log.misc("Adding a new known agent " + peer_id)
        assert self.agent is not None
        if not (await self.agent.add_agent(peer_id=peer_id, profile=profile)):
            await self.__purge(peer_id, reason=LOCAL_ERROR)
            return False

        if self.conn.is_public(peer_id):
            self.agent.behav_lone_wolf.update_wildcard(Custom.PARTNER_WILDCARD, peer_id)
            self.agent.behav_lone_wolf.apply_wildcards()

        return True

    # NOTE: __interview_enqueue / __interview_check_profile / __interview_clean /
    # __handle_connected_without_ack have been REMOVED. Profile exchange is Stage-2
    # of the Session handshake (PROFILE_OFFER/DECISION); acceptance/role decisions
    # are in _session_decision_hook; timeouts/retry/reconnection are owned by the
    # Session FSM. The profile-acceptance evaluation that lived in
    # __interview_check_profile (cv_hash / certified / hosted.evaluate_profile) rides
    # with the deferred auth model and is intentionally light in the hook for now.

    def set_decision_policy(self, policy) -> None:
        """Override the responder decision policy at runtime. The new policy takes effect on
        the next inbound PROFILE_OFFER for every Session, existing or future (the Sessions
        hold a lambda that re-reads self._decision_policy on each call, not a frozen hook).

        Args:
            policy: an async callable with the same signature as _session_decision_hook,
                ``(peer_id, remote_node_id, remote_profile, p2p_name, verified_claims) ->
                ProfileDecisionResult``. Pass self._session_decision_hook to restore the
                built-in policy.
        """
        self._decision_policy = policy

    async def _session_decision_hook(self, peer_id: str, remote_node_id, remote_profile: dict,
                                     p2p_name: str,
                                     verified_claims: VerifiedTokenClaims | None = None):
        """Responder decision hook called by a Session on an inbound PROFILE_OFFER.

        Returns a ProfileDecisionResult: accept/reject + (world host, public
        instance) a WorldGrantSpec. This is the Session-era replacement for the
        legacy interview/approval block — the role-assignment and grant payload
        assembly are ported here.

        ``verified_claims`` carries the root-attested token claims (RT) when a
        token-verifier vouched for the peer: ``remote_node_id`` is then the attested
        node-id, the offered profile's CV is checked against the attested cv_hash,
        and the joiner's world-side peer-id is bound to the attested p2p_peers. None
        on the v1 / no-verifier path, where the checks below fall back to the
        self-asserted profile.
        """

        # Allow-list gate (node-id based). A peer whose node-id could not be
        # determined must NOT slip past a configured allow-list.
        if (self.allowed_node_ids is not None and
                (remote_node_id is None or remote_node_id not in self.allowed_node_ids)):
            log.misc(f"[session] decision: node_id {remote_node_id} not in allow-list → reject")
            return ProfileDecisionResult(accepted=False, reason_code=NOT_IN_ALLOW_LIST)

        try:
            profile = NodeProfile.from_dict(remote_profile)
        except Exception as e:
            log.error(f"Bad remote profile from peer_id {peer_id}: {e}")
            return ProfileDecisionResult(accepted=False, reason_code=HOSTED_REJECTED)

        # RT profile-integrity gate: the root attests the node's cv_hash, so an
        # offered profile whose CV does not hash to the attested value is forged →
        # reject (this restores the world access filter the profile alone cannot be
        # trusted for). Skipped on the v1 / no-verifier path.
        if verified_claims is not None:
            match, _ = profile.verify_cv_hash(verified_claims.cv_hash)
            if not match:
                log.misc(f"[session] decision: cv_hash mismatch for ...{peer_id[-5:]} → reject")
                return ProfileDecisionResult(accepted=False, reason_code=CV_HASH_MISMATCH)

        # World host
        if self.node_type is Node.WORLD:
            # On the private world instance, an inbound offer is a joiner redeeming
            # a grant: accept (no new grant). The Session ESTABLISHes on PROFILE and
            # the agent is registered on that event; PRIVATE_JOIN was dropped.
            if p2p_name == NodeConn.P2P_WORLD:
                return ProfileDecisionResult(accepted=True)

            # On the public instance: assign a role and mint a grant.
            is_world_master = (self.world_masters_node_ids is not None and
                               remote_node_id in self.world_masters_node_ids)
            assert self.world is not None
            role_str = self.world.assign_role(profile=profile, is_world_master=is_world_master)
            if role_str is None:
                return ProfileDecisionResult(accepted=False, reason_code=ROLE_INCOMPATIBLE)
            role = self.world.ROLE_STR_TO_BITS[role_str]
            role = role | (Agent.ROLE_WORLD_MASTER if is_world_master else Agent.ROLE_WORLD_AGENT)

            # Admit the joiner's world-side identity BEFORE it dials the private
            # instance: the world perimeter cut (the world SessionManager's reconcile)
            # disconnects unknown peers, and the post-handshake registration
            # (_world_host_register_agent) would come too late to save the
            # in-flight world handshake. The legacy flow did this at approval time.
            joiner_dyn = profile.get_dynamic_profile()
            joiner_private_peer_id: str | None = joiner_dyn.get('private_peer_id')
            joiner_private_addr: list[str] | None = joiner_dyn.get('private_peer_addresses')
            # RT binding: the joiner's world-side peer-id must be the one the root
            # attested in p2p_peers, so the host preregisters a root-vouched identity
            # instead of a self-claimed one (the public peer-id was already checked
            # inside the token verification). Skipped on the v1 / no-verifier path.
            # A joiner with no root-vouched private identity/addresses cannot be a routable
            # world member: reject structurally instead of asserting (which would tear down
            # the world host, and under python -O would preregister a None-addressed member).
            if not joiner_private_peer_id or not joiner_private_addr:
                log.misc(f"[session] decision: joiner ...{peer_id[-5:]} has no private "
                         f"peer-id/addresses → reject")
                return ProfileDecisionResult(accepted=False, reason_code=NOT_AUTHORIZED)
            assert self.world.stats is not None
            if (verified_claims is not None and
                    joiner_private_peer_id not in verified_claims.p2p_peers):
                log.misc(f"[session] decision: world binding mismatch for ...{peer_id[-5:]} → reject")
                return ProfileDecisionResult(accepted=False, reason_code=NOT_AUTHORIZED)
            # Provisional: the entry must exist before the joiner dials the world
            # instance (the perimeter would cut an unknown peer mid-handshake), but
            # only the world-side registration confirms it — unredeemed grants are
            # swept once the TTL expires.
            if is_world_master:
                self.conn.add_to_world_masters_list(joiner_private_peer_id, joiner_private_addr, role,
                                                    provisional_ttl_s=Custom.WORLD_PREADMISSION_TTL)
            else:
                self.conn.add_to_world_agents_list(joiner_private_peer_id, joiner_private_addr, role,
                                                   provisional_ttl_s=Custom.WORLD_PREADMISSION_TTL)

            is_human = profile.get_static_profile().get("node_type") == self.hosted.HUMAN
            # World-definition bundle, the single channel for everything the joiner
            # executes: the pruned agent files (stats.py rides inside them, so there
            # is no stats_code field) and the role FSMs (no longer in the profile).
            code_bundle = json.dumps({
                "agent_files": self.world.packed_agent_files,
                "roles_fsm": self.world.role_to_behav,
            }).encode("utf-8")
            initial_stats = (json.dumps(self.world.stats.plot()).encode("utf-8")
                             if is_human else b"")
            spec = WorldGrantSpec(
                role=role,
                world_profile_bytes=json.dumps(self.profile.get_all_profile()).encode("utf-8"),
                rendezvous_tag=clock.get_cycle(),
                code_bundle=code_bundle,
                initial_stats=initial_stats,
            )
            # No role stash needed here: the pre-admission (add_to_world_*_list above)
            # records the role under the joiner's PRIVATE peer-id, and the world-side
            # ESTABLISHED reads it back from that record (conn.get_role) — the grant is
            # unsigned and PRIVATE_JOIN no longer exists.
            log.misc(f"[session] decision: granting world role '{role_str}' to ...{peer_id[-5:]}")
            return ProfileDecisionResult(accepted=True, world_grant=spec)

        # Agent host
        # v1: accept (allow-list already gated above). Custom hosted.evaluate_profile
        # / certified / cv_hash gating returns with the deferred auth model.
        return ProfileDecisionResult(accepted=True)

    async def _dispatch_session_event(self, ev) -> None:
        """Single dispatch site translating SessionEvents into application action.

        ESTABLISHED drives the world handoff (joiner side) and the ``add_agent`` /
        ``__join_world`` / world-list side effects that the legacy handshake used to
        perform inline. RECONNECTING is logged; RESUMED re-enrols the agent if the
        disconnect path already purged it. The terminal branches all end in a purge
        (pool evict + hosted removal; the transport was already settled by the SM
        before these events surfaced, so the purge's drop lands on the tearing-down
        no-op branch): FAILED first rolls back the handoff correlation, unlists the
        world member and re-arms the rejoin of a never-completed join; LOST/CLOSED
        first unlist and possibly run the full leave_world (which early-returns past
        the purge); a CLOSED carrying WORLD_HANDOFF_COMPLETE is the intentional
        retirement of the public carrier after a successful join and must not touch
        anything (complete_world_handoff already evicted its pools)."""
        try:
            log.misc(f"[session] {ev.type.value} peer=...{ev.peer_id[-5:]} session={ev.session_id[:8]}… "
                     f"role={ev.role.value} p2p={ev.p2p_name}",
                     sub="prv" if "world" in ev.p2p_name else "pub")
        except Exception:
            pass

        if ev.type == SessionEventType.ESTABLISHED:
            await self._on_session_established(ev)
        elif ev.type == SessionEventType.RESUMED:
            await self._on_session_resumed(ev)
        elif ev.type == SessionEventType.FAILED:
            # Roll back any in-flight world-handoff correlation (Coordinator-owned;
            # aborts the in-flight world redemption, no-ops if it already established).
            await self.conn.rollback_handoff(ev.peer_id)
            self._world_host_unlist_member(ev)
            await self._maybe_leave_world_on_session_end(ev)
            self._maybe_rearm_rejoin_on_failed_join(ev)
            log.error(f"[session] handshake to ...{ev.peer_id[-5:]} failed "
                      f"(p2p={ev.p2p_name}, reason={ev.data.get('reason', 'unknown')})",
                      sub="prv" if "world" in ev.p2p_name else "pub")
            # Evict the pools right away: a FAILED peer must not linger "connected"
            # (pool-resident, chat-admitted, re-dial suppressed) for the settlement
            # linger. The transport cut is the SM settlement's job; hosted removal is
            # a no-op (enrolment only happens at ESTABLISHED); the purge's drop lands
            # on the tearing-down / no-session branch. Gated on a real session_id: the
            # quarantine gate synthesizes FAILED(session_id="") without any Session
            # ending, and purging there could tear down a fresh inbound relation the
            # peer re-established while quarantined (inbound is deliberately not gated).
            if ev.session_id:
                try:
                    await self.__purge(ev.peer_id, settled=True)
                except Exception as e:
                    log.error(f"[session] dispatch FAILED purge failed for ...{ev.peer_id[-5:]}: {e}")
        elif ev.type == SessionEventType.CLOSED:
            # WORLD_HANDOFF_COMPLETE is the intentional teardown of the public
            # Session after a successful world join — it must NOT purge the node
            # (complete_world_handoff already retired the carrier and its pools).
            if "WORLD_HANDOFF_COMPLETE" in ev.data.get("reason", ""):
                return
            # Deliberately NO rollback_handoff here: a carrier CLOSED mid-handoff (e.g.
            # the host GOAWAYing everything while shutting down) must not abort the
            # in-flight world redemption — the world Session decides the join's fate:
            # ESTABLISHED completes the join (complete_world_handoff cleans the maps),
            # a failure rolls back and re-arms via its own FAILED/LOST dispatch. A
            # clean CLOSED never re-arms by design, so rolling back here would strand
            # the pending join forever (silent abort, no terminal event, no retry).
            self._world_host_unlist_member(ev)
            if await self._maybe_leave_world_on_session_end(ev):
                return
            try:
                await self.__purge(ev.peer_id, settled=True)
            except Exception as e:
                log.error(f"[session] dispatch CLOSED purge failed for ...{ev.peer_id[-5:]}: {e}")
        elif ev.type == SessionEventType.LOST:
            await self.conn.rollback_handoff(ev.peer_id)   # a mid-handoff carrier lost = dead handoff
            self._world_host_unlist_member(ev)
            if await self._maybe_leave_world_on_session_end(ev):
                return
            # LOST is involuntary: a pending join whose carrier (or world Session)
            # rode its grace to LOST must retry, same as a FAILED one. A clean
            # CLOSED still never re-arms.
            self._maybe_rearm_rejoin_on_failed_join(ev)
            try:
                await self.__purge(ev.peer_id, settled=True)
            except Exception as e:
                log.error(f"[session] dispatch LOST purge failed for ...{ev.peer_id[-5:]}: {e}")

    async def _tick_rejoin_watchdog(self) -> None:
        """One tick of the world-rejoin watchdog, run once per main-loop cycle.

        A scheduled rejoin (rejoining_time armed by _maybe_leave_world_on_session_end on an
        involuntary world-link end) fires ask_to_join_world, which dials and then hands off
        ASYNChronously — so a dial that returns is NOT yet a completed join. We therefore
        RESCHEDULE (not disarm) each time, at the earliest of the default wait and any sooner
        retry armed by the attempt's own failure dispatch (a quarantine refusal schedules
        just past its ban via _maybe_rearm_rejoin_on_failed_join): the timer is disarmed
        across the attempt precisely so that sooner target survives the default re-anchor.
        The true 'disarm' happens ONLY on completion (__join_world clears the timer when the
        role bit is set). Deliberately NOT gated on in_world(): a leave_world that raised
        before clearing the role bit would otherwise keep in_world() True and let this tick
        cancel the rejoin, wedging the node out of the world with no retry — so we let it
        re-fire and recover instead. A duplicate tick while a join is in flight is harmless:
        initiate() no-ops on the matching-intent carrier. Any error from the dial is caught
        so a transient failure retries rather than unwinding the main loop."""
        if not (0. < self.rejoining_time <= clock.get_time()):
            return
        if not self.keep_rejoining:
            self.rejoining_time = -1.
            return
        self.rejoining_time = 0.
        failed = False
        try:
            failed = await self.ask_to_join_world(**self.rejoining_kwargs) is None
        except Exception as e:
            log.error(f"[rejoin] ask_to_join_world raised, will retry: {e}")
        if self.rejoining_time <= 0.:
            self.rejoining_time = clock.get_time() + Custom.REJOINING_WAIT
        if failed:
            log.user(f"Failed to rejoin, retrying in "
                     f"{self.rejoining_time - clock.get_time():.0f} seconds...")

    def _maybe_rearm_rejoin_on_failed_join(self, ev) -> None:
        """Joiner side: an INVOLUNTARY terminal (FAILED, LOST, or a failed join body)
        on a world we are still trying to join (joining_world_info set, role bit not
        yet on) re-arms the rejoin. Without this a world unreachable at the first join
        leaves the node up but never joined and never retrying (the dial is off-task,
        so ask_to_join_world no longer fails synchronously). An involuntary end of an
        ALREADY-joined world is instead handled by _maybe_leave_world_on_session_end;
        a clean CLOSED never retries. The watchdog then re-runs the whole
        ask_to_join_world flow until the world is up."""
        if self.node_type is not Node.AGENT or not self.keep_rejoining:
            return
        if self.joining_world_info is None:
            return
        if self.agent is not None and self.agent.in_world():
            return  # in-world already → this FAILED is some other peer, not our join
        # The failure belongs to our in-flight join iff it is on the pending public carrier
        # (case A, world offline: its dial failed) or on a world-instance Session (case B,
        # public reached but the private world dial failed after the grant). Before we are
        # in-world there are no other world peers, so a world-instance FAILED here is the
        # join's own private Session.
        pending_carrier = self.joining_world_info.get("world_public_peer_id")
        is_pending_carrier = pending_carrier is not None and ev.peer_id == pending_carrier
        is_world_instance = NodeConn.P2P_WORLD == ev.p2p_name or "world" in ev.p2p_name
        if not (is_pending_carrier or is_world_instance):
            return
        if not self.rejoining_kwargs:
            self.rejoining_kwargs = copy.deepcopy(self.joining_world_info["options"])
        # A quarantine refusal names its own horizon (retry_after_s): schedule just past
        # the ban instead of a blind full wait, and PULL IN an already-armed retry when
        # this target is sooner — never push it out.
        retry_after = ev.data.get("retry_after_s")
        wait_s = (Custom.REJOINING_WAIT if retry_after is None
                  else min(float(retry_after) + _REJOIN_RETRY_MARGIN_S, Custom.REJOINING_WAIT))
        target = clock.get_time() + wait_s
        if self.rejoining_time <= 0.:
            self.rejoining_time = target
            log.misc(f"[session] pending world join failed ({ev.type.value}) → retry in {wait_s:.0f}s",
                     sub="prv")
        elif target < self.rejoining_time:
            self.rejoining_time = target
            log.misc(f"[session] pending world join refused → retry pulled in to {wait_s:.0f}s",
                     sub="prv")

    async def _maybe_leave_world_on_session_end(self, ev) -> bool:
        """Joiner side: a terminal event on the Session toward the CURRENT world
        host means we are out of the world — run the full leave_world cleanup
        (world lists, profile connections, role, agent-instance restore, code
        bundle unload) instead of a plain purge, otherwise the agent keeps
        believing it is in-world (in_world() reads the role bit) and e.g. the
        stats sender fires against a dead peer forever. The world host is read from
        the Coordinator (the single runtime source of truth). A terminal event caused
        by our own leave_world does not re-enter: leave_world clears the host
        synchronously (conn.set_world(None)) before its CLOSED surfaces through the pump."""
        if self.node_type is not Node.AGENT:
            return False
        if not (NodeConn.P2P_WORLD == ev.p2p_name or "world" in ev.p2p_name):
            return False
        host = self.conn.get_world_node_peer_id()
        if host is None or ev.peer_id != host:
            return False
        # conn holds the host from handoff-initiate onward, i.e. BEFORE the join
        # completes. A terminal event on the host before we are actually in the world
        # (in_world() reads the role bit set by __join_world) is a failed join, not a
        # departure: fall through to the plain purge, leaving the application state
        # untouched (matching the pre-refactor guard, whose profile mirror stayed unset
        # until the join succeeded).
        if self.agent is None or not self.agent.in_world():
            return False
        log.misc(f"[session] world link terminated ({ev.type.value}, "
                 f"reason={ev.data.get('reason', 'unknown')}) → leaving world", sub="prv")
        # Arm the rejoin BEFORE leaving (leave_world is best-effort and may raise; the rejoin
        # must survive that). Only an INVOLUNTARY end re-arms: a clean CLOSED is a GOAWAY from
        # the host (kick / role-revoke / graceful shutdown / our own token-expiry close), where
        # re-joining would just be bounced again. LOST (host vanished) and FAILED (e.g. an
        # inbound-dispatch error on the live world Session) do re-arm.
        if self.keep_rejoining and ev.type != SessionEventType.CLOSED:
            self.rejoining_time = clock.get_time() + Custom.REJOINING_WAIT
        try:
            await self.leave_world()
        except Exception as e:
            log.error(f"[session] leave_world after world-session end failed: {e}")
        return True

    def _world_host_unlist_member(self, ev) -> None:
        """World host only: a member's world Session terminated — stop advertising
        it (world agents/masters lists feed routing and the rendezvous publish).
        Covers grant-time preregistration too (a joiner that failed mid-handshake)."""
        if self.node_type is Node.WORLD and (NodeConn.P2P_WORLD == ev.p2p_name or
                                             "world" in ev.p2p_name):
            try:
                self.conn.remove_from_world_lists(ev.peer_id)
            except Exception as e:
                log.error(f"[session] world unlist failed for ...{ev.peer_id[-5:]}: {e}")

    async def _on_session_resumed(self, ev) -> None:
        """A Session survived a transport drop (RESUME acked). The removed-diff
        path deliberately KEEPS the hosted enrolment while the Session rides its
        reconnect grace (the purge there is gated on not-in-grace, see
        __handle_network_connections), so this is usually a no-op; it repairs the
        hosted side only when the peer WAS purged before the resume (e.g. the
        grace was entered late, after a non-grace purge already ran), re-enrolling
        from the profile the Session still holds. Pool membership self-heals on
        the next conn.update()."""
        if ev.peer_id in self.hosted.all_agents:
            return
        is_world = NodeConn.P2P_WORLD == ev.p2p_name or "world" in ev.p2p_name
        # A world host's public Sessions are grant carriers: they never enrolled
        # an agent, so there is nothing to restore.
        if self.node_type is Node.WORLD and not is_world:
            return
        profile_dict = ev.data.get("profile")
        if not profile_dict:
            return
        try:
            profile = NodeProfile.from_dict(profile_dict)
        except Exception as e:
            log.error(f"[session] resumed: bad stored profile for ...{ev.peer_id[-5:]}: {e}")
            return
        try:
            if not (await self.hosted.add_agent(peer_id=ev.peer_id, profile=profile)):
                await self.__purge(ev.peer_id, reason=LOCAL_ERROR)
        except Exception as e:
            log.error(f"[session] resumed: re-enrol failed for ...{ev.peer_id[-5:]}: {e}")

    async def _on_session_established(self, ev) -> None:
        """Run the application side effects for an ESTABLISHED Session, branching on
        (role, p2p instance, grant presence)."""
        is_world = NodeConn.P2P_WORLD == ev.p2p_name or "world" in ev.p2p_name
        profile_b = ev.data.get("profile_b")

        # Initiator on the PUBLIC instance
        if ev.role == SessionRole.INITIATOR and not is_world:
            grant = ev.data.get("world_grant")
            if grant is not None:
                # World join: open the private world Session to redeem the grant.
                await self._start_world_handoff(ev.peer_id, grant)
            elif profile_b is not None:
                # Plain agent connection.
                try:
                    await self.__join_agent(profile=NodeProfile.from_dict(profile_b), peer_id=ev.peer_id)
                except Exception as e:
                    log.error(f"[session] __join_agent failed for ...{ev.peer_id[-5:]}: {e}")
                # Interactive lone-wolf mode: the targeted peer we dialed is now
                # connected (mirrors the pre-Session AGENT_APPROVAL handler).
                opts = self._interact_mode_opts
                if isinstance(opts, dict) and 'lone_wolf_peer_id' in opts:
                    opts['ready_to_interact'] = True
            return

        # Initiator on the WORLD instance
        if ev.role == SessionRole.INITIATOR and is_world:
            grant = ev.data.get("grant")
            if grant is None and not self.conn.is_pending_handoff(ev.peer_id):
                # In-world agent-to-agent connection (connect_to/connect_by_role):
                # same enrolment as the public initiator path. Only the handoff
                # Session to the world host carries a grant.
                if profile_b is not None:
                    try:
                        await self.__join_agent(profile=NodeProfile.from_dict(profile_b), peer_id=ev.peer_id)
                    except Exception as e:
                        log.error(f"[session] __join_agent failed for ...{ev.peer_id[-5:]}: {e}")
                return
            # We joined a world: run the join body and retire the public Session.
            ok = False
            try:
                ok = await self._join_world_after_session(ev.peer_id, grant)
            except Exception as e:
                log.error(f"[session] __join_world failed for ...{ev.peer_id[-5:]}: {e}")
            if ok:
                # Interactive world mode: we are in the world now (mirrors the
                # pre-Session WORLD_APPROVAL handler). Gated on a successful join
                # so we never enter the REPL against a world we failed to join.
                opts = self._interact_mode_opts
                if isinstance(opts, dict) and 'world_peer_id' in opts:
                    opts['ready_to_interact'] = True
            # Whether or not the join body fully succeeded, the public Session has
            # done its job — retire it: GOAWAY(WORLD_HANDOFF_COMPLETE) on the still-live
            # public transport + pool eviction, with the settlement cut severing the
            # transport once the CLOSED lands. The Coordinator owns the correlation.
            await self.conn.complete_world_handoff(ev.peer_id)
            if not ok:
                # The grant was redeemed but the join body failed (bad code bundle,
                # subscribe failure...): tear the half-join down instead of leaving an
                # orphaned ESTABLISHED world Session, stale world routing and no retry.
                # GOAWAY the world Session (the host unlists us on its CLOSED),
                # un-point routing, and re-arm the rejoin (involuntary end).
                try:
                    await self.__purge(ev.peer_id, reason=LOCAL_ERROR)
                except Exception as e:
                    log.error(f"[session] failed-join teardown purge failed for ...{ev.peer_id[-5:]}: {e}")
                self.conn.set_world(None)
                self._maybe_rearm_rejoin_on_failed_join(ev)
            return

        # Responder on an agent node (public or in-world peer connection)
        # On a world host the world-side responder is the grant-redemption path
        # handled below; everything else lands here.
        if ev.role == SessionRole.RESPONDER and (not is_world or self.node_type is not Node.WORLD):
            if profile_b is not None and self.node_type is Node.AGENT:
                added = False
                try:
                    if await self.hosted.add_agent(peer_id=ev.peer_id,
                                                   profile=NodeProfile.from_dict(profile_b)):
                        added = True
                    else:
                        await self.__purge(ev.peer_id, reason=LOCAL_ERROR)
                except Exception as e:
                    log.error(f"[session] add_agent failed for ...{ev.peer_id[-5:]}: {e}")
                # Interactive mode waiting for a lone wolf: latch onto the first
                # inbound peer and arm interactive mode (mirrors the pre-Session
                # PROFILE handler). Only after a successful enrolment.
                if added:
                    opts = self._interact_mode_opts
                    if (isinstance(opts, dict) and 'lone_wolf_peer_id' in opts and
                            opts['lone_wolf_peer_id'] is None):
                        opts['lone_wolf_peer_id'] = ev.peer_id
                        opts['ready_to_interact'] = True
            return

        # Responder on the WORLD instance: a joiner redeemed a grant
        if ev.role == SessionRole.RESPONDER and is_world and self.node_type is Node.WORLD:
            await self._world_host_register_agent(ev, profile_b)

    async def _start_world_handoff(self, public_peer_id: str, grant: dict) -> None:
        """Joiner side: the public Session yielded a WorldGrant. Parse the world profile
        (profile domain, stays on Node) and hand the public<->world correlation + the
        world Session open to the Coordinator, dispatching the events it returns."""
        try:
            world_profile = NodeProfile.from_dict(
                json.loads(grant["world_profile_bytes"].decode("utf-8")))
        except Exception as e:
            log.error(f"World handoff, bad world profile: {e}")
            return
        world_dyn = world_profile.get_dynamic_profile()
        world_world_peer_id: str | None = world_dyn.get("private_peer_id")
        world_addrs = world_dyn.get("private_peer_addresses") or []
        if not world_world_peer_id:
            log.error("World handoff, world profile lacks private_peer_id")
            return

        blob = dict(grant)
        # The Coordinator records the correlation, points pool routing at the world host
        # (set_world) and opens the world Session; we just dispatch its events.
        for ev in await self.conn.start_world_handoff(
                public_peer_id, world_world_peer_id, list(world_addrs), blob,
                reject_relay=not self.talk_to_relay_based_nodes):
            await self._dispatch_session_event(ev)

    async def _world_host_register_agent(self, ev, profile_b) -> None:
        """World host: a joiner's world Session reached ESTABLISHED on the world
        instance — add it to the world agents/masters list (replaces the legacy
        post-approval bookkeeping). The grant is unsigned and there is no PRIVATE_JOIN,
        so the authoritative role is read from the host's own
        pre-admission record (set at the public decision), not from any join echo. A
        peer that was never pre-admitted reads role 0 here and is purged by the
        world-bit check below."""
        if profile_b is None:
            return
        try:
            profile = NodeProfile.from_dict(profile_b)
        except Exception as e:
            log.error(f"[session] world register: bad joiner profile: {e}")
            return
        role = self.conn.get_role(ev.peer_id)
        if role & Agent.ROLE_WORLD_AGENT != Agent.ROLE_WORLD_AGENT:
            log.error(f"[session] world register: grant role {role} lacks the world "
                      f"bit for ...{ev.peer_id[-5:]}; purging")
            await self.__purge(ev.peer_id, reason=HOSTED_REJECTED)
            return
        private_peer_id = ev.peer_id
        private_addr: list[str] | None = profile.get_dynamic_profile().get("private_peer_addresses")
        if self.world is None:
            return
        if not private_addr:
            # A joiner with no private addresses can't be a routable member: reject here
            # instead of asserting (which would tear down the world host's loop, and under
            # python -O would list a member with a None address). Unlist explicitly because
            # __purge does not touch the world lists, so an earlier pre-admission would leak.
            log.error(f"[session] world register: joiner ...{private_peer_id[-5:]} has no private "
                      f"addresses; rejecting")
            self._world_host_unlist_member(ev)
            await self.__purge(private_peer_id, reason=HOSTED_REJECTED)
            return
        try:
            # conn pools (routing) + world object (membership). The periodic rendezvous
            # publish (PUBLISH_RENDEZVOUS_EVERY) advertises the new member to the world.
            # Re-adding WITHOUT a TTL confirms the grant-time pre-admission (the
            # provisional deadline is dropped: the member is now registered).
            if role & Agent.ROLE_WORLD_MASTER == Agent.ROLE_WORLD_MASTER:
                self.conn.add_to_world_masters_list(private_peer_id, private_addr, role)
            else:
                self.conn.add_to_world_agents_list(private_peer_id, private_addr, role)
            if private_addr:
                self.conn.set_addresses_in_peer_info(private_peer_id, private_addr)
            if not (await self.world.add_agent(peer_id=private_peer_id, profile=profile)):
                log.error(f"[session] world.add_agent rejected ...{private_peer_id[-5:]}; purging")
                await self.__purge(private_peer_id, reason=HOSTED_REJECTED)
        except Exception as e:
            log.error(f"[session] world register failed for ...{private_peer_id[-5:]}: {e}")

    async def _join_world_after_session(self, world_peer_id: str, grant: dict | None) -> bool:
        """Joiner side: run the world-join body against an already-connected world
        Session, decoding the grant blob carried from the public Session."""
        if not grant:
            log.error("join_world: missing grant blob")
            return False
        try:
            raw_world_profile = json.loads(grant["world_profile_bytes"].decode("utf-8"))
            world_profile = NodeProfile.from_dict(raw_world_profile)
        except Exception as e:
            log.error(f"join_world: bad world profile: {e}")
            return False
        role = int(grant.get("role", 0))
        rendezvous_tag = int(grant.get("rendezvous_tag", 0))

        agent_files = roles_fsm = legacy_stats_code = None
        code = grant.get("code_bundle") or b""
        if code:
            try:
                bundle = json.loads(code.decode("utf-8"))
                agent_files: str | None = bundle.get("agent_files")
                roles_fsm: dict | None = bundle.get("roles_fsm")
                # Legacy worlds shipped stats.py as a dedicated field (and their
                # single-agent.py repack dropped it from the packed files)
                legacy_stats_code: str | None = bundle.get("stats_code")
            except Exception as e:
                log.error(f"[session] join_world: bad code_bundle: {e}")
        if not roles_fsm:
            # Legacy worlds shipped the FSMs inside the dynamic profile. The current
            # NodeProfile template drops that key at parse time, so read the raw dict.
            roles_fsm = (raw_world_profile.get('dynamic') or {}).get('world_roles_fsm')

        # Root-attested code hash from the public Session's verified token claims
        # (annotated by the joiner itself when the grant event was emitted)
        attested_code_hash = grant.get("world_code_hash_claim")

        initial_stats = None
        istats = grant.get("initial_stats") or b""
        if istats:
            try:
                initial_stats = json.loads(istats.decode("utf-8"))
            except Exception:
                initial_stats = None

        if agent_files is None:
            log.error("Missing agent files when joining the world")
            return False

        return await self.__join_world(
            profile=world_profile, role=role,
            packed_agent_files=agent_files, roles_fsm=roles_fsm,
            rendezvous_tag=rendezvous_tag, initial_stats=initial_stats,
            peer_id=world_peer_id, attested_code_hash=attested_code_hash,
            legacy_stats_code=legacy_stats_code,
        )

    async def connect_via_session(
        self,
        peer_id: str,
        addresses: list[str],
        p2p_name: str = NodeConn.P2P_PUBLIC,
        is_world_join: bool = False,
        remote_node_id_hint: str | None = None,
        reject_relay: bool = False,
    ) -> bool:
        """Start an initiator Session to `peer_id` via SessionManager and return.

        Fire-and-forget: Session.start launches the dial off-task and returns, and the whole
        dial → HELLO → PROFILE handshake completes asynchronously in the main loop's pump
        (deliver/tick). Returns True once the Session has been started; False only for the one
        synchronous rejection left, a quarantined peer. Dial failure and relay rejection now
        surface later as a FAILED event via the pump, not here. The Session owns the dial, so
        callers must NOT pre-dial.
        """
        events = await self.conn.initiate_session(
            p2p_name,
            peer_id=peer_id,
            addresses=addresses,
            is_world_join=is_world_join,
            remote_node_id_hint=remote_node_id_hint,
            reject_relay=reject_relay,
        )
        # The only synchronous FAILED among the start-up events is a quarantine refusal;
        # dial failure and relay rejection surface asynchronously via the pump.
        started = True
        for ev in events:
            if ev.type == SessionEventType.FAILED:
                started = False
            await self._dispatch_session_event(ev)
        return started

    async def __purge(self, peer_id: str, reason: int | None = None,
                      settled: bool = False) -> None:
        """Tear a peer down: drop its relation at the Coordinator and remove it from
        the hosted entity. (async)

        The teardown is Session-driven, the Connection is not visible here, and the
        transport is never severed under a farewell: ``conn.drop_peer`` dispatches on
        the Session state (see SessionManager.drop). With a ``reason`` on a live
        established Session it only GOAWAY-closes — the remote really does observe
        Session.Closed now, because the transport falls with the deferred settlement
        cut, after the ACK — while a purge of a mid-handshake, KNOWN or sessionless
        peer aborts/severs inline. With ``reason=None`` it reacts to a Session that
        already ended (the terminal-event dispatch, the polled non-grace transport
        drop): the SM has already settled it, so the drop is the tearing-down no-op
        and only the pool evict + hosted removal remain. The frequent double-purge
        is idempotent for the same reason. The events drop_peer returns are
        dispatched before the hosted removal.

        Args:
            peer_id: The peer ID of the node to purge.
            reason: Optional ReasonCode; when set and the Session is closable, it is
                GOAWAY-closed with this reason (inside drop_peer).
            settled: True for the reckoning of an already-settled end (the terminal
                dispatch, the polled transport drop): pools + hosted only, the SM is
                never touched — see NodeConn.drop_peer.
        """
        for ev in await self.conn.drop_peer(peer_id, reason, settled=settled):
            await self._dispatch_session_event(ev)

        await self.hosted.remove_agent(peer_id)

    @staticmethod
    def __sort_messages_by_priority(messages: list) -> list:
        """Sort messages by priority: action (interaction) requests first, then the rest.

        Args:
            messages: The list of Msg objects to sort.

        Returns:
            A new list with action requests first, then all other messages. The legacy
            world-approval / agent-approval priority tiers are gone with the pre-Session
            handshake (those content-types no longer exist).
        """

        _action_messages = []
        _other_messages = []
        for _msg in messages:
            if _msg.content_type == Msg.INTERACTION:
                _action_messages.append(_msg)
            else:
                _other_messages.append(_msg)
        return _action_messages + _other_messages

    def __root(self, api: str, payload: dict) -> dict | list:
        """Sends a POST request to the root server's API endpoint.

        Args:
            api: The API endpoint to send the request to.
            payload: The data to be sent in the request body.

        Returns:
            The 'data' field from the server's JSON response.

        Raises:
            RootServerError: on transport failures and, with api_rejected set, on
                API-level refusals (the API answers HTTP 200 with state.code != "ok");
                refusals carry the response envelope (state code and message, the
                data.code discriminator, the flags) for callers to branch on.
        """
        url = self.root_endpoint.rstrip("/") + "/" + api.lstrip("/")
        payload["node_token"] = self.node_token  # Adding token to let the server verify

        try:
            response = requests.post(url,
                                     json=payload,
                                     headers={"Content-Type": "application/json"},
                                     timeout=Custom.ROOT_REQUEST_TIMEOUT)
        except Exception as e:
            log.error(f"Request {url} failed: {e}")
            raise RootServerError(f"Request {api} failed: {e}", api=api) from e

        if response.status_code != 200:
            log.error(f"Request {url} failed with status code {response.status_code}")
            raise RootServerError(f"Request {api} failed with status code {response.status_code}",
                                  api=api, status=response.status_code)

        try:
            ret = response.json()
        except Exception as e:
            log.error(f"Request {url} returned a non-JSON body: {e}")
            raise RootServerError(f"Request {api} returned a non-JSON body", api=api,
                                  status=response.status_code) from e

        for field in ("state", "flags", "data"):
            if field not in ret:
                log.error(f"Missing key '{field}' in the response to {url}: {ret}")
                raise RootServerError(f"Missing key '{field}' in the response to {api}",
                                      api=api, status=response.status_code)

        state = ret['state'] if isinstance(ret['state'], dict) else {}
        if state.get('code') != "ok":
            data = ret['data']
            log.error("[" + url + "] " + str(state.get('message')))
            raise RootServerError("[" + api + "] " + str(state.get('message')),
                                  api=api, api_rejected=True,
                                  state_code=state.get('code'),
                                  state_message=state.get('message'),
                                  data_code=data.get('code') if isinstance(data, dict) else None,
                                  data=data,
                                  flags=ret['flags'] if isinstance(ret['flags'], dict) else {},
                                  status=response.status_code)
        return ret['data']

    def __replace_agent_instance_by_role(self, role: int,
                                         world_peer_id: str | None = None,
                                         world_profile: NodeProfile | None = None,
                                         stats_class=None,
                                         initial_stats=None,
                                         changing_an_existing_role_in_world: bool = False):
        """Generating a new instance of the Agent class accordingly to the given role (when in-world only)."""
        assert self.agent is not None

        if world_profile is None:
            world_profile = self.agent.world_profile

        if world_peer_id is None:
            world_peer_id = self.conn.get_world_node_peer_id()

        if self.world_agent_files is not None and len(self.world_agent_files) > 0:

            # Getting the roles delivered by the grant bundle at join time
            world_roles = (self.world_role_fsms or {}).keys()
            role_bits_to_str, role_str_to_bits = Agent.build_augmented_roles_dictionaries(world_roles)

            # Creating a new agent with the received actions
            if not self.agent.__class__.__name__ == "WAgent" or changing_an_existing_role_in_world:
                base_role_str = role_bits_to_str[(role >> 2) << 2]
                new_agent, new_agent_memory_finder = load_agent_in_memory(self.world_agent_files,
                                                                          base_role_str,
                                                                          proc=None)
            else:
                new_agent = self.agent
                new_agent_memory_finder = None

            # Saving new roles from the world ("custom roles") and building the augmented sets
            new_agent.CUSTOM_ROLES.clear()
            for r in world_roles:
                new_agent.CUSTOM_ROLES.append(r)
            new_agent.augment_roles(role_bits_to_str, role_str_to_bits)

            # Cloning attributes of the existing agent
            for key, value in self.agent.__dict__.items():
                if hasattr(new_agent, key):  # This will skip ROLE_BITS_TO_STR, CUSTOM_ROLES, etc...
                    if key == 'stats' and stats_class is not None:
                        new_agent.stats = stats_class(is_world=False)
                    else:
                        setattr(new_agent, key, value)

            # Fixing internal references to agent
            new_agent.im.agent = new_agent
            new_agent.proc.agent = new_agent

            # Telling the FSM that actions are related to this new agent
            # (the world FSM here is an empty FSM, keep it here to ensure its link with the new agent, in case of
            # a no-role world)
            new_agent.behav.set_actionable(new_agent)
            new_agent.behav_lone_wolf.set_actionable(new_agent)

            # Inheriting the pre-defined policy filter (if any)
            new_agent.set_policy_filter(self.agent.policy_filter, public=False)
            new_agent.set_policy_filter(self.agent.policy_filter_lone_wolf, public=True)

            # Updating node-level references
            old_agent = self.agent
            old_memory_finder = self.memory_finder
            self.agent = new_agent
            self.hosted = new_agent
            self.memory_finder = new_agent_memory_finder

        else:
            old_agent = self.agent
            old_memory_finder = None
            if stats_class is not None:
                log.misc("Replacing default stats with custom WStats from world.")
                old_agent.stats = stats_class(is_world=False)

        # Inject the stats history
        # If initial_stats is a string of an already generated HTML file, for example, we have to skip this part,
        # that's why we check if initial_stats is an instance of dict.
        assert self.agent is not None
        if initial_stats is not None and isinstance(initial_stats, dict):
            self.agent.update_stats_view(initial_stats, overwrite=True)

        # Saving the world profile and the role FSMs (accept_new_role reads the FSM
        # of the assigned role from them, at join time and on later role changes)
        self.agent.world_profile = world_profile
        self.agent.world_role_fsms = self.world_role_fsms

        # Setting the assigned role and default behavior (do it after having recreated the new agent object)
        self.agent.accept_new_role(role)  # Do this after having set world_profile/world_role_fsms

        # Updating wildcards
        self.agent.behav.update_wildcard(Custom.AGENT_WILDCARD, f"{self.get_world_peer_id()}")
        self.agent.behav.update_wildcard(Custom.WORLD_WILDCARD, f"{world_peer_id}")
        self.agent.behav.add_wildcards(old_agent.behav_wildcards)
        self.agent.behav.apply_wildcards()

        # Clearing memory
        if old_memory_finder is not None:
            old_memory_finder.cleanup()

    async def __handle_inspector_command(self, cmd: str, arg: str | None) -> None:
        """Handles commands received from an inspector node. (async)

        Args:
            cmd: The command string.
            arg: The argument for the command, or None if no argument is provided.
        """
        log.misc(f"Handling inspector message {cmd}, with arg {arg}")

        if arg is not None and not isinstance(arg, str):
            log.error("Expecting a string argument from the inspector!")
        else:
            if cmd == "ask_to_join_world":
                log.user(f"Inspector asked to join world: {arg}")
                if arg is None:
                    log.error("Missing inspector arguments!")
                else:
                    await self.ask_to_join_world(node_name=arg)
            elif cmd == "ask_to_get_in_touch":
                log.user(f"Inspector asked to get in touch with an agent: {arg}")
                await self.ask_to_get_in_touch(node_name=arg, public=True)
            elif cmd == "leave":
                log.user(f"Inspector asked to leave an agent: {arg}")
                if arg is None:
                    log.error("Missing inspector arguments!")
                else:
                    await self.__purge(arg, reason=WORLD_DISENGAGEMENT)
            elif cmd == "leave_world":
                log.user("Inspector asked to leave the current world")
                await self.leave_world()
            elif cmd == "pause":
                log.user("Inspector asked to pause")
                self.__inspector_told_to_pause = True
            elif cmd == "play":
                log.user("Inspector asked to play")
                self.__inspector_told_to_pause = False
            elif cmd == "save":
                log.user("Inspector asked to save")
                if arg is None:
                    log.error("Missing inspector arguments!")
                else:
                    self.hosted.save(arg)
            else:
                log.error(f"Unknown inspector command: {cmd}")

    async def __send_to_inspector(self) -> None:
        """Sends status updates and data to the connected inspector node. (async)"""

        # Collecting console
        console = log.get_inspector_console()

        # Collecting the HSM
        if self.__inspector_cache['behav'] != self.hosted.behav:
            self.__inspector_cache['behav'] = self.hosted.behav
            behav = str(self.hosted.behav)
        else:
            behav = None

        # Collecting status of the HSM
        if self.hosted.behav is not None:
            _behav = self.hosted.behav
            state = _behav.get_state().id if _behav.get_state() is not None else None
            action = _behav.get_action().id if _behav.get_action() is not None else None
            behav_status = {'state': state, 'action': action,
                            'state_with_action': _behav.get_state().has_action()
                            if (state is not None) else False}
        else:
            behav_status = None

        # Collecting known agents
        if self.__inspector_cache['all_agents_count'] != len(self.hosted.all_agents):
            self.__inspector_cache['all_agents_count'] = len(self.hosted.all_agents)
            all_agents_profiles = {k: v.get_all_profile() for k, v in self.hosted.all_agents.items()}

            # Inspector expects also to have access to the profile of the world,
            # so we patch this thing by adding it here. This is the REMOTE world-host's
            # peer-id (conn.get_world_node_peer_id), NOT our own local world peer-id
            # (self.get_world_peer_id / conn.world_peer_id).
            world_node_peer_id = self.conn.get_world_node_peer_id()
            if self.hosted.in_world() and world_node_peer_id is not None:
                all_agents_profiles[world_node_peer_id] = self.hosted.world_profile.get_all_profile()
        else:
            all_agents_profiles = None

        # Collecting known streams info
        if self.__inspector_cache['known_streams_count'] != len(self.hosted.known_streams):
            self.__inspector_cache['known_streams_count'] = len(self.hosted.known_streams)
            known_streams_props = {(k + "-" + name): v.get_props().to_dict() for k, stream_dict in
                                   self.hosted.known_streams.items() for name, v in stream_dict.items()}
        else:
            known_streams_props = None

        # Packing console, HSM status, and possibly HSM
        console_behav_status_and_behav = {'console': console,
                                          'behav': behav,
                                          'behav_status': behav_status,
                                          'all_agents_profiles': all_agents_profiles,
                                          'known_streams_props': known_streams_props}

        # Sending console, HSM status, and possibly HSM to the inspector
        if not (await self.conn.send(self.inspector_peer_id, channel_trail=None,
                                     content_type=Msg.CONSOLE_AND_BEHAV_STATUS,
                                     content=console_behav_status_and_behav)):
            log.error("Failed to send data to the inspector")

        # Sending stream data (not pubsub) to the inspector
        my_peer_ids = (self.get_public_peer_id(), self.get_world_peer_id())
        for net_hash, streams_dict in self.hosted.known_streams.items():
            peer_id = DataProps.peer_id_from_net_hash(net_hash)

            # Preparing sample dict
            something_to_send = False
            content = {name: {} for name in streams_dict.keys()}
            for name, stream in streams_dict.items():
                data = stream.get(requested_by="__send_to_inspector")

                if data is not None:
                    something_to_send = True

                log.debug(f"[__send_to_inspector] Preparing to send stream samples from {net_hash}, {name}")
                content[(peer_id + "|" + name) if peer_id not in my_peer_ids else name] = \
                    {'data': data, 'data_tag': stream.get_tag(), 'data_uuid': None}

            # Checking if there is something valid in this group of streams to send to inspector
            if not something_to_send:
                log.debug(f"[__send_to_inspector] No stream samples to send to inspector for {net_hash}, "
                          f"all internal streams returned None")
                continue

            log.debug(f"[__send_to_inspector] Sending samples of {net_hash} by direct message, to inspector")
            name_or_group = DataProps.name_or_group_from_net_hash(net_hash)
            if not (await self.conn.send(self.inspector_peer_id, channel_trail=name_or_group,
                                         content_type=Msg.STREAM_SAMPLE, content=content)):
                log.error(f"Failed to send stream sample data to the inspector (hash: {net_hash})")


class NodeSynchronizer:

    def __init__(self):
        """Initializes a new instance of the NodeSynchronizer class."""
        self.nodes = []
        self.agent_nodes = {}
        self.world_node = None  # Added to allow get_console() to access the world node from server.py (synch only)
        self.streams = {}
        self.world = None
        self.world_masters = set()
        self.world_masters_node_ids = None
        self.agent_name_to_profile = {}
        self.synch_cycle = -1
        self.synch_cycles = -1
        self.gap = 1.0

    def add_node(self, node: 'Node') -> None:
        """Adds a new node to the synchronizer.

        Args:
            node: The node to add.
        """
        self.nodes.append(node)

        if node.node_type == Node.AGENT:
            assert node.agent is not None
            self.agent_nodes[node.agent.get_name()] = node
            if self.world_masters_node_ids is not None:
                if node.node_id in self.world_masters_node_ids:
                    self.world_masters.add(node.agent.get_name())
            self.agent_name_to_profile[node.agent.get_name()] = node.agent.get_profile()
        elif node.node_type == Node.WORLD:
            self.world_node = node
            self.world = node.world
            self.world_masters_node_ids = node.world_masters_node_ids
            if self.world_masters_node_ids is None:
                self.world_masters_node_ids = set()
            for node in self.nodes:
                if node.node_id in self.world_masters_node_ids:
                    self.world_masters.add(node.agent.get_name())

    async def run(self, addresses: list[str] | None, synch_cycles: int | None = None) -> None:
        """Starts the main execution loop for the synchronizer. (async)

        Args:
            addresses: Addresses of the world to connect to, or None if not joining a world.
            synch_cycles: The number of synchronized clock cycles to run. If None, runs indefinitely.
        """
        if self.world is None:
            log.critical("Missing world node")

        # Main loop
        self.synch_cycles = synch_cycles
        self.synch_cycle = 0

        try:
            while True:
                state_changed = False
                world_node: Node | None = None
                for node in self.nodes:
                    if node.node_type == Node.AGENT:
                        await node.run_async(cycles=1, join_world=addresses if self.synch_cycle == 0 else None)
                        if self.gap > 0.:
                            time.sleep(self.gap)
                        state_changed = state_changed or node.agent.behav.get_state_changed()
                    else:
                        world_node = node
                if world_node is not None:
                    await world_node.run_async(cycles=1)
                    if self.gap > 0.:
                        time.sleep(self.gap)

                self.synch_cycle += 1

                # Stop condition on the number of cycles
                if self.synch_cycles is not None and self.synch_cycle == self.synch_cycles:
                    break
        except KeyboardInterrupt:
            pass
        finally:

            # The synchronizer drives each node one cycle at a time, so it is the only
            # one that knows when they are done: shut them down here (world last, as the
            # agents still GOAWAY it while leaving).
            for node in reversed(self.nodes):
                await node.aclose()
