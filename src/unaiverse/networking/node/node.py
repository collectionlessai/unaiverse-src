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
import ast
import cv2
import copy
import json
import math
import time
import html
import queue
import types
import requests
import threading
import traceback
from PIL import Image
from typing import Optional
from collections import deque
from unaiverse.clock import Clock
from unaiverse.world import World
from unaiverse.agent import Agent
from unaiverse.networking.p2p import P2P, P2PError
from unaiverse.networking.p2p.messages import Msg
from datetime import datetime, timezone, timedelta
from unaiverse.networking.node.connpool import NodeConn
from unaiverse.networking.node.profile import NodeProfile
from unaiverse.streams import DataProps, BufferedDataStream
from unaiverse.utils.misc import GenException, get_key_considering_multiple_sources, save_node_addresses_to_file


class Node:

    # Each node can host an agent or a world
    AGENT = "agent"  # Artificial agent
    WORLD = "world"  # World agent

    # Each node outputs console text with a different color
    TEXT_COLORS = ('\033[91m', '\033[94m', '\033[92m', '\033[93m')
    TEXT_LAST_USED_COLOR = 0
    TEXT_LOCK = threading.Lock()

    def __init__(self,
                 hosted: Agent | World,
                 unaiverse_key: str | None = None,
                 node_name: str | None = None,
                 node_id: str | None = None,
                 hidden: bool = False,
                 clock_delta: float = 1. / 25.,
                 base_identity_dir: str = "./unaiverse_nodes_identity",
                 only_certified_agents: bool = False,
                 allowed_node_ids: list[str] | set[str] = None,  # Optional: it is loaded from the online profile
                 world_masters_node_ids: list[str] | set[str] = None,  # Optional: it is loaded from the online profile
                 world_masters_node_names: list[str] | set[str] = None,  # Optional: it will be converted to node IDs
                 allow_connection_through_relay: bool = True,
                 talk_to_relay_based_nodes: bool = True):
        """Initializes a new instance of the Node class.

        Args:
            hosted: The Agent or World entity hosted by this node.
            unaiverse_key: The UNaIVERSE key for authentication (if None, it will be loaded from env var or cache file,
                or you will be asked for it).
            node_name: A human-readable name for the node (using node ID is preferable; use this or node ID, not both).
            node_id: A unique identifier for the node (use this or the node name, not both).
            hidden: A flag to determine if the node is hidden (i.e., only the owner of the account can see it).
            clock_delta: The minimum time delta for the node's clock.
            only_certified_agents: A flag to allow only certified agents to connect.
            allowed_node_ids: A list or set of allowed node IDs to connect (t is loaded from the online profile).
            world_masters_node_ids: A list or set of world masters' node IDs (it is also loaded from online profile).
            world_masters_node_names: A list or set of world masters' node names (using IDs is preferable).
            allow_connection_through_relay: A flag to allow connections through a relay.
            talk_to_relay_based_nodes: A flag to allow talking to relay-based nodes.
        """

        # Checking main arguments
        if not (isinstance(hosted, Agent) or isinstance(hosted, World)):
            raise GenException("Invalid hosted entity, must be Agent or World")
        if not (node_id is None or isinstance(node_id, str)):
            raise GenException("Invalid node ID")
        if not (node_name is None or isinstance(node_name, str)):
            raise GenException("Invalid node name")
        if not (node_name is None or node_id is None):
            raise GenException("Cannot specify both node ID and node name")
        if not (node_name is not None or node_id is not None):
            raise GenException("You must specify either node ID or node name: both are missing")
        if not (unaiverse_key is None or isinstance(unaiverse_key, str)):
            raise GenException("Invalid UNaIVERSE key")

        # Main attributes
        self.node_id = node_id
        self.unaiverse_key = unaiverse_key
        self.hosted = hosted
        self.node_type = Node.AGENT if (isinstance(hosted, Agent) and not isinstance(hosted, World)) else Node.WORLD
        self.agent = hosted if self.node_type is Node.AGENT else None
        self.world = hosted if self.node_type is Node.WORLD else None
        self.clock = Clock(min_delta=clock_delta)  # Node clock
        self.conn = None  # Manages the network operations in the P2P network
        self.talk_to_relay_based_nodes = talk_to_relay_based_nodes

        # Expected properties of the nodes that will try to connect to this one
        self.only_certified_agents = only_certified_agents
        self.allowed_node_ids = set(allowed_node_ids) if allowed_node_ids is not None else None
        self.world_masters_node_ids = set(world_masters_node_ids) if world_masters_node_ids is not None else None

        # Profile
        self.profile = None
        self.send_dynamic_profile_every = 10. if self.node_type is Node.WORLD else 10.  # Seconds
        self.get_new_token_every = 23 * 60. * 60. + 30 * 60.  # Seconds (23 hours and 30 minutes, safer)

        # Rendezvous
        self.publish_rendezvous_every = 10.
        self.last_rendezvous_time = 0.

        # Automatic address update and relay refresh (if needed)
        self.relay_reservation_expiry: Optional[datetime] = None
        self.address_check_every = 5 * 60.  # Check every 5 minutes

        # Interview of newly connected nodes
        self.interview_timeout = 45.  # Seconds
        self.connect_without_ack_timeout = 45.  # Seconds
        
        # Alive messaging
        self.send_alive_every = 2.5 * 60.  # Seconds
        self.last_alive_time = 0.
        self.skip_was_alive_check = os.getenv("NODE_IGNORE_ALIVE", "0") == "1"

        # Alive messaging
        self.run_start_time = 0.

        # Root server-related
        self.root_endpoint = 'https://unaiverse.io/api'  # WARNING: EDITING THIS ADDRESS VIOLATES THE LICENSE
        self.node_token = ""
        self.public_key = ""

        # Output console text
        print_level = int(os.getenv("NODE_PRINT", "0"))  # 0, 1, 2
        self.print_enabled = print_level > 0
        self.cursor_hidden = False
        NodeSynchronizer.DEBUG = print_level > 1
        NodeConn.DEBUG = print_level > 1
        if print_level == 0:
            self.cursor_hidden = True
        with Node.TEXT_LOCK:
            self.text_color = Node.TEXT_COLORS[Node.TEXT_LAST_USED_COLOR]
            Node.TEXT_LAST_USED_COLOR = (Node.TEXT_LAST_USED_COLOR + 1) % len(Node.TEXT_COLORS)

        # Print-related logging (for inspector only)
        self._output_messages = [""] * 20
        self._output_messages_ids = [-1] * 20
        self._output_messages_count = 0
        self._output_messages_last_pos = -1

        # Attributes: handshake-related
        self.agents_to_interview: dict[str, [float, NodeProfile | None]] = {}  # Peer_id -> [time, profile | None]
        self.agents_expected_to_send_ack = {}
        self.last_rejected_agents = deque(maxlen=self.conn)
        self.joining_world_info = None
        self.first = True

        # Inspector related
        self.inspector_activated = False
        self.inspector_peer_id = None
        self.debug_server_running = False
        self.__inspector_cache = {"behav": None, "known_streams_count": 0, "all_agents_count": 0}
        self.__inspector_told_to_pause = False

        # Get key
        self.unaiverse_key = get_key_considering_multiple_sources(self.unaiverse_key)

        # Getting node ID (retrieving by name), if it was not provided (the node is created if not existing)
        if self.node_id is None:
            node_ids, were_alive = self.get_node_id_by_name([node_name],
                                                            create_if_missing=True)
            self.node_id = node_ids[0]
            if were_alive[0]:
                raise GenException(f"Cannot access node {node_name}, it is already running! "
                                   f"(set env variable NODE_IGNORE_ALIVE=1 to ignore this control)")
        
        # Automatically create a unique data directory for this specific node
        node_identity_dir = os.path.join(base_identity_dir, self.node_id)
        p2p_u_identity_dir = os.path.join(node_identity_dir, "p2p_public")
        p2p_w_identity_dir = os.path.join(node_identity_dir, "p2p_private")

        # Getting node ID of world masters, if needed
        if world_masters_node_names is not None and len(world_masters_node_names) > 0:
            master_node_ids, were_alive = self.get_node_id_by_name(world_masters_node_names,
                                                                   create_if_missing=True, node_type=Node.AGENT)
            for master_node_name, master_node_id in zip(world_masters_node_names, master_node_ids):
                if master_node_id is None:
                    raise GenException(f"Cannot find world master node ID given its name: {master_node_name}")
                else:
                    if self.world_masters_node_ids is None:
                        self.world_masters_node_ids = set()
                    self.world_masters_node_ids.add(master_node_id)

        # Here you can setup max_instances, max_channels, enable_logging at libp2p level etc.
        P2P.setup_library(enable_logging=os.getenv("NODE_LIBP2PLOG", "0") == "1")

        offer_relay_facilities = self.node_type is Node.WORLD  # Only world nodes offer relay facilities
        
        # --- PARALLEL P2P NODE CREATION ---
        # 1. Define configurations for both nodes
        p2p_u_config = {
            "identity_dir": p2p_u_identity_dir,
            "port": int(os.getenv("NODE_STARTING_PORT", "0")),
            "ips": None,
            "enable_relay_client": allow_connection_through_relay,
            "enable_relay_service": offer_relay_facilities,
            "knows_is_public": os.getenv("NODE_IS_PUBLIC", "0") == "1",
            "max_connections": 1000,
            "enable_tls": os.getenv("NODE_USE_TLS", "0") == "1",
            "domain_name": os.getenv("DOMAIN", None),
            "tls_cert_path": os.getenv("TLS_CERT_PATH", None),
            "tls_key_path": os.getenv("TLS_KEY_PATH", None),
        }
        
        p2p_w_config = {
            "identity_dir": p2p_w_identity_dir,
            "port": (int(os.getenv("NODE_STARTING_PORT", "0")) + 4)
                    if int(os.getenv("NODE_STARTING_PORT", "0")) > 0 else 0,
            "ips": None,
            "enable_relay_client": allow_connection_through_relay,
            "enable_relay_service": offer_relay_facilities,
            "knows_is_public": os.getenv("NODE_IS_PUBLIC", "0") == "1",
            "max_connections": 1000,
            "enable_tls": os.getenv("NODE_USE_TLS", "0") == "1",
            "domain_name": os.getenv("DOMAIN", None),
            "tls_cert_path": os.getenv("TLS_CERT_PATH", None),
            "tls_key_path": os.getenv("TLS_KEY_PATH", None),
        }

        # 2. Prepare a dictionary to store results or exceptions
        results = {
            "p2p_u": None,
            "p2p_w": None
        }
        
        # 3. Define the worker function for the threads
        def create_p2p_instance(name: str, config: dict):
            try:
                # This is the slow, blocking call
                instance = P2P(**config)
                results[name] = instance
            except Exception as e:
                # Store the exception if creation fails
                results[name] = e
        
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
        p2p_u = results["p2p_u"]
        p2p_w = results["p2p_w"]

        if isinstance(p2p_u, Exception):
            # We must re-raise the exception to fail the Node creation
            raise P2PError(f"Failed to initialize public P2P node (p2p_u): {p2p_u}") from p2p_u
        if isinstance(p2p_w, Exception):
            raise P2PError(f"Failed to initialize private P2P node (p2p_w): {p2p_w}") from p2p_w
        if p2p_u is None or p2p_w is None:
            # This should not happen if threads ran, but it's a safe check
            raise P2PError("P2P node creation did not complete, but no exception was caught.")

        # Get first node token
        self.get_node_token(peer_ids=[p2p_u.peer_id, p2p_w.peer_id])  # Passing both the peer IDs

        # Get first badge token
        if self.node_type is Node.WORLD:
            self.badge_token = self.__root(api="account/node/cv/badge/token/get", payload={"node_id": self.node_id})
        else:
            self.badge_token = None

        # Get profile (static)
        profile_static = self.__root(api="/account/node/profile/static/get", payload={"node_id": self.node_id})

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
                             p2p_u=p2p_u,
                             p2p_w=p2p_w,
                             is_world_node=self.node_type is Node.WORLD,
                             public_key=self.public_key,
                             token=self.node_token)

        # Get CV
        cv = self.get_cv()

        # Creating full node profile putting together static info, dynamic profile, adding P2P node info, CV
        self.profile = NodeProfile(static=profile_static,
                                   dynamic={'peer_id': p2p_u.peer_id,
                                            'peer_addresses': p2p_u.addresses,
                                            'private_peer_id': p2p_w.peer_id,
                                            'private_peer_addresses': p2p_w.addresses,
                                            'connections': {
                                                'role': self.hosted.ROLE_BITS_TO_STR[self.hosted.ROLE_PUBLIC]
                                            },
                                            'world_summary': {
                                                'world_name':
                                                    profile_static['node_name']
                                                    if self.node_type is Node.WORLD else None
                                            },
                                            "world_roles_fsm": None,  # This will be filled later if this is a world
                                            "world_stats_dynamic": None,  # This will be filled later if this is a world
                                            "hidden": hidden  # Marking the node as hidden (or not)
                                            },
                                   cv=cv)  # Adding CV here

        # Sharing node-level info with the hosted entity
        self.hosted.set_node_info(self.clock, self.conn, self.profile, self.out, self.ask_to_get_in_touch,
                                  self.__purge, self.agents_expected_to_send_ack, print_level)

        # Finally, sending dynamic profile to the root server
        # (send AFTER set_node_info, not before, since set_node_info updates the profile,
        # adding world roles and state machines)
        self.send_dynamic_profile()

        # Save public addresses
        path_to_append_addresses = os.getenv("NODE_SAVE_RUNNING_ADDRESSES")
        if path_to_append_addresses is not None and os.path.exists(path_to_append_addresses):
            save_node_addresses_to_file(self, public=True, dir_path=path_to_append_addresses,
                                        filename="running.csv", append=True)

        # Update lone-wolf machines to replace default wildcards (like <agent>) - the private one will be handled when
        # joining a world
        if self.node_type is Node.AGENT:
            self.agent.behav_lone_wolf.update_wildcard("<agent>", f"{self.get_public_peer_id()}")

    def out(self, msg: str):
        """Prints a formatted message to the console if printing is enabled.

        Args:
            msg: The message to be printed.
        """
        if self.print_enabled:
            s = (f"{self.node_type[0:2]}: " +
                 ((self.hosted.get_name())[0:6] + ",").ljust(7) +
                 f" cy: {self.clock.get_cycle()}")
            s = f"[{s}] {msg}"
            print(f"{self.text_color}{s}\033[0m")

        if self.inspector_activated or self.debug_server_running:
            last_id = self._output_messages_ids[self._output_messages_last_pos]
            self._output_messages_last_pos = (self._output_messages_last_pos + 1) % len(self._output_messages)
            self._output_messages_count = min(self._output_messages_count + 1, len(self._output_messages))
            self._output_messages_ids[self._output_messages_last_pos] = last_id + 1
            self._output_messages[self._output_messages_last_pos] = html.escape(str(msg), quote=True)

    def err(self, msg: str):
        """Prints a formatted error message to the console.

        Args:
            msg: The error message to be printed.
        """
        when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        if self.print_enabled:
            self.out(f"<ERROR> [{when}] " + msg)
        else:
            print(f"<ERROR> [{when}] " + msg)

    def get_node_id_by_name(self, node_names: list[str], create_if_missing: bool = False,
                            node_type: str | None = None) -> tuple[list[str], list[bool]]:
        """Retrieves the node ID by its name from the root server, creating a new node if it's missing and specified.

        Args:
            node_names: The list with the names of the nodes to retrieve.
            create_if_missing: A flag to create the node if it doesn't exist (only valid for your own nodes).
            node_type: The type of the node to create if missing (when create_if_missing is True) - default: the type of
                the current node.

        Returns:
            The list of node IDs and the list of boolean flags telling if a node was already alive,
            or an exception if an error occurs.
        """
        try:
            response = self.__root("/account/node/get/id",
                                   payload={"node_name": node_names,
                                            "account_token": self.unaiverse_key})
            node_ids = []
            were_alive = []
            missing = []
            for i in range(0, len(response["nodes"])):
                if response["nodes"][i] is not None:
                    node_ids.append(response["nodes"][i]["node_id"])
                    were_alive.append(response["nodes"][i]["was_alive"])
                else:
                    node_ids.append(None)
                    were_alive.append(None)
                missing.append(i)
        except Exception as e:
            raise GenException(f"Error while retrieving nodes named {node_names} from server! [{e}]")

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
                    raise GenException(f"Error while registering node named {node_name} in server! [{e}]")
        return node_ids, were_alive

    def send_alive(self) -> bool:
        """Send an alive message to the root server.

        Returns:
            A boolean flag indicating whether the node was already live before sending this.
        """
        try:
            response = self.__root("/account/node/alive",
                                   payload={"node_id": self.node_id,
                                            "account_token": self.unaiverse_key})
            return response["was_alive"]
        except Exception as e:
            self.err(f"Error while sending alive message to server! [{e}]")

    def get_node_token(self, peer_ids):
        """Generates and retrieves a node token from the root server.

        Args:
            peer_ids: A list of public and private peer IDs.
        """
        response = None

        for i in range(0, 3):  # It will try 3 times before raising the exception...
            try:
                response = self.__root("/account/node/token/generate",
                                       payload={"node_id": self.node_id,
                                                "account_token": self.unaiverse_key
                                                if self.node_token is None or len(self.node_token) == 0 else None,
                                                "node_token": self.node_token, "peer_ids": json.dumps(peer_ids)})
                break
            except Exception as e:
                if i < 2:
                    self.err(f"Error while getting token from server, retrying...")
                    time.sleep(1)  # Wait a little bit
                else:
                    raise GenException(f"Error while getting token from server [{e}]")  # Raise the exception

        self.node_token = response["token"]
        self.public_key = response["public_key"]

        # Sharing the token with the connection manager
        if self.conn is not None:
            self.conn.set_token(self.node_token)

    def get_cv(self):
        """Retrieves the node's CV (Curriculum Vitae) from the root server

        Returns:
            The node's CV as a dictionary.
        """
        for i in range(0, 3):  # It will try 3 times before raising the exception...
            try:
                return self.__root(api="/account/node/cv/get", payload={"node_id": self.node_id})
            except Exception as e:
                self.err(f"Error while getting CV from server [{e}]")
                if i < 2:
                    self.out("Retrying...")
                    time.sleep(1)  # Wait a little bit
                else:
                    raise GenException(f"Error while getting CV from server [{e}]")

    def send_dynamic_profile(self):
        """Sends the node's dynamic profile to the root server."""
        try:
            self.__root(api="/account/node/profile/dynamic/post", payload={"node_id": self.node_id,
                                                                           "profile":
                                                                               self.profile.get_dynamic_profile()})
        except Exception as e:
            self.err(f"Error while sending dynamic profile to from server [{e}]")

    def send_badges(self):
        """Sends new badges assigned by a world node to the root server and notifies the agents."""
        if self.node_type is Node.WORLD:
            peer_id_to_badges = self.world.get_all_badges()
            if len(peer_id_to_badges) > 0:
                self.out(f"Sending {len(peer_id_to_badges)} badges to root server")
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
                                self.err(f"Error while posting a badge assigned to {peer_ids[z]}. Badge: {badges[z]}. "
                                         f"Error message: invalid response format")
                            else:
                                if ret['state']['code'] != "ok":
                                    self.err(f"Error while posting a badge assigned to {peer_ids[z]}. "
                                             f"Badge: {badges[z]}. "
                                             f"Error message: {ret['state']['message']}")
                                else:
                                    peer_ids_to_notify.add(peer_ids[z])

                        # Notify agents
                        for peer_id in peer_ids_to_notify:
                            if not self.conn.send(peer_id, channel_trail=None, content=None,
                                                  content_type=Msg.GET_CV_FROM_ROOT):
                                self.err(f"Error while sending the request to re-download CV to peer {peer_id}")

                        # Clearing
                        self.world.clear_badges()
                        break
                    except Exception as e:
                        self.err(f"Error while sending badges to server or when notifying peers [{e}]")
                        if i < 2:
                            self.out("Retrying...")
                            time.sleep(1)  # Wait a little bit
                        else:
                            self.err(f"Couldn't complete badge sending or notification procedure (stop trying)")

    def get_public_addresses(self) -> list[str]:
        """Returns the public addresses of the P2P node

        Returns:
            The list of public addresses.
        """
        return self.conn[NodeConn.P2P_PUBLIC].addresses

    def get_world_addresses(self) -> list[str]:
        """Returns the world addresses of the P2P node

        Returns:
            The list of world addresses.
        """
        return self.conn[NodeConn.P2P_WORLD].addresses

    def get_public_peer_id(self) -> str:
        """Returns the public peer ID of the P2P node

        Returns:
            The public peer ID.
        """
        return self.conn[NodeConn.P2P_PUBLIC].peer_id

    def get_world_peer_id(self) -> str:
        """Returns the world peer ID of the P2P node

        Returns:
            The world peer ID.
        """
        return self.conn[NodeConn.P2P_WORLD].peer_id

    def ask_to_get_in_touch(self, node_name: str | None = None, addresses: list[str] | None = None, public: bool = True,
                            before_updating_pools_fcn=None, run_count: int = 0):
        """Tries to connect to another agent or world node.

        Args:
            node_name: Name of the node to join (alternative to addresses below)
            addresses: A list of network addresses to connect to (alternative to node_name).
            public: A boolean flag indicating whether to use the public or world P2P network.
            before_updating_pools_fcn: A function to call before updating the connection pools.
            run_count: The number of connection attempts made.

        Returns:
            The peer ID of the connected node if successful, otherwise None.
        """

        # Checking arguments
        if (node_name is None and addresses is None) or (node_name is not None and addresses is not None):
            raise GenException("Cannot specify both node_name and addresses or none of them, check your code!")

        # Getting addresses, if needed
        if addresses is None:
            addresses = self.__root(api="account/node/get/addresses",
                                    payload={"node_name": node_name, "account_token": self.unaiverse_key})["addresses"]

        # Connecting
        self.out("Connecting to another agent/world...")
        peer_id, through_relay = self.conn.connect(addresses,
                                                   p2p_name=NodeConn.P2P_PUBLIC if public else NodeConn.P2P_WORLD)

        if through_relay:
            print("Warning: this connection goes through a relay-based circuit, "
                  "so a third-party node is involved in the communication")

        if peer_id is not None and (not through_relay or self.talk_to_relay_based_nodes):

            # Ping to test the readiness of the established connection
            self.out(f"Connected, ping-pong...")
            if not self.conn.send(peer_id, channel_trail=None, content_type=Msg.MISC, content={"ping": "pong"},
                                  p2p=self.conn.p2p_name_to_p2p[NodeConn.P2P_PUBLIC if public else NodeConn.P2P_WORLD]):
                if run_count < 2:
                    return self.ask_to_get_in_touch(addresses=addresses, public=public,
                                                    before_updating_pools_fcn=before_updating_pools_fcn,
                                                    run_count=run_count+1)
                else:
                    self.err("Connection failed! (ping-pong max trials exceeded)")
                    return None

            self.out("Connected, updating pools...")
            if before_updating_pools_fcn is not None:
                before_updating_pools_fcn(peer_id)
            self.conn.update()

            self.agents_expected_to_send_ack[peer_id] = self.clock.get_time()
            self.out(f"Current set of {len(self.agents_expected_to_send_ack)} connected peer IDs that will get our "
                     f"profile and are expected to send a confirmation: "
                     f"{self.agents_expected_to_send_ack.keys()}")
            return peer_id
        else:
            self.err("Connection failed!")
            return None

    def ask_to_join_world(self, node_name: str | None = None, addresses: list[str] | None = None, **kwargs):
        """Initiates a request to join a world.

        Args:
            node_name: The name of the node hosting the world to join (alternative to addresses below).
            addresses: A list of network addresses of the world node (alternative to world_name).
            **kwargs: Additional options for joining the world.

        Returns:
            The public peer ID of the world node if the connection request is successful, otherwise None.
        """
        print("Asking to join world...")

        # Leave an already entered world (if any)
        world_peer_id = self.profile.get_dynamic_profile()['connections']['world_peer_id']
        if world_peer_id is not None:
            self.leave(world_peer_id)

        # Connecting to the world (public)
        peer_id = self.ask_to_get_in_touch(node_name=node_name, addresses=addresses, public=True)

        # Saving info
        if peer_id is not None:
            print("Connected on the public network, waiting for handshake...")
            self.joining_world_info = {"world_public_peer_id": peer_id, "options": kwargs}
        else:
            print("Failed to join world!")
        return peer_id

    def leave(self, peer_id: str):
        """Disconnects the node from a specific peer, typically a world.

        Args:
            peer_id: The peer ID of the node to leave.
        """

        if not isinstance(peer_id, str):
            self.err(f"Invalid argument provided to leave(...): {peer_id}")
            return

        print(f"Leaving {peer_id}...")

        dynamic_profile = self.profile.get_dynamic_profile()

        if peer_id == dynamic_profile['connections']['world_peer_id']:
            print("Leaving world...")

            # Clearing world-related lists in the connection manager (to avoid world agent to connect again)
            self.conn.set_world(None)
            self.conn.set_world_agents_list(None)
            self.conn.set_world_masters_list(None)

            # Disconnecting all connected world-related agents, including world node (it clears roles too)
            self.conn.remove_all_world_agents()

            # Better clear this as well
            if peer_id in self.agents_expected_to_send_ack:
                del self.agents_expected_to_send_ack[peer_id]

            # Clear profile
            dynamic_profile['connections']['world_peer_id'] = None
            dynamic_profile['connections']['world_agents'] = None
            dynamic_profile['connections']['world_masters'] = None
            self.profile.mark_change_in_connections()

            # Clearing agent-level info
            self.agent.clear_world_related_data()

            # Clearing all joining options
            self.joining_world_info = None
        else:
            if peer_id in self.hosted.all_agents:
                self.hosted.remove_agent(peer_id)
            self.conn.remove(peer_id)

    def leave_world(self):
        """Initiates the process of leaving a world.

        Returns:
            None.
        """
        if self.profile.get_dynamic_profile()['connections']['world_peer_id'] is not None:
            self.agent.accept_new_role(self.agent.ROLE_PUBLIC)
            self.agent.world_profile = None
            self.leave(self.profile.get_dynamic_profile()['connections']['world_peer_id'])

    def run(self, cycles: int | None = None, max_time: float | None = None, interact_mode_opts: dict | None = None):
        """Starts the main execution loop for the node.

        Args:
            cycles: The number of clock cycles to run the loop for. If None, runs indefinitely.
            max_time: The maximum time in seconds to run the loop. If None, runs indefinitely.
            interact_mode_opts: A dictionary of options to enable interactive mode.
        """
        try:
            if self.cursor_hidden:
                sys.stdout.write("\033[?25l")  # Hide cursor

            last_dynamic_profile_time = self.clock.get_time()
            last_get_token_time = self.clock.get_time()
            last_address_check_time = self.clock.get_time()
            if not (cycles is None or cycles > 0):
                raise GenException("Invalid number of cycles")

            # Interactive mode (useful when chatting with lone wolves)
            keyboard_queue = None
            keyboard_listener = None
            processor_net_hash = None
            processor_img_stream = None
            processor_text_stream = None
            cap = None
            ready_to_interact = False

            if interact_mode_opts is not None:
                if self.agent is None:
                    raise GenException("Interactive mode is only valid for agents")

                processor_text_net_hash = None
                processor_img_net_hash = None
                looking_for_public_streams = "lone_wolf_peer_id" in interact_mode_opts
                for net_hash, stream_dict in self.agent.proc_streams.items():
                    for stream in stream_dict.values():
                        if (processor_text_stream is None and stream.props.is_public() == looking_for_public_streams
                                and stream.props.is_text()):
                            processor_text_stream = stream
                            processor_text_net_hash = net_hash
                        if (processor_img_stream is None and stream.props.is_public() == looking_for_public_streams
                                and stream.props.is_img()):
                            processor_img_stream = stream
                            processor_img_net_hash = net_hash

                if processor_text_net_hash is None:
                    raise GenException("Interactive mode requires a processor that generates a text stream")
                if not (processor_img_net_hash is None or (processor_text_net_hash == processor_img_net_hash)):
                    raise GenException("Interactive mode requires the same processor to generate text and img streams")
                processor_net_hash = processor_text_net_hash

                def keyboard_listener(k_queue):
                    while True:
                        webcam_shot = None
                        keyboard_msg = input()  # Get from keyboards
                        if cap is not None:
                            ret, got_shot = cap.read()  # Get from webcam
                            if ret:
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
                            k_queue.put((keyboard_msg, webcam_shot))  # Store in the asynch queue

                        if keyboard_msg.strip() == "exit" or keyboard_msg.strip() == "quit":
                            break

                        if not self.agent.in_world():  # If the world disconnected
                            break

                keyboard_queue = queue.Queue()  # Create a thread-safe queue for communication
                keyboard_listener = threading.Thread(target=keyboard_listener, args=(keyboard_queue,), daemon=True)

            if self.clock.get_cycle() == -1:
                print("Running " + ("agent node" if self.agent else "world node") + " " +
                      f"(public: {self.get_public_peer_id()}, private: {self.get_world_peer_id()})...")

            # Main loop
            must_quit = False
            self.run_start_time = self.clock.get_time()
            while not must_quit:

                # Sending alive message every "K" seconds
                if self.clock.get_time() - self.last_alive_time >= self.send_alive_every:
                    was_alive = self.send_alive()

                    # Checking only at the first run
                    if self.last_alive_time == 0 and was_alive and not self.skip_was_alive_check:
                        print(f"The node is already alive, maybe running in a different machine? "
                              f"(set env variable NODE_IGNORE_ALIVE=1 to ignore this control)")
                        break  # Stopping the running cycle
                    self.last_alive_time = self.clock.get_time()

                # Check inspector
                if self.inspector_activated:
                    if self.__inspector_told_to_pause:
                        print("Paused by the inspector, waiting...")

                        while self.__inspector_told_to_pause:
                            if not self.inspector_activated:  # Disconnected
                                self.__inspector_told_to_pause = False
                                print("Inspector is not active/connected anymore, resuming...")
                                break

                            public_messages = self.conn.get_messages(p2p_name=NodeConn.P2P_PUBLIC)
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
                                        self.__handle_inspector_command(msg.content['cmd'], msg.content['arg'])
                                    else:
                                        self.err("Inspector command was not sent by the expected inspector node ID "
                                                 "or no inspector connected")
                                        self.__purge(msg.sender)
                            time.sleep(0.1)

                # Move to the next cycle
                while not self.clock.next_cycle():
                    time.sleep(0.001)  # Seconds (lowest possible granularity level)

                self.out(f">>> Starting clock cycle {self.clock.get_cycle()} <<<")

                # Handle new connections or lost connections
                self.__handle_network_connections()

                # Handle (read, execute) received network data/commands
                self.__handle_network_messages(interact_mode_opts=interact_mode_opts)

                # Stream live data (generated and environmental)
                if len(self.hosted.all_agents) > 0:
                    if self.node_type is Node.WORLD:
                        if self.first is True:
                            self.first = False
                            for net_hash, stream_dict in self.hosted.known_streams.items():
                                for stream_obj in stream_dict.values():
                                    if isinstance(stream_obj, BufferedDataStream):
                                        stream_obj.restart()
                self.hosted.send_stream_samples()

                # Trigger HSM of the agent
                if self.node_type is Node.AGENT:
                    if interact_mode_opts is not None:
                        try:

                            # Waiting until we meet a state named "ready"
                            if not ready_to_interact:
                                behav = self.agent.behav_lone_wolf \
                                    if "lone_wolf_peer_id" in interact_mode_opts else self.agent.behav
                                if behav.state == "ready":
                                    ready_to_interact = True
                                    keyboard_listener.start()
                                    cap = cv2.VideoCapture(0) if processor_img_stream is not None else None
                                    print(f"\n*** Entering interactive text mode ***\n\n👉 ", end="")

                                    original_stdout = sys.stdout  # Valid screen-related stream
                                    sys.stdout = open('interact_stdout.txt',
                                                      'w')  # Open(os.devnull, 'w')  # null stream
                                    interact_mode_opts["stdout"] = [original_stdout, sys.stdout]

                            # Getting message from keyboard
                            msg, image_pil = keyboard_queue.get_nowait()

                            # Quit?
                            msg = msg.strip()
                            if msg == "exit" or msg == "quit":
                                must_quit = True
                                interact_mode_opts["stdout"][1].close()
                                if cap is not None:
                                    cap.release()
                            else:

                                # Asking the to generate (the request will be immediately sent)
                                if "lone_wolf_peer_id" in interact_mode_opts:
                                    behav = self.agent.behav_lone_wolf
                                    other_behav = self.agent.behav

                                    other_behav.enable(False)
                                    behav.enable(True)
                                    self.agent.ask_gen(agent=interact_mode_opts["lone_wolf_peer_id"],
                                                       u_hashes=[processor_net_hash],
                                                       samples=1)
                                    behav.enable(False)
                                else:
                                    self.agent.behav.request_action(action_name="ask_gen",
                                                                    args={},
                                                                    signature=self.get_world_peer_id(),
                                                                    timestamp=self.clock.get_time(),
                                                                    uuid=None)
                                    behav = self.agent.behav
                                    other_behav = self.agent.behav_lone_wolf
                                    self.agent.behave()

                                # Loading the message and image to the processor's output streams
                                # they will be sent at the next clock cycle
                                other_behav.enable(False)
                                behav.enable(True)
                                if processor_img_stream is not None:
                                    [msg, image_pil], _ = self.agent.generate(input_net_hashes=None,
                                                                              inputs=[msg, image_pil])
                                    processor_text_stream.set(msg)
                                    processor_img_stream.set(image_pil)
                                else:
                                    [msg], _ = self.agent.generate(input_net_hashes=None,
                                                                   inputs=[msg])
                                    processor_text_stream.set(msg)
                                behav.enable(False)
                        except queue.Empty:
                            self.agent.behave()  # If nothing has been typed (+ enter)
                    else:

                        # Ordinary behaviour
                        self.agent.behave()

                # Send dynamic profile every "N" seconds
                if (self.clock.get_time() - last_dynamic_profile_time >= self.send_dynamic_profile_every
                        and self.profile.connections_changed()):
                    try:
                        last_dynamic_profile_time = self.clock.get_time()
                        self.profile.unmark_change_in_connections()
                        self.send_badges()  # Sending and clearing badges
                        self.send_dynamic_profile()  # Sending
                    except Exception as e:
                        self.err(f"Error while sending the update dynamic profile (or badges) to the server "
                                 f"(trying to go ahead...) [{e}]")

                # Getting a new token every "N" seconds
                if self.clock.get_time() - last_get_token_time >= self.get_new_token_every:
                    self.get_node_token(peer_ids=[self.get_public_peer_id(), self.get_world_peer_id()])
                    last_get_token_time = self.clock.get_time()

                # Check for address changes every "N" seconds
                if self.clock.get_time() - last_address_check_time >= self.address_check_every:
                    try:
                        self.out("Performing periodic check for address changes...")
                        last_address_check_time = self.clock.get_time()
                        current_public_addrs = self.conn.p2p_public.get_node_addresses()
                        current_private_addrs = self.conn.p2p_world.get_node_addresses()
                        profile_public_addrs = self.profile.get_dynamic_profile().get('peer_addresses', [])
                        profile_private_addrs = self.profile.get_dynamic_profile().get('private_peer_addresses', [])

                        # TODO: if public addresses changed... (if this makes any sense)
                        if set(current_public_addrs) != set(profile_public_addrs):
                            self.out(f"Address change detected for the public instance! "
                                     f"New addresses: {current_public_addrs}")

                            # Update profile in-place
                            # address_list = self.profile.get_dynamic_profile()['peer_addresses']
                            # address_list.clear()
                            # address_list.extend(current_public_addrs)
                            # self.profile.mark_change_in_connections()

                        # If private addresses changed, update the profile and notify the world
                        elif set(current_private_addrs) != set(profile_private_addrs):
                            self.out(f"Address change detected for the private instance! "
                                     f"New addresses: {current_public_addrs}")

                            # Update profile in-place
                            address_list = self.profile.get_dynamic_profile()['private_peer_addresses']
                            address_list.clear()
                            address_list.extend(current_private_addrs)
                            # self.profile.mark_change_in_connections()

                            world_peer_id = (
                                self.profile.get_dynamic_profile().get('connections', {}).get('world_peer_id'))
                            if self.node_type is Node.AGENT and world_peer_id:
                                self.out("Notifying world of address change...")
                                self.conn.send(
                                    world_peer_id, content_type=Msg.ADDRESS_UPDATE, channel_trail=None,
                                    content={'addresses': self.profile.get_dynamic_profile()['private_peer_addresses']}
                                )
                        else:
                            self.out("No address changes detected.")
                    except Exception as e:
                        self.err(f"Failed to check for address updates: {e}")

                # Refresh relay reservation if nearing expiration
                if self.relay_reservation_expiry is not None:
                    time_to_expiry = self.relay_reservation_expiry - datetime.now(timezone.utc)
                    if time_to_expiry < timedelta(minutes=15):
                        self.out("Relay reservation nearing expiration. Attempting to renew...")
                        try:
                            world_private_peer_id = self.profile.get_dynamic_profile()['connections']['world_peer_id']
                            new_expiry_utc = self.conn.p2p_world.reserve_on_relay(world_private_peer_id)
                            self.relay_reservation_expiry = datetime.fromisoformat(
                                new_expiry_utc.replace('Z', '+00:00'))
                            self.out(f"Relay reservation renewed. New expiration: "
                                     f"{self.relay_reservation_expiry.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                        except Exception as e:
                            self.err(f"Failed to renew relay reservation: {e}. Node may become unreachable.")
                            self.relay_reservation_expiry = None  # Stop trying if it fails

                # Taking to the inspector
                if self.inspector_activated:
                    self.__send_to_inspector()

                # Stop conditions
                if cycles is not None and ((self.clock.get_cycle() + 1) >= cycles):
                    break
                if max_time is not None and (self.clock.get_time() - self.run_start_time) >= max_time:
                    break

        except KeyboardInterrupt:
            if self.cursor_hidden:
                sys.stdout.write("\033[?25h")  # Re-enabling cursor
            if cycles == 1:
                raise KeyboardInterrupt  # Node synch will catch this
            else:
                print("\nDetected Ctrl+C! Exiting gracefully...")

        except Exception as e:
            if self.cursor_hidden:
                sys.stdout.write("\033[?25h")  # Re-enabling cursor
            print(f"An error occurred: {e}")
            traceback.print_exc()

    def __handle_network_connections(self):
        """Manages new and lost network connections."""
        
        # Getting fresh lists of existing world agents and world masters (from the rendezvous)
        if self.node_type is Node.AGENT:
            self.out("Updating list of world agents and world masters by using data from the rendezvous")
            self.conn.set_world_agents_and_world_masters_lists_from_rendezvous()

        # Updating connection pools, getting back the lists (well, dictionaries) of new agents and lost agents
        new_peer_ids_by_pool, removed_peer_ids_by_pool = self.conn.update()
        if len(new_peer_ids_by_pool) > 0 or len(removed_peer_ids_by_pool) > 0:
            self.out("Current status of the pools, right after the update:\n" + str(self.conn))

        # Checking if some peers were removed
        an_agent_left_the_world = False
        removed_peers = False
        for pool_name, removed_peer_ids in removed_peer_ids_by_pool.items():
            for peer_id in removed_peer_ids:
                removed_peers = True
                self.out("Removing a not-connected-anymore peer, "
                         "pool_name: " + pool_name + ", peer_id: " + peer_id + "...")
                self.__purge(peer_id)

                # Checking if we removed an agent from this world
                if self.node_type is Node.WORLD and pool_name in self.conn.WORLD:
                    an_agent_left_the_world = True

                # Check if the world disconnected: in that case, disconnect all the other agents in the world and leave
                if self.node_type is Node.AGENT and pool_name in self.conn.WORLD_NODE:
                    self.leave_world()

                # Checking if the inspector disconnected
                if peer_id == self.inspector_peer_id:
                    self.inspector_activated = False
                    self.inspector_peer_id = None
                    self.__inspector_cache = {"behav": None, "known_streams_count": 0, "all_agents_count": 0}
                    print("Inspector disconnected")

        # Handling newly connected peers
        an_agent_joined_the_world = False
        added_peers = False
        for pool_name, new_peer_ids in new_peer_ids_by_pool.items():
            for peer_id in new_peer_ids:
                added_peers = True
                self.out("Processing a newly connected peers, "
                         "pool_name: " + pool_name + ", peer_id: " + peer_id + "...")

                # If this is a world node, it is time to tell the world object that a new agent is there
                if self.node_type is Node.WORLD and pool_name in self.conn.WORLD:
                    self.out("Not considering interviewing since this is a world and the considered peer is in the"
                             " world pools")

                    if peer_id in self.agents_to_interview:

                        # Getting the new agent profile
                        profile = self.agents_to_interview[peer_id][1]  # [time, profile]

                        # Adding the new agent to the world object
                        if not self.world.add_agent(peer_id=peer_id, profile=profile):
                            self.__purge(peer_id)
                            continue

                        # Clearing the profile from the interviews
                        del self.agents_to_interview[peer_id]  # Removing from the queue (private peer id)
                        an_agent_joined_the_world = True

                        # Replacing multi-address with what comes from the profile (there are more addresses there!)
                        self.conn.set_addresses_in_peer_info(peer_id,
                                                             profile.get_dynamic_profile()['private_peer_addresses'])
                    else:

                        # This agent tried to connect to a world "directly", without passing through the
                        # public handshake
                        self.__purge(peer_id)
                        continue

                    continue  # Nothing else to do

                # Both if this is an agent or a world, checks if the newly connected agent can be added or not to the
                # queue of agents to interview
                if pool_name not in self.conn.OUTGOING:

                    # Trying to add to the queue
                    enqueued_for_interview = self.__interview_enqueue(peer_id)

                    # If the agent is rejected at this stage, we disconnect from its peer
                    if not enqueued_for_interview:
                        self.out(f"Not enqueued for interview, removing peer (disconnecting {peer_id})")
                        self.__purge(peer_id)
                    else:
                        self.out("Enqueued for interview")

        # Updating list of world agents & friends, if needed
        # (it happens only if the node hosts a world, otherwise 'an_agent_joined_the_world' and
        # 'an_agent_left_the_world' are certainly False)
        world_agents_peer_infos = None
        world_masters_peer_infos = None
        if self.node_type is Node.WORLD:
            enter_left = an_agent_joined_the_world or an_agent_left_the_world
            timeout = (self.clock.get_time() - self.last_rendezvous_time) >= self.publish_rendezvous_every

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
                if not self.conn.publish(self.conn.p2p_world.peer_id, f"{self.conn.p2p_world.peer_id}::ps:rv",
                                         content_type=Msg.WORLD_AGENTS_LIST,
                                         content={"peers": world_all_peer_infos,
                                                  "update_count": self.clock.get_cycle()}):
                    self.err("Failed to publish the updated list of (all) world agents (ignoring)")
                else:
                    self.last_rendezvous_time = self.clock.get_time()
                    self.out(f"Rendezvous messages just published "
                             f"(tag: {self.clock.get_cycle()}, peers: {len(world_all_peer_infos)})")

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

    def __handle_network_messages(self, interact_mode_opts=None):
        """Handles and processes all incoming network messages.

        Args:
            interact_mode_opts: A dictionary of options for interactive mode.
        """
        # Fetching all messages,
        public_messages = self.conn.get_messages(p2p_name=NodeConn.P2P_PUBLIC)
        world_messages = self.conn.get_messages(p2p_name=NodeConn.P2P_WORLD)

        self.out("Got " + str(len(public_messages)) + " messages from the public net")
        self.out("Got " + str(len(world_messages)) + " messages from the world/private net")

        # Process all messages
        all_messages = public_messages + world_messages
        if len(all_messages) > 0:
            self.out("Processing all messages...")

        for i, msg in enumerate(all_messages):
            if i < len(public_messages):
                self.out("Processing public message " + str(i + 1) + "/"
                         + str(len(public_messages)) + ": " + str(msg))
            else:
                self.out("Processing world/private message " + str(i - len(public_messages) + 1)
                         + "/" + str(len(world_messages)) + ": " + str(msg))

            # Checking
            if not isinstance(msg, Msg):
                self.err("Expected message of type Msg, got {} (skipping)".format(type(msg)))
                continue

            # Unpacking piggyback
            sender_node_id, sender_inspector_mode_on = (msg.piggyback[0:-1],
                                                        msg.piggyback[-1] == "1")

            # Is message from inspector?
            sender_is_inspector = (sender_node_id == self.profile.get_static_profile()['inspector_node_id'] and
                                   sender_inspector_mode_on)

            # (A) received a profile
            if msg.content_type == Msg.PROFILE:
                self.out("Received a profile...")

                # Checking the received profile
                # (recall that a profile sent through the world connection to the world node will be considered
                # not acceptable)
                profile = NodeProfile.from_dict(msg.content)
                is_an_already_known_agent = msg.sender in self.hosted.all_agents

                if is_an_already_known_agent:
                    self.out("Editing information of an already added agent " + msg.sender)

                    if not self.hosted.add_agent(peer_id=msg.sender, profile=profile):
                        self.__purge(msg.sender)
                else:
                    is_expected_and_acceptable_profile = self.__interview_check_profile(peer_id=msg.sender,
                                                                                        node_id=sender_node_id,
                                                                                        profile=profile)

                    if not is_expected_and_acceptable_profile:
                        self.err("Unexpected or unacceptable profile, removing (disconnecting) " + msg.sender)
                        self.__purge(msg.sender)
                    else:

                        # If the node hosts a world and gets an expected and acceptable profile from the public network,
                        # assigns a role and sends the world profile (which includes private peer ID) and role to the
                        # requester
                        if (self.node_type is Node.WORLD and self.conn.is_public(peer_id=msg.sender) and
                                not sender_is_inspector):
                            self.out("Sending world approval message, profile, and assigned role to " + msg.sender +
                                     " (and switching peer ID in the interview queue)...")
                            is_world_master = (self.world_masters_node_ids is not None and
                                               sender_node_id in self.world_masters_node_ids)

                            # Assigning a role
                            role_str = self.world.assign_role(profile=profile, is_world_master=is_world_master)
                            if role_str is None:
                                self.err("Unable to determine what role to assign, removing (disconnecting) "
                                         + msg.sender)
                                self.__purge(msg.sender)
                            else:
                                role = self.world.ROLE_STR_TO_BITS[role_str]  # The role is a bit-wise-interpretable int
                                role = role | (Agent.ROLE_WORLD_MASTER if is_world_master else Agent.ROLE_WORLD_AGENT)

                                # Clearing temporary options (if any)
                                dynamic_profile = profile.get_dynamic_profile()
                                keys_to_delete = [key for key in dynamic_profile if key.startswith('tmp_')]
                                for key in keys_to_delete:
                                    del dynamic_profile[key]

                                if not self.conn.send(msg.sender, channel_trail=None,
                                                      content={
                                                          'world_profile': self.profile.get_all_profile(),
                                                          'rendezvous_tag': self.clock.get_cycle(),
                                                          'your_role': role,
                                                          'agent_actions': self.world.agent_actions
                                                      },
                                                      content_type=Msg.WORLD_APPROVAL):
                                    self.err("Failed to send world approval, removing (disconnecting) " + msg.sender)
                                    self.__purge(msg.sender)
                                else:
                                    private_peer_id = profile.get_dynamic_profile()['private_peer_id']
                                    private_addr = profile.get_dynamic_profile()['private_peer_addresses']
                                    if is_world_master:
                                        role = role | Agent.ROLE_WORLD_MASTER
                                        self.conn.add_to_world_masters_list(private_peer_id, private_addr, role)
                                    else:
                                        role = role | Agent.ROLE_WORLD_AGENT
                                        self.conn.add_to_world_agents_list(private_peer_id, private_addr, role)

                                    # Removing from the queue of public interviews
                                    # and adding to the private ones (refreshing timer)
                                    del self.agents_to_interview[msg.sender]  # Removing from public queue
                                    self.agents_to_interview[private_peer_id] = [self.clock.get_time(), profile]  # Add

                        # If the node is an agent, it is time to tell the agent object that a new agent is now known,
                        # and send our profile to the agent that asked for out contact
                        elif self.node_type is Node.AGENT or sender_is_inspector:
                            self.out("Sending agent approval message and profile...")

                            if not self.conn.send(msg.sender, channel_trail=None,
                                                  content={
                                                      'my_profile': self.profile.get_all_profile()
                                                  },
                                                  content_type=Msg.AGENT_APPROVAL):
                                self.err("Failed to send agent approval, removing (disconnecting) " + msg.sender)
                                self.__purge(msg.sender)
                            else:
                                self.out("Adding known agent and removing it from the interview queue " + msg.sender)
                                if not self.hosted.add_agent(peer_id=msg.sender, profile=profile):  # keep "hosted" here
                                    self.__purge(msg.sender)
                                else:

                                    # Removing from the queues
                                    del self.agents_to_interview[msg.sender]  # Removing from queue

            # (B) received a world-join-approval
            elif msg.content_type == Msg.WORLD_APPROVAL:
                self.out("Received a world-join-approval message...")

                # Checking if it is the world we asked for
                # moreover, it must be on the public network, and this must not be a world-node (of course)
                # and you must not be already in another world
                if (not self.conn.is_public(peer_id=msg.sender) or self.node_type is Node.WORLD
                        or msg.sender not in self.agents_expected_to_send_ack or
                        self.profile.get_dynamic_profile()['connections']['world_peer_id'] is not None):
                    self.err("Unexpected world approval, removing (disconnecting) " + msg.sender)
                    self.__purge(msg.sender)
                else:
                    if msg.sender != self.joining_world_info["world_public_peer_id"]:
                        self.err(f"Unexpected world approval: asked to join "
                                 f"{self.joining_world_info['world_public_peer_id']} got approval from {msg.sender} "
                                 f"(disconnecting)")
                        self.__purge(msg.sender)
                    else:

                        # Getting world profile (includes private addresses) and connecting to the world (privately)
                        self.__join_world(profile=NodeProfile.from_dict(msg.content['world_profile']),
                                          role=msg.content['your_role'],
                                          agent_actions=msg.content['agent_actions'],
                                          rendezvous_tag=msg.content['rendezvous_tag'])

            # (C) received an agent-connect-approval
            elif msg.content_type == Msg.AGENT_APPROVAL:
                self.out("Received an agent-connect-approval message...")

                # Checking if it is the agent we asked for
                if msg.sender not in self.agents_expected_to_send_ack:
                    self.err("Unexpected agent-connect approval, removing (disconnecting) " + msg.sender)
                    self.__purge(msg.sender)
                else:

                    # Adding the agent
                    self.__join_agent(profile=NodeProfile.from_dict(msg.content['my_profile']),
                                      peer_id=msg.sender)

            # (D) requested for a profile
            elif msg.content_type == Msg.PROFILE_REQUEST:
                self.out("Received a profile request...")

                # If this is a world-node, it expects profile requests only on the public network
                # if this is not a world or not, we only send profile to agents who are involved in the handshake
                if ((self.node_type is Node.WORLD and not self.conn.is_public(peer_id=msg.sender)) or
                        (msg.sender not in self.agents_expected_to_send_ack)):
                    self.err("Unexpected profile request, removing (disconnecting) " + msg.sender)
                    self.__purge(msg.sender)
                else:

                    # If a preference was defined, we temporarily add it to the profile
                    if (self.joining_world_info is not None and
                            msg.sender == self.joining_world_info["world_public_peer_id"] and
                            self.joining_world_info["options"] is not None and
                            len(self.joining_world_info["options"]) > 0):
                        my_profile = copy.deepcopy(self.profile)
                        for k, v in self.joining_world_info["options"].items():
                            my_profile.get_dynamic_profile()['tmp_' + str(k)] = v
                        my_profile = my_profile.get_all_profile()
                    else:
                        my_profile = self.profile.get_all_profile()

                    # Sending the profile
                    self.out("Sending profile")
                    if not self.conn.send(msg.sender, channel_trail=None,
                                          content=my_profile,
                                          content_type=Msg.PROFILE):
                        self.err("Failed to send profile, removing (disconnecting) " + msg.sender)
                        self.__purge(msg.sender)

            # (E) the world node received an ADDRESS_UPDATE from an agent
            elif msg.content_type == Msg.ADDRESS_UPDATE:
                self.out("Received an address update from " + msg.sender)

                if self.node_type is Node.WORLD and msg.sender in self.world.all_agents:
                    all_addresses = msg.content.get('addresses')
                    if all_addresses and isinstance(all_addresses, list):

                        # Update the address both in the connection and in the profile
                        self.conn.set_addresses_in_peer_info(msg.sender, all_addresses)
                        self.world.set_addresses_in_profile(msg.sender, all_addresses)
                        self.out(f"Waiting rendezvous publish after address update from {msg.sender}")

            # (F) got stream data
            elif msg.content_type == Msg.STREAM_SAMPLE:
                self.out("Received a stream sample...")

                if self.node_type is Node.AGENT:  # Handling the received samples
                    self.agent.get_stream_sample(net_hash=msg.channel, sample_dict=msg.content)

                    # Printing messages to screen, if needed (useful when chatting with lone wolves)
                    if interact_mode_opts is not None and "stdout" in interact_mode_opts:
                        net_hash = DataProps.normalize_net_hash(msg.channel)
                        if net_hash in self.agent.known_streams:
                            stream_dict = self.agent.known_streams[net_hash]
                            sys.stdout = interact_mode_opts["stdout"][0]  # Output on
                            for name, stream_obj in stream_dict.items():
                                if stream_obj.props.is_text():
                                    msg = stream_obj.get(requested_by="print")  # Getting message
                                    print(f"\n『 {msg} 』")  # Printing to screen
                                if stream_obj.props.is_img():
                                    img = stream_obj.get(requested_by="print")  # Getting image
                                    filename = "wolf_img.png"
                                    img.save(filename)
                                    msg = f"(saved image to {filename})"
                                    print(f"\n『 {msg} 』")  # Printing to screen
                            print("\n👉 ", end="")
                            sys.stdout = interact_mode_opts["stdout"][1]  # Output off

                elif self.node_type is Node.WORLD:
                    self.err("Unexpected stream samples received by this world node, sent by: " + msg.sender)
                    self.__purge(msg.sender)

            # (G) got action request
            elif msg.content_type == Msg.ACTION_REQUEST:
                self.out("Received an action request...")

                if self.node_type is Node.AGENT:
                    if msg.sender not in self.agent.all_agents:
                        self.err("Unexpected action request received by a unknown node: " + msg.sender)
                    else:
                        behav = self.agent.behav_lone_wolf \
                            if msg.sender in self.agent.public_agents else self.agent.behav
                        behav.request_action(action_name=msg.content['action_name'],
                                             args=msg.content['args'],
                                             signature=msg.sender,
                                             timestamp=self.clock.get_time(),
                                             uuid=msg.content['uuid'])

                elif self.node_type is Node.WORLD:
                    self.err("Unexpected action request received by this world node, sent by: " + msg.sender)
                    self.__purge(msg.sender)

            # (H) got role suggestion
            elif msg.content_type == Msg.ROLE_SUGGESTION:
                self.out("Received a role suggestion/new role...")

                if self.node_type is Node.AGENT:
                    if msg.sender == self.conn.get_world_peer_id():
                        new_role_indication = msg.content
                        if new_role_indication['peer_id'] == self.get_world_peer_id():
                            self.agent.accept_new_role(new_role_indication['role'])

                            self.agent.behav.update_wildcard("<agent>", f"{self.get_world_peer_id()}")
                            self.agent.behav.update_wildcard("<world>", f"{msg.sender}")

                elif self.node_type is Node.WORLD:
                    if msg.sender in self.world.world_masters:
                        for role_suggestion in msg.content:
                            self.world.set_role(peer_id=role_suggestion['peer_id'], role=role_suggestion['role'])

            # (I) got request to alter the HSM
            elif msg.content_type == Msg.HSM:
                self.out("Received a request to alter the HSM...")

                if self.node_type is Node.AGENT:
                    if msg.sender in self.agent.world_masters:  # This must be coherent with what we do in set_role
                        ret = getattr(self.agent.behav, msg.content['method'])(*msg.content['args'])
                        if not ret:
                            self.err(f"Cannot run HSM action named {msg.content['method']} with args "
                                     f"{msg.content['args']}")
                    else:
                        self.err("Only world-master can alter HSMs of other agents: " + msg.sender)  # No need to purge

                elif self.node_type is Node.WORLD:
                    self.err("Unexpected request to alter the HSM received by this world node, sent by: " + msg.sender)
                    self.__purge(msg.sender)

            # (J) misc
            elif msg.content_type == Msg.MISC:
                self.out("Received a misc message...")
                self.out(msg.content)

            # (K) got a request to re-download the CV from the root server
            elif msg.content_type == Msg.GET_CV_FROM_ROOT:
                self.out("Received a notification to re-download the CV...")

                # Downloading CV
                self.profile.update_cv(self.get_cv())

                # Re-downloading token (it will include the new CV hash)
                self.get_node_token(peer_ids=[self.get_public_peer_id(), self.get_world_peer_id()])

            # (L) got one or more badge suggestions
            elif msg.content_type == Msg.BADGE_SUGGESTIONS:
                self.out("Received badge suggestions...")

                if self.node_type is Node.WORLD:
                    for badge_dict in msg.content:

                        # Right now, we accept all the suggestions
                        self.world.add_badge(**badge_dict)  # Adding to the list of badges
                elif self.node_type is Node.AGENT:
                    self.err("Receiving badge suggestions is not expected for an agent node")

            # (M) got a special connection/presence message for an inspector
            elif msg.content_type == Msg.INSPECT_ON:
                self.out("Received an inspector-activation message...")

                if sender_is_inspector:
                    self.inspector_activated = True
                    self.inspector_peer_id = msg.sender
                    print("Inspector activated")
                else:
                    self.err("Inspector-activation message was not sent by the expected inspector node ID")
                    self.__purge(msg.sender)

            # (N) got a command from an inspector
            elif msg.content_type == Msg.INSPECT_CMD:
                self.out("Received a command from the inspector...")

                if sender_is_inspector and self.inspector_activated:
                    self.__handle_inspector_command(msg.content['cmd'], msg.content['arg'])
                else:
                    self.err("Inspector command was not sent by the expected inspector node ID "
                             "or the inspector was not yet activated (Msg.INSPECT_ON not received yet)")
                    self.__purge(msg.sender)

        self.__interview_clean()
        self.__connected_without_ack_clean()

    def __join_world(self, profile: NodeProfile, role: int,
                     agent_actions: str | None, rendezvous_tag: int):
        """Performs the actual operation of joining a world after receiving confirmation.

        Args:
            profile: The profile of the world to join.
            role: The role assigned to the agent in the world (int).
            agent_actions: A string of code defining the agent's actions.
            rendezvous_tag: The rendezvous tag from the world's profile.

        Returns:
            True if the join operation is successful, otherwise False.
        """
        addresses = profile.get_dynamic_profile()['private_peer_addresses']
        world_public_peer_id = profile.get_dynamic_profile()['peer_id']
        self.out(f"Actually joining world, role will be {role}")

        # Connecting to the world (private)
        # notice that we also communicate the world node private peer ID to the connection manager,
        # to avoid filtering it out when updating pools
        peer_id = self.ask_to_get_in_touch(addresses=addresses, public=False,
                                           before_updating_pools_fcn=self.conn.set_world)

        if peer_id is not None:

            # Relay reservation logic for non-public peers
            if not self.conn.p2p_world.is_public and self.conn.p2p_world.relay_is_enabled:
                self.out("Node is not publicly reachable. Attempting to reserve a slot on the world's private network.")
                try:
                    expiry_utc = self.conn.p2p_world.reserve_on_relay(peer_id)
                    self.relay_reservation_expiry = (
                        datetime.fromisoformat(expiry_utc.replace('Z', '+00:00')))
                    self.out(f"Reserved relay slot. Expires at "
                             f"{self.relay_reservation_expiry.strftime('%Y-%m-%d %H:%M:%S')} UTC")

                    self.out("Fetching updated address list from transport layer...")
                    complete_private_addrs = self.conn.p2p_world.get_node_addresses()

                    # Update the profile with this definitive list (IN-PLACE)
                    address_list = self.profile.get_dynamic_profile()['private_peer_addresses']
                    address_list.clear()
                    address_list.extend(complete_private_addrs)

                    self.out("Notifying world of the complete updated address list...")
                    self.conn.send(peer_id, channel_trail=None,
                                   content_type=Msg.ADDRESS_UPDATE,
                                   content={'addresses': complete_private_addrs})
                except Exception as e:
                    self.err(f"An error occurred during relay reservation: {e}.")

            # Subscribing to the world rendezvous topic, from which we will get fresh information
            # about the world agents and masters
            self.out("Subscribing to the world-members topic...")
            if not self.conn.subscribe(peer_id, channel=f"{peer_id}::ps:rv"):  # Special rendezvous (ps:rv)
                self.leave(peer_id)  # If subscribing fails, we quit everything (safer)
                return False

            # Killing the public connection to the world node
            self.out("Disconnecting from the public world network (since we joined the private one)")
            self.__purge(world_public_peer_id)

            # Removing the private world peer id from the list of connected-but-not-managed peer
            del self.agents_expected_to_send_ack[peer_id]

            # Subscribing to all the other world topics, from which we will get fresh information
            # about the streams
            self.out("Subscribing to the world-streams topics...")
            dynamic_profile = profile.get_dynamic_profile()
            list_of_props = []
            list_of_props += dynamic_profile['streams'] if dynamic_profile['streams'] is not None else []
            list_of_props += dynamic_profile['proc_outputs'] if dynamic_profile['proc_outputs'] is not None else []

            if not self.agent.add_compatible_streams(peer_id, list_of_props, buffered=False, public=False):
                self.leave(peer_id)
                return False

            # Setting actions
            if agent_actions is not None and len(agent_actions) > 0:

                # Checking code
                if not Node.__analyze_code(agent_actions):
                    self.err("Invalid agent actions code (syntax errors or unsafe code) was provided by the world, "
                             "blocking the join operation")
                    return False

                # Creating a new agent with the received actions
                mod = types.ModuleType("dynamic_module")
                exec(agent_actions, mod.__dict__)
                sys.modules["dynamic_module"] = mod
                new_agent = mod.WAgent(proc=None)

                # Cloning attributes of the existing agent
                for key, value in self.agent.__dict__.items():
                    if hasattr(new_agent, key):  # This will skip ROLE_BITS_TO_STR, CUSTOM_ROLES, etc...
                        setattr(new_agent, key, value)

                # Telling the FSM that actions are related to this new agent
                new_agent.behav.set_actionable(new_agent)
                new_agent.behav_lone_wolf.set_actionable(new_agent)

                # Setting up roles
                roles = profile.get_dynamic_profile()['world_roles_fsm'].keys()
                new_agent.CUSTOM_ROLES = roles
                new_agent.augment_roles()

                # Setting up world stats
                world_stats_dynamic_by_peer = profile.get_dynamic_profile()['world_stats_dynamic']
                if world_stats_dynamic_by_peer:
                    new_agent.STATS_DYNAMIC_SENT_BY_PEER.clear()
                    new_agent.STATS_DYNAMIC_SENT_BY_PEER.update(world_stats_dynamic_by_peer)  # in-place
                    new_agent.clear_stats()

                # Updating node-level references
                old_agent = self.agent
                self.agent = new_agent
                self.hosted = new_agent
            else:
                old_agent = self.agent

            # Saving the world profile
            self.agent.world_profile = profile

            # Setting the assigned role and default behavior (do it after having recreated the new agent object)
            self.agent.accept_new_role(role)  # Do this after having done 'self.agent.world_profile = profile'

            # Updating wildcards
            self.agent.behav.update_wildcard("<agent>", f"{self.get_world_peer_id()}")
            self.agent.behav.update_wildcard("<world>", f"{peer_id}")
            self.agent.behav.add_wildcards(old_agent.behav_wildcards)

            # Telling the connection manager the info needed to discriminate peers (getting them from the world profile)
            # notice that the world node private ID was already told to the connection manager (see a few lines above)
            self.out(f"Rendezvous tag received with profile: {rendezvous_tag} "
                     f"(in conn pool: {self.conn.rendezvous_tag})")
            if self.conn.rendezvous_tag < rendezvous_tag:
                self.conn.rendezvous_tag = rendezvous_tag
                num_world_masters = len(dynamic_profile['world_summary']['world_masters']) \
                    if dynamic_profile['world_summary']['world_masters'] is not None else 'none'
                num_world_agents = len(dynamic_profile['world_summary']['world_agents']) \
                    if dynamic_profile['world_summary']['world_agents'] is not None else 'none'
                self.out(f"Rendezvous from profile (tag: {rendezvous_tag}), world masters: {num_world_masters}")
                self.out(f"Rendezvous from profile (tag: {rendezvous_tag}), world agents: {num_world_agents}")
                self.conn.set_world_masters_list(dynamic_profile['world_summary']['world_masters'])
                self.conn.set_world_agents_list(dynamic_profile['world_summary']['world_agents'])

            # Updating our profile to set the world we are in
            self.profile.get_dynamic_profile()['connections']['world_peer_id'] = peer_id
            self.profile.mark_change_in_connections()

            print("Handshake completed, world joined!")
            return True
        else:
            return False

    def __join_agent(self, profile: NodeProfile, peer_id: str):
        """Adds a new known agent after receiving an approval message.

        Args:
            profile: The profile of the agent to join.
            peer_id: The peer ID of the agent.

        Returns:
            True if the agent is successfully added, otherwise False.
        """
        self.out("Adding known agent " + peer_id)
        if not self.agent.add_agent(peer_id=peer_id, profile=profile):
            self.__purge(peer_id)
            return False

        del self.agents_expected_to_send_ack[peer_id]
        return True

    def __interview_enqueue(self, peer_id: str):
        """Adds a newly connected peer to the queue of agents to be interviewed.

        Args:
            peer_id: The peer ID of the agent to interview.

        Returns:
            True if the agent is successfully enqueued, otherwise False.
        """

        # If the peer_id is not in the same world were we are, we early stop the interview process
        if (not self.conn.is_public(peer_id) and peer_id not in self.conn.world_agents_list and
                peer_id not in self.conn.world_masters_list and peer_id != self.conn.world_node_peer_id):
            self.out(f"Interview failed: "
                     f"peer ID {peer_id} is not in the world agents/masters list, and it is not the world node")
            return False

        # Ask for the profile
        ret = self.conn.send(peer_id, channel_trail=None,
                             content_type=Msg.PROFILE_REQUEST, content=None)
        if not ret:
            self.out(f"Interview failed: "
                     f"unable to send a profile request to peer ID {peer_id}")
            return False

        # Put the agent in the list of agents to interview
        self.agents_to_interview[peer_id] = [self.clock.get_time(), None]  # Peer ID -> [time, profile]; no profile yet
        return True

    def __interview_check_profile(self, peer_id: str, node_id: str, profile: NodeProfile):
        """Checks if a received profile is acceptable and valid.

        Args:
            peer_id: The peer ID of the node that sent the profile.
            node_id: The node ID of the node that sent the profile.
            profile: The NodeProfile object to be checked.

        Returns:
            True if the profile is acceptable, otherwise False.
        """

        # If the node ID was not on the list of allowed ones (if the list exists), then stop it
        # notice that we do not get the node ID from the profile, but from outside (it comes from the token, so safe)
        if ((self.allowed_node_ids is not None and node_id not in self.allowed_node_ids) or
                (peer_id not in self.agents_to_interview)):
            self.out(f"Profile of f{peer_id} not in the list of agents to interview or its node ID is not allowed")
            return False
        else:

            # Getting the parts of profile needed
            eval_static_profile = profile.get_static_profile()
            eval_dynamic_profile = profile.get_dynamic_profile()
            my_dynamic_profile = self.profile.get_dynamic_profile()

            # Checking if CV was altered
            cv_hash = self.conn.get_cv_hash_from_last_token(peer_id)
            sanity_ok, pairs_of_hashes = profile.verify_cv_hash(cv_hash)
            if not sanity_ok:
                self.out(f"The CV in the profile of f{peer_id} failed the sanity check {pairs_of_hashes},"
                         f" {profile.get_cv()}")
                return False

            # Determining type of agent, checking the connection pools
            role = self.conn.get_role(peer_id)

            if role & 1 == 0:

                # Ensuring that the interviewed agent is out of every world
                # (if it were in the same world in which we are, it would connect in a private manner) and
                # possibly fulfilling the optional constraint of accepting only certified agent,
                # then asking the hosted entity for additional custom evaluation
                if (eval_dynamic_profile['connections']['world_peer_id'] is None and
                        (not self.only_certified_agents or eval_static_profile['certified'] is True)):
                    return self.hosted.evaluate_profile(role, profile)
                else:
                    self.out(f"Peer f{peer_id} is already living in a world, of it is not certified "
                             f"and maybe I expect certified only")
                    return False
            else:

                if self.node_type is Node.AGENT:

                    # Ensuring that the interviewed agent is in the same world where we are and
                    # possibly fulfilling the optional constraint of accepting only certified agent
                    if (eval_dynamic_profile['connections']['world_peer_id'] is not None and
                            eval_dynamic_profile['connections']['world_peer_id'] ==
                            my_dynamic_profile['connections']['world_peer_id'] and
                            (not self.only_certified_agents or 'certified' in eval_static_profile and
                             eval_static_profile['certified'] is True)):
                        return self.hosted.evaluate_profile(role, profile)
                    else:
                        self.out(f"Peer f{peer_id} is living in a different world, of it is not certified "
                                 f"and maybe I expect certified only")
                        return False

                elif self.node_type is Node.WORLD:

                    # If this node hosts a world, we do not expect to interview agents in the private world connection,
                    # so something went wrong here, let's reject it
                    self.out(f"Peer f{peer_id} sent a profile in the private network, unexpected")
                    return False

    def __interview_clean(self):
        """Removes outdated or timed-out interview requests from the queue."""
        cur_time = self.clock.get_time()
        agents_to_remove = []
        for peer_id, (profile_time, profile) in self.agents_to_interview.items():

            # Checking timeout
            if (cur_time - profile_time) > self.interview_timeout:
                self.out("Removing (disconnecting) due to timeout in interview queue: " + peer_id)
                agents_to_remove.append(peer_id)

        # Updating
        for peer_id in agents_to_remove:
            self.__purge(peer_id)  # This will also remove the peer from the queue of peers to interview

    def __connected_without_ack_clean(self):
        """Removes connected peers from the queue if they haven't sent an acknowledgment within the timeout period."""
        cur_time = self.clock.get_time()
        agents_to_remove = []
        for peer_id, connection_time in self.agents_expected_to_send_ack.items():

            # Checking timeout
            if (cur_time - connection_time) > self.connect_without_ack_timeout:
                self.out("Removing (disconnecting) due to timeout in the connected-without-ack queue: " + peer_id)
                agents_to_remove.append(peer_id)

        # Updating
        for peer_id in agents_to_remove:
            self.__purge(peer_id)  # This will also remove the peer from the queue of in the connected-without-ack queue

    def __purge(self, peer_id: str):
        """Removes a peer from all relevant connection lists and queues.

        Args:
            peer_id: The peer ID of the node to purge.
        """
        self.hosted.remove_agent(peer_id)
        self.conn.remove(peer_id)

        # Clearing also the contents of the list of interviews
        if peer_id in self.agents_to_interview:
            del self.agents_to_interview[peer_id]

        # Clearing the temporary list of connected agents
        if peer_id in self.agents_expected_to_send_ack:
            del self.agents_expected_to_send_ack[peer_id]

    def __root(self, api: str, payload: dict):
        """Sends a POST request to the root server's API endpoint.

        Args:
            api: The API endpoint to send the request to.
            payload: The data to be sent in the request body.

        Returns:
            The 'data' field from the server's JSON response.
        """
        response_fields = ["state", "flags", "data"]

        try:
            api = self.root_endpoint + ("/" if self.root_endpoint[-1] != "/" and api[0] != "/" else "") + api
            payload["node_token"] = self.node_token  # Adding token to let the server verify
            response = requests.post(api,
                                     json=payload,
                                     headers={"Content-Type": "application/json"})

            if response.status_code == 200:
                ret = response.json()
                if response_fields is not None:
                    for field in response_fields:
                        if field not in ret:
                            raise GenException(f"Missing key '{field}' in the response to {api}: {ret}")
            else:
                raise GenException(f"Request {api} failed with status code {response.status_code}")
        except Exception as e:
            self.err(f"An error occurred while making the POST request: {e}")
            raise GenException(f"An error occurred while making the POST request: {e}")

        if ret['state']['code'] != "ok":
            raise GenException("[" + api + "] " + ret['state']['message'])

        return ret['data']

    @staticmethod
    def __analyze_code(file_in_memory):
        """Analyzes a string of Python code for dangerous or unsafe functions and modules.

        Args:
            file_in_memory: The string of Python code to analyze.

        Returns:
            True if the code is considered safe, otherwise False.
        """
        dangerous_functions = {"eval", "exec", "compile", "system", "__import__", "input"}
        dangerous_modules = {"subprocess"}

        def is_suspicious(ast_node):

            # Detect bare function calls like eval(...)
            if isinstance(ast_node, ast.Call):
                # case: eval(...)  (ast.Name)
                if isinstance(ast_node.func, ast.Name):
                    return ast_node.func.id in dangerous_functions
                # case: something.eval(...)  (ast.Attribute)
                elif isinstance(ast_node.func, ast.Attribute):
                    attr_name = ast_node.func.attr
                    # 1) If attribute name is one of the dangerous_functions, only flag it
                    #    if the object is a suspicious module (os, subprocess, etc.)
                    if attr_name in dangerous_functions:
                        value = ast_node.func.value
                        # example: os.system(...)  => ast.Name(id='os')
                        if isinstance(value, ast.Name):
                            if value.id in dangerous_modules:
                                return True
                        # example: package.subpackage.func(...) => ast.Attribute
                        # check top-level name if possible: walk down to the leftmost Name
                        left = value
                        while isinstance(left, ast.Attribute):
                            left = left.value
                        if isinstance(left, ast.Name) and left.id in dangerous_modules:
                            return True
                    # 2) Also catch explicit module imports used directly:
                    #    subprocess.run(...), os.system(...), etc.
                    if isinstance(ast_node.func.value, ast.Name):
                        if ast_node.func.value.id in dangerous_modules:
                            # if the module is suspicious, any attribute call is risky
                            return True

            # Detect imports
            if isinstance(ast_node, (ast.Import, ast.ImportFrom)):
                for alias in ast_node.names:
                    if alias.name.split('.')[0] in dangerous_modules:
                        return True

            return False

        try:
            tree = ast.parse(file_in_memory)
        except SyntaxError:
            return False

        for _ast_node in ast.walk(tree):
            if is_suspicious(_ast_node):
                return False

        return True

    def __handle_inspector_command(self, cmd: str, arg):
        """Handles commands received from an inspector node.

        Args:
            cmd: The command string.
            arg: The argument for the command.
        """
        self.out(f"Handling inspector message {cmd}, with arg {arg}")
        print(f"Handling inspector message {cmd}, with arg {arg}")

        if arg is not None and not isinstance(arg, str):
            self.err(f"Expecting a string argument from the inspector!")
        else:
            if cmd == "ask_to_join_world":
                print(f"Inspector asked to join world: {arg}")
                self.ask_to_join_world(node_name=arg)
            elif cmd == "ask_to_get_in_touch":
                print(f"Inspector asked to get in touch with an agent: {arg}")
                self.ask_to_get_in_touch(node_name=arg, public=True)
            elif cmd == "leave":
                print(f"Inspector asked to leave an agent: {arg}")
                self.leave(arg)
            elif cmd == "leave_world":
                print(f"Inspector asked to leave the current world")
                self.leave_world()
            elif cmd == "pause":
                print("Inspector asked to pause")
                self.__inspector_told_to_pause = True
            elif cmd == "play":
                print("Inspector asked to play")
                self.__inspector_told_to_pause = False
            elif cmd == "save":
                print("Inspector asked to save")
                self.hosted.save(arg)
            else:
                self.err("Unknown inspector command")

    def __send_to_inspector(self):
        """Sends status updates and data to the connected inspector node."""

        # Collecting console
        f = self._output_messages_last_pos - self._output_messages_count + 1  # Included
        t = self._output_messages_last_pos  # Included
        ff = -1
        tt = -1
        if t >= 0 > f:  # If there is something, and we incurred in the circular organization (t: valid; f: negative)
            ff = len(self._output_messages) + f  # Included
            tt = len(self._output_messages) - 1  # Included
            f = 0
        elif t < 0:  # If there are no messages at all (t: -1; f: 0 - due to the way we initialized class attributes)
            f = -1
            t = -1
        console = {'output_messages': self._output_messages[ff:tt+1] + self._output_messages[f:t+1]}

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
            # so we patch this thing by adding it here
            if self.hosted.in_world() and self.conn.world_node_peer_id is not None:
                all_agents_profiles[self.conn.world_node_peer_id] = self.hosted.world_profile.get_all_profile()
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
        if not self.conn.send(self.inspector_peer_id, channel_trail=None,
                              content_type=Msg.CONSOLE_AND_BEHAV_STATUS,
                              content=console_behav_status_and_behav):
            self.err("Failed to send data to the inspector")

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

                self.hosted.deb(f"[__send_to_inspector] Preparing to send stream samples from {net_hash}, {name}")
                content[(peer_id + "|" + name) if peer_id not in my_peer_ids else name] = \
                    {'data': data, 'data_tag': stream.get_tag(), 'data_uuid': stream.get_uuid()}

            # Checking if there is something valid in this group of streams to send to inspector
            if not something_to_send:
                self.hosted.deb(f"[__send_to_inspector] No stream samples to send to inspector for {net_hash}, "
                                f"all internal streams returned None")
                continue

            self.hosted.deb(f"[__send_to_inspector] Sending samples of {net_hash} by direct message, to inspector")
            name_or_group = DataProps.name_or_group_from_net_hash(net_hash)
            if not self.conn.send(self.inspector_peer_id, channel_trail=name_or_group,
                                  content_type=Msg.STREAM_SAMPLE, content=content):
                self.err(f"Failed to send stream sample data to the inspector (hash: {net_hash})")


class NodeSynchronizer:
    DEBUG = True

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
        self.clock = Clock()
        self.synch_cycle = -1
        self.synch_cycles = -1

        # Visualization-related attributes
        self.using_server = False
        self.server_checkpoints = None
        self.skip_clear_for = 0
        self.step_event = None  # Event that triggers a new step (manipulated by the server)
        self.wait_event = None  # Event that triggers a new "wait-for-step-event" case (manipulated by the server)
        self.next_checkpoint = 0
        self.server_checkpoints = None
        self.gap = 0.  # Seconds

    def add_node(self, node: Node):
        """Adds a new node to the synchronizer.

        Args:
            node: The node to add.
        """
        self.nodes.append(node)

        if node.node_type == Node.AGENT:
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
        node.debug_server_running = True

    def run(self, synch_cycles: int | None = None):
        """Starts the main execution loop for the node.

        Args:
            synch_cycles: The number of clock cycles to run the loop for. If None, runs indefinitely.
        """
        if self.world is None:
            raise GenException("Missing world node")

        # External events
        if self.using_server:
            self.step_event = threading.Event()
            self.wait_event = threading.Event()

        # Main loop
        self.synch_cycles = synch_cycles
        self.synch_cycle = 0

        try:
            while True:

                # In server mode, we wait for an external event to go ahead (step_event.set())
                if self.using_server:
                    self.wait_event.set()
                    self.step_event.wait()
                    self.wait_event.clear()

                state_changed = False
                world_node = None
                for node in self.nodes:
                    if node.node_type == Node.AGENT:
                        node.run(cycles=1)
                        if self.gap > 0.:
                            time.sleep(self.gap)
                        state_changed = state_changed or node.agent.behav.get_state_changed()
                    else:
                        world_node = node
                if world_node is not None:
                    world_node.run(cycles=1)
                    if self.gap > 0.:
                        time.sleep(self.gap)

                if NodeSynchronizer.DEBUG and state_changed:
                    for node in self.nodes:
                        if node.node_type == Node.AGENT:
                            print(f"[DEBUG NODE SYNCHRONIZER] {node.agent.get_name()} "
                                  f"state: {node.agent.behav.get_state_name()}")

                # Matching checkpoints
                if self.server_checkpoints is not None and self.server_checkpoints["current"] >= 0:
                    self.server_checkpoints["matched"] = -1
                    checkpoint = self.server_checkpoints["checkpoints"][self.server_checkpoints["current"]]
                    agent = checkpoint["agent"]
                    state = checkpoint["state"] if "state" in checkpoint else None

                    if agent not in self.nodes:
                        raise GenException(f"Unknown agent in the checkpoint list: {agent}")
                    behav = self.nodes[agent].agent.behav
                    if not (state is None or state in behav.states):
                        raise GenException(f"Unknown state in the checkpoint list: {state}")

                    if state is None or behav.state == state:
                        if "skip" not in checkpoint:
                            self.server_checkpoints["matched"] = self.server_checkpoints["current"]
                            self.server_checkpoints["current"] += 1
                            if self.server_checkpoints["current"] >= len(self.server_checkpoints["checkpoints"]):
                                self.server_checkpoints["current"] = -1  # This means: no more checkpoints
                        else:
                            checkpoint["skip"] -= 1
                            if checkpoint["skip"] <= 0:
                                self.server_checkpoints["current"] += 1
                                if self.server_checkpoints["current"] >= len(self.server_checkpoints["checkpoints"]):
                                    self.server_checkpoints["current"] = -1  # This means: no more checkpoints

                # In step mode, we clear the external event to be able to wait for a new one
                if self.using_server:
                    if self.skip_clear_for == 0:
                        self.step_event.clear()
                    elif self.skip_clear_for == -2:  # Infinite play
                        pass
                    elif self.skip_clear_for == -1:  # Play until next state
                        if state_changed:
                            self.step_event.clear()
                    elif self.skip_clear_for == -3:  # Play until next checkpoint:
                        if self.server_checkpoints["matched"] >= 0:
                            self.step_event.clear()
                    else:
                        self.skip_clear_for -= 1

                self.synch_cycle += 1

                # Stop condition on the number of cycles
                if self.synch_cycles is not None and self.synch_cycle == self.synch_cycles:
                    break
        except KeyboardInterrupt:
            print("\nDetected Ctrl+C! Exiting gracefully...")
