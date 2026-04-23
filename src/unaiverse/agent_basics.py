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
import asyncio
import os
import torch
import pickle
import uuid as _uuid
import importlib.resources
from PIL.Image import Image
from unaiverse.clock import clock
from unaiverse.stats import Stats
from unaiverse.custom import Custom
from collections.abc import Callable
from unaiverse.utils.logger import log
from unaiverse.hsm.action import Action
from typing_extensions import deprecated
from unaiverse.hsm.hsm import HybridStateMachine
from unaiverse.networking.p2p.messages import Msg
from unaiverse.networking.node.connpool import NodeConn
from unaiverse.networking.node.profile import NodeProfile
from unaiverse.streams.streams import Stream, BufferedStream
from unaiverse.streams.dataprops import DataProps, StreamType
from unaiverse.interaction import Interaction, InteractionManager
from unaiverse.streams.streamproxy import StreamProxy, StreamsProxyWithDefaults
from unaiverse.modules.utils import AgentProcessorChecker, ModuleWrapper, HumanModule, has_human_processor
from unaiverse.utils.misc import (GenException, FileTracker, collect_py_files, load_agent_in_memory, pack_py_files,
                                  unpack_py_files)


class AgentBasics:
    """This class contains those methods and properties that are about building the agent, known agents,
    known streams, etc., and no actions at all (see the class "Agent" for actions)."""

    # Role bits (a.k.a. role int): default roles, shared by every possible agent
    ROLE_PUBLIC = 0 << 0  # 00000000 = 0 means "public"
    ROLE_WORLD_MASTER = (1 << 0) | (1 << 1)  # 00000011 = 3 means "world master" (the first bit means "about world")
    ROLE_WORLD_AGENT = (1 << 0) | (0 << 1)  # 00000001 = 1 means "world agent" (the first bit means "about world")
    CUSTOM_ROLES = []

    # From role bits (int) to string
    ROLE_BITS_TO_STR = {
        ROLE_PUBLIC: "public_agent",
        ROLE_WORLD_MASTER: "world_master",
        ROLE_WORLD_AGENT: "world_agent",
    }

    # From role string to bits (int)
    ROLE_STR_TO_BITS = {
        "public_agent": ROLE_PUBLIC,
        "world_master": ROLE_WORLD_MASTER,
        "world_agent": ROLE_WORLD_AGENT,
    }

    # Types of badges
    BADGE_TYPES = {'completed', 'attended', 'intermediate', 'pro'}

    # The type associated to a human: it is not exploited at a node-level, only at agent level
    HUMAN = "human"  # Human agent

    def __init__(self,
                 proc: ModuleWrapper | torch.nn.Module | None,
                 proc_inputs: list[StreamType | str] | None = None,
                 proc_outputs: list[StreamType | str] | None = None,
                 proc_opts: dict | None = None,
                 behav: HybridStateMachine | None = None,
                 behav_lone_wolf: HybridStateMachine | str = "serve",
                 merge_flat_stream_labels: bool = False,
                 buffer_generated: bool = False,
                 buffer_generated_by_others: str = "none",
                 world_folder: str | None = None,
                 policy_filter: callable = None,
                 policy_filter_lone_wolf: callable = None):
        """Create a new agent.

        Args:
            proc: The processing module (e.g., a neural network) for the agent. Can be None or "default".
            proc_inputs: list of DataProps defining the expected inputs for processor (if None it will be guessed).
            proc_outputs: list of DataProps defining the expected outputs from processor (if None it will be guessed).
            proc_opts: A dictionary of options for the processor.
            behav: The HybridStateMachine that describes the agent's behavior when joining a world.
            behav_lone_wolf: The HybridStateMachine that describes the agent's behavior when in the public net
                (it can also be a string "serve" or "ask", that will load pre-designed HSMs).
            merge_flat_stream_labels: If True, merges flat stream labels across all owned streams.
            buffer_generated: If True, generated streams will be buffered.
            buffer_generated_by_others: If set to "one" or "all", streams generated by other agents will be buffered
                ("one" per peer or "all"). If set to "none", no buffering will happen (default).
            world_folder: World only. Folder where the world data is (role files, represented by *.json behavior files).
            policy_filter: The name of a method of the Agent class or a function that implements a
                policy filtering function, overriding what the action-selection-policy decided (when in a world).
            policy_filter_lone_wolf: Same as policy_filter, but for the public network.
        """

        # Agent-related features
        self.behav = behav  # HSM that describes the agent behavior in the private/world net
        self.behav_lone_wolf = behav_lone_wolf  # HSM that describes the agent behavior in the public net
        self.behav_wildcards = {}
        self.proc = proc
        self.proc_updated_since_last_save = False
        self.proc_inputs = proc_inputs
        self.proc_outputs = proc_outputs
        self.proc_opts = proc_opts
        self.proc_last_inputs = None
        self.proc_last_outputs = None
        self.proc_human_peer_id_to_interaction = {}
        self.proc_optional_inputs: list[dict] | None = None
        self.proc_net_hash: dict[str, None | str] = {'public': None, 'private': None}
        self.proc_in_net_hash: dict[str, None | str] = {'public': None, 'private': None}
        self.merge_flat_stream_labels = merge_flat_stream_labels
        self.buffer_generated = buffer_generated
        self.buffer_generated_by_others = buffer_generated_by_others
        self.world_folder = world_folder
        self.policy_filter = policy_filter
        self.policy_filter_opts = {}
        self.policy_filter_lone_wolf = policy_filter_lone_wolf
        self.policy_filter_lone_wolf_opts = {}
        self.clock = clock  # Backward compatibility

        if self.buffer_generated_by_others not in {"one", "all", "none"}:
            raise GenException("Param buffer_generated_by_others can be set to 'one', 'all', or 'none' only.")

        # Streams
        self.known_streams = {}  # All streams that are known to this agent, organized by net hash
        self.owned_streams = {}  # The streams that are generated/offered by this agent, organized by net hash
        self.env_streams = {}  # The owned streams that come from environmental sources (e.g., a camera), by net hash
        self.proc_streams = {}  # The owned streams that are generated by the agent's processor, organized by net hash
        self.proc_in_streams = {}  # The owned streams that are input to the agent's processor, organized by net hash
        self.known_streams_by_user_hash = {}  # From peer_id:name to stream object
        self.owned_streams_by_user_hash = {}  # From stream name to stream object
        self.env_streams_by_user_hash = {}  # From peer_id:name to stream object
        self.proc_streams_by_user_hash = {}  # From peer_id:name to stream object
        self.proc_in_streams_by_user_hash = {}  # From peer_id:name to stream object
        self.compat_in_streams = set()  # Streams compatible with the processor input (dynamically set)
        self.compat_out_streams = set()  # Streams compatible with the processor output (dynamically set)

        # Agents, world masters, expected world masters
        self.all_agents = {}  # ID -> profile (all types of agent)
        self.public_agents = {}  # ID -> profile of lone wolves talking to this world in a public manner (profile)
        self.world_agents = {}  # ID -> profile of all agents living in this world (profile)
        self.world_masters = {}  # ID -> profile of all master-agents living in this world (profile)
        self.human_agents = {}  # ID -> profile (human agents)
        self.artificial_agents = {}  # ID -> profile (artificial agent)
        self.world_profile = None
        self.is_world = False  # If this instance is about a world: it will be discovered at creation time

        # World specific attributes (they are only used if this agent is actually a world)
        self.packed_agent_files: str | None = None
        self.role_to_behav = {}
        self.agent_badges: dict[str, list[dict]] = {}  # Peer_id -> collected badges for other agents
        self.role_changed_by_world: bool = False
        self.received_address_update: bool = False

        # Internal properties about the way streams are used
        self.last_buffered_peer_id_to_info = {}  # If buffering was turned on
        self.last_ref_uuid = None
        self.overridden_action_step = None
        self.locked_set_proc_input = False

        # Stats
        self.stats: Stats | None = None
        self.agent_stats_code = None

        # Information inherited from the node that hosts this agent
        self._node_name = "unk"
        self._node_conn = None
        self._node_profile = None
        self._node_out_fcn = print
        self._node_ask_to_get_in_touch_fcn = None
        self._node_purge_fcn = None
        self._node_agents_waiting = None
        self._node_identity_dir = ''

        # Utilities
        self.__proc_in_streams_by_user_hash_pub = {}
        self.__proc_streams_by_user_hash_pub = {}
        self.__env_streams_by_user_hash_pub = {}
        self.__proc_in_streams_by_user_hash_prv = {}
        self.__proc_streams_by_user_hash_prv = {}
        self.__env_streams_by_user_hash_prv = {}

        # Checking
        if not (self.proc is None or
                (isinstance(self.proc, torch.nn.Module) or (isinstance(self.proc, str) and self.proc == "default"))):
            raise GenException("Invalid data processor: it must be either the string 'default' or a torch.nn.module")
        if not (self.behav is None or isinstance(self.behav, HybridStateMachine)):
            raise GenException("Invalid behavior: it must be either None or a HybridStateMachine")

        # Filling (guessing) missing processor-related info (proc_inputs and proc_outputs)
        # and allocating a dummy processor if it was not specified (if None)
        AgentProcessorChecker(self)

        # The stream_hash of compatible streams for each data_props are stored in a set
        self.compat_in_streams = [set() for _ in range(len(self.proc_inputs))] \
            if self.proc_inputs is not None else None
        self.compat_out_streams = [set() for _ in range(len(self.proc_outputs))] \
            if self.proc_outputs is not None else None

        # Interaction Manager, stdin/stdout
        self.im = InteractionManager(self, max_interactions=Custom.MAX_INTERACTIONS)
        self.stdin = StreamsProxyWithDefaults(default_values=[item["default_value"] if item["has_default"] else None
                                                              for item in self.proc_optional_inputs])
        self.stdtar = StreamProxy()
        self.stdext = StreamProxy()
        self.stdout = StreamProxy()

        # Loading default public HSM
        if hasattr(self, "do_gen"):  # Trick to distinguish if this is an Agent or a World (both sons of this class)
            self.is_world = False

            # Setting an empty HSM as default is not provided (private/world)
            if self.behav is None:
                self.behav = HybridStateMachine(self, policy=self.policy_default)
                self.behav.add_state("empty")

            if self.behav_lone_wolf is not None and isinstance(self.behav_lone_wolf, str):
                template_string = self.behav_lone_wolf
                if template_string == "serve":
                    json_to_load = "public.json"
                elif template_string == "ask":
                    json_to_load = "public.json"
                else:
                    raise ValueError("Invalid behav_lone_wolf: it must be an HybridStateMachine or a string "
                                     "in ('serve', 'ask')")

                # Safe way to load a file packed in a pip package
                self.behav_lone_wolf = HybridStateMachine(self, policy=self.policy_default)
                utils_path = importlib.resources.files("unaiverse.utils")
                json_file = utils_path.joinpath(json_to_load)
                file = json_file.open()
                self.behav_lone_wolf.load(file)
                file.close()
                self.set_policy_filter(self.policy_filter_lone_wolf, public=True)
        else:
            self.is_world = True
            if self.world_folder is None:
                raise GenException("No world folder was indicated (world_folder argument)")

    def set_node_info(self, conn: NodeConn, profile: NodeProfile,
                      ask_to_get_in_touch_fcn: Callable, purge_fcn: Callable, node_identity_dir: str,
                      agents_waiting: dict) -> bool:
        """Set the required information from the node that hosts this agent.

        Args:
            conn: The connection pool manager from the node.
            profile: The profile of the hosting node.
            ask_to_get_in_touch_fcn: The function to call to request getting in touch with another peer.
            purge_fcn: The function to call to purge (kill/disconnect) a connection.
            node_identity_dir: The folder where the node identity files are stored.
            agents_waiting: Dict of agents that connected to this node but have not been evaluated yet to be added.

        Returns:
            True upon successful setup.
        """

        # Getting basic references
        self._node_conn = conn
        self._node_profile = profile
        self._node_name = profile.get_static_profile()['node_name']
        self._node_ask_to_get_in_touch_fcn = ask_to_get_in_touch_fcn
        self._node_purge_fcn = purge_fcn
        self._node_identity_dir = node_identity_dir
        self._node_agents_waiting = agents_waiting

        # Checking if human, marking the node
        if self.proc is not None and isinstance(self.proc.module, HumanModule):
            self._node_profile.get_static_profile()["node_type"] = self.HUMAN

        # Adding peer_id information into the already existing stream data (if any)
        # (initially marked with generic wildcards like <public_peer_id>, ...)
        net_hashes = list(self.known_streams.keys())
        for net_hash in net_hashes:
            if net_hash.startswith("<public_peer_id>") or net_hash.startswith("<private_peer_id>"):
                stream_dict = self.known_streams[net_hash]
                for stream_obj in stream_dict.values():
                    self.add_stream(stream_obj, owned=True)  # This will also re-add streams using the node clock

        # Removing place-holder streams
        for peer_id in ["<public_peer_id>", "<private_peer_id>"]:
            to_remove = []
            for net_hash in self.known_streams.keys():
                if DataProps.peer_id_from_net_hash(net_hash) == peer_id:
                    for _name, _stream in self.known_streams[net_hash].items():
                        to_remove.append((net_hash, _name))

            # Removing
            for (net_hash, name) in to_remove:
                user_hash = DataProps.user_hash_from_net_hash(net_hash, name)
                del self.known_streams[net_hash][name]
                if len(self.known_streams[net_hash]) == 0:
                    del self.known_streams[net_hash]
                if user_hash in self.known_streams_by_user_hash:
                    del self.known_streams_by_user_hash[user_hash]

                # Removing all the owned streams (environment and processor streams are of course "owned")
                if net_hash in self.owned_streams:
                    if name in self.owned_streams[net_hash]:
                        del self.owned_streams[net_hash][name]
                        if len(self.owned_streams[net_hash]) == 0:
                            del self.owned_streams[net_hash]
                if user_hash in self.owned_streams_by_user_hash:
                    del self.owned_streams_by_user_hash[user_hash]
                if net_hash in self.env_streams:
                    if name in self.env_streams[net_hash]:
                        del self.env_streams[net_hash][name]
                        if len(self.env_streams[net_hash]) == 0:
                            del self.env_streams[net_hash]
                if user_hash in self.env_streams_by_user_hash:
                    del self.env_streams_by_user_hash[user_hash]
                if net_hash in self.proc_streams:
                    if name in self.proc_streams[net_hash]:
                        del self.proc_streams[net_hash][name]
                        if len(self.proc_streams[net_hash]) == 0:
                            del self.proc_streams[net_hash]
                if user_hash in self.proc_streams_by_user_hash:
                    del self.proc_streams_by_user_hash[user_hash]
                if net_hash in self.proc_in_streams:
                    if name in self.proc_in_streams[net_hash]:
                        del self.proc_in_streams[net_hash][name]
                        if len(self.proc_in_streams[net_hash]) == 0:
                            del self.proc_in_streams[net_hash]
                if user_hash in self.proc_in_streams_by_user_hash:
                    del self.proc_in_streams_by_user_hash[user_hash]
                log.misc(f"Successfully removed known stream with network hash {net_hash}, stream name: {name}")

        # World only: loading action files and refactoring (or building) JSON files of the different roles.
        # This where the world guesses roles.
        if self.is_world:

            # Check role-JSON files in the world folder
            role_json_tracker = FileTracker(self.world_folder, ext=".json")

            # This usually does nothing, but if you like to dynamically create JSON files, overload this method
            self.create_behav_files()

            # Loading and refactoring roles and behaviors
            self.load_and_refactor_action_file_and_behav_files(force_save=role_json_tracker.something_changed())

            # Building combination of default roles (considering public, world_agent, world_master default roles), and
            # agent/world specific roles
            self.augment_roles()

            # Loading the custom Stats code
            if self.world_folder is not None:
                stats_file = os.path.join(self.world_folder, 'stats.py')
                if os.path.exists(stats_file):
                    log.misc(f"Found custom stats.py at {stats_file}")
                    try:
                        with open(stats_file, 'r', encoding='utf-8') as file:
                            self.agent_stats_code = file.read()
                    except Exception as e:
                        raise GenException(f'Error while reading/loading the stats.py file: {stats_file} [{e}]')

        # Creating streams associated to the processor input (right now we assume there is no need to buffer them)
        self.create_proc_input_streams(buffered=False)

        # Creating streams associated to the processor output
        self.create_proc_output_streams(buffered=self.buffer_generated)

        # Updating node profile by indicating the processor-related streams
        self.update_streams_in_profile()

        # Blocking stdin for humans
        if has_human_processor(self):
            self.set_default_stdin_binding(public=True)
            self.stdin.disable()
            self.set_default_stdin_binding(public=False)
            self.stdin.disable()

        return True

    def get_connection_pool_manager(self) -> NodeConn:
        return self._node_conn

    def set_default_stdin_binding(self, public: bool | None = None) -> None:
        """Set default bindings for the stdin."""

        if (public is not None and not public) or (public is None and self.behaving_in_world()):
            self.stdin.bind(self.__proc_in_streams_by_user_hash_prv, uuid=Custom.SYSTEM_INTERACTION_UUID)

        if (public is not None and public) or (public is None and not self.behaving_in_world()):
            self.stdin.bind(self.__proc_in_streams_by_user_hash_pub, uuid=Custom.SYSTEM_INTERACTION_UUID)

    def set_default_stream_binding(self, public: bool | None = None) -> None:
        """Set default bindings for the stdin, stdtar, stdext, stdout stream proxies."""

        if (public is not None and not public) or (public is None and self.behaving_in_world()):
            self.stdin.bind(self.__proc_in_streams_by_user_hash_prv, uuid=Custom.SYSTEM_INTERACTION_UUID)
            self.stdtar.bind({}, uuid=Custom.SYSTEM_INTERACTION_UUID)
            self.stdext.bind(self.__env_streams_by_user_hash_prv, uuid=Custom.SYSTEM_INTERACTION_UUID)
            self.stdout.bind(self.__proc_streams_by_user_hash_prv, uuid=Custom.SYSTEM_INTERACTION_UUID)

        if (public is not None and public) or (public is None and not self.behaving_in_world()):
            self.stdin.bind(self.__proc_in_streams_by_user_hash_pub, uuid=Custom.SYSTEM_INTERACTION_UUID)
            self.stdtar.bind({}, uuid=Custom.SYSTEM_INTERACTION_UUID)
            self.stdext.bind(self.__env_streams_by_user_hash_pub, uuid=Custom.SYSTEM_INTERACTION_UUID)
            self.stdout.bind(self.__proc_streams_by_user_hash_pub, uuid=Custom.SYSTEM_INTERACTION_UUID)

    def get_default_stdin_bindings(self, public: bool | None = None) -> dict:
        """Returns the streams that are bind by default on the stdin."""
        if (public is not None and not public) or (public is None and self.behaving_in_world()):
            return self.__proc_in_streams_by_user_hash_prv

        if (public is not None and public) or (public is None and not self.behaving_in_world()):
            return self.__proc_in_streams_by_user_hash_pub

    def get_default_stream_bindings(self, public: bool | None = None) -> tuple[dict, dict, dict, dict]:
        """Returns the streams that are bind by default on the standard streams (stdin, stdtar, stdext, stdout)."""
        if (public is not None and not public) or (public is None and self.behaving_in_world()):
            return (self.__proc_in_streams_by_user_hash_prv, {},
                    self.__env_streams_by_user_hash_prv, self.__proc_streams_by_user_hash_prv)

        if (public is not None and public) or (public is None and not self.behaving_in_world()):
            return (self.__proc_in_streams_by_user_hash_pub, {},
                    self.__env_streams_by_user_hash_pub, self.__proc_streams_by_user_hash_pub)

    def get_proc_output_net_hash(self, public: bool = True) -> str | None:
        """Return the network hash of the processor output streams.

        Args:
            public: If True, returns the public net hash, otherwise the private one.

        Returns:
            The network hash string, or None if not yet set.
        """
        return self.proc_net_hash['public'] if public else self.proc_net_hash['private']

    def get_proc_input_net_hash(self, public: bool = True) -> str | None:
        """Return the network hash of the processor input streams.

        Args:
            public: If True, returns the public net hash, otherwise the private one.

        Returns:
            The network hash string, or None if not yet set.
        """
        return self.proc_in_net_hash['public'] if public else self.proc_in_net_hash['private']

    def prepare_stdin_if_human(self, public: bool, peer_id: str):
        self.set_default_stdin_binding(public)

        # Zero-step: if we are in a world, the peer ID used for indexing the interaction cache is the world peer ID
        peer_id = peer_id if public else self._node_conn.get_world_peer_id()

        # First of all, clearing residuals of interactions that were completed in the past
        to_remove = []
        for _peer_id, _interaction in self.proc_human_peer_id_to_interaction.items():
            if _interaction.is_completed():
                to_remove.append(_peer_id)
        for _peer_id in to_remove:
            del self.proc_human_peer_id_to_interaction[_peer_id]

        # Getting existing interaction or defaulting to the system one
        if peer_id in self.proc_human_peer_id_to_interaction:
            interaction = self.proc_human_peer_id_to_interaction[peer_id]
            return interaction.uuid
        else:
            return Custom.SYSTEM_INTERACTION_UUID

    @staticmethod
    def generate_id() -> str:
        """Generate a random 8-character hexadecimal ID string.

        Returns:
            An 8-character hexadecimal string.
        """
        return _uuid.uuid4().hex[0:8]

    def inject_received_interaction(self, sender: str, interaction_dict: dict):

        # Creating the interaction object
        interaction = Interaction.from_dict(interaction_dict)

        # Also register with the HSM for action selection (backward compat bridge)
        behav = self.behav_lone_wolf \
            if sender in self.public_agents else self.behav
        if not behav.request_action(interaction):
            log.error(f"Cannot enqueue the interaction, incompatible action. "
                      f"Interaction is: {interaction}")
        else:
            # Register with the InteractionManager
            if hasattr(self, 'im'):
                log.misc(f"Registering received interaction for action "
                         f"{interaction.action_name}")
                self.im.register_received(interaction)

    @staticmethod
    def build_augmented_roles_dictionaries(custom_roles: list[str] | set[str]) -> tuple[dict[int, str], dict[str, int]]:
        """Augment the custom roles (role1, role2, etc.) with the default ones (public, world_master, etc.), generating
        all the mixed roles (world_master~role1, world_master~role2, ...)"""

        role_bits_to_str = {k: v for k, v in AgentBasics.ROLE_BITS_TO_STR.items()}
        role_str_to_bits = {k: v for k, v in AgentBasics.ROLE_STR_TO_BITS.items()}

        # Both Agent and World: Fusing basic roles and custom roles
        if len(custom_roles) > 0:
            if len(custom_roles) > 30:  # Safe value, could be increased
                log.critical("Maximum number of custom role overcame (max is 30)")
            for i, role_str in enumerate(custom_roles):
                role_int = 1 << (i + 2)  # 000000100, then 00001000, etc. (recall that the first two bits are reserved)
                role_str_to_bits[role_str] = role_int
                role_bits_to_str[role_int] = role_str
                AgentBasics.ROLE_STR_TO_BITS[role_str] = role_int

        # Both Agent and World: Augmenting roles
        roles_not_to_be_augmented = {AgentBasics.ROLE_PUBLIC,
                                     AgentBasics.ROLE_WORLD_AGENT,
                                     AgentBasics.ROLE_WORLD_MASTER}
        role_bits_to_str_original = {k: v for k, v in role_bits_to_str.items()}
        for role_int, role_str in role_bits_to_str_original.items():
            if role_int not in roles_not_to_be_augmented and "~" not in role_str:
                for role_base_int in {AgentBasics.ROLE_WORLD_AGENT, AgentBasics.ROLE_WORLD_MASTER}:
                    augmented_role_int = role_base_int | role_int
                    augmented_role_str = role_bits_to_str[role_base_int] + "~" + role_str
                    if augmented_role_str not in role_str_to_bits:
                        role_str_to_bits[augmented_role_str] = augmented_role_int
                        role_bits_to_str[augmented_role_int] = augmented_role_str

        return role_bits_to_str, role_str_to_bits

    def augment_roles(self,
                      role_bits_to_str: dict[int, str] | None = None,
                      role_str_to_bits: dict[str, int] | None = None):
        """Augment the custom roles (role1, role2, etc.) with the default ones (public, world_master, etc.), generating
        all the mixed roles (world_master~role1, world_master~role2, ...)"""

        if role_bits_to_str is None or role_str_to_bits is None:
            role_bits_to_str, role_str_to_bits = AgentBasics.build_augmented_roles_dictionaries(self.CUSTOM_ROLES)

        self.ROLE_BITS_TO_STR.clear()
        for k, v in role_bits_to_str.items():
            self.ROLE_BITS_TO_STR[k] = v
        self.ROLE_STR_TO_BITS.clear()
        for k, v in role_str_to_bits.items():
            self.ROLE_STR_TO_BITS[k] = v

    async def clear_world_related_data(self):
        """Destroy all the cached information that is about a world (useful when leaving a world) (async)."""

        # Clearing status variables
        self.reset_agent_status_attrs()

        # Clear/reset
        await self.__remove_all_world_private_streams()
        await self.__remove_all_world_related_agents()
        self._node_conn.reset_rendezvous_tag()

    def load_and_refactor_action_file_and_behav_files(self, force_save: bool = False):
        """This method is called when building a world object. It loads the behavior files and refactors them.
        It loads the action file agent.py. It checks consistency between the agent action files agent.py and the roles
        in the behavior files.

            Args:
                force_save: Boolean to force the saving of the JSON and of a "pdf" folder with the PDFs of the state
                    machines.
        """

        # World only: the world discovers CUSTOM_ROLES from the JSON files in the world folder
        if self.world_folder is not None and self.is_world:

            # Guessing roles from the list of json files
            listdir = os.listdir(self.world_folder)  # Do this only once
            self.CUSTOM_ROLES = [os.path.splitext(f)[0] for f in listdir
                                 if os.path.isfile(os.path.join(self.world_folder, f))
                                 and f.lower().endswith(".json")]
            if len(self.CUSTOM_ROLES) == 0:
                raise GenException(f"No world-role files (*.json) were found in the world folder {self.world_folder}")

            # Default behaviours (getting roles, that are the names of the files with extension "json")
            default_behav_files = [os.path.join(self.world_folder, f) for f in listdir
                                   if os.path.isfile(os.path.join(self.world_folder, f)) and
                                   f.lower().endswith(".json")]

            # Loading Python files in the "world" folder
            try:
                self.packed_agent_files = pack_py_files(collect_py_files(self.world_folder,
                                                                         exclude_dirs={'pdf', 'stats'}))
            except Exception as e:
                raise GenException(f'Error while reading the agent-related python files in {self.world_folder} [{e}]')

            # Loading and refactoring behaviors
            agent_files = unpack_py_files(self.packed_agent_files)

            # Backward compatibility
            if len(self.CUSTOM_ROLES) > 0:
                found = 0
                found_agent_py = False
                for role in self.CUSTOM_ROLES:
                    for file_name in agent_files.keys():
                        if file_name == "agent.py":
                            found_agent_py = True
                        if file_name == f"{role}.py":
                            found += 1
                            break
                if found != len(self.CUSTOM_ROLES):
                    if found_agent_py:
                        code_content = agent_files["agent.py"]

                        # All roles go to the content of agent.py
                        agent_files = {f"{k}.py": code_content for k in self.CUSTOM_ROLES}
                        self.packed_agent_files = pack_py_files(agent_files)  # Repacking
                    else:
                        raise GenException(f'Mismatching JSON-file roles and .py files in {self.world_folder}')

            for role, default_behav_file in zip(self.CUSTOM_ROLES, default_behav_files):

                # Creating a dummy agent which supports the actions of the following state machines
                try:
                    dummy_agent, dummy_agent_memory_finder = load_agent_in_memory(agent_files, role, proc=None)
                    dummy_agent.CUSTOM_ROLES = self.CUSTOM_ROLES
                except Exception as e:
                    raise GenException(f'Unable to create a valid dummy agent object for agent with role {role} [{e}]')

                # Checking if the roles you wrote in agent.py are coherent with the JSON files in this folder
                if dummy_agent.CUSTOM_ROLES != self.CUSTOM_ROLES:
                    raise GenException(f"Mismatching roles. "
                                       f"Roles in JSON files: {self.CUSTOM_ROLES}. "
                                       f"Roles specified in the agent.py file: {dummy_agent.CUSTOM_ROLES}")

                try:
                    behav = HybridStateMachine(dummy_agent)
                    behav.load(default_behav_file)
                    self.role_to_behav[role] = str(behav)

                    # Adding roles and machines to profile
                    self._node_profile.get_dynamic_profile()['world_roles_fsm'] = self.role_to_behav
                except Exception as e:
                    raise GenException(f'Error while loading or handling '
                                       f'behav file {default_behav_file} for role {role} [{e}]')

                # Refactoring and saving PDF
                try:
                    if (force_save or
                            behav.save(os.path.join(self.world_folder, f'{role}.json'), only_if_changed=dummy_agent)):
                        os.makedirs(os.path.join(self.world_folder, 'pdf'), exist_ok=True)
                        behav.save_pdf(os.path.join(self.world_folder, 'pdf', f'{role}.pdf'))
                except Exception as e:
                    raise GenException(f'Error while saving the behav file {default_behav_file} for role {role} [{e}]')

                # Clearing memory
                dummy_agent_memory_finder.cleanup()

    def create_behav_files(self):
        """This method is called when building a world object. In your custom world-class, you can overload this method
        and create the JSON files with the role-related behaviors, if you like. Recall that acting like this is not
        mandatory at all: you can just manually create the JSON  files, and this method will simply do nothing."""
        pass

    def resolve_agent_ref(self, ref: str) -> str | None:
        """Resolve an agent name or peer_id to a peer_id.

        If ``ref`` is already a known peer_id, it is returned as-is.
        Otherwise, the method searches all known agents by node_name.

        Args:
            ref: A peer_id or agent name.

        Returns:
            The peer_id if found, None otherwise.
        """

        # If already a valid peer_id
        if ref in self.all_agents:
            return ref

        # Search by email@node_name (if only node_name is provided, assumes it is one of your agents)
        if '@' not in ref:
            your_email = self.get_profile().get_static_profile()['email']
            ref = your_email + '@' + ref

        for peer_id, profile in self.all_agents.items():
            static_profile = profile.get_static_profile() if hasattr(profile, 'get_static_profile') else {}
            if isinstance(static_profile, dict):
                if (static_profile.get('email') + '@' + static_profile.get('node_name')) == ref:
                    return peer_id

        return None

    def resolve_stream_ref(self, ref: str) -> str | None:
        """Resolve a stream name to a stream user or net hash (heuristic, gives priority to owned streams).

        Args:
            ref: A stream name, without peer IDs, just the name.

        Returns:
            The stream user hash or net hash if found, None otherwise.
        """

        # If already a valid user hash
        if ref in self.known_streams_by_user_hash:
            return ref

        # Search your own streams first (priority)
        for user_hash, stream_obj in self.owned_streams_by_user_hash.items():
            if ref == Stream.name_from_user_hash(user_hash):
                return user_hash
        for net_hash, stream_dict in self.owned_streams.items():
            if ref == Stream.name_or_group_from_net_hash(net_hash):
                return net_hash

        # Search all streams then
        for user_hash, stream_obj in self.known_streams_by_user_hash.items():
            if ref == Stream.name_from_user_hash(user_hash):
                return user_hash
        for net_hash, stream_dict in self.known_streams.items():
            if ref == Stream.name_or_group_from_net_hash(net_hash):
                return net_hash

        return None

    def get_name(self) -> str:
        """Returns the name of the agent or world from the node's profile.

        Args:
            None.

        Returns:
            The name of the agent or world.
        """
        return self._node_name

    def get_profile(self) -> NodeProfile:
        """Returns the profile of the node hosting this agent/world.

            Returns:
                The NodeProfile of this node.
        """
        return self._node_profile

    def get_role(self, agent: str) -> str | None:
        role_int = self._node_conn.get_role(agent)
        base_role_int = (role_int >> 2) << 2
        return self.ROLE_BITS_TO_STR[base_role_int] if base_role_int in self.ROLE_BITS_TO_STR else None

    def get_agents_by_role(self, role: str | list[str], handshake_completed: bool = True):
        role_list = role if isinstance(role, list) else [role]
        searched_roles_int = set()

        for role in role_list:
            is_augmented = "~" in role
            role_int = self.ROLE_STR_TO_BITS[role]
            if not is_augmented:
                role_int = self.ROLE_STR_TO_BITS[role]
                for role_prefix_int in {AgentBasics.ROLE_WORLD_AGENT, AgentBasics.ROLE_WORLD_MASTER}:
                    searched_roles_int.add(role_prefix_int | role_int)
            else:
                searched_roles_int = {role_int}

        found_agents = []
        for role_int in searched_roles_int:
            _, _found_connected_peer_ids = self._node_conn.find_addrs_by_role(role_int, return_peer_ids_too=True)
            if handshake_completed:
                for agent in _found_connected_peer_ids:
                    if agent in self.all_agents:
                        found_agents.append(agent)
            else:
                found_agents += _found_connected_peer_ids

        return found_agents

    def get_current_role(self, return_int: bool = False, ignore_base_role: bool = True) -> str | int | None:
        """Returns the current role of the agent.

        Args:
            return_int: If True, returns the integer representation of the role.
            ignore_base_role: If True, returns only the specific role part, not the base.

        Returns:
            The role as a string or integer, or None if the agent is not living in any worlds.
        """
        if self.in_world():
            role_str = self._node_profile.get_dynamic_profile()['connections']['role']
            if ignore_base_role:
                role_str = role_str.split("~")[-1]
            if not return_int:
                return role_str
            else:
                return self.ROLE_STR_TO_BITS[role_str]
        else:
            return None

    async def add_agent(self, peer_id: str, profile: NodeProfile,
                        add_proc_streams: bool = True,
                        add_env_streams: bool = True,
                        add_pubsub_streams: bool = True) -> bool:
        """Add a new known agent (async).

        Args:
            peer_id: The unique identifier of the peer.
            profile: The NodeProfile object containing the peer's/agent's information.
            add_proc_streams: True if the processor's streams of the peer must be added to the list of known streams,
                if compatible.
            add_env_streams: True if the environmental streams of the peer must be added to the list of known streams,
                if compatible.
            add_pubsub_streams: True if pubsub streams must be added/subscribed.

        Returns:
            True if the agent was successfully added, False otherwise.
        """

        # If the agent was already there, we remove it and add it again (in case of changes)
        await self.remove_agent(peer_id)  # It has no effects if the agent is not existing

        # Guessing the type of agent to add (accordingly to the default roles shared by every agent)
        role = self._node_conn.get_role(peer_id)
        self.all_agents[peer_id] = profile
        if role & 1 == self.ROLE_PUBLIC:
            self.public_agents[peer_id] = profile
            public = True
        elif role & 3 == self.ROLE_WORLD_AGENT:
            self.world_agents[peer_id] = profile
            public = False
        elif role & 3 == self.ROLE_WORLD_MASTER:
            self.world_masters[peer_id] = profile
            public = False
        else:
            log.error(f"Cannot add agent with peer ID {peer_id} - unknown role: {role}")
            return False

        # Human or artificial?
        if profile.get_static_profile()["node_type"] == AgentBasics.HUMAN:
            self.human_agents[peer_id] = profile
        else:
            self.artificial_agents[peer_id] = profile

        # Check compatibility of the streams owned by the agent we are adding with our-agent's processor
        if self.proc_outputs is not None and self.proc_inputs is not None:

            # Check compatibility of the environmental streams of the agent we are adding with our-agent's processor
            environmental_streams = profile.get_dynamic_profile()['streams']
            if (add_env_streams and environmental_streams is not None and
                    not (await self.add_compatible_streams(peer_id, environmental_streams,
                                                           buffered=False,
                                                           public=public,
                                                           skip_pubsub=not add_pubsub_streams))):
                return False

            # Check compatibility of the generated streams of the agent we are adding with our-agent's processor
            proc_streams = profile.get_dynamic_profile()['proc_outputs']
            if (add_proc_streams and proc_streams is not None and
                    not (await self.add_compatible_streams(peer_id, profile.get_dynamic_profile()['proc_outputs'],
                                                           buffered=False,
                                                           public=public,
                                                           skip_pubsub=not add_pubsub_streams))):
                return False

        log.misc(f"Successfully added agent {profile.get_static_profile()['node_name']} "
                 f"with peer ID {peer_id} (public: {public})")
        return True

    async def remove_agent(self, peer_id: str):
        """Remove an agent (async).

        Args:
            peer_id: The unique identifier of the peer to remove.
        """
        if peer_id in self.all_agents:

            # Removing from agent list
            del self.all_agents[peer_id]
            if peer_id in self.world_agents:
                del self.world_agents[peer_id]
            elif peer_id in self.world_masters:
                del self.world_masters[peer_id]
            elif peer_id in self.public_agents:
                del self.public_agents[peer_id]

            if peer_id in self.artificial_agents:
                del self.artificial_agents[peer_id]
            elif peer_id in self.human_agents:
                del self.human_agents[peer_id]

            # Clearing from the list of processor-input-compatible-streams
            if self.compat_in_streams is not None:
                for i, _ in enumerate(self.compat_in_streams):
                    to_remove = []
                    for net_hash_name in self.compat_in_streams[i]:
                        if DataProps.peer_id_from_net_hash(net_hash_name[0]) == peer_id:
                            to_remove.append(net_hash_name)
                    for net_hash_name in to_remove:
                        self.compat_in_streams[i].remove(net_hash_name)

            # Clearing from the list of processor-output-compatible-streams
            if self.compat_out_streams is not None:
                for i, _ in enumerate(self.compat_out_streams):
                    to_remove = []
                    for net_hash_name in self.compat_out_streams[i]:
                        if DataProps.peer_id_from_net_hash(net_hash_name[0]) == peer_id:
                            to_remove.append(net_hash_name)
                    for net_hash_name in to_remove:
                        self.compat_out_streams[i].remove(net_hash_name)

            # Clearing streams owned by the removed agent from the list of known streams
            await self.remove_streams(peer_id)

            # Removing from the status variables
            self.remove_peer_from_agent_status_attrs(peer_id)

            # Updating buffered stream index
            if peer_id in self.last_buffered_peer_id_to_info:
                del self.last_buffered_peer_id_to_info[peer_id]  # Only if present

            # Clearing pending interactions
            await self.im.remove_interactions_of_agent(peer_id)

            log.misc(f"Successfully removed agent with peer ID {peer_id}")

    def remove_all_agents(self):
        """Remove all known agents."""

        # Clearing all agents
        self.all_agents = {}
        self.public_agents = {}
        self.world_masters = {}
        self.world_agents = {}
        self.human_agents = {}
        self.artificial_agents = {}

        # Clearing the list of processor-output-compatible-streams
        if self.compat_in_streams is not None and self.proc_inputs is not None:
            self.compat_in_streams = [set() for _ in range(len(self.proc_inputs))]
        if self.compat_out_streams is not None and self.proc_outputs is not None:
            self.compat_out_streams = [set() for _ in range(len(self.proc_outputs))]

        # Clearing the list of known streams (not our own streams!)
        self.remove_all_streams(owned_too=False)
        log.misc(f"Successfully removed all agents")

    def add_behav_wildcard(self, wildcard_from: str, wildcard_to: object):
        """Adds a wildcard mapping for the agent's behavior state machine, use when joining worlds.

        Args:
            wildcard_from: The string to be used as a wildcard.
            wildcard_to: The object to replace the wildcard.
        """
        self.behav_wildcards[wildcard_from] = wildcard_to

    def add_stream(self, stream: Stream, owned: bool = True, net_hash: str | None = None) -> dict[str, Stream]:
        """Add a new stream to the set of known streams.

        Args:
            stream: The Stream object to add.
            owned: If True, the streams are considered owned by this agent.
            net_hash: Optional network hash for the streams. If None, it will be generated.

        Returns:
            A dictionary containing the added stream and the possibly already present streams belonging to the same
            group (stream name -> stream object).
        """

        # Stream net hash
        if net_hash is None:
            public_peer_id, private_peer_id = self.get_peer_ids()
            peer_id = public_peer_id if stream.is_public() else private_peer_id
            net_hash = stream.net_hash(peer_id)
        user_hash = stream.user_hash(stream.peer_id_from_net_hash(net_hash))

        # It is fine to overwrite an existing stream, but the net hash cannot change
        if net_hash not in self.known_streams and user_hash in self.known_streams_by_user_hash:
            raise GenException(f"Naming clash in streams. "
                               f"Net hash {net_hash} is not known (stream name: {stream.props.get_name()}), "
                               f"but user hash {user_hash} is already existing. "
                               f"Are you using the same name for two different streams? (not allowed)")

        # Adding the new stream
        if net_hash not in self.known_streams:
            self.known_streams[net_hash] = {}
        else:
            for _stream in self.known_streams[net_hash].values():
                public = _stream.get_props().is_public()
                pubsub = _stream.get_props().is_pubsub()
                if public and not stream.get_props().is_public():
                    log.error(f"Cannot add a stream to a group with different properties (public): "
                              f"hash: {net_hash}, name: {stream.get_props().get_name()}, "
                              f"public: {stream.get_props().is_public()}")
                    return {}
                if pubsub and not stream.get_props().is_pubsub():
                    log.error(f"Cannot add a stream to a group with different properties (pubsub): "
                              f"hash: {net_hash}, name: {stream.get_props().get_name()}, "
                              f"public: {stream.get_props().is_public()}")
                    return {}
                break
        self.known_streams[net_hash][stream.get_props().get_name()] = stream  # Overwrite if existing
        self.known_streams_by_user_hash[user_hash] = stream  # Overwrite if existing

        if owned:
            is_proc_outputs_stream = False
            is_proc_inputs_stream = False

            # Adding an 'owned' processor output stream (i.e., the stream coming from OUR OWN processor)
            if self.proc_outputs is not None:
                proc_outputs_name_and_group = set()
                for props in self.proc_outputs:
                    proc_outputs_name_and_group.add((props.get_name(), props.get_group()))
                if (stream.get_props().get_name(), stream.get_props().get_group()) in proc_outputs_name_and_group:
                    if net_hash not in self.proc_streams:
                        self.proc_streams[net_hash] = {}
                    self.proc_streams[net_hash][stream.get_props().get_name()] = stream
                    self.proc_streams_by_user_hash[user_hash] = stream
                    if stream.is_public():
                        self.__proc_streams_by_user_hash_pub[user_hash] = stream
                    else:
                        self.__proc_streams_by_user_hash_prv[user_hash] = stream
                    is_proc_outputs_stream = True

            # Adding an 'owned' processor input stream (i.e., the stream entering OUR OWN processor)
            if self.proc_inputs is not None and not is_proc_outputs_stream:
                proc_inputs_name_and_group = set()
                for props in self.proc_inputs:
                    proc_inputs_name_and_group.add((props.get_name(), props.get_group()))
                if (stream.get_props().get_name(), stream.get_props().get_group()) in proc_inputs_name_and_group:
                    if net_hash not in self.proc_in_streams:
                        self.proc_in_streams[net_hash] = {}
                    self.proc_in_streams[net_hash][stream.get_props().get_name()] = stream
                    self.proc_in_streams_by_user_hash[user_hash] = stream
                    if stream.is_public():
                        self.__proc_in_streams_by_user_hash_pub[user_hash] = stream
                    else:
                        self.__proc_in_streams_by_user_hash_prv[user_hash] = stream
                    is_proc_inputs_stream = True

            if net_hash not in self.owned_streams:
                self.owned_streams[net_hash] = {}
            self.owned_streams[net_hash][stream.get_props().get_name()] = stream
            self.owned_streams_by_user_hash[user_hash] = stream

            if not is_proc_outputs_stream and not is_proc_inputs_stream:
                if net_hash not in self.env_streams:
                    self.env_streams[net_hash] = {}
                self.env_streams[net_hash][stream.get_props().get_name()] = stream
                self.env_streams_by_user_hash[user_hash] = stream
                if stream.is_public():
                    self.__env_streams_by_user_hash_pub[user_hash] = stream
                else:
                    self.__env_streams_by_user_hash_prv[user_hash] = stream

        # If needed, merging descriptor labels (attribute labels) and sharing them with all streams
        if self.merge_flat_stream_labels:
            self.merge_flat_data_stream_props()

        return self.known_streams[net_hash]

    def add_streams(self, streams: list[Stream], owned: bool = True, net_hash: str | None = None) \
            -> list[dict[str, Stream]]:
        """Add a list of new streams to this environment.

        Args:
            streams: A list of Stream objects to add.
            owned: If True, the streams are considered owned by this agent.
            net_hash: Optional network hash for the streams. If None, it will be generated for each.

        Returns:
            A list of dictionaries (it could be empty in case of issues), where each dictionary is what
            is returned by add_stream().
        """

        # Adding the new stream
        ret = []
        for stream in streams:
            stream_dict = self.add_stream(stream, owned, net_hash)
            if len(stream_dict) == 0:
                return []
            ret.append(stream_dict)
        return ret

    async def remove_streams(self, peer_id: str, name: str | None = None, owned_too: bool = False):
        """Remove a known stream (async).

        Args:
            peer_id: The hash of each stream included the peer ID of the owner, so this is the peer ID associated with
                the stream(s) to remove.
            name: The optional name of the stream to remove. If None, all streams with this peer_id are removed.
            owned_too: If True, also removes streams from the owned stream dict (so also environmental and processor).
        """

        # Identifying what to remove
        to_remove = []
        for net_hash in self.known_streams.keys():
            if DataProps.peer_id_from_net_hash(net_hash) == peer_id:
                for _name, _stream in self.known_streams[net_hash].items():
                    if name is None or name == _name:
                        to_remove.append((net_hash, _name))

        # Removing
        for (net_hash, name) in to_remove:
            if not owned_too and net_hash in self.owned_streams:
                continue

            group = DataProps.name_or_group_from_net_hash(net_hash)
            user_hash = DataProps.user_hash_from_net_hash(net_hash, name if name is not None else group)

            del self.known_streams[net_hash][name]
            if len(self.known_streams[net_hash]) == 0:
                del self.known_streams[net_hash]
            if user_hash in self.known_streams_by_user_hash:
                del self.known_streams_by_user_hash[user_hash]

            # Unsubscribing to pubsub
            if DataProps.is_pubsub_from_net_hash(net_hash):
                if peer_id != "<private_peer_id>" and peer_id != "<public_peer_id>":
                    if not (await self._node_conn.unsubscribe(peer_id, channel=net_hash)):
                        # log.error(f"Failed in unsubscribing from pubsub, peer_id: {peer_id}, channel: {net_hash}")
                        pass
                    else:
                        log.misc(f"Successfully unsubscribed from pubsub, peer_id: {peer_id}, channel: {net_hash}")

            # Removing all the owned streams (environment and processor streams are of course "owned")
            if net_hash in self.owned_streams:
                if name in self.owned_streams[net_hash]:
                    del self.owned_streams[net_hash][name]
                    if len(self.owned_streams[net_hash]) == 0:
                        del self.owned_streams[net_hash]
            if user_hash in self.owned_streams_by_user_hash:
                del self.owned_streams_by_user_hash[user_hash]
            if net_hash in self.env_streams:
                if name in self.env_streams[net_hash]:
                    del self.env_streams[net_hash][name]
                    if len(self.env_streams[net_hash]) == 0:
                        del self.env_streams[net_hash]
            if user_hash in self.env_streams_by_user_hash:
                del self.env_streams_by_user_hash[user_hash]
            if net_hash in self.proc_streams:
                if name in self.proc_streams[net_hash]:
                    del self.proc_streams[net_hash][name]
                    if len(self.proc_streams[net_hash]) == 0:
                        del self.proc_streams[net_hash]
            if user_hash in self.proc_streams_by_user_hash:
                del self.proc_streams_by_user_hash[user_hash]
            log.misc(f"Successfully removed known stream with network hash {net_hash}, stream name: {name}")

    def remove_all_streams(self, owned_too: bool = False):
        """Remove all not-owned streams.

        Args:
            owned_too: If True, also removes the owned streams of this agent (so also environmental and processor ones).
        """
        if not owned_too:
            self.known_streams = {k: v for k, v in self.owned_streams.items()}
            self.known_streams_by_user_hash = {k: v for k, v in self.known_streams_by_user_hash.items()}
        else:
            self.known_streams = {}
            self.owned_streams = {}
            self.env_streams = {}
            self.proc_streams = {}
            self.proc_in_streams = {}
            self.known_streams_by_user_hash = {}
            self.owned_streams_by_user_hash = {}
            self.env_streams_by_user_hash = {}
            self.proc_streams_by_user_hash = {}
            self.proc_in_streams_by_user_hash = {}
        log.misc(f"Successfully removed all streams!")

    def get_stream(self, name_or_group: str, peer_id: str | None = None, data_type: str | None = None) -> Stream | None:
        if peer_id is None:
            peer_id = self.get_peer_id()
        user_hash = DataProps.build_user_hash(peer_id, name_or_group)
        if user_hash in self.known_streams_by_user_hash:
            return self.known_streams_by_user_hash[user_hash]
        else:
            streams = self.get_streams(group_name=name_or_group, peer_id=peer_id)
            if streams is not None and len(streams) > 0:
                if data_type is not None:
                    for stream in streams:
                        if stream.props.data_type == data_type:
                            return stream
                    return None
                elif len(streams) == 1:
                    return streams[0]
                else:
                    return None  # Ambiguous case
            else:
                return None

    def get_streams(self, group_name: str, peer_id: str | None = None) -> list[Stream] | None:
        if peer_id is None:
            peer_id = self.get_peer_id()
        net_hash_ps = DataProps.build_net_hash(peer_id, pubsub=True, name_or_group=group_name)
        if net_hash_ps in self.known_streams:
            return list(self.known_streams[net_hash_ps].values())
        else:
            net_hash_dm = DataProps.build_net_hash(peer_id, pubsub=False, name_or_group=group_name)
            if net_hash_dm in self.known_streams:
                return list(self.known_streams[net_hash_dm].values())
            else:
                return None

    def find_streams(self, peer_id: str, name_or_group: str | None = None, discard_owned: bool = False) \
            -> dict[str, dict[str, Stream]]:
        """Find streams associated with a given peer ID and optionally by name or group.

        Args:
            peer_id: The peer ID of the (owner of the) streams to find.
            name_or_group: Optional name or group of the streams to find.
            discard_owned: If True, the owned streams are not searched (default False).

        Returns:
            A dictionary where keys are network hashes and values are dictionaries of streams
            (stream name to Stream object) matching the criteria.
        """
        ret = {}
        for net_hash, streams_dict in self.known_streams.items():
            if discard_owned and net_hash in self.owned_streams:
                continue
            _peer_id = Stream.peer_id_from_net_hash(net_hash)
            _name_or_group = Stream.name_or_group_from_net_hash(net_hash)
            if peer_id == _peer_id:
                if name_or_group is None or name_or_group == _name_or_group:
                    ret[net_hash] = streams_dict
                else:
                    for _name, _stream in streams_dict.items():
                        if name_or_group == _name:
                            if net_hash not in ret:
                                ret[net_hash] = {}
                            ret[net_hash][name_or_group] = _stream
        return ret

    def get_last_streamed_data(self, agent_name: str):
        """Get the last data samples from the processor streams of a given agent.

        Args:
            agent_name: The name of the agent.

        Returns:
            A list of data samples taken from all the known streams associated to the provided agent.
        """
        data_list = []
        for peer_id, profile in self.all_agents.items():
            if profile.get_static_profile()['node_name'] == agent_name:
                net_hash_to_stream_dict = self.find_streams(peer_id, name_or_group="processor")
                for net_hash, streams_dict in net_hash_to_stream_dict.items():
                    for stream_name, stream_obj in streams_dict.items():
                        data_list.append(stream_obj.get())
        return data_list

    def merge_flat_data_stream_props(self):
        """Merge the labels of the descriptor components, across all streams, sharing them."""

        # Set of pivot labels
        superset_labels = []

        # Checking the whole list of streams, but considering only the ones with generic data, flat, and labels
        considered_streams = []

        for stream_dict in self.owned_streams.values():
            for stream in stream_dict.values():

                # Skipping not flat, or not generic, or unlabeled streams
                if not stream.props.is_flat_tensor_with_labels():
                    continue

                # Saving list of considered streams
                considered_streams.append(stream)

                # Adding the current stream-labels to the pivot labels
                for label in stream.props.tensor_labels:
                    if label not in superset_labels:
                        superset_labels.append(label)

        # Telling each stream in which positions their labels fall, given the pivot labels
        for stream in considered_streams:

            # In the case of BufferedStream, we have to update the data buffer by clearing previously applied
            # adaptation first (I know it looks similar to what is done below, but we must clear first!)
            if isinstance(stream, BufferedStream):
                for uuid in stream.data_buffer_by_uuid.keys():
                    for i, (data, data_tag) in enumerate(stream.data_buffer_by_uuid[uuid]):
                        stream.data_buffer_by_uuid[uuid][i] = (stream.props.clear_label_adaptation(data),
                                                               data_tag)

            # Updating labels
            stream.props.tensor_labels.interleave_with(superset_labels)

            # In the case of BufferedStream, we have to update the data buffer with the new labels
            if isinstance(stream, BufferedStream):
                for uuid in stream.data_buffer_by_uuid.keys():
                    for i, (data, data_tag) in enumerate(stream.data_buffer_by_uuid[uuid]):
                        stream.data_buffer_by_uuid[uuid][i] = (stream.props.adapt_tensor_to_tensor_labels(data),
                                                               data_tag)

    def user_stream_hash_to_net_hash(self, user_stream_hash: str) -> str | None:
        """Converts a user-defined stream hash (peer_id:name_or_group) to a network hash
        (peer_id::dm:... or peer_id::ps:name_or_group) by searching the known hashes in the known streams.

        Args:
            user_stream_hash: The user-defined stream hash string (peer_id:name).

        Returns:
            The corresponding network hash string (peer_id::dm:... or peer_id::ps:name_or_group), or None if not found.
        """
        if user_stream_hash is None:
            return None
        if "::" in user_stream_hash:
            return user_stream_hash  # It was already fine
        components = user_stream_hash.split(":")
        peer_id = components[0]
        name = components[-1]
        for net_hash, stream_dict in self.known_streams.items():
            _peer_id = Stream.peer_id_from_net_hash(net_hash)
            _name_or_group = Stream.name_or_group_from_net_hash(net_hash)
            if _peer_id == peer_id and _name_or_group == name:
                return net_hash
            for _name in stream_dict.keys():
                if _peer_id == peer_id and _name == name:
                    return net_hash
        return None

    def create_proc_input_streams(self, buffered: bool = False):
        """Creates the processor input streams based on the `proc_inputs` defined for the agent.

        Args:
            buffered: If True, the created streams will be of type BufferedStream.
        """

        # Adding input streams (grouped together), passing the node clock
        if self.proc_inputs is not None:
            for i, procs in enumerate(self.proc_inputs):
                procs.set_group("processor_in")  # Adding default group info, forced, do not change this!

                # Creating the streams
                for props in procs.props:
                    if not buffered:
                        stream = Stream(props=props.clone())
                    else:
                        stream = BufferedStream(props=props.clone())

                    self.add_stream(stream, owned=True)

                    public_peer_id, private_peer_id = self.get_peer_ids()
                    peer_id = public_peer_id if stream.is_public() else private_peer_id
                    net_hash = stream.net_hash(peer_id)
                    if stream.is_public():
                        self.proc_in_net_hash['public'] = net_hash
                    else:
                        self.proc_in_net_hash['private'] = net_hash

                    # forcing the input stream to be compatible with proc inputs
                    self.compat_in_streams[i].add((net_hash, props.get_name()))

    def create_proc_output_streams(self, buffered: bool = False):
        """Creates the processor output streams based on the `proc_outputs` defined for the agent.

        Args:
            buffered: If True, the created streams will be of type BufferedStream.
        """

        # Adding generated streams (grouped together), passing the node clock
        if self.proc_outputs is not None:
            for i, procs in enumerate(self.proc_outputs):
                procs.set_group("processor")  # Adding default group info, forced, do not change this!

                # Creating the streams
                for props in procs.props:
                    if not buffered:
                        stream = Stream(props=props.clone())
                    else:
                        stream = BufferedStream(props=props.clone())

                    self.add_stream(stream, owned=True)

                    public_peer_id, private_peer_id = self.get_peer_ids()
                    peer_id = public_peer_id if stream.is_public() else private_peer_id
                    net_hash = stream.net_hash(peer_id)
                    if stream.is_public():
                        self.proc_net_hash['public'] = net_hash
                    else:
                        self.proc_net_hash['private'] = net_hash

    async def add_compatible_streams(self, peer_id: str,
                                     streams_in_profile: list[dict], buffered: bool = False,
                                     add_all: bool = False, public: bool = True, skip_pubsub: bool = False) -> bool:
        """Add to the list of processor-compatible-streams those streams provided as arguments that are actually
        found to be compatible with the processor (if they are pubsub, it also subscribes to them) (async).

        Args:
            peer_id: The peer ID of the agent providing the streams.
            streams_in_profile: A list of dictionaries (DataProps to dict) representing the streams in peer's profile.
            buffered: If True, the added streams will be of type BufferedStream.
            add_all: If True, all streams from the profile are added, regardless of processor compatibility.
            public: Consider public streams only (or private streams only).
            skip_pubsub: Skip pubsub streams.

        Returns:
            True if compatible streams were successfully added and subscribed to, False otherwise.
        """
        added_streams = []

        if add_all:

            # This is the case in which we add all streams, storing all pairs (DataProps, net_hash)
            for j in streams_in_profile:
                jj = DataProps.from_dict(j)
                if (public == jj.is_public()
                        and (not jj.is_pubsub() or not skip_pubsub)):
                    net_hash = jj.net_hash(peer_id)
                    added_streams.append((jj, net_hash))
        else:

            # This is the case in which a processor is present, hence storing pairs (DataProps, net_hash)
            # of the found compatible streams
            added_net_hash_to_prop_name = {}

            # Find streams that are compatible with our 'proc_inputs'
            for i, in_proc in enumerate(self.proc_inputs):
                for j in streams_in_profile:
                    jj = DataProps.from_dict(j)
                    if (public == jj.is_public() and in_proc.is_compatible(jj)
                            and (not jj.is_pubsub() or not skip_pubsub)):
                        net_hash = jj.net_hash(peer_id)

                        if net_hash not in added_net_hash_to_prop_name:
                            added_net_hash_to_prop_name[net_hash] = set()
                        if jj.name not in added_net_hash_to_prop_name[net_hash]:
                            added_net_hash_to_prop_name[net_hash].add(jj.name)
                            added_streams.append((jj, net_hash))

                        # Saving the position in the proc_input list
                        self.compat_in_streams[i].add((net_hash, jj.get_name()))

            # Find streams that are compatible with our 'proc_outputs'
            has_cross_entropy = []
            if 'losses' in self.proc_opts:
                for i in range(0, len(self.proc_outputs)):
                    if self.proc_opts['losses'][i] is not None and \
                            (self.proc_opts['losses'][i] == torch.nn.functional.cross_entropy or
                             isinstance(self.proc_opts['losses'][i], torch.nn.CrossEntropyLoss) or
                             "cross_entropy" in self.proc_opts['losses'][i].__name__):
                        has_cross_entropy.append(True)
                    else:
                        has_cross_entropy.append(False)

            for i, out_proc in enumerate(self.proc_outputs):
                for j in streams_in_profile:
                    jj = DataProps.from_dict(j)
                    if ((public == jj.is_public() and
                            (out_proc.is_compatible(jj) or (jj.is_tensor_target_id() and has_cross_entropy[i])))
                            and (not jj.is_pubsub() or not skip_pubsub)):
                        net_hash = jj.net_hash(peer_id)

                        if net_hash not in added_net_hash_to_prop_name:
                            added_net_hash_to_prop_name[net_hash] = set()
                        if jj.name not in added_net_hash_to_prop_name[net_hash]:
                            added_net_hash_to_prop_name[net_hash].add(jj.name)
                            added_streams.append((jj, net_hash))

                        # Saving the position in the proc_output list
                        self.compat_out_streams[i].add((net_hash, jj.get_name()))

        net_hashes_to_subscribe = set()

        # For each compatible stream found...
        for (props, net_hash) in added_streams:

            # Check if it is a new stream or a data stream to add to an already known stream
            already_known_stream = net_hash in self.known_streams

            # Creating the stream object
            if not buffered:
                stream = Stream(props=props.clone())
            else:
                stream = BufferedStream(props=props.clone())

            # Add the data stream to the list of known streams
            # if the stream already exists it will be overwritten (which is fine in case of changes)
            self.add_stream(stream, owned=False, net_hash=net_hash)

            # If the stream is over PubSub, and we are not already subscribed, we will subscribe
            if props.is_pubsub() and not already_known_stream:
                net_hashes_to_subscribe.add(net_hash)

        # Opening PubSubs
        for net_hash in net_hashes_to_subscribe:
            log.misc(f"Opening channel for the not-owned but processor-compatible stream {net_hash}")
            if not (await self._node_conn.subscribe(peer_id, channel=net_hash)):
                log.error(f"Error subscribing to {net_hash}")
                return False

        return True

    async def subscribe_to_pubsub_owned_streams(self) -> bool:
        """Subscribes to all owned streams that are marked as PubSub (async).

        Returns:
            True if all subscriptions were successful, False otherwise.
        """

        # Opening channels for all the (groups of) owned streams (generated and not)
        for net_hash in self.owned_streams.keys():
            is_pubsub = Stream.is_pubsub_from_net_hash(net_hash)

            if is_pubsub:
                log.misc(f"Opening channel for the owned stream {net_hash}")
                peer_id = Stream.peer_id_from_net_hash(net_hash)  # Guessing peer ID from the net hash

                if not (await self._node_conn.subscribe(peer_id, channel=net_hash)):
                    log.error(f"Cannot open a channel for owned stream hash {net_hash}")
                    return False
        return True

    def update_streams_in_profile(self):
        """Updates the agent's profile with information about its owned (environmental and processor) streams."""

        # Filling the information about the streams that can be generated and handled
        dynamic_profile = self._node_profile.get_dynamic_profile()
        if hasattr(self, 'proc_outputs') and hasattr(self, 'proc_inputs'):
            dynamic_profile['proc_outputs'] = \
                [dct for d in self.proc_outputs for dct in d.to_list_of_dicts()]  # List of dict of DataProp
            dynamic_profile['proc_inputs'] = \
                [dct for d in self.proc_inputs for dct in d.to_list_of_dicts()]  # List of dict of DataProp

        # Adding the list of locally-created ("environmental") streams to the profile
        list_of_props = []
        public_peer_id, private_peer_id = self.get_peer_ids()
        for net_hash, streams_dict in self.owned_streams.items():
            if net_hash not in self.proc_streams.keys() and net_hash not in self.proc_in_streams.keys():
                if (DataProps.peer_id_from_net_hash(net_hash) == public_peer_id or
                        DataProps.peer_id_from_net_hash(net_hash) == private_peer_id):
                    for stream in streams_dict.values():
                        list_of_props.append(stream.get_props().to_dict())  # DataProp
        if len(list_of_props) > 0:
            dynamic_profile['streams'] = list_of_props

    async def send_profile_to_all(self):
        """Sends the agent's profile to all known agents (async)."""

        agents = list(self.all_agents.keys())
        for peer_id in agents:
            log.misc(f"Sending profile to {peer_id}")
            if not (await self._node_conn.send(peer_id, channel_trail=None,
                                               content=self._node_profile.get_all_profile(),
                                               content_type=Msg.PROFILE)):
                log.error("Failed to send profile, removing (disconnecting) " + peer_id)
                await self.remove_agent(peer_id)

    async def behave(self):
        """Behave in the current environment, calling the state-machines of the public and private networks (async)."""

        if self.in_world():
            log.set_sub("prv")
            log.misc("Behaving (world)...")
            if self.behav is None:
                log.error("No behaviour specified")
            else:
                if self.behav_lone_wolf is not None:
                    self.behav_lone_wolf.enable(False)
                self.behav.enable(True)
                self.stdin.bind(self.__proc_in_streams_by_user_hash_prv)
                self.stdout.bind(self.__proc_streams_by_user_hash_prv)
                self.stdext.bind(self.__env_streams_by_user_hash_prv)
                await self.on_tick()
                await self.behav.act()
                self.behav.enable(False)

        log.set_sub("pub")
        log.misc("Behaving (public)...")
        if self.behav_lone_wolf is None:
            log.error("No behaviour specified")
        else:
            if self.behav is not None:
                self.behav.enable(False)
            self.behav_lone_wolf.enable(True)
            self.stdin.bind(self.__proc_in_streams_by_user_hash_pub, uuid=Custom.SYSTEM_INTERACTION_UUID)
            self.stdout.bind(self.__proc_streams_by_user_hash_pub, uuid=Custom.SYSTEM_INTERACTION_UUID)
            self.stdext.bind(self.__env_streams_by_user_hash_pub, uuid=Custom.SYSTEM_INTERACTION_UUID)
            await self.behav_lone_wolf.act()
            self.behav_lone_wolf.enable(False)

        log.set_sub("gen")

        # Clear expired interactions at every clock cycle and notify originators
        await self.im.complete_expired()

        # Also drain recently completed interactions for status notification
        completed = self.im.drain_completed()

        # Clear stream-expired data
        self.im.clear_expired_stream_data()

        for interaction in completed:
            if (interaction.requester is not None and interaction.send_status and
                    interaction.requester in self.all_agents):
                log.inter(f"Communicating completion of interaction {interaction.to_code_str(True)}")
                my_public_peer_id, my_private_peer_id = self.get_peer_ids()
                if interaction.requester in self.public_agents:
                    my_peer_id = my_public_peer_id
                else:
                    my_peer_id = my_private_peer_id
                ret = await self._node_conn.send(interaction.requester, channel_trail=None,
                                                 content=interaction.to_status_dict(status_generated_by=my_peer_id),
                                                 content_type=Msg.INTERACTION_STATUS)
                if not ret:
                    log.error(
                        f"Failed to send interaction status to peer {interaction.requester}, disconnecting it")
                    await self._node_purge_fcn(interaction.requester)

    def learn_behave(self, state: int, last_action: int, prev_state: int):
        """A placeholder method for behavioral learning, intended to be implemented by child classes.
        It receives state and action information to update a behavioral model.

        Args:
            state: The current state of the agent.
            last_action: The last action taken.
            prev_state: The previous state of the agent.

        Returns:
            None in the base implementation; subclasses are expected to return state-related feedback.
        """
        pass

    async def on_tick(self):
        pass

    def get_peer_id(self) -> str:
        """Return the current peer ID of the agent, depending on whether it is behaving in a world or not.

        Returns:
            The private peer ID if the agent is behaving in a world, the public peer ID otherwise.
        """
        public_peer_id, private_peer_id = self.get_peer_ids()
        if self.behaving_in_world():
            return private_peer_id
        else:
            return public_peer_id

    def get_peer_ids(self):
        """Retrieve the public and private peer IDs of the agent, from the underlying node's dynamic profile.

        Returns:
            A tuple containing the public peer ID and the private peer ID.
            If either ID is not available, a placeholder string is returned <public_peer_id>, <private_peer_id>.
        """
        public_peer_id = None
        private_peer_id = None
        if self._node_profile is not None:
            dynamic_profile = self._node_profile.get_dynamic_profile()
            public_peer_id = dynamic_profile['peer_id']  # Public
            private_peer_id = dynamic_profile['private_peer_id']  # Private
        public_peer_id = '<public_peer_id>' if public_peer_id is None else public_peer_id
        private_peer_id = '<private_peer_id>' if private_peer_id is None else private_peer_id
        return public_peer_id, private_peer_id

    def evaluate_profile(self, role: int, profile: NodeProfile) -> bool:
        """Evaluate if a given profile is valid for this agent based on its role. It helps in identifying and filtering
         out invalid or 'cheating' profiles.

        Args:
            role: The expected integer role (e.g., ROLE_PUBLIC, ROLE_WORLD_MASTER) for the profile.
            profile: The NodeProfile object to be evaluated.

        Returns:
            True if the profile is considered valid for the specified role, False otherwise.
        """

        # If the role in the profile is not the provided role, a profile-cheater was found
        if (profile.get_dynamic_profile()['connections']['role'] in self.ROLE_STR_TO_BITS and
                self.ROLE_STR_TO_BITS[profile.get_dynamic_profile()['connections']['role']] != role):
            log.user(f"Cheater found: "
                     f"{profile.get_dynamic_profile()['connections']['role']} != {self.ROLE_BITS_TO_STR[role]}")
            return False  # Cheater found

        # These are just examples: you are expected to reimplement this method in your custom agent file
        if (role & 1 == self.ROLE_PUBLIC and
                profile.get_dynamic_profile()['guessed_location'] == 'Some Dummy Location, Just An Example Here'):
            return False
        elif (role & 3 == self.ROLE_WORLD_MASTER and
              profile.get_dynamic_profile()['guessed_location'] == 'Some Other Location, Just Another Example Here'):
            return False
        else:
            return True

    def accept_new_role(self, role: int):
        """Set the agent's role and optionally load a default behavior (private/world behaviour).

        Args:
            role: The integer role to assign to the agent (e.g., ROLE_PUBLIC, ROLE_WORLD_MASTER).
        """
        base_role_str = self.ROLE_BITS_TO_STR[(role >> 2) << 2]
        full_role_str = self.ROLE_BITS_TO_STR[role]

        self._node_profile.get_dynamic_profile()['connections']['role'] = full_role_str

        default_behav = None  # A public role will not be found in the world map
        if self.world_profile is not None:
            base_role_to_behav = self.world_profile.get_dynamic_profile()['world_roles_fsm']
            if base_role_str in base_role_to_behav:
                default_behav = self.world_profile.get_dynamic_profile()['world_roles_fsm'][base_role_str]

        if default_behav is not None and len(default_behav) > 0:
            default_behav_hsm = HybridStateMachine(self)
            default_behav_hsm.load(default_behav)
            self.behav = HybridStateMachine(self, policy=self.policy_default)
            self.behav.include(default_behav_hsm, make_a_copy=True)
            self.behav.set_role(base_role_str)
            self.set_policy_filter(self.policy_filter, public=False)

    def is_human(self):
        """Check if the agent is marked as human in its node.

        Returns:
            True if the agent is in a human, False otherwise.
        """
        if self._node_profile is not None:
            return self._node_profile.get_static_profile()["node_type"] == self.HUMAN
        else:
            return False

    def in_world(self):
        """Check if the agent is currently operating within a 'world'.

        Returns:
            True if the agent is in a world, False otherwise.
        """
        if self._node_profile is not None:
            return self.ROLE_STR_TO_BITS[self._node_profile.get_dynamic_profile()['connections']['role']] & 1 == 1
        else:
            return False

    def behaving_in_world(self):
        """Checks if the agent's world-specific behavior state machine is currently active.

        Returns:
            True if the world behavior is active, False otherwise.
        """
        return self.behav.is_enabled()

    def get_stream_sample(self, net_hash: str,
                          sample_dict: dict[str, dict[str, torch.Tensor | None | int | str]]) \
            -> list[tuple[str, str]]:
        """Receive and process stream samples that were provided by another agent.

        Args:
            net_hash: The network hash identifying the source of the stream samples.
            sample_dict: A dictionary where keys are stream names and values are dictionaries
                         containing 'data', 'data_tag', and 'data_uuid' for each sample.

        Returns:
            The list of tuples (stream user hash, data UUID) for each added data sample.
        """

        # Let's be sure that the net hash is converted from the user's perspective to the one of the code here
        net_hash = DataProps.normalize_net_hash(net_hash)
        added_data = []

        log.misc(f"Got a stream sample from {net_hash}...")
        if sample_dict is None:
            log.error(f"Invalid sample (expected a dictionary, got {type(sample_dict)})")
            return added_data

        if net_hash in self.known_streams:
            for name, data_and_tag_and_uuid in sample_dict.items():
                if ('data' not in data_and_tag_and_uuid or
                        'data_tag' not in data_and_tag_and_uuid or
                        'data_uuid' not in data_and_tag_and_uuid):
                    log.error(f"Invalid sample in data stream named {name} (missing one or more keys)")
                    continue

                data, data_tag, data_uuid = (data_and_tag_and_uuid['data'],
                                             data_and_tag_and_uuid['data_tag'],
                                             data_and_tag_and_uuid['data_uuid'])

                # - data must be not None
                # - the stream name must be known
                # - if the UUID associated to our local stream is the same of the data, then we check tag order
                # - if the UUID associated to our local stream is the expected one, we don't check tag order
                skip = False
                reason = None
                has_data = False
                if not skip:
                    if data is None:
                        skip = True
                        reason = "Data is None"
                if not skip:
                    if net_hash not in self.known_streams:
                        skip = True
                        reason = f"The net hash {net_hash} is not a known stream hash"
                if not skip:
                    if name not in self.known_streams[net_hash]:
                        skip = True
                        reason = (f"The data stream named {name} is not present for net hash {net_hash} "
                                  f"(names: {list(self.known_streams[net_hash].keys())})")
                if not skip:
                    has_data = self.known_streams[net_hash][name].has_data(data_uuid)
                    if has_data and data_tag <= self.known_streams[net_hash][name].get_tag(data_uuid):
                        skip = True
                        reason = (f"The data tag {data_tag} is less or equal to the already present one "
                                  f"for UUID {data_uuid} "
                                  f"({self.known_streams[net_hash][name].get_tag(data_uuid)})")

                # If we sample can be accepted...
                if not skip:
                    log.misc(f"Accepted sample named {name}: tag={data_tag}, uuid={data_uuid}" +
                             (": it is a new UUID" if not has_data else ""))

                    # Saving the data sample on the known stream objects
                    self.known_streams[net_hash][name].set(data, data_tag, uuid=data_uuid)

                    # Data to return
                    added_data.append((DataProps.user_hash_from_net_hash(net_hash, name), data_uuid))

                    # Buffering data, if it was requested and if this sample comes from somebody's processor
                    if (self.buffer_generated_by_others != "none" and
                            DataProps.name_or_group_from_net_hash(net_hash) == "processor"):
                        log.debug(f"[get_stream_sample] Buffering others' processor generated data...")

                        # Getting the streams of the processor of the source agent
                        _processor_stream_dict = self.known_streams[net_hash]
                        _peer_id = DataProps.peer_id_from_net_hash(net_hash)

                        # Setting buffered stream counter
                        clear = False
                        if _peer_id in self.last_buffered_peer_id_to_info:
                            if self.buffer_generated_by_others == "one":
                                _buffered_uuid_to_id = self.last_buffered_peer_id_to_info[_peer_id]["uuid_to_id"]
                                if data_uuid not in _buffered_uuid_to_id:
                                    _id = next(iter(_buffered_uuid_to_id.values()))
                                    _buffered_uuid_to_id.clear()
                                    _buffered_uuid_to_id[data_uuid] = _id
                                    clear = True
                        else:
                            self.last_buffered_peer_id_to_info[_peer_id] = {"uuid_to_id": {}, "net_hash": None}
                        _buffered_uuid_to_id = self.last_buffered_peer_id_to_info[_peer_id]["uuid_to_id"]
                        if data_uuid not in _buffered_uuid_to_id:
                            _buffered_uuid_to_id[data_uuid] = sum(
                                len(v["uuid_to_id"]) for v in self.last_buffered_peer_id_to_info.values()) + 1
                        _buffered_id = _buffered_uuid_to_id[data_uuid]

                        # Building net hash to retrieve the buffered stream
                        _group = "buffered" + str(_buffered_id)
                        _name = name + "@" + _group
                        _net_hash = DataProps.build_net_hash(
                            _peer_id,
                            pubsub=False,
                            name_or_group=_group)

                        # If the buffered stream was not created before
                        if _net_hash not in self.known_streams:
                            log.debug(f"[get_stream_sample] Adding a new buffered stream to the list of known "
                                      f"streams, hash: {_net_hash}")
                            for stream_obj in _processor_stream_dict.values():
                                # Same properties of the stream of the processor of the source agent
                                props = stream_obj.get_props().clone()
                                props.set_group("buffered" + str(_buffered_id))
                                props.set_name(props.get_name() + "@" + props.get_group())

                                # Adding the newly created stream
                                self.add_stream(BufferedStream(props=props),
                                                owned=False,
                                                net_hash=_net_hash)

                            # Saving hash of the new buffered stream
                            self.last_buffered_peer_id_to_info[_peer_id]["net_hash"] = _net_hash
                        else:
                            if clear:
                                for stream_obj in self.known_streams[_net_hash].values():
                                    stream_obj.clear_buffer()

                        # Saving sample
                        self.known_streams[_net_hash][_name].set(data, data_tag, uuid=data_uuid)
                        if self.known_streams[_net_hash][_name].get_interaction(data_uuid) is None:
                            interaction = self.known_streams[net_hash][name].get_interaction(data_uuid)
                            if interaction is not None:
                                self.im.add_lazy_stream_to_interaction(
                                    DataProps.user_hash_from_net_hash(_net_hash, _name), interaction)

                # If we decided to skip this sample...
                else:
                    log.error(f"Skipping sample named {name} received in net hash {net_hash}, "
                              f"tag={data_tag}, uuid={data_uuid}: {reason}")

                    if net_hash not in self.known_streams:
                        log.debug(f"[get_stream_sample] "
                                  f"The net hash {net_hash} was not found in the set of known streams")
                    else:
                        if name not in self.known_streams[net_hash]:
                            log.debug(f"[get_stream_sample] The net hash was known, but the data stream "
                                      f"named {name} is not known")
                        else:
                            log.debug(f"[get_stream_sample] "
                                      f"data={self.known_streams[net_hash][name].props.to_text(data)}")
            return added_data

        # If this stream is not known at all...
        else:
            log.misc(f"Skipping sample from {net_hash} (data stream is unknown)")
            return added_data

    async def send_stream_samples(self):
        """Collect and send stream samples from all owned streams to appropriate recipients (async)."""

        # For each owned net hash...
        for net_hash, streams_dict in self.owned_streams.items():

            # Skipping our processor input
            if DataProps.name_or_group_from_net_hash(net_hash) == "processor_in":
                continue

            # Get samples from all the owned streams: taking notes about the recipient and of what to send (contents)
            interactions_by_uuid = {}
            recipients_by_uuid = {}
            contents_by_uuid = {}
            contents_data_by_uuid = {}

            # Preparing content to send
            empty_contents: dict[str, dict | None] = {name: None for name in streams_dict.keys()}
            empty_data = {name: None for name in streams_dict.keys()}
            valid_samples_count = {}

            # For each stream within this net hash...
            for name, stream in streams_dict.items():
                if stream.is_pubsub():
                    uuids = stream.get_data_uuids()  # We send by PubSub whenever there is some data
                else:
                    uuids = stream.get_interaction_uuids()  # We send direct messages only if there is an interaction

                # For each agent who triggered the interaction that generated this data (recipient)

                # Backward compatibility: set None as last UUID, since it is sometimes replaced by the last set
                # reference, and we prefer to give priority to the not-None cases
                if None in uuids:
                    uuids = [x for x in uuids if x is not None] + [None]

                for uuid in uuids:
                    interaction = stream.get_interaction(uuid)
                    recipient = None

                    # Invalid agent? Skip!
                    if not stream.is_pubsub():
                        recipient = self.im.get_recipients(interaction)
                        if len(recipient) == 0:
                            continue

                    # Skipping system-triggered interactions
                    if recipient == [Custom.SYSTEM_INTERACTION_LABEL]:
                        continue

                    # Get data
                    data = stream.get(requested_by="send_stream_samples", uuid=uuid)
                    data_tag = stream.get_tag(uuid=uuid)

                    if data is not None:
                        log.debug(f"[send_stream_samples] Found something in stream {stream.props.get_name()} "
                                  f"uuid={uuid}, data_tag={data_tag}, recipient={recipient}")

                        # Prepare data structures the first time we meet a new recipient
                        if uuid not in contents_by_uuid:
                            interactions_by_uuid[uuid] = interaction
                            recipients_by_uuid[uuid] = recipient
                            contents_by_uuid[uuid] = empty_contents.copy()
                            contents_data_by_uuid[uuid] = empty_data.copy()
                            valid_samples_count[uuid] = 0

                        # Pack into the prepared data structures
                        contents_by_uuid[uuid][name] = {'data': data,
                                                        'data_tag': data_tag,
                                                        'data_uuid': uuid}
                        contents_data_by_uuid[uuid][name] = data
                        valid_samples_count[uuid] += 1
                    else:
                        log.debug(
                            f"[send_stream_samples] None data in stream {stream.props.get_name()} uuid={uuid}")

            # Remove recipients of not-net-hash-full-of-contents messages
            uuid_to_remove = []
            for uuid, count in valid_samples_count.items():
                if count != len(streams_dict):
                    uuid_to_remove.append(uuid)
                    log.debug(f"[send_stream_samples] Cannot send data for {net_hash}, uuid {uuid}, "
                              f"since it is incomplete ({valid_samples_count[uuid]} vs {len(streams_dict)})")
            for uuid in uuid_to_remove:
                del interactions_by_uuid[uuid]
                del recipients_by_uuid[uuid]
                del contents_by_uuid[uuid]
                del contents_data_by_uuid[uuid]
                del valid_samples_count[uuid]

            # If direct message...
            if not Stream.is_pubsub_from_net_hash(net_hash):
                name_or_group = DataProps.name_or_group_from_net_hash(net_hash)

                for uuid, recipients in recipients_by_uuid.items():
                    content = contents_by_uuid[uuid]
                    content_data = contents_data_by_uuid[uuid]

                    for recipient in recipients:
                        log.debug(f"[send_stream_samples] "
                                  f"Sending samples of {net_hash} by direct message to {recipient}...")
                        for name in content.keys():
                            content[name]['data'] = self.hook_before_sending_sample(content_data[name],
                                                                                    content[name]['data_tag'],
                                                                                    net_hash, name, recipient)
                            log.debug(f"[send_stream_samples] "
                                      f"(data_tag={content[name]['data_tag']}, data_uuid={uuid}, "
                                      f"data is None?={content[name]['data'] is None})")

                        ret = await self._node_conn.send(recipient, channel_trail=name_or_group,
                                                         content_type=Msg.STREAM_SAMPLE,
                                                         content=content)

                        log.debug(f"[send_stream_samples] Sending returned: " + str(ret))

            # If pubsub...
            if Stream.is_pubsub_from_net_hash(net_hash):
                for uuid, recipients in recipients_by_uuid.items():
                    log.debug(f"[send_stream_samples] Sending stream samples of the whole {net_hash} by pubsub...")

                    content = contents_by_uuid[uuid]
                    content_data = contents_data_by_uuid[uuid]

                    for name in content.keys():
                        content[name]['data'] = self.hook_before_sending_sample(content_data[name],
                                                                                content[name]['data_tag'],
                                                                                net_hash, name, None)
                        log.debug(
                            f"[send_stream_samples] (data_tag={content[name]['data_tag']}, data_uuid={uuid}, "
                            f"data is None?={content[name]['data'] is None}")

                    peer_id = Stream.peer_id_from_net_hash(net_hash)  # Guessing agent peer ID from the net hash
                    ret = await self._node_conn.publish(peer_id, channel=net_hash,
                                                        content_type=Msg.STREAM_SAMPLE,
                                                        content=content)

                    log.debug(f"[send_stream_samples] Sending returned: " + str(ret))

    def disable_proc_input(self, public: bool):
        """Disable the processor input stream of this agent.

        Args:
            public: If True, disables the public input stream, otherwise the private one.
        """
        stream_dict = self.owned_streams[self.get_proc_input_net_hash(public=public)]
        for stream_obj in stream_dict.values():
            if stream_obj.is_public() == public:
                stream_obj.disable()

    def enable_proc_input(self, public: bool):
        """Enable the processor input stream of this agent.

        Args:
            public: If True, enables the public input stream, otherwise the private one.
        """
        stream_dict = self.owned_streams[self.get_proc_input_net_hash(public=public)]
        for stream_obj in stream_dict.values():
            if stream_obj.is_public() == public:
                stream_obj.enable()

    def set_proc_input(self, data: str | Image | torch.Tensor | None, public: bool = False,
                       uuid: str | None = None, data_type: str = "auto", data_tag: int = -1) -> bool:
        """Set the data in the processor input stream.

        Args:
            data: The data to set in the stream (text, image, or tensor).
            public: If True, targets the public input stream, otherwise the private one.
            uuid: Optional UUID to associate with the data sample.
            data_type: The type of data ("text", "img", "tensor", or "auto" to infer from the data).
            data_tag: The tag associated with the sample (default -1).

        Returns:
            True if the data was successfully set, False otherwise.
        """
        peer_id = self.get_peer_ids()[0] if public else self.get_peer_ids()[1]
        proc_in = self.find_streams(peer_id, "processor_in")
        if proc_in is None or len(proc_in) == 0:
            return False
        for net_hash, stream_dict in proc_in.items():
            if not DataProps.is_pubsub_from_net_hash(net_hash):
                for stream_name, stream_obj in stream_dict.items():
                    if stream_obj.props.is_public() == public:
                        if (data is None or
                                ((data_type == "text" or isinstance(data, str)) and stream_obj.props.is_text()) or
                                ((data_type == "img" or isinstance(data, Image)) and stream_obj.props.is_img()) or
                                ((data_type == "tensor" or isinstance(data, torch.Tensor))
                                 and stream_obj.props.is_tensor())):
                            stream_obj.set(data, uuid=uuid)  # This might fail if the stream is disabled
                            stream_obj.set_tag(data_tag, uuid=uuid)
                            return True
        return False

    def get_tag(self, net_hash: str, uuid: str | None = None) -> int:
        """Get the maximum data tag across all streams associated with the given network hash.

        Args:
            net_hash: The network hash identifying the stream group.
            uuid: Optional UUID of the data sample. If None, the latest tag is returned.

        Returns:
            The maximum data tag found, or -1 if the stream is unknown.
        """
        if net_hash in self.known_streams:
            data_tag = -1
            stream_dict = self.known_streams[net_hash]
            for stream_obj in stream_dict.values():
                t = stream_obj.get_tag(uuid)
                if t is not None:
                    data_tag = max(data_tag, t)
            return data_tag
        return -1

    def set_tag(self, net_hash: str, data_tag: int, uuid: str | None = None):
        """Set the data tag on all streams associated with the given network hash.

        Args:
            net_hash: The network hash identifying the stream group.
            data_tag: The tag value to set.
            uuid: Optional UUID of the data sample to tag. If None, the latest sample is tagged.
        """
        if net_hash in self.known_streams:
            stream_dict = self.known_streams[net_hash]
            for stream_obj in stream_dict.values():
                stream_obj.set_tag(data_tag, uuid=uuid)

    def get_current_interaction(self):
        return self.im.get_current()

    def force_action_step(self, step: int):
        """Override the current action step index used in the active behavior.

        Args:
            step: The action step index to force. If negative, the override is cleared.
        """
        self.overridden_action_step = step if step >= 0 else None

    def get_action_step(self):
        """Retrieve the current action step from the agent's private/world behavior.

        Returns:
            The current action step object from the HybridStateMachine's active action, or None if no action.
        """
        behav = self.behav if self.behav.is_enabled() else self.behav_lone_wolf
        return behav.get_action_step_idx() if self.overridden_action_step is None else self.overridden_action_step

    def is_last_action_step(self):
        """Check if the agent's current action (private/world behaviour) is on its last step.

        Returns:
            True if the current action was its last step, False otherwise. Returns None if there is no active action.
        """
        if self.im.current is not None:
            return self.im.current.was_last_step_done()
        else:
            return None

    def is_multi_steps_action(self):
        """Determines if the current action is a multistep action.

        Returns:
            True if the action is multistep, False otherwise.
        """
        if self.im.current is not None:
            return self.im.current.is_multi_steps()
        else:
            return False

    async def set_policy(self,
                         policy_method_name_or_policy_fcn: str | Callable[[list[Action]], [int, Interaction | None]],
                         public: bool = False) -> bool:
        """Sets the policy to be used in selecting what action to perform in the current state (async).

        Args:
            policy_method_name_or_policy_fcn: The name of a method of the Agent class that implements a policy function.
                It is a function that takes a list of `Action` objects that are candidates for execution, and returns
                the index of the selected action and an ActionRequest object with the action-requester details
                (requester, arguments, time, and UUID), or -1 and None if no action is selected.
                By design, every agent implements a basic policy function named "policy_default".
            public: If True, the policy will be applied to the public HSM, otherwise to the private/world one.

        Returns:
            True if the policy was successfully set, False otherwise.
        """
        if isinstance(policy_method_name_or_policy_fcn, str):
            policy_fcn = getattr(self, policy_method_name_or_policy_fcn, None)
            if not callable(policy_fcn):
                return False
            behav = self.behav if not public else self.behav_lone_wolf
            behav.set_policy(policy_fcn)
            return True
        elif callable(policy_method_name_or_policy_fcn):
            policy_fcn = policy_method_name_or_policy_fcn
            behav = self.behav if not public else self.behav_lone_wolf
            behav.set_policy(policy_fcn)
            return True
        return False

    def set_policy_filter(self,
                          filter_method_name_or_policy_fcn: str | Callable[
                              [int, Interaction | None, list[Action], dict], [int, Interaction | None]],
                          public: bool = False) -> bool:
        """Sets the policy filter function, which overrides the action selected by the policy in the current state.

        Args:
            filter_method_name_or_policy_fcn: The name of a method of the Agent class or a function that implements a
                policy filtering function, overriding what the policy decided.
                It is a function that takes what the policy decided, a list of `Action` objects that are candidates
                for execution, and a dictionary with customizable field (always including the "agent" key, with a ref
                to the current agent) and returns the index of the selected action and an ActionRequest object with the
                action-requester details (requester, arguments, time, and UUID), or -1 and None
                if no action is selected.
                By design, every agent comes with no filtering active.
            public: If True, the filter will be applied to the public HSM, otherwise to the private/world one.

        Returns:
            True if the filter was successfully set, False otherwise.
        """
        if isinstance(filter_method_name_or_policy_fcn, str):
            filter_fcn = getattr(self, filter_method_name_or_policy_fcn, None)
            if not callable(filter_fcn):
                return False
            if public:
                self.policy_filter_lone_wolf = filter_fcn
                self.behav_lone_wolf.set_policy_filter(self.policy_filter_lone_wolf, self.policy_filter_lone_wolf_opts)
                self.policy_filter_lone_wolf_opts['agent'] = self  # Forced (do it *after* set_policy_filter)
                self.policy_filter_lone_wolf_opts['public'] = True
            else:
                self.policy_filter = filter_fcn
                self.behav.set_policy_filter(self.policy_filter, self.policy_filter_opts)
                self.policy_filter_opts['agent'] = self  # Forced (do it *after* set_policy_filter)
                self.policy_filter_opts['public'] = False
            return True
        elif callable(filter_method_name_or_policy_fcn):
            if public:
                self.policy_filter_lone_wolf = filter_method_name_or_policy_fcn
                self.behav_lone_wolf.set_policy_filter(self.policy_filter_lone_wolf, self.policy_filter_lone_wolf_opts)
                self.policy_filter_lone_wolf_opts['agent'] = self  # Forced (do it *after* set_policy_filter)
                self.policy_filter_lone_wolf_opts['public'] = True
            else:
                self.policy_filter = filter_method_name_or_policy_fcn
                self.behav.set_policy_filter(self.policy_filter, self.policy_filter_opts)
                self.policy_filter_opts['agent'] = self  # Forced (do it *after* set_policy_filter)
                self.policy_filter_opts['public'] = False
            return True
        return False

    def policy_default(self, actions_list: list[Action]) -> tuple[int, Interaction | None]:
        """This is the default policy for selecting which action to execute from a list of feasible actions.
        It prioritizes actions that have been explicitly requested (i.e., have pending requests) on a first-come,
        first-served basis. If no requested actions are found, it then selects the first action in the list that is
        marked as `ready`.

        Args:
            actions_list: A list of `Action` objects that are candidates for execution.

        Returns:
            The index of the selected action and an ActionRequest object with the requester details (requester,
                arguments, time, and UUID), or -1 and None if no action is selected.
        """
        for i, action in enumerate(actions_list):
            _list_of_requests = action.get_list_of_interactions()
            if len(_list_of_requests) > 0:
                _selected_action_idx = i
                _selected_interaction = _list_of_requests.get_oldest_interaction()
                return _selected_action_idx, _selected_interaction
        for i, action in enumerate(actions_list):
            if action.is_ready(consider_interactions=False):
                _selected_action_idx = i
                _selected_interaction = None
                return _selected_action_idx, _selected_interaction
        _selected_action_idx = -1
        _selected_interaction = None
        return _selected_action_idx, _selected_interaction

    def hook_proc_tweak_inputs(self, inputs):
        """A callback method that saves the inputs to the processor right before execution.

        Args:
            inputs: The data inputs for the processor.

        Returns:
            The same inputs passed to the function.
        """
        self.proc_last_inputs = inputs
        return inputs

    def hook_proc_tweak_outputs(self, outputs):
        """A callback method that saves the outputs from the processor right after execution.

        Args:
            outputs: The data outputs from the processor.

        Returns:
            The same outputs passed to the function.
        """
        self.proc_last_outputs = outputs
        return outputs

    def hook_before_sending_sample(self, data, data_tag: int,
                                   net_hash: str, stream_name: str, recipient: str | None):
        """A callback method that handles the steam data right before sending it through the network.

        Args:
            data: The stream data sample.
            data_tag: The tag of the sample.
            stream_name: The name of the data stream.
            net_hash: The net hash of the whole stream.
            recipient: The (planned) recipient of this sample (or None in case of pubsub).

        Returns:
            The same data passed to the function.
        """
        return data

    def remove_peer_from_agent_status_attrs(self, peer_id: str):
        """Remove a peer ID from the status of the agent, assuming it to be the represented by attributes that start
        with '_'."""
        for attr_name in dir(self):
            if attr_name.startswith("_") and (not attr_name.startswith("__") and not attr_name.startswith("_Agent")
                                              and not attr_name.startswith("_WAgent")):
                try:
                    value = getattr(self, attr_name)
                    if isinstance(value, list):
                        setattr(self, attr_name, [v for v in value if v != peer_id])
                    elif isinstance(value, set):
                        value.discard(peer_id)
                    elif isinstance(value, dict):
                        if peer_id in value:
                            del value[peer_id]
                except AttributeError:
                    continue  # Skip read-only attributes

    def reset_agent_status_attrs(self):
        """Resets attributes that represent the status of the agent, assuming to be the ones that start with '_'."""
        for attr_name in dir(self):
            if attr_name.startswith("_") and (not attr_name.startswith("__") and not attr_name.startswith("_Agent")
                                              and not attr_name.startswith("_WAgent")):
                try:
                    value = getattr(self, attr_name)
                    if isinstance(value, list):
                        setattr(self, attr_name, [])
                    elif isinstance(value, set):
                        setattr(self, attr_name, set())
                    elif isinstance(value, dict):
                        setattr(self, attr_name, {})
                    elif isinstance(value, int):
                        setattr(self, attr_name, 0)
                    elif isinstance(value, float):
                        setattr(self, attr_name, 0.)
                    elif isinstance(value, bool):
                        setattr(self, attr_name, False)
                except AttributeError:
                    continue  # Skip read-only attributes

    def streams_to_str(self) -> str:
        """Return a formatted string listing all known streams, marking owned ones with an asterisk.

        Returns:
            A multi-line string representation of the known streams.
        """
        return "Streams:\n" + "\n".join([(("   @" if user_hash in self.owned_streams_by_user_hash else "   ") +
                                          DataProps.peer_id_from_user_hash(user_hash) + " => " +
                                          str(stream).replace("\n", "\n   "))
                                         for user_hash, stream in self.known_streams_by_user_hash.items()])

    def agent_state_dict(self) -> dict:
        """Returns a dictionary containing an instance of the agent's state that can be saved."""
        save_in_state = ['world_profile', ]
        return {k: getattr(self, k) for k in save_in_state}

    def save(self, where: str = "") -> bool:
        """Save the agent's state, including its processor and other attributes, to a specified location.

        Args:
            where: The directory path where the agent's state should be saved. Defaults to "".

        Returns:
            True upon successful saving.

        Raises:
            IOError: If there is an issue with file operations (e.g., directory creation, writing files).
            TypeError, ValueError, RuntimeError: For other potential issues during serialization or saving.
        """

        if where == '':
            if self._node_identity_dir is None or len(self._node_identity_dir) == 0:
                return False
            where = os.path.join(self._node_identity_dir, "agent_state")  # Default save path

        os.makedirs(where, exist_ok=True)

        # Saving the processor
        if self.proc is not None and self.proc_updated_since_last_save:
            pt_final = os.path.join(where, f"{self._node_name}.pt")
            pt_tmp = pt_final + ".tmp"
            try:
                checkpoint = {
                    'model_state_dict': self.proc.state_dict(),
                }

                # If your agent has an optimizer, save its state too
                if self.proc_opts.get('optimizer') is not None:
                    checkpoint['optimizer_state_dict'] = self.proc_opts['optimizer'].state_dict()

                torch.save(checkpoint, pt_tmp)
                os.replace(pt_tmp, pt_final)  # Atomic move
                self.proc_updated_since_last_save = False
            except Exception as e:
                if os.path.exists(pt_tmp):
                    os.remove(pt_tmp)
                log.error(f"Error saving processor: {e}")
                raise e

        # Save Agent State
        pkl_final = os.path.join(where, f"{self._node_name}.pkl")
        pkl_tmp = pkl_final + ".tmp"
        try:
            state = self.agent_state_dict()
            with open(pkl_tmp, "wb") as f:
                pickle.dump(state, f)
            os.replace(pkl_tmp, pkl_final)
        except Exception as e:
            log.error(f"Could not save " + ("agent" if not self.is_world else "world") + f": {e}")
            if os.path.exists(pkl_tmp):
                os.remove(pkl_tmp)
            raise e

        return True

    def load(self, where: str = "") -> bool:
        """Load the agent's state from a specified location.

        Args:
            where: The directory path from which the agent's state should be loaded. Defaults to "".

        Returns:
            True if loading succeeded.
        """

        if where == '':
            if self._node_identity_dir is None or len(self._node_identity_dir) == 0:
                return False
            where = os.path.join(self._node_identity_dir, "agent_state")  # Default save path

        # Check if directory exists
        if not os.path.exists(where):
            log.error("No state folder found for " + ("agent" if not self.is_world else "world") +
                      f" {self._node_name}.")
            return False

        # Check if the specific pickle file exists
        pkl_path = os.path.join(where, f"{self._node_name}.pkl")
        if not os.path.exists(pkl_path):
            log.error("No saved state found for " + ("agent" if not self.is_world else "world") +
                      f" {self._node_name}.")
            return False

        # Loading the agent state dictionary
        try:
            with open(pkl_path, "rb") as f:
                agent_state_dict = pickle.load(f)
        except Exception as e:
            raise Exception(f"Error loading pickle file at {pkl_path}: {e}")

        # Update self's attributes with the loaded object's attributes
        self.__dict__.update(agent_state_dict)

        # Check if we also need to load the processor state
        pt_path = os.path.join(where, f"{self._node_name}.pt")
        load_proc = self.proc is not None and os.path.exists(pt_path)
        if load_proc:
            try:
                checkpoint = torch.load(pt_path)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.proc.load_state_dict(checkpoint['model_state_dict'])

                    # Restore Optimizer to proc_opts
                    if 'optimizer_state_dict' in checkpoint and self.proc_opts.get('optimizer') is not None:
                        self.proc_opts['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])
                else:
                    self.proc.load_state_dict(checkpoint)
            except Exception as e:
                raise Exception(f"Error loading processor state: {e}")

        return True

    async def _send(self, interaction: Interaction | None = None,
                    action_name: str | None = None,
                    target: str | list[str] | None = None,
                    action_kwargs: dict | None = None,
                    streams: dict[str, str] | list[str] | None = None,
                    data_samples: list[str | Image | torch.Tensor] | dict[
                        str, str | Image | torch.Tensor] | None = None,
                    num_steps: int = -1,
                    max_time: float = -1.,
                    from_state: str | None = None,
                    to_state: str | None = None,
                    callback: str | None = None,
                    forced_uuid: str | None = "do_not_force",
                    id: str | None = "random",
                    copy_sys: bool = False) -> Interaction | None:
        """Send an interaction request to one or more target agents (async).

        Accepts either a pre-built :class:`Interaction` object *or* raw arguments from which one will be created.

        Args:
            interaction: A pre-built Interaction.
            action_name: The action name (str) if building one from raw args.
                Ignored when a pre-built Interaction is provided.
            target: Peer ID (or wildcard such as ``"<valid_cmp>"``) of the target
                agent. It can be a list of agents. Ignored when a pre-built Interaction is provided.
            action_kwargs: Action arguments dict.  Ignored when a pre-built Interaction
                is provided.
            streams: List of ``(stream_hash, num_samples)`` tuples (or just stream_hash if num_samples = 1).
                Ignored when a pre-built Interaction is provided. A lot of freedom here: stream_hash can be a net hash
                or a user hash. It is also fine to specify just the stream name or the stream
                group, then the user hash or the net hash will be automatically estimated.
            data_samples: List of actual data samples. Ignored when a pre-built Interaction is provided.
                It can also be a dictionary stream user hash -> data sample, to actually send the samples as if they
                were produced by a stream.
            num_steps: Number of steps the interaction spans (used when building from raw args).
                Ignored when a pre-built Interaction is provided.
            max_time: Maximum time for the interaction. Ignored when a pre-built Interaction is provided.
            from_state: Optional source state. Ignored when a pre-built Interaction provided.
            to_state: Optional destination state. Ignored when a pre-built Interaction provided.
            callback: Callback action.
            forced_uuid: A very custom option to force a specific UUID, that is by default a mixture of the id (below)
                and the target fields. To be used only in very special cases. Defaults to None, meaning "don't force".
            id: Optional ID of the interaction. Ignored when a pre-built Interaction is provided.
            copy_sys: True if the interaction must copy the last added stream data (with system UUID)
                to prepare initial data form the current interaction.

        Returns:
            The Interaction on success, ``None`` on failure.
        """
        # Build Interaction from raw args if needed
        if interaction is None:
            interaction = Interaction(action_name=action_name, action_kwargs=action_kwargs,
                                      streams=streams, data_samples=data_samples,
                                      num_steps=num_steps,
                                      requester=self.get_peer_id(), target=target,
                                      from_state=from_state, to_state=to_state,
                                      timeout=max_time, callback=callback, forced_uuid=forced_uuid, id=id)

        # Resolve target agent(s)
        if len(interaction.target) == 0:
            return None

        # Ensure requester is set
        if interaction.requester is None:
            interaction.requester = self.get_peer_id()

        # Registering
        if not self.im.register_sent(interaction):
            return None

        # Build the content for the interaction message
        content = interaction.to_dict()
        at_least_one_completed = False
        correctly_contacted = []
        for _peer_id in interaction.target:
            ret = asyncio.create_task(
                self._node_conn.send(_peer_id, channel_trail=None,
                                     content=content, content_type=Msg.INTERACTION, fire_and_forget=True)
            )
            if ret:
                correctly_contacted.append(_peer_id)
                at_least_one_completed = at_least_one_completed or ret
                log.debug(f"[send] Sent interaction {interaction.uuid} "
                          f"(action={interaction.action_name}) to {_peer_id}, result={ret}")

        if at_least_one_completed:

            # Updating targets with the list of actually contacted agents
            interaction.update_target(correctly_contacted)

            # If we asked to copy the last-added stream data by a system interaction,
            # then we copy it using the current interaction UUID
            if copy_sys:
                for stream in interaction.owned_streams.values():
                    last_stream_data = stream.get(uuid=Custom.SYSTEM_INTERACTION_UUID)
                    if last_stream_data is not None:
                        stream.set(last_stream_data, uuid=interaction.uuid)

            return interaction
        else:
            self.im.unregister(interaction)
            return None

    def __str__(self):
        """String representation of an agent.

        Returns:
            A formatted string describing the agent's current state and relationships.
        """
        s = ("[" + ("Agent" if not self.is_world else "World") + "]"
             + f" {self._node_name} (role: {self._node_profile.get_dynamic_profile()['connections']['role']})")
        if len(self.world_masters) > 0:
            s += "\n\t- known world masters:"
            for _s in self.world_masters.keys():
                s += "\n\t\t" + str(_s)
        if len(self.world_agents) > 0:
            s += "\n\t- known agents living in the same world (non-world-masters):"
            for _s in self.world_agents.keys():
                s += "\n\t\t" + str(_s)
        if len(self.public_agents) > 0:
            s += "\n\t- known lone wolves:"
            for _s in self.public_agents.keys():
                s += "\n\t\t" + str(_s)
        if len(self.known_streams) > 0:
            s += "\n\t- known_streams:"
            for _s in self.known_streams:
                s += "\n\t\t" + str(_s)
        s += "\n\t- behaviour (public):"
        s += "\n\t\t" + (str(self.behav_lone_wolf).replace("\n", "\n\t\t")
                         if self.behav_lone_wolf is not None else "none")
        s += "\n\t- behaviour (private):"
        s += "\n\t\t" + (str(self.behav).replace("\n", "\n\t\t") if self.behav is not None else "none")
        s += "\n\t- processor:"
        s += "\n\t\t" + (str(self.proc).replace("\n", "\n\t\t") if self.proc is not None else "none")
        return s

    async def __remove_all_world_related_agents(self):
        """Remove all world-related agents (masters and regular agents) from the agent's known lists (async)."""

        to_remove = list(self.world_masters.keys())
        for peer_id in to_remove:
            await self.remove_agent(peer_id)

        to_remove = list(self.world_agents.keys())
        for peer_id in to_remove:
            await self.remove_agent(peer_id)

    async def __remove_all_world_private_streams(self):
        """Remove all known streams that are flagged as not-public and are not owned by this agent (async)."""

        # Find what to remove
        to_remove = []
        for net_hash, stream_dict in self.known_streams.items():
            for name, stream_obj in stream_dict.items():
                if not stream_obj.get_props().is_public() and net_hash not in self.owned_streams:
                    to_remove.append((DataProps.peer_id_from_net_hash(net_hash), name))

        # Remove it
        for (peer_id, name) in to_remove:
            await self.remove_streams(peer_id, name)

    # ==================================================================================================================
    # BEGIN OF DEPRECATED METHODS
    # ==================================================================================================================
    @deprecated("Use the provided logger")
    def out(self, msg):
        """DEPRECATED"""
        log.misc(msg)

    @deprecated("Use the provided logger")
    def err(self, msg):
        """DEPRECATED"""
        log.error(msg)

    @deprecated("Use the provided logger")
    def deb(self, msg):
        """DEPRECATED"""
        log.debug(msg)

    @deprecated("Use the interaction-based design")
    def set_uuid(self, net_hash: str, uuid: int | None, expected: bool = False):
        """DEPRECATED"""
        if net_hash in self.known_streams:
            stream_dict = self.known_streams[net_hash]
            for stream_obj in stream_dict.values():
                stream_obj.set_uuid(uuid, expected=expected)

    @deprecated("Use hook_proc_tweak_inputs")
    def proc_callback_inputs(self, inputs):
        """DEPRECATED"""
        return self.hook_proc_tweak_inputs(inputs)

    @deprecated("Use hook_proc_tweak_outputs")
    def proc_callback_outputs(self, outputs):
        """DEPRECATED"""
        return self.hook_proc_tweak_outputs(outputs)

    @deprecated("Use the interaction-based methods")
    def generate(self, input_net_hashes: list[str] | None = None,
                 inputs: list[str | torch.Tensor | Image] | None = None,
                 first: bool = False, last: bool = False, ref_uuid: str | None = None) -> (
            tuple[tuple[torch.Tensor] | None, int]):
        """DEPRECATED: Generate new signals.

        Args:
            input_net_hashes: A list of network hashes to be considered as input streams (they will be sub-selected).
            inputs: A list of data to be directly provided as input to the processor (if not None, input_net_hashes is
                ignored).
            first: If True, indicates this is the first generation call in a sequence.
            last: If True, indicates this is the last generation call in a sequence.
            ref_uuid: An optional UUID to match against input stream UUIDs (it can be None).

        Returns:
            A tuple containing:
                - A tuple of torch.Tensor objects representing the generated output, or None if generation failed.
                - An integer representing a data tag or status.
        """

        # Preparing processor input
        if inputs is None:
            inputs = [None] * len(self.proc_inputs)
            matched = set()
            data_tag = None

            if input_net_hashes is None:
                input_net_hashes = []

            # Checking UUIDs and searching the provided input streams: we look to match them with the processor input
            for net_hash in input_net_hashes:
                if net_hash is None:
                    return None, - 1

                stream_dict = self.known_streams[net_hash]
                for stream_name, stream in stream_dict.items():

                    # Checking the UUID in our known streams, comparing it with the UUID provided as input:
                    # if they are not compatible, we don't generate at all
                    if ref_uuid is not None and ref_uuid not in stream.get_data_uuids():
                        log.debug(f"[generate] The UUIDs of stream {net_hash} ({list(stream.get_data_uuids())}) "
                                  f"does not include the one we were  "
                                  f"looking for ({ref_uuid}), skipping this stream")
                        continue

                    # Matching the currently checked input stream with one of the processor inputs
                    stream_sample = stream.get(requested_by="generate", uuid=ref_uuid)
                    for i in range(len(self.proc_inputs)):

                        # If the current input stream is compatible with the i-th input slot...
                        if (net_hash, stream_name) in self.compat_in_streams[i]:

                            # If the current input stream was already assigned to another input slot
                            # (different from "i") we skip the generation
                            if (net_hash, stream_name) in matched:
                                log.error("Cannot generate: ambiguous input streams provided "
                                          "(they can match multiple processor inputs)")
                                return None, -1

                            # Found a valid assignment: getting stream sample
                            log.debug(f"[generate] Setting the {i}-th network input to stream with "
                                      f"net_hash: {net_hash}, name: {stream_name}")
                            if stream_sample is None:
                                log.debug(f"[generate] Failed setting the {i}-th input, got a None sample "
                                          f"(uuid={ref_uuid})")
                            else:
                                log.debug(f"[generate] Going ahead setting the {i}-th input, got a valid sample")

                                # Found a valid assignment: associating it to the i-th input slot
                                try:
                                    inputs[i] = self.proc_inputs[i].check_and_preprocess(stream_sample,
                                                                                         device=self.proc.device)
                                except Exception as e:
                                    log.error(f"Error while checking and preprocessing the {i}-th input [{e}]")
                                    continue

                                log.debug(f"[generate] Finished setting the {i}-th input, preprocessing complete")

                                # Found a valid assignment: saving match
                                matched.add((net_hash, stream_name))

                                # If all the inputs share the same data tag, we will return it,
                                # otherwise we set it at -1 (meaning no tag)
                                if data_tag is None:
                                    data_tag = stream.get_tag(uuid=ref_uuid)
                                elif data_tag != stream.get_tag(uuid=ref_uuid):
                                    data_tag = -1

                                if stream.props.is_text():
                                    log.debug(f"[generate] Input {i} of the network: {stream_sample}")
                                break

            # Checking if we were able to match some data for each input slot of the network (processor)
            for i in range(len(self.proc_inputs)):
                if inputs[i] is None:
                    if self.proc_optional_inputs[i]["has_default"]:
                        inputs[i] = self.proc_optional_inputs[i]["default_value"]
                    else:
                        return None, -1

            # If we reach this point and data_tag is still None, it means that inputs were all taken from defaults
            if data_tag is None:
                return None, -1
        else:
            data_tag = -1

        if inputs is not None:
            input_shapes = []
            for x in inputs:
                if isinstance(x, torch.Tensor):
                    input_shapes.append(x.shape)
                else:
                    input_shapes.append("<non-tensor>")
            log.debug(f"[generate] Input shapes: {input_shapes}")
            log.debug(f"[generate] Input data tag: {data_tag}")

        # Calling processor (inference) passing the collected inputs
        if "proc_callback_inputs" in type(self).__dict__:
            inputs = self.proc_callback_inputs(inputs)  # Backward compatibility
        else:
            inputs = self.hook_proc_tweak_inputs(inputs)
        try:
            outputs = self.proc(*inputs, first=first, last=last)

            # Ensuring the output is a tuple, even if composed by a single tensor
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
        except Exception as e:
            log.error(f"Error while calling the processor [{e}]")
            outputs = (None,) * len(self.proc_outputs)
        outputs = self.hook_proc_tweak_outputs(outputs)

        # Saving
        self.last_ref_uuid = ref_uuid

        if outputs is not None:
            i = 0
            for net_hash, stream_dict in self.proc_streams.items():
                for stream in stream_dict.values():
                    if self.behaving_in_world() != stream.props.is_public():
                        if outputs[i] is not None:
                            if stream.props.is_tensor() or stream.props.is_text():
                                log.debug(f"[generate] outputs[{i}]: {str(stream.props.to_text(outputs[i]))}")
                            else:
                                log.debug(
                                    f"[generate] outputs[{i}]: not None, but it cannot be converted to text")
                        else:
                            log.debug(f"[generate] outputs[{i}]: None")
                        i += 1
            log.debug(f"[generate] Output shapes: "
                      f"{[x.shape if isinstance(x, torch.Tensor) else '<non-tensor>' for x in outputs]}")

        return outputs, data_tag

    @deprecated("Use the interaction-based methods")
    def learn_generate(self,
                       outputs: tuple[torch.Tensor],
                       targets_net_hashes: list[str] | None,
                       ref_uuid: str | None = None) -> tuple[list[float] | None, list[float] | None]:
        """DEPRECATED: Learn (i.e., update model params) by matching the given processor outputs with a set of targets.

        Args:
            outputs: A tuple of torch.Tensor representing the outputs generated by the agent's processor.
            targets_net_hashes: An optional list of network hashes identifying the streams
                                from which target data should be retrieved for learning.
                                If None, losses are evaluated without explicit targets.
            ref_uuid: UUID of the interaction associated to the learning procedure.

        Returns:
            A tuple containing:
            - A list of float values representing the individual loss values for each output.
              Returns None if targets are specified but cannot be found.
            - A list of integers representing the data tags of the given target streams (None if no targets were given).
        """

        # Cannot learn without optimizer and losses
        if (self.proc_opts['optimizer'] is None or self.proc_opts['losses'] is None or
                len(self.proc_opts['losses']) == 0):
            return None, None

        # Matching targets with the output slots of the processor
        at_least_one_target_found = False
        if targets_net_hashes is not None:
            targets = [None] * len(self.proc_outputs)
            matched = set()
            data_tags = [-1] * len(self.proc_outputs)

            # For each target stream group...
            for net_hash in targets_net_hashes:
                stream_dict = self.known_streams[net_hash]

                # For each stream of the current target group....
                for stream_name, stream in stream_dict.items():
                    stream_sample = None

                    # For each output slot of our processor... (index "i")
                    for i in range(len(self.proc_outputs)):

                        # Check if the i-th target was already assigned or if the i-th output is not a tensor
                        if targets[i] is not None or not isinstance(outputs[i], torch.Tensor):
                            continue

                        # If the target stream is compatible with the i-th output of the processor...
                        if (net_hash, stream_name) in self.compat_out_streams[i]:

                            # If the current target was already assigned to another output slot (different from "i)"
                            # we skip learning
                            if (net_hash, stream_name) in matched:
                                log.error("Cannot generate: ambiguous target streams provided "
                                          "(they can match multiple processor outputs)")
                                return None, None

                            # Found a valid assignment: getting stream sample
                            if stream_sample is None:
                                stream_sample = stream.get(requested_by="learn_generate", uuid=ref_uuid)
                                if stream_sample is None:
                                    return None, None

                            # Found a valid assignment: associating target to the i-th output slot
                            try:
                                targets[i] = self.proc_outputs[i].check_and_preprocess(stream_sample,
                                                                                       allow_class_ids=True,
                                                                                       targets=True,
                                                                                       device=self.proc.device)
                            except Exception as e:
                                log.error(f"Error while checking and preprocessing the {i}-th targets [{e}]")

                            # Found a valid assignment: saving match
                            matched.add((net_hash, stream_name))

                            # Saving tag
                            data_tags[i] = stream.get_tag(uuid=ref_uuid)

                            # Confirming
                            at_least_one_target_found = True

                            if stream.props.is_tensor():
                                log.debug("[generate] Target of the network: " +
                                          str(stream.props.to_text(targets[i])))
                            elif stream.props.is_text():
                                log.debug("[generate] Target of the network: " + stream_sample)
                            break

            # If no targets were matched, we skip learning
            if not at_least_one_target_found:
                log.error(f"Cannot learn: cannot find a valid target for any output positions of the processor")
                return None, None
        else:

            # If no targets were provided, it is expected to be the case of fully unsupervised learning
            data_tags = None
            targets = None

        # Retrieving custom elements from the option dictionary
        loss_functions: list = self.proc_opts['losses']
        optimizer: torch.optim.optimizer.Optimizer | None = self.proc_opts['optimizer']

        # Evaluating loss function(s), one for each processor output slot (they are set to 0. if no targets are there)
        if targets_net_hashes is not None:

            # Supervised or partly supervised learning
            loss_values = [loss_fcn(outputs[i], targets[i]) if targets[i] is not None else
                           torch.tensor(0., device=self.proc.device)
                           for i, loss_fcn in enumerate(loss_functions)]
            loss = torch.stack(loss_values).sum()  # Sum of losses
        else:

            # Unsupervised learning
            loss_values = [loss_fcn(outputs[i]) for i, loss_fcn in enumerate(loss_functions)]
            loss = torch.stack(loss_values).sum()  # Sum of losses

        # Learning step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # This is where parameters are actually updated, but the flag is set
        # upon success in do_learn which is the outer method calling this one
        # self.proc_updated_since_last_save = True

        # Teaching (for autoregressive models, expected to have attribute "y")
        if hasattr(self.proc, 'y'):
            self.proc.y = targets[0]

        # Returning a list of float values and the data tags of the targets
        return [loss_value.item() for loss_value in loss_values], data_tags
    # ==================================================================================================================
    # END OF DEPRECATED METHODS
    # ==================================================================================================================
