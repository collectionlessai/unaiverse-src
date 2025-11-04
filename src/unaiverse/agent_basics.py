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
import json
import torch
import types
import pickle
import importlib.resources
from PIL.Image import Image
from unaiverse.clock import Clock
from unaiverse.hsm import HybridStateMachine
from unaiverse.networking.p2p.messages import Msg
from unaiverse.dataprops import DataProps, Data4Proc
from unaiverse.networking.node.profile import NodeProfile
from unaiverse.streams import BufferedDataStream, DataStream
from unaiverse.networking.node.connpool import ConnectionPools
from unaiverse.modules.utils import AgentProcessorChecker, ModuleWrapper
from unaiverse.utils.misc import GenException, FileTracker, StatLoadedSaver


class AgentBasics:
    """This class contains those methods and properties that are about building the agent, known agents,
    known streams, etc., and no actions at all (see the class "Agent" for actions)."""

    DEBUG = True  # Turns on/off extra logging

    # Role bits (a.k.a. role int): default roles, shared by every possible agent
    ROLE_PUBLIC = 0 << 0  # 00000000 = 0 means "public"
    ROLE_WORLD_MASTER = (1 << 0) | (1 << 1)  # 00000011 = 3 means "world master" (the first bit means "about world")
    ROLE_WORLD_AGENT = (1 << 0) | (0 << 1)  # 00000001 = 2 means "world agent" (the first bit means "about world")
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

    # Stats (world perspective - i.e., as the world organizes them)
    WORLD_STATS_STATIC = {'graph'}
    WORLD_STATS_STATIC_BY_PEER = {'state', 'action', 'last_action'}
    WORLD_STATS_DYNAMIC = {'world_masters', 'world_agents', 'human_agents', 'artificial_agents'}
    WORLD_STATS_DYNAMIC_BY_PEER = {}  # modify this in-place to introduce custom stats

    # Stats (agent perspective - i.e., sent by every peer to the world)
    STATS_STATIC_SENT_BY_PEER = {'connected_peers'} | WORLD_STATS_STATIC_BY_PEER
    STATS_DYNAMIC_SENT_BY_PEER = WORLD_STATS_DYNAMIC_BY_PEER

    # The type associated to a human: it is not exploited at a node-level, only at agent level
    HUMAN = "human"  # Human agent

    def __init__(self,
                 proc: ModuleWrapper | torch.nn.Module | None,
                 proc_inputs: list[Data4Proc] | None = None,
                 proc_outputs: list[Data4Proc] | None = None,
                 proc_opts: dict | None = None,
                 behav: HybridStateMachine | None = None,
                 behav_lone_wolf: HybridStateMachine | str = "serve",
                 merge_flat_stream_labels: bool = False,
                 buffer_generated: bool = False,
                 buffer_generated_by_others: str = "none",
                 world_folder: str | None = None):
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
            buffer_generated_by_others: If set to "one" or "last", streams generated by other agents will be buffered
                ("one" per peer or "all"). If set to "none", no buffering will happen (default).
            world_folder: World only. Folder where the world data is (role files, represented by *.json behavior files).
        """

        # Agent-related features
        self.behav = behav  # HSM that describes the agent behavior in the private/world net
        self.behav_lone_wolf = behav_lone_wolf  # HSM that describes the agent behavior in the public net
        self.behav_wildcards = {}
        self.proc = proc
        self.proc_inputs = proc_inputs
        self.proc_outputs = proc_outputs
        self.proc_opts = proc_opts
        self.proc_last_inputs = None
        self.proc_last_outputs = None
        self.proc_optional_inputs = None
        self.merge_flat_stream_labels = merge_flat_stream_labels
        self.buffer_generated = buffer_generated
        self.buffer_generated_by_others = buffer_generated_by_others
        self.world_folder = world_folder

        if self.buffer_generated_by_others not in {"one", "all", "none"}:
            raise GenException("Param buffer_generated_by_others can be set to 'one', 'all', or 'none' only.")

        # Streams
        self.known_streams = {}  # All streams that are known to this agent
        self.owned_streams = {}  # The streams that are generated/offered by this agent
        self.env_streams = {}  # The owned streams that come from environmental sources (e.g., a camera)
        self.proc_streams = {}  # The owned streams that are generated by the agent's processor
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
        self.agent_actions = None
        self.role_to_behav = {}
        self.agent_badges: dict[str, list[dict]] = {}  # Peer_id -> collected badges for other agents
        self.role_changed_by_world: bool = False
        self.received_address_update: bool = False

        # Internal properties about the way streams are used
        self.last_buffered_peer_id_to_info = {}   # If buffering was turned on
        self.last_ref_uuid = None
        self.recipients = {}  # The peer IDs of the recipients of the next batch of direct messages

        # Stats
        self.stats = {}
        self.clear_stats()
        self.stats_loader_saver = None

        # Information inherited from the node that hosts this agent
        self._node_name = "unk"
        self._node_clock = None
        self._node_conn = None
        self._node_profile = None
        self._node_out_fcn = print
        self._node_ask_to_get_in_touch_fcn = None
        self._node_purge_fcn = None
        self._node_agents_waiting = None

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

        # Loading default public HSM
        if hasattr(self, "do_gen"):  # Trick to distinguish if this is an Agent or a World (both sons of this class)
            self.is_world = False

            # Setting an empty HSM as default is not provided (private/world)
            if self.behav is None:
                self.behav = HybridStateMachine(self)
                self.behav.add_state("empty")

            if self.behav_lone_wolf is not None and isinstance(self.behav_lone_wolf, str):
                template_string = self.behav_lone_wolf
                if template_string == "serve":
                    json_to_load = "lone_wolf.json"
                elif template_string == "ask":
                    json_to_load = "ask_lone_wolf.json"
                else:
                    raise ValueError("Invalid behav_lone_wolf: it must be an HybridStateMachine or a string "
                                     "in ('serve', 'ask')")

                # Safe way to load a file packed in a pip package
                self.behav_lone_wolf = HybridStateMachine(self)
                utils_path = importlib.resources.files("unaiverse.utils")
                json_file = utils_path.joinpath(json_to_load)
                file = json_file.open()
                self.behav_lone_wolf.load(file)
                file.close()
        else:
            self.is_world = True
            if self.world_folder is None:
                raise GenException("No world folder was indicated (world_folder argument)")

    def set_node_info(self, clock: Clock, conn: ConnectionPools, profile: NodeProfile,
                      out_fcn, ask_to_get_in_touch_fcn, purge_fcn, agents_waiting, print_level):
        """Set the required information from the node that hosts this agent.

        Args:
            clock: The global clock instance from the node.
            conn: The connection pool manager from the node.
            profile: The profile of the hosting node.
            out_fcn: The function to use for general output messages.
            ask_to_get_in_touch_fcn: The function to call to request getting in touch with another peer.
            purge_fcn: The function to call to purge (kill/disconnect) a connection.
            agents_waiting: Set of agents that connected to this node but have not been evaluated yet to be added.
            print_level: The level of output printing verbosity (0, 1, 2).
        """

        # Getting basic references
        self._node_name = profile.get_static_profile()['node_name']
        self._node_clock = clock
        self._node_profile = profile
        self._node_conn = conn
        self._node_out_fcn = out_fcn
        self._node_ask_to_get_in_touch_fcn = ask_to_get_in_touch_fcn
        self._node_purge_fcn = purge_fcn
        self._node_agents_waiting = agents_waiting

        # Adding peer_id information into the already existing stream data (if any)
        # (initially marked with generic wildcards like <public_peer_id>, ...)
        net_hashes = list(self.known_streams.keys())
        for net_hash in net_hashes:
            if net_hash.startswith("<public_peer_id>") or net_hash.startswith("<private_peer_id>"):
                stream_dict = self.known_streams[net_hash]
                for stream_obj in stream_dict.values():
                    self.add_stream(stream_obj, owned=True)  # This will also re-add streams using the node clock
        self.remove_streams("<public_peer_id>", owned_too=True)
        self.remove_streams("<private_peer_id>", owned_too=True)

        # World only: loading action files and refactoring (or building) JSON files of the different roles.
        # This where the world guesses roles.
        if self.is_world:

            # Check role-JSON files in the world folder
            role_json_tracker = FileTracker(self.world_folder, ext=".json", skip="world.stats.json")

            # Check stats-JSON file in the world folder
            stats_json_tracker = FileTracker(self.world_folder, ext=".json", prefix="world.stats")

            # This usually does nothing, but if you like to dynamically create JSON files, overload this method
            self.create_behav_files()

            # Loading and refactoring roles and behaviors
            self.load_and_refactor_action_file_and_behav_files(force_save=role_json_tracker.something_changed())

            # Building combination of default roles (considering public, world_agent, world_master default roles), and
            # agent/world specific roles
            self.augment_roles()

            # This usually does nothing, but if you like to dynamically create the JSON file "world.stats.json",
            # overload this method
            self.create_stats_file()

            # Loading and refactoring stats
            self.load_and_refactor_stats_file(force_save=stats_json_tracker.something_changed())

        # Creating streams associated to the processor output
        self.create_proc_output_streams(buffered=self.buffer_generated)

        # Updating node profile by indicating the processor-related streams
        self.update_streams_in_profile()

        # Print level
        AgentBasics.DEBUG = print_level > 1
        ConnectionPools.DEBUG = print_level > 1
        HybridStateMachine.DEBUG = print_level > 1

        # Subscribing/creating our own pubsub
        return self.subscribe_to_pubsub_owned_streams()

    def augment_roles(self):
        """Augment the custom roles (role1, role2, etc.) with the default ones (public, world_master, etc.), generating
        all the mixed roles (world_master~role1, world_master~role2, ...)"""

        # Both Agent and World: Fusing basic roles and custom roles
        if len(self.CUSTOM_ROLES) > 0:
            if len(self.CUSTOM_ROLES) > 30:  # Safe value, could be increased
                raise GenException("Maximum number of custom role overcame (max is 30)")
            for i, role_str in enumerate(self.CUSTOM_ROLES):
                role_int = 1 << (i + 2)  # 000000100, then 00001000, etc. (recall that the first two bits are reserved)
                self.ROLE_BITS_TO_STR[role_int] = role_str
                self.ROLE_STR_TO_BITS[role_str] = role_int

        # Both Agent and World: Augmenting roles
        roles_not_to_be_augmented = {self.ROLE_PUBLIC, self.ROLE_WORLD_AGENT, self.ROLE_WORLD_MASTER}
        role_bits_to_str_original = {k: v for k, v in self.ROLE_BITS_TO_STR.items()}
        for role_int, role_str in role_bits_to_str_original.items():
            if role_int not in roles_not_to_be_augmented and "~" not in role_str:
                for role_base_int in {self.ROLE_WORLD_AGENT, self.ROLE_WORLD_MASTER}:
                    augmented_role_int = role_base_int | role_int
                    augmented_role_str = self.ROLE_BITS_TO_STR[role_base_int] + "~" + role_str
                    if augmented_role_str not in self.ROLE_STR_TO_BITS:
                        self.ROLE_STR_TO_BITS[augmented_role_str] = augmented_role_int
                        self.ROLE_BITS_TO_STR[augmented_role_int] = augmented_role_str

    def clear_world_related_data(self):
        """Destroy all the cached information that is about a world (useful when leaving a world)."""

        # Clearing status variables
        self.reset_agent_status_attrs()

        # Clear/reset
        self.__remove_all_world_private_streams()
        self.__remove_all_world_related_agents()
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
            self.CUSTOM_ROLES = [os.path.splitext(f)[0] for f in os.listdir(self.world_folder)
                                 if os.path.isfile(os.path.join(self.world_folder, f))
                                 and f.lower().endswith(".json")
                                 and f.lower() != "world.stats.json"]
            if len(self.CUSTOM_ROLES) == 0:
                raise GenException(f"No world-role files (*.json) were found in the world folder {self.world_folder}")

            # Default behaviours (getting roles, that are the names of the files with extension "json")
            default_behav_files = [os.path.join(self.world_folder, f) for f in os.listdir(self.world_folder)
                                   if os.path.isfile(os.path.join(self.world_folder, f)) and
                                   f.lower().endswith(".json")
                                   and f.lower() != "world.stats.json"]

            # Loading action file
            action_file = os.path.join(self.world_folder, 'agent.py')
            try:
                with open(action_file, 'r', encoding='utf-8') as file:
                    self.agent_actions = file.read()
            except Exception as e:
                raise GenException(f'Error while reading the agent.py file: {action_file} [{e}]')

            # Creating a dummy agent which supports the actions of the following state machines
            mod = types.ModuleType("dynamic_module")
            try:
                exec(self.agent_actions, mod.__dict__)
                dummy_agent = mod.WAgent(proc=None)
                dummy_agent.CUSTOM_ROLES = self.CUSTOM_ROLES
            except Exception as e:
                raise GenException(f'Unable to create a valid agent object from the agent action file '
                                   f'{action_file} [{e}]')

            # Checking if the roles you wrote in agent.py are coherent with the JSON files in this folder
            if dummy_agent.CUSTOM_ROLES != self.CUSTOM_ROLES:
                raise GenException(f"Mismatching roles. "
                                   f"Roles in JSON files: {self.CUSTOM_ROLES}. "
                                   f"Roles specified in the agent.py file: {dummy_agent.CUSTOM_ROLES}")

            # Loading and refactoring behaviors
            for role, default_behav_file in zip(self.CUSTOM_ROLES, default_behav_files):
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

    def create_behav_files(self):
        """This method is called when building a world object. In your custom world-class, you can overload this method
        and create the JSON files with the role-related behaviors, if you like. Recall that acting like this is not
        mandatory at all: you can just manually create the JSON  files, and this method will simply do nothing."""
        pass

    def load_and_refactor_stats_file(self, force_save: bool = False):
        """This method is called when building a world object. It loads file "world.stats.json" (JSON) and refactors it.
        It checks the validity of the file contents.

            Args:
                force_save: Boolean to force the saving of the JSON.
        """

        # World only: world discovers WORLD_STATS_DYNAMIC_BY_PEER from the JSON file 'world.stats.json' in world folder
        if self.world_folder is not None and self.is_world:

            # Loading stats
            stats_definition = None
            valid = False
            stats_file_name = 'world.stats.json'
            stats_file = os.path.join(self.world_folder, stats_file_name)
            try:
                if os.path.exists(stats_file):
                    with open(stats_file, 'r', encoding='utf-8') as file:
                        stats_definition = json.load(file)
            except Exception as e:
                raise GenException(f'Error while reading the {stats_file_name} file [{e}]')

            # Checking stats
            if stats_definition is not None:
                valid = isinstance(stats_definition, dict)
                if valid:
                    valid &= len(stats_definition) > 0
                    for stat_name, stat in stats_definition.items():
                        valid &= isinstance(stat_name, str)
                        valid &= isinstance(stat, dict)
                        if valid:
                            keys = set(list(stat.keys()))
                            keys.add("min")
                            keys.add("max")
                            valid &= keys == {"desc", "type", "max", "min"}
                            if valid:
                                valid &= isinstance(stat_name, str)
                                valid &= isinstance(stat["desc"], str)
                                valid &= isinstance(stat["type"], str)
                                if valid:
                                    valid &= stat["type"] in ["number", "str"]
                                    if "max" in stat.keys() and "min" in stat.keys():
                                        try:
                                            _max = float(stat["max"])
                                            if _max != stat["max"]:
                                                stat["max"] = _max
                                                force_save = True
                                        except ValueError:
                                            valid &= False
                                        try:
                                            _min = float(stat["min"])
                                            if _min != stat["min"]:
                                                stat["min"] = _min
                                        except ValueError:
                                            valid &= False
                                    else:
                                        if "max" in stat.keys():
                                            del stat["max"]
                                            force_save = True
                                        if "min" in stat.keys():
                                            del stat["min"]
                                            force_save = True

                if valid and force_save:
                    try:
                        with open(stats_file, 'w', encoding='utf-8') as file:
                            json.dump(stats_definition, file, indent=4)
                    except Exception as e:
                        raise GenException(f'Error while saving the {stats_file_name} file [{e}]')

            # Adding dynamic stats to world profile
            stats_definition = stats_definition if valid else {}
            self._node_profile.get_dynamic_profile()['world_stats_dynamic'] = stats_definition

            # Saving dynamic stats to the world object
            self.WORLD_STATS_DYNAMIC_BY_PEER.clear()  # in-place
            self.WORLD_STATS_DYNAMIC_BY_PEER.update(stats_definition)  # in-place

            # Loading existing stats (if any)
            self.clear_stats()
            stats_folder_name = "stats"
            stats_folder = os.path.join(self.world_folder, stats_folder_name)
            if not os.path.exists(stats_folder) or not os.path.isdir(stats_folder):
                os.makedirs(stats_folder, exist_ok=True)

            self.stats_loader_saver = \
                StatLoadedSaver(base_filename="dynamic",
                                save_dir=stats_folder,
                                max_size_mb=5,
                                dynamic_stats=list(set(self.WORLD_STATS_DYNAMIC) |
                                                    set(self.WORLD_STATS_DYNAMIC_BY_PEER.keys())),
                                static_stats=list(set(self.WORLD_STATS_STATIC) |
                                                    set(self.WORLD_STATS_STATIC_BY_PEER)),
                                group_indexed_stats=list(set(self.WORLD_STATS_STATIC_BY_PEER) |
                                                            set(self.WORLD_STATS_DYNAMIC_BY_PEER.keys())),
                                group_key="peer_stats")
            self.stats_loader_saver.load_existing_data()

    def create_stats_file(self):
        """This method is called when building a world object. In your custom world-class, you can overload this method
        and create the JSON file "world.stats.json" with the stats you want agents to log and send-back to the world,
        if you like. Recall that acting like this is not mandatory at all: you can just manually create the JSON file
        "world.stats.json", and this method will simply do nothing."""
        pass

    def out(self, msg: str):
        """Print a message to the console, if enabled at node level (it reuses the node-out-function).

        Args:
            msg: The message string to print.
        """
        self._node_out_fcn(msg)

    def err(self, msg: str):
        """Print an error message to the console, if enabled at node level (it reuses the node-err-function).

        Args:
            msg: The error message string to print.
        """
        self.out("<ERROR> " + msg)

    def deb(self, msg: str):
        """Print an error message to the console, if debug is enabled for this agent (it reuses the agent-out-function).

        Args:
            msg: The error message string to print.
        """
        if AgentBasics.DEBUG:
            self.out("[DEBUG " + ("AGENT" if not self.is_world else "WORLD") + "] " + msg)

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

    def add_agent(self, peer_id: str, profile: NodeProfile) -> bool:
        """Add a new known agent.

        Args:
            peer_id: The unique identifier of the peer.
            profile: The NodeProfile object containing the peer's/agent's information.

        Returns:
            True if the agent was successfully added, False otherwise.
        """

        # If the agent was already there, we remove it and add it again (in case of changes)
        self.remove_agent(peer_id)  # It has no effects if the agent is not existing

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
            self.err(f"Cannot add agent with peer ID {peer_id} - unknown role: {role}")
            return False

        # Human or artificial?
        if profile.get_static_profile()["node_type"] is AgentBasics.HUMAN:
            self.human_agents[peer_id] = profile
        else:
            self.artificial_agents[peer_id] = profile

        # Check compatibility of the streams owned by the agent we are adding with our-agent's processor
        if self.proc_outputs is not None and self.proc_inputs is not None:

            # Check compatibility of the environmental streams of the agent we are adding with our-agent's processor
            environmental_streams = profile.get_dynamic_profile()['streams']
            if (environmental_streams is not None and
                    not self.add_compatible_streams(peer_id, environmental_streams,
                                                    buffered=False, public=public)):  # This will also "add" the stream
                return False

            # Check compatibility of the generated streams of the agent we are adding with our-agent's processor
            proc_streams = profile.get_dynamic_profile()['proc_outputs']
            if (proc_streams is not None and
                    not self.add_compatible_streams(peer_id, profile.get_dynamic_profile()['proc_outputs'],
                                                    buffered=False, public=public)):  # This will also "add" the stream
                return False

        self.out(f"Successfully added agent with peer ID {peer_id} (public: {public})")
        return True

    def remove_agent(self, peer_id: str):
        """Remove an agent.

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
                del self.artificial_agents[peer_id]

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
            self.remove_streams(peer_id)

            # Removing from the status variables
            self.remove_peer_from_agent_status_attrs(peer_id)

            # Updating buffered stream index
            if peer_id in self.last_buffered_peer_id_to_info:
                del self.last_buffered_peer_id_to_info[peer_id]  # Only if present
            self.out(f"Successfully removed agent with peer ID {peer_id}")

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
        self.out(f"Successfully removed all agents")

    def add_behav_wildcard(self, wildcard_from: str, wildcard_to: object):
        """Adds a wildcard mapping for the agent's behavior state machine.

        Args:
            wildcard_from: The string to be used as a wildcard.
            wildcard_to: The object to replace the wildcard.
        """
        self.behav_wildcards[wildcard_from] = wildcard_to

    def add_stream(self, stream: DataStream, owned: bool = True, net_hash: str | None = None) -> dict[str, DataStream]:
        """Add a new stream to the set of known streams.

        Args:
            stream: The DataStream object to add.
            owned: If True, the streams are considered owned by this agent.
            net_hash: Optional network hash for the streams. If None, it will be generated.

        Returns:
            A dictionary containing the added stream and the possibly already present streams belonging to the same
            group (stream name -> stream object).
        """

        # Forcing clock
        stream.clock = self._node_clock

        # Stream net hash
        if net_hash is None:
            public_peer_id, private_peer_id = self.get_peer_ids()
            peer_id = public_peer_id if stream.is_public() else private_peer_id
            net_hash = stream.net_hash(peer_id)

        # Adding the new stream
        if net_hash not in self.known_streams:
            self.known_streams[net_hash] = {}
        else:
            for _stream in self.known_streams[net_hash].values():
                public = _stream.get_props().is_public()
                pubsub = _stream.get_props().is_pubsub()
                if public and not stream.get_props().is_public():
                    self.err(f"Cannot add a stream to a group with different properties (public): "
                             f"hash: {net_hash}, name: {stream.get_props().get_name()}, "
                             f"public: {stream.get_props().is_public()}")
                    return {}
                if pubsub and not stream.get_props().is_pubsub():
                    self.err(f"Cannot add a stream to a group with different properties (pubsub): "
                             f"hash: {net_hash}, name: {stream.get_props().get_name()}, "
                             f"public: {stream.get_props().is_public()}")
                    return {}
                break
        self.known_streams[net_hash][stream.get_props().get_name()] = stream

        if owned:

            # Adding an 'owned' processor output stream (i.e., the stream coming from OUR OWN processor)
            is_proc_outputs_stream = False
            if self.proc_outputs is not None:
                proc_outputs_name_and_group = set()
                for props in self.proc_outputs:
                    proc_outputs_name_and_group.add((props.get_name(), props.get_group()))
                if (stream.get_props().get_name(), stream.get_props().get_group()) in proc_outputs_name_and_group:
                    if net_hash not in self.proc_streams:
                        self.proc_streams[net_hash] = {}
                    self.proc_streams[net_hash][stream.get_props().get_name()] = stream
                    is_proc_outputs_stream = True

            if net_hash not in self.owned_streams:
                self.owned_streams[net_hash] = {}
            self.owned_streams[net_hash][stream.get_props().get_name()] = stream

            if not is_proc_outputs_stream:
                if net_hash not in self.env_streams:
                    self.env_streams[net_hash] = {}
                self.env_streams[net_hash][stream.get_props().get_name()] = stream

        # Adding empty recipient slot
        if net_hash not in self.recipients:
            self.recipients[net_hash] = None

        # If needed, merging descriptor labels (attribute labels) and sharing them with all streams
        if self.merge_flat_stream_labels:
            self.merge_flat_data_stream_props()

        return self.known_streams[net_hash]

    def add_streams(self, streams: list[DataStream], owned: bool = True, net_hash: str | None = None) \
            -> list[dict[str, DataStream]]:
        """Add a list of new streams to this environment.

        Args:
            streams: A list of DataStream objects to add.
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

    def remove_streams(self, peer_id: str, name: str | None = None, owned_too: bool = False):
        """Remove a known stream.

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

            del self.known_streams[net_hash][name]
            if len(self.known_streams[net_hash]) == 0:
                del self.known_streams[net_hash]

            # Unsubscribing to pubsub
            if DataProps.is_pubsub_from_net_hash(net_hash):
                if peer_id != "<private_peer_id>" and peer_id != "<public_peer_id>":
                    if not self._node_conn.unsubscribe(peer_id, channel=net_hash):
                        self.err(f"Failed in unsubscribing from pubsub, peer_id: {peer_id}, channel: {net_hash}")
                    else:
                        self.out(f"Successfully unsubscribed from pubsub, peer_id: {peer_id}, channel: {net_hash}")

            # Removing all the owned streams (environment and processor streams are of course "owned")
            if net_hash in self.owned_streams:
                if name in self.owned_streams[net_hash]:
                    del self.owned_streams[net_hash][name]
                    if len(self.owned_streams[net_hash]) == 0:
                        del self.owned_streams[net_hash]
            if net_hash in self.env_streams:
                if name in self.env_streams[net_hash]:
                    del self.env_streams[net_hash][name]
                    if len(self.env_streams[net_hash]) == 0:
                        del self.env_streams[net_hash]
            if net_hash in self.proc_streams:
                if name in self.proc_streams[net_hash]:
                    del self.proc_streams[net_hash][name]
                    if len(self.proc_streams[net_hash]) == 0:
                        del self.proc_streams[net_hash]
            self.out(f"Successfully removed known stream with network hash {net_hash}, stream name: {name}")

    def remove_all_streams(self, owned_too: bool = False):
        """Remove all not-owned streams.

        Args:
            owned_too: If True, also removes the owned streams of this agent (so also environmental and processor ones).
        """
        if not owned_too:
            self.known_streams = {k: v for k, v in self.owned_streams}
        else:
            self.known_streams = {}
            self.owned_streams = {}
            self.env_streams = {}
            self.proc_streams = {}
        self.out(f"Successfully removed all streams!")

    def find_streams(self, peer_id: str, name_or_group: str | None = None) -> dict[str, dict[str, DataStream]]:
        """Find streams associated with a given peer ID and optionally by name or group.

        Args:
            peer_id: The peer ID of the (owner of the) streams to find.
            name_or_group: Optional name or group of the streams to find.

        Returns:
            A dictionary where keys are network hashes and values are dictionaries of streams
            (stream name to DataStream object) matching the criteria.
        """
        ret = {}
        for net_hash, streams_dict in self.known_streams.items():
            _peer_id = DataStream.peer_id_from_net_hash(net_hash)
            _name_or_group = DataStream.name_or_group_from_net_hash(net_hash)
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
        """Find streams associated with a given peer ID and optionally by name or group.

        Args:
            agent_name: The name of the agent.

        Returns:
            A list of data samples taken from all the known streams associated to the provided agent.
        """
        data_list = []
        for peer_id, profile in self.all_agents.items():
            if profile.get_static_profile()['node_name'] == agent_name:
                net_hash_to_stream_dict = self.find_streams(peer_id)
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

            # In the case of BufferedDataStream, we have to update the data buffer by clearing previously applied
            # adaptation first (I know it looks similar to what is done below, but we must clear first!)
            if isinstance(stream, BufferedDataStream):
                for i, (data, data_tag) in enumerate(stream.data_buffer):
                    stream.data_buffer[i] = (stream.props.clear_label_adaptation(data), data_tag)

            # Updating labels
            stream.props.tensor_labels.interleave_with(superset_labels)

            # In the case of BufferedDataStream, we have to update the data buffer with the new labels
            if isinstance(stream, BufferedDataStream):
                for i, (data, data_tag) in enumerate(stream.data_buffer):
                    stream.data_buffer[i] = (stream.props.adapt_tensor_to_tensor_labels(data), data_tag)

    def user_stream_hash_to_net_hash(self, user_stream_hash: str) -> str | None:
        """Converts a user-defined stream hash (peer_id:name_or_group) to a network hash
        (peer_id::dm:... or peer_id::ps:name_or_group) by searching the known hashes in the known streams.

        Args:
            user_stream_hash: The user-defined stream hash string (peer_id:name_or_group).

        Returns:
            The corresponding network hash string (peer_id::dm:... or peer_id::ps:name_or_group), or None if not found.
        """
        components = user_stream_hash.split(":")
        peer_id = components[0]
        name_or_group = components[-1]
        for net_hash in self.known_streams.keys():
            _peer_id = DataStream.peer_id_from_net_hash(net_hash)
            _name_or_group = DataStream.name_or_group_from_net_hash(net_hash)
            if _peer_id == peer_id and _name_or_group == name_or_group:
                return net_hash
        return None

    def create_proc_output_streams(self, buffered: bool = False):
        """Creates the processor output streams based on the `proc_outputs` defined for the agent.

        Args:
            buffered: If True, the created streams will be of type BufferedDataStream.
        """

        # Adding generated streams (grouped together), passing the node clock
        if self.proc_outputs is not None:
            for i, procs in enumerate(self.proc_outputs):
                procs.set_group("processor")  # Adding default group info, forced, do not change this!

                # Creating the streams
                for props in procs.props:
                    if not buffered:
                        stream = DataStream(props=props.clone(), clock=self._node_clock)
                    else:
                        stream = BufferedDataStream(props=props.clone(), clock=self._node_clock)

                    self.add_stream(stream, owned=True)

    def add_compatible_streams(self, peer_id: str,
                               streams_in_profile: list[DataProps], buffered: bool = False,
                               add_all: bool = False, public: bool = True) -> bool:
        """Add to the list of processor-compatible-streams those streams provided as arguments that are actually
        found to be compatible with the processor (if they are pubsub, it also subscribes to them).

        Args:
            peer_id: The peer ID of the agent providing the streams.
            streams_in_profile: A list of DataProps objects representing the streams from the peer's profile.
            buffered: If True, the added streams will be of type BufferedDataStream.
            add_all: If True, all streams from the profile are added, regardless of processor compatibility.
            public: Consider public streams only (or private streams only).

        Returns:
            True if compatible streams were successfully added and subscribed to, False otherwise.
        """
        added_streams = []

        if add_all:

            # This is the case in which we add all streams, storing all pairs (DataProps, net_hash)
            for j in streams_in_profile:
                jj = DataProps.from_dict(j)
                if public == jj.is_public():
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
                    if public == jj.is_public() and in_proc.is_compatible(jj):
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
                    if (public == jj.is_public() and
                            (out_proc.is_compatible(jj) or (jj.is_tensor_target_id() and has_cross_entropy[i]))):
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
                stream = DataStream(props=props.clone(), clock=self._node_clock)
            else:
                stream = BufferedDataStream(props=props.clone(), clock=self._node_clock)

            # Add the data stream to the list of known streams
            # if the stream already exists it will be overwritten (which is fine in case of changes)
            self.add_stream(stream, owned=False, net_hash=net_hash)

            # If the stream is over PubSub, and we are not already subscribed, we will subscribe
            if props.is_pubsub() and not already_known_stream:
                net_hashes_to_subscribe.add(net_hash)

        # Opening PubSubs
        for net_hash in net_hashes_to_subscribe:
            self.out(f"Opening channel for the not-owned but processor-compatible stream {net_hash}")
            if not self._node_conn.subscribe(peer_id, channel=net_hash):
                self.err(f"Error subscribing to {net_hash}")
                return False

        return True

    def subscribe_to_pubsub_owned_streams(self) -> bool:
        """Subscribes to all owned streams that are marked as PubSub.

        Returns:
            True if all subscriptions were successful, False otherwise.
        """

        # Opening channels for all the (groups of) owned streams (generated and not)
        for net_hash in self.owned_streams.keys():
            is_pubsub = DataStream.is_pubsub_from_net_hash(net_hash)

            if is_pubsub:
                self.out(f"Opening channel for the owned stream {net_hash}")
                peer_id = DataStream.peer_id_from_net_hash(net_hash)  # Guessing peer ID from the net hash

                if not self._node_conn.subscribe(peer_id, channel=net_hash):
                    self.err(f"Cannot open a channel for owned stream hash {net_hash}")
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
        for net_hash, streams_dict in self.known_streams.items():
            if net_hash not in self.proc_streams.keys():
                if (DataProps.peer_id_from_net_hash(net_hash) == public_peer_id or
                        DataProps.peer_id_from_net_hash(net_hash) == private_peer_id):
                    for stream in streams_dict.values():
                        list_of_props.append(stream.get_props().to_dict())  # DataProp
        if len(list_of_props) > 0:
            dynamic_profile['streams'] = list_of_props

    def send_profile_to_all(self):
        """Sends the agent's profile to all known agents."""

        for peer_id in self.all_agents.keys():
            self.out(f"Sending profile to {peer_id}")
            if not self._node_conn.send(peer_id, channel_trail=None,
                                        content=self._node_profile.get_all_profile(),
                                        content_type=Msg.PROFILE):
                self.err("Failed to send profile, removing (disconnecting) " + peer_id)
                self.remove_agent(peer_id)

    def generate(self, input_net_hashes: list[str] | None = None,
                 inputs: list[str | torch.Tensor | Image] | None = None,
                 first: bool = False, last: bool = False, ref_uuid: str | None = None) -> (
            tuple[tuple[torch.Tensor] | None, int]):
        """Generate new signals.

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
                stream_dict = self.known_streams[net_hash]
                for stream_name, stream in stream_dict.items():

                    # Checking the UUID in our known streams, comparing it with the UUID provided as input:
                    # if they are not compatible, we don't generate at all
                    if ref_uuid is not None and stream.get_uuid(expected=False) != ref_uuid:
                        self.deb(f"[generate] The UUID ({stream.get_uuid(expected=False)}, expected: "
                                 f"{stream.get_uuid(expected=True)}) of stream {net_hash} is not the one we were "
                                 f"looking for ({ref_uuid}), skipping this data stream")
                        continue

                    # Matching the currently checked input stream with one of the processor inputs
                    stream_sample = None
                    for i in range(len(self.proc_inputs)):

                        # If the current input stream is compatible with the i-th input slot...
                        if (net_hash, stream_name) in self.compat_in_streams[i]:

                            # If the current input stream was already assigned to another input slot
                            # (different from "i") we skip the generation
                            if (net_hash, stream_name) in matched:
                                self.err("Cannot generate: ambiguous input streams provided "
                                         "(they can match multiple processor inputs)")
                                return None, -1

                            # Found a valid assignment: getting stream sample
                            self.deb(f"[generate] Setting the {i}-th network input to stream with "
                                     f"net_hash: {net_hash}, name: {stream_name}")
                            if stream_sample is None:
                                stream_sample = stream.get(requested_by="generate")
                                if stream_sample is None:  # If there are no samples in this stream, we cannot generate
                                    self.err(
                                        f"Cannot generate: got nothing (None) from what we would provide to the {i}-th "
                                        f"input position of the processor")
                                    return None, -1

                            # Found a valid assignment: associating it to the i-th input slot
                            try:
                                inputs[i] = self.proc_inputs[i].check_and_preprocess(stream_sample,
                                                                                     device=self.proc.device)
                            except Exception as e:
                                self.err(f"Error while checking and preprocessing the {i}-th input [{e}]")

                            # Found a valid assignment: saving match
                            matched.add((net_hash, stream_name))

                            # If all the inputs share the same data tag, we will return it,
                            # otherwise we set it at -1 (meaning no tag)
                            if data_tag is None:
                                data_tag = stream.get_tag()
                            elif data_tag != stream.get_tag():
                                data_tag = -1

                            if AgentBasics.DEBUG:
                                if stream.props.is_text():
                                    self.deb(f"[generate] Input of the network: {stream_sample}")

                            break

            # Checking if we were able to match some data for each input slot of the network (processor)
            for i in range(len(self.proc_inputs)):
                if inputs[i] is None:
                    if self.proc_optional_inputs[i]["has_default"]:
                        inputs[i] = self.proc_optional_inputs[i]["default_value"]
                    else:
                        self.err(
                            f"Cannot generate: couldn't find a valid input for the "
                            f"{i}-th input position of the processor (and no default values are present)")
                        return None, -1
        else:
            data_tag = -1

        if AgentBasics.DEBUG:
            if inputs is not None:
                self.deb(f"[generate] Input shapes: {[x.shape for x in inputs if isinstance(x, torch.Tensor)]}")
                self.deb(f"[generate] Input data tag: {data_tag}")

        # Calling processor (inference) passing the collected inputs
        inputs = self.proc_callback_inputs(inputs)
        try:
            outputs = self.proc(*inputs, first=first, last=last)

            # Ensuring the output is a tuple, even if composed by a single tensor
            if not isinstance(outputs, tuple):
                outputs = (outputs, )
        except Exception as e:
            self.err(f"Error while calling the processor [{e}]")
            outputs = None
        outputs = self.proc_callback_outputs(outputs)

        # Saving
        self.last_ref_uuid = ref_uuid

        if AgentBasics.DEBUG:
            if outputs is not None:
                for net_hash, stream_dict in self.proc_streams.items():
                    for stream in stream_dict.values():
                        if stream.props.is_tensor() or stream.props.is_text():
                            if outputs is not None:
                                self.deb(f"[generate] Text outputs: "
                                         f"{[str(stream.props.to_text(x)) for i, x in enumerate(outputs)]}")
                            break
                self.deb(f"[generate] Output shapes: {[x.shape for x in outputs if isinstance(x, torch.Tensor)]}")

        return outputs, data_tag

    def learn_generate(self,
                       outputs: tuple[torch.Tensor],
                       targets_net_hashes: list[str] | None) -> tuple[list[float] | None, list[float] | None]:
        """Learn (i.e., update model params) by matching the given processor outputs with a set of targets (if any).

        Args:
            outputs: A tuple of torch.Tensor representing the outputs generated by the agent's processor.
            targets_net_hashes: An optional list of network hashes identifying the streams
                                from which target data should be retrieved for learning.
                                If None, losses are evaluated without explicit targets.

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
                                self.err("Cannot generate: ambiguous target streams provided "
                                         "(they can match multiple processor outputs)")
                                return None, None

                            # Found a valid assignment: getting stream sample
                            if stream_sample is None:
                                stream_sample = stream.get(requested_by="learn_generate")
                                if stream_sample is None:
                                    return None, None

                            # Found a valid assignment: associating target to the i-th output slot
                            try:
                                targets[i] = self.proc_outputs[i].check_and_preprocess(stream_sample,
                                                                                       allow_class_ids=True,
                                                                                       targets=True,
                                                                                       device=self.proc.device)
                            except Exception as e:
                                self.err(f"Error while checking and preprocessing the {i}-th targets [{e}]")

                            # Found a valid assignment: saving match
                            matched.add((net_hash, stream_name))

                            # Saving tag
                            data_tags[i] = stream.get_tag()

                            # Confirming
                            at_least_one_target_found = True

                            if AgentBasics.DEBUG:
                                if stream.props.is_tensor():
                                    self.deb("[generate] Target of the network: " +
                                             str(stream.props.to_text(targets[i])))
                                elif stream.props.is_text():
                                    self.deb("[generate] Target of the network: " + stream_sample)
                            break

            # If no targets were matched, we skip learning
            if not at_least_one_target_found:
                self.err(f"Cannot learn: cannot find a valid target for any output positions of the processor")
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

        # Teaching (for autoregressive models, expected to have attribute "y")
        if hasattr(self.proc, 'y'):
            self.proc.y = targets[0]

        # Returning a list of float values and the data tags of the targets
        return [loss_value.item() for loss_value in loss_values], data_tags

    def behave(self):
        """Behave in the current environment, calling the state-machines of the public and private networks."""

        if self.in_world():
            self.out("Behaving (world)...")
            if self.behav is None:
                self.err("No behaviour specified")
            else:
                self.behav_lone_wolf.enable(False)
                self.behav.enable(True)
                self.behav.act()
                self.behav.enable(False)

        self.out("Behaving (public)...")
        if self.behav_lone_wolf is None:
            self.err("No behaviour specified")
        else:
            self.behav.enable(False)
            self.behav_lone_wolf.enable(True)
            self.behav_lone_wolf.act()
            self.behav_lone_wolf.enable(False)

    def learn_behave(self, state: int, last_action: int, prev_state: int):
        """A placeholder method for behavioral learning, intended to be implemented by child classes.
        It receives state and action information to update a behavioral model.

        Args:
            state: The current state of the agent.
            last_action: The last action taken.
            prev_state: The previous state of the agent.

        Returns:
            An integer representing a new state, or similar feedback.
        """
        pass

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
        if self.ROLE_STR_TO_BITS[profile.get_dynamic_profile()['connections']['role']] != role:
            self.out(f"Cheater found: "
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

        base_role_to_behav = self.world_profile.get_dynamic_profile()['world_roles_fsm']
        if base_role_str in base_role_to_behav:
            default_behav = self.world_profile.get_dynamic_profile()['world_roles_fsm'][base_role_str]
        else:
            default_behav = None  # A public role will not be found in the map

        if default_behav is not None and len(default_behav) > 0:
            default_behav_hsm = HybridStateMachine(self)
            default_behav_hsm.load(default_behav)
            self.behav = HybridStateMachine(self)
            self.behav.include(default_behav_hsm, make_a_copy=True)

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

    def get_stream_sample(self, net_hash: str, sample_dict: dict[str, dict[str, torch.Tensor | None | int | str]]):
        """Receive and process stream samples that were provided by another agent.

        Args:
            net_hash: The network hash identifying the source of the stream samples.
            sample_dict: A dictionary where keys are stream names and values are dictionaries
                         containing 'data', 'data_tag', and 'data_uuid' for each sample.

        Returns:
            True if the stream samples were successfully processed and stored, False otherwise
            (e.g., if the stream is unknown, not compatible, or data is None/stale).
        """

        # Let's be sure that the net hash is converted from the user's perspective to the one of the code here
        net_hash = DataProps.normalize_net_hash(net_hash)

        self.out(f"Got a stream sample from {net_hash}...")

        if net_hash in self.known_streams:
            for name, data_and_tag_and_uuid in sample_dict.items():
                if AgentBasics.DEBUG:
                    if net_hash in self.known_streams and name in self.known_streams[net_hash]:
                        self.deb(f"[get_stream_sample] Local data stream {name} status: tag="
                                 f"{self.known_streams[net_hash][name].get_tag()}, uuid="
                                 f"{self.known_streams[net_hash][name].get_uuid(expected=False)}, uuid-expected="
                                 f"{self.known_streams[net_hash][name].get_uuid(expected=True)}")

                data, data_tag, data_uuid = (data_and_tag_and_uuid['data'],
                                             data_and_tag_and_uuid['data_tag'],
                                             data_and_tag_and_uuid['data_uuid'])

                # - data must be not None
                # - the stream name must be known
                # - if the UUID associated to our local stream is the same of the data, then we check tag order
                # - if the UUID associated to our local stream is the expected one, we don't check tag order
                skip = data is None
                skip = skip or net_hash not in self.known_streams
                skip = skip or name not in self.known_streams[net_hash]
                skip = (skip or (self.known_streams[net_hash][name].get_uuid(expected=True) is not None and
                        data_uuid != self.known_streams[net_hash][name].get_uuid(expected=True)))
                skip = (skip or (self.known_streams[net_hash][name].get_uuid(expected=True) is None and
                        self.known_streams[net_hash][name].get_uuid(expected=False) is not None and
                        data_uuid != self.known_streams[net_hash][name].get_uuid(expected=False)))
                skip = (skip or (self.known_streams[net_hash][name].get_uuid(expected=True) is None and
                        self.known_streams[net_hash][name].get_uuid(expected=False) is not None and
                        data_uuid == self.known_streams[net_hash][name].get_uuid(expected=False) and
                        data_tag <= self.known_streams[net_hash][name].get_tag()))

                # If we sample can be accepted...
                if not skip:
                    self.out(f"Accepted sample named {name}: tag={data_tag}, uuid={data_uuid}")

                    # Saving the data sample on the known stream objects
                    self.known_streams[net_hash][name].set(data, data_tag)

                    # If the local stream was expecting data with a certain UUID, and we got it ...
                    # OR
                    # if the local stream was not expecting anything and was also not set to any UUID, and we got data
                    # with some UUID ...
                    # THEN
                    # we clear expectations and set the current UUID to the one of the data.
                    # (the second part of the OR above is the case of data that arrives before an action request,
                    # since action requests set expectations only)
                    if ((self.known_streams[net_hash][name].get_uuid(expected=True) is not None and
                         data_uuid == self.known_streams[net_hash][name].get_uuid(expected=True)) or
                            (self.known_streams[net_hash][name].get_uuid(expected=True) is None and
                             self.known_streams[net_hash][name].get_uuid(expected=False) is None and
                             data_uuid is not None)):

                        # Setting what was the expected UUID as the local UUID from now on
                        self.known_streams[net_hash][name].set_uuid(data_uuid, expected=False)  # Setting current
                        self.known_streams[net_hash][name].set_uuid(None, expected=True)  # Clearing expected

                        if AgentBasics.DEBUG:
                            self.deb(f"[get_stream_sample] Switched uuid of the local data stream!")
                            self.deb(f"[get_stream_sample] New local data stream status: tag="
                                     f"{self.known_streams[net_hash][name].get_tag()}, uuid="
                                     f"{self.known_streams[net_hash][name].get_uuid(expected=False)}, uuid-expected="
                                     f"{self.known_streams[net_hash][name].get_uuid(expected=True)}")

                    # Buffering data, if it was requested and if this sample comes from somebody's processor
                    if (self.buffer_generated_by_others != "none" and
                            DataProps.name_or_group_from_net_hash(net_hash) == "processor"):
                        self.deb(f"[get_stream_sample] Buffering others' processor generated data...")

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
                        _net_hash = DataProps.build_net_hash(
                            _peer_id,
                            pubsub=False,
                            name_or_group=("buffered" + str(_buffered_id)))

                        # If the buffered stream was not created before
                        if _net_hash not in self.known_streams:
                            self.deb(f"[get_stream_sample] Adding a new buffered stream to the list of known "
                                     f"streams, hash: {_net_hash}")
                            for stream_obj in _processor_stream_dict.values():

                                # Same properties of the stream of the processor of the source agent
                                props = stream_obj.get_props().clone()
                                props.set_group("buffered" + str(_buffered_id))

                                # Adding the newly created stream
                                self.add_stream(BufferedDataStream(props=props, clock=self._node_clock),
                                                owned=False,
                                                net_hash=_net_hash)

                            # Saving hash of the new buffered stream
                            self.last_buffered_peer_id_to_info[_peer_id]["net_hash"] = _net_hash
                        else:
                            if clear:
                                for stream_obj in self.known_streams[_net_hash].values():
                                    stream_obj.clear_buffer()

                        # Saving sample
                        self.known_streams[_net_hash][name].set(data, data_tag)

                        # Clearing all UUID of the locally buffered stream
                        self.known_streams[_net_hash][name].set_uuid(None, expected=False)
                        self.known_streams[_net_hash][name].set_uuid(None, expected=True)

                # If we decided to skip this sample...
                else:
                    self.out(f"Skipping sample named {name}: tag={data_tag}, uuid={data_uuid}" +
                             (", data is None!" if data is None else ""))
            return True

        # If this stream is not known at all...
        else:
            self.out(f"Skipping sample from {net_hash} (data stream is unknown)")
            return False

    def send_stream_samples(self):
        """Collect and send stream samples from all owned streams to appropriate recipients."""

        # Get samples from all the owned streams
        for net_hash, streams_dict in self.owned_streams.items():

            # Preparing content to send
            something_to_send = False
            content = {name: {} for name in streams_dict.keys()}
            for name, stream in streams_dict.items():
                data = stream.get(requested_by="send_stream_samples")

                if data is not None:
                    something_to_send = True
                    self.deb(f"[send_stream_samples] Preparing to send stream samples from {net_hash}, named {name} "
                             f"(tag={stream.get_tag()}, uuid={stream.get_uuid()})")

                content[name] = {'data': data, 'data_tag': stream.get_tag(), 'data_uuid': stream.get_uuid()}

                stream.clear_uuid_if_marked_as_clearable()

            # Checking if there is something valid in this group of streams
            if not something_to_send:
                continue

            # Guessing recipient of direct message (if None, then PubSub)
            recipient = self.recipients[net_hash]

            # Debug: force pubsub to be sent as direct message to the first agent
            # if self._recipients[net_hash] is None:
            # for peer_id in self.all_agents.keys():
            # recipient = peer_id
            # break

            # If pubsub...
            if recipient is None:
                if DataStream.is_pubsub_from_net_hash(net_hash):
                    self.deb(f"[send_stream_samples] Sending stream samples of the whole {net_hash} by pubsub")

                    peer_id = DataStream.peer_id_from_net_hash(net_hash)  # Guessing agent peer ID from the net hash
                    ret = self._node_conn.publish(peer_id, channel=net_hash,
                                                  content_type=Msg.STREAM_SAMPLE,
                                                  content=content)

                    self.deb(f"[send_stream_samples] Sending returned: " + str(ret))

            # If direct message...
            else:
                if not DataStream.is_pubsub_from_net_hash(net_hash):
                    _recipients = recipient if isinstance(recipient, list) else [recipient]
                    for i, _recipient in enumerate(_recipients):
                        self.deb(f"[send_stream_samples] Sending samples by direct message, to {_recipient}")

                        peer_id = _recipient  # Peer ID from the recipient information
                        name_or_group = DataProps.name_or_group_from_net_hash(net_hash)
                        ret = self._node_conn.send(peer_id, channel_trail=name_or_group,
                                                   content_type=Msg.STREAM_SAMPLE,
                                                   content=content)

                        self.deb(f"[send_stream_samples] Sending returned: " + str(ret))
                else:
                    raise ValueError(f"Unexpected scenario: recipient set ({recipient}) and sending on a pubsub stream")

    def get_action_step(self):
        """Retrieve the current action step from the agent's private/world behavior.

        Returns:
            The current action step object from the HybridStateMachine's active action, or None if no action.
        """
        behav = self.behav if self.behav.is_enabled() else self.behav_lone_wolf
        return behav.get_action_step()

    def is_last_action_step(self):
        """Check if the agent's current action (private/world behaviour) is on its last step.

        Returns:
            True if the current action was its last step, False otherwise. Returns None if there is no active action.
        """
        behav = self.behav if self.behav.is_enabled() else self.behav_lone_wolf
        action = behav.get_action()
        if action is not None:
            return action.was_last_step_done()
        else:
            return None

    def is_multi_steps_action(self):
        """Determines if the current action is a multistep action.

        Returns:
            True if the action is multistep, False otherwise.
        """
        behav = self.behav if self.behav.is_enabled() else self.behav_lone_wolf
        action = behav.get_action()
        return action.is_multi_steps() if action is not None else False

    def proc_callback_inputs(self, inputs):
        """A callback method that saves the inputs to the processor right before execution.

        Args:
            inputs: The data inputs for the processor.

        Returns:
            The same inputs passed to the function.
        """
        self.proc_last_inputs = inputs
        return inputs

    def proc_callback_outputs(self, outputs):
        """A callback method that saves the outputs from the processor right after execution.

        Args:
            outputs: The data outputs from the processor.

        Returns:
            The same outputs passed to the function.
        """
        self.proc_last_outputs = outputs
        return outputs

    def save(self, where: str = "output"):
        """Save the agent's state, including its processor and other attributes, to a specified location.

        Args:
            where: The directory path where the agent's state should be saved. Defaults to "output".

        Returns:
            The string "<SAVE_OK>" upon successful saving.

        Raises:
            IOError: If there is an issue with file operations (e.g., directory creation, writing files).
            TypeError, ValueError, RuntimeError: For other potential issues during serialization or saving.
        """

        # Saving the processor
        if self.proc is not None:
            torch.save(self.proc.state_dict(), os.path.join(where, f"{self._node_name}.pt"))

        try:

            # Creating output folder
            if not os.path.exists(where):
                os.makedirs(where)

            # Saving the whole thing (excluding the processor)
            proc = self.proc
            self.proc = None
            with open(os.path.join(where, f"{self._node_name}.pkl"), "wb") as f:
                pickle.dump(self, f)
            self.proc = proc
        except (TypeError, ValueError, RuntimeError, IOError, FileNotFoundError) as e:
            self.out("Could not save " + ("agent" if not self.is_world else "world") + f" {self._node_name}: {e}")
            raise e

        return "<SAVE_OK>"  # This means OK

    def load(self, where: str = "output"):
        """Load the agent's state from a specified location.

        Args:
            where: The directory path from which the agent's state should be loaded. Defaults to "output".

        Returns:
            The loaded AgentBasics object.

        Raises:
            AssertionError: If the specified load path does not exist.
            IOError: If there is an issue with file operations (e.g., reading files).
        """

        # Checking output folder
        if not os.path.exists(where):
            raise GenException(f"Invalid load path: {where}")
        load_proc = self.proc is not None

        # Loading the whole object (no processor)
        with open(os.path.join(where, f"{self._node_name}.pkl"), "rb") as f:
            loaded = pickle.load(f)

        # Update self's attributes with the loaded object's attributes
        self.__dict__.update(loaded.__dict__)

        # Loading the processor
        if load_proc:
            self.proc.load_state_dict(torch.load(os.path.join(where, f"{self._node_name}.pt")))

        return loaded

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

    def __remove_all_world_related_agents(self):
        """Remove all world-related agents (masters and regular agents) from the agent's known lists."""

        to_remove = list(self.world_masters.keys())
        for peer_id in to_remove:
            self.remove_agent(peer_id)

        to_remove = list(self.world_agents.keys())
        for peer_id in to_remove:
            self.remove_agent(peer_id)

    def __remove_all_world_private_streams(self):
        """Remove all known streams that are flagged as not-public and are not owned by this agent."""

        # Find what to remove
        to_remove = []
        for net_hash, stream_dict in self.known_streams.items():
            for name, stream_obj in stream_dict.items():
                if not stream_obj.get_props().is_public() and net_hash not in self.owned_streams:
                    to_remove.append((DataProps.peer_id_from_net_hash(net_hash), name))

        # Remove it
        for (peer_id, name) in to_remove:
            self.remove_streams(peer_id, name)

        # Clear recipients associated to these streams
        recipient_net_hashes = list(self.recipients.keys())
        for net_hash in recipient_net_hashes:
            if net_hash not in self.known_streams:
                del self.recipients[net_hash]

    def remove_peer_from_agent_status_attrs(self, peer_id):
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

    def clear_stats(self):
        self.stats = {}
        if self.is_world:
            for k in self.WORLD_STATS_STATIC:
                self.stats[k] = None
            for k in self.WORLD_STATS_STATIC_BY_PEER:
                self.stats[k] = None
            for k in self.WORLD_STATS_DYNAMIC:
                self.stats[k] = {}  # every dynamic stats is a dictionary: timestamp -> value
            for k in self.WORLD_STATS_DYNAMIC_BY_PEER:
                self.stats[k] = {}  # peer ID -> dynamic stat; every dynamic stats is a dictionary: timestamp -> value
        else:
            for k in self.STATS_STATIC_SENT_BY_PEER:
                self.stats[k] = None
            for k in self.STATS_DYNAMIC_SENT_BY_PEER:
                self.stats[k] = {}  # every dynamic stats is a dictionary: timestamp -> value

    def store_static_stats(self):
        if not self.is_world:
            for k in self.STATS_STATIC_SENT_BY_PEER:
                if k == 'connected_peers':  # TODO don't send if the same (None)
                    self.stats[k] = list(self.world_masters.keys()) + list(self.world_agents.keys())
                elif k == 'state':
                    self.stats[k] = self.behav.get_state_name()
                elif k == 'action':
                    self.stats[k] = self.behav.get_action_name()
                elif k == 'last_action':
                    self.stats[k] = self.behav.get_last_completed_action_name()

    def store_dynamic_stat(self, stat_name: str, stat_value: int | float | str):
        if not self.is_world:
            if (stat_name in self.STATS_DYNAMIC_SENT_BY_PEER and
                    (isinstance(stat_value, int) or isinstance(stat_value, float) or isinstance(stat_value, str))):
                t = self._node_clock.get_time()
                self.stats[stat_name][t] = float(stat_value) \
                    if self.STATS_DYNAMIC_SENT_BY_PEER[stat_name]["type"] == "number" \
                    else str(stat_value)

    def save_stats_to_disk(self):
        if self.is_world:
            self.stats_loader_saver.save_incremental(self.stats)
