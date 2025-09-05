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
import types
from typing import Optional
from unaiverse.agent import AgentBasics
from unaiverse.hsm import HybridStateMachine
from unaiverse.networking.p2p.messages import Msg
from unaiverse.networking.node.profile import NodeProfile


class World(AgentBasics):

    def __init__(self, *args, **kwargs):
        """Initializes a World object, which acts as a special agent without a processor or behavior.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments, including 'role_to_behav' and 'agent_actions' which are used to
                configure the world.
        """

        # Deleting the "proc" parameter, if provided (it is the second one)
        if len(args) == 2:
            args = (args[0])

        # Clearing keyword-level processor-related attributes
        kwargs['proc'] = None  # Processor = None (existing)
        kwargs['proc_inputs'] = None
        kwargs['proc_outputs'] = None
        kwargs['proc_opts'] = None

        # Clearing behavior
        kwargs['behav'] = None

        # Removing world-specific arguments
        if 'role_to_behav' in kwargs:
            role_to_behav_files = kwargs['role_to_behav']
            del kwargs['role_to_behav']
        else:
            role_to_behav_files = None

        if 'agent_actions' in kwargs:
            agent_actions_file = kwargs['agent_actions']
            del kwargs['agent_actions']
        else:
            agent_actions_file = None

        # Creating a "special" agent with no processor and no behavior, which is our world
        super().__init__(*args, **kwargs)

        # Clearing processor (world must have no processor, and, maybe, a dummy processor was allocated when building
        # the agent in the init call above)
        self.proc = None
        self.proc_inputs = []  # Do not set it to None
        self.proc_outputs = []  # Do not set it to None
        self.compat_in_streams = None
        self.compat_out_streams = None

        # World specific attributes
        self.agent_badges: dict[str, list[dict]] = {}  # Peer_id -> collected badges for other agents
        self.role_changed_by_world: bool = False
        self.received_address_update: bool = False

        # Loading agent (actions) file
        if agent_actions_file is not None:
            path_of_this_file = str(os.path.dirname(os.path.abspath(__file__)))
            with open(os.path.join(path_of_this_file, 'library', 'worlds', agent_actions_file),
                      'r', encoding='utf-8') as file:
                self.agent_actions = file.read()
        else:
            self.agent_actions = ""  # Empty string

        # Loading default behaviours
        self.role_to_behav = {}
        if role_to_behav_files is not None:

            # Creating a dummy agent which supports the actions of the following state machines
            mod = types.ModuleType("dynamic_module")
            exec(self.agent_actions, mod.__dict__)
            dummy_agent = mod.WAgent(proc=None)
            path_of_this_file = str(os.path.dirname(os.path.abspath(__file__)))

            for role, default_behav_file in role_to_behav_files.items():
                behav = HybridStateMachine(dummy_agent)
                behav.load(os.path.join(path_of_this_file, 'library', 'worlds', default_behav_file))
                self.role_to_behav[role] = str(behav)
        else:
            self.role_to_behav = None

    def assign_role(self, profile: NodeProfile, is_world_master: bool):
        """Assigns an initial role to a newly connected agent.

        In this basic implementation, the role is determined based on whether the agent is a world master or a regular
        world agent, ensuring there's only one master.

        Args:
            profile: The NodeProfile of the new agent.
            is_world_master: A boolean indicating if the new agent is attempting to be a master.

        Returns:
            An integer representing the assigned role (e.g., ROLE_WORLD_MASTER or ROLE_WORLD_AGENT).
        """
        assert self.is_world, "Assigning a role is expected to be done by the world"

        if profile.get_dynamic_profile()['guessed_location'] == 'Some Dummy Location, Just An Example Here':
            pass

        # Currently, roles are only world masters and world agents
        if is_world_master:
            if len(self.world_masters) <= 1:
                return AgentBasics.ROLE_WORLD_MASTER
            else:
                return AgentBasics.ROLE_WORLD_AGENT
        else:
            return AgentBasics.ROLE_WORLD_AGENT

    def set_role(self, peer_id: str, role: int):
        """Sets a new role for a specific agent and broadcasts this change to the agent.

        It computes the new role and sends a message containing the new role and the corresponding default behavior
        for that role.

        Args:
            peer_id: The ID of the agent whose role is to be set.
            role: The new role to be assigned (as an integer).
        """
        assert self.is_world, "Setting the role is expected to be done by the world, which will broadcast such info"

        # Computing new role (keeping the first two bits as before)
        cur_role = self._node_conn.get_role(peer_id)
        new_role_without_base_int = (role >> 2) << 2
        new_role = (cur_role & 3) | new_role_without_base_int

        if new_role != role:
            self._node_conn.set_role(peer_id, new_role)
            self.out("Telling an agent that his role changed")
            if not self._node_conn.send(peer_id, channel_trail=None,
                                        content={'peer_id': peer_id, 'role': new_role,
                                                 'default_behav':
                                                     self.role_to_behav[
                                                         self.ROLE_BITS_TO_STR[new_role_without_base_int]]
                                                     if self.role_to_behav is not None else
                                                     str(HybridStateMachine(None))},
                                        content_type=Msg.ROLE_SUGGESTION):
                self.err("Failed to send role change, removing (disconnecting) " + peer_id)
                self._node_purge_fcn(peer_id)
            else:
                self.role_changed_by_world = True

    def set_addresses_in_profile(self, peer_id, addresses):
        """Updates the network addresses in an agent's profile.

        Args:
            peer_id: The ID of the agent whose profile is being updated.
            addresses: A list of new addresses to set.
        """
        if peer_id in self.all_agents:
            profile = self.all_agents[peer_id]
            addrs = profile.get_dynamic_profile()['private_peer_addresses']
            addrs.clear()  # Warning: do not allocate a new list, keep the current one (it is referenced by others)
            for _addrs in addresses:
                addrs.append(_addrs)
            self.received_address_update = True
        else:
            self.err(f"Cannot set addresses in profile, unknown peer_id {peer_id}")

    def add_badge(self, peer_id, score: float, badge_type: str, agent_token: str,
                  badge_description: Optional[str] = None):
        """Requests a badge for a specific agent, which can be used to track and reward agent performance.
        It validates the score and badge type and stores the badge information in an internal dictionary.

        Args:
            peer_id: The ID of the agent for whom the badge is requested.
            score: The score associated with the badge (must be in [0, 1]).
            badge_type: The type of badge to be awarded.
            agent_token: The token of the agent receiving the badge.
            badge_description: An optional text description for the badge.
        """

        # Validate score
        if score < 0. or score > 1.:
            raise ValueError(f"Score must be in [0.0, 1.0], got {score}")

        # Validate badge_type
        if badge_type not in AgentBasics.BADGE_TYPES:
            raise ValueError(f"Invalid badge_type '{badge_type}'. Must be one of {AgentBasics.BADGE_TYPES}.")

        if badge_description is None:
            badge_description = ""

        # The world not necessarily knows the token of the agents, since they usually do not send messages to the world
        badge = {
            'agent_node_id': self.all_agents[peer_id].get_static_profile()['node_id'],
            'agent_token': agent_token,
            'badge_type': badge_type,
            'score': score,
            'badge_description': badge_description,
            'last_edit_utc': self._node_clock.get_time_as_string(),
        }

        if peer_id not in self.agent_badges:
            self.agent_badges[peer_id] = [badge]
        else:
            self.agent_badges[peer_id].append(badge)

    # Get all the badges requested by the world
    def get_all_badges(self):
        """Retrieves all badges that have been added to the world's record for all agents.
        This provides a central log of achievements or performance metrics.

        Returns:
            A dictionary where keys are agent peer IDs and values are lists of badge dictionaries.
        """
        return self.agent_badges

    def clear_badges(self):
        """Clears all badge records from the world's memory.
        This can be used to reset competition results or clean up state after a specific event.
        """
        self.agent_badges = {}
