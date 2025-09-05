import os
import types
from typing import Optional
from unaiverse.agent import AgentBasics
from unaiverse.hsm import HybridStateMachine
from unaiverse.networking.p2p.messages import Msg
from unaiverse.networking.node.profile import NodeProfile


class World(AgentBasics):

    def __init__(self, *args, **kwargs):

        # deleting the "proc" parameter, if provided (it is the second one)
        if len(args) == 2:
            args = (args[0])

        # clearing keyword-level processor-related attributes
        kwargs['proc'] = None  # processor = None (existing)
        kwargs['proc_inputs'] = None
        kwargs['proc_outputs'] = None
        kwargs['proc_opts'] = None

        # clearing behavior
        kwargs['behav'] = None

        # removing world-specific arguments
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

        # creating a "special" agent with no processor and no behavior, which is our world
        super().__init__(*args, **kwargs)

        # clearing processor (world must have no processor, and, maybe, a dummy processor was allocated when building
        # the agent in the init call above)
        self.proc = None
        self.proc_inputs = []  # do not set it to None
        self.proc_outputs = [] # do not set it to None
        self.compat_in_streams = None
        self.compat_out_streams = None

        # world specific attributes
        self.agent_badges: dict[str, list[dict]] = {}  # peer_id -> collected badges for other agents
        self.role_changed_by_world: bool = False
        self.received_address_update: bool = False

        # loading agent (actions) file
        if agent_actions_file is not None:
            path_of_this_file = str(os.path.dirname(os.path.abspath(__file__)))
            with open(os.path.join(path_of_this_file, 'library', 'worlds', agent_actions_file),
                      'r', encoding='utf-8') as file:
                self.agent_actions = file.read()
        else:
            self.agent_actions = ""  # empty string

        # loading default behaviours
        self.role_to_behav = {}
        if role_to_behav_files is not None:

            # creating a dummy agent which supports the actions of the following state machines
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
        assert self.is_world, "Assigning a role is expected to be done by the world"

        if profile.get_dynamic_profile()['guessed_location'] == 'Some Dummy Location, Just An Example Here':
            pass

        # currently, roles are only world masters and world agents
        if is_world_master:
            if len(self.world_masters) <= 1:
                return AgentBasics.ROLE_WORLD_MASTER
            else:
                return AgentBasics.ROLE_WORLD_AGENT
        else:
            return AgentBasics.ROLE_WORLD_AGENT

    def set_role(self, peer_id: str, role: int):
        assert self.is_world, "Setting the role is expected to be done by the world, which will broadcast such info"

        # computing new role (keeping the first two bits as before)
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
        if peer_id in self.all_agents:
            profile = self.all_agents[peer_id]
            addrs = profile.get_dynamic_profile()['private_peer_addresses']
            addrs.clear()  # warning: do not allocate a new list, keep the current one (it is referenced by others)
            for _addrs in addresses:
                addrs.append(_addrs)
            self.received_address_update = True
        else:
            self.err(f"Cannot set addresses in profile, unknown peer_id {peer_id}")

    def add_badge(self, peer_id, score: float, badge_type: str, agent_token: str,
                  badge_description: Optional[str] = None):
        """Request a badge for an agent.

        Args:
            peer_id: agent peer_id for which the badge is requested.
            score: must be in [0,1]
            badge_type: must be one of 'completed', 'intermediate', 'attended', 'pro'.
            agent_token: token of the agent that will receive this badge.
            badge_description: optional text description defining the badge.
        """

        # validate score
        if score < 0. or score > 1.:
            raise ValueError(f"Score must be in [0.0, 1.0], got {score}")
        
        # Validate badge_type
        if badge_type not in AgentBasics.BADGE_TYPES:
            raise ValueError(f"Invalid badge_type '{badge_type}'. Must be one of {AgentBasics.BADGE_TYPES}.")

        if badge_description is None:
            badge_description = ""

        # the world not necessarily knows the token of the agents, since they usually do not send messages to the world
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
    
    # get all the badges requested by the world
    def get_all_badges(self):
        return self.agent_badges

    def clear_badges(self):
        self.agent_badges = {}
