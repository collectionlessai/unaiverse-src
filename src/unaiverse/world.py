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
import json
import math
import bisect
from unaiverse.agent import AgentBasics
from unaiverse.hsm import HybridStateMachine
from unaiverse.networking.p2p.messages import Msg
from unaiverse.networking.node.profile import NodeProfile


class World(AgentBasics):

    def __init__(self, world_folder: str, merge_flat_stream_labels: bool = False):
        """Initializes a World object, which acts as a special agent without a processor or behavior.

        Args:
            world_folder: The path of the world folder, with JSON files of the behaviors (per role) and agent.py.
        """

        # Creating a "special" agent with no processor and no behavior, but with a "world_folder", which is our world
        super().__init__(proc=None, proc_inputs=None, proc_outputs=None, proc_opts=None, behav=None,
                         world_folder=world_folder, merge_flat_stream_labels=merge_flat_stream_labels)

        # Clearing processor (world must have no processor, and, maybe, a dummy processor was allocated when building
        # the agent in the init call above)
        self.proc = None
        self.proc_inputs = []  # Do not set it to None
        self.proc_outputs = []  # Do not set it to None
        self.compat_in_streams = None
        self.compat_out_streams = None

    def assign_role(self, profile: NodeProfile, is_world_master: bool) -> str:
        """Assigns an initial role to a newly connected agent.

        In this basic implementation, the role is determined based on whether the agent is a world master or a regular
        world agent, ensuring there's only one master.

        Args:
            profile: The NodeProfile of the new agent.
            is_world_master: A boolean indicating if the new agent is attempting to be a master.

        Returns:
            A string representing the assigned role.
        """
        assert self.is_world, "Assigning a role is expected to be done by the world"

        if profile.get_dynamic_profile()['guessed_location'] == 'Some Dummy Location, Just An Example Here':
            pass

        # Currently, roles are only world masters and world agents
        if is_world_master:
            if len(self.world_masters) <= 1:
                return AgentBasics.ROLE_BITS_TO_STR[AgentBasics.ROLE_WORLD_MASTER]
            else:
                return AgentBasics.ROLE_BITS_TO_STR[AgentBasics.ROLE_WORLD_AGENT]
        else:
            return AgentBasics.ROLE_BITS_TO_STR[AgentBasics.ROLE_WORLD_AGENT]

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

    def add_badge(self, peer_id: str, score: float, badge_type: str, agent_token: str,
                  badge_description: str | None = None):
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

        # This will force the sending of the dynamic profile at the defined time instants
        self._node_profile.mark_change_in_connections()

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

    def get_stats(self, from_timestamp: float | None = None,
                  compute_aggregated_custom_stats: bool = False) -> dict:
        # TODO: filter by peer?
        full_stats_requested = from_timestamp is None or from_timestamp < 0.

        if full_stats_requested:
            stats = self.stats
        else:
            sliced_stats = {
                'peer_graph': self.stats['peer_graph'],
                'peer_stats': {}
            }
            world_peer_stats = self.stats['peer_stats']
            sliced_peer_stats = sliced_stats['peer_stats']
            for peer_id, stat_name_tv_dict in world_peer_stats.items():
                sliced_stat_name_tv_dict = {}
                sliced_peer_stats[peer_id] = sliced_stat_name_tv_dict
                for stat_name, tv_dict in stat_name_tv_dict.items():
                    ts = list(tv_dict.keys())
                    vs = list(tv_dict.values())
                    i = bisect.bisect_left(ts, from_timestamp)
                    sliced_stat_name_tv_dict[stat_name] = dict(zip(ts[i:], vs[i:]))
            stats = sliced_stats

        if compute_aggregated_custom_stats and len(self.WORLD_STATS_DYNAMIC_BY_PEER) > 0:
            mean, std = self.__aggregate_time_indexed_stats_over_peers(stats)
            stats['peer_stats']['<mean>'] = mean
            stats['peer_stats']['<std>'] = std

        # TODO: remove this, debug only
        import os
        t = self._node_clock.get_time_as_string()
        json.dump(stats, open(os.path.join(self.world_folder, f"[{t}]_get_stats.json")), indent=4)

        return stats

    def add_stats_from_peer(self, peer_id: str, peer_stats: dict):
        t = self._node_clock.get_time()

        # World stats
        if len(self.world_masters) != next(reversed(self.stats["world_masters"].values()), -1):
            self.stats["world_masters"][t] = len(self.world_masters)
        if len(self.world_agents) != next(reversed(self.stats["world_agents"].values()), -1):
            self.stats["world_agents"][t] = len(self.world_agents)
        if len(self.human_agents) != next(reversed(self.stats["human_agents"].values()), -1):
            self.stats["human_agents"][t] = len(self.human_agents)
        if len(self.artificial_agents) != next(reversed(self.stats["artificial_agents"].values()), -1):
            self.stats["artificial_agents"][t] = len(self.artificial_agents)

        # Static stats, without groups (i.e., the peer graph)
        graph = self.stats["graph"]
        if peer_id not in graph:
            graph[peer_id] = set()
        prev_connected_peers = graph[peer_id]
        graph_changed = False
        if "connected_peers" in peer_stats:
            connected_peers = set(peer_stats["connected_peers"])
            to_remove = prev_connected_peers - connected_peers
            for _peer_id in connected_peers:
                if _peer_id in graph:
                    if peer_id not in graph[_peer_id]:
                        graph[_peer_id].add(peer_id)
                        graph_changed = True
                else:
                    graph[peer_id] = {peer_id}
                    graph_changed = True
            for _peer_id in to_remove:
                if _peer_id in graph:
                    if peer_id in graph[_peer_id]:
                        graph[_peer_id].remove(peer_id)
                        graph_changed = True
        if graph_changed:
            self.stats_loader_saver.mark_stat_as_changed("graph")

        # Static stats, per peer
        stats_per_peer = self.stats["stats_per_peer"]
        if peer_id not in stats_per_peer:
            stats_per_peer[peer_id] = {"state": None, "action": None, "last_action": None}
        for stat_name, v in peer_stats.items():
            if stat_name in self.WORLD_STATS_STATIC_BY_PEER:
                if v != stats_per_peer[peer_id][stat_name]:
                    stats_per_peer[peer_id][stat_name] = v
                    self.stats_loader_saver.mark_stat_as_changed(stat_name)

        # Dynamic stats, per peer (appending them)
        for stat_name, tv_dict in peer_stats.items():
            if stat_name in self.WORLD_STATS_DYNAMIC_BY_PEER.keys():
                expected_type = self.WORLD_STATS_DYNAMIC_BY_PEER[stat_name]["type"]
                if stat_name not in stats_per_peer[peer_id]:
                    stats_per_peer[peer_id][stat_name] = {}
                tv_dict_clean = {}
                if len(stats_per_peer[peer_id][stat_name]) > 0:
                    last_t = next(reversed(stats_per_peer[peer_id][stat_name].keys()))
                else:
                    last_t = -1.
                for t, v in tv_dict.items():
                    if (isinstance(t, float) and t > 0. and
                            ((isinstance(v, float) or isinstance(v, int) and expected_type == 'number') or
                             (isinstance(v, str) and expected_type == 'str')) and t > last_t):
                        tv_dict_clean[t] = v if expected_type != "number" else float(v)
                stats_per_peer[peer_id][stat_name].update(tv_dict_clean)

        # TODO: remove this, debug only
        import os
        t = self._node_clock.get_time_as_string()
        json.dump(self.stats, open(os.path.join(self.world_folder, f"[{t}]_add_stats_from_peer.json")), indent=4)
        self.get_stats(None, compute_aggregated_custom_stats=True)
        self.get_stats(self._node_clock.get_time() - 100.0, compute_aggregated_custom_stats=True)

    def __aggregate_time_indexed_stats_over_peers(self, stats: dict) -> tuple[dict, dict]:
        """Aggregate all time-indexed peer stats by aligning timestamps across peers, forward-filling missing values,
        and computing mean + std at each timestamp.

        Args:
            stats: dictionary with structure stats['peer_stats'][peer_id][custom_key] = {timestamp: value}.

        Returns:
            Two dictionaries, mean_dict, std_dict where: mean_dict[custom_key][timestamp] = mean_value,
            std_dict[custom_key][timestamp] = std_value
        """
        mean_dict = {}
        std_dict = {}
        peer_stats = stats.get("peer_stats", {})

        for custom_key in self.WORLD_STATS_DYNAMIC_BY_PEER.keys():

            # Skipping strings: we can only aggregate numbers
            if self.WORLD_STATS_DYNAMIC_BY_PEER[custom_key]['type'] != 'number':
                continue

            # Collecting time series
            peer_series = []
            for peer_id, peer_data in peer_stats.items():
                if custom_key in peer_data:
                    tv_dict = peer_data[custom_key]
                    if tv_dict:
                        peer_series.append(tv_dict)

            if not peer_series:
                continue

            # Merging all timestamps
            all_times = sorted({t for series in peer_series for t in series.keys()})  # Set to list

            # Forward filling of the timestamp that are not present in a series
            aligned_values = []
            for series in peer_series:
                if len(series) == 0:
                    continue
                filled = []
                last_val = next(iter(series.values()))  # Get the first value (going ahead...exception)
                for t in all_times:
                    if t in series:
                        last_val = series[t]
                    filled.append(last_val)
                aligned_values.append(filled)

            # Aggregate (mean + std) at each timestamp
            mean_dict[custom_key] = {}
            std_dict[custom_key] = {}
            for i, t in enumerate(all_times):
                vals = [peer_vals[i] for peer_vals in aligned_values if peer_vals[i] is not None]
                if vals:
                    mean_val = sum(vals) / float(len(vals))
                    var = sum((x - mean_val) ** 2 for x in vals) / len(vals)
                    std_val = math.sqrt(var)
                else:
                    mean_val = None
                    std_val = None

                mean_dict[custom_key][t] = mean_val
                std_dict[custom_key][t] = std_val

        return mean_dict, std_dict
