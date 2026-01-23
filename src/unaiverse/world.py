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
from unaiverse.stats import Stats
from typing import List, Dict, Any
from unaiverse.agent import AgentBasics
from unaiverse.hsm import HybridStateMachine
from unaiverse.networking.p2p.messages import Msg
from unaiverse.networking.node.profile import NodeProfile


class World(AgentBasics):

    def __init__(self, world_folder: str, merge_flat_stream_labels: bool = False, stats: Stats | None = None):
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

        # Map from public peer IDs to private peer IDs
        self.private_peer_of = {}
        
        # Stats
        if stats is not None:
            self.stats = stats
        else:
            # fallback to default Stats class
            self.stats = Stats(is_world=True, db_path=f"{self.world_folder}/stats/world_stats.db",
                               cache_window_hours=2.0)

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

    async def set_role(self, peer_id: str, role: int):
        """Sets a new role for a specific agent and broadcasts this change to the agent (async).

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
            if not (await self._node_conn.send(peer_id, channel_trail=None,
                                               content={'peer_id': peer_id, 'role': new_role,
                                                        'default_behav':
                                                            self.role_to_behav[
                                                                self.ROLE_BITS_TO_STR[new_role_without_base_int]]
                                                            if self.role_to_behav is not None else
                                                            str(HybridStateMachine(None))},
                                               content_type=Msg.ROLE_SUGGESTION)):
                self.err("Failed to send role change, removing (disconnecting) " + peer_id)
                await self._node_purge_fcn(peer_id)
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

    async def add_agent(self, peer_id: str, profile: NodeProfile) -> bool:
        if await super().add_agent(peer_id, profile):
            public_peer_id = profile.get_dynamic_profile()["peer_id"]
            self.private_peer_of[public_peer_id] = peer_id
            return True
        else:
            return False

    async def remove_agent(self, peer_id: str):
        profile = None
        if peer_id in self.all_agents:
            profile = self.all_agents[peer_id]
        if await super().remove_agent(peer_id):
            public_peer_id = profile.get_dynamic_profile()["peer_id"]
            if public_peer_id in self.private_peer_of:
                del self.private_peer_of[public_peer_id]
            return True
        else:
            return False

    def collect_and_store_own_stats(self):
        """Collects this world's own stats and pushes them to the stats recorder."""
        if self.stats is None:
            return
        
        t = self._node_clock.get_time_ms()
        _, own_private_pid = self.get_peer_ids()
        
        # Helper to add if value changed
        def store_if_changed(stat_name, new_value):
            last_value = self.stats.get_last_value(stat_name)
            if last_value != new_value:
                # Note: We pass the world's *own* peer_id for its *own* stats
                self.stats.store_stat(stat_name, new_value, peer_id=own_private_pid, timestamp=t)
        
        try:
            store_if_changed("world_masters", len(self.world_masters))
            store_if_changed("world_agents", len(self.world_agents))
            store_if_changed("human_agents", len(self.human_agents))
            store_if_changed("artificial_agents", len(self.artificial_agents))
        except Exception as e:
            self.err(f"[Stats] Error updating own world stats: {e}")
    
    def _process_custom_stat(self, stat_name, value, peer_id, timestamp) -> bool:
        """Hook for subclasses to intercept a stat. Return True if handled."""
        return False
    
    def _extract_graph_node_info(self, peer_id: str) -> Dict[str, Any]:
        """Helper to extract lightweight visualization data from NodeProfile."""

        if peer_id == self.get_peer_ids()[1]:
            # this is the world itself
            profile = self._node_profile
        else:
            profile = self.all_agents.get(peer_id)
        if profile is None:
            return {}
        
        # Accessing the inner private dict of NodeProfile based on your class structure
        static_profile = profile.get_static_profile()
        dynamic_profile = profile.get_dynamic_profile()
        
        return {
            'Name': static_profile.get('node_name', '~'),
            'Owner': static_profile.get('email', '~'),
            'Role': dynamic_profile.get('connections', {}).get('role', 'unknown').split('~')[-1],
            'Type': static_profile.get('node_type', '~'),
            'Number of Badges': len(dynamic_profile.get('cv', [])),
            'Current Action': self.stats.get_last_value('action', peer_id=peer_id) or '~',
            'Current State': self.stats.get_last_value('state', peer_id=peer_id) or '~',
        }
    
    def _update_graph(self, peer_id: str, connected_peers_list: List[str], timestamp: int):
        """Updates both graph connectivity (edges) and node metadata."""
        
        # 1. initialize structure if missing (e.g. first run or after DB load)
        graph_stat = self.stats.get_stats().setdefault("graph", {'nodes': {}, 'edges': {}})
            
        nodes = graph_stat.setdefault('nodes', {})
        edges = graph_stat.setdefault('edges', {})
        
        # 2. Update Node Metadata
        # We update the sender's info
        node_data = self._extract_graph_node_info(peer_id)
        node_data['last_seen'] = timestamp
        nodes[peer_id] = node_data

        # We also ensure connected peers exist in 'nodes', even if we don't have their full profile yet
        connected_peers = set(connected_peers_list)
        for target_id in connected_peers:
            if target_id not in nodes:
                # Try to fetch profile if we have it, otherwise placeholder
                nodes[target_id] = self._extract_graph_node_info(target_id)

        # 3. Update Edges (Logic adapted from your previous code)
        prev_connected_peers = edges.setdefault(peer_id, set())

        # Add reverse connections (Undirected/Bidirectional logic)
        for _peer_id in connected_peers:
            edges.setdefault(_peer_id, set()).add(peer_id)
        
        # Remove dropped reverse connections
        to_remove = prev_connected_peers - connected_peers
        for _peer_id in to_remove:
            if _peer_id in edges and peer_id in edges[_peer_id]:
                edges[_peer_id].remove(peer_id)
        
        # Update peer's own forward connections
        edges[peer_id] = connected_peers
        
        # 4. Store
        world_peer_id = self.get_peer_ids()[1]
        self.stats.store_stat('graph', graph_stat, peer_id=world_peer_id, timestamp=timestamp)
    
    def _prune_graph(self):
        """Removes nodes that are no longer connected to the World."""
        graph_stat = self.stats.get_stats().get("graph")
        if not graph_stat:
            return

        nodes = graph_stat.get('nodes', {})
        edges = graph_stat.get('edges', {})

        # Get the active/inactive peers
        active_peers = set(self.all_agents.keys())
        # active_peers_in_world = set(self.world_agents.keys()) | set(self.world_masters.keys())
        active_peers.add(self.get_peer_ids()[1])
        current_graph_nodes = set(nodes.keys())
        dead_peers = current_graph_nodes - active_peers

        if not dead_peers:
            return

        # 2. Kill Zombies
        for pid in dead_peers:
            nodes.pop(pid, None)  # Nodes
            edges.pop(pid, None)  # Outgoing edges

            # Remove incoming edges
            for other_pid in edges:
                edges[other_pid].discard(pid)
    
    def add_peer_stats(self, peer_stats_batch: List[Dict[str, Any]], sender_peer_id: str | None = None):
        """(World-only) Processes a batch of stats received from a peer."""
        
        # 1. Update own stats (this logic is now in the World)
        self.collect_and_store_own_stats()

        # 2. Process peer stats
        connected_peers = []
        for update in peer_stats_batch:
            try:
                p_id = update['peer_id']
                if p_id != sender_peer_id:
                    # TODO: decide if we want to filter the stats
                    pass
                stat_name = update['stat_name']
                t = int(update['timestamp'])
                v = update['value']
                
                # Call the hook (which also lives in the World now)
                if self._process_custom_stat(stat_name, v, p_id, t):
                    continue  # The custom processor handled it
                
                # Generate the graph and handle the connected_peers stat
                if stat_name == 'connected_peers':
                    # We need to wait for all the info to arrive before updating the graph.
                    # Otherwise, _extract_graph_node_info may not find data yet.
                    connected_peers.append((p_id, v, t))
                    continue

                # 3. Push to the "dumb" Stats recorder
                if stat_name in self.stats.all_keys:
                    self.stats.store_stat(stat_name, v, peer_id=p_id, timestamp=t)
                else:
                    self.err(f"[World] Unknown stat received: {stat_name}")

            except Exception as e:
                self.err(f"[World] Error processing stats update {update}: {e}")
        
        # Now update the graph for all collected connected_peers stats
        for p_id, v, t in connected_peers:
            self._update_graph(p_id, v, t)
        
        # Clean the graph from potentially stale peers
        self._prune_graph()
    
    def debug_stats_dashboard(self):
        """Helper to verify the dashboard looks correct during development."""
        import plotly.io as pio
        
        print("[DEBUG] Rendering Dashboard...")
        json_str = self.stats.plot()
        if json_str:
            pio.from_json(json_str).show()
