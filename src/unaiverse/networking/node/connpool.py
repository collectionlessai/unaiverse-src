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
import math
from unaiverse.networking.p2p.messages import Msg
from unaiverse.networking.p2p.p2p import P2P, P2PError
from unaiverse.networking.node.tokens import TokenVerifier


class ConnectionPools:
    DEBUG = True

    def __init__(self, max_connections: int, pool_name_to_p2p_name_and_ratio: dict[str, [str, float]],
                 p2p_name_to_p2p: dict[str, P2P], public_key: str | None = None, token: str | None = None):
        """Initializes a new instance of the ConnectionPools class.

        Args:
            max_connections: The maximum total number of connections allowed across all pools.
            pool_name_to_p2p_name_and_ratio: A dictionary mapping pool names to a list containing the associated P2P network name and its connection ratio.
            p2p_name_to_p2p: A dictionary mapping P2P network names to their corresponding P2P objects.
            public_key: An optional public key for token verification.
            token: An optional initial token for authentication.

        Returns:
            None.

        """
        # Common terms: a "pool triple" is [pool_contents, max_connections_in_such_a_pool, p2p_object_of_the_pool
        self.max_con = max_connections
        self.pool_count = len(pool_name_to_p2p_name_and_ratio)
        self.pool_names = list(pool_name_to_p2p_name_and_ratio.keys())
        self.pool_ratios = [p2p_name_and_ratio[1] for p2p_name_and_ratio in pool_name_to_p2p_name_and_ratio.values()]

        # Indices involving the P2P object or its name
        self.p2p_name_to_p2p = p2p_name_to_p2p
        self.p2p_name_and_pool_name_to_pool_triple = {}
        self.p2p_to_pool_names = {}

        # Indices rooted around the pool name
        self.pool_name_to_pool_triple = {}
        self.pool_name_to_added_in_last_update = {}
        self.pool_name_to_removed_in_last_update = {}
        self.pool_name_to_peer_infos = {p: {} for p in self.pool_names}

        # Indices rooted around the peer ID
        self.peer_id_to_pool_name = {}
        self.peer_id_to_p2p = {}
        self.peer_id_to_misc = {}
        self.peer_id_to_token = {}

        # Token-related stuff, super private
        self.__token = token if token is not None else ""
        self.__token_verifier = TokenVerifier(public_key) if public_key is not None else None

        # Checking
        for p2p_name_and_ratio in pool_name_to_p2p_name_and_ratio.values():
            assert p2p_name_and_ratio[0] in self.p2p_name_to_p2p, f"Cannot find p2p named {p2p_name_and_ratio[0]} "
        assert self.max_con >= len(self.pool_names), "Too small number of max connections"
        assert sum([x for x in self.pool_ratios if x > 0]) == 1.0, "Pool ratios must sum to 1.0"

        # Preparing the pool triples
        self.pool_name_to_pool_triple = \
            {k: [set(), 0, self.p2p_name_to_p2p[pool_name_to_p2p_name_and_ratio[k][0]]] for k in self.pool_names}
        num_zero_ratio_pools = len([x for x in self.pool_ratios if x == 0])
        assert num_zero_ratio_pools <= self.max_con, "Cannot create pools given the provided max connection count"

        # Edit: to solve the teacher not engaging with more than two students.
        pools_max_sizes = {k: max(math.floor(self.pool_ratios[i] * (self.max_con - num_zero_ratio_pools)),
                                  1 if self.pool_ratios[i] >= 0. else 0)
                           for i, k in enumerate(self.pool_names)}

        # Pools_max_sizes = {k: self.max_con for k in self.pool_names}

        # Fixing sizes
        tot = 0
        for i, (k, v) in enumerate(pools_max_sizes.items()):
            assert v > 0 or self.pool_ratios[i] < 0, "Cannot create pools given the provided max connection count"

        # Edit: to solve the teacher not engaging with more than two students.
            tot += v
        assert tot <= self.max_con, \
            "Cannot create pools given the provided max connection count"
        pools_max_sizes[self.pool_names[-1]] += (self.max_con - tot)

        # Storing fixed sizes in the previously created pool triples & building additional index
        for pool_name, pool_contents_max_con_and_p2p in self.pool_name_to_pool_triple.items():
            pool_contents_max_con_and_p2p[1] = pools_max_sizes[pool_name]  # Fixing the second element of the triple

            pool, _, p2p = pool_contents_max_con_and_p2p
            p2p_name = None
            for k, v in self.p2p_name_to_p2p.items():
                if v == p2p:
                    p2p_name = k
                    break
            if p2p_name not in self.p2p_name_and_pool_name_to_pool_triple:
                self.p2p_name_and_pool_name_to_pool_triple[p2p_name] = {}
                self.p2p_to_pool_names[p2p] = []
            self.p2p_name_and_pool_name_to_pool_triple[p2p_name][pool_name] = (
                pool_contents_max_con_and_p2p)
            self.p2p_to_pool_names[p2p].append(pool_name)

    def __str__(self):
        """Returns a human-readable string representation of the connection pools' status.

        Args:
            None.

        Returns:
            A formatted string showing the number of connections and peer IDs in each pool.
        """
        max_len = max(len(s) for s in self.pool_names)
        s = f"[ConnectionPool] max_con={self.max_con}, pool_count={self.pool_count}"
        for k, (pool, max_con, _) in self.pool_name_to_pool_triple.items():
            kk = k + ","
            s += f"\n\tpool={kk.ljust(max_len + 1)}\tmax_con={max_con},\tcurrent_con={len(pool)},\tpeer_ids="
            ss = "("
            first = True
            for c in pool:
                if not first:
                    ss += ", "
                ss += c
                first = False
            ss += ")"
            s += ss
        return s

    def __getitem__(self, p2p_name):
        """Retrieves a P2P object by its name.

        Args:
            p2p_name: The name of the P2P network.

        Returns:
            The P2P object.
        """
        return self.p2p_name_to_p2p[p2p_name]

    def conn_routing_fcn(self, connected_peer_infos: list, p2p: P2P):
        """A placeholder function that must be implemented to route connected peers to the correct pool.

        Args:
            connected_peer_infos: A list of dictionaries containing information about connected peers.
            p2p: The P2P network object from which the peers were connected.

        Returns:
            A dictionary mapping pool names to a dictionary of peer IDs and their information.
        """
        raise NotImplementedError("You must implement conn_routing_fcn!")

    @staticmethod
    def __connect(p2p: P2P, addresses: list[str]):
        """Establishes a connection to a peer via a P2P network.

        Args:
            p2p: The P2P network object to use for the connection.
            addresses: A list of addresses of the peer to connect to.

        Returns:
            A tuple containing the peer ID and a boolean indicating if the connection was established through a relay.
        """
        if ConnectionPools.DEBUG:
            print(f"[DEBUG CONNECTIONS-POOL] Connecting to {addresses}")

        try:
            winning_addr_info_dict = p2p.connect_to(addresses)
            peer_id = winning_addr_info_dict.get('ID')
            connected_addr_str = winning_addr_info_dict.get('Addrs')[0]
            through_relay = '/p2p-circuit/' in connected_addr_str
            if ConnectionPools.DEBUG:
                print(f"[DEBUG CONNECTIONS-POOL] Connected to peer {peer_id} via {connected_addr_str} "
                      f"(through relay: {through_relay})")

            return peer_id, through_relay
        except P2PError:
            if ConnectionPools.DEBUG:
                print(f"[DEBUG CONNECTIONS-POOL] Connection failed!")
            return None, False

    @staticmethod
    def disconnect(p2p: P2P, peer_id: str):
        """Disconnects from a specific peer on a P2P network.

        Args:
            p2p: The P2P network object to use for disconnection.
            peer_id: The peer ID to disconnect from.

        Returns:
            True if the disconnection is successful, otherwise False.
        """
        try:
            p2p.disconnect_from(peer_id)
        except P2PError:
            return False
        return True

    def set_token(self, token: str):
        """Sets the authentication token for the connection pools.

        Args:
            token: The new token string.
        """
        self.__token = token

    def verify_token(self, token: str, peer_id: str):
        """Verifies a received token using the provided public key.

        Args:
            token: The token string to verify.
            peer_id: The peer ID associated with the token.

        Returns:
            A tuple containing the node ID and CV hash if the token is valid, otherwise None.
        """
        if self.__token_verifier is None:
            return None
        else:
            node_id, cv_hash = self.__token_verifier.verify_token(token, p2p_peer=peer_id)
            return node_id, cv_hash  # If the verification fails, this is None, None

    def connect(self, addresses: list[str], p2p_name: str):
        """Connects to a peer on a specified P2P network.

        Args:
            addresses: A list of addresses of the peer to connect to.
            p2p_name: The name of the P2P network to use.

        Returns:
            A tuple containing the peer ID of the connected peer and a boolean indicating if a relay was used.
        """
        p2p = self.p2p_name_to_p2p[p2p_name]

        # Connecting
        peer_id, through_relay = ConnectionPools.__connect(p2p, addresses)
        return peer_id, through_relay

    def add(self, peer_info: dict, pool_name: str):
        """Adds a connected peer to a specified connection pool.

        Args:
            peer_info: A dictionary containing information about the peer.
            pool_name: The name of the pool to add the peer to.

        Returns:
            True if the peer is successfully added, otherwise False.
        """
        peer_id = peer_info['id']
        pool, max_size, p2p = self.pool_name_to_pool_triple[pool_name]
        if len(pool) < max_size:

            # "hoping" peer IDs are unique, and stopping duplicate cases
            if peer_id in self.peer_id_to_pool_name and self.peer_id_to_pool_name[peer_id] != pool_name:
                return False

            self.peer_id_to_pool_name[peer_id] = pool_name
            self.peer_id_to_p2p[peer_id] = p2p

            # Setting 'misc' field (default is 0, where 0 means public)
            peer_info['misc'] = self.peer_id_to_misc.get(peer_id, 0)

            # Storing (only)
            pool.add(peer_id)
            self.pool_name_to_peer_infos[pool_name][peer_id] = peer_info
            return True
        else:
            return False

    def remove(self, peer_id: str):
        """Removes a peer from its connection pool and disconnects from it.

        Args:
            peer_id: The peer ID to remove.

        Returns:
            True if the peer is successfully removed, otherwise False.
        """
        if peer_id in self.peer_id_to_pool_name:
            pool_name = self.peer_id_to_pool_name[peer_id]
            pool, _, p2p = self.pool_name_to_pool_triple[pool_name]

            # Disconnecting
            disc = ConnectionPools.disconnect(p2p, peer_id)
            pool.remove(peer_id)
            del self.pool_name_to_peer_infos[pool_name][peer_id]
            del self.peer_id_to_pool_name[peer_id]
            del self.peer_id_to_p2p[peer_id]
            if peer_id in self.peer_id_to_misc:
                del self.peer_id_to_misc[peer_id]
            if peer_id in self.peer_id_to_token:
                del self.peer_id_to_token[peer_id]
            return disc
        else:
            return False

    def get_all_connected_peer_infos(self, pool_name: str):
        """Retrieves a list of peer information dictionaries for a given pool.

        Args:
            pool_name: The name of the pool to query.

        Returns:
            A list of dictionaries, each containing information about a peer in the pool.
        """
        return list(self.pool_name_to_peer_infos[pool_name].values())

    def get_pool_status(self):
        """Returns a dictionary showing the set of peer IDs in each pool.

        Args:
            None.

        Returns:
            A dictionary mapping pool names to the set of peer IDs in that pool.
        """
        return {k: v[0] for k, v in self.pool_name_to_pool_triple.items()}

    def get_all_connected_peer_ids(self):
        """Retrieves a list of all peer IDs currently connected across all pools.

        Args:
            None.

        Returns:
            A list of all connected peer IDs.
        """
        return list(self.peer_id_to_pool_name.keys())

    def update(self):
        """Refreshes the connection pools by checking for new and lost connections.

        Args:
            None.

        Returns:
            A tuple containing two dictionaries: one for newly added peers and one for removed peers, both keyed by
            pool name.
        """
        self.pool_name_to_added_in_last_update = {}
        self.pool_name_to_removed_in_last_update = {}

        for p2p_name, p2p in self.p2p_name_to_p2p.items():
            connected_peer_infos = p2p.get_connected_peers_info()

            if connected_peer_infos is not None:

                # Routing to the right queue / filtering
                pool_name_and_peer_ids_to_peer_info = self.conn_routing_fcn(connected_peer_infos, p2p)

                # Parsing the generated index
                for pool_name, connected_peer_ids_to_connected_peer_infos \
                        in pool_name_and_peer_ids_to_peer_info.items():
                    pool, _, pool_p2p = self.p2p_name_and_pool_name_to_pool_triple[p2p_name][pool_name]
                    connected_peer_ids = connected_peer_ids_to_connected_peer_infos.keys()
                    new_peer_ids = connected_peer_ids - pool
                    lost_peer_ids = pool - connected_peer_ids

                    # Clearing disconnected agents
                    for lost_peer_id in lost_peer_ids:
                        self.pool_name_to_removed_in_last_update.setdefault(pool_name, set()).add(lost_peer_id)

                    # Adding new agents
                    for new_peer_id in new_peer_ids:
                        peer_info = connected_peer_ids_to_connected_peer_infos[new_peer_id]
                        if not self.add(peer_info, pool_name=pool_name):
                            break
                        self.pool_name_to_added_in_last_update.setdefault(pool_name, set()).add(new_peer_id)

        return self.pool_name_to_added_in_last_update, self.pool_name_to_removed_in_last_update

    def get_messages(self, p2p_name: str, allowed_not_connected_peers: set | None = None):
        """Retrieves and verifies all messages from a specified P2P network.

        Args:
            p2p_name: The name of the P2P network to fetch messages from.
            allowed_not_connected_peers: An optional set of peer IDs to allow messages from, even if they are not in the pools.

        Returns:
            A list of verified and processed message objects.
        """
        # Pop all messages
        messages: list[Msg] = self[p2p_name].pop_messages()  # Pop all messages (list of messages - list[Msg])
        ret = []
        for m in messages:
            if (m.sender in self.peer_id_to_pool_name or  # Check if expected sender
                    (allowed_not_connected_peers is not None and m.sender in allowed_not_connected_peers)):
                try:
                    token_with_inspector_final_bit = m.piggyback
                    token = token_with_inspector_final_bit[0:-1]
                    inspector_mode = token_with_inspector_final_bit[-1]
                    node_id, _ = self.verify_token(token, m.sender)
                    if node_id is not None:

                        # Replacing piggyback with the node ID and the flag telling if it is inspector
                        m.piggyback = node_id + inspector_mode
                        ret.append(m)
                        if m.sender in self.peer_id_to_pool_name:
                            self.peer_id_to_token[m.sender] = token
                    else:
                        print("Received a message missing expected info in the token payload (discarding it)")
                except Exception as e:
                    print(f"Received a message with an invalid piggyback token! (discarding it) [{e}]")
            else:
                if ConnectionPools.DEBUG:
                    print("Received a message from a unknown sender (discarding it)")
        return ret

    def get_added_after_updating(self, pool_name: str | None = None):
        """Retrieves the peers that were added in the last update cycle.

        Args:
            pool_name: The name of a specific pool to query. If None, returns data for all pools.

        Returns:
            A set of added peer IDs for the specified pool, or a dictionary of sets for all pools.
        """
        if pool_name is not None:
            return self.pool_name_to_added_in_last_update[pool_name]
        else:
            return self.pool_name_to_added_in_last_update

    def get_removed_after_updating(self, pool_name: str | None = None):
        """Retrieves the peers that were removed in the last update cycle.

        Args:
            pool_name: The name of a specific pool to query. If None, returns data for all pools.

        Returns:
            A set of removed peer IDs for the specified pool, or a dictionary of sets for all pools.
        """
        if pool_name is not None:
            return self.pool_name_to_removed_in_last_update[pool_name]
        else:
            return self.pool_name_to_removed_in_last_update

    def get_last_token(self, peer_id):
        """Retrieves the last known token for a given peer.

        Args:
            peer_id: The peer ID to query.

        Returns:
            The token string if found, otherwise None.
        """
        return self.peer_id_to_token[peer_id] if peer_id in self.peer_id_to_token else None

    def is_connected(self, peer_id: str, pool_name: str | None = None):
        """Checks if a peer is currently connected, optionally in a specific pool.

        Args:
            peer_id: The peer ID to check.
            pool_name: An optional pool name to check within.

        Returns:
            True if the peer is connected, otherwise False.
        """
        if pool_name is None:
            return peer_id in self.peer_id_to_pool_name
        else:
            return peer_id in self.peer_id_to_pool_name and pool_name == self.peer_id_to_pool_name[peer_id]

    def get_pool_of(self, peer_id: str):
        """Gets the pool name for a given connected peer.

        Args:
            peer_id: The peer ID to query.

        Returns:
            The name of the pool the peer is in.
        """
        return self.peer_id_to_pool_name[peer_id]

    def size(self, pool_name: str | None = None):
        """Returns the number of connections in a specific pool or the total number across all pools.

        Args:
            pool_name: An optional pool name to get the size of. If None, returns the total size.

        Returns:
            The size of the pool or the total number of connections.
        """
        if pool_name is not None:
            return len(self.pool_name_to_pool_triple[pool_name])
        else:
            c = 0
            for v in self.pool_name_to_pool_triple.values():
                c += len(v)
            return c

    def send(self, peer_id: str, channel_trail: str | None,
             content_type: str, content: bytes | dict | None = None, p2p: P2P | None = None):
        """Sends a direct message to a specific peer.

        Args:
            peer_id: The peer ID to send the message to.
            channel_trail: An optional string to append to the channel name.
            content_type: The type of content in the message.
            content: The message content.
            p2p: An optional P2P object to use for sending. If None, it is derived from the peer_id.

        Returns:
            True if the message is sent successfully, otherwise False.
        """
        # Getting the right p2p object
        if p2p is None:
            p2p = self.peer_id_to_p2p[peer_id] if peer_id in self.peer_id_to_p2p else None
            if p2p is None:
                if ConnectionPools.DEBUG:
                    print("[DEBUG CONNECTIONS-POOL] P2P non found for peer id: " + str(peer_id))
                return False

        # Defining channel
        if channel_trail is not None and len(channel_trail) > 0:
            channel = f"{p2p.peer_id}::dm:{peer_id}-{content_type}~{channel_trail}"
        else:
            channel = f"{p2p.peer_id}::dm:{peer_id}-{content_type}"

        # Adding sender info here
        msg = Msg(sender=p2p.peer_id,
                  content_type=content_type,
                  content=content,
                  channel=channel,
                  piggyback=self.__token + "0")  # Adding inspector-mode bit (dummy bit here)
        if ConnectionPools.DEBUG:
            print("[DEBUG CONNECTIONS-POOL] Sending message: " + str(msg))

        # Sending direct message
        try:
            p2p.send_message_to_peer(channel, msg)

            # If the line above executes without raising an error, it was successful.
            return True
        except P2PError as e:

            # If send_message_to_peer fails, it will raise a P2PError. We catch it here.
            if ConnectionPools.DEBUG:
                print("[DEBUG CONNECTIONS-POOL] Sending error is: " + str(e))
            return False

    def subscribe(self, peer_id: str, channel: str, default_p2p_name: str | None = None):
        """Subscribes to a topic/channel on a P2P network.

        Args:
            peer_id: The peer ID associated with the topic/channel.
            channel: The name of the channel to subscribe to.
            default_p2p_name: An optional P2P network name to use if the peer's network is unknown.

        Returns:
            True if the subscription is successful, otherwise False.
        """

        # Getting the right p2p object
        p2p = None
        for _p2p in self.p2p_to_pool_names.keys():
            if _p2p.peer_id == peer_id:
                p2p = _p2p
                break
        if p2p is None and peer_id in self.peer_id_to_p2p:
            p2p = self.peer_id_to_p2p[peer_id]
        if p2p is None:
            if default_p2p_name is not None:
                p2p = self.p2p_name_to_p2p[default_p2p_name]
            else:
                return False

        try:
            p2p.subscribe_to_topic(channel)
        except (P2PError, ValueError) as e:
            return False
        return True

    def unsubscribe(self, peer_id: str, channel: str):
        """Unsubscribes from a topic/channel on a P2P network.

        Args:
            peer_id: The peer ID associated with the topic/channel.
            channel: The name of the channel to unsubscribe from.

        Returns:
            True if the unsubscription is successful, otherwise False.
        """

        # Getting the right p2p object
        p2p = None
        for _p2p in self.p2p_to_pool_names.keys():
            if _p2p.peer_id == peer_id:
                p2p = _p2p
                break
        if p2p is None and peer_id in self.peer_id_to_p2p:
            p2p = self.peer_id_to_p2p[peer_id]
        if p2p is None:
            return False

        try:
            p2p.unsubscribe_from_topic(channel)
        except (P2PError, ValueError):
            return False
        return True

    def publish(self, peer_id: str, channel: str,
                content_type: str, content: bytes | dict | tuple | None = None):
        """Publishes a message to a topic/channel on a P2P network.

        Args:
            peer_id: The peer ID associated with the topic/channel.
            channel: The name of the channel to publish to.
            content_type: The type of content in the message.
            content: The message content.

        Returns:
            True if the message is published successfully, otherwise False.
        """

        # Getting the right p2p object
        p2p = None
        for _p2p in self.p2p_to_pool_names.keys():
            if _p2p.peer_id == peer_id:
                p2p = _p2p
                break
        if p2p is None:
            p2p = self.peer_id_to_p2p[peer_id]
        if p2p is None:
            return False

        # Adding sender info here
        msg = Msg(sender=p2p.peer_id,
                  content_type=content_type,
                  content=content,
                  channel=channel,
                  piggyback=self.__token + "0")  # Adding inspector-mode bit (dummy bit here)
        if ConnectionPools.DEBUG:
            print("[DEBUG CONNECTIONS-POOL] Sending (publish) message: " + str(msg))

        # Sending message via GossipSub
        try:
            p2p.broadcast_message(channel, msg)

            # If the line above executes without raising an error, it was successful.
            return True
        except P2PError as e:

            # If send_message_to_peer fails, it will raise a P2PError. We catch it here.
            return False


class NodeConn(ConnectionPools):

    # Basic name
    __ALL_UNIVERSE = "all_universe"
    __WORLD_AGENTS_ONLY = "world_agents"
    __WORLD_NODE_ONLY = "world_node"
    __WORLD_MASTERS_ONLY = "world_masters"

    # Suffixes
    __PUBLIC_NET = "_public"
    __PRIVATE_NET = "_private"

    # Prefixes
    __INBOUND = "in_"
    __OUTBOUND = "out_"

    # P2p names
    P2P_PUBLIC = "p2p_public"
    P2P_WORLD = "p2p_world"

    # All pools (prefix + basic name + suffix)
    IN_PUBLIC = __INBOUND + __ALL_UNIVERSE + __PUBLIC_NET
    OUT_PUBLIC = __OUTBOUND + __ALL_UNIVERSE + __PUBLIC_NET
    IN_WORLD_AGENTS = __INBOUND + __WORLD_AGENTS_ONLY + __PRIVATE_NET
    OUT_WORLD_AGENTS = __OUTBOUND + __WORLD_AGENTS_ONLY + __PRIVATE_NET
    IN_WORLD_NODE = __INBOUND + __WORLD_NODE_ONLY + __PRIVATE_NET
    OUT_WORLD_NODE = __OUTBOUND + __WORLD_NODE_ONLY + __PRIVATE_NET
    IN_WORLD_MASTERS = __INBOUND + __WORLD_MASTERS_ONLY + __PRIVATE_NET
    OUT_WORLD_MASTERS = __OUTBOUND + __WORLD_MASTERS_ONLY + __PRIVATE_NET

    # Aggregated pools
    PUBLIC = {IN_PUBLIC, OUT_PUBLIC}
    WORLD_NODE = {IN_WORLD_NODE, OUT_WORLD_NODE}
    WORLD_AGENTS = {IN_WORLD_AGENTS, OUT_WORLD_AGENTS}
    WORLD_MASTERS = {IN_WORLD_MASTERS, OUT_WORLD_MASTERS}
    WORLD = WORLD_NODE | WORLD_AGENTS | WORLD_MASTERS
    ALL = PUBLIC | WORLD
    OUTGOING = {OUT_PUBLIC, OUT_WORLD_NODE, OUT_WORLD_AGENTS, OUT_WORLD_MASTERS}
    INCOMING = {IN_PUBLIC, IN_WORLD_NODE, IN_WORLD_AGENTS, IN_WORLD_MASTERS}

    def __init__(self, max_connections: int, p2p_u: P2P, p2p_w: P2P,
                 is_world_node: bool, public_key: str, token: str):
        """Initializes a new instance of the NodeConn class.

        Args:
            max_connections: The total number of connections the node can handle.
            p2p_u: The P2P object for the public network.
            p2p_w: The P2P object for the world/private network.
            is_world_node: A boolean flag indicating if this node is a world node.
            public_key: The public key for token verification.
            token: The node's authentication token.
        """
        super().__init__(max_connections=max_connections,
                         p2p_name_to_p2p={
                             NodeConn.P2P_PUBLIC: p2p_u,
                             NodeConn.P2P_WORLD: p2p_w,
                         },
                         pool_name_to_p2p_name_and_ratio={
                             NodeConn.IN_PUBLIC: [NodeConn.P2P_PUBLIC, 0.25 / 2. if not is_world_node else 0.25 / 2.],
                             NodeConn.OUT_PUBLIC: [NodeConn.P2P_PUBLIC, 0.25 / 2. if not is_world_node else 0.25 / 2.],
                             NodeConn.IN_WORLD_AGENTS: [NodeConn.P2P_WORLD, .75 / 2 if not is_world_node else 0.5 / 2],
                             NodeConn.OUT_WORLD_AGENTS: [NodeConn.P2P_WORLD, .75 / 2 if not is_world_node else 0.5 / 2],
                             NodeConn.IN_WORLD_NODE: [NodeConn.P2P_WORLD, 0. if not is_world_node else -1.],
                             NodeConn.OUT_WORLD_NODE: [NodeConn.P2P_WORLD, 0. if not is_world_node else -1],
                             NodeConn.IN_WORLD_MASTERS: [NodeConn.P2P_WORLD, 0. if not is_world_node else 0.25 / 2.],
                             NodeConn.OUT_WORLD_MASTERS: [NodeConn.P2P_WORLD, 0. if not is_world_node else 0.25 / 2.]
                         },
                         public_key=public_key, token=token)

        # Just for convenience
        self.p2p_public = p2p_u
        self.p2p_world = p2p_w

        # These are the list of all the possible agents that might try to connect when we are in world
        self.world_agents_list = set()
        self.world_masters_list = set()
        self.world_agents_and_world_masters_list = set()
        self.world_node_peer_id = None
        self.inspector_peer_id = None
        self.role_to_peer_ids = {}
        self.peer_id_to_addrs = {}

        # Rendezvous
        self.rendezvous_tag = -1

    def reset_rendezvous_tag(self):
        """Resets the rendezvous tag to its initial state."""
        self.rendezvous_tag = -1

    def conn_routing_fcn(self, connected_peer_infos: list, p2p: P2P):
        """Routes connected peers to the correct connection pool based on their network and role.

        Args:
            connected_peer_infos: A list of dictionaries with information about connected peers.
            p2p: The P2P network object where the connections were found.

        Returns:
            A dictionary mapping pool names to a dictionary of peer IDs and their information.
        """
        pool_name_and_peer_id_to_peer_info = {k: {} for k in self.p2p_to_pool_names[p2p]}
        public = p2p == self.p2p_public

        for c in connected_peer_infos:
            inbound = c['direction'] == "incoming"
            outbound = c['direction'] == "outgoing"
            peer_id = c['id']  # Other fields are: c['addrs'], c['connected_at']

            if public:
                if inbound:
                    pool_name_and_peer_id_to_peer_info[NodeConn.IN_PUBLIC][peer_id] = c
                elif outbound:
                    pool_name_and_peer_id_to_peer_info[NodeConn.OUT_PUBLIC][peer_id] = c
                else:
                    raise ValueError(f"Connection direction is undefined: {c['direction']}")
            else:
                is_world_agent = peer_id in self.world_agents_list
                is_world_master = peer_id in self.world_masters_list
                is_world_node = self.world_node_peer_id is not None and peer_id == self.world_node_peer_id
                is_inspector = self.inspector_peer_id is not None and peer_id == self.inspector_peer_id
                if not is_world_node and not is_world_master and not is_world_agent and not is_inspector:
                    if ConnectionPools.DEBUG:
                        print("[DEBUG CONNECTIONS-POOL] World agents list:  " + str(self.world_agents_list))
                        print("[DEBUG CONNECTIONS-POOL] World masters list: " + str(self.world_masters_list))
                        print("[DEBUG CONNECTIONS-POOL] World node peer id: " + str(self.world_node_peer_id))
                        print("[DEBUG CONNECTIONS-POOL] Inspector peer id: " + str(self.inspector_peer_id))
                        print(f"[DEBUG CONNECTIONS-POOL] Unable to determine the peer type for {peer_id}: "
                              f"cannot say if world agent, master, world node, inspector (disconnecting it)")
                    ConnectionPools.disconnect(p2p, peer_id)
                    continue

                if inbound:
                    pool_name_and_peer_id_to_peer_info[NodeConn.IN_WORLD_AGENTS if is_world_agent else (
                            NodeConn.IN_WORLD_NODE if is_world_node else
                            NodeConn.IN_WORLD_MASTERS)][peer_id] = c
                elif outbound:
                    pool_name_and_peer_id_to_peer_info[NodeConn.OUT_WORLD_AGENTS if is_world_agent else (
                            NodeConn.OUT_WORLD_NODE if is_world_node else
                            NodeConn.OUT_WORLD_MASTERS)][peer_id] = c
                else:
                    raise ValueError(f"Connection direction is undefined: {c}")

        return pool_name_and_peer_id_to_peer_info

    def set_world(self, world_peer_id: str | None):
        """Sets the peer ID of the world node.

        Args:
            world_peer_id: The peer ID of the world node, or None to clear it.
        """
        self.world_node_peer_id = world_peer_id

    def set_inspector(self, inspector_peer_id: str | None):
        """Sets the peer ID of the inspector.

        Args:
            inspector_peer_id: The peer ID of the inspector node.
        """
        self.inspector_peer_id = inspector_peer_id

    def get_world_peer_id(self):
        """Returns the peer ID of the world node.

        Args:
            None.

        Returns:
            The world node's peer ID.
        """
        return self.world_node_peer_id

    def set_addresses_in_peer_info(self, peer_id, addresses):
        """Updates the list of addresses for a given peer.

        Args:
            peer_id: The peer ID to update.
            addresses: A new list of addresses for the peer.
        """
        if self.in_connection_queues(peer_id):
            addrs = self.pool_name_to_peer_infos[self.get_pool_of(peer_id)][peer_id]['addrs']
            addrs.clear()  # Warning: do not allocate a new list, keep the current one (it is referenced by others)
            for _addrs in addresses:
                addrs.append(_addrs)

    def set_role(self, peer_id, new_role: int):
        """Updates the role of a peer and its associated role-based lists.

        Args:
            peer_id: The peer ID to update.
            new_role: The new role for the peer.
        """
        cur_role = self.get_role(peer_id)

        # Updating
        self.peer_id_to_misc[peer_id] = new_role

        if self.in_connection_queues(peer_id):
            self.pool_name_to_peer_infos[self.get_pool_of(peer_id)][peer_id]['misc'] = new_role

        # Updating
        if cur_role in self.role_to_peer_ids:
            if peer_id in self.role_to_peer_ids[cur_role]:
                self.role_to_peer_ids[cur_role].remove(peer_id)
            if len(self.role_to_peer_ids[cur_role]) == 0:
                del self.role_to_peer_ids[cur_role]
        if new_role not in self.role_to_peer_ids:
            self.role_to_peer_ids[new_role] = set()
        self.role_to_peer_ids[new_role].add(peer_id)

    def set_world_agents_list(self, world_agents_list_peer_infos: list[dict] | None):
        """Sets the list of all world agents based on a provided list of peer information.

        Args:
            world_agents_list_peer_infos: A list of dictionaries containing peer information for world agents.
        """

        # Clearing previous information
        to_remove = []
        for peer_id, misc in self.peer_id_to_misc.items():
            if misc & 1 == 1 and misc & 2 == 0:
                to_remove.append((peer_id, misc))

        for peer_id, misc in to_remove:
            del self.peer_id_to_misc[peer_id]
            if peer_id in self.peer_id_to_addrs:
                del self.peer_id_to_addrs[peer_id]
            self.role_to_peer_ids[misc].discard(peer_id)

        # Setting new information
        if world_agents_list_peer_infos is not None and len(world_agents_list_peer_infos) > 0:
            self.world_agents_list = {x['id'] for x in world_agents_list_peer_infos}
            for x in world_agents_list_peer_infos:
                self.peer_id_to_addrs[x['id']] = x['addrs']
                self.set_role(x['id'], x['misc'])
        else:
            self.world_agents_list = set()

        self.world_agents_and_world_masters_list = self.world_agents_list | self.world_masters_list

    def set_world_masters_list(self, world_masters_list_peer_infos: list[dict] | None):
        """Sets the list of all world masters based on a provided list of peer information.

        Args:
            world_masters_list_peer_infos: A list of dictionaries containing peer information for world masters.
        """

        # Clearing previous information
        to_remove = []
        for peer_id, misc in self.peer_id_to_misc.items():
            if misc & 1 == 1 and misc & 2 == 2:
                to_remove.append((peer_id, misc))

        for peer_id, misc in to_remove:
            del self.peer_id_to_misc[peer_id]
            if peer_id in self.peer_id_to_addrs:
                del self.peer_id_to_addrs[peer_id]
            self.role_to_peer_ids[misc].discard(peer_id)

        # Setting new information
        if world_masters_list_peer_infos is not None and len(world_masters_list_peer_infos) > 0:
            self.world_masters_list = {x['id'] for x in world_masters_list_peer_infos}
            for x in world_masters_list_peer_infos:
                self.peer_id_to_addrs[x['id']] = x['addrs']
                self.set_role(x['id'], x['misc'])
        else:
            self.world_masters_list = set()

        self.world_agents_and_world_masters_list = self.world_agents_list | self.world_masters_list

    def add_to_world_agents_list(self, peer_id: str, addrs: list[str], role: int = -1):
        """Adds a new world agent to the list.

        Args:
            peer_id: The peer ID of the new agent.
            addrs: A list of addresses for the new agent.
            role: The role assigned to the agent.
        """
        self.world_agents_list.add(peer_id)

        # This assumes that the WORLD MASTER/AGENT BIT is the first one
        assert role & 1 == 1, "Expecting the first bit of the role to be 1 for world agents"
        assert role & 2 == 0, "Expecting the second bit of the role to be 0 for world agents"
        self.peer_id_to_addrs[peer_id] = addrs
        self.set_role(peer_id, role)
        self.world_agents_and_world_masters_list = self.world_agents_list | self.world_masters_list

    def add_to_world_masters_list(self, peer_id: str, addrs: list[str], role: int = -1):
        """Adds a new world master to the list.

        Args:
            peer_id: The peer ID of the new master.
            addrs: A list of addresses for the new master.
            role: The role assigned to the master.
        """
        self.world_masters_list.add(peer_id)

        # This assumes that the WORLD MASTER/AGENT BIT is the first one
        assert role & 1 == 1, "Expecting the first bit of the role to be 1 for world masters"
        assert role & 2 == 2, "Expecting the second bit of the role to be 1 for world masters"
        self.peer_id_to_addrs[peer_id] = addrs
        self.set_role(peer_id, role)
        self.world_agents_and_world_masters_list = self.world_agents_list | self.world_masters_list

    def get_added_after_updating(self, pool_names: list[str] | None = None):
        """Retrieves the set of peers added after the last update cycle for specified pools.

        Args:
            pool_names: A list of pool names to check. If None, checks all pools.

        Returns:
            A dictionary mapping pool names to sets of added peer IDs, or a single set if only one pool is specified.
        """
        if pool_names is not None:
            ret = {}
            for p in pool_names:
                ret[p] = super().get_added_after_updating(p)
            return ret
        else:
            return super().get_added_after_updating()

    def get_removed_after_updating(self, pool_names: list[str] | None = None):
        """Retrieves the set of peers removed after the last update cycle for specified pools.

        Args:
            pool_names: A list of pool names to check. If None, checks all pools.

        Returns:
            A dictionary mapping pool names to sets of removed peer IDs, or a single set if only one pool is specified.
        """
        if pool_names is not None:
            ret = {}
            for p in pool_names:
                ret[p] = super().get_removed_after_updating(p)
            return ret
        else:
            return super().get_removed_after_updating()

    def size(self, pool_names: list[str] | None = None):
        """Returns the total number of connections across all specified pools.

        Args:
            pool_names: A list of pool names to sum the size of. If None, returns the total size of all pools.

        Returns:
            The total number of connections.
        """
        if pool_names is not None:
            return super().size()
        else:
            c = 0
            for p in self.pool_names:
                c += super().size(p)
            return c

    def is_connected(self, peer_id: str, pool_names: list[str] | None = None):
        """Checks if a peer is connected in any of the specified pools.

        Args:
            peer_id: The peer ID to check.
            pool_names: A list of pool names to search within. If None, searches all pools.

        Returns:
            True if the peer is found in any of the pools, otherwise False.
        """
        if pool_names is None:
            return super().is_connected(peer_id)
        else:
            for p in pool_names:
                if super().is_connected(peer_id, p):
                    return True
            return False

    def is_public(self, peer_id):
        """Checks if a peer is connected via the public network.

        Args:
            peer_id: The peer ID to check.

        Returns:
            True if the peer is in a public pool, otherwise False.
        """
        pool_name = self.get_pool_of(peer_id)
        return pool_name in NodeConn.PUBLIC

    def is_world_master(self, peer_id):
        """Checks if a peer is a world master.

        Args:
            peer_id: The peer ID to check.

        Returns:
            True if the peer is in a world master pool, otherwise False.
        """
        pool_name = self.get_pool_of(peer_id)
        return pool_name in NodeConn.WORLD_MASTERS

    def is_world_node(self, peer_id):
        """Checks if a peer is the world node.

        Args:
            peer_id: The peer ID to check.

        Returns:
            True if the peer is in a world node pool, otherwise False.
        """
        pool_name = self.get_pool_of(peer_id)
        return pool_name in NodeConn.WORLD_NODE

    def is_in_world(self, peer_id):
        """Checks if a peer is connected to the world network.

        Args:
            peer_id: The peer ID to check.

        Returns:
            True if the peer is in any world pool, otherwise False.
        """
        pool_name = self.get_pool_of(peer_id)
        return pool_name in NodeConn.WORLD

    def get_role(self, peer_id):
        """Retrieves the role of a given peer.

        Args:
            peer_id: The peer ID to query.

        Returns:
            The integer role of the peer.
        """
        role = self.peer_id_to_misc.get(peer_id, 0)  # 0 means public
        assert role >= 0, "Expecting role to be >= 0"
        assert role & 1 != 0 or role == 0, "Expecting public role to be zero (all-zero-bits)"
        return role

    def get_addrs(self, peer_id):
        """Retrieves the list of addresses for a given peer.

        Args:
            peer_id: The peer ID to query.

        Returns:
            A list of addresses for the peer.
        """
        return self.peer_id_to_addrs.get(peer_id)

    def in_connection_queues(self, peer_id):
        """Checks if a peer ID exists in any connection pool.

        Args:
            peer_id: The peer ID to check.

        Returns:
            True if the peer is found in any pool, otherwise False.
        """
        return peer_id in self.peer_id_to_pool_name

    def find_addrs_by_role(self, role, return_peer_ids_too: bool = False):
        """Finds all addresses of peers with a specific role.

        Args:
            role: The integer role to search for.
            return_peer_ids_too: A boolean to also return the peer IDs.

        Returns:
            A list of lists of addresses, and optionally a list of peer IDs.
        """
        if role in self.role_to_peer_ids:
            peer_ids = self.role_to_peer_ids[role]
        else:
            if not return_peer_ids_too:
                return []
            else:
                return [], []
        ret_addrs = []
        ret_peer_ids = []
        for peer_id in peer_ids:
            addrs = self.get_addrs(peer_id)
            if addrs is not None:
                ret_addrs.append(addrs)
                ret_peer_ids.append(peer_id)
        if not return_peer_ids_too:
            return ret_addrs
        else:
            return ret_addrs, ret_peer_ids

    def count_by_role(self, role: int):
        """Counts the number of peers with a specific role.

        Args:
            role: The integer role to count.

        Returns:
            The number of peers with that role.
        """
        if role in self.role_to_peer_ids:
            return len(self.role_to_peer_ids[role])
        else:
            return 0

    def get_all_connected_peer_infos(self, pool_names: list[str] | set[str]):
        """Retrieves a list of all peer info dictionaries for the specified pools.

        Args:
            pool_names: A list or set of pool names to query.

        Returns:
            A list of dictionaries containing peer information.
        """
        ret = []
        for p in pool_names:
            ret += super().get_all_connected_peer_infos(p)
        return ret

    def set_world_agents_and_world_masters_lists_from_rendezvous(self):
        """Updates the lists of world agents and masters using data from the rendezvous topic."""
        rendezvous_state = self.p2p_world.get_rendezvous_peers_info()

        if rendezvous_state is not None:
            tag = rendezvous_state.get('update_count', -1)

            if tag > self.rendezvous_tag:
                self.rendezvous_tag = tag
                rendezvous_peer_infos = rendezvous_state.get('peers', [])

                world_agents_peer_infos = []
                world_masters_peer_infos = []

                if ConnectionPools.DEBUG:
                    print(f"[DEBUG CONNECTIONS-POOL] Rendezvous peer infos (tag: {tag}, peers: "
                          f"{len(rendezvous_peer_infos)} peers)")

                for c in rendezvous_peer_infos:
                    if c['addrs'] is None or len(c['addrs']) == 0:
                        print(f"[DEBUG CONNECTIONS-POOL] Skipping a peer with missing addrs: {c}")
                        continue
                    if (c['misc'] & 1) == 1 and (c['misc'] & 2) == 0:
                        world_agents_peer_infos.append(c)
                    elif (c['misc'] & 1) == 1 and (c['misc'] & 2) == 2:
                        world_masters_peer_infos.append(c)
                    else:
                        raise ValueError("Unexpected value of the 'misc' field: " + str(c))

                # Updating lists
                self.set_world_agents_list(world_agents_peer_infos)
                self.set_world_masters_list(world_masters_peer_infos)

    def get_cv_hash_from_last_token(self, peer_id):
        """Retrieves the CV hash from the last token received from a peer.

        Args:
            peer_id: The peer ID to query.

        Returns:
            The CV hash string, or None if not found.
        """
        token = self.get_last_token(peer_id)
        if token is not None:
            _, cv_hash = self.verify_token(token, peer_id)
            return cv_hash
        else:
            return None

    def remove(self, peer_id: str):
        """Removes a peer and its associated information from all lists and pools.

        Args:
            peer_id: The peer ID to remove.
        """
        super().remove(peer_id)
        if peer_id in self.peer_id_to_addrs:
            del self.peer_id_to_addrs[peer_id]

    def remove_all_world_agents(self):
        """Removes all connected world agents from the pools and role lists."""
        peer_infos = self.get_all_connected_peer_infos(NodeConn.WORLD)
        for c in peer_infos:
            peer_id = c['id']
            self.remove(peer_id)
            for role, peer_ids in self.role_to_peer_ids.items():
                if role & 1 == NodeConn.WORLD:
                    peer_ids.remove(peer_id)

    def subscribe(self, peer_id: str, channel: str, default_p2p_name: str | None = None):
        """Subscribes to a channel, defaulting to the world P2P network if a network is not specified.

        Args:
            peer_id: The peer ID associated with the channel.
            channel: The channel to subscribe to.
            default_p2p_name: An optional P2P name to use for the subscription.

        Returns:
            True if successful, False otherwise.
        """
        return super().subscribe(peer_id, channel,
                                 default_p2p_name=NodeConn.P2P_WORLD if default_p2p_name is None else default_p2p_name)

    def get_messages(self, p2p_name: str, allowed_not_connected_peers: set | None = None):
        """Retrieves messages, allowing for messages from known world agents and masters even if not in a connection pool.

        Args:
            p2p_name: The name of the P2P network to get messages from.
            allowed_not_connected_peers: This parameter is ignored in this implementation.

        Returns:
            A list of verified and processed message objects.
        """
        assert allowed_not_connected_peers is None, "This param (allowed_not_connected_peers is ignored in NodeConn"
        return super().get_messages(p2p_name, allowed_not_connected_peers=self.world_agents_and_world_masters_list)
