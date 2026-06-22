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
import time
import math
import json
import binascii
from collections import namedtuple
from unaiverse.utils.logger import log
from unaiverse.networking.p2p.messages import Msg
from unaiverse.networking.p2p.p2p import P2P, P2PError
from unaiverse.networking.node.tokens import TokenVerifier


# The session-protocol wire codec, its protocol IDs and the SessionInbound value object
# are the session layer's single source of truth (sess_msg.py, the leaf module). connpool
# imports them rather than redefining them; the inbound codec itself (pop + decode) now
# lives in P2P.get_messages, and connpool only runs the Node-bound chat path. The protobuf
# dependency stays inside SessMsg.to_bytes/from_bytes (lazy import), never at this top.
from unaiverse.networking.session.sess_msg import (
    SessMsg, SESSION_PROTOCOL_IDS, SessionInbound,
)


# Immutable per-cycle view of the world membership that the perimeter routing
# classifies against. Built once per update cycle so conn_routing_fcn reads a single
# consistent snapshot instead of the Coordinator's mutable sets — and so the same view
# can be handed down to each per-instance reconcile when the perimeter moves into the
# SessionManagers. The world side is fed by the rendezvous; the public side ignores it.
WorldMembership = namedtuple(
    "WorldMembership",
    ["world_agents", "world_masters", "world_node_peer_id", "inspector_peer_id"],
)


class NodeConn:
    """The networking Coordinator. Owns the two per-instance P2P handles, the two
    per-instance SessionManagers and the cross-instance world membership, and exposes
    the pool/routing facade the hosted entity speaks to. This used to be a NodeConn
    subclass over a generic ``ConnectionPools`` base, but the base was never
    instantiated on its own (only NodeConn was), so the two have been collapsed into a
    single class: the pool/transport primitives and the Coordinator logic now live
    together here."""

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
                 is_world_node: bool, public_key: str, token: str,
                 local_node_id: str = None) -> None:
        """Initializes a new instance of the NodeConn class.

        Args:
            max_connections: The total number of connections the node can handle.
            p2p_u: The P2P object for the public network.
            p2p_w: The P2P object for the world/private network.
            is_world_node: A boolean flag indicating if this node is a world node.
            public_key: The public key for token verification.
            token: The node's authentication token.
            local_node_id: this node's application node-id, used when building the
                per-instance SessionManagers in attach_session_policy.
        """
        self._init_pools(max_connections=max_connections,
                         p2p_name_to_p2p={
                             NodeConn.P2P_PUBLIC: p2p_u,
                             NodeConn.P2P_WORLD: p2p_w,
                         },
                         pool_name_to_p2p_name_and_ratio={
                             NodeConn.IN_PUBLIC: [NodeConn.P2P_PUBLIC, 0.25 / 2. if not is_world_node else 0.25 / 2.],
                             NodeConn.OUT_PUBLIC: [NodeConn.P2P_PUBLIC, 0.25 / 2. if not is_world_node else 0.25 / 2.],
                             NodeConn.IN_WORLD_AGENTS: [NodeConn.P2P_WORLD, .75 / 2. if not is_world_node else 0.5 / 2],
                             NodeConn.OUT_WORLD_AGENTS: [NodeConn.P2P_WORLD, .75 / 2 if not is_world_node else 0.5 / 2],
                             NodeConn.IN_WORLD_NODE: [NodeConn.P2P_WORLD, 0. if not is_world_node else -1.],
                             NodeConn.OUT_WORLD_NODE: [NodeConn.P2P_WORLD, 0. if not is_world_node else -1.],
                             NodeConn.IN_WORLD_MASTERS: [NodeConn.P2P_WORLD, 0. if not is_world_node else 0.25 / 2.],
                             NodeConn.OUT_WORLD_MASTERS: [NodeConn.P2P_WORLD, 0. if not is_world_node else 0.25 / 2.]
                         },
                         public_key=public_key, token=token)

        # Just for convenience
        self.p2p_public = p2p_u
        self.p2p_world = p2p_w

        # The pure world membership (world_agents_set / world_masters_set /
        # world_node_peer_id) now lives in sm_world, next to the Sessions it gates (built
        # in attach_session_policy); the Coordinator reads it back through
        # membership_snapshot(). inspector_peer_id stays here: it is not pure world
        # membership but an orthogonal, currently-inert chat exception (D-8a).
        self.inspector_peer_id = None
        # role_to_peer_ids and peer_id_to_addrs stay Coordinator-owned and do NOT descend
        # into sm_world with the world membership (D-2). role_to_peer_ids is the inverse
        # index of peer_id_to_misc (the global cross-pool taxonomy, kept here) and set_role
        # updates both together; find_addrs_by_role reads role_to_peer_ids and
        # peer_id_to_addrs jointly, and get_addrs feeds the dial path. Only the pure world
        # membership (the two sets + world_node_peer_id, now in sm_world) is what the
        # perimeter classifies against.
        self.role_to_peer_ids = {}
        self.peer_id_to_addrs = {}

        # The per-cycle world-membership snapshot, frozen at the start of each update()
        # and read by BOTH the perimeter (via _reconcile_instance) and the chat-admission
        # gate in get_messages, so the two views never diverge inside a cycle. A roster
        # mutation written mid-cycle (pre-admission) takes effect only at the next
        # update()'s snapshot, for both views. Bootstrapped to an explicit empty snapshot
        # so get_messages admits no not-yet-connected peer before the first update(): the
        # roster lives in sm_world, built later in attach_session_policy, so there is no
        # manager to snapshot from yet.
        self._cycle_membership = WorldMembership(
            world_agents=frozenset(), world_masters=frozenset(),
            world_node_peer_id=None, inspector_peer_id=None,
        )

        # Rendezvous
        self.rendezvous_tag = -1

        # Application node-id, used when this Coordinator builds the SessionManagers.
        self._local_node_id = local_node_id

        # Per-instance SessionManagers. The Coordinator owns them: it builds both in
        # attach_session_policy (once the Node policy is available) and consults them
        # so the hosted entity (which holds conn, not the managers) can ask about
        # per-peer Session state (see is_peer_session_pending).
        self._sm_public = None
        self._sm_world = None
        self._session_managers = []

    def _init_pools(self, max_connections: int, pool_name_to_p2p_name_and_ratio: dict[str, list[str | float]],
                    p2p_name_to_p2p: dict[str, P2P],
                    public_key: str | None = None, token: str | None = None) -> None:
        """Set up the pool triples and the per-peer / per-p2p indices for this Coordinator.

        Args:
            max_connections: The maximum total number of connections allowed across all pools.
            pool_name_to_p2p_name_and_ratio: A dictionary mapping pool names to a list containing the associated P2P
                network name and its connection ratio.
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
        self.pool_ratios: list[float] = []
        for p2p_name_and_ratio in pool_name_to_p2p_name_and_ratio.values():
            ratio = p2p_name_and_ratio[1]
            assert isinstance(ratio, float)
            self.pool_ratios.append(ratio)

        # Indices involving the P2P object or its name
        self.p2p_name_to_p2p = p2p_name_to_p2p
        self.p2p_name_and_pool_name_to_pool_triple = {}
        self.p2p_to_pool_names = {}

        # Indices rooted around the pool name
        self.pool_name_to_pool_triple = {}
        self.pool_name_to_added_in_last_update = {}
        self.pool_name_to_removed_in_last_update = {}
        self.pool_name_to_peer_infos = {p: {} for p in self.pool_names}

        # Indices rooted around the peer ID. peer_id_to_pool_name is the connection
        # index (which pool a connected peer lives in): it backs is_connected,
        # get_pool_of, in_connection_queues, get_all_connected_peer_ids and the chat
        # admission gate in get_messages. It is no longer the primary source of the
        # peer -> instance routing: that is Session-anchored now (_p2p_of asks the
        # SessionManagers first and only falls back to this map for a peer connected
        # without a Session yet). So it stays load-bearing and must not be dropped when
        # the world roster and the pool state later move down into the SessionManagers.
        self.peer_id_to_pool_name = {}
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
        self.pool_name_to_pool_triple = {}
        for k in self.pool_names:
            p2p_name = pool_name_to_p2p_name_and_ratio[k][0]
            assert isinstance(p2p_name, str)
            p2p: P2P = self.p2p_name_to_p2p[p2p_name]
            self.pool_name_to_pool_triple[k] = [set(), 0, p2p]
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
                p2p: P2P
                self.p2p_to_pool_names[p2p] = []
            self.p2p_name_and_pool_name_to_pool_triple[p2p_name][pool_name] = (
                pool_contents_max_con_and_p2p)
            self.p2p_to_pool_names[p2p].append(pool_name)

        # Time markers
        self.__last_sent_time = 0.
        self.__last_recv_time = 0.

    def __str__(self) -> str:
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

    def __getitem__(self, p2p_name: str) -> P2P:
        """Retrieves a P2P object by its name.

        Args:
            p2p_name: The name of the P2P network.

        Returns:
            The P2P object.
        """
        return self.p2p_name_to_p2p[p2p_name]

    @staticmethod
    async def disconnect(p2p: P2P, peer_id: str) -> bool:
        """Disconnects from a specific peer on a P2P network (async).

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

    def set_token(self, token: str) -> None:
        """Sets the authentication token for the connection pools.

        Args:
            token: The new token string.
        """
        self.__token = token

    async def verify_token(self, token: str, peer_id: str) -> tuple[str | None, str | None]:
        """Verifies a received token using the provided public key (async).

        Args:
            token: The token string to verify.
            peer_id: The peer ID associated with the token.

        Returns:
            A tuple containing the node ID and CV hash if the token is valid, otherwise None.
        """
        if self.__token_verifier is None:
            return None, None
        else:
            node_id, cv_hash = self.__token_verifier.verify_token(token, p2p_peer=peer_id)
            return node_id, cv_hash  # If the verification fails, this is None, None

    async def verify_token_claims(self, token: str, peer_id: str) -> tuple:
        """Verify a token and return the full claims, including the root-attested
        `p2p_peers` binding, for the Session handshake (RT). The `p2p_peer` check is
        always against `peer_id` (the transport-verified peer-id of the speaker).

        Returns:
            ``(node_id, cv_hash, p2p_peers, exp)`` on success, ``(None, None, (), 0)``
            on failure or when no verifier is configured.
        """
        if self.__token_verifier is None:
            return None, None, (), 0
        return self.__token_verifier.verify_token_claims(token, p2p_peer=peer_id)

    async def reserve(self, peer_id: str, p2p_name: str) -> str | None:
        """Attempts to reserve a slot on a specified relay node via a P2P network (async).

        Args:
            peer_id: The peer ID of the relay node.
            p2p_name: The name of the P2P network to use.

        Returns:
            The UTC expiration timestamp of the reservation, or None if it fails.
        """
        p2p = self.p2p_name_to_p2p[p2p_name]
        try:
            return p2p.reserve_on_relay(peer_id)
        except P2PError:
            return None

    def add(self, peer_info: dict, pool_name: str) -> bool:
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

            # Setting 'misc' field (default is 0, where 0 means public)
            peer_info['misc'] = self.peer_id_to_misc.get(peer_id, 0)

            # Storing (only)
            pool.add(peer_id)
            self.pool_name_to_peer_infos[pool_name][peer_id] = peer_info
            return True
        else:
            return False

    async def remove(self, peer_id: str) -> bool:
        """Removes a peer from its connection pool and disconnects from it (async).

        Args:
            peer_id: The peer ID to remove.

        Returns:
            True if the peer is successfully removed, otherwise False.
        """
        if peer_id in self.peer_id_to_pool_name:
            pool_name = self.peer_id_to_pool_name[peer_id]
            pool, _, p2p = self.pool_name_to_pool_triple[pool_name]

            # Disconnecting
            disc = await NodeConn.disconnect(p2p, peer_id)
            pool.remove(peer_id)
            del self.pool_name_to_peer_infos[pool_name][peer_id]
            del self.peer_id_to_pool_name[peer_id]
            if peer_id in self.peer_id_to_token:
                del self.peer_id_to_token[peer_id]

            # Remember to NOT del peer_id_to_misc[peer_id]!
            return disc
        else:
            return False

    def get_pool_status(self) -> dict[str, set]:
        """Returns a dictionary showing the set of peer IDs in each pool.

        Args:
            None.

        Returns:
            A dictionary mapping pool names to the set of peer IDs in that pool.
        """
        return {k: v[0] for k, v in self.pool_name_to_pool_triple.items()}

    def get_all_connected_peer_ids(self) -> list[str]:
        """Retrieves a list of all peer IDs currently connected across all pools.

        Args:
            None.

        Returns:
            A list of all connected peer IDs.
        """
        return list(self.peer_id_to_pool_name.keys())

    async def get_messages(
        self,
        p2p_name: str,
        allowed_not_connected_peers: set | None = None,
    ) -> tuple[list[Msg], list[SessionInbound]]:
        """Retrieves and verifies all messages from a specified P2P network (async).

        Returns a tuple `(chat_messages, session_messages)`:
          * `chat_messages` are decoded `Msg` objects whose `piggyback` JWT has been
            verified (the existing chat / pubsub flow, unchanged).
          * `session_messages` are raw per-stage Session payloads delivered on one of
            the SESSION_PROTOCOL_IDS — these skip Msg decoding and JWT verification
            because identity is established cryptographically inside the Session
            handshake (sig_b / sig_a / token).

        Args:
            p2p_name: The name of the P2P network to fetch messages from.
            allowed_not_connected_peers: must be None at the call site. The Coordinator
                admits chat from the world agents/masters of THIS CYCLE'S membership
                snapshot (the same frozen view the perimeter classifies against, not the
                live mutable set), so a mid-cycle pre-admission only enables chat at the
                next update(). Session messages are never gated on pool membership, the
                handshake itself is what enrols a peer.
        """
        assert allowed_not_connected_peers is None, \
            "allowed_not_connected_peers is set by the Coordinator, not by the caller"
        allowed_not_connected_peers = (self._cycle_membership.world_agents
                                       | self._cycle_membership.world_masters)

        # P2P pops this instance's inbound batch and decodes the Session-protocol frames
        # (the wire codec lives in the transport tier); chat_raw stays raw for the
        # Node-bound chat path below, and session_messages is passed straight through.
        p2p = self[p2p_name]
        chat_raw, session_messages = p2p.get_messages()
        processed_messages: list[Msg] = []

        if chat_raw or session_messages:
            self.__last_recv_time = time.time()

        for i, msg_dict in enumerate(chat_raw):
            msg_dict: dict
            try:
                # The binary transport delivers "data" already as raw bytes (no base64).
                # Go structure: {"from":"Qm...", "data":<bytes>, "protocol":"..."}
                verified_sender_id = msg_dict.get("from")
                decoded_data = msg_dict.get("data")

                # ── Chat path (Node-bound) ───────────────────────────────────
                # Session-protocol frames were already split off and decoded by
                # P2P.get_messages; everything here is a chat Msg that needs
                # Msg.from_bytes + sender-mismatch + membership gate + JWT verify.
                msg_obj = Msg.from_bytes(decoded_data)

                log.network(f"<<< RECEIVED {msg_obj.content_type} through channel {msg_obj.channel}", sub=p2p.log_sub)

                # --- CRITICAL SECURITY CHECK ---
                # Verify that the sender claimed inside the message payload
                # matches the cryptographically verified sender from the network layer.
                if msg_obj.sender != verified_sender_id:
                    log.error(f"SENDER MISMATCH! Network sender '{verified_sender_id}' "
                              f"does not match payload sender '{msg_obj.sender}'. Discarding message.",
                              sub=p2p.log_sub)

                    # In a real-world scenario, you might also want to penalize or disconnect
                    # from a peer that sends such malformed/spoofed messages.
                    continue  # Discard this message

                # filter only valid messages to return
                if (msg_obj.sender in self.peer_id_to_pool_name or  # Check if expected sender
                        (allowed_not_connected_peers is not None and msg_obj.sender in allowed_not_connected_peers)):

                    try:
                        token_with_inspector_final_bit = msg_obj.piggyback
                        token = token_with_inspector_final_bit[0:-1]
                        inspector_mode = token_with_inspector_final_bit[-1]
                        node_id, _ = await self.verify_token(token, msg_obj.sender)
                        if node_id is not None:
                            # Replacing piggyback with the node ID and the flag telling if it is inspector
                            msg_obj.piggyback = node_id + inspector_mode
                            processed_messages.append(msg_obj)
                            if msg_obj.sender in self.peer_id_to_pool_name:
                                self.peer_id_to_token[msg_obj.sender] = token
                        else:
                            log.error("Received a message missing expected info in the token payload "
                                      "(discarding it)", sub=p2p.log_sub)
                    except Exception as e:
                        log.error(f"Received a message with an invalid piggyback token! (discarding it) [{e}]",
                                  sub=p2p.log_sub)

            except ValueError as ve:
                log.error(f"Invalid message created, stopping. Error: {ve}", sub=p2p.log_sub)
                continue  # Skip problematic message
            except (TypeError, binascii.Error) as decode_err:
                log.error(f"Failed to decode Base64 data for a message in batch: {decode_err}. "
                          f"Message dict: {msg_dict}", sub=p2p.log_sub)
                continue  # Skip problematic message
            except Exception as msg_proc_err:  # Catch errors from Msg.from_bytes or attribute setting
                log.error(f"Error processing popped message item {i}: {msg_proc_err}. "
                          f"Message dict: {msg_dict}", sub=p2p.log_sub)
                continue  # Skip problematic message

        return processed_messages, session_messages

    def passed_time_since_last_communication(self):
        passed = time.time() - max(self.__last_sent_time, self.__last_recv_time)
        return 0. if passed < 0. else passed

    def get_last_token(self, peer_id: str) -> str | None:
        """Retrieves the last known token for a given peer.

        Args:
            peer_id: The peer ID to query.

        Returns:
            The token string if found, otherwise None.
        """
        return self.peer_id_to_token[peer_id] if peer_id in self.peer_id_to_token else None

    def get_pool_of(self, peer_id: str) -> str | None:
        """Gets the pool name for a given connected peer.

        Args:
            peer_id: The peer ID to query.

        Returns:
            The name of the pool the peer is in.
        """
        if peer_id in self.peer_id_to_pool_name:
            return self.peer_id_to_pool_name[peer_id]
        else:
            return None

    async def send(self, peer_id: str, channel_trail: str | None,
                   content_type: str, content: bytes | dict | str | None = None, p2p: P2P | None = None) -> bool:
        """Sends a direct message to a specific peer (async).

        Args:
            peer_id: The peer ID to send the message to.
            channel_trail: An optional string to append to the channel name.
            content_type: The type of content in the message.
            content: The message content.
            p2p: An optional P2P object to use for sending. If None, it is derived from the peer_id.

        Returns:
            True if the message is sent successfully, otherwise False.
        """
        if content_type not in [Msg.STATS_UPDATE, Msg.STATS_REQUEST, Msg.STATS_RESPONSE]:
            self.__last_sent_time = time.time()

        # Getting the right p2p object
        if p2p is None:
            p2p = self._p2p_of(peer_id)
            if p2p is None:
                log.error("P2P non found for peer id: " + str(peer_id))
                return False

        # Defining channel
        if channel_trail is not None and len(channel_trail) > 0:
            channel = f"{p2p.peer_id}::dm:{peer_id}-{content_type}~{channel_trail}"
        else:
            channel = f"{p2p.peer_id}::dm:{peer_id}-{content_type}"

        log.network(f">>> SENDING {content_type} to {peer_id}", sub=p2p.log_sub)

        # Adding sender info here
        msg = Msg(sender=p2p.peer_id,
                  content_type=content_type,
                  content=content,
                  channel=channel,
                  piggyback=self.__token + "0")  # Adding inspector-mode bit (dummy bit here)

        # Sending direct message (offloaded to a thread so the event loop stays unblocked)
        try:
            await asyncio.to_thread(p2p.send_message_to_peer, channel, msg_bytes=msg.to_bytes())
            return True
        except P2PError as e:
            log.error("Sending error is: " + str(e), sub=p2p.log_sub)
            return False

    async def subscribe(self, peer_id: str, channel: str, default_p2p_name: str | None = None) -> bool:
        """Subscribes to a topic/channel on a P2P network (async).

        Args:
            peer_id: The peer ID associated with the topic/channel.
            channel: The name of the channel to subscribe to.
            default_p2p_name: The P2P network to fall back on when the peer's network is
                unknown; defaults to the world network when not given.

        Returns:
            True if the subscription is successful, otherwise False.
        """
        if default_p2p_name is None:
            default_p2p_name = NodeConn.P2P_WORLD

        # Getting the right p2p object
        p2p = None
        for _p2p in self.p2p_to_pool_names.keys():
            if _p2p.peer_id == peer_id:
                p2p = _p2p
                break
        if p2p is None:
            p2p = self._p2p_of(peer_id)
        if p2p is None:
            if default_p2p_name is not None:
                p2p = self.p2p_name_to_p2p[default_p2p_name]
            else:
                return False

        try:
            p2p.subscribe_to_topic(channel)
        except (P2PError, ValueError):
            return False
        return True

    async def unsubscribe(self, peer_id: str, channel: str, default_p2p_name: str | None = None) -> bool:
        """Unsubscribes from a topic/channel on a P2P network (async).

        Args:
            peer_id: The peer ID associated with the topic/channel.
            channel: The name of the channel to unsubscribe from.
            default_p2p_name: An optional P2P network name to use if the peer's network is unknown.

        Returns:
            True if the unsubscription is successful, otherwise False.
        """

        # Getting the right p2p object
        p2p = None
        for _p2p in self.p2p_to_pool_names.keys():
            if _p2p.peer_id == peer_id:
                p2p = _p2p
                break
        if p2p is None:
            p2p = self._p2p_of(peer_id)
        if p2p is None:
            if default_p2p_name is None:
                default_p2p_name = NodeConn.P2P_WORLD
            p2p = self.p2p_name_to_p2p[default_p2p_name]
        if p2p is None:
            return False

        try:
            p2p.unsubscribe_from_topic(channel)
        except (P2PError, ValueError):
            return False
        return True

    async def publish(self, peer_id: str, channel: str,
                      content_type: str, content: bytes | dict | tuple | None = None) -> bool:
        """Publishes a message to a topic/channel on a P2P network (async).

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
            p2p = self._p2p_of(peer_id)
        if p2p is None:
            return False

        p2p: P2P
        log.network(f">>> PUBLISHING {content_type} to {channel}", sub=p2p.log_sub)

        # Adding sender info here
        msg = Msg(sender=p2p.peer_id,
                  content_type=content_type,
                  content=content,
                  channel=channel,
                  piggyback=self.__token + "0")  # Adding inspector-mode bit (dummy bit here)

        # Sending message via GossipSub
        try:
            p2p.broadcast_message(channel, msg_bytes=msg.to_bytes())

            # If the line above executes without raising an error, it was successful.
            return True
        except P2PError:

            # If send_message_to_peer fails, it will raise a P2PError. We catch it here.
            return False

    def attach_session_policy(self, local_profile_factory, decision_hook, token_factory=None) -> None:
        """Build and own the two per-instance SessionManagers, now that the Node
        policy born late is available (the profile factory samples self.profile, the
        decision hook reads world/agent/allow-list state). Each manager owns its P2P
        instance."""
        from unaiverse.networking.session import SessionManager
        # Per-instance perimeter routing injected into each manager: the public one
        # routes by direction (no membership, never cuts); the world one classifies
        # against the snapshot and cuts unclassifiable peers on its P2P instance.
        self._sm_public = SessionManager(
            p2p=self.p2p_public,
            p2p_name=NodeConn.P2P_PUBLIC,
            local_peer_id=self.p2p_public.peer_id,
            local_node_id=self._local_node_id,
            local_profile_factory=local_profile_factory,
            decision_hook=decision_hook,
            token_verifier=self.verify_token_claims,
            token_factory=token_factory,
            routing_classifier=NodeConn._route_public,
            perimeter_disconnect=None,
        )
        self._sm_world = SessionManager(
            p2p=self.p2p_world,
            p2p_name=NodeConn.P2P_WORLD,
            local_peer_id=self.p2p_world.peer_id,
            local_node_id=self._local_node_id,
            local_profile_factory=local_profile_factory,
            decision_hook=decision_hook,
            token_verifier=self.verify_token_claims,
            token_factory=token_factory,
            routing_classifier=NodeConn._route_world,
            perimeter_disconnect=lambda peer_id: NodeConn.disconnect(self.p2p_world, peer_id),
        )
        self.set_session_managers(self._sm_public, self._sm_world)

    def set_session_managers(self, *session_managers) -> None:
        """Register the per-instance SessionManagers so conn can answer per-peer
        Session-state queries on behalf of the hosted entity. Called by
        attach_session_policy (and directly by unit tests with stub managers)."""
        self._session_managers = [sm for sm in session_managers if sm is not None]

    def is_peer_session_pending(self, peer_id: str) -> bool:
        """True if `peer_id` has a Session whose *first* handshake is still in
        flight (it exists but has never reached ESTABLISHED) on any registered
        instance. This is the session-era replacement for the old "connected but
        not yet managed" notion behind Agent.agents_are_waiting.

        RECONNECTING is deliberately excluded: such a peer was already established
        (already managed) and is merely re-establishing, so it is not "waiting to
        be managed". KNOWN is excluded too: it is a peer known from the rendezvous
        whose initiator has not dialed yet, so no handshake is in flight. Returns
        False for no Session / KNOWN / ESTABLISHED / RECONNECTING / terminal
        (FAILED, CLOSED)."""
        from unaiverse.networking.session.session import SessionState
        for sm in self._session_managers:
            sess = sm.get_session(peer_id)
            if sess is None:
                continue
            if sess.state not in (SessionState.KNOWN,
                                  SessionState.ESTABLISHED,
                                  SessionState.RECONNECTING,
                                  SessionState.FAILED,
                                  SessionState.CLOSED):
                return True
        return False

    def requires_fast_clock(self) -> bool:
        """True if any registered SessionManager has a Session that needs the main loop
        at full speed (see SessionManager.requires_fast_clock). The Coordinator aggregates
        across its instances so the Node's clock-pacing decision is one state-based query
        instead of reaching into each manager. This supersedes the old idle check that
        counted only ESTABLISHED/RECONNECTING and so let a slowed clock starve handshakes
        in flight."""
        return any(sm.requires_fast_clock() for sm in self._session_managers)

    def _p2p_of(self, peer_id: str) -> "P2P | None":
        """Resolve a peer's P2P instance by asking the SessionManagers first: the SM
        that holds a Session for this peer-id owns the instance the peer lives on
        (public before world, matching the cross-pool first-network-wins). Falls back to
        the connection pool for a peer that is connected but has no Session yet (an
        in-flight or just-evicted handshake), and None when nothing knows the peer."""
        for sm in (self._sm_public, self._sm_world):
            if sm is not None and sm.get_session(peer_id) is not None:
                return sm.p2p
        # Each pool is bound to one P2P instance, so the peer's pool resolves it.
        pool_name = self.peer_id_to_pool_name.get(peer_id)
        if pool_name is None:
            return None
        return self.pool_name_to_pool_triple[pool_name][2]

    def reset_rendezvous_tag(self) -> None:
        """Resets the rendezvous tag to its initial state."""
        self.rendezvous_tag = -1

    def membership_snapshot(self) -> WorldMembership:
        """Freeze the current world membership into an immutable per-cycle view. The pure
        membership (the two sets + world_node_peer_id) is read from sm_world, where it now
        lives next to the Sessions it gates; inspector_peer_id is the Coordinator's own
        orthogonal exception (D-8a). The sets are copied into frozensets so a snapshot
        already taken cannot change if the membership mutates later (and cannot be mutated
        by the routing itself). Before attach_session_policy builds sm_world the membership
        reads empty."""
        sm = self._sm_world
        if sm is None:
            world_agents, world_masters, world_node_peer_id = frozenset(), frozenset(), None
        else:
            world_agents = frozenset(sm.world_agents_set)
            world_masters = frozenset(sm.world_masters_set)
            world_node_peer_id = sm.world_node_peer_id
        return WorldMembership(
            world_agents=world_agents,
            world_masters=world_masters,
            world_node_peer_id=world_node_peer_id,
            inspector_peer_id=self.inspector_peer_id,
        )

    @staticmethod
    def _route_public(peer_info: dict, membership) -> "str | None":
        """Per-instance routing for the public network: by direction only, no
        membership and no perimeter cut. Returns the pool name, or None for an
        undefined direction (the caller skips it without disconnecting)."""
        direction = peer_info['direction']
        if direction == "inbound":
            return NodeConn.IN_PUBLIC
        if direction == "outbound":
            return NodeConn.OUT_PUBLIC
        return None

    @staticmethod
    def _route_world(peer_info: dict, membership: WorldMembership) -> "str | None":
        """Per-instance routing for the world network: classify the peer against the
        membership snapshot (precedence agent > node > master) and return its pool
        name, or None when the peer is not a known world member (the caller cuts it
        from the perimeter) or has an undefined direction."""
        peer_id = peer_info['id']
        is_agent = peer_id in membership.world_agents
        is_master = peer_id in membership.world_masters
        is_node = membership.world_node_peer_id is not None and peer_id == membership.world_node_peer_id
        is_inspector = membership.inspector_peer_id is not None and peer_id == membership.inspector_peer_id
        if not (is_node or is_master or is_agent or is_inspector):
            return None
        direction = peer_info['direction']
        if direction == "inbound":
            return NodeConn.IN_WORLD_AGENTS if is_agent else (
                NodeConn.IN_WORLD_NODE if is_node else NodeConn.IN_WORLD_MASTERS)
        if direction == "outbound":
            return NodeConn.OUT_WORLD_AGENTS if is_agent else (
                NodeConn.OUT_WORLD_NODE if is_node else NodeConn.OUT_WORLD_MASTERS)
        return None

    async def update(self) -> tuple[dict, dict]:
        """Coordinator per-cycle update. First refresh the world SessionManager's KNOWN
        roster (and the role/addrs it carries) from the rendezvous, so the perimeter and
        facade see this cycle's roster rather than the previous one and the KNOWN sync
        stays out of the pool/diff path. Then run the transport-driven per-instance
        reconcile: one immutable membership snapshot for the whole cycle, fanned out to
        each P2P instance in insertion order (public then world), merging the per-instance
        diffs. The pool_name keys are disjoint across the two instances (IN/OUT_PUBLIC vs
        IN/OUT_WORLD_*), so the merge is a plain dict.update, asserted to be sure."""
        self._sync_known_world_peers()

        self.pool_name_to_added_in_last_update = {}
        self.pool_name_to_removed_in_last_update = {}
        membership = self.membership_snapshot()
        self._cycle_membership = membership  # publish this cycle's snapshot for get_messages' gate

        for p2p_name, p2p in self.p2p_name_to_p2p.items():
            added_local, removed_local = await self._reconcile_instance(p2p_name, p2p, membership)
            assert self.pool_name_to_added_in_last_update.keys().isdisjoint(added_local.keys()), \
                "pool_name keys must be disjoint across P2P instances"
            assert self.pool_name_to_removed_in_last_update.keys().isdisjoint(removed_local.keys()), \
                "pool_name keys must be disjoint across P2P instances"
            self.pool_name_to_added_in_last_update.update(added_local)
            self.pool_name_to_removed_in_last_update.update(removed_local)

        return self.pool_name_to_added_in_last_update, self.pool_name_to_removed_in_last_update

    def _sync_known_world_peers(self) -> None:
        """Feed the world SessionManager the peers currently advertised by the rendezvous
        (``peer_id -> (addresses, app_role)``, minus ourselves) so it registers a KNOWN
        Session per known peer, refreshes role/addrs on the Sessions that exist, and
        prunes the vanished ones. World-only: the public instance carries no rendezvous
        membership."""
        if self._sm_world is None:
            return
        self_peer_id = self.p2p_world.peer_id
        known = {peer_id: (addrs, self.peer_id_to_misc.get(peer_id, 0))
                 for peer_id, addrs in self.peer_id_to_addrs.items()
                 if peer_id != self_peer_id}
        self._sm_world.sync_known_peers(known)

    async def _reconcile_instance(self, p2p_name: str, p2p: P2P, membership) -> tuple[dict, dict]:
        """Reconcile ONE P2P instance and report its per-pool added/removed diff.

        Asks this instance's SessionManager to classify its connected peers
        (sm.reconcile, which also applies the world perimeter cut), then for each of
        this instance's pools computes the new/lost set-difference and admits the new
        peers via add() — which keeps the cross-pool guard and the shared peer_id_to_*
        indices here in the Coordinator. Returns two dicts keyed by pool_name with only
        the pools that changed (empty pools omitted), like the legacy update().
        """
        added_local: dict = {}
        removed_local: dict = {}

        connected_peer_infos = p2p.get_connected_peers_info()
        if connected_peer_infos is None:
            return added_local, removed_local

        # The world manager classifies against the snapshot (and cuts the perimeter);
        # the public manager has no membership.
        if p2p_name == NodeConn.P2P_WORLD:
            sm, sm_membership = self._sm_world, membership
        else:
            sm, sm_membership = self._sm_public, None
        classified = await sm.reconcile(connected_peer_infos, sm_membership)

        # Iterate this instance's own pools (the Coordinator owns them), treating a
        # pool absent from `classified` as empty so removals are still detected.
        for pool_name, (pool, _, pool_p2p) in self.p2p_name_and_pool_name_to_pool_triple[p2p_name].items():
            connected_peer_ids = classified.get(pool_name, {}).keys()
            new_peer_ids = connected_peer_ids - pool
            lost_peer_ids = pool - connected_peer_ids

            # Clearing disconnected agents
            for lost_peer_id in lost_peer_ids:
                removed_local.setdefault(pool_name, set()).add(lost_peer_id)

            # Adding new agents
            for new_peer_id in new_peer_ids:
                peer_info = classified[pool_name][new_peer_id]
                if not self.add(peer_info, pool_name=pool_name):
                    continue
                added_local.setdefault(pool_name, set()).add(new_peer_id)

        return added_local, removed_local

    async def conn_routing_fcn(self, connected_peer_infos: list, p2p: P2P,
                               membership: WorldMembership = None) -> dict:
        """Routes connected peers to the correct connection pool based on their network and role (async).

        Args:
            connected_peer_infos: A list of dictionaries with information about connected peers.
            p2p: The P2P network object where the connections were found.
            membership: the immutable world-membership snapshot to classify against.
                Defaults to a fresh snapshot of the current membership when not passed
                (so a standalone call still works); update() passes the per-cycle one.

        Returns:
            A dictionary mapping pool names to a dictionary of peer IDs and their information.
        """
        if membership is None:
            membership = self.membership_snapshot()
        public = p2p == self.p2p_public
        classify = self._route_public if public else self._route_world
        result = {k: {} for k in self.p2p_to_pool_names[p2p]}
        for c in connected_peer_infos:
            pool_name = classify(c, membership)
            if pool_name is None:
                # A world peer that is not a known member gets cut from the perimeter;
                # the public side never returns None for a real (in/out) connection.
                if not public:
                    await NodeConn.disconnect(p2p, c['id'])
                continue
            result[pool_name][c['id']] = c
        return result

    # ── World-roster writers (S9: writes are centralized here) ─────────────────
    #
    # set_world / set_world_agents_list / set_world_masters_list / add_to_world_agents_list /
    # add_to_world_masters_list / remove_from_world_lists are the ONLY places the world roster
    # is mutated. node.py never touches world_agents_set/world_masters_set/world_node_peer_id
    # directly: it calls these Coordinator methods, which write sm_world. That is what keeps the
    # membership single-sourced in sm_world (S8). The rendezvous READ stays on the Coordinator
    # (set_world_agents_and_world_masters_lists_from_rendezvous, D-6); only the write lands here.
    # If a new roster mutation is ever needed, add it as a method here, never as a direct set.

    def set_world(self, world_peer_id: str | None) -> None:
        """Sets the peer ID of the world node.

        Args:
            world_peer_id: The peer ID of the world node, or None to clear it.
        """
        self._sm_world.world_node_peer_id = world_peer_id

    def get_world_node_peer_id(self) -> str | None:
        """Returns the PRIVATE peer ID of the world node, i.e. the REMOTE world host we joined,
        as written by set_world. This is deliberately NOT our own world/private peer-id:
        that one is Node.get_world_peer_id (conn[P2P_WORLD].peer_id). The name was
        get_world_peer_id, renamed to drop the collision with Node's same-named accessor
        that means the opposite thing.

        Args:
            None.

        Returns:
            The world node's peer ID, or None before sm_world is built / when unset.
        """
        return self._sm_world.world_node_peer_id if self._sm_world is not None else None

    def set_addresses_in_peer_info(self, peer_id: str, addresses: list[str]) -> None:
        """Updates the list of addresses for a given peer.

        Args:
            peer_id: The peer ID to update.
            addresses: A new list of addresses for the peer.
        """
        if self.in_connection_queues(peer_id):
            pool_name = self.get_pool_of(peer_id)
            assert pool_name is not None
            addrs = self.pool_name_to_peer_infos[pool_name][peer_id]['addrs']
            addrs.clear()  # Warning: do not allocate a new list, keep the current one (it is referenced by others)
            for _addrs in addresses:
                addrs.append(_addrs)

    def set_role(self, peer_id: str, new_role: int) -> None:
        """Updates the role of a peer and its associated role-based lists.

        Args:
            peer_id: The peer ID to update.
            new_role: The new role for the peer.
        """
        cur_role = self.get_role(peer_id)

        # Updating
        self.peer_id_to_misc[peer_id] = new_role

        if self.in_connection_queues(peer_id):
            pool_name = self.get_pool_of(peer_id)
            assert pool_name is not None
            self.pool_name_to_peer_infos[pool_name][peer_id]['misc'] = new_role

        # Updating
        if cur_role in self.role_to_peer_ids:
            if peer_id in self.role_to_peer_ids[cur_role]:
                self.role_to_peer_ids[cur_role].remove(peer_id)
            if len(self.role_to_peer_ids[cur_role]) == 0:
                del self.role_to_peer_ids[cur_role]
        if new_role not in self.role_to_peer_ids:
            self.role_to_peer_ids[new_role] = set()
        self.role_to_peer_ids[new_role].add(peer_id)

    def set_world_agents_list(self, world_agents_list_peer_infos: list[dict] | None) -> None:
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
            self._sm_world.world_agents_set = {x['id'] for x in world_agents_list_peer_infos}
            for x in world_agents_list_peer_infos:
                self.peer_id_to_addrs[x['id']] = x['addrs']
                self.set_role(x['id'], x['misc'])
        else:
            self._sm_world.world_agents_set = set()

    def remove_from_world_lists(self, peer_id: str) -> None:
        """Drops a peer from the world agents/masters lists and the role/address
        bookkeeping. The world host calls this when a member's world Session
        terminates, so the routing pass and the rendezvous publish stop
        advertising it."""
        self._sm_world.world_agents_set.discard(peer_id)
        self._sm_world.world_masters_set.discard(peer_id)
        self.peer_id_to_addrs.pop(peer_id, None)
        misc = self.peer_id_to_misc.pop(peer_id, None)
        if misc in self.role_to_peer_ids:
            self.role_to_peer_ids[misc].discard(peer_id)
            if not self.role_to_peer_ids[misc]:
                del self.role_to_peer_ids[misc]

    def set_world_masters_list(self, world_masters_list_peer_infos: list[dict] | None) -> None:
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
            self._sm_world.world_masters_set = {x['id'] for x in world_masters_list_peer_infos}
            for x in world_masters_list_peer_infos:
                self.peer_id_to_addrs[x['id']] = x['addrs']
                self.set_role(x['id'], x['misc'])
        else:
            self._sm_world.world_masters_set = set()

    def add_to_world_agents_list(self, peer_id: str, addrs: list[str], role: int = -1) -> None:
        """Adds a new world agent to the list.

        Args:
            peer_id: The peer ID of the new agent.
            addrs: A list of addresses for the new agent.
            role: The role assigned to the agent.
        """
        self._sm_world.world_agents_set.add(peer_id)

        # This assumes that the WORLD MASTER/AGENT BIT is the first one
        assert role & 1 == 1, "Expecting the first bit of the role to be 1 for world agents"
        assert role & 2 == 0, "Expecting the second bit of the role to be 0 for world agents"
        self.peer_id_to_addrs[peer_id] = addrs
        self.set_role(peer_id, role)

    def add_to_world_masters_list(self, peer_id: str, addrs: list[str], role: int = -1) -> None:
        """Adds a new world master to the list.

        Args:
            peer_id: The peer ID of the new master.
            addrs: A list of addresses for the new master.
            role: The role assigned to the master.
        """
        self._sm_world.world_masters_set.add(peer_id)

        # This assumes that the WORLD MASTER/AGENT BIT is the first one
        assert role & 1 == 1, "Expecting the first bit of the role to be 1 for world masters"
        assert role & 2 == 2, "Expecting the second bit of the role to be 1 for world masters"
        self.peer_id_to_addrs[peer_id] = addrs
        self.set_role(peer_id, role)

    def get_added_after_updating(self, pool_names: list[str] | None = None) -> dict[str, set] | set:
        """Retrieves the set of peers added after the last update cycle for specified pools.

        Args:
            pool_names: A list of pool names to check. If None, returns the per-pool mapping.

        Returns:
            A dictionary mapping pool names to sets of added peer IDs, or the full per-pool
            mapping when no pool names are given.
        """
        if pool_names is not None:
            ret = {}
            for p in pool_names:
                ret[p] = self.pool_name_to_added_in_last_update[p]
            return ret
        else:
            return self.pool_name_to_added_in_last_update

    def get_removed_after_updating(self, pool_names: list[str] | None = None) -> dict[str, set] | set:
        """Retrieves the set of peers removed after the last update cycle for specified pools.

        Args:
            pool_names: A list of pool names to check. If None, returns the per-pool mapping.

        Returns:
            A dictionary mapping pool names to sets of removed peer IDs, or the full per-pool
            mapping when no pool names are given.
        """
        if pool_names is not None:
            ret = {}
            for p in pool_names:
                ret[p] = self.pool_name_to_removed_in_last_update[p]
            return ret
        else:
            return self.pool_name_to_removed_in_last_update

    def size(self, pool_names: list[str] | None = None) -> int:
        """Returns the total number of connections across all specified pools.

        Args:
            pool_names: A list of pool names to sum the size of. If None, returns the total size of all pools.

        Returns:
            The total number of connections.
        """
        if pool_names is None:
            c = 0
            for v in self.pool_name_to_pool_triple.values():
                c += len(v[0])
            return c
        else:
            c = 0
            for p in pool_names:
                c += len(self.pool_name_to_pool_triple[p][0])
            return c

    def is_connected(self, peer_id: str, pool_names: list[str] | None = None) -> bool:
        """Checks if a peer is connected in any of the specified pools.

        Args:
            peer_id: The peer ID to check.
            pool_names: A list of pool names to search within. If None, searches all pools.

        Returns:
            True if the peer is found in any of the pools, otherwise False.
        """
        if pool_names is None:
            return peer_id in self.peer_id_to_pool_name
        else:
            for p in pool_names:
                if peer_id in self.peer_id_to_pool_name and p == self.peer_id_to_pool_name[peer_id]:
                    return True
            return False

    def is_public(self, peer_id: str) -> bool:
        """Checks if a peer is connected via the public network.

        Args:
            peer_id: The peer ID to check.

        Returns:
            True if the peer is in a public pool, otherwise False.
        """
        pool_name = self.get_pool_of(peer_id)
        return pool_name in NodeConn.PUBLIC

    def is_world_master(self, peer_id: str) -> bool:
        """Checks if a peer is a world master.

        Args:
            peer_id: The peer ID to check.

        Returns:
            True if the peer is in a world master pool, otherwise False.
        """
        pool_name = self.get_pool_of(peer_id)
        return pool_name in NodeConn.WORLD_MASTERS

    def is_world_node(self, peer_id: str) -> bool:
        """Checks if a peer is the world node.

        Args:
            peer_id: The peer ID to check.

        Returns:
            True if the peer is in a world node pool, otherwise False.
        """
        pool_name = self.get_pool_of(peer_id)
        return pool_name in NodeConn.WORLD_NODE

    def is_in_world(self, peer_id: str) -> bool:
        """Checks if a peer is connected to the world network.

        Args:
            peer_id: The peer ID to check.

        Returns:
            True if the peer is in any world pool, otherwise False.
        """
        pool_name = self.get_pool_of(peer_id)
        return pool_name in NodeConn.WORLD

    def get_role(self, peer_id: str) -> int:
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

    def get_addrs(self, peer_id: str) -> list[str] | None:
        """Retrieves the list of addresses for a given peer.

        Args:
            peer_id: The peer ID to query.

        Returns:
            A list of addresses for the peer.
        """
        if peer_id in self.peer_id_to_addrs:
            return self.peer_id_to_addrs.get(peer_id)
        else:
            return None

    def in_connection_queues(self, peer_id: str) -> bool:
        """Checks if a peer ID exists in any connection pool.

        Args:
            peer_id: The peer ID to check.

        Returns:
            True if the peer is found in any pool, otherwise False.
        """
        return peer_id in self.peer_id_to_pool_name

    def find_addrs_by_role(self, role: int, return_peer_ids_too: bool = False,
                           discard_yourself: bool = False) -> list | tuple[list, list]:
        """Finds all addresses of peers with a specific role.

        Args:
            role: The integer role to search for.
            return_peer_ids_too: A boolean to also return the peer IDs.
            discard_yourself: True to avoid reporting your info on the returned data.

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
            if discard_yourself and (peer_id == self.p2p_public.peer_id or peer_id == self.p2p_world.peer_id):
                continue
            addrs = self.get_addrs(peer_id)
            if addrs is not None:
                ret_addrs.append(addrs)
                ret_peer_ids.append(peer_id)
        if not return_peer_ids_too:
            return ret_addrs
        else:
            return ret_addrs, ret_peer_ids

    def count_by_role(self, role: int) -> int:
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

    def get_all_connected_peer_infos(self, pool_names: list[str] | set[str]) -> list[dict]:
        """Retrieves a list of all peer info dictionaries for the specified pools.

        Args:
            pool_names: A list or set of pool names to query.

        Returns:
            A list of dictionaries containing peer information.
        """
        ret = []
        for p in pool_names:
            ret += list(self.pool_name_to_peer_infos[p].values())
        return ret

    async def set_world_agents_and_world_masters_lists_from_rendezvous(self) -> None:
        """Updates the lists of world agents and masters using data from the rendezvous topic (async)."""
        rendezvous_state = self.p2p_world.get_rendezvous_peers_info()

        if rendezvous_state is not None:
            tag = rendezvous_state.get('update_count', -1)

            if tag > self.rendezvous_tag:
                self.rendezvous_tag = tag
                rendezvous_peer_infos = rendezvous_state.get('peers', [])

                world_agents_peer_infos = []
                world_masters_peer_infos = []

                log.cpool(f"Rendezvous peer infos (tag: {tag}, peers: "
                          f"{len(rendezvous_peer_infos)} peers)", sub=self.p2p_world.log_sub)
                log.cpool(f"Rendezvous message included peer IDs: "
                          f"{[p['id'] for p in rendezvous_peer_infos]})", sub="prv")

                for c in rendezvous_peer_infos:
                    if c['addrs'] is None:
                        log.cpool(f"Skipping a peer with None addrs (unexpected)", sub=self.p2p_world.log_sub)
                        continue
                    # if len(c['addrs']) == 0:
                    #    log.cpool(f"Skipping a peer with zero-length addrs-list (unexpected)")
                    #    continue
                    if (c['misc'] & 1) == 1 and (c['misc'] & 2) == 0:
                        world_agents_peer_infos.append(c)
                    elif (c['misc'] & 1) == 1 and (c['misc'] & 2) == 2:
                        world_masters_peer_infos.append(c)
                    else:
                        log.critical("Unexpected value of the 'misc' field: " + str(c), sub=self.p2p_world.log_sub)

                # Updating lists
                self.set_world_agents_list(world_agents_peer_infos)
                self.set_world_masters_list(world_masters_peer_infos)

    async def get_cv_hash_from_last_token(self, peer_id: str) -> str | None:
        """Retrieves the CV hash from the last token received from a peer (async).

        Args:
            peer_id: The peer ID to query.

        Returns:
            The CV hash string, or None if not found.
        """
        token = self.get_last_token(peer_id)
        if token is not None:
            _, cv_hash = await self.verify_token(token, peer_id)
            return cv_hash
        else:
            return None

    async def remove_all_world_agents(self) -> None:
        """Removes all connected world agents from the pools and role lists (async)."""
        peer_infos = self.get_all_connected_peer_infos(NodeConn.WORLD)
        for c in peer_infos:
            peer_id = c['id']
            await self.remove(peer_id)
            if peer_id in self.peer_id_to_addrs:
                del self.peer_id_to_addrs[peer_id]
            for role, peer_ids in self.role_to_peer_ids.items():
                if peer_id not in peer_ids:
                    continue
                if role & 1 == 1:
                    peer_ids.remove(peer_id)
