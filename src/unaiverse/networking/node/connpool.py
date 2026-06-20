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


# A SessionInbound is a per-stage Session message lifted off the inbound queue
# without going through Msg.from_bytes or JWT verification. Identity is established
# inside Stage-2 of the Session handshake (sig_a + signed_peer_record), not here.
# `sess_msg` is a decoded SessMsg (see SessMsg.from_bytes) — the Session FSM never
# sees protobuf, which keeps session.py/session_manager.py Pyodide-portable.
SessionInbound = namedtuple("SessionInbound", ["peer_id", "protocol_id", "sess_msg"])

# libp2p protocol IDs that belong to the Session abstraction. Keep in sync with
# session/session.py SESSION_PROTOCOL_IDS and unailib/types.go Session*Protocol.
SESSION_PROTOCOL_IDS = frozenset({
    "/unaiverse/session/hello/1.0.0",
    "/unaiverse/session/profile/1.0.0",
    "/unaiverse/session/world/1.0.0",
    "/unaiverse/session/control/1.0.0",
})


# `SessMsg` owns its own wire codec now (SessMsg.to_bytes / SessMsg.from_bytes, in
# session/sess_msg.py), so connpool no longer imports session_pb2: the protobuf
# dependency lives inside those methods (lazy import) and never at this module top.
# These two imports are the session layer's portable seam — SessMsg for send/decode,
# _sess_msg for the P2P_PUBLIC/P2P_WORLD aliases used below.
from unaiverse.networking.session.sess_msg import SessMsg  # noqa: E402
from unaiverse.networking.session import sess_msg as _sess_msg  # noqa: E402


# Immutable per-cycle view of the world membership that the perimeter routing
# classifies against. Built once per update cycle so conn_routing_fcn reads a single
# consistent snapshot instead of the Coordinator's mutable sets — and so the same view
# can be handed down to each per-instance reconcile when the perimeter moves into the
# SessionManagers. The world side is fed by the rendezvous; the public side ignores it.
WorldMembership = namedtuple(
    "WorldMembership",
    ["world_agents", "world_masters", "world_node_peer_id", "inspector_peer_id"],
)


class ConnectionPools:

    def __init__(self, max_connections: int, pool_name_to_p2p_name_and_ratio: dict[str, list[str | float]],
                 p2p_name_to_p2p: dict[str, P2P],
                 public_key: str | None = None, token: str | None = None) -> None:
        """Initializes a new instance of the ConnectionPools class.

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

        # Indices rooted around the peer ID
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

    def membership_snapshot(self) -> "WorldMembership | None":
        """The immutable per-cycle world-membership view handed to conn_routing_fcn.
        The base pool has no world membership, so it returns None; NodeConn overrides
        it with the real snapshot."""
        return None

    async def conn_routing_fcn(self, connected_peer_infos: list, p2p: P2P,
                               membership: "WorldMembership | None" = None) -> dict:
        """A placeholder function that must be implemented to route connected peers to the correct pool (async).

        Args:
            connected_peer_infos: A list of dictionaries containing information about connected peers.
            p2p: The P2P network object from which the peers were connected.
            membership: the immutable world-membership snapshot to classify against.

        Returns:
            A dictionary mapping pool names to a dictionary of peer IDs and their information.
        """
        raise NotImplementedError("You must implement conn_routing_fcn!")

    async def __connect(self, p2p: P2P, addresses: list[str]) -> tuple[str | None, bool]:
        """Establishes a connection to a peer via a P2P network (async).

        Args:
            p2p: The P2P network object to use for the connection.
            addresses: A list of addresses of the peer to connect to.

        Returns:
            A tuple containing the peer ID and a boolean indicating if the connection was established through a relay.
        """

        force: str | None = None
        # force: str | None = '/tcp'
        if force is not None:
            _addresses = [a for a in addresses if force in a]
            addresses.clear()
            for a in _addresses:
                addresses.append(a)

        def sort_priority(s):
            if "/udp/" in s:
                return 0  # Highest priority
            elif "/ws/" in s:
                return 1  # Middle priority
            else:
                return 2  # Lowest priority

        # Sort in place (sorting does not seem to have an effect, but it makes things more readable)
        kept_addresses = sorted(
            [s for s in addresses if "127.0.0.1" not in s and ("/tcp/" not in s or "/ws/" in s)],
            key=sort_priority
        )
        discarded_addresses = sorted(
            [s for s in addresses if "127.0.0.1" not in s and ("/tcp/" in s and "/ws/" not in s)],
            key=sort_priority
        )
        if len(kept_addresses) > 0:
            addresses = kept_addresses
        else:
            addresses = discarded_addresses
            discarded_addresses = []

        log.cpool(f"Connecting to {addresses}", sub=p2p.log_sub)

        if addresses is None or len(addresses) == 0:
            log.error(f"Connection failed! (not-event tried, invalid addresses: {addresses})",
                      sub=p2p.log_sub)
            return None, False

        try:
            # Off the event loop: the dial can block up to its timeout, and other
            # sessions must keep ticking meanwhile (e.g. during a re-dial).
            winning_addr_info_dict = await asyncio.to_thread(p2p.connect_to, addresses)
            peer_id = winning_addr_info_dict.get('ID')
            addrs = winning_addr_info_dict.get('Addrs')
            assert addrs is not None and isinstance(addrs, (tuple, list))
            connected_addr_str = addrs[0]
            through_relay = '/p2p-circuit/' in connected_addr_str
            log.cpool(f"Connected to peer {peer_id} via {connected_addr_str} "
                      f"(through relay: {through_relay})", sub=p2p.log_sub)

            return peer_id, through_relay
        except P2PError:
            log.error(f"Connection failed!", sub=p2p.log_sub)
            if len(discarded_addresses) > 0:
                return await self.__connect(p2p, discarded_addresses)
            else:
                return None, False

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

    async def connect(self, addresses: list[str], p2p_name: str) -> tuple[str | None, bool]:
        """Connects to a peer on a specified P2P network (async).

        Args:
            addresses: A list of addresses of the peer to connect to.
            p2p_name: The name of the P2P network to use.

        Returns:
            A tuple containing the peer ID of the connected peer and a boolean indicating if a relay was used.
        """
        p2p = self.p2p_name_to_p2p[p2p_name]

        # Connecting
        peer_id, through_relay = await self.__connect(p2p, addresses)
        return peer_id, through_relay
    
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
            disc = await ConnectionPools.disconnect(p2p, peer_id)
            pool.remove(peer_id)
            del self.pool_name_to_peer_infos[pool_name][peer_id]
            del self.peer_id_to_pool_name[peer_id]
            if peer_id in self.peer_id_to_token:
                del self.peer_id_to_token[peer_id]

            # Remember to NOT del peer_id_to_misc[peer_id]!
            return disc
        else:
            return False

    def get_all_connected_peer_infos(self, pool_name: str) -> list[dict]:
        """Retrieves a list of peer information dictionaries for a given pool.

        Args:
            pool_name: The name of the pool to query.

        Returns:
            A list of dictionaries, each containing information about a peer in the pool.
        """
        return list(self.pool_name_to_peer_infos[pool_name].values())

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

    async def update(self) -> tuple[dict, dict]:
        """Per-cycle reconcile of the connection pools. The base pool has no world
        membership nor per-instance reconcile; the real loop is implemented by NodeConn
        (the Coordinator), which owns the SessionManagers and the membership snapshot."""
        raise NotImplementedError("ConnectionPools.update is implemented by NodeConn")

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
            handshake (sig_b / sig_a / signed_peer_record).

        Args:
            p2p_name: The name of the P2P network to fetch messages from.
            allowed_not_connected_peers: An optional set of peer IDs to allow chat
                messages from even if they are not yet in the pools. Session messages
                are never gated on pool membership — the handshake itself is what
                enrols a peer.
        """
        # Pop all messages
        p2p = self[p2p_name]
        byte_messages = p2p.pop_messages()  # Pop all messages
        # Process the list of message dictionaries
        processed_messages: list[Msg] = []
        session_messages: list[SessionInbound] = []

        if len(byte_messages) > 0:
            self.__last_recv_time = time.time()

        for i, msg_dict in enumerate(byte_messages):
            msg_dict: dict
            try:
                # Extract and validate required fields from the Go message structure.
                # The binary transport delivers "data" already as raw bytes (no base64).
                # Go structure: {"from":"Qm...", "data":<bytes>, "protocol":"..."}
                verified_sender_id = msg_dict.get("from")
                decoded_data = msg_dict.get("data")
                protocol_id = msg_dict.get("protocol") or ""

                # ── Session-protocol fast path ────────────────────────────────
                # Bypass Msg.from_bytes + JWT verify; Session handshake establishes
                # identity itself in Stage-2 (sig_a + signed_peer_record).
                if protocol_id in SESSION_PROTOCOL_IDS:
                    try:
                        sess_msg = SessMsg.from_bytes(decoded_data, protocol_id)
                    except Exception as codec_err:
                        log.error(f"Failed to decode session message on {protocol_id} "
                                  f"from {verified_sender_id}: {codec_err}", sub=p2p.log_sub)
                        continue
                    session_messages.append(SessionInbound(
                        peer_id=verified_sender_id,
                        protocol_id=protocol_id,
                        sess_msg=sess_msg,
                    ))
                    continue

                # ── Legacy chat path ─────────────────────────────────────────
                # Attempt to create the higher-level Msg object
                # This assumes Msg.from_bytes can parse your message protocol from decoded_data
                # and that Msg objects store sender, type, channel intrinsically or can be set.
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

    async def send_session(self, peer_id: str, protocol_id: str, sess_msg: SessMsg,
                           p2p_name: str) -> bool:
        """Serialize `sess_msg` and open a fresh outbound libp2p stream on
        `protocol_id` to `peer_id` on the `p2p_name` instance, writing a single
        length-prefixed payload and closing the stream. Returns True on send success.

        The Session FSM passes a `SessMsg` (not bytes) and a `p2p_name` (not a P2P
        object) so it stays free of protobuf and of any Python-only P2P handle — the
        JS `NodeConn` mirror exposes the same signature. Identity is bound by the
        Session handshake (sig + signed_peer_record), so no JWT piggyback is inserted.
        """
        if protocol_id not in SESSION_PROTOCOL_IDS:
            raise ValueError(f"send_session: protocol_id '{protocol_id}' is not a Session protocol")
        p2p = self[p2p_name]
        try:
            payload = sess_msg.to_bytes()
        except Exception as e:
            log.error(f"send_session(...{peer_id[-5:]}, {protocol_id}) encode failed: {e}", sub=p2p.log_sub)
            return False
        try:
            # Off the event loop (same pattern as `send`): the Go side opens a
            # fresh stream and can block for seconds toward a dead peer, which
            # would freeze every other session's keep-alive.
            return await asyncio.to_thread(p2p.send_on_protocol, peer_id, protocol_id, payload)
        except Exception as e:
            log.error(f"send_session(...{peer_id[-5:]}, {protocol_id}) failed: {e}", sub=p2p.log_sub)
            return False

    # ── Async crypto / ping interface (legacy seam) ──────────────────────────────
    # The Session/SessionManager no longer reach the transport through these conn
    # wrappers: they call the high-level methods on their injected `P2P` instance
    # directly (see p2p.py). These thin wrappers are kept for now (orphaned) to
    # minimise churn in this step; they delegate to the synchronous Go FFI on the
    # `p2p_name` instance, keyed by name. ─────────────────────────────────────────

    def get_local_peer_id(self, p2p_name: str) -> str:
        """Our own libp2p peer-id on the `p2p_name` instance."""
        return self[p2p_name].peer_id

    async def sign_with_host_key(self, p2p_name: str, data: bytes):
        """Ed25519-sign `data` with the host key of the `p2p_name` instance."""
        return self[p2p_name]._sign_with_host_key_sync(data)

    async def verify_peer_signature(self, p2p_name: str, peer_id: str,
                                    data: bytes, signature: bytes) -> bool:
        """Verify `signature` over `data` under `peer_id`'s public key."""
        return self[p2p_name]._verify_peer_signature_sync(peer_id, data, signature)

    async def verify_signed_peer_record(self, p2p_name: str, envelope: bytes,
                                        expected_peer_id: str) -> bool:
        """Verify a SignedPeerRecord envelope binds `expected_peer_id`."""
        return self[p2p_name]._verify_signed_peer_record_sync(envelope, expected_peer_id)

    async def generate_signed_peer_record(self, p2p_name: str):
        """SignedPeerRecord envelope bytes over the `p2p_name` host's addresses."""
        return self[p2p_name]._generate_signed_peer_record_sync()

    def passed_time_since_last_communication(self):
        passed = time.time() - max(self.__last_sent_time, self.__last_recv_time)
        return 0. if passed < 0. else passed

    def get_added_after_updating(self, pool_name: str | None = None) -> set | dict:
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

    def get_removed_after_updating(self, pool_name: str | None = None) -> set | dict:
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

    def get_last_token(self, peer_id: str) -> str | None:
        """Retrieves the last known token for a given peer.

        Args:
            peer_id: The peer ID to query.

        Returns:
            The token string if found, otherwise None.
        """
        return self.peer_id_to_token[peer_id] if peer_id in self.peer_id_to_token else None

    def is_connected(self, peer_id: str, pool_name: str | None = None) -> bool:
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

    def size(self, pool_name: str | None = None) -> int:
        """Returns the number of connections in a specific pool or the total number across all pools.

        Args:
            pool_name: An optional pool name to get the size of. If None, returns the total size.

        Returns:
            The size of the pool or the total number of connections.
        """
        if pool_name is not None:
            return len(self.pool_name_to_pool_triple[pool_name][0])
        else:
            c = 0
            for v in self.pool_name_to_pool_triple.values():
                c += len(v[0])
            return c

    def _p2p_of(self, peer_id: str) -> "P2P | None":
        """Resolve which P2P instance a peer is reachable on, for the peer_id-only I/O
        (send / subscribe / publish). The base pool resolves it from the pool a connected
        peer lives in (each pool is bound to one P2P instance); NodeConn overrides this to
        ask the SessionManagers first (Session-anchored routing) and falls back here."""
        pool_name = self.peer_id_to_pool_name.get(peer_id)
        if pool_name is None:
            return None
        return self.pool_name_to_pool_triple[pool_name][2]

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

    # P2p names (canonical strings live in the protobuf-free sess_msg so the
    # Session FSM can read them without importing this conn layer)
    P2P_PUBLIC = _sess_msg.P2P_PUBLIC
    P2P_WORLD = _sess_msg.P2P_WORLD

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
        super().__init__(max_connections=max_connections,
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

        # These are the list of all the possible agents that might try to connect when we are in world
        self.world_agents_set = set()
        self.world_masters_set = set()
        self.world_agents_and_world_masters_set = set()
        self.world_node_peer_id = None
        self.inspector_peer_id = None
        self.role_to_peer_ids = {}
        self.peer_id_to_addrs = {}

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

    def attach_session_policy(self, local_profile_factory, decision_hook, token_factory=None) -> None:
        """Build and own the two per-instance SessionManagers, now that the Node
        policy born late is available (the profile factory samples self.profile, the
        decision hook reads world/agent/allow-list state). Each manager owns its P2P
        instance and receives a cross-instance crypto capability over the *other*
        instance: a public Session signs its other-network artifacts (SignedPeerRecord
        / world-grant) with the WORLD host key and vice-versa."""
        from unaiverse.networking.session import SessionManager
        from unaiverse.networking.p2p import CrossInstanceCrypto
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
            xcrypto=CrossInstanceCrypto(self.p2p_world),
            token_verifier=self.verify_token,
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
            xcrypto=CrossInstanceCrypto(self.p2p_public),
            token_verifier=self.verify_token,
            token_factory=token_factory,
            routing_classifier=NodeConn._route_world,
            perimeter_disconnect=lambda peer_id: ConnectionPools.disconnect(self.p2p_world, peer_id),
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

    def _p2p_of(self, peer_id: str):
        """Resolve a peer's P2P instance by asking the SessionManagers first: the SM
        that holds a Session for this peer-id owns the instance the peer lives on
        (public before world, matching the cross-pool first-network-wins). Falls back to
        the base resolution from the connection pool for a peer that is connected but has
        no Session yet (an in-flight or just-evicted handshake)."""
        for sm in (self._sm_public, self._sm_world):
            if sm is not None and sm.get_session(peer_id) is not None:
                return sm.p2p
        return super()._p2p_of(peer_id)

    def reset_rendezvous_tag(self) -> None:
        """Resets the rendezvous tag to its initial state."""
        self.rendezvous_tag = -1

    def membership_snapshot(self) -> WorldMembership:
        """Freeze the current world membership into an immutable per-cycle view. The
        sets are copied into frozensets so a snapshot already taken cannot change if
        the membership mutates later (and cannot be mutated by the routing itself)."""
        return WorldMembership(
            world_agents=frozenset(self.world_agents_set),
            world_masters=frozenset(self.world_masters_set),
            world_node_peer_id=self.world_node_peer_id,
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
                    await ConnectionPools.disconnect(p2p, c['id'])
                continue
            result[pool_name][c['id']] = c
        return result

    def set_world(self, world_peer_id: str | None) -> None:
        """Sets the peer ID of the world node.

        Args:
            world_peer_id: The peer ID of the world node, or None to clear it.
        """
        self.world_node_peer_id = world_peer_id

    def get_world_peer_id(self) -> str | None:
        """Returns the peer ID of the world node.

        Args:
            None.

        Returns:
            The world node's peer ID.
        """
        return self.world_node_peer_id

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
            self.world_agents_set = {x['id'] for x in world_agents_list_peer_infos}
            for x in world_agents_list_peer_infos:
                self.peer_id_to_addrs[x['id']] = x['addrs']
                self.set_role(x['id'], x['misc'])
        else:
            self.world_agents_set = set()

        self.world_agents_and_world_masters_set = self.world_agents_set | self.world_masters_set

    def remove_from_world_lists(self, peer_id: str) -> None:
        """Drops a peer from the world agents/masters lists and the role/address
        bookkeeping. The world host calls this when a member's world Session
        terminates, so the routing pass and the rendezvous publish stop
        advertising it."""
        self.world_agents_set.discard(peer_id)
        self.world_masters_set.discard(peer_id)
        self.world_agents_and_world_masters_set = self.world_agents_set | self.world_masters_set
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
            self.world_masters_set = {x['id'] for x in world_masters_list_peer_infos}
            for x in world_masters_list_peer_infos:
                self.peer_id_to_addrs[x['id']] = x['addrs']
                self.set_role(x['id'], x['misc'])
        else:
            self.world_masters_set = set()

        self.world_agents_and_world_masters_set = self.world_agents_set | self.world_masters_set

    def add_to_world_agents_list(self, peer_id: str, addrs: list[str], role: int = -1) -> None:
        """Adds a new world agent to the list.

        Args:
            peer_id: The peer ID of the new agent.
            addrs: A list of addresses for the new agent.
            role: The role assigned to the agent.
        """
        self.world_agents_set.add(peer_id)

        # This assumes that the WORLD MASTER/AGENT BIT is the first one
        assert role & 1 == 1, "Expecting the first bit of the role to be 1 for world agents"
        assert role & 2 == 0, "Expecting the second bit of the role to be 0 for world agents"
        self.peer_id_to_addrs[peer_id] = addrs
        self.set_role(peer_id, role)
        self.world_agents_and_world_masters_set = self.world_agents_set | self.world_masters_set

    def add_to_world_masters_list(self, peer_id: str, addrs: list[str], role: int = -1) -> None:
        """Adds a new world master to the list.

        Args:
            peer_id: The peer ID of the new master.
            addrs: A list of addresses for the new master.
            role: The role assigned to the master.
        """
        self.world_masters_set.add(peer_id)

        # This assumes that the WORLD MASTER/AGENT BIT is the first one
        assert role & 1 == 1, "Expecting the first bit of the role to be 1 for world masters"
        assert role & 2 == 2, "Expecting the second bit of the role to be 1 for world masters"
        self.peer_id_to_addrs[peer_id] = addrs
        self.set_role(peer_id, role)
        self.world_agents_and_world_masters_set = self.world_agents_set | self.world_masters_set

    def get_added_after_updating(self, pool_names: list[str] | None = None) -> dict[str, set] | set:
        """Retrieves the set of peers added after the last update cycle for specified pools.

        Args:
            pool_names: A list of pool names to check. If None, checks all pools.

        Returns:
            A dictionary mapping pool names to sets of added peer IDs.
        """
        if pool_names is not None:
            ret = {}
            for p in pool_names:
                ret[p] = super().get_added_after_updating(p)
            return ret
        else:
            return super().get_added_after_updating()

    def get_removed_after_updating(self, pool_names: list[str] | None = None) -> dict[str, set] | set:
        """Retrieves the set of peers removed after the last update cycle for specified pools.

        Args:
            pool_names: A list of pool names to check. If None, checks all pools.

        Returns:
            A dictionary mapping pool names to sets of removed peer IDs.
        """
        if pool_names is not None:
            ret = {}
            for p in pool_names:
                ret[p] = super().get_removed_after_updating(p)
            return ret
        else:
            return super().get_removed_after_updating()

    def size(self, pool_names: list[str] | None = None) -> int:
        """Returns the total number of connections across all specified pools.

        Args:
            pool_names: A list of pool names to sum the size of. If None, returns the total size of all pools.

        Returns:
            The total number of connections.
        """
        if pool_names is None:
            return super().size()
        else:
            c = 0
            for p in pool_names:
                c += super().size(p)
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
            return super().is_connected(peer_id)
        else:
            for p in pool_names:
                if super().is_connected(peer_id, p):
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
            ret += super().get_all_connected_peer_infos(p)
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

    async def remove(self, peer_id: str) -> None:
        """Removes a peer and its associated information from all lists and pools (async).

        Args:
            peer_id: The peer ID to remove.
        """
        await super().remove(peer_id)
        # if peer_id in self.peer_id_to_addrs:
        #    del self.peer_id_to_addrs[peer_id]

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

    async def subscribe(self, peer_id: str, channel: str, default_p2p_name: str | None = None) -> bool:
        """Subscribes to a channel, defaulting to the world P2P network if a network is not specified (async).

        Args:
            peer_id: The peer ID associated with the channel.
            channel: The channel to subscribe to.
            default_p2p_name: An optional P2P name to use for the subscription.

        Returns:
            True if successful, False otherwise.
        """
        return await super().subscribe(peer_id, channel,
                                       default_p2p_name=NodeConn.P2P_WORLD
                                       if default_p2p_name is None else default_p2p_name)

    async def get_messages(self, p2p_name: str, allowed_not_connected_peers: set | None = None) -> list[Msg]:
        """Retrieves messages, allowing for messages from known world agents and masters even if not in a
        connection pool (async).

        Args:
            p2p_name: The name of the P2P network to get messages from.
            allowed_not_connected_peers: This parameter is ignored in this implementation.

        Returns:
            A list of verified and processed message objects.
        """
        assert allowed_not_connected_peers is None, "This param (allowed_not_connected_peers is ignored in NodeConn"
        return await super().get_messages(p2p_name,
                                          allowed_not_connected_peers=self.world_agents_and_world_masters_set)
