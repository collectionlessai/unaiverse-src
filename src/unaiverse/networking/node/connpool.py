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
import time
import math
import asyncio
from collections import namedtuple
from unaiverse.utils.logger import log
from unaiverse.networking.p2p.messages import Msg
from unaiverse.networking.p2p.p2p import P2P, P2PError
from unaiverse.networking.node.tokens import TokenVerifier
from unaiverse.networking.session.session_manager import SessionManager


# sess_msg.py (the session layer's leaf module) is the single source of truth for the
# session-protocol value objects. The inbound codec itself (pop + decode) now lives in
# P2P.get_messages, so connpool only needs the SessionInbound value object it carries
# through and the WORLD_HANDOFF_COMPLETE reason code for the handoff close. The wire codec
# (SessMsg) and its protocol IDs (SESSION_PROTOCOL_IDS) are no longer referenced here.
from unaiverse.networking.session.sess_msg import (
    SessionInbound, WORLD_HANDOFF_COMPLETE, WORLD_DISENGAGEMENT,
)


# Immutable per-cycle view of the world membership that the perimeter routing
# classifies against. Built once per update cycle so each per-instance reconcile reads a
# single consistent snapshot instead of the Coordinator's mutable sets, the same frozen
# view handed both to update()'s reconcile and to the get_messages chat gate. The world
# side is fed by the rendezvous; the public side ignores it.
WorldMembership = namedtuple(
    "WorldMembership",
    ["world_agents", "world_masters", "world_node_peer_id", "inspector_peer_id"],
)


# Dial budget of the world-join redeemer: a single attempt (one p2p.connect already
# spans every address in parallel; multi-attempt retries belong to the rejoin
# watchdog, which re-arms with a FRESH grant) bounded by the grant window. The floor
# lets a nearly-expired grant still dial (expiry is only checked at grant receipt,
# and the host-side pre-admission outlives it); the ceiling covers one full dial
# timeout plus margin, so a wedged dial fails while a watchdog retry is still cheap.
_WORLD_DIAL_MIN_BUDGET_S = 15.0
_WORLD_DIAL_MAX_BUDGET_S = 35.0


class HandoffCorrelation:
    """Ledger of an in-flight world-join handoff: the public carrier that yielded the
    grant, correlated with the world Session that redeems it, kept as twin indexes
    (forward ``public -> world``, reverse ``world -> public``) with the three
    lifecycle operations. Shared VERBATIM by the real Coordinator and the
    test-harness mirror (NodeFakeConn), which used to hand-copy this logic and
    drifted more than once; one implementation is the lockstep guarantee.

    External effects are injected as callables so the ledger stays pure bookkeeping
    (and the harness can wire its fakes):

      * ``is_world_established(world_peer_id)``: True when the redemption already
        reached ESTABLISHED — rollback then no-ops, the completing join owns the
        cleanup (see ``rollback``).
      * ``abort_world(world_peer_id)``: awaitable, aborts the in-flight redemption.
      * ``current_world()`` / ``clear_world()``: read / un-point the pool-routing
        pointer that ``start_world_handoff`` set optimistically.
    """

    def __init__(self, is_world_established, abort_world, current_world, clear_world):
        self.pending_world_handoff: dict[str, str] = {}
        self.world_to_public: dict[str, str] = {}
        self._is_world_established = is_world_established
        self._abort_world = abort_world
        self._current_world = current_world
        self._clear_world = clear_world

    def start(self, public_peer_id: str, world_peer_id: str) -> None:
        """Record the correlation at handoff-initiate (both directions)."""
        self.pending_world_handoff[public_peer_id] = world_peer_id
        self.world_to_public[world_peer_id] = public_peer_id

    def is_pending(self, world_peer_id: str) -> bool:
        """True if this world peer-id is the world side of an in-flight handoff."""
        return world_peer_id in self.world_to_public

    async def rollback(self, peer_id: str) -> None:
        """Roll back the correlation for a terminal Session, whichever side died.

        When the PUBLIC carrier dies mid-handoff, the fate of the correlated world
        Session decides everything: already ESTABLISHED -> NO-OP (the join is
        completing; rolling back would strand it half-rolled-back — in-world with
        no world host — and the perimeter would then coercively cut our own host
        with no event, a permanent wedge); still in flight -> pop both indexes AND
        abort the redemption (a dial/handshake left running would complete into a
        headless join). Also un-points the routing pointer when the rolled-back
        world peer is the current one."""
        pending_world = self.pending_world_handoff.get(peer_id)
        if pending_world is not None and self._is_world_established(pending_world):
            return
        world_peer_id = self.pending_world_handoff.pop(peer_id, None)
        if world_peer_id is not None:
            self.world_to_public.pop(world_peer_id, None)
            await self._abort_world(world_peer_id)
        public_peer_id = self.world_to_public.pop(peer_id, None)
        if public_peer_id is not None:
            self.pending_world_handoff.pop(public_peer_id, None)
        rolled_back_world = world_peer_id if world_peer_id is not None else (
            peer_id if public_peer_id is not None else None)
        if rolled_back_world is not None and rolled_back_world == self._current_world():
            self._clear_world()

    def complete(self, world_peer_id: str) -> "str | None":
        """Drop the correlation on success. Returns the public carrier's peer-id for
        the caller to retire (GOAWAY on its live transport + pool evict), or None if
        this world peer-id was not part of a handoff."""
        public_peer_id = self.world_to_public.pop(world_peer_id, None)
        if public_peer_id is not None:
            self.pending_world_handoff.pop(public_peer_id, None)
        return public_peer_id


class NodeConn:
    """The networking Coordinator. Owns the two per-instance P2P handles, the two
    per-instance SessionManagers and the cross-instance world membership, and exposes
    the pool/routing facade the hosted entity speaks to. This used to be a NodeConn
    subclass over a generic ``ConnectionPools`` base, but the base was never
    instantiated on its own (only NodeConn was), so the two have been collapsed into a
    single class: the pool/transport primitives and the Coordinator logic now live
    together here."""

    # Basic name
    __ALL_UNAIVERSE = "all_unaiverse"
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
    IN_PUBLIC = __INBOUND + __ALL_UNAIVERSE + __PUBLIC_NET
    OUT_PUBLIC = __OUTBOUND + __ALL_UNAIVERSE + __PUBLIC_NET
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

    def __init__(self, max_connections: int,
                 is_world_node: bool, public_key: str, token: str,
                 local_node_id: str = None) -> None:
        """Initializes a new instance of the NodeConn class.

        The two transport handles are NOT taken here: the constructor builds only the pool
        taxonomy (keyed on the P2P NAMES, which are class constants) and the empty indices.
        The P2P objects arrive later, together with the Node policy, in attach_session_policy,
        which is where they are first used (to build the SessionManagers). A NodeConn before
        attach_session_policy is an inert shell with no transport.

        Args:
            max_connections: The total number of connections the node can handle.
            is_world_node: A boolean flag indicating if this node is a world node.
            public_key: The public key for token verification.
            token: The node's authentication token.
            local_node_id: this node's application node-id, used when building the
                per-instance SessionManagers in attach_session_policy.
        """
        self._init_pools(max_connections=max_connections,
                         valid_p2p_names={NodeConn.P2P_PUBLIC, NodeConn.P2P_WORLD},
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

        # The pure world membership (world_agents_set / world_masters_set /
        # world_node_peer_id) now lives in sm_world, next to the Sessions it gates (built
        # in attach_session_policy); the Coordinator reads it back through
        # membership_snapshot(). inspector_peer_id stays here: it is not pure world
        # membership but an orthogonal, currently-inert chat exception.
        self.inspector_peer_id = None
        # role_to_peer_ids and peer_id_to_addrs stay Coordinator-owned and do NOT descend
        # into sm_world with the world membership. role_to_peer_ids is the inverse
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

        # Pre-admissions still waiting for their world-side redemption: peer_id →
        # monotonic deadline. Written by the grant-time pre-admission
        # (add_to_world_*_list with provisional_ttl_s), cleared by the world-side
        # registration (same writers, no TTL) or by remove_from_world_lists; the
        # host sweeps the expired ones each cycle, so a grant that is never
        # redeemed does not leave a ghost member in the roster.
        self.world_preadmissions: dict[str, float] = {}

        # Application node-id, used when this Coordinator builds the SessionManagers.
        self._local_node_id = local_node_id

        # Per-instance SessionManagers. The Coordinator owns them: it builds both in
        # attach_session_policy (once the Node policy is available) and consults them
        # so the hosted entity (which holds conn, not the managers) can ask about
        # per-peer Session state (see is_peer_session_pending).
        self._sm_public = None
        self._sm_world = None
        self._session_managers = []

        # World-handoff correlation, owned here so it sits next to the two SessionManagers
        # it spans. The ledger itself (the twin maps + start/rollback/complete) lives in
        # HandoffCorrelation, shared with the test-harness mirror; the effect callables
        # close over self so they see the managers wired later by attach_session_policy.
        # Node drives the handoff through the start/rollback/complete facades below.
        self._handoff = HandoffCorrelation(
            is_world_established=lambda wpid: (self._sm_world is not None
                                               and self._sm_world.is_established(wpid)),
            abort_world=self._abort_world_redemption,
            current_world=self.get_world_node_peer_id,
            clear_world=lambda: self.set_world(None),
        )

    @property
    def _pending_world_handoff(self) -> dict:
        """The forward handoff index (public -> world); lives in the shared ledger."""
        return self._handoff.pending_world_handoff

    @property
    def _world_to_public(self) -> dict:
        """The reverse handoff index (world -> public); lives in the shared ledger."""
        return self._handoff.world_to_public

    async def _abort_world_redemption(self, world_peer_id: str) -> None:
        """Effect injected into the handoff ledger: abort the in-flight grant
        redemption (drop dispatches on its state: mid-handshake -> silent abort)."""
        if self._sm_world is not None:
            await self._sm_world.drop(world_peer_id)

    def _init_pools(self, max_connections: int, pool_name_to_p2p_name_and_ratio: dict[str, list[str | float]],
                    valid_p2p_names: set[str],
                    public_key: str | None = None, token: str | None = None) -> None:
        """Set up the pool triples and the per-peer / per-p2p indices for this Coordinator.

        Args:
            max_connections: The maximum total number of connections allowed across all pools.
            pool_name_to_p2p_name_and_ratio: A dictionary mapping pool names to a list containing the associated P2P
                network name and its connection ratio.
            valid_p2p_names: The set of known P2P network names, used only to validate that every
                pool references an existing network. The P2P objects themselves are not needed
                here, they arrive in attach_session_policy.
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

        # The SM is the instance: the per-instance index maps a p2p_name to its
        # SessionManager, not to the raw P2P handle. It is empty until
        # set_session_managers runs (the SMs are built late in attach_session_policy),
        # so it is seeded here and populated there. The pool->p2p_name mapping captured
        # at init lets set_session_managers wire each pool triple's instance slot to the
        # right SM. The Coordinator keeps no separate transport handle: each P2P lives inside
        # its owning SM, and the Coordinator reaches it (and the instance identity) through
        # the SM, never as a raw channel.
        self._p2p_name_to_sm = {}
        self._pool_name_to_p2p_name = {}

        # Indices rooted around the pool name
        self.pool_name_to_pool_triple = {}
        self.pool_name_to_added_in_last_update = {}
        self.pool_name_to_removed_in_last_update = {}
        self.pool_name_to_peer_infos = {p: {} for p in self.pool_names}

        # Indices rooted around the peer ID. peer_id_to_pool_name is the connection
        # index (which pool a connected peer lives in): it backs is_connected,
        # get_pool_of, in_connection_queues, get_all_connected_peer_ids and the chat
        # admission gate in get_messages. It is no longer the primary source of the
        # peer -> instance routing: that is Session-anchored now (_sm_of asks the
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
            assert p2p_name_and_ratio[0] in valid_p2p_names, f"Cannot find p2p named {p2p_name_and_ratio[0]} "
        assert self.max_con >= len(self.pool_names), "Too small number of max connections"
        assert sum([x for x in self.pool_ratios if x > 0]) == 1.0, "Pool ratios must sum to 1.0"

        # Preparing the pool triples. The instance slot (triple[2]) holds the
        # SessionManager that owns the pool, but the SMs do not exist yet at __init__
        # (they are built in attach_session_policy): seed it with None and remember which
        # p2p_name each pool belongs to, so set_session_managers can wire the right SM in.
        self.pool_name_to_pool_triple = {}
        for k in self.pool_names:
            p2p_name = pool_name_to_p2p_name_and_ratio[k][0]
            assert isinstance(p2p_name, str)
            self._pool_name_to_p2p_name[k] = p2p_name
            self.pool_name_to_pool_triple[k] = [set(), 0, None]
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

        # Storing the fixed cap in each pool triple (triple = [peer_id set, cap, SM]).
        for pool_name, pool_contents_max_con_and_sm in self.pool_name_to_pool_triple.items():
            pool_contents_max_con_and_sm[1] = pools_max_sizes[pool_name]  # Fixing the second element of the triple

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

    # Identity / posture facade. Node reads its own peer-ids, addresses and world posture
    # through the owning SessionManager (the manager is the instance, so its identity is the
    # instance's identity), never reaching into a raw P2P handle. These are the peer-ids /
    # addresses of THIS node on each network, distinct from the remote world host's peer-id
    # exposed by get_world_node_peer_id. peer_id may be None before the libp2p host is up:
    # the None guard stays on Node (get_public_peer_id / get_world_peer_id), so these return raw.
    @property
    def public_peer_id(self) -> str | None:
        return self._sm_public.local_peer_id

    @property
    def world_peer_id(self) -> str | None:
        return self._sm_world.local_peer_id

    @property
    def public_addresses(self) -> list[str]:
        return self._sm_public.addresses

    @property
    def world_addresses(self) -> list[str]:
        return self._sm_world.addresses

    @property
    def world_is_public(self) -> bool | None:
        return self._sm_world.is_public

    @property
    def world_relay_enabled(self) -> bool:
        return self._sm_world.relay_enabled

    def get_world_connected_peers_info(self) -> list | None:
        """Raw transport-level connected peers on the world instance, for the stats path
        (Agent.collect_and_store_own_stats). Reads through the world manager as internal
        custody and returns the data, not the handle, so callers stop reaching into the P2P.
        None on an FFI error (transport state unknown): callers must skip, not treat it as
        zero peers, or a transient failure poisons the stats with a false empty sample."""
        return self._sm_world.connected_peers_info()

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
        sm = self._sm_by_name(p2p_name)
        if sm is None:
            return None
        try:
            return await sm.reserve(peer_id)
        except P2PError:
            return None

    def set_turn_credentials(self, ice_turn_servers) -> None:
        """Push TURN credentials to BOTH instances, fanning out through the SMs
        (Node -> NodeConn -> SM -> P2P). The raw transport handles stay custody-only
        behind the managers, so Node sets credentials once and both instances update."""
        self._sm_public.set_turn_credentials(ice_turn_servers)
        self._sm_world.set_turn_credentials(ice_turn_servers)

    def add(self, peer_info: dict, pool_name: str) -> bool:
        """Adds a connected peer to a specified connection pool.

        Args:
            peer_info: A dictionary containing information about the peer.
            pool_name: The name of the pool to add the peer to.

        Returns:
            True if the peer is successfully added, otherwise False.
        """
        peer_id = peer_info['id']
        pool, max_size, _ = self.pool_name_to_pool_triple[pool_name]
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

    async def drop_peer(self, peer_id: str, reason: int | None = None,
                        settled: bool = False) -> list:
        """Voluntary per-peer teardown, the single teardown verb the hosted entity
        drives through (Node.__purge): ``sm.drop`` on the owning instance + the pool
        eviction. What happens to the Session and the transport is the manager's
        state dispatch (see SessionManager.drop): a live established Session with a
        reason is GOAWAY-closed only, and the transport falls with the deferred
        settlement cut once the CLOSED lands — never a disconnect in the same stack
        as the farewell write, which would destroy the GOAWAY on the wire; the
        coercive branches (mid-handshake, reasonless on a live Session, KNOWN,
        sessionless) abort/forget and sever the transport inline.

        Returns the SessionEvents for Node to dispatch (always ``[]`` today: the
        CLOSED arrives later via GOAWAY-ACK / the CLOSING timeout). The grace case
        (transport drop with a surviving RECONNECTING Session) does NOT come through
        here: it is handled by ``on_peer_disconnected``, which syncs the pool without
        closing the Session.

        The owning manager is resolved BEFORE the pool eviction (the pool index feeds
        ``_sm_of``'s fallback); a peer unknown to both registries is a no-op, which
        tolerates the frequent double-purge (a CLOSED dispatched after an earlier
        purge — also a no-op inside sm.drop, on the tearing-down branch).

        Args:
            peer_id: The peer ID to tear down.
            reason: Optional ReasonCode; when set and the Session is closable, it is
                GOAWAY-closed with it.
            settled: True for the RECKONING purge of a terminal SessionEvent (or of
                the polled transport drop): the SM settlement already handled the
                Session and scheduled/executed the transport cut, so only the pool
                bookkeeping is evicted here — sm.drop is skipped entirely. This is
                what keeps the reckoning from touching a SUCCESSOR Session born in
                the same pump batch (the settled relation's teardown must never
                leak onto the fresh one that replaced it).

        Returns:
            The list of SessionEvents to dispatch.
        """
        if not settled:
            sm = self._sm_of(peer_id)
            events = await sm.drop(peer_id, reason) if sm is not None else []
        else:
            events = []
        self._evict_from_pools(peer_id)
        return events

    def _evict_from_pools(self, peer_id: str) -> bool:
        """Internal: drop the peer from the pool indices only (the pool set, the per-pool
        peer infos, the pool-name index and the last-token cache), WITHOUT touching the
        transport or the Session. This is the pure pool-sync shared by the voluntary drop
        (``drop_peer``) and the involuntary transport-drop reaction (``on_peer_disconnected``).

        Remember to NOT del peer_id_to_misc[peer_id]! (the role taxonomy survives a drop)."""
        pool_name = self.peer_id_to_pool_name.get(peer_id)
        if pool_name is None:
            return False
        pool, _, _ = self.pool_name_to_pool_triple[pool_name]
        pool.discard(peer_id)
        self.pool_name_to_peer_infos[pool_name].pop(peer_id, None)
        del self.peer_id_to_pool_name[peer_id]
        self.peer_id_to_token.pop(peer_id, None)
        return True

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
          * `session_messages` are raw per-stage Session payloads already split off and
            decoded upstream by P2P.get_messages (the session-protocol partition lives in
            the transport tier); the Coordinator forwards them as-is, skipping the chat Msg
            decoding and JWT verification, because identity is established cryptographically
            inside the Session handshake (sig_b / sig_a / token).

        The whole wire codec (Msg.from_bytes and the sender-mismatch check) lives in
        P2P.get_messages now (the rewrite boundary toward OP2P.js); here the Coordinator
        receives already-decoded, sender-verified Msg and does only the Node-bound
        application work: membership gate plus piggyback JWT verification.

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

        # The instance is the SM; go through its passthrough (never call P2P directly).
        # P2P has already popped the inbound batch and run the whole codec: chat frames are
        # decoded, sender-verified Msg, session frames are SessionInbound passed straight
        # through. Here we only gate membership and verify the piggyback token.
        sm = self._sm_by_name(p2p_name)
        chat_messages, session_messages = sm.get_messages()
        processed_messages: list[Msg] = []

        if chat_messages or session_messages:
            self.__last_recv_time = time.time()

        for msg_obj in chat_messages:
            log.network(f"<<< RECEIVED {msg_obj.content_type} through channel {msg_obj.channel}", sub=sm.log_sub)

            # filter only valid messages to return: a sender already in a pool, or one
            # of this cycle's admitted world agents/masters (the frozen snapshot above).
            if (msg_obj.sender in self.peer_id_to_pool_name or
                    msg_obj.sender in allowed_not_connected_peers):

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
                                  "(discarding it)", sub=sm.log_sub)
                except Exception as e:
                    log.error(f"Received a message with an invalid piggyback token! (discarding it) [{e}]",
                              sub=sm.log_sub)

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
                   content_type: str, content: bytes | dict | str | None = None) -> bool:
        """Sends a direct message to a specific peer (async).

        Args:
            peer_id: The peer ID to send the message to.
            channel_trail: An optional string to append to the channel name.
            content_type: The type of content in the message.
            content: The message content.

        Returns:
            True if the message is sent successfully, otherwise False.
        """
        if content_type not in [Msg.STATS_UPDATE, Msg.STATS_REQUEST, Msg.STATS_RESPONSE]:
            self.__last_sent_time = time.time()

        # Resolve the owning instance's manager and send through it (NodeConn -> SM -> P2P).
        sm = self._sm_of(peer_id)
        if sm is None:
            log.error("P2P non found for peer id: " + str(peer_id))
            return False

        # Defining channel (the instance's local peer-id, read from the SM, not the P2P handle)
        if channel_trail is not None and len(channel_trail) > 0:
            channel = f"{sm.local_peer_id}::dm:{peer_id}-{content_type}~{channel_trail}"
        else:
            channel = f"{sm.local_peer_id}::dm:{peer_id}-{content_type}"

        log.network(f">>> SENDING {content_type} to {peer_id}", sub=sm.log_sub)

        # Adding sender info here
        msg = Msg(sender=sm.local_peer_id,
                  content_type=content_type,
                  content=content,
                  channel=channel,
                  piggyback=self.__token + "0")  # Adding inspector-mode bit (dummy bit here)

        # Hand the Msg object to the SM; the codec and the thread offload live in P2P.
        try:
            await sm.send(channel, msg)
            return True
        except P2PError as e:
            log.error("Sending error is: " + str(e), sub=sm.log_sub)
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

        # Resolve the owning instance's manager and subscribe through it.
        sm = self._sm_for_topic(peer_id, default_p2p_name)
        if sm is None:
            return False
        try:
            sm.subscribe_to_topic(channel)
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

        if default_p2p_name is None:
            default_p2p_name = NodeConn.P2P_WORLD

        # Resolve the owning instance's manager and unsubscribe through it.
        sm = self._sm_for_topic(peer_id, default_p2p_name)
        if sm is None:
            return False
        try:
            sm.unsubscribe_from_topic(channel)
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

        # Resolve the owning instance's manager (our own topic or the peer's instance).
        sm = self._sm_for_topic(peer_id)
        if sm is None:
            return False

        log.network(f">>> PUBLISHING {content_type} to {channel}", sub=sm.log_sub)

        # Adding sender info here
        msg = Msg(sender=sm.local_peer_id,
                  content_type=content_type,
                  content=content,
                  channel=channel,
                  piggyback=self.__token + "0")  # Adding inspector-mode bit (dummy bit here)

        # Hand the Msg object to the SM; the codec and the thread offload live in P2P.
        try:
            await sm.broadcast(channel, msg)

            # If the line above executes without raising an error, it was successful.
            return True
        except P2PError:

            # If broadcast_message fails, it will raise a P2PError. We catch it here.
            return False

    def attach_session_policy(self, p2p_u: P2P, p2p_w: P2P,
                              local_profile_factory, decision_hook, token_factory=None) -> None:
        """Take the two transport handles and build the two per-instance SessionManagers,
        now that both the transports and the Node policy born late are available (the profile
        factory samples self.profile, the decision hook reads world/agent/allow-list state).
        This is the single point where the Coordinator becomes operational: each manager is
        built around its P2P and owns it. The Coordinator keeps no separate transport handle,
        it reaches each instance (and its identity) through the owning manager."""
        # Per-instance perimeter routing injected into each manager: the public one
        # routes by direction (no membership, never cuts); the world one classifies
        # against the snapshot and cuts unclassifiable peers on its P2P instance.
        self._sm_public = SessionManager(
            p2p=p2p_u,
            p2p_name=NodeConn.P2P_PUBLIC,
            local_peer_id=p2p_u.peer_id,
            local_node_id=self._local_node_id,
            local_profile_factory=local_profile_factory,
            decision_hook=decision_hook,
            token_verifier=self.verify_token_claims,
            token_factory=token_factory,
            routing_classifier=NodeConn._route_public,
        )
        self._sm_world = SessionManager(
            p2p=p2p_w,
            p2p_name=NodeConn.P2P_WORLD,
            local_peer_id=p2p_w.peer_id,
            local_node_id=self._local_node_id,
            local_profile_factory=local_profile_factory,
            decision_hook=decision_hook,
            token_verifier=self.verify_token_claims,
            token_factory=token_factory,
            routing_classifier=NodeConn._route_world,
        )
        self.set_session_managers(self._sm_public, self._sm_world)

    def set_session_managers(self, *session_managers) -> None:
        """Register the per-instance SessionManagers so conn can answer per-peer
        Session-state queries on behalf of the hosted entity. Called by
        attach_session_policy (and directly by unit tests with stub managers).

        This is also where the SM (the instance) is wired into the per-instance index and
        into each pool triple's instance slot, now that the managers exist: __init__ seeded
        the triples with None and recorded each pool's p2p_name. Unit tests that build a
        bare NodeConn via __new__ (no _init_pools) have no triples to wire, hence the
        guard."""
        self._session_managers = [sm for sm in session_managers if sm is not None]
        self._p2p_name_to_sm = {sm.p2p_name: sm for sm in self._session_managers
                               if getattr(sm, "p2p_name", None) is not None}
        triples = getattr(self, "pool_name_to_pool_triple", None)
        if triples:
            for pool_name, triple in triples.items():
                triple[2] = self._p2p_name_to_sm.get(self._pool_name_to_p2p_name[pool_name])

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
        (FAILED, CLOSED).

        The per-state classification lives in SessionManager.is_handshake_in_flight
        (next to SessionState); the Coordinator only aggregates across its instances, so
        connpool no longer imports the session enum."""
        return any(sm.is_handshake_in_flight(peer_id) for sm in self._session_managers)

    def requires_fast_clock(self) -> bool:
        """True if any registered SessionManager has a Session that needs the main loop
        at full speed (see SessionManager.requires_fast_clock). The Coordinator aggregates
        across its instances so the Node's clock-pacing decision is one state-based query
        instead of reaching into each manager. This supersedes the old idle check that
        counted only ESTABLISHED/RECONNECTING and so let a slowed clock starve handshakes
        in flight."""
        return any(sm.requires_fast_clock() for sm in self._session_managers)

    async def pump_sessions(self, inbound_by_p2p_name: dict, now: float) -> list:
        """Per-cycle Session pump, owned by the Coordinator: deliver every inbound
        Session message to its instance's manager, then tick every manager, returning a
        FLAT list of SessionEvent in generation order (all delivers, public before world,
        then all ticks, same order). Node keeps the single dispatch site
        (_dispatch_session_event), so this only moves the per-instance iteration off Node.

        Mirrors the legacy order exactly: every deliver fires before any tick, and the
        managers are swept in registration order (public then world). The SessionInbound
        items are keyed by p2p_name; a manager with no inbound this cycle still ticks."""
        events: list = []
        # Deliver pass (public then world), every inbound of each instance in order.
        for sm in self._session_managers:
            for inb in inbound_by_p2p_name.get(sm.p2p_name, []):
                events.extend(await sm.deliver(inb.peer_id, inb.protocol_id, inb.sess_msg))
        # Tick pass (same order): timeouts + keep-alive PING/PONG.
        for sm in self._session_managers:
            events.extend(await sm.tick(now))
        return events

    async def on_peer_disconnected(self, peer_id: str) -> list:
        """React to a transport disconnect: fan it to every instance's manager
        (ESTABLISHED -> RECONNECTING grace, in-handshake states -> FAILED) and sync the
        pool bookkeeping (``_evict_from_pools``), returning the flat list of resulting
        SessionEvent (public before world) for Node to dispatch.

        The link is already gone, so this evicts the pool WITHOUT a transport disconnect
        (unlike the voluntary ``drop_peer``). This is the involuntary pool-sync: it runs in
        BOTH the grace case (the RECONNECTING Session survives, the pool self-heals on the
        next update() if the peer RESUMEs) and the terminal case, so Node never issues a
        bare pool-remove itself."""
        events: list = []
        for sm in self._session_managers:
            events.extend(await sm.on_peer_disconnected(peer_id))
        self._evict_from_pools(peer_id)
        return events

    async def on_peer_reconnected(self, peer_id: str) -> list:
        """Fan a transport reconnect to every instance's manager and return the flat list
        of resulting SessionEvent (public before world). A no-op unless a Session for the
        peer is RECONNECTING (mid-grace), so feeding fresh first-time connections is safe."""
        events: list = []
        for sm in self._session_managers:
            events.extend(await sm.on_peer_reconnected(peer_id))
        return events

    def is_in_grace(self, peer_id: str) -> bool:
        """True if any instance holds a Session for this peer riding its reconnect grace
        (RECONNECTING). The Node uses this to keep the enrolment alive across a transport
        blip (pool-remove) instead of a full purge, without reaching into Session state."""
        return any(sm.is_in_grace(peer_id) for sm in self._session_managers)

    # Session lifecycle
    #
    # The Session lifecycle verbs, kept separate from the pump and the world-handoff:
    # open an initiator Session, gracefully close one, close them all on shutdown.
    # Node drives these through the Coordinator rather than on the SessionManagers
    # directly. Events come back as the return
    # value for Node's single dispatch site; the lifecycle POLICY (when to connect / leave /
    # purge / close) stays on Node, only the SM resolution and the call move here.

    async def initiate_session(self, p2p_name: str, peer_id: str, addresses: list,
                               is_world_join: bool = False, remote_node_id_hint: str | None = None,
                               reject_relay: bool = False) -> list:
        """Open an initiator Session to `peer_id` on the named instance and return the
        start-up events for Node to dispatch (and to read a terminal FAILED from). The
        grant-carrying world initiate stays on the handoff path (start_world_handoff).

        A world-join dial to a peer that already holds an ESTABLISHED world-join carrier
        with NO pending handoff is a SPENT carrier (a previous join's handoff failed and was
        rolled back): the plain intent match in initiate() would no-op and the join would
        never restart. The pending-handoff correlation lives here on the Coordinator, so we
        detect it here and tell the SM to replace the stale carrier. An in-flight handoff
        stays in _pending_world_handoff, so a live join is protected (no-op), not torn down."""
        sm = self._sm_by_name(p2p_name)
        replace_stale = (is_world_join and peer_id not in self._pending_world_handoff
                         and sm.has_established_world_join_carrier(peer_id))
        return await sm.initiate(
            peer_id=peer_id, addresses=addresses, is_world_join=is_world_join,
            remote_node_id_hint=remote_node_id_hint, reject_relay=reject_relay,
            replace_stale=replace_stale)

    async def close_all_sessions(self) -> None:
        """Shutdown path: GOAWAY every Session on every instance so peers see a clean CLOSED
        instead of a liveness timeout. Events are intentionally dropped (the node is going
        down); close_all skips Sessions already CLOSING/terminal and silently aborts the
        in-flight handshakes (no farewell owed there, and the drain must be able to end)."""
        for sm in self._session_managers:
            await sm.close_all()

    async def drain_teardown(self, budget_s: float = 3.0) -> None:
        """Shutdown drain: keep pumping ONLY the session traffic until every farewell
        settled — no Session left CLOSING (its GOAWAY_ACK arrived, or the timeout
        finalized it) and no settlement cut still pending (the transports of ended
        Sessions actually severed) — or the budget expires. Events are discarded: the
        node is going down, there is no application state left worth dispatching to.
        World instance first: a host's hosted GOAWAYs ride the world instance.

        This is what makes the shutdown GOAWAYs real on a network with latency: the
        send returning only means 'queued in the local muxer', and exiting (or closing
        the Go hosts) right after would destroy the queued frames. The GOAWAY_ACK
        round-trip is the end-to-end proof of delivery; the budget bounds the wait
        when a peer is gone and the 10s CLOSING timeout exceeds the shutdown patience.
        """
        deadline = time.monotonic() + budget_s
        managers = [sm for sm in reversed(self._session_managers)]
        while time.monotonic() < deadline:
            for sm in managers:
                try:
                    _, session_inbounds = sm.get_messages()
                    for inb in session_inbounds:
                        await sm.deliver(inb.peer_id, inb.protocol_id, inb.sess_msg)
                    await sm.tick()
                except Exception as e:
                    log.error(f"drain_teardown: pump on {sm.p2p_name} failed: {e}")
            if not any(sm.has_pending_teardown() for sm in managers):
                return
            await asyncio.sleep(0.05)

    def close_transports(self) -> None:
        """Shutdown: close the Go host of every instance (world first), freeing its
        slot — until now nothing ever called P2P.close, so embedding several Nodes in
        one process leaked slots. Only the shutdown path calls this, after
        drain_teardown has let the farewells settle."""
        for sm in reversed(self._session_managers):
            try:
                sm.close_transport()
            except Exception as e:
                log.error(f"close_transports: closing {sm.p2p_name} failed: {e}")

    # World-handoff correlation
    #
    # The join handshake spans two Sessions on two instances: the public one yields a
    # WorldGrant, the world one redeems it. These facades own the public<->world peer-id
    # correlation so Node drives the handoff without holding the maps. The world-profile
    # parse stays on Node (profile domain); only the correlation + transport open/close
    # live here.

    async def start_world_handoff(self, public_peer_id: str, world_peer_id: str,
                                  world_addrs: list, grant_blob: dict, reject_relay: bool) -> list:
        """Joiner side: record the public<->world correlation, point pool routing at the
        world host (set_world), and open the world Session to redeem the grant. Returns
        the initiate events for Node to dispatch. The redeemer's dial is a single
        attempt bounded by the grant window (see _WORLD_DIAL_*_BUDGET_S): time.time()
        matches the wall-clock the host minted expiry_unix_ms with."""
        try:
            self.set_world(world_peer_id)
        except Exception:
            pass
        self._handoff.start(public_peer_id, world_peer_id)
        remaining_s = int(grant_blob.get("expiry_unix_ms", 0)) / 1000.0 - time.time()
        budget_s = max(_WORLD_DIAL_MIN_BUDGET_S, min(remaining_s, _WORLD_DIAL_MAX_BUDGET_S))
        return await self._sm_world.initiate(
            world_peer_id, world_addrs, grant=grant_blob, reject_relay=reject_relay,
            dial_attempts=1, dial_budget_s=budget_s)

    async def rollback_handoff(self, peer_id: str) -> None:
        """Roll back any in-flight handoff correlation for a terminal Session,
        whichever side died. The mechanics (established-skip, abort of the in-flight
        redemption, routing un-point) live in HandoffCorrelation.rollback, shared
        with the test harness."""
        await self._handoff.rollback(peer_id)

    def is_pending_handoff(self, world_peer_id: str) -> bool:
        """True if this world peer-id is the world side of an in-flight handoff (its world
        Session was opened by start_world_handoff), as opposed to a plain in-world
        agent-to-agent connection that carries no grant correlation."""
        return self._handoff.is_pending(world_peer_id)

    async def complete_world_handoff(self, world_peer_id: str) -> None:
        """The world Session established: retire the correlated public carrier — GOAWAY
        it (WORLD_HANDOFF_COMPLETE) on its still-live transport, evict it from the pools,
        and drop the correlation. No-op if this world peer-id was not part of a handoff.
        The transport itself falls with the settlement cut once the CLOSED lands (ACK or
        the CLOSING timeout); the dispatch early-returns on WORLD_HANDOFF_COMPLETE, so
        the pool eviction must happen here, not in a purge."""
        public_peer_id = self._handoff.complete(world_peer_id)
        if public_peer_id is None:
            return
        await self._sm_public.close_session(public_peer_id, WORLD_HANDOFF_COMPLETE)
        self._evict_from_pools(public_peer_id)

    def _sm_of(self, peer_id: str) -> "SessionManager | None":
        """Resolve the SessionManager that owns the instance this peer lives on (the SM
        IS the instance): the manager holding a Session for the peer-id first (public
        before world, matching the cross-pool first-network-wins), falling back to the
        manager whose P2P backs the peer's pool for a peer connected without a Session yet
        (an in-flight or just-evicted handshake), and None when nothing knows the peer.
        This is the SM-keyed resolver the transport facades (drop/send/...) delegate
        to so the chain stays Node -> NodeConn -> SM -> P2P."""
        for sm in (self._sm_public, self._sm_world):
            if sm is not None and sm.get_session(peer_id) is not None:
                return sm
        pool_name = self.peer_id_to_pool_name.get(peer_id)
        if pool_name is None:
            return None
        # The pool triple's instance slot already holds the owning SM (None pre-attach).
        return self.pool_name_to_pool_triple[pool_name][2]

    def _sm_by_name(self, p2p_name: str) -> "SessionManager | None":
        """The SessionManager (the instance) for a named network, via the per-instance
        index populated at attach time. None before the managers are registered."""
        return self._p2p_name_to_sm.get(p2p_name)

    def _sm_for_topic(self, peer_id: str, default_p2p_name: "str | None" = None) -> "SessionManager | None":
        """The SessionManager to use for a topic/dm keyed by peer_id: the instance whose
        OWN peer-id is peer_id (our own topics), else the peer's owning instance (_sm_of),
        else the named default. None if nothing resolves."""
        for sm in (self._sm_public, self._sm_world):
            if sm is not None and sm.p2p.peer_id == peer_id:
                return sm
        sm = self._sm_of(peer_id)
        if sm is not None:
            return sm
        return self._sm_by_name(default_p2p_name) if default_p2p_name is not None else None

    def reset_rendezvous_tag(self) -> None:
        """Resets the rendezvous tag to its initial state."""
        self.rendezvous_tag = -1

    def membership_snapshot(self) -> WorldMembership:
        """Freeze the current world membership into an immutable per-cycle view. The pure
        membership (the two sets + world_node_peer_id) is read from sm_world, where it now
        lives next to the Sessions it gates; inspector_peer_id is the Coordinator's own
        orthogonal exception. The sets are copied into frozensets so a snapshot
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
        each SessionManager in registration order (public then world), merging the
        per-instance diffs. The pool_name keys are disjoint across the two instances (IN/OUT_PUBLIC vs
        IN/OUT_WORLD_*), so the merge is a plain dict.update, asserted to be sure."""
        self._sync_known_world_peers()

        self.pool_name_to_added_in_last_update = {}
        self.pool_name_to_removed_in_last_update = {}
        membership = self.membership_snapshot()
        self._cycle_membership = membership  # publish this cycle's snapshot for get_messages' gate

        # The SM is the instance: iterate the managers (public then world) instead of the
        # raw P2P handles. _session_managers preserves the public-before-world order the
        # cross-pool guard relies on (a peer-id enters at most one pool, public winning).
        for sm in self._session_managers:
            added_local, removed_local = await self._reconcile_instance(sm, membership)
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
        self_peer_id = self._sm_world.local_peer_id
        known = {peer_id: (addrs, self.peer_id_to_misc.get(peer_id, 0))
                 for peer_id, addrs in self.peer_id_to_addrs.items()
                 if peer_id != self_peer_id}
        self._sm_world.sync_known_peers(known)

    async def _reconcile_instance(self, sm, membership) -> tuple[dict, dict]:
        """Reconcile ONE instance (its SessionManager) and report its per-pool
        added/removed diff.

        Asks the manager to classify its connected peers (sm.reconcile, which also
        applies the world perimeter cut), then for each of this instance's pools computes
        the new/lost set-difference and admits the new peers via add() — which keeps the
        cross-pool guard and the shared peer_id_to_* indices here in the Coordinator.
        Returns two dicts keyed by pool_name with only the pools that changed (empty pools
        omitted), like the legacy update().
        """
        added_local: dict = {}
        removed_local: dict = {}

        connected_peer_infos = sm.connected_peers_info()
        if connected_peer_infos is None:
            return added_local, removed_local

        # The world manager classifies against the snapshot (and cuts the perimeter);
        # the public manager has no membership.
        sm_membership = membership if sm.p2p_name == NodeConn.P2P_WORLD else None
        classified = await sm.reconcile(connected_peer_infos, sm_membership)

        # This instance's own pools are the ones whose triple is bound to this manager
        # (the SM is the instance). Treat a pool absent from `classified` as empty so
        # removals are still detected.
        for pool_name, (pool, _, pool_sm) in self.pool_name_to_pool_triple.items():
            if pool_sm is not sm:
                continue
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

    # World-roster writers: every world-membership mutation lands here
    #
    # set_world / set_world_agents_list / set_world_masters_list / add_to_world_agents_list /
    # add_to_world_masters_list / remove_from_world_lists are the ONLY places the world roster
    # is mutated. node.py never touches world_agents_set/world_masters_set/world_node_peer_id
    # directly: it calls these Coordinator methods, which write sm_world. That is what keeps the
    # membership single-sourced in sm_world. The rendezvous READ stays on the Coordinator
    # (set_world_agents_and_world_masters_lists_from_rendezvous); only the write lands here.
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
        self.world_preadmissions.pop(peer_id, None)
        self.peer_id_to_addrs.pop(peer_id, None)
        misc = self.peer_id_to_misc.pop(peer_id, None)
        if misc in self.role_to_peer_ids:
            self.role_to_peer_ids[misc].discard(peer_id)
            if not self.role_to_peer_ids[misc]:
                del self.role_to_peer_ids[misc]

    def sweep_world_preadmissions(self, now: float | None = None) -> list[str]:
        """World host: drop the pre-admitted joiners whose deadline expired without
        a world-side registration (the grant was never redeemed). The public
        carrier's own FAILED cannot clean these up — it carries the PUBLIC peer-id
        while the pre-admission is keyed on the private one — so the deadline is
        the collector. Returns the removed peer-ids (for the caller's log)."""
        if not self.world_preadmissions:
            return []
        if now is None:
            now = time.monotonic()
        expired = [pid for pid, deadline in self.world_preadmissions.items()
                   if deadline <= now]
        for pid in expired:
            self.remove_from_world_lists(pid)   # pops the deadline too
        return expired

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

    def add_to_world_agents_list(self, peer_id: str, addrs: list[str], role: int = -1,
                                 provisional_ttl_s: float | None = None) -> None:
        """Adds a new world agent to the list.

        Args:
            peer_id: The peer ID of the new agent.
            addrs: A list of addresses for the new agent.
            role: The role assigned to the agent.
            provisional_ttl_s: When set, the entry is a grant-time pre-admission
                that expires after this many seconds unless confirmed; calling
                again WITHOUT it (the world-side registration) confirms the entry.
        """
        self._sm_world.world_agents_set.add(peer_id)

        # This assumes that the WORLD MASTER/AGENT BIT is the first one
        assert role & 1 == 1, "Expecting the first bit of the role to be 1 for world agents"
        assert role & 2 == 0, "Expecting the second bit of the role to be 0 for world agents"
        self.peer_id_to_addrs[peer_id] = addrs
        self.set_role(peer_id, role)
        if provisional_ttl_s is not None:
            self.world_preadmissions[peer_id] = time.monotonic() + provisional_ttl_s
        else:
            self.world_preadmissions.pop(peer_id, None)

    def add_to_world_masters_list(self, peer_id: str, addrs: list[str], role: int = -1,
                                  provisional_ttl_s: float | None = None) -> None:
        """Adds a new world master to the list.

        Args:
            peer_id: The peer ID of the new master.
            addrs: A list of addresses for the new master.
            role: The role assigned to the master.
            provisional_ttl_s: When set, the entry is a grant-time pre-admission
                that expires after this many seconds unless confirmed; calling
                again WITHOUT it (the world-side registration) confirms the entry.
        """
        self._sm_world.world_masters_set.add(peer_id)

        # This assumes that the WORLD MASTER/AGENT BIT is the first one
        assert role & 1 == 1, "Expecting the first bit of the role to be 1 for world masters"
        assert role & 2 == 2, "Expecting the second bit of the role to be 1 for world masters"
        self.peer_id_to_addrs[peer_id] = addrs
        self.set_role(peer_id, role)
        if provisional_ttl_s is not None:
            self.world_preadmissions[peer_id] = time.monotonic() + provisional_ttl_s
        else:
            self.world_preadmissions.pop(peer_id, None)

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
            if discard_yourself and (peer_id == self._sm_public.local_peer_id
                                     or peer_id == self._sm_world.local_peer_id):
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
        rendezvous_state = self._sm_world.rendezvous_peers()

        if rendezvous_state is not None:
            tag = rendezvous_state.get('update_count', -1)

            if tag > self.rendezvous_tag:
                self.rendezvous_tag = tag
                rendezvous_peer_infos = rendezvous_state.get('peers', [])

                world_agents_peer_infos = []
                world_masters_peer_infos = []

                log.cpool(f"Rendezvous peer infos (tag: {tag}, peers: "
                          f"{len(rendezvous_peer_infos)} peers)", sub=self._sm_world.log_sub)
                log.cpool(f"Rendezvous message included peer IDs: "
                          f"{[p['id'] for p in rendezvous_peer_infos]})", sub="prv")

                for c in rendezvous_peer_infos:
                    if c['addrs'] is None:
                        log.cpool(f"Skipping a peer with None addrs (unexpected)", sub=self._sm_world.log_sub)
                        continue
                    # if len(c['addrs']) == 0:
                    #    log.cpool(f"Skipping a peer with zero-length addrs-list (unexpected)")
                    #    continue
                    if (c['misc'] & 1) == 1 and (c['misc'] & 2) == 0:
                        world_agents_peer_infos.append(c)
                    elif (c['misc'] & 1) == 1 and (c['misc'] & 2) == 2:
                        world_masters_peer_infos.append(c)
                    else:
                        # An unexpected 'misc' from the rendezvous must not be fatal (this runs
                        # every cycle from the main loop): log and skip the entry.
                        log.error("Unexpected value of the 'misc' field, skipping: " + str(c),
                                  sub=self._sm_world.log_sub)

                # Updating lists
                self.set_world_agents_list(world_agents_peer_infos)
                self.set_world_masters_list(world_masters_peer_infos)

    async def remove_all_world_agents(self) -> None:
        """GOAWAY-close and drop every connected world peer (the WORLD pools: agents,
        masters and node alike) from the pools and role lists (async). The reason
        WORLD_DISENGAGEMENT closes each ex-companion's Session cleanly — the GOAWAY
        actually reaches them now that the transport falls with the deferred
        settlement cut instead of being reset under the write — so they observe
        CLOSED (no reconnect) instead of a drop that would leave them ESTABLISHED
        to re-dial and re-enrol peers of a world we just left."""
        peer_infos = self.get_all_connected_peer_infos(NodeConn.WORLD)
        for c in peer_infos:
            peer_id = c['id']
            await self.drop_peer(peer_id, WORLD_DISENGAGEMENT)
            if peer_id in self.peer_id_to_addrs:
                del self.peer_id_to_addrs[peer_id]
            for role, peer_ids in self.role_to_peer_ids.items():
                if peer_id not in peer_ids:
                    continue
                if role & 1 == 1:
                    peer_ids.remove(peer_id)
