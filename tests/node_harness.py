"""Harness to exercise Node's Session integration without booting a real Node.

``Node.__init__`` is untestable in CI (NTP sync with interactive fallback,
root-server HTTPS calls, real Go P2P instances), so ``build_agent_node_shell``
builds a *shell*: ``Node.__new__(Node)`` plus the closed set of attributes that
the three integration entry points actually touch —

  * ``node._Node__handle_network_messages()`` (session pump + SessionManager tick)
  * ``node._Node__handle_network_connections()`` (pool diff → transport events)
  * ``node._dispatch_session_event(...)`` / ``node._session_decision_hook(...)``

The SessionManagers are REAL (so is the Session FSM); only the surfaces below
them (conn / hosted / profile / world) are fakes. A remote peer is represented
by a plain SessionManager on a FakeConn wired into the shell's inbox, exactly
like the e2e handshake tests.
"""

from collections import deque

from conftest import FakeP2P, FakeConn

from unaiverse.networking.node.connpool import NodeConn, SessionInbound
from unaiverse.networking.session import SessionManager


def full_profile_dict(node_id: str, **dynamic_overrides) -> dict:
    """A remote-profile dict shaped exactly like ``NodeProfile.get_all_profile()``
    output: no top-level ``node_id`` (the Session FSM must dig it out of
    ``static``), and the full static key set ``NodeProfile.from_dict`` requires."""
    static = {
        "node_id": node_id,
        "node_type": "agent",
        "node_name": f"{node_id}-name",
        "node_description": "",
        "created_utc": "2026-01-01T00:00:00+00:00",
        "name": "Test",
        "surname": "Peer",
        "title": "",
        "organization": "",
        "email": f"{node_id}@example.org",
        "max_nr_connections": 8,
    }
    dynamic = {
        "peer_id": f"{node_id}-peer",
        "peer_addresses": [],
        "private_peer_id": f"{node_id}-world-peer",
        "private_peer_addresses": [f"/ip4/10.0.0.9/tcp/4001/p2p/{node_id}-world-peer"],
    }
    dynamic.update(dynamic_overrides)
    return {"static": static, "dynamic": dynamic, "cv": []}


class FakeHosted:
    """Stands in for the hosted Agent: records add/remove calls."""

    HUMAN = "human"

    def __init__(self, add_agent_returns: bool = True):
        self.add_agent_returns = add_agent_returns
        self.added = []           # (peer_id, profile)
        self.removed = []         # peer_id
        self.all_agents = {}

    async def add_agent(self, peer_id, profile):
        self.added.append((peer_id, profile))
        if self.add_agent_returns:
            self.all_agents[peer_id] = profile
        return self.add_agent_returns

    async def remove_agent(self, peer_id):
        self.removed.append(peer_id)
        self.all_agents.pop(peer_id, None)
        return True


class FakeNodeProfile:
    """Just enough of NodeProfile for the connection-summary updates and the
    grant flow (which binds signatures to ``dynamic.private_peer_id``)."""

    def __init__(self, node_id: str, world_peer_id: str | None = None):
        self._node_id = node_id
        self._world_peer_id = world_peer_id
        self._dynamic = {
            "private_peer_id": world_peer_id,
            "connections": {"public_agents": None, "world_agents": None,
                            "world_masters": None, "world_peer_id": None, "role": None},
            "world_summary": {},
        }
        self.connection_changes = 0

    def get_static_profile(self):
        return {"node_id": self._node_id, "inspector_node_id": None}

    def get_dynamic_profile(self):
        return self._dynamic

    def get_all_profile(self):
        return {"static": {"node_id": self._node_id},
                "dynamic": {"private_peer_id": self._world_peer_id}, "cv": []}

    def mark_change_in_connections(self):
        self.connection_changes += 1


class FakeWorld:
    """The slice of World the world-host decision hook touches."""

    class _Stats:
        def plot(self, *a, **kw):
            return {}

    ROLE_STR_TO_BITS = {"player": 1 << 2}

    def __init__(self):
        self.stats = self._Stats()
        self.packed_agent_files = "packed-files"
        self.agent_stats_code = ""
        self.assign_role_returns = "player"
        self.assigned = []
        self.added = []
        self.add_agent_returns = True

    def assign_role(self, profile, is_world_master):
        self.assigned.append((profile, is_world_master))
        return self.assign_role_returns

    async def add_agent(self, peer_id, profile):
        self.added.append((peer_id, profile))
        return self.add_agent_returns


class NodeFakeConn(FakeConn):
    """FakeConn extended with the surface Node's loop methods consume."""

    P2P_PUBLIC = NodeConn.P2P_PUBLIC
    P2P_WORLD = NodeConn.P2P_WORLD
    PUBLIC = NodeConn.PUBLIC
    WORLD = NodeConn.WORLD
    WORLD_NODE = NodeConn.WORLD_NODE
    WORLD_AGENTS = NodeConn.WORLD_AGENTS
    WORLD_MASTERS = NodeConn.WORLD_MASTERS
    IN_PUBLIC = NodeConn.IN_PUBLIC
    OUT_PUBLIC = NodeConn.OUT_PUBLIC

    def __init__(self, p2p_public: FakeP2P, p2p_world: FakeP2P):
        super().__init__(p2p_public=p2p_public, p2p_world=p2p_world)
        # Inbound session messages per p2p instance, as (sender, protocol, SessMsg)
        # tuples — the same shape the bidirectional FakeConn wiring produces.
        self.inbox_by_p2p = {self.P2P_PUBLIC: deque(), self.P2P_WORLD: deque()}
        # Next update() diff, preset by the test: {pool_name: {peer_id, ...}}.
        self.next_added: dict = {}
        self.next_removed: dict = {}
        self.peer_id_to_pool = {}
        # Call recorders.
        self.removed_calls = []
        self.set_world_calls = []
        self.world_agents_set = set()
        self.world_masters_set = set()
        self.world_roles = {}
        self.addresses_set = []

    async def get_messages(self, p2p_name: str, allowed_not_connected_peers=None):
        inbox = self.inbox_by_p2p[p2p_name]
        session_inbound = []
        while inbox:
            peer_id, protocol_id, sess_msg = inbox.popleft()
            session_inbound.append(SessionInbound(peer_id, protocol_id, sess_msg))
        return [], session_inbound

    async def update(self):
        added, removed = self.next_added, self.next_removed
        self.next_added, self.next_removed = {}, {}
        return added, removed

    async def remove(self, peer_id: str) -> bool:
        self.removed_calls.append(peer_id)
        self.peer_id_to_pool.pop(peer_id, None)
        return True

    async def set_world_agents_and_world_masters_lists_from_rendezvous(self):
        return None

    def get_pool_of(self, peer_id: str):
        return self.peer_id_to_pool.get(peer_id)

    def get_all_connected_peer_infos(self, pool_name):
        return []

    def is_public(self, peer_id: str) -> bool:
        return False

    def set_world(self, world_peer_id):
        self.set_world_calls.append(world_peer_id)

    def add_to_world_agents_list(self, peer_id, addrs, role=-1):
        self.world_agents_set.add(peer_id)
        self.world_roles[peer_id] = role

    def add_to_world_masters_list(self, peer_id, addrs, role=-1):
        self.world_masters_set.add(peer_id)
        self.world_roles[peer_id] = role

    def remove_from_world_lists(self, peer_id):
        self.world_agents_set.discard(peer_id)
        self.world_masters_set.discard(peer_id)
        self.world_roles.pop(peer_id, None)

    def set_addresses_in_peer_info(self, peer_id, addresses):
        self.addresses_set.append((peer_id, addresses))


def build_agent_node_shell(node_peer_id: str, node_id: str = "shell-node",
                           hosted: FakeHosted | None = None,
                           world: FakeWorld | None = None):
    """Build a Node shell wired to real SessionManagers over a NodeFakeConn.

    Returns (node, conn). When `world` is given the shell behaves as a world
    host (node_type WORLD) for decision-hook tests; otherwise it is an AGENT.
    """
    from unaiverse.networking.node.node import Node

    node = Node.__new__(Node)
    conn = NodeFakeConn(p2p_public=FakeP2P(peer_id=node_peer_id, log_sub="shell-pub"),
                        p2p_world=FakeP2P(peer_id=node_peer_id + "-w", log_sub="shell-prv"))

    node.node_id = node_id
    node.node_type = Node.WORLD if world is not None else Node.AGENT
    node.hosted = hosted if hosted is not None else FakeHosted()
    node.agent = node.hosted if world is None else None
    node.world = world
    node.conn = conn
    node.profile = FakeNodeProfile(node_id, world_peer_id=node_peer_id + "-w")
    node.allowed_node_ids = None
    node.world_masters_node_ids = None
    node.only_certified_agents = False
    node.talk_to_relay_based_nodes = True
    node.inspector_peer_id = None
    node.inspector_activated = False
    node.reconnected = set()
    node.joining_world_info = None
    node.memory_finder = None
    node._pending_world_handoff = {}
    node._world_to_public = {}

    node._sm_public = SessionManager(
        conn=conn,
        p2p_name=NodeConn.P2P_PUBLIC,
        local_peer_id=node_peer_id,
        local_node_id=node_id,
        local_profile_factory=lambda: node.profile.get_all_profile(),
        decision_hook=node._session_decision_hook,
    )
    node._sm_world = SessionManager(
        conn=conn,
        p2p_name=NodeConn.P2P_WORLD,
        local_peer_id=node_peer_id + "-w",
        local_node_id=node_id,
        local_profile_factory=lambda: node.profile.get_all_profile(),
        decision_hook=node._session_decision_hook,
    )
    return node, conn


def make_remote_peer(remote_peer_id: str, node_id: str, node_inbox: deque,
                     p2p_name: str = NodeConn.P2P_PUBLIC, profile_factory=None):
    """A remote peer talking to the shell: a bare SessionManager on a FakeConn
    whose sends land in `node_inbox` (one of conn.inbox_by_p2p). Returns
    (sm, conn, remote_inbox); pump the remote with apump(remote_inbox, sm)."""
    remote_conn = FakeConn(p2p_public=FakeP2P(peer_id=remote_peer_id, log_sub="remote"))
    remote_conn.send_inbox = node_inbox
    remote_inbox = deque()
    sm = SessionManager(
        conn=remote_conn,
        p2p_name=p2p_name,
        local_peer_id=remote_peer_id,
        local_node_id=node_id,
        local_profile_factory=profile_factory or (lambda: full_profile_dict(node_id)),
    )
    return sm, remote_conn, remote_inbox
