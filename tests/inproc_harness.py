"""In-process application chat-path harness for the unaiverse framework.

This module is imported explicitly (like ``tests/realnet_helpers.py``), NOT a
conftest. It wires two REAL ``Agent`` instances back-to-back over an in-process
``InProcNodeConn`` that faithfully reproduces the application chat path of the
real ``NodeConn`` (src/unaiverse/networking/node/connpool.py): the EXACT channel
string of ``send`` (connpool.py:613-617), the codec round-trip through
``Msg.to_bytes``/``Msg.from_bytes`` (the wire boundary that lives in P2P), the
membership gate of ``get_messages`` (connpool.py:539-540), and the piggyback
rewrite from ``token+bit`` to ``node_id+bit`` (connpool.py:543-549).

The receive side is driven by the REAL ``Node.__handle_network_messages``
(receive-side mode A): the harness feeds Msg into ``conn.get_messages`` and lets
the real Node loop do the sort + dispatch, so the real gates/ordering are reused
rather than re-implemented.

Design (a switchboard shared by both conns):

  * one chat inbox deque per (node, p2p_name);
  * ``send`` builds the DM channel with the sender's per-network local peer-id,
    constructs a real ``Msg``, ROUND-TRIPS it through the protobuf codec, and
    appends it to the recipient's inbox for that network;
  * ``publish`` fans a Msg out to every subscriber of the exact channel;
  * ``get_messages`` drains the inbox, enforces the membership gate, rewrites
    each piggyback to ``sender_node_id + trailing_inspector_bit`` (JWT verify is
    skipped, identity is trusted in-process), and returns ``(list[Msg], [])``.
"""

import os
import json
from collections import deque
from collections.abc import Callable

from unaiverse.clock import clock
from unaiverse.custom import Custom
from unaiverse.agent import Agent
from unaiverse.networking.p2p.messages import Msg
from unaiverse.networking.node.connpool import NodeConn
from unaiverse.networking.node.profile import NodeProfile
from unaiverse.streams.streams import Stream
from unaiverse.streams.dataprops import DataProps


# Role bits mirrored from AgentBasics (part of the public protocol): bit0 = "about
# world", bit1 = "master". 0 = public, 1 = world agent, 3 = world master.
ROLE_PUBLIC = 0
ROLE_WORLD_AGENT = (1 << 0) | (0 << 1)   # 1
ROLE_WORLD_MASTER = (1 << 0) | (1 << 1)  # 3


# Deterministic clock (same shape as tests/test_streams_io.py::FakeClock)

class HarnessClock:
    """A fully controllable stand-in for ``unaiverse.clock._Clock``.

    Only the surface the streams + Msg path touch is implemented:
    ``get_cycle``/``get_cycle_time`` (a timestamp bound 1:1 to the cycle, so
    distinct cycles always yield distinct timestamps, avoiding the real-clock
    duplicate-timestamp footgun), ``get_time`` (irrelevant when delta<=0) and
    ``next_cycle`` to advance. Inject it with ``install_clock`` and advance it
    between ticks via ``next_cycle``."""

    def __init__(self):
        self.cycle = -1
        self._time = 1000.0

    def get_cycle(self) -> int:
        return self.cycle

    def get_cycle_time(self) -> float:
        return float(self.cycle)

    def get_time(self, passed: bool = False) -> float:
        return self._time

    def get_time_ms(self, monotonic: bool = False) -> int:
        return int(self._time * 1000)

    def next_cycle(self) -> bool:
        self.cycle += 1
        self._time += 1.0
        return True


def install_clock() -> HarnessClock:
    """Inject a deterministic clock into the global ``clock`` singleton and return
    it. Mirrors the fixture pattern of tests/test_streams_io.py; the caller is
    responsible for restoring the previous instance (see the smoke test fixture)."""
    fc = HarnessClock()
    clock.set(fc)
    return fc


# The in-process switchboard shared by two InProcNodeConn instances

class InProcSwitchboard:
    """Shared in-memory bus between InProcNodeConn instances.

    Per network: ``owner[net][local_peer_id] -> conn`` resolves a recipient to
    its owning conn (this is what ``send`` routes on, mirroring NodeConn._sm_of),
    and ``subs[net][channel] -> set(local_peer_id)`` is the pubsub subscriber set.
    """

    def __init__(self):
        self.owner = {NodeConn.P2P_PUBLIC: {}, NodeConn.P2P_WORLD: {}}
        self.subs = {NodeConn.P2P_PUBLIC: {}, NodeConn.P2P_WORLD: {}}

    def register(self, conn: "InProcNodeConn") -> None:
        for net, pid in conn.local_peer_id_by_net.items():
            if pid is not None:
                self.owner[net][pid] = conn

    def net_of(self, peer_id: str) -> str | None:
        """Which network is ``peer_id`` reachable on? World is checked first
        (world DMs use the private peer-id), then public."""
        for net in (NodeConn.P2P_WORLD, NodeConn.P2P_PUBLIC):
            if peer_id in self.owner[net]:
                return net
        return None


# The faithful application chat-path conn

class InProcNodeConn:
    """An in-process stand-in for ``NodeConn`` covering the application chat path.

    One instance per node; two (or more) instances share a single
    ``InProcSwitchboard``. ``send``/``publish`` push fully-formed, codec-round-
    tripped ``Msg`` objects into the recipient's per-network chat inbox;
    ``get_messages`` drains it, gates membership and rewrites the piggyback, and
    hands back ``(list[Msg], list[SessionInbound])`` exactly like the real conn.
    """

    # P2P names + the pool-name constants a real Node shell may reference.
    P2P_PUBLIC = NodeConn.P2P_PUBLIC
    P2P_WORLD = NodeConn.P2P_WORLD
    PUBLIC = NodeConn.PUBLIC
    WORLD = NodeConn.WORLD
    WORLD_NODE = NodeConn.WORLD_NODE
    WORLD_AGENTS = NodeConn.WORLD_AGENTS
    WORLD_MASTERS = NodeConn.WORLD_MASTERS
    IN_PUBLIC = NodeConn.IN_PUBLIC
    OUT_PUBLIC = NodeConn.OUT_PUBLIC
    IN_WORLD_AGENTS = NodeConn.IN_WORLD_AGENTS
    OUT_WORLD_AGENTS = NodeConn.OUT_WORLD_AGENTS

    def __init__(self, node_id: str, public_peer_id: str, world_peer_id: str,
                 switchboard: InProcSwitchboard):
        self.node_id = node_id
        self.local_peer_id_by_net = {self.P2P_PUBLIC: public_peer_id,
                                     self.P2P_WORLD: world_peer_id}
        self.switchboard = switchboard

        # Per-network chat inbox (the application Msg list get_messages drains).
        self.chat_inbox = {self.P2P_PUBLIC: deque(), self.P2P_WORLD: deque()}

        # Membership gate: a sender is admitted iff in a pool OR in this cycle's
        # frozen world membership snapshot (connpool.py:539-540).
        self.peer_id_to_pool_name: dict[str, str] = {}
        self.cycle_membership_peers: set[str] = set()

        # The per-peer node-id table (substitutes verify_token in-process) and the
        # role/misc + token tables the real conn maintains.
        self.peer_id_to_node_id: dict[str, str] = {}
        self.peer_id_to_misc: dict[str, int] = {}
        self.peer_id_to_token: dict[str, str] = {}

        # The remote world host's peer-id (get_world_node_peer_id), addresses, and
        # the close_session record needed later for the bug-3 e2e.
        self._world_node_peer_id: str | None = None
        self.addresses_set: list[tuple[str, list]] = []
        self.closed_sessions: list[tuple[str, int]] = []

        switchboard.register(self)

    # Egress (async)

    async def send(self, peer_id: str, channel_trail: str | None,
                   content_type: str, content=None) -> bool:
        """Direct message egress. Replicates connpool.py:608-634 verbatim: the
        channel prefix is the SENDER's local peer-id on the recipient's network
        (NOT the recipient), piggyback = token('node_id') + '0', and the Msg is
        round-tripped through the protobuf codec before delivery."""
        net = self.switchboard.net_of(peer_id)
        if net is None:
            return False  # No instance owns this peer (NodeConn._sm_of -> None).
        local_peer_id = self.local_peer_id_by_net[net]

        if channel_trail is not None and len(channel_trail) > 0:
            channel = f"{local_peer_id}::dm:{peer_id}-{content_type}~{channel_trail}"
        else:
            channel = f"{local_peer_id}::dm:{peer_id}-{content_type}"

        msg = Msg(sender=local_peer_id, content_type=content_type, content=content,
                  channel=channel, piggyback=self.node_id + "0")

        # Codec round-trip (the wire boundary lives in P2P.send/get_messages): a
        # fresh object, JSON/stream-sample coercions applied, no cross-talk.
        msg = Msg.from_bytes(msg.to_bytes())

        self.switchboard.owner[net][peer_id].chat_inbox[net].append(msg)
        return True

    async def publish(self, peer_id: str, channel: str,
                      content_type: str, content=None) -> bool:
        """Pubsub egress. The channel is passed through verbatim (the net_hash),
        and the Msg is delivered to every subscriber of that exact channel on the
        owner's network. The publisher is NOT self-delivered (gossipsub does not
        loop a publish back to the publisher)."""
        net = self.switchboard.net_of(peer_id) or self.P2P_WORLD
        local_peer_id = self.local_peer_id_by_net[net]
        msg = Msg(sender=local_peer_id, content_type=content_type, content=content,
                  channel=channel, piggyback=self.node_id + "0")
        msg_bytes = msg.to_bytes()
        for sub_pid in self.switchboard.subs[net].get(channel, set()):
            sub_conn = self.switchboard.owner[net].get(sub_pid)
            if sub_conn is None or sub_conn is self:
                continue
            sub_conn.chat_inbox[net].append(Msg.from_bytes(msg_bytes))
        return True

    async def subscribe(self, peer_id: str, channel: str,
                        default_p2p_name: str | None = None) -> bool:
        net = default_p2p_name or self.P2P_WORLD
        self.switchboard.subs[net].setdefault(channel, set()).add(self.local_peer_id_by_net[net])
        return True

    async def unsubscribe(self, peer_id: str, channel: str,
                          default_p2p_name: str | None = None) -> bool:
        net = default_p2p_name or self.P2P_WORLD
        self.switchboard.subs[net].get(channel, set()).discard(self.local_peer_id_by_net[net])
        return True

    # Ingress (async)

    async def get_messages(self, p2p_name: str, allowed_not_connected_peers=None):
        """Drain one network's chat inbox. Reproduces connpool.py:518-560: assert
        the caller passed None, gate the sender against pools | cycle membership,
        and REWRITE each kept Msg's piggyback to ``node_id + inspector_bit`` (real
        JWT verify is skipped; the node-id is looked up from the trusted table).
        Returns ``(processed_messages, session_messages=[])``."""
        assert allowed_not_connected_peers is None, \
            "allowed_not_connected_peers is set by the Coordinator, not by the caller"
        allowed = self.cycle_membership_peers

        inbox = self.chat_inbox[p2p_name]
        processed: list[Msg] = []
        while inbox:
            msg = inbox.popleft()

            # Membership gate (silently drop unknown senders, like production).
            if msg.sender not in self.peer_id_to_pool_name and msg.sender not in allowed:
                continue

            node_id = self.peer_id_to_node_id.get(msg.sender)
            if node_id is None:
                continue  # verify_token -> None: discard.

            # Keep the trailing inspector bit, replace the token with the node-id
            # so node.py's piggyback[:-1] / piggyback[-1] stay correct.
            inspector_bit = msg.piggyback[-1] if msg.piggyback else "0"
            msg.piggyback = node_id + inspector_bit
            processed.append(msg)
            if msg.sender in self.peer_id_to_pool_name:
                self.peer_id_to_token[msg.sender] = node_id

        return processed, []

    async def pump_sessions(self, inbound_by_p2p_name: dict, now: float) -> list:
        """No Session FSM is exercised on the pure chat path (peers are pre-joined),
        so this returns no events."""
        return []

    async def drop_peer(self, peer_id: str, reason: int | None = None,
                        settled: bool = False) -> list:
        """Voluntary per-peer teardown (mirrors NodeConn.drop_peer): with a reason,
        RECORD the GOAWAY (close) for the bug-3 e2e; then drop the peer from the
        pool/identity indices (the evict half of the real drop_peer). A settled
        reckoning never touches the session side (mirrors NodeConn). Returns [] (the
        CLOSED event would arrive later via the FSM, not modelled here)."""
        if reason is not None and not settled:
            self.closed_sessions.append((peer_id, reason))
        self.peer_id_to_pool_name.pop(peer_id, None)
        self.peer_id_to_token.pop(peer_id, None)
        self.peer_id_to_node_id.pop(peer_id, None)
        return []

    async def close_session(self, peer_id: str, reason: int) -> list:
        """GOAWAY a Session. RECORDS (peer_id, reason) for the bug-3 e2e and returns
        [] (the CLOSED event would arrive later via the FSM, not modelled here)."""
        self.closed_sessions.append((peer_id, reason))
        return []

    async def remove(self, peer_id: str) -> bool:
        """Drop a peer from the pool/token indices (keeps peer_id_to_misc, like
        the real ``remove``). Returns True."""
        self.peer_id_to_pool_name.pop(peer_id, None)
        self.peer_id_to_token.pop(peer_id, None)
        self.peer_id_to_node_id.pop(peer_id, None)
        return True

    # Read facade (sync)

    def get_role(self, peer_id: str) -> int:
        return self.peer_id_to_misc.get(peer_id, 0)

    def set_role(self, peer_id: str, new_role: int) -> None:
        self.peer_id_to_misc[peer_id] = new_role

    def get_world_node_peer_id(self) -> str | None:
        return self._world_node_peer_id

    def set_world(self, world_peer_id: str | None) -> None:
        self._world_node_peer_id = world_peer_id

    def get_pool_of(self, peer_id: str) -> str | None:
        return self.peer_id_to_pool_name.get(peer_id)

    def is_connected(self, peer_id: str, pool_names=None) -> bool:
        return peer_id in self.peer_id_to_pool_name

    def get_last_token(self, peer_id: str) -> str | None:
        return self.peer_id_to_token.get(peer_id)

    def set_addresses_in_peer_info(self, peer_id: str, addresses: list) -> None:
        self.addresses_set.append((peer_id, list(addresses)))

    def reset_rendezvous_tag(self) -> None:
        pass

    # Harness wiring helpers (not part of the NodeConn surface)

    def register_peer(self, peer_id: str, node_id: str,
                      role: int = ROLE_WORLD_AGENT,
                      pool: str | None = None) -> None:
        """Seed the membership gate + identity tables so a peer's chat survives
        ``get_messages`` (pool admission) and gets a valid node-id (piggyback
        rewrite). ``role`` feeds ``get_role`` (used by ``add_agent``)."""
        self.peer_id_to_pool_name[peer_id] = pool or self.IN_WORLD_AGENTS
        self.peer_id_to_node_id[peer_id] = node_id
        self.peer_id_to_misc[peer_id] = role


# NodeProfile skeleton (offline, no machine-spec network calls)

def _build_node_profile(node_id: str, public_peer_id: str, world_peer_id: str,
                        role: str = "world_agent",
                        world_node_peer_id: str | None = None) -> NodeProfile:
    """Build a real ``NodeProfile`` whose dynamic profile already carries
    ``peer_id``/``private_peer_id`` (so ``get_peer_ids`` resolves real ids).

    Every top-level dynamic key is pre-filled with a non-None placeholder so
    ``NodeProfile._fill_missing_specs`` never calls ``_get_current_specs`` (which
    would hit the network for the public IP / geolocation). ``proc_inputs`` /
    ``proc_outputs`` / ``streams`` are overwritten by ``update_streams_in_profile``
    during ``set_node_info``.
    """
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
        "inspector_node_id": None,
    }
    dynamic = {
        "os": "test-os",
        "cpu_cores": 1,
        "logical_cpus": 1,
        "memory_gb": 1.0,
        "memory_avail": 1.0,
        "memory_used": 0.0,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "public_ip_address": "127.0.0.1",
        "guessed_location": {},
        "peer_id": public_peer_id,
        "peer_addresses": [],
        "private_peer_id": world_peer_id,
        "private_peer_addresses": [],
        "proc_inputs": [],
        "proc_outputs": [],
        "streams": [],
        "world_roles_fsm": {},
        "hidden": False,
        "connections": {
            "public_agents": [],
            "world_agents": [],
            "world_masters": [],
            "world_peer_id": world_node_peer_id,
            "role": role,
        },
    }
    return NodeProfile(static=static, dynamic=dynamic, cv=[])


async def _noop_async(*args, **kwargs):
    return None


# Build a REAL Agent + Node shell wired onto an InProcNodeConn

def build_real_agent_node(peer_id: str, proc: Callable | None,
                          proc_inputs, proc_outputs, conn: InProcNodeConn,
                          node_id: str = "node", world_peer_id: str | None = None,
                          role: str = "public_agent"):
    """Build a REAL ``Agent`` wired via ``set_node_info`` onto ``conn``, wrapped in
    a ``Node`` shell (``Node.__new__``) of node_type AGENT with the startup
    wildcard applied (node.py:442-445 equivalent). Returns ``(node, agent)``.

    ``proc`` should be a trivial echo (str in -> str out) so STREAM_SAMPLE content
    stays a plain str (codec-stable). ``conn`` must already have been built with
    this node's ``public_peer_id``/``world_peer_id`` registered on the switchboard.

    ``role`` defaults to ``"public_agent"`` so a bare agent is drivable by ``tick``
    out of the box (the public HSM is fully loaded from public.json). For world
    chat tests pass ``role="world_agent"`` AND load the agent's world FSM (e.g. via
    ``accept_new_role``) before driving ``behave``: a bare world-agent's private
    HSM is the empty placeholder and ``behav.act`` would raise on it. Note that the
    chat *delivery* path (send/get_messages/STREAM_SAMPLE dispatch) does not depend
    on ``role`` at all (routing is by peer-id, the gate is seeded explicitly).
    """
    from unaiverse.networking.node.node import Node

    if world_peer_id is None:
        world_peer_id = peer_id + "-w"

    profile = _build_node_profile(node_id, peer_id, world_peer_id, role=role,
                                  world_node_peer_id=conn.get_world_node_peer_id())

    agent = Agent(proc=proc, proc_inputs=proc_inputs, proc_outputs=proc_outputs)
    agent.set_node_info(conn, profile, _noop_async, _noop_async, "")

    # Startup wildcard substitution (node.py:442-445): replace <agent> with the
    # agent's public peer-id in the lone-wolf HSM so public-path actions resolve.
    agent.behav_lone_wolf.update_wildcard(Custom.AGENT_WILDCARD, f"{peer_id}")
    agent.behav_lone_wolf.apply_wildcards()

    node = Node.__new__(Node)
    node.node_id = node_id
    node.node_type = Node.AGENT
    node.agent = agent
    node.hosted = agent
    node.world = None
    node.conn = conn
    node.profile = profile
    node._interact_mode_opts = None
    return node, agent


def register_known_processor_stream(agent: Agent, sender_world_peer_id: str,
                                    name: str = "processor") -> str:
    """Register, on ``agent``, a known (not-owned) text stream under the net_hash a
    DM STREAM_SAMPLE from ``sender_world_peer_id`` (channel_trail='processor')
    normalizes to: ``{sender_world_peer_id}::dm:processor``. This is what
    ``add_compatible_streams`` does internally for a peer's processor output, but
    with an explicit stream ``name`` so the content key matches. Returns the
    net_hash key under which the sample will land."""
    net_hash = DataProps.build_net_hash(sender_world_peer_id, pubsub=False,
                                         name_or_group="processor")
    stream = Stream(name=name, group="processor", data_type="text")
    agent.add_stream(stream, owned=False, net_hash=net_hash)
    return net_hash


def cross_register(node_a, agent_a, node_b, agent_b,
                   role: int = ROLE_WORLD_AGENT) -> None:
    """Make each agent know the other as a world peer.

    Seeds each conn's membership/identity tables (so the get_messages gate passes
    and the piggyback rewrite finds a node-id), sets the counterpart's role (so
    ``add_agent`` classifies it), then calls ``add_agent(other_world_peer,
    other_profile)`` on each side. ``add_agent`` reads the counterpart profile's
    ``proc_outputs`` and registers its processor streams via
    ``add_compatible_streams`` (under name 'proc_output_0')."""
    import asyncio

    conn_a: InProcNodeConn = node_a.conn
    conn_b: InProcNodeConn = node_b.conn
    a_world = conn_a.local_peer_id_by_net[InProcNodeConn.P2P_WORLD]
    b_world = conn_b.local_peer_id_by_net[InProcNodeConn.P2P_WORLD]

    conn_a.register_peer(b_world, node_b.node_id, role=role)
    conn_b.register_peer(a_world, node_a.node_id, role=role)

    async def _wire():
        await agent_a.add_agent(b_world, agent_b.node_profile)
        await agent_b.add_agent(a_world, agent_a.node_profile)

    asyncio.run(_wire())


# Per-tick driver (canonical order)

async def tick(node) -> None:
    """Run ONE real cycle in the canonical order (node.py:1274-1363):
    ``__handle_network_messages`` (deliver + dispatch) -> ``send_stream_samples``
    (outbound data egress) -> ``behave`` (HSM step). Advance the deterministic
    clock BEFORE the cycle so each tick records a distinct timestamp."""
    clock.next_cycle()
    await node._Node__handle_network_messages()
    await node.agent.send_stream_samples()
    await node.agent.behave()


# Minimal world folder (for later world tests)

def make_min_world_folder(tmp_path) -> str:
    """Write a minimal ``<role>.json`` HSM template + ``agent.py`` companion into a
    fresh subdir of ``tmp_path`` and return its path. Minimal placeholder for
    future world tests; not exercised by the chat-path smoke test."""
    folder = os.path.join(str(tmp_path), "world")
    os.makedirs(folder, exist_ok=True)

    # A one-state HSM template (the shape World.load_and_refactor_* consumes).
    role_hsm = {
        "states": [{"name": "empty", "actions": []}],
        "transitions": [],
        "initial_state": "empty",
    }
    with open(os.path.join(folder, "world_agent.json"), "w", encoding="utf-8") as f:
        json.dump(role_hsm, f)

    companion = (
        "from unaiverse.agent import Agent\n\n\n"
        "class WorldAgent(Agent):\n"
        "    pass\n"
    )
    with open(os.path.join(folder, "agent.py"), "w", encoding="utf-8") as f:
        f.write(companion)

    return folder
