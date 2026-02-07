from typing import TypedDict
from unaiverse.networking.node.connpool import ExtendedPeerInfosData
from unaiverse.world import WorldBadgeData
from unaiverse.dataprops import DatapropsData

class GeoLocationData(TypedDict):
    """ Location data dictionary type, containing all the information about the node's location. 
        This information can be determined using different methods: actually in python using the "ip" or "manual" method.
    """
    country: str # The country where the node is located.
    city: str # The city where the node is located.
    road: str # The road or street of the node's location.
    latitude: float # The latitude coordinate of the node's location.
    longitude: float # The longitude coordinate of the node's location.

class AccountData(TypedDict):
    """ Account data dictionary type, containing all the information about the node's owner. 
        This information comes from the registration form on the platform, therefore they are static.
    """
    name: str # The name of the node's owner.
    surname: str # The surname of the node's owner.
    title: str # The title of the node's owner.
    organization: str # The organization the node's owner is affiliated with.
    email: str # The email address of the node's owner.
    inspector_node_id: str # The inspector node ID is the human agent associated to the node's owner, which is responsible for inspecting the node's behavior
    
class StaticProfileData(TypedDict):
    """ Static data dictionary type, containing all the information about the node that does not change over time. 
        This information is set during the node registration and remains constant throughout the node's lifecycle.
    """
    node_id: str # Unique identifier for the node.
    node_type: str # The type of the node (human, agent, world).
    node_name: str # A human-readable name for the node.
    node_description: str # A description of the node's purpose or functionality.
    created_utc: str # The UTC timestamp when the node was created.
    max_nr_connections: int # The maximum number of connections this node can handle.
    allowed_node_ids: list[str] # A list of node IDs that are allowed to connect to this node. An empty list means no restrictions.
    world_masters_node_ids: list[str] # A list of World master node IDs. An empty list means no world masters.
    certified: bool # Indicates whether the node is certified by the platform.
    location_method: str # The method used to determine the node's location (e.g., "ip", "manual").
    name: str # The name of the node's owner.
    surname: str # The surname of the node's owner.
    title: str # The title of the node's owner.
    organization: str # The organization the node's owner is affiliated with.
    email: str # The email address of the node's owner.
    inspector_node_id: str # The inspector node ID is the human agent associated to the node

class CVData(TypedDict):
    """ CV data dictionary type, containing all the information about the node's curriculum vitae. 
        This information is not currently supported by the platform, but it is defined here for future implementation.
    """
    pass

class ConnectionsData(TypedDict):
    """ Connections data dictionary type, containing all the information about the node's connections in the p2p network. 
    """
    public_agents: list[ExtendedPeerInfosData] # A list of ExtendedPeerInfosData instances representing the current public agent connections of the node in the p2p network.
    world_agents: list[ExtendedPeerInfosData] # A list of ExtendedPeerInfosData instances representing the current private agent connections of the node in the p2p network.
    world_masters: list[ExtendedPeerInfosData] # A list of ExtendedPeerInfosData instances representing the current world master connections of the node in the p2p network.
    world_peer_id: str # The peer ID of the world node, which is a unique identifier used in the p2p network to identify the world node during interactions with other nodes.
    role: str # The role of the node in the p2p network 

class WorldSummaryData(TypedDict):
    """ World summary data dictionary type, containing all the information about the world node's summary. 
        This information is not currently supported by the platform, but it is defined here for future implementation.
    """
    world_title: str # The title of the world.
    world_agents: list[ExtendedPeerInfosData] # A list of ExtendedPeerInfosData instances representing the agents currently present in the world.
    world_agents_count: int # The total number of world agents currently present in the world.
    world_masters: list[ExtendedPeerInfosData] # A list of ExtendedPeerInfosData instances representing the world masters currently managing the world.
    world_masters_count: int # The total number of world masters currently managing the world.
    total_agents: int # The total number of agents (agents+masters) that is currently in the world.
    agent_badges: list[WorldBadgeData] # A list of badges given to agents in the world.
    agent_badges_count: int # The total number of badges given to agents in the world.
    streams_count: int # The total number of streams currently active in the world.
    
class DynamicProfileData(TypedDict):
    """ Dynamic data dictionary type, containing all the information about the node that can change over time. 
        This information is not currently supported by the platform, but it is defined here for future implementation.
    """
    os: str # The operating system of the node.
    cpu_cores: int # The number of CPU cores available on the node.
    logical_cpus: int # The number of logical CPUs available on the node.
    memory_gb: float # The total memory available on the node in gigabytes.
    memory_avail: float # The available memory on the node in gigabytes.
    memory_used: float # The used memory on the node in gigabytes.
    timestamp: str # The UTC timestamp when the dynamic data was last updated.
    public_ip_address: str # The public IP address of the node.
    guessed_location: GeoLocationData # The guessed location of the node based on its public IP address.
    peer_id: str # The peer ID of the node, which is a unique identifier used in the p2p network to identify the node during interactions with other nodes.
    peer_addresses: list[str] # A list of peer multi-addresses that the node is listening on the p2p network.
    private_peer_id: str # The private peer ID of the node, which is used for internal communications within the node itself.
    private_peer_addresses: list[str] # A list of private peer multi-addresses that the node is listening on for private communications.
    proc_inputs: list[DatapropsData] # A list of DatapropsData instances representing the input data properties that the agent processor is capable of processing.
    proc_outputs: list[DatapropsData] # A list of DatapropsData instances representing the output data properties that the agent processor is capable of producing.
    streams: list[DatapropsData] # A list of DatapropsData instances representing the data properties that the agent processor can stream in real-time.
    connections: ConnectionsData # The connections data of the node in the p2p network.
    world_summary: WorldSummaryData # The world summary data of the node, if the node is a world.
    world_roles_fsm: