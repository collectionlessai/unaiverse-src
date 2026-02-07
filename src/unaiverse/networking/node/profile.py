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
import re
import json
import psutil
import hashlib
import platform
import datetime
import requests
import ipaddress
from datetime import timezone
from enum import Enum
from typing import TypedDict
from functools import lru_cache
from math import radians, cos, sin, sqrt, atan2
from ....unaiverse.dataprops import DatapropsData
from .connpool import ExtendedPeerInfosData
from ....unaiverse.world import WorldBadgeData

from pydantic import BaseModel, Field, model_validator, EmailStr, UUID4, IPvAnyAddress

# ---- STATIC INFOS ---
IP_SERVICES: list[str] = [
    "https://api.ipify.org",
    "https://icanhazip.com",
    "https://ident.me",
    "https://checkip.amazonaws.com",
]
CREATED_UTC_PATTERN = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$'
EMAIL_PATTERN = r'^[\w\.-]+@[\w\.-]+\.\w+$'
# ---

# ---- ENUMS AND DATA CLASSES ----
class NodeType(Enum):
    """Defines constants for different node types in the network."""
    HUMAN = "human"
    AGENT = "agent"
    WORLD = "world"

class GeoLocationMethod(Enum):
    """Defines constants for different methods of determining a node's location."""
    IP = "ip"
    MANUAL = "manual"
    
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
    badge_id: str # The unique identifier of the badge.
    badge_type: str # The type of badge awarded.
    description: str # A textual description of the badge.
    last_edit_utc: str # The UTC timestamp of the last edit to the badge.
    score: float # The score associated with the badge.
    world_node_id: str # The unique identifier of the world node that awarded the badge.
    world_node_name: str # The human-readable name of the world node that awarded the badge

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
    world_roles_fsm: dict[str, str] # A dictionary representing the finite state machine (FSM) of world roles, where keys are role names and values are their current states.
    hidden: bool # Indicates whether the node is hidden in the network (hidden means not visible in the map for other users).
# ------


# --- UTILS ---
@lru_cache(maxsize=1) # Never heard about this, credit to Gemini for the idea.
def get_public_ip() -> str:
    """Fetches the public IP address of the current machine using predefined external services.

    The function iterates through a list of known IP services, attempting to retrieve the public IP address.
    If a service fails (due to network issues or service unavailability), it moves on to the next one.
    If all services fail, it returns a string indicating the failure.

    Args:
        None
        
    Returns:
        The public IP address as a string.
    """
    
    for service in IP_SERVICES:
        try:
            response = requests.get(service, timeout=5)
            response.raise_for_status()
            
            # Validate the IP address format
            ip = response.text.strip()
            ipaddress.ip_address(ip)

            return ip
        except (requests.RequestException, ValueError):
            continue
    return "N/A: Unable to fetch public IP from all services."

@lru_cache(maxsize=1)
def get_location_by_ip() -> GeoLocationData | None:
    """Fetches the geographical location of the current machine based on its public IP address.

    The function uses the 'ip-api.com' service to retrieve location data such as country, city, latitude, and longitude.
    If the request is successful and the data is valid, it returns a dictionary containing the location information.
    If the request fails or the data is invalid, it returns None.

    Args:
        None
            
    Returns:
        A dictionary containing location data if successful, otherwise None.
    """
    
    current_ip = get_public_ip()
    
    try:
        response = requests.get(f"http://ip-api.com/json/{current_ip}", timeout=5)
        response.raise_for_status()
        data = response.json()
        if data["status"] == "success":
            location_data: GeoLocationData = {
                "country": data.get("country", ""),
                "city": data.get("city", ""),
                "road": data.get("road", ""),
                "latitude": data.get("lat", 0.0),
                "longitude": data.get("lon", 0.0),
            }
            return location_data
        else:
            return None
    except (requests.RequestException, ValueError):
        return None
    
def get_location_by_os() -> GeoLocationData | None:
    """Fetches the geographical location of the current machine based on its operating system settings.
    This function is a placeholder for future implementation. Currently, it returns None.
    Args:
    Raises:
        NotImplementedError: This function is not yet implemented.
    """
    return None

# ----
    
class GeoLocation(BaseModel):
    """Represents the location of a node, which can be determined using different methods.
    
    Args:
        method (GeoLocationMethod | None): The method used to determine the node's location (e.g., IP-based or manual).
        country (str): The country where the node is located.
        city (str): The city where the node is located.
        latitude (float): The latitude coordinate of the node's location.
        longitude (float): The longitude coordinate of the node's location.
        road (str): The road or street of the node's location.

    Raises:        
        ValueError: If the method is not an instance of GeoLocationMethod Enum.
        ValueError: If pre_fetched_data is provided but is missing required keys.
    """
    
    method: GeoLocationMethod | None = Field(default=None, description="The method used to determine the node's location (e.g., IP-based or manual).")
    country: str = Field(..., description="The country where the node is located.")
    city: str = Field(..., description="The city where the node is located.")
    latitude: float = Field(..., description="The latitude coordinate of the node's location.")
    longitude: float = Field(..., description="The longitude coordinate of the node's location.")
    road: str = Field(default="", description="The road or street of the node's location.")
    
    
    @model_validator(mode="before")
    def fetch_location_data(cls, data: GeoLocationData | str) -> GeoLocationData | None:
        """ Validates and fetches location data if the method is IP-based.

        Args:
            data (GeoLocationData | str): A dictionary containing the method and pre_fetched_data fields.
        Returns:
            GeoLocationData | None: A dictionary with the method and location data fields populated. If the method is IP-based, the location data is fetched using the get_location_by_ip function.
        Raises:
            ValueError: If the method is not an instance of GeoLocationMethod Enum.
            ValueError: If pre_fetched_data is provided but is missing required keys.
        """
        if isinstance(data, str): # Case in which we got a dumped string
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                raise ValueError(f"[NodeProfile][GeoLocation] Input data string is not valid JSON.")
        
        if not isinstance(data, dict):
            raise ValueError(f"[NodeProfile][GeoLocation] Input data must be a dictionary, got: {type(data)}")
        
        method = data.get("location_method", None)
        
        if method == GeoLocationMethod.IP or method == GeoLocationMethod.IP.value:
            location_data: GeoLocationData | None = get_location_by_ip()
            if location_data is None:
                raise ValueError(f"[NodeProfile][GeoLocation] Unable to fetch location data using IP method.")
            data = location_data
        
        return data
    
    def __eq__(self, value: object) -> bool:
        """ Compares two GeoLocationData instances for equality.

        Args:
            value (object): The other object to compare with.
        
        Returns:
            bool: True if both instances share the same country/city, False otherwise.
        """
        if not isinstance(value, GeoLocation):
            return False
        return (self.country == value.country) and (self.city == value.city)
    
    def same_country(self, value: 'GeoLocation') -> bool:
        """ Checks if two GeoLocationData instances are located in the same country.

        Args:
            value (GeoLocation): The other object to compare with.
        Returns:
            bool: True if both instances are in the same country, False otherwise.
        """
        if not isinstance(value, GeoLocation):
            return False
        return self.country == value.country
    
    def same_road(self, value: 'GeoLocation') -> bool:
        """ Checks if two GeoLocationData instances are located in the same city.

        Args:
            value (GeoLocation): The other object to compare with.
        
        Returns:
            bool: True if both instances are in the same city, False otherwise.
        """
        if not isinstance(value, GeoLocation):
            return False
        if self.road and value.road:
            return self.road == value.road and self.city == value.city and self.country == value.country
        return False
    
    def same_by_radius(self, value: 'GeoLocation', max_radius_km: float = 10.0) -> bool:
        """ Checks if two GeoLocationData instances are located in the same radius, centered on obj1 latitude and longitude.

        Args:
            value (GeoLocation): The other object to compare with.
            max_radius_km (float): The maximum radius in kilometers to consider as "same region". Default is 10 km.
        
        Returns:
            bool: True if both instances are in the same region, False otherwise.
        """
        if not isinstance(value, GeoLocation):
            return False
        # Radius check
        R = 6371.0  # Earth radius in kilometers (roughly)
        lat1 = radians(self.latitude) # Convert latitude to radians
        lon1 = radians(self.longitude) # Convert longitude to radians
        lat2 = radians(value.latitude) # same for obj2
        lon2 = radians(value.longitude) # same for obj2
        dlon = lon2 - lon1 # Difference in longitude
        dlat = lat2 - lat1 # Difference in latitude
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2 # Haversine formula
        c = 2 * atan2(sqrt(a), sqrt(1 - a)) # Angular distance in radians
        distance = R * c # Distance in kilometers
        return distance <= max_radius_km # If we do math correctly, distance should be positive


class Account(BaseModel):
    """ Account data class, containing all the information about the node's owner. 
        This information comes from the registration form on the platform, therefore they are static.
    
    Raises:
        ValueError: If the email format is invalid.
        ValueError: If input data is missing required keys.
    """
    
    name: str = Field(..., description="The name of the node's owner.", gt=1)
    surname: str = Field(..., description="The surname of the node's owner.", gt=1)
    title: str = Field(..., description="The title of the node's owner.")
    organization: str = Field(..., description="The organization the node's owner is affiliated with.")
    email: EmailStr = Field(..., description="The email address of the node's owner.")
    inspector_node_id: UUID4 = Field(..., description="The inspector node ID is the human agent associated to the node's owner, which is responsible for inspecting the node's behavior.")
    
    def __eq__(self, value: object) -> bool:
        """ Compares two Account instances for equality.

        Args:
            value (object): The other object to compare with.
        Returns:
            bool: True if both instances share the same email, False otherwise.
        """
        if not isinstance(value, Account):
            return False
        return self.email == value.email
    
class Badge(BaseModel):
    """ Represents a badge awarded to an agent in a world.

    Raises:
        ValueError: If any of the input parameters are of the wrong type.
    """
    
    badge_id: str = Field(..., description="The unique identifier of the badge.")
    badge_type: str = Field(..., description="The type of badge awarded.")
    description: str = Field(..., description="A textual description of the badge.")
    last_edit_utc: str = Field(..., description="The UTC timestamp of the last edit to the badge.")
    score: float = Field(..., description="The score associated with the badge.")
    world_node_id: str = Field(..., description="The unique identifier of the world node that awarded the badge.")
    world_node_name: str = Field(..., description="The human-readable name of the world node that awarded the badge.")
    
class WorldSummary(BaseModel):
    """ Represents the world summary data of a world node.
        This class is currently a placeholder for future implementation, as the current version of the platform does not support world summary data.
    """
    
    world_title: str = Field(description="The title of the world.", default="")
    world_agents: list[ExtendedPeerInfosData] = Field(..., description="A list of ExtendedPeerInfosData instances representing the agents currently present in the world.")
    world_agents_count: int = Field(..., description="The total number of world agents currently present in the world.", ge=0)
    world_masters: list[ExtendedPeerInfosData] = Field(..., description="A list of ExtendedPeerInfosData instances representing the world masters currently managing the world.")
    world_masters_count: int = Field(..., description="The total number of world masters currently managing the world.", ge=0)
    total_agents: int = Field(..., description="The total number of agents (agents+masters) that is currently in the world.", ge=0)
    agent_badges: list[Badge] = Field(..., description="A list of badges given to agents in the world.")
    agent_badges_count: int = Field(..., description="The total number of badges given to agents in the world.", ge=0)
    streams_count: int = Field(..., description="The total number of streams currently active in the world.", ge=0)
    

class StaticProfile(BaseModel):
    """ Represents the static profile data of a node, which includes all the information that does not change over time.

    Raises:
        ValueError: If any of the input parameters are of the wrong type or if created_utc is not in the correct format.
        ValueError: If input data is missing required keys.
    """
    
    node_id: UUID4 = Field(..., description="Unique identifier for the node.")
    node_type: NodeType = Field(..., description="The type of the node (human, agent, world).")
    node_name: str = Field(..., description="A human-readable name for the node.", gt=1)
    node_description: str = Field(..., description="A description of the node's purpose or functionality.", gt=1)
    created_utc: str = Field(..., description="The UTC timestamp when the node was created.", pattern=CREATED_UTC_PATTERN)
    max_nr_connections: int = Field(..., description="The maximum number of connections this node can handle.", gt=0)
    allowed_node_ids: list[UUID4] = Field(..., description="A list of node IDs that are allowed to connect to this node. An empty list means no restrictions.")
    world_masters_node_ids : list[UUID4] = Field(..., description="A list of World master node IDs. An empty list means no world masters.")
    certified: bool = Field(..., description="Indicates whether the node is certified by the platform.")
    geo_location_method: GeoLocationMethod = Field(..., description="The method used to determine the node's location (e.g., 'ip', 'manual').")
    account: Account = Field(..., description="An instance of the Account class containing information about the node's owner.")

class DynamicProfile(BaseModel):
    """ Represents the dynamic profile data of a node, which includes all the information that can change over time.
        This class is currently a placeholder for future implementation, as the current version of the platform does not support dynamic profile data.
    """
    os: str = Field(..., description="The operating system of the node.")
    cpu_cores: int = Field(..., description="The number of CPU cores available on the node.", ge=0)
    logical_cpus: int = Field(..., description="The number of logical CPUs available on the node.", ge=0)
    memory_gb: float = Field(..., description="The total memory available on the node in gigabytes.", ge=0)
    memory_avail: float = Field(..., description="The available memory on the node in gigabytes.", ge=0)
    memory_used: float = Field(..., description="The used memory on the node in gigabytes.", ge=0)
    timestamp: str = Field(..., description="The UTC timestamp when the dynamic data was last updated.", pattern=CREATED_UTC_PATTERN)
    public_ip_address: IPvAnyAddress = Field(..., description="The public IP address of the node.")
    guessed_location: GeoLocation = Field(..., description="The guessed location of the node based on its public IP address or manual input (depending on the geolocation method).")
    peer_id: str = Field(..., description="The peer ID of the node, which is a unique identifier used in the p2p network to identify the node during interactions with other nodes.", )