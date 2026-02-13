"""Node Profile Module - UNaIVERSE Networking.

This module provides data structures and utilities for managing node profiles in the UNaIVERSE network.
It includes Pydantic models for static and dynamic profile data, geolocation handling, account information,
and CV/badge management.

A Collectionless AI Project (https://collectionless.ai) / UNaIVERSE SRL (https://unaiverse.ai)

- Registration/Login: https://unaiverse.io
- Code Repositories: https://github.com/collectionlessai/
- Main Developers: Stefano Melacci (Project Leader), Christian Di Maio, Tommaso Guidi
"""

# Standard library imports
import hashlib
import ipaddress
import json
import platform
from datetime import datetime, timezone
from enum import Enum
from unaiverse.utils.misc import ttl_cache
from math import atan2, cos, radians, sin, sqrt
from typing import Dict, TypedDict, Union

# 3rd party imports
import psutil
import requests
from pydantic import (
    UUID4,
    BaseModel,
    EmailStr,
    Field,
    IPvAnyAddress,
    PrivateAttr,
    field_validator,
    field_serializer,
    model_validator,
)
from typing_extensions import Self

# 1st party imports
from ....unaiverse.dataprops import DatapropsData
from .connpool import ExtendedPeerInfosData


# ---- STATIC INFOS ---
IP_SERVICES: list[str] = [
    "https://api.ipify.org",
    "https://icanhazip.com",
    "https://ident.me",
    "https://checkip.amazonaws.com",
]  # List of services to fetch public IP address, used in get_public_ip function. The order is important for the fallback mechanism.
# ---


# ---- ENUMS AND DATA CLASSES ----
class NodeType(Enum):
    """Defines constants for different node types in the network.

    Attributes:
        HUMAN: Represents a human node, which is typically associated with a human user or operator in the network.
        AGENT: Represents an agent node, which is an autonomous entity that can perform tasks and interact with other nodes in the network.
        WORLD: Represents a world node, which is a special type of node that can host multiple agents and manage interactions between them in a shared environment.
    """

    HUMAN = "human"
    AGENT = "agent"
    WORLD = "world"


class GeoLocationMethod(Enum):
    """Defines constants for different methods of determining a node's location.

    Attributes:
        IP: Indicates that the node's location is determined based on its public IP address.
        MANUAL: Indicates that the node's location is manually provided by the user.
    """

    IP = "ip"
    MANUAL = "manual"


class ConnectionsData(TypedDict):
    """Connections data dictionary type, containing all the information about the node's connections in the p2p network.

    Attributes:
       public_agents (list[ExtendedPeerInfosData]): A list of ExtendedPeerInfosData instances representing the current public agent connections of the node in the p2p network.
       world_agents (list[ExtendedPeerInfosData]): A list of ExtendedPeerInfosData instances representing the current private agent connections of the node in the p2p network.
       world_masters (list[ExtendedPeerInfosData]): A list of ExtendedPeerInfosData instances representing the current world master connections of the node in the p2p network.
       world_peer_id (str): The peer ID of the world node, which is a unique identifier used in the p2p network to identify the world node during interactions with other nodes.
       role (str): The role of the node in the p2p network, which can be "agent", "world_master" or custom role inherited in the world the agent is currently in.
    """

    public_agents: list[
        ExtendedPeerInfosData
    ]  # A list of ExtendedPeerInfosData instances representing the current public agent connections of the node in the p2p network.
    world_agents: list[
        ExtendedPeerInfosData
    ]  # A list of ExtendedPeerInfosData instances representing the current private agent connections of the node in the p2p network.
    world_masters: list[
        ExtendedPeerInfosData
    ]  # A list of ExtendedPeerInfosData instances representing the current world master connections of the node in the p2p network.
    world_peer_id: str  # The peer ID of the world node, which is a unique identifier used in the p2p network to identify the world node during interactions with other nodes.
    role: str  # The role of the node in the p2p network


# ------


# --- UTILS ---
@ttl_cache(seconds=600)  # Cache the result of this function for 10 minutes
def get_public_ip() -> str:
    """Fetches the public IP address of the current machine using predefined external services.

    The function iterates through a list of known IP services, attempting to retrieve the public IP address.
    If a service fails (due to network issues or service unavailability), it moves on to the next one.
    If all services fail, it returns a string indicating the failure.

    Returns:
        str: The public IP address as a string.
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


@ttl_cache(seconds=600)  # Cache the result of this function for 10 minutes
def get_location_by_ip() -> "GeoLocation | None":
    """Fetches the geographical location of the current machine based on its public IP address.

    The function uses the 'ip-api.com' service to retrieve location data such as country, city, latitude, and longitude.
    If the request is successful and the data is valid, it returns a dictionary containing the location information.
    If the request fails or the data is invalid, it returns None.

    Returns:
        (GeoLocation | None): A GeoLocation object containing location data if successful, otherwise None.
    """

    current_ip = get_public_ip()

    try:
        response = requests.get(f"http://ip-api.com/json/{current_ip}", timeout=5)
        response.raise_for_status()
        data = response.json()
        if data["status"] == "success":
            return GeoLocation(
                method=GeoLocationMethod.IP,
                country=data.get("country", ""),
                city=data.get("city", ""),
                road=data.get("road", ""),
                latitude=data.get("lat", 0.0),
                longitude=data.get("lon", 0.0),
            )
        else:
            return None
    except (requests.RequestException, ValueError):
        return None


def get_location_by_os() -> "GeoLocation | None":
    """Fetches the geographical location of the current machine based on its operating system settings.
    This function is a placeholder for future implementation. Currently, it returns None.

    Raises:
        NotImplementedError: This function is not yet implemented.
    """
    return None


# ----


class GeoLocation(BaseModel):
    """Represents the location of a node, which can be determined using different methods.

    Attributes:
        method (GeoLocationMethod | None): The method used to determine the node's location (e.g., IP-based or manual).
        country (str): The country where the node is located.
        city (str): The city where the node is located.
        latitude (float): The latitude coordinate of the node's location.
        longitude (float): The longitude coordinate of the node's location.
        road (str): The road or street of the node's location.

    Raises:
        ValueError: If the method is not an instance of GeoLocationMethod Enum.
        ValueError: If pre_fetched_data is provided but is missing required keys.

    Example:
        ```python
            # Example of creating a GeoLocation instance with manual method
            manual_location = GeoLocation(
                method=GeoLocationMethod.MANUAL,
                country="USA",
                city="New York",
                latitude=40.7128,
                longitude=-74.0060,
                road="Broadway"
            )

            # Example of creating a GeoLocation instance with IP method (location data will be fetched automatically)
            ip_location = GeoLocation(method=GeoLocationMethod.IP)
        ```
    """

    method: Union[GeoLocationMethod, None] = Field(
        default=None,
        description="The method used to determine the node's location (e.g., IP-based or manual).",
    )
    country: str = Field(..., description="The country where the node is located.")
    city: str = Field(..., description="The city where the node is located.")
    latitude: float = Field(
        ..., description="The latitude coordinate of the node's location."
    )
    longitude: float = Field(
        ..., description="The longitude coordinate of the node's location."
    )
    road: str = Field(
        default="", description="The road or street of the node's location."
    )

    @model_validator(mode="before")
    @classmethod
    def fetch_location_data(cls, data: "GeoLocation | str") -> dict:
        """Validates and fetches location data if the method is IP-based.

        Args:
            data (GeoLocation | str): A dictionary containing the method and pre_fetched_data fields, or a string representation of such a dictionary.
        Returns:
            A GeoLocation object with the method and location data fields populated. If the method is IP-based, the location data is fetched using the get_location_by_ip function.
        Raises:
            ValueError: If the method is not an instance of GeoLocationMethod Enum.
            ValueError: If pre_fetched_data is provided but is missing required keys.
        """
        if isinstance(data, str):  # Case in which we got a dumped string
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                raise ValueError(
                    "[NodeProfile][GeoLocation] Input data string is not valid JSON."
                )

        if not isinstance(data, dict):
            raise ValueError(
                f"[NodeProfile][GeoLocation] Input data must be a dictionary, got: {type(data)}"
            )

        method = data.get("location_method", None)

        if method == GeoLocationMethod.IP or method == GeoLocationMethod.IP.value:
            location_data: GeoLocation | None = get_location_by_ip()
            if location_data is None:
                raise ValueError(
                    "[NodeProfile][GeoLocation] Unable to fetch location data using IP method."
                )
            return location_data.model_dump()  # type: ignore[return-value]

        return data

    def update(self) -> bool:
        """Update the geolocation

        Returns:
            bool: The outcome, True the update went good, False otherwise.
        """
        if self.method == GeoLocationMethod.IP:
            location_data: GeoLocation | None = get_location_by_ip()
            if location_data is None:
                return False
            self.country = location_data.country
            self.city = location_data.city
            self.latitude = location_data.latitude
            self.longitude = location_data.longitude
            self.road = location_data.road
            return True

        if self.method == GeoLocationMethod.MANUAL:
            return True  # For manual method, we assume that the user will update the location data manually, so we return True to indicate that the update is successful.

        # For other methods, we currently do not have an implementation to update the location data, so we return False.
        return False

    def __eq__(self, value: object) -> bool:
        """Compares two GeoLocation instances for equality.

        Args:
            value (object): The other object to compare with.

        Returns:
            bool: True if both instances share the same country/city, False otherwise.

        Example:
            ```python
                loc1 = GeoLocation(method=GeoLocationMethod.MANUAL, country="USA", city="New York", latitude=40.7128, longitude=-74.0060)
                loc2 = GeoLocation(method=GeoLocationMethod.IP, country="USA", city="New York", latitude=40.7128, longitude=-74.0060)
                loc3 = GeoLocation(method=GeoLocationMethod.IP, country="USA", city="Los Angeles", latitude=34.0522, longitude=-118.2437)

                print(loc1 == loc2)  # True
                print(loc1 == loc3)  # False
            ```
        """
        if not isinstance(value, GeoLocation):
            return False
        return (self.country == value.country) and (self.city == value.city)

    def __hash__(self) -> int:
        """Returns a hash based on country and city, consistent with __eq__."""
        return hash((self.country, self.city))

    def same_country(self, value: "GeoLocation") -> bool:
        """Checks if two GeoLocation instances are located in the same country.

        Args:
            value (GeoLocation): The other object to compare with.
        Returns:
            bool: True if both instances are in the same country, False otherwise.
        """
        if not isinstance(value, GeoLocation):
            return False
        return self.country == value.country

    def same_road(self, value: "GeoLocation") -> bool:
        """Checks if two GeoLocation instances are located on the same road.

        Args:
            value (GeoLocation): The other object to compare with.

        Returns:
            bool: True if both instances are on the same road in the same city/country, False otherwise.
        """
        if not isinstance(value, GeoLocation):
            return False
        if self.road and value.road:
            return (
                self.road == value.road
                and self.city == value.city
                and self.country == value.country
            )
        return False

    def same_by_radius(self, value: "GeoLocation", max_radius_km: float = 10.0) -> bool:
        """Checks if two GeoLocation instances are located within a given radius using the Haversine formula.

        Args:
            value (GeoLocation): The other object to compare with.
            max_radius_km (float): The maximum radius in kilometers to consider as "same region". Default is 10 km.

        Returns:
            bool: True if both instances are within the given radius, False otherwise.

        Example:
            ```python
                loc1 = GeoLocation(method=GeoLocationMethod.MANUAL, country="USA", city="New York", latitude=40.7128, longitude=-74.0060)
                loc2 = GeoLocation(method=GeoLocationMethod.MANUAL, country="USA", city="New York", latitude=40.7130, longitude=-74.0070)
                loc3 = GeoLocation(method=GeoLocationMethod.MANUAL, country="USA", city="Los Angeles", latitude=34.0522, longitude=-118.2437)

                print(loc1.same_by_radius(loc2, max_radius_km=1.0))  # True (very close)
                print(loc1.same_by_radius(loc3, max_radius_km=4000.0))  # True (within 4000 km)
                print(loc1.same_by_radius(loc3, max_radius_km=3000.0))  # False (not within 3000 km)
            ```
        """
        if not isinstance(value, GeoLocation):
            return False
        R = 6371.0  # Earth radius in kilometers
        lat1 = radians(self.latitude)
        lon1 = radians(self.longitude)
        lat2 = radians(value.latitude)
        lon2 = radians(value.longitude)
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c
        return distance <= max_radius_km


class Account(BaseModel):
    """Account data class, containing all the information about the node's owner.
    This information comes from the registration form on the platform, therefore they are static.

    Attributes:
        name (str): The name of the node's owner.
        surname (str): The surname of the node's owner.
        title (str): The title of the node's owner.
        organization (str): The organization the node's owner is affiliated with.
        email (EmailStr): The email address of the node's owner. Must be a valid email format.
        inspector_node_id (UUID4): The inspector node ID is the human agent associated to the node's owner, which is responsible for inspecting the node's behavior. Must be a valid UUID4 format.

    Raises:
        ValueError: If the email format is invalid.
        ValueError: If input data is missing required keys.
    """

    name: str = Field(..., description="The name of the node's owner.", min_length=1)
    surname: str = Field(
        ..., description="The surname of the node's owner.", min_length=1
    )
    title: str = Field(..., description="The title of the node's owner.")
    organization: str = Field(
        ..., description="The organization the node's owner is affiliated with."
    )
    email: EmailStr = Field(..., description="The email address of the node's owner.")
    inspector_node_id: UUID4 = Field(
        ...,
        description="The inspector node ID is the human agent associated to the node's owner, which is responsible for inspecting the node's behavior.",
    )

    def __eq__(self, value: object) -> bool:
        """Compares two Account instances for equality.

        Args:
            value (object): The other object to compare with.
        Returns:
            bool: True if both instances share the same email, False otherwise.
        """
        if not isinstance(value, Account):
            return False
        return self.email == value.email

    def __hash__(self) -> int:
        """Returns a hash based on email, consistent with __eq__."""
        return hash(self.email)


class Badge(BaseModel):
    """Represents a badge awarded to an agent in a world.

    Attributes:
        badge_id (str): The unique identifier of the badge.
        badge_type (str): The type of badge awarded.
        description (str): A textual description of the badge.
        last_edit_utc (datetime): The UTC timestamp of the last edit to the badge.
        score (float): The score associated with the badge.
        world_node_id (str): The unique identifier of the world node that awarded the badge.
        world_node_name (str): The human-readable name of the world node that awarded the badge.

    Raises:
        ValueError: If any of the input parameters are of the wrong type.
    """

    badge_id: str = Field(..., description="The unique identifier of the badge.")
    badge_type: str = Field(..., description="The type of badge awarded.")
    description: str = Field(..., description="A textual description of the badge.")
    last_edit_utc: datetime = Field(
        ..., description="The UTC timestamp of the last edit to the badge."
    )
    score: float = Field(..., description="The score associated with the badge.")
    world_node_id: str = Field(
        ...,
        description="The unique identifier of the world node that awarded the badge.",
    )
    world_node_name: str = Field(
        ...,
        description="The human-readable name of the world node that awarded the badge.",
    )

    @field_validator("last_edit_utc", mode="before")
    @classmethod
    def validate_last_edit_utc(cls, value: datetime | str) -> datetime:
        if isinstance(value, str):
            try:
                # Handle 'Z' manually if strictly required by Python < 3.11 for fromisoformat
                if value.endswith("Z"):
                    value = value[:-1] + "+00:00"
                value = datetime.fromisoformat(value)
            except ValueError:
                raise ValueError("last_edit_utc must be a valid ISO 8601 string.")

        if value.tzinfo is None:
            # Assume UTC if naive, or raise error depending on your strictness preference
            value = value.replace(tzinfo=timezone.utc)
        elif value.tzinfo != timezone.utc:
            # Convert to UTC
            value = value.astimezone(timezone.utc)

        return value

    # 3. SERIALIZER (Output): Converts datetime back to ISO string for dumps
    @field_serializer("last_edit_utc")
    def serialize_dt(self, dt: datetime, _info):
        # Returns a string like '2023-01-01T12:00:00Z'
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class WorldSummary(BaseModel):
    """Represents the world summary data of a world node.

    Attributes:
        world_title (str): The title of the world.
        world_agents (list[ExtendedPeerInfosData]): A list of ExtendedPeerInfosData instances representing the agents currently present in the world.
        world_agents_count (int): The total number of world agents currently present in the world.
        world_masters (list[ExtendedPeerInfosData]): A list of ExtendedPeerInfosData instances representing the world masters currently managing the world.
        world_masters_count (int): The total number of world masters currently managing the world.
        total_agents (int): The total number of agents (agents+masters) that is currently in the world.
        agent_badges (list[Badge]): A list of badges given to agents in the world.
        agent_badges_count (int): The total number of badges given to agents in the world.
        streams_count (int): The total number of streams currently active in the world.

    Raises:
        ValueError: If any of the input parameters are of the wrong type.
    """

    world_title: str = Field(description="The title of the world.", default="")
    world_agents: list[ExtendedPeerInfosData] = Field(
        ...,
        description="A list of ExtendedPeerInfosData instances representing the agents currently present in the world.",
    )
    world_agents_count: int = Field(
        ...,
        description="The total number of world agents currently present in the world.",
        ge=0,
    )
    world_masters: list[ExtendedPeerInfosData] = Field(
        ...,
        description="A list of ExtendedPeerInfosData instances representing the world masters currently managing the world.",
    )
    world_masters_count: int = Field(
        ...,
        description="The total number of world masters currently managing the world.",
        ge=0,
    )
    total_agents: int = Field(
        ...,
        description="The total number of agents (agents+masters) that is currently in the world.",
        ge=0,
    )
    agent_badges: list[Badge] = Field(
        ..., description="A list of badges given to agents in the world."
    )
    agent_badges_count: int = Field(
        ...,
        description="The total number of badges given to agents in the world.",
        ge=0,
    )
    streams_count: int = Field(
        ...,
        description="The total number of streams currently active in the world.",
        ge=0,
    )


class StaticProfile(BaseModel):
    """Represents the static profile data of a node, which includes all the information that does not change over time.

    Attributes:
        node_id (UUID4): Unique identifier for the node.
        node_type (NodeType): The type of the node (human, agent, world).
        node_name (str): A human-readable name for the node.
        node_description (str): A description of the node's purpose or functionality.
        created_utc (str): The UTC timestamp when the node was created.
        max_nr_connections (int): The maximum number of connections this node can handle.
        allowed_node_ids (list[UUID4]): A list of node IDs that are allowed to connect to this node. An empty list means no restrictions.
        world_masters_node_ids (list[UUID4]): A list of World master node IDs. An empty list means no world masters.
        certified (bool): Indicates whether the node is certified by the platform.
        geo_location_method (GeoLocationMethod): The method used to determine the node's location (e.g., 'ip', 'manual').
        account (Account): An instance of the Account class containing information about the node's owner.

    Raises:
        ValueError: If any of the input parameters are of the wrong type or if created_utc is not in the correct format.
        ValueError: If input data is missing required keys.
    """

    node_id: UUID4 = Field(..., description="Unique identifier for the node.")
    node_type: NodeType = Field(
        ..., description="The type of the node (human, agent, world)."
    )
    node_name: str = Field(
        ..., description="A human-readable name for the node.", min_length=1
    )
    node_description: str = Field(
        ...,
        description="A description of the node's purpose or functionality.",
        min_length=1,
    )
    created_utc: datetime = Field(
        ...,
        description="The UTC timestamp when the node was created.",
    )
    max_nr_connections: int = Field(
        ..., description="The maximum number of connections this node can handle.", gt=0
    )
    allowed_node_ids: list[UUID4] = Field(
        ...,
        description="A list of node IDs that are allowed to connect to this node. An empty list means no restrictions.",
    )
    world_masters_node_ids: list[UUID4] = Field(
        ...,
        description="A list of World master node IDs. An empty list means no world masters.",
    )
    certified: bool = Field(
        ..., description="Indicates whether the node is certified by the platform."
    )
    geo_location_method: GeoLocationMethod = Field(
        ...,
        description="The method used to determine the node's location (e.g., 'ip', 'manual').",
    )
    account: Account = Field(
        ...,
        description="An instance of the Account class containing information about the node's owner.",
    )

    @field_validator("created_utc", mode="before")
    @classmethod
    def validate_created_utc(cls, value: datetime | str) -> datetime:
        if isinstance(value, str):
            try:
                # Handle 'Z' manually if strictly required by Python < 3.11 for fromisoformat
                if value.endswith("Z"):
                    value = value[:-1] + "+00:00"
                value = datetime.fromisoformat(value)
            except ValueError:
                raise ValueError("created_utc must be a valid ISO 8601 string.")

        if value.tzinfo is None:
            # Assume UTC if naive, or raise error depending on your strictness preference
            value = value.replace(tzinfo=timezone.utc)
        elif value.tzinfo != timezone.utc:
            # Convert to UTC
            value = value.astimezone(timezone.utc)

        return value

    @field_serializer("created_utc")
    def serialize_dt(self, dt: datetime, _info):
        # Returns a string like '2023-01-01T12:00:00Z'
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class DynamicProfile(BaseModel):
    """Represents the dynamic profile data of a node, which includes all the information that can change over time.

    Attributes:
        os (str): The operating system of the node.
        cpu_cores (int): The number of CPU cores available on the node.
        logical_cpus (int): The number of logical CPUs available on the node.
        memory_gb (float): The total memory available on the node in gigabytes.
        memory_avail (float): The available memory on the node in gigabytes.
        memory_used (float): The used memory on the node in gigabytes.
        timestamp (datetime): The UTC timestamp when the dynamic data was last updated.
        public_ip_address (IPvAnyAddress): The public IP address of the node.
        guessed_location (GeoLocation): The guessed location of the node based on its public IP address or manual input (depending on the geolocation method).
        peer_id (str): The peer ID of the node, which is a unique identifier used in the p2p network to identify the node during interactions with other nodes.
        peer_addresses (list[str]): A list of peer multi-addresses that the node is listening on the p2p network.
        private_peer_id (str): The private peer ID of the node, which is used for internal communications within the node itself.
        private_peer_addresses (list[str]): A list of private peer multi-addresses that the node is listening on for private communications.
        proc_inputs (list[DatapropsData]): A list of DatapropsData instances representing the input data properties that the agent processor is capable of processing.
        proc_outputs (list[DatapropsData]): A list of DatapropsData instances representing the output data properties that the agent processor is capable of producing.
        streams (list[DatapropsData]): A list of DatapropsData instances representing the data properties that the agent processor can stream in real-time.
        connections (ConnectionsData): The connections data of the node in the p2p network.
        world_summary (WorldSummary): The world summary data of the node, if the node is a world.
        world_roles_fsm (dict[str, str]): A dictionary representing the finite state machine of world roles for the node.
        hidden (bool): Indicates whether the node is hidden in the network (hidden means not visible in the map for other users).
    """

    os: str = Field(..., description="The operating system of the node.")
    cpu_cores: int = Field(
        ..., description="The number of CPU cores available on the node.", ge=0
    )
    logical_cpus: int = Field(
        ..., description="The number of logical CPUs available on the node.", ge=0
    )
    memory_gb: float = Field(
        ..., description="The total memory available on the node in gigabytes.", ge=0
    )
    memory_avail: float = Field(
        ..., description="The available memory on the node in gigabytes.", ge=0
    )
    memory_used: float = Field(
        ..., description="The used memory on the node in gigabytes.", ge=0
    )
    timestamp: datetime = Field(
        ..., description="The UTC timestamp when the dynamic data was last updated."
    )
    public_ip_address: IPvAnyAddress = Field(
        ..., description="The public IP address of the node."
    )
    guessed_location: GeoLocation = Field(
        ...,
        description="The guessed location of the node based on its public IP address or manual input (depending on the geolocation method).",
    )
    peer_id: str = Field(
        ...,
        description="The peer ID of the node, which is a unique identifier used in the p2p network to identify the node during interactions with other nodes.",
    )
    peer_addresses: list[str] = Field(
        ...,
        description="A list of peer multi-addresses that the node is listening on the p2p network.",
    )
    private_peer_id: str = Field(
        ...,
        description="The private peer ID of the node, which is used for internal communications within the node itself.",
    )
    private_peer_addresses: list[str] = Field(
        ...,
        description="A list of private peer multi-addresses that the node is listening on for private communications.",
    )
    proc_inputs: list[DatapropsData] = Field(
        ...,
        description="A list of DatapropsData instances representing the input data properties that the agent processor is capable of processing.",
    )  # TODO change it after DataProps will be a BaseModel
    proc_outputs: list[DatapropsData] = Field(
        ...,
        description="A list of DatapropsData instances representing the output data properties that the agent processor is capable of producing.",
    )  # TODO change it after DataProps will be a BaseModel
    streams: list[DatapropsData] = Field(
        ...,
        description="A list of DatapropsData instances representing the data properties that the agent processor can stream in real-time.",
    )  # TODO change it after DataProps will be a BaseModel
    connections: ConnectionsData = Field(
        ..., description="The connections data of the node in the p2p network."
    )
    world_summary: WorldSummary = Field(
        ..., description="The world summary data of the node, if the node is a world."
    )
    world_roles_fsm: dict[str, str] = Field(
        ...,
        description="A dictionary representing the finite state machine of world roles for the node.",
    )
    hidden: bool = Field(
        ...,
        description="Indicates whether the node is hidden in the network (hidden means not visible in the map for other users).",
    )

    @model_validator(mode="before")
    @classmethod
    def filling_missing_specs(
        cls, data: "DynamicProfile | str"
    ) -> "DynamicProfile | dict[str, str | int | float | GeoLocation | None]":
        """Validates and fills missing specifications in the dynamic profile data.

        Args:
            data (DynamicProfileData | str): A dictionary containing the dynamic profile data of the node or a JSON string representation of it.
            A ghost field named "filling_missing_specs" is expected in the input data, which is used to trigger the filling of missing specifications. Defaults to False if not provided.

        Returns:
            DynamicProfileData: A dictionary with missing specifications filled in.
        """
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                raise ValueError(
                    "[NodeProfile][DynamicProfile] Input data string is not valid JSON."
                )

        if not isinstance(data, dict):
            raise ValueError(
                f"[NodeProfile][DynamicProfile] Input data must be a dictionary, got: {type(data)}"
            )

        do_i_need_to_fill: bool = data.get("filling_missing_specs", False)

        if not do_i_need_to_fill:
            return data

        system_specs: Dict | None = None
        if "os" not in data or data.get("os") is None:
            system_specs = DynamicProfile._get_current_specs()

        public_ip: str | None = None
        if not data.get("public_ip_address"):
            public_ip = get_public_ip()

        guessed_location: GeoLocation | None = None
        location = data.get("guessed_location")
        if (
            location is None
            or location == {}
            or (isinstance(location, dict) and not location.get("country"))
        ):
            guessed_location = get_location_by_ip()

        if system_specs:
            data.update(system_specs)

        if public_ip:
            data["public_ip_address"] = public_ip

        if guessed_location:
            data["guessed_location"] = guessed_location

        return data

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_timestamp(cls, value: datetime | str) -> datetime:
        """Validates and converts the timestamp field to a datetime object.

        Args:
            value (datetime | str): The input value for the timestamp field, which can be either a datetime object or a string in ISO 8601 format.
        Returns:
            datetime: A datetime object representing the timestamp.
        Raises:
            ValueError: If the input value is not a valid datetime object or a string in ISO 8601 format.
        """
        if isinstance(value, datetime):
            if value.tzinfo is None:
                # Assume UTC if naive, or raise error depending on your strictness preference
                value = value.replace(tzinfo=timezone.utc)
            elif value.tzinfo != timezone.utc:
                # Convert to UTC
                value = value.astimezone(timezone.utc)
            return value

        if isinstance(value, str):
            try:
                # Handle 'Z' manually if strictly required by Python < 3.11 for fromisoformat
                if value.endswith("Z"):
                    value = value[:-1] + "+00:00"
                dt = datetime.fromisoformat(value)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                elif dt.tzinfo != timezone.utc:
                    dt = dt.astimezone(timezone.utc)
                return dt
            except ValueError:
                raise ValueError(
                    "timestamp must be a valid ISO 8601 string or a datetime object."
                )

        raise ValueError(
            f"Invalid type for timestamp: expected datetime or ISO 8601 string, got {type(value)}"
        )

    @field_serializer("timestamp")
    def serialize_timestamp(self, dt: datetime, _info) -> str:
        """Serializes the timestamp field to an ISO 8601 string.

        Args:
            dt (datetime): The datetime object to serialize.
            _info: Additional information provided by Pydantic during serialization (not used in this method).
        Returns:
            str: A string representing the timestamp in ISO 8601 format.
        """
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    def check_and_update_specs(self, update_only: bool = True) -> list[str]:
        """Checks for changes in the dynamic specifications of the node and updates them if necessary.

        Args:
            update_only (bool): If True, unconditionally updates all specs without tracking changes.
                If False, compares current vs new and only updates on change. Defaults to True.

        Returns:
            list[str]: A list of strings describing the changes detected in the dynamic specifications.
            Each string is in the format "spec_name: old_value -> new_value". If no changes are detected
            (or update_only is True), the list is empty.

        Example:
            ```python
                dynamic_profile = DynamicProfile(
                    os="Linux",
                    cpu_cores=4,
                    logical_cpus=8,
                    memory_gb=16.0,
                    memory_avail=10.0,
                    memory_used=6.0,
                    timestamp=datetime.now(timezone.utc),
                    public_ip_address="192.168.1.1",
                    guessed_location=GeoLocation(method=GeoLocationMethod.IP),
                    peer_id="peer1",
                    peer_addresses=["/ip4/234/tcp/4001"],
                    private_peer_id="private_peer1",
                    private_peer_addresses=["/ip4/345/tcp/4002"],
                    proc_inputs=[],
                    proc_outputs=[],
                    streams=[],
                    connections=ConnectionsData(...),
                    world_summary=WorldSummary(...),
                    world_roles_fsm={},
                    hidden=False,
                )
                changes = dynamic_profile.check_and_update_specs(update_only=False)
                if changes:
                    print("Detected changes in dynamic specifications:")
                    for change in changes:
                        print(change)
                else:
                    print("No changes detected in dynamic specifications.")
            ```
        """
        new_specs = self._get_current_specs()
        self.guessed_location.update()

        if update_only:
            for key, val in new_specs.items():
                if hasattr(self, key):
                    setattr(self, key, val)
            return []

        changes: list[str] = []
        for key, new_val in new_specs.items():
            if key == "timestamp":
                continue

            old_val = getattr(self, key, None)

            if isinstance(new_val, float) and isinstance(old_val, float):
                if abs(new_val - old_val) < 1e-6:
                    continue

            if old_val != new_val:
                changes.append(f"{key}: {old_val} -> {new_val}")
                setattr(self, key, new_val)

        if changes:
            self.timestamp = datetime.now(timezone.utc).isoformat()

        return changes

    @staticmethod
    def _get_current_specs() -> dict[str, str | int | float | GeoLocation | None]:
        """Gathers current system specifications.

        Returns:
             A dictionary containing the current system specifications.
        """
        cpu_info: Dict[str, int | None] = DynamicProfile._get_cpu_info()
        memory_info: Dict[str, float] = DynamicProfile._get_memory_info()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "os": DynamicProfile._get_os_spec(),
            "cpu_cores": cpu_info.get("physical_cores"),
            "logical_cpus": cpu_info.get("logical_cores"),
            "memory_gb": memory_info.get("total"),
            "memory_avail": memory_info.get("available"),
            "memory_used": memory_info.get("used"),
            "public_ip_address": get_public_ip(),
        }

    @staticmethod
    def _get_os_spec() -> str:
        """Extracts operating system information.

        Returns:
            str: A string representing the operating system specifications.
        """
        return platform.platform()

    @staticmethod
    def _get_cpu_info() -> dict[str, int | None]:
        """Extracts CPU core information.

        Returns:
            A dictionary containing the number of physical and logical CPU cores.
        """
        try:
            return {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
            }
        except Exception as e:
            print(f"Error getting CPU info: {e}")
            return {"physical_cores": None, "logical_cores": None}

    @staticmethod
    def _get_memory_info() -> dict[str, float]:
        """Extracts memory information in GB.

        Returns:
            A dictionary containing the total, available, and used memory in gigabytes.
        """
        try:
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024**3)
            available_gb = mem.available / (1024**3)
            used_gb = mem.used / (1024**3)
            return {
                "total": float(total_gb),
                "available": float(available_gb),
                "used": float(used_gb),
            }
        except Exception as e:
            print(f"Error getting memory info: {e}")
            return {"total": 0.0, "available": 0.0, "used": 0.0}


class NodeProfile(BaseModel):
    """Represents the complete profile of a node, including both static and dynamic data.

    Attributes:
        static (StaticProfile): An instance of the StaticProfile class containing all the static information about the node.
        dynamic (DynamicProfile): An instance of the DynamicProfile class containing all the dynamic information about the node.
        cv (list[Badge]): A list of badges representing the node's curriculum vitae (CV).
        _last_updated_utc (datetime): A private attribute to track the last time the profile was updated, used for internal purposes.
        _connections_updated (bool): A private attribute to track if there has been a change in connections, used for internal purposes.

    Raises:
        ValueError: If any of the input parameters are of the wrong type or if created_utc is not in the correct format.
        ValueError: If input data is missing required keys.

    Example:
        ```python
            static_profile = StaticProfile(
                node_id=UUID4("123e4567-e89b-12d3-a456-426614174000"),
                node_type=NodeType.AGENT,
                node_name="Test Agent",
                node_description="An agent for testing purposes.",
                created_utc=datetime.now(timezone.utc),
                max_nr_connections=10,
                allowed_node_ids=[],
                world_masters_node_ids=[],
                certified=True,
                geo_location_method=GeoLocationMethod.IP,
                account=Account(
                    name="John",
                    surname="Doe",
                    title="Mr.",
                    organization="Test Org",
                    email="asdf@asdfsd.com",
                    inspector_node_id=UUID4("123e4567-e89b-12d3-a456-426614174001"),
                ),
            )
            dynamic_profile = DynamicProfile(
                os="Linux",
                cpu_cores=4,
                logical_cpus=8,
                memory_gb=16.0,
                ...
            )
            node_profile = NodeProfile(
                static=static_profile,
                dynamic=dynamic_profile,
                cv=[],
            )
        ```
    """

    static: StaticProfile = Field(
        ...,
        description="An instance of the StaticProfile class containing all the static information about the node.",
    )
    dynamic: DynamicProfile = Field(
        ...,
        description="An instance of the DynamicProfile class containing all the dynamic information about the node.",
    )
    cv: list[Badge] = Field(
        ...,
        description="A list of badges representing the node's curriculum vitae (CV).",
    )
    _last_updated_utc: datetime = PrivateAttr(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    _connections_updated: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def order_cv(self) -> Self:
        """Orders the CV badges by last_edit_utc and enforces consistent key ordering within each badge
        for deterministic hashing (must match server-side hash computation).

        Returns:
            Self: The NodeProfile instance with the CV list ordered.
        """
        self.cv = sorted(self.cv, key=lambda badge: badge.last_edit_utc)
        return self

    def update_cv(self, new_cv: list[Badge]) -> None:
        """Updates the node's curriculum vitae (CV) with a new list of badges.

        Args:
            new_cv (list[Badge]): The new list of badges to set as the CV.
        """
        self.cv = sorted(new_cv, key=lambda badge: badge.last_edit_utc)

    def get_static_profile(self) -> StaticProfile:
        """Get static profile data.

        Returns:
            StaticProfile: An instance of StaticProfile containing all the static information about the node.
        """
        return self.static.model_dump()  # type: ignore[return-value]

    def get_dynamic_profile(self) -> DynamicProfile:
        """Get dynamic profile data.

        Returns:
            DynamicProfile: An instance of DynamicProfile containing all the dynamic information about the node.
        """
        return self.dynamic.model_dump()  # type: ignore[return-value]

    def get_cv(self) -> list[Badge]:
        """Get the node's curriculum vitae (CV).

        Returns:
            list[Badge]: A list of badges representing the node's CV.
        """
        return self.cv

    def get_all_profile(self) -> dict:
        """Get the complete profile data.

        Returns:
            dict: A dictionary containing all the profile data (static, dynamic, cv).
        """
        return {
            "static": self.get_static_profile(),
            "dynamic": self.get_dynamic_profile(),
            "cv": [badge.model_dump() for badge in self.cv],
        }

    def verify_cv_hash(self, cv_hash: str) -> tuple[bool, tuple[str, str]]:
        """Verifies if the provided hash matches the computed hash of the current CV.

        Important: The CV is serialized with sorted keys within each badge dict to ensure
        hash consistency with the server-side computation.

        Args:
            cv_hash (str): The hash to verify against the current CV.

        Returns:
            tuple[bool, tuple[str, str]]: A tuple containing a boolean indicating if the hash matches,
            and a tuple with the provided hash and the computed hash.

        Example:
            ```python
                node_profile = NodeProfile(
                    static=StaticProfile(...),
                    dynamic=DynamicProfile(...),
                    cv=[
                        Badge(
                            badge_id="badge1",
                            badge_type="type1",
                            description="First badge",
                            last_edit_utc=datetime(2023, 1, 1, tzinfo=timezone.utc),
                            score=10.0,
                            world_node_id="world1",
                            world_node_name="World One",
                        ),
                    ],
                )
                cv_hash = "..provided_hash_from_rootserver.."
                is_valid, (provided_hash, computed_hash) = node_profile.verify_cv_hash(cv_hash)
                if is_valid:
                    print("CV hash is valid.")
                else:
                    print(f"CV hash is invalid. Provided: {provided_hash}, Computed: {computed_hash}")
            ```
        """
        # Serialize with sorted keys to match server-side hash computation
        # (mirrors the old code's {k: _cv[k] for k in sorted(_cv)} pattern)
        cv_dicts = [badge.model_dump() for badge in self.cv]
        cv_sorted = [{k: d[k] for k in sorted(d)} for d in cv_dicts]
        cv_json = json.dumps(cv_sorted, separators=(",", ":"))
        computed_hash = hashlib.blake2b(
            cv_json.encode("utf-8"), digest_size=16
        ).hexdigest()
        return cv_hash == computed_hash, (cv_hash, computed_hash)

    def mark_change_in_connections(self) -> None:
        """Mark the change in connections, setting the flag to True."""
        self._connections_updated = True

    def unmark_change_in_connections(self) -> None:
        """Unmark the change in connections, setting the flag back to False."""
        self._connections_updated = False

    def connections_changed(self) -> bool:
        """Check if the connections have changed.

        Returns:
            bool: True if the connections have changed, False otherwise.
        """
        return self._connections_updated

    def check_and_update_specs(self, update_only: bool = True) -> list[str]:
        """Delegates spec checking to the DynamicProfile and updates the last updated timestamp.

        Args:
            update_only (bool): If True, unconditionally updates all specs. Defaults to True.

        Returns:
            list[str]: A list of change descriptions, empty if update_only is True or no changes detected.
        """
        changes = self.dynamic.check_and_update_specs(update_only=update_only)
        self._last_updated_utc = datetime.now(timezone.utc)
        return changes
