"""P2P Channel Module - UNaIVERSE Networking.

This module provides the P2PChannel class for building and parsing P2P communication channels.
Channels are used for both direct messaging (DM) and pub/sub (PS) communication patterns.

Channel Format:
    {source_peer_id}::{channel_type}:{destination_info}[-{content_type}][~{channel_trail}]

Where:
    - source_peer_id: The Peer ID of the sender/owner
    - channel_type: Either 'dm' (direct message) or 'ps' (pub/sub)
    - destination_info: Peer ID for DM, topic name for PS
    - content_type: Optional type identifier (e.g., 'world_approval', 'agent_approval')
    - channel_trail: Optional additional identifier for sub-channels

Examples:
    Direct Message: "QmABC123::dm:QmXYZ789-json"
    Direct Message with trail: "QmABC123::dm:QmXYZ789-protobuf~channel1"
    PubSub: "QmABC123::ps:my_topic-json"
    Rendezvous: "QmABC123::ps:rv"

A Collectionless AI Project (https://collectionless.ai) / UNaIVERSE SRL (https://unaiverse.ai)

- Registration/Login: https://unaiverse.io
- Code Repositories: https://github.com/collectionlessai/
- Main Developers: Stefano Melacci (Project Leader), Christian Di Maio, Tommaso Guidi
"""

from enum import Enum
from typing import Callable, Optional, Any
from pydantic import BaseModel, Field, field_validator, model_validator, model_serializer
from typing_extensions import Self
from unaiverse.networking.messages import CONTENT_TYPES, ContentTypes


class ChannelType(str, Enum):
    """Channel communication type.

    Attributes:
        DM: Direct message - one-to-one communication
        PS: Pub/Sub - one-to-many broadcast communication
    """
    DM = "dm"
    PS = "ps"


class Channel(BaseModel):
    """Represents a P2P communication channel.

    This class encapsulates all the information needed to construct a valid
    channel string for the UNaIVERSE networking protocol. It supports
    both direct messaging (DM) and pub/sub (PS) patterns.

    The model serializes directly to the channel string format and can parse
    channel strings using model_validate().

    Attributes:
        source_peer_id: The Peer ID of the sender/channel owner
        channel_type: Type of channel (DM or PS)
        destination_peer_id: Target Peer ID (for DM) or empty string (for PS)
        topic_name: Topic name (for PS) or empty string (for DM)
        content_type: Optional content type identifier (e.g., 'json', 'protobuf')
        channel_trail: Optional additional identifier for sub-channels

    Examples:
        Create a direct message channel:
        ```python
        channel = Channel(
            source_peer_id="QmABC123",
            channel_type=ChannelType.DM,
            destination_peer_id="QmXYZ789",
            content_type=ContentTypes.WORLD_APPROVAL
        )
        print(channel.model_dump(mode='json'))  # "QmABC123::dm:QmXYZ789-world_approval"
        ```

        Create a pub/sub channel with trail:
        ```python
        channel = Channel(
            source_peer_id="QmABC123",
            channel_type=ChannelType.PS,
            topic_name="my_topic",
            content_type=ContentTypes.AGENT_APPROVAL,
            channel_trail="pluto"
        )
        print(channel.model_dump(mode='json'))  # "QmABC123::ps:my_topic-agent_approval~pluto"
        ```

        Parse an existing channel string:
        ```python
        channel = Channel.model_validate("QmABC123::dm:QmXYZ789-world_approval~pluto")
        print(channel.destination_peer_id)  # "QmXYZ789"
        print(channel.content_type)  # "world_approval"
        print(channel.channel_trail)  # "pluto"
        ```
    """

    source_peer_id: str = Field(
        ...,
        description="The Peer ID of the sender/channel owner",
        min_length=1
    )
    channel_type: ChannelType = Field(
        ...,
        description="Type of channel: 'dm' for direct messages or 'ps' for pub/sub"
    )
    destination_peer_id: str = Field(
        default="",
        description="Target Peer ID (only for DM channels, empty for PS)"
    )
    topic_name: str = Field(
        default="",
        description="Topic name (only for PS channels, empty for DM)"
    )
    content_type: Optional[CONTENT_TYPES] = Field(
        default=None,
        description="Optional content type identifier"
    )
    channel_trail: Optional[str] = Field(
        default=None,
        description="Optional additional identifier for sub-channels"
    )

    @model_validator(mode="after")
    def validate_channel_configuration(self) -> Self:
        """Validates that channel is properly configured based on type.

        Rules:
            - DM channels must have destination_peer_id and no topic_name
            - PS channels must have topic_name and no destination_peer_id

        Raises:
            ValueError: If channel configuration is invalid
        """
        if self.channel_type == ChannelType.DM:
            if not self.destination_peer_id:
                raise ValueError(
                    "DM channels must have a destination_peer_id"
                )
            if self.topic_name:
                raise ValueError(
                    "DM channels should not have a topic_name"
                )
        elif self.channel_type == ChannelType.PS:
            if not self.topic_name:
                raise ValueError(
                    "PS channels must have a topic_name"
                )
            if self.destination_peer_id:
                raise ValueError(
                    "PS channels should not have a destination_peer_id"
                )

        return self

    @model_validator(mode='wrap')
    @classmethod
    def validate_from_string(cls, value: Any, handler: Callable[[Any], Self], info: Any) -> Self:
        """Validates and parses channel strings or dict input.

        Handles both dictionary input (standard Pydantic) and string input
        (channel string format).

        Args:
            value: Either a dict or a channel string
            handler: The default validation handler
            info: Validation context info

        Returns:
            Validated Channel instance

        Raises:
            ValueError: If string format is invalid
        
        Examples:
            ```python
            # Validating from a channel string
            channel_str = "QmABC123::dm:QmXYZ789-world_approval~pluto"
            channel = Channel.model_validate(channel_str)
            print(channel.source_peer_id)  # "QmABC123"
            print(channel.channel_type)  # ChannelType.DM
            print(channel.destination_peer_id)  # "QmXYZ789"
            '''
        """
        # If it's a string, parse it into dict format
        if isinstance(value, str):
            # Split by :: to get source and rest
            try:
                source_part, rest = value.split("::", 1)
            except ValueError:
                raise ValueError(
                    f"Invalid channel format: '{value}' - missing '::' separator"
                )

            # Split rest by : to get type and destination info
            try:
                type_part, destination_info = rest.split(":", 1)
            except ValueError:
                raise ValueError(
                    f"Invalid channel format: '{value}' - missing channel type separator"
                )

            # Validate channel type
            try:
                channel_type = ChannelType(type_part)
            except ValueError:
                raise ValueError(
                    f"Invalid channel type: '{type_part}' - must be 'dm' or 'ps'"
                )

            # Parse destination info (may contain content_type and trail)
            content_type = None
            channel_trail = None

            # Check for trail (~)
            if "~" in destination_info:
                destination_info, channel_trail = destination_info.split("~", 1)

            # Check for content type (-)
            if "-" in destination_info:
                destination_info, content_type = destination_info.split("-", 1)

            # Create dict based on channel type
            if channel_type == ChannelType.DM:
                value = {
                    "source_peer_id": source_part,
                    "channel_type": channel_type,
                    "destination_peer_id": destination_info,
                    "content_type": content_type,
                    "channel_trail": channel_trail
                }
            else:  # PS
                value = {
                    "source_peer_id": source_part,
                    "channel_type": channel_type,
                    "topic_name": destination_info,
                    "content_type": content_type,
                    "channel_trail": channel_trail
                }

        # Proceed with standard validation
        return handler(value)

    @field_validator("source_peer_id", "destination_peer_id")
    @classmethod
    def validate_peer_id_format(cls, v: str) -> str:
        """Validates peer ID format (basic check).

        Peer IDs should start with 'Qm' (v0) or '12D3' (v1).
        Empty strings are allowed for optional fields.

        Args:
            v: Peer ID string to validate

        Returns:
            Validated peer ID string

        Raises:
            ValueError: If peer ID format is invalid
        """
        if v and not (v.startswith("Qm") or v.startswith("12D3")):
            Warning(
                f"Peer ID '{v}' does not appear to be a valid libp2p peer ID "
                f"(should start with 'Qm' or '12D3')"
            )
        return v

    @model_serializer(mode='wrap', when_used='json')
    def serialize_to_string(self, serializer: Callable[[Any], str], info: Any) -> str:
        """Serializes the channel to its string representation for JSON mode.

        Format: {source}::{type}:{destination}[-{content_type}][~{trail}]

        Returns:
            Channel string suitable for use with P2P send/subscribe methods

        Examples:
            ```python
                channel = Channel(
                    source_peer_id="QmABC",
                    channel_type=ChannelType.DM,
                    destination_peer_id="QmXYZ",
                    content_type="json"
                )
                print(channel.model_dump(mode='json')) # 'QmABC::dm:QmXYZ-json'
            ```
        """
        # Base: source::type:destination
        if self.channel_type == ChannelType.DM:
            destination = self.destination_peer_id
        else:  # PS
            destination = self.topic_name

        channel_str = f"{self.source_peer_id}::{self.channel_type.value}:{destination}"

        # Add content type if present
        if self.content_type:
            channel_str += f"-{self.content_type.value}"

        # Add channel trail if present
        if self.channel_trail:
            channel_str += f"~{self.channel_trail}"

        return channel_str

    @classmethod
    def create_dm_channel(
        cls,
        source_peer_id: str,
        destination_peer_id: str,
        content_type: Optional[ContentTypes] = None,
        channel_trail: Optional[str] = None
    ) -> "Channel":
        """Convenience method to create a Direct Message channel.

        Args:
            source_peer_id: Sender's Peer ID
            destination_peer_id: Recipient's Peer ID
            content_type: Optional content type identifier
            channel_trail: Optional channel trail

        Returns:
            Channel configured for direct messaging

        Example:
            ```python
                channel = Channel.create_dm_channel(
                source_peer_id="QmABC",
                    destination_peer_id="QmXYZ",
                    content_type=ContentTypes.PROFILE_REQUEST
                )
                print(channel.model_dump(mode='json')) # 'QmABC::dm:QmXYZ-profile_request'
            ```
        """
        return cls(
            source_peer_id=source_peer_id,
            channel_type=ChannelType.DM,
            destination_peer_id=destination_peer_id,
            content_type=content_type,
            channel_trail=channel_trail
        )

    @classmethod
    def create_pubsub_channel(
        cls,
        source_peer_id: str,
        topic_name: str,
        content_type: Optional[ContentTypes] = None,
        channel_trail: Optional[str] = None
    ) -> "Channel":
        """Convenience method to create a Pub/Sub channel.

        Args:
            source_peer_id: Channel owner's Peer ID
            topic_name: Topic name to publish/subscribe to
            content_type: Optional content type identifier
            channel_trail: Optional channel trail

        Returns:
            Channel configured for pub/sub

        Example:
            ```python
                channel = Channel.create_pubsub_channel(
                    source_peer_id="QmABC",
                    topic_name="my_topic",
                    content_type=ContentTypes.STREAM_SAMPLE
                )
                print(channel.model_dump(mode='json')) # 'QmABC::ps:my_topic-stream_sample'
            ```
            
        """
        return cls(
            source_peer_id=source_peer_id,
            channel_type=ChannelType.PS,
            topic_name=topic_name,
            content_type=content_type,
            channel_trail=channel_trail
        )

    @classmethod
    def create_rendezvous_channel(cls, source_peer_id: str) -> "Channel":
        """Convenience method to create a Rendezvous channel.

        Rendezvous is a special pub/sub topic used for peer discovery.

        Args:
            source_peer_id: Channel owner's Peer ID

        Returns:
            Channel configured for rendezvous

        Example:
            ```python
                channel = Channel.create_rendezvous("QmABC")
                print(channel.model_dump(mode='json')) # 'QmABC::ps:rv'
            ```
        """
        return cls(
            source_peer_id=source_peer_id,
            channel_type=ChannelType.PS,
            topic_name="rv"  # Special rendezvous topic
        )

    def __str__(self) -> str:
        """Returns the string representation of the channel."""
        return self.model_dump(mode='json') # type: ignore

    def __repr__(self) -> str:
        """Returns a detailed representation of the channel."""
        return (
            f"Channel(type={self.channel_type.value}, "
            f"source={self.source_peer_id[:8]}..., "
            f"{'dest=' + self.destination_peer_id[:8] + '...' if self.destination_peer_id else ''}"
            f"{'topic=' + self.topic_name if self.topic_name else ''}, "
            f"content={self.content_type.value if self.content_type else 'none'}, "
            f"trail={self.channel_trail or 'none'})"
        )

    def is_dm(self) -> bool:
        """Check if this is a direct message channel.

        Returns:
            True if channel is for direct messaging
        """
        return self.channel_type == ChannelType.DM

    def is_pubsub(self) -> bool:
        """Check if this is a pub/sub channel.

        Returns:
            True if channel is for pub/sub
        """
        return self.channel_type == ChannelType.PS

    def is_rendezvous(self) -> bool:
        """Check if this is a rendezvous channel.

        Returns:
            True if channel is the special rendezvous topic
        """
        return self.channel_type == ChannelType.PS and self.topic_name == "rv"

    def get_destination(self) -> str:
        """Get the destination identifier (peer ID or topic name).

        Returns:
            Destination peer ID for DM, topic name for PS
        """
        if self.channel_type == ChannelType.DM:
            return self.destination_peer_id
        return self.topic_name

    def with_content_type(self, content_type: ContentTypes) -> "Channel":
        """Create a new channel with a different content type.

        Args:
            content_type: New content type identifier

        Returns:
            New Channel instance with updated content_type
        """
        return self.model_copy(update={"content_type": content_type})

    def with_trail(self, channel_trail: str) -> "Channel":
        """Create a new channel with a different trail.

        Args:
            channel_trail: New channel trail identifier

        Returns:
            New Channel instance with updated channel_trail
        """
        return self.model_copy(update={"channel_trail": channel_trail})