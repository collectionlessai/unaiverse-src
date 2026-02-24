# P2PChannel Complete Definition

## Overview

The `P2PChannel` class is a comprehensive Pydantic model for building and parsing P2P communication channels in the UNaIVERSE networking protocol. It handles both Direct Messaging (DM) and Pub/Sub (PS) patterns with full validation and type safety.

## Channel Format Specification

```
{source_peer_id}::{channel_type}:{destination_info}[-{content_type}][~{channel_trail}]
```

### Components

1. **source_peer_id** (required): Sender's/owner's libp2p Peer ID
   - Must start with 'Qm' (CIDv0) or '12D3' (CIDv1)

2. **channel_type** (required): Communication pattern
   - `dm`: Direct Message (one-to-one)
   - `ps`: Pub/Sub (one-to-many)

3. **destination_info** (required): Varies by type
   - For DM: Destination Peer ID
   - For PS: Topic name

4. **content_type** (optional): Content format identifier
   - Examples: `json`, `protobuf`, `stream`, `binary`

5. **channel_trail** (optional): Sub-channel identifier
   - Used to create multiple sub-channels

### Valid Channel Examples

```python
# Direct Messages
"QmABC123::dm:QmXYZ789"                    # Simple DM
"QmABC123::dm:QmXYZ789-json"              # DM with content type
"QmABC123::dm:QmXYZ789-protobuf~stream1"  # DM with type and trail

# Pub/Sub
"QmABC123::ps:my_topic"                   # Simple topic
"QmABC123::ps:my_topic-json"              # Topic with content type
"QmABC123::ps:rv"                         # Rendezvous (special topic)
"QmABC123::ps:world_updates-json~agent1"  # Topic with trail
```

## Class Definition

```python
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Self

class ChannelType(str, Enum):
    """Channel communication type."""
    DM = "dm"  # Direct Message
    PS = "ps"  # Pub/Sub

class P2PChannel(BaseModel):
    """Represents a P2P communication channel with full validation."""

    # Required fields
    source_peer_id: str = Field(..., min_length=1)
    channel_type: ChannelType

    # Type-specific fields (one must be set)
    destination_peer_id: str = Field(default="")  # For DM
    topic_name: str = Field(default="")            # For PS

    # Optional fields
    content_type: Optional[str] = Field(default=None)
    channel_trail: Optional[str] = Field(default=None)

    # Validators ensure proper configuration
    # (DM must have destination, PS must have topic)
```

## Usage Examples

### 1. Creating Channels - Constructor

```python
from unaiverse.networking.p2p.channel import P2PChannel, ChannelType

# Direct Message channel
dm_channel = P2PChannel(
    source_peer_id="QmSourcePeer123",
    channel_type=ChannelType.DM,
    destination_peer_id="QmDestPeer456",
    content_type="json"
)
print(dm_channel.to_string())
# Output: "QmSourcePeer123::dm:QmDestPeer456-json"

# Pub/Sub channel
ps_channel = P2PChannel(
    source_peer_id="QmSourcePeer123",
    channel_type=ChannelType.PS,
    topic_name="world_updates",
    content_type="protobuf",
    channel_trail="stream1"
)
print(ps_channel.to_string())
# Output: "QmSourcePeer123::ps:world_updates-protobuf~stream1"
```

### 2. Creating Channels - Convenience Methods

```python
# Direct Message
dm = P2PChannel.create_dm(
    source_peer_id="QmSource",
    destination_peer_id="QmDest",
    content_type="json"
)

# Pub/Sub
ps = P2PChannel.create_pubsub(
    source_peer_id="QmSource",
    topic_name="my_topic",
    content_type="protobuf"
)

# Rendezvous (special pub/sub for discovery)
rv = P2PChannel.create_rendezvous("QmSource")
print(rv.to_string())  # "QmSource::ps:rv"
```

### 3. Parsing Existing Channel Strings

```python
# Parse from string
channel_str = "QmABC::dm:QmXYZ-json~trail1"
channel = P2PChannel.from_string(channel_str)

# Access fields with full type safety
print(channel.source_peer_id)        # "QmABC"
print(channel.destination_peer_id)   # "QmXYZ"
print(channel.content_type)          # "json"
print(channel.channel_trail)         # "trail1"
print(channel.is_dm())               # True
print(channel.is_pubsub())           # False
```

### 4. Common Patterns Helper

```python
from unaiverse.networking.p2p.channel import CommonChannels

# Quick channel string generation
rv_channel = CommonChannels.rendezvous("QmPeer123")
# "QmPeer123::ps:rv"

dm_json = CommonChannels.dm_json("QmSource", "QmDest", trail="sub1")
# "QmSource::dm:QmDest-json~sub1"

dm_proto = CommonChannels.dm_protobuf("QmSource", "QmDest")
# "QmSource::dm:QmDest-protobuf"

topic = CommonChannels.pubsub_topic("QmSource", "updates", "json")
# "QmSource::ps:updates-json"
```

### 5. Channel Inspection Methods

```python
channel = P2PChannel.from_string("QmABC::dm:QmXYZ-json")

# Type checking
channel.is_dm()          # True
channel.is_pubsub()      # False
channel.is_rendezvous()  # False

# Get destination (works for both DM and PS)
dest = channel.get_destination()  # "QmXYZ" for DM, topic for PS

# String representations
str(channel)   # "QmABC::dm:QmXYZ-json"
repr(channel)  # "P2PChannel(type=dm, source=QmABC..., dest=QmXYZ..., ...)"
```

### 6. Immutable Updates

```python
original = P2PChannel.create_dm("QmSrc", "QmDst", "json")

# Create new channel with different content type
updated = original.with_content_type("protobuf")
print(updated.to_string())  # "QmSrc::dm:QmDst-protobuf"

# Create new channel with trail
with_trail = original.with_trail("stream1")
print(with_trail.to_string())  # "QmSrc::dm:QmDst-json~stream1"

# Original unchanged (immutable)
print(original.to_string())  # "QmSrc::dm:QmDst-json"
```

## Integration with P2P Class

### Sending Direct Messages

```python
from unaiverse.networking.p2p import P2P
from unaiverse.networking.p2p.channel import P2PChannel

# Create channel
channel = P2PChannel.create_dm(
    source_peer_id=p2p.peer_id,
    destination_peer_id="QmDestPeer",
    content_type="json"
)

# Send message
p2p.send_message_to_peer(
    channel=channel.to_string(),
    msg_bytes=b"Hello, peer!"
)
```

### Subscribing to Topics

```python
# Create pub/sub channel
channel = P2PChannel.create_pubsub(
    source_peer_id=p2p.peer_id,
    topic_name="world_updates",
    content_type="json"
)

# Subscribe
p2p.subscribe_to_topic(channel.to_string())

# Later, unsubscribe
p2p.unsubscribe_from_topic(channel.to_string())
```

### Rendezvous Pattern

```python
# Subscribe to rendezvous for peer discovery
rv_channel = P2PChannel.create_rendezvous(p2p.peer_id)
p2p.subscribe_to_topic(rv_channel.to_string())

# Get discovered peers
rendezvous_state = p2p.get_rendezvous_peers_info()
```

## Integration with ConnectionPools

### In connpool.py Usage

```python
from unaiverse.networking.p2p.channel import P2PChannel

async def send(self, peer_id: str, channel_trail: str | None,
               content_type: str, content: bytes, p2p: P2P | None = None):
    """Send message using P2PChannel."""

    # Build channel using P2PChannel
    channel = P2PChannel.create_dm(
        source_peer_id=p2p.peer_id,
        destination_peer_id=peer_id,
        content_type=content_type,
        channel_trail=channel_trail
    )

    # Create message
    msg = Msg(
        sender=p2p.peer_id,
        content_type=content_type,
        content=content,
        channel=channel.to_string(),  # Convert to string
        piggyback=self.__token + "0"
    )

    # Send
    p2p.send_message_to_peer(channel.to_string(), msg.to_bytes())
```

## Validation Rules

### Automatic Validation

```python
# ✅ Valid - DM with destination
P2PChannel(
    source_peer_id="QmABC",
    channel_type=ChannelType.DM,
    destination_peer_id="QmXYZ"
)

# ✅ Valid - PS with topic
P2PChannel(
    source_peer_id="QmABC",
    channel_type=ChannelType.PS,
    topic_name="my_topic"
)

# ❌ Invalid - DM without destination
P2PChannel(
    source_peer_id="QmABC",
    channel_type=ChannelType.DM
)  # Raises ValidationError

# ❌ Invalid - PS without topic
P2PChannel(
    source_peer_id="QmABC",
    channel_type=ChannelType.PS
)  # Raises ValidationError

# ❌ Invalid - DM with topic (wrong type)
P2PChannel(
    source_peer_id="QmABC",
    channel_type=ChannelType.DM,
    destination_peer_id="QmXYZ",
    topic_name="topic"  # DM shouldn't have topic
)  # Raises ValidationError
```

### Peer ID Format Validation

```python
# ✅ Valid peer IDs
"Qm..."   # CIDv0 format
"12D3..."  # CIDv1 format

# ⚠️ Warning - Invalid format
"InvalidPeerID"  # Raises ValidationError
```

## Advanced Use Cases

### 1. Channel Namespacing with Trails

```python
# Multiple sub-channels for different data streams
base = P2PChannel.create_dm("QmSrc", "QmDst", "json")

video_channel = base.with_trail("video")
audio_channel = base.with_trail("audio")
metadata_channel = base.with_trail("metadata")

# Each has unique string representation
print(video_channel.to_string())     # "QmSrc::dm:QmDst-json~video"
print(audio_channel.to_string())     # "QmSrc::dm:QmDst-json~audio"
print(metadata_channel.to_string())  # "QmSrc::dm:QmDst-json~metadata"
```

### 2. Content Type Negotiation

```python
# Start with one content type
channel = P2PChannel.create_dm("QmSrc", "QmDst", "json")

# Upgrade to different serialization
if supports_protobuf:
    channel = channel.with_content_type("protobuf")
elif supports_msgpack:
    channel = channel.with_content_type("msgpack")
```

### 3. Topic-based Routing

```python
# World-specific topics
world_channel = P2PChannel.create_pubsub(
    source_peer_id=p2p.peer_id,
    topic_name=f"world_{world_id}",
    content_type="json"
)

# Role-based topics
agent_channel = P2PChannel.create_pubsub(
    source_peer_id=p2p.peer_id,
    topic_name=f"agents_{role}",
    content_type="protobuf"
)
```

### 4. Parsing Unknown Channels

```python
# Receive channel string from network
received_channel_str = "QmUnknown::ps:some_topic-protobuf~trail"

# Parse and inspect
channel = P2PChannel.from_string(received_channel_str)

if channel.is_pubsub():
    print(f"Pub/Sub topic: {channel.topic_name}")
    print(f"Content type: {channel.content_type}")

    # Route based on topic
    if channel.topic_name == "rv":
        handle_rendezvous(channel)
    elif "world_" in channel.topic_name:
        handle_world_updates(channel)
```

## Error Handling

```python
from pydantic import ValidationError

# Invalid channel string
try:
    channel = P2PChannel.from_string("invalid_format")
except ValueError as e:
    print(f"Parse error: {e}")

# Invalid configuration
try:
    channel = P2PChannel(
        source_peer_id="QmABC",
        channel_type=ChannelType.DM
        # Missing destination_peer_id!
    )
except ValidationError as e:
    print(f"Validation error: {e}")

# Invalid peer ID format
try:
    channel = P2PChannel.create_dm(
        source_peer_id="NotAValidPeerID",
        destination_peer_id="QmDest"
    )
except ValidationError as e:
    print(f"Invalid peer ID: {e}")
```

## Performance Considerations

### String Caching

```python
# Cache channel strings for frequently used channels
class ChannelCache:
    def __init__(self):
        self._cache = {}

    def get_dm_channel(self, source: str, dest: str,
                       content_type: str) -> str:
        key = (source, dest, content_type)
        if key not in self._cache:
            channel = P2PChannel.create_dm(source, dest, content_type)
            self._cache[key] = channel.to_string()
        return self._cache[key]

# Usage
cache = ChannelCache()
channel_str = cache.get_dm_channel("QmSrc", "QmDst", "json")
```

### Batch Parsing

```python
# Parse multiple channels efficiently
channel_strings = [
    "QmA::dm:QmB-json",
    "QmC::ps:topic1-protobuf",
    "QmD::ps:rv"
]

channels = [P2PChannel.from_string(s) for s in channel_strings]

# Filter by type
dm_channels = [c for c in channels if c.is_dm()]
ps_channels = [c for c in channels if c.is_pubsub()]
```

## Type Safety with mypy/pyright

```python
from unaiverse.networking.p2p.channel import P2PChannel, ChannelType

def send_to_channel(channel: P2PChannel, data: bytes) -> None:
    """Type-safe function signature."""
    channel_str: str = channel.to_string()
    # ... send logic ...

# Type checker will catch errors
channel: P2PChannel = P2PChannel.create_dm("QmSrc", "QmDst")
send_to_channel(channel, b"data")  # ✅ OK

send_to_channel("string", b"data")  # ❌ Type error!
```

## Summary

The `P2PChannel` class provides:

✅ **Type Safety**: Full Pydantic validation
✅ **Parsing**: Bidirectional string ↔ object conversion
✅ **Convenience**: Factory methods for common patterns
✅ **Validation**: Automatic checking of channel configuration
✅ **Immutability**: Safe updates via copy methods
✅ **Inspection**: Type checking and destination extraction
✅ **Integration**: Works seamlessly with existing P2P code

This eliminates string manipulation errors and provides a clean, type-safe API for channel management throughout the UNaIVERSE networking stack.
