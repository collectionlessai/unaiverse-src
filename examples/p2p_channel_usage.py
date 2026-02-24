"""P2PChannel Usage Examples.

This script demonstrates comprehensive usage of the P2PChannel class
for building and parsing P2P communication channels.

Run this to see P2PChannel in action!
"""

from unaiverse.networking.channel import (
    P2PChannel,
    ChannelType,
    CommonChannels
)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def example_1_basic_creation():
    """Example 1: Basic channel creation."""
    print_section("Example 1: Basic Channel Creation")

    # Direct Message (DM) channel
    dm_channel = P2PChannel(
        source_peer_id="QmSourcePeer123",
        channel_type=ChannelType.DM,
        destination_peer_id="QmDestPeer456"
    )
    print(f"DM Channel: {dm_channel.to_string()}")

    # Pub/Sub (PS) channel
    ps_channel = P2PChannel(
        source_peer_id="QmSourcePeer123",
        channel_type=ChannelType.PS,
        topic_name="world_updates"
    )
    print(f"PS Channel: {ps_channel.to_string()}")


def example_2_convenience_methods():
    """Example 2: Using convenience methods."""
    print_section("Example 2: Convenience Methods")

    # Quick DM channel
    dm = P2PChannel.create_dm(
        source_peer_id="QmSrc",
        destination_peer_id="QmDst",
        content_type="json"
    )
    print(f"DM with JSON: {dm.to_string()}")

    # Quick Pub/Sub channel
    ps = P2PChannel.create_pubsub(
        source_peer_id="QmSrc",
        topic_name="announcements",
        content_type="protobuf"
    )
    print(f"PS with Protobuf: {ps.to_string()}")

    # Rendezvous channel (special topic for peer discovery)
    rv = P2PChannel.create_rendezvous("QmSrc")
    print(f"Rendezvous: {rv.to_string()}")


def example_3_parsing_channels():
    """Example 3: Parsing existing channel strings."""
    print_section("Example 3: Parsing Channel Strings")

    # Parse various channel formats
    channels = [
        "QmABC::dm:QmXYZ",
        "QmABC::dm:QmXYZ-json",
        "QmABC::dm:QmXYZ-protobuf~stream1",
        "QmABC::ps:my_topic",
        "QmABC::ps:rv"
    ]

    for channel_str in channels:
        channel = P2PChannel.from_string(channel_str)
        print(f"\nParsed: {channel_str}")
        print(f"  Type: {channel.channel_type.value}")
        print(f"  Source: {channel.source_peer_id}")
        if channel.is_dm():
            print(f"  Destination: {channel.destination_peer_id}")
        else:
            print(f"  Topic: {channel.topic_name}")
        if channel.content_type:
            print(f"  Content Type: {channel.content_type}")
        if channel.channel_trail:
            print(f"  Trail: {channel.channel_trail}")


def example_4_channel_inspection():
    """Example 4: Inspecting channels."""
    print_section("Example 4: Channel Inspection")

    # Create various channels
    dm = P2PChannel.create_dm("QmSrc", "QmDst", "json")
    ps = P2PChannel.create_pubsub("QmSrc", "updates", "protobuf")
    rv = P2PChannel.create_rendezvous("QmSrc")

    channels = [
        ("DM Channel", dm),
        ("PS Channel", ps),
        ("Rendezvous", rv)
    ]

    for name, channel in channels:
        print(f"\n{name}: {channel.to_string()}")
        print(f"  Is DM? {channel.is_dm()}")
        print(f"  Is PubSub? {channel.is_pubsub()}")
        print(f"  Is Rendezvous? {channel.is_rendezvous()}")
        print(f"  Destination: {channel.get_destination()}")


def example_5_immutable_updates():
    """Example 5: Creating modified copies (immutable pattern)."""
    print_section("Example 5: Immutable Updates")

    # Start with a basic channel
    original = P2PChannel.create_dm("QmSrc", "QmDst", "json")
    print(f"Original: {original.to_string()}")

    # Create new channel with different content type
    with_protobuf = original.with_content_type("protobuf")
    print(f"With Protobuf: {with_protobuf.to_string()}")

    # Create new channel with trail
    with_trail = original.with_trail("stream1")
    print(f"With Trail: {with_trail.to_string()}")

    # Create with both
    with_both = original.with_content_type("protobuf").with_trail("stream1")
    print(f"With Both: {with_both.to_string()}")

    # Original unchanged!
    print(f"Original Still: {original.to_string()}")


def example_6_common_patterns():
    """Example 6: Using CommonChannels helpers."""
    print_section("Example 6: Common Channel Patterns")

    # Quick rendezvous
    rv = CommonChannels.rendezvous("QmPeer")
    print(f"Rendezvous: {rv}")

    # Quick DM with JSON
    dm_json = CommonChannels.dm_json("QmSrc", "QmDst")
    print(f"DM JSON: {dm_json}")

    # DM with Protobuf and trail
    dm_proto = CommonChannels.dm_protobuf("QmSrc", "QmDst", trail="sub1")
    print(f"DM Protobuf: {dm_proto}")

    # Pub/Sub topic
    topic = CommonChannels.pubsub_topic("QmSrc", "announcements", "json")
    print(f"PS Topic: {topic}")


def example_7_real_world_usage():
    """Example 7: Real-world integration patterns."""
    print_section("Example 7: Real-World Integration")

    # Simulate P2P peer IDs
    my_peer_id = "QmMyNodePeer123"
    remote_peer_id = "QmRemotePeer456"
    world_peer_id = "QmWorldNode789"

    print("\nScenario 1: Sending a direct message")
    dm_channel = P2PChannel.create_dm(
        source_peer_id=my_peer_id,
        destination_peer_id=remote_peer_id,
        content_type="json",
        channel_trail="chat"
    )
    print(f"  Channel: {dm_channel.to_string()}")
    print(f"  Usage: p2p.send_message_to_peer('{dm_channel.to_string()}', msg_bytes)")

    print("\nScenario 2: Subscribing to world updates")
    world_channel = P2PChannel.create_pubsub(
        source_peer_id=world_peer_id,
        topic_name="world_updates",
        content_type="protobuf"
    )
    print(f"  Channel: {world_channel.to_string()}")
    print(f"  Usage: p2p.subscribe_to_topic('{world_channel.to_string()}')")

    print("\nScenario 3: Peer discovery via rendezvous")
    rv_channel = P2PChannel.create_rendezvous(my_peer_id)
    print(f"  Channel: {rv_channel.to_string()}")
    print(f"  Usage: p2p.subscribe_to_topic('{rv_channel.to_string()}')")

    print("\nScenario 4: Multiple data streams (video/audio/metadata)")
    base_channel = P2PChannel.create_dm(my_peer_id, remote_peer_id, "protobuf")
    video = base_channel.with_trail("video")
    audio = base_channel.with_trail("audio")
    metadata = base_channel.with_trail("metadata")

    print(f"  Video: {video.to_string()}")
    print(f"  Audio: {audio.to_string()}")
    print(f"  Metadata: {metadata.to_string()}")


def example_8_error_handling():
    """Example 8: Error handling."""
    print_section("Example 8: Error Handling")

    from pydantic import ValidationError

    print("\nAttempting invalid channel configurations:")

    # Invalid: DM without destination
    try:
        P2PChannel(
            source_peer_id="QmSrc",
            channel_type=ChannelType.DM
            # Missing destination!
        )
    except ValidationError as e:
        print("✗ DM without destination - ValidationError (expected)")

    # Invalid: PS without topic
    try:
        P2PChannel(
            source_peer_id="QmSrc",
            channel_type=ChannelType.PS
            # Missing topic!
        )
    except ValidationError as e:
        print("✗ PS without topic - ValidationError (expected)")

    # Invalid: Bad peer ID format
    try:
        P2PChannel.create_dm("BadPeerID", "QmDst")
    except ValidationError as e:
        print("✗ Invalid peer ID format - ValidationError (expected)")

    # Invalid: Malformed channel string
    try:
        P2PChannel.from_string("invalid_format")
    except ValueError as e:
        print("✗ Malformed string - ValueError (expected)")

    print("\n✅ All errors handled correctly!")


def example_9_advanced_patterns():
    """Example 9: Advanced usage patterns."""
    print_section("Example 9: Advanced Patterns")

    my_peer = "QmMyPeer"
    dest_peer = "QmDestPeer"

    print("\nPattern 1: Channel namespacing with trails")
    base = P2PChannel.create_dm(my_peer, dest_peer, "json")
    channels = {
        "commands": base.with_trail("cmd"),
        "responses": base.with_trail("resp"),
        "events": base.with_trail("events"),
        "logs": base.with_trail("logs")
    }

    for name, channel in channels.items():
        print(f"  {name}: {channel.to_string()}")

    print("\nPattern 2: Content type negotiation")
    base_channel = P2PChannel.create_dm(my_peer, dest_peer, "json")
    print(f"  Start: {base_channel.to_string()}")

    # Upgrade to protobuf
    upgraded = base_channel.with_content_type("protobuf")
    print(f"  Upgraded: {upgraded.to_string()}")

    print("\nPattern 3: Topic-based routing")
    world_id = "world_123"
    role = "agent"

    world_topic = P2PChannel.create_pubsub(
        my_peer,
        f"world_{world_id}_updates",
        "json"
    )
    print(f"  World topic: {world_topic.to_string()}")

    role_topic = P2PChannel.create_pubsub(
        my_peer,
        f"role_{role}_commands",
        "protobuf"
    )
    print(f"  Role topic: {role_topic.to_string()}")


def example_10_roundtrip_conversion():
    """Example 10: String ↔ Object roundtrip."""
    print_section("Example 10: Roundtrip Conversion")

    # Original strings
    original_strings = [
        "QmSrc::dm:QmDst",
        "QmSrc::dm:QmDst-json",
        "QmSrc::dm:QmDst-protobuf~trail1",
        "QmSrc::ps:topic",
        "QmSrc::ps:rv"
    ]

    print("Testing roundtrip conversions:")
    for original in original_strings:
        # String → Object
        channel = P2PChannel.from_string(original)

        # Object → String
        result = channel.to_string()

        # Verify
        match = "✓" if original == result else "✗"
        print(f"  {match} {original}")
        if original != result:
            print(f"    Got: {result}")


def main():
    """Run all examples."""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                      P2PChannel Usage Examples                             ║
║                    UNaIVERSE Networking - P2P Channels                     ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    example_1_basic_creation()
    example_2_convenience_methods()
    example_3_parsing_channels()
    example_4_channel_inspection()
    example_5_immutable_updates()
    example_6_common_patterns()
    example_7_real_world_usage()
    example_8_error_handling()
    example_9_advanced_patterns()
    example_10_roundtrip_conversion()

    print("\n" + "=" * 80)
    print(" All examples completed successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
