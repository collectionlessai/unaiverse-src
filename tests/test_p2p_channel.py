"""Tests for P2PChannel class.

This module tests the P2PChannel class functionality including:
- Channel creation (DM and PubSub)
- String parsing and generation
- Validation rules
- Convenience methods
- Integration patterns
"""

import pytest
from pydantic import ValidationError
from unaiverse.networking.channel import (
    P2PChannel,
    ChannelType,
    CommonChannels
)


class TestChannelCreation:
    """Test channel creation methods."""

    def test_create_dm_basic(self):
        """Test basic direct message channel creation."""
        channel = P2PChannel(
            source_peer_id="QmSourcePeer",
            channel_type=ChannelType.DM,
            destination_peer_id="QmDestPeer"
        )
        assert channel.to_string() == "QmSourcePeer::dm:QmDestPeer"

    def test_create_dm_with_content_type(self):
        """Test DM channel with content type."""
        channel = P2PChannel(
            source_peer_id="QmSourcePeer",
            channel_type=ChannelType.DM,
            destination_peer_id="QmDestPeer",
            content_type="json"
        )
        assert channel.to_string() == "QmSourcePeer::dm:QmDestPeer-json"

    def test_create_dm_with_trail(self):
        """Test DM channel with trail."""
        channel = P2PChannel(
            source_peer_id="QmSourcePeer",
            channel_type=ChannelType.DM,
            destination_peer_id="QmDestPeer",
            content_type="protobuf",
            channel_trail="stream1"
        )
        assert channel.to_string() == "QmSourcePeer::dm:QmDestPeer-protobuf~stream1"

    def test_create_pubsub_basic(self):
        """Test basic pub/sub channel creation."""
        channel = P2PChannel(
            source_peer_id="QmSourcePeer",
            channel_type=ChannelType.PS,
            topic_name="my_topic"
        )
        assert channel.to_string() == "QmSourcePeer::ps:my_topic"

    def test_create_pubsub_with_content_type(self):
        """Test pub/sub channel with content type."""
        channel = P2PChannel(
            source_peer_id="QmSourcePeer",
            channel_type=ChannelType.PS,
            topic_name="updates",
            content_type="json"
        )
        assert channel.to_string() == "QmSourcePeer::ps:updates-json"

    def test_create_rendezvous(self):
        """Test rendezvous channel creation."""
        channel = P2PChannel(
            source_peer_id="QmSourcePeer",
            channel_type=ChannelType.PS,
            topic_name="rv"
        )
        assert channel.to_string() == "QmSourcePeer::ps:rv"
        assert channel.is_rendezvous()


class TestConvenienceMethods:
    """Test convenience factory methods."""

    def test_create_dm_convenience(self):
        """Test DM creation via convenience method."""
        channel = P2PChannel.create_dm(
            source_peer_id="QmSrc",
            destination_peer_id="QmDst",
            content_type="json"
        )
        assert channel.to_string() == "QmSrc::dm:QmDst-json"

    def test_create_pubsub_convenience(self):
        """Test pub/sub creation via convenience method."""
        channel = P2PChannel.create_pubsub(
            source_peer_id="QmSrc",
            topic_name="topic1",
            content_type="protobuf"
        )
        assert channel.to_string() == "QmSrc::ps:topic1-protobuf"

    def test_create_rendezvous_convenience(self):
        """Test rendezvous creation via convenience method."""
        channel = P2PChannel.create_rendezvous("QmSrc")
        assert channel.to_string() == "QmSrc::ps:rv"


class TestStringParsing:
    """Test channel string parsing."""

    def test_parse_dm_basic(self):
        """Test parsing basic DM channel."""
        channel = P2PChannel.from_string("QmSrc::dm:QmDst")
        assert channel.source_peer_id == "QmSrc"
        assert channel.channel_type == ChannelType.DM
        assert channel.destination_peer_id == "QmDst"
        assert channel.content_type is None
        assert channel.channel_trail is None

    def test_parse_dm_with_content_type(self):
        """Test parsing DM with content type."""
        channel = P2PChannel.from_string("QmSrc::dm:QmDst-json")
        assert channel.destination_peer_id == "QmDst"
        assert channel.content_type == "json"

    def test_parse_dm_full(self):
        """Test parsing DM with all components."""
        channel = P2PChannel.from_string("QmSrc::dm:QmDst-protobuf~trail1")
        assert channel.destination_peer_id == "QmDst"
        assert channel.content_type == "protobuf"
        assert channel.channel_trail == "trail1"

    def test_parse_pubsub_basic(self):
        """Test parsing basic pub/sub channel."""
        channel = P2PChannel.from_string("QmSrc::ps:topic1")
        assert channel.source_peer_id == "QmSrc"
        assert channel.channel_type == ChannelType.PS
        assert channel.topic_name == "topic1"

    def test_parse_pubsub_full(self):
        """Test parsing pub/sub with all components."""
        channel = P2PChannel.from_string("QmSrc::ps:topic1-json~trail1")
        assert channel.topic_name == "topic1"
        assert channel.content_type == "json"
        assert channel.channel_trail == "trail1"

    def test_parse_invalid_format_no_separator(self):
        """Test parsing invalid format (no :: separator)."""
        with pytest.raises(ValueError, match="missing '::' separator"):
            P2PChannel.from_string("invalid_format")

    def test_parse_invalid_format_no_type(self):
        """Test parsing invalid format (no channel type)."""
        with pytest.raises(ValueError, match="missing channel type separator"):
            P2PChannel.from_string("QmSrc::invalid")

    def test_parse_invalid_channel_type(self):
        """Test parsing invalid channel type."""
        with pytest.raises(ValueError, match="Invalid channel type"):
            P2PChannel.from_string("QmSrc::invalid:dest")


class TestValidation:
    """Test validation rules."""

    def test_dm_requires_destination(self):
        """Test that DM requires destination_peer_id."""
        with pytest.raises(ValidationError):
            P2PChannel(
                source_peer_id="QmSrc",
                channel_type=ChannelType.DM
                # Missing destination_peer_id
            )

    def test_dm_cannot_have_topic(self):
        """Test that DM cannot have topic_name."""
        with pytest.raises(ValidationError):
            P2PChannel(
                source_peer_id="QmSrc",
                channel_type=ChannelType.DM,
                destination_peer_id="QmDst",
                topic_name="topic"  # Should not be set for DM
            )

    def test_ps_requires_topic(self):
        """Test that pub/sub requires topic_name."""
        with pytest.raises(ValidationError):
            P2PChannel(
                source_peer_id="QmSrc",
                channel_type=ChannelType.PS
                # Missing topic_name
            )

    def test_ps_cannot_have_destination(self):
        """Test that pub/sub cannot have destination_peer_id."""
        with pytest.raises(ValidationError):
            P2PChannel(
                source_peer_id="QmSrc",
                channel_type=ChannelType.PS,
                topic_name="topic",
                destination_peer_id="QmDst"  # Should not be set for PS
            )

    def test_peer_id_validation_qm(self):
        """Test peer ID validation for Qm prefix."""
        channel = P2PChannel.create_dm("QmValidPeer", "QmAnotherPeer")
        assert channel.source_peer_id == "QmValidPeer"

    def test_peer_id_validation_12d3(self):
        """Test peer ID validation for 12D3 prefix."""
        channel = P2PChannel.create_dm("12D3ValidPeer", "12D3AnotherPeer")
        assert channel.source_peer_id == "12D3ValidPeer"

    def test_peer_id_validation_invalid(self):
        """Test peer ID validation for invalid format."""
        with pytest.raises(ValidationError, match="not appear to be a valid"):
            P2PChannel.create_dm("InvalidPeerID", "QmDst")


class TestInspectionMethods:
    """Test channel inspection methods."""

    def test_is_dm(self):
        """Test is_dm() method."""
        dm = P2PChannel.create_dm("QmSrc", "QmDst")
        ps = P2PChannel.create_pubsub("QmSrc", "topic")
        assert dm.is_dm()
        assert not ps.is_dm()

    def test_is_pubsub(self):
        """Test is_pubsub() method."""
        dm = P2PChannel.create_dm("QmSrc", "QmDst")
        ps = P2PChannel.create_pubsub("QmSrc", "topic")
        assert not dm.is_pubsub()
        assert ps.is_pubsub()

    def test_is_rendezvous(self):
        """Test is_rendezvous() method."""
        rv = P2PChannel.create_rendezvous("QmSrc")
        regular = P2PChannel.create_pubsub("QmSrc", "topic")
        assert rv.is_rendezvous()
        assert not regular.is_rendezvous()

    def test_get_destination_dm(self):
        """Test get_destination() for DM."""
        channel = P2PChannel.create_dm("QmSrc", "QmDst")
        assert channel.get_destination() == "QmDst"

    def test_get_destination_pubsub(self):
        """Test get_destination() for pub/sub."""
        channel = P2PChannel.create_pubsub("QmSrc", "my_topic")
        assert channel.get_destination() == "my_topic"


class TestImmutableUpdates:
    """Test immutable update methods."""

    def test_with_content_type(self):
        """Test creating new channel with different content type."""
        original = P2PChannel.create_dm("QmSrc", "QmDst", "json")
        updated = original.with_content_type("protobuf")

        assert original.content_type == "json"
        assert updated.content_type == "protobuf"
        assert original.to_string() == "QmSrc::dm:QmDst-json"
        assert updated.to_string() == "QmSrc::dm:QmDst-protobuf"

    def test_with_trail(self):
        """Test creating new channel with trail."""
        original = P2PChannel.create_dm("QmSrc", "QmDst", "json")
        updated = original.with_trail("stream1")

        assert original.channel_trail is None
        assert updated.channel_trail == "stream1"
        assert original.to_string() == "QmSrc::dm:QmDst-json"
        assert updated.to_string() == "QmSrc::dm:QmDst-json~stream1"


class TestCommonChannels:
    """Test CommonChannels helper class."""

    def test_rendezvous(self):
        """Test rendezvous helper."""
        channel_str = CommonChannels.rendezvous("QmPeer")
        assert channel_str == "QmPeer::ps:rv"

    def test_dm_json(self):
        """Test dm_json helper."""
        channel_str = CommonChannels.dm_json("QmSrc", "QmDst")
        assert channel_str == "QmSrc::dm:QmDst-json"

    def test_dm_json_with_trail(self):
        """Test dm_json helper with trail."""
        channel_str = CommonChannels.dm_json("QmSrc", "QmDst", trail="sub1")
        assert channel_str == "QmSrc::dm:QmDst-json~sub1"

    def test_dm_protobuf(self):
        """Test dm_protobuf helper."""
        channel_str = CommonChannels.dm_protobuf("QmSrc", "QmDst")
        assert channel_str == "QmSrc::dm:QmDst-protobuf"

    def test_pubsub_topic(self):
        """Test pubsub_topic helper."""
        channel_str = CommonChannels.pubsub_topic("QmSrc", "topic1", "json")
        assert channel_str == "QmSrc::ps:topic1-json"


class TestStringRepresentations:
    """Test string representations."""

    def test_str_method(self):
        """Test __str__() method."""
        channel = P2PChannel.create_dm("QmSrc", "QmDst", "json")
        assert str(channel) == "QmSrc::dm:QmDst-json"

    def test_repr_method_dm(self):
        """Test __repr__() method for DM."""
        channel = P2PChannel.create_dm("QmSourcePeer", "QmDestPeer", "json")
        repr_str = repr(channel)
        assert "P2PChannel" in repr_str
        assert "type=dm" in repr_str
        assert "content=json" in repr_str

    def test_repr_method_pubsub(self):
        """Test __repr__() method for pub/sub."""
        channel = P2PChannel.create_pubsub("QmSourcePeer", "topic1", "protobuf")
        repr_str = repr(channel)
        assert "P2PChannel" in repr_str
        assert "type=ps" in repr_str
        assert "topic=topic1" in repr_str


class TestRealWorldPatterns:
    """Test real-world usage patterns from codebase."""

    def test_connpool_send_pattern(self):
        """Test channel pattern used in ConnectionPools.send()."""
        # From connpool.py line 616-618
        peer_id = "QmDestPeer"
        source = "QmSourcePeer"
        content_type = "json"
        channel_trail = "stream1"

        # With trail
        channel = P2PChannel.create_dm(
            source_peer_id=source,
            destination_peer_id=peer_id,
            content_type=content_type,
            channel_trail=channel_trail
        )
        assert channel.to_string() == f"{source}::dm:{peer_id}-{content_type}~{channel_trail}"

        # Without trail
        channel_no_trail = P2PChannel.create_dm(
            source_peer_id=source,
            destination_peer_id=peer_id,
            content_type=content_type
        )
        assert channel_no_trail.to_string() == f"{source}::dm:{peer_id}-{content_type}"

    def test_rendezvous_pattern(self):
        """Test rendezvous pattern from node.py."""
        # From node.py line 1468, 2050
        peer_id = "QmPeerID"
        channel = P2PChannel.create_rendezvous(peer_id)
        assert channel.to_string() == f"{peer_id}::ps:rv"

    def test_world_updates_pattern(self):
        """Test world updates pub/sub pattern."""
        source = "QmWorldNode"
        topic = "world_updates"
        content_type = "json"

        channel = P2PChannel.create_pubsub(
            source_peer_id=source,
            topic_name=topic,
            content_type=content_type
        )
        assert channel.to_string() == f"{source}::ps:{topic}-{content_type}"


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_content_type(self):
        """Test channel with no content type."""
        channel = P2PChannel.create_dm("QmSrc", "QmDst")
        assert channel.content_type is None
        assert "-" not in channel.to_string()

    def test_empty_trail(self):
        """Test channel with no trail."""
        channel = P2PChannel.create_dm("QmSrc", "QmDst", "json")
        assert channel.channel_trail is None
        assert "~" not in channel.to_string()

    def test_topic_with_special_chars(self):
        """Test topic name with underscores and numbers."""
        channel = P2PChannel.create_pubsub("QmSrc", "world_123_updates")
        assert channel.topic_name == "world_123_updates"

    def test_roundtrip_conversion(self):
        """Test string -> object -> string conversion."""
        original_str = "QmSrc::dm:QmDst-protobuf~trail1"
        channel = P2PChannel.from_string(original_str)
        result_str = channel.to_string()
        assert original_str == result_str

    def test_multiple_hyphens_in_content(self):
        """Test handling of content type with hyphens."""
        # Content type should only be split on first hyphen
        channel_str = "QmSrc::ps:my-topic-name-json"
        channel = P2PChannel.from_string(channel_str)
        # Should split on first hyphen only
        assert channel.topic_name == "my"
        assert channel.content_type == "topic-name-json"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
