"""P2P Message Module - UNaIVERSE Networking.

This module provides the Msg class for creating, validating, and serializing P2P messages
with Protocol Buffer support. Messages are used for all communication in the UNaIVERSE
networking layer.

Message Types (ContentTypes):
    - PROFILE: Share an agent's profile
    - WORLD_APPROVAL: World approves agent's join request
    - AGENT_APPROVAL: Agent approves another agent's join request
    - PROFILE_REQUEST: Request an agent's profile
    - ADDRESS_UPDATE: Share updated network addresses
    - STREAM_SAMPLE: Share data stream samples (observations, tensors, images)
    - ACTION_REQUEST: Request an action from another agent
    - ROLE_SUGGESTION: Suggest a role for another agent
    - HSM: Share Hybrid State Machine definition
    - MISC: Generic message type
    - And more... (see ContentTypes enum for complete list)

The Msg class uses Pydantic for validation and type safety while maintaining full
Protocol Buffer compatibility for efficient network serialization.

Examples:
    Create a simple message:
    ```python
    msg = Msg(
        sender="QmABC123",
        content={"action": "ping"},
        channel="QmABC::dm:QmXYZ",
        content_type=ContentTypess.MISC
    )
    ```

    Serialize and deserialize:
    ```python
    msg_bytes = msg.model_dump(mode='json')  # To bytes
    msg = Msg.model_validate(msg_bytes)      # From bytes
    content = msg.get_content()              # Access content
    ```

    Create with convenience methods:
    ```python
    msg = Msg.create_simple(
        sender="QmABC",
        content={"key": "value"},
        channel="QmABC::dm:QmXYZ"
    )
    ```

A Collectionless AI Project (https://collectionless.ai) / UNaIVERSE SRL (https://unaiverse.ai)

- Registration/Login: https://unaiverse.io
- Code Repositories: https://github.com/collectionlessai/
- Main Developers: Stefano Melacci (Project Leader), Christian Di Maio, Tommaso Guidi
"""

import gzip
import io
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Union

import torch
from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.struct_pb2 import NULL_VALUE, ListValue, Value
from PIL import Image
from pydantic import (
    BaseModel,
    PrivateAttr,
    model_serializer,
    model_validator,
)
from typing_extensions import Self

from unaiverse.dataprops import FileContainer
from unaiverse.networking.channel import Channel

# Import the Protobuf-generated module
try:
    from unaiverse.networking.p2p import message_pb2 as pb
except ImportError:
    try:
        from unaiverse.networking.p2p import message_pb2 as pb
    except ImportError:
        print("Error: message_pb2.py not found. Please compile the .proto file first.")
        raise


class ContentTypes(str, Enum):
    """All possible types of message which UNaIVERSE supports.

    Attributes:
        PROFILE: Used to share an agent's profile.
        WORLD_APPROVAL: Used by the world to approve an agent's request to join.
        AGENT_APPROVAL: Used by an agent to approve another agent's request to join.
        PROFILE_REQUEST: Used to request an agent's profile.
        ADDRESS_UPDATE: Used to share updated network addresses.
        STREAM_SAMPLE: Used to share a sample of a data stream (e.g. an observation).
        ACTION_REQUEST: Used by an agent to request an action from another agent.
        ROLE_SUGGESTION: Used by an agent to suggest a role for another agent.
        HSM: Used to share a Hybrid State Machine (HSM) definition.
        MISC: A generic type for messages that don't fit into the other categories.
        GET_CV_FROM_ROOT: Used to request a CV from the root node.
        BADGE_SUGGESTIONS: Used to share badge suggestions for an agent.
        INSPECT_ON: Used to request inspection of an agent or world.
        INSPECT_CMD: Used to send a command to an inspector.
        WORLD_AGENTS_LIST: Used to share the list of agents in a world.
        CONSOLE_AND_BEHAV_STATUS: Used to share console output and behavior status.
        STATS_UPDATE: Used by an agent to send a batch of stats updates to the world.
        STATS_REQUEST: Used by an agent to request stats from the world.
        STATS_RESPONSE: Used by the world to respond to a stats request with the requested data.
    """

    PROFILE = "profile"
    WORLD_APPROVAL = "world_approval"
    AGENT_APPROVAL = "agent_approval"
    PROFILE_REQUEST = "profile_request"
    ADDRESS_UPDATE = "address_update"
    STREAM_SAMPLE = "stream_sample"
    ACTION_REQUEST = "action_request"
    ROLE_SUGGESTION = "role_suggestion"
    HSM = "hsm"
    MISC = "misc"
    GET_CV_FROM_ROOT = "get_cv_from_root"
    BADGE_SUGGESTIONS = "badge_suggestions"
    INSPECT_ON = "inspect_on"
    INSPECT_CMD = "inspect_cmd"
    WORLD_AGENTS_LIST = "world_agents_list"
    CONSOLE_AND_BEHAV_STATUS = "console_and_behav_status"
    STATS_UPDATE = "stats_update"
    STATS_REQUEST = "stats_request"
    STATS_RESPONSE = "stats_response"


class Msg(BaseModel):
    """UNaIVERSE Message container with Protobuf serialization support.

    Three main content types are supported:
        - STREAM_SAMPLE: Complex data structures (tensors, images, files)
        - STATS_UPDATE: Batched statistics updates
        - Others: Generic JSON-serializable content

    Examples:
        Create a simple message:
        ```python
        msg = Msg(
            sender="QmABC123",
            content={"key": "value"},
            content_type=ContentTypes.MISC,
            channel=Channel.direct("QmABC123", "QmXYZ456")
        )
        ```

        Create from bytes:
        ```python
        msg = Msg.model_validate(raw_bytes)
        ```

        Serialize to bytes:
        ```python
        raw_bytes = msg.model_dump(mode='json')
        # or
        raw_bytes = msg.to_bytes()
        ```
    """

    # Internal state - protobuf message is the single source of truth
    _proto_msg: pb.Message = PrivateAttr(default=None)
    _decoded_content: Optional[Any] = PrivateAttr(default=None)

    model_config = {
        "arbitrary_types_allowed": True,  # Allow torch.Tensor, PIL.Image, etc.
        "validate_assignment": True,
        "validate_default": False,  # Don't validate None for _proto_msg
    }

    @model_validator(mode="wrap")
    @classmethod
    def validate_from_bytes_or_dict(cls, value: Any, handler, info) -> Self:
        """Validates and constructs from either bytes, dict, or protobuf message.

        Handles three input modes:
        1. bytes: Deserialize protobuf message
        2. dict with '_proto_msg': Use existing protobuf message
        3. dict: Standard Pydantic validation + protobuf construction

        Args:
            value: Input value (bytes, dict, or other)
            handler: Default validation handler
            info: Validation context

        Returns:
            Validated Msg instance
        """
        # Mode 1: Deserialize from bytes
        if isinstance(value, bytes):
            pb_msg = pb.Message()
            pb_msg.ParseFromString(value)

            # Create instance with empty dict (Pydantic won't use fields)
            instance = handler({})
            instance._proto_msg = pb_msg
            return instance

        # Mode 2: Using existing protobuf message (from deserialization)
        if isinstance(value, dict) and "_proto_msg" in value:
            proto_msg = value.get("_proto_msg")
            if proto_msg is not None:
                # Create instance with empty dict
                instance = handler({})
                instance._proto_msg = proto_msg
                return instance

        # Mode 3: Standard dict - validate and build protobuf
        # Extract fields for validation
        sender = value.get("sender")    
        content = value.get("content")
        timestamp_net = value.get("timestamp_net")
        channel = value.get("channel", "<unknown>")
        content_type = value.get("content_type", ContentTypes.MISC)
        piggyback = value.get("piggyback", "")

        # Validate required fields
        if sender is None:
            raise ValueError("sender is required")
        if not isinstance(sender, str) or len(sender) < 1:
            raise ValueError("sender must be a non-empty string")

        # Validate content_type
        if isinstance(content_type, str):
            try:
                content_type = ContentTypes(content_type)
            except ValueError:
                raise ValueError(f"Invalid content_type: {content_type}")

        # Build protobuf message
        pb_msg = pb.Message()
        pb_msg.sender = sender
        pb_msg.timestamp_net = (
            timestamp_net
            if timestamp_net is not None
            else datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
        )
        pb_msg.content_type = content_type.value
        pb_msg.channel = channel
        pb_msg.piggyback = piggyback

        # Build content based on type
        if content is not None and content != "<empty>":
            if content_type == ContentTypes.STREAM_SAMPLE:
                cls._build_stream_sample_content_static(pb_msg, content)
            elif content_type == ContentTypes.STATS_UPDATE:
                cls._build_stats_update_content_static(pb_msg, content)
            else:
                cls._build_json_content_static(pb_msg, content)

        # Create instance with empty dict and set _proto_msg
        instance = handler({})
        instance._proto_msg = pb_msg
        return instance

    # --- Properties that route to protobuf message (single source of truth) ---

    @property
    def sender(self) -> str:
        """Get sender from protobuf message."""
        return self._proto_msg.sender if self._proto_msg else ""

    @sender.setter
    def sender(self, value: str):
        """Set sender in protobuf message."""
        if not self._proto_msg:
            self._proto_msg = pb.Message()
        self._proto_msg.sender = value if value is not None else ""

    @property
    def content_type(self) -> ContentTypes:
        """Get content_type from protobuf message."""
        if not self._proto_msg:
            return ContentTypes.MISC
        return ContentTypes(self._proto_msg.content_type)

    @content_type.setter
    def content_type(self, value: Union[ContentTypes, str]):
        """Set content_type in protobuf message."""
        if not self._proto_msg:
            self._proto_msg = pb.Message()
        if isinstance(value, ContentTypes):
            self._proto_msg.content_type = value.value
        else:
            self._proto_msg.content_type = (
                value if value is not None else ContentTypes.MISC.value
            )

    @property
    def channel(self) -> Channel:
        """Get channel from protobuf message."""
        return Channel.model_validate(self._proto_msg.channel) if self._proto_msg else "<unknown>"

    @channel.setter
    def channel(self, value: str):
        """Set channel in protobuf message."""
        if not self._proto_msg:
            self._proto_msg = pb.Message()
        self._proto_msg.channel = value if value is not None else "<unknown>"

    @property
    def piggyback(self) -> str:
        """Get piggyback from protobuf message."""
        return self._proto_msg.piggyback if self._proto_msg else ""

    @piggyback.setter
    def piggyback(self, value: str):
        """Set piggyback in protobuf message."""
        if not self._proto_msg:
            self._proto_msg = pb.Message()
        self._proto_msg.piggyback = value if value is not None else ""

    @property
    def timestamp_net(self) -> str:
        """Get timestamp_net from protobuf message."""
        return self._proto_msg.timestamp_net if self._proto_msg else ""

    @timestamp_net.setter
    def timestamp_net(self, value: str):
        """Set timestamp_net in protobuf message."""
        if not self._proto_msg:
            self._proto_msg = pb.Message()
        self._proto_msg.timestamp_net = value if value is not None else ""

    def to_bytes(self) -> bytes:
        """Serializes the message to Protocol Buffer bytes.

        Returns:
            Serialized protobuf message as bytes
        """
        if self._proto_msg is None:
            raise ValueError("Cannot serialize message without protobuf data")
        return self._proto_msg.SerializeToString()

    @model_serializer(mode="plain", when_used="json")
    def serialize_to_bytes(self) -> bytes:
        """Serializes the message to Protocol Buffer bytes for JSON mode.

        Returns:
            Serialized protobuf message as bytes
        """
        return self.to_bytes()

    def get_content(self) -> Any:
        """Lazy-loads and returns the decoded message content.

        The content is cached after first access for performance.

        Returns:
            Decoded message content based on content_type
        """
        if self._decoded_content is not None:
            return self._decoded_content

        if self._proto_msg is None:
            return None

        payload_type = self._proto_msg.WhichOneof("content")
        if payload_type == "stream_sample":
            self._decoded_content = self._parse_stream_sample_content()
        elif payload_type == "stats_update":
            self._decoded_content = self._parse_stats_update_content()
        elif payload_type == "json_content":
            self._decoded_content = self._parse_json_content()
        else:
            self._decoded_content = "<empty>"

        return self._decoded_content

    # --- Static Helper Methods for Protobuf Content Building ---

    @staticmethod
    def _build_json_content_static(proto_msg: pb.Message, content: dict):
        """Populates the generic json_content field."""
        proto_msg.json_content = json.dumps(content)

    def _parse_json_content(self) -> dict:
        """Parses the generic json_content field back to a dict."""
        return json.loads(self._proto_msg.json_content)

    @staticmethod
    def _build_stream_sample_content_static(proto_msg: pb.Message, samples_dict: dict):
        """Builds the complex StreamSampleContent message from a dict."""
        content_pb = proto_msg.stream_sample
        for name, sample_info in samples_dict.items():
            data = sample_info.get("data")
            if data is None:
                continue

            stream_sample_pb = content_pb.samples[name]
            stream_sample_pb.data_tag = sample_info.get("data_tag", -1)
            uuid = sample_info.get("data_uuid")

            if uuid is not None:
                stream_sample_pb.data_uuid = uuid

            if isinstance(data, torch.Tensor):
                raw_bytes = data.detach().cpu().numpy().tobytes()
                with io.BytesIO() as buffer:
                    with gzip.GzipFile(fileobj=buffer, mode="wb") as f:
                        f.write(raw_bytes)
                    stream_sample_pb.data.tensor_data.data = buffer.getvalue()
                stream_sample_pb.data.tensor_data.dtype = str(data.dtype).split(".")[-1]
                stream_sample_pb.data.tensor_data.shape.extend(list(data.shape))

            elif isinstance(data, Image.Image):
                with io.BytesIO() as buffer:
                    data.save(buffer, format="PNG", optimize=True, compress_level=9)
                    stream_sample_pb.data.image_data.data = buffer.getvalue()

            elif isinstance(data, str):
                stream_sample_pb.data.text_data.data = data

            elif isinstance(data, FileContainer):
                raw_bytes = (
                    data.content.encode("utf-8")
                    if isinstance(data.content, str)
                    else data.content
                )
                stream_sample_pb.data.file_data.content = raw_bytes
                stream_sample_pb.data.file_data.filename = data.filename
                stream_sample_pb.data.file_data.mime_type = data.mime_type

    def _parse_stream_sample_content(self) -> dict:
        """Parses the internal StreamSampleContent message back into a Python dictionary."""
        py_dict = {}

        for name, sample_pb in self._proto_msg.stream_sample.samples.items():
            data_payload = sample_pb.data
            data = None

            payload_type = data_payload.WhichOneof("data_payload")

            if payload_type == "tensor_data":
                tensor_data = data_payload.tensor_data
                with gzip.GzipFile(
                    fileobj=io.BytesIO(tensor_data.data), mode="rb"
                ) as f:
                    raw_bytes = f.read()
                data = torch.frombuffer(
                    bytearray(raw_bytes), dtype=getattr(torch, tensor_data.dtype)
                ).reshape(list(tensor_data.shape))

            elif payload_type == "image_data":
                data = Image.open(io.BytesIO(data_payload.image_data.data))

            elif payload_type == "text_data":
                data = data_payload.text_data.data

            elif payload_type == "file_data":
                f_data = data_payload.file_data
                data = FileContainer(
                    content=f_data.content,
                    filename=f_data.filename,
                    mime_type=f_data.mime_type,
                )

            py_dict[name] = {
                "data": data,
                "data_tag": sample_pb.data_tag,
                "data_uuid": sample_pb.data_uuid
                if sample_pb.HasField("data_uuid")
                else None,
            }
        return py_dict

    @staticmethod
    def _py_value_to_proto_value(py_val: Any) -> Value:
        """Helper to convert a Python type into a google.protobuf.Value."""
        if py_val is None:
            return Value(null_value=NULL_VALUE)
        if isinstance(py_val, bool):  # Check bool before int/float
            return Value(bool_value=py_val)
        if isinstance(py_val, (int, float)):
            return Value(number_value=py_val)
        if isinstance(py_val, str):
            return Value(string_value=py_val)
        if isinstance(py_val, list):
            lv = ListValue()
            for item in py_val:
                lv.values.append(Msg._py_value_to_proto_value(item))
            return Value(list_value=lv)
        if isinstance(py_val, dict):
            s = Value(struct_value={})
            ParseDict(py_val, s.struct_value)
            return s

        # Fallback
        return Value(string_value=str(py_val))

    @staticmethod
    def _build_stats_update_content_static(proto_msg: pb.Message, payload_list: list):
        """Builds the StatBatch message from a List[Dict]."""
        batch_pb = proto_msg.stats_update

        for update_dict in payload_list:
            update_pb = batch_pb.updates.add()
            update_pb.peer_id = update_dict["peer_id"]
            update_pb.stat_name = update_dict["stat_name"]
            update_pb.timestamp = int(update_dict["timestamp"])

            py_value = update_dict["value"]
            update_pb.value.CopyFrom(Msg._py_value_to_proto_value(py_value))

    def _proto_value_to_py_value(self, proto_val: Value) -> Any:
        """Helper to convert a google.protobuf.Value into a Python type."""
        kind = proto_val.WhichOneof("kind")
        if kind == "null_value":
            return None
        if kind == "number_value":
            return proto_val.number_value
        if kind == "string_value":
            return proto_val.string_value
        if kind == "bool_value":
            return proto_val.bool_value
        if kind == "list_value":
            return [
                self._proto_value_to_py_value(v) for v in proto_val.list_value.values
            ]
        if kind == "struct_value":
            return MessageToDict(proto_val.struct_value)
        return None

    def _parse_stats_update_content(self) -> list:
        """Parses the StatBatch message back into a List[Dict]."""
        py_list = []
        batch_pb = self._proto_msg.stats_update

        for update_pb in batch_pb.updates:
            py_list.append({
                "peer_id": update_pb.peer_id,
                "stat_name": update_pb.stat_name,
                "timestamp": update_pb.timestamp,
                "value": self._proto_value_to_py_value(update_pb.value),
            })
        return py_list

    # --- Convenience Methods ---

    @classmethod
    def create_simple(
        cls,
        sender: str,
        content: dict,
        channel: str,
        content_type: ContentTypes = ContentTypes.MISC,
        piggyback: str = "",
    ) -> "Msg":
        """Convenience method to create a simple message with JSON content.

        Args:
            sender: Peer ID of sender
            content: Dictionary content
            channel: Channel identifier
            content_type: Type of content (default: MISC)
            piggyback: Optional piggyback data

        Returns:
            New Msg instance

        Example:
            >>> msg = Msg.create_simple(
            ...     sender="QmABC",
            ...     content={"action": "ping"},
            ...     channel="QmABC::dm:QmXYZ"
            ... )
        """
        return cls.model_validate({
            "sender": sender,
            "content": content,
            "channel": channel,
            "content_type": content_type,
            "piggyback": piggyback,
        })

    @classmethod
    def create_stream_sample(
        cls, sender: str, samples: dict, channel: str, piggyback: str = ""
    ) -> "Msg":
        """Convenience method to create a stream sample message.

        Args:
            sender: Peer ID of sender
            samples: Dictionary of sample data (tensors, images, etc.)
            channel: Channel identifier
            piggyback: Optional piggyback data

        Returns:
            New Msg instance

        Example:
            >>> import torch
            >>> msg = Msg.create_stream_sample(
            ...     sender="QmABC",
            ...     samples={"obs": {
            ...         "data": torch.randn(3, 32, 32),
            ...         "data_tag": 1,
            ...         "data_uuid": "uuid-123"
            ...     }},
            ...     channel="QmABC::ps:observations"
            ... )
        """
        return cls.model_validate({
            "sender": sender,
            "content": samples,
            "channel": channel,
            "content_type": ContentTypes.STREAM_SAMPLE,
            "piggyback": piggyback,
        })

    @classmethod
    def create_stats_update(
        cls, sender: str, updates: list, channel: str, piggyback: str = ""
    ) -> "Msg":
        """Convenience method to create a stats update message.

        Args:
            sender: Peer ID of sender
            updates: List of stat update dictionaries
            channel: Channel identifier
            piggyback: Optional piggyback data

        Returns:
            New Msg instance

        Example:
            >>> msg = Msg.create_stats_update(
            ...     sender="QmABC",
            ...     updates=[{
            ...         "peer_id": "QmABC",
            ...         "stat_name": "cpu_usage",
            ...         "timestamp": 1234567890,
            ...         "value": 45.2
            ...     }],
            ...     channel="QmABC::ps:stats"
            ... )
        """
        return cls.model_validate({
            "sender": sender,
            "content": updates,
            "channel": channel,
            "content_type": ContentTypes.STATS_UPDATE,
            "piggyback": piggyback,
        })

    def __str__(self) -> str:
        """Returns a concise string representation."""
        byte_len = len(self.model_dump(mode="json")) if self._proto_msg else 0
        return (
            f"Msg(sender={self.sender[:12]}..., content_type={self.content_type.value}, "
            f"channel='{self.channel}', content_len={byte_len} bytes)"
        )

    def __repr__(self) -> str:
        """Returns a detailed representation."""
        return (
            f"Msg(sender='{self.sender}', content_type={self.content_type.value}, "
            f"channel='{self.channel}', timestamp='{self.timestamp_net}')"
        )
