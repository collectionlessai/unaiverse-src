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
import io
import gzip
import json
import torch
from PIL import Image
from typing import Any
from datetime import datetime, timezone
from unaiverse.dataprops import FileContainer
from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.struct_pb2 import Value, ListValue, NULL_VALUE

# Import the Protobuf-generated module
try:
    from . import message_pb2 as pb
except ImportError:
    print("Error: message_pb2.py not found. Please compile the .proto file first.")
    raise


class Msg:

    # Message content types
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
    STATS_UPDATE = "stats_update"  # agent -> world
    STATS_REQUEST = "stats_request"
    STATS_RESPONSE = 'stats_response'  # world -> agent

    # Collections
    CONTENT_TYPES = {PROFILE, WORLD_APPROVAL, AGENT_APPROVAL, PROFILE_REQUEST, ADDRESS_UPDATE,
                     STREAM_SAMPLE, ACTION_REQUEST, ROLE_SUGGESTION, HSM, MISC, GET_CV_FROM_ROOT,
                     BADGE_SUGGESTIONS, INSPECT_ON, INSPECT_CMD, WORLD_AGENTS_LIST, CONSOLE_AND_BEHAV_STATUS,
                     STATS_UPDATE, STATS_REQUEST, STATS_RESPONSE}

    def __init__(self,
                 sender: str | None = None,
                 content: any = None,
                 timestamp_net: str | None = None,
                 channel: str | None = None,
                 content_type: str = MISC,
                 piggyback: str | None = None,
                 _proto_msg: pb.Message = None):
        """The constructor should be used either to create a new message filling the fields,
        or to parse an existing Protobuf message (passing _proto_msg). In the latter case,
        the other fields are ignored and the Protobuf message is used as-is. The message is
        simply stored in the internal `_proto_msg` field and other fields can be accessed
        through properties."""

        self._decoded_content: any = None  # Cache for decompressed content

        if _proto_msg is not None:

            # Check if any other arguments were simultaneously provided
            other_args = [sender, content, timestamp_net, channel, piggyback]
            if any(arg is not None for arg in other_args):
                raise ValueError("Cannot specify other arguments when creating a Msg from a _proto_msg.")

            # This path is used by from_bytes, message is already built
            self._proto_msg = _proto_msg
            return

        # Sanity checks
        assert sender is not None, "Sender must be specified for a new message."
        assert isinstance(sender, str), "Sender must be a string"
        assert timestamp_net is None or isinstance(timestamp_net, str), "Invalid timestamp_net"
        assert channel is None or isinstance(channel, str), "Invalid channel"
        assert content_type in Msg.CONTENT_TYPES, "Invalid content type"

        # --- SMART CONSTRUCTOR: Populates the correct 'oneof' field ---
        self._proto_msg = pb.Message()
        self._proto_msg.sender = sender if sender is not None else ""
        self._proto_msg.timestamp_net = timestamp_net if timestamp_net is not None else \
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
        self._proto_msg.content_type = content_type if content_type is not None else self.MISC
        self._proto_msg.channel = channel if channel is not None else "<unknown>"
        self._proto_msg.piggyback = piggyback if piggyback is not None else ""

        if content is None or content == "<empty>":
            return  # Nothing to set in the 'oneof'

        # Route the content to the correct builder
        if content_type == Msg.STREAM_SAMPLE:
            self._build_stream_sample_content(content)
        elif content_type == Msg.STATS_UPDATE:
            self._build_stats_update_content(content)
        else:
            # All other structured types use the generic json_content field
            self._build_json_content(content)

    def __str__(self):
        return (f"Msg(sender={self.sender[:12]}..., content_type={self.content_type}, "
                f"channel='{self.channel}', content_len={len(self.to_bytes())} bytes)")

    # --- Properties for lazy-loading and easy access ---
    @property
    def sender(self):
        return self._proto_msg.sender

    @sender.setter
    def sender(self, value: str):
        self._proto_msg.sender = value if value is not None else ""

    @property
    def content_type(self):
        return self._proto_msg.content_type

    @content_type.setter
    def content_type(self, value: str):
        self._proto_msg.content_type = value if value is not None else self.MISC

    @property
    def channel(self):
        return self._proto_msg.channel

    @channel.setter
    def channel(self, value: str):
        self._proto_msg.channel = value if value is not None else "<unknown>"

    @property
    def piggyback(self):
        return self._proto_msg.piggyback

    @piggyback.setter
    def piggyback(self, value: str):
        self._proto_msg.piggyback = value if value is not None else ""

    @property
    def timestamp_net(self):
        return self._proto_msg.timestamp_net

    @timestamp_net.setter
    def timestamp_net(self, value: str):
        self._proto_msg.timestamp_net = value if value is not None else ""

    @property
    def content(self) -> any:
        """The main content of the message, decoded on-the-fly with caching."""
        if self._decoded_content is not None:
            return self._decoded_content

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

    # --- Serialization / Deserialization ---
    def to_bytes(self) -> bytes:
        """Serializes the internal Protobuf message to bytes."""
        return self._proto_msg.SerializeToString()

    @classmethod
    def from_bytes(cls, msg_bytes: bytes) -> 'Msg':
        """Deserializes a byte array into a new Msg instance."""
        pb_msg = pb.Message()
        pb_msg.ParseFromString(msg_bytes)

        # Pass the parsed protobuf message to the constructor
        return cls(_proto_msg=pb_msg)

    # --- Internal Helper Methods ---
    def _build_json_content(self, content: dict):
        """Populates the generic json_content field."""
        self._proto_msg.json_content = json.dumps(content)

    def _parse_json_content(self) -> dict:
        """Parses the generic json_content field back to a dict."""
        return json.loads(self._proto_msg.json_content)

    def _build_stream_sample_content(self, samples_dict: dict):
        """Builds the complex StreamSampleContent message from a dict."""
        content_pb = self._proto_msg.stream_sample
        for name, sample_info in samples_dict.items():
            data = sample_info.get('data')
            if data is None:
                continue

            stream_sample_pb = content_pb.samples[name]
            stream_sample_pb.data_tag = sample_info.get('data_tag', -1)
            uuid = sample_info.get('data_uuid')

            # Only set the field if the uuid is not None
            if uuid is not None:
                stream_sample_pb.data_uuid = uuid

            if isinstance(data, torch.Tensor):
                raw_bytes = data.detach().cpu().numpy().tobytes()
                with io.BytesIO() as buffer:
                    with gzip.GzipFile(fileobj=buffer, mode='wb') as f:
                        f.write(raw_bytes)
                    stream_sample_pb.data.tensor_data.data = buffer.getvalue()
                stream_sample_pb.data.tensor_data.dtype = str(data.dtype).split('.')[-1]
                stream_sample_pb.data.tensor_data.shape.extend(list(data.shape))

            elif isinstance(data, Image.Image):
                with io.BytesIO() as buffer:
                    data.save(buffer, format="PNG", optimize=True, compress_level=9)
                    stream_sample_pb.data.image_data.data = buffer.getvalue()

            elif isinstance(data, str):
                stream_sample_pb.data.text_data.data = data
            
            elif isinstance(data, FileContainer):
                # Auto-convert string content to bytes if needed
                raw_bytes = data.content.encode('utf-8') if isinstance(data.content, str) else data.content
                stream_sample_pb.data.file_data.content = raw_bytes
                stream_sample_pb.data.file_data.filename = data.filename
                stream_sample_pb.data.file_data.mime_type = data.mime_type

    def _parse_stream_sample_content(self) -> dict:
        """
        Parses the internal StreamSampleContent message back into a Python
        dictionary of tensors and images.
        """
        py_dict = {}

        # Iterate through the Protobuf map ('samples')
        for name, sample_pb in self._proto_msg.stream_sample.samples.items():
            data_payload = sample_pb.data
            data = None

            # Check which field in the 'oneof' is set
            payload_type = data_payload.WhichOneof("data_payload")

            if payload_type == "tensor_data":
                tensor_data = data_payload.tensor_data

                # Decompress and reconstruct the tensor
                with gzip.GzipFile(fileobj=io.BytesIO(tensor_data.data), mode='rb') as f:
                    raw_bytes = f.read()
                data = torch.frombuffer(
                    bytearray(raw_bytes),
                    dtype=getattr(torch, tensor_data.dtype)
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
                    mime_type=f_data.mime_type
                )

            # Build the final Python dictionary for this sample
            py_dict[name] = {
                'data': data,
                'data_tag': sample_pb.data_tag,
                'data_uuid': sample_pb.data_uuid if sample_pb.HasField("data_uuid") else None
            }
        return py_dict
    
    def _py_value_to_proto_value(self, py_val: Any) -> Value:
        """Helper to convert a Python type into a google.protobuf.Value."""
        if py_val is None:
            return Value(null_value=NULL_VALUE)
        if isinstance(py_val, (int, float)):
            return Value(number_value=py_val)
        if isinstance(py_val, str):
            return Value(string_value=py_val)
        if isinstance(py_val, bool):
            return Value(bool_value=py_val)
        if isinstance(py_val, list):
            lv = ListValue()
            for item in py_val:
                lv.values.append(self._py_value_to_proto_value(item))
            return Value(list_value=lv)
        if isinstance(py_val, dict):
            # This is recursive for dicts/structs
            s = Value(struct_value={})
            ParseDict(py_val, s.struct_value)
            return s
        
        # Fallback
        return Value(string_value=str(py_val))

    def _build_stats_update_content(self, payload_list: list):
        """Builds the StatBatch message from a List[Dict]."""
        batch_pb = self._proto_msg.stats_update
        
        for update_dict in payload_list:
            update_pb = batch_pb.updates.add()
            update_pb.peer_id = update_dict['peer_id']
            update_pb.stat_name = update_dict['stat_name']
            update_pb.timestamp = int(update_dict['timestamp'])  # Ensure int
            
            # Convert the Python 'value' to a Protobuf 'Value'
            py_value = update_dict['value']
            update_pb.value.CopyFrom(self._py_value_to_proto_value(py_value))

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
            return [self._proto_value_to_py_value(v) for v in proto_val.list_value.values]
        if kind == "struct_value":
            # Use the helper to convert a Struct to a dict
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
                "value": self._proto_value_to_py_value(update_pb.value)
            })
        return py_list
