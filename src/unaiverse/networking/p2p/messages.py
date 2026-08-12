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
# Msg — self-contained value object + wire codec for the Node-bound chat path.
#
# Msg is a plain value object: the scalar
# fields (sender, content_type, channel, piggyback, timestamp_net) and the `content`
# are stored as native Python objects, never as a protobuf message. The wire codec
# lives entirely inside to_bytes / from_bytes, which import the generated protobuf
# (and the heavy stream_sample deps: torch, PIL, gzip, FileContainer) LAZILY. The
# module top stays stdlib-only so importing it is safe under Pyodide, where the
# protobuf runtime is mocked and message_pb2 cannot be imported at module level.
#
# In CPython the codec runs in the transport tier (P2P.get_messages / P2P.send). Under
# Pyodide the FSM/Coordinator only construct and read Msg value objects, never serialize:
# the JS twin (Messages.js) reads the value object over the proxy and does the codec. So
# do NOT hoist the lazy imports below to module top, and do not import anything that pulls
# protobuf/torch at module level.
import json
from typing import Any
from datetime import datetime, timezone


class Msg:

    # Message content types
    PROFILE = "profile"
    WORLD_APPROVAL = "world_approval"
    AGENT_APPROVAL = "agent_approval"
    PROFILE_REQUEST = "profile_request"
    ADDRESS_UPDATE = "address_update"
    STREAM_SAMPLE = "stream_sample"
    ROLE_SUGGESTION = "role_suggestion"
    INTERACTION = "interaction"
    INTERACTION_STATUS = "interaction_status"
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
                     STREAM_SAMPLE, ROLE_SUGGESTION, HSM, MISC, GET_CV_FROM_ROOT,
                     BADGE_SUGGESTIONS, INSPECT_ON, INSPECT_CMD, WORLD_AGENTS_LIST, CONSOLE_AND_BEHAV_STATUS,
                     STATS_UPDATE, STATS_REQUEST, STATS_RESPONSE, INTERACTION, INTERACTION_STATUS}

    # Sentinel for "no content set" (mirrors the unset protobuf oneof on the wire).
    _EMPTY = "<empty>"

    def __init__(self,
                 sender: str | None = None,
                 content: Any = None,
                 timestamp_net: str | None = None,
                 channel: str | None = None,
                 content_type: str = MISC,
                 piggyback: str | None = None):
        """Create a new message from native Python fields. The content is kept as-is
        (a dict for json types, a dict of samples for stream_sample, a list for
        stats_update); it is encoded to protobuf only in to_bytes. Use from_bytes to
        rebuild a Msg off the wire."""

        # Sanity checks
        assert sender is not None, "Sender must be specified for a new message."
        assert isinstance(sender, str), "Sender must be a string"
        assert timestamp_net is None or isinstance(timestamp_net, str), "Invalid timestamp_net"
        assert channel is None or isinstance(channel, str), "Invalid channel"
        assert content_type in Msg.CONTENT_TYPES, "Invalid content type"

        self._sender = sender
        self._timestamp_net = timestamp_net if timestamp_net is not None else \
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
        self._content_type = content_type if content_type is not None else self.MISC
        self._channel = channel if channel is not None else "<unknown>"
        self._piggyback = piggyback if piggyback is not None else ""
        # None or the sentinel mean "no content" (unset oneof on the wire).
        self._content = self._EMPTY if (content is None or content == self._EMPTY) else content

    def __str__(self):
        # Deliberately does NOT serialize (no to_bytes): __str__ runs on log paths and
        # rebuilding the protobuf just to report a byte length would be wasteful.
        return (f"Msg(sender={self.sender[:12]}..., content_type={self.content_type}, "
                f"channel='{self.channel}')")

    # --- Properties (plain field access; no protobuf, no decoding) ---
    @property
    def sender(self):
        return self._sender

    @sender.setter
    def sender(self, value: str):
        self._sender = value if value is not None else ""

    @property
    def content_type(self):
        return self._content_type

    @content_type.setter
    def content_type(self, value: str):
        self._content_type = value if value is not None else self.MISC

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value: str):
        self._channel = value if value is not None else "<unknown>"

    @property
    def piggyback(self):
        return self._piggyback

    @piggyback.setter
    def piggyback(self, value: str):
        self._piggyback = value if value is not None else ""

    @property
    def timestamp_net(self):
        return self._timestamp_net

    @timestamp_net.setter
    def timestamp_net(self, value: str):
        self._timestamp_net = value if value is not None else ""

    @property
    def content(self) -> Any:
        """The message content as a native Python object (dict / list / sentinel)."""
        return self._content

    # --- Serialization / Deserialization (the ONLY place protobuf is touched) ---
    def to_bytes(self) -> bytes:
        """Encode the value object into the protobuf wire format. protobuf (and the
        heavy stream_sample deps) are imported lazily here so the module top stays
        stdlib-only and importable under Pyodide."""
        from . import message_pb2 as pb  # lazy: CPython conn layer only, never under Pyodide

        proto = pb.Message()
        proto.sender = self._sender if self._sender is not None else ""
        proto.timestamp_net = self._timestamp_net if self._timestamp_net is not None else ""
        proto.content_type = self._content_type if self._content_type is not None else self.MISC
        proto.channel = self._channel if self._channel is not None else "<unknown>"
        proto.piggyback = self._piggyback if self._piggyback is not None else ""

        content = self._content
        if content is not None and content != self._EMPTY:
            # Route the content to the correct encoder.
            if self._content_type == Msg.STREAM_SAMPLE:
                self._encode_stream_sample(proto, content)
            elif self._content_type == Msg.STATS_UPDATE:
                self._encode_stats_update(proto, content)
            else:
                # All other structured types use the generic json_content field
                proto.json_content = json.dumps(content)

        return proto.SerializeToString()

    @classmethod
    def from_bytes(cls, msg_bytes: bytes) -> 'Msg':
        """Decode a byte array into a new Msg. Builds the value object directly via
        __new__ (no field validation): a frame off the wire is taken as-is, exactly
        like the previous protobuf-backed path. protobuf is imported lazily."""
        from . import message_pb2 as pb  # lazy: CPython conn layer only, never under Pyodide

        proto = pb.Message()
        proto.ParseFromString(msg_bytes)

        obj = cls.__new__(cls)
        obj._sender = proto.sender
        obj._timestamp_net = proto.timestamp_net
        obj._content_type = proto.content_type
        obj._channel = proto.channel
        obj._piggyback = proto.piggyback

        payload_type = proto.WhichOneof("content")
        if payload_type == "stream_sample":
            obj._content = cls._decode_stream_sample(proto)
        elif payload_type == "stats_update":
            obj._content = cls._decode_stats_update(proto)
        elif payload_type == "json_content":
            obj._content = json.loads(proto.json_content)
        else:
            obj._content = cls._EMPTY
        return obj

    # --- Internal codec helpers (run only inside to_bytes / from_bytes) ---
    @staticmethod
    def _encode_stream_sample(proto, samples_dict: dict) -> None:
        """Builds the complex StreamSampleContent message from a dict of samples."""
        import io
        import gzip
        import torch
        from PIL import Image
        from unaiverse.streams.dataprops import FileContainer

        content_pb = proto.stream_sample
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

    @staticmethod
    def _decode_stream_sample(proto) -> dict:
        """Parses the StreamSampleContent message back into a Python dict of tensors,
        images, strings and file containers."""
        import io
        import gzip
        import torch
        from PIL import Image
        from unaiverse.streams.dataprops import FileContainer

        py_dict = {}

        # Iterate through the Protobuf map ('samples')
        for name, sample_pb in proto.stream_sample.samples.items():
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

    @staticmethod
    def _py_value_to_proto_value(py_val: Any):
        """Helper to convert a Python type into a google.protobuf.Value."""
        from google.protobuf.struct_pb2 import Value, ListValue, NULL_VALUE
        from google.protobuf.json_format import ParseDict

        if py_val is None:
            return Value(null_value=NULL_VALUE)
        if isinstance(py_val, bool):
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
            # This is recursive for dicts/structs
            s = Value(struct_value={})
            ParseDict(py_val, s.struct_value)
            return s

        # Fallback
        return Value(string_value=str(py_val))

    @staticmethod
    def _encode_stats_update(proto, payload_list: list) -> None:
        """Builds the StatBatch message from a List[Dict]."""
        batch_pb = proto.stats_update

        for update_dict in payload_list:
            update_pb = batch_pb.updates.add()
            update_pb.peer_id = update_dict['group_key']
            update_pb.stat_name = update_dict['stat_name']
            update_pb.timestamp = int(update_dict['timestamp'])  # Ensure int

            # Convert the Python 'value' to a Protobuf 'Value'
            py_value = update_dict['value']
            update_pb.value.CopyFrom(Msg._py_value_to_proto_value(py_value))

    @staticmethod
    def _proto_value_to_py_value(proto_val) -> Any:
        """Helper to convert a google.protobuf.Value into a Python type."""
        from google.protobuf.json_format import MessageToDict

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
            return [Msg._proto_value_to_py_value(v) for v in proto_val.list_value.values]
        if kind == "struct_value":
            # Use the helper to convert a Struct to a dict
            return MessageToDict(proto_val.struct_value)
        return None

    @staticmethod
    def _decode_stats_update(proto) -> list:
        """Parses the StatBatch message back into a List[Dict]."""
        py_list = []
        batch_pb = proto.stats_update

        for update_pb in batch_pb.updates:
            py_list.append({
                "group_key": update_pb.peer_id,
                "stat_name": update_pb.stat_name,
                "timestamp": update_pb.timestamp,
                "value": Msg._proto_value_to_py_value(update_pb.value)
            })
        return py_list
