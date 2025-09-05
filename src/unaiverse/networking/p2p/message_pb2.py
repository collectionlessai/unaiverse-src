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
from google.protobuf import descriptor as _descriptor
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    6,
    31,
    1,
    '',
    'message.proto'
)

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rmessage.proto\x12\x03p2p\"\xc0\x01\n\x07Message\x12\x0e\n\x06sender\x18\x01 \x01(\t\x12\x14\n\x0c\x63ontent_type\x18\x02 \x01(\t\x12\x0f\n\x07\x63hannel\x18\x03 \x01(\t\x12\x11\n\tpiggyback\x18\x04 \x01(\t\x12\x15\n\rtimestamp_net\x18\x05 \x01(\t\x12\x31\n\rstream_sample\x18\x06 \x01(\x0b\x32\x18.p2p.StreamSampleContentH\x00\x12\x16\n\x0cjson_content\x18\x07 \x01(\tH\x00\x42\t\n\x07\x63ontent\"\x90\x01\n\x13StreamSampleContent\x12\x36\n\x07samples\x18\x01 \x03(\x0b\x32%.p2p.StreamSampleContent.SamplesEntry\x1a\x41\n\x0cSamplesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12 \n\x05value\x18\x02 \x01(\x0b\x32\x11.p2p.StreamSample:\x02\x38\x01\"e\n\x0cStreamSample\x12\x1d\n\x04\x64\x61ta\x18\x01 \x01(\x0b\x32\x0f.p2p.SampleData\x12\x10\n\x08\x64\x61ta_tag\x18\x02 \x01(\x05\x12\x16\n\tdata_uuid\x18\x03 \x01(\tH\x00\x88\x01\x01\x42\x0c\n\n_data_uuid\"\x8e\x01\n\nSampleData\x12&\n\x0btensor_data\x18\x01 \x01(\x0b\x32\x0f.p2p.TensorDataH\x00\x12$\n\nimage_data\x18\x02 \x01(\x0b\x32\x0e.p2p.ImageDataH\x00\x12\"\n\ttext_data\x18\x03 \x01(\x0b\x32\r.p2p.TextDataH\x00\x42\x0e\n\x0c\x64\x61ta_payload\"8\n\nTensorData\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x12\r\n\x05\x64type\x18\x02 \x01(\t\x12\r\n\x05shape\x18\x03 \x03(\x05\"\x19\n\tImageData\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\"\x18\n\x08TextData\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\tB\x0cZ\n./proto-gob\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'message_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'Z\n./proto-go'
  _globals['_STREAMSAMPLECONTENT_SAMPLESENTRY']._loaded_options = None
  _globals['_STREAMSAMPLECONTENT_SAMPLESENTRY']._serialized_options = b'8\001'
  _globals['_MESSAGE']._serialized_start=23
  _globals['_MESSAGE']._serialized_end=215
  _globals['_STREAMSAMPLECONTENT']._serialized_start=218
  _globals['_STREAMSAMPLECONTENT']._serialized_end=362
  _globals['_STREAMSAMPLECONTENT_SAMPLESENTRY']._serialized_start=297
  _globals['_STREAMSAMPLECONTENT_SAMPLESENTRY']._serialized_end=362
  _globals['_STREAMSAMPLE']._serialized_start=364
  _globals['_STREAMSAMPLE']._serialized_end=465
  _globals['_SAMPLEDATA']._serialized_start=468
  _globals['_SAMPLEDATA']._serialized_end=610
  _globals['_TENSORDATA']._serialized_start=612
  _globals['_TENSORDATA']._serialized_end=668
  _globals['_IMAGEDATA']._serialized_start=670
  _globals['_IMAGEDATA']._serialized_end=695
  _globals['_TEXTDATA']._serialized_start=697
  _globals['_TEXTDATA']._serialized_end=721

# @@protoc_insertion_point(module_scope)
