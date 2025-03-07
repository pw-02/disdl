# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: protos/minibatch_service.proto
# Protobuf Python Version: 5.29.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    29,
    0,
    '',
    'protos/minibatch_service.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1eprotos/minibatch_service.proto\x1a\x1bgoogle/protobuf/empty.proto\"\r\n\x0bPingRequest\"\x1f\n\x0cPingResponse\x12\x0f\n\x07message\x18\x01 \x01(\t\".\n\x12RegisterJobRequest\x12\x18\n\x10\x64\x61taset_location\x18\x01 \x01(\t\"Q\n\x13RegisterJobResponse\x12\x0e\n\x06job_id\x18\x01 \x01(\t\x12\x14\n\x0c\x64\x61taset_info\x18\x02 \x01(\t\x12\x14\n\x0c\x65rrorMessage\x18\x03 \x01(\t\"E\n\x19GetNextBatchForJobRequest\x12\x0e\n\x06job_id\x18\x01 \x01(\t\x12\x18\n\x10\x64\x61taset_location\x18\x02 \x01(\t\"*\n\x06Sample\x12\x11\n\tdata_path\x18\x01 \x01(\t\x12\r\n\x05label\x18\x02 \x01(\t\"=\n\x05\x42\x61tch\x12\x10\n\x08\x62\x61tch_id\x18\x01 \x01(\t\x12\x0f\n\x07samples\x18\x02 \x01(\t\x12\x11\n\tis_cached\x18\x03 \x01(\x08\"3\n\x1aGetNextBatchForJobResponse\x12\x15\n\x05\x62\x61tch\x18\x01 \x01(\x0b\x32\x06.Batch\";\n\x0fJobEndedRequest\x12\x0e\n\x06job_id\x18\x01 \x01(\t\x12\x18\n\x10\x64\x61taset_location\x18\x02 \x01(\t\"\xe3\x01\n\x10JobUpdateRequest\x12\x0e\n\x06job_id\x18\x01 \x01(\t\x12\x12\n\ndataset_id\x18\x02 \x01(\t\x12\x1e\n\x16previous_step_batch_id\x18\x03 \x01(\t\x12(\n previous_step_wait_for_data_time\x18\x04 \x01(\x01\x12\"\n\x1aprevious_step_is_cache_hit\x18\x05 \x01(\x08\x12\x1e\n\x16previous_step_gpu_time\x18\x06 \x01(\x01\x12\x1d\n\x15prefetched_next_batch\x18\x07 \x01(\x08\x32\xae\x02\n\x10MiniBatchService\x12#\n\x04Ping\x12\x0c.PingRequest\x1a\r.PingResponse\x12\x38\n\x0bRegisterJob\x12\x13.RegisterJobRequest\x1a\x14.RegisterJobResponse\x12M\n\x12GetNextBatchForJob\x12\x1a.GetNextBatchForJobRequest\x1a\x1b.GetNextBatchForJobResponse\x12\x34\n\x08JobEnded\x12\x10.JobEndedRequest\x1a\x16.google.protobuf.Empty\x12\x36\n\tJobUpdate\x12\x11.JobUpdateRequest\x1a\x16.google.protobuf.Emptyb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'protos.minibatch_service_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_PINGREQUEST']._serialized_start=63
  _globals['_PINGREQUEST']._serialized_end=76
  _globals['_PINGRESPONSE']._serialized_start=78
  _globals['_PINGRESPONSE']._serialized_end=109
  _globals['_REGISTERJOBREQUEST']._serialized_start=111
  _globals['_REGISTERJOBREQUEST']._serialized_end=157
  _globals['_REGISTERJOBRESPONSE']._serialized_start=159
  _globals['_REGISTERJOBRESPONSE']._serialized_end=240
  _globals['_GETNEXTBATCHFORJOBREQUEST']._serialized_start=242
  _globals['_GETNEXTBATCHFORJOBREQUEST']._serialized_end=311
  _globals['_SAMPLE']._serialized_start=313
  _globals['_SAMPLE']._serialized_end=355
  _globals['_BATCH']._serialized_start=357
  _globals['_BATCH']._serialized_end=418
  _globals['_GETNEXTBATCHFORJOBRESPONSE']._serialized_start=420
  _globals['_GETNEXTBATCHFORJOBRESPONSE']._serialized_end=471
  _globals['_JOBENDEDREQUEST']._serialized_start=473
  _globals['_JOBENDEDREQUEST']._serialized_end=532
  _globals['_JOBUPDATEREQUEST']._serialized_start=535
  _globals['_JOBUPDATEREQUEST']._serialized_end=762
  _globals['_MINIBATCHSERVICE']._serialized_start=765
  _globals['_MINIBATCHSERVICE']._serialized_end=1067
# @@protoc_insertion_point(module_scope)
