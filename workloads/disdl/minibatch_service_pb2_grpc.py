# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
import disdl.minibatch_service_pb2 as protos_dot_minibatch__service__pb2

GRPC_GENERATED_VERSION = '1.70.0'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in protos/minibatch_service_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class MiniBatchServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Ping = channel.unary_unary(
                '/MiniBatchService/Ping',
                request_serializer=protos_dot_minibatch__service__pb2.PingRequest.SerializeToString,
                response_deserializer=protos_dot_minibatch__service__pb2.PingResponse.FromString,
                _registered_method=True)
        self.RegisterJob = channel.unary_unary(
                '/MiniBatchService/RegisterJob',
                request_serializer=protos_dot_minibatch__service__pb2.RegisterJobRequest.SerializeToString,
                response_deserializer=protos_dot_minibatch__service__pb2.RegisterJobResponse.FromString,
                _registered_method=True)
        self.GetNextBatchForJob = channel.unary_unary(
                '/MiniBatchService/GetNextBatchForJob',
                request_serializer=protos_dot_minibatch__service__pb2.GetNextBatchForJobRequest.SerializeToString,
                response_deserializer=protos_dot_minibatch__service__pb2.GetNextBatchForJobResponse.FromString,
                _registered_method=True)
        self.JobEnded = channel.unary_unary(
                '/MiniBatchService/JobEnded',
                request_serializer=protos_dot_minibatch__service__pb2.JobEndedRequest.SerializeToString,
                response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                _registered_method=True)
        self.JobUpdate = channel.unary_unary(
                '/MiniBatchService/JobUpdate',
                request_serializer=protos_dot_minibatch__service__pb2.JobUpdateRequest.SerializeToString,
                response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                _registered_method=True)


class MiniBatchServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Ping(self, request, context):
        """Health check
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RegisterJob(self, request, context):
        """Register a dataset for a job
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetNextBatchForJob(self, request, context):
        """Get the next batch for a given job
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def JobEnded(self, request, context):
        """Notify server when a job ends
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def JobUpdate(self, request, context):
        """Update job status (e.g., batch consumption rate)
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_MiniBatchServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Ping': grpc.unary_unary_rpc_method_handler(
                    servicer.Ping,
                    request_deserializer=protos_dot_minibatch__service__pb2.PingRequest.FromString,
                    response_serializer=protos_dot_minibatch__service__pb2.PingResponse.SerializeToString,
            ),
            'RegisterJob': grpc.unary_unary_rpc_method_handler(
                    servicer.RegisterJob,
                    request_deserializer=protos_dot_minibatch__service__pb2.RegisterJobRequest.FromString,
                    response_serializer=protos_dot_minibatch__service__pb2.RegisterJobResponse.SerializeToString,
            ),
            'GetNextBatchForJob': grpc.unary_unary_rpc_method_handler(
                    servicer.GetNextBatchForJob,
                    request_deserializer=protos_dot_minibatch__service__pb2.GetNextBatchForJobRequest.FromString,
                    response_serializer=protos_dot_minibatch__service__pb2.GetNextBatchForJobResponse.SerializeToString,
            ),
            'JobEnded': grpc.unary_unary_rpc_method_handler(
                    servicer.JobEnded,
                    request_deserializer=protos_dot_minibatch__service__pb2.JobEndedRequest.FromString,
                    response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            ),
            'JobUpdate': grpc.unary_unary_rpc_method_handler(
                    servicer.JobUpdate,
                    request_deserializer=protos_dot_minibatch__service__pb2.JobUpdateRequest.FromString,
                    response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'MiniBatchService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('MiniBatchService', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class MiniBatchService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Ping(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/MiniBatchService/Ping',
            protos_dot_minibatch__service__pb2.PingRequest.SerializeToString,
            protos_dot_minibatch__service__pb2.PingResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def RegisterJob(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/MiniBatchService/RegisterJob',
            protos_dot_minibatch__service__pb2.RegisterJobRequest.SerializeToString,
            protos_dot_minibatch__service__pb2.RegisterJobResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def GetNextBatchForJob(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/MiniBatchService/GetNextBatchForJob',
            protos_dot_minibatch__service__pb2.GetNextBatchForJobRequest.SerializeToString,
            protos_dot_minibatch__service__pb2.GetNextBatchForJobResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def JobEnded(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/MiniBatchService/JobEnded',
            protos_dot_minibatch__service__pb2.JobEndedRequest.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def JobUpdate(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/MiniBatchService/JobUpdate',
            protos_dot_minibatch__service__pb2.JobUpdateRequest.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
