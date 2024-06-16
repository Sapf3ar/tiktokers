# pylint: skip-file
# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import inference_pb2 as inference__pb2


class InferenceServerStub(object):
    """The inference service definition.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.inference = channel.unary_unary(
                '/InferenceServer/inference',
                request_serializer=inference__pb2.InferenceRequest.SerializeToString,
                response_deserializer=inference__pb2.InferenceReply.FromString,
                )


class InferenceServerServicer(object):
    """The inference service definition.
    """

    def inference(self, request, context):
        """Sends a inference reply
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_InferenceServerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'inference': grpc.unary_unary_rpc_method_handler(
                    servicer.inference,
                    request_deserializer=inference__pb2.InferenceRequest.FromString,
                    response_serializer=inference__pb2.InferenceReply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'InferenceServer', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class InferenceServer(object):
    """The inference service definition.
    """

    @staticmethod
    def inference(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/InferenceServer/inference',
            inference__pb2.InferenceRequest.SerializeToString,
            inference__pb2.InferenceReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
