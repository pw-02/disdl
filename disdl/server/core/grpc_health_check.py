import grpc
from google.protobuf.empty_pb2 import Empty

# Update these imports based on your project structure
# import disdl.minibatch_service_pb2 as pb
# import disdl.minibatch_service_pb2_grpc as pb_grpc
import minibatch_service_pb2 as pb
import minibatch_service_pb2_grpc as pb_grpc
def check_server(address="localhost:50051"):
    try:
        print(f"Trying to connect to {address}...")
        channel = grpc.insecure_channel(address)
        grpc.channel_ready_future(channel).result(timeout=5)  # Wait until channel is ready

        stub = pb_grpc.MiniBatchServiceStub(channel)
        response = stub.Ping(pb.PingRequest())
        print(f"✅ Ping successful: {response.message}")

    except grpc.FutureTimeoutError:
        print("❌ Timeout: gRPC server is not reachable.")
    except grpc.RpcError as e:
        print(f"❌ RPC Error: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    check_server("localhost:50051")  # Change this if your server uses a different address
