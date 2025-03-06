import grpc
import protos.minibatch_service_pb2 as minibatch_service_pb2
import protos.minibatch_service_pb2_grpc as minibatch_service_pb2_grpc
import ast
import json
class DisDLClient:
    def __init__(self, address: str):
        self.address = address
        self.channel = grpc.insecure_channel(address)
        self.stub = minibatch_service_pb2_grpc.MiniBatchServiceStub(self.channel)
        self.job_id = None
        self.dataset_location = None

    def registerJob(self, dataset_location: str, dataset_kind: str):
        response = self.stub.RegisterJob(minibatch_service_pb2.RegisterJobRequest(
            dataset_location=dataset_location
        ))
        self.job_id = response.job_id
        self.dataset_location = dataset_location
        dataset_info = ast.literal_eval(response.dataset_info)
        
        return self.job_id, dataset_info

    def sendJobEndNotification(self):
        response = self.stub.JobEnded(minibatch_service_pb2.JobEndedRequest(
            job_id=self.job_id,
            dataset_location=self.dataset_location))
        

    def sampleNetMinibatch(self):
        response = self.stub.GetNextBatchForJob(minibatch_service_pb2.GetNextBatchForJobRequest(
            job_id=self.job_id,
            dataset_location=self.dataset_location
        ))
        batch_id = response.batch.batch_id
        samples = [(sample[0], sample[1]) for sample in json.loads(response.batch.samples)] #data, label pairs

        is_cached = response.batch.is_cached
        return batch_id, samples, is_cached

    def sendUpdate(self, batch_id, wait_for_data_time: float, is_cache_hit: bool, gpu_time: float, cached_batch_on_miss: bool):
        if not self.job_id or not self.data_dir:
            print("Error: Job ID or Data Directory not set.")
            return

        try:
            self.stub.JobUpdate(minibatch_service_pb2.JobUpdateRequest(
                job_id=self.job_id,
                data_dir=self.data_dir,  # Fixed
                previous_step_batch_id=batch_id,
                previous_step_wait_for_data_time=wait_for_data_time,
                previous_step_is_cache_hit=is_cache_hit,
                previous_step_gpu_time=gpu_time,
                cached_previous_batch=cached_batch_on_miss
            ))
        except grpc.RpcError as e:
            print(f"Failed to send job update info to SUPER: {e.details()}")

    def close(self):
        """Closes the gRPC channel."""
        self.channel.close()

if __name__ == "__main__":
    num_epochs = 2
    client = DisDLClient("localhost:50051")
    jobid, dataset_info = client.registerJob("s3://sdl-cifar10/test/", "cifar10")
    print(f"Job ID: {jobid}, Dataset Info: {dataset_info}")

    num_batchs = dataset_info["num_batches"]
    for j in range(num_epochs):
        for i in range(num_batchs):
            batch_id, samples, is_cached = client.sampleNetMinibatch()
            print(f"Batch ID: {batch_id}, Num Samples: {len(samples)}, Is Cached: {is_cached}")
    client.sendJobEndNotification()
    client.close()