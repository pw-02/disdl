import grpc
import sys
# sys.path.append(".")
# sys.path.append("disdl\protos")
import disdl.protos.minibatch_service_pb2 as pb
import disdl.protos.minibatch_service_pb2_grpc as pb_grpc
from google.protobuf.empty_pb2 import Empty
import json

class MiniBatchClient:
    def __init__(self, address="localhost:50051"):
        self.channel = grpc.insecure_channel(address)
        self.stub = pb_grpc.MiniBatchServiceStub(self.channel)
        self.ping()

    def close(self):
        self.channel.close()

    def ping(self):
        return self.stub.Ping(pb.PingRequest()).message

    def list_datasets(self):
        response = self.stub.ListDatasets(Empty())
        return [d.name for d in response.datasets]

    def register_job(self, dataset_name, processing_speed=1.0):
        request = pb.RegisterJobRequest(dataset_name=dataset_name,
                                        processing_speed=processing_speed)
        response = self.stub.RegisterJob(request)

        if response.errorMessage:
            raise RuntimeError(response.errorMessage)

        dataset_info = {
            "name": response.dataset_info.name,
            "location": response.dataset_info.location,
            "num_samples": response.dataset_info.num_samples,
            "num_batches": response.dataset_info.num_batches,
            "num_partitions": response.dataset_info.num_partitions,
        }

        return response.job_id, dataset_info


    def get_next_batch_metadata(self, job_id):
        request = pb.GetNextBatchForJobRequest(job_id=job_id)
        response = self.stub.GetNextBatchForJob(request)
        batch = response.batch
        should_cache = response.should_cache
        eviction_candidate = response.eviction_candidate if response.eviction_candidate else None
        batch_id = batch.batch_id
        samples = json.loads(batch.samples)
        if batch.batch_id == "None":
            return None
        # print(f"Batch ID: {batch_id} for job {job_id}")
        return batch_id, samples, should_cache, eviction_candidate

    def end_job(self, job_id):
        request = pb.JobEndedRequest(job_id=job_id)
        self.stub.JobEnded(request)


    def report_job_update(self,
                            job_id: str,
                            processed_batch_id: str,
                            batch_is_cached: str,
                            eviction_candidate_batch_id: str = None,
                            evicted_batch_id: str = None):
       
        request = pb.JobUpdateRequest(
            job_id=job_id,
            processed_batch_id=processed_batch_id,
            batch_is_cached=batch_is_cached,
            eviction_candidate_batch_id=eviction_candidate_batch_id,
            evicted_batch_id=evicted_batch_id
        )
        self.stub.JobUpdate(request)


def test_client():
    client = MiniBatchClient()
    response = client.ping()
    print(f"✅ Ping successful: {response}")
    datasets = client.list_datasets()
    print("Datasets:", datasets)
    
    if datasets:
        job_id = client.register_job(datasets[0])
        print("Registered job ID:", job_id)
        
        while True:
            batch_id, samples, should_cache, eviction_candidate = client.get_next_batch_metadata(job_id)
          
            print("Batch:", batch_id)
            client.report_job_update(job_id, False, eviction_candidate)
        
        client.end_job(job_id)
    
    client.close()
if __name__ == "__main__":
    test_client()

