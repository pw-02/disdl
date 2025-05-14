import grpc
import minibatch_service_pb2 as pb
import minibatch_service_pb2_grpc as pb_grpc
from google.protobuf.empty_pb2 import Empty
import json

class MiniBatchClient:
    def __init__(self, address="localhost:50051"):
        self.channel = grpc.insecure_channel(address)
        self.stub = pb_grpc.MiniBatchServiceStub(self.channel)

    def close(self):
        self.channel.close()

    def ping(self):
        return self.stub.Ping(pb.PingRequest()).message

    def list_datasets(self):
        response = self.stub.ListDatasets(Empty())
        return [d.name for d in response.datasets]

    def register_job(self, dataset_name):
        request = pb.RegisterJobRequest(dataset_name=dataset_name)
        response = self.stub.RegisterJob(request)
        if response.errorMessage:
            raise RuntimeError(response.errorMessage)
        return response.job_id

    def get_next_batch(self, job_id):
        request = pb.GetNextBatchForJobRequest(job_id=job_id)
        response = self.stub.GetNextBatchForJob(request)
        batch = response.batch
        if batch.batch_id == "None":
            return None
        return json.loads(batch.samples), batch.is_cached

    def end_job(self, job_id):
        request = pb.JobEndedRequest(job_id=job_id)
        self.stub.JobEnded(request)

    def get_next_batch_metadata(self, job_id):
        request = pb.GetNextBatchForJobRequest(job_id=job_id)
        response = self.stub.GetNextBatchForJob(request)
        batch = response.batch
        if batch.batch_id == "None":
            return None
        return {"batch_id": batch.batch_id}
    

    def report_job_progress(self,
                            job_id: str,
                            batch_id: str,
                            data_fetch_time: float,
                            was_cache_hit: bool,
                            evicted_batch_id: str = ""):
    
        request = pb.JobUpdateRequest(
            job_id=job_id,
            batch_id=batch_id,
            data_fetch_time=data_fetch_time,
            was_cache_hit=was_cache_hit,
            evicted_batch_id=evicted_batch_id or ""
        )
        self.stub.JobUpdate(request)


