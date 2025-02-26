import grpc
from concurrent import futures
import protos.minibatch_service_pb2 as minibatch_service_pb2
import protos.minibatch_service_pb2_grpc as minibatch_service_pb2_grpc
import google.protobuf.empty_pb2
from logger_config import logger
import hydra
from omegaconf import DictConfig
from args import DisDLArgs
from server.bacth_manager import CentralBatchManager
from omegaconf import OmegaConf
from typing import Dict

# from batch import Batch
# from typing import Dict, List
# from dataset import Dataset
# from central_batch_manager import CentralBatchManager, DLTJob, PrefetchService

class CacheAwareMiniBatchService(minibatch_service_pb2_grpc.MiniBatchServiceServicer):
    def __init__(self, args:DisDLArgs):
        self.args = args
        self.datasets: Dict[str,CentralBatchManager] = {}
        self.jobs: Dict[DLTJob] = {}

    def Ping(self, request, context):
        return minibatch_service_pb2.PingResponse(message = 'pong')
    
    def RegisterDataset(self, request, context):

        if request.data_dir in self.datasets: # if dataset is already registered
            dataset =  self.datasets[request.data_dir].dataset
            message = f"Dataset '{request.data_dir}' already registered. Files: {len(dataset)}, Batches: {dataset.num_batches}, Partitions:{len(dataset.partitions)}"
            success = True
        else:
            dataset = Dataset(request.data_dir, self.args.batch_size, False, self.args.partitions_per_dataset, request.dataset_kind, max_dataset_size=None)
            self.datasets[request.data_dir] = CentralBatchManager(dataset=dataset, args=self.args)
            message = f"Dataset '{request.data_dir}'. Total Files: {len(dataset)}, Total Batches:{dataset.num_batches} Partitions:{len(dataset.partitions)}"
            success = True
        logger.info(message)
        return minibatch_service_pb2.RegisterDatasetResponse(dataset_is_registered=success, 
                                                             total_batches=dataset.num_batches, 
                                                             message=message)
    
    

    def JobUpdate(self, request, context):
        job_id = request.job_id
        data_dir = request.data_dir
        previous_step_batch_id = request.previous_step_batch_id
        previous_step_wait_for_data_time = request.previous_step_wait_for_data_time
        previous_step_is_cache_hit = request.previous_step_is_cache_hit
        previous_step_gpu_time = request.previous_step_gpu_time
        cached_previous_batch = request.cached_previous_batch
        
        if isinstance(self.args, CoorDLArgs):
            self.datasets[data_dir].update_job_progess(
                previous_step_batch_id,
                previous_step_is_cache_hit,
                cached_previous_batch)
        else:
            self.datasets[data_dir].update_job_progess(
                job_id,
                previous_step_batch_id,
                previous_step_wait_for_data_time,
                previous_step_is_cache_hit,
                previous_step_gpu_time,
                cached_previous_batch)
        return google.protobuf.empty_pb2.Empty()
    
    def GetNextBatchForJob(self, request, context):
        job_id = request.job_id
        data_dir = request.data_dir
       
        if data_dir not in self.datasets:
            message = f"Failed to register job with id '{job_id}' because data dir '{data_dir}' was not found in SUPER."
            logger.info(message)
        
        next_batch:Batch = self.datasets[data_dir].get_next_batch(job_id)
        if next_batch is None:
            return minibatch_service_pb2.GetNextBatchForJobResponse(
                job_id=job_id,
                batch=minibatch_service_pb2.Batch(batch_id='None', indicies=[], is_cached=False))
        else:
                
            return minibatch_service_pb2.GetNextBatchForJobResponse(
                job_id=request.job_id,
                batch=minibatch_service_pb2.Batch(batch_id=next_batch.batch_id, 
                                                indicies=next_batch.indicies, 
                                                is_cached=next_batch.is_cached))
        # return response
    
    def JobEnded(self, request, context):
        job_id = request.job_id
        data_dir = request.data_dir
        self.datasets[data_dir].job_ended(job_id=job_id)
        return google.protobuf.empty_pb2.Empty()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def serve(cfg: DictConfig):
    try:
        logger.info("Starting DisDL ML Dataloader Service")
        logger.info(f"Config: {OmegaConf.to_yaml(cfg, resolve=True)}")
        args:DisDLArgs = DisDLArgs(
            batch_size = cfg.workload.batch_size,
            num_dataset_partitions = cfg.workload.num_dataset_partitions,
            lookahead_steps = cfg.lookahead_steps,
            serverless_cache_address = cfg.serverless_cache_address,
            use_prefetching = cfg.use_prefetching,
            use_keep_alive = cfg.use_keep_alive,
            prefetch_lambda_name = cfg.workload.prefetch_lambda_name,
            prefetch_cost_cap_per_hour=cfg.prefetch_cost_cap_per_hour,
            cache_evition_ttl_threshold = cfg.cache_evition_ttl_threshold,
            prefetch_simulation_time = cfg.prefetch_simulation_time,
            evict_from_cache_simulation_time = cfg.evict_from_cache_simulation_time,
            shuffle = cfg.workload.shuffle,
            drop_last = cfg.workload.drop_last,
            workload_kind = cfg.workload.kind)
        
        cache_service = CacheAwareMiniBatchService(args) 
        max_workers = 1
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

        minibatch_service_pb2_grpc.add_MiniBatchServiceServicer_to_server(cache_service, server)
        server.add_insecure_port('[::]:50051')
        server.start()
        logger.info(f"Server started. Listening on port 50051 with {max_workers} workers.")

        # Keep the server running until interrupted
        server.wait_for_termination()

    except KeyboardInterrupt:
        logger.info("Server stopped due to keyboard interrupt")
        server.stop(0)
    except Exception as e:
            logger.exception(f"{e}. Shutting Down.")
            if  'server' in locals():
                server.stop(0)
    finally:
        pass
        # if 'coordinator' in locals():
        #     logger.info(f"Total Lambda Invocations:{coordinator.lambda_invocation_count}")
        #     coordinator.stop_workers()

if __name__ == '__main__':
    serve()
