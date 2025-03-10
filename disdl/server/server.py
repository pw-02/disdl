import grpc
from concurrent import futures
import minibatch_service_pb2 as minibatch_service_pb2
import minibatch_service_pb2_grpc as minibatch_service_pb2_grpc
import google.protobuf.empty_pb2
from logger_config import logger
import hydra
from omegaconf import DictConfig
from args import DisDLArgs
from batch_manager import CentralBatchManager
from omegaconf import OmegaConf
from typing import Dict
from job import DLTJob
from dataset import ImageNetDataset,MSCOCODataset
from batch import Batch, CacheStatus
import json

class CacheAwareMiniBatchService(minibatch_service_pb2_grpc.MiniBatchServiceServicer):
    def __init__(self, args:DisDLArgs):
        self.args = args
        self.datasets: Dict[str,CentralBatchManager] = {}
        self.jobs: Dict[DLTJob] = {}

    def Ping(self, request, context):
        return minibatch_service_pb2.PingResponse(message = 'pong')
    
    def RegisterJob(self, request, context):
        try:
            error_message = None
            dataset_location = request.dataset_location
            transforms = None
            if dataset_location in self.datasets:
                #datasets already registered by another job, add new job to dataset and return job_id, and dataset info
                job_id = self.datasets[dataset_location].add_job()
                dataset_info = self.datasets[dataset_location].dataset_info()
            else:
                #register dataset and add job
                dataset = MSCOCODataset(dataset_location, transforms)
                self.datasets[dataset_location] = CentralBatchManager(dataset=dataset, args=self.args)
                dataset_info = self.datasets[dataset_location].dataset_info()
                logger.info(f"Dataset '{dataset_location}' added. Total Files: {len(dataset)}, Total Batches:{dataset_info['num_batches']}")
                job_id = self.datasets[dataset_location].add_job()
            message = f"Job '{job_id}' registered for dataset '{dataset_location}'. Total Jobs: {len(self.datasets[dataset_location].active_jobs)}"
            logger.info(message)
        except Exception as e:
            error_message = f"Failed to register job with dataset '{dataset_location}'. Error: {e}"
            logger.error(error_message)
        return minibatch_service_pb2.RegisterJobResponse(
            job_id=job_id, 
            dataset_info=str(dataset_info),
            errorMessage=error_message)
    
    def GetNextBatchForJob(self, request, context):
        job_id = request.job_id
        dataset_location = request.dataset_location
       
        next_batch:Batch = self.datasets[dataset_location].get_next_batch_for_job(job_id)
        samples = self.datasets[dataset_location].dataset.get_samples(next_batch.indices)
        use_cache = True if next_batch.cache_status == CacheStatus.CACHED or next_batch.cache_status == CacheStatus.CACHING_IN_PROGRESS else False
        if next_batch is None:
            return minibatch_service_pb2.GetNextBatchForJobResponse(
                batch=minibatch_service_pb2.Batch(batch_id='None', indicies=[], is_cached=False))
        else:
            return minibatch_service_pb2.GetNextBatchForJobResponse(
                batch=minibatch_service_pb2.Batch(
                    batch_id=next_batch.batch_id, 
                    samples=json.dumps(samples), 
                    is_cached=use_cache))
        # return response
    
    def JobEnded(self, request, context):
        job_id = request.job_id
        dataset_location = request.dataset_location
        self.datasets[dataset_location].handle_job_ended(job_id=job_id)
        return google.protobuf.empty_pb2.Empty()
    

    

    
    

    # def JobUpdate(self, request, context):
    #     job_id = request.job_id
    #     data_dir = request.data_dir
    #     previous_step_batch_id = request.previous_step_batch_id
    #     previous_step_wait_for_data_time = request.previous_step_wait_for_data_time
    #     previous_step_is_cache_hit = request.previous_step_is_cache_hit
    #     previous_step_gpu_time = request.previous_step_gpu_time
    #     cached_previous_batch = request.cached_previous_batch
        
    #     if isinstance(self.args, CoorDLArgs):
    #         self.datasets[data_dir].update_job_progess(
    #             previous_step_batch_id,
    #             previous_step_is_cache_hit,
    #             cached_previous_batch)
    #     else:
    #         self.datasets[data_dir].update_job_progess(
    #             job_id,
    #             previous_step_batch_id,
    #             previous_step_wait_for_data_time,
    #             previous_step_is_cache_hit,
    #             previous_step_gpu_time,
    #             cached_previous_batch)
    #     return google.protobuf.empty_pb2.Empty()
    
    # def GetNextBatchForJob(self, request, context):
    #     job_id = request.job_id
    #     data_dir = request.data_dir
       
    #     if data_dir not in self.datasets:
    #         message = f"Failed to register job with id '{job_id}' because data dir '{data_dir}' was not found in SUPER."
    #         logger.info(message)
        
    #     next_batch:Batch = self.datasets[data_dir].get_next_batch(job_id)
    #     if next_batch is None:
    #         return minibatch_service_pb2.GetNextBatchForJobResponse(
    #             job_id=job_id,
    #             batch=minibatch_service_pb2.Batch(batch_id='None', indicies=[], is_cached=False))
    #     else:
                
    #         return minibatch_service_pb2.GetNextBatchForJobResponse(
    #             job_id=request.job_id,
    #             batch=minibatch_service_pb2.Batch(batch_id=next_batch.batch_id, 
    #                                             indicies=next_batch.indicies, 
    #                                             is_cached=next_batch.is_cached))
    #     # return response
    
    # def JobEnded(self, request, context):
    #     job_id = request.job_id
    #     data_dir = request.data_dir
    #     self.datasets[data_dir].job_ended(job_id=job_id)
    #     return google.protobuf.empty_pb2.Empty()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def serve(cfg: DictConfig):
    try:
        logger.info("Starting DisDL ML Dataloader Service")
        logger.info(f"Config: {OmegaConf.to_yaml(cfg, resolve=True)}")
        args:DisDLArgs = DisDLArgs(
            num_dataset_partitions = cfg.workload.num_dataset_partitions,
            batch_size = cfg.workload.batch_size,
            lookahead_steps = cfg.lookahead_steps,
            serverless_cache_address = cfg.serverless_cache_address,
            cache_keep_alive_timeout = cfg.cache_keep_alive_timeout,
            use_prefetching = cfg.use_prefetching,
            use_keep_alive = cfg.use_keep_alive,
            prefetch_lambda_name = cfg.workload.prefetch_lambda_name,
            prefetch_cost_cap_per_hour=cfg.prefetch_cost_cap_per_hour,
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
