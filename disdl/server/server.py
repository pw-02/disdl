import grpc
from concurrent import futures
import sys
sys.path.append(".")
sys.path.append("disdl\protos")
import minibatch_service_pb2 as minibatch_service_pb2
import minibatch_service_pb2_grpc as minibatch_service_pb2_grpc
import google.protobuf.empty_pb2
from logger_config import configure_logger
import hydra
from omegaconf import DictConfig
from args import DisDLArgs, DatasetConfig
from batch_manager import BatchManager
from omegaconf import OmegaConf
from typing import Dict, List
from job import DLTJob
from dataset import S3DatasetFactory
from batch import Batch
from cache_status import CacheStatus
import json
from omegaconf import OmegaConf
from omegaconf import DictConfig
from dacite import from_dict
logger = configure_logger(__name__)
from google.protobuf.empty_pb2 import Empty
import uuid

class CacheAwareMiniBatchService(minibatch_service_pb2_grpc.MiniBatchServiceServicer):
    def __init__(self, args:DisDLArgs):
        self.args = args
        self.datasets: Dict[str, BatchManager] = {}  # dataset_name -> BatchManager
        self.job_to_dataset: Dict[str, str] = {}

        for cfg in args.available_datasets:
            dataset = S3DatasetFactory.create_dataset(
                dataset_location=cfg.storage_backend,
                batch_size=cfg.batch_size,
                num_partitions=cfg.num_partitions,
                shuffle=cfg.shuffle,
                drop_last=cfg.drop_last,
                min_lookahead_steps=50,  # or cfg.min_lookahead_steps if available
                transforms=None  # or cfg.transforms if you add support
            )
            self.datasets[cfg.name] = BatchManager(
                dataset=dataset,
                use_prefetching=args.enable_prefetching,
                prefetch_lambda_name=cfg.prefetch_lambda_name,
                prefetch_simulation_time=None,
                cache_address=args.cache_address,
                shared_cache=None
            )
            logger.info(f"Registered dataset '{cfg.name}' with {len(dataset)} samples.")
        logger.info("All datasets registered successfully.")

    
    def ListDatasets(self, request: Empty, context):
        dataset_infos = []
        for name, batch_manager in self.datasets.items():
            info = batch_manager.dataset.dataset_info()
            dataset_infos.append(minibatch_service_pb2.DatasetInfo(
                name=name,
                location=info["location"],
                num_samples=info["num_samples"],
                num_batches=info["num_batches"],
                num_partitions=info["num_partitions"]
            ))
        
        return minibatch_service_pb2.ListDatasetsResponse(datasets=dataset_infos)
    
    def Ping(self, request, context):
        return minibatch_service_pb2.PingResponse(message = 'pong')
    
    def RegisterJob(self, request, context):
        try:
            dataset_name = request.dataset_name
            if dataset_name not in self.datasets:
                return minibatch_service_pb2.RegisterJobResponse(
                    job_id="",
                    dataset_info="",
                    errorMessage=f"Dataset '{dataset_name}' not registered."
                )
            
            job_id = uuid.uuid4().hex[:8]  # Generate a unique job ID
            self.job_to_dataset[job_id] = dataset_name
            self.datasets[dataset_name].add_job(job_id=job_id)
            logger.info(f"Registered job {job_id} for dataset '{dataset_name}'")

            return minibatch_service_pb2.RegisterJobResponse(
                job_id=job_id,
                errorMessage=""
            )
        except Exception as e:
            logger.exception(f"Failed to register job for dataset '{dataset_name}': {e}")
            return minibatch_service_pb2.RegisterJobResponse(
                job_id="",
                errorMessage=str(e)
            )
        
    def GetNextBatchForJob(self, request, context):
        job_id = request.job_id
        dataset_name = self.job_to_dataset.get(job_id)

        if dataset_name is None:
            logger.warning(f"Unknown job_id {job_id}")
            return minibatch_service_pb2.GetNextBatchForJobResponse(
                batch=minibatch_service_pb2.Batch(batch_id="None", samples="", is_cached=False)
            )
        try:
            next_batch, should_cache, eviction_candidate = self.datasets[dataset_name].get_next_batch_for_job(job_id)
            is_cached = False
            if next_batch is None:
                return minibatch_service_pb2.GetNextBatchForJobResponse(
                    batch=minibatch_service_pb2.Batch(batch_id="None", samples="", is_cached=False)
                )

            samples = self.datasets[dataset_name].dataset.get_samples(next_batch.indices)
            is_cached = next_batch.cache_status in [CacheStatus.CACHED]

            return minibatch_service_pb2.GetNextBatchForJobResponse(
                batch=minibatch_service_pb2.Batch(
                    batch_id=next_batch.batch_id,
                    samples=json.dumps(samples),
                    is_cached=is_cached
                ),
                should_cache=should_cache,
                eviction_candidate=eviction_candidate,
            )
        except Exception as e:
            logger.exception(f"Error getting batch for job {job_id}: {e}")
            return minibatch_service_pb2.GetNextBatchForJobResponse(
                batch=minibatch_service_pb2.Batch(batch_id="None", samples="", is_cached=False)
            )
        
    def JobEnded(self, request, context):
        job_id = request.job_id
        dataset_name = self.job_to_dataset.get(job_id)
        if dataset_name is None:
            logger.warning(f"Unknown job_id {job_id}")
            return Empty()

        # self.datasets[dataset_name].handle_job_ended(job_id)
        logger.info(f"Job {job_id} ended for dataset {dataset_name}")
        return Empty()
    
    def JobUpdate(self, request, context):
        job_id = request.job_id
        dataset_name = self.job_to_dataset.get(job_id)
        if dataset_name is None:
            logger.warning(f"Unknown job_id {job_id}")
            return Empty()

        batch_is_cached = request.batch_is_cached
        evicted_batch_id = request.evicted_batch_id

        self.datasets[dataset_name].processed_batch_update(
            job_id=job_id,
            batch_is_cached=batch_is_cached,
            evicited_batch_id=evicted_batch_id
        )
        logger.info(f"Job {job_id} update reported for dataset {dataset_name}")
        return Empty()


    

@hydra.main(version_base=None, config_path="conf", config_name="config")
def serve(cfg: DictConfig):
    try:
        logger.info("Starting DisDL ML Dataloader Service")
        logger.info(f"Loaded Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

        # Automatically convert DictConfig to DisDLArgs dataclass
        # args = OmegaConf.to_object(cfg)  # This assumes cfg matches DisDLArgs structure exactly
        # disdl_args = DisDLArgs(**args)
        args_dict = OmegaConf.to_container(cfg, resolve=True)
        disdl_args = from_dict(data_class=DisDLArgs, data=args_dict)
        # Start the minibatch service with all datasets registered
        cache_service = CacheAwareMiniBatchService(disdl_args)

        # Launch the gRPC server
        max_workers = 1
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        minibatch_service_pb2_grpc.add_MiniBatchServiceServicer_to_server(cache_service, server)
        server.add_insecure_port('[::]:50051')
        server.start()
        logger.info(f"Server started. Listening on port 50051 with {max_workers} worker(s).")

        # Keep the server running until interrupted
        server.wait_for_termination()

    except KeyboardInterrupt:
        logger.info("Server stopped due to keyboard interrupt")
        if 'server' in locals():
            server.stop(0)
    except Exception as e:
            logger.exception(f"Exception occurred during server execution: {e}")
            if 'server' in locals():
                server.stop(0)
    finally:
        pass
        # if 'coordinator' in locals():
        #     logger.info(f"Total Lambda Invocations:{coordinator.lambda_invocation_count}")
        #     coordinator.stop_workers()

if __name__ == '__main__':
    serve()
