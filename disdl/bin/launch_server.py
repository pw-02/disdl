# disdl/bin/launch_server.py

import grpc
from concurrent import futures
import hydra
import uuid
import json
import logging
from typing import Dict
from omegaconf import DictConfig, OmegaConf
from dacite import from_dict
from google.protobuf.empty_pb2 import Empty

# Import DisDL logic
from disdl.utils.logger_config import configure_logger
from disdl.core.args import DisDLArgs, DatasetConfig
from disdl.core.batch_manager import BatchManager
from disdl.core.job import DLTJob
from disdl.cache.cache_status import CacheStatus
from disdl.datasets.s3_dataset import S3DatasetFactory
from disdl.datasets.disk_dataset import DiskDatasetFactory
from disdl.protos import minibatch_service_pb2, minibatch_service_pb2_grpc

logger = configure_logger(__name__)


class CacheAwareMiniBatchService(minibatch_service_pb2_grpc.MiniBatchServiceServicer):
    def __init__(self, args: DisDLArgs):
        self.args = args
        self.datasets: Dict[str, BatchManager] = {}
        self.job_to_dataset: Dict[str, str] = {}

        for cfg in args.available_datasets:
            dataset = self._create_data_factory(cfg)
            self.datasets[cfg.name] = BatchManager(
                dataset=dataset,
                use_prefetching=args.enable_prefetching,
                prefetch_lambda_name=cfg.prefetch_lambda_name,
                prefetch_simulation_time=None,
                cache_address=args.cache_address,
                shared_cache=None,
            )
            logger.info(f"Registered dataset '{cfg.name}' with {len(dataset)} samples.")
        logger.info("All datasets registered successfully.")

    def _create_data_factory(self, config: DatasetConfig):
        if config.storage_backend == "s3":
            return S3DatasetFactory.create_dataset(
                dataset_location=config.s3_storage_path,
                batch_size=config.batch_size,
                num_partitions=config.num_partitions,
                shuffle=config.shuffle,
                drop_last=config.drop_last,
                min_lookahead_steps=50,
                transforms=None,
            )
        elif config.storage_backend == "local":
            return DiskDatasetFactory.create_dataset(
                dataset_location=config.local_storage_path,
                batch_size=config.batch_size,
                num_partitions=config.num_partitions,
                shuffle=config.shuffle,
                drop_last=config.drop_last,
                min_lookahead_steps=50,
                transforms=None,
            )
        else:
            raise ValueError(f"Unsupported storage backend: {config.storage_backend}")

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
        return minibatch_service_pb2.PingResponse(message="pong")

    def RegisterJob(self, request, context):
        dataset_name = request.dataset_name
        if dataset_name not in self.datasets:
            return minibatch_service_pb2.RegisterJobResponse(
                job_id="",
                dataset_info=None,
                errorMessage=f"Dataset '{dataset_name}' not registered."
            )

        job_id = uuid.uuid4().hex[:8]
        self.job_to_dataset[job_id] = dataset_name
        self.datasets[dataset_name].add_job(job_id=job_id)

        info = self.datasets[dataset_name].dataset.dataset_info()
        dataset_info_msg = minibatch_service_pb2.DatasetInfo(
            name=dataset_name,
            location=info["location"],
            num_samples=info["num_samples"],
            num_batches=info["num_batches"],
            num_partitions=info["num_partitions"]
        )

        logger.info(f"Registered job {job_id} for dataset '{dataset_name}'")

        return minibatch_service_pb2.RegisterJobResponse(
            job_id=job_id,
            dataset_info=dataset_info_msg,
            errorMessage=""
        )


    def GetNextBatchForJob(self, request, context):
        job_id = request.job_id
        dataset_name = self.job_to_dataset.get(job_id)

        if dataset_name is None:
            return minibatch_service_pb2.GetNextBatchForJobResponse(
                batch=minibatch_service_pb2.Batch(batch_id="None", samples="", is_cached=False)
            )

        try:
            next_batch, should_cache, eviction_candidate = self.datasets[dataset_name].get_next_batch_for_job(job_id)
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
        if dataset_name:
            logger.info(f"Job {job_id} ended for dataset {dataset_name}")
        return Empty()

    def JobUpdate(self, request, context):
        job_id = request.job_id
        dataset_name = self.job_to_dataset.get(job_id)
        if dataset_name:
            self.datasets[dataset_name].processed_batch_update(
                job_id=job_id,
                batch_is_cached=request.batch_is_cached,
                evicited_batch_id=request.evicted_batch_id
            )
            # logger.info(f"Job {job_id} update reported for dataset {dataset_name}")
        return Empty()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def serve(cfg: DictConfig):
    logger.info("Starting DisDL ML Dataloader Service")
    logger.info(f"Loaded Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    args_dict = OmegaConf.to_container(cfg, resolve=True)
    disdl_args = from_dict(data_class=DisDLArgs, data=args_dict)
    cache_service = CacheAwareMiniBatchService(disdl_args)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    minibatch_service_pb2_grpc.add_MiniBatchServiceServicer_to_server(cache_service, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info("Server listening on port 50051")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
