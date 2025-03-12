# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch.nn as nn
from collections import OrderedDict
import os
import time
from typing import Any, List, Optional, Tuple, Union
import hydra
from omegaconf import DictConfig
from lightning.fabric import Fabric, seed_everything
from lightning.fabric.loggers import CSVLogger
import torch
from models.albef import albef_model_for_retrieval, ALBEFTextTransform
from torch.utils.data import DataLoader
import torch.optim as optim

# from torch.nn import Sequential
from torchvision import transforms
from lightning.pytorch.core.saving import save_hparams_to_yaml
from disdl.disdl_client import DisDLClient

from baselines.tensorsocket.tensorsocket_sampler import TensorSocketSampler
from baselines.tensorsocket.tensorsocket_coco_dataset import TensorSocketCocoDataset
from baselines.tensorsocket.producer import TensorProducer
from baselines.tensorsocket.consumer import TensorConsumer
from disdl.disdl_iterable_dataset import DisDLCocoIterableDataset
from datetime import datetime, timezone
import numpy as np
# mean and standard deviation from the ALBEF repo:
# https://github.com/salesforce/ALBEF/blob/main/dataset/__init__.py#L16
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD_DEV = (0.26862954, 0.26130258, 0.27577711)


def finetune_multi_modal(config: DictConfig, train_logger: CSVLogger, val_logger: CSVLogger):
    if config.simulation_mode:
        config.accelerator = 'cpu'
    
    fabric = Fabric(accelerator=config.accelerator, 
                    devices=config.devices, 
                    precision=config.workload.precision)
    if config.seed is not None:
        seed_everything(config.seed) 
    else:
        seed_everything(config.job_id) # instead of torch.manual_seed(...)
    
    #setup model and optimizer
    model = albef_model_for_retrieval(config.workload, pretrained=False)
    optimizer = optim.Adam(model.parameters(), lr=config.workload.learning_rate)
    model, optimizer = fabric.setup(model, optimizer)
    # fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
    # fabric.print(f"Number of non-trainable parameters: {num_parameters(model, requires_grad=False):,}")
    
    train_dataloader = None
    val_dataloader = None
    tensorsocket_procuder:TensorProducer = None
    tensorsoket_consumer:TensorConsumer = None
    
    if config.dataloader.name == 'disdl':
        client = DisDLClient(address=config.dataloader.grpc_server_address)
        job_id, dataset_info = client.registerJob(dataset_location=config.workload.s3_train_prefix)
        num_batchs = dataset_info["num_batches"]
        client.close()

        train_dataset = DisDLCocoIterableDataset(
            job_id=client.job_id,
            dataset_location=client.dataset_location,
            num_samples=num_batchs,
            batch_size=config.workload.batch_size,
            image_transform=image_transform(),
            text_transform=ALBEFTextTransform(
                truncate=True, 
                pad_to_max_seq_len=True, 
                max_seq_len=30, 
                add_end_token=False),
            disdl_service_address=config.dataloader.grpc_server_address,
            cache_address=config.dataloader.cache_address,
            ssl=config.dataloader.ssl_enabled,
            use_compression=config.dataloader.use_compression,
            use_local_folder=config.dataloader.use_local_folder
            )
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=None,
            num_workers=config.workload.num_pytorch_workers,
            pin_memory=True if config.accelerator != 'cpu' else False)
        
        train_dataloader = fabric.setup_dataloaders(train_dataloader, move_to_device=True if config.accelerator != 'cpu' else False)

    elif config.dataloader.name == 'tensorsocket':
        # PyTorch DataLoader
        if config.dataloader.mode == 'producer':
            train_dataset = TensorSocketCocoDataset(
                s3_data_dir=config.workload.s3_train_prefix,
                image_transform=image_transform(),
                text_transform=ALBEFTextTransform(
                    truncate=True, 
                    pad_to_max_seq_len=True, 
                    max_seq_len=30, 
                    add_end_token=False))
            
            tensor_socket_sampler = TensorSocketSampler(
                data_source=train_dataset,
                batch_size=config.workload.batch_size)
            
            train_dataloader = DataLoader(
                train_dataset,
                sampler=tensor_socket_sampler,
                batch_size=None,
                num_workers=config.workload.num_pytorch_workers,
                pin_memory=True if config.accelerator != 'cpu' else False)
            
            train_dataloader = fabric.setup_dataloaders(train_dataloader, move_to_device=False)

            tensorsocket_procuder = TensorProducer(
                data_loader=train_dataloader,
                port=config.dataloader.producer_port,
                consumer_max_buffer_size=config.dataloader.consumer_maxbuffersize,
                ack_port=config.dataloader.producer_ackport,
                producer_batch_size=config.dataloader.producer_batch_size,
                )
            
        elif config.dataloader.mode == 'consumer':
            tensorsoket_consumer = TensorConsumer(
                port=config.dataloader.consumer_port,
                ack_port=config.dataloader.consumer_ackport,
                batch_size=config.workload.batch_size,
            )

    global_train_step_count = 0
    global_val_step_count = 0
    current_epoch=0
    should_stop = False
    train_start_time = time.perf_counter()
    if tensorsoket_consumer is not None:
        train_dataloader = tensorsoket_consumer

    if config.workload.limit_train_batches is None:
        config.workload.limit_train_batches = len(train_dataloader)
        
        
    while not should_stop:
        current_epoch += 1
        global_train_step_count = train_loop(
            fabric=fabric,
            job_id=config.job_id,
            train_logger=train_logger,
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            train_start_time=train_start_time,
            current_epoch=current_epoch,
            global_step_count=global_train_step_count,
            max_steps=config.workload.max_steps,
            limit_train_batches=config.workload.limit_train_batches,
            criterion=nn.CrossEntropyLoss(reduction = 'none'),
            tensorsocker_procuder=tensorsocket_procuder,
            tensorsocket_consumer=tensorsoket_consumer,
            sim=config.simulation_mode,
            sim_time=config.workload.gpu_time
            )
        
        if config.workload.max_steps is not None and global_train_step_count >= config.workload.max_steps:
            should_stop = True
        if config.workload.max_epochs is not None and current_epoch >= config.workload.max_epochs:
            should_stop = True

    # if not isinstance(train_dataloader, TensorConsumer):
    #     train_dataloader.sampler.send_job_ended_notfication()
    
    if config.dataloader.name == 'tensorsocket' and config.dataloader.mode == 'producer':
        tensorsocket_procuder.join() #shutdown the producer

    elapsed_time = time.perf_counter() - train_start_time
    fabric.print(f"Training completed in {elapsed_time:.2f} seconds")


def image_transform(
    image_size: int = 384,
    scale: Tuple[float, float] = (0.5, 1.0),
    image_interpolation=transforms.InterpolationMode.BICUBIC,
    mean: Tuple[float, float, float] = MEAN,
    std_dev: Tuple[float, float, float] = STD_DEV,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size, scale=scale, interpolation=image_interpolation
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(2, 7),
            transforms.ToTensor(),
            transforms.Normalize(mean, std_dev),
        ]
    )


def train_loop(fabric:Fabric, 
               job_id, 
               train_logger:CSVLogger,
               model,
               optimizer,
               train_dataloader:DataLoader,
               train_start_time,
               current_epoch,
               global_step_count,
               max_steps = None, 
               limit_train_batches = np.inf, 
               criterion=nn.CrossEntropyLoss(),
               tensorsocker_procuder:TensorProducer=None,
               tensorsocket_consumer:TensorConsumer=None,
               sim=False,
               sim_time=0):
    
    model.train()
    total_samples = 0
    total_train_loss = 0.0
    correct_preds = 0
    alpha = 0.4
    end = time.perf_counter()
    if tensorsocker_procuder is not None:
        for i, _ in enumerate(tensorsocker_procuder):
            #dont do anything as the producer will send the data to gpu of the consumers
            time.sleep(0.001)
    else:
        to_enmerate = tensorsocket_consumer if tensorsocket_consumer is not None else train_dataloader
        for batch_idx, (batch, data_load_time, transformation_time, is_cache_hit, cached_on_miss) in enumerate(to_enmerate):
            wait_for_data_time = time.perf_counter() - end
            if limit_train_batches is not None and batch_idx >= limit_train_batches:
                break
        
            if isinstance(to_enmerate, TensorConsumer):
                images, captions, text_atts, image_ids = batch
                batch_id = batch_idx
            else:
                images, captions, text_atts, image_ids, batch_id = batch
            
            if fabric.device.type == 'cuda':
                    torch.cuda.synchronize()
                    #remove batch from GPU
                    # if isinstance(train_dataloader, TensorConsumer):
                    #     #needed to free the memory for the producer to send more data
                    #     images = images.cpu()
                    #     captions = captions.cpu()
                    #     text_atts = text_atts.cpu()
                    #     image_ids = image_ids.cpu()

            gpu_processing_started = time.perf_counter()
            if sim:
                    time.sleep(sim_time)
            else:
                loss = model(images, captions, text_atts, image_ids, alpha, is_train=True)
                # Backpropagation and optimization
                optimizer.zero_grad()  # Clear previous gradients
                fabric.backward(loss)  # Backpropagation
                optimizer.step()  # Update weights
                total_train_loss += loss.item() * images.size(0)

                # if fabric.device.type == 'cuda':
                #         torch.cuda.synchronize()

            # Track time taken for GPU processing
            gpu_processing_time = time.perf_counter() - gpu_processing_started
            # Metrics calculation
            total_samples += images.size(0)
            avg_train_loss = total_train_loss / total_samples if not sim else 0
            global_step_count +=1
            cache_hit_samples = batch[0].size(0) if is_cache_hit == True else 0
            cache_hit_bacth = 1 if is_cache_hit == True else 0

            # if not isinstance(train_dataloader, TensorConsumer):
            #     train_dataloader.sampler.send_job_update_to_super(
            #         batch_id,
            #         data_load_time,
            #         is_cache_hit,
            #         gpu_processing_time,
            #         cached_on_miss)

            
            metrics= OrderedDict({
                                "Batch Id": batch_id,
                                "Elapsed Time (s)": time.perf_counter() - train_start_time,
                                # "Num Torch Workers": train_dataloader.num_workers,
                                "Device": fabric.global_rank,
                                "Epoch Index": current_epoch,
                                "Batch Index": batch_idx+1,
                                "Batch Size": batch[0].size(0),
                                "Iteration Time (s)": time.perf_counter() - end,
                                "Wait for Data Time (s)": wait_for_data_time,
                                "GPU Processing Time (s)": gpu_processing_time,
                                "Data Load Time (s)": data_load_time,
                                "Transformation Time (s)": transformation_time,
                                "Cache_Hit (Batch)": cache_hit_bacth,
                                "Cache_Hits (Samples)": cache_hit_samples,
                                "Cache_Size": 0,
                                "Cache_Memory (Mb)": 0,
                                "Train Loss (Avg)": avg_train_loss, #calculates the average training loss across all batches.
                                # "Train Accuracy (Avg)": avg_train_acc, #calculates the average training accuracy across all batches.
                                "Timestamp (UTC)": datetime.now(timezone.utc)  # Adds UTC timestamp
                                })
            train_logger.log_metrics(metrics,step=global_step_count)
            fabric.print(
                        f" Job {job_id} | Epoch:{metrics['Epoch Index']}({metrics['Batch Index']}/{min(len(train_dataloader),limit_train_batches)}) |"
                        # f" loss train: {metrics['Train Loss']:.3f} |"
                        # f" val: {val_loss} |"
                        # f" batch:{batch_id} |"
                        f" iter:{metrics['Iteration Time (s)']:.2f}s |"
                        f" delay:{metrics['Wait for Data Time (s)']:.2f}s |"
                        f" fetch:{metrics['Data Load Time (s)']:.2f}s |"
                        f" transform:{metrics['Transformation Time (s)']:.2f}s |"
                        f" gpu:{metrics['GPU Processing Time (s)']:.2f}s |"
                        f" elapsed:{metrics['Elapsed Time (s)']:.2f}s |"
                        f" loss: {metrics['Train Loss (Avg)']:.3f} |"
                        # f" acc: {metrics['Train Accuracy (Avg)']:.3f} |"
                        F" cache hit: {metrics['Cache_Hit (Batch)']} |"
                        )
            
            # stopping criterion on step level
            if max_steps is not None and global_step_count >= max_steps:
                break
            end = time.perf_counter()
   
    return global_step_count

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(config: DictConfig):
    log_dir = f"{config.log_dir}/{config.workload.name}/{config.job_id}".lower()

    log_dir = os.path.normpath(log_dir)  # Normalize path for Windows
    train_logger = CSVLogger(root_dir=log_dir, name="train", prefix='', flush_logs_every_n_steps=config.log_interval)
    val_logger = CSVLogger(root_dir=log_dir, name="val", prefix='', flush_logs_every_n_steps=config.log_interval)
    #cree log dir if does not exist
    # os.makedirs(log_dir, exist_ok=True)
    #save config
    finetune_multi_modal(config, train_logger,val_logger)
    save_hparams_to_yaml(os.path.join(log_dir, "hparms.yaml"), config)

if __name__ == "__main__":
    main()
