import torch
import sys
print(sys.path)
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
from lightning.fabric import Fabric, seed_everything
from lightning.fabric.loggers import CSVLogger
from torchvision.models import get_model as get_torchvision_model
from torchvision.models import list_models as list_torchvision_models

import time
import os
from collections import OrderedDict
import numpy as np
from datetime import datetime, timezone
import timm
from baselines.tensorsocket.tensorsocket_openimages_dataset import TensorSocketOpenImagesDataset
from baselines.tensorsocket.tensorsocket_coco_dataset import TensorSocketCocoDataset
from baselines.tensorsocket.tensorsocket_imagenet_dataset import TensorSocketImageNetDataset

from baselines.tensorsocket.producer import TensorProducer
from baselines.tensorsocket.consumer import TensorConsumer
from baselines.tensorsocket.tensorsocket_sampler import TensorSocketSampler
from lightning.pytorch.core.saving import save_hparams_to_yaml
from disdl.disdl_client import DisDLClient
from disdl.disdl_iterable_dataset import DisDLImageNetIterableDataset, DisDLOpenImagesDataset, DisDLCocoIterableDataset
from models.albef import albef_model_for_retrieval, albef_image_transform, ALBEFTextTransform


def run_training_job(config: DictConfig, train_logger: CSVLogger, val_logger: CSVLogger):
    if config.simulation_mode:
        config.accelerator = 'cpu'
        
    fabric = Fabric(accelerator=config.accelerator, devices=config.devices, precision=config.workload.precision)
    if config.seed is not None:
        seed_everything(config.seed) 
    else:
        seed_everything(config.job_id) # instead of torch.manual_seed(...)

    model = get_model(config=config)
    optimizer = optim.Adam(model.parameters(), lr=config.workload.learning_rate)
    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader = None
    val_dataloader = None
    tensorsocket_producer:TensorProducer = None
    tensorsocket_consumer:TensorConsumer = None

    if config.dataloader.name == 'disdl':
        client = DisDLClient(address=config.dataloader.grpc_server_address)
        job_id, dataset_info = client.registerJob(dataset_location=config.workload.s3_train_prefix)
        num_batchs = dataset_info["num_batches"]
        client.close()
        train_dataset = get_disdl_dataset(config, client, num_batchs)
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=None,
            num_workers=config.workload.num_pytorch_workers,
            pin_memory=True if config.accelerator != 'cpu' else False)
        train_dataloader = fabric.setup_dataloaders(train_dataloader, move_to_device=True if config.accelerator != 'cpu' else False)
    
    elif config.dataloader.name == 'tensorsocket':
        if config.dataloader.mode == 'producer':
            print("Creating TensorSocket producer..")
            train_dataset = get_tensorsocker_dataset(config)
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
            
            tensorsocket_producer = TensorProducer(
                data_loader=train_dataloader,
                port=config.dataloader.producer_port,
                consumer_max_buffer_size=config.dataloader.consumer_maxbuffersize,
                ack_port=config.dataloader.producer_ackport,
                producer_batch_size=config.dataloader.producer_batch_size,
                )
        elif config.dataloader.mode == 'consumer':
            tensorsocket_consumer = TensorConsumer(
                port=config.dataloader.consumer_port,
                ack_port=config.dataloader.consumer_ackport,
                batch_size=config.workload.batch_size)
    else:
        raise ValueError("Invalid dataloader name")
    
    global_train_step_count = 0
    global_val_step_count = 0
    current_epoch=0
    should_stop = False
    train_start_time = time.perf_counter()
    
    if tensorsocket_consumer is not None:
        train_dataloader = tensorsocket_consumer

    if config.workload.limit_train_batches is None:
        config.workload.limit_train_batches = len(train_dataloader)
    
    while not should_stop:
        current_epoch += 1
        global_train_step_count = train_loop(
            fabric=fabric,
            job_id=config.job_id,
            dataset_location = config.workload.s3_train_prefix,
            train_logger=train_logger,
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            train_start_time=train_start_time,
            current_epoch=current_epoch,
            global_step_count=global_train_step_count,
            max_steps=config.workload.max_steps,
            limit_train_batches=config.workload.limit_train_batches,
            criterion=nn.CrossEntropyLoss(reduction = 'none'), # if isinstance(train_dataloader.sampler, ShadeSampler) else nn.CrossEntropyLoss(),
            tensorsocket_producer=tensorsocket_producer,
            tensorsocket_consumer=tensorsocket_consumer,
            sim=config.simulation_mode,
            sim_time=config.workload.gpu_time)
    
        # if val_dataloader is not None and current_epoch % config.workload.validation_frequency == 0:
        #     global_val_step_count=  validate_loop(fabric, config.job_id,val_logger,model, val_dataloader,
        #                                           train_start_time, 
        #                                           current_epoch, 
        #                                           global_val_step_count, 
        #                                           config.workload.limit_val_batches)
  
        if current_epoch % config.workload.checkpoint_frequency == 0:
            checkpoint = {'epoch': current_epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict()}
            fabric.save(os.path.join(config.checkpoint_dir, f"epoch-{current_epoch:04d}.ckpt"), checkpoint)
        
        if config.workload.max_steps is not None and global_train_step_count >= config.workload.max_steps:
            should_stop = True
        if config.workload.max_epochs is not None and current_epoch >= config.workload.max_epochs:
            should_stop = True

    # if not isinstance(train_dataloader, TensorConsumer):
    #     train_dataloader.sampler.send_job_ended_notfication()
    
    if config.dataloader.name == 'tensorsocket' and config.dataloader.mode == 'producer':
        tensorsocket_producer.join() #shutdown the producer

    elapsed_time = time.perf_counter() - train_start_time
    fabric.print(f"Training completed in {elapsed_time:.2f} seconds")

def train_loop(fabric:Fabric, 
               job_id,
               dataset_location,
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
               tensorsocket_producer:TensorProducer=None,
               tensorsocket_consumer:TensorConsumer=None,
               sim=False,
               sim_time=0):
    
    model.train()
    total_samples = 0
    total_train_loss = 0.0
    alpha = 0.4

    end = time.perf_counter()

    if tensorsocket_producer is not None:
        for i, _ in enumerate(tensorsocket_producer):
            #dont do anything as the producer will send the data to gpu of the consumers
            time.sleep(0.001)
    else:
        to_enmerate = tensorsocket_consumer if tensorsocket_consumer is not None else train_dataloader
        for batch_idx, (batch, data_load_time, transformation_time, is_cache_hit, cached_on_miss) in enumerate(to_enmerate):
            wait_for_data_time = time.perf_counter() - end
            # end epoch if stopping training completely or max batches for this epoch reached
            if limit_train_batches is not None and batch_idx >= limit_train_batches:
                break
            
            # Unpack batch
            if isinstance(to_enmerate, TensorConsumer):
                if 'imagenet' in dataset_location:
                    inputs, labels = batch
                elif 'disdlopenimages' in dataset_location:
                    inputs, labels = batch
                elif 'coco' in dataset_location:
                    inputs, captions, text_atts, image_ids = batch
                batch_id = batch_idx
            else:
                if 'coco' in dataset_location:
                    inputs, captions, text_atts, image_ids, batch_id = batch
                else:
                    inputs, labels, batch_id = batch
            
            if fabric.device.type == 'cuda':
                torch.cuda.synchronize()
                #remove batch from GPU
                # if isinstance(train_dataloader, TensorConsumer):
                #     #need to free the memory for the producer to send more data
                #     inputs = inputs.cpu()
                #     labels = labels.cpu()

            # Forward pass: Compute model output and loss
            gpu_processing_started = time.perf_counter()
            if sim:
                time.sleep(sim_time)
            else:
                error = False
                if 'coco' in dataset_location:
                    try:
                        loss = model(inputs, captions, text_atts, image_ids, alpha, is_train=True)
                    except Exception as e:
                        error = True
                        print(e)
                else:
                    outputs  = model(inputs)
                    item_loss = criterion(outputs, labels)
                    loss = item_loss.mean()

                # Backpropagation and optimization
                optimizer.zero_grad()  # Clear previous gradients
                fabric.backward(loss)  # Backpropagation
                optimizer.step()  # Update weights
                total_train_loss += loss.item() * inputs.size(0)  # Convert loss to CPU for accumulation
                
                # Accumulate metrics directly on GPU to avoid synchronization
                # correct_preds += (outputs.argmax(dim=1) == labels).sum().item()  # No .item(), stays on GPU
                   
            # Track time taken for GPU processing
            gpu_processing_time = time.perf_counter() - gpu_processing_started
            # Metrics calculation
            total_samples += inputs.size(0)
            avg_train_loss = total_train_loss / total_samples if not error else 0
            global_step_count +=1
            cache_hit_samples = batch[0].size(0) if is_cache_hit == True else 0
            cache_hit_bacth = 1 if is_cache_hit == True else 0

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
                            "Train Loss (Avg)": avg_train_loss, 
                            # "Train Accuracy (Avg)": avg_train_acc, 
                            "Timestamp (UTC)": datetime.now(timezone.utc)
                            })
            train_logger.log_metrics(metrics,step=global_step_count)
            
            fabric.print(
                    f" Job {job_id} | Epoch:{metrics['Epoch Index']}({metrics['Batch Index']}/{min(len(train_dataloader),limit_train_batches)}) |"
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
            
    return  global_step_count

def get_tansform(dataset_location):
    if 'imagenet' in dataset_location:
        train_transform = transforms.Compose([
                transforms.Resize(256), 
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly change brightness, contrast, saturation, and hue
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    elif 'disdlopenimages' in dataset_location:
        train_transform = transforms.Compose([
            transforms.Resize(256), 
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return train_transform

def get_tensorsocker_dataset(config: DictConfig):
    dataset_location = config.workload.s3_train_prefix
    if 'imagenet' in dataset_location:
        train_dataset = TensorSocketImageNetDataset(
            s3_data_dir=dataset_location,
            transform=get_tansform(dataset_location))
    
    if 'disdlopenimages' in dataset_location:
        train_dataset = TensorSocketOpenImagesDataset(
            s3_data_dir=config.workload.s3_train_prefix,
            transform=get_tansform(dataset_location))
        
    elif 'coco' in dataset_location:
        train_dataset = TensorSocketCocoDataset(
                        s3_data_dir=config.workload.s3_train_prefix,
                        image_transform=albef_image_transform(),
                        text_transform=ALBEFTextTransform(
                            truncate=True, 
                            pad_to_max_seq_len=True, 
                            max_seq_len=30, 
                            add_end_token=False))
    return train_dataset





def get_disdl_dataset(config: DictConfig, client: DisDLClient, num_batchs: int):
    dataset_location = config.workload.s3_train_prefix
    if 'imagenet' in dataset_location:
        train_dataset = DisDLImageNetIterableDataset(
            job_id=client.job_id,
            dataset_location=client.dataset_location,
            num_samples=num_batchs,
            batch_size=config.workload.batch_size,
            transform=get_tansform(dataset_location),
            disdl_service_address=config.dataloader.grpc_server_address,
            cache_address=config.dataloader.cache_address,
            ssl=config.dataloader.ssl_enabled,
            use_compression=config.dataloader.use_compression,
            use_local_folder=config.dataloader.use_local_folder
            )
    elif 'coco' in dataset_location:
       
        train_dataset = DisDLCocoIterableDataset(
            job_id=client.job_id,
            dataset_location=client.dataset_location,
            num_samples=num_batchs,
            batch_size=config.workload.batch_size,
            image_transform=albef_image_transform(),
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
        
    elif 'disdlopenimages' in dataset_location:
        train_dataset = DisDLOpenImagesDataset(
            job_id=client.job_id,
            dataset_location=client.dataset_location,
            num_samples=num_batchs,
            batch_size=config.workload.batch_size,
            transform=get_tansform(dataset_location),
            disdl_service_address=config.dataloader.grpc_server_address,
            cache_address=config.dataloader.cache_address,
            ssl=config.dataloader.ssl_enabled,
            use_compression=config.dataloader.use_compression,
            use_local_folder=config.dataloader.use_local_folder
            )
    return train_dataset


def get_model(config: DictConfig):
    model_architecture = config.workload.model_architecture
    if model_architecture in list_torchvision_models():
        model = get_torchvision_model(name=model_architecture, weights=None, num_classes=config.workload.num_classes)
    elif model_architecture == 'levit_128':
        model = timm.create_model('levit_128', pretrained=False, num_classes=config.workload.num_classes)
    elif model_architecture == 'vit_small_patch32_224':
        model = timm.create_model('vit_small_patch32_224', pretrained=False, num_classes=config.workload.num_classes)
    elif model_architecture == 'mixer_b32_224':
        model = timm.create_model('mixer_b32_224', pretrained=False, num_classes=config.workload.num_classes)
    elif model_architecture == 'albef_retrieval':
        model = albef_model_for_retrieval(config.workload, pretrained=False)
    
    return model


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(config: DictConfig):
    log_dir = f"{config.log_dir}/{config.workload.name}/{config.job_id}".lower()

    log_dir = os.path.normpath(log_dir)  # Normalize path for Windows
    train_logger = CSVLogger(root_dir=log_dir, name="train", prefix='', flush_logs_every_n_steps=config.log_interval)
    val_logger = CSVLogger(root_dir=log_dir, name="val", prefix='', flush_logs_every_n_steps=config.log_interval)
    #cree log dir if does not exist
    os.makedirs(log_dir, exist_ok=True)
    #save config
    save_hparams_to_yaml(os.path.join(log_dir, "hparms.yaml"), config)
    run_training_job(config, train_logger,val_logger)


if __name__ == "__main__":
    main()
