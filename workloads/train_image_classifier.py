import torch
import sys
print(sys.path)
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler
import hydra
from omegaconf import DictConfig
from lightning.fabric import Fabric, seed_everything
from lightning.fabric.loggers import CSVLogger
from torchvision.models import get_model
import time
import os
from collections import OrderedDict
import numpy as np
from datetime import datetime, timezone
import timm
from workloads.baselines.tensorsocket.tensorsocket_imagenet_dataset import TensorSocketImageNetDataset
from baselines.tensorsocket.producer import TensorProducer
from baselines.tensorsocket.consumer import TensorConsumer
from baselines.tensorsocket.tensorsocket_sampler import TensorSocketSampler
from lightning.pytorch.core.saving import save_hparams_to_yaml
from disdl.disdl_client import DisDLClient
from disdl.disdl_iterable_dataset import DisDLIterableDataset

def train_image_classifer(config: DictConfig,  train_logger: CSVLogger, val_logger: CSVLogger):
    if config.simulation_mode:
        config.accelerator = 'cpu'
        
    fabric = Fabric(accelerator=config.accelerator, devices=config.devices, precision=config.workload.precision)

    if config.seed is not None:
        seed_everything(config.seed) 
    else:
        seed_everything(config.job_id) # instead of torch.manual_seed(...)

    if config.workload.model_architecture == 'levit_128':
        model = timm.create_model('levit_128', pretrained=False, num_classes=config.workload.num_classes)
    elif config.workload.model_architecture == 'vit_small_patch32_224':
        model = timm.create_model('vit_small_patch32_224', pretrained=False, num_classes=config.workload.num_classes)
    # elif config.workload.model_architecture == 'vit_b_32':
    #     model = timm.create_model('vit_base_patch32_384', pretrained=False)
    elif config.workload.model_architecture == 'mixer_b32_224':
        model = timm.create_model('mixer_b32_224', pretrained=False, num_classes=config.workload.num_classes)
    else:
        model = get_model(name=config.workload.model_architecture, weights=None, num_classes=config.workload.num_classes)
    
    optimizer = optim.Adam(model.parameters(), lr=config.workload.learning_rate)
    model, optimizer = fabric.setup(model, optimizer)

    train_transform, val_transform = get_transforms(config.workload.name)

    train_dataloader = None
    val_dataloader = None
    tensorsocket_procuder:TensorProducer = None
    tensorsoket_consumer:TensorConsumer = None

    if config.dataloader.name == 'disdl':
        client = DisDLClient(address=config.dataloader.grpc_server_address)
        job_id, dataset_info = client.registerJob(dataset_location=config.workload.s3_train_prefix)
        num_batchs = dataset_info["num_batches"]
        client.close()

        train_dataset = DisDLIterableDataset(
            job_id=client.job_id,
            dataset_location=client.dataset_location,
            num_samples=num_batchs,
            batch_size=config.workload.batch_size,
            transform=train_transform,
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
            train_dataset = TensorSocketImageNetDataset(
                s3_data_dir=config.workload.s3_train_prefix,
                transform=train_transform,)
            
            tensor_socket_sampler = TensorSocketSampler(
                data_source=train_dataset,
                batch_size=config.workload.batch_size)
            
            train_dataloader = DataLoader(
                train_dataset,
                sampler=tensor_socket_sampler,
                batch_size=None,
                num_workers=config.workload.num_pytorch_workers,
                pin_memory=True if config.accelerator != 'cpu' else False)
            
            train_dataloader = fabric.setup_dataloaders(train_dataloader,  move_to_device=True if config.accelerator != 'cpu' else False)

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
            criterion=nn.CrossEntropyLoss(reduction = 'none'), # if isinstance(train_dataloader.sampler, ShadeSampler) else nn.CrossEntropyLoss(),
            tensorsocker_procuder=tensorsocket_procuder,
            tensorsocket_consumer=tensorsoket_consumer,
            sim=config.simulation_mode,
            sim_time=config.workload.gpu_time)
    
        if val_dataloader is not None and current_epoch % config.workload.validation_frequency == 0:
            global_val_step_count=  validate_loop(fabric, config.job_id,val_logger,model, val_dataloader,
                                                  train_start_time, 
                                                  current_epoch, 
                                                  global_val_step_count, 
                                                  config.workload.limit_val_batches)
  
        if current_epoch % config.workload.checkpoint_frequency == 0:
            checkpoint = {'epoch': current_epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict()}
            fabric.save(os.path.join(config.checkpoint_dir, f"epoch-{current_epoch:04d}.ckpt"), checkpoint)
        
        if config.workload.max_steps is not None and global_train_step_count >= config.workload.max_steps:
            should_stop = True
        if config.workload.max_epochs is not None and current_epoch >= config.workload.max_epochs:
            should_stop = True

    if not isinstance(train_dataloader, TensorConsumer):
        train_dataloader.sampler.send_job_ended_notfication()
    
    if config.dataloader.name == 'tensorsocket' and config.dataloader.mode == 'producer':
        tensorsocket_procuder.join() #shutdown the producer

    elapsed_time = time.perf_counter() - train_start_time
    fabric.print(f"Training completed in {elapsed_time:.2f} seconds")
    # metric_collector.stop()

def get_transforms():    
    # Set up data transforms for ImageNet on transformer workloads
    train_transform = transforms.Compose([
        transforms.Resize(256), 
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly change brightness, contrast, saturation, and hue
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform

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
    end = time.perf_counter()
    if tensorsocker_procuder is not None:
        for i, _ in enumerate(tensorsocker_procuder):
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
                inputs, labels = batch
                batch_id = batch_idx
            else:
                inputs, labels, batch_id = batch
            
            if fabric.device.type == 'cuda':
                torch.cuda.synchronize()
                #remove batch from GPU
                if isinstance(train_dataloader, TensorConsumer):
                    #need to free the memory for the producer to send more data
                    inputs = inputs.cpu()
                    labels = labels.cpu()

            
            # print(inputs)

            # Forward pass: Compute model output and loss
            gpu_processing_started = time.perf_counter()
            if sim:
                time.sleep(sim_time)
            else:
                outputs  = model(inputs)
                item_loss = criterion(outputs, labels)
                loss = item_loss.mean()

                # Backpropagation and optimization
                optimizer.zero_grad()  # Clear previous gradients
                fabric.backward(loss)  # Backpropagation
                optimizer.step()  # Update weights

                # Accumulate metrics directly on GPU to avoid synchronization
                correct_preds += (outputs.argmax(dim=1) == labels).sum().item()  # No .item(), stays on GPU
                total_train_loss += loss.item() * inputs.size(0)  # Convert loss to CPU for accumulation
                    
                # if fabric.device.type == 'cuda':
                #     torch.cuda.synchronize()

            # Track time taken for GPU processing
            gpu_processing_time = time.perf_counter() - gpu_processing_started
            # Metrics calculation
            total_samples += inputs.size(0)
            avg_train_loss = total_train_loss / total_samples if not sim else 0
            avg_train_acc = correct_preds / total_samples if not sim else 0
            # Calculate average loss and accuracy across all batches
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
                            "Train Accuracy (Avg)": avg_train_acc, #calculates the average training accuracy across all batches.
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
                    f" acc: {metrics['Train Accuracy (Avg)']:.3f} |"
                    F" cache hit: {metrics['Cache_Hit (Batch)']} |"
                    )

            # stopping criterion on step level
            if max_steps is not None and global_step_count >= max_steps:
                break

            end = time.perf_counter()
            
    return  global_step_count


def validate_loop(fabric,job_id, val_logger:CSVLogger, model, dataloader, val_start_time, current_epoch, global_step_count, limit_val_batches=np.inf, criterion=nn.CrossEntropyLoss()):
    model.eval()
    end = time.perf_counter()

    total_val_loss = 0.0
    correct_preds = 0
    total_samples = 0

    for batch_idx, (batch, data_load_time, transformation_time, is_cache_hit) in enumerate(dataloader):
        if batch_idx >= limit_val_batches:
            break
        
        inputs, labels = batch

        # Forward pass
        gpu_processing_started = time.perf_counter()
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        gpu_processing_time = time.perf_counter() - gpu_processing_started

        # Metrics calculation
        val_loss = loss.item()
        val_acc = (outputs.argmax(dim=1) == labels).float().mean().item()
        
        total_val_loss += val_loss * inputs.size(0)  # accumulate total loss
        correct_preds += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += inputs.size(0)

        # Calculate average loss and accuracy
        avg_val_loss = total_val_loss / total_samples #Calculates the average validation loss across all batches.
        avg_val_acc = correct_preds / total_samples #Calculates the average validation accuracy across all batches.
        global_step_count +=1

        metrics = OrderedDict({
            "Elapsed Time (s)": time.perf_counter() - val_start_time,
            "Device": fabric.global_rank,
            "Epoch Index": current_epoch,
            "Batch Index": batch_idx + 1,
            "Batch Size": batch[0].size(0),
            "Total Iteration Time (s)": time.perf_counter() - end,
            "Data Fetch Time (s)": data_load_time,
            "Transformation Time (s)": transformation_time,
            "Cache Hit/Miss": 1 if is_cache_hit else 0,
            "Avg Validation Loss": avg_val_loss,
            "Avg Validation Accuracy": avg_val_acc
        })
        
        val_logger.log_metrics(metrics, step=global_step_count)

        fabric.print(
            f" Job {job_id} | Epoch {metrics['Epoch Index']}({metrics['Batch Index']}/{min(len(dataloader), limit_val_batches)}) |"
            f" iter time: {metrics['Total Iteration Time (s)']:.2f} |"
            f" dataload time: {metrics['Data Fetch Time (s)']:.2f} |"
            f" dataload time: {metrics['Data Fetch Time (s)']:.2f} |"
            f" gpu time: {metrics['Data Fetch Time (s)']:.2f} |"
            f" elapsed time: {metrics['Elapsed Time (s)']:.2f} |"
            f" val loss: {metrics['Avg Validation Loss']:.3f} |"
            f" val acc: {metrics['Avg Validation Accuracy']:.3f}"
        )

        end = time.perf_counter()

    return global_step_count


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
    train_image_classifer(config, train_logger,val_logger)


if __name__ == "__main__":
    main()
