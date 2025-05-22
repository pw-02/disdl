import os
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Dict

import hydra
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms
from omegaconf import DictConfig, OmegaConf

from lightning.fabric import Fabric, seed_everything
from lightning.fabric.loggers import CSVLogger
from lightning.pytorch.core.saving import save_hparams_to_yaml

from disdl.client.disdl_iterable_dataset import DISDLDataset
from disdl.client.minibatch_client import MiniBatchClient
from disdl.client.s3_loader_factory import S3LoaderFactory
from disdl.client.disk_loader_factory import DiskLoaderFactory
from baselines.coordl.coordl_dataset import CoorDLDataset
from baselines.coordl.coordl_sampler import CoorDLBatchSampler
from workloads.args import FullConfig, JobConfig, DisDLConfig, CoorDLConfig


def run_training_job(config: FullConfig, train_logger: CSVLogger, val_logger: CSVLogger):
    # Set up Fabric
    if config.simulation_mode:
        config.accelerator = "cpu"

    fabric = Fabric(accelerator=config.accelerator, devices=config.devices, precision=config.job.precision)
    
    if config.job.seed is not None:
        seed_everything(config.job.seed)

    # Model and optimizer
    model = get_model(model_arch=config.job.model_name, 
                      num_classes=config.job.num_classes, 
                      pretrained=False)

    optimizer = optim.Adam(model.parameters(), lr=config.job.learning_rate)
    model, optimizer = fabric.setup(model, optimizer)

    if config.dataloader.name == 'disdl':
        train_dataloader = setup_disdl_dataloader(config, fabric)  # optional: extract this logic into a helper
    elif config.dataloader.name == 'coordl':
        train_dataloader = setup_coordl_dataloader(config, fabric)
    else:
        raise ValueError("Invalid dataloader name")
    
    # Train loop metadata
    global_step = 0
    current_epoch = 0
    train_start_time = time.perf_counter()
    should_stop = False
    max_time = config.job.max_training_time_sec
    max_steps = config.job.max_training_steps 
    max_epochs = config.job.max_epochs
    sim_time = config.job.sim_gpu_time  if config.simulation_mode else None

    while not should_stop:
        current_epoch += 1
        global_step = train_loop(
            fabric=fabric,
            job_id=config.job.job_id,
            train_dataloader=train_dataloader,
            model=model,
            optimizer=optimizer, 
            criterion=nn.CrossEntropyLoss(),  #nn.CrossEntropyLoss(reduction = 'none')
            train_logger=train_logger,
            train_start_time=train_start_time,
            current_epoch=current_epoch,
            global_step_count=global_step,
            max_steps=max_steps,
            max_training_time=max_time,
            sim_time=sim_time
        )
        # # Save checkpoint
        # if current_epoch % config.checkpoint_frequency == 0:
        #     checkpoint = {
        #         "epoch": current_epoch,
        #         "model_state_dict": model.state_dict(),
        #         "optimizer_state_dict": optimizer.state_dict(),
        #     }
        #     fabric.save(os.path.join(config.checkpoint_dir, f"epoch-{current_epoch:04d}.ckpt"), checkpoint)

        # Exit conditions
        if max_steps is not None and global_step >= max_steps:
            should_stop = True
        if max_epochs is not None and current_epoch >= max_epochs:
            should_stop = True
        if max_time is not None and (time.perf_counter() - train_start_time) >= max_time:
            should_stop = True

    elapsed = time.perf_counter() - train_start_time
    fabric.print(f"Training finished after {elapsed:.2f} seconds.")

def train_loop(fabric:Fabric, 
               job_id,
               train_dataloader:DataLoader,
               model,
               optimizer,
               criterion,
               train_logger:CSVLogger,
               train_start_time,
               current_epoch,
               global_step_count,
               max_steps = None, 
               max_training_time = None,
               sim_time=None
):
    
    model.train()
    total_samples = 0
    total_train_loss = 0.0
    last_step_time  = time.perf_counter()
    
    for batch_idx, (batch, meta) in enumerate(train_dataloader, start=1):
        
        inputs, labels = batch
        wait_for_data_time = time.perf_counter() - last_step_time 
        if fabric.device.type == "cuda":
            torch.cuda.synchronize()
        
        gpu_start = time.perf_counter()
        if sim_time is not None:
            time.sleep(sim_time)
            loss = torch.tensor(0.0)
            gpu_time = time.perf_counter() - gpu_start
            acc = {"top1": 0.0, "top5": 0.0}

        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            fabric.backward(loss)
            optimizer.step()
            torch.cuda.synchronize()
            gpu_time = time.perf_counter() - gpu_start
            acc = compute_topk_accuracy(outputs, labels, topk=(1, 5))
        total_samples += inputs.size(0)
        total_train_loss += loss.item() * inputs.size(0)
        avg_loss = total_train_loss / total_samples
        global_step_count += 1
        elapsed = time.perf_counter() - train_start_time
        

        metrics = OrderedDict({
            "Epoch": current_epoch,
            "Batch Id": meta.batch_id,
            "Num Torch Workers": train_dataloader.num_workers,
            "Device": fabric.global_rank,
            "Batch Index": batch_idx,
            "Batch Size": inputs.size(0),
            "Iteration Time (s)": time.perf_counter() - last_step_time,
            "Wait for Data Time (s)": wait_for_data_time,
            "GPU Processing Time (s)": gpu_time,
            "Train Loss (Avg)": avg_loss,
            "Top-1 Accuracy": acc["top1"],
            "Top-5 Accuracy": acc["top5"],
            "Data Fetch Time (s)": meta.data_fetch_time,
            "Transform Time (s)": meta.preprocess_time,
            "GRPC Get (s)": meta.grpc_get_overhead,
            "GRPC Report (s)": meta.grpc_report_overhead,
            "Other Time (s)": meta.other,
            "Cache Hit (Batch)": int(meta.cache_hit),
            "Timestamp (UTC)": datetime.now(timezone.utc),
            "Elapsed Time (s)": elapsed,
        })
        train_logger.log_metrics(metrics, step=global_step_count)
        fabric.print(
                    f" Job {job_id} | Epoch:{metrics['Epoch']}({metrics['Batch Index']}/{len(train_dataloader)}) |"
                    f" Batch:{metrics['Batch Id']} |"
                    f" iter:{metrics['Iteration Time (s)']:.2f}s |"
                    f" gpu:{metrics['GPU Processing Time (s)']:.2f}s |"
                    f" delay:{metrics['Wait for Data Time (s)']:.2f}s |"
                    f" fetch:{metrics['Data Fetch Time (s)']:.2f}s |"
                    f" transform:{metrics['Transform Time (s)']:.2f}s |"
                    f" elapsed:{metrics['Elapsed Time (s)']:.2f}s |"
                    # f" loss: {metrics['Train Loss (Avg)']:.3f} |"
                    # f" acc: {metrics['Train Accuracy (Avg)']:.3f} |"
                    f" hit: {metrics['Cache Hit (Batch)']} |"
                    f" worker: {meta.worker_id} |"
                    )

        if (max_training_time and elapsed >= max_training_time) or (max_steps and global_step_count >= max_steps):
            break

        if batch_idx >= len(train_dataloader):
            # fabric.print(f"Completed epoch {current_epoch} with {global_step_count} steps.")
            break


        last_step_time  = time.perf_counter()
    return global_step_count


def compute_topk_accuracy(outputs: torch.Tensor, targets: torch.Tensor, topk=(1, 5)) -> Dict[str, float]:
    """
    Computes the top-k accuracy for the specified values of k.
    Returns a dict like {'top1': ..., 'top5': ...}
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        # Get top-k predictions
        _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()  # shape [maxk, batch_size]
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        accuracies = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            accuracies[f"top{k}"] = (correct_k / batch_size).item()
        return accuracies


def get_transform(dataset_name: str, is_training: bool = True):
    if 'imagenet' in dataset_name.lower():
        return transforms.Compose([
            transforms.Resize(256), 
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    
    elif 'cifar10' in dataset_name.lower():
        return transforms.Compose([
            transforms.Resize(256), 
            transforms.RandomResizedCrop(224),

            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2470, 0.2435, 0.2616]),
        ])
    
    elif 'openimages' in dataset_name.lower():  # consistent naming
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    
    else:
        raise ValueError(f"No transform defined for dataset at: {dataset_name}")


def get_dataset_loader(config: FullConfig):
    if 's3:' in config.job.dataset_dir:
        return S3LoaderFactory.create(
            dataset_name=config.job.dataset_name,
            dataset_location=config.job.dataset_dir,
            transform=get_transform(config.job.dataset_name),
        )
    else:
        return DiskLoaderFactory.create(
            dataset_name=config.job.dataset_name,
            dataset_location=config.job.dataset_dir,
            transform=get_transform(config.job.dataset_name),
        )


def setup_coordl_dataloader(config: FullConfig, fabric: Fabric):
   
     # Create iterable dataset
    train_dataset = CoorDLDataset(
        job_id=config.job.job_id,
        total_jobs=config.num_jobs,
        s3_loader=get_dataset_loader(config),
        redis_host=config.dataloader.cache_host,
        redis_port=config.dataloader.cache_port,
        ssl=config.dataloader.ssl_enabled,
        use_compression=config.dataloader.use_compression,
        syncronized_mode=config.dataloader.syncronized_mode
    )
    train_sampler = CoorDLBatchSampler(
        data_source=train_dataset,
        batch_size=config.job.batch_size,
        job_idx=config.job.job_id,
        num_jobs=config.num_jobs ,
        shuffle=config.dataloader.shuffle,
        drop_last=config.dataloader.drop_last,
        seed=config.job.seed
    )
     # Wrap in PyTorch DataLoader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=None,  # CoorDLDataset yields full batches
        num_workers=config.dataloader.num_workers,
        pin_memory=config.accelerator != "cpu"
    )
     # Fabric handles device placement if needed
    train_dataloader = fabric.setup_dataloaders(
        train_dataloader, move_to_device=config.accelerator != "cpu"
    )
    return train_dataloader




def setup_disdl_dataloader(config: FullConfig, fabric: Fabric):
    client = MiniBatchClient(address=config.dataloader.grpc_server)
    grpc_job_id, dataset_info = client.register_job(dataset_name=config.dataset_name,
                                                    processing_speed= config.job.sim_gpu_time)
    # config.job_id = grpc_job_id
    client.close()
   
     # Create iterable dataset
    train_dataset = DISDLDataset(
        job_id=grpc_job_id,
        dataset_name=config.dataset_name,
        grpc_address=config.dataloader.grpc_server,
        s3_loader=get_dataset_loader(config),
        redis_host=config.dataloader.cache_host,
        redis_port=config.dataloader.cache_port,
        num_batches_per_epoch=dataset_info['num_batches'],
        use_compression=config.dataloader.use_compression,
    )
     # Wrap in PyTorch DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=None,  # DISDLDataset yields full batches
        num_workers=config.dataloader.num_workers,
        pin_memory=config.accelerator != "cpu",
        persistent_workers=True if config.dataloader.num_workers > 0 else False,
    )
     # Fabric handles device placement if needed
    train_dataloader = fabric.setup_dataloaders(
        train_dataloader, move_to_device=config.accelerator != "cpu"
    )
    return train_dataloader



def get_model(model_arch: str, num_classes: int, pretrained: bool = False):
    from torchvision.models import get_model as get_torchvision_model
    from torchvision.models import list_models as list_torchvision_models

    if model_arch == "albef_retrieval":
        raise NotImplementedError("ALBEF model is not implemented yet.")

    if model_arch not in timm.list_models():
        raise ValueError(f"Unsupported model architecture: '{model_arch}'")

    model = timm.create_model(
        model_name=model_arch,
        pretrained=pretrained,
        num_classes=num_classes
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"{model_arch} - Total Parameters: {num_params:,}")
    return model


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig):

    # cfg = OmegaConf.to_object(config)  # or manually create FullConfig if you want strict typing
    dataloader_config = None

    if cfg.dataloader.name == 'disdl':
        dataloader_config = DisDLConfig(
            name=cfg.dataloader.name,
            cache_host=cfg.dataloader.cache_host,
            cache_port=cfg.dataloader.cache_port,
            grpc_server=cfg.dataloader.grpc_server,
            ssl_enabled=cfg.dataloader.ssl_enabled,
            use_compression=cfg.dataloader.use_compression,
            num_workers=cfg.dataloader.num_torch_workers
        )
    elif cfg.dataloader.name == 'coordl':
        dataloader_config = CoorDLConfig(
            name=cfg.dataloader.name,
            cache_host=cfg.dataloader.cache_host,
            cache_port=cfg.dataloader.cache_port,
            syncronized_mode=cfg.dataloader.syncronized_mode,
            ssl_enabled=cfg.dataloader.ssl_enabled ,
            shuffle=cfg.dataloader.shuffle,
            drop_last=cfg.dataloader.drop_last,
            use_compression=cfg.dataloader.use_compression,
            num_workers=cfg.dataloader.num_torch_workers
        )
    # Create job config
    job_config = JobConfig(
        job_id=cfg.workload.job_id,
        gpu_id=cfg.workload.gpu_id,
        model_name=cfg.workload.model_name,
        dataset_name=cfg.workload.dataset_name,
        dataset_dir=cfg.workload.dataset_path,
        num_classes=cfg.workload.num_classes,
        learning_rate=cfg.workload.learning_rate,
        batch_size=cfg.workload.batch_size,
        weight_decay=cfg.workload.weight_decay,
        sim_gpu_time=cfg.workload.sim_gpu_time  if cfg.simulation_mode else None,
        max_training_time_sec=cfg.workload.max_training_time_sec,
        max_training_steps=cfg.workload.max_training_steps,
        max_epochs=cfg.workload.max_epochs,
        precision=cfg.workload.precision,
        seed=cfg.workload.seed
    )

    full_config = FullConfig(
    # log_dir = os.path.join(cfg.log_dir, cfg.workload.dataset_name, cfg.workload.workload_type , cfg.dataloader.name, job_config.model_name),
    log_dir = os.path.join(cfg.log_dir, f"{job_config.job_id}_{job_config.model_name}"),

    log_interval = cfg.log_interval,
    dataloader_name = cfg.dataloader.name,
    dataset_name = cfg.workload.dataset_name,
    accelerator= cfg.accelerator,
    simulation_mode = cfg.simulation_mode,
    job = job_config,
    dataloader= dataloader_config,
    devices = cfg.devices,
    num_jobs = cfg.num_jobs,
    checkpoint_frequency= cfg.checkpoint_frequency)

    train_logger = CSVLogger(root_dir=full_config.log_dir, name="train", prefix='', flush_logs_every_n_steps=full_config.log_interval)
    val_logger = CSVLogger(root_dir=full_config.log_dir, name="val", prefix='', flush_logs_every_n_steps=full_config.log_interval)
    os.makedirs(full_config.log_dir, exist_ok=True)
    save_hparams_to_yaml(os.path.join(full_config.log_dir, "hparms.yaml"), cfg)

    # log_dir = f"{config.log_dir}/{config.workload.name}/{config.job_id}".lower()
    # log_dir = os.path.normpath(log_dir)  # Normalize path for Windows
    # log_dir = os.path.join(config.log_dir, config.workload.workload_name, config.workload.dataset.dataset_name, config.dataloader.name, config.workload.model_name)
    # train_logger = CSVLogger(root_dir=log_dir, name="train", prefix='', flush_logs_every_n_steps=config.log_interval)
    # val_logger = CSVLogger(root_dir=log_dir, name="val", prefix='', flush_logs_every_n_steps=config.log_interval)
    #cree log dir if does not exist
    #save
    run_training_job(full_config, train_logger,val_logger)


if __name__ == "__main__":
    main()
