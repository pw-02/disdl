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

from lightning.fabric import Fabric, seed_everything
from lightning.fabric.loggers import CSVLogger
from lightning.pytorch.core.saving import save_hparams_to_yaml

from disdl.disdl_iterable_dataset import DISDLDataset
from disdl.minibatch_client import MiniBatchClient
from disdl.s3_loader_factory import S3LoaderFactory

def run_training_job(config: DictConfig, train_logger: CSVLogger, val_logger: CSVLogger):
    # Set up Fabric
    if config.simulation_mode:
        config.accelerator = "cpu"

    fabric = Fabric(accelerator=config.accelerator, devices=config.devices, precision=config.workload.precision)
    
    if config.seed is not None:
        seed_everything(config.seed)

    # Model and optimizer
    model = get_model(config=config)

    optimizer = optim.Adam(model.parameters(), lr=config.workload.learning_rate)
    model, optimizer = fabric.setup(model, optimizer)

    if config.dataloader.name == 'disdl':
        train_dataloader = setup_disdl_dataloader(config, fabric)  # optional: extract this logic into a helper
    else:
        raise ValueError("Invalid dataloader name")
    
    # Train loop metadata
    global_step = 0
    current_epoch = 0
    train_start_time = time.perf_counter()
    should_stop = False
    max_time = config.workload.max_training_time_sec
    max_steps = config.workload.max_steps
    max_epochs = config.workload.max_epochs
    sim_time = config.workload.sim_time

    while not should_stop:
        current_epoch += 1
        global_step = train_loop(
            fabric=fabric,
            job_id=config.job_id,
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
        # Save checkpoint
        if current_epoch % config.workload.checkpoint_frequency == 0:
            checkpoint = {
                "epoch": current_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            fabric.save(os.path.join(config.checkpoint_dir, f"epoch-{current_epoch:04d}.ckpt"), checkpoint)

        # Exit conditions
        if (
            (config.workload.max_steps and global_step >= config.workload.max_steps)
            or (config.workload.max_epochs and current_epoch >= config.workload.max_epochs)
            or (max_time and (time.perf_counter() - train_start_time) >= max_time)
        ):
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
               sim_time=None):
    
    model.train()
    total_samples = 0
    total_train_loss = 0.0
    last_step_time  = time.perf_counter()
    
    for batch_idx, (batch, meta) in enumerate(train_dataloader, start=1):
        
        inputs, labels = batch
        wait_for_data_time = time.perf_counter() - last_step_time 
        if fabric.device.type == "cuda":
            torch.cuda.synchronize()

        if sim_time is not None:
            time.sleep(sim_time)
            loss = torch.tensor(0.0)
            gpu_time = sim_time
        else:
            gpu_start = time.perf_counter()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            fabric.backward(loss)
            optimizer.step()
            gpu_time = time.perf_counter() - gpu_start
        
        total_samples += inputs.size(0)
        total_train_loss += loss.item() * inputs.size(0)
        avg_loss = total_train_loss / total_samples
        global_step_count += 1
        elapsed = time.perf_counter() - train_start_time
        acc = compute_topk_accuracy(outputs, labels, topk=(1, 5))

        metrics = OrderedDict({
            "Epoch": current_epoch,
            "Batch Id": meta.batch_id,
            "Batch Index": batch_idx,
            "Batch Size": inputs.size(0),
            "Train Loss (Avg)": avg_loss,
            "Top-1 Accuracy": acc["top1"],
            "Top-5 Accuracy": acc["top5"],
            "GPU Processing Time (s)": gpu_time,
            "Wait for Data Time (s)": wait_for_data_time,
            "Data Load Time (s)": meta.data_fetch_time,
            "Transform Time (s)": meta.preprocess_time,
            "Cache Time (s)": meta.cache_time,
            "Cache Hit (Batch)": int(meta.cache_hit),
            "Timestamp (UTC)": datetime.now(timezone.utc),
        })
        train_logger.log_metrics(metrics, step=global_step_count)
        fabric.print(
            f"Job {job_id} | [{current_epoch}:{batch_idx}] batch_id={meta.batch_id} "
            f"loss={avg_loss:.3f} gpu={gpu_time:.2f}s fetch={meta.data_fetch_time:.2f}s "
            f"preprocess={meta.preprocess_time:.2f}s cache_hit={meta.cache_hit}, elapsed={elapsed:.2f}s "
        )

        if (max_training_time and elapsed >= max_training_time) or (max_steps and global_step_count >= max_steps):
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


def get_transform(dataset_location: str, is_training: bool = True):
    if 'imagenet' in dataset_location.lower():
        return transforms.Compose([
            transforms.Resize(256), 
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    
    elif 'cifar10' in dataset_location.lower():
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2470, 0.2435, 0.2616]),
        ])
    
    elif 'openimages' in dataset_location.lower():  # consistent naming
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    
    else:
        raise ValueError(f"No transform defined for dataset at: {dataset_location}")


def setup_disdl_dataloader(config: DictConfig, fabric: Fabric):
    client = MiniBatchClient(address=config.dataloader.grpc_server_address)
    job_id = client.register_job(dataset_name=config.workload.name)
    config.job_id = job_id
    client.close()
     # Set up S3 data loader
    s3_loader = S3LoaderFactory.create(
        dataset_name=config.workload.name,
        dataset_location=config.workload.dataset_location,
        transform=get_transform(config.workload.dataset_location)
    )
     # Create iterable dataset
    train_dataset = DISDLDataset(
        job_id=job_id,
        dataset_name=config.workload.name,
        grpc_address=config.dataloader.grpc_server_address,
        s3_loader=s3_loader,
        redis_host=config.cache.redis_host,
        redis_port=config.cache.redis_port
    )
     # Wrap in PyTorch DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=None,  # DISDLDataset yields full batches
        num_workers=config.workload.num_pytorch_workers,
        pin_memory=config.accelerator != "cpu"
    )
     # Fabric handles device placement if needed
    train_dataloader = fabric.setup_dataloaders(
        train_dataloader, move_to_device=config.accelerator != "cpu"
    )
    return train_dataloader



def get_model(config: DictConfig):
    model_arch = config.workload.model_architecture

    if model_arch == "albef_retrieval":
        raise NotImplementedError("ALBEF model is not implemented yet.")

    if model_arch not in timm.list_models():
        raise ValueError(f"Unsupported model architecture: '{model_arch}'")

    model = timm.create_model(
        model_name=model_arch,
        pretrained=False,
        num_classes=config.workload.num_classes
    )
    print_model_stats(model, model_arch)
    return model








def print_model_stats(model, model_name=""):
    num_params = sum(p.numel() for p in model.parameters())
    print(f"{model_name} - Total Parameters: {num_params:,}")

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(config: DictConfig):

    # log_dir = f"{config.log_dir}/{config.workload.name}/{config.job_id}".lower()
    # log_dir = os.path.normpath(log_dir)  # Normalize path for Windows
    log_dir = os.path.join(config.log_dir, config.workload.model_architecture)
    train_logger = CSVLogger(root_dir=log_dir, name="train", prefix='', flush_logs_every_n_steps=config.log_interval)
    val_logger = CSVLogger(root_dir=log_dir, name="val", prefix='', flush_logs_every_n_steps=config.log_interval)
    #cree log dir if does not exist
    os.makedirs(log_dir, exist_ok=True)
    #save config
    save_hparams_to_yaml(os.path.join(log_dir, "hparms.yaml"), config)
    run_training_job(config, train_logger,val_logger)


if __name__ == "__main__":
    main()
