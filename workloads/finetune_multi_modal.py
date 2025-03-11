# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch.nn as nn
import redis
from collections import OrderedDict
import math
import os
import time
from typing import Any, List, Optional, Tuple, Union
import hydra
from omegaconf import DictConfig
from lightning.fabric import Fabric, seed_everything
from lightning.fabric.loggers import CSVLogger
import torch
from models.albef import albef_model_for_retrieval
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.nn import Module
import re
# from torch.nn import Sequential
from torchvision import transforms
from lightning.pytorch.core.saving import save_hparams_to_yaml

from baselines.tensorsocket.tensorsocket_sampler import TensorSocketSampler
from baselines.tensorsocket.tensorsocket_coco_dataset import TensorSocketCocoDataset
from baselines.tensorsocket.producer import TensorProducer
from baselines.tensorsocket.consumer import TensorConsumer
from transformers.models.bert.tokenization_bert import BertTokenizer

from datetime import datetime, timezone
import numpy as np
# mean and standard deviation from the ALBEF repo:
# https://github.com/salesforce/ALBEF/blob/main/dataset/__init__.py#L16
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD_DEV = (0.26862954, 0.26130258, 0.27577711)


class Truncate(Module):
    r"""Truncate input sequence
    :param max_seq_len: The maximum allowable length for input sequence
    :type max_seq_len: int
    """
    def __init__(self, max_seq_len: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sequence or batch of sequence to be truncated
        :type input: Union[List[Union[str, int]], List[List[Union[str, int]]]]
        :return: Truncated sequence
        :rtype: Union[List[Union[str, int]], List[List[Union[str, int]]]]
        """
        if torch.jit.isinstance(input, List[int]):
            return input[:self.max_seq_len]
        elif torch.jit.isinstance(input, List[str]):
            return input[:self.max_seq_len]
        elif torch.jit.isinstance(input, List[List[int]]):
            output: List[List[int]] = []
            for ids in input:
                output.append(ids[:self.max_seq_len])
            return output
        elif torch.jit.isinstance(input, List[List[str]]):
            output: List[List[str]] = []
            for ids in input:
                output.append(ids[:self.max_seq_len])
            return output
        else:
            raise TypeError("Input type not supported")


class Sequential(torch.nn.Sequential):
    r"""A container to host a sequence of text transforms."""

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sequence or batch. The input type must be supported by the first transform in the sequence.
        :type input: `Any`
        """
        for module in self:
            input = module(input)
        return input

class ToTensor(Module):
    r"""Convert input to torch tensor

    :param padding_value: Pad value to make each input in the batch of length equal to the longest sequence in the batch.
    :type padding_value: Optional[int]
    :param dtype: :class:`torch.dtype` of output tensor
    :type dtype: :class:`torch.dtype`
    """

    def __init__(self, padding_value: Optional[int] = None, dtype: torch.dtype = torch.long) -> None:
        super().__init__()
        self.padding_value = padding_value
        self.dtype = dtype

    def forward(self, input: Any) -> torch.Tensor:
        """
        :param input: Sequence or batch of token ids
        :type input: Union[List[int], List[List[int]]]
        :rtype: Tensor
        """
        if torch.jit.isinstance(input, List[int]):
            return torch.tensor(input, dtype=torch.long)
        elif torch.jit.isinstance(input, List[List[int]]):
            if self.padding_value is None:
                output = torch.tensor(input, dtype=self.dtype)
                return output
            else:
                output = pad_sequence(
                    [torch.tensor(ids, dtype=self.dtype) for ids in input], batch_first=True, padding_value=float(self.padding_value)
                )
                return output
        else:
            raise TypeError("Input type not supported")


class PadTransform(Module):
    """Pad tensor to a fixed length with given padding value.

    :param max_length: Maximum length to pad to
    :type max_length: int
    :param pad_value: Value to pad the tensor with
    :type pad_value: bool
    """

    def __init__(self, max_length: int, pad_value: int) -> None:
        super().__init__()
        self.max_length = max_length
        self.pad_value = float(pad_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: The tensor to pad
        :type x: Tensor
        :return: Tensor padded up to max_length with pad_value
        :rtype: Tensor
        """
        max_encoded_length = x.size(-1)
        if max_encoded_length < self.max_length:
            pad_amount = self.max_length - max_encoded_length
            x = torch.nn.functional.pad(x, (0, pad_amount), value=self.pad_value)
        return x

class ALBEFTextTransform:
    def __init__(
        self,
        pretrained_tokenizer: str = "bert-base-uncased",
        do_pre_process: bool = True,
        truncate: bool = False,
        pad_to_max_seq_len: bool = False,
        add_end_token: bool = True,
        max_seq_len: int = 25,
        cls_token_id: int = 101,
        sep_token_id: int = 102,
        pad_token_id: int = 0,
    ):
        self.do_pre_process = do_pre_process
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.add_end_token = add_end_token

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_tokenizer)
        self.transform = Sequential(
            Truncate(max_seq_len=max_seq_len) if truncate else torch.nn.Identity(),
            ToTensor(padding_value=self.pad_token_id),
            (
                PadTransform(max_length=max_seq_len, pad_value=self.pad_token_id)
                if pad_to_max_seq_len
                else torch.nn.Identity()
            ),
        )

    def pre_process(self, text: str) -> str:
        text = (
            re.sub(
                r"([,.'!?\"()*#:;~])",
                "",
                text,
            )
            .replace("-", " ")
            .replace("/", " ")
        )
        text = text.rstrip(" ")

        return text

    def __call__(self, text: Union[List[str], str]) -> torch.Tensor:
        if self.do_pre_process:
            if isinstance(text, str):
                text = self.pre_process(text)
            else:
                text = [self.pre_process(t) for t in text]
        tokens = self.tokenizer(text)["input_ids"]
        if not self.add_end_token and tokens[-1] == self.sep_token_id:
            tokens = tokens[:-1]
        input_ids = self.transform(tokens)

        return input_ids


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
    
    if config.dataloader.name == 'disdl':
        #exit
        pass
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

    if not isinstance(train_dataloader, TensorConsumer):
        train_dataloader.sampler.send_job_ended_notfication()
    
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
                    if isinstance(train_dataloader, TensorConsumer):
                        #needed to free the memory for the producer to send more data
                        images = images.cpu()
                        captions = captions.cpu()
                        text_atts = text_atts.cpu()
                        image_ids = image_ids.cpu()

            gpu_processing_started = time.perf_counter()
            if sim:
                    time.sleep(sim_time)
            else:
                loss = model(images, captions, text_atts, image_ids, alpha, is_train=True)
                # Backpropagation and optimization
                optimizer.zero_grad()  # Clear previous gradients
                fabric.backward(loss)  # Backpropagation
                optimizer.step()  # Update weights
                total_train_loss += loss.item() * image.size(0)
                
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
   
    return global_step_count

def retrieval_train_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    image_list = []
    text_list = []
    idx_list = []
    transformation_time_list = []
    fetch_duration_list = []
    cache_hit_list = []
    cached_after_fetch_list = []
    for image, text, idx, fetch_duration, transformation_time, cache_hit, cached_after_fetch in batch:
        image_list.append(image)
        text_list.append(text)
        idx_list.append(idx)
        transformation_time_list.append(transformation_time)
        fetch_duration_list.append(fetch_duration)
        cache_hit_list.append(cache_hit)
        cached_after_fetch_list.append(cached_after_fetch)
        # print(f'Image shape: {image.shape}, Text shape: {text.shape}, Index: {idx}')
    images = torch.stack(image_list, dim=0)
    text = pad_sequence(text_list, batch_first=True)  # You can specify your padding value
    text_atts = (text != 0).type(torch.long)
    idx = torch.Tensor(idx_list).type(torch.long)

    return (
        (images,
        text,
        text_atts,
        idx),
        fetch_duration_list,
        transformation_time_list,
        cache_hit_list,
        False
    )
    

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
    finetune_multi_modal(config, train_logger,val_logger)
 
if __name__ == "__main__":
    main()
