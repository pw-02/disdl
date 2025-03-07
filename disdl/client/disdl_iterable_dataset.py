from disdl_client import DisDLClient
import torch
import boto3
import io
import json
from PIL import Image
import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from typing import List, Dict, Tuple
import functools
import time
from urllib.parse import urlparse
import redis
from io import BytesIO
import lz4.frame
import botocore.config
import os
import threading
import queue
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
class S3Url(object):
    def __init__(self, url):
        self._parsed = urlparse(url, allow_fragments=False)

    @property
    def bucket(self):
        return self._parsed.netloc

    @property
    def key(self):
        if self._parsed.query:
            return self._parsed.path.lstrip('/') + '?' + self._parsed.query
        else:
            return self._parsed.path.lstrip('/')

    @property
    def url(self):
        return self._parsed.geturl()
    
class DisDLIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, 
                 job_id,
                 dataset_location,
                 num_samples,
                 batch_size, 
                 disdl_service_address,
                 prefetch_buffer_size,
                 transform=None,
                 cache_address= None,
                 ssl = True,
                 use_compression = True,
                 use_local_folder=False):
        
        self.disdl_service_address = disdl_service_address
        self.job_id = job_id
        self.dataset_location = dataset_location
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.ssl = ssl
        self.transform = transform
        if cache_address is not None:
            self.cache_host, self.cache_port = cache_address.split(":")
            self.cache_port = int(self.cache_port)
            self.use_cache = True
        else:
            self.use_cache = False
        self.use_local_folder = use_local_folder
        self.client:DisDLClient = None
        #prefetching parameters
        self.prefetch_buffer_size = prefetch_buffer_size
        # self.prefetch_queue = queue.Queue(maxsize=self.prefetch_buffer_size)
        # self.stop_thread = False
        # self._prefetch_thread = threading.Thread(target=self._prefetch_data)
        self.use_compression = use_compression
        self.s3_client = None
        self.cache_client = None
    
    def check_dsdl_client(self):
        if self.client is None:
            self.client = DisDLClient(
                address=self.disdl_service_address, 
                job_id=self.job_id, 
                dataset_location=self.dataset_location)

    def _prefetch_data(self):
        """
        This function will run in a separate thread and pre-load batches into the prefetch_queue.
        """
        while not self.stop_thread:
            if self.prefetch_queue.full():
                time.sleep(0.1)
                continue
            try:
                print(self.prefetch_queue.qsize())
                batch, data_fetch_time, transformation_time, cache_hit, cached_after_fetch = self._get_next_batch()
                #add all of the data to the prefetch queue
                self.prefetch_queue.put((batch,  data_fetch_time, transformation_time, cache_hit, cached_after_fetch))  # ✅ Wrap in a tuple
            except Exception as e:
                print(f"Error prefetching batch: {e}")
                continue

    def __iter__(self):
        self.check_dsdl_client()
        self.stop_thread = False
        self.prefetch_queue = queue.Queue(maxsize=self.prefetch_buffer_size)
        self._prefetch_thread = threading.Thread(target=self._prefetch_data)
        self._prefetch_thread.daemon = True  # Ensure the thread stops when the process exits
        self._prefetch_thread.start()
     
        while not self.stop_thread or not self.prefetch_queue.empty():
            yield self.prefetch_queue.get()
    
    def _set_s3_client(self):
        if self.s3_client is None:
            self.s3_client = boto3.client('s3', config=botocore.config.Config(
                max_pool_connections=100))
            
    def stop(self):
        """
        Stops the prefetching thread.
        """
        self.stop_thread = True
        self._prefetch_thread.join()

    
    def _initialize_cache_client(self):
        """Initialize Redis cache client if not already connected."""
        if self.cache_client is None:
            if self.ssl:
                self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port, ssl=True)
            else:
                self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port)

    def _load_batch_from_cache(self, batch_id):
        try:
            self._initialize_cache_client()   
            return self.cache_client.get(batch_id)
        except Exception as e:
            # print(f"Error fetching from cache: {e}, batch_id: {batch_id}")
            return None
    
    def get_cached_minibatch_with_retries(self, batch_id, max_retries=3, retry_interval=0.1):
        """Attempts to load a batch from cache, retrying if it fails."""
        attempt = 0
        next_minibatch = None
        while attempt < max_retries:
            try:
                next_minibatch = self._load_batch_from_cache(batch_id)
                # If successfully loaded, break out of the loop
                if next_minibatch:
                    break
            except Exception as e:
                # Handle exception (log it, etc.)
                # print(f"Attempt {attempt + 1} failed with error: {e}")
                pass
            # Increment retry count
            attempt += 1
            time.sleep(retry_interval)  # Sleep before retrying
        return next_minibatch
    
    def _get_next_batch(self):
        start_time = time.perf_counter()
        cached_after_fetch = False
        minibatch_bytes = None
        batch_id, samples, is_cached = self.client.sampleNextMinibatch()
        if self.use_cache and is_cached:
            minibatch_bytes = self.get_cached_minibatch_with_retries(batch_id, max_retries=3, retry_interval=0.05)
        
        if minibatch_bytes  is not None and (isinstance(minibatch_bytes , bytes) or isinstance(minibatch_bytes , str)):
            start_transformation_time   = time.perf_counter()
            batch_data, batch_labels = self.convert_bytes_to_torch_tensor(minibatch_bytes)
            transformation_time  =  time.perf_counter() - start_transformation_time
            cache_hit = True
        else:
            batch_data, batch_labels = self.load_batch_data(samples)
            cache_hit = False
             # Apply transformations if provided
            start_transformation_time = time.perf_counter()
            if self.transform is not None:
                for i in range(len(batch_data)):
                    batch_data[i] = self.transform(batch_data[i])        
            transformation_time =  time.perf_counter() - start_transformation_time
            batch_data= torch.stack(batch_data)
            batch_labels = torch.tensor(batch_labels)
            if self.use_cache:
                bytes_tensor = self.convert_torch_tensor_to_bytes((batch_data, batch_labels))
                if self.cache_minibatch_with_retries(batch_id, bytes_tensor, max_retries=0):
                    cached_after_fetch = True
        data_fetch_time = time.perf_counter() - start_time - transformation_time
        return (batch_data,batch_labels,batch_id), data_fetch_time, transformation_time, cache_hit, cached_after_fetch
            
    def cache_minibatch_with_retries(self, batch_id, minibatch, max_retries=4, retry_interval=0.1):
        retries = 0
        while retries <= max_retries:
            try:
                # Attempt to cache the minibatch in Redis
                self.cache_client.set(batch_id, minibatch)
                return True # Exit the function on success
            except Exception as e:
                print(f"Error saving to cache: {e}, batch_id: {batch_id}, retrying {retries}...")
            # Increment the retry count
            retries += 1
            # Wait before retrying
            time.sleep(retry_interval)
        return False



    def convert_torch_tensor_to_bytes(self, data:Tuple[torch.Tensor, torch.Tensor]):    
        with BytesIO() as buffer:
            torch.save(data, buffer)
        bytes_tensor = buffer.getvalue()
        if self.use_compression:
            bytes_tensor = lz4.frame.compress(bytes_tensor,  compression_level=0)
        return bytes_tensor

    def convert_bytes_to_torch_tensor(self, data:bytes):
        if self.use_compression:
            data = lz4.frame.decompress(data)
        with BytesIO(data) as buffer:
            batch_data, batch_labels = torch.load(buffer)
        return batch_data, batch_labels
    
    def load_batch_data(self, samples) -> Tuple[List[torch.Tensor], List[int]]:
        batch_data, batch_labels = [], []
        
        if self.use_local_folder:
            for  data_path, label in samples:
                with open(data_path, 'rb') as f:
                    data = Image.open(f).convert("RGB")
                batch_data.append(data)
                batch_labels.append(label)
            return batch_data, batch_labels
        else:
            self._set_s3_client()
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(self.read_data_from_s3, data_path, label): (data_path, label) for data_path, label in samples}
                for future in as_completed(futures):
                    data_sample, label = future.result()
                    batch_data.append(data_sample)
                    batch_labels.append(label)
            return batch_data, batch_labels
        
    def read_data_from_s3(self,data_path, label) -> tuple: 
        s3_bucket = S3Url(self.dataset_location).bucket
        obj = self.s3_client.get_object(Bucket=s3_bucket, Key=data_path)
        data = Image.open(BytesIO(obj['Body'].read())).convert("RGB")
        return data, label


if __name__ == "__main__":
    dataset_location = "s3://sdl-cifar10/test/"
    service_address = 'localhost:50051'
    client = DisDLClient(address=service_address)
    job_id, dataset_info = client.registerJob(dataset_location="s3://sdl-cifar10/test/")
    num_batchs = dataset_info["num_batches"]
    client.close()
    train_transform = transforms.Compose([
            transforms.Resize(256), 
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    dataset = DisDLIterableDataset(
        job_id=job_id,
        dataset_location=dataset_location,
        num_samples=num_batchs,
        batch_size=128,
        prefetch_buffer_size=1,
        transform=train_transform,
        disdl_service_address=service_address,
        cache_address=None,
        ssl=True,
        use_compression=True,
        use_local_folder=False
    )

    dataloader = DataLoader(dataset, batch_size=None, num_workers=1, pin_memory=True)

    end = time.perf_counter()
    for idx, (bach, data_fetch_time, transformation_time, cache_hit, cached_after_fetch) in enumerate(dataloader):
        bacth_data, batch_labels, batch_id = bach
        delay = time.perf_counter() - end
        print(f"Batch {idx}: id: {batch_id} delay: {delay:.2f}s, fetch(s): {data_fetch_time:.2f}s, transform(s): {transformation_time:.2f}s, hit: {cache_hit}")
        end = time.perf_counter()


        

