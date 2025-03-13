
# No 'default_generator' in torch/__init__.pyi
from typing import TypeVar, List, Tuple, Dict
import PIL.Image as Image
import numpy as np
import io
import numpy as np
import PIL
from urllib.parse import urlparse
import boto3
import functools
from torch.utils.data import Dataset
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import botocore.config
import os
import redis
import torch
from io import BytesIO
from torch.nn.utils.rnn import pad_sequence
import csv

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


class TensorSocketCocoDataset(Dataset):
    def __init__(self, 
                 s3_data_dir: str,
                 
                image_transform=None,
                 text_transform=None,                  
                 cache_address=None,  
                 simulate_mode=False, 
                 simulate_time_for_cache_miss=0,
                 simulate_time_for_cache_hit=0,
                 cache_transformations=True,
                 use_compression=False,
                 use_local_folder=False,
                 ssl=True,
                log_dir='logs'):
        
        self.log_dir = log_dir
        self.s3_bucket = S3Url(s3_data_dir).bucket
        self.s3_prefix = S3Url(s3_data_dir).key
        self.s3_data_dir = s3_data_dir
        self.s3_client = None
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.using_local_folder = use_local_folder
        self.samples = self._get_samples_from_s3()
        self.simulate_mode = simulate_mode
        self._simlute_time_for_cache_miss = simulate_time_for_cache_miss
        self._simlute_time_for_cache_hit = simulate_time_for_cache_hit
        self.cache_transformations = cache_transformations
        self.use_compression = use_compression
        self.ssl = ssl
        self.cache_client = None

        if cache_address is not None:
            self.cache_host, self.cache_port = cache_address.split(":")
            self.cache_port = int(self.cache_port)
            self.use_cache = True
        else:
            self.use_cache = False

    def record_metrics(self, line):
        file_name = os.path.join(self.log_dir, 'tensordataset_coco.csv')
        file_exists = os.path.isfile(file_name)
        with open(file_name, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=line.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(line)
    
    def __getstate__(self):
            state = self.__dict__.copy()
            if self.use_cache:
                del state['cache_client']  # Remove the Redis connection before pickling
            return state

    def __setstate__(self, state):
            self.__dict__.update(state)
            if self.use_cache:
                if self.ssl:
                    self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port, ssl=True)
                else:
                    self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port)

    
    def check_s3_client(self):
        if self.s3_client is None:
            self.s3_client = boto3.client('s3', config=botocore.config.Config(
                max_pool_connections=100))
    
    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [(blob, class_index)
            for class_index, blob_class in enumerate(self.samples)
            for blob in self.samples[blob_class]]
            

    def _get_samples_from_s3(self, use_index_file=True, images_only=True) -> Dict[str, List[str]]:
        s3_client = boto3.client('s3')
        index_object = s3_client.get_object(Bucket=self.s3_bucket, Key=self.s3_prefix)
        file_content = index_object['Body'].read().decode('utf-8')
        # samples = json.loads(file_content)
        paired_samples = json.loads(file_content)
        return paired_samples


    def __len__(self) -> int:
        return sum(len(class_items) for class_items in self.samples.values())
    

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        batch_id, batch_indices = idx
        start_loading_time = time.perf_counter()
        images, captions, image_ids, cache_hit_count = self.fetch_batch_data(batch_indices)
         # Apply transformations if provided
        start_transformation_time = time.perf_counter()
        for i in range(len(images)):
            images[i] = self.image_transform(images[i])      
        
        for i in range(len(captions)):
            captions[i] = self.text_transform(captions[i]) 
         # Convert to tensors
        images = torch.stack(images, dim=0)
        captions = pad_sequence(captions, batch_first=True)
        text_atts = (captions != 0).type(torch.long)
        image_ids =  torch.Tensor(image_ids).type(torch.long)
        
        transformation_time =  time.perf_counter() - start_transformation_time
        data_fetch_time  = time.perf_counter() - start_loading_time - transformation_time
        self.record_metrics({'s3': self.s3_data_dir, 'batch_id': batch_id, 'data_fetch_time': data_fetch_time, 'transformation_time': transformation_time, 'cache_hit_count': cache_hit_count, 'total_time': data_fetch_time + transformation_time})
        return images, captions, text_atts, image_ids


        # return samples, labels,batch_id,data_fetch_time,transformation_time

    def fetch_batch_data(self, batch_indices: List[str]):
        data_samples,captions, image_ids = [], [], []
        cache_hits = 0

        self.check_s3_client()
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.get_data_sample, idx): idx for idx in batch_indices}
            for future in as_completed(futures):
                data_sample, caption, imageid, cahe_hit = future.result()
                if cahe_hit:
                    cache_hits += 1
                data_samples.append(data_sample)
                captions.append(caption)
                image_ids.append(imageid)

        return data_samples, captions, image_ids, cache_hits
        

    def _initialize_cache_client(self):
        """Initialize Redis cache client if not already connected."""
        if self.cache_client is None:
            # self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port)
            if self.ssl:
                self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port, ssl=True)
            else:
                self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port)

    def fetch_item_from_cache(self, idx: int):
          self._initialize_cache_client()
          byte_image = self.cache_client.get(idx)
          if byte_image is None:
                return None
          byteImgIO = io.BytesIO(byte_image)
          data = Image.open(byteImgIO)
          return data
    
    def put_item_in_cache(self, idx: int, data):
        self._initialize_cache_client()
        # Serialize the image using BytesIO
        img_byte_arr = BytesIO()
        data.save(img_byte_arr, format='PNG')  # Save as PNG to the byte array
        img_byte_arr.seek(0)  # Reset pointer to the start of the byte array
        self.cache_client.set(idx, img_byte_arr.read())

    def get_data_sample(self,idx) -> tuple:  
        sample, image_id = self._classed_items[idx]
        data_path, caption = sample
        cache_hit = False
        if self.use_cache:
            data = self.fetch_item_from_cache(idx)
            cache_hit = True
        else:
            obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=data_path)
            data = Image.open(BytesIO(obj['Body'].read())).convert("RGB")
            if self.use_cache:
                self.put_item_in_cache(idx, data)
        return data.convert("RGB"), caption, image_id, cache_hit

    def fetch_image_from_s3(self, data_path):
        if self.s3_client is None:
            self.s3_client = boto3.client('s3')
        obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=data_path)
        img_data = obj['Body'].read()
        image = Image.open(io.BytesIO(img_data)) #.convert('RGB')
        return image
    
    def _get_sample_list_from_s3(self, use_index_file=True, images_only=True) -> Dict[str, List[str]]:
        s3_client = boto3.client('s3')

        index_file_key = f"{self.s3_prefix}_paired_index.json"
        paired_samples = {}

        if use_index_file:
            try:
                index_object = s3_client.get_object(Bucket=self.s3_bucket, Key=index_file_key)
                file_content = index_object['Body'].read().decode('utf-8')
                paired_samples = json.loads(file_content)
                return paired_samples
            except Exception as e:
                print(f"Error reading index file '{index_file_key}': {e}")

        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=self.s3_prefix):
            for blob in page.get('Contents', []):
                blob_path = blob.get('Key')
                
                if blob_path.endswith("/"):
                    continue  # Skip folders
                
                stripped_path = blob_path[len(self.s3_prefix):].lstrip("/")
                if stripped_path == blob_path:
                    continue  # No matching prefix, skip

                if images_only and not blob_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue  # Skip non-image files
                
                if 'index.json' in blob_path:
                    continue  # Skip index file

                blob_class = stripped_path.split("/")[0]
                if blob_class not in paired_samples:
                    paired_samples[blob_class] = []
                paired_samples[blob_class].append(blob_path)

        if use_index_file and paired_samples:
            s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=index_file_key,
                Body=json.dumps(paired_samples, indent=4).encode('utf-8')
            )

        return paired_samples
       

