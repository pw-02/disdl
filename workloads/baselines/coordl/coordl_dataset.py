import torch
import boto3
from PIL import Image
import torch
from typing import List, Dict, Tuple
import time
from urllib.parse import urlparse
import redis
from io import BytesIO
import lz4.frame
import botocore.config
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.nn.utils.rnn import pad_sequence
import json
from typing import  Dict, Sized
import functools
import pandas as pd

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
    
class CoorDLDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 job_id,
                 dataset_location,
                 batch_size,
                 cache_address= None,
                 ssl = True,
                 use_compression = True,
                 use_local_folder=False):
        
        self.job_id = job_id
        self.dataset_location = dataset_location
        self.batch_size = batch_size
        self.ssl = ssl
        
        if cache_address is not None:
            self.cache_host, self.cache_port = cache_address.split(":")
            self.cache_port = int(self.cache_port)
            self.use_cache = True
        else:
            self.use_cache = False
        self.use_local_folder = use_local_folder
        self.use_compression = use_compression
        self.s3_client = None
        self.cache_client = None
        self.num_samples = None
    
    def __len__(self):
        return self.num_samples
    
    def _set_s3_client(self):
        if self.s3_client is None:
            self.s3_client = boto3.client('s3', config=botocore.config.Config(
                max_pool_connections=100))
            
    
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
        self._initialize_cache_client()
        attempt = 0
        next_minibatch = None
        while attempt <= max_retries:
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
                
    def cache_minibatch_with_retries(self, batch_id, minibatch, max_retries=4, retry_interval=0.1):
        retries = 0
        while retries <= max_retries:
            try:
                # Attempt to cache the minibatch in Redis
                self.cache_client.set(batch_id, minibatch)
                return True # Exit the function on success
            except Exception as e:
                # print(f"Error saving to cache: {e}, batch_id: {batch_id}, retrying {retries}...")
                pass
            # Increment the retry count
            retries += 1
            # Wait before retrying
            time.sleep(retry_interval)
        return False


    def convert_torch_tensor_to_bytes(self, data):    
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
    
#-------------------------------------------------------------------------------


class CoorDLMSCOCODataset(CoorDLDataset):
    def __init__(self, 
                     job_id,
                     dataset_location,
                     batch_size, 
                     image_transform=None,
                     text_transform=None,       
                     cache_address=None,
                     ssl=True,
                     use_compression=True,
                     use_local_folder=False
                     ):
            super().__init__(job_id, dataset_location, batch_size, cache_address, ssl, use_compression, use_local_folder)
            self.s3_client = None
            self.cache_client = None
            self.s3_bucket = S3Url(dataset_location).bucket
            self.s3_prefix = S3Url(dataset_location).key
            self.s3_data_dir = dataset_location
            self.samples = self._get_samples_from_s3()
            self.image_transform = image_transform
            self.text_transform = text_transform
            # print(self.job_id, self.batches)
            pass
    
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['cache_client']  # Remove the Redis connection before pickling
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.ssl:
            self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port, ssl=True)
        else:
            self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port)

    
    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [(blob, class_index)
            for class_index, blob_class in enumerate(self.samples)
            for blob in self.samples[blob_class]]
    
    def __len__(self) -> int:
        return sum(len(class_items) for class_items in self.samples.values())
    
    def _get_samples_from_s3(self, use_index_file=True, images_only=True) -> Dict[str, List[str]]:
        s3_client = boto3.client('s3')
        index_object = s3_client.get_object(Bucket=self.s3_bucket, Key=self.s3_prefix)
        file_content = index_object['Body'].read().decode('utf-8')
        # samples = json.loads(file_content)
        paired_samples = json.loads(file_content)
        return paired_samples


    def __getitem__(self, next_batch):
        start_time = time.perf_counter()
        cached_after_fetch = False
        minibatch_bytes = None

        # next_batch, is_cached = self._find_next_batch_to_process()
        batch_indices, batch_id, is_cached = next_batch
        # return batch_id
   

        if self.use_cache:
            minibatch_bytes = self.get_cached_minibatch_with_retries(batch_id, max_retries=3, retry_interval=0.25)
        
        if minibatch_bytes  is not None and (isinstance(minibatch_bytes , bytes) or isinstance(minibatch_bytes , str)):
            start_transformation_time   = time.perf_counter()
            images, captions, text_atts, image_ids = self.convert_bytes_to_torch_tensor(minibatch_bytes)
            processed_count = int(self.cache_client.get(f"{batch_id}_count"))
            if processed_count == 4:
                #remove from cache
                self.cache_client.delete(batch_id)
                # self.cache_client.delete(f"{batch_id}_count")
            else:
                self.cache_client.set(f"{batch_id}_count", processed_count+1)

            transformation_time  =  time.perf_counter() - start_transformation_time
            cache_hit = True
        else:
            samples = []
            for i in batch_indices:
                sample, image_id = self._classed_items[i]
                image, cpation = sample
                samples.append((image, cpation, image_id))

            images, captions, image_ids = self.load_batch_data(samples)
            cache_hit = False
             # Apply transformations if provided
            start_transformation_time = time.perf_counter()
            if self.image_transform is not None:
                for i in range(len(images)):
                    images[i] = self.image_transform(images[i])  
            if self.text_transform is not None:
                for i in range(len(captions)):
                    captions[i] = self.text_transform(captions[i])   

            images = torch.stack(images, dim=0)
            captions = pad_sequence(captions, batch_first=True)
            text_atts = (captions != 0).type(torch.long)
            image_ids =  torch.Tensor(image_ids).type(torch.long)
            transformation_time =  time.perf_counter() - start_transformation_time

            if self.use_cache:
                 bytes_tensor = self.convert_torch_tensor_to_bytes((images, captions, text_atts, image_ids)) 
                 if self.cache_minibatch_with_retries(batch_id, bytes_tensor, max_retries=0):
                    cached_after_fetch = True
                    self.cache_client.set(f"{batch_id}_count", 1)
        data_fetch_time = time.perf_counter() - start_time - transformation_time
        return (images, captions,text_atts,image_ids,batch_id), data_fetch_time, transformation_time, cache_hit, cached_after_fetch
    
    def convert_bytes_to_torch_tensor(self, data:bytes):
        if self.use_compression:
            data = lz4.frame.decompress(data)
        with BytesIO(data) as buffer:
            images, captions, text_atts, image_ids = torch.load(buffer)
        return  images, captions, text_atts, image_ids
    
    
    
    def load_batch_data(self, samples) -> Tuple[List[torch.Tensor], List[int]]:
        images, captions, image_ids = [],[],[]
        if self.use_local_folder:
            for  data_path, label in samples:
                with open(data_path, 'rb') as f:
                    data = Image.open(f).convert("RGB")
                images.append(data)
                captions.append(label)
            return images, captions
        else:
            self._set_s3_client()
            with ThreadPoolExecutor(max_workers=None) as executor:
                futures = {executor.submit(self.read_data_from_s3, image, caption, imageid): (image, caption, imageid) for image, caption, imageid in samples}
                for future in as_completed(futures):
                    image, caption, imageid = future.result()
                    images.append(image)
                    captions.append(caption)
                    image_ids.append(imageid)
            return images, captions, image_ids
    
    def read_data_from_s3(self,data_path, caption, imageid) -> tuple: 
        s3_bucket = S3Url(self.dataset_location).bucket
        obj = self.s3_client.get_object(Bucket=s3_bucket, Key=data_path)
        data = Image.open(BytesIO(obj['Body'].read())).convert("RGB")
        return data, caption, imageid



#-------------------------------------------------------------------------------

class CoorDLOpenImagesIterableDataset(CoorDLDataset):
    def __init__(self, 
                     job_id,
                     dataset_location,
                     batch_size, 
                     transform=None,
                     cache_address=None,
                     ssl=True,
                     use_compression=True,
                     use_local_folder=False
                     ):
            super().__init__(job_id, dataset_location, batch_size, cache_address, ssl, use_compression, use_local_folder)
            self.s3_client = None
            self.cache_client = None
            self.s3_bucket = S3Url(self.dataset_location).bucket
            self.s3_prefix = S3Url(self.dataset_location).key
            self.samples = self._get_samples_from_s3()
            self.transform = transform    
            # print(self.job_id, self.batches)
            pass
    
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['cache_client']  # Remove the Redis connection before pickling
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.ssl:
            self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port, ssl=True)
        else:
            self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port)

    
    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [(blob, class_index)
            for class_index, blob_class in enumerate(self.samples)
            for blob in self.samples[blob_class]]
    
    def __len__(self) -> int:
        return sum(len(class_items) for class_items in self.samples.values())
    
    def count_image_samples(self):
        """Count the number of image samples in the dataset."""
        return sum(len(class_items) for class_items in self.samples.values())
    
    def _get_samples_from_s3(self, use_index_file=True, images_only=True) -> Dict[str, List[str]]:
        s3_client = boto3.client('s3')
        index_file_key = f"{self.s3_prefix}_paired_index.json"
        paired_samples = {}

        if use_index_file:
            try:
                index_object = s3_client.get_object(Bucket=self.s3_bucket, Key=index_file_key)
                file_content = index_object['Body'].read().decode('utf-8')
                paired_samples = json.loads(file_content)
                return list(paired_samples.values())
            except Exception as e:
                print(f"Error reading index file '{index_file_key}': {e}")

        #fist lets get all of the images ids and paths
        images ={}
        image_label_dict = {}
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
                    print(f"Skipping non-image file: {blob_path}")
                    if 'annotations' in blob_path:
                        response = s3_client.get_object(Bucket=self.s3_bucket, Key=blob_path)
                        csv_content = response["Body"].read().decode("utf-8")  # Decode bytes to string
                        df = pd.read_csv(StringIO(csv_content))  # Convert string to DataFrame
                        image_label_dict = df.set_index("ImageID")["Ids"].to_dict()

                    continue  # Skip non-image files
                if 'index.json' in blob_path:
                    continue  # Skip index file

                fileid = stripped_path.split("/")[-1].split(".")[0]
                images[fileid] = blob_path

        for image_id in image_label_dict:
            if image_id in images:
                paired_samples[image_id] = (images[image_id], image_label_dict[image_id])

        if not use_index_file and paired_samples:
            s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=index_file_key,
                Body=json.dumps(paired_samples, indent=4).encode('utf-8'))

        return list(paired_samples.values())
    
    
    def __getitem__(self, next_batch):
        start_time = time.perf_counter()
        cached_after_fetch = False
        minibatch_bytes = None

        # next_batch, is_cached = self._find_next_batch_to_process()
        batch_indices, batch_id, is_cached = next_batch
    
        if self.use_cache:
            minibatch_bytes = self.get_cached_minibatch_with_retries(batch_id, max_retries=3, retry_interval=0.25)
        
        if minibatch_bytes  is not None and (isinstance(minibatch_bytes , bytes) or isinstance(minibatch_bytes , str)):
            start_transformation_time   = time.perf_counter()
            batch_data, batch_labels = self.convert_bytes_to_torch_tensor(minibatch_bytes)
            processed_count = int(self.cache_client.get(f"{batch_id}_count"))
            if processed_count == 4:
                #remove from cache
                self.cache_client.delete(batch_id)
                # self.cache_client.delete(f"{batch_id}_count")
            else:
                self.cache_client.set(f"{batch_id}_count", processed_count+1)

            transformation_time  =  time.perf_counter() - start_transformation_time
            cache_hit = True
        else:
            samples = []
            for i in batch_indices:
                samples.append(self.samples[i])
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
                bytes_tensor = self.convert_torch_tensor_to_bytes((batch_data, batch_labels)) #1 is the number of jobs that have processed this batc
                if self.cache_minibatch_with_retries(batch_id, bytes_tensor, max_retries=0):
                    cached_after_fetch = True
                    self.cache_client.set(f"{batch_id}_count", 1)
        data_fetch_time = time.perf_counter() - start_time - transformation_time
        return (batch_data,batch_labels,batch_id), data_fetch_time, transformation_time, cache_hit, cached_after_fetch
    
    
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
            with ThreadPoolExecutor(max_workers=None) as executor:
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

#-------------------------------------------------------------------------------









    

class CoorDLImageNetIterableDataset(CoorDLDataset):
    def __init__(self, 
                     job_id,
                     dataset_location,
                     batch_size, 
                     transform=None,
                     cache_address=None,
                     ssl=True,
                     use_compression=True,
                     use_local_folder=False
                     ):
            super().__init__(job_id, dataset_location, batch_size, cache_address, ssl, use_compression, use_local_folder)
            self.s3_client = None
            self.cache_client = None
            self.s3_bucket = S3Url(self.dataset_location).bucket
            self.s3_prefix = S3Url(self.dataset_location).key
            self.samples = self._get_samples_from_s3()
            self.transform = transform    
            # print(self.job_id, self.batches)
            pass
    
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['cache_client']  # Remove the Redis connection before pickling
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.ssl:
            self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port, ssl=True)
        else:
            self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port)

    
    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [(blob, class_index)
            for class_index, blob_class in enumerate(self.samples)
            for blob in self.samples[blob_class]]
    
    def __len__(self) -> int:
        return sum(len(class_items) for class_items in self.samples.values())
    
    def count_image_samples(self):
        """Count the number of image samples in the dataset."""
        return sum(len(class_items) for class_items in self.samples.values())
    
    def _get_samples_from_s3(self, use_index_file=True, images_only=True) -> Dict[str, List[str]]:
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
                    print(f"Skipping non-image file: {blob_path}")
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
                Body=json.dumps(paired_samples, indent=4).encode('utf-8'))

        return paired_samples
    
    def _find_next_batch_to_process(self):
        next_batch = None
        cache_hit = False
        for batch in self.batches:
            batch_indices, batch_id, this_job_fetch = batch
            if self.use_cache:
                self._initialize_cache_client()
                if self.cache_client.exists(batch_id): #cached by another job use it
                    next_batch = batch
                    cache_hit = True
                    break
            
            if this_job_fetch:
                next_batch = batch
                break
        if next_batch is not None:
            next_batch = self.batches[0]

        self.batches.remove(next_batch)
        return next_batch, cache_hit


    def __getitem__(self, next_batch):
        start_time = time.perf_counter()
        cached_after_fetch = False
        minibatch_bytes = None

        # next_batch, is_cached = self._find_next_batch_to_process()
        batch_indices, batch_id, is_cached = next_batch
        # return batch_id
        samples = [self._classed_items[i] for i in batch_indices]
        
        if self.use_cache:
            minibatch_bytes = self.get_cached_minibatch_with_retries(batch_id, max_retries=3, retry_interval=0.25)
        
        if minibatch_bytes  is not None and (isinstance(minibatch_bytes , bytes) or isinstance(minibatch_bytes , str)):
            start_transformation_time   = time.perf_counter()
            batch_data, batch_labels = self.convert_bytes_to_torch_tensor(minibatch_bytes)
            processed_count = int(self.cache_client.get(f"{batch_id}_count"))
            if processed_count == 4:
                #remove from cache
                self.cache_client.delete(batch_id)
                # self.cache_client.delete(f"{batch_id}_count")
            else:
                self.cache_client.set(f"{batch_id}_count", processed_count+1)

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
                bytes_tensor = self.convert_torch_tensor_to_bytes((batch_data, batch_labels)) #1 is the number of jobs that have processed this batc
                if self.cache_minibatch_with_retries(batch_id, bytes_tensor, max_retries=0):
                    cached_after_fetch = True
                    self.cache_client.set(f"{batch_id}_count", 1)
        data_fetch_time = time.perf_counter() - start_time - transformation_time
        return (batch_data,batch_labels,batch_id), data_fetch_time, transformation_time, cache_hit, cached_after_fetch
    
    
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
            with ThreadPoolExecutor(max_workers=None) as executor:
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
    import torchvision.transforms as transforms
    #import torhc dataloader
    from torch.utils.data import DataLoader
    from coordl_sampler import CoorDLBatchSampler

    cache_address = '127.0.0.1:6379'
    test_dataset = 'ImageNet' # 'COCO', 'ImageNet', 'LibriSpeech'

    if test_dataset == 'ImageNet':
        dataset_location = " s3://imagenet1k-sdl/train/"

       
        train_transform = transforms.Compose([
                transforms.Resize(256), 
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        dataset = CoorDLImageNetIterableDataset(
            job_id=1,
            dataset_location=dataset_location,
            batch_size=128,
            transform=train_transform,
            cache_address=cache_address,
            ssl=False,
            use_compression=True,
            use_local_folder=False
        )

        sampler = CoorDLBatchSampler(
            dataset.count_image_samples(), 
            dataset.batch_size, 
            jobid=dataset.job_id,   
            drop_last=False, 
            shuffle=True,
            cache_address=cache_address,
            ssl=False)

        laoder = DataLoader(dataset=dataset, sampler=sampler, batch_size=None, num_workers=0)

        # Iterate through the dataset
        for batch_idx, (batchid) in enumerate(laoder):
            print(batchid)
        # for batch_idx, (batch, data_load_time, transformation_time, is_cache_hit, cached_on_miss) in enumerate(dataset):
        #     items, labels, batc_id = batch
        #     print(batc_id)

            # print(batch_idx, batch[0].shape, batch[1].shape, data_load_time, transformation_time, is_cache_hit, cached_on_miss)
            # if batch_idx > 10:

