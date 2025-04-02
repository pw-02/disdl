from disdl.disdl_client import DisDLClient
#from disdl_client import DisDLClient
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
# import torchaudio
# import torchaudio.transforms as T

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
       
        if cache_address is not None:
            self.cache_host, self.cache_port = cache_address.split(":")
            self.cache_port = int(self.cache_port)
            self.use_cache = True
        else:
            self.use_cache = False
        self.use_local_folder = use_local_folder
        self.client:DisDLClient = None
      
        self.use_compression = use_compression
        self.s3_client = None
        self.cache_client = None
    
    def check_dsdl_client(self):
        if self.client is None:
            self.client = DisDLClient(
                address=self.disdl_service_address, 
                job_id=self.job_id, 
                dataset_location=self.dataset_location)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        self.check_dsdl_client()
        while True:
            yield self._get_next_batch()
    
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

# class DisDLLibrSpeechIterableDataset(DisDLIterableDataset):
#     def __init__(self, 
#                  job_id,
#                  dataset_location,
#                  num_samples,
#                  batch_size, 
#                  disdl_service_address,
#                  transform=None,
#                  cache_address=None,
#                  ssl=True,
#                  use_compression=True,
#                  use_local_folder=False,
#                  custom_attribute=None):
#         super().__init__(job_id, dataset_location, num_samples, batch_size, 
#                          disdl_service_address, cache_address, 
#                          ssl, use_compression, use_local_folder)
#         self.transform = transform
        
#         # Define additional attributes
#         self.custom_attribute = custom_attribute

#     def _get_next_batch(self):

#         start_time = time.perf_counter()
#         cached_after_fetch = False
#         minibatch_bytes = None
#         batch_id, samples, is_cached = self.client.sampleNextMinibatch()
#         if self.use_cache and is_cached:
#             minibatch_bytes = self.get_cached_minibatch_with_retries(batch_id, max_retries=3, retry_interval=0.25)
        
#         if minibatch_bytes  is not None and (isinstance(minibatch_bytes , bytes) or isinstance(minibatch_bytes , str)):
#             start_transformation_time   = time.perf_counter()
#             waveforms, sample_rates, transcripts = self.convert_bytes_to_torch_tensor(minibatch_bytes)
#             transformation_time  =  time.perf_counter() - start_transformation_time
#             cache_hit = True
#         else:
#             waveforms, sample_rates, transcripts = self.load_batch_data(samples)
#             cache_hit = False
#              # Apply transformations if provided
#             start_transformation_time = time.perf_counter()
#             if self.transform is not None:
#                 for i in range(len(waveforms)):
#                     waveforms[i] = self.transform(waveforms[i])      

#             transformation_time =  time.perf_counter() - start_transformation_time
#             waveforms= torch.stack(waveforms)
#             sample_rates = torch.tensor(sample_rates)
#             transcripts = torch.tensor(transcripts)
#             # if self.use_cache:
#             #     bytes_tensor = self.convert_torch_tensor_to_bytes((batch_data, batch_labels))
#             #     if self.cache_minibatch_with_retries(batch_id, bytes_tensor, max_retries=0):
#             #         cached_after_fetch = True
#         data_fetch_time = time.perf_counter() - start_time - transformation_time
#         return (waveforms,sample_rates,transcripts,batch_id), data_fetch_time, transformation_time, cache_hit, cached_after_fetch
    
    
#     def load_batch_data(self, samples) -> Tuple[List[torch.Tensor], List[int]]:
#         waveforms, sample_rates, transcripts = [], [], []
#         if self.use_local_folder:
#             for  data_path, label in samples:
#                 with open(data_path, 'rb') as f:
#                     data = Image.open(f)

#                 waveforms.append(data)
#                 transcripts.append(label)
#             return waveforms, sample_rates, transcripts
#         else:
#             self._set_s3_client()
#             with ThreadPoolExecutor(max_workers=None) as executor:
#                 futures = {executor.submit(self.read_data_from_s3, data_path, label): (data_path, label) for data_path, label in samples}
#                 for future in as_completed(futures):
#                     waveform,sample_rate, transcript = future.result()
#                     waveforms.append(waveform)
#                     sample_rates.append(sample_rate)
#                     transcripts.append(transcript)
#             return waveforms, sample_rates, transcripts
        
#     def read_data_from_s3(self,data_path, transcript) -> tuple: 
#         s3_bucket = S3Url(self.dataset_location).bucket
#         audio_data = self.s3_client.get_object(Bucket=s3_bucket, Key=data_path)["Body"].read()
#         waveform, sample_rate = torchaudio.load(io.BytesIO(audio_data))

#         return waveform, sample_rate, transcript

    
  
class DisDLImageNetIterableDataset(DisDLIterableDataset):
    def __init__(self, 
                 job_id,
                 dataset_location,
                 num_samples,
                 batch_size, 
                 disdl_service_address,
                 transform=None,
                 cache_address=None,
                 ssl=True,
                 use_compression=True,
                 use_local_folder=False,
                 custom_attribute=None):
        super().__init__(job_id, dataset_location, num_samples, batch_size, 
                         disdl_service_address, cache_address, 
                         ssl, use_compression, use_local_folder)
        self.transform = transform
        
        # Define additional attributes
        self.custom_attribute = custom_attribute

    def _get_next_batch(self):

        start_time = time.perf_counter()
        cached_after_fetch = False
        minibatch_bytes = None
        batch_id, samples, is_cached = self.client.sampleNextMinibatch()
        if self.use_cache and is_cached:
            minibatch_bytes = self.get_cached_minibatch_with_retries(batch_id, max_retries=3, retry_interval=0.25)
        
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
    
      
class DisDLOpenImagesDataset(DisDLIterableDataset):
    def __init__(self, 
                 job_id,
                 dataset_location,
                 num_samples,
                 batch_size, 
                 disdl_service_address,
                 transform=None,
                 cache_address=None,
                 ssl=True,
                 use_compression=True,
                 use_local_folder=False,
                 custom_attribute=None):
        super().__init__(job_id, dataset_location, num_samples, batch_size, 
                         disdl_service_address, cache_address, 
                         ssl, use_compression, use_local_folder)
        self.transform = transform
        
        # Define additional attributes
        self.custom_attribute = custom_attribute

    def _get_next_batch(self):

        start_time = time.perf_counter()
        cached_after_fetch = False
        minibatch_bytes = None
        batch_id, samples, is_cached = self.client.sampleNextMinibatch()
        if self.use_cache and is_cached:
            minibatch_bytes = self.get_cached_minibatch_with_retries(batch_id, max_retries=3, retry_interval=0.25)
        
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




















class DisDLCocoIterableDataset(DisDLIterableDataset):

    def __init__(self, 
                 job_id,
                 dataset_location,
                 num_samples,
                 batch_size, 
                 disdl_service_address,
                 image_transform=None,
                 text_transform=None,
                 cache_address=None,
                 ssl=True,
                 use_compression=True,
                 use_local_folder=False,
                 custom_attribute=None):
        super().__init__(job_id, dataset_location, num_samples, batch_size, 
                         disdl_service_address, cache_address, 
                         ssl, use_compression, use_local_folder)
        self.image_transform = image_transform
        self.text_transform = text_transform
        
        # Define additional attributes
        self.custom_attribute = custom_attribute

    def _get_next_batch(self):
        start_time = time.perf_counter()
        cached_after_fetch = False
        minibatch_bytes = None
        batch_id, samples, is_cached = self.client.sampleNextMinibatch()

        if self.use_cache and is_cached:
            minibatch_bytes = self.get_cached_minibatch_with_retries(batch_id, max_retries=3, retry_interval=0.25)
        
        if minibatch_bytes  is not None and (isinstance(minibatch_bytes , bytes) or isinstance(minibatch_bytes , str)):
            start_transformation_time   = time.perf_counter()
            images, captions, text_atts, image_ids = self.convert_bytes_to_torch_tensor(minibatch_bytes)
            transformation_time  =  time.perf_counter() - start_transformation_time
            cache_hit = True
        else:
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
            
             # Convert to tensors
            images = torch.stack(images, dim=0)
            captions = pad_sequence(captions, batch_first=True)
            text_atts = (captions != 0).type(torch.long)
            image_ids =  torch.Tensor(image_ids).type(torch.long)
            transformation_time =  time.perf_counter() - start_transformation_time
         
            # if self.use_cache:
            #     bytes_tensor = self.convert_torch_tensor_to_bytes((batch_data, batch_labels))
            #     if self.cache_minibatch_with_retries(batch_id, bytes_tensor, max_retries=0):
            #         cached_after_fetch = True
        data_fetch_time = time.perf_counter() - start_time - transformation_time
        return (images, captions,text_atts,image_ids,batch_id), data_fetch_time, transformation_time, cache_hit, cached_after_fetch
    
    
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
    
    
    def convert_bytes_to_torch_tensor(self, data:bytes):
        if self.use_compression:
            data = lz4.frame.decompress(data)
        with BytesIO(data) as buffer:
            images, captions, text_atts, image_ids = torch.load(buffer)
        return  images, captions, text_atts, image_ids




# if __name__ == "__main__":
#     service_address = 'localhost:50051'
#     cache_address = None
#     test_dataset = 'LibriSpeech' # 'COCO', 'ImageNet', 'LibriSpeech'

#     if test_dataset == 'ImageNet':
#         dataset_location = " s3://imagenet1k-sdl/train/"
#         client = DisDLClient(address=service_address)
#         job_id, dataset_info = client.registerJob(
#             dataset_location=dataset_location)
#         num_batchs = dataset_info["num_batches"]
#         client.close()

#         train_transform = transforms.Compose([
#                 transforms.Resize(256), 
#                 transforms.RandomResizedCrop(224),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ])

#         dataset = DisDLImageNetIterableDataset(
#             job_id=job_id,
#             dataset_location=dataset_location,
#             num_samples=num_batchs,
#             batch_size=128,
#             transform=train_transform,
#             disdl_service_address=service_address,
#             cache_address=cache_address,
#             ssl=False,
#             use_compression=True,
#             use_local_folder=False
#         )
#     elif test_dataset == 'LibriSpeech':
#         dataset_location = "s3://disdlspeech/test-clean/"
#         client = DisDLClient(address=service_address)
#         job_id, dataset_info = client.registerJob(
#             dataset_location=dataset_location)
#         num_batchs = dataset_info["num_batches"]
#         client.close()

#         transform = T.MelSpectrogram(sample_rate=16000, n_mels=128)
#         dataset = DisDLLibrSpeechIterableDataset(
#             job_id=job_id,
#             dataset_location=dataset_location,
#             num_samples=num_batchs,
#             batch_size=128,
#             transform=transform,
#             disdl_service_address=service_address,
#             cache_address=cache_address,
#             ssl=False,
#             use_compression=True,
#             use_local_folder=False
#         )

#     elif test_dataset == 'COCO':
#         from albef_transforms import ALBEFTextTransform, image_transform
#         dataset_location = "s3://coco-dataset/coco_train.json"
#         client = DisDLClient(address=service_address)
#         job_id, dataset_info = client.registerJob(
#             dataset_location=dataset_location)
#         num_batchs = dataset_info["num_batches"]
#         client.close()
     

#         dataset = DisDLCocoIterableDataset(
#             job_id=job_id,
#             dataset_location=dataset_location,
#             num_samples=num_batchs,
#             batch_size=128,
#             image_transform=image_transform(),
#             text_transform=ALBEFTextTransform(
#             truncate=True, pad_to_max_seq_len=True, max_seq_len=30, add_end_token=False),
#             disdl_service_address=service_address,
#             cache_address=cache_address,
#             ssl=False,
#             use_compression=True,
#             use_local_folder=False
#         )

#     dataloader = DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=True)

#     end = time.perf_counter()
#     for idx, (batch, data_fetch_time, transformation_time, cache_hit, cached_after_fetch) in enumerate(dataloader):
#         delay = time.perf_counter() - end
#         bacth_data, batch_labels, batch_id = batch
#         print(f"Batch {idx}: id: {batch_id} delay: {delay:.2f}s, fetch(s): {data_fetch_time:.2f}s, transform(s): {transformation_time:.2f}s, hit: {cache_hit}")
#         time.sleep(0.05)
#         end = time.perf_counter()

        

