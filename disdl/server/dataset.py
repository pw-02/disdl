from utils import S3Url
import boto3
import json
from typing import List, Tuple, Dict
import functools
import pandas as pd
from io import StringIO

class S3DatasetBase:
    def __init__(self, dataset_location: str,batch_size:int, num_partitions: int = 1, transforms=None,  ):
        self.dataset_location = dataset_location
        self.transforms = transforms
        self.s3_bucket = S3Url(self.dataset_location).bucket
        self.s3_prefix = S3Url(self.dataset_location).key
        self.batch_size = batch_size
        self.num_partitions = num_partitions
        self.samples = self._get_samples_from_s3()

    def __len__(self) -> int:
        return len(self.samples)

    def get_samples(self, indices: List[int]):
        return [self.samples[i] for i in indices]

    def dataset_info(self):
        return {
            'location': self.dataset_location,
            'num_samples': len(self),
            'batch_size': self.batch_size,
            'num_partitions': self.num_partitions}
    
    def _get_samples_from_s3(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement _get_samples_from_s3")


class MSCOCODataset(S3DatasetBase):
    def __init__(self, 
                 dataset_location: str, 
                 batch_size:int,
                 num_partitions: int = 1,
                 transforms=None):
            super().__init__(dataset_location, batch_size, num_partitions, transforms)
            self.annotation_file = self.s3_prefix
            self.samples = self._get_samples_from_s3()
            pass
    
    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [(blob, class_index)
            for class_index, blob_class in enumerate(self.samples)
            for blob in self.samples[blob_class]]
    
    def get_samples(self, indices: List[int]):
        result = []
        for i in indices:
            sample, image_id = self._classed_items[i]
            image, caption = sample
            result.append((image, caption, image_id))
        return result
    
    def _get_samples_from_s3(self) -> Dict[str, List[str]]:
        s3_client = boto3.client('s3')
        index_object = s3_client.get_object(Bucket=self.s3_bucket, Key=self.annotation_file)
        file_content = index_object['Body'].read().decode('utf-8')
        return json.loads(file_content)
    
    def __len__(self) -> int:
        return sum(len(class_items) for class_items in self.samples.values())

class OpenImagesDataset(S3DatasetBase):
    def __init__(self, dataset_location: str, batch_size:int, num_partitions: int = 1, transforms=None):
        super().__init__(dataset_location, batch_size, num_partitions, transforms)
        
    def _get_samples_from_s3(self, use_index_file=True, images_only=True) -> Dict[str, List[str]]:
        s3_client = boto3.client('s3')
        index_file_key = f"{self.s3_prefix}_paired_index.json"
        paired_samples = {}

        if use_index_file:
            try:
                obj = s3_client.get_object(Bucket=self.s3_bucket, Key=index_file_key)
                return list(json.loads(obj['Body'].read().decode('utf-8')).values())
            except Exception as e:
                print(f"Index load failed: {e}")

        images = {}
        labels = {}

        for page in s3_client.get_paginator('list_objects_v2').paginate(Bucket=self.s3_bucket, Prefix=self.s3_prefix):
            for blob in page.get('Contents', []):
                key = blob.get('Key')
                
                if 'annotations' in key:
                    response = s3_client.get_object(Bucket=self.s3_bucket, Key=key)
                    df = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
                    labels = df.set_index("ImageID")["Ids"].to_dict()
                elif key.lower().endswith(('.jpg', '.jpeg', '.png')):
                    fileid = key.split("/")[-1].split(".")[0]
                    images[fileid] = key
        
        for image_id in labels:
            if image_id in images:
                paired_samples[image_id] = (images[image_id], labels[image_id])

        if paired_samples:
            s3_client.put_object(Bucket=self.s3_bucket, Key=index_file_key, Body=json.dumps(paired_samples, indent=4).encode('utf-8'))
        return list(paired_samples.values())
    
    def get_samples(self, indices: List[int]):
        samples = []
        for i in indices:
            samples.append(self.samples[i])
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)


class ImageNetDataset(S3DatasetBase):
    def __init__(self, dataset_location: str, batch_size:int, num_partitions: int = 1, transforms=None):
        super().__init__(dataset_location, batch_size, num_partitions, transforms)
        
    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [(blob, class_index)
            for class_index, blob_class in enumerate(self.samples)
            for blob in self.samples[blob_class]]
    
    def __len__(self) -> int:
        return sum(len(class_items) for class_items in self.samples.values())
    
    def _get_samples_from_s3(self, use_index_file=True, images_only=True) -> Dict[str, List[str]]:
        s3_client = boto3.client('s3')
        index_file_key = f"{self.s3_prefix}_paired_index.json"
        paired_samples = {}
        if use_index_file:
            try:
                obj = s3_client.get_object(Bucket=self.s3_bucket, Key=index_file_key)
                return json.loads(obj['Body'].read().decode('utf-8'))
            except Exception as e:
                print(f"Error loading index: {e}")

        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=self.s3_prefix):
            for blob in page.get('Contents', []):
                key = blob['Key']
                if key.lower().endswith(('.jpg', '.jpeg', '.png')):
                    stripped = key[len(self.s3_prefix):].lstrip('/')
                    blob_class = stripped.split("/")[0]
                    paired_samples.setdefault(blob_class, []).append(key)
                    
        if paired_samples:
            s3_client.put_object(Bucket=self.s3_bucket, Key=index_file_key, Body=json.dumps(paired_samples).encode('utf-8'))
        return paired_samples
    
    def get_samples(self, indices: List[int]):
        samples = []
        for i in indices:
            samples.append(self._classed_items[i])
        return samples
    

# class LibSpeechDataset():
#     def __init__(self, 
#                  dataset_location: str, 
#                  transforms=None):
        
#         self.dataset_location = dataset_location
#         self.s3_bucket = S3Url(self.dataset_location).bucket
#         self.s3_prefix = S3Url(self.dataset_location).key
#         self.transforms = transforms
#         self.samples = self._get_samples_from_s3()
    
#     def __len__(self) -> int:
#         return len(self.samples)
    
#     def _get_samples_from_s3(self, use_index_file=False) -> Dict[str, List[str]]:
#         s3_client = boto3.client('s3')
#         paginator = s3_client.get_paginator('list_objects_v2')
#         transcripts ={}
#         paired_samples = {}
#         index_file_key = f"{self.s3_prefix}_paired_index.json"

#         if use_index_file:
#             try:
#                 index_object = s3_client.get_object(Bucket=self.s3_bucket, Key=index_file_key)
#                 file_content = index_object['Body'].read().decode('utf-8')
#                 paired_samples = json.loads(file_content)
#                 return list(paired_samples.values())
#             except Exception as e:
#                 print(f"Error reading index file '{index_file_key}': {e}")
        
#         for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=self.s3_prefix):
#             for blob in page.get('Contents', []):
#                 key = blob.get('Key')
#                 if key.endswith(".txt"):
#                     response = s3_client.get_object(Bucket=self.s3_bucket, Key=key)
#                     transcript_stream = response["Body"].iter_lines()

#                     for line in transcript_stream:
#                         # Decode the line from bytes to string
#                         line = line.decode("utf-8").strip()
                        
#                         # Assuming each line is structured like: 'fileid_text transcript'
#                         try:
#                             fileid_text, transcript = line.split(" ", 1)
#                             transcripts[fileid_text] = transcript
#                         except ValueError:
#                             # Handle lines that don't have the expected format
#                             continue
#         #now get the audio paths
#         for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=self.s3_prefix):
#             for blob in page.get('Contents', []):
#                 key = blob.get('Key')
#                 if key.endswith(".flac"):
#                     #get file name withput_Extenstion FROM KEY
#                     fileid = key.split("/")[-1].split(".")[0]
#                     transcript = None
#                     if fileid in transcripts:
#                         transcript = transcripts[fileid]
#                     if transcript:
#                         paired_samples[fileid] = (key, transcript)

#         if use_index_file and paired_samples:
#             s3_client.put_object(
#                 Bucket=self.s3_bucket,
#                 Key=index_file_key,
#                 Body=json.dumps(paired_samples, indent=4).encode('utf-8'))
        
#         #returnd values as a list
#         return list(paired_samples.values())

#     def get_samples(self, indices: List[int]):
#         samples = []
#         for i in indices:
#             samples.append(self.samples[i])
#         return samples

#main
if __name__ == "__main__":
    dataset_location = 's3://coco-dataset/coco_train.json'
    dataset = MSCOCODataset(dataset_location)