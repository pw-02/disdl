from common.utils import S3Url
import boto3
import json
from typing import List, Tuple, Dict
import functools
from itertools import cycle
import random
from common.batch import Batch
import random

class Dataset():
    def __init__(self, 
                 dataset_location: str, 
                 transforms=None):
        
        self.dataset_location = dataset_location
        self.s3_bucket = S3Url(self.dataset_location).bucket
        self.s3_prefix = S3Url(self.dataset_location).key
        self.transforms = transforms
        self.samples = self._get_samples_from_s3()

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
                Body=json.dumps(paired_samples, indent=4).encode('utf-8'))

        return paired_samples
    
    def get_samples(self, indices: List[int]):
        samples = []
        for i in indices:
            samples.append(self._classed_items[i])
        return samples
    
    def dataset_info(self):
        return {
            "num_samples": len(self),
            
        }

#main
if __name__ == "__main__":
    pass