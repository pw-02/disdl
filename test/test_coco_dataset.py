import os
import io
import boto3
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio.transforms as T
import torch.nn.functional as F
from typing import Dict, List
import json
from urllib.parse import urlparse
import functools
from typing import List, Dict, Tuple

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


class MSCOCODataset(Dataset):
    def __init__(self, 
                 dataset_location,
                 image_transform=None,
                 text_transform=None,):
        """
        Args:
            dataset_location (str): S3 bucket name (e.g., 'my-coco-bucket').
            transform (callable, optional): Transform to apply to the images.
        """
        self.dataset_location = dataset_location
        self.s3_bucket = S3Url(self.dataset_location).bucket
        self.s3_prefix = S3Url(self.dataset_location).key
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.samples = self._get_samples_from_s3()
    
    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [(blob, class_index)
            for class_index, blob_class in enumerate(self.samples)
            for blob in self.samples[blob_class]]
    
    def _get_sample_list_from_s3(self, use_index_file=True, images_only=True) -> Dict[str, List[str]]:
        s3_client = boto3.client('s3')
        index_object = s3_client.get_object(Bucket=self.s3_bucket, Key=self.annotation_file)
        file_content = index_object['Body'].read().decode('utf-8')
        # samples = json.loads(file_content)
        paired_samples = json.loads(file_content)
        return paired_samples
    
    def __len__(self) -> int:
        return sum(len(class_items) for class_items in self.samples.values())