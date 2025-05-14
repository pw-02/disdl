from abc import ABC, abstractmethod
from typing import List, Tuple, Any
import boto3
import torch
from PIL import Image
from io import BytesIO
from typing import List, Tuple
import time
from concurrent.futures import ThreadPoolExecutor
import torchvision.transforms as T
import logging
from urllib.parse import urlparse


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


class BaseS3Loader(ABC):
    @abstractmethod
    def load_batch(self, samples: List[Tuple]) -> Tuple[Any, Any, float, float]:
        """
        Load and return data, labels, and timing info.
        """
        pass

class ImageNetS3Loader(BaseS3Loader):
    def __init__(self, dataset_location: str, transform=None, use_local_folder=False):
        self.dataset_location = dataset_location
        self.transform = transform or T.ToTensor()
        self.use_local_folder = use_local_folder
        self.s3_client = None  # Delay init
    
    def _ensure_s3_client(self):
        if self.s3_client is None and not self.use_local_folder:
            self.s3_client = boto3.client("s3")
    
    def load_batch(self, samples: List[Tuple[str, int]]) -> Tuple[List[torch.Tensor], List[int], float, float]:
        """
        Load a batch of ImageNet-style (image_path, label) pairs.
        Returns: images, labels, retrieval_time, transform_time
        """
        self._ensure_s3_client()
        retrieval_start = time.perf_counter()

        def load_image(path):
            try:
                if self.use_local_folder:
                    with open(path, 'rb') as f:
                        return Image.open(f).convert("RGB")
                else:
                    bucket = S3Url(self.dataset_location).bucket
                    obj = self.s3_client.get_object(Bucket=bucket, Key=path)
                    return Image.open(BytesIO(obj["Body"].read())).convert("RGB")
            except Exception as e:
                logging.warning(f"[FETCH FAIL] {path}: {e}")
                return Image.new("RGB", (224, 224))

        with ThreadPoolExecutor() as executor:
            images = list(executor.map(lambda s: load_image(s[0]), samples))

        retrieval_time = time.perf_counter() - retrieval_start

        transform_start = time.perf_counter()
        batch_data, batch_labels = [], []

        for img, (_, label) in zip(images, samples):
            try:
                batch_data.append(self.transform(img))
            except Exception as e:
                logging.warning(f"[TRANSFORM FAIL] {e}")
                batch_data.append(torch.zeros(3, 224, 224))
            batch_labels.append(label)

        transform_time = time.perf_counter() - transform_start
        return batch_data, batch_labels, retrieval_time, transform_time
    
class S3LoaderFactory:
    @staticmethod
    def create(dataset_name: str, dataset_location: str, transform=None, use_local_folder=False) -> BaseS3Loader:
        if "imagenet" in dataset_name.lower():
            return ImageNetS3Loader(dataset_location, transform, use_local_folder)
        elif "cifar" in dataset_name.lower():
            return ImageNetS3Loader(dataset_location, transform, use_local_folder)
        
        # Later:
        # elif "coco" in dataset_name.lower():
        #     return CocoS3Loader(...)
        # elif "librispeech" in dataset_name.lower():
        #     return LibriSpeechS3Loader(...)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
