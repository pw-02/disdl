import functools
import json
import logging
import boto3
import pandas as pd
from io import StringIO
from typing import List, Tuple, Dict
from disdl.utils.utils import S3Url

logger = logging.getLogger(__name__)
class S3DatasetBase:
    def __init__(self, dataset_location: str, batch_size: int, num_partitions: int = 1,
                 shuffle: bool = False, drop_last: bool = False, min_lookahead_steps: int = 50, transforms=None):
        
        self.dataset_location = dataset_location.rstrip('/') + '/'
        self.transforms = transforms
        self.s3_bucket = S3Url(self.dataset_location).bucket
        self.s3_prefix = S3Url(self.dataset_location).key
        self.batch_size = batch_size
        self.num_partitions = num_partitions
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.min_lookahead_steps = min_lookahead_steps
        self.samples = self._get_samples_from_s3()
    
    def get_samples(self, indices: List[int]):
        raise NotImplementedError()

    def _get_samples_from_s3(self):
        raise NotImplementedError()

    def dataset_info(self):
        return {
            "location": self.dataset_location,
            "num_samples": len(self),
            "num_batches": len(self) // self.batch_size,
            "num_partitions": self.num_partitions
        }

    def __len__(self):
        return len(self.samples)

class ImageNetDataset(S3DatasetBase):
    def _get_samples_from_s3(self):
        s3_client = boto3.client('s3')
        index_key = f"{self.s3_prefix}_paired_index.json"

        try:
            obj = s3_client.get_object(Bucket=self.s3_bucket, Key=index_key)
            return json.loads(obj['Body'].read().decode('utf-8'))
        except Exception as e:
            logger.warning(f"Falling back to full scan for data location: {e}")

        samples = {}
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=self.s3_prefix):
            for blob in page.get('Contents', []):
                key = blob['Key']
                if key.lower().endswith(('.jpg', '.jpeg', '.png')):
                    class_name = key[len(self.s3_prefix):].lstrip('/').split('/')[0]
                    samples.setdefault(class_name, []).append(key)

        s3_client.put_object(Bucket=self.s3_bucket, Key=index_key, Body=json.dumps(samples).encode('utf-8'))
        return samples

    @functools.cached_property
    def _classed_items(self):
        return [(blob, class_idx)
                for class_idx, class_name in enumerate(self.samples)
                for blob in self.samples[class_name]]

    def get_samples(self, indices: List[int]):
        return [self._classed_items[i] for i in indices]

    def __len__(self):
        return len(self._classed_items)


class CIFAR10Dataset(ImageNetDataset):
    pass  # Inherits logic from ImageNetDataset, which is appropriate for this structure





class MSCOCODataset(S3DatasetBase):
    def _get_samples_from_s3(self):
        s3_client = boto3.client('s3')
        try:
            obj = s3_client.get_object(Bucket=self.s3_bucket, Key=self.s3_prefix)
            return json.loads(obj['Body'].read().decode('utf-8'))
        except Exception as e:
            logger.error(f"Failed to load COCO index: {e}")
            return {}

    @functools.cached_property
    def _classed_items(self):
        return [(entry, class_idx)
                for class_idx, class_name in enumerate(self.samples)
                for entry in self.samples[class_name]]

    def get_samples(self, indices: List[int]):
        results = []
        for i in indices:
            sample, image_id = self._classed_items[i]
            image, caption = sample
            results.append((image, caption, image_id))
        return results

    def __len__(self):
        return len(self._classed_items)


class OpenImagesDataset(S3DatasetBase):
    def _get_samples_from_s3(self):
        s3_client = boto3.client('s3')
        index_key = f"{self.s3_prefix}_paired_index.json"
        try:
            obj = s3_client.get_object(Bucket=self.s3_bucket, Key=index_key)
            return list(json.loads(obj['Body'].read().decode('utf-8')).values())
        except Exception as e:
            logger.warning(f"Falling back to scanning OpenImages dataset: {e}")

        images, labels, paired_samples = {}, {}, {}
        paginator = s3_client.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=self.s3_prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
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

        s3_client.put_object(Bucket=self.s3_bucket, Key=index_key, Body=json.dumps(paired_samples).encode('utf-8'))
        return list(paired_samples.values())

    def get_samples(self, indices: List[int]):
        return [self.samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)
    
class S3DatasetFactory:
    @staticmethod
    def create_dataset(dataset_location: str, batch_size: int, num_partitions: int = 1,
                       shuffle: bool = False, drop_last: bool = False, min_lookahead_steps: int = 50,
                       transforms=None) -> S3DatasetBase:
        if "imagenet" in dataset_location.lower():
            return ImageNetDataset(dataset_location, batch_size, num_partitions, shuffle, drop_last, min_lookahead_steps, transforms)
        elif "cifar10" in dataset_location.lower():
            return CIFAR10Dataset(dataset_location, batch_size, num_partitions, shuffle, drop_last, min_lookahead_steps, transforms)
        elif "mscoco" in dataset_location.lower():
            return MSCOCODataset(dataset_location, batch_size, num_partitions, shuffle, drop_last, min_lookahead_steps, transforms)
        elif "openimages" in dataset_location.lower():
            return OpenImagesDataset(dataset_location, batch_size, num_partitions, shuffle, drop_last, min_lookahead_steps, transforms)
        else:
            raise ValueError(f"Unsupported dataset type for location: {dataset_location}")
 