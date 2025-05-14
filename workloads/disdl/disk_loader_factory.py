from abc import ABC, abstractmethod
from typing import List, Tuple, Any
import torch
from PIL import Image
import torchvision.transforms as T
import time
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# --- Abstract Base Class ---
class BaseDiskLoader(ABC):
    @abstractmethod
    def load_sample_list(self) -> List[Tuple[str, int]]:
        pass

    @abstractmethod
    def load_batch(self, samples: List[Tuple[str, int]]) -> Tuple[Any, Any, float, float]:
        pass

# --- ImageNet-style Loader for Local Disk ---
class ImageNetDiskLoader(BaseDiskLoader):
    def __init__(self, dataset_location: str, transform=None):
        self.dataset_location = Path(dataset_location).expanduser().resolve()
        self.transform = transform or T.ToTensor()

    def load_sample_list(self) -> List[Tuple[str, int]]:
        samples = {}

        for class_dir in sorted(self.dataset_location.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            samples[class_name] = []
            for img_path in class_dir.rglob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    samples[class_name].append(str(img_path.resolve()))

        return samples

    def load_batch(self, samples: List[Tuple[str, int]]) -> Tuple[List[torch.Tensor], List[int], float, float]:
        retrieval_start = time.perf_counter()

        def load_image(path):
            try:
                return Image.open(path).convert("RGB")
            except Exception as e:
                logger.warning(f"[FETCH FAIL] {path}: {e}")
                return Image.new("RGB", (224, 224))

        with ThreadPoolExecutor(max_workers=None) as executor:
            images = list(executor.map(lambda s: load_image(s[0]), samples))

        retrieval_time = time.perf_counter() - retrieval_start

        transform_start = time.perf_counter()
        batch_data, batch_labels = [], []
        for img, (_, label) in zip(images, samples):
            try:
                batch_data.append(self.transform(img))
            except Exception as e:
                logger.warning(f"[TRANSFORM FAIL] {e}")
                batch_data.append(torch.zeros(3, 224, 224))
            batch_labels.append(label)

        transform_time = time.perf_counter() - transform_start
        return batch_data, batch_labels, retrieval_time, transform_time

# --- Factory for Disk Loaders ---
class DiskLoaderFactory:
    @staticmethod
    def create(dataset_name: str, dataset_location: str, transform=None) -> BaseDiskLoader:
        if "imagenet" in dataset_name.lower() or "cifar" in dataset_name.lower():
            return ImageNetDiskLoader(dataset_location, transform)
        else:
            raise ValueError(f"Unsupported dataset for disk loader: {dataset_name}")
