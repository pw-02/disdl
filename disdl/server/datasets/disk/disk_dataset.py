import functools
import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)


class DiskDatasetBase:
    def __init__(self, dataset_location: str, batch_size: int, num_partitions: int = 1,
                 shuffle: bool = False, drop_last: bool = False, min_lookahead_steps: int = 50, transforms=None):
        
        self.dataset_location = Path(dataset_location).expanduser().resolve()
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_partitions = num_partitions
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.min_lookahead_steps = min_lookahead_steps
        self.samples = self._get_samples_from_disk()

    def get_samples(self, indices: List[int]):
        raise NotImplementedError()

    def _get_samples_from_disk(self):
        raise NotImplementedError()

    def dataset_info(self):
        return {
            "location": str(self.dataset_location),
            "num_samples": len(self),
            "num_batches": len(self) // self.batch_size,
            "num_partitions": self.num_partitions
        }

    def __len__(self):
        return len(self.samples)


class ImageNetDataset(DiskDatasetBase):
    def _get_samples_from_disk(self):
        index_file = self.dataset_location / "_paired_index.json"
        if index_file.exists():
            return json.loads(index_file.read_text())

        samples = {}
        for class_dir in sorted(self.dataset_location.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            for img_path in class_dir.rglob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    samples.setdefault(class_name, []).append(str(img_path.resolve()))

        index_file.write_text(json.dumps(samples))
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
    pass  # Same directory structure


class MSCOCODataset(DiskDatasetBase):
    def _get_samples_from_disk(self):
        index_file = self.dataset_location / "_paired_index.json"
        if index_file.exists():
            return json.loads(index_file.read_text())

        annotations_file = self.dataset_location / "annotations.csv"
        if not annotations_file.exists():
            raise FileNotFoundError(f"Expected COCO-style annotations.csv at {annotations_file}")

        df = pd.read_csv(annotations_file)
        labels = df.set_index("ImageID")["Ids"].to_dict()

        samples = {}
        for img_path in self.dataset_location.rglob("*.jpg"):
            fileid = img_path.stem
            if fileid in labels:
                class_name = labels[fileid]
                samples.setdefault(class_name, []).append((str(img_path.resolve()), fileid))

        index_file.write_text(json.dumps(samples))
        return samples

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


class OpenImagesDataset(DiskDatasetBase):
    def _get_samples_from_disk(self):
        index_file = self.dataset_location / "_paired_index.json"
        if index_file.exists():
            return list(json.loads(index_file.read_text()).values())

        images, labels, paired_samples = {}, {}, {}

        for f in self.dataset_location.rglob("*.csv"):
            df = pd.read_csv(f)
            labels.update(df.set_index("ImageID")["Ids"].to_dict())

        for img_path in self.dataset_location.rglob("*.jpg"):
            fileid = img_path.stem
            images[fileid] = str(img_path.resolve())

        for image_id in labels:
            if image_id in images:
                paired_samples[image_id] = (images[image_id], labels[image_id])

        index_file.write_text(json.dumps(paired_samples))
        return list(paired_samples.values())

    def get_samples(self, indices: List[int]):
        return [self.samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)


class DiskDatasetFactory:
    @staticmethod
    def create_dataset(dataset_location: str, batch_size: int, num_partitions: int = 1,
                       shuffle: bool = False, drop_last: bool = False, min_lookahead_steps: int = 50,
                       transforms=None) -> DiskDatasetBase:
        path = Path(dataset_location).as_posix().lower()

        if "imagenet" in path:
            return ImageNetDataset(dataset_location, batch_size, num_partitions, shuffle, drop_last, min_lookahead_steps, transforms)
        elif "cifar10" in path:
            return CIFAR10Dataset(dataset_location, batch_size, num_partitions, shuffle, drop_last, min_lookahead_steps, transforms)
        elif "mscoco" in path:
            return MSCOCODataset(dataset_location, batch_size, num_partitions, shuffle, drop_last, min_lookahead_steps, transforms)
        elif "openimages" in path:
            return OpenImagesDataset(dataset_location, batch_size, num_partitions, shuffle, drop_last, min_lookahead_steps, transforms)
        else:
            raise ValueError(f"Unsupported dataset type for location: {dataset_location}")
