from utils import S3Url
import boto3
import json
from typing import List, Tuple, Dict
import functools
import pandas as pd
from io import StringIO


class MSCOCODataset():
    def __init__(self, 
                 dataset_location: str, 
                 transforms=None):
            self.dataset_location = dataset_location
            self.s3_bucket = S3Url(self.dataset_location).bucket
            self.s3_prefix = S3Url(self.dataset_location).key
            self.annotation_file = self.s3_prefix
            self.samples = self._get_samples_from_s3()
            pass
    
    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [(blob, class_index)
            for class_index, blob_class in enumerate(self.samples)
            for blob in self.samples[blob_class]]
    
    def _get_samples_from_s3(self, use_index_file=True, images_only=True) -> Dict[str, List[str]]:
        s3_client = boto3.client('s3')
        index_object = s3_client.get_object(Bucket=self.s3_bucket, Key=self.annotation_file)
        file_content = index_object['Body'].read().decode('utf-8')
        # samples = json.loads(file_content)
        paired_samples = json.loads(file_content)
        return paired_samples
    
    def __len__(self) -> int:
        return sum(len(class_items) for class_items in self.samples.values())
    
    def get_samples(self, indices: List[int]):
        samples = []
        for i in indices:
            sample, image_id = self._classed_items[i]
            image, cpation = sample
            samples.append((image, cpation, image_id))
        return samples
    
    def dataset_info(self):
        return {
            "num_samples": len(self)}


class OpenImagesDataset():
    def __init__(self, 
                 dataset_location: str, 
                 transforms=None):
        self.dataset_location = dataset_location
        self.s3_bucket = S3Url(self.dataset_location).bucket
        self.s3_prefix = S3Url(self.dataset_location).key
        self.transforms = transforms
        self.samples = self._get_samples_from_s3()
        pass
        
    def _get_samples_from_s3(self, use_index_file=False, images_only=True) -> Dict[str, List[str]]:
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
                        image_label_dict = df.set_index("ImageID")["LabelName"].to_dict()

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
    
    def get_samples(self, indices: List[int]):
        samples = []
        for i in indices:
            samples.append(self._classed_items[i])
        return samples
    
    def dataset_info(self):
        return {
            "num_samples": len(self)}

class ImageNetDataset():
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
    
    def _get_samples_from_s3(self, use_index_file=False, images_only=True) -> Dict[str, List[str]]:
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
    
    def get_samples(self, indices: List[int]):
        samples = []
        for i in indices:
            samples.append(self._classed_items[i])
        return samples
    
    def dataset_info(self):
        return {
            "num_samples": len(self)}

class LibSpeechDataset():
    def __init__(self, 
                 dataset_location: str, 
                 transforms=None):
        
        self.dataset_location = dataset_location
        self.s3_bucket = S3Url(self.dataset_location).bucket
        self.s3_prefix = S3Url(self.dataset_location).key
        self.transforms = transforms
        self.samples = self._get_samples_from_s3()
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _get_samples_from_s3(self, use_index_file=False) -> Dict[str, List[str]]:
        s3_client = boto3.client('s3')
        paginator = s3_client.get_paginator('list_objects_v2')
        transcripts ={}
        paired_samples = {}
        index_file_key = f"{self.s3_prefix}_paired_index.json"

        if use_index_file:
            try:
                index_object = s3_client.get_object(Bucket=self.s3_bucket, Key=index_file_key)
                file_content = index_object['Body'].read().decode('utf-8')
                paired_samples = json.loads(file_content)
                return list(paired_samples.values())
            except Exception as e:
                print(f"Error reading index file '{index_file_key}': {e}")
        
        for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=self.s3_prefix):
            for blob in page.get('Contents', []):
                key = blob.get('Key')
                if key.endswith(".txt"):
                    response = s3_client.get_object(Bucket=self.s3_bucket, Key=key)
                    transcript_stream = response["Body"].iter_lines()

                    for line in transcript_stream:
                        # Decode the line from bytes to string
                        line = line.decode("utf-8").strip()
                        
                        # Assuming each line is structured like: 'fileid_text transcript'
                        try:
                            fileid_text, transcript = line.split(" ", 1)
                            transcripts[fileid_text] = transcript
                        except ValueError:
                            # Handle lines that don't have the expected format
                            continue
        #now get the audio paths
        for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=self.s3_prefix):
            for blob in page.get('Contents', []):
                key = blob.get('Key')
                if key.endswith(".flac"):
                    #get file name withput_Extenstion FROM KEY
                    fileid = key.split("/")[-1].split(".")[0]
                    transcript = None
                    if fileid in transcripts:
                        transcript = transcripts[fileid]
                    if transcript:
                        paired_samples[fileid] = (key, transcript)

        if use_index_file and paired_samples:
            s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=index_file_key,
                Body=json.dumps(paired_samples, indent=4).encode('utf-8'))
        
        #returnd values as a list
        return list(paired_samples.values())

    def get_samples(self, indices: List[int]):
        samples = []
        for i in indices:
            samples.append(self.samples[i])
        return samples

#main
if __name__ == "__main__":
    dataset_location = 's3://coco-dataset/coco_train.json'
    dataset = MSCOCODataset(dataset_location)