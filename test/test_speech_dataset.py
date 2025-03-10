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
import torchaudio.datasets.librispeech as librispeech

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
    


class LibriSpeechS3Dataset(Dataset):
    def __init__(self, dataset_location, transform=None):
        """
        Args:
            s3_bucket (str): S3 bucket name (e.g., 'my-librispeech-bucket').
            dataset_prefix (str): Path inside S3 bucket (e.g., 'train-clean-100/').
            transform (callable, optional): Transform to apply to the audio samples.
        """
        self.dataset_location = dataset_location
        self.s3_bucket = S3Url(self.dataset_location).bucket
        self.s3_prefix = S3Url(self.dataset_location).key
        self.transform = transform
        #get all samples from S3
        self.paired_samples = self._get_samples_from_s3(use_index_file=True)
        pass

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

    def __len__(self):
        return len(self.paired_samples)
    
    def load_s3_file(self, s3_path):
        """Loads a file from S3 into memory."""
        obj = self.s3.get_object(Bucket=self.s3_bucket, Key=s3_path)
        return io.BytesIO(obj["Body"].read())

    def __getitem__(self, idx):
        # Load audio
        audio_key, transcript = self.paired_samples[idx]

        # Load audio from S3
        audio_data = self.s3_client.get_object(Bucket=self.s3_bucket, Key=audio_key)["Body"].read()
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_data))

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, sample_rate, transcript


def train_speech_model(dataset_location, num_epochs=5):

    transform = T.MelSpectrogram(sample_rate=16000, n_mels=128)
    # Create dataset and dataloader
    dataset = LibriSpeechS3Dataset(dataset_location, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = torchaudio.models.Conformer()
     # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            waveforms, sample_rates, transcripts = batch
            # Convert audio to Mel spectrograms
            outputs = model(waveforms)  # Pass through model
            print(outputs.shape)  # (batch, time, classes)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")



if __name__ == "__main__":
    # Initialize the dataset

    train_speech_model("s3://disdlspeech/test-clean", num_epochs=5)

