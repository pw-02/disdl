from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import boto3
import redis
from PIL import Image
import torch
import torchvision.transforms as transforms
import botocore.config
import re
from transformers.models.bert.tokenization_bert import BertTokenizer
from typing import Any, List, Optional, Tuple, Union
from torch.nn.utils.rnn import pad_sequence
from torch.nn import Module
import time
import lz4.frame

# mean and standard deviation from the ALBEF repo:
# https://github.com/salesforce/ALBEF/blob/main/dataset/__init__.py#L16
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD_DEV = (0.26862954, 0.26130258, 0.27577711)

tokenizer = BertTokenizer.from_pretrained("/var/task/bert-tokenizer")

# Create the S3 client with the custom config
# Create the S3 client with the custom config
s3_client = boto3.client('s3', config=botocore.config.Config(
    max_pool_connections=500
))
redis_client = None

class Truncate(Module):
    r"""Truncate input sequence

    :param max_seq_len: The maximum allowable length for input sequence
    :type max_seq_len: int
    """

    def __init__(self, max_seq_len: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sequence or batch of sequence to be truncated
        :type input: Union[List[Union[str, int]], List[List[Union[str, int]]]]
        :return: Truncated sequence
        :rtype: Union[List[Union[str, int]], List[List[Union[str, int]]]]
        """
        if torch.jit.isinstance(input, List[int]):
            return input[:self.max_seq_len]
        elif torch.jit.isinstance(input, List[str]):
            return input[:self.max_seq_len]
        elif torch.jit.isinstance(input, List[List[int]]):
            output: List[List[int]] = []
            for ids in input:
                output.append(ids[:self.max_seq_len])
            return output
        elif torch.jit.isinstance(input, List[List[str]]):
            output: List[List[str]] = []
            for ids in input:
                output.append(ids[:self.max_seq_len])
            return output
        else:
            raise TypeError("Input type not supported")


class Sequential(torch.nn.Sequential):
    r"""A container to host a sequence of text transforms."""

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sequence or batch. The input type must be supported by the first transform in the sequence.
        :type input: `Any`
        """
        for module in self:
            input = module(input)
        return input

class ToTensor(Module):
    r"""Convert input to torch tensor

    :param padding_value: Pad value to make each input in the batch of length equal to the longest sequence in the batch.
    :type padding_value: Optional[int]
    :param dtype: :class:`torch.dtype` of output tensor
    :type dtype: :class:`torch.dtype`
    """

    def __init__(self, padding_value: Optional[int] = None, dtype: torch.dtype = torch.long) -> None:
        super().__init__()
        self.padding_value = padding_value
        self.dtype = dtype

    def forward(self, input: Any) -> torch.Tensor:
        """
        :param input: Sequence or batch of token ids
        :type input: Union[List[int], List[List[int]]]
        :rtype: Tensor
        """
        if torch.jit.isinstance(input, List[int]):
            return torch.tensor(input, dtype=torch.long)
        elif torch.jit.isinstance(input, List[List[int]]):
            if self.padding_value is None:
                output = torch.tensor(input, dtype=self.dtype)
                return output
            else:
                output = pad_sequence(
                    [torch.tensor(ids, dtype=self.dtype) for ids in input], batch_first=True, padding_value=float(self.padding_value)
                )
                return output
        else:
            raise TypeError("Input type not supported")


class PadTransform(Module):
    """Pad tensor to a fixed length with given padding value.

    :param max_length: Maximum length to pad to
    :type max_length: int
    :param pad_value: Value to pad the tensor with
    :type pad_value: bool
    """

    def __init__(self, max_length: int, pad_value: int) -> None:
        super().__init__()
        self.max_length = max_length
        self.pad_value = float(pad_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: The tensor to pad
        :type x: Tensor
        :return: Tensor padded up to max_length with pad_value
        :rtype: Tensor
        """
        max_encoded_length = x.size(-1)
        if max_encoded_length < self.max_length:
            pad_amount = self.max_length - max_encoded_length
            x = torch.nn.functional.pad(x, (0, pad_amount), value=self.pad_value)
        return x



class ALBEFTextTransform:
    def __init__(
        self,
        do_pre_process: bool = True,
        truncate: bool = False,
        pad_to_max_seq_len: bool = False,
        add_end_token: bool = True,
        max_seq_len: int = 25,
        cls_token_id: int = 101,
        sep_token_id: int = 102,
        pad_token_id: int = 0,
    ):
        self.do_pre_process = do_pre_process
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.add_end_token = add_end_token

        self.transform = Sequential(
            Truncate(max_seq_len=max_seq_len) if truncate else torch.nn.Identity(),
            ToTensor(padding_value=self.pad_token_id),
            (
                PadTransform(max_length=max_seq_len, pad_value=self.pad_token_id)
                if pad_to_max_seq_len
                else torch.nn.Identity()
            ),
        )

    def pre_process(self, text: str) -> str:
        text = (
            re.sub(
                r"([,.'!?\"()*#:;~])",
                "",
                text,
            )
            .replace("-", " ")
            .replace("/", " ")
        )
        text = text.rstrip(" ")

        return text

    def __call__(self, text: Union[List[str], str]) -> torch.Tensor:
        if self.do_pre_process:
            if isinstance(text, str):
                text = self.pre_process(text)
            else:
                text = [self.pre_process(t) for t in text]
        tokens = tokenizer(text)["input_ids"]
        if not self.add_end_token and tokens[-1] == self.sep_token_id:
            tokens = tokens[:-1]
        input_ids = self.transform(tokens)

        return input_ids


def training_image_transform(
    image_size: int = 384,
    scale: Tuple[float, float] = (0.5, 1.0),
    image_interpolation=transforms.InterpolationMode.BICUBIC,
    mean: Tuple[float, float, float] = MEAN,
    std_dev: Tuple[float, float, float] = STD_DEV,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size, scale=scale, interpolation=image_interpolation
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(2, 7),
            transforms.ToTensor(),
            transforms.Normalize(mean, std_dev),
        ]
    )

def bytes_to_mb(byte_data):
    size_in_bytes = len(byte_data)  # Get size in bytes
    size_in_mb = size_in_bytes / (1024 * 1024)  # Convert to megabytes
    return size_in_mb

def get_data_sample(bucket_name: str, data_sample: tuple, image_transform, text_transform, s3_client) -> tuple:
    """
    Retrieves and transforms a sample from S3.
    """
    # sample, image_id = data_sample
    image_path, caption, image_id= data_sample

    obj = s3_client.get_object(Bucket=bucket_name, Key=image_path)
    image = Image.open(BytesIO(obj['Body'].read())).convert("RGB")
    # Apply transformations
    if image_transform:
        image = image_transform(image)
    if text_transform:
        caption = text_transform(caption)

    return image, caption, image_id

def create_minibatch(bucket_name: str, samples: list, image_transform, text_transform, s3_client) -> str:
    """
    Creates a minibatch from the samples, compresses it, and encodes it in base64.
    """
    image_list, text_list, image_id_list  = [], [], []
    with ThreadPoolExecutor(max_workers=None) as executor:
        futures = {executor.submit(get_data_sample, bucket_name, sample, image_transform, text_transform, s3_client): sample for sample in samples}
        for future in as_completed(futures):
            image, caption, id = future.result()
            image_list.append(image)
            text_list.append(caption)
            image_id_list.append(id)
    
    text = pad_sequence(text_list, batch_first=True)
    minibatch = torch.stack(image_list, dim=0), text, (text != 0).type(torch.long),  torch.Tensor(image_id_list).type(torch.long)
    with BytesIO() as buffer:
        torch.save(minibatch, buffer)
        bytes_minibatch = buffer.getvalue()
        # Encode the serialized tensor with base64
        # compressed_minibatch = lz4.frame.compress(bytes_minibatch)
        compressed_minibatch = lz4.frame.compress(bytes_minibatch)

    return compressed_minibatch

def cache_minibatch_with_retries(redis_client, batch_id, minibatch, max_retries=4, retry_interval=0.1):
    retries = 0
    execption = None
    while retries < max_retries:
        try:
            # Attempt to cache the minibatch in Redis
            redis_client.set(batch_id, minibatch)
            return  # Exit the function on success
        except Exception as e:
            execption = e
            pass
        # Increment the retry count
        retries += 1
        # Wait before retrying
        time.sleep(retry_interval)
    raise execption

def lambda_handler(event, context):
    """
    AWS Lambda handler function that processes a batch of images from an S3 bucket and caches the results in Redis.
    """
    global s3_client
    global redis_client

    try:
        task = event.get('task')
        if task == 'warmup':
            return {'success': True, 'message': 'function warmed'}

        bucket_name = event.get('bucket_name')
        batch_samples = event.get('batch_samples')
        batch_id = event.get('batch_id')
        cache_address = event.get('cache_address', None)

        if not all([bucket_name, batch_samples, batch_id, cache_address]):
            return {'success': False, 'message': 'Missing parameters'}
        
        cache_host, cache_port = cache_address.split(":")
        image_transform = training_image_transform()
        text_transform = ALBEFTextTransform(
            truncate=True, pad_to_max_seq_len=True, max_seq_len=30, add_end_token=False)

        minibatch = create_minibatch(bucket_name, batch_samples, image_transform, text_transform, s3_client)
        # minibatch_size_mb = bytes_to_mb(minibatch)
        # redis_client = None
        if redis_client is None:
            redis_client = redis.StrictRedis(host=cache_host, port=int(cache_port))
            # redis_client = redis.StrictRedis(host=cache_host, port=int(cache_port), ssl=True )
      
        cache_minibatch_with_retries(redis_client, batch_id, minibatch)

        return {
            'success': True,
            'is_cached': True,
            'message': f"Successfully cached '{batch_id}'"
        }
    except Exception as e:
        return {
            'success': False,
            'message': str(e)
        }


if __name__ == '__main__':
    # Define the data dictionary with detailed formatting

    data = {"bucket_name": "coco-dataset", 
            "batch_id": "1_1_5_a459f276d19e8810", 
            "batch_samples": [[["train2014/COCO_train2014_000000239736.jpg", "Blue and white toothbrush in a cage of beetles."], 74562], [["train2014/COCO_train2014_000000509128.jpg", "A man prepares to throw a white Frisbee."], 81341], [["train2014/COCO_train2014_000000303311.jpg", "A group of women standing next to each other holding a glass of wine."], 37885], [["train2014/COCO_train2014_000000401428.jpg", "A snow boarder riding down a snow covered hill."], 33426]], 
            "cache_address": "10.0.17.5:6378", 
            "task": "prefetch"}
    # data = {
    #     {"bucket_name": "coco-dataset", "batch_id": "1_1_6_f5be86d838f7f48f", "batch_samples": [[["train2014/COCO_train2014_000000240457.jpg", "There is a an exit sign next to a clock"], 19426], [["train2014/COCO_train2014_000000096450.jpg", "A close up of two tennis players with one holding a racket."], 37589], [["train2014/COCO_train2014_000000290110.jpg", "Fire and Rescue Command Truck parked in parking lot.\\n "], 80244], [["train2014/COCO_train2014_000000279129.jpg", "A baseball player holding a bat on top of a field."], 45941]], "cache_address": "35.91.49.129:6378", "task": "prefetch"}
    # }

    # Call the lambda_handler function with the defined data
    lambda_handler(data, None)
