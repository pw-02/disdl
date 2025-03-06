from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import boto3
import redis
from PIL import Image
import torch
import torchvision.transforms as transforms
import lz4.frame
import botocore.config
import time
# import zstandard as zstd

# Create the S3 client with the custom config
s3_client = boto3.client('s3', config=botocore.config.Config(
    max_pool_connections=500
))
redis_client = None
# compressor = zstd.ZstdCompressor(level=-1)

# def dict_to_torchvision_transform(transform_dict):
#     """
#     Converts a dictionary of transformations to a PyTorch transform object.
#     """
#     transform_list = []
#     for transform_name, params in transform_dict.items():
#         if transform_name == 'Resize':
#             transform_list.append(transforms.Resize(params))
#         elif transform_name == 'Normalize':
#             transform_list.append(transforms.Normalize(mean=params['mean'], std=params['std']))
#         elif params is None:
#             transform_list.append(getattr(transforms, transform_name)())
#         else:
#             raise ValueError(f"Unsupported transform: {transform_name}")
#     return transforms.Compose(transform_list)

def is_image_file(path: str) -> bool:
    """
    Checks if the file is an image based on its extension.
    """
    return any(path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.ppm', '.bmp'])

def get_transform(bucket_name: str):
    # Load image
    if 'imagenet1k-sdl' in bucket_name or 'sdl-cifar10' in bucket_name:
       normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225],
        )
       return transforms.Compose([
            transforms.Resize(256),                    # Resize the image to 256x256 pixels
            transforms.RandomResizedCrop(224),   # Randomly crop a 224x224 patch
            transforms.RandomHorizontalFlip(), # Randomly flip the image horizontally
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly change brightness, contrast, saturation, and hue
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            normalize,
        ])

    # elif 'sdl-cifar10' in bucket_name:
    #      return transforms.Compose([
    #     transforms.Resize(224),
    #         transforms.RandomHorizontalFlip(),        # Random horizontal flip
    #         transforms.RandomCrop(32, padding=4),
    #         # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly change brightness, contrast, saturation, and hue
    #         # transforms.RandomRotation(15),      # Randomly rotate images by up to 15 degrees
    #         transforms.ToTensor(),                    # Convert to tensor
    #         transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]) ]) # Normalize
    #     # return transforms.Compose([
    #     #     transforms.Resize(224),                    # Resize the image to 224x224 pixels
    #     #     transforms.RandomHorizontalFlip(),     # Randomly flip the image horizontally
    #     #     transforms.ToTensor(),                 # Convert the image to a PyTorch tensor
    #     #      transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  # Normalize
    #     #     ])
    elif 'coco' in bucket_name:
        return transforms.Compose([
            transforms.Resize(256),                    # Resize the image to 256x256 pixels
            transforms.RandomResizedCrop(224),   # Randomly crop a 224x224 patch
            transforms.RandomHorizontalFlip(), # Randomly flip the image horizontally
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        ])
    else:
        return None
 

def bytes_to_mb(byte_data):
    size_in_bytes = len(byte_data)  # Get size in bytes
    size_in_mb = size_in_bytes / (1024 * 1024)  # Convert to megabytes
    return size_in_mb

def get_data_sample(bucket_name: str, data_sample: tuple, transform, s3_client) -> tuple:
    """
    Retrieves and transforms a sample from S3.
    """
    sample_path, sample_label = data_sample
    obj = s3_client.get_object(Bucket=bucket_name, Key=sample_path)
    data = Image.open(BytesIO(obj['Body'].read())).convert("RGB")
    # Apply transformations
    if transform:
        data = transform(data)
    return data, sample_label

def create_minibatch(bucket_name: str, samples: list, transform, s3_client) -> str:
    """
    Creates a minibatch from the samples, compresses it, and encodes it in base64.
    """
    batch_data, batch_labels = [], []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_data_sample, bucket_name, sample, transform, s3_client): sample for sample in samples}
        for future in as_completed(futures):
            data_sample, label = future.result()
            batch_data.append(data_sample)
            batch_labels.append(label)
            
    minibatch = torch.stack(batch_data), torch.tensor(batch_labels)
    with BytesIO() as buffer:
        torch.save(minibatch, buffer)
        bytes_minibatch = buffer.getvalue()
        # Encode the serialized tensor with base64
        compressed_minibatch = lz4.frame.compress(bytes_minibatch)
        # compressed_minibatch = compressor.compress(bytes_minibatch)
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
        transformformation = get_transform(bucket_name)
        minibatch = create_minibatch(bucket_name, batch_samples, transformformation, s3_client)
        minibatch_size_mb = bytes_to_mb(minibatch)
        
        if redis_client is None:
            redis_client = redis.StrictRedis(host=cache_host, port=int(cache_port))
            # redis_client = redis.StrictRedis(host=cache_host, port=int(cache_port), ssl=True )
        cache_minibatch_with_retries(redis_client, batch_id, minibatch)

        # redis_client.set(batch_id, minibatch)

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

    data = {
        "bucket_name": "imagenet1k-sdl",
        "batch_id": 2,
        "batch_samples": [
            ["train/n01440764/n01440764_10026.JPEG", 0],
            ["train/n01440764/n01440764_10026.JPEG", 0],
            ["train/n01440764/n01440764_10026.JPEG", 0],
        ],
        "cache_address": "54.184.21.219:6378",
        "task": "vision"
    }

    # Call the lambda_handler function with the defined data
    lambda_handler(data, None)
