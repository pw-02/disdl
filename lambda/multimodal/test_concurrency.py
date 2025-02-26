import boto3
import concurrent.futures
import time
import json
from urllib.parse import urlparse
from botocore.exceptions import ClientError

# AWS Lambda client setup
lambda_client = boto3.client('lambda')
LAMBDA_FUNCTION_NAME = 'CreateVisionTrainingBatch'
NUM_CONCURRENT_INVOCATIONS = 500
# NUM_BATCHES = 200   

class S3Url:
    def __init__(self, url):
        self._parsed = urlparse(url, allow_fragments=False)

    @property
    def bucket(self):
        return self._parsed.netloc

    @property
    def key(self):
        return self._parsed.path.lstrip('/')

    @property
    def url(self):
        return self._parsed.geturl()

def split_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def invoke_lambda(payload):
    start_time = time.perf_counter()
    response = lambda_client.invoke(
        FunctionName=LAMBDA_FUNCTION_NAME,
        InvocationType='RequestResponse',
        Payload=payload
    )
    duration = time.perf_counter() - start_time
    response_data = json.loads(response['Payload'].read().decode('utf-8'))
    response_data['duration'] = duration
    return response_data

def is_image_file(path):
    return path.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm', '.bmp'))

def load_paired_s3_object_keys(s3_uri, images_only=True, use_index_file=True):
    s3_client = boto3.client('s3')
    s3url = S3Url(s3_uri)
    index_file_key = s3url.key + '_paired_index.json'
    paired_samples = {}

    if use_index_file:
        try:
            index_object = s3_client.get_object(Bucket=s3url.bucket, Key=index_file_key)
            paired_samples = json.loads(index_object['Body'].read().decode('utf-8'))
            return paired_samples
        except ClientError as e:
            print(f"Error reading index file '{index_file_key}': {e}")

    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=s3url.bucket, Prefix=s3url.key):
        for blob in page.get('Contents', []):
            blob_path = blob.get('Key')
            if blob_path.endswith("/"):
                continue  # Ignore folders

            stripped_path = blob_path[len(s3url.key):].lstrip("/")
            if stripped_path == blob_path or (images_only and not is_image_file(blob_path)):
                continue

            if 'index.json' in blob_path:
                continue

            blob_class = stripped_path.split("/")[0]
            paired_samples.setdefault(blob_class, []).append(blob_path)

    if use_index_file and paired_samples:
        s3_client.put_object(Bucket=s3url.bucket, Key=index_file_key, 
                             Body=json.dumps(paired_samples, indent=4).encode('UTF-8'))

    return paired_samples

def process_batches(payloads):
    total_latency = 0

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(invoke_lambda, payload) for payload in payloads]
        for future in concurrent.futures.as_completed(futures):
            try:
                response = future.result()
                total_latency += response["duration"]
                print(f'Invocation response: {response}')
            except Exception as e:
                print(f'Invocation error: {e}')
    return total_latency
def test_concurrency():


    # invoke_lambda(json.dumps({'task': 'warmup'}))
    dataset = load_paired_s3_object_keys('s3://imagenet1k-sdl/train/')
    all_samples = [(path, 0) for key in dataset for path in dataset[key]]
    payloads = []
    batches = split_list(all_samples, 128)[:NUM_CONCURRENT_INVOCATIONS]

    for idx, batch in enumerate(batches):
        payload = {
            'bucket_name': 'imagenet1k-sdl',
            'batch_id': str(idx + 1),
            'batch_samples': batch,
            'task': 'prefetch',
            'cache_address': '10.0.19.221:6378'
        }
        payloads.append(json.dumps(payload))

    return process_batches(payloads)

if __name__ == '__main__':
    start_time = time.time()
    total_latency = test_concurrency()
    end_time = time.time()
    total_time = end_time - start_time
    avg_latency = total_latency / NUM_CONCURRENT_INVOCATIONS * 1000
    throuhgput = NUM_CONCURRENT_INVOCATIONS/total_time
    print(f'Total time: {total_time} seconds, Throuhgput {throuhgput}, Avg Latency {avg_latency}')
