import boto3
import os
import time
import argparse
from datetime import datetime
from datetime import datetime, timezone
import concurrent.futures
from create_bill import prarse_exported_logs

def export_logs_to_s3(log_group, s3_bucket_name, s3_prefix, from_time, to_time):

    if to_time is None:
          to_time = int(time.time() * 1000)
    else:
        to_time = convert_to_millis(to_time)

    if from_time is None:
        from_time = 0
    else:
        from_time = convert_to_millis(from_time)

    # Set toTime to current timestamp in milliseconds
    to_time = int(time.time() * 1000)



    task_name = f"export-{log_group}-{int(time.time())}"
    client = boto3.client("logs")

    response = client.create_export_task(
        taskName=task_name,
        logGroupName=log_group,
        fromTime=from_time,
        to=to_time,
        destination=s3_bucket_name,
        destinationPrefix=s3_prefix
    )
    task_id = response['taskId']
    while True:
        request = client.describe_export_tasks(taskId=task_id)
        status = request['exportTasks'][0]['status']['code']
        print(f'Task ID {task_id} status: {status}')

        if status in ['COMPLETED', 'FAILED']:
            break  
        # Wait for a while before checking the status again
        time.sleep(5)
    print(f"Export task created: {response['taskId']} for log group {log_group}")



def download_logs_from_s3(s3_bucket_name, s3_prefix, download_path):
    s3_client = boto3.client('s3')

    """Downloads exported logs from S3."""
    response = s3_client.list_objects_v2(Bucket=s3_bucket_name, Prefix=s3_prefix)
    if 'Contents' not in response:
        print(f"No logs found in S3 bucket {s3_bucket_name} with prefix {s3_prefix}.")
        return
    
    # Iterate over each file in the S3 prefix
    for obj in response['Contents']:
        # Get the S3 object key
        s3_key = obj['Key']
        
        # Create corresponding local path
        local_path = os.path.join(download_path, *s3_key.split('/'))
        local_path = os.path.normpath(local_path)  # Normalize path for Windows

        # Ensure the local directory exists
        local_dirname = os.path.dirname(local_path)
        os.makedirs(local_dirname, exist_ok=True)
        
        # Download the S3 object to the local path
        s3_client.download_file(s3_bucket_name, s3_key, local_path)
# Function to retrieve all log groups
def get_all_log_groups():
    log_groups = []
    next_token = None
    client = boto3.client('logs')
    paginator = client.get_paginator("describe_log_groups")

    for page in paginator.paginate():
        for group in page.get("logGroups", []):
            log_groups.append(group["logGroupName"])
    
    return log_groups


# Function to list all Lambda functions
def list_lambda_functions():
    client = boto3.client('lambda')

    functions = []
    paginator = client.get_paginator('list_functions')


    for page in paginator.paginate():
        for function in page['Functions']:
            functions.append(f"/aws/lambda/{function['FunctionName']}")

    return functions


def get_cloud_watch_logs_for_experiment(download_dir, s3_bucket_name, from_time, to_time=None):
    os.makedirs(download_dir, exist_ok=True)
    log_groups = get_all_log_groups()
    lambda_functions = list_lambda_functions()
    # log_groups.append({'logGroupName': '/aws/lambda/lambda_function'})
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        # Export logs in parallel
        futures = []
        for log_group_name in log_groups:
            if log_group_name not in lambda_functions:
                continue

            s3_prefix = f'disdl/{log_group_name.replace("/", "_")}'
            futures.append(executor.submit(export_logs_to_s3, 
                                           log_group_name, 
                                           s3_bucket_name, 
                                           s3_prefix,
                                           '2025-03-11 00:00:00', 
                                            None))
        
        # Wait for all export tasks to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # re-raise any exceptions that occurred during export
            except Exception as e:
                print(f'Exception during log export: {e}')

        # Download logs from S3 in parallel
        for log_group_name in log_groups:
            if log_group_name not in lambda_functions:
                continue
            s3_prefix = f'cloudwatchresnet/{log_group_name.replace("/", "_")}'
            executor.submit(download_logs_from_s3, s3_bucket_name, s3_prefix, download_dir)
            
        # Wait for all download tasks to complete
        # (You may need to use additional synchronization here depending on your needs)

def convert_to_millis(timestamp_str):
    # Format: "YYYY-MM-DD HH:MM:SS"
    dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    return int(time.mktime(dt.timetuple()) * 1000)

# def convert_to_millis(date_str):
#     # Parse the input string as a naive datetime object
#     dt = datetime.strptime(date_str, '%Y-%m-%d_%H-%M-%S')
    
#     # Make the datetime object timezone-aware (UTC)
#     dt_utc = dt.replace(tzinfo=timezone.utc)
    
#     # Convert to milliseconds since the Unix epoch
#     return int(dt_utc.timestamp() * 1000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CloudWatch logs to S3 and download them.")
    parser.add_argument("--download_dir", help="Directory to download the logs to", default="logs")
    parser.add_argument("--s3_bucket_name", help="S3 bucket name for exporting logs", default="supercloudwtachexports")
    parser.add_argument("--start_time", help="", default='2025-03-11_00-32-19')
    parser.add_argument("--end_time", help="",  default='2025-03-05_13-06-19')
    args = parser.parse_args()
    #10/23/2024  11:44:40 PM
    # 2024-10-24_18-27-38
    get_cloud_watch_logs_for_experiment(args.download_dir, args.s3_bucket_name, args.start_time, args.end_time)
    #prarse_exported_logs(args.download_dir)
