import boto3
from datetime import datetime, timedelta
import time
import os
import gzip
import glob
import csv
import re
# Initialize the CloudWatch Logs client
# Example timestamps (replace these with your experiment start and end times)
# experiment_start_time = '2024-09-02T10:00:00Z'
# experiment_end_time = '2024-09-03T12:00:00Z'
# # Convert ISO 8601 format to datetime object
# start_time = datetime.fromisoformat(experiment_start_time.replace('Z', '+00:00'))
# end_time = datetime.fromisoformat(experiment_end_time.replace('Z', '+00:00'))

# # Convert datetime to Unix timestamp (milliseconds)
# start_timestamp = int(start_time.timestamp() * 1000)
# end_timestamp = int(end_time.timestamp() * 1000)

# Function to get current AWS costs using Cost Explorer

def get_lambda_cost_last_day():
    # Create a Cost Explorer client
    client = boto3.client('ce')

    # Define the date range for the last day
    end_date = datetime.utcnow().date()  # Current date in UTC
    start_date = end_date - timedelta(days=1)  # One day before the current date

    # Convert dates to string format
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Query Cost Explorer
    response = client.get_cost_and_usage(
        TimePeriod={
            'Start': start_date_str,
            'End': end_date_str
        },
        Granularity='DAILY',
        Metrics=['UnblendedCost'],
        GroupBy=[
            {
                'Type': 'DIMENSION',
                'Key': 'SERVICE'
            }
        ]
    )

    # Extract Lambda costs
    cost_data = response.get('ResultsByTime', [])
    lambda_cost = 0.0

    for result in cost_data:
        groups = result.get('Groups', [])
        for group in groups:
            service_name = group.get('Keys', [])[0]
            if service_name == 'AWS Lambda':
                lambda_cost = float(group.get('Metrics', {}).get('UnblendedCost', {}).get('Amount', 0.0))

    return lambda_cost





def get_current_aws_costs():
    ce_client = boto3.client('ce')  # Cost Explorer client
    response = ce_client.get_cost_and_usage(
        TimePeriod={'Start': '2024-09-16', 'End': '2024-09-17'},  # Example dates
        Granularity='DAILY',
        Metrics=['UnblendedCost']
    )
    total_cost = sum(float(day['Total']['UnblendedCost']['Amount']) for day in response['ResultsByTime'])
    return total_cost


def compute_lambda_costs(total_requests, cost_per_request=0.20, memory_size_mb=128, duration_ms=100):
    """
    Compute the cost of AWS Lambda invocations.

    Parameters:
    - total_requests (int): The total number of Lambda invocations.
    - cost_per_request (float): The cost per million Lambda requests (default is $0.20).
    - memory_size_mb (int): The amount of memory allocated to the Lambda function (default is 128 MB).
    - duration_ms (int): The duration of each Lambda execution in milliseconds (default is 100 ms).

    Returns:
    - total_cost (float): The total cost of Lambda invocations.
    """

    # Cost per request calculation
    cost_per_request_total = (total_requests / 1_000_000) * cost_per_request
    
    # Convert duration to GB-seconds (memory_size_mb / 1024 = memory_size_gb)
    memory_size_gb = memory_size_mb / 1024
    duration_seconds = duration_ms / 1000
    cost_per_invocation = memory_size_gb * duration_seconds * 0.00001667  # $0.00001667 per GB-second

    # Total invocation cost
    invocation_cost_total = total_requests * cost_per_invocation
    
    # Combine the cost per request and the cost per invocation
    total_cost = cost_per_request_total + invocation_cost_total

    return cost_per_request_total, invocation_cost_total, total_cost



def prarse_exported_logs(destination_folder = 'cloudwatch', export_prefix = 'cloudwatchlogs', skip_unzip = False):
    base_dir = os.path.join(destination_folder, export_prefix)
    output_file = os.path.join(base_dir, "bill.csv")
    decompressed_file_paths = []
    base_dir = os.path.join(destination_folder, export_prefix)

    if not skip_unzip:
         # Traverse the directory hierarchy
        for dirpath, _, filenames in os.walk(base_dir):
            for filename in filenames:
                if filename.endswith('.gz'):
                    gz_file_path = os.path.join(dirpath, filename)
                    decompressed_file_path = gz_file_path[:-3]  # Remove '.gz' suffix
                    
                    # Decompress the file
                    with gzip.open(gz_file_path, 'rb') as f_in:
                        with open(decompressed_file_path, 'wb') as f_out:
                            f_out.write(f_in.read())
                    
                    # Save the decompressed file path
                    decompressed_file_paths.append(decompressed_file_path)

    log_data = []
    start_pattern = re.compile(r'START RequestId: ([\w-]+)')
    report_pattern = re.compile(r'REPORT RequestId: ([\w-]+)\s*Duration: ([\d.]+) ms\s*Billed Duration: ([\d.]+) ms\s*Memory Size: (\d+) MB\s*Max Memory Used: (\d+) MB(?:\s*Init Duration: ([\d.]+) ms)?')
    timestamp_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z)')

    for log_file in decompressed_file_paths:
        with open(log_file, 'r') as log_file:
            content = log_file.read()
            # Extracting logs using regex
            current_request = {}
            for line in content.splitlines():
                start_match = start_pattern.search(line)
                report_match = report_pattern.search(line)
                
                if start_match:
                    request_id = start_match.group(1)
                    current_request = {
                        'System':'InfiniSore' if 'CacheNode' in str(log_file) else 'PREFETCH',
                        'Timestamp': '',
                        'RequestId': request_id,
                        'Duration': '0',
                        'Billed Duration': '0',
                        'Memory Size': '0',
                        'Max Memory Used': '0',
                        'Init Duration': '0'
                    }
                
                if report_match:
                    request_id, duration, billed_duration, memory_size, max_memory_used, init_duration = report_match.groups()
                    if request_id == current_request.get('RequestId'):
                        match = timestamp_pattern.search(line)
                        timestamp = match.group(1)
                        current_request.update({
                           'Timestamp': timestamp,
                            'RequestId': request_id,
                            'Duration': duration,
                            'Billed Duration': billed_duration,
                            'Memory Size': memory_size,
                            'Max Memory Used': max_memory_used,
                            'Init Duration': init_duration if init_duration else '0'
                        })
                        compute_cost, request_cost, total_cost = compute_lambda_costs(
                            total_requests=1,
                            cost_per_request=0.20,
                            memory_size_mb=int(memory_size),
                            duration_ms=int(billed_duration))
                                
                        current_request['Compute Cost'] = compute_cost
                        current_request['Request Cost'] = request_cost
                        current_request['Total Cost'] = total_cost

                        # Append data when report is fully parsed
                        log_data.append(current_request)
                        current_request = {}

    
    # Write parsed data to CSV
    with open(output_file, 'w', newline='') as csv_file:
        fieldnames = log_data[0].keys()
        # fieldnames = ['System', 'Timestamp', 'RequestId', 'Duration', 'Billed Duration', 'Memory Size', 'Max Memory Used', 'Init Duration']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_data)
    
if __name__ == '__main__':

    # test = compute_lambda_costs(total_requests=1071,
    #                             cost_per_request=0.20,
    #                             memory_size_mb=3072,
    #                             duration_ms=931)
    
    # cost = get_lambda_cost_last_day()

    export_prefix = 'cloudwatchresnet'
    destination_folder = 'C:\\Users\\pw\\Desktop\\super_results\\image_classification\\super\\imagenet\\multi_job_2025-02-06_19-13-12'
    prarse_exported_logs(destination_folder,export_prefix, False)
