import boto3
from datetime import datetime, timedelta
import time
import os
import gzip
import glob
import csv
import re
import pandas as pd
from datetime import datetime
def compute_lambda_cost(lambda_data, request_cost_per_million=0.20, compute_cost_per_gb_second=0.00001667):
    """
    Computes AWS Lambda invocation cost based on duration and memory usage.
    
    :param lambda_data: Dictionary with Billed Duration and Memory Size
    :param request_cost_per_million: Cost per 1M requests ($0.20 by default)
    :param compute_cost_per_gb_second: Cost per GB-second ($0.00001667 by default)
    :return: Dictionary with computed cost breakdown
    """
    
    # Extracting values
    billed_duration_s = lambda_data["Billed_Duration_ms"] / 1000  # Convert ms to seconds
    memory_gb = lambda_data["Memory_Size_MB"] / 1024  # Convert MB to GB
    
    # Compute cost (GB-seconds * price per GB-second)
    compute_cost = billed_duration_s * memory_gb * compute_cost_per_gb_second
    
    # Request cost (1 request)
    request_cost = request_cost_per_million / 1_000_000  # Cost per single request
    
    # Total cost
    total_cost = compute_cost + request_cost
    
    return {
        "Compute Cost ($)": round(compute_cost, 6),
        "Request Cost ($)": round(request_cost, 6),
        "Total Cost ($)": round(total_cost, 6),
    }

def parse_report_line(report_line):
    """
    Extracts details from a CloudWatch REPORT log entry.
    
    :param report_line: The log line containing the REPORT entry.
    :return: A dictionary with extracted information.
    """
    report_pattern = re.compile(
        r"REPORT RequestId: ([\w-]+)\s+Duration: ([\d\.]+) ms\s+Billed Duration: (\d+) ms\s+"
        r"Memory Size: (\d+) MB\s+Max Memory Used: (\d+) MB"
    )
    
    match = report_pattern.search(report_line)
    if match:
        return {
            "RequestId": match.group(1),
            "Duration_ms": float(match.group(2)),
            "Billed_Duration_ms": int(match.group(3)),
            "Memory_Size_MB": int(match.group(4)),
            "Max_Memory_Used_MB": int(match.group(5)),
        }
    return None

def extract_lambda_name(path):
    match = re.search(r'_aws_lambda_([^\\\/]+)', path)
    return match.group(1) if match else None


def parse_cloudwatch_logs(data_folder = 'data', prefix = 'disdl', skip_unzip = False):
    base_dir = os.path.join(data_folder, prefix)
    output_file = os.path.join(base_dir, "bill.csv")
    decompressed_file_paths = []
    base_dir = os.path.join(data_folder, prefix)

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

    log_entries = []
    log_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z)\s+(.*)")

    for file_path in decompressed_file_paths:
       with open(file_path, "r") as file:
        for line in file:
            match = log_pattern.match(line)
            if match:
                timestamp, message = match.groups()
                log_type = "LOG"
                if "START RequestId" in message:
                    log_type = "START"
                elif "END RequestId" in message:
                    log_type = "END"
                elif "REPORT RequestId" in message:
                    log_type = "REPORT"
                    name = extract_lambda_name(file_path)
                    if 'CacheNode' in name:
                        log_type='GET/Putr Request'
                    else:
                        log_type='Prefetch Request'
                    report_details = parse_report_line(message)
                    costs = compute_lambda_cost(report_details)
                    report_details.update(costs)
                    log_entries.append({
                        "name": name,
                        "timestamp": timestamp,
                        "log_type": log_type,
                        **report_details,
                    })
       
        # Write parsed data to CSV
        with open(output_file, 'w', newline='') as csv_file:
            fieldnames = log_entries[0].keys()
            # fieldnames = ['System', 'Timestamp', 'RequestId', 'Duration', 'Billed Duration', 'Memory Size', 'Max Memory Used', 'Init Duration']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(log_entries)

if __name__ == '__main__':

    export_prefix = 'disdlin'
    destination_folder = 'data'
    parse_cloudwatch_logs(destination_folder,export_prefix, False)
