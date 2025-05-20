import glob
import pandas as pd
import os
import csv
from pathlib import Path


def convert_csv_to_dict(csv_file, start_timestamp = None, end_timestamp = None):
    df = pd.read_csv(csv_file)
    if 'bill.csv' in csv_file:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        # Filter the DataFrame based on the timestamp range
        # filtered_df = df[(df['Timestamp'] >= start_timestamp) & (df['Timestamp'] <= end_timestamp)]
        return df.to_dict(orient='list')

    return df.to_dict(orient='list')

def save_dict_list_to_csv(dict_list, output_file):
    if not dict_list:
        print("No data to save.")
        return
    headers = dict_list[0].keys()
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for data in dict_list:
            writer.writerow(data)

def gen_workload_level_report(exp_folder_path):
    # Get the list of all CSV files in the directory
    job_summaries = []
    workload_summary = {}
    csv_files = glob.glob(os.path.join(exp_folder_path, '**', '*metrics.csv'), recursive=True)
    for job_level_report in csv_files:
        model_name = model_folder = Path(job_level_report).parents[2].name  # Adjust index depending on folder depth
        job_metrics = convert_csv_to_dict(job_level_report)
        job_summaries.append({
            'model_name': model_name,
            # 'dataloader_name': job_metrics['dataloader_name'],
            'batch_size': job_metrics['Batch Size'][0],
            'batches_processed': len(job_metrics['Batch Index']),
            'total_time(s)': sum(job_metrics['Iteration Time (s)']),
            'total_gpu_time(s)': sum(job_metrics['GPU Processing Time (s)']),
            # 'average_gpu_time(s)': sum(job_metrics['GPU Processing Time (s)']) / len(job_metrics['GPU Processing Time (s)']),
            'total_data_delay(s)': sum(job_metrics['Wait for Data Time (s)']),
            #'total_data_delay(s)' : sum(job_metrics["Iteration Time (s)"]) - sum(job_metrics["GPU Processing Time (s)"]),
            # 'average_data_delay(s)': sum(job_metrics['Wait for Data Time (s)']) / len(job_metrics['Wait for Data Time (s)']),
            'total_data_fetch_time(s)': sum(job_metrics['Data Fetch Time (s)']),
            # 'average_data_fetch_time(s)': sum(job_metrics['Data Fetch Time (s)']) / len(job_metrics['Data Fetch Time (s)']),
            'total_transformation_time(s)': sum(job_metrics['Transform Time (s)']),
            # 'average_transformation_time(s)': sum(job_metrics['Transform Time (s)']) / len(job_metrics['Transform Time (s)']),
            'total_grpc_overhead(s)': sum(job_metrics['GRPC Report (s)']) + sum(job_metrics['GRPC Get (s)']),
            'total_cache_hits': sum(job_metrics['Cache Hit (Batch)']),
            'total_cache_misses': len(job_metrics['Cache Hit (Batch)']) - sum(job_metrics['Cache Hit (Batch)']),
            'cache_hit(%)': sum(job_metrics['Cache Hit (Batch)']) / len(job_metrics['Cache Hit (Batch)']),
            'gpu_time(%)': sum(job_metrics['GPU Processing Time (s)']) / sum(job_metrics['Iteration Time (s)']),
            'delay_time(%)': sum(job_metrics['Wait for Data Time (s)']) / sum(job_metrics['Iteration Time (s)']),
            'transformation_time(%)': (sum(job_metrics['Transform Time (s)']) / (sum(job_metrics['Transform Time (s)']) + sum(job_metrics['Data Fetch Time (s)']))) * (sum(job_metrics['Wait for Data Time (s)']) / sum(job_metrics['Iteration Time (s)'])),
            'data_fetch_time(%)':  (sum(job_metrics['Data Fetch Time (s)']) / (sum(job_metrics['Transform Time (s)']) + sum(job_metrics['Data Fetch Time (s)']))) * (sum(job_metrics['Wait for Data Time (s)']) / sum(job_metrics['Iteration Time (s)'])),
            'optimal_throughput(batches/s)': 1 / (sum(job_metrics['GPU Processing Time (s)']) / len(job_metrics['GPU Processing Time (s)'])),
            'throughput(batches/s)':  len(job_metrics['Batch Index']) / sum(job_metrics['Iteration Time (s)']),
            })
        # Append the data to the list
    workload_summary['num_jobs'] = len(job_summaries)
    workload_summary['model_names'] = [job['model_name'] for job in job_summaries]
    workload_summary['total_batches'] = sum(job['batches_processed'] for job in job_summaries)
    workload_summary['total_time(s)'] = sum(job['total_time(s)'] for job in job_summaries)
    workload_summary['total_gpu_time(s)'] = sum(job['total_gpu_time(s)'] for job in job_summaries)
    workload_summary['total_data_delay(s)'] = sum(job['total_data_delay(s)'] for job in job_summaries)
    workload_summary['total_data_fetch_time(s)'] = sum(job['total_data_fetch_time(s)'] for job in job_summaries)
    workload_summary['total_transformation_time(s)'] = sum(job['total_transformation_time(s)'] for job in job_summaries)
    workload_summary['total_grpc_overhead(s)'] = sum(job['total_grpc_overhead(s)'] for job in job_summaries)
    workload_summary['total_cache_hits'] = sum(job['total_cache_hits'] for job in job_summaries)
    workload_summary['total_cache_misses'] = sum(job['total_cache_misses'] for job in job_summaries)
    workload_summary['cache_hit(%)'] = sum(job['cache_hit(%)'] for job in job_summaries) / len(job_summaries)
    workload_summary['gpu_time(%)'] = sum(job['gpu_time(%)'] for job in job_summaries) / len(job_summaries)
    workload_summary['delay_time(%)'] = sum(job['delay_time(%)'] for job in job_summaries) / len(job_summaries)
    workload_summary['transformation_time(%)'] = sum(job['transformation_time(%)'] for job in job_summaries) / len(job_summaries)
    workload_summary['data_fetch_time(%)'] = sum(job['data_fetch_time(%)'] for job in job_summaries) / len(job_summaries)
    workload_summary['optimal_throughput(batches/s)'] = sum(job['optimal_throughput(batches/s)'] for job in job_summaries)
    workload_summary['throughput(batches/s)'] = sum(job['throughput(batches/s)'] for job in job_summaries)

    job_summary_file = os.path.join(exp_folder_path, 'job_summary.csv')
    save_dict_list_to_csv(job_summaries, job_summary_file)

    workload_summary_file = os.path.join(exp_folder_path, 'workload_summary.csv')
    save_dict_list_to_csv([workload_summary], workload_summary_file)



if __name__ == "__main__":
    example_folder = r"C:\Users\pw\projects\disdl\logs\cifar10\nas\disdl\2025-05-20_22-07-02"
    gen_workload_level_report(example_folder)