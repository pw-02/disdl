import glob
import pandas as pd
import os
from collections import OrderedDict
import csv
from pathlib import Path
import itertools
import datetime
import numpy as np

def convert_csv_to_dict(csv_file, start_timestamp = None, end_timestamp = None):
    df = pd.read_csv(csv_file)
    if 'bill.csv' in csv_file:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        # Filter the DataFrame based on the timestamp range
        # filtered_df = df[(df['Timestamp'] >= start_timestamp) & (df['Timestamp'] <= end_timestamp)]
        return df.to_dict(orient='list')
    # Filter the rows where 'Epoch Index' is equal to 1

    # if df[df['Epoch Index'] > 1].empty:
    #         filtered_df = df[df['Epoch Index'] == 1]
    # else:
    #         filtered_df = df[df['Epoch Index'] > 1]

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


def get_subfolder_names(folder_path, include_children = False):
    subfolders = glob.glob(os.path.join(folder_path, '*'))
    basenames = []
    for subfolder in subfolders:
        if include_children:
            subfolder_names = glob.glob(os.path.join(subfolder, '*'))
            for subfolder_name in subfolder_names:
                if os.path.isdir(subfolder_name):
                    basenames.append(os.path.basename(os.path.normpath(subfolder_name)))
        else:
            if os.path.isdir(subfolder):
                basenames.append(os.path.basename(os.path.normpath(subfolder)))
    return basenames

def get_throughput_over_epoch_timepoints(metrics_csv, total_epochs):
    throughput_over_time_points =[]
    for i in range(1, total_epochs + 1):
        epoch_metrics = {}
        epoch_metrics['epoch'] = i
        df = pd.read_csv(metrics_csv)
        filtered_df = df[df['Epoch Index'] == i]
        epoch_dict = filtered_df.to_dict(orient='list')
        if i == 1:
            epoch_metrics['total_samples'] = sum(epoch_dict["Batch Size"])
            epoch_metrics['total_time(s)'] = sum(epoch_dict["Iteration Time (s)"])
            epoch_metrics['throughput(samples/s)'] = epoch_metrics['total_samples'] / epoch_metrics['total_time(s)']
        else:
            epoch_metrics['total_samples'] = sum(epoch_dict["Batch Size"]) + throughput_over_time_points[-1]['total_samples']
            epoch_metrics['total_time(s)'] = sum(epoch_dict["Iteration Time (s)"]) + throughput_over_time_points[-1]['total_time(s)']
            epoch_metrics['throughput(samples/s)'] = epoch_metrics['total_samples'] / epoch_metrics['total_time(s)']
        throughput_over_time_points.append(epoch_metrics)
        
    return throughput_over_time_points


def get_batches_processed_over_time(csv_data):
    #get elapsed time in each row
    elapsed_times = list(itertools.accumulate(csv_data['Iteration Time (s)']))
    #remove the first element
    #get a dict of elapsed time and timestamp for each row
    # elapsed_times = csv_data['Elapsed Time (s)']
    time_stamps = csv_data['Timestamp (UTC)']

    #take these two list and create a dict of elapsed time and timestamp
    data = {}
    for idx, time in enumerate(elapsed_times):
        data[time] = time_stamps[idx]
  

    return  data


def get_optimal_batches_processed_over_time(csv_data):
    #get elapsed time in each row
    cumulative_iteration_times = list(itertools.accumulate(csv_data['GPU Processing Time (s)']))
    return cumulative_iteration_times


def compute_ec2_costs(instance_type: str, time_seconds: float):
    instance_prices = {'p3.8xlarge':  12.24, 'c5n.xlarge': 0.4,}
    hours = time_seconds / 3600
    hourly_rate = instance_prices[instance_type]
    instance_cost = hourly_rate * hours
    return instance_cost

def compute_cost_efficiency(time_seconds, batches_processed, dataloader, timestamp, time_cost_df):
    instance_prices = {'p3.8xlarge':  12.24, 'c5n.xlarge': 0.4} #per hour
    cache_cost = 0
    prefetch_cost = 0
    if dataloader == 'disdl' and time_cost_df is not None:
        proxy_cost_per_second = instance_prices['c5n.xlarge'] / 3600
        proxy_cost = proxy_cost_per_second * time_seconds
        # Convert reference timestamp to datetime
        # reference_timestamp = np.datetime64(timestamp)
        reference_timestamp = datetime.datetime.fromisoformat(timestamp)
        # Verify types again
        # Compute sum of costs for timestamps before the reference
        lambda_cost = time_cost_df.loc[time_cost_df['Timestamp'] < reference_timestamp, 'Cost'].sum()
        cache_cost = lambda_cost + proxy_cost

    compute_cost_per_second = instance_prices['p3.8xlarge'] / 3600
    compute_cost = compute_cost_per_second * time_seconds
    total_cost = compute_cost + cache_cost + prefetch_cost
    cost_efficiency = batches_processed / total_cost
    return total_cost, cost_efficiency, compute_cost, cache_cost, prefetch_cost



def get_training_summary(folder_path, dataloader, max_batches =None):
    search_pattern = os.path.join(folder_path, '**', 'metrics.csv')
    jobs_metric_list = []
    elapsed_times_and_timestamps = {}
    optimal_times = []
    lambda_cost_data = None

    if dataloader == 'disdl':
        for cost_csv in Path(folder_path).rglob('*_bill.csv'):
            lambda_cost_data = convert_csv_to_dict(str(cost_csv))

    for metrics_csv in glob.iglob(search_pattern, recursive=True):
        job_metrics = {}
        csv_data = convert_csv_to_dict(metrics_csv)
        #remove the first 20 element of each list as its warm up
        csv_data = {k: v[:] for k, v in csv_data.items()}
        dataloader_name = Path(metrics_csv).parts[-5]
        model_name = Path(metrics_csv).parts[-2]
        job_metrics['dataloader_name'] = dataloader_name
        job_metrics['model_name'] = model_name
        job_metrics['path'] = metrics_csv
        job_metrics['start_time'] = csv_data['Timestamp (UTC)'][0]
        job_metrics['end_time'] = csv_data['Timestamp (UTC)'][-1]
        job_metrics['num_batches'] = len(csv_data["Batch Index"])
        job_metrics['total_samples'] = sum(csv_data["Batch Size"])
        job_metrics['total_time(s)'] = sum(csv_data["Iteration Time (s)"])
        job_metrics['total_epochs'] = max(csv_data["Epoch Index"])
        job_metrics['wait_on_data_time(s)'] = sum(csv_data["Iteration Time (s)"]) - sum(csv_data["GPU Processing Time (s)"])
        job_metrics['gpu_processing_time(s)'] = sum(csv_data["GPU Processing Time (s)"])
        job_metrics['data_fetch_time(s)'] = sum(csv_data["Data Load Time (s)"])
        job_metrics['transformation_time(s)'] = sum(csv_data["Transformation Time (s)"])
        job_metrics['cache_hits'] = sum(csv_data["Cache_Hits (Samples)"])
        job_metrics['max_cached_batches'] = max(csv_data["Cache_Size"])
        job_metrics["throughput(samples/s)"] = job_metrics["total_samples"] / job_metrics["total_time(s)"]
        job_metrics["throughput(batch/s)"] = job_metrics["num_batches"] / job_metrics["total_time(s)"]
        job_metrics["optiaml_throughput(batches/s)"] = 1 / (sum(csv_data["GPU Processing Time (s)"])/len(csv_data["GPU Processing Time (s)"])) 
        job_metrics["optiaml_throughput(samples/s)"] =  job_metrics["optiaml_throughput(batches/s)"] * csv_data["Batch Size"][0]
        job_metrics["cache_hit(%)"] = job_metrics["cache_hits"] / job_metrics["total_samples"]
        job_metrics["compute_time(%)"] = job_metrics["gpu_processing_time(s)"] / job_metrics["total_time(s)"]
        job_metrics["waiting_on_data_time(%)"] = job_metrics["wait_on_data_time(s)"] / job_metrics["total_time(s)"]
        job_metrics["transformation_time(s)"] / (job_metrics["transformation_time(s)"] + job_metrics["data_fetch_time(s)"])
        job_metrics["data_fetch_time(s)"] / (job_metrics["transformation_time(s)"] + job_metrics["data_fetch_time(s)"])
        transform_percent = job_metrics["transformation_time(s)"] / (job_metrics["transformation_time(s)"] + job_metrics["data_fetch_time(s)"])
        data_fetch_percent = job_metrics["data_fetch_time(s)"] / (job_metrics["transformation_time(s)"] + job_metrics["data_fetch_time(s)"])
        job_metrics["transform_delay(%)"] = transform_percent *  job_metrics["waiting_on_data_time(%)"] 
        job_metrics["data_fetch_delay(%)"] = data_fetch_percent *  job_metrics["waiting_on_data_time(%)"]
        # job_metrics["throughout_over_time"] = get_throughput_over_epoch_timepoints(metrics_csv, job_metrics['total_epochs'])
        jobs_metric_list.append(job_metrics)
        elapsed_times_and_timestamps.update(get_batches_processed_over_time(csv_data))
        optimal_times.extend(get_optimal_batches_processed_over_time(csv_data))
        # epoch_throughputs = get_epoch_throughput(metrics_csv, job_metrics['total_epochs'])
        pass
    #now get the overall summary for all jobs
    start_time_stamp = None
    end_time_stamp = None
    overall_metrics = OrderedDict({
         "num_jobs": 0,
         "total_batches": 0,
         "total_samples": 0,
         "total_tokens": 0,
         "max_cached_batches": 0,
         "total_time(s)": 0,
         "wait_on_data_time(s)": 0,
         "gpu_processing_time(s)": 0,
         "data_fetch_time(s)": 0,
         "transformation_time(s)": 0,
         "cache_hits": 0,
          "optimal_throughput(batches/s)": 0,
          "optimal_throughput(samples/s)": 0,
    })

    aggegared_throughput_overtime = {}

    for csv_data in jobs_metric_list:
        overall_metrics["num_jobs"] += 1
        if not start_time_stamp or csv_data['start_time'] < start_time_stamp:
            start_time_stamp = csv_data['start_time']
        if not end_time_stamp or csv_data['end_time'] > end_time_stamp:
            end_time_stamp = csv_data['end_time']
       
        overall_metrics["total_batches"] += csv_data["num_batches"]
        overall_metrics["total_samples"] += csv_data["total_samples"]
        overall_metrics["total_time(s)"] += csv_data["total_time(s)"]
        overall_metrics["wait_on_data_time(s)"] += csv_data["wait_on_data_time(s)"]
        overall_metrics["gpu_processing_time(s)"] += csv_data["gpu_processing_time(s)"]
        overall_metrics["data_fetch_time(s)"] += csv_data["data_fetch_time(s)"]
        overall_metrics["transformation_time(s)"] += csv_data["transformation_time(s)"]
        overall_metrics["cache_hits"] += csv_data["cache_hits"]
        overall_metrics["optimal_throughput(batches/s)"] += csv_data["optiaml_throughput(batches/s)"]
        overall_metrics["optimal_throughput(samples/s)"] += csv_data["optiaml_throughput(samples/s)"]
        if csv_data["max_cached_batches"] > overall_metrics["max_cached_batches"]:
            overall_metrics["max_cached_batches"] = csv_data["max_cached_batches"]
        
    #     for epoch_throughput in csv_data["throughout_over_time"]:
    #         if epoch_throughput['epoch'] not in aggegared_throughput_overtime:
    #             aggegared_throughput_overtime[epoch_throughput['epoch']] = {'epoch_id': epoch_throughput['epoch'], 'total_samples': 0, 'total_time(s)': 0}
    #         aggegared_throughput_overtime[epoch_throughput['epoch']]['total_samples'] += epoch_throughput['total_samples']
    #         aggegared_throughput_overtime[epoch_throughput['epoch']]['total_time(s)'] += epoch_throughput['total_time(s)']
    #         aggegared_throughput_overtime[epoch_throughput['epoch']]['throughput(samples/s)'] = aggegared_throughput_overtime[epoch_throughput['epoch']]['total_samples'] / aggegared_throughput_overtime[epoch_throughput['epoch']]['total_time(s)']
    
    # overall_metrics["throughout_over_time"] = aggegared_throughput_overtime
    if overall_metrics['num_jobs'] > 0:
        for key in ['total_time(s)', "wait_on_data_time(s)", "gpu_processing_time(s)", "data_fetch_time(s)", "transformation_time(s)"]:
            overall_metrics[key] = overall_metrics[key] / overall_metrics['num_jobs']
        
        # metrics["throughput(batches/s)"] = metrics["total_batches"] / metrics["total_time(s)"]
        overall_metrics["throughput(samples/s)"] = overall_metrics["total_samples"] / overall_metrics["total_time(s)"]
        overall_metrics["throughput(batches/s)"] = overall_metrics["total_batches"] / overall_metrics["total_time(s)"]

        overall_metrics["cache_hit(%)"] = overall_metrics["cache_hits"] / overall_metrics["total_samples"]
        overall_metrics["compute_time(%)"] = overall_metrics["gpu_processing_time(s)"] / overall_metrics["total_time(s)"]
        overall_metrics["waiting_on_data_time(%)"] = overall_metrics["wait_on_data_time(s)"] / overall_metrics["total_time(s)"]

        transform_percent = overall_metrics["transformation_time(s)"] / (overall_metrics["transformation_time(s)"] + overall_metrics["data_fetch_time(s)"])
        data_fetch_percent = overall_metrics["data_fetch_time(s)"] / (overall_metrics["transformation_time(s)"] + overall_metrics["data_fetch_time(s)"])
        # metrics["transform_time(%)"] = metrics["transformation_time(s)"] / (metrics["transformation_time(s)"] + metrics["data_fetch_time(s)"])
        # metrics["data_fetch_time(%)"] = metrics["data_fetch_time(s)"] / (metrics["transformation_time(s)"] + metrics["data_fetch_time(s)"])
        overall_metrics["transform_delay(%)"] = transform_percent *  overall_metrics["waiting_on_data_time(%)"] 
        overall_metrics["data_fetch_delay(%)"] = data_fetch_percent *  overall_metrics["waiting_on_data_time(%)"] 
    aggegared_throughput_overtime_list = []
    for key, vlaue in aggegared_throughput_overtime.items():
            dict_line = {'path': folder_path, 'epoch_id': key, 'total_samples': vlaue['total_samples'], 
                         'total_time(s)': vlaue['total_time(s)'] / overall_metrics['num_jobs'], 
                         'throughput(samples/s)': vlaue['total_samples']/(vlaue['total_time(s)'] / overall_metrics['num_jobs'])}
            aggegared_throughput_overtime_list.append(dict_line)


    elapsed_times_and_timestamps = sorted(elapsed_times_and_timestamps.items())
    _, timestamps = zip(*elapsed_times_and_timestamps)

    trimmed =1
    batches_over_time = []

    if lambda_cost_data is not None:
        time_cost_df = pd.DataFrame({
        'Timestamp': lambda_cost_data['Timestamp'],
        'Cost': lambda_cost_data['Total Cost']})
    else:
        time_cost_df = None
    for idx, (elpasedtime, timestamp) in enumerate(elapsed_times_and_timestamps):
        total_cost, cost_efficiency, compute_cost, cache_cost, prefetch_cost = compute_cost_efficiency(elpasedtime, 
                                                                                                 idx + trimmed, 
                                                                                                 dataloader, 
                                                                                                 timestamp, 
                                                                                                 time_cost_df)        
        batches_over_time.append({'index': idx + trimmed, 
                                  'elapsed_time': elpasedtime, 
                                  'throughput': (idx + trimmed)/elpasedtime,
                                  'cost': total_cost,
                                  'compute_cost': compute_cost,
                                  'cache_cost': cache_cost,
                                  'cost_efficiency': cost_efficiency})
    

    optimal_times = sorted(optimal_times)
    return overall_metrics,jobs_metric_list,batches_over_time, lambda_cost_data

def get_avergae_batch_size_gb(workload):
    if 'cifar10' in workload:
        return 0.039 #GB
    elif 'imagenet' in workload:
        return 0.039 #GB
    


def calculaute_costs(
        dataloader_name, 
        elapsed_time, 
        exp_folder_path,
        start_timestamp = None,
        end_timestamp = None):
    
    ec2_cost = compute_ec2_costs('p3.8xlarge', elapsed_time['elapsed_time'])
    cache_cost = 0
    prefetch_cost = 0

    if dataloader_name == 'disdl':
        search_pattern = os.path.join(exp_folder_path, '**', 'bill.csv')
        for cost_csv in glob.iglob(search_pattern, recursive=True):
            #comute data loading costs
            csv_data = convert_csv_to_dict(cost_csv, start_timestamp = start_timestamp, end_timestamp = end_timestamp)
            systems = list(csv_data["System"])
            for idx, system in enumerate(systems):
                if 'InfiniSore' in system:
                    cache_cost += csv_data["Total Cost"][idx]
                elif 'PREFETCH' in system:
                    prefetch_cost += csv_data["Total Cost"][idx]
    total_cost = ec2_cost + cache_cost + prefetch_cost
    return total_cost, ec2_cost, cache_cost, prefetch_cost


def get_dataset_size(dataset):
    if 'imagenet' in dataset:
        return 1.3e6
    else:
        return 1.3e6
 
def gen_job_level_repot(jobs_metric_list, exp_folder_path):

    final_data = []
    for job_metrics in jobs_metric_list:
        model_name = job_metrics['model_name']
        dataloader_name = job_metrics['dataloader_name']
        samples_per_sec = job_metrics['throughput(samples/s)']
        sample_size_per_epoch = get_dataset_size(job_metrics['path'])
        epoch_time_seconds = (sample_size_per_epoch / samples_per_sec)
        compute_cost = compute_ec2_costs('p3.8xlarge', epoch_time_seconds)
        compute_cost_efficiency = job_metrics['num_batches'] / compute_cost
        report_line = {
            'model_name': model_name,
            'dataloader_name': dataloader_name,
            'samples_per_sec': samples_per_sec,
            'sample_size_per_epoch': sample_size_per_epoch,
            'epoch_time_seconds': epoch_time_seconds,
            'compute_cost': compute_cost,

        }
        final_data.append(report_line)
    

       
    report_name = "job_level_report.csv"
    report_path = os.path.join(exp_folder_path, report_name)
    save_dict_list_to_csv(final_data, report_path)





def compute_serverless_redis_costs(total_durtion_seconds, cache_size_gb, throughput_per_s, avg_size_per_request_kb):
    # Duration is in seconds
    # Memory size is in GB
    # Cost is in USD
    hours_in_a_month = 730
    seconds_in_a_month = 2628000
    # round_duartion_tonearest_hour = total_durtion_seconds / 3600
    # rounded_duarion = round_duartion_tonearest_hour * 3600
    data_storage_cost_monthly = cache_size_gb * hours_in_a_month * 0.125

    requests = throughput_per_s * seconds_in_a_month * avg_size_per_request_kb
    ecpu_monthly_cost = requests * 0.0000000034

    total_monhtly_cost = data_storage_cost_monthly + ecpu_monthly_cost

    exp_cost = total_monhtly_cost/seconds_in_a_month * total_durtion_seconds
    return exp_cost


if __name__ == "__main__":
 
    paths = [
        Path(r"C:\Users\pw\Desktop\disdl(600)\imagenet_nas"),
        # Path(r"C:\Users\pw\Desktop\disdl(600)\coco_nas"),
        # Path(r"C:\Users\pw\Desktop\disdl(600)\openimages_nas"),
        # Path(r"C:\Users\pw\Desktop\disdl(batchsizes)")
        ]
    
    for folder_path in paths:
        experiment_folders = [str(folder) for folder in folder_path.rglob("2025*") if folder.is_dir()]
        workload_kind = os.path.basename(os.path.normpath(folder_path))
        overall_summary = []
        job_summary = []
        throuhgput_over_time_summary = []
        for exp_folder in experiment_folders:
            exp_name = os.path.basename(os.path.normpath(exp_folder))
            dataloader = os.path.basename(os.path.dirname(exp_folder))
            dataset = os.path.basename(os.path.dirname(os.path.dirname(exp_folder)))

            exp_summary  = {}
            exp_summary['name'] = exp_name
            exp_summary['dataloader'] = dataloader
            exp_summary['dataset'] = dataset
            exp_summary['path'] = exp_folder
            
            if 'hpo' in str(folder_path):
                #get the first fodler name under  exp_summary['path'] and use it as the model name
                model_name = get_subfolder_names(exp_folder, include_children=False)[0]
                exp_summary['model_name'] = model_name

            overall_metrics,jobs_metric_list, batches_over_time, lambda_cost_data = get_training_summary(exp_folder, dataloader)
            exp_summary['lamda_requests'] = 0 if lambda_cost_data is None else len(lambda_cost_data['Total Cost'])
            exp_summary['ec2_cost'] =  compute_ec2_costs('p3.8xlarge', overall_metrics['total_time(s)'])
            exp_summary['lambda_cost'] = 0 if lambda_cost_data is None else sum(lambda_cost_data['Total Cost'])
            exp_summary['total_cost'] =  exp_summary['lambda_cost']+ exp_summary['ec2_cost']
            exp_summary['cost_efficiency'] = overall_metrics['total_batches'] / exp_summary['total_cost']
            job_summary.extend(jobs_metric_list)
            save_dict_list_to_csv(jobs_metric_list, os.path.join(exp_folder, f'{exp_name}_{dataset}_{dataloader}_summary.csv'))
            save_dict_list_to_csv(batches_over_time, os.path.join(exp_folder, f'{exp_name}_{dataset}_{dataloader}_over_time.csv'))

            exp_summary.update(overall_metrics)
            # save_dict_list_to_csv([exp_summary], os.path.join(exp_folder, f'{exp_name}_summary.csv'))
            overall_summary.append(exp_summary)
        
        gen_job_level_repot(job_summary, folder_path)

        save_dict_list_to_csv(overall_summary, os.path.join(folder_path, f'overall_summary_{workload_kind}.csv'))