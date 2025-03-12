import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from omegaconf import DictConfig
import hydra
from lightning.fabric.loggers import CSVLogger

# Detect the correct Python version
def get_python_command():
    try:
        subprocess.run(["python", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return "python"
    except subprocess.CalledProcessError:
        try:
            subprocess.run(["python3", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return "python3"
        except subprocess.CalledProcessError:
            print("Error: Python is not installed.", file=sys.stderr)
            sys.exit(1)

#get config
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    workload = "coco_nas"
    dataloader = "tensorsocket" # or "tensorsocket", "disdl"
    vision_encoder_hiddern_layer_sizes = [4, 8, 16, 32]
    # vision_encoder_hiddern_layer_sizes = [4]
    producer_only = False

    # Generate experiment ID and log directory
    current_datetime = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    expid = f"{current_datetime}"
    root_log_dir = "logs"
    # log_dir = os.path.join(root_log_dir, wokload_name, dataset, dataloader, expid)
    log_dir = os.path.join(root_log_dir, workload, dataloader, expid)

    os.makedirs(log_dir, exist_ok=True)  # Ensure the log directory exists

    # Start resource monitoring
    print("Starting Resource Monitor...")
    python_cmd = get_python_command()
    monitor_cmd = f"{python_cmd} workloads/resource_monitor.py start --interval 1 --flush_interval 10 --file_path {log_dir}/resource_usage_metrics.json"
    with open(os.path.join(log_dir, "resource_usage.log"), "w") as log_file:
        monitor_process = subprocess.Popen(monitor_cmd, shell=True, stdout=log_file, stderr=log_file)
    monitor_pid = monitor_process.pid

    if dataloader == 'tensorsocket':
        # print("Starting TensorSocket producer...")
        producer_cmd = f"{python_cmd} workloads/finetune_multi_modal.py workload={workload} dataloader={dataloader} dataloader.mode=producer"
        producer_process = subprocess.Popen(producer_cmd, shell=True)
        producer_pid = producer_process.pid
        time.sleep(5)  # Adjust as necessary

        if producer_only:
            job_pids = []
            job_pids.append(producer_process)
            for process in job_pids:
                process.wait()


    # Track training start time
    training_started_datetime =  datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Training started UTC Time: {training_started_datetime}")

    # Loop over jobs
    job_pids = []
    for idx, num_hidden_layers in enumerate(vision_encoder_hiddern_layer_sizes):
        print(f"Starting job on GPU {idx} with albef model with {num_hidden_layers} hidden layers and exp_id {expid}_{idx}")
        if dataloader == 'disdl':
            run_cmd = f"CUDA_VISIBLE_DEVICES={idx} {python_cmd} workloads/finetune_multi_modal.py workload={workload} exp_id={expid} job_id={idx} dataloader={dataloader} log_dir={log_dir} workload.vision_encoder_args.num_hidden_layers={num_hidden_layers}"
        elif dataloader == 'tensorsocket' and not producer_only:
            run_cmd = f"CUDA_VISIBLE_DEVICES={idx} {python_cmd} workloads/finetune_multi_modal.py workload={workload} exp_id={expid} job_id={idx} dataloader={dataloader} log_dir={log_dir} workload.vision_encoder_args.num_hidden_layers={num_hidden_layers} dataloader.mode=consumer"

        
        #run_cmd = f"{python_cmd} workloads/image_classification.py workload={workload} exp_id={expid} job_id={idx} dataloader={dataloader} log_dir={log_dir} workload.model_architecture={model}"
        process = subprocess.Popen(run_cmd, shell=True)
        job_pids.append(process)
        time.sleep(2)  # Adjust as necessary
    
    # Wait for all jobs to complete
    for process in job_pids:
        process.wait()  

        # Track training end time
    training_ended_datetime =  datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Training started UTC Time: {training_started_datetime}")
    print(f"Training ended UTC Time: {training_ended_datetime}")
    
    # Stop resource monitor
    print("Stopping Resource Monitor...")
    monitor_process.kill()

    if dataloader == 'tensorsocket':
        producer_process.kill()


    print("Experiment completed.")

if __name__ == "__main__":
    main()