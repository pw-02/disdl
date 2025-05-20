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
def main():
    #global variables
    root_log_dir = "logs"
    workload_type = "single"
    workload = "cifar10"
    dataloader = 'coordl' #tensorsocket, disdl, coordl
    batch_size = 128
    max_training_time_sec = None
    max_epochs = 3
    #job-level variables
    jobs = [
    {
        "job_id": 0,
        "gpu": 0,
        "seed": 0,
        "model_name": "resnet18",
        "learning_rate": 0.1,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "weight_decay": 0.0001,
        "sim_gpu_time": 0.0,
    }]
    # Generate global experiment ID and log directory
    training_started_datetime =  datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")

    # current_datetime = f"{datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")}"
    log_dir = os.path.join(root_log_dir, workload, workload_type, dataloader, training_started_datetime)
    os.makedirs(log_dir, exist_ok=True)  # Ensure the log directory exists
    
    python_cmd = sys.executable  # Use current python interpreter
    processes = []
    for job in jobs:
        # Create per-job log directory
        job_log_dir = os.path.join(log_dir, f"job_{job['job_id']}_{job['model_name']}")
        # Build command line arguments for the training script
        cmd = [
            python_cmd,
            "workloads/run.py",  # your training script filename here
            f"workload.job_id={job['job_id']}",
            f"workload.gpu_id={job['gpu']}",
            f"workload.seed={job['seed']}",
            f"workload.model_name={job['model_name']}",
            f"workload.learning_rate={job['learning_rate']}",
            f"workload.batch_size={job['batch_size']}",
            f"workload.max_epochs={job['max_epochs']}",
            f"workload.weight_decay={job['weight_decay']}",
            f"workload.sim_gpu_time={job['sim_gpu_time']}",
            f"log_dir={log_dir}"
        ]

        # Set environment variable to control visible GPU for this process
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(job['gpu'])
        print(f"Launching job {job['job_id']} on GPU {job['gpu']} with command:\n{' '.join(cmd)}")
        # Launch process
        proc = subprocess.Popen(cmd, env=env)
        processes.append(proc)

    # Wait for all jobs to finish
    for proc in processes:
        proc.wait()
    
    training_ended_datetime =  datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Training started UTC Time: {training_started_datetime}")
    print(f"Training ended UTC Time: {training_ended_datetime}")
    print("All training jobs completed.")

if __name__ == "__main__":
    main()