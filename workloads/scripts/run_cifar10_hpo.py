import os
import subprocess
import sys
from datetime import datetime, timezone
import time
def main():
    # Global constants
    ROOT_LOG_DIR = "logs"
    WORKLOAD_TYPE = "hpo"
    WORKLOAD = "cifar10"
    DATALOADER = 'coordl'  # tensorsocket, disdl, coordl
    DEFAULT_BATCH_SIZE = 128
    DEFAULT_MAX_EPOCHS = 1
    DEFAULT_MODEL = "resnet18"
    DEFAULT_SEED = 42
    DEFAULT_SIM_GPU_TIME = 0.1

    # Define jobs with only differing params
    jobs = [
        {"job_id": 0, "gpu": 0, "learning_rate": 0.1},
        {"job_id": 1, "gpu": 1, "learning_rate": 0.1},
        {"job_id": 2, "gpu": 2, "learning_rate": 0.01},
        {"job_id": 3, "gpu": 3, "learning_rate": 0.005},
    ]

    # Create global log directory with timestamp
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    base_log_dir = os.path.join(ROOT_LOG_DIR, WORKLOAD, WORKLOAD_TYPE, DATALOADER, timestamp)
    os.makedirs(base_log_dir, exist_ok=True)

    python_cmd = sys.executable
    processes = []

    for job in jobs:
        # Prepare per-job parameters by merging with defaults
        job_params = {
            "job_id": job["job_id"],
            "gpu_id": job["gpu"],
            "seed": DEFAULT_SEED,
            "model_name": DEFAULT_MODEL,
            "learning_rate": job.get("learning_rate", 0.1),
            "batch_size": DEFAULT_BATCH_SIZE,
            "max_epochs": DEFAULT_MAX_EPOCHS,
            "weight_decay": 0.0001,
            "sim_gpu_time": DEFAULT_SIM_GPU_TIME
        }

        # Create per-job log directory
        job_log_dir = os.path.join(base_log_dir, f"job_{job_params['job_id']}_{job_params['model_name']}")
        os.makedirs(job_log_dir, exist_ok=True)

        # Build command line with Hydra-style overrides
        cmd = [
            python_cmd,
            "workloads/run.py",
            *(f"workload.{key}={value}" for key, value in job_params.items()),
            f"log_dir={job_log_dir}",
            f"num_jobs={len(jobs)}",
        ]

        # Set CUDA_VISIBLE_DEVICES env var to isolate GPUs
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(job_params["gpu_id"])

        print(f"Launching job {job_params['job_id']} on GPU {job_params['gpu_id']} with command:\n{' '.join(cmd)}")
        proc = subprocess.Popen(cmd, env=env)
        processes.append(proc)
        time.sleep(2)

    # Wait for all jobs to complete
    for proc in processes:
        proc.wait()

    end_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Training started at UTC: {timestamp}")
    print(f"Training ended at UTC: {end_timestamp}")
    print("All training jobs completed.")

if __name__ == "__main__":
    main()
