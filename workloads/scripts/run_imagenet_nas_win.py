import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from omegaconf import DictConfig
import hydra
from lightning.fabric.loggers import CSVLogger

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

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    workload = "imagenet_nas"
    dataloader = "disdl"  # tensorsocket, disdl, coordl
    producer_only = False
    max_train_time_seconds = 600  # 15min
    models = ["shufflenet_v2_x1_0", "resnet18", "resnet50", "vgg16"]
    sim_gpu_times = [0.05, 0.1, 0.34, 0.5]  # Simulated GPU times

    current_datetime = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    expid = f"{current_datetime}"
    root_log_dir = "logs"
    log_dir = os.path.join(root_log_dir, workload, dataloader, expid)
    os.makedirs(log_dir, exist_ok=True)

    print("Starting Resource Monitor...")
    python_cmd = get_python_command()
    monitor_cmd = [
        python_cmd, "workloads/resource_monitor.py", "start",
        "--interval", "1",
        "--flush_interval", "10",
        "--file_path", f"{log_dir}/resource_usage_metrics.json"
    ]
    with open(os.path.join(log_dir, "resource_usage.log"), "w") as log_file:
        monitor_process = subprocess.Popen(monitor_cmd, stdout=log_file, stderr=log_file)
    monitor_pid = monitor_process.pid

    if dataloader == 'tensorsocket':
        producer_cmd = [
            python_cmd, "workloads/run.py",
            f"workload={workload}",
            f"dataloader={dataloader}",
            "dataloader.mode=producer",
            f"workload.model_architecture={models[0]}"
        ]
        producer_process = subprocess.Popen(producer_cmd)
        producer_pid = producer_process.pid
        time.sleep(5)
        if producer_only:
            producer_process.wait()
            return

    training_started_datetime = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Training started UTC Time: {training_started_datetime}")

    job_pids = []
    for idx, model in enumerate(models):
        print(f"Starting job on GPU {idx} with model {model} and exp_id {expid}_{idx}")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(idx)

        run_cmd = [
            python_cmd, "workloads/run.py",
            f"workload={workload}",
            f"exp_id={expid}",
            f"job_id={idx}",
            f"dataloader={dataloader}",
            f"log_dir={log_dir}",
            f"workload.model_architecture={model}",
            f"workload.max_training_time_sec={max_train_time_seconds}"
        ]

        if dataloader == 'disdl' or dataloader == 'coordl':
            run_cmd.append(f"workload.gpu_time={sim_gpu_times[idx]}")
        elif dataloader == 'tensorsocket' and not producer_only:
            run_cmd.append("dataloader.mode=consumer")

        process = subprocess.Popen(run_cmd, env=env)
        job_pids.append(process)

    for process in job_pids:
        process.wait()

    training_ended_datetime = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Training started UTC Time: {training_started_datetime}")
    print(f"Training ended UTC Time: {training_ended_datetime}")

    print("Stopping Resource Monitor...")
    monitor_process.kill()

    if dataloader == 'tensorsocket':
        os.kill(monitor_pid, 9)

    print("Experiment completed.")

if __name__ == "__main__":
    main()
