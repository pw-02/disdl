import os
import sys
import subprocess
from pathlib import Path
from omegaconf import OmegaConf
import hydra

@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    python_cmd = sys.executable
    exp_id = "exp_" + OmegaConf.to_container(cfg).get("exp_id", "default")

    base_log_dir = Path("logs") / exp_id
    base_log_dir.mkdir(parents=True, exist_ok=True)

    processes = []

    for job in cfg.jobs:
        # Build command line args overrides for this job
        overrides = [
            f"job_id={job.job_id}",
            f"workload.jobs=[{{job_id: {job.job_id}, gpu: {job.gpu}, model_name: '{job.model_name}', learning_rate: {job.learning_rate}, batch_size: {job.batch_size}}}]"
        ]

        log_dir = base_log_dir / f"job_{job.job_id}"
        log_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            python_cmd,
            "workloads/run.py",  # Your training script entrypoint
            f"log_dir={str(log_dir)}",
        ] + overrides

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)

        print(f"Launching job {job.job_id} on GPU {job.gpu} with command:\n{' '.join(cmd)}")
        proc = subprocess.Popen(cmd, env=env)
        processes.append(proc)

    # Wait for all jobs to complete
    for proc in processes:
        proc.wait()

    print("All training jobs finished.")

if __name__ == "__main__":
    main()
