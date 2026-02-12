"""Submit training job to Ray cluster via Job Submission API.

Uploads local code to the cluster without rebuilding container images.
Usage:
    python scripts/submit_job.py [--stop]

Requires: pip install ray[default]
Requires: oc port-forward svc/zelda-rl-head-svc 8265:8265 -n zelda-rl
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from ray.job_submission import JobSubmissionClient, JobStatus

DASHBOARD_URL = os.getenv("RAY_DASHBOARD_URL", "http://localhost:8265")

ENV_VARS = {
    "EPOCH": "0",
    "MAX_EPOCHS": "48",
    "EPOCH_STEPS": "5000000",
    "RAY_WORKERS": "20",
    "ENVS_PER_WORKER": "5",
    "EPISODE_LENGTH": "30000",
    "BATCH_SIZE": "12288",
    "REWARD_MODEL_PATH": "",
    "RUN_EVAL": "true",
    "CLEAN_START": "false",
    "ENTROPY_START": "0.05",
    "ENTROPY_END": "0.03",
    "WALKTHROUGH_PATH": "data/zelda_oos_walkthrough.txt",
    "PYTHONPATH": ".",  # Use uploaded code, not image's baked-in code
    "PYTHONUNBUFFERED": "1",  # Force real-time log output
    "ROM_PATH": "/home/ray/roms/zelda.gbc",
    "SAVE_STATE_PATH": "/tmp/zelda.gbc.state",
}


def main():
    parser = argparse.ArgumentParser(description="Submit Ray training job")
    parser.add_argument("--stop", action="store_true", help="Stop the running job")
    parser.add_argument("--status", action="store_true", help="Check job status")
    parser.add_argument("--logs", action="store_true", help="Stream job logs")
    parser.add_argument("--tail", type=int, default=50, help="Tail N lines of logs")
    parser.add_argument("--clean", action="store_true", help="Clean MinIO before start")
    args = parser.parse_args()

    client = JobSubmissionClient(DASHBOARD_URL)

    # List running jobs
    jobs = client.list_jobs()
    running = [j for j in jobs if j.status in (JobStatus.RUNNING, JobStatus.PENDING)]

    if args.stop:
        for job in running:
            print(f"Stopping job {job.submission_id}...")
            client.stop_job(job.submission_id)
        print("Done.")
        return

    if args.status:
        if running:
            for job in running:
                print(f"Job {job.submission_id}: {job.status}")
        else:
            print("No running jobs.")
        return

    if args.logs:
        if not running:
            # Show last completed job
            completed = [j for j in jobs if j.status == JobStatus.SUCCEEDED]
            if completed:
                job = completed[-1]
            else:
                print("No jobs found.")
                return
        else:
            job = running[0]
        logs = client.get_job_logs(job.submission_id)
        lines = logs.strip().split("\n")
        for line in lines[-args.tail:]:
            print(line)
        return

    # Stop any existing jobs before submitting
    for job in running:
        print(f"Stopping existing job {job.submission_id}...")
        client.stop_job(job.submission_id)
        time.sleep(2)

    if args.clean:
        ENV_VARS["CLEAN_START"] = "true"

    # Submit new job — working_dir uploads local code to the cluster
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print(f"Submitting job from {project_root}...")
    print(f"Dashboard: {DASHBOARD_URL}")

    submission_id = client.submit_job(
        entrypoint="python scripts/run_rollouts.py",
        runtime_env={
            "working_dir": project_root,
            "env_vars": ENV_VARS,
            "excludes": [
                "*.ipynb", "__pycache__", "*.pyc", ".git",
                ".DS_Store", "*.state", "*.gbc", "*.gbc.ram",
                "new/", "old/", "ROM-MEMORY-MAP.md",
                "gitops/", "1", "=2.6.0",
            ],
        },
    )

    print(f"Job submitted: {submission_id}")
    print(f"Monitor: {DASHBOARD_URL}/#/jobs/{submission_id}")
    print()
    print("Tailing logs (Ctrl+C to stop)...")
    print()

    # Tail logs
    try:
        prev_len = 0
        while True:
            status = client.get_job_status(submission_id)
            logs = client.get_job_logs(submission_id)
            if len(logs) > prev_len:
                new_lines = logs[prev_len:]
                # Filter autoscaler spam
                for line in new_lines.split("\n"):
                    if "autoscaler" not in line and line.strip():
                        print(line)
                prev_len = len(logs)

            if status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED):
                print(f"\nJob finished: {status}")
                break
            time.sleep(10)
    except KeyboardInterrupt:
        print(f"\nJob still running: {submission_id}")
        print("Use: python scripts/submit_job.py --logs")


if __name__ == "__main__":
    main()
