"""Epoch orchestrator: train → judge → adjust → repeat.

Coordinates the full RL training loop:
  1. Submit a training RayJob for the current epoch
  2. Wait for completion
  3. Submit the evaluator job (LLM judges score episodes)
  4. Wait for completion
  5. Increment epoch and repeat

Designed to run as a long-lived pod or CronJob on the cluster.

Environment variables:
  MAX_EPOCHS       — Maximum epochs to run (default 100)
  START_EPOCH      — Starting epoch (default 0, for resumption)
  NAMESPACE        — Kubernetes namespace (default zelda-rl)
  RAY_CLUSTER      — RayCluster name (default zelda-rl)
  RAY_WORKERS      — Workers per epoch (default 10)
  EPISODE_LENGTH   — Steps per episode (default 30000)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NAMESPACE = os.getenv("NAMESPACE", "zelda-rl")
RAY_CLUSTER = os.getenv("RAY_CLUSTER", "zelda-rl")
WORKER_IMAGE = os.getenv("WORKER_IMAGE", "quay.io/cnuland/zelda-kuberay-worker:latest")
GATEWAY_URL = os.getenv(
    "LLM_GATEWAY_URL",
    "http://openshift-ai-inference-openshift-default.openshift-ingress.svc.cluster.local",
)


def create_training_job(epoch: int, prev_checkpoint: str = "") -> str:
    """Create a RayJob manifest for one training epoch and apply it."""
    job_name = f"zelda-train-epoch-{epoch}"
    n_workers = int(os.getenv("RAY_WORKERS", "10"))
    ep_length = int(os.getenv("EPISODE_LENGTH", "30000"))
    epoch_steps = n_workers * ep_length  # 1 episode per worker = 1 epoch

    rm_path = f"/tmp/reward_model/rm.pt" if epoch > 0 else ""

    manifest = f"""apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: {job_name}
  namespace: {NAMESPACE}
  labels:
    app.kubernetes.io/part-of: zelda-rl
    app.kubernetes.io/component: rl-training
    zelda-rl/epoch: "{epoch}"
    zelda-rl/phase: training
spec:
  entrypoint: python scripts/run_rollouts.py
  runtimeEnvYAML: |
    pip:
      - boto3>=1.28.0
      - pyboy>=2.6.0
      - Pillow>=9.0.0
      - PyYAML>=6.0
      - httpx>=0.25.0
      - pydantic>=2.0.0
      - torch>=2.0.0
    working_dir: "."
    excludes:
      - "old/"
      - "new/"
      - "*.ipynb"
      - "*.md"
    env_vars:
      EPOCH: "{epoch}"
      EPOCH_STEPS: "{epoch_steps}"
      RAY_WORKERS: "{n_workers}"
      ENVS_PER_WORKER: "1"
      EPISODE_LENGTH: "{ep_length}"
      REWARD_MODEL_PATH: "{rm_path}"
      LLM_GATEWAY_URL: "{GATEWAY_URL}"
      LLM_NAMESPACE: "{NAMESPACE}"
  shutdownAfterJobFinishes: false
  ttlSecondsAfterFinished: 600
  clusterSelector:
    app.kubernetes.io/part-of: zelda-rl
"""
    return _apply_manifest(manifest, job_name)


def create_evaluator_job(epoch: int) -> str:
    """Create a RayJob manifest for the evaluator/judge phase."""
    job_name = f"zelda-eval-epoch-{epoch}"

    manifest = f"""apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: {job_name}
  namespace: {NAMESPACE}
  labels:
    app.kubernetes.io/part-of: zelda-rl
    app.kubernetes.io/component: evaluator
    zelda-rl/epoch: "{epoch}"
    zelda-rl/phase: evaluation
spec:
  entrypoint: python scripts/run_evaluator.py
  runtimeEnvYAML: |
    pip:
      - boto3>=1.28.0
      - httpx>=0.25.0
      - pydantic>=2.0.0
      - torch>=2.0.0
    working_dir: "."
    excludes:
      - "old/"
      - "new/"
      - "*.ipynb"
      - "*.md"
    env_vars:
      EPOCH: "{epoch}"
      LLM_GATEWAY_URL: "{GATEWAY_URL}"
      LLM_NAMESPACE: "{NAMESPACE}"
  shutdownAfterJobFinishes: false
  ttlSecondsAfterFinished: 600
  clusterSelector:
    app.kubernetes.io/part-of: zelda-rl
"""
    return _apply_manifest(manifest, job_name)


def _apply_manifest(manifest: str, name: str) -> str:
    """Apply a K8s manifest and return the resource name."""
    result = subprocess.run(
        ["oc", "apply", "-f", "-"],
        input=manifest,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("Failed to apply %s: %s", name, result.stderr)
        raise RuntimeError(f"oc apply failed for {name}: {result.stderr}")
    logger.info("Applied: %s", name)
    return name


def wait_for_rayjob(job_name: str, timeout_s: int = 7200) -> bool:
    """Wait for a RayJob to complete. Returns True if succeeded."""
    logger.info("Waiting for RayJob %s...", job_name)
    start = time.time()
    while time.time() - start < timeout_s:
        result = subprocess.run(
            [
                "oc", "get", "rayjob", job_name,
                "-n", NAMESPACE,
                "-o", "jsonpath={.status.jobStatus}",
            ],
            capture_output=True,
            text=True,
        )
        status = result.stdout.strip()
        if status == "SUCCEEDED":
            logger.info("RayJob %s completed successfully", job_name)
            return True
        if status == "FAILED":
            logger.error("RayJob %s failed", job_name)
            return False
        time.sleep(30)

    logger.error("RayJob %s timed out after %ds", job_name, timeout_s)
    return False


def download_reward_model(epoch: int) -> str:
    """Download reward model from MinIO for use in next epoch."""
    try:
        from agent.utils.s3 import S3Client
        from agent.utils.config import S3Config

        s3 = S3Client(S3Config())
        rm_data = s3.download_bytes(
            "zelda-models",
            f"reward_model/epoch_{epoch}/rm.pt",
        )
        os.makedirs("/tmp/reward_model", exist_ok=True)
        path = "/tmp/reward_model/rm.pt"
        with open(path, "wb") as f:
            f.write(rm_data)
        logger.info("Downloaded reward model for epoch %d (%d bytes)", epoch, len(rm_data))
        return path
    except Exception as e:
        logger.warning("Could not download reward model for epoch %d: %s", epoch, e)
        return ""


def main():
    max_epochs = int(os.getenv("MAX_EPOCHS", "100"))
    start_epoch = int(os.getenv("START_EPOCH", "0"))

    logger.info("=== EPOCH ORCHESTRATOR ===")
    logger.info("Starting from epoch %d, max %d epochs", start_epoch, max_epochs)

    for epoch in range(start_epoch, max_epochs):
        logger.info("========================================")
        logger.info("  EPOCH %d / %d", epoch, max_epochs)
        logger.info("========================================")

        # Phase 1: Training
        logger.info("--- Phase 1: Training ---")
        train_job = create_training_job(epoch)
        if not wait_for_rayjob(train_job, timeout_s=7200):
            logger.error("Training failed at epoch %d. Stopping.", epoch)
            break

        # Phase 2: Judge/Evaluation
        logger.info("--- Phase 2: Judge/Evaluation ---")
        eval_job = create_evaluator_job(epoch)
        if not wait_for_rayjob(eval_job, timeout_s=3600):
            logger.warning("Evaluation failed at epoch %d. Continuing to next epoch.", epoch)

        # Phase 3: Prepare for next epoch
        logger.info("--- Phase 3: Prepare next epoch ---")
        rm_path = download_reward_model(epoch)
        if rm_path:
            logger.info("Reward model ready for epoch %d", epoch + 1)
        else:
            logger.info("No reward model available. Next epoch runs without shaping.")

        # Cleanup old RayJobs
        for old_job in [train_job, eval_job]:
            subprocess.run(
                ["oc", "delete", "rayjob", old_job, "-n", NAMESPACE, "--ignore-not-found"],
                capture_output=True,
            )

        logger.info("Epoch %d complete.", epoch)

    logger.info("=== ORCHESTRATOR FINISHED ===")


if __name__ == "__main__":
    main()
