"""Full RLAIF training pipeline — runs train/eval loop until time limit.

Combines RL training, LLM judge evaluation, and reward model updates
in a single long-running process. Designed for multi-day runs on KubeRay.

Environment variables:
  RUN_HOURS        — Total runtime in hours (default 48)
  EPOCH_STEPS      — Timesteps per training epoch (default 300,000)
  RAY_WORKERS      — Number of Ray workers (default 3)
  ENVS_PER_WORKER  — Envs per worker (default 1)
  EPISODE_LENGTH   — Steps per episode (default 30,000)
  BATCH_SIZE       — PPO train batch size (default 4096)
  EVAL_INTERVAL    — Evaluate every N epochs (default 1)
  CHECKPOINT_DIR   — Local checkpoint directory (default /tmp/ray-checkpoints)
"""

from __future__ import annotations

import glob as globmod
import json
import logging
import os
import time
from datetime import datetime, timedelta

import ray
from ray import tune
from ray.tune.registry import register_env as ray_register

from agent.evaluator.ingest import EvaluatorIngest
from agent.evaluator.reward_model import RewardModel, build_preferences
from agent.planner.llm_client import LLMClient
from agent.rl.trainer import create_ppo_config
from agent.utils.config import S3Config
from agent.utils.s3 import S3Client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("pipeline")


def make_wrapped_env(config: dict):
    """Create ZeldaEnv wrapped with RewardWrapper."""
    from agent.env.zelda_env import ZeldaEnv
    from agent.env.reward_wrapper import RewardWrapper

    env_config = {
        k: v
        for k, v in config.items()
        if k in ("rom_path", "headless", "frame_skip", "max_steps",
                  "save_state_path", "render_mode", "seed")
    }
    base_env = ZeldaEnv(**env_config)
    wrapped = RewardWrapper(
        base_env,
        reward_config=config.get("reward_config"),
        enable_rnd=config.get("enable_rnd", True),
        enable_shaping=config.get("enable_shaping", False),
        reward_model_path=config.get("reward_model_path"),
        enable_export=config.get("enable_export", True),
        s3_config=config.get("s3_config"),
        epoch=config.get("epoch", 0),
    )
    return wrapped


def download_reward_model(s3: S3Client, epoch: int) -> str | None:
    """Download latest reward model from MinIO. Returns local path or None."""
    for e in range(epoch - 1, -1, -1):
        key = f"reward_model/epoch_{e}/rm.pt"
        try:
            data = s3.download_bytes("zelda-models", key)
            local_path = f"/tmp/reward_model_epoch{e}.pt"
            with open(local_path, "wb") as f:
                f.write(data)
            logger.info("Downloaded reward model from epoch %d (%d bytes)", e, len(data))
            return local_path
        except Exception:
            continue
    return None


def run_training_epoch(
    epoch: int,
    epoch_steps: int,
    n_workers: int,
    n_envs: int,
    ep_length: int,
    batch_size: int,
    rm_path: str | None,
    checkpoint_dir: str,
    s3: S3Client,
) -> dict:
    """Run one epoch of PPO training. Returns metadata dict."""
    logger.info("=" * 60)
    logger.info("TRAINING EPOCH %d", epoch)
    logger.info("=" * 60)

    env_config = {
        "rom_path": os.getenv("ROM_PATH", "roms/zelda.gbc"),
        "headless": True,
        "frame_skip": 4,
        "max_steps": ep_length,
        "render_mode": "rgb_array",
        "save_state_path": os.getenv("SAVE_STATE_PATH", ""),
        "epoch": epoch,
        "enable_rnd": True,
        "enable_shaping": bool(rm_path),
        "reward_model_path": rm_path,
        "enable_export": True,
    }

    config = create_ppo_config(
        env_config=env_config,
        num_workers=n_workers,
        envs_per_worker=n_envs,
        batch_size=batch_size,
    )

    logger.info("Workers: %d, Envs/worker: %d", n_workers, n_envs)
    logger.info("Epoch timesteps: %s, Episode length: %s", f"{epoch_steps:,}", f"{ep_length:,}")
    if rm_path:
        logger.info("Reward model: %s", rm_path)

    t0 = time.time()
    results = tune.run(
        "PPO",
        name=f"PPO_Zelda_epoch{epoch}",
        stop={"timesteps_total": epoch_steps},
        checkpoint_freq=5,
        checkpoint_at_end=True,
        storage_path=checkpoint_dir,
        config=config.to_dict(),
    )
    train_time = time.time() - t0

    # Extract results
    reward_metric = "env_runners/episode_return_mean"
    try:
        best_trial = results.get_best_trial(reward_metric, "max", "last")
    except Exception:
        best_trial = results.get_best_trial("training_iteration", "max", "last")

    try:
        best_checkpoint = results.get_best_checkpoint(best_trial, reward_metric, "max")
    except Exception:
        best_checkpoint = results.get_best_checkpoint(best_trial, "training_iteration", "max")

    last = best_trial.last_result if best_trial else {}
    metadata = {
        "epoch": epoch,
        "checkpoint": str(best_checkpoint) if best_checkpoint else "",
        "reward_mean": last.get("env_runners", {}).get("episode_return_mean", 0),
        "reward_max": last.get("env_runners", {}).get("episode_return_max", 0),
        "timesteps": last.get("num_env_steps_sampled_lifetime", 0),
        "episodes": last.get("env_runners", {}).get("num_episodes_lifetime", 0),
        "training_seconds": round(train_time, 1),
        "iterations": last.get("training_iteration", 0),
    }

    logger.info("Epoch %d complete in %.0fs", epoch, train_time)
    logger.info("  Mean reward: %.2f", metadata["reward_mean"])
    logger.info("  Max reward: %.2f", metadata["reward_max"])
    logger.info("  Timesteps: %s, Iterations: %d",
                f"{metadata['timesteps']:,}", metadata["iterations"])

    # Upload checkpoint to MinIO
    try:
        s3.ensure_bucket("zelda-models")
        s3.upload_bytes(
            "zelda-models",
            f"checkpoints/epoch_{epoch}/metadata.json",
            json.dumps(metadata, indent=2).encode(),
        )
        if best_checkpoint:
            checkpoint_path = getattr(best_checkpoint, "path", None)
            if checkpoint_path is None:
                cp_str = str(best_checkpoint)
                if "path=" in cp_str:
                    checkpoint_path = cp_str.split("path=")[1].rstrip(")")
            if checkpoint_path and os.path.isdir(checkpoint_path):
                for fpath in globmod.glob(os.path.join(checkpoint_path, "**"), recursive=True):
                    if os.path.isfile(fpath):
                        rel = os.path.relpath(fpath, checkpoint_path)
                        s3.upload_file("zelda-models", f"checkpoints/epoch_{epoch}/{rel}", fpath)
                logger.info("Checkpoint uploaded to MinIO: checkpoints/epoch_%d/", epoch)
    except Exception as e:
        logger.warning("Checkpoint upload failed: %s", e)

    return metadata


def run_evaluation(epoch: int, s3: S3Client) -> list[dict]:
    """Run LLM judge evaluation on all available segments. Returns scored results."""
    logger.info("=" * 60)
    logger.info("EVALUATION EPOCH %d", epoch)
    logger.info("=" * 60)

    t0 = time.time()
    llm = LLMClient()
    evaluator = EvaluatorIngest(
        llm_client=llm,
        s3_client=s3,
        episodes_bucket="zelda-episodes",
        consistency_m=3,
    )

    # List all segments
    all_keys = s3.list_keys("zelda-episodes", prefix="")
    segment_keys = []
    for key in all_keys:
        if key.endswith("/manifest.json") and not key.startswith("scores/"):
            segment_prefix = key.rsplit("/manifest.json", 1)[0]
            segment_keys.append(segment_prefix)

    if not segment_keys:
        logger.warning("No segments found for evaluation")
        llm.close()
        return []

    logger.info("Found %d segments to evaluate", len(segment_keys))

    results = evaluator.batch_evaluate(segment_keys)
    logger.info("Evaluated %d segments", len(results))

    # Write scores
    scores_key = f"scores/epoch_{epoch}/scores.jsonl"
    evaluator.write_scores(results, output_key=scores_key)

    # Train reward model
    if len(results) >= 2:
        prefs = build_preferences(results)
        if prefs:
            rm = RewardModel()

            # Load previous model for continual learning
            if epoch > 0:
                try:
                    rm_data = s3.download_bytes(
                        "zelda-models",
                        f"reward_model/epoch_{epoch - 1}/rm.pt",
                    )
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                        f.write(rm_data)
                        rm.load(f.name)
                    logger.info("Loaded reward model from epoch %d", epoch - 1)
                except Exception as e:
                    logger.info("No previous reward model: %s", e)

            avg_loss = rm.train_on_preferences(prefs)
            logger.info("Reward model loss: %.4f (%d pairs)", avg_loss, len(prefs))

            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                rm.save(f.name)
                with open(f.name, "rb") as rf:
                    rm_bytes = rf.read()

            s3.upload_bytes("zelda-models", f"reward_model/epoch_{epoch}/rm.pt", rm_bytes)
            logger.info("Saved reward model: reward_model/epoch_%d/rm.pt", epoch)

    # Write summary
    summary = {
        "epoch": epoch,
        "segments_evaluated": len(results),
        "mean_weighted_score": (
            sum(r.get("weighted_score", 0) for r in results) / len(results) if results else 0
        ),
        "score_distribution": {
            k: sum(r.get("scores", {}).get(k, 0) for r in results) / max(len(results), 1)
            for k in ["progress", "dialog", "puzzle", "novelty", "efficiency"]
        },
        "eval_seconds": round(time.time() - t0, 1),
    }
    s3.upload_bytes(
        "zelda-models",
        f"evaluations/epoch_{epoch}/summary.json",
        json.dumps(summary, indent=2).encode(),
    )

    logger.info("Evaluation complete in %.0fs", time.time() - t0)
    logger.info("  Mean score: %.3f", summary["mean_weighted_score"])
    for k, v in summary["score_distribution"].items():
        logger.info("  %s: %.3f", k, v)

    llm.close()
    return results


def main():
    run_hours = float(os.getenv("RUN_HOURS", "48"))
    epoch_steps = int(os.getenv("EPOCH_STEPS", "300000"))
    n_workers = int(os.getenv("RAY_WORKERS", "3"))
    n_envs = int(os.getenv("ENVS_PER_WORKER", "1"))
    ep_length = int(os.getenv("EPISODE_LENGTH", "30000"))
    batch_size = int(os.getenv("BATCH_SIZE", "4096"))
    eval_interval = int(os.getenv("EVAL_INTERVAL", "1"))
    checkpoint_dir = os.getenv("CHECKPOINT_DIR", "/tmp/ray-checkpoints")

    deadline = datetime.now() + timedelta(hours=run_hours)

    logger.info("=" * 60)
    logger.info("ZELDA RLAIF PIPELINE")
    logger.info("=" * 60)
    logger.info("Run hours: %.1f (deadline: %s)", run_hours, deadline.isoformat())
    logger.info("Epoch steps: %s, Episode length: %s", f"{epoch_steps:,}", f"{ep_length:,}")
    logger.info("Workers: %d, Envs/worker: %d, Batch size: %d", n_workers, n_envs, batch_size)
    logger.info("Eval every %d epoch(s)", eval_interval)

    # Init
    s3 = S3Client(S3Config())
    register_env = ray_register
    register_env("zelda_env", make_wrapped_env)
    ray.init()

    # Track progress across epochs
    epoch = 0
    total_timesteps = 0
    reward_history = []

    while datetime.now() < deadline:
        remaining = (deadline - datetime.now()).total_seconds()
        logger.info("--- Time remaining: %.1f hours ---", remaining / 3600)

        if remaining < 300:  # less than 5 minutes left
            logger.info("Less than 5 minutes remaining. Stopping.")
            break

        # Download latest reward model
        rm_path = download_reward_model(s3, epoch) if epoch > 0 else None

        # Training phase
        try:
            metadata = run_training_epoch(
                epoch=epoch,
                epoch_steps=epoch_steps,
                n_workers=n_workers,
                n_envs=n_envs,
                ep_length=ep_length,
                batch_size=batch_size,
                rm_path=rm_path,
                checkpoint_dir=checkpoint_dir,
                s3=s3,
            )
            total_timesteps += metadata.get("timesteps", 0)
            reward_history.append({
                "epoch": epoch,
                "reward_mean": metadata.get("reward_mean", 0),
                "reward_max": metadata.get("reward_max", 0),
                "timesteps": total_timesteps,
            })
        except Exception as e:
            logger.error("Training epoch %d failed: %s", epoch, e)
            epoch += 1
            continue

        # Evaluation phase
        if epoch % eval_interval == 0:
            remaining = (deadline - datetime.now()).total_seconds()
            if remaining < 600:  # not enough time for eval
                logger.info("Not enough time for evaluation. Skipping.")
            else:
                try:
                    run_evaluation(epoch, s3)
                except Exception as e:
                    logger.error("Evaluation epoch %d failed: %s", epoch, e)

        # Progress report
        logger.info("=" * 60)
        logger.info("PROGRESS REPORT")
        logger.info("=" * 60)
        logger.info("Completed epochs: %d", epoch + 1)
        logger.info("Total timesteps: %s", f"{total_timesteps:,}")
        if len(reward_history) >= 2:
            first = reward_history[0]["reward_mean"]
            latest = reward_history[-1]["reward_mean"]
            delta = latest - first
            logger.info("Reward improvement: %.2f -> %.2f (delta: %+.2f)", first, latest, delta)
        logger.info("Reward history (last 10):")
        for entry in reward_history[-10:]:
            logger.info("  Epoch %d: mean=%.2f, max=%.2f",
                        entry["epoch"], entry["reward_mean"], entry["reward_max"])

        # Upload progress to MinIO
        try:
            s3.upload_bytes(
                "zelda-models",
                "pipeline/progress.json",
                json.dumps({
                    "current_epoch": epoch,
                    "total_timesteps": total_timesteps,
                    "deadline": deadline.isoformat(),
                    "elapsed_hours": run_hours - (deadline - datetime.now()).total_seconds() / 3600,
                    "reward_history": reward_history,
                }, indent=2).encode(),
            )
        except Exception:
            pass

        epoch += 1

    # Final summary
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Total epochs: %d", epoch)
    logger.info("Total timesteps: %s", f"{total_timesteps:,}")
    if reward_history:
        logger.info("Final mean reward: %.2f", reward_history[-1]["reward_mean"])
        if len(reward_history) >= 2:
            logger.info("Total improvement: %+.2f",
                        reward_history[-1]["reward_mean"] - reward_history[0]["reward_mean"])

    ray.shutdown()


if __name__ == "__main__":
    main()
