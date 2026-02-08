"""Ray RLlib PPO training entry point — epoch-aware.

Submitted as a RayJob to the KubeRay cluster.
Runs one epoch of PPO training, saves checkpoint to MinIO,
and writes episode segment data for the judge/eval pipeline.

Environment variables:
  EPOCH          — Current epoch number (default 0)
  EPOCH_STEPS    — Timesteps per epoch (default 307,200 = 10 episodes × 30k steps)
  RAY_WORKERS    — Number of Ray workers (default 10)
  ENVS_PER_WORKER — Envs per worker (default 1, one PyBoy per worker)
  EPISODE_LENGTH — Steps per episode (default 30,000)
  REWARD_MODEL_PATH — Path to reward model .pt (empty = no shaping)
  CHECKPOINT_DIR — Local dir for Ray checkpoints (default /tmp/ray-checkpoints)
  PREV_CHECKPOINT — Path to restore from previous epoch checkpoint
"""

from __future__ import annotations

import json
import os

import ray
from ray import tune

from agent.rl.trainer import create_ppo_config


def make_wrapped_env(config: dict):
    """Create ZeldaEnv wrapped with RewardWrapper."""
    from agent.env.zelda_env import ZeldaEnv
    from agent.env.reward_wrapper import RewardWrapper

    env_config = {
        k: v
        for k, v in config.items()
        if k in ("rom_path", "headless", "frame_skip", "max_steps", "save_state_path", "render_mode", "seed")
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


def main():
    epoch = int(os.getenv("EPOCH", "0"))
    epoch_steps = int(os.getenv("EPOCH_STEPS", str(30_000 * 10)))
    n_workers = int(os.getenv("RAY_WORKERS", "10"))
    n_envs = int(os.getenv("ENVS_PER_WORKER", "1"))
    ep_length = int(os.getenv("EPISODE_LENGTH", "30000"))
    rm_path = os.getenv("REWARD_MODEL_PATH", "")
    checkpoint_dir = os.getenv("CHECKPOINT_DIR", "/tmp/ray-checkpoints")
    prev_checkpoint = os.getenv("PREV_CHECKPOINT", "")

    # Download reward model from MinIO if path starts with s3://
    if rm_path.startswith("s3://"):
        from agent.utils.s3 import S3Client
        from agent.utils.config import S3Config

        s3 = S3Client(S3Config())
        # Parse s3://bucket/key
        parts = rm_path.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        local_rm = "/tmp/reward_model.pt"
        data = s3.download_bytes(bucket, key)
        with open(local_rm, "wb") as f:
            f.write(data)
        print(f"Downloaded reward model from {rm_path} to {local_rm}")
        rm_path = local_rm

    # Register wrapped env
    from ray.tune.registry import register_env as ray_register
    ray_register("zelda_env", make_wrapped_env)

    ray.init()

    # Env config with epoch awareness
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
        "reward_model_path": rm_path or None,
        "enable_export": True,
    }

    config = create_ppo_config(
        env_config=env_config,
        num_workers=n_workers,
        envs_per_worker=n_envs,
    )

    print(f"=== EPOCH {epoch} ===")
    print(f"Workers: {n_workers}, Envs/worker: {n_envs}")
    print(f"Parallel environments: {n_workers * n_envs}")
    print(f"Epoch timesteps: {epoch_steps:,}")
    print(f"Episode length: {ep_length:,}")
    if rm_path:
        print(f"Reward model: {rm_path}")
    if prev_checkpoint:
        print(f"Restoring from: {prev_checkpoint}")

    # Run one epoch of training
    restore_path = prev_checkpoint if prev_checkpoint else None
    results = tune.run(
        "PPO",
        name=f"PPO_Zelda_epoch{epoch}",
        stop={"timesteps_total": epoch_steps},
        checkpoint_freq=1,
        checkpoint_at_end=True,
        storage_path=checkpoint_dir,
        restore=restore_path,
        config=config.to_dict(),
    )

    # Save epoch metadata for the orchestrator
    # Ray 2.9+ uses nested metric keys under env_runners/
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
    }

    metadata_path = os.path.join(checkpoint_dir, f"epoch_{epoch}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Epoch {epoch} complete. Metadata: {metadata_path}")
    print(f"  Mean reward: {metadata['reward_mean']:.2f}")
    print(f"  Max reward: {metadata['reward_max']:.2f}")
    print(f"  Total episodes: {metadata['episodes']}")

    # Upload checkpoint to MinIO for persistence
    try:
        from agent.utils.s3 import S3Client
        from agent.utils.config import S3Config

        s3 = S3Client(S3Config())
        s3.ensure_bucket("zelda-models")
        s3.upload_bytes(
            "zelda-models",
            f"checkpoints/epoch_{epoch}/metadata.json",
            json.dumps(metadata).encode(),
        )
        if best_checkpoint:
            # Ray 2.9+ Checkpoint: try .path, then parse string repr
            checkpoint_path = getattr(best_checkpoint, "path", None)
            if checkpoint_path is None:
                cp_str = str(best_checkpoint)
                if "path=" in cp_str:
                    checkpoint_path = cp_str.split("path=")[1].rstrip(")")
            if checkpoint_path and os.path.isdir(checkpoint_path):
                import glob as globmod
                for fpath in globmod.glob(os.path.join(checkpoint_path, "**"), recursive=True):
                    if os.path.isfile(fpath):
                        rel = os.path.relpath(fpath, checkpoint_path)
                        s3.upload_file(
                            "zelda-models",
                            f"checkpoints/epoch_{epoch}/{rel}",
                            fpath,
                        )
        print(f"Checkpoint uploaded to MinIO: checkpoints/epoch_{epoch}/")
    except Exception as e:
        print(f"Warning: Could not upload checkpoint to MinIO: {e}")


if __name__ == "__main__":
    main()
