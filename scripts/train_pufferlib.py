"""PufferLib PPO training entry point -- multi-epoch with RLAIF.

Replaces scripts/run_rollouts.py for PufferLib-based training.
Single-process with multiprocessing env vectorization.
Runs on 1 GPU + N CPU cores in a single Kubernetes pod.

Environment variables:
  EPOCH          -- Starting epoch number (default 0)
  MAX_EPOCHS     -- Number of epochs to run (default 48)
  EPOCH_STEPS    -- Timesteps per epoch (default 500,000)
  NUM_ENVS       -- Total parallel environments (default 24)
  NUM_WORKERS    -- Worker processes for env vectorization (default 8)
  EPISODE_LENGTH -- Steps per episode (default 30,000)
  BATCH_SIZE     -- PPO batch size (default: num_envs * 128)
  MINIBATCH_SIZE -- PPO minibatch size (default 2048)
  USE_LSTM       -- Use LSTM recurrent policy (default true)
  ENTROPY_START  -- Starting entropy coefficient (default 0.05)
  ENTROPY_END    -- Final entropy coefficient (default 0.015)
  LR_START       -- Starting learning rate (default 3e-4)
  LR_END         -- Final learning rate (default 1e-5)
  GOD_MODE_EPOCHS -- Infinite health curriculum epochs (default 6)
  RUN_EVAL       -- Run LLM eval after training (default true)
  RUN_ADVISOR    -- Run LLM reward advisor (default true)
  CLEAN_START    -- Delete old MinIO data (default true)
  ROM_PATH       -- Path to ROM file
  SAVE_STATE_PATH -- Path to game save state
  SAVE_STATE_S3_KEY -- MinIO key for save state download
  CHECKPOINT_DIR -- Local checkpoint directory
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from functools import partial

import numpy as np
import torch

import pufferlib
import pufferlib.emulation
import pufferlib.vector

# Force unbuffered stdout for real-time logs
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Re-use Ray-independent functions from run_rollouts.py
from scripts.run_rollouts import (
    clean_minio,
    cleanup_after_epoch,
    compute_entropy,
    compute_lr,
    download_reward_model,
    run_evaluation,
    run_reward_advisor,
)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(
    rom_path: str = "roms/zelda.gbc",
    headless: bool = True,
    frame_skip: int = 4,
    max_steps: int = 30_000,
    save_state_path: str = "",
    god_mode: bool = False,
    epoch: int = 0,
    reward_config: dict | None = None,
    enable_rnd: bool = True,
    enable_shaping: bool = False,
    reward_model_path: str = "",
    enable_export: bool = True,
    stagnation_limit: int = 1500,
):
    """Create a GymnasiumPufferEnv-wrapped RewardWrapper(ZeldaEnv).

    Used as the factory function for pufferlib.vector.make().
    """
    from agent.env.reward_wrapper import RewardWrapper
    from agent.env.zelda_env import ZeldaEnv

    base = ZeldaEnv(
        rom_path=rom_path,
        headless=headless,
        frame_skip=frame_skip,
        max_steps=max_steps,
        save_state_path=save_state_path if save_state_path else None,
        render_mode="rgb_array",
        god_mode=god_mode,
    )
    wrapped = RewardWrapper(
        base,
        reward_config=reward_config,
        enable_rnd=enable_rnd,
        enable_shaping=enable_shaping,
        reward_model_path=reward_model_path if reward_model_path else None,
        enable_export=enable_export,
        epoch=epoch,
        stagnation_limit=stagnation_limit,
    )
    return pufferlib.emulation.GymnasiumPufferEnv(wrapped)


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def save_checkpoint(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    mean_reward: float,
    checkpoint_dir: str,
) -> str:
    """Save model + optimizer state to local disk."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "mean_reward": mean_reward,
        },
        path,
    )
    logger.info("Saved checkpoint: %s (reward=%.1f)", path, mean_reward)
    return path


def load_checkpoint(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: str = "cuda",
) -> dict:
    """Restore model + optimizer from checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Handle compiled model state dicts (module. prefix)
    state_dict = {
        k.replace("_orig_mod.", "").replace("module.", ""): v
        for k, v in ckpt["policy_state_dict"].items()
    }
    policy.load_state_dict(state_dict, strict=False)
    if optimizer and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e:
            logger.warning("Could not restore optimizer state: %s", e)
    logger.info(
        "Loaded checkpoint: %s (epoch %d, reward %.1f)",
        path, ckpt.get("epoch", -1), ckpt.get("mean_reward", 0),
    )
    return ckpt


def upload_checkpoint_to_minio(local_path: str, epoch: int) -> None:
    """Upload checkpoint .pt file to MinIO for persistence."""
    try:
        from agent.utils.config import S3Config
        from agent.utils.s3 import S3Client

        s3 = S3Client(S3Config())
        s3.ensure_bucket("zelda-models")
        with open(local_path, "rb") as f:
            s3.upload_bytes(
                "zelda-models",
                f"pufferlib-checkpoints/epoch_{epoch}/model.pt",
                f.read(),
            )
        logger.info("Uploaded checkpoint to MinIO: epoch_%d", epoch)
    except Exception as e:
        logger.warning("MinIO upload failed: %s", e)


def download_checkpoint_from_minio(epoch: int, checkpoint_dir: str) -> str | None:
    """Download checkpoint from MinIO if available."""
    try:
        from agent.utils.config import S3Config
        from agent.utils.s3 import S3Client

        s3 = S3Client(S3Config())
        key = f"pufferlib-checkpoints/epoch_{epoch}/model.pt"
        data = s3.download_bytes("zelda-models", key)
        local_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(data)
        logger.info("Downloaded checkpoint from MinIO: %s (%d bytes)", key, len(data))
        return local_path
    except Exception:
        return None


def upload_metadata(epoch: int, metadata: dict) -> None:
    """Upload epoch metadata to MinIO."""
    try:
        from agent.utils.config import S3Config
        from agent.utils.s3 import S3Client

        s3 = S3Client(S3Config())
        s3.ensure_bucket("zelda-models")
        s3.upload_bytes(
            "zelda-models",
            f"checkpoints/epoch_{epoch}/metadata.json",
            json.dumps(metadata, indent=2).encode(),
        )
    except Exception as e:
        logger.warning("Could not upload metadata: %s", e)


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def run_training_epoch(
    epoch: int,
    num_envs: int,
    num_workers: int,
    epoch_steps: int,
    ep_length: int,
    device: torch.device,
    checkpoint_dir: str,
    lr: float,
    entropy_coeff: float,
    god_mode: bool,
    use_lstm: bool,
    reward_config: dict | None,
    rm_path: str,
    prev_checkpoint: str,
    save_state_path: str,
    minibatch_size: int = 2048,
) -> tuple[str, float, dict]:
    """Run one epoch of PufferLib PPO training.

    Returns:
        (checkpoint_path, mean_reward, metadata_dict)
    """
    from agent.rl.puffer_policy import ZeldaPolicy

    logger.info("=== EPOCH %d TRAINING ===", epoch)
    logger.info(
        "Envs: %d, Workers: %d, Steps: %s, LR: %.2e, Entropy: %.4f",
        num_envs, num_workers, f"{epoch_steps:,}", lr, entropy_coeff,
    )
    if god_mode:
        logger.info("GOD MODE: enabled")
    if rm_path:
        logger.info("Reward model shaping: %s", rm_path)

    # Create vectorized environment
    env_fn = partial(
        make_env,
        rom_path=os.getenv("ROM_PATH", "roms/zelda.gbc"),
        headless=True,
        frame_skip=4,
        max_steps=ep_length,
        save_state_path=save_state_path,
        god_mode=god_mode,
        epoch=epoch,
        reward_config=reward_config,
        enable_rnd=True,
        enable_shaping=bool(rm_path),
        reward_model_path=rm_path,
        enable_export=True,
        stagnation_limit=1500,
    )

    vecenv = pufferlib.vector.make(
        env_fn,
        backend=pufferlib.vector.Multiprocessing,
        num_envs=num_envs,
        num_workers=num_workers,
    )

    # Create policy
    policy = ZeldaPolicy(vecenv, hidden_size=256).to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

    # Load from previous checkpoint if available
    if prev_checkpoint and os.path.exists(prev_checkpoint):
        load_checkpoint(policy, optimizer, prev_checkpoint, str(device))
        # Override LR from schedule (checkpoint may have different LR)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

    # PufferLib training config
    batch_size = num_envs * 128
    batch_size_env = os.getenv("BATCH_SIZE", "")
    if batch_size_env and batch_size_env != "auto":
        batch_size = int(batch_size_env)

    config = pufferlib.namespace(
        device=str(device),
        learning_rate=lr,
        gamma=0.999,
        gae_lambda=0.95,
        update_epochs=2,
        clip_coef=0.2,
        vf_clip_coef=10.0,
        vf_coef=0.5,
        ent_coef=entropy_coeff,
        max_grad_norm=0.5,
        batch_size=batch_size,
        minibatch_size=min(minibatch_size, batch_size),
        bptt_horizon=32 if use_lstm else 128,
        anneal_lr=False,
        total_timesteps=epoch_steps,
        checkpoint_interval=0,
        seed=42 + epoch,
        compile=False,
        data_dir=checkpoint_dir,
    )

    # Create PufferLib trainer
    trainer = pufferlib.PuffeRL(
        config=config,
        vecenv=vecenv,
        policy=policy,
        optimizer=optimizer,
    )

    # Training loop
    steps_this_epoch = 0
    rewards_history = []
    episode_lengths = []
    start_time = time.time()
    log_interval = 10
    iteration = 0

    logger.info("Starting training loop (target: %s steps)...", f"{epoch_steps:,}")

    while not trainer.done:
        trainer.evaluate()
        trainer.train()
        stats = trainer.log()
        iteration += 1

        if stats:
            steps_this_epoch = stats.get("overview/agent_steps", steps_this_epoch)
            mean_reward = stats.get("environment/episode_return", 0)
            mean_length = stats.get("environment/episode_length", 0)
            sps = stats.get("overview/SPS", 0)

            if mean_reward != 0:
                rewards_history.append(mean_reward)
            if mean_length != 0:
                episode_lengths.append(mean_length)

            if iteration % log_interval == 0:
                elapsed = time.time() - start_time
                logger.info(
                    "Epoch %d | Iter %d | Steps: %s/%s | SPS: %.0f | "
                    "Reward: %.1f | EpLen: %.0f | Time: %.0fs",
                    epoch, iteration, f"{steps_this_epoch:,}",
                    f"{epoch_steps:,}", sps, mean_reward, mean_length,
                    elapsed,
                )

    # Compute final stats
    elapsed = time.time() - start_time
    avg_reward = float(np.mean(rewards_history[-20:])) if rewards_history else 0.0
    avg_length = float(np.mean(episode_lengths[-20:])) if episode_lengths else 0.0

    logger.info("Epoch %d complete: %.0fs, avg_reward=%.1f, avg_length=%.0f",
                epoch, elapsed, avg_reward, avg_length)

    # Save checkpoint
    ckpt_path = save_checkpoint(
        policy, optimizer, epoch, steps_this_epoch, avg_reward, checkpoint_dir,
    )
    upload_checkpoint_to_minio(ckpt_path, epoch)

    # Build metadata
    metadata = {
        "epoch": epoch,
        "checkpoint": ckpt_path,
        "reward_mean": avg_reward,
        "reward_max": float(max(rewards_history)) if rewards_history else 0,
        "reward_min": float(min(rewards_history)) if rewards_history else 0,
        "timesteps": steps_this_epoch,
        "elapsed_seconds": elapsed,
        "entropy_coeff": entropy_coeff,
        "learning_rate": lr,
        "god_mode": god_mode,
        "num_envs": num_envs,
        "best_mean": avg_reward,
    }

    metadata_path = os.path.join("/tmp", f"epoch_{epoch}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    upload_metadata(epoch, metadata)

    logger.info("  Mean reward: %.2f", avg_reward)
    logger.info("  Mean episode length: %.0f", avg_length)
    logger.info("  Total steps: %s", f"{steps_this_epoch:,}")

    # Cleanup
    trainer.close()

    return ckpt_path, avg_reward, metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start_epoch = int(os.getenv("EPOCH", "0"))
    max_epochs = int(os.getenv("MAX_EPOCHS", "48"))
    epoch_steps = int(os.getenv("EPOCH_STEPS", "500000"))
    num_envs = int(os.getenv("NUM_ENVS", "24"))
    num_workers = int(os.getenv("NUM_WORKERS", "8"))
    ep_length = int(os.getenv("EPISODE_LENGTH", "30000"))
    minibatch_size = int(os.getenv("MINIBATCH_SIZE", "2048"))
    use_lstm = os.getenv("USE_LSTM", "false").lower() in ("true", "1", "yes")
    run_eval = os.getenv("RUN_EVAL", "true").lower() in ("true", "1", "yes")
    clean_start = os.getenv("CLEAN_START", "true").lower() in ("true", "1", "yes")

    entropy_start = float(os.getenv("ENTROPY_START", "0.05"))
    entropy_end = float(os.getenv("ENTROPY_END", "0.015"))
    lr_start = float(os.getenv("LR_START", "3e-4"))
    lr_end = float(os.getenv("LR_END", "1e-5"))
    god_mode_epochs = int(os.getenv("GOD_MODE_EPOCHS", "6"))
    checkpoint_dir = os.getenv("CHECKPOINT_DIR", "/tmp/pufferlib-checkpoints")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # Download save state from MinIO if needed
    save_state_path = os.getenv("SAVE_STATE_PATH", "")
    if save_state_path and not os.path.exists(save_state_path):
        save_state_key = os.getenv(
            "SAVE_STATE_S3_KEY", "save-states/zelda-maku-gate.gbc.state"
        )
        logger.info("Downloading save state: %s -> %s", save_state_key, save_state_path)
        try:
            from agent.utils.config import S3Config
            from agent.utils.s3 import S3Client

            s3 = S3Client(S3Config())
            data = s3.download_bytes("zelda-models", save_state_key)
            os.makedirs(os.path.dirname(save_state_path) or ".", exist_ok=True)
            with open(save_state_path, "wb") as f:
                f.write(data)
            logger.info("Save state downloaded (%d bytes)", len(data))
        except Exception as e:
            logger.warning("Failed to download save state: %s", e)

    logger.info("=== ZELDA RL TRAINING PIPELINE (PufferLib) ===")
    logger.info("Epochs: %d-%d, Steps/epoch: %s", start_epoch,
                start_epoch + max_epochs - 1, f"{epoch_steps:,}")
    logger.info("Envs: %d, Workers: %d, LSTM: %s", num_envs, num_workers, use_lstm)
    logger.info("Entropy: %.4f -> %.4f, LR: %.2e -> %.2e",
                entropy_start, entropy_end, lr_start, lr_end)
    logger.info("God mode: first %d epochs", god_mode_epochs)

    if clean_start and start_epoch == 0:
        logger.info("Cleaning old MinIO data...")
        clean_minio()

    # Global best tracking -- prevents cascading regressions
    global_best_checkpoint = ""
    global_best_mean = -float("inf")
    reward_config = None
    rm_path = os.getenv("REWARD_MODEL_PATH", "")

    # Check for existing checkpoint from MinIO
    if start_epoch > 0:
        ckpt = download_checkpoint_from_minio(start_epoch - 1, checkpoint_dir)
        if ckpt:
            global_best_checkpoint = ckpt
            logger.info("Resuming from MinIO checkpoint: %s", ckpt)

    for epoch in range(start_epoch, start_epoch + max_epochs):
        logger.info("=" * 60)
        logger.info("  EPOCH %d / %d", epoch, start_epoch + max_epochs - 1)
        logger.info("=" * 60)

        entropy = compute_entropy(epoch, start_epoch, max_epochs,
                                  entropy_start, entropy_end)
        scheduled_lr = compute_lr(epoch, start_epoch, max_epochs,
                                  lr_start, lr_end)
        god_mode = epoch < (start_epoch + god_mode_epochs)

        # Phase 1: Training
        checkpoint, mean_reward, metadata = run_training_epoch(
            epoch=epoch,
            num_envs=num_envs,
            num_workers=num_workers,
            epoch_steps=epoch_steps,
            ep_length=ep_length,
            device=device,
            checkpoint_dir=checkpoint_dir,
            lr=scheduled_lr,
            entropy_coeff=entropy,
            god_mode=god_mode,
            use_lstm=use_lstm,
            reward_config=reward_config,
            rm_path=rm_path,
            prev_checkpoint=global_best_checkpoint,
            save_state_path=save_state_path,
            minibatch_size=minibatch_size,
        )

        # Update global best
        if mean_reward > global_best_mean:
            global_best_checkpoint = checkpoint
            global_best_mean = mean_reward
            logger.info("New global best: mean=%.1f (epoch %d)", mean_reward, epoch)
        else:
            logger.info(
                "Epoch %d mean=%.1f below global best=%.1f -- "
                "next epoch restores from global best",
                epoch, mean_reward, global_best_mean,
            )

        # Phase 2: Evaluation + reward model + reward advisor
        if run_eval:
            run_evaluation(epoch)
            rm_path = download_reward_model(epoch)

            epoch_metadata = metadata
            try:
                metadata_path = os.path.join("/tmp", f"epoch_{epoch}_metadata.json")
                with open(metadata_path) as f:
                    epoch_metadata = json.load(f)
            except Exception:
                pass

            advised_config = run_reward_advisor(epoch, epoch_metadata)
            if advised_config:
                reward_config = advised_config
                logger.info("Next epoch uses LLM-adjusted reward config")

        # Phase 3: Cleanup storage
        cleanup_after_epoch(epoch, global_best_checkpoint)

        logger.info("Epoch %d complete.", epoch)

    logger.info("=== PIPELINE FINISHED ===")


if __name__ == "__main__":
    main()
