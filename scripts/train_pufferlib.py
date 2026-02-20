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
  NUM_STEPS      -- Rollout length per env per update (default 128)
  MINIBATCH_SIZE -- PPO minibatch size (default 2048)
  USE_LSTM       -- Use LSTM recurrent policy (default false)
  ENTROPY_START  -- Starting entropy coefficient (default 0.02)
  ENTROPY_END    -- Final entropy coefficient (default 0.005)
  LR_START       -- Starting learning rate (default 3e-4)
  LR_END         -- Final learning rate (default 1e-5)
  GOD_MODE_EPOCHS -- Infinite health curriculum epochs (default 6)
  STAGNATION_LIMIT -- Steps without new tile before episode truncation (default 5000)
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
import torch.nn as nn

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
    stagnation_limit: int = 5000,
    vis_ws_url: str = "",
    buf=None,
    seed=None,
):
    """Create a GymnasiumPufferEnv-wrapped RewardWrapper(ZeldaEnv).

    Used as the factory function for pufferlib.vector.make().
    PufferLib 3.0 passes ``buf`` and ``seed`` to the factory when
    creating environments inside worker processes.

    If ``vis_ws_url`` is set, wraps the environment with StreamWrapper
    to broadcast position telemetry to the map visualization server.
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

    env = wrapped

    # Optionally stream position telemetry to the map visualization server
    if vis_ws_url:
        from visualization.stream_wrapper import StreamWrapper
        env_id = seed if seed is not None else 0
        # Generate a unique color per env from its ID
        hue = (env_id * 137) % 360  # golden-angle spacing
        color = f"hsl({hue}, 70%, 55%)"
        env = StreamWrapper(
            env,
            ws_address=vis_ws_url,
            stream_metadata={"user": "pufferlib", "env_id": env_id, "color": color},
        )

    return pufferlib.emulation.GymnasiumPufferEnv(
        env, buf=buf, seed=seed if seed is not None else 0,
    )


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
# Info extraction
# ---------------------------------------------------------------------------

def _extract_env_info(infos, env_idx: int) -> dict:
    """Safely extract per-env info dict from PufferLib vectorized infos.

    PufferLib may return infos as:
      - list of dicts: infos[i] is a dict
      - dict of arrays: infos["key"][i] is a scalar
      - None / empty
    """
    if infos is None:
        return {}
    # List of dicts (most common with PufferLib Multiprocessing)
    if isinstance(infos, (list, tuple)) and len(infos) > env_idx:
        entry = infos[env_idx]
        if isinstance(entry, dict):
            return entry
        return {}
    # Dict of arrays / lists
    if isinstance(infos, dict):
        result = {}
        for key, val in infos.items():
            try:
                if hasattr(val, "__getitem__") and len(val) > env_idx:
                    result[key] = val[env_idx]
            except (TypeError, IndexError):
                pass
        return result
    return {}


# ---------------------------------------------------------------------------
# PPO Training
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
    num_steps: int = 128,
    minibatch_size: int = 2048,
    gamma: float = 0.999,
    gae_lambda: float = 0.95,
    update_epochs: int = 2,
    clip_coef: float = 0.2,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
) -> tuple[str, float, dict]:
    """Run one epoch of CleanRL-style PPO training with PufferLib envs.

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
        stagnation_limit=int(os.getenv("STAGNATION_LIMIT", "5000")),
        vis_ws_url=os.getenv("VIS_WS_URL", ""),
    )

    vecenv = pufferlib.vector.make(
        env_fn,
        backend=pufferlib.vector.Multiprocessing,
        num_envs=num_envs,
        num_workers=num_workers,
        overwork=True,  # container cgroup limits != physical cores
    )

    # Get observation size from the vecenv
    obs_size = int(np.prod(vecenv.single_observation_space.shape))

    # Create policy
    policy = ZeldaPolicy(vecenv, hidden_size=256).to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

    # Load from previous checkpoint if available
    if prev_checkpoint and os.path.exists(prev_checkpoint):
        load_checkpoint(policy, optimizer, prev_checkpoint, str(device))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

    # PPO batch dimensions
    batch_size = num_envs * num_steps
    if minibatch_size > batch_size:
        minibatch_size = batch_size
    num_iterations = max(1, epoch_steps // batch_size)

    logger.info(
        "PPO: num_steps=%d, batch_size=%d, minibatch=%d, iterations=%d, "
        "update_epochs=%d, gamma=%.3f, gae_lambda=%.2f, clip=%.2f",
        num_steps, batch_size, minibatch_size, num_iterations,
        update_epochs, gamma, gae_lambda, clip_coef,
    )

    # Rollout storage (on device)
    # MultiDiscrete: actions have shape (num_envs, 2) per step
    num_action_dims = len(policy.nvec)  # 2: [movement, button]
    obs_buf = torch.zeros((num_steps, num_envs, obs_size), dtype=torch.float32, device=device)
    actions_buf = torch.zeros((num_steps, num_envs, num_action_dims), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    rewards_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    dones_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    values_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)

    # Reset environment
    obs_np, _ = vecenv.reset(seed=42 + epoch)
    obs = torch.tensor(np.asarray(obs_np), dtype=torch.float32, device=device)
    done = torch.zeros(num_envs, dtype=torch.float32, device=device)

    # Episode tracking
    ep_rewards = np.zeros(num_envs, dtype=np.float64)
    ep_lengths = np.zeros(num_envs, dtype=np.int64)
    completed_rewards = []
    completed_lengths = []

    # Game milestone tracking — aggregated across all completed episodes.
    # "any_*" tracks whether ANY episode achieved a milestone (binary).
    # "max_*" tracks the best value seen across all episodes.
    # "total_*" counts how many episodes achieved a milestone.
    milestones = {
        "total_got_sword": 0,
        "total_entered_dungeon": 0,
        "total_visited_maku_tree": 0,
        "total_gate_slashed": 0,
        "total_maku_stage": 0,
        "total_maku_dialog": 0,
        "total_gnarled_key": 0,
        "total_maku_seed": 0,
        "max_rooms": 0,
        "max_tiles": 0,
        "max_maku_rooms": 0,
        "max_essences": 0,
        "max_dungeon_keys": 0,
        # Save-state baseline — what the agent already has at episode start.
        # Captured once from the first completed episode (same for all episodes
        # since they share the same save state).  Used by the phase detector
        # to skip phases the save state has already completed.
        "baseline_has_sword": False,
        "baseline_has_maku_dialog": False,
        "baseline_has_gnarled_key": False,
        "baseline_has_maku_seed": False,
        "baseline_maku_stage": 0,
        "baseline_captured": False,
    }

    global_step = 0
    start_time = time.time()
    log_interval = max(1, num_iterations // 20)

    logger.info("Starting training loop (target: %s steps, %d iterations)...",
                f"{epoch_steps:,}", num_iterations)

    for iteration in range(1, num_iterations + 1):
        # ---- Collect rollout ----
        for step in range(num_steps):
            with torch.no_grad():
                logits_list, value = policy(obs)
                # MultiDiscrete: sample each dimension independently
                dists = [torch.distributions.Categorical(logits=lg) for lg in logits_list]
                actions = [d.sample() for d in dists]
                logprob = sum(d.log_prob(a) for d, a in zip(dists, actions))
                action = torch.stack(actions, dim=-1)  # (num_envs, 2)

            obs_buf[step] = obs
            actions_buf[step] = action
            logprobs_buf[step] = logprob
            values_buf[step] = value
            dones_buf[step] = done

            next_obs_np, reward_np, terminal_np, truncated_np, infos = vecenv.step(
                action.cpu().numpy()
            )

            reward_np = np.asarray(reward_np, dtype=np.float32)
            terminal_np = np.asarray(terminal_np, dtype=bool)
            truncated_np = np.asarray(truncated_np, dtype=bool)
            done_np = np.logical_or(terminal_np, truncated_np)

            rewards_buf[step] = torch.tensor(reward_np, device=device)
            done = torch.tensor(done_np, dtype=torch.float32, device=device)

            # Track episode stats and game milestones
            ep_rewards += reward_np
            ep_lengths += 1
            for i in range(num_envs):
                if done_np[i]:
                    completed_rewards.append(float(ep_rewards[i]))
                    completed_lengths.append(int(ep_lengths[i]))
                    ep_rewards[i] = 0.0
                    ep_lengths[i] = 0

                    # Extract game milestones from env info.
                    # PufferLib returns infos as list-of-dicts or dict-of-arrays.
                    env_info = _extract_env_info(infos, i)
                    if env_info:
                        if env_info.get("milestone_got_sword", 0) > 0:
                            milestones["total_got_sword"] += 1
                        if env_info.get("milestone_entered_dungeon", 0) > 0:
                            milestones["total_entered_dungeon"] += 1
                        if env_info.get("milestone_visited_maku_tree", 0) > 0:
                            milestones["total_visited_maku_tree"] += 1
                        if env_info.get("milestone_maku_dialog", 0) > 0:
                            milestones["total_maku_dialog"] += 1
                        if env_info.get("milestone_gate_slashed", 0) > 0:
                            milestones["total_gate_slashed"] += 1
                        if env_info.get("milestone_maku_stage", 0) > 0:
                            milestones["total_maku_stage"] += 1
                        if env_info.get("milestone_gnarled_key", 0) > 0:
                            milestones["total_gnarled_key"] += 1
                        if env_info.get("milestone_maku_seed", 0) > 0:
                            milestones["total_maku_seed"] += 1
                        milestones["max_maku_rooms"] = max(
                            milestones["max_maku_rooms"],
                            int(env_info.get("milestone_maku_rooms", 0)),
                        )
                        milestones["max_rooms"] = max(
                            milestones["max_rooms"],
                            int(env_info.get("milestone_max_rooms", 0)),
                        )
                        milestones["max_tiles"] = max(
                            milestones["max_tiles"],
                            int(env_info.get("unique_tiles", 0)),
                        )
                        milestones["max_essences"] = max(
                            milestones["max_essences"],
                            int(env_info.get("milestone_essences", 0)),
                        )
                        milestones["max_dungeon_keys"] = max(
                            milestones["max_dungeon_keys"],
                            int(env_info.get("milestone_dungeon_keys", 0)),
                        )

                        # Capture save-state baseline once (identical for all
                        # episodes since they share the same save state).
                        if not milestones["baseline_captured"]:
                            milestones["baseline_has_sword"] = bool(
                                env_info.get("baseline_has_sword", 0)
                            )
                            milestones["baseline_has_maku_dialog"] = bool(
                                env_info.get("baseline_has_maku_dialog", 0)
                            )
                            milestones["baseline_has_gnarled_key"] = bool(
                                env_info.get("baseline_has_gnarled_key", 0)
                            )
                            milestones["baseline_has_maku_seed"] = bool(
                                env_info.get("baseline_has_maku_seed", 0)
                            )
                            milestones["baseline_maku_stage"] = int(
                                env_info.get("baseline_maku_stage", 0)
                            )
                            milestones["baseline_captured"] = True

            obs = torch.tensor(np.asarray(next_obs_np), dtype=torch.float32, device=device)
            global_step += num_envs

        # ---- Compute GAE advantages ----
        with torch.no_grad():
            _, next_value = policy(obs)
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = 0.0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = (
                    delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values_buf

        # ---- PPO update ----
        b_obs = obs_buf.reshape(-1, obs_size)
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1, num_action_dims)  # (batch, 2)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        num_updates = 0

        for _update_epoch in range(update_epochs):
            indices = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]

                new_logits_list, new_value = policy(b_obs[mb_idx])
                # MultiDiscrete: reconstruct per-dimension distributions
                new_dists = [
                    torch.distributions.Categorical(logits=lg)
                    for lg in new_logits_list
                ]
                mb_acts = b_actions[mb_idx]  # (mb, 2)
                new_logprob = sum(
                    d.log_prob(mb_acts[:, i])
                    for i, d in enumerate(new_dists)
                )
                entropy = sum(d.entropy() for d in new_dists)

                logratio = new_logprob - b_logprobs[mb_idx]
                ratio = logratio.exp()

                # Approximate KL for diagnostics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                mb_advantages = b_advantages[mb_idx]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                # Clipped policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * ((new_value - b_returns[mb_idx]) ** 2).mean()

                # Entropy bonus
                entropy_loss = entropy.mean()

                loss = pg_loss - entropy_coeff * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += entropy_loss.item()
                total_approx_kl += approx_kl.item()
                num_updates += 1

        # ---- Logging ----
        if iteration % log_interval == 0 or iteration == num_iterations:
            elapsed = time.time() - start_time
            sps = global_step / elapsed if elapsed > 0 else 0
            recent_rewards = completed_rewards[-50:] if completed_rewards else [0]
            recent_lengths = completed_lengths[-50:] if completed_lengths else [0]
            logger.info(
                "Epoch %d | Iter %d/%d | Steps: %s/%s | SPS: %.0f | "
                "Reward: %.1f | EpLen: %.0f | PgLoss: %.4f | VLoss: %.4f | "
                "Entropy: %.3f | KL: %.4f | Rooms: %d | Tiles: %d",
                epoch, iteration, num_iterations,
                f"{global_step:,}", f"{epoch_steps:,}", sps,
                np.mean(recent_rewards), np.mean(recent_lengths),
                total_pg_loss / max(num_updates, 1),
                total_v_loss / max(num_updates, 1),
                total_entropy / max(num_updates, 1),
                total_approx_kl / max(num_updates, 1),
                milestones["max_rooms"],
                milestones["max_tiles"],
            )

    # ---- Epoch summary ----
    elapsed = time.time() - start_time
    avg_reward = float(np.mean(completed_rewards[-20:])) if completed_rewards else 0.0
    avg_length = float(np.mean(completed_lengths[-20:])) if completed_lengths else 0.0

    logger.info(
        "Epoch %d complete: %.0fs, avg_reward=%.1f, avg_length=%.0f, "
        "episodes=%d, global_steps=%s",
        epoch, elapsed, avg_reward, avg_length, len(completed_rewards),
        f"{global_step:,}",
    )

    # Save checkpoint
    ckpt_path = save_checkpoint(
        policy, optimizer, epoch, global_step, avg_reward, checkpoint_dir,
    )
    upload_checkpoint_to_minio(ckpt_path, epoch)

    # Build metadata (includes game milestones for monitoring)
    metadata = {
        "epoch": epoch,
        "checkpoint": ckpt_path,
        "reward_mean": avg_reward,
        "reward_max": float(max(completed_rewards)) if completed_rewards else 0,
        "reward_min": float(min(completed_rewards)) if completed_rewards else 0,
        "timesteps": global_step,
        "elapsed_seconds": elapsed,
        "entropy_coeff": entropy_coeff,
        "learning_rate": lr,
        "god_mode": god_mode,
        "num_envs": num_envs,
        "episodes_completed": len(completed_rewards),
        "best_mean": avg_reward,
        # Game progression milestones
        "milestones": milestones,
    }

    metadata_path = os.path.join("/tmp", f"epoch_{epoch}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    upload_metadata(epoch, metadata)

    logger.info("  Mean reward: %.2f", avg_reward)
    logger.info("  Mean episode length: %.0f", avg_length)
    logger.info("  Total steps: %s", f"{global_step:,}")

    # Game progression summary
    n_eps = max(len(completed_rewards), 1)
    logger.info("  --- Game Progress ---")
    logger.info("  Max rooms explored:   %d", milestones["max_rooms"])
    logger.info("  Max tiles explored:   %d", milestones["max_tiles"])
    logger.info("  Got sword:            %d/%d episodes", milestones["total_got_sword"], n_eps)
    logger.info("  Visited Maku Tree:    %d/%d episodes", milestones["total_visited_maku_tree"], n_eps)
    logger.info("  Slashed gate:         %d/%d episodes", milestones["total_gate_slashed"], n_eps)
    logger.info("  Maku Tree rooms:      %d max", milestones["max_maku_rooms"])
    logger.info("  Maku Tree stage up:   %d/%d episodes", milestones["total_maku_stage"], n_eps)
    logger.info("  Maku Tree dialog:     %d/%d episodes", milestones["total_maku_dialog"], n_eps)
    logger.info("  Got Gnarled Key:      %d/%d episodes", milestones["total_gnarled_key"], n_eps)
    logger.info("  Got Maku Seed:        %d/%d episodes", milestones["total_maku_seed"], n_eps)
    logger.info("  Entered dungeon:      %d/%d episodes", milestones["total_entered_dungeon"], n_eps)
    logger.info("  Max essences:         %d", milestones["max_essences"])
    logger.info("  Max dungeon keys:     %d", milestones["max_dungeon_keys"])

    # Cleanup
    vecenv.close()

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
    num_steps = int(os.getenv("NUM_STEPS", "128"))
    minibatch_size = int(os.getenv("MINIBATCH_SIZE", "2048"))
    use_lstm = os.getenv("USE_LSTM", "false").lower() in ("true", "1", "yes")
    run_eval = os.getenv("RUN_EVAL", "true").lower() in ("true", "1", "yes")
    clean_start = os.getenv("CLEAN_START", "true").lower() in ("true", "1", "yes")

    entropy_start = float(os.getenv("ENTROPY_START", "0.02"))
    entropy_end = float(os.getenv("ENTROPY_END", "0.005"))
    lr_start = float(os.getenv("LR_START", "3e-4"))
    lr_end = float(os.getenv("LR_END", "1e-5"))
    god_mode_epochs = int(os.getenv("GOD_MODE_EPOCHS", "6"))
    checkpoint_dir = os.getenv("CHECKPOINT_DIR", "/tmp/pufferlib-checkpoints")

    # PPO core hyperparameters — configurable via env vars for tuning
    update_epochs = int(os.getenv("UPDATE_EPOCHS", "2"))
    clip_coef = float(os.getenv("CLIP_COEF", "0.2"))
    gamma = float(os.getenv("GAMMA", "0.999"))

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

    # Initial reward config from env vars (overridden later by reward advisor)
    reward_config: dict | None = None
    time_penalty = float(os.getenv("TIME_PENALTY", "0"))
    backtrack_penalty = float(os.getenv("BACKTRACK_PENALTY", "0"))
    distance_bonus = float(os.getenv("DISTANCE_BONUS", "0"))
    directional_bonus = float(os.getenv("DIRECTIONAL_BONUS", "0"))
    grid_exploration = float(os.getenv("GRID_EXPLORATION", "0"))
    reward_overrides = {}
    if time_penalty != 0:
        reward_overrides["time_penalty"] = time_penalty
        logger.info("Time penalty: %.4f per step", time_penalty)
    if backtrack_penalty != 0:
        reward_overrides["backtrack_penalty"] = backtrack_penalty
        logger.info("Backtrack penalty: %.2f per room re-entry", backtrack_penalty)
    if distance_bonus != 0:
        reward_overrides["distance_bonus"] = distance_bonus
        logger.info("Distance bonus: %.1f per new max Manhattan distance", distance_bonus)
    if directional_bonus != 0:
        reward_overrides["directional_bonus"] = directional_bonus
        logger.info("Directional bonus: %.1f per new eastward column", directional_bonus)
    if grid_exploration != 0:
        reward_overrides["grid_exploration"] = grid_exploration
        logger.info("Grid exploration: %.3f per tile step", grid_exploration)
    if reward_overrides:
        reward_config = reward_overrides

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
            num_steps=num_steps,
            minibatch_size=minibatch_size,
            update_epochs=update_epochs,
            clip_coef=clip_coef,
            gamma=gamma,
        )

        # Always continue from latest checkpoint to avoid reverting
        # learning progress. During early training, mean reward dips
        # as the policy transitions from random to directed behavior —
        # reverting to the "best" random checkpoint prevents specialization.
        global_best_checkpoint = checkpoint
        if mean_reward > global_best_mean:
            global_best_mean = mean_reward
            logger.info("New best mean reward: %.1f (epoch %d)", mean_reward, epoch)
        else:
            logger.info(
                "Epoch %d mean=%.1f (best=%.1f) -- continuing from latest",
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
