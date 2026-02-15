"""Ray RLlib PPO training entry point — multi-epoch with RLAIF.

Runs the full training loop:
  1. PPO training for one epoch (with entropy scheduling)
  2. LLM judge evaluation of episode segments
  3. Bradley-Terry reward model training
  4. Next epoch with potential-based shaping

Improvements over previous versions:
  - Entropy scheduling: linear decay from 0.05 → 0.005 across epochs
  - Stagnation truncation: episodes end early if no new tiles for 3000 steps
  - Coordinate decay: tile bonuses decay over time (0.9995/step)
  - Gamma 0.999: longer horizon for exploration
  - MinIO cleanup: old data is purged on fresh starts
  - Per-epoch timestep counting: each epoch trains for exactly EPOCH_STEPS

Environment variables:
  EPOCH          — Starting epoch number (default 0)
  MAX_EPOCHS     — Number of epochs to run (default 48)
  EPOCH_STEPS    — Timesteps per epoch (default 500,000)
  RAY_WORKERS    — Number of Ray workers (default 10)
  ENVS_PER_WORKER — Envs per worker (default 1)
  EPISODE_LENGTH — Steps per episode (default 30,000)
  REWARD_MODEL_PATH — Path to reward model .pt (empty = no shaping)
  CHECKPOINT_DIR — Local dir for Ray checkpoints
  PREV_CHECKPOINT — Path to restore from previous epoch checkpoint
  RUN_EVAL       — Run LLM eval after training (default true)
  CLEAN_START    — Delete old MinIO data before training (default true)
  ENTROPY_START  — Starting entropy coefficient (default 0.05)
  ENTROPY_END    — Final entropy coefficient (default 0.005)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile

import ray
from ray.tune import Stopper

from agent.rl.trainer import create_ppo_config

# Force unbuffered stdout so logs appear in real-time
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class EpochStopper(Stopper):
    """Stop after a fixed number of NEW timesteps per epoch.

    Unlike a raw timesteps_total stop, this tracks the starting timestep
    when the first result arrives (which includes restored checkpoint steps)
    and stops after epoch_steps additional steps.
    """

    def __init__(self, epoch_steps: int):
        self._epoch_steps = epoch_steps
        self._start_ts: int | None = None

    def __call__(self, trial_id: str, result: dict) -> bool:
        ts = result.get("timesteps_total", 0)
        if self._start_ts is None:
            self._start_ts = ts
            logger.info(
                "EpochStopper: starting at %d, will stop at %d",
                self._start_ts,
                self._start_ts + self._epoch_steps,
            )
        return (ts - self._start_ts) >= self._epoch_steps

    def stop_all(self) -> bool:
        return False


def make_wrapped_env(config: dict):
    """Create ZeldaEnv wrapped with RewardWrapper."""
    from agent.env.zelda_env import ZeldaEnv
    from agent.env.reward_wrapper import RewardWrapper

    env_config = {
        k: v
        for k, v in config.items()
        if k
        in (
            "rom_path",
            "headless",
            "frame_skip",
            "max_steps",
            "save_state_path",
            "render_mode",
            "seed",
            "god_mode",
        )
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
        stagnation_limit=config.get("stagnation_limit", 5000),
    )
    return wrapped


def compute_entropy(epoch: int, start_epoch: int, max_epochs: int,
                    entropy_start: float, entropy_end: float) -> float:
    """Linear entropy decay across epochs."""
    if max_epochs <= 1:
        return entropy_start
    progress = (epoch - start_epoch) / (max_epochs - 1)
    return entropy_start + (entropy_end - entropy_start) * progress


def compute_lr(epoch: int, start_epoch: int, max_epochs: int,
               lr_start: float, lr_end: float) -> float:
    """Linear learning rate decay across epochs."""
    if max_epochs <= 1:
        return lr_start
    progress = (epoch - start_epoch) / (max_epochs - 1)
    return lr_start + (lr_end - lr_start) * progress


def clean_minio():
    """Delete all old episodes and models from MinIO for a fresh start."""
    try:
        from agent.utils.s3 import S3Client
        from agent.utils.config import S3Config

        s3 = S3Client(S3Config())
        for bucket in ["zelda-episodes", "zelda-models"]:
            logger.info("Resetting bucket %s...", bucket)
            s3.force_reset_bucket(bucket)
        logger.info("MinIO cleanup complete")
    except Exception as e:
        logger.warning("MinIO cleanup failed: %s", e)


def cleanup_after_epoch(epoch: int, global_best_checkpoint: str):
    """Free MinIO storage after each epoch to prevent disk-full crashes.

    1. Deletes episode recordings for this epoch (already scored by judge).
    2. Deletes old Ray checkpoint experiments, keeping only the global best.
    """
    try:
        from agent.utils.s3 import S3Client
        from agent.utils.config import S3Config

        s3 = S3Client(S3Config())

        # 1. Clean episode recordings for this epoch (already evaluated)
        ep_prefix = f"epoch_{epoch}/"
        deleted = s3.delete_all_objects("zelda-episodes", prefix=ep_prefix)
        if deleted:
            logger.info("Cleaned %d episode objects for epoch %d", deleted, epoch)

        # 2. Clean old Ray checkpoint experiments from zelda-models
        # Keep the experiment directory containing the global best checkpoint.
        # Checkpoints live under ray-checkpoints/PPO_Zelda_epoch{N}/
        all_keys = s3.list_keys("zelda-models", prefix="ray-checkpoints/")

        # Find which experiment dirs exist (e.g. ray-checkpoints/PPO_Zelda_epoch0)
        experiment_dirs = set()
        for key in all_keys:
            parts = key.split("/")
            if len(parts) >= 2:
                experiment_dirs.add(parts[0] + "/" + parts[1])

        # Determine which experiment dir holds the global best by checking
        # if the dir name appears anywhere in the checkpoint path
        keep_dirs = set()
        if global_best_checkpoint:
            for d in experiment_dirs:
                # Check both "d in checkpoint" and "checkpoint contains d"
                # The checkpoint path may or may not include bucket prefix
                dir_name = d.split("/")[-1]  # e.g. "PPO_Zelda_epoch0"
                if dir_name in global_best_checkpoint or d in global_best_checkpoint:
                    keep_dirs.add(d)
                    logger.info("Keeping checkpoint dir %s (matches global best)", d)

        if not keep_dirs and global_best_checkpoint:
            # Safety: if no match found, keep everything to avoid data loss
            logger.warning(
                "Could not match global best checkpoint to any experiment dir. "
                "Checkpoint: %s, Dirs: %s — skipping checkpoint cleanup",
                global_best_checkpoint, experiment_dirs,
            )
            return

        # Delete all experiment dirs except the ones we need to keep
        for exp_dir in experiment_dirs:
            if exp_dir in keep_dirs:
                continue
            deleted = s3.delete_all_objects("zelda-models", prefix=exp_dir + "/")
            if deleted:
                logger.info("Cleaned %d checkpoint objects from %s", deleted, exp_dir)

    except Exception as e:
        logger.warning("Post-epoch cleanup failed (non-fatal): %s", e)


def _download_s3_checkpoint(s3_path: str, s3_fs) -> str:
    """Download an S3 checkpoint to a local temp directory for restore."""
    import shutil
    from pyarrow.fs import FileSelector, FileType

    local_dir = os.path.join(tempfile.mkdtemp(prefix="ray-cp-"), "checkpoint")
    os.makedirs(local_dir, exist_ok=True)

    logger.info("Downloading checkpoint from S3: %s -> %s", s3_path, local_dir)
    selector = FileSelector(s3_path, recursive=True)
    for f in s3_fs.get_file_info(selector):
        if f.type == FileType.File:
            rel = f.path[len(s3_path):].lstrip("/")
            local_path = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with s3_fs.open_input_stream(f.path) as stream:
                with open(local_path, "wb") as lf:
                    lf.write(stream.read())

    logger.info("Checkpoint downloaded to %s", local_dir)
    return local_dir


def run_training_epoch(
    epoch, start_epoch, epoch_steps, n_workers, n_envs, ep_length,
    rm_path, prev_checkpoint, entropy_coeff, reward_config=None,
    god_mode_epochs=0, lr=None
):
    """Run one epoch of PPO training. Returns best checkpoint path."""
    # God mode curriculum: infinite health for first N epochs to isolate
    # exploration learning from survival (inspired by Pokemon Red RL).
    god_mode = epoch < (start_epoch + god_mode_epochs)

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
        "stagnation_limit": 1500,  # Truncate after 1.5K steps without new tile
        "god_mode": god_mode,
    }

    # Pass LLM-adjusted reward config to workers (None = use defaults)
    if reward_config:
        env_config["reward_config"] = reward_config
        logger.info("Using LLM-adjusted reward config with %d parameters", len(reward_config))

    config = create_ppo_config(
        env_config=env_config,
        num_workers=n_workers,
        envs_per_worker=n_envs,
        entropy_coeff=entropy_coeff,
        lr=lr,
    )

    logger.info("=== EPOCH %d TRAINING ===", epoch)
    logger.info("Workers: %d, Envs/worker: %d", n_workers, n_envs)
    logger.info("Epoch timesteps: %s (per-epoch stopper)", f"{epoch_steps:,}")
    logger.info("Episode length: %s, Entropy: %.4f, LR: %.2e", f"{ep_length:,}", entropy_coeff, lr or 3e-4)
    if god_mode:
        logger.info("GOD MODE: enabled (epoch %d < %d)", epoch, start_epoch + god_mode_epochs)
    if rm_path:
        logger.info("Reward model shaping: %s", rm_path)

    # Use MinIO (S3-compatible) for checkpoint storage so all nodes can access it
    from pyarrow.fs import S3FileSystem
    from agent.utils.config import S3Config

    s3_cfg = S3Config()
    # endpoint_override needs host:port without scheme
    endpoint = s3_cfg.endpoint_url.replace("http://", "").replace("https://", "")
    s3_fs = S3FileSystem(
        access_key=s3_cfg.access_key,
        secret_key=s3_cfg.secret_key,
        endpoint_override=endpoint,
        scheme="http",
    )

    # Download previous checkpoint from S3 to local for restore
    restore_path = None
    if prev_checkpoint:
        try:
            restore_path = _download_s3_checkpoint(prev_checkpoint, s3_fs)
        except Exception as e:
            logger.warning("Could not download checkpoint: %s — starting fresh", e)

    # --- Train with peak-reward checkpoint tracking via callback ---
    from ray import tune

    class PeakCheckpointTracker:
        """Tracks the checkpoint with the highest mean reward."""
        def __init__(self):
            self.best_mean = -float("inf")
            self.best_checkpoint_path = ""

    tracker = PeakCheckpointTracker()

    # Try with restore first; if the PPO actor lands on a different node
    # than where the checkpoint was downloaded, it will fail with a
    # ValueError. In that case, retry without restore.
    tune_kwargs = dict(
        run_or_experiment="PPO",
        name=f"PPO_Zelda_epoch{epoch}",
        stop=EpochStopper(epoch_steps),
        checkpoint_freq=10,
        checkpoint_at_end=True,
        storage_path="zelda-models/ray-checkpoints",
        storage_filesystem=s3_fs,
        config=config.to_dict(),
    )

    if restore_path:
        try:
            results = tune.run(**tune_kwargs, restore=restore_path)
        except Exception as restore_err:
            logger.warning(
                "Restore failed (cross-node issue): %s — restarting fresh",
                restore_err,
            )
            # Reset the stopper so it counts from scratch
            tune_kwargs["stop"] = EpochStopper(epoch_steps)
            results = tune.run(**tune_kwargs)
    else:
        results = tune.run(**tune_kwargs)

    # Extract the LAST checkpoint (most trained).  Previous approach of
    # selecting "best by mean reward" was unreliable because early
    # iterations have NaN returns and the metric fluctuates heavily.
    best_trial = results.trials[0] if results.trials else None
    best_checkpoint = None

    if best_trial:
        try:
            best_checkpoint = results.get_best_checkpoint(
                best_trial, "training_iteration", "max"
            )
            logger.info("Using latest checkpoint (by training_iteration)")
        except Exception as e:
            logger.error("Could not get checkpoint: %s", e)

        if best_checkpoint:
            tracker.best_checkpoint_path = best_checkpoint.path if hasattr(best_checkpoint, "path") else str(best_checkpoint)
            # Extract peak mean reward from trial dataframe
            try:
                reward_metric = "env_runners/episode_return_mean"
                df = results.trial_dataframes.get(best_trial.logdir)
                if df is not None and reward_metric in df.columns:
                    tracker.best_mean = df[reward_metric].max()
            except Exception:
                pass

    logger.info("Peak mean reward: %.1f (checkpoint: %s)", tracker.best_mean, tracker.best_checkpoint_path)

    last = best_trial.last_result if best_trial else {}
    env_runner_stats = last.get("env_runners", {})
    metadata = {
        "epoch": epoch,
        "checkpoint": tracker.best_checkpoint_path,
        "reward_mean": env_runner_stats.get("episode_return_mean", 0),
        "reward_max": env_runner_stats.get("episode_return_max", 0),
        "reward_min": env_runner_stats.get("episode_return_min", 0),
        "timesteps": last.get("num_env_steps_sampled_lifetime", 0),
        "episodes": env_runner_stats.get("num_episodes_lifetime", 0),
        "entropy_coeff": entropy_coeff,
        "best_mean": tracker.best_mean,
    }

    metadata_path = os.path.join("/tmp", f"epoch_{epoch}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Extract milestone custom metrics (reported by ZeldaCallbacks)
    # Check both top-level and env_runners locations
    custom = last.get("custom_metrics", {})
    if not custom:
        custom = env_runner_stats.get("custom_metrics", {})
    if not custom:
        # Try sampler_results for older RLlib
        custom = last.get("sampler_results", {}).get("custom_metrics", {})
    if custom:
        logger.info("Custom metrics keys: %s", list(custom.keys())[:10])
    else:
        logger.info("No custom_metrics found in result")
    # Custom metrics keys may have _mean suffix (aggregated) or not (raw).
    # Values can be floats or lists (per-episode); handle both.
    def _get_metric(d, name, default=0):
        val = d.get(f"{name}_mean", d.get(name, default))
        if isinstance(val, (list, tuple)):
            return sum(val) / len(val) if val else default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    milestones = {
        "got_sword_pct": _get_metric(custom, "got_sword") * 100,
        "entered_dungeon_pct": _get_metric(custom, "entered_dungeon") * 100,
        "visited_maku_tree_pct": _get_metric(custom, "visited_maku_tree") * 100,
        "maku_dialog_pct": _get_metric(custom, "maku_dialog") * 100,
        "gnarled_key_pct": _get_metric(custom, "gnarled_key") * 100,
        "avg_essences": _get_metric(custom, "essences"),
        "avg_dungeon_keys": _get_metric(custom, "dungeon_keys"),
        "avg_rooms": _get_metric(custom, "max_rooms"),
    }
    metadata["milestones"] = milestones

    logger.info("Epoch %d training complete:", epoch)
    logger.info("  Mean reward: %.2f", metadata["reward_mean"])
    logger.info("  Best mean reward: %.2f", tracker.best_mean)
    logger.info("  Max reward: %.2f", metadata["reward_max"])
    logger.info("  === MILESTONES ===")
    logger.info("  Got Sword: %.1f%% of episodes", milestones["got_sword_pct"])
    logger.info("  Entered Dungeon: %.1f%% of episodes", milestones["entered_dungeon_pct"])
    logger.info("  Visited Maku Tree: %.1f%% of episodes", milestones["visited_maku_tree_pct"])
    logger.info("  Maku Tree Dialog: %.1f%% of episodes", milestones["maku_dialog_pct"])
    logger.info("  Got Gnarled Key: %.1f%% of episodes", milestones["gnarled_key_pct"])
    logger.info("  Avg Essences: %.2f", milestones["avg_essences"])
    logger.info("  Avg Dungeon Keys: %.2f", milestones["avg_dungeon_keys"])
    logger.info("  Avg Rooms Explored: %.1f", milestones["avg_rooms"])
    logger.info("  Total episodes: %d", metadata["episodes"])
    logger.info("  Entropy coeff: %.4f", entropy_coeff)

    # Upload metadata to MinIO (checkpoints already stored via S3 storage)
    _upload_checkpoint(epoch, metadata, None)

    # Return checkpoint path and mean reward so the main loop can
    # track the global best and avoid cascading regressions.
    return tracker.best_checkpoint_path, metadata["reward_mean"]


def _upload_checkpoint(epoch, metadata, best_checkpoint):
    """Upload epoch metadata to MinIO. Checkpoints are stored via algo.save()."""
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
        logger.info("Metadata uploaded to MinIO: checkpoints/epoch_%d/", epoch)
    except Exception as e:
        logger.warning("Could not upload metadata to MinIO: %s", e)


def run_evaluation(epoch):
    """Run LLM judge evaluation on episode segments."""
    logger.info("=== EPOCH %d EVALUATION ===", epoch)

    try:
        from agent.evaluator.ingest import EvaluatorIngest
        from agent.evaluator.reward_model import RewardModel, build_preferences
        from agent.planner.llm_client import LLMClient
        from agent.utils.config import S3Config
        from agent.utils.s3 import S3Client

        max_segments = int(os.getenv("MAX_EVAL_SEGMENTS", "30"))

        s3 = S3Client(S3Config())
        llm = LLMClient()
        walkthrough_path = os.getenv("WALKTHROUGH_PATH", "data/zelda_oos_walkthrough.txt")
        evaluator = EvaluatorIngest(
            llm_client=llm,
            s3_client=s3,
            episodes_bucket="zelda-episodes",
            consistency_m=1,
            walkthrough_path=walkthrough_path if os.path.exists(walkthrough_path) else None,
        )

        # List only current epoch's segment manifests
        epoch_prefix = f"epoch_{epoch}/"
        logger.info("Listing segment manifests for %s (max %d)...", epoch_prefix, max_segments)
        manifest_keys = s3.list_manifests(
            "zelda-episodes", prefix=epoch_prefix, max_count=max_segments
        )
        segment_keys = [k.rsplit("/manifest.json", 1)[0] for k in manifest_keys]
        logger.info("Found %d segment manifests for epoch %d", len(manifest_keys), epoch)

        if not segment_keys:
            logger.warning("No segments found for evaluation. Skipping.")
            llm.close()
            return

        logger.info("Found %d segments to evaluate", len(segment_keys))

        # Evaluate through LLM judges
        results = evaluator.batch_evaluate(segment_keys)
        logger.info("Evaluated %d segments", len(results))

        # Write scores
        scores_key = f"scores/epoch_{epoch}/scores.jsonl"
        evaluator.write_scores(results, output_key=scores_key)

        # Train reward model on pairwise preferences
        if len(results) >= 2:
            logger.info(
                "Training reward model on %d scored segments...", len(results)
            )
            prefs = build_preferences(results)
            if prefs:
                rm = RewardModel()

                # Load previous epoch's model if exists
                if epoch > 0:
                    try:
                        rm_data = s3.download_bytes(
                            "zelda-models",
                            f"reward_model/epoch_{epoch - 1}/rm.pt",
                        )
                        with tempfile.NamedTemporaryFile(
                            suffix=".pt", delete=False
                        ) as f:
                            f.write(rm_data)
                            rm.load(f.name)
                        logger.info("Loaded reward model from epoch %d", epoch - 1)
                    except Exception as e:
                        logger.info("No previous reward model: %s", e)

                avg_loss = rm.train_on_preferences(prefs)
                logger.info(
                    "Reward model loss: %.4f (%d pairs)", avg_loss, len(prefs)
                )

                # Save reward model to MinIO
                with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                    rm.save(f.name)
                    with open(f.name, "rb") as rf:
                        rm_bytes = rf.read()

                s3.upload_bytes(
                    "zelda-models",
                    f"reward_model/epoch_{epoch}/rm.pt",
                    rm_bytes,
                )
                logger.info(
                    "Saved reward model: reward_model/epoch_%d/rm.pt", epoch
                )
            else:
                logger.warning(
                    "No valid preference pairs from %d results", len(results)
                )
        else:
            logger.warning(
                "Not enough segments (%d) for reward model", len(results)
            )

        # Write evaluation summary
        summary = {
            "epoch": epoch,
            "segments_evaluated": len(results),
            "mean_weighted_score": (
                sum(r.get("weighted_score", 0) for r in results) / len(results)
                if results
                else 0
            ),
            "score_distribution": {
                k: sum(r.get("scores", {}).get(k, 0) for r in results)
                / max(len(results), 1)
                for k in ["progress", "dialog", "puzzle", "novelty", "efficiency"]
            },
        }
        s3.upload_bytes(
            "zelda-models",
            f"evaluations/epoch_{epoch}/summary.json",
            json.dumps(summary, indent=2).encode(),
        )
        logger.info(
            "Evaluation summary: mean_score=%.3f", summary["mean_weighted_score"]
        )
        for k, v in summary["score_distribution"].items():
            logger.info("  %s: %.3f", k, v)

        llm.close()
    except Exception as e:
        logger.error("Evaluation failed: %s", e, exc_info=True)


def download_reward_model(epoch):
    """Check if reward model exists in MinIO for the next epoch.

    Returns the S3 key (not local path) so workers can download it themselves.
    """
    try:
        from agent.utils.s3 import S3Client
        from agent.utils.config import S3Config

        s3 = S3Client(S3Config())
        s3_key = f"reward_model/epoch_{epoch}/rm.pt"
        rm_data = s3.download_bytes("zelda-models", s3_key)
        logger.info(
            "Reward model available for epoch %d (%d bytes)", epoch, len(rm_data)
        )
        return s3_key
    except Exception as e:
        logger.info("No reward model available for epoch %d: %s", epoch, e)
        return ""


def run_reward_advisor(epoch: int, epoch_metadata: dict) -> dict | None:
    """Run the LLM reward advisor to adjust reward weights for the next epoch.

    Args:
        epoch: Current epoch number (just completed).
        epoch_metadata: Training metadata dict with reward_mean, milestones, etc.

    Returns:
        Updated reward config dict, or None if advisor fails/skips.
    """
    run_advisor = os.getenv("RUN_ADVISOR", "true").lower() in ("true", "1", "yes")
    if not run_advisor:
        logger.info("Reward advisor disabled (RUN_ADVISOR=false)")
        return None

    logger.info("=== EPOCH %d REWARD ADVISOR ===", epoch)

    try:
        from agent.evaluator.reward_advisor import RewardAdvisor
        from agent.planner.llm_client import LLMClient
        from agent.utils.config import RewardConfig, S3Config
        from agent.utils.s3 import S3Client

        walkthrough_path = os.getenv("WALKTHROUGH_PATH", "data/zelda_oos_walkthrough.txt")
        llm = LLMClient()
        advisor = RewardAdvisor(
            llm_client=llm,
            walkthrough_path=walkthrough_path if os.path.exists(walkthrough_path) else None,
        )

        # Gather segment summaries from this epoch's evaluation scores
        segment_summaries = []
        try:
            s3 = S3Client(S3Config())
            scores_key = f"scores/epoch_{epoch}/scores.jsonl"
            scores_data = s3.download_bytes("zelda-episodes", scores_key)
            lines = scores_data.decode().strip().split("\n")
            for line in lines[:5]:
                seg = json.loads(line)
                segment_summaries.append({
                    "rooms": seg.get("scores", {}).get("progress", 0),
                    "dialog_count": seg.get("scores", {}).get("dialog", 0),
                    "area": "overworld",
                    "total_reward": seg.get("weighted_score", 0),
                    "scores": seg.get("scores", {}),
                })
        except Exception as e:
            logger.info("Could not load segment scores for advisor: %s", e)

        # Get multipliers from LLM
        multipliers = advisor.advise(epoch_metadata, segment_summaries)

        if not multipliers:
            logger.info("Advisor returned no multipliers — keeping defaults")
            llm.close()
            return None

        # Apply multipliers to base config
        base = RewardConfig().model_dump()
        updated = advisor.apply_multipliers(base, multipliers)

        # Upload advice to MinIO for auditing
        try:
            s3 = S3Client(S3Config())
            advice_data = {
                "epoch": epoch,
                "multipliers": multipliers,
                "base_config": RewardConfig().model_dump(),
                "updated_config": updated,
                "epoch_stats": epoch_metadata,
            }
            s3.upload_bytes(
                "zelda-models",
                f"advice/epoch_{epoch}/advice.json",
                json.dumps(advice_data, indent=2).encode(),
            )
            logger.info("Saved reward advice to MinIO: advice/epoch_%d/advice.json", epoch)
        except Exception as e:
            logger.warning("Could not save advice to MinIO: %s", e)

        llm.close()

        logger.info("Reward advisor complete — %d parameters adjusted", len(multipliers))
        return updated

    except Exception as e:
        logger.error("Reward advisor failed: %s", e, exc_info=True)
        return None


def main():
    start_epoch = int(os.getenv("EPOCH", "0"))
    max_epochs = int(os.getenv("MAX_EPOCHS", "48"))
    epoch_steps = int(os.getenv("EPOCH_STEPS", "500000"))
    n_workers = int(os.getenv("RAY_WORKERS", "10"))
    n_envs = int(os.getenv("ENVS_PER_WORKER", "1"))
    ep_length = int(os.getenv("EPISODE_LENGTH", "30000"))
    run_eval = os.getenv("RUN_EVAL", "true").lower() in ("true", "1", "yes")
    clean_start = os.getenv("CLEAN_START", "true").lower() in ("true", "1", "yes")

    # Entropy scheduling parameters
    entropy_start = float(os.getenv("ENTROPY_START", "0.05"))
    entropy_end = float(os.getenv("ENTROPY_END", "0.015"))

    # God mode curriculum: infinite health for first N epochs
    god_mode_epochs = int(os.getenv("GOD_MODE_EPOCHS", "6"))

    # LR scheduling parameters
    lr_start = float(os.getenv("LR_START", "3e-4"))
    lr_end = float(os.getenv("LR_END", "1e-5"))

    # Register wrapped env
    from ray.tune.registry import register_env as ray_register

    ray_register("zelda_env", make_wrapped_env)

    ray.init()

    logger.info("=== ZELDA RL TRAINING PIPELINE ===")
    logger.info(
        "Epochs: %d-%d, Steps/epoch: %s",
        start_epoch,
        start_epoch + max_epochs - 1,
        f"{epoch_steps:,}",
    )
    logger.info("Entropy schedule: %.4f → %.4f", entropy_start, entropy_end)
    logger.info("LR schedule: %.2e → %.2e", lr_start, lr_end)
    logger.info("God mode curriculum: first %d epochs", god_mode_epochs)
    logger.info("Gamma: 0.999, Stagnation limit: 3000 steps (coord-based)")
    logger.info("Evaluation: %s", "enabled" if run_eval else "disabled")

    # Clean old MinIO data for fresh start
    if clean_start and start_epoch == 0:
        logger.info("Cleaning old MinIO data for fresh start...")
        clean_minio()

    prev_checkpoint = os.getenv("PREV_CHECKPOINT", "")
    rm_path = os.getenv("REWARD_MODEL_PATH", "")

    # Track the global best checkpoint across all epochs to prevent
    # cascading regressions: if an epoch collapses, the next epoch
    # restores from the best-ever checkpoint instead of the collapsed one.
    global_best_checkpoint = prev_checkpoint
    global_best_mean = -float("inf")
    reward_config = None  # LLM advisor will populate after first epoch

    for epoch in range(start_epoch, start_epoch + max_epochs):
        logger.info("=" * 60)
        logger.info("  EPOCH %d / %d", epoch, start_epoch + max_epochs - 1)
        logger.info("=" * 60)

        # Compute scheduled entropy and LR for this epoch
        entropy = compute_entropy(epoch, start_epoch, max_epochs, entropy_start, entropy_end)
        scheduled_lr = compute_lr(epoch, start_epoch, max_epochs, lr_start, lr_end)

        # Phase 1: Training — always restore from the global best
        checkpoint, mean_reward = run_training_epoch(
            epoch,
            start_epoch,
            epoch_steps,
            n_workers,
            n_envs,
            ep_length,
            rm_path,
            global_best_checkpoint,
            entropy,
            reward_config=reward_config,
            god_mode_epochs=god_mode_epochs,
            lr=scheduled_lr,
        )

        # Update global best if this epoch improved
        if mean_reward > global_best_mean:
            global_best_checkpoint = checkpoint
            global_best_mean = mean_reward
            logger.info(
                "New global best: mean=%.1f (epoch %d)", mean_reward, epoch
            )
        else:
            logger.info(
                "Epoch %d mean=%.1f below global best=%.1f — next epoch "
                "will restore from global best checkpoint",
                epoch, mean_reward, global_best_mean,
            )

        # Phase 2: Evaluation + reward model
        if run_eval:
            run_evaluation(epoch)

            # Phase 3: Download reward model for next epoch
            rm_path = download_reward_model(epoch)
            if rm_path:
                logger.info("Next epoch will use reward model shaping")
            else:
                logger.info("Next epoch runs without shaping")

            # Phase 3b: LLM reward advisor — adjust reward weights for next epoch
            epoch_metadata = {}
            metadata_path = os.path.join("/tmp", f"epoch_{epoch}_metadata.json")
            try:
                with open(metadata_path) as f:
                    epoch_metadata = json.load(f)
            except Exception as e:
                logger.warning("Could not load epoch metadata: %s", e)
                epoch_metadata = {"epoch": epoch, "reward_mean": mean_reward}

            advised_config = run_reward_advisor(epoch, epoch_metadata)
            if advised_config:
                reward_config = advised_config
                logger.info("Next epoch will use LLM-adjusted reward config")
            else:
                logger.info("Next epoch keeps current reward config")

        # Phase 4: Clean up storage to prevent MinIO disk-full crashes
        cleanup_after_epoch(epoch, global_best_checkpoint)

        logger.info("Epoch %d complete.", epoch)

    logger.info("=== PIPELINE FINISHED ===")


if __name__ == "__main__":
    main()
