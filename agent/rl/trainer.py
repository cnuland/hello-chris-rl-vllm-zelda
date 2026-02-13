"""PPO Trainer via Ray RLlib.

Registers custom Gymnasium env, configures PPO, and runs training.
CPU-only (num_gpus=0) to preserve GPUs for LLM inference.
Episodes export to MinIO on completion.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def make_env_config(
    rom_path: str = "roms/zelda.gbc",
    headless: bool = True,
    frame_skip: int = 4,
    max_steps: int = 30_000,
) -> dict:
    """Build env config dict for Ray."""
    return {
        "rom_path": rom_path,
        "headless": headless,
        "frame_skip": frame_skip,
        "max_steps": max_steps,
    }


class MilestoneCallbacks:
    """RLlib callback to track game progression milestones as custom metrics.

    Reports per-episode: got_sword, entered_dungeon, visited_maku_tree,
    essences collected, dungeon keys found, and rooms explored.
    """

    @staticmethod
    def on_episode_end(*, episode, **kwargs):
        """Extract milestone flags from the last info dict."""
        info = episode.last_info_for() or {}
        episode.custom_metrics["got_sword"] = info.get("milestone_got_sword", 0.0)
        episode.custom_metrics["entered_dungeon"] = info.get("milestone_entered_dungeon", 0.0)
        episode.custom_metrics["visited_maku_tree"] = info.get("milestone_visited_maku_tree", 0.0)
        episode.custom_metrics["essences"] = info.get("milestone_essences", 0.0)
        episode.custom_metrics["dungeon_keys"] = info.get("milestone_dungeon_keys", 0.0)
        episode.custom_metrics["max_rooms"] = info.get("milestone_max_rooms", 0.0)


def _make_callbacks_class():
    """Build a DefaultCallbacks subclass with milestone tracking."""
    from ray.rllib.algorithms.callbacks import DefaultCallbacks

    class ZeldaCallbacks(DefaultCallbacks):
        def on_episode_end(self, *, episode, **kwargs):
            # Try multiple access patterns for the last info dict
            info = None
            try:
                info = episode.last_info_for()
            except Exception:
                pass
            if not info:
                try:
                    info = episode.last_info_for("default_policy")
                except Exception:
                    pass
            if not info:
                try:
                    # Access via _last_infos in old API
                    infos = getattr(episode, "_last_infos", {})
                    if infos:
                        info = list(infos.values())[0] if isinstance(infos, dict) else infos
                except Exception:
                    pass
            if not info:
                return

            episode.custom_metrics["got_sword"] = info.get("milestone_got_sword", 0.0)
            episode.custom_metrics["entered_dungeon"] = info.get("milestone_entered_dungeon", 0.0)
            episode.custom_metrics["visited_maku_tree"] = info.get("milestone_visited_maku_tree", 0.0)
            episode.custom_metrics["essences"] = info.get("milestone_essences", 0.0)
            episode.custom_metrics["dungeon_keys"] = info.get("milestone_dungeon_keys", 0.0)
            episode.custom_metrics["max_rooms"] = info.get("milestone_max_rooms", 0.0)

    return ZeldaCallbacks


def create_ppo_config(
    env_config: dict | None = None,
    num_workers: int | None = None,
    envs_per_worker: int | None = None,
    batch_size: int | None = None,
    entropy_coeff: float | None = None,
) -> "PPOConfig":
    """Create Ray RLlib PPOConfig.

    All knobs are overridable via env vars:
      RAY_WORKERS, ENVS_PER_WORKER, EPISODE_LENGTH, BATCH_SIZE.

    Args:
        entropy_coeff: Override entropy coefficient for entropy scheduling.
            If None, uses 0.05 default.
    """
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.models import ModelCatalog

    from agent.rl.model import ZeldaMLPModel

    ModelCatalog.register_custom_model("zelda_mlp", ZeldaMLPModel)

    n_workers = num_workers or int(os.getenv("RAY_WORKERS", "6"))
    n_envs = envs_per_worker or int(os.getenv("ENVS_PER_WORKER", "6"))
    bs = batch_size or int(os.getenv("BATCH_SIZE", "4096"))
    ent = entropy_coeff if entropy_coeff is not None else 0.05

    if env_config is None:
        ep_len = int(os.getenv("EPISODE_LENGTH", str(2048 * 15)))
        env_config = make_env_config(max_steps=ep_len)

    callbacks_cls = _make_callbacks_class()

    config = (
        PPOConfig()
        .environment(env="zelda_env", env_config=env_config)
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .env_runners(
            num_env_runners=n_workers,
            num_envs_per_env_runner=n_envs,
            rollout_fragment_length=2048,
            sample_timeout_s=300.0,
        )
        .callbacks(callbacks_cls)
        .reporting(
            keep_per_episode_custom_metrics=True,
        )
        .training(
            model={
                "custom_model": "zelda_mlp",
                "fcnet_hiddens": [256],
            },
            lr=5e-5,
            gamma=0.999,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=200.0,
            entropy_coeff=ent,
            train_batch_size_per_learner=bs,
            minibatch_size=min(bs, 2048),
            num_epochs=2,
            grad_clip=0.5,
        )
        .resources(num_gpus=0)
    )
    return config


def register_env() -> None:
    """Register the Zelda env with Ray."""
    from ray.tune.registry import register_env as ray_register

    from agent.env.zelda_env import ZeldaEnv

    def _make(config):
        return ZeldaEnv(**config)

    ray_register("zelda_env", _make)
