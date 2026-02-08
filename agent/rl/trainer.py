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


def create_ppo_config(
    env_config: dict | None = None,
    num_workers: int | None = None,
    envs_per_worker: int | None = None,
    batch_size: int | None = None,
) -> "PPOConfig":
    """Create Ray RLlib PPOConfig.

    All knobs are overridable via env vars:
      RAY_WORKERS, ENVS_PER_WORKER, EPISODE_LENGTH, BATCH_SIZE.
    """
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.models import ModelCatalog

    from agent.rl.model import ZeldaMLPModel

    ModelCatalog.register_custom_model("zelda_mlp", ZeldaMLPModel)

    n_workers = num_workers or int(os.getenv("RAY_WORKERS", "6"))
    n_envs = envs_per_worker or int(os.getenv("ENVS_PER_WORKER", "6"))
    bs = batch_size or int(os.getenv("BATCH_SIZE", "4096"))

    if env_config is None:
        ep_len = int(os.getenv("EPISODE_LENGTH", str(2048 * 15)))
        env_config = make_env_config(max_steps=ep_len)

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
            rollout_fragment_length=200,
            sample_timeout_s=300.0,
        )
        .training(
            model={
                "custom_model": "zelda_mlp",
                "fcnet_hiddens": [256],
            },
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            train_batch_size_per_learner=bs,
            minibatch_size=min(bs, 512),
            num_epochs=10,
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
