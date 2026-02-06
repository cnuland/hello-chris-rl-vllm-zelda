"""Ray RLlib PPO training entry point.

Submitted as a RayJob to the KubeRay cluster.
Mirrors old/run-ray-zelda.py with new module structure.
"""

from __future__ import annotations

import os

import ray
from ray import tune

from agent.rl.trainer import create_ppo_config, register_env


def main():
    register_env()
    ray.init()

    config = create_ppo_config()
    ep_length = int(os.getenv("EPISODE_LENGTH", str(2048 * 15)))
    total_timesteps = ep_length * 10_000  # ~300M

    n_workers = int(os.getenv("RAY_WORKERS", "6"))
    n_envs = int(os.getenv("ENVS_PER_WORKER", "6"))
    print(f"Starting PPO training: {n_workers} workers x {n_envs} envs")
    print(f"Total parallel environments: {n_workers * n_envs}")
    print(f"Target timesteps: {total_timesteps:,}")

    tune.run(
        "PPO",
        name="PPO_ZeldaOracleSeasons",
        stop={"timesteps_total": total_timesteps},
        checkpoint_freq=0,
        config=config.to_dict(),
    )
    print("Training complete.")


if __name__ == "__main__":
    main()
