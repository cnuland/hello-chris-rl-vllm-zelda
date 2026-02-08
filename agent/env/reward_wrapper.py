"""Reward wrapper: integrates coverage, game events, RND, and potential shaping.

Wraps ZeldaEnv to compute composite rewards from:
  1. Game event rewards (health, rupees, keys, rooms, sword, death)
  2. Coverage rewards (tile exploration, new rooms)
  3. RND curiosity bonus (clamped)
  4. Potential-based shaping from RLAIF reward model (when available)

Also handles episode export to MinIO for the judge/eval pipeline.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import gymnasium as gym
import numpy as np

from agent.rl.rewards import CoverageReward, PotentialShaping, RNDCuriosity

logger = logging.getLogger(__name__)


class RewardWrapper(gym.Wrapper):
    """Wraps ZeldaEnv with composite reward computation and episode export."""

    def __init__(
        self,
        env: gym.Env,
        reward_config: dict[str, float] | None = None,
        enable_rnd: bool = True,
        enable_shaping: bool = False,
        reward_model_path: str | None = None,
        enable_export: bool = True,
        s3_config: dict[str, str] | None = None,
        epoch: int = 0,
    ):
        super().__init__(env)
        cfg = reward_config or {}

        # Game event reward scales
        self._rupee_scale = cfg.get("rupee", 0.01)
        self._key_scale = cfg.get("key", 0.5)
        self._death_penalty = cfg.get("death", -50.0)
        self._health_loss_scale = cfg.get("health_loss", -0.1)
        self._time_penalty = cfg.get("time_penalty", -0.0001)
        self._sword_bonus = cfg.get("sword", 200.0)
        self._dungeon_bonus = cfg.get("dungeon", 150.0)
        self._maku_tree_bonus = cfg.get("maku_tree", 100.0)

        # Sub-reward modules
        self._coverage = CoverageReward(
            bonus_per_tile=cfg.get("grid_exploration", 5.0),
            bonus_per_room=cfg.get("new_room", 20.0),
            revisit_penalty=cfg.get("revisit", -0.5),
        )

        self._rnd = RNDCuriosity() if enable_rnd else None

        self._shaping = PotentialShaping() if enable_shaping else None
        self._reward_model = None
        if enable_shaping and reward_model_path:
            self._load_reward_model(reward_model_path)

        # Episode export
        self._exporter = None
        self._enable_export = enable_export
        if enable_export:
            self._init_exporter(s3_config)

        # Tracking
        self._prev_health = 0
        self._prev_rupees = 0
        self._prev_keys = 0
        self._prev_sword = 0
        self._prev_essences = 0
        self._epoch = epoch
        self._episode_id = ""

    def _load_reward_model(self, path: str) -> None:
        try:
            from agent.evaluator.reward_model import RewardModel

            self._reward_model = RewardModel()
            self._reward_model.load(path)
            logger.info("Loaded reward model from %s", path)
        except Exception as e:
            logger.warning("Could not load reward model: %s", e)
            self._reward_model = None

    def _init_exporter(self, s3_config: dict[str, str] | None) -> None:
        try:
            from agent.evaluator.exporter import EpisodeExporter
            from agent.utils.s3 import S3Client
            from agent.utils.config import S3Config

            if s3_config:
                cfg = S3Config(**s3_config)
            else:
                cfg = S3Config()
            s3 = S3Client(cfg)
            s3.ensure_bucket(cfg.episodes_bucket)
            self._exporter = EpisodeExporter(
                s3_client=s3,
                bucket=cfg.episodes_bucket,
                frames_per_segment=300,
                png_interval=30,
            )
        except Exception as e:
            logger.warning("Episode export disabled: %s", e)
            self._exporter = None

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)

        # Snapshot baseline state
        self._prev_health = info.get("health", 0)
        self._prev_rupees = 0
        self._prev_keys = 0
        self._prev_sword = self.env._read(0xC668) if hasattr(self.env, "_read") else 0
        self._prev_essences = 0

        # Reset sub-modules
        self._coverage.reset()
        if self._shaping:
            self._shaping.reset()

        # Start episode recording
        if self._exporter:
            self._episode_id = self._exporter.begin_episode()
            info["episode_id"] = self._episode_id

        info["epoch"] = self._epoch
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, _, terminated, truncated, info = self.env.step(action)

        reward = self._compute_reward(obs, info, terminated)

        # Record frame for export
        if self._exporter:
            screen = self.env.render() if hasattr(self.env, "render") else None
            self._exporter.record_frame(
                step=info.get("step", 0),
                state=info,
                action=action,
                reward=reward,
                screen_array=screen,
            )

        # Flush on episode end
        if (terminated or truncated) and self._exporter:
            self._exporter.end_episode()

        info["epoch"] = self._epoch
        return obs, reward, terminated, truncated, info

    def _compute_reward(
        self, obs: np.ndarray, info: dict[str, Any], terminated: bool
    ) -> float:
        reward = 0.0

        # --- Game event rewards ---
        health = info.get("health", 0)
        health_delta = health - self._prev_health
        if health_delta < 0:
            reward += health_delta * abs(self._health_loss_scale)
        self._prev_health = health

        # Death
        if terminated:
            reward += self._death_penalty

        # Time penalty
        reward += self._time_penalty

        # Rupees (read from env if available)
        if hasattr(self.env, "_read16"):
            rupees = self.env._read16(0xC6A5)
            rupee_delta = rupees - self._prev_rupees
            if rupee_delta > 0:
                reward += rupee_delta * self._rupee_scale
            self._prev_rupees = rupees

        # Keys
        if hasattr(self.env, "_read"):
            keys = self.env._read(0xC694)
            if keys > self._prev_keys:
                reward += (keys - self._prev_keys) * self._key_scale
            self._prev_keys = keys

            # Sword upgrade
            sword = self.env._read(0xC668)
            if sword > self._prev_sword:
                reward += self._sword_bonus
            self._prev_sword = sword

            # Essences
            essences = bin(self.env._read(0xC692)).count("1")
            if essences > self._prev_essences:
                reward += self._dungeon_bonus
            self._prev_essences = essences

        # --- Coverage reward ---
        coverage = self._coverage.step(
            info.get("room_id", 0),
            info.get("pixel_x", 0),
            info.get("pixel_y", 0),
        )
        reward += coverage

        # --- RND curiosity ---
        if self._rnd is not None:
            curiosity = self._rnd.compute(obs, reward)
            reward += curiosity

        # --- Potential-based shaping ---
        if self._shaping is not None and self._reward_model is not None:
            phi = self._reward_model.predict(obs)
            reward = self._shaping.shape(reward, phi)

        return reward
