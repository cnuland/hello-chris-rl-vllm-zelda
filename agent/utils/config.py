"""Configuration loading and validation."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class EmulatorConfig(BaseModel):
    headless: bool = True
    frame_skip: int = 4
    rom_path: str = Field(default_factory=lambda: os.getenv("ROM_PATH", "roms/zelda.gbc"))


class ObservationConfig(BaseModel):
    vector_size: int = 128
    normalize: bool = True


class RewardConfig(BaseModel):
    rupee: float = 0.01
    key: float = 0.5
    death: float = -50.0
    health_loss: float = -0.1
    time_penalty: float = -0.0001
    new_room: float = 20.0
    movement: float = 0.1
    grid_exploration: float = 5.0
    revisit: float = -0.5
    maku_tree: float = 100.0
    sword: float = 200.0
    dungeon: float = 150.0


class LLMConfig(BaseModel):
    text_probability: float = 0.02
    vision_probability: float = 0.03
    text_alignment_bonus: float = 5.0
    vision_alignment_bonus: float = 50.0
    guidance_multiplier: float = 5.0


class TrainingConfig(BaseModel):
    ray_workers: int = Field(default_factory=lambda: int(os.getenv("RAY_WORKERS", "6")))
    envs_per_worker: int = Field(default_factory=lambda: int(os.getenv("ENVS_PER_WORKER", "6")))
    episode_length: int = Field(
        default_factory=lambda: int(os.getenv("EPISODE_LENGTH", str(2048 * 15)))
    )
    batch_size: int = Field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "4096")))
    lr: float = 3e-4
    gamma: float = 0.99
    lambda_: float = 0.95
    clip_param: float = 0.2
    entropy_coeff: float = 0.01
    num_gpus: int = 0


class S3Config(BaseModel):
    endpoint_url: str = Field(
        default_factory=lambda: os.getenv(
            "S3_ENDPOINT_URL", "http://minio-api.zelda-rl.svc.cluster.local:9000"
        )
    )
    access_key: str = Field(default_factory=lambda: os.getenv("S3_ACCESS_KEY", "admin"))
    secret_key: str = Field(
        default_factory=lambda: os.getenv("S3_SECRET_KEY", "zelda-rl-minio-2024")
    )
    episodes_bucket: str = Field(
        default_factory=lambda: os.getenv("S3_EPISODES_BUCKET", "zelda-episodes")
    )
    models_bucket: str = Field(
        default_factory=lambda: os.getenv("S3_MODELS_BUCKET", "zelda-models")
    )


class AgentConfig(BaseModel):
    emulator: EmulatorConfig = EmulatorConfig()
    observation: ObservationConfig = ObservationConfig()
    rewards: RewardConfig = RewardConfig()
    llm: LLMConfig = LLMConfig()
    training: TrainingConfig = TrainingConfig()
    s3: S3Config = S3Config()


def load_config(path: str | Path | None = None) -> AgentConfig:
    """Load config from YAML, falling back to defaults."""
    if path is None:
        return AgentConfig()
    path = Path(path)
    if not path.exists():
        return AgentConfig()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return AgentConfig(**data)
