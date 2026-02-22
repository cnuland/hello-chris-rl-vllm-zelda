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
    key: float = 0.5
    death: float = -1.0
    health_loss: float = -0.005
    new_room: float = 10.0
    movement: float = 0.1
    grid_exploration: float = 0.1
    maku_tree: float = 100.0
    sword: float = 15.0
    dungeon: float = 100.0
    dungeon_entry: float = 100.0
    maku_tree_visit: float = 100.0
    indoor_entry: float = 5.0
    dungeon_floor: float = 10.0
    dialog_bonus: float = 10.0
    dialog_advance: float = 25.0
    maku_dialog: float = 500.0
    gnarled_key: float = 500.0
    maku_seed: float = 1000.0
    gate_slash: float = 250.0
    maku_room: float = 100.0
    maku_stage: float = 300.0
    directional_bonus: float = 0.0
    area_boost_overworld: float = 1.0
    area_boost_subrosia: float = 1.5
    area_boost_maku: float = 3.0
    area_boost_indoors: float = 1.5
    area_boost_dungeon: float = 2.0


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
    gamma: float = 0.999
    lambda_: float = 0.95
    clip_param: float = 0.2
    entropy_coeff: float = 0.05
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
