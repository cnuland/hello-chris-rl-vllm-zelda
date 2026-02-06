"""Train reward model from judge scores.

Reads scores.jsonl, builds pairwise preferences,
trains Bradley-Terry reward model, saves to MinIO.
"""

from __future__ import annotations

import json
import logging

from agent.evaluator.reward_model import RewardModel, build_preferences
from agent.utils.config import S3Config
from agent.utils.s3 import S3Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    s3_config = S3Config()
    s3 = S3Client(s3_config)

    # Load scores
    bucket = s3_config.episodes_bucket
    scores_raw = s3.download_bytes(bucket, "scores.jsonl")
    scores = [json.loads(line) for line in scores_raw.decode().strip().split("\n") if line.strip()]
    logger.info("Loaded %d scored segments", len(scores))

    # Build preferences
    prefs = build_preferences(scores)
    logger.info("Built %d pairwise preferences", len(prefs))

    if not prefs:
        logger.warning("No preferences to train on (need >= 2 non-tied segments)")
        return

    # Train reward model
    rm = RewardModel()
    epochs = 50
    for epoch in range(epochs):
        loss = rm.train_on_preferences(prefs)
        if (epoch + 1) % 10 == 0:
            logger.info("Epoch %d/%d, loss=%.4f", epoch + 1, epochs, loss)

    # Save model
    rm.save("/tmp/rm.pt")
    s3.upload_file(s3_config.models_bucket, "reward_model/rm.pt", "/tmp/rm.pt")
    logger.info("Reward model saved to s3://%s/reward_model/rm.pt", s3_config.models_bucket)


if __name__ == "__main__":
    main()
