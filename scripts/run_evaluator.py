"""Judge/evaluator phase — runs after each training epoch.

Reads episode segments from MinIO, sends them through the LLM judge
ensemble (vision + puzzle + state models), collects scores, trains
the Bradley-Terry reward model, and uploads it for the next epoch.

Environment variables:
  EPOCH              — Current epoch number
  LLM_GATEWAY_URL    — RHOAI 3 gateway URL
  LLM_NAMESPACE      — Namespace for LLMInferenceService models
  S3_ENDPOINT_URL    — MinIO endpoint
  S3_ACCESS_KEY      — MinIO access key
  S3_SECRET_KEY      — MinIO secret key
"""

from __future__ import annotations

import json
import logging
import os
import tempfile

from agent.evaluator.ingest import EvaluatorIngest
from agent.evaluator.reward_model import RewardModel, build_preferences
from agent.planner.llm_client import LLMClient
from agent.utils.config import S3Config
from agent.utils.s3 import S3Client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    epoch = int(os.getenv("EPOCH", "0"))
    logger.info("=== EVALUATOR PHASE — EPOCH %d ===", epoch)

    s3 = S3Client(S3Config())
    llm = LLMClient()
    evaluator = EvaluatorIngest(
        llm_client=llm,
        s3_client=s3,
        episodes_bucket="zelda-episodes",
        consistency_m=3,
    )

    # 1. List all episode segments from this epoch
    logger.info("Listing segments for epoch %d...", epoch)
    all_keys = s3.list_keys("zelda-episodes", prefix="")
    segment_keys = []
    for key in all_keys:
        if key.endswith("/manifest.json"):
            segment_prefix = key.rsplit("/manifest.json", 1)[0]
            segment_keys.append(segment_prefix)

    if not segment_keys:
        logger.warning("No segments found for evaluation. Skipping judge phase.")
        return

    logger.info("Found %d segments to evaluate", len(segment_keys))

    # 2. Evaluate segments through LLM judges
    results = evaluator.batch_evaluate(segment_keys)
    logger.info("Evaluated %d segments", len(results))

    # 3. Write scores
    scores_key = f"scores/epoch_{epoch}/scores.jsonl"
    evaluator.write_scores(results, output_key=scores_key)

    # 4. Train reward model on pairwise preferences
    if len(results) >= 2:
        logger.info("Training reward model on %d scored segments...", len(results))
        prefs = build_preferences(results)
        if prefs:
            rm = RewardModel()

            # Load previous epoch's model if it exists
            if epoch > 0:
                try:
                    rm_data = s3.download_bytes(
                        "zelda-models",
                        f"reward_model/epoch_{epoch - 1}/rm.pt",
                    )
                    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                        f.write(rm_data)
                        rm.load(f.name)
                    logger.info("Loaded reward model from epoch %d", epoch - 1)
                except Exception as e:
                    logger.info("No previous reward model found: %s", e)

            avg_loss = rm.train_on_preferences(prefs)
            logger.info("Reward model training loss: %.4f (%d pairs)", avg_loss, len(prefs))

            # Save reward model
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                rm.save(f.name)
                with open(f.name, "rb") as rf:
                    rm_bytes = rf.read()

            s3.upload_bytes(
                "zelda-models",
                f"reward_model/epoch_{epoch}/rm.pt",
                rm_bytes,
            )
            logger.info("Saved reward model to s3://zelda-models/reward_model/epoch_%d/rm.pt", epoch)
        else:
            logger.warning("No valid preference pairs from %d results", len(results))
    else:
        logger.warning("Not enough segments (%d) for reward model training", len(results))

    # 5. Write epoch evaluation summary
    summary = {
        "epoch": epoch,
        "segments_evaluated": len(results),
        "mean_weighted_score": (
            sum(r.get("weighted_score", 0) for r in results) / len(results)
            if results
            else 0
        ),
        "score_distribution": {
            k: sum(r.get("scores", {}).get(k, 0) for r in results) / max(len(results), 1)
            for k in ["progress", "dialog", "puzzle", "novelty", "efficiency"]
        },
    }
    s3.upload_bytes(
        "zelda-models",
        f"evaluations/epoch_{epoch}/summary.json",
        json.dumps(summary, indent=2).encode(),
    )
    logger.info("Epoch %d evaluation summary:", epoch)
    logger.info("  Segments: %d", summary["segments_evaluated"])
    logger.info("  Mean score: %.3f", summary["mean_weighted_score"])
    for k, v in summary["score_distribution"].items():
        logger.info("  %s: %.3f", k, v)

    llm.close()
    logger.info("=== EVALUATOR PHASE COMPLETE ===")


if __name__ == "__main__":
    main()
