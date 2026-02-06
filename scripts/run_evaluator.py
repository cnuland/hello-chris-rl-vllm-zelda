"""Evaluator pipeline entry point.

Reads segments from MinIO, scores them via llm-d judges,
and writes scores.jsonl back.
"""

from __future__ import annotations

import logging
import os

from agent.evaluator.ingest import EvaluatorIngest
from agent.planner.llm_client import LLMClient
from agent.utils.config import S3Config
from agent.utils.s3 import S3Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    s3_config = S3Config()
    s3 = S3Client(s3_config)
    llm = LLMClient()
    evaluator = EvaluatorIngest(llm_client=llm, s3_client=s3)

    # List all segment prefixes
    bucket = s3_config.episodes_bucket
    keys = s3.list_keys(bucket)
    segment_prefixes = set()
    for k in keys:
        parts = k.split("/")
        if len(parts) >= 2 and parts[1] != "scores.jsonl":
            segment_prefixes.add(f"{parts[0]}/{parts[1]}")

    logger.info("Found %d segments to evaluate", len(segment_prefixes))
    results = evaluator.batch_evaluate(list(segment_prefixes))
    evaluator.write_scores(results)
    logger.info("Evaluation complete: %d segments scored", len(results))


if __name__ == "__main__":
    main()
