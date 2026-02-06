"""Evaluator ingest: batch segments through llm-d judges.

Pipeline:
  1. Read segments from MinIO.
  2. Fan out to vision/state/rule judges via llm-d.
  3. Self-consistency M=3 (three passes), majority vote.
  4. Write scores.jsonl back to MinIO.
"""

from __future__ import annotations

import json
import logging
import statistics
from typing import Any

logger = logging.getLogger(__name__)

# Rubric weights (from new/JUDGES_RUBRIC.md)
RUBRIC_WEIGHTS = {
    "progress": 0.4,
    "dialog": 0.2,
    "puzzle": 0.2,
    "novelty": 0.1,
    "efficiency": 0.1,
}

SCORE_KEYS = list(RUBRIC_WEIGHTS.keys())


class EvaluatorIngest:
    """Batch evaluate episode segments using LLM judges."""

    def __init__(
        self,
        llm_client: Any = None,
        s3_client: Any = None,
        episodes_bucket: str = "zelda-episodes",
        consistency_m: int = 3,
    ):
        self._llm = llm_client
        self._s3 = s3_client
        self._bucket = episodes_bucket
        self._m = consistency_m

    def evaluate_segment(self, segment_data: dict[str, Any]) -> dict[str, Any]:
        """Evaluate a single segment with M=3 self-consistency.

        Args:
            segment_data: Segment manifest + state data.

        Returns:
            Judge output matching SCHEMAS.md Judge Output format.
        """
        all_scores: list[dict[str, float]] = []

        for trial in range(self._m):
            scores = self._judge_once(segment_data, trial)
            all_scores.append(scores)

        # Majority vote: median per dimension
        final_scores = {}
        for key in SCORE_KEYS:
            values = [s.get(key, 0.0) for s in all_scores]
            final_scores[key] = statistics.median(values)

        # Weighted aggregate
        weighted = sum(final_scores[k] * RUBRIC_WEIGHTS[k] for k in SCORE_KEYS)

        result = {
            "segment_id": segment_data.get("segment_id", "unknown"),
            "scores": final_scores,
            "weighted_score": weighted,
            "rationale": self._generate_rationale(final_scores),
            "raw_trials": all_scores,
        }
        return result

    def _judge_once(self, segment_data: dict[str, Any], trial: int) -> dict[str, float]:
        """Run one judge pass over a segment."""
        if self._llm is None:
            return {k: 0.5 for k in SCORE_KEYS}

        prompt = self._build_judge_prompt(segment_data)
        try:
            result = self._llm.state({"judge_prompt": prompt, "trial": trial})
            if "scores" in result:
                scores = result["scores"]
                return {k: float(scores.get(k, 0.0)) for k in SCORE_KEYS}
        except Exception as e:
            logger.warning("Judge trial %d failed: %s", trial, e)

        return {k: 0.5 for k in SCORE_KEYS}

    def _build_judge_prompt(self, segment_data: dict[str, Any]) -> str:
        """Build evaluation prompt for the judge model."""
        states = segment_data.get("states", [])
        summary = f"Segment {segment_data.get('segment_id', '?')}: "
        summary += f"{len(states)} frames, "
        summary += f"reward={segment_data.get('total_reward', 0):.1f}. "

        if states:
            first = states[0].get("state", {})
            last = states[-1].get("state", {})
            summary += f"Start room={first.get('room_id', '?')}, "
            summary += f"End room={last.get('room_id', '?')}."

        return (
            f"Rate this game segment on: progress, dialog, puzzle, novelty, efficiency. "
            f"Each 0.0-1.0. {summary} "
            f"Output JSON: {{scores: {{progress, dialog, puzzle, novelty, efficiency}}, rationale: str}}"
        )

    @staticmethod
    def _generate_rationale(scores: dict[str, float]) -> str:
        """Generate a short rationale from scores."""
        best = max(scores, key=scores.get)
        worst = min(scores, key=scores.get)
        return f"Best: {best} ({scores[best]:.2f}), Weakest: {worst} ({scores[worst]:.2f})"

    def batch_evaluate(self, segment_keys: list[str]) -> list[dict[str, Any]]:
        """Evaluate multiple segments and return all results."""
        results = []
        for key in segment_keys:
            try:
                if self._s3:
                    manifest = self._s3.download_json(self._bucket, f"{key}/manifest.json")
                    states_raw = self._s3.download_bytes(self._bucket, f"{key}/states.jsonl")
                    states = [json.loads(line) for line in states_raw.decode().strip().split("\n")]
                    manifest["states"] = states
                else:
                    manifest = {"segment_id": key}

                result = self.evaluate_segment(manifest)
                results.append(result)
            except Exception as e:
                logger.error("Failed to evaluate segment %s: %s", key, e)

        return results

    def write_scores(self, results: list[dict[str, Any]], output_key: str = "scores.jsonl") -> None:
        """Write evaluation results as JSONL to MinIO."""
        if self._s3 is None:
            return
        lines = [json.dumps(r) for r in results]
        self._s3.upload_bytes(self._bucket, output_key, "\n".join(lines).encode())
        logger.info("Wrote %d scores to s3://%s/%s", len(results), self._bucket, output_key)
