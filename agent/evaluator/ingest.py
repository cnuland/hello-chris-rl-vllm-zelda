"""Evaluator ingest: batch segments through llm-d judges.

Pipeline:
  1. Read segments from MinIO.
  2. Fan out to vision/state/puzzle judges via llm-d gateway.
  3. Self-consistency M=3 (three passes), majority vote.
  4. Write scores.jsonl back to MinIO.

Judge model mapping:
  - state (qwen25-7b):     progress, dialog, efficiency scoring
  - puzzle (qwen25-32b):   puzzle scoring
  - vision (qwen25-vl-32b): novelty scoring (frame analysis)
"""

from __future__ import annotations

import base64
import json
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
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

        Uses multiple judge models:
          - State judge (qwen25-7b) for progress, dialog, efficiency
          - Puzzle judge (qwen25-32b) for puzzle scoring
          - Vision judge (qwen25-vl-32b) for novelty (if frames available)
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
        """Run one judge pass over a segment using multiple models in parallel."""
        if self._llm is None:
            return {k: 0.5 for k in SCORE_KEYS}

        scores = {}

        def _call_state():
            state_prompt = self._build_state_prompt(segment_data)
            result = self._llm.state({"judge_prompt": state_prompt, "trial": trial})
            partial = {}
            if "scores" in result:
                for k in ["progress", "dialog", "efficiency"]:
                    partial[k] = float(result["scores"].get(k, 0.5))
            return partial

        def _call_puzzle():
            puzzle_prompt = self._build_puzzle_prompt(segment_data)
            result = self._llm.puzzle({"judge_prompt": puzzle_prompt, "trial": trial})
            if "scores" in result:
                return {"puzzle": float(result["scores"].get("puzzle", 0.5))}
            elif "confidence" in result:
                return {"puzzle": float(result.get("confidence", 0.5))}
            return {}

        def _call_vision():
            frame_b64 = segment_data.get("frame_b64")
            if not frame_b64:
                return {}
            vision_prompt = self._build_vision_prompt(segment_data)
            game_state = {}
            if segment_data.get("states"):
                mid = len(segment_data["states"]) // 2
                game_state = segment_data["states"][mid].get("state", {})
            result = self._llm.vision(frame_b64, game_state, prompt=vision_prompt)
            if "scores" in result:
                return {"novelty": float(result["scores"].get("novelty", 0.5))}
            elif "novelty" in result:
                return {"novelty": float(result["novelty"])}
            return {}

        # Run all 3 judges in parallel
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(_call_state): "state",
                pool.submit(_call_puzzle): "puzzle",
                pool.submit(_call_vision): "vision",
            }
            for fut in as_completed(futures):
                judge_name = futures[fut]
                try:
                    partial = fut.result()
                    scores.update(partial)
                except Exception as e:
                    logger.warning("%s judge trial %d failed: %s", judge_name, trial, e)

        # Fill defaults for any missing scores
        for k in SCORE_KEYS:
            if k not in scores:
                scores[k] = 0.5

        return scores

    def _build_state_prompt(self, segment_data: dict[str, Any]) -> str:
        """Build evaluation prompt for the state judge model."""
        states = segment_data.get("states", [])
        summary = f"Segment {segment_data.get('segment_id', '?')}: "
        summary += f"{len(states)} frames, "
        summary += f"reward={segment_data.get('total_reward', 0):.1f}. "

        if states:
            first = states[0].get("state", {})
            last = states[-1].get("state", {})
            summary += f"Start room={first.get('room_id', '?')}, "
            summary += f"End room={last.get('room_id', '?')}. "

            # Compute some metrics
            rooms = set()
            dialogs = 0
            for s in states:
                st = s.get("state", {})
                rooms.add(st.get("room_id", 0))
                if st.get("dialog_active"):
                    dialogs += 1
            summary += f"Unique rooms visited: {len(rooms)}. "
            summary += f"Dialog interactions: {dialogs}."

        return (
            f"Rate this Zelda game segment on: progress, dialog, efficiency. "
            f"Each score 0.0-1.0. {summary} "
            f"Output JSON: {{\"scores\": {{\"progress\": float, \"dialog\": float, \"efficiency\": float}}}}"
        )

    def _build_puzzle_prompt(self, segment_data: dict[str, Any]) -> str:
        """Build evaluation prompt for the puzzle judge model."""
        states = segment_data.get("states", [])
        summary = f"Segment {segment_data.get('segment_id', '?')}: "
        summary += f"{len(states)} frames. "

        if states:
            puzzle_flags = set()
            for s in states:
                st = s.get("state", {})
                pf = st.get("puzzle_flags", 0)
                if pf:
                    puzzle_flags.add(pf)
            summary += f"Puzzle flag changes: {len(puzzle_flags)}."

        return (
            f"Rate this Zelda game segment on puzzle-solving skill. "
            f"Score 0.0-1.0. {summary} "
            f"Output JSON: {{\"scores\": {{\"puzzle\": float}}, \"rationale\": str}}"
        )

    def _build_vision_prompt(self, segment_data: dict[str, Any]) -> str:
        """Build evaluation prompt for the vision judge model."""
        states = segment_data.get("states", [])
        summary = f"Segment {segment_data.get('segment_id', '?')}: "
        summary += f"{len(states)} frames. "

        if states:
            rooms = set()
            for s in states:
                rooms.add(s.get("state", {}).get("room_id", 0))
            summary += f"Rooms visited: {len(rooms)}. "
            summary += f"Total reward: {segment_data.get('total_reward', 0):.1f}."

        return (
            f"Analyze this Zelda game screenshot for novelty and exploration. "
            f"Rate how novel/interesting the game state is on a 0.0-1.0 scale. "
            f"Consider: new areas explored, unique enemy encounters, items found, "
            f"environmental variety. {summary} "
            f'Output JSON: {{"scores": {{"novelty": float}}, "rationale": str}}'
        )

    @staticmethod
    def _generate_rationale(scores: dict[str, float]) -> str:
        """Generate a short rationale from scores."""
        best = max(scores, key=scores.get)
        worst = min(scores, key=scores.get)
        return f"Best: {best} ({scores[best]:.2f}), Weakest: {worst} ({scores[worst]:.2f})"

    def _load_and_evaluate(self, key: str) -> dict[str, Any] | None:
        """Load a single segment from S3 and evaluate it."""
        try:
            if self._s3:
                manifest = self._s3.download_json(self._bucket, f"{key}/manifest.json")
                states_raw = self._s3.download_bytes(self._bucket, f"{key}/states.jsonl")
                states = [json.loads(line) for line in states_raw.decode().strip().split("\n")]
                manifest["states"] = states

                # Download a representative frame PNG for the vision judge
                frame_keys = [
                    k for k in self._s3.list_keys(self._bucket, prefix=f"{key}/frames/")
                    if k.endswith(".png")
                ]
                if frame_keys:
                    mid_frame = frame_keys[len(frame_keys) // 2]
                    png_bytes = self._s3.download_bytes(self._bucket, mid_frame)
                    manifest["frame_b64"] = base64.b64encode(png_bytes).decode("ascii")
                    logger.info("Loaded frame %s for vision judge", mid_frame)
            else:
                manifest = {"segment_id": key}

            return self.evaluate_segment(manifest)
        except Exception as e:
            logger.error("Failed to evaluate segment %s: %s", key, e)
            return None

    def batch_evaluate(self, segment_keys: list[str], max_workers: int = 4) -> list[dict[str, Any]]:
        """Evaluate multiple segments in parallel and return all results."""
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self._load_and_evaluate, key): key for key in segment_keys}
            done = 0
            for fut in as_completed(futures):
                done += 1
                result = fut.result()
                if result is not None:
                    results.append(result)
                if done % 20 == 0:
                    logger.info("Evaluated %d/%d segments...", done, len(segment_keys))

        return results

    def write_scores(self, results: list[dict[str, Any]], output_key: str = "scores.jsonl") -> None:
        """Write evaluation results as JSONL to MinIO."""
        if self._s3 is None:
            return
        lines = [json.dumps(r) for r in results]
        self._s3.upload_bytes(self._bucket, output_key, "\n".join(lines).encode())
        logger.info("Wrote %d scores to s3://%s/%s", len(results), self._bucket, output_key)
