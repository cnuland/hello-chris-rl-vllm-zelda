"""Evaluator ingest: batch segments through phase-aware llm-d judges.

Pipeline:
  1. Detect the agent's current game phase from epoch milestones.
  2. Read segments from MinIO.
  3. Fan out to vision/state/puzzle judges via llm-d gateway with
     phase-specific prompts and dynamic rubric weights.
  4. Self-consistency M=3 (three passes), majority vote.
  5. Write scores.jsonl back to MinIO.

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

from agent.evaluator.phase_detector import detect_phase

logger = logging.getLogger(__name__)

# Default rubric weights (fallback when phase is unknown)
DEFAULT_RUBRIC_WEIGHTS = {
    "progress": 0.40,
    "dialog": 0.20,
    "puzzle": 0.20,
    "novelty": 0.10,
    "efficiency": 0.10,
}

# Phase-specific rubric weights — shift emphasis based on what matters
# most at each stage of the game.  Key design decisions:
#   - pre_sword: novelty high (0.25) because exploration is the core task
#   - pre_maku: progress high (0.45) because directional movement matters
#   - maku_interaction: dialog jumps to 0.40 — this is THE critical blocker
#   - dungeon: puzzle at 0.40 because puzzles ARE the dungeon gameplay
PHASE_WEIGHTS: dict[str, dict[str, float]] = {
    "pre_sword": {
        "progress": 0.30,
        "dialog": 0.10,
        "puzzle": 0.15,
        "novelty": 0.25,
        "efficiency": 0.20,
    },
    "pre_maku": {
        "progress": 0.45,
        "dialog": 0.10,
        "puzzle": 0.10,
        "novelty": 0.20,
        "efficiency": 0.15,
    },
    "maku_interaction": {
        "progress": 0.20,
        "dialog": 0.40,
        "puzzle": 0.20,
        "novelty": 0.05,
        "efficiency": 0.15,
    },
    "pre_dungeon": {
        "progress": 0.45,
        "dialog": 0.05,
        "puzzle": 0.10,
        "novelty": 0.25,
        "efficiency": 0.15,
    },
    "dungeon": {
        "progress": 0.25,
        "dialog": 0.05,
        "puzzle": 0.40,
        "novelty": 0.15,
        "efficiency": 0.15,
    },
}

# Phase-specific judge prompt guidance — tells each judge what
# "good behavior" looks like at each stage of the game.
PHASE_PROMPTS: dict[str, dict[str, str]] = {
    "pre_sword": {
        "state": (
            "The agent is in the EARLY GAME and needs to find the Hero's Cave "
            "to obtain the sword. Good progress means: exploring indoor areas "
            "(active_group 3), finding new rooms, moving toward caves. "
            "Dialog is less important at this stage. Efficiency means not "
            "standing still or looping through the same rooms."
        ),
        "puzzle": (
            "In the early game, puzzle interaction is minimal. Score based on "
            "whether the agent interacts with environmental objects (pushing, "
            "slashing bushes to find cave entrances)."
        ),
        "vision": (
            "Score novelty based on whether the screenshot shows new areas "
            "the agent hasn't visited before. Indoor areas and cave entrances "
            "are especially valuable."
        ),
    },
    "pre_maku": {
        "state": (
            "The agent HAS the sword and needs to travel EAST then NORTH to "
            "reach the Maku Tree (active_group 2). Good progress means: "
            "increasing column number in the room grid (room_id %% 16 should "
            "increase), discovering new overworld rooms, moving toward the "
            "northeast quadrant. The Maku Tree area is around row 5, col 12."
        ),
        "puzzle": (
            "Overworld puzzles at this stage involve cutting bushes, pushing "
            "rocks, and navigating around obstacles to head east. Score any "
            "progress-enabling environmental interaction."
        ),
        "vision": (
            "Score novelty highly for screenshots showing new overworld areas, "
            "especially areas to the east and north. The Maku Tree area has "
            "distinctive grove/tree visuals."
        ),
    },
    "maku_interaction": {
        "state": (
            "The agent is AT or NEAR the Maku Tree (active_group 2). "
            "DIALOG IS CRITICAL — the agent must interact with the Maku Tree "
            "NPC to receive the Gnarled Key quest. Score dialog very highly "
            "(0.8-1.0) if the agent triggers ANY dialog in group 2. "
            "Score progress highly for staying in group 2 and exploring new "
            "rooms within it. Penalize leaving group 2 unnecessarily."
        ),
        "puzzle": (
            "At the Maku Tree, the agent needs to: slash the gate (sword on "
            "a specific tile), navigate through the grove, pop a bubble "
            "(attack), and advance dialog. Score ANY interaction with "
            "objects in group 2 highly."
        ),
        "vision": (
            "At the Maku Tree, the screen should show the grove area with "
            "the large tree NPC. Score screenshots showing the Maku Tree "
            "and dialog boxes very highly. Score screenshots of the overworld "
            "(outside group 2) low."
        ),
    },
    "pre_dungeon": {
        "state": (
            "The agent has the Gnarled Key quest and needs to travel WEST to "
            "find Dungeon 1 (Gnarled Root). Good progress: decreasing column "
            "number, discovering rooms to the west, entering active_group "
            "4 or 5 (dungeon). Dialog is less important now."
        ),
        "puzzle": (
            "The agent may need to solve environmental puzzles to reach the "
            "dungeon entrance (season changes, bush cutting, rock pushing). "
            "Score any puzzle-solving behavior that opens paths westward."
        ),
        "vision": (
            "Score novelty for new western overworld areas. The dungeon "
            "entrance is a distinctive cave opening. Score dungeon interior "
            "screenshots very highly."
        ),
    },
    "dungeon": {
        "state": (
            "The agent is IN a dungeon (active_group 4-5). Progress means: "
            "advancing floors, finding keys, defeating enemies, reaching the "
            "boss. Score room exploration within the dungeon highly."
        ),
        "puzzle": (
            "Dungeon puzzles are the core challenge: pushing blocks, hitting "
            "switches, using keys on locked doors, navigating dark rooms, "
            "defeating mini-bosses. Score any puzzle interaction very highly."
        ),
        "vision": (
            "Dungeon interiors have distinctive tile patterns and objects. "
            "Score screenshots showing new dungeon rooms, boss encounters, "
            "puzzle elements (blocks, switches), and treasure chests highly."
        ),
    },
}

SCORE_KEYS = list(DEFAULT_RUBRIC_WEIGHTS.keys())


class EvaluatorIngest:
    """Phase-aware batch evaluator for episode segments using LLM judges."""

    def __init__(
        self,
        llm_client: Any = None,
        s3_client: Any = None,
        episodes_bucket: str = "zelda-episodes",
        consistency_m: int = 3,
        walkthrough_path: str | None = None,
        milestones: dict | None = None,
        weight_overrides: dict[str, float] | None = None,
    ):
        self._llm = llm_client
        self._s3 = s3_client
        self._bucket = episodes_bucket
        self._m = consistency_m

        # Detect game phase from milestones
        self._phase = detect_phase(milestones or {})
        logger.info("Evaluator phase: %s", self._phase)

        # Validate and store advisor weight overrides
        self._weight_overrides = self._validate_weights(weight_overrides)

        # Load game walkthrough for judge context
        self._walkthrough = ""
        if walkthrough_path:
            try:
                with open(walkthrough_path) as f:
                    self._walkthrough = f.read()
                logger.info("Loaded walkthrough (%d chars) from %s", len(self._walkthrough), walkthrough_path)
            except Exception as e:
                logger.warning("Could not load walkthrough: %s", e)

    @staticmethod
    def _validate_weights(overrides: dict[str, float] | None) -> dict[str, float] | None:
        """Validate weight overrides: all positive, sum ≈ 1.0, max 0.5."""
        if not overrides:
            return None
        if not all(isinstance(v, (int, float)) and v > 0 for v in overrides.values()):
            return None
        total = sum(overrides.values())
        if not (0.8 < total < 1.2):
            return None
        normed = {k: v / total for k, v in overrides.items()}
        if max(normed.values()) > 0.5:
            return None
        return normed

    def _get_weights(self) -> dict[str, float]:
        """Get rubric weights for the current phase, with advisor overrides."""
        weights = dict(PHASE_WEIGHTS.get(self._phase, DEFAULT_RUBRIC_WEIGHTS))

        if self._weight_overrides:
            weights.update(self._weight_overrides)
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}

        return weights

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

        # Weighted aggregate using phase-specific weights
        weights = self._get_weights()
        weighted = sum(final_scores[k] * weights[k] for k in SCORE_KEYS)

        result = {
            "segment_id": segment_data.get("segment_id", "unknown"),
            "phase": self._phase,
            "scores": final_scores,
            "weights_used": weights,
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
        # Phase-specific guidance
        phase_guidance = PHASE_PROMPTS.get(self._phase, {}).get("state", "")

        states = segment_data.get("states", [])
        summary = f"Segment {segment_data.get('segment_id', '?')}: "
        summary += f"{len(states)} frames, "
        summary += f"reward={segment_data.get('total_reward', 0):.1f}. "

        if states:
            first = states[0].get("state", {})
            last = states[-1].get("state", {})
            summary += f"Start room={first.get('room_id', '?')}, "
            summary += f"End room={last.get('room_id', '?')}. "

            rooms = set()
            dialogs = 0
            for s in states:
                st = s.get("state", {})
                rooms.add(st.get("room_id", 0))
                if st.get("dialog_active"):
                    dialogs += 1
            summary += f"Unique rooms visited: {len(rooms)}. "
            summary += f"Dialog interactions: {dialogs}. "

            active_group = last.get("active_group", 0)
            if active_group in (4, 5):
                summary += "Agent is inside a DUNGEON. "
            elif active_group == 2:
                summary += "Agent is at the MAKU TREE (key story location). "
            elif active_group == 3:
                summary += "Agent is INDOORS (cave, house, or shop). "
            else:
                summary += "Agent is in the OVERWORLD. "
            summary += f"Health: {last.get('health', '?')}/{last.get('max_health', '?')}. "

        context = ""
        if self._walkthrough:
            context = (
                "You are an expert on Zelda: Oracle of Seasons. Use this game guide "
                "to evaluate the agent's progress accurately:\n\n"
                f"{self._walkthrough}\n\n"
            )

        phase_section = ""
        if phase_guidance:
            phase_section = (
                f"CURRENT GAME PHASE: {self._phase}\n"
                f"PHASE GUIDANCE: {phase_guidance}\n\n"
            )

        return (
            f"{context}"
            f"{phase_section}"
            f"Rate this Zelda: Oracle of Seasons gameplay segment on: progress, dialog, efficiency. "
            f"progress: How much the agent advances toward the next quest milestone. "
            f"dialog: Quality of NPC dialog interactions (talking to the right NPCs, advancing story). "
            f"efficiency: Steps used productively vs wasted (backtracking, standing still). "
            f"Each score 0.0-1.0. {summary} "
            f'Output JSON: {{"scores": {{"progress": float, "dialog": float, "efficiency": float}}}}'
        )

    def _build_puzzle_prompt(self, segment_data: dict[str, Any]) -> str:
        """Build evaluation prompt for the puzzle judge model."""
        phase_guidance = PHASE_PROMPTS.get(self._phase, {}).get("puzzle", "")

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
            summary += f"Puzzle flag changes: {len(puzzle_flags)}. "

            last = states[-1].get("state", {})
            if last.get("active_group", 0) in (4, 5):
                summary += f"Currently in dungeon (floor {last.get('dungeon_floor', 0)}). "

        context = ""
        if self._walkthrough:
            context = (
                "You are an expert on Zelda: Oracle of Seasons puzzle mechanics. "
                "Use this guide for context:\n\n"
                f"{self._walkthrough}\n\n"
            )

        phase_section = ""
        if phase_guidance:
            phase_section = (
                f"CURRENT GAME PHASE: {self._phase}\n"
                f"PHASE GUIDANCE: {phase_guidance}\n\n"
            )

        return (
            f"{context}"
            f"{phase_section}"
            f"Rate this Zelda: Oracle of Seasons segment on puzzle-solving. "
            f"Puzzles include: pushing blocks, hitting switches, using season changes to open paths, "
            f"navigating dungeon rooms, finding keys and boss keys. "
            f"Score 0.0-1.0 (0=no puzzle progress, 1=solved complex puzzle). {summary} "
            f'Output JSON: {{"scores": {{"puzzle": float}}, "rationale": str}}'
        )

    def _build_vision_prompt(self, segment_data: dict[str, Any]) -> str:
        """Build evaluation prompt for the vision judge model."""
        phase_guidance = PHASE_PROMPTS.get(self._phase, {}).get("vision", "")

        states = segment_data.get("states", [])
        summary = f"Segment {segment_data.get('segment_id', '?')}: "
        summary += f"{len(states)} frames. "

        if states:
            rooms = set()
            for s in states:
                rooms.add(s.get("state", {}).get("room_id", 0))
            summary += f"Rooms visited: {len(rooms)}. "
            summary += f"Total reward: {segment_data.get('total_reward', 0):.1f}. "

            last = states[-1].get("state", {})
            if last.get("active_group", 0) in (4, 5):
                summary += "This is a DUNGEON screenshot. "
            elif last.get("active_group", 0) == 2:
                summary += "This is a MAKU TREE screenshot. "
            else:
                summary += "This is an OVERWORLD screenshot. "

        context = ""
        if self._walkthrough:
            context = (
                "You are evaluating a Zelda: Oracle of Seasons gameplay screenshot. "
                "Use this game guide for context:\n\n"
                f"{self._walkthrough}\n\n"
            )

        phase_section = ""
        if phase_guidance:
            phase_section = (
                f"CURRENT GAME PHASE: {self._phase}\n"
                f"PHASE GUIDANCE: {phase_guidance}\n\n"
            )

        return (
            f"{context}"
            f"{phase_section}"
            f"Analyze this Zelda: Oracle of Seasons screenshot for novelty and exploration. "
            f"Rate how novel/interesting the game state is on a 0.0-1.0 scale. "
            f"Consider: new areas explored (dungeons, caves, new overworld screens), "
            f"unique enemy encounters, items found, NPCs present, "
            f"environmental variety (different seasons change the landscape). {summary} "
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
