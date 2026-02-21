"""Evaluator ingest: batch segments through phase-aware llm-d judges.

Pipeline:
  1. Detect the agent's current game phase from epoch milestones.
  2. Read segments from MinIO.
  3. Fan out to vision/state/puzzle judges via llm-d gateway with
     phase-specific prompts and dynamic rubric weights.
  4. Self-consistency M=3 (three passes), majority vote.
  5. Write scores.jsonl back to MinIO.

Judge model mapping (vision-primary architecture):
  - vision (qwen25-vl-32b): PRIMARY — multi-frame analysis scoring
    progress, novelty, efficiency, dialog (~70% effective weight)
  - state (qwen25-7b):      cross-validator for progress, dialog, efficiency
  - puzzle (qwen25-32b):    puzzle scoring (text-only, no vision blend)
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
# most at each stage of the game.  These are the DIMENSION importance
# weights; the vision/text split within each dimension is handled by
# DIMENSION_BLEND separately.
#
# Key design decisions:
#   - pre_sword: novelty + exploration (0.30) to encourage finding Hero's Cave
#   - pre_maku: progress (0.35) + efficiency (0.20) for directional travel
#   - maku_interaction: dialog (0.40) — THE critical Maku Tree blocker
#   - dungeon: puzzle (0.30) + efficiency (0.25) for puzzle-solving gameplay
PHASE_WEIGHTS: dict[str, dict[str, float]] = {
    "pre_sword": {
        "progress": 0.30,
        "novelty": 0.30,
        "dialog": 0.10,
        "efficiency": 0.20,
        "puzzle": 0.10,
    },
    "pre_maku": {
        "progress": 0.35,
        "novelty": 0.25,
        "dialog": 0.10,
        "efficiency": 0.20,
        "puzzle": 0.10,
    },
    "maku_interaction": {
        "progress": 0.20,
        "novelty": 0.10,
        "dialog": 0.40,
        "efficiency": 0.15,
        "puzzle": 0.15,
    },
    "pre_dungeon": {
        "progress": 0.35,
        "novelty": 0.25,
        "dialog": 0.05,
        "efficiency": 0.25,
        "puzzle": 0.10,
    },
    "dungeon": {
        "progress": 0.25,
        "novelty": 0.15,
        "dialog": 0.05,
        "efficiency": 0.25,
        "puzzle": 0.30,
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
            "EARLY GAME — the agent needs to explore and find the Hero's Cave. "
            "PROGRESS: Score highly if frames show the agent moving through "
            "different screens, especially entering indoor/cave areas. "
            "NOVELTY: Score highly if frames show visually distinct areas — "
            "caves, indoor tile sets, forest clearings the agent hasn't seen. "
            "EFFICIENCY: Score highly if the agent moves purposefully between "
            "frames (positions change meaningfully). Score low if Link stands "
            "still or oscillates in the same spot across frames. "
            "DIALOG: Score only if a dialog box is visible on screen."
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
            "OVERWORLD TRAVEL — the agent has the sword and must reach the "
            "Maku Tree (northeast). "
            "PROGRESS: Score highly if frames show the agent traversing east "
            "and north through overworld screens. New screens appearing across "
            "the frame sequence = strong progress. Score low if all frames "
            "show the same screen. "
            "NOVELTY: Score highly for new overworld areas, especially eastern "
            "and northern screens with village buildings, trees, water. "
            "EFFICIENCY: Score highly if frame-to-frame movement shows clear "
            "directional intent (heading east/north). Score low for wandering "
            "back and forth or circling. "
            "DIALOG: Score if any frame shows a text dialog box on screen."
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
            "MAKU TREE AREA — dialog interaction is critical here. "
            "PROGRESS: Score highly if frames show the grove area with the "
            "large Maku Tree NPC. Score low if frames show overworld outside "
            "the grove. "
            "NOVELTY: Score for exploring new rooms within the grove area. "
            "EFFICIENCY: Score highly if the agent stays in the grove and "
            "approaches the tree NPC. Score low if leaving and re-entering. "
            "DIALOG: Score VERY HIGHLY (0.8-1.0) if ANY frame shows a dialog "
            "box on screen — this is THE critical objective at this phase."
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
            "HEADING TO DUNGEON — the agent needs to travel west. "
            "PROGRESS: Score highly if frames show westward overworld travel "
            "or the distinctive dungeon cave entrance. Score very highly if "
            "any frame shows a dungeon interior tile set. "
            "NOVELTY: Score highly for new western areas and especially "
            "dungeon entrances or interiors. "
            "EFFICIENCY: Score highly for clear westward movement across "
            "frames. Score low for meandering or retracing. "
            "DIALOG: Score only if dialog boxes appear on screen."
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
            "DUNGEON INTERIOR — puzzle-solving and exploration are key. "
            "PROGRESS: Score highly if frames show different dungeon rooms, "
            "defeated enemies, opened chests, keys collected. "
            "NOVELTY: Score highly for new dungeon rooms with distinct tile "
            "patterns, puzzle elements (blocks, switches, locked doors). "
            "EFFICIENCY: Score highly if the agent moves through rooms "
            "purposefully. Score low if stuck in one room across all frames. "
            "DIALOG: Score if dialog boxes appear (item descriptions, etc.)."
        ),
    },
}

SCORE_KEYS = list(DEFAULT_RUBRIC_WEIGHTS.keys())

# Number of frames to sample from each segment for multi-frame vision analysis.
NUM_VISION_FRAMES = 7

# Per-dimension blending of vision + text judge scores.
# Vision is the primary scorer for progress, novelty, dialog, efficiency.
# Puzzle remains text-only (requires game-mechanic knowledge, not pixel inspection).
DIMENSION_BLEND: dict[str, dict[str, float]] = {
    "progress":   {"vision": 0.75, "text": 0.25},
    "novelty":    {"vision": 0.85, "text": 0.15},
    "dialog":     {"vision": 0.60, "text": 0.40},
    "efficiency": {"vision": 0.80, "text": 0.20},
    "puzzle":     {"vision": 0.00, "text": 1.00},
}


def _select_segment_frames(frame_keys: list[str], num_frames: int = NUM_VISION_FRAMES) -> list[str]:
    """Select frames spanning a segment: first, last, + evenly spaced middle.

    Provides temporal coverage of the gameplay segment for multi-frame
    vision analysis without sending every saved PNG.
    """
    n = len(frame_keys)
    if n <= num_frames:
        return list(frame_keys)

    # Always include first and last
    indices = {0, n - 1}

    # Fill remaining slots evenly from the middle
    remaining = num_frames - 2
    step = (n - 2) / (remaining + 1)
    for i in range(1, remaining + 1):
        indices.add(int(round(i * step)))

    return [frame_keys[i] for i in sorted(indices)[:num_frames]]


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

        Uses vision-primary architecture:
          - Vision judge (qwen25-vl-32b): PRIMARY — multi-frame scoring of
            progress, novelty, efficiency, dialog (~70% effective weight)
          - State judge (qwen25-7b):  cross-validator for progress, dialog, efficiency
          - Puzzle judge (qwen25-32b): puzzle scoring (text-only)

        Per-dimension scores are blended via DIMENSION_BLEND, then weighted
        by phase-specific PHASE_WEIGHTS for the final aggregate score.
        """
        all_scores: list[dict[str, Any]] = []

        for trial in range(self._m):
            scores = self._judge_once(segment_data, trial)
            all_scores.append(scores)

        # Majority vote: median per dimension
        final_scores: dict[str, float] = {}
        for key in SCORE_KEYS:
            values = [s.get(key, 0.0) for s in all_scores]
            final_scores[key] = statistics.median(values)

        # Collect vision rationale from the first trial that has one
        vision_rationale = ""
        for s in all_scores:
            if s.get("_vision_rationale"):
                vision_rationale = s["_vision_rationale"]
                break

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
            "vision_rationale": vision_rationale,
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
            # Prefer multi-frame list; fall back to single frame
            frames = segment_data.get("frames_b64") or []
            if not frames:
                single = segment_data.get("frame_b64")
                if single:
                    frames = [single]
                else:
                    return {}

            vision_prompt = self._build_vision_prompt(segment_data)

            # Build a compact game-state summary from first + last state
            game_state: dict[str, Any] = {}
            states = segment_data.get("states", [])
            if states:
                first_s = states[0].get("state", {})
                last_s = states[-1].get("state", {})
                game_state = {
                    "start_room": first_s.get("room_id"),
                    "end_room": last_s.get("room_id"),
                    "active_group": last_s.get("active_group"),
                    "has_sword": last_s.get("has_sword", False),
                    "total_reward": segment_data.get("total_reward", 0),
                }

            result = self._llm.vision(frames, game_state, prompt=vision_prompt)

            partial: dict[str, Any] = {}
            vision_dims = ["progress", "novelty", "efficiency", "dialog"]
            if "scores" in result:
                for dim in vision_dims:
                    val = result["scores"].get(dim)
                    if val is not None:
                        partial[f"_vision_{dim}"] = float(val)
            else:
                # Flat keys fallback
                for dim in vision_dims:
                    val = result.get(dim)
                    if val is not None:
                        partial[f"_vision_{dim}"] = float(val)

            # Capture rationale for forwarding to reward advisor
            if result.get("rationale"):
                partial["_vision_rationale"] = str(result["rationale"])

            return partial

        # Run all 3 judges in parallel
        raw: dict[str, Any] = {}
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
                    raw.update(partial)
                except Exception as e:
                    logger.warning("%s judge trial %d failed: %s", judge_name, trial, e)

        # ---- Dimension blending: vision + text per DIMENSION_BLEND ----
        for dim in SCORE_KEYS:
            blend = DIMENSION_BLEND.get(dim, {"vision": 0.0, "text": 1.0})
            vision_val = raw.get(f"_vision_{dim}")
            text_val = raw.get(dim)  # state or puzzle judge

            if vision_val is not None and text_val is not None:
                # Both judges provided a score — blend them
                scores[dim] = (
                    blend["vision"] * vision_val + blend["text"] * text_val
                )
            elif vision_val is not None:
                # Only vision available — use it fully
                scores[dim] = vision_val
            elif text_val is not None:
                # Only text available — use it fully
                scores[dim] = text_val
            else:
                # Neither judge scored this dimension
                scores[dim] = 0.5

        # Carry vision rationale through for the advisor
        if raw.get("_vision_rationale"):
            scores["_vision_rationale"] = raw["_vision_rationale"]

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
        """Build multi-dimension evaluation prompt for the vision judge.

        The vision judge is the PRIMARY evaluator — it sees 7 frames spanning
        the gameplay segment and scores progress, novelty, efficiency, and
        dialog from direct pixel observation.  It also provides a natural-
        language rationale that gets forwarded to the reward advisor.
        """
        phase_guidance = PHASE_PROMPTS.get(self._phase, {}).get("vision", "")

        states = segment_data.get("states", [])
        num_frames = len(segment_data.get("frames_b64", []))
        summary = f"Segment {segment_data.get('segment_id', '?')}: "
        summary += f"{len(states)} game steps, {num_frames} frames shown. "

        if states:
            rooms = set()
            for s in states:
                rooms.add(s.get("state", {}).get("room_id", 0))
            summary += f"Rooms visited: {len(rooms)}. "
            summary += f"Total reward: {segment_data.get('total_reward', 0):.1f}. "

            last = states[-1].get("state", {})
            if last.get("active_group", 0) in (4, 5):
                summary += "Location: DUNGEON interior. "
            elif last.get("active_group", 0) == 2:
                summary += "Location: MAKU TREE grove. "
            else:
                summary += "Location: OVERWORLD. "

        context = ""
        if self._walkthrough:
            context = (
                "You are the PRIMARY evaluator for a Zelda: Oracle of Seasons "
                "RL agent. You are analyzing a sequence of gameplay frames. "
                "Use this game guide for context:\n\n"
                f"{self._walkthrough}\n\n"
            )

        phase_section = ""
        if phase_guidance:
            phase_section = (
                f"CURRENT GAME PHASE: {self._phase}\n"
                f"PHASE GUIDANCE:\n{phase_guidance}\n\n"
            )

        return (
            f"{context}"
            f"{phase_section}"
            f"You are analyzing {num_frames} frames from a Zelda: Oracle of Seasons "
            f"gameplay segment, shown in chronological order. "
            f"The frames are labeled [Frame 1/{num_frames}] through "
            f"[Frame {num_frames}/{num_frames}]. "
            f"The game state JSON is also provided for context.\n\n"
            f"{summary}\n\n"
            f"Score this segment on FOUR dimensions (each 0.0 to 1.0):\n"
            f"  progress  — Did the player advance? New rooms, items, enemies "
            f"defeated, or meaningful position changes across frames?\n"
            f"  novelty   — Are the frames showing new, visually distinct areas? "
            f"Or revisiting the same places repeatedly?\n"
            f"  efficiency — Is movement purposeful between frames? Heading in a "
            f"clear direction vs wandering/backtracking/standing still?\n"
            f"  dialog    — Are any dialog boxes visible on screen in any frame? "
            f"Score 0.0 if no dialog, higher if dialog boxes appear.\n\n"
            f"Also provide a 2-3 sentence 'rationale' describing what you observe "
            f"across the frame sequence — where is Link, what is he doing, "
            f"is he making progress?\n\n"
            f"Output ONLY valid JSON:\n"
            f'{{"scores": {{"progress": float, "novelty": float, '
            f'"efficiency": float, "dialog": float}}, '
            f'"rationale": "string describing observations across frames"}}'
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

                frame_keys = sorted(
                    k for k in self._s3.list_keys(self._bucket, prefix=f"{key}/frames/")
                    if k.endswith(".png")
                )
                if frame_keys:
                    selected = _select_segment_frames(frame_keys)
                    frames_b64 = []
                    for fk in selected:
                        png_bytes = self._s3.download_bytes(self._bucket, fk)
                        frames_b64.append(base64.b64encode(png_bytes).decode("ascii"))
                    manifest["frames_b64"] = frames_b64
                    # Keep single frame for backward compat
                    manifest["frame_b64"] = frames_b64[len(frames_b64) // 2]
                    logger.info(
                        "Loaded %d frames for vision judge (%d available)",
                        len(frames_b64), len(frame_keys),
                    )
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
