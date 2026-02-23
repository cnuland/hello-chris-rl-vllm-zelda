"""LLM Reward Advisor — phase-aware reward tuning between epochs.

After each training epoch, the advisor:
  1. Detects the agent's current game phase (pre_sword → dungeon)
  2. Reviews epoch stats (milestones, mean return, rooms explored)
  3. Samples episode segment summaries from MinIO
  4. Calls qwen25-32b with phase-specific context to produce:
     - Reward parameter multipliers (0.5-2.0)
     - Directional target (row/col for the agent to aim toward)
     - Structured directives (seek_room, trigger_action, avoid_region)
     - Rubric weight adjustments for the next eval pass

This creates a full coaching loop: the LLM observes agent behavior,
understands WHERE the agent is in the quest, and provides spatial
guidance and reward shaping to push it toward the next milestone.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any

from agent.evaluator.phase_detector import (
    PHASE_DESCRIPTIONS,
    PHASE_DIRECTIONAL_TARGETS,
    detect_phase,
)

logger = logging.getLogger(__name__)

# Base reward values — must match RewardWrapper defaults exactly.
# The advisor outputs multipliers on these, not absolute values.
BASE_REWARDS: dict[str, float] = {
    "key": 0.5,
    "death": -1.0,
    "health_loss": -0.005,
    "sword": 15.0,
    "dungeon": 100.0,
    "maku_tree": 100.0,
    "new_room": 50.0,
    "grid_exploration": 0.02,
    "dungeon_entry": 100.0,
    "maku_tree_visit": 100.0,
    "indoor_entry": 5.0,
    "dungeon_floor": 10.0,
    "dialog_bonus": 10.0,
    "dialog_advance": 25.0,
    "maku_dialog": 500.0,
    "gnarled_key": 500.0,
    "maku_seed": 1000.0,
    "gate_slash": 250.0,
    "maku_room": 100.0,
    "maku_stage": 300.0,
    "directional_bonus": 0.0,
    "snow_region": 0.0,
    "maku_loiter_penalty": 1.0,
    "area_boost_overworld": 1.0,
    "area_boost_subrosia": 1.5,
    "area_boost_maku": 3.0,
    "area_boost_indoors": 1.5,
    "area_boost_dungeon": 2.0,
}

# Absolute value clamps — prevent any single reward from becoming
# disproportionately large regardless of multiplier.
ABS_CLAMPS: dict[str, tuple[float, float]] = {
    "new_room": (10.0, 100.0),
    "dialog_bonus": (3.0, 20.0),
    "dialog_advance": (15.0, 100.0),
    "dungeon_entry": (30.0, 200.0),
    "maku_tree_visit": (30.0, 200.0),
    "indoor_entry": (1.0, 10.0),
    "maku_dialog": (150.0, 1000.0),
    "gnarled_key": (150.0, 1000.0),
    "maku_seed": (300.0, 2000.0),
    "gate_slash": (75.0, 500.0),
    "maku_room": (30.0, 200.0),
    "maku_stage": (100.0, 600.0),
    "sword": (5.0, 30.0),
    "dungeon": (30.0, 200.0),
    "maku_tree": (30.0, 200.0),
    "grid_exploration": (0.01, 0.1),
    "area_boost_overworld": (0.5, 2.0),
    "area_boost_subrosia": (0.5, 3.0),
    "area_boost_maku": (1.0, 8.0),
    "area_boost_indoors": (0.5, 3.0),
    "area_boost_dungeon": (1.0, 6.0),
    "snow_region": (0.0, 5000.0),
    "maku_loiter_penalty": (0.0, 5.0),
}

MULTIPLIER_BOUNDS = (0.5, 2.0)

# Valid rubric weight keys (must match ingest.py SCORE_KEYS)
RUBRIC_KEYS = {"progress", "dialog", "puzzle", "novelty", "efficiency"}

# Valid directive types
DIRECTIVE_TYPES = {"seek_room", "trigger_action", "avoid_region"}


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DirectionalTarget:
    """Target room coordinates for the directional bonus."""

    row: int
    col: int
    scale: float = 1.0
    rationale: str = ""


@dataclass
class Directive:
    """Structured action directive for the reward wrapper."""

    type: str  # "seek_room" | "trigger_action" | "avoid_region"
    target_group: int | None = None
    target_room: int | None = None
    bonus: float = 0.0
    condition: str = ""
    rationale: str = ""


@dataclass
class AdvisorOutput:
    """Complete advisor response with multipliers, targets, and directives."""

    multipliers: dict[str, float] = field(default_factory=dict)
    directional_target: DirectionalTarget | None = None
    directives: list[Directive] = field(default_factory=list)
    weight_adjustments: dict[str, float] | None = None
    phase_overrides: dict[str, dict] | None = None
    rationale: str = ""


# ---------------------------------------------------------------------------
# Advisor
# ---------------------------------------------------------------------------

class RewardAdvisor:
    """Analyzes agent behavior and produces phase-aware reward adjustments."""

    def __init__(
        self,
        llm_client: Any = None,
        walkthrough_path: str | None = None,
    ):
        self._llm = llm_client
        self._walkthrough = ""
        if walkthrough_path:
            try:
                with open(walkthrough_path) as f:
                    self._walkthrough = f.read()[:4000]
                logger.info(
                    "Loaded walkthrough (%d chars) for reward advisor",
                    len(self._walkthrough),
                )
            except Exception as e:
                logger.warning("Could not load walkthrough for advisor: %s", e)

    def advise(
        self,
        epoch_stats: dict[str, Any],
        segment_summaries: list[dict[str, Any]] | None = None,
    ) -> AdvisorOutput:
        """Call LLM to get reward adjustments based on epoch performance.

        Returns an AdvisorOutput with multipliers, directional target,
        directives, and weight adjustments.  If the LLM call fails,
        returns an AdvisorOutput with empty multipliers.
        """
        if self._llm is None:
            logger.warning("No LLM client — skipping reward advice")
            return AdvisorOutput()

        prompt = self._build_prompt(epoch_stats, segment_summaries or [])

        try:
            result = self._llm.advise(prompt)
        except Exception as e:
            logger.error("Reward advisor LLM call failed: %s", e)
            return AdvisorOutput()

        if "error" in result:
            logger.warning("Reward advisor returned error: %s", result)
            return AdvisorOutput()

        rationale = result.get("rationale", "")
        if rationale:
            logger.info("Reward advisor rationale: %s", rationale)

        # Parse and validate each component
        multipliers = self._parse_multipliers(result.get("multipliers", {}))
        directional_target = self._parse_directional_target(
            result.get("directional_target")
        )
        directives = self._parse_directives(result.get("directives", []))
        weight_adjustments = self._parse_weight_adjustments(
            result.get("weight_adjustments")
        )

        # Phase profile overrides — optional per-phase reward profile tweaks
        phase_overrides = result.get("phase_overrides")
        if isinstance(phase_overrides, dict):
            logger.info("Advisor phase overrides: %s", list(phase_overrides.keys()))
        else:
            phase_overrides = None

        output = AdvisorOutput(
            multipliers=multipliers,
            directional_target=directional_target,
            directives=directives,
            weight_adjustments=weight_adjustments,
            phase_overrides=phase_overrides,
            rationale=rationale,
        )

        logger.info(
            "Advisor output: %d multipliers, target=%s, %d directives, weights=%s",
            len(multipliers),
            f"({directional_target.row},{directional_target.col})" if directional_target else "none",
            len(directives),
            "yes" if weight_adjustments else "no",
        )
        return output

    def apply_multipliers(
        self,
        base_config: dict[str, float] | None,
        multipliers: dict[str, float],
    ) -> dict[str, float]:
        """Apply multipliers to base config with absolute clamping."""
        base = dict(base_config) if base_config else dict(BASE_REWARDS)

        for key, mult in multipliers.items():
            if key not in base:
                continue
            new_val = base[key] * mult

            if key in ABS_CLAMPS:
                lo, hi = ABS_CLAMPS[key]
                new_val = max(lo, min(hi, new_val))

            base[key] = new_val
            logger.info(
                "  %s: %.4f * %.2f = %.4f",
                key, BASE_REWARDS.get(key, 0), mult, new_val,
            )

        return base

    # ------------------------------------------------------------------
    # Parsing / validation helpers
    # ------------------------------------------------------------------

    def _parse_multipliers(self, raw: Any) -> dict[str, float]:
        """Validate and clamp multipliers from LLM response."""
        if not isinstance(raw, dict):
            return {}

        clamped = {}
        for key, val in raw.items():
            if key not in BASE_REWARDS:
                logger.warning("Unknown reward key from advisor: %s", key)
                continue
            try:
                val = float(val)
            except (TypeError, ValueError):
                logger.warning("Non-numeric multiplier for %s: %s", key, val)
                continue
            val = max(MULTIPLIER_BOUNDS[0], min(MULTIPLIER_BOUNDS[1], val))
            clamped[key] = val

        if clamped:
            logger.info(
                "Parsed %d multipliers: %s",
                len(clamped),
                {k: f"{v:.2f}" for k, v in clamped.items()},
            )
        return clamped

    def _parse_directional_target(self, raw: Any) -> DirectionalTarget | None:
        """Validate directional target from LLM response."""
        if not isinstance(raw, dict):
            return None

        try:
            row = int(raw["row"])
            col = int(raw["col"])
        except (KeyError, TypeError, ValueError):
            logger.warning("Invalid directional target (missing row/col): %s", raw)
            return None

        if not (0 <= row <= 15 and 0 <= col <= 15):
            logger.warning("Directional target out of range: row=%d, col=%d", row, col)
            return None

        scale = max(0.5, min(3.0, float(raw.get("scale", 1.0))))
        rationale = str(raw.get("rationale", ""))

        logger.info("Parsed directional target: row=%d, col=%d, scale=%.1f", row, col, scale)
        return DirectionalTarget(row=row, col=col, scale=scale, rationale=rationale)

    def _parse_directives(self, raw: Any) -> list[Directive]:
        """Validate directives from LLM response."""
        if not isinstance(raw, list):
            return []

        directives = []
        for item in raw[:10]:  # limit to 10 directives
            if not isinstance(item, dict):
                continue

            dtype = item.get("type", "")
            if dtype not in DIRECTIVE_TYPES:
                logger.warning("Unknown directive type: %s", dtype)
                continue

            bonus = max(-50.0, min(100.0, float(item.get("bonus", 0))))
            directives.append(Directive(
                type=dtype,
                target_group=item.get("target_group"),
                target_room=item.get("target_room"),
                bonus=bonus,
                condition=str(item.get("condition", "")),
                rationale=str(item.get("rationale", "")),
            ))

        if directives:
            logger.info("Parsed %d directives: %s", len(directives),
                        [d.type for d in directives])
        return directives

    def _parse_weight_adjustments(self, raw: Any) -> dict[str, float] | None:
        """Validate rubric weight adjustments from LLM response."""
        if not isinstance(raw, dict):
            return None

        weights = {}
        for key, val in raw.items():
            if key not in RUBRIC_KEYS:
                continue
            try:
                weights[key] = float(val)
            except (TypeError, ValueError):
                continue

        if not weights:
            return None

        # Validate: all positive, sum ≈ 1.0, no single weight > 0.5
        if any(v <= 0 for v in weights.values()):
            logger.warning("Weight adjustments contain non-positive values: %s", weights)
            return None

        total = sum(weights.values())
        if not (0.8 < total < 1.2):
            logger.warning("Weight adjustments don't sum to ~1.0 (sum=%.2f): %s", total, weights)
            return None

        # Normalize to exactly 1.0
        normed = {k: v / total for k, v in weights.items()}
        if max(normed.values()) > 0.5:
            logger.warning("Weight adjustment has single weight > 0.5: %s", normed)
            return None

        logger.info("Parsed weight adjustments: %s",
                    {k: f"{v:.2f}" for k, v in normed.items()})
        return normed

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        epoch_stats: dict[str, Any],
        segment_summaries: list[dict[str, Any]],
    ) -> str:
        """Build the advisor prompt with phase context, stats, and instructions."""
        parts = []

        # Detect current phase
        milestones = epoch_stats.get("milestones", {})
        phase = detect_phase(milestones)
        phase_desc = PHASE_DESCRIPTIONS.get(phase, "")
        default_target = PHASE_DIRECTIONAL_TARGETS.get(phase)

        # Phase context (most important for the LLM to understand)
        parts.append(f"CURRENT GAME PHASE: {phase}")
        parts.append(f"{phase_desc}")
        if default_target:
            parts.append(
                f"Default directional target for this phase: "
                f"row={default_target[0]}, col={default_target[1]} "
                f"(in the 16x16 room grid, where room_id = row*16 + col)"
            )
        parts.append("")

        # Walkthrough context
        if self._walkthrough:
            parts.append(
                "GAME WALKTHROUGH (Zelda: Oracle of Seasons):\n"
                f"{self._walkthrough}\n"
            )

        # Current reward parameters
        parts.append("CURRENT REWARD PARAMETERS (base values):")
        for key, val in BASE_REWARDS.items():
            parts.append(f"  {key}: {val}")
        parts.append("")

        # Epoch stats
        parts.append("EPOCH TRAINING STATS:")
        parts.append(f"  Epoch: {epoch_stats.get('epoch', '?')}")
        reward_mean = epoch_stats.get("reward_mean", "?")
        reward_max = epoch_stats.get("reward_max", "?")
        parts.append(f"  Mean return: {reward_mean}")
        parts.append(f"  Max return: {reward_max}")
        episodes = epoch_stats.get(
            "episodes_completed", epoch_stats.get("episodes", "?")
        )
        parts.append(f"  Total episodes: {episodes}")

        if milestones:
            n_eps = max(int(episodes) if isinstance(episodes, (int, float)) else 1, 1)
            parts.append("  MILESTONES:")
            for key, label in [
                ("total_got_sword", "Got Sword"),
                ("total_visited_maku_tree", "Visited Maku Tree"),
                ("total_gate_slashed", "Slashed Gate"),
                ("total_maku_stage", "Maku Tree Stage Up"),
                ("total_maku_dialog", "Maku Tree Dialog"),
                ("total_gnarled_key", "Got Gnarled Key"),
                ("total_entered_snow_region", "Entered Snow Region"),
                ("total_entered_dungeon", "Entered Dungeon"),
                ("total_maku_seed", "Got Maku Seed"),
            ]:
                count = milestones.get(key, 0)
                pct = 100.0 * count / n_eps if n_eps > 0 else 0
                parts.append(f"    {label}: {count}/{n_eps} ({pct:.0f}%)")
            parts.append(f"    Max Rooms Explored: {milestones.get('max_rooms', 0)}")
            parts.append(f"    Max Tiles Explored: {milestones.get('max_tiles', 0)}")
            parts.append(f"    Max Maku Rooms: {milestones.get('max_maku_rooms', 0)}")
        parts.append("")

        # Segment summaries
        if segment_summaries:
            parts.append(f"SAMPLE EPISODE SEGMENTS ({len(segment_summaries)} segments):")
            for i, seg in enumerate(segment_summaries[:5]):
                parts.append(f"  Segment {i+1}:")
                parts.append(f"    Rooms visited: {seg.get('rooms', '?')}")
                parts.append(f"    Dialog count: {seg.get('dialog_count', 0)}")
                parts.append(f"    Area: {seg.get('area', 'overworld')}")
                parts.append(f"    Total reward: {seg.get('total_reward', 0):.1f}")
                scores = seg.get("scores", {})
                if scores:
                    parts.append(f"    Judge scores: {scores}")
                # Vision rationale — qualitative description from the vision
                # model's multi-frame analysis of this segment
                vis_rationale = seg.get("vision_rationale", "")
                if vis_rationale:
                    parts.append(f"    Visual observation: {vis_rationale}")
            parts.append("")

        # Instructions — phase-specific and expanded output schema
        parts.append(
            "TASK: You are a reward engineering advisor for a reinforcement learning "
            "agent playing Zelda: Oracle of Seasons. Based on the current phase, "
            "training stats, walkthrough, and segment analysis above, provide "
            "comprehensive guidance for the next epoch.\n\n"
            "The agent's quest progression is:\n"
            "1. Get the sword from the Hero's Cave\n"
            "2. Head EAST then NORTH to find the Maku Tree\n"
            "3. Slash the gate, pop the bubble, talk to the Maku Tree\n"
            "4. Get the Gnarled Key quest from dialog\n"
            "5. Head WEST to find Dungeon 1 (Gnarled Root)\n"
            "6. Complete Dungeon 1 to get the first Essence\n\n"
            "ROOM GRID: 16x16, room_id = row*16 + col. Agent starts at row=13, col=9.\n"
            "AREA GROUPS: 0=overworld, 1=subrosia, 2=maku tree, 3=indoors, 4-5=dungeons\n\n"
            "Output a JSON object with ALL of these fields:\n\n"
            '1. "multipliers": dict mapping parameter names to float multipliers (0.5-2.0)\n'
            '   1.0 = keep current, >1.0 = increase, <1.0 = decrease\n'
            '   Only include parameters you want to change\n\n'
            '2. "directional_target": {"row": int, "col": int, "scale": float, "rationale": str}\n'
            '   Which room in the 16x16 grid should the agent aim toward?\n'
            '   scale multiplies the directional_bonus (1.0 = normal, 2.0 = double)\n\n'
            '3. "directives": list of action directives:\n'
            '   - {"type": "seek_room", "target_group": int, "bonus": float, "rationale": str}\n'
            '     One-time bonus when the agent enters a specific area group\n'
            '   - {"type": "trigger_action", "condition": str, "bonus": float, "rationale": str}\n'
            '     One-time bonus for performing an action (e.g. "dialog_in_group_2")\n'
            '   - {"type": "avoid_region", "target_group": int, "bonus": float, "rationale": str}\n'
            '     Per-step penalty (negative bonus) for being in an undesirable area\n\n'
            '4. "weight_adjustments": {"progress": float, "dialog": float, "puzzle": float, '
            '"novelty": float, "efficiency": float}\n'
            '   Rubric weights for the next judge evaluation pass (must sum to 1.0, max 0.5 per weight)\n\n'
            '5. "rationale": brief explanation of your strategy\n\n'
            f"Focus on the agent's CURRENT PHASE ({phase}). "
            "What is the biggest bottleneck preventing progress to the next phase?"
        )

        # Phase-specific supplemental notes
        if phase == "pre_dungeon":
            parts.append(
                "\n\nPHASE-SPECIFIC NOTES (pre_dungeon):\n"
                "The agent has the Gnarled Key. The reward wrapper now SUPPRESSES "
                "all Maku Tree rewards (visit bonus, dialog advance, room discovery, "
                "stage change) and applies a per-step penalty for loitering in group 2. "
                "A directional bonus toward Dungeon 1 at (row=10, col=4) is ACTIVE.\n"
                "Focus your advice on:\n"
                "- Increasing directional_target scale if the agent is not moving west\n"
                "- Adding avoid_region directive for group 2 (Maku Tree) to reinforce\n"
                "- Increasing new_room and grid_exploration multipliers to incentivize "
                "overworld exploration\n"
                "- The path is: EXIT Maku Tree → go WEST → then NORTH to snowy region\n"
                "- Dungeon 1 entrance is at approximately (row=10, col=4) on the overworld"
            )

        return "\n".join(parts)
