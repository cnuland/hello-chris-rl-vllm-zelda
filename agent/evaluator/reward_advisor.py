"""LLM Reward Advisor — dynamically tunes reward weights between epochs.

After each training epoch, the advisor:
  1. Reviews epoch stats (milestones, mean return, rooms explored)
  2. Samples episode segment summaries from MinIO
  3. Calls qwen25-32b with walkthrough context to produce reward multipliers
  4. Applies bounded multipliers (0.5-2.0) to the base RewardConfig

This creates a feedback loop: the LLM observes agent behavior and adjusts
rewards to guide it toward walkthrough-informed objectives (e.g., boost
directional bonus if the agent never goes east toward the Maku Tree).
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Base reward values — must match RewardWrapper defaults exactly.
# The advisor outputs multipliers on these, not absolute values.
BASE_REWARDS: dict[str, float] = {
    "rupee": 0.01,
    "key": 0.5,
    "death": -1.0,
    "health_loss": -0.005,
    "time_penalty": 0.0,
    "sword": 15.0,
    "dungeon": 15.0,
    "maku_tree": 15.0,
    "new_room": 10.0,
    "grid_exploration": 0.005,
    "exit_seeking": 0.5,
    "dungeon_entry": 15.0,
    "maku_tree_visit": 15.0,
    "indoor_entry": 5.0,
    "dungeon_floor": 2.0,
    "dialog_bonus": 3.0,
    "maku_dialog": 30.0,
    "gnarled_key": 30.0,
    "distance_bonus": 5.0,
    "directional_bonus": 20.0,
    "directional_decay": 0.999,
    "coord_decay_factor": 0.9998,
    "coord_decay_floor": 0.60,
    "area_boost_overworld": 1.0,
    "area_boost_subrosia": 1.5,
    "area_boost_maku": 3.0,
    "area_boost_indoors": 1.5,
    "area_boost_dungeon": 2.0,
}

# Absolute value clamps — prevent any single reward from becoming
# disproportionately large regardless of multiplier.
ABS_CLAMPS: dict[str, tuple[float, float]] = {
    "new_room": (3.0, 20.0),
    "directional_bonus": (10.0, 40.0),
    "distance_bonus": (2.0, 10.0),
    "dialog_bonus": (1.0, 6.0),
    "dungeon_entry": (5.0, 30.0),
    "maku_tree_visit": (5.0, 30.0),
    "indoor_entry": (1.0, 10.0),
    "maku_dialog": (10.0, 60.0),
    "gnarled_key": (10.0, 60.0),
    "sword": (5.0, 30.0),
    "dungeon": (5.0, 30.0),
    "maku_tree": (5.0, 30.0),
    "grid_exploration": (0.001, 0.02),
    "exit_seeking": (0.1, 2.0),
    "directional_decay": (0.995, 0.9999),
    "coord_decay_factor": (0.999, 0.99999),
    "coord_decay_floor": (0.30, 0.90),
    "area_boost_overworld": (0.5, 2.0),
    "area_boost_subrosia": (0.5, 3.0),
    "area_boost_maku": (1.0, 5.0),
    "area_boost_indoors": (0.5, 3.0),
    "area_boost_dungeon": (1.0, 4.0),
}

MULTIPLIER_BOUNDS = (0.5, 2.0)


class RewardAdvisor:
    """Analyzes agent behavior and produces reward weight adjustments."""

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
    ) -> dict[str, float]:
        """Call LLM to get reward multipliers based on epoch performance.

        Args:
            epoch_stats: Training metrics (mean_return, milestones, rooms, etc.)
            segment_summaries: Optional list of segment summary dicts from eval.

        Returns:
            Dict mapping reward parameter names to multiplier values (0.5-2.0).
            Returns empty dict if LLM call fails.
        """
        if self._llm is None:
            logger.warning("No LLM client — skipping reward advice")
            return {}

        prompt = self._build_prompt(epoch_stats, segment_summaries or [])

        try:
            result = self._llm.advise(prompt)
        except Exception as e:
            logger.error("Reward advisor LLM call failed: %s", e)
            return {}

        if "error" in result:
            logger.warning("Reward advisor returned error: %s", result)
            return {}

        multipliers = result.get("multipliers", {})
        if not isinstance(multipliers, dict):
            logger.warning("Invalid multipliers format: %s", type(multipliers))
            return {}

        rationale = result.get("rationale", "")
        if rationale:
            logger.info("Reward advisor rationale: %s", rationale)

        # Validate and clamp multipliers
        clamped = {}
        for key, val in multipliers.items():
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

        logger.info(
            "Reward advisor produced %d multipliers: %s",
            len(clamped),
            {k: f"{v:.2f}" for k, v in clamped.items()},
        )
        return clamped

    def apply_multipliers(
        self,
        base_config: dict[str, float] | None,
        multipliers: dict[str, float],
    ) -> dict[str, float]:
        """Apply multipliers to base config with absolute clamping.

        Args:
            base_config: Base reward config dict. If None, uses BASE_REWARDS.
            multipliers: Multiplier dict from advise().

        Returns:
            Updated reward config dict with multiplied and clamped values.
        """
        base = dict(base_config) if base_config else dict(BASE_REWARDS)

        for key, mult in multipliers.items():
            if key not in base:
                continue
            new_val = base[key] * mult

            # Apply absolute clamps if defined
            if key in ABS_CLAMPS:
                lo, hi = ABS_CLAMPS[key]
                new_val = max(lo, min(hi, new_val))

            base[key] = new_val
            logger.info(
                "  %s: %.4f * %.2f = %.4f",
                key, BASE_REWARDS.get(key, 0), mult, new_val,
            )

        return base

    def _build_prompt(
        self,
        epoch_stats: dict[str, Any],
        segment_summaries: list[dict[str, Any]],
    ) -> str:
        """Build the advisor prompt with context, stats, and instructions."""
        parts = []

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
        parts.append(f"  Mean return: {epoch_stats.get('reward_mean', '?'):.1f}")
        parts.append(f"  Max return: {epoch_stats.get('reward_max', '?'):.1f}")
        parts.append(f"  Total episodes: {epoch_stats.get('episodes', '?')}")

        milestones = epoch_stats.get("milestones", {})
        if milestones:
            parts.append("  MILESTONES:")
            parts.append(f"    Got Sword: {milestones.get('got_sword_pct', 0):.1f}%")
            parts.append(f"    Entered Dungeon: {milestones.get('entered_dungeon_pct', 0):.1f}%")
            parts.append(f"    Visited Maku Tree: {milestones.get('visited_maku_tree_pct', 0):.1f}%")
            parts.append(f"    Maku Tree Dialog: {milestones.get('maku_dialog_pct', 0):.1f}%")
            parts.append(f"    Got Gnarled Key: {milestones.get('gnarled_key_pct', 0):.1f}%")
            parts.append(f"    Avg Rooms Explored: {milestones.get('avg_rooms', 0):.1f}")
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
            parts.append("")

        # Instructions
        parts.append(
            "TASK: You are a reward engineering advisor for a reinforcement learning "
            "agent playing Zelda: Oracle of Seasons. Based on the training stats, "
            "walkthrough, and segment analysis above, suggest multipliers for each "
            "reward parameter to guide the agent toward the next game milestone.\n\n"
            "The agent's current goal progression should be:\n"
            "1. Get the sword from the Hero's Cave\n"
            "2. Head EAST then NORTH to find the Maku Tree\n"
            "3. Talk to the Maku Tree to get the Gnarled Key quest\n"
            "4. Head WEST to find Dungeon 1 (Gnarled Root)\n"
            "5. Complete Dungeon 1 to get the first Essence\n\n"
            "Output a JSON object with:\n"
            '- "multipliers": dict mapping parameter names to float multipliers (0.5-2.0)\n'
            '  - 1.0 = keep current value, >1.0 = increase, <1.0 = decrease\n'
            '  - Only include parameters you want to change (omit ones that should stay at 1.0)\n'
            '- "rationale": brief explanation of why you chose these multipliers\n\n'
            "Focus on the BIGGEST bottleneck. If the agent never reaches the Maku Tree, "
            "boost directional_bonus and dialog_bonus. If it gets the sword but doesn't "
            "explore, boost new_room and distance_bonus. If it explores but ignores NPCs, "
            "boost dialog_bonus and maku_dialog."
        )

        return "\n".join(parts)
