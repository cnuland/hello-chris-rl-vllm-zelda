"""Phase detection for RLAIF feedback loop.

Determines the agent's current game phase from epoch milestone statistics.
Both the evaluator (ingest.py) and reward advisor (reward_advisor.py) use
this to generate phase-appropriate judge prompts, rubric weights, and
directional targets.

Phases follow the Oracle of Seasons quest progression:
  pre_sword → pre_maku → maku_interaction → pre_dungeon → dungeon
"""

from __future__ import annotations

import os

PHASES = ["pre_sword", "pre_maku", "maku_interaction", "pre_dungeon", "dungeon"]

# Human-readable descriptions included in LLM judge/advisor prompts so
# the models understand what the agent is currently trying to accomplish.
PHASE_DESCRIPTIONS: dict[str, str] = {
    "pre_sword": (
        "The agent is in the EARLY GAME. It needs to find and enter the "
        "Hero's Cave (an indoor area south of the starting village) to "
        "obtain the sword. Without the sword, the agent cannot slash "
        "bushes, hit switches, or interact with the Maku Tree gate. "
        "Good behavior: exploring indoor areas (active_group 3), finding "
        "cave entrances, picking up items."
    ),
    "pre_maku": (
        "The agent HAS the sword and needs to travel EAST then NORTH to "
        "reach the Maku Tree (active_group 2). The Maku Tree is the "
        "central quest hub -- talking to it gives the Gnarled Key quest "
        "needed to enter Dungeon 1. Good behavior: increasing room column "
        "(room_id %% 16 should increase), discovering new overworld rooms, "
        "moving toward the northeast quadrant (approximately row 5, col 12)."
    ),
    "maku_interaction": (
        "The agent has REACHED the Maku Tree area (active_group 2) but has "
        "NOT yet completed the dialog to receive the Gnarled Key quest. "
        "This is the CRITICAL BOTTLENECK. The agent must: (1) slash the "
        "gate with the sword, (2) navigate the grove, (3) pop the bubble "
        "around the Maku Tree, (4) advance dialog to receive the quest. "
        "DIALOG IS THE MOST IMPORTANT ACTION at this stage. Good behavior: "
        "staying in group 2, triggering dialog, interacting with objects."
    ),
    "pre_dungeon": (
        "The agent has the Gnarled Key quest from the Maku Tree and needs "
        "to travel WEST to find Dungeon 1 (Gnarled Root). The dungeon "
        "entrance is in the western part of the overworld. Good behavior: "
        "decreasing room column, discovering new rooms to the west, "
        "entering active_group 4 or 5 (dungeon areas)."
    ),
    "dungeon": (
        "The agent is IN a dungeon (active_group 4 or 5). It needs to "
        "navigate dungeon rooms, solve puzzles, find keys, defeat enemies, "
        "and reach the boss. Good behavior: advancing floors, collecting "
        "keys, hitting switches, pushing blocks, defeating mini-bosses."
    ),
}

# Default directional targets for each phase -- the room grid coordinates
# (row, col in the 16x16 room grid) the agent should be guided toward.
# These defaults can be overridden by the reward advisor each epoch.
PHASE_DIRECTIONAL_TARGETS: dict[str, tuple[int, int] | None] = {
    "pre_sword": (14, 8),     # Hero's Cave area (south of village)
    "pre_maku": (5, 12),      # Maku Tree path (northeast)
    "maku_interaction": (5, 12),  # Stay at Maku Tree
    "pre_dungeon": (10, 4),   # Dungeon 1 entrance (west)
    "dungeon": None,          # No directional target inside dungeons
}


def detect_phase(milestones: dict) -> str:
    """Detect the agent's current game phase from epoch milestone counts.

    Uses conservative thresholds: 60% for easy milestones (sword, maku visit)
    and 30% for hard milestones (maku dialog, dungeon entry).  The lower
    threshold for hard milestones prevents the system from getting stuck
    in a phase that the agent will never reach 60% on.

    Args:
        milestones: Dict from epoch metadata with keys like total_got_sword,
                    total_visited_maku_tree, total_maku_dialog,
                    total_entered_dungeon, and total_episodes (or
                    episodes_completed).

    Returns:
        Phase string: one of PHASES.
    """
    # Allow manual override for testing/debugging
    override = os.getenv("RLAIF_PHASE", "")
    if override in PHASES:
        return override

    # Extract episode count for percentage calculation
    total_eps = max(
        milestones.get("episodes_completed",
                       milestones.get("total_episodes", 0)),
        1,
    )

    got_sword_pct = 100.0 * milestones.get("total_got_sword", 0) / total_eps
    visited_maku_pct = 100.0 * milestones.get("total_visited_maku_tree", 0) / total_eps
    maku_dialog_pct = 100.0 * milestones.get("total_maku_dialog", 0) / total_eps
    entered_dungeon_pct = 100.0 * milestones.get("total_entered_dungeon", 0) / total_eps

    if got_sword_pct < 60:
        return "pre_sword"
    if visited_maku_pct < 60:
        return "pre_maku"
    if maku_dialog_pct < 30:
        return "maku_interaction"
    if entered_dungeon_pct < 30:
        return "pre_dungeon"
    return "dungeon"
