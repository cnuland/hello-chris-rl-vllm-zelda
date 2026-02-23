"""Advancing checkpoint selection for curriculum-based save state progression.

At the end of each training epoch, this module evaluates which game milestones
the agent has consistently achieved and selects the best captured PyBoy save
state to advance to for the next epoch.  This eliminates wasted steps
re-traversing already-mastered sections of the game.

Milestone hierarchy (ordered by game progression):
    got_sword → visited_maku_tree → gate_slashed → maku_dialog
    → gnarled_key → entered_snow_region → entered_dungeon

Save states are captured by RewardWrapper._capture_milestone_state() during
training and written to MILESTONE_STATE_DIR as .state files with .json
metadata sidecars.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Ordered from earliest to latest in game progression.
MILESTONE_ORDER = [
    "got_sword",
    "visited_maku_tree",
    "gate_slashed",
    "maku_dialog",
    "gnarled_key",
    "entered_snow_region",
    "entered_dungeon",
]

# Mapping from milestone names to the milestones dict keys used in
# train_pufferlib.py for counting total episodes that achieved each.
MILESTONE_TO_COUNT_KEY = {
    "got_sword": "total_got_sword",
    "visited_maku_tree": "total_visited_maku_tree",
    "gate_slashed": "total_gate_slashed",
    "maku_dialog": "total_maku_dialog",
    "gnarled_key": "total_gnarled_key",
    "entered_snow_region": "total_entered_snow_region",
    "entered_dungeon": "total_entered_dungeon",
}


@dataclass
class AdvancementDecision:
    """Result of the advancing checkpoint selection."""

    should_advance: bool
    milestone_name: str | None = None
    state_path: str | None = None
    percentage: float = 0.0
    reason: str = ""


def select_advancing_checkpoint(
    milestones: dict,
    episodes_completed: int,
    milestone_state_dir: str,
    current_milestone: str | None = None,
    threshold: float = 0.40,
    min_episodes: int = 20,
) -> AdvancementDecision:
    """Select the best advancing checkpoint from this epoch's results.

    Walks the milestone hierarchy from latest to earliest.  For each
    milestone that exceeds the threshold percentage, checks if a
    captured state file exists.  Returns the *highest* milestone that
    is consistently achieved AND has a state file available.

    Args:
        milestones: Dict from epoch metadata (total_got_sword, etc.).
        episodes_completed: Total episodes completed this epoch.
        milestone_state_dir: Directory containing captured .state files.
        current_milestone: The milestone the current save state starts at
            (None = original pre-milestone state).  We only advance
            forward, never backward.
        threshold: Fraction of episodes required (default 0.40 = 40%).
        min_episodes: Minimum episodes before considering advancement.

    Returns:
        AdvancementDecision with the selection result.
    """
    if episodes_completed < min_episodes:
        return AdvancementDecision(
            should_advance=False,
            reason=f"Too few episodes ({episodes_completed} < {min_episodes})",
        )

    # Determine which milestones are "after" the current one
    current_idx = -1
    if current_milestone and current_milestone in MILESTONE_ORDER:
        current_idx = MILESTONE_ORDER.index(current_milestone)

    # Walk milestones from latest to earliest, looking for the highest
    # consistently-achieved milestone beyond current position.
    for milestone in reversed(MILESTONE_ORDER):
        ms_idx = MILESTONE_ORDER.index(milestone)
        if ms_idx <= current_idx:
            continue  # don't go backwards

        count_key = MILESTONE_TO_COUNT_KEY[milestone]
        count = milestones.get(count_key, 0)
        pct = count / episodes_completed

        if pct >= threshold:
            # Find best state file for this milestone
            state_path = _find_best_state(milestone_state_dir, milestone)
            if state_path:
                return AdvancementDecision(
                    should_advance=True,
                    milestone_name=milestone,
                    state_path=state_path,
                    percentage=pct,
                    reason=(
                        f"Milestone '{milestone}' achieved in {pct:.0%} of "
                        f"episodes ({count}/{episodes_completed}), "
                        f"above threshold {threshold:.0%}"
                    ),
                )
            else:
                logger.warning(
                    "Milestone '%s' at %.0f%% but no state file found in %s",
                    milestone,
                    pct * 100,
                    milestone_state_dir,
                )

    return AdvancementDecision(
        should_advance=False,
        reason="No milestone exceeded threshold or no state files available",
    )


def _find_best_state(state_dir: str, milestone: str) -> str | None:
    """Find the best .state file for a given milestone.

    Picks the state file with the highest ``reward_so_far`` from its
    metadata sidecar.  If no metadata exists, picks any matching file.
    """
    if not os.path.isdir(state_dir):
        return None

    candidates: list[tuple[str, float, int]] = []
    for fname in os.listdir(state_dir):
        if fname.startswith(milestone + "_") and fname.endswith(".state"):
            state_path = os.path.join(state_dir, fname)
            meta_path = state_path + ".json"
            reward = 0.0
            tiles = 0
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    reward = meta.get("reward_so_far", 0.0)
                    tiles = meta.get("unique_tiles", 0)
                except Exception:
                    pass
            candidates.append((state_path, reward, tiles))

    if not candidates:
        return None

    # Sort by reward (primary), then tiles (secondary), pick best
    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    best_path = candidates[0][0]
    logger.info(
        "Selected best state for '%s': %s (reward=%.1f, tiles=%d, %d candidates)",
        milestone,
        os.path.basename(best_path),
        candidates[0][1],
        candidates[0][2],
        len(candidates),
    )
    return best_path


def upload_advancing_state(
    state_path: str,
    milestone: str,
    epoch: int,
    metadata: dict,
) -> str:
    """Upload the selected advancing state to MinIO for persistence.

    Returns the S3 key for the uploaded state.
    """
    from agent.utils.config import S3Config
    from agent.utils.s3 import S3Client

    s3 = S3Client(S3Config())
    s3.ensure_bucket("zelda-models")

    # Upload state bytes
    s3_key = f"save-states/advancing/epoch_{epoch}_{milestone}.state"
    with open(state_path, "rb") as f:
        s3.upload_bytes("zelda-models", s3_key, f.read())

    # Upload metadata
    meta_key = f"save-states/advancing/epoch_{epoch}_{milestone}.json"
    meta = {
        **metadata,
        "milestone": milestone,
        "epoch": epoch,
        "s3_key": s3_key,
    }
    s3.upload_bytes(
        "zelda-models", meta_key, json.dumps(meta, indent=2).encode()
    )

    logger.info(
        "Uploaded advancing checkpoint: %s (milestone=%s, epoch=%d)",
        s3_key,
        milestone,
        epoch,
    )
    return s3_key


def clear_milestone_states(state_dir: str) -> int:
    """Remove all milestone state files from the temp directory.

    Called at the start of each epoch to prevent stale state files
    from previous epochs from being selected.

    Returns number of files removed.
    """
    if not os.path.isdir(state_dir):
        return 0
    removed = 0
    for fname in os.listdir(state_dir):
        if fname.endswith(".state") or fname.endswith(".state.json"):
            try:
                os.remove(os.path.join(state_dir, fname))
                removed += 1
            except OSError:
                pass
    return removed
