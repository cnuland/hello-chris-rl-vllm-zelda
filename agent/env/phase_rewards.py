"""Declarative phase-driven reward profiles for milestone-based training.

Replaces scattered ``if has_gnarled_key_now:`` checks in reward_wrapper.py
with a lookup table of reward profiles keyed by game phase.  Each profile
declares which rewards are active, which are suppressed, directional
targets, loiter penalties, and dialog area restrictions.

Phase detection operates at **step granularity** using live RAM state,
not epoch-level aggregate statistics (that's phase_detector.py's job for
the LLM reward advisor).

Usage in RewardWrapper:
    manager = PhaseManager()
    manager.reset(sword_level=..., has_gnarled_key=..., ...)

    # On milestone achievement:
    changed = manager.update_phase(...)
    if changed:
        profile = manager.active_profile
        # apply directional target, etc.

    # In reward computation:
    if not manager.is_reward_suppressed("maku_tree_visit"):
        reward += bonus
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Canonical phase ordering — matches quest progression within a single
# episode.  Each phase corresponds to a state the agent can be IN at any
# given step, determined by live RAM reads (sword level, key flags, area
# group).
EPISODE_PHASES = [
    "pre_sword",
    "pre_maku",
    "maku_interaction",
    "post_key",
    "snow_region",
    "dungeon",
]


@dataclass
class PhaseRewardProfile:
    """Declarative reward configuration for a single game phase.

    Controls *which* rewards fire (suppression) and *how* spatial guidance
    works (directional targets, loiter penalties).  Reward *magnitudes*
    are still set by env vars / reward_config — profiles just gate them.
    """

    # Reward keys that produce zero output in this phase.
    suppressed: list[str] = field(default_factory=list)

    # Which area groups qualify for dialog_advance rewards.
    dialog_advance_groups: frozenset[int] = field(
        default_factory=lambda: frozenset({2, 3})
    )

    # Target room coordinates for directional bonus (row, col in 16x16 grid).
    # None means no phase-default directional guidance.
    directional_target: tuple[int, int] | None = None

    # Directional bonus magnitude (used as fallback when reward_config
    # doesn't set directional_bonus).
    directional_bonus: float = 0.0

    # Directional scale multiplier.
    directional_scale: float = 1.0

    # Per-step penalties by area group.  Key = active_group, value = penalty
    # per step.  Applied every step the agent is in the specified group.
    loiter_penalties: dict[int, float] = field(default_factory=dict)

    # Grace period (in steps) before loiter penalties start applying.
    # When the agent enters a penalized area group, the first N steps
    # are penalty-free.  This teaches "enter, interact, leave quickly"
    # instead of "never enter at all."
    loiter_grace_steps: int = 0

    # Phase-specific reward parameter overrides (key → value).
    # Applied on top of reward_config when this phase is active.
    param_overrides: dict[str, float] = field(default_factory=dict)

    # Area boost overrides (active_group → multiplier).
    area_boost_overrides: dict[int, float] = field(default_factory=dict)

    # Maximum cumulative coverage reward per episode in this phase.
    # None = unlimited (default, backward compatible).
    coverage_reward_cap: float | None = None


# ---------------------------------------------------------------------------
# Default profiles — encode the same behavior as the hardcoded checks
# in reward_wrapper.py _compute_reward().
# ---------------------------------------------------------------------------

DEFAULT_PHASE_PROFILES: dict[str, PhaseRewardProfile] = {
    "pre_sword": PhaseRewardProfile(
        directional_target=(14, 8),  # Hero's Cave area (south of village)
        dialog_advance_groups=frozenset({2, 3}),
        coverage_reward_cap=2000.0,  # Cap exploration so milestone rewards dominate
    ),
    "pre_maku": PhaseRewardProfile(
        directional_target=(5, 12),  # Maku Tree path (northeast)
        dialog_advance_groups=frozenset({2, 3}),
        coverage_reward_cap=2000.0,  # Cap exploration so gate slash reward is visible
    ),
    "maku_interaction": PhaseRewardProfile(
        directional_target=(5, 12),  # Stay near Maku Tree
        dialog_advance_groups=frozenset({2, 3}),
        # All Maku sub-events active (default behavior, nothing suppressed)
    ),
    "post_key": PhaseRewardProfile(
        # Suppress Maku Tree farming rewards
        suppressed=["maku_tree_visit", "maku_room", "maku_stage"],
        # Restrict dialog advance to group 3 only (Maku dialog no longer relevant)
        dialog_advance_groups=frozenset({3}),
        # Direct toward Dungeon 1
        directional_target=(10, 4),
        directional_bonus=50.0,
        directional_scale=1.0,
        # Penalize loitering in Maku Tree area — reduced from 1.0 to 0.1
        # to avoid PPO learning "never enter the Maku Tree."  The penalty
        # should be low enough that the gnarled_key bonus (500) + maku_tree
        # visit (100) still dominate a brief visit (~2K steps = -200 penalty).
        loiter_penalties={2: 0.1},
        # Grace period: first 3000 steps in group 2 are penalty-free,
        # giving the agent time to interact with the Maku Tree and exit.
        loiter_grace_steps=3000,
    ),
    "snow_region": PhaseRewardProfile(
        # Same suppressions as post_key — still heading to dungeon
        suppressed=["maku_tree_visit", "maku_room", "maku_stage"],
        dialog_advance_groups=frozenset({3}),
        directional_target=(10, 4),
        directional_bonus=50.0,
        directional_scale=1.0,
        loiter_penalties={2: 0.1},
        loiter_grace_steps=3000,
    ),
    "dungeon": PhaseRewardProfile(
        suppressed=["maku_tree_visit", "maku_room", "maku_stage"],
        dialog_advance_groups=frozenset({3, 4, 5}),
        directional_target=None,  # No directional guidance inside dungeons
        area_boost_overrides={4: 3.0, 5: 3.0},
    ),
}


# ---------------------------------------------------------------------------
# Step-level phase detection
# ---------------------------------------------------------------------------

def detect_episode_phase(
    *,
    sword_level: int,
    has_gnarled_key: bool,
    active_group: int,
    entered_snow_region: bool,
    baseline_sword: int = 0,
    baseline_gnarled_key: bool = False,
) -> str:
    """Detect the current game phase from live RAM state.

    This runs at reset() and when milestones fire (hybrid Option C).
    NOT the same as ``phase_detector.detect_phase()`` which uses epoch-level
    aggregate milestone percentages.

    Args:
        sword_level: Current SWORD_LEVEL RAM value (0 = no sword).
        has_gnarled_key: Whether GNARLED_KEY_OBTAINED flag is set in RAM.
        active_group: Current ACTIVE_GROUP (0=overworld, 2=maku, 4-5=dungeon).
        entered_snow_region: Whether the agent has reached the snow region
            this episode (sticky per-episode flag).
        baseline_sword: Sword level at episode start (from save state).
        baseline_gnarled_key: Whether the save state already had the key.

    Returns:
        Phase string from EPISODE_PHASES.
    """
    # Dungeon takes priority — if we're in a dungeon, we're in dungeon phase
    if active_group in (4, 5):
        return "dungeon"

    # Check Gnarled Key (from current RAM or save state baseline)
    if has_gnarled_key or baseline_gnarled_key:
        if entered_snow_region:
            return "snow_region"
        return "post_key"

    # In the Maku Tree area — interacting with the tree
    if active_group == 2:
        return "maku_interaction"

    # No sword yet
    if sword_level == 0 and baseline_sword == 0:
        return "pre_sword"

    # Has sword, heading to Maku Tree
    return "pre_maku"


# ---------------------------------------------------------------------------
# Phase manager
# ---------------------------------------------------------------------------

class PhaseManager:
    """Manages within-episode phase transitions and reward profile lookups.

    The manager maintains a current phase, provides profile queries, and
    handles transitions when milestones fire.  It also supports LLM advisor
    overrides that modify profiles between epochs.
    """

    def __init__(
        self,
        profiles: dict[str, PhaseRewardProfile] | None = None,
    ):
        self._profiles: dict[str, PhaseRewardProfile] = {}
        # Deep-copy default profiles so mutations don't affect the module-level dict
        src = profiles if profiles is not None else DEFAULT_PHASE_PROFILES
        for name, profile in src.items():
            self._profiles[name] = PhaseRewardProfile(
                suppressed=list(profile.suppressed),
                dialog_advance_groups=frozenset(profile.dialog_advance_groups),
                directional_target=profile.directional_target,
                directional_bonus=profile.directional_bonus,
                directional_scale=profile.directional_scale,
                loiter_penalties=dict(profile.loiter_penalties),
                loiter_grace_steps=profile.loiter_grace_steps,
                param_overrides=dict(profile.param_overrides),
                area_boost_overrides=dict(profile.area_boost_overrides),
                coverage_reward_cap=profile.coverage_reward_cap,
            )
        self._current_phase: str = "pre_sword"
        self._phase_history: list[tuple[str, int]] = []

    @property
    def current_phase(self) -> str:
        """The current game phase within this episode."""
        return self._current_phase

    @property
    def active_profile(self) -> PhaseRewardProfile:
        """The reward profile for the current phase."""
        return self._profiles.get(self._current_phase, PhaseRewardProfile())

    @property
    def phase_history(self) -> list[tuple[str, int]]:
        """List of (phase, step_number) transitions this episode."""
        return list(self._phase_history)

    def reset(
        self,
        *,
        sword_level: int,
        has_gnarled_key: bool,
        active_group: int,
        baseline_sword: int = 0,
        baseline_gnarled_key: bool = False,
    ) -> str:
        """Detect initial phase at episode start.

        Returns:
            The detected initial phase string.
        """
        self._current_phase = detect_episode_phase(
            sword_level=sword_level,
            has_gnarled_key=has_gnarled_key,
            active_group=active_group,
            entered_snow_region=False,
            baseline_sword=baseline_sword,
            baseline_gnarled_key=baseline_gnarled_key,
        )
        self._phase_history = [(self._current_phase, 0)]
        return self._current_phase

    def update_phase(
        self,
        *,
        sword_level: int,
        has_gnarled_key: bool,
        active_group: int,
        entered_snow_region: bool,
        baseline_sword: int = 0,
        baseline_gnarled_key: bool = False,
        step: int = 0,
    ) -> bool:
        """Re-detect phase after a milestone event.

        Returns:
            True if the phase changed.
        """
        new_phase = detect_episode_phase(
            sword_level=sword_level,
            has_gnarled_key=has_gnarled_key,
            active_group=active_group,
            entered_snow_region=entered_snow_region,
            baseline_sword=baseline_sword,
            baseline_gnarled_key=baseline_gnarled_key,
        )
        if new_phase != self._current_phase:
            old_phase = self._current_phase
            self._current_phase = new_phase
            self._phase_history.append((new_phase, step))
            logger.info(
                "PHASE TRANSITION: %s -> %s (step %d)",
                old_phase, new_phase, step,
            )
            return True
        return False

    def is_reward_suppressed(self, reward_key: str) -> bool:
        """Check if a reward is suppressed in the current phase."""
        return reward_key in self.active_profile.suppressed

    def get_loiter_penalty(self, active_group: int) -> float:
        """Get per-step loiter penalty for the given area group."""
        return self.active_profile.loiter_penalties.get(active_group, 0.0)

    def get_dialog_advance_groups(self) -> frozenset[int]:
        """Get the set of area groups that qualify for dialog advance rewards."""
        return self.active_profile.dialog_advance_groups

    def merge_advisor_overrides(
        self, phase: str, overrides: dict[str, Any]
    ) -> None:
        """Merge partial overrides from the LLM advisor into a phase profile.

        Args:
            phase: Phase name to override (must exist in profiles).
            overrides: Dict with optional keys: ``suppressed``,
                ``dialog_advance_groups``, ``directional_target``,
                ``directional_bonus``, ``directional_scale``,
                ``loiter_penalties``, ``param_overrides``,
                ``area_boost_overrides``.
        """
        if phase not in self._profiles:
            logger.warning("Cannot override unknown phase: %s", phase)
            return

        profile = self._profiles[phase]

        if "suppressed" in overrides:
            profile.suppressed = list(overrides["suppressed"])
        if "dialog_advance_groups" in overrides:
            profile.dialog_advance_groups = frozenset(
                overrides["dialog_advance_groups"]
            )
        if "directional_target" in overrides:
            val = overrides["directional_target"]
            if val is None:
                profile.directional_target = None
            else:
                profile.directional_target = (int(val[0]), int(val[1]))
        if "directional_bonus" in overrides:
            profile.directional_bonus = float(overrides["directional_bonus"])
        if "directional_scale" in overrides:
            profile.directional_scale = float(overrides["directional_scale"])
        if "loiter_penalties" in overrides:
            profile.loiter_penalties = {
                int(k): float(v)
                for k, v in overrides["loiter_penalties"].items()
            }
        if "loiter_grace_steps" in overrides:
            profile.loiter_grace_steps = int(overrides["loiter_grace_steps"])
        if "param_overrides" in overrides:
            profile.param_overrides.update(overrides["param_overrides"])
        if "area_boost_overrides" in overrides:
            profile.area_boost_overrides = {
                int(k): float(v)
                for k, v in overrides["area_boost_overrides"].items()
            }
        if "coverage_reward_cap" in overrides:
            val = overrides["coverage_reward_cap"]
            profile.coverage_reward_cap = float(val) if val is not None else None

        logger.info(
            "Applied advisor overrides to phase '%s': %s",
            phase, list(overrides.keys()),
        )
