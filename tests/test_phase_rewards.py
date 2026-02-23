"""Tests for the milestone-driven phase reward system.

Verifies:
  - detect_episode_phase() returns correct phase for each game state
  - PhaseManager transitions correctly on milestone events
  - DEFAULT_PHASE_PROFILES encode the same behavior as the old
    hardcoded ``if has_gnarled_key_now:`` blocks
  - Advisor overrides merge correctly into profiles
"""

from __future__ import annotations

import pytest

from agent.env.phase_rewards import (
    DEFAULT_PHASE_PROFILES,
    EPISODE_PHASES,
    PhaseManager,
    PhaseRewardProfile,
    detect_episode_phase,
)


# ---------------------------------------------------------------------------
# detect_episode_phase()
# ---------------------------------------------------------------------------

class TestDetectEpisodePhase:
    """Unit tests for step-level phase detection from RAM state."""

    def test_no_sword_returns_pre_sword(self):
        assert detect_episode_phase(
            sword_level=0, has_gnarled_key=False,
            active_group=0, entered_snow_region=False,
            baseline_sword=0, baseline_gnarled_key=False,
        ) == "pre_sword"

    def test_has_sword_returns_pre_maku(self):
        assert detect_episode_phase(
            sword_level=1, has_gnarled_key=False,
            active_group=0, entered_snow_region=False,
            baseline_sword=0, baseline_gnarled_key=False,
        ) == "pre_maku"

    def test_baseline_sword_returns_pre_maku(self):
        """Save state with sword should skip pre_sword."""
        assert detect_episode_phase(
            sword_level=0, has_gnarled_key=False,
            active_group=0, entered_snow_region=False,
            baseline_sword=1, baseline_gnarled_key=False,
        ) == "pre_maku"

    def test_in_maku_group_returns_maku_interaction(self):
        assert detect_episode_phase(
            sword_level=1, has_gnarled_key=False,
            active_group=2, entered_snow_region=False,
            baseline_sword=1, baseline_gnarled_key=False,
        ) == "maku_interaction"

    def test_has_key_returns_post_key(self):
        assert detect_episode_phase(
            sword_level=1, has_gnarled_key=True,
            active_group=0, entered_snow_region=False,
            baseline_sword=1, baseline_gnarled_key=False,
        ) == "post_key"

    def test_baseline_key_returns_post_key(self):
        """Save state with Gnarled Key should start in post_key."""
        assert detect_episode_phase(
            sword_level=1, has_gnarled_key=False,
            active_group=0, entered_snow_region=False,
            baseline_sword=1, baseline_gnarled_key=True,
        ) == "post_key"

    def test_snow_region_returns_snow_region(self):
        assert detect_episode_phase(
            sword_level=1, has_gnarled_key=True,
            active_group=0, entered_snow_region=True,
            baseline_sword=1, baseline_gnarled_key=False,
        ) == "snow_region"

    def test_in_dungeon_returns_dungeon(self):
        assert detect_episode_phase(
            sword_level=1, has_gnarled_key=True,
            active_group=4, entered_snow_region=False,
            baseline_sword=1, baseline_gnarled_key=False,
        ) == "dungeon"

    def test_dungeon_group_5(self):
        assert detect_episode_phase(
            sword_level=1, has_gnarled_key=True,
            active_group=5, entered_snow_region=True,
            baseline_sword=1, baseline_gnarled_key=False,
        ) == "dungeon"

    def test_dungeon_takes_priority_over_key(self):
        """Dungeon phase should take priority even with Gnarled Key."""
        assert detect_episode_phase(
            sword_level=1, has_gnarled_key=True,
            active_group=4, entered_snow_region=True,
            baseline_sword=1, baseline_gnarled_key=True,
        ) == "dungeon"


# ---------------------------------------------------------------------------
# PhaseManager
# ---------------------------------------------------------------------------

class TestPhaseManager:
    """Tests for the phase state machine."""

    def test_reset_detects_initial_phase(self):
        manager = PhaseManager()
        phase = manager.reset(
            sword_level=1, has_gnarled_key=False,
            active_group=0, baseline_sword=1,
            baseline_gnarled_key=False,
        )
        assert phase == "pre_maku"
        assert manager.current_phase == "pre_maku"

    def test_reset_with_key_starts_post_key(self):
        manager = PhaseManager()
        phase = manager.reset(
            sword_level=1, has_gnarled_key=True,
            active_group=0, baseline_sword=1,
            baseline_gnarled_key=False,
        )
        assert phase == "post_key"

    def test_phase_transition_on_key_acquisition(self):
        manager = PhaseManager()
        manager.reset(
            sword_level=1, has_gnarled_key=False,
            active_group=0, baseline_sword=1,
            baseline_gnarled_key=False,
        )
        assert manager.current_phase == "pre_maku"

        changed = manager.update_phase(
            sword_level=1, has_gnarled_key=True,
            active_group=0, entered_snow_region=False,
            baseline_sword=1, baseline_gnarled_key=False,
            step=100,
        )
        assert changed is True
        assert manager.current_phase == "post_key"

    def test_no_transition_when_phase_unchanged(self):
        manager = PhaseManager()
        manager.reset(
            sword_level=1, has_gnarled_key=False,
            active_group=0, baseline_sword=1,
            baseline_gnarled_key=False,
        )

        changed = manager.update_phase(
            sword_level=1, has_gnarled_key=False,
            active_group=0, entered_snow_region=False,
            baseline_sword=1, baseline_gnarled_key=False,
        )
        assert changed is False
        assert manager.current_phase == "pre_maku"

    def test_phase_history_tracking(self):
        manager = PhaseManager()
        manager.reset(
            sword_level=0, has_gnarled_key=False,
            active_group=0, baseline_sword=0,
            baseline_gnarled_key=False,
        )
        assert len(manager.phase_history) == 1
        assert manager.phase_history[0] == ("pre_sword", 0)

        manager.update_phase(
            sword_level=1, has_gnarled_key=False,
            active_group=0, entered_snow_region=False,
            baseline_sword=0, baseline_gnarled_key=False,
            step=50,
        )
        assert len(manager.phase_history) == 2
        assert manager.phase_history[1] == ("pre_maku", 50)

    def test_snow_region_transition(self):
        manager = PhaseManager()
        manager.reset(
            sword_level=1, has_gnarled_key=True,
            active_group=0, baseline_sword=1,
            baseline_gnarled_key=False,
        )
        assert manager.current_phase == "post_key"

        changed = manager.update_phase(
            sword_level=1, has_gnarled_key=True,
            active_group=0, entered_snow_region=True,
            baseline_sword=1, baseline_gnarled_key=False,
        )
        assert changed is True
        assert manager.current_phase == "snow_region"


# ---------------------------------------------------------------------------
# Suppression rules (backward compatibility with hardcoded checks)
# ---------------------------------------------------------------------------

class TestDefaultProfileBackwardCompat:
    """Verify DEFAULT_PHASE_PROFILES encode the same behavior as the old
    hardcoded has_gnarled_key_now checks in reward_wrapper.py."""

    def test_pre_sword_no_suppressions(self):
        profile = DEFAULT_PHASE_PROFILES["pre_sword"]
        assert "maku_tree_visit" not in profile.suppressed
        assert "maku_room" not in profile.suppressed
        assert "maku_stage" not in profile.suppressed

    def test_pre_maku_no_suppressions(self):
        profile = DEFAULT_PHASE_PROFILES["pre_maku"]
        assert "maku_tree_visit" not in profile.suppressed

    def test_maku_interaction_allows_maku_rewards(self):
        profile = DEFAULT_PHASE_PROFILES["maku_interaction"]
        assert "maku_tree_visit" not in profile.suppressed
        assert "maku_room" not in profile.suppressed
        assert "maku_stage" not in profile.suppressed

    def test_post_key_suppresses_maku_rewards(self):
        """After Gnarled Key: maku_tree_visit, maku_room, maku_stage suppressed."""
        profile = DEFAULT_PHASE_PROFILES["post_key"]
        assert "maku_tree_visit" in profile.suppressed
        assert "maku_room" in profile.suppressed
        assert "maku_stage" in profile.suppressed

    def test_snow_region_suppresses_maku_rewards(self):
        profile = DEFAULT_PHASE_PROFILES["snow_region"]
        assert "maku_tree_visit" in profile.suppressed

    def test_dungeon_suppresses_maku_rewards(self):
        profile = DEFAULT_PHASE_PROFILES["dungeon"]
        assert "maku_tree_visit" in profile.suppressed

    def test_post_key_dialog_restricted_to_group3(self):
        """After key: dialog advance only in group 3 (indoors)."""
        profile = DEFAULT_PHASE_PROFILES["post_key"]
        assert 3 in profile.dialog_advance_groups
        assert 2 not in profile.dialog_advance_groups

    def test_maku_interaction_dialog_includes_group2(self):
        """Before key: dialog advance in groups 2 and 3."""
        profile = DEFAULT_PHASE_PROFILES["maku_interaction"]
        assert 2 in profile.dialog_advance_groups
        assert 3 in profile.dialog_advance_groups

    def test_post_key_loiter_penalty_in_group2(self):
        """After key: per-step penalty for being in Maku Tree area."""
        profile = DEFAULT_PHASE_PROFILES["post_key"]
        assert profile.loiter_penalties.get(2, 0) > 0

    def test_pre_maku_no_loiter_penalty(self):
        profile = DEFAULT_PHASE_PROFILES["pre_maku"]
        assert profile.loiter_penalties.get(2, 0) == 0

    def test_post_key_directional_target_dungeon1(self):
        """After key: directional target at (10, 4) for Dungeon 1."""
        profile = DEFAULT_PHASE_PROFILES["post_key"]
        assert profile.directional_target == (10, 4)

    def test_pre_maku_directional_target_maku_tree(self):
        profile = DEFAULT_PHASE_PROFILES["pre_maku"]
        assert profile.directional_target == (5, 12)

    def test_dungeon_no_directional_target(self):
        profile = DEFAULT_PHASE_PROFILES["dungeon"]
        assert profile.directional_target is None


# ---------------------------------------------------------------------------
# PhaseManager suppression queries
# ---------------------------------------------------------------------------

class TestPhaseManagerSuppression:
    """Test is_reward_suppressed and get_loiter_penalty on PhaseManager."""

    def test_suppressed_in_post_key(self):
        manager = PhaseManager()
        manager.reset(
            sword_level=1, has_gnarled_key=True,
            active_group=0, baseline_sword=1,
            baseline_gnarled_key=False,
        )
        assert manager.is_reward_suppressed("maku_tree_visit") is True
        assert manager.is_reward_suppressed("maku_room") is True
        assert manager.is_reward_suppressed("maku_stage") is True

    def test_not_suppressed_in_pre_maku(self):
        manager = PhaseManager()
        manager.reset(
            sword_level=1, has_gnarled_key=False,
            active_group=0, baseline_sword=1,
            baseline_gnarled_key=False,
        )
        assert manager.is_reward_suppressed("maku_tree_visit") is False
        assert manager.is_reward_suppressed("sword") is False

    def test_loiter_penalty_in_post_key(self):
        manager = PhaseManager()
        manager.reset(
            sword_level=1, has_gnarled_key=True,
            active_group=0, baseline_sword=1,
            baseline_gnarled_key=False,
        )
        assert manager.get_loiter_penalty(2) > 0
        assert manager.get_loiter_penalty(0) == 0

    def test_no_loiter_penalty_in_pre_maku(self):
        manager = PhaseManager()
        manager.reset(
            sword_level=1, has_gnarled_key=False,
            active_group=0, baseline_sword=1,
            baseline_gnarled_key=False,
        )
        assert manager.get_loiter_penalty(2) == 0

    def test_dialog_groups_in_post_key(self):
        manager = PhaseManager()
        manager.reset(
            sword_level=1, has_gnarled_key=True,
            active_group=0, baseline_sword=1,
            baseline_gnarled_key=False,
        )
        groups = manager.get_dialog_advance_groups()
        assert 3 in groups
        assert 2 not in groups

    def test_dialog_groups_in_maku_interaction(self):
        manager = PhaseManager()
        manager.reset(
            sword_level=1, has_gnarled_key=False,
            active_group=2, baseline_sword=1,
            baseline_gnarled_key=False,
        )
        groups = manager.get_dialog_advance_groups()
        assert 2 in groups
        assert 3 in groups


# ---------------------------------------------------------------------------
# Advisor overrides
# ---------------------------------------------------------------------------

class TestAdvisorOverrides:
    """Test that LLM advisor can modify phase profiles."""

    def test_override_directional_bonus(self):
        manager = PhaseManager()
        manager.merge_advisor_overrides("post_key", {
            "directional_bonus": 200.0,
        })
        profile = manager._profiles["post_key"]
        assert profile.directional_bonus == 200.0

    def test_override_loiter_penalties(self):
        manager = PhaseManager()
        manager.merge_advisor_overrides("post_key", {
            "loiter_penalties": {2: 3.0},
        })
        profile = manager._profiles["post_key"]
        assert profile.loiter_penalties[2] == 3.0

    def test_override_suppressed_list(self):
        manager = PhaseManager()
        manager.merge_advisor_overrides("pre_maku", {
            "suppressed": ["maku_tree_visit"],
        })
        profile = manager._profiles["pre_maku"]
        assert "maku_tree_visit" in profile.suppressed

    def test_override_dialog_groups(self):
        manager = PhaseManager()
        manager.merge_advisor_overrides("post_key", {
            "dialog_advance_groups": [3, 4, 5],
        })
        profile = manager._profiles["post_key"]
        assert 4 in profile.dialog_advance_groups

    def test_override_directional_target(self):
        manager = PhaseManager()
        manager.merge_advisor_overrides("post_key", {
            "directional_target": [8, 3],
        })
        profile = manager._profiles["post_key"]
        assert profile.directional_target == (8, 3)

    def test_override_unknown_phase_logs_warning(self):
        """Overriding a non-existent phase should not raise."""
        manager = PhaseManager()
        manager.merge_advisor_overrides("nonexistent_phase", {
            "directional_bonus": 100.0,
        })
        # Should not crash, just warn

    def test_profiles_are_independent_copies(self):
        """Ensure modifying one PhaseManager's profiles doesn't affect another."""
        manager1 = PhaseManager()
        manager2 = PhaseManager()
        manager1.merge_advisor_overrides("post_key", {
            "directional_bonus": 999.0,
        })
        assert manager2._profiles["post_key"].directional_bonus == 50.0


# ---------------------------------------------------------------------------
# EPISODE_PHASES ordering
# ---------------------------------------------------------------------------

class TestEpisodePhases:
    """Verify the phase list is correctly ordered."""

    def test_all_default_profiles_have_entries(self):
        for phase in EPISODE_PHASES:
            assert phase in DEFAULT_PHASE_PROFILES, f"Missing profile for {phase}"

    def test_phase_order(self):
        assert EPISODE_PHASES.index("pre_sword") < EPISODE_PHASES.index("pre_maku")
        assert EPISODE_PHASES.index("pre_maku") < EPISODE_PHASES.index("maku_interaction")
        assert EPISODE_PHASES.index("maku_interaction") < EPISODE_PHASES.index("post_key")
        assert EPISODE_PHASES.index("post_key") < EPISODE_PHASES.index("snow_region")
        assert EPISODE_PHASES.index("snow_region") < EPISODE_PHASES.index("dungeon")
