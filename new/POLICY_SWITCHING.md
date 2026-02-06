# Policy Switching

**Into planner** when:
- Dialog window detected **and** cursor/choice visible for ≥3 frames.
- Puzzle flags active (switches/blocks/cutscene preflags) for ≥3 frames.

**Back to PPO** when:
- Dialog cleared, puzzle flag flipped, timeout hit, HP dropped, or planner `fallback`.

**Budgets**
- Dialog ≤ 2s; Puzzle ≤ 6s. Confidence < 0.5 → no takeover.
