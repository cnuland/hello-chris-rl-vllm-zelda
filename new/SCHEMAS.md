# Schemas (authoritative)

## Planner Output (Dialog)
```json
{
  "intent": "advance|select|yes|no|shop_buy|end",
  "presses": [{"btn": "A|B|UP|DOWN|LEFT|RIGHT", "frames": 2}],
  "guard": {"expect": "more_text|choice|end", "timeout_s": 2.0},
  "confidence": 0.0
}
```

## Planner Output (Puzzle)
```json
{
  "subgoal": "open_door|trigger_switch|align_block|start_cutscene|set_season|leave_room",
  "macros": [
    {"name": "MOVE_TO", "args": {"x": 112, "y": 80}, "timeout_s": 1.5},
    {"name": "PUSH_BLOCK", "args": {"dir": "RIGHT", "steps": 1}}
  ],
  "fallback": "handoff_controller",
  "confidence": 0.0
}
```

## Judge Output (per segment)
```json
{
  "segment_id": "string",
  "scores": {
    "progress": 0..1,
    "dialog": 0..1,
    "puzzle": 0..1,
    "novelty": 0..1,
    "efficiency": 0..1
  },
  "rationale": "short string"
}
```
