# Evaluator Pipeline (llm-d) — Cycle B

This pipeline runs **after each KubeRay RL training burst** (Cycle A) completes and episodes have been exported to MinIO.

1. **Trigger**: Cycle A `RayJob` completes → episode segments are in MinIO (`episodes/{id}/frames/*.png`, `ram.jsonl`, `manifest.json`).
2. **Vision Judge** (`/vision`) — **Qwen2.5-VL-32B-Instruct** (vLLM/KServe) reads selected crops + mini RAM JSON; outputs rubric scores. Strong OCR/grounding for HUD, dialog, and interactable detection.
3. **State Judge** (`/state`) — small text LLM (or rules-only) validates flags from RAM JSON.
4. **Rule Judge** (`/rules`) — deterministic checks (keys, room transitions).
5. **Aggregation**: self-consistency M=3 per LLM judge; majority vote; produce `scores.jsonl`.
6. **Preferences**: build pairwise prefs; fit \(R_\phi\) (Bradley-Terry). Save `rm.pt` to MinIO.
7. **Feedback → next Cycle A**: the next KubeRay `RayJob` loads `rm.pt` for shaped reward (\(r' = r + \lambda R_\phi\)); optional SIL/AWR on top-K judged segments.

**Metrics**: agreement rate, p50/p95 latency, cost/100 segments, RM drift.
