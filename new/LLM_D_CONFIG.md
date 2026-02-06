# llm-d Configuration (targets)

## Model Assignments
Each route uses the Qwen2.5 model best suited to the task:

- `/vision` → **Qwen2.5-VL-32B-Instruct** (multimodal vision-language; prod) / **Qwen2.5-VL-7B-Instruct** (dev). Reads 160×144 game frames + RAM JSON; outputs rubric scores. Strong OCR/grounding for HUD parsing, dialog detection, interactable identification.
- `/dialog` → **Qwen2.5-7B-Instruct** (text-only). Fast structured JSON output for dialog navigation. No vision needed; state provided as JSON.
- `/puzzle` → **Qwen2.5-32B-Instruct** (text-only; prod) / **Qwen2.5-7B-Instruct** (dev). Spatial reasoning for block puzzles, switch sequences, pathfinding macros.
- `/state` → **Qwen2.5-7B-Instruct** (text-only). Lightweight flag validation from RAM JSON; fast and cheap.

## KServe vLLM Services
- `qwen25-vl-32b` — serves `/vision` (prod). Image: vLLM with `Qwen/Qwen2.5-VL-32B-Instruct`. TP=2 or TP=4 depending on GPU. Requires vLLM ≥0.7.2 for Qwen2.5-VL support.
- `qwen25-32b` — serves `/puzzle` (prod). Image: vLLM with `Qwen/Qwen2.5-32B-Instruct`. TP=2.
- `qwen25-7b` — serves `/dialog` and `/state`. Image: vLLM with `Qwen/Qwen2.5-7B-Instruct`. Single GPU.

## Gateway Env
- `ENABLE_KVCACHE_AWARE_SCORER=true`
- `PD_ENABLED=true`
- `SESSION_AFFINITY_KEY=room_id,mode`

## Decoding
- temperature 0.1–0.2; fixed seed; batch-invariance enabled.
- `max_tokens`: vision ≤256, dialog ≤96, puzzle ≤160, state ≤64.
