# Claude Task Plan (authoritative build list)

> Claude: Execute tasks **in order**. For each task, produce code/YAML, pass unit checks where applicable, and update test files.

## Phase 0 — Bootstrap
1. **Repo Scaffolding**
   - Create Python package `agent/` with subpackages: `env/`, `rl/`, `planner/`, `evaluator/`, `utils/`.
   - Generate `pyproject.toml` (set Python 3.11), `ruff` + `black` config, and `pre-commit`.
   - Output: code + minimal placeholder tests.

## Phase 1 — Environment & State
2. **Gymnasium Wrapper**
   - Implement `agent/env/zelda_env.py` integrating **PyBoy** (headless) with reset/save/load state, fixed step, and deterministic seeding.
   - Implement RAM taps for `room_id`, `tile_x/y`, dialog flags, OAM presence.
   - Tests: step/reset determinism; RAM read sanity.

3. **State Encoder**
   - Implement `agent/env/state_encoder.py` that yields:
     - Compact vector obs (128-D), and
     - Structured JSON (for LLM) with coords, flags, inventory bits.
   - Tests: shape checks; JSON schema validation (see `new/SCHEMAS.md`).

## Phase 2 — RL Core (KubeRay + Ray RLlib)
4. **PPO Trainer (Ray RLlib)**
   - Implement PPO via **Ray RLlib** (`PPOConfig`), mirroring `old/run-ray-zelda.py`.
   - Register custom Gymnasium env with Ray (`register_env`).
   - Configurable via env vars: `RAY_WORKERS`, `ENVS_PER_WORKER`, `EPISODE_LENGTH`, `BATCH_SIZE`.
   - Sticky actions + latched holds; frame_skip configurable.
   - Add coverage reward and small RND curiosity (clamped to ≤30% extrinsic).
   - CPU-only training (num_gpus=0) to preserve GPUs for LLM inference.
   - Episode export to MinIO on completion.
   - Tests: training loop smoke test (local Ray); coverage reward increments on movement.

5. **Archive (Go-Explore-lite)**
   - Implement archive keyed by `(room_id, tile_bin)`; periodic restart from frontier.
   - Tests: archive update & resume.

## Phase 3 — Planner (LLM micro-domains)
6. **Policy Switch**
   - Implement `agent/planner/policy_switch.py` predicates: `is_dialog_state`, `is_puzzle_state`, with hysteresis.
   - Implement `macro_executor.py` for MOVE_TO, PUSH_BLOCK, USE_ITEM, with timeouts.
   - Tests: predicate and macro unit tests with synthetic frames.

7. **LLM Client (llm-d)**
   - Implement HTTP clients for routes `/vision`, `/dialog`, `/puzzle` with retries, seed, JSON grammar enforcement.
   - Tests: mock server; JSON validation reject/repair.

## Phase 4 — Evaluator Fleet & Reward Model
8. **Episode Exporter**
   - Export segments (video PNGs + RAM JSONL) to MinIO.
   - Tests: write/read & manifest integrity.

9. **Evaluator Ingest**
   - Implement `agent/evaluator/ingest.py` to batch segments through llm-d judges; self-consistency M=3; majority vote; write scores JSONL.
   - Tests: mock 3-judge ensemble → aggregate.

10. **Preference Builder & RM**
    - Build pairwise prefs; train reward model \(R_\phi\) (Bradley-Terry); wrap as potential-based shaping.
    - Tests: basic fitting on synthetic ordered segments; invariance to potential.

11. **SIL/AWR Self-Imitation**
    - Add top-K segment buffer; imitation updates periodically.
    - Tests: target/behavior policy update smoke test.

## Phase 5 — Infra & Deployment (GitOps + Kustomize)
12. **Kustomize Base Manifests**
    - Scaffold `gitops/base/` with sub-directories per component: `llm-inference/`, `llm-d-gateway/`, `rl-training/`, `evaluator/`, `storage/`, `networking/`, `monitoring/`.
    - Each sub-dir: `kustomization.yaml` + resource YAMLs.
    - `llm-inference/`: Three KServe `InferenceService` CRs (vLLM): **Qwen2.5-VL-32B-Instruct** (vision, TP=2), **Qwen2.5-32B-Instruct** (puzzle, TP=2), **Qwen2.5-7B-Instruct** (dialog + state, 1 GPU). Decode `Deployment`, `Service`s, `ServiceAccount`, RBAC. Follow deployment patterns from `old/openshift/llm-deployments/` and `old/llama4scout-17b-inferenceservice.yaml`.
    - `llm-d-gateway/`: `Gateway`, per-route `HTTPRoute` (vision/dialog/puzzle/state), each to the correct model's `InferencePool` with `PrefixHash` session affinity, EPP `Deployment`. Follow patterns from `old/openshift/llm-deployments/llama4-scout-inference-gateway.yaml`.
    - `rl-training/`: KubeRay `RayCluster` CR + `RayJob` CR, Kueue `LocalQueue`, `Service`, `ServiceAccount`, RBAC (including raycluster/rayjob verbs). Mirror `old/run-kuberay-zelda.ipynb` cluster config and `old/run-ray-zelda.py` job submission.
    - `evaluator/`: evaluator batch `Job`, reward-model training `Job`.
    - `storage/`: MinIO `Deployment`, `Service`, `PVC`.
    - `networking/`: `NetworkPolicy` set. Mirror `old/llm-d-network-policy.yaml`.
    - `monitoring/`: `ServiceMonitor`, Grafana dashboard `ConfigMap`.
    - Validate with `kustomize build gitops/base/`.

13. **Kustomize Overlays (dev / staging / prod)**
    - `gitops/overlays/dev/`: single LLM replica, RayCluster with 1 worker × 3 envs, reduced GPU requests.
    - `gitops/overlays/staging/`: 2 LLM replicas, RayCluster with 3 workers × 6 envs.
    - `gitops/overlays/prod/`: 3+ LLM replicas, RayCluster with 5+ workers × 6 envs (30+ rollouts), ROSA `MachineSet` YAMLs for `llm-inference` and `pyboy-training` nodes (mirror `old/openshift/machinesets/` and `old/PRODUCTION_CLUSTER_CONFIG.md`).
    - Each overlay: `kustomization.yaml` + `patches/` directory.
    - Validate each overlay with `kustomize build`.

14. **ArgoCD Application CRs**
    - `gitops/argocd/project.yaml`: `AppProject` scoped to `zelda-rl` namespace + cluster resources.
    - `gitops/argocd/applications/{dev,staging,prod}.yaml`: ArgoCD `Application` pointing to respective overlay path, automated sync with self-heal + prune.
    - Validate specs per `new/GITOPS.md`.

15. **Metrics & Dashboards**
    - Expose Prom/OTel metrics per `new/METRICS.md`; produce Grafana JSON exemplars.
    - Include ArgoCD sync health and drift alerts.

## Phase 6 — Tests & CI
16. **Kustomize Validation**
    - `kustomize build` succeeds for base and all overlays.
    - `kubeconform` or `kubeval` validates generated manifests against OpenShift/K8s schemas.

17. **End-to-End Dry-Run**
    - Simulated short rollout with mocks; evaluator pass; reward model update → second training burst.
    - All tests green.
