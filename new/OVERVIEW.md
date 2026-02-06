# Overview

### Training Loop (KubeRay → LLM Evaluators → Feedback)
The system runs in iterative cycles. Each cycle improves the RL agent using LLM-generated feedback from the previous cycle's gameplay.

**Cycle A — RL Training Burst (KubeRay)**
1. A `RayJob` CR is submitted to the **KubeRay** operator, which spins up a `RayCluster` (managed via CodeFlare SDK from an OpenShift AI Workbench notebook).
2. Ray RLlib PPO runs distributed rollouts across N workers × M parallel PyBoy envs.
3. Episodes are exported to **MinIO** as segments (PNG frames + RAM JSONL + manifest).
4. The RayJob completes; the RayCluster scales down (or is reused).

**Cycle B — LLM Evaluator Pass (llm-d)**
5. An evaluator `Job` reads completed segments from MinIO.
6. Segments are fanned out to the **llm-d** judge fleet (vision / state / rule judges) via the Inference Gateway.
7. Judge scores are aggregated (self-consistency M=3, majority vote) → `scores.jsonl`.
8. Pairwise preferences are built; the **reward model** \(R_\phi\) is trained (Bradley-Terry).
9. `rm.pt` is saved to MinIO.

**Feedback → Next Cycle**
10. The next Cycle A `RayJob` loads the updated reward model and uses it for reward shaping (\(r' = r + \lambda R_\phi\)), plus optional SIL/AWR on top-K judged segments.
11. Repeat.

### Control (within each RL episode)
- PPO drives by default.
- Planner (LLM) handles **dialog** & **puzzle/critical** via constrained JSON macros.
- Watchdogs + confidence gating + time budgets.

### Deployment
- All OpenShift infra managed via **GitOps** (ArgoCD + Kustomize).
- RL training orchestrated by **KubeRay operator** (`RayCluster` / `RayJob` CRs), launched from **CodeFlare SDK** in OpenShift AI Workbench. **Kueue** manages job queuing.
- `gitops/base/` = shared manifests; `gitops/overlays/{dev,staging,prod}` = env patches.
- Mirrors old project's OpenShift patterns (KubeRay, KServe, vLLM, llm-d, MachineSets, NetworkPolicies) but organized declaratively.
- See `GITOPS.md` and `DEPLOY_YAML.md`.
