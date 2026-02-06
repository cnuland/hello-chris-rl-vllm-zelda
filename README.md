# hello-chris-rl-zelda — Hybrid RL + LLM with llm-d (OpenShift AI)

This project revives our Game Boy *Zelda: Oracle of Seasons* agent with an iterative two-cycle pipeline:
1) **RL training burst** — **KubeRay** `RayJob` runs distributed PPO (Ray RLlib) with exploration fixes (coverage, curiosity, archive). Episodes export to MinIO.
2) **LLM evaluator pass** — a fleet of **llm-d** judge models (vision + state + rules) scores the completed trajectories; we train a reward model (RLAIF) and optionally self-imitate the best segments.
3) **Feedback loop** — the updated reward model is loaded into the next RL burst, improving each cycle.

**Why this will do better than the last attempt**
- Sparse rewards become learnable via **preference-trained reward model** (LLM evaluators score each run, then reward model improves the next run).
- **Go-Explore-lite** and coverage rewards end "ramming into trees."
- **Dialog/puzzle micro-domains** handled by constrained planners (**Qwen2.5-7B** for dialog, **Qwen2.5-32B** for puzzles) via vLLM/KServe, routed by **llm-d**. Vision judge uses **Qwen2.5-VL-32B** for frame-level scoring.
- Distributed training via **KubeRay** (Ray RLlib PPO on OpenShift, managed by CodeFlare SDK + Kueue).
- YAML-first deployments (OpenShift AI + KServe + vLLM + llm-d).
- **GitOps-driven** via OpenShift GitOps (ArgoCD) + Kustomize — all infra is declarative and environment-aware.

## Quickstart (high level)
1. Put your previous project under `old/` (see `old/README.md`).
2. Ask Claude to follow **CLAUDE_TASKS.md** in order.
3. Deploy infra via **OpenShift GitOps**: ArgoCD syncs `gitops/overlays/{dev,staging,prod}` (see `new/GITOPS.md`).
4. Launch RL training burst per `new/WORKBENCH.md`.
5. Run evaluator pass per `new/EVAL_PIPELINE.md`.

## Deployment (GitOps)
All OpenShift resources live under `gitops/` using **Kustomize** (base + overlays).
- `gitops/base/` — shared manifests: KServe InferenceService, vLLM decode Deployments, llm-d Gateway + EPP, KubeRay RayCluster/RayJob, evaluator Jobs, MinIO, NetworkPolicies, monitoring.
- `gitops/overlays/dev|staging|prod` — environment patches (replica counts, GPU resources, MachineSets for prod).
- `gitops/argocd/` — ArgoCD `Application` and `AppProject` CRs. Apply once; ArgoCD keeps the cluster in sync with git.

```bash
# Bootstrap ArgoCD applications (one-time)
oc apply -f gitops/argocd/project.yaml
oc apply -f gitops/argocd/applications/dev.yaml   # or staging / prod
```

See `new/GITOPS.md` and `new/DEPLOY_YAML.md` for full specs.

## Directory map
- `old/` — previous baseline (unmodified; includes original flat OpenShift YAMLs for reference).
- `new/` — new hybrid system specs (Claude will generate code + YAML from these).
- `gitops/` — Kustomize bases, overlays, and ArgoCD Application CRs.
- Top-level docs: tasks, testing, contributing, and structure.
