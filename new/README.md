# New Hybrid System

This directory contains the authoritative specifications for the **new** Zelda agent. Claude will generate code and YAML from these docs.

**Key tech**
- **RL**: PPO + coverage + RND curiosity + archive restarts.
- **Models**: Qwen2.5 family via **vLLM/KServe**, routed by **llm-d** (4 routes). **Qwen2.5-VL-32B** (vision judge), **Qwen2.5-32B** (puzzle planner), **Qwen2.5-7B** (dialog planner + state judge).
- **Evaluators**: vision judge, state judge, and rule judge; **RLAIF** reward model; optional SIL/AWR.
- **Deployment**: **OpenShift GitOps** (ArgoCD) + **Kustomize** — base manifests in `gitops/base/`, env overlays in `gitops/overlays/{dev,staging,prod}`. No ad-hoc `oc apply`; all infra is git-driven. See `GITOPS.md`.
