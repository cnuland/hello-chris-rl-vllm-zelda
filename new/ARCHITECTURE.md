# Architecture

## Runtime Components
- **Env**: Gymnasium + PyBoy; deterministic seedable; RAM taps for room/flags.
- **Encoder**: 128-D vector obs + compact JSON for planner.
- **Policy Switch**: `is_dialog_state` / `is_puzzle_state` with 3-frame hysteresis.
- **Planner**: vLLM (Qwen2.5 models), routes `/vision` (Qwen2.5-VL-32B), `/dialog` (Qwen2.5-7B), `/puzzle` (Qwen2.5-32B).
- **Executor**: MOVE_TO (A*/greedy), PUSH_BLOCK, USE_ITEM; abort on timeout or HP drop.
- **Exploration**: coverage map (per room), RND curiosity, archive restarts.
- **Evaluator Fleet (llm-d)**: ensemble judges, self-consistency M=3, majority vote; preference learning for reward model.
- **Metrics**: per-mode latency, takeover success, unique rooms/episode, judge agreement, reward drift.

## RL Training Orchestration (KubeRay)
Distributed RL training runs on **KubeRay** (mirrors `old/run-kuberay-zelda.ipynb` and `old/run-ray-zelda.py`):
- **KubeRay operator** manages `RayCluster` and `RayJob` CRs on OpenShift.
- **CodeFlare SDK** (`codeflare-sdk`) creates and submits Ray clusters from an OpenShift AI Workbench notebook.
- **Kueue** handles job admission and resource quota.
- Ray RLlib PPO distributes rollouts across workers; CPU-only training preserves GPUs for LLM inference.
- Each training burst produces episode segments exported to MinIO. On completion, the evaluator pass (Cycle B) runs, and the resulting reward model feeds back into the next `RayJob`.

## Deployment Layer (OpenShift + GitOps)
All infrastructure runs on **OpenShift** (ROSA) and is managed via **OpenShift GitOps** (ArgoCD) + **Kustomize**.
- Manifests in `gitops/base/` (Kustomize), environment patches in `gitops/overlays/{dev,staging,prod}`.
- ArgoCD `Application` CRs auto-sync cluster state from git.
- **LLM inference**: KServe `InferenceService` + vLLM decode `Deployment` on dedicated GPU nodes (`llm-inference` MachineSet).
- **llm-d gateway**: Inference Gateway + EPP with `PrefixHash` session affinity + per-route `HTTPRoute`s.
- **RL training**: **KubeRay** `RayCluster` + `RayJob` CRs on dedicated training nodes (`pyboy-training` MachineSet). Kueue `LocalQueue` for admission control.
- **Evaluator**: batch `Job` + reward-model training `Job` triggered after each RL burst.
- **Storage**: MinIO for episode segments, models, and judge outputs.
- **Networking**: `NetworkPolicy` set for cross-namespace LLM access.
- **Monitoring**: `ServiceMonitor` + Grafana dashboards.

See `GITOPS.md` and `DEPLOY_YAML.md` for full specs.
