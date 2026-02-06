# GitOps (OpenShift GitOps + Kustomize)

All cluster resources are managed declaratively under `gitops/` and deployed via the **OpenShift GitOps operator** (ArgoCD). No ad-hoc `oc apply` or shell deploy scripts in production.

## Kustomize Layout

### `gitops/base/`
Shared manifests referenced by every overlay. One sub-directory per component, each with its own `kustomization.yaml`:

- **`llm-inference/`** — Three KServe `InferenceService` CRs (vLLM, prefix caching): **Qwen2.5-VL-32B-Instruct** (vision judge, TP=2), **Qwen2.5-32B-Instruct** (puzzle planner, TP=2), **Qwen2.5-7B-Instruct** (dialog planner + state judge, 1 GPU). Decode `Deployment`, `Service`s, `ServiceAccount`, RBAC. Follows deployment patterns from `old/llama4scout-17b-inferenceservice.yaml` and `old/openshift/llm-deployments/`.
- **`llm-d-gateway/`** — `Gateway` CR, per-route `HTTPRoute` for `/vision`, `/dialog`, `/puzzle`, `/state`, each routing to the appropriate model's `InferencePool`. Three `InferencePool` CRs with `PrefixHash` session affinity (follows patterns from `old/openshift/llm-deployments/llama4-scout-inference-gateway.yaml`). EPP `Deployment` with env: `ENABLE_KVCACHE_AWARE_SCORER`, `PD_ENABLED`, `ROUTING_STRATEGY`.
- **`rl-training/`** — KubeRay `RayCluster` CR (headless PyBoy workers, Xvfb, ROM mount via ConfigMap, MinIO export) + `RayJob` CR for each training burst. Kueue `LocalQueue` for admission. `Service` for Ray dashboard + TensorBoard. `ServiceAccount` + RBAC (including raycluster verbs). Mirrors `old/run-kuberay-zelda.ipynb` and `old/run-ray-zelda.py`.
- **`evaluator/`** — Evaluator batch `Job` (ingests segments from MinIO → llm-d judges → `scores.jsonl`). Reward-model training `Job`.
- **`storage/`** — MinIO `Deployment`, `Service`, `PVC` for episode segments.
- **`networking/`** — `NetworkPolicy` set allowing cross-namespace LLM access, HUD access, inference gateway access. Mirrors `old/llm-d-network-policy.yaml`.
- **`monitoring/`** — `ServiceMonitor` for vLLM + EPP + training metrics. Grafana dashboard `ConfigMap`.

Root `kustomization.yaml` composes all sub-directories.

### `gitops/overlays/`
Environment-specific patches. Each overlay has `kustomization.yaml` + `patches/` directory.

**dev/**
- 1 LLM decode replica (1 GPU); `max-model-len=2048`.
- RayCluster: 1 worker × 3 envs; `RAY_WORKERS=3`, `ENVS_PER_WORKER=3`.
- MinIO ephemeral (emptyDir).
- Namespace: `zelda-rl-dev`.

**staging/**
- 2 LLM decode replicas (1 GPU each); `max-model-len=4096`.
- RayCluster: 3 workers × 6 envs; `RAY_WORKERS=18`, `ENVS_PER_WORKER=6`.
- MinIO with PVC.
- Namespace: `zelda-rl-staging`.

**prod/**
- 3+ LLM decode replicas on dedicated `llm-inference` nodes (4 GPU tensor-parallel per pod).
- RayCluster: 5+ workers × 6 envs = 30+ rollouts on dedicated `pyboy-training` nodes; `RAY_WORKERS=30`, `ENVS_PER_WORKER=6`, `BATCH_SIZE=32768`.
- ROSA `MachineSet` YAMLs (mirrors `old/openshift/machinesets/`): `llm-inference-machineset.yaml` (p4d.xlarge × 3), `pyboy-training-machineset.yaml` (g4dn.4xlarge × 5).
- MinIO with persistent PVC + backup.
- Namespace: `zelda-rl`.

## ArgoCD

### Prerequisites
- OpenShift GitOps operator installed (`openshift-gitops` namespace).
- Cluster admin has granted the ArgoCD service account permissions for the target namespaces.

### `gitops/argocd/project.yaml`
`AppProject` CR:
- **Name**: `zelda-rl`
- **Source repos**: this git repo.
- **Destinations**: `zelda-rl`, `zelda-rl-dev`, `zelda-rl-staging` namespaces + cluster-scoped resources (MachineSets, ClusterRoles).
- **Cluster resource allow list**: `MachineSet`, `ClusterRole`, `ClusterRoleBinding`.

### `gitops/argocd/applications/{dev,staging,prod}.yaml`
Each is an ArgoCD `Application` CR:
- **Source**: `gitops/overlays/{env}` path in this repo.
- **Destination**: the respective namespace on the target cluster.
- **Sync policy**: automated, self-heal enabled, prune enabled.
- **Sync options**: `CreateNamespace=true`, `ServerSideApply=true`.

### Workflow
1. Developer pushes changes to `gitops/base/` or `gitops/overlays/`.
2. ArgoCD detects drift and auto-syncs (or manual sync via UI/CLI for prod).
3. Sync status and health visible in ArgoCD dashboard.
4. Rollback: `argocd app rollback zelda-rl-prod` or revert git commit.

## Validation
- `kustomize build gitops/base/` must succeed.
- `kustomize build gitops/overlays/{env}` must succeed for each env.
- `kubeconform` against OpenShift API schemas for generated manifests.
- ArgoCD dry-run sync before merge for prod changes.
