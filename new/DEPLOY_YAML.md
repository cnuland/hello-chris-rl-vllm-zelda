# Deployment (Kustomize base specs Claude must generate)

All manifests live under `gitops/base/` and are deployed via ArgoCD (see `GITOPS.md`). Claude generates these into the Kustomize structure, **not** as flat YAML files.

## `gitops/base/llm-inference/`
Three vLLM-backed services, one per model size. Each follows the same pattern as `old/llama4scout-17b-inferenceservice.yaml` but uses Qwen2.5 models.

### Vision model (multimodal)
- **`inferenceservice-vision.yaml`**: KServe `InferenceService` — vLLM for **Qwen2.5-VL-32B-Instruct** (`Qwen/Qwen2.5-VL-32B-Instruct`). Args: `--tensor-parallel-size=2`, `--enable-prefix-caching`, `--block-size=16`, `--gpu-memory-utilization=0.9`, `--max-model-len=4096`, `--max-num-seqs=32`, `--trust-remote-code`. Requires vLLM ≥0.7.2 for Qwen2.5-VL. HF token from Secret `llm-d-hf-token`. Readiness probe `/v1/models` (180s initial); liveness `/health` (300s initial).

### Puzzle model (text, large)
- **`inferenceservice-puzzle.yaml`**: KServe `InferenceService` — vLLM for **Qwen2.5-32B-Instruct** (`Qwen/Qwen2.5-32B-Instruct`). Args: `--tensor-parallel-size=2`, `--enable-prefix-caching`, `--gpu-memory-utilization=0.9`, `--max-model-len=4096`, `--max-num-seqs=64`.

### Dialog + State model (text, small)
- **`inferenceservice-text.yaml`**: KServe `InferenceService` — vLLM for **Qwen2.5-7B-Instruct** (`Qwen/Qwen2.5-7B-Instruct`). Single GPU. Args: `--enable-prefix-caching`, `--gpu-memory-utilization=0.85`, `--max-model-len=4096`, `--max-num-seqs=128`. Serves both `/dialog` and `/state` routes.

### Shared resources
- **`decode-deployment.yaml`**: Standalone vLLM decode `Deployment` (for environments without KServe). Labels: `llm-d.ai/inferenceServing: "true"`, `llm-d.ai/role: decode`. Node selector for `gpu-inference` nodes. Mirrors `old/ms-llm-d-modelservice-decode-deployment.yaml`.
- **`service.yaml`**: `ClusterIP` Services exposing port 8000 (vLLM) per model. Mirrors `old/openshift/llm-deployments/llama4-scout-service.yaml`.
- **`serviceaccount.yaml`** + **`rbac.yaml`**: SA + `ClusterRole`/`ClusterRoleBinding` for pods, services, endpoints, InferencePool access.

## `gitops/base/llm-d-gateway/`
- **`gateway.yaml`**: `Gateway` CR (`gateway.networking.k8s.io/v1`), `gatewayClassName: istio`.
- **`httproute-vision.yaml`**: `HTTPRoute` — path `/vision` → `InferencePool` `qwen25-vl-pool` (Qwen2.5-VL-32B vision model).
- **`httproute-dialog.yaml`**: `HTTPRoute` — path `/dialog` → `InferencePool` `qwen25-text-pool` (Qwen2.5-7B text model).
- **`httproute-puzzle.yaml`**: `HTTPRoute` — path `/puzzle` → `InferencePool` `qwen25-puzzle-pool` (Qwen2.5-32B text model).
- **`httproute-state.yaml`**: `HTTPRoute` — path `/state` → `InferencePool` `qwen25-text-pool` (shares Qwen2.5-7B with dialog).
- **`inference-pool-vision.yaml`**: `InferencePool` for vision model with `PrefixHash` session affinity.
- **`inference-pool-puzzle.yaml`**: `InferencePool` for puzzle model.
- **`inference-pool-text.yaml`**: `InferencePool` for dialog + state (shared 7B model).
- **`epp-deployment.yaml`**: EPP `Deployment` with env: `ENABLE_KVCACHE_AWARE_SCORER: "true"`, `PD_ENABLED: "true"`, `BATCH_MAX_TOKENS: "1024"`, `ROUTING_STRATEGY: "route_by_path"`. Prom + OTel metrics enabled.

## `gitops/base/rl-training/`
**Prerequisite**: KubeRay operator installed on the cluster (via OperatorHub or Helm). Kueue operator installed for job admission.
- **`raycluster.yaml`**: KubeRay `RayCluster` CR — Ray head (1 pod) + N worker group pods. Headless PyBoy (Xvfb), ROM from `ConfigMap`, image `quay.io/cnuland/zelda-kuberay-worker:latest`. Workers: CPU-only (preserves GPUs for LLM). Env vars: `S3_ENDPOINT_URL`, `S3_BUCKET_NAME`, `LLM_ENDPOINT`. Kueue label `kueue.x-k8s.io/queue-name`. Mirrors `old/run-kuberay-zelda.ipynb` ClusterConfiguration.
- **`rayjob.yaml`**: KubeRay `RayJob` CR — submits `scripts/run_rollouts.py` to the RayCluster. Configurable via env vars: `RAY_WORKERS`, `ENVS_PER_WORKER`, `EPISODE_LENGTH`, `BATCH_SIZE`. On completion, exports episodes to MinIO. Mirrors `old/run-ray-zelda.py`.
- **`kueue-localqueue.yaml`**: Kueue `LocalQueue` CR for Ray job admission in the target namespace.
- **`service.yaml`**: `ClusterIP` for Ray dashboard + TensorBoard.
- **`serviceaccount.yaml`** + **`rbac.yaml`**: SA + roles for pods, services, configmaps, rayclusters, metrics access. Mirrors `old/run-kuberay-zelda.ipynb` RBAC requirements.

## `gitops/base/evaluator/`
- **`evaluator-job.yaml`**: Batch `Job` — reads segments from MinIO, fans out to llm-d judge routes, writes `scores.jsonl`.
- **`reward-model-job.yaml`**: Training `Job` — reads `scores.jsonl`, builds preferences, fits reward model, saves `rm.pt` to MinIO.

## `gitops/base/storage/`
- **`minio.yaml`**: MinIO `Deployment` + `Service` + `PVC`. Buckets: `zelda-episodes`, `zelda-models`.

## `gitops/base/networking/`
- **`network-policies.yaml`**: Allow all intra-namespace traffic; allow cross-namespace access to LLM services and inference gateway. Mirrors `old/llm-d-network-policy.yaml`.

## `gitops/base/monitoring/`
- **`servicemonitor.yaml`**: Prometheus `ServiceMonitor` for vLLM, EPP, training, and evaluator pods.
- **`grafana-dashboard-cm.yaml`**: Grafana dashboard JSON as `ConfigMap`.

## Overlay patches (see `GITOPS.md` for full env specs)
- **dev**: 1 LLM replica, RayCluster with 1 worker × 3 envs, reduced resources.
- **staging**: 2 LLM replicas, RayCluster with 3 workers × 6 envs.
- **prod**: 3+ LLM replicas with tensor-parallel=4, RayCluster with 5+ workers × 6 envs (30+ rollouts), ROSA MachineSets for dedicated node pools.
