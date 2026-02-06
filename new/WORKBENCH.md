# OpenShift AI Workbench

## Prerequisites
- OpenShift GitOps has synced the target environment (`oc apply -f gitops/argocd/applications/{env}.yaml` if not already bootstrapped).
- **KubeRay operator** installed on the cluster (via OperatorHub).
- **Kueue operator** installed; `LocalQueue` configured in namespace (deployed by ArgoCD from `gitops/base/rl-training/kueue-localqueue.yaml`).
- LLM inference, llm-d gateway, MinIO, and networking are healthy (ArgoCD dashboard shows green sync).

## Training Loop (iterative)
Each iteration runs Cycle A (RL burst) then Cycle B (LLM evaluator), feeding results back.

### Cycle A — RL Training Burst (KubeRay)
1. Launch notebook image with Python 3.11 + CUDA from OpenShift AI dashboard.
2. `pip install codeflare-sdk ray[rllib]` (+ project `requirements.txt`).
3. Use **CodeFlare SDK** to create a `RayCluster` via `ClusterConfiguration` (mirrors `old/run-kuberay-zelda.ipynb`):
   - Authenticate with `TokenAuthentication` (OpenShift token + server).
   - Configure workers, CPU/memory, image (`quay.io/cnuland/zelda-kuberay-worker:latest`).
   - Set env vars: `S3_ENDPOINT_URL`, `S3_BUCKET_NAME`, `LLM_ENDPOINT`, `RAY_WORKERS`, `ENVS_PER_WORKER`, `BATCH_SIZE`.
   - `cluster.up()` → KubeRay operator provisions the RayCluster.
4. Submit `RayJob` (or `JobSubmissionClient.submit_job`) running `scripts/run_rollouts.py` → Ray RLlib PPO distributes rollouts across workers.
5. Training completes → episodes exported to MinIO as segments (PNG frames + RAM JSONL + manifest).
6. Optionally scale down: `cluster.down()` (or leave cluster warm for the next cycle).

### Cycle B — LLM Evaluator Pass
7. Run `scripts/run_evaluator.py` → reads segments from MinIO → fans out to llm-d judge routes (vision/state/rules) → aggregates scores → writes `scores.jsonl` to MinIO.
8. Run `scripts/train_reward_model.py` → builds pairwise preferences → trains \(R_\phi\) → saves `rm.pt` to MinIO.

### Feedback → Next Cycle A
9. Re-submit a new `RayJob` with updated config: load `rm.pt` for reward shaping + optional SIL/AWR on top-K segments.
10. Repeat from step 4.

## Scaling
- Adjust RayCluster worker count via overlay patches in `gitops/overlays/{env}/patches/training-scale.yaml` and push to git. ArgoCD will reconcile.
- For prod: scale ROSA MachineSet replicas in `gitops/overlays/prod/machinesets/`.
- Env vars `RAY_WORKERS`, `ENVS_PER_WORKER`, `EPISODE_LENGTH`, `BATCH_SIZE` control per-job scale (see `old/PRODUCTION_CLUSTER_CONFIG.md` for reference values).
