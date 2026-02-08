# Zelda RLAIF Agent — Hybrid RL + LLM on OpenShift AI

A reinforcement learning agent that plays *Zelda: Oracle of Seasons* on a Game Boy emulator, guided by LLM judge models that score its gameplay and feed improvements back into training. The entire system runs distributed on OpenShift using KubeRay for RL training and Red Hat OpenShift AI (RHOAI) with llm-d for LLM inference.

## Approach

The agent uses a three-phase feedback loop that runs continuously:

```
   Train (PPO)          Evaluate (LLM Judges)       Improve
  +-----------+        +-------------------+       +-------------+
  | PyBoy     |  eps   | qwen25-7b (state) |  RM   | Next epoch  |
  | emulators +------->| qwen25-32b (puzzle)+----->| uses reward  |
  | on Ray    |  to    | qwen25-vl (vision) | .pt  | model for   |
  | workers   | MinIO  |                   | to   | shaping     |
  +-----------+        +-------------------+ MinIO +-------------+
       ^                                                 |
       +-------------------------------------------------+
```

### Phase 1: RL Training

PPO (Proximal Policy Optimization) runs on a KubeRay cluster with multiple workers, each running a headless PyBoy Game Boy emulator. The agent receives a 128-D observation vector built from RAM state (player position, room ID, health, dialog flags, inventory) and outputs one of 8 actions (d-pad, A, B, Start, NOP).

Rewards come from multiple sources:
- **Game events** — room transitions, health changes, rupees, keys, sword upgrades, dungeon essences, death penalties
- **Coverage** — tile-level and room-level exploration bonuses with revisit penalties
- **RND curiosity** — Random Network Distillation provides intrinsic motivation for novel states
- **RLAIF shaping** — potential-based reward shaping from the Bradley-Terry reward model trained by LLM judges (available from epoch 1 onward)

Episodes export frame PNGs and JSONL state logs to MinIO as 300-frame segments for the judges.

### Phase 2: LLM Judge Evaluation

Three Qwen2.5 models score each episode segment with M=3 self-consistency voting:

| Judge | Model | Scores | Input |
|-------|-------|--------|-------|
| State | qwen25-7b | progress, dialog, efficiency | JSONL state data |
| Puzzle | qwen25-32b | puzzle-solving skill | JSONL state + puzzle flags |
| Vision | qwen25-vl-32b | novelty / exploration | PNG screenshots + state |

Rubric weights: progress (0.4), dialog (0.2), puzzle (0.2), novelty (0.1), efficiency (0.1).

Scores are written to `s3://zelda-episodes/scores/epoch_N/scores.jsonl`.

### Phase 3: Reward Model Update

A Bradley-Terry reward model (3-layer MLP, 128 -> 64 -> 64 -> 1) is trained on pairwise preferences derived from the judge scores. The model learns P(segment A > segment B) = sigmoid(R(A) - R(B)). Each epoch's model is saved to MinIO and loaded by the next training epoch for potential-based reward shaping, closing the RLAIF feedback loop.

## Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| RL framework | [Ray RLlib](https://docs.ray.io/en/latest/rllib/) (PPO) | Distributed policy optimization |
| Cluster orchestration | [KubeRay](https://ray-project.github.io/kuberay/) | RayCluster + RayJob management on K8s |
| Game emulator | [PyBoy](https://github.com/Baekalfen/PyBoy) | Headless Game Boy emulation |
| Gym interface | [Gymnasium](https://gymnasium.farama.org/) | Standard RL environment API |
| LLM inference | [Red Hat OpenShift AI](https://www.redhat.com/en/technologies/cloud-computing/openshift/openshift-ai) (RHOAI 3) | LLMInferenceService CRD with vLLM backends |
| LLM load balancing | [llm-d](https://github.com/llm-d/llm-d) (Gateway API + InferencePool) | Request routing with KV-cache and prefix-cache scoring |
| LLM models | Qwen2.5-7B, Qwen2.5-32B, Qwen2.5-VL-32B | State, puzzle, and vision judges |
| Object storage | [MinIO](https://min.io/) | Episode data, checkpoints, reward models |
| Policy network | [PyTorch](https://pytorch.org/) | MLP policy + reward model |
| GitOps | [OpenShift GitOps](https://docs.openshift.com/gitops/) (ArgoCD) + Kustomize | Declarative infrastructure management |
| Container platform | [Red Hat OpenShift](https://www.redhat.com/en/technologies/cloud-computing/openshift) (ROSA) | Managed Kubernetes with GPU support |
| GPU management | NVIDIA GPU Operator + Node Feature Discovery | A100 GPU scheduling and allocation |

## Project Structure

```
agent/
  env/
    zelda_env.py          # Gymnasium wrapper around PyBoy
    reward_wrapper.py     # Composite reward: events + coverage + RND + RLAIF shaping
    state_encoder.py      # RAM -> 128-D float32 observation vector
  evaluator/
    ingest.py             # Batch segments through 3 LLM judges with M=3 voting
    exporter.py           # Episode segment export (PNGs + JSONL) to MinIO
    reward_model.py       # Bradley-Terry reward model + preference builder
  planner/
    llm_client.py         # HTTP client for llm-d gateway (OpenAI-compatible)
  rl/
    trainer.py            # PPO config builder for Ray RLlib
    model.py              # Custom MLP policy network (256 hidden)
    rewards.py            # CoverageReward, RNDCuriosity, PotentialShaping
  utils/
    config.py             # Pydantic configs (S3, training, rewards, LLM)
    s3.py                 # MinIO/S3 client wrapper

scripts/
  run_pipeline.py         # Full RLAIF loop: train -> eval -> feedback (long-running)
  run_rollouts.py         # Single training epoch
  run_evaluator.py        # Single evaluation pass
  run_epoch.py            # Epoch orchestrator (external job submission)

gitops/
  argocd/                 # ArgoCD Application + AppProject CRs
  base/                   # Kustomize base: RayCluster, LLMInferenceService, MinIO, RBAC
  overlays/dev|staging|prod  # Environment patches (replicas, GPU, storage)
  cluster/                # Cluster-wide operator subscriptions
```

## Prerequisites

- OpenShift cluster (ROSA or self-managed) with GPU nodes (tested on p4d.24xlarge with A100s)
- Red Hat OpenShift AI (RHOAI) 3.x installed with distributed inference (llm-d) enabled
- NVIDIA GPU Operator and Node Feature Discovery installed
- KubeRay operator installed
- `oc` CLI authenticated to the cluster
- `podman` for building container images
- A Zelda: Oracle of Seasons ROM file (`zelda.gbc`) placed in `new/ignored/`

## Installation

### 1. Clone and install Python dependencies

```bash
git clone https://github.com/cnuland/hello-chris-rl-zelda.git
cd hello-chris-rl-zelda
pip install -e ".[dev]"
```

### 2. Build and push the worker container image

The container packages the agent code, PyBoy, Ray, and the ROM into a single image used by the RayCluster head and worker pods.

```bash
# Place ROM files in new/ignored/
cp /path/to/zelda.gbc new/ignored/zelda.gbc
# If you have a save state:
cp /path/to/zelda.gbc.state new/ignored/zelda.gbc.state

# Build and push
podman build -t quay.io/<your-org>/zelda-kuberay-worker:latest -f Containerfile .
podman push quay.io/<your-org>/zelda-kuberay-worker:latest
```

Update the image reference in `gitops/base/rl-training/raycluster.yaml` to match your registry.

### 3. Deploy infrastructure via GitOps

```bash
# Create the namespace
oc new-project zelda-rl

# Apply ArgoCD project and application
oc apply -f gitops/argocd/project.yaml
oc apply -f gitops/argocd/applications/dev.yaml   # or staging/prod

# Or apply directly with Kustomize
oc apply -k gitops/overlays/dev
```

This deploys:
- **RayCluster** (`zelda-rl`) with head + worker pods running the emulator image
- **LLMInferenceServices** for qwen25-7b, qwen25-32b, qwen25-vl-32b
- **MinIO** for episode storage and model checkpoints
- **Gateway API** routing for llm-d inference load balancing
- **RBAC**, ServiceAccounts, NetworkPolicies

### 4. Verify the deployment

```bash
# Check RayCluster is ready
oc get raycluster -n zelda-rl

# Check LLM models are serving
oc get llminferenceservice -n zelda-rl

# Check MinIO is running
oc get pods -n zelda-rl -l app=minio

# Verify GPU allocation
oc describe nodes -l nvidia.com/gpu.present=true | grep -A5 "Allocated resources"
```

## Running the Pipeline

### Full RLAIF pipeline (recommended)

Submit a long-running RayJob that handles the complete train/eval/feedback loop:

```bash
cat << 'EOF' | oc apply -f -
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: zelda-pipeline
  namespace: zelda-rl
spec:
  entrypoint: python scripts/run_pipeline.py
  runtimeEnvYAML: |
    env_vars:
      RUN_HOURS: "48"
      EPOCH_STEPS: "300000"
      RAY_WORKERS: "3"
      ENVS_PER_WORKER: "1"
      EPISODE_LENGTH: "30000"
      BATCH_SIZE: "4096"
      EVAL_INTERVAL: "1"
      S3_ENDPOINT_URL: "http://minio-api.zelda-rl.svc.cluster.local:9000"
      S3_ACCESS_KEY: "admin"
      S3_SECRET_KEY: "zelda-rl-minio-2024"
      LLM_NAMESPACE: "zelda-rl"
      LLM_USE_DIRECT: "true"
  shutdownAfterJobFinishes: false
  ttlSecondsAfterFinished: 86400
  submitterPodTemplate:
    spec:
      restartPolicy: Never
      containers:
        - name: ray-job-submitter
          image: quay.io/<your-org>/zelda-kuberay-worker:latest
          resources:
            requests:
              cpu: "500m"
              memory: "200Mi"
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      serviceAccountName: zelda-rl-training
  clusterSelector:
    ray.io/cluster: zelda-rl
EOF
```

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RUN_HOURS` | 48 | Total pipeline runtime |
| `EPOCH_STEPS` | 300,000 | Environment steps per training epoch |
| `EPISODE_LENGTH` | 30,000 | Steps per episode (game frames = steps x frame_skip) |
| `RAY_WORKERS` | 3 | Parallel PyBoy emulator workers |
| `ENVS_PER_WORKER` | 1 | Environments per worker (1 PyBoy per worker) |
| `BATCH_SIZE` | 4096 | PPO train batch size |
| `EVAL_INTERVAL` | 1 | Run evaluation every N epochs |
| `LLM_USE_DIRECT` | true | Use direct workload service URLs (bypasses gateway) |

### Single training epoch

```bash
cat << 'EOF' | oc apply -f -
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: zelda-train-epoch-0
  namespace: zelda-rl
spec:
  entrypoint: python scripts/run_rollouts.py
  runtimeEnvYAML: |
    env_vars:
      EPOCH: "0"
      EPOCH_STEPS: "300000"
      RAY_WORKERS: "3"
      EPISODE_LENGTH: "30000"
      BATCH_SIZE: "4096"
  submitterPodTemplate:
    spec:
      restartPolicy: Never
      containers:
        - name: ray-job-submitter
          image: quay.io/<your-org>/zelda-kuberay-worker:latest
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      serviceAccountName: zelda-rl-training
  clusterSelector:
    ray.io/cluster: zelda-rl
EOF
```

### Single evaluation pass

```bash
cat << 'EOF' | oc apply -f -
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: zelda-eval-epoch-0
  namespace: zelda-rl
spec:
  entrypoint: python scripts/run_evaluator.py
  runtimeEnvYAML: |
    env_vars:
      EPOCH: "0"
      S3_ENDPOINT_URL: "http://minio-api.zelda-rl.svc.cluster.local:9000"
      S3_ACCESS_KEY: "admin"
      S3_SECRET_KEY: "zelda-rl-minio-2024"
      LLM_NAMESPACE: "zelda-rl"
      LLM_USE_DIRECT: "true"
  submitterPodTemplate:
    spec:
      restartPolicy: Never
      containers:
        - name: ray-job-submitter
          image: quay.io/<your-org>/zelda-kuberay-worker:latest
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      serviceAccountName: zelda-rl-training
  clusterSelector:
    ray.io/cluster: zelda-rl
EOF
```

## Monitoring

### Job logs

```bash
# Get the Ray job ID
oc get rayjob -n zelda-rl

# Stream logs from the head pod
HEAD_POD=$(oc get pods -n zelda-rl -l ray.io/node-type=head -o name)
JOB_ID=$(oc get rayjob zelda-pipeline -n zelda-rl -o jsonpath='{.status.jobId}')
oc exec -n zelda-rl $HEAD_POD -- ray job logs $JOB_ID
```

### Training progress

The pipeline writes a live progress file to MinIO:

```bash
# Port-forward MinIO
oc port-forward -n zelda-rl svc/minio-api 9000:9000 &

# Check progress
aws --endpoint-url http://localhost:9000 s3 cp s3://zelda-models/pipeline/progress.json -
```

### MinIO data

```bash
# List episode segments
aws --endpoint-url http://localhost:9000 s3 ls s3://zelda-episodes/ --recursive | head

# List checkpoints and reward models
aws --endpoint-url http://localhost:9000 s3 ls s3://zelda-models/ --recursive

# Download evaluation summary
aws --endpoint-url http://localhost:9000 s3 cp s3://zelda-models/evaluations/epoch_0/summary.json -
```

### LLM model health

```bash
# Check all LLMInferenceService status
oc get llminferenceservice -n zelda-rl

# Test inference directly
oc exec -n zelda-rl $HEAD_POD -- curl -sk \
  https://qwen25-7b-kserve-workload-svc.zelda-rl.svc.cluster.local:8000/v1/models
```

## MinIO Bucket Layout

| Bucket | Contents |
|--------|----------|
| `zelda-episodes` | Episode segments: `{episode_id}/{segment_id}/manifest.json`, `states.jsonl`, `frames/*.png` |
| `zelda-episodes` | Judge scores: `scores/epoch_N/scores.jsonl` |
| `zelda-models` | Checkpoints: `checkpoints/epoch_N/` (policy_state.pkl, algorithm_state.pkl) |
| `zelda-models` | Reward models: `reward_model/epoch_N/rm.pt` |
| `zelda-models` | Eval summaries: `evaluations/epoch_N/summary.json` |
| `zelda-models` | Live progress: `pipeline/progress.json` |
