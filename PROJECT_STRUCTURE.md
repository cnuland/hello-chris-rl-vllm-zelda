# Project Structure

```
hello-chris-rl-zelda/
├─ README.md
├─ PROJECT_STRUCTURE.md
├─ CLAUDE_TASKS.md
├─ TESTING.md
├─ CONTRIBUTING.md
├─ GOVERNANCE.md
├─ LICENSE.md
├─ old/                          # Previous baseline (read-only reference)
│  ├─ README.md
│  ├─ NOTES.md
│  ├─ openshift/                 # Old flat YAML deployments
│  └─ ...
├─ new/                          # Authoritative specs for the new hybrid system
│  ├─ README.md
│  ├─ OVERVIEW.md
│  ├─ ARCHITECTURE.md
│  ├─ GITOPS.md
│  ├─ POLICY_SWITCHING.md
│  ├─ STATE_ENCODER.md
│  ├─ SCHEMAS.md
│  ├─ REWARD_SHAPING.md
│  ├─ EXPLORATION.md
│  ├─ EVAL_PIPELINE.md
│  ├─ JUDGES_RUBRIC.md
│  ├─ LLM_D_CONFIG.md
│  ├─ DEPLOY_YAML.md
│  ├─ WORKBENCH.md
│  ├─ METRICS.md
│  ├─ RISKS.md
│  ├─ ROADMAP.md
│  └─ PROMPTS/
│     ├─ VISION_PROMPT.md
│     ├─ DIALOG_PROMPT.md
│     └─ PUZZLE_PROMPT.md
└─ gitops/                       # Kustomize + ArgoCD (OpenShift GitOps)
   ├─ argocd/
   │  ├─ project.yaml            # AppProject CR
   │  └─ applications/
   │     ├─ dev.yaml             # ArgoCD Application — dev overlay
   │     ├─ staging.yaml         # ArgoCD Application — staging overlay
   │     └─ prod.yaml            # ArgoCD Application — prod overlay
   ├─ base/
   │  ├─ kustomization.yaml
   │  ├─ namespace.yaml
   │  ├─ llm-inference/          # KServe InferenceService + vLLM decode Deployments
   │  │  ├─ kustomization.yaml
   │  │  ├─ inferenceservice.yaml
   │  │  ├─ decode-deployment.yaml
   │  │  ├─ service.yaml
   │  │  ├─ serviceaccount.yaml
   │  │  └─ rbac.yaml
   │  ├─ llm-d-gateway/          # Inference Gateway + EPP + HTTPRoutes
   │  │  ├─ kustomization.yaml
   │  │  ├─ gateway.yaml
   │  │  ├─ httproute-vision.yaml
   │  │  ├─ httproute-dialog.yaml
   │  │  ├─ httproute-puzzle.yaml
   │  │  ├─ inference-pool.yaml
   │  │  └─ epp-deployment.yaml
   │  ├─ rl-training/            # KubeRay RayCluster + RayJob for RL training bursts
   │  │  ├─ kustomization.yaml
   │  │  ├─ raycluster.yaml
   │  │  ├─ rayjob.yaml
   │  │  ├─ kueue-localqueue.yaml
   │  │  ├─ service.yaml
   │  │  ├─ serviceaccount.yaml
   │  │  └─ rbac.yaml
   │  ├─ evaluator/              # Evaluator batch Job + reward model Job
   │  │  ├─ kustomization.yaml
   │  │  ├─ evaluator-job.yaml
   │  │  └─ reward-model-job.yaml
   │  ├─ storage/                # MinIO for episodes/segments
   │  │  ├─ kustomization.yaml
   │  │  └─ minio.yaml
   │  ├─ networking/
   │  │  ├─ kustomization.yaml
   │  │  └─ network-policies.yaml
   │  └─ monitoring/
   │     ├─ kustomization.yaml
   │     ├─ servicemonitor.yaml
   │     └─ grafana-dashboard-cm.yaml
   └─ overlays/
      ├─ dev/
      │  ├─ kustomization.yaml
      │  └─ patches/
      │     ├─ llm-replicas.yaml
      │     └─ training-scale.yaml
      ├─ staging/
      │  ├─ kustomization.yaml
      │  └─ patches/
      │     ├─ llm-replicas.yaml
      │     └─ training-scale.yaml
      └─ prod/
         ├─ kustomization.yaml
         ├─ machinesets/
         │  ├─ llm-inference-machineset.yaml
         │  └─ pyboy-training-machineset.yaml
         └─ patches/
            ├─ llm-replicas.yaml
            └─ training-scale.yaml
```
