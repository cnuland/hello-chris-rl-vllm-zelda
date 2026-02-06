# Risks & Mitigations

- Judge drift → calibration set + deterministic decoding + self-consistency.
- Reward hacking → potential-based shaping + audit rationales.
- Cost spikes → batch segments; adaptive M for self-consistency.
- Config drift → ArgoCD auto-sync + self-heal; no manual `oc apply` in staging/prod.
- Bad manifest merge → `kustomize build` + `kubeconform` in CI pre-merge; ArgoCD dry-run for prod.
- MachineSet mis-scale → prod overlay changes require two reviewers; ArgoCD manual sync for prod.
