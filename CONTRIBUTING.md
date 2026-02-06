# Contributing

- Use Python 3.11; prefer type hints; run `ruff`, `black`, and tests locally.
- Keep YAML declarative; no ad-hoc Python deploy scripts or `oc apply` in production.
- Any evaluator/judge prompt change → bump rubric version & run calibration.

## GitOps Workflow
- All cluster manifests live under `gitops/`. Edit base or overlay YAMLs and push to git.
- Run `kustomize build gitops/overlays/{env}` locally before pushing to validate.
- ArgoCD auto-syncs dev/staging; prod requires manual sync or PR approval.
- Never apply manifests directly with `oc apply` in staging/prod — let ArgoCD manage state.
- MachineSet changes (prod only) require two reviewers.
