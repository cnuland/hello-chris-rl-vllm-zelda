# Test Strategy

## Levels
- **Unit**: env wrappers, state encoder, predicates, macro executor, evaluator aggregation, reward model.
- **Integration**: end-to-end mock: rollout → export → judge aggregation → reward model → PPO step.
- **Determinism**: fixed seeds, stable episode hashes.
- **Performance**: p50/p95 LLM call latency < 200/400 ms (dialog/puzzle), eval throughput ≥ 100 segments/min/node.
- **GitOps / Kustomize**: `kustomize build` succeeds for `gitops/base/` and all overlays; `kubeconform` validates against OpenShift API schemas.

## Commands (Claude: implement)
- `pytest -q`
- Linters: `ruff`, `black --check`
- Kustomize: `kustomize build gitops/base/ > /dev/null && kustomize build gitops/overlays/dev/ > /dev/null && kustomize build gitops/overlays/staging/ > /dev/null && kustomize build gitops/overlays/prod/ > /dev/null`
- Schema validation: `kustomize build gitops/overlays/dev/ | kubeconform -strict -kubernetes-version 1.28.0`
- Pre-commit hooks configured (includes kustomize build check).

## Artifacts
- JUnit XML for CI; artifacts to `artifacts/` (episode GIFs, judge JSONL).
