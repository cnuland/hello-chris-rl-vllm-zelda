# Metrics & Dashboards

- **Control**: `mode_active{controller,dialog,puzzle}`, takeover success, giveback reasons, stuck watchdog trips.
- **Exploration**: unique rooms/10k steps; unique tiles/room; doorway ping-pong count.
- **LLM**: latency histograms by route; token usage; KV-cache hit rate; session affinity hits.
- **Evaluator**: agreement rate; self-consistency delta; cost/100 segments.
- **Learning**: episodic return; PPO KL, entropy; reward model correlation to milestones.
- **GitOps**: ArgoCD sync status per app (`Synced`/`OutOfSync`/`Degraded`); last sync time; drift count; rollback count.

Provide a Grafana dashboard JSON (`gitops/base/monitoring/grafana-dashboard-cm.yaml`) with panels for each category.
`ServiceMonitor` CRs in `gitops/base/monitoring/` scrape vLLM, EPP, training, and evaluator pods.
