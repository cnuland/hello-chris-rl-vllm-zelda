"""Lightweight training dashboard -- Flask app reading epoch data from MinIO.

Runs as a separate Deployment using the same container image as training.
Port 8080 (OpenShift non-root default).

Endpoints:
  GET /           -- HTML dashboard with Chart.js charts
  GET /api/epochs -- All epoch metadata as JSON array
  GET /api/evals  -- All evaluation summaries as JSON array
  GET /healthz    -- Liveness / readiness probe
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

from flask import Flask, jsonify

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# MinIO helpers (with simple time-based cache)
# ---------------------------------------------------------------------------

CACHE_TTL = int(os.getenv("DASHBOARD_CACHE_TTL", "30"))
_cache: dict[str, tuple[object, float]] = {}


def _get_s3():
    from agent.utils.config import S3Config
    from agent.utils.s3 import S3Client
    return S3Client(S3Config())


def _cached(key: str, fetcher):
    now = time.time()
    if key in _cache:
        data, ts = _cache[key]
        if now - ts < CACHE_TTL:
            return data
    try:
        data = fetcher()
    except Exception as exc:
        logger.warning("Fetch %s failed: %s", key, exc)
        if key in _cache:
            return _cache[key][0]
        return []
    _cache[key] = (data, now)
    return data


def fetch_all_epoch_metadata() -> list[dict]:
    """Fetch all epoch metadata, tolerant of gaps."""
    s3 = _get_s3()
    keys = s3.list_keys("zelda-models", prefix="checkpoints/epoch_")
    metadata_keys = sorted(k for k in keys if k.endswith("/metadata.json"))
    results = []
    for key in metadata_keys:
        try:
            data = s3.download_json("zelda-models", key)
            results.append(data)
        except Exception:
            continue
    return results


def fetch_all_eval_summaries() -> list[dict]:
    """Fetch all evaluation summaries, tolerant of gaps."""
    s3 = _get_s3()
    keys = s3.list_keys("zelda-models", prefix="evaluations/epoch_")
    eval_keys = sorted(k for k in keys if k.endswith("/summary.json"))
    results = []
    for key in eval_keys:
        try:
            data = s3.download_json("zelda-models", key)
            results.append(data)
        except Exception:
            continue
    return results


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.route("/api/epochs")
def api_epochs():
    data = _cached("epochs", fetch_all_epoch_metadata)
    return jsonify(data)


@app.route("/api/evals")
def api_evals():
    data = _cached("evals", fetch_all_eval_summaries)
    return jsonify(data)


# ---------------------------------------------------------------------------
# HTML dashboard
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Zelda RL Training Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root {
    --bg: #0d1117;
    --card: #161b22;
    --border: #30363d;
    --text: #c9d1d9;
    --text-muted: #8b949e;
    --accent: #58a6ff;
    --green: #3fb950;
    --red: #f85149;
    --orange: #d29922;
    --purple: #bc8cff;
    --pink: #f778ba;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    font-size: 14px;
    padding: 16px;
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
  }
  header h1 { font-size: 20px; font-weight: 600; }
  header .meta { color: var(--text-muted); font-size: 12px; }
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }
  @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
  }
  .card h2 {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted);
    margin-bottom: 12px;
  }
  .status-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }
  .stat { padding: 8px; background: var(--bg); border-radius: 6px; }
  .stat .label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; }
  .stat .value { font-size: 22px; font-weight: 700; margin-top: 2px; }
  .stat .value.accent { color: var(--accent); }
  .stat .value.green { color: var(--green); }
  .stat .value.orange { color: var(--orange); }
  .milestone-list { list-style: none; }
  .milestone-list li {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid var(--border);
  }
  .milestone-list li:last-child { border-bottom: none; }
  .milestone-list .label { color: var(--text-muted); }
  .milestone-list .value { font-weight: 600; }
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
  }
  .badge.on { background: #1f6f2b; color: var(--green); }
  .badge.off { background: #3d1f1f; color: var(--red); }
  canvas { width: 100% !important; }
  .no-data {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 200px;
    color: var(--text-muted);
    font-style: italic;
  }
</style>
</head>
<body>

<header>
  <h1>Zelda RL Training Dashboard</h1>
  <div class="meta">
    <span id="last-update">Loading...</span>
    &nbsp;|&nbsp; Auto-refresh: 30s
  </div>
</header>

<div class="grid">
  <!-- Status Card -->
  <div class="card">
    <h2>Current Status</h2>
    <div id="status-content" class="no-data">Waiting for training data...</div>
  </div>

  <!-- Milestones Card -->
  <div class="card">
    <h2>Game Milestones (Latest Epoch)</h2>
    <div id="milestones-content" class="no-data">No milestone data</div>
  </div>

  <!-- Reward Chart -->
  <div class="card">
    <h2>Reward Over Time</h2>
    <canvas id="rewardChart"></canvas>
  </div>

  <!-- Exploration Chart -->
  <div class="card">
    <h2>Exploration Progress</h2>
    <canvas id="explorationChart"></canvas>
  </div>

  <!-- Training Dynamics -->
  <div class="card">
    <h2>Training Dynamics</h2>
    <canvas id="trainingChart"></canvas>
  </div>

  <!-- Episode Stats -->
  <div class="card">
    <h2>Episode Stats</h2>
    <canvas id="episodeChart"></canvas>
  </div>
</div>

<script>
const REFRESH_MS = 30000;
const COLORS = {
  blue: '#58a6ff', green: '#3fb950', red: '#f85149',
  orange: '#d29922', purple: '#bc8cff', pink: '#f778ba',
  cyan: '#56d4dd', yellow: '#e3b341',
};

const chartDefaults = {
  responsive: true,
  animation: { duration: 300 },
  plugins: {
    legend: { labels: { color: '#c9d1d9', font: { size: 11 } } },
  },
  scales: {
    x: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
    y: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
  },
};

// ---- Charts ----

const rewardChart = new Chart(document.getElementById('rewardChart'), {
  type: 'line',
  data: { labels: [], datasets: [] },
  options: {
    ...chartDefaults,
    plugins: { ...chartDefaults.plugins, title: { display: false } },
    scales: {
      x: { ...chartDefaults.scales.x, title: { display: true, text: 'Epoch', color: '#8b949e' } },
      y: { ...chartDefaults.scales.y, title: { display: true, text: 'Reward', color: '#8b949e' } },
    },
  },
});

const explorationChart = new Chart(document.getElementById('explorationChart'), {
  type: 'line',
  data: { labels: [], datasets: [] },
  options: {
    ...chartDefaults,
    scales: {
      x: { ...chartDefaults.scales.x, title: { display: true, text: 'Epoch', color: '#8b949e' } },
      rooms: {
        type: 'linear', position: 'left',
        ticks: { color: COLORS.orange }, grid: { color: '#21262d' },
        title: { display: true, text: 'Rooms', color: COLORS.orange },
      },
      tiles: {
        type: 'linear', position: 'right',
        ticks: { color: COLORS.purple }, grid: { drawOnChartArea: false },
        title: { display: true, text: 'Tiles', color: COLORS.purple },
      },
    },
  },
});

const trainingChart = new Chart(document.getElementById('trainingChart'), {
  type: 'line',
  data: { labels: [], datasets: [] },
  options: {
    ...chartDefaults,
    scales: {
      x: { ...chartDefaults.scales.x, title: { display: true, text: 'Epoch', color: '#8b949e' } },
      entropy: {
        type: 'linear', position: 'left',
        ticks: { color: COLORS.cyan }, grid: { color: '#21262d' },
        title: { display: true, text: 'Entropy Coeff', color: COLORS.cyan },
      },
      lr: {
        type: 'linear', position: 'right',
        ticks: { color: COLORS.yellow }, grid: { drawOnChartArea: false },
        title: { display: true, text: 'Learning Rate', color: COLORS.yellow },
      },
    },
  },
});

const episodeChart = new Chart(document.getElementById('episodeChart'), {
  type: 'bar',
  data: { labels: [], datasets: [] },
  options: {
    ...chartDefaults,
    scales: {
      x: { ...chartDefaults.scales.x, title: { display: true, text: 'Epoch', color: '#8b949e' } },
      y: { ...chartDefaults.scales.y, title: { display: true, text: 'Episodes', color: '#8b949e' } },
      score: {
        type: 'linear', position: 'right',
        ticks: { color: COLORS.pink }, grid: { drawOnChartArea: false },
        title: { display: true, text: 'Judge Score', color: COLORS.pink },
        min: 0, max: 1,
      },
    },
  },
});

// ---- Update functions ----

function updateStatus(epochs) {
  const el = document.getElementById('status-content');
  if (!epochs.length) {
    el.className = 'no-data';
    el.textContent = 'Waiting for training data...';
    return;
  }
  el.className = 'status-grid';
  const latest = epochs[epochs.length - 1];
  const sps = latest.elapsed_seconds > 0
    ? (latest.timesteps / latest.elapsed_seconds).toFixed(0) : '?';
  const godBadge = latest.god_mode
    ? '<span class="badge on">ON</span>' : '<span class="badge off">OFF</span>';

  el.innerHTML = `
    <div class="stat"><div class="label">Epoch</div><div class="value accent">${latest.epoch}</div></div>
    <div class="stat"><div class="label">Mean Reward</div><div class="value green">${latest.reward_mean?.toFixed(1) ?? '?'}</div></div>
    <div class="stat"><div class="label">Max Reward</div><div class="value">${latest.reward_max?.toFixed(1) ?? '?'}</div></div>
    <div class="stat"><div class="label">SPS</div><div class="value">${sps}</div></div>
    <div class="stat"><div class="label">Episodes</div><div class="value">${latest.episodes_completed ?? '?'}</div></div>
    <div class="stat"><div class="label">Total Steps</div><div class="value">${(latest.timesteps ?? 0).toLocaleString()}</div></div>
    <div class="stat"><div class="label">Entropy</div><div class="value">${latest.entropy_coeff?.toFixed(4) ?? '?'}</div></div>
    <div class="stat"><div class="label">God Mode</div><div class="value">${godBadge}</div></div>
  `;
}

function updateMilestones(epochs) {
  const el = document.getElementById('milestones-content');
  if (!epochs.length) {
    el.className = 'no-data';
    el.textContent = 'No milestone data';
    return;
  }
  const latest = epochs[epochs.length - 1];
  const ms = latest.milestones || {};
  const nEps = Math.max(latest.episodes_completed || 1, 1);

  el.className = '';
  el.innerHTML = `<ul class="milestone-list">
    <li><span class="label">Max Rooms Explored</span><span class="value">${ms.max_rooms ?? 0}</span></li>
    <li><span class="label">Max Tiles Explored</span><span class="value">${ms.max_tiles ?? 0}</span></li>
    <li><span class="label">Got Sword</span><span class="value">${ms.total_got_sword ?? 0} / ${nEps} eps</span></li>
    <li><span class="label">Visited Maku Tree</span><span class="value">${ms.total_visited_maku_tree ?? 0} / ${nEps} eps</span></li>
    <li><span class="label">Maku Tree Dialog</span><span class="value">${ms.total_maku_dialog ?? 0} / ${nEps} eps</span></li>
    <li><span class="label">Got Gnarled Key</span><span class="value">${ms.total_gnarled_key ?? 0} / ${nEps} eps</span></li>
    <li><span class="label">Entered Dungeon</span><span class="value">${ms.total_entered_dungeon ?? 0} / ${nEps} eps</span></li>
    <li><span class="label">Max Essences</span><span class="value">${ms.max_essences ?? 0}</span></li>
    <li><span class="label">Max Dungeon Keys</span><span class="value">${ms.max_dungeon_keys ?? 0}</span></li>
  </ul>`;
}

function updateRewardChart(epochs) {
  if (!epochs.length) return;
  const labels = epochs.map(e => e.epoch);
  rewardChart.data.labels = labels;
  rewardChart.data.datasets = [
    { label: 'Mean', data: epochs.map(e => e.reward_mean), borderColor: COLORS.blue, backgroundColor: COLORS.blue + '33', fill: true, tension: 0.3 },
    { label: 'Max', data: epochs.map(e => e.reward_max), borderColor: COLORS.green, borderDash: [5, 3], tension: 0.3 },
    { label: 'Min', data: epochs.map(e => e.reward_min), borderColor: COLORS.red, borderDash: [5, 3], tension: 0.3 },
  ];
  rewardChart.update();
}

function updateExplorationChart(epochs) {
  if (!epochs.length) return;
  const labels = epochs.map(e => e.epoch);
  explorationChart.data.labels = labels;
  explorationChart.data.datasets = [
    { label: 'Max Rooms', data: epochs.map(e => (e.milestones || {}).max_rooms || 0), borderColor: COLORS.orange, yAxisID: 'rooms', tension: 0.3 },
    { label: 'Max Tiles', data: epochs.map(e => (e.milestones || {}).max_tiles || 0), borderColor: COLORS.purple, yAxisID: 'tiles', tension: 0.3 },
  ];
  explorationChart.update();
}

function updateTrainingChart(epochs) {
  if (!epochs.length) return;
  const labels = epochs.map(e => e.epoch);
  trainingChart.data.labels = labels;
  trainingChart.data.datasets = [
    { label: 'Entropy Coeff', data: epochs.map(e => e.entropy_coeff), borderColor: COLORS.cyan, yAxisID: 'entropy', tension: 0.3 },
    { label: 'Learning Rate', data: epochs.map(e => e.learning_rate), borderColor: COLORS.yellow, yAxisID: 'lr', tension: 0.3 },
  ];
  trainingChart.update();
}

function updateEpisodeChart(epochs, evals) {
  if (!epochs.length) return;
  const labels = epochs.map(e => e.epoch);
  const datasets = [
    { label: 'Episodes', data: epochs.map(e => e.episodes_completed || 0), backgroundColor: COLORS.blue + '99', yAxisID: 'y' },
  ];
  if (evals && evals.length) {
    const scoreMap = {};
    evals.forEach(ev => { scoreMap[ev.epoch] = ev.mean_weighted_score; });
    datasets.push({
      label: 'Judge Score',
      type: 'line',
      data: labels.map(ep => scoreMap[ep] ?? null),
      borderColor: COLORS.pink,
      yAxisID: 'score',
      tension: 0.3,
      spanGaps: true,
    });
  }
  episodeChart.data.labels = labels;
  episodeChart.data.datasets = datasets;
  episodeChart.update();
}

// ---- Data fetching ----

async function refresh() {
  try {
    const [epochsResp, evalsResp] = await Promise.all([
      fetch('/api/epochs'), fetch('/api/evals'),
    ]);
    const epochs = await epochsResp.json();
    const evals = await evalsResp.json();

    updateStatus(epochs);
    updateMilestones(epochs);
    updateRewardChart(epochs);
    updateExplorationChart(epochs);
    updateTrainingChart(epochs);
    updateEpisodeChart(epochs, evals);

    document.getElementById('last-update').textContent =
      'Updated: ' + new Date().toLocaleTimeString();
  } catch (err) {
    console.error('Refresh failed:', err);
    document.getElementById('last-update').textContent = 'Update failed';
  }
}

refresh();
setInterval(refresh, REFRESH_MS);
</script>
</body>
</html>"""


@app.route("/")
def index():
    return DASHBOARD_HTML, 200, {"Content-Type": "text/html"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("DASHBOARD_PORT", "8080"))
    logger.info("Starting dashboard on port %d", port)
    app.run(host="0.0.0.0", port=port)
