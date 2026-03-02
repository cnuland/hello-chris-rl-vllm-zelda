/**
 * WebSocket relay server for Oracle of Seasons RL map visualization.
 *
 * Receives position telemetry from the Python RL training process
 * (via StreamWrapper) and forwards it to all connected browser clients
 * (the PixiJS map visualizer).
 *
 * Architecture inspired by:
 *   - LinkMapViz WebSocket relay (https://github.com/Xe-Xo/LinkMapViz)
 *     Original pub/sub relay architecture for Link's Awakening DX.
 *   - PokemonRedExperiments (https://github.com/PWhiddy/PokemonRedExperiments)
 *     WebSocket streaming concept for RL agent visualization.
 *
 * Endpoints:
 *   /broadcast  - Python trainer sends position data here
 *   /receive    - Browser clients connect here to receive data
 *   /health     - Health check endpoint
 *
 * License: MIT
 */

const path = require("path");
const express = require("express");
const expressWs = require("express-ws");

const app = express();
expressWs(app);

const PORT = process.env.PORT || 3344;

// Serve the built PixiJS web app from ./public
app.use(express.static(path.join(__dirname, "public")));

// Latest state per agent: stores the most recent message from each env_id
// so that newly connected browsers immediately see ALL agents' positions.
let latestByAgent = new Map();

// Track connected receivers
let receivers = new Set();

// Track connected broadcasters (trainers) for epoch-review detection
let broadcasters = new Set();

/**
 * Broadcast a system status message to all connected browser clients.
 */
function broadcastStatus(state, reason) {
  const msg = JSON.stringify({ name: "status", state, reason: reason || "" });
  for (const receiver of receivers) {
    if (receiver.readyState === 1) {
      receiver.send(msg);
    }
  }
}

// --- /broadcast endpoint: Python trainer sends data here ---
app.ws("/broadcast", (ws, req) => {
  const wasEmpty = broadcasters.size === 0;
  broadcasters.add(ws);
  console.log(`[broadcast] Trainer connected (${broadcasters.size} active)`);

  // If trainers just came back from an epoch review, notify browsers
  if (wasEmpty) {
    broadcastStatus("active");
  }

  ws.on("message", (msg) => {
    try {
      const data = JSON.parse(msg);

      // Store latest message per env_id for instant replay on connect
      const envId = data.metadata?.env_id;
      if (envId !== undefined) {
        latestByAgent.set(envId, msg);
      }

      // Forward to all connected receivers
      for (const receiver of receivers) {
        if (receiver.readyState === 1) {
          // OPEN
          receiver.send(msg);
        }
      }
    } catch (e) {
      console.error("[broadcast] Invalid message:", e.message);
    }
  });

  ws.on("close", () => {
    broadcasters.delete(ws);
    console.log(`[broadcast] Trainer disconnected (${broadcasters.size} active)`);

    // When all trainers disconnect, an epoch is being reviewed
    if (broadcasters.size === 0) {
      broadcastStatus(
        "epoch_review",
        "Epoch complete — reviewing results and preparing next epoch…"
      );
    }
  });
});

// --- /receive endpoint: Browser clients connect here ---
app.ws("/receive", (ws, req) => {
  receivers.add(ws);
  console.log(`[receive] Client connected (${receivers.size} total)`);

  // Replay latest state for every known agent — the browser gets all
  // cursor positions immediately instead of waiting for each env to flush.
  for (const msg of latestByAgent.values()) {
    ws.send(msg);
  }

  // If no trainers are currently connected, immediately tell the new
  // browser client that an epoch review is in progress.
  if (broadcasters.size === 0) {
    const status = JSON.stringify({
      name: "status",
      state: "epoch_review",
      reason: "Epoch complete — reviewing results and preparing next epoch…",
    });
    ws.send(status);
  }

  ws.on("close", () => {
    receivers.delete(ws);
    console.log(`[receive] Client disconnected (${receivers.size} total)`);
  });
});

// --- Health check ---
app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    receivers: receivers.size,
    agents: latestByAgent.size,
  });
});

// --- Debug: dump latest position per agent ---
app.get("/debug/positions", (req, res) => {
  const positions = {};
  for (const [envId, msg] of latestByAgent) {
    try {
      const data = JSON.parse(msg);
      const last = data.pos_data?.[data.pos_data.length - 1];
      if (last) {
        positions[envId] = {
          x: last.x, y: last.y,
          fx: last.fx, fy: last.fy,
          z: last.z,
          room_id: last.room_id,
          tile_x: last.tile_x, tile_y: last.tile_y,
          direction: last.direction,
        };
      }
    } catch (e) { /* skip */ }
  }
  res.json(positions);
});

// --- Start server ---
app.listen(PORT, () => {
  console.log(`Zelda OoS Map Relay Server running on port ${PORT}`);
  console.log(`  Broadcast: ws://localhost:${PORT}/broadcast`);
  console.log(`  Receive:   ws://localhost:${PORT}/receive`);
  console.log(`  Health:    http://localhost:${PORT}/health`);
});
