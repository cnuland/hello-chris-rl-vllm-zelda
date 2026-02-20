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
const REPLAY_BUFFER_SIZE = 16;

// Replay buffer: stores last N messages so late-joining viewers
// immediately see recent agent activity.
let replayBuffer = [];

// Track connected receivers
let receivers = new Set();

// --- /broadcast endpoint: Python trainer sends data here ---
app.ws("/broadcast", (ws, req) => {
  console.log("[broadcast] Trainer connected");

  ws.on("message", (msg) => {
    try {
      const data = JSON.parse(msg);

      // Add to replay buffer (circular)
      replayBuffer.push(msg);
      if (replayBuffer.length > REPLAY_BUFFER_SIZE) {
        replayBuffer.shift();
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
    console.log("[broadcast] Trainer disconnected");
  });
});

// --- /receive endpoint: Browser clients connect here ---
app.ws("/receive", (ws, req) => {
  receivers.add(ws);
  console.log(`[receive] Client connected (${receivers.size} total)`);

  // Send replay buffer to new client
  for (const msg of replayBuffer) {
    ws.send(msg);
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
    replayBufferSize: replayBuffer.length,
  });
});

// --- Start server ---
app.listen(PORT, () => {
  console.log(`Zelda OoS Map Relay Server running on port ${PORT}`);
  console.log(`  Broadcast: ws://localhost:${PORT}/broadcast`);
  console.log(`  Receive:   ws://localhost:${PORT}/receive`);
  console.log(`  Health:    http://localhost:${PORT}/health`);
});
