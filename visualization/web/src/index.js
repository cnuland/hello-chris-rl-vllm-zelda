/**
 * Main entry point for the Oracle of Seasons RL map visualizer.
 *
 * Connects to the WebSocket relay server to receive real-time position
 * telemetry from the RL training process, and renders it on a full
 * overworld map using PixiJS.
 *
 * Architecture inspired by:
 *   - LinkMapViz (https://github.com/Xe-Xo/LinkMapViz)
 *     Real-time PixiJS visualization for Link's Awakening DX RL agents.
 *   - PokemonRedExperiments (https://github.com/PWhiddy/PokemonRedExperiments)
 *     WebSocket streaming and overworld map overlay concept.
 *
 * Memory addresses and coordinate system from oracles-disasm:
 *   https://github.com/Stewmath/oracles-disasm
 *
 * Coordinate system:
 *   world_x = (room_id % 16) * 10 + tile_x   (0-159)
 *   world_y = (room_id // 16) * 8  + tile_y   (0-127)
 *   Pixel on map: (world_x * 16, world_y * 16) on a 2560x2048 image
 *
 * License: MIT
 */

import "./style.css";
import { Application, Assets } from "pixi.js";
import { MapPanel } from "./map_panel.js";
import { SidePanel } from "./side_panel.js";

// --- Configuration ---
// Auto-detect wss:// when served over HTTPS (e.g. behind OpenShift Route)
const _wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
const WS_URL =
  new URLSearchParams(window.location.search).get("ws") ||
  `${_wsProto}//${window.location.host}/receive`;
const PROCESS_INTERVAL_MS = 20; // 50Hz processing rate
const RECONNECT_INTERVAL_MS = 2000;

// --- State ---
let app;
let mapPanel;
let sidePanel;
let ws = null;
let dataStream = [];
let totalReceived = 0;
let statusEl;

async function init() {
  // Create PixiJS application
  app = new Application();
  await app.init({
    background: "#111111",
    resizeTo: window,
    antialias: false,
    resolution: window.devicePixelRatio || 1,
    autoDensity: true,
  });

  document.getElementById("app").appendChild(app.canvas);
  statusEl = document.getElementById("status");

  // Preload assets
  await Assets.load([
    { alias: "bg_overworld", src: "assets/overworld.png" },
    { alias: "link_sprites", src: "assets/link_sprites.png" },
  ]);

  // Create UI panels
  mapPanel = new MapPanel(app);
  mapPanel.addBackground("bg_overworld");

  sidePanel = new SidePanel(app, window.innerWidth, window.innerHeight);

  // Center the map initially
  mapPanel.container.x = window.innerWidth / 2 - 2560 / 2;
  mapPanel.container.y = window.innerHeight / 2 - 2048 / 2;

  // Start processing loop
  setInterval(processDataStream, PROCESS_INTERVAL_MS);
  setInterval(reconnectWebSocket, RECONNECT_INTERVAL_MS);

  // Initial connection
  connectWebSocket();

  updateStatus("Connected to visualizer. Waiting for data...");
}

function connectWebSocket() {
  if (ws && ws.readyState <= 1) return; // Already connected or connecting

  try {
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      updateStatus(`Connected to ${WS_URL}`);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.pos_data && Array.isArray(data.pos_data)) {
          // Attach metadata (env_id, color) to each element so we can
          // identify which agent produced it downstream.
          const meta = data.metadata || {};
          for (const element of data.pos_data) {
            element._envId = meta.env_id ?? "unknown";
            element._color = meta.color || "";
            element._direction = element.direction ?? 2; // default: facing down
            dataStream.push(element);
          }
          totalReceived += data.pos_data.length;
        }
      } catch (e) {
        console.warn("Invalid message:", e);
      }
    };

    ws.onclose = () => {
      updateStatus("Disconnected. Reconnecting...");
      ws = null;
    };

    ws.onerror = () => {
      ws = null;
    };
  } catch (e) {
    updateStatus(`Connection failed: ${e.message}`);
    ws = null;
  }
}

function reconnectWebSocket() {
  if (!ws || ws.readyState > 1) {
    connectWebSocket();
  }
}

/**
 * Process incoming data stream at 50Hz.
 * Rate-limits to avoid overwhelming the renderer.
 * Inspired by LinkMapViz's refreshNotifications() approach.
 */
function processDataStream() {
  // Update existing position rectangles (fade/destroy)
  mapPanel.updatePosSeenRect();

  // Scroll notifications
  sidePanel.moveNotifications();

  // Process a chunk of incoming data
  const len = dataStream.length;
  if (len === 0) return;

  // Rate limit: process at most len/6000 or 1 element per tick
  const batchSize = Math.max(1, Math.min(len, Math.ceil(len / 6000)));
  const batch = dataStream.splice(0, batchSize);

  for (const element of batch) {
    // Add heatmap trail rectangle
    mapPanel.addPosSeenRect(
      element.x,
      element.y,
      element.z || 0,
      element.notable || ""
    );

    // Update per-agent cursor position (Link sprite)
    // Use sub-tile float coords (fx/fy) for precise placement,
    // falling back to integer tile coords if not available
    mapPanel.updateAgent(
      element._envId,
      element.fx ?? element.x,
      element.fy ?? element.y,
      element._color,
      element._direction
    );

    // Only show meaningful events in the sidebar — skip new_room
    // since it fires constantly with 24 parallel agents
    if (
      element.notable &&
      element.notable !== "" &&
      element.notable !== "new_room"
    ) {
      sidePanel.addNotification(element.notable);
    }
  }

  updateStatus(
    `Received: ${totalReceived} | Buffer: ${dataStream.length} | Agents: ${mapPanel.agentCursors.size} | Tiles: ${mapPanel.posSeenMap.size}`
  );
}

function updateStatus(text) {
  if (statusEl) {
    statusEl.textContent = text;
  }
}

// Handle window resize
window.addEventListener("resize", () => {
  if (app) {
    app.renderer.resize(window.innerWidth, window.innerHeight);
  }
});

// Launch
init().catch(console.error);
