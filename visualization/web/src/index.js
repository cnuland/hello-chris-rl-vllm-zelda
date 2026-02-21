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

// --- Configuration ---
// Auto-detect wss:// when served over HTTPS (e.g. behind OpenShift Route)
const _wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
const WS_URL =
  new URLSearchParams(window.location.search).get("ws") ||
  `${_wsProto}//${window.location.host}/receive`;
const PROCESS_INTERVAL_MS = 20; // 50Hz processing rate
const RECONNECT_INTERVAL_MS = 2000;
const MAX_TRAIL_BUFFER = 20000; // Cap trail buffer to prevent memory growth

// --- State ---
let app;
let mapPanel;
let ws = null;
let trailStream = []; // Queued trail positions (rate-limited processing)
let totalReceived = 0;
let statusEl;

// Latest cursor position per env_id — updated IMMEDIATELY on WebSocket
// message arrival, bypassing the rate-limited trail queue entirely.
// This ensures cursors always reflect the agent's current room/position.
let latestCursorPos = new Map();

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
          const meta = data.metadata || {};
          const envId = meta.env_id ?? "unknown";
          const color = meta.color || "";

          // Find the last overworld position in this batch for the cursor.
          // This is the agent's CURRENT position — skip interiors (z != 0)
          // since interior room_ids don't map to the overworld grid.
          let lastOverworld = null;
          for (let i = data.pos_data.length - 1; i >= 0; i--) {
            const p = data.pos_data[i];
            if (!p.z || p.z === 0) {
              lastOverworld = p;
              break;
            }
          }

          // Update cursor immediately (bypasses trail queue).
          // The StreamWrapper's -16px HUD correction over-corrects by ~1
          // tile — PLAYER_Y is room-relative in OoS, not screen-relative.
          // Shift Y down by +1 tile here to compensate, then clamp to
          // avoid cliff/wall boundary tiles.
          if (lastOverworld) {
            let fx = lastOverworld.fx ?? lastOverworld.x;
            let fy = lastOverworld.fy ?? lastOverworld.y;

            // Shift Y down 1 tile (compensate server-side HUD over-correction)
            const roomBaseY = Math.floor(fy / 8) * 8;
            let adjInRoomY = Math.min((fy - roomBaseY) + 1.0, 7.0);

            // Clamp to room interior — bottom rows 6-7 are often cliff/wall
            if (adjInRoomY > 6.0) adjInRoomY = 6.0;

            fy = roomBaseY + adjInRoomY;

            // Shift X left by 1 tile — entity X coordinates in OoS have
            // the same type of offset as Y (position represents the right
            // side of the collision box rather than the center).
            const roomBaseX = Math.floor(fx / 10) * 10;
            let adjInRoomX = Math.max((fx - roomBaseX) - 1.0, 0.0);

            // Clamp to room interior — left/right border tiles are walls
            if (adjInRoomX > 8.5) adjInRoomX = 8.5;

            fx = roomBaseX + adjInRoomX;

            latestCursorPos.set(envId, {
              fx: fx,
              fy: fy,
              color: color,
              direction: lastOverworld.direction ?? 2,
            });
          }

          // Queue overworld positions for trail heatmap (rate-limited)
          // Skip boundary tiles (top/bottom of rooms) to avoid heatmap
          // marks on non-walkable transition tiles.
          for (const element of data.pos_data) {
            if (element.z && element.z !== 0) continue; // skip interiors
            const trailInRoomY = element.y % 8;
            if (trailInRoomY === 0 || trailInRoomY >= 7) continue; // skip boundary rows
            trailStream.push(element);
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
 *
 * Cursor updates are applied immediately from latestCursorPos (no rate
 * limit), so agents always show their current position.  Trail heatmap
 * rectangles are rate-limited to avoid overwhelming the renderer.
 */
function processDataStream() {
  // Advance trail fade/destroy + cursor lerp
  mapPanel.updateTrails();

  // --- Cursor updates (immediate, every tick) ---
  for (const [envId, pos] of latestCursorPos) {
    mapPanel.updateAgent(envId, pos.fx, pos.fy, pos.color, pos.direction);
  }

  // --- Trail heatmap (rate-limited) ---
  const len = trailStream.length;
  if (len === 0) {
    updateStatus(
      `Received: ${totalReceived} | Agents: ${mapPanel.agentCursors.size} | Tiles: ${mapPanel.posSeenMap.size}`
    );
    return;
  }

  // Process up to 100 trail points per tick (5000/sec at 50Hz)
  const batchSize = Math.min(len, 100);
  const batch = trailStream.splice(0, batchSize);

  for (const element of batch) {
    mapPanel.addTrailRect(element.x, element.y, element.z || 0);
  }

  // Cap buffer to prevent unbounded memory growth
  if (trailStream.length > MAX_TRAIL_BUFFER) {
    trailStream.splice(0, trailStream.length - MAX_TRAIL_BUFFER);
  }

  updateStatus(
    `Received: ${totalReceived} | Buffer: ${trailStream.length} | Agents: ${mapPanel.agentCursors.size} | Tiles: ${mapPanel.posSeenMap.size}`
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
