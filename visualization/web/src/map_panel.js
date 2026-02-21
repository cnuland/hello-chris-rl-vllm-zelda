/**
 * Map panel for Oracle of Seasons RL agent visualization.
 *
 * Renders the overworld map background with two overlay layers:
 *   1. Agent cursors — bright colored circles showing each agent's
 *      current position, updated in real-time.
 *   2. Heatmap trail — fading translucent rectangles showing where
 *      agents have recently visited (green → red over 15s TTL).
 *
 * Architecture inspired by:
 *   - LinkMapViz map_panel.js (https://github.com/Xe-Xo/LinkMapViz)
 *     PosSeen class with time-based color fading and PixiJS rendering.
 *   - PokemonRedExperiments (https://github.com/PWhiddy/PokemonRedExperiments)
 *     Overworld map overlay concept.
 *
 * License: MIT
 */

import { Container, Graphics, Sprite, Assets } from "pixi.js";

const TILE_SIZE = 16;
const TTL_MS = 15000; // 15 seconds before trail positions fade out
const CURSOR_RADIUS = 5;

// --- Color utilities ---

function hslToHex(h, s, l) {
  s /= 100;
  l /= 100;
  const a = s * Math.min(l, 1 - l);
  const f = (n) => {
    const k = (n + h / 30) % 12;
    const color = l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
    return Math.round(255 * color);
  };
  return (f(0) << 16) | (f(8) << 8) | f(4);
}

function parseColor(colorStr) {
  if (!colorStr) return 0x44aa77;
  const m = colorStr.match(/hsl\(\s*(\d+)\s*,\s*(\d+)%?\s*,\s*(\d+)%?\s*\)/);
  if (m) return hslToHex(parseInt(m[1]), parseInt(m[2]), parseInt(m[3]));
  if (colorStr.startsWith("#")) return parseInt(colorStr.slice(1), 16) || 0x44aa77;
  return 0x44aa77;
}

/**
 * Represents a single visited position on the heatmap trail.
 */
class PosSeen {
  constructor(mapPanel, worldX, worldY, worldZ) {
    this.mapPanel = mapPanel;
    this.worldX = worldX;
    this.worldY = worldY;
    this.worldZ = worldZ;
    this.createdAt = Date.now();

    this.graphics = new Graphics();
    this.graphics.x = this.worldX * TILE_SIZE;
    this.graphics.y = this.worldY * TILE_SIZE;
    this.graphics.rect(0, 0, TILE_SIZE, TILE_SIZE);
    this.graphics.fill({ color: 0x00ff00, alpha: 0.25 });

    mapPanel.trailContainer.addChild(this.graphics);
  }

  getColor(age) {
    const t = Math.min(Math.max(age, 0), 1);
    let r, g;
    if (t < 0.5) {
      r = Math.floor(255 * t * 2);
      g = 255;
    } else {
      r = 255;
      g = Math.floor(255 * (1 - (t - 0.5) * 2));
    }
    return (r << 16) | (g << 8);
  }

  update() {
    const age = (Date.now() - this.createdAt) / TTL_MS;
    if (age >= 1.0) return false;

    const color = this.getColor(age);
    const alpha = 0.25 * (1.0 - age * 0.9);

    this.graphics.clear();
    this.graphics.rect(0, 0, TILE_SIZE, TILE_SIZE);
    this.graphics.fill({ color, alpha });
    return true;
  }

  destroy() {
    this.mapPanel.trailContainer.removeChild(this.graphics);
    this.graphics.destroy();
  }
}

/**
 * Agent cursor — a bright colored circle showing an agent's current position.
 */
class AgentCursor {
  constructor(mapPanel, envId, color) {
    this.mapPanel = mapPanel;
    this.envId = envId;
    this.color = color;

    this.container = new Container();

    // Outer glow ring
    this.glow = new Graphics();
    this.glow.circle(0, 0, CURSOR_RADIUS + 3);
    this.glow.fill({ color: this.color, alpha: 0.3 });
    this.container.addChild(this.glow);

    // Main dot
    this.dot = new Graphics();
    this.dot.circle(0, 0, CURSOR_RADIUS);
    this.dot.fill({ color: this.color, alpha: 1.0 });
    this.dot.stroke({ color: 0xffffff, width: 1.5, alpha: 0.9 });
    this.container.addChild(this.dot);

    mapPanel.cursorContainer.addChild(this.container);
  }

  moveTo(worldX, worldY) {
    this.container.x = worldX * TILE_SIZE + TILE_SIZE / 2;
    this.container.y = worldY * TILE_SIZE + TILE_SIZE / 2;
  }

  destroy() {
    this.mapPanel.cursorContainer.removeChild(this.container);
    this.container.destroy({ children: true });
  }
}

/**
 * Main map panel component.
 */
export class MapPanel {
  constructor(app) {
    this.app = app;
    this.container = new Container();
    this.bgContainer = new Container();
    this.trailContainer = new Container();
    this.cursorContainer = new Container();

    this.container.addChild(this.bgContainer);
    this.container.addChild(this.trailContainer);
    this.container.addChild(this.cursorContainer);

    app.stage.addChild(this.container);

    // Heatmap trail: key → PosSeen
    this.posSeenMap = new Map();

    // Agent cursors: envId → AgentCursor
    this.agentCursors = new Map();

    // Interaction state
    this._dragging = false;
    this._dragStart = { x: 0, y: 0 };
    this._panStart = { x: 0, y: 0 };

    this._setupInteraction();
  }

  addBackground(alias) {
    for (const child of this.bgContainer.children) {
      child.visible = false;
    }
    const sprite = Sprite.from(alias);
    this.bgContainer.addChild(sprite);
  }

  /**
   * Update an agent's cursor position (or create it if new).
   */
  updateAgent(envId, worldX, worldY, colorStr) {
    let cursor = this.agentCursors.get(envId);
    if (!cursor) {
      const color = parseColor(colorStr);
      cursor = new AgentCursor(this, envId, color);
      this.agentCursors.set(envId, cursor);
    }
    cursor.moveTo(worldX, worldY);
  }

  /**
   * Add a heatmap trail rectangle.
   */
  addTrailRect(x, y, z) {
    const key = `${x}_${y}_${z}`;
    if (this.posSeenMap.has(key)) {
      const existing = this.posSeenMap.get(key);
      existing.destroy();
      this.posSeenMap.delete(key);
    }
    const posSeen = new PosSeen(this, x, y, z);
    this.posSeenMap.set(key, posSeen);
  }

  /**
   * Update all trail rectangles (fade/destroy expired ones).
   */
  updateTrails() {
    for (const [key, posSeen] of this.posSeenMap) {
      if (!posSeen.update()) {
        posSeen.destroy();
        this.posSeenMap.delete(key);
      }
    }
  }

  // Keep old name as alias for backward compat with index.js
  addPosSeenRect(x, y, z, notable) {
    this.addTrailRect(x, y, z);
  }
  updatePosSeenRect() {
    this.updateTrails();
  }

  _setupInteraction() {
    const canvas = this.app.canvas;

    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      const worldX = (mouseX - this.container.x) / this.container.scale.x;
      const worldY = (mouseY - this.container.y) / this.container.scale.y;

      this.container.scale.x *= zoomFactor;
      this.container.scale.y *= zoomFactor;

      const minScale = 0.1;
      const maxScale = 10;
      this.container.scale.x = Math.max(minScale, Math.min(maxScale, this.container.scale.x));
      this.container.scale.y = Math.max(minScale, Math.min(maxScale, this.container.scale.y));

      this.container.x = mouseX - worldX * this.container.scale.x;
      this.container.y = mouseY - worldY * this.container.scale.y;
    });

    canvas.addEventListener("mousedown", (e) => {
      this._dragging = true;
      this._dragStart = { x: e.clientX, y: e.clientY };
      this._panStart = { x: this.container.x, y: this.container.y };
    });

    canvas.addEventListener("mousemove", (e) => {
      if (!this._dragging) return;
      this.container.x = this._panStart.x + (e.clientX - this._dragStart.x);
      this.container.y = this._panStart.y + (e.clientY - this._dragStart.y);
    });

    canvas.addEventListener("mouseup", () => { this._dragging = false; });
    canvas.addEventListener("mouseleave", () => { this._dragging = false; });

    let lastTouchDist = 0;
    canvas.addEventListener("touchstart", (e) => {
      if (e.touches.length === 1) {
        this._dragging = true;
        this._dragStart = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        this._panStart = { x: this.container.x, y: this.container.y };
      } else if (e.touches.length === 2) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        lastTouchDist = Math.sqrt(dx * dx + dy * dy);
      }
    });

    canvas.addEventListener("touchmove", (e) => {
      e.preventDefault();
      if (e.touches.length === 1 && this._dragging) {
        this.container.x = this._panStart.x + (e.touches[0].clientX - this._dragStart.x);
        this.container.y = this._panStart.y + (e.touches[0].clientY - this._dragStart.y);
      } else if (e.touches.length === 2) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (lastTouchDist > 0) {
          this.container.scale.x *= dist / lastTouchDist;
          this.container.scale.y *= dist / lastTouchDist;
        }
        lastTouchDist = dist;
      }
    });

    canvas.addEventListener("touchend", () => {
      this._dragging = false;
      lastTouchDist = 0;
    });
  }
}
