/**
 * Map panel for Oracle of Seasons RL agent visualization.
 *
 * Renders the overworld map background and overlays position heatmap
 * rectangles showing where the agent has been. Recent positions are green,
 * fading to red over time (30-second TTL).
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
const TTL_MS = 30000; // 30 seconds before positions fade out

/**
 * Represents a single visited position on the map.
 */
class PosSeen {
  constructor(mapPanel, worldX, worldY, worldZ, notable) {
    this.mapPanel = mapPanel;
    this.worldX = worldX;
    this.worldY = worldY;
    this.worldZ = worldZ;
    this.notable = notable;
    this.createdAt = Date.now();

    this.container = new Container();
    this.container.x = this.worldX * TILE_SIZE;
    this.container.y = this.worldY * TILE_SIZE;

    // Draw colored rectangle
    this.graphics = new Graphics();
    this.graphics.rect(0, 0, TILE_SIZE, TILE_SIZE);
    this.graphics.fill({ color: 0x00ff00, alpha: 0.5 });
    this.container.addChild(this.graphics);

    // Notable event icon (if any)
    if (this.notable && this.notable !== "") {
      const iconAlias = this._notableToIcon(this.notable);
      if (iconAlias) {
        try {
          this.sprite = Sprite.from(iconAlias);
          this.sprite.setSize(TILE_SIZE, TILE_SIZE);
          this.container.addChild(this.sprite);
        } catch (e) {
          // Icon not loaded, skip
        }
      }
    }

    mapPanel.posContainer.addChild(this.container);
  }

  _notableToIcon(notable) {
    const iconMap = {
      gate_slash: "icon_gate",
      item_obtained: "icon_item",
      new_room: "icon_room",
    };
    return iconMap[notable] || null;
  }

  /**
   * Get interpolated color based on age (green → yellow → red).
   */
  getColor(age) {
    // age: 0 = fresh, 1 = expired
    const t = Math.min(Math.max(age, 0), 1);
    let r, g, b;
    if (t < 0.5) {
      // Green → Yellow
      const f = t * 2;
      r = Math.floor(255 * f);
      g = 255;
      b = 0;
    } else {
      // Yellow → Red
      const f = (t - 0.5) * 2;
      r = 255;
      g = Math.floor(255 * (1 - f));
      b = 0;
    }
    return (r << 16) | (g << 8) | b;
  }

  /**
   * Update visual appearance based on age. Returns false if expired.
   */
  update() {
    const elapsed = Date.now() - this.createdAt;
    const age = elapsed / TTL_MS;

    if (age >= 1.0) {
      return false; // Expired
    }

    const color = this.getColor(age);
    const alpha = 0.5 * (1.0 - age * 0.8);

    this.graphics.clear();
    this.graphics.rect(0, 0, TILE_SIZE, TILE_SIZE);
    this.graphics.fill({ color, alpha });

    // Float notable icons upward as they age
    if (this.sprite) {
      this.sprite.y = -age * 10;
      this.sprite.alpha = 1.0 - age;
    }

    return true; // Still alive
  }

  destroy() {
    this.mapPanel.posContainer.removeChild(this.container);
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
    this.posContainer = new Container();

    this.container.addChild(this.bgContainer);
    this.container.addChild(this.posContainer);

    app.stage.addChild(this.container);

    // Position tracking: key → PosSeen
    this.posSeenMap = new Map();

    // Interaction state
    this._dragging = false;
    this._dragStart = { x: 0, y: 0 };
    this._panStart = { x: 0, y: 0 };

    this._setupInteraction();
  }

  /**
   * Add a background map image.
   */
  addBackground(alias) {
    // Hide previous backgrounds
    for (const child of this.bgContainer.children) {
      child.visible = false;
    }
    const sprite = Sprite.from(alias);
    this.bgContainer.addChild(sprite);
  }

  /**
   * Add a visited position rectangle.
   */
  addPosSeenRect(x, y, z, notable) {
    const key = `${x}_${y}_${z}`;

    // If position already tracked, refresh it
    if (this.posSeenMap.has(key)) {
      const existing = this.posSeenMap.get(key);
      existing.destroy();
      this.posSeenMap.delete(key);
    }

    const posSeen = new PosSeen(this, x, y, z, notable);
    this.posSeenMap.set(key, posSeen);
  }

  /**
   * Update all position rectangles (fade/destroy expired ones).
   */
  updatePosSeenRect() {
    for (const [key, posSeen] of this.posSeenMap) {
      const alive = posSeen.update();
      if (!alive) {
        posSeen.destroy();
        this.posSeenMap.delete(key);
      }
    }
  }

  /**
   * Set up pan and zoom interactions.
   */
  _setupInteraction() {
    const canvas = this.app.canvas;

    // Mouse wheel zoom
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;

      // Zoom toward cursor position
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const worldX = (mouseX - this.container.x) / this.container.scale.x;
      const worldY = (mouseY - this.container.y) / this.container.scale.y;

      this.container.scale.x *= zoomFactor;
      this.container.scale.y *= zoomFactor;

      // Clamp scale
      const minScale = 0.1;
      const maxScale = 10;
      this.container.scale.x = Math.max(minScale, Math.min(maxScale, this.container.scale.x));
      this.container.scale.y = Math.max(minScale, Math.min(maxScale, this.container.scale.y));

      this.container.x = mouseX - worldX * this.container.scale.x;
      this.container.y = mouseY - worldY * this.container.scale.y;
    });

    // Mouse drag pan
    canvas.addEventListener("mousedown", (e) => {
      this._dragging = true;
      this._dragStart = { x: e.clientX, y: e.clientY };
      this._panStart = { x: this.container.x, y: this.container.y };
    });

    canvas.addEventListener("mousemove", (e) => {
      if (!this._dragging) return;
      const dx = e.clientX - this._dragStart.x;
      const dy = e.clientY - this._dragStart.y;
      this.container.x = this._panStart.x + dx;
      this.container.y = this._panStart.y + dy;
    });

    canvas.addEventListener("mouseup", () => { this._dragging = false; });
    canvas.addEventListener("mouseleave", () => { this._dragging = false; });

    // Touch support
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
        const dx = e.touches[0].clientX - this._dragStart.x;
        const dy = e.touches[0].clientY - this._dragStart.y;
        this.container.x = this._panStart.x + dx;
        this.container.y = this._panStart.y + dy;
      } else if (e.touches.length === 2) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (lastTouchDist > 0) {
          const scale = dist / lastTouchDist;
          this.container.scale.x *= scale;
          this.container.scale.y *= scale;
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
