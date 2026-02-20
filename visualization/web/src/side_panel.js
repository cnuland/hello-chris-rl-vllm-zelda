/**
 * Side notification panel for Oracle of Seasons RL map visualization.
 *
 * Displays scrolling notification icons for notable game events
 * (gate slashes, item pickups, new rooms discovered) in a vertical
 * sidebar on the left side of the screen.
 *
 * Architecture inspired by:
 *   - LinkMapViz side_panel.js (https://github.com/Xe-Xo/LinkMapViz)
 *     Scrolling notification sidebar with event icons.
 *
 * License: MIT
 */

import { Container, Graphics, Text } from "pixi.js";

const PANEL_WIDTH = 50;
const PANEL_ALPHA = 0.6;
const SCROLL_SPEED = 0.8;
const ICON_SIZE = 20;

/**
 * Notification entry that scrolls downward.
 */
class Notification {
  constructor(panel, eventType) {
    this.panel = panel;
    this.eventType = eventType;
    this.container = new Container();
    this.container.y = 0;

    // Simple colored circle with event label
    const colors = {
      gate_slash: 0xff4444,
      item_obtained: 0xffdd44,
      new_room: 0x44ff44,
    };

    const gfx = new Graphics();
    const color = colors[eventType] || 0xffffff;
    gfx.circle(ICON_SIZE / 2, ICON_SIZE / 2, ICON_SIZE / 2 - 2);
    gfx.fill({ color, alpha: 0.9 });
    gfx.stroke({ color: 0xffffff, width: 1, alpha: 0.6 });
    this.container.addChild(gfx);

    // Short label
    const labels = {
      gate_slash: "G",
      item_obtained: "I",
      new_room: "R",
    };
    const label = new Text({
      text: labels[eventType] || "?",
      style: {
        fontSize: 10,
        fontFamily: "monospace",
        fill: 0xffffff,
      },
    });
    label.x = ICON_SIZE / 2 - label.width / 2;
    label.y = ICON_SIZE / 2 - label.height / 2;
    this.container.addChild(label);

    panel.notifContainer.addChild(this.container);
  }

  /**
   * Move notification downward. Returns false when off-screen.
   */
  update(maxY) {
    this.container.y += SCROLL_SPEED;
    this.container.alpha = Math.max(0, 1.0 - this.container.y / maxY);
    return this.container.y < maxY;
  }

  destroy() {
    this.panel.notifContainer.removeChild(this.container);
    this.container.destroy({ children: true });
  }
}

/**
 * Side panel component for event notifications.
 */
export class SidePanel {
  constructor(app, width, height) {
    this.app = app;
    this.screenWidth = width;
    this.screenHeight = height;
    this.notifications = [];

    // Panel background
    this.container = new Container();
    this.container.x = 0;
    this.container.y = 0;

    const bg = new Graphics();
    bg.rect(0, 0, PANEL_WIDTH, height);
    bg.fill({ color: 0x000000, alpha: PANEL_ALPHA });
    this.container.addChild(bg);

    // Notification container (children scroll within)
    this.notifContainer = new Container();
    this.notifContainer.x = (PANEL_WIDTH - ICON_SIZE) / 2;
    this.container.addChild(this.notifContainer);

    app.stage.addChild(this.container);
  }

  /**
   * Add a new notification at the top.
   */
  addNotification(eventType) {
    const notif = new Notification(this, eventType);
    this.notifications.push(notif);
  }

  /**
   * Scroll and remove expired notifications.
   */
  moveNotifications() {
    const toRemove = [];
    for (const notif of this.notifications) {
      const alive = notif.update(this.screenHeight);
      if (!alive) {
        toRemove.push(notif);
      }
    }
    for (const notif of toRemove) {
      notif.destroy();
      const idx = this.notifications.indexOf(notif);
      if (idx !== -1) this.notifications.splice(idx, 1);
    }
  }
}
