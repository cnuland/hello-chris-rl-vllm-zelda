#!/usr/bin/env python3
"""Offline video renderer for Oracle of Seasons RL agent visualization.

Reads coordinate logs (CSV from StreamWrapper or JSONL from EpisodeExporter)
and renders animated videos showing Link walking across the full overworld map.

Architecture inspired by:
  - PokemonRedExperiments BetterMapVis_script_version.py
    (https://github.com/PWhiddy/PokemonRedExperiments)
    Animated sprite overlay on full overworld map with multi-process rendering.
  - PokemonRedExperiments BetterMapVis_script_version_FLOW.py
    Flow field visualization showing aggregate movement directions.
  - LinkMapViz (https://github.com/Xe-Xo/LinkMapViz)
    Heatmap-style position visualization with time-based fading.

Memory addresses derived from oracles-disasm:
  https://github.com/Stewmath/oracles-disasm

License: MIT
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Map layout constants (must match generate_overworld.py)
ROOM_W = 160
ROOM_H = 128
GRID_COLS = 16
GRID_ROWS = 16
TILE_SIZE = 16
TILES_PER_ROOM_X = 10
TILES_PER_ROOM_Y = 8
MAP_W = GRID_COLS * ROOM_W   # 2560
MAP_H = GRID_ROWS * ROOM_H   # 2048

# Direction sprites (simple colored arrows as fallback)
DIR_UP = 0
DIR_RIGHT = 1
DIR_DOWN = 2
DIR_LEFT = 3

# Heatmap colors (green → yellow → red, matching LinkMapViz style)
HEATMAP_COLORS = [
    (0, 255, 0),      # Fresh (green)
    (128, 255, 0),    # Recent
    (255, 255, 0),    # Moderate (yellow)
    (255, 128, 0),    # Aging
    (255, 0, 0),      # Stale (red)
]


def load_coordinates_csv(csv_path: str) -> list[dict]:
    """Load coordinate data from StreamWrapper CSV log.

    Expected format:
        step,world_x,world_y,world_z,room_id,tile_x,tile_y,direction,notable
    """
    import csv

    coords = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            coords.append({
                "step": int(row["step"]),
                "x": int(row["world_x"]),
                "y": int(row["world_y"]),
                "z": int(row["world_z"]),
                "room_id": int(row["room_id"]),
                "tile_x": int(row["tile_x"]),
                "tile_y": int(row["tile_y"]),
                "direction": int(row["direction"]),
                "notable": row.get("notable", ""),
            })
    return coords


def load_coordinates_jsonl(jsonl_path: str) -> list[dict]:
    """Load coordinate data from EpisodeExporter JSONL state logs.

    Each line is a JSON object with a 'state' field containing RAM readings.
    """
    coords = []
    step = 0
    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            state = record.get("state", record)

            room_id = state.get("room_id", 0)
            pixel_x = state.get("player_x", 80)
            pixel_y = state.get("player_y", 72)
            group = state.get("active_group", 0)
            direction = state.get("direction", 2)

            tile_x = pixel_x // 16
            tile_y = pixel_y // 16
            room_col = room_id % 16
            room_row = room_id // 16
            world_x = room_col * 10 + min(tile_x, 9)
            world_y = room_row * 8 + min(tile_y, 7)

            coords.append({
                "step": step,
                "x": world_x,
                "y": world_y,
                "z": group,
                "room_id": room_id,
                "tile_x": tile_x,
                "tile_y": tile_y,
                "direction": direction,
                "notable": "",
            })
            step += 1
    return coords


def create_marker_sprite(size: int = 12, color: tuple = (0, 255, 0)) -> Image.Image:
    """Create a simple colored circle marker sprite."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([1, 1, size - 2, size - 2], fill=(*color, 200), outline=(255, 255, 255, 255))
    return img


def create_arrow_sprite(size: int = 14, direction: int = DIR_DOWN,
                        color: tuple = (0, 255, 0)) -> Image.Image:
    """Create a directional arrow sprite."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    r = size // 2 - 2

    if direction == DIR_UP:
        points = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]
    elif direction == DIR_DOWN:
        points = [(cx, cy + r), (cx - r, cy - r), (cx + r, cy - r)]
    elif direction == DIR_LEFT:
        points = [(cx - r, cy), (cx + r, cy - r), (cx + r, cy + r)]
    elif direction == DIR_RIGHT:
        points = [(cx + r, cy), (cx - r, cy - r), (cx - r, cy + r)]
    else:
        points = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]

    draw.polygon(points, fill=(*color, 220), outline=(255, 255, 255, 255))
    return img


def load_link_sprites(sprite_path: str | None) -> dict[int, Image.Image] | None:
    """Load Link character sprites from a sprite sheet.

    Expected format: 4 directional sprites arranged horizontally,
    each 16x16 pixels. If not available, returns None and the renderer
    falls back to arrow markers.
    """
    if sprite_path is None or not Path(sprite_path).exists():
        return None

    try:
        sheet = Image.open(sprite_path).convert("RGBA")
        sprites = {}
        sprite_w = 16
        sprite_h = 16
        # Expected order: down, up, left, right
        dir_order = [DIR_DOWN, DIR_UP, DIR_LEFT, DIR_RIGHT]
        for i, d in enumerate(dir_order):
            x = i * sprite_w
            sprites[d] = sheet.crop((x, 0, x + sprite_w, sprite_h))
        return sprites
    except Exception as e:
        logger.warning(f"Failed to load sprites from {sprite_path}: {e}")
        return None


def interpolate_color(t: float) -> tuple[int, int, int]:
    """Interpolate heatmap color for age value t (0=new, 1=old)."""
    t = max(0.0, min(1.0, t))
    n = len(HEATMAP_COLORS) - 1
    idx = t * n
    i = int(idx)
    frac = idx - i
    if i >= n:
        return HEATMAP_COLORS[-1]
    c1 = HEATMAP_COLORS[i]
    c2 = HEATMAP_COLORS[i + 1]
    return (
        int(c1[0] + (c2[0] - c1[0]) * frac),
        int(c1[1] + (c2[1] - c1[1]) * frac),
        int(c1[2] + (c2[2] - c1[2]) * frac),
    )


def render_heatmap_video(
    coords: list[dict],
    map_path: str,
    output_path: str,
    fps: int = 30,
    steps_per_frame: int = 10,
    trail_length: int = 300,
    scale: float = 1.0,
    sprite_path: str | None = None,
):
    """Render a heatmap-style video showing agent exploration.

    Like LinkMapViz, recent positions are green and fade to red over time.
    The current position shows a directional sprite or arrow.

    Args:
        coords: List of coordinate dicts from load_coordinates_*.
        map_path: Path to overworld.png background image.
        output_path: Output video file path (.mp4 or .mov).
        fps: Video frame rate.
        steps_per_frame: How many coordinate steps per video frame.
        trail_length: How many past positions to show in the trail.
        scale: Scale factor for output resolution.
        sprite_path: Optional path to Link sprite sheet.
    """
    import cv2

    logger.info(f"Loading map from {map_path}")
    bg = Image.open(map_path).convert("RGB")
    bg_w, bg_h = bg.size

    if scale != 1.0:
        bg = bg.resize((int(bg_w * scale), int(bg_h * scale)), Image.NEAREST)
        bg_w, bg_h = bg.size

    # Load sprites
    link_sprites = load_link_sprites(sprite_path)

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_w, out_h = bg_w, bg_h
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    total_steps = len(coords)
    total_frames = total_steps // steps_per_frame
    logger.info(
        f"Rendering {total_frames} frames "
        f"({total_steps} steps, {steps_per_frame} steps/frame, {fps} FPS)"
    )

    for frame_idx in range(total_frames):
        step_end = (frame_idx + 1) * steps_per_frame
        step_start = max(0, step_end - trail_length)

        # Create frame from background
        frame = bg.copy()
        overlay = Image.new("RGBA", (bg_w, bg_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Draw trail (older positions first, newer on top)
        trail_coords = coords[step_start:step_end]
        trail_len = len(trail_coords)

        for i, c in enumerate(trail_coords):
            if c["z"] != 0:  # Only draw overworld positions
                continue

            age = 1.0 - (i / max(trail_len - 1, 1))  # 0=newest, 1=oldest
            color = interpolate_color(age)
            alpha = int(180 * (1.0 - age * 0.7))

            px = int(c["x"] * TILE_SIZE * scale)
            py = int(c["y"] * TILE_SIZE * scale)
            tile_sz = int(TILE_SIZE * scale)

            draw.rectangle(
                [px, py, px + tile_sz - 1, py + tile_sz - 1],
                fill=(*color, alpha),
            )

        # Draw current position marker
        if step_end > 0 and step_end <= total_steps:
            cur = coords[step_end - 1]
            if cur["z"] == 0:
                px = int(cur["x"] * TILE_SIZE * scale)
                py = int(cur["y"] * TILE_SIZE * scale)

                if link_sprites and cur["direction"] in link_sprites:
                    sprite = link_sprites[cur["direction"]]
                    if scale != 1.0:
                        sprite = sprite.resize(
                            (int(16 * scale), int(16 * scale)),
                            Image.NEAREST,
                        )
                    overlay.paste(sprite, (px, py), sprite)
                else:
                    arrow = create_arrow_sprite(
                        size=int(14 * scale),
                        direction=cur["direction"],
                        color=(255, 255, 255),
                    )
                    overlay.paste(arrow, (px + 1, py + 1), arrow)

        # Composite overlay onto frame
        frame = frame.convert("RGBA")
        frame = Image.alpha_composite(frame, overlay)
        frame = frame.convert("RGB")

        # Convert to OpenCV BGR and write
        frame_arr = np.array(frame)
        frame_bgr = cv2.cvtColor(frame_arr, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

        if (frame_idx + 1) % 100 == 0:
            logger.info(f"  Frame {frame_idx + 1}/{total_frames}")

    writer.release()
    logger.info(f"Video saved to {output_path}")


def render_flow_field(
    coords: list[dict],
    map_path: str,
    output_path: str,
    scale: float = 1.0,
):
    """Render a static flow field image showing movement directions.

    Inspired by PokemonRedExperiments BetterMapVis_script_version_FLOW.py.
    Each visited tile gets a colored arrow showing the aggregate movement
    direction at that position.

    Args:
        coords: List of coordinate dicts.
        map_path: Path to overworld.png.
        output_path: Output image path.
        scale: Scale factor.
    """
    logger.info("Computing flow field...")

    # Accumulate movement vectors at each position
    flows: dict[tuple[int, int], np.ndarray] = {}
    visit_counts: dict[tuple[int, int], int] = {}

    for i in range(1, len(coords)):
        prev = coords[i - 1]
        cur = coords[i]

        if prev["z"] != 0 or cur["z"] != 0:
            continue

        dx = cur["x"] - prev["x"]
        dy = cur["y"] - prev["y"]

        # Skip teleports (large jumps)
        if abs(dx) > 2 or abs(dy) > 2:
            continue

        key = (cur["x"], cur["y"])
        if key not in flows:
            flows[key] = np.array([0.0, 0.0])
            visit_counts[key] = 0

        flows[key] += np.array([dx, dy])
        visit_counts[key] += 1

    logger.info(f"Flow data for {len(flows)} unique tiles")

    # Load background
    bg = Image.open(map_path).convert("RGBA")
    if scale != 1.0:
        bg = bg.resize(
            (int(bg.width * scale), int(bg.height * scale)),
            Image.NEAREST,
        )

    overlay = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    max_visits = max(visit_counts.values()) if visit_counts else 1

    for (wx, wy), flow in flows.items():
        visits = visit_counts[(wx, wy)]
        magnitude = np.linalg.norm(flow)

        if magnitude < 0.1:
            continue

        # Direction → hue (HUSL-like color mapping)
        angle = math.atan2(-flow[1], flow[0])  # -y because screen Y is inverted
        hue = (angle / (2 * math.pi)) % 1.0

        # Convert hue to RGB (simplified HSV with S=1, V=1)
        h6 = hue * 6
        i_h = int(h6)
        f = h6 - i_h
        if i_h == 0:
            color = (255, int(255 * f), 0)
        elif i_h == 1:
            color = (int(255 * (1 - f)), 255, 0)
        elif i_h == 2:
            color = (0, 255, int(255 * f))
        elif i_h == 3:
            color = (0, int(255 * (1 - f)), 255)
        elif i_h == 4:
            color = (int(255 * f), 0, 255)
        else:
            color = (255, 0, int(255 * (1 - f)))

        # Intensity based on visit count
        alpha = int(100 + 155 * min(visits / max_visits, 1.0))

        px = int(wx * TILE_SIZE * scale)
        py = int(wy * TILE_SIZE * scale)
        tile_sz = int(TILE_SIZE * scale)

        draw.rectangle(
            [px, py, px + tile_sz - 1, py + tile_sz - 1],
            fill=(*color, alpha),
        )

    result = Image.alpha_composite(bg, overlay)
    result = result.convert("RGB")
    result.save(output_path)
    logger.info(f"Flow field saved to {output_path}")


def render_coverage_image(
    coords: list[dict],
    map_path: str,
    output_path: str,
    scale: float = 1.0,
):
    """Render a static coverage image showing all visited tiles.

    Args:
        coords: List of coordinate dicts.
        map_path: Path to overworld.png.
        output_path: Output image path.
        scale: Scale factor.
    """
    visited: dict[tuple[int, int], int] = {}
    for c in coords:
        if c["z"] != 0:
            continue
        key = (c["x"], c["y"])
        visited[key] = visited.get(key, 0) + 1

    logger.info(f"Coverage: {len(visited)} unique tiles visited")

    bg = Image.open(map_path).convert("RGBA")
    if scale != 1.0:
        bg = bg.resize(
            (int(bg.width * scale), int(bg.height * scale)),
            Image.NEAREST,
        )

    overlay = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    max_visits = max(visited.values()) if visited else 1

    for (wx, wy), count in visited.items():
        intensity = min(count / max_visits, 1.0)
        color = interpolate_color(1.0 - intensity)  # Green=high, Red=low
        alpha = int(100 + 100 * intensity)

        px = int(wx * TILE_SIZE * scale)
        py = int(wy * TILE_SIZE * scale)
        tile_sz = int(TILE_SIZE * scale)

        draw.rectangle(
            [px, py, px + tile_sz - 1, py + tile_sz - 1],
            fill=(*color, alpha),
        )

    result = Image.alpha_composite(bg, overlay)
    result = result.convert("RGB")
    result.save(output_path)
    logger.info(f"Coverage image saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Offline renderer for Oracle of Seasons RL agent visualization.",
        epilog=(
            "Credits:\n"
            "  Inspired by PokemonRedExperiments "
            "(https://github.com/PWhiddy/PokemonRedExperiments)\n"
            "  and LinkMapViz (https://github.com/Xe-Xo/LinkMapViz)\n"
            "  Memory addresses from oracles-disasm "
            "(https://github.com/Stewmath/oracles-disasm)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "coords", type=str,
        help="Path to coordinate log (.csv from StreamWrapper, or .jsonl from exporter)",
    )
    parser.add_argument(
        "--map", type=str, default=None,
        help="Path to overworld.png (default: visualization/assets/overworld.png)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output file path (.mp4 for video, .png for static)",
    )
    parser.add_argument(
        "--mode", choices=["video", "flow", "coverage"], default="video",
        help="Rendering mode: video (animated), flow (direction field), coverage (heatmap)",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Video frame rate (default: 30)",
    )
    parser.add_argument(
        "--steps-per-frame", type=int, default=10,
        help="Coordinate steps per video frame (default: 10)",
    )
    parser.add_argument(
        "--trail", type=int, default=300,
        help="Trail length in steps for video mode (default: 300)",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="Output scale factor (default: 1.0, use 2.0 for higher res)",
    )
    parser.add_argument(
        "--sprites", type=str, default=None,
        help="Path to Link sprite sheet (16x16 x 4 directions)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load coordinates
    coords_path = Path(args.coords)
    if coords_path.suffix == ".csv":
        coords = load_coordinates_csv(str(coords_path))
    elif coords_path.suffix in (".jsonl", ".json"):
        coords = load_coordinates_jsonl(str(coords_path))
    else:
        logger.error(f"Unknown coordinate format: {coords_path.suffix}")
        sys.exit(1)

    logger.info(f"Loaded {len(coords)} coordinate records")

    # Find map
    if args.map:
        map_path = args.map
    else:
        map_path = str(
            Path(__file__).resolve().parent / "assets" / "overworld.png"
        )

    if not Path(map_path).exists():
        logger.error(
            f"Map not found at {map_path}. "
            "Run generate_overworld.py first!"
        )
        sys.exit(1)

    # Output path
    if args.output:
        output_path = args.output
    else:
        suffix = ".mp4" if args.mode == "video" else ".png"
        output_path = str(coords_path.with_suffix(suffix))

    # Render
    if args.mode == "video":
        render_heatmap_video(
            coords, map_path, output_path,
            fps=args.fps,
            steps_per_frame=args.steps_per_frame,
            trail_length=args.trail,
            scale=args.scale,
            sprite_path=args.sprites,
        )
    elif args.mode == "flow":
        render_flow_field(coords, map_path, output_path, scale=args.scale)
    elif args.mode == "coverage":
        render_coverage_image(coords, map_path, output_path, scale=args.scale)


if __name__ == "__main__":
    main()
