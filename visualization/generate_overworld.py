#!/usr/bin/env python3
"""Generate full overworld map images for Oracle of Seasons.

Uses PyBoy to warp to every room in the 16x16 overworld grid and captures
screenshots, then stitches them into a single composite image.

Architecture inspired by:
  - PokemonRedExperiments (https://github.com/PWhiddy/PokemonRedExperiments)
    Map stitching approach from visualization/Map_Stitching.ipynb
  - LinkMapViz / LADXExperiments (https://github.com/Xe-Xo/LinkMapViz)
    create_world_map.py / get_room_map.py for Link's Awakening DX

Memory addresses derived from oracles-disasm:
  https://github.com/Stewmath/oracles-disasm

License: MIT
"""

import argparse
import io
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# =============================================================================
# Oracle of Seasons RAM addresses (confirmed via oracles-disasm wram.s)
# Format in disasm is Ages/Seasons — we use the Seasons column.
# =============================================================================

# Warp destination system (Seasons addresses)
WARP_DEST_GROUP = 0xCC63      # wWarpDestGroup — bit 7 set = direct warp
WARP_DEST_ROOM = 0xCC64       # wWarpDestRoom — room index
WARP_TRANSITION = 0xCC65      # wWarpTransition — transition type (bits 0-3)
WARP_DEST_POS = 0xCC66        # wWarpDestPos — encoded YX position
WARP_TRANSITION2 = 0xCC67     # wWarpTransition2 — 0x01 = instant (skip anim)

# Active state (Seasons addresses)
ACTIVE_GROUP = 0xCC49         # wActiveGroup
ACTIVE_ROOM = 0xCC4C          # wActiveRoom
LOADING_ROOM = 0xCC4B         # wLoadingRoom

# Player position
PLAYER_X = 0xD00D             # w1Link.xh (pixel X)
PLAYER_Y = 0xD00B             # w1Link.yh (pixel Y)
PLAYER_DIR = 0xD008           # w1Link.direction (0=up,1=right,2=down,3=left)

# Screen/game state
SCROLL_MODE = 0xCD00          # wScrollMode (bit 7 set = transitioning)
GAME_STATE = 0xC2EE           # wGameState (2 = gameplay)
MENU_STATE = 0xCBCB           # wOpenedMenuType (0 = none)
KEYS_PRESSED = 0xC481         # wKeysPressed
KEYS_JUST_PRESSED = 0xC482    # wKeysJustPressed

# Season
ROOM_STATE_MODIFIER = 0xCC4E  # wRoomStateModifier (0=spring..3=winter)

# GBC hardware registers for VRAM/WRAM bank switching
VBK_REG = 0xFF4F              # VRAM bank select (0 or 1)
SVBK_REG = 0xFF70             # WRAM bank select (0-7)

# VRAM addresses
TILE_DATA_START = 0x8000      # Tile pixel data ($8000-$97FF, 6144 bytes per bank)
BG_MAP_ADDR = 0x9800          # BG tilemap (32×32 tile indices/attributes)
VRAM_TILE_BYTES = 0x1800      # 6144 bytes of tile data per bank

# Palette WRAM buffer (accessible via WRAM bank 2)
WRAM_TILESET_PALETTES = 0xDE80  # wTilesetPalettes — 8 BG palettes × 8 bytes

# Room flags
OVERWORLD_ROOM_FLAGS = 0xC700 # wOverworldRoomFlags (256 bytes)
SUBROSIA_ROOM_FLAGS = 0xC800  # wSubrosiaRoomFlags

# Screen dimensions
GB_SCREEN_W = 160
GB_SCREEN_H = 144
HUD_HEIGHT = 16               # Top 16px is the HUD
ROOM_W = 160                  # Pixels per room (width)
ROOM_H = 128                  # Pixels per room (height, after HUD crop)

# Grid layout
GRID_COLS = 16
GRID_ROWS = 16
TOTAL_ROOMS = GRID_COLS * GRID_ROWS  # 256

# Season constants
SEASONS = {0: "spring", 1: "summer", 2: "autumn", 3: "winter"}


# =============================================================================
# Community map download (VGMaps.com, maps by TerraEsperZ)
# =============================================================================

# Pixel-accurate maps hosted on VGMaps.com. These are 2720x2048 PNG images
# with an extra 160px padding column on the left that must be cropped.
VGMAPS_BASE = "https://www.vgmaps.com/Atlas/GB-GBC"
COMMUNITY_MAPS = {
    "default": f"{VGMAPS_BASE}/LegendOfZelda-OracleOfSeasons-Holodrum(Default).png",
    "spring": f"{VGMAPS_BASE}/LegendOfZelda-OracleOfSeasons-Holodrum(Spring).png",
    "summer": f"{VGMAPS_BASE}/LegendOfZelda-OracleOfSeasons-Holodrum(Summer).png",
    "autumn": f"{VGMAPS_BASE}/LegendOfZelda-OracleOfSeasons-Holodrum(Fall).png",
    "winter": f"{VGMAPS_BASE}/LegendOfZelda-OracleOfSeasons-Holodrum(Winter).png",
    "subrosia": f"{VGMAPS_BASE}/LegendOfZelda-OracleOfSeasons-Subrosia.png",
}


def download_community_map(
    variant: str = "default",
    output_path: Path | None = None,
) -> Image.Image:
    """Download a pixel-accurate overworld map from VGMaps.com.

    Maps by TerraEsperZ. The raw images are 2720x2048 with a 160px
    padding column on the left; we crop to get a clean 2560x2048 grid.

    Args:
        variant: Map variant — "default", "spring", "summer", "autumn",
                 "winter", or "subrosia".
        output_path: Where to save the cropped PNG.

    Returns:
        PIL Image (2560x2048 RGB).
    """
    url = COMMUNITY_MAPS.get(variant)
    if url is None:
        raise ValueError(
            f"Unknown map variant '{variant}'. "
            f"Available: {list(COMMUNITY_MAPS.keys())}"
        )

    logger.info(f"Downloading {variant} map from VGMaps.com...")
    logger.info(f"  URL: {url}")

    # Download to a temp buffer
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw_bytes = resp.read()

    raw_img = Image.open(io.BytesIO(raw_bytes))
    logger.info(f"  Raw size: {raw_img.size}, mode: {raw_img.mode}")

    # Convert from palette to RGB BEFORE cropping (palette mode crops
    # can produce incorrect colors)
    rgb_img = raw_img.convert("RGB")

    # Crop the 160px left padding to get clean 2560x2048
    w, h = rgb_img.size
    if w == 2720 and h == 2048:
        cropped = rgb_img.crop((160, 0, 2720, 2048))
    elif w == 2560 and h == 2048:
        cropped = rgb_img  # Already correct size
    else:
        logger.warning(f"  Unexpected size {w}x{h}, using as-is")
        cropped = rgb_img

    logger.info(f"  Final size: {cropped.size}")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(str(output_path), "PNG")
        logger.info(f"  Saved to {output_path}")

    return cropped


def find_rom_and_state(project_root: Path) -> tuple[Path, Path]:
    """Locate the ROM and save state files."""
    rom_candidates = [
        project_root / "roms" / "zelda.gbc",
        project_root / "new" / "ignored" / "zelda.gbc",
    ]
    state_candidates = [
        project_root / "new" / "ignored" / "zelda.gbc.pre-maku.state",
        project_root / "new" / "ignored" / "zelda.gbc.sword.state",
        project_root / "new" / "ignored" / "zelda.gbc.start.state",
    ]
    rom_path = None
    for p in rom_candidates:
        if p.exists():
            rom_path = p
            break
    state_path = None
    for p in state_candidates:
        if p.exists():
            state_path = p
            break
    if rom_path is None:
        raise FileNotFoundError(
            f"ROM not found. Searched: {[str(p) for p in rom_candidates]}"
        )
    if state_path is None:
        raise FileNotFoundError(
            f"Save state not found. Searched: {[str(p) for p in state_candidates]}"
        )
    return rom_path, state_path


# =============================================================================
# VRAM rendering — software PPU for reading tile data when LCD is disabled
# =============================================================================
#
# After a fadeout warp (WARP_TRANSITION2=0x03), the game loads the correct
# tileset and tilemap into VRAM but never re-enables the LCD (the fade-in
# step is skipped).  Instead of fighting the game engine, we read the raw
# VRAM data and reconstruct the room image in Python.
#
# GBC VRAM layout:
#   Bank 0: $8000-$97FF  tile pixel data (2bpp, 16 bytes per 8×8 tile)
#           $9800-$9BFF  BG tilemap — 32×32 grid of tile indices
#   Bank 1: $8000-$97FF  additional tile pixel data
#           $9800-$9BFF  BG tile attributes (palette, bank, flip flags)
#
# Tile attribute byte (VRAM bank 1 at $9800):
#   Bits 0-2: CGB palette number (0-7)
#   Bit 3:    Tile pixel data VRAM bank (0 or 1)
#   Bit 5:    Horizontal flip
#   Bit 6:    Vertical flip
#
# CGB color format: RGB555 little-endian (2 bytes per color, 4 colors/palette)
# =============================================================================


def _read_vram_bank(pyboy, bank: int, start: int, length: int) -> bytes:
    """Read a byte range from a specific VRAM bank (0 or 1)."""
    old = pyboy.memory[VBK_REG]
    pyboy.memory[VBK_REG] = bank & 1
    data = bytes(pyboy.memory[start + i] for i in range(length))
    pyboy.memory[VBK_REG] = old
    return data


def _read_wram_palettes(pyboy) -> list[tuple[int, int, int]]:
    """Read 8 BG palettes from WRAM bank 2 (wTilesetPalettes at $DE80).

    Returns a flat list of 32 RGB tuples (8 palettes × 4 colors).
    The game stores the correct room-specific palette data here even
    during fadeout transitions when hardware palettes are faded to white.
    """
    old = pyboy.memory[SVBK_REG]
    pyboy.memory[SVBK_REG] = 2
    raw = bytes(pyboy.memory[WRAM_TILESET_PALETTES + i] for i in range(64))
    pyboy.memory[SVBK_REG] = old

    colors = []
    for i in range(32):  # 8 palettes × 4 colors
        lo, hi = raw[i * 2], raw[i * 2 + 1]
        rgb555 = lo | (hi << 8)
        r = (rgb555 & 0x1F) << 3
        g = ((rgb555 >> 5) & 0x1F) << 3
        b = ((rgb555 >> 10) & 0x1F) << 3
        colors.append((r, g, b))
    return colors


def _tile_data_offset(tile_index: int) -> int:
    """Map a tilemap index to an offset into the $8000-$97FF tile data region.

    Oracle of Seasons uses LCDC bit 4 = 0 ($8800 addressing mode):
      Index 0..127   → $9000-$97F0  (signed positive)
      Index 128..255 → $8800-$8FF0  (signed negative / unsigned high)

    Returns offset relative to $8000.
    """
    signed = tile_index if tile_index < 128 else tile_index - 256
    return (0x9000 + signed * 16) - TILE_DATA_START


def _render_room_from_vram(pyboy) -> np.ndarray:
    """Render a 160×128 pixel room image directly from VRAM data.

    Reads the BG tilemap, tile attributes, tile pixel data, and palette
    colors, then reconstructs the image without relying on the LCD.

    The game area occupies tilemap rows 0-15, columns 0-19 (the Window
    layer at $9C00 covers the top 16px HUD, so the BG at $9800 is purely
    game world).

    Returns:
        128×160×3 numpy array (RGB).
    """
    # Read tilemap indices (VRAM bank 0) and attributes (VRAM bank 1)
    tilemap = _read_vram_bank(pyboy, 0, BG_MAP_ADDR, 32 * 32)
    attrs = _read_vram_bank(pyboy, 1, BG_MAP_ADDR, 32 * 32)

    # Read tile pixel data from both VRAM banks
    tile_data_b0 = _read_vram_bank(pyboy, 0, TILE_DATA_START, VRAM_TILE_BYTES)
    tile_data_b1 = _read_vram_bank(pyboy, 1, TILE_DATA_START, VRAM_TILE_BYTES)

    # Read palette colors from WRAM bank 2
    palette_colors = _read_wram_palettes(pyboy)

    # Render 20 columns × 16 rows of 8×8 tiles = 160×128 pixels
    img = np.zeros((ROOM_H, ROOM_W, 3), dtype=np.uint8)

    for tile_row in range(16):
        for tile_col in range(20):
            map_idx = tile_row * 32 + tile_col
            tidx = tilemap[map_idx]
            attr = attrs[map_idx]

            pal_num = attr & 0x07
            vram_bank = (attr >> 3) & 1
            h_flip = (attr >> 5) & 1
            v_flip = (attr >> 6) & 1

            # Look up tile pixel data (16 bytes, 2bpp)
            offset = _tile_data_offset(tidx)
            td = (tile_data_b1 if vram_bank else tile_data_b0)
            tile_bytes = td[offset:offset + 16]
            if len(tile_bytes) < 16:
                continue

            # Decode 2bpp tile: each row = 2 bytes (lo, hi)
            # Pixel value = (hi_bit << 1) | lo_bit  → 0..3
            pal_base = pal_num * 4
            for py in range(8):
                lo_byte = tile_bytes[py * 2]
                hi_byte = tile_bytes[py * 2 + 1]
                src_y = (7 - py) if v_flip else py
                iy = tile_row * 8 + src_y
                for px in range(8):
                    bit = 7 - px
                    ci = ((lo_byte >> bit) & 1) | (((hi_byte >> bit) & 1) << 1)
                    src_x = (7 - px) if h_flip else px
                    ix = tile_col * 8 + src_x
                    img[iy, ix] = palette_colors[pal_base + ci]

    return img


def warp_to_room(pyboy, save_state_bytes: bytes, room_id: int,
                 group: int = 0, season: int | None = None,
                 max_wait_frames: int = 500,
                 warp_pos: int = 0x55) -> np.ndarray | None:
    """Warp to a specific overworld room and render it from VRAM.

    Uses the game's fadeout warp transition (WARP_TRANSITION2=0x03) which
    runs the full room loading sequence — loadTilesetData, loadTilesetGraphics,
    generateVramTilesWithRoomChanges — populating VRAM with correct tile data.

    The fadeout leaves the LCD disabled (LCDC=0x00), so we bypass the screen
    entirely and reconstruct the room image by reading raw VRAM tile data
    and WRAM palette buffers.

    Approach developed by analyzing the oracles-disasm cutscene03 path and
    verified against rooms where the LCD naturally re-enables (identical output).

    Args:
        pyboy: PyBoy emulator instance.
        save_state_bytes: Save state to reload for clean starting point.
        room_id: Room index (0-255). Row = room_id // 16, Col = room_id % 16.
        group: Map group (0 = overworld, 1 = Subrosia).
        season: Season to force (0-3), or None for default from save state.
        max_wait_frames: Maximum frames to wait for room to load.

    Returns:
        128x160x3 numpy array (RGB) of the room, or None if capture failed.
    """
    # 1. Reload save state for a clean game state
    pyboy.load_state(io.BytesIO(save_state_bytes))
    for _ in range(10):
        pyboy.tick()

    # 2. Clear any menus or input state
    pyboy.memory[MENU_STATE] = 0
    pyboy.memory[KEYS_PRESSED] = 0x00
    pyboy.memory[KEYS_JUST_PRESSED] = 0x00
    for _ in range(5):
        pyboy.tick()

    # 3. Set season if requested (before warp so room loads with correct tiles)
    if season is not None:
        pyboy.memory[ROOM_STATE_MODIFIER] = season & 0x03

    # 4. Trigger fadeout warp — this runs the game's full room loading
    #    sequence (tileset + tilemap + palette) but stops before fade-in.
    #    Bit 7 on wWarpDestGroup = direct warp (bypass warpDestTable lookup)
    pyboy.memory[WARP_DEST_GROUP] = 0x80 | group
    pyboy.memory[WARP_DEST_ROOM] = room_id
    pyboy.memory[WARP_TRANSITION] = 0x00   # TRANSITION_DEST_BASIC
    pyboy.memory[WARP_DEST_POS] = warp_pos  # default: Row 5, Col 5 (center)
    pyboy.memory[WARP_TRANSITION2] = 0x03  # Fadeout (loads tiles fully)

    # 5. Wait for room change (~33 frames) + tilemap population (~10 more)
    room_changed = False
    for frame in range(max_wait_frames):
        pyboy.tick()
        if pyboy.memory[ACTIVE_ROOM] == room_id:
            room_changed = True
            # Clear warp state immediately — applyWarpDest has already
            # consumed these values.  If we leave WARP_DEST_GROUP=0x80,
            # the game detects a "pending warp" during the extra ticks
            # and re-triggers it, overwriting correct VRAM tile data.
            pyboy.memory[WARP_DEST_GROUP] = 0xFF
            pyboy.memory[WARP_DEST_ROOM] = 0x00
            pyboy.memory[WARP_TRANSITION] = 0x00
            pyboy.memory[WARP_TRANSITION2] = 0x00
            pyboy.memory[KEYS_PRESSED] = 0x00
            pyboy.memory[KEYS_JUST_PRESSED] = 0x00
            # Extra frames for generateVramTilesWithRoomChanges to finish.
            # Monitor for secondary warps (some rooms have auto-cave scripts
            # that change the group during tile loading).
            for extra in range(60):
                pyboy.tick()
                if pyboy.memory[ACTIVE_GROUP] != group:
                    # Secondary warp detected — render what we have so far.
                    # The tileset was loaded for the target room before the
                    # script changed the group, so VRAM may be usable.
                    logger.info(
                        f"  Room {room_id}: secondary warp at frame +{extra}, "
                        f"rendering early VRAM snapshot"
                    )
                    # Restore group/room so _render_room_from_vram reads
                    # the correct palettes
                    pyboy.memory[ACTIVE_GROUP] = group
                    pyboy.memory[ACTIVE_ROOM] = room_id
                    return _render_room_from_vram(pyboy)
            break

    if not room_changed:
        logger.warning(f"Room {room_id}: fadeout warp did not complete")
        return None

    # 6. Verify room
    actual_room = pyboy.memory[ACTIVE_ROOM]
    actual_group = pyboy.memory[ACTIVE_GROUP]
    if actual_room != room_id or actual_group != group:
        logger.warning(
            f"Room mismatch for room {room_id}: "
            f"got group={actual_group}, room={actual_room}"
        )
        return None

    # 7. Render from VRAM (bypasses disabled LCD entirely)
    return _render_room_from_vram(pyboy)


def generate_overworld(rom_path: Path, state_path: Path,
                       output_path: Path, season: int | None = None,
                       subrosia: bool = False) -> Image.Image:
    """Generate a full overworld map image.

    Args:
        rom_path: Path to the Oracle of Seasons ROM.
        state_path: Path to a save state file.
        output_path: Where to save the resulting PNG.
        season: Force a specific season (0-3), or None for default.
        subrosia: If True, generate Subrosia map (group 1) instead.

    Returns:
        PIL Image of the complete map.
    """
    from pyboy import PyBoy

    group = 1 if subrosia else 0
    group_name = "Subrosia" if subrosia else "Overworld"
    season_name = SEASONS.get(season, "default") if not subrosia else "n/a"

    logger.info(f"Generating {group_name} map (season={season_name})")
    logger.info(f"ROM: {rom_path}")
    logger.info(f"State: {state_path}")
    logger.info(f"Output: {output_path}")

    # Initialize PyBoy in headless mode
    pyboy = PyBoy(str(rom_path), window="null", sound_emulated=False)

    # Boot past logo
    for _ in range(1000):
        pyboy.tick()

    # Load and cache the save state
    with open(state_path, "rb") as f:
        save_state_bytes = f.read()
    pyboy.load_state(io.BytesIO(save_state_bytes))

    # Create the composite image
    map_w = GRID_COLS * ROOM_W  # 2560
    map_h = GRID_ROWS * ROOM_H  # 2048
    composite = Image.new("RGB", (map_w, map_h), color=(0, 0, 0))

    success_count = 0
    fail_count = 0
    room_metadata = {}
    start_time = time.time()

    for room_id in range(TOTAL_ROOMS):
        row = room_id // GRID_COLS
        col = room_id % GRID_COLS
        progress = (room_id + 1) / TOTAL_ROOMS * 100

        logger.info(
            f"[{progress:5.1f}%] Room {room_id:3d} (row={row}, col={col})"
        )

        room_frame = warp_to_room(
            pyboy, save_state_bytes, room_id,
            group=group, season=season,
        )

        # Retry with different spawn positions if the default fails
        # (some rooms have warp tiles at the center that redirect to caves)
        if room_frame is None:
            for retry_pos in [0x00, 0x09, 0x70, 0x77]:
                logger.info(f"  Retrying room {room_id} with pos=0x{retry_pos:02X}")
                room_frame = warp_to_room(
                    pyboy, save_state_bytes, room_id,
                    group=group, season=season,
                    warp_pos=retry_pos,
                )
                if room_frame is not None:
                    break

        if room_frame is not None:
            room_img = Image.fromarray(room_frame)
            x_offset = col * ROOM_W
            y_offset = row * ROOM_H
            composite.paste(room_img, (x_offset, y_offset))
            success_count += 1
            room_metadata[room_id] = {"status": "ok", "row": row, "col": col}
        else:
            fail_count += 1
            room_metadata[room_id] = {"status": "failed", "row": row, "col": col}

    elapsed = time.time() - start_time
    logger.info(
        f"Done! {success_count}/{TOTAL_ROOMS} rooms captured "
        f"({fail_count} failed) in {elapsed:.1f}s"
    )

    # Save the composite
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(str(output_path), "PNG")
    logger.info(f"Saved {map_w}x{map_h} map to {output_path}")

    # Save metadata alongside
    meta_path = output_path.with_suffix(".json")
    metadata = {
        "group": group,
        "group_name": group_name,
        "season": season,
        "season_name": season_name,
        "grid_cols": GRID_COLS,
        "grid_rows": GRID_ROWS,
        "room_width_px": ROOM_W,
        "room_height_px": ROOM_H,
        "total_width_px": map_w,
        "total_height_px": map_h,
        "tiles_per_room_x": 10,
        "tiles_per_room_y": 8,
        "tile_size_px": 16,
        "success_count": success_count,
        "fail_count": fail_count,
        "elapsed_seconds": round(elapsed, 1),
        "rooms": {str(k): v for k, v in room_metadata.items()},
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {meta_path}")

    pyboy.stop()
    return composite


def main():
    parser = argparse.ArgumentParser(
        description="Generate Oracle of Seasons overworld map images.",
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
        "--rom", type=str, default=None,
        help="Path to Oracle of Seasons ROM (auto-detected if omitted)",
    )
    parser.add_argument(
        "--state", type=str, default=None,
        help="Path to PyBoy save state (auto-detected if omitted)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output PNG path (default: visualization/assets/overworld.png)",
    )
    parser.add_argument(
        "--season", type=int, choices=[0, 1, 2, 3], default=None,
        help="Force season: 0=spring, 1=summer, 2=autumn, 3=winter",
    )
    parser.add_argument(
        "--all-seasons", action="store_true",
        help="Generate maps for all 4 seasons",
    )
    parser.add_argument(
        "--subrosia", action="store_true",
        help="Generate Subrosia map instead of overworld",
    )
    parser.add_argument(
        "--download", type=str, nargs="?", const="default",
        choices=list(COMMUNITY_MAPS.keys()),
        help="Download a community map from VGMaps.com instead of generating "
             "from ROM. Maps by TerraEsperZ. Options: default, spring, summer, "
             "autumn, winter, subrosia",
    )
    parser.add_argument(
        "--download-all", action="store_true",
        help="Download all community map variants from VGMaps.com",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    assets_dir = Path(__file__).resolve().parent / "assets"

    # --- Download mode: fetch community maps from VGMaps.com ---
    if args.download_all:
        for variant, url in COMMUNITY_MAPS.items():
            if variant == "default":
                out = assets_dir / "overworld.png"
            elif variant == "subrosia":
                out = assets_dir / "subrosia.png"
            else:
                out = assets_dir / f"overworld_{variant}.png"
            download_community_map(variant, out)
        logger.info("All community maps downloaded!")
        return

    if args.download is not None:
        variant = args.download
        if args.output:
            out = Path(args.output)
        elif variant == "subrosia":
            out = assets_dir / "subrosia.png"
        elif variant == "default":
            out = assets_dir / "overworld.png"
        else:
            out = assets_dir / f"overworld_{variant}.png"
        download_community_map(variant, out)
        return

    # --- Generate mode: use PyBoy to warp and screenshot ---
    # Locate project root
    project_root = Path(__file__).resolve().parent.parent

    # Find ROM and state
    if args.rom:
        rom_path = Path(args.rom)
    else:
        rom_path, _ = find_rom_and_state(project_root)

    if args.state:
        state_path = Path(args.state)
    else:
        _, state_path = find_rom_and_state(project_root)

    if args.all_seasons:
        for season_id, season_name in SEASONS.items():
            out = assets_dir / f"overworld_{season_name}.png"
            logger.info(f"\n{'='*60}\nGenerating {season_name} map...\n{'='*60}")
            generate_overworld(rom_path, state_path, out, season=season_id)
        # Also generate default (from save state)
        out = assets_dir / "overworld.png"
        logger.info(f"\n{'='*60}\nGenerating default season map...\n{'='*60}")
        generate_overworld(rom_path, state_path, out, season=None)
    else:
        if args.output:
            out = Path(args.output)
        elif args.subrosia:
            out = assets_dir / "subrosia.png"
        elif args.season is not None:
            out = assets_dir / f"overworld_{SEASONS[args.season]}.png"
        else:
            out = assets_dir / "overworld.png"

        generate_overworld(
            rom_path, state_path, out,
            season=args.season,
            subrosia=args.subrosia,
        )


if __name__ == "__main__":
    main()
