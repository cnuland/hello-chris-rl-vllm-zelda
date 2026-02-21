#!/usr/bin/env python3
"""Extract Link's directional sprites from Oracle of Seasons ROM.

Loads the ROM + save state, faces Link in each direction by writing
directly to the direction RAM address, captures the screen, and crops
the 16×16 character sprite. Produces a 64×16 sprite sheet:

  [down 16×16] [up 16×16] [right 16×16] [left 16×16]

Direction indices match the game's w1Link.direction values:
  0 = up, 1 = right, 2 = down, 3 = left

Usage:
    python visualization/extract_link_sprites.py [--output PATH] [--scale N]

License: MIT
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Oracle of Seasons RAM addresses
PLAYER_Y     = 0xD00B   # w1Link.yPixel
PLAYER_X     = 0xD00D   # w1Link.xPixel
PLAYER_DIR   = 0xD008   # w1Link.direction (0=up, 1=right, 2=down, 3=left)
PLAYER_ANIM  = 0xD004   # w1Link.animMode
LCDC_REG     = 0xFF40   # LCD Control register (bit 1 = OBJ/sprite enable)

# Link's sprite is 16×16 pixels on the 160×144 screen.
SPRITE_W = 16
SPRITE_H = 16


def find_rom_and_state(project_root: Path):
    """Locate ROM and save state (same logic as generate_overworld.py)."""
    rom_candidates = [
        project_root / "roms" / "zelda.gbc",
        project_root / "new" / "ignored" / "zelda.gbc",
    ]
    state_candidates = [
        project_root / "new" / "ignored" / "zelda.gbc.pre-maku.state",
        project_root / "new" / "ignored" / "zelda.gbc.sword.state",
        project_root / "new" / "ignored" / "zelda.gbc.start.state",
    ]
    rom_path = next((p for p in rom_candidates if p.exists()), None)
    state_path = next((p for p in state_candidates if p.exists()), None)
    if rom_path is None:
        raise FileNotFoundError(f"ROM not found: {rom_candidates}")
    if state_path is None:
        raise FileNotFoundError(f"State not found: {state_candidates}")
    return rom_path, state_path


def extract_sprites(rom_path: Path, state_path: Path, scale: int = 1) -> Image.Image:
    """Extract Link's 4 directional sprites from the ROM.

    Returns a 64×16 RGBA sprite sheet (or scaled up by `scale`).
    """
    from pyboy import PyBoy

    pyboy = PyBoy(str(rom_path), window="null", sound_emulated=False)

    # Boot past logo
    for _ in range(1000):
        pyboy.tick()

    # Load save state
    with open(state_path, "rb") as f:
        state_bytes = f.read()
    pyboy.load_state(io.BytesIO(state_bytes))

    # Let the game settle for a moment
    for _ in range(60):
        pyboy.tick()

    # Direction order for the sprite sheet: down, up, right, left
    # (matches the offline_renderer.py convention)
    # PyBoy button names for each direction
    dir_buttons = ["down", "up", "right", "left"]
    dir_names   = ["down", "up", "right", "left"]
    dir_values  = [2, 0, 1, 3]  # game's internal direction codes

    sprites = []
    for dir_idx, (btn, dir_val) in enumerate(zip(dir_buttons, dir_values)):
        # Reload state for clean capture each time
        pyboy.load_state(io.BytesIO(state_bytes))
        for _ in range(30):
            pyboy.tick()

        # Press the direction button for a few frames to make Link face
        # that way (this triggers the game engine's animation system
        # which updates the OAM sprite tiles properly)
        for _ in range(8):
            pyboy.button(btn)
            pyboy.tick()

        # Release and let the standing animation settle
        for _ in range(6):
            pyboy.tick()

        # --- Dual-capture technique ---
        # Capture WITH sprites, then move all OAM sprites off-screen
        # and capture WITHOUT. The pixel difference = sprite-only pixels.

        # 1) Capture with sprites (normal rendering)
        screen_with = pyboy.screen.image.copy()

        # 2) Save OAM data, then zero all 40 OAM Y-positions to move
        #    sprites off-screen (Y=0 → screen_y=-16 → invisible)
        oam_backup = bytes(pyboy.memory[0xFE00 + i] for i in range(160))
        for i in range(40):
            pyboy.memory[0xFE00 + i * 4] = 0  # Y = 0 → off-screen
        pyboy.tick()
        screen_without = pyboy.screen.image.copy()

        # 3) Restore OAM
        for i, b in enumerate(oam_backup):
            pyboy.memory[0xFE00 + i] = b

        # Read Link's position from RAM
        px = pyboy.memory[PLAYER_X]
        py = pyboy.memory[PLAYER_Y]

        # The visual sprite top-left offset from position registers.
        # Empirically verified via OAM readout:
        #   screen_x = PLAYER_X - 8
        #   screen_y = PLAYER_Y + 8  (= room_y + 16_hud - 8_anchor)
        sprite_left = px - 8
        sprite_top = py + 8

        # Clamp to screen bounds
        sprite_left = max(0, min(sprite_left, 160 - SPRITE_W))
        sprite_top = max(0, min(sprite_top, 144 - SPRITE_H))

        box = (sprite_left, sprite_top,
               sprite_left + SPRITE_W, sprite_top + SPRITE_H)

        crop_with = np.array(screen_with.convert("RGB").crop(box))
        crop_without = np.array(screen_without.convert("RGB").crop(box))

        # Build alpha mask: wherever pixels differ, that's a sprite pixel
        diff = np.any(crop_with != crop_without, axis=2)
        rgba = np.zeros((SPRITE_H, SPRITE_W, 4), dtype=np.uint8)
        rgba[:, :, :3] = crop_with
        rgba[:, :, 3] = np.where(diff, 255, 0)

        sprite_img = Image.fromarray(rgba, "RGBA")
        sprites.append(sprite_img)

        opaque = int(diff.sum())
        print(f"  [{dir_names[dir_idx]}] pos=({px},{py}) crop=({sprite_left},{sprite_top}) opaque_px={opaque}/{SPRITE_W*SPRITE_H}")

    pyboy.stop()

    # Stitch into a 64×16 sprite sheet
    sheet = Image.new("RGBA", (SPRITE_W * 4, SPRITE_H), (0, 0, 0, 0))
    for i, sprite in enumerate(sprites):
        sheet.paste(sprite, (i * SPRITE_W, 0))

    if scale > 1:
        sheet = sheet.resize(
            (sheet.width * scale, sheet.height * scale),
            Image.NEAREST,
        )

    return sheet


def main():
    parser = argparse.ArgumentParser(description="Extract Link sprites from Oracle of Seasons ROM")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output path (default: visualization/web/assets/link_sprites.png)")
    parser.add_argument("--scale", type=int, default=1,
                        help="Scale factor for output (default: 1, native 16×16)")
    parser.add_argument("--rom", type=str, default=None, help="Path to ROM file")
    parser.add_argument("--state", type=str, default=None, help="Path to save state")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    if args.rom and args.state:
        rom_path = Path(args.rom)
        state_path = Path(args.state)
    else:
        rom_path, state_path = find_rom_and_state(project_root)

    output = Path(args.output) if args.output else (
        project_root / "visualization" / "web" / "assets" / "link_sprites.png"
    )

    print(f"ROM:    {rom_path}")
    print(f"State:  {state_path}")
    print(f"Output: {output}")
    print(f"Scale:  {args.scale}x")
    print()

    sheet = extract_sprites(rom_path, state_path, scale=args.scale)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(str(output), "PNG")
    print(f"\nSaved sprite sheet: {output} ({sheet.size[0]}×{sheet.size[1]})")


if __name__ == "__main__":
    main()
