#!/usr/bin/env python3
"""Interactive PyBoy tracer for Zelda: Oracle of Seasons.

Loads a ROM + save state, opens an SDL2 window for keyboard play,
and logs every room transition, milestone flag change, and player
coordinates.  Output: a route map from Maku Tree → Snow Region → Dungeon 1.

Controls (PyBoy SDL2 defaults):
  Arrow keys  = D-pad
  Z           = A button
  X           = B button
  Enter       = Start
  Backspace   = Select
  Esc         = Quit

Usage:
  python scripts/trace_route.py [--state STATE_FILE]
"""

import argparse
import datetime
import sys
import os
from pathlib import Path

# ── RAM addresses (from agent/env/ram_addresses.py) ─────────────────
PLAYER_ROOM       = 0xCC4C  # wActiveRoom
PLAYER_X           = 0xD00D  # pixel X
PLAYER_Y           = 0xD00B  # pixel Y
ACTIVE_GROUP       = 0xCC49
SWORD_LEVEL        = 0xC6AC
GNARLED_KEY_ADDR   = 0xC69A
GNARLED_KEY_MASK   = 0x04
GNARLED_KEY_GIVEN  = 0xC6CD
GNARLED_KEY_GIVEN_MASK = 0x01
OVERWORLD_ROOM_FLAGS = 0xC700
MAKU_GATE_ROOM     = 0xD9
ROOMFLAG_GATE_HIT  = 0x80
CURRENT_SEASON     = 0xCC4E
DUNGEON_INDEX      = 0xCC55
HEALTH             = 0xC6AA
MAX_HEALTH         = 0xC6AB
RUPEES_HIGH        = 0xC6A4
RUPEES_LOW         = 0xC6A5

# ── Season names ────────────────────────────────────────────────────
SEASONS = {0: "Spring", 1: "Summer", 2: "Autumn", 3: "Winter"}

# ── Known landmark rooms (overworld group 0) ────────────────────────
LANDMARKS = {
    0xD9: "Maku Gate",
    0xD8: "Horon Village (west)",
    0xC8: "North Horon",
    0xB8: "Eyeglass Lake area",
    0x98: "Tarm Ruins approach",
    0x88: "North Tarm",
    0x28: "Goron Mountain area",
    0x48: "Temple Remains area",
    0x90: "Spool Swamp",
    0x80: "Sunken City",
}

# ── Known dungeon names ─────────────────────────────────────────────
DUNGEONS = {
    0: "Hero's Cave (tutorial)",
    1: "Gnarled Root Dungeon (D1)",
    2: "Snake's Remains (D2)",
    3: "Poison Moth's Lair (D3)",
    4: "Dancing Dragon Dungeon (D4)",
    5: "Unicorn's Cave (D5)",
    6: "Ancient Ruins (D6)",
    7: "Explorer's Crypt (D7)",
    8: "Sword & Shield Maze (D8)",
}


class RewardEstimator:
    """Estimates what rewards the RewardWrapper would give at each room transition.

    Uses the same reward config as V40 training to show the agent's reward
    landscape as you walk the route.
    """

    def __init__(self):
        # Reward values matching V40 Job manifest
        self.directional_bonus = float(os.getenv("DIRECTIONAL_BONUS", "50"))
        self.new_room_bonus = float(os.getenv("NEW_ROOM_BONUS", "10"))
        self.snow_region_bonus = float(os.getenv("SNOW_REGION_BONUS", "2000"))
        self.dungeon_entry_bonus = float(os.getenv("DUNGEON_ENTRY", "3000"))
        self.gate_slash_bonus = float(os.getenv("GATE_SLASH", "2000"))
        self.maku_tree_visit = float(os.getenv("MAKU_TREE_VISIT", "1000"))
        self.gnarled_key_bonus = float(os.getenv("GNARLED_KEY", "2000"))
        self.grid_exploration = float(os.getenv("GRID_EXPLORATION", "0.1"))

        # Snow region bounds (V40: rows 9-11, cols 6-7)
        self.snow_min_row = int(os.getenv("SNOW_REGION_MIN_ROW", "9"))
        self.snow_max_row = int(os.getenv("SNOW_REGION_MAX_ROW", "11"))
        self.snow_min_col = int(os.getenv("SNOW_REGION_MIN_COL", "6"))
        self.snow_max_col = int(os.getenv("SNOW_REGION_MAX_COL", "7"))

        # Directional target (D1 entrance)
        self.target_row = int(os.getenv("DIRECTIONAL_TARGET_ROW", "9"))
        self.target_col = int(os.getenv("DIRECTIONAL_TARGET_COL", "6"))

        # Tracking state
        self.min_target_distance = None
        self.visited_rooms: set[int] = set()
        self.entered_snow = False
        self.entered_dungeon = False
        self.total_reward = 0.0

    def estimate_room_reward(self, state: dict, prev_group: int | None) -> list[str]:
        """Return list of reward descriptions for a room transition."""
        rewards = []
        group = state["group"]
        room = state["room"]
        row = state["row"]
        col = state["col"]

        if group != 0:
            # Dungeon entry check
            if (group >= 4
                    and 1 <= state["dungeon"] < 0xFF
                    and not self.entered_dungeon):
                self.entered_dungeon = True
                self.total_reward += self.dungeon_entry_bonus
                rewards.append(
                    f"  💰 DUNGEON ENTRY: +{self.dungeon_entry_bonus:.0f}"
                )
            return rewards

        # --- Overworld rewards ---

        # New room bonus
        if room not in self.visited_rooms:
            self.visited_rooms.add(room)
            self.total_reward += self.new_room_bonus
            rewards.append(f"  💰 New room: +{self.new_room_bonus:.0f}")

        # Directional bonus (Manhattan distance reduction to target)
        target_dist = abs(row - self.target_row) + abs(col - self.target_col)
        if self.min_target_distance is None:
            self.min_target_distance = target_dist
            rewards.append(
                f"  📍 Distance to ({self.target_row},{self.target_col}): "
                f"{target_dist} rooms (baseline set)"
            )
        elif target_dist < self.min_target_distance:
            delta = self.min_target_distance - target_dist
            dir_rew = delta * self.directional_bonus
            self.total_reward += dir_rew
            self.min_target_distance = target_dist
            rewards.append(
                f"  💰 Directional (closer by {delta}): +{dir_rew:.0f}  "
                f"[dist now {target_dist}]"
            )
        elif target_dist > self.min_target_distance:
            rewards.append(
                f"  ⚠️  Directional: moved AWAY (dist {target_dist}, "
                f"best was {self.min_target_distance}) — no reward"
            )
        else:
            rewards.append(
                f"  📍 Directional: same distance ({target_dist}) — no reward"
            )

        # Snow region check
        if (not self.entered_snow
                and self.snow_min_row <= row <= self.snow_max_row
                and self.snow_min_col <= col <= self.snow_max_col):
            self.entered_snow = True
            self.total_reward += self.snow_region_bonus
            rewards.append(
                f"  💰 SNOW REGION ENTRY: +{self.snow_region_bonus:.0f}"
            )

        # D1 entrance room check (currently NO bonus — the gap!)
        if row == self.target_row and col == self.target_col:
            rewards.append(
                f"  ⭐ D1 ENTRANCE ROOM ({row},{col}) — "
                f"NO specific bonus exists! (gap in reward)"
            )

        rewards.append(f"  📊 Running total: {self.total_reward:.0f}")
        return rewards


class RouteTracer:
    """Tracks room transitions and game state during interactive play."""

    def __init__(self):
        self.route: list[dict] = []
        self.milestones: list[dict] = []
        self.prev_room = None
        self.prev_group = None
        self.prev_gate_slashed = False
        self.prev_has_key = False
        self.prev_key_given = False
        self.prev_dungeon = 0
        self.frame_count = 0
        self.start_time = datetime.datetime.now()
        self.reward_estimator = RewardEstimator()

    def read_state(self, pyboy) -> dict:
        """Read all relevant RAM values."""
        mem = pyboy.memory
        room = mem[PLAYER_ROOM]
        group = mem[ACTIVE_GROUP]
        gate_flags = mem[OVERWORLD_ROOM_FLAGS + MAKU_GATE_ROOM]

        return {
            "room": room,
            "group": group,
            "row": room >> 4,
            "col": room & 0x0F,
            "px": mem[PLAYER_X],
            "py": mem[PLAYER_Y],
            "season": mem[CURRENT_SEASON],
            "sword": mem[SWORD_LEVEL],
            "gate_slashed": bool(gate_flags & ROOMFLAG_GATE_HIT),
            "has_key": bool(mem[GNARLED_KEY_ADDR] & GNARLED_KEY_MASK),
            "key_given": bool(mem[GNARLED_KEY_GIVEN] & GNARLED_KEY_GIVEN_MASK),
            "dungeon": mem[DUNGEON_INDEX],
            "health": mem[HEALTH],
            "max_health": mem[MAX_HEALTH],
            "rupees": mem[RUPEES_HIGH] * 256 + mem[RUPEES_LOW],
        }

    def check_transitions(self, state: dict):
        """Detect and log room transitions and milestone changes."""
        room = state["room"]
        group = state["group"]

        # ── Room transition ─────────────────────────────────────────
        if room != self.prev_room or group != self.prev_group:
            landmark = ""
            if group == 0 and room in LANDMARKS:
                landmark = f"  [{LANDMARKS[room]}]"
            elif group >= 4 and state["dungeon"] > 0:
                dname = DUNGEONS.get(state["dungeon"], f"Dungeon {state['dungeon']}")
                landmark = f"  [{dname}]"
            elif group == 2:
                landmark = "  [Indoor / Maku Tree]"
            elif group == 1:
                landmark = "  [Cave / Interior]"

            season_name = SEASONS.get(state["season"], "?")
            entry = {
                "frame": self.frame_count,
                "elapsed": str(datetime.datetime.now() - self.start_time).split(".")[0],
                "group": group,
                "room": room,
                "row": state["row"],
                "col": state["col"],
                "px": state["px"],
                "py": state["py"],
                "season": season_name,
                "landmark": landmark.strip(" []") if landmark else "",
            }
            self.route.append(entry)

            direction = ""
            if self.prev_room is not None and self.prev_group == group:
                dr = state["row"] - (self.prev_room >> 4)
                dc = state["col"] - (self.prev_room & 0x0F)
                dirs = []
                if dr < 0: dirs.append("North")
                if dr > 0: dirs.append("South")
                if dc < 0: dirs.append("West")
                if dc > 0: dirs.append("East")
                direction = " → " + "/".join(dirs) if dirs else " → Same"

            print(
                f"[{entry['elapsed']}] ROOM: group={group} room=0x{room:02X} "
                f"({state['row']},{state['col']}) px=({state['px']},{state['py']}) "
                f"season={season_name}{direction}{landmark}"
            )

            # Show estimated rewards for this room transition
            reward_lines = self.reward_estimator.estimate_room_reward(
                state, self.prev_group
            )
            for line in reward_lines:
                print(line)

            self.prev_room = room
            self.prev_group = group

        # ── Milestone: Gate slashed ─────────────────────────────────
        if state["gate_slashed"] and not self.prev_gate_slashed:
            ms = {"frame": self.frame_count, "type": "GATE_SLASHED",
                  "room": room, "group": group}
            self.milestones.append(ms)
            print(f"  *** MILESTONE: Gate slashed! ***")
            self.prev_gate_slashed = True

        # ── Milestone: Gnarled Key obtained ─────────────────────────
        if state["has_key"] and not self.prev_has_key:
            ms = {"frame": self.frame_count, "type": "GNARLED_KEY_OBTAINED",
                  "room": room, "group": group}
            self.milestones.append(ms)
            print(f"  *** MILESTONE: Gnarled Key obtained! ***")
            self.prev_has_key = True

        # ── Milestone: Key given to Maku Tree ───────────────────────
        if state["key_given"] and not self.prev_key_given:
            ms = {"frame": self.frame_count, "type": "GNARLED_KEY_GIVEN",
                  "room": room, "group": group}
            self.milestones.append(ms)
            print(f"  *** MILESTONE: Gnarled Key given to Maku Tree! ***")
            self.prev_key_given = True

        # ── Milestone: Entered dungeon ──────────────────────────────
        if state["dungeon"] > 0 and state["dungeon"] != self.prev_dungeon:
            dname = DUNGEONS.get(state["dungeon"], f"Dungeon {state['dungeon']}")
            ms = {"frame": self.frame_count, "type": "ENTERED_DUNGEON",
                  "dungeon": state["dungeon"], "name": dname,
                  "room": room, "group": group}
            self.milestones.append(ms)
            print(f"  *** MILESTONE: Entered {dname}! ***")
        self.prev_dungeon = state["dungeon"]

    def print_summary(self):
        """Print the complete route summary and ASCII map."""
        elapsed = str(datetime.datetime.now() - self.start_time).split(".")[0]
        print("\n" + "=" * 70)
        print(f"ROUTE SUMMARY  ({len(self.route)} rooms visited in {elapsed})")
        print("=" * 70)

        # ── Route list ──────────────────────────────────────────────
        print("\nRoom sequence:")
        for i, r in enumerate(self.route):
            landmark = f"  [{r['landmark']}]" if r.get("landmark") else ""
            print(
                f"  {i+1:3d}. [{r['elapsed']}] group={r['group']} "
                f"room=0x{r['room']:02X} ({r['row']:2d},{r['col']:2d}) "
                f"season={r['season']}{landmark}"
            )

        # ── Milestones ──────────────────────────────────────────────
        if self.milestones:
            print(f"\nMilestones achieved ({len(self.milestones)}):")
            for ms in self.milestones:
                print(f"  - {ms['type']} (frame {ms['frame']})")

        # ── ASCII overworld map ─────────────────────────────────────
        overworld_rooms = [
            r for r in self.route if r["group"] == 0
        ]
        if overworld_rooms:
            print("\nOverworld route map (group 0):")
            print("  Row/Col", end="")
            min_col = min(r["col"] for r in overworld_rooms)
            max_col = max(r["col"] for r in overworld_rooms)
            min_row = min(r["row"] for r in overworld_rooms)
            max_row = max(r["row"] for r in overworld_rooms)

            # Header
            for c in range(min_col, max_col + 1):
                print(f" {c:3d}", end="")
            print()

            # Build visit order map
            visit_order = {}
            for i, r in enumerate(overworld_rooms):
                key = (r["row"], r["col"])
                if key not in visit_order:
                    visit_order[key] = i + 1

            # Grid
            for row in range(min_row, max_row + 1):
                print(f"  {row:3d}   ", end="")
                for col in range(min_col, max_col + 1):
                    key = (row, col)
                    if key in visit_order:
                        print(f" {visit_order[key]:3d}", end="")
                    else:
                        print("   .", end="")
                print()

        # ── Directional target analysis ─────────────────────────────
        if len(overworld_rooms) >= 2:
            print("\nDirectional target analysis:")
            start = overworld_rooms[0]
            end = overworld_rooms[-1]
            print(f"  Start: ({start['row']},{start['col']})  "
                  f"room=0x{start['room']:02X}")
            print(f"  End:   ({end['row']},{end['col']})  "
                  f"room=0x{end['room']:02X}")
            print(f"  Net direction: "
                  f"dRow={end['row'] - start['row']:+d} "
                  f"dCol={end['col'] - start['col']:+d}")

            # Snow region analysis
            snow_rooms = [r for r in overworld_rooms if r["row"] <= 7]
            if snow_rooms:
                print(f"\n  Snow region rooms visited (row <= 7): {len(snow_rooms)}")
                for r in snow_rooms:
                    print(f"    ({r['row']},{r['col']}) room=0x{r['room']:02X}")

        # ── Reward summary ─────────────────────────────────────────
        est = self.reward_estimator
        print(f"\nReward estimate (V40 config):")
        print(f"  Total accumulated:    {est.total_reward:.0f}")
        print(f"  Rooms visited:        {len(est.visited_rooms)}")
        print(f"  Snow region entered:  {'YES' if est.entered_snow else 'NO'}")
        print(f"  Dungeon entered:      {'YES' if est.entered_dungeon else 'NO'}")
        print(f"  Best distance to D1:  {est.min_target_distance}")
        print(f"\n  Reward breakdown (V40 values):")
        print(f"    Snow region bonus:  {est.snow_region_bonus:.0f} (one-time)")
        print(f"    Dungeon entry:      {est.dungeon_entry_bonus:.0f} (one-time)")
        print(f"    Directional/room:   {est.directional_bonus:.0f} per Manhattan step")
        print(f"    New room/room:      {est.new_room_bonus:.0f} per unique room")
        if not est.entered_dungeon and est.min_target_distance == 0:
            print(f"\n  ⚠️  Reached D1 entrance room but NO specific room bonus!")
            print(f"      Consider adding a D1_ENTRANCE_BONUS (e.g. +1500)")

        # ── Output route as CSV for easy import ─────────────────────
        csv_path = Path(__file__).parent / "route_trace.csv"
        with open(csv_path, "w") as f:
            f.write("step,elapsed,group,room_hex,row,col,px,py,season,landmark\n")
            for i, r in enumerate(self.route):
                f.write(
                    f"{i+1},{r['elapsed']},{r['group']},0x{r['room']:02X},"
                    f"{r['row']},{r['col']},{r['px']},{r['py']},"
                    f"{r['season']},{r['landmark']}\n"
                )
        print(f"\nRoute saved to: {csv_path}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Interactive Zelda route tracer")
    parser.add_argument(
        "--rom",
        default="new/ignored/zelda.gbc",
        help="Path to Zelda GBC ROM",
    )
    parser.add_argument(
        "--state",
        default="new/ignored/zelda.gbc.after-maku.state",
        help="Path to save state file (use after-maku for D1 route testing)",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=1,
        help="Emulation speed multiplier (0=unlimited, 1=normal, 2=2x, etc.)",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    rom_path = Path(args.rom)
    state_path = Path(args.state)

    if not rom_path.is_absolute():
        rom_path = project_root / rom_path
    if not state_path.is_absolute():
        state_path = project_root / state_path

    if not rom_path.exists():
        print(f"ERROR: ROM not found: {rom_path}")
        sys.exit(1)
    if not state_path.exists():
        print(f"ERROR: Save state not found: {state_path}")
        sys.exit(1)

    # ── Launch PyBoy ────────────────────────────────────────────────
    try:
        from pyboy import PyBoy
    except ImportError:
        print("ERROR: PyBoy not installed. Run: pip install pyboy")
        sys.exit(1)

    print(f"Loading ROM: {rom_path}")
    print(f"Loading state: {state_path}")
    print()
    print("Controls:")
    print("  Arrow keys = D-pad")
    print("  Z = A button (sword/confirm)")
    print("  X = B button (cancel/use item)")
    print("  Enter = Start")
    print("  Backspace = Select")
    print("  Esc = Quit")
    print()
    print("Walk from Maku Tree area → Snow Region → Dungeon 1")
    print("Room transitions will be logged below.")
    print("-" * 70)

    pyboy = PyBoy(str(rom_path), window="SDL2", sound=False)

    # Load save state
    with open(state_path, "rb") as f:
        pyboy.load_state(f)
    print("Save state loaded successfully!")

    # Set emulation speed
    pyboy.set_emulation_speed(args.speed)

    # ── Main loop ───────────────────────────────────────────────────
    tracer = RouteTracer()

    # Read initial state
    initial = tracer.read_state(pyboy)
    print(
        f"\nInitial state: group={initial['group']} room=0x{initial['room']:02X} "
        f"({initial['row']},{initial['col']}) "
        f"sword={initial['sword']} gate={'YES' if initial['gate_slashed'] else 'NO'} "
        f"key={'YES' if initial['has_key'] else 'NO'} "
        f"health={initial['health']}/{initial['max_health']} "
        f"season={SEASONS.get(initial['season'], '?')}"
    )
    print()

    # Set initial baselines
    tracer.prev_gate_slashed = initial["gate_slashed"]
    tracer.prev_has_key = initial["has_key"]
    tracer.prev_key_given = initial["key_given"]
    tracer.prev_dungeon = initial["dungeon"]

    try:
        while pyboy.tick():
            tracer.frame_count += 1
            # Check every 4 frames (15 Hz) to reduce overhead
            if tracer.frame_count % 4 == 0:
                state = tracer.read_state(pyboy)
                tracer.check_transitions(state)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    # ── Summary ─────────────────────────────────────────────────────
    tracer.print_summary()
    pyboy.stop()


if __name__ == "__main__":
    main()
