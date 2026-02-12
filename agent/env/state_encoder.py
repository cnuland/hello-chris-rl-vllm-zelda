"""State encoder: RAM → 128-D vector + structured JSON for LLM.

Vector features (128-D normalized [0,1]):
  0-3:   Position & direction (x, y, dir, room)
  4-6:   Health (current, max, ratio)
  7-12:  Resources (rupees, keys, sword, shield, bombs, ore)
  13-18: Seeds (6 types)
  19-23: Equipment flags
  24-25: Season (current, spirits)
  26-29: Dungeon progress
  30-33: Flags (dialog, puzzle, transition, overworld_pos)
  34-36: Entity counts
  37-44: Boss keys bitfield (8)
  45-64: OAM sprite proximity (20)
  65-68: Edge exit indicators (up, down, left, right)
  69-72: Ray-cast distances (tiles walkable in each direction)
  73-76: Exit distances (Manhattan dist to nearest exit per edge)
  77-78: Direction to nearest exit (dx, dy)
  79:    Active tile type
  80-83: Screen edge distances (left, right, up, down)
  84-87: Neighbor visited flags (N, S, E, W) — frontier awareness
  88-89: Frontier exit distance, any exit distance
  90-127: Reserved

JSON planner format:
  player:{x,y,dir,hp,max_hp}, room_id, inventory:{...}, flags:{dialog,puzzle,cutscene},
  interactables:[{type,x,y}].
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from agent.env.ram_addresses import (
    A_BUTTON_ITEM,
    ACTIVE_TILE_TYPE,
    B_BUTTON_ITEM,
    BOSS_KEYS,
    CURRENT_BOMBS,
    CURRENT_SEASON,
    DIALOG_STATE,
    DIRECTIONS,
    DUNGEON_COMPASS,
    DUNGEON_FLOOR,
    DUNGEON_KEYS,
    DUNGEON_MAP,
    EMBER_SEEDS,
    ENEMIES_COUNT,
    ESSENCES_COLLECTED,
    GALE_SEEDS,
    GASHA_SEEDS,
    ITEMS_COUNT,
    MAGNETIC_GLOVES,
    MAX_BOMBS,
    MYSTERY_SEEDS,
    NPCS_COUNT,
    OAM_BASE,
    ORE_CHUNKS,
    OVERWORLD_POSITION,
    PEGASUS_SEEDS,
    PLAYER_DIRECTION,
    PLAYER_HEALTH,
    PLAYER_MAX_HEALTH,
    PLAYER_ROOM,
    PLAYER_X,
    PLAYER_Y,
    PUZZLE_FLAGS,
    ROCS_FEATHER_LEVEL,
    RUPEES,
    SCENT_SEEDS,
    SCREEN_TRANSITION,
    SEASONS,
    SEASON_SPIRITS,
    SHIELD_LEVEL,
    SLINGSHOT_LEVEL,
    SWORD_LEVEL,
)

if TYPE_CHECKING:
    from agent.env.zelda_env import ZeldaEnv

# ---------------------------------------------------------------------------
# Vector encoder (128-D)
# ---------------------------------------------------------------------------

VECTOR_SIZE = 128


def encode_vector(env: ZeldaEnv) -> np.ndarray:
    """Build a 128-D normalized float32 observation from the environment.

    The layout packs the most decision-relevant features first so that
    even a shallow MLP can learn useful representations quickly.
    """
    v = np.zeros(VECTOR_SIZE, dtype=np.float32)
    r = env._read
    r16 = env._read16
    idx = 0

    # --- Player position & direction (4) ---
    v[idx] = r(PLAYER_X) / 255.0
    v[idx + 1] = r(PLAYER_Y) / 255.0
    v[idx + 2] = r(PLAYER_DIRECTION) / 3.0
    v[idx + 3] = r(PLAYER_ROOM) / 255.0
    idx += 4

    # --- Health (3) ---
    health_raw = r(PLAYER_HEALTH)
    max_health_raw = r(PLAYER_MAX_HEALTH)
    v[idx] = (health_raw / 4) / 20.0
    v[idx + 1] = (max_health_raw / 4) / 20.0
    v[idx + 2] = (health_raw / max(max_health_raw, 1))  # health ratio
    idx += 3

    # --- Resources (6) ---
    v[idx] = min(r16(RUPEES) / 999.0, 1.0)
    v[idx + 1] = r(DUNGEON_KEYS) / 9.0
    v[idx + 2] = r(SWORD_LEVEL) / 4.0
    v[idx + 3] = r(SHIELD_LEVEL) / 3.0
    v[idx + 4] = r(CURRENT_BOMBS) / max(r(MAX_BOMBS), 1)
    v[idx + 5] = min(r16(ORE_CHUNKS) / 99.0, 1.0)
    idx += 6

    # --- Seeds (6) ---
    for addr in [EMBER_SEEDS, SCENT_SEEDS, PEGASUS_SEEDS, GALE_SEEDS, MYSTERY_SEEDS, GASHA_SEEDS]:
        v[idx] = min(r(addr) / 99.0, 1.0)
        idx += 1

    # --- Equipment flags (5) ---
    v[idx] = min(r(SLINGSHOT_LEVEL) / 2.0, 1.0)
    v[idx + 1] = min(r(ROCS_FEATHER_LEVEL) / 2.0, 1.0)
    v[idx + 2] = min(r(MAGNETIC_GLOVES), 1.0)
    v[idx + 3] = r(A_BUTTON_ITEM) / 255.0
    v[idx + 4] = r(B_BUTTON_ITEM) / 255.0
    idx += 5

    # --- Season (2) ---
    v[idx] = r(CURRENT_SEASON) / 3.0
    v[idx + 1] = r(SEASON_SPIRITS) / 4.0
    idx += 2

    # --- Dungeon progress (4) ---
    v[idx] = 1.0 if r(DUNGEON_MAP) else 0.0
    v[idx + 1] = 1.0 if r(DUNGEON_COMPASS) else 0.0
    v[idx + 2] = r(DUNGEON_FLOOR) / 10.0
    v[idx + 3] = bin(r(ESSENCES_COLLECTED)).count("1") / 8.0
    idx += 4

    # --- Flags (4) ---
    v[idx] = 1.0 if r(DIALOG_STATE) != 0 else 0.0
    v[idx + 1] = 1.0 if r(PUZZLE_FLAGS) != 0 else 0.0
    v[idx + 2] = 1.0 if r(SCREEN_TRANSITION) != 0 else 0.0
    v[idx + 3] = r(OVERWORLD_POSITION) / 255.0
    idx += 4

    # --- Entity counts (3) ---
    v[idx] = min(r(ENEMIES_COUNT) / 10.0, 1.0)
    v[idx + 1] = min(r(NPCS_COUNT) / 10.0, 1.0)
    v[idx + 2] = min(r(ITEMS_COUNT) / 10.0, 1.0)
    idx += 3

    # --- Boss keys bitfield (8) ---
    bk = r(BOSS_KEYS)
    for bit in range(8):
        v[idx] = 1.0 if (bk >> bit) & 1 else 0.0
        idx += 1

    # --- OAM sprite proximity (up to 40 → pack first 20 y-coords) ---
    # This gives the agent a sense of "how many things are on screen"
    sprites_packed = 20
    for i in range(sprites_packed):
        y = r(OAM_BASE + i * 4)
        v[idx] = y / 160.0 if 0 < y < 160 else 0.0
        idx += 1

    # --- Navigation features (dims 65+) ---
    # Edge exit indicators (4): can the agent exit the room in each direction?
    exit_up, exit_down, exit_left, exit_right = env.check_edge_exits()
    v[idx] = exit_up
    v[idx + 1] = exit_down
    v[idx + 2] = exit_left
    v[idx + 3] = exit_right
    idx += 4

    # Ray-cast distances (4): how far can the agent walk in each direction?
    ray_up, ray_down, ray_left, ray_right = env.ray_cast_distances()
    v[idx] = ray_up
    v[idx + 1] = ray_down
    v[idx + 2] = ray_left
    v[idx + 3] = ray_right
    idx += 4

    # Exit distances (4) + direction to nearest exit (2):
    # Manhattan distance to nearest walkable edge tile per direction,
    # plus a 2-D vector pointing toward the overall nearest exit.
    ed_up, ed_down, ed_left, ed_right, dir_x, dir_y = env.exit_distances()
    v[idx] = ed_up
    v[idx + 1] = ed_down
    v[idx + 2] = ed_left
    v[idx + 3] = ed_right
    v[idx + 4] = dir_x
    v[idx + 5] = dir_y
    idx += 6

    # Active tile type (1): what Link is standing on
    v[idx] = min(r(ACTIVE_TILE_TYPE) / 24.0, 1.0)
    idx += 1

    # Screen edge distances (4): proximity to each room boundary
    px = r(PLAYER_X)
    py = r(PLAYER_Y)
    v[idx] = px / 160.0
    v[idx + 1] = max(160 - px, 0) / 160.0
    v[idx + 2] = py / 144.0
    v[idx + 3] = max(144 - py, 0) / 144.0
    idx += 4

    # --- Frontier awareness (dims 84-89) ---
    # Neighbor visited flags (4): is the adjacent room already explored?
    # 0.0 = frontier (unvisited), 1.0 = visited (backtracking)
    n_north, n_south, n_east, n_west = env.neighbor_room_visited()
    v[idx] = n_north
    v[idx + 1] = n_south
    v[idx + 2] = n_east
    v[idx + 3] = n_west
    idx += 4

    # Frontier vs any exit distance (2): how far to nearest unexplored exit?
    visited = getattr(env, "_visited_rooms_set", set())
    frontier_d = env.frontier_exit_dist(visited)
    any_d = env.nearest_exit_dist
    v[idx] = min(frontier_d / 16.0, 1.0)
    v[idx + 1] = min(any_d / 16.0, 1.0) if any_d < 999 else 1.0
    idx += 2

    # Remaining slots (idx should be ~90) stay zero-filled
    return np.clip(v, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Structured JSON encoder (for LLM planner)
# ---------------------------------------------------------------------------


def encode_json(env: ZeldaEnv) -> dict[str, Any]:
    """Produce a JSON-serializable dict for the LLM planner.

    Schema follows new/SCHEMAS.md and new/STATE_ENCODER.md.
    """
    r = env._read
    r16 = env._read16

    state: dict[str, Any] = {
        "player": {
            "x": r(PLAYER_X),
            "y": r(PLAYER_Y),
            "dir": DIRECTIONS.get(r(PLAYER_DIRECTION), "unknown"),
            "hp": r(PLAYER_HEALTH) // 4,
            "max_hp": r(PLAYER_MAX_HEALTH) // 4,
        },
        "room_id": r(PLAYER_ROOM),
        "inventory": {
            "sword": r(SWORD_LEVEL),
            "shield": r(SHIELD_LEVEL),
            "feather": r(ROCS_FEATHER_LEVEL),
            "bracelet": r(MAGNETIC_GLOVES),
            "bombs": r(CURRENT_BOMBS),
            "keys": r(DUNGEON_KEYS),
            "rupees": r16(RUPEES),
        },
        "flags": {
            "dialog": r(DIALOG_STATE) != 0,
            "puzzle": r(PUZZLE_FLAGS) != 0,
            "cutscene": r(SCREEN_TRANSITION) != 0,
        },
        "season": SEASONS.get(r(CURRENT_SEASON), "unknown"),
        "dungeon_floor": r(DUNGEON_FLOOR),
    }

    # Interactables from OAM
    interactables: list[dict[str, Any]] = []
    for i in range(40):
        base = OAM_BASE + i * 4
        y = r(base)
        x = r(base + 1)
        tile_id = r(base + 2)
        if 0 < y < 160 and 0 < x < 168:
            interactables.append(
                {"type": _classify_oam(tile_id), "x": x - 8, "y": y - 16}
            )
    state["interactables"] = interactables

    return state


def _classify_oam(tile_id: int) -> str:
    """Rough sprite classification from tile ID."""
    if 0x20 <= tile_id < 0x40:
        return "enemy"
    if 0x40 <= tile_id < 0x50:
        return "item"
    if 0x60 <= tile_id < 0x90:
        return "npc"
    return "unknown"
