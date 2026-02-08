"""Oracle of Seasons RAM addresses — oracles-disasm confirmed.

Reference: https://github.com/Drenn1/oracles-disasm
Link object struct at w1Link = $D000, SpecialObjectStruct layout.
When two addresses are listed in disasm, format is Ages/Seasons.
"""

# --- Player (w1Link struct at $D000) ---
PLAYER_X = 0xD00D          # w1Link + $0D (xh - pixel X)
PLAYER_Y = 0xD00B          # w1Link + $0B (yh - pixel Y)
PLAYER_DIRECTION = 0xD008  # w1Link + $08 (direction: 0=up, 1=right, 2=down, 3=left)
PLAYER_ROOM = 0xCC4C       # wActiveRoom (Seasons)
PLAYER_HEALTH = 0xC6A2     # wLinkHealth (Seasons) — quarter-hearts
PLAYER_MAX_HEALTH = 0xC6A3 # wLinkMaxHealth (Seasons)
HEART_PIECES = 0xC6A4      # wNumHeartPieces

# --- Resources ---
RUPEES = 0xC6A5  # 2 bytes, little-endian
ORE_CHUNKS = 0xC6AA  # 2 bytes
CURRENT_BOMBS = 0xC674
MAX_BOMBS = 0xC675
CURRENT_BOMBCHUS = 0xC676
SWORD_LEVEL = 0xC668
SHIELD_LEVEL = 0xC669
SEED_SATCHEL_LEVEL = 0xC66E

# --- Active items ---
A_BUTTON_ITEM = 0xC620
B_BUTTON_ITEM = 0xC621

# --- Seeds ---
EMBER_SEEDS = 0xC66F
SCENT_SEEDS = 0xC670
PEGASUS_SEEDS = 0xC671
GALE_SEEDS = 0xC672
MYSTERY_SEEDS = 0xC673
GASHA_SEEDS = 0xC674

# --- Equipment ---
BOOMERANG_LEVEL = 0xC66A
SLINGSHOT_LEVEL = 0xC66B
ROCS_FEATHER_LEVEL = 0xC66C
FLUTE_TYPE = 0xC66D
MAGNETIC_GLOVES = 0xC66E

# --- Rings ---
VASU_RING_FLAGS = 0xC6A0
RING_BOX_LEVEL = 0xC6A1

# --- Progress ---
ESSENCES_COLLECTED = 0xC692
TOTAL_DEATHS = 0xC61E  # 2 bytes
ENEMIES_KILLED = 0xC6B0  # 2 bytes
RUPEES_COLLECTED = 0xC6B2  # 2 bytes

# --- World ---
CURRENT_LEVEL_BANK = 0xC63A
OVERWORLD_POSITION = 0xC63C
DUNGEON_POSITION = 0xCC56   # wDungeonMapPosition
DUNGEON_FLOOR = 0xCC57      # wDungeonFloor (Seasons)
MAPLE_COUNTER = 0xC6B4
ENEMIES_ON_SCREEN = 0xCC30

# --- Dialog / UI ---
DIALOG_STATE = 0xCBA0    # wTextIsActive (0 = no text)
MENU_STATE = 0xC2F0
SCREEN_TRANSITION = 0xCD00  # wScrollMode
LOADING_SCREEN = 0xC2F2

# --- Season ---
CURRENT_SEASON = 0xC680  # 0=spring, 1=summer, 2=autumn, 3=winter
SEASON_SPIRITS = 0xC681

# --- Dungeon ---
DUNGEON_KEYS = 0xC694
BOSS_KEYS = 0xC696
DUNGEON_MAP = 0xC697
DUNGEON_COMPASS = 0xC698

# --- Entity counts ---
ENEMIES_COUNT = 0xCC30
NPCS_COUNT = 0xCC31
ITEMS_COUNT = 0xCC32

# --- OAM (Sprite) base ---
OAM_BASE = 0xFE00  # 40 sprites × 4 bytes each

# --- Puzzle flags ---
PUZZLE_FLAGS = 0xC6C0  # Bitfield for puzzle room completion

# --- Look-up tables ---
DIRECTIONS = {0: "up", 1: "right", 2: "down", 3: "left"}
SEASONS = {0: "spring", 1: "summer", 2: "autumn", 3: "winter"}
