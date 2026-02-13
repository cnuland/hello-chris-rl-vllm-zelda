"""Oracle of Seasons RAM addresses — oracles-disasm confirmed.

Reference: https://github.com/Drenn1/oracles-disasm
Link object struct at w1Link = $D000, SpecialObjectStruct layout.
When two addresses are listed in disasm, format is Ages/Seasons.
"""

# =============================================================================
# Player (w1Link struct at $D000)
# =============================================================================
PLAYER_X = 0xD00D          # w1Link + $0D (xh - pixel X)
PLAYER_Y = 0xD00B          # w1Link + $0B (yh - pixel Y)
PLAYER_DIRECTION = 0xD008  # w1Link + $08 (direction: 0=up, 1=right, 2=down, 3=left)
PLAYER_ROOM = 0xCC4C       # wActiveRoom (Seasons)
PLAYER_HEALTH = 0xC6A2     # wLinkHealth (Seasons) — quarter-hearts
PLAYER_MAX_HEALTH = 0xC6A3 # wLinkMaxHealth (Seasons)
HEART_PIECES = 0xC6A4      # wNumHeartPieces

# --- Player state (Link object struct fields) ---
PLAYER_STATE = 0xD004      # w1Link + $04 (state machine, see zelda_oos_ram_map.LINK_STATES)
PLAYER_SUBSTATE = 0xD005   # w1Link + $05 (sub-state within current state)
PLAYER_Z = 0xD00F          # w1Link + $0F (height, 0=ground)
PLAYER_SPEED = 0xD010      # w1Link + $10 (movement speed)
PLAYER_INVINCIBILITY = 0xD02B  # w1Link + $2B (invincibility counter)
PLAYER_KNOCKBACK = 0xD02D  # w1Link + $2D (knockback frames remaining)
PLAYER_ANIM_MODE = 0xD030  # w1Link + $30 (animation mode)

# --- Player WRAM state ---
LINK_ANGLE = 0xCC47        # wLinkAngle ($FF when not moving)
LINK_IN_AIR = 0xCC77       # wLinkInAir (upper nibble: flags, lower: 0=ground,1=jumped,2=in air)
LINK_SWIMMING = 0xCC78     # wLinkSwimmingState (bit 7=diving, bit 6=lava)
LINK_GRAB_STATE = 0xCC75   # wLinkGrabState ($00=normal, $41=grabbing, $C2=lifting, $83=holding)
LINK_DEATH_TRIGGER = 0xCC34  # wLinkDeathTrigger (nonzero = kill Link)

# =============================================================================
# Resources
# =============================================================================
RUPEES = 0xC6A5            # wNumRupees, 2 bytes little-endian
ORE_CHUNKS = 0xC6A7        # wNumOreChunks, 2 bytes (Subrosia currency)
CURRENT_BOMBS = 0xC6AA     # wNumBombs
MAX_BOMBS = 0xC6AB         # wMaxBombs
CURRENT_BOMBCHUS = 0xC6AD  # wNumBombchus
SWORD_LEVEL = 0xC6AC       # wSwordLevel (0=none, 1=wooden, 2=noble, 3=master)
SHIELD_LEVEL = 0xC6A9      # wShieldLevel (1-3)
SEED_SATCHEL_LEVEL = 0xC6AE  # wSeedSatchelLevel

# =============================================================================
# Active items
# =============================================================================
A_BUTTON_ITEM = 0xC681     # wInventoryA (item ID on A button)
B_BUTTON_ITEM = 0xC680     # wInventoryB (item ID on B button)

# =============================================================================
# Seeds
# =============================================================================
EMBER_SEEDS = 0xC6B5       # wNumEmberSeeds
SCENT_SEEDS = 0xC6B6       # wNumScentSeeds
PEGASUS_SEEDS = 0xC6B7     # wNumPegasusSeeds
GALE_SEEDS = 0xC6B8        # wNumGaleSeeds
MYSTERY_SEEDS = 0xC6B9     # wNumMysterySeeds
GASHA_SEEDS = 0xC6BA       # wNumGashaSeeds

# =============================================================================
# Equipment
# =============================================================================
BOOMERANG_LEVEL = 0xC6B1   # wBoomerangLevel (1-2)
SLINGSHOT_LEVEL = 0xC6B3   # wSlingshotLevel
ROCS_FEATHER_LEVEL = 0xC6B4  # wFeatherLevel (1=feather, 2=cape)
FLUTE_TYPE = 0xC6AF        # wFluteIcon
MAGNETIC_GLOVES = 0xC6B2   # wMagnetGlovePolarity (0=S, 1=N)

# =============================================================================
# Rings
# =============================================================================
RING_BOX_CONTENTS = 0xC6C0 # wRingBoxContents (5 ring slots)
ACTIVE_RING = 0xC6C5       # wActiveRing (current ring ID)
RING_BOX_LEVEL = 0xC6C6    # wRingBoxLevel
RINGS_OBTAINED = 0xC616    # wRingsObtained (8 bytes, 64 rings bitset)

# =============================================================================
# Progress
# =============================================================================
ESSENCES_COLLECTED = 0xC6BB  # wEssencesObtained (8-bit bitset, 1 bit per essence)
OBTAINED_TREASURES = 0xC692  # wObtainedTreasureFlags (16 bytes bitset)
TOTAL_DEATHS = 0xC61E        # wDeathCounter, 2 bytes BCD
ENEMIES_KILLED = 0xC620      # wTotalEnemiesKilled, 2 bytes
RUPEES_COLLECTED = 0xC627    # wTotalRupeesCollected, 2 bytes
SIGNS_DESTROYED = 0xC626     # wTotalSignsDestroyed
GLOBAL_FLAGS = 0xC6CA        # wGlobalFlags (16 bytes, 128 flags bitset)

# --- Maku Tree quest progression (oracles-disasm confirmed) ---
# GLOBALFLAG_GNARLED_KEY_GIVEN = flag 0x18 → byte 3 (0xC6CD), bit 0
GNARLED_KEY_GIVEN_FLAG = 0xC6CD   # wGlobalFlags + 3, bit 0 = Maku Tree gave Gnarled Key
GNARLED_KEY_GIVEN_MASK = 0x01
# TREASURE_GNARLED_KEY = 0x42 → byte 8 (0xC69A), bit 2
GNARLED_KEY_OBTAINED = 0xC69A     # wObtainedTreasureFlags + 8, bit 2 = picked up key
GNARLED_KEY_OBTAINED_MASK = 0x04
# Maku Tree stage: 0=first meeting, increases with essences collected
MAKU_TREE_STAGE = 0xCDDA          # ws_cc39 (Seasons only)

# =============================================================================
# World
# =============================================================================
ACTIVE_GROUP = 0xCC49        # wActiveGroup (0=overworld, 1=subrosia, 2=maku, 3=indoors, 4-5=dungeons)
MINIMAP_GROUP = 0xC63A       # wMinimapGroup
MINIMAP_ROOM = 0xC63B        # wMinimapRoom
OVERWORLD_POSITION = 0xC63C  # wMinimapDungeonMapPosition
DUNGEON_POSITION = 0xCC56    # wDungeonMapPosition
DUNGEON_FLOOR = 0xCC57       # wDungeonFloor (Seasons)
DUNGEON_INDEX = 0xCC55       # wDungeonIndex ($FF = overworld)
DUNGEON_ROOM_PROPERTIES = 0xCC58  # wDungeonRoomProperties (exits, key, chest, boss, dark)
MAPLE_COUNTER = 0xC63E       # wMapleKillCounter (Maple appears at 30)
ENEMIES_ON_SCREEN = 0xCC30   # wNumEnemies

# =============================================================================
# Dialog / UI
# =============================================================================
DIALOG_STATE = 0xCBA0        # wTextIsActive (0 = no text, $80 = finished+nonexitable)
MENU_STATE = 0xCBCB          # wOpenedMenuType (1=inventory, 2=map, 3=save)
GAME_STATE = 0xC2EE          # wGameState (0=loading, 2=gameplay)
CUTSCENE_INDEX = 0xC2EF      # wCutsceneIndex
SCREEN_TRANSITION = 0xCD00   # wScrollMode
LOADING_SCREEN = 0xC2F2      # Loading state byte
DISABLED_OBJECTS = 0xCCA4    # wDisabledObjects (bit 0=Link, 2=enemies, etc.)

# =============================================================================
# Season
# =============================================================================
CURRENT_SEASON = 0xCC4E      # wRoomStateModifier (0=spring, 1=summer, 2=autumn, 3=winter)
OBTAINED_SEASONS = 0xC6B0    # wObtainedSeasons (bitset: bit 0=spring, 1=summer, 2=autumn, 3=winter)
SEASON_SPIRITS = OBTAINED_SEASONS  # Legacy alias

# =============================================================================
# Dungeon
# =============================================================================
DUNGEON_KEYS = 0xC66E        # wDungeonSmallKeys (12-byte array, 1 per dungeon)
BOSS_KEYS = 0xC67A           # wDungeonBossKeys (2-byte bitset, 1 bit per dungeon)
DUNGEON_MAP = 0xC67E         # wDungeonMaps (2-byte bitset)
DUNGEON_COMPASS = 0xC67C     # wDungeonCompasses (2-byte bitset)

# =============================================================================
# Entity counts / Room puzzle state
# =============================================================================
ENEMIES_COUNT = 0xCC30       # wNumEnemies (triggers events when reaches 0)
TOGGLE_BLOCKS_STATE = 0xCC31 # wToggleBlocksState (orb-activated block state)
SWITCH_STATE = 0xCC32        # wSwitchState (bitset of switches pressed in room)

# =============================================================================
# Tile interaction
# =============================================================================
ACTIVE_TILE_TYPE = 0xCCB6   # wActiveTileType (what Link stands on)
ACTIVE_TILE_INDEX = 0xCCB4  # wActiveTileIndex
ACTIVE_TILE_POS = 0xCCB3    # wActiveTilePos

# =============================================================================
# Room layout
# =============================================================================
ROOM_LAYOUT = 0xCF00        # wRoomLayout (192 bytes, room tile data)
ROOM_COLLISIONS = 0xCE00    # wRoomCollisions (192 bytes)

# =============================================================================
# OAM (Sprite) base
# =============================================================================
OAM_BASE = 0xFE00           # 40 sprites × 4 bytes each

# =============================================================================
# Puzzle flags
# =============================================================================
PUZZLE_FLAGS = 0xCC58        # wDungeonRoomProperties (exits, key, chest, boss, dark)

# =============================================================================
# Input
# =============================================================================
KEYS_PRESSED = 0xC481        # wKeysPressed (currently held buttons)
KEYS_JUST_PRESSED = 0xC482   # wKeysJustPressed (buttons pressed this frame)

# =============================================================================
# Enemy object slots
# =============================================================================
ENEMY_SLOT_BASE = 0xD080     # First enemy slot ($D[N]80 for slot N, 16 slots)
ENEMY_STRUCT_SIZE = 0x100    # Distance between enemy slots

# =============================================================================
# Look-up tables
# =============================================================================
DIRECTIONS = {0: "up", 1: "right", 2: "down", 3: "left"}
SEASONS = {0: "spring", 1: "summer", 2: "autumn", 3: "winter"}
