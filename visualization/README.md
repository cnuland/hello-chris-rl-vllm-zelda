# Oracle of Seasons — RL Map Visualization

Real-time and offline map visualization for a reinforcement learning agent
playing *The Legend of Zelda: Oracle of Seasons* on Game Boy Color.

Watch the RL agent explore the overworld of Holodrum in real-time on an
interactive web map, or generate shareable videos of exploration runs.

## Credits & Inspiration

This visualization system is built on the shoulders of two excellent projects:

### [PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments)

By **Peter Whidden** — The original RL-plays-Pokemon project that pioneered the
approach of streaming agent coordinates over WebSocket and rendering animated
sprite overlays on a full overworld map image. Key ideas borrowed:

- **`stream_agent_wrapper.py`**: Lightweight `gym.Wrapper` that reads position
  data from emulator RAM and streams coordinate triples `[x, y, map_id]` over
  WebSocket — zero impact on training performance.
- **`BetterMapVis_script_version.py`**: Offline video renderer that loads
  coordinate logs and renders frame-by-frame animations with character sprites
  walking across the full Kanto overworld map.
- **`BetterMapVis_script_version_FLOW.py`**: Flow field visualization showing
  aggregate movement directions with color-coded arrows.

### [LinkMapViz](https://github.com/Xe-Xo/LinkMapViz)

By **Xe-Xo** — Real-time browser-based map visualization for *The Legend of
Zelda: Link's Awakening DX* RL agents. This is the most direct architectural
inspiration since Link's Awakening uses the same screen-based room system as
Oracle of Seasons (16×16 room grid, 10×8 tiles per room). Key ideas borrowed:

- **PixiJS web frontend**: Full overworld map as a background sprite with
  colored position rectangles that fade from green (fresh) to red (stale)
  over a 30-second TTL. Pan/zoom interaction for exploring the map.
- **WebSocket pub/sub relay**: Express + express-ws server that receives from
  the Python trainer (`/broadcast`) and forwards to browser clients (`/receive`)
  with a 16-message replay buffer for late joiners.
- **Coordinate system**: `world_x = room_col * 10 + tile_x`,
  `world_y = room_row * 8 + tile_y` — positions map to pixels at `(x*16, y*16)`
  on the overworld image.
- **Notable events**: Icon overlays for gameplay events (gate slashes, item
  pickups, new room discoveries).

The companion training project **[LADXExperiments](https://github.com/Xe-Xo/LADXExperiments)**
provided the `StreamWrapper` pattern and `get_world_pos()` coordinate
translation that our `stream_wrapper.py` is modeled after.

### [oracles-disasm](https://github.com/Stewmath/oracles-disasm)

By **Stewmath** — The Oracle of Seasons/Ages disassembly project that documents
all RAM addresses, warp system mechanics, room loading routines, and memory
layouts used throughout this visualization code. Specific contributions:

- Warp destination variables (`wWarpDestGroup`, `wWarpDestRoom`, etc.) and the
  bit 7 direct-warp mechanism used in `generate_overworld.py`.
- Room state modifier (season control) for generating seasonal map variants.
- Transition system constants (`TRANSITION_DEST_*`, `wWarpTransition2`) for
  fast room loading during map generation.

## Architecture

```
┌─────────────────────┐    WebSocket     ┌──────────────────┐    WebSocket     ┌─────────────────┐
│  Python RL Trainer  │  ──────────────> │  Relay Server    │  ──────────────> │  Web Visualizer  │
│  (StreamWrapper)    │   /broadcast     │  (Node.js)       │   /receive      │  (PixiJS)        │
│                     │                  │                  │                  │                  │
│  Reads RAM:         │                  │  Buffers last    │                  │  Renders map +   │
│  - room_id          │   {pos_data:     │  16 messages     │                  │  heatmap overlay │
│  - player_x/y       │    [{x,y,z}]}   │  for late join   │                  │  Pan/zoom/touch  │
│  - active_group     │                  │                  │                  │                  │
└─────────────────────┘                  └──────────────────┘                  └─────────────────┘

                                  ┌──────────────────────┐
                                  │  Offline Renderer    │
                                  │  (Python)            │
                                  │                      │
                                  │  Reads CSV/JSONL     │
                                  │  logs and renders    │
                                  │  videos or images    │
                                  └──────────────────────┘
```

## Components

| Component | Path | Language | Purpose |
|-----------|------|----------|---------|
| Map Generator | `generate_overworld.py` | Python | Generate overworld/Subrosia map images from ROM |
| Stream Wrapper | `stream_wrapper.py` | Python | `gym.Wrapper` that streams position data |
| Offline Renderer | `offline_renderer.py` | Python | Generate videos/images from coordinate logs |
| WS Relay Server | `ws-server/` | Node.js | WebSocket pub/sub relay |
| Web Visualizer | `web/` | JS/PixiJS | Real-time browser map visualization |

## Quick Start

### 1. Generate the Overworld Map

```bash
# Default season from save state
python visualization/generate_overworld.py

# Specific season
python visualization/generate_overworld.py --season 0  # Spring
python visualization/generate_overworld.py --season 3  # Winter

# All seasons
python visualization/generate_overworld.py --all-seasons

# Subrosia
python visualization/generate_overworld.py --subrosia

# Verbose output
python visualization/generate_overworld.py -v
```

This generates `visualization/assets/overworld.png` (2560×2048 pixels) by
warping to all 256 rooms in the 16×16 grid and stitching screenshots.

### 2. Real-Time Streaming Mode

**Start the relay server:**

```bash
cd visualization/ws-server
npm install
npm start
# → Relay running on ws://localhost:3344
```

Or with Docker:

```bash
cd visualization/ws-server
docker build -t zelda-ws-relay .
docker run -p 3344:3344 zelda-ws-relay
```

**Start the web visualizer:**

```bash
cd visualization/web
npm install
# Copy overworld.png to web assets
cp ../assets/overworld.png assets/
npm run dev
# → Opens http://localhost:8080
```

**Wrap your environment with StreamWrapper:**

```python
from visualization.stream_wrapper import StreamWrapper

env = YourZeldaEnv(...)
env = StreamWrapper(
    env,
    ws_address="ws://localhost:3344/broadcast",
    stream_metadata={
        "user": "my-agent",
        "env_id": 0,
        "color": "#44aa77",
    },
)

# Training loop — StreamWrapper is transparent
obs, info = env.reset()
for _ in range(100000):
    action = agent.act(obs)
    obs, reward, done, truncated, info = env.step(action)
    if done:
        obs, info = env.reset()
```

### 3. Offline Video Rendering

**Record coordinates during training:**

```python
env = StreamWrapper(
    env,
    csv_log_path="run_coords.csv",  # Log to CSV
)
```

**Render a video:**

```bash
# Heatmap video (green → red trail)
python visualization/offline_renderer.py run_coords.csv --mode video

# Flow field (directional arrows)
python visualization/offline_renderer.py run_coords.csv --mode flow

# Coverage image (all visited tiles)
python visualization/offline_renderer.py run_coords.csv --mode coverage

# High resolution with custom settings
python visualization/offline_renderer.py run_coords.csv \
    --mode video --scale 2.0 --fps 60 --trail 500
```

## Coordinate System

The coordinate system maps Game Boy RAM values to pixel positions on the
overworld image:

```
RAM Data                         World Coordinates
─────────────────────────────────────────────────────
wActiveRoom (0xCC4C) = room_id → room_col = room_id % 16
                                  room_row = room_id // 16
w1Link.xh  (0xD00D) = pixel_x → tile_x = pixel_x // 16
w1Link.yh  (0xD00B) = pixel_y → tile_y = pixel_y // 16

world_x = room_col * 10 + tile_x    (range: 0-159)
world_y = room_row * 8  + tile_y    (range: 0-127)

Pixel position on overworld.png:
  px = world_x * 16    (range: 0-2543 on 2560px image)
  py = world_y * 16    (range: 0-2031 on 2048px image)
```

## Map Image Specs

| Map | Dimensions | Rooms | Tiles/Room | Tile Size |
|-----|-----------|-------|------------|-----------|
| Overworld (Holodrum) | 2560 × 2048 px | 16 × 16 = 256 | 10 × 8 = 80 | 16 × 16 px |
| Subrosia | 2560 × 2048 px | 16 × 16 = 256 | 10 × 8 = 80 | 16 × 16 px |

## Dependencies

### Python

```
pyboy>=2.6.0        # Game Boy emulator
numpy>=1.24.0       # Array operations
Pillow>=10.0.0      # Image processing
gymnasium>=0.29.0   # RL environment interface
websockets>=12.0    # WebSocket client
opencv-python>=4.8  # Video encoding (offline renderer)
```

### Node.js (WebSocket relay)

```
express ^4.21.0
express-ws ^5.0.2
```

### Browser (web visualizer)

```
pixi.js ^8.1.2
webpack ^5.94.0
```

## License

MIT — See individual source files for specific credits.
