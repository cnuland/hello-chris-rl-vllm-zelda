# Exploration

- **Coverage**: per-room 8×8 tile bins; reward first visits; log revisit ratio.
- **Archive**: hash `(room_id, tile_bin)`; periodic frontier restarts.
- **Anti-loop**: n-gram loop penalty; sticky actions; longer holds for navigation.
- **Adaptive LLM budget**: base 1%; up to 20% when stuck/entropy high; 0.5% when progressing.
