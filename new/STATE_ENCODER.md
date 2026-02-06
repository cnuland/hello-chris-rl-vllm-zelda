# State Encoder

- Vector features (128): normalized coords, velocities, collisions, enemy proximity, room one-hot, inventory bits.
- JSON (planner): 
  - `player:{x,y,dir,hp,max_hp}`, `room_id`, `inventory:{sword,feather,bracelet,keys,rupees}`, 
  - `flags:{dialog,puzzle,cutscene}`, `interactables:[{type,x,y}]`.

See `SCHEMAS.md` for exact schema and validator requirements.
