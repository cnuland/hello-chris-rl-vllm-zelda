"""Prometheus/OpenTelemetry metrics helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ExplorationMetrics:
    """Track exploration statistics for a single episode."""

    unique_rooms: set[int] = field(default_factory=set)
    unique_tiles: dict[int, set[tuple[int, int]]] = field(default_factory=dict)
    doorway_pingpong: int = 0
    _last_room: int | None = None
    _room_history: list[int] = field(default_factory=list)

    def record_position(self, room_id: int, tile_x: int, tile_y: int) -> bool:
        """Record a position visit, return True if it's a new tile."""
        self.unique_rooms.add(room_id)
        if room_id not in self.unique_tiles:
            self.unique_tiles[room_id] = set()
        tile = (tile_x, tile_y)
        is_new = tile not in self.unique_tiles[room_id]
        self.unique_tiles[room_id].add(tile)

        # Detect doorway ping-pong
        if self._last_room is None:
            self._room_history.append(room_id)
        elif room_id != self._last_room:
            self._room_history.append(room_id)
            if len(self._room_history) >= 4:
                recent = self._room_history[-4:]
                if recent[0] == recent[2] and recent[1] == recent[3]:
                    self.doorway_pingpong += 1
        self._last_room = room_id
        return is_new

    @property
    def total_unique_tiles(self) -> int:
        return sum(len(tiles) for tiles in self.unique_tiles.values())

    @property
    def num_unique_rooms(self) -> int:
        return len(self.unique_rooms)


@dataclass
class LLMCallMetrics:
    """Track LLM call latency and usage."""

    call_count: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    cache_hits: int = 0

    def record_call(self, latency_ms: float, tokens: int = 0, cache_hit: bool = False) -> None:
        self.call_count += 1
        self.total_latency_ms += latency_ms
        self.total_tokens += tokens
        if cache_hit:
            self.cache_hits += 1

    @property
    def avg_latency_ms(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.total_latency_ms / self.call_count

    @property
    def cache_hit_rate(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.cache_hits / self.call_count


class Timer:
    """Context manager for timing operations."""

    def __init__(self):
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
