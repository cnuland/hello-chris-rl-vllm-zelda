"""Go-Explore-lite archive: keyed by (room_id, tile_bin).

Periodically restart rollouts from frontier states (highest novelty).
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ArchiveEntry:
    """A single archive cell."""

    room_id: int
    tile_bin: tuple[int, int]
    save_state: bytes  # PyBoy save-state blob
    visit_count: int = 0
    best_reward: float = float("-inf")
    total_coverage: int = 0  # unique tiles reached from this cell

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.room_id, *self.tile_bin)

    @property
    def novelty_score(self) -> float:
        """Lower visit count + more coverage = higher novelty."""
        return self.total_coverage / max(self.visit_count, 1)


class Archive:
    """Go-Explore-lite archive mapping (room_id, tile_bin) → state.

    On each episode start (or periodically), the agent can restart from
    a frontier cell — the one with the highest novelty score.
    """

    def __init__(self, max_size: int = 10_000):
        self._cells: dict[tuple[int, int, int], ArchiveEntry] = {}
        self.max_size = max_size

    def update(
        self,
        room_id: int,
        pixel_x: int,
        pixel_y: int,
        save_state: bytes,
        episode_reward: float,
        coverage: int,
    ) -> bool:
        """Update archive with a new observation.

        Returns True if this is a new cell or improved an existing one.
        """
        bin_x = pixel_x // 20
        bin_y = pixel_y // 18
        key = (room_id, bin_x, bin_y)

        if key in self._cells:
            entry = self._cells[key]
            entry.visit_count += 1
            improved = False
            if episode_reward > entry.best_reward:
                entry.best_reward = episode_reward
                entry.save_state = save_state
                improved = True
            if coverage > entry.total_coverage:
                entry.total_coverage = coverage
                improved = True
            return improved

        # New cell
        if len(self._cells) >= self.max_size:
            self._evict_least_novel()

        self._cells[key] = ArchiveEntry(
            room_id=room_id,
            tile_bin=(bin_x, bin_y),
            save_state=save_state,
            visit_count=1,
            best_reward=episode_reward,
            total_coverage=coverage,
        )
        return True

    def sample_frontier(self, k: int = 1) -> list[ArchiveEntry]:
        """Return the k cells with highest novelty score (restart candidates)."""
        if not self._cells:
            return []
        entries = list(self._cells.values())
        return heapq.nlargest(k, entries, key=lambda e: e.novelty_score)

    def get_restart_state(self) -> bytes | None:
        """Get the save state of the top frontier cell."""
        frontier = self.sample_frontier(1)
        if frontier:
            entry = frontier[0]
            entry.visit_count += 1  # penalise re-use
            logger.info(
                "Archive restart: room=%d bin=(%d,%d) novelty=%.2f",
                entry.room_id,
                *entry.tile_bin,
                entry.novelty_score,
            )
            return entry.save_state
        return None

    def _evict_least_novel(self) -> None:
        """Remove the cell with lowest novelty to make room."""
        if not self._cells:
            return
        worst_key = min(self._cells, key=lambda k: self._cells[k].novelty_score)
        del self._cells[worst_key]

    def __len__(self) -> int:
        return len(self._cells)

    @property
    def unique_rooms(self) -> int:
        return len({k[0] for k in self._cells})

    def reset(self) -> None:
        self._cells.clear()
