"""Episode exporter: video PNGs + RAM JSONL to MinIO.

After each training episode, exports segment data:
  - Frame PNGs (downsampled)
  - RAM state JSONL (one line per frame)
  - manifest.json with metadata

Stored in: s3://zelda-episodes/{episode_id}/
"""

from __future__ import annotations

import io
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _sanitize(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    return obj


@dataclass
class FrameRecord:
    """A single frame's data for export."""

    step: int
    state: dict[str, Any]
    action: int
    reward: float
    frame_png: bytes | None = None


@dataclass
class EpisodeSegment:
    """A segment of an episode for evaluation."""

    segment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    episode_id: str = ""
    start_step: int = 0
    end_step: int = 0
    frames: list[FrameRecord] = field(default_factory=list)
    total_reward: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class EpisodeExporter:
    """Exports episode segments to MinIO for evaluator consumption."""

    def __init__(
        self,
        s3_client: Any = None,
        bucket: str = "zelda-episodes",
        frames_per_segment: int = 300,
        png_interval: int = 10,  # save PNG every N frames
        epoch: int = 0,
    ):
        self._s3 = s3_client
        self._bucket = bucket
        self._frames_per_segment = frames_per_segment
        self._png_interval = png_interval
        self._epoch = epoch

        # Buffer
        self._current_frames: list[FrameRecord] = []
        self._episode_id = ""
        self._episode_reward = 0.0

    def begin_episode(self, episode_id: str | None = None) -> str:
        """Start tracking a new episode."""
        self._episode_id = episode_id or str(uuid.uuid4())[:12]
        self._current_frames.clear()
        self._episode_reward = 0.0
        return self._episode_id

    def record_frame(
        self,
        step: int,
        state: dict[str, Any],
        action: int,
        reward: float,
        screen_array: np.ndarray | None = None,
    ) -> list[EpisodeSegment]:
        """Record a frame. Returns exported segments when buffer is full."""
        frame_png = None
        if screen_array is not None and step % self._png_interval == 0:
            frame_png = self._encode_png(screen_array)

        self._current_frames.append(
            FrameRecord(step=step, state=state, action=action, reward=reward, frame_png=frame_png)
        )
        self._episode_reward += reward

        exported = []
        if len(self._current_frames) >= self._frames_per_segment:
            seg = self._flush_segment()
            if seg:
                exported.append(seg)
        return exported

    def end_episode(self) -> list[EpisodeSegment]:
        """Flush remaining frames as a final segment."""
        segments = []
        if self._current_frames:
            seg = self._flush_segment()
            if seg:
                segments.append(seg)
        return segments

    def _flush_segment(self) -> EpisodeSegment | None:
        if not self._current_frames:
            return None

        seg = EpisodeSegment(
            episode_id=self._episode_id,
            start_step=self._current_frames[0].step,
            end_step=self._current_frames[-1].step,
            frames=list(self._current_frames),
            total_reward=sum(f.reward for f in self._current_frames),
            metadata={
                "episode_id": self._episode_id,
                "total_episode_reward": self._episode_reward,
            },
        )
        self._current_frames.clear()

        if self._s3 is not None:
            self._upload_segment(seg)

        return seg

    def _upload_segment(self, seg: EpisodeSegment) -> None:
        """Upload segment to MinIO."""
        prefix = f"epoch_{self._epoch}/{seg.episode_id}/{seg.segment_id}"

        # Upload manifest
        manifest = {
            "segment_id": seg.segment_id,
            "episode_id": seg.episode_id,
            "start_step": int(seg.start_step),
            "end_step": int(seg.end_step),
            "num_frames": len(seg.frames),
            "total_reward": float(seg.total_reward),
            "metadata": _sanitize(seg.metadata),
        }
        self._s3.upload_json(self._bucket, f"{prefix}/manifest.json", manifest)

        # Upload RAM state JSONL
        lines = []
        for f in seg.frames:
            lines.append(
                json.dumps(
                    {"step": int(f.step), "action": int(f.action), "reward": float(f.reward), "state": _sanitize(f.state)}
                )
            )
        self._s3.upload_bytes(
            self._bucket, f"{prefix}/states.jsonl", "\n".join(lines).encode()
        )

        # Upload PNGs
        for f in seg.frames:
            if f.frame_png is not None:
                self._s3.upload_bytes(
                    self._bucket,
                    f"{prefix}/frames/frame_{f.step:06d}.png",
                    f.frame_png,
                )

        logger.info("Exported segment %s (%d frames)", seg.segment_id, len(seg.frames))

    @staticmethod
    def _encode_png(screen_array: np.ndarray) -> bytes:
        """Encode numpy screen array to PNG bytes."""
        try:
            from PIL import Image

            img = Image.fromarray(screen_array.astype(np.uint8))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except ImportError:
            return b""
