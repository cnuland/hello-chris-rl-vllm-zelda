"""Training monitor — fetches progress from MinIO and Ray dashboard.

Prints a summary of game progress, judge model scores, and training metrics.
Run locally while training is active on the cluster.

Usage:
  python scripts/monitor_training.py              # one-shot report
  python scripts/monitor_training.py --loop 1800   # repeat every 30 min
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s"
)
logger = logging.getLogger(__name__)


def get_s3_client():
    """Create S3Client using env vars or defaults."""
    from agent.utils.s3 import S3Client
    from agent.utils.config import S3Config
    return S3Client(S3Config())


def fetch_epoch_metadata(s3, max_epoch: int = 100) -> list[dict]:
    """Download all epoch metadata files from MinIO."""
    results = []
    for epoch in range(max_epoch):
        try:
            data = s3.download_bytes(
                "zelda-models",
                f"checkpoints/epoch_{epoch}/metadata.json",
            )
            results.append(json.loads(data))
        except Exception:
            break
    return results


def fetch_eval_summaries(s3, max_epoch: int = 100) -> list[dict]:
    """Download all evaluation summaries from MinIO."""
    results = []
    for epoch in range(max_epoch):
        try:
            data = s3.download_bytes(
                "zelda-models",
                f"evaluations/epoch_{epoch}/summary.json",
            )
            results.append(json.loads(data))
        except Exception:
            continue
    return results


def count_segments_per_epoch(s3, max_epoch: int = 100) -> dict[int, int]:
    """Count exported segments per epoch."""
    counts = {}
    for epoch in range(max_epoch):
        try:
            prefix = f"epoch_{epoch}/"
            manifests = s3.list_manifests("zelda-episodes", prefix=prefix, max_count=1000)
            if manifests:
                counts[epoch] = len(manifests)
        except Exception:
            continue
    return counts


def fetch_latest_segment_stats(s3, epoch: int) -> dict:
    """Read a sample of segment manifests to extract game state info."""
    stats = {
        "rooms_seen": set(),
        "max_reward": float("-inf"),
        "total_segments": 0,
        "sample_states": [],
    }
    try:
        prefix = f"epoch_{epoch}/"
        manifests = s3.list_manifests("zelda-episodes", prefix=prefix, max_count=50)
        stats["total_segments"] = len(manifests)

        for mkey in manifests[:10]:
            try:
                data = s3.download_bytes("zelda-episodes", mkey)
                manifest = json.loads(data)
                stats["max_reward"] = max(stats["max_reward"], manifest.get("total_reward", 0))

                # Try to read states.jsonl for game state details
                states_key = mkey.replace("manifest.json", "states.jsonl")
                try:
                    states_data = s3.download_bytes("zelda-episodes", states_key)
                    for line in states_data.decode().strip().split("\n")[:5]:
                        state = json.loads(line)
                        s = state.get("state", {})
                        room = s.get("room_id")
                        if room is not None:
                            stats["rooms_seen"].add(room)
                        if not stats["sample_states"]:
                            stats["sample_states"].append(s)
                except Exception:
                    pass
            except Exception:
                continue
    except Exception:
        pass

    stats["rooms_seen"] = list(stats["rooms_seen"])
    return stats


def print_report(metadata: list[dict], evals: list[dict], segment_counts: dict[int, int],
                 latest_stats: dict, latest_epoch: int):
    """Print formatted training report."""
    print("\n" + "=" * 70)
    print("  ZELDA RL TRAINING MONITOR")
    print("=" * 70)

    if not metadata:
        print("\nNo training data found yet. Training may not have started.")
        return

    latest = metadata[-1]
    print(f"\n--- Training Progress (Epoch {latest.get('epoch', '?')}) ---")
    print(f"  Total timesteps:    {latest.get('timesteps', 0):,}")
    print(f"  Episodes completed: {latest.get('episodes_completed', latest.get('episodes', 0)):,}")
    print(f"  Mean reward:        {latest.get('reward_mean', 0):.2f}")
    print(f"  Max reward:         {latest.get('reward_max', 0):.2f}")
    print(f"  Min reward:         {latest.get('reward_min', 'N/A')}")
    print(f"  Entropy coeff:      {latest.get('entropy_coeff', 'N/A')}")

    # Game milestones from metadata
    ms = latest.get("milestones", {})
    if ms:
        n_eps = max(latest.get("episodes_completed", 1), 1)
        print(f"\n--- Game Milestones (Epoch {latest.get('epoch', '?')}) ---")
        print(f"  Max rooms explored:   {ms.get('max_rooms', 0)}")
        print(f"  Max tiles explored:   {ms.get('max_tiles', 0)}")
        print(f"  Got sword:            {ms.get('total_got_sword', 0)}/{n_eps} episodes")
        print(f"  Visited Maku Tree:    {ms.get('total_visited_maku_tree', 0)}/{n_eps} episodes")
        print(f"  Maku Tree dialog:     {ms.get('total_maku_dialog', 0)}/{n_eps} episodes")
        print(f"  Got Gnarled Key:      {ms.get('total_gnarled_key', 0)}/{n_eps} episodes")
        print(f"  Entered dungeon:      {ms.get('total_entered_dungeon', 0)}/{n_eps} episodes")
        print(f"  Max essences:         {ms.get('max_essences', 0)}")
        print(f"  Max dungeon keys:     {ms.get('max_dungeon_keys', 0)}")

    # Milestone trend across epochs
    milestone_epochs = [m for m in metadata if m.get("milestones")]
    if len(milestone_epochs) >= 2:
        print(f"\n--- Exploration Trend (last 5 epochs) ---")
        for m in milestone_epochs[-5:]:
            ep = m.get("epoch", "?")
            ms_e = m.get("milestones", {})
            rooms = ms_e.get("max_rooms", 0)
            tiles = ms_e.get("max_tiles", 0)
            sword = ms_e.get("total_got_sword", 0)
            maku = ms_e.get("total_visited_maku_tree", 0)
            dung = ms_e.get("total_entered_dungeon", 0)
            room_bar = "#" * min(rooms, 40)
            flags = []
            if sword: flags.append("SWORD")
            if maku: flags.append("MAKU")
            if dung: flags.append("DUNGEON")
            flag_str = f"  [{', '.join(flags)}]" if flags else ""
            print(f"  Epoch {ep:3d}: rooms={rooms:3d} tiles={tiles:4d} {room_bar}{flag_str}")

    # Reward trend
    if len(metadata) >= 2:
        print("\n--- Reward Trend (last 5 epochs) ---")
        for m in metadata[-5:]:
            ep = m.get("epoch", "?")
            mean = m.get("reward_mean", 0)
            mx = m.get("reward_max", 0)
            ent = m.get("entropy_coeff", "?")
            bar = "+" * max(0, int(mean / 50)) if mean > 0 else "-" * max(0, int(abs(mean) / 50))
            print(f"  Epoch {ep:3d}: mean={mean:8.1f}  max={mx:8.1f}  ent={ent}  {bar}")

    # Game progress from latest segments
    print(f"\n--- Game Progress (Epoch {latest_epoch}) ---")
    rooms = latest_stats.get("rooms_seen", [])
    print(f"  Rooms discovered (sample): {len(rooms)}")
    if rooms:
        print(f"  Room IDs: {sorted(rooms)[:20]}")

    seg_reward = latest_stats.get("max_reward", 0)
    if seg_reward > float("-inf"):
        print(f"  Best segment reward:       {seg_reward:.1f}")

    segs = latest_stats.get("total_segments", 0)
    print(f"  Segments exported:         {segs}")

    # Check for game state details
    for s in latest_stats.get("sample_states", []):
        health = s.get("health", "?")
        max_hp = s.get("max_health", "?")
        group = s.get("active_group", "?")
        dungeon = s.get("dungeon_index", 255)
        tiles = s.get("unique_tiles", "?")
        urooms = s.get("unique_rooms", "?")

        group_name = {0: "Overworld", 1: "Subrosia", 2: "Maku Tree", 3: "Indoors",
                      4: "Dungeon", 5: "Dungeon"}.get(group, f"Group {group}")

        print(f"\n  Latest state snapshot:")
        print(f"    Location:    {group_name} (group={group})")
        if dungeon != 255 and dungeon != "?":
            print(f"    Dungeon:     #{dungeon}")
        print(f"    Health:      {health}/{max_hp}")
        print(f"    Unique rooms: {urooms}")
        print(f"    Unique tiles: {tiles}")

    # Dungeon / Maku Tree detection from group changes
    dungeon_epochs = []
    maku_epochs = []
    for m in metadata:
        ep = m.get("epoch", 0)
        # We can't directly read group from metadata, but eval summaries may help

    # Segment counts per epoch
    if segment_counts:
        print(f"\n--- Segments Per Epoch ---")
        for ep in sorted(segment_counts.keys())[-5:]:
            print(f"  Epoch {ep:3d}: {segment_counts[ep]} segments")

    # Judge model evaluation
    if evals:
        print(f"\n--- LLM Judge Evaluation ---")
        for ev in evals[-5:]:
            ep = ev.get("epoch", "?")
            segs_eval = ev.get("segments_evaluated", 0)
            mean_score = ev.get("mean_weighted_score", 0)
            dist = ev.get("score_distribution", {})
            print(f"  Epoch {ep:3d}: score={mean_score:.3f} ({segs_eval} segments)")
            for dim, val in dist.items():
                bar = "#" * int(val * 20)
                print(f"    {dim:12s}: {val:.3f} {bar}")

        # Score trend
        if len(evals) >= 2:
            scores = [e.get("mean_weighted_score", 0) for e in evals]
            trend = scores[-1] - scores[0]
            direction = "improving" if trend > 0 else "declining" if trend < 0 else "stable"
            print(f"\n  Judge score trend: {direction} ({scores[0]:.3f} -> {scores[-1]:.3f})")
    else:
        print("\n--- LLM Judge Evaluation ---")
        print("  No evaluations completed yet.")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Monitor Zelda RL training")
    parser.add_argument("--loop", type=int, default=0,
                        help="Repeat every N seconds (0 = one-shot)")
    args = parser.parse_args()

    while True:
        try:
            s3 = get_s3_client()
            metadata = fetch_epoch_metadata(s3)
            evals = fetch_eval_summaries(s3)
            segment_counts = count_segments_per_epoch(s3)

            latest_epoch = metadata[-1].get("epoch", 0) if metadata else 0
            latest_stats = fetch_latest_segment_stats(s3, latest_epoch)

            print_report(metadata, evals, segment_counts, latest_stats, latest_epoch)
        except Exception as e:
            logger.error("Monitor error: %s", e, exc_info=True)

        if args.loop <= 0:
            break
        print(f"\nNext update in {args.loop} seconds...")
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
