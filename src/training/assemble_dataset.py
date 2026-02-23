"""Assemble JSONL training dataset from curated episode JSONs.

For each episode, renders C_t canvas frames and encodes actions as compact
token strings, producing one JSONL record per action step.

Usage:
    python3 -m src.training.assemble_dataset \
        --episodes data/processed/video-2.json \
        --output   data/processed/dataset.jsonl \
        --frames-dir data/processed/frames
"""

import argparse
import json
import os
import sys

from src.dataset.canvas_render import CanvasRenderer
from src.training.encode_actions import encode_action, ACTION_TYPE_TO_ID


def _transcript_at(transcript: list, step_timestamp: float) -> str:
    """Return text of the last transcript segment whose start <= timestamp."""
    text = ""
    for seg in transcript:
        if seg["start"] <= step_timestamp:
            text = seg["text"]
    return text


def assemble_episode(
    episode_path: str,
    frames_dir: str,
) -> list:
    """Process one episode JSON into a list of training record dicts.

    Also saves C_t frame PNGs to frames_dir/<episode_id>/.
    """
    with open(episode_path) as f:
        episode = json.load(f)

    episode_id = episode["id"]
    actions = episode["actions"]
    transcript = episode.get("transcript", [])

    renderer = CanvasRenderer(
        width=episode.get("canvas", {}).get("w", 256),
        height=episode.get("canvas", {}).get("h", 256),
    )

    # Render all C_t frames (N+1 frames for N actions; frame[i] = canvas BEFORE action i)
    frames = renderer.render_all(actions)

    # Save frame PNGs
    ep_frames_dir = os.path.join(frames_dir, episode_id)
    os.makedirs(ep_frames_dir, exist_ok=True)

    records = []
    for step, action in enumerate(actions):
        # Save C_t frame (canvas state before this action)
        frame_filename = f"frame_{step:04d}.png"
        frame_path = os.path.join(ep_frames_dir, frame_filename)
        frames[step].save(frame_path)

        # Active transcript at this step's timestamp
        ts = action.get("timestamp", 0.0)
        active_transcript = _transcript_at(transcript, ts)

        # Encode action
        action_str = encode_action(action)
        action_type = ACTION_TYPE_TO_ID[action["type"]]

        records.append({
            "episode_id": episode_id,
            "step": step,
            "image_path": frame_path,
            "transcript": active_transcript,
            "action_str": action_str,
            "action_type": action_type,
        })

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Assemble JSONL training dataset from episode JSONs."
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        required=True,
        help="Path(s) to episode JSON files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--frames-dir",
        required=True,
        help="Directory to save C_t frame PNGs.",
    )
    args = parser.parse_args()

    all_records = []
    for ep_path in args.episodes:
        print(f"Processing {ep_path} ...")
        records = assemble_episode(ep_path, args.frames_dir)
        all_records.extend(records)
        print(f"  -> {len(records)} steps")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(all_records)} records to {args.output}")


if __name__ == "__main__":
    main()
