"""CLI orchestrator for the dataset curation pipeline.

Usage:
    python3 -m src.dataset.curate \\
        --excalidraw data/raw/video-2.excalidraw \\
        --video      data/raw/updated-video.mp4 \\
        --output     data/processed/video-2.json \\
        --id         video-2 \\
        --verbose

    # Skip Whisper (dev without API key)
    python3 -m src.dataset.curate ... --skip-transcribe

    # Manual offset override
    python3 -m src.dataset.curate ... --drawing-offset 2.133
"""

import argparse
import json
import os
import sys


def get_video_duration(video_path: str) -> float:
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0 and frame_count > 0:
        return round(frame_count / fps, 3)
    raise RuntimeError("Could not determine video duration")


def main():
    parser = argparse.ArgumentParser(description="Curate a WhiteboardGym training episode")
    parser.add_argument("--excalidraw", required=True, help="Path to .excalidraw file")
    parser.add_argument("--video", required=True, help="Path to video file (MP4)")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--id", default="episode", help="Episode ID")
    parser.add_argument("--drawing-offset", type=float, default=None,
                        help="Manual override for drawing start offset (seconds)")
    parser.add_argument("--skip-transcribe", action="store_true",
                        help="Skip Whisper transcription entirely")
    parser.add_argument("--whisper-api", action="store_true",
                        help="Use OpenAI Whisper API instead of local model (requires OPENAI_API_KEY)")
    parser.add_argument("--whisper-model", default="base",
                        help="Local whisper model size: tiny, base, small, medium, large (default: base)")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    args = parser.parse_args()

    def log(msg: str):
        if args.verbose:
            print(msg)

    # Step 1: Parse excalidraw
    log("Step 1: Parsing excalidraw...")
    from .parse_excalidraw import get_sorted_elements
    elements = get_sorted_elements(args.excalidraw)
    log(f"  Found {len(elements)} elements")

    # Step 2: Detect drawing start offset
    if args.drawing_offset is not None:
        drawing_start_offset = args.drawing_offset
        log(f"Step 2: Using manual drawing offset = {drawing_start_offset}s")
    else:
        log("Step 2: Detecting drawing start offset via frame differencing...")
        from .detect_offset import detect_drawing_start_offset
        drawing_start_offset = detect_drawing_start_offset(args.video)
        log(f"  Detected offset = {drawing_start_offset}s")

    # Get video duration
    log("       Getting video duration...")
    video_duration = get_video_duration(args.video)
    log(f"  Video duration = {video_duration}s")

    # Step 3: Transcribe
    if args.skip_transcribe:
        log("Step 3: Skipping transcription (--skip-transcribe)")
        transcript = []
    elif args.whisper_api:
        log("Step 3: Transcribing audio via OpenAI Whisper API...")
        from .transcribe import transcribe_video, segments_to_dicts
        segments = transcribe_video(args.video, use_api=True)
        transcript = segments_to_dicts(segments)
        log(f"  Got {len(transcript)} segments")
    else:
        log(f"Step 3: Transcribing audio via local Whisper (model={args.whisper_model})...")
        from .transcribe import transcribe_video, segments_to_dicts
        segments = transcribe_video(args.video, use_api=False, model_name=args.whisper_model)
        transcript = segments_to_dicts(segments)
        log(f"  Got {len(transcript)} segments")

    # Step 4 & 5: Normalize + build actions
    log("Step 4: Computing normalization (256×256)...")
    from .normalize import build_normalizer
    normalizer = build_normalizer(elements)
    log(f"  Bounding box: x=[{normalizer.min_x:.1f}, {normalizer.max_x:.1f}], "
        f"y=[{normalizer.min_y:.1f}, {normalizer.max_y:.1f}]")

    log("Step 5: Building action sequence...")
    from .build_actions import build_actions
    actions = build_actions(
        elements, normalizer, drawing_start_offset, video_duration,
        transcript=transcript or None,
    )
    log(f"  Generated {len(actions)} actions")

    from collections import Counter
    counts = Counter(a["type"] for a in actions)
    log(f"  Counts: {dict(counts)}")
    if counts.get("ERASE"):
        log(f"  ({counts['ERASE']} ERASE actions from deleted elements)")
    if counts.get("AUDIOSYNC"):
        log(f"  ({counts['AUDIOSYNC']} AUDIOSYNC actions from transcript segments)")

    # Assemble output
    excalidraw_basename = os.path.basename(args.excalidraw)
    video_basename = os.path.basename(args.video)

    output = {
        "id": args.id,
        "video_file": video_basename,
        "excalidraw_file": excalidraw_basename,
        "video_duration": video_duration,
        "drawing_start_offset": drawing_start_offset,
        "canvas": {"w": 256, "h": 256},
        "transcript": transcript,
        "actions": actions,
    }

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    log(f"\nWrote {args.output}")

    # Quick verification
    log("\nVerifying output...")
    assert output["actions"][-1]["type"] == "FINISH", "Last action must be FINISH"
    assert abs(output["actions"][-1]["timestamp"] - video_duration) < 0.01, "FINISH timestamp mismatch"
    ts = [a["timestamp"] for a in output["actions"]]
    assert all(ts[i] <= ts[i + 1] for i in range(len(ts) - 1)), "Timestamps not monotonic"
    for a in output["actions"]:
        if a["type"] == "STROKE":
            assert all(0 <= p[0] <= 255 and 0 <= p[1] <= 255 for p in a["points"]), \
                "STROKE point out of [0,255]"
    log("  All checks passed!")


if __name__ == "__main__":
    main()
