"""Detect drawing start offset in video using frame differencing."""

import cv2
import numpy as np


def detect_drawing_start_offset(video_path: str, threshold: float = 0.01) -> float:
    """Return the timestamp (seconds) when drawing first appears in the video.

    Algorithm:
    1. Read frame 0 as the blank reference (mean pixel >= 250 confirms white canvas).
    2. Walk frames forward, comparing each to the blank reference.
    3. First frame where changed_pixels / total_pixels > threshold is the offset.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Read reference frame (frame 0)
    ret, ref_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Cannot read frame 0 from video")

    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    total_pixels = ref_gray.size

    # Warn if frame 0 is not a white canvas
    if ref_gray.mean() < 200:
        print(f"  Warning: frame 0 mean pixel = {ref_gray.mean():.1f} (expected ~255 for white canvas)")

    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.abs(gray - ref_gray)
        changed_ratio = np.sum(diff > 10) / total_pixels

        if changed_ratio > threshold:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            cap.release()
            return round(timestamp, 3)

        frame_idx += 1

    cap.release()
    # Fallback: no change detected, return 0
    return 0.0
