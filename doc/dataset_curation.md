# WhiteboardGym — Dataset Curation Pipeline

## Overview

This pipeline converts a raw whiteboard recording (Excalidraw JSON + video with audio) into a structured JSON training episode for the WhiteboardGym RL agent.

**Key insight:** Excalidraw elements already contain `updated` Unix-millisecond timestamps, so action ordering requires no computer vision. The video is only needed for:
1. Audio transcription (what is being said)
2. Finding the drawing start offset (when in the video the first mark appears)

---

## Pipeline Steps

### Step 1 — `parse_excalidraw.py`

Parses `.excalidraw` JSON into sorted `ExcalidrawElement` dataclasses.

- Filters out `isDeleted=True` elements
- Computes `rel_time = (updated - min_updated) / 1000.0` for each element
- Sorts by `(rel_time, index)` — the `index` field (lexicographic: "a0", "a1", ..., "aA", "aB", ...) breaks ties

**Key function:** `get_sorted_elements(filepath) -> List[ExcalidrawElement]`

### Step 2 — `detect_offset.py`

Uses OpenCV frame differencing to find when drawing first appears in the video.

**Algorithm:**
1. Read frame 0 as blank reference (white canvas, mean pixel ≈ 255)
2. Walk frames forward, compare each to blank reference
3. First frame where `changed_pixels / total_pixels > 0.01` → that timestamp is `drawing_start_offset`

For the demo video (`updated-video.mp4`, 35.38s), this returns **2.133s** (frame 128 at ~60fps).

**CLI override:** `--drawing-offset 2.133` skips auto-detection.

### Step 3 — `transcribe.py`

Calls OpenAI Whisper API with the MP4 file (AAC audio, under 25MB limit).

Returns `List[TranscriptSegment]` with `start`, `end`, `text`, and word-level timestamps.

**Requirements:** `OPENAI_API_KEY` environment variable.
**Dev mode:** `--skip-transcribe` skips this step entirely.

### Step 4 — `normalize.py`

Computes global bounding box across ALL elements, then provides a `Normalizer` for 256×256 coordinate scaling.

**Point extraction:**
- `freedraw`: absolute point = `(element.x + pt[0], element.y + pt[1])`
- `text`: bounding box corners `(element.x, element.y)` and `(element.x + width, element.y + height)`

**Normalization:**
```
norm_x(x) = clamp(int((x - min_x) / dx * 255), 0, 255)
norm_y(y) = clamp(int((y - min_y) / dy * 255), 0, 255)
```

For the demo: min_x=565, max_x=1055, min_y=211, max_y=554 (dx=490, dy=343).

### Step 5 — `build_actions.py`

State machine converting sorted elements into a flat action sequence.

**State tracked:** `current_color`, `current_width`

**For each element (in order):**
1. **Gap check** (not for first element): if `rel_time[i] - rel_time[i-1] > 0.1s` → emit `WAIT(duration, timestamp=prev_video_t)`
2. **Color change**: if `element.strokeColor != current_color` → emit `SETCOLOR(color, timestamp=video_t)`
3. **Width change**: if `element.strokeWidth != current_width` → emit `SETWIDTH(width, timestamp=video_t)`
4. **Primary action** at `timestamp=video_t`:
   - `text` → `TEXT(content, x_center_norm, y_center_norm, font_size)`
   - `freedraw` → `STROKE(points_normalized, color, width)`

**After last element:**
- If `video_duration - last_video_t > 0.1s` → emit `WAIT(tail_gap, last_video_t)`
- Emit `FINISH(timestamp=video_duration)`

**Video timestamp:** `video_t = drawing_start_offset + rel_time`

### Step 6 — `curate.py` (CLI)

Orchestrates all steps. See usage examples below.

---

## Usage

```bash
# Full pipeline (requires OPENAI_API_KEY)
python3 -m src.dataset.curate \
    --excalidraw data/raw/video-2.excalidraw \
    --video      data/raw/updated-video.mp4 \
    --output     data/processed/video-2.json \
    --id         video-2 \
    --verbose

# Skip Whisper (dev/testing without API key)
python3 -m src.dataset.curate \
    --excalidraw data/raw/video-2.excalidraw \
    --video      data/raw/updated-video.mp4 \
    --output     data/processed/video-2.json \
    --id         video-2 \
    --skip-transcribe \
    --verbose

# Manual offset override (skip auto-detection)
python3 -m src.dataset.curate \
    --excalidraw data/raw/video-2.excalidraw \
    --video      data/raw/updated-video.mp4 \
    --output     data/processed/video-2.json \
    --id         video-2 \
    --drawing-offset 2.133 \
    --skip-transcribe \
    --verbose
```

---

## Output JSON Schema

```json
{
  "id": "video-2",
  "video_file": "updated-video.mp4",
  "excalidraw_file": "video-2.excalidraw",
  "video_duration": 35.377,
  "drawing_start_offset": 2.133,
  "canvas": {"w": 256, "h": 256},
  "transcript": [
    {"start": 0.0, "end": 3.2, "text": "...", "words": [{"word": "Today", "start": 0.0, "end": 0.3}]}
  ],
  "actions": [
    {"type": "SETCOLOR", "color": "#1e1e1e", "timestamp": 2.133},
    {"type": "SETWIDTH", "width": 2, "timestamp": 2.133},
    {"type": "TEXT", "content": "Nuclear", "x": 17, "y": 9, "font_size": 20.0, "timestamp": 2.133},
    {"type": "WAIT", "duration": 4.463, "timestamp": 2.133},
    {"type": "STROKE", "points": [[149,121],[149,120]], "color": "#1e1e1e", "width": 2, "timestamp": 6.596},
    "...",
    {"type": "FINISH", "timestamp": 35.377}
  ]
}
```

### Action Types

| Type | Fields | Meaning |
|------|--------|---------|
| `SETCOLOR` | `color`, `timestamp` | Change pen color |
| `SETWIDTH` | `width`, `timestamp` | Change pen width |
| `TEXT` | `content`, `x`, `y`, `font_size`, `timestamp` | Place text label |
| `STROKE` | `points`, `color`, `width`, `timestamp` | Draw a freedraw stroke |
| `WAIT` | `duration`, `timestamp` | Pause for given seconds |
| `FINISH` | `timestamp` | End of episode |

---

## Expected Counts (demo: `video-2`)

| Action | Count |
|--------|-------|
| SETCOLOR | 4 (#1e1e1e → #c2255c → #1e1e1e → #6741d9) |
| SETWIDTH | 1 (width=2, set once at start) |
| TEXT | 1 ("Nuclear") |
| STROKE | 20 (freedraw elements) |
| WAIT | 21 (20 inter-element gaps + 1 tail) |
| FINISH | 1 |
| **Total** | **48** |

---

## Verification

```python
import json
from collections import Counter

with open("data/processed/video-2.json") as f:
    d = json.load(f)

assert d["actions"][-1]["type"] == "FINISH"
assert abs(d["actions"][-1]["timestamp"] - d["video_duration"]) < 0.01
ts = [a["timestamp"] for a in d["actions"]]
assert all(ts[i] <= ts[i+1] for i in range(len(ts)-1))
for a in d["actions"]:
    if a["type"] == "STROKE":
        assert all(0 <= p[0] <= 255 and 0 <= p[1] <= 255 for p in a["points"])

counts = Counter(a["type"] for a in d["actions"])
assert counts["SETCOLOR"] == 4
assert counts["SETWIDTH"] == 1
assert counts["TEXT"] == 1
assert counts["STROKE"] == 20
assert counts["FINISH"] == 1
assert counts["WAIT"] >= 1
print("All checks passed!")
```

---

## Dependencies

All already installed in the project environment:

```
opencv-python>=4.8.0   # frame differencing for offset detection
numpy>=1.26.0          # array operations
openai>=2.0.0          # Whisper API for transcription
```
