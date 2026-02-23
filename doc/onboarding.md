# WhiteboardGym — Onboarding Guide

This document gives a new contributor everything they need to understand and use
the WhiteboardGym codebase. Read it top-to-bottom; by the end you will know the
project goal, architecture, every module, every data format, and how to run
each pipeline stage.

---

## 1. Project Overview

WhiteboardGym reproduces the Stanford 2026 paper (included as `fp.pdf`) which
trains an RL agent to draw whiteboard lessons from human demonstrations. A
teacher records an Excalidraw drawing session while narrating; the pipeline
converts that recording into a training episode and fine-tunes a
vision-language model to predict drawing actions given the current canvas and
transcript.

### MDP Formulation (from the paper)

| Component | Definition |
|-----------|-----------|
| **State** | `(rasterized_canvas, transcript_index)` — a 256×256 image + which transcript segment is active |
| **Action** | One of 8 primitives: STROKE, TEXT, ERASE, SETCOLOR, SETWIDTH, WAIT, AUDIOSYNC, FINISH |
| **Reward** | Clarity + Pacing + Engagement + Alignment (distilled from GPT-4o into ViT + RoBERTa) |
| **Policy** | Qwen3-8B with ViT-B/16 canvas encoder (we use Qwen2-VL-2B for dev iteration) |
| **Training** | SFT → PPO-BC (decaying λ) → GRPO-BC |

### Roadmap

| # | Stage | Status |
|---|-------|--------|
| 1 | Parse Excalidraw | Done |
| 2 | Detect drawing offset | Done |
| 3 | Transcribe audio | Done |
| 4 | Normalize coordinates | Done |
| 5 | Build action sequence | Done |
| 6 | Curate episode JSON | Done |
| 7 | Encode actions + assemble JSONL | Done |
| 8 | SFT fine-tune | Code written, awaiting model download |
| 9 | Reward model | Not started |
| 10 | PPO-BC / GRPO-BC | Not started |

---

## 2. Architecture

```
Raw Recording                    Curation Pipeline                       Training Pipeline
─────────────                    ─────────────────                       ─────────────────

video-2.excalidraw ─┐
                    ├─→ curate.py ─→ video-2.json ─→ assemble_dataset.py ─→ dataset.jsonl ─┐
updated-video.mp4 ──┘     │              │                    │                             │
                          │              │                    ├─→ frames/*.png              │
                          │              │                    │                             │
                  ┌───────┴────────┐     │                   │                             │
                  │  Sub-modules:  │     │            encode_actions.py                    │
                  │  parse_excali  │     │            (Bézier fitting +                    │
                  │  detect_offset │     │             token encoding)                     │
                  │  transcribe    │     │                                                 │
                  │  normalize     │     │                                          sft_train.py
                  │  build_actions │     │                                          (LoRA fine-tune
                  │  canvas_render │     │                                           Qwen2-VL-2B)
                  └────────────────┘     │                                                 │
                                         │                                                 ▼
                                         │                                          checkpoints/
                                         │                                          sft-v1/
                                         │
                                  Episode JSON
                                  (48 actions, transcript,
                                   metadata)
```

**Data flow in one sentence:** An Excalidraw file and its screen-recording
video are curated into an episode JSON, which is then encoded into a JSONL
dataset of `(canvas_image, transcript, action_token)` triples used to SFT a
vision-language model.

---

## 3. Directory Structure

```
white-board-gym/
├── src/
│   ├── __init__.py
│   ├── dataset/                        # Stage 1-6: curation pipeline
│   │   ├── __init__.py
│   │   ├── parse_excalidraw.py         # Parse .excalidraw → sorted elements
│   │   ├── detect_offset.py            # OpenCV frame diff → drawing start time
│   │   ├── transcribe.py               # Whisper (local or API) → transcript segments
│   │   ├── normalize.py                # Global bbox → 256×256 coordinate mapping
│   │   ├── build_actions.py            # State machine → 8 action types
│   │   ├── canvas_render.py            # Action sequence → PIL canvas frames
│   │   └── curate.py                   # CLI orchestrator for the full pipeline
│   └── training/                       # Stage 7-8: encoding + SFT
│       ├── __init__.py
│       ├── encode_actions.py           # Bézier fitting + token encode/decode
│       ├── assemble_dataset.py         # Episode JSON → JSONL + frame PNGs
│       └── sft_train.py               # LoRA fine-tune Qwen2-VL-2B-Instruct
├── data/
│   └── processed/
│       ├── video-2.json                # Curated episode (48 actions)
│       ├── dataset.jsonl               # Training records (48 rows)
│       ├── frames/video-2/             # C_t canvas PNGs (frame_0000–0047)
│       └── frames_example/             # Example renders for visual inspection
├── doc/
│   ├── onboarding.md                   # ← You are here
│   ├── dataset_curation.md             # Deep-dive: curation pipeline design
│   └── qa_discussion.md               # Research Q&A log, gap analysis
├── requirements.txt                    # Python dependencies
├── fp.pdf                              # The paper being reproduced
├── video-2.excalidraw                  # Source drawing (21 elements)
├── updated-video.mp4                   # Source screen recording (35.4s, 688×480)
└── .env                                # OPENAI_API_KEY (for Whisper API)
```

---

## 4. Dataset Curation Pipeline (`src/dataset/`)

The curation pipeline takes a raw recording (Excalidraw file + video) and
produces a structured episode JSON. It runs as a single CLI command via
`curate.py`, which calls six sub-modules in sequence.

### 4.1 `parse_excalidraw.py`

**Purpose:** Parse `.excalidraw` JSON into sorted `ExcalidrawElement` dataclasses.

**Key function:** `get_sorted_elements(filepath) → List[ExcalidrawElement]` (line 29)
- Reads ALL elements including deleted ones (`is_deleted=True`)
- Computes `rel_time = (updated - min_updated) / 1000.0` from Excalidraw timestamps
- Sorts by `(rel_time, index)` for chronological order

**Dataclass fields:** `id`, `type`, `x`, `y`, `width`, `height`, `stroke_color`,
`stroke_width`, `updated`, `index`, `rel_time`, `is_deleted`, `text`,
`font_size`, `points`

### 4.2 `detect_offset.py`

**Purpose:** Find the timestamp when drawing first appears in the video.

**Key function:** `detect_drawing_start_offset(video_path, threshold=0.01) → float` (line 7)
- Uses OpenCV frame differencing against frame 0 (assumed blank)
- Returns the first frame where >1% of pixels have changed
- For `video-2`: returns **2.133 seconds**

### 4.3 `transcribe.py`

**Purpose:** Transcribe video audio to timed segments with word-level timestamps.

**Key functions:**
- `transcribe_video_local(video_path, model_name="base")` (line 23) — local Whisper
- `transcribe_video_api(video_path)` (line 52) — OpenAI Whisper API
- `transcribe_video(video_path, use_api=False)` (line 104) — high-level wrapper
- `segments_to_dicts(segments)` (line 111) — serialize to JSON-compatible dicts

**Dataclasses:** `WordTimestamp(word, start, end)`, `TranscriptSegment(start, end, text, words)`

### 4.4 `normalize.py`

**Purpose:** Map raw Excalidraw coordinates into a 256×256 canvas.

**Key function:** `build_normalizer(elements) → Normalizer` (line 28)
- Scans all points (freedraw vertices + text bounding boxes) for global min/max
- Returns a `Normalizer` with `norm_x(x)`, `norm_y(y)`, `norm_point(x, y)` methods
- Formula: `norm = max(0, min(255, int((val - min) / range * 255)))`

### 4.5 `build_actions.py`

**Purpose:** Convert sorted elements into a flat action sequence via a state machine.

**Key function:** `build_actions(elements, normalizer, drawing_start_offset, video_duration, transcript=None) → List[dict]` (line 30)

**State machine logic (main loop, lines 61–144):**
1. If gap > `GAP_THRESHOLD` (0.1s, line 12) → emit WAIT
2. If transcript segment changed → emit AUDIOSYNC
3. If element is deleted → emit ERASE
4. If color changed → emit SETCOLOR
5. If width changed → emit SETWIDTH
6. Emit primary action (STROKE or TEXT)
7. After all elements → final WAIT (if gap) + FINISH

**Timestamp formula:** `video_t = drawing_start_offset + elem.rel_time`

### 4.6 `canvas_render.py`

**Purpose:** Render action sequences into canvas state images for training.

**Key class:** `CanvasRenderer(width=256, height=256)` (line 45)
- `render_all(actions) → List[Image]` (line 95) — returns N+1 images (C₀ blank … C_N final)
- `render_at(actions, step) → Image` (line 111) — single frame before step
- `save_frames(actions, output_dir, steps)` (line 118) — write PNGs to disk

**Rendering:** STROKE → `ImageDraw.line`; TEXT → scaled font; ERASE → white rectangle.
Non-visual actions (WAIT, SETCOLOR, etc.) return the canvas unchanged.

**Why render instead of extracting video frames?** At inference time the agent
won't have a video — it generates actions on a blank canvas. Using rendered
frames ensures train/inference consistency. See `doc/qa_discussion.md` Q4.

### 4.7 `curate.py`

**Purpose:** CLI orchestrator that chains all sub-modules.

See [Section 7](#7-running-the-pipeline) for full CLI usage.

**Pipeline steps (lines 58–150):**
1. Parse excalidraw → sorted elements
2. Detect drawing start offset (or use `--drawing-offset` override)
3. Get video duration + transcribe audio
4. Build normalizer from global bounding box
5. Build action sequence
6. Assemble output JSON with metadata
7. Verify: FINISH is last, timestamps monotonic, STROKE points in [0, 255]

For the full schema and deep-dive, see `doc/dataset_curation.md`.

---

## 5. Training Pipeline (`src/training/`)

The training pipeline converts curated episode JSONs into model-ready data and
runs supervised fine-tuning.

### 5.1 `encode_actions.py`

**Purpose:** Encode/decode actions as compact token strings for language model training.

**Key functions:**
- `encode_action(action) → str` (line 61) — action dict → token string
- `decode_action(token) → dict` (line 108) — token string → action dict
- `_fit_bezier(pts) → list` (line 30) — fit cubic Bézier to variable-length stroke points

**Token format:**

| Action | Token | Example |
|--------|-------|---------|
| STROKE | `S(x0),(y0),(x1),(y1),(x2),(y2),(x3),(y3),(width)` | `S149,121,143,126,143,139,154,139,2` |
| TEXT | `T(x),(y),(fs),(content)` | `T17,9,14,Nuclear` |
| ERASE | `E(cx),(cy),(w),(h)` | `E128,128,50,30` |
| SETCOLOR | `C(r),(g),(b)` | `C30,30,30` |
| SETWIDTH | `L(width)` | `L2` |
| WAIT | `W(bins)` | `W45` (= 4.5 seconds) |
| AUDIOSYNC | `A(segment_idx)` | `A3` |
| FINISH | `F` | `F` |

**Bézier fitting (line 30):** Variable-length stroke polylines are fit to a
cubic spline (`scipy.interpolate.splprep`, s=0) and sampled at
t=[0, 0.333, 0.667, 1.0] → 4 control points (8 integers). Consecutive
duplicate points are removed before fitting to avoid `splprep` errors.

**WAIT bins:** `bins = min(int(duration / 0.1), 300)`. One bin = 100ms; max 30 seconds.

### 5.2 `assemble_dataset.py`

**Purpose:** Build JSONL training dataset and C_t frame PNGs from episode JSONs.

**Key function:** `assemble_episode(episode_path, frames_dir) → list` (line 31)
1. Loads episode JSON
2. Renders all canvas frames via `CanvasRenderer.render_all()`
3. Saves each C_t as `frames_dir/<episode_id>/frame_NNNN.png`
4. Encodes each action via `encode_action()`
5. Looks up active transcript text at each step's timestamp
6. Returns list of training record dicts

### 5.3 `sft_train.py`

**Purpose:** LoRA fine-tune Qwen2-VL-2B-Instruct to predict action tokens from
canvas images + transcript.

**Model:** `Qwen/Qwen2-VL-2B-Instruct` (line 29) — smaller than the paper's
Qwen3-8B, chosen for faster dev iteration.

**Prompt template (line 31):**
```
Transcript: {transcript}
Action:
```

The model receives a chat with two turns:
1. **User:** `[image]` + prompt text
2. **Assistant:** action token string (e.g. `S149,121,...`)

**LoRA config (lines 170–179):**
- `r=8`, `lora_alpha=16`, `lora_dropout=0.05`
- Target modules: `q_proj`, `v_proj`

**Label masking (lines 127–128):** Prompt tokens are set to `-100` so only the
action token string contributes to the cross-entropy loss.

**Dataset class:** `WhiteboardSFTDataset` (lines 42–142) — loads JSONL, returns
tokenized + masked training samples.

---

## 6. Data Formats

### 6.1 Episode JSON (`data/processed/video-2.json`)

Top-level structure:

```json
{
  "id": "video-2",
  "video_file": "updated-video.mp4",
  "excalidraw_file": "video-2.excalidraw",
  "video_duration": 35.367,
  "drawing_start_offset": 2.133,
  "canvas": { "w": 256, "h": 256 },
  "transcript": [],
  "actions": [ ... ]
}
```

### 6.2 Action Types

Every action has a `type` and `timestamp` (seconds into the video). Additional
fields depend on the type:

| Type | Extra fields | Description |
|------|-------------|-------------|
| `STROKE` | `points`, `color`, `width` | Freedraw polyline (normalized [0,255] coords) |
| `TEXT` | `content`, `x`, `y`, `font_size`, `font_size_norm` | Place text at position |
| `ERASE` | `cx`, `cy`, `w`, `h` | Clear a bounding-box region |
| `SETCOLOR` | `color` | Change active stroke color (`#rrggbb`) |
| `SETWIDTH` | `width` | Change active stroke width |
| `WAIT` | `duration` | Pause (seconds); emitted when gap > 0.1s |
| `AUDIOSYNC` | `segment_idx` | Advance to transcript segment |
| `FINISH` | — | End of episode |

**`video-2.json` action counts:** SETCOLOR ×4, SETWIDTH ×1, TEXT ×1,
STROKE ×20, WAIT ×21, FINISH ×1 = **48 total**.

### 6.3 Token Encoding

See the table in [Section 5.1](#51-encode_actionspy). Each action dict is
encoded to a single token string (e.g. `S149,121,143,126,143,139,154,139,2`).
The model predicts these strings autoregressively.

### 6.4 JSONL Training Records (`data/processed/dataset.jsonl`)

One JSON object per line, one line per action step:

```json
{
  "episode_id": "video-2",
  "step": 0,
  "image_path": "data/processed/frames/video-2/frame_0000.png",
  "transcript": "",
  "action_str": "C30,30,30",
  "action_type": 3
}
```

| Field | Description |
|-------|-------------|
| `episode_id` | Source episode identifier |
| `step` | 0-indexed action position |
| `image_path` | Path to C_t canvas frame (state before this action) |
| `transcript` | Active transcript text at this step's timestamp |
| `action_str` | Encoded action token string |
| `action_type` | Integer ID (0=STROKE, 1=TEXT, 2=ERASE, 3=SETCOLOR, 4=SETWIDTH, 5=WAIT, 6=AUDIOSYNC, 7=FINISH) |

---

## 7. Running the Pipeline

### Prerequisites

```bash
pip install -r requirements.txt
```

Dependencies: `opencv-python`, `numpy`, `openai`, `openai-whisper`, `Pillow`,
`scipy`, `transformers`, `peft`, `accelerate`.

For Whisper API transcription, set `OPENAI_API_KEY` in `.env` or your shell.

### Step 1: Curate an episode

```bash
python3 -m src.dataset.curate \
    --excalidraw video-2.excalidraw \
    --video      updated-video.mp4 \
    --output     data/processed/video-2.json \
    --id         video-2 \
    --skip-transcribe \
    --verbose
```

Remove `--skip-transcribe` to run Whisper. Add `--whisper-api` to use the
OpenAI API instead of the local model.

**Expected output:** `data/processed/video-2.json` with 48 actions.

**Verify:**
```python
import json
ep = json.load(open("data/processed/video-2.json"))
assert ep["actions"][-1]["type"] == "FINISH"
assert len(ep["actions"]) == 48
print("OK:", len(ep["actions"]), "actions")
```

### Step 2: Assemble training dataset

```bash
python3 -m src.training.assemble_dataset \
    --episodes   data/processed/video-2.json \
    --output     data/processed/dataset.jsonl \
    --frames-dir data/processed/frames
```

**Expected output:**
- `data/processed/dataset.jsonl` — 48 lines
- `data/processed/frames/video-2/` — 48 PNG files (`frame_0000.png` … `frame_0047.png`)

**Verify:**
```bash
wc -l data/processed/dataset.jsonl
# 48 data/processed/dataset.jsonl

ls data/processed/frames/video-2/ | wc -l
# 48
```

### Step 3: Run SFT (requires GPU or MPS)

```bash
python3 -m src.training.sft_train \
    --dataset    data/processed/dataset.jsonl \
    --output     checkpoints/sft-v1 \
    --epochs     2 \
    --batch-size 1
```

This downloads `Qwen/Qwen2-VL-2B-Instruct` (~4 GB) on first run and saves
LoRA adapter weights to `checkpoints/sft-v1/`.

---

## 8. Key Design Decisions

### Canvas is rendered, not extracted from video

At inference time the agent draws on a blank canvas — there is no video to
extract frames from. Using programmatically rendered C_t frames ensures
train/inference consistency. Video is only used for audio transcription and
drawing-start offset detection.

### Bézier fitting for strokes

Raw Excalidraw strokes have variable numbers of points (2 to hundreds). The
model needs fixed-size inputs, so each polyline is fit to a cubic Bézier curve
(4 control points = 8 integers + 1 width = 9 values total). This follows the
paper's approach of using spline approximations.

### WAIT uses the preceding action's timestamp

A WAIT represents idle time between actions. Its `timestamp` field equals the
timestamp of the action that preceded it (i.e., when the pause began), and its
`duration` field captures how long the pause lasted.

### Deduplication before `splprep`

`scipy.interpolate.splprep` fails on consecutive duplicate points. Before
fitting, `_fit_bezier` removes consecutive duplicates. Short strokes (≤1 unique
point) are handled by replicating the single point into 4 control points.

### Smaller model for dev iteration

The paper uses Qwen3-8B; this implementation uses Qwen2-VL-2B-Instruct for
faster iteration with limited hardware. The architecture (LoRA on q/v
projections, vision-language chat format) mirrors the paper's approach.

---

## 9. Current Status & Next Steps

### Implemented and verified

- Full curation pipeline (parse → offset → transcribe → normalize → actions → curate)
- Canvas renderer producing C_t frames from action sequences
- Action token encoding/decoding with Bézier fitting
- JSONL dataset assembly with frame PNGs
- SFT training script with LoRA (code complete, not yet executed)
- One demo curated: `video-2` (48 actions, 35.4s)

### Pending

| Item | Notes |
|------|-------|
| Execute SFT | Download Qwen2-VL-2B, run `sft_train.py`, verify loss convergence |
| Curate 49 more demos | Paper uses 50 demos, 5–10s each, 20–40 actions each |
| ADDIMAGE action type | Not yet implemented (no image elements in current demos) |
| Reward model | Clarity + Pacing + Engagement + Alignment; distill from GPT-4o into ViT + RoBERTa |
| PPO-BC training | Decaying behavioral cloning regularization λ |
| GRPO-BC training | Final RL stage from the paper |

---

## 10. Further Reading

- **`doc/dataset_curation.md`** — Deep-dive into curation pipeline design,
  output schema, and verification steps.
- **`doc/qa_discussion.md`** — Research Q&A log covering FPS handling, WAIT
  semantics, Bézier rationale, canvas rendering justification, and paper gap
  analysis (Gaps A–J).
- **`fp.pdf`** — The original WhiteboardGym paper.
