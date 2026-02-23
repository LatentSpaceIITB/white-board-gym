# WhiteboardGym — Q&A Discussion

A running log of questions, answers, and design insights discovered while building the dataset curation pipeline and reading the paper.

---

## Q1: Are we taking FPS into account?

**Short answer: FPS only matters in one place, and we handle it correctly there.**

The pipeline has two independent timing systems:

| System | Source | FPS involved? |
|--------|--------|---------------|
| Action timestamps | Excalidraw `updated` (Unix ms) | No |
| Drawing start offset | Video frame scan (OpenCV) | Yes, but handled correctly |

For offset detection (`detect_offset.py`), we scan frames one-by-one but read `cap.get(cv2.CAP_PROP_POS_MSEC)` — the actual millisecond position — rather than computing `frame_index / fps`. So the result (2.133s) is accurate regardless of whether the video is 30fps, 60fps, or variable.

For everything else — STROKE timestamps, WAITs, gaps — we use pure Excalidraw timestamps. The video's framerate never touches those.

**Where FPS will matter later (not yet):** When rendering the rasterized canvas for training, you'll need to produce canvas images at each action step. That's a rendering concern, not a dataset curation concern.

---

## Q2: Freedraw tracing — all coordinates or just start/finish?

**We store ALL points.** For example, the first STROKE has 16 points:
```json
[[149,121],[149,120],[148,118],[147,120],[146,121],[145,124],[143,126],...]
```

The paper confirms this design (Section IV.B):

> "the stroke primitive takes in as input an **ordered set of coordinates** and sketches between those coordinates"

A STROKE is an **atomic action** — the agent outputs "draw this entire polyline" as a single decision. The environment renders all points at once. There is no per-point timing.

**Critical subtlety: we only have ONE timestamp per stroke — the completion time.** The `updated` field in Excalidraw records when the pen lifted, not when each point was drawn. We have full spatial accuracy (all points) but no within-stroke temporal resolution. The paper doesn't require within-stroke timing — each stroke is one indivisible MDP transition.

---

## Q3: What do the gaps/WAITs actually mean?

WAIT actions represent measured pauses between consecutive stroke **completion** timestamps. They do NOT precisely equal "thinking time" because:

1. Stroke a1 completes at t=6.596s
2. Stroke a2 completes at t=7.265s
3. Gap = 0.669s → we emit `WAIT(0.669)`

But the user may have started drawing a2 at t=6.6s (immediately after lifting pen from a1). The 0.669s gap includes the drawing time of a2, not just idle time. This is a known approximation — the Excalidraw format only stores completion timestamps.

For the RL agent, this is acceptable: the agent learns to pace its output at roughly the same rate as the demonstrator.

---

## Gaps Identified from Reading the Paper

### Gap A: The MDP formulation (Section III)

The agent operates as a Markov Decision Process:

```
State:      s_t = (C_t, τ_t)
              C_t = rasterized canvas image (what is drawn so far)
              τ_t = current transcript index (what is being said)

Action:     a_t ∈ {STROKE, TEXT, ERASE, SETCOLOR, SETWIDTH, ADDIMAGE, WAIT, FINISH}

Transition: deterministic — environment renders action onto canvas, advances transcript clock

Reward:     r(s_t, a_t) = weighted(Clarity, Pacing, Engagement, Alignment)
```

Key insight: this is NOT frame-based. The agent takes one action per step (~1-4 actions/second), not one action per video frame.

### Gap B: What the agent sees (observation space)

The agent sees two things at each step:

1. **Rasterized canvas C_t** — a pixel image of everything drawn so far (processed by ViT-B/16)
2. **Transcript sliding window τ_t** — the narration text around the current position (processed by Qwen3-8B's text encoder)

This means we will eventually need a **canvas renderer** that replays actions 0..t and produces an image. Our JSON stores the action sequence; rendering is a separate step for training.

### Gap C: The agent's action output

The policy model is **Qwen3-8B** (a language model). It predicts action tokens autoregressively. The canvas image is processed by ViT and injected as embeddings.

Actions are encoded as **fixed-length integer vectors**. Our variable-length STROKE points will need to be padded/truncated or tokenized before model training. The current JSON is the correct intermediate format.

### Gap D: The reward function (Section V)

Four components, scored by a distilled ViT-B/16 + RoBERTa network:

| Component | What it measures |
|-----------|-----------------|
| **Clarity** | Are diagrams legible? Is spatial layout clean? |
| **Pacing** | Does drawing happen at the right speed relative to speech? |
| **Engagement** | Is there enough visual activity to hold attention? |
| **Alignment** | Does what's drawn match what's being said? |

The reward network was trained on 15,000 (canvas, transcript) pairs scored by GPT-4o, then distilled into the lightweight network for interactive-speed evaluation during training (Spearman ρ ≥ 0.86 vs GPT-4o).

### Gap E: Three-stage training pipeline

| Stage | Method | Purpose |
|-------|--------|---------|
| 1. SFT-30 | Behavior Cloning on 50 demos, 2 epochs | Teach basic drawing via imitation |
| 2. PPO-BC | PPO + decaying BC loss (λ decays via exp(-k/2000)) | RL improves on demos while staying grounded |
| 3. GRPO-BC | Group-relative policy optimization | Final refinement, no critic needed |

The BC loss starts at λ₀ = 1.0 and decays exponentially. Early in training, the agent closely imitates demos. By step 2000+, RL reward dominates and the agent surpasses human quality.

### Gap F: Our demo is longer than the paper's

The paper uses 50 demos of **5-10 seconds** each with **20-40 actions** (~4 actions/sec). Our demo is **35 seconds** with **48 actions** (~1.4 actions/sec). It's 3-7x longer with lower action density. Future recordings should aim for shorter, denser demos.

### Gap G: Missing action types (now fixed)

The paper defines 8 action types. Our original pipeline had 6. Now updated:

| Paper action | Pipeline status |
|-------------|----------------|
| STROKE | ✓ from v1 |
| TEXT | ✓ from v1 |
| SETCOLOR | ✓ from v1 |
| SETWIDTH | ✓ from v1 |
| WAIT | ✓ from v1 |
| FINISH | ✓ from v1 |
| **ERASE** | ✓ added in v2 |
| **AUDIOSYNC** | ✓ added in v2 |
| ADDIMAGE | Not yet (no image elements in current demos) |

### Gap H: ERASE — what we can and can't recover

Excalidraw marks deleted elements with `isDeleted: true`. The `updated` timestamp on a deleted element is the **deletion time**, not the creation time. So:

- We CAN emit an ERASE action at the correct deletion time with the correct bounding box
- We CANNOT recover when the erased element was originally drawn (that timestamp is overwritten)

For the RL agent: the ERASE action tells it "at time T, erase this region." The agent must learn to draw the element earlier (from other cues) and then erase it. The pairing of draw-then-erase is learned implicitly.

Our current demo (video-2) has **zero deletions**, so ERASE actions don't affect its output.

### Gap I: audioSync — transcript-action alignment

The AUDIOSYNC action explicitly synchronizes the agent with the narration timeline. When a new transcript segment becomes active during the action sequence, we emit:

```json
{"type": "AUDIOSYNC", "segment_index": 2, "text": "the nuclear reaction...", "timestamp": 5.4}
```

This teaches the agent: "at this timestamp, you should be attending to transcript segment #2." Without audioSync, the agent would have to infer the transcript position purely from timing.

### Gap J: Pressure data is ignored

Every freedraw element has a `pressures` array. In our demo, all values are uniform (0.12). The paper doesn't mention pressure. We discard it. If future demos have variable pressure, this would need revisiting.

---

## Open Questions (for future work)

1. ~~Canvas renderer~~ → Resolved below (Q4)

2. **Action tokenization:** How to encode variable-length STROKE points into fixed-length integer vectors for Qwen3-8B? Truncation? Subsampling? Bezier approximation?

3. **More demos:** Need 49 more recordings. What's the target domain? Nuclear physics only, or diverse topics?

4. **Reward model data:** The 15,000 GPT-4o scored pairs — do we generate these from our demos, or is this a separate data collection?

5. **Demo length:** Should we re-record video-2 as a shorter 5-10 second clip, or is the pipeline flexible enough to handle varied lengths?

---

## Q4: Are we not feeding video snapshots? Where does canvas rendering come in?

**We do NOT use video frames as canvas states.** The video is used for exactly two things — both already done:

1. Audio → Whisper → transcript segments (what is being said)
2. Frame differencing → drawing start offset (when drawing begins)

After that, the video is set aside entirely.

### Why not use video frames?

The problem is a **train/inference mismatch**. Consider what each looks like:

```
Video frame (688×480):          Clean rendered canvas (256×256):
- Teacher's hand in frame       - White background only
- Stylus/pen visible            - Pure colored strokes
- Lighting variation            - No hand, no noise
- Camera shake                  - Crisp normalized coordinates
- Background behind canvas
```

At inference time, the agent has no video — it draws on a fresh programmatic canvas and sees only what it has drawn itself. If we trained on video frames, the model would learn features (hand shapes, lighting, camera noise) that simply don't exist at inference time. The model would fail immediately.

### The correct pipeline

```
Video
  ├── Audio ──────────────────────────────→ Transcript (τ_t)
  └── Frame differencing ─────────────────→ Drawing start offset

Excalidraw JSON
  └── Curation pipeline ───────────────────→ Action sequence [a_0, a_1, ..., a_N]

Action sequence
  └── Canvas renderer (replay step-by-step) → Clean C_t images (256×256)

Training pairs:  (C_t, τ_t)  →  a_t      ← what the model learns
```

### What C_t actually looks like

C_t is the canvas state **before** action t is applied. For video-2:

| Step | C_t (what agent sees) | Action taken |
|------|----------------------|--------------|
| 0 | blank white 256×256 | AUDIOSYNC seg 0 |
| 1 | blank white 256×256 | SETCOLOR #1e1e1e |
| 2 | blank white 256×256 | SETWIDTH 2 |
| 3 | blank white 256×256 | TEXT "Nuclear" at (17,9) |
| 4 | canvas with "Nuclear" label | WAIT 4.463s |
| 5 | canvas with "Nuclear" label | STROKE (first black circle, 16 pts) |
| 6 | canvas with "Nuclear" + circle 1 | STROKE (circle 2) |
| ... | ... | ... |
| 49 | nearly complete diagram | FINISH |

The agent never sees a video frame — it always sees the programmatic render of its own prior actions.

### Implementation

Canvas renderer (`src/dataset/canvas_render.py`) uses Pillow (PIL) to replay
the action sequence on a 256×256 white canvas:
- `STROKE` → `ImageDraw.line(points, fill=color, width=width)`
- `TEXT` → `ImageDraw.text(center, content, font=scaled_font)`
- `ERASE` → `ImageDraw.rectangle(bbox, fill="white")`
- `WAIT / SETCOLOR / SETWIDTH / AUDIOSYNC / FINISH` → no pixel change

`CanvasRenderer.render_all(actions)` returns N+1 images: C_0 (blank) through C_N (final canvas).

---

## Q5: What comes after canvas rendering? What is the full roadmap?

```
✅ 1. Dataset curation      raw recording → action sequence JSON
✅ 2. Canvas renderer       action sequence → C_t images (256×256)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
→  3. Action encoding       action dicts → fixed-length integer vectors
   4. Training data assembly (C_t, τ_t, action_vec) triplets per step
   5. SFT on Qwen3-8B       behavior cloning on demos (Stage 1)
   6. Reward model           distilled ViT+RoBERTa scorer
   7. PPO-BC                 RL with decaying imitation loss (Stage 2)
   8. GRPO-BC                final refinement (Stage 3)
```

**Step 3 (Action Encoding)** is the next code milestone. The challenge: our JSON has
variable-length data — a STROKE can have 9 points or 75 points — but Qwen3-8B needs
fixed-size input.

### The two paths forward

**Path A — go deeper (code first):**
Build action encoding → training data assembly → run SFT on 1 demo to prove the
pipeline works end-to-end. Then collect more demos.

**Path B — go wider (data first):**
Record more demos now (pipeline handles them automatically), reach 10-20 demos,
then build encoding + SFT with a healthier dataset.

**Decision: Path B** — collect more recordings first, then encode.

---

## Q6: How does the paper solve variable-length stroke points?

This is the action encoding design question for STROKE actions. Three naive approaches were considered:
1. Truncate/pad to a fixed max N (e.g. 64 points), fill unused slots with a sentinel
2. Subsample to exactly N points equally spaced along the stroke
3. Bézier approximation — fit a curve, encode control points only

**The paper uses Bézier approximation.** From Section IV.B:

> *"Parameter values (e.g., **Bézier points**, colour) are predicted in a subsequent decoder
> but the policy optimization treats the primitive symbolically."*

### Two-stage prediction architecture

The action prediction is split into two stages:

**Stage 1 — Qwen3-8B selects the action type symbolically**
```
Input:  (C_t image, transcript τ_t)
Output: action type integer  →  e.g. 1 (STROKE)
```

**Stage 2 — A separate parameter decoder fills in coordinates and parameters**
```
Input:  action type + (C_t, τ_t)
Output: Bézier control points, color, width, etc.
```

This clean separation means the main language model (Qwen3-8B) never has to deal
with variable-length point lists at all — it just picks a type. The decoder handles
the geometry.

### What Bézier approximation does

A cubic Bézier curve represents ANY smooth curve with exactly 4 control points,
regardless of how many raw sample points the original stroke had:

```
Raw freedraw stroke (variable):     Cubic Bézier (always 4 points):
[p0, p1, p2, ..., p47]         →    [P0, P1, P2, P3]
16 pts or 75 pts — doesn't matter    always 4 control points = 8 integers
```

The control points are mathematical handles, not points on the curve itself. The
curve passes through P0 and P3 (endpoints), and P1/P2 pull it toward them:

```
P1 ●                    ● P2
    \                  /
     \   smooth arc   /
      ●──────────────●
     P0              P3
```

For strokes that change direction sharply (squiggles, complex shapes), multiple
Bézier segments are chained. The paper likely uses a fixed number of segments per
stroke (e.g., 1 or 2 cubic segments = 8 or 16 integers total).

### Implementation note (for when we build Step 3)

Python's `scipy.interpolate` handles least-squares Bézier fitting:
```python
from scipy.interpolate import splprep, splev
# Fit spline to raw points, then evaluate at fixed parameter values
# → gives fixed N resampled points OR convert to Bézier control points
```

This is Step 3 of the roadmap — deferred until after data collection (Path B).
