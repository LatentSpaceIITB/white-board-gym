# Stage 1 Spatial RL MVP

This repository implements an MVP for Stage 1 from the formalization docs:

- deterministic drawing environment
- visual policy loop over canvas + concept
- terminal reward (`r_vis` + `r_struct`)
- GRPO-style grouped policy updates with KL anchor
- dual runtime profiles:
  - `nvidia_24gb` for full MVP training
  - `apple_silicon_dev` for local smoke/dev runs

## Quick start

### 1) Recommended Python

Use Python 3.11 or 3.12 for ML dependencies (PyTorch + optional CLIP judge).

### 2) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[ml]
```

Optional CLIP visual judge:

```bash
pip install -e .[clip]
```

### 3) Run smoke training (Apple profile)

```bash
python -m spatial_rl.train.run_stage1 --profile apple_silicon_dev --smoke
```

### 4) Run MVP training (NVIDIA profile)

```bash
python -m spatial_rl.train.run_stage1 --profile nvidia_24gb
```

Artifacts are written to `runs/<timestamp>/`:

- `resolved_config.yaml`
- `metrics.jsonl`
- `samples/` image snapshots
- `checkpoints/` policy checkpoints

## Notes

- The default policy backend is `structured_torch` for MVP reliability.
- The renderer is Python/Pillow deterministic and easy to debug.
- `r_vis` supports two backends:
  - `heuristic` (no extra model, fast)
  - `clip` (requires transformers + torch, better signal)
