# Architecture Plan & Session Findings

**Date:** 2026-03-15  
**Context:** Switched from local MacBook development to SSH machine with 4× NVIDIA RTX A6000 (48 GB each). Full end-to-end execution now possible including model training, evaluation, and GPU-accelerated reward computation.

---

## 1. Project Context

The repository (`white-board`) is a Stage 1 spatial-RL MVP. A structured PyTorch policy learns to draw educational diagrams from concept prompts, trained with a terminal reward composed of a visual score and a structural score.

### Current State (as of this session)

| Component | File | Status |
|---|---|---|
| Policy backend | `src/spatial_rl/policy/factory.py:12` | Only `structured_torch` supported |
| Training loop | `src/spatial_rl/train/grpo.py:116` | Eval mixed into training |
| Episode execution | `src/spatial_rl/train/grpo.py:157` | Not extracted; no shared runner |
| Visual judge | `src/spatial_rl/reward/visual.py:35` | Local CLIP only |
| Eval CLI | — | Does not exist |
| Baseline runner | — | Does not exist |
| Data splits | `data/concepts_train.txt`, `data/concepts_val.txt` | Flat text, no family metadata |
| GPU profile | `configs/profiles/nvidia_24gb.yaml` | Scaffolded, not used in practice |
| Runtime | Python 3.13.11 (active shell) | Mismatched; repo targets 3.11/3.12 |

### Hardware

- 4× RTX A6000 48 GB
- CUDA available, BF16 supported
- Current plan: single-GPU bring-up first, multi-GPU later

### Installed Packages (current env)

```
torch          2.10.0
transformers   4.46.1
accelerate     1.12.0
peft           0.18.1
trl            0.27.1
datasets       3.1.0
flash_attn     NOT INSTALLED
vllm           NOT INSTALLED
bitsandbytes   NOT INSTALLED
```

---

## 2. Full Discussion Summary

### Decision: Include new model + eval architecture before the first full run

Rather than just running the existing MVP on CUDA, the plan was expanded to include:

- A new VLM policy backend (`Qwen3-VL`)
- A standalone evaluation architecture (separate from training)
- A local committee judge (replacing CLIP-only)
- A baseline comparison matrix
- A phased experiment ladder

### Decision: Keep everything fully local

No API judges (no GPT-4o, no external rescoring). All reward computation, evaluation, and judging runs on local hardware. This changes the judge strategy toward a multi-model local committee (SigLIP2 + structural) and separating train judge config from eval judge config to avoid leakage.

### Decision: Use Qwen3-VL, not Qwen3 text-only

`Qwen3` (text-only) was the original suggestion. Web research confirmed that `Qwen3-VL` exists as a separate multimodal family with full HF `transformers` support via `Qwen3VLForConditionalGeneration`. This is the correct model for image-conditioned action generation and visual scoring.

**Recommended default:** `Qwen/Qwen3-VL-4B-Instruct`  
**Reason:** Best balance of quality and VRAM headroom on a single A6000 for LoRA + online RL. `8B` is possible but is a worse first RL target.

### Decision: Recommendation → implementation spec first, then runbook

Writing the runbook before the architecture is refactored would produce a plan that changes as soon as the rollout/eval/judge paths are separated. The implementation spec forces risk mitigations into the design upfront.

**Sequence:**
1. File-by-file implementation spec ← **next artifact**
2. Eval/baseline matrix
3. Add `Qwen3-VL` backend
4. Add constrained generation + scoring
5. Add synthetic warm-start
6. Run small GRPO smoke experiments
7. Write experiment runbook from implemented architecture

---

## 3. Risk Register

All five risks identified. Mitigations confirmed from web research.

### Risk 1 — Invalid JSON from VLM during generation

**Problem:** VLM GRPO training collapses early if the model emits syntactically invalid actions before the environment can score them. Malformed outputs must never silently pass to the reward.

**Mitigation:**
- Use schema-constrained decoding from the first training step
- Implement a `format_reward` that returns a large negative penalty for any unparseable output, applied before visual or structural scores
- Parse every generation in the rollout runner before scoring; log and track `invalid_action_rate` as a first-class metric
- References: `https://docs.vllm.ai/en/latest/features/structured_outputs.html`, `https://huggingface.co/docs/trl/main/en/grpo_trainer`

### Risk 2 — Token-level logprob instability in VLM GRPO

**Problem:** GRPO requires sequence-level log-probabilities from the policy and reference model. VLMs process image tokens before text tokens, which can cause numerical instability or off-by-one errors when slicing logprobs.

**Mitigation:**
- Use `Qwen3-VL-4B-Instruct` (non-thinking variant); avoid `Qwen3-VL-4B-Thinking` for RL initially
- Score only the action text tokens, not image tokens; use processor output to identify exact token span boundaries
- Debug logprob computation in plain `transformers` before introducing `vLLM`; add a unit test that asserts logprob consistency between generation and scoring forward passes
- References: `https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_vl`

### Risk 3 — Reward hacking / judge leakage

**Problem:** If the training judge and evaluation judge are from the same model family (or the same checkpoint), the model can overfit the judge without improving on the actual task.

**Mitigation:**
- Training judge: `SigLIP2 + structural reward`
- Evaluation judge: separately frozen config; loaded in isolation; never shared with training config
- Always report component metrics independently: `reward_vis`, `reward_struct`, `format_penalty`, `invalid_action_rate`
- Save per-rollout JSONL + rendered images for offline re-scoring with a different judge family if needed later
- References: `https://huggingface.co/google/siglip2-so400m-patch14-384`

### Risk 4 — Excessive VRAM / latency during online rollouts

**Problem:** `Qwen3-VL-4B-Instruct` in BF16 uses ~9–12 GB for weights alone; with a reference policy copy, optimizer states, and rollout buffers it can exceed 48 GB.

**Mitigation:**
- LoRA only (no full fine-tuning); target `q_proj, k_proj, v_proj, o_proj` at minimum
- One image per rollout step initially; do not batch rollouts until memory is profiled
- Keep `max_new_tokens` short (e.g. ≤ 256) during initial GRPO runs
- Freeze the vision encoder initially; only fine-tune the LLM text decoder layers
- `flash_attn` and `vLLM` are optimizations for later, not day-1 requirements
- References: `https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct`, `https://github.com/modelscope/ms-swift`

### Risk 5 — Environment / Python version drift

**Problem:** The active shell runs Python 3.13.11. The repo targets 3.11/3.12. The installed `transformers 4.46.1` predates Qwen3-VL support (Qwen3-VL requires latest `transformers` from source or `>=4.51`). `bitsandbytes`, `flash_attn`, and `vllm` are all missing.

**Mitigation:**
- Create a clean Python 3.11 virtual environment before starting implementation
- Install `transformers` from source or pin to a version that includes `Qwen3VLForConditionalGeneration`
- Do not depend on `bitsandbytes` (QLoRA) in the implementation plan; treat it as optional
- Pin the new env to a `requirements-gpu.txt` checked into the repo
- Verify with: `python -c "from transformers import Qwen3VLForConditionalGeneration; print('ok')"` before any other work

---

## 4. Qwen3-VL Findings

**Source:** `https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_vl`, `https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct`

### Model Family

| Model | VRAM (BF16) | A6000 Fit | RL Target |
|---|---|---|---|
| `Qwen3-VL-2B-Instruct` | ~5 GB | Comfortable | Best for RL prototyping |
| `Qwen3-VL-4B-Instruct` | ~9–12 GB | Good | **Recommended default** |
| `Qwen3-VL-8B-Instruct` | ~18–22 GB | Tight (LoRA only) | Later milestone |
| `Qwen3-VL-30B-A3B` | >48 GB | No | Out of scope |
| `Qwen3-VL-72B` | Out of scope | No | Out of scope |

### Key Architecture Facts

- Full multimodal: native image + video + text; no custom projector needed
- `Qwen3VLForConditionalGeneration` is the main class for generation and logprob scoring
- `Qwen3VLProcessor` wraps image processor + tokenizer; always use the same processor for generation and logprob scoring to avoid position ID mismatches
- Supports `logits_to_keep` parameter for memory-efficient logprob computation
- Enhanced MRoPE with interleaved layout; DeepStack integration for multi-level visual features
- Use `apply_chat_template(..., return_dict=True, return_tensors="pt")` for rollout inputs

### HF Transformers Integration

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="sdpa"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
```

### LoRA Target Modules

Use `peft` with `target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` on the LLM text decoder. Freeze vision encoder layers initially.

### Recommended Toolchain

- Development/debugging: `transformers + peft` (already installed pattern)
- Multimodal SFT/RL: `ms-swift` (Qwen-aligned, supports multimodal GRPO)
- Structured serving (later): `vLLM` for constrained JSON decoding
- Custom Python reward logic for GRPO: `trl` `GRPOTrainer` (already installed)

---

## 5. SigLIP2 Findings

**Source:** `https://huggingface.co/google/siglip2-so400m-patch14-384`, `https://huggingface.co/docs/transformers/main/en/model_doc/siglip`

### Why SigLIP2 Over Current CLIP

- SigLIP2 uses pairwise sigmoid loss instead of softmax contrastive; better at small batch sizes and single-image scoring
- `so400m-patch14-384` has 400M params and operates at 384px resolution; better spatial fidelity than ViT-B/32 CLIP
- SigLIP2 adds decoder loss, global-local prediction, and aspect ratio adaptability over SigLIP1
- Designated explicitly as a vision encoder for VLMs — suitable for diagram-quality scoring
- Fully supported in `transformers` via `AutoModel`; no custom code needed

### Usage as a Local Judge

```python
from transformers import AutoProcessor, AutoModel
import torch

model = AutoModel.from_pretrained(
    "google/siglip2-so400m-patch14-384",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"
).eval()
processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch14-384")

# score rendered diagram against concept string
inputs = processor(
    text=[f"a whiteboard diagram showing {concept}"],
    images=[rendered_image],
    padding="max_length",
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    score = torch.sigmoid(outputs.logits_per_image).item()
```

### Judge Architecture Decision

| Judge Role | Model | Notes |
|---|---|---|
| Training online reward | `SigLIP2-so400m-patch14-384` | Always-on, fast |
| Eval offline judge | Frozen `SigLIP2` (separate instance) | Never shares config with training |
| Structural component | `src/spatial_rl/reward/structural.py` | Unchanged initially |
| Format penalty | Custom parse check | Applied before any scoring |

---

## 6. Architecture Plan

### New Modules

| File | Purpose |
|---|---|
| `src/spatial_rl/policy/types.py` | Shared policy input/output types |
| `src/spatial_rl/policy/vlm_qwen3.py` | `Qwen3-VL-4B-Instruct` + LoRA backend |
| `src/spatial_rl/rollout/runner.py` | Shared episode execution (extracted from grpo.py) |
| `src/spatial_rl/rollout/collector.py` | Rollout buffer, JSONL + image artifact saving |
| `src/spatial_rl/eval/runner.py` | Standalone checkpoint evaluation CLI |
| `src/spatial_rl/eval/judges.py` | Local judge committee (SigLIP2 + structural) |
| `src/spatial_rl/eval/baselines.py` | Baseline runners (finish_immediately, random_valid, structured_torch, zero-shot VLM) |
| `src/spatial_rl/train/run_eval.py` | CLI: evaluate a checkpoint on held-out split |
| `src/spatial_rl/train/run_baselines.py` | CLI: run the full baseline matrix |

### Modified Files

| File | Change |
|---|---|
| `src/spatial_rl/policy/factory.py:12` | Add `qwen3_vl_lora` backend |
| `src/spatial_rl/policy/base.py:16` | Extend interface to `generate_text → parse → score` |
| `src/spatial_rl/train/grpo.py:116` | Extract eval loop; delegate to `rollout/runner.py` |
| `src/spatial_rl/reward/visual.py:35` | Add SigLIP2 path alongside existing CLIP |
| `configs/common.yaml:1` | Add `model`, `rollout`, `reward.train_judge`, `eval.judges`, `eval.baselines` sections |
| `configs/profiles/nvidia_24gb.yaml:1` | Rename to `structured_torch_a6000.yaml`; add `qwen3_vl_4b_a6000.yaml` |

### New Config Files

| File | Purpose |
|---|---|
| `configs/profiles/qwen3_vl_4b_a6000.yaml` | Qwen3-VL-4B single-GPU LoRA profile |
| `configs/profiles/qwen3_vl_4b_dev.yaml` | Minimal dev/smoke-test profile |
| `configs/reward/siglip2_judge.yaml` | SigLIP2 judge configuration |
| `configs/reward/committee_judge.yaml` | Committee judge (SigLIP2 + structural) |

### Data Upgrade

| File | Change |
|---|---|
| `data/concepts_train.txt` | Add family labels (e.g. `biology:mitosis`) |
| `data/concepts_val.txt` | Add family labels |
| `data/concepts_test.txt` | New fixed held-out split (never touched during training) |

---

## 7. Phase Plan

### Phase 0 — Runtime Normalization

- Create a clean Python 3.11 virtual environment
- Install `transformers` from source or pinned version with Qwen3-VL support
- Verify `Qwen3VLForConditionalGeneration` imports cleanly
- Verify `SigLIP2` loads and scores a test image
- Freeze a `requirements-gpu.txt`
- All existing tests pass: `pytest tests/`

**Exit criteria:** single-image Qwen3-VL inference works locally; all existing tests green

### Phase 1 — Refactor Without Behavior Change

- Extract shared rollout execution from `src/spatial_rl/train/grpo.py:157` into `src/spatial_rl/rollout/runner.py`
- Extract eval from `src/spatial_rl/train/grpo.py:116` into `src/spatial_rl/eval/runner.py`
- Separate train judge config from eval judge config in `configs/common.yaml`
- No change to reward values, model, or metrics
- All existing tests still pass

**Exit criteria:** `structured_torch` training run produces identical metrics to pre-refactor

### Phase 2 — Standalone Eval + Baseline Matrix + Richer Data

- Implement `src/spatial_rl/train/run_eval.py` CLI
- Implement `src/spatial_rl/eval/baselines.py` with `finish_immediately` and `random_valid`
- Implement `src/spatial_rl/rollout/collector.py` to save per-rollout JSONL and images
- Add family labels to data splits; add `data/concepts_test.txt`
- Add `src/spatial_rl/train/run_baselines.py` CLI
- Run baseline matrix on `structured_torch`; save as `runs/baselines/`

**Exit criteria:** baseline report generated; per-rollout artifacts saved; `structured_torch` eval score matches training log

### Phase 3 — Qwen3-VL Backend

- Implement `src/spatial_rl/policy/types.py`
- Implement `src/spatial_rl/policy/vlm_qwen3.py`:
  - Load `Qwen3-VL-4B-Instruct` with LoRA via `peft`
  - Generate action text using `apply_chat_template` with JSON schema prompt
  - Parse output through existing `src/spatial_rl/parser.py:35`
  - Return format penalty for malformed outputs
  - Compute sequence logprobs for action tokens only
- Register in `src/spatial_rl/policy/factory.py:12`
- Add zero-shot VLM baseline to `src/spatial_rl/eval/baselines.py`
- Add `configs/profiles/qwen3_vl_4b_a6000.yaml`

**Exit criteria:** zero-shot `Qwen3-VL-4B` runs one episode without crash; logprob output matches manual check; invalid action rate logged

### Phase 4 — SigLIP2 Judge + Committee

- Implement SigLIP2 path in `src/spatial_rl/reward/visual.py`
- Implement `src/spatial_rl/eval/judges.py` with committee scoring
- Add `configs/reward/siglip2_judge.yaml`
- Add separate frozen eval judge in `src/spatial_rl/eval/runner.py`
- Compare SigLIP2 vs CLIP scores on 100 saved rollout images

**Exit criteria:** SigLIP2 score correlation with CLIP documented; committee score variance measured

### Phase 5 — GRPO Training Run

- Wire GRPO to `Qwen3-VL-4B-Instruct` backend via `trl` or custom loop
- Add format reward component (applied before visual/structural scores)
- Run synthetic SFT warm-start before GRPO (format compliance only)
- Run smoke GRPO experiment: 50 concepts, 200 updates, single A6000
- Monitor: `reward_total`, `reward_vis`, `reward_struct`, `format_penalty`, `invalid_action_rate`, token length, VRAM usage

**Exit criteria:** GRPO training runs 200 steps without OOM; reward curve shows learning signal; invalid action rate < 20% after warm-start

### Phase 6 — Experiment Ladder + Runbook

- Run full baseline matrix: `finish_immediately`, `random_valid`, `structured_torch`, zero-shot VLM, best-of-n VLM, GRPO-trained VLM
- Run family transfer split: train on seen families, eval on held-out families
- Write experiment runbook from the implemented architecture
- Produce results table: reward, component breakdown, invalid rate, per-family transfer

---

## 8. Experiment Ladder

| Model | Config | Notes |
|---|---|---|
| `finish_immediately` | no training | Lower bound |
| `random_valid` | no training | Lower bound (valid JSON, random content) |
| `structured_torch` trained | current MVP | Control baseline |
| `Qwen3-VL-4B` zero-shot | frozen, no RL | Reference baseline |
| `Qwen3-VL-4B` best-of-n | frozen, n=8 | Search control |
| `Qwen3-VL-4B` GRPO-trained | LoRA, SigLIP2 + struct reward | **Main result** |

### Metrics Per Run

- `reward_total` — composite terminal reward
- `reward_vis` — SigLIP2 visual score
- `reward_struct` — structural reward
- `format_penalty` — fraction of invalid JSON outputs
- `invalid_action_rate` — fraction of unparseable rollout steps
- `avg_steps` — average episode length
- `avg_token_length` — average action token count
- `termination_reason` — histogram: success / max_steps / invalid
- `judge_agreement` — SigLIP2 vs CLIP score correlation (eval only)
- `family_transfer_delta` — reward on held-out families minus seen families

---

## 9. Explicit Non-Recommendations

The following were considered and explicitly rejected:

| Rejected Option | Reason |
|---|---|
| `Qwen3-VL-8B` as first RL target | Too tight on 48 GB with reference policy + optimizer states; `4B` is safer |
| Custom Qwen3 text + separate vision projector | `Qwen3-VL` already exists; building from scratch adds months of risk |
| Same model family for policy and main judge | Reward hacking risk; SigLIP2 is a different family entirely |
| QLoRA / bitsandbytes on day 1 | `bitsandbytes` is not installed; not needed for `4B` in BF16 |
| Start experiments before eval architecture is standalone | Metrics would be meaningless without a fixed held-out judge |
| Multi-GPU as first milestone | Single-device stability is harder than it looks for VLM RL; parallelism is a later optimization |
| API judges (GPT-4o, etc.) | Explicitly requested to keep everything fully local |
| Text-only `Qwen3` | `Qwen3-VL` has native multimodal support; text-only is a step backward |

---

## 10. Next Step

Write the **file-by-file implementation spec** covering each new and modified file listed in Section 6, with:

- Exact class/function signatures
- Interface contracts between modules
- Concrete schema-constrained generation prompt template for `Qwen3-VL`
- Logprob scoring implementation pattern
- LoRA config recommendation
- `requirements-gpu.txt` contents

This spec will be the direct input to implementation work in Phase 1–3.
