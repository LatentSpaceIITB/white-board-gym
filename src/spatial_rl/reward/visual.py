from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class VisualJudge(ABC):
    @abstractmethod
    def score(self, image: np.ndarray, concept: str) -> float:
        raise NotImplementedError


class HeuristicVisualJudge(VisualJudge):
    def score(self, image: np.ndarray, concept: str) -> float:
        gray = image.mean(axis=2)
        ink_ratio = float(np.mean(gray < 245.0))

        dx = np.abs(np.diff(gray, axis=1))
        dy = np.abs(np.diff(gray, axis=0))
        edge_density = float((dx.mean() + dy.mean()) / 255.0)

        concept_tokens = [token for token in concept.lower().split() if token]
        target_ink = min(0.28, 0.08 + 0.012 * len(concept_tokens))
        score_ink = max(0.0, 1.0 - abs(ink_ratio - target_ink) / max(0.08, target_ink))
        score_edge = min(1.0, edge_density * 8.0)

        return float(np.clip(0.65 * score_ink + 0.35 * score_edge, 0.0, 1.0))


class ClipVisualJudge(VisualJudge):
    def __init__(
        self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"
    ):
        try:
            import torch
            from PIL import Image
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(
                "CLIP judge requires torch and transformers. Install with: pip install -e .[ml,clip]"
            ) from exc

        self._torch = torch
        self._Image = Image
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model = CLIPModel.from_pretrained(model_name)
        self._device = device

        if device != "auto":
            self._model.to(device)
        self._model.eval()

    def score(self, image: np.ndarray, concept: str) -> float:
        text = f"diagram of {concept}"
        pil_image = self._Image.fromarray(image)
        inputs = self._processor(
            text=[text], images=[pil_image], return_tensors="pt", padding=True
        )

        for key, value in list(inputs.items()):
            if hasattr(value, "to") and self._device != "auto":
                inputs[key] = value.to(self._device)

        with self._torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits_per_image.squeeze().item()

        # Smooth map from CLIP logit scale to [0, 1].
        score = 1.0 / (1.0 + np.exp(-logits / 10.0))
        return float(np.clip(score, 0.0, 1.0))


class CachedVisualJudge(VisualJudge):
    def __init__(self, base: VisualJudge, cache_path: str | None):
        self._base = base
        self._cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, float] = {}

        if self._cache_path and self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text())
            except json.JSONDecodeError:
                self._cache = {}

    def score(self, image: np.ndarray, concept: str) -> float:
        key = self._hash(image, concept)
        if key in self._cache:
            return self._cache[key]

        score = float(self._base.score(image, concept))
        self._cache[key] = score
        self._flush()
        return score

    def _flush(self) -> None:
        if not self._cache_path:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(self._cache, indent=2, sort_keys=True))

    @staticmethod
    def _hash(image: np.ndarray, concept: str) -> str:
        digest = hashlib.sha256()
        digest.update(image.tobytes())
        digest.update(concept.encode("utf-8"))
        return digest.hexdigest()


def build_visual_judge(
    config: Mapping[str, Any], *, device: str = "cpu"
) -> VisualJudge:
    backend = str(config.get("backend", "heuristic")).lower()
    cache_path = config.get("cache_path")

    if backend == "clip":
        model_name = str(config.get("model_name", "openai/clip-vit-base-patch32"))
        base = ClipVisualJudge(model_name=model_name, device=device)
    else:
        base = HeuristicVisualJudge()

    return CachedVisualJudge(base=base, cache_path=cache_path)
