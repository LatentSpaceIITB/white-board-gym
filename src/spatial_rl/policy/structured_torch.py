from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from spatial_rl.types import ActionSample, Observation

from spatial_rl.policy.base import BasePolicy, PolicyContext

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical, Normal
except ImportError as exc:
    raise RuntimeError(
        "StructuredTorchPolicy requires torch. Install with: pip install -e .[ml]"
    ) from exc


ACTION_TYPES = ["freedraw", "rect", "ellipse", "text", "arrow", "finish"]
ACTION_TO_INDEX = {name: idx for idx, name in enumerate(ACTION_TYPES)}
NUMERIC_FIELDS = [
    "x",
    "y",
    "w",
    "h",
    "rx",
    "ry",
    "x1",
    "y1",
    "x2",
    "y2",
    "f",
    "stroke_w",
]
FIELD_TO_INDEX = {name: idx for idx, name in enumerate(NUMERIC_FIELDS)}


@dataclass(slots=True)
class StructuredPolicyConfig:
    width: int
    height: int
    hidden_dim: int
    image_encoder_channels: list[int]
    image_feature_dim: int
    concept_vocab_size: int
    concept_embed_dim: int
    concept_feature_dim: int
    color_palette: list[str]
    text_vocab: list[str]
    temperature: float = 1.0


class StructuredTorchPolicy(nn.Module, BasePolicy):
    def __init__(
        self,
        config: StructuredPolicyConfig,
        *,
        device: str = "cpu",
        dtype: Any = torch.float32,
    ):
        super().__init__()
        self.config = config
        self.context = PolicyContext(width=config.width, height=config.height)
        self._device = torch.device(device)
        self._dtype = dtype

        channels = [3, *config.image_encoder_channels]
        conv_layers: list[nn.Module] = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            conv_layers.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
            )
            conv_layers.append(nn.ReLU())

        self.image_encoder = nn.Sequential(
            *conv_layers,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels[-1], config.image_feature_dim),
            nn.ReLU(),
        )

        self.concept_embedding = nn.Embedding(
            config.concept_vocab_size, config.concept_embed_dim
        )
        self.concept_mlp = nn.Sequential(
            nn.Linear(config.concept_embed_dim, config.concept_feature_dim),
            nn.ReLU(),
        )

        fused_in = config.image_feature_dim + config.concept_feature_dim
        self.context_mlp = nn.Sequential(
            nn.Linear(fused_in, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )

        self.action_head = nn.Linear(config.hidden_dim, len(ACTION_TYPES))
        self.numeric_mean_head = nn.Linear(config.hidden_dim, len(NUMERIC_FIELDS))
        self.numeric_scale_head = nn.Linear(config.hidden_dim, len(NUMERIC_FIELDS))
        self.stroke_color_head = nn.Linear(config.hidden_dim, len(config.color_palette))
        self.fill_color_head = nn.Linear(config.hidden_dim, len(config.color_palette))
        self.text_head = nn.Linear(config.hidden_dim, len(config.text_vocab))

        self.to(device=self._device, dtype=self._dtype)

    def sample_action(
        self, observation: Observation, *, temperature: float = 1.0
    ) -> ActionSample:
        self.eval_mode()
        with torch.no_grad():
            context = self._encode_observation(observation)
            action_dist = Categorical(
                logits=self.action_head(context) / max(temperature, 1e-3)
            )
            action_index = int(action_dist.sample().item())
            action_type = ACTION_TYPES[action_index]

            if action_type == "finish":
                action = {"type": "finish"}
                return ActionSample(
                    action=action, action_text=json.dumps(action), valid=True
                )

            numeric = self._sample_numeric_fields(context)
            stroke_color = self._sample_palette_color(
                context, fill=False, temperature=temperature
            )
            fill_color = self._sample_palette_color(
                context, fill=True, temperature=temperature
            )
            text_token = self._sample_text_token(context, temperature=temperature)
            action = self._materialize_action(
                action_type, numeric, stroke_color, fill_color, text_token
            )
            action_text = json.dumps(action, separators=(",", ":"))
            return ActionSample(action=action, action_text=action_text, valid=True)

    def logprob_action(self, observation: Observation, action: Mapping[str, object]):
        context = self._encode_observation(observation)
        action_type = str(action.get("type", "finish"))
        action_index = ACTION_TO_INDEX.get(action_type, ACTION_TO_INDEX["finish"])

        action_logits = self.action_head(context)
        logp = Categorical(logits=action_logits).log_prob(
            torch.tensor(action_index, device=self._device)
        )

        if action_type == "finish":
            return logp

        means, stds = self._numeric_distributions(context)
        for field_name in self._fields_for_action(action_type):
            if field_name not in action:
                continue
            value = float(action[field_name])
            idx = FIELD_TO_INDEX[field_name]
            field_dist = Normal(loc=means[idx], scale=stds[idx])
            logp = logp + field_dist.log_prob(
                torch.tensor(value, dtype=self._dtype, device=self._device)
            )

        if action_type in {"rect", "ellipse"}:
            stroke_color = str(action.get("cs", self.config.color_palette[0]))
        else:
            stroke_color = str(action.get("col", self.config.color_palette[0]))
        stroke_index = self._palette_index(stroke_color)
        stroke_logits = self.stroke_color_head(context)
        logp = logp + Categorical(logits=stroke_logits).log_prob(
            torch.tensor(stroke_index, device=self._device)
        )

        if action_type in {"rect", "ellipse"}:
            fill_color = str(action.get("cf", self.config.color_palette[0]))
            fill_index = self._palette_index(fill_color)
            fill_logits = self.fill_color_head(context)
            logp = logp + Categorical(logits=fill_logits).log_prob(
                torch.tensor(fill_index, device=self._device)
            )

        if action_type == "text":
            label_index = self._text_index(
                str(action.get("s", self.config.text_vocab[0]))
            )
            text_logits = self.text_head(context)
            logp = logp + Categorical(logits=text_logits).log_prob(
                torch.tensor(label_index, device=self._device)
            )

        if action_type == "arrow":
            arrow_label = str(action.get("l", self.config.text_vocab[0]))
            label_index = self._text_index(arrow_label)
            text_logits = self.text_head(context)
            logp = logp + Categorical(logits=text_logits).log_prob(
                torch.tensor(label_index, device=self._device)
            )

        return logp

    def train_mode(self) -> None:
        self.train()

    def eval_mode(self) -> None:
        self.eval()

    def clone_frozen(self) -> BasePolicy:
        clone = copy.deepcopy(self)
        clone.eval_mode()
        for param in clone.parameters():
            param.requires_grad_(False)
        return clone

    def parameters(self):
        return super().parameters()

    def _encode_observation(self, observation: Observation):
        image = torch.from_numpy(observation.image).to(device=self._device)
        image = image.permute(2, 0, 1).to(dtype=self._dtype) / 255.0
        image = image.unsqueeze(0)
        image_feature = self.image_encoder(image).squeeze(0)

        concept_feature = self._encode_concept(observation.concept)
        fused = torch.cat([image_feature, concept_feature], dim=-1)
        return self.context_mlp(fused)

    def _encode_concept(self, concept: str):
        tokens = re.findall(r"[a-zA-Z0-9]+", concept.lower())
        if not tokens:
            tokens = ["concept"]

        indices = [self._stable_token_index(token) for token in tokens]
        token_tensor = torch.tensor(indices, device=self._device, dtype=torch.long)
        embeds = self.concept_embedding(token_tensor)
        pooled = embeds.mean(dim=0)
        return self.concept_mlp(pooled)

    def _stable_token_index(self, token: str) -> int:
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
        value = int(digest[:8], 16)
        return value % self.config.concept_vocab_size

    def _numeric_distributions(self, context):
        raw_mean = self.numeric_mean_head(context)
        raw_scale = self.numeric_scale_head(context)

        means = []
        stds = []
        for field_name in NUMERIC_FIELDS:
            idx = FIELD_TO_INDEX[field_name]
            low, high = self._field_range(field_name)
            span = high - low
            mean = low + span * torch.sigmoid(raw_mean[idx])
            std = span * (0.02 + 0.18 * torch.sigmoid(raw_scale[idx]))
            means.append(mean)
            stds.append(std)

        mean_tensor = torch.stack(means)
        std_tensor = torch.stack(stds)
        return mean_tensor, std_tensor

    def _sample_numeric_fields(self, context) -> dict[str, float]:
        means, stds = self._numeric_distributions(context)
        values: dict[str, float] = {}
        for field_name in NUMERIC_FIELDS:
            idx = FIELD_TO_INDEX[field_name]
            dist = Normal(means[idx], stds[idx])
            sample = float(dist.sample().item())
            low, high = self._field_range(field_name)
            values[field_name] = float(max(low, min(high, sample)))
        return values

    def _sample_palette_color(self, context, *, fill: bool, temperature: float) -> str:
        head = self.fill_color_head if fill else self.stroke_color_head
        logits = head(context) / max(temperature, 1e-3)
        idx = int(Categorical(logits=logits).sample().item())
        return self.config.color_palette[idx]

    def _sample_text_token(self, context, *, temperature: float) -> str:
        logits = self.text_head(context) / max(temperature, 1e-3)
        idx = int(Categorical(logits=logits).sample().item())
        return self.config.text_vocab[idx]

    def _fields_for_action(self, action_type: str) -> list[str]:
        if action_type == "freedraw":
            return ["x", "y", "w", "h", "stroke_w"]
        if action_type == "rect":
            return ["x", "y", "w", "h"]
        if action_type == "ellipse":
            return ["x", "y", "rx", "ry"]
        if action_type == "text":
            return ["x", "y", "f"]
        if action_type == "arrow":
            return ["x1", "y1", "x2", "y2"]
        return []

    def _materialize_action(
        self,
        action_type: str,
        numeric: Mapping[str, float],
        stroke_color: str,
        fill_color: str,
        text_token: str,
    ) -> dict[str, object]:
        if action_type == "freedraw":
            cx = numeric["x"]
            cy = numeric["y"]
            width = numeric["w"]
            height = numeric["h"]
            points = self._free_points(cx, cy, width, height)
            return {
                "type": "freedraw",
                "pts": points,
                "col": stroke_color,
                "w": numeric["stroke_w"],
            }

        if action_type == "rect":
            w = min(numeric["w"], self.context.width - 1)
            h = min(numeric["h"], self.context.height - 1)
            x = max(0.0, min(self.context.width - w, numeric["x"]))
            y = max(0.0, min(self.context.height - h, numeric["y"]))
            return {
                "type": "rect",
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "cs": stroke_color,
                "cf": fill_color,
            }

        if action_type == "ellipse":
            rx = numeric["rx"]
            ry = numeric["ry"]
            x = max(rx, min(self.context.width - rx, numeric["x"]))
            y = max(ry, min(self.context.height - ry, numeric["y"]))
            return {
                "type": "ellipse",
                "x": x,
                "y": y,
                "rx": rx,
                "ry": ry,
                "cs": stroke_color,
                "cf": fill_color,
            }

        if action_type == "text":
            x = max(0.0, min(self.context.width, numeric["x"]))
            y = max(0.0, min(self.context.height, numeric["y"]))
            return {
                "type": "text",
                "s": text_token,
                "x": x,
                "y": y,
                "f": numeric["f"],
                "col": stroke_color,
            }

        if action_type == "arrow":
            x1 = max(0.0, min(self.context.width, numeric["x1"]))
            y1 = max(0.0, min(self.context.height, numeric["y1"]))
            x2 = max(0.0, min(self.context.width, numeric["x2"]))
            y2 = max(0.0, min(self.context.height, numeric["y2"]))
            return {
                "type": "arrow",
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "l": text_token,
                "col": stroke_color,
            }

        return {"type": "finish"}

    def _field_range(self, field_name: str) -> tuple[float, float]:
        width = float(self.context.width)
        height = float(self.context.height)

        if field_name in {"x", "x1", "x2"}:
            return 0.0, width
        if field_name in {"y", "y1", "y2"}:
            return 0.0, height
        if field_name == "w":
            return 6.0, max(8.0, width * 0.75)
        if field_name == "h":
            return 6.0, max(8.0, height * 0.75)
        if field_name == "rx":
            return 4.0, max(6.0, width * 0.35)
        if field_name == "ry":
            return 4.0, max(6.0, height * 0.35)
        if field_name == "f":
            return 8.0, 24.0
        if field_name == "stroke_w":
            return 1.0, 6.0
        return 0.0, 1.0

    def _free_points(
        self, cx: float, cy: float, w: float, h: float
    ) -> list[list[float]]:
        points: list[list[float]] = []
        for i in range(12):
            theta = 2.0 * math.pi * (i / 12.0)
            jitter = 1.0 + 0.08 * math.sin(3.0 * theta)
            x = cx + 0.5 * w * math.cos(theta) * jitter
            y = cy + 0.5 * h * math.sin(theta) * jitter
            x = max(0.0, min(float(self.context.width), x))
            y = max(0.0, min(float(self.context.height), y))
            points.append([x, y])
        return points

    def _palette_index(self, color: str) -> int:
        try:
            return self.config.color_palette.index(color)
        except ValueError:
            return 0

    def _text_index(self, token: str) -> int:
        token_norm = token.strip().lower()
        if token_norm in self.config.text_vocab:
            return self.config.text_vocab.index(token_norm)
        return 0
