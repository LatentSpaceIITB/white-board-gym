from __future__ import annotations

from typing import Any, Mapping

from spatial_rl.policy.base import BasePolicy
from spatial_rl.policy.structured_torch import (
    StructuredPolicyConfig,
    StructuredTorchPolicy,
)


def build_policy(
    policy_cfg: Mapping[str, Any], env_cfg: Mapping[str, Any], *, device: str, dtype
) -> BasePolicy:
    backend = str(policy_cfg.get("backend", "structured_torch")).lower()
    if backend != "structured_torch":
        raise ValueError(f"unsupported policy backend: {backend}")

    text_vocab = [
        str(token).strip().lower()
        for token in policy_cfg.get("text_vocab", [])
        if str(token).strip()
    ]
    if not text_vocab:
        text_vocab = ["label", "part", "input", "output", "process"]

    color_palette = [
        str(color).strip().lower()
        for color in policy_cfg.get("color_palette", [])
        if str(color).strip()
    ]
    if not color_palette:
        color_palette = ["#1f2933", "#0ea5e9", "#22c55e", "#ef4444"]

    config = StructuredPolicyConfig(
        width=int(env_cfg["width"]),
        height=int(env_cfg["height"]),
        hidden_dim=int(policy_cfg.get("hidden_dim", 256)),
        image_encoder_channels=[
            int(v) for v in policy_cfg.get("image_encoder_channels", [16, 32, 64])
        ],
        image_feature_dim=int(policy_cfg.get("image_feature_dim", 128)),
        concept_vocab_size=int(policy_cfg.get("concept_vocab_size", 4096)),
        concept_embed_dim=int(policy_cfg.get("concept_embed_dim", 64)),
        concept_feature_dim=int(policy_cfg.get("concept_feature_dim", 128)),
        color_palette=color_palette,
        text_vocab=text_vocab,
        temperature=float(policy_cfg.get("default_temperature", 1.0)),
    )
    return StructuredTorchPolicy(config=config, device=device, dtype=dtype)
