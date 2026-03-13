from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image
from tqdm import tqdm

from spatial_rl.env import SpatialDrawingEnv
from spatial_rl.policy.base import BasePolicy
from spatial_rl.types import Rollout, TrajectoryStep
from spatial_rl.utils.torch_utils import require_torch


@dataclass(slots=True)
class GRPOConfig:
    updates: int
    concepts_per_update: int
    group_size: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    kl_weight: float
    kl_anneal_start: int
    kl_anneal_end: int
    eval_every: int
    checkpoint_every: int
    temperature: float


@dataclass(slots=True)
class EvalConfig:
    concepts: int
    rollouts_per_concept: int


class GRPOTrainer:
    def __init__(
        self,
        *,
        env: SpatialDrawingEnv,
        policy: BasePolicy,
        reference_policy: BasePolicy,
        train_concepts: list[str],
        val_concepts: list[str],
        run_dir: Path,
        config: GRPOConfig,
        eval_config: EvalConfig,
        seed: int,
    ):
        self.env = env
        self.policy = policy
        self.reference_policy = reference_policy
        self.train_concepts = train_concepts
        self.val_concepts = val_concepts
        self.run_dir = run_dir
        self.config = config
        self.eval_config = eval_config
        self.random = random.Random(seed)

        torch = require_torch()
        self._torch = torch
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.metrics_path = run_dir / "metrics.jsonl"
        self.samples_dir = run_dir / "samples"
        self.checkpoint_dir = run_dir / "checkpoints"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self, *, smoke: bool = False) -> None:
        updates = min(self.config.updates, 6) if smoke else self.config.updates

        progress = tqdm(range(1, updates + 1), desc="GRPO")
        for update_index in progress:
            rollout_groups = self._collect_rollout_groups(update_index)
            loss, train_metrics = self._optimize_step(rollout_groups, update_index)

            metrics = {
                "update": update_index,
                "loss": float(loss),
                **train_metrics,
            }

            if (
                update_index % self.config.eval_every == 0
                or update_index == 1
                or update_index == updates
            ):
                eval_metrics = self.evaluate(update_index)
                metrics.update(eval_metrics)

            self._append_metrics(metrics)
            progress.set_postfix(
                {
                    "loss": f"{metrics['loss']:.4f}",
                    "train_reward": f"{metrics['train/reward_total']:.3f}",
                    "kl": f"{metrics['train/kl']:.3f}",
                }
            )

            if (
                update_index % self.config.checkpoint_every == 0
                or update_index == updates
            ):
                self._save_checkpoint(update_index)

    def evaluate(self, update_index: int) -> dict[str, float]:
        self.policy.eval_mode()

        concepts = self.val_concepts[
            : min(len(self.val_concepts), self.eval_config.concepts)
        ]
        rollouts: list[Rollout] = []
        for concept in concepts:
            for _ in range(self.eval_config.rollouts_per_concept):
                rollout = self._run_rollout(concept, temperature=0.75)
                rollouts.append(rollout)

        if not rollouts:
            return {}

        reward_total = float(np.mean([r.reward_total for r in rollouts]))
        reward_vis = float(np.mean([r.reward_vis for r in rollouts]))
        reward_struct = float(np.mean([r.reward_struct for r in rollouts]))
        avg_steps = float(np.mean([len(r.steps) for r in rollouts]))

        best_rollout = max(rollouts, key=lambda rollout: rollout.reward_total)
        image_path = self.samples_dir / f"eval_update_{update_index:04d}.png"
        Image.fromarray(best_rollout.terminal_image).save(image_path)

        return {
            "eval/reward_total": reward_total,
            "eval/reward_vis": reward_vis,
            "eval/reward_struct": reward_struct,
            "eval/avg_steps": avg_steps,
        }

    def _collect_rollout_groups(self, update_index: int) -> dict[str, list[Rollout]]:
        concepts = self._sample_train_concepts(self.config.concepts_per_update)
        grouped: dict[str, list[Rollout]] = {}
        for concept in concepts:
            grouped[concept] = [
                self._run_rollout(concept, temperature=self.config.temperature)
                for _ in range(self.config.group_size)
            ]
        return grouped

    def _run_rollout(self, concept: str, *, temperature: float) -> Rollout:
        observation = self.env.reset(concept)
        rollout = Rollout(concept=concept)

        done = False
        while not done:
            sample = self.policy.sample_action(observation, temperature=temperature)
            step_result = self.env.step(sample.action)
            rollout.steps.append(
                TrajectoryStep(
                    observation=observation,
                    action=sample.action,
                    action_text=sample.action_text,
                    valid_action=bool(step_result.info.get("valid_action", True)),
                )
            )
            if not step_result.info.get("valid_action", True):
                rollout.invalid_actions += 1

            done = step_result.terminated or step_result.truncated
            observation = step_result.observation

            if done:
                rollout.reward_total = float(step_result.reward)
                rollout.reward_vis = float(step_result.info.get("reward_vis", 0.0))
                rollout.reward_struct = float(
                    step_result.info.get("reward_struct", 0.0)
                )
                rollout.terminal_image = observation.image.copy()
                rollout.terminated_reason = str(
                    step_result.info.get("termination_reason", "unknown")
                )

        return rollout

    def _optimize_step(
        self, rollout_groups: Mapping[str, list[Rollout]], update_index: int
    ) -> tuple[float, dict[str, float]]:
        torch = self._torch
        self.policy.train_mode()

        loss_terms = []
        kl_terms = []
        pg_terms = []
        groups_count = 0

        reward_values: list[float] = []
        reward_vis_values: list[float] = []
        reward_struct_values: list[float] = []
        steps_values: list[int] = []
        invalid_values: list[int] = []

        for _, rollouts in rollout_groups.items():
            if not rollouts:
                continue

            rewards = torch.tensor(
                [rollout.reward_total for rollout in rollouts], dtype=torch.float32
            )
            advantages = (rewards - rewards.mean()) / (
                rewards.std(unbiased=False) + 1e-8
            )

            group_pg_terms = []
            group_kl_terms = []
            valid_rollout_count = 0

            for idx, rollout in enumerate(rollouts):
                if not rollout.steps:
                    continue

                adv = advantages[idx]
                step_logps = []
                step_kls = []

                for step in rollout.steps:
                    logp = self.policy.logprob_action(step.observation, step.action)
                    with torch.no_grad():
                        ref_logp = self.reference_policy.logprob_action(
                            step.observation, step.action
                        )

                    step_logps.append(logp)
                    step_kls.append(logp - ref_logp)

                if not step_logps:
                    continue

                logp_mean = torch.stack(step_logps).mean()
                kl_mean = torch.stack(step_kls).mean()
                adv = adv.to(device=logp_mean.device, dtype=logp_mean.dtype)

                group_pg_terms.append(-adv * logp_mean)
                group_kl_terms.append(kl_mean)
                valid_rollout_count += 1

                reward_values.append(rollout.reward_total)
                reward_vis_values.append(rollout.reward_vis)
                reward_struct_values.append(rollout.reward_struct)
                steps_values.append(len(rollout.steps))
                invalid_values.append(rollout.invalid_actions)

            if valid_rollout_count == 0:
                continue

            group_pg = torch.stack(group_pg_terms).mean()
            group_kl = torch.stack(group_kl_terms).mean()

            current_kl = self._annealed_kl_weight(update_index)
            group_loss = group_pg + current_kl * group_kl

            pg_terms.append(group_pg)
            kl_terms.append(group_kl)
            loss_terms.append(group_loss)
            groups_count += 1

        if groups_count == 0:
            return 0.0, {
                "train/reward_total": 0.0,
                "train/reward_vis": 0.0,
                "train/reward_struct": 0.0,
                "train/avg_steps": 0.0,
                "train/invalid_actions": 0.0,
                "train/kl": 0.0,
            }

        loss = torch.stack(loss_terms).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.parameters()), self.config.grad_clip_norm
        )
        self.optimizer.step()

        metrics = {
            "train/reward_total": float(np.mean(reward_values))
            if reward_values
            else 0.0,
            "train/reward_vis": float(np.mean(reward_vis_values))
            if reward_vis_values
            else 0.0,
            "train/reward_struct": float(np.mean(reward_struct_values))
            if reward_struct_values
            else 0.0,
            "train/avg_steps": float(np.mean(steps_values)) if steps_values else 0.0,
            "train/invalid_actions": float(np.mean(invalid_values))
            if invalid_values
            else 0.0,
            "train/kl": float(torch.stack(kl_terms).mean().item()),
            "train/pg": float(torch.stack(pg_terms).mean().item()),
            "train/kl_weight": float(self._annealed_kl_weight(update_index)),
        }
        return float(loss.item()), metrics

    def _annealed_kl_weight(self, update_index: int) -> float:
        start = self.config.kl_anneal_start
        end = self.config.kl_anneal_end
        base = self.config.kl_weight

        if update_index <= start:
            return base
        if update_index >= end:
            return 0.0
        span = max(1, end - start)
        frac = (update_index - start) / span
        return base * (1.0 - frac)

    def _append_metrics(self, metrics: Mapping[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(metrics), sort_keys=True) + "\n")

    def _save_checkpoint(self, update_index: int) -> None:
        torch = self._torch
        ckpt_path = self.checkpoint_dir / f"policy_update_{update_index:04d}.pt"
        payload = {
            "update": update_index,
            "policy_state_dict": self.policy.state_dict(),
        }
        torch.save(payload, ckpt_path)

    def _sample_train_concepts(self, count: int) -> list[str]:
        if count >= len(self.train_concepts):
            return self.train_concepts.copy()
        return self.random.sample(self.train_concepts, count)
