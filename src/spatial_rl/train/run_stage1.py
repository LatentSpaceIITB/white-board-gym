from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from spatial_rl.config import build_run_dir, load_experiment_config
from spatial_rl.datasets import load_concepts
from spatial_rl.env import EnvConfig, SpatialDrawingEnv
from spatial_rl.policy.factory import build_policy
from spatial_rl.reward import TerminalReward, TerminalRewardConfig, build_visual_judge
from spatial_rl.train.grpo import EvalConfig, GRPOConfig, GRPOTrainer
from spatial_rl.utils.random import set_global_seed
from spatial_rl.utils.torch_utils import require_torch, resolve_device, resolve_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 1 Spatial RL MVP training")
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Directory containing common/mvp/profile configs",
    )
    parser.add_argument(
        "--profile", type=str, default="apple_silicon_dev", help="Runtime profile name"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Run a tiny smoke training loop"
    )
    return parser.parse_args()


def _resolve_data_path(path: str, root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def main() -> None:
    args = parse_args()
    root = Path.cwd()

    config = load_experiment_config(args.config_dir, args.profile)
    seed = int(config["experiment"]["seed"])
    set_global_seed(seed)

    torch = require_torch()
    requested_device = str(config["train"].get("device", "auto"))
    device = resolve_device(requested_device)

    requested_dtype = str(config["train"].get("dtype", "float32"))
    dtype = resolve_dtype(torch, requested_dtype)
    if device == "mps" and dtype != torch.float32:
        dtype = torch.float32

    env_cfg = config["env"]
    reward_cfg = config["reward"]
    judge_cfg = reward_cfg["visual_judge"]

    try:
        visual_judge = build_visual_judge(judge_cfg, device=device)
    except RuntimeError as exc:
        if str(judge_cfg.get("backend", "heuristic")).lower() == "clip":
            print(
                f"[warning] clip judge unavailable ({exc}); falling back to heuristic"
            )
            fallback_cfg = dict(judge_cfg)
            fallback_cfg["backend"] = "heuristic"
            visual_judge = build_visual_judge(fallback_cfg, device=device)
        else:
            raise

    reward_fn = TerminalReward(
        TerminalRewardConfig(
            alpha=float(reward_cfg["alpha"]),
            beta=float(reward_cfg["beta"]),
            width=int(env_cfg["width"]),
            height=int(env_cfg["height"]),
        ),
        visual_judge=visual_judge,
    )

    env = SpatialDrawingEnv(
        config=EnvConfig(
            width=int(env_cfg["width"]),
            height=int(env_cfg["height"]),
            tmax=int(env_cfg["tmax"]),
            background_color=str(env_cfg.get("background_color", "#ffffff")),
        ),
        reward_fn=reward_fn,
    )

    data_cfg = config["data"]
    train_concepts = load_concepts(
        _resolve_data_path(str(data_cfg["train_concepts_path"]), root)
    )
    val_concepts = load_concepts(
        _resolve_data_path(str(data_cfg["val_concepts_path"]), root)
    )

    policy = build_policy(config["policy"], env_cfg, device=device, dtype=dtype)
    reference_policy = policy.clone_frozen()

    run_dir = build_run_dir(config)
    run_config = dict(config)
    run_config["runtime"] = {
        "profile": args.profile,
        "device": device,
        "dtype": str(dtype),
        "smoke": bool(args.smoke),
    }
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(run_config, sort_keys=False)
    )

    train_cfg = config["train"]
    eval_cfg = config["eval"]
    trainer = GRPOTrainer(
        env=env,
        policy=policy,
        reference_policy=reference_policy,
        train_concepts=train_concepts,
        val_concepts=val_concepts,
        run_dir=run_dir,
        config=GRPOConfig(
            updates=int(train_cfg["updates"]),
            concepts_per_update=int(train_cfg["concepts_per_update"]),
            group_size=int(train_cfg["group_size"]),
            learning_rate=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg["weight_decay"]),
            grad_clip_norm=float(train_cfg["grad_clip_norm"]),
            kl_weight=float(train_cfg["kl_weight"]),
            kl_anneal_start=int(train_cfg["kl_anneal_start"]),
            kl_anneal_end=int(train_cfg["kl_anneal_end"]),
            eval_every=int(train_cfg["eval_every"]),
            checkpoint_every=int(train_cfg["checkpoint_every"]),
            temperature=float(config["policy"].get("default_temperature", 1.0)),
        ),
        eval_config=EvalConfig(
            concepts=int(eval_cfg["concepts"]),
            rollouts_per_concept=int(eval_cfg["rollouts_per_concept"]),
        ),
        seed=seed,
    )

    trainer.train(smoke=args.smoke)
    print(f"Training complete. Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
