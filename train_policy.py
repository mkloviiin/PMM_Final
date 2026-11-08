#!/usr/bin/env python3
"""Unified training script for LeRobot policies (SmolVLA, ACT, Diffusion) on recorded iCub datasets.

Supports:
  - SmolVLA (VLA policy with vision, proprioception, and language task conditioning)
  - ACT (Action Chunking with Transformers)
  - Diffusion Policy
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

repo_root = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LeRobot Policy on iCub Datasets")
    parser.add_argument(
        "--dataset-repo-id",
        required=True,
        help="Dataset identifier / path used during recording (e.g. local/icub_demo)",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(repo_root / "data"),
        help="Root directory where recorded datasets are stored",
    )
    parser.add_argument(
        "--policy-type",
        choices=["smolvla", "act", "diffusion"],
        default="smolvla",
        help="Policy architecture to train (default: smolvla)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Run directory for checkpoints and logs (must not exist; default: output/train/<dataset>_<policy>)",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--training-steps", type=int, default=20000, help="Total training steps")
    parser.add_argument("--save-freq", type=int, default=5000, help="Checkpoint save frequency")
    parser.add_argument("--log-freq", type=int, default=100, help="Logging step frequency")
    parser.add_argument("--device", default="cuda", help="Training device: cuda or cpu")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Strictly verify presence of meta/control_config.json before training
    ds_root = Path(args.dataset_root).expanduser().resolve()
    clean_id = args.dataset_repo_id.replace("local/", "").replace("/", "_")
    candidates = [
        Path(args.dataset_repo_id).expanduser().resolve(),
        ds_root / args.dataset_repo_id,
        ds_root / args.dataset_repo_id.replace("/", "_"),
        ds_root / clean_id,
    ]
    candidates.extend(ds_root.glob(f"{args.dataset_repo_id.replace('/', '_')}_*"))
    candidates.extend(ds_root.glob(f"{clean_id}_*"))
    candidates.extend(ds_root.glob(f"local_{clean_id}_*"))

    found_dataset_dir: Path | None = None
    for cand in candidates:
        if cand.is_dir() and (cand / "meta" / "info.json").is_file():
            found_dataset_dir = cand
            break

    if found_dataset_dir is None:
        print(f"[train][error] Dataset '{args.dataset_repo_id}' not found under '{args.dataset_root}'.", file=sys.stderr)
        sys.exit(1)

    control_config_file = found_dataset_dir / "meta" / "control_config.json"
    if not control_config_file.is_file():
        print(
            f"\n[train][FATAL ERROR] Dataset at '{found_dataset_dir}' is missing required 'meta/control_config.json'.\n"
            "All datasets must be characterized with controller and solver parameters before training.\n"
            "Training is cancelled.\n",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    print(f"[train] Verified dataset control metadata: {control_config_file}")

    if args.output_dir is None:
        base = repo_root / "output" / "train" / f"{found_dataset_dir.name}_{args.policy_type}"
        out, n = base, 2
        while out.exists():  # lerobot aborta si output_dir ya existe (salvo --resume)
            out, n = base.with_name(f"{base.name}_{n}"), n + 1
        args.output_dir = str(out)

    # SmolVLA: igual que el tab Train del Hub. Por defecto lerobot recorta el estado a 32 dims y
    # no carga los pesos del VLM.
    policy_extra: list[str] = []
    if args.policy_type == "smolvla":
        import json
        feats = json.loads((found_dataset_dir / "meta" / "info.json").read_text())["features"]
        policy_extra = [
            "--policy.load_vlm_weights=true",
            f"--policy.max_state_dim={max(32, feats['observation.state']['shape'][0])}",
            f"--policy.max_action_dim={max(32, feats['action']['shape'][0])}",
        ]

    # lerobot-train sin cambios. eval_policy.py lee los metadatos de control desde el dataset
    # (dataset.root en train_config.json de cada checkpoint).
    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id=local/{found_dataset_dir.name}",
        f"--dataset.root={found_dataset_dir}",
        f"--policy.type={args.policy_type}",
        f"--policy.device={args.device}",
        "--policy.push_to_hub=false",
        *policy_extra,
        f"--output_dir={args.output_dir}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.training_steps}",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        f"--wandb.enable={'true' if args.wandb else 'false'}",
    ]

    print("\n=======================================================")
    print(f" Starting LeRobot Policy Training")
    print(f" Policy:    {args.policy_type.upper()}")
    print(f" Dataset:   {found_dataset_dir}")
    print(f" Steps:     {args.training_steps} (batch size: {args.batch_size})")
    print(f" Output:    {args.output_dir}")
    print(f" Command:   {' '.join(cmd)}")
    print("=======================================================\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{str(repo_root)}:{env.get('PYTHONPATH', '')}"

    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

