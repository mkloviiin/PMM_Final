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
        default=str(repo_root / "outputs" / "train"),
        help="Directory to save checkpoints and logs",
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

    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.train",
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--dataset.root={args.dataset_root}",
        f"--policy.type={args.policy_type}",
        f"--output_dir={args.output_dir}",
        f"--training.batch_size={args.batch_size}",
        f"--training.steps={args.training_steps}",
        f"--training.save_checkpoint_interval={args.save_freq}",
        f"--training.log_freq={args.log_freq}",
        f"--device={args.device}",
        f"--wandb.enable={'true' if args.wandb else 'false'}",
    ]

    print("\n=======================================================")
    print(f" Starting LeRobot Policy Training")
    print(f" Policy:    {args.policy_type.upper()}")
    print(f" Dataset:   {args.dataset_repo_id} (root: {args.dataset_root})")
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

