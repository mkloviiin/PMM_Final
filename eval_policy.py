#!/usr/bin/env python3
"""Unified policy evaluation for trained LeRobot VLA models across MuJoCo, Gazebo, and Real iCub.

Supported Backends:
  - mujoco: Evaluates policy inside MuJoCo physics with TSIK (ik) or TSID (id).
  - gazebo: Evaluates policy against Gazebo Sim via YARP / teleop_module_sm.
  - real: Evaluates policy against the Real iCub robot via YARP / teleop_module_sm.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch

repo_root = Path(__file__).resolve().parent

for plugin in [
    "src/plugins/lerobot-robot-icub",
    "src/plugins/lerobot-teleoperator-icubteleop",
    "src/plugins/lerobot-robot-icubrg",
    "src/plugins/lerobot-teleoperator-icubrgteleop",
]:
    p = repo_root / plugin
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from lerobot.configs.policies import PreTrainedConfig
try:
    from lerobot.utils.feature_utils import build_dataset_frame
except ImportError:
    from lerobot.datasets.feature_utils import build_dataset_frame
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.utils.constants import ACTION, OBS_STR
try:
    from lerobot.common.control_utils import predict_action
except ImportError:
    from lerobot.utils.control_utils import predict_action
from lerobot.utils.device_utils import get_safe_torch_device

from play import _resolve_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Trained LeRobot Policy on iCub")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path or HuggingFace repo ID for the trained policy checkpoint",
    )
    parser.add_argument(
        "--backend",
        choices=["mujoco", "gazebo", "real"],
        default="mujoco",
        help="Evaluation backend",
    )
    parser.add_argument(
        "--controller",
        choices=["ik", "id"],
        default="ik",
        help="Whole-body controller type: 'ik' (TSIK) or 'id' (TSID)",
    )
    parser.add_argument(
        "--impedance",
        action="store_true",
        help="Enable compliant Cartesian impedance mode when --controller id",
    )
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--episode-len", type=int, default=600, help="Max steps per episode")
    parser.add_argument("--task", default=None, help="Language task instruction prompt")
    parser.add_argument("--scene", default=None, help="Scene index or label (e.g. 1 for lift)")
    parser.add_argument("--control-arms", choices=["right", "left", "both"], default="right", help="Active arms")
    parser.add_argument("--robot-name", choices=["icub", "icubSim"], default=None, help="YARP robot name")
    parser.add_argument("--show-viewer", action="store_true", default=True, help="Display live camera window")
    parser.add_argument("--device", default="cuda", help="Torch device: cuda or cpu")
    parser.add_argument("--config", default=str(repo_root / "config" / "control_config.yaml"), help="Config path")
    return parser.parse_args()


def load_dataset_features(checkpoint_path: Path) -> dict | None:
    train_cfg_path = checkpoint_path / "train_config.json"
    if train_cfg_path.exists():
        try:
            with open(train_cfg_path, "r") as f:
                train_cfg = json.load(f)
            ds_root = train_cfg.get("dataset", {}).get("root", "")
            info_path = Path(ds_root) / "meta" / "info.json"
            if info_path.exists():
                with open(info_path, "r") as f:
                    return json.load(f).get("features", None)
        except Exception:
            pass
    return None


def main():
    args = parse_args()
    device = get_safe_torch_device(args.device)

    chk_path = Path(args.checkpoint)
    if chk_path.is_dir():
        cfg_dir = chk_path
    else:
        cfg_dir = chk_path

    policy_cfg = PreTrainedConfig.from_pretrained(str(cfg_dir))
    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(str(cfg_dir), config=policy_cfg)
    policy.to(device)
    policy.eval()

    model_path, default_task, scene_objects, scene_joints = _resolve_scene(args.scene, controller=args.controller)
    task_desc = args.task or default_task

    if args.backend == "mujoco":
        from lerobot_robot_icub.config_icub_mujoco import iCubMuJoCoConfig
        from lerobot_robot_icub.icub_mujoco import iCubMuJoCo

        robot_cfg = iCubMuJoCoConfig(
            model_path=str(model_path),
            config_path=str(args.config),
            control_arms=args.control_arms,
            controller_type=args.controller,
            impedance_mode=args.impedance,
            vr_enabled=False,
            scene_objects=scene_objects,
            scene_joints=scene_joints,
        )
        robot = iCubMuJoCo(robot_cfg)
    else:
        from lerobot_robot_icubrg.config_icub import iCubConfig
        from lerobot_robot_icubrg.icub import iCub

        robot_name = args.robot_name or ("icub" if args.backend == "real" else "icubSim")
        os.environ["ICUB_ROBOT_NAME"] = robot_name

        robot_cfg = iCubConfig(
            config_path=str(args.config),
            control_arms=args.control_arms,
        )
        robot = iCub(robot_cfg)

    print(f"\n=======================================================")
    print(f" Starting Policy Evaluation")
    print(f" Backend:    {args.backend.upper()}")
    print(f" Controller: {args.controller.upper()} {'(Impedance)' if args.impedance else ''}")
    print(f" Task:       {task_desc}")
    print(f" Checkpoint: {args.checkpoint}")
    print(f" Device:     {device}")
    print(f"=======================================================\n")

    robot.connect()

    ds_features = load_dataset_features(chk_path) if chk_path.is_dir() else None
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        dataset_features=ds_features,
    )

    try:
        for ep in range(args.num_episodes):
            print(f"\n--- Episode {ep + 1}/{args.num_episodes} ---")
            policy.reset()

            if args.backend == "mujoco":
                robot.teleop_core._snap_targets_to_ee()
            else:
                robot.reset_command_state()

            obs = robot.get_observation()
            dt = robot_cfg.control_dt if hasattr(robot_cfg, "control_dt") else 1.0 / 30.0

            for step in range(args.episode_len):
                t0 = time.perf_counter()

                frame = {
                    OBS_STR: obs,
                    "task": task_desc,
                }

                if preprocessor is not None:
                    batch = preprocessor(frame)
                else:
                    batch = frame

                batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                with torch.inference_mode():
                    action_tensor = policy.select_action(batch)

                if postprocessor is not None:
                    action_dict = postprocessor(action_tensor)
                else:
                    action_dict = make_robot_action(action_tensor, robot.action_features)

                robot.send_action(action_dict)
                obs = robot.get_observation()

                if args.show_viewer:
                    cam_img = obs.get("front") or obs.get("left")
                    if cam_img is not None:
                        bgr = cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR)
                        cv2.imshow("iCub Policy Eval", bgr)
                        if cv2.waitKey(1) & 0xFF == 27:
                            print("[Eval] ESC pressed -> Stopping.")
                            return

                elapsed = time.perf_counter() - t0
                sleep_s = dt - elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)

            print(f"Episode {ep + 1} finished.")

    finally:
        robot.disconnect()
        if args.show_viewer:
            cv2.destroyAllWindows()
        print("[Eval] Completed.")


if __name__ == "__main__":
    main()

