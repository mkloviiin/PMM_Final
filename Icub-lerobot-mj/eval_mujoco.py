#!/usr/bin/env python3
"""
eval_mujoco.py – Run a trained SmolVLA policy on the iCub MuJoCo simulation.

The robot acts autonomously based on visual + state observations fed to SmolVLA.

Usage (lerobotenv, from /home/icub/mujoco_ws/REPO_ICUB/icub-lerobot/):
    conda activate lerobotenv
    python eval_mujoco.py --policy-repo mklovin/my_smolvla

Optional flags:
    --task "Pick up the blue cube"   Task instruction for the policy
    --n-episodes 5                   Number of evaluation episodes
    --episode-steps 300              Max steps per episode (at --fps Hz)
    --fps 30
    --scene 1                        1=lift, 2=stack
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

# ── Inject local robot / teleop plugins ─────────────────────────────────────
repo_root = Path(__file__).resolve().parent

local_robot_path = repo_root / "lerobot-robot-icub"
if local_robot_path.exists():
    sys.path.insert(0, str(local_robot_path))

local_teleop_path = repo_root / "lerobot-teleoperator-icubteleop"
if local_teleop_path.exists():
    sys.path.insert(0, str(local_teleop_path))

# Also expose icubenv packages (MuJoCo bindings, dependencies, etc.)
_icubenv_sp = os.path.expanduser(
    "~/miniconda3/envs/icubenv/lib/python3.12/site-packages"
)
if _icubenv_sp not in sys.path:
    sys.path.append(_icubenv_sp)


# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained SmolVLA policy on the iCub MuJoCo simulation"
    )
    p.add_argument(
        "--policy-repo",
        default="mklovin/my_smolvla",
        help="HF Hub repo ID or local path to the trained checkpoint directory",
    )
    p.add_argument(
        "--task",
        default="Pick up the blue cube",
        help="Task instruction fed to the language-conditioned policy",
    )
    p.add_argument("--n-episodes", type=int, default=5, help="Evaluation episodes")
    p.add_argument(
        "--episode-steps",
        type=int,
        default=300,
        help="Maximum steps per episode at --fps Hz",
    )
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--scene",
        type=int,
        default=1,
        choices=[1, 2],
        help="1=lift cube, 2=stack cubes",
    )
    p.add_argument(
        "--config",
        default=str(repo_root / "utils" / "control_config.yaml"),
        help="Path to control_config.yaml",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for policy inference",
    )
    p.add_argument(
        "--lift-threshold",
        type=float,
        default=0.03,
        help="Lift success threshold in meters (max_cube_z - initial_cube_z)",
    )
    p.add_argument(
        "--min-hand-z",
        type=float,
        default=0.55,
        help="Minimum z for rh/lh mocap targets during eval",
    )
    p.add_argument(
        "--min-gaze-z",
        type=float,
        default=0.55,
        help="Minimum z for gaze mocap target during eval",
    )
    return p.parse_args()


# ── Observation → policy batch ───────────────────────────────────────────────

CAMERA_NAMES = ("head_cam", "head_cam_track_hand", "frontview")

FULL_ACTION_ORDER = [
    "rh_pos_x", "rh_pos_y", "rh_pos_z",
    "rh_quat_w", "rh_quat_x", "rh_quat_y", "rh_quat_z",
    "rh_gripper",
    "lh_pos_x", "lh_pos_y", "lh_pos_z",
    "lh_quat_w", "lh_quat_x", "lh_quat_y", "lh_quat_z",
    "lh_gripper",
    "gaze_x", "gaze_y", "gaze_z",
]


def build_raw_observation_dict(obs: dict, device: str) -> dict:
    """
    Convert the raw iCub-MuJoCo observation dict into exactly the state
    and image formats expected by the PreTrained Policy inputs, before 
    any normalizations or tokenization strings are handled by the preprocessor.
    """
    batch: dict = {}
    state_vals: list[float] = []

    for key, val in obs.items():
        if key in CAMERA_NAMES:
            # val is a (C, H, W) torch.Tensor with values in [0, 255]
            # Must remain on CPU for raw formatting; preprocessor moves it to target device
            img = val.float() / 255.0           # expected to be in [0, 1] usually
            batch[f"observation.images.{key}"] = img.unsqueeze(0).to(device)
        else:
            # Scalar observation feature (joint pos, vel, torque, EE pose, …)
            if isinstance(val, torch.Tensor):
                state_vals.append(val.item())
            else:
                state_vals.append(float(val))

    batch["observation.state"] = torch.tensor(
        state_vals, dtype=torch.float32
    ).unsqueeze(0).to(device)

    return batch


def to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def read_expected_dims_from_train_config(policy_repo: str) -> tuple[int | None, int | None]:
    policy_path = Path(policy_repo).expanduser()
    train_cfg = policy_path / "train_config.json"
    if not train_cfg.exists():
        return None, None

    try:
        cfg = json.loads(train_cfg.read_text())
        expected_state = cfg["policy"]["input_features"]["observation.state"]["shape"][0]
        expected_action = cfg["policy"]["output_features"]["action"]["shape"][0]
        return int(expected_state), int(expected_action)
    except Exception:
        return None, None


def actions_tensor_to_dict(actions: torch.Tensor, action_keys: list[str]) -> dict[str, float]:
    """
    Convert the 1-D action tensor returned by `policy.select_action`
    into the dict format expected by `robot.send_action`.
    """
    a = actions.detach().cpu().numpy().flatten()

    if len(a) == len(FULL_ACTION_ORDER):
        full_action = {name: float(a[idx]) for idx, name in enumerate(FULL_ACTION_ORDER)}
        return {key: full_action.get(key, 0.0) for key in action_keys}

    return {key: float(a[i]) for i, key in enumerate(action_keys) if i < len(a)}


def clamp_action_targets(action_dict: dict[str, float], min_hand_z: float, min_gaze_z: float) -> bool:
    clamped = False

    for key in ("rh_pos_z", "lh_pos_z"):
        if key in action_dict and action_dict[key] < min_hand_z:
            action_dict[key] = min_hand_z
            clamped = True

    if "gaze_z" in action_dict and action_dict["gaze_z"] < min_gaze_z:
        action_dict["gaze_z"] = min_gaze_z
        clamped = True

    return clamped


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Pick the right scene XML
    if args.scene == 1:
        model_file = (
            repo_root
            / "dependencies"
            / "assets"
            / "scenes"
            / "icub_table_scene_lift.xml"
        )
        task = args.task or "Pick up the blue cube"
    else:
        model_file = (
            repo_root
            / "dependencies"
            / "assets"
            / "scenes"
            / "icub_table_scene_stack.xml"
        )
        task = args.task or "Stack the blue cube"

    # Set env vars used by the robot plugin
    os.environ.setdefault("ICUB_LEROBOT_CONFIG", args.config)
    os.environ.setdefault("ICUB_MUJOCO_MODEL_PATH", str(model_file))
    os.environ.setdefault("ICUB_WORKSPACE_ROOT", str(repo_root.parent))
    os.environ["ICUB_MUJOCO_VR_ENABLED"] = "0"

    # ── Load policy ──────────────────────────────────────────────────────────
    print(f"[eval] Loading SmolVLA policy from: {args.policy_repo!r}")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy: SmolVLAPolicy = SmolVLAPolicy.from_pretrained(args.policy_repo)
    policy.eval()
    policy.to(args.device)
    
    # Preprocessor and Postprocessor for data normalization (MEAN_STD)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.policy_repo,
    )
    
    print(f"[eval] Policy and processors loaded on {args.device} ✓")

    # ── Connect robot ────────────────────────────────────────────────────────
    from lerobot_robot_icub.config_icub_mujoco import iCubMuJoCoConfig
    from lerobot.robots import make_robot_from_config
    from lerobot.utils.robot_utils import precise_sleep

    robot_cfg = iCubMuJoCoConfig(
        model_path=str(model_file),
        config_path=args.config,
    )
    robot = make_robot_from_config(robot_cfg)
    robot.connect()
    print("[eval] Robot connected ✓")
    print(f"[eval] Task: {task!r}")
    print(f"[eval] Running {args.n_episodes} episodes × {args.episode_steps} steps @ {args.fps} fps\n")

    action_keys = list(robot.action_features.keys())

    expected_state_dim, expected_action_dim = read_expected_dims_from_train_config(args.policy_repo)
    obs_probe = robot.get_observation()
    current_state_dim = sum(1 for key in obs_probe if key not in CAMERA_NAMES)

    if expected_state_dim is not None and expected_state_dim != current_state_dim:
        print(
            "[eval][warn] State dim mismatch: "
            f"trained={expected_state_dim}, eval={current_state_dim}. "
            "This can cause out-of-distribution actions (e.g., mocaps below scenario)."
        )

    if expected_action_dim is not None and expected_action_dim != len(FULL_ACTION_ORDER):
        print(
            "[eval][warn] Unexpected trained action dim: "
            f"trained={expected_action_dim}, expected_full={len(FULL_ACTION_ORDER)}."
        )

    # Open the MuJoCo passive viewer (same window you see when recording)
    tc = robot.teleop_core
    mj_viewer = mujoco.viewer.launch_passive(tc.model, tc.data)
    mj_viewer.cam.azimuth = 135
    mj_viewer.cam.elevation = -20
    mj_viewer.cam.distance = 2.0
    print("[eval] MuJoCo viewer opened ✓")

    # ── Evaluation loop ───────────────────────────────────────────────────────
    episode_metrics: list[dict] = []

    try:
        for ep in range(args.n_episodes):
            print(f"── Episode {ep + 1}/{args.n_episodes} ──")

            # Reset scenario: place the cube
            if hasattr(robot, "teleop_core") and robot.teleop_core:
                tc = robot.teleop_core
                if hasattr(tc, "_reset_scenario"):
                    tc._reset_scenario()
                
                # Set fixed position
                pos = np.array([0.40, -0.20, 0.72])
                quat = np.array([1.0, 0.0, 0.0, 0.0])
                if hasattr(tc, "set_cube_pose"):
                    tc.set_cube_pose(pos, quat)
                print(f"[Reset] Fixed blue-cube → {pos}")

            # Reset policy hidden state / action chunk
            policy.reset()

            ep_initial_cube_z: float | None = None
            ep_max_cube_z: float | None = None
            ep_touch_frames = 0
            ep_clamp_steps = 0

            for step in range(args.episode_steps):
                t0 = time.perf_counter()

                # 1. Get observation from the simulator
                obs = robot.get_observation()

                if "object_pos_z" in obs:
                    cube_z = to_float(obs["object_pos_z"])
                    if ep_initial_cube_z is None:
                        ep_initial_cube_z = cube_z
                    if ep_max_cube_z is None:
                        ep_max_cube_z = cube_z
                    else:
                        ep_max_cube_z = max(ep_max_cube_z, cube_z)

                touched = False
                for touch_key in ("rh_touch_thumb", "rh_touch_index", "rh_touch_middle"):
                    if touch_key in obs and to_float(obs[touch_key]) > 0.5:
                        touched = True
                        break
                if touched:
                    ep_touch_frames += 1

                # 2. Build the policy input batch
                raw_obs = build_raw_observation_dict(obs, args.device)

                # Preprocessor loaded from checkpoint expects flat batch format
                # (observation.* keys + complementary fields like "task").
                policy_input = {**raw_obs, "task": task}

                # Apply normalization and tokenization
                batch = preprocessor(policy_input)

                # 3. Run policy inference
                with torch.no_grad():
                    action_tensor = policy.select_action(batch)
                
                # 4. Unnormalize action to get simulator metric coordinates
                action_tensor = postprocessor(action_tensor)

                # 5. Convert tensor → dict and send to robot
                action_dict = actions_tensor_to_dict(action_tensor, action_keys)
                if clamp_action_targets(action_dict, args.min_hand_z, args.min_gaze_z):
                    ep_clamp_steps += 1
                robot.send_action(action_dict)

                # 5. Sync viewer and rate-limit to --fps
                mj_viewer.sync()
                elapsed = time.perf_counter() - t0
                precise_sleep(max(1.0 / args.fps - elapsed, 0.0))

                if step % args.fps == 0:      # print every second
                    print(f"  step {step:4d}/{args.episode_steps}")

            if ep_initial_cube_z is not None and ep_max_cube_z is not None:
                lift_delta = ep_max_cube_z - ep_initial_cube_z
                lifted = lift_delta >= args.lift_threshold
            else:
                lift_delta = float("nan")
                lifted = False

            touch_ratio = ep_touch_frames / max(args.episode_steps, 1)
            episode_metrics.append(
                {
                    "episode": ep + 1,
                    "initial_cube_z": ep_initial_cube_z,
                    "max_cube_z": ep_max_cube_z,
                    "lift_delta": lift_delta,
                    "touch_ratio": touch_ratio,
                    "clamp_ratio": ep_clamp_steps / max(args.episode_steps, 1),
                    "lifted": lifted,
                }
            )

            init_str = f"{ep_initial_cube_z:.3f}" if ep_initial_cube_z is not None else "n/a"
            max_str = f"{ep_max_cube_z:.3f}" if ep_max_cube_z is not None else "n/a"
            lift_str = f"{lift_delta:.3f}" if not np.isnan(lift_delta) else "n/a"
            print(
                f"  [metrics] z0={init_str} zmax={max_str} lift={lift_str}m "
                f"touch={touch_ratio:.1%} clamp={ep_clamp_steps / max(args.episode_steps, 1):.1%} "
                f"success={'yes' if lifted else 'no'}"
            )
            print(f"  Episode {ep + 1} done.\n")

        if episode_metrics:
            total = len(episode_metrics)
            successes = sum(1 for m in episode_metrics if m["lifted"])
            valid_lifts = [m["lift_delta"] for m in episode_metrics if not np.isnan(m["lift_delta"])]
            avg_lift = float(np.mean(valid_lifts)) if valid_lifts else float("nan")
            avg_touch = float(np.mean([m["touch_ratio"] for m in episode_metrics]))
            avg_clamp = float(np.mean([m["clamp_ratio"] for m in episode_metrics]))
            avg_lift_str = f"{avg_lift:.3f}m" if not np.isnan(avg_lift) else "n/a"

            print("[eval] Summary")
            print(f"  success_rate: {successes}/{total} ({successes / total:.1%})")
            print(f"  avg_lift_delta: {avg_lift_str}")
            print(f"  avg_touch_ratio: {avg_touch:.1%}")
            print(f"  avg_clamp_ratio: {avg_clamp:.1%}")

    finally:
        print("[eval] Cleaning up …")
        mj_viewer.close()
        if robot.is_connected:
            robot.disconnect()
        print("[eval] Finished.")


if __name__ == "__main__":
    main()
