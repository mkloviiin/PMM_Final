#!/usr/bin/env python3
"""Evaluate a trained LeRobot policy (e.g. SmolVLA) on iCub in MuJoCo.

This script loads a pretrained checkpoint, connects to the iCub MuJoCo simulation,
gets observations from the robot, feeds them through the policy to get actions,
and sends the actions back to the robot — closing the loop for autonomous execution.

It uses the same observation→preprocess→policy→postprocess→action pipeline that
lerobot-record uses internally, ensuring full compatibility with training.

Usage examples
--------------
# Basic (uses default checkpoint path and scene):
python eval_policy_mujoco.py

EXECUTE: python eval_policy_mujoco.py --task "Pick up the blue cube"

# Custom checkpoint and scene:
python eval_policy_mujoco.py \\
    --checkpoint /home/icub/mujoco_ws/REPO_ICUB/outputs/train/my_smolvla/checkpoints/020000/pretrained_model \\
    --scene 1 \\
    --num-episodes 5 \\
    --episode-len 600

# With a specific task prompt (must match what was used during training):
python eval_policy_mujoco.py --task "Pick up the blue cube"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Path setup — mirror play_mujoco.py so local plugins are found
# ---------------------------------------------------------------------------
script_dir = Path(__file__).resolve().parent

local_teleop_path = script_dir / "lerobot-teleoperator-icubteleop"
if local_teleop_path.exists():
    sys.path.insert(0, str(local_teleop_path))

local_robot_path = script_dir / "lerobot-robot-icub"
if local_robot_path.exists():
    sys.path.insert(0, str(local_robot_path))


# Add icubenv site-packages so mujoco / robotology are importable
icubenv_sp = os.path.expanduser("~/miniconda3/envs/icubenv/lib/python3.12/site-packages")
if icubenv_sp not in sys.path:
    sys.path.append(icubenv_sp)

# ---------------------------------------------------------------------------
# Imports that depend on the path setup above
# ---------------------------------------------------------------------------
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.robots import make_robot_from_config
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.common.control_utils import predict_action
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.device_utils import get_safe_torch_device

from lerobot_robot_icub.config_icub_mujoco import iCubMuJoCoConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset_features(checkpoint_path: Path) -> dict | None:
    """Load the dataset features dict from train_config.json → dataset meta/info.json."""
    train_cfg_path = checkpoint_path / "train_config.json"
    if not train_cfg_path.exists():
        return None
    with open(train_cfg_path, "r") as f:
        train_cfg = json.load(f)
    ds_root = train_cfg.get("dataset", {}).get("root", "")
    info_path = Path(ds_root) / "meta" / "info.json"
    if not info_path.exists():
        return None
    with open(info_path, "r") as f:
        info = json.load(f)
    return info.get("features", None)


def get_checkpoint_experiment_name(checkpoint_path: Path) -> str:
    """Infer experiment folder name from a checkpoint path.

    Expected shape:
      .../outputs/train/<experiment_name>/checkpoints/<step>/pretrained_model
    """
    for parent in checkpoint_path.parents:
        if parent.name == "checkpoints":
            exp_name = parent.parent.name
            if exp_name:
                return exp_name

    # Fallbacks for non-standard checkpoint layouts
    if checkpoint_path.name:
        return checkpoint_path.name
    return "eval_run"


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def tensor_or_scalar_to_float(val: Any) -> float | None:
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        return float(val.item())
    try:
        return float(val)
    except Exception:
        return None


def get_cube_height_m(obs: dict[str, Any], tc: Any) -> float | None:
    """Get cube/object height (z, meters) from observation, with model fallback."""
    z = tensor_or_scalar_to_float(obs.get("object_pos_z"))
    if z is not None:
        return z

    # Fallback to direct model read when observation key is unavailable.
    try:
        bid = tc.model.body("blue-cube").id
        return float(tc.data.xpos[bid][2])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    default_ckpt = str(
        script_dir.parent
        / "outputs"
        / "train"
        / "my_smolvla"
        / "checkpoints"
        / "020000"
        / "pretrained_model"
    )
    default_model = str(script_dir / "dependencies" / "assets" / "scenes" / "icub_table_scene.xml")
    default_config = str(script_dir / "utils" / "control_config.yaml")

    parser = argparse.ArgumentParser(
        description="Evaluate a trained policy on iCub MuJoCo"
    )
    parser.add_argument(
        "--checkpoint",
        default=default_ckpt,
        help="Path to pretrained_model directory (contains config.json + model.safetensors)",
    )
    parser.add_argument(
        "--scene",
        type=int,
        choices=[1, 2],
        default=None,
        help="Scene: 1=lift, 2=stack (overrides --model)",
    )
    parser.add_argument(
        "--model",
        default=default_model,
        help="Path to MuJoCo scene XML",
    )
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to control_config.yaml",
    )
    parser.add_argument(
        "--task",
        default="Pick up the blue cube",
        help="Language task prompt (should match what was used during training)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes to run",
    )
    parser.add_argument(
        "--episode-len",
        type=int,
        default=600,
        help="Max frames per episode (at 30 FPS → 600 = 20 s)",
    )
    parser.add_argument("--fps", type=int, default=30, help="Control frequency")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for policy inference",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve scene shortcut
    if args.scene == 1:
        args.model = str(script_dir / "dependencies" / "assets" / "scenes" / "icub_table_scene_lift.xml")
        if args.task == "Pick up the blue cube":
            args.task = "levantar el cubo"
    elif args.scene == 2:
        args.model = str(script_dir / "dependencies" / "assets" / "scenes" / "icub_table_scene_stack.xml")
        if args.task == "Pick up the blue cube":
            args.task = "apilar el cubo"

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo scene not found: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Control config not found: {config_path}")

    train_cfg_path = checkpoint_path / "train_config.json"
    policy_cfg_path = checkpoint_path / "config.json"
    train_cfg_raw = load_json_if_exists(train_cfg_path) or {}
    policy_cfg_raw = load_json_if_exists(policy_cfg_path) or {}
    ds_root_from_train = train_cfg_raw.get("dataset", {}).get("root", "")
    ds_repo_id_from_train = train_cfg_raw.get("dataset", {}).get("repo_id", "")
    dataset_info_path = Path(ds_root_from_train) / "meta" / "info.json" if ds_root_from_train else None
    dataset_stats_path = Path(ds_root_from_train) / "meta" / "stats.json" if ds_root_from_train else None
    dataset_info_raw = load_json_if_exists(dataset_info_path) if dataset_info_path else None
    dataset_stats_raw = load_json_if_exists(dataset_stats_path) if dataset_stats_path else None

    # ------------------------------------------------------------------
    # 1.  Load dataset features (needed to build observation frames)
    # ------------------------------------------------------------------
    ds_features = load_dataset_features(checkpoint_path)
    if ds_features is None:
        raise RuntimeError(
            "Could not load dataset features from train_config.json. "
            "Make sure the dataset used for training is still available at its original path."
        )

    action_names = ds_features.get("action", {}).get("names", [])
    if not action_names:
        # Fallback: typical right-arm + gaze action names
        action_names = [
            "rh_pos_x", "rh_pos_y", "rh_pos_z",
            "rh_quat_w", "rh_quat_x", "rh_quat_y", "rh_quat_z",
            "rh_gripper",
            "gaze_x", "gaze_y", "gaze_z",
        ]
    print(f"[eval] Action names ({len(action_names)}): {action_names}")

    # ------------------------------------------------------------------
    # 2.  Load the trained policy
    # ------------------------------------------------------------------
    print(f"[eval] Loading policy from {checkpoint_path} ...")

    policy_cfg = PreTrainedConfig.from_pretrained(str(checkpoint_path))
    policy_cfg.device = args.device
    policy_cfg.pretrained_path = str(checkpoint_path)

    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(
        pretrained_name_or_path=str(checkpoint_path),
        config=policy_cfg,
    )
    policy.to(args.device)
    policy.eval()
    print(f"[eval] Policy loaded: {policy_cfg.type}  (device={args.device})")

    # Load pre/post processors (normalisation, tokeniser, etc.)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg,
        pretrained_path=str(checkpoint_path),
    )
    print("[eval] Pre/post processors loaded")

    device = get_safe_torch_device(args.device)

    # ------------------------------------------------------------------
    # 3.  Set up iCub MuJoCo robot
    # ------------------------------------------------------------------
    print("[eval] Setting up iCub MuJoCo robot ...")
    os.environ.setdefault("ICUB_LEROBOT_CONFIG", str(config_path))
    os.environ.setdefault("ICUB_MUJOCO_MODEL_PATH", str(model_path))
    os.environ["ICUB_MUJOCO_VR_ENABLED"] = "0"

    robot_cfg = iCubMuJoCoConfig(
        model_path=str(model_path),
        config_path=str(config_path),
        vr_enabled=False,
    )
    robot = make_robot_from_config(robot_cfg)
    robot.connect()
    print("[eval] Robot connected")

    # Hide specific mocap target bodies from renders
    tc = robot.teleop_core
    model = tc.model
    hide_bodies = ["com_target", "l_hand_target", "r_hand_target"]
    for bname in hide_bodies:
        try:
            bid = model.body(bname).id
        except Exception:
            continue
        # Hide all geoms of this body
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == bid:
                model.geom_rgba[gid, 3] = 0.0
        # Hide all sites of this body
        for sid in range(model.nsite):
            if model.site_bodyid[sid] == bid:
                model.site_rgba[sid, 3] = 0.0
    print("[eval] Mocap targets hidden (com_target, l_hand_target, r_hand_target)")

    # Open a MuJoCo viewer for visual feedback
    mj_viewer = mujoco.viewer.launch_passive(tc.model, tc.data)
    # camera_name = "frontview"  # Make sure this camera is defined in your scene XML
    # camera_id = mujoco.mj_name2id(tc.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    # mj_viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    # mj_viewer.cam.fixedcamid = camera_id
    mj_viewer.cam.azimuth = -160
    mj_viewer.cam.elevation = -10
    mj_viewer.cam.distance = 1.5
    mj_viewer.cam.lookat[2] = 0.7
    print("[eval] MuJoCo viewer opened")

    # ------------------------------------------------------------------
    # 4.  Prepare output directory for logging
    # ------------------------------------------------------------------
    run_stamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_name = get_checkpoint_experiment_name(checkpoint_path)
    output_base = script_dir.parent / "outputs" / "tests" / f"{experiment_name}_{run_stamp}"
    output_base.mkdir(parents=True, exist_ok=True)
    print(f"[eval] Saving episode data to {output_base}")

    # Success criterion: cube height above threshold for a sustained duration.
    success_height_threshold_m = 0.65
    success_hold_seconds = 1.0
    success_hold_steps = max(1, int(np.ceil(success_hold_seconds * args.fps)))

    sim_dt_s = float(tc.model.opt.timestep)
    sim_hz = float(1.0 / sim_dt_s) if sim_dt_s > 0 else None
    robot_control_dt_s = float(getattr(robot.config, "control_dt", 1.0 / max(1, args.fps)))
    robot_control_hz = float(1.0 / robot_control_dt_s) if robot_control_dt_s > 0 else None
    sim_steps_per_action = max(1, int(round(robot_control_dt_s / sim_dt_s))) if sim_dt_s > 0 else None

    # Gather observation state & action column names from dataset features
    obs_state_names: list[str] = ds_features.get("observation.state", {}).get("names", [])
    all_touch_feature_keys = [
        key for key in (
            "rh_touch_thumb", "rh_touch_index", "rh_touch_middle",
            "lh_touch_thumb", "lh_touch_index", "lh_touch_middle",
        )
        if key in obs_state_names
    ]
    if not all_touch_feature_keys:
        all_touch_feature_keys = [
            "rh_touch_thumb", "rh_touch_index", "rh_touch_middle",
            "lh_touch_thumb", "lh_touch_index", "lh_touch_middle",
        ]

    # "Tomar" el cubo: contacto táctil sostenido con varios dedos.
    # Preferimos mano derecha; si no existen esas keys, usamos todas.
    preferred_take_keys = ["rh_touch_thumb", "rh_touch_index", "rh_touch_middle"]
    take_touch_keys = [key for key in preferred_take_keys if key in all_touch_feature_keys]
    if not take_touch_keys:
        take_touch_keys = all_touch_feature_keys

    touch_threshold = 0.5
    take_min_fingers = 2
    take_hold_seconds = 0.2
    take_hold_steps = max(1, int(np.ceil(take_hold_seconds * args.fps)))

    # ------------------------------------------------------------------
    # 5.  Run evaluation loop
    # ------------------------------------------------------------------
    # Pre-initialise for safe access in except/finally
    vid_writers: dict[str, cv2.VideoWriter] = {}
    obs_rows: list[dict] = []
    action_rows: list[dict] = []
    ep_dir = output_base / "episode_000"
    episode_summaries: list[dict[str, Any]] = []
    interrupted = False
    run_start_wall_time = time.time()

    current_ep_idx = -1
    current_ep_start_wall = 0.0
    current_ep_steps = 0
    current_ep_success = False
    current_ep_first_success_step: int | None = None
    current_ep_consecutive_above = 0
    current_ep_max_cube_height = float("-inf")
    current_ep_valid_cube_samples = 0
    current_ep_first_take_step: int | None = None
    current_ep_consecutive_take = 0

    try:
        for ep in range(args.num_episodes):
            print(f"\n{'='*60}")
            print(f"  Episode {ep + 1}/{args.num_episodes}")
            print(f"  Task: \"{args.task}\"")
            print(f"{'='*60}")

            current_ep_idx = ep
            current_ep_start_wall = time.time()
            current_ep_steps = 0
            current_ep_success = False
            current_ep_first_success_step = None
            current_ep_consecutive_above = 0
            current_ep_max_cube_height = float("-inf")
            current_ep_valid_cube_samples = 0
            current_ep_first_take_step = None
            current_ep_consecutive_take = 0

            # Reset the scene (place cube on table, snap targets to EE)
            if hasattr(tc, "_reset_scenario"):
                tc._reset_scenario()
            if hasattr(tc, "_snap_targets_to_ee"):
                tc._snap_targets_to_ee()

            # Warm-up physics
            for _ in range(10):
                mujoco.mj_step(tc.model, tc.data)
            mj_viewer.sync()

            # Reset policy action queue (important for chunk-based policies)
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()

            # --- Per-episode logging structures ---
            ep_dir = output_base / f"episode_{ep:03d}"
            ep_dir.mkdir(parents=True, exist_ok=True)

            obs_rows: list[dict] = []
            action_rows: list[dict] = []

            # Video writers per camera (will be initialised on first frame)
            record_cams = ["frontview", "head_cam", "head_cam_track_hand"]
            vid_writers: dict[str, cv2.VideoWriter] = {}

            for step in range(args.episode_len):
                t0 = time.perf_counter()

                # 1) Get raw observation from robot (dict of torch tensors)
                obs = robot.get_observation()

                # Fill in any observation keys that the dataset expects but the
                # robot didn't provide (e.g. object_pos_x when the cube body
                # name differs between scenes).  Use zero as fallback.
                for _feat_key, _feat_info in ds_features.items():
                    if not _feat_key.startswith(OBS_STR):
                        continue
                    for _name in _feat_info.get("names", []):
                        if _name not in obs:
                            obs[_name] = torch.tensor(0.0, dtype=torch.float32)

                touch_values = {
                    key: (tensor_or_scalar_to_float(obs.get(key)) or 0.0)
                    for key in all_touch_feature_keys
                }
                touch_any = any(val > touch_threshold for val in touch_values.values())

                take_touch_count = sum(
                    1 for key in take_touch_keys
                    if (tensor_or_scalar_to_float(obs.get(key)) or 0.0) > touch_threshold
                )
                take_contact = take_touch_count >= take_min_fingers
                if take_contact:
                    current_ep_consecutive_take += 1
                else:
                    current_ep_consecutive_take = 0

                if current_ep_first_take_step is None and current_ep_consecutive_take >= take_hold_steps:
                    current_ep_first_take_step = step - take_hold_steps + 1

                cube_height_m = get_cube_height_m(obs, tc)
                print(f"  Step {step}: cube height = {cube_height_m:.3f} m" if cube_height_m is not None else f"  Step {step}: cube height = N/A")  
                current_ep_steps = step + 1
                if cube_height_m is not None:
                    current_ep_valid_cube_samples += 1
                    current_ep_max_cube_height = max(current_ep_max_cube_height, float(cube_height_m))
                    if cube_height_m > success_height_threshold_m:
                        current_ep_consecutive_above += 1
                    else:
                        current_ep_consecutive_above = 0
                else:
                    current_ep_consecutive_above = 0

                if (not current_ep_success) and current_ep_consecutive_above >= success_hold_steps:
                    current_ep_success = True
                    current_ep_first_success_step = step

                # 2) Build dataset-format observation frame (numpy arrays
                #    with "observation.state", "observation.images.*" keys).
                #    This is the same transform used during recording.
                observation_frame = build_dataset_frame(ds_features, obs, prefix=OBS_STR)

                # Ensure all values are numpy arrays (images come as torch
                # tensors from the robot but predict_action expects numpy).
                for _k, _v in observation_frame.items():
                    if isinstance(_v, torch.Tensor):
                        # Images: (C, H, W) → (H, W, C) as uint8 numpy
                        # (prepare_observation_for_inference will re-permute & normalise)
                        if _v.dim() == 3:
                            observation_frame[_k] = _v.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                        else:
                            observation_frame[_k] = _v.cpu().numpy()

                # 3) Run the full lerobot inference pipeline:
                #    prepare_observation → preprocessor → policy.select_action → postprocessor
                action_tensor = predict_action(
                    observation=observation_frame,
                    policy=policy,
                    device=device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=policy.config.use_amp,
                    task=args.task,
                    robot_type=robot.name,
                )

                # 4) Convert flat action tensor → named dict for robot.send_action()
                action_dict = make_robot_action(action_tensor, ds_features)

                # 5) Log observation & action data
                obs_row = {"step": step}
                for _name in obs_state_names:
                    val = obs.get(_name)
                    if isinstance(val, torch.Tensor):
                        obs_row[_name] = val.item()
                    elif val is not None:
                        obs_row[_name] = float(val)
                    else:
                        obs_row[_name] = 0.0
                obs_row["cube_height_z"] = float(cube_height_m) if cube_height_m is not None else np.nan
                obs_row["cube_above_0p65m"] = int(bool(cube_height_m is not None and cube_height_m > success_height_threshold_m))
                obs_row["touch_any"] = int(touch_any)
                obs_row["take_contact"] = int(take_contact)
                obs_rows.append(obs_row)

                act_row = {"step": step}
                for _name in action_names:
                    val = action_dict.get(_name)
                    if isinstance(val, torch.Tensor):
                        act_row[_name] = val.item()
                    elif val is not None:
                        act_row[_name] = float(val)
                    else:
                        act_row[_name] = 0.0
                action_rows.append(act_row)

                # 6) Display + record camera views
                cam_imgs = []
                for _cam in record_cams:
                    if _cam in obs and isinstance(obs[_cam], torch.Tensor):
                        img_rgb = obs[_cam].permute(1, 2, 0).cpu().numpy()  # (H,W,C) RGB
                        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                        cam_imgs.append(img_bgr)
                        # Initialise video writer on first frame
                        if _cam not in vid_writers:
                            h, w = img_bgr.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            vid_path = str(ep_dir / f"{_cam}.mp4")
                            vid_writers[_cam] = cv2.VideoWriter(vid_path, fourcc, args.fps, (w, h))
                        vid_writers[_cam].write(img_bgr)
                if cam_imgs:
                    combined = np.concatenate(cam_imgs, axis=0)
                    try:
                        cv2.imshow("iCub Cameras: frontview | head_cam | track_hand", combined)
                        cv2.waitKey(1)
                    except cv2.error:
                        pass  # Headless environment — skip display, videos saved to disk

                # 7) Send action to robot (sets mocap targets + steps physics)
                robot.send_action(action_dict)

                # 8) Update viewer
                mj_viewer.sync()

                # 9) Maintain target FPS
                elapsed = time.perf_counter() - t0
                precise_sleep(max(1.0 / args.fps - elapsed, 0.0))

                # Print progress every 5 seconds
                if step > 0 and step % (args.fps * 5) == 0:
                    secs = step / args.fps
                    print(f"  [{secs:.0f}s] step {step}/{args.episode_len}")

            # --- Save episode data ---
            for vw in vid_writers.values():
                vw.release()
            vid_writers.clear()

            pd.DataFrame(obs_rows).to_csv(ep_dir / "observations.csv", index=False)
            pd.DataFrame(action_rows).to_csv(ep_dir / "actions.csv", index=False)

            ep_duration_s = time.time() - current_ep_start_wall
            episode_summary = {
                "episode_index": int(ep),
                "episode_dir": str(ep_dir),
                "completed": True,
                "interrupted": False,
                "steps_executed": int(current_ep_steps),
                "episode_len_limit": int(args.episode_len),
                "duration_s": float(ep_duration_s),
                "success": bool(current_ep_success),
                "success_threshold_m": float(success_height_threshold_m),
                "success_hold_seconds": float(success_hold_seconds),
                "success_hold_steps": int(success_hold_steps),
                "first_success_step": int(current_ep_first_success_step) if current_ep_first_success_step is not None else None,
                "first_success_time_s": float(current_ep_first_success_step / args.fps) if current_ep_first_success_step is not None else None,
                "first_take_step": int(current_ep_first_take_step) if current_ep_first_take_step is not None else None,
                "first_take_time_s": float(current_ep_first_take_step / args.fps) if current_ep_first_take_step is not None else None,
                "max_cube_height_m": float(current_ep_max_cube_height) if current_ep_valid_cube_samples > 0 else None,
                "cube_height_samples": int(current_ep_valid_cube_samples),
            }
            episode_summaries.append(episode_summary)

            print(f"  Episode {ep + 1} finished ({args.episode_len / args.fps:.0f}s)")
            print(f"  Success: {current_ep_success}  |  Max cube z: {episode_summary['max_cube_height_m']}")
            print(f"  Saved → {ep_dir}")

    except KeyboardInterrupt:
        interrupted = True
        print("\n[eval] Interrupted by user.")
        # Save any partial episode data
        for vw in vid_writers.values():
            vw.release()
        if obs_rows:
            pd.DataFrame(obs_rows).to_csv(ep_dir / "observations.csv", index=False)
        if action_rows:
            pd.DataFrame(action_rows).to_csv(ep_dir / "actions.csv", index=False)
            print(f"  Partial episode saved → {ep_dir}")

        if current_ep_idx >= 0 and (not episode_summaries or episode_summaries[-1].get("episode_index") != current_ep_idx):
            ep_duration_s = time.time() - current_ep_start_wall
            episode_summaries.append({
                "episode_index": int(current_ep_idx),
                "episode_dir": str(ep_dir),
                "completed": False,
                "interrupted": True,
                "steps_executed": int(current_ep_steps),
                "episode_len_limit": int(args.episode_len),
                "duration_s": float(ep_duration_s),
                "success": bool(current_ep_success),
                "success_threshold_m": float(success_height_threshold_m),
                "success_hold_seconds": float(success_hold_seconds),
                "success_hold_steps": int(success_hold_steps),
                "first_success_step": int(current_ep_first_success_step) if current_ep_first_success_step is not None else None,
                "first_success_time_s": float(current_ep_first_success_step / args.fps) if current_ep_first_success_step is not None else None,
                "first_take_step": int(current_ep_first_take_step) if current_ep_first_take_step is not None else None,
                "first_take_time_s": float(current_ep_first_take_step / args.fps) if current_ep_first_take_step is not None else None,
                "max_cube_height_m": float(current_ep_max_cube_height) if current_ep_valid_cube_samples > 0 else None,
                "cube_height_samples": int(current_ep_valid_cube_samples),
            })

    finally:
        run_end_wall_time = time.time()
        completed_episodes = int(sum(1 for ep in episode_summaries if ep.get("completed", False)))
        successful_episodes = int(sum(1 for ep in episode_summaries if ep.get("success", False)))
        executed_episodes = int(len(episode_summaries))
        successful_take_times_s = [
            float(ep["first_take_time_s"])
            for ep in episode_summaries
            if ep.get("success", False) and ep.get("first_take_time_s") is not None
        ]
        mean_time_to_take_success_s = float(np.mean(successful_take_times_s)) if successful_take_times_s else None
        std_time_to_take_success_s = float(np.std(successful_take_times_s)) if successful_take_times_s else None

        dataset_meta_summary = None
        if dataset_info_raw is not None:
            dataset_meta_summary = {
                "total_episodes": dataset_info_raw.get("total_episodes"),
                "total_frames": dataset_info_raw.get("total_frames"),
                "fps": dataset_info_raw.get("fps"),
                "robot_type": dataset_info_raw.get("robot_type"),
                "codebase_version": dataset_info_raw.get("codebase_version"),
                "chunks_size": dataset_info_raw.get("chunks_size"),
                "total_tasks": dataset_info_raw.get("total_tasks"),
                "video_path": dataset_info_raw.get("video_path"),
                "data_path": dataset_info_raw.get("data_path"),
            }

        summary = {
            "run": {
                "timestamp_start_unix": float(run_start_wall_time),
                "timestamp_end_unix": float(run_end_wall_time),
                "duration_s": float(run_end_wall_time - run_start_wall_time),
                "output_dir": str(output_base),
                "task": args.task,
                "scene": int(args.scene) if args.scene is not None else None,
                "model_xml_path": str(model_path),
                "control_config_path": str(config_path),
                "interrupted": bool(interrupted),
            },
            "evaluation": {
                "requested_num_episodes": int(args.num_episodes),
                "executed_episodes": executed_episodes,
                "completed_episodes": completed_episodes,
                "successful_episodes": successful_episodes,
                "success_rate_over_completed": float(successful_episodes / completed_episodes) if completed_episodes > 0 else 0.0,
                "success_rate_over_executed": float(successful_episodes / executed_episodes) if executed_episodes > 0 else 0.0,
                "mean_time_to_take_blue_cube_success_s": mean_time_to_take_success_s,
                "std_time_to_take_blue_cube_success_s": std_time_to_take_success_s,
                "successful_episodes_with_detected_take": int(len(successful_take_times_s)),
                "episode_len_limit_steps": int(args.episode_len),
                "requested_control_hz": float(args.fps),
                "requested_control_dt_s": float(1.0 / args.fps),
                "simulation_hz": sim_hz,
                "simulation_dt_s": sim_dt_s,
                "robot_internal_control_hz": robot_control_hz,
                "robot_internal_control_dt_s": robot_control_dt_s,
                "sim_steps_per_action": int(sim_steps_per_action) if sim_steps_per_action is not None else None,
                "success_criterion": {
                    "cube_height_threshold_m": float(success_height_threshold_m),
                    "min_hold_time_s": float(success_hold_seconds),
                    "required_consecutive_steps": int(success_hold_steps),
                },
                "touch_detection": {
                    "feature_keys_all": all_touch_feature_keys,
                    "feature_keys_for_take": take_touch_keys,
                    "touch_threshold": float(touch_threshold),
                    "take_min_fingers": int(take_min_fingers),
                    "take_hold_seconds": float(take_hold_seconds),
                    "take_hold_steps": int(take_hold_steps),
                    "definition": "Tomar = contacto tactil sostenido con al menos take_min_fingers dedos sobre threshold.",
                    "metric_scope": "Media y desviacion estandar calculadas solo sobre episodios exitosos.",
                },
            },
            "model": {
                "policy_type": getattr(policy_cfg, "type", None),
                "checkpoint_path": str(checkpoint_path),
                "policy_config_path": str(policy_cfg_path),
                "policy_config": policy_cfg_raw,
                "device": args.device,
            },
            "training": {
                "train_config_path": str(train_cfg_path),
                "train_config": train_cfg_raw,
            },
            "dataset_used_for_training": {
                "repo_id": ds_repo_id_from_train,
                "root": ds_root_from_train,
                "meta_info_path": str(dataset_info_path) if dataset_info_path is not None else None,
                "meta_stats_path": str(dataset_stats_path) if dataset_stats_path is not None else None,
                "meta_info_summary": dataset_meta_summary,
                "meta_info": dataset_info_raw,
                "meta_stats_keys": sorted(list(dataset_stats_raw.keys())) if dataset_stats_raw is not None else None,
            },
            "episodes": episode_summaries,
        }

        summary_path = output_base / "evaluation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[eval] Summary JSON saved → {summary_path}")

        print("[eval] Cleaning up ...")
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # Headless environment
        mj_viewer.close()
        if robot.is_connected:
            robot.disconnect()
        print("[eval] Done!")


if __name__ == "__main__":
    main()
