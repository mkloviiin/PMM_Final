import argparse
import os
import time
import numpy as np

import mujoco
import mujoco.viewer
import torch

# We need to import the local plugins dynamically like play_mujoco does!
import sys
from pathlib import Path

# Add icubenv site-packages to sys.path to resolve mujoco / robotology
icubenv_site_packages = os.path.expanduser("~/miniconda3/envs/icubenv/lib/python3.12/site-packages")
if icubenv_site_packages not in sys.path:
    sys.path.append(icubenv_site_packages)
    
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir / "lerobot-robot-icub"))
sys.path.append(str(script_dir / "lerobot-teleoperator-icubteleop"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots import make_robot_from_config
from lerobot.utils.robot_utils import precise_sleep
from lerobot_robot_icub.config_icub_mujoco import iCubMuJoCoConfig


def resolve_dataset_root(repo_id: str, requested_root: str | None) -> Path | None:
    if requested_root:
        return Path(requested_root).expanduser().resolve()

    if not repo_id.startswith("local/"):
        return None

    data_root = script_dir.parent / "data"
    pattern = f"{repo_id.replace('/', '_')}_*"
    candidates = sorted(
        data_root.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, default="mklovin/icub_mujoco_demo")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Local dataset root path (if omitted and repo-id is local/*, latest matching run is auto-selected)",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to replay")
    parser.add_argument(
        "--model",
        type=str,
        default=str(script_dir / "dependencies" / "assets" / "scenes" / "icub_table_scene_lift.xml"),
        help="Path to MuJoCo scene XML",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(script_dir / "utils" / "control_config.yaml"),
        help="Path to control_config.yaml",
    )
    args = parser.parse_args()

    dataset_root = resolve_dataset_root(args.repo_id, args.root)
    if dataset_root is not None:
        print(f"[replay] Loading local dataset {args.repo_id} from {dataset_root}...")
        dataset = LeRobotDataset(args.repo_id, root=dataset_root)
    else:
        print(f"[replay] Loading dataset {args.repo_id}...")
        dataset = LeRobotDataset(args.repo_id)

    dataset_action_names = dataset.features.get("action", {}).get("names", [])
    dataset_action_idx = {name: idx for idx, name in enumerate(dataset_action_names)}
    
    # Try to find the structure of observation.state
    # dataset.features is typically where the feature dictionaries are stored in LeRobotDataset
    state_names = dataset.features.get("observation.state", {}).get("names", [])
    
    # Find the indices of the cube position and quaternion
    cube_pos_indices = []
    cube_quat_indices = []
    for i, name in enumerate(state_names):
        if name in ["object_pos_x", "object_pos_y", "object_pos_z"]:
            cube_pos_indices.append(i)
        elif name in ["object_quat_w", "object_quat_x", "object_quat_y", "object_quat_z"]:
            cube_quat_indices.append(i)
            
    has_cube_state = len(cube_pos_indices) == 3 and len(cube_quat_indices) == 4

    print("[replay] Setting up robot...")
    model_path = str(Path(args.model).expanduser().resolve())
    config_path = str(Path(args.config).expanduser().resolve())
    robot_cfg = iCubMuJoCoConfig(
        model_path=model_path,
        config_path=config_path,
    )
    robot = make_robot_from_config(robot_cfg)
    robot.connect()
    print("[replay] Robot connected ✓")
    
    action_keys = list(robot.action_features.keys())

    missing_action_keys = [name for name in action_keys if name not in dataset_action_idx]
    if missing_action_keys:
        print(
            "[replay] Warning: missing action keys in dataset; they will be set to 0.0 -> "
            + ", ".join(missing_action_keys)
        )
    
    tc = robot.teleop_core
    mj_viewer = mujoco.viewer.launch_passive(tc.model, tc.data)
    mj_viewer.cam.azimuth = 135
    mj_viewer.cam.elevation = -20
    mj_viewer.cam.distance = 2.0
    print("[replay] MuJoCo viewer opened ✓")

    try:
        total_ep = dataset.meta.total_episodes
        ep_to_play = min(total_ep, args.episodes)
        
        for ep in range(ep_to_play):
            print(f"── Replaying Episode {ep + 1}/{total_ep} ──")
            
            if hasattr(tc, "_reset_scenario"):
                tc._reset_scenario()
                
            # Get episode bounds
            ep_info = dataset.meta.episodes[ep]
            start_i = ep_info["dataset_from_index"]
            end_i = ep_info["dataset_to_index"]
            num_frames = end_i - start_i
            
            # Extract first state to set the cube exactly
            if has_cube_state and hasattr(tc, "set_cube_pose"):
                first_frame = dataset[start_i]
                first_state = first_frame["observation.state"]
                
                pos = first_state[cube_pos_indices].numpy()
                quat = first_state[cube_quat_indices].numpy()
                tc.set_cube_pose(pos, quat)
                print(f"[Reset] Restored exact recorded blue-cube → pos:{pos}")
            
            # Let's run a few warmup steps so the environment settles
            for _ in range(5):
                mujoco.mj_step(tc.model, tc.data)
                
            mj_viewer.sync()
            
            for f in range(num_frames):
                t0 = time.perf_counter()
                frame_data = dataset[start_i + f]
                action_tensor = frame_data["action"]
                
                # action_tensor is actually (action_dim,) if we fetched one frame
                if action_tensor.dim() >= 1:
                    action_tensor = action_tensor.squeeze()

                action_dict = {}
                for key in action_keys:
                    idx = dataset_action_idx.get(key)
                    action_dict[key] = action_tensor[idx].item() if idx is not None else 0.0
                
                # IMPORTANT: In LeRobot dataset replay, we often just want to send the action
                # directly to the PD controller and step.
                robot.send_action(action_dict)
                
                # Wait: icub_mujoco robot.send_action only *sets* the target positions. 
                # The robot teleop background thread automatically steps MuJoCo and tracks it.
                
                mj_viewer.sync()
                # Use elapsed time for precise fps sleep
                elapsed = time.perf_counter() - t0
                precise_sleep(max(1.0 / args.fps - elapsed, 0.0))

    finally:
        print("[replay] Cleaning up …")
        mj_viewer.close()
        if robot.is_connected:
            robot.disconnect()
        print("[replay] Finished.")

if __name__ == "__main__":
    main()
