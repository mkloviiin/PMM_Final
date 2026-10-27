#!/usr/bin/env python3
"""Unified launcher for teleoperating and recording datasets with iCub across MuJoCo, Gazebo, and Real Robot.

Supported Backends:
  - mujoco: Pure MuJoCo physics simulation driven by TSIK (QP IK) or TSID (QP Inverse Dynamics / Impedance).
  - gazebo: Gazebo Sim + YARP simulation connected via BLFRobotBridge / teleop_module_sm.
  - real: Real iCub robot connected via BLFRobotBridge / teleop_module_sm.

Supported Controllers:
  - ik: Task-Space Inverse Kinematics (blf.ik.QPFixedBaseInverseKinematics)
  - id: Task-Space Inverse Dynamics (blf.tsid.QPFixedBaseTSID) with optional Cartesian impedance (--impedance)
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

# Ensure headless/offscreen MuJoCo rendering uses EGL if not explicitly set
os.environ.setdefault("MUJOCO_GL", "egl")

repo_root = Path(__file__).resolve().parent

# Ensure plugin and src paths are in sys.path
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

metrics_path = repo_root / "metrics"
if metrics_path.exists() and str(metrics_path) not in sys.path:
    sys.path.insert(0, str(metrics_path))

# Pre-import av to avoid FFmpeg symbol clashes between conda and PyAV
try:
    import av
    import av.container
    import av.video
except ImportError:
    pass

SCENES_YAML = repo_root / "mujoco_sim" / "assets" / "scenes" / "scenes.yaml"


class _NoMetrics:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _make_metrics(**kwargs):
    try:
        from frame_metrics import FrameMetrics
        return FrameMetrics.create(**kwargs)
    except Exception:
        return _NoMetrics()


def _discard_episode(dataset) -> None:
    import shutil

    if not hasattr(dataset, "clear_episode_buffer"):
        if hasattr(dataset, "clear_episode"):
            dataset.clear_episode()
        return

    ep = (getattr(dataset, "episode_buffer", None) or {}).get("episode_index", dataset.num_episodes)
    if hasattr(ep, "size"):
        ep = ep.item() if ep.size == 1 else ep[0]
    dataset.clear_episode_buffer()

    for key in getattr(dataset.meta, "video_keys", []):
        img_dir = dataset._get_image_file_dir(int(ep), key)
        if img_dir.is_dir():
            shutil.rmtree(img_dir)


def _resolve_scene(scene_arg: str | int | None, controller: str = "ik") -> tuple[Path, str, list, list]:
    import yaml

    scenes_data = []
    if SCENES_YAML.exists():
        with open(SCENES_YAML, "r", encoding="utf-8") as f:
            scenes_data = yaml.safe_load(f) or []

    selected = None
    if scene_arg is not None:
        try:
            idx = int(scene_arg) - 1
            if 0 <= idx < len(scenes_data):
                selected = scenes_data[idx]
        except ValueError:
            for s in scenes_data:
                if s.get("label", "").lower() == str(scene_arg).lower():
                    selected = s
                    break

    if selected is None and scenes_data:
        selected = scenes_data[0]

    if selected:
        xml_name = selected.get("model", "scene_icub_lift.xml")
        task = selected.get("task", "Pick up the blue cube")
        objects = selected.get("objects", [])
        joints = selected.get("joints", [])
    else:
        xml_name = "scene_icub_lift.xml"
        task = "Pick up the blue cube"
        objects = [{"body": "blue-cube"}]
        joints = []

    model_dir = repo_root / "mujoco_sim" / "assets" / "scenes"
    if controller == "id":
        # Check if dedicated idyn scene exists, or use default visuomanip_idyn_limits
        idyn_candidate = model_dir / xml_name.replace("ikin", "idyn")
        if idyn_candidate.exists():
            model_path = idyn_candidate
        else:
            model_path = model_dir / xml_name
    else:
        model_path = model_dir / xml_name

    return model_path, task, objects, joints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Teleoperation & Dataset Recording for iCub")
    parser.add_argument(
        "--backend",
        choices=["mujoco", "gazebo", "real"],
        default="mujoco",
        help="Backend to run against (default: mujoco)",
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
        help="Enable compliant Cartesian impedance mode when --controller id is selected",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Run interactive teleoperation only without recording a LeRobot dataset",
    )
    parser.add_argument("--repo-id", default="local/icub_demo", help="Dataset repository / experiment name")
    parser.add_argument("--root", default=str(repo_root / "data"), help="Dataset output directory")
    parser.add_argument("--dataset-root", default=None, help="Explicit dataset output path")
    parser.add_argument("--fps", type=int, default=30, help="Control & recording FPS (default: 30)")
    parser.add_argument("--num-episodes", type=int, default=50, help="Number of episodes to record")
    parser.add_argument("--episode-time-s", type=int, default=0, help="Max episode duration in seconds (0 = manual)")
    parser.add_argument("--scene", default=None, help="Scene index or label (e.g. 1 for lift, 2 for stack)")
    parser.add_argument("--single-task", default=None, help="Task description string")
    parser.add_argument(
        "--control-arms",
        choices=["right", "left", "both"],
        default=None,
        help="Active arms (default: auto-detect from active teleop module or control_config.yaml)",
    )
    parser.add_argument("--vr", action="store_true", help="Enable Meta Quest VR teleoperation")
    parser.add_argument("--vr-ip", default=None, help="Meta Quest headset IP for streaming")
    parser.add_argument("--vr-cable", action="store_true", help="Connect Meta Quest via USB reverse-tether ADB")
    parser.add_argument("--robot-name", choices=["icub", "icubSim"], default=None, help="YARP robot name")
    parser.add_argument("--robot-model", default=None, help="iCub model name (e.g. iCubGazeboV2_5_visuomanip, iCubGazeboV2_5_fixed)")
    parser.add_argument("--model", default=None, help="Override MJCF XML scene path")
    parser.add_argument("--config", default=str(repo_root / "config" / "control_config.yaml"), help="Config path")
    return parser.parse_args()


def _resolve_control_arms(explicit_arms: str | None, config_path: str | Path | None = None) -> str:
    if explicit_arms:
        return explicit_arms
    # 1. Try to auto-detect from active teleop module
    tmp_arms_file = Path("/tmp/icub_teleop_arms")
    if tmp_arms_file.exists():
        try:
            m = tmp_arms_file.read_text(encoding="utf-8").strip().lower()
            if m in ("right", "left", "both"):
                print(f"[play.py] Auto-detected control_arms='{m}' from active teleop module.")
                return m
        except Exception:
            pass
    # 2. Try environment variable
    env_arms = os.environ.get("ICUB_CONTROL_ARMS")
    if env_arms and env_arms.strip().lower() in ("right", "left", "both"):
        return env_arms.strip().lower()
    # 3. Fallback to control_config.yaml or 'right'
    if config_path and Path(config_path).exists():
        import yaml
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                m = data.get("control_arms")
                if m and str(m).strip().lower() in ("right", "left", "both"):
                    return str(m).strip().lower()
        except Exception:
            pass
    return "right"


def run_mujoco_teleop_loop(args, model_path: Path, scene_task: str, scene_objects: list, scene_joints: list):
    args.control_arms = _resolve_control_arms(args.control_arms, args.config)
    from lerobot_robot_icub.config_icub_mujoco import iCubMuJoCoConfig
    from lerobot_robot_icub.icub_mujoco import iCubMuJoCo
    from lerobot_teleoperator_icubteleop.config_icubteleop import iCubTeleopConfig
    from lerobot_teleoperator_icubteleop.icubteleop import iCubTeleop

    fps = max(1, int(args.fps))
    control_dt = 1.0 / fps

    robot_cfg = iCubMuJoCoConfig(
        model_path=str(model_path),
        config_path=str(args.config),
        control_arms=args.control_arms,
        controller_type=args.controller,
        impedance_mode=args.impedance,
        control_dt=control_dt,
        vr_enabled=bool(args.vr or args.vr_ip),
        vr_ip=args.vr_ip,
        scene_objects=scene_objects,
        scene_joints=scene_joints,
    )

    teleop_cfg = iCubTeleopConfig(
        model_path=str(model_path),
        config_path=str(args.config),
        control_arms=args.control_arms,
        vr_enabled=False,
        scene_objects=scene_objects,
        scene_joints=scene_joints,
    )

    robot = iCubMuJoCo(robot_cfg)
    teleop = iCubTeleop(teleop_cfg)

    print(f"\n=======================================================")
    print(f" Starting iCub Teleoperation (Backend: MuJoCo)")
    print(f" Controller: {args.controller.upper()} {'(Impedance Mode)' if args.impedance else ''}")
    print(f" Model:      {model_path.name}")
    print(f" Control:    {args.control_arms} arm(s)")
    print(f" Recording:  {'DISABLED (--no-record)' if args.no_record else 'ENABLED'}")
    print(f"=======================================================\n")

    robot.connect()
    teleop.connect()

    teleop._reset_scenario()
    if robot.teleop_core:
        reset_poses = getattr(teleop, "_last_reset_poses", None)
        if reset_poses:
            robot.teleop_core.set_object_poses(reset_poses)

    dataset = None
    if not args.no_record:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
        try:
            from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
        except ImportError:
            from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts
        from lerobot.processor import make_default_processors

        has_cameras = bool(getattr(robot.config, "camera_ports", None) or getattr(robot, "_cameras_ft", None))
        teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
        dataset_features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=teleop_action_processor,
                initial_features=create_initial_features(action=robot.action_features),
                use_videos=has_cameras,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=robot_observation_processor,
                initial_features=create_initial_features(observation=robot.observation_features),
                use_videos=has_cameras,
            ),
        )

        dataset_root = args.dataset_root or str(Path(args.root) / f"{args.repo_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        dataset = LeRobotDataset.create(
            args.repo_id,
            args.fps,
            root=dataset_root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=has_cameras,
        )
        print(f"[Dataset] Recording to: {dataset_root}")

    is_recording = False
    episode_index = 0
    ep_start_t = 0.0

    print("[Controls] Console: 1 = START, 2 = STOP/SAVE, 3 = EXIT, h = HOME, n = RESET objects, t = TOGGLE teleop, i = TOGGLE impedance, 8/9 = Right Hand, 6/7 = Left Hand")
    print("[Controls] Viewer:  Mouse (Ctrl+Drag) = Move/Rotate Mocaps, 8/9 = Right Hand, 6/7 = Left Hand")
    print("[Controls] VR:      A = START episode, B = STOP/SAVE episode (hold B 1.5s to DISCARD), Y = RESET objects, Index Trigger = Gripper, Lower Trigger = Freeze")

    cmd_queue = []
    def _console_reader():
        while True:
            try:
                c = input().strip()
                if c:
                    cmd_queue.append(c)
            except Exception:
                break
    threading.Thread(target=_console_reader, daemon=True).start()

    dt = 1.0 / args.fps
    try:
        while episode_index < args.num_episodes or args.no_record:
            loop_t0 = time.perf_counter()

            if teleop.viewer is not None and not teleop.viewer.is_running():
                print("[Viewer] Window closed by user. Exiting...")
                break

            c_cmd = cmd_queue.pop(0) if cmd_queue else None
            cmd_lower = c_cmd.strip().lower() if c_cmd else ""

            if cmd_lower == "3":
                print("[Exit] Quitting...")
                break

            if cmd_lower in ("t", "teleop"):
                new_state = robot.toggle_teleop()
                print(f"\n[Teleop] Toggled teleoperation: {'ACTIVE' if new_state else 'IDLE (Paused)'}")

            if cmd_lower in ("h", "home"):
                print("\n[Teleop] Home requested (h): Resetting mocaps and moving robot smoothly to home...")
                teleop._reset_mocaps_to_xml_pose(strict=True)
                robot.go_home()

            reset_scen = (cmd_lower in ("n", "reset")) or teleop.consume_world_reset_event()
            if reset_scen:
                print("\n[Teleop] Reset scenario (n): Re-randomizing scene objects...")
                teleop._reset_scenario()
                if robot.teleop_core and getattr(teleop, "_last_reset_poses", None):
                    robot.teleop_core.set_object_poses(teleop._last_reset_poses)

            if cmd_lower in ("i", "impedance"):
                robot.toggle_impedance()

            if cmd_lower == "8":
                teleop.set_gripper("rh", 0.0)
                print("[Console] Right Gripper: OPEN")
            elif cmd_lower == "9":
                teleop.set_gripper("rh", 1.0)
                print("[Console] Right Gripper: CLOSE")
            elif cmd_lower == "6":
                teleop.set_gripper("lh", 0.0)
                print("[Console] Left Gripper: OPEN")
            elif cmd_lower == "7":
                teleop.set_gripper("lh", 1.0)
                print("[Console] Left Gripper: CLOSE")

            vr_start, vr_stop, vr_discard = robot.consume_vr_record_events()
            if not vr_start and not vr_stop:
                vr_start, vr_stop, vr_discard = teleop.consume_vr_record_events()

            if (c_cmd == "1" or vr_start) and not is_recording and not args.no_record:
                is_recording = True
                ep_start_t = time.time()
                print(f"\n[Record] Episode {episode_index} STARTED.")

            if is_recording and (c_cmd == "2" or vr_stop):
                is_recording = False
                dataset.save_episode()
                print(f"[Record] Episode {episode_index} SAVED ({dataset.num_frames} total frames).")
                episode_index += 1
                teleop._reset_scenario()
                if robot.teleop_core and getattr(teleop, "_last_reset_poses", None):
                    robot.teleop_core.set_object_poses(teleop._last_reset_poses)

            if is_recording and vr_discard:
                is_recording = False
                _discard_episode(dataset)
                print(f"[Record] Episode {episode_index} DISCARDED.")
                teleop._reset_scenario()
                if robot.teleop_core and getattr(teleop, "_last_reset_poses", None):
                    robot.teleop_core.set_object_poses(teleop._last_reset_poses)

            action = teleop.get_action()
            robot.send_action(action)
            obs = robot.get_observation()
            teleop.send_feedback(obs)

            if is_recording and dataset is not None:
                frame = build_dataset_frame(
                    dataset_features,
                    action=action,
                    observation=obs,
                    task=scene_task,
                )
                dataset.add_frame(frame)

                if args.episode_time_s > 0 and (time.time() - ep_start_t) >= args.episode_time_s:
                    is_recording = False
                    dataset.save_episode()
                    print(f"[Record] Episode {episode_index} reached time limit -> SAVED.")
                    episode_index += 1
                    teleop._reset_scenario()
                    if robot.teleop_core and getattr(teleop, "_last_reset_poses", None):
                        robot.teleop_core.set_object_poses(teleop._last_reset_poses)

            elapsed = time.perf_counter() - loop_t0
            sleep_s = dt - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    finally:
        if is_recording and dataset is not None:
            dataset.save_episode()
        if getattr(robot, "is_teleop_active", False) and hasattr(robot, "stop_teleop"):
            robot.stop_teleop()
        try:
            teleop.disconnect()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass
        if getattr(args, "vr_cable", False):
            try:
                from interfaces.vr_usb import remove_reverse_ports
                remove_reverse_ports()
            except Exception:
                pass
        print("[Play] Session ended cleanly.")


def run_yarp_teleop_loop(args, scene_task: str):
    args.control_arms = _resolve_control_arms(args.control_arms, args.config)
    from lerobot_robot_icubrg.config_icub import iCubConfig
    from lerobot_robot_icubrg.icub import iCub
    from lerobot_teleoperator_icubrgteleop.config_icubteleop import iCubTeleopConfig
    from lerobot_teleoperator_icubrgteleop.icubteleop import iCubTeleop

    robot_name = args.robot_name or ("icub" if args.backend == "real" else "icubSim")
    os.environ["ICUB_ROBOT_NAME"] = robot_name
    if args.robot_model:
        os.environ["ICUB_ROBOT_MODEL"] = args.robot_model

    robot_cfg = iCubConfig(
        config_path=str(args.config),
        control_arms=args.control_arms,
        controller_type=args.controller,
        impedance_mode=args.impedance,
        robot_model=args.robot_model,
    )
    teleop_cfg = iCubTeleopConfig(
        config_path=str(args.config),
        control_arms=args.control_arms,
        vr_enabled=bool(args.vr or args.vr_ip),
        vr_ip=args.vr_ip,
    )

    robot = iCub(robot_cfg)
    teleop = iCubTeleop(teleop_cfg)

    print(f"\n=======================================================")
    print(f" Starting iCub Teleoperation (Backend: {args.backend.upper()})")
    print(f" Robot Name: {robot_name}")
    print(f" Control:    {args.control_arms} arm(s)")
    print(f" Recording:  {'DISABLED (--no-record)' if args.no_record else 'ENABLED'}")
    print(f"=======================================================\n")

    # If teleop_module_sm is not running on YARP, launch it automatically in background
    import yarp
    yarp.Network.init()
    teleop_proc = None
    if not yarp.Network.exists("/teleop/state:o"):
        print(f"[play.py] /teleop/state:o not found. Auto-launching teleop_module_sm.py ({args.controller.upper()})...")
        teleop_script = repo_root / "src" / "interfaces" / "teleop_module_sm.py"
        teleop_cmd = [
            sys.executable,
            str(teleop_script),
            "--robot", robot_name,
            "--controller", args.controller,
            "--control-arms", args.control_arms,
        ]
        if args.impedance:
            teleop_cmd.append("--impedance")
        teleop_proc = subprocess.Popen(teleop_cmd)
        print("[play.py] Waiting for /teleop/state:o to appear on YARP network...")
        for _ in range(40):
            if yarp.Network.exists("/teleop/state:o"):
                print("[play.py] ✓ /teleop/state:o is ready.")
                break
            time.sleep(0.5)
        else:
            print("[play.py] Warning: Timeout waiting for /teleop/state:o. Proceeding to connect...")

    robot.connect()
    teleop.connect()

    dataset = None
    if not args.no_record:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
        try:
            from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
        except ImportError:
            from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts
        from lerobot.processor import make_default_processors

        has_cameras = bool(getattr(robot.config, "camera_ports", None) or getattr(robot, "_cameras_ft", None))
        teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
        dataset_features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=teleop_action_processor,
                initial_features=create_initial_features(action=robot.action_features),
                use_videos=has_cameras,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=robot_observation_processor,
                initial_features=create_initial_features(observation=robot.observation_features),
                use_videos=has_cameras,
            ),
        )

        dataset_root = args.dataset_root or str(Path(args.root) / f"{args.repo_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        dataset = LeRobotDataset.create(
            args.repo_id,
            args.fps,
            root=dataset_root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=has_cameras,
        )
        print(f"[Dataset] Recording to: {dataset_root}")

    is_recording = False
    episode_index = 0
    ep_start_t = 0.0

    print("[Controls] Console: 1 = START, 2 = STOP/SAVE, 3 = EXIT, h = HOME, n = RESET objects, t = TOGGLE teleop, i = TOGGLE impedance, 8/9 = Right Hand, 6/7 = Left Hand")
    print("[Controls] Viewer:  Mouse (Ctrl+Drag) = Move/Rotate Mocaps, 8/9 = Right Hand, 6/7 = Left Hand")
    print("[Controls] VR:      A = START episode, B = STOP/SAVE episode (hold B 1.5s to DISCARD), Y = RESET objects, Index Trigger = Gripper, Lower Trigger = Freeze")

    cmd_queue = []
    def _console_reader():
        while True:
            try:
                c = input().strip()
                if c:
                    cmd_queue.append(c)
            except Exception:
                break
    threading.Thread(target=_console_reader, daemon=True).start()

    dt = 1.0 / args.fps
    try:
        while episode_index < args.num_episodes or args.no_record:
            loop_t0 = time.perf_counter()

            c_cmd = cmd_queue.pop(0) if cmd_queue else None
            cmd_lower = c_cmd.strip().lower() if c_cmd else ""

            if cmd_lower == "3":
                print("[Exit] Quitting...")
                break

            if cmd_lower in ("t", "teleop"):
                if robot.is_teleop_active:
                    robot.stop_teleop()
                    print("\n[Teleop] Status: STOPPED (Idle - holding current posture)\n")
                else:
                    robot.start_teleop()
                    print("\n[Teleop] Status: ACTIVE (Tracking mocaps)\n")

            if cmd_lower in ("h", "home"):
                print("\n[Teleop] Home requested (h): Resetting mocaps and moving robot smoothly to home...")
                teleop._reset_mocaps_to_xml_pose()
                robot.go_home()
                print("[Teleop] Status: IDLE (Robot at home pose - press 't' in console to resume teleoperation)\n")

            world_reset = (
                (cmd_lower in ("n", "reset"))
                or teleop.consume_world_reset_event()
            )
            if world_reset:
                if args.backend == "gazebo":
                    print("\n[Gazebo] Reset requested (console 'n' / VR 'Y'): Resetting objects via reset_objects.sh...")
                    reset_script = repo_root / "gazebo_sim" / "scripts" / "reset_objects.sh"
                    if reset_script.exists():
                        subprocess.run([str(reset_script)], check=False)
                    else:
                        print(f"[Gazebo] Warning: reset script not found at {reset_script}")

            if cmd_lower in ("i", "impedance"):
                robot.toggle_impedance()

            if cmd_lower == "8":
                teleop.set_gripper("rh", 0.0)
                print("[Console] Right Gripper: OPEN")
            elif cmd_lower == "9":
                teleop.set_gripper("rh", 1.0)
                print("[Console] Right Gripper: CLOSE")
            elif cmd_lower == "6":
                teleop.set_gripper("lh", 0.0)
                print("[Console] Left Gripper: OPEN")
            elif cmd_lower == "7":
                teleop.set_gripper("lh", 1.0)
                print("[Console] Left Gripper: CLOSE")

            vr_start, vr_stop, vr_discard = teleop.consume_vr_record_events()

            if (c_cmd == "1" or vr_start) and not is_recording and not args.no_record:
                is_recording = True
                ep_start_t = time.time()
                robot.reset_command_state()
                print(f"\n[Record] Episode {episode_index} STARTED.")

            if is_recording and (c_cmd == "2" or vr_stop):
                is_recording = False
                dataset.save_episode()
                print(f"[Record] Episode {episode_index} SAVED.")
                episode_index += 1
                if args.backend == "gazebo":
                    subprocess.run([str(repo_root / "gazebo_sim" / "scripts" / "reset_objects.sh")], check=False)

            if is_recording and vr_discard:
                is_recording = False
                _discard_episode(dataset)
                print(f"[Record] Episode {episode_index} DISCARDED.")
                if args.backend == "gazebo":
                    subprocess.run([str(repo_root / "gazebo_sim" / "scripts" / "reset_objects.sh")], check=False)

            action = teleop.get_action()
            robot.send_action(action)
            obs = robot.get_observation()
            teleop.send_feedback(obs)

            if is_recording and dataset is not None:
                frame = build_dataset_frame(
                    dataset_features,
                    action=action,
                    observation=obs,
                    task=scene_task,
                )
                dataset.add_frame(frame)

                if args.episode_time_s > 0 and (time.time() - ep_start_t) >= args.episode_time_s:
                    is_recording = False
                    dataset.save_episode()
                    print(f"[Record] Episode {episode_index} reached time limit -> SAVED.")
                    episode_index += 1
                    if args.backend == "gazebo":
                        subprocess.run([str(repo_root / "gazebo_sim" / "scripts" / "reset_objects.sh")], check=False)

            elapsed = time.perf_counter() - loop_t0
            sleep_s = dt - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    finally:
        if is_recording and dataset is not None:
            dataset.save_episode()
        if getattr(robot, "is_teleop_active", False) and hasattr(robot, "stop_teleop"):
            try:
                robot.stop_teleop()
            except Exception:
                pass
        try:
            teleop.disconnect()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass
        if teleop_proc is not None:
            print("[play.py] Terminating background teleop_module_sm...")
            teleop_proc.terminate()
            try:
                teleop_proc.wait(timeout=3)
            except Exception:
                teleop_proc.kill()
        if getattr(args, "vr_cable", False):
            try:
                from interfaces.vr_usb import remove_reverse_ports
                remove_reverse_ports()
            except Exception:
                pass
        print("[Play] Session ended cleanly.")


def main():
    args = parse_args()
    if args.vr_cable:
        from interfaces.vr_usb import connect_cable, CABLE_IP
        print("[VR] Conectando visor Meta Quest por cable USB (ADB reverse)...")
        if not connect_cable():
            raise SystemExit(
                "\n[ERROR] Falló la conexión USB con el Meta Quest 2.\n"
                "Verifica que el visor esté conectado por cable USB-C, encendido,\n"
                "y que hayas aceptado el mensaje 'Permitir depuración USB' dentro del visor.\n"
            )
        args.vr_ip = CABLE_IP
        args.vr = True

    vr_enabled = bool(args.vr or args.vr_ip)
    os.environ["ICUB_MUJOCO_VR_ENABLED"] = "1" if vr_enabled else "0"
    if args.vr_ip:
        os.environ["ICUB_MUJOCO_VR_IP"] = str(args.vr_ip)

    model_path, scene_task, scene_objects, scene_joints = _resolve_scene(args.scene, controller=args.controller)
    if args.model:
        model_path = Path(args.model)
    if args.single_task:
        scene_task = args.single_task

    if args.backend == "mujoco":
        run_mujoco_teleop_loop(args, model_path, scene_task, scene_objects, scene_joints)
    else:
        run_yarp_teleop_loop(args, scene_task)


if __name__ == "__main__":
    main()

