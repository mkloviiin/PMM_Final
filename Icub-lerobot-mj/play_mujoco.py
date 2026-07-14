#!/usr/bin/env python3
"""One-command launcher for lerobot-record using local iCub plugins.

This script sets local defaults for iCub config/model and forwards arguments
into LeRobot's official `lerobot-record` entrypoint.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
import time
import threading

repo_root = Path(__file__).resolve().parent

# Force usage of local lerobot plugins (overriding site-packages)
local_teleop_path = repo_root / "lerobot-teleoperator-icubteleop"
if local_teleop_path.exists():
    sys.path.insert(0, str(local_teleop_path))


local_robot_path = repo_root / "lerobot-robot-icub"
if local_robot_path.exists():
    sys.path.insert(0, str(local_robot_path))

# Add icubenv site-packages to sys.path to generic python
sys.path.append(os.path.expanduser("~/miniconda3/envs/icubenv/lib/python3.12/site-packages"))


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    default_cfg = repo_root / "utils" / "control_config.yaml"
    default_model = repo_root / "dependencies" / "assets" / "scenes" / "icub_table_scene.xml"

    parser = argparse.ArgumentParser(
        description="Record dataset with lerobot-record and iCub plugins"
    )
    parser.add_argument(
        "--repo-id",
        default="local/icub_mujoco_demo",
        help="Dataset repo_id/name",
    )
    parser.add_argument(
        "--root",
        default=str(repo_root.parent / "data"),
        help="Dataset root directory",
    )
    parser.add_argument("--fps", type=int, default=30, help="Dataset FPS")
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=50,
        help="Number of episodes to record",
    )
    parser.add_argument(
        "--scene",
        type=int,
        choices=[1, 2],
        help="Scene to load: 1 for lift (levantar el cubo), 2 for stack (apilar el cubo)"
    )
    parser.add_argument(
        "--single-task",
        default="Pick up the blue cube",
        help="Value for --dataset.single_task",
    )
    parser.add_argument(
        "--episode-time-s",
        type=int,
        default=0,
        help="Episode duration in seconds (0 = manual stop)",
    )
    parser.add_argument(
        "--config",
        default=str(default_cfg),
        help="Path to control_config.yaml",
    )
    parser.add_argument(
        "--model",
        default=str(default_model),
        help="Path to MuJoCo scene.xml",
    )
    parser.add_argument(
        "--vr",
        action="store_true",
        help="Enable VR control in MuJoCo robot plugin",
    )
    parser.add_argument(
        "--vr-ip",
        default=None,
        help="Quest/VR publisher IP for ZMQ connect mode",
    )
    parser.add_argument(
        "--vr-cable",
        action="store_true",
        help="Connect the Quest over USB cable via adb reverse instead of WiFi "
             "(implies --vr, forces --vr-ip 127.0.0.1)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push dataset to Hugging Face Hub (disabled by default)",
    )
    return parser.parse_args()


def _manual_vr_record(
    *,
    args: argparse.Namespace,
    repo_id: str,
    dataset_root: Path,
    model_path: Path,
    cfg_path: Path,
    cmd_source=None,    # callable()->str|None para GUI; None = leer stdin
    on_features=None,   # callable(dict) llamado con dataset_features al conectar
    on_status=None,     # callable(str) para eventos: "waiting","recording","saved:N"
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.pipeline_features import (
        aggregate_pipeline_dataset_features,
        create_initial_features,
    )
    from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts
    from lerobot.processor import make_default_processors
    from lerobot.teleoperators import make_teleoperator_from_config
    from lerobot.robots import make_robot_from_config
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.robot_utils import precise_sleep

    from lerobot_robot_icub.config_icub_mujoco import iCubMuJoCoConfig
    from lerobot_teleoperator_icubteleop.config_icubteleop import iCubTeleopConfig

    vr_enabled = bool(args.vr or args.vr_ip)

    robot_cfg = iCubMuJoCoConfig(
        model_path=str(model_path),
        config_path=str(cfg_path),
        vr_enabled=vr_enabled,
        vr_ip=args.vr_ip,
    )
    teleop_cfg = iCubTeleopConfig(
        model_path=str(model_path),
        config_path=str(cfg_path),
        vr_enabled=False,
        vr_ip=None,
    )
    teleop_cfg.vr_enabled = False
    teleop_cfg.vr_ip = None

    robot = make_robot_from_config(robot_cfg)
    teleop = make_teleoperator_from_config(teleop_cfg)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    if on_features is not None:
        on_features(dataset_features)

    dataset = LeRobotDataset.create(
        repo_id,
        args.fps,
        root=dataset_root,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=True,
    )

    robot.connect()
    teleop.connect()

    # Initial reset: place cube on table so it's visible from the start
    teleop._reset_scenario()
    # Sync initial cube position to physics model too
    if hasattr(robot, "teleop_core") and robot.teleop_core:
        robot.teleop_core.set_cube_pose(teleop._last_reset_pos, teleop._last_reset_quat)
    teleop._reset_request = False

    cmd_state = {
        "start": False,
        "stop": False,
        "exit": False,
    }
    cmd_lock = threading.Lock()

    if cmd_source is not None:
        def _console_listener() -> None:
            while True:
                with cmd_lock:
                    if cmd_state["exit"]:
                        return
                cmd = cmd_source()
                if cmd:
                    with cmd_lock:
                        if cmd == "1":
                            cmd_state["start"] = True
                        elif cmd == "2":
                            cmd_state["stop"] = True
                        elif cmd == "3":
                            cmd_state["exit"] = True
                            return
                time.sleep(0.02)
    else:
        def _console_listener() -> None:
            print("[Manual] Console controls: 1=START, 2=STOP, 3=EXIT")
            while True:
                try:
                    cmd = input("[Manual] COMMAND (1/2/3): ").strip()
                except EOFError:
                    break
                except Exception:
                    continue
                with cmd_lock:
                    if cmd == "1":
                        cmd_state["start"] = True
                    elif cmd == "2":
                        cmd_state["stop"] = True
                    elif cmd == "3":
                        cmd_state["exit"] = True
                        break

    listener_t = threading.Thread(target=_console_listener, daemon=True)
    listener_t.start()

    try:
        if args.vr or args.vr_ip:
            print("\n[Manual] VR controls: A=start episode, B=stop episode")
        print("[Manual] Console controls: 1=start, 2=stop, 3=exit")
        print(f"[Manual] dataset root: {dataset_root}")

        recorded = 0
        while recorded < args.num_episodes:
            if on_status is not None:
                on_status("waiting")
            print(f"\n[Manual] Waiting START for episode {dataset.num_episodes}...")
            while True:
                with cmd_lock:
                    if cmd_state["exit"]:
                        return
                start_t = time.perf_counter()
                obs = robot.get_observation()
                teleop.send_feedback(obs)
                
                # Check for reset request (KEY_R from teleop viewer)
                if getattr(teleop, "_reset_request", False):
                    try:
                        pos = teleop._last_reset_pos
                        quat = teleop._last_reset_quat
                        # Sync cube to physics model (teleop_core)
                        if hasattr(robot, "teleop_core") and robot.teleop_core:
                            robot.teleop_core.set_cube_pose(pos, quat)
                        else:
                            robot.set_cube_pose(pos, quat)
                    except Exception as e:
                        print(f"[Record] Sync failed: {e}")
                    teleop._reset_request = False

                act = teleop.get_action()
                act_processed = teleop_action_processor((act, obs))
                robot_action_to_send = robot_action_processor((act_processed, obs))
                robot.send_action(robot_action_to_send)

                start_ev, _, _ = robot.consume_vr_record_events() if hasattr(robot, "consume_vr_record_events") else (False, False, False)
                with cmd_lock:
                    start_cmd = cmd_state["start"]
                    if start_cmd:
                        cmd_state["start"] = False
                if start_ev or start_cmd:
                    break

                precise_sleep(max(1.0 / args.fps - (time.perf_counter() - start_t), 0.0))

            if on_status is not None:
                on_status("recording")
            print(f"[Manual] Recording episode {dataset.num_episodes}...")
            episode_start = time.perf_counter()

            while True:
                with cmd_lock:
                    if cmd_state["exit"]:
                        return
                loop_start = time.perf_counter()

                obs = robot.get_observation()
                teleop.send_feedback(obs)
                obs_processed = robot_observation_processor(obs)

                # Check for reset request (KEY_R from teleop viewer)
                if getattr(teleop, "_reset_request", False):
                    try:
                        pos = teleop._last_reset_pos
                        quat = teleop._last_reset_quat
                        # Sync cube to physics model (teleop_core)
                        if hasattr(robot, "teleop_core") and robot.teleop_core:
                            robot.teleop_core.set_cube_pose(pos, quat)
                        else:
                            robot.set_cube_pose(pos, quat)
                        print(f"[Record] Synced cube to {pos}")
                    except Exception as e:
                        print(f"[Record] Sync failed: {e}")
                    teleop._reset_request = False

                act = teleop.get_action()
                act_processed = teleop_action_processor((act, obs))
                robot_action_to_send = robot_action_processor((act_processed, obs))
                robot.send_action(robot_action_to_send)

                observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
                action_frame = build_dataset_frame(dataset.features, act_processed, prefix=ACTION)
                frame = {**observation_frame, **action_frame, "task": args.single_task}
                dataset.add_frame(frame)

                _, stop_ev, discard_ev = robot.consume_vr_record_events() if hasattr(robot, "consume_vr_record_events") else (False, False, False)
                with cmd_lock:
                    stop_cmd = cmd_state["stop"]
                    if stop_cmd:
                        cmd_state["stop"] = False
                if args.episode_time_s > 0 and (time.perf_counter() - episode_start) >= args.episode_time_s:
                    stop_ev = True

                # B mantenido → descartar episodio y volver a esperar
                if discard_ev:
                    print("[Manual] Episodio DESCARTADO (B mantenido) — re-grabando...")
                    if hasattr(dataset, "clear_episode_buffer"):
                        dataset.clear_episode_buffer()
                    else:
                        dataset.clear_episode() # Por si en versiones viejas se llamaba así
                    break

                if stop_ev or stop_cmd:
                    break

                precise_sleep(max(1.0 / args.fps - (time.perf_counter() - loop_start), 0.0))

            if discard_ev:
                continue   # no guardar, volver al bucle de espera

            dataset.save_episode()
            recorded += 1
            if on_status is not None:
                on_status(f"saved:{recorded}")
            print(f"[Manual] Episode saved ({recorded}/{args.num_episodes})")

    finally:
        try:
            dataset.finalize()
        except Exception:
            pass
        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    args = parse_args()

    if args.scene == 1:
        args.model = str(repo_root / "dependencies" / "assets" / "scenes" / "icub_table_scene_lift.xml")
        args.single_task = "levantar el cubo"
    elif args.scene == 2:
        args.model = str(repo_root / "dependencies" / "assets" / "scenes" / "icub_table_scene_stack.xml")
        args.single_task = "apilar el cubo"

    cfg_path = Path(args.config).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    os.environ.setdefault("ICUB_LEROBOT_CONFIG", str(cfg_path))
    os.environ.setdefault("ICUB_MUJOCO_MODEL_PATH", str(model_path))
    os.environ.setdefault("ICUB_WORKSPACE_ROOT", str(repo_root.parent))

    if args.vr_cable:
        from dependencies.vr_usb import connect_cable, CABLE_IP
        if not connect_cable():
            raise SystemExit(
                "USB cable connection failed. Check the Quest is plugged in, "
                "'USB debugging' is enabled, and the popup on the headset is accepted."
            )
        args.vr_ip = CABLE_IP
        args.vr = True
        print(f"[VR] USB cable ready. Set BeaVR's IP to {CABLE_IP} on the headset.")

    vr_enabled = bool(args.vr or args.vr_ip)
    os.environ["ICUB_MUJOCO_VR_ENABLED"] = "1" if vr_enabled else "0"
    if args.vr_ip:
        os.environ["ICUB_MUJOCO_VR_IP"] = str(args.vr_ip)

    repo_id = args.repo_id.strip()
    if "/" not in repo_id:
        repo_id = f"local/{repo_id}"

    base_root = Path(args.root).expanduser()
    run_suffix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dataset_root = base_root / f"{repo_id.replace('/', '_')}_{run_suffix}"

    _manual_vr_record(
        args=args,
        repo_id=repo_id,
        dataset_root=dataset_root,
        model_path=model_path,
        cfg_path=cfg_path,
    )


if __name__ == "__main__":
    main()
