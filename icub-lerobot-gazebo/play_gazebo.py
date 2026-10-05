#!/usr/bin/env python3
"""One-command launcher for LeRobot recording against the real/Gazebo iCub.

Flow:  VR  <->  MuJoCo (solo visualizador / fuente de targets)  <->  Gazebo o robot real.

The *robot* is ``lerobot_robot_icubrg.iCub`` — a YARP client that talks to
``common/yarp_module/teleop_module_sm.py``. The MuJoCo teleoperator
(``lerobot_teleoperator_icubrgteleop``) only produces Cartesian targets and
shows a feedback window; it does not simulate the task, and the camera
images sent to the headset come from the robot, never from MuJoCo.

Prerequisites (see README.md):
  1. Either the real robot or ``gazebo/scripts/start_sim.sh`` must be
     running, selected via ``robot_name`` in ``config/control_config.yaml``
     (``icub`` = real, ``icubSim`` = Gazebo). That is the only switch.
  2. ``common/yarp_module/teleop_module_sm.py`` must be running and
     connected to that robot.
  3. Then run this script to teleoperate (and optionally record).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

repo_root = Path(__file__).resolve().parent

# Force usage of local lerobot plugins (overriding site-packages)
local_teleop_path = repo_root / "common" / "lerobot-teleoperator-icubrgteleop"
if local_teleop_path.exists():
    sys.path.insert(0, str(local_teleop_path))

local_robot_path = repo_root / "common" / "lerobot-robot-icubrg"
if local_robot_path.exists():
    sys.path.insert(0, str(local_robot_path))

# Add icubenv site-packages to sys.path (generic python, not the conda env's own)
sys.path.append(os.path.expanduser("~/miniconda3/envs/icubenv/lib/python3.12/site-packages"))

# PyAV (con el que LeRobot codifica los videos) trae su propio FFmpeg 61, pero
# los plugins de imagen de YARP enlazan el FFmpeg 62 de conda. Si YARP recibe
# imágenes antes de que PyAV esté cargado, los símbolos se mezclan y av.open()
# hace segfault al guardar el episodio. Cargar PyAV primero lo evita.
import av  # noqa: E402,F401
import av.container  # noqa: E402,F401
import av.video  # noqa: E402,F401

SCENES_DIR = repo_root / "mujoco" / "assets" / "scenes"

# The MuJoCo side is only a mirror of the Gazebo robot, so its table is left
# empty: the manipulated cube lives in the Gazebo world (gazebo/worlds/icub_world.sdf)
# and MuJoCo has no way to observe its pose.
SCENE_MODEL = SCENES_DIR / "scene_icub_empty_table.xml"
SCENE_TASK = "Pick up the blue cube"
SCENE_OBJECTS: list = []


def parse_args() -> argparse.Namespace:
    default_cfg = repo_root / "config" / "control_config.yaml"

    parser = argparse.ArgumentParser(
        description="Teleoperate/record the real or Gazebo iCub with lerobot-record-style plugins"
    )
    parser.add_argument("--repo-id", default="local/icub_gazebo_demo", help="Dataset repo_id/name")
    parser.add_argument("--root", default=str(repo_root.parent / "data"), help="Dataset root directory")
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Explicit output dataset directory. If set, overrides <root>/<repo>_<timestamp>. "
        "Used by the Hub to control/know the exact dataset path.",
    )
    parser.add_argument(
        "--robot-name",
        default=None,
        choices=["icub", "icubSim"],
        help="Override robot_name for the client side (camera ports) without editing the YAML. "
        "icub = real robot, icubSim = Gazebo. The whole-stack switch still lives in control_config.yaml.",
    )
    parser.add_argument("--fps", type=int, default=30, help="Dataset FPS")
    parser.add_argument("--num-episodes", type=int, default=50, help="Number of episodes to record")
    parser.add_argument(
        "--single-task",
        default=SCENE_TASK,
        help="Value for --dataset.single_task",
    )
    parser.add_argument(
        "--episode-time-s",
        type=int,
        default=0,
        help="Episode duration in seconds (0 = manual stop)",
    )
    parser.add_argument("--config", default=str(default_cfg), help="Path to control_config.yaml")
    parser.add_argument(
        "--model",
        default=str(SCENE_MODEL),
        help="Path to the MuJoCo scene.xml used by the teleoperator visualizer "
        "(only one scenario is wired up for now: lifting the blue cube)",
    )
    parser.add_argument(
        "--control-arms",
        choices=["auto", "right", "left", "both"],
        default="auto",
        help="Arms to control. 'auto' detects available YARP ports on the robot",
    )
    parser.add_argument("--vr", action="store_true", help="Enable VR control in the teleoperator")
    parser.add_argument("--vr-ip", default=None, help="Quest/VR publisher IP for ZMQ connect mode")
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
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Do not subscribe to/record the cameras (eyes + Gazebo front camera); saves bandwidth",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Run the teleop loop without recording a dataset (simple mode)",
    )
    return parser.parse_args()


def _filter_actuators(full_actuators: dict, control_arms: str) -> dict:
    filtered = {}
    if "torso" in full_actuators:
        filtered["torso"] = full_actuators["torso"]
    if control_arms in ["left", "both"] and "left_arm" in full_actuators:
        filtered["left_arm"] = full_actuators["left_arm"]
    if control_arms in ["right", "both"] and "right_arm" in full_actuators:
        filtered["right_arm"] = full_actuators["right_arm"]
    if "head" in full_actuators:
        filtered["head"] = full_actuators["head"]
    return filtered


def _build_robot_and_teleop(
    args: argparse.Namespace,
    cfg_path: Path,
    model_path: Path,
    scene_objects: list | None = None,
    scene_joints: list | None = None,
):
    """Build robot_cfg/teleop_cfg, auto-detect arms, and instantiate both plugins."""
    import yarp
    from lerobot.robots import make_robot_from_config
    from lerobot.teleoperators import make_teleoperator_from_config

    from lerobot_robot_icubrg.config_icub import iCubConfig, load_yarp_config
    from lerobot_teleoperator_icubrgteleop.config_icubteleop import iCubTeleopConfig

    vr_enabled = bool(args.vr or args.vr_ip)

    robot_cfg = iCubConfig(config_path=str(cfg_path))
    teleop_cfg = iCubTeleopConfig(
        model_path=str(model_path),
        config_path=str(cfg_path),
        vr_enabled=vr_enabled,
        vr_ip=args.vr_ip,
        scene_objects=scene_objects or [],
        scene_joints=scene_joints or [],
    )

    # Detect available arms on the real/Gazebo robot (YARP ports must already exist)
    selected_control = args.control_arms
    if selected_control == "auto":
        yarp.Network.init()
        has_rh = bool(yarp.Network.exists(robot_cfg.remote_rh_port))
        has_lh = bool(yarp.Network.exists(robot_cfg.remote_lh_port))
        if has_rh and has_lh:
            selected_control = "both"
        elif has_rh:
            selected_control = "right"
        elif has_lh:
            selected_control = "left"
        else:
            selected_control = robot_cfg.control_arms
        print(f"[play_gazebo] control arms auto -> {selected_control} (rh={has_rh}, lh={has_lh})")

    if selected_control != robot_cfg.control_arms:
        cfg_dict = load_yarp_config(str(cfg_path))
        full_actuators = cfg_dict.get("actuators", {})
        filtered_actuators = _filter_actuators(full_actuators, selected_control)
        robot_cfg.control_arms = selected_control
        teleop_cfg.control_arms = selected_control
        if filtered_actuators:
            robot_cfg.actuators_to_use = filtered_actuators
            teleop_cfg.actuators_to_use = filtered_actuators

    if args.no_camera:
        robot_cfg.camera_ports = {}
        print("[play_gazebo] camera subscriptions disabled (--no-camera)")
    else:
        print(f"[play_gazebo] cameras: {robot_cfg.camera_ports}")

    robot = make_robot_from_config(robot_cfg)
    teleop = make_teleoperator_from_config(teleop_cfg)
    return robot, teleop, selected_control, vr_enabled


def _vr_status(robot, *lines: str) -> None:
    """Texto de la pantalla chica del VR cuando no hay cámara frontal (robot real).

    Lo dibuja teleop_module_sm (RPC vr_status). Sin tildes: cv2.putText no las soporta.
    """
    text = "|".join(lines).replace('"', "'")
    robot.send_rpc_command(f'vr_status "{text}"')


def _forward_world_reset(robot, teleop) -> None:
    """Botón de reset (Y en VR / R en el visor) -> reposiciona el cubo en Gazebo.

    El teleoperador solo resetea su espejo MuJoCo (mesa vacía); el objeto real
    vive en Gazebo, así que se le pide al teleop_module_sm vía RPC, que corre
    gazebo/scripts/reset_objects.sh (posición aleatoria sobre la mesa).
    """
    if hasattr(teleop, "consume_world_reset_event") and teleop.consume_world_reset_event():
        reply = robot.send_rpc_command("reset_scenario")
        print(f"[Reset] reset_scenario -> {reply}", flush=True)


def _manual_vr_record(
    *,
    args: argparse.Namespace,
    repo_id: str,
    dataset_root: Path,
    robot,
    teleop,
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.pipeline_features import (
        aggregate_pipeline_dataset_features,
        create_initial_features,
    )
    from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts
    from lerobot.processor import make_default_processors
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.robot_utils import precise_sleep

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

    dataset = LeRobotDataset.create(
        repo_id,
        args.fps,
        root=dataset_root,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=True,
    )

    cmd_state = {"start": False, "stop": False, "exit": False}
    cmd_lock = threading.Lock()

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
        vr_enabled = bool(args.vr or args.vr_ip)
        if vr_enabled:
            print("\n[Manual] VR controls: A=start episode, B=stop episode")
        print("[Manual] Console controls: 1=start, 2=stop, 3=exit")
        print(f"[Manual] dataset root: {dataset_root}")

        recorded = 0
        while recorded < args.num_episodes:
            print(f"\n[Manual] Waiting START for episode {dataset.num_episodes}...", flush=True)
            _vr_status(robot, "EN ESPERA", f"Episodios grabados: {recorded}/{args.num_episodes}",
                       "A = grabar")
            discard_ev = False
            while True:
                with cmd_lock:
                    if cmd_state["exit"]:
                        return
                start_t = time.perf_counter()
                obs = robot.get_observation()
                teleop.send_feedback(obs)
                act = teleop.get_action()
                act_processed = teleop_action_processor((act, obs))
                robot_action_to_send = robot_action_processor((act_processed, obs))
                robot.send_action(robot_action_to_send)
                _forward_world_reset(robot, teleop)

                # VR record events (A/B) come from the teleoperator: the real/Gazebo
                # robot never receives the VR stream directly.
                start_ev = False
                if vr_enabled and hasattr(teleop, "consume_vr_record_events"):
                    start_ev, _, _ = teleop.consume_vr_record_events()
                with cmd_lock:
                    start_cmd = cmd_state["start"]
                    if start_cmd:
                        cmd_state["start"] = False
                if start_ev or start_cmd:
                    break

                precise_sleep(max(1.0 / args.fps - (time.perf_counter() - start_t), 0.0))

            print(f"[Manual] Recording episode {dataset.num_episodes}...", flush=True)
            _vr_status(robot, "GRABANDO", f"Episodio {recorded + 1}/{args.num_episodes}",
                       f"Episodios grabados: {recorded}", "B = guardar")
            episode_start = time.perf_counter()

            while True:
                with cmd_lock:
                    if cmd_state["exit"]:
                        return
                loop_start = time.perf_counter()

                obs = robot.get_observation()
                teleop.send_feedback(obs)
                obs_processed = robot_observation_processor(obs)

                act = teleop.get_action()
                act_processed = teleop_action_processor((act, obs))
                robot_action_to_send = robot_action_processor((act_processed, obs))
                robot.send_action(robot_action_to_send)
                _forward_world_reset(robot, teleop)

                observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
                action_frame = build_dataset_frame(dataset.features, act_processed, prefix=ACTION)
                frame = {**observation_frame, **action_frame, "task": args.single_task}
                dataset.add_frame(frame)

                stop_ev = False
                discard_ev = False
                if vr_enabled and hasattr(teleop, "consume_vr_record_events"):
                    _, stop_ev, discard_ev = teleop.consume_vr_record_events()
                with cmd_lock:
                    stop_cmd = cmd_state["stop"]
                    if stop_cmd:
                        cmd_state["stop"] = False
                if args.episode_time_s > 0 and (time.perf_counter() - episode_start) >= args.episode_time_s:
                    stop_ev = True

                if discard_ev:
                    print("[Manual] Episode DISCARDED (B held) — re-recording...")
                    if hasattr(dataset, "clear_episode_buffer"):
                        dataset.clear_episode_buffer()
                    else:
                        dataset.clear_episode()
                    break

                if stop_ev or stop_cmd:
                    break

                precise_sleep(max(1.0 / args.fps - (time.perf_counter() - loop_start), 0.0))

            if discard_ev:
                continue  # Don't save, go back to wait loop

            # Sin pool de procesos: con >1 cámara LeRobot hace fork para codificar en
            # paralelo, y el fork de un proceso con hilos YARP/CUDA vivos deja a los
            # hijos bloqueados en un lock heredado (el episodio nunca se guarda).
            _vr_status(robot, "GUARDANDO...", f"Episodio {recorded + 1}/{args.num_episodes}")
            dataset.save_episode(parallel_encoding=False)
            recorded += 1
            if recorded >= args.num_episodes:
                _vr_status(robot, "SESION COMPLETA",
                           f"Episodios grabados: {recorded}/{args.num_episodes}")
            print(f"[Manual] Episode saved ({recorded}/{args.num_episodes})", flush=True)

    finally:
        if recorded == 0:
            import shutil
            try:
                shutil.rmtree(dataset_root, ignore_errors=True)
                print(f"[Manual] No se grabó nada — carpeta eliminada: {dataset_root}")
            except Exception as e:
                print(f"[Manual] No se pudo eliminar la carpeta vacía: {e}")
        else:
            try:
                dataset.finalize()
            except Exception:
                pass


def _simple_teleop_loop(*, args, robot, teleop) -> None:
    """Teleoperate without recording — just move the robot and mirror feedback."""
    print("Starting Gazebo/real-robot teleoperation loop (no recording)...")
    try:
        while True:
            start_t = time.perf_counter()
            obs = robot.get_observation()
            teleop.send_feedback(obs)
            act = teleop.get_action()
            robot.send_action(act)
            _forward_world_reset(robot, teleop)
            time.sleep(max(1.0 / args.fps - (time.perf_counter() - start_t), 0.0))
    except KeyboardInterrupt:
        print("\nStopping teleoperation loop...")


def main() -> None:
    args = parse_args()

    # Only one scenario wired up for now (lift the blue cube) — see SCENE_OBJECTS above.
    scene_objects = SCENE_OBJECTS
    scene_joints: list = []

    cfg_path = Path(args.config).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    os.environ.setdefault("ICUB_LEROBOT_CONFIG", str(cfg_path))
    os.environ.setdefault("ICUB_MUJOCO_MODEL_PATH", str(model_path))
    os.environ.setdefault("ICUB_WORKSPACE_ROOT", str(repo_root.parent))

    # --robot-name (icub|icubSim) overrides the client-side robot_name (camera ports)
    # without touching the shared YAML. iCubConfig reads ICUB_ROBOT_NAME first.
    if args.robot_name:
        os.environ["ICUB_ROBOT_NAME"] = args.robot_name
        print(f"[play_gazebo] robot_name override -> {args.robot_name}")

    if args.vr_cable:
        from vr.vr_usb import connect_cable, CABLE_IP
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

    robot, teleop, selected_control, vr_enabled = _build_robot_and_teleop(
        args, cfg_path, model_path, scene_objects=scene_objects, scene_joints=scene_joints
    )

    robot.connect()
    teleop.connect()

    # Activar gaze y teleop en el state machine, igual que en common/yarp_module/teleop_module_sm.py
    time.sleep(0.5)
    robot.send_rpc_command("start_gaze")
    time.sleep(0.1)
    robot.send_rpc_command("look_at -0.5 0.0 0.3")
    time.sleep(0.1)
    robot.send_rpc_command("start_teleop")

    print("=======================================================")
    print(f"  iCub Gazebo/Real Teleop — Recording {'ON' if not args.no_record else 'OFF'}")
    print(f"  Config: {cfg_path}")
    print(f"  Model : {model_path}  (visualizer only)")
    print(f"  Arms  : {selected_control}")
    print(f"  VR    : {'ON' if vr_enabled else 'OFF'}")
    print(f"  Task  : {args.single_task}")
    print("=======================================================")

    try:
        if args.no_record:
            _simple_teleop_loop(args=args, robot=robot, teleop=teleop)
        else:
            repo_id = args.repo_id.strip()
            if "/" not in repo_id:
                repo_id = f"local/{repo_id}"

            if args.dataset_root:
                # El Hub pasa la ruta exacta para conocerla (curación/subida).
                dataset_root = Path(args.dataset_root).expanduser()
            else:
                base_root = Path(args.root).expanduser()
                run_suffix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                dataset_root = base_root / f"{repo_id.replace('/', '_')}_{run_suffix}"

            _manual_vr_record(
                args=args,
                repo_id=repo_id,
                dataset_root=dataset_root,
                robot=robot,
                teleop=teleop,
            )
    finally:
        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()
        if args.vr_cable:
            from vr.vr_usb import remove_reverse_ports, VR_PORTS, IMG_PORTS
            remove_reverse_ports(VR_PORTS)
            remove_reverse_ports(IMG_PORTS)


if __name__ == "__main__":
    main()
