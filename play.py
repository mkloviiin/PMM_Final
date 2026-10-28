#!/usr/bin/env python3
"""Unified launcher for teleoperating and recording datasets with iCub across MuJoCo, Gazebo, and Real Robot.

Supported Backends:
  - mujoco: Pure MuJoCo physics simulation driven by TSIK (QP IK) or TSID (QP Inverse Dynamics / Impedance).
  - gazebo: Gazebo Sim + YARP simulation connected via BLFRobotBridge / teleop_module_sm.
  - real: Real iCub robot connected via BLFRobotBridge / teleop_module_sm.

Supported Controllers:
  - ik: Task-Space Inverse Kinematics (blf.ik.QPFixedBaseInverseKinematics)
  - id: Task-Space Inverse Dynamics (blf.tsid.QPFixedBaseTSID) with optional Cartesian impedance (--impedance)

Both backends share one record loop (_record_loop). Episodes start/stop with A/B on the VR
or 1/2/3 on the console; the Hub (src/hub_dataset/launcher_gui/robots/icub_rg.py, _bridge_play)
writes those commands to stdin and reads the "[Manual] ..." and "[Live] {json}" lines.
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
    """Per-frame recorder of metrics/frame_metrics.py (per-episode metrics in the Hub logbook).
    Without it the session records the same, only without metrics."""
    try:
        from frame_metrics import FrameMetrics
        return FrameMetrics.create(**kwargs)
    except Exception as e:  # noqa: BLE001
        print(f"[Metrics] frame metrics unavailable ({e}): recording without metrics.", flush=True)
        return _NoMetrics()


def _discard_episode(dataset) -> None:
    """Drop the episode being recorded (B held on the VR) so it can be recorded again.

    LeRobot >= 0.6: clear_episode_buffer() removes the temporary frames of every camera
    (image and video) and cancels the streaming encoder. LeRobot 0.5 only removed the 'image'
    cameras' PNGs: the 'video' ones (images/<cam>/episode-N, written before encoding) are
    removed here, otherwise re-recording the same episode mixes frames of the discarded attempt.
    """
    import shutil

    if not hasattr(dataset, "clear_episode_buffer"):
        if hasattr(dataset, "clear_episode"):
            dataset.clear_episode()
        return

    ep = (getattr(dataset, "episode_buffer", None) or {}).get("episode_index", dataset.num_episodes)
    if hasattr(ep, "size"):
        ep = ep.item() if ep.size == 1 else ep[0]
    dataset.clear_episode_buffer()

    image_dir = getattr(dataset, "_get_image_file_dir", None)  # LeRobot 0.5 only
    if image_dir is not None:
        for key in getattr(dataset.meta, "video_keys", []):
            img_dir = image_dir(int(ep), key)
            if img_dir.is_dir():
                shutil.rmtree(img_dir)


def _load_scene_entries() -> list[dict]:
    import yaml

    if not SCENES_YAML.exists():
        return []
    with open(SCENES_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or []


def _resolve_scene(scene_arg: str | int | None, controller: str = "ik",
                   model_file: str | None = None) -> tuple[Path, str, list, list, list]:
    """(model path, task, objects, joints, cameras) of a scenes.yaml entry, by 1-based index or
    label, or else by the scene XML file name (model_file, e.g. --model from the Hub).

    The ID controller loads icub_visuomanip_idyn_limits.xml instead of the ikin robot model
    (MuJoCoTeleop swaps the include, see load_mujoco_model_for_controller)."""
    scenes_data = _load_scene_entries()

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

    if selected is None and model_file:
        selected = next((s for s in scenes_data if s.get("file") == Path(model_file).name), None)
    if selected is None and scenes_data:
        selected = scenes_data[0]

    if selected:
        xml_name = selected.get("file", "scene_icub_lift.xml")
        task = selected.get("task", "lift the blue block")
        objects = selected.get("objects", [])
        joints = selected.get("joints", [])
        cameras = selected.get("cameras", [])
    else:
        xml_name = "scene_icub_lift.xml"
        task = "lift the blue block"
        objects = [{"body": "blue-cube"}]
        joints = []
        cameras = []

    model_path = repo_root / "mujoco_sim" / "assets" / "scenes" / xml_name
    return model_path, task, objects, joints, cameras



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
        choices=["right", "left", "both", "auto"],
        default=None,
        help="Active arms (default / auto: detect from the active teleop module or control_config.yaml)",
    )
    parser.add_argument(
        "--cameras",
        default=None,
        help="Comma-separated cameras of the task (cameras in scenes.yaml, e.g. left,right,top). "
             "Default: the scene's cameras, else the robot's (left, right, front)",
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
    if explicit_arms and explicit_arms != "auto":
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


# ── Backend sessions: what differs between MuJoCo and Gazebo / real in the record loop ──────
# Video encoding without a process pool: with > 1 camera LeRobot forks to encode in parallel,
# and forking a process with live threads (YARP, EGL renderers, ZMQ, CUDA) leaves the children
# stuck on an inherited lock: the episode is never saved.
_SAVE_KWARGS = {"parallel_encoding": False}


class _MuJoCoSession:
    backend = "mujoco"

    def __init__(self, args, robot, teleop):
        self.args, self.robot, self.teleop = args, robot, teleop
        self.vr_enabled = bool(robot.config.vr_enabled)

    def closed(self) -> bool:
        viewer = getattr(self.teleop, "viewer", None)
        if viewer is not None and not viewer.is_running():
            print("[Viewer] Window closed by user. Exiting...")
            return True
        return False

    def reset_world(self) -> None:
        self.teleop._reset_scenario()
        poses = getattr(self.teleop, "_last_reset_poses", None)
        if self.robot.teleop_core and poses:
            self.robot.teleop_core.set_object_poses(poses)

    def poll_world_reset(self, requested: bool) -> None:
        if requested or self.teleop.consume_world_reset_event():
            print("\n[Teleop] Reset scenario: re-randomizing scene objects...")
            self.reset_world()

    def vr_status(self, *lines: str, push: bool = False) -> None:
        """Recording status in the top-right corner of the big VR screen (no accents:
        cv2.putText). push=True sends a frame now: save_episode blocks send_action."""
        tc = self.robot.teleop_core
        if tc is None:
            return
        tc.vr_status_lines = lines
        if push and self.vr_enabled:
            tc._stream_images(force=True)

    def forward_vr_help(self) -> None:
        pass  # MuJoCoTeleop draws the help itself

    def vr_record_events(self) -> tuple[bool, bool, bool]:
        events = self.robot.consume_vr_record_events()
        return events if any(events) else self.teleop.consume_vr_record_events()

    def recorded_action(self, sent_action: dict, act: dict) -> dict:
        # With VR the robot moves the targets inside send_action and returns what it applied
        # (already in the root frame and canonicalized): the observation then carries the
        # previous command, as in Gazebo.
        if self.vr_enabled:
            return sent_action
        from lerobot_robot_icub.icub_mujoco import canonicalize_quats
        if self.robot.config.canonical_quats and self.robot.config.pose_frame == "root":
            return canonicalize_quats(dict(act))
        return act

    def toggle_teleop(self) -> None:
        state = self.robot.toggle_teleop()
        print(f"\n[Teleop] Teleoperation: {'ACTIVE' if state else 'IDLE (paused)'}")

    def go_home(self) -> None:
        self.teleop._reset_mocaps_to_xml_pose(strict=True)
        self.robot.go_home()

    def on_episode_start(self) -> None:
        pass


class _YarpSession:
    def __init__(self, args, robot, teleop):
        self.args, self.robot, self.teleop = args, robot, teleop
        self.backend = args.backend
        self.vr_enabled = bool(args.vr or args.vr_ip)
        self._vr_help_sent = None

    def closed(self) -> bool:
        return False

    def reset_world(self) -> None:
        # The teleoperator only resets its MuJoCo mirror (empty table): the objects live in
        # Gazebo, where the YARP module runs gazebo_sim/scripts/reset_objects.sh (asynchronous).
        if self.backend == "gazebo":
            reply = self.robot.send_rpc_command("reset_scenario")
            print(f"[Gazebo] reset_scenario -> {reply}", flush=True)

    def poll_world_reset(self, requested: bool) -> None:
        if requested or self.teleop.consume_world_reset_event():
            self.reset_world()

    def vr_status(self, *lines: str, push: bool = False) -> None:
        """Drawn by teleop_module_sm (RPC vr_status): big-screen corner in Gazebo, small screen
        when there is no front camera (real robot). No accents: cv2.putText."""
        text = "|".join(lines).replace('"', "'")
        self.robot.send_rpc_command(f'vr_status "{text}"')

    def forward_vr_help(self) -> None:
        """The teleoperator detects the paused app (or X); teleop_module_sm draws the panel.
        The RPC is only sent when it changes."""
        if not self.vr_enabled or not hasattr(self.teleop, "vr_help_active"):
            return
        active = self.teleop.vr_help_active()
        if active != self._vr_help_sent:
            self.robot.send_rpc_command(f"vr_help {int(active)}")
            self._vr_help_sent = active

    def vr_record_events(self) -> tuple[bool, bool, bool]:
        # VR record events (A/B) come from the teleoperator: the robot never sees the VR stream
        return self.teleop.consume_vr_record_events()

    def recorded_action(self, sent_action: dict, act: dict) -> dict:
        from lerobot_robot_icubrg.icub import canonicalize_quats
        return canonicalize_quats(dict(act)) if self.robot.config.canonical_quats else act

    def toggle_teleop(self) -> None:
        if self.robot.is_teleop_active:
            self.robot.stop_teleop()
            print("\n[Teleop] Status: STOPPED (holding current posture)\n")
        else:
            self.robot.start_teleop()
            print("\n[Teleop] Status: ACTIVE (tracking targets)\n")

    def go_home(self) -> None:
        self.teleop._reset_mocaps_to_xml_pose()
        self.robot.go_home()
        print("[Teleop] Status: IDLE (robot at home - press 't' to resume teleoperation)\n")

    def on_episode_start(self) -> None:
        self.robot.reset_command_state()


def _console_commands() -> list[str]:
    """Commands typed on the console (or written to stdin by the Hub), read in a thread."""
    queue: list[str] = []

    def _reader() -> None:
        while True:
            try:
                line = input()
            except EOFError:
                break
            except Exception:  # noqa: BLE001
                continue
            if line.strip():
                queue.append(line.strip())

    threading.Thread(target=_reader, daemon=True).start()
    return queue


class _LivePreview:
    """Hub session without VR: the recorded robot-eye and scene cameras to the Hub's live view
    (ZMQ, PREVIEW_PORTS, ~15 fps). With VR the Hub shows the headset streams instead."""

    PERIOD_S = 0.06  # every 2nd step of the 30 Hz loop: ~15 fps

    def __init__(self, cameras: list[str]):
        import cv2
        import numpy as np
        from interfaces.zmq_image_publisher import PREVIEW_PORTS, ZMQCompressedImageTransmitter

        self._cv2, self._np = cv2, np
        self._pubs = []
        self._last = 0.0
        eye = next((c for c in ("left", "right") if c in cameras), None)
        scene = next((c for c in ("front", "top") if c in cameras), None)
        for view, cam in (("robot", eye), ("scene", scene)):
            if cam is None:
                continue
            try:
                pub = ZMQCompressedImageTransmitter(host="*", port=PREVIEW_PORTS[view], quality=70)
                self._pubs.append((cam, pub))
            except Exception as e:  # noqa: BLE001
                print(f"[Preview] '{cam}' camera not sent to the Hub: {e}")

    def send(self, obs: dict) -> None:
        now = time.perf_counter()
        if not self._pubs or now - self._last < self.PERIOD_S:
            return
        self._last = now
        for cam, pub in self._pubs:
            img = obs.get(cam)
            if img is None:
                continue
            try:
                pub.send_image(self._cv2.cvtColor(self._np.asarray(img), self._cv2.COLOR_RGB2BGR))
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        for _, pub in self._pubs:
            pub.stop()


def _record_loop(args, session, *, scene_task: str, cameras: list[str]) -> None:
    """Teleoperation loop shared by every backend; records a LeRobot dataset unless --no-record.

    Console: 1 = START, 2 = STOP/SAVE, 3 = EXIT, h = HOME, n = RESET objects, t = TOGGLE teleop,
    i = TOGGLE impedance, 8/9 = right hand open/close, 6/7 = left hand open/close.
    VR: A = START, B = STOP/SAVE (held 1.5 s = DISCARD), Y = RESET objects."""
    from lerobot.processor import make_default_processors
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.robot_utils import precise_sleep
    try:
        from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
    except ImportError:
        from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts

    robot, teleop = session.robot, session.teleop
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset = dataset_root = None
    repo_id = args.repo_id if "/" in args.repo_id else f"local/{args.repo_id}"
    if not args.no_record:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features

        has_cameras = bool(getattr(robot, "_cameras_ft", None))
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
        dataset_root = Path(args.dataset_root or Path(args.root) / (
            f"{repo_id.replace('/', '_')}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"))
        dataset = LeRobotDataset.create(
            repo_id, args.fps, root=dataset_root, robot_type=robot.name,
            features=dataset_features, use_videos=has_cameras,
        )
        print(f"[Manual] dataset root: {dataset_root}")

    fm = _make_metrics(
        backend=session.backend, fps=args.fps,
        name=dataset_root.name if dataset_root else "no_record",
        repo_id=repo_id, task=scene_task, vr=session.vr_enabled,
        robot_name=os.environ.get("ICUB_ROBOT_NAME", ""),
        scene=Path(args.model).name if args.model else None,
        controller=args.controller, impedance=bool(args.impedance),
        control_arms=getattr(robot.config, "control_arms", None), cameras=list(cameras),
    )
    # The Hub reads these lines for its "Real-time Metrics" panel
    live_every = max(int(round(args.fps / 2)), 1) if os.environ.get("PMM_LIVE_METRICS") == "1" else 0

    print("[Controls] Console: 1 = START, 2 = STOP/SAVE, 3 = EXIT, h = HOME, n = RESET objects, "
          "t = TOGGLE teleop, i = TOGGLE impedance, 8/9 = Right Hand, 6/7 = Left Hand")
    if session.vr_enabled:
        print("[Controls] VR: A = START episode, B = STOP/SAVE (hold B 1.5 s to DISCARD), Y = RESET objects, "
              "Index trigger = hand, Lower trigger = clutch, X = help")

    commands = _console_commands()
    recorded = 0
    recording = False
    frame_i = 0
    episode_start = 0.0
    n_eps = args.num_episodes
    preview = (_LivePreview(cameras) if os.environ.get("PMM_LIVE_VIEW") == "1" and not session.vr_enabled
               else None)

    def _wait_status(title: str = "EN ESPERA") -> None:
        print(f"\n[Manual] Waiting START for episode {dataset.num_episodes}...", flush=True)
        session.vr_status(title, f"Episodios grabados: {recorded}/{n_eps}", "A = grabar")

    if dataset is not None:
        _wait_status()
    try:
        while dataset is None or recorded < n_eps:
            loop_t0 = time.perf_counter()
            if session.closed():
                break

            cmd = commands.pop(0).lower() if commands else ""
            if cmd == "3":
                print("[Exit] Quitting...")
                break
            if cmd in ("t", "teleop"):
                session.toggle_teleop()
            elif cmd in ("h", "home"):
                print("\n[Teleop] Home requested: moving the robot smoothly to home...")
                session.go_home()
            elif cmd in ("i", "impedance"):
                robot.toggle_impedance()
            elif cmd in ("8", "9", "6", "7"):
                side, level = {"8": ("rh", 0.0), "9": ("rh", 1.0), "6": ("lh", 0.0), "7": ("lh", 1.0)}[cmd]
                teleop.set_gripper(side, level)
                print(f"[Console] {'Right' if side == 'rh' else 'Left'} hand: {'CLOSE' if level else 'OPEN'}")
            session.poll_world_reset(cmd in ("n", "reset"))

            fm.begin(recording=recording, episode=dataset.num_episodes if dataset is not None else 0)
            obs = robot.get_observation()
            fm.mark("obs")
            teleop.send_feedback(obs)
            act = teleop.get_action()
            act_processed = teleop_action_processor((act, obs))
            robot_action_to_send = robot_action_processor((act_processed, obs))
            fm.mark("teleop")
            sent_action = robot.send_action(robot_action_to_send)
            fm.mark("send")
            session.forward_vr_help()
            recorded_action = session.recorded_action(sent_action, act_processed)

            start_ev, stop_ev, discard_ev = session.vr_record_events()
            if dataset is not None and recording:
                obs_processed = robot_observation_processor(obs)
                frame = {
                    **build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR),
                    **build_dataset_frame(dataset.features, recorded_action, prefix=ACTION),
                    "task": scene_task,
                }
                dataset.add_frame(frame)
                fm.mark("record")
            fm.end(robot=robot, obs=obs, action=recorded_action)
            if preview is not None:
                preview.send(obs)

            if dataset is not None and recording:
                frame_i += 1
                if live_every and frame_i % live_every == 0:
                    live = fm.live()
                    if live:
                        print("[Live] " + json.dumps(live), flush=True)
                if args.episode_time_s > 0 and time.perf_counter() - episode_start >= args.episode_time_s:
                    print(f"[Record] Episode {dataset.num_episodes} reached the time limit.")
                    stop_ev = True

                if discard_ev:
                    # Shown right away (B is still held: the operator can let go now) and kept
                    # until the next A
                    session.vr_status("DESCARTADO", "Suelta B", push=True)
                    print("[Manual] Episode DISCARDED (B held) - re-recording...", flush=True)
                    fm.end_episode(dataset.num_episodes, saved=False, reason="discarded")
                    _discard_episode(dataset)
                    recording = False
                    session.reset_world()
                    _wait_status("DESCARTADO")
                elif stop_ev or cmd == "2":
                    session.vr_status("GUARDANDO...", f"Episodio {recorded + 1}/{n_eps}", push=True)
                    ep_idx = dataset.num_episodes
                    t_save = time.perf_counter()
                    dataset.save_episode(**_SAVE_KWARGS)
                    fm.end_episode(ep_idx, saved=True, save_ms=(time.perf_counter() - t_save) * 1000.0)
                    recorded += 1
                    recording = False
                    print(f"[Manual] Episode saved ({recorded}/{n_eps})", flush=True)
                    if recorded >= n_eps:
                        session.vr_status("SESION COMPLETA", f"Episodios grabados: {recorded}/{n_eps}", push=True)
                        break
                    session.reset_world()
                    _wait_status()
            elif dataset is not None and (start_ev or cmd == "1"):
                recording = True
                frame_i = 0
                episode_start = time.perf_counter()
                session.on_episode_start()
                print(f"[Manual] Recording episode {dataset.num_episodes}...", flush=True)
                session.vr_status("GRABANDO", f"Episodio {recorded + 1}/{n_eps}",
                                  f"Episodios grabados: {recorded}", "B = guardar")

            precise_sleep(max(1.0 / args.fps - (time.perf_counter() - loop_t0), 0.0))
    finally:
        fm.close()
        if preview is not None:
            preview.close()
        if dataset is not None:
            if recording:
                # Unfinished episode (EXIT / window closed): not saved, like a discard
                _discard_episode(dataset)
            if recorded == 0:
                import shutil
                shutil.rmtree(dataset_root, ignore_errors=True)
                print(f"[Manual] Nothing was recorded - folder removed: {dataset_root}")
            else:
                try:
                    dataset.finalize()
                except Exception:  # noqa: BLE001
                    pass


def _resolve_cameras(args, scene_cameras: list) -> list[str]:
    if args.cameras:
        return [c.strip() for c in args.cameras.split(",") if c.strip()]
    return list(scene_cameras or [])


def run_mujoco_teleop_loop(args, model_path: Path, scene_task: str, scene_objects: list,
                           scene_joints: list, scene_cameras: list):
    args.control_arms = _resolve_control_arms(args.control_arms, args.config)
    from lerobot_robot_icub.config_icub_mujoco import iCubMuJoCoConfig
    from lerobot_robot_icub.icub_mujoco import iCubMuJoCo
    from lerobot_teleoperator_icubteleop.config_icubteleop import iCubTeleopConfig
    from lerobot_teleoperator_icubteleop.icubteleop import iCubTeleop

    robot_cfg = iCubMuJoCoConfig(
        model_path=str(model_path),
        config_path=str(args.config),
        control_arms=args.control_arms,
        controller_type=args.controller,
        impedance_mode=args.impedance,
        control_dt=1.0 / max(1, int(args.fps)),
        vr_enabled=bool(args.vr or args.vr_ip),
        vr_ip=args.vr_ip,
        scene_objects=scene_objects,
        scene_joints=scene_joints,
    )
    # Task cameras (cameras in scenes.yaml), e.g. the door: eyes + overhead
    cameras = _resolve_cameras(args, scene_cameras)
    if cameras:
        robot_cfg.camera_names = cameras
    teleop_cfg = iCubTeleopConfig(
        model_path=str(model_path),
        config_path=str(args.config),
        control_arms=args.control_arms,
        vr_enabled=False,  # the VR goes to the robot (MuJoCoTeleop core), the viewer only mirrors
        scene_objects=scene_objects,
        scene_joints=scene_joints,
    )

    robot = iCubMuJoCo(robot_cfg)
    teleop = iCubTeleop(teleop_cfg)

    print("\n=======================================================")
    print(" Starting iCub Teleoperation (Backend: MuJoCo)")
    print(f" Controller: {args.controller.upper()} {'(Impedance Mode)' if args.impedance else ''}")
    print(f" Model:      {model_path.name}")
    print(f" Control:    {args.control_arms} arm(s)")
    print(f" Cameras:    {list(robot_cfg.camera_names)}")
    print(f" Recording:  {'DISABLED (--no-record)' if args.no_record else 'ENABLED'}")
    print("=======================================================\n")

    # If a connection fails, release what was connected (the robot opens the VR and stream ports)
    try:
        robot.connect()
        teleop.connect()
    except BaseException:
        for dev in (teleop, robot):
            if dev.is_connected:
                dev.disconnect()
        raise

    session = _MuJoCoSession(args, robot, teleop)
    session.reset_world()
    try:
        _record_loop(args, session, scene_task=scene_task, cameras=list(robot_cfg.camera_names))
    finally:
        if getattr(robot, "is_teleop_active", False) and hasattr(robot, "stop_teleop"):
            robot.stop_teleop()
        for dev in (teleop, robot):
            try:
                dev.disconnect()
            except Exception:  # noqa: BLE001
                pass
        print("[Play] Session ended cleanly.")


def run_yarp_teleop_loop(args, model_path: Path, scene_task: str, scene_cameras: list):
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
    cameras = _resolve_cameras(args, scene_cameras)
    if cameras:
        # Task cameras (cameras in scenes.yaml): e.g. the door records the eyes and the overhead one
        missing = robot_cfg.select_cameras(cameras)
        if missing:
            print(f"[play.py][warn] cameras not available on this robot, not recorded: {missing}")
    teleop_kwargs = dict(
        config_path=str(args.config),
        control_arms=args.control_arms,
        vr_enabled=bool(args.vr or args.vr_ip),
        vr_ip=args.vr_ip,
    )
    if args.model:
        # MuJoCo mirror of the robot: the Hub passes the empty table (objects live in Gazebo / real world)
        teleop_kwargs["model_path"] = str(model_path)
    teleop_cfg = iCubTeleopConfig(**teleop_kwargs)

    robot = iCub(robot_cfg)
    teleop = iCubTeleop(teleop_cfg)

    print("\n=======================================================")
    print(f" Starting iCub Teleoperation (Backend: {args.backend.upper()})")
    print(f" Robot Name: {robot_name}")
    print(f" Controller: {args.controller.upper()} {'(Impedance Mode)' if args.impedance else ''}")
    print(f" Control:    {args.control_arms} arm(s)")
    print(f" Cameras:    {list(robot_cfg.camera_ports)}")
    print(f" Recording:  {'DISABLED (--no-record)' if args.no_record else 'ENABLED'}")
    print("=======================================================\n")

    # If teleop_module_sm is not running on YARP (the Hub starts it in step 2.2), launch it
    # automatically in background. Its RPC port tells whether it is up.
    import yarp
    yarp.Network.init()
    module_port = "/teleop/rpc:i"
    teleop_proc = None
    if not yarp.Network.exists(module_port):
        print(f"[play.py] {module_port} not found. Auto-launching teleop_module_sm.py ({args.controller.upper()})...")
        teleop_cmd = [
            sys.executable,
            str(repo_root / "src" / "interfaces" / "teleop_module_sm.py"),
            "--robot", robot_name,
            "--controller", args.controller,
            "--control-arms", args.control_arms,
        ]
        if args.impedance:
            teleop_cmd.append("--impedance")
        teleop_proc = subprocess.Popen(teleop_cmd)
        print(f"[play.py] Waiting for {module_port} to appear on YARP network...")
        for _ in range(120):
            if yarp.Network.exists(module_port):
                print(f"[play.py] {module_port} is ready.")
                break
            time.sleep(0.5)
        else:
            print(f"[play.py] Warning: Timeout waiting for {module_port}. Proceeding to connect...")

    try:
        robot.connect()
        teleop.connect()
        # The YARP module starts idle: start tracking the targets right away
        time.sleep(0.5)
        robot.start_teleop()

        session = _YarpSession(args, robot, teleop)
        _record_loop(args, session, scene_task=scene_task, cameras=list(robot_cfg.camera_ports))
    finally:
        if getattr(robot, "is_teleop_active", False) and hasattr(robot, "stop_teleop"):
            try:
                robot.stop_teleop()
            except Exception:  # noqa: BLE001
                pass
        for dev in (teleop, robot):
            try:
                if dev.is_connected:
                    dev.disconnect()
            except Exception:  # noqa: BLE001
                pass
        if teleop_proc is not None:
            print("[play.py] Terminating background teleop_module_sm...")
            teleop_proc.terminate()
            try:
                teleop_proc.wait(timeout=3)
            except Exception:  # noqa: BLE001
                teleop_proc.kill()
        print("[Play] Session ended cleanly.")


def main():
    args = parse_args()
    if args.vr_cable:
        from interfaces.vr_usb import connect_cable, CABLE_IP
        print("[VR] Connecting the Meta Quest over USB (ADB reverse)...")
        if not connect_cable():
            raise SystemExit(
                "\n[ERROR] USB connection with the Meta Quest failed.\n"
                "Check the headset is plugged in over USB-C and turned on, and that\n"
                "'Allow USB debugging' was accepted inside the headset.\n"
            )
        args.vr_ip = CABLE_IP
        args.vr = True

    vr_enabled = bool(args.vr or args.vr_ip)
    os.environ["ICUB_MUJOCO_VR_ENABLED"] = "1" if vr_enabled else "0"
    if args.vr_ip:
        os.environ["ICUB_MUJOCO_VR_IP"] = str(args.vr_ip)
    if args.impedance and args.controller != "id":
        print("[play.py] --impedance only applies to --controller id: ignored.")
        args.impedance = False

    model_path, scene_task, scene_objects, scene_joints, scene_cameras = _resolve_scene(
        args.scene, controller=args.controller, model_file=args.model)
    if args.model:
        model_path = Path(args.model)
    if args.single_task:
        scene_task = args.single_task

    try:
        if args.backend == "mujoco":
            run_mujoco_teleop_loop(args, model_path, scene_task, scene_objects, scene_joints, scene_cameras)
        else:
            run_yarp_teleop_loop(args, model_path, scene_task, scene_cameras)
    finally:
        if args.vr_cable:
            try:
                from interfaces.vr_usb import remove_reverse_ports
                remove_reverse_ports()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    main()
