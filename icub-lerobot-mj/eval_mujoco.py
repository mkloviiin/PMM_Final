#!/usr/bin/env python3
"""
eval_mujoco.py – Run a trained SmolVLA policy on the iCub MuJoCo simulation.

The robot acts autonomously based on visual + state observations fed to SmolVLA.

Usage (smolvla, from icub-lerobot-mj/):
    conda activate smolvla
    python eval_mujoco.py --policy-repo mklovin/my_smolvla

Optional flags:
    --task "Pick up the blue cube"   Task instruction for the policy
    --n-episodes 5                   Number of evaluation episodes
    --episode-steps 300              Max steps per episode (at --fps Hz)
    --fps 30
    --scene 1                        1=lift, 2=stack, 3=door (con pestillo)
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
import yaml

# ── Inject local robot / teleop plugins ─────────────────────────────────────
repo_root = Path(__file__).resolve().parent

local_robot_path = repo_root / "lerobot-robot-icub"
if local_robot_path.exists():
    sys.path.insert(0, str(local_robot_path))

local_teleop_path = repo_root / "lerobot-teleoperator-icubteleop"
if local_teleop_path.exists():
    sys.path.insert(0, str(local_teleop_path))

# Also expose icubenv packages (MuJoCo bindings, dependencies, etc.)
_icubenv_sp = os.environ.get("ICUBENV_SITE_PACKAGES", os.path.expanduser(
    "~/miniconda3/envs/icubenv/lib/python3.12/site-packages"
))
if _icubenv_sp not in sys.path:
    sys.path.append(_icubenv_sp)

from dependencies.door_scene import handle_side, place_door, set_handle_side  # noqa: E402


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
        default=None,
        help="Task instruction fed to the language-conditioned policy (default: the scene's "
             "task in scenes.yaml, e.g. 'open the door' for --scene 3)",
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
        choices=[1, 2, 3],
        help="1=lift cube, 2=stack cubes, 3=open the door (latch; success = door_hinge > 0.3 rad)",
    )
    p.add_argument(
        "--control-arms",
        choices=["right", "left", "both"],
        default=None,
        help="Arms to control (default: inferred from the training dataset action names, "
             "else control_arms in control_config.yaml)",
    )
    p.add_argument(
        "--cube-pos",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Fixed blue-cube spawn position (default: random within cube_spawn of "
             "control_config.yaml, as during recording)",
    )
    p.add_argument(
        "--door-xy",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="--scene 3: moves the door so the center of its panel is at (X, Y) in the MuJoCo "
             "world (table center = 0.5 0). Default: XML position, panel at (0.47, -0.13)",
    )
    p.add_argument(
        "--door-side",
        choices=["right", "left", "alternate"],
        default=None,
        help="--scene 3: side of the handle (default: XML, left). alternate = right in odd "
             "episodes, left in even ones (dataset local_door_both_sides)",
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
        help="Minimum z (MuJoCo world, m) for rh/lh targets during eval; converted to the robot's "
             "pose frame",
    )
    p.add_argument(
        "--camera-source",
        nargs="*",
        default=[],
        metavar="TRAINED=SOURCE",
        help="Ablation: feed the policy camera SOURCE where it expects TRAINED, e.g. "
             "head_cam_track_hand=head_cam (Gazebo has no hand-tracking camera)",
    )
    p.add_argument(
        "--legacy", "--legacy-head-cams",
        dest="legacy",
        action="store_true",
        help="Policy trained before 2026-10-20 (e.g. mujocotrainPMM): world-frame poses, the old "
             "head cameras / camera names and the old gripper convention (0 open / 0.5 close)",
    )
    p.add_argument(
        "--zero-touch",
        action="store_true",
        help="Ablation: rh_touch_* always 0, like Gazebo's iCub (no skin)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the cube spawn, to compare runs on the same positions",
    )
    p.add_argument(
        "--min-gaze-z",
        type=float,
        default=None,
        help="Minimum z for gaze mocap target during eval (default: no clamp; "
             "trained gaze_z is mostly 0.1-0.5, so a clamp pushes it out of distribution)",
    )
    p.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Actions executed per predicted chunk before replanning (default: the trained "
             "n_action_steps, e.g. 60). Lower = more closed-loop",
    )
    p.add_argument(
        "--realtime",
        action="store_true",
        help="Advance the physics by the wall time each step really took (inference included), "
             "holding the previous targets, like Gazebo and the real robot. Default: the sim "
             "waits for the policy (exactly 1/fps per step)",
    )
    return p.parse_args()


REALTIME_MAX_GAP_S = 1.0


def advance_physics(robot, seconds: float) -> None:
    """Avanza la simulación `seconds` con los targets actuales (mismo bucle que send_action)."""
    tc = robot.teleop_core
    for i in range(int(round(seconds / tc.model.opt.timestep))):
        if i % max(1, tc.frame_skip) == 0:
            tc._solve_ik()
            tc._interpolate_hands()
        mujoco.mj_step(tc.model, tc.data)


class _Tee:
    """Copia stdout/stderr a un archivo (eval.log del run), para revisar la corrida después."""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, data):
        self.stream.write(data)
        self.log.write(data)
        self.log.flush()

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def log_to(path: Path) -> None:
    log = open(path, "a", buffering=1)
    sys.stdout, sys.stderr = _Tee(sys.stdout, log), _Tee(sys.stderr, log)
    print(f"[eval] command: {' '.join(sys.argv)}")


# ── Observation → policy batch ───────────────────────────────────────────────

# Cámaras de los modelos --legacy: nombres viejos, y las de cabeza en su pose antigua (ver dependencies/assets/scenes/icub_visuomanip_ikin_limits.xml)
LEGACY_CAMERA_NAMES = ["head_cam", "head_cam_track_hand", "frontview"]
LEGACY_CAMERA_SOURCES = {"head_cam": "head_cam_v1", "head_cam_track_hand": "head_cam_track_hand_v1"}

SCENES_DIR = repo_root / "dependencies" / "assets" / "scenes"
SCENE_FILES = {1: "scene_icub_lift.xml", 2: "scene_icub_stack.xml", 3: "scene_icub_door.xml"}
# Puerta: éxito como robosuite (Door._check_success), bisagra sobre 0.3 rad en algún paso
DOOR_SUCCESS_RAD = 0.3

FULL_ACTION_ORDER = [
    "rh_pos_x", "rh_pos_y", "rh_pos_z",
    "rh_quat_w", "rh_quat_x", "rh_quat_y", "rh_quat_z",
    "rh_gripper",
    "lh_pos_x", "lh_pos_y", "lh_pos_z",
    "lh_quat_w", "lh_quat_x", "lh_quat_y", "lh_quat_z",
    "lh_gripper",
    "gaze_x", "gaze_y", "gaze_z",
]


def build_raw_observation_dict(
    obs: dict, device: str, camera_names, state_names: list[str] | None = None
) -> dict:
    """
    Raw iCub-MuJoCo observation dict -> policy input batch (images and
    observation.state), before the preprocessor normalizes and tokenizes.

    If `state_names` (from the training dataset) is given, the state vector is
    built in that exact order; otherwise the observation dict order is used.
    """
    batch: dict = {}
    state_vals: list[float] = []

    for key, val in obs.items():
        if key in camera_names:
            # val is an (H, W, C) uint8 image (like the Gazebo robot); the policy wants
            # (C, H, W) floats in [0, 1]
            img = torch.from_numpy(np.ascontiguousarray(val)).permute(2, 0, 1).float() / 255.0
            batch[f"observation.images.{key}"] = img.unsqueeze(0).to(device)
        elif state_names is None:
            # Scalar observation feature (joint pos, vel, torque, EE pose, …)
            if isinstance(val, torch.Tensor):
                state_vals.append(val.item())
            else:
                state_vals.append(float(val))

    if state_names is not None:
        state_vals = [to_float(obs[name]) for name in state_names]

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


def read_trained_feature_names(policy_repo: str) -> tuple[list[str] | None, list[str] | None]:
    """State/action names of the training dataset (via train_config.json → dataset root)."""
    try:
        cfg = json.loads((Path(policy_repo).expanduser() / "train_config.json").read_text())
        info_path = Path(cfg["dataset"]["root"]).expanduser() / "meta" / "info.json"
        feats = json.loads(info_path.read_text())["features"]
        return feats["observation.state"]["names"], feats["action"]["names"]
    except Exception:
        return None, None


def read_trained_stats(policy_repo: str) -> dict:
    """meta/stats.json del dataset de entrenamiento ({} si no está)."""
    try:
        cfg = json.loads((Path(policy_repo).expanduser() / "train_config.json").read_text())
        return json.loads((Path(cfg["dataset"]["root"]).expanduser() / "meta" / "stats.json").read_text())
    except Exception:
        return {}


def infer_control_arms(action_names: list[str] | None) -> str | None:
    if not action_names:
        return None
    has_r = any(n.startswith("rh_") for n in action_names)
    has_l = any(n.startswith("lh_") for n in action_names)
    if has_r and has_l:
        return "both"
    return "right" if has_r else "left" if has_l else None


def load_scene_entry(xml_name: str) -> dict:
    """Entry of scenes.yaml for this XML (objects/joints to reset, default task)."""
    try:
        entries = yaml.safe_load((SCENES_DIR / "scenes.yaml").read_text()) or []
    except Exception:
        return {}
    return next((e for e in entries if e.get("file") == xml_name), {})


def actions_tensor_to_dict(
    actions: torch.Tensor, action_keys: list[str], action_names: list[str] | None = None
) -> dict[str, float]:
    """
    Convert the 1-D action tensor returned by `policy.select_action`
    into the dict format expected by `robot.send_action`.
    """
    a = actions.detach().cpu().numpy().flatten()

    if action_names and len(action_names) == len(a):
        named = {name: float(a[idx]) for idx, name in enumerate(action_names)}
        return {key: named.get(key, 0.0) for key in action_keys}

    if len(a) == len(FULL_ACTION_ORDER):
        full_action = {name: float(a[idx]) for idx, name in enumerate(FULL_ACTION_ORDER)}
        return {key: full_action.get(key, 0.0) for key in action_keys}

    return {key: float(a[i]) for i, key in enumerate(action_keys) if i < len(a)}


def clamp_action_targets(
    action_dict: dict[str, float], min_hand_z: float, min_gaze_z: float | None
) -> bool:
    clamped = False

    for key in ("rh_pos_z", "lh_pos_z"):
        if key in action_dict and action_dict[key] < min_hand_z:
            action_dict[key] = min_hand_z
            clamped = True

    if min_gaze_z is not None and "gaze_z" in action_dict and action_dict["gaze_z"] < min_gaze_z:
        action_dict["gaze_z"] = min_gaze_z
        clamped = True

    return clamped


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    policy_path = Path(args.policy_repo).expanduser()
    exp_name = policy_path.resolve().parents[2].name if policy_path.exists() else "policy"
    run_dir = repo_root.parent / "outputs" / "tests" / f"{exp_name}_mujoco_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_to(run_dir / "eval.log")

    # Same scene XML / objects as the hub uses for recording
    model_file = SCENES_DIR / SCENE_FILES[args.scene]
    if not model_file.exists():
        raise FileNotFoundError(f"MuJoCo scene not found: {model_file}")
    scene_entry = load_scene_entry(model_file.name)
    task = args.task or scene_entry.get("task", "lift the blue block")

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
    if args.n_action_steps is not None:
        if not 1 <= args.n_action_steps <= policy.config.chunk_size:
            raise SystemExit(f"--n-action-steps must be in 1..{policy.config.chunk_size}")
        policy.config.n_action_steps = args.n_action_steps  # policy.reset() rehace la cola
    print(f"[eval] n_action_steps={policy.config.n_action_steps} (chunk_size="
          f"{policy.config.chunk_size}) | {'realtime' if args.realtime else 'synchronous'} physics")
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

    from lerobot_robot_icub.config_icub_mujoco import _get_actuators_logic, load_control_config
    from lerobot_robot_icub.icub_mujoco import dataset_uses_canonical_quats

    state_names, action_names = read_trained_feature_names(args.policy_repo)
    if state_names is None:
        print("[eval][warn] Training dataset meta/info.json not found: state/action order "
              "falls back to the robot's feature order.")

    robot_cfg = iCubMuJoCoConfig(
        model_path=str(model_file),
        config_path=args.config,
        scene_objects=scene_entry.get("objects", []),
        scene_joints=scene_entry.get("joints", []),
    )
    # Arms used in training override control_arms from the YAML (read in __post_init__)
    control_arms = args.control_arms or infer_control_arms(action_names)
    if control_arms:
        ycfg = dict(load_control_config(args.config) or {})
        ycfg["control_arms"] = control_arms
        robot_cfg.control_arms = control_arms
        robot_cfg.actuators_to_use = _get_actuators_logic(ycfg)
    print(f"[eval] Control arms: {robot_cfg.control_arms}")
    if args.legacy:
        robot_cfg.pose_frame = "world"
        robot_cfg.gripper_legacy = True
        robot_cfg.camera_names = list(LEGACY_CAMERA_NAMES)
        robot_cfg.camera_sources = dict(LEGACY_CAMERA_SOURCES)
    else:
        # Las cámaras con que se entrenó (p. ej. la puerta: left/right/top, no front)
        policy_cams = [k.removeprefix("observation.images.") for k in policy.config.input_features
                       if k.startswith("observation.images.")]
        if policy_cams:
            robot_cfg.camera_names = policy_cams
    # Misma convención de signo de cuaterniones que el dataset de entrenamiento
    robot_cfg.canonical_quats = dataset_uses_canonical_quats(
        read_trained_stats(args.policy_repo), state_names or [], action_names or [])
    print(f"[eval] Quaternion sign: {'canonical' if robot_cfg.canonical_quats else 'raw (dataset before 2026-10-21)'}")
    trained_cams = list(robot_cfg.camera_names)
    # Ablación --camera-source: la cámara fuente también hay que renderizarla
    camera_sources = dict(pair.split("=", 1) for pair in args.camera_source)
    robot_cfg.camera_names = list(dict.fromkeys([*trained_cams, *camera_sources.values()]))
    robot = make_robot_from_config(robot_cfg)
    robot.connect()
    print("[eval] Robot connected ✓")
    if args.seed is not None:
        np.random.seed(args.seed)
    print(f"[eval] Pose frame: {robot_cfg.pose_frame} | cameras: {trained_cams}")
    if camera_sources:
        print(f"[eval][ablation] Cameras (trained <- source): {camera_sources}")
    # --min-hand-z / --min-gaze-z vienen en el mundo MuJoCo; las acciones, en pose_frame
    z_offset = float(robot._waist_pos[2]) if robot_cfg.pose_frame == "root" else 0.0
    min_hand_z = args.min_hand_z - z_offset
    min_gaze_z = None if args.min_gaze_z is None else args.min_gaze_z - z_offset
    print(f"[eval] Task: {task!r}")
    print(f"[eval] Running {args.n_episodes} episodes × {args.episode_steps} steps @ {args.fps} fps\n")

    action_keys = list(robot.action_features.keys())

    expected_state_dim, expected_action_dim = read_expected_dims_from_train_config(args.policy_repo)
    obs_probe = robot.get_observation()
    # Torques of actuators outside control_arms (e.g. l_pinky_torque, left over by the
    # curation of a dataset recorded with both arms) are read straight from MuJoCo.
    extra_torque_ids: dict[str, int] = {}
    if state_names is not None:
        missing = []
        for name in state_names:
            if name in obs_probe:
                continue
            try:
                extra_torque_ids[name] = robot.teleop_core.model.actuator(name.removesuffix("_torque")).id
            except KeyError:
                missing.append(name)
        if missing:
            # p. ej. rh_ext_force_* de un modelo --legacy: los datasets actuales no los graban
            print(f"[eval][warn] Robot observation lacks {len(missing)} trained state features "
                  f"{missing[:6]}: fed as 0. Check --scene / --control-arms / --legacy.")
        if extra_torque_ids:
            print(f"[eval] Read from MuJoCo actuator_force: {list(extra_torque_ids)}")
        current_state_dim = len(state_names)
    else:
        current_state_dim = sum(1 for key in obs_probe if key not in trained_cams)
    if action_names and set(action_names) != set(action_keys):
        print(
            "[eval][warn] Action mismatch: "
            f"trained={action_names}, robot={action_keys}."
        )

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

    # Same passive viewer as during recording
    tc = robot.teleop_core
    mj_viewer = mujoco.viewer.launch_passive(tc.model, tc.data)
    mj_viewer.cam.azimuth = 135
    mj_viewer.cam.elevation = -20
    mj_viewer.cam.distance = 2.0
    print("[eval] MuJoCo viewer opened ✓")

    # Escena de la puerta: la métrica es la bisagra (no hay cubo)
    door_adr = handle_adr = None
    if "door_hinge" in [tc.model.joint(j).name for j in range(tc.model.njnt)]:
        door_adr = tc.model.jnt_qposadr[tc.model.joint("door_hinge").id]
        handle_adr = tc.model.jnt_qposadr[tc.model.joint("handle_hinge").id]
        print(f"[eval] Door scene: success = door_hinge > {DOOR_SUCCESS_RAD} rad "
              f"({np.degrees(DOOR_SUCCESS_RAD):.0f} deg) at some step")
        if args.door_xy is not None:
            # Se mueve todo el montaje (marco, hoja, manilla); cambiar de lado la manilla
            # espeja respecto del centro de la hoja, así que basta con ubicarla una vez.
            panel_xy = tc.data.xpos[tc.model.body("door_panel").id][:2].copy()
            place_door(tc.model, tc.data, args.door_xy)
            mujoco.mj_forward(tc.model, tc.data)
            print(f"[eval] Door moved: panel center {np.round(panel_xy, 3)} -> {args.door_xy}")
    elif args.door_xy is not None or args.door_side is not None:
        raise SystemExit("--door-xy / --door-side only apply to the door scene (--scene 3)")

    # ── Evaluation loop ───────────────────────────────────────────────────────
    episode_metrics: list[dict] = []

    try:
        for ep in range(args.n_episodes):
            print(f"── Episode {ep + 1}/{args.n_episodes} ──")

            # Reset scenario: randomize the cube (cube_spawn of control_config.yaml)
            if hasattr(robot, "teleop_core") and robot.teleop_core:
                tc = robot.teleop_core
                if door_adr is not None and args.door_side is not None:
                    side = (args.door_side if args.door_side != "alternate"
                            else ("right" if ep % 2 == 0 else "left"))
                    set_handle_side(tc.model, side)
                if door_adr is not None:
                    # Las demos parten en home; una mano que quedó en la manilla sostendría
                    # la puerta durante el reset.
                    tc.data.qpos[:] = tc.home_qpos
                    tc.data.ctrl[:] = tc.home_ctrl
                    tc._apply_home_pose_to_arrays(tc.data.qpos, tc.data.ctrl)
                    tc.data.qvel[:] = 0
                    tc._set_hand_level("right", 0.0)
                    tc._set_hand_level("left", 0.0)
                    mujoco.mj_forward(tc.model, tc.data)
                if hasattr(tc, "_reset_scenario"):
                    tc._reset_scenario()

                if args.cube_pos is not None and hasattr(tc, "set_cube_pose"):
                    pos = np.array(args.cube_pos)
                    quat = np.array([1.0, 0.0, 0.0, 0.0])
                    tc.set_cube_pose(pos, quat)
                    print(f"[Reset] Fixed blue-cube → {pos}")

                # Targets on the current EE pose + physics warm-up (as eval_policy_mujoco.py)
                if hasattr(tc, "_snap_targets_to_ee"):
                    tc._snap_targets_to_ee()
                for _ in range(10):
                    mujoco.mj_step(tc.model, tc.data)

            # Reset policy hidden state / action chunk
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()

            ep_initial_cube_z: float | None = None
            ep_max_cube_z: float | None = None
            ep_max_door = ep_min_handle = 0.0
            ep_touch_frames = 0
            ep_clamp_steps = 0
            infer_ms: list[float] = []
            rt_gap_s = 0.0
            t_last_send = None

            for step in range(args.episode_steps):
                t0 = time.perf_counter()

                obs = robot.get_observation()
                for trained_cam, source_cam in camera_sources.items():
                    obs[trained_cam] = obs[source_cam]
                for extra_cam in set(camera_sources.values()) - set(trained_cams):
                    obs.pop(extra_cam, None)
                for name in missing if state_names is not None else []:
                    obs[name] = torch.tensor(0.0)
                if args.zero_touch:
                    for touch_key in ("rh_touch_thumb", "rh_touch_index", "rh_touch_middle"):
                        if touch_key in obs:
                            obs[touch_key] = torch.tensor(0.0)
                for name, aid in extra_torque_ids.items():
                    obs[name] = float(robot.teleop_core.data.actuator_force[aid])

                if "obj0_pos_z" in obs:
                    cube_z = to_float(obs["obj0_pos_z"])
                    if ep_initial_cube_z is None:
                        ep_initial_cube_z = cube_z
                    if ep_max_cube_z is None:
                        ep_max_cube_z = cube_z
                    else:
                        ep_max_cube_z = max(ep_max_cube_z, cube_z)

                if door_adr is not None:
                    ep_max_door = max(ep_max_door, float(tc.data.qpos[door_adr]))
                    ep_min_handle = min(ep_min_handle, float(tc.data.qpos[handle_adr]))

                touched = False
                for touch_key in ("rh_touch_thumb", "rh_touch_index", "rh_touch_middle"):
                    if touch_key in obs and to_float(obs[touch_key]) > 0.5:
                        touched = True
                        break
                if touched:
                    ep_touch_frames += 1

                raw_obs = build_raw_observation_dict(obs, args.device, trained_cams, state_names)

                # Preprocessor loaded from checkpoint expects flat batch format
                # (observation.* keys + complementary fields like "task").
                policy_input = {**raw_obs, "task": task}

                # Apply normalization and tokenization
                batch = preprocessor(policy_input)

                t_infer = time.perf_counter()
                with torch.no_grad():
                    action_tensor = policy.select_action(batch)
                infer_ms.append((time.perf_counter() - t_infer) * 1000.0)
                
                # Unnormalize: back to simulator metric coordinates
                action_tensor = postprocessor(action_tensor)

                action_dict = actions_tensor_to_dict(action_tensor, action_keys, action_names)
                if clamp_action_targets(action_dict, min_hand_z, min_gaze_z):
                    ep_clamp_steps += 1
                if args.realtime and t_last_send is not None:
                    # Lo que el paso tardó de más (sobre todo la inferencia de un chunk nuevo)
                    # también pasa en el simulador, con los targets anteriores.
                    gap = min(time.perf_counter() - t_last_send - 1.0 / args.fps, REALTIME_MAX_GAP_S)
                    if gap > 0:
                        advance_physics(robot, gap)
                        rt_gap_s += gap
                t_last_send = time.perf_counter()
                robot.send_action(action_dict)

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
            door_metrics = {}
            if door_adr is not None:
                door_metrics = {
                    "handle_side": handle_side(tc.model),
                    "door_max_deg": float(np.degrees(ep_max_door)),
                    "door_final_deg": float(np.degrees(tc.data.qpos[door_adr])),
                    "handle_min_deg": float(np.degrees(ep_min_handle)),
                }
            success = ep_max_door > DOOR_SUCCESS_RAD if door_adr is not None else lifted
            episode_metrics.append(
                {
                    "episode": ep + 1,
                    "initial_cube_z": ep_initial_cube_z,
                    "max_cube_z": ep_max_cube_z,
                    "lift_delta": lift_delta,
                    "touch_ratio": touch_ratio,
                    "clamp_ratio": ep_clamp_steps / max(args.episode_steps, 1),
                    "lifted": lifted,
                    **door_metrics,
                    "success": success,
                    "infer_ms_max": max(infer_ms) if infer_ms else None,
                    "realtime_extra_s": rt_gap_s,
                }
            )

            init_str = f"{ep_initial_cube_z:.3f}" if ep_initial_cube_z is not None else "n/a"
            max_str = f"{ep_max_cube_z:.3f}" if ep_max_cube_z is not None else "n/a"
            lift_str = f"{lift_delta:.3f}" if not np.isnan(lift_delta) else "n/a"
            task_str = (f"door ({door_metrics['handle_side']} handle) max={door_metrics['door_max_deg']:.1f}deg "
                        f"final={door_metrics['door_final_deg']:.1f}deg "
                        f"handle min={door_metrics['handle_min_deg']:.1f}deg"
                        if door_metrics else f"z0={init_str} zmax={max_str} lift={lift_str}m")
            print(
                f"  [metrics] {task_str} "
                f"touch={touch_ratio:.1%} clamp={ep_clamp_steps / max(args.episode_steps, 1):.1%} "
                f"success={'yes' if success else 'no'}"
            )
            chunk_ms = sorted(infer_ms)[-max(1, len(infer_ms) // policy.config.n_action_steps):]
            print(f"  [timing] inference: chunk {np.mean(chunk_ms):.0f} ms (max {max(infer_ms):.0f}), "
                  f"other steps {np.median(infer_ms):.1f} ms | realtime extra sim time "
                  f"{rt_gap_s:.1f} s")
            print(f"  Episode {ep + 1} done.\n")

        if episode_metrics:
            total = len(episode_metrics)
            successes = sum(1 for m in episode_metrics if m["success"])
            valid_lifts = [m["lift_delta"] for m in episode_metrics if not np.isnan(m["lift_delta"])]
            avg_lift = float(np.mean(valid_lifts)) if valid_lifts else float("nan")
            avg_touch = float(np.mean([m["touch_ratio"] for m in episode_metrics]))
            avg_clamp = float(np.mean([m["clamp_ratio"] for m in episode_metrics]))
            avg_lift_str = f"{avg_lift:.3f}m" if not np.isnan(avg_lift) else "n/a"

            print("[eval] Summary")
            print(f"  success_rate: {successes}/{total} ({successes / total:.1%})")
            if door_adr is not None:
                print(f"  avg_door_max: {np.mean([m['door_max_deg'] for m in episode_metrics]):.1f}deg")
            else:
                print(f"  avg_lift_delta: {avg_lift_str}")
            print(f"  avg_touch_ratio: {avg_touch:.1%}")
            print(f"  avg_clamp_ratio: {avg_clamp:.1%}")

        (run_dir / "summary.json").write_text(json.dumps(
            {"args": vars(args), "episodes": episode_metrics,
             "success_rate": (sum(m["success"] for m in episode_metrics) / len(episode_metrics))
             if episode_metrics else None},
            indent=2, default=float))
        print(f"[eval] Summary saved to {run_dir / 'summary.json'}")

    finally:
        print("[eval] Cleaning up …")
        mj_viewer.close()
        if robot.is_connected:
            robot.disconnect()
        print("[eval] Finished.")


if __name__ == "__main__":
    main()
