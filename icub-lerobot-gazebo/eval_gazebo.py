#!/usr/bin/env python3
"""Evalúa en Gazebo una política entrenada en MuJoCo o en Gazebo (sim-to-sim).

Requisitos: Gazebo (gazebo/scripts/start_sim.sh) y common/yarp_module/teleop_module_sm.py
corriendo, igual que para play_gazebo.py, con las observaciones extra activadas en
config/control_config.yaml (use_joint_vel, use_eef_pose, ...) y los modelos de objetos con
PosePublisher (gazebo/models/<body>/model.sdf).

Si la política viene de MuJoCo (robot_type icub_mujoco en el dataset de entrenamiento):
  * Poses (manos, targets, mirada, objetos) y acciones se convierten entre el mundo MuJoCo y
    el frame root de Gazebo (girado 180 deg en z). Por defecto (--anchor robot) el root de
    Gazebo es la cintura de MuJoCo (0.73 m): mano, cubo, articulaciones y cámara de la cabeza
    quedan consistentes respecto del cuerpo. Con --anchor table se usa la altura real del
    root sobre la mesa (gazebo_root_xyz, 0.77 m): como el robot de Gazebo está 4 cm más
    alto, la política se detiene ~4 cm sobre el cubo.
  * Cámaras por --camera-map (por defecto head_cam=left, head_cam_track_hand=right,
    frontview=front), redimensionadas a la resolución de entrenamiento.
  * Lo que Gazebo no entrega (p. ej. rh_ext_force_*) se rellena con la media del dataset.
  * El cubo se coloca dentro del rango de cube_spawn de MuJoCo (--cube-x/--cube-y).

Uso:
    python eval_gazebo.py \\
        --policy-repo ../output/train/mujocotrainPMM/checkpoints/last/pretrained_model \\
        --task "lift the blue block" --n-episodes 5 --episode-steps 900
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parent

local_robot_path = repo_root / "common" / "lerobot-robot-icubrg"
if local_robot_path.exists():
    sys.path.insert(0, str(local_robot_path))

# Add icubenv site-packages to sys.path (generic python, not the conda env's own)
sys.path.append(os.environ.get("ICUBENV_SITE_PACKAGES", os.path.expanduser("~/miniconda3/envs/icubenv/lib/python3.12/site-packages")))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

# Rotación de 180 deg en z (w, x, y, z): mundo MuJoCo <-> mundo/root de Gazebo
Q_Z180 = (0.0, 0.0, 0.0, 1.0)
Q_Z180_INV = (0.0, 0.0, 0.0, -1.0)

# (prefijo de posición, prefijo de cuaternión) de las poses que hay que convertir.
# La orientación de los objetos no se gira: cada simulador los hace aparecer alineados con
# su propio mundo (cuaternión identidad) y los cubos son simétricos ante 180 deg en z, así
# que girarla daría (0, 0, 0, 1), un valor que la política nunca vio para el mismo cubo.
STATE_POSE_GROUPS = [
    ("rh_ee_pos", "rh_ee_quat"), ("lh_ee_pos", "lh_ee_quat"),
    ("rh_target_pos", "rh_target_quat"), ("lh_target_pos", "lh_target_quat"),
    ("gaze_target_pos", None),
] + [(f"obj{i}_pos", None) for i in range(8)]
ACTION_POSE_GROUPS = [("rh_pos", "rh_quat"), ("lh_pos", "lh_quat"), ("gaze", None)]

OOD_REPORT_STEP = 30
# Cintura (body "icub") en el mundo MuJoCo, igual en icub-lerobot-mj y en el espejo de Gazebo
MUJOCO_WAIST_XYZ = (0.0, 0.0, 0.73)
# Robot MuJoCo (mismo XML que icub-lerobot-mj) para el adaptador de mirada
MIRROR_SCENE_XML = repo_root / "mujoco" / "assets" / "scenes" / "scene_icub_empty_table.xml"
CONSTANT_STD = 1e-3

# Columnas de estado que se guardan por paso en episode_XXX.csv
LOG_STATE_KEYS = [f"{p}_{ax}" for p in ("rh_ee_pos", "rh_target_pos", "obj0_pos") for ax in "xyz"] + [
    "rh_gripper", "rh_touch_thumb", "rh_touch_index", "rh_touch_middle"]

# Mismo criterio que eval_policy_mujoco.py: cubo sobre 0.80 m (mundo, tapa de la mesa en
# 0.72 en ambos simuladores) durante 1 s.
SUCCESS_HEIGHT_M = 0.80
SUCCESS_HOLD_S = 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained policy on the Gazebo iCub")
    p.add_argument("--policy-repo", required=True, help="Local pretrained_model directory")
    p.add_argument("--task", default=None, help="Task prompt (default: first task of the dataset)")
    p.add_argument("--n-episodes", type=int, default=5)
    p.add_argument("--episode-steps", type=int, default=900, help="Max steps per episode at --fps")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--config", default=str(repo_root / "config" / "control_config.yaml"))
    p.add_argument("--robot-name", default="icubSim", help="icubSim (Gazebo) or icub (real robot)")
    p.add_argument("--source-sim", choices=["auto", "mujoco", "gazebo"], default="auto",
                   help="Simulator the policy was trained on (auto: robot_type of the dataset)")
    p.add_argument("--camera-map", default="head_cam=left,head_cam_track_hand=right,frontview=front",
                   help="trained_camera=gazebo_camera pairs, used when --source-sim is mujoco")
    p.add_argument("--cube-x", type=float, nargs=2, default=[0.35, 0.45],
                   help="Cube spawn x range, MuJoCo frame (cube_spawn of the MuJoCo YAML)")
    p.add_argument("--cube-y", type=float, nargs=2, default=[-0.15, -0.05],
                   help="Cube spawn y range, MuJoCo frame")
    p.add_argument("--scene-spawn", action="store_true",
                   help="Keep reset_objects.sh's position (scenes.yaml ranges) instead")
    p.add_argument("--home-wait", type=float, default=4.0, help="Seconds to wait after go_home")
    p.add_argument("--fill-mean", nargs="*", default=[], metavar="PATTERN",
                   help="State names (globs, e.g. '*_torque' '*_vel') replaced by the dataset "
                        "mean, to test which Gazebo inputs throw the policy off")
    p.add_argument("--anchor", choices=["robot", "table"], default="table",
                   help="Reference for MuJoCo <-> Gazebo poses: 'robot' (the waist; hand, cube, joints "
                        "and head camera stay consistent) or 'table' (heights relative to the table; "
                        "the Gazebo robot sits 4 cm higher, so the policy stops 4 cm above the cube)")
    p.add_argument("--legacy", "--legacy-head-cams", dest="legacy", action="store_true",
                   help="MuJoCo policy trained before 2026-10-20 (e.g. mujocotrainPMM): world-frame "
                        "poses, old camera names and head cameras. Converts poses, maps cameras "
                        "(--camera-map) and aims the gaze adapter along head_cam_v1. Newer MuJoCo "
                        "datasets already use the root frame and left/right/front, like Gazebo")
    p.add_argument("--legacy-gripper", action="store_true",
                   help="Policy with the old gripper convention (0 open / 0.5 close / 1 hold), e.g. "
                        "gazebotrainPMM; implied by --legacy. Since 2026-10-21 rh_gripper is the "
                        "closure fraction 0..1")
    p.add_argument("--no-gaze-adapter", action="store_true",
                   help="Send MuJoCo gaze targets as plain look_at points (no MujocoGazeAdapter)")
    p.add_argument("--touch-oracle", action="store_true",
                   help="Gazebo's iCub has no skin: emulate rh_touch_* from the hand-cube "
                        "geometry with a rule fitted on the MuJoCo dataset")
    p.add_argument("--clip-z", type=float, default=None, metavar="Z",
                   help="Clip every state value to mean ± Z·std of the training data")
    p.add_argument("--lift-threshold", type=float, default=0.03, help="Lift success threshold [m]")
    p.add_argument("--grasp-offset", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                   help="Right-hand-frame offset [m] = cube-in-hand at grasp in the eval sim minus "
                        "the training sim (tools/grasp_offset.py). The policy sees the hand shifted "
                        "by it and its targets are shifted back, so it grasps where the eval sim's "
                        "fingers need the cube. MuJoCo model in Gazebo: 0.030 -0.017 -0.005")
    p.add_argument("--n-action-steps", type=int, default=None,
                   help="Actions executed per predicted chunk before replanning (default: the "
                        "trained n_action_steps, e.g. 60). Lower = more closed-loop")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


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


def quat_mul(a, b) -> list[float]:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def mujoco_from_root(pos, quat, root_xyz):
    """Frame root de Gazebo -> mundo MuJoCo (anclado a la mesa)."""
    tx, ty, tz = root_xyz
    pos_w = [-(pos[0] + tx), -(pos[1] + ty), pos[2] + tz]
    return pos_w, (quat_mul(Q_Z180, quat) if quat is not None else None)


def root_from_mujoco(pos, quat, root_xyz):
    """Mundo MuJoCo -> frame root de Gazebo (inversa de mujoco_from_root)."""
    tx, ty, tz = root_xyz
    pos_r = [-pos[0] - tx, -pos[1] - ty, pos[2] - tz]
    return pos_r, (quat_mul(Q_Z180_INV, quat) if quat is not None else None)


def convert_poses(values: dict, groups, fn, root_xyz) -> None:
    """Aplica fn a cada grupo (pos [+ quat]) presente en values, en el lugar."""
    for pos_prefix, quat_prefix in groups:
        pos_keys = [f"{pos_prefix}_{ax}" for ax in "xyz"]
        if not all(k in values for k in pos_keys):
            continue
        quat_keys = [f"{quat_prefix}_{ax}" for ax in "wxyz"] if quat_prefix else None
        if quat_keys and not all(k in values for k in quat_keys):
            quat_keys = None
        pos = [float(values[k]) for k in pos_keys]
        quat = [float(values[k]) for k in quat_keys] if quat_keys else None
        if quat is not None:
            norm = float(np.linalg.norm(quat))
            quat = [q / norm for q in quat] if norm > 1e-6 else [1.0, 0.0, 0.0, 0.0]
        new_pos, new_quat = fn(pos, quat, root_xyz)
        values.update(zip(pos_keys, new_pos))
        if quat_keys:
            values.update(zip(quat_keys, new_quat))


def quat_rotate(quat, vec) -> np.ndarray:
    """Rota vec por el cuaternión (w, x, y, z)."""
    w, x, y, z = np.asarray(quat, dtype=float) / max(float(np.linalg.norm(quat)), 1e-9)
    u = np.array([x, y, z])
    v = np.asarray(vec, dtype=float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


# Poses de la mano derecha a las que se aplica --grasp-offset: (posición, cuaternión)
GRASP_OFFSET_STATE = [("rh_ee_pos", "rh_ee_quat"), ("rh_target_pos", "rh_target_quat")]
GRASP_OFFSET_ACTION = [("rh_pos", "rh_quat")]


def shift_hand(values: dict, groups, offset, sign: float) -> None:
    """pos += sign · R(quat) · offset, en el lugar: la mano que ve la política queda donde
    estaría con la geometría de dedos del simulador de entrenamiento (offset en el frame de
    la mano, ver --grasp-offset). +1 para el estado, -1 para devolver la acción."""
    for pos_prefix, quat_prefix in groups:
        pos_keys = [f"{pos_prefix}_{ax}" for ax in "xyz"]
        quat_keys = [f"{quat_prefix}_{ax}" for ax in "wxyz"]
        if not all(k in values for k in pos_keys + quat_keys):
            continue
        delta = sign * quat_rotate([float(values[k]) for k in quat_keys], offset)
        values.update({k: float(values[k]) + d for k, d in zip(pos_keys, delta)})


class MujocoGazeAdapter:
    """Objetivo de mirada de MuJoCo -> punto para iKinGazeCtrl (ojos bloqueados).

    El IK de mirada de MuJoCo (icub-lerobot-mj/dependencies/mujoco_ik.py, GazeIKSolver)
    apunta el eje z del cuerpo `head` al objetivo, con pitch/yaw del cuello y roll en 0;
    head_cam va inclinada dentro de la cabeza. iKinGazeCtrl apunta el eje de los ojos. Con
    el mismo objetivo el cuello queda muy distinto (en Gazebo -50 deg de pitch contra -10 en
    MuJoCo), y la cámara ve otra escena. Aquí se resuelve el IK de MuJoCo hasta converger
    (mismo robot y torso que Gazebo) y se devuelve un punto sobre el eje de head_cam, en el
    frame root: iKinGazeCtrl deja entonces el cuello como lo dejaría MuJoCo.
    """

    NECK = ("neck_pitch", "neck_yaw")
    TORSO = ("torso_pitch", "torso_roll", "torso_yaw")

    def __init__(self, xml_path: Path, camera: str = "head_cam", distance: float = 2.0):
        import mujoco

        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(str(xml_path))
        self.d = mujoco.MjData(self.m)
        try:
            mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("home").id)
        except KeyError:
            pass
        mujoco.mj_forward(self.m, self.d)
        self.head = self.m.body("head").id
        self.cam = self.m.camera(camera).id
        self.distance = distance
        waist = self.m.body("icub").id
        self.waist_pos = self.d.xpos[waist].copy()
        self.waist_R = self.d.xmat[waist].reshape(3, 3).copy()
        self.qadr = {n: self.m.jnt_qposadr[self.m.joint(n).id] for n in (*self.NECK, "neck_roll", *self.TORSO)}
        self.dofs = np.array([self.m.jnt_dofadr[self.m.joint(n).id] for n in self.NECK])
        self.ranges = np.array([self.m.jnt_range[self.m.joint(n).id] for n in self.NECK])

    def neck_for(self, target_world, torso: dict, iters: int = 100) -> np.ndarray:
        """Pitch/yaw de cuello a los que converge el IK de MuJoCo (roll fijo en 0)."""
        mj, m, d = self.mj, self.m, self.d
        for name, val in torso.items():
            d.qpos[self.qadr[name]] = val
        d.qpos[self.qadr["neck_roll"]] = 0.0
        target = np.asarray(target_world, dtype=float)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            mj.mj_kinematics(m, d)
            mj.mj_comPos(m, d)
            forward = d.xmat[self.head].reshape(3, 3)[:, 2]
            direction = target - d.xpos[self.head]
            direction /= max(np.linalg.norm(direction), 1e-9)
            axis = np.cross(forward, direction)
            sin_a = np.linalg.norm(axis)
            if sin_a < 1e-5:
                break
            err = axis / sin_a * np.arctan2(sin_a, float(forward @ direction))
            mj.mj_jacBody(m, d, jacp, jacr, self.head)
            jac = jacr[:, self.dofs]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(3), err)
            for i, name in enumerate(self.NECK):
                adr = self.qadr[name]
                d.qpos[adr] = np.clip(d.qpos[adr] + 0.5 * dq[i], *self.ranges[i])
        return np.array([d.qpos[self.qadr[n]] for n in self.NECK])

    def fixation_root(self, target_world, torso: dict) -> list[float]:
        """Punto (frame root, anclado a la cintura) sobre el eje óptico de head_cam."""
        self.neck_for(target_world, torso)
        self.mj.mj_kinematics(self.m, self.d)
        self.mj.mj_comPos(self.m, self.d)
        self.mj.mj_camlight(self.m, self.d)  # cam_xpos/cam_xmat no los calcula mj_kinematics
        cam_pos = self.d.cam_xpos[self.cam]
        cam_fwd = -self.d.cam_xmat[self.cam].reshape(3, 3)[:, 2]  # las cámaras miran a -z
        point = cam_pos + self.distance * cam_fwd
        return list(self.waist_R.T @ (point - self.waist_pos))


def load_training_meta(policy_repo: Path) -> tuple[dict, dict, str, list[str]]:
    """features, stats, robot_type y tareas del dataset con que se entrenó la política."""
    train_cfg = json.loads((policy_repo / "train_config.json").read_text())
    root = Path(train_cfg["dataset"]["root"]).expanduser()
    info = json.loads((root / "meta" / "info.json").read_text())
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    tasks: list[str] = []
    try:
        import pandas as pd
        tasks = list(pd.read_parquet(root / "meta" / "tasks.parquet").index)
    except Exception:
        pass
    return info["features"], stats, info.get("robot_type", ""), tasks


def image_hw(feature: dict) -> tuple[int, int]:
    info = feature.get("info", {})
    if "video.height" in info:
        return int(info["video.height"]), int(info["video.width"])
    shape = feature["shape"]
    return (shape[0], shape[1]) if shape[-1] == 3 else (shape[1], shape[2])


def place_cube(body: str, x_mj: float, y_mj: float, z: float = 0.745) -> bool:
    """Pone el cubo en (x, y) del marco MuJoCo; en Gazebo es (-x, -y) (reset_objects.sh)."""
    from lerobot_robot_icubrg.gz_pose import _find_gz

    gz = _find_gz()
    if gz is None:
        return False
    req = (f'name: "{body}", position {{x: {-x_mj:.4f}, y: {-y_mj:.4f}, z: {z:.4f}}}, '
           f'orientation {{w: 1}}')
    out = subprocess.run(
        [gz, "service", "-s", "/world/default/set_pose", "--reqtype", "gz.msgs.Pose",
         "--reptype", "gz.msgs.Boolean", "--req", req, "--timeout", "2000"],
        capture_output=True, text=True,
    )
    return "true" in out.stdout


def report_out_of_distribution(values: dict, names: list[str], mean: list, std: list,
                               top: int = 15) -> None:
    """Variables de estado más lejos de la distribución de entrenamiento (|z| = desvíos)."""
    z = {n: (values[n] - mean[i]) / max(std[i], 1e-6) for i, n in enumerate(names)}
    worst = sorted(z.items(), key=lambda kv: -abs(kv[1]))[:top]
    print(f"[eval] Out-of-distribution state at step {OOD_REPORT_STEP} (|z| = std devs from training):")
    for n, zv in worst:
        i = names.index(n)
        print(f"    {n:28s} z={zv:+8.1f}   gazebo={values[n]:+.3f}   train mean={mean[i]:+.3f} std={std[i]:.3f}")


# Regla ajustada sobre el dataset de MuJoCo (F1 0.85 pulgar, 0.78 índice/medio): contacto si
# la mano está baja y cerca del cubo (el tacto de MuJoCo cuenta también la mesa), o si el cubo
# ya está levantado con la mano cerrada. (radio xy, altura máx. de la mano sobre el cubo) [m]
TOUCH_ORACLE = {"thumb": (0.12, 0.05), "index": (0.07, 0.04), "middle": (0.07, 0.04)}


def oracle_touch(values: dict) -> dict[str, float]:
    """rh_touch_* a partir de la mano y el cubo (marco MuJoCo), para un iCub sin piel."""
    xy = float(np.hypot(values["rh_ee_pos_x"] - values["obj0_pos_x"],
                        values["rh_ee_pos_y"] - values["obj0_pos_y"]))
    dz = values["rh_ee_pos_z"] - values["obj0_pos_z"]
    holding = values.get("rh_gripper") == 0.5 and values["obj0_pos_z"] > 0.755 and xy < 0.06
    return {f"rh_touch_{f}": float(holding or (xy < r and dz < h)) for f, (r, h) in TOUCH_ORACLE.items()}


def wait_object_pose(robot, body: str, timeout_s: float = 3.0) -> bool:
    """True cuando llega una pose del objeto *nueva* (posterior a esta llamada): tras
    reset_objects.sh el cubo se recrea y una pose anterior ya no vale."""
    t0 = time.time()
    while time.time() < t0 + timeout_s:
        reader = robot._pose_reader
        if reader is not None and (reader.stamp(body) or 0.0) > t0:
            return True
        time.sleep(0.05)
    return False


def start_episode(robot, args, cube_body: str) -> None:
    """Brazo a home, mano abierta, mirada y teleop activos, objetos reposicionados."""
    robot.send_rpc_command("go_home")
    if args.scene_spawn:
        # reset_objects.sh borra y recrea los objetos (corre en segundo plano, 1-3 s)
        robot.send_rpc_command("reset_scenario")
    time.sleep(args.home_wait)
    robot.send_rpc_command("open_right")
    robot.send_rpc_command("start_gaze")
    time.sleep(0.1)
    gx, gy, gz = robot.config.gaze_target_init
    robot.send_rpc_command(f"look_at {gx} {gy} {gz}")
    time.sleep(0.1)
    robot.send_rpc_command("start_teleop")
    if not args.scene_spawn:
        # Solo set_pose: con reset_scenario además, el cubo aparecía dos veces y, si el
        # script terminaba tarde, quedaba en la posición de la escena y no en esta.
        x, y = random.uniform(*args.cube_x), random.uniform(*args.cube_y)
        ok = place_cube(cube_body, x, y)
        print(f"[Reset] {cube_body} -> MuJoCo frame ({x:.3f}, {y:.3f})" + ("" if ok else "  (set_pose FAILED)"))
    time.sleep(0.5)
    robot.reset_command_state()


def main() -> None:
    args = parse_args()
    os.environ["ICUB_ROBOT_NAME"] = args.robot_name

    try:  # LeRobot >= 0.6 (smolvla); 0.5.x (icubenv / Docker) lo tiene en lerobot.utils
        from lerobot.common.control_utils import predict_action
    except ImportError:
        from lerobot.utils.control_utils import predict_action
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.policies.utils import make_robot_action
    from lerobot.utils.constants import OBS_STR
    from lerobot.utils.device_utils import get_safe_torch_device
    try:  # LeRobot >= 0.6; en 0.5.x está en lerobot.datasets.feature_utils
        from lerobot.utils.feature_utils import build_dataset_frame
    except ImportError:
        from lerobot.datasets.feature_utils import build_dataset_frame
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot_robot_icubrg.config_icub import iCubConfig
    from lerobot_robot_icubrg.icub import dataset_uses_canonical_quats, iCub

    policy_repo = Path(args.policy_repo).expanduser().resolve()
    run_dir = repo_root.parent / "outputs" / "tests" / (
        f"{policy_repo.parents[2].name if len(policy_repo.parents) > 2 else 'policy'}"
        f"_gazebo_{time.strftime('%Y%m%d-%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_to(run_dir / "eval.log")
    ds_features, ds_stats, robot_type, tasks = load_training_meta(policy_repo)
    source = args.source_sim
    if source == "auto":
        source = "mujoco" if robot_type == "icub_mujoco" else "gazebo"
    # Desde el 2026-10-20 MuJoCo graba en el frame root y con cámaras left/right/front, igual
    # que Gazebo: solo los modelos --legacy necesitan conversión.
    convert = source == "mujoco" and args.legacy
    task = args.task or (tasks[0] if tasks else "")
    state_names: list[str] = ds_features["observation.state"]["names"]
    image_keys = [k for k in ds_features if k.startswith("observation.images.")]

    # ── Política ──────────────────────────────────────────────────────────────
    print(f"[eval] Loading policy from {policy_repo} (trained on {source}, robot_type={robot_type!r})")
    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_repo))
    policy_cfg.device = args.device
    policy_cfg.pretrained_path = str(policy_repo)
    if args.n_action_steps is not None:
        if not 1 <= args.n_action_steps <= policy_cfg.chunk_size:
            raise SystemExit(f"--n-action-steps must be in 1..{policy_cfg.chunk_size}")
        policy_cfg.n_action_steps = args.n_action_steps
    print(f"[eval] n_action_steps={policy_cfg.n_action_steps} (chunk_size={policy_cfg.chunk_size})")
    policy = get_policy_class(policy_cfg.type).from_pretrained(str(policy_repo), config=policy_cfg)
    policy.to(args.device).eval()
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg, pretrained_path=str(policy_repo))
    device = get_safe_torch_device(args.device)

    # ── Robot ─────────────────────────────────────────────────────────────────
    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    robot_cfg = iCubConfig(config_path=str(cfg_path))
    robot_cfg.gripper_legacy = args.legacy or args.legacy_gripper
    # Misma convención de signo de cuaterniones que el dataset de entrenamiento
    robot_cfg.canonical_quats = dataset_uses_canonical_quats(
        ds_stats, state_names, ds_features["action"]["names"])
    print(f"[eval] Quaternion sign: {'canonical' if robot_cfg.canonical_quats else 'raw (dataset before 2026-10-21)'}")
    root_xyz = robot_cfg.gazebo_root_xyz  # root real de Gazebo (alturas del cubo)
    # Root de Gazebo en el mundo MuJoCo para convertir poses: la cintura de MuJoCo (anclado al
    # robot) o la posición real del root sobre la mesa (anclado a la mesa).
    conv_xyz = MUJOCO_WAIST_XYZ if args.anchor == "robot" else root_xyz
    robot = iCub(robot_cfg)
    robot_state = [k for k, v in robot.observation_features.items() if v is float]

    camera_map = {}
    for key in image_keys:
        cam = key.removeprefix("observation.images.")
        camera_map[cam] = cam
    if convert:
        camera_map.update(dict(pair.split("=") for pair in args.camera_map.split(",") if pair))
    missing_cams = [g for t, g in camera_map.items() if g not in robot_cfg.camera_ports]
    if missing_cams:
        raise SystemExit(f"Gazebo cameras {missing_cams} not in camera_ports {list(robot_cfg.camera_ports)}")

    # Lo que la política espera y Gazebo no entrega: media del dataset (≈ 0 normalizado)
    state_mean = ds_stats.get("observation.state", {}).get("mean")
    fill = {n: (float(state_mean[i]) if state_mean else 0.0)
            for i, n in enumerate(state_names)
            if n not in robot_state or any(fnmatch.fnmatch(n, pat) for pat in args.fill_mean)}
    state_std = ds_stats.get("observation.state", {}).get("std")
    # Variables constantes en el entrenamiento (std ≈ 0, p. ej. neck_roll en MuJoCo): no
    # aportan información y cualquier valor distinto, normalizado, explota (0.03 rad / 1e-4
    # = 300 desvíos). Se fijan en su valor de entrenamiento.
    if state_mean and state_std:
        fill.update({n: float(state_mean[i]) for i, n in enumerate(state_names)
                     if state_std[i] < CONSTANT_STD and n not in fill})
    action_names = ds_features["action"]["names"]
    print(f"[eval] Task: {task!r}")
    print(f"[eval] State {len(state_names)} dims; filled with the dataset mean: {list(fill)}")
    print(f"[eval] Cameras (trained <- gazebo): {camera_map}")
    gaze_adapter = None
    if convert:
        print(f"[eval] MuJoCo world <-> Gazebo root anchored to the {args.anchor} (root at {conv_xyz})")
        if not args.no_gaze_adapter and any(n.startswith("gaze_") for n in action_names):
            gaze_adapter = MujocoGazeAdapter(
                MIRROR_SCENE_XML, camera="head_cam_v1")
            print("[eval] Gaze: MuJoCo head_cam axis -> iKinGazeCtrl fixation point (MujocoGazeAdapter)")
    action_groups = [g for g in ACTION_POSE_GROUPS if not (gaze_adapter and g[0] == "gaze")]
    grasp_offset = np.array(args.grasp_offset) if args.grasp_offset else None
    if grasp_offset is not None:
        print(f"[eval] Grasp offset (right hand frame): {np.round(grasp_offset, 4).tolist()} m; "
              f"CSV real_* columns hold the hand and command actually used in Gazebo")
    if any(n.startswith("lh_") for n in action_names) and robot_cfg.control_arms == "right":
        print("[eval][warn] Policy has left-arm actions but control_arms is 'right' in the YAML.")

    robot.connect()
    cube_body = robot_cfg.object_bodies[0] if robot_cfg.object_bodies else "blue-cube"
    hold_steps = max(1, int(np.ceil(SUCCESS_HOLD_S * args.fps)))
    episodes = []

    try:
        for ep in range(args.n_episodes):
            print(f"\n── Episode {ep + 1}/{args.n_episodes} ──")
            start_episode(robot, args, cube_body)
            if robot_cfg.use_object_pose and not wait_object_pose(robot, cube_body):
                raise SystemExit(
                    f"No pose for '{cube_body}' from Gazebo (/model/{cube_body}/pose): the policy "
                    f"would see a stale or zero pose. Check `gz topic -l | grep pose` "
                    f"with icubenv's gz, and restart Gazebo if model.sdf lacks the PosePublisher."
                )
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()

            z0 = z_max = None
            # Mirada que pidió la política (marco MuJoCo): es lo que MuJoCo grababa como
            # gaze_target_pos; con el adaptador, el look_at real de Gazebo es otro punto.
            last_gaze_mj = None
            log_rows: list[dict] = []
            above = 0
            held = False
            t_ep = t_prev = time.perf_counter()

            for step in range(args.episode_steps):
                t0 = time.perf_counter()
                obs = robot.get_observation()

                values = {n: float(obs[n]) for n in robot_state if n in obs}
                if convert:
                    convert_poses(values, STATE_POSE_GROUPS, mujoco_from_root, conv_xyz)
                real_ee = {f"real_rh_ee_pos_{ax}": values.get(f"rh_ee_pos_{ax}") for ax in "xyz"}
                if grasp_offset is not None:
                    shift_hand(values, GRASP_OFFSET_STATE, grasp_offset, +1.0)
                values.update(fill)  # ya en el marco de entrenamiento
                if args.touch_oracle:
                    values.update(oracle_touch(values))
                if gaze_adapter is not None:
                    gaze_state = last_gaze_mj or [state_mean[state_names.index(f"gaze_target_pos_{ax}")]
                                                  for ax in "xyz"]
                    values.update({f"gaze_target_pos_{ax}": v for ax, v in zip("xyz", gaze_state)})
                if args.clip_z is not None and state_mean and state_std:
                    for i, n in enumerate(state_names):
                        lo = state_mean[i] - args.clip_z * state_std[i]
                        hi = state_mean[i] + args.clip_z * state_std[i]
                        values[n] = min(max(values[n], lo), hi)

                # Altura del cubo en el mundo (misma en ambos simuladores)
                if "obj0_pos_z" in obs:
                    z = float(obs["obj0_pos_z"]) + root_xyz[2]
                    z0 = z if z0 is None else z0
                    z_max = z if z_max is None else max(z_max, z)
                    above = above + 1 if z > SUCCESS_HEIGHT_M else 0
                    held = held or above >= hold_steps

                for trained_cam, gz_cam in camera_map.items():
                    h, w = image_hw(ds_features[f"observation.images.{trained_cam}"])
                    img = obs[gz_cam]
                    if img.shape[:2] != (h, w):
                        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                    values[trained_cam] = img

                frame = build_dataset_frame(ds_features, values, prefix=OBS_STR)
                t_infer = time.perf_counter()
                action_tensor = predict_action(
                    observation=frame, policy=policy, device=device,
                    preprocessor=preprocessor, postprocessor=postprocessor,
                    use_amp=policy.config.use_amp, task=task, robot_type=robot_type,
                )
                infer_ms = (time.perf_counter() - t_infer) * 1000.0
                action = make_robot_action(action_tensor, ds_features)
                # Lo que pide la política y dónde está la mano/el cubo, en el marco de
                # entrenamiento (comparable con las stats del dataset). Tiempos: t_s desde el
                # inicio del episodio, period_ms entre pasos, infer_ms de la política (alto =
                # chunk nuevo; Gazebo sigue corriendo mientras tanto).
                row = {"step": step, "t_s": round(t0 - t_ep, 4),
                       "period_ms": round((t0 - t_prev) * 1000.0, 2), "infer_ms": round(infer_ms, 2)}
                t_prev = t0
                row.update({f"cmd_{k}": v for k, v in action.items()})
                row.update({k: values[k] for k in LOG_STATE_KEYS if k in values})
                row.update({f"state.{n}": values[n] for n in state_names})
                log_rows.append(row)
                if ep == 0 and step == OOD_REPORT_STEP and state_mean and state_std:
                    report_out_of_distribution(values, state_names, state_mean, state_std)
                if gaze_adapter is not None:
                    last_gaze_mj = [action[f"gaze_{ax}"] for ax in "xyz"]
                    torso = {n: float(obs[n]) for n in MujocoGazeAdapter.TORSO if n in obs}
                    action.update(zip(("gaze_x", "gaze_y", "gaze_z"),
                                      gaze_adapter.fixation_root(last_gaze_mj, torso)))
                if grasp_offset is not None:
                    shift_hand(action, GRASP_OFFSET_ACTION, grasp_offset, -1.0)
                    row.update(real_ee)
                    row.update({f"real_cmd_rh_pos_{ax}": float(action[f"rh_pos_{ax}"]) for ax in "xyz"})
                if convert:
                    convert_poses(action, action_groups, root_from_mujoco, conv_xyz)
                robot.send_action(action)

                if step % args.fps == 0:
                    z_str = f"{z:.3f}" if z0 is not None else "n/a"
                    print(f"  step {step:4d}/{args.episode_steps}  cube_z={z_str}")
                precise_sleep(max(1.0 / args.fps - (time.perf_counter() - t0), 0.0))

            if log_rows:
                with open(run_dir / f"episode_{ep:03d}.csv", "w", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=list(log_rows[0]))
                    writer.writeheader()
                    writer.writerows(log_rows)
            lift = (z_max - z0) if z0 is not None else float("nan")
            result = {"episode": ep + 1, "z0": z0, "z_max": z_max, "lift_delta": lift,
                      "lifted": bool(lift >= args.lift_threshold) if z0 is not None else False,
                      "held_above_0p80_1s": held}
            episodes.append(result)
            print(f"  [metrics] lift={lift:.3f} m  lifted={result['lifted']}  held={held}")
            if log_rows:
                inf = np.array([r["infer_ms"] for r in log_rows])
                per = np.array([r["period_ms"] for r in log_rows[1:]])
                print(f"  [timing] {len(log_rows) / log_rows[-1]['t_s']:.1f} Hz | inference: chunk "
                      f"{np.sort(inf)[-max(1, len(inf) // policy_cfg.n_action_steps):].mean():.0f} ms "
                      f"(max {inf.max():.0f}), other steps {np.median(inf):.1f} ms | period p99 "
                      f"{np.percentile(per, 99):.0f} ms")
    finally:
        summary = {
            "policy": str(policy_repo), "source_sim": source, "task": task,
            "camera_map": camera_map, "filled_state": list(fill), "root_xyz": list(root_xyz),
            "anchor": args.anchor, "conv_xyz": list(conv_xyz),
            "grasp_offset": args.grasp_offset,
            "episodes": episodes,
            "success_rate_lift": float(np.mean([e["lifted"] for e in episodes])) if episodes else None,
            "success_rate_held": float(np.mean([e["held_above_0p80_1s"] for e in episodes])) if episodes else None,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\n[eval] Summary saved to {run_dir / 'summary.json'}")
        if episodes:
            print(f"  lift success: {summary['success_rate_lift']:.0%}   held success: {summary['success_rate_held']:.0%}")
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
