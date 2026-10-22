from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Any

import mujoco
import numpy as np
import torch

from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_icub_mujoco import iCubMuJoCoConfig

# Articulaciones sin torque: Gazebo los entrega siempre en 0 en los dedos y los ojos (y el
# iCub real no los mide), así que ni Gazebo ni MuJoCo los graban (mismo dataset sim-to-sim).
NO_TORQUE_TAGS = ("_hand_finger", "_thumb_", "_index_", "_middle_", "_pinky", "eyes_")


def has_torque(joint: str) -> bool:
    return not any(tag in joint for tag in NO_TORQUE_TAGS)


# Signo de los cuaterniones: q y -q son la misma orientación, pero para una política son
# valores opuestos, y cada fuente elegía uno distinto (getPose de iKin da w >= 0, el VR no).
# Ambos simuladores graban el mismo hemisferio: manos (ee, target y acción, frame root) con
# quat_y <= 0, es decir q·(0, 0, -1, 0) >= 0, la orientación típica de la mano (todas las
# poses grabadas quedan a |q_y| >= 0.65 del cambio de signo); objetos con quat_w >= 0
# (aparecen en la identidad). Mismo criterio en lerobot_robot_icubrg (Gazebo) y en
# icub-lerobot-gazebo/tools/canonicalize_quats.py.
QUAT_HEMISPHERE = "hand_y<=0,obj_w>=0"


def canonicalize_quats(values: dict) -> dict:
    """Pasa al hemisferio de QUAT_HEMISPHERE cada cuaternión {prefijo}_quat_{w,x,y,z}
    presente en values (tensores o floats), en el lugar. Devuelve values."""
    prefixes = {k[:-2] for k in values if k.endswith("_quat_w")}
    for prefix in prefixes:
        keys = [f"{prefix}_{ax}" for ax in "wxyz"]
        if not all(k in values for k in keys):
            continue
        if prefix.startswith("obj"):
            flip = float(values[f"{prefix}_w"]) < 0.0
        else:
            flip = float(values[f"{prefix}_y"]) > 0.0
        if flip:
            for k in keys:
                values[k] = -values[k]
    return values


def dataset_uses_canonical_quats(stats: dict, state_names: list[str], action_names: list[str]) -> bool:
    """True si el dataset ya viene en el hemisferio de QUAT_HEMISPHERE, según su stats.json
    (manos con max(quat_y) <= 0, objetos con min(quat_w) >= 0). Los eval lo usan para
    darle a una política la misma convención con que se entrenó: los datasets anteriores al
    2026-10-21 (sin pasar por tools/canonicalize_quats.py) dan False."""
    groups = []
    for key, names in (("observation.state", state_names), ("action", action_names)):
        st = stats.get(key) or {}
        for i, name in enumerate(names):
            if name.endswith("_quat_y") and not name.startswith("obj"):
                groups.append(st.get("max", [None] * len(names))[i] is not None
                              and float(st["max"][i]) <= 1e-6)
            elif name.endswith("_quat_w") and name.startswith("obj"):
                groups.append(st.get("min", [None] * len(names))[i] is not None
                              and float(st["min"][i]) >= -1e-6)
    return bool(groups) and all(groups)


class iCubMuJoCo(Robot):
    config_class = iCubMuJoCoConfig
    name = "icub_mujoco"

    def __init__(self, config: iCubMuJoCoConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._calibrated = True

        self.teleop_core = None
        self.renderers: dict[str, mujoco.Renderer] = {}
        self.cameras: dict[str, Any] = {}

        self.actuator_list: list[str] = []
        for part in self.config.actuators_to_use:
            self.actuator_list.extend(self.config.actuators_to_use[part])

        self._actuator_id_by_name: dict[str, int] = {}
        self._missing_actuator_names: set[str] = set()
        self._camera_name_resolved: dict[str, str] = {}

        self._last_action: dict[str, float] = {}
        # Métricas por frame que lee metrics/frame_metrics.py (tiempo simulado del
        # estado y del render, costo de física y render). Se rehace en cada get_observation.
        self.metrics_meta: dict[str, Any] = {}
        self._last_touch_state: dict[str, float] = {}
        self._vr_rec_start_event   = False
        self._vr_rec_stop_event    = False
        self._vr_rec_discard_event = False   # B mantenido 1.5s → descarta episodio
        self._vr_a_held      = False
        self._vr_b_held      = False
        self._vr_b_hold_start: float | None = None   # timestamp cuando se empezó a p​resar B

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {actuator_name: float for actuator_name in self.actuator_list}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        # (alto, ancho, canales) como lerobot_robot_icubrg y la convención de LeRobot: con
        # la misma forma, los datasets de MuJoCo y Gazebo se pueden combinar.
        return {
            name: (self.config.camera_height, self.config.camera_width, 3)
            for name in self.config.camera_names
        }

    def _arm_sides(self) -> list[str]:
        return [side for side, arms in (("rh", ("right", "both")), ("lh", ("left", "both")))
                if self.config.control_arms in arms]

    # (prefijo de posición, prefijo de cuaternión) de las poses que pasan al frame root
    _POSE_GROUPS = [("rh_ee_pos", "rh_ee_quat"), ("lh_ee_pos", "lh_ee_quat"),
                    ("rh_target_pos", "rh_target_quat"), ("lh_target_pos", "lh_target_quat"),
                    ("gaze_target_pos", None)] + [(f"obj{i}_pos", f"obj{i}_quat") for i in range(8)]
    _ACTION_POSE_GROUPS = [("rh_pos", "rh_quat"), ("lh_pos", "lh_quat"), ("gaze", None)]

    def _convert_poses(self, values: dict, groups, to_root: bool) -> None:
        """Mundo MuJoCo <-> frame root del iCub (cintura, body "icub"), en el lugar.

        Misma conversión que el teleoperador de Gazebo (_world_to_root): Gazebo y el robot
        real usan el frame root, así los tres graban las poses en el mismo marco.
        """
        for pos_prefix, quat_prefix in groups:
            pos_keys = [f"{pos_prefix}_{ax}" for ax in "xyz"]
            if not all(k in values for k in pos_keys):
                continue
            pos = np.array([self._to_float(values[k]) for k in pos_keys])
            if to_root:
                new_pos = self._waist_R.T @ (pos - self._waist_pos)
            else:
                new_pos = self._waist_R @ pos + self._waist_pos
            for k, v in zip(pos_keys, new_pos):
                values[k] = torch.tensor(float(v), dtype=torch.float32)
            quat_keys = [f"{quat_prefix}_{ax}" for ax in "wxyz"] if quat_prefix else []
            if quat_keys and all(k in values for k in quat_keys):
                quat = np.array([self._to_float(values[k]) for k in quat_keys])
                new_quat = np.zeros(4)
                mujoco.mju_mulQuat(new_quat, self._waist_quat_inv if to_root else self._waist_quat, quat)
                for k, v in zip(quat_keys, new_quat):
                    values[k] = torch.tensor(float(v), dtype=torch.float32)

    @property
    def observation_features(self) -> dict[str, type]:
        # ── Features are ordered so that the training-relevant dims come FIRST ──
        # This lets SmolVLA use the first N dims via max_state_dim without filtering.
        #
        # ORDER (with VR, 118 dims total):
        #   [0 – 21]   joint positions           (22)  ← from _motors_ft
        #   [22– 43]   joint velocities           (22)
        #   [44– 57]   EEF pose  rh + lh          (14)
        #   [58– 76]   VR targets (if vr_enabled) (19)
        #   [77– 82]   touch contact rh + lh       (6)
        #   ── 83 training dims ──────────────────────
        #   [83– 89]   object pose                 (7)
        #   [90–102]   joint torques              (13, sin dedos)
        #   [112–117]  external forces rh + lh     (6)
        #   ── 118 total (+ cameras as separate keys) ─
        features = {**self._motors_ft}   # [0-21] joint positions

        # [22-43] joint velocities
        for name in self.actuator_list:
            features[f"{name}_vel"] = float

        # Solo los brazos de control_arms (lo que elige el Hub), igual que en Gazebo:
        # una tarea de un brazo no graba la otra mano.
        sides = self._arm_sides()

        # [44-57] End-effector poses (x,y,z + quat w,x,y,z)
        for side in sides:
            for ax in ("x", "y", "z"):
                features[f"{side}_ee_pos_{ax}"] = float
            for ax in ("w", "x", "y", "z"):
                features[f"{side}_ee_quat_{ax}"] = float

        # [58-76] Mocap target poses (VR moves them during teleoperation; policy moves
        #         them during eval via send_action)
        for side in sides:
            for ax in ("x", "y", "z"):
                features[f"{side}_target_pos_{ax}"] = float
            for ax in ("w", "x", "y", "z"):
                features[f"{side}_target_quat_{ax}"] = float
        if self.config.use_gaze:
            for ax in ("x", "y", "z"):
                features[f"gaze_target_pos_{ax}"] = float
        for side in sides:
            features[f"{side}_gripper"] = float

        # [77-82] Tactile contact (binary per finger)
        for side in sides:
            for finger in ("thumb", "index", "middle"):
                features[f"{side}_touch_{finger}"] = float

        # ── less critical dims below (not needed for training) ──

        # object poses — indexed by order in scenes.yaml
        scene_objects = getattr(self.config, "scene_objects", []) or []
        num_objects = max(len(scene_objects), 1)  # at least 1 slot for legacy compat
        for i in range(num_objects):
            features[f"obj{i}_pos_x"] = float
            features[f"obj{i}_pos_y"] = float
            features[f"obj{i}_pos_z"] = float
            features[f"obj{i}_quat_w"] = float
            features[f"obj{i}_quat_x"] = float
            features[f"obj{i}_quat_y"] = float
            features[f"obj{i}_quat_z"] = float

        # joint torques / actuator forces (sin dedos: Gazebo no los entrega)
        for name in filter(has_torque, self.actuator_list):
            features[f"{name}_torque"] = float

        # (Sin fuerza externa en la mano: solo existe en MuJoCo, no en Gazebo ni en el robot
        #  real sin wholeBodyDynamics, y un dataset sim-to-sim no puede depender de ella.)

        # cameras (separate observation.images.* keys, not part of state vector)
        features.update(self._cameras_ft)

        return features

    @property
    def action_features(self) -> dict[str, type]:
        features = {}
        if self.config.control_arms in ["right", "both"]:
            features["rh_pos_x"] = float
            features["rh_pos_y"] = float
            features["rh_pos_z"] = float
            features["rh_quat_w"] = float
            features["rh_quat_x"] = float
            features["rh_quat_y"] = float
            features["rh_quat_z"] = float
            features["rh_gripper"] = float
        if self.config.control_arms in ["left", "both"]:
            features["lh_pos_x"] = float
            features["lh_pos_y"] = float
            features["lh_pos_z"] = float
            features["lh_quat_w"] = float
            features["lh_quat_x"] = float
            features["lh_quat_y"] = float
            features["lh_quat_z"] = float
            features["lh_gripper"] = float
        if self.config.use_gaze:
            features["gaze_x"] = float
            features["gaze_y"] = float
            features["gaze_z"] = float
        return features

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Try to import from the local dependencies/ package first (self-contained repo),
        # then fall back to icub_examples in the workspace.
        try:
            # dependencies/ lives at <repo_root>/dependencies/
            _repo_root = Path(__file__).resolve().parents[3]
            _deps_root = str(_repo_root)
            if _deps_root not in sys.path:
                sys.path.insert(0, _deps_root)
            from dependencies.teleop_mujoco import MuJoCoTeleop
        except (ImportError, ModuleNotFoundError):
            # Fallback: look for icub_examples in the workspace
            env_root = os.getenv("ICUB_WORKSPACE_ROOT", "").strip()
            candidate_roots = []
            if env_root:
                candidate_roots.append(Path(env_root).expanduser())
            candidate_roots.append(Path(__file__).resolve().parents[3])
            for workspace_root in candidate_roots:
                ws = str(workspace_root)
                if ws not in sys.path:
                    sys.path.insert(0, ws)
            from icub_examples.mu_teleop.teleop_mujoco import MuJoCoTeleop

        model_path = str(Path(self.config.model_path).expanduser().resolve())
        self.teleop_core = MuJoCoTeleop(
            model_path=model_path,
            vr_enabled=bool(self.config.vr_enabled),
            record_enabled=False,
            control_arms=self.config.control_arms,
            primary_arm=self.config.primary_arm,
            config_path=self.config.config_path,
            vr_ip=self.config.vr_ip,
            scene_objects=self.config.scene_objects,
        )
        # Objetos/joints de scenes.yaml: el reset del núcleo (botón de VR) usa los mismos
        # rangos que la escena del Hub, no el cube_spawn heredado del YAML.
        self.teleop_core._scene_joints = list(self.config.scene_joints or [])

        self.teleop_core.state = "teleop_active"

        model = self.teleop_core.model

        # Cintura (frame root del iCub) en el mundo MuJoCo, para pose_frame="root"
        waist_id = model.body("icub").id
        self._waist_pos = self.teleop_core.data.xpos[waist_id].copy()
        self._waist_R = self.teleop_core.data.xmat[waist_id].reshape(3, 3).copy()
        self._waist_quat = np.zeros(4)
        mujoco.mju_mat2Quat(self._waist_quat, self.teleop_core.data.xmat[waist_id].copy())
        self._waist_quat_inv = np.zeros(4)
        mujoco.mju_negQuat(self._waist_quat_inv, self._waist_quat)

        self._actuator_id_by_name = {}
        self._missing_actuator_names = set()
        for name in self.actuator_list:
            try:
                self._actuator_id_by_name[name] = model.actuator(name).id
            except KeyError:
                self._missing_actuator_names.add(name)

        self.renderers = {}
        self.cameras = {}
        # Scene option for offscreen rendering: hide mocap target geoms (group 2)
        self._cam_scene_option = mujoco.MjvOption()
        self._cam_scene_option.geomgroup[2] = 0
        # y sus sites (puntos de los targets): no existen en Gazebo ni en el robot real
        self._cam_scene_option.sitegroup[2] = 0
        for requested_name in self.config.camera_names:
            resolved_name = self.config.camera_sources.get(requested_name, requested_name)
            try:
                model.camera(resolved_name).id
            except Exception:
                print(f"[WARN] Camera '{resolved_name}' ({requested_name}) not found in the MuJoCo "
                      f"model: using head_cam.")
                resolved_name = "head_cam"
            self._camera_name_resolved[requested_name] = resolved_name
            self.renderers[requested_name] = mujoco.Renderer(
                model,
                height=self.config.camera_height,
                width=self.config.camera_width,
            )
            self.cameras[requested_name] = self.renderers[requested_name]

        self._connected = True
        if calibrate and not self.is_calibrated:
            self.calibrate()

    def calibrate(self) -> None:
        if not self.teleop_core:
            return
        self.teleop_core._snap_targets_to_ee()
        self._calibrated = True

    def configure(self) -> None:
        pass

    def _to_float(self, val: Any) -> float:
        if isinstance(val, torch.Tensor):
            return float(val.item())
        return float(val)

    def _action_value(self, action: dict[str, Any], *keys: str) -> float:
        for key in keys:
            if key in action:
                return self._to_float(action[key])
        raise KeyError(f"None of keys found in action: {keys}")

    def _set_targets_from_action(self, action: dict[str, Any]) -> None:
        data = self.teleop_core.data
        if self.config.pose_frame == "root":
            action = dict(action)
            self._convert_poses(action, self._ACTION_POSE_GROUPS, to_root=False)

        def apply_grip(side: str, grip_value: float) -> None:
            """rh/lh_gripper = fracción de cierre 0..1. Con gripper_legacy (modelos y datasets
            anteriores al 2026-10-21): <=0.1 abrir, 0.4-0.6 cerrar, resto = mantener."""
            if self.config.gripper_legacy:
                if grip_value <= 0.1:
                    level = 0.0
                elif 0.4 <= grip_value <= 0.6:
                    level = 1.0
                else:
                    return
            else:
                level = float(np.clip(grip_value, 0.0, 1.0))
            self.teleop_core._set_hand_level(side, level)

        if self.config.control_arms in ["right", "both"]:
            rh_pos = np.array([
                self._action_value(action, "rh_pos_x", "rh_x"),
                self._action_value(action, "rh_pos_y", "rh_y"),
                self._action_value(action, "rh_pos_z", "rh_z"),
            ], dtype=np.float64)
            rh_quat = np.array([
                self._action_value(action, "rh_quat_w", "rh_qw"),
                self._action_value(action, "rh_quat_x", "rh_qx"),
                self._action_value(action, "rh_quat_y", "rh_qy"),
                self._action_value(action, "rh_quat_z", "rh_qz"),
            ], dtype=np.float64)
            # Normalize quaternion — networks don't guarantee unit norm;
            # a non-unit quat passed to the IK causes CTRL NaN/Inf.
            rh_quat_norm = np.linalg.norm(rh_quat)
            if rh_quat_norm > 1e-6:
                rh_quat = rh_quat / rh_quat_norm
            else:
                rh_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity fallback
            data.mocap_pos[self.teleop_core.rh_mocap_id] = rh_pos
            data.mocap_quat[self.teleop_core.rh_mocap_id] = rh_quat

            if "rh_gripper" in action:
                apply_grip("right", self._to_float(action["rh_gripper"]))

        if self.config.control_arms in ["left", "both"]:
            # Left-hand keys are optional: if the policy only controls the right
            # arm, skip updating the left mocap target (it stays in place).
            _lh_has_pos = any(k in action for k in ("lh_pos_x", "lh_x"))
            if _lh_has_pos:
                lh_pos = np.array([
                    self._action_value(action, "lh_pos_x", "lh_x"),
                    self._action_value(action, "lh_pos_y", "lh_y"),
                    self._action_value(action, "lh_pos_z", "lh_z"),
                ], dtype=np.float64)
                lh_quat = np.array([
                    self._action_value(action, "lh_quat_w", "lh_qw"),
                    self._action_value(action, "lh_quat_x", "lh_qx"),
                    self._action_value(action, "lh_quat_y", "lh_qy"),
                    self._action_value(action, "lh_quat_z", "lh_qz"),
                ], dtype=np.float64)
                lh_quat_norm = np.linalg.norm(lh_quat)
                if lh_quat_norm > 1e-6:
                    lh_quat = lh_quat / lh_quat_norm
                else:
                    lh_quat = np.array([1.0, 0.0, 0.0, 0.0])
                data.mocap_pos[self.teleop_core.lh_mocap_id] = lh_pos
                data.mocap_quat[self.teleop_core.lh_mocap_id] = lh_quat

            if "lh_gripper" in action:
                apply_grip("left", self._to_float(action["lh_gripper"]))

        if self.config.use_gaze and all(k in action for k in ("gaze_x", "gaze_y", "gaze_z")):
            gaze_pos = np.array([
                self._to_float(action["gaze_x"]),
                self._to_float(action["gaze_y"]),
                self._to_float(action["gaze_z"]),
            ], dtype=np.float64)
            data.mocap_pos[self.teleop_core.gaze_mocap_id] = gaze_pos

    # Duración mínima (s) que hay que mantener B para descartar el episodio.
    # Una pulsación corta (<B_HOLD_DISCARD_S) sigue siendo el stop/save normal.
    _B_HOLD_DISCARD_S: float = 1.5

    def _update_vr_record_events(self) -> None:
        import time as _time
        if self.teleop_core is None:
            return
        b = self.teleop_core.latest_buttons
        if b is None or not isinstance(b, dict):
            return

        a_pressed = (b.get("BTN_ONE_RIGHT", 0) > 0.5) or (b.get("A", 0) > 0.8)
        b_pressed = (b.get("BTN_TWO_RIGHT", 0) > 0.5) or (b.get("B", 0) > 0.8)

        # ── Botón A: pulsar → start ───────────────────────────────────────
        if a_pressed and not self._vr_a_held:
            self._vr_rec_start_event = True
        self._vr_a_held = a_pressed

        # ── Botón B: tap → stop/save  |  hold 1.5s → discard ─────────────
        if b_pressed and not self._vr_b_held:
            # B acaba de presionarse: empezar a contar
            self._vr_b_hold_start = _time.monotonic()

        if not b_pressed and self._vr_b_held:
            # B acaba de soltarse
            held_s = _time.monotonic() - (self._vr_b_hold_start or 0.0)
            if held_s >= self._B_HOLD_DISCARD_S:
                # Mantenido suficiente → descartar episodio
                self._vr_rec_discard_event = True
                print(f"[VR] B mantenido {held_s:.1f}s → DESCARTAR episodio")
            else:
                # Pulsación corta → guardar episodio (comportamiento original)
                self._vr_rec_stop_event = True
            self._vr_b_hold_start = None

        self._vr_b_held = b_pressed

    def consume_vr_record_events(self) -> tuple[bool, bool, bool]:
        """Retorna (start, stop/save, discard/rerecord).

        - start   : A pulsado  → empieza a grabar
        - stop    : B tap      → guarda el episodio
        - discard : B hold 1.5s → descarta el episodio (re-grabar)
        """
        start   = self._vr_rec_start_event
        stop    = self._vr_rec_stop_event
        discard = self._vr_rec_discard_event
        self._vr_rec_start_event   = False
        self._vr_rec_stop_event    = False
        self._vr_rec_discard_event = False
        return start, stop, discard

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected or self.teleop_core is None:
            raise DeviceNotConnectedError("iCub MuJoCo robot is not connected")

        if self.config.vr_enabled:
            self.teleop_core._process_vr_head()
            self.teleop_core._process_vr_hands()
            self._update_vr_record_events()
            if self.teleop_core.latest_buttons:
                self.teleop_core._process_button_commands()
                self.teleop_core.latest_buttons = None
            # El VR mueve los targets aquí, después de get_observation: la acción de este
            # paso es lo que se aplicó ahora (no lo que leyó el teleoperador, que es el target
            # del paso anterior, ya presente en la observación).
            action = self._applied_action()
        else:
            self._set_targets_from_action(action)

        # Advance physics precisely by the expected evaluation Delta Time (e.g. 1/30 fps)
        # to ensure hands fully close and physics speeds match real-time
        dt = self.teleop_core.model.opt.timestep
        sim_steps_needed = max(1, int(round(self.config.control_dt / dt)))
        t_phys = time.perf_counter()
        
        for i in range(sim_steps_needed):
            if i % max(1, self.teleop_core.frame_skip) == 0:
                self.teleop_core._solve_ik()
                self.teleop_core._interpolate_hands()
            mujoco.mj_step(self.teleop_core.model, self.teleop_core.data)
        self.metrics_meta["physics_ms"] = (time.perf_counter() - t_phys) * 1000.0
        self.metrics_meta["sim_steps"] = sim_steps_needed

        self._last_action = {
            k: self._to_float(v) if isinstance(v, torch.Tensor) else float(v)
            for k, v in action.items()
        }

        if self.config.vr_enabled:
             self.teleop_core._stream_images()

        return action

    def _applied_action(self) -> dict[str, Any]:
        """Targets y pinzas que tiene ahora el teleop_core, con las claves de action_features."""
        tc = self.teleop_core
        data = tc.data
        action: dict[str, Any] = {}
        for side, mocap_id in (("rh", tc.rh_mocap_id), ("lh", tc.lh_mocap_id)):
            for ax, v in zip("xyz", data.mocap_pos[mocap_id]):
                action[f"{side}_pos_{ax}"] = float(v)
            for ax, v in zip("wxyz", data.mocap_quat[mocap_id]):
                action[f"{side}_quat_{ax}"] = float(v)
            action[f"{side}_gripper"] = float(getattr(tc, f"{side}_grip_state", 0.0))
        for ax, v in zip("xyz", data.mocap_pos[tc.gaze_mocap_id]):
            action[f"gaze_{ax}"] = float(v)
        if self.config.pose_frame == "root":
            self._convert_poses(action, self._ACTION_POSE_GROUPS, to_root=True)
            if self.config.canonical_quats:
                canonicalize_quats(action)
        return {k: torch.tensor(self._to_float(action[k]), dtype=torch.float32)
                for k in self.action_features}

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected or self.teleop_core is None:
            raise DeviceNotConnectedError("iCub MuJoCo robot is not connected")

        obs: dict[str, Any] = {}
        tc = self.teleop_core
        model = tc.model
        data = tc.data
        # Estado y cámaras salen del mismo mjData sin avanzar la física entre medio:
        # state_sim_t == cam_sim_t por construcción (se registra para comprobarlo).
        self.metrics_meta = {"state_sim_t": float(data.time)}

        # 1. Joint positions
        state_vec = tc._get_robot_state()
        for name in self.actuator_list:
            if name in self._actuator_id_by_name:
                aid = self._actuator_id_by_name[name]
                obs[name] = torch.tensor(float(state_vec[aid]), dtype=torch.float32)
            else:
                obs[name] = torch.tensor(0.0, dtype=torch.float32)

        # 2. Joint velocity  — data.qvel indexed like qpos via jnt_dofadr 
        for name in self.actuator_list:
            if name in self._actuator_id_by_name:
                aid = self._actuator_id_by_name[name]
                trntype = model.actuator_trntype[aid]
                trnid   = model.actuator_trnid[aid, 0]
                if trntype == 0:  # mjTRN_JOINT
                    dof_adr = model.jnt_dofadr[trnid]
                    vel = float(data.qvel[dof_adr])
                elif trntype == 3:  # mjTRN_TENDON
                    vel = float(data.ten_velocity[trnid])
                else:
                    vel = 0.0
            else:
                vel = 0.0
            obs[f"{name}_vel"] = torch.tensor(vel, dtype=torch.float32)

        # 3. Joint torque  — data.actuator_force (computed after mj_step)
        for name in filter(has_torque, self.actuator_list):
            if name in self._actuator_id_by_name:
                aid = self._actuator_id_by_name[name]
                torque = float(data.actuator_force[aid])
            else:
                torque = 0.0
            obs[f"{name}_torque"] = torch.tensor(torque, dtype=torch.float32)

        # 4. End-effector poses  — site_xpos + mju_mat2Quat(site_xmat)
        #    Uses the same site IDs as _snap_targets_to_ee() in teleop_core
        ee_sites = {
            "rh": getattr(tc, "rh_site_id", None),
            "lh": getattr(tc, "lh_site_id", None),
        }
        for side, site_id in ee_sites.items():
            if site_id is not None:
                pos  = data.site_xpos[site_id].copy()
                quat = np.zeros(4, dtype=np.float64)
                mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
            else:
                pos  = np.zeros(3)
                quat = np.array([1.0, 0.0, 0.0, 0.0])
            obs[f"{side}_ee_pos_x"]  = torch.tensor(float(pos[0]),  dtype=torch.float32)
            obs[f"{side}_ee_pos_y"]  = torch.tensor(float(pos[1]),  dtype=torch.float32)
            obs[f"{side}_ee_pos_z"]  = torch.tensor(float(pos[2]),  dtype=torch.float32)
            obs[f"{side}_ee_quat_w"] = torch.tensor(float(quat[0]), dtype=torch.float32)
            obs[f"{side}_ee_quat_x"] = torch.tensor(float(quat[1]), dtype=torch.float32)
            obs[f"{side}_ee_quat_y"] = torch.tensor(float(quat[2]), dtype=torch.float32)
            obs[f"{side}_ee_quat_z"] = torch.tensor(float(quat[3]), dtype=torch.float32)

        # 5. Estimated external force on EEF bodies  — data.cfrc_ext
        #    cfrc_ext shape: (nbody, 6)  → [torque(3), force(3)]  in world frame
        #    We expose only the 3D force component (indices 3:6).
        ee_body_names = {"rh": "r_hand", "lh": "l_hand"}
        for side, body_name in ee_body_names.items():
            try:
                bid = model.body(body_name).id
                force = data.cfrc_ext[bid, 3:6].copy()   # (fx, fy, fz) world frame
            except Exception:
                force = np.zeros(3)
            obs[f"{side}_ext_force_x"] = torch.tensor(float(force[0]), dtype=torch.float32)
            obs[f"{side}_ext_force_y"] = torch.tensor(float(force[1]), dtype=torch.float32)
            obs[f"{side}_ext_force_z"] = torch.tensor(float(force[2]), dtype=torch.float32)

        # 6. Object poses — indexed by order in scenes.yaml
        scene_objects = getattr(self.config, "scene_objects", []) or []
        if not scene_objects:
            # Legacy fallback: try blue-cube as obj0
            scene_objects = [{"body": "blue-cube"}]

        for i, obj_def in enumerate(scene_objects):
            body_name = obj_def.get("body", "")
            prefix = f"obj{i}"
            try:
                bid = model.body(body_name).id
                if model.body_jntnum[bid] > 0:
                    jnt_adr = model.jnt_qposadr[model.body_jntadr[bid]]
                    pos = data.qpos[jnt_adr:jnt_adr + 3]
                    quat = data.qpos[jnt_adr + 3:jnt_adr + 7]
                else:
                    pos = data.xpos[bid]
                    quat = data.xquat[bid]

                obs[f"{prefix}_pos_x"] = torch.tensor(float(pos[0]), dtype=torch.float32)
                obs[f"{prefix}_pos_y"] = torch.tensor(float(pos[1]), dtype=torch.float32)
                obs[f"{prefix}_pos_z"] = torch.tensor(float(pos[2]), dtype=torch.float32)
                obs[f"{prefix}_quat_w"] = torch.tensor(float(quat[0]), dtype=torch.float32)
                obs[f"{prefix}_quat_x"] = torch.tensor(float(quat[1]), dtype=torch.float32)
                obs[f"{prefix}_quat_y"] = torch.tensor(float(quat[2]), dtype=torch.float32)
                obs[f"{prefix}_quat_z"] = torch.tensor(float(quat[3]), dtype=torch.float32)
            except Exception:
                # Body not in this scene — fill with zeros
                for suffix in ("pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z"):
                    obs[f"{prefix}_{suffix}"] = torch.tensor(0.0, dtype=torch.float32)

        # 7. Mocap target positions — always read from data.mocap_pos regardless
        #    of vr_enabled. During VR teleoperation the VR device moves these;
        #    during eval the policy moves them via _set_targets_from_action.
        #    Either way the current values are the ground truth for the observation.
        rh_id = tc.rh_mocap_id
        obs["rh_target_pos_x"]  = torch.tensor(float(data.mocap_pos[rh_id][0]),  dtype=torch.float32)
        obs["rh_target_pos_y"]  = torch.tensor(float(data.mocap_pos[rh_id][1]),  dtype=torch.float32)
        obs["rh_target_pos_z"]  = torch.tensor(float(data.mocap_pos[rh_id][2]),  dtype=torch.float32)
        obs["rh_target_quat_w"] = torch.tensor(float(data.mocap_quat[rh_id][0]), dtype=torch.float32)
        obs["rh_target_quat_x"] = torch.tensor(float(data.mocap_quat[rh_id][1]), dtype=torch.float32)
        obs["rh_target_quat_y"] = torch.tensor(float(data.mocap_quat[rh_id][2]), dtype=torch.float32)
        obs["rh_target_quat_z"] = torch.tensor(float(data.mocap_quat[rh_id][3]), dtype=torch.float32)

        lh_id = tc.lh_mocap_id
        obs["lh_target_pos_x"]  = torch.tensor(float(data.mocap_pos[lh_id][0]),  dtype=torch.float32)
        obs["lh_target_pos_y"]  = torch.tensor(float(data.mocap_pos[lh_id][1]),  dtype=torch.float32)
        obs["lh_target_pos_z"]  = torch.tensor(float(data.mocap_pos[lh_id][2]),  dtype=torch.float32)
        obs["lh_target_quat_w"] = torch.tensor(float(data.mocap_quat[lh_id][0]), dtype=torch.float32)
        obs["lh_target_quat_x"] = torch.tensor(float(data.mocap_quat[lh_id][1]), dtype=torch.float32)
        obs["lh_target_quat_y"] = torch.tensor(float(data.mocap_quat[lh_id][2]), dtype=torch.float32)
        obs["lh_target_quat_z"] = torch.tensor(float(data.mocap_quat[lh_id][3]), dtype=torch.float32)

        gaze_id = tc.gaze_mocap_id
        obs["gaze_target_pos_x"] = torch.tensor(float(data.mocap_pos[gaze_id][0]), dtype=torch.float32)
        obs["gaze_target_pos_y"] = torch.tensor(float(data.mocap_pos[gaze_id][1]), dtype=torch.float32)
        obs["gaze_target_pos_z"] = torch.tensor(float(data.mocap_pos[gaze_id][2]), dtype=torch.float32)

        # Fracción de cierre pedida (0 = abierta, 1 = cerrada), igual que la acción
        obs["rh_gripper"] = torch.tensor(float(getattr(tc, "rh_grip_state", 0.0)), dtype=torch.float32)
        obs["lh_gripper"] = torch.tensor(float(getattr(tc, "lh_grip_state", 0.0)), dtype=torch.float32)

        # 8. Camera images  (mocap target geoms hidden via group 2 filter)
        t_render = time.perf_counter()
        self.metrics_meta["cam_sim_t"] = float(data.time)
        for requested_name, renderer in self.renderers.items():
            cam_name = self._camera_name_resolved[requested_name]
            renderer.update_scene(data, camera=cam_name, scene_option=self._cam_scene_option)
            img = renderer.render()
            obs[requested_name] = img.copy()  # HWC uint8, igual que Gazebo
        self.metrics_meta["render_ms"] = (time.perf_counter() - t_render) * 1000.0

        # 9. Tactile contact — per-finger binary contact detection via data.contact.
        #    Only the fingertip (last phalanx _3) is used for touch detection.
        #      r_hand_{thumb,index,middle,ring,little}_3
        #      l_hand_{thumb,index,middle,ring,little}_3

        FINGERS = ("thumb", "index", "middle")
        HAND_PREFIX = {"rh": "r_hand", "lh": "l_hand"}

        def _geoms_of_fingertip(model: mujoco.MjModel, tip_body_name: str) -> set[int]:
            """Geom IDs for a fingertip body (last phalanx only)."""
            try:
                tip_id = model.body(tip_body_name).id
            except Exception:
                return set()
            return {gid for gid in range(model.ngeom) if model.geom_bodyid[gid] == tip_id}

        # Build and cache geom sets per fingertip (only once per session)
        if not hasattr(self, "_finger_geom_ids"):
            self._finger_geom_ids: dict[str, set[int]] = {}
            for side, prefix in HAND_PREFIX.items():
                for finger in FINGERS:
                    key = f"{side}_{finger}"
                    tip_name = f"{prefix}_{finger}_3"  # e.g. r_hand_thumb_3
                    self._finger_geom_ids[key] = _geoms_of_fingertip(model, tip_name)

        # Collect all active contact geom IDs
        active_geoms: set[int] = set()
        for i in range(data.ncon):
            active_geoms.add(data.contact[i].geom1)
            active_geoms.add(data.contact[i].geom2)

        # Check each finger
        for side in ("rh", "lh"):
            for finger in FINGERS:
                key = f"{side}_{finger}"
                touching = 1.0 if self._finger_geom_ids[key] & active_geoms else 0.0
                self._last_touch_state[key] = touching
                obs[f"{side}_touch_{finger}"] = torch.tensor(touching, dtype=torch.float32)

        # 10. Poses al frame root (como Gazebo y el robot real) y solo lo declarado en
        #     observation_features (brazos de control_arms, sin fuerza externa).
        if self.config.pose_frame == "root":
            self._convert_poses(obs, self._POSE_GROUPS, to_root=True)
            if self.config.canonical_quats:
                canonicalize_quats(obs)
        if getattr(self, "_obs_keys", None) is None:
            self._obs_keys = set(self.observation_features)
        return {k: v for k, v in obs.items() if k in self._obs_keys}

    def disconnect(self) -> None:
        if not self._connected:
            return

        for renderer in self.renderers.values():
            try:
                renderer.close()
            except Exception:
                pass
        self.renderers.clear()
        self.cameras.clear()

        # Liberar los puertos del VR y del streaming: si no, la sesión siguiente del Hub
        # (MuJoCo o Gazebo) falla con "Address already in use"
        if self.teleop_core is not None:
            self.teleop_core.close()
        self.teleop_core = None
        self._connected = False
