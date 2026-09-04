from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np
import torch

from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_icub_mujoco import iCubMuJoCoConfig


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
        self._last_touch_state: dict[str, float] = {}
        self._rh_grip_mode: str = "stop"
        self._lh_grip_mode: str = "stop"
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
        return {
            name: (3, self.config.camera_height, self.config.camera_width)
            for name in self.config.camera_names
        }

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
        #   [90–111]   joint torques              (22)
        #   [112–117]  external forces rh + lh     (6)
        #   ── 118 total (+ cameras as separate keys) ─
        features = {**self._motors_ft}   # [0-21] joint positions

        # [22-43] joint velocities
        for name in self.actuator_list:
            features[f"{name}_vel"] = float

        # [44-57] End-effector poses (x,y,z + quat w,x,y,z)
        for side in ("rh", "lh"):
            for ax in ("x", "y", "z"):
                features[f"{side}_ee_pos_{ax}"] = float
            for ax in ("w", "x", "y", "z"):
                features[f"{side}_ee_quat_{ax}"] = float

        # [58-76] Mocap target poses — always present (VR moves them during
        #         teleoperation; policy moves them during eval via send_action)
        for ax in ("x", "y", "z"):
            features[f"rh_target_pos_{ax}"] = float
        for ax in ("w", "x", "y", "z"):
            features[f"rh_target_quat_{ax}"] = float
        for ax in ("x", "y", "z"):
            features[f"lh_target_pos_{ax}"] = float
        for ax in ("w", "x", "y", "z"):
            features[f"lh_target_quat_{ax}"] = float
        for ax in ("x", "y", "z"):
            features[f"gaze_target_pos_{ax}"] = float
        features["rh_gripper"] = float
        features["lh_gripper"] = float

        # [77-82] Tactile contact (binary per finger)
        for side in ("rh", "lh"):
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

        # joint torques / actuator forces
        for name in self.actuator_list:
            features[f"{name}_torque"] = float

        # estimated external forces on EEF bodies
        for side in ("rh", "lh"):
            for ax in ("x", "y", "z"):
                features[f"{side}_ext_force_{ax}"] = float

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
        )

        self.teleop_core.state = "teleop_active"

        model = self.teleop_core.model

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
        for requested_name in self.config.camera_names:
            resolved_name = requested_name
            try:
                model.camera(requested_name).id
            except Exception:
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

        def grip_to_mode(grip_value: float) -> str:
            if grip_value <= 0.1:
                return "open"
            if 0.4 <= grip_value <= 0.6:
                return "close"
            return "stop"

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
                grip = self._to_float(action["rh_gripper"])
                self.teleop_core.rh_grip_state = grip
                mode = grip_to_mode(grip)
                if mode != self._rh_grip_mode:
                    if mode in ("open", "close"):
                        self.teleop_core._set_hand("right", mode)
                    self._rh_grip_mode = mode

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
                grip = self._to_float(action["lh_gripper"])
                self.teleop_core.lh_grip_state = grip
                mode = grip_to_mode(grip)
                if mode != self._lh_grip_mode:
                    if mode in ("open", "close"):
                        self.teleop_core._set_hand("left", mode)
                    self._lh_grip_mode = mode

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
        else:
            self._set_targets_from_action(action)

        # Advance physics precisely by the expected evaluation Delta Time (e.g. 1/30 fps)
        # to ensure hands fully close and physics speeds match real-time
        dt = self.teleop_core.model.opt.timestep
        sim_steps_needed = max(1, int(round(self.config.control_dt / dt)))
        
        for i in range(sim_steps_needed):
            if i % max(1, self.teleop_core.frame_skip) == 0:
                self.teleop_core._solve_ik()
                self.teleop_core._interpolate_hands()
            mujoco.mj_step(self.teleop_core.model, self.teleop_core.data)

        self._last_action = {
            k: self._to_float(v) if isinstance(v, torch.Tensor) else float(v)
            for k, v in action.items()
        }

        if self.config.vr_enabled:
             self.teleop_core._stream_images()

        return action

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected or self.teleop_core is None:
            raise DeviceNotConnectedError("iCub MuJoCo robot is not connected")

        obs: dict[str, Any] = {}
        tc = self.teleop_core
        model = tc.model
        data = tc.data

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
        for name in self.actuator_list:
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

        obs["rh_gripper"] = torch.tensor(float(getattr(tc, "rh_grip_state", 1.0)), dtype=torch.float32)
        obs["lh_gripper"] = torch.tensor(float(getattr(tc, "lh_grip_state", 1.0)), dtype=torch.float32)

        # 8. Camera images  (mocap target geoms hidden via group 2 filter)
        for requested_name, renderer in self.renderers.items():
            cam_name = self._camera_name_resolved[requested_name]
            renderer.update_scene(data, camera=cam_name, scene_option=self._cam_scene_option)
            img = renderer.render()
            obs[requested_name] = torch.from_numpy(img.copy()).permute(2, 0, 1)

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

        return obs

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

        self.teleop_core = None
        self._connected = False
