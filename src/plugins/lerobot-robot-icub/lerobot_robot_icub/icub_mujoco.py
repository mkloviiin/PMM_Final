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
from interfaces.teleop_mujoco import MuJoCoTeleop

from .config_icub_mujoco import iCubMuJoCoConfig


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
        self.metrics_meta: dict[str, Any] = {}
        self._last_touch_state: dict[str, float] = {}
        self._vr_rec_start_event = False
        self._vr_rec_stop_event = False
        self._vr_rec_discard_event = False
        self._vr_a_held = False
        self._vr_b_held = False
        self._vr_b_hold_start: float | None = None

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {actuator_name: float for actuator_name in self.actuator_list}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            name: (self.config.camera_height, self.config.camera_width, 3)
            for name in self.config.camera_names
        }

    def _arm_sides(self) -> list[str]:
        return [side for side, arms in (("rh", ("right", "both")), ("lh", ("left", "both")))
                if self.config.control_arms in arms]

    _POSE_GROUPS = [
        ("rh_ee_pos", "rh_ee_quat"),
        ("lh_ee_pos", "lh_ee_quat"),
        (None, "head_ee_quat"),
        ("rh_target_pos", "rh_target_quat"),
        ("lh_target_pos", "lh_target_quat"),
        (None, "head_target_quat"),
    ] + [(f"obj{i}_pos", f"obj{i}_quat") for i in range(8)]

    _ACTION_POSE_GROUPS = [
        ("rh_pos", "rh_quat"),
        ("lh_pos", "lh_quat"),
        (None, "head_quat"),
    ]

    def _convert_poses(self, values: dict, groups, to_root: bool) -> None:
        """Convert positions and quaternions between MuJoCo world frame and iCub root_link frame in-place."""
        for pos_prefix, quat_prefix in groups:
            if pos_prefix:
                pos_keys = [f"{pos_prefix}_{ax}" for ax in "xyz"]
                if all(k in values for k in pos_keys):
                    pos = np.array([self._to_float(values[k]) for k in pos_keys], dtype=np.float64)
                    if to_root:
                        new_pos = self._waist_R.T @ (pos - self._waist_pos)
                    else:
                        new_pos = self._waist_R @ pos + self._waist_pos
                    for k, v in zip(pos_keys, new_pos):
                        values[k] = torch.tensor(float(v), dtype=torch.float32)

            if quat_prefix:
                quat_keys = [f"{quat_prefix}_{ax}" for ax in "wxyz"]
                if all(k in values for k in quat_keys):
                    quat = np.array([self._to_float(values[k]) for k in quat_keys], dtype=np.float64)
                    new_quat = np.zeros(4)
                    mujoco.mju_mulQuat(new_quat, self._waist_quat_inv if to_root else self._waist_quat, quat)
                    for k, v in zip(quat_keys, new_quat):
                        values[k] = torch.tensor(float(v), dtype=torch.float32)

    @property
    def observation_features(self) -> dict[str, type]:
        features = {**self._motors_ft}

        # Joint velocities
        for name in self.actuator_list:
            features[f"{name}_vel"] = float

        sides = self._arm_sides()

        # End-effector poses (r_hand, l_hand SE(3) and head SO(3) in root_link frame)
        for side in sides:
            for ax in ("x", "y", "z"):
                features[f"{side}_ee_pos_{ax}"] = float
            for ax in ("w", "x", "y", "z"):
                features[f"{side}_ee_quat_{ax}"] = float
        if self.config.use_gaze:
            for ax in ("w", "x", "y", "z"):
                features[f"head_ee_quat_{ax}"] = float

        # Mocap target poses
        for side in sides:
            for ax in ("x", "y", "z"):
                features[f"{side}_target_pos_{ax}"] = float
            for ax in ("w", "x", "y", "z"):
                features[f"{side}_target_quat_{ax}"] = float
        if self.config.use_gaze:
            for ax in ("w", "x", "y", "z"):
                features[f"head_target_quat_{ax}"] = float
        for side in sides:
            features[f"{side}_gripper"] = float

        # Tactile contacts
        for side in sides:
            for finger in ("thumb", "index", "middle"):
                features[f"{side}_touch_{finger}"] = float

        # Scene object poses
        scene_objects = getattr(self.config, "scene_objects", []) or []
        num_objects = max(len(scene_objects), 1)
        for i in range(num_objects):
            for ax in ("x", "y", "z"):
                features[f"obj{i}_pos_{ax}"] = float
            for ax in ("w", "x", "y", "z"):
                features[f"obj{i}_quat_{ax}"] = float

        # Joint torques
        for name in self.actuator_list:
            features[f"{name}_torque"] = float

        # Cameras
        features.update(self._cameras_ft)
        return features

    @property
    def action_features(self) -> dict[str, type]:
        features = {}
        if self.config.control_arms in ["right", "both"]:
            for ax in ("x", "y", "z"):
                features[f"rh_pos_{ax}"] = float
            for ax in ("w", "x", "y", "z"):
                features[f"rh_quat_{ax}"] = float
            features["rh_gripper"] = float
        if self.config.control_arms in ["left", "both"]:
            for ax in ("x", "y", "z"):
                features[f"lh_pos_{ax}"] = float
            for ax in ("w", "x", "y", "z"):
                features[f"lh_quat_{ax}"] = float
            features["lh_gripper"] = float
        if self.config.use_gaze:
            for ax in ("w", "x", "y", "z"):
                features[f"head_quat_{ax}"] = float
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

        model_path = str(Path(self.config.model_path).expanduser().resolve())
        self.teleop_core = MuJoCoTeleop(
            model_path=model_path,
            controller_type=self.config.controller_type,
            impedance_mode=self.config.impedance_mode,
            vr_enabled=bool(self.config.vr_enabled),
            record_enabled=False,
            control_arms=self.config.control_arms,
            primary_arm=self.config.primary_arm,
            config_path=self.config.config_path,
            vr_ip=self.config.vr_ip,
            scene_objects=self.config.scene_objects,
            control_dt=self.config.control_dt,
        )
        self.teleop_core._scene_joints = list(self.config.scene_joints or [])
        self.teleop_core.state = "teleop_active"

        model = self.teleop_core.model

        # Cache waist (root_link) transform
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
        self._cam_scene_option = mujoco.MjvOption()
        self._cam_scene_option.geomgroup[2] = 0
        self._cam_scene_option.sitegroup[2] = 0

        for requested_name in self.config.camera_names:
            resolved_name = self.config.camera_sources.get(requested_name, requested_name)
            try:
                model.camera(resolved_name).id
            except Exception:
                print(f"[WARN] Camera '{resolved_name}' ({requested_name}) not found: fallback to head_cam.")
                resolved_name = "head_cam"
            self._camera_name_resolved[requested_name] = resolved_name
            try:
                self.renderers[requested_name] = mujoco.Renderer(
                    model,
                    height=self.config.camera_height,
                    width=self.config.camera_width,
                )
                self.cameras[requested_name] = self.renderers[requested_name]
            except Exception as e:
                print(f"[WARN] Failed to initialize renderer for '{requested_name}' (running headless/no GL): {e}")

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
            norm = np.linalg.norm(rh_quat)
            rh_quat = (rh_quat / norm) if norm > 1e-6 else np.array([1.0, 0.0, 0.0, 0.0])
            data.mocap_pos[self.teleop_core.rh_mocap_id] = rh_pos
            data.mocap_quat[self.teleop_core.rh_mocap_id] = rh_quat

            if "rh_gripper" in action:
                apply_grip("right", self._to_float(action["rh_gripper"]))

        if self.config.control_arms in ["left", "both"]:
            if any(k in action for k in ("lh_pos_x", "lh_x")):
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
                norm = np.linalg.norm(lh_quat)
                lh_quat = (lh_quat / norm) if norm > 1e-6 else np.array([1.0, 0.0, 0.0, 0.0])
                data.mocap_pos[self.teleop_core.lh_mocap_id] = lh_pos
                data.mocap_quat[self.teleop_core.lh_mocap_id] = lh_quat

            if "lh_gripper" in action:
                apply_grip("left", self._to_float(action["lh_gripper"]))

        if self.config.use_gaze and any(k in action for k in ("head_quat_w", "gaze_w")):
            head_quat = np.array([
                self._action_value(action, "head_quat_w", "gaze_w"),
                self._action_value(action, "head_quat_x", "gaze_x"),
                self._action_value(action, "head_quat_y", "gaze_y"),
                self._action_value(action, "head_quat_z", "gaze_z"),
            ], dtype=np.float64)
            norm = np.linalg.norm(head_quat)
            head_quat = (head_quat / norm) if norm > 1e-6 else np.array([1.0, 0.0, 0.0, 0.0])
            data.mocap_quat[self.teleop_core.head_mocap_id] = head_quat

    _B_HOLD_DISCARD_S: float = 1.5

    def _update_vr_record_events(self) -> None:
        if self.teleop_core is None:
            return
        b = self.teleop_core.latest_buttons
        if b is None or not isinstance(b, dict):
            return

        a_pressed = (b.get("BTN_ONE_RIGHT", 0) > 0.5) or (b.get("A", 0) > 0.8)
        b_pressed = (b.get("BTN_TWO_RIGHT", 0) > 0.5) or (b.get("B", 0) > 0.8)

        if a_pressed and not self._vr_a_held:
            self._vr_rec_start_event = True
        self._vr_a_held = a_pressed

        if b_pressed and not self._vr_b_held:
            self._vr_b_hold_start = time.monotonic()

        # Discard as soon as B has been held long enough (the VR status changes: the operator
        # knows when to let go); releasing it afterwards does nothing. A short press = STOP/SAVE.
        if (b_pressed and self._vr_b_hold_start is not None
                and time.monotonic() - self._vr_b_hold_start >= self._B_HOLD_DISCARD_S):
            self._vr_rec_discard_event = True
            self._vr_b_hold_start = None
            print(f"[VR] B held {self._B_HOLD_DISCARD_S:.1f}s → DISCARD episode")

        if not b_pressed and self._vr_b_held:
            if self._vr_b_hold_start is not None:
                self._vr_rec_stop_event = True
            self._vr_b_hold_start = None

        self._vr_b_held = b_pressed

    def consume_vr_record_events(self) -> tuple[bool, bool, bool]:
        start = self._vr_rec_start_event
        stop = self._vr_rec_stop_event
        discard = self._vr_rec_discard_event
        self._vr_rec_start_event = False
        self._vr_rec_stop_event = False
        self._vr_rec_discard_event = False
        return start, stop, discard

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
        for ax, v in zip("wxyz", data.mocap_quat[tc.head_mocap_id]):
            action[f"head_quat_{ax}"] = float(v)
        if self.config.pose_frame == "root":
            self._convert_poses(action, self._ACTION_POSE_GROUPS, to_root=True)
            if self.config.canonical_quats:
                canonicalize_quats(action)
        return {k: torch.tensor(self._to_float(action[k]), dtype=torch.float32)
                for k in self.action_features}

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
            action = self._applied_action()
        else:
            self._set_targets_from_action(action)

        dt = self.teleop_core.model.opt.timestep
        sim_steps_needed = max(1, int(round(self.config.control_dt / dt)))
        t_phys = time.perf_counter()

        if self.teleop_core.state == "going_home":
            self.teleop_core._interpolate_home()
        elif self.teleop_core.state == "teleop_active":
            self.teleop_core._solve_whole_body()
        self.teleop_core._interpolate_hands()

        for _ in range(sim_steps_needed):
            if self.teleop_core.controller_type == "id":
                self.teleop_core._update_solver_from_mj()
                tau_opt = self.teleop_core.solver.step()
                self.teleop_core.data.ctrl[self.teleop_core._upper_act_ids] = tau_opt
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

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected or self.teleop_core is None:
            raise DeviceNotConnectedError("iCub MuJoCo robot is not connected")

        obs: dict[str, Any] = {}
        tc = self.teleop_core
        model = tc.model
        data = tc.data
        self.metrics_meta = {"state_sim_t": float(data.time)}

        # 1. Joint positions
        state_vec = tc._get_robot_state()
        for name in self.actuator_list:
            if name in self._actuator_id_by_name:
                aid = self._actuator_id_by_name[name]
                obs[name] = torch.tensor(float(state_vec[aid]), dtype=torch.float32)
            else:
                obs[name] = torch.tensor(0.0, dtype=torch.float32)

        # 2. Joint velocities
        for name in self.actuator_list:
            if name in self._actuator_id_by_name:
                aid = self._actuator_id_by_name[name]
                trntype = model.actuator_trntype[aid]
                trnid = model.actuator_trnid[aid, 0]
                if trntype == 0:
                    dof_adr = model.jnt_dofadr[trnid]
                    vel = float(data.qvel[dof_adr])
                elif trntype == 3:
                    vel = float(data.ten_velocity[trnid])
                else:
                    vel = 0.0
            else:
                vel = 0.0
            obs[f"{name}_vel"] = torch.tensor(vel, dtype=torch.float32)

        # 3. End-effector poses (r_hand, l_hand SE(3) and head SO(3))
        ee_sites = {
            "rh": getattr(tc, "rh_site_id", None),
            "lh": getattr(tc, "lh_site_id", None),
        }
        for side, site_id in ee_sites.items():
            if site_id is not None:
                pos = data.site_xpos[site_id].copy()
                quat = np.zeros(4, dtype=np.float64)
                mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
                if quat[0] < 0:
                    quat = -quat
            else:
                pos = np.zeros(3)
                quat = np.array([1.0, 0.0, 0.0, 0.0])
            obs[f"{side}_ee_pos_x"] = torch.tensor(float(pos[0]), dtype=torch.float32)
            obs[f"{side}_ee_pos_y"] = torch.tensor(float(pos[1]), dtype=torch.float32)
            obs[f"{side}_ee_pos_z"] = torch.tensor(float(pos[2]), dtype=torch.float32)
            obs[f"{side}_ee_quat_w"] = torch.tensor(float(quat[0]), dtype=torch.float32)
            obs[f"{side}_ee_quat_x"] = torch.tensor(float(quat[1]), dtype=torch.float32)
            obs[f"{side}_ee_quat_y"] = torch.tensor(float(quat[2]), dtype=torch.float32)
            obs[f"{side}_ee_quat_z"] = torch.tensor(float(quat[3]), dtype=torch.float32)

        if self.config.use_gaze:
            head_site_id = getattr(tc, "head_site_id", None)
            if head_site_id is not None:
                quat_head = np.zeros(4, dtype=np.float64)
                mujoco.mju_mat2Quat(quat_head, data.site_xmat[head_site_id])
                if quat_head[0] < 0:
                    quat_head = -quat_head
            else:
                quat_head = np.array([1.0, 0.0, 0.0, 0.0])
            obs["head_ee_quat_w"] = torch.tensor(float(quat_head[0]), dtype=torch.float32)
            obs["head_ee_quat_x"] = torch.tensor(float(quat_head[1]), dtype=torch.float32)
            obs["head_ee_quat_y"] = torch.tensor(float(quat_head[2]), dtype=torch.float32)
            obs["head_ee_quat_z"] = torch.tensor(float(quat_head[3]), dtype=torch.float32)

        # 4. Mocap target poses
        rh_id = tc.rh_mocap_id
        obs["rh_target_pos_x"] = torch.tensor(float(data.mocap_pos[rh_id][0]), dtype=torch.float32)
        obs["rh_target_pos_y"] = torch.tensor(float(data.mocap_pos[rh_id][1]), dtype=torch.float32)
        obs["rh_target_pos_z"] = torch.tensor(float(data.mocap_pos[rh_id][2]), dtype=torch.float32)
        obs["rh_target_quat_w"] = torch.tensor(float(data.mocap_quat[rh_id][0]), dtype=torch.float32)
        obs["rh_target_quat_x"] = torch.tensor(float(data.mocap_quat[rh_id][1]), dtype=torch.float32)
        obs["rh_target_quat_y"] = torch.tensor(float(data.mocap_quat[rh_id][2]), dtype=torch.float32)
        obs["rh_target_quat_z"] = torch.tensor(float(data.mocap_quat[rh_id][3]), dtype=torch.float32)

        lh_id = tc.lh_mocap_id
        obs["lh_target_pos_x"] = torch.tensor(float(data.mocap_pos[lh_id][0]), dtype=torch.float32)
        obs["lh_target_pos_y"] = torch.tensor(float(data.mocap_pos[lh_id][1]), dtype=torch.float32)
        obs["lh_target_pos_z"] = torch.tensor(float(data.mocap_pos[lh_id][2]), dtype=torch.float32)
        obs["lh_target_quat_w"] = torch.tensor(float(data.mocap_quat[lh_id][0]), dtype=torch.float32)
        obs["lh_target_quat_x"] = torch.tensor(float(data.mocap_quat[lh_id][1]), dtype=torch.float32)
        obs["lh_target_quat_y"] = torch.tensor(float(data.mocap_quat[lh_id][2]), dtype=torch.float32)
        obs["lh_target_quat_z"] = torch.tensor(float(data.mocap_quat[lh_id][3]), dtype=torch.float32)

        if self.config.use_gaze:
            head_id = tc.head_mocap_id
            obs["head_target_quat_w"] = torch.tensor(float(data.mocap_quat[head_id][0]), dtype=torch.float32)
            obs["head_target_quat_x"] = torch.tensor(float(data.mocap_quat[head_id][1]), dtype=torch.float32)
            obs["head_target_quat_y"] = torch.tensor(float(data.mocap_quat[head_id][2]), dtype=torch.float32)
            obs["head_target_quat_z"] = torch.tensor(float(data.mocap_quat[head_id][3]), dtype=torch.float32)

        obs["rh_gripper"] = torch.tensor(float(getattr(tc, "rh_grip_state", 0.0)), dtype=torch.float32)
        obs["lh_gripper"] = torch.tensor(float(getattr(tc, "lh_grip_state", 0.0)), dtype=torch.float32)

        # 5. Tactile contacts
        FINGERS = ("thumb", "index", "middle")
        HAND_PREFIX = {"rh": "r_hand", "lh": "l_hand"}

        def _geoms_of_fingertip(model: mujoco.MjModel, tip_body_name: str) -> set[int]:
            try:
                tip_id = model.body(tip_body_name).id
            except Exception:
                return set()
            return {gid for gid in range(model.ngeom) if model.geom_bodyid[gid] == tip_id}

        if not hasattr(self, "_finger_geom_ids"):
            self._finger_geom_ids: dict[str, set[int]] = {}
            for side, prefix in HAND_PREFIX.items():
                for finger in FINGERS:
                    key = f"{side}_{finger}"
                    tip_name = f"{prefix}_{finger}_3"
                    self._finger_geom_ids[key] = _geoms_of_fingertip(model, tip_name)

        active_geoms: set[int] = set()
        for i in range(data.ncon):
            active_geoms.add(data.contact[i].geom1)
            active_geoms.add(data.contact[i].geom2)

        for side in ("rh", "lh"):
            for finger in FINGERS:
                key = f"{side}_{finger}"
                touching = 1.0 if self._finger_geom_ids[key] & active_geoms else 0.0
                self._last_touch_state[key] = touching
                obs[f"{side}_touch_{finger}"] = torch.tensor(touching, dtype=torch.float32)

        # 6. Scene objects
        scene_objects = getattr(self.config, "scene_objects", []) or []
        if not scene_objects:
            scene_objects = [{"body": "blue-cube"}]

        absent_objects = []
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
                absent_objects.append(prefix)
                for suffix in ("pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z"):
                    obs[f"{prefix}_{suffix}"] = torch.tensor(0.0, dtype=torch.float32)

        # 7. Joint torques
        for name in self.actuator_list:
            if name in self._actuator_id_by_name:
                aid = self._actuator_id_by_name[name]
                torque = float(data.actuator_force[aid])
            else:
                torque = 0.0
            obs[f"{name}_torque"] = torch.tensor(torque, dtype=torch.float32)

        # 8. Cameras
        t_render = time.perf_counter()
        self.metrics_meta["cam_sim_t"] = float(data.time)
        for requested_name in self.config.camera_names:
            if requested_name in self.renderers:
                renderer = self.renderers[requested_name]
                cam_name = self._camera_name_resolved[requested_name]
                renderer.update_scene(data, camera=cam_name, scene_option=self._cam_scene_option)
                img = renderer.render()
                obs[requested_name] = img.copy()
            else:
                obs[requested_name] = np.zeros((self.config.camera_height, self.config.camera_width, 3), dtype=np.uint8)
        self.metrics_meta["render_ms"] = (time.perf_counter() - t_render) * 1000.0

        # 9. Frame conversion to root_link
        if self.config.pose_frame == "root":
            self._convert_poses(obs, self._POSE_GROUPS, to_root=True)
            if self.config.canonical_quats:
                canonicalize_quats(obs)
            # Absent object (e.g. the door scene, no cube): zeros in the root frame too, like
            # Gazebo; the conversion would otherwise leave obj_pos_z = -0.73 (the waist).
            for prefix in absent_objects:
                for suffix in ("pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z"):
                    obs[f"{prefix}_{suffix}"] = torch.tensor(0.0, dtype=torch.float32)

        if getattr(self, "_obs_keys", None) is None:
            self._obs_keys = set(self.observation_features)
        return {k: v for k, v in obs.items() if k in self._obs_keys}

    @property
    def is_teleop_active(self) -> bool:
        if self.teleop_core is not None:
            return self.teleop_core.state == "teleop_active"
        return False

    def toggle_teleop(self) -> bool:
        if self.teleop_core is None:
            return False
        if self.teleop_core.state == "teleop_active":
            self.stop_teleop()
            return False
        else:
            self.start_teleop()
            return True

    def start_teleop(self) -> None:
        if self.teleop_core is not None and self.teleop_core.state != "teleop_active":
            self.teleop_core._snap_targets_to_ee()
            self.teleop_core.state = "teleop_active"
            print("[State] → teleop_active (resumed)")

    def stop_teleop(self) -> None:
        if self.teleop_core is not None and self.teleop_core.state == "teleop_active":
            self.teleop_core._snap_targets_to_ee()
            self.teleop_core.state = "idle"
            print("[State] teleop → idle (paused)")

    def go_home(self) -> bool:
        if self.teleop_core is not None:
            self.teleop_core._start_go_home()
            return True
        return False

    def toggle_impedance(self) -> bool:
        if self.teleop_core is None or getattr(self.teleop_core, "controller_type", "") != "id":
            print("[Teleop] Modo impedancia solo disponible con --controller id")
            return False
        new_mode = not getattr(self.teleop_core, "impedance_mode", False)
        self.teleop_core.set_impedance_mode(new_mode)
        print(f"[Teleop] Modo impedancia: {'ACTIVADO' if new_mode else 'DESACTIVADO'}")
        return new_mode

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

        if self.teleop_core is not None:
            self.teleop_core.close()
        self.teleop_core = None
        self._connected = False

