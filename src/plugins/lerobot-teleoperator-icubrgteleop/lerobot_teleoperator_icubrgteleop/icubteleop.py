from collections import deque
import json
import os
import threading
import time
from typing import Any

import glfw
import mujoco
import mujoco.viewer
import numpy as np
import torch

try:
    import zmq
except ImportError:
    zmq = None

try:
    from pyquaternion import Quaternion as PyQuaternion
except ImportError:
    PyQuaternion = None

try:
    from lerobot.teleoperators.teleoperator import Teleoperator
except ImportError:
    class Teleoperator:
        def __init__(self, config):
            self.config = config

from .config_icubteleop import iCubTeleopConfig, load_control_config

VR_HEAD_PORT = 8115
VR_HAND_RIGHT_PORT = 8117
VR_BUTTONS_PORT = 8119
VR_HAND_LEFT_PORT = 8121

# Convert Unity left-handed (RUF: +X right, +Y up, +Z forward) quaternion
# to MuJoCo / ROS FLU right-handed (+X forward, +Y left, +Z up) quaternion:
# Unity: (x, y, z, w) -> ROS FLU: (z, -x, y, -w)
def unity_to_world_quat(q_unity: PyQuaternion) -> PyQuaternion:
    return PyQuaternion(w=-q_unity.w, x=q_unity.z, y=-q_unity.x, z=q_unity.y)


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(q, R.flatten())
    if q[0] < 0:
        q = -q
    return q


class iCubTeleop(Teleoperator):
    config_class = iCubTeleopConfig
    name = "icubrg_teleop"
    frame_skip = 1

    _B_HOLD_DISCARD_S: float = 1.5

    def __init__(self, config: iCubTeleopConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._calibrated = False
        self._reset_request = False
        self._pending_visual_reset = False
        self._teleop_toggle_requested = False
        self._reset_home_requested = False

        self.rh_grip_state = 0.0
        self.lh_grip_state = 0.0
        self._grip_pending = {"rh": None, "lh": None}

        self.model = None
        self.data = None
        self.viewer = None

        self.mocap_ids = {}
        self.site_ids = {}

        self.actuator_list = []
        for part in self.config.actuators_to_use:
            self.actuator_list.extend(self.config.actuators_to_use[part])

        self.qpos_map = []

        self._vr_running = True
        self._vr_sockets: dict = {}
        self._vr_thread = None
        self.latest_rh_data = None
        self.latest_lh_data = None
        self.latest_buttons = None
        self._last_btn_recv_time = 0.0
        self._btn_lock = threading.Lock()
        self._btn_state: dict = {}
        self._btn_pressed: dict[str, bool] = {}
        self._btn_events: deque = deque()
        self.calibrated_origin_rh = None
        self.calibrated_origin_lh = None
        self.calibrated_quat_rh = None
        self.calibrated_quat_lh = None
        self.calibrated_quat_head = None
        self._vr_home = {"right": np.array([0.35, -0.15, 0.91]), "left": np.array([0.35, 0.15, 0.91])}
        self._vr_home_rot = {"right": np.eye(3), "left": np.eye(3), "head": np.eye(3)}
        self._mocap_from_robot = {"right": False, "left": False, "head": False}
        self._prev_target_pos = {"right": None, "left": None}
        self._target_filter_alpha = 0.25
        self.freeze_rh = False
        self.freeze_lh = False
        self.freeze_head = False

        self._vr_rec_start_event = False
        self._vr_rec_stop_event = False
        self._vr_rec_discard_event = False
        self._vr_b_hold_start: float | None = None

        self._scene_objects: list[dict] = getattr(self.config, "scene_objects", []) or []
        self._scene_joints: list[dict] = getattr(self.config, "scene_joints", []) or []
        self._last_reset_poses: dict[str, tuple] = {}
        self._world_reset_event = False
        self._last_world_reset_t = 0.0

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
        if getattr(self.config, "use_gaze", False):
            for ax in ("w", "x", "y", "z"):
                features[f"head_quat_{ax}"] = float
        return features

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {actuator_name: float for actuator_name in self.actuator_list}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {name: (3, 240, 320) for name in ["left", "right"]}

    @property
    def feedback_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            return

        print(f"[Teleop] Loading MuJoCo model from: {self.config.model_path}")
        try:
            self.model = mujoco.MjModel.from_xml_path(self.config.model_path)
            self.data = mujoco.MjData(self.model)

            cfg = load_control_config(self.config.config_path)
            sim_dt = cfg.get("sim_dt", 0.005)
            control_dt = cfg.get("control_dt", 0.02)

            self.model.opt.timestep = sim_dt
            self.frame_skip = max(1, int(round(control_dt / sim_dt)))
        except Exception as e:
            raise ConnectionError(f"Failed to load MuJoCo model: {e}")

        try:
            home_id = self.model.key("home").id
            mujoco.mj_resetDataKeyframe(self.model, self.data, home_id)
        except Exception:
            self.data.qpos[:] = self.model.qpos0
            self.data.qvel[:] = 0

        self._apply_home_pose_overrides(cfg)
        mujoco.mj_forward(self.model, self.data)

        # Cache waist (root_link) transform
        waist_id = self.model.body("icub").id
        self._waist_pos = self.data.xpos[waist_id].copy()
        self._waist_R = self.data.xmat[waist_id].reshape(3, 3).copy()
        waist_quat = np.zeros(4)
        mujoco.mju_mat2Quat(waist_quat, self.data.xmat[waist_id].copy())
        self._waist_quat_inv = np.zeros(4)
        mujoco.mju_negQuat(self._waist_quat_inv, waist_quat)

        self.qpos_map = [self._resolve_qpos_index(name) for name in self.actuator_list]

        if self.config.control_arms in ["right", "both"]:
            self.mocap_ids["right"] = self.model.body("r_hand_target").mocapid[0]
            try:
                self.site_ids["right"] = self.model.site("r_hand_dh_frame_site").id
            except Exception:
                self.site_ids["right"] = self.model.site("r_hand_site").id

        if self.config.control_arms in ["left", "both"]:
            self.mocap_ids["left"] = self.model.body("l_hand_target").mocapid[0]
            try:
                self.site_ids["left"] = self.model.site("l_hand_dh_frame_site").id
            except Exception:
                self.site_ids["left"] = self.model.site("l_hand_site").id

        if self.config.use_gaze:
            try:
                self.mocap_ids["head"] = self.model.body("head_target").mocapid[0]
            except Exception:
                self.mocap_ids["head"] = self.model.body("com_target").mocapid[0]

        self._reset_mocaps_to_xml_pose()

        can_render = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if can_render:
            try:
                import glfw
                can_render = bool(glfw.init())
            except Exception:
                can_render = False

        if can_render:
            try:
                print("Launching MuJoCo Viewer... (keys: 8/9 open/close right hand, 6/7 left hand, N reset)")
                print("Mouse: Ctrl + Right-click drag to move mocap targets")
                self.viewer = mujoco.viewer.launch_passive(
                    self.model,
                    self.data,
                    key_callback=self._key_callback,
                    show_left_ui=True,
                    show_right_ui=True,
                )
                self.viewer.cam.lookat[:] = [0.35, 0.0, 0.9]
                self.viewer.cam.distance = 2.5
                self.viewer.cam.elevation = -65
                self.viewer.cam.azimuth = 0
                self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            except Exception as e:
                print(f"[Teleop] Viewer launch skipped: {e}")
                self.viewer = None
        else:
            print("[Teleop] No display or GLFW platform detected — running headless (viewer disabled).")
            self.viewer = None
        self._connected = True

        if getattr(self.config, "vr_enabled", False):
            self._init_vr()

        if calibrate:
            self.calibrate()

    def disconnect(self) -> None:
        self._vr_running = False
        if self._vr_thread is not None and self._vr_thread.is_alive():
            self._vr_thread.join(timeout=1.0)
        for _, (ctx, sock) in self._vr_sockets.items():
            try:
                sock.close(linger=0)
                ctx.term()
            except Exception:
                pass
        self._vr_sockets = {}
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        self._connected = False

    def calibrate(self) -> None:
        if self.is_connected:
            self._calibrated = True

    def configure(self) -> None:
        pass

    def _world_to_root(self, pos, quat_wxyz=None):
        pos_root = self._waist_R.T @ (np.asarray(pos, dtype=float) - self._waist_pos)
        if quat_wxyz is None:
            return pos_root, None
        quat_root = np.zeros(4)
        mujoco.mju_mulQuat(quat_root, self._waist_quat_inv, np.asarray(quat_wxyz, dtype=float))
        return pos_root, quat_root

    def _root_to_world(self, pos_root, quat_root_wxyz):
        pos_w = self._waist_R @ np.asarray(pos_root, dtype=float) + self._waist_pos
        waist_quat = np.zeros(4)
        mujoco.mju_negQuat(waist_quat, self._waist_quat_inv)
        quat_w = np.zeros(4)
        mujoco.mju_mulQuat(quat_w, waist_quat, np.asarray(quat_root_wxyz, dtype=float))
        return pos_w, quat_w

    def _init_mocaps_from_robot(self, feedback: dict[str, Any]) -> None:
        for side, key in (("right", "rh_current_pose"), ("left", "lh_current_pose")):
            if self._mocap_from_robot[side] or side not in self.mocap_ids or key not in feedback:
                continue
            vals = feedback[key]
            pos_w, quat_w = self._root_to_world(vals[:3], vals[3:7])
            mid = self.mocap_ids[side]
            self.data.mocap_pos[mid] = pos_w
            self.data.mocap_quat[mid] = quat_w
            self._vr_home[side] = pos_w.copy()
            mat = np.zeros(9)
            mujoco.mju_quat2Mat(mat, quat_w)
            self._vr_home_rot[side] = mat.reshape(3, 3)
            self._prev_target_pos[side] = None
            if side == "right":
                self.calibrated_origin_rh = None
                self.calibrated_quat_rh = None
            else:
                self.calibrated_origin_lh = None
                self.calibrated_quat_lh = None
            self._mocap_from_robot[side] = True
            print(f"[Teleop] {side} mocap initialized to robot hand pose.")

        if "head" in self.mocap_ids and not self._mocap_from_robot["head"] and "head_ee_quat_w" in feedback:
            quat_root = [feedback[f"head_ee_quat_{ax}"].item() for ax in "wxyz"]
            _, quat_w = self._root_to_world([0, 0, 0], quat_root)
            self.data.mocap_quat[self.mocap_ids["head"]] = quat_w
            mat = np.zeros(9)
            mujoco.mju_quat2Mat(mat, quat_w)
            self._vr_home_rot["head"] = mat.reshape(3, 3)
            self.calibrated_quat_head = None
            self._mocap_from_robot["head"] = True

    def _apply_home_pose_overrides(self, cfg: dict[str, Any]) -> None:
        home_pose = (cfg or {}).get("home_pose", {}) or {}
        if not home_pose:
            return

        for joint_name, deg_value in home_pose.items():
            try:
                rad_value = np.deg2rad(float(deg_value))
            except Exception:
                continue

            try:
                joint_id = self.model.joint(joint_name).id
                qpos_adr = self.model.jnt_qposadr[joint_id]
                self.data.qpos[qpos_adr] = rad_value
            except Exception:
                pass

            try:
                actuator_id = self.model.actuator(joint_name).id
                self.data.ctrl[actuator_id] = rad_value
            except Exception:
                pass

        self.data.qvel[:] = 0

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError("Teleoperator not connected")

        if getattr(self.config, "vr_enabled", False):
            self._process_vr_hands()
            self._process_vr_head()
            self._process_vr_button_commands()
            self._update_vr_record_events()

        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()
            mujoco.mj_forward(self.model, self.data)

        actions = {}

        if self._pending_visual_reset:
            self._reset_scenario()
            self._pending_visual_reset = False
            now = time.time()
            if now - self._last_world_reset_t > 2.0:
                self._world_reset_event = True
                self._last_world_reset_t = now

        if self.config.control_arms in ["right", "both"]:
            mid = self.mocap_ids["right"]
            pos, quat = self._world_to_root(self.data.mocap_pos[mid], self.data.mocap_quat[mid])
            actions["rh_pos_x"] = torch.tensor(pos[0], dtype=torch.float32)
            actions["rh_pos_y"] = torch.tensor(pos[1], dtype=torch.float32)
            actions["rh_pos_z"] = torch.tensor(pos[2], dtype=torch.float32)
            actions["rh_quat_w"] = torch.tensor(quat[0], dtype=torch.float32)
            actions["rh_quat_x"] = torch.tensor(quat[1], dtype=torch.float32)
            actions["rh_quat_y"] = torch.tensor(quat[2], dtype=torch.float32)
            actions["rh_quat_z"] = torch.tensor(quat[3], dtype=torch.float32)
            actions["rh_gripper"] = torch.tensor(self.rh_grip_state, dtype=torch.float32)

        if self.config.control_arms in ["left", "both"]:
            mid = self.mocap_ids["left"]
            pos, quat = self._world_to_root(self.data.mocap_pos[mid], self.data.mocap_quat[mid])
            actions["lh_pos_x"] = torch.tensor(pos[0], dtype=torch.float32)
            actions["lh_pos_y"] = torch.tensor(pos[1], dtype=torch.float32)
            actions["lh_pos_z"] = torch.tensor(pos[2], dtype=torch.float32)
            actions["lh_quat_w"] = torch.tensor(quat[0], dtype=torch.float32)
            actions["lh_quat_x"] = torch.tensor(quat[1], dtype=torch.float32)
            actions["lh_quat_y"] = torch.tensor(quat[2], dtype=torch.float32)
            actions["lh_quat_z"] = torch.tensor(quat[3], dtype=torch.float32)
            actions["lh_gripper"] = torch.tensor(self.lh_grip_state, dtype=torch.float32)

        if "head" in self.mocap_ids:
            mid = self.mocap_ids["head"]
            _, quat_h = self._world_to_root([0, 0, 0], self.data.mocap_quat[mid])
            actions["head_quat_w"] = torch.tensor(quat_h[0], dtype=torch.float32)
            actions["head_quat_x"] = torch.tensor(quat_h[1], dtype=torch.float32)
            actions["head_quat_y"] = torch.tensor(quat_h[2], dtype=torch.float32)
            actions["head_quat_z"] = torch.tensor(quat_h[3], dtype=torch.float32)

        return actions

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        if not self.is_connected:
            return

        if "rh_gripper" in feedback:
            val = feedback["rh_gripper"]
            robot_level = float(val.item() if hasattr(val, "item") else val)
            pending = self._grip_pending["rh"]
            if pending is None:
                self.rh_grip_state = robot_level
            elif abs(robot_level - pending) < 1e-3:
                self._grip_pending["rh"] = None

        if "lh_gripper" in feedback:
            val = feedback["lh_gripper"]
            robot_level = float(val.item() if hasattr(val, "item") else val)
            pending = self._grip_pending["lh"]
            if pending is None:
                self.lh_grip_state = robot_level
            elif abs(robot_level - pending) < 1e-3:
                self._grip_pending["lh"] = None

        update_visuals = False
        self._init_mocaps_from_robot(feedback)

        # Object sync (always track ground-truth object poses from robot simulation)
        if not hasattr(self, "_obj_jnt_adrs"):
            self._obj_jnt_adrs: dict[int, int | None] = {}
            scene_objects = self._scene_objects or [{"body": "blue-cube"}]
            for i, obj_def in enumerate(scene_objects):
                body_name = obj_def.get("body", "")
                try:
                    bid = self.model.body(body_name).id
                    self._obj_jnt_adrs[i] = self.model.jnt_qposadr[self.model.body_jntadr[bid]]
                except Exception:
                    self._obj_jnt_adrs[i] = None

        for idx, jnt_adr in self._obj_jnt_adrs.items():
            if jnt_adr is None:
                continue
            prefix = f"obj{idx}"
            if f"{prefix}_pos_x" in feedback:
                try:
                    pos_w, quat_w = self._root_to_world(
                        [float(feedback[f"{prefix}_pos_{ax}"].item() if hasattr(feedback[f"{prefix}_pos_{ax}"], "item") else feedback[f"{prefix}_pos_{ax}"]) for ax in "xyz"],
                        [float(feedback[f"{prefix}_quat_{ax}"].item() if hasattr(feedback[f"{prefix}_quat_{ax}"], "item") else feedback[f"{prefix}_quat_{ax}"]) for ax in "wxyz"],
                    )
                    self.data.qpos[jnt_adr : jnt_adr + 3] = pos_w
                    self.data.qpos[jnt_adr + 3 : jnt_adr + 7] = quat_w
                    update_visuals = True
                except Exception:
                    pass
            elif "objects" in feedback and isinstance(feedback["objects"], dict):
                scene_objs = self._scene_objects or [{"body": "blue-cube"}]
                bname = scene_objs[idx].get("body", "") if idx < len(scene_objs) else ""
                obj_data = feedback["objects"].get(bname) or feedback["objects"].get(f"obj{idx}")
                if obj_data is not None:
                    if "qpos" in obj_data:
                        self.data.qpos[jnt_adr : jnt_adr + 7] = obj_data["qpos"]
                        update_visuals = True
                    elif "xpos" in obj_data and "quat" in obj_data:
                        self.data.qpos[jnt_adr : jnt_adr + 3] = obj_data["xpos"]
                        self.data.qpos[jnt_adr + 3 : jnt_adr + 7] = obj_data["quat"]
                        update_visuals = True
        self._reset_request = False
        if update_visuals:
            mujoco.mj_forward(self.model, self.data)

        for i, name in enumerate(self.actuator_list):
            if name in feedback:
                q_idx = self.qpos_map[i]
                if q_idx != -1:
                    val = feedback[name]
                    self.data.qpos[q_idx] = float(val.item() if hasattr(val, "item") else val)
                    update_visuals = True

        if update_visuals:
            mujoco.mj_forward(self.model, self.data)

    def _init_vr(self):
        if zmq is None:
            print("[VR] Warning: zmq not available, VR disabled")
            return
        self._vr_sockets = {}
        for name, port in [("head", VR_HEAD_PORT), ("rh", VR_HAND_RIGHT_PORT), ("lh", VR_HAND_LEFT_PORT), ("btn", VR_BUTTONS_PORT)]:
            ctx = zmq.Context()
            sock = ctx.socket(zmq.PULL)
            endpoint = f"tcp://*:{port}"
            sock.bind(endpoint)
            sock.setsockopt(zmq.LINGER, 0)
            self._vr_sockets[name] = (ctx, sock)

        self._vr_running = True
        self._vr_thread = threading.Thread(target=self._vr_receive_loop, daemon=True)
        self._vr_thread.start()
        print("[VR] VR receiver thread started")

    def _vr_receive_loop(self):
        while self._vr_running:
            for name in ("rh", "lh", "btn"):
                if name not in self._vr_sockets:
                    continue
                _, sock = self._vr_sockets[name]
                try:
                    while True:
                        raw = sock.recv_string(flags=zmq.NOBLOCK)
                        if name == "rh":
                            self.latest_rh_data = self._parse_controller(raw)
                        elif name == "lh":
                            self.latest_lh_data = self._parse_controller(raw)
                        elif name == "btn":
                            parsed = self._parse_buttons(raw)
                            if parsed:
                                self._ingest_buttons(parsed)
                            self._last_btn_recv_time = time.time()
                except Exception:
                    pass
            time.sleep(0.001)

    @staticmethod
    def _button_levels(b: dict) -> dict[str, bool]:
        return {
            "freeze": b.get("INDEX_RIGHT", 0) > 0.8 or b.get("INDEX_LEFT", 0) > 0.8,
            "reset": b.get("BTN_TWO_LEFT", 0) > 0.5 or b.get("Y", 0) > 0.8,
            "rec_a": b.get("BTN_ONE_RIGHT", 0) > 0.5 or b.get("A", 0) > 0.8,
            "rec_b": b.get("BTN_TWO_RIGHT", 0) > 0.5 or b.get("B", 0) > 0.8,
        }

    def _ingest_buttons(self, parsed: dict) -> None:
        now = time.monotonic()
        with self._btn_lock:
            self._btn_state.update(parsed)
            for name, pressed in self._button_levels(self._btn_state).items():
                was = self._btn_pressed.get(name, False)
                if pressed != was:
                    self._btn_events.append((name, "press" if pressed else "release", now))
                    self._btn_pressed[name] = pressed
            self.latest_buttons = dict(self._btn_state)

    @staticmethod
    def _parse_controller(data_str):
        try:
            data_str = data_str.strip()
            if not data_str or data_str.startswith("DIAGNOSTIC"):
                return None
            if data_str.startswith("{"):
                j = json.loads(data_str)
                if all(k in j for k in ("x", "y", "z")):
                    pos = np.array([float(j["x"]), float(j["y"]), float(j["z"])])
                elif all(k in j for k in ("X", "Y", "Z")):
                    pos = np.array([float(j["X"]), float(j["Y"]), float(j["Z"])])
                elif all(k in j for k in ("px", "py", "pz")):
                    pos = np.array([float(j["px"]), float(j["py"]), float(j["pz"])])
                else:
                    return None
                if "qw" in j:
                    quat = PyQuaternion(w=float(j["qw"]), x=float(j["qx"]), y=float(j["qy"]), z=float(j["qz"]))
                    return {"pos": pos, "quat": quat}
                if "w" in j and "x" in j and "y" in j and "z" in j:
                    quat = PyQuaternion(w=float(j["w"]), x=float(j["x"]), y=float(j["y"]), z=float(j["z"]))
                    return {"pos": pos, "quat": quat}
                return {"pos": pos}
            else:
                parts = data_str.split(",")
                if len(parts) >= 3:
                    pos = np.array([float(p) for p in parts[:3]])
                    if len(parts) >= 7:
                        quat = PyQuaternion(w=float(parts[6]), x=float(parts[3]), y=float(parts[4]), z=float(parts[5]))
                        return {"pos": pos, "quat": quat}
                    return {"pos": pos}
        except Exception:
            pass
        return None

    @staticmethod
    def _parse_buttons(data_str):
        try:
            data_str = data_str.strip()
            if not data_str or data_str.startswith("DIAGNOSTIC"):
                return None
            if data_str.startswith("{"):
                return json.loads(data_str)
            if ":" in data_str:
                key, val_str = data_str.split(":", 1)
                try:
                    val = float(val_str.strip())
                except ValueError:
                    val = 1.0 if "PRESS" in val_str.upper() else 0.0
                return {key.strip().upper(): val}
        except Exception:
            pass
        return None

    def _vr_hand_to_target(
        self,
        hand_data,
        calib_origin,
        calib_quat,
        home_pos,
        home_rot,
        side="right",
    ):
        if hand_data is None or PyQuaternion is None:
            return None, None, calib_origin, calib_quat
        vr_pos = hand_data["pos"]
        if np.linalg.norm(vr_pos) < 0.001:
            return None, None, calib_origin, calib_quat
        if calib_origin is None:
            calib_origin = vr_pos.copy()

        # Delta position: VR (+X right, +Y up, +Z fwd in Unity/OpenXR) -> Robot (+X fwd, +Y left, +Z up)
        delta_vr = vr_pos - calib_origin
        delta_robot = np.array([delta_vr[2], -delta_vr[0], delta_vr[1]])
        raw_target_pos = home_pos + delta_robot

        if self._prev_target_pos[side] is None:
            self._prev_target_pos[side] = raw_target_pos

        filtered_pos = (
            self._target_filter_alpha * raw_target_pos
            + (1.0 - self._target_filter_alpha) * self._prev_target_pos[side]
        )
        self._prev_target_pos[side] = filtered_pos
        target_pos = filtered_pos

        target_quat = None
        if "quat" in hand_data:
            q_vr = hand_data["quat"]
            q_world = unity_to_world_quat(q_vr)
            if calib_quat is None:
                calib_quat = PyQuaternion(q_world)
            # Relative rotation in world frame from neutral calibration pose
            q_rel = q_world * calib_quat.inverse
            delta_R_world = q_rel.rotation_matrix
            # Target orientation relative to robot home pose
            R_target = delta_R_world @ home_rot
            target_quat = _rotmat_to_quat_wxyz(R_target)
        else:
            target_quat = _rotmat_to_quat_wxyz(home_rot)

        return target_pos, target_quat, calib_origin, calib_quat

    def _process_vr_hands(self):
        if not self.freeze_rh and self.latest_rh_data is not None and "right" in self.mocap_ids:
            home_pos_rh = self._vr_home.get("right", np.array([0.35, -0.15, 0.91]))
            home_rot_rh = self._vr_home_rot.get("right", np.eye(3))
            pos, quat, self.calibrated_origin_rh, self.calibrated_quat_rh = self._vr_hand_to_target(
                self.latest_rh_data,
                self.calibrated_origin_rh,
                self.calibrated_quat_rh,
                home_pos_rh,
                home_rot_rh,
                side="right",
            )
            if pos is not None:
                pos[0] = np.clip(pos[0], 0.1, 0.8)
                pos[1] = np.clip(pos[1], -0.4, 0.1)
                pos[2] = np.clip(pos[2], 0.4, 1.1)
                mid = self.mocap_ids["right"]
                self.data.mocap_pos[mid] = pos
                if quat is not None:
                    self.data.mocap_quat[mid] = quat

        if not self.freeze_lh and self.latest_lh_data is not None and "left" in self.mocap_ids:
            home_pos_lh = self._vr_home.get("left", np.array([0.35, 0.15, 0.91]))
            home_rot_lh = self._vr_home_rot.get("left", np.eye(3))
            pos, quat, self.calibrated_origin_lh, self.calibrated_quat_lh = self._vr_hand_to_target(
                self.latest_lh_data,
                self.calibrated_origin_lh,
                self.calibrated_quat_lh,
                home_pos_lh,
                home_rot_lh,
                side="left",
            )
            if pos is not None:
                pos[0] = np.clip(pos[0], 0.1, 0.8)
                pos[1] = np.clip(pos[1], -0.1, 0.4)
                pos[2] = np.clip(pos[2], 0.4, 1.1)
                mid = self.mocap_ids["left"]
                self.data.mocap_pos[mid] = pos
                if quat is not None:
                    self.data.mocap_quat[mid] = quat

    def _process_vr_head(self):
        if self.freeze_head or PyQuaternion is None:
            return
        if "head" not in self._vr_sockets or "head" not in self.mocap_ids:
            return
        _, sock = self._vr_sockets["head"]
        last_data = None
        try:
            while True:
                try:
                    last_data = sock.recv_string(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
        except Exception:
            return

        if last_data is None:
            return
        data = last_data.strip()
        if not data or data.startswith("DIAGNOSTIC"):
            return

        q_vr = None
        try:
            if data.startswith("{"):
                j = json.loads(data)
                if "w" in j:
                    q_vr = PyQuaternion(w=float(j["w"]), x=float(j["x"]), y=float(j["y"]), z=float(j["z"]))
            else:
                parts = data.split(",")
                if len(parts) == 4:
                    q_vr = PyQuaternion(w=float(parts[3]), x=float(parts[0]), y=float(parts[1]), z=float(parts[2]))
        except Exception:
            return

        if q_vr is not None:
            q_world = unity_to_world_quat(q_vr)
            if self.calibrated_quat_head is None:
                self.calibrated_quat_head = PyQuaternion(q_world)
            q_rel = q_world * self.calibrated_quat_head.inverse
            delta_R_world = q_rel.rotation_matrix

            home_rot_head = self._vr_home_rot.get("head")
            if home_rot_head is not None:
                R_head_world = delta_R_world @ home_rot_head
            else:
                R_head_zero = np.array([
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ])
                R_head_world = delta_R_world @ R_head_zero

            self.data.mocap_quat[self.mocap_ids["head"]] = _rotmat_to_quat_wxyz(R_head_world)

    def _process_vr_button_commands(self):
        with self._btn_lock:
            events = list(self._btn_events)
            self._btn_events.clear()
            levels = dict(self._btn_pressed)
            state = dict(self._btn_state)

        for side, key in (("rh", "HAND_RIGHT"), ("lh", "HAND_LEFT")):
            if key in state:
                level = self._trigger_to_level(state[key])
                if abs(level - getattr(self, f"{side}_grip_state")) > 1e-3:
                    self._set_grip(side, level)

        for name, kind, t in events:
            if kind == "press":
                if name == "reset":
                    self._pending_visual_reset = True
                elif name == "rec_a":
                    self._vr_rec_start_event = True
                elif name == "rec_b":
                    self._vr_b_hold_start = t
            elif kind == "release" and name == "rec_b":
                held_s = t - (self._vr_b_hold_start or t)
                if held_s >= self._B_HOLD_DISCARD_S:
                    self._vr_rec_discard_event = True
                    print(f"[VR] B held {held_s:.1f}s → DISCARD episode")
                else:
                    self._vr_rec_stop_event = True
                self._vr_b_hold_start = None

        self.freeze_rh = levels.get("freeze", False)
        self.freeze_lh = levels.get("freeze", False)
        self.freeze_head = levels.get("freeze", False)

    def _update_vr_record_events(self) -> None:
        if self._vr_sockets and (time.time() - self._last_btn_recv_time) > 0.5:
            self.freeze_rh = False
            self.freeze_lh = False
            self.freeze_head = False

    def consume_vr_record_events(self) -> tuple[bool, bool, bool]:
        start = self._vr_rec_start_event
        stop = self._vr_rec_stop_event
        discard = self._vr_rec_discard_event
        self._vr_rec_start_event = False
        self._vr_rec_stop_event = False
        self._vr_rec_discard_event = False
        return start, stop, discard

    def _resolve_qpos_index(self, name: str) -> int:
        try:
            return int(self.model.joint(name).qposadr[0])
        except Exception:
            pass
        try:
            act_id = self.model.actuator(name).id
            trn_type = int(self.model.actuator_trntype[act_id])
            if trn_type == 0:
                joint_id = int(self.model.actuator_trnid[act_id, 0])
                return int(self.model.jnt_qposadr[joint_id])
        except Exception:
            pass
        return -1

    def _reset_mocaps_to_xml_pose(self):
        body_by_side = {
            "right": "r_hand_target",
            "left": "l_hand_target",
            "head": "head_target",
        }
        for side, body_name in body_by_side.items():
            if side not in self.mocap_ids:
                continue
            try:
                mid = self.mocap_ids[side]
                bid = self.model.body(body_name).id
                self.data.mocap_pos[mid] = self.model.body_pos[bid]
                self.data.mocap_quat[mid] = self.model.body_quat[bid]
            except Exception:
                continue

        if not getattr(self.config, "strict_xml_init", False):
            if "right" in self.mocap_ids and "right" in self.site_ids:
                mid = self.mocap_ids["right"]
                sid = self.site_ids["right"]
                mujoco.mju_mat2Quat(self.data.mocap_quat[mid], self.data.site_xmat[sid])

            if "left" in self.mocap_ids and "left" in self.site_ids:
                mid = self.mocap_ids["left"]
                sid = self.site_ids["left"]
                mujoco.mju_mat2Quat(self.data.mocap_quat[mid], self.data.site_xmat[sid])

        mujoco.mj_forward(self.model, self.data)

    def _reset_scenario(self):
        objects = self._scene_objects
        if not objects and not self._scene_joints:
            spawn = getattr(self.config, "cube_spawn", None)
            if spawn:
                objects = [{"body": "blue-cube", "spawn": spawn}]
            else:
                return

        def _sample(val, default=0.0):
            if isinstance(val, (list, tuple)) and len(val) == 2:
                return np.random.uniform(val[0], val[1])
            return float(val) if val is not None else default

        reset_poses = {}
        any_reset = False

        for obj_def in objects:
            body_name = obj_def.get("body", "")
            spawn = obj_def.get("spawn", {})
            try:
                body = self.model.body(body_name)
                bid = body.id
                jnt_adr = body.jntadr[0]
                if jnt_adr == -1:
                    continue
                qpos_adr = self.model.jnt_qposadr[jnt_adr]
                dof_adr = self.model.jnt_dofadr[jnt_adr]

                if any(k in spawn for k in ("x", "y", "z")):
                    x = _sample(spawn.get("x"), self.data.qpos[qpos_adr + 0])
                    y = _sample(spawn.get("y"), self.data.qpos[qpos_adr + 1])
                    z = _sample(spawn.get("z"), self.data.qpos[qpos_adr + 2])
                    self.data.qpos[qpos_adr : qpos_adr + 3] = [x, y, z]

                quat = spawn.get("quat", [1, 0, 0, 0])
                self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat

                ndof = self.model.jnt_type[self.model.body_jntadr[bid]]
                num_dof = 6 if ndof == 0 else 1
                self.data.qvel[dof_adr : dof_adr + num_dof] = 0

                pos_arr = self.data.qpos[qpos_adr : qpos_adr + 3].copy()
                quat_arr = self.data.qpos[qpos_adr + 3 : qpos_adr + 7].copy()
                reset_poses[body_name] = (pos_arr, quat_arr)
                any_reset = True
                print(f"[Teleop] Reset {body_name} → ({pos_arr[0]:.2f}, {pos_arr[1]:.2f}, {pos_arr[2]:.2f})")
            except Exception as e:
                print(f"[Teleop] Reset skip '{body_name}': {e}")

        if any_reset:
            self._last_reset_poses = reset_poses
            self._reset_request = True
            mujoco.mj_forward(self.model, self.data)

        for jnt_def in self._scene_joints:
            jnt_name = jnt_def.get("name", "")
            target_qpos = float(jnt_def.get("qpos", 0.0))
            try:
                jnt_id = self.model.joint(jnt_name).id
                qpos_adr = self.model.jnt_qposadr[jnt_id]
                dof_adr = self.model.jnt_dofadr[jnt_id]
                self.data.qpos[qpos_adr] = target_qpos
                self.data.qvel[dof_adr] = 0.0
                print(f"[Teleop] Reset joint '{jnt_name}' → {target_qpos:.3f} rad")
            except Exception as e:
                print(f"[Teleop] Reset joint skip '{jnt_name}': {e}")

        if self._scene_joints:
            mujoco.mj_forward(self.model, self.data)

    def _set_grip(self, side: str, level: float) -> None:
        level = float(np.clip(level, 0.0, 1.0))
        setattr(self, f"{side}_grip_state", level)
        self._grip_pending[side] = level

    def _trigger_to_level(self, value) -> float:
        lo, hi = self.config.vr_grip_deadband
        return float(np.clip((float(value) - lo) / max(hi - lo, 1e-6), 0.0, 1.0))

    def consume_teleop_toggle_event(self) -> bool:
        if getattr(self, "_teleop_toggle_requested", False):
            self._teleop_toggle_requested = False
            return True
        return False

    def consume_reset_home_event(self) -> bool:
        if getattr(self, "_reset_home_requested", False):
            self._reset_home_requested = False
            return True
        return False

    def _key_callback(self, keycode):
        if keycode == glfw.KEY_8:
            self._set_grip("rh", 0.0)
            print("Right Gripper: OPEN")
        elif keycode == glfw.KEY_9:
            self._set_grip("rh", 1.0)
            print("Right Gripper: CLOSE")
        elif keycode == glfw.KEY_N:
            self._pending_visual_reset = True
        elif keycode == glfw.KEY_6:
            self._set_grip("lh", 0.0)
            print("Left Gripper: OPEN")
        elif keycode == glfw.KEY_7:
            self._set_grip("lh", 1.0)
            print("Left Gripper: CLOSE")
        elif keycode == glfw.KEY_T:
            self._teleop_toggle_requested = True
            print("[Teleop] Toggle teleoperation requested via viewer 'T' key.")
        elif keycode == glfw.KEY_SPACE:
            self._reset_mocaps_to_xml_pose()
            self._reset_home_requested = True
            print("[Teleop] SPACE pressed: Mocaps reset to XML home. Requesting smooth robot homing...")
        elif keycode == glfw.KEY_C:
            self.calibrated_origin_rh = None
            self.calibrated_origin_lh = None
            self.calibrated_quat_rh = None
            self.calibrated_quat_lh = None
            self.calibrated_quat_head = None
            self._prev_target_pos = {"right": None, "left": None}
            print("[Teleop] VR recalibration requested via viewer 'C' key.")

