#!/usr/bin/env python3
"""
MuJoCo Whole-Body Teleoperation & Simulation Core for iCub
==========================================================
Powered exclusively by Bipedal Locomotion Framework (BLF) Whole-Body Controllers:
  - iCubWholeBodyIK (TSIK): Uses `icub_visuomanip_ikin_limits.xml` (position actuators)
  - iCubWholeBodyID (TSID / Cartesian Impedance): Uses `icub_visuomanip_idyn_limits.xml`
    (torque `<motor>` actuators on the 20 upper-body joints + `<position>` actuators on the 18 hand joints)

End-effector & Head Tasks:
  - Right Hand (`r_hand`): 6D SE(3) pose tracked via `r_hand_target` mocap body
  - Left Hand (`l_hand`):  6D SE(3) pose tracked via `l_hand_target` mocap body
  - Head (`head`):         3D SO(3) orientation tracked via `head_target` mocap body quaternion
"""

import os
import sys
import time
import argparse
import threading
import json
import weakref
import tempfile
import faulthandler
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import yaml
import mujoco
import mujoco.viewer
import glfw

faulthandler.enable()

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from controllers.whole_body_ik import (
    iCubWholeBodyIK,
    rotmat_to_quat_wxyz,
    quat_wxyz_to_rotmat,
)
from controllers.whole_body_id import iCubWholeBodyID
from controllers.joint_limits import load_joint_limits

import cv2
import zmq
from pyquaternion import Quaternion as PyQuaternion
from interfaces.zmq_image_publisher import ZMQCompressedImageTransmitter
from interfaces.vr_overlay import VR_PAUSE_TIMEOUT_S, draw_help_hint, draw_help_panel

# Hand actuator names (order matches the actuator block in icub_visuomanip_*_limits.xml)
RIGHT_HAND_ACTUATORS = [
    "r_hand_finger", "r_thumb_oppose", "r_thumb_proximal", "r_thumb_distal",
    "r_index_proximal", "r_index_distal",
    "r_middle_proximal", "r_middle_distal", "r_pinky",
]
LEFT_HAND_ACTUATORS = [
    "l_hand_finger", "l_thumb_oppose", "l_thumb_proximal", "l_thumb_distal",
    "l_index_proximal", "l_index_distal",
    "l_middle_proximal", "l_middle_distal", "l_pinky",
]

HAND_OPEN = np.array([0.000, 1.309, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000])
HAND_CLOSE = np.array([0.000, 1.047, 0.785, 1.571, 1.396, 1.745, 1.396, 1.745, 0.000])

VR_HEAD_PORT = 8115
VR_HAND_RIGHT_PORT = 8117
VR_BUTTONS_PORT = 8119
VR_HAND_LEFT_PORT = 8121

# Convert Unity left-handed (RUF: +X right, +Y up, +Z forward) quaternion
# to MuJoCo / ROS FLU right-handed (+X forward, +Y left, +Z up) quaternion:
# Unity: (x, y, z, w) -> ROS FLU: (z, -x, y, -w)
def unity_to_world_quat(q_unity: PyQuaternion) -> PyQuaternion:
    return PyQuaternion(w=-q_unity.w, x=q_unity.z, y=-q_unity.x, z=q_unity.y)


def draw_status_corner(img: np.ndarray, lines) -> np.ndarray:
    """Draw recording status overlay in the top-right corner of BGR image `img`."""
    lines = [ln.strip() for ln in lines[:2] if ln.strip()]
    if not lines:
        return img
    h, w = img.shape[:2]
    k = w / 640.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    thick = max(1, round(2 * k))
    pad, margin = max(4, int(10 * k)), max(4, int(14 * k))
    recording = lines[0].upper().startswith("GRABANDO") or lines[0].upper().startswith("RECORDING")
    colors = [(0, 0, 255) if recording else (0, 220, 255)] + [(255, 255, 255)]
    scales = [0.7 * k, 0.5 * k]
    sizes = [cv2.getTextSize(t, font, sc, thick)[0] for t, sc in zip(lines, scales)]
    r = sizes[0][1] // 2 + 1
    dot_w = 2 * r + pad if recording else 0
    box_w = max([sizes[0][0] + dot_w] + [tw for tw, _ in sizes[1:]]) + 2 * pad
    box_h = sum(th for _, th in sizes) + pad * (len(lines) + 1)
    x1, y0 = w - margin, margin
    x0, y1 = max(0, x1 - box_w), min(h, y0 + box_h)
    img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * 0.35).astype(img.dtype)
    y = y0
    for i, (text, sc, (tw, th), color) in enumerate(zip(lines, scales, sizes, colors)):
        y += pad + th
        cv2.putText(img, text, (x1 - pad - tw, y), font, sc, color, thick, cv2.LINE_AA)
        if i == 0 and recording:
            cv2.circle(img, (x1 - pad - tw - pad // 2 - r, y - th // 2), r, (0, 0, 255), -1, cv2.LINE_AA)
    return img


_LIVE_CORES: "weakref.WeakSet[MuJoCoTeleop]" = weakref.WeakSet()


def close_all_live() -> int:
    """Close all live MuJoCoTeleop instances in this process."""
    cores = list(_LIVE_CORES)
    for core in cores:
        core.close()
    return len(cores)


def load_mujoco_model_for_controller(model_path: str, controller_type: str) -> mujoco.MjModel:
    """
    Load the MuJoCo XML model corresponding to `controller_type`:
      - 'ik': uses `icub_visuomanip_ikin_limits.xml` (position actuators on all joints)
      - 'id': uses `icub_visuomanip_idyn_limits.xml` (torque motors on 20 upper-body joints + position on fingers)
    """
    p = Path(model_path).resolve()
    ctrl_type = str(controller_type).lower()
    if ctrl_type == "id":
        if p.name == "icub_visuomanip_ikin_limits.xml":
            idyn_path = p.with_name("icub_visuomanip_idyn_limits.xml")
            if idyn_path.exists():
                return mujoco.MjModel.from_xml_path(str(idyn_path))
        xml_text = p.read_text(encoding="utf-8")
        if "icub_visuomanip_ikin_limits.xml" in xml_text:
            xml_idyn = xml_text.replace(
                "icub_visuomanip_ikin_limits.xml",
                "icub_visuomanip_idyn_limits.xml",
            )
            tmp_file = p.parent / f".tmp_idyn_{os.getpid()}_{p.name}"
            try:
                tmp_file.write_text(xml_idyn, encoding="utf-8")
                return mujoco.MjModel.from_xml_path(str(tmp_file))
            finally:
                if tmp_file.exists():
                    try:
                        tmp_file.unlink()
                    except Exception:
                        pass
    return mujoco.MjModel.from_xml_path(str(p))


class MuJoCoTeleop:
    """
    Unified MuJoCo Whole-Body Teleoperation & Control Engine for iCub using
    iCubWholeBodyIK (TSIK) or iCubWholeBodyID (TSID / Cartesian Impedance).
    """

    JOINTS_LIST = iCubWholeBodyIK.JOINTS_LIST

    def __init__(
        self,
        model_path: str,
        vr_enabled: bool = False,
        record_enabled: bool = False,
        repo_id: str = "icub_mujoco",
        control_arms: Optional[str] = None,
        primary_arm: str = "right",
        config_path: Optional[str] = None,
        vr_ip: Optional[str] = None,
        scene_objects: Optional[ list] = None,
        controller_type: Optional[str] = None,
        impedance_mode: Optional[bool] = None,
        control_dt: Optional[float] = None,
    ):
        _LIVE_CORES.add(self)
        cfg = {}
        if config_path is None:
            default_cfg = _PROJECT_ROOT / "config" / "control_config.yaml"
            if default_cfg.exists():
                config_path = str(default_cfg)
        if config_path is not None and Path(config_path).exists():
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            print(f"  [Config] Loaded {config_path}")

        self.sim_dt = float(cfg.get("sim_dt", 0.005))
        if control_dt is not None:
            self.control_dt = float(control_dt)
        else:
            self.control_dt = float(cfg.get("period", cfg.get("control_dt", 0.02)))
        self.frame_skip = max(1, int(round(self.control_dt / self.sim_dt)))

        self.controller_type = (
            str(controller_type).lower()
            if controller_type is not None
            else str(cfg.get("controller_type", "ik")).lower()
        )
        if self.controller_type not in ("ik", "id"):
            raise ValueError(f"Unsupported controller_type='{self.controller_type}'. Expected 'ik' or 'id'.")

        self.impedance_mode = (
            bool(impedance_mode)
            if impedance_mode is not None
            else bool(cfg.get("impedance_mode", False))
        )

        # Load appropriate MuJoCo model (ikin_limits for IK, idyn_limits for ID)
        self.model_path = str(model_path)
        self.model = load_mujoco_model_for_controller(self.model_path, self.controller_type)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.sim_dt

        # Joint limits: joint_limits_deg of the config is the only source (controllers/joint_limits.py).
        # They overwrite the XML ranges and the position actuators' ctrlrange (IK model; the ID
        # model drives these joints with torque motors).
        self._joint_limits = load_joint_limits(cfg)
        for j_name, (lo, hi) in self._joint_limits.items():
            jid = self.model.joint(j_name).id
            self.model.jnt_range[jid] = (lo, hi)
            self.model.jnt_limited[jid] = 1
            act = self.model.actuator(j_name)
            if self.model.actuator_gaintype[act.id] == mujoco.mjtGain.mjGAIN_FIXED and self.model.actuator_biastype[act.id] == mujoco.mjtBias.mjBIAS_AFFINE:
                self.model.actuator_ctrlrange[act.id] = (lo, hi)

        # Map the 20 upper-body joints (JOINTS_LIST) to MuJoCo qpos, qvel (dof), and actuator indices
        self._upper_jnt_ids = np.array([self.model.joint(n).id for n in self.JOINTS_LIST], dtype=np.int32)
        self._upper_qpos_idxs = np.array([self.model.jnt_qposadr[jid] for jid in self._upper_jnt_ids], dtype=np.int32)
        self._upper_qvel_idxs = np.array([self.model.jnt_dofadr[jid] for jid in self._upper_jnt_ids], dtype=np.int32)
        self._upper_act_ids = np.array([self.model.actuator(n).id for n in self.JOINTS_LIST], dtype=np.int32)
        self._upper_jnt_min = np.array([self.model.jnt_range[jid, 0] for jid in self._upper_jnt_ids], dtype=np.float64)
        self._upper_jnt_max = np.array([self.model.jnt_range[jid, 1] for jid in self._upper_jnt_ids], dtype=np.float64)


        # Home pose overrides from config
        self._home_pose_overrides: Dict[str, float] = {}
        for joint_name, deg_value in (cfg.get("home_pose", {}) or {}).items():
            try:
                self._home_pose_overrides[joint_name] = float(np.deg2rad(float(deg_value)))
            except Exception:
                continue

        self.home_qpos = self.model.qpos0.copy()
        self.home_ctrl = np.zeros(self.model.nu, dtype=np.float64)
        self._apply_home_pose_to_arrays(self.home_qpos, self.home_ctrl)

        # Initialize simulation state at home pose
        self.data.qpos[:] = self.home_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.home_ctrl
        mujoco.mj_forward(self.model, self.data)

        # Root body ('icub') world transform for BLF solver
        self.waist_id = self.model.body("icub").id
        waist_pos = self.data.xpos[self.waist_id].copy()
        waist_mat = self.data.xmat[self.waist_id].reshape(3, 3).copy()

        # Instantiate Whole-Body Controller (iCubWholeBodyIK or iCubWholeBodyID)
        robot_model_name = str(cfg.get("robot_model", "iCubGazeboV2_5_visuomanip"))
        if self.controller_type == "id":
            tsid_cfg = cfg.get("tsid", {}) or {}
            self.solver = iCubWholeBodyID(
                sampling_time=self.sim_dt,
                robot_model_name=robot_model_name,
                kp_cartesian=float(tsid_cfg.get("kp_cartesian", 100.0)),
                kp_angular=float(tsid_cfg.get("kp_angular", 80.0)),
                kp_posture=float(tsid_cfg.get("kp_posture", 25.0)),
                weight_hands=float(tsid_cfg.get("weight_hands", 100.0)),
                weight_head=float(tsid_cfg.get("weight_head", 15.0)),
                weight_posture_torso=float(tsid_cfg.get("weight_posture_torso", 2.0)),
                weight_posture_neck=float(tsid_cfg.get("weight_posture_neck", 0.05)),
                weight_posture_arms=float(tsid_cfg.get("weight_posture_arms", 0.1)),
                weight_torque_reg=float(tsid_cfg.get("weight_torque_reg", 1e-5)),
                impedance_mode=self.impedance_mode,
                compensate_urdf_damping=True,
                joint_limits=self._joint_limits,
                joint_limit_horizon_s=float(tsid_cfg.get("joint_limit_horizon_s", 0.1)),
            )
        else:
            tsik_cfg = cfg.get("tsik", {}) or {}
            self.solver = iCubWholeBodyIK(
                sampling_time=self.control_dt,
                robot_model_name=robot_model_name,
                kp_cartesian=float(tsik_cfg.get("kp_cartesian", 20.0)),
                kp_angular=float(tsik_cfg.get("kp_angular", 15.0)),
                weight_hands=float(tsik_cfg.get("weight_hands", 180.0)),
                weight_head=float(tsik_cfg.get("weight_head", 5.0)),
                weight_posture_torso=float(tsik_cfg.get("weight_posture_torso", 80.0)),
                weight_posture_neck=float(tsik_cfg.get("weight_posture_neck", 0.05)),
                weight_posture_arms=float(tsik_cfg.get("weight_posture_arms", 0.05)),
                max_joint_velocity=float(tsik_cfg.get("max_joint_velocity", 1.5)),
                joint_limits=self._joint_limits,
            )

        self.solver.set_base_pose(waist_pos, waist_mat)
        self._q_home_20 = self.home_qpos[self._upper_qpos_idxs].copy()
        self.solver.set_posture_reference(self._q_home_20)
        self._q_cmd_20 = self._q_home_20.copy()

        # Sync solver state with initial MuJoCo state
        self._update_solver_from_mj()

        # In TSID mode, initialize home_ctrl with gravity compensation torques
        if self.controller_type == "id":
            tau_grav = self.solver.get_gravity_compensation_torques()
            self.home_ctrl[self._upper_act_ids] = tau_grav
            self.data.ctrl[self._upper_act_ids] = tau_grav

        # End-effector & head bodies (r_hand_dh_frame, l_hand_dh_frame, head)
        try:
            self.rh_body_id = self.model.body("r_hand_dh_frame").id
        except KeyError:
            self.rh_body_id = self.model.body("r_hand").id
        try:
            self.lh_body_id = self.model.body("l_hand_dh_frame").id
        except KeyError:
            self.lh_body_id = self.model.body("l_hand").id
        self.head_body_id = self.model.body("head").id
        self.head_body_name = "head"

        # Sites (r_hand_dh_frame_site, l_hand_dh_frame_site, head_site)
        try:
            self.rh_site_id = self.model.site("r_hand_dh_frame_site").id
        except KeyError:
            self.rh_site_id = self.model.site("r_hand_site").id
        try:
            self.lh_site_id = self.model.site("l_hand_dh_frame_site").id
        except KeyError:
            self.lh_site_id = self.model.site("l_hand_site").id
        try:
            self.head_site_id = self.model.site("head_site").id
        except KeyError:
            self.head_site_id = None

        # Mocap bodies (r_hand_target, l_hand_target, head_target)
        self.rh_mocap_id = self.model.body("r_hand_target").mocapid[0]
        self.lh_mocap_id = self.model.body("l_hand_target").mocapid[0]
        try:
            self.head_mocap_id = self.model.body("head_target").mocapid[0]
        except KeyError:
            self.head_mocap_id = self.model.body("com_target").mocapid[0]
        self.gaze_mocap_id = self.head_mocap_id

        # Arm & head control flags
        self.control_arms = control_arms or cfg.get("control_arms", "both")
        self.primary_arm = primary_arm if primary_arm != "right" else cfg.get("primary_arm", primary_arm)
        if self.primary_arm in ("right_arm", "r", "rh"):
            self.primary_arm = "right"
        elif self.primary_arm in ("left_arm", "l", "lh"):
            self.primary_arm = "left"
        self.use_head_control = bool(cfg.get("use_head_control", cfg.get("gaze_ctrl", True)))

        # Hand finger actuator IDs
        self.rh_hand_act_ids = np.array([self.model.actuator(n).id for n in RIGHT_HAND_ACTUATORS])
        self.lh_hand_act_ids = np.array([self.model.actuator(n).id for n in LEFT_HAND_ACTUATORS])

        # Scene objects & joints
        self._cube_spawn = cfg.get("cube_spawn", {"x": [0.35, 0.45], "y": [-0.15, -0.05], "z": 0.745})
        self._scene_objects: list = scene_objects or []
        self._scene_joints: list = []

        # Reference home poses for relative VR tracking
        self._home_pos_rh: Optional[np.ndarray] = None
        self._home_rot_rh: Optional[np.ndarray] = None
        self._home_pos_lh: Optional[np.ndarray] = None
        self._home_rot_lh: Optional[np.ndarray] = None
        self._home_rot_head: Optional[np.ndarray] = None

        # VR calibration baselines & target filters
        self.calibrated_origin_rh: Optional[np.ndarray] = None
        self.calibrated_origin_lh: Optional[np.ndarray] = None
        self.calibrated_quat_rh: Optional[PyQuaternion] = None
        self.calibrated_quat_lh: Optional[PyQuaternion] = None
        self.calibrated_quat_head: Optional[PyQuaternion] = None
        self._target_filter_alpha = 0.25
        self._prev_target_pos = {"right": None, "left": None}

        # Snap mocap targets to initial r_hand, l_hand, and head poses
        self._snap_targets_to_ee()

        # State machine
        self.state = "idle"
        self.running = True
        self._pending_cmds = []
        self._pending_cmds_lock = threading.Lock()
        self.cli_controls_enabled = sys.stdin.isatty()
        self._cli_thread = None

        # Hand closure state
        self._hand_close_time_s = float(cfg.get("hand_close_time_s", 3.8))
        self._vr_grip_deadband = tuple(cfg.get("vr_grip_deadband", (0.1, 0.9)))

        def _hand_rad(key, default):
            vals = cfg.get(key)
            arr = np.radians(np.asarray(vals, dtype=float).ravel()) if vals is not None else None
            return arr if arr is not None and arr.size == default.size else default

        self._hand_open = _hand_rad("hand_open_values", HAND_OPEN)
        self._hand_close_full = _hand_rad("hand_close_values", HAND_CLOSE)
        self._hand_close_ratio = float(np.clip(cfg.get("hand_close_ratio", 0.5), 0.0, 1.0))
        self._hand_close_goal = self._hand_open + self._hand_close_ratio * (self._hand_close_full - self._hand_open)
        self._hand_goal = {"right": self._hand_open.copy(), "left": self._hand_open.copy()}
        self._hand_last_t = None
        self.rh_grip_state = 0.0
        self.lh_grip_state = 0.0

        # Pre-initialize hand actuator controls and thumb oppose joints to open state immediately
        for aid, val in zip(self.rh_hand_act_ids, self._hand_open):
            self.home_ctrl[aid] = val
            self.data.ctrl[aid] = val
        for aid, val in zip(self.lh_hand_act_ids, self._hand_open):
            self.home_ctrl[aid] = val
            self.data.ctrl[aid] = val

        for side, j_name, val in (("r", "r_hand_thumb_0_joint", self._hand_open[1]), ("l", "l_hand_thumb_0_joint", self._hand_open[1])):
            try:
                jid = self.model.joint(j_name).id
                adr = self.model.jnt_qposadr[jid]
                self.home_qpos[adr] = val
                self.data.qpos[adr] = val
            except Exception:
                pass
        mujoco.mj_forward(self.model, self.data)

        # Go-home trajectory state
        self._home_start_q = None
        self._home_target_q = None
        self._home_step = 0
        self._home_total_steps = int(cfg.get("go_home_steps", 100))

        # Image streaming to VR (head_cam -> 10505, viewer -> 15001)
        self.img_stream_enabled = True
        self._head_cam_pub = None
        self._viewer_pub = None
        self._head_renderer = None
        self._viewer_renderer = None
        if self.img_stream_enabled:
            try:
                stream_quality = int(cfg.get("stream_jpeg_quality", 70))
                self._head_cam_pub = ZMQCompressedImageTransmitter(host="*", port=10505, quality=stream_quality)
                self._viewer_pub = ZMQCompressedImageTransmitter(host="*", port=15001, quality=stream_quality)
                self._head_renderer = mujoco.Renderer(self.model, height=480, width=640)
                self._head_scene_option = mujoco.MjvOption()
                if not cfg.get("vr_show_targets", True):
                    self._head_scene_option.geomgroup[2] = 0
                    self._head_scene_option.sitegroup[2] = 0
                # head_cam (real eye FOV, ~50 deg horizontal) stretched in the lenses looks
                # zoomed in: the VR uses head_cam_vr, same pose with fovy vr_head_fovy.
                self._vr_head_cam = "head_cam"
                vr_fovy = cfg.get("vr_head_fovy")
                vr_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam_vr")
                if vr_fovy and vr_cam_id >= 0:
                    self.model.cam_fovy[vr_cam_id] = float(vr_fovy)
                    self._vr_head_cam = "head_cam_vr"
                self._viewer_renderer = mujoco.Renderer(self.model, height=480, width=640)
                # Small VR screen (15001): the fixed front camera (frontview), as in Gazebo
                # (teleop_module_sm.py sends its front camera there), without the targets like
                # the recorded camera. Scenes without it: the viewer's view.
                self._vr_front_cam = "frontview" if mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_CAMERA, "frontview") >= 0 else None
                self._front_scene_option = mujoco.MjvOption()
                self._front_scene_option.geomgroup[2] = 0
                self._front_scene_option.sitegroup[2] = 0
                vr_fovy_txt = f" fovy {self.model.cam_fovy[vr_cam_id]:g} deg" if self._vr_head_cam == "head_cam_vr" else ""
                print(f"  [Stream] {self._vr_head_cam}{vr_fovy_txt} -> port 10505, "
                      f"{self._vr_front_cam or 'viewer'} -> port 15001 (jpeg quality={stream_quality})")
            except Exception as e:
                print(f"  [Stream] Failed to init image streaming: {e}")
                self.img_stream_enabled = False
        self._last_stream_time = 0.0
        self._stream_errors: set[str] = set()
        self._stream_first_ok = False
        self.vr_status_lines: tuple[str, ...] = ()
        # VR help on the big screen (vr_help in the YAML)
        self._vr_help_enabled = bool(cfg.get("vr_help", True))

        # VR setup
        self.vr_ip = vr_ip if vr_ip else cfg.get("vr_ip", None)
        self.vr_enabled = bool(vr_enabled)
        self.latest_rh_data = None
        self.latest_lh_data = None
        self.latest_buttons = None
        self.freeze_rh = False
        self.freeze_lh = False
        self.freeze_head = False
        self._last_btn_recv_time = 0.0
        self._last_ctrl_recv_time = 0.0    # last controller message (0 = never)
        self._vr_help_toggled = False       # help panel opened with X (if BeaVR sends X to the PC)
        self._vr_x_held = False
        # Last value of each lower (grip) trigger: BeaVR may send one key per message
        self._grip_state = {"HAND_RIGHT": 0.0, "HAND_LEFT": 0.0}
        # Clutch (lower trigger): controller position when it was pressed, None = released
        self._clutch_start = {"right": None, "left": None}
        if self.vr_enabled:
            self._init_vr()

        self.record_enabled = bool(record_enabled)
        self.recording_active = False
        self.recorder = None
        self.episode_count = 0

        print("=" * 62)
        print("  MuJoCo iCub Whole-Body Teleop — Initialised")
        print(f"  Model      : {Path(self.model_path).name}")
        print(f"  Controller : BLF {self.controller_type.upper()} (impedance={self.impedance_mode})")
        print(f"  Arms       : {self.control_arms}  |  Head SO(3): {self.use_head_control}")
        print(f"  VR         : {'ON' if self.vr_enabled else 'OFF'}")
        print("=" * 62)

    def _apply_home_pose_to_arrays(self, qpos_array: np.ndarray, ctrl_array: np.ndarray) -> None:
        for joint_name, rad_value in self._home_pose_overrides.items():
            try:
                joint_id = self.model.joint(joint_name).id
                qpos_adr = self.model.jnt_qposadr[joint_id]
                qpos_array[qpos_adr] = rad_value
            except Exception:
                pass
            if self.controller_type == "ik":
                try:
                    actuator_id = self.model.actuator(joint_name).id
                    ctrl_array[actuator_id] = rad_value
                except Exception:
                    pass

    def _update_solver_from_mj(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read the 20 upper-body joint positions and velocities from MjData and update BLF solver."""
        q_20 = self.data.qpos[self._upper_qpos_idxs].copy()
        dq_20 = self.data.qvel[self._upper_qvel_idxs].copy()
        self.solver.update_robot_state(q_20, dq_20)
        return q_20, dq_20

    def _get_robot_state(self) -> np.ndarray:
        """Read current joint positions for all actuators (nu-dim vector)."""
        state = np.zeros(self.model.nu, dtype=np.float32)
        for i in range(self.model.nu):
            trntype = self.model.actuator_trntype[i]
            trnid = self.model.actuator_trnid[i, 0]
            if trntype == 0:  # mjTRN_JOINT
                state[i] = self.data.qpos[self.model.jnt_qposadr[trnid]]
            elif trntype == 3:  # mjTRN_TENDON
                state[i] = self.data.ten_length[trnid]
            else:
                state[i] = self.data.ctrl[i]
        return state

    def set_impedance_mode(self, enabled: bool, stiffness_scale: Optional[float] = None):
        """Toggle compliant Cartesian impedance mode when using iCubWholeBodyID."""
        self.impedance_mode = bool(enabled)
        if isinstance(self.solver, iCubWholeBodyID):
            self.solver.set_impedance_mode(enabled, stiffness_scale=stiffness_scale)

    def _snap_targets_to_ee(self):
        """Snap mocap targets (`r_hand_target`, `l_hand_target`, `head_target`) to current `r_hand_dh_frame`, `l_hand_dh_frame`, `head` poses."""
        mujoco.mj_forward(self.model, self.data)
        self._update_solver_from_mj()

        # Right hand (`r_hand_dh_frame` palm)
        rh_pos = self.data.xpos[self.rh_body_id].copy()
        rh_quat = self.data.xquat[self.rh_body_id].copy()
        if rh_quat[0] < 0:
            rh_quat = -rh_quat
        self.data.mocap_pos[self.rh_mocap_id] = rh_pos
        self.data.mocap_quat[self.rh_mocap_id] = rh_quat

        # Left hand (`l_hand_dh_frame` palm)
        lh_pos = self.data.xpos[self.lh_body_id].copy()
        lh_quat = self.data.xquat[self.lh_body_id].copy()
        if lh_quat[0] < 0:
            lh_quat = -lh_quat
        self.data.mocap_pos[self.lh_mocap_id] = lh_pos
        self.data.mocap_quat[self.lh_mocap_id] = lh_quat

        # Head (`head` SO(3) orientation)
        head_pos = self.data.xpos[self.head_body_id].copy()
        head_quat = self.data.xquat[self.head_body_id].copy()
        if head_quat[0] < 0:
            head_quat = -head_quat
        # Place head_target mocap box slightly above the head so it is easy to grab in GUI
        self.data.mocap_pos[self.head_mocap_id] = head_pos + np.array([0.0, 0.0, 0.18])
        self.data.mocap_quat[self.head_mocap_id] = head_quat

        # Lock current poses as fallback references for unselected arms/head
        self._locked_rh_pos = rh_pos.copy()
        self._locked_rh_rot = quat_wxyz_to_rotmat(rh_quat)
        self._locked_lh_pos = lh_pos.copy()
        self._locked_lh_rot = quat_wxyz_to_rotmat(lh_quat)
        self._locked_head_rot = quat_wxyz_to_rotmat(head_quat)

        # Reference home poses for relative VR tracking
        self._home_pos_rh = rh_pos.copy()
        self._home_rot_rh = quat_wxyz_to_rotmat(rh_quat)
        self._home_pos_lh = lh_pos.copy()
        self._home_rot_lh = quat_wxyz_to_rotmat(lh_quat)
        self._home_rot_head = quat_wxyz_to_rotmat(head_quat)

        # Reset VR calibration baselines so they recalibrate relative to this home pose
        self.calibrated_origin_rh = None
        self.calibrated_origin_lh = None
        self.calibrated_quat_rh = None
        self.calibrated_quat_lh = None
        self.calibrated_quat_head = None
        self._prev_target_pos = {"right": None, "left": None}
        self._clutch_start = {"right": None, "left": None}

        self._q_cmd_20 = self.data.qpos[self._upper_qpos_idxs].copy()
        self.solver.set_desired_right_hand(self._locked_rh_pos, self._locked_rh_rot)
        self.solver.set_desired_left_hand(self._locked_lh_pos, self._locked_lh_rot)
        self.solver.set_desired_head_orientation(self._locked_head_rot)

    def _update_solver_setpoints_from_mocap(self):
        """Transfer current mocap poses (`r_hand_target`, `l_hand_target`, `head_target`) into BLF solver setpoints."""
        if self.control_arms in ("both", "right"):
            rh_pos = self.data.mocap_pos[self.rh_mocap_id].copy()
            rh_rot = quat_wxyz_to_rotmat(self.data.mocap_quat[self.rh_mocap_id])
            self._locked_rh_pos = rh_pos
            self._locked_rh_rot = rh_rot
        self.solver.set_desired_right_hand(self._locked_rh_pos, self._locked_rh_rot)

        if self.control_arms in ("both", "left"):
            lh_pos = self.data.mocap_pos[self.lh_mocap_id].copy()
            lh_rot = quat_wxyz_to_rotmat(self.data.mocap_quat[self.lh_mocap_id])
            self._locked_lh_pos = lh_pos
            self._locked_lh_rot = lh_rot
        self.solver.set_desired_left_hand(self._locked_lh_pos, self._locked_lh_rot)

        if self.use_head_control:
            head_rot = quat_wxyz_to_rotmat(self.data.mocap_quat[self.head_mocap_id])
            self._locked_head_rot = head_rot
        self.solver.set_desired_head_orientation(self._locked_head_rot)

    def _solve_whole_body(self):
        """
        Execute one outer-loop control update when `state == 'teleop_active'`.
        - For TSIK (`controller_type == 'ik'`): solves QP IK for joint velocities dq*, integrates q_cmd,
          and writes position commands to `data.ctrl`. Decouples reference state from delayed physics
          measurements to eliminate delay-induced positive feedback oscillation (PI windup).
        - For TSID (`controller_type == 'id'`): updates the Cartesian setpoints from mocap; actual torque
          QP solve runs inside `_step_physics()` at the 200 Hz physics rate.
        """
        self._update_solver_setpoints_from_mocap()
        if self.controller_type == "ik":
            q_meas = self.data.qpos[self._upper_qpos_idxs].copy()
            max_drift = 0.35  # ~20 deg safety clamp against external collisions
            drift = self._q_cmd_20 - q_meas
            if np.any(np.abs(drift) > max_drift):
                self._q_cmd_20 = np.clip(self._q_cmd_20, q_meas - max_drift, q_meas + max_drift)

            self.solver.update_robot_state(self._q_cmd_20, np.zeros(20, dtype=np.float64))
            dq_opt = self.solver.step()
            self._q_cmd_20 = np.clip(
                self._q_cmd_20 + dq_opt * self.control_dt,
                self._upper_jnt_min,
                self._upper_jnt_max,
            )
            self.data.ctrl[self._upper_act_ids] = self._q_cmd_20

    _solve_ik = _solve_whole_body

    def _step_physics(self):
        """
        Advance MuJoCo physics by `self.frame_skip` substeps.
        When `controller_type == 'id'` (TSID), computes optimal torques tau* at every physics substep.
        """
        for _ in range(self.frame_skip):
            if self.controller_type == "id":
                self._update_solver_from_mj()
                tau_opt = self.solver.step()
                self.data.ctrl[self._upper_act_ids] = tau_opt
            mujoco.mj_step(self.model, self.data)

    # ========================= GO HOME =====================================

    def _start_go_home(self):
        """Begin a minimum-jerk trajectory back to the home joint configuration."""
        self._home_start_q = self.data.qpos[self._upper_qpos_idxs].copy()
        self._home_target_q = self._q_home_20.copy()
        self._home_step = 0
        self.state = "going_home"
        print("[State] → going_home")

    def _interpolate_home(self):
        """Advance one step of the minimum-jerk go-home trajectory."""
        if self._home_step >= self._home_total_steps:
            if self.controller_type == "ik":
                self.data.ctrl[self._upper_act_ids] = self._home_target_q
                self._q_cmd_20 = self._home_target_q.copy()
            else:
                self.solver.set_posture_reference(self._home_target_q)
            self._snap_targets_to_ee()
            self.state = "idle"
            print("[State] Home reached → idle")
            return

        tau = self._home_step / float(self._home_total_steps)
        s = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
        q_des = self._home_start_q + (self._home_target_q - self._home_start_q) * s
        if self.controller_type == "ik":
            self.data.ctrl[self._upper_act_ids] = q_des
            self._q_cmd_20 = q_des.copy()
        else:
            # Update FK targets along q_des for smooth TSID homing
            self.solver.update_robot_state(q_des)
            pos_l, rot_l = self.solver.get_current_left_hand_pose()
            pos_r, rot_r = self.solver.get_current_right_hand_pose()
            rot_h = self.solver.get_current_head_orientation()
            self.solver.set_desired_left_hand(pos_l, rot_l)
            self.solver.set_desired_right_hand(pos_r, rot_r)
            self.solver.set_desired_head_orientation(rot_h)
            self.solver.set_posture_reference(q_des)
        self._home_step += 1

    # ========================= VR SETUP & PROCESSING =======================

    def _init_vr(self):
        self._vr_sockets = {}
        for name, port in [
            ("head", VR_HEAD_PORT),
            ("rh", VR_HAND_RIGHT_PORT),
            ("lh", VR_HAND_LEFT_PORT),
            ("btn", VR_BUTTONS_PORT),
        ]:
            ctx = zmq.Context()
            sock = ctx.socket(zmq.PULL)
            endpoint = f"tcp://*:{port}"
            sock.bind(endpoint)
            sock.setsockopt(zmq.LINGER, 0)
            self._vr_sockets[name] = (ctx, sock)

        self._vr_thread = threading.Thread(target=self._vr_receive_loop, daemon=True)
        self._vr_thread.start()

    def _vr_receive_loop(self):
        while self.running:
            for name in ("rh", "lh", "btn"):
                _, sock = self._vr_sockets[name]
                try:
                    while True:
                        raw = sock.recv_string(flags=zmq.NOBLOCK)
                        if name == "rh":
                            self.latest_rh_data = self._parse_controller(raw)
                            self._last_ctrl_recv_time = time.time()
                        elif name == "lh":
                            self.latest_lh_data = self._parse_controller(raw)
                            self._last_ctrl_recv_time = time.time()
                        elif name == "btn":
                            self.latest_buttons = self._parse_buttons(raw)
                            self._last_btn_recv_time = time.time()
                            if isinstance(self.latest_buttons, dict):
                                for k in self._grip_state.keys() & self.latest_buttons.keys():
                                    self._grip_state[k] = float(self.latest_buttons[k])
                            self._ingest_help_button(self.latest_buttons)
                except zmq.Again:
                    pass
            time.sleep(0.001)

    def _ingest_help_button(self, b):
        """(VR thread) X opens/closes the help panel, if BeaVR sends it to the PC."""
        if not isinstance(b, dict) or not any(k in b for k in ("BTN_ONE_LEFT", "X")):
            return
        pressed = b.get("BTN_ONE_LEFT", 0) > 0.5 or b.get("X", 0) > 0.8
        if pressed and not self._vr_x_held:
            self._vr_help_toggled = not self._vr_help_toggled
        self._vr_x_held = pressed

    def vr_help_active(self) -> bool:
        """True = show the help panel: X pressed, or the controllers sent data and then stopped
        for longer than VR_PAUSE_TIMEOUT_S (app paused)."""
        if self._vr_help_toggled:
            return True
        last = self._last_ctrl_recv_time
        return last > 0 and time.time() - last > VR_PAUSE_TIMEOUT_S

    @staticmethod
    def _parse_controller(data_str: str):
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
                if all(k in j for k in ("w", "x", "y", "z")):
                    quat = PyQuaternion(w=float(j["w"]), x=float(j["x"]), y=float(j["y"]), z=float(j["z"]))
                    return {"pos": pos, "quat": quat}
                return {"pos": pos}
            else:
                parts = data_str.split(",")
                if len(parts) >= 3:
                    pos = np.array([float(p) for p in parts[:3]])
                    if len(parts) >= 7:
                        quat = PyQuaternion(
                            w=float(parts[6]), x=float(parts[3]), y=float(parts[4]), z=float(parts[5])
                        )
                        return {"pos": pos, "quat": quat}
                    return {"pos": pos}
        except Exception:
            pass
        return None

    @staticmethod
    def _parse_buttons(data_str: str):
        try:
            data_str = data_str.strip()
            if data_str.startswith("DIAGNOSTIC"):
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
        if hand_data is None:
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

        target_quat, calib_quat = self._vr_hand_quat(hand_data, calib_quat, home_rot)
        return target_pos, target_quat, calib_origin, calib_quat

    @staticmethod
    def _vr_hand_quat(hand_data, calib_quat, home_rot):
        """Target orientation (wxyz) from the controller, relative to its calibration pose."""
        if "quat" not in hand_data:
            return rotmat_to_quat_wxyz(home_rot), calib_quat
        q_world = unity_to_world_quat(hand_data["quat"])
        if calib_quat is None:
            calib_quat = PyQuaternion(q_world)
        # Relative rotation in world frame from neutral calibration pose
        q_rel = q_world * calib_quat.inverse
        R_target = q_rel.rotation_matrix @ home_rot
        return rotmat_to_quat_wxyz(R_target), calib_quat

    def _clutch_hand(self, side: str, hand_data, engaged: bool) -> bool:
        """Position clutch on the lower (grip) trigger: while pressed the target position does
        not move (the operator repositions the controller); on release the origin shifts by
        what the controller moved, so the target continues from where it stopped, without a
        jump. Orientation has no clutch. True = position held this step."""
        vr_pos = hand_data.get("pos")
        if vr_pos is None or np.linalg.norm(vr_pos) < 0.001:
            return engaged
        if engaged:
            if self._clutch_start[side] is None:
                self._clutch_start[side] = vr_pos.copy()
            return True
        start = self._clutch_start[side]
        if start is not None:
            attr = "calibrated_origin_rh" if side == "right" else "calibrated_origin_lh"
            origin = getattr(self, attr)
            if origin is not None:
                setattr(self, attr, origin + (vr_pos - start))
            self._clutch_start[side] = None
        return False

    def _process_vr_hands(self):
        if self._home_pos_rh is None or self._home_pos_lh is None:
            return

        rh = self.latest_rh_data
        if rh is not None and self._clutch_hand("right", rh, self.freeze_rh):
            quat, self.calibrated_quat_rh = self._vr_hand_quat(rh, self.calibrated_quat_rh, self._home_rot_rh)
            self.data.mocap_quat[self.rh_mocap_id] = quat
        elif rh is not None:
            pos, quat, self.calibrated_origin_rh, self.calibrated_quat_rh = self._vr_hand_to_target(
                self.latest_rh_data,
                self.calibrated_origin_rh,
                self.calibrated_quat_rh,
                self._home_pos_rh,
                self._home_rot_rh,
                side="right",
            )
            if pos is not None:
                pos[0] = np.clip(pos[0], 0.1, 0.75)
                pos[1] = np.clip(pos[1], -0.4, 0.1)
                pos[2] = np.clip(pos[2], 0.4, 1.1)
                self.data.mocap_pos[self.rh_mocap_id] = pos
                if quat is not None:
                    self.data.mocap_quat[self.rh_mocap_id] = quat

        lh = self.latest_lh_data
        if lh is not None and self._clutch_hand("left", lh, self.freeze_lh):
            quat, self.calibrated_quat_lh = self._vr_hand_quat(lh, self.calibrated_quat_lh, self._home_rot_lh)
            self.data.mocap_quat[self.lh_mocap_id] = quat
        elif lh is not None:
            pos, quat, self.calibrated_origin_lh, self.calibrated_quat_lh = self._vr_hand_to_target(
                self.latest_lh_data,
                self.calibrated_origin_lh,
                self.calibrated_quat_lh,
                self._home_pos_lh,
                self._home_rot_lh,
                side="left",
            )
            if pos is not None:
                pos[0] = np.clip(pos[0], 0.1, 0.75)
                pos[1] = np.clip(pos[1], -0.1, 0.4)
                pos[2] = np.clip(pos[2], 0.4, 1.1)
                self.data.mocap_pos[self.lh_mocap_id] = pos
                if quat is not None:
                    self.data.mocap_quat[self.lh_mocap_id] = quat

    def _process_vr_head(self):
        """Map VR headset quaternion directly to SO(3) `head_target` mocap orientation (`mocap_quat`)."""
        if self.freeze_head:
            return
        _, sock = self._vr_sockets.get("head", (None, None))
        if sock is None:
            return
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

            if self._home_rot_head is not None:
                R_head_world = delta_R_world @ self._home_rot_head
            else:
                R_head_zero = np.array([
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ])
                R_head_world = delta_R_world @ R_head_zero

            self.data.mocap_quat[self.head_mocap_id] = rotmat_to_quat_wxyz(R_head_world)

    def _process_button_commands(self):
        b = self.latest_buttons
        if b is None or not isinstance(b, dict):
            return

        if "INDEX_RIGHT" in b:
            self._set_hand_level("right", self._trigger_to_level(b["INDEX_RIGHT"]))
        if "INDEX_LEFT" in b:
            self._set_hand_level("left", self._trigger_to_level(b["INDEX_LEFT"]))

        if b.get("BTN_TWO_LEFT", 0) > 0.5 or b.get("Y", 0) > 0.8:
            self._reset_scenario()

        # Lower (grip) trigger, either hand: clutch of both hands' position (_clutch_hand)
        # and head freeze. _grip_state accumulates both (BeaVR may send one key per message).
        any_active = any(v > 0.8 for v in self._grip_state.values())
        self.freeze_rh = any_active
        self.freeze_lh = any_active
        self.freeze_head = any_active

    # ========================= HAND CONTROL ================================

    def _set_hand(self, side: str, action: str):
        self._set_hand_level(side, 0.0 if action == "open" else 1.0)

    def _set_hand_level(self, side: str, level: float):
        level = float(np.clip(level, 0.0, 1.0))
        self._hand_goal[side] = self._hand_open + level * (self._hand_close_goal - self._hand_open)
        if side == "right":
            self.rh_grip_state = level
        else:
            self.lh_grip_state = level

    def _trigger_to_level(self, value: float) -> float:
        lo, hi = self._vr_grip_deadband
        return float(np.clip((float(value) - lo) / max(hi - lo, 1e-6), 0.0, 1.0))

    def _interpolate_hands(self):
        t = float(self.data.time)
        dt = 0.0 if self._hand_last_t is None else max(t - self._hand_last_t, 0.0)
        self._hand_last_t = t
        max_step = np.abs(self._hand_close_goal - self._hand_open) / max(self._hand_close_time_s, 1e-3) * dt
        for side, act_ids in (("right", self.rh_hand_act_ids), ("left", self.lh_hand_act_ids)):
            goal = self._hand_goal[side]
            if goal is None:
                continue
            cur = np.array([self.data.ctrl[aid] for aid in act_ids])
            new = cur + np.clip(goal - cur, -max_step, max_step)
            for aid, val in zip(act_ids, new):
                self.data.ctrl[aid] = val

    # ========================= SCENE RESET =================================

    def _reset_scenario(self):
        objects = getattr(self, "_scene_objects", [])
        if not objects and not self._scene_joints:
            spawn = getattr(self, "_cube_spawn", None)
            if spawn:
                objects = [{"body": "blue-cube", "spawn": spawn}]
            else:
                return

        def _sample(val, default=0.0):
            if isinstance(val, list) and len(val) == 2:
                return np.random.uniform(val[0], val[1])
            return float(val) if val is not None else default

        # spawn.mirror_y: one coin per reset for all those objects; heads -> y negated (layout
        # mirrored to the other side of the robot)
        mirror = 1.0
        if any(o.get("spawn", {}).get("mirror_y") for o in objects) and np.random.uniform() < 0.5:
            mirror = -1.0

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
                    if spawn.get("mirror_y"):
                        y *= mirror
                    self.data.qpos[qpos_adr : qpos_adr + 3] = [x, y, z]

                quat = spawn.get("quat", [1, 0, 0, 0])
                self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat

                ndof = self.model.jnt_type[self.model.body_jntadr[bid]]
                num_dof = 6 if ndof == 0 else 1
                self.data.qvel[dof_adr : dof_adr + num_dof] = 0
                any_reset = True
            except Exception as e:
                print(f"[Reset] Skip '{body_name}': {e}")

        if any_reset:
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
            except Exception:
                pass
        if self._scene_joints:
            mujoco.mj_forward(self.model, self.data)

    def set_object_poses(self, poses: dict):
        for body_name, (pos, quat) in poses.items():
            try:
                body = self.model.body(body_name)
                jnt_adr = body.jntadr[0]
                if jnt_adr == -1:
                    continue
                qpos_adr = self.model.jnt_qposadr[jnt_adr]
                dof_adr = self.model.jnt_dofadr[jnt_adr]
                self.data.qpos[qpos_adr : qpos_adr + 3] = pos
                self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat if quat is not None else [1, 0, 0, 0]
                ndof = self.model.jnt_type[self.model.body_jntadr[body.id]]
                num_dof = 6 if ndof == 0 else 1
                self.data.qvel[dof_adr : dof_adr + num_dof] = 0
            except Exception as e:
                print(f"[Teleop] Error setting '{body_name}' pose: {e}")
        mujoco.mj_forward(self.model, self.data)

    def set_cube_pose(self, pos, quat):
        self.set_object_poses({"blue-cube": (pos, quat)})

    # ========================= KEYBOARD & CLI ==============================

    def _enqueue_cmd(self, cmd: str):
        with self._pending_cmds_lock:
            self._pending_cmds.append(cmd)

    def key_callback(self, keycode):
        if keycode == glfw.KEY_F:
            self._enqueue_cmd("toggle_teleop")
        elif keycode == glfw.KEY_8:
            self._enqueue_cmd("rh_open")
        elif keycode == glfw.KEY_9:
            self._enqueue_cmd("rh_close")
        elif keycode == glfw.KEY_6:
            self._enqueue_cmd("lh_open")
        elif keycode == glfw.KEY_7:
            self._enqueue_cmd("lh_close")
        elif keycode == glfw.KEY_0:
            self._enqueue_cmd("go_home")
        elif keycode == glfw.KEY_N:
            self._enqueue_cmd("reset_scenario")
        elif keycode == glfw.KEY_C:
            self._enqueue_cmd("recalibrate_vr")
        elif keycode == glfw.KEY_I:
            self._enqueue_cmd("toggle_impedance")

    def _process_pending_cmds(self):
        while True:
            with self._pending_cmds_lock:
                if not self._pending_cmds:
                    break
                cmd = self._pending_cmds.pop(0)

            if cmd == "toggle_teleop":
                if self.state == "teleop_active":
                    self._snap_targets_to_ee()
                    self.state = "idle"
                    print("[State] teleop → idle")
                else:
                    self._snap_targets_to_ee()
                    self.state = "teleop_active"
                    print("[State] → teleop_active")
            elif cmd == "rh_open":
                self._set_hand("right", "open")
            elif cmd == "rh_close":
                self._set_hand("right", "close")
            elif cmd == "lh_open":
                self._set_hand("left", "open")
            elif cmd == "lh_close":
                self._set_hand("left", "close")
            elif cmd == "go_home":
                self._start_go_home()
            elif cmd == "reset_scenario":
                self._reset_scenario()
            elif cmd == "recalibrate_vr":
                self.calibrated_origin_rh = None
                self.calibrated_origin_lh = None
                self.calibrated_quat_rh = None
                self.calibrated_quat_lh = None
                self.calibrated_quat_head = None
                self._prev_target_pos = {"right": None, "left": None}
                print("[VR] Recalibrated origins and orientations")
            elif cmd == "toggle_impedance":
                if self.controller_type == "id":
                    self.set_impedance_mode(not self.impedance_mode)
                    print(f"[TSID] Impedance mode toggled to: {self.impedance_mode}")

    def _stream_images(self, viewer_cam=None, force: bool = False):
        if self.img_stream_enabled and (force or time.time() - self._last_stream_time > 0.066):
            try:
                self._head_renderer.update_scene(
                    self.data, camera=self._vr_head_cam, scene_option=self._head_scene_option
                )
                head_img = cv2.cvtColor(self._head_renderer.render(), cv2.COLOR_RGB2BGR)
                head_img = draw_status_corner(head_img, self.vr_status_lines)
                if self._vr_help_enabled:
                    if self.vr_help_active():
                        draw_help_panel(head_img)
                    else:
                        draw_help_hint(head_img)
                self._head_cam_pub.send_image(head_img)

                if self._vr_front_cam is not None:
                    self._viewer_renderer.update_scene(
                        self.data, camera=self._vr_front_cam, scene_option=self._front_scene_option)
                else:
                    camera_arg = viewer_cam
                    if camera_arg is None:
                        cam = mujoco.MjvCamera()
                        cam.lookat = [0.3, 0.0, 0.7]
                        cam.distance = 1.5
                        cam.elevation = -15
                        cam.azimuth = 180
                        camera_arg = cam
                    self._viewer_renderer.update_scene(self.data, camera=camera_arg)
                viewer_img = self._viewer_renderer.render()
                self._viewer_pub.send_image(cv2.cvtColor(viewer_img, cv2.COLOR_RGB2BGR))
                self._last_stream_time = time.time()
                if not self._stream_first_ok:
                    self._stream_first_ok = True
                    print("  [Stream] Transmitiendo flujo de video hacia Meta Quest 2...")
            except Exception as e:
                key = f"{type(e).__name__}: {e}"[:300]
                if key not in self._stream_errors:
                    self._stream_errors.add(key)
                    print(f"  [Stream ERROR] Image streaming failed: {e}", file=sys.stderr)

    def run(self):
        """Launch interactive MuJoCo viewer and enter the whole-body control loop."""
        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=self.key_callback,
            show_left_ui=True,
            show_right_ui=True,
        ) as viewer:
            viewer.cam.lookat[:] = [0.3, 0.0, 0.7]
            viewer.cam.distance = 1.5
            viewer.cam.elevation = -15
            viewer.cam.azimuth = 180
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

            while viewer.is_running() and self.running:
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0
                step_start = time.time()

                self._process_pending_cmds()

                if self.vr_enabled:
                    self._process_vr_head()
                    self._process_vr_hands()

                if self.state == "teleop_active":
                    self._solve_ik()
                elif self.state == "going_home":
                    self._interpolate_home()

                self._interpolate_hands()
                self._step_physics()

                if self.latest_buttons:
                    self._process_button_commands()
                    self.latest_buttons = None

                if self.vr_enabled:
                    btn_age = (time.time() - self._last_btn_recv_time) if self._last_btn_recv_time > 0 else np.inf
                    if btn_age > 0.5:
                        self.freeze_rh = False
                        self.freeze_lh = False
                        self.freeze_head = False

                self._stream_images(viewer.cam)
                viewer.sync()

                elapsed = time.time() - step_start
                remaining = self.control_dt - elapsed
                if remaining > 0:
                    time.sleep(remaining)

        self.close()

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        _LIVE_CORES.discard(self)
        self.running = False
        vr_thread = getattr(self, "_vr_thread", None)
        if vr_thread is not None and vr_thread.is_alive():
            vr_thread.join(timeout=1.0)
        for renderer in (getattr(self, "_head_renderer", None), getattr(self, "_viewer_renderer", None)):
            if renderer is not None:
                try:
                    renderer.close()
                except Exception:
                    pass
        for pub in (getattr(self, "_head_cam_pub", None), getattr(self, "_viewer_pub", None)):
            if pub is not None:
                try:
                    pub.stop()
                except Exception:
                    pass
        for _, (ctx, sock) in list(getattr(self, "_vr_sockets", {}).items()):
            try:
                sock.close(linger=0)
                ctx.term()
            except Exception:
                pass
        if hasattr(self, "_vr_sockets"):
            self._vr_sockets.clear()


def main():
    default_config = str(_PROJECT_ROOT / "config" / "control_config.yaml")
    default_scene = str(_PROJECT_ROOT / "mujoco_sim" / "assets" / "scenes" / "scene_icub_lift.xml")
    parser = argparse.ArgumentParser(description="MuJoCo iCub Whole-Body Teleoperation (BLF TSIK / TSID)")
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--model", type=str, default=default_scene)
    parser.add_argument("--controller", type=str, default=None, choices=["ik", "id"])
    parser.add_argument("--impedance", action="store_true", help="Enable compliant Cartesian impedance (for --controller id)")
    parser.add_argument("--vr", action="store_true")
    parser.add_argument("--vr-ip", "--quest-ip", dest="vr_ip", type=str, default=None)
    parser.add_argument("--arms", type=str, default=None, choices=["both", "right", "left"])
    parser.add_argument("--primary-arm", type=str, default="right", choices=["right", "left"])
    args = parser.parse_args()

    teleop = MuJoCoTeleop(
        model_path=args.model,
        vr_enabled=args.vr or bool(args.vr_ip),
        control_arms=args.arms,
        primary_arm=args.primary_arm,
        config_path=args.config,
        vr_ip=args.vr_ip,
        controller_type=args.controller,
        impedance_mode=True if args.impedance else None,
    )
    try:
        teleop.run()
    except KeyboardInterrupt:
        teleop.running = False
    finally:
        teleop.close()


if __name__ == "__main__":
    main()
