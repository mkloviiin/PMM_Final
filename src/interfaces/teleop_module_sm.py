#!/usr/bin/env python3
"""
teleop_module_sm.py
===================
Unified YARP Teleoperation & Policy Execution RFModule for Gazebo Sim and Real iCub,
powered exclusively by BLFRobotBridge and BLF Whole-Body Controllers:
  - iCubWholeBodyIK (TSIK - QP Fixed-Base Inverse Kinematics)
  - iCubWholeBodyID (TSID - QP Fixed-Base Task-Space Inverse Dynamics & Cartesian Impedance)

All legacy iKin dependencies (iKinCartesianSolver, iKinGazeCtrl, icubyarpinterface.py)
are completely eliminated:
  - Right & Left Hands: Tracked in SE(3) on frames `r_hand` and `l_hand` (root_link frame).
  - Head Orientation: Tracked in SO(3) on frame `head` via quaternion [qw, qx, qy, qz] (root_link frame).
  - Hand Fingers (9 DoFs/hand) & Eyes (3 DoFs locked at 0 deg): Managed by BLFRobotBridge.
"""

import sys
import os
import time
import subprocess
import faulthandler
from pathlib import Path
from collections import deque
from typing import Optional, Dict, Tuple

import numpy as np
import cv2
import yaml
import yarp

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
from interfaces.blf_robot_bridge import BLFRobotBridge
from controllers.joint_limits import load_joint_limits

from interfaces.zmq_image_publisher import ZMQCompressedImageTransmitter
# VR help (RPC vr_help, forwarded by play.py when the BeaVR app pauses or X is pressed)
from interfaces.vr_overlay import draw_help_hint, draw_help_panel

CAMERA_RELAY_DEFAULT_HW = (240, 320)
CONFIG_FILE = os.getenv(
    "ICUB_LEROBOT_CONFIG",
    str(_PROJECT_ROOT / "config" / "control_config.yaml"),
)

VR_HEAD_CAM_PORT = 10505
VR_VIEWER_PORT = 15001
FRONT_RETRY_S = 2.0
SLOW_CYCLE_S = 0.05
STALL_DUMP_S = 1.0


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


class TeleopModuleSM(yarp.RFModule):
    """
    Whole-Body Teleoperation & VLA Execution YARP RFModule using BLFRobotBridge
    and iCubWholeBodyIK (TSIK) or iCubWholeBodyID (TSID / Cartesian Impedance).
    """

    def configure(self, rf) -> bool:
        try:
            print("=" * 68)
            print("STARTING ICUB-LEROBOT WHOLE-BODY TELEOP MODULE (BLF TSIK / TSID)")
            print("=" * 68)

            self.state = "idle"
            self._pending_state = None
            self._pending_impedance = None
            self.pending_hand_cmds = deque()
            self._closed = False

            print(f"[1/7] Loading configuration from {CONFIG_FILE}...")
            with open(CONFIG_FILE, "r") as f:
                self.cfg = yaml.safe_load(f)

            robot_name = (
                rf.find("robot").asString()
                if rf.check("robot")
                else self.cfg.get("robot_name", "icubSim")
            )
            if rf.check("model"):
                robot_model = rf.find("model").asString()
            elif os.environ.get("ICUB_ROBOT_MODEL"):
                robot_model = os.environ.get("ICUB_ROBOT_MODEL")
            elif Path("/tmp/icub_gazebo_model").exists():
                try:
                    m = Path("/tmp/icub_gazebo_model").read_text(encoding="utf-8").strip()
                    robot_model = m if m else self.cfg.get("robot_model", "iCubGazeboV2_5_visuomanip")
                except Exception:
                    robot_model = self.cfg.get("robot_model", "iCubGazeboV2_5_visuomanip")
            else:
                robot_model = self.cfg.get("robot_model", "iCubGazeboV2_5_visuomanip")
            self.controller_type = (
                rf.find("controller").asString().lower()
                if rf.check("controller")
                else str(self.cfg.get("controller_type", "ik")).lower()
            )
            self.impedance_mode = (
                rf.check("impedance")
                or bool(self.cfg.get("impedance_mode", False))
            )
            yarp_mode_cfg = str(self.cfg.get("yarp_control_mode", "position_direct")).lower()
            if rf.check("mode"):
                yarp_mode_cfg = rf.find("mode").asString().lower()
            if self.controller_type == "id":
                self.yarp_mode = "torque"
            else:
                self.yarp_mode = yarp_mode_cfg if yarp_mode_cfg in ("position_direct", "velocity") else "position_direct"

            self.is_gazebo = robot_name.lower() == "icubsim"
            if rf.check("control-arms"):
                control_arms = rf.find("control-arms").asString()
            elif rf.check("control_arms"):
                control_arms = rf.find("control_arms").asString()
            elif rf.check("control"):
                control_arms = rf.find("control").asString()
            elif rf.check("arms"):
                control_arms = rf.find("arms").asString()
            else:
                control_arms = self.cfg.get("control_arms", "right")
            self.control_arms = str(control_arms).strip().lower()
            try:
                Path("/tmp/icub_teleop_arms").write_text(self.control_arms, encoding="utf-8")
            except Exception:
                pass
            control_arms = self.control_arms
            self.use_head_control = bool(self.cfg.get("use_head_control", True))
            self.period = float(self.cfg.get("period", 0.02))
            is_real = robot_name.strip().lower() == "icub"
            is_visuomanip = "visuomanip" in robot_model.lower()
            if not is_real and not is_visuomanip:
                self.show_cameras = False
            else:
                self.show_cameras = bool(self.cfg.get("show_cameras", True))
            self.home_pos_deg = self.cfg.get("home_pose", {})
            self.use_tactile = bool(self.cfg.get("tactile_enabled", True))
            self.publish_torques = bool(self.cfg.get("use_joint_torque", True))

            self.tactile_source_port = self.cfg.get(
                "tactile_source_port", f"/{robot_name}/skin/right_hand_comp"
            )
            self.tactile_include_palm = bool(self.cfg.get("tactile_include_palm", True))
            self.tactile_palm_source_port = self.cfg.get(
                "tactile_palm_source_port", f"/{robot_name}/skin/right_hand"
            )
            self.tactile_auto_connect = bool(self.cfg.get("tactile_auto_connect", True))

            # Build ordered list of actuators for /teleop/robot_state:o matching lerobot_robot_icubrg
            full_actuators = self.cfg.get("actuators", {})
            self.actuators_to_use = {}
            self.actuators_to_use["torso"] = full_actuators.get("torso", [])
            self.cartesian_arms = []
            if control_arms in ("left", "both"):
                self.actuators_to_use["left_arm"] = full_actuators.get("left_arm", [])
                self.cartesian_arms.append("left_arm")
            if control_arms in ("right", "both"):
                self.actuators_to_use["right_arm"] = full_actuators.get("right_arm", [])
                self.cartesian_arms.append("right_arm")
            self.actuators_to_use["head"] = full_actuators.get("head", [])

            self.actuators_order = [
                j for part_joints in self.actuators_to_use.values() for j in part_joints
            ]

            print(
                f"✓ Robot: /{robot_name} ({robot_model}) | Controller: {self.controller_type.upper()} "
                f"(mode={self.yarp_mode}, impedance={self.impedance_mode}) | Arms: {self.cartesian_arms} | Cameras: {self.show_cameras}"
            )

            # [2/7] Connect BLFRobotBridge
            print("[2/7] Connecting BLFRobotBridge...")
            hand_open = self.cfg.get("hand_open_values", [[0, 75, 0, 0, 0, 0, 0, 0, 0]])[0]
            hand_close = self.cfg.get("hand_close_values", [[0, 40, 40, 50, 75, 75, 75, 75, 0]])[0]
            close_ratio = float(self.cfg.get("hand_close_ratio", 0.5))

            self.bridge = BLFRobotBridge(
                robot_name=robot_name,
                local_prefix="teleop_module_blf",
                frequency=1.0 / self.period,
                default_mode=self.yarp_mode,
                max_admissible_error=0.15,
                use_network_clock=False,
                hand_open_deg=hand_open,
                hand_close_deg=hand_close,
                hand_close_ratio=close_ratio,
            )
            if not self.bridge.connect(
                timeout=30.0,
                enable_hands_and_eyes=True,
                enable_cameras=self.show_cameras,
            ):
                print("✗ ERROR: BLFRobotBridge failed to connect.")
                return False
            # Hand speed: a full close in hand_close_time_s, same as MuJoCo
            close_time = self.cfg.get("hand_close_time_s")
            if close_time:
                for arm in ("left_arm", "right_arm"):
                    if arm in self.actuators_to_use:
                        ok = self.bridge.set_hand_ref_speeds(arm, float(close_time))
                        print(f"  Hand {arm} close time {float(close_time):.1f} s -> {'ok' if ok else 'not set'}")

            # [3/7] Initialize Whole-Body Controller (iCubWholeBodyIK or iCubWholeBodyID)
            print(f"[3/7] Initializing Whole-Body Solver ({self.controller_type.upper()})...")
            tsik_cfg = self.cfg.get("tsik", {}) or {}
            tsid_cfg = self.cfg.get("tsid", {}) or {}

            # Joint limits: joint_limits_deg of the config is the only source (controllers/joint_limits.py)
            self.joint_limits = load_joint_limits(self.cfg)

            if self.controller_type == "id":
                self.solver = iCubWholeBodyID(
                    sampling_time=self.period,
                    robot_model_name=robot_model,
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
                    joint_limits=self.joint_limits,
                    joint_limit_horizon_s=float(tsid_cfg.get("joint_limit_horizon_s", 0.1)),
                )
            else:
                self.solver = iCubWholeBodyIK(
                    sampling_time=self.period,
                    robot_model_name=robot_model,
                    kp_cartesian=float(tsik_cfg.get("kp_cartesian", 20.0)),
                    kp_angular=float(tsik_cfg.get("kp_angular", 15.0)),
                    weight_hands=float(tsik_cfg.get("weight_hands", 180.0)),
                    weight_head=float(tsik_cfg.get("weight_head", 5.0)),
                    weight_posture_torso=float(tsik_cfg.get("weight_posture_torso", 80.0)),
                    weight_posture_neck=float(tsik_cfg.get("weight_posture_neck", 0.05)),
                    weight_posture_arms=float(tsik_cfg.get("weight_posture_arms", 0.05)),
                    max_joint_velocity=float(tsik_cfg.get("max_joint_velocity", 1.5)),
                    joint_limits=self.joint_limits,
                )

            # Bounds of the 20 joints to clamp q_cmd safely (the solver model already has joint_limits)
            model = self.solver.model_loader.model()
            self.jnt_min_20 = np.array([model.getJoint(model.getJointIndex(j)).getMinPosLimit(0) for j in self.solver.JOINTS_LIST], dtype=np.float64)
            self.jnt_max_20 = np.array([model.getJoint(model.getJointIndex(j)).getMaxPosLimit(0) for j in self.solver.JOINTS_LIST], dtype=np.float64)
            # Gazebo: same limits on the simulated controlboards. Real robot: solver only.
            if self.is_gazebo and self.joint_limits:
                self.bridge.apply_joint_limits(self.joint_limits)

            # Compute home posture in radians (20 DoFs)
            self.q_home_20 = self.bridge.home_pose_dict_to_vector(
                self.home_pos_deg,
                fallback_rad=self.solver.DEFAULT_HOME_POSTURE,
            )
            self.solver.set_posture_reference(self.q_home_20)

            # Move smoothly to home posture at startup
            print("[4/7] Moving robot smoothly to home posture...")
            self.bridge.move_to_posture(self.q_home_20, duration=3.0)
            self.bridge.set_hand_closure("right_arm", 0.0)
            self.bridge.set_hand_closure("left_arm", 0.0)
            self.bridge.lock_eyes_forward()

            # Sync solver state and lock initial Cartesian targets
            self._sync_solver_and_lock_targets()

            # [5/7] Open YARP RPC & Data Ports
            print("[5/7] Opening YARP RPC and streaming ports...")
            self.cmd_port = yarp.Port()
            if not self.cmd_port.open("/teleop/rpc:i"):
                print("✗ ERROR: Failed to open /teleop/rpc:i")
                return False
            self.attach(self.cmd_port)

            self.ports: Dict[str, yarp.Port] = {}

            # Hand target & current pose ports (always open for right and left so plugins can connect)
            for prefix in ("rh", "lh"):
                t_port = yarp.BufferedPortBottle()
                t_port.open(f"/teleop/{prefix}_target:i")
                self.ports[f"{prefix}_target"] = t_port

                c_port = yarp.BufferedPortBottle()
                c_port.open(f"/teleop/{prefix}_current_pose:o")
                self.ports[f"{prefix}_current_pose"] = c_port

            # Head SO(3) orientation target & current quaternion ports
            head_t_port = yarp.BufferedPortBottle()
            head_t_port.open("/teleop/head_target:i")
            self.ports["head_target"] = head_t_port

            head_c_port = yarp.BufferedPortBottle()
            head_c_port.open("/teleop/head_current_quat:o")
            self.ports["head_current_quat"] = head_c_port

            # Robot state port
            self.ports["robot_state"] = yarp.Port()
            self.ports["robot_state"].open("/teleop/robot_state:o")
            self._state_seq = 0

            # Command ACK port (for latency telemetry)
            self.ports["cmd_ack"] = yarp.BufferedPortBottle()
            self.ports["cmd_ack"].open("/teleop/cmd_ack:o")

            # Tactile ports
            if self.use_tactile:
                self.ports["r_hand_touch"] = yarp.BufferedPortVector()
                self.ports["r_hand_touch"].open("/teleop/r_hand_touch:i")
                if self.tactile_include_palm:
                    self.ports["r_hand_palm_touch"] = yarp.BufferedPortVector()
                    self.ports["r_hand_palm_touch"].open("/teleop/r_hand_palm_touch:i")
                self.ports["touch_state"] = yarp.Port()
                self.ports["touch_state"].open("/teleop/touch:o")

                if self.tactile_auto_connect:
                    if yarp.Network.exists(self.tactile_source_port):
                        yarp.Network.connect(self.tactile_source_port, "/teleop/r_hand_touch:i", "fast_tcp")
                    if self.tactile_include_palm and yarp.Network.exists(self.tactile_palm_source_port):
                        yarp.Network.connect(self.tactile_palm_source_port, "/teleop/r_hand_palm_touch:i", "fast_tcp")

            # Camera relay ports
            self.ports["view_out"] = yarp.BufferedPortImageRgb()
            self.ports["view_out"].open("/teleop/view:o")

            shapes = self.cfg.get("camera_shapes") or {}
            self.cam_relay_hw = {}
            for eye in ("left", "right"):
                port_name = f"/teleop/cam/{eye}:o"
                self.ports[f"cam_relay_{eye}"] = yarp.BufferedPortImageRgb()
                self.ports[f"cam_relay_{eye}"].open(port_name)
                h, w = shapes.get(eye, CAMERA_RELAY_DEFAULT_HW)
                self.cam_relay_hw[eye] = (int(h), int(w))

            # External front camera port (Gazebo camera-stand)
            self.front_source_port = self.cfg.get(
                "front_camera_port", f"/{robot_name}/cam/front/rgbImage:o"
            )
            self.front_local_port = "/teleop/front:i"
            self.ports["front_in"] = yarp.BufferedPortImageRgb()
            self.ports["front_in"].open(self.front_local_port)
            self.front_buf = yarp.ImageRgb()
            self.front_arr = None
            self._front_last_try = 0.0
            self._front_last_frame_t = 0.0
            self._front_fallback_warned = False
            self.vr_status_text = "EN ESPERA"
            self.vr_help_enabled = bool(self.cfg.get("vr_help", True))
            self.vr_help_panel = False

            # Wider-FOV camera for the headset (Gazebo: vr_camera_nws.xml in the iCub model).
            # Without frames (real robot) the VR shows the eye camera.
            self.vr_cam_source_port = self.cfg.get("vr_camera_port", f"/{robot_name}/cam/vr/rgbImage:o")
            self.vr_cam_local_port = "/teleop/vr_cam:i"
            self.ports["vr_cam_in"] = yarp.BufferedPortImageRgb()
            self.ports["vr_cam_in"].open(self.vr_cam_local_port)
            self.vr_cam_buf = yarp.ImageRgb()
            self.vr_cam_arr = None
            self.vr_cam_bgr = None
            self._vr_cam_last_try = 0.0
            self._vr_cam_last_frame_t = 0.0
            if self._connect_vr_camera():
                print(f"  ✓ VR camera connected: {self.vr_cam_source_port}")
            self._status_img = None
            self._status_img_text = None
            self._status_last_send = 0.0
            self._connect_front_camera()

            # Internal state & caches
            self.l_img = None
            self.r_img = None
            self.last_touch = np.zeros(60, dtype=np.float64)
            self.last_touch_fingers = np.zeros(60, dtype=np.float64)
            self.last_touch_palm = np.zeros(0, dtype=np.float64)
            self._vr_eye_sent = 0

            # [6/7] Initialize VR ZMQ publishers
            print("[6/7] Initializing VR ZMQ image publishers...")
            if ZMQCompressedImageTransmitter is not None:
                try:
                    self.vr_pub = ZMQCompressedImageTransmitter(host="*", port=VR_HEAD_CAM_PORT)
                    print(f"  ✓ VR eye publisher on port {VR_HEAD_CAM_PORT}")
                except Exception as e:
                    print(f"  ⚠ VR eye publisher not started: {e}")
                try:
                    self.vr_viewer_pub = ZMQCompressedImageTransmitter(host="*", port=VR_VIEWER_PORT)
                    print(f"  ✓ VR front publisher on port {VR_VIEWER_PORT}")
                except Exception as e:
                    print(f"  ⚠ VR front publisher not started: {e}")

            print("[7/7] Module ready! Current state:", self.state)
            print("=" * 68)
            return True

        except Exception as e:
            print(f"✗ ERROR during configuration: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _sync_solver_and_lock_targets(self):
        """Read current joint state from BLFRobotBridge and lock current Cartesian poses as setpoints."""
        q_curr = self.bridge.get_joint_positions()
        dq_curr = self.bridge.get_joint_velocities()
        self.q_integrated = q_curr.copy()
        self.solver.update_robot_state(q_curr, dq_curr)

        pos_l, rot_l = self.solver.get_current_left_hand_pose()
        pos_r, rot_r = self.solver.get_current_right_hand_pose()
        rot_h = self.solver.get_current_head_orientation()

        self.des_lh_pos = pos_l.copy()
        self.des_lh_rot = rot_l.copy()
        self.des_rh_pos = pos_r.copy()
        self.des_rh_rot = rot_r.copy()
        self.des_head_rot = rot_h.copy()

        self.solver.set_desired_left_hand(self.des_lh_pos, self.des_lh_rot)
        self.solver.set_desired_right_hand(self.des_rh_pos, self.des_rh_rot)
        self.solver.set_desired_head_orientation(self.des_head_rot)

    def respond(self, command, reply) -> bool:
        """Handle RPC commands on /teleop/rpc:i in a thread-safe manner."""
        try:
            cmd = command.get(0).asString()
            if cmd not in ("head_quat", "look_at"):
                print(f"[RPC] Command received: '{cmd}'")

            if cmd == "quit":
                self._pending_state = "quit"
                reply.addString("Quitting module.")

            elif cmd == "go_home":
                self._pending_state = "going_home"
                reply.addString("Going to home pose.")

            elif cmd == "start_teleop":
                self._pending_state = "start_teleop"
                reply.addString("Teleop active. Processing continuous whole-body targets.")

            elif cmd == "stop_teleop":
                self._pending_state = "stop_teleop"
                reply.addString("Teleop stopped. Idle.")

            elif cmd in ("start_gaze", "stop_gaze"):
                reply.addString("ok")

            elif cmd == "set_impedance":
                if command.size() >= 2:
                    enabled = bool(command.get(1).asInt32())
                    self._pending_impedance = enabled
                    reply.addString(f"Impedance mode set to {enabled}")
                else:
                    reply.addString("Usage: set_impedance <0|1>")

            elif cmd == "get_control_arms":
                reply.addString(self.control_arms)

            elif cmd == "open_right":
                self.pending_hand_cmds.append(("right_arm", 0.0))
                reply.addString("Opening right hand.")

            elif cmd == "close_right":
                self.pending_hand_cmds.append(("right_arm", 1.0))
                reply.addString("Closing right hand.")

            elif cmd == "open_left":
                self.pending_hand_cmds.append(("left_arm", 0.0))
                reply.addString("Opening left hand.")

            elif cmd == "close_left":
                self.pending_hand_cmds.append(("left_arm", 1.0))
                reply.addString("Closing left hand.")

            elif cmd in ("hand_right", "hand_left"):
                if command.size() >= 2:
                    level = float(np.clip(command.get(1).asFloat64(), 0.0, 1.0))
                    arm = "right_arm" if cmd == "hand_right" else "left_arm"
                    self.pending_hand_cmds.append((arm, level))
                    reply.addString("ok")
                else:
                    reply.addString(f"Usage: {cmd} <level 0..1>")

            elif cmd == "head_quat":
                if command.size() >= 5:
                    quat_wxyz = np.array([command.get(i).asFloat64() for i in range(1, 5)], dtype=np.float64)
                    if np.isfinite(quat_wxyz).all() and np.linalg.norm(quat_wxyz) > 1e-6:
                        self.des_head_rot = quat_wxyz_to_rotmat(quat_wxyz)
                    reply.addString("ok")
                else:
                    reply.addString("Usage: head_quat qw qx qy qz")

            elif cmd == "move_right_arm":
                if command.size() >= 8:
                    pos = np.array([command.get(i).asFloat64() for i in range(1, 4)], dtype=np.float64)
                    quat = np.array([command.get(i).asFloat64() for i in range(4, 8)], dtype=np.float64)
                    if np.isfinite(pos).all() and np.isfinite(quat).all():
                        self.des_rh_pos = pos
                        self.des_rh_rot = quat_wxyz_to_rotmat(quat)
                    reply.addString("ok")
                else:
                    reply.addString("Usage: move_right_arm x y z qw qx qy qz")

            elif cmd == "move_left_arm":
                if command.size() >= 8:
                    pos = np.array([command.get(i).asFloat64() for i in range(1, 4)], dtype=np.float64)
                    quat = np.array([command.get(i).asFloat64() for i in range(4, 8)], dtype=np.float64)
                    if np.isfinite(pos).all() and np.isfinite(quat).all():
                        self.des_lh_pos = pos
                        self.des_lh_rot = quat_wxyz_to_rotmat(quat)
                    reply.addString("ok")
                else:
                    reply.addString("Usage: move_left_arm x y z qw qx qy qz")

            elif cmd == "vr_status":
                self.vr_status_text = command.get(1).asString() if command.size() > 1 else ""
                reply.addString("ok")

            elif cmd == "vr_help":
                self.vr_help_panel = command.size() > 1 and command.get(1).asInt32() != 0
                reply.addString("ok")

            elif cmd == "reset_scenario":
                if not self.is_gazebo:
                    reply.addString("Reset not available on real robot.")
                    return True
                script_path = str(_PROJECT_ROOT / "gazebo_sim" / "scripts" / "reset_objects.sh")
                if os.path.exists(script_path):
                    subprocess.Popen(["bash", script_path])
                    reply.addString("Scenario reset.")
                else:
                    reply.addString("Reset script not found.")

            else:
                reply.addString(f"Command '{cmd}' not recognized.")

        except Exception as e:
            print(f"[RPC] Exception in respond: {e}")
            import traceback
            traceback.print_exc()
            reply.clear()
            reply.addString(f"ERROR: {e}")
            return True

        return True

    def getPeriod(self) -> float:
        return self.period

    def updateModule(self) -> bool:
        if self.state == "quit":
            self.close()
            return False

        faulthandler.dump_traceback_later(STALL_DUMP_S, exit=False)
        try:
            return self._update_state_machine()
        finally:
            faulthandler.cancel_dump_traceback_later()

    def _update_state_machine(self) -> bool:
        if self._pending_impedance is not None:
            if isinstance(self.solver, iCubWholeBodyID):
                self.solver.set_impedance_mode(self._pending_impedance)
                self.impedance_mode = self._pending_impedance
            self._pending_impedance = None

        if self._pending_state is not None:
            req = self._pending_state
            self._pending_state = None
            if req == "quit":
                self.state = "quit"
            elif req == "going_home":
                self.state = "going_home"
            elif req == "start_teleop":
                self._sync_solver_and_lock_targets()
                self.state = "teleop_active"
                print("[State] Teleop started -> teleop_active")
            elif req == "stop_teleop":
                self._sync_solver_and_lock_targets()
                self.state = "idle"
                print("[State] Teleop stopped -> idle")

        if self.state == "going_home":
            print("[State] Moving smoothly to home posture...")
            self.bridge.move_to_posture(self.q_home_20, duration=2.5, step_callback=self._send_robot_state)
            self.bridge.set_hand_closure("right_arm", 0.0)
            self.bridge.set_hand_closure("left_arm", 0.0)
            self.bridge.lock_eyes_forward()
            self._sync_solver_and_lock_targets()
            self.state = "idle"
            print("[State] Home reached. Transitioning to: idle")

        elif self.state == "teleop_active":
            # Read non-blocking Cartesian hand targets (in root_link frame)
            if "right_arm" in self.cartesian_arms and "rh_target" in self.ports:
                rh_b = self.ports["rh_target"].read(False)
                if rh_b is not None and rh_b.size() >= 7:
                    pos = np.array([rh_b.get(i).asFloat64() for i in range(3)], dtype=np.float64)
                    quat = np.array([rh_b.get(i).asFloat64() for i in range(3, 7)], dtype=np.float64)
                    if np.isfinite(pos).all() and np.isfinite(quat).all() and np.linalg.norm(quat) > 1e-6:
                        t_recv = time.time()
                        self.des_rh_pos = pos
                        self.des_rh_rot = quat_wxyz_to_rotmat(quat)
                        self._ack_target("rh", "rh_target", t_recv, True)

            if "left_arm" in self.cartesian_arms and "lh_target" in self.ports:
                lh_b = self.ports["lh_target"].read(False)
                if lh_b is not None and lh_b.size() >= 7:
                    pos = np.array([lh_b.get(i).asFloat64() for i in range(3)], dtype=np.float64)
                    quat = np.array([lh_b.get(i).asFloat64() for i in range(3, 7)], dtype=np.float64)
                    if np.isfinite(pos).all() and np.isfinite(quat).all() and np.linalg.norm(quat) > 1e-6:
                        t_recv = time.time()
                        self.des_lh_pos = pos
                        self.des_lh_rot = quat_wxyz_to_rotmat(quat)
                        self._ack_target("lh", "lh_target", t_recv, True)

            # Read non-blocking SO(3) head orientation target [qw, qx, qy, qz] (in root_link frame)
            if self.use_head_control and "head_target" in self.ports:
                head_b = self.ports["head_target"].read(False)
                if head_b is not None and head_b.size() >= 4:
                    quat = np.array([head_b.get(i).asFloat64() for i in range(4)], dtype=np.float64)
                    if np.isfinite(quat).all() and np.linalg.norm(quat) > 1e-6:
                        self.des_head_rot = quat_wxyz_to_rotmat(quat)

            # Step Whole-Body Solver (TSIK or TSID)
            self._step_whole_body_controller()

        elif self.state == "idle":
            # In TSID (torque) mode, keep actively compensating gravity and holding locked pose
            if self.controller_type == "id":
                self._step_whole_body_controller()

        # Process queued hand commands, camera relays, and state broadcast
        self._process_hand_cmds()
        self._check_cameras()
        self._send_robot_state()
        return True

    def _step_whole_body_controller(self):
        """Execute one step of iCubWholeBodyIK or iCubWholeBodyID and command BLFRobotBridge."""
        q_curr = self.bridge.get_joint_positions()
        dq_curr = self.bridge.get_joint_velocities()
        self.solver.update_robot_state(q_curr, dq_curr)

        self.solver.set_desired_left_hand(self.des_lh_pos, self.des_lh_rot)
        self.solver.set_desired_right_hand(self.des_rh_pos, self.des_rh_rot)
        self.solver.set_desired_head_orientation(self.des_head_rot)

        if self.controller_type == "id":
            torques = self.solver.step()
            self.bridge.send_torques(torques)
        else:
            dq_opt = self.solver.step()
            if self.yarp_mode == "velocity":
                self.bridge.send_velocities(dq_opt)
            else:
                q_cmd = np.clip(q_curr + dq_opt * self.period, self.jnt_min_20, self.jnt_max_20)
                self.bridge.send_position_direct(q_cmd, q_curr)

    def _ack_target(self, side: str, port_key: str, t_recv: float, ok: bool):
        port = self.ports.get("cmd_ack")
        if port is None or port.getOutputCount() == 0:
            return
        stamp = yarp.Stamp()
        if not (self.ports[port_key].getEnvelope(stamp) and stamp.isValid()):
            return
        bot = port.prepare()
        bot.clear()
        bot.addString(side)
        bot.addInt64(stamp.getCount())
        bot.addFloat64(stamp.getTime())
        bot.addFloat64(t_recv)
        bot.addFloat64(time.time())
        bot.addInt32(1 if ok else -1)
        port.write()

    def _process_hand_cmds(self):
        latest = {}
        while self.pending_hand_cmds:
            arm, level = self.pending_hand_cmds.popleft()
            latest[arm] = float(level)
        for arm, level in latest.items():
            self.bridge.set_hand_closure(arm, level)

    def _check_cameras(self):
        if not self.show_cameras:
            return
        self._stream_front_camera()
        l_rgb, r_rgb = self.bridge.get_camera_images()
        self._relay_cameras({"left": l_rgb, "right": r_rgb})

        if l_rgb is not None:
            self.l_img = cv2.cvtColor(l_rgb, cv2.COLOR_RGB2BGR)
        if r_rgb is not None:
            self.r_img = cv2.cvtColor(r_rgb, cv2.COLOR_RGB2BGR)

        self._stream_status_screen()

        img_view = self.r_img.copy() if self.r_img is not None else (self.l_img.copy() if self.l_img is not None else None)
        if img_view is not None:
            vr_src = self._read_vr_camera()
            img_vr = vr_src.copy() if vr_src is not None else img_view.copy()
            if time.time() - self._front_last_frame_t <= 1.0:
                img_vr = draw_status_corner(img_vr, self.vr_status_text.split("|"))
            if self.vr_help_enabled:
                if self.vr_help_panel:
                    draw_help_panel(img_vr)
                else:
                    draw_help_hint(img_vr)

            if hasattr(self, "vr_pub") and self.vr_pub is not None:
                try:
                    self.vr_pub.send_image(img_vr)
                    self._vr_eye_sent += 1
                except Exception:
                    pass

            if "view_out" in self.ports and self.ports["view_out"].getOutputCount() > 0:
                h_i, w_i = img_view.shape[:2]
                y_img = self.ports["view_out"].prepare()
                y_img.resize(w_i, h_i)
                img_rgb_yarp = cv2.cvtColor(img_view, cv2.COLOR_BGR2RGB)
                temp_wrapper = yarp.ImageRgb()
                temp_wrapper.setExternal(img_rgb_yarp.data, w_i, h_i)
                y_img.copy(temp_wrapper)
                self.ports["view_out"].write()

    def _relay_cameras(self, frames: Dict[str, Optional[np.ndarray]]):
        for eye, img in frames.items():
            port = self.ports.get(f"cam_relay_{eye}")
            if img is None or port is None or port.getOutputCount() == 0:
                continue
            h, w = self.cam_relay_hw[eye]
            out = img if img.shape[:2] == (h, w) else cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            out = np.ascontiguousarray(out)
            wrapper = yarp.ImageRgb()
            wrapper.setExternal(out.data, w, h)
            y_img = port.prepare()
            y_img.resize(w, h)
            y_img.copy(wrapper)
            stamp = self.bridge.cam_stamps.get(eye)
            if stamp is not None:
                port.setEnvelope(yarp.Stamp(*stamp))
            port.write()

    def _connect_vr_camera(self) -> bool:
        self._vr_cam_last_try = time.time()
        if yarp.Network.isConnected(self.vr_cam_source_port, self.vr_cam_local_port):
            return True
        return (
            yarp.Network.exists(self.vr_cam_source_port, True)
            and yarp.Network.connect(self.vr_cam_source_port, self.vr_cam_local_port, "fast_tcp", True)
        )

    def _read_vr_camera(self) -> Optional[np.ndarray]:
        """Last VR camera frame in BGR, or None if no frame for more than 1 s (real robot)."""
        img = self.ports["vr_cam_in"].read(False)
        now = time.time()
        if img is None:
            if now - self._vr_cam_last_frame_t > FRONT_RETRY_S and now - self._vr_cam_last_try > FRONT_RETRY_S:
                if self._connect_vr_camera():
                    print(f"  ✓ VR camera connected: {self.vr_cam_source_port}")
        else:
            w, h = img.width(), img.height()
            if self.vr_cam_arr is None or self.vr_cam_arr.shape[:2] != (h, w):
                self.vr_cam_arr = np.zeros((h, w, 3), dtype=np.uint8)
                self.vr_cam_buf.resize(w, h)
                self.vr_cam_buf.setExternal(self.vr_cam_arr.data, w, h)
            self.vr_cam_buf.copy(img)
            self.vr_cam_bgr = cv2.cvtColor(self.vr_cam_arr, cv2.COLOR_RGB2BGR)
            self._vr_cam_last_frame_t = now
        return self.vr_cam_bgr if now - self._vr_cam_last_frame_t <= 1.0 else None

    def _connect_front_camera(self) -> bool:
        self._front_last_try = time.time()
        if yarp.Network.isConnected(self.front_source_port, self.front_local_port):
            return True
        return (
            yarp.Network.exists(self.front_source_port, True)
            and yarp.Network.connect(self.front_source_port, self.front_local_port, "fast_tcp", True)
        )

    def _stream_front_camera(self):
        if "front_in" not in self.ports:
            return
        img = self.ports["front_in"].read(False)
        if img is None:
            now = time.time()
            if now - self._front_last_frame_t > FRONT_RETRY_S and now - self._front_last_try > FRONT_RETRY_S:
                self._connect_front_camera()
            return

        w, h = img.width(), img.height()
        if self.front_arr is None or self.front_arr.shape[:2] != (h, w):
            self.front_arr = np.zeros((h, w, 3), dtype=np.uint8)
            self.front_buf.resize(w, h)
            self.front_buf.setExternal(self.front_arr.data, w, h)
        self.front_buf.copy(img)
        self._front_last_frame_t = time.time()

        if getattr(self, "vr_viewer_pub", None) is not None:
            try:
                self.vr_viewer_pub.send_image(cv2.cvtColor(self.front_arr, cv2.COLOR_RGB2BGR))
            except Exception:
                pass

    def _stream_status_screen(self):
        now = time.time()
        if (
            getattr(self, "vr_viewer_pub", None) is None
            or now - getattr(self, "_front_last_frame_t", 0.0) <= 1.0
            or now - self._status_last_send < 0.1
        ):
            return
        if self._status_img is None or self._status_img_text != self.vr_status_text:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            lines = [ln.strip() for ln in self.vr_status_text.split("|") if ln.strip()]
            font = cv2.FONT_HERSHEY_SIMPLEX
            scales = [1.6] + [1.1] * max(len(lines) - 1, 0)
            heights = [cv2.getTextSize(t, font, sc, 3)[0][1] + 30 for t, sc in zip(lines, scales)]
            y = (480 - sum(heights)) // 2
            for text, sc, hgt in zip(lines, scales, heights):
                (tw, _), _ = cv2.getTextSize(text, font, sc, 3)
                y += hgt
                cv2.putText(img, text, ((640 - tw) // 2, y - 15), font, sc, (255, 255, 255), 3, cv2.LINE_AA)
            self._status_img = img
            self._status_img_text = self.vr_status_text
        try:
            self.vr_viewer_pub.send_image(self._status_img)
            self._status_last_send = now
        except Exception:
            pass

    def _send_robot_state(self):
        """Publish full joint state, current hand SE(3) poses, head SO(3) quaternion, and tactile."""
        q, dq, tau = self.bridge.get_full_joint_state(
            actuators_order=self.actuators_order,
            include_torques=self.publish_torques,
        )
        self._state_seq += 1
        now_ts = time.time()
        self.ports["robot_state"].setEnvelope(yarp.Stamp(self._state_seq, now_ts))
        bot = yarp.Bottle()
        for v in (*q, *dq, *tau):
            bot.addFloat64(float(v))
        self.ports["robot_state"].write(bot)

        # Keep solver kinematics updated for FK reporting even when idle
        q_20 = self.bridge.last_q
        dq_20 = self.bridge.last_dq
        self.solver.update_robot_state(q_20, dq_20)

        # Publish r_hand pose in root_link frame: [x, y, z, qw, qx, qy, qz]
        if "rh_current_pose" in self.ports:
            pos_r, rot_r = self.solver.get_current_right_hand_pose()
            quat_r = rotmat_to_quat_wxyz(rot_r)
            pbot = self.ports["rh_current_pose"].prepare()
            pbot.clear()
            for v in (*pos_r, *quat_r):
                pbot.addFloat64(float(v))
            self.ports["rh_current_pose"].setEnvelope(yarp.Stamp(self._state_seq, now_ts))
            self.ports["rh_current_pose"].write()

        # Publish l_hand pose in root_link frame: [x, y, z, qw, qx, qy, qz]
        if "lh_current_pose" in self.ports:
            pos_l, rot_l = self.solver.get_current_left_hand_pose()
            quat_l = rotmat_to_quat_wxyz(rot_l)
            pbot = self.ports["lh_current_pose"].prepare()
            pbot.clear()
            for v in (*pos_l, *quat_l):
                pbot.addFloat64(float(v))
            self.ports["lh_current_pose"].setEnvelope(yarp.Stamp(self._state_seq, now_ts))
            self.ports["lh_current_pose"].write()

        # Publish head SO(3) orientation in root_link frame: [qw, qx, qy, qz]
        if "head_current_quat" in self.ports:
            quat_h = self.solver.get_current_head_quaternion()
            hbot = self.ports["head_current_quat"].prepare()
            hbot.clear()
            for v in quat_h:
                hbot.addFloat64(float(v))
            self.ports["head_current_quat"].setEnvelope(yarp.Stamp(self._state_seq, now_ts))
            self.ports["head_current_quat"].write()

        # Publish tactile state
        if self.use_tactile and "r_hand_touch" in self.ports and "touch_state" in self.ports:
            touch_vec = self.ports["r_hand_touch"].read(False)
            if touch_vec is not None:
                self.last_touch_fingers = np.array(
                    [touch_vec.get(i) for i in range(touch_vec.size())], dtype=np.float64
                )
            if self.tactile_include_palm and "r_hand_palm_touch" in self.ports:
                palm_vec = self.ports["r_hand_palm_touch"].read(False)
                if palm_vec is not None:
                    self.last_touch_palm = np.array(
                        [palm_vec.get(i) for i in range(palm_vec.size())], dtype=np.float64
                    )
            if self.tactile_include_palm and self.last_touch_palm.size > 0:
                self.last_touch = np.concatenate((self.last_touch_fingers, self.last_touch_palm))
            else:
                self.last_touch = self.last_touch_fingers

            tbot = yarp.Bottle()
            for v in self.last_touch:
                tbot.addFloat64(float(v))
            self.ports["touch_state"].write(tbot)

    def interruptModule(self) -> bool:
        if hasattr(self, "ports"):
            for p in self.ports.values():
                try:
                    p.interrupt()
                except Exception:
                    pass
        if hasattr(self, "cmd_port"):
            try:
                self.cmd_port.interrupt()
            except Exception:
                pass
        return True

    def close(self) -> bool:
        if getattr(self, "_closed", False):
            return True
        self._closed = True

        tmp_arms_file = Path("/tmp/icub_teleop_arms")
        if tmp_arms_file.exists():
            try:
                tmp_arms_file.unlink()
            except Exception:
                pass

        print("\n[Cleanup] Returning robot to Home and releasing YARP resources...")
        if hasattr(self, "bridge") and self.bridge is not None and self.bridge.is_connected:
            try:
                self.bridge.move_to_posture(self.q_home_20, duration=2.5)
            except Exception:
                pass

        if hasattr(self, "cmd_port"):
            try:
                self.cmd_port.close()
            except Exception:
                pass

        if hasattr(self, "ports"):
            for p in self.ports.values():
                try:
                    p.close()
                except Exception:
                    pass

        if hasattr(self, "bridge") and self.bridge is not None:
            self.bridge.close()

        for pub_name in ("vr_pub", "vr_viewer_pub"):
            pub = getattr(self, pub_name, None)
            if pub is not None:
                pub.stop()

        cv2.destroyAllWindows()
        return True


if __name__ == "__main__":
    module = None
    try:
        yarp.Network.init()
        if not yarp.Network.checkNetwork():
            print("✗ ERROR: YARP network is not available! Start `yarpserver --write` first.")
            sys.exit(1)

        module = TeleopModuleSM()
        rf = yarp.ResourceFinder()
        rf.configure(sys.argv)
        module.runModule(rf)
    except KeyboardInterrupt:
        print("\n✓ Module stopped by user (Ctrl+C)")
    finally:
        if module is not None:
            module.close()
        yarp.Network.fini()
