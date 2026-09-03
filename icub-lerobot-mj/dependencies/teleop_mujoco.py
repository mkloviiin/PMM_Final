#!/usr/bin/env python3
"""
MuJoCo-Only Teleoperation for iCub
====================================

Instead the arm IK is computed via iKinIpOptMin (mujoco_ik.py) and
gaze IK via MuJoCo Jacobians; physics are stepped by MuJoCo position actuators.

Supports:
  - Mouse control: Ctrl + Right-click to drag mocap targets in the viewer
  - VR control:    Meta Quest controllers via ZMQ (optional, --vr flag)
  - Keyboard:      Hand open/close, go home, reset scene, recording

Usage:
    python3 teleop_mujoco.py                          # Mouse only
    python3 teleop_mujoco.py --quest-ip 192.168.1.50       # Same as above (--vr implied)
    python3 teleop_mujoco.py --record --repo-id demo  # With recording

Keyboard Shortcuts:
    F       Toggle teleop on/off
    1 / 2   Open / Close right hand
    4 / 5   Open / Close left hand
    0       Go home (minimum-jerk trajectory)
    R       Reset scenario (random cube position)
    SPACE   Start / Stop recording an episode
    C       Recalibrate VR controllers
    ESC     Quit
"""

import sys
import time
import argparse
import threading
import json
import faulthandler
import numpy as np
import yaml
import mujoco
import mujoco.viewer
import glfw
from pathlib import Path

# Enable faulthandler to get a traceback on segfault
faulthandler.enable()

# Optional imports ----------------------------------------------------------
try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
    _CV2_IMPORT_ERROR = None
except ImportError as e:
    CV2_AVAILABLE = False
    _CV2_IMPORT_ERROR = e

try:
    from dependencies.zmq_image_publisher import ZMQCompressedImageTransmitter
    BEAVR_IMG_AVAILABLE = True
    _BEAVR_IMG_IMPORT_ERROR = None
except ImportError as e:
    BEAVR_IMG_AVAILABLE = False
    _BEAVR_IMG_IMPORT_ERROR = e

try:
    from pyquaternion import Quaternion as PyQuaternion
    PYQUAT_AVAILABLE = True
except ImportError:
    PYQUAT_AVAILABLE = False

# Local imports -------------------------------------------------------------
from dependencies.mujoco_ik import IKinArmSolver, GazeIKSolver

try:
    from dependencies.recorder_mujoco import MuJoCoRecorder
    RECORDER_AVAILABLE = True
except ImportError:
    RECORDER_AVAILABLE = False

# ========================== JOINT DEFINITIONS ==============================

# Right arm IK joints (includes torso when right is primary)
RIGHT_ARM_JOINTS = [
    "torso_pitch", "torso_roll", "torso_yaw",
    "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw",
    "r_elbow", "r_wrist_prosup", "r_wrist_pitch", "r_wrist_yaw",
]

# Left arm IK joints — two variants depending on torso ownership
LEFT_ARM_JOINTS_WITH_TORSO = [
    "torso_pitch", "torso_roll", "torso_yaw",
    "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw",
    "l_elbow", "l_wrist_prosup", "l_wrist_pitch", "l_wrist_yaw",
]
LEFT_ARM_JOINTS_NO_TORSO = [
    "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw",
    "l_elbow", "l_wrist_prosup", "l_wrist_pitch", "l_wrist_yaw",
]

NECK_JOINTS = ["neck_pitch", "neck_roll", "neck_yaw"]

# Hand actuator names (order matches the actuator block in icub_fixed.xml)
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

# Hand open/close ctrl values (radians)
# open  (deg):  [  0, 75,  0,  0,  0,   0,  0,   0,   0]
# close (deg):  [  0, 60, 45, 90, 80, 100, 80, 100,   0]
HAND_OPEN  = np.array([0.000, 1.309, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000])
HAND_CLOSE = np.array([0.000, 1.047, 0.785, 1.571, 1.396, 1.745, 1.396, 1.745, 0.000])

# VR ZMQ ports (same as vr_controller_sm.py)
VR_HEAD_PORT       = 8115
VR_HAND_RIGHT_PORT = 8117
VR_BUTTONS_PORT    = 8119
VR_HAND_LEFT_PORT  = 8121


# ===========================================================================
#                            MAIN TELEOP CLASS
# ===========================================================================

class MuJoCoTeleop:
    """
    All-in-one MuJoCo teleoperation controller for iCub.

    States:
        idle           – Waiting for commands, no IK running
        teleop_active  – IK tracking mocap targets every control step
        going_home     – Minimum-jerk trajectory back to home keyframe
    """

    def __init__(self, model_path, vr_enabled=False,
                 record_enabled=False, repo_id="icub_mujoco",
                 control_arms="both", primary_arm="right",
                 config_path=None, vr_ip=None, scene_objects=None):
        # ----- Load config.yaml (if provided) -----
        cfg = {}
        if config_path is not None:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            print(f"  [Config] Loaded {config_path}")

        # ----- Simulation timing -----
        self.sim_dt = cfg.get("sim_dt", 0.005)
        self.control_dt = cfg.get("control_dt", 0.02)  # 50 Hz control loop
        self.frame_skip = int(self.control_dt / self.sim_dt)

        # ----- Load MuJoCo model / data -----
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.sim_dt

        # ----- Home keyframe -----
        try:
            self.key_id = self.model.key("home").id
            self.home_ctrl = self.model.key("home").ctrl.copy()
            self.home_qpos = self.model.key("home").qpos.copy()
            self.has_home_keyframe = True
        except KeyError:
            self.key_id = None
            self.home_ctrl = self.data.ctrl.copy()
            self.home_qpos = self.model.qpos0.copy()
            self.has_home_keyframe = False
            print("  [Warn] Keyframe 'home' not found. Using current initial state as home.")

        self._cube_spawn = cfg.get("cube_spawn", {"x": [0.4, 0.5], "y": [-0.15, 0.15], "z": 0.725})
        # Modular scene objects from scenes.yaml (overrides _cube_spawn when present)
        self._scene_objects: list = scene_objects or []
        # Modular joint resets from scenes.yaml (hinge/slide joints, e.g. door)
        self._scene_joints: list = []

        self._home_pose_overrides: dict[str, float] = {}
        for joint_name, deg_value in (cfg.get("home_pose", {}) or {}).items():
            try:
                self._home_pose_overrides[joint_name] = np.deg2rad(float(deg_value))
            except Exception:
                continue

        self._apply_home_pose_to_arrays(self.home_qpos, self.home_ctrl)

        # ----- Arm configuration -----
        # CLI args override config.yaml values
        self.control_arms = control_arms if control_arms != "both" else cfg.get("control_arms", control_arms)
        self.primary_arm = primary_arm if primary_arm != "right" else cfg.get("primary_arm", primary_arm)
        if self.primary_arm in ("right_arm", "r", "rh"):
            self.primary_arm = "right"
        elif self.primary_arm in ("left_arm", "l", "lh"):
            self.primary_arm = "left"

        # ----- IK solver -----
        ik_damping = cfg.get("ik_damping", 1e-4)
        ik_dt = cfg.get("ik_integration_dt", 0.1)
        ik_maxv = cfg.get("ik_max_angvel", 2.0)
        # Number of control steps per IK recomputation (like total_num_steps
        # in cartesian_controller.py).  IK is solved once, then min-jerk
        # interpolation drives ctrl toward the goal over this many steps.
        self._ik_recompute_steps = cfg.get("ik_recompute_steps", 60)

        # Waist (root body) pose — needed for world ↔ iKin frame transform
        if self.has_home_keyframe:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
            self._apply_home_pose_to_arrays(self.data.qpos, self.data.ctrl)
        else:
            # No keyframe: start from qpos0 then apply home_pose YAML overrides
            self.data.qpos[:] = self.model.qpos0
            self.data.qvel[:] = 0
            self._apply_home_pose_to_arrays(self.data.qpos, self.data.ctrl)
        mujoco.mj_forward(self.model, self.data)
        waist_id = self.model.body("icub").id
        waist_pos = self.data.xpos[waist_id].copy()
        waist_mat = self.data.xmat[waist_id].reshape(3, 3).copy()

        # Arm IK via iKinIpOptMin (one solver per arm)
        if self.control_arms in ("both", "right"):
            rh_joints = (RIGHT_ARM_JOINTS if self.primary_arm == "right"
                         else RIGHT_ARM_JOINTS[3:])
            self._rh_solver = IKinArmSolver(
                side="right", joints_to_control=rh_joints,
                real_robot_limits=True, limit_torso_pitch=True,
                waist_pos=waist_pos, waist_rot_mat=waist_mat)
        else:
            self._rh_solver = None

        if self.control_arms in ("both", "left"):
            lh_joints = (LEFT_ARM_JOINTS_WITH_TORSO
                         if self.primary_arm == "left"
                         else LEFT_ARM_JOINTS_NO_TORSO)
            self._lh_solver = IKinArmSolver(
                side="left", joints_to_control=lh_joints,
                real_robot_limits=True, limit_torso_pitch=True,
                waist_pos=waist_pos, waist_rot_mat=waist_mat)
        else:
            self._lh_solver = None

        # Gaze IK (differential, for neck joints)
        self.gaze_solver = GazeIKSolver(
            self.model, self.data,
            damping=ik_damping, integration_dt=ik_dt, max_angvel=ik_maxv)

        # ----- End-effector sites -----
        self.rh_site_id = self.model.site("r_hand_dh_frame_site").id
        self.lh_site_id = self.model.site("l_hand_dh_frame_site").id

        # ----- Mocap bodies (targets) -----
        self.rh_mocap_id = self.model.body("r_hand_target").mocapid[0]
        self.lh_mocap_id = self.model.body("l_hand_target").mocapid[0]
        self.gaze_mocap_id = self.model.body("com_target").mocapid[0]

        # ----- Head body for gaze -----
        self.head_body_name = "head"

        # ----- Hand actuator IDs -----
        self.rh_hand_act_ids = np.array(
            [self.model.actuator(n).id for n in RIGHT_HAND_ACTUATORS])
        self.lh_hand_act_ids = np.array(
            [self.model.actuator(n).id for n in LEFT_HAND_ACTUATORS])

        # ----- State machine -----
        self.state = "idle"
        self.running = True

        # Pending commands from key_callback (executed in main loop)
        self._pending_cmds = []
        self._pending_cmds_lock = threading.Lock()

        # Optional CLI controls (stdin): 1=REC, 2=STOP, 3=EXIT
        self.cli_controls_enabled = sys.stdin.isatty()
        self._cli_thread = None

        # IK interpolation state  (cartesian_controller.py pattern)
        self._ik_step = 0          # current step in interpolation window
        self._rh_ctrl_start = None  # start of interpolation (right arm)
        self._rh_ctrl_goal  = None  # goal of interpolation (right arm)
        self._lh_ctrl_start = None
        self._lh_ctrl_goal  = None

        # Hand interpolation state (gradual open/close)
        self._hand_total_steps = cfg.get("hand_steps", 150)  # ~1.5s at 10ms ctrl_dt
        self._rh_hand_start = None
        self._rh_hand_goal  = None
        self._rh_hand_step  = 0
        self._rh_hand_active = False
        self.rh_grip_state = 1.0  # 0.0=open, 0.5=close, 1.0=stop/idle
        self._lh_hand_start = None
        self._lh_hand_goal  = None
        self._lh_hand_step  = 0
        self._lh_hand_active = False
        self.lh_grip_state = 1.0  # 0.0=open, 0.5=close, 1.0=stop/idle

        # Go-home trajectory state
        self._home_start_ctrl = None
        self._home_target_ctrl = None
        self._home_step = 0
        self._home_total_steps = cfg.get("go_home_steps", 120)

        # ----- Image streaming (head_cam → 10505, viewer → 15001) -----
        self.img_stream_enabled = CV2_AVAILABLE and BEAVR_IMG_AVAILABLE
        self._head_cam_pub = None
        self._viewer_pub = None
        self._head_renderer = None
        self._viewer_renderer = None
        if self.img_stream_enabled:
            try:
                stream_quality = cfg.get("stream_jpeg_quality", 90)
                self._head_cam_pub = ZMQCompressedImageTransmitter(host="*", port=10505, quality=stream_quality)
                self._viewer_pub = ZMQCompressedImageTransmitter(host="*", port=15001, quality=stream_quality)
                self._head_renderer = mujoco.Renderer(self.model, height=480, width=640)
                self._viewer_renderer = mujoco.Renderer(self.model, height=480, width=640)
                print(f"  [Stream] head_cam → port 10505, viewer → port 15001 (jpeg quality={stream_quality})")
            except Exception as e:
                print(f"  [Stream] Failed to init image streaming: {e}")
                self.img_stream_enabled = False
        else:
            if not CV2_AVAILABLE:
                print(f"  [Stream] cv2 not available — image streaming disabled ({_CV2_IMPORT_ERROR!r})")
            if not BEAVR_IMG_AVAILABLE:
                print(f"  [Stream] ZMQCompressedImageTransmitter not available — streaming disabled ({_BEAVR_IMG_IMPORT_ERROR!r})")
        self._last_stream_time = 0.0

        # ----- VR -----
        self.vr_ip = vr_ip if vr_ip else cfg.get("vr_ip", None)
        self.vr_enabled = vr_enabled and ZMQ_AVAILABLE and PYQUAT_AVAILABLE
        
        # --- Smoothing Config ---
        self._target_filter_alpha = 0.15
        self._prev_target_pos = {"right": None, "left": None}
        
        # --- Cache Joint IDs for IK ---
        self._rh_act_ids = []
        self._rh_qpos_idxs = []
        if self._rh_solver:
            self._rh_act_ids = np.array([self.model.actuator(n).id for n in self._rh_solver.joints_to_control])
            # For qpos, we need joint address.
            # NOTE: qpos address != joint ID for ball joints, but here we only use hinge/slide so it's likely 1:1 map
            # Safer to use jnt_qposadr
            self._rh_qpos_idxs = [self.model.jnt_qposadr[self.model.joint(n).id] for n in self._rh_solver.joints_to_control]

        self._lh_act_ids = []
        self._lh_qpos_idxs = []
        if self._lh_solver:
            self._lh_act_ids = np.array([self.model.actuator(n).id for n in self._lh_solver.joints_to_control])
            self._lh_qpos_idxs = [self.model.jnt_qposadr[self.model.joint(n).id] for n in self._lh_solver.joints_to_control]

        self.latest_rh_data = None
        self.latest_lh_data = None
        self.latest_buttons = None
        self.calibrated_origin_rh = None
        self.calibrated_origin_lh = None
        self.freeze_rh = False
        self.freeze_lh = False
        self.freeze_head = False
        self.gaze_distance = 1.5
        self._last_btn_recv_time = 0.0
        if self.vr_enabled:
            self._init_vr()

        # ----- Recording -----
        self.record_enabled = record_enabled and RECORDER_AVAILABLE
        self.recording_active = False
        self.recorder = None
        self.episode_count = 0
        if self.record_enabled:
            self.recorder = MuJoCoRecorder(
                self.model, self.data, fps=30, repo_id=repo_id,
                camera_name="head_cam")

        # ----- Print summary -----
        print("=" * 55)
        print("  MuJoCo iCub Teleop — Initialised")
        print(f"  Config : {config_path or 'defaults'}")
        print(f"  Model  : {model_path}")
        print(f"  Arms   : {self.control_arms}  (primary={self.primary_arm})")
        print(f"  IK     : iKinIpOptMin  recompute_steps={self._ik_recompute_steps}  gaze_dt={ik_dt}")
        print(f"  VR     : {'ON' if self.vr_enabled else 'OFF'}")
        if self.vr_enabled:
            vr_mode = "connect" if self.vr_ip else "bind"
            vr_target = self.vr_ip if self.vr_ip else "*"
            print(f"  VR ZMQ : mode={vr_mode} target={vr_target}")
        print(f"  Record : {'ON' if self.record_enabled else 'OFF'}")
        print(f"  CLI    : {'ON' if self.cli_controls_enabled else 'OFF'}")
        print("=" * 55)

    def _apply_home_pose_to_arrays(self, qpos_array, ctrl_array) -> None:
        if not self._home_pose_overrides:
            return

        for joint_name, rad_value in self._home_pose_overrides.items():
            try:
                joint_id = self.model.joint(joint_name).id
                qpos_adr = self.model.jnt_qposadr[joint_id]
                qpos_array[qpos_adr] = rad_value
            except Exception:
                pass

            try:
                actuator_id = self.model.actuator(joint_name).id
                ctrl_array[actuator_id] = rad_value
            except Exception:
                pass

    def _enqueue_cmd(self, cmd):
        """Thread-safe enqueue for commands processed in main loop."""
        with self._pending_cmds_lock:
            self._pending_cmds.append(cmd)

    def _start_cli_listener(self):
        """Start non-blocking stdin listener for REC/STOP/EXIT commands."""
        if not self.cli_controls_enabled or self._cli_thread is not None:
            return
        self._cli_thread = threading.Thread(
            target=self._cli_command_listener,
            daemon=True,
        )
        self._cli_thread.start()

    def _cli_command_listener(self):
        """Background stdin listener.

        Commands:
            1 | rec   | start  -> start recording
            2 | stop             -> stop recording
            3 | exit  | quit      -> exit application
        """
        print("\n[CLI] Commands: 1=REC, 2=STOP, 3=EXIT")
        while self.running:
            try:
                cmd = input(">> COMMAND (1=REC, 2=STOP, 3=EXIT): ").strip().lower()
            except EOFError:
                break
            except Exception:
                continue

            if not cmd:
                continue

            if cmd in ("1", "rec", "start"):
                self._enqueue_cmd("start_recording")
            elif cmd in ("2", "stop"):
                self._enqueue_cmd("stop_recording")
            elif cmd in ("3", "exit", "quit", "q"):
                self._enqueue_cmd("exit_program")
                break
            elif cmd in ("f", "teleop"):
                self._enqueue_cmd("toggle_teleop")
            elif cmd in ("r", "reset"):
                self._enqueue_cmd("reset_scenario")
            elif cmd in ("0", "home"):
                self._enqueue_cmd("go_home")
            else:
                print("[CLI] Unknown command. Use: 1=REC, 2=STOP, 3=EXIT")

    # ========================= VR SETUP =================================

    def _init_vr(self):
        """Initialise ZMQ sockets for Meta Quest controller input."""
        self._vr_sockets = {}
        for name, port in [("head", VR_HEAD_PORT),
                           ("rh", VR_HAND_RIGHT_PORT),
                           ("lh", VR_HAND_LEFT_PORT),
                           ("btn", VR_BUTTONS_PORT)]:
            ctx = zmq.Context()
            sock = ctx.socket(zmq.PULL)
            endpoint = f"tcp://*:{port}"
            sock.bind(endpoint)
            if self.vr_ip:
                print(f"  [VR] Socket '{name}' bound to {endpoint} (quest_ip={self.vr_ip})")
            else:
                print(f"  [VR] Socket '{name}' bound to {endpoint}")
            sock.setsockopt(zmq.LINGER, 0)
            self._vr_sockets[name] = (ctx, sock)

        # Start receiver thread
        self._vr_thread = threading.Thread(target=self._vr_receive_loop,
                                           daemon=True)
        self._vr_thread.start()

    def _vr_receive_loop(self):
        """Background thread: drain all VR sockets continuously."""
        while self.running:
            for name in ("rh", "lh", "btn"):
                _, sock = self._vr_sockets[name]
                try:
                    while True:
                        raw = sock.recv_string(flags=zmq.NOBLOCK)
                        if name == "rh":
                            self.latest_rh_data = self._parse_controller(raw)
                        elif name == "lh":
                            self.latest_lh_data = self._parse_controller(raw)
                        elif name == "btn":
                            self.latest_buttons = self._parse_buttons(raw)
                            self._last_btn_recv_time = time.time()
                except zmq.Again:
                    pass
            time.sleep(0.001)

    # ---------------------- VR data parsers --------------------------------

    @staticmethod
    def _parse_controller(data_str):
        """Parse hand controller data from Meta Quest (JSON or CSV)."""
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
                    quat = PyQuaternion(w=float(j["qw"]), x=float(j["qx"]),
                                        y=float(j["qy"]), z=float(j["qz"]))
                    return {"pos": pos, "quat": quat}
                if "w" in j and "x" in j and "y" in j and "z" in j:
                    quat = PyQuaternion(w=float(j["w"]), x=float(j["x"]),
                                        y=float(j["y"]), z=float(j["z"]))
                    return {"pos": pos, "quat": quat}
                return {"pos": pos}
            else:
                parts = data_str.split(",")
                if len(parts) >= 3:
                    pos = np.array([float(p) for p in parts[:3]])
                    if len(parts) >= 7:
                        quat = PyQuaternion(w=float(parts[6]),
                                            x=float(parts[3]),
                                            y=float(parts[4]),
                                            z=float(parts[5]))
                        return {"pos": pos, "quat": quat}
                    return {"pos": pos}
        except Exception:
            pass
        return None

    @staticmethod
    def _parse_buttons(data_str):
        """Parse button state data from Meta Quest."""
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

    # ---------------------- VR data processing -----------------------------

    def _vr_hand_to_target(self, hand_data, calib_origin, home_pos,
                           roll_offset=0, side="right"):
        """Convert VR controller data to robot-frame target with Smoothing."""
        if hand_data is None:
            return None, None, calib_origin
        vr_pos = hand_data["pos"]
        if np.linalg.norm(vr_pos) < 0.001:
            return None, None, calib_origin
        if calib_origin is None:
            calib_origin = vr_pos.copy()
        
        delta_vr = vr_pos - calib_origin
        # VR → robot frame mapping (inverted X)
        # Robot X (Fwd) = VR Z (Back/Fwd)
        # Robot Y (Left) = -VR X (Right)  <-- FIX: Inverted sign based on user feedback
        # Robot Z (Up)   = VR Y (Up)
        delta_robot = np.array([delta_vr[2], -delta_vr[0], delta_vr[1]])
        raw_target_pos = home_pos + delta_robot

        # --- Low Pass Filter on Position ---
        if self._prev_target_pos[side] is None:
             self._prev_target_pos[side] = raw_target_pos
        
        # EMA Filter: filtered = alpha * raw + (1-alpha) * prev
        filtered_pos = (self._target_filter_alpha * raw_target_pos + 
                        (1.0 - self._target_filter_alpha) * self._prev_target_pos[side])
        self._prev_target_pos[side] = filtered_pos
        
        target_pos = filtered_pos

        if "quat" in hand_data:
            q_vr = hand_data["quat"]
            # Standard mapping: -Z_vr -> X_rob (Fwd), -X_vr -> Y_rob (Right), Y_vr -> Z_rob (Up)
            q_remapped = PyQuaternion(w=q_vr.w, x=-q_vr.z, y=-q_vr.x, z=q_vr.y)
            
            # Rotate 180 degrees around X axis as requested
            q_fix_x = PyQuaternion(axis=[1, 0, 0], degrees=180)
            q_final = q_fix_x * q_remapped

            if roll_offset != 0:
                q_fix = PyQuaternion(axis=[1, 0, 0], degrees=roll_offset)
                q_final = q_final * q_fix
            # Additional 45° Z-axis rotation to align VR wrist orientation with robot hand
            #q_fix_z = PyQuaternion(axis=[0, 1, 0], degrees=-45)
            #q_final = q_final * q_fix_z
            target_quat = np.array([q_final.w, q_final.x,
                                    q_final.y, q_final.z])
        else:
            target_quat = np.array([1.0, 0.0, 0.0, 0.0])

        return target_pos, target_quat, calib_origin

    def _process_vr_hands(self):
        """Map VR hand controller data → mocap target positions."""
        ROBOT_HOME_RH = np.array([0.4, -0.2, 0.7])
        ROBOT_HOME_LH = np.array([0.4,  0.2, 0.7])

        # Right hand
        if not self.freeze_rh and self.latest_rh_data is not None:
            pos, quat, self.calibrated_origin_rh = self._vr_hand_to_target(
                self.latest_rh_data, self.calibrated_origin_rh,
                ROBOT_HOME_RH, roll_offset=90, side="right")
            if pos is not None:
                pos[0] = np.clip(pos[0], 0.1, 0.8)
                pos[1] = np.clip(pos[1], -0.4, 0.1)
                pos[2] = np.clip(pos[2], 0.4, 1.0)
                self.data.mocap_pos[self.rh_mocap_id] = pos
                self.data.mocap_quat[self.rh_mocap_id] = quat

        # Left hand
        if not self.freeze_lh and self.latest_lh_data is not None:
            pos, quat, self.calibrated_origin_lh = self._vr_hand_to_target(
                self.latest_lh_data, self.calibrated_origin_lh,
                ROBOT_HOME_LH, roll_offset=90, side="left")
            if pos is not None:
                pos[0] = np.clip(pos[0], 0.1, 0.8)
                pos[1] = np.clip(pos[1], -0.1, 0.4)
                pos[2] = np.clip(pos[2], 0.4, 1.0)
                self.data.mocap_pos[self.lh_mocap_id] = pos
                self.data.mocap_quat[self.lh_mocap_id] = quat

    def _solve_ik(self):
        """Solve IK and SMOOTHLY update joint controls."""
        
        # 1. Update Gaze (Differential IK)
        # Check if self.config exists and has use_gaze attribute
        if hasattr(self, 'config') and hasattr(self.config, 'use_gaze') and self.config.use_gaze:
            gaze_target = self.data.mocap_pos[self.gaze_mocap_id]
            gaze_ctrl_delta = self.gaze_solver.solve(
                self.head_body_name, gaze_target, self.neck_joint_names)
            for jname, val in gaze_ctrl_delta.items():
                act_id = self.model.actuator(jname).id
                self.data.ctrl[act_id] = val

        # Always lock neck_roll to 0 to prevent head from tilting sideways
        try:
            neck_roll_act_id = self.model.actuator("neck_roll").id
            self.data.ctrl[neck_roll_act_id] = 0.0
        except Exception:
            pass

        # 2. Update Arms (iKin)
        q_current = self.data.qpos.copy() # Current simulation state

        # Base interpolation factor
        base_alpha = 0.2
        deadband_m = 0.05  # 5mm deadband for settling

        # Helper to get current torso values for the localized solver
        # (Needed if the solver does NOT control the torso, so it knows where the shoulder is)
        current_torso_values = {}
        for t_name in ["torso_pitch", "torso_roll", "torso_yaw"]:
            # Find qpos address for this joint
            # We can cache this, but lookup is fast enough for 3 joints
            try:
                adr = self.model.jnt_qposadr[self.model.joint(t_name).id]
                current_torso_values[t_name] = q_current[adr]
            except KeyError:
                pass

        if self._rh_solver:
            rh_target_pos = self.data.mocap_pos[self.rh_mocap_id]
            rh_target_quat = self.data.mocap_quat[self.rh_mocap_id]
            
            q_arm = [q_current[i] for i in self._rh_qpos_idxs]
            
            # If right arm is secondary, it needs torso hint from left
            # But usually right is primary. We pass it anyway if the solver accepts it.
            # In mujoco_ik.py, solve() takes torso_values.
            
            sol_r, solved_r = self._rh_solver.solve(
                rh_target_pos, rh_target_quat, 
                np.array(q_arm),
                torso_values=current_torso_values 
            )
            
            # --- Deadband & Smooth Interpolation ---
            curr_ee_pos = self.data.site_xpos[self.rh_site_id]
            dist_error = np.linalg.norm(rh_target_pos - curr_ee_pos)
            
            if dist_error < deadband_m:
                 interp_alpha = 0.01 
            else:
                 interp_alpha = base_alpha

            curr_ctrl = self.data.ctrl[self._rh_act_ids]
            goal_ctrl = sol_r
            next_ctrl = curr_ctrl + interp_alpha * (goal_ctrl - curr_ctrl)
            
            if not np.any(np.isnan(next_ctrl)):
                self.data.ctrl[self._rh_act_ids] = next_ctrl

        if self._lh_solver:
            lh_target_pos = self.data.mocap_pos[self.lh_mocap_id]
            lh_target_quat = self.data.mocap_quat[self.lh_mocap_id]
            
            q_arm = [q_current[i] for i in self._lh_qpos_idxs]
            
            # CRITICAL FIX: Pass current torso state to Left Arm IK
            # Since Right Arm (Primary) moves the torso, Left Arm needs to know!
            sol_l, solved_l = self._lh_solver.solve(
                lh_target_pos, lh_target_quat,
                np.array(q_arm),
                torso_values=current_torso_values
            )
            
            curr_ee_pos = self.data.site_xpos[self.lh_site_id]
            dist_error = np.linalg.norm(lh_target_pos - curr_ee_pos)
            
            if dist_error < deadband_m:
                 interp_alpha = 0.01
            else:
                 interp_alpha = base_alpha
            
            curr_ctrl = self.data.ctrl[self._lh_act_ids]
            goal_ctrl = sol_l
            next_ctrl = curr_ctrl + interp_alpha * (goal_ctrl - curr_ctrl)

            if not np.any(np.isnan(next_ctrl)):
                self.data.ctrl[self._lh_act_ids] = next_ctrl

    def _process_vr_head(self):
        """Process VR head rotation → gaze mocap target."""
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
                    q_vr = PyQuaternion(w=float(j["w"]), x=float(j["x"]),
                                        y=float(j["y"]), z=float(j["z"]))
            else:
                parts = data.split(",")
                if len(parts) == 4:
                    q_vr = PyQuaternion(w=float(parts[3]), x=float(parts[0]),
                                        y=float(parts[1]), z=float(parts[2]))
        except Exception:
            return

        if q_vr is not None:
            # Yaw offset correction: when looking straight ahead in real life,
            # gaze was deviating ~20° to the right in the robot frame.
            # Rotating the VR quaternion by +20° around Y (VR up) compensates.
            q_fix_yaw = PyQuaternion(axis=[0, 1, 0], degrees=-20)
            q_vr = q_fix_yaw * q_vr

            v_fwd_vr = q_vr.rotate(np.array([0.0, 0.0, -1.0]))
            
            # Gaze forward vector mapping
            # Robot X (Fwd)  = -VR Z (Fwd)
            # Robot Y (Left) =  VR X (Right) (flipped sign to fix mirror)
            # Robot Z (Up)   = -VR Y (Down?) 
            v_fwd_robot = np.array([-v_fwd_vr[2], v_fwd_vr[0], -v_fwd_vr[1]])
            norm = np.linalg.norm(v_fwd_robot)
            if norm > 0:
                v_fwd_robot /= norm

            head_bid = self.model.body(self.head_body_name).id
            head_pos = self.data.xpos[head_bid]
            target = head_pos + v_fwd_robot * self.gaze_distance
            self.data.mocap_pos[self.gaze_mocap_id] = target

    def _process_button_commands(self):
        """React to VR button states."""
        b = self.latest_buttons
        if b is None or not isinstance(b, dict):
            return

        # Hands
        if b.get("INDEX_RIGHT", 0) > 0.8:
            self._set_hand("right", "close")
        elif b.get("HAND_RIGHT", 0) > 0.8:
            self._set_hand("right", "open")
        if b.get("INDEX_LEFT", 0) > 0.8:
            self._set_hand("left", "close")
        elif b.get("HAND_LEFT", 0) > 0.8:
            self._set_hand("left", "open")

        # Reset
        if b.get("BTN_TWO_LEFT", 0) > 0.5 or b.get("Y", 0) > 0.8:
            self._reset_scenario()

        # Recording A → start, B → stop
        if b.get("BTN_ONE_RIGHT", 0) > 0.5 or b.get("A", 0) > 0.8:
            if not getattr(self, "_rec_a_held", False):
                self._rec_a_held = True
                if not self.recording_active:
                    self._toggle_recording()
        else:
            self._rec_a_held = False

        if b.get("BTN_TWO_RIGHT", 0) > 0.5 or b.get("B", 0) > 0.8:
            if not getattr(self, "_rec_b_held", False):
                self._rec_b_held = True
                if self.recording_active:
                    self._toggle_recording()
        else:
            self._rec_b_held = False

        # Freeze while squeezing triggers
        any_active = any(b.get(k, 0) > 0.8
                         for k in ("INDEX_RIGHT", "HAND_RIGHT",
                                   "INDEX_LEFT", "HAND_LEFT"))
        self.freeze_rh = any_active
        self.freeze_lh = any_active
        self.freeze_head = any_active

    # ========================= HAND CONTROL ================================

    def _set_hand(self, side, action):
        """Start a gradual min-jerk open/close for the given hand."""
        goal = HAND_OPEN if action == "open" else HAND_CLOSE
        act_ids = self.rh_hand_act_ids if side == "right" else self.lh_hand_act_ids
        start = np.array([self.data.ctrl[aid] for aid in act_ids])

        if side == "right":
            self._rh_hand_start = start
            self._rh_hand_goal = goal.copy()
            self._rh_hand_step = 0
            self._rh_hand_active = True
            self.rh_grip_state = 0.0 if action == "open" else 0.5
        else:
            self._lh_hand_start = start
            self._lh_hand_goal = goal.copy()
            self._lh_hand_step = 0
            self._lh_hand_active = True
            self.lh_grip_state = 0.0 if action == "open" else 0.5
        print(f"[Hand] {side} {action}")

    def _interpolate_hands(self):
        """Advance hand interpolation one step (called every control cycle)."""
        N = self._hand_total_steps
        for side in ("right", "left"):
            if side == "right":
                if not self._rh_hand_active:
                    continue
                step = self._rh_hand_step
                start, goal = self._rh_hand_start, self._rh_hand_goal
                act_ids = self.rh_hand_act_ids
            else:
                if not self._lh_hand_active:
                    continue
                step = self._lh_hand_step
                start, goal = self._lh_hand_start, self._lh_hand_goal
                act_ids = self.lh_hand_act_ids

            # min-jerk scalar
            if step >= N:
                s = 1.0
            else:
                tau = step / max(1, N)
                s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5

            q_t = start + (goal - start) * s
            for i, aid in enumerate(act_ids):
                self.data.ctrl[aid] = q_t[i]

            # Advance step
            if side == "right":
                self._rh_hand_step += 1
                if self._rh_hand_step > N:
                    self._rh_hand_active = False
            else:
                self._lh_hand_step += 1
                if self._lh_hand_step > N:
                    self._lh_hand_active = False

    # ========================= SCENE RESET =================================

    def _reset_scenario(self):
        """Randomize all declared scene objects (driven by scenes.yaml).

        Falls back to legacy _cube_spawn config if no scene_objects are declared
        but a blue-cube body exists in the model.
        """
        objects = getattr(self, "_scene_objects", [])

        # Legacy fallback: only when BOTH scene_objects AND scene_joints are absent.
        # Joint-only scenes (e.g. door) must not trigger this or they'd try to reset
        # a 'blue-cube' that doesn't exist in the model.
        if not objects and not self._scene_joints:
            spawn = getattr(self, "_cube_spawn", None)
            if spawn:
                objects = [{"body": "blue-cube", "spawn": spawn}]
            else:
                return  # Nothing to reset

        def _sample(val, default=0.0):
            """Uniform random if [min, max], fixed if scalar."""
            if isinstance(val, list) and len(val) == 2:
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

                # Position: only write axes that are specified
                if any(k in spawn for k in ("x", "y", "z")):
                    x = _sample(spawn.get("x"), self.data.qpos[qpos_adr + 0])
                    y = _sample(spawn.get("y"), self.data.qpos[qpos_adr + 1])
                    z = _sample(spawn.get("z"), self.data.qpos[qpos_adr + 2])
                    self.data.qpos[qpos_adr:qpos_adr + 3] = [x, y, z]

                # Orientation
                quat = spawn.get("quat", [1, 0, 0, 0])
                self.data.qpos[qpos_adr + 3:qpos_adr + 7] = quat

                # Zero velocities
                ndof = self.model.jnt_type[self.model.body_jntadr[bid]]
                num_dof = 6 if ndof == 0 else 1  # mjtJoint.mjJNT_FREE = 0
                self.data.qvel[dof_adr:dof_adr + num_dof] = 0

                pos_arr = self.data.qpos[qpos_adr:qpos_adr + 3].copy()
                quat_arr = self.data.qpos[qpos_adr + 3:qpos_adr + 7].copy()
                reset_poses[body_name] = (pos_arr, quat_arr)
                any_reset = True
                print(f"[Reset] {body_name} → ({pos_arr[0]:.2f}, {pos_arr[1]:.2f}, {pos_arr[2]:.2f})")
            except Exception as e:
                print(f"[Reset] Skip '{body_name}': {e}")

        if any_reset:
            mujoco.mj_forward(self.model, self.data)

        # ── Joint reset (hinge/slide joints declared under `joints:` in scenes.yaml) ──
        for jnt_def in self._scene_joints:
            jnt_name = jnt_def.get("name", "")
            target_qpos = float(jnt_def.get("qpos", 0.0))
            try:
                jnt_id = self.model.joint(jnt_name).id
                qpos_adr = self.model.jnt_qposadr[jnt_id]
                dof_adr = self.model.jnt_dofadr[jnt_id]
                self.data.qpos[qpos_adr] = target_qpos
                self.data.qvel[dof_adr] = 0.0
                print(f"[Reset] Joint '{jnt_name}' → {target_qpos:.3f} rad")
            except Exception as e:
                print(f"[Reset] Skip joint '{jnt_name}': {e}")

        if self._scene_joints:
            mujoco.mj_forward(self.model, self.data)

    def set_object_poses(self, poses: dict):
        """Force one or more objects to specific poses.

        Args:
            poses: {body_name: (pos_array, quat_array)} dict.
        """
        for body_name, (pos, quat) in poses.items():
            try:
                body = self.model.body(body_name)
                jnt_adr = body.jntadr[0]
                if jnt_adr == -1:
                    continue
                qpos_adr = self.model.jnt_qposadr[jnt_adr]
                dof_adr = self.model.jnt_dofadr[jnt_adr]

                self.data.qpos[qpos_adr:qpos_adr + 3] = pos
                if quat is not None:
                    self.data.qpos[qpos_adr + 3:qpos_adr + 7] = quat
                else:
                    self.data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]

                ndof = self.model.jnt_type[self.model.body_jntadr[body.id]]
                num_dof = 6 if ndof == 0 else 1
                self.data.qvel[dof_adr:dof_adr + num_dof] = 0
            except Exception as e:
                print(f"[Teleop] Error setting '{body_name}' pose: {e}")

        mujoco.mj_forward(self.model, self.data)

    def set_cube_pose(self, pos, quat):
        """Legacy wrapper — delegates to set_object_poses."""
        self.set_object_poses({"blue-cube": (pos, quat)})
    # ========================= GO HOME =====================================

    def _start_go_home(self):
        """Begin a minimum-jerk trajectory to the home keyframe ctrl."""
        self._home_start_ctrl = self.data.ctrl.copy()
        self._home_target_ctrl = self.home_ctrl.copy()
        self._home_step = 0
        self.state = "going_home"
        print("[State] → going_home")

    def _interpolate_home(self):
        """Advance one step of the min-jerk go-home trajectory."""
        if self._home_step >= self._home_total_steps:
            self.data.ctrl[:] = self._home_target_ctrl
            self.state = "idle"
            print("[State] Home reached → idle")
            return

        tau = self._home_step / self._home_total_steps
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        self.data.ctrl[:] = (self._home_start_ctrl
                             + (self._home_target_ctrl - self._home_start_ctrl) * s)
        self._home_step += 1

    # ========================= TARGET SYNC =================================

    def _snap_targets_to_ee(self):
        """Move mocap targets to current end-effector positions.

        This prevents sudden jumps when teleop is activated — the IK
        starts with zero error and only tracks incremental changes.
        """
        mujoco.mj_forward(self.model, self.data)

        # Right hand
        rh_pos = self.data.site(self.rh_site_id).xpos.copy()
        rh_quat = np.zeros(4)
        mujoco.mju_mat2Quat(rh_quat, self.data.site(self.rh_site_id).xmat)
        self.data.mocap_pos[self.rh_mocap_id] = rh_pos
        self.data.mocap_quat[self.rh_mocap_id] = rh_quat

        # Left hand
        lh_pos = self.data.site(self.lh_site_id).xpos.copy()
        lh_quat = np.zeros(4)
        mujoco.mju_mat2Quat(lh_quat, self.data.site(self.lh_site_id).xmat)
        self.data.mocap_pos[self.lh_mocap_id] = lh_pos
        self.data.mocap_quat[self.lh_mocap_id] = lh_quat

        # Gaze: snap target in front of the current head orientation
        # so teleop activation does not send gaze behind the robot.
        try:
            head_bid = self.model.body(self.head_body_name).id
            head_pos = self.data.xpos[head_bid].copy()
            head_rot = self.data.xmat[head_bid].reshape(3, 3)
            gaze_dir = head_rot[:, 2].copy()  # iCub forward axis (+X) in this model frame
            norm = np.linalg.norm(gaze_dir)
            if norm < 1e-8:
                gaze_dir = np.array([1.0, 0.0, 0.0])
            else:
                gaze_dir /= norm
            gaze_pos = head_pos + gaze_dir * self.gaze_distance
            self.data.mocap_pos[self.gaze_mocap_id] = gaze_pos
        except Exception:
            gaze_pos = self.data.mocap_pos[self.gaze_mocap_id].copy()

        print(f"  [Snap] RH target → {rh_pos}")
        print(f"  [Snap] LH target → {lh_pos}")
        print(f"  [Snap] Gaze kept → {gaze_pos}")

        # Reset arm solvers so first IK call starts fresh
        if self._rh_solver:
            self._rh_solver.reset()
        if self._lh_solver:
            self._lh_solver.reset()
        # Force immediate IK recomputation
        self._ik_step = self._ik_recompute_steps
        self._rh_ctrl_goal = None
        self._lh_ctrl_goal = None

    # ========================= IK ==========================================

    def _solve_ik(self):
        """Arm IK + min-jerk interpolation (same pattern as cartesian_controller.py).

        Every ``_ik_recompute_steps`` control cycles the IK is re-solved for
        the current target.  Between solves, ``data.ctrl`` is driven toward
        the goal via a minimum-jerk trajectory — no exponential blend.
        """
        N = self._ik_recompute_steps

        # --- Recompute IK when interpolation window is done (or first call) ---
        need_recompute = (self._ik_step >= N
                          or (self._rh_solver is not None and self._rh_ctrl_goal is None)
                          or (self._lh_solver is not None and self._lh_ctrl_goal is None))

        if need_recompute:
            torso_ctrl = {
                'torso_pitch': self.data.ctrl[self.model.actuator('torso_pitch').id],
                'torso_roll':  self.data.ctrl[self.model.actuator('torso_roll').id],
                'torso_yaw':   self.data.ctrl[self.model.actuator('torso_yaw').id],
            }

            # ---- Right arm ----
            if self._rh_solver is not None and self.control_arms in ("both", "right"):
                rh_pos = self.data.mocap_pos[self.rh_mocap_id]
                rh_quat = self.data.mocap_quat[self.rh_mocap_id]
                rh_joints = (RIGHT_ARM_JOINTS if self.primary_arm == "right"
                             else RIGHT_ARM_JOINTS[3:])
                # Use data.ctrl (commanded position — deterministic, no physics jitter)
                current_q = np.array([
                    self.data.ctrl[self.model.actuator(j).id] for j in rh_joints])
                goal_q, _ = self._rh_solver.solve(
                    rh_pos, rh_quat, current_q,
                    torso_values=torso_ctrl if self.primary_arm != "right" else None)
                self._rh_ctrl_start = current_q.copy()
                self._rh_ctrl_goal = goal_q.copy()
            if self._lh_solver is not None and self.control_arms in ("both", "left"):
                lh_pos = self.data.mocap_pos[self.lh_mocap_id]
                lh_quat = self.data.mocap_quat[self.lh_mocap_id]
                lh_joints = (LEFT_ARM_JOINTS_WITH_TORSO
                             if self.primary_arm == "left"
                             else LEFT_ARM_JOINTS_NO_TORSO)
                # Use data.ctrl (commanded position — deterministic, no physics jitter)
                current_q = np.array([
                    self.data.ctrl[self.model.actuator(j).id] for j in lh_joints])
                goal_q, _ = self._lh_solver.solve(
                    lh_pos, lh_quat, current_q,
                    torso_values=torso_ctrl if self.primary_arm != "left" else None)
                self._lh_ctrl_start = current_q.copy()
                self._lh_ctrl_goal = goal_q.copy()

            self._ik_step = 0

        # --- Min-jerk interpolation (exactly like go_to in cartesian_controller.py) ---
        if self._ik_step > N * 80 // 100:
            tau = 1.0
        else:
            T = max(1, N * 80 // 100)
            tau = self._ik_step / T
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5  # min-jerk scalar

        # Right arm
        if self._rh_ctrl_goal is not None and self.control_arms in ("both", "right"):
            rh_joints = (RIGHT_ARM_JOINTS if self.primary_arm == "right"
                         else RIGHT_ARM_JOINTS[3:])
            q_t = self._rh_ctrl_start + (self._rh_ctrl_goal - self._rh_ctrl_start) * s
            for i, jname in enumerate(rh_joints):
                act_id = self.model.actuator(jname).id
                jnt_id = self.model.actuator_trnid[act_id, 0]
                self.data.ctrl[act_id] = np.clip(
                    q_t[i], self.model.jnt_range[jnt_id, 0],
                    self.model.jnt_range[jnt_id, 1])

        # Left arm
        if self._lh_ctrl_goal is not None and self.control_arms in ("both", "left"):
            lh_joints = (LEFT_ARM_JOINTS_WITH_TORSO
                         if self.primary_arm == "left"
                         else LEFT_ARM_JOINTS_NO_TORSO)
            q_t = self._lh_ctrl_start + (self._lh_ctrl_goal - self._lh_ctrl_start) * s
            for i, jname in enumerate(lh_joints):
                act_id = self.model.actuator(jname).id
                jnt_id = self.model.actuator_trnid[act_id, 0]
                self.data.ctrl[act_id] = np.clip(
                    q_t[i], self.model.jnt_range[jnt_id, 0],
                    self.model.jnt_range[jnt_id, 1])

        self._ik_step += 1

        # ---- Gaze (head) ----
        gaze_target = self.data.mocap_pos[self.gaze_mocap_id]
        gaze_ctrl = self.gaze_solver.solve(
            self.head_body_name, gaze_target, NECK_JOINTS)
        gaze_ctrl.pop("neck_roll", None)  # Lock head roll
        for jname, val in gaze_ctrl.items():
            self.data.ctrl[self.model.actuator(jname).id] = val

    # ========================= RECORDING ===================================

    def _toggle_recording(self):
        if not self.record_enabled or self.recorder is None:
            print("[Record] Disabled.  Relaunch with --record")
            return
        if self.recording_active:
            self.recording_active = False
            self.recorder.stop_episode()
            self.episode_count += 1
            print(f"[Record] Episode saved.  Total: {self.episode_count}")
        else:
            self.recording_active = True
            self.recorder.start_episode()
            print(f"[Record] ● REC episode {self.episode_count}…")

    def _get_robot_state(self):
        """Read current joint positions for all actuators (38-dim vector)."""
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

    def _get_world_state(self):
        """World state vector: blue cube pose [x, y, z, qw, qx, qy, qz]."""
        if not hasattr(self, "_world_cube_body_id"):
            self._world_cube_body_id = None
            try:
                self._world_cube_body_id = self.model.body("blue-cube").id
            except Exception:
                self._world_cube_body_id = None

        if self._world_cube_body_id is None:
            return np.zeros(7, dtype=np.float32)

        pos = self.data.xpos[self._world_cube_body_id].copy()
        quat = self.data.xquat[self._world_cube_body_id].copy()
        return np.concatenate([pos, quat]).astype(np.float32)

    def _get_action_vector(self):
        """Action = [rh_pos(3), rh_quat(4), rh_gripper(1), lh_pos(3), lh_quat(4), lh_gripper(1), gaze_pos(3)] = 19."""
        rh_pos = self.data.mocap_pos[self.rh_mocap_id]
        rh_quat = self.data.mocap_quat[self.rh_mocap_id]
        lh_pos = self.data.mocap_pos[self.lh_mocap_id]
        lh_quat = self.data.mocap_quat[self.lh_mocap_id]
        gaze_pos = self.data.mocap_pos[self.gaze_mocap_id]
        return np.concatenate([
            rh_pos,
            rh_quat,
            np.array([self.rh_grip_state], dtype=np.float32),
            lh_pos,
            lh_quat,
            np.array([self.lh_grip_state], dtype=np.float32),
            gaze_pos,
        ]).astype(np.float32)

    # ========================= KEYBOARD ====================================

    def key_callback(self, keycode):
        """Handle keyboard events from the MuJoCo viewer.

        NOTE: This runs in the viewer's render thread.  Do NOT call any
        mujoco.mj_*  functions here — that would race with `mj_step` in
        the main thread and segfault.  Instead, queue a command string
        and process it in the main loop.
        """
        if keycode == glfw.KEY_F:
            self._enqueue_cmd("toggle_teleop")
        elif keycode == glfw.KEY_1:
            self._enqueue_cmd("rh_open")
        elif keycode == glfw.KEY_2:
            self._enqueue_cmd("rh_close")
        elif keycode == glfw.KEY_4:
            self._enqueue_cmd("lh_open")
        elif keycode == glfw.KEY_5:
            self._enqueue_cmd("lh_close")
        elif keycode == glfw.KEY_0:
            self._enqueue_cmd("go_home")
        elif keycode == glfw.KEY_R:
            self._enqueue_cmd("reset_scenario")
        elif keycode == glfw.KEY_SPACE:
            self._enqueue_cmd("toggle_recording")
        elif keycode == glfw.KEY_C:
            self._enqueue_cmd("recalibrate_vr")

    def _process_pending_cmds(self):
        """Execute queued keyboard commands (called from the main loop)."""
        while True:
            with self._pending_cmds_lock:
                if not self._pending_cmds:
                    break
                cmd = self._pending_cmds.pop(0)

            if cmd == "toggle_teleop":
                if self.state == "teleop_active":
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
            elif cmd == "toggle_recording":
                self._toggle_recording()
            elif cmd == "start_recording":
                if not self.recording_active:
                    self._toggle_recording()
            elif cmd == "stop_recording":
                if self.recording_active:
                    self._toggle_recording()
            elif cmd == "recalibrate_vr":
                self.calibrated_origin_rh = None
                self.calibrated_origin_lh = None
                print("[VR] Recalibrated origins")
            elif cmd == "exit_program":
                print("[CLI] Exit requested")
                if self.recording_active:
                    self._toggle_recording()
                self.running = False

    # ========================= MAIN LOOP ===================================

    def _stream_images(self, viewer_cam=None):
        """Stream images to ZMQ if enabled."""
        if self.img_stream_enabled and (time.time() - self._last_stream_time > 0.066):
            try:
                # head_cam → port 10505
                self._head_renderer.update_scene(self.data, camera="head_cam")
                head_img = self._head_renderer.render()
                self._head_cam_pub.send_image(cv2.cvtColor(head_img, cv2.COLOR_RGB2BGR))

                # viewer scene → port 15001
                # viewer scene → port 15001
                camera_arg = viewer_cam
                if camera_arg is None:
                     # Use a fixed 3rd person camera if no specific camera provided
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
            except Exception:
                pass

    # ========================= MAIN LOOP ===================================

    def run(self):
        """Launch the MuJoCo viewer and enter the control loop."""

        with mujoco.viewer.launch_passive(
            self.model, self.data,
            key_callback=self.key_callback,
            show_left_ui=True,
            show_right_ui=True,
        ) as viewer:
            # Reset to home keyframe
            if self.has_home_keyframe:
                mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
                self._apply_home_pose_to_arrays(self.data.qpos, self.data.ctrl)
                self.data.qvel[:] = 0
                mujoco.mj_forward(self.model, self.data)
            else:
                self.data.qpos[:] = self.home_qpos
                self.data.ctrl[:] = self.home_ctrl
                self.data.qvel[:] = 0
                mujoco.mj_forward(self.model, self.data)

            # Camera
            viewer.cam.lookat[:] = [0.3, 0.0, 0.7]
            viewer.cam.distance = 1.5
            viewer.cam.elevation = -15
            viewer.cam.azimuth = 180

            # Show sites, hide contact forces
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

            # -- Print help --
            print("\n" + "=" * 55)
            print("  MuJoCo iCub Teleoperation")
            print("  " + "-" * 51)
            print("  F       Toggle teleop on / off")
            print("  1 / 2   Open / Close right hand")
            print("  4 / 5   Open / Close left hand")
            print("  0       Go home")
            print("  R       Reset scenario (random cube)")
            print("  SPACE   Start / Stop recording")
            print("  C       Recalibrate VR controllers")
            print("  CLI     1=REC, 2=STOP, 3=EXIT")
            print("  Mouse   Ctrl+Right-click to drag targets")
            print("=" * 55 + "\n")

            # Optional stdin controls in a background thread.
            self._start_cli_listener()

            # ---- Control loop ----
            rec_frame_interval = max(1, int((1.0 / 30.0) / self.control_dt))
            ctrl_step = 0

            while viewer.is_running() and self.running:
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0

                step_start = time.time()

                # 1. Process keyboard commands (thread-safe)
                self._process_pending_cmds()

                # 2. VR input
                if self.vr_enabled:
                    self._process_vr_head()
                    self._process_vr_hands()

                # 3. State machine
                if self.state == "teleop_active":
                    self._solve_ik()
                elif self.state == "going_home":
                    self._interpolate_home()
                # idle: do nothing

                # 3b. Gradual hand open/close (runs in any state)
                self._interpolate_hands()

                # 3. Physics step
                for _ in range(self.frame_skip):
                    mujoco.mj_step(self.model, self.data)

                # 4. VR buttons
                if self.latest_buttons:
                    self._process_button_commands()
                    self.latest_buttons = None

                # If button stream is stale, release freeze to avoid latching controls.
                if self.vr_enabled:
                    btn_age = (time.time() - self._last_btn_recv_time) if self._last_btn_recv_time > 0 else np.inf
                    if btn_age > 0.5:
                        self.freeze_rh = False
                        self.freeze_lh = False
                        self.freeze_head = False

                # 5. Recording (at ~30 fps, not every ctrl step)
                if (self.recording_active and self.recorder
                        and ctrl_step % rec_frame_interval == 0):
                    s = self._get_robot_state()
                    w = self._get_world_state()
                    a = self._get_action_vector()
                    self.recorder.add_frame(s, a, w)

                ctrl_step += 1

                # 6. Stream images (~15 fps)
                self._stream_images(viewer.cam)

                # 7. Viewer sync
                viewer.sync()

                # 8. Rate limit
                elapsed = time.time() - step_start
                remaining = self.control_dt - elapsed
                if remaining > 0:
                    time.sleep(remaining)

        # -- Cleanup --
        if self.recording_active and self.recorder:
            self.recorder.stop_episode()
            self.episode_count += 1
        self.running = False
        # Close image streaming
        if self._head_renderer is not None:
            self._head_renderer.close()
        if self._viewer_renderer is not None:
            self._viewer_renderer.close()
        # Release the ZMQ ports so the next session can bind them again
        # (otherwise the next MuJoCoTeleop's bind() fails silently and BeaVR
        # is left showing the last frame from this now-dead publisher).
        if self._head_cam_pub is not None:
            self._head_cam_pub.stop()
        if self._viewer_pub is not None:
            self._viewer_pub.stop()
        # Close VR sockets
        if self.vr_enabled:
            for _, (ctx, sock) in self._vr_sockets.items():
                sock.close()
                ctx.term()
        print("\nTeleop session ended.")


# ===========================================================================
#                                CLI
# ===========================================================================

def main():
    _script_dir = Path(__file__).resolve().parent
    default_config = str(_script_dir / "config.yaml")
    parser = argparse.ArgumentParser(
        description="MuJoCo-only iCub Teleoperation")
    parser.add_argument(
        "--config", type=str,
        default=default_config if Path(default_config).exists() else None,
        help="Path to config.yaml (auto-detected in mu_teleop/)")
    parser.add_argument(
        "--model", type=str,
        default=str(_script_dir / "scenes" / "icub_table_scene.xml"),
        help="Path to MuJoCo scene XML")
    parser.add_argument("--vr", action="store_true",
                        help="Enable VR controller input via ZMQ (optional; --quest-ip also enables VR)")
    parser.add_argument("--vr-ip", "--quest-ip", dest="vr_ip", type=str, default=None,
                        help="VR publisher IP for ZMQ connect mode (aliases: --vr-ip, --quest-ip). If provided, VR is enabled automatically.")
    parser.add_argument("--record", action="store_true",
                        help="Enable dataset recording (LeRobot)")
    parser.add_argument("--repo-id", type=str, default="icub_mujoco",
                        help="Dataset repository / name")
    parser.add_argument("--arms", type=str, default="both",
                        choices=["both", "right", "left"],
                        help="Which arms to control (overrides config.yaml)")
    parser.add_argument("--primary-arm", type=str, default="right",
                        choices=["right", "left"],
                        help="Which arm owns the torso joints (overrides config.yaml)")
    args = parser.parse_args()
    vr_enabled = args.vr or bool(args.vr_ip)

    teleop = MuJoCoTeleop(
        model_path=args.model,
        vr_enabled=vr_enabled,
        record_enabled=args.record,
        repo_id=args.repo_id,
        control_arms=args.arms,
        primary_arm=args.primary_arm,
        config_path=args.config,
        vr_ip=args.vr_ip,
    )
    try:
        teleop.run()
    except KeyboardInterrupt:
        teleop.running = False
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
