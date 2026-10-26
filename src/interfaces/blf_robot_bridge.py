#!/usr/bin/env python3
"""
BLF Robot Bridge: IIT-Standard Robot Interface for iCub / ergoCub.
Uses Bipedal Locomotion Framework (BLF) native bindings:
  - blf.robot_interface.construct_remote_control_board_remapper
  - blf.robot_interface.YarpRobotControl (SI units, max admissible jump safety)
  - blf.robot_interface.YarpSensorBridge (streaming UDP non-blocking reads)
  - Strict monotonic timing with deadline miss watchdog (AdvanceableRunner pattern)

Enhanced for icub-lerobot to also manage:
  - Hand finger position control (9 DoFs per hand on left_arm / right_arm, indices 7..15)
  - Eye stabilization (3 DoFs on head, indices 3..5 locked at 0 deg)
  - Full 41-DoF state & torque reporting (20 upper-body via BLF + 18 fingers + 3 eyes)
  - Non-blocking YARP RGB stereo camera acquisition (left, right) for Gazebo & Real iCub

Compatible with Gazebo Sim (Harmonic) across all 5 iCub models and the Real iCub Robot.
"""

import os
import sys
import time
import tempfile
import pathlib
from typing import Optional, List, Tuple, Dict
import numpy as np
import yarp
import bipedal_locomotion_framework.bindings as blf


class BLFRobotBridge:
    """
    Standard Robot Interface leveraging BLF YarpRobotControl and YarpSensorBridge
    for the 20 upper-body DoFs, plus auxiliary YARP interfaces for hand fingers,
    eye stabilization, and RGB stereo cameras.
    """

    JOINTS_LIST = [
        "torso_yaw", "torso_roll", "torso_pitch",
        "neck_pitch", "neck_roll", "neck_yaw",
        "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow",
        "l_wrist_prosup", "l_wrist_pitch", "l_wrist_yaw",
        "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow",
        "r_wrist_prosup", "r_wrist_pitch", "r_wrist_yaw",
    ]

    PARTS = ["torso", "head", "left_arm", "right_arm"]
    TOTAL_AXES = 20

    HAND_JOINTS_LEFT = [
        "l_hand_finger", "l_thumb_oppose", "l_thumb_proximal", "l_thumb_distal",
        "l_index_proximal", "l_index_distal", "l_middle_proximal", "l_middle_distal",
        "l_pinky",
    ]
    HAND_JOINTS_RIGHT = [
        "r_hand_finger", "r_thumb_oppose", "r_thumb_proximal", "r_thumb_distal",
        "r_index_proximal", "r_index_distal", "r_middle_proximal", "r_middle_distal",
        "r_pinky",
    ]
    EYE_JOINTS = ["eyes_tilt", "eyes_version", "eyes_vergence"]

    # Nominal physical torque limits [N*m] for iCub upper-body joints
    MAX_TORQUE_LIMITS = np.array([
        # Torso (yaw, roll, pitch)
        40.0, 40.0, 40.0,
        # Neck (pitch, roll, yaw)
        6.0, 6.0, 6.0,
        # Left arm (3 shoulder, 1 elbow, 3 wrist)
        25.0, 25.0, 20.0, 18.0, 6.0, 6.0, 6.0,
        # Right arm (3 shoulder, 1 elbow, 3 wrist)
        25.0, 25.0, 20.0, 18.0, 6.0, 6.0, 6.0,
    ], dtype=np.float64)

    def __init__(
        self,
        robot_name: str = "icubSim",
        local_prefix: str = "teleop_blf",
        frequency: float = 50.0,
        default_mode: str = "position_direct",
        max_admissible_error: float = 0.087,  # ~5 degrees max jump per cycle (safety safeguard)
        max_deadline_misses: int = 5,
        use_network_clock: bool = True,
        max_torque_limits: Optional[np.ndarray] = None,
        hand_open_deg: Optional[List[float]] = None,
        hand_close_deg: Optional[List[float]] = None,
        hand_close_ratio: float = 0.5,
    ):
        self.robot_name = robot_name.strip("/")
        # Append PID to client prefix to ensure unique YARP port names
        self.local_prefix = f"{local_prefix}_{os.getpid()}"
        self.frequency = frequency
        self.dt = 1.0 / frequency
        self.default_mode = default_mode.lower()
        self.max_admissible_error = max_admissible_error
        self.max_deadline_misses = max_deadline_misses
        self.use_network_clock = use_network_clock

        if max_torque_limits is not None:
            self.max_torque_limits = np.asarray(max_torque_limits, dtype=np.float64).copy()
        else:
            self.max_torque_limits = self.MAX_TORQUE_LIMITS.copy()

        # Hand open/close profiles [deg] (9 joints per hand)
        default_open = np.array([0, 75, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        default_close = np.array([0, 40, 40, 50, 75, 75, 75, 75, 0], dtype=np.float64)
        self.hand_open_deg = (
            np.asarray(hand_open_deg, dtype=np.float64).ravel()
            if hand_open_deg is not None
            else default_open
        )
        full_close = (
            np.asarray(hand_close_deg, dtype=np.float64).ravel()
            if hand_close_deg is not None
            else default_close
        )
        ratio = float(np.clip(hand_close_ratio, 0.0, 1.0))
        self.hand_close_deg = self.hand_open_deg + ratio * (full_close - self.hand_open_deg)

        # BLF and YARP objects
        self.network = yarp.Network()
        self.param_handler = None
        self.poly_remapper = None
        self.robot_control: Optional[blf.robot_interface.YarpRobotControl] = None
        self.sensor_bridge: Optional[blf.robot_interface.YarpSensorBridge] = None

        # Auxiliary YARP PolyDrivers for hand fingers (left_arm, right_arm) & eyes (head)
        self.aux_drivers: Dict[str, yarp.PolyDriver] = {}
        self.aux_encoders: Dict[str, yarp.IEncoders] = {}
        self.aux_pos_control: Dict[str, yarp.IPositionControl] = {}
        self.aux_pos_direct: Dict[str, yarp.IPositionDirect] = {}
        self.aux_control_modes: Dict[str, yarp.IControlMode] = {}
        self.aux_axis_info: Dict[str, yarp.IAxisInfo] = {}
        self.aux_num_axes: Dict[str, int] = {}

        # Virtual fallback cache for hand finger positions [deg] when using a 7-DoF arm model
        self._virtual_hand_deg: Dict[str, np.ndarray] = {
            "left_arm": self.hand_open_deg.copy(),
            "right_arm": self.hand_open_deg.copy(),
        }

        # Camera ports & buffers
        self.cam_ports: Dict[str, yarp.BufferedPortImageRgb] = {}
        self.img_buffers: Dict[str, yarp.ImageRgb] = {}
        self.np_arrays: Dict[str, np.ndarray] = {}
        self.cam_frames: Dict[str, int] = {"left": 0, "right": 0}
        self.cam_stamps: Dict[str, Optional[Tuple[int, float]]] = {"left": None, "right": None}

        # State cache for 20 upper-body joints
        self.last_q = np.zeros(self.TOTAL_AXES, dtype=np.float64)
        self.last_dq = np.zeros(self.TOTAL_AXES, dtype=np.float64)
        self.last_tau = np.zeros(self.TOTAL_AXES, dtype=np.float64)
        self.is_connected = False
        self.current_mode = self.default_mode

        # Real-time clock and watchdog state (IIT AdvanceableRunner pattern)
        self.wake_up_time = 0.0
        self.deadline_miss_count = 0
        self.total_deadline_misses = 0

        # Temporary config file
        self._tmp_config_file = None
        self._using_network_clock = False

    def connect(
        self,
        timeout: float = 15.0,
        enable_hands_and_eyes: bool = True,
        enable_cameras: bool = False,
    ) -> bool:
        """
        Connect to robot control boards using BLF YarpRobotControl and YarpSensorBridge,
        and optionally open auxiliary handles for hand fingers, eyes, and stereo cameras.
        """
        if not yarp.Network.checkNetwork(2.0):
            print("[BLFRobotBridge] Error: YARP name server is not running or not reachable.")
            print("[BLFRobotBridge] Please ensure yarpserver is active.")
            return False

        print(f"[BLFRobotBridge] Connected to nameserver. Locating controlboards for /{self.robot_name}...")

        # Verify remote controlboard RPC ports are advertised
        start_time = time.time()
        for part in self.PARTS:
            rpc_port = f"/{self.robot_name}/{part}/rpc:i"
            remote_port = f"/{self.robot_name}/{part}"
            port_found = False
            while time.time() - start_time < timeout:
                if yarp.Network.exists(rpc_port) or yarp.Network.exists(remote_port):
                    port_found = True
                    break
                time.sleep(0.5)

            if not port_found:
                print(f"[BLFRobotBridge] Error: Controlboard port '{rpc_port}' not found within {timeout}s.")
                self.close()
                return False

        # If in simulation, synchronize with /clock if available
        if self.use_network_clock and yarp.Network.exists("/clock"):
            yarp.useNetworkClock("/clock")
            self._using_network_clock = True
            print("[BLFRobotBridge] Synchronized with YARP network clock (/clock).")
        else:
            self._using_network_clock = False

        # Generate standard BLF configuration
        joints_str = " ".join(self.JOINTS_LIST)
        parts_str = " ".join(self.PARTS)

        config_content = f"""
[REMOTE_CONTROL_BOARD]
robot_name                             {self.robot_name}
remote_control_boards                  ({parts_str})
joints_list                            ({joints_str})
position_direct_max_admissible_error   {self.max_admissible_error}
positioning_duration                   5.0
positioning_tolerance                  0.05

[SENSOR_BRIDGE]
check_for_nan                          true
stream_joint_states                    true
stream_motor_states                    false
stream_forcetorque_sensors             false
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
            f.write(config_content)
            self._tmp_config_file = f.name

        # Load parameters with BLF YarpParametersHandler
        self.param_handler = blf.parameters_handler.YarpParametersHandler()
        if not self.param_handler.set_from_filename(self._tmp_config_file):
            print("[BLFRobotBridge] Error: Failed to parse configuration in YarpParametersHandler.")
            self.close()
            return False

        rcb_group = self.param_handler.get_group("REMOTE_CONTROL_BOARD")
        rcb_group.set_parameter_string("local_prefix", self.local_prefix)

        # 1. Construct centralized RemoteControlBoardRemapper
        print("[BLFRobotBridge] Constructing centralized RemoteControlBoardRemapper...")
        self.poly_remapper = blf.robot_interface.construct_remote_control_board_remapper(rcb_group)
        if self.poly_remapper is None or not self.poly_remapper.is_valid():
            print("[BLFRobotBridge] Error: Failed to construct remote control board remapper.")
            self.close()
            return False

        # 2. Initialize YarpRobotControl
        print("[BLFRobotBridge] Initializing YarpRobotControl...")
        self.robot_control = blf.robot_interface.YarpRobotControl()
        if not self.robot_control.initialize(rcb_group):
            print("[BLFRobotBridge] Error: Failed to initialize YarpRobotControl.")
            self.close()
            return False

        if not self.robot_control.set_driver(self.poly_remapper.poly):
            print("[BLFRobotBridge] Error: Failed to set driver on YarpRobotControl.")
            self.close()
            return False

        # 3. Initialize YarpSensorBridge (streaming UDP non-blocking reads)
        print("[BLFRobotBridge] Initializing YarpSensorBridge (streaming mode)...")
        sb_group = self.param_handler.get_group("SENSOR_BRIDGE")
        self.sensor_bridge = blf.robot_interface.YarpSensorBridge()
        if not self.sensor_bridge.initialize(sb_group):
            print("[BLFRobotBridge] Error: Failed to initialize YarpSensorBridge.")
            self.close()
            return False

        if not self.sensor_bridge.set_drivers_list([self.poly_remapper]):
            print("[BLFRobotBridge] Error: Failed to link driver to YarpSensorBridge.")
            self.close()
            return False

        # Read initial sample to ensure stream is active
        time.sleep(0.3)
        self.sensor_bridge.advance()
        ok, q_init, _ = self.sensor_bridge.get_joint_positions()
        if not ok or len(q_init) != self.TOTAL_AXES:
            print("[BLFRobotBridge] Warning: Initial sensor read did not return valid data yet. Retrying...")
            for _ in range(10):
                time.sleep(0.1)
                self.sensor_bridge.advance()
                ok, q_init, _ = self.sensor_bridge.get_joint_positions()
                if ok and len(q_init) == self.TOTAL_AXES:
                    break

        if ok and len(q_init) == self.TOTAL_AXES:
            self.last_q = np.array(q_init, dtype=np.float64)
            print(f"[BLFRobotBridge] Initial joint state read successfully ({self.TOTAL_AXES} DoFs, SI units [rad]).")
        else:
            print("[BLFRobotBridge] Error: Could not obtain valid joint readings from YarpSensorBridge.")
            self.close()
            return False

        # 4. Set initial control mode
        # IMPORTANT: If default_mode is 'torque', do NOT switch to torque mode during connect()
        # before gravity-compensation torques are ready; otherwise Gazebo sets referenceTorque=0.0
        # and the arms collapse under gravity before the control loop starts.
        if self.default_mode == "torque":
            self.set_control_mode("position")
            print("[BLFRobotBridge] Holding posture in POSITION mode until first send_torques() call.")
        else:
            self.set_control_mode(self.default_mode)

        # 5. Open auxiliary handles for hand fingers & eye joints if requested
        if enable_hands_and_eyes:
            self._connect_auxiliary_parts()
            self.lock_eyes_forward()

        # 6. Connect stereo cameras if requested
        if enable_cameras:
            self.connect_cameras()

        self.is_connected = True
        self.init_clock()
        print("[BLFRobotBridge] Robot successfully connected and configured via BLF.")
        return True

    def _connect_auxiliary_parts(self):
        """
        Open auxiliary remote_controlboard handles on head, left_arm, and right_arm
        for eye locking (head indices 3..5) and hand finger actuation (arm indices 7..15).
        Does NOT alter the control modes of the 20 BLF-controlled upper-body joints.
        """
        for part_name in ("head", "left_arm", "right_arm"):
            props = yarp.Property()
            props.put("device", "remote_controlboard")
            props.put("local", f"/{self.local_prefix}/aux/{part_name}")
            props.put("remote", f"/{self.robot_name}/{part_name}")
            driver = yarp.PolyDriver(props)
            if not driver.isValid():
                print(f"[BLFRobotBridge] Warning: Could not open auxiliary driver for {part_name}.")
                continue

            ienc = driver.viewIEncoders()
            ipos = driver.viewIPositionControl()
            iposd = driver.viewIPositionDirect()
            imode = driver.viewIControlMode()
            iinfo = driver.viewIAxisInfo()
            num_axes = ienc.getAxes() if ienc is not None else 0

            self.aux_drivers[part_name] = driver
            self.aux_encoders[part_name] = ienc
            self.aux_pos_control[part_name] = ipos
            self.aux_pos_direct[part_name] = iposd
            self.aux_control_modes[part_name] = imode
            self.aux_axis_info[part_name] = iinfo
            self.aux_num_axes[part_name] = num_axes

    def lock_eyes_forward(self, tilt_deg: float = 0.0, version_deg: float = 0.0, vergence_deg: float = 0.0):
        """
        Lock the 3 eye joints (eyes_tilt, eyes_version, eyes_vergence) on the head part
        to fixed angles (default 0 deg) so the eye cameras move rigidly with the head frame.
        """
        if "head" not in self.aux_drivers:
            return
        num_j = self.aux_num_axes.get("head", 0)
        if num_j <= 3:
            return

        imode = self.aux_control_modes.get("head")
        ipos = self.aux_pos_control.get("head")
        iinfo = self.aux_axis_info.get("head")
        if imode is None or ipos is None:
            return

        targets = {
            "tilt": float(tilt_deg),
            "version": float(version_deg),
            "vergence": float(vergence_deg),
        }
        for idx in range(3, num_j):
            jname = iinfo.getAxisName(idx).lower() if iinfo is not None else ""
            val = 0.0
            for key, v in targets.items():
                if key in jname:
                    val = v
                    break
            imode.setControlMode(idx, yarp.VOCAB_CM_POSITION)
            ipos.positionMove(idx, val)

    def send_hand_positions(self, arm: str, target_positions_deg: np.ndarray) -> bool:
        """
        Command the 9 hand finger joints (indices 7..15) of 'left_arm' or 'right_arm' in degrees.
        If the simulated model has only 7 arm joints (no fingers), caches the target virtually.
        """
        target_deg = np.asarray(target_positions_deg, dtype=np.float64).ravel()
        if arm in self._virtual_hand_deg and len(target_deg) == 9:
            self._virtual_hand_deg[arm] = target_deg.copy()

        num_total = self.aux_num_axes.get(arm, 0)
        if num_total <= 7:
            return True

        num_hand = num_total - 7
        if len(target_deg) != num_hand:
            return False

        imode = self.aux_control_modes.get(arm)
        ipos = self.aux_pos_control.get(arm)
        if imode is None or ipos is None:
            return False

        modes = yarp.VectorInt(num_total)
        imode.getControlModes(modes.data())
        for i in range(7, num_total):
            if modes[i] != yarp.VOCAB_CM_POSITION:
                imode.setControlMode(i, yarp.VOCAB_CM_POSITION)

        for i in range(num_hand):
            ipos.positionMove(i + 7, float(target_deg[i]))
        return True

    def set_hand_closure(self, arm: str, closure_fraction: float) -> bool:
        """
        Set hand closure level in [0.0, 1.0] where 0.0 = hand_open_deg and 1.0 = hand_close_deg.
        Accepts arm='right_arm'|'right'|'rh' or 'left_arm'|'left'|'lh'.
        """
        arm_norm = "right_arm" if arm.lower().startswith("r") else "left_arm"
        level = float(np.clip(closure_fraction, 0.0, 1.0))
        target_deg = self.hand_open_deg + level * (self.hand_close_deg - self.hand_open_deg)
        return self.send_hand_positions(arm_norm, target_deg)

    def get_hand_positions_deg(self, arm: str) -> np.ndarray:
        """
        Return the 9 hand finger joint positions in degrees for 'left_arm' or 'right_arm'.
        """
        arm_norm = "right_arm" if arm.lower().startswith("r") else "left_arm"
        num_total = self.aux_num_axes.get(arm_norm, 0)
        ienc = self.aux_encoders.get(arm_norm)
        if num_total > 7 and ienc is not None:
            encs = yarp.Vector(num_total)
            if ienc.getEncoders(encs.data()):
                return np.array([encs[i] for i in range(7, num_total)], dtype=np.float64)
        return self._virtual_hand_deg[arm_norm].copy()

    def get_full_joint_state(
        self,
        actuators_order: Optional[List[str]] = None,
        include_torques: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (q_rad, dq_rad, tau_nm) for the requested actuator list (e.g. all 41 actuators
        in control_config.yaml), combining the 20 BLF streaming joints with auxiliary finger/eye encoders.
        """
        q_20 = self.get_joint_positions()
        dq_20 = self.get_joint_velocities()
        tau_20 = self.get_joint_torques() if include_torques else np.full(self.TOTAL_AXES, np.nan)

        if actuators_order is None:
            return q_20, dq_20, tau_20

        q_map: Dict[str, float] = {}
        dq_map: Dict[str, float] = {}
        tau_map: Dict[str, float] = {}

        for idx, jname in enumerate(self.JOINTS_LIST):
            q_map[jname] = float(q_20[idx])
            dq_map[jname] = float(dq_20[idx])
            tau_map[jname] = float(tau_20[idx])

        # Populate eye joints (default 0.0 rad)
        for jname in self.EYE_JOINTS:
            q_map[jname] = 0.0
            dq_map[jname] = 0.0
            tau_map[jname] = np.nan

        # Populate left and right hand joints from auxiliary encoders or virtual cache
        for arm_name, hand_names in (
            ("left_arm", self.HAND_JOINTS_LEFT),
            ("right_arm", self.HAND_JOINTS_RIGHT),
        ):
            num_total = self.aux_num_axes.get(arm_name, 0)
            ienc = self.aux_encoders.get(arm_name)
            if num_total >= 16 and ienc is not None:
                pos_vec = yarp.Vector(num_total)
                vel_vec = yarp.Vector(num_total)
                ienc.getEncoders(pos_vec.data())
                ienc.getEncoderSpeeds(vel_vec.data())
                for k, hname in enumerate(hand_names):
                    q_map[hname] = float(np.deg2rad(pos_vec[7 + k]))
                    dq_map[hname] = float(np.deg2rad(vel_vec[7 + k]))
                    tau_map[hname] = np.nan
            else:
                virt_deg = self._virtual_hand_deg[arm_name]
                for k, hname in enumerate(hand_names):
                    q_map[hname] = float(np.deg2rad(virt_deg[k]))
                    dq_map[hname] = 0.0
                    tau_map[hname] = np.nan

        n = len(actuators_order)
        q_out = np.zeros(n, dtype=np.float64)
        dq_out = np.zeros(n, dtype=np.float64)
        tau_out = np.full(n, np.nan, dtype=np.float64)
        for i, name in enumerate(actuators_order):
            q_out[i] = q_map.get(name, 0.0)
            dq_out[i] = dq_map.get(name, 0.0)
            tau_out[i] = tau_map.get(name, np.nan)

        return q_out, dq_out, tau_out

    def connect_cameras(self, cams: Tuple[str, ...] = ("left", "right"), width: int = 320, height: int = 240):
        """
        Open non-blocking YARP BufferedPortImageRgb receivers for the robot's stereo cameras.
        Supports both Gazebo (/{robot_name}/cam/{eye}/rgbImage:o) and Real iCub (/{robot_name}/camcalib/{eye}/out).
        """
        robot_norm = self.robot_name.lower()
        for cam in cams:
            if cam in self.cam_ports:
                continue
            port = yarp.BufferedPortImageRgb()
            local_port = f"/{self.local_prefix}/cam/{cam}:i"
            if not port.open(local_port):
                print(f"[BLFRobotBridge] Warning: Could not open local camera port {local_port}")
                continue
            self.cam_ports[cam] = port

            if robot_norm == "icub":
                source_port = f"/{self.robot_name}/camcalib/{cam}/out"
            else:
                source_port = f"/{self.robot_name}/cam/{cam}/rgbImage:o"

            if yarp.Network.exists(source_port):
                if yarp.Network.connect(source_port, local_port, "fast_tcp", False):
                    print(f"[BLFRobotBridge] Connected camera '{cam}': {source_port} -> {local_port}")
            else:
                print(f"[BLFRobotBridge] Camera source port '{source_port}' not yet online (will auto-connect when available).")

            arr = np.zeros((height, width, 3), dtype=np.uint8)
            self.np_arrays[cam] = arr
            buf = yarp.ImageRgb()
            buf.resize(width, height)
            self.img_buffers[cam] = buf
            buf.setExternal(arr.data, arr.shape[1], arr.shape[0])
            self.cam_frames[cam] = 0
            self.cam_stamps[cam] = None

    def get_camera_images(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Non-blocking read of (left_rgb, right_rgb) camera frames as (H, W, 3) uint8 numpy arrays.
        """
        out: Dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        robot_norm = self.robot_name.lower()
        for eye in ("left", "right"):
            port = self.cam_ports.get(eye)
            if port is None:
                continue
            # Auto-connect if not yet connected
            if port.getInputCount() == 0:
                src = (
                    f"/{self.robot_name}/camcalib/{eye}/out"
                    if robot_norm == "icub"
                    else f"/{self.robot_name}/cam/{eye}/rgbImage:o"
                )
                if yarp.Network.exists(src):
                    yarp.Network.connect(src, port.getName(), "fast_tcp", False)

            img = port.read(False)
            if img is not None:
                w, h = img.width(), img.height()
                if self.np_arrays[eye].shape[:2] != (h, w):
                    self.np_arrays[eye] = np.zeros((h, w, 3), dtype=np.uint8)
                    self.img_buffers[eye].resize(w, h)
                    self.img_buffers[eye].setExternal(
                        self.np_arrays[eye].data,
                        self.np_arrays[eye].shape[1],
                        self.np_arrays[eye].shape[0],
                    )
                self.img_buffers[eye].copy(img)
                self.cam_frames[eye] = self.cam_frames.get(eye, 0) + 1
                stamp = yarp.Stamp()
                self.cam_stamps[eye] = (
                    (stamp.getCount(), stamp.getTime())
                    if port.getEnvelope(stamp) and stamp.isValid()
                    else None
                )
                out[eye] = self.np_arrays[eye]
        return out["left"], out["right"]

    def set_control_mode(self, mode: str) -> bool:
        """
        Switch control mode: 'position_direct', 'velocity', 'position', or 'torque'.
        """
        if self.robot_control is None or not self.robot_control.is_valid():
            return False

        mode = mode.lower()
        if mode == "position_direct":
            target_mode = blf.robot_interface.YarpRobotControl.PositionDirect
        elif mode == "velocity":
            target_mode = blf.robot_interface.YarpRobotControl.Velocity
        elif mode == "position":
            target_mode = blf.robot_interface.YarpRobotControl.Position
        elif mode == "torque":
            target_mode = blf.robot_interface.YarpRobotControl.Torque
        else:
            raise ValueError(
                f"Unsupported mode: {mode}. Choose 'position_direct', 'velocity', 'position', or 'torque'."
            )

        ok = self.robot_control.set_control_mode(target_mode)
        if ok:
            self.current_mode = mode
            print(f"[BLFRobotBridge] Active control mode set to: '{mode.upper()}'.")
        else:
            print(f"[BLFRobotBridge] Warning: Failed to set control mode to '{mode}'.")
        return ok

    def get_joint_positions(self) -> np.ndarray:
        """
        Reads joint positions in radians [rad] for all 20 upper-body DoFs via streaming.
        """
        if not self.is_connected:
            raise RuntimeError("BLFRobotBridge is not connected.")

        self.sensor_bridge.advance()
        ok, q, _ = self.sensor_bridge.get_joint_positions()
        if ok and len(q) == self.TOTAL_AXES:
            self.last_q = np.array(q, dtype=np.float64)
        return np.copy(self.last_q)

    def get_joint_velocities(self) -> np.ndarray:
        """
        Reads joint velocities in radians/s [rad/s] for all 20 upper-body DoFs via streaming.
        """
        if not self.is_connected:
            raise RuntimeError("BLFRobotBridge is not connected.")

        ok, dq, _ = self.sensor_bridge.get_joint_velocities()
        if ok and len(dq) == self.TOTAL_AXES:
            self.last_dq = np.array(dq, dtype=np.float64)
        return np.copy(self.last_dq)

    def get_joint_torques(self) -> np.ndarray:
        """
        Reads measured joint torques in [N*m] for all 20 upper-body DoFs via streaming.
        """
        if not self.is_connected:
            raise RuntimeError("BLFRobotBridge is not connected.")

        ok, tau, _ = self.sensor_bridge.get_joint_torques()
        if ok and len(tau) == self.TOTAL_AXES:
            self.last_tau = np.array(tau, dtype=np.float64)
        return np.copy(self.last_tau)

    def send_position_direct(self, positions_rad: np.ndarray, current_positions_rad: Optional[np.ndarray] = None) -> bool:
        """
        Send direct position references (20 DoFs) [rad].
        Enforces maximum admissible error safety check against current_positions_rad.
        """
        if not self.is_connected:
            raise RuntimeError("BLFRobotBridge is not connected.")

        curr_q = current_positions_rad if current_positions_rad is not None else self.last_q

        if self.current_mode != "position_direct":
            self.set_control_mode("position_direct")

        ok = self.robot_control.set_references(
            positions_rad,
            blf.robot_interface.YarpRobotControl.PositionDirect,
            curr_q,
        )

        if not ok:
            print("[BLFRobotBridge][SAFETY] set_references rejected! Step exceeded position_direct_max_admissible_error.")
        return ok

    def send_velocities(self, velocities_rad_s: np.ndarray) -> bool:
        """
        Send direct joint velocities (20 DoFs) [rad/s].
        """
        if not self.is_connected:
            raise RuntimeError("BLFRobotBridge is not connected.")

        if self.current_mode != "velocity":
            self.set_control_mode("velocity")

        return self.robot_control.set_references(
            velocities_rad_s,
            blf.robot_interface.YarpRobotControl.Velocity,
        )

    def send_torques(self, torques_nm: np.ndarray) -> bool:
        """
        Send joint torque references (20 DoFs) [N*m].
        Performs NaN/Inf safety validation and clamps commands to [-max_torque_limits, +max_torque_limits].
        Automatically transitions to TORQUE mode immediately before applying the first torque vector
        to prevent any zero-torque free-fall gap.
        """
        if not self.is_connected:
            raise RuntimeError("BLFRobotBridge is not connected.")

        tau = np.asarray(torques_nm, dtype=np.float64)
        if tau.shape != (self.TOTAL_AXES,):
            raise ValueError(f"torques_nm must have shape ({self.TOTAL_AXES},), got {tau.shape}")

        if np.any(np.isnan(tau)) or np.any(np.isinf(tau)):
            print("[BLFRobotBridge][SAFETY] NaN or Inf detected in torque references! Command rejected.")
            return False

        if self.current_mode != "torque":
            if not self.set_control_mode("torque"):
                return False

        tau_clamped = np.clip(tau, -self.max_torque_limits, self.max_torque_limits)
        return self.robot_control.set_references(
            tau_clamped,
            blf.robot_interface.YarpRobotControl.Torque,
        )

    def home_pose_dict_to_vector(self, home_pose_deg: Dict[str, float], fallback_rad: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convert a {joint_name: deg} dictionary (e.g. from control_config.yaml) into a (20,) array in radians
        ordered according to BLFRobotBridge.JOINTS_LIST.
        """
        q_target = (
            np.asarray(fallback_rad, dtype=np.float64).copy()
            if fallback_rad is not None
            else self.last_q.copy()
        )
        for i, jname in enumerate(self.JOINTS_LIST):
            if jname in home_pose_deg:
                q_target[i] = np.deg2rad(float(home_pose_deg[jname]))
        return q_target

    def move_to_posture(self, target_positions_rad: np.ndarray, duration: float = 3.0, step_callback=None) -> bool:
        """
        Smoothly move the robot from its current joint configuration to target_positions_rad
        in PositionDirect mode using a quintic minimum-jerk polynomial trajectory.
        If step_callback is provided, it is invoked after each step (e.g. for streaming state).
        """
        if not self.is_connected:
            raise RuntimeError("BLFRobotBridge is not connected.")

        q_start = self.get_joint_positions()
        q_target = np.asarray(target_positions_rad, dtype=np.float64)
        max_diff = float(np.max(np.abs(q_target - q_start)))
        if max_diff < 0.03:
            return True

        prev_mode = self.current_mode
        if self.current_mode != "position_direct":
            if not self.set_control_mode("position_direct"):
                return False

        steps = max(1, int(duration * self.frequency))
        self.init_clock()
        for step in range(1, steps + 1):
            tau = step / float(steps)
            s = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
            q_des = (1.0 - s) * q_start + s * q_target
            q_curr = self.get_joint_positions()
            self.send_position_direct(q_des, q_curr)
            if step_callback is not None:
                try:
                    step_callback()
                except Exception:
                    pass
            self.wait_next_cycle()

        if prev_mode not in ("position_direct", "torque"):
            self.set_control_mode(prev_mode)
        return True

    def init_clock(self):
        """Initialize monotonic start time for fixed-rate control loop."""
        self.wake_up_time = yarp.now()
        self.deadline_miss_count = 0
        self.total_deadline_misses = 0

    def wait_next_cycle(self) -> Tuple[bool, float, float]:
        """
        Strict monotonic real-time timer adhering to IIT AdvanceableRunner pattern:
          - wake_up_time += dt (zero drift accumulation)
          - monitors deadline misses
          - triggers watchdog if consecutive deadline misses exceed max_deadline_misses
        Returns: (is_ok, current_time, delay_or_lag_seconds)
        """
        self.wake_up_time += self.dt
        now = yarp.now()

        lag = now - self.wake_up_time
        if lag > 0.0:
            # Deadline missed
            self.deadline_miss_count += 1
            self.total_deadline_misses += 1
            if self.deadline_miss_count >= self.max_deadline_misses:
                print(
                    f"[BLFRobotBridge][WATCHDOG] Exceeded {self.max_deadline_misses} consecutive deadline misses! "
                    f"Last lag: {lag*1000:.1f} ms. Aborting for robot safety."
                )
                return False, now, lag
            # Yield CPU briefly before next iteration
            yarp.delay(0.001)
            return True, now, lag
        else:
            # Met deadline: sleep exactly until next cycle
            self.deadline_miss_count = 0
            yarp.delay(-lag)
            return True, now, lag

    def close(self):
        """Safely bring robot to halt and release YARP drivers."""
        if not self.is_connected and self.poly_remapper is None and not self.aux_drivers:
            return

        self.is_connected = False

        # 1. Switch robot control to Position holding mode to safely lock joints
        if self.robot_control is not None and self.robot_control.is_valid():
            try:
                self.robot_control.set_control_mode(blf.robot_interface.YarpRobotControl.Position)
            except Exception:
                pass

        # 2. Reset clock to system clock BEFORE closing devices and ports
        if getattr(self, "_using_network_clock", False) or yarp.isNetworkClock():
            try:
                yarp.useSystemClock()
                self._using_network_clock = False
                print("[BLFRobotBridge] Restored standard system clock.")
            except Exception:
                pass

        # 3. Close camera ports
        for p in self.cam_ports.values():
            try:
                p.close()
            except Exception:
                pass
        self.cam_ports.clear()

        # 4. Close auxiliary drivers
        for d in self.aux_drivers.values():
            try:
                if d.isValid():
                    d.close()
            except Exception:
                pass
        self.aux_drivers.clear()
        self.aux_encoders.clear()
        self.aux_pos_control.clear()
        self.aux_pos_direct.clear()
        self.aux_control_modes.clear()
        self.aux_axis_info.clear()

        # 5. Cleanly delete BLF higher-level objects before closing poly driver
        self.sensor_bridge = None
        self.robot_control = None

        # 6. Close the poly driver
        if self.poly_remapper is not None:
            try:
                if self.poly_remapper.poly is not None and self.poly_remapper.poly.isValid():
                    self.poly_remapper.poly.close()
            except Exception:
                pass
            self.poly_remapper = None

        # 7. Clean up temporary configuration file
        if self._tmp_config_file and os.path.exists(self._tmp_config_file):
            try:
                os.remove(self._tmp_config_file)
            except Exception:
                pass
            self._tmp_config_file = None

        time.sleep(0.1)
        print("[BLFRobotBridge] Robot interface cleanly closed.")

