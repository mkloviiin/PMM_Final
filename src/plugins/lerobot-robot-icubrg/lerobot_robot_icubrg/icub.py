import math
import time
from typing import Any

import numpy as np
import torch

from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
import yarp

from .config_icub import iCubConfig, load_control_config
from .gz_pose import GzContactReader, GzPoseReader

CAMERA_FIRST_FRAME_TIMEOUT_S = 5.0
CAMERA_STALE_S = 1.0
TOUCH_FINGER_SLOT = {"thumb": 4, "index": 0, "middle": 1}
TAXELS_PER_FINGER = 12

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


class iCub(Robot):
    config_class = iCubConfig
    name = "icub_remote"

    def __init__(self, config: iCubConfig):
        super().__init__(config)
        self.yarp = yarp
        self.config = config

        self.ports = {}
        self.img_ports = {}
        self.img_buffers = {}
        self.np_arrays = {}
        self._native_bufs: dict[str, tuple] = {}
        self._cam_last_t: dict[str, float] = {}
        self._cam_stale_warn_t = 0.0
        self._connected = False
        self._calibrated = True
        self.is_teleop_active = False
        self._grip_sent = {"rh": None, "lh": None}
        self._last_target: dict[str, list[float]] = {}
        self._last_head_target: list[float] | None = None
        self._grip_level = {"rh": 0.0, "lh": 0.0}
        self._last_touch: list[float] = []
        self._pose_reader: GzPoseReader | None = None
        self._contact_reader: GzContactReader | None = None
        self._warned: set[str] = set()
        self._state_layout: tuple[int, int, dict[str, int]] | None = None
        self._rpc_enabled = False
        self.metrics_meta: dict[str, Any] = {}
        self._cmd_seq = {"rh": 0, "lh": 0, "head": 0}
        self._cam_stamp: dict[str, tuple[int, float]] = {}
        self._pose_stamp: dict[str, float] = {}

        self.actuator_list = []
        for part in self.config.actuators_to_use:
            self.actuator_list.extend(self.config.actuators_to_use[part])

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {actuator_name: float for actuator_name in self.actuator_list}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            name: (*self._camera_hw(name), 3) for name in self.config.camera_ports.keys()
        }

    def _camera_hw(self, name: str) -> tuple[int, int]:
        return tuple(self.config.camera_shapes.get(name, (240, 320)))

    @property
    def observation_features(self) -> dict[str, type]:
        features = {**self._motors_ft}

        for name in self.actuator_list:
            features[f"{name}_vel"] = float

        sides = self._arm_sides()

        for side in sides:
            for ax in ("x", "y", "z"):
                features[f"{side}_ee_pos_{ax}"] = float
            for ax in ("w", "x", "y", "z"):
                features[f"{side}_ee_quat_{ax}"] = float
        if self.config.use_gaze:
            for ax in ("w", "x", "y", "z"):
                features[f"head_ee_quat_{ax}"] = float

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

        for side in self._touch_sides():
            for finger in TOUCH_FINGER_SLOT:
                features[f"{side}_touch_{finger}"] = float

        for i in range(len(self.config.object_bodies)):
            for ax in ("x", "y", "z"):
                features[f"obj{i}_pos_{ax}"] = float
            for ax in ("w", "x", "y", "z"):
                features[f"obj{i}_quat_{ax}"] = float

        for name in self.actuator_list:
            features[f"{name}_torque"] = float

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

    def _arm_sides(self) -> list[str]:
        return [side for side, arms in (("rh", ("right", "both")), ("lh", ("left", "both")))
                if self.config.control_arms in arms]

    def _touch_sides(self) -> list[str]:
        sides = self._arm_sides()
        return sides if self.config.touch_source == "gazebo" else [s for s in sides if s == "rh"]

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            print(f"[WARN] {msg}")

    def _layout_for(self, n: int) -> tuple[int, dict[str, int]]:
        if self._state_layout and self._state_layout[0] == n:
            return self._state_layout[1], self._state_layout[2]
        full = load_control_config(self.config.config_path).get("actuators", {})
        layouts = {}
        for arms in (("right_arm",), ("left_arm",), ("left_arm", "right_arm"), ()):
            names = [j for part in ("torso", *arms, "head") for j in full.get(part, [])]
            layouts.setdefault(len(names), names)
        layouts[len(self.actuator_list)] = list(self.actuator_list)
        block = n // 3 if n % 3 == 0 and n // 3 in layouts else n
        names = layouts.get(block)
        missing = [j for j in self.actuator_list if names is None or j not in names]
        if missing:
            raise ConnectionError(
                f"/teleop/robot_state:o sends {n} values, which does not match joints "
                f"({len(self.actuator_list)}, control_arms={self.config.control_arms})."
            )
        index = {j: i for i, j in enumerate(names)}
        self._state_layout = (n, block, index)
        return block, index

    GRIP_STEP = 0.05

    def _command_grip(self, side: str, value: float, send_rpc) -> None:
        if self.config.gripper_legacy:
            if value <= 0.1:
                val = 0.0
            elif 0.4 <= value <= 0.6:
                val = 1.0
            else:
                return
        else:
            val = float(np.clip(value, 0.0, 1.0))
        self._grip_level[side] = val
        step = round(val / self.GRIP_STEP) * self.GRIP_STEP
        if self._grip_sent[side] is not None and abs(step - self._grip_sent[side]) < 1e-6:
            return
        word = "right" if side == "rh" else "left"
        if send_rpc(f"hand_{word} {step:.2f}"):
            self._grip_sent[side] = step
        elif self.ports.get("hand_cmd") is not None and step in (0.0, 1.0):
            cmd = self.yarp.Bottle()
            cmd.addString(f"{'open' if step == 0.0 else 'close'}_{word}")
            self.ports["hand_cmd"].write(cmd)
            self._grip_sent[side] = step

    def reset_command_state(self) -> None:
        self._last_target.clear()
        self._last_head_target = None
        self._grip_level = {"rh": 0.0, "lh": 0.0}
        self._grip_sent = {"rh": None, "lh": None}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        yarp = self.yarp
        yarp.Network.init()
        prefix = "/lerobot_client"

        self.ports["state"] = yarp.BufferedPortBottle()
        self.ports["state"].open(f"{prefix}/state:i")
        if not yarp.Network.connect(self.config.remote_state_port, f"{prefix}/state:i"):
            raise ConnectionError(f"Failed to connect robot_state port: {self.config.remote_state_port}")

        if self.config.control_arms in ["right", "both"]:
            self.ports["rh_out"] = yarp.Port()
            self.ports["rh_out"].open(f"{prefix}/rh_target:o")
            if not yarp.Network.connect(f"{prefix}/rh_target:o", self.config.remote_rh_port):
                raise ConnectionError("Failed to connect right hand target port.")

        if self.config.control_arms in ["left", "both"]:
            self.ports["lh_out"] = yarp.Port()
            self.ports["lh_out"].open(f"{prefix}/lh_target:o")
            if not yarp.Network.connect(f"{prefix}/lh_target:o", self.config.remote_lh_port):
                raise ConnectionError("Failed to connect left hand target port.")

        if self.config.use_gaze:
            self.ports["head_out"] = yarp.Port()
            self.ports["head_out"].open(f"{prefix}/head_target:o")
            if not yarp.Network.connect(f"{prefix}/head_target:o", self.config.remote_head_port):
                raise ConnectionError(f"Failed to connect head target port: {self.config.remote_head_port}")

            p_h = yarp.BufferedPortBottle()
            p_h.open(f"{prefix}/head_current_quat:i")
            if yarp.Network.connect("/teleop/head_current_quat:o", f"{prefix}/head_current_quat:i"):
                self.ports["head_current_quat"] = p_h
            else:
                p_h.close()

        self.ports["hand_cmd"] = yarp.Port()
        self.ports["hand_cmd"].open(f"{prefix}/hand_cmd:o")
        if not yarp.Network.connect(f"{prefix}/hand_cmd:o", self.config.remote_hand_cmd_port):
            self.ports["hand_cmd"].close()
            self.ports["hand_cmd"] = None

        self.ports["rpc"] = yarp.RpcClient()
        self.ports["rpc"].open(f"{prefix}/rpc:o")
        self._rpc_enabled = bool(yarp.Network.connect(f"{prefix}/rpc:o", self.config.remote_rpc_port))

        ack = yarp.BufferedPortBottle()
        ack.open(f"{prefix}/cmd_ack:i")
        ack.setStrict(True)
        if yarp.Network.connect("/teleop/cmd_ack:o", f"{prefix}/cmd_ack:i"):
            self.ports["cmd_ack"] = ack
        else:
            ack.close()

        connected_camera_ports = {}
        for name, remote_port in list(self.config.camera_ports.items()):
            if not yarp.Network.exists(remote_port):
                print(f"[iCub] Camera '{name}' source port '{remote_port}' not found on YARP network (skipping).")
                continue

            port = yarp.BufferedPortImageRgb()
            local_port_name = f"{prefix}/cam/{name}:i"
            if not port.open(local_port_name):
                print(f"[iCub] Warning: Failed to open local port {local_port_name} for camera '{name}'.")
                continue

            if not yarp.Network.connect(remote_port, local_port_name, "fast_tcp", False):
                port.close()
                print(f"[iCub] Warning: Failed to connect camera '{name}' ({remote_port} -> {local_port_name}).")
                continue

            h, w = self._camera_hw(name)
            self.np_arrays[name] = np.zeros((h, w, 3), dtype=np.uint8)
            self.img_buffers[name] = yarp.ImageRgb()
            self.img_buffers[name].resize(w, h)
            self.img_buffers[name].setExternal(
                self.np_arrays[name].data,
                self.np_arrays[name].shape[1],
                self.np_arrays[name].shape[0],
            )
            self.img_ports[name] = port
            connected_camera_ports[name] = remote_port
            print(f"[iCub] Connected camera '{name}': {remote_port} -> {local_port_name}")

        self.config.camera_ports = connected_camera_ports

        if self.img_ports:
            deadline = time.time() + CAMERA_FIRST_FRAME_TIMEOUT_S
            for name in list(self.img_ports.keys()):
                while name not in self._cam_last_t and time.time() < deadline:
                    self._read_camera(name)
                    time.sleep(0.01)

        self._current_pose = {}
        for side in ("rh", "lh"):
            if self.config.control_arms in (["right", "both"] if side == "rh" else ["left", "both"]):
                p = yarp.BufferedPortBottle()
                p.open(f"{prefix}/{side}_current_pose:i")
                if yarp.Network.connect(f"/teleop/{side}_current_pose:o", f"{prefix}/{side}_current_pose:i"):
                    self.ports[f"{side}_current_pose"] = p
                else:
                    p.close()

        if self.config.use_tactile and self.config.touch_source == "gazebo":
            keys = [f"{side}_{finger}" for side in self._touch_sides() for finger in TOUCH_FINGER_SLOT]
            self._contact_reader = GzContactReader(keys)
            self._contact_reader.start()
        elif self.config.use_tactile and "rh" in self._arm_sides():
            p = yarp.BufferedPortBottle()
            p.open(f"{prefix}/touch:i")
            if yarp.Network.connect(self.config.remote_touch_port, f"{prefix}/touch:i"):
                self.ports["touch"] = p
            else:
                p.close()

        if self.config.use_object_pose:
            self._pose_reader = GzPoseReader(self.config.object_bodies, self.config.gazebo_root_xyz)
            self._pose_reader.start()

        self._connected = True
        print("[iCub] Robot Connected via YARP")

        if calibrate and not self.is_calibrated:
            self.calibrate()

    def calibrate(self) -> None:
        self._calibrated = True

    def configure(self) -> None:
        pass

    def send_rpc_command(self, cmd: str) -> str | None:
        if not self._rpc_enabled or "rpc" not in self.ports:
            return None
        request = self.yarp.Bottle()
        reply = self.yarp.Bottle()
        request.fromString(cmd)
        if self.ports["rpc"].write(request, reply):
            return reply.toString()
        return None

    def start_teleop(self) -> bool:
        if not self._connected or not self._rpc_enabled:
            print("[iCub] Cannot start teleop: robot not connected or RPC unavailable.")
            return False
        reply = self.send_rpc_command("start_teleop")
        print(f"[iCub] Sent 'start_teleop' -> Reply: {reply}")
        # Impedance only exists with the ID (torque) controller, and only when it was asked for
        if getattr(self.config, "controller_type", "ik") == "id":
            val = 1 if self.config.impedance_mode else 0
            imp_reply = self.send_rpc_command(f"set_impedance {val}")
            print(f"[iCub] Sent 'set_impedance {val}' -> Reply: {imp_reply}")
        self.is_teleop_active = True
        return True

    def stop_teleop(self) -> bool:
        if not self._connected or not self._rpc_enabled:
            return False
        reply = self.send_rpc_command("stop_teleop")
        print(f"[iCub] Sent 'stop_teleop' -> Reply: {reply}")
        self.is_teleop_active = False
        return True

    def go_home(self) -> bool:
        if not self._connected or not self._rpc_enabled:
            return False
        self.is_teleop_active = False
        reply = self.send_rpc_command("go_home")
        print(f"[iCub] Sent 'go_home' -> Reply: {reply}")
        return True

    def toggle_impedance(self) -> bool:
        if not self._connected or not self._rpc_enabled:
            return False
        if getattr(self.config, "controller_type", "ik") != "id":
            print("[Teleop] Modo impedancia solo disponible con --controller id")
            return False
        self.config.impedance_mode = not getattr(self.config, "impedance_mode", False)
        val = "1" if self.config.impedance_mode else "0"
        reply = self.send_rpc_command(f"set_impedance {val}")
        print(f"[iCub] Modo impedancia: {'ACTIVADO' if self.config.impedance_mode else 'DESACTIVADO'} -> {reply}")
        return self.config.impedance_mode

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError("iCub robot is not connected.")

        obs = {}
        meta = self.metrics_meta = {}
        t0 = time.time()
        state_bot = self.ports["state"].read(True)
        t_state = time.time()
        meta["state_wait_ms"] = (t_state - t0) * 1000.0

        if state_bot is None:
            raise ConnectionError("Critical: Failed to read robot state (YARP port might be closed).")

        n = state_bot.size()
        block, index = self._layout_for(n)
        vals = [state_bot.get(i).asFloat64() for i in range(n)]

        for name in self.actuator_list:
            obs[name] = torch.tensor(vals[index[name]], dtype=torch.float32)

        for name in self.actuator_list:
            obs[f"{name}_vel"] = torch.tensor(vals[block + index[name]], dtype=torch.float32)

        for name in self.actuator_list:
            tau = vals[2 * block + index[name]]
            if math.isnan(tau):
                tau = 0.0
            obs[f"{name}_torque"] = torch.tensor(tau, dtype=torch.float32)

        now = time.time()
        for name in self.img_ports:
            meta[f"cam_{name}_new"] = int(self._read_camera(name))
            obs[name] = self.np_arrays[name].copy()
            if name in self._cam_last_t:
                meta[f"cam_{name}_age_ms"] = (time.time() - self._cam_last_t[name]) * 1000.0

        stamp = self.yarp.Stamp()
        for side in ("rh", "lh"):
            port = self.ports.get(f"{side}_current_pose")
            if port is not None and (b := port.read(False)) is not None and b.size() >= 7:
                self._current_pose[side] = [b.get(i).asFloat64() for i in range(7)]
                if port.getEnvelope(stamp) and stamp.isValid():
                    self._pose_stamp[side] = stamp.getTime()
            if side in self._current_pose:
                obs[f"{side}_current_pose"] = list(self._current_pose[side])

        head_port = self.ports.get("head_current_quat")
        if head_port is not None and (b := head_port.read(False)) is not None and b.size() >= 4:
            self._current_pose["head"] = [b.get(i).asFloat64() for i in range(4)]

        self._add_extra_observations(obs)
        if self.config.canonical_quats:
            canonicalize_quats(obs)
        return obs

    def _add_extra_observations(self, obs: dict[str, Any]) -> None:
        def put(prefix: str, axes: str, values) -> None:
            for ax, val in zip(axes, values):
                obs[f"{prefix}_{ax}"] = torch.tensor(float(val), dtype=torch.float32)

        sides = self._arm_sides()
        for side in sides:
            pose = self._current_pose.get(side) or [0.0] * 7
            put(f"{side}_ee_pos", "xyz", pose[:3])
            put(f"{side}_ee_quat", "wxyz", pose[3:7])

        if self.config.use_gaze:
            head_quat = self._current_pose.get("head") or [1.0, 0.0, 0.0, 0.0]
            put("head_ee_quat", "wxyz", head_quat[:4])

        for side in sides:
            target = self._last_target.get(side) or self._current_pose.get(side) or [0.0] * 7
            put(f"{side}_target_pos", "xyz", target[:3])
            put(f"{side}_target_quat", "wxyz", target[3:7])
            obs[f"{side}_gripper"] = torch.tensor(self._grip_level[side], dtype=torch.float32)

        if self.config.use_gaze:
            head_target = self._last_head_target or self._current_pose.get("head") or [1.0, 0.0, 0.0, 0.0]
            put("head_target_quat", "wxyz", head_target[:4])

        if self.config.use_tactile and self._contact_reader is not None:
            for side in self._touch_sides():
                for finger in TOUCH_FINGER_SLOT:
                    touching = self._contact_reader.touching(f"{side}_{finger}")
                    obs[f"{side}_touch_{finger}"] = torch.tensor(float(touching), dtype=torch.float32)
        elif self.config.use_tactile and "rh" in sides:
            port = self.ports.get("touch")
            if port is not None and (b := port.read(False)) is not None:
                self._last_touch = [b.get(i).asFloat64() for i in range(b.size())]
            for finger, slot in TOUCH_FINGER_SLOT.items():
                taxels = self._last_touch[slot * TAXELS_PER_FINGER : (slot + 1) * TAXELS_PER_FINGER]
                touching = bool(taxels) and sum(taxels) / len(taxels) >= self.config.touch_threshold
                obs[f"rh_touch_{finger}"] = torch.tensor(float(touching), dtype=torch.float32)

        if self.config.use_object_pose:
            for i, body in enumerate(self.config.object_bodies):
                pose = self._pose_reader.get(body) if self._pose_reader else None
                if pose is None:
                    pose = ([0.0] * 3, [0.0] * 4)
                put(f"obj{i}_pos", "xyz", pose[0])
                put(f"obj{i}_quat", "wxyz", pose[1])

    def _read_camera(self, name: str) -> bool:
        img_yarp = self.img_ports[name].read(False)
        if img_yarp is None:
            return False
        h_in, w_in = img_yarp.height(), img_yarp.width()
        if (h_in, w_in) == self.np_arrays[name].shape[:2]:
            self.img_buffers[name].copy(img_yarp)
        else:
            self._copy_resized(name, img_yarp, h_in, w_in)
        self._cam_last_t[name] = time.time()
        return True

    def _copy_resized(self, name: str, img_yarp, h_in: int, w_in: int) -> None:
        import cv2

        buf = self._native_bufs.get(name)
        if buf is None or buf[0].shape[:2] != (h_in, w_in):
            arr = np.zeros((h_in, w_in, 3), dtype=np.uint8)
            yimg = self.yarp.ImageRgb()
            yimg.resize(w_in, h_in)
            yimg.setExternal(arr.data, w_in, h_in)
            self._native_bufs[name] = buf = (arr, yimg)
        arr, yimg = buf
        yimg.copy(img_yarp)
        h, w = self.np_arrays[name].shape[:2]
        cv2.resize(arr, (w, h), dst=self.np_arrays[name], interpolation=cv2.INTER_AREA)

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError("iCub robot is not connected.")

        yarp = self.yarp

        def action_value(*keys):
            for key in keys:
                if key in action:
                    return action[key]
            raise KeyError(f"None of keys found in action: {keys}")

        def to_float(val):
            return val.item() if isinstance(val, torch.Tensor) else float(val)

        def send_rpc(cmd_string: str) -> bool:
            if not self._rpc_enabled:
                return False
            cmd = yarp.Bottle()
            reply = yarp.Bottle()
            cmd.fromString(cmd_string)
            try:
                return bool(self.ports["rpc"].write(cmd, reply))
            except Exception:
                return False

        if self.config.control_arms in ["right", "both"]:
            pos_rh = [to_float(action_value("rh_pos_x", "rh_x")), to_float(action_value("rh_pos_y", "rh_y")), to_float(action_value("rh_pos_z", "rh_z"))]
            quat_rh = [to_float(action_value("rh_quat_w", "rh_qw")), to_float(action_value("rh_quat_x", "rh_qx")), to_float(action_value("rh_quat_y", "rh_qy")), to_float(action_value("rh_quat_z", "rh_qz"))]
            self._last_target["rh"] = pos_rh + quat_rh
            if "rh_out" in self.ports:
                bottle_rh = yarp.Bottle()
                for v in pos_rh:
                    bottle_rh.addFloat64(v)
                for v in quat_rh:
                    bottle_rh.addFloat64(v)
                self._write_target("rh", bottle_rh)

            if "rh_gripper" in action:
                self._command_grip("rh", to_float(action["rh_gripper"]), send_rpc)

        if self.config.control_arms in ["left", "both"]:
            pos_lh = [to_float(action_value("lh_pos_x", "lh_x")), to_float(action_value("lh_pos_y", "lh_y")), to_float(action_value("lh_pos_z", "lh_z"))]
            quat_lh = [to_float(action_value("lh_quat_w", "lh_qw")), to_float(action_value("lh_quat_x", "lh_qx")), to_float(action_value("lh_quat_y", "lh_qy")), to_float(action_value("lh_quat_z", "lh_qz"))]
            self._last_target["lh"] = pos_lh + quat_lh
            if "lh_out" in self.ports:
                bottle_lh = yarp.Bottle()
                for v in pos_lh:
                    bottle_lh.addFloat64(v)
                for v in quat_lh:
                    bottle_lh.addFloat64(v)
                self._write_target("lh", bottle_lh)

            if "lh_gripper" in action:
                self._command_grip("lh", to_float(action["lh_gripper"]), send_rpc)

        if self.config.use_gaze and any(k in action for k in ("head_quat_w", "gaze_w")):
            quat_head = [
                to_float(action_value("head_quat_w", "gaze_w")),
                to_float(action_value("head_quat_x", "gaze_x")),
                to_float(action_value("head_quat_y", "gaze_y")),
                to_float(action_value("head_quat_z", "gaze_z")),
            ]
            self._last_head_target = quat_head
            if "head_out" in self.ports:
                bottle_head = yarp.Bottle()
                for v in quat_head:
                    bottle_head.addFloat64(v)
                self._write_target("head", bottle_head)

        if self.config.canonical_quats:
            canonicalize_quats(action)
        return action

    def _write_target(self, target_name: str, bottle) -> None:
        self._cmd_seq[target_name] = self._cmd_seq.get(target_name, 0) + 1
        port = self.ports.get(f"{target_name}_out")
        if port is not None:
            t_send = time.time()
            port.setEnvelope(self.yarp.Stamp(self._cmd_seq[target_name], t_send))
            port.write(bottle)

    def disconnect(self) -> None:
        if not self._connected:
            return
        if getattr(self, "is_teleop_active", False):
            try:
                self.stop_teleop()
            except Exception:
                pass
        yarp = self.yarp
        if self._pose_reader is not None:
            self._pose_reader.stop()
            self._pose_reader = None
        if self._contact_reader is not None:
            self._contact_reader.stop()
            self._contact_reader = None
        for name, port in self.ports.items():
            if port is not None:
                port.interrupt()
                port.close()
        for name, port in self.img_ports.items():
            port.interrupt()
            port.close()
        self.img_buffers.clear()
        self.np_arrays.clear()
        yarp.Network.fini()
        self._calibrated = False
        self._connected = False

