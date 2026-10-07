from typing import Any
import math
import time
import torch
import numpy as np

try:
    from lerobot.robots.robot import Robot
except Exception:
    class Robot:
        def __init__(self, config):
            self.config = config

try:
    from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
except Exception:
    class DeviceNotConnectedError(RuntimeError):
        pass

    class DeviceAlreadyConnectedError(RuntimeError):
        pass

from .config_icub import iCubConfig

# Espera máxima al primer frame de cada cámara en connect(), y antigüedad a partir de
# la cual se avisa que una cámara dejó de mandar frames.
CAMERA_FIRST_FRAME_TIMEOUT_S = 5.0
CAMERA_STALE_S = 1.0

class iCub(Robot):
    config_class = iCubConfig
    name = "icub_remote"

    def __init__(self, config: iCubConfig):
        super().__init__(config)
        try:
            import yarp
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Python module 'yarp' is required for iCub robot plugin. "
                "Install YARP Python bindings in this environment before using --robot.type=lerobot_robot_icubrg."
            ) from exc

        self.yarp = yarp
        self.config = config
        
        
        
        self.ports = {}
        self.img_ports = {}
        self.img_buffers = {}
        self.np_arrays = {}
        # Buffers a la resolución nativa de cada cámara cuando difiere de camera_shapes
        # (p.ej. el robot real publica 640x480 y Gazebo 320x240): se redimensiona.
        self._native_bufs: dict[str, tuple] = {}
        # Último frame recibido por cámara y último aviso de cámara sin frames
        self._cam_last_t: dict[str, float] = {}
        self._cam_stale_warn_t = 0.0
        self._connected = False
        self._calibrated = True 
        self._prev_r_cmd = None
        self._prev_l_cmd = None
        self._warned_missing_gaze = False
        self._warned_extra_joints = False
        self._rpc_enabled = False
        # Métricas por frame que lee metrics/frame_metrics.py (timestamps YARP de estado
        # y cámaras, confirmaciones de comandos). Se rehace en cada get_observation.
        self.metrics_meta: dict[str, Any] = {}
        self._cmd_seq = {"rh": 0, "lh": 0}
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
        # Dimensiones esperadas (H, W, C): es el formato que LeRobot exige para imágenes
        return {
            name: (*self._camera_hw(name), 3) for name in self.config.camera_ports.keys()
        }

    def _camera_hw(self, name: str) -> tuple[int, int]:
        return tuple(self.config.camera_shapes.get(name, (240, 320)))

    @property
    def observation_features(self) -> dict[str, type]:     
        features = {**self._motors_ft, **self._cameras_ft}

        if getattr(self.config, "use_joint_vel", False):
            for name in self.actuator_list:
                features[f"vel_{name}"] = float

        if getattr(self.config, "use_joint_torque", False):
            for name in self.actuator_list:
                features[f"torque_{name}"] = float

        if getattr(self.config, "use_external_wrench", False):
            if self.config.control_arms in ["left", "both"]:
                for i, comp in enumerate(["fx", "fy", "fz", "tx", "ty", "tz"]):
                    features[f"lh_wrench_{comp}"] = float
            if self.config.control_arms in ["right", "both"]:
                for i, comp in enumerate(["fx", "fy", "fz", "tx", "ty", "tz"]):
                    features[f"rh_wrench_{comp}"] = float

        if getattr(self.config, "use_eef_pose", False):
            if self.config.control_arms in ["left", "both"]:
                for comp in ["x", "y", "z", "qw", "qx", "qy", "qz"]:
                    features[f"lh_eef_{comp}"] = float
            if self.config.control_arms in ["right", "both"]:
                for comp in ["x", "y", "z", "qw", "qx", "qy", "qz"]:
                    features[f"rh_eef_{comp}"] = float

        if getattr(self.config, "use_tactile", False):
            if self.config.control_arms in ["left", "both"]:
                features["tactile_left"] = float
            if self.config.control_arms in ["right", "both"]:
                features["tactile_right"] = float

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
        if getattr(self.config, "use_gaze", False):
            features["gaze_x"] = float
            features["gaze_y"] = float
            features["gaze_z"] = float
        return features

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Connect to robot ports
        yarp = self.yarp
        yarp.Network.init()
        prefix = "/lerobot_client"
        
        # 1. Puertos de Control
        self.ports["state"] = yarp.BufferedPortBottle()
        self.ports["state"].open(f"{prefix}/state:i")
        if not yarp.Network.connect(self.config.remote_state_port, f"{prefix}/state:i"):
            raise ConnectionError("Failed to connect robot_state ports.")

        if self.config.control_arms in ["right", "both"]:
            self.ports["rh_out"] = yarp.Port()
            self.ports["rh_out"].open(f"{prefix}/rh_target:o")
            if not yarp.Network.connect(f"{prefix}/rh_target:o", self.config.remote_rh_port):
                raise ConnectionError("Failed to connect right hand target ports.")

        if self.config.control_arms in ["left", "both"]:
            self.ports["lh_out"] = yarp.Port()
            self.ports["lh_out"].open(f"{prefix}/lh_target:o")
            if not yarp.Network.connect(f"{prefix}/lh_target:o", self.config.remote_lh_port):
                raise ConnectionError("Failed to connect left hand target ports.")
            
        self.ports["hand_cmd"] = yarp.Port()
        self.ports["hand_cmd"].open(f"{prefix}/hand_cmd:o")
        if not yarp.Network.connect(f"{prefix}/hand_cmd:o", self.config.remote_hand_cmd_port):
            self.ports["hand_cmd"].close()
            self.ports["hand_cmd"] = None
            print("[WARN] Hand command port not connected. Gripper commands will be ignored.")

        if self.config.use_gaze:
            self.ports["gaze_target"] = yarp.Port()
            self.ports["gaze_target"].open(f"{prefix}/gaze_target:o")
            if not yarp.Network.connect(f"{prefix}/gaze_target:o", self.config.remote_gaze_port):
                raise ConnectionError("Failed to connect gaze target ports.")

        # RPC client (icub_teleop state machine command interface)
        self.ports["rpc"] = yarp.RpcClient()
        self.ports["rpc"].open(f"{prefix}/rpc:o")
        if yarp.Network.connect(f"{prefix}/rpc:o", self.config.remote_rpc_port):
            self._rpc_enabled = True
        else:
            self._rpc_enabled = False
            print(f"[WARN] RPC port not connected: {self.config.remote_rpc_port}. Falling back to legacy target ports.")

        # Confirmaciones de targets (teleop_module_sm → /teleop/cmd_ack:o). Solo métricas:
        # sin este puerto (módulo antiguo) el teleop funciona igual.
        ack = yarp.BufferedPortBottle()
        ack.open(f"{prefix}/cmd_ack:i")
        ack.setStrict(True)  # encolar todas: cada una es una muestra de latencia
        if yarp.Network.connect("/teleop/cmd_ack:o", f"{prefix}/cmd_ack:i"):
            self.ports["cmd_ack"] = ack
        else:
            ack.close()
            print("[WARN] /teleop/cmd_ack:o not connected: no command-latency metrics.")

        # 2. Puertos de Cámara
        for name, remote_port in self.config.camera_ports.items():
            port = yarp.BufferedPortImageRgb()
            local_port_name = f"{prefix}/cam/{name}:i"
            port.open(local_port_name)
            if not yarp.Network.connect(remote_port, local_port_name, 'fast_tcp', False):
                raise ConnectionError(
                    f"Failed to connect camera '{name}' ({remote_port}). Is Gazebo/the robot "
                    f"publishing it? Check with `yarp name list`, or record with --no-camera."
                )

            h, w = self._camera_hw(name)
            self.np_arrays[name] = np.zeros((h, w, 3), dtype=np.uint8)
            self.img_buffers[name] = yarp.ImageRgb()
            self.img_buffers[name].resize(w,h)
            self.img_buffers[name].setExternal(
                self.np_arrays[name].data, 
                self.np_arrays[name].shape[1],
                self.np_arrays[name].shape[0]
            )
            self.img_ports[name] = port

        # Espera el primer frame de cada cámara (lectura bloqueante solo aquí, con
        # timeout): así el dataset no arranca con imágenes negras.
        deadline = time.time() + CAMERA_FIRST_FRAME_TIMEOUT_S
        for name in self.img_ports:
            while name not in self._cam_last_t and time.time() < deadline:
                self._read_camera(name)
                time.sleep(0.01)
            if name not in self._cam_last_t:
                print(f"[WARN] Camera '{name}' sent no frame in {CAMERA_FIRST_FRAME_TIMEOUT_S:.0f} s; "
                      f"recording black frames until it does.")

        # Pose actual de cada mano que publica teleop_module_sm (x y z qw qx qy qz, frame
        # root). No va al dataset: el teleoperador la usa para arrancar los mocaps donde
        # está la mano real y evitar el salto inicial.
        self._current_pose = {}
        for side in ("rh", "lh"):
            if self.config.control_arms in (["right", "both"] if side == "rh" else ["left", "both"]):
                p = yarp.BufferedPortBottle()
                p.open(f"{prefix}/{side}_current_pose:i")
                if yarp.Network.connect(f"/teleop/{side}_current_pose:o", f"{prefix}/{side}_current_pose:i"):
                    self.ports[f"{side}_current_pose"] = p
                else:
                    p.close()
                    print(f"[WARN] /teleop/{side}_current_pose:o not connected: mocaps start at the XML pose.")

        # 3. Puertos opcionales de observación
        def _open_optional(key, local_suffix, remote_port):
            """Abre un BufferedPortBottle opcional. Si falla, lo omite con un warning."""
            p = yarp.BufferedPortBottle()
            local_name = f"{prefix}/{local_suffix}"
            p.open(local_name)
            if not yarp.Network.connect(remote_port, local_name):
                p.close()
                print(f"[WARN] Could not connect optional port '{remote_port}' → skipped.")
                return None
            return p

        if getattr(self.config, "use_joint_vel", False):
            if self.config.control_arms in ["left", "both"]:
                self.ports["vel_l"] = _open_optional("vel_l", "vel_l:i", self.config.remote_vel_port_l)
            if self.config.control_arms in ["right", "both"]:
                self.ports["vel_r"] = _open_optional("vel_r", "vel_r:i", self.config.remote_vel_port_r)

        if getattr(self.config, "use_joint_torque", False):
            if self.config.control_arms in ["left", "both"]:
                self.ports["torque_l"] = _open_optional("torque_l", "torque_l:i", self.config.remote_torque_port_l)
            if self.config.control_arms in ["right", "both"]:
                self.ports["torque_r"] = _open_optional("torque_r", "torque_r:i", self.config.remote_torque_port_r)

        if getattr(self.config, "use_external_wrench", False):
            if self.config.control_arms in ["left", "both"]:
                self.ports["wrench_l"] = _open_optional("wrench_l", "wrench_l:i", self.config.remote_wrench_port_l)
            if self.config.control_arms in ["right", "both"]:
                self.ports["wrench_r"] = _open_optional("wrench_r", "wrench_r:i", self.config.remote_wrench_port_r)

        if getattr(self.config, "use_eef_pose", False):
            if self.config.control_arms in ["left", "both"]:
                self.ports["eef_l"] = _open_optional("eef_l", "eef_l:i", self.config.remote_eef_port_l)
            if self.config.control_arms in ["right", "both"]:
                self.ports["eef_r"] = _open_optional("eef_r", "eef_r:i", self.config.remote_eef_port_r)

        if getattr(self.config, "use_tactile", False):
            if self.config.control_arms in ["left", "both"]:
                self.ports["tactile_l"] = _open_optional("tactile_l", "tactile_l:i", self.config.remote_tactile_port_l)
            if self.config.control_arms in ["right", "both"]:
                self.ports["tactile_r"] = _open_optional("tactile_r", "tactile_r:i", self.config.remote_tactile_port_r)

        self._connected = True
        print("iCub Robot Connected via YARP")

        self.configure()
        if calibrate and not self.is_calibrated:
            self.calibrate()

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def send_rpc_command(self, cmd: str) -> str | None:
        """Envía un comando RPC al teleop_module_sm y retorna la respuesta."""
        if not self._rpc_enabled or "rpc" not in self.ports:
            print(f"[RPC] Skipped (not connected): {cmd}")
            return None
        request = self.yarp.Bottle()
        reply = self.yarp.Bottle()
        request.fromString(cmd)
        if self.ports["rpc"].write(request, reply):
            return reply.toString()
        return None

    def calibrate(self) -> None:
        self._calibrated = True


    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError("iCub robot is not connected.")

        obs = {}
        meta = self.metrics_meta = {}
        # Leer Estado Articular (BLOQUEANTE)
        t0 = time.time()
        state_bot = self.ports["state"].read(True)
        t_state = time.time()
        meta["state_wait_ms"] = (t_state - t0) * 1000.0
        stamp = self.yarp.Stamp()
        if state_bot is not None and self.ports["state"].getEnvelope(stamp) and stamp.isValid():
            meta["state_seq"] = stamp.getCount()
            meta["state_stamp"] = stamp.getTime()
            meta["state_age_ms"] = (t_state - stamp.getTime()) * 1000.0
        # Validación de seguridad: Si el puerto se cierra o hay error grave, podría retornar None
        if state_bot is None:
             raise ConnectionError("Critical: Failed to read robot state (YARP port might be closed).")
        # Convertir Bottle a lista
        n = state_bot.size()
        expected = len(self.actuator_list)
        if n < expected:
            raise ValueError(f"Joint mismatch: Received {n}, expected at least {expected}")
        if n > expected and not self._warned_extra_joints:
            print(f"[WARN] Joint state has extra values ({n}>{expected}). Trimming to expected actuator list.")
            self._warned_extra_joints = True
        vals = [state_bot.get(i).asFloat64() for i in range(n)]
        # Asignación rapida
        for i, name in enumerate(self.actuator_list):
            obs[name] = torch.tensor(vals[i], dtype=torch.float32)

        # Leer Camaras (NO bloqueante): si no llegó un frame nuevo se repite el último.
        # Con read(True) el loop de teleop (visor + targets) se congelaba cada vez que
        # las cámaras se atrasaban.
        now = time.time()
        for name in self.img_ports:
            meta[f"cam_{name}_new"] = int(self._read_camera(name))
            # HWC uint8 (copia: el buffer se reutiliza en el siguiente frame)
            obs[name] = self.np_arrays[name].copy()
            # Frame que se graba: cuánto hace que llegó y su envelope (captura en origen)
            if name in self._cam_last_t:
                meta[f"cam_{name}_age_ms"] = (time.time() - self._cam_last_t[name]) * 1000.0
            if name in self._cam_stamp:
                seq, t_cap = self._cam_stamp[name]
                meta[f"cam_{name}_seq"], meta[f"cam_{name}_stamp"] = seq, t_cap
                # Antigüedad desde la captura: solo tiene sentido si la fuente usa el mismo
                # reloj que este PC (metrics_report.py lo comprueba antes de usarla).
                meta[f"cam_{name}_stamp_age_ms"] = (time.time() - t_cap) * 1000.0
        stale = [n for n in self.img_ports
                 if now - self._cam_last_t.get(n, 0.0) > CAMERA_STALE_S]
        if stale and now - self._cam_stale_warn_t > 5.0:
            print(f"[WARN] No new frames for >{CAMERA_STALE_S:.0f} s from camera(s) {stale}: "
                  f"repeating the last frame.")
            self._cam_stale_warn_t = now

        # Pose actual de las manos (solo feedback para el teleoperador, no es feature)
        for side in ("rh", "lh"):
            port = self.ports.get(f"{side}_current_pose")
            if port is not None and (b := port.read(False)) is not None and b.size() >= 7:
                self._current_pose[side] = [b.get(i).asFloat64() for i in range(7)]
                if port.getEnvelope(stamp) and stamp.isValid():
                    self._pose_stamp[side] = stamp.getTime()
            if side in self._current_pose:
                obs[f"{side}_current_pose"] = list(self._current_pose[side])
            if side in self._pose_stamp:
                meta[f"{side}_pose_stamp"] = self._pose_stamp[side]

        self._drain_acks(meta)

        # --- Observaciones opcionales ---

        def _read_bottle_floats(port_key: str) -> list[float] | None:
            """Lee una Bottle del puerto indicado y retorna lista de floats, o None si falla."""
            port = self.ports.get(port_key)
            if port is None:
                return None
            bottle = port.read(False)  # no bloqueante para puertos opcionales
            if bottle is None:
                return None
            return [bottle.get(i).asFloat64() for i in range(bottle.size())]

        def _axis_angle_to_quat(ax: float, ay: float, az: float, theta: float) -> tuple:
            """Convierte eje-ángulo (ax,ay,az,theta) a quaternion (qw,qx,qy,qz)."""
            s = math.sin(theta / 2.0)
            return (math.cos(theta / 2.0), ax * s, ay * s, az * s)

        # Joint Velocity
        if getattr(self.config, "use_joint_vel", False):
            # El estado del robot puede dar posición y velocidad juntos dependiendo del puerto.
            # Si hay puertos separados por brazo, los leemos y los asignamos por nombre.
            for side, port_key in [("l", "vel_l"), ("r", "vel_r")]:
                vals = _read_bottle_floats(port_key)
                if vals is not None:
                    # Los joints del lado corresponden al subgrupo del actuator_list
                    side_joints = [
                        name for name in self.actuator_list
                        if name.startswith(("l_" if side == "l" else "r_",
                                            "torso" if side == "l" else "__never__"))
                    ]
                    # Asignación por índice (el orden del puerto debe coincidir con actuator_list)
                    for i, val in enumerate(vals):
                        if i < len(self.actuator_list):
                            obs[f"vel_{self.actuator_list[i]}"] = torch.tensor(val, dtype=torch.float32)

        # Joint Torque
        if getattr(self.config, "use_joint_torque", False):
            for side, port_key in [("l", "torque_l"), ("r", "torque_r")]:
                vals = _read_bottle_floats(port_key)
                if vals is not None:
                    for i, val in enumerate(vals):
                        if i < len(self.actuator_list):
                            obs[f"torque_{self.actuator_list[i]}"] = torch.tensor(val, dtype=torch.float32)

        # External Wrench [Fx, Fy, Fz, Tx, Ty, Tz]
        if getattr(self.config, "use_external_wrench", False):
            comps = ["fx", "fy", "fz", "tx", "ty", "tz"]
            for prefix_key, port_key in [("lh", "wrench_l"), ("rh", "wrench_r")]:
                vals = _read_bottle_floats(port_key)
                if vals is not None:
                    for i, comp in enumerate(comps):
                        val = vals[i] if i < len(vals) else 0.0
                        obs[f"{prefix_key}_wrench_{comp}"] = torch.tensor(val, dtype=torch.float32)

        # EEF Pose: el Cartesian Controller publica [x y z ax ay az theta]
        if getattr(self.config, "use_eef_pose", False):
            for prefix_key, port_key in [("lh", "eef_l"), ("rh", "eef_r")]:
                vals = _read_bottle_floats(port_key)
                if vals is not None and len(vals) >= 7:
                    x, y, z, ax, ay, az, theta = vals[:7]
                    qw, qx, qy, qz = _axis_angle_to_quat(ax, ay, az, theta)
                    for comp, val in zip(["x","y","z","qw","qx","qy","qz"], [x,y,z,qw,qx,qy,qz]):
                        obs[f"{prefix_key}_eef_{comp}"] = torch.tensor(val, dtype=torch.float32)

        # Tactile (skin): cada taxel es un float en la Bottle
        if getattr(self.config, "use_tactile", False):
            for obs_key, port_key in [("tactile_left", "tactile_l"), ("tactile_right", "tactile_r")]:
                vals = _read_bottle_floats(port_key)
                if vals is not None:
                    obs[obs_key] = torch.tensor(vals, dtype=torch.float32)
                    max_touch = max(vals) if vals else 0
                    if max_touch > 5.0: # Ajusta este número si hay ruido en el sensor
                        print(f"[{obs_key}] Tacto detectado! Max valor: {max_touch:.2f}")

        return obs

    def _drain_acks(self, meta: dict) -> None:
        """Lee las confirmaciones de targets llegadas desde el frame anterior.

        transport_ms: del write del cliente a que el módulo lo lee (incluye la espera
        al siguiente ciclo del módulo); exec_ms: lo que tarda go_to_pose_async."""
        port = self.ports.get("cmd_ack")
        if port is None:
            return
        acks = []
        while port.getPendingReads() > 0:
            b = port.read(False)
            if b is None or b.size() < 6:
                break
            t_send, t_recv, t_done = (b.get(i).asFloat64() for i in (2, 3, 4))
            acks.append({
                "arm": b.get(0).asString(), "seq": b.get(1).asInt64(),
                "transport_ms": (t_recv - t_send) * 1000.0,
                "exec_ms": (t_done - t_recv) * 1000.0,
                "sent": b.get(5).asInt32(),
            })
        if acks:
            meta["acks"] = acks

    def _read_camera(self, name: str) -> bool:
        """Copia a np_arrays[name] el frame nuevo de la cámara, si llegó uno."""
        img_yarp = self.img_ports[name].read(False)
        if img_yarp is None:
            return False
        stamp = self.yarp.Stamp()
        if self.img_ports[name].getEnvelope(stamp) and stamp.isValid():
            self._cam_stamp[name] = (stamp.getCount(), stamp.getTime())
        h_in, w_in = img_yarp.height(), img_yarp.width()
        if (h_in, w_in) == self.np_arrays[name].shape[:2]:
            self.img_buffers[name].copy(img_yarp)
        else:
            # copy() a img_buffers reasignaría memoria fuera del buffer externo:
            # se copia a un buffer nativo y se redimensiona a camera_shapes, para
            # que el dataset tenga la misma forma en Gazebo y en el robot real.
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
            h, w = self.np_arrays[name].shape[:2]
            print(f"[iCub] Camera '{name}' sends {w_in}x{h_in}; resizing to {w}x{h} "
                  f"(set camera_shapes in the YAML to keep the native resolution).")
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
            raise KeyError(f"None of the keys found in action: {keys}")

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

        # Lógica Mano Derecha
        if self.config.control_arms in ["right", "both"]:
            pos_rh = [
                to_float(action_value("rh_pos_x", "rh_x")),
                to_float(action_value("rh_pos_y", "rh_y")),
                to_float(action_value("rh_pos_z", "rh_z")),
                ]
            quat_rh = [
                to_float(action_value("rh_quat_w", "rh_qw")),
                to_float(action_value("rh_quat_x", "rh_qx")),
                to_float(action_value("rh_quat_y", "rh_qy")),
                to_float(action_value("rh_quat_z", "rh_qz")),
            ]
            if "rh_out" in self.ports:
                bottle_rh = yarp.Bottle()
                for v in pos_rh: bottle_rh.addFloat64(v)
                for v in quat_rh: bottle_rh.addFloat64(v)
                self._write_target("rh", bottle_rh)
            elif self._rpc_enabled:
                send_rpc(
                    f"move_right_arm {pos_rh[0]:.6f} {pos_rh[1]:.6f} {pos_rh[2]:.6f} "
                    f"{quat_rh[0]:.6f} {quat_rh[1]:.6f} {quat_rh[2]:.6f} {quat_rh[3]:.6f}"
                )
            # Gripper Derecho
            if "rh_gripper" in action:
                r_cmd = to_float(action["rh_gripper"])
                if r_cmd == 0.0:     cmd_str = "open_right"
                elif r_cmd == 0.5:   cmd_str = "close_right"
                elif r_cmd == 1.0:   cmd_str = None
                else:                cmd_str = None
                # Se envía en cada flanco (el teleop manda un pulso 0.5/0.0 y vuelve a
                # 1.0=stop), así cerrar→cerrar también se reenvía.
                if cmd_str is not None and r_cmd != self._prev_r_cmd:
                    sent_gripper_rpc = send_rpc(cmd_str)
                    if (not sent_gripper_rpc) and self.ports.get("hand_cmd") is not None:
                        cmd = yarp.Bottle(); cmd.addString(cmd_str)
                        self.ports["hand_cmd"].write(cmd)
                    print(f"[Gripper] {cmd_str} -> {'rpc' if sent_gripper_rpc else 'NOT SENT'}", flush=True)
                self._prev_r_cmd = r_cmd

        # Lógica Mano Izquierda
        if self.config.control_arms in ["left", "both"]:
            pos_lh = [
                to_float(action_value("lh_pos_x", "lh_x")),
                to_float(action_value("lh_pos_y", "lh_y")),
                to_float(action_value("lh_pos_z", "lh_z")),
            ]
            quat_lh = [
                to_float(action_value("lh_quat_w", "lh_qw")),
                to_float(action_value("lh_quat_x", "lh_qx")),
                to_float(action_value("lh_quat_y", "lh_qy")),
                to_float(action_value("lh_quat_z", "lh_qz")),
            ]

            if "lh_out" in self.ports:
                bottle_lh = yarp.Bottle()
                for v in pos_lh: bottle_lh.addFloat64(v)
                for v in quat_lh: bottle_lh.addFloat64(v)
                self._write_target("lh", bottle_lh)
            elif self._rpc_enabled:
                send_rpc(
                    f"move_left_arm {pos_lh[0]:.6f} {pos_lh[1]:.6f} {pos_lh[2]:.6f} "
                    f"{quat_lh[0]:.6f} {quat_lh[1]:.6f} {quat_lh[2]:.6f} {quat_lh[3]:.6f}"
                )

            # Gripper Izquierdo
            if "lh_gripper" in action:
                l_cmd = to_float(action["lh_gripper"])
                if l_cmd == 0.0:     cmd_str = "open_left"
                elif l_cmd == 0.5:   cmd_str = "close_left"
                elif l_cmd == 1.0:   cmd_str = None
                else:                cmd_str = None
                # Se envía en cada flanco (el teleop manda un pulso 0.5/0.0 y vuelve a
                # 1.0=stop), así cerrar→cerrar también se reenvía.
                if cmd_str is not None and l_cmd != self._prev_l_cmd:
                    sent_gripper_rpc = send_rpc(cmd_str)
                    if (not sent_gripper_rpc) and self.ports.get("hand_cmd") is not None:
                        cmd = yarp.Bottle(); cmd.addString(cmd_str)
                        self.ports["hand_cmd"].write(cmd)
                    print(f"[Gripper] {cmd_str} -> {'rpc' if sent_gripper_rpc else 'NOT SENT'}", flush=True)
                self._prev_l_cmd = l_cmd
                    
        # Logica Gaze ctrl
        if getattr(self.config, "use_gaze", False):
            if not all(k in action for k in ("gaze_x", "gaze_y", "gaze_z")):
                if not self._warned_missing_gaze:
                    print("[WARN] Gaze enabled in config, but action has no gaze keys. Skipping gaze commands.")
                    self._warned_missing_gaze = True
                return action
            pos_gaze = [
                to_float(action["gaze_x"]),
                to_float(action["gaze_y"]),
                to_float(action["gaze_z"])
            ]
            
            t_rpc = time.perf_counter()
            sent_gaze_rpc = send_rpc(
                f"look_at {pos_gaze[0]:.6f} {pos_gaze[1]:.6f} {pos_gaze[2]:.6f}"
            )
            if sent_gaze_rpc:
                self.metrics_meta["gaze_rpc_ms"] = (time.perf_counter() - t_rpc) * 1000.0
            if (not sent_gaze_rpc) and "gaze_target" in self.ports:
                bottle_gaze = yarp.Bottle()
                for v in pos_gaze: bottle_gaze.addFloat64(v)
                self.ports["gaze_target"].write(bottle_gaze)

        return action

    def _write_target(self, side: str, bottle) -> None:
        """Envía un target cartesiano con envelope (seq, t_send) para medir su latencia."""
        self._cmd_seq[side] += 1
        port = self.ports[f"{side}_out"]
        t_send = time.time()
        port.setEnvelope(self.yarp.Stamp(self._cmd_seq[side], t_send))
        port.write(bottle)
        self.metrics_meta[f"{side}_cmd_seq"] = self._cmd_seq[side]
        self.metrics_meta[f"{side}_cmd_write_ms"] = (time.time() - t_send) * 1000.0

    def disconnect(self) -> None:
        if not self._connected: return
        yarp = self.yarp
        print("Disconnecting YARP ports...")
        for name, port in self.ports.items():
            if port is None:
                continue
            port.interrupt() 
            port.close()
        for name, port in self.img_ports.items():
            port.interrupt()
            port.close()
        # Limpieza de buffers
        self.img_buffers.clear()
        self.np_arrays.clear()
        yarp.Network.fini()
        self._calibrated = False
        self._connected = False
