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

from .config_icub import iCubConfig, load_yarp_config
from .gz_pose import GzContactReader, GzPoseReader

# Espera máxima al primer frame de cada cámara en connect(), y antigüedad a partir de
# la cual se avisa que una cámara dejó de mandar frames.
CAMERA_FIRST_FRAME_TIMEOUT_S = 5.0
CAMERA_STALE_S = 1.0
# /teleop/touch:o: 12 taxels por dedo en este orden (teleop_module_sm._touch_names),
# luego la palma. MuJoCo solo registra pulgar, índice y medio.
TOUCH_FINGER_SLOT = {"thumb": 4, "index": 0, "middle": 1}
TAXELS_PER_FINGER = 12

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
# (aparecen en la identidad). Mismo criterio en lerobot_robot_icub (MuJoCo) y en
# tools/canonicalize_quats.py.
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


def _grip_mode(value: float) -> str:
    """Mismos umbrales que el plugin de MuJoCo: el teleop manda 0.0/0.5/1.0 exactos,
    pero una política entrega valores continuos."""
    if value <= 0.1:
        return "open"
    if 0.4 <= value <= 0.6:
        return "close"
    return "stop"

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
        self._grip_sent = {"rh": None, "lh": None}
        # Último comando enviado (observaciones use_targets). Pinza = fracción de cierre
        # pedida (0 = abierta, 1 = cerrada), como en MuJoCo.
        self._last_target: dict[str, list[float]] = {}
        self._last_gaze: list[float] | None = None
        self._grip_level = {"rh": 0.0, "lh": 0.0}
        self._last_touch: list[float] = []
        self._pose_reader: GzPoseReader | None = None
        self._contact_reader: GzContactReader | None = None
        self._warned: set[str] = set()
        # (tamaño de la Bottle, articulaciones por bloque, nombre -> índice) de robot_state
        self._state_layout: tuple[int, int, dict[str, int]] | None = None
        self._warned_missing_gaze = False
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

        sides = self._arm_sides()
        if self.config.use_joint_vel:
            features.update({f"{name}_vel": float for name in self.actuator_list})
        if self.config.use_eef_pose:
            for side in sides:
                features.update({f"{side}_ee_pos_{ax}": float for ax in "xyz"})
                features.update({f"{side}_ee_quat_{ax}": float for ax in "wxyz"})
        if self.config.use_targets:
            for side in sides:
                features.update({f"{side}_target_pos_{ax}": float for ax in "xyz"})
                features.update({f"{side}_target_quat_{ax}": float for ax in "wxyz"})
            if self.config.use_gaze:
                features.update({f"gaze_target_pos_{ax}": float for ax in "xyz"})
            for side in sides:
                features[f"{side}_gripper"] = float
        if self.config.use_tactile:
            for side in self._touch_sides():
                features.update({f"{side}_touch_{finger}": float for finger in TOUCH_FINGER_SLOT})
        if self.config.use_object_pose:
            for i in range(len(self.config.object_bodies)):
                features.update({f"obj{i}_pos_{ax}": float for ax in "xyz"})
                features.update({f"obj{i}_quat_{ax}": float for ax in "wxyz"})
        if self.config.use_joint_torque:
            features.update({f"{name}_torque": float for name in self.actuator_list
                             if has_torque(name)})

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

    def _arm_sides(self) -> list[str]:
        return [side for side, arms in (("rh", ("right", "both")), ("lh", ("left", "both")))
                if self.config.control_arms in arms]

    def _layout_for(self, n: int) -> tuple[int, dict[str, int]]:
        """Articulaciones por bloque y nombre -> índice de /teleop/robot_state:o.

        teleop_module_sm publica torso, [left_arm], [right_arm], head según SUS brazos
        (--control), no los de este cliente: con el módulo en 'both' y el robot en 'right'
        tomar las primeras 25 era leer el brazo izquierdo como el derecho. Se deduce qué
        publica a partir del tamaño (q, o q|dq|tau) y se lee por nombre.
        """
        if self._state_layout and self._state_layout[0] == n:
            return self._state_layout[1], self._state_layout[2]
        full = load_yarp_config(self.config.config_path).get("actuators", {})
        layouts = {}
        for arms in (("right_arm",), ("left_arm",), ("left_arm", "right_arm"), ()):
            names = [j for part in ("torso", *arms, "head") for j in full.get(part, [])]
            layouts.setdefault(len(names), names)
        layouts[len(self.actuator_list)] = list(self.actuator_list)  # mismos brazos: orden propio
        block = n // 3 if n % 3 == 0 and n // 3 in layouts else n
        names = layouts.get(block)
        missing = [j for j in self.actuator_list if names is None or j not in names]
        if missing:
            raise ConnectionError(
                f"/teleop/robot_state:o sends {n} values, which does not match this robot's "
                f"joints ({len(self.actuator_list)}, control_arms={self.config.control_arms}). "
                f"Start teleop_module_sm.py with the same or more arms (--control)."
            )
        if block != len(self.actuator_list):
            print(f"[iCub] robot_state has {block} joints per block (module controls more arms); "
                  f"reading the {len(self.actuator_list)} of this robot by name.")
        index = {j: i for i, j in enumerate(names)}
        self._state_layout = (n, block, index)
        return block, index

    # Paso con que se manda la fracción de cierre al módulo: el gatillo cambia en cada frame
    # y la mano real se mueve a ~10 deg/s, así que basta con mandar cambios de 5 %.
    GRIP_STEP = 0.05

    def _command_grip(self, side: str, value: float, send_rpc) -> None:
        """rh/lh_gripper = fracción de cierre 0..1 -> RPC hand_right/hand_left <fracción>.

        Con gripper_legacy (modelos/datasets anteriores al 2026-10-21): <=0.1 abrir,
        0.4-0.6 cerrar, el resto mantiene.
        """
        if self.config.gripper_legacy:
            mode = _grip_mode(value)
            if mode == "stop":
                return
            value = 0.0 if mode == "open" else 1.0
        level = float(np.clip(value, 0.0, 1.0))
        self._grip_level[side] = level
        step = round(level / self.GRIP_STEP) * self.GRIP_STEP
        if self._grip_sent[side] is not None and abs(step - self._grip_sent[side]) < 1e-6:
            return
        word = "right" if side == "rh" else "left"
        if send_rpc(f"hand_{word} {step:.2f}"):
            self._grip_sent[side] = step
        elif self.ports.get("hand_cmd") is not None and step in (0.0, 1.0):
            # Puerto heredado: solo entiende abrir/cerrar
            cmd = self.yarp.Bottle(); cmd.addString(f"{'open' if step == 0.0 else 'close'}_{word}")
            self.ports["hand_cmd"].write(cmd)
            self._grip_sent[side] = step

    def reset_command_state(self) -> None:
        """Olvida el último comando (targets, mirada, pinza) al empezar un episodio nuevo."""
        self._last_target.clear()
        self._last_gaze = None
        self._grip_level = {"rh": 0.0, "lh": 0.0}
        self._grip_sent = {"rh": None, "lh": None}

    def _touch_sides(self) -> list[str]:
        # La piel la republica teleop_module_sm solo para la mano derecha; los sensores de
        # contacto de Gazebo están en ambas manos.
        sides = self._arm_sides()
        return sides if self.config.touch_source == "gazebo" else [s for s in sides if s == "rh"]

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            print(f"[WARN] {msg}")

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

        # 3. Observaciones extra
        if self.config.use_joint_vel or self.config.use_joint_torque:
            # q | dq | tau desde teleop_module_sm._send_robot_state; un módulo anterior
            # publica solo q y el dataset quedaría con velocidades/torques falsos.
            first = self.ports["state"].read(True)
            if first is not None:
                block, _ = self._layout_for(first.size())
                if first.size() != 3 * block:
                    raise ConnectionError(
                        f"/teleop/robot_state:o sends {first.size()} values ({block} joints, positions "
                        f"only); use_joint_vel/use_joint_torque need q|dq|tau. Restart "
                        f"teleop_module_sm.py so it runs the current version."
                    )

        if self.config.use_tactile and self.config.touch_source == "gazebo":
            keys = [f"{side}_{finger}" for side in self._touch_sides() for finger in TOUCH_FINGER_SLOT]
            self._contact_reader = GzContactReader(keys)
            missing = self._contact_reader.start()
            if missing:
                print(f"[WARN] Gazebo does not publish {missing}: restart Gazebo so it loads the "
                      f"fingertip contact sensors of model.urdf, or the touch will be 0.")
            else:
                print(f"[iCub] Touch from Gazebo fingertip contact sensors: {keys}")
        elif self.config.use_tactile and "rh" in self._arm_sides():
            p = yarp.BufferedPortBottle()
            p.open(f"{prefix}/touch:i")
            if yarp.Network.connect(self.config.remote_touch_port, f"{prefix}/touch:i"):
                self.ports["touch"] = p
            else:
                p.close()
                print(f"[WARN] {self.config.remote_touch_port} not connected: rh_touch_* will be 0.")

        if self.config.use_object_pose:
            self._pose_reader = GzPoseReader(self.config.object_bodies, self.config.gazebo_root_xyz)
            self._pose_reader.start()
            print(f"[iCub] Object poses from Gazebo: {self.config.object_bodies}")

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
        block, index = self._layout_for(n)
        vals = [state_bot.get(i).asFloat64() for i in range(n)]
        # Por nombre: el módulo puede publicar más brazos que los de este robot
        for name in self.actuator_list:
            obs[name] = torch.tensor(vals[index[name]], dtype=torch.float32)
        # Bloques q | dq | tau (ver connect); NaN = la parte no entrega torques
        if self.config.use_joint_vel:
            for name in self.actuator_list:
                obs[f"{name}_vel"] = torch.tensor(vals[block + index[name]], dtype=torch.float32)
        if self.config.use_joint_torque:
            no_torque = []
            for name in filter(has_torque, self.actuator_list):
                tau = vals[2 * block + index[name]]
                if math.isnan(tau):
                    no_torque.append(name)
                    tau = 0.0
                obs[f"{name}_torque"] = torch.tensor(tau, dtype=torch.float32)
            if no_torque:
                self._warn_once("torque", f"No joint torques for {no_torque} (set use_joint_torque "
                                          f"in the YAML read by teleop_module_sm); recording 0.")

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

        self._add_extra_observations(obs)
        if self.config.canonical_quats:
            canonicalize_quats(obs)
        return obs

    def _add_extra_observations(self, obs: dict[str, Any]) -> None:
        """Observaciones use_eef_pose / use_targets / use_tactile / use_object_pose."""
        def put(prefix: str, axes: str, values) -> None:
            for ax, val in zip(axes, values):
                obs[f"{prefix}_{ax}"] = torch.tensor(float(val), dtype=torch.float32)

        sides = self._arm_sides()
        if self.config.use_eef_pose:
            for side in sides:
                pose = self._current_pose.get(side)
                if pose is None:
                    self._warn_once(f"ee_{side}", f"No /teleop/{side}_current_pose:o data yet: {side}_ee_* = 0.")
                    pose = [0.0] * 7
                put(f"{side}_ee_pos", "xyz", pose[:3])
                put(f"{side}_ee_quat", "wxyz", pose[3:7])

        if self.config.use_targets:
            for side in sides:
                # Antes del primer comando el target es la mano (como el snap de MuJoCo)
                target = self._last_target.get(side) or self._current_pose.get(side) or [0.0] * 7
                put(f"{side}_target_pos", "xyz", target[:3])
                put(f"{side}_target_quat", "wxyz", target[3:7])
                obs[f"{side}_gripper"] = torch.tensor(self._grip_level[side], dtype=torch.float32)
            if self.config.use_gaze:
                put("gaze_target_pos", "xyz", self._last_gaze or self.config.gaze_target_init)

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
                taxels = self._last_touch[slot * TAXELS_PER_FINGER:(slot + 1) * TAXELS_PER_FINGER]
                touching = bool(taxels) and sum(taxels) / len(taxels) >= self.config.touch_threshold
                obs[f"rh_touch_{finger}"] = torch.tensor(float(touching), dtype=torch.float32)

        if self.config.use_object_pose:
            for i, body in enumerate(self.config.object_bodies):
                pose = self._pose_reader.get(body) if self._pose_reader else None
                if pose is None:
                    self._warn_once(f"obj_{body}", f"No Gazebo pose for '{body}' yet (PosePublisher in "
                                                   f"gazebo/models/{body}/model.sdf?): obj{i}_* = 0.")
                    pose = ([0.0] * 3, [0.0] * 4)
                put(f"obj{i}_pos", "xyz", pose[0])
                put(f"obj{i}_quat", "wxyz", pose[1])

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
            self._last_target["rh"] = pos_rh + quat_rh
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
                self._command_grip("rh", to_float(action["rh_gripper"]), send_rpc)

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

            self._last_target["lh"] = pos_lh + quat_lh
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
                self._command_grip("lh", to_float(action["lh_gripper"]), send_rpc)
                    
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
            self._last_gaze = pos_gaze
            
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
        if self._pose_reader is not None:
            self._pose_reader.stop()
            self._pose_reader = None
        if self._contact_reader is not None:
            self._contact_reader.stop()
            self._contact_reader = None
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
