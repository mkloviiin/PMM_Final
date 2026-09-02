import mujoco
import mujoco.viewer
import numpy as np
import torch
import glfw
import time
import json
import threading
from typing import Any

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
except Exception:
    class Teleoperator:
        def __init__(self, config):
            self.config = config
from .config_icubteleop import iCubTeleopConfig, load_yarp_config

# VR ZMQ ports (must match Meta Quest publisher)
VR_HEAD_PORT = 8115
VR_HAND_RIGHT_PORT = 8117
VR_BUTTONS_PORT = 8119
VR_HAND_LEFT_PORT = 8121


class iCubTeleop(Teleoperator):
    config_class = iCubTeleopConfig
    name = "icub_teleop_mujoco"
    frame_skip = 1  # Default to 1 if not set by config

    # Duración mínima (s) que hay que mantener B para descartar el episodio.
    _B_HOLD_DISCARD_S: float = 1.5

    def __init__(self, config: iCubTeleopConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._calibrated = False
        self._reset_request = False  # Flag to signal external reset
        self._pending_visual_reset = False # Flag for internal thread-safe reset
        
        # Estado interno de botones (Gripper)
        self.rh_grip_state = 1.0 # Default stop
        self.lh_grip_state = 1.0 
        self._rh_grip_pulse = False
        self._lh_grip_pulse = False
        
        # Mujoco variables
        self.model = None
        self.data = None
        self.viewer = None
        
        # IDs cacheados para acceso rápido
        self.mocap_ids = {}
        self.site_ids = {}

        self.actuator_list = []
        for part in self.config.actuators_to_use:
            self.actuator_list.extend(self.config.actuators_to_use[part])

        # Mapa para relacionar nombres de YARP con IDs de Mujoco
        self.qpos_map = []

        # --- VR state ---
        self._vr_running = True
        self._vr_sockets: dict = {}
        self._vr_thread = None
        self.latest_rh_data = None
        self.latest_lh_data = None
        self.latest_buttons = None
        self._last_btn_recv_time = 0.0
        self.calibrated_origin_rh = None
        self.calibrated_origin_lh = None
        self._prev_target_pos = {"right": None, "left": None}
        self._target_filter_alpha = 0.25
        self.freeze_rh = False
        self.freeze_lh = False
        self.freeze_head = False
        self.head_body_name = "head"
        self.gaze_distance = 1.0

        # VR record events
        self._vr_rec_start_event = False
        self._vr_rec_stop_event = False
        self._vr_rec_discard_event = False
        self._vr_a_held = False
        self._vr_b_held = False
        self._vr_b_hold_start: float | None = None

        # Reset scenario state — generic for any scene objects from scenes.yaml
        self._scene_objects: list[dict] = getattr(self.config, "scene_objects", []) or []
        self._scene_joints: list[dict] = getattr(self.config, "scene_joints", []) or []
        self._last_reset_poses: dict[str, tuple] = {}  # {body_name: (pos, quat)}
        self._reset_request = False
        self._pending_visual_reset = False

    @property
    def action_features(self) -> dict[str, type]:

        features = {}        
        # Brazo Derecho
        if self.config.control_arms in ["right", "both"]:
            features["rh_pos_x"] = float
            features["rh_pos_y"] = float
            features["rh_pos_z"] = float
            features["rh_quat_w"] = float
            features["rh_quat_x"] = float
            features["rh_quat_y"] = float
            features["rh_quat_z"] = float
            features["rh_gripper"] = float
        # Brazo Izquierdo
        if self.config.control_arms in ["left", "both"]:
            features["lh_pos_x"] = float
            features["lh_pos_y"] = float
            features["lh_pos_z"] = float
            features["lh_quat_w"] = float
            features["lh_quat_x"] = float
            features["lh_quat_y"] = float
            features["lh_quat_z"] = float
            features["lh_gripper"] = float    
        # Gaze controller
        if getattr(self.config, "use_gaze", False):
            features["gaze_x"] = float
            features["gaze_y"] = float
            features["gaze_z"] = float
        
        return features

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {actuator_name: float for actuator_name in self.actuator_list}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        # Definimos dimensiones esperadas (H, W, C)
        return {
            name: (3, 240, 320) for name in ["left","right"]
        }

    @property
    def feedback_features(self) -> dict:
        return {**self._motors_ft,**self._cameras_ft}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def connect(self, calibrate: bool = True) -> None:
        if self._connected: return

        print(f"Loading Mujoco model from: {self.config.model_path}")
        try:
            self.model = mujoco.MjModel.from_xml_path(self.config.model_path)
            self.data = mujoco.MjData(self.model)

            # --- Sincronizar Física (TimeStep) con Teleop Standalone ---
            yarp_cfg = load_yarp_config(self.config.config_path)
            sim_dt = yarp_cfg.get("sim_dt", 0.005)      # 200 Hz Physics
            control_dt = yarp_cfg.get("control_dt", 0.02) # 50 Hz Control
            
            # Forzar timestep en MuJoCo
            self.model.opt.timestep = sim_dt
            
            # Calcular frame_skip para que el bucle de control coincida
            self.frame_skip = int(control_dt / sim_dt)
            print(f"[Teleop] Physics Sync: sim_dt={sim_dt}, control_dt={control_dt}, frame_skip={self.frame_skip}")
        except Exception as e:
            raise ConnectionError(f"Failed to load Mujoco model: {e}")
        
        # Reset inicial (home si existe; fallback a qpos0)
        try:
            home_id = self.model.key('home').id
            mujoco.mj_resetDataKeyframe(self.model, self.data, home_id)
        except Exception:
            self.data.qpos[:] = self.model.qpos0
            self.data.qvel[:] = 0

        self._apply_home_pose_overrides(yarp_cfg)
        mujoco.mj_forward(self.model, self.data)
        
        # --- Crear Mapa de qpos robusto (joint names o actuator names) ---
        self.qpos_map = [self._resolve_qpos_index(name) for name in self.actuator_list]

        # Cachear IDs para evitar búsquedas por string en cada ciclo (Optimización)
        if self.config.control_arms in ["right", "both"]:
            self.mocap_ids['right'] = self.model.body('r_hand_target').mocapid[0]
            self.site_ids['right'] = self.model.site('r_hand_dh_frame_site').id
            
        if self.config.control_arms in ["left", "both"]:
            self.mocap_ids['left'] = self.model.body('l_hand_target').mocapid[0]
            self.site_ids['left'] = self.model.site('l_hand_dh_frame_site').id
        
        if self.config.use_gaze:
            self.mocap_ids['gaze'] = self.model.body('com_target').mocapid[0]

        # Inicializar mocaps desde la pose definida en XML (home/base)
        self._reset_mocaps_to_xml_pose()


        # Iniciar Viewer Pasivo
        print("Launching MuJoCo Viewer... (Press 1/2/3 for Right Hand, 4/5/6 for Left Hand)")
        print("Mouse: Ctrl + Right-click drag to move mocap targets")
        self.viewer = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=self._key_callback,
            show_left_ui=True,
            show_right_ui=True,
        )

        # Camera — vista aérea sobre la mesa (elevation negativo = mirar hacia abajo)
        self.viewer.cam.lookat[:] = [0.35, 0.0, 0.9]
        self.viewer.cam.distance = 2.5
        self.viewer.cam.elevation = -65
        self.viewer.cam.azimuth = 0

        # Show sites, hide contact forces
        self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE        
        self._connected = True

        # --- Initialize VR if enabled ---
        if getattr(self.config, "vr_enabled", False):
            self._init_vr()

        if calibrate:
            self.calibrate()

    def disconnect(self) -> None:
        self._vr_running = False
        # Close VR sockets
        for _, (ctx, sock) in self._vr_sockets.items():
            try:
                sock.close()
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

        # Process VR input if enabled (before reading mocaps)
        if getattr(self.config, "vr_enabled", False):
            self._process_vr_hands()
            self._process_vr_head()
            self._update_vr_record_events()
            if self.latest_buttons:
                self._process_vr_button_commands()
                self.latest_buttons = None

        # IMPORTANTE: Sincronizar viewer para recibir inputs del usuario (movimiento de mocap con mouse)
        if self.viewer.is_running():
            self.viewer.sync()
            mujoco.mj_forward(self.model, self.data)
        else:
            print("Warning: Viewer closed.")
            pass

        actions = {}

        # Handle queued reset from viewer thread
        if self._pending_visual_reset:
            self._reset_scenario()
            self._pending_visual_reset = False
        
        # --- Lectura Mano Derecha ---
        if self.config.control_arms in ["right", "both"]:
            mid = self.mocap_ids['right']
            pos = self.data.mocap_pos[mid]   # [x, y, z]
            quat = self.data.mocap_quat[mid] # [w, x, y, z]
            
            # Asignación plana (Flat Dictionary)
            actions["rh_pos_x"] = torch.tensor(pos[0], dtype=torch.float32)
            actions["rh_pos_y"] = torch.tensor(pos[1], dtype=torch.float32)
            actions["rh_pos_z"] = torch.tensor(pos[2], dtype=torch.float32)
            actions["rh_quat_w"] = torch.tensor(quat[0], dtype=torch.float32)
            actions["rh_quat_x"] = torch.tensor(quat[1], dtype=torch.float32)
            actions["rh_quat_y"] = torch.tensor(quat[2], dtype=torch.float32)
            actions["rh_quat_z"] = torch.tensor(quat[3], dtype=torch.float32)
            actions["rh_gripper"] = torch.tensor(self.rh_grip_state, dtype=torch.float32)
            if self._rh_grip_pulse:
                self.rh_grip_state = 1.0
                self._rh_grip_pulse = False

        # --- Lectura Mano Izquierda ---
        if self.config.control_arms in ["left", "both"]:
            mid = self.mocap_ids['left']
            pos = self.data.mocap_pos[mid]
            quat = self.data.mocap_quat[mid]
            
            actions["lh_pos_x"] = torch.tensor(pos[0], dtype=torch.float32)
            actions["lh_pos_y"] = torch.tensor(pos[1], dtype=torch.float32)
            actions["lh_pos_z"] = torch.tensor(pos[2], dtype=torch.float32)
            actions["lh_quat_w"] = torch.tensor(quat[0], dtype=torch.float32)
            actions["lh_quat_x"] = torch.tensor(quat[1], dtype=torch.float32)
            actions["lh_quat_y"] = torch.tensor(quat[2], dtype=torch.float32)
            actions["lh_quat_z"] = torch.tensor(quat[3], dtype=torch.float32)
            actions["lh_gripper"] = torch.tensor(self.lh_grip_state, dtype=torch.float32)
            if self._lh_grip_pulse:
                self.lh_grip_state = 1.0
                self._lh_grip_pulse = False

        if 'gaze' in self.mocap_ids:
            gid = self.mocap_ids['gaze']
            gaze_pos = self.data.mocap_pos[gid]

            actions["gaze_x"] = torch.tensor(gaze_pos[0], dtype=torch.float32)
            actions["gaze_y"] = torch.tensor(gaze_pos[1], dtype=torch.float32)
            actions["gaze_z"] = torch.tensor(gaze_pos[2], dtype=torch.float32)

        return actions

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        #Recibe las observaciones del robot y actualiza el modelo visual en Mujoco.        
        if not self.is_connected: return

        if "rh_gripper" in feedback:
            val = feedback["rh_gripper"]
            self.rh_grip_state = float(val.item() if hasattr(val, "item") else val)
            self._rh_grip_pulse = False
        if "lh_gripper" in feedback:
            val = feedback["lh_gripper"]
            self.lh_grip_state = float(val.item() if hasattr(val, "item") else val)
            self._lh_grip_pulse = False
        
        update_visuals = False

        # --- Object Sync (generic, indexed by scenes.yaml order) ---
        if not self._reset_request:
            if not hasattr(self, "_obj_jnt_adrs"):
                # Cache qpos addresses for each declared scene object
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
                prefix = f"obj{idx}"
                if jnt_adr is not None and f"{prefix}_pos_x" in feedback:
                    try:
                        self.data.qpos[jnt_adr + 0] = feedback[f"{prefix}_pos_x"].item()
                        self.data.qpos[jnt_adr + 1] = feedback[f"{prefix}_pos_y"].item()
                        self.data.qpos[jnt_adr + 2] = feedback[f"{prefix}_pos_z"].item()
                        self.data.qpos[jnt_adr + 3] = feedback[f"{prefix}_quat_w"].item()
                        self.data.qpos[jnt_adr + 4] = feedback[f"{prefix}_quat_x"].item()
                        self.data.qpos[jnt_adr + 5] = feedback[f"{prefix}_quat_y"].item()
                        self.data.qpos[jnt_adr + 6] = feedback[f"{prefix}_quat_z"].item()
                        update_visuals = True
                    except Exception:
                        pass

        # --- VR Mocap Sync ---
        # If the robot is sending VR target positions, update our local mocaps to match
        if self.is_connected:
            # Right Hand
            if "rh_target_pos_x" in feedback and "right" in self.mocap_ids:
                mid = self.mocap_ids['right']
                self.data.mocap_pos[mid][0] = feedback["rh_target_pos_x"].item()
                self.data.mocap_pos[mid][1] = feedback["rh_target_pos_y"].item()
                self.data.mocap_pos[mid][2] = feedback["rh_target_pos_z"].item()
                self.data.mocap_quat[mid][0] = feedback["rh_target_quat_w"].item()
                self.data.mocap_quat[mid][1] = feedback["rh_target_quat_x"].item()
                self.data.mocap_quat[mid][2] = feedback["rh_target_quat_y"].item()
                self.data.mocap_quat[mid][3] = feedback["rh_target_quat_z"].item()
                update_visuals = True

            # Left Hand
            if "lh_target_pos_x" in feedback and "left" in self.mocap_ids:
                mid = self.mocap_ids['left']
                self.data.mocap_pos[mid][0] = feedback["lh_target_pos_x"].item()
                self.data.mocap_pos[mid][1] = feedback["lh_target_pos_y"].item()
                self.data.mocap_pos[mid][2] = feedback["lh_target_pos_z"].item()
                self.data.mocap_quat[mid][0] = feedback["lh_target_quat_w"].item()
                self.data.mocap_quat[mid][1] = feedback["lh_target_quat_x"].item()
                self.data.mocap_quat[mid][2] = feedback["lh_target_quat_y"].item()
                self.data.mocap_quat[mid][3] = feedback["lh_target_quat_z"].item()
                update_visuals = True
            
            # Gaze
            if "gaze_target_pos_x" in feedback and "gaze" in self.mocap_ids:
                gid = self.mocap_ids['gaze']
                self.data.mocap_pos[gid][0] = feedback["gaze_target_pos_x"].item()
                self.data.mocap_pos[gid][1] = feedback["gaze_target_pos_y"].item()
                self.data.mocap_pos[gid][2] = feedback["gaze_target_pos_z"].item()
                update_visuals = True
        
        # Iteramos sobre los datos que llegan del robot
        for i, name in enumerate(self.actuator_list):
            if name in feedback:
                # Obtenemos el índice pre-calculado
                q_idx = self.qpos_map[i]
                
                # Si el mapeo fue exitoso (no es -1)
                if q_idx != -1:
                    val_tensor = feedback[name]
                    # Conversión: Tensor -> Float (Radianes)
                    val_rad = val_tensor.item()
                    
                    # Actualización directa en memoria
                    self.data.qpos[q_idx] = val_rad
                    update_visuals = True

        # Si actualizamos algún joint, recalculamos la cinemática directa
        if update_visuals:
            mujoco.mj_forward(self.model, self.data)

    # ========================= VR SUPPORT ==================================

    def _init_vr(self):
        """Initialise ZMQ sockets for Meta Quest controller input."""
        if zmq is None:
            print("[VR] Warning: zmq not available, VR disabled")
            return
        self._vr_sockets = {}
        for name, port in [("head", VR_HEAD_PORT),
                           ("rh", VR_HAND_RIGHT_PORT),
                           ("lh", VR_HAND_LEFT_PORT),
                           ("btn", VR_BUTTONS_PORT)]:
            ctx = zmq.Context()
            sock = ctx.socket(zmq.PULL)
            endpoint = f"tcp://*:{port}"
            sock.bind(endpoint)
            print(f"  [VR] Socket '{name}' bound to {endpoint}")
            sock.setsockopt(zmq.LINGER, 0)
            self._vr_sockets[name] = (ctx, sock)

        self._vr_running = True
        self._vr_thread = threading.Thread(target=self._vr_receive_loop, daemon=True)
        self._vr_thread.start()
        print("[VR] VR receiver thread started")

    def _vr_receive_loop(self):
        """Background thread: drain all VR sockets continuously."""
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
                            self.latest_buttons = self._parse_buttons(raw)
                            self._last_btn_recv_time = time.time()
                except Exception:
                    pass
            time.sleep(0.001)

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

    def _vr_hand_to_target(self, hand_data, calib_origin, home_pos,
                           roll_offset=0, side="right"):
        """Convert VR controller data to robot-frame target with Smoothing."""
        if hand_data is None or PyQuaternion is None:
            return None, None, calib_origin
        vr_pos = hand_data["pos"]
        if np.linalg.norm(vr_pos) < 0.001:
            return None, None, calib_origin
        if calib_origin is None:
            calib_origin = vr_pos.copy()
        
        delta_vr = vr_pos - calib_origin
        # VR → robot frame mapping
        delta_robot = np.array([delta_vr[2], -delta_vr[0], delta_vr[1]])
        raw_target_pos = home_pos + delta_robot

        # --- Low Pass Filter on Position ---
        if self._prev_target_pos[side] is None:
             self._prev_target_pos[side] = raw_target_pos
        
        filtered_pos = (self._target_filter_alpha * raw_target_pos + 
                        (1.0 - self._target_filter_alpha) * self._prev_target_pos[side])
        self._prev_target_pos[side] = filtered_pos
        
        target_pos = filtered_pos

        if "quat" in hand_data:
            q_vr = hand_data["quat"]
            q_remapped = PyQuaternion(w=q_vr.w, x=-q_vr.z, y=-q_vr.x, z=q_vr.y)
            q_fix_x = PyQuaternion(axis=[1, 0, 0], degrees=180)
            q_final = q_fix_x * q_remapped
            if roll_offset != 0:
                q_fix = PyQuaternion(axis=[1, 0, 0], degrees=roll_offset)
                q_final = q_final * q_fix
            target_quat = np.array([q_final.w, q_final.x,
                                    q_final.y, q_final.z])
        else:
            target_quat = np.array([1.0, 0.0, 0.0, 0.0])

        return target_pos, target_quat, calib_origin

    def _process_vr_hands(self):
        """Map VR hand controller data → mocap target positions."""
        ROBOT_HOME_RH = np.array([-0.3,  0.2, 1.2])
        ROBOT_HOME_LH = np.array([-0.3, -0.2, 1.2])

        # Right hand
        if not self.freeze_rh and self.latest_rh_data is not None and "right" in self.mocap_ids:
            pos, quat, self.calibrated_origin_rh = self._vr_hand_to_target(
                self.latest_rh_data, self.calibrated_origin_rh,
                ROBOT_HOME_RH, roll_offset=90, side="right")
            if pos is not None:
                pos[0] = np.clip(pos[0], 0.1, 0.8)
                pos[1] = np.clip(pos[1], -0.4, 0.1)
                pos[2] = np.clip(pos[2], 0.4, 1.0)
                mid = self.mocap_ids['right']
                self.data.mocap_pos[mid] = pos
                self.data.mocap_quat[mid] = quat

        # Left hand
        if not self.freeze_lh and self.latest_lh_data is not None and "left" in self.mocap_ids:
            pos, quat, self.calibrated_origin_lh = self._vr_hand_to_target(
                self.latest_lh_data, self.calibrated_origin_lh,
                ROBOT_HOME_LH, roll_offset=90, side="left")
            if pos is not None:
                pos[0] = np.clip(pos[0], 0.1, 0.8)
                pos[1] = np.clip(pos[1], -0.1, 0.4)
                pos[2] = np.clip(pos[2], 0.4, 1.0)
                mid = self.mocap_ids['left']
                self.data.mocap_pos[mid] = pos
                self.data.mocap_quat[mid] = quat

    def _process_vr_head(self):
        """Process VR head rotation → gaze mocap target."""
        if self.freeze_head or PyQuaternion is None:
            return
        if "head" not in self._vr_sockets:
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
                    q_vr = PyQuaternion(w=float(j["w"]), x=float(j["x"]),
                                        y=float(j["y"]), z=float(j["z"]))
            else:
                parts = data.split(",")
                if len(parts) == 4:
                    q_vr = PyQuaternion(w=float(parts[3]), x=float(parts[0]),
                                        y=float(parts[1]), z=float(parts[2]))
        except Exception:
            return

        if q_vr is not None and "gaze" in self.mocap_ids:
            q_fix_yaw = PyQuaternion(axis=[0, 1, 0], degrees=-20)
            q_vr = q_fix_yaw * q_vr
            v_fwd_vr = q_vr.rotate(np.array([0.0, 0.0, -1.0]))
            v_fwd_robot = np.array([-v_fwd_vr[2], v_fwd_vr[0], -v_fwd_vr[1]])
            norm = np.linalg.norm(v_fwd_robot)
            if norm > 0:
                v_fwd_robot /= norm

            try:
                head_bid = self.model.body(self.head_body_name).id
                head_pos = self.data.xpos[head_bid]
            except Exception:
                head_pos = np.array([0.0, 0.0, 1.0])
            target = head_pos + v_fwd_robot * self.gaze_distance
            self.data.mocap_pos[self.mocap_ids['gaze']] = target

    def _process_vr_button_commands(self):
        """React to VR button states for gripper and freeze."""
        b = self.latest_buttons
        if b is None or not isinstance(b, dict):
            return

        # Hands
        if b.get("INDEX_RIGHT", 0) > 0.8:
            self.rh_grip_state = 0.5  # close
            self._rh_grip_pulse = True
        elif b.get("HAND_RIGHT", 0) > 0.8:
            self.rh_grip_state = 0.0  # open
            self._rh_grip_pulse = True
        if b.get("INDEX_LEFT", 0) > 0.8:
            self.lh_grip_state = 0.5  # close
            self._lh_grip_pulse = True
        elif b.get("HAND_LEFT", 0) > 0.8:
            self.lh_grip_state = 0.0  # open
            self._lh_grip_pulse = True

        # Reset
        if b.get("BTN_TWO_LEFT", 0) > 0.5 or b.get("Y", 0) > 0.8:
            self._pending_visual_reset = True

        # Freeze while squeezing triggers
        any_active = any(b.get(k, 0) > 0.8
                         for k in ("INDEX_RIGHT", "HAND_RIGHT",
                                   "INDEX_LEFT", "HAND_LEFT"))
        self.freeze_rh = any_active
        self.freeze_lh = any_active
        self.freeze_head = any_active

    def _update_vr_record_events(self) -> None:
        """Track A/B button presses for recording control."""
        b = self.latest_buttons
        if b is None or not isinstance(b, dict):
            # If button stream is stale, release freeze
            if self._vr_sockets and (time.time() - self._last_btn_recv_time) > 0.5:
                self.freeze_rh = False
                self.freeze_lh = False
                self.freeze_head = False
            return

        a_pressed = (b.get("BTN_ONE_RIGHT", 0) > 0.5) or (b.get("A", 0) > 0.8)
        b_pressed = (b.get("BTN_TWO_RIGHT", 0) > 0.5) or (b.get("B", 0) > 0.8)

        # A: tap → start
        if a_pressed and not self._vr_a_held:
            self._vr_rec_start_event = True
        self._vr_a_held = a_pressed

        # B: tap → stop/save  |  hold 1.5s → discard
        if b_pressed and not self._vr_b_held:
            self._vr_b_hold_start = time.monotonic()

        if not b_pressed and self._vr_b_held:
            held_s = time.monotonic() - (self._vr_b_hold_start or 0.0)
            if held_s >= self._B_HOLD_DISCARD_S:
                self._vr_rec_discard_event = True
                print(f"[VR] B held {held_s:.1f}s → DISCARD episode")
            else:
                self._vr_rec_stop_event = True
            self._vr_b_hold_start = None

        self._vr_b_held = b_pressed

    def consume_vr_record_events(self) -> tuple[bool, bool, bool]:
        """Return (start, stop/save, discard/rerecord) and clear flags."""
        start = self._vr_rec_start_event
        stop = self._vr_rec_stop_event
        discard = self._vr_rec_discard_event
        self._vr_rec_start_event = False
        self._vr_rec_stop_event = False
        self._vr_rec_discard_event = False
        return start, stop, discard

    # ========================= Helpers & Callbacks =========================

    def _resolve_qpos_index(self, name: str) -> int:
        """Resolve qpos index from either a joint name or an actuator name."""
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
        """Setea mocaps exactamente a la pose declarada en el XML para cada target."""
        body_by_side = {
            'right': 'r_hand_target',
            'left': 'l_hand_target',
            'gaze': 'com_target',
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
            if 'right' in self.mocap_ids and 'right' in self.site_ids:
                mid = self.mocap_ids['right']
                sid = self.site_ids['right']
                mujoco.mju_mat2Quat(self.data.mocap_quat[mid], self.data.site_xmat[sid])

            if 'left' in self.mocap_ids and 'left' in self.site_ids:
                mid = self.mocap_ids['left']
                sid = self.site_ids['left']
                mujoco.mju_mat2Quat(self.data.mocap_quat[mid], self.data.site_xmat[sid])

        mujoco.mj_forward(self.model, self.data)

    def _reset_scenario(self):
        """Randomize all declared scene objects (driven by scenes.yaml).

        Falls back to legacy cube_spawn config if no scene_objects are declared
        but a blue-cube body exists in the model.
        """
        objects = self._scene_objects

        # Legacy fallback: only when BOTH scene_objects AND scene_joints are absent.
        # Joint-only scenes (e.g. door) must not trigger this or they'd try to reset
        # a 'blue-cube' that doesn't exist in the model.
        if not objects and not self._scene_joints:
            spawn = getattr(self.config, "cube_spawn", None)
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
                # freejoint = 6 DOF, hinge = 1 DOF
                num_dof = 6 if ndof == 0 else 1  # mjtJoint.mjJNT_FREE = 0
                self.data.qvel[dof_adr:dof_adr + num_dof] = 0

                pos_arr = self.data.qpos[qpos_adr:qpos_adr + 3].copy()
                quat_arr = self.data.qpos[qpos_adr + 3:qpos_adr + 7].copy()
                reset_poses[body_name] = (pos_arr, quat_arr)
                any_reset = True
                print(f"[Teleop] Reset {body_name} → ({pos_arr[0]:.2f}, {pos_arr[1]:.2f}, {pos_arr[2]:.2f})")
            except Exception as e:
                print(f"[Teleop] Reset skip '{body_name}': {e}")

        if any_reset:
            self._last_reset_poses = reset_poses
            self._reset_request = True
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
                print(f"[Teleop] Reset joint '{jnt_name}' → {target_qpos:.3f} rad")
            except Exception as e:
                print(f"[Teleop] Reset joint skip '{jnt_name}': {e}")

        if self._scene_joints:
            mujoco.mj_forward(self.model, self.data)

    def _key_callback(self, keycode):
        """Callback para el viewer pasivo."""
        if keycode == glfw.KEY_1: 
            self.rh_grip_state = 0.0 # Close
            self._rh_grip_pulse = True
            print("Right Gripper: OPEN")
        elif keycode == glfw.KEY_2: 
            self.rh_grip_state = 0.5 # Open
            self._rh_grip_pulse = True
            print("Right Gripper: CLOSE")
        elif keycode == glfw.KEY_3: 
            self.rh_grip_state = 1.0 # Open
            self._rh_grip_pulse = False
            print("Right Gripper: STOP")
        
        elif keycode == glfw.KEY_R:
            self._pending_visual_reset = True

        # 4: Close Left, 5: Open Left
        elif keycode == glfw.KEY_4: 
            self.lh_grip_state = 0.0
            self._lh_grip_pulse = True
            print("Left Gripper: OPEN")
        elif keycode == glfw.KEY_5: 
            self.lh_grip_state = 0.5
            self._lh_grip_pulse = True
            print("Left Gripper: CLOSE")
        elif keycode == glfw.KEY_6: 
            self.lh_grip_state = 1.0
            self._lh_grip_pulse = False
            print("Left Gripper: STOP")
        
        # Space: Re-sincronizar mocaps (Reset visual)
        elif keycode == glfw.KEY_SPACE:
            self._reset_mocaps_to_xml_pose()
            print("Mocaps reset to XML home pose.")
