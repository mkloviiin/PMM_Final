from dataclasses import dataclass, field
import os
from pathlib import Path

import yaml

try:
    from lerobot.robots.config import RobotConfig
except Exception:
    class RobotConfig:
        @classmethod
        def register_subclass(cls, _name: str):
            def _decorator(subcls):
                return subcls
            return _decorator


def _project_root() -> Path:
    """Sube hasta la carpeta que contiene config/control_config.yaml."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "config" / "control_config.yaml").exists():
            return parent
    return Path(__file__).resolve().parents[3]


def _default_config_path() -> Path:
    env_path = os.getenv("ICUB_LEROBOT_CONFIG")
    if env_path:
        return Path(env_path).expanduser()
    return _project_root() / "config" / "control_config.yaml"


def load_yarp_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path).expanduser() if config_path else _default_config_path()
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _normalize_control_arms(mode: str | None) -> str:
    value = str(mode or "both").strip().lower()
    aliases = {
        "r": "right",
        "rh": "right",
        "right_arm": "right",
        "l": "left",
        "lh": "left",
        "left_arm": "left",
        "both_arms": "both",
    }
    normalized = aliases.get(value, value)
    return normalized if normalized in {"left", "right", "both", "none"} else "both"


def _get_actuators_logic(cfg: dict) -> dict[str, list[str]]:
    full = cfg.get("actuators", {})
    mode = _normalize_control_arms(cfg.get("control_arms", "both"))
    use_gaze = bool(cfg.get("gaze_ctrl", False))
    used: dict[str, list[str]] = {}
    
    if "torso" in full: used["torso"] = full["torso"]
    if mode in ["left", "both"] and "left_arm" in full: used["left_arm"] = full["left_arm"]
    if mode in ["right", "both"] and "right_arm" in full: used["right_arm"] = full["right_arm"]
    if use_gaze and "head" in full: used["head"] = full["head"]
    
    return used


def _default_camera_ports() -> dict[str, str]:
    return {
        "left": "",   # derivado de robot_name en __post_init__
        "right": "",
    }


def _camera_ports_from_robot_name(robot_name: str) -> dict[str, str]:
    # Gazebo (gz-sim) publica los ojos en .../rgbImage:o y además tiene la
    # cámara frontal del mundo (models/camera-stand). El robot real no tiene
    # cámara frontal en YARP: se agrega vía `camera_ports` en el YAML.
    if robot_name.strip().lower() == "icubsim":
        return {
            "left": f"/{robot_name}/cam/left/rgbImage:o",
            "right": f"/{robot_name}/cam/right/rgbImage:o",
            "front": f"/{robot_name}/cam/front/rgbImage:o",
        }
    # Robot real: el relay local de teleop_module_sm (que ya recibe camcalib), no
    # una segunda conexión a las cámaras del robot. Duplicar las imágenes crudas
    # (~110 Mbit/s por cámara y cliente) saturaba la red del robot: se perdían los
    # stateExt por UDP y el iKinCartesianSolver se suspendía (el brazo no se movía).
    return {
        "left": "/teleop/cam/left:o",
        "right": "/teleop/cam/right:o",
    }


# Resolución (alto, ancho) por cámara; la que no aparezca usa _DEFAULT_CAMERA_SHAPE.
_DEFAULT_CAMERA_SHAPE = (240, 320)
_CAMERA_SHAPES = {"front": (480, 640)}

def _get_state(robot_name: str, part_name: str) -> str:
    return f"/{robot_name}/{part_name}/state:o"

@dataclass
@RobotConfig.register_subclass("lerobot_robot_icubrg")
class iCubConfig(RobotConfig):
    name: str = "icub_remote"
    config_path: str = field(default_factory=lambda: str(_default_config_path()))

    # Puertos remotos (deben coincidir con common/yarp_module/teleop_module_sm.py)
    remote_state_port: str = "/teleop/robot_state:o"
    remote_rh_port: str = "/teleop/rh_target:i"
    remote_lh_port: str = "/teleop/lh_target:i"
    remote_hand_cmd_port: str = "/teleop/hand_cmd:i"
    remote_gaze_port: str = "/teleop/gaze_target:i"
    remote_rpc_port: str = "/teleop/rpc:i"

    # --- Observaciones opcionales ---
    use_joint_vel: bool = False
    use_joint_torque: bool = False
    use_external_wrench: bool = False
    use_eef_pose: bool = False
    use_tactile: bool = False

    # Puertos para joint velocity (wholeBodyDynamics o el puerto de estado del robot)
    remote_vel_port_l: str = "/wholeBodyDynamics/left_arm/jointVelocities:o"
    remote_vel_port_r: str = "/wholeBodyDynamics/right_arm/jointVelocities:o"

    # Puertos para joint torque
    remote_torque_port_l: str = "/wholeBodyDynamics/left_arm/torques:o"
    remote_torque_port_r: str = "/wholeBodyDynamics/right_arm/torques:o"

    # Puertos para wrench (fuerza/torque externo estimado en el EEF)
    remote_wrench_port_l: str = "/wholeBodyDynamics/left_arm/cartesianEndEffectorWrench:o"
    remote_wrench_port_r: str = "/wholeBodyDynamics/right_arm/cartesianEndEffectorWrench:o"

    # Puertos para EEF pose del Cartesian Controller [x y z ax ay az theta]
    remote_eef_port_l: str = ""   # derivado de robot_name en __post_init__
    remote_eef_port_r: str = ""

    # Puertos de tacto (skin)
    remote_tactile_port_l: str = ""   # derivado de robot_name en __post_init__
    remote_tactile_port_r: str = ""
    

    # _YAML_CFG = load_yarp_config(config_path)

    control_arms: str = "both"
    use_gaze: bool = False
    actuators_to_use: dict[str, list[str]] = field(default_factory=dict)
    camera_ports: dict[str, str] = field(default_factory=_default_camera_ports)
    camera_shapes: dict[str, tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cfg = load_yarp_config(self.config_path)
        # ICUB_ROBOT_NAME permite al Hub elegir real/Gazebo sin reescribir el YAML.
        robot_name = os.environ.get("ICUB_ROBOT_NAME") or cfg.get("robot_name", "icubSim")
        self.control_arms = _normalize_control_arms(cfg.get("control_arms", self.control_arms))
        self.use_gaze = cfg.get("gaze_ctrl", self.use_gaze)
        self.remote_state_port = cfg.get("remote_state_port", self.remote_state_port)
        self.remote_rh_port = cfg.get("remote_rh_port", self.remote_rh_port)
        self.remote_lh_port = cfg.get("remote_lh_port", self.remote_lh_port)
        self.remote_hand_cmd_port = cfg.get("remote_hand_cmd_port", self.remote_hand_cmd_port)
        self.remote_gaze_port = cfg.get("remote_gaze_port", self.remote_gaze_port)
        self.remote_rpc_port = cfg.get("remote_rpc_port", self.remote_rpc_port)
        if not self.actuators_to_use:
            self.actuators_to_use = _get_actuators_logic(cfg)
        # Derivados de robot_name; el YAML solo los sobreescribe si lo pide explícitamente
        # (camera_ports: {name: port}, camera_shapes: {name: [alto, ancho]}).
        self.camera_ports = cfg.get("camera_ports") or _camera_ports_from_robot_name(robot_name)
        yaml_shapes = cfg.get("camera_shapes") or {}
        self.camera_shapes = {
            name: tuple(yaml_shapes.get(name, _CAMERA_SHAPES.get(name, _DEFAULT_CAMERA_SHAPE)))
            for name in self.camera_ports
        }
        # Flags de observaciones opcionales
        self.use_joint_vel      = cfg.get("use_joint_vel",       self.use_joint_vel)
        self.use_joint_torque   = cfg.get("use_joint_torque",    self.use_joint_torque)
        self.use_external_wrench= cfg.get("use_external_wrench", self.use_external_wrench)
        self.use_eef_pose       = cfg.get("use_eef_pose",        self.use_eef_pose)
        self.use_tactile        = cfg.get("use_tactile",         self.use_tactile)
        # Puertos opcionales sobreescribibles desde YAML
        self.remote_vel_port_l     = cfg.get("remote_vel_port_l",     self.remote_vel_port_l)
        self.remote_vel_port_r     = cfg.get("remote_vel_port_r",     self.remote_vel_port_r)
        self.remote_torque_port_l  = cfg.get("remote_torque_port_l",  self.remote_torque_port_l)
        self.remote_torque_port_r  = cfg.get("remote_torque_port_r",  self.remote_torque_port_r)
        self.remote_wrench_port_l  = cfg.get("remote_wrench_port_l",  self.remote_wrench_port_l)
        self.remote_wrench_port_r  = cfg.get("remote_wrench_port_r",  self.remote_wrench_port_r)
        # Derivados de robot_name: el YAML solo los sobreescribe si lo pide explicitamente.
        self.remote_eef_port_l     = cfg.get("remote_eef_port_l",     f"/{robot_name}/cartesianController/left_arm/state:o")
        self.remote_eef_port_r     = cfg.get("remote_eef_port_r",     f"/{robot_name}/cartesianController/right_arm/state:o")
        self.remote_tactile_port_l = cfg.get("remote_tactile_port_l", f"/{robot_name}/skin/left_arm_comp")
        self.remote_tactile_port_r = cfg.get("remote_tactile_port_r", f"/{robot_name}/skin/right_arm_comp")



    ##############################

    # print(f"Loading configuration from {config_file}...")
    # with open(config_file, 'r') as f:
    #     cfg = yaml.safe_load(f)

    # # --- Parámetros desde YAML ---
    # control_arms = cfg.get("control_arms", "both") # left, right, both
    # primary_arm = cfg.get("primary_arm", "right_arm")
    
    # period = cfg.get("period", 0.02)
    # show_cameras = cfg.get("show_cameras", True)
    # home_pos_deg = cfg.get("home_pose", {})
    # hand_open_values = cfg.get("hand_open_values")
    # hand_closed_values = cfg.get("hand_close_values")
    # use_gaze = cfg.get("gaze_ctrl", False)

    # # --- Filtrar actuadores según configuración ---
    # full_actuators = cfg.get("actuators", {})
    # actuators_to_use = {}
    
    # # Siempre necesitamos torso si usamos algún brazo
    # actuators_to_use["torso"] = full_actuators["torso"]
    
    # cartesian_arms = []
    # if control_arms in ["left", "both"]:
    #     actuators_to_use["left_arm"] = full_actuators["left_arm"]
    #     cartesian_arms.append("left_arm")
    # if control_arms in ["right", "both"]:
    #     actuators_to_use["right_arm"] = full_actuators["right_arm"]
    #     cartesian_arms.append("right_arm")

    # actuators_to_use["head"] = full_actuators["head"]
    # print(f"Active Cartesian Arms: {cartesian_arms}")
    # print(f"Opening drivers for: {list(actuators_to_use.keys())}")
    # print(f"Opening actuators: {list(actuators_to_use.values())}")
