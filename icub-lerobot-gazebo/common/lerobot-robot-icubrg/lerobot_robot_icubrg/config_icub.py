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

def _scene_object_bodies() -> list[str]:
    """Cuerpos de la escena activa en el orden de scenes.yaml (obj0, obj1, ...).

    La escena es la etiqueta que el Hub escribe en $ICUB_SCENE_FILE (la misma que lee
    gazebo/scripts/reset_objects.sh). Sin escena: blue-cube, como el fallback de MuJoCo.
    """
    scene_file = Path(os.environ.get("ICUB_SCENE_FILE", "/tmp/icub_lerobot_scene"))
    scenes_yaml = _project_root() / "mujoco" / "assets" / "scenes" / "scenes.yaml"
    try:
        label = scene_file.read_text(encoding="utf-8").strip()
        scenes = yaml.safe_load(scenes_yaml.read_text(encoding="utf-8")) or []
        scene = next(s for s in scenes if s.get("label") == label)
        bodies = [obj["body"] for obj in scene.get("objects", [])]
    except (OSError, StopIteration, KeyError, TypeError, yaml.YAMLError):
        bodies = []
    return bodies or ["blue-cube"]


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

    # --- Observaciones extra ---
    # Mismos nombres y unidades que el plugin de MuJoCo (lerobot_robot_icub), para poder
    # evaluar en un simulador políticas entrenadas en el otro. Diferencia: aquí todo va
    # en el frame root del iCub y en MuJoCo en el mundo MuJoCo (ver README, sim-to-sim).
    use_joint_vel: bool = False      # {joint}_vel [rad/s]      ← /teleop/robot_state:o
    use_joint_torque: bool = False   # {joint}_torque [Nm]      ← /teleop/robot_state:o
    use_eef_pose: bool = False       # {rh,lh}_ee_pos_*/_ee_quat_*  ← /teleop/{rh,lh}_current_pose:o
    use_targets: bool = False        # {rh,lh}_target_*, gaze_target_pos_*, {rh,lh}_gripper (último comando)
    use_tactile: bool = False        # {rh,lh}_touch_{thumb,index,middle} (0/1), ver touch_source
    use_object_pose: bool = False    # obj{i}_pos_*/_quat_* (ground truth) ← gz /model/<body>/pose

    # De dónde sale el tacto: "gazebo" = sensores de contacto de las puntas (ambas manos,
    # model.urdf), "skin" = piel del robot republicada por teleop_module_sm (solo mano
    # derecha). "auto": gazebo con robot_name icubSim, skin con el robot real.
    touch_source: str = "auto"
    remote_touch_port: str = "/teleop/touch:o"
    # Media de los 12 taxels de un dedo a partir de la cual cuenta como contacto
    # (el mismo umbral que usa teleop_module_sm para sus avisos).
    touch_threshold: float = 100.0 / 12.0
    # Mirada antes del primer comando: la que fija play_gazebo.py (look_at -0.5 0.0 0.3).
    gaze_target_init: tuple[float, float, float] = (-0.5, 0.0, 0.3)
    # Modelos de Gazebo en el orden obj0, obj1, ... (vacío = objetos de la escena activa
    # en scenes.yaml, igual que MuJoCo; sin escena, blue-cube).
    object_bodies: list[str] = field(default_factory=list)
    # root_link del iCub en el mundo Gazebo (<pose> del iCub en gazebo/worlds/icub_world.sdf).
    gazebo_root_xyz: tuple[float, float, float] = (0.0, 0.0, 0.73)
    # rh/lh_gripper: False = fracción de cierre 0..1 (desde el 2026-10-21, gatillo gradual);
    # True = convención anterior 0 abrir / 0.5 cerrar / 1 mantener (modelos legacy).
    gripper_legacy: bool = False
    # Signo de los cuaterniones (QUAT_HEMISPHERE en icub.py): True desde el 2026-10-21. Los
    # eval de políticas entrenadas con datasets anteriores lo apagan (no marcados en info.json).
    canonical_quats: bool = True


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
        # Flags de observaciones extra
        self.use_joint_vel    = cfg.get("use_joint_vel",    self.use_joint_vel)
        self.use_joint_torque = cfg.get("use_joint_torque", self.use_joint_torque)
        self.use_eef_pose     = cfg.get("use_eef_pose",     self.use_eef_pose)
        self.use_targets      = cfg.get("use_targets",      self.use_targets)
        self.use_tactile      = cfg.get("use_tactile",      self.use_tactile)
        # Ground truth del simulador: en el robot real no hay de dónde leerla
        self.use_object_pose  = (cfg.get("use_object_pose", self.use_object_pose)
                                 and robot_name.strip().lower() == "icubsim")
        self.remote_touch_port = cfg.get("remote_touch_port", self.remote_touch_port)
        self.touch_source = cfg.get("touch_source", self.touch_source)
        if self.touch_source == "auto":
            self.touch_source = "gazebo" if robot_name.strip().lower() == "icubsim" else "skin"
        self.touch_threshold = float(cfg.get("touch_threshold", self.touch_threshold))
        self.gaze_target_init = tuple(cfg.get("gaze_target_init", self.gaze_target_init))
        self.gazebo_root_xyz = tuple(cfg.get("gazebo_root_xyz", self.gazebo_root_xyz))
        if not self.object_bodies:
            self.object_bodies = cfg.get("object_bodies") or _scene_object_bodies()



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
