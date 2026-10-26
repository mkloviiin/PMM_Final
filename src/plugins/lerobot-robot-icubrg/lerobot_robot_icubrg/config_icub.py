from dataclasses import dataclass, field
import os
from pathlib import Path
import yaml

try:
    from lerobot.robots.config import RobotConfig
except ImportError:
    class RobotConfig:
        @classmethod
        def register_subclass(cls, _name: str):
            def _decorator(subcls):
                return subcls
            return _decorator


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "config" / "control_config.yaml").exists():
            return parent
    return Path(__file__).resolve().parents[3]


def _default_config_path() -> Path:
    env_path = os.getenv("ICUB_LEROBOT_CONFIG")
    if env_path:
        return Path(env_path).expanduser()
    return _project_root() / "config" / "control_config.yaml"


def load_control_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path).expanduser() if config_path else _default_config_path()
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _normalize_control_arms(mode: str | None) -> str:
    value = str(mode or "right").strip().lower()
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
    return normalized if normalized in {"left", "right", "both", "none"} else "right"


def _get_actuators_logic(cfg: dict, mode: str | None = None) -> dict[str, list[str]]:
    full = cfg.get("actuators", {})
    if mode is None:
        mode = _normalize_control_arms(cfg.get("control_arms", "right"))
    else:
        mode = _normalize_control_arms(mode)
    use_gaze = bool(cfg.get("gaze_ctrl", True))
    used: dict[str, list[str]] = {}

    if "torso" in full:
        used["torso"] = full["torso"]
    if mode in ["left", "both"] and "left_arm" in full:
        used["left_arm"] = full["left_arm"]
    if mode in ["right", "both"] and "right_arm" in full:
        used["right_arm"] = full["right_arm"]
    if use_gaze and "head" in full:
        used["head"] = full["head"]

    return used


def _detect_robot_model(cfg: dict, explicit_model: str | None = None) -> str:
    if explicit_model:
        return explicit_model
    env_model = os.environ.get("ICUB_ROBOT_MODEL")
    if env_model:
        return env_model
    tmp_model_file = Path("/tmp/icub_gazebo_model")
    if tmp_model_file.exists():
        try:
            m = tmp_model_file.read_text(encoding="utf-8").strip()
            if m:
                return m
        except Exception:
            pass
    return str(cfg.get("robot_model", "iCubGazeboV2_5_visuomanip"))


def _camera_ports_from_robot_name(robot_name: str, robot_model: str | None = None) -> dict[str, str]:
    name_clean = robot_name.strip().lower()
    is_real = name_clean == "icub"
    model_str = (robot_model or "").strip().lower()
    is_visuomanip = "visuomanip" in model_str

    # If running Gazebo simulation and model does not have camera sensors, return empty
    if not is_real and not is_visuomanip:
        return {}

    if name_clean == "icubsim":
        return {
            "left": f"/{robot_name}/cam/left/rgbImage:o",
            "right": f"/{robot_name}/cam/right/rgbImage:o",
            "front": f"/{robot_name}/cam/front/rgbImage:o",
        }
    return {
        "left": "/teleop/cam/left:o",
        "right": "/teleop/cam/right:o",
    }


def _scene_object_bodies() -> list[str]:
    scene_file = Path(os.environ.get("ICUB_SCENE_FILE", "/tmp/icub_lerobot_scene"))
    scenes_yaml = _project_root() / "mujoco_sim" / "assets" / "scenes" / "scenes.yaml"
    try:
        label = scene_file.read_text(encoding="utf-8").strip()
        scenes = yaml.safe_load(scenes_yaml.read_text(encoding="utf-8")) or []
        scene = next(s for s in scenes if s.get("label") == label)
        bodies = [obj["body"] for obj in scene.get("objects", [])]
    except Exception:
        bodies = []
    return bodies or ["blue-cube"]


@dataclass
@RobotConfig.register_subclass("lerobot_robot_icubrg")
class iCubConfig(RobotConfig):
    name: str = "icub_remote"
    config_path: str = field(default_factory=lambda: str(_default_config_path()))

    remote_state_port: str = "/teleop/robot_state:o"
    remote_rh_port: str = "/teleop/rh_target:i"
    remote_lh_port: str = "/teleop/lh_target:i"
    remote_head_port: str = "/teleop/head_target:i"
    remote_hand_cmd_port: str = "/teleop/hand_cmd:i"
    remote_rpc_port: str = "/teleop/rpc:i"

    use_joint_vel: bool = True
    use_joint_torque: bool = True
    use_eef_pose: bool = True
    use_targets: bool = True
    use_tactile: bool = True
    use_object_pose: bool = False

    touch_source: str = "auto"
    remote_touch_port: str = "/teleop/touch:o"
    touch_threshold: float = 100.0 / 12.0
    object_bodies: list[str] = field(default_factory=list)
    gazebo_root_xyz: tuple[float, float, float] = (0.0, 0.0, 0.73)
    gripper_legacy: bool = False

    control_arms: str | None = None
    controller_type: str = "ik"
    impedance_mode: bool = False
    canonical_quats: bool = True
    robot_model: str | None = None
    use_gaze: bool | None = None
    actuators_to_use: dict[str, list[str]] = field(default_factory=dict)
    camera_ports: dict[str, str] = field(default_factory=dict)
    camera_shapes: dict[str, tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cfg = load_control_config(self.config_path)
        robot_name = os.environ.get("ICUB_ROBOT_NAME") or cfg.get("robot_name", "icubSim")
        self.robot_model = _detect_robot_model(cfg, self.robot_model)
        self.controller_type = cfg.get("controller_type", self.controller_type)
        self.impedance_mode = bool(cfg.get("impedance_mode", self.impedance_mode))
        if self.control_arms is None:
            self.control_arms = _normalize_control_arms(cfg.get("control_arms", "right"))
        else:
            self.control_arms = _normalize_control_arms(self.control_arms)
        if self.use_gaze is None:
            self.use_gaze = cfg.get("gaze_ctrl", True)
        self.remote_state_port = cfg.get("remote_state_port", self.remote_state_port)
        self.remote_rh_port = cfg.get("remote_rh_port", self.remote_rh_port)
        self.remote_lh_port = cfg.get("remote_lh_port", self.remote_lh_port)
        self.remote_head_port = cfg.get("remote_head_port", self.remote_head_port)
        self.remote_hand_cmd_port = cfg.get("remote_hand_cmd_port", self.remote_hand_cmd_port)
        self.remote_rpc_port = cfg.get("remote_rpc_port", self.remote_rpc_port)

        if not self.actuators_to_use:
            self.actuators_to_use = _get_actuators_logic(cfg, mode=self.control_arms)

        show_cameras = bool(cfg.get("show_cameras", True))
        if not show_cameras:
            self.camera_ports = {}
        elif not self.camera_ports:
            ports = cfg.get("camera_ports")
            if ports is None:
                ports = _camera_ports_from_robot_name(robot_name, self.robot_model)
            self.camera_ports = ports or {}

        yaml_shapes = cfg.get("camera_shapes") or {}
        self.camera_shapes = {
            name: tuple(yaml_shapes.get(name, (240, 320)))
            for name in self.camera_ports
        }

        self.use_joint_vel = cfg.get("use_joint_vel", self.use_joint_vel)
        self.use_joint_torque = cfg.get("use_joint_torque", self.use_joint_torque)
        self.use_eef_pose = cfg.get("use_eef_pose", self.use_eef_pose)
        self.use_targets = cfg.get("use_targets", self.use_targets)
        self.use_tactile = cfg.get("use_tactile", self.use_tactile)
        self.use_object_pose = (cfg.get("use_object_pose", self.use_object_pose) and robot_name.strip().lower() == "icubsim")

        self.touch_source = cfg.get("touch_source", self.touch_source)
        if self.touch_source == "auto":
            self.touch_source = "gazebo" if robot_name.strip().lower() == "icubsim" else "skin"
        self.touch_threshold = float(cfg.get("touch_threshold", self.touch_threshold))
        self.gazebo_root_xyz = tuple(cfg.get("gazebo_root_xyz", self.gazebo_root_xyz))
        if not self.object_bodies:
            self.object_bodies = cfg.get("object_bodies") or _scene_object_bodies()
