from dataclasses import dataclass, field
import os
from pathlib import Path

import yaml

from lerobot.robots.config import RobotConfig


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


def _default_model_path() -> str:
    env_path = os.getenv("ICUB_MUJOCO_MODEL_PATH")
    if env_path:
        return str(Path(env_path).expanduser())
    return str(_project_root() / "mujoco_sim" / "assets" / "scenes" / "icub_Activate_table_scene.xml")


def _default_vr_enabled() -> bool:
    val = os.getenv("ICUB_MUJOCO_VR_ENABLED", "0").strip().lower()
    return val in {"1", "true", "yes", "on"}


def _default_vr_ip() -> str | None:
    val = os.getenv("ICUB_MUJOCO_VR_IP", "").strip()
    return val or None


def load_control_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path).expanduser() if config_path else _default_config_path()
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _get_actuators_logic(cfg: dict, mode: str | None = None) -> dict[str, list[str]]:
    full = cfg.get("actuators", {})
    if mode is None:
        mode = cfg.get("control_arms", "both")
    used: dict[str, list[str]] = {}

    if "torso" in full:
        used["torso"] = full["torso"]
    if mode in ["left", "both"] and "left_arm" in full:
        used["left_arm"] = full["left_arm"]
    if mode in ["right", "both"] and "right_arm" in full:
        used["right_arm"] = full["right_arm"]
    if "head" in full:
        used["head"] = full["head"]

    return used


@RobotConfig.register_subclass("lerobot_robot_icub_mujoco")
@dataclass
class iCubMuJoCoConfig(RobotConfig):
    name: str = "icub_mujoco"

    model_path: str = field(default_factory=_default_model_path)
    config_path: str = field(default_factory=lambda: str(_default_config_path()))
    control_arms: str | None = None
    use_gaze: bool | None = None
    controller_type: str | None = None       # "ik" (TSIK) or "id" (TSID)
    impedance_mode: bool | None = None      # Compliant Cartesian impedance when controller_type == "id"
    actuators_to_use: dict[str, list[str]] = field(default_factory=dict)

    # top = overhead camera (topview); a task picks its cameras with `cameras` in scenes.yaml
    # (the door records left/right/top instead of front).
    camera_names: list[str] = field(default_factory=lambda: ["left", "right", "front"])
    camera_sources: dict[str, str] = field(default_factory=lambda: {
        "left": "head_cam", "right": "head_cam_right", "front": "frontview", "top": "topview"
    })
    camera_width: int = 320
    camera_height: int = 240
    pose_frame: str = "root"
    gripper_legacy: bool = False
    canonical_quats: bool = True

    sim_dt: float = 0.005
    control_dt: float = 1.0 / 30  # 30 Hz
    primary_arm: str = "right"
    vr_enabled: bool = field(default_factory=_default_vr_enabled)
    vr_ip: str | None = field(default_factory=_default_vr_ip)
    scene_objects: list = field(default_factory=list)
    scene_joints: list = field(default_factory=list)

    def __post_init__(self) -> None:
        cfg = load_control_config(self.config_path)
        if self.control_arms is None:
            self.control_arms = cfg.get("control_arms", "both")
        if self.use_gaze is None:
            self.use_gaze = cfg.get("gaze_ctrl", True)
        if self.controller_type is None:
            self.controller_type = os.getenv("ICUB_CONTROLLER_TYPE", cfg.get("controller_type", "ik"))
        if self.impedance_mode is None:
            env_imp = os.getenv("ICUB_IMPEDANCE_MODE")
            if env_imp is not None:
                self.impedance_mode = env_imp.strip().lower() in {"1", "true", "yes", "on"}
            else:
                self.impedance_mode = bool(cfg.get("impedance_mode", False))
        self.vr_enabled = cfg.get("vr_enabled", self.vr_enabled)
        self.vr_ip = cfg.get("vr_ip", self.vr_ip)
        self.pose_frame = cfg.get("pose_frame", self.pose_frame)
        if not self.actuators_to_use:
            self.actuators_to_use = _get_actuators_logic(cfg, mode=self.control_arms)

