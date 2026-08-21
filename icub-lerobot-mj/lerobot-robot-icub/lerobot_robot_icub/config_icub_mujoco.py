from dataclasses import dataclass, field
import os
from pathlib import Path

import yaml

from lerobot.robots.config import RobotConfig


def _default_config_path() -> Path:
    env_path = os.getenv("ICUB_LEROBOT_CONFIG")
    if env_path:
        return Path(env_path).expanduser()

    candidates = [
        Path.cwd() / "icub-lerobot" / "utils" / "control_config.yaml",
        Path.cwd() / "utils" / "control_config.yaml",
        Path(__file__).resolve().parents[2] / "utils" / "control_config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _default_model_path() -> str:
    env_path = os.getenv("ICUB_MUJOCO_MODEL_PATH")
    if env_path:
        return str(Path(env_path).expanduser())

    candidates = [
        Path.cwd() / "icub-lerobot" / "model" / "xml" / "scene.xml",
        Path.cwd() / "model" / "xml" / "scene.xml",
        Path(__file__).resolve().parents[2] / "model" / "xml" / "scene.xml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


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


def _get_actuators_logic(cfg: dict) -> dict[str, list[str]]:
    full = cfg.get("actuators", {})
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
    control_arms: str = "both"
    use_gaze: bool = False
    actuators_to_use: dict[str, list[str]] = field(default_factory=dict)

    # head_cam           — cámara fija en la cabeza del robot (vista global)
    # head_cam_track_hand — siempre apuntar a la mano derecha
    # frontview           — cámara fija frontal externa
    camera_names: list[str] = field(default_factory=lambda: ["head_cam", "head_cam_track_hand", "frontview"])
    camera_width: int = 320
    camera_height: int = 240

    sim_dt: float = 0.005
    control_dt: float = 1.0 / 30  # 30 Hz
    primary_arm: str = "right"
    vr_enabled: bool = field(default_factory=_default_vr_enabled)
    vr_ip: str | None = field(default_factory=_default_vr_ip)

    def __post_init__(self) -> None:
        cfg = load_control_config(self.config_path)
        self.control_arms = cfg.get("control_arms", self.control_arms)
        self.use_gaze = cfg.get("gaze_ctrl", self.use_gaze)
        self.vr_enabled = cfg.get("vr_enabled", self.vr_enabled)
        self.vr_ip = cfg.get("vr_ip", self.vr_ip)
        if not self.actuators_to_use:
            self.actuators_to_use = _get_actuators_logic(cfg)
