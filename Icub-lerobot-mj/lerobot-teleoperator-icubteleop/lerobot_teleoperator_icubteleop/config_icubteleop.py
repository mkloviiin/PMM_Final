from dataclasses import dataclass, field
import os
from pathlib import Path
import yaml

try:
    from lerobot.teleoperators.config import TeleoperatorConfig
except Exception:
    class TeleoperatorConfig:
        @classmethod
        def register_subclass(cls, _name: str):
            def _decorator(subcls):
                return subcls
            return _decorator

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
        Path.cwd() / "icub-lerobot" / "dependencies" / "assets" / "scenes" / "icub_table_scene.xml",
        Path.cwd() / "dependencies" / "assets" / "scenes" / "icub_table_scene.xml",
        Path.cwd() / "icub-lerobot" / "model" / "xml" / "scene.xml",
        Path.cwd() / "model" / "xml" / "scene.xml",
        Path(__file__).resolve().parents[2] / "model" / "xml" / "scene.xml",
        Path(__file__).resolve().parents[2] / "dependencies" / "assets" / "scenes" / "icub_table_scene.xml",
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


def load_yarp_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path).expanduser() if config_path else _default_config_path()
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _get_actuators_logic(cfg: dict) -> dict[str, list[str]]:
    full = cfg.get("actuators", {})
    mode = cfg.get("control_arms", "both")
    used: dict[str, list[str]] = {}
    if "torso" in full: used["torso"] = full["torso"]
    if mode in ["left", "both"] and "left_arm" in full: used["left_arm"] = full["left_arm"]
    if mode in ["right", "both"] and "right_arm" in full: used["right_arm"] = full["right_arm"]
    if "head" in full: used["head"] = full["head"]
    return used

@TeleoperatorConfig.register_subclass("icub_teleop_mujoco")
@dataclass
class iCubTeleopConfig(TeleoperatorConfig):
    name: str = "icub_teleop_mujoco"

    model_path: str = field(default_factory=_default_model_path)
    config_path: str = field(default_factory=lambda: str(_default_config_path()))
    control_arms: str = "both"
    use_gaze: bool = False
    strict_xml_init: bool = False
    vr_enabled: bool = field(default_factory=_default_vr_enabled)
    vr_ip: str | None = field(default_factory=_default_vr_ip)
    actuators_to_use: dict[str, list[str]] = field(default_factory=dict)
    cube_spawn: dict = field(default_factory=lambda: {"x": [0.3, 0.4], "y": [-0.2, 0.2], "z": 0.725})

    def __post_init__(self) -> None:
        cfg = load_yarp_config(self.config_path)
        self.control_arms = cfg.get("control_arms", self.control_arms)
        self.use_gaze = cfg.get("gaze_ctrl", self.use_gaze)
        self.strict_xml_init = cfg.get("strict_xml_init", self.strict_xml_init)
        self.vr_enabled = cfg.get("vr_enabled", self.vr_enabled)
        self.vr_ip = cfg.get("vr_ip", self.vr_ip)
        self.cube_spawn = cfg.get("cube_spawn", self.cube_spawn)
        if not self.actuators_to_use:
            self.actuators_to_use = _get_actuators_logic(cfg)