from .config_icub_mujoco import iCubMuJoCoConfig

try:
    from .icub_mujoco import iCubMuJoCo
except ModuleNotFoundError:
    iCubMuJoCo = None

__all__ = ["iCubMuJoCoConfig", "iCubMuJoCo"]
