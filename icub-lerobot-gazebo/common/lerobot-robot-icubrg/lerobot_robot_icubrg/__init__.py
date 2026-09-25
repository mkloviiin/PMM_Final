from .config_icub import iCubConfig

try:
    from .icub import iCub
except ModuleNotFoundError:
    iCub = None

__all__ = ["iCubConfig", "iCub"]
