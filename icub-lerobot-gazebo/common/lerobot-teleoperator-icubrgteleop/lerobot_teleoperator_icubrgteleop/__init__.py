from .config_icubteleop import iCubTeleopConfig

try:
	from .icubteleop import iCubTeleop
except ModuleNotFoundError:
	iCubTeleop = None

__all__ = ["iCubTeleopConfig", "iCubTeleop"]