"""Rutas compartidas del Hub de experimentacion."""

from __future__ import annotations

from pathlib import Path

# launcher_gui/ vive en Hub_dataset/, pero dependencies/, play_mujoco.py, utils/
# y los paquetes lerobot-* viven en la carpeta hermana Icub-lerobot-mj/.
HUB_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = HUB_ROOT.parent / "icub-lerobot-mj"
ASSETS_DIR = HUB_ROOT / "assets"
