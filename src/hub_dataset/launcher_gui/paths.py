"""Rutas compartidas del Hub de experimentación.

El Hub es agnóstico al robot: aquí viven solo la ruta del propio Hub, la de
sus assets y las raíces de los proyectos hermanos.
"""

from __future__ import annotations

from pathlib import Path

# launcher_gui/ vive dentro de src/hub_dataset/.
HUB_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = HUB_ROOT / "assets"

REPO_ROOT = HUB_ROOT.parent.parent
PLAY_SCRIPT = REPO_ROOT / "play.py"
MUJOCO_ASSETS = REPO_ROOT / "mujoco_sim" / "assets"
GAZEBO_ROOT = REPO_ROOT / "gazebo_sim"
