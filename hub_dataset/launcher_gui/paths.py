"""Rutas compartidas del Hub de experimentacion.

El Hub es **agnóstico al robot**: aquí viven solo la ruta del propio Hub, la de
sus assets y las raíces de los proyectos hermanos (una por backend de robot).
No hay un robot "base" — cada backend en `robots/*.py` usa la raíz que le
corresponde y hace sus imports de forma perezosa.
"""

from __future__ import annotations

from pathlib import Path

# launcher_gui/ vive dentro de hub_dataset/.
HUB_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = HUB_ROOT / "assets"

# Proyectos hermanos, uno por backend de robot (ver robots/registry.py):
#   - icub-lerobot-mj      → backend icub_sim  (MuJoCo, corre in-process)
#   - icub-lerobot-gazebo  → backend icub_rg   (real/Gazebo vía YARP, subproceso)
MJ_ROOT = HUB_ROOT.parent / "icub-lerobot-mj"
GAZEBO_ROOT = HUB_ROOT.parent / "icub-lerobot-gazebo"
