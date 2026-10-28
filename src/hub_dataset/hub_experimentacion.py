#!/usr/bin/env python3
"""
Web GUI para grabar datasets LeRobot con iCub (real, Gazebo o MuJoCo).

Los backends disponibles se declaran en
``launcher_gui/robots/registry.py`` y cada uno hace sus imports de forma
perezosa al lanzar. Este arranque solo prepara el sys.path del propio Hub.

Ejecutar con:  python hub_experimentacion.py
Abrir en:      http://localhost:7860
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

# ── hub_dataset/ al FRENTE de sys.path ───────────────────────────────────────
# Sin esto, `from launcher_gui... import ...` puede importar un launcher_gui
# stale de otra ruta (a/PMM_Final, Papelera, etc.) que ya esté en sys.path.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
for _p in (_THIS_DIR, _REPO_ROOT, _REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── Desalojar cualquier launcher_gui ya cacheado ─────────────────────────────
# Python no vuelve a buscar un módulo ya en sys.modules aunque cambie sys.path:
# si YARP u otro paquete lo importó desde otra ruta, se descarta aquí.
_STALE_PREFIXES = ("launcher_gui",)
for _key in list(sys.modules.keys()):
    if any(_key == p or _key.startswith(p + ".") for p in _STALE_PREFIXES):
        del sys.modules[_key]

# Fallback para entornos donde el python del Hub no trae yarp/lerobot en su
# propio site-packages (no es específico de ningún robot).
sys.path.append(os.environ.get("ICUBENV_SITE_PACKAGES", os.path.expanduser("~/miniconda3/envs/icubenv/lib/python3.12/site-packages")))

# gradio 6.18.0 usa internamente el nombre deprecado de Starlette
warnings.filterwarnings("ignore", message=r".*HTTP_422_UNPROCESSABLE_ENTITY.*")

# OpenCV antes que cualquier otra cosa: si torchvision (lo importa lerobot al lanzar MuJoCo
# en este mismo proceso) se carga primero, trae su propia libjpeg y después `import cv2`
# falla con "libtiff.so.6: undefined symbol: jpeg12_write_raw_data" -> sin streaming al
# visor VR.
try:
    import cv2  # noqa: F401,E402
except Exception as _cv2_err:  # noqa: BLE001
    print(f"[Hub] OpenCV not available (no image streaming to the VR): {_cv2_err!r}")

from launcher_gui.app import build_ui, WELCOME_CSS  # noqa: E402
from launcher_gui import live_view, state  # noqa: E402

if __name__ == "__main__":
    print("Opening the GUI at: http://localhost:7860")
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0", server_port=7860, share=False,
        theme="EGOsnm/AMGBitch", css=WELCOME_CSS,
        # Los videos recortados se sirven via /gradio_api/file= desde un <video> HTML.
        allowed_paths=[str(state.viz_tmp_dir)],
        prevent_thread_lock=True,
    )
    # Vista en vivo de Episode Control: su stream lo sirve el mismo servidor de Gradio
    live_view.mount(demo.app)
    demo.block_thread()
