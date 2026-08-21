#!/usr/bin/env python3
"""
Web GUI para sesiones de grabacion de iCub MuJoCo.

Ejecutar con:  python hub_experimentacion.py
Abrir en:      http://localhost:7860
"""

from __future__ import annotations

import os
import sys
import warnings

from launcher_gui.paths import PROJECT_ROOT

# hub_experimentacion.py vive en Hub_dataset/, pero dependencies/, play_mujoco.py,
# utils/ y los paquetes lerobot-* viven en la carpeta hermana Icub-lerobot-mj/.
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "lerobot-teleoperator-icubteleop"))
sys.path.insert(0, str(PROJECT_ROOT / "lerobot-robot-icub"))
sys.path.append(os.path.expanduser("~/miniconda3/envs/icubenv/lib/python3.12/site-packages"))

# gradio 6.18.0 usa internamente el nombre deprecado de Starlette
# HTTP_422_UNPROCESSABLE_ENTITY (routes.py:1402) — cosmetico, no afecta funcionalidad.
warnings.filterwarnings("ignore", message=r".*HTTP_422_UNPROCESSABLE_ENTITY.*")

# ── Warm-up de librerias nativas ──────────────────────────────────────────────
# dependencies.teleop_mujoco (cv2/mujoco/glfw) se importa por primera vez dentro
# del hilo de sesion (robots/icub_sim.py). Si esa carga inicial de librerias nativas
# coincide con el hilo principal de Gradio cargando las suyas, el dynamic linker
# puede fallar con "undefined symbol" de forma espuria. Importarlo aqui, en el
# hilo principal y antes de levantar Gradio, evita la carrera.
try:
    import dependencies.teleop_mujoco  # noqa: F401
except Exception as _warmup_err:
    print(f"[Warmup] No se pudo precargar dependencies.teleop_mujoco: {_warmup_err!r}")

from launcher_gui.app import build_ui, WELCOME_CSS  # noqa: E402  (import tras el warm-up, a proposito)

if __name__ == "__main__":
    print("Abriendo GUI en: http://localhost:7860")
    build_ui().launch(
        server_name="0.0.0.0", server_port=7860, share=False,
        theme="EGOsnm/AMGBitch", css=WELCOME_CSS,
    )
