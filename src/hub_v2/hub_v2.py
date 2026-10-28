#!/usr/bin/env python3
"""Hub v2 (prototipo de rediseño de la UI). Corre junto al Hub original, en otro puerto.

    pixi run python src/hub_v2/hub_v2.py          # http://localhost:7861  (HUB_V2_PORT para cambiarlo)

Reutiliza la lógica del Hub original (src/hub_dataset/launcher_gui); no correr una sesión
desde los dos Hubs a la vez: usan los mismos puertos del VR y del streaming.
"""

import os
import sys
import warnings
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
# Al correr este archivo, Python pone su carpeta (src/hub_v2/) al frente de sys.path y
# `import hub_v2` encontraría este mismo hub_v2.py en vez del paquete: se saca.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _THIS_DIR]
# hub_dataset/ (launcher_gui), src/ (hub_v2, interfaces) y la raíz al frente de sys.path, como
# hub_experimentacion.py
for _p in (_REPO_ROOT / "src" / "hub_dataset", _REPO_ROOT, _REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
for _key in list(sys.modules):
    if _key == "launcher_gui" or _key.startswith("launcher_gui."):
        del sys.modules[_key]
sys.path.append(os.environ.get("ICUBENV_SITE_PACKAGES",
                               os.path.expanduser("~/miniconda3/envs/icubenv/lib/python3.12/site-packages")))
warnings.filterwarnings("ignore", message=r".*HTTP_422_UNPROCESSABLE_ENTITY.*")
try:  # OpenCV antes que torchvision (ver hub_experimentacion.py)
    import cv2  # noqa: F401,E402
except Exception as _cv2_err:  # noqa: BLE001
    print(f"[Hub v2] OpenCV not available: {_cv2_err!r}")

from launcher_gui import live_view, state  # noqa: E402

from hub_v2.app import build_ui  # noqa: E402
from hub_v2.theme import CSS, JS, THEME  # noqa: E402

if __name__ == "__main__":
    port = int(os.environ.get("HUB_V2_PORT", "7861"))
    print(f"Opening Hub v2 at: http://localhost:{port}")
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=port, share=False, theme=THEME, css=CSS, js=JS,
                allowed_paths=[str(state.viz_tmp_dir)], prevent_thread_lock=True)
    live_view.mount(demo.app)
    demo.block_thread()
