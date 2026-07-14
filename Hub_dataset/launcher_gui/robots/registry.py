"""Registro de robots disponibles en el Hub.

Para agregar un robot nuevo: crear `robots/<robot>.py` con su propio
`build_config_form()`/`launch()` que respeten el contrato de `RobotBackend`
(ver base.py) y sumar su `BACKEND` aca. No hace falta tocar la UI compartida
(welcome, tab_config, tab_episodes, tab_curation) ni las pestanas de
curacion/subida a Hub -- se arman solas a partir de esta lista.
"""

from __future__ import annotations

from ..paths import ASSETS_DIR
from .base import RobotBackend
from . import icub_sim

# Placeholder: iCub real todavia no tiene backend implementado. Cuando se
# agregue (con su propio control remoto en vez de VR), crear
# robots/icub_real.py siguiendo el mismo patron que icub_sim.py y reemplazar
# esta entrada por `icub_real.BACKEND`.
_ICUB_REAL_PLACEHOLDER = RobotBackend(
    id="icub_real",
    label="iCub Real",
    image=ASSETS_DIR / "icub.png",
    available=False,
)

REGISTRY: list[RobotBackend] = [
    icub_sim.BACKEND,
    _ICUB_REAL_PLACEHOLDER,
]
