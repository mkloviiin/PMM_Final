"""Registro de robots disponibles en el Hub.

Para agregar un robot nuevo: crear `robots/<robot>.py` con su propio
`build_config_form()`/`launch()` que respeten el contrato de `RobotBackend`
(ver base.py) y sumar su `BACKEND` aca. No hace falta tocar la UI compartida
(welcome, tab_config, tab_episodes, tab_curation) ni las pestanas de
curacion/subida a Hub -- se arman solas a partir de esta lista.

Patron para escalar a N robots: cada backend es independiente y decide como
corre (in-process como `icub_sim`, o en subproceso aislado como `icub_rg`).
No hay un robot "base". El Hub serializa: solo una sesion de grabacion activa
a la vez (ver session.py), asi que sumar robots no multiplica procesos vivos.
"""

from __future__ import annotations

from .base import RobotBackend
from . import icub_rg, icub_sim

REGISTRY: list[RobotBackend] = [
    icub_sim.BACKEND,  # iCub MuJoCo (simulacion de fisica, in-process)
    icub_rg.BACKEND,   # iCub Real / Gazebo (via YARP, subproceso aislado)
]
