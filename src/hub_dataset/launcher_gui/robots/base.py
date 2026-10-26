"""Contrato que debe cumplir cualquier backend de robot para integrarse al Hub.

Dos partes son obligatorias para que el resto del Hub (Control de episodios,
Visualizar y curar, Subir a Hugging Face) funcione sin modificarse sin
importar el robot:

  1. Formato de dataset -- `launch()` (via `session.start_session`) debe
     terminar escribiendo el dataset grabado en formato LeRobot (meta/info.json,
     data/*.parquet, videos/*.mp4) bajo el `dataset_root` que arma el backend.
     Esto es lo unico que necesitan curation.py y hub_push.py.

  2. Protocolo de sesion -- la grabacion real corre a traves de
     `session.start_session(record_fn, ...)`, y `record_fn(cmd_source, on_status)`
     debe reportar su estado y escuchar comandos por esos mismos callbacks (ver
     launcher_gui/session.py). No importa si el robot se controla por VR, por
     un control remoto fisico o de otra forma: mientras respete ese protocolo,
     "Control de episodios" no necesita saber nada del robot.

Lo unico que varia por robot es su formulario de configuracion propio
(build_config_form) y como arma `record_fn`/`launch` a partir de esos valores
-- eso vive encapsulado en el modulo de cada robot (ver icub_sim.py).

`config_input_order` es lo que le permite a app.py conectar el boton
"LANZAR SESIÓN" de cualquier robot sin conocer de antemano los parametros de
su `launch()`: es la lista de keys de `build_config_form()` en el mismo orden
en que `launch()` espera recibirlas como argumentos posicionales.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import gradio as gr


@dataclass
class RobotBackend:
    id: str
    label: str
    image: Path
    # False -> el boton de bienvenida solo avisa "Aun no disponible".
    available: bool = True
    # Crea los componentes propios de este robot en la pestana "Configuración"
    # (debe llamarse dentro de un `with gr.Column():` ya abierto) y devuelve un
    # dict {nombre_campo: componente} para que app.py arme el wiring.
    build_config_form: Callable[[], dict[str, gr.components.Component]] | None = None
    # launch() recibe los valores de esos componentes como argumentos
    # posicionales, en el mismo orden que config_input_order (una lista de
    # keys de build_config_form). Devuelve el mensaje de estado a mostrar.
    launch: Callable[..., str] | None = None
    config_input_order: list[str] = field(default_factory=list)
