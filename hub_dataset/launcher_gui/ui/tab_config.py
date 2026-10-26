"""Pestana 'Configuración': delega en el formulario propio de cada robot
disponible. Solo queda visible el del robot elegido en la bienvenida -- ver
app.py, que alterna la Column visible al mismo tiempo que navega a main_col.
"""

from __future__ import annotations

import gradio as gr

from ..robots.registry import REGISTRY

CONFIG_TAB_ID = "config"
ROBOT_SHOWN = ["robot-col"]
ROBOT_HIDDEN = ["robot-col", "robot-hidden"]


def build_config_tab():
    """Devuelve (columns, forms):
      columns = {robot_id: gr.Column}      -- para alternar visibilidad
      forms   = {robot_id: {campo: componente}}  -- para el wiring de cada robot
    """
    columns: dict[str, gr.Column] = {}
    forms: dict[str, dict] = {}

    with gr.Tab("1. Configuration", id=CONFIG_TAB_ID):
        available = [r for r in REGISTRY if r.available and r.build_config_form]
        for robot in available:
            # Todas "visibles" para Gradio y ocultas con la clase CSS .robot-hidden; el botón del
            # robot en la bienvenida (app.py) cambia la clase. Con visible=False/True Gradio 6 no
            # dibuja la columna que se muestra por primera vez después de elegir otro robot
            # (p. ej. MuJoCo -> Back -> Gazebo deja Configuration en blanco).
            with gr.Column(elem_classes=ROBOT_HIDDEN) as col:
                forms[robot.id] = robot.build_config_form()
            columns[robot.id] = col

    return columns, forms
