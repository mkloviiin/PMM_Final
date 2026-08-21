"""Pestana 'Configuración': delega en el formulario propio de cada robot
disponible. Solo queda visible el del robot elegido en la bienvenida -- ver
app.py, que alterna la Column visible al mismo tiempo que navega a main_col.
"""

from __future__ import annotations

import gradio as gr

from ..robots.registry import REGISTRY


def build_config_tab():
    """Devuelve (columns, forms):
      columns = {robot_id: gr.Column}      -- para alternar visibilidad
      forms   = {robot_id: {campo: componente}}  -- para el wiring de cada robot
    """
    columns: dict[str, gr.Column] = {}
    forms: dict[str, dict] = {}

    with gr.Tab("Configuration"):
        available = [r for r in REGISTRY if r.available and r.build_config_form]
        for i, robot in enumerate(available):
            with gr.Column(visible=(i == 0)) as col:
                forms[robot.id] = robot.build_config_form()
            columns[robot.id] = col

    return columns, forms
