"""Pantalla de bienvenida: un boton por robot en robots.registry.REGISTRY.

Agregar un robot al registro alcanza para que aparezca aca -- no hay que
tocar este archivo.
"""

from __future__ import annotations

import gradio as gr
from PIL import Image

from ..paths import ASSETS_DIR
from ..robots.registry import REGISTRY

_LOGO_PAD = 10


def _trimmed_logo(filename: str) -> Image.Image:
    """Recorta el margen transparente de un logo para que todos se vean con
    un peso visual comparable al ponerlos uno al lado del otro."""
    im = Image.open(ASSETS_DIR / filename).convert("RGBA")
    bbox = im.getbbox()
    if not bbox:
        return im
    left, top, right, bottom = bbox
    left = max(left - _LOGO_PAD, 0)
    top = max(top - _LOGO_PAD, 0)
    right = min(right + _LOGO_PAD, im.width)
    bottom = min(bottom + _LOGO_PAD, im.height)
    return im.crop((left, top, right, bottom))


def build_welcome_screen():
    """Devuelve (welcome_col, buttons) con buttons = {robot_id: gr.Button}."""
    buttons: dict[str, gr.Button] = {}

    with gr.Column(visible=True, elem_classes=["welcome-screen"]) as welcome_col:
        with gr.Row(elem_classes=["welcome-institutional-logos"]):
            gr.Image(
                _trimmed_logo("usm.png"),
                show_label=False, interactive=False, container=False,
                height=70, scale=0, buttons=[],
            )
            gr.Image(
                _trimmed_logo("ac3e.png"),
                show_label=False, interactive=False, container=False,
                height=70, scale=0, buttons=[],
            )
        gr.Markdown(
            "<h1 style='text-align:center; margin-bottom:0; font-size:3.2rem;'>"
            "Bienvenido/a al Hub de Experimentación</h1>"
            "<h3 style='text-align:center; font-weight:normal; font-size:1.5rem; "
            "color:var(--body-text-color-subdued);'>"
            "Selecciona tu robot</h3>"
        )
        with gr.Row():
            for robot in REGISTRY:
                with gr.Column():
                    gr.Image(
                        str(robot.image),
                        show_label=False, interactive=False,
                        container=False, height=280, buttons=[],
                    )
                    variant = "primary" if robot.available else "secondary"
                    buttons[robot.id] = gr.Button(robot.label, variant=variant, size="lg")

    return welcome_col, buttons
