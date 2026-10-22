"""Pantalla de bienvenida: un boton por robot en robots.registry.REGISTRY.

Agregar un robot al registro alcanza para que aparezca aca -- no hay que
tocar este archivo.
"""

from __future__ import annotations

import base64
import io

import gradio as gr
from PIL import Image

from ..paths import ASSETS_DIR
from ..robots.registry import REGISTRY

_LOGO_PAD = 10
_LOGO_HEIGHT_PX = 70
_LOGO_GAP = "25rem"  
_INSTITUTIONAL_LOGOS = (
    ("usm.png", "https://usm.cl", "Universidad Técnica Federico Santa María"),
    ("ac3e.png", "https://ac3e.usm.cl", "AC3E"),
)


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


def _logo_links_html() -> str:
    """Logos institucionales como enlaces (pestaña nueva). gr.Image no admite links, así que
    van en un gr.HTML con cada PNG recortado embebido en base64."""
    links = []
    for filename, url, alt in _INSTITUTIONAL_LOGOS:
        buf = io.BytesIO()
        _trimmed_logo(filename).save(buf, format="PNG")
        src = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        links.append(
            f'<a href="{url}" target="_blank" rel="noopener noreferrer" title="{alt}">'
            f'<img src="{src}" alt="{alt}" style="height:{_LOGO_HEIGHT_PX}px; width:auto; '
            f'display:block;"></a>'
        )
    return ('<div style="display:flex; justify-content:center; align-items:center; '
            f'gap:{_LOGO_GAP};">' + "".join(links) + "</div>")


def build_welcome_screen():
    """Devuelve (welcome_col, buttons) con buttons = {robot_id: gr.Button}."""
    buttons: dict[str, gr.Button] = {}

    with gr.Column(visible=True, elem_classes=["welcome-screen"]) as welcome_col:
        gr.HTML(_logo_links_html(), elem_classes=["welcome-institutional-logos"])
        gr.Markdown(
            "<h1 style='text-align:center; margin-bottom:0; font-size:3.2rem;'>"
            "Welcome to the Experimentation Hub</h1>"
            "<h3 style='text-align:center; font-weight:normal; font-size:1.5rem; "
            "color:var(--body-text-color-subdued);'>"
            "Select your robot</h3>"
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
