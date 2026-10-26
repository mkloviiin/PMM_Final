"""Campo de texto con un botón de carpeta dentro (como la flecha de un Dropdown) que abre
el selector de carpetas del sistema. Lo usan "Dataset root directory" (Configuration) y
"Datasets directory" (Visualize & Curate). El estilo está en app.py (.dir-tb).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import gradio as gr

_TK_PICKER = (
    "import sys, tkinter as tk\n"
    "from tkinter import filedialog\n"
    "root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)\n"
    "print(filedialog.askdirectory(initialdir=sys.argv[1], title=sys.argv[2]) or '')\n"
)


def _ask_directory(start: Path, title: str) -> str | None:
    """Abre el selector de carpetas en la pantalla del PC donde corre el Hub (el navegador
    no puede entregar rutas del servidor). zenity si está (Ubuntu); si no, tkinter (p. ej.
    en Docker). Devuelve None si se cancela."""
    if shutil.which("zenity"):
        cmd = ["zenity", "--file-selection", "--directory", f"--title={title}", f"--filename={start}/"]
    else:
        cmd = [sys.executable, "-c", _TK_PICKER, str(start), title]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.stdout.strip() or None


def pick_directory(current: str, title: str = "Select a folder"):
    if not os.environ.get("DISPLAY"):
        gr.Warning("No display available on the Hub machine: type the path instead.")
        return gr.update()
    start = Path(current).expanduser() if current else Path.home()
    if not start.is_dir():
        start = Path.home()
    try:
        chosen = _ask_directory(start, title)
    except Exception as e:  # noqa: BLE001
        gr.Warning(f"Could not open the folder picker: {e}")
        return gr.update()
    return gr.update(value=chosen) if chosen else gr.update()


def dir_textbox(label: str, value: str, title: str | None = None, **kwargs):
    """Crea el campo con el botón de carpeta dentro. El botón va en `buttons=` del propio
    Textbox (no en una fila aparte), así no corta el panel del formulario; el CSS de app.py
    (.dir-tb) lo baja dentro del cuadro de texto. Devuelve (textbox,
    evento_click) para encadenar acciones con `.then(...)` después de elegir la carpeta."""
    btn = gr.Button("", render=False)   # el ícono de carpeta lo dibuja el CSS
    tb = gr.Textbox(value=value, label=label, buttons=[btn], elem_classes=["dir-tb"], **kwargs)
    event = btn.click(fn=lambda cur: pick_directory(cur, title or f"Select: {label}"),
                      inputs=[tb], outputs=[tb])
    return tb, event
