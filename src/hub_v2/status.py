"""Estado visible del Hub v2: barra superior, panel de grabación y progreso.

Todo sale de launcher_gui.state (lo mismo que usa el Hub original); aquí solo se dibuja.
"""

from __future__ import annotations

import base64
import html
import io

from launcher_gui import state
from launcher_gui.ui.welcome import _trimmed_logo

# estado de la sesión -> (texto, clase de color)
_STATE = {
    "idle": ("No session", "v2-off"),
    "starting": ("Starting…", "v2-warn"),
    "waiting": ("Ready", "v2-warn"),
    "recording": ("Recording", "v2-rec"),
    "done": ("Session finished", "v2-ok"),
    "error": ("Error", "v2-err"),
}


def _logos_html() -> str:
    out = []
    for name, alt in (("usm.png", "USM"), ("ac3e.png", "AC3E")):
        try:
            buf = io.BytesIO()
            _trimmed_logo(name).save(buf, format="PNG")
            out.append(f'<img alt="{alt}" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">')
        except Exception:  # noqa: BLE001
            pass
    return "".join(out)


_LOGOS = _logos_html()


_ready_seq = -1  # última sesión (state.session_seq) en que play.py ya pidió START


def _play_ready() -> bool:
    """La sesión marca 'waiting' apenas se lanza, pero play.py tarda 10-20 s en conectar: está
    lista recién cuando su log dice "Waiting START" (el log se limpia al lanzar cada sesión)."""
    global _ready_seq
    if _ready_seq != state.session_seq:
        with state.log_lock:
            if any("Waiting START" in line for line in state.log_lines):
                _ready_seq = state.session_seq
    return _ready_seq == state.session_seq


def current_state() -> str:
    """Estado a mostrar: 'idle' si no hay sesión y la última no terminó en done/error."""
    if state.running:
        st = state.status if state.status in _STATE else "waiting"
        return "starting" if st == "waiting" and not _play_ready() else st
    return state.status if state.status in ("done", "error") else "idle"


def _chip(text: str, cls: str = "", dot: bool = False) -> str:
    return f'<span class="v2-chip {cls}">{"<span class=v2-dot></span>" if dot else ""}{html.escape(text)}</span>'


def header_html(platform: str) -> str:
    st = current_state()
    label, cls = _STATE[st]
    chips = [_chip(platform, "")]
    chips.append(_chip(label, cls, dot=True))
    if state.running or st in ("done", "error"):
        chips.append(_chip(f"Episodes {state.ep_current} / {state.ep_total}"))
    if st == "recording":
        m = state.live_metrics
        chips.append(_chip(f"{m['fps']:.0f} FPS · {m['latency_ms']:.0f} ms"))
    if state.running:
        chips.append(_chip("VR" if state.live_view_vr else "No VR"))
    return (
        '<div class="v2-header">'
        f'<div class="v2-brand">{_LOGOS}<div><h1>iCub Experimentation Hub</h1>'
        '<small>Setup → Record → Review &amp; Curate → Export</small></div></div>'
        f'<div class="v2-chips">{"".join(chips)}</div></div>'
    )


def state_panel_html() -> str:
    """Panel grande de Record: qué está pasando y qué hacer ahora."""
    st = current_state()
    vr = state.live_view_vr
    nxt = state.ep_current + 1
    title, cls = {
        "idle": ("No session", "v2-off"),
        "starting": ("Starting the session…", "v2-warn"),
        "waiting": (f"Ready for episode {nxt}", "v2-warn"),
        "recording": (f"Recording episode {nxt}", "v2-rec"),
        "done": ("Session finished", "v2-ok"),
        "error": ("Session error", "v2-err"),
    }[st]
    hint = {
        "idle": "Configure and start a session in <b>Setup</b>.",
        "starting": "Connecting to the robot (about 10–20 s).",
        "waiting": ("Press <b>Record</b> (<span class=v2-kbd>R</span>)"
                    + (" or <b>A</b> on the controller" if vr else "") + " to start."),
        "recording": ("Press <b>Stop &amp; save</b> (<span class=v2-kbd>S</span>)"
                      + (" or <b>B</b>. Hold <b>B</b> 1.5 s to discard the episode." if vr else ".")),
        "done": "Review the recorded episodes in <b>Review &amp; Curate</b>.",
        "error": "Open the <b>Log</b> below to see what failed.",
    }[st]
    return (f'<div class="v2-state {cls}"><div class="v2-state-title"><span class="v2-dot"></span>'
            f'{html.escape(title)}</div><div class="v2-state-hint">{hint}</div></div>')


def progress_html() -> str:
    total = max(state.ep_total, 1)
    done = min(state.ep_current, total)
    pct = int(100 * done / total)
    return (f'<div class="v2-progress"><div class="v2-bar"><div class="v2-fill" style="width:{pct}%"></div></div>'
            f'<div class="v2-progress-text"><span>{done} of {total} episodes saved</span><span>{pct}%</span></div></div>')


def light_html(level: str, text: str) -> str:
    """Indicador de un paso del stack: level = ok | warn | err | off."""
    return f'<div class="v2-step-status v2-{level}"><span class="v2-dot"></span>{html.escape(text)}</div>'
