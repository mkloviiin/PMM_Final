"""Pestana 'Control de episodios': generica, no depende del robot activo. El total de
la barra de progreso es state.ep_total, que sigue al "Number of episodes" de la plataforma
activa (session.set_planned_episodes).

Diseño (traído del Hub v2, src/hub_v2/record_tab.py): todo lo que se usa al grabar en una
sola vista sin scroll. Video y bitácora a la izquierda; a la derecha un panel grande que dice
qué hacer ahora, el progreso, RECORD / STOP (atajos R / S), comandos del robot y la chuleta
del control VR. Métricas y log quedan plegados abajo.
"""

from __future__ import annotations

import html

import gradio as gr

from .. import live_view, session, session_log, state

EPISODES_TAB_ID = "episodes"

_VR_CHEATSHEET = """
| Controller | Action |
|---|---|
| **A** | Start recording |
| **B** | Stop & save the episode |
| **B** held 1.5 s | Discard the episode (the headset shows *DESCARTADO*) |
| **Y** | Reset the scene |
| **Index trigger** | Open / close the hand |
| **Lower trigger** | Clutch: re-position the controller without moving the robot |
| **X** | Pause / help overlay |
"""

# Comandos de consola de play.py (los dos backends los reenvían por stdin)
_ROBOT_CMDS = (("home", "h", "go home"), ("reset", "n", "reset scene"),
               ("teleop", "t", "pause / resume teleop"),
               ("r_open", "8", "right hand open"), ("r_close", "9", "right hand close"),
               ("l_open", "6", "left hand open"), ("l_close", "7", "left hand close"))

EPISODES_CSS = """
:root {
    --rec-ok: #2f9e44;
    --rec-warn: #f59f00;
    --rec-err: #e03131;
    --rec-off: #868e96;
}
.rec-hidden { display: none !important; }
.rec-dot { width: 16px; height: 16px; border-radius: 50%; background: var(--rec-off); flex: none; }
.rec-ok .rec-dot { background: var(--rec-ok); }
.rec-warn .rec-dot { background: var(--rec-warn); }
.rec-err .rec-dot { background: var(--rec-err); }
.rec-rec .rec-dot { background: var(--rec-err); animation: rec-pulse 1s infinite; }
@keyframes rec-pulse { 0%, 100% { opacity: 1; } 50% { opacity: .25; } }
/* Panel "qué hacer ahora" */
.rec-state {
    border-radius: 12px; padding: 16px 18px; border: 2px solid var(--border-color-primary);
    background: var(--background-fill-secondary);
}
.rec-state .rec-state-title { font-size: 1.6rem; font-weight: 800; display: flex; align-items: center; gap: 10px; }
.rec-state .rec-state-hint { margin-top: 6px; color: var(--body-text-color-subdued); }
.rec-state.rec-rec { border-color: var(--rec-err); }
.rec-state.rec-rec .rec-state-title { color: var(--rec-err); }
.rec-state.rec-ok { border-color: var(--rec-ok); }
.rec-state.rec-warn { border-color: var(--rec-warn); }
.rec-state.rec-err { border-color: var(--rec-err); }
/* Progreso */
.rec-progress .rec-bar { height: 12px; border-radius: 6px; background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary); overflow: hidden; }
.rec-progress .rec-fill { height: 100%; background: var(--rec-ok); transition: width .4s; }
.rec-progress .rec-progress-text { display: flex; justify-content: space-between; font-size: .9rem;
    margin-top: 4px; color: var(--body-text-color-subdued); }
/* Tarjeta de la bitácora y textos secundarios */
.rec-card {
    border: 1px solid var(--border-color-primary) !important; border-radius: 12px !important;
    padding: 14px 16px !important; background: var(--block-background-fill) !important;
}
.rec-card-title p, .rec-card-title { margin: 0 0 4px 0 !important; font-weight: 700; font-size: 1.02rem; }
.rec-muted, .rec-muted p { color: var(--body-text-color-subdued) !important; font-size: .9rem; }
.rec-big button, button.rec-big { min-height: 64px !important; font-size: 1.15rem !important; font-weight: 700 !important; }
.rec-kbd { font-family: var(--font-mono); border: 1px solid var(--border-color-primary); border-radius: 4px;
    padding: 0 5px; font-size: .8rem; }
.rec-cheatsheet table { font-size: .9rem; }
"""

# Atajos de teclado: R = grabar, S = detener y guardar. Solo con Episode Control a la vista y
# sin un campo de texto enfocado (si no, escribir una nota dispararía la grabación).
EPISODES_JS = """
() => {
    if (window.__hubRecKeys) return;
    window.__hubRecKeys = true;
    document.addEventListener('keydown', e => {
        if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
        if (e.target.closest && e.target.closest('input, textarea, select, [contenteditable="true"]')) return;
        const panel = document.getElementById('rec-panel');
        if (!panel || panel.offsetParent === null) return;
        const id = {r: 'rec-start-btn', s: 'rec-stop-btn'}[e.key.toLowerCase()];
        const btn = id && document.getElementById(id);
        if (!btn) return;
        (btn.tagName === 'BUTTON' ? btn : btn.querySelector('button'))?.click();
        e.preventDefault();
    });
}
"""


def build_episodes_tab():
    # Pestañas en el orden de uso (pedido del cuestionario SUS)
    with gr.Tab("2. Episode Control", id=EPISODES_TAB_ID) as episodes_tab:
        with gr.Row(equal_height=False, elem_id="rec-panel"):
            # ── Video y bitácora ─────────────────────────────────────────────
            with gr.Column(scale=3):
                # With VR: what the headset shows + its front screen; without VR: robot eye + scene camera
                gr.HTML(live_view.html())
                gr.Markdown("With VR: what the headset shows and its front screen. "
                            "Without VR: robot eye and scene camera.", elem_classes=["rec-muted"])
                with gr.Column(elem_classes=["rec-card"]):
                    gr.Markdown("Logbook", elem_classes=["rec-card-title"])
                    with gr.Row(equal_height=True):
                        note_tb = gr.Textbox(show_label=False, lines=2, scale=5,
                                             placeholder="Observation about the current episode "
                                                         "(saved with its number in the dataset folder)")
                        add_note_btn = gr.Button("Add note", scale=1)
                    notes_md = gr.Markdown("*No notes yet.*")
            # ── Control ──────────────────────────────────────────────────────
            with gr.Column(scale=2):
                state_html = gr.HTML(state_panel_html())
                progress_html_c = gr.HTML(progress_html())
                with gr.Row():
                    start_btn = gr.Button("● Record", variant="primary", elem_id="rec-start-btn",
                                          elem_classes=["btn-green", "rec-big"])
                    stop_btn = gr.Button("■ Stop & save", elem_id="rec-stop-btn",
                                         elem_classes=["btn-red", "rec-big"])
                gr.Markdown("Shortcuts: <span class=rec-kbd>R</span> record · "
                            "<span class=rec-kbd>S</span> stop & save", elem_classes=["rec-muted"])
                cmd_feedback = gr.Markdown("", elem_classes=["rec-muted"])
                # Solo con algo grabado y sin sesión en curso (lo muestra/oculta el poller)
                review_btn = gr.Button("Review the recorded episodes →", variant="primary",
                                       elem_classes=["rec-hidden"])

                cmd_btns = {}
                with gr.Accordion("Robot commands", open=False):
                    with gr.Row():
                        cmd_btns["home"] = gr.Button("⌂ Go home", size="sm")
                        cmd_btns["reset"] = gr.Button("↺ Reset scene", size="sm")
                        cmd_btns["teleop"] = gr.Button("⏯ Pause / resume teleop", size="sm")
                    with gr.Row():
                        cmd_btns["r_open"] = gr.Button("Right hand: open", size="sm")
                        cmd_btns["r_close"] = gr.Button("Right hand: close", size="sm")
                    with gr.Row():
                        cmd_btns["l_open"] = gr.Button("Left hand: open", size="sm")
                        cmd_btns["l_close"] = gr.Button("Left hand: close", size="sm")
                with gr.Accordion("VR controller buttons", open=False, elem_classes=["rec-cheatsheet"]):
                    gr.Markdown(_VR_CHEATSHEET)

                # "End session" pide confirmación: descarta un episodio en curso
                exit_btn = gr.Button("End session", variant="stop", size="sm")
                with gr.Row(elem_classes=["rec-hidden"]) as exit_confirm:
                    gr.Markdown("**End the session?** An episode being recorded is discarded.")
                    exit_yes = gr.Button("Yes, end it", variant="stop", size="sm")
                    exit_no = gr.Button("Cancel", size="sm")

        with gr.Accordion("Session metrics (saved in <dataset>/hub_session/)", open=False):
            with gr.Row():
                fps_nb = gr.Number(label="FPS", elem_classes=["metric-nb"], value=0.0, interactive=False, precision=1)
                lat_nb = gr.Number(label="Latency (ms)", elem_classes=["metric-nb"], value=0.0, interactive=False, precision=1)
                size_nb = gr.Number(label="Dataset size (MB)", elem_classes=["metric-nb"], value=0.0, interactive=False, precision=2)
            episode_metrics_plot = gr.Plot(label="Per-episode metrics", show_label=False)
        with gr.Accordion("Live log", open=False) as log_acc:
            log_box = gr.Textbox(show_label=False, lines=18, max_lines=18, interactive=False, autoscroll=True)

    # ── Wiring propio de la pestaña ──────────────────────────────────────────
    start_btn.click(fn=session.cmd_start, outputs=[cmd_feedback])
    stop_btn.click(fn=session.cmd_stop, outputs=[cmd_feedback])
    for key, cmd, label in _ROBOT_CMDS:
        cmd_btns[key].click(fn=lambda c=cmd, lb=label: _send(c, lb), outputs=[cmd_feedback])
    add_note_btn.click(fn=session.add_note, inputs=[note_tb], outputs=[note_tb, notes_md])

    show, hide = gr.update(elem_classes=[]), gr.update(elem_classes=["rec-hidden"])
    exit_btn.click(fn=lambda: (hide, show), outputs=[exit_btn, exit_confirm])
    exit_no.click(fn=lambda: (show, hide), outputs=[exit_btn, exit_confirm])
    exit_yes.click(fn=lambda: (session.cmd_exit(), show, hide), outputs=[cmd_feedback, exit_btn, exit_confirm])
    # Gráfico y notas completos al abrir la pestaña (el poller solo manda cambios)
    episodes_tab.select(fn=session.episode_panels, outputs=[episode_metrics_plot, notes_md])

    return dict(
        tab=episodes_tab, state_html=state_html, progress_html=progress_html_c,
        cmd_feedback=cmd_feedback, review_btn=review_btn, log_box=log_box, log_acc=log_acc,
        fps_nb=fps_nb, lat_nb=lat_nb, size_nb=size_nb,
        episode_metrics_plot=episode_metrics_plot, note_tb=note_tb, notes_md=notes_md,
    )


def _send(cmd: str, label: str) -> str:
    if not state.running:
        return "No active session."
    state.cmd_queue.put(cmd)
    return f"Sent: {label}"


# ── Estado a mostrar ─────────────────────────────────────────────────────────

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
    """idle | starting | waiting | recording | done | error."""
    if state.running:
        st = state.status if state.status in ("waiting", "recording") else "waiting"
        return "starting" if st == "waiting" and not _play_ready() else st
    return state.status if state.status in ("done", "error") else "idle"


def state_panel_html() -> str:
    """Panel grande: qué está pasando y qué hacer ahora."""
    st = current_state()
    vr = state.live_view_vr
    nxt = state.ep_current + 1
    title, cls = {
        "idle": ("No session", "rec-off"),
        "starting": ("Starting the session…", "rec-warn"),
        "waiting": (f"Ready for episode {nxt}", "rec-warn"),
        "recording": (f"Recording episode {nxt}", "rec-rec"),
        "done": ("Session finished", "rec-ok"),
        "error": ("Session error", "rec-err"),
    }[st]
    hint = {
        "idle": "Configure and launch a session in <b>Configuration</b>.",
        "starting": "Connecting to the robot (about 10–20 s).",
        "waiting": ("Press <b>Record</b> (<span class=rec-kbd>R</span>)"
                    + (" or <b>A</b> on the controller" if vr else "") + " to start."),
        "recording": ("Press <b>Stop &amp; save</b> (<span class=rec-kbd>S</span>)"
                      + (" or <b>B</b>. Hold <b>B</b> 1.5 s to discard the episode." if vr else ".")),
        "done": "Review the recorded episodes in <b>Visualize &amp; Curate</b>.",
        "error": "Open the <b>Live log</b> below to see what failed.",
    }[st]
    return (f'<div class="rec-state {cls}"><div class="rec-state-title"><span class="rec-dot"></span>'
            f'{html.escape(title)}</div><div class="rec-state-hint">{hint}</div></div>')


def progress_html() -> str:
    total = max(state.ep_total, 1)
    done = min(state.ep_current, total)
    pct = int(100 * done / total)
    return (f'<div class="rec-progress"><div class="rec-bar"><div class="rec-fill" style="width:{pct}%"></div></div>'
            f'<div class="rec-progress-text"><span>{done} of {total} episodes saved</span><span>{pct}%</span></div></div>')


# ── Poller ───────────────────────────────────────────────────────────────────

_ui_log_version = -1  # versión de session_log ya enviada a la UI (gráfico y notas)
_last_state = None


def poll():
    """Valores para el Timer de 0.5 s, en el orden de poll_outputs()."""
    global _ui_log_version, _last_state
    if session_log.version != _ui_log_version:
        _ui_log_version = session_log.version
        plot_upd, notes_upd = session._episode_metrics_figure(), session._notes_markdown()
    else:
        plot_upd, notes_upd = gr.update(), gr.update()
    with state.log_lock:
        log_text = "\n".join(state.log_lines[-200:])
    st = current_state()
    # Al pasar a error, el log se abre solo
    log_open = gr.update(open=True) if st == "error" and _last_state != "error" else gr.update()
    _last_state = st
    m = state.live_metrics
    review = gr.update(elem_classes=[] if (not state.running and state.last_dataset_root) else ["rec-hidden"])
    return (
        state_panel_html(), progress_html(), m["fps"], m["latency_ms"], m["size_mb"],
        plot_upd, notes_upd, log_text, log_open, review,
        gr.update(interactive=not state.running),  # "Back to robot selection"
    )


def poll_outputs(episodes: dict, back_btn) -> list:
    return [episodes[k] for k in ("state_html", "progress_html", "fps_nb", "lat_nb", "size_nb",
                                  "episode_metrics_plot", "notes_md", "log_box", "log_acc",
                                  "review_btn")] + [back_btn]
