"""Pestaña Record del Hub v2: lo que se usa mientras se graba, en una sola vista sin scroll.

Diferencias con Episode Control original:
- Video en grande a la izquierda y panel de control a la derecha (antes: botones arriba, video
  al medio, métricas y log abajo, había que bajar para ver todo).
- Un estado grande que dice qué hacer ahora ("Ready for episode 4 — press Record or A").
- Atajos de teclado R / S y la chuleta de botones del control VR a mano.
- Comandos del robot que antes solo existían en la consola de play.py (home, reset de escena,
  manos), para grabar sin VR.
- "End session" pide confirmación (cierra la sesión y descarta un episodio en curso).
- Métricas, bitácora y log plegados: están, pero no compiten con lo principal.
"""

from __future__ import annotations

import gradio as gr

from launcher_gui import live_view, session, state, session_log

from . import status

RECORD_TAB_ID = "record"

_VR_CHEATSHEET = """
| Controller | Action |
|---|---|
| **A** | Start recording |
| **B** | Stop & save the episode |
| **B** held 1.5 s | Discard the episode (status turns *DESCARTADO*) |
| **Y** | Reset the scene |
| **Index trigger** | Open / close the hand |
| **Lower trigger** | Clutch: re-position the controller without moving the robot |
| **X** | Pause / help overlay |
"""


def _send(cmd: str, label: str) -> str:
    if not state.running:
        return "No active session."
    state.cmd_queue.put(cmd)
    return f"Sent: {label}"


def build_record_tab() -> dict:
    with gr.Tab("2 · Record", id=RECORD_TAB_ID, render_children=True) as tab:
        with gr.Row(equal_height=False, elem_id="v2-record-panel"):
            # ── Video ────────────────────────────────────────────────────────
            with gr.Column(scale=3):
                gr.HTML(live_view.html())
                gr.Markdown("With VR: what the headset shows and its front screen. "
                            "Without VR: robot eye and scene camera.", elem_classes=["v2-muted"])
                with gr.Column(elem_classes=["v2-card"]):
                    gr.Markdown("Logbook", elem_classes=["v2-card-title"])
                    with gr.Row(equal_height=True):
                        note_tb = gr.Textbox(show_label=False, lines=2, scale=5,
                                             placeholder="Observation about the current episode "
                                                         "(saved with its number in the dataset folder)")
                        add_note_btn = gr.Button("Add note", scale=1)
                    notes_md = gr.Markdown(session._notes_markdown())
            # ── Control ──────────────────────────────────────────────────────
            with gr.Column(scale=2):
                state_html = gr.HTML(status.state_panel_html())
                progress = gr.HTML(status.progress_html())
                with gr.Row():
                    rec_btn = gr.Button("● Record", variant="primary", elem_id="v2-rec-btn",
                                        elem_classes=["btn-green", "v2-big"])
                    stop_btn = gr.Button("■ Stop & save", elem_id="v2-stop-btn",
                                         elem_classes=["btn-red", "v2-big"])
                gr.Markdown("Shortcuts: <span class=v2-kbd>R</span> record · "
                            "<span class=v2-kbd>S</span> stop & save", elem_classes=["v2-muted"])
                feedback = gr.Markdown("", elem_classes=["v2-muted"])
                # Solo con algo grabado y sin sesión en curso (lo muestra/oculta el poller)
                review_btn = gr.Button("Review the recorded episodes →", variant="primary",
                                       elem_classes=["v2-hidden"])

                with gr.Accordion("Robot commands", open=False):
                    with gr.Row():
                        home_btn = gr.Button("⌂ Go home", size="sm")
                        reset_btn = gr.Button("↺ Reset scene", size="sm")
                        teleop_btn = gr.Button("⏯ Pause / resume teleop", size="sm")
                    with gr.Row():
                        r_open = gr.Button("Right hand: open", size="sm")
                        r_close = gr.Button("Right hand: close", size="sm")
                    with gr.Row():
                        l_open = gr.Button("Left hand: open", size="sm")
                        l_close = gr.Button("Left hand: close", size="sm")
                with gr.Accordion("VR controller buttons", open=False, elem_classes=["v2-cheatsheet"]):
                    gr.Markdown(_VR_CHEATSHEET)


                end_btn = gr.Button("End session", variant="stop", size="sm")
                with gr.Row(elem_classes=["v2-hidden"]) as end_confirm:
                    gr.Markdown("**End the session?** An episode being recorded is discarded.")
                    end_yes = gr.Button("Yes, end it", variant="stop", size="sm")
                    end_no = gr.Button("Cancel", size="sm")

        with gr.Accordion("Session metrics", open=False):
            with gr.Row():
                fps_nb = gr.Number(label="FPS", interactive=False, precision=1, elem_classes=["metric-nb"])
                lat_nb = gr.Number(label="Latency (ms)", interactive=False, precision=1, elem_classes=["metric-nb"])
                size_nb = gr.Number(label="Dataset size (MB)", interactive=False, precision=2,
                                    elem_classes=["metric-nb"])
            metrics_plot = gr.Plot(show_label=False)
        with gr.Accordion("Log", open=False) as log_acc:
            log_box = gr.Textbox(show_label=False, lines=18, max_lines=18, interactive=False, autoscroll=True)

    # ── Wiring interno ───────────────────────────────────────────────────────
    rec_btn.click(session.cmd_start, None, [feedback])
    stop_btn.click(session.cmd_stop, None, [feedback])
    for btn, cmd, label in ((home_btn, "h", "go home"), (reset_btn, "n", "reset scene"),
                            (teleop_btn, "t", "pause / resume teleop"),
                            (r_open, "8", "right hand open"), (r_close, "9", "right hand close"),
                            (l_open, "6", "left hand open"), (l_close, "7", "left hand close")):
        btn.click(lambda c=cmd, lb=label: _send(c, lb), None, [feedback])
    add_note_btn.click(session.add_note, [note_tb], [note_tb, notes_md])

    show, hide = gr.update(elem_classes=[]), gr.update(elem_classes=["v2-hidden"])
    end_btn.click(lambda: (hide, show), None, [end_btn, end_confirm])
    end_no.click(lambda: (show, hide), None, [end_btn, end_confirm])
    end_yes.click(lambda: (session.cmd_exit(), show, hide), None, [feedback, end_btn, end_confirm])
    # Gráfico y notas completos al abrir la pestaña (el poller solo manda cambios)
    tab.select(session.episode_panels, None, [metrics_plot, notes_md])

    return dict(tab=tab, state_html=state_html, progress=progress, fps_nb=fps_nb, lat_nb=lat_nb,
                size_nb=size_nb, metrics_plot=metrics_plot, notes_md=notes_md, log_box=log_box,
                log_acc=log_acc, note_tb=note_tb, feedback=feedback, review_btn=review_btn)


_ui_version = -1
_last_status = None


def poll(record: dict):
    """Valores para el Timer de 0.5 s (mismo orden que outputs_of(record))."""
    global _ui_version, _last_status
    if session_log.version != _ui_version:
        _ui_version = session_log.version
        plot, notes = session._episode_metrics_figure(), session._notes_markdown()
    else:
        plot, notes = gr.update(), gr.update()
    with state.log_lock:
        log_text = "\n".join(state.log_lines[-200:])
    st = status.current_state()
    # Al pasar a error, el log se abre solo
    log_open = gr.update(open=True) if st == "error" and _last_status != "error" else gr.update()
    _last_status = st
    m = state.live_metrics
    review = gr.update(elem_classes=[] if (not state.running and state.last_dataset_root) else ["v2-hidden"])
    return (status.state_panel_html(), status.progress_html(), m["fps"], m["latency_ms"], m["size_mb"],
            plot, notes, log_text, log_open, review)


def outputs_of(record: dict) -> list:
    return [record[k] for k in ("state_html", "progress", "fps_nb", "lat_nb", "size_nb",
                                "metrics_plot", "notes_md", "log_box", "log_acc", "review_btn")]
