"""Pestana 'Control de episodios': generica, no depende del robot activo mas
alla del componente de "numero de episodios" que le pasa cada robot para
acotar la barra de progreso.
"""

from __future__ import annotations

import gradio as gr


def build_episodes_tab(num_eps_component: gr.components.Component):
    with gr.Tab("Control de episodios"):
        with gr.Row():
            status_lbl = gr.Textbox(
                value="En espera", label="Estado",
                interactive=False, scale=2,
            )
            progress_lbl = gr.Textbox(
                value="Episodio: 0 / 0  (0%)", label="Progreso",
                interactive=False, scale=3,
            )
        progress_bar = gr.Slider(
            0, num_eps_component.value, value=0, label="Completado", interactive=False,
            buttons=[], elem_classes=["progress-completado"],
        )

        num_eps_component.change(
            fn=lambda n: gr.update(maximum=max(int(n or 1), 1)),
            inputs=[num_eps_component],
            outputs=[progress_bar],
        )

        gr.Markdown("---")
        gr.Markdown("### Controles de episodio")
        with gr.Row():
            start_btn = gr.Button("GRABAR episodio", variant="primary", scale=2)
            stop_btn = gr.Button("PARAR episodio", variant="secondary", scale=2)
            exit_btn = gr.Button("TERMINAR sesion", variant="stop", scale=1)
        cmd_feedback = gr.Textbox(label="Respuesta", interactive=False)

        gr.Markdown("---")
        log_box = gr.Textbox(
            label="Log en vivo  (ultimas 80 lineas)",
            lines=22,
            max_lines=22,
            interactive=False,
        )

    return dict(
        status_lbl=status_lbl, progress_lbl=progress_lbl, progress_bar=progress_bar,
        start_btn=start_btn, stop_btn=stop_btn, exit_btn=exit_btn,
        cmd_feedback=cmd_feedback, log_box=log_box,
    )
