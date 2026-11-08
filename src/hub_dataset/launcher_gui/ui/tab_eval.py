"""Pestaña 'Evaluate': evalúa una política entrenada en MuJoCo o Gazebo (lógica en evaluation.py).

Mismo diseño que 'Train': arriba el modelo a la izquierda y estado + Start / Stop a la derecha;
después la tarea y el simulador; lo que casi no se cambia va plegado en "Advanced settings".
Mientras evalúa, app.py bloquea las otras pestañas y "Back to robot selection".
"""

from __future__ import annotations

import gradio as gr

from .. import evaluation
from ..curation import MODAL_HIDDEN, MODAL_SHOWN
from ..robots.icub_sim import SCENES
from ..robots.solver import SOLVERS
from .dir_picker import dir_textbox

EVAL_TAB_ID = "eval"
D = evaluation.DEFAULTS


def build_eval_tab():
    # Como Train: parte en "Waiting to start" salvo que haya una evaluación en curso
    evaluation.clear_finished()
    scenes = list(SCENES)
    with gr.Tab("5. Evaluate", id=EVAL_TAB_ID) as eval_tab:
        gr.Markdown("Evaluate a trained policy in simulation: it runs the task for a number of episodes "
                    "and measures the success rate. While it runs, the rest of the Hub is locked.")

        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                # Mismo título con botón de refrescar que Train (.train-ds-head / .train-refresh)
                with gr.Row(elem_classes=["train-ds-head"]):
                    gr.Markdown("### 1. Model")
                    refresh_btn = gr.Button("", scale=0, min_width=34, elem_classes=["train-refresh"])
                models_tb, browse_ev = dir_textbox("Models directory", str(evaluation.DEFAULT_MODELS_DIR),
                                                   title="Select the trained models directory",
                                                   info="Folder with the training runs (Train tab)")
                with gr.Row():
                    model_dd = gr.Dropdown(choices=[], value=None, label="Model", allow_custom_value=True,
                                           scale=3,
                                           info="Training run to evaluate, or a Hugging Face model ID")
                    ckpt_dd = gr.Dropdown(choices=[], value=None, label="Checkpoint", scale=1,
                                          info="'last' = end of the training", visible=False)
                model_md = gr.Markdown("Select a model.")
            with gr.Column(scale=2):
                status_html = gr.HTML(evaluation.status_html(evaluation.current_run()),
                                      elem_classes=["train-status"])
                start_btn = gr.Button("▶ Start evaluation", variant="primary",
                                      elem_classes=["btn-green", "rec-big"])
                stop_btn = gr.Button("■ Stop evaluation", elem_classes=["btn-red", "rec-big"], visible=False)

        gr.Markdown("### 2. Task")
        with gr.Row():
            backend_rd = gr.Radio(choices=list(evaluation.BACKENDS), value=evaluation.BACKEND_MUJOCO,
                                  label="Simulator",
                                  info="Gazebo: the simulation and the YARP module are started here "
                                       "(with the model's arms) and closed at the end")
            scene_dd = gr.Dropdown(choices=scenes, value=scenes[0] if scenes else None, label="Task",
                                   info="Scene the policy is run in (set from the model when it matches)")
            solver_dd = gr.Dropdown(choices=list(SOLVERS), value=None, label="Controller (solver)",
                                    interactive=False,
                                    info="Set from the model: the controller its dataset was recorded with")
        task_tb = gr.Textbox(label="Task instruction",
                             value=SCENES[scenes[0]][1] if scenes else "",
                             info="Language instruction given to the policy (by default, the one it was trained with)")
        with gr.Row():
            eps_nb = gr.Number(value=D["num_episodes"], label="Episodes", precision=0, minimum=1,
                               info="The scene is reset before each one")
            len_nb = gr.Number(value=D["episode_len"], label="Max steps per episode", precision=0, minimum=1,
                               info=f"At {D['fps']} steps per second: 600 = 20 s")
            side_dd = gr.Dropdown(choices=list(evaluation.SIDES), value=evaluation.SIDE_ALTERNATE,
                                  label="Side of the task",
                                  info="Two-sided tasks: where the object / door handle is in each "
                                       "episode (the Gazebo door is fixed)")
        with gr.Row():
            viewer_cb = gr.Checkbox(value=True, label="Show the simulation window",
                                    info="MuJoCo viewer (Gazebo: its own window + the policy's first camera)")
            videos_cb = gr.Checkbox(value=True, label="Save videos",
                                    info="One video per policy camera with every episode, in the results folder")

        with gr.Accordion("Advanced settings", open=False):
            with gr.Row():
                nact_nb = gr.Number(value=None, label="Action steps per inference", precision=0, minimum=1,
                                    info="Actions executed from each predicted chunk before predicting "
                                         "again (set from the model); fewer = more corrections")
                fps_nb = gr.Number(value=D["fps"], label="Control rate (Hz)", precision=0, minimum=1,
                                   info="Steps per second, as recorded")
                seed_nb = gr.Number(value=None, label="Seed (MuJoCo)", precision=0, minimum=0,
                                    info="Seed of the object positions; empty = different every time")
            results_tb, _ = dir_textbox("Results directory", str(evaluation.DEFAULT_RESULTS_DIR),
                                        title="Select where to save evaluation results",
                                        info="Each evaluation saves summary.json, eval.log and videos/ "
                                             "in its own folder here")

        with gr.Accordion("Results per episode", open=True):
            episodes_md = gr.Markdown(evaluation.episodes_table(evaluation.current_run()))

        with gr.Accordion("Evaluation log", open=False):
            log_box = gr.Textbox(show_label=False, lines=18, max_lines=18, interactive=False, autoscroll=True)

        # Confirmación de "Stop evaluation" (pop-up centrado, .pmm-modal de app.py)
        with gr.Column(elem_classes=MODAL_HIDDEN) as stop_confirm:
            stop_confirm_md = gr.Markdown("")
            with gr.Row():
                stop_cancel_btn = gr.Button("Cancel")
                stop_yes_btn = gr.Button("Stop", variant="primary", elem_classes=["btn-red"])

    # ── Wiring propio de la pestaña ──────────────────────────────────────────
    eval_tab.select(fn=evaluation.refresh_models, inputs=[models_tb, model_dd], outputs=[model_dd])
    refresh_btn.click(fn=evaluation.refresh_models, inputs=[models_tb, model_dd], outputs=[model_dd])
    browse_ev.then(fn=evaluation.refresh_models, inputs=[models_tb], outputs=[model_dd])
    model_outputs = [ckpt_dd, model_md, scene_dd, task_tb, nact_nb, solver_dd]
    model_dd.change(fn=evaluation.on_model_change, inputs=[model_dd, ckpt_dd], outputs=model_outputs)
    # Otro checkpoint del mismo modelo: solo cambia el resumen (y action steps si difiere)
    ckpt_dd.input(fn=evaluation.on_model_change, inputs=[model_dd, ckpt_dd], outputs=model_outputs)
    scene_dd.input(fn=evaluation.on_scene_change, inputs=[scene_dd, model_dd, ckpt_dd], outputs=[task_tb])
    # La semilla solo la usa MuJoCo (el lado de la tarea, los dos simuladores)
    backend_rd.change(fn=lambda b: gr.update(interactive=b == evaluation.BACKEND_MUJOCO),
                      inputs=[backend_rd], outputs=[seed_nb])

    def _start(*args):
        msg = evaluation.start_evaluation(*args)
        (gr.Info if evaluation.is_active() else gr.Warning)(msg)

    start_inputs = [model_dd, ckpt_dd, backend_rd, scene_dd, task_tb, eps_nb, len_nb,
                    viewer_cb, videos_cb, nact_nb, side_dd, seed_nb, fps_nb, results_tb]
    start_btn.click(fn=_start, inputs=start_inputs)

    def _ask_stop():
        run = evaluation.current_run()
        if not run or run.get("status") != "running":
            return gr.update(elem_classes=MODAL_HIDDEN), ""
        pr = evaluation.progress(run)
        done = (evaluation.summary(run) or {}).get("n_episodes_completed", 0)
        return gr.update(elem_classes=MODAL_SHOWN), (
            f"### Stop the evaluation?\n"
            f"**{run['run_name']}** is in episode {pr['episode']} of {pr['total']}. "
            f"The {done} finished episode(s) are kept in the results folder; the current one is dropped:\n\n"
            f"`{run['output_dir']}`")

    stop_btn.click(fn=_ask_stop, outputs=[stop_confirm, stop_confirm_md])
    stop_cancel_btn.click(fn=lambda: gr.update(elem_classes=MODAL_HIDDEN), outputs=[stop_confirm])
    stop_yes_btn.click(fn=lambda: gr.update(elem_classes=MODAL_HIDDEN), outputs=[stop_confirm]).then(
        fn=lambda: gr.Info(evaluation.stop_evaluation()))

    def _poll():
        run = evaluation.current_run()
        active = bool(run and run.get("status") in ("running", "stopping"))
        return (evaluation.status_html(run), evaluation.episodes_table(run), evaluation.log_text(run),
                gr.update(interactive=not active),
                gr.update(visible=bool(run and run.get("status") == "running")))

    # El gr.Timer lo crea app.py fuera de gr.Tabs (ver ui/tab_train.py)
    return dict(tab=eval_tab, poll=_poll,
                poll_outputs=[status_html, episodes_md, log_box, start_btn, stop_btn])
