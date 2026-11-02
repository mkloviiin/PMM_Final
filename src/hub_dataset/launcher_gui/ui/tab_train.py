"""Pestana 'Train': entrena SmolVLA sobre un dataset grabado (lógica en training.py).

Arriba lo que se cambia siempre (dataset, nombres, steps, batch, W&B y Hugging Face); el resto
de hiperparámetros va plegado en "Advanced settings". Mientras entrena, app.py bloquea las
otras pestañas y "Back to robot selection".
"""

from __future__ import annotations

import gradio as gr

from .. import curation, training
from ..curation import MODAL_HIDDEN, MODAL_SHOWN
from ..paths import REPO_ROOT
from .dir_picker import dir_textbox

TRAIN_TAB_ID = "train"
D = training.DEFAULTS

# El panel de estado sin el margen de gr.HTML (padding=False no lo quita): mismo ancho que el
# botón y el log. "Refresh" es un botón cuadrado con ícono, en la fila del título "1. Dataset".
TRAIN_CSS = """
.train-status .html-container { padding: 0 !important; }
.train-ds-head { align-items: center !important; }
button.train-refresh, .train-refresh button {
    width: 34px !important; height: 34px !important; min-width: 34px !important; max-width: 34px !important;
    padding: 0 !important; border-radius: 8px !important;
    background: var(--input-background-fill) url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23f97316' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 12a9 9 0 1 1-2.64-6.36'/%3E%3Cpath d='M21 3v6h-6'/%3E%3C/svg%3E") center / 20px 20px no-repeat !important;
}
button.train-refresh:hover, .train-refresh button:hover { background-color: rgba(127, 127, 127, 0.28) !important; }
"""


def build_train_tab():
    # build_ui() corre una vez al abrir el Hub: la pestaña parte en "Waiting to start" salvo
    # que haya una corrida en curso
    training.clear_finished()
    with gr.Tab("4. Train", id=TRAIN_TAB_ID) as train_tab:
        gr.Markdown("Train a **SmolVLA** policy on a recorded dataset. Training uses the whole GPU: "
                    "while it runs, the rest of the Hub is locked.")

        # Dataset a la izquierda; estado y Start / Stop a la derecha
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                with gr.Row(elem_classes=["train-ds-head"]):
                    gr.Markdown("### 1. Dataset")
                    # Ícono de refrescar: lo dibuja TRAIN_CSS
                    refresh_btn = gr.Button("", scale=0, min_width=34, elem_classes=["train-refresh"])
                root_tb, browse_ev = dir_textbox("Datasets directory", str(REPO_ROOT / "data"),
                                                 title="Select the datasets directory",
                                                 info="Folder with the datasets: data/ (recorded) "
                                                      "or data_curated/ (curated)")
                dataset_dd = gr.Dropdown(choices=[], value=None, label="Dataset", allow_custom_value=True,
                                         info="Demonstrations the policy learns from")
                dataset_md = gr.Markdown("Select a dataset.")
            with gr.Column(scale=2):
                status_html = gr.HTML(training.status_html(training.current_run()),
                                      elem_classes=["train-status"])
                start_btn = gr.Button("▶ Start training", variant="primary",
                                      elem_classes=["btn-green", "rec-big"])
                stop_btn = gr.Button("■ Stop training", elem_classes=["btn-red", "rec-big"], visible=False)

        gr.Markdown("### 2. Run")
        runs_tb, _ = dir_textbox("Runs directory", str(training.DEFAULT_RUNS_DIR),
                                 title="Select where to save training runs",
                                 info="Folder where the trained models are saved")
        with gr.Row():
            run_tb = gr.Textbox(label="Run name", info="Folder of this run inside the runs directory")
            job_tb = gr.Textbox(label="Job name", info="Name of the run in the training log and in W&B")
        with gr.Row():
            steps_nb = gr.Number(value=D["steps"], label="Training steps", precision=0, minimum=1,
                                 info="Number of model updates; more steps take longer")
            batch_nb = gr.Number(value=D["batch_size"], label="Batch size", precision=0, minimum=1,
                                 info="Samples used in each step; larger needs more GPU memory")
            save_nb = gr.Number(value=D["save_freq"], label="Save checkpoint every (steps)",
                                precision=0, minimum=1, info="How often a copy of the model is saved (~1.3 GB each)")

        gr.Markdown("### 3. Logging and upload")
        with gr.Group():
            wandb_cb = gr.Checkbox(value=True, label="Log to Weights & Biases",
                                   info="Sends the training curves (loss, learning rate) to wandb.ai")
            with gr.Row() as wandb_row:
                wandb_key_tb = gr.Textbox(label="W&B API key (session only, not saved to disk)",
                                          type="password", placeholder="Paste your key from wandb.ai/authorize",
                                          info="Key of your W&B account; empty = the account logged in on this PC")
                wandb_proj_tb = gr.Textbox(label="W&B project", info="W&B project where this run is grouped")
        with gr.Group():
            hf_cb = gr.Checkbox(value=False, label="Push the trained model to Hugging Face",
                                info="Uploads the final model when the training ends")
            with gr.Row(visible=False) as hf_row:
                hf_token_tb = gr.Textbox(label="Hugging Face token (session only, not saved to disk)",
                                         type="password", placeholder="hf_...",
                                         info="Access token with Write permission; checked before the training starts")
                hf_repo_tb = gr.Textbox(label="Model repo ID", placeholder="user_or_org/model_name",
                                        info="Where the model is uploaded: user or organization / model name")
                hf_private_cb = gr.Checkbox(value=True, label="Private", scale=0, min_width=160,
                                            info="Only you (or your organization) can see it")

        with gr.Accordion("Advanced settings", open=False):
            with gr.Row():
                vlm_tb = gr.Textbox(value=D["vlm_model"], label="Base vision-language model",
                                    info="Pretrained model SmolVLA is built on (Hugging Face ID)")
                load_vlm_cb = gr.Checkbox(value=D["load_vlm_weights"], label="Load its pretrained weights",
                                          info="Start from the pretrained model instead of random weights")
                amp_cb = gr.Checkbox(value=D["use_amp"], label="Mixed precision (AMP)",
                                     info="Half-precision math: faster and uses less GPU memory")
            with gr.Row():
                chunk_nb = gr.Number(value=D["chunk_size"], label="Chunk size", precision=0, minimum=1,
                                     info="Future actions predicted at once")
                nact_nb = gr.Number(value=D["n_action_steps"], label="Action steps per inference",
                                    precision=0, minimum=1, info="Actions executed before predicting again")
                max_state_nb = gr.Number(value=32, label="Max state dim", precision=0, minimum=1,
                                         info="Size the robot state is padded to (set from the dataset)")
                max_action_nb = gr.Number(value=32, label="Max action dim", precision=0, minimum=1,
                                          info="Size the action is padded to (set from the dataset)")
            with gr.Row():
                lr_nb = gr.Number(value=D["lr"], label="Learning rate", info="Size of each model update")
                warmup_nb = gr.Number(value=D["warmup_steps"], label="Warmup steps", precision=0, minimum=0,
                                      info="Steps where the learning rate ramps up from 0")
                decay_nb = gr.Number(value=D["decay_steps"], label="LR decay steps", precision=0, minimum=1,
                                     info="Steps over which the learning rate decreases")
            with gr.Row():
                freeze_cb = gr.Checkbox(value=D["freeze_vision"], label="Freeze the vision encoder",
                                        info="Keep the image encoder as pretrained")
                expert_cb = gr.Checkbox(value=D["expert_only"], label="Train the action expert only",
                                        info="Train only the part that outputs actions, not the language model")
            with gr.Row():
                workers_nb = gr.Number(value=D["num_workers"], label="Dataloader workers", precision=0, minimum=0,
                                       info="Processes that load the data in parallel")
                logf_nb = gr.Number(value=D["log_freq"], label="Log every (steps)", precision=0, minimum=1,
                                    info="How often the step and loss are written to the log")
                seed_nb = gr.Number(value=D["seed"], label="Seed", precision=0, minimum=0,
                                    info="Random seed: same seed, same data order")

        with gr.Accordion("Training log", open=False):
            log_box = gr.Textbox(show_label=False, lines=18, max_lines=18, interactive=False, autoscroll=True)

        # Confirmación de "Stop training" (pop-up centrado, .pmm-modal de app.py)
        with gr.Column(elem_classes=MODAL_HIDDEN) as stop_confirm:
            stop_confirm_md = gr.Markdown("")
            with gr.Row():
                stop_cancel_btn = gr.Button("Cancel")
                stop_yes_btn = gr.Button("Stop and delete", variant="primary", elem_classes=["btn-red"])

    # ── Wiring propio de la pestaña ──────────────────────────────────────────
    def _refresh(root, current=None):
        return curation.refresh_datasets(root, current)[0]

    train_tab.select(fn=_refresh, inputs=[root_tb, dataset_dd], outputs=[dataset_dd])
    refresh_btn.click(fn=_refresh, inputs=[root_tb, dataset_dd], outputs=[dataset_dd])
    browse_ev.then(fn=_refresh, inputs=[root_tb], outputs=[dataset_dd])
    dataset_dd.change(fn=training.on_dataset_change, inputs=[dataset_dd, runs_tb],
                      outputs=[dataset_md, run_tb, job_tb, wandb_proj_tb, hf_repo_tb, max_state_nb, max_action_nb])
    wandb_cb.change(fn=lambda on: gr.update(visible=on), inputs=[wandb_cb], outputs=[wandb_row])
    hf_cb.change(fn=lambda on: gr.update(visible=on), inputs=[hf_cb], outputs=[hf_row])

    # Sin texto bajo los botones: el panel de estado dice cómo va y los avisos salen como toast
    def _start(*args):
        msg = training.start_training(*args)
        (gr.Info if training.is_active() else gr.Warning)(msg)

    start_inputs = [dataset_dd, runs_tb, run_tb, job_tb, steps_nb, batch_nb, save_nb,
                    wandb_cb, wandb_key_tb, wandb_proj_tb, hf_cb, hf_token_tb, hf_repo_tb, hf_private_cb,
                    vlm_tb, load_vlm_cb, amp_cb, chunk_nb, nact_nb, max_state_nb, max_action_nb,
                    lr_nb, warmup_nb, decay_nb, freeze_cb, expert_cb, workers_nb, logf_nb, seed_nb]
    start_btn.click(fn=_start, inputs=start_inputs)

    def _ask_stop():
        run = training.current_run()
        if not run or run.get("status") != "running":
            return gr.update(elem_classes=MODAL_HIDDEN), ""
        pr = training.progress(run)
        return gr.update(elem_classes=MODAL_SHOWN), (
            f"### Stop the training?\n"
            f"**{run['run_name']}** is at step {pr['step']:,} of {pr['total']:,}. "
            f"Stopping it ends the run and **moves its folder to the trash**:\n\n"
            f"`{run['output_dir']}`\n\n"
            f"Checkpoints saved so far go with it, and the run can't be resumed from the Hub. "
            f"The training log is kept.")

    stop_btn.click(fn=_ask_stop, outputs=[stop_confirm, stop_confirm_md])
    stop_cancel_btn.click(fn=lambda: gr.update(elem_classes=MODAL_HIDDEN), outputs=[stop_confirm])
    stop_yes_btn.click(fn=lambda: gr.update(elem_classes=MODAL_HIDDEN), outputs=[stop_confirm]).then(
        fn=lambda: gr.Info(training.stop_training()))

    def _poll():
        run = training.current_run()
        active = bool(run and run.get("status") in ("running", "stopping"))
        return (training.status_html(run), training.log_text(run),
                gr.update(interactive=not active),
                gr.update(visible=bool(run and run.get("status") == "running")))

    gr.Timer(1.0).tick(fn=_poll, outputs=[status_html, log_box, start_btn, stop_btn])

    return dict(tab=train_tab)
