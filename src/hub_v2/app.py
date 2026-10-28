"""Ensamblado del Hub v2: barra de estado fija, cuatro pestañas en el orden del trabajo
(Setup → Record → Review & Curate → Export) y los temporizadores que las mantienen al día.
"""

from __future__ import annotations

import gradio as gr

from launcher_gui import curation, state
from launcher_gui.ui.input_filters import INPUT_FILTER_JS
from launcher_gui.video_sync import SYNC_JS

from . import curate_tab, export_tab, record_tab, setup_tab, status
from .curate_tab import CURATE_TAB_ID, build_curate_tab
from .export_tab import EXPORT_TAB_ID, build_export_components, changes_summary
from .record_tab import RECORD_TAB_ID, build_record_tab
from .setup_tab import build_setup_tab
from .stack import P_MUJOCO


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="iCub Experimentation Hub v2") as demo:
        demo.load(fn=None, js=SYNC_JS)          # videos sincronizados de Review (como el Hub original)
        demo.load(fn=None, js=INPUT_FILTER_JS)  # campos numéricos / IP: solo caracteres válidos
        header = gr.HTML(status.header_html(P_MUJOCO))

        with gr.Tabs() as tabs:
            setup = build_setup_tab()
            record = build_record_tab()
            c = build_curate_tab()
            with gr.Tab("4 · Export", id=EXPORT_TAB_ID, render_children=True) as export:
                e = build_export_components()

        curate_tab.wire(c, e)
        export_tab.wire(e, c)
        export.select(changes_summary,
                      [c["whole_feature_cbg"], c["dims_action_cbg"], c["dims_state_cbg"],
                       c["task_dirty_state"], c["task_tb"]], [e["summary_md"]])

        # Lanzar: si la sesión partió se pasa a Record; si no, el error queda a la vista en Setup
        def _launch(*values):
            seq = state.session_seq
            msg = setup_tab.launch(*values)
            return msg, gr.update(selected=RECORD_TAB_ID) if state.session_seq != seq else gr.update()

        setup["launch_btn"].click(_launch, setup["form"], [setup["launch_msg"], tabs])

        # Record -> Review: abre el último dataset grabado en este Hub
        record["review_btn"].click(
            lambda root: (gr.update(selected=CURATE_TAB_ID), curation.use_last_dataset(root)),
            [c["viz_root_tb"]], [tabs, c["dataset_dd"]],
        ).then(curation.load_viz_dataset, [c["dataset_dd"]], c["load_outputs"]).then(
            lambda: gr.update(visible=not state.viz_ep_rows), None, [c["empty_md"]])

        gr.Timer(0.5).tick(lambda p: (status.header_html(p), *record_tab.poll(record)),
                           [setup["platform"]], [header] + record_tab.outputs_of(record))
        gr.Timer(2.0).tick(setup_tab.poll_steps, [setup["platform"]],
                           [setup["step1_light"], setup["step2_light"], setup["step1"], setup["step2"],
                            setup["launch_btn"], setup["ready_hint"]])
    return demo
