"""Backend de robot: iCub en simulacion MuJoCo (VR).

Encapsula todo lo que es propio de este robot: las escenas de MuJoCo (leidas
desde el manifiesto scenes.yaml), los campos VR del formulario de
"Configuración" y como arma la sesion de grabacion. El resto del Hub
(Control de episodios, Visualizar y curar, Subir a Hugging Face) no sabe nada
de esto -- solo le importa que `launch()` termine escribiendo un dataset
LeRobot y que la grabacion hable el protocolo de `session.start_session`
(ver robots/base.py).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import yaml

from .. import session
from ..paths import ASSETS_DIR, HUB_ROOT, PROJECT_ROOT
from .base import RobotBackend

_SCENES_DIR = PROJECT_ROOT / "dependencies" / "assets" / "scenes"
_SCENES_MANIFEST = _SCENES_DIR / "scenes.yaml"


def _load_scenes() -> dict[str, tuple[str, str, list]]:
    """Lee scenes.yaml: {label: (archivo_xml, tarea_default, objects_list)}.

    Las escenas se declaran a mano en el manifiesto (no se auto-detectan todos
    los *.xml de la carpeta) porque esa carpeta tambien tiene XMLs que no son
    escenas elegibles (includes compartidos, versiones viejas). Si aparece un
    *.xml suelto que no esta en el manifiesto, se avisa por consola para que
    no quede una escena "huerfana" sin declarar.
    """
    entries = yaml.safe_load(_SCENES_MANIFEST.read_text(encoding="utf-8")) or []
    scenes = {
        e["label"]: (e["file"], e["task"], e.get("objects", []), e.get("joints", []))
        for e in entries
    }

    declared_files = {e["file"] for e in entries}
    all_xml = {p.name for p in _SCENES_DIR.glob("*.xml")}
    orphans = all_xml - declared_files
    if orphans:
        print(f"[icub_sim] Warning: {len(orphans)} XML(s) in {_SCENES_DIR} are not "
              f"declared in scenes.yaml and will not appear in the dropdown: "
              f"{', '.join(sorted(orphans))}")


    return scenes


SCENES = _load_scenes()


def build_config_form() -> dict[str, gr.components.Component]:
    with gr.Row():
        with gr.Column(scale=1):
            scene_dd = gr.Dropdown(
                choices=list(SCENES), value=list(SCENES)[0],
                label="Scene",
                info="Determines the MuJoCo XML and the task",
            )
            repo_id_tb = gr.Textbox(
                value="local/icub_mujoco_demo", label="Repo ID",
                info="Dataset name (e.g. local/my_dataset)",
            )
            num_eps_nb = gr.Number(
                value=50, label="Number of episodes (Default: 50)",
                precision=0, minimum=1,
            )
            fps_nb = gr.Number(
                value=30, label="Dataset FPS (Default: 30)",
                precision=0, minimum=5, maximum=60,
            )

        with gr.Column(scale=1):
            ep_time_sl = gr.Number(
                value=0, label="Max episode duration (s)", placeholder="No limit",
                precision=0, minimum=0, info="0 = no time limit"
            )
            root_tb = gr.Textbox(
                value=str(HUB_ROOT.parent / "data"),
                label="Dataset root directory",
            )
            vr_cb = gr.Checkbox(value=False, label="Enable VR")
            vr_cable_cb = gr.Checkbox(
                value=False, visible=False,
                label="Connect MetaQuest via USB cable",
                info="Tunnels VR ports over USB; set IP = 127.0.0.1 in BeaVR",
            )
            vr_ip_tb = gr.Textbox(
                value="", placeholder="192.168.x.x", visible=False,
                label="VR headset IP",
            )

    gr.Markdown("---")
    launch_btn = gr.Button("LAUNCH SESSION", variant="primary", size="lg")
    launch_status = gr.Textbox(label="Launch status", interactive=False)

    def _on_vr_toggle(enabled):
        if enabled:
            return gr.update(visible=True), gr.update(visible=True)
        return gr.update(visible=False, value=False), gr.update(visible=False, value="")

    vr_cb.change(
        fn=_on_vr_toggle,
        inputs=[vr_cb],
        outputs=[vr_cable_cb, vr_ip_tb],
    )
    vr_cable_cb.change(
        fn=lambda cable: gr.update(
            value="127.0.0.1" if cable else "",
            interactive=not cable,
        ),
        inputs=[vr_cable_cb],
        outputs=[vr_ip_tb],
    )

    return dict(
        scene=scene_dd, repo_id=repo_id_tb, num_eps=num_eps_nb, fps=fps_nb,
        ep_time=ep_time_sl, root=root_tb, vr=vr_cb, vr_cable=vr_cable_cb,
        vr_ip=vr_ip_tb, launch_btn=launch_btn, launch_status=launch_status,
    )


def launch(scene_name, repo_id, num_eps, fps, ep_time, vr, vr_ip, vr_cable, root_dir) -> str:
    if session.state.running:
        return "A session is already running."

    xml_file, default_task, scene_objects, scene_joints = SCENES.get(
        scene_name, ("icub_table_scene.xml", "tarea libre", [], [])
    )
    model_path = _SCENES_DIR / xml_file
    cfg_path = PROJECT_ROOT / "utils" / "control_config.yaml"

    if not model_path.exists():
        return f"Scene not found: {model_path}"
    if not cfg_path.exists():
        return f"Config not found: {cfg_path}"

    rid = (repo_id.strip() or "local/icub_mujoco_demo")
    rid = rid if "/" in rid else f"local/{rid}"

    resolved_vr_ip = vr_ip.strip() if vr_ip else None
    resolved_vr = bool(vr)

    if vr_cable:
        from dependencies.vr_usb import connect_cable, CABLE_IP
        if not connect_cable(log=session.log):
            return ("USB connection failed. Check the log: connect the Quest, "
                     "enable 'USB Debugging' and accept the popup on the headset.")
        resolved_vr_ip = CABLE_IP
        resolved_vr = True
        session.log(f"[USB] Cable listo. En BeaVR, pon la IP en {CABLE_IP}.")

    args = SimpleNamespace(
        repo_id=rid,
        root=root_dir or str(HUB_ROOT.parent / "data"),
        fps=int(fps),
        num_episodes=int(num_eps),
        scene=None,
        single_task=default_task,
        episode_time_s=int(ep_time),
        config=str(cfg_path),
        model=str(model_path),
        vr=resolved_vr,
        vr_ip=resolved_vr_ip,
        push_to_hub=False,
        scene_objects=scene_objects,
        scene_joints=scene_joints,
    )

    run_suffix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_root = Path(args.root).expanduser()
    dataset_root = base_root / f"{rid.replace('/', '_')}_{run_suffix}"

    def _record(cmd_source, on_status) -> None:
        from play_mujoco import _manual_vr_record
        _manual_vr_record(
            args=args,
            repo_id=rid,
            dataset_root=dataset_root,
            model_path=model_path,
            cfg_path=cfg_path,
            cmd_source=cmd_source,
            on_status=on_status,
        )

    return session.start_session(
        _record, repo_id=rid, dataset_root=dataset_root, num_episodes=int(num_eps),
    )


BACKEND = RobotBackend(
    id="icub_sim",
    label="iCub Simulated MJ",

    image=ASSETS_DIR / "icubsim.png",
    available=True,
    build_config_form=build_config_form,
    launch=launch,
    config_input_order=[
        "scene", "repo_id", "num_eps", "fps", "ep_time",
        "vr", "vr_ip", "vr_cable", "root",
    ],
)
