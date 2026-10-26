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

import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import yaml

from .. import ports, session
from ..paths import ASSETS_DIR, HUB_ROOT, REPO_ROOT, PLAY_SCRIPT
from ..ui.dir_picker import dir_textbox
from .base import RobotBackend
from .icub_rg import _bridge_play

PROJECT_ROOT = REPO_ROOT

_SCENES_DIR = PROJECT_ROOT / "mujoco_sim" / "assets" / "scenes"
_SCENES_MANIFEST = _SCENES_DIR / "scenes.yaml"
_CFG_PATH = PROJECT_ROOT / "config" / "control_config.yaml"


def _ensure_mj_on_path() -> None:
    """Inserta el proyecto y sus plugins al frente de sys.path (idempotente)."""
    import sys
    for _p in (
        PROJECT_ROOT / "src" / "plugins" / "lerobot-robot-icub",
        PROJECT_ROOT / "src" / "plugins" / "lerobot-teleoperator-icubteleop",
        PROJECT_ROOT / "src",
        PROJECT_ROOT,
    ):
        _s = str(_p)
        if _s not in sys.path:
            sys.path.insert(0, _s)


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
    gr.Markdown("### 1. Session settings")
    # .cfg-row (app.py): con "Enable VR" marcado, las dos columnas quedan de la misma altura
    with gr.Row(elem_classes=["cfg-row"]):
        with gr.Column(scale=1):
            scene_dd = gr.Dropdown(
                choices=list(SCENES), value=list(SCENES)[0],
                label="Tasks",
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
            root_tb, _ = dir_textbox(
                "Dataset root directory", str(HUB_ROOT.parent / "data"),
                title="Select the dataset root directory",
            )
            arms_dd = gr.Dropdown(
                choices=["both", "right", "left"], value=_yaml_arms(),
                label="Control arms",
                info="Arms driven by the teleoperation (and recorded in the dataset)",
            )
            # Cable e IP solo con "Enable VR" marcado: los muestra/oculta el CSS .vr-extra de
            # app.py según la casilla (visible= de Gradio no era confiable y a veces quedaban
            # a la vista sin VR).
            vr_cb = gr.Checkbox(value=False, label="Enable VR", elem_classes=["vr-cb"])
            vr_cable_cb = gr.Checkbox(
                value=False,
                label="Connect MetaQuest via USB cable",
                info="Tunnels VR ports over USB; set IP = 127.0.0.1 in BeaVR",
                elem_classes=["vr-extra"],
            )
            vr_ip_tb = gr.Textbox(
                value="", placeholder="192.168.x.x",
                label="VR headset IP",
                elem_classes=["ip-input", "vr-extra"],
            )

    gr.Markdown("---")
    gr.Markdown("### 2. Launch session")
    launch_btn = gr.Button("LAUNCH SESSION", variant="primary", size="lg")
    with gr.Row():
        play_stop_btn = gr.Button("■ Stop teleop / recording", size="sm")
    launch_status = gr.Textbox(label="Launch status", interactive=False)
    # Igual que en icub_rg: cierra la sesión activa (comando "3" de la cola de session)
    play_stop_btn.click(fn=session.cmd_exit, inputs=[], outputs=[launch_status])

    def _on_vr_toggle(enabled):
        # La visibilidad la maneja el CSS (.vr-extra); al desactivar el VR se limpian los valores
        if enabled:
            return gr.update(), gr.update()
        return gr.update(value=False), gr.update(value="", interactive=True)

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
        ep_time=ep_time_sl, root=root_tb, vr=vr_cb, vr_cable=vr_cable_cb, control_arms=arms_dd,
        vr_ip=vr_ip_tb, launch_btn=launch_btn, launch_status=launch_status,
    )


def _yaml_arms() -> str:
    """control_arms de utils/control_config.yaml de MuJoCo (valor inicial del selector)."""
    try:
        cfg = yaml.safe_load((MJ_ROOT / "utils" / "control_config.yaml").read_text(encoding="utf-8")) or {}
        arms = cfg.get("control_arms", "both")
        return arms if arms in ("both", "right", "left") else "both"
    except Exception:  # noqa: BLE001
        return "both"


def launch(scene_name, repo_id, num_eps, fps, ep_time, vr, vr_ip, vr_cable, control_arms, root_dir) -> str:
    if session.state.running:
        return "A session is already running."
    # El núcleo MuJoCo abre los puertos del VR y del streaming (también los usan el
    # teleoperador de Gazebo y el módulo YARP): libres, o liberados si son de una sesión
    # anterior de este Hub.
    busy = ports.preflight(ports.VR_PORTS + ports.STREAM_PORTS, "MuJoCo", log=session.log)
    if busy:
        return busy

    _ensure_mj_on_path()

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
        from interfaces.vr_usb import connect_cable, CABLE_IP
        if not connect_cable(log=session.log):
            return ("USB connection failed. Check the log: connect the Quest, "
                     "enable 'USB Debugging' and accept the popup on the headset.")
        resolved_vr_ip = CABLE_IP
        resolved_vr = True
        session.log(f"[USB] Cable listo. En BeaVR, pon la IP en {CABLE_IP}.")

    run_suffix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_root = Path(root_dir or str(HUB_ROOT.parent / "data")).expanduser()
    dataset_root = base_root / f"{rid.replace('/', '_')}_{run_suffix}"

    argv = [
        sys.executable, "-u", str(PLAY_SCRIPT),
        "--backend", "mujoco",
        "--config", str(cfg_path),
        "--model", str(model_path),
        "--repo-id", rid,
        "--dataset-root", str(dataset_root),
        "--fps", str(int(fps)),
        "--num-episodes", str(int(num_eps)),
        "--single-task", default_task,
        "--episode-time-s", str(int(ep_time)),
        "--control-arms", control_arms or "both",
    ]
    if resolved_vr:
        argv.append("--vr")
    if resolved_vr_ip:
        argv += ["--vr-ip", resolved_vr_ip]

    def _record(cmd_source, on_status) -> None:
        try:
            _bridge_play(argv, cmd_source=cmd_source, on_status=on_status, dataset_root=dataset_root)
        finally:
            # Red de seguridad: ningún núcleo MuJoCo de esta sesión queda reteniendo los
            # puertos (si no, la próxima sesión de cualquier plataforma no podría abrirlos)
            n = ports.release_own()
            if n:
                session.log(f"[Hub] Closed {n} MuJoCo core(s) left open by the session.")

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
        "vr", "vr_ip", "vr_cable", "control_arms", "root",
    ],
)
