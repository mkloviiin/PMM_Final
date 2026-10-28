"""Pestaña Setup del Hub v2: una sola pantalla para elegir plataforma, configurar y lanzar.

Diferencias con la Configuration original:
- La plataforma (MuJoCo / Gazebo / real) se elige aquí con tarjetas, no en una pantalla previa
  con "Back": cambiarla no obliga a volver atrás, y Gazebo / real dejan de estar en un formulario
  aparte con un selector de modo adentro.
- Los campos se agrupan por la pregunta que responden (qué tarea, qué dataset, cómo se controla)
  y lo técnico (FPS, carpeta) queda plegado en "Advanced".
- VR en un solo control (sin VR / Wi-Fi / cable USB) en vez de casilla + casilla + IP.
- Gazebo / real: los procesos previos como pasos numerados con un indicador de estado vivo.
- Antes de lanzar, un resumen en lenguaje natural de lo que se va a grabar y dónde.
"""

from __future__ import annotations

import re
from pathlib import Path

import gradio as gr

from launcher_gui import session, state
from launcher_gui.paths import REPO_ROOT
from launcher_gui.robots import icub_rg, icub_sim
from launcher_gui.robots.solver import solver_dropdown
from launcher_gui.ui.dir_picker import dir_textbox

from . import stack
from .stack import P_GAZEBO, P_MUJOCO, P_REAL, PLATFORMS

SETUP_TAB_ID = "setup"
HEADSET_OFF, HEADSET_WIFI, HEADSET_USB = "No VR", "VR over Wi-Fi", "VR over USB cable"
_ARMS = {P_MUJOCO: (["both", "right", "left"], icub_sim._yaml_arms()),
         P_GAZEBO: (["auto", "both", "right", "left"], "auto"),
         P_REAL: (["auto", "both", "right", "left"], "auto")}
_TAG = {P_MUJOCO: "mj", P_GAZEBO: "gz", P_REAL: "real"}
_SCENES = icub_sim.SCENES  # label -> (xml, instrucción, objetos, joints, cámaras); mismos labels que icub_rg

_last_platform: str | None = None


def _instruction_md(task: str) -> str:
    instr = _SCENES.get(task, (None, ""))[1]
    return f"Language instruction recorded with each episode: *“{instr}”*" if instr else ""


def suggest_repo(platform: str, task: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (task or "task").lower()).strip("_")
    return f"local/icub_{_TAG.get(platform, 'mj')}_{slug}"


def summary_md(platform, task, repo, num_eps, ep_time, arms, solver, headset, vr_ip, fps, root) -> str:
    rid = (repo or "").strip() or suggest_repo(platform, task)
    rid = rid if "/" in rid else f"local/{rid}"
    folder = Path(root or REPO_ROOT / "data").expanduser() / f"{rid.replace('/', '_')}_<date-time>"
    try:
        folder = folder.relative_to(REPO_ROOT)
    except ValueError:
        pass
    if headset == HEADSET_OFF:
        how = ("keyboard / mouse — drag the hand targets in the MuJoCo window (Ctrl + drag)"
               if platform == P_MUJOCO else "no headset (console commands only)")
    elif headset == HEADSET_USB:
        how = "Meta Quest over USB cable (BeaVR IP = 127.0.0.1)"
    else:
        how = f"Meta Quest over Wi-Fi ({vr_ip.strip() or 'headset IP not set'})"
    limit = f", up to {int(ep_time)} s each" if ep_time else ""
    return (
        f"**You are about to record {int(num_eps or 0)} episodes{limit}**\n\n"
        f"> **{task}** — {_SCENES.get(task, (None, '?'))[1]}\n\n"
        f"- **Platform:** {platform}\n"
        f"- **Controller:** {solver} · **arms:** {arms}\n"
        f"- **Teleoperation:** {how}\n"
        f"- **Saved to:** `{folder}/` at {int(fps or 30)} FPS"
    )


def _vr_args(headset: str, vr_ip: str) -> tuple[bool, str, bool]:
    """(vr, vr_ip, vr_cable) como los espera el launch() de los backends."""
    return headset != HEADSET_OFF, (vr_ip or "").strip() if headset == HEADSET_WIFI else "", headset == HEADSET_USB


def launch(platform, task, repo, num_eps, ep_time, arms, solver, headset, vr_ip, fps, root) -> str:
    vr, ip, cable = _vr_args(headset, vr_ip)
    repo = (repo or "").strip() or suggest_repo(platform, task)
    if platform == P_MUJOCO:
        return icub_sim.launch(task, repo, num_eps, fps, ep_time, vr, ip, cable, arms, root, solver)
    return icub_rg.launch(stack.rg_mode(platform), task, repo, num_eps, fps, ep_time,
                          vr, ip, cable, arms, root, solver)


def build_setup_tab() -> dict:
    tasks = list(_SCENES)
    with gr.Tab("1 · Setup", id=SETUP_TAB_ID, render_children=True):
        with gr.Row(equal_height=False):
            # ── Izquierda: qué grabar y cómo ─────────────────────────────────
            with gr.Column(scale=3):
                platform = gr.Radio(PLATFORMS, value=P_MUJOCO, label="Where do you want to record?",
                                    elem_classes=["v2-platform"])
                with gr.Column(elem_classes=["v2-card"]):
                    gr.Markdown("Task", elem_classes=["v2-card-title"])
                    task = gr.Dropdown(tasks, value=tasks[0], show_label=False, container=False)
                    instr = gr.Markdown(_instruction_md(tasks[0]), elem_classes=["v2-muted"])
                with gr.Column(elem_classes=["v2-card"]):
                    gr.Markdown("Dataset", elem_classes=["v2-card-title"])
                    with gr.Row():
                        repo = gr.Textbox(suggest_repo(P_MUJOCO, tasks[0]), label="Dataset name", scale=3,
                                          info="Suggested from platform + task; the date-time is appended")
                        num_eps = gr.Number(50, label="Episodes", precision=0, minimum=1, scale=1)
                        ep_time = gr.Number(0, label="Max length (s)", precision=0, minimum=0, scale=1,
                                            info="0 = no limit")
                with gr.Column(elem_classes=["v2-card"]):
                    gr.Markdown("Control", elem_classes=["v2-card-title"])
                    with gr.Row():
                        arms_choices, arms_value = _ARMS[P_MUJOCO]
                        arms = gr.Dropdown(arms_choices, value=arms_value, label="Arms",
                                           info="Teleoperated and recorded")
                        solver = solver_dropdown(icub_sim._CFG_PATH, "IK = joint positions · ID = joint torques")
                    with gr.Row(equal_height=True):
                        headset = gr.Radio([HEADSET_OFF, HEADSET_WIFI, HEADSET_USB], value=HEADSET_OFF,
                                           label="Headset", scale=3)
                        vr_ip = gr.Textbox("", label="Headset IP", placeholder="192.168.x.x", scale=2,
                                           elem_classes=["ip-input", "v2-hidden"])
                with gr.Accordion("Advanced", open=False):
                    with gr.Row():
                        fps = gr.Number(30, label="Dataset FPS", precision=0, minimum=5, maximum=60)
                        root, _ = dir_textbox("Dataset folder", str(REPO_ROOT / "data"),
                                              title="Select the dataset folder")

            # ── Derecha: preparar y lanzar ───────────────────────────────────
            with gr.Column(scale=2):
                with gr.Column(elem_classes=["v2-hidden"]) as steps_col:
                    gr.Markdown("Gazebo and the real robot run as separate processes. "
                                "Start them in order, then the session.", elem_classes=["v2-muted"])
                    with gr.Column(elem_classes=["v2-card", "v2-step"]) as step1:
                        step1_title = gr.Markdown("<span class=v2-num>1</span>Gazebo simulation",
                                                  elem_classes=["v2-card-title"])
                        step1_light = gr.HTML(stack.link_light(P_GAZEBO))
                        with gr.Row():
                            step1_start = gr.Button("▶ Start Gazebo", size="sm", variant="primary")
                            step1_stop = gr.Button("■ Stop", size="sm")
                        step1_msg = gr.Markdown("", elem_classes=["v2-muted"])
                    with gr.Column(elem_classes=["v2-card", "v2-step"]) as step2:
                        gr.Markdown("<span class=v2-num>2</span>Control module (YARP)",
                                    elem_classes=["v2-card-title"])
                        gr.Markdown("Runs the whole-body controller with the solver and arms chosen "
                                    "on the left.", elem_classes=["v2-muted"])
                        step2_light = gr.HTML(stack.module_light())
                        with gr.Row():
                            step2_start = gr.Button("▶ Start module", size="sm", variant="primary")
                            step2_stop = gr.Button("■ Stop", size="sm")
                        step2_msg = gr.Markdown("", elem_classes=["v2-muted"])
                launch_title = gr.Markdown("Start the session", elem_classes=["v2-card-title"])
                with gr.Column(elem_classes=["v2-card", "v2-summary"]):
                    summary = gr.Markdown(summary_md(P_MUJOCO, tasks[0], suggest_repo(P_MUJOCO, tasks[0]), 50, 0,
                                                     _ARMS[P_MUJOCO][1], solver.value, HEADSET_OFF, "", 30,
                                                     str(REPO_ROOT / "data")))
                ready_hint = gr.Markdown("")
                launch_btn = gr.Button("● Start session", variant="primary", elem_classes=["v2-big"])
                launch_msg = gr.Markdown("", elem_classes=["v2-muted"])
        # Dentro de la pestaña: un componente suelto entre las pestañas de gr.Tabs (aunque sea un
        # State) corre el conteo y la barra muestra una pestaña con el nombre de otra
        last_suggest = gr.State(suggest_repo(P_MUJOCO, tasks[0]))

    form = [platform, task, repo, num_eps, ep_time, arms, solver, headset, vr_ip, fps, root]

    # ── Wiring interno ───────────────────────────────────────────────────────
    def _on_platform(p, t, r, last, n):
        global _last_platform
        # Otra plataforma: Record queda como nuevo (igual que Episode Control en el Hub original)
        if _last_platform not in (None, p):
            session.reset_episode_control(n)
        _last_platform = p
        choices, value = _ARMS[p]
        new = suggest_repo(p, t)
        rg = p != P_MUJOCO
        return (
            gr.update(choices=choices, value=value),
            gr.update(value=new) if r in ("", last) else gr.update(), new,
            gr.update(elem_classes=[] if rg else ["v2-hidden"]),
            "<span class=v2-num>1</span>" + ("Gazebo simulation" if p == P_GAZEBO else "Robot connection"),
            gr.update(value="▶ Start Gazebo" if p == P_GAZEBO else "▶ Connect"),
            gr.update(value="■ Stop" if p == P_GAZEBO else "■ Disconnect"),
            stack.link_light(p) if rg else "",
            ("<span class=v2-num>3</span>" if rg else "") + "Start the session",
        )

    platform.change(_on_platform, [platform, task, repo, last_suggest, num_eps],
                    [arms, repo, last_suggest, steps_col, step1_title, step1_start, step1_stop,
                     step1_light, launch_title])

    def _on_task(p, t, r, last):
        new = suggest_repo(p, t)
        return _instruction_md(t), gr.update(value=new) if r in ("", last) else gr.update(), new

    task.change(_on_task, [platform, task, repo, last_suggest], [instr, repo, last_suggest])
    headset.change(lambda h: gr.update(elem_classes=["ip-input"] + ([] if h == HEADSET_WIFI else ["v2-hidden"])),
                   [headset], [vr_ip])
    num_eps.change(session.set_planned_episodes, [num_eps])
    for comp in form:
        comp.change(summary_md, form, [summary])

    step1_start.click(stack.start_link, [platform], [step1_msg]).then(
        lambda p: stack.link_light(p), [platform], [step1_light])
    step1_stop.click(stack.stop_link, [platform], [step1_msg]).then(
        lambda p: stack.link_light(p), [platform], [step1_light])
    step2_start.click(stack.start_module, [platform, arms, solver], [step2_msg]).then(
        lambda: stack.module_light(), None, [step2_light])
    step2_stop.click(stack.stop_module, None, [step2_msg]).then(
        lambda: stack.module_light(), None, [step2_light])

    return dict(form=form, platform=platform, num_eps=num_eps, summary=summary, launch_btn=launch_btn,
                launch_msg=launch_msg, ready_hint=ready_hint, step1=step1, step2=step2,
                step1_light=step1_light, step2_light=step2_light)


def poll_steps(platform: str):
    """Timer (cada 2 s): indicadores de los pasos, borde verde del paso listo y el botón de lanzar
    (deshabilitado con una sesión en curso, con aviso si falta un paso)."""
    running = state.running
    btn = gr.update(value="Session running — see Record" if running else "● Start session",
                    interactive=not running)
    if platform == P_MUJOCO:
        return gr.update(), gr.update(), gr.update(), gr.update(), btn, gr.update()
    l_ok, m_ok = stack.link_ok(platform), stack.module_ok()
    missing = [s for s, ok in (("step 1", l_ok), ("step 2", m_ok)) if not ok]
    hint = "" if running or not missing else f"⚠ Complete {' and '.join(missing)} before starting the session."
    cls = lambda ok: ["v2-card", "v2-step"] + (["v2-step-ok"] if ok else [])  # noqa: E731
    return (stack.link_light(platform), stack.module_light(), gr.update(elem_classes=cls(l_ok)),
            gr.update(elem_classes=cls(m_ok)), btn, hint)
