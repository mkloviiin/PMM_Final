"""Backend de robot: iCub **Real / Gazebo** (vía YARP).

Encapsula todo lo propio de este robot. A diferencia de `icub_sim` (MuJoCo,
in-process), aquí la grabación corre en un **subproceso aislado**
(`icub-lerobot-gazebo/play_gazebo.py`) — así los paquetes de este proyecto
(`lerobot_robot_icubrg`, `lerobot_teleoperator_icubrgteleop`) nunca comparten
el intérprete con los del backend MuJoCo, y una caída de la sesión no arrastra
al Hub. El resto del Hub (Control de episodios, Visualizar y curar, Subir a
Hugging Face) no sabe nada de esto: solo le importa que el dataset quede en
formato LeRobot bajo `dataset_root` y que la grabación hable el protocolo de
`session.start_session` (ver robots/base.py).

El stack real/Gazebo son tres procesos que el usuario levanta por partes desde
la pestaña de configuración (ver README de icub-lerobot-gazebo):
  1. Simulación Gazebo (`gazebo/scripts/start_sim.sh`)  — solo en modo Gazebo.
  2. Módulo YARP (`common/yarp_module/teleop_module_sm.py`) — servidor.
  3. Teleop/Grabación (`play_gazebo.py`) — el cliente que graba (= la "sesión").

El único switch real/Gazebo es `robot_name` (`icub`/`icubSim`): se pasa como
`--robot` al módulo YARP y `--robot-name` al play, sin reescribir el YAML.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import yaml

from .. import procman, session
from ..paths import ASSETS_DIR, GAZEBO_ROOT, HUB_ROOT
from .base import RobotBackend

# ── Rutas del proyecto gazebo ────────────────────────────────────────────────
_SCENES_DIR = GAZEBO_ROOT / "mujoco" / "assets" / "scenes"
_SCENES_MANIFEST = _SCENES_DIR / "scenes.yaml"
# En gazebo/real el visualizador MuJoCo es solo un espejo: siempre mesa vacía,
# porque los objetos se colocan en el mundo real (o en Gazebo), no en MuJoCo.
_EMPTY_TABLE = _SCENES_DIR / "scene_icub_empty_table.xml"
_CFG_PATH = GAZEBO_ROOT / "config" / "control_config.yaml"
_PLAY = GAZEBO_ROOT / "play_gazebo.py"
_YARP_MODULE = GAZEBO_ROOT / "common" / "yarp_module" / "teleop_module_sm.py"
_SCRIPTS_DIR = GAZEBO_ROOT / "gazebo" / "scripts"
_SIM_START = _SCRIPTS_DIR / "start_sim.sh"
_SIM_STOP = _SCRIPTS_DIR / "stop_sim.sh"
_TMUX_SESSION = "icub_sim"

# ── Modo real/Gazebo ─────────────────────────────────────────────────────────
MODE_GAZEBO = "Gazebo (Simulation)"
MODE_REAL = "Real Robot"


# Name server YARP del robot real. En Gazebo no hace falta: el `yarpserver --write`
# de start_sim.sh ya deja `yarp conf` apuntando al servidor local.
REAL_YARP_IP = "10.0.0.1"
REAL_YARP_PORT = 10000


def _robot_name(mode: str) -> str:
    return "icub" if mode == MODE_REAL else "icubSim"


def _yarp_bin() -> str:
    """El `yarp` del mismo entorno conda que el Hub (puede no estar en el PATH)."""
    local = Path(sys.executable).parent / "yarp"
    return str(local) if local.exists() else "yarp"


def _connect_real_robot() -> tuple[bool, str]:
    """`yarp conf <ip> <port>` + `yarp detect`. Devuelve (conectado, mensaje)."""
    yarp = _yarp_bin()
    rc, out = procman.run_once([yarp, "conf", REAL_YARP_IP, str(REAL_YARP_PORT)], timeout=10)
    if rc != 0:
        return False, f"`yarp conf {REAL_YARP_IP} {REAL_YARP_PORT}` failed (rc={rc}). Check the log."
    rc, out = procman.run_once([yarp, "detect"], timeout=20)
    if rc == 0 and "FOUND" in out and "NOT FOUND" not in out:
        return True, f"Connected to the robot's YARP name server at {REAL_YARP_IP}:{REAL_YARP_PORT}."
    return False, (f"Robot name server not reachable at {REAL_YARP_IP}:{REAL_YARP_PORT} "
                   f"(`yarp detect` rc={rc}). Is the robot on and the network cable connected?")


def _load_scenes() -> dict[str, tuple[str, str]]:
    """Lee scenes.yaml del proyecto gazebo: {label: (archivo_xml, tarea_default)}.

    En Gazebo/real el objeto no tiene pose observable, así que `objects`/`joints`
    del manifiesto no se usan aquí (play_gazebo no los recibe): solo importan la
    escena del visualizador MuJoCo (`--model`) y la instrucción (`--single-task`).
    """
    if not _SCENES_MANIFEST.exists():
        print(f"[icub_rg] Warning: {_SCENES_MANIFEST} not found; no tasks.")
        return {}
    entries = yaml.safe_load(_SCENES_MANIFEST.read_text(encoding="utf-8")) or []
    scenes = {e["label"]: (e["file"], e["task"]) for e in entries}

    declared = {e["file"] for e in entries}
    all_xml = {p.name for p in _SCENES_DIR.glob("*.xml")}
    orphans = all_xml - declared
    if orphans:
        print(f"[icub_rg] Warning: {len(orphans)} XML(s) in {_SCENES_DIR} are not "
              f"declared in scenes.yaml and will not appear in the dropdown: "
              f"{', '.join(sorted(orphans))}")
    return scenes


SCENES = _load_scenes()


def _tmux_alive() -> bool:
    try:
        r = subprocess.run(
            ["tmux", "has-session", "-t", _TMUX_SESSION],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


# ── Puente subproceso play_gazebo.py ↔ protocolo de sesión ───────────────────
def _dir_size_mb(path: Path) -> float:
    try:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)
    except Exception:  # noqa: BLE001
        return 0.0


def _bridge_play(argv, *, cmd_source, on_status, dataset_root: Path | None = None) -> None:
    """Corre play_gazebo.py como subproceso y traduce entre el protocolo del
    Hub (cmd_source/on_status) y el stdin/stdout del hijo.

    Se ejecuta dentro del thread de `session.start_session`, cuyo stdout está
    redirigido al log del Hub: por eso los `print()` de aquí ya se ven en el log.
    """
    proc = subprocess.Popen(
        [str(a) for a in argv],
        cwd=str(GAZEBO_ROOT),
        env={**os.environ, "PYTHONUNBUFFERED": "1", "PMM_LIVE_METRICS": "1"},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    print(f"[play] launched (pid={proc.pid})")

    stop_flag = {"exit": False}

    def _forward_cmds() -> None:
        # cmd_source() devuelve "1"/"2"/"3" (START/STOP/EXIT) o None. Se los
        # escribimos por stdin al listener de consola de play_gazebo.
        while proc.poll() is None and not stop_flag["exit"]:
            cmd = cmd_source() if cmd_source else None
            if cmd:
                try:
                    proc.stdin.write(f"{cmd}\n")
                    proc.stdin.flush()
                except Exception:  # noqa: BLE001
                    break
                if cmd == "3":
                    stop_flag["exit"] = True
                    break
            else:
                time.sleep(0.02)

    threading.Thread(target=_forward_cmds, daemon=True).start()

    saved_re = re.compile(r"Episode saved \((\d+)/")
    live = {"fps": 0.0, "latency_ms": 0.0, "size_mb": 0.0}
    t_size = 0.0
    try:
        for raw in iter(proc.stdout.readline, ""):
            line = raw.rstrip("\n")
            if not line:
                continue
            if line.startswith("[Live] "):
                # Métricas en vivo para el panel; no van al log (llegan 2 veces por segundo)
                try:
                    live.update(json.loads(line[7:]))
                except ValueError:
                    continue
                if dataset_root is not None and time.time() - t_size > 2.0:
                    live["size_mb"] = _dir_size_mb(dataset_root)
                    t_size = time.time()
                if on_status is not None:
                    on_status({"metrics": dict(live)})
                continue
            print(f"[play] {line}")
            if on_status is None:
                continue
            if "Waiting START" in line:
                live.update(fps=0.0, latency_ms=0.0)
                if dataset_root is not None:
                    live["size_mb"] = _dir_size_mb(dataset_root)
                on_status({"status": "waiting", "metrics": dict(live)})
            elif "Recording episode" in line:
                on_status("recording")
            else:
                m = saved_re.search(line)
                if m:
                    on_status(f"saved:{m.group(1)}")
    finally:
        stop_flag["exit"] = True
        if proc.poll() is None:
            # Pedir salida ordenada; si no muere, señal al grupo y kill.
            try:
                proc.stdin.write("3\n")
                proc.stdin.flush()
            except Exception:  # noqa: BLE001
                pass
            try:
                proc.wait(timeout=12)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                    proc.wait(timeout=5)
                except Exception:  # noqa: BLE001
                    proc.kill()
        print(f"[play] play_gazebo finished (rc={proc.poll()})")


# ── Formulario de configuración ──────────────────────────────────────────────
def build_config_form() -> dict[str, gr.components.Component]:
    scene_choices = list(SCENES) or ["(no tasks)"]

    with gr.Row():
        with gr.Column(scale=1):
            mode_dd = gr.Radio(
                choices=[MODE_GAZEBO, MODE_REAL], value=MODE_GAZEBO,
                label="Mode",
                info="Gazebo → robot_name=icubSim · Real → robot_name=icub",
            )
            scene_dd = gr.Dropdown(
                choices=scene_choices, value=scene_choices[0],
                label="Tasks",
                info="Instruction to record."
                     "table; objects are placed in the real world / Gazebo.",
            )
            repo_id_tb = gr.Textbox(
                value="local/icub_gazebo_demo", label="Repo ID",
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
            ep_time_nb = gr.Number(
                value=0, label="Max episode duration (s)",
                precision=0, minimum=0, info="0 = no time limit",
            )
            root_tb = gr.Textbox(
                value=str(HUB_ROOT.parent / "data"),
                label="Dataset root directory",
            )
            arms_dd = gr.Dropdown(
                choices=["auto", "right", "left", "both"], value="auto",
                label="Control arms",
                info="'auto' detects the arms available in YARP",
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
    gr.Markdown("### Stack — Start each part in order")
    with gr.Row():
        with gr.Column():
            gr.Markdown("**1. Gazebo simulation / Real robot connection**")
            with gr.Row():
                sim_start_btn = gr.Button(
                    "▶ Connect to robot" if mode_dd.value == MODE_REAL else "▶ Start sim",
                    size="sm")
                sim_stop_btn = gr.Button("■ Stop sim", size="sm")
            sim_status = gr.Textbox(label="Simulation / connection status", interactive=False)
        with gr.Column():
            gr.Markdown("**2. YARP module**")
            with gr.Row():
                yarp_start_btn = gr.Button("▶ Start YARP module", size="sm")
                yarp_stop_btn = gr.Button("■ Stop YARP module", size="sm")
            yarp_status = gr.Textbox(label="YARP module status", interactive=False)

    gr.Markdown("---")
    gr.Markdown("### 3. Teleop / recording")
    launch_btn = gr.Button("LAUNCH TELEOP / RECORDING", variant="primary", size="lg")
    with gr.Row():
        play_stop_btn = gr.Button("■ Stop teleop / recording", size="sm")
    launch_status = gr.Textbox(label="Launch status", interactive=False)

    # ── Wiring interno (autocontenido, como icub_sim con los toggles de VR) ──
    def _on_vr_toggle(enabled):
        if enabled:
            return gr.update(visible=True), gr.update(visible=True)
        return gr.update(visible=False, value=False), gr.update(visible=False, value="")

    vr_cb.change(fn=_on_vr_toggle, inputs=[vr_cb], outputs=[vr_cable_cb, vr_ip_tb])
    vr_cable_cb.change(
        fn=lambda cable: gr.update(value="127.0.0.1" if cable else "", interactive=not cable),
        inputs=[vr_cable_cb], outputs=[vr_ip_tb],
    )

    def _on_mode_change(mode):
        is_gz = (mode == MODE_GAZEBO)
        return (gr.update(value="▶ Start sim" if is_gz else "▶ Connect to robot"),
                gr.update(interactive=is_gz))

    mode_dd.change(fn=_on_mode_change, inputs=[mode_dd], outputs=[sim_start_btn, sim_stop_btn])

    # 1. Simulación Gazebo (fire-and-forget: crean/cierran la sesión tmux)
    def _sim_start(mode):
        if mode == MODE_REAL:
            return _connect_real_robot()[1]
        if not _SIM_START.exists():
            return f"Not found: {_SIM_START}"
        if _tmux_alive():
            return f"Simulation is already running (tmux '{_TMUX_SESSION}')."
        procman.run_once(["bash", str(_SIM_START)], cwd=str(_SCRIPTS_DIR))
        return f"start_sim.sh executed — starting tmux '{_TMUX_SESSION}' (~15 s). Check the log."

    def _sim_stop():
        if not _SIM_STOP.exists():
            return f"Not found: {_SIM_STOP}"
        procman.run_once(["bash", str(_SIM_STOP)], cwd=str(_SCRIPTS_DIR))
        return "stop_sim.sh executed."

    sim_start_btn.click(fn=_sim_start, inputs=[mode_dd], outputs=[sim_status])
    sim_stop_btn.click(fn=_sim_stop, inputs=[], outputs=[sim_status])

    # 2. Módulo YARP (long-lived, gestionado por procman)
    def _yarp_start(mode):
        if not _YARP_MODULE.exists():
            return f"Not found: {_YARP_MODULE}"
        if mode == MODE_REAL:
            # Se repite aquí por si se saltó el paso 1: el módulo no debe arrancar
            # apuntando al name server equivocado.
            ok, msg = _connect_real_robot()
            if not ok:
                return msg
        robot_name = _robot_name(mode)
        if not procman.is_running("yarp_module"):
            procman.stop_stale("yarp_module", str(_YARP_MODULE))
        argv = [sys.executable, "-u", str(_YARP_MODULE), "--robot", robot_name]
        return procman.start(
            "yarp_module", argv,
            cwd=str(_YARP_MODULE.parent),
            env={"ICUB_LEROBOT_CONFIG": str(_CFG_PATH), "ICUB_ROBOT_NAME": robot_name},
        )

    yarp_start_btn.click(fn=_yarp_start, inputs=[mode_dd], outputs=[yarp_status])
    yarp_stop_btn.click(fn=lambda: procman.stop("yarp_module"), inputs=[], outputs=[yarp_status])

    # 3. Detener play/grabación = EXIT de la sesión (mismo protocolo genérico)
    play_stop_btn.click(fn=session.cmd_exit, inputs=[], outputs=[launch_status])

    return dict(
        mode=mode_dd, scene=scene_dd, repo_id=repo_id_tb, num_eps=num_eps_nb,
        fps=fps_nb, ep_time=ep_time_nb, root=root_tb, control_arms=arms_dd,
        vr=vr_cb, vr_cable=vr_cable_cb, vr_ip=vr_ip_tb,
        launch_btn=launch_btn, launch_status=launch_status,
    )


# ── Lanzamiento de la parte "play" (la sesión de grabación) ──────────────────
def launch(mode, scene_name, repo_id, num_eps, fps, ep_time,
           vr, vr_ip, vr_cable, control_arms, root_dir) -> str:
    if session.state.running:
        return "A session is already running."

    if scene_name not in SCENES:
        return f"Invalid task: {scene_name}. Check {_SCENES_MANIFEST}."
    # Solo importa la instrucción; el XML de la escena se ignora a propósito:
    # en gazebo/real MuJoCo es un espejo y siempre carga la mesa vacía.
    _xml_file, default_task = SCENES[scene_name]
    model_path = _EMPTY_TABLE

    if not model_path.exists():
        return f"Scene not found: {model_path}"
    if not _CFG_PATH.exists():
        return f"Config not found: {_CFG_PATH}"
    if not _PLAY.exists():
        return f"play_gazebo.py not found: {_PLAY}"

    rid = (repo_id.strip() or "local/icub_gazebo_demo")
    rid = rid if "/" in rid else f"local/{rid}"
    robot_name = _robot_name(mode)

    resolved_vr_ip = vr_ip.strip() if vr_ip else None
    resolved_vr = bool(vr)
    resolved_cable = bool(vr_cable)

    run_suffix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_root = Path(root_dir or str(HUB_ROOT.parent / "data")).expanduser()
    dataset_root = base_root / f"{rid.replace('/', '_')}_{run_suffix}"

    argv = [
        sys.executable, "-u", str(_PLAY),
        "--config", str(_CFG_PATH),
        "--model", str(model_path),
        "--repo-id", rid,
        "--dataset-root", str(dataset_root),
        "--fps", str(int(fps)),
        "--num-episodes", str(int(num_eps)),
        "--single-task", default_task,
        "--episode-time-s", str(int(ep_time)),
        "--control-arms", control_arms or "auto",
        "--robot-name", robot_name,
    ]
    if resolved_vr:
        argv.append("--vr")
    if resolved_vr_ip:
        argv += ["--vr-ip", resolved_vr_ip]
    if resolved_cable:
        argv.append("--vr-cable")

    def _record(cmd_source, on_status) -> None:
        _bridge_play(argv, cmd_source=cmd_source, on_status=on_status, dataset_root=dataset_root)

    return session.start_session(
        _record, repo_id=rid, dataset_root=dataset_root, num_episodes=int(num_eps),
    )


BACKEND = RobotBackend(
    id="icub_rg",
    label="iCub Real / Gazebo",
    image=ASSETS_DIR / "icub.png",
    available=True,
    build_config_form=build_config_form,
    launch=launch,
    config_input_order=[
        "mode", "scene", "repo_id", "num_eps", "fps", "ep_time",
        "vr", "vr_ip", "vr_cable", "control_arms", "root",
    ],
)
