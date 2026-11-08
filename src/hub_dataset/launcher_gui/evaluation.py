"""Pestaña 'Evaluate': evalúa una política entrenada con eval_policy.py (MuJoCo o Gazebo).

Misma lógica que training.py: la evaluación corre desacoplada del Hub (sesión propia, salida a
un archivo de log); si se cierra el Hub sigue corriendo y al volver a abrirlo se detecta por
logs/eval/current.json. Mientras evalúa, el resto del Hub queda bloqueado (app.py): usa el
simulador / el módulo YARP y la GPU.

Diferencia con el entrenamiento: al detenerla los resultados NO se borran. eval_policy.py
reescribe summary.json después de cada episodio, así que una evaluación detenida deja los
episodios terminados. Se detiene con SIGINT (KeyboardInterrupt -> su `finally` cierra los .mp4
y el robot); SIGTERM / SIGKILL solo si no termina.

Gazebo: todo se levanta desde aquí, sin pasar por Configuration. Se lanza eval_gazebo_stack.py,
que levanta la simulación (si no está), el módulo YARP con los brazos del modelo y el solver
elegido, corre eval_policy.py y al final cierra lo que levantó.

El robot real no está: eval_policy.py le pide al operador por consola (input) colocar los
objetos y decir si hubo éxito, y el proceso desacoplado no tiene consola.
"""

from __future__ import annotations

import html
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import gradio as gr

from . import procman, state, training
from .paths import HUB_ROOT, REPO_ROOT
from .robots.icub_sim import SCENES
from .robots.solver import SOLVERS, solver_argv
from interfaces.control_metadata import controller_from_metadata, find_model_control_config  # noqa: E402  (src/ en sys.path vía robots.icub_sim)
from .session import log as hub_log
from .training import _fmt_dur, _group_alive, _int, _log_lines, _rc, _tail

EVAL_LOG_DIR = HUB_ROOT / "logs" / "eval"
CURRENT_FILE = EVAL_LOG_DIR / "current.json"
EVAL_SCRIPT = REPO_ROOT / "eval_policy.py"
GAZEBO_STACK = Path(__file__).resolve().parent / "eval_gazebo_stack.py"
DEFAULT_MODELS_DIR = training.DEFAULT_RUNS_DIR
DEFAULT_RESULTS_DIR = REPO_ROOT / "output" / "eval"  # el mismo default que eval_policy.py

BACKEND_MUJOCO = "MuJoCo"
BACKEND_GAZEBO = "Gazebo"
BACKENDS = {BACKEND_MUJOCO: "mujoco", BACKEND_GAZEBO: "gazebo"}

# Lado de la tarea en escenas de dos lados (solo MuJoCo) -> argumentos de eval_policy.py
SIDE_ALTERNATE = "Alternate (right, left, ...)"
SIDES = {
    SIDE_ALTERNATE: ["--mirror", "alternate"],
    "Random (50/50)": ["--mirror", "random"],
    "Right only": ["--side", "right"],
    "Left only": ["--side", "left"],
}

DEFAULTS = dict(num_episodes=10, episode_len=600, fps=30)

_lock = threading.Lock()


# ── Modelos ──────────────────────────────────────────────────────────────────

def _ckpt_names(model_dir: Path) -> list[str]:
    """Checkpoints con modelo: "last" primero, después los numerados del más nuevo al más viejo."""
    ck = model_dir / "checkpoints"
    if not ck.is_dir():
        return []
    names = [p.name for p in ck.iterdir() if (p / "pretrained_model" / "config.json").exists()]
    return sorted(names, key=lambda n: (n != "last", -int(n) if n.isdigit() else 0))


def refresh_models(models_dir: str, current: str | None = None):
    """Dropdown "Model": corridas de entrenamiento con al menos un checkpoint (la más nueva
    primero). Conserva la selección si sigue existiendo (o si es un ID de Hugging Face)."""
    root = Path(models_dir or DEFAULT_MODELS_DIR).expanduser()
    runs = []
    if root.is_dir():
        runs = [p for p in root.iterdir() if p.is_dir() and _ckpt_names(p)]
        runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    choices = [(p.name, str(p)) for p in runs]
    paths = [c[1] for c in choices]
    value = current if current and (current in paths or not Path(current).exists()) else (
        paths[0] if paths else None)
    return gr.update(choices=choices, value=value)


def resolve_checkpoint(model: str | None, ckpt: str | None) -> str | None:
    """Carpeta pretrained_model (o ID de Hugging Face) que recibe eval_policy.py --checkpoint."""
    model = (model or "").strip()
    if not model:
        return None
    p = Path(model).expanduser()
    if (p / "checkpoints").is_dir():
        names = _ckpt_names(p)
        name = ckpt if ckpt in names else (names[0] if names else None)
        return str(p / "checkpoints" / name / "pretrained_model") if name else None
    if (p / "pretrained_model" / "config.json").exists():
        return str(p / "pretrained_model")
    if (p / "config.json").exists():
        return str(p)
    # No es una carpeta local: ID de Hugging Face (user_or_org/model)
    return model if re.fullmatch(r"[\w.\-]+/[\w.\-]+", model) and not p.exists() else None


def _model_name(model: str) -> str:
    return Path(model).name if Path(model).expanduser().exists() else model.replace("/", "_")


def _scene_for_task(task: str) -> str | None:
    t = task.strip().lower()
    return next((label for label, v in SCENES.items() if v[1].strip().lower() == t), None) if t else None


def model_info(checkpoint: str | None) -> dict:
    """Lo que se sabe del modelo local: cámaras, chunk, dataset de entrenamiento, su tarea y brazos."""
    if not checkpoint or not Path(checkpoint).exists():
        return {}
    info: dict = {}
    try:
        cfg = json.loads((Path(checkpoint) / "config.json").read_text())
        info["cameras"] = [k.removeprefix("observation.images.") for k in cfg.get("input_features", {})
                           if k.startswith("observation.images.")]
        info["chunk_size"], info["n_action_steps"] = cfg.get("chunk_size"), cfg.get("n_action_steps")
        info["type"] = cfg.get("type")
    except Exception:  # noqa: BLE001
        return {}
    try:
        tc = json.loads((Path(checkpoint) / "train_config.json").read_text())
        info["steps"] = tc.get("steps")
        root = Path(tc["dataset"]["root"]).expanduser()
        info["dataset"] = root.name
        info["task"] = training._dataset_task(root)
        ds = training.dataset_info(str(root)) or {}
        actions = ds.get("features", {}).get("action", {}).get("names") or []
        r, l = any(n.startswith("rh_") for n in actions), any(n.startswith("lh_") for n in actions)
        info["arms"] = "both" if r and l else "right" if r else "left" if l else None
    except Exception:  # noqa: BLE001
        pass
    return info


NO_SOLVER_MSG = ("This model has no controller metadata (control_config.json in the checkpoint, or "
                 "meta/control_config.json in the dataset it was trained on), so it can't be evaluated.")


def model_solver(checkpoint: str | None) -> str | None:
    """Opción del selector que corresponde al controlador con que se grabó el dataset del modelo
    (misma búsqueda que eval_policy.py). None si no hay metadatos (o es un ID de Hugging Face)."""
    if not checkpoint or not Path(checkpoint).exists():
        return None
    meta_file = find_model_control_config(checkpoint)
    if meta_file is None:
        return None
    try:
        key = controller_from_metadata(json.loads(meta_file.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None
    return next((label for label, v in SOLVERS.items() if v == key), None)


def on_model_change(model: str | None, ckpt: str | None):
    """Al elegir modelo: sus checkpoints, resumen, y la tarea / instrucción con que se entrenó.
    Orden de salida: checkpoint, resumen, tarea, instrucción, action steps, controlador."""
    model = (model or "").strip()
    p = Path(model).expanduser() if model else None
    names = _ckpt_names(p) if p and p.is_dir() else []
    ck_upd = (gr.update(choices=names, value=ckpt if ckpt in names else names[0], visible=True)
              if names else gr.update(choices=[], value=None, visible=False))
    if not model:
        return ck_upd, "Select a model.", gr.update(), gr.update(), gr.update(), gr.update(value=None)
    checkpoint = resolve_checkpoint(model, ckpt if ckpt in names else (names[0] if names else None))
    if checkpoint is None:
        return ck_upd, "Not a trained model: choose a run folder or type a Hugging Face model ID.", \
            gr.update(), gr.update(), gr.update(), gr.update(value=None)
    info = model_info(checkpoint)
    if not info:
        return ck_upd, (f"Hugging Face model **{html.escape(model)}**: it is downloaded when the "
                        f"evaluation starts. Choose the task below."), gr.update(), gr.update(), gr.update(), \
            gr.update(value=None)
    parts = [f"**{info.get('type') or 'policy'}**"]
    if info.get("steps"):
        parts.append(f"{info['steps']:,} training steps")
    parts.append(f"cameras: {', '.join(info['cameras']) or 'none'}")
    if info.get("arms"):
        parts.append(f"arms: {info['arms']}")
    parts.append(f"chunk {info['chunk_size']} · {info['n_action_steps']} actions per inference")
    summary = " · ".join(parts)
    if info.get("dataset"):
        summary += f"  \nTrained on `{html.escape(info['dataset'])}`"
    if info.get("task"):
        summary += f"  \nTask: *{html.escape(info['task'])}*"
    solver = model_solver(checkpoint)
    if solver is None:
        summary += f"  \n⚠️ {NO_SOLVER_MSG}"
    scene = _scene_for_task(info.get("task") or "")
    return (ck_upd, summary,
            gr.update(value=scene) if scene else gr.update(),
            gr.update(value=info["task"]) if info.get("task") else gr.update(),
            gr.update(value=info.get("n_action_steps")),
            gr.update(value=solver))


def on_scene_change(scene: str | None, model: str | None, ckpt: str | None) -> str:
    """Instrucción al cambiar de tarea: la del entrenamiento si el modelo es de esa tarea; si
    no, la de scenes.yaml."""
    info = model_info(resolve_checkpoint(model, ckpt))
    if info.get("task") and _scene_for_task(info["task"]) == scene:
        return info["task"]
    return SCENES[scene][1] if scene in SCENES else ""


# ── Estado de la evaluación (current.json + archivo de rc) ──────────────────

def _read_current() -> dict | None:
    try:
        return json.loads(CURRENT_FILE.read_text())
    except Exception:  # noqa: BLE001
        return None


def _write_current(run: dict) -> None:
    EVAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CURRENT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(run, indent=2))
    tmp.replace(CURRENT_FILE)


def _finalize(run: dict) -> dict:
    """Evaluación sin proceso vivo: fija su estado final una sola vez (los resultados quedan)."""
    if run.get("status") not in ("running", "stopping"):
        return run
    rc = _rc(run)
    if run.get("status") == "stopping":
        run["status"] = "stopped"
    elif rc == 0:
        run["status"] = "done"
    else:
        run["status"] = "failed"
        run["rc"] = rc
    run["ended"] = time.time()
    _write_current(run)
    return run


def current_run() -> dict | None:
    with _lock:
        run = _read_current()
        if run and run.get("status") in ("running", "stopping") and (
                _rc(run) is not None or not _group_alive(run["pgid"])):
            run = _finalize(run)
        return run


def clear_finished() -> None:
    """Al abrir el Hub la pestaña parte en "Waiting to start" (como Train): solo se retoma una
    evaluación en curso. Los resultados de las anteriores quedan en su carpeta."""
    run = current_run()
    if run and run.get("status") not in ("running", "stopping"):
        with _lock:
            CURRENT_FILE.unlink(missing_ok=True)


def is_active() -> bool:
    run = current_run()
    return bool(run and run.get("status") in ("running", "stopping"))


# ── Lanzar ────────────────────────────────────────────────────────────────────

def build_command(p: dict) -> list[str]:
    argv = [
        sys.executable, "-u", str(EVAL_SCRIPT),
        "--checkpoint", p["checkpoint"],
        "--backend", p["backend"],
        "--scene", p["scene"],
        "--task", p["task"],
        *solver_argv(p["solver"]),
        "--num-episodes", str(p["num_episodes"]),
        "--episode-len", str(p["episode_len"]),
        "--fps", str(p["fps"]),
        "--out", p["output_dir"],
    ]
    if p["n_action_steps"] is not None:
        argv += ["--n-action-steps", str(p["n_action_steps"])]
    argv += SIDES.get(p["side"], SIDES[SIDE_ALTERNATE])
    if p["backend"] == "mujoco" and p["seed"] is not None:
        argv += ["--seed", str(p["seed"])]
    if not p["viewer"]:
        argv.append("--no-viewer")
    if not p["videos"]:
        argv.append("--no-video")
    if p["backend"] == "gazebo":
        # Simulación + módulo YARP + evaluación en un solo proceso (eval_gazebo_stack.py)
        argv = [sys.executable, "-u", str(GAZEBO_STACK), "--arms", p["arms"], *solver_argv(p["solver"]),
                "--module-log", p["module_log"], "--scene", p["scene"], "--", *argv[3:]]
    return argv


def start_evaluation(model, ckpt, backend_label, scene, task, num_episodes, episode_len,
                     viewer, videos, n_action_steps, side, seed, fps, results_dir):
    """Devuelve el mensaje para el toast."""
    if is_active():
        return "An evaluation is already in progress."
    if training.is_active():
        return "A training run is in progress: the GPU is busy until it ends."
    if state.running:
        return "Stop the recording session first (Episode Control)."
    if state.curation_running or state.push_running:
        return "Wait for the dataset packaging / upload to finish."
    checkpoint = resolve_checkpoint(model, ckpt)
    if checkpoint is None:
        return "Select a trained model (run folder with checkpoints/) or type a Hugging Face model ID."
    backend = BACKENDS.get(backend_label)
    if backend is None:
        return "Choose the simulator."
    if scene not in SCENES:
        return "Choose the task."
    task = (task or "").strip() or SCENES[scene][1]
    # El controlador no se elige: es el del dataset del modelo (en Gazebo lo usa el módulo YARP)
    solver = model_solver(checkpoint)
    if solver is None:
        return NO_SOLVER_MSG
    try:
        p = dict(
            checkpoint=checkpoint, backend=backend, scene=scene, task=task, solver=solver,
            num_episodes=_int(num_episodes, "Episodes"), episode_len=_int(episode_len, "Max steps per episode"),
            fps=_int(fps, "Control rate"), viewer=bool(viewer) and bool(os.environ.get("DISPLAY")),
            videos=bool(videos), side=side,
            n_action_steps=None if n_action_steps in (None, "") else _int(n_action_steps, "Action steps per inference"),
            seed=None if seed in (None, "") else _int(seed, "Seed", 0),
        )
    except ValueError as e:
        return str(e)
    info = model_info(checkpoint)
    if p["n_action_steps"] is not None and info.get("chunk_size") and p["n_action_steps"] > info["chunk_size"]:
        return f"Action steps per inference can't be larger than the model's chunk size ({info['chunk_size']})."
    if not EVAL_SCRIPT.exists():
        return f"Not found: {EVAL_SCRIPT}"
    if backend == "gazebo":
        if not GAZEBO_STACK.exists():
            return f"Not found: {GAZEBO_STACK}"
        # El módulo YARP lo levanta la evaluación (con los brazos del modelo): el del Hub, si
        # quedó corriendo desde Configuration, se detiene (eval_gazebo_stack.py detiene los demás)
        if procman.is_running("yarp_module"):
            hub_log("[Hub] Evaluate: stopping the YARP module (the evaluation starts its own).")
            procman.stop("yarp_module")
        p["arms"] = info.get("arms") or "both"

    name = _model_name(model.strip())
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{name}_{backend}_{stamp}"
    out = Path(results_dir or DEFAULT_RESULTS_DIR).expanduser() / run_name
    p["output_dir"] = str(out)

    EVAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = EVAL_LOG_DIR / f"{run_name}.log"
    rc_file = EVAL_LOG_DIR / f"{run_name}.rc"
    p["module_log"] = str(EVAL_LOG_DIR / f"{run_name}_yarp_module.log")
    argv = build_command(p)
    wrapper = ["sh", "-c", '"$@"; echo $? > "$0"', str(rc_file), *argv]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with _lock:
        with open(log_file, "w") as lf:
            lf.write("$ " + shlex.join(argv) + "\n\n")
            lf.flush()
            try:
                proc = subprocess.Popen(wrapper, cwd=str(REPO_ROOT), env=env, stdin=subprocess.DEVNULL,
                                        stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            except Exception as e:  # noqa: BLE001
                return f"Could not launch the evaluation: {e}"
        _write_current(dict(
            status="running", pid=proc.pid, pgid=proc.pid, started=time.time(),
            log_file=str(log_file), rc_file=str(rc_file), output_dir=str(out), run_name=run_name,
            model=model.strip(), checkpoint=checkpoint, backend=backend, scene=scene, task=task,
            solver=solver, num_episodes=p["num_episodes"], episode_len=p["episode_len"],
        ))
    threading.Thread(target=proc.wait, daemon=True).start()
    return f"Evaluation started: {run_name}."


# ── Detener ───────────────────────────────────────────────────────────────────

def stop_evaluation():
    """SIGINT al grupo (eval_policy.py cierra videos y robot en su finally; en Gazebo,
    eval_gazebo_stack.py cierra el módulo YARP y la simulación que levantó); SIGTERM a los 45 s y
    SIGKILL a los 55 s si sigue vivo. Los episodios terminados quedan en summary.json."""
    with _lock:
        run = _read_current()
        if not run or run.get("status") != "running":
            return "No evaluation in progress."
        run["status"] = "stopping"
        _write_current(run)
    pgid = run["pgid"]
    t0 = time.time()
    for sig, until in ((signal.SIGINT, 45), (signal.SIGTERM, 55), (signal.SIGKILL, 65)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            break
        while _group_alive(pgid) and time.time() - t0 < until:
            time.sleep(0.3)
        if not _group_alive(pgid):
            break
    run = current_run() or run
    done = (summary(run) or {}).get("n_episodes_completed", 0)
    return f"Evaluation stopped after {done} episode(s). Results kept in {run['output_dir']}."


# ── Progreso (log + summary.json) ────────────────────────────────────────────

_EP_RE = re.compile(r"^--- Episode (\d+)/(\d+) ---", re.M)
_STEP_RE = re.compile(r"^\s+step\s+(\d+)/(\d+)", re.M)
_STACK_RE = re.compile(r"^\[stack\] (.+)$", re.M)


def summary(run: dict) -> dict | None:
    """summary.json de eval_policy.py (se reescribe después de cada episodio)."""
    try:
        return json.loads((Path(run["output_dir"]) / "summary.json").read_text())
    except Exception:  # noqa: BLE001
        return None


def progress(run: dict) -> dict:
    text = "\n".join(_log_lines(_tail(run["log_file"])))
    eps = _EP_RE.findall(text)
    steps = _STEP_RE.findall(text[text.rfind("--- Episode"):]) if eps else []
    stack = _STACK_RE.findall(text)
    out = dict(episode=int(eps[-1][0]) if eps else 0, total=run.get("num_episodes", 0),
               step=int(steps[-1][0]) if steps else 0, episode_len=run.get("episode_len", 0),
               elapsed=(run.get("ended") or time.time()) - run["started"],
               stage=stack[-1] if stack else None)  # Gazebo: paso del arranque del stack
    return out


def _results_line(s: dict | None) -> str:
    if not s:
        return ""
    r = s.get("results", {})
    n = s.get("n_episodes_completed", 0)
    if r.get("success_rate_pct") is None:
        return f"{n} episode(s) finished (success not measured for this task)"
    line = f"<b>Success {r['n_success']} / {n} ({r['success_rate_pct']}%)</b>"
    if r.get("success_at_end_rate_pct") is not None:
        line += f" · at the end {r['success_at_end_rate_pct']}%"
    sides = r.get("success_rate_by_side") or r.get("success_rate_by_blue_side") or {}
    for sd, v in sides.items():
        line += f" · {html.escape(sd)} {v['success_rate_pct']}% ({v['episodes']})"
    if r.get("avg_success_time_s") is not None:
        line += f" · avg time to success {r['avg_success_time_s']} s"
    return line


def status_html(run: dict | None) -> str:
    if run is None:
        return ('<div class="rec-state"><div class="rec-state-title"><span class="rec-dot"></span>Waiting to start</div>'
                '<div class="rec-state-hint">Choose a model and a task, check the settings and press '
                '<b>Start evaluation</b>.</div></div>')
    st = run["status"]
    pr = progress(run)
    s = summary(run)
    name = html.escape(run["run_name"])
    res = _results_line(s)
    if st in ("running", "stopping"):
        cls, title = ("rec-rec", "Evaluating") if st == "running" else ("rec-warn", "Stopping")
        if pr["episode"] == 0 and st == "running":
            hint = (html.escape(pr["stage"]) if pr["stage"] and pr["stage"] != "YARP module ready"
                    else "Loading the policy and the scene…")
            if not Path(run["checkpoint"]).exists():
                hint += " (the Hugging Face model is downloaded first)"
            hint += f" · {_fmt_dur(pr['elapsed'])}"
        else:
            hint = (f"Episode {pr['episode']} / {pr['total']} · step {pr['step']} / {pr['episode_len']}"
                    f" · elapsed {_fmt_dur(pr['elapsed'])}")
            if res:
                hint += f"<br>{res}"
        hint += "<br><b>The rest of the Hub is locked until the evaluation ends.</b>"
    elif st == "done":
        cls, title = "rec-ok", "Finished"
        hint = (res or "No results") + f" · {_fmt_dur(pr['elapsed'])}"
    elif st == "stopped":
        cls, title = "rec-warn", "Stopped"
        hint = res or "Stopped before the first episode ended."
    else:
        cls, title = "rec-err", "Failed"
        hint = f"Exit code {run.get('rc')}. See the evaluation log below." + (f"<br>{res}" if res else "")
    if st != "running" or pr["episode"]:
        hint += f"<br>Results: <code>{html.escape(run['output_dir'])}</code>"
    done = (s or {}).get("n_episodes_completed", 0)
    total = pr["total"] or 1
    frac = 1.0 if st == "done" else min(1.0, (done + (pr["step"] / pr["episode_len"] if pr["episode_len"]
                                                       and pr["episode"] > done else 0)) / total)
    return (f'<div class="rec-state {cls}"><div class="rec-state-title"><span class="rec-dot"></span>'
            f'{title}</div><div class="rec-run-name">{name}</div>'
            f'<div class="rec-state-hint">{hint}</div></div>'
            f'<div class="rec-progress" style="margin-top:10px"><div class="rec-bar">'
            f'<div class="rec-fill" style="width:{int(100 * frac)}%"></div></div></div>')


def episodes_table(run: dict | None) -> str:
    """Tabla por episodio (Markdown) desde summary.json."""
    s = summary(run) if run else None
    eps = (s or {}).get("episodes") or []
    if not eps:
        return "*No finished episodes yet.*"
    has_side = any(e.get("side") or e.get("blue_side") for e in eps)
    head = "| Episode |" + (" Side |" if has_side else "") + " Success | At the end | Time to success | Duration |"
    rows = [head, "|---" * (head.count("|") - 1) + "|"]
    yn = {True: "✅", False: "❌", None: "—"}
    for e in eps:
        t = f"{e['success_time_s']} s" if e.get("success_time_s") is not None else "—"
        row = f"| {e['episode']} |"
        if has_side:
            row += f" {e.get('side') or e.get('blue_side') or '—'} |"
        row += f" {yn[e.get('success')]} | {yn[e.get('success_at_end')]} | {t} | {e.get('wall_time_s', '—')} s |"
        rows.append(row)
    return "\n".join(rows)


def log_text(run: dict | None, n: int = 200) -> str:
    if run is None:
        return ""
    lines = [ln for ln in _log_lines(_tail(run["log_file"], 262144)) if ln.strip()]
    return "\n".join(lines[-n:])
