"""Pestana 'Train': entrena SmolVLA con lerobot-train sobre un dataset grabado.

El entrenamiento corre desacoplado del Hub (sesión propia, salida a un archivo de log): si se
cierra el Hub sigue entrenando, y al volver a abrirlo se detecta por logs/training/current.json.
Mientras entrena, el resto del Hub queda bloqueado (app.py) porque usa casi toda la GPU
(~19 de 32 GB con batch 64 en la RTX 5090).

Las claves de W&B y Hugging Face las pone el usuario en la pestaña: van al proceso como
variables de entorno (WANDB_API_KEY, HF_TOKEN), nunca en la línea de comandos ni en disco.

Si el entrenamiento se detiene o falla, la carpeta de la corrida se manda a la papelera
(incompleta no sirve); el log queda en logs/training/.
"""

from __future__ import annotations

import html
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import gradio as gr

from . import state
from .curation import _move_to_trash
from .paths import HUB_ROOT, REPO_ROOT

TRAIN_LOG_DIR = HUB_ROOT / "logs" / "training"
CURRENT_FILE = TRAIN_LOG_DIR / "current.json"
DEFAULT_RUNS_DIR = REPO_ROOT / "output" / "train"
DEFAULT_HF_ORG = "CAMADA-UTFSM"

# Valores por defecto para la RTX 5090 (32 GB) = receta de fine-tuning de SmolVLA (batch 64,
# 20k steps), que no cabía en la RTX 3070 de 8 GB (batch 22, ~4.8 h). Medido en la 5090
# (TwoArmPegInHole_v2, 3 cámaras): el throughput está limitado por la GPU en ~112 muestras/s
# con cualquier batch (los dataloaders no frenan), así que batch 64 -> ~0.57 s/step, ~3.2 h
# para 20k steps, pico ~19 GB. Con batch 22 serían ~65 min (mismas muestras/s, 1/3 de los datos).
# No sirven en este commit de lerobot: --accelerator.mixed_precision=bf16 (~6% más lento: el
# VLM ya está en bf16) ni --policy.compile_model (loss NaN). --policy.use_amp no cambia el
# entrenamiento (lerobot solo lo usa al inferir).
DEFAULTS = dict(
    steps=20000, batch_size=64, save_freq=20000,
    vlm_model="HuggingFaceTB/SmolVLM2-500M-Video-Instruct", load_vlm_weights=True, use_amp=True,
    chunk_size=60, n_action_steps=60, lr=1e-4, warmup_steps=1000, decay_steps=30000,
    freeze_vision=True, expert_only=True, num_workers=min(32, os.cpu_count() or 4),
    log_freq=30, seed=1000,
)

_lock = threading.Lock()
_proc: subprocess.Popen | None = None  # solo si lo lanzó este proceso del Hub


# ── Datasets y nombres ────────────────────────────────────────────────────────

def dataset_info(dataset_path: str | None) -> dict | None:
    if not dataset_path:
        return None
    try:
        return json.loads((Path(dataset_path) / "meta" / "info.json").read_text())
    except Exception:  # noqa: BLE001
        return None


def _dataset_task(path: Path) -> str:
    try:
        import pandas as pd
        tasks = pd.read_parquet(path / "meta" / "tasks.parquet")
        return str(tasks.index[0]) if len(tasks) else ""
    except Exception:  # noqa: BLE001
        return ""


def base_name(dataset_path: str) -> str:
    """local_TwoArmPegInHole_v2_2026-10-29_16-32-22 -> TwoArmPegInHole_v2"""
    m = re.match(r"^(?:local_)?(.+?)(?:_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})?$", Path(dataset_path).name)
    return m.group(1) if m else Path(dataset_path).name


def _free_run_name(runs_dir: Path, name: str) -> str:
    """lerobot-train no arranca si la carpeta de salida ya existe: _2, _3, ..."""
    if not (runs_dir / name).exists():
        return name
    i = 2
    while (runs_dir / f"{name}_{i}").exists():
        i += 1
    return f"{name}_{i}"


def on_dataset_change(dataset_path: str | None, runs_dir: str):
    """Al elegir dataset: resumen y nombres sugeridos (editables). Orden de salida: resumen,
    run name, job name, W&B project, HF repo, max state dim, max action dim."""
    info = dataset_info(dataset_path)
    if info is None:
        return ("Select a dataset.",) + (gr.update(),) * 6
    feats = info.get("features", {})
    state_dim = int(feats.get("observation.state", {}).get("shape", [0])[0])
    action_dim = int(feats.get("action", {}).get("shape", [0])[0])
    cams = [k.removeprefix("observation.images.") for k in feats if k.startswith("observation.images.")]
    task = _dataset_task(Path(dataset_path))
    summary = (f"**{info.get('total_episodes', '?')} episodes** · {info.get('total_frames', '?')} frames · "
               f"{info.get('fps', '?')} fps · action {action_dim} · state {state_dim} · "
               f"cameras: {', '.join(cams) or 'none'}")
    if task:
        summary += f"  \nTask: *{html.escape(task)}*"
    if not (Path(dataset_path) / "meta" / "control_config.json").is_file():
        summary += "  \n⚠️ **No controller metadata** (meta/control_config.json): training will not start."
    name = base_name(dataset_path)
    job = f"icub_smolvla_{name.lower()}"
    run = _free_run_name(Path(runs_dir or DEFAULT_RUNS_DIR).expanduser(), f"{name}_smolvla")
    return (summary, run, job, job, f"{DEFAULT_HF_ORG}/{job}",
            max(32, state_dim), max(32, action_dim))


# ── Estado de la corrida (current.json + archivo de rc) ──────────────────────

def _read_current() -> dict | None:
    try:
        return json.loads(CURRENT_FILE.read_text())
    except Exception:  # noqa: BLE001
        return None


def _write_current(run: dict) -> None:
    TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CURRENT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(run, indent=2))
    tmp.replace(CURRENT_FILE)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _rc(run: dict) -> int | None:
    try:
        return int(Path(run["rc_file"]).read_text().strip())
    except Exception:  # noqa: BLE001
        return None


def _finalize(run: dict) -> dict:
    """Corrida sin proceso vivo: fija su estado final una sola vez (y si falló, manda la
    carpeta a la papelera)."""
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
    if run["status"] != "done":
        run["trash"] = _trash_run_dir(run)
    _write_current(run)
    return run


def _trash_run_dir(run: dict) -> str:
    out = Path(run["output_dir"])
    if not out.exists():
        return "no run folder was created"
    try:
        return f"run folder {_move_to_trash(out)}"
    except Exception as e:  # noqa: BLE001
        return f"could not delete the run folder {out}: {e}"


def current_run() -> dict | None:
    """La última corrida (viva o terminada), con su estado al día."""
    with _lock:
        run = _read_current()
        # Con rc_file escrito terminó, aunque quede algo vivo en el grupo (p. ej. wandb-core)
        if run and run.get("status") in ("running", "stopping") and (
                _rc(run) is not None or not _group_alive(run["pgid"])):
            run = _finalize(run)
        return run


def clear_finished() -> None:
    """Al abrir el Hub la pestaña parte en "Waiting to start": una corrida terminada, detenida o fallida no
    se sigue mostrando (su log queda en logs/training/). Solo una en curso se retoma."""
    run = current_run()
    if run and run.get("status") not in ("running", "stopping"):
        with _lock:
            CURRENT_FILE.unlink(missing_ok=True)


def is_active() -> bool:
    run = current_run()
    return bool(run and run.get("status") in ("running", "stopping"))


# ── Lanzar ────────────────────────────────────────────────────────────────────

def _int(v, name: str, minimum: int = 1) -> int:
    try:
        n = int(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a whole number.") from None
    if n < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return n


def _float(v, name: str) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number.") from None
    if x <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return x


def build_command(p: dict) -> list[str]:
    """argv de lerobot-train (el mismo comando que training_smolvla.ipynb)."""
    b = lambda v: "true" if v else "false"  # noqa: E731
    argv = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        "--policy.type=smolvla",
        f"--policy.vlm_model_name={p['vlm_model']}",
        f"--policy.load_vlm_weights={b(p['load_vlm_weights'])}",
        f"--policy.use_amp={b(p['use_amp'])}",
        f"--policy.chunk_size={p['chunk_size']}",
        f"--policy.n_action_steps={p['n_action_steps']}",
        f"--policy.max_state_dim={p['max_state_dim']}",
        f"--policy.max_action_dim={p['max_action_dim']}",
        f"--policy.optimizer_lr={p['lr']}",
        f"--policy.scheduler_warmup_steps={p['warmup_steps']}",
        f"--policy.scheduler_decay_steps={p['decay_steps']}",
        f"--policy.freeze_vision_encoder={b(p['freeze_vision'])}",
        f"--policy.train_expert_only={b(p['expert_only'])}",
        "--policy.device=cuda",
        f"--policy.push_to_hub={b(p['hf_push'])}",
        f"--dataset.repo_id=local/{Path(p['dataset']).name}",
        f"--dataset.root={p['dataset']}",
        f"--batch_size={p['batch_size']}",
        f"--num_workers={p['num_workers']}",
        f"--steps={p['steps']}",
        f"--save_freq={p['save_freq']}",
        f"--log_freq={p['log_freq']}",
        f"--seed={p['seed']}",
        f"--output_dir={p['output_dir']}",
        f"--job_name={p['job_name']}",
        f"--wandb.enable={b(p['wandb'])}",
    ]
    if p["wandb"]:
        argv.append(f"--wandb.project={p['wandb_project']}")
    if p["hf_push"]:
        argv += [f"--policy.repo_id={p['hf_repo']}", f"--policy.private={b(p['hf_private'])}"]
    return argv


def _check_hf(token: str, repo_id: str) -> str | None:
    """El modelo se sube al final (horas después): token y namespace se revisan antes."""
    if "/" not in repo_id or repo_id.startswith("local/"):
        return "Hugging Face repo ID must be user_or_org/model_name."
    try:
        from huggingface_hub import HfApi, get_token
        tok = token or get_token()
        if not tok:
            return "Enter your Hugging Face token to push the model."
        me = HfApi().whoami(token=tok)
    except Exception as e:  # noqa: BLE001
        return f"Hugging Face token check failed: {e}"
    names = {me.get("name")} | {o.get("name") for o in me.get("orgs", [])}
    ns = repo_id.split("/", 1)[0]
    if ns not in names:
        return f"'{ns}' is not your Hugging Face user or one of your organizations ({', '.join(sorted(n for n in names if n))})."
    return None


def _wandb_logged_in() -> bool:
    netrc = Path.home() / ".netrc"
    return bool(os.environ.get("WANDB_API_KEY")) or (netrc.exists() and "api.wandb.ai" in netrc.read_text())


def start_training(dataset, runs_dir, run_name, job_name, steps, batch_size, save_freq,
                   wandb_on, wandb_key, wandb_project, hf_on, hf_token, hf_repo, hf_private,
                   vlm_model, load_vlm_weights, use_amp, chunk_size, n_action_steps,
                   max_state_dim, max_action_dim, lr, warmup_steps, decay_steps,
                   freeze_vision, expert_only, num_workers, log_freq, seed):
    """Devuelve el mensaje para el campo de estado."""
    global _proc
    if is_active():
        return "A training run is already in progress."
    from . import evaluation  # import local: evaluation.py importa este módulo
    if evaluation.is_active():
        return "An evaluation is in progress: wait for it to finish or stop it (Evaluate)."
    if state.running:
        return "Stop the recording session first (Episode Control): training needs the whole GPU."
    if state.curation_running or state.push_running:
        return "Wait for the dataset packaging / upload to finish."
    if dataset_info(dataset) is None:
        return "Select a valid dataset (folder with meta/info.json)."
    if not (Path(dataset) / "meta" / "control_config.json").is_file():
        return ("This dataset has no meta/control_config.json (controller metadata), so the model "
                "could not be evaluated. Record it again with the current version.")
    run_name = (run_name or "").strip()
    job_name = (job_name or "").strip() or run_name
    if not run_name or not re.fullmatch(r"[\w.\-]+", run_name):
        return "Run name: use letters, numbers, '_', '-' or '.' (no spaces)."
    runs = Path(runs_dir or DEFAULT_RUNS_DIR).expanduser()
    out = runs / run_name
    if out.exists():
        return f"{out} already exists: choose another run name."
    try:
        p = dict(
            dataset=str(Path(dataset).resolve()), output_dir=str(out), job_name=job_name,
            steps=_int(steps, "Steps"), batch_size=_int(batch_size, "Batch size"),
            save_freq=_int(save_freq, "Save checkpoint every"),
            vlm_model=(vlm_model or DEFAULTS["vlm_model"]).strip(),
            load_vlm_weights=bool(load_vlm_weights), use_amp=bool(use_amp),
            chunk_size=_int(chunk_size, "Chunk size"), n_action_steps=_int(n_action_steps, "Action steps"),
            max_state_dim=_int(max_state_dim, "Max state dim"), max_action_dim=_int(max_action_dim, "Max action dim"),
            lr=_float(lr, "Learning rate"), warmup_steps=_int(warmup_steps, "Warmup steps", 0),
            decay_steps=_int(decay_steps, "Decay steps"),
            freeze_vision=bool(freeze_vision), expert_only=bool(expert_only),
            num_workers=_int(num_workers, "Dataloader workers", 0), log_freq=_int(log_freq, "Log every"),
            seed=_int(seed, "Seed", 0),
            wandb=bool(wandb_on), wandb_project=(wandb_project or "").strip() or job_name,
            hf_push=bool(hf_on), hf_repo=(hf_repo or "").strip(), hf_private=bool(hf_private),
        )
    except ValueError as e:
        return str(e)
    if p["n_action_steps"] > p["chunk_size"]:
        return "Action steps per inference can't be larger than the chunk size."
    feats = dataset_info(dataset)["features"]
    for key, dim in (("observation.state", p["max_state_dim"]), ("action", p["max_action_dim"])):
        n = int(feats.get(key, {}).get("shape", [0])[0])
        if dim < n:
            return f"Max {key.split('.')[-1]} dim ({dim}) is smaller than the dataset's {key} ({n})."

    env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PYTHONUNBUFFERED": "1"}
    if p["wandb"]:
        if (wandb_key or "").strip():
            env["WANDB_API_KEY"] = wandb_key.strip()
        elif not _wandb_logged_in():
            return "Enter your W&B API key (or turn off W&B logging)."
    if p["hf_push"]:
        err = _check_hf((hf_token or "").strip(), p["hf_repo"])
        if err:
            return err
        if (hf_token or "").strip():
            env["HF_TOKEN"] = hf_token.strip()

    TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_file = TRAIN_LOG_DIR / f"{run_name}_{stamp}.log"
    rc_file = TRAIN_LOG_DIR / f"{run_name}_{stamp}.rc"
    argv = build_command(p)
    # sh deja el código de salida en rc_file aunque el Hub ya no esté (no puede hacer wait())
    wrapper = ["sh", "-c", '"$@"; echo $? > "$0"', str(rc_file), *argv]
    full_env = os.environ.copy()
    full_env.update(env)
    runs.mkdir(parents=True, exist_ok=True)
    with _lock:
        with open(log_file, "w") as lf:
            lf.write("$ " + " ".join(argv) + "\n\n")
            lf.flush()
            try:
                proc = subprocess.Popen(wrapper, cwd=str(REPO_ROOT), env=full_env, stdin=subprocess.DEVNULL,
                                        stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            except Exception as e:  # noqa: BLE001
                return f"Could not launch the training: {e}"
        _proc = proc
        _write_current(dict(
            status="running", pid=proc.pid, pgid=proc.pid, started=time.time(),
            log_file=str(log_file), rc_file=str(rc_file), output_dir=str(out),
            run_name=run_name, job_name=job_name, dataset=p["dataset"], steps=p["steps"],
            batch_size=p["batch_size"], wandb=p["wandb"], wandb_project=p["wandb_project"] if p["wandb"] else None,
            hf_repo=p["hf_repo"] if p["hf_push"] else None, hf_private=p["hf_private"],
        ))
    # Recoge el proceso (si no, queda zombie y _group_alive lo ve vivo)
    threading.Thread(target=proc.wait, daemon=True).start()
    return f"Training started: {run_name}."


# ── Detener ───────────────────────────────────────────────────────────────────

def stop_training():
    """Detiene la corrida (SIGTERM al grupo, SIGKILL a los 15 s) y manda su carpeta a la
    papelera. Devuelve el mensaje de estado."""
    with _lock:
        run = _read_current()
        if not run or run.get("status") != "running":
            return "No training in progress."
        run["status"] = "stopping"
        _write_current(run)
    pgid = run["pgid"]
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.time() + 15
    while _group_alive(pgid) and time.time() < deadline:
        time.sleep(0.3)
    if _group_alive(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        while _group_alive(pgid) and time.time() < deadline + 10:
            time.sleep(0.3)
    run = current_run() or run
    return f"Training stopped ({run.get('trash', 'run folder removed')})."


# ── Progreso (leído del log) ──────────────────────────────────────────────────

# Solo la barra "Training:" (antes salen otras, p. ej. "Loading weights: 489/489")
_TQDM_RE = re.compile(r"Training:[^|]*\|[^|]*\|\s*(\d+)/(\d+) \[([\d:]+)<([\d:?]+)")
_LOSS_RE = re.compile(r"step:\S+ .*?loss:([\d.]+)")


def _tail(path: str, nbytes: int = 65536) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - nbytes))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _log_lines(text: str) -> list[str]:
    """Líneas del log tal como se verían en una terminal (tqdm reescribe con \\r y los INFO
    de lerobot salen pegados a la barra)."""
    text = re.sub(r"(?<!\n)(?=INFO \d{4}-\d{2}-\d{2} )", "\n", text.replace("\r\n", "\n"))
    return [ln.rsplit("\r", 1)[-1] for ln in text.split("\n")]


def _fmt_dur(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}h {s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m {s % 60:02d}s"


def progress(run: dict) -> dict:
    text = _tail(run["log_file"])
    m_step = None
    for m_step in _TQDM_RE.finditer(text):
        pass
    losses = _LOSS_RE.findall(text)
    out = dict(step=0, total=run.get("steps", 0), loss=float(losses[-1]) if losses else None,
               eta=None, elapsed=time.time() - run["started"])
    if m_step:
        out["step"], out["total"] = int(m_step.group(1)), int(m_step.group(2))
        out["eta"] = m_step.group(4)
    if run.get("ended"):
        out["elapsed"] = run["ended"] - run["started"]
    return out


def _model_dir(run: dict) -> Path:
    return Path(run["output_dir"]) / "checkpoints" / "last" / "pretrained_model"


def status_html(run: dict | None) -> str:
    if run is None:
        return ('<div class="rec-state"><div class="rec-state-title"><span class="rec-dot"></span>Waiting to start</div>'
                '<div class="rec-state-hint">Choose a dataset, check the settings and press '
                '<b>Start training</b>.</div></div>')
    st = run["status"]
    pr = progress(run)
    name = html.escape(run["run_name"])
    if st in ("running", "stopping"):
        cls, title = ("rec-rec", f"Training — {name}") if st == "running" else ("rec-warn", f"Stopping — {name}")
        if pr["step"] == 0 and st == "running":
            hint = "Loading the dataset and the base model… (the first run downloads it)"
        else:
            hint = f"Step {pr['step']:,} / {pr['total']:,}"
            if pr["loss"] is not None:
                hint += f" · loss {pr['loss']:.3f}"
            hint += f" · elapsed {_fmt_dur(pr['elapsed'])}"
            if pr["eta"] and "?" not in pr["eta"]:
                hint += f" · remaining {pr['eta']}"
        hint += "<br><b>The rest of the Hub is locked until the training ends.</b>"
    elif st == "done":
        cls, title = "rec-ok", f"Finished — {name}"
        hint = (f"{pr['total']:,} steps in {_fmt_dur(pr['elapsed'])}"
                + (f" · final loss {pr['loss']:.3f}" if pr["loss"] is not None else "")
                + f"<br>Model: <code>{html.escape(str(_model_dir(run)))}</code>")
        if run.get("hf_repo"):
            vis = "private" if run.get("hf_private", True) else "public"
            hint += f"<br>Pushed to Hugging Face ({vis}): <code>{html.escape(run['hf_repo'])}</code>"
    elif st == "stopped":
        cls, title = "rec-warn", f"Stopped — {name}"
        hint = f"Stopped at step {pr['step']:,} / {pr['total']:,} ({html.escape(run.get('trash', ''))})."
    else:
        cls, title = "rec-err", f"Failed — {name}"
        hint = (f"Exit code {run.get('rc')}. See the training log below "
                f"({html.escape(run.get('trash', ''))}).")
    pct = 100 if st == "done" else int(100 * pr["step"] / pr["total"]) if pr["total"] else 0
    return (f'<div class="rec-state {cls}"><div class="rec-state-title"><span class="rec-dot"></span>'
            f'{title}</div><div class="rec-state-hint">{hint}</div></div>'
            f'<div class="rec-progress" style="margin-top:10px"><div class="rec-bar">'
            f'<div class="rec-fill" style="width:{pct}%"></div></div></div>')


def log_text(run: dict | None, n: int = 200) -> str:
    if run is None:
        return ""
    lines = [ln for ln in _log_lines(_tail(run["log_file"], 262144)) if ln.strip()]
    return "\n".join(lines[-n:])
