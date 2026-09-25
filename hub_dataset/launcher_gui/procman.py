"""Gestor genérico de subprocesos externos del Hub.

No sabe nada de robots concretos: cada backend decide qué lanzar. Maneja dos
tipos de proceso que un backend puede necesitar levantar *fuera* del proceso
del Hub:

  * **Long-lived con nombre** (p. ej. el módulo YARP ``teleop_module_sm.py``):
    ``start``/``stop``/``is_running``, con un hilo que vuelca su stdout al log
    compartido (``session.log``, el mismo panel que ve el usuario).
  * **Fire-and-forget** (p. ej. ``start_sim.sh`` / ``stop_sim.sh``, que crean o
    cierran una sesión tmux y retornan enseguida): ``run_once`` corre el script,
    espera a que termine y loguea su salida.

Se apoya en ``session.log`` para no inventar otro panel de logs. Un proceso por
nombre; arrancar dos veces el mismo nombre no relanza si sigue vivo.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from pathlib import Path

from . import session  # reutiliza el panel de log compartido (session.log)

_procs: dict[str, subprocess.Popen] = {}
_lock = threading.Lock()


def _pump(name: str, proc: subprocess.Popen) -> None:
    """Vuelca stdout del proceso al log compartido, línea a línea."""
    try:
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            if line:
                session.log(f"[{name}] {line.rstrip()}")
    except Exception as e:  # noqa: BLE001
        session.log(f"[{name}] lector detenido: {e!r}")
    finally:
        session.log(f"[{name}] proceso terminado (rc={proc.poll()}).")


def is_running(name: str) -> bool:
    with _lock:
        proc = _procs.get(name)
    return proc is not None and proc.poll() is None


def start(name: str, argv, cwd=None, env=None) -> str:
    """Lanza un proceso long-lived con nombre. Si ya hay uno vivo, no relanza."""
    if is_running(name):
        return f"'{name}' ya está corriendo."
    full_env = os.environ.copy()
    full_env["PYTHONUNBUFFERED"] = "1"
    if env:
        full_env.update(env)
    try:
        proc = subprocess.Popen(
            [str(a) for a in argv],
            cwd=str(cwd) if cwd else None,
            env=full_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,  # grupo propio → stop limpio con señales
        )
    except Exception as e:  # noqa: BLE001
        session.log(f"[{name}] no se pudo lanzar: {e!r}")
        return f"Error al lanzar '{name}': {e}"
    with _lock:
        _procs[name] = proc
    threading.Thread(target=_pump, args=(name, proc), daemon=True).start()
    session.log(f"[{name}] lanzado (pid={proc.pid}): {' '.join(str(a) for a in argv)}")
    return f"'{name}' iniciado (pid={proc.pid})."


def stop(name: str, sig=signal.SIGINT, timeout: float = 8.0) -> str:
    """Detiene un proceso long-lived: señal al grupo, y SIGKILL si no muere."""
    with _lock:
        proc = _procs.get(name)
    if proc is None or proc.poll() is not None:
        return f"'{name}' no está corriendo."
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except Exception:  # noqa: BLE001
        proc.send_signal(sig)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            proc.kill()
    session.log(f"[{name}] detenido.")
    return f"'{name}' detenido."


def run_once(argv, cwd=None, env=None, timeout: float = 180.0) -> tuple[int, str]:
    """Ejecuta un script que retorna solo (start_sim.sh/stop_sim.sh) y loguea su salida."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    label = Path(str(argv[-1])).name if argv else "run_once"
    try:
        res = subprocess.run(
            [str(a) for a in argv],
            cwd=str(cwd) if cwd else None,
            env=full_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except Exception as e:  # noqa: BLE001
        session.log(f"[{label}] error: {e!r}")
        return 1, str(e)
    for line in (res.stdout or "").splitlines():
        session.log(f"[{label}] {line}")
    return res.returncode, res.stdout or ""
