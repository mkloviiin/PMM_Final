"""Stack Gazebo / robot real del Hub v2 (simulación o conexión, módulo YARP).

Misma lógica que los botones 2.1 / 2.2 del Hub original (robots/icub_rg.py, que los define
dentro de su formulario): aquí quedan como funciones sueltas para la UI nueva. Usa las mismas
rutas y helpers de icub_rg, y deja en icub_rg._module_arms / _module_solver los brazos y el
solver con que se arrancó el módulo, que su launch() revisa antes de grabar.
"""

from __future__ import annotations

import subprocess
import sys

from launcher_gui import ports, procman, session
from launcher_gui.robots import icub_rg
from launcher_gui.robots.solver import SOLVERS, solver_argv

from .status import light_html

P_MUJOCO = "MuJoCo simulation"
P_GAZEBO = "Gazebo simulation"
P_REAL = "Real robot"
PLATFORMS = [P_MUJOCO, P_GAZEBO, P_REAL]

_real_link: dict = {"ok": None, "msg": ""}  # último resultado de "Connect" (robot real)


def rg_mode(platform: str) -> str:
    return icub_rg.MODE_REAL if platform == P_REAL else icub_rg.MODE_GAZEBO


# ── Paso 1: simulación Gazebo / conexión con el robot real ───────────────────
def start_link(platform: str) -> str:
    if platform == P_REAL:
        ok, msg = icub_rg._connect_real_robot()
        _real_link.update(ok=ok, msg=msg)
        return msg
    if not icub_rg._SIM_START.exists():
        return f"Not found: {icub_rg._SIM_START}"
    if icub_rg._tmux_alive():
        return "The simulation is already running."
    procman.run_once(["bash", str(icub_rg._SIM_START)], cwd=str(icub_rg._SCRIPTS_DIR))
    return "Starting Gazebo (~15 s)…"


def stop_link(platform: str) -> str:
    if platform == P_REAL:
        _real_link.update(ok=None, msg="")
        return "Disconnected (nothing to stop on the robot side)."
    if not icub_rg._SIM_STOP.exists():
        return f"Not found: {icub_rg._SIM_STOP}"
    procman.run_once(["bash", str(icub_rg._SIM_STOP)], cwd=str(icub_rg._SCRIPTS_DIR))
    return "Gazebo stopped."


# ── Paso 2: módulo YARP (corre el solver whole-body) ─────────────────────────
def start_module(platform: str, arms: str, solver: str) -> str:
    module = icub_rg._YARP_MODULE
    if not module.exists():
        return f"Not found: {module}"
    if platform == P_REAL:
        ok, msg = icub_rg._connect_real_robot()  # por si se saltó el paso 1
        _real_link.update(ok=ok, msg=msg)
        if not ok:
            return msg
    robot_name = icub_rg._robot_name(rg_mode(platform))
    if not procman.is_running("yarp_module"):
        procman.stop_stale("yarp_module", str(module))
        busy = ports.preflight(ports.STREAM_PORTS, "the YARP module (VR streaming)", log=session.log)
        if busy:
            return busy
    if arms in ("right", "left", "both"):
        resolved, how = arms, "selected"
    else:
        resolved = icub_rg._detect_arms(robot_name)
        resolved, how = (resolved, "auto-detected") if resolved else (icub_rg._yaml_arms(), "from YAML")
    if procman.is_running("yarp_module") and icub_rg._module_solver not in (None, solver):
        return f"Running with '{icub_rg._module_solver}'. Stop it first to switch to '{solver}'."
    if platform == P_REAL and SOLVERS.get(solver, ("ik",))[0] == "id":
        session.log("[Hub] WARNING: the real robot will be driven in TORQUE mode (whole-body ID). "
                    "Keep the emergency stop at hand.")
    msg = procman.start(
        "yarp_module",
        [sys.executable, "-u", str(module), "--robot", robot_name, "--control", resolved, *solver_argv(solver)],
        cwd=str(module.parent),
        env={"ICUB_LEROBOT_CONFIG": str(icub_rg._CFG_PATH), "ICUB_ROBOT_NAME": robot_name},
    )
    icub_rg._module_arms = resolved
    icub_rg._module_solver = solver
    return f"{msg} Arms: {resolved} ({how})."


def stop_module() -> str:
    return procman.stop("yarp_module")


def _external_module() -> bool:
    """Módulo YARP corriendo pero no lanzado desde este Hub (p. ej. desde el Hub original)."""
    try:
        return subprocess.run(["pgrep", "-f", "teleop_module_sm.py"], capture_output=True).returncode == 0
    except Exception:  # noqa: BLE001
        return False


# ── Estado de los pasos (lo consulta un Timer) ───────────────────────────────
def link_ok(platform: str) -> bool:
    return icub_rg._tmux_alive() if platform == P_GAZEBO else bool(_real_link["ok"])


def module_ok() -> bool:
    return procman.is_running("yarp_module") or _external_module()


def link_light(platform: str) -> str:
    if platform == P_GAZEBO:
        return (light_html("ok", "Gazebo is running") if icub_rg._tmux_alive()
                else light_html("off", "Not running"))
    if _real_link["ok"] is None:
        return light_html("off", f"Not connected (name server {icub_rg.REAL_YARP_IP}:{icub_rg.REAL_YARP_PORT})")
    return light_html("ok" if _real_link["ok"] else "err", _real_link["msg"])


def module_light() -> str:
    if procman.is_running("yarp_module"):
        return light_html("ok", f"Running · {icub_rg._module_solver} · arms: {icub_rg._module_arms}")
    if _external_module():
        return light_html("warn", "Running, started outside this Hub (solver / arms unknown)")
    return light_html("off", "Not running")
