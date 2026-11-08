"""Evaluación en Gazebo con todo el stack, en un solo proceso (lo lanza evaluation.py).

Pestaña Evaluate del Hub: el usuario no tiene que levantar nada en Configuration. Este script
corre en el grupo de procesos de la evaluación (desacoplado del Hub, ver evaluation.py):
  1. Simulación Gazebo (gazebo_sim/scripts/start_sim.sh, sesión tmux icub_sim) si no está
     corriendo; espera a que estén los controlboards del iCub en YARP.
  2. Módulo YARP (src/interfaces/teleop_module_sm.py) con los brazos del modelo y el solver
     elegido. Si ya había uno (del Hub o de antes) se detiene: con otros brazos el estado que
     publica no calza con el de la política (p. ej. 75 valores = solo brazo derecho).
  3. eval_policy.py --backend gazebo (los argumentos después de `--`); resetea la escena él
     mismo en cada episodio (reset_objects.sh, con el lado de la tarea como en MuJoCo).
Al terminar (o al detenerla: SIGINT al grupo) cierra el módulo YARP y, si la simulación la
levantó él, también la simulación: deja todo como lo encontró.

    python eval_gazebo_stack.py --arms both --controller ik --module-log x.log -- <args de eval_policy.py>
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "gazebo_sim" / "scripts"
YARP_MODULE = REPO_ROOT / "src" / "interfaces" / "teleop_module_sm.py"
CFG_PATH = REPO_ROOT / "config" / "control_config.yaml"
EVAL_SCRIPT = REPO_ROOT / "eval_policy.py"
TMUX_SESSION = "icub_sim"
ROBOT = "icubSim"
SIM_TIMEOUT_S = 180
SPAWN_TIMEOUT_S = 120
MODULE_TIMEOUT_S = 120

# Controlboards del iCub en YARP (los abre el plugin de Gazebo al cargar el modelo)
_QUERY_PY = (
    "import sys, yarp\n"
    "yarp.Network.init()\n"
    "ok = yarp.Network.checkNetwork(1.0) and all(yarp.Network.queryName(n).isValid() for n in sys.argv[1:])\n"
    "print('READY' if ok else 'WAIT')\n"
)


# Sensores listos = la cámara entrega una imagen (Gazebo terminó de iniciar el render). Antes de
# eso, borrar un modelo del mundo (reset_objects.sh) hace caer al servidor (segfault).
_PROBE_CAM_PY = (
    "import sys, time, yarp\n"
    "yarp.Network.init()\n"
    "p = yarp.BufferedPortImageRgb(); p.open('/eval_stack/probe:i')\n"
    "img = None\n"
    "if yarp.Network.connect(sys.argv[1], '/eval_stack/probe:i'):\n"
    "    t = time.time()\n"
    "    while img is None and time.time() - t < 5:\n"
    "        img = p.read(False); time.sleep(0.1)\n"
    "p.close()\n"
    "print('READY' if img is not None else 'WAIT')\n"
)
SENSOR_SETTLE_S = 3.0


def say(msg: str) -> None:
    print(f"[stack] {msg}", flush=True)


def tmux_alive() -> bool:
    return subprocess.run(["tmux", "has-session", "-t", TMUX_SESSION],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def ports_ready(names: list[str]) -> bool:
    try:
        out = subprocess.run([sys.executable, "-c", _QUERY_PY, *names], capture_output=True,
                             text=True, timeout=15).stdout
    except subprocess.TimeoutExpired:
        return False
    return any(ln.strip() == "READY" for ln in out.splitlines())


def camera_ready(port: str) -> bool:
    try:
        out = subprocess.run([sys.executable, "-c", _PROBE_CAM_PY, port], capture_output=True,
                             text=True, timeout=20).stdout
    except subprocess.TimeoutExpired:
        return False
    return any(ln.strip() == "READY" for ln in out.splitlines())


def gz_server_alive() -> bool:
    return subprocess.run(["pgrep", "-f", "gz sim -s"], stdout=subprocess.DEVNULL).returncode == 0


def stop_group(proc: subprocess.Popen, timeout: float = 8.0) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def stop_leftover_modules() -> None:
    """Módulos YARP vivos (del Hub o de una sesión anterior): chocarían con los puertos."""
    out = subprocess.run(["pgrep", "-f", str(YARP_MODULE)], stdout=subprocess.PIPE, text=True).stdout
    pids = [int(p) for p in out.split() if p.isdigit() and int(p) != os.getpid()]
    if not pids:
        return
    say(f"Stopping the YARP module already running (pid {', '.join(map(str, pids))})")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    deadline = time.time() + 10
    while time.time() < deadline and any(_alive(p) for p in pids):
        time.sleep(0.3)
    for pid in pids:
        if _alive(pid):
            os.kill(pid, signal.SIGKILL)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> int:
    if "--" not in sys.argv:
        raise SystemExit("usage: eval_gazebo_stack.py [options] -- <eval_policy.py args>")
    split = sys.argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", choices=["right", "left", "both"], default="both")
    ap.add_argument("--controller", choices=["ik", "id"], default="ik")
    ap.add_argument("--impedance", action="store_true")
    ap.add_argument("--module-log", required=True)
    ap.add_argument("--scene", default=None, help="scenes.yaml label: spawned once Gazebo is up")
    args = ap.parse_args(sys.argv[1:split])
    eval_args = sys.argv[split + 1:]

    started_sim = False
    module = eval_proc = None
    try:
        # 1. Simulación
        if tmux_alive():
            say("Gazebo simulation already running: using it")
        else:
            say("Starting the Gazebo simulation…")
            subprocess.run(["bash", str(SCRIPTS / "start_sim.sh")], cwd=str(SCRIPTS), check=False)
            started_sim = True
        arms = {"right": ["right_arm"], "left": ["left_arm"], "both": ["right_arm", "left_arm"]}[args.arms]
        boards = [f"/{ROBOT}/{p}/state:o" for p in ("torso", *arms)]
        say("Waiting for Gazebo and the robot…")
        deadline = time.time() + SIM_TIMEOUT_S
        cam = f"/{ROBOT}/cam/left/rgbImage:o"
        while not (ports_ready(boards) and camera_ready(cam)):
            if time.time() > deadline:
                say(f"Gazebo did not come up in {SIM_TIMEOUT_S} s (robot {boards} / camera {cam}).")
                return 3
            if not tmux_alive() or (time.time() > deadline - SIM_TIMEOUT_S + 20 and not gz_server_alive()):
                say("The Gazebo server is not running (it closed or crashed while starting): "
                    "see /tmp/tmux_output_pipe_gz_server.")
                return 3
            time.sleep(2)
        time.sleep(SENSOR_SETTLE_S)

        # Los controlboards aparecen antes de que el mundo acepte crear modelos ("Service call
        # timed out"): se crea la escena hasta que Gazebo responda
        if args.scene:
            say("Waiting for Gazebo to spawn the scene…")
            scene_file = Path(args.module_log).with_name("gazebo_scene_warmup.txt")
            scene_file.write_text(args.scene)
            env = dict(os.environ, ICUB_SCENE_FILE=str(scene_file), ICUB_SCENE_SIDE="right")
            deadline = time.time() + SPAWN_TIMEOUT_S
            while True:
                out = subprocess.run(["bash", str(SCRIPTS / "reset_objects.sh")], env=env,
                                     capture_output=True, text=True, timeout=120)
                if out.returncode == 0 and "AVISO" not in out.stdout:
                    break
                if not gz_server_alive():
                    say("The Gazebo server crashed while spawning the scene: see /tmp/tmux_output_pipe_gz_server.")
                    return 3
                if time.time() > deadline:
                    say(f"Gazebo could not spawn the scene in {SPAWN_TIMEOUT_S} s:\n{out.stdout[-1500:]}")
                    return 3
                time.sleep(3)
            scene_file.unlink(missing_ok=True)

        # 2. Módulo YARP, con los brazos del modelo y el solver elegido
        stop_leftover_modules()
        solver = [f"--controller", args.controller] + (["--impedance"] if args.impedance else [])
        say(f"Starting the YARP module (arms: {args.arms}, controller: {args.controller.upper()}"
            f"{' + impedance' if args.impedance else ''})…")
        env = dict(os.environ, PYTHONUNBUFFERED="1", ICUB_LEROBOT_CONFIG=str(CFG_PATH), ICUB_ROBOT_NAME=ROBOT)
        # Plugins de YARP (remotecontrolboardremapper, ...): los define la activación de pixi; si el
        # Hub no se lanzó con `pixi run`, se toman del mismo entorno que este Python
        env.setdefault("YARP_DATA_DIRS", f"{sys.prefix}/share/iCub:{sys.prefix}/share/yarp")
        log_path = Path(args.module_log)
        with open(log_path, "w") as log:
            module = subprocess.Popen(
                [sys.executable, "-u", str(YARP_MODULE), "--robot", ROBOT, "--control", args.arms, *solver],
                cwd=str(YARP_MODULE.parent), env=env, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT)
        deadline = time.time() + MODULE_TIMEOUT_S
        while "Module ready!" not in log_path.read_text(errors="replace"):
            if module.poll() is not None:
                say(f"The YARP module exited (rc={module.returncode}). Its log: {log_path}")
                print(log_path.read_text(errors="replace")[-3000:], flush=True)
                return 4
            if time.time() > deadline:
                say(f"The YARP module was not ready in {MODULE_TIMEOUT_S} s. Its log: {log_path}")
                return 4
            time.sleep(0.5)
        say("YARP module ready")

        # 3. Evaluación
        eval_proc = subprocess.Popen([sys.executable, "-u", str(EVAL_SCRIPT), *eval_args], cwd=str(REPO_ROOT))
        while True:
            try:
                return eval_proc.wait()
            except KeyboardInterrupt:
                # Stop del Hub: el SIGINT le llega también a eval_policy.py (mismo grupo), que
                # cierra videos y robot; aquí solo se espera a que termine
                continue
    except KeyboardInterrupt:
        say("Stopped while starting")
        return 130
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # que un segundo SIGINT no corte la limpieza
        if eval_proc is not None and eval_proc.poll() is None:
            stop_group(eval_proc, timeout=15)
        if module is not None:
            say("Stopping the YARP module")
            stop_group(module)
        if started_sim:
            say("Closing the Gazebo simulation (it was started for this evaluation)")
            subprocess.run(["bash", str(SCRIPTS / "stop_sim.sh")], cwd=str(SCRIPTS),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


if __name__ == "__main__":
    sys.exit(main())
