"""Puertos ZMQ que comparten las plataformas, y su revisión antes de lanzar una sesión.

MuJoCo y Gazebo/robot real usan los mismos puertos para el VR, pero desde procesos
distintos:

- 8115-8121 (entrada del VR: cabeza, manos, botones): el núcleo de MuJoCo
  (src/interfaces/teleop_mujoco.py) o el teleoperador de Gazebo/real, ambos dentro de
  play.py (subproceso de la sesión).
- 10505 / 15001 (streaming de imagen a los lentes): el núcleo de MuJoCo, o el módulo YARP
  (teleop_module_sm.py, proceso aparte).

Si una sesión anterior no se cerró bien, sus puertos siguen tomados y la sesión siguiente
(de cualquier plataforma) falla con "Address already in use". Por eso, antes de lanzar algo
se revisan los puertos que va a usar: los que retiene el propio Hub se liberan
(release_own); los de otro proceso se informan en vez de dejar que la sesión se caiga.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys

VR_PORTS = (8115, 8117, 8119, 8121)
STREAM_PORTS = (10505, 15001)


def listeners(ports) -> dict[int, int | None]:
    """{puerto: pid} de los puertos TCP en escucha (pid None si no se pudo saber)."""
    wanted = set(ports)
    found: dict[int, int | None] = {}
    try:
        out = subprocess.run(["ss", "-ltnpH"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        out = None
    if out is not None:
        for line in out.splitlines():
            cols = line.split()
            if len(cols) < 4:
                continue
            port = cols[3].rsplit(":", 1)[-1]
            if port.isdigit() and int(port) in wanted:
                pid = re.search(r"pid=(\d+)", line)
                found[int(port)] = int(pid.group(1)) if pid else None
        return found
    # Sin `ss` (p. ej. un contenedor mínimo): probar si se puede abrir el puerto
    for port in wanted:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
            except OSError:
                found[port] = None
    return found


def release_own() -> int:
    """Cierra los núcleos MuJoCo que sigan abiertos en el proceso del Hub."""
    core_mod = (
        sys.modules.get("interfaces.teleop_mujoco")
        or sys.modules.get("teleop_mujoco")
    )
    if core_mod is None or not hasattr(core_mod, "close_all_live"):
        return 0
    return core_mod.close_all_live()


def _describe(pid: int | None) -> str:
    if pid is None:
        return "another process"
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            cmd = fh.read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        cmd = "?"
    return f"PID {pid} ({cmd[:80]})"


def preflight(ports, label: str, log=print) -> str | None:
    """None si los puertos están libres (liberando antes los que retenga el Hub), o un
    mensaje para la UI con quién los tiene."""
    busy = listeners(ports)
    if any(pid == os.getpid() for pid in busy.values()):
        n = release_own()
        log(f"[Hub] Released {n} leftover MuJoCo session(s) holding ports "
            f"{sorted(p for p, pid in busy.items() if pid == os.getpid())}.")
        busy = listeners(ports)
    if not busy:
        return None
    owners = {}
    for port, pid in sorted(busy.items()):
        owners.setdefault(pid, []).append(port)
    detail = "; ".join(f"{ports_} held by {_describe(pid)}" for pid, ports_ in owners.items())
    return (f"Cannot start {label}: ports {detail}. Close that session/process "
            f"(another Hub, play.py, teleop_mujoco.py or a YARP module) and try again.")
