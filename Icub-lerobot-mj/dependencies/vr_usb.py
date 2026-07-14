"""Conexion del Meta Quest por cable USB (adb reverse).

La app BeaVR en el Quest necesita una IP a la que enviar los comandos.
En modo cable, en vez de la IP WiFi del PC, tunelizamos los puertos ZMQ de
VR (ver dependencies/teleop_mujoco.py) por el cable USB con `adb reverse`,
para que BeaVR pueda apuntar a 127.0.0.1 y todo viaje por USB.

Requiere: adb (Android Platform Tools) y el Quest con "Depuracion USB"
habilitada y autorizada.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Callable

# Mismos puertos que dependencies/teleop_mujoco.py
VR_PORTS = {
    "HEAD": 8115,
    "RIGHT_HAND": 8117,
    "BUTTONS": 8119,
    "LEFT_HAND": 8121,
}

# Puertos de streaming de imagen (ver dependencies/teleop_mujoco.py: head_cam=10505, viewer=15001).
# Tambien hay que tunelizarlos por USB o BeaVR nunca vera la camara aunque el
# PC este publicando video correctamente.
IMG_PORTS = {
    "HEAD_CAM": 10505,
    "VIEWER": 15001,
}

CABLE_IP = "127.0.0.1"


def _run_adb(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["adb", *args], capture_output=True, text=True)


def check_adb_device(log: Callable[[str], None] = print) -> bool:
    """Verifica que haya un Quest conectado por USB y autorizado."""
    if shutil.which("adb") is None:
        log("[USB] 'adb' no esta instalado o no esta en el PATH.")
        return False

    result = _run_adb(["devices"])
    lines = [l.strip() for l in result.stdout.splitlines()[1:] if l.strip()]

    if not lines:
        log("[USB] No se detecto ningun dispositivo. Conecta el Quest 2 por "
            "USB-C y habilita 'Depuracion USB' (Ajustes > Sistema > Opciones de desarrollador).")
        return False

    for line in lines:
        serial, _, state = line.partition("\t")
        if state == "unauthorized":
            log(f"[USB] Dispositivo {serial} conectado pero NO autorizado. "
                "Ponte el visor y acepta el popup 'Allow USB debugging'.")
            return False
        if state == "device":
            log(f"[USB] Dispositivo autorizado: {serial}")
            return True

    log(f"[USB] Estado de dispositivo desconocido: {lines}")
    return False


def setup_reverse_ports(ports: dict[str, int] = VR_PORTS,
                         log: Callable[[str], None] = print) -> bool:
    """Tunel-iza cada puerto VR del Quest hacia el mismo puerto en el PC."""
    ok = True
    for name, port in ports.items():
        result = _run_adb(["reverse", f"tcp:{port}", f"tcp:{port}"])
        if result.returncode != 0:
            log(f"[USB] adb reverse tcp:{port} fallo: {result.stderr.strip()}")
            ok = False
        else:
            log(f"[USB] adb reverse tcp:{port} -> tcp:{port}  ({name})")
    return ok


def remove_reverse_ports(ports: dict[str, int] = VR_PORTS) -> None:
    for port in ports.values():
        _run_adb(["reverse", "--remove", f"tcp:{port}"])


def connect_cable(log: Callable[[str], None] = print) -> bool:
    """Verifica el dispositivo y configura adb reverse para los puertos VR
    (control) y de imagen (camara).

    Devuelve True si el Quest queda listo para transmitir por cable
    (BeaVR debe apuntar su IP a CABLE_IP = 127.0.0.1).
    """
    if not check_adb_device(log):
        return False
    ok = setup_reverse_ports(VR_PORTS, log=log)
    ok = setup_reverse_ports(IMG_PORTS, log=log) and ok
    return ok
