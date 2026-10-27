"""Pose de los objetos de la escena leída desde Gazebo (ground truth, como obj{i}_* en MuJoCo).

Cada modelo publica su pose con el sistema PosePublisher (ver gazebo/models/<body>/model.sdf):
en /model/<body>/pose si viene del mundo, pero en /<body>/pose si lo recreó reset_objects.sh
(EntityFactory). Se escuchan ambos con `gz topic -e --json-output`, un hilo por tópico, para
no depender de los bindings Python de gz-transport (no están en icubenv), y vale el último
mensaje que llegó.

La pose se entrega en el frame root del iCub: en el mundo Gazebo el robot mira hacia -x y su
root (x hacia atrás, y a la derecha, z arriba) tiene la misma orientación que el mundo, así que
basta restar la posición del root_link (gazebo/worlds/icub_world.sdf).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path


def _die_with_parent() -> None:
    """preexec_fn: el `gz topic -e` muere si el proceso que lo lanzó se cae sin disconnect()."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").prctl(1, 15)  # PR_SET_PDEATHSIG, SIGTERM
    except Exception:
        pass


def _find_gz() -> str | None:
    """gz del entorno icubenv (ICUBENV_PREFIX en Docker) antes que el de PATH, como
    reset_objects.sh: /usr/bin/gz puede ser Gazebo Classic 11, que no ve gz-sim."""
    prefix = os.environ.get("ICUBENV_PREFIX", str(Path.home() / "miniconda3" / "envs" / "icubenv"))
    candidate = Path(prefix) / "bin" / "gz"
    return str(candidate) if candidate.exists() else shutil.which("gz")


class GzPoseReader:
    def __init__(self, bodies: list[str], root_xyz: tuple[float, float, float]):
        self.bodies = list(bodies)
        self.root_xyz = tuple(float(v) for v in root_xyz)
        self._latest: dict[str, tuple[list[float], list[float]]] = {}
        self._stamp: dict[str, float] = {}  # time.time() del último mensaje por cuerpo
        self._lock = threading.Lock()
        self._procs: list[subprocess.Popen] = []

    def start(self) -> None:
        gz = _find_gz()
        if gz is None:
            print("[WARN] 'gz' not found (PATH / ICUBENV_PREFIX): object poses will be zero.")
            return
        for body in self.bodies:
            for topic in (f"/model/{body}/pose", f"/{body}/pose"):
                proc = subprocess.Popen(
                    [gz, "topic", "-e", "-t", topic, "--json-output"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
                    preexec_fn=_die_with_parent,
                )
                self._procs.append(proc)
                threading.Thread(target=self._reader, args=(body, proc), daemon=True).start()

    def _reader(self, body: str, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            # protobuf→JSON omite los campos en cero
            p, q = msg.get("position", {}), msg.get("orientation", {})
            pos = [p.get(ax, 0.0) - r for ax, r in zip("xyz", self.root_xyz)]
            quat = [q.get("w", 0.0), q.get("x", 0.0), q.get("y", 0.0), q.get("z", 0.0)]
            with self._lock:
                self._latest[body] = (pos, quat)
                self._stamp[body] = time.time()

    def get(self, body: str) -> tuple[list[float], list[float]] | None:
        """(pos xyz, quat wxyz) en frame root, o None si aún no llegó ningún mensaje."""
        with self._lock:
            return self._latest.get(body)

    def stamp(self, body: str) -> float | None:
        """Instante (time.time()) del último mensaje recibido para body."""
        with self._lock:
            return self._stamp.get(body)

    def stop(self) -> None:
        for proc in self._procs:
            proc.terminate()
        for proc in self._procs:
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._procs.clear()


class GzContactReader:
    """Tacto binario por dedo desde los sensores de contacto de las puntas de los dedos.

    gazebo/models/iCubGazeboV2_5_visuomanip_camada/model.urdf publica gz.msgs.Contacts en
    /icub/touch/<side>_<finger> a 30 Hz solo mientras hay contacto; un dedo cuenta como
    tocando si llegó un mensaje con contactos hace menos de hold_s. `topics` cambia el tópico
    de una clave (p. ej. el sensor de models/red-cube, /red-cube/contact), y touching(key,
    model) dice además con qué modelo: los contactos traen las colisiones como
    "<modelo>::<link>::<colisión>".
    """

    def __init__(self, keys: list[str], hold_s: float = 0.1, topics: dict[str, str] | None = None):
        self.keys = list(keys)
        self.hold_s = hold_s
        self._topics = dict(topics or {})
        self._last: dict[str, float] = {}
        self._last_with: dict[str, dict[str, float]] = {}
        self._lock = threading.Lock()
        self._procs: list[subprocess.Popen] = []

    def topic(self, key: str) -> str:
        return self._topics.get(key, f"/icub/touch/{key}")

    def start(self) -> list[str]:
        """Arranca los lectores; devuelve los tópicos que Gazebo no anuncia (sin sensor)."""
        gz = _find_gz()
        if gz is None:
            print("[WARN] 'gz' not found (PATH / ICUBENV_PREFIX): touch will be zero.")
            return [self.topic(k) for k in self.keys]
        listed = subprocess.run([gz, "topic", "-l"], capture_output=True, text=True, timeout=10).stdout.split()
        for key in self.keys:
            proc = subprocess.Popen(
                [gz, "topic", "-e", "-t", self.topic(key), "--json-output"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
                    preexec_fn=_die_with_parent,
            )
            self._procs.append(proc)
            threading.Thread(target=self._reader, args=(key, proc), daemon=True).start()
        return [self.topic(k) for k in self.keys if self.topic(k) not in listed]

    def _reader(self, key: str, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            contacts = msg.get("contact")
            if contacts:
                now = time.time()
                models = {c.get(side, {}).get("name", "").split("::")[0]
                          for c in contacts for side in ("collision1", "collision2")}
                with self._lock:
                    self._last[key] = now
                    seen = self._last_with.setdefault(key, {})
                    for model in models:
                        seen[model] = now

    def touching(self, key: str, model: str | None = None) -> bool:
        """Contacto reciente del sensor `key` (con el modelo `model`, si se da)."""
        with self._lock:
            last = self._last.get(key, 0.0) if model is None else self._last_with.get(key, {}).get(model, 0.0)
            return time.time() - last < self.hold_s

    def stop(self) -> None:
        for proc in self._procs:
            proc.terminate()
        for proc in self._procs:
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._procs.clear()


class GzJointReader:
    """Ángulos de las articulaciones de un modelo de la escena (p. ej. door_hinge de la puerta),
    publicados por su JointStatePublisher en /world/<world>/model/<model>/joint_state. Mismo
    esquema que GzPoseReader: `gz topic -e --json-output` en un hilo; vale el último mensaje."""

    def __init__(self, model: str, world: str = "default"):
        self.topic = f"/world/{world}/model/{model}/joint_state"
        self._latest: dict[str, float] = {}
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None

    def start(self) -> None:
        gz = _find_gz()
        if gz is None:
            print("[WARN] 'gz' not found (PATH / ICUBENV_PREFIX): joint states unavailable.")
            return
        self._proc = subprocess.Popen(
            [gz, "topic", "-e", "-t", self.topic, "--json-output"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
            preexec_fn=_die_with_parent,
        )
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        for line in self._proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            # protobuf→JSON omite los campos en cero (position 0 = sin "position")
            pos = {j["name"]: float(j["axis1"].get("position", 0.0)) for j in msg.get("joint", [])
                   if "axis1" in j}
            with self._lock:
                self._latest = pos

    def get(self, joint: str) -> float | None:
        """Ángulo [rad] del último mensaje, o None si aún no llegó."""
        with self._lock:
            return self._latest.get(joint)

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
