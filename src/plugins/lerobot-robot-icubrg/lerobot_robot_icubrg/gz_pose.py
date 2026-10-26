from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time


def _die_with_parent() -> None:
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").prctl(1, 15)
    except Exception:
        pass


def _find_gz() -> str | None:
    proj_root = Path(__file__).resolve().parents[3]
    pixi_candidate = proj_root / ".pixi" / "envs" / "default" / "bin" / "gz"
    if pixi_candidate.exists():
        return str(pixi_candidate)
    bimanual_candidate = proj_root.parent / "icub-bimanual" / ".pixi" / "envs" / "default" / "bin" / "gz"
    if bimanual_candidate.exists():
        return str(bimanual_candidate)
    prefix = os.environ.get("ICUBENV_PREFIX", str(Path.home() / "miniconda3" / "envs" / "icubenv"))
    conda_candidate = Path(prefix) / "bin" / "gz"
    if conda_candidate.exists():
        return str(conda_candidate)
    return shutil.which("gz")


class GzPoseReader:
    def __init__(self, bodies: list[str], root_xyz: tuple[float, float, float]):
        self.bodies = list(bodies)
        self.root_xyz = tuple(float(v) for v in root_xyz)
        self._latest: dict[str, tuple[list[float], list[float]]] = {}
        self._stamp: dict[str, float] = {}
        self._lock = threading.Lock()
        self._procs: list[subprocess.Popen] = []

    def start(self) -> None:
        gz = _find_gz()
        if gz is None:
            print("[WARN] 'gz' not found: object poses will be zero.")
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
            p, q = msg.get("position", {}), msg.get("orientation", {})
            pos = [p.get(ax, 0.0) - r for ax, r in zip("xyz", self.root_xyz)]
            quat = [q.get("w", 0.0), q.get("x", 0.0), q.get("y", 0.0), q.get("z", 0.0)]
            with self._lock:
                self._latest[body] = (pos, quat)
                self._stamp[body] = time.time()

    def get(self, body: str) -> tuple[list[float], list[float]] | None:
        with self._lock:
            return self._latest.get(body)

    def stamp(self, body: str) -> float | None:
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
    def __init__(self, keys: list[str], hold_s: float = 0.1):
        self.keys = list(keys)
        self.hold_s = hold_s
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()
        self._procs: list[subprocess.Popen] = []

    @staticmethod
    def topic(key: str) -> str:
        return f"/icub/touch/{key}"

    def start(self) -> list[str]:
        gz = _find_gz()
        if gz is None:
            print("[WARN] 'gz' not found: touch will be zero.")
            return [self.topic(k) for k in self.keys]
        try:
            listed = subprocess.run([gz, "topic", "-l"], capture_output=True, text=True, timeout=10).stdout.split()
        except Exception:
            listed = []
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
            if msg.get("contact"):
                with self._lock:
                    self._last[key] = time.time()

    def touching(self, key: str) -> bool:
        with self._lock:
            return time.time() - self._last.get(key, 0.0) < self.hold_s

    def stop(self) -> None:
        for proc in self._procs:
            proc.terminate()
        for proc in self._procs:
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._procs.clear()

