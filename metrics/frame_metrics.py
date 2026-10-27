"""Métricas por frame del loop de teleoperación/grabación (MuJoCo, Gazebo y real).

Los loops de ``play_mujoco.py`` y ``play_gazebo.py`` llaman a este registrador en
cada iteración. Se escribe un archivo JSON Lines por sesión (una línea por frame,
más líneas de episodio y de sesión) en ``src/hub_dataset/logs/metrics/``: fuera del
dataset, para que no se suba a Hugging Face con él. ``metrics_report.py`` lo
convierte en tablas y figuras.

Por qué un archivo aparte y no el log: el jitter necesita cada muestra, no
promedios cada 5 s, y la columna ``timestamp`` del dataset es ``frame_index / fps``
(ideal, no medida), así que no sirve para verificar la sincronización.

Uso en un loop::

    fm = FrameMetrics.create(backend="gazebo", fps=30, name=dataset_root.name)
    fm.begin(recording=True, episode=ep)
    obs = robot.get_observation();  fm.mark("obs")
    act = teleop.get_action();      fm.mark("teleop")
    robot.send_action(act);         fm.mark("send")
    dataset.add_frame(frame);       fm.mark("record")
    fm.end(robot=robot, obs=obs, action=act)
    ...
    fm.end_episode(ep, saved=True, save_ms=...)
    fm.close()

El robot puede exponer ``robot.metrics_meta`` (dict) con datos propios del backend
(timestamps YARP de estado y cámaras, confirmaciones de comandos, tiempos de render
o de física). Se copia tal cual en la línea del frame.

``PMM_METRICS=0`` desactiva el registro; ``PMM_METRICS_DIR`` cambia la carpeta.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from collections import deque
from pathlib import Path

import numpy as np

METRICS_DIR = Path(
    os.environ.get("PMM_METRICS_DIR")
    or Path(__file__).resolve().parent.parent / "src" / "hub_dataset" / "logs" / "metrics"
)

# Un frame cuyo periodo supera este múltiplo del nominal cuenta como frame perdido
# (el loop no alcanzó a correr a tiempo).
MISSED_FRAME_FACTOR = 1.5


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent, capture_output=True, text=True, timeout=2,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _num(v):
    """Tensor/np/float -> float redondeado; listas y dicts recursivos; resto tal cual."""
    if isinstance(v, dict):
        return {k: _num(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_num(x) for x in v]
    if isinstance(v, (bool, int, str)) or v is None:
        return v
    try:
        f = float(v)
    except Exception:  # noqa: BLE001
        return str(v)
    if not np.isfinite(f):
        return None
    # Timestamps absolutos (época) necesitan µs; el resto basta con 4 decimales.
    return round(f, 6) if abs(f) > 1e6 else round(f, 4)


def _xyz(src: dict | None, keys: tuple[str, str, str]) -> list[float] | None:
    if not src or not all(k in src for k in keys):
        return None
    try:
        return [float(src[k]) for k in keys]
    except Exception:  # noqa: BLE001
        return None


def _hand_positions(obs: dict | None, action: dict | None) -> dict:
    """Posición mano derecha: comandada (acción) y medida (observación), en m."""
    out = {}
    act = _xyz(action, ("rh_pos_x", "rh_pos_y", "rh_pos_z"))
    cur = None
    if obs:
        pose = obs.get("rh_current_pose")  # Gazebo/real: [x y z qw qx qy qz]
        if pose is not None and len(pose) >= 3:
            cur = [float(pose[i]) for i in range(3)]
        else:  # MuJoCo
            cur = _xyz(obs, ("rh_ee_pos_x", "rh_ee_pos_y", "rh_ee_pos_z"))
    mocap = _xyz(obs, ("rh_target_pos_x", "rh_target_pos_y", "rh_target_pos_z"))
    if act is not None:
        out["act"] = act
    if cur is not None:
        out["cur"] = cur
    if mocap is not None:
        out["mocap"] = mocap
    if act is not None and cur is not None:
        out["track_err_cm"] = float(np.linalg.norm(np.subtract(act, cur)) * 100.0)
    return out


def _stats(values) -> str:
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if a.size == 0:
        return "n/a"
    return f"{a.mean():.1f}±{a.std():.1f} ms (p99 {np.percentile(a, 99):.1f}, max {a.max():.1f})"


class _NullMetrics:
    """Registrador desactivado: mismos métodos, no hace nada."""

    path = None

    def begin(self, *a, **k): pass
    def mark(self, *a, **k): pass
    def end(self, *a, **k): pass
    def end_episode(self, *a, **k): pass
    def event(self, *a, **k): pass
    def live(self, *a, **k): return None
    def close(self, *a, **k): pass


class FrameMetrics:
    def __init__(self, path: Path, *, backend: str, fps: float, meta: dict):
        self.path = path
        self.backend = backend
        self.fps = float(fps)
        self.nominal_ms = 1000.0 / self.fps
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8", buffering=1)
        self._write({
            "type": "session", "t": time.time(), "backend": backend, "fps": self.fps,
            "host": socket.gethostname(), "pid": os.getpid(), "git": _git_commit(), **meta,
        })
        self._prev_key = None      # (recording, episode) del frame anterior
        self._t_prev_begin = None
        self._row: dict | None = None
        self._ep_stats = self._new_ep_stats()
        self._frame_i = 0
        # Ventana corta (~1 s) para el panel "Real-time Metrics" del Hub.
        self._live_period = deque(maxlen=int(round(self.fps)))
        self._live_lat = deque(maxlen=int(round(self.fps)))

    @classmethod
    def create(cls, *, backend: str, fps: float, name: str, **meta) -> "FrameMetrics | _NullMetrics":
        """Abre ``<METRICS_DIR>/<name>.jsonl``. Nunca lanza: si falla, devuelve un no-op."""
        if os.environ.get("PMM_METRICS", "1").strip() == "0":
            return _NullMetrics()
        try:
            fm = cls(METRICS_DIR / f"{name}.jsonl", backend=backend, fps=fps, meta=_num(meta))
            print(f"[Metrics] per-frame metrics -> {fm.path}", flush=True)
            return fm
        except Exception as e:  # noqa: BLE001 -- las métricas nunca deben tumbar la grabación
            print(f"[Metrics] disabled: {e!r}", flush=True)
            return _NullMetrics()

    # ── Frames ───────────────────────────────────────────────────────────────
    def begin(self, *, recording: bool, episode: int) -> None:
        t = time.perf_counter()
        key = (bool(recording), int(episode))
        # El periodo solo vale entre frames consecutivos de la misma fase: al pasar de
        # espera a grabación (o tras guardar un episodio) hay un hueco que no es jitter.
        period = None
        if self._t_prev_begin is not None and key == self._prev_key:
            period = (t - self._t_prev_begin) * 1000.0
        if key != self._prev_key:
            self._frame_i = 0
        self._prev_key = key
        self._t_prev_begin = t
        self._t_begin = self._t_mark = t
        self._row = {
            "type": "frame", "rec": int(recording), "ep": int(episode), "i": self._frame_i,
            "t": time.time(), "period_ms": period,
        }
        self._frame_i += 1

    def mark(self, stage: str) -> None:
        """Cierra la etapa ``stage``: guarda el tiempo desde la marca anterior."""
        if self._row is None:
            return
        t = time.perf_counter()
        self._row[f"{stage}_ms"] = self._row.get(f"{stage}_ms", 0.0) + (t - self._t_mark) * 1000.0
        self._t_mark = t

    def end(self, *, robot=None, obs: dict | None = None, action: dict | None = None) -> None:
        """Cierra el frame (antes del sleep del loop) y lo escribe."""
        row = self._row
        if row is None:
            return
        self._row = None
        try:
            work = (time.perf_counter() - self._t_begin) * 1000.0
            row["work_ms"] = work
            meta = getattr(robot, "metrics_meta", None)
            if meta:
                row.update(meta)
            row.update(_hand_positions(obs, action))
            self._live_update(row)
            if row["rec"]:
                self._accumulate(row)
            self._write(_num(row))
        except Exception as e:  # noqa: BLE001
            print(f"[Metrics] frame not written: {e!r}", flush=True)

    def _live_update(self, row: dict) -> None:
        if row.get("period_ms") is not None:
            self._live_period.append(row["period_ms"])
        acks = [a["transport_ms"] + (a.get("exec_ms") or 0.0)
                for a in row.get("acks") or () if a.get("transport_ms") is not None]
        if acks:
            self._live_lat.extend(acks)
        elif not self._live_lat and "send_ms" in row:
            # Sin confirmaciones del módulo (MuJoCo o módulo antiguo): obs→send, como play_mujoco
            self._live_lat.append(sum(row.get(f"{k}_ms", 0.0) for k in ("obs", "teleop", "send")))

    def live(self) -> dict | None:
        """FPS y latencia del último segundo, para mostrar en vivo. Latencia = comando →
        goToPose si el módulo YARP confirma los comandos; si no, obs → send_action."""
        if not self._live_period:
            return None
        return {"fps": 1000.0 / float(np.mean(self._live_period)),
                "latency_ms": float(np.mean(self._live_lat)) if self._live_lat else 0.0}

    def event(self, name: str, **data) -> None:
        """Línea suelta (p. ej. reset de escena) para correlacionar con los frames."""
        self._write(_num({"type": "event", "t": time.time(), "name": name, **data}))

    # ── Episodios ────────────────────────────────────────────────────────────
    def _new_ep_stats(self) -> dict:
        return {"n": 0, "period": [], "work": [], "transport": [], "track": [], "cam_new": {}}

    def _accumulate(self, row: dict) -> None:
        s = self._ep_stats
        s["n"] += 1
        if row.get("period_ms") is not None:
            s["period"].append(row["period_ms"])
        s["work"].append(row["work_ms"])
        for ack in row.get("acks") or ():
            if ack.get("transport_ms") is not None:
                s["transport"].append(ack["transport_ms"] + (ack.get("exec_ms") or 0.0))
        if row.get("track_err_cm") is not None:
            s["track"].append(row["track_err_cm"])
        for k, v in row.items():
            if k.startswith("cam_") and k.endswith("_new"):
                c = s["cam_new"].setdefault(k[4:-4], [0, 0])
                c[0] += int(v)
                c[1] += 1

    def end_episode(self, episode: int, *, saved: bool, save_ms: float | None = None,
                    reason: str | None = None) -> None:
        s = self._ep_stats
        self._ep_stats = self._new_ep_stats()
        self._prev_key = None
        if s["n"] == 0:
            return
        period = np.asarray(s["period"], dtype=float)
        work = np.asarray(s["work"], dtype=float)
        missed = float(np.mean(period > MISSED_FRAME_FACTOR * self.nominal_ms) * 100) if period.size else None
        overrun = float(np.mean(work > self.nominal_ms) * 100) if work.size else None
        summary = {
            "type": "episode", "t": time.time(), "ep": int(episode), "saved": bool(saved),
            "reason": reason, "frames": s["n"], "save_ms": save_ms,
            "hz": float(1000.0 / period.mean()) if period.size else None,
            "period_mean_ms": float(period.mean()) if period.size else None,
            "period_std_ms": float(period.std()) if period.size else None,
            "missed_pct": missed, "overrun_pct": overrun,
        }
        self._write(_num(summary))

        parts = [f"{s['n']} frames"]
        if period.size:
            parts.append(f"{summary['hz']:.1f} Hz, period {_stats(period)}, missed {missed:.1f} %")
        if s["transport"]:
            parts.append(f"cmd->goToPose {_stats(s['transport'])}")
        if s["cam_new"]:
            parts.append("new cam frames " + ", ".join(
                f"{n} {100 * a / max(b, 1):.0f} %" for n, (a, b) in s["cam_new"].items()))
        if s["track"]:
            parts.append(f"track err {np.mean(s['track']):.1f} cm")
        tag = "saved" if saved else (reason or "discarded")
        print(f"[Metrics] ep {episode} ({tag}): " + " | ".join(parts), flush=True)

    def close(self) -> None:
        if self._ep_stats["n"] and self._prev_key is not None:
            self.end_episode(self._prev_key[1], saved=False, reason="session ended")
        self._write({"type": "session_end", "t": time.time()})
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass

    def _write(self, rec: dict) -> None:
        try:
            self._fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        except Exception:  # noqa: BLE001
            pass
