"""Métricas por episodio y bitácora de la sesión de grabación.

Genérico (MuJoCo, Gazebo y robot real): se alimenta de los mismos eventos que
`session._on_status` recibe de cada backend ("recording", métricas en vivo, "saved:N").

En la carpeta del dataset, subcarpeta `hub_session/`:
  episode_metrics.json / .csv / .png   promedios por episodio guardado (no cuenta los descartados)
  logbook.json / .txt                  notas escritas durante la grabación

Las notas o filas que llegan antes de que lerobot cree la carpeta del dataset (no se puede
crear antes: LeRobotDataset.create falla si ya existe) quedan en memoria y se escriben apenas exista.
"""

from __future__ import annotations

import csv
import json
import statistics
import threading
import time
from pathlib import Path

SUBDIR = "hub_session"

_lock = threading.Lock()
_root: Path | None = None
_samples: list[tuple[float, float]] = []  # (fps, latency_ms) del episodio en curso
_rec_t0: float | None = None
_rows: list[dict] = []
_notes: list[dict] = []
version = 0  # sube con cada fila/nota nueva; el poller de la UI solo redibuja si cambia


def begin(dataset_root: Path) -> None:
    """Nueva sesión: se olvida lo anterior y se apunta a la carpeta del nuevo dataset."""
    global _root, _rec_t0, version
    with _lock:
        _root = Path(dataset_root)
        _samples.clear(); _rows.clear(); _notes.clear()
        _rec_t0 = None
        version += 1


def reset() -> None:
    """Sin sesión (al cambiar de plataforma en el Hub): gráfico y bitácora vacíos. Lo ya
    escrito en hub_session/ del dataset anterior no se toca."""
    global _root, _rec_t0, version
    with _lock:
        _root = None
        _samples.clear(); _rows.clear(); _notes.clear()
        _rec_t0 = None
        version += 1


def on_recording_start() -> None:
    global _rec_t0
    with _lock:
        _samples.clear()
        _rec_t0 = time.time()


def on_metrics(metrics: dict, recording: bool) -> None:
    fps = float(metrics.get("fps") or 0.0)
    if recording and fps > 0:
        with _lock:
            _samples.append((fps, float(metrics.get("latency_ms") or 0.0)))


def on_saved(n_saved: int, size_mb: float) -> None:
    """Episodio guardado: n_saved = cantidad de episodios guardados (el índice es n_saved-1)."""
    global version
    with _lock:
        fps = [s[0] for s in _samples]
        lat = [s[1] for s in _samples]
        _rows.append({
            "episode": n_saved - 1,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_s": round(time.time() - _rec_t0, 2) if _rec_t0 else None,
            "fps_mean": round(statistics.fmean(fps), 2) if fps else None,
            "fps_std": round(statistics.pstdev(fps), 2) if len(fps) > 1 else 0.0,
            "latency_ms_mean": round(statistics.fmean(lat), 2) if lat else None,
            "latency_ms_std": round(statistics.pstdev(lat), 2) if len(lat) > 1 else 0.0,
            "samples": len(fps),
            "dataset_size_mb": round(float(size_mb), 2),
        })
        _samples.clear()
        version += 1
    _flush()


def add_note(text: str, episode: int, phase: str) -> dict | None:
    global version
    text = (text or "").strip()
    if not text:
        return None
    entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "episode": int(episode),
             "phase": phase, "note": text}
    with _lock:
        _notes.append(entry)
        version += 1
    _flush()
    return entry


def has_session() -> bool:
    return _root is not None


def rows() -> list[dict]:
    with _lock:
        return list(_rows)


def notes() -> list[dict]:
    with _lock:
        return list(_notes)


def dataset_root() -> Path | None:
    return _root


def folder() -> Path | None:
    return _root / SUBDIR if _root else None


# ── Escritura en la carpeta del dataset ───────────────────────────────────────

def _flush() -> None:
    """Escribe todo si la carpeta del dataset ya existe; si no, queda en memoria."""
    with _lock:
        root, rows_, notes_ = _root, list(_rows), list(_notes)
    if root is None or not (root / "meta" / "info.json").exists():
        return
    out = root / SUBDIR
    out.mkdir(exist_ok=True)
    try:
        if rows_:
            (out / "episode_metrics.json").write_text(json.dumps(rows_, indent=2), encoding="utf-8")
            with open(out / "episode_metrics.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows_[0]))
                w.writeheader(); w.writerows(rows_)
            _save_png(rows_, out / "episode_metrics.png", _nominal_fps(root))
        if notes_:
            (out / "logbook.json").write_text(json.dumps(notes_, indent=2, ensure_ascii=False), encoding="utf-8")
            (out / "logbook.txt").write_text(
                "".join(f"[{n['time']}] Episode {n['episode']} ({n['phase']}): {n['note']}\n" for n in notes_),
                encoding="utf-8")
    except Exception as e:  # noqa: BLE001 - la bitácora nunca debe tumbar la grabación
        print(f"[Hub] Could not write {out}: {e!r}")


def _nominal_fps(root: Path) -> float | None:
    try:
        return float(json.loads((root / "meta" / "info.json").read_text())["fps"])
    except Exception:  # noqa: BLE001
        return None


def _save_png(rows_: list[dict], path: Path, nominal_fps: float | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [r["episode"] for r in rows_]
    def col(k):
        return [r[k] if r[k] is not None else float("nan") for r in rows_]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    ax = axes[0, 0]
    ax.errorbar(ep, col("fps_mean"), yerr=col("fps_std"), fmt="o-", capsize=3, color="#2a6fdb")
    if nominal_fps:
        ax.axhline(nominal_fps, ls="--", color="gray", lw=1, label=f"nominal {nominal_fps:g}")
        ax.legend(loc="lower right", fontsize=8)
    ax.set_title("FPS (mean ± std)")
    ax = axes[0, 1]
    ax.errorbar(ep, col("latency_ms_mean"), yerr=col("latency_ms_std"), fmt="o-", capsize=3, color="#e8590c")
    ax.set_title("Latency [ms] (mean ± std)")
    ax = axes[1, 0]
    ax.bar(ep, col("duration_s"), color="#37b24d")
    ax.set_title("Episode duration [s]")
    ax = axes[1, 1]
    ax.plot(ep, col("dataset_size_mb"), "o-", color="#7048e8")
    ax.set_title("Dataset size [MB]")
    for a in axes.flat:
        a.grid(alpha=0.3)
    from matplotlib.ticker import MaxNLocator
    for a in axes.flat:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
    for a in axes[1]:
        a.set_xlabel("Episode")
    fig.suptitle(f"Per-episode metrics — {path.parent.parent.name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
