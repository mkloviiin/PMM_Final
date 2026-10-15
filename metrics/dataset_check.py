#!/usr/bin/env python3
"""Chequeo de sincronización e integridad de datasets LeRobot ya grabados.

Sirve también para datasets grabados antes de existir frame_metrics.py. Por
episodio y por cámara:

  * filas del parquet == ``length`` de meta/episodes == frames de video decodificados
  * ``timestamp`` monótono y igual a frame_index / fps (en LeRobot es ideal, no medido:
    no prueba sincronización, solo que el dataset es consistente)
  * frames de video idénticos al anterior (cámara congelada / imagen repetida), y en
    particular los idénticos *mientras el estado articular cambia*: el robot se movía
    pero la imagen no → la cámara iba atrasada respecto del estado
  * filas de ``observation.state`` idénticas a la anterior (estado no actualizado)

Ejemplos:
  python metrics/dataset_check.py data/local_icub_gazebo_demo_2026-10-05_17-45-30
  python metrics/dataset_check.py data/* --out hub_dataset/logs/metrics/report
  python metrics/dataset_check.py data/X --max-episodes 5      # decodificar menos
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Diferencia media por píxel (0-255) bajo la cual dos frames se consideran iguales.
# La compresión AV1 mete ruido: un frame repetido no sale exactamente igual tras decodificar.
FROZEN_THRESH = 0.5
STATE_EPS = 1e-6


def _decode(video_path: Path, t_from: float, t_to: float, fps: float):
    """Frames (gris, uint8) del intervalo [t_from, t_to) de un mp4."""
    import av

    frames = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        container.seek(int(max(t_from - 1.0, 0) / stream.time_base), stream=stream, any_frame=False)
        half = 0.5 / fps
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t < t_from - half:
                continue
            if t >= t_to - half:
                break
            frames.append(frame.to_ndarray(format="gray"))
    return frames


def check_dataset(root: Path, max_episodes: int | None):
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    data = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet"))],
                     ignore_index=True)
    eps = pd.concat([pd.read_parquet(p) for p in sorted((root / "meta" / "episodes").rglob("*.parquet"))],
                    ignore_index=True)
    rows = []
    for _, ep in eps.iterrows():
        idx = int(ep["episode_index"])
        if max_episodes is not None and len(rows) >= max_episodes:
            break
        d = data[data["episode_index"] == idx].sort_values("frame_index")
        ts = d["timestamp"].to_numpy(dtype=float)
        expected = d["frame_index"].to_numpy() / fps
        r = {"dataset": root.name, "episode": idx, "length": int(ep["length"]), "rows": len(d),
             "timestamps_ok": bool(np.all(np.diff(ts) > 0) and np.allclose(ts, expected, atol=1e-3))}
        moving = None
        if "observation.state" in d.columns and len(d) > 1:
            st = np.stack(d["observation.state"].to_numpy())
            dstate = np.abs(np.diff(st, axis=0)).max(axis=1)
            moving = dstate > STATE_EPS
            r["state_repeated_pct"] = float((~moving).mean() * 100)
            r["moving_pct"] = float(moving.mean() * 100)
        for key in video_keys:
            cam = key.split(".")[-1]
            chunk, file = int(ep[f"videos/{key}/chunk_index"]), int(ep[f"videos/{key}/file_index"])
            path = root / info["video_path"].format(video_key=key, chunk_index=chunk, file_index=file)
            try:
                frames = _decode(path, float(ep[f"videos/{key}/from_timestamp"]),
                                 float(ep[f"videos/{key}/to_timestamp"]), fps)
            except Exception as e:  # noqa: BLE001
                r[f"{cam}_error"] = repr(e)
                continue
            r[f"{cam}_video_frames"] = len(frames)
            if len(frames) > 1:
                diffs = np.array([np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()
                                  for a, b in zip(frames[:-1], frames[1:])])
                frozen = diffs < FROZEN_THRESH
                r[f"{cam}_frozen_pct"] = float(frozen.mean() * 100)
                if moving is not None and len(moving) == len(frozen) and moving.any():
                    r[f"{cam}_frozen_while_moving_pct"] = float(frozen[moving].mean() * 100)
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("datasets", nargs="+", help="Carpetas de datasets LeRobot (con meta/info.json)")
    ap.add_argument("--max-episodes", type=int, default=None, help="Máximo de episodios por dataset")
    ap.add_argument("--out", default=None, help="Carpeta donde escribir dataset_check.md/.csv")
    a = ap.parse_args()

    frames = []
    for ds in a.datasets:
        root = Path(ds)
        if not (root / "meta" / "info.json").exists():
            print(f"[skip] {root}: no es un dataset LeRobot (falta meta/info.json)")
            continue
        print(f"[check] {root.name} ...", flush=True)
        frames.append(check_dataset(root, a.max_episodes))
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)

    pd.set_option("display.width", 200)
    print(df.round(2).to_string(index=False))

    cams = sorted({c.split("_frozen_pct")[0] for c in df.columns if c.endswith("_frozen_pct")})
    lines = ["# Chequeo de datasets", "",
             "| Dataset | Episodios | Frames | Parquet = length = video | Timestamps OK | "
             "Estado repetido | " + " | ".join(f"{c}: congelado / congelado en movimiento" for c in cams) + " |",
             "|" + "---|" * (6 + len(cams))]
    for name, g in df.groupby("dataset", sort=False):
        consistent = all(
            (g["rows"] == g["length"]).tolist()
            + [(g[f"{c}_video_frames"] == g["length"]).all() for c in cams if f"{c}_video_frames" in g])
        cells = [name, str(len(g)), str(int(g["length"].sum())), "sí" if consistent else "**NO**",
                 "sí" if g["timestamps_ok"].all() else "**NO**",
                 f"{g['state_repeated_pct'].mean():.1f} %" if "state_repeated_pct" in g else "—"]
        for c in cams:
            fz = g.get(f"{c}_frozen_pct")
            fm = g.get(f"{c}_frozen_while_moving_pct")
            cells.append(f"{fz.mean():.1f} % / {fm.mean():.1f} %" if fz is not None and fm is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", f"_Congelado = frame con diferencia media < {FROZEN_THRESH} niveles de gris respecto del "
                  "anterior. En movimiento = frames donde observation.state cambió. Promedios por episodio._", ""]
    md = "\n".join(lines)
    print("\n" + md)
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "dataset_check.md").write_text(md, encoding="utf-8")
        df.to_csv(out / "dataset_check.csv", index=False)
        print(f"Escrito: {out / 'dataset_check.md'}")


if __name__ == "__main__":
    main()
