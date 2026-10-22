#!/usr/bin/env python3
"""Copia un dataset pasando sus cuaterniones al hemisferio común de MuJoCo y Gazebo.

q y -q son la misma orientación, pero para una política son valores opuestos. Hasta el
2026-10-21 cada fuente elegía un signo distinto (en Gazebo rh_ee_quat salía con w >= 0 y
rh_target_quat / la acción con w < 0: opuestos en el 92 % de los frames). Desde entonces ambos
grabadores usan QUAT_HEMISPHERE (lerobot_robot_icubrg.icub / lerobot_robot_icub.icub_mujoco):
manos (ee, target y acción) con quat_y <= 0 y objetos con quat_w >= 0. Este script deja un
dataset anterior igual que uno grabado ahora: cambia el signo frame a frame (sin pérdida) en
observation.state y action y recalcula las estadísticas (meta/stats.json y las de cada
episodio). El original no se toca. Solo para datasets en el frame root (desde el 2026-10-20).

Uso:
    python tools/canonicalize_quats.py ../data/local_final_gazebo_2026-10-20_15-25-43 \\
        [--dst ../data_curated/<nombre>]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}
VECTOR_KEYS = ("observation.state", "action")


def quat_groups(names: list[str]) -> list[tuple[str, list[int]]]:
    """(prefijo, índices w x y z) de cada cuaternión completo en names."""
    groups = []
    for name in names:
        if name.endswith("_quat_w"):
            prefix = name[:-2]
            keys = [f"{prefix}_{ax}" for ax in "wxyz"]
            if all(k in names for k in keys):
                groups.append((prefix, [names.index(k) for k in keys]))
    return groups


def canonicalize(values: np.ndarray, names: list[str]) -> dict[str, int]:
    """Cambia el signo en el lugar (mismo criterio que canonicalize_quats); frames cambiados."""
    flipped = {}
    for prefix, idx in quat_groups(names):
        w, _, y, _ = idx
        rows = values[:, w] < 0 if prefix.startswith("obj") else values[:, y] > 0
        values[np.ix_(rows, idx)] *= -1
        flipped[prefix] = int(rows.sum())
    return flipped


def stats_of(values: np.ndarray, keys) -> dict:
    stats = {
        "min": values.min(0), "max": values.max(0), "mean": values.mean(0),
        "std": values.std(0), "count": np.array([len(values)]),
    }
    stats.update({k: np.quantile(values, q, axis=0) for k, q in QUANTILES.items()})
    return {k: stats[k] for k in keys}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("src", type=Path, help="Dataset root (with meta/info.json)")
    p.add_argument("--dst", type=Path, default=None, help="Output root (default: <src>_quat)")
    args = p.parse_args()

    src = args.src.expanduser().resolve()
    dst = (args.dst or src.with_name(src.name + "_quat")).expanduser().resolve()
    if dst.exists():
        raise SystemExit(f"{dst} already exists")

    info = json.loads((src / "meta" / "info.json").read_text())
    names = {k: info["features"][k]["names"] for k in VECTOR_KEYS if k in info["features"]}
    if not any(quat_groups(n) for n in names.values()):
        raise SystemExit(f"No quaternions in {list(names)}: nothing to do")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("meta_backup*"))

    # Datos
    all_values = {k: [] for k in names}
    per_episode: dict[str, dict[int, np.ndarray]] = {k: {} for k in names}
    flipped = {k: {} for k in names}
    for f in sorted((dst / "data").glob("**/*.parquet")):
        df = pd.read_parquet(f)
        for key, key_names in names.items():
            values = np.stack(df[key].values).astype(np.float32)
            for prefix, n in canonicalize(values, key_names).items():
                flipped[key][prefix] = flipped[key].get(prefix, 0) + n
            df[key] = list(values)
            all_values[key].append(values)
            for ep in df["episode_index"].unique():
                rows = values[df["episode_index"].values == ep]
                prev = per_episode[key].get(int(ep))  # un episodio puede ocupar dos archivos
                per_episode[key][int(ep)] = rows if prev is None else np.concatenate([prev, rows])
        df.to_parquet(f, index=False)

    # Estadísticas globales (normalización del entrenamiento)
    stats_path = dst / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    for key in names:
        values = np.concatenate(all_values[key])
        stats[key] = {k: v.tolist() for k, v in stats_of(values, stats[key].keys()).items()}
    stats_path.write_text(json.dumps(stats, indent=4))

    # Estadísticas por episodio (meta/episodes)
    for f in sorted((dst / "meta" / "episodes").glob("**/*.parquet")):
        ep_df = pd.read_parquet(f)
        for key in names:
            cols = [c.removeprefix(f"stats/{key}/") for c in ep_df.columns if c.startswith(f"stats/{key}/")]
            new_cols = {k: [] for k in cols}
            for row, ep in enumerate(ep_df["episode_index"].values):
                ep_stats = stats_of(per_episode[key][int(ep)], cols)
                for k in cols:
                    old = np.asarray(ep_df[f"stats/{key}/{k}"].iloc[row])  # mismo tipo y forma
                    new_cols[k].append(np.asarray(ep_stats[k], dtype=old.dtype).reshape(old.shape))
            for k, values in new_cols.items():
                ep_df[f"stats/{key}/{k}"] = values
        ep_df.to_parquet(f, index=False)

    total = sum(len(v) for v in per_episode["action"].values()) if "action" in per_episode else None
    for key, counts in flipped.items():
        for prefix, n in counts.items():
            print(f"{key:18s} {prefix:16s} sign flipped in {n} frames" + (f" of {total}" if total else ""))
    print(f"Saved to {dst}")


if __name__ == "__main__":
    main()
