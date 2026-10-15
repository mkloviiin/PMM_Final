#!/usr/bin/env python3
"""Informe de latencia, jitter y sincronización para MuJoCo, Gazebo y el robot real.

Lee lo que dejan las sesiones en ``hub_dataset/logs/metrics/``:
  * ``<dataset>.jsonl``              métricas por frame (frame_metrics.py, se generan al grabar)
  * ``probe_<backend>_*_trials.csv``  pruebas de escalón (latency_probe.py)
y opcionalmente los logs del Hub (``hub_dataset/logs/hub_*.txt``) para el módulo YARP.

Genera en ``--out`` (default ``hub_dataset/logs/metrics/report``):
  resumen.md      tablas + definiciones, listo para copiar al informe
  tablas.tex      las mismas tablas en LaTeX (booktabs)
  resumen.json    todos los números
  fig_*.png       figuras

Ejemplos:
  python metrics/metrics_report.py                       # todo lo que haya
  python metrics/metrics_report.py --frames a.jsonl b.jsonl --logs hub_dataset/logs/hub_*.txt
  python metrics/metrics_report.py --include-waiting     # incluir frames fuera de grabación
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
METRICS_DIR = ROOT / "hub_dataset" / "logs" / "metrics"

BACKENDS = ["mujoco", "gazebo", "real"]
LABEL = {"mujoco": "MuJoCo", "gazebo": "Gazebo", "real": "Real"}
# Color fijo por backend en todas las figuras (paleta categórica validada, 3 primeros slots)
COLOR = {"mujoco": "#2a78d6", "gazebo": "#eb6834", "real": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MISSED_FRAME_FACTOR = 1.5

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
})


# ── Carga ────────────────────────────────────────────────────────────────────
def load_frames(paths):
    """-> {backend: {"frames": df, "episodes": df, "sessions": [meta]}}"""
    out = {}
    for path in paths:
        session, frames, episodes = None, [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                typ = rec.get("type")
                if typ == "session":
                    session = rec
                elif typ == "frame":
                    frames.append(rec)
                elif typ == "episode":
                    episodes.append(rec)
        if session is None or not frames:
            continue
        b = session["backend"]
        fdf = pd.DataFrame(frames)
        fdf["session"] = Path(path).stem
        fdf["fps"] = session["fps"]
        edf = pd.DataFrame(episodes)
        if not edf.empty:
            edf["session"] = Path(path).stem
        d = out.setdefault(b, {"frames": [], "episodes": [], "sessions": []})
        d["frames"].append(fdf)
        d["episodes"].append(edf)
        d["sessions"].append({**session, "file": str(path), "n_frames": len(fdf)})
    for b, d in out.items():
        d["frames"] = pd.concat(d["frames"], ignore_index=True)
        eps = [e for e in d["episodes"] if not e.empty]
        d["episodes"] = pd.concat(eps, ignore_index=True) if eps else pd.DataFrame()
    return out


def load_probes(paths):
    """-> {backend: {"trials": df, "samples": df, "meta": dict}}"""
    out = {}
    for path in paths:
        path = Path(path)
        with open(path) as f:
            first = f.readline()
        meta = dict(kv.split("=", 1) for kv in first.lstrip("# ").split())
        trials = pd.read_csv(path, comment="#")
        trials["probe"] = path.name
        samples_path = path.with_name(path.name.replace("_trials.csv", "_samples.csv"))
        samples = pd.read_csv(samples_path) if samples_path.exists() else pd.DataFrame()
        if not samples.empty:
            samples["probe"] = path.name
        b = meta.get("backend", "?")
        d = out.setdefault(b, {"trials": [], "samples": [], "meta": meta})
        d["trials"].append(trials)
        d["samples"].append(samples)
    for d in out.values():
        d["trials"] = pd.concat(d["trials"], ignore_index=True)
        d["samples"] = pd.concat(d["samples"], ignore_index=True)
    return out


# ── Estadística ──────────────────────────────────────────────────────────────
def desc(values) -> dict | None:
    a = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if a.size == 0:
        return None
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
            "max": float(a.max())}


def col(df, name):
    return df[name] if name in df.columns else pd.Series(dtype=float)


def explode_acks(df) -> pd.DataFrame:
    if "acks" not in df.columns:
        return pd.DataFrame()
    rows = [a for lst in df["acks"].dropna() for a in lst]
    if not rows:
        return pd.DataFrame()
    a = pd.DataFrame(rows)
    a["total_ms"] = a["transport_ms"] + a["exec_ms"]
    return a


def same_clock(t_local, stamp) -> bool:
    """¿El timestamp de origen está en el mismo reloj que este PC? (diferencia mediana
    entre 0 y 5 s). En Gazebo las cámaras pueden usar tiempo simulado; en el robot
    real, el reloj del PC del robot (sin NTP puede haber desfase)."""
    d = (pd.to_numeric(t_local, errors="coerce") - pd.to_numeric(stamp, errors="coerce")).dropna()
    return bool(len(d)) and 0.0 <= float(d.median()) < 5.0


def camera_names(df):
    return sorted({c[4:-4] for c in df.columns if c.startswith("cam_") and c.endswith("_new")})


def lag_xcorr(df, max_lag=90):
    """Retardo comando→posición medida por correlación cruzada de velocidades, por
    episodio con movimiento suficiente. Devuelve lista de retardos en ms.
    max_lag en frames (90 ≈ 3 s a 30 fps); un máximo en el borde de la ventana se descarta
    porque el retardo real puede ser mayor."""
    lags = []
    for (_, _), ep in df.groupby(["session", "ep"]):
        ep = ep.sort_values("i")
        src = None
        for key in ("act", "mocap"):
            if key in ep.columns and ep[key].notna().all() and len(ep) > 3 * max_lag:
                arr = np.array(ep[key].tolist(), dtype=float)
                if arr.std(axis=0).max() > 0.005:  # > 5 mm de movimiento
                    src = (key, arr)
                    break
        if src is None or "cur" not in ep.columns or not ep["cur"].notna().all():
            continue
        key, cmd = src
        cur = np.array(ep["cur"].tolist(), dtype=float)
        dc, dm = np.diff(cmd, axis=0), np.diff(cur, axis=0)
        best, best_k = -np.inf, None
        for k in range(0, max_lag + 1):
            a, b = dc[: len(dc) - k], dm[k:]
            num = float((a * b).sum())
            den = float(np.sqrt((a * a).sum() * (b * b).sum())) or 1.0
            if num / den > best:
                best, best_k = num / den, k
        if best_k is None or best < 0.3 or best_k == max_lag:
            continue
        # mocap(obs k) es el target aplicado en el frame k-1: un frame más de retardo
        frames = best_k + (1 if key == "mocap" else 0)
        period = pd.to_numeric(ep["period_ms"], errors="coerce").median()
        lags.append(frames * period)
    return lags


def analyze_backend(b, d, include_waiting=False):
    df = d["frames"]
    if not include_waiting:
        df = df[df["rec"] == 1]
    res = {"backend": b, "sessions": [s["file"] for s in d["sessions"]],
           "frames": int(len(df)), "episodes": int(df.groupby(["session", "ep"]).ngroups)}
    if df.empty:
        return res, df
    fps = float(df["fps"].iloc[0])
    nominal = 1000.0 / fps
    period = pd.to_numeric(col(df, "period_ms"), errors="coerce").dropna()
    work = pd.to_numeric(col(df, "work_ms"), errors="coerce").dropna()
    res["fps"] = fps
    res["loop"] = {
        "nominal_ms": nominal,
        "period": desc(period),
        "hz": float(1000.0 / period.mean()) if len(period) else None,
        "jitter_std_ms": float(period.std()) if len(period) else None,
        "jitter_mad_ms": float((period - nominal).abs().mean()) if len(period) else None,
        "missed_pct": float((period > MISSED_FRAME_FACTOR * nominal).mean() * 100) if len(period) else None,
        "overrun_pct": float((work > nominal).mean() * 100) if len(work) else None,
    }
    res["stages"] = {s: desc(col(df, f"{s}_ms")) for s in ("obs", "teleop", "send", "record", "work")}
    ep = d["episodes"]
    if not ep.empty and "save_ms" in ep.columns:
        res["save_episode_ms"] = desc(ep.loc[ep["saved"] == True, "save_ms"])  # noqa: E712
        res["episodes_saved"] = int((ep["saved"] == True).sum())  # noqa: E712
        res["episodes_discarded"] = int((ep["saved"] != True).sum())  # noqa: E712

    # Comandos (YARP)
    acks = explode_acks(df)
    if not acks.empty:
        cmd = {"transport": desc(acks["transport_ms"]), "exec": desc(acks["exec_ms"]),
               "total": desc(acks["total_ms"])}
        counts = acks["sent"].value_counts()
        cmd["sent_pct"] = float(counts.get(1, 0) / len(acks) * 100)
        cmd["deadband_pct"] = float(counts.get(0, 0) / len(acks) * 100)
        cmd["failed_pct"] = float(counts.get(-1, 0) / len(acks) * 100)
        for side in ("rh", "lh"):
            sent = pd.to_numeric(col(df, f"{side}_cmd_seq"), errors="coerce").dropna().astype(int)
            acked = set(acks.loc[acks["arm"] == side, "seq"].astype(int))
            if len(sent):
                lost = 1 - len(acked & set(sent)) / len(set(sent))
                cmd[f"{side}_not_processed_pct"] = float(lost * 100)
        res["command"] = cmd
    if "gaze_rpc_ms" in df.columns:
        res["gaze_rpc"] = desc(df["gaze_rpc_ms"])

    # Sincronización
    sync = {}
    if "state_age_ms" in df.columns:
        sync["state_age"] = desc(df["state_age_ms"])
        sync["state_wait"] = desc(col(df, "state_wait_ms"))
    for cam in camera_names(df):
        new = pd.to_numeric(df[f"cam_{cam}_new"], errors="coerce").fillna(0)
        c = {"new_pct": float(new.mean() * 100), "repeated_pct": float((1 - new.mean()) * 100),
             "age_arrival": desc(col(df, f"cam_{cam}_age_ms"))}
        seq_col = f"cam_{cam}_seq"
        if seq_col in df.columns:
            dropped = total = 0
            # Por episodio: entre episodios la cámara sigue publicando (save_episode, reset)
            # y ese salto de secuencia no es una pérdida.
            ep_keys = [k for k in ("session", "rec", "ep") if k in df.columns]
            for _, g in df[new == 1].groupby(ep_keys):
                s = pd.to_numeric(g[seq_col], errors="coerce").dropna().to_numpy()
                gaps = np.diff(s)
                gaps = gaps[(gaps > 0) & (gaps < 1000)]
                dropped += int((gaps - 1).sum())
                total += int(gaps.sum())
            c["dropped_pct"] = float(dropped / total * 100) if total else None
            stamps = pd.to_numeric(df.loc[new == 1, f"cam_{cam}_stamp"], errors="coerce")
            c["source_fps"] = float(1.0 / np.median(np.diff(stamps.dropna().to_numpy())))\
                if stamps.notna().sum() > 2 else None
            c["same_clock"] = same_clock(df["t"], df.get(f"cam_{cam}_stamp"))
            if c["same_clock"]:
                c["age_capture"] = desc(col(df, f"cam_{cam}_stamp_age_ms"))
                if "state_stamp" in df.columns:
                    c["skew_vs_state"] = desc((df[f"cam_{cam}_stamp"] - df["state_stamp"]) * 1000)
        # fps efectivos de cámara vistos por el loop
        dur = df.groupby(["session", "ep"])["t"].agg(lambda t: t.max() - t.min()).sum()
        c["effective_fps"] = float(new.sum() / dur) if dur > 0 else None
        sync[f"cam_{cam}"] = c
    cams = camera_names(df)
    if "left" in cams and "right" in cams and "cam_left_stamp" in df.columns and "cam_right_stamp" in df.columns:
        sync["stereo_skew"] = desc(((df["cam_left_stamp"] - df["cam_right_stamp"]) * 1000).abs())
    if "cam_sim_t" in df.columns:  # MuJoCo
        sync["cam_vs_state_sim_ms"] = desc((df["cam_sim_t"] - df["state_sim_t"]).abs() * 1000)
        sync["render"] = desc(col(df, "render_ms"))
        sync["physics"] = desc(col(df, "physics_ms"))
        ratios = []
        for _, g in df.groupby(["session", "ep"]):
            g = g.sort_values("i")
            if len(g) > 10:
                ratios.append((g["state_sim_t"].iloc[-1] - g["state_sim_t"].iloc[0])
                              / (g["t"].iloc[-1] - g["t"].iloc[0]))
        sync["sim_time_ratio"] = float(np.median(ratios)) if ratios else None
    res["sync"] = sync

    # Seguimiento
    if "track_err_cm" in df.columns:
        lags = lag_xcorr(df)
        res["tracking"] = {"error_cm": desc(df["track_err_cm"]), "lag_ms": desc(lags)}
    return res, df


def analyze_probe(p):
    t = p["trials"]
    res = {"meta": p["meta"], "n": int(len(t))}
    for k in ("t_move_ms", "t10_ms", "t50_ms", "t90_ms", "t_settle_ms", "overshoot_pct",
              "reached_pct", "final_err_mm", "cmd_transport_ms", "cmd_exec_ms", "sample_dt_ms"):
        res[k] = desc(col(t, k))
    res["settled_pct"] = float(t["t_settle_ms"].notna().mean() * 100) if "t_settle_ms" in t else None
    return res


LOOP_RE = re.compile(r"\[Loop\](?: SLOW:)? ([\d.]+) Hz \(target (\d+)\), worst cycle (\d+) ms")
CAM_RE = re.compile(r"cam left ([\d.]+) fps, right ([\d.]+) fps")
ARM_RE = re.compile(r"\[Arm\] (\w+): (\d+) targets in 2s \((\d+) goToPose FAILED\).*?hand-target ([\d.]+) cm")


def analyze_logs(paths):
    out = []
    for path in paths:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        hz, worst, caml, camr, targets, failed, dist = [], [], [], [], 0, 0, []
        target = None
        for line in text.splitlines():
            if "[yarp_module]" not in line and "[Loop]" not in line and "[Arm]" not in line:
                continue
            if m := LOOP_RE.search(line):
                hz.append(float(m.group(1)))
                target = int(m.group(2))
                worst.append(float(m.group(3)))
                if c := CAM_RE.search(line):
                    caml.append(float(c.group(1)))
                    camr.append(float(c.group(2)))
            if m := ARM_RE.search(line):
                targets += int(m.group(2))
                failed += int(m.group(3))
                dist.append(float(m.group(4)))
        if not hz and not targets:
            continue
        robot = "real" if re.search(r"--robot icub\b|robot_name override -> icub\b", text) else (
            "gazebo" if "icubSim" in text else "?")
        out.append({
            "file": Path(path).name, "robot": robot, "windows_5s": len(hz), "target_hz": target,
            "hz": desc(hz), "worst_cycle_ms": desc(worst),
            "cam_left_fps": desc([v for v in caml if v > 0]), "cam_right_fps": desc([v for v in camr if v > 0]),
            "slow_lines": text.count("[SLOW]"), "stale_camera_warnings": text.count("No new frames"),
            "ik_targets": targets, "ik_failed_pct": failed / targets * 100 if targets else None,
            "hand_target_cm": desc(dist),
        })
    return out


# ── Figuras ──────────────────────────────────────────────────────────────────
def _present(results):
    return [b for b in BACKENDS if b in results and results[b].get("frames")]


def fig_period(results, frames, out):
    bs = _present(results)
    if not bs:
        return None
    fig, axes = plt.subplots(len(bs), 1, figsize=(6.4, 1.7 * len(bs) + 0.6), sharex=True, squeeze=False)
    allp = pd.concat([pd.to_numeric(frames[b]["period_ms"], errors="coerce").dropna() for b in bs])
    nominal0 = results[bs[0]]["loop"]["nominal_ms"]
    lo = min(np.percentile(allp, 0.5), nominal0 * 0.85)
    hi = max(np.percentile(allp, 99.5), nominal0 * 1.15)
    bins = np.linspace(lo, hi, 80)
    for ax, b in zip(axes[:, 0], bs):
        p = pd.to_numeric(frames[b]["period_ms"], errors="coerce").dropna()
        nominal = results[b]["loop"]["nominal_ms"]
        w = np.ones(len(p)) / max(len(p), 1) * 100
        ax.hist(p.clip(bins[0], bins[-1]), bins=bins, weights=w, color=COLOR[b], edgecolor="white", linewidth=0.5)
        ax.axvline(nominal, color=INK2, linestyle="--", linewidth=1)
        ax.text(nominal, ax.get_ylim()[1] * 0.92, f" nominal {nominal:.1f} ms", color=INK2, fontsize=8, va="top")
        lp = results[b]["loop"]
        ax.set_title(f"{LABEL[b]} — {lp['hz']:.1f} Hz, σ = {lp['jitter_std_ms']:.2f} ms, "
                     f"p99 = {lp['period']['p99']:.1f} ms", loc="left", fontsize=9)
        ax.set_ylabel("% de frames")
    axes[-1, 0].set_xlabel("Periodo entre frames del loop de teleoperación [ms]")
    fig.tight_layout()
    path = out / "fig_periodo_hist.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_period_series(results, frames, out):
    bs = _present(results)
    if not bs:
        return None
    fig, axes = plt.subplots(len(bs), 1, figsize=(6.4, 1.6 * len(bs) + 0.6), squeeze=False)
    for ax, b in zip(axes[:, 0], bs):
        df = frames[b]
        # Episodio más largo como ejemplo representativo
        key = df.groupby(["session", "ep"]).size().idxmax()
        ep = df[(df["session"] == key[0]) & (df["ep"] == key[1])].sort_values("i")
        t = ep["t"] - ep["t"].iloc[0]
        ax.plot(t, ep["period_ms"], color=COLOR[b], linewidth=1)
        ax.axhline(results[b]["loop"]["nominal_ms"], color=INK2, linestyle="--", linewidth=1)
        ax.set_title(f"{LABEL[b]} — episodio {key[1]} ({key[0]})", loc="left", fontsize=9)
        ax.set_ylabel("periodo [ms]")
    axes[-1, 0].set_xlabel("Tiempo desde el inicio del episodio [s]")
    fig.tight_layout()
    path = out / "fig_periodo_serie.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_stages(results, out):
    bs = _present(results)
    stages = [("obs", "get_\nobservation"), ("teleop", "teleop +\nprocesadores"), ("send", "send_\naction"),
              ("record", "add_frame"), ("work", "total\n(sin sleep)")]
    stages = [(s, l) for s, l in stages if any(results[b]["stages"].get(s) for b in bs)]
    if not bs or not stages:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(stages))
    w = 0.8 / len(bs)
    for k, b in enumerate(bs):
        means = [(results[b]["stages"].get(s) or {}).get("mean", np.nan) for s, _ in stages]
        p95 = [(results[b]["stages"].get(s) or {}).get("p95", np.nan) for s, _ in stages]
        xs = x - 0.4 + w * (k + 0.5)
        ax.bar(xs, means, width=w * 0.92, color=COLOR[b], label=LABEL[b])
        ax.scatter(xs, p95, marker="_", s=120, color=INK, linewidths=1.5, zorder=3,
                   label="p95" if k == 0 else None)
    nominal = results[bs[0]]["loop"]["nominal_ms"]
    ax.axhline(nominal, color=INK2, linestyle="--", linewidth=1)
    ax.text(len(stages) - 0.5, nominal, f"presupuesto {nominal:.1f} ms", color=INK2, fontsize=8,
            ha="right", va="bottom")
    ax.set_xticks(x, [l for _, l in stages])
    ax.set_ylabel("tiempo por frame [ms] (media; marca = p95)")
    ax.legend(ncol=len(bs) + 1, loc="upper left", bbox_to_anchor=(0, 1.13))
    fig.tight_layout()
    path = out / "fig_etapas.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _ecdf(ax, values, color, label):
    a = np.sort(pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy())
    if a.size:
        ax.step(a, np.arange(1, a.size + 1) / a.size * 100, where="post", color=color, label=label)


def fig_command(frames, out):
    bs = [b for b in ("gazebo", "real") if b in frames and not explode_acks(frames[b]).empty]
    if not bs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    for b in bs:
        a = explode_acks(frames[b])
        _ecdf(axes[0], a["transport_ms"], COLOR[b], LABEL[b])
        _ecdf(axes[1], a["total_ms"], COLOR[b], LABEL[b])
    axes[0].set_title("Cliente → módulo YARP", loc="left", fontsize=9)
    axes[1].set_title("Cliente → goToPose emitido", loc="left", fontsize=9)
    for ax in axes:
        ax.set_xlabel("latencia [ms]")
    axes[0].set_ylabel("% de comandos ≤ x")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    path = out / "fig_latencia_comando_cdf.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_step(probes, out):
    bs = [b for b in BACKENDS if b in probes and not probes[b]["samples"].empty]
    if not bs:
        return None
    fig, axes = plt.subplots(1, len(bs), figsize=(2.6 * len(bs) + 0.8, 2.8), sharey=True, squeeze=False)
    for ax, b in zip(axes[0], bs):
        tr, sm = probes[b]["trials"], probes[b]["samples"]
        for _, r in tr.iterrows():
            if pd.isna(r.get("t_cmd")):
                continue
            s = sm[(sm["probe"] == r["probe"]) & (sm["trial"] == r["trial"]) & (sm["phase"] == "step")]
            s = s[s["t"] >= r["t_cmd"]]
            delta = np.array([r["delta_x"], r["delta_y"], r["delta_z"]])
            L = np.linalg.norm(delta)
            start = np.array([r["start_x"], r["start_y"], r["start_z"]])
            along = (s[["x", "y", "z"]].to_numpy() - start) @ (delta / L)
            ax.plot((s["t"] - r["t_cmd"]) * 1000, along * 1000, color=COLOR[b], linewidth=1, alpha=0.45)
        L_mm = float(np.median(tr["step_mm"]))
        ax.axhline(L_mm, color=INK2, linestyle="--", linewidth=1)
        ax.text(ax.get_xlim()[1], L_mm, "comandado", color=INK2, fontsize=8, ha="right", va="bottom")
        res = analyze_probe(probes[b])
        tm = res["t_move_ms"]
        ax.set_title(f"{LABEL[b]} — 1er mov. {tm['mean']:.0f}±{tm['std']:.0f} ms" if tm else LABEL[b],
                     loc="left", fontsize=9)
        ax.set_xlabel("tiempo desde el comando [ms]")
    axes[0, 0].set_ylabel("desplazamiento de la mano [mm]")
    fig.tight_layout()
    path = out / "fig_escalon.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_camera_age(results, frames, out):
    bs = [b for b in _present(results) if camera_names(frames[b]) and
          any(f"cam_{c}_age_ms" in frames[b] for c in camera_names(frames[b]))]
    if not bs:
        return None
    fig, axes = plt.subplots(len(bs), 1, figsize=(6.4, 1.7 * len(bs) + 0.6), sharex=True, squeeze=False)
    for ax, b in zip(axes[:, 0], bs):
        df = frames[b]
        cams = [c for c in camera_names(df) if f"cam_{c}_age_ms" in df]
        vals = [pd.to_numeric(df[f"cam_{c}_age_ms"], errors="coerce").dropna() for c in cams]
        # Una sola serie por panel (todas las cámaras juntas): el color sigue siendo el backend
        allv = pd.concat(vals)
        bins = np.linspace(0, max(np.percentile(allv, 99.5), 1), 60)
        ax.hist(allv.clip(upper=bins[-1]), bins=bins, weights=np.ones(len(allv)) / len(allv) * 100,
                color=COLOR[b], edgecolor="white", linewidth=0.5)
        rep = np.mean([results[b]["sync"][f"cam_{c}"]["repeated_pct"] for c in cams])
        ax.set_title(f"{LABEL[b]} — cámaras {', '.join(cams)}; frames repetidos {rep:.1f} %",
                     loc="left", fontsize=9)
        ax.set_ylabel("% de frames")
    axes[-1, 0].set_xlabel("Antigüedad de la imagen grabada desde que llegó al cliente [ms]")
    fig.tight_layout()
    path = out / "fig_camara_edad.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_tracking(results, frames, out):
    bs = [b for b in _present(results) if "act" in frames[b] and "cur" in frames[b]]
    if not bs:
        return None
    fig, axes = plt.subplots(len(bs), 1, figsize=(6.4, 1.7 * len(bs) + 0.6), squeeze=False)
    for ax, b in zip(axes[:, 0], bs):
        df = frames[b].dropna(subset=["act", "cur"])
        if df.empty:
            continue
        # Episodio con más movimiento comandado
        def motion(g):
            return np.array(g["act"].tolist()).std(axis=0).max()
        key = df.groupby(["session", "ep"]).apply(motion).idxmax()
        ep = df[(df["session"] == key[0]) & (df["ep"] == key[1])].sort_values("i")
        t = ep["t"] - ep["t"].iloc[0]
        act, cur = np.array(ep["act"].tolist()), np.array(ep["cur"].tolist())
        axis = int(np.argmax(act.std(axis=0)))
        ax.plot(t, act[:, axis] * 100, color=INK2, linewidth=1.2, linestyle="--", label="comandado")
        ax.plot(t, cur[:, axis] * 100, color=COLOR[b], linewidth=1.5, label="medido")
        ax.set_title(f"{LABEL[b]} — posición mano derecha, eje {'xyz'[axis]}", loc="left", fontsize=9)
        ax.set_ylabel("[cm]")
        ax.legend(loc="upper right", ncol=2)
    axes[-1, 0].set_xlabel("Tiempo desde el inicio del episodio [s]")
    fig.tight_layout()
    path = out / "fig_seguimiento.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ── Tablas ───────────────────────────────────────────────────────────────────
def ms(d, key="mean", nd=1):
    return "—" if not d else f"{d[key]:.{nd}f}"


def ms_pm(d, nd=1):
    return "—" if not d else f"{d['mean']:.{nd}f} ± {d['std']:.{nd}f}"


def pct(v, nd=1):
    return "—" if v is None else f"{v:.{nd}f} %"


def build_tables(results, probes, logs):
    """-> lista de (título, encabezados, filas, nota)"""
    T = []
    bs = _present(results)
    hdr = ["Métrica"] + [LABEL[b] for b in bs]

    def row(name, fn):
        return [name] + [fn(results[b]) for b in bs]

    if bs:
        T.append(("Loop de teleoperación/grabación", hdr, [
            row("Sesiones / episodios / frames", lambda r: f"{len(r['sessions'])} / {r['episodes']} / {r['frames']}"),
            row("Frecuencia nominal [Hz]", lambda r: f"{r['fps']:.0f}"),
            row("Frecuencia efectiva [Hz]", lambda r: f"{r['loop']['hz']:.2f}" if r["loop"]["hz"] else "—"),
            row("Periodo media ± σ [ms]", lambda r: ms_pm(r["loop"]["period"], 2)),
            row("Periodo p50 / p95 / p99 [ms]", lambda r: "—" if not r["loop"]["period"] else
                "{p50:.1f} / {p95:.1f} / {p99:.1f}".format(**r["loop"]["period"])),
            row("Periodo máx [ms]", lambda r: ms(r["loop"]["period"], "max")),
            row("Jitter σ [ms]", lambda r: f"{r['loop']['jitter_std_ms']:.2f}" if r["loop"]["jitter_std_ms"] is not None else "—"),
            row("Jitter |T − T₀| medio [ms]", lambda r: f"{r['loop']['jitter_mad_ms']:.2f}" if r["loop"]["jitter_mad_ms"] is not None else "—"),
            row(f"Frames perdidos (T > {MISSED_FRAME_FACTOR}·T₀)", lambda r: pct(r["loop"]["missed_pct"], 2)),
            row("Frames con trabajo > T₀", lambda r: pct(r["loop"]["overrun_pct"], 2)),
            row("save_episode media [s]", lambda r: ms(r.get("save_episode_ms")) if not r.get("save_episode_ms")
                else f"{r['save_episode_ms']['mean'] / 1000:.1f}"),
        ], "T₀ = 1/fps. Periodo = tiempo entre inicios de frames consecutivos dentro de un episodio."))

        T.append(("Tiempo por etapa del loop (media / p95) [ms]", hdr, [
            row(name, lambda r, s=s: "—" if not r["stages"].get(s) else
                f"{r['stages'][s]['mean']:.2f} / {r['stages'][s]['p95']:.2f}")
            for s, name in (("obs", "get_observation"), ("teleop", "teleop.get_action + procesadores"),
                            ("send", "send_action"), ("record", "add_frame"), ("work", "Total (sin sleep)"))
        ], "En MuJoCo send_action incluye la física (IK + mj_step); en Gazebo/real solo escribe en YARP."))

    yb = [b for b in ("gazebo", "real") if b in results and results[b].get("command")]
    if yb:
        h = ["Métrica"] + [LABEL[b] for b in yb]
        def crow(name, fn):
            return [name] + [fn(results[b]["command"]) for b in yb]
        T.append(("Latencia de comando cliente → módulo YARP → controlador cartesiano [ms]", h, [
            crow("Transporte media ± σ", lambda c: ms_pm(c["transport"], 2)),
            crow("Transporte p95 / p99 / máx", lambda c: "{p95:.1f} / {p99:.1f} / {max:.1f}".format(**c["transport"])),
            crow("go_to_pose_async media ± σ", lambda c: ms_pm(c["exec"], 2)),
            crow("Total hasta goToPose media ± σ", lambda c: ms_pm(c["total"], 2)),
            crow("Total p95 / p99 / máx", lambda c: "{p95:.1f} / {p99:.1f} / {max:.1f}".format(**c["total"])),
            crow("Comandos → goToPose enviado", lambda c: pct(c["sent_pct"])),
            crow("Comandos en banda muerta (<1 cm/5°)", lambda c: pct(c["deadband_pct"])),
            crow("goToPose fallido", lambda c: pct(c["failed_pct"], 2)),
            crow("Comandos no procesados (mano der.)", lambda c: pct(c.get("rh_not_processed_pct"), 2)),
        ] + ([["RPC look_at (ida y vuelta)"] + [ms_pm(results[b].get("gaze_rpc"), 2) for b in yb]]
             if any(results[b].get("gaze_rpc") for b in yb) else []),
            "Transporte = desde el write del cliente hasta que teleop_module_sm lo lee (incluye la espera "
            "a su siguiente ciclo, ≤ 20 ms). No procesados = reemplazados por uno más nuevo antes de ser leídos."))

    pb = [b for b in BACKENDS if b in probes]
    if pb:
        P = {b: analyze_probe(probes[b]) for b in pb}
        h = ["Métrica [ms]"] + [LABEL[b] for b in pb]
        def prow(name, k, nd=1):
            return [name] + [ms_pm(P[b][k], nd) for b in pb]
        T.append(("Respuesta al escalón (comando → movimiento de la mano)", h, [
            ["Escalones"] + [f"{P[b]['n']} × {float(P[b]['meta'].get('step', 0)) * 100:.1f} cm" for b in pb],
            prow("Primer movimiento (> 1 mm)", "t_move_ms"),
            prow("10 % del valor final (≈ tiempo muerto)", "t10_ms"),
            prow("50 % del valor final", "t50_ms"),
            prow("90 % del valor final", "t90_ms"),
            prow("Asentamiento (±max(2 mm, 5 %))", "t_settle_ms"),
            ["Escalones asentados en la ventana"] + [pct(P[b]["settled_pct"], 0) for b in pb],
            prow("Sobrepaso [%]", "overshoot_pct"),
            prow("Escalón logrado [%]", "reached_pct"),
            prow("Error estacionario al target [mm]", "final_err_mm", 2),
            prow("Transporte 1er comando", "cmd_transport_ms", 2),
            prow("Periodo de muestreo de la pose", "sample_dt_ms"),
        ], "Media ± σ entre escalones (σ = jitter de la respuesta). Tiempos desde la escritura del "
           "comando; 10/50/90 % y sobrepaso respecto del desplazamiento final alcanzado. MuJoCo en tiempo "
           "simulado; Gazebo/real en tiempo de pared fechado con el envelope de la pose."))

    if bs:
        def srow(name, fn):
            return [name] + [fn(results[b]["sync"]) for b in bs]
        rows = [
            srow("Antigüedad del estado articular [ms]", lambda s: ms_pm(s.get("state_age"), 2)),
            srow("Espera del estado (lectura bloqueante) [ms]", lambda s: ms_pm(s.get("state_wait"), 2)),
        ]
        cams = sorted({c for b in bs for c in camera_names(results[b]["_df"])})
        for c in cams:
            def g(s, k, c=c):
                return (s.get(f"cam_{c}") or {}).get(k)
            rows += [
                srow(f"Cámara {c}: frames nuevos / repetidos", lambda s, g=g: "—" if g(s, "new_pct") is None else
                     f"{g(s, 'new_pct'):.1f} % / {g(s, 'repeated_pct'):.1f} %"),
                srow(f"Cámara {c}: frames perdidos en origen", lambda s, g=g: pct(g(s, "dropped_pct"), 2)),
                srow(f"Cámara {c}: fps fuente / vistos por el loop", lambda s, g=g: "—" if g(s, "effective_fps") is None else
                     (f"{g(s, 'source_fps'):.1f} / " if g(s, "source_fps") else "— / ") + f"{g(s, 'effective_fps'):.1f}"),
                srow(f"Cámara {c}: antigüedad desde llegada [ms]", lambda s, g=g: ms_pm(g(s, "age_arrival"), 1)),
                srow(f"Cámara {c}: antigüedad desde captura [ms]", lambda s, g=g: ms_pm(g(s, "age_capture"), 1)
                     if g(s, "same_clock") else ("reloj distinto" if g(s, "same_clock") is False else "—")),
                srow(f"Cámara {c}: desfase captura − estado [ms]", lambda s, g=g: ms_pm(g(s, "skew_vs_state"), 1)),
            ]
        rows += [
            srow("Desfase estéreo |izq − der| media / máx [ms]", lambda s: "—" if not s.get("stereo_skew") else
                 f"{s['stereo_skew']['mean']:.1f} / {s['stereo_skew']['max']:.1f}"),
            srow("Desfase cámara − estado (t. simulado) [ms]", lambda s: ms(s.get("cam_vs_state_sim_ms"), "max", 3)),
            srow("Render cámaras [ms]", lambda s: ms_pm(s.get("render"), 2)),
            srow("Física por frame [ms]", lambda s: ms_pm(s.get("physics"), 2)),
            srow("Tiempo simulado / tiempo real", lambda s: f"{s['sim_time_ratio']:.3f}" if s.get("sim_time_ratio") else "—"),
        ]
        T.append(("Sincronización cámara / estado / acción", hdr, rows,
                  "Repetido = el frame grabado reutiliza la imagen del frame anterior (no llegó una nueva). "
                  "Antigüedad desde captura y desfases solo se calculan si la fuente comparte reloj con el PC."))

        tb = [b for b in bs if results[b].get("tracking")]
        if tb:
            h = ["Métrica"] + [LABEL[b] for b in tb]
            T.append(("Seguimiento de la mano derecha durante la teleoperación", h, [
                ["Error comandado − medido media ± σ [cm]"] + [ms_pm(results[b]["tracking"]["error_cm"], 2) for b in tb],
                ["Error p95 [cm]"] + [ms(results[b]["tracking"]["error_cm"], "p95", 2) for b in tb],
                ["Retardo por correlación cruzada mediana [ms]"] + [ms(results[b]["tracking"]["lag_ms"], "p50", 0) for b in tb],
                ["Episodios usados para el retardo"] + [str((results[b]["tracking"]["lag_ms"] or {}).get("n", 0)) for b in tb],
            ], "El error incluye el retardo de seguimiento. Retardo = desplazamiento que maximiza la correlación "
               "entre las velocidades comandada y medida (episodios con > 5 mm de movimiento)."))

    if logs:
        h = ["Log", "Robot", "Hz (media / mín)", "Peor ciclo p95 / máx [ms]", "Cámaras fps izq / der",
             "[SLOW]", "IK fallidos", "Mano−target mediana [cm]"]
        T.append(("Módulo YARP teleop_module_sm (desde los logs del Hub)", h, [[
            l["file"], LABEL.get(l["robot"], l["robot"]),
            "—" if not l["hz"] else f"{l['hz']['mean']:.1f} / {l['hz']['min']:.1f} (obj. {l['target_hz']})",
            "—" if not l["worst_cycle_ms"] else f"{l['worst_cycle_ms']['p95']:.0f} / {l['worst_cycle_ms']['max']:.0f}",
            f"{ms(l['cam_left_fps'])} / {ms(l['cam_right_fps'])}",
            str(l["slow_lines"]), pct(l["ik_failed_pct"], 2), ms(l["hand_target_cm"], "p50", 1),
        ] for l in logs], "Agregados cada 5 s que imprime el propio módulo; complementan las métricas por frame."))
    return T


def to_markdown(T, figs):
    out = ["# Métricas de latencia, jitter y sincronización", ""]
    for title, h, rows, note in T:
        out += [f"## {title}", "", "| " + " | ".join(h) + " |", "|" + "---|" * len(h)]
        out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
        out += ["", f"_{note}_", ""]
    if figs:
        out += ["## Figuras", ""] + [f"![{p.stem}]({p.name})" for p in figs] + [""]
    return "\n".join(out)


def _tex(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("%", r"\%"), ("_", r"\_"), ("&", r"\&"), ("#", r"\#"),
                 ("±", r"$\pm$"), ("σ", r"$\sigma$"), ("≤", r"$\leq$"), ("→", r"$\rightarrow$"),
                 ("−", "--"), ("T₀", r"$T_0$"), ("·", r"$\cdot$"), ("×", r"$\times$")):
        s = s.replace(a, b)
    return s


def to_latex(T):
    out = ["% Generado por metrics/metrics_report.py — requiere \\usepackage{booktabs}", ""]
    for title, h, rows, note in T:
        out += [r"\begin{table}[htbp]", r"\centering\small", f"\\caption{{{_tex(title)}}}",
                r"\begin{tabular}{l" + "c" * (len(h) - 1) + "}", r"\toprule",
                " & ".join(_tex(c) for c in h) + r" \\", r"\midrule"]
        out += [" & ".join(_tex(c) for c in r) + r" \\" for r in rows]
        out += [r"\bottomrule", r"\end{tabular}", f"\\par\\footnotesize {_tex(note)}", r"\end{table}", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--frames", nargs="*", help="Archivos .jsonl (default: todos en hub_dataset/logs/metrics)")
    ap.add_argument("--probes", nargs="*", help="Archivos probe_*_trials.csv (default: todos)")
    ap.add_argument("--logs", nargs="*", default=None, help="Logs del Hub hub_*.txt (default: ninguno)")
    ap.add_argument("--include-waiting", action="store_true", help="Incluir frames fuera de grabación")
    ap.add_argument("--out", default=str(METRICS_DIR / "report"))
    a = ap.parse_args()

    frame_paths = a.frames if a.frames is not None else sorted(glob.glob(str(METRICS_DIR / "*.jsonl")))
    probe_paths = a.probes if a.probes is not None else sorted(glob.glob(str(METRICS_DIR / "probe_*_trials.csv")))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    data = load_frames(frame_paths)
    probes = load_probes(probe_paths)
    logs = analyze_logs(a.logs) if a.logs else []

    results, frames = {}, {}
    for b, d in data.items():
        r, df = analyze_backend(b, d, a.include_waiting)
        r["sessions"] = d["sessions"]
        r["_df"] = df
        results[b], frames[b] = r, df

    figs = [f for f in (
        fig_period(results, frames, out), fig_period_series(results, frames, out),
        fig_stages(results, out), fig_command(frames, out), fig_step(probes, out),
        fig_camera_age(results, frames, out), fig_tracking(results, frames, out),
    ) if f is not None]

    T = build_tables(results, probes, logs)
    (out / "resumen.md").write_text(to_markdown(T, figs), encoding="utf-8")
    (out / "tablas.tex").write_text(to_latex(T), encoding="utf-8")
    dump = {b: {k: v for k, v in r.items() if k != "_df"} for b, r in results.items()}
    dump["probes"] = {b: analyze_probe(p) for b, p in probes.items()}
    dump["logs"] = logs
    (out / "resumen.json").write_text(json.dumps(dump, indent=2, default=str), encoding="utf-8")

    sessions = ", ".join(f"{LABEL.get(b, b)}={len(d['sessions'])}" for b, d in data.items())
    probe_names = ", ".join(LABEL.get(b, b) for b in probes)
    print(f"Sesiones: {sessions or 'ninguna'} | pruebas de escalón: {probe_names or 'ninguna'} | logs: {len(logs)}")
    print(f"Informe: {out / 'resumen.md'}\nLaTeX:   {out / 'tablas.tex'}\nFiguras: {len(figs)} en {out}")


if __name__ == "__main__":
    main()
