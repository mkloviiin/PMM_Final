#!/usr/bin/env python3
"""Prueba de escalón: latencia comando → respuesta del robot (MuJoCo, Gazebo, real).

Manda a la mano derecha un desplazamiento pequeño (``--step``, 2 cm por defecto) a
lo largo de un eje, lo mantiene, vuelve al punto de partida, y repite ``--reps``
veces. Los comandos pasan por el mismo camino que en la teleoperación (``send_action``
del plugin LeRobot: en Gazebo/real, port YARP → teleop_module_sm → cartesianController).
De la posición medida de la mano se obtiene, por escalón:

  * t_move   : primer movimiento detectable (> ``--move-thresh``, 1 mm)  ← latencia
  * t10/t50/t90 : 10 / 50 / 90 % del desplazamiento FINAL alcanzado (definición
                  clásica; t10 ≈ tiempo muerto, t90 − t10 = tiempo de subida)
  * t_settle : desde cuándo queda a menos de max(``--settle-tol``, 5 %) del valor final
  * overshoot (sobre el valor final), % del escalón logrado y error estacionario
    (distancia final al target: en Gazebo/real la mano suele quedar a ~1 cm)
  * cmd_transport_ms / cmd_exec_ms (solo YARP): latencia del primer comando hasta
    el módulo y lo que tarda go_to_pose_async (de /teleop/cmd_ack:o)

Todos los tiempos se miden desde el instante en que se escribió el primer comando
del escalón. En Gazebo/real las muestras se fechan con el envelope de la pose
(instante en que el módulo la leyó del controlador) y el cruce de cada umbral se
interpola linealmente entre muestras. En MuJoCo se usa el tiempo simulado
(``data.time``); como el loop corre a ``--rate`` con ``control_dt = 1/rate``, coincide
con el tiempo real.

Requisitos:
  * mujoco : nada (carga la escena en este proceso, sin visor).
  * gazebo : start_sim.sh + teleop_module_sm.py --robot icubSim corriendo.
  * real   : yarp conf al robot + teleop_module_sm.py --robot icub corriendo.

Salida (en hub_dataset/logs/metrics/): probe_<backend>_<fecha>_samples.csv (todas
las muestras), probe_<backend>_<fecha>_trials.csv (una fila por escalón) y un
resumen por consola. ``metrics_report.py`` los incluye en el informe.

Ejemplos:
  python metrics/latency_probe.py --backend mujoco
  python metrics/latency_probe.py --backend gazebo --reps 10
  python metrics/latency_probe.py --backend real --step 0.02 --axis z
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GZ_ROOT = ROOT / "icub-lerobot-gazebo"
MJ_ROOT = ROOT / "icub-lerobot-mj"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from frame_metrics import METRICS_DIR  # noqa: E402

AXES = {"x": 0, "y": 1, "z": 2}
MAX_STEP_M = 0.05


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--backend", required=True, choices=["mujoco", "gazebo", "real"])
    p.add_argument("--reps", type=int, default=10, help="Pares ida/vuelta (default 10 → 20 escalones)")
    p.add_argument("--step", type=float, default=0.02, help=f"Desplazamiento en m (máx {MAX_STEP_M})")
    p.add_argument("--axis", choices=list(AXES), default="z", help="Eje del root frame (default z = arriba)")
    p.add_argument("--sign", type=int, choices=[1, -1], default=1, help="Sentido del primer escalón")
    p.add_argument("--rate", type=float, default=30.0, help="Frecuencia de envío de comandos (Hz), como el teleop")
    p.add_argument("--hold", type=float, default=1.5, help="Pausa quieto antes de cada escalón (s)")
    p.add_argument("--hold-jitter", type=float, default=1.2,
                   help="Pausa extra aleatoria U(0, x) s: muestrea la fase de procesos periódicos "
                        "(p. ej. la ventana de IK de MuJoCo) en vez de caer siempre en la misma")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window", type=float, default=4.0, help="Ventana de medición tras cada escalón (s)")
    p.add_argument("--move-thresh", type=float, default=0.001, help="Umbral de movimiento detectable (m)")
    p.add_argument("--settle-tol", type=float, default=0.002, help="Tolerancia de asentamiento (m)")
    p.add_argument("--config", default=None, help="control_config.yaml (default: el del backend)")
    p.add_argument("--model", default=None, help="MuJoCo: escena XML (default: primera de scenes.yaml)")
    p.add_argument("--no-render", action="store_true", help="MuJoCo: no renderizar cámaras en cada frame")
    p.add_argument("--yes", action="store_true", help="Real: no pedir confirmación")
    p.add_argument("--out", default=str(METRICS_DIR))
    a = p.parse_args()
    if not 0 < a.step <= MAX_STEP_M:
        p.error(f"--step debe estar en (0, {MAX_STEP_M}] m")
    return a


# ── Adaptadores: misma interfaz para MuJoCo y YARP ───────────────────────────
class _Adapter:
    """read() -> (t, pos[3], quat[4] wxyz, extra); send(pos, quat) -> t_send."""

    time_base = "wall"

    def __init__(self):
        self.robot = None
        self.other_arm: dict = {}

    def _action(self, pos, quat) -> dict:
        a = {f"rh_pos_{c}": float(v) for c, v in zip("xyz", pos)}
        a.update({f"rh_quat_{c}": float(v) for c, v in zip("wxyz", quat)})
        a["rh_gripper"] = 1.0  # 1.0 = mano quieta
        a.update(self.other_arm)
        return a

    def close(self):
        if self.robot is not None and self.robot.is_connected:
            self.robot.disconnect()


class YarpAdapter(_Adapter):
    def __init__(self, args):
        super().__init__()
        os.environ["ICUB_ROBOT_NAME"] = "icub" if args.backend == "real" else "icubSim"
        sys.path.insert(0, str(GZ_ROOT / "common" / "lerobot-robot-icubrg"))
        from lerobot_robot_icubrg.config_icub import iCubConfig
        from lerobot_robot_icubrg.icub import iCub

        cfg = iCubConfig(config_path=str(args.config or GZ_ROOT / "config" / "control_config.yaml"))
        cfg.camera_ports = {}   # sin cámaras: solo interesa el brazo
        cfg.use_gaze = False    # la mirada no se toca
        self.robot = iCub(cfg)
        self.robot.connect()
        reply = self.robot.send_rpc_command("start_teleop")
        print(f"[probe] start_teleop -> {reply}")
        self._last_pose_t = None
        self.acks: dict[int, dict] = {}

        # Esperar la primera pose de la mano (y la de la otra mano si se controla)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            obs = self.robot.get_observation()
            if "rh_current_pose" in obs:
                break
        else:
            raise RuntimeError("No llega /teleop/rh_current_pose:o: ¿está corriendo teleop_module_sm?")
        if "lh_current_pose" in obs:
            lh = obs["lh_current_pose"]
            self.other_arm = {**{f"lh_pos_{c}": lh[i] for i, c in enumerate("xyz")},
                              **{f"lh_quat_{c}": lh[3 + i] for i, c in enumerate("wxyz")}}

    def initial_target(self):
        t, pos, quat, _ = self.read(wait_new=False)
        return pos, quat

    def read(self, wait_new=True):
        """Bloquea hasta el siguiente estado del módulo (~50 Hz)."""
        while True:
            obs = self.robot.get_observation()
            meta = self.robot.metrics_meta
            for ack in meta.get("acks") or ():
                if ack["arm"] == "rh":
                    self.acks[ack["seq"]] = ack
            t = meta.get("rh_pose_stamp")
            pose = obs.get("rh_current_pose")
            if pose is None:
                continue
            if t is None:
                t = time.time()
            if not wait_new or t != self._last_pose_t:
                self._last_pose_t = t
                return t, np.array(pose[:3]), np.array(pose[3:7]), {"state_age_ms": meta.get("state_age_ms")}

    def send(self, pos, quat):
        t = time.time()
        self.robot.send_action(self._action(pos, quat))
        return t, self.robot.metrics_meta.get("rh_cmd_seq")

    def pace(self, t_loop_start, period):
        pass  # read() ya bloquea al ritmo del módulo


class MujocoAdapter(_Adapter):
    time_base = "sim"

    def __init__(self, args):
        super().__init__()
        import yaml
        os.environ["ICUB_MUJOCO_VR_ENABLED"] = "0"
        for p in (MJ_ROOT / "lerobot-robot-icub", MJ_ROOT):
            sys.path.insert(0, str(p))
        from lerobot_robot_icub.config_icub_mujoco import iCubMuJoCoConfig
        from lerobot_robot_icub.icub_mujoco import iCubMuJoCo

        scenes_dir = MJ_ROOT / "dependencies" / "assets" / "scenes"
        model = args.model
        if model is None:
            entries = yaml.safe_load((scenes_dir / "scenes.yaml").read_text(encoding="utf-8")) or []
            model = scenes_dir / entries[0]["file"]
        cfg = iCubMuJoCoConfig(
            model_path=str(model),
            config_path=str(args.config or MJ_ROOT / "utils" / "control_config.yaml"),
            vr_enabled=False,
        )
        cfg.vr_enabled = False  # __post_init__ lo toma del YAML
        cfg.control_dt = 1.0 / args.rate
        if args.no_render:
            cfg.camera_names = []
        self.robot = iCubMuJoCo(cfg)
        self.robot.connect()
        print(f"[probe] MuJoCo scene: {model}")
        self._obs = self.robot.get_observation()
        o = self._obs
        if cfg.control_arms in ("left", "both"):
            self.other_arm = {**{f"lh_pos_{c}": float(o[f"lh_target_pos_{c}"]) for c in "xyz"},
                              **{f"lh_quat_{c}": float(o[f"lh_target_quat_{c}"]) for c in "wxyz"}}

    def initial_target(self):
        o = self._obs
        return (np.array([float(o[f"rh_target_pos_{c}"]) for c in "xyz"]),
                np.array([float(o[f"rh_target_quat_{c}"]) for c in "wxyz"]))

    def read(self, wait_new=True):
        o = self.robot.get_observation()
        t = float(self.robot.teleop_core.data.time)
        return (t, np.array([float(o[f"rh_ee_pos_{c}"]) for c in "xyz"]),
                np.array([float(o[f"rh_ee_quat_{c}"]) for c in "wxyz"]),
                {"render_ms": self.robot.metrics_meta.get("render_ms"), "wall": time.time()})

    def send(self, pos, quat):
        # La acción se aplica en el instante simulado actual y luego avanza control_dt.
        t = float(self.robot.teleop_core.data.time)
        self.robot.send_action(self._action(pos, quat))
        return t, None

    def pace(self, t_loop_start, period):
        rest = period - (time.perf_counter() - t_loop_start)
        if rest > 0:
            time.sleep(rest)


# ── Análisis de un escalón ───────────────────────────────────────────────────
def _crossing(t, y, level):
    """Primer instante (interpolado) en que y >= level; None si nunca."""
    idx = np.nonzero(y >= level)[0]
    if idx.size == 0:
        return None
    i = idx[0]
    if i == 0:
        return t[0]
    t0, t1, y0, y1 = t[i - 1], t[i], y[i - 1], y[i]
    return t0 + (level - y0) * (t1 - t0) / (y1 - y0) if y1 != y0 else t1


def analyze_step(samples, t_cmd, start_pos, delta, move_thresh, settle_tol):
    L = float(np.linalg.norm(delta))
    u = delta / L
    t = np.array([s["t"] for s in samples]) - t_cmd
    pos = np.array([[s["x"], s["y"], s["z"]] for s in samples])
    along = (pos - start_pos) @ u                        # m recorridos en la dirección del escalón
    goal = start_pos + delta
    res = {"t_move_ms": _crossing(t, along, move_thresh), "t10_ms": None, "t50_ms": None,
           "t90_ms": None, "t_settle_ms": None, "overshoot_pct": None, "reached_pct": None,
           "final_err_mm": None}
    if along.size < 3:
        return res
    # Valor final = media del último 0,5 s de la ventana
    final = float(along[t >= t[-1] - 0.5].mean())
    res["reached_pct"] = final / L * 100.0
    res["final_err_mm"] = float(np.linalg.norm(pos[-1] - goal) * 1000.0)
    if final > move_thresh:
        for pc in (10, 50, 90):
            res[f"t{pc}_ms"] = _crossing(t, along, pc / 100 * final)
        res["overshoot_pct"] = max(0.0, (along.max() / final - 1.0) * 100.0)
        # Asentado = dentro de la banda desde ese instante hasta el final de la ventana
        band = max(settle_tol, 0.05 * final)
        outside = np.nonzero(np.abs(along - final) > band)[0]
        if not outside.size:
            res["t_settle_ms"] = t[0]
        elif outside[-1] + 1 < len(t):
            res["t_settle_ms"] = t[outside[-1] + 1]
    for k in ("t_move_ms", "t10_ms", "t50_ms", "t90_ms", "t_settle_ms"):
        if res[k] is not None:
            res[k] = float(res[k] * 1000.0)
    return res


def _fmt(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return "n/a"
    return f"{a.mean():7.1f} ± {a.std():5.1f}  (min {a.min():.1f}, max {a.max():.1f}, n={a.size})"


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    if args.backend == "real" and not args.yes:
        print(f"*** ROBOT REAL: la mano derecha se moverá {args.step * 100:.1f} cm en "
              f"{'+' if args.sign > 0 else '-'}{args.axis}, {args.reps * 2} veces. "
              f"Despeja el espacio y ten el paro de emergencia a mano.")
        if input("Escribe 'si' para continuar: ").strip().lower() not in ("si", "sí", "s", "yes", "y"):
            sys.exit("Cancelado.")

    adapter = MujocoAdapter(args) if args.backend == "mujoco" else YarpAdapter(args)
    period = 1.0 / args.rate
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"probe_{args.backend}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    samples_path = out_dir / f"{stem}_samples.csv"
    trials_path = out_dir / f"{stem}_trials.csv"

    p0, q0 = adapter.initial_target()
    delta = np.zeros(3)
    delta[AXES[args.axis]] = args.sign * args.step
    print(f"[probe] backend={args.backend} start={np.round(p0, 4).tolist()} "
          f"step={np.round(delta, 3).tolist()} m, {args.reps} reps, rate {args.rate:.0f} Hz")

    all_samples, trials = [], []
    rng = np.random.default_rng(args.seed)

    def run_phase(target, duration, trial, phase):
        """Envía `target` a `rate` Hz durante `duration` s leyendo la pose; devuelve
        (t_cmd, cmd_seq, muestras)."""
        samples, t_cmd, seq = [], None, None
        t_end = time.perf_counter() + duration
        next_send = time.perf_counter()
        while time.perf_counter() < t_end:
            t_loop = time.perf_counter()
            if t_loop >= next_send:
                ts, sq = adapter.send(target, q0)
                if t_cmd is None:
                    t_cmd, seq = ts, sq
                next_send += period
            t, pos, quat, extra = adapter.read()
            s = {"trial": trial, "phase": phase, "t": t, "x": pos[0], "y": pos[1], "z": pos[2],
                 "tx": target[0], "ty": target[1], "tz": target[2], **extra}
            samples.append(s)
            adapter.pace(t_loop, period)
        all_samples.extend(samples)
        return t_cmd, seq, samples

    try:
        run_phase(p0, 2.0, -1, "init")  # asentarse en el punto de partida
        current = p0
        for k in range(args.reps * 2):
            target = p0 + delta if k % 2 == 0 else p0
            _, _, hold = run_phase(current, args.hold + rng.uniform(0, args.hold_jitter), k, "hold")
            start = np.mean([[s["x"], s["y"], s["z"]] for s in hold[-5:]], axis=0)
            t_cmd, seq, step_samples = run_phase(target, args.window, k, "step")
            # Solo muestras posteriores al comando (la primera puede ser previa)
            post = [s for s in step_samples if s["t"] >= t_cmd]
            res = analyze_step(post, t_cmd, start, target - current,
                               args.move_thresh, args.settle_tol)
            ack = getattr(adapter, "acks", {}).get(seq) if seq is not None else None
            res.update({
                "trial": k, "direction": "out" if k % 2 == 0 else "back", "t_cmd": t_cmd,
                **{f"start_{c}": float(v) for c, v in zip("xyz", start)},
                **{f"delta_{c}": float(v) for c, v in zip("xyz", target - current)},
                "step_mm": float(np.linalg.norm(target - current) * 1000.0),
                "cmd_transport_ms": ack["transport_ms"] if ack else None,
                "cmd_exec_ms": ack["exec_ms"] if ack else None,
                "cmd_sent": ack["sent"] if ack else None,
                "sample_dt_ms": float(np.median(np.diff([s["t"] for s in post])) * 1000.0) if len(post) > 1 else None,
            })
            trials.append(res)
            print(f"[probe] step {k + 1:2d}/{args.reps * 2} ({res['direction']:4s}): "
                  f"move {res['t_move_ms'] or float('nan'):6.1f} ms, t10 {res['t10_ms'] or float('nan'):6.1f}, "
                  f"t90 {res['t90_ms'] or float('nan'):7.1f}, settle {res['t_settle_ms'] or float('nan'):7.1f} ms, "
                  f"final err {res['final_err_mm']:.1f} mm"
                  + (f", cmd transport {res['cmd_transport_ms']:.1f} ms" if res["cmd_transport_ms"] is not None else ""),
                  flush=True)
            current = target
    except KeyboardInterrupt:
        print("\n[probe] interrumpido: volviendo al punto de partida")
        try:
            run_phase(p0, 1.0, -1, "abort")
        except Exception:  # noqa: BLE001
            pass
    finally:
        adapter.close()

    if all_samples:
        keys = list(dict.fromkeys(k for s in all_samples for k in s))
        with open(samples_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_samples)
    if trials:
        keys = ["trial", "direction", "step_mm", "t_move_ms", "t10_ms", "t50_ms", "t90_ms",
                "t_settle_ms", "overshoot_pct", "reached_pct", "final_err_mm", "cmd_transport_ms",
                "cmd_exec_ms", "cmd_sent", "sample_dt_ms", "t_cmd",
                "start_x", "start_y", "start_z", "delta_x", "delta_y", "delta_z"]
        with open(trials_path, "w", newline="") as f:
            f.write(f"# backend={args.backend} time_base={adapter.time_base} rate={args.rate} "
                    f"step={args.step} axis={args.axis} move_thresh={args.move_thresh} "
                    f"settle_tol={args.settle_tol}\n")
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(trials)

        print("\n" + "=" * 78)
        print(f"Respuesta al escalón — {args.backend} ({len(trials)} escalones de {args.step * 100:.1f} cm, "
              f"tiempo {'simulado' if adapter.time_base == 'sim' else 'real'}) [ms]")
        for k, label in (("t_move_ms", "primer movimiento"), ("t10_ms", "10 % del final"),
                         ("t50_ms", "50 %"), ("t90_ms", "90 %"), ("t_settle_ms", "asentamiento"),
                         ("cmd_transport_ms", "transporte comando"), ("cmd_exec_ms", "go_to_pose_async")):
            print(f"  {label:22s} {_fmt([t[k] for t in trials])}")
        print(f"  {'escalón logrado [%]':22s} {_fmt([t['reached_pct'] for t in trials])}")
        print(f"  {'error final [mm]':22s} {_fmt([t['final_err_mm'] for t in trials])}")
        print(f"  {'sobrepaso [%]':22s} {_fmt([t['overshoot_pct'] for t in trials])}")
        print(f"  {'periodo de muestreo':22s} {_fmt([t['sample_dt_ms'] for t in trials])}")
        print(f"\n  muestras: {samples_path}\n  escalones: {trials_path}")


if __name__ == "__main__":
    main()
