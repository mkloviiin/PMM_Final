"""Prueba aislada del controlador cartesiano (sin teleop_module_sm ni el Hub).

Sube la mano 5 cm (eje z del frame root), espera y vuelve a la pose inicial, usando
solo la ICartesianControl estándar: no toca modos de control ni hace stop().
Sirve para saber si un problema de movimiento es del robot o de nuestro código.

Uso (con el YARP module DETENIDO y alguien junto al robot):
    conda activate icubenv
    python test_cartesian.py --robot icub --arm right_arm [--dz 0.05]
"""

import argparse
import time

import numpy as np
import yarp


def control_modes(robot, arm):
    p = yarp.Property()
    p.put("device", "remote_controlboard")
    p.put("local", f"/test_cartesian/{arm}")
    p.put("remote", f"/{robot}/{arm}")
    d = yarp.PolyDriver(p)
    if not d.isValid():
        return d, lambda: "n/a"
    cm, n = d.viewIControlMode(), d.viewIEncoders().getAxes()

    def read():
        m = yarp.VectorInt(n)
        cm.getControlModes(m.data())
        dec = lambda v: bytes((v >> (8 * k)) & 0xFF for k in range(4)).rstrip(b"\0").decode()
        return [dec(m[i]) for i in range(7)]
    time.sleep(0.3)
    return d, read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="icub")
    ap.add_argument("--arm", default="right_arm")
    ap.add_argument("--dz", type=float, default=0.05, help="desplazamiento en z [m]")
    args = ap.parse_args()

    yarp.Network.init()
    p = yarp.Property()
    p.put("device", "cartesiancontrollerclient")
    p.put("local", f"/test_cartesian/cart/{args.arm}")
    p.put("remote", f"/{args.robot}/cartesianController/{args.arm}")
    p.put("timeout", 10.0)
    drv = yarp.PolyDriver(p)
    if not drv.isValid():
        raise SystemExit("No se pudo abrir el cliente cartesiano (¿controlador colgado o sin solver?).")
    ic = drv.viewICartesianControl()
    board, modes = control_modes(args.robot, args.arm)

    x0, o0 = yarp.Vector(3), yarp.Vector(4)
    time.sleep(0.5)
    if not ic.getPose(x0, o0):
        raise SystemExit("getPose falló.")
    start = np.array([x0[i] for i in range(3)])
    print(f"Pose inicial: {np.round(start, 3).tolist()}   modos j0-6: {modes()}")

    input(f"ENTER para mover la mano {args.dz * 100:.0f} cm en z (Ctrl+C para cancelar)... ")
    ic.setTrajTime(2.0)
    xd = yarp.Vector(3)
    for i in range(3):
        xd.set(i, start[i] + (args.dz if i == 2 else 0.0))
    print("goToPoseSync:", ic.goToPoseSync(xd, o0))
    t0 = time.time()
    while time.time() - t0 < 6.0:
        time.sleep(0.5)
        ic.getPose(x0, o0)
        cur = np.array([x0[i] for i in range(3)])
        print(f"  t={time.time() - t0:.1f}s  movido {np.linalg.norm(cur - start) * 100:.1f} cm  "
              f"modos j0-6: {modes()}")
    moved = np.linalg.norm(cur - start)
    print("RESULTADO:", "SE MOVIO" if moved > 0.5 * abs(args.dz) else "NO SE MOVIO",
          f"({moved * 100:.1f} cm de {abs(args.dz) * 100:.0f} cm)")

    print("Volviendo a la pose inicial...")
    xd = yarp.Vector(3)
    for i in range(3):
        xd.set(i, start[i])
    ic.goToPoseSync(xd, o0)
    time.sleep(3.0)
    ic.stopControl()
    drv.close()
    board.close()


if __name__ == "__main__":
    main()
