#!/usr/bin/env python3
"""Muestra lo que manda la app BeaVR del Quest a los puertos del teleop, para saber qué datos
llegan (botones, formato de la cabeza y de los controles).

Escucha los mismos puertos que teleop_mujoco.py / el teleoperador de Gazebo (cabeza 8115,
mano derecha 8117, botones 8119, mano izquierda 8121), así que el Hub o el teleop tienen que
estar cerrados. Imprime:
  - cada mensaje de botones distinto del anterior (apretar y soltar X, Y, A, B, gatillos...);
  - el formato de los mensajes de cabeza y de manos la primera vez y cada vez que cambia el
    número de campos (p. ej. si la cabeza trae posición además de orientación);
  - cada 5 s, cuántos mensajes llegaron por puerto (si la app deja de mandar al pausar).

Uso (con BeaVR conectado por Wi-Fi o cable; Ctrl+C para salir):
    python -m dependencies.vr_sniffer
"""

from __future__ import annotations

import time

import zmq

PORTS = {"head": 8115, "rh": 8117, "btn": 8119, "lh": 8121}


def shape(msg: str) -> str:
    """Formato del mensaje sin los números: claves JSON o cantidad de campos CSV."""
    msg = msg.strip()
    if msg.startswith("{"):
        return "json " + ",".join(sorted(k.strip('"') for k in msg.strip("{}").split(",") for k in [k.split(":")[0]]))
    if ":" in msg and "," not in msg:
        return "clave:valor"
    return f"csv con {len(msg.split(','))} campos"


def main() -> None:
    ctx = zmq.Context()
    socks = {}
    for name, port in PORTS.items():
        s = ctx.socket(zmq.PULL)
        s.setsockopt(zmq.LINGER, 0)
        s.bind(f"tcp://*:{port}")
        socks[name] = s
    print("Escuchando", PORTS, "- aprieta los botones del control (X, Y, A, B, gatillos)...")
    last_btn, shapes, counts, t_count = None, {}, {k: 0 for k in PORTS}, time.time()
    try:
        while True:
            for name, s in socks.items():
                try:
                    while True:
                        msg = s.recv_string(flags=zmq.NOBLOCK)
                        counts[name] += 1
                        if name == "btn":
                            if msg != last_btn:
                                print(f"[{time.strftime('%H:%M:%S')}] botones: {msg.strip()}")
                                last_btn = msg
                        else:
                            sh = shape(msg)
                            if shapes.get(name) != sh:
                                shapes[name] = sh
                                print(f"[{time.strftime('%H:%M:%S')}] {name}: {sh} | ej.: {msg.strip()[:160]}")
                except zmq.Again:
                    pass
            if time.time() - t_count >= 5.0:
                print(f"   mensajes en 5 s: {counts}")
                counts, t_count = {k: 0 for k in PORTS}, time.time()
            time.sleep(0.002)
    except KeyboardInterrupt:
        pass
    finally:
        for s in socks.values():
            s.close()
        ctx.term()


if __name__ == "__main__":
    main()
