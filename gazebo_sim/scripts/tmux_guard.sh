#!/bin/bash
# Ejecuta un comando (gz sim -s / gz sim -g) y lo cierra cuando se cierra su panel de tmux.
#
# gz sim no se cierra con SIGHUP: el lanzador Ruby de gz captura la señal pero nunca la atiende,
# porque el intérprete queda bloqueado dentro del bucle C++ del servidor; Gazebo solo se cierra con
# SIGINT o SIGTERM. Por eso cerrar la sesión tmux sin stop_sim.sh (tmux kill-session, kill-server,
# cerrar la terminal) dejaba el servidor huérfano simulando: el 2026-11-05 quedó 4 días al ~290 %
# de CPU. Este guardián recibe el SIGHUP que manda tmux al cerrar el panel (o el Ctrl+C del panel)
# y se lo pasa al comando como SIGTERM, que Gazebo atiende y con el que se cierra limpio.
#
# Uso (lo llama start_sim.sh): tmux_guard.sh <comando> [args...]

"$@" &
pid=$!

# SIGTERM y no SIGINT: bash deja SIGINT ignorado en los comandos lanzados con '&' (sin control de
# trabajos), así que el comando podría no recibirlo.
trap 'kill -TERM "$pid" 2>/dev/null' HUP INT TERM

# wait vuelve en cuanto llega una señal atrapada: seguir esperando hasta que el comando termine.
status=0
while kill -0 "$pid" 2>/dev/null; do
    wait "$pid"
    status=$?
done
exit "$status"
