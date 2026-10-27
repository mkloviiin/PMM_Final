#!/bin/bash
# Cierra la simulación lanzada por start_sim.sh (sesión tmux icub_sim).

SESSION_NAME="icub_sim"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")"
PIXI_ENV="${PIXI_ENV:-$PROJECT_ROOT/.pixi/envs/default}"

echo "Iniciando secuencia de cierre para la sesión de tmux '$SESSION_NAME'..."

# 1. Cierra Gazebo y fuerza la terminación de sus procesos
echo "Cerrando Gazebo..."
tmux send-keys -t $SESSION_NAME:gazebo.0 C-c 2>/dev/null
tmux send-keys -t $SESSION_NAME:gazebo.1 C-c 2>/dev/null
sleep 1
echo "Asegurando el cierre forzoso del servidor y cliente de Gazebo..."
pkill -9 -f "gz sim"
sleep 2

# 2. Cierra el 'yarpserver' y limpia YARP
echo "Cerrando yarpserver y limpiando YARP..."
tmux send-keys -t $SESSION_NAME:server C-c 2>/dev/null
sleep 1
PATH="$PIXI_ENV/bin:$PATH" yarp clean --timeout 0.5 >/dev/null 2>&1

# 3. Cierra la sesión de tmux completa
echo "Cerrando la sesión de tmux..."
tmux kill-session -t $SESSION_NAME 2>/dev/null
rm -f /tmp/icub_gazebo_model

echo "Cierre completado."
