#!/bin/bash

SESSION_NAME="icub_sim"

echo "Iniciando secuencia de cierre para la sesión de tmux '$SESSION_NAME'..."

# 1. Cierra los controladores de alto nivel (brazos y mirada)
echo "Cerrando controladores (brazos y mirada)..."
tmux send-keys -t $SESSION_NAME:right_arm C-c
tmux send-keys -t $SESSION_NAME:left_arm C-c
tmux send-keys -t $SESSION_NAME:gaze C-c

# 2. Cierra el 'robotinterface'
echo "Cerrando yarprobotinterface..."
tmux send-keys -t $SESSION_NAME:robotinterface C-c

# 3. Cierra Gazebo y fuerza la terminación de sus procesos
echo "Cerrando Gazebo..."
tmux send-keys -t $SESSION_NAME:gazebo C-c
sleep 1 # Pequeña pausa
echo "Asegurando el cierre forzoso del servidor y cliente de Gazebo..."
pkill -9 -f "gz sim"

# Pausa para asegurar que los procesos terminen
sleep 2

# 4. Cierra el 'yarpserver' y limpia YARP
echo "Cerrando yarpserver y limpiando YARP..."
tmux send-keys -t $SESSION_NAME:server C-c
sleep 1 # Pausa para que el servidor muera antes de limpiar
tmux send-keys -t $SESSION_NAME:server "yarp clean" C-m
sleep 1

# 5. Cierra la sesión de tmux completa
echo "Cerrando la sesión de tmux..."
tmux kill-session -t $SESSION_NAME

echo "Cierre completado."