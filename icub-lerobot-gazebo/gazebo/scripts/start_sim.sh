#!/bin/bash

SESSION_NAME="icub_sim"
CONTEXT="gazeboCartesianControl"
CONDA_ENV="icubenv"
GAZEBO_EXAMPLE_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")"
WORLD="$GAZEBO_EXAMPLE_DIR/worlds/icub_world.sdf"
MODELS_DIR="$GAZEBO_EXAMPLE_DIR/models"
PLUGINS_DIR="$GAZEBO_EXAMPLE_DIR/plugins"

echo "Iniciando sesión de tmux '$SESSION_NAME' con ventanas separadas..."

# 1. Crea la sesión con la primera ventana para 'yarpserver'
tmux new-session -d -s $SESSION_NAME -n server
tmux send-keys -t $SESSION_NAME:server "conda activate $CONDA_ENV" C-m
tmux send-keys -t $SESSION_NAME:server "yarpserver --write" C-m

# 2. Crea una ventana para Gazebo
tmux new-window -t $SESSION_NAME -n gazebo
tmux send-keys -t $SESSION_NAME:gazebo "conda activate $CONDA_ENV" C-m
tmux send-keys -t $SESSION_NAME:gazebo "export GZ_SIM_RESOURCE_PATH=\${GZ_SIM_RESOURCE_PATH}:\${GAZEBO_MODEL_PATH}:$MODELS_DIR" C-m
# Controlboard parchado (modo MIXED para los ojos de iKinGazeCtrl), antes que el de conda.
# Ver gazebo/plugins/build_controlboard.sh.
tmux send-keys -t $SESSION_NAME:gazebo "export GZ_SIM_SYSTEM_PLUGIN_PATH=$PLUGINS_DIR:\${GZ_SIM_SYSTEM_PLUGIN_PATH}" C-m

# LANZAR SERVIDOR (Headless - Maneja física y sensores)
# tmux send-keys -t $SESSION_NAME:gazebo "gz sim -r $WORLD" C-m
tmux send-keys -t $SESSION_NAME:gazebo "__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia gz sim -s -r $WORLD -v 4" C-m
tmux pipe-pane -o -t $SESSION_NAME:1.0 'cat >> /tmp/tmux_output_pipe_gz_server'

# 2.1 Crear una sub-ventana (pane) para la GUI
tmux split-window -h -t $SESSION_NAME:gazebo
tmux send-keys -t $SESSION_NAME:gazebo.1 "conda activate $CONDA_ENV" C-m
tmux send-keys -t $SESSION_NAME:gazebo.1 "__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia gz sim -g" C-m
tmux pipe-pane -o -t $SESSION_NAME:1.1 'cat >> /tmp/tmux_output_pipe_gz_gui'

# 3. Crea una ventana para el 'robotinterface'
tmux new-window -t $SESSION_NAME -n robotinterface
tmux send-keys -t $SESSION_NAME:robotinterface "conda activate $CONDA_ENV" C-m
tmux send-keys -t $SESSION_NAME:robotinterface "sleep 5 # Espera a que Gazebo inicie" C-m
tmux send-keys -t $SESSION_NAME:robotinterface "yarprobotinterface --context $CONTEXT --config no_legs.xml" C-m

# 4. Crea una ventana para el brazo derecho
tmux new-window -t $SESSION_NAME -n right_arm
tmux send-keys -t $SESSION_NAME:right_arm "conda activate $CONDA_ENV" C-m
tmux send-keys -t $SESSION_NAME:right_arm "sleep 10 # Espera a que el robot interface inicie" C-m
tmux send-keys -t $SESSION_NAME:right_arm "iKinCartesianSolver --context $CONTEXT --part right_arm" C-m

# 5. Crea una ventana para el brazo izquierdo
tmux new-window -t $SESSION_NAME -n left_arm
tmux send-keys -t $SESSION_NAME:left_arm "conda activate $CONDA_ENV" C-m
tmux send-keys -t $SESSION_NAME:left_arm "sleep 10 # Espera a que el robot interface inicie" C-m
tmux send-keys -t $SESSION_NAME:left_arm "iKinCartesianSolver --context $CONTEXT --part left_arm" C-m

# 6. Crea una ventana para el control de la mirada (gaze)
tmux new-window -t $SESSION_NAME -n gaze
tmux send-keys -t $SESSION_NAME:gaze "conda activate $CONDA_ENV" C-m
tmux send-keys -t $SESSION_NAME:gaze "sleep 10 # Espera a que el robot interface inicie" C-m
# YARP_CLOCK=/clock: el gaze usa el tiempo simulado (lo publica gzyarp::Clock). Con
# reloj de pared y RTF < 1 los ojos (control en velocidad) oscilan y se desorbitan.
tmux send-keys -t $SESSION_NAME:gaze "YARP_CLOCK=/clock iKinGazeCtrl --context $CONTEXT --from iKinGazeCtrl.ini" C-m

echo "Sesión de simulación iniciada."
echo "Usa 'tmux attach -t $SESSION_NAME' y navega entre ventanas con 'Ctrl+b, n' (siguiente) o 'Ctrl+b, p' (anterior)."

# # Opcional: Adjuntar a la sesión automáticamente
# tmux attach-session -t $SESSION_NAME:robotinterface