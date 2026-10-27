#!/bin/bash
# Simulación Gazebo del Hub en una sesión tmux (icub_sim): yarpserver + servidor gz sim
# (física y sensores) + GUI. Usa el entorno Pixi del proyecto, como launch_sim.sh.
#
# El control lo hace el módulo YARP (src/interfaces/teleop_module_sm.py, BLF whole-body),
# conectado directo a los controlboards que abre el RobotInterface del modelo dentro de
# Gazebo: ya no se lanzan yarprobotinterface, iKinCartesianSolver ni iKinGazeCtrl.

SESSION_NAME="icub_sim"
GAZEBO_SIM_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")"
PROJECT_ROOT="$(dirname "$GAZEBO_SIM_DIR")"
WORLD="$GAZEBO_SIM_DIR/worlds/icub_world.sdf"
PIXI_ENV="${PIXI_ENV:-$PROJECT_ROOT/.pixi/envs/default}"
# Variables para renderizar con la GPU NVIDIA de un notebook con GPU híbrida (PRIME).
# GZ_RENDER_ENV="" las desactiva (PC de escritorio).
GZ_RENDER_ENV="${GZ_RENDER_ENV-__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia}"

if [ ! -d "$PIXI_ENV/bin" ]; then
    echo "Error: Pixi environment not found at $PIXI_ENV (run 'pixi install')."
    exit 1
fi

# Mismo entorno que launch_sim.sh, exportado en cada ventana de tmux
ENV_CMD="export PATH=$PIXI_ENV/bin:\$PATH"
ENV_CMD+=" LD_LIBRARY_PATH=$PIXI_ENV/lib:\${LD_LIBRARY_PATH:-}"
ENV_CMD+=" YARP_DATA_DIRS=$PIXI_ENV/share/iCub:$PIXI_ENV/share/yarp"
ENV_CMD+=" GZ_SIM_RESOURCE_PATH=$GAZEBO_SIM_DIR/models:$GAZEBO_SIM_DIR/worlds:$PIXI_ENV/share/iCub/robots:$PIXI_ENV/share:$PIXI_ENV/share/iCub"
ENV_CMD+=" GZ_SIM_SYSTEM_PLUGIN_PATH=$PIXI_ENV/lib"
ENV_CMD+=" GZ_CONFIG_PATH=$PIXI_ENV/share/gz"

echo "Iniciando sesión de tmux '$SESSION_NAME' (entorno: $PIXI_ENV)..."
echo "${MODEL:-iCubGazeboV2_5_visuomanip}" > /tmp/icub_gazebo_model

# 1. yarpserver
tmux new-session -d -s $SESSION_NAME -n server
tmux send-keys -t $SESSION_NAME:server "$ENV_CMD" C-m
tmux send-keys -t $SESSION_NAME:server "yarpserver --write" C-m

# 2. Servidor de Gazebo (headless: física y sensores)
tmux new-window -t $SESSION_NAME -n gazebo
tmux send-keys -t $SESSION_NAME:gazebo "$ENV_CMD" C-m
tmux send-keys -t $SESSION_NAME:gazebo "sleep 2; $GZ_RENDER_ENV gz sim -s -r $WORLD -v 4" C-m
# '>' (no '>>'): el log se reinicia en cada arranque; con '>>' crece sin límite (llegó a 1.3 GB).
tmux pipe-pane -o -t $SESSION_NAME:1.0 'cat > /tmp/tmux_output_pipe_gz_server'

# 2.1 GUI de Gazebo en otro panel
tmux split-window -h -t $SESSION_NAME:gazebo
tmux send-keys -t $SESSION_NAME:gazebo.1 "$ENV_CMD" C-m
tmux send-keys -t $SESSION_NAME:gazebo.1 "sleep 3; $GZ_RENDER_ENV gz sim -g" C-m
tmux pipe-pane -o -t $SESSION_NAME:1.1 'cat > /tmp/tmux_output_pipe_gz_gui'

echo "Sesión de simulación iniciada."
echo "Usa 'tmux attach -t $SESSION_NAME' y navega entre ventanas con 'Ctrl+b, n' (siguiente) o 'Ctrl+b, p' (anterior)."
