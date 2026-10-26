#!/usr/bin/env bash
set -e

usage() {
    cat <<EOF
Usage: ./gazebo/scripts/launch_sim.sh [OPTIONS]

Launch the iCub humanoid simulation in Gazebo Sim + YARP.

Options:
  -w, --world <WORLD>   Select the world type (default: manip):
                          * manip : Manipulation arena (table, objects, external camera, iCubGazeboV2_5_visuomanip)
                          * empty : Empty world with the selected --model
  -m, --model <MODEL>   Select the iCub model to launch (default: iCubGazeboV2_5_visuomanip for manip, iCubGazeboV2_5_fixed for empty):
                          * iCubGazeboV2_5_visuomanip [FIXED BASE]    V2.5 with articulated fingers, eyes & cameras
                          * iCubGazeboV2_5_fixed      [FIXED BASE]    Standard V2.5 upper-body
                          * iCubGazeboV2_7_fixed      [FIXED BASE]    Standard V2.7 upper-body
                          * iCubGazeboV2_5            [FLOATING BASE] Full-body V2.5 standing on ground
                          * iCubGazeboV2_7            [FLOATING BASE] Full-body V2.7 standing on ground
  --stop, --kill        Stop any running Gazebo Sim (gz sim) and YARP (yarpserver) processes
  -h, --help            Show this help message and exit

Examples:
  ./gazebo/scripts/launch_sim.sh
  ./gazebo/scripts/launch_sim.sh --world manip
  ./gazebo/scripts/launch_sim.sh --world empty --model iCubGazeboV2_5_fixed
  ./gazebo/scripts/launch_sim.sh --world empty --model iCubGazeboV2_7_fixed
  ./gazebo/scripts/launch_sim.sh --world empty --model iCubGazeboV2_5
  ./gazebo/scripts/launch_sim.sh --world empty --model iCubGazeboV2_7
  ./gazebo/scripts/launch_sim.sh --stop
EOF
}

SELECTED_WORLD="manip"
SELECTED_MODEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stop|--kill|--clean)
            echo "=========================================================="
            echo " Stopping any running Gazebo Sim and YARP processes..."
            echo "=========================================================="
            pkill -f "gz sim" 2>/dev/null && echo "[Gazebo] Stopped gz sim processes." || echo "[Gazebo] No gz sim processes found."
            pkill -f "yarpserver" 2>/dev/null && echo "[YARP] Stopped yarpserver." || echo "[YARP] No yarpserver found."
            echo "Done."
            exit 0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -w|--world)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --world requires an argument (manip or empty)."
                usage
                exit 1
            fi
            SELECTED_WORLD="$2"
            shift 2
            ;;
        -m|--model)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --model requires a model name argument."
                usage
                exit 1
            fi
            SELECTED_MODEL="$2"
            shift 2
            ;;
        iCubGazeboV2_5|iCubGazeboV2_5_fixed|iCubGazeboV2_5_visuomanip|iCubGazeboV2_7|iCubGazeboV2_7_fixed)
            SELECTED_MODEL="$1"
            shift
            ;;
        *)
            echo "Error: Unknown argument '$1'."
            usage
            exit 1
            ;;
    esac
done

# Determine base directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PIXI_ENV="${PROJECT_ROOT}/.pixi/envs/default"

if [ ! -d "${PIXI_ENV}" ] && [ -d "${PROJECT_ROOT}/../icub-bimanual/.pixi/envs/default" ]; then
    PIXI_ENV="${PROJECT_ROOT}/../icub-bimanual/.pixi/envs/default"
fi

if [ ! -d "${PIXI_ENV}" ]; then
    echo "Error: Pixi environment not found at ${PIXI_ENV}"
    echo "Please run 'pixi install' first."
    exit 1
fi

# Resolve default model if not explicitly provided
if [ -z "${SELECTED_MODEL}" ]; then
    if [ "${SELECTED_WORLD}" = "manip" ]; then
        SELECTED_MODEL="iCubGazeboV2_5_visuomanip"
    else
        SELECTED_MODEL="iCubGazeboV2_5_fixed"
    fi
else
    # If user explicitly picked a non-visuomanip model without specifying --world, use empty world
    if [ "${SELECTED_MODEL}" != "iCubGazeboV2_5_visuomanip" ] && [ "${SELECTED_WORLD}" = "manip" ]; then
        SELECTED_WORLD="empty"
    fi
fi

if [ "${SELECTED_WORLD}" = "manip" ]; then
    WORLD_FILE="${PROJECT_ROOT}/gazebo/worlds/icub_world.sdf"
    MODEL_NAME="iCubGazeboV2_5_visuomanip (Manipulation Arena with Table, Objects & Cameras)"
    BASE_TYPE="FIXED BASE (z = 0.73 m at table)"
else
    case "${SELECTED_MODEL}" in
        iCubGazeboV2_5_fixed)
            WORLD_FILE="${PROJECT_ROOT}/gazebo/worlds/icub_fixed.sdf"
            MODEL_NAME="iCubGazeboV2_5_fixed (Standard V2.5)"
            BASE_TYPE="FIXED BASE (Root link welded in air)"
            ;;
        iCubGazeboV2_5_visuomanip)
            WORLD_FILE="${PROJECT_ROOT}/gazebo/worlds/icub_visuomanip_fixed.sdf"
            MODEL_NAME="iCubGazeboV2_5_visuomanip (with Fingers & Eyes)"
            BASE_TYPE="FIXED BASE (Root link welded in air)"
            ;;
        iCubGazeboV2_7_fixed)
            WORLD_FILE="${PROJECT_ROOT}/gazebo/worlds/icub_v2_7_fixed.sdf"
            MODEL_NAME="iCubGazeboV2_7_fixed (Standard V2.7)"
            BASE_TYPE="FIXED BASE (Root link welded in air)"
            ;;
        iCubGazeboV2_5)
            WORLD_FILE="${PROJECT_ROOT}/gazebo/worlds/icub_v2_5_floating.sdf"
            MODEL_NAME="iCubGazeboV2_5 (Full-Body V2.5)"
            BASE_TYPE="FLOATING BASE (Standing on ground plane)"
            ;;
        iCubGazeboV2_7)
            WORLD_FILE="${PROJECT_ROOT}/gazebo/worlds/icub_v2_7_floating.sdf"
            MODEL_NAME="iCubGazeboV2_7 (Full-Body V2.7)"
            BASE_TYPE="FLOATING BASE (Standing on ground plane)"
            ;;
        *)
            echo "Error: Unsupported model '${SELECTED_MODEL}'."
            echo "Supported models: iCubGazeboV2_5, iCubGazeboV2_5_fixed, iCubGazeboV2_5_visuomanip, iCubGazeboV2_7, iCubGazeboV2_7_fixed"
            exit 1
            ;;
    esac
fi

# Record current active Gazebo model for teleop modules and clients
echo "${SELECTED_MODEL}" > /tmp/icub_gazebo_model

# Activate pixi environment variables
export PATH="${PIXI_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${PIXI_ENV}/lib:${LD_LIBRARY_PATH:-}"

# Configure YARP data directories
export YARP_DATA_DIRS="${PIXI_ENV}/share/iCub:${PIXI_ENV}/share/yarp:${YARP_DATA_DIRS:-}"

# Configure Gazebo Sim resource and plugin search paths (including gazebo/models and gazebo/worlds)
export GZ_SIM_RESOURCE_PATH="${PROJECT_ROOT}/gazebo/models:${PROJECT_ROOT}/gazebo/worlds:${PIXI_ENV}/share/iCub/robots:${PIXI_ENV}/share:${PIXI_ENV}/share/iCub:${GZ_SIM_RESOURCE_PATH:-}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="${PIXI_ENV}/lib:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
export GZ_CONFIG_PATH="${PIXI_ENV}/share/gz:${GZ_CONFIG_PATH:-}"

# Automatically offload rendering to NVIDIA discrete GPU if present
if command -v nvidia-smi > /dev/null 2>&1; then
    export __NV_PRIME_RENDER_OFFLOAD=1
    export __GLX_VENDOR_LIBRARY_NAME=nvidia
fi

echo "=========================================================="
echo " Starting iCub Simulation in Gazebo Sim + YARP"
echo " World Mode:  ${SELECTED_WORLD}"
echo " Model:       ${MODEL_NAME}"
echo " Base Type:   ${BASE_TYPE}"
echo " Environment: ${PIXI_ENV}"
echo " World file:  ${WORLD_FILE}"
if [ -n "${__NV_PRIME_RENDER_OFFLOAD:-}" ]; then
    echo " Rendering:   NVIDIA PRIME Offload (Dedicated GPU)"
fi
echo "=========================================================="

YARP_PID=""
GZ_PID=""
STARTED_YARP=false

cleanup() {
    trap - EXIT INT TERM
    echo ""
    echo "=========================================================="
    echo " [CLEANUP] Stopping simulation processes..."
    echo "=========================================================="

    if [ -n "${GZ_PID}" ] && kill -0 "${GZ_PID}" 2>/dev/null; then
        echo "[Gazebo] Terminating Gazebo Sim (PID ${GZ_PID})..."
        kill -INT "${GZ_PID}" 2>/dev/null || true
        wait "${GZ_PID}" 2>/dev/null || true
    fi

    if [ "${STARTED_YARP}" = true ] && [ -n "${YARP_PID}" ]; then
        if kill -0 "${YARP_PID}" 2>/dev/null; then
            echo "[YARP] Terminating yarpserver (PID ${YARP_PID})..."
            kill -TERM "${YARP_PID}" 2>/dev/null || true
            for i in {1..6}; do
                if ! kill -0 "${YARP_PID}" 2>/dev/null; then
                    break
                fi
                sleep 0.5
            done
            if kill -0 "${YARP_PID}" 2>/dev/null; then
                kill -9 "${YARP_PID}" 2>/dev/null || true
            fi
        fi
    fi

    yarp clean --quiet --timeout 0.5 2>/dev/null || true
    rm -f /tmp/icub_gazebo_model
    echo "[CLEANUP] Shutdown complete."
}

trap cleanup EXIT INT TERM

# 1. Check if yarpserver is already running
if yarp where > /dev/null 2>&1; then
    echo "[YARP] yarpserver is already active."
    STARTED_YARP=false
    yarp clean --quiet --timeout 0.5 2>/dev/null || true
else
    echo "[YARP] Starting yarpserver in the background..."
    yarpserver --write &
    YARP_PID=$!
    STARTED_YARP=true
    sleep 2
    if ! yarp where > /dev/null 2>&1; then
        echo "[YARP] Error: Failed to start yarpserver."
        exit 1
    fi
    echo "[YARP] yarpserver started with PID ${YARP_PID}."
fi

# 2. Launch Gazebo Sim with full GUI in background and wait for it
echo "[Gazebo] Launching gz sim with GUI..."
gz sim -r "${WORLD_FILE}" &
GZ_PID=$!

wait "${GZ_PID}" || true

