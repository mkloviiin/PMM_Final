#!/bin/bash
export PATH=$HOME/miniconda3/envs/icubenv/bin:$PATH

# Define paths
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
MODELS_DIR="$(dirname "$SCRIPT_DIR")/models"

MODE="2"
#MODE 1 = cylinder, MODE 2 = cube, MODE 3 = sphere

POSITION="2"
#POSITION 1 = fixed, POSITION 2 = random area

if [ "$MODE" = "1" ]; then
    OBJECT_NAME="blue-cylinder"
elif [ "$MODE" = "2" ]; then
    OBJECT_NAME="blue-cube"
elif [ "$MODE" = "3" ]; then
    OBJECT_NAME="red-ball"
fi

if [ "$POSITION" = "1" ]; then
    # Fixed positions
    OBJ_X="-0.4"
    OBJ_Y="0.0"
    OBJ_Z="0.745"
elif [ "$POSITION" = "2" ]; then
    # Random positions
    # Area: X -0.3 to -0.5; Y 0.2 to -0.1; Z 0.745 (table top 0.72 + half cube)
    OBJ_X=$(python3 -c "import random; print(round(random.uniform(-0.5, -0.3), 3))")
    OBJ_Y=$(python3 -c "import random; print(round(random.uniform(-0.1, 0.2), 3))")
    OBJ_Z="0.745"
fi

# 1. Delete existing objects (silence output to avoid confusion if they don't exist)
# We try to remove them to ensure a clean slate, especially if they fell off the table or user wants to reset.
gz service -s /world/default/remove --reqtype gz.msgs.Entity --reptype gz.msgs.Boolean --req 'name: "red-ball", type: MODEL' --timeout 1000 > /dev/null 2>&1
gz service -s /world/default/remove --reqtype gz.msgs.Entity --reptype gz.msgs.Boolean --req 'name: "blue-cylinder", type: MODEL' --timeout 1000 > /dev/null 2>&1
gz service -s /world/default/remove --reqtype gz.msgs.Entity --reptype gz.msgs.Boolean --req 'name: "blue-cube", type: MODEL' --timeout 1000 > /dev/null 2>&1

# Wait briefly for deletion to process
sleep 0.2

# 2. Spawn Objects using absolute paths to SDFs
# # Red Ball
# gz service -s /world/default/create \
#     --reqtype gz.msgs.EntityFactory \
#     --reptype gz.msgs.Boolean \
#     --req "sdf_filename: \"$MODELS_DIR/red-ball/model.sdf\", pose: {position: {x: $RED_X, y: $RED_Y, z: $RED_Z}}, name: \"red-ball\"" \
#     --timeout 2000 &

# Blue Cylinder
gz service -s /world/default/create \
    --reqtype gz.msgs.EntityFactory \
    --reptype gz.msgs.Boolean \
    --req "sdf_filename: \"$MODELS_DIR/$OBJECT_NAME/model.sdf\", pose: {position: {x: $OBJ_X, y: $OBJ_Y, z: $OBJ_Z}}, name: \"$OBJECT_NAME\"" \
    --timeout 2000 &

wait
echo "Objects reset (respawned) command sent."
