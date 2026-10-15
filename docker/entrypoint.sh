#!/bin/bash
# Activa icubenv antes de ejecutar el comando: los scripts de activación de conda
# definen YARP_DATA_DIRS, GZ_SIM_SYSTEM_PLUGIN_PATH, rutas de icub-models, etc.,
# que necesitan yarprobotinterface, Gazebo y el módulo YARP.
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate icubenv
exec "$@"
