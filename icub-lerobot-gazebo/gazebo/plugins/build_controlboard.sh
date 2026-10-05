#!/bin/bash
# Recompila libgz-sim-yarp-controlboard-system.so (gz-sim-yarp-plugins v0.5.3) con
# controlboard_mixed_mode.patch: implementa el modo MIXED, que usa iKinGazeCtrl para
# los ojos y que el plugin oficial no implementa (los ojos quedan sin control).
# start_sim.sh antepone esta carpeta a GZ_SIM_SYSTEM_PLUGIN_PATH.
set -e
eval "$(conda shell.bash hook)"
conda activate icubenv
HERE="$(dirname "$(readlink -f "$0")")"
SRC="$(mktemp -d)"
git clone -q --depth 1 --branch v0.5.3 https://github.com/robotology/gz-sim-yarp-plugins.git "$SRC"
git -C "$SRC" apply "$HERE/controlboard_mixed_mode.patch"
cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DGZ_SIM_YARP_PLUGINS_BUILD_TOOLS=OFF -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH="$CONDA_PREFIX/lib" \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF
cmake --build "$SRC/build" --target gz-sim-yarp-controlboard-system -j"$(nproc)"
cp "$SRC/build/lib/libgz-sim-yarp-controlboard-system.so" "$HERE/"
rm -rf "$SRC"
echo "OK: $HERE/libgz-sim-yarp-controlboard-system.so"
