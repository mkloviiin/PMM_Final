#!/bin/bash
# Congela el entorno conda que usa el proyecto (por defecto icubenv) en los dos
# lockfiles que consume el Dockerfile:
#   docker/conda-linux-64.lock   paquetes conda con build exacto (conda list --explicit)
#   docker/requirements.lock.txt paquetes instalados con pip, version exacta
#
# Se excluyen de pip los paquetes instalados desde una ruta local o un repo
# (lerobot, beavr, pyzed, lerobot_robot_icub...): lerobot se instala en el
# Dockerfile desde GitHub en un commit fijo y el resto no lo usa el proyecto
# (los plugins lerobot_* del repo se importan por sys.path).
#
# Uso: docker/freeze_env.sh [ruta_del_env]   (default ~/miniconda3/envs/icubenv)
set -euo pipefail

ENV_PREFIX="${1:-${ICUBENV_PREFIX:-$HOME/miniconda3/envs/icubenv}}"
HERE="$(dirname "$(readlink -f "$0")")"
CONDA_BIN="$(command -v conda || echo "$HOME/miniconda3/bin/conda")"

[ -x "$ENV_PREFIX/bin/python" ] || { echo "No existe el entorno: $ENV_PREFIX" >&2; exit 1; }

"$CONDA_BIN" list -p "$ENV_PREFIX" --explicit --md5 > "$HERE/conda-linux-64.lock"

"$ENV_PREFIX/bin/python" - "$HERE/requirements.lock.txt" <<'EOF'
import sys
from importlib import metadata as md

# Dependencias exclusivas de paquetes que no van en la imagen (beavr), y que
# además no traen wheel para Python 3.12 (se compilarían desde el fuente).
EXCLUDE = {"pybullet"}

skipped, lines = [], []
for d in md.distributions():
    if (d.read_text("INSTALLER") or "").strip() != "pip":
        continue  # lo instaló conda: ya está en conda-linux-64.lock
    name = d.metadata["Name"]
    if name.lower() in EXCLUDE:
        skipped.append(name)
        continue
    if d.read_text("direct_url.json"):
        skipped.append(name)  # ruta local / git: no reproducible desde PyPI
        continue
    lines.append(f"{name}=={d.version}")

with open(sys.argv[1], "w") as f:
    f.write("# Generado por docker/freeze_env.sh — no editar a mano.\n")
    f.write("\n".join(sorted(set(lines), key=str.lower)) + "\n")
print(f"pip: {len(lines)} paquetes; omitidos (ruta local/git o EXCLUDE): {', '.join(sorted(skipped))}")
EOF

echo "conda: $(grep -c '^http' "$HERE/conda-linux-64.lock") paquetes -> $HERE/conda-linux-64.lock"
