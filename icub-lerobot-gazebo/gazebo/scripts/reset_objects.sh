#!/bin/bash
# Reset de los objetos de la escena en Gazebo.
#
# Escena: etiqueta de scenes.yaml guardada en $ICUB_SCENE_FILE (por defecto /tmp/icub_lerobot_scene);
#   la escribe el dashboard al lanzar la sesion. Sin ese archivo se comporta como antes:
#   Block Lifting con el cubo azul en el area original de este script.
# Posiciones: al azar dentro de los rangos de scenes.yaml (marco MuJoCo: x hacia adelante, y a la
#   izquierda del robot), convertidas al marco de Gazebo, donde el iCub mira hacia -x: (x, y) -> (-x, -y).
# Cada objeto del yaml necesita su modelo en gazebo/models/<body>/model.sdf; si falta, avisa y lo salta.
# Necesita 'gz' en el PATH (entorno icub_lerobot_env activo) y la simulacion corriendo.

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
MODELS_DIR="$(dirname "$SCRIPT_DIR")/models"
SCENES_YAML="$(dirname "$(dirname "$SCRIPT_DIR")")/mujoco/assets/scenes/scenes.yaml"
SCENE_FILE="${ICUB_SCENE_FILE:-/tmp/icub_lerobot_scene}"

PLAN=$(python3 - "$SCENES_YAML" "$SCENE_FILE" <<'PYEOF'
import os, random, sys
import yaml

yaml_path, scene_file = sys.argv[1], sys.argv[2]
LEGACY = ["blue-cube", "blue-cylinder", "red-ball"]      # objetos del script original

def pick(v):
    return random.uniform(*v) if isinstance(v, (list, tuple)) else v

if not os.path.exists(scene_file):
    # comportamiento original: Block Lifting, cubo azul en el area del script anterior
    known = set(LEGACY)
    spawns = [("blue-cube", round(random.uniform(-0.5, -0.3), 3), round(random.uniform(-0.1, 0.2), 3), 0.745)]
else:
    label = open(scene_file).read().strip()
    scenes = yaml.safe_load(open(yaml_path))
    scene = next((s for s in scenes if s.get("label") == label), None)
    if scene is None:
        print(f"ERROR escena desconocida: {label!r}", file=sys.stderr)
        sys.exit(2)
    known = set(LEGACY) | {o["body"] for s in scenes for o in s.get("objects", [])}
    spawns = []
    for obj in scene.get("objects", []):
        sp = obj["spawn"]
        x, y = -pick(sp["x"]), -pick(sp["y"])                          # marco MuJoCo -> marco Gazebo
        z = 0.745 if obj["body"].endswith("cube") else pick(sp["z"])   # cubos: tapa 0.72 + 0.025
        spawns.append((obj["body"], round(x, 3), round(y, 3), round(z, 3)))

for name in sorted(known):
    print("DELETE", name)
for name, x, y, z in spawns:
    print("SPAWN", name, x, y, z)
PYEOF
)
if [ $? -ne 0 ] || [ -z "$PLAN" ]; then
    echo "[reset] no se pudo leer la escena"
    exit 1
fi

# 1. Borrar todos los objetos de escena conocidos (silencioso: si no existen, no pasa nada)
while read -r kind name x y z; do
    [ "$kind" = "DELETE" ] || continue
    gz service -s /world/default/remove --reqtype gz.msgs.Entity --reptype gz.msgs.Boolean \
        --req "name: \"$name\", type: MODEL" --timeout 1000 > /dev/null 2>&1
done <<< "$PLAN"
sleep 0.2

# 2. Crear los objetos de la escena (con aviso visible si algo falla)
while read -r kind name x y z; do
    [ "$kind" = "SPAWN" ] || continue
    if [ ! -f "$MODELS_DIR/$name/model.sdf" ]; then
        echo "[reset] AVISO: falta el modelo $MODELS_DIR/$name/model.sdf, se omite"
        continue
    fi
    OUT=$(gz service -s /world/default/create --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean \
        --req "sdf_filename: \"$MODELS_DIR/$name/model.sdf\", pose: {position: {x: $x, y: $y, z: $z}}, name: \"$name\"" \
        --timeout 2000 2>&1)
    case "$OUT" in
        *true*) echo "[reset] $name -> ($x, $y, $z)" ;;
        *)      echo "[reset] AVISO: no se pudo crear $name (simulacion corriendo y 'gz' en el PATH?): $OUT" ;;
    esac
done <<< "$PLAN"
echo "[reset] listo"
