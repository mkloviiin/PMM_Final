#!/bin/bash
# Reset de los objetos de la escena en Gazebo.
#
# Escena: etiqueta de scenes.yaml guardada en $ICUB_SCENE_FILE (por defecto /tmp/icub_lerobot_scene);
#   la escribe el dashboard al lanzar la sesion. Sin ese archivo: Block Lifting con el cubo azul
#   en x [-0.5, -0.3], y [-0.1, 0.2] (marco Gazebo).
# Posiciones: al azar dentro de los rangos de scenes.yaml (marco MuJoCo: x hacia adelante, y a la
#   izquierda del robot), convertidas al marco de Gazebo, donde el iCub mira hacia -x: (x, y) -> (-x, -y).
# Cada objeto del yaml necesita su modelo en gazebo/models/<body>/model.sdf; si falta, avisa y lo salta.
# Lado de la tarea (`sides` en scenes.yaml, como scene_reset.py de MuJoCo): "left" espeja en y los
#   rangos de los objetos; en la puerta elige el modelo con la manilla de ese lado. Con alternate, el
#   lado del reset anterior queda en $SCENE_FILE.side: si la escena se volvio a escribir despues
#   (lanzamiento del Hub, eval_gazebo.py), se parte de nuevo por la derecha, como en MuJoCo.
# Puerta (`door` en scenes.yaml): centro de la hoja en panel_xy ± jitter (marco MuJoCo).
# Necesita 'gz' y python3 con yaml: se antepone el entorno icubenv (ICUBENV_PREFIX en Docker)
# al PATH. Requiere la simulacion corriendo.
export PATH="${ICUBENV_PREFIX:-$HOME/miniconda3/envs/icubenv}/bin:$PATH"

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
MODELS_DIR="$(dirname "$SCRIPT_DIR")/models"
SCENES_YAML="$(dirname "$SCRIPT_DIR")/scenes.yaml"
SCENE_FILE="${ICUB_SCENE_FILE:-/tmp/icub_lerobot_scene}"
FIXTURES_YAML="$(dirname "$SCRIPT_DIR")/scene_fixtures.yaml"

PLAN=$(python3 - "$SCENES_YAML" "$SCENE_FILE" "$FIXTURES_YAML" <<'PYEOF'
import os, random, sys
import yaml

yaml_path, scene_file, fixtures_path = sys.argv[1], sys.argv[2], sys.argv[3]
side_file = scene_file + ".side"
LEGACY = ["blue-cube", "blue-cylinder", "red-ball"]      # se borran siempre, aunque no esten en scenes.yaml
Z_GAZEBO = {"blue-cube": 0.745, "red-cube": 0.745, "pick-cube": 0.74}   # tapa 0.72 + semialtura

def pick(v):
    return random.uniform(*v) if isinstance(v, (list, tuple)) else v

def mirror_y(v):
    # rango (o valor) y del lado derecho -> el del lado izquierdo
    return [-v[1], -v[0]] if isinstance(v, (list, tuple)) else -v

def task_side(scene):
    mode = scene.get("sides")
    if mode is None:
        return None
    if mode in ("right", "left"):
        return mode
    if mode != "alternate":
        print(f"ERROR sides: {mode!r} (right, left o alternate)", file=sys.stderr)
        sys.exit(2)
    prev = None
    try:
        if os.path.getmtime(side_file) >= os.path.getmtime(scene_file):
            label_prev, prev = open(side_file).read().split("\t")
            prev = prev.strip() if label_prev == scene["label"] else None
    except (OSError, ValueError):
        prev = None
    side = "left" if prev == "right" else "right"
    with open(side_file, "w") as f:
        f.write(f"{scene['label']}\t{side}\n")
    return side

if not os.path.exists(scene_file):
    # sin escena: Block Lifting con el cubo azul
    known = set(LEGACY)
    spawns = [("blue-cube", "blue-cube", round(random.uniform(-0.5, -0.3), 3), round(random.uniform(-0.1, 0.2), 3), 0.745)]
else:
    label = open(scene_file).read().strip()
    scenes = yaml.safe_load(open(yaml_path))
    scene = next((s for s in scenes if s.get("label") == label), None)
    if scene is None:
        print(f"ERROR escena desconocida: {label!r}", file=sys.stderr)
        sys.exit(2)
    fixtures = (yaml.safe_load(open(fixtures_path)) or {}) if os.path.exists(fixtures_path) else {}
    known = set(LEGACY) | {o["body"] for s in scenes for o in s.get("objects", [])} \
            | {f["model"] for lst in fixtures.values() for f in lst}
    side = task_side(scene)
    print(f"SIDE {side}")
    spawns = []
    for obj in scene.get("objects", []):
        sp = obj["spawn"]
        y_range = mirror_y(sp["y"]) if side == "left" else sp["y"]
        x, y = -pick(sp["x"]), -pick(y_range)                          # marco MuJoCo -> marco Gazebo
        z = Z_GAZEBO.get(obj["body"], pick(sp["z"]))
        spawns.append((obj["body"], obj["body"], round(x, 3), round(y, 3), round(z, 3)))
    door = scene.get("door")
    for fx in fixtures.get(label, []):                                    # piezas fijas de la escena
        model = fx.get("side_models", {}).get(side, fx["model"])
        x, y, z = fx["pose"]
        if door and "panel_offset" in fx:
            # door_mount (marco MuJoCo) = centro de la hoja (± jitter) - panel_offset; Gazebo (-x, -y)
            j = float(door.get("jitter", 0.0))
            px, py = (float(v) + random.uniform(-j, j) for v in door["panel_xy"])
            x, y = -(px - fx["panel_offset"][0]), -(py - fx["panel_offset"][1])
        spawns.append((fx["model"], model, round(x, 4), round(y, 4), z))

for name in sorted(known):
    print("DELETE", name)
for name, model, x, y, z in spawns:
    print("SPAWN", name, model, x, y, z)
PYEOF
)
if [ $? -ne 0 ] || [ -z "$PLAN" ]; then
    echo "[reset] no se pudo leer la escena"
    exit 1
fi

SIDE=$(sed -n 's/^SIDE //p' <<< "$PLAN")
[ -n "$SIDE" ] && [ "$SIDE" != "None" ] && echo "[reset] lado: $SIDE"

# 1. Borrar todos los objetos de escena conocidos (silencioso: si no existen, no pasa nada)
while read -r kind name rest; do
    [ "$kind" = "DELETE" ] || continue
    gz service -s /world/default/remove --reqtype gz.msgs.Entity --reptype gz.msgs.Boolean \
        --req "name: \"$name\", type: MODEL" --timeout 1000 > /dev/null 2>&1
done <<< "$PLAN"
sleep 0.2

# 2. Crear los objetos de la escena (con aviso visible si algo falla). La entidad se llama $name;
#    el modelo ($model) puede ser otro (la puerta con la manilla a la derecha: door-right).
while read -r kind name model x y z; do
    [ "$kind" = "SPAWN" ] || continue
    if [ ! -f "$MODELS_DIR/$model/model.sdf" ]; then
        echo "[reset] AVISO: falta el modelo $MODELS_DIR/$model/model.sdf, se omite"
        continue
    fi
    OUT=$(gz service -s /world/default/create --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean \
        --req "sdf_filename: \"$MODELS_DIR/$model/model.sdf\", pose: {position: {x: $x, y: $y, z: $z}}, name: \"$name\"" \
        --timeout 2000 2>&1)
    case "$OUT" in
        *true*) echo "[reset] $name ($model) -> ($x, $y, $z)" ;;
        *)      echo "[reset] AVISO: no se pudo crear $name (simulacion corriendo y 'gz' en el PATH?): $OUT" ;;
    esac
done <<< "$PLAN"
echo "[reset] listo"
