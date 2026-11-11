#!/bin/bash
# Reset de los objetos de la escena en Gazebo.
#
# Escena: etiqueta de mujoco_sim/assets/scenes/scenes.yaml (el manifiesto único de MuJoCo y Gazebo)
#   guardada en $ICUB_SCENE_FILE (por defecto /tmp/icub_lerobot_scene); la escribe el Hub al lanzar
#   la sesion. Sin ese archivo: Block Lifting con el cubo azul en x [-0.5, -0.3], y [-0.1, 0.2]
#   (marco Gazebo).
# Lado: $ICUB_SCENE_SIDE = right | left (opcional, lo pasa eval_policy.py en cada episodio) fija el
#   lado de la tarea como en MuJoCo (src/tools/scene_reset.py): en escenas con `sides`, "left"
#   espeja en y todos los objetos; en las que solo tienen mirror_y, esos objetos. Sin la variable:
#   el sorteo 50/50 de mirror_y (`sides` no se usa). La puerta es fija en Gazebo (no cambia de lado).
# Posiciones: al azar dentro de los rangos de scenes.yaml (marco MuJoCo: x hacia adelante, y a la
#   izquierda del robot), convertidas al marco de Gazebo, donde el iCub mira hacia -x: (x, y) -> (-x, -y).
# Cada objeto del yaml necesita su modelo en gazebo_sim/models/<body>/model.sdf; si falta, avisa y lo
#   salta. Las piezas fijas (la puerta) estan en gazebo_sim/scene_fixtures.yaml.
# Pick-and-Place (scenes.yaml: variants, spawn.yaw, fixtures; como src/tools/scene_reset.py):
#   - variants: una al azar; se crea models/<body>-<variante>/model.sdf con el nombre <body> (la
#     pose sigue en /model/<body>/pose) y la variante queda en $ICUB_SCENE_FILE.variants (JSON),
#     que lerobot_robot_icubrg graba como obj{i}_id y eval_policy.py usa para el exito.
#   - spawn.yaw: giro en z (en Gazebo, sobre la identidad: su mundo ya esta girado 180 grados).
#   - fixtures: modelos estaticos (las cajas y las replicas), espejados en y con el lado.
# Necesita 'gz' y python3 con yaml: usa el entorno Pixi del proyecto (o icubenv / ICUBENV_PREFIX).
# Requiere la simulacion corriendo.
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODELS_DIR="$(dirname "$SCRIPT_DIR")/models"
FIXTURES_YAML="$(dirname "$SCRIPT_DIR")/scene_fixtures.yaml"
SCENE_FILE="${ICUB_SCENE_FILE:-/tmp/icub_lerobot_scene}"

PIXI_ENV="${PROJECT_ROOT}/.pixi/envs/default"
if [ -d "${PIXI_ENV}/bin" ]; then
    export PATH="${PIXI_ENV}/bin:$PATH"
    export LD_LIBRARY_PATH="${PIXI_ENV}/lib:${LD_LIBRARY_PATH:-}"
elif [ -d "${ICUBENV_PREFIX:-$HOME/miniconda3/envs/icubenv}/bin" ]; then
    export PATH="${ICUBENV_PREFIX:-$HOME/miniconda3/envs/icubenv}/bin:$PATH"
fi

SCENES_YAML="${PROJECT_ROOT}/mujoco_sim/assets/scenes/scenes.yaml"

PLAN=$(python3 - "$SCENES_YAML" "$SCENE_FILE" "$FIXTURES_YAML" <<'PYEOF'
import json, math, os, random, sys
import yaml

yaml_path, scene_file, fixtures_path = sys.argv[1], sys.argv[2], sys.argv[3]
forced_side = os.environ.get("ICUB_SCENE_SIDE", "").strip().lower()
# se borran siempre, aunque no esten en scenes.yaml (pick-cube / place-bin: Pick-and-Place hasta 2026-11)
LEGACY = ["blue-cube", "blue-cylinder", "red-ball", "pick-cube", "place-bin"]
Z_GAZEBO = {"blue-cube": 0.745, "red-cube": 0.745}   # tapa 0.72 + semialtura

def pick(v):
    return random.uniform(*v) if isinstance(v, (list, tuple)) else v

if not os.path.exists(scene_file):
    # sin escena: Block Lifting con el cubo azul
    known = set(LEGACY)
    spawns = [("blue-cube", "blue-cube", round(random.uniform(-0.5, -0.3), 3), round(random.uniform(-0.1, 0.2), 3), 0.745, 1.0, 0.0)]
else:
    label = open(scene_file).read().strip()
    scenes = yaml.safe_load(open(yaml_path))
    scene = next((s for s in scenes if s.get("label") == label), None)
    if scene is None:
        print(f"ERROR escena desconocida: {label!r}", file=sys.stderr)
        sys.exit(2)
    fixtures = (yaml.safe_load(open(fixtures_path)) or {}) if os.path.exists(fixtures_path) else {}
    known = set(LEGACY) | {o["body"] for s in scenes for o in s.get("objects", [])} \
            | {f["body"] for s in scenes for f in s.get("fixtures") or []} \
            | {f["model"] for lst in fixtures.values() for f in lst}
    spawns = []
    # spawn.mirror_y: un solo sorteo por reset para todos esos objetos; si sale, y cambia de signo
    mirror = -1 if any(o["spawn"].get("mirror_y") for o in scene.get("objects", [])) \
             and random.random() < 0.5 else 1
    with_sides = scene.get("sides") is not None
    if forced_side in ("right", "left"):                                  # lado fijado (eval_policy.py)
        mirror = -1 if forced_side == "left" else 1
    sided = with_sides and forced_side in ("right", "left")
    variants = {}
    for obj in scene.get("objects", []):
        sp = dict(obj["spawn"])
        model = obj["body"]
        if obj.get("variants"):                                           # una al azar (scene_reset.py)
            v = random.choice(obj["variants"])
            variants[obj["body"]] = v["name"]
            model = f'{obj["body"]}-{v["name"]}'
            sp["z"], r = v["z"], float(v.get("radius", 0.0))
            for ax in "xy":
                lo, hi = sp[ax]
                sp[ax] = [lo + r, hi - r] if hi - lo > 2 * r else [(lo + hi) / 2] * 2
        mirrored = sp.get("mirror_y") or sided
        x = -pick(sp["x"])                                                # marco MuJoCo -> marco Gazebo
        y = -pick(sp["y"]) * (mirror if mirrored else 1)
        z = Z_GAZEBO.get(obj["body"], pick(sp["z"]))
        h = math.radians(pick(sp["yaw"])) / 2 if sp.get("yaw") is not None else 0.0
        spawns.append((obj["body"], model, round(x, 3), round(y, 3), round(z, 4), round(math.cos(h), 6), round(math.sin(h), 6)))
    for fx in scene.get("fixtures") or []:                                # cajas / replicas (Pick-and-Place)
        fxp = fx["pos"]
        y = -fxp[1] * mirror                                              # el mismo lado que los objetos
        spawns.append((fx["body"], fx["body"], round(-fxp[0], 4), round(y, 4), round(fxp[2], 4), 1.0, 0.0))
    for fx in fixtures.get(label, []):                                    # piezas fijas de la escena
        spawns.append((fx["model"], fx["model"], *fx["pose"], 1.0, 0.0))
    if variants:
        with open(scene_file + ".variants", "w") as fv:
            json.dump(variants, fv)

for name in sorted(known):
    print("DELETE", name)
for name, model, x, y, z, qw, qz in spawns:
    print("SPAWN", name, model, x, y, z, qw, qz)
PYEOF
)
if [ $? -ne 0 ] || [ -z "$PLAN" ]; then
    echo "[reset] no se pudo leer la escena"
    exit 1
fi

# 1. Borrar todos los objetos de escena conocidos (silencioso: si no existen, no pasa nada)
while read -r kind name _rest; do
    [ "$kind" = "DELETE" ] || continue
    gz service -s /world/default/remove --reqtype gz.msgs.Entity --reptype gz.msgs.Boolean \
        --req "name: \"$name\", type: MODEL" --timeout 1000 > /dev/null 2>&1
done <<< "$PLAN"
sleep 0.2

# 2. Crear los objetos de la escena (con aviso visible si algo falla)
while read -r kind name model x y z qw qz; do
    [ "$kind" = "SPAWN" ] || continue
    if [ ! -f "$MODELS_DIR/$model/model.sdf" ]; then
        echo "[reset] AVISO: falta el modelo $MODELS_DIR/$model/model.sdf, se omite"
        continue
    fi
    OUT=$(gz service -s /world/default/create --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean \
        --req "sdf_filename: \"$MODELS_DIR/$model/model.sdf\", pose: {position: {x: $x, y: $y, z: $z}, orientation: {w: $qw, x: 0, y: 0, z: $qz}}, name: \"$name\"" \
        --timeout 2000 2>&1)
    case "$OUT" in
        *true*) if [ "$model" = "$name" ]; then echo "[reset] $name -> ($x, $y, $z)"
                else echo "[reset] $name ($model) -> ($x, $y, $z)"; fi ;;
        *)      echo "[reset] AVISO: no se pudo crear $name (simulacion corriendo y 'gz' en el PATH?): $OUT" ;;
    esac
done <<< "$PLAN"
echo "[reset] listo"
