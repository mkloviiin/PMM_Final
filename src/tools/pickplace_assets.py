#!/usr/bin/env python3
"""Genera la escena Pick-and-Place del iCub a partir de PickPlace de robosuite (single_object_mode=1).

Una sola especificación (este archivo) para los dos simuladores:
  - mujoco_sim/assets/meshes/pickplace/<obj>.obj       mallas de robosuite, escaladas y centradas
  - mujoco_sim/assets/scenes/scene_icub_pickplace.xml  escena de MuJoCo
  - gazebo_sim/models/pick-object-<obj>/               objeto que se toma (dinámico, PosePublisher)
  - gazebo_sim/models/visual-<obj>/                    réplica visual en su compartimento (estática)
  - gazebo_sim/models/bin-source/, bin-target/         contenedores (estáticos)
e imprime el bloque `variants` / `fixtures` de scenes.yaml (pegarlo a mano si cambia algo aquí).

Qué se toma de robosuite (robosuite/environments/manipulation/pick_place.py, models/assets):
  - Los 4 objetos (leche, pan, cereal, lata) con sus mallas, texturas y densidades, en el orden de
    `object_to_id` (milk 0, bread 1, cereal 2, can 3): ese índice se graba como obj0_id.
  - BinsArena: bin1 (origen, a la derecha) y bin2 (destino, a la izquierda), de 40 x 50 cm con
    paredes de 10 cm; bin2 dividido en 4 compartimentos, uno por objeto, con la réplica visual del
    objeto en su centro. Compartimento del objeto i: x cercano al robot si i es 0 o 2, y del lado
    de bin1 si i < 2.
Qué cambia para el iCub (su brazo alcanza ~1/3 de lo que alcanza el Panda):
  - BIN_SCALE: contenedores a escala 0.4 en x y 0.48 en y (16 x 24 cm); WALL_SCALE: tapa y paredes
    a escala 0.3 (paredes de 3 cm de alto, 1 cm entre los contenedores). La profundidad (x) la fija
    el iCub: con la cabeza bajada al tope (neck_pitch -30°) los ojos ven la mesa desde x ~ 0.30-0.35
    y el punto de agarre llega con precisión hasta x ~ 0.41 (TSIK); por eso BIN_X = 0.37. El ancho
    (y) es el mínimo para que el cereal (10 cm) entre en su compartimento (12 cm, réplica girada
    90°, REPLICA_YAW). Destino centrado en y = +0.06 para que la mano del lado del origen llegue al
    compartimento exterior.
  - OBJECT_SCALE: objetos al tamaño de robosuite (1.0).
  - Filas de bin-target invertidas en x: los objetos altos (leche 14 cm, cereal 15 cm) van en los
    compartimentos lejanos y los bajos (pan, lata) en los cercanos; al revés que en robosuite, las
    réplicas altas cercanas tapaban a las bajas en la vista de los ojos del iCub. El lado en y es el
    de robosuite (leche y pan del lado de bin-source).
  - Colisión con primitivas (caja / cilindro del tamaño de la malla) en ambos simuladores, para que
    la física sea la misma; la malla solo se ve.

Uso (entorno Pixi, desde la raíz del repo; robosuite clonado al lado de PMM_Final por defecto):
    pixi run python src/tools/pickplace_assets.py [--robosuite ../../robosuite]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENES_DIR = REPO_ROOT / "mujoco_sim" / "assets" / "scenes"
MESH_DIR = REPO_ROOT / "mujoco_sim" / "assets" / "meshes" / "pickplace"
TEXTURE_DIR = REPO_ROOT / "mujoco_sim" / "assets" / "textures"
MODELS_DIR = REPO_ROOT / "gazebo_sim" / "models"

BIN_SCALE = np.array([0.4, 0.48])   # (x, y) del contenedor
WALL_SCALE = 0.3                     # alto y grosor de tapa y paredes, separación entre contenedores
OBJECT_SCALE = 1.0
TABLE_Z = 0.72

# robosuite: (malla, escala de su XML, densidad [kg/m3], textura, primitiva de colisión)
OBJECTS = {
    "milk": ("milk.obj", 0.9, 100.0, "ceramic.png", "box"),
    "bread": ("bread.obj", 0.8, 50.0, "bread.png", "box"),
    "cereal": ("cereal.obj", 1.0, 150.0, "cereal.png", "box"),
    "can": ("can.obj", 1.0, 100.0, "soda.png", "cylinder"),
}

# BinsArena de robosuite (medias medidas) escalado
BIN_HALF = np.array([0.20, 0.25]) * BIN_SCALE            # 16 x 24 cm
BASE_HALF_Z = 0.02 * WALL_SCALE                           # tapa de 1.2 cm
WALL_HALF_T = 0.01 * WALL_SCALE                           # paredes de 0.6 cm
WALL_HALF_H = 0.05 * WALL_SCALE                           # paredes de 3 cm
BIN_Z = TABLE_Z + BASE_HALF_Z                             # origen = centro de la tapa (como bin1_pos)
BASE_TOP_Z = BIN_Z + BASE_HALF_Z
# Lado derecho (marco MuJoCo: x adelante, y a la izquierda); el izquierdo espeja y
BIN_X = 0.37
TARGET_Y = 0.06
GAP = 0.03 * WALL_SCALE                                   # robosuite: 3 cm entre bin1 y bin2
SOURCE_Y = TARGET_Y - 2 * BIN_HALF[1] - GAP
# robosuite: los objetos aparecen en todo bin1 menos 5 cm por lado (y su radio horizontal)
SPAWN_MARGIN = 0.05 * BIN_SCALE

ROBOSUITE_QUAT = (0, 0, 0, 1)  # 180° en z: identidad en el frame root, como los objetos de Gazebo
# Giro extra en z de la réplica (grados): el cereal mide 10 cm en x y el compartimento solo 8 en x,
# así que su réplica (y el lugar donde cabe) va a lo largo de y (12 cm)
REPLICA_YAW = {"cereal": 90.0}


def replica_quat(name: str) -> np.ndarray:
    """Orientación (w, x, y, z) de la réplica en MuJoCo: ROBOSUITE_QUAT seguido de REPLICA_YAW."""
    h = np.radians(REPLICA_YAW.get(name, 0.0)) / 2
    w0, x0, y0, z0 = ROBOSUITE_QUAT
    c, s = np.cos(h), np.sin(h)                  # (w0 + z0 k)(c + s k), x0 = y0 = 0
    return np.array([w0 * c - z0 * s, 0.0, 0.0, w0 * s + z0 * c])


# ── Mallas ───────────────────────────────────────────────────────────────────

def read_obj(path: Path) -> tuple[np.ndarray, list[str]]:
    verts, rest = [], []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith(("vt ", "vn ", "f ", "s ", "o ", "g ")):
            rest.append(line)
    return np.array(verts), rest


def write_obj(path: Path, verts: np.ndarray, rest: list[str], mtl: str | None) -> None:
    head = [f"# robosuite (MIT), escalado y centrado por src/tools/pickplace_assets.py"]
    if mtl:
        head += [f"mtllib {mtl}", "usemtl material"]
    body = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    path.write_text("\n".join(head + body + rest) + "\n")


def object_geometry(robosuite: Path) -> dict:
    """Malla escalada y centrada en su caja (z arriba), tamaño, primitiva, masa e inercia."""
    out = {}
    for name, (mesh, rs_scale, density, texture, shape) in OBJECTS.items():
        verts, rest = read_obj(robosuite / "models" / "assets" / "objects" / "meshes" / mesh)
        verts = verts * rs_scale * OBJECT_SCALE
        verts -= (verts.min(0) + verts.max(0)) / 2
        size = verts.max(0) - verts.min(0)
        if shape == "box":
            volume = float(np.prod(size))
            a, b, c = size
            mass = density * volume
            inertia = mass / 12 * np.array([b * b + c * c, a * a + c * c, a * a + b * b])
            col = {"type": "box", "half": size / 2}
        else:
            r, h = float(max(size[:2]) / 2), float(size[2])
            mass = density * np.pi * r * r * h
            inertia = np.array([mass * (3 * r * r + h * h) / 12] * 2 + [mass * r * r / 2])
            col = {"type": "cylinder", "r": r, "h": h}
        out[name] = dict(verts=verts, rest=rest, size=size, mass=mass, inertia=inertia, col=col,
                         texture=texture, radius=float(np.linalg.norm(size[:2]) / 2))
    return out


# ── Disposición (lado derecho) ───────────────────────────────────────────────

def compartment_centers() -> dict[str, np.ndarray]:
    """Centro (x, y) del compartimento de cada objeto en bin-target (robosuite _setup_references)."""
    centers = {}
    for i, name in enumerate(OBJECTS):
        # robosuite pone 0 y 2 (leche y cereal) cerca del robot; aquí lejos, por ser los más altos
        dx = BIN_HALF[0] / 2 if i in (0, 2) else -BIN_HALF[0] / 2
        dy = -BIN_HALF[1] / 2 if i < 2 else BIN_HALF[1] / 2          # 0, 1: del lado de bin-source
        centers[name] = np.array([BIN_X + dx, TARGET_Y + dy])
    return centers


def bin_geoms(dividers: bool) -> list[tuple[str, tuple, tuple]]:
    """(nombre, pos, medias) de la tapa y las paredes (y los tabiques de bin-target)."""
    hx, hy = BIN_HALF
    g = [("base", (0, 0, 0), (hx, hy, BASE_HALF_Z))]
    wz = WALL_HALF_H                                   # robosuite: pared en z = 0.05 desde el origen
    g += [("wall_xp", (hx, 0, wz), (WALL_HALF_T, hy, WALL_HALF_H)),
          ("wall_xn", (-hx, 0, wz), (WALL_HALF_T, hy, WALL_HALF_H)),
          ("wall_yp", (0, hy, wz), (hx + WALL_HALF_T, WALL_HALF_T, WALL_HALF_H)),
          ("wall_yn", (0, -hy, wz), (hx + WALL_HALF_T, WALL_HALF_T, WALL_HALF_H))]
    if dividers:
        g += [("divider_x", (0, 0, wz), (WALL_HALF_T, hy, WALL_HALF_H)),
              ("divider_y", (0, 0, wz), (hx, WALL_HALF_T, WALL_HALF_H))]
    return g


def fixtures(geo: dict) -> list[tuple[str, np.ndarray]]:
    out = [("bin-source", np.array([BIN_X, SOURCE_Y, BIN_Z])),
           ("bin-target", np.array([BIN_X, TARGET_Y, BIN_Z]))]
    for name, (x, y) in compartment_centers().items():
        out.append((f"visual-{name}", np.array([x, y, BASE_TOP_Z + geo[name]["size"][2] / 2])))
    return out


def f(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in np.ravel(v))


# ── MuJoCo ───────────────────────────────────────────────────────────────────

def mujoco_xml(geo: dict) -> str:
    assets = ['    <texture name="pp-tex-light-wood" file="../textures/light-wood.png" type="2d"/>',
              '    <texture name="pp-tex-dark-wood" file="../textures/dark-wood.png" type="2d"/>',
              '    <material name="pp-bin-source" texture="pp-tex-light-wood" texrepeat="2 2" texuniform="true"/>',
              '    <material name="pp-bin-target" texture="pp-tex-dark-wood" texrepeat="2 2" texuniform="true"/>']
    for name, g in geo.items():
        assets += [f'    <mesh name="pp-{name}" file="../pickplace/{name}.obj"/>',
                   f'    <texture name="pp-tex-{name}" file="../textures/{g["texture"]}" type="2d"/>',
                   f'    <material name="pp-{name}" texture="pp-tex-{name}" reflectance="0.3"/>']

    def bin_body(name: str, pos, dividers: bool, mat: str) -> list[str]:
        lines = [f'    <body name="{name}" pos="{f(pos)}">']
        for gname, gpos, half in bin_geoms(dividers):
            lines.append(f'      <geom name="{name}_{gname}" type="box" pos="{f(gpos)}" size="{f(half)}" '
                         f'material="{mat}" group="1" friction="1 0.005 0.0001"/>')
        return lines + ["    </body>"]

    fx = dict(fixtures(geo))
    body = bin_body("bin-source", fx["bin-source"], False, "pp-bin-source")
    body += bin_body("bin-target", fx["bin-target"], True, "pp-bin-target")
    for name in geo:
        body += [f'    <body name="visual-{name}" pos="{f(fx[f"visual-{name}"])}" quat="{f(np.round(replica_quat(name), 6))}">',
                 f'      <geom name="visual-{name}" type="mesh" mesh="pp-{name}" material="pp-{name}" '
                 f'contype="0" conaffinity="0" group="1"/>',
                 "    </body>"]
    # El objeto que se toma: un cuerpo con las 4 variantes; scene_reset.set_variant deja activa una
    # (colisión, visibilidad, masa e inercia). Todas las colisiones compilan activas (MuJoCo solo
    # mete en el BVH las geoms que colisionan al compilar); al cargar el modelo,
    # scene_reset.init_variants desactiva las que no son la primera.
    first = next(iter(geo))
    g0 = geo[first]
    z0 = BASE_TOP_Z + g0["size"][2] / 2 + 0.002
    body += [f'    <body name="pick-object" pos="{BIN_X} {SOURCE_Y:.4f} {z0:.4f}" quat="{f(ROBOSUITE_QUAT)}">',
             "      <freejoint/>",
             f'      <inertial pos="0 0 0" mass="{g0["mass"]:.6g}" diaginertia="{f(g0["inertia"])}"/>']
    for name, g in geo.items():
        on = name == first
        col = g["col"]
        size = f(col["half"]) if col["type"] == "box" else f"{col['r']:.6g} {col['h'] / 2:.6g}"
        body += [f'      <geom name="pick-object-{name}-col" type="{col["type"]}" size="{size}" group="3" '
                 f'contype="1" conaffinity="1" friction="0.9 0.005 0.0001" condim="4" '
                 f'solimp="0.9 0.95 0.001" solref="0.02 1" rgba="0.5 0.5 0.5 1"/>',
                 f'      <geom name="pick-object-{name}-vis" type="mesh" mesh="pp-{name}" material="pp-{name}" '
                 f'contype="0" conaffinity="0" group="{1 if on else 5}"/>']
    body += ["    </body>"]
    return f"""<?xml version='1.0' encoding='utf-8'?>
<!-- GENERADO por src/tools/pickplace_assets.py (no editar a mano): PickPlace de robosuite
     (single_object_mode=1) a escala del iCub. Lado derecho (origen a la derecha); el lado izquierdo
     lo pone scene_reset.py espejando en y (scenes.yaml: sides, fixtures). -->
<mujoco model="scene_icub_pickplace">

  <include file="icub_visuomanip_ikin_limits.xml"/>
  <include file="icub_shared_materials.xml"/>
  <include file="icub_room_arena.xml"/>
  <include file="icub_standard_table.xml"/>

  <!-- multiccd: el cilindro de la lata contra la base de la caja (colisión convexa general) da un solo
       punto de contacto y la lata se balancea sin parar en reposo (~0.5 rad/s); con multiccd tiene
       varios puntos y queda quieta. Solo añade el flag: el resto de <option> viene del modelo. -->
  <option>
    <flag multiccd="enable"/>
  </option>

  <visual>
    <map force="0.1" zfar="30" />
    <rgba haze="0.15 0.25 0.35 1" />
    <quality shadowsize="2048" />
    <global offwidth="800" offheight="800" />
  </visual>

  <asset>
{chr(10).join(assets)}
  </asset>

  <worldbody>
{chr(10).join(body)}
  </worldbody>

</mujoco>
"""


# ── Gazebo ───────────────────────────────────────────────────────────────────

POSE_PUBLISHER = """    <!-- Pose del modelo en /model/<nombre>/pose (mundo Gazebo) a 30 Hz: la graba
         lerobot_robot_icubrg como obj{i}_* y la lee eval_policy.py (criterio de éxito). -->
    <plugin filename="gz-sim-pose-publisher-system" name="gz::sim::systems::PosePublisher">
      <publish_link_pose>false</publish_link_pose>
      <publish_model_pose>true</publish_model_pose>
      <use_pose_vector_msg>false</use_pose_vector_msg>
      <update_frequency>30</update_frequency>
    </plugin>"""


def write_model(name: str, sdf_body: str, description: str) -> Path:
    d = MODELS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.sdf").write_text(f'<?xml version="1.0" ?>\n<sdf version="1.7">\n'
                                 f"  <!-- GENERADO por src/tools/pickplace_assets.py: {description} -->\n"
                                 f"{sdf_body}</sdf>\n")
    (d / "model.config").write_text(f"""<?xml version="1.0"?>
<model>
  <name>{name}</name>
  <version>1.0</version>
  <sdf version="1.7">model.sdf</sdf>
  <description>{description}</description>
</model>
""")
    return d


def mesh_files(model_dir: Path, name: str, g: dict) -> None:
    m = model_dir / "meshes"
    m.mkdir(exist_ok=True)
    write_obj(m / f"{name}.obj", g["verts"], g["rest"], f"{name}.mtl")
    (m / f"{name}.mtl").write_text(f"newmtl material\nKa 1 1 1\nKd 1 1 1\nKs 0.3 0.3 0.3\nmap_Kd {g['texture']}\n")
    shutil.copy2(TEXTURE_DIR / g["texture"], m / g["texture"])


def mesh_visual(name: str) -> str:
    return (f'      <visual name="visual">\n'
            f'        <geometry><mesh><uri>meshes/{name}.obj</uri></mesh></geometry>\n'
            f"      </visual>\n")


def gazebo_models(geo: dict) -> None:
    for name, g in geo.items():
        col = g["col"]
        shape = (f"<box><size>{f(2 * col['half'])}</size></box>" if col["type"] == "box"
                 else f"<cylinder><radius>{col['r']:.6g}</radius><length>{col['h']:.6g}</length></cylinder>")
        i = g["inertia"]
        d = write_model(f"pick-object-{name}", f"""  <model name="pick-object">
    <link name="link">
      <inertial>
        <mass>{g['mass']:.6g}</mass>
        <inertia><ixx>{i[0]:.6g}</ixx><ixy>0</ixy><ixz>0</ixz><iyy>{i[1]:.6g}</iyy><iyz>0</iyz><izz>{i[2]:.6g}</izz></inertia>
      </inertial>
      <collision name="collision">
        <geometry>{shape}</geometry>
        <!-- mu=5 como los cubos: DART toma el mínimo con las yemas del iCub (mu=5) -->
        <surface><friction><ode><mu>5.0</mu><mu2>5.0</mu2></ode></friction></surface>
      </collision>
{mesh_visual(name)}    </link>
{POSE_PUBLISHER}
  </model>
""", f"{name} de robosuite PickPlace (pick-object de scene_icub_pickplace.xml)")
        mesh_files(d, name, g)
        # Gazebo: el mundo está girado 180° respecto de MuJoCo, así que ROBOSUITE_QUAT es la identidad
        # y solo queda REPLICA_YAW (el espejo del lado izquierdo es en y, no cambia esta orientación)
        yaw = np.radians(REPLICA_YAW.get(name, 0.0))
        d = write_model(f"visual-{name}", f"""  <model name="visual-{name}">
    <static>true</static>
    <link name="link">
      <pose>0 0 0 0 0 {yaw:.6g}</pose>
{mesh_visual(name)}    </link>
  </model>
""", f"réplica visual de {name} en su compartimento de bin-target (sin colisión)")
        mesh_files(d, name, g)

    for name, dividers, texture in (("bin-source", False, "light-wood.png"), ("bin-target", True, "dark-wood.png")):
        links = []
        for gname, gpos, half in bin_geoms(dividers):
            box = f"<box><size>{f(2 * np.asarray(half))}</size></box>"
            links.append(f"""      <collision name="{gname}_collision">
        <pose>{f(gpos)} 0 0 0</pose>
        <geometry>{box}</geometry>
      </collision>
      <visual name="{gname}_visual">
        <pose>{f(gpos)} 0 0 0</pose>
        <geometry>{box}</geometry>
        <material>
          <diffuse>1 1 1 1</diffuse>
          <pbr><metal><albedo_map>materials/{texture}</albedo_map></metal></pbr>
        </material>
      </visual>""")
        d = write_model(name, f"""  <model name="{name}">
    <static>true</static>
    <link name="link">
{chr(10).join(links)}
    </link>
  </model>
""", f"{'bin1' if name == 'bin-source' else 'bin2 (4 compartimentos)'} de robosuite PickPlace, escala {f(BIN_SCALE)} (x y)")
        (d / "materials").mkdir(exist_ok=True)
        shutil.copy2(TEXTURE_DIR / texture, d / "materials" / texture)


# ── scenes.yaml ──────────────────────────────────────────────────────────────

def yaml_block(geo: dict) -> str:
    hx, hy = BIN_HALF - SPAWN_MARGIN
    lines = ['    - body: "pick-object"',
             "      variants:",
             ]
    for name, g in geo.items():
        lines.append(f"        - {{name: {name}, z: {BASE_TOP_Z + g['size'][2] / 2 + 0.002:.4f}, "
                     f"radius: {g['radius']:.4f}, mass: {g['mass']:.5f}}}")
    lines += ["      spawn:",
              f"        x: [{BIN_X - hx:.4f}, {BIN_X + hx:.4f}]",
              f"        y: [{SOURCE_Y - hy:.4f}, {SOURCE_Y + hy:.4f}]",
              "        yaw: [-180, 180]",
              f"        quat: [{', '.join(str(v) for v in ROBOSUITE_QUAT)}]",
              "  fixtures:"]
    for name, pos in fixtures(geo):
        lines.append(f'    - {{body: "{name}", pos: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]}}')
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robosuite", type=Path, default=REPO_ROOT.parents[1] / "robosuite",
                    help="Clon de robosuite (carpeta con robosuite/models/assets)")
    args = ap.parse_args()
    rs = args.robosuite / "robosuite" if (args.robosuite / "robosuite").is_dir() else args.robosuite
    geo = object_geometry(rs)

    MESH_DIR.mkdir(parents=True, exist_ok=True)
    for name, g in geo.items():
        write_obj(MESH_DIR / f"{name}.obj", g["verts"], g["rest"], None)
    (SCENES_DIR / "scene_icub_pickplace.xml").write_text(mujoco_xml(geo))
    gazebo_models(geo)

    for name, g in geo.items():
        print(f"{name:7s} {f(np.round(g['size'] * 100, 2))} cm  {g['mass'] * 1000:.1f} g  "
              f"radius {g['radius'] * 100:.2f} cm  collision {g['col']['type']}")
    print("\n# scenes.yaml (Pick-and-Place: objects / fixtures)\n" + yaml_block(geo))


if __name__ == "__main__":
    main()
