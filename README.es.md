# Experimentation Hub

[English](README.md) | **Español**

Plataforma unificada para **grabar, revisar, curar y publicar datasets de robótica** en formato
[LeRobot](https://github.com/huggingface/lerobot). Diseñada para **cualquier robot**:
el flujo de grabación, curación y subida a Hugging Face es compartido, y cada robot
se conecta como un *backend* independiente (ver [Cómo agregar un robot](#cómo-agregar-un-robot-pendiente-de-comprobación)).

Actualmente cuenta con integración completa para el robot humanoide **iCub** en tres plataformas:
simulación en **MuJoCo**, simulación en **Gazebo Sim** y el **iCub real**. Las tres pueden teleoperarse
con un visor de realidad virtual (Meta Quest 2 / Pro) mediante BeaVR, o de forma interactiva usando teclado y mouse en el visor gráfico.

- [Plataformas disponibles](#plataformas-disponibles)
- [Gafas VR](#gafas-vr)
- [Controles en el visor](#controles-en-el-visor)
- [Controles de consola](#controles-de-consola)
- [Instalación](#instalación)
- [Cómo usarlo](#cómo-usarlo)
- [Arquitectura](#arquitectura)
- [Módulos del Hub](#módulos-del-hub)
- [Cómo agregar un robot (Pendiente de comprobación)](#cómo-agregar-un-robot-pendiente-de-comprobación)
- [Datos grabados](#datos-grabados)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Documentación relacionada](#documentación-relacionada)

---

## Plataformas disponibles

| Plataforma (pantalla de inicio) | Backend | Modo de ejecución | Tareas |
|---|---|---|---|
| **iCub Simulated MJ** | `robots/icub_sim.py` | Subproceso independiente (`play.py --backend mujoco`) | Block Lifting, Block Stacking, Pick-and-Place, Door Opening, Two-Arm Peg-In-Hole, Two-Arm Lifting |
| **iCub Real / Gazebo** | `robots/icub_rg.py` | Subproceso independiente conectado por YARP | Block Lifting, Block Stacking, Pick-and-Place, Door Opening, Two-Arm Peg-In-Hole, Two-Arm Lifting |

- Las tareas se declaran de manera unificada en `mujoco_sim/assets/scenes/scenes.yaml`, y Gazebo utiliza el mismo archivo.
  Cada tarea puede elegir sus cámaras (`cameras`): Door Opening graba `left`, `right` y la cámara
  cenital `top` en vez de `front`.
- En *Configuración*, las dos plataformas eligen el **Controller (solver)**, el controlador
  whole-body de BLF de `config/control_config.yaml`: **Whole-Body IK (position)**, **Whole-Body ID
  (torque)** o **Whole-Body ID (torque) + impedance**. En Gazebo / real el solver corre en el
  módulo YARP (paso 2.2), que se inicia con la opción elegida.
- Criterios de éxito de las seis tareas (`_check_success` de robosuite): [metrics/README.md](metrics/README.md).
- Ambos backends graban **exactamente el mismo dataset**: mismas variables, en el mismo marco de coordenadas y con las mismas cámaras (ver [Datos grabados](#datos-grabados)).

---

## Gafas VR

La teleoperación usa la app BeaVR en unas Meta Quest 2 / Quest Pro, conectadas por Wi-Fi
(con la IP de las gafas en *Configuración*) o por cable USB.

### Controles

| Control VR | Acción |
|---|---|
| **Gatillo índice (frontal)** | Cierre de la mano, proporcional: suelto = abierta, a fondo = cerrada (un cierre completo dura `hand_close_time_s` = 2 s, en todas las plataformas) |
| **Gatillo inferior (de agarre)** | Embrague: mientras está apretado la posición de las manos no se mueve y la cabeza queda congelada, para reacomodar el control; al soltarlo la mano sigue desde donde quedó, sin salto. La orientación sigue al control |
| **A** | Iniciar grabación de episodio |
| **B** | Detener y guardar episodio (mantener presionado **B** por 1.5s para descartar) |
| **Y** | Reposicionar / aleatorizar la posición de los objetos en la mesa |
| **X** | Panel de ayuda con estos botones en la pantalla grande (también aparece con la app BeaVR en pausa) |

La pantalla grande muestra el ojo del robot con un campo de visión más amplio que la cámara que
se graba (`vr_head_fovy`, `head_cam_vr` en MuJoCo / cámara VR en Gazebo) y el estado de la
grabación en la esquina; la pantalla chica muestra la cámara frontal.

### Conexión por cable USB (una sola vez)

La conexión por cable pasa por ADB, así que el computador necesita las herramientas de Android
y permiso para acceder a las gafas por USB. Estos pasos se hacen en la máquina anfitriona,
también cuando el Hub corre en Docker.

1. Activa el **modo desarrollador** de las gafas desde la app Meta Horizon del celular
   (*Dispositivos → Configuración de las gafas → Modo desarrollador*).
2. Instala ADB.
   ```bash
   sudo apt install adb
   ```
3. Agrega una regla udev para que tu usuario acceda a las gafas (`2833` es el vendor ID USB de
   Meta) y súmate al grupo `plugdev`. Luego cierra sesión y vuelve a entrar.
   ```bash
   echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="2833", MODE="0666", GROUP="plugdev"' \
     | sudo tee /etc/udev/rules.d/51-meta-quest.rules
   sudo udevadm control --reload-rules && sudo udevadm trigger
   sudo usermod -aG plugdev $USER
   ```
4. Conecta el cable, ponte las gafas y acepta **Permitir depuración USB**
   (marca *Permitir siempre desde este computador*).
5. Revisa que las gafas aparezcan como `device`.
   ```bash
   adb devices
   ```
   Si aparecen como `unauthorized`, vuelve a aceptar el aviso dentro de las gafas. Si no
   aparecen, revisa el cable y la regla udev.

---

## Controles en el visor

Teleoperación interactiva con mouse y teclado dentro de la ventana del visor de MuJoCo (válido para backends MuJoCo y Gazebo):

| Entrada | Acción |
|---|---|
| **Doble clic izquierdo** | Seleccionar efector mocap (`r_hand_target` o `l_hand_target`) |
| **Ctrl + Clic Derecho (arrastrar)** | Trasladar posición cartesiana del efector ($X, Y, Z$) |
| **Ctrl + Clic Izquierdo (arrastrar)** | Rotar orientación del efector ($SO(3)$) |
| Tecla **8** | Abrir pinza de la mano derecha |
| Tecla **9** | Cerrar pinza de la mano derecha |
| Tecla **6** | Abrir pinza de la mano izquierda |
| Tecla **7** | Cerrar pinza de la mano izquierda |

*(Nota: Los comandos de estado del sistema como volver a home, pausar/activar teleoperación y resetear objetos se gestionan desde la consola).*

---

## Controles de consola

Comandos interactivos ingresados en la terminal donde se ejecuta `play.py` (idénticos para MuJoCo y Gazebo):

| Tecla / Comando | Acción |
|---|---|
| `1` | Iniciar grabación de episodio |
| `2` | Detener y guardar episodio grabado |
| `3` | Salir limpiamente de la teleoperación |
| `h` | Volver a la postura de inicio (*home*) y reiniciar los mocaps |
| `n` | Resetear la posición de los objetos en la mesa |
| `t` | Activar o pausar la teleoperación (*Active* / *Idle*) |
| `i` | Alternar modo de impedancia cartesiana (cuando se usa `--controller id`) |
| `8` / `9` | Abrir / cerrar mano derecha |
| `6` / `7` | Abrir / cerrar mano izquierda |

---

## Instalación

### Con Pixi (Recomendado)

El workspace utiliza [Pixi](https://pixi.sh) para gestionar de forma nativa y hermética todas las dependencias de robótica, física y ML (MuJoCo, YARP, Gazebo Sim, BLF, LeRobot, PyTorch con CUDA):

```bash
# 1. Instalar dependencias
pixi install

# 2. Iniciar la interfaz web del Experimentation Hub
pixi run hub            # http://localhost:7860
```

### Con Docker

Una sola imagen contiene el Hub, MuJoCo, Gazebo y el cliente YARP para el robot real:

```bash
docker compose build                  # construcción inicial o tras cambios de código
xhost +SI:localuser:$(id -un)         # permite al contenedor abrir ventanas en X11
docker compose up                     # inicia el Hub en http://localhost:7860
```

---

## Cómo usarlo

El Hub sigue el orden de sus pestañas numeradas. En la pantalla de inicio se elige la
plataforma, en **1. Configuración** se prepara y lanza la sesión (tarea, brazos, solver, VR),
en **2. Control de episodios** se graban los episodios con las gafas VR y en
**3. Visualizar y curar** se revisa, cura y publica el dataset. El recorrido completo, con
capturas de cada pestaña, está en el [Manual de usuario](docs/MANUAL.es.md).

Desde la línea de comandos, `play.py` graba (o solo teleopera, con `--no-record`) de la misma forma:

### MuJoCo

Lanzar la teleoperación bímana interactiva en MuJoCo:

```bash
# 1. Con Cinemática Inversa QP (TSIK - iCubWholeBodyIK):
pixi run python play.py --backend mujoco --controller ik --control-arms both --no-record

# 2. Con Dinámica Inversa QP (TSID - iCubWholeBodyID):
pixi run python play.py --backend mujoco --controller id --control-arms both --no-record

# 3. Con TSID + Modo de Impedancia Cartesiana compliant:
pixi run python play.py --backend mujoco --controller id --impedance --control-arms both --no-record
```

#### Argumentos soportados por `play.py`:

| Opción | Valores / Por defecto | Descripción |
|---|---|---|
| `--backend` | `mujoco`, `gazebo`, `real` (defecto: `mujoco`) | Backend de simulación o robot físico |
| `--controller` | `ik`, `id` (defecto: `ik`) | Controlador de cuerpo completo: `ik` (cinemática TSIK) o `id` (dinámica TSID) |
| `--impedance` | bandera | Activa el modo de impedancia cartesiana pasiva (requiere `--controller id`) |
| `--control-arms` | `right`, `left`, `both` (defecto: `both`) | Brazo(s) activos a teleoperar |
| `--no-record` | bandera | Ejecutar teleoperación interactiva sin grabar dataset |
| `--vr` | bandera | Habilita teleoperación con casco de realidad virtual Meta Quest |
| `--vr-cable` | bandera | Configura automáticamente la conexión por cable USB-C mediante ADB reverse |
| `--vr-ip` | IP (ej. `192.168.1.50`) | Conexión inalámbrica por Wi-Fi con el visor Meta Quest |
| `--scene` | entero o etiqueta (ej. `1` o `lift`) | Escena de tarea predefinida en `scenes.yaml` |
| `--single-task` | texto | Descripción textual en lenguaje natural de la tarea |
| `--fps` | entero (defecto: `30`) | Frecuencia en Hz del bucle de control y grabación |
| `--num-episodes` | entero (defecto: `50`) | Número objetivo de episodios a grabar antes de terminar |
| `--episode-time-s` | decimal (defecto: `0.0`) | Duración máxima por episodio en segundos (`0` = control manual) |
| `--repo-id` | texto | Identificador / nombre del repositorio del dataset LeRobot |
| `--root` | ruta de carpeta | Directorio base de salida para los datasets grabados |

---

### Gazebo / Real

Para teleoperar el iCub en Gazebo Sim con ambos brazos:

#### Paso 1: Lanzar Gazebo Sim (Terminal 1)

```bash
./gazebo_sim/scripts/launch_sim.sh --world manip
```

##### Argumentos soportados por `launch_sim.sh`:

| Opción | Valores / Por defecto | Descripción |
|---|---|---|
| `-w, --world` | `manip`, `empty` (defecto: `manip`) | Mundo de simulación: `manip` (arena con mesa, objetos y cámara) o `empty` |
| `-m, --model` | nombre de modelo | Modelo del iCub: `iCubGazeboV2_5_visuomanip` (defecto en manip), `iCubGazeboV2_5_fixed`, `iCubGazeboV2_7_fixed`, `iCubGazeboV2_5`, `iCubGazeboV2_7` |
| `--stop`, `--kill` | bandera | Detiene y limpia todos los procesos de Gazebo Sim (`gz sim`) y YARP (`yarpserver`) |
| `-h, --help` | bandera | Muestra el mensaje de ayuda y finaliza |

#### Paso 2: Lanzar la Teleoperación (Terminal 2)

```bash
# 1. Con Cinemática Inversa QP (TSIK - iCubWholeBodyIK):
pixi run python play.py --backend gazebo --controller ik --control-arms both --no-record

# 2. Con Dinámica Inversa QP (TSID - iCubWholeBodyID):
pixi run python play.py --backend gazebo --controller id --control-arms both --no-record

# 3. Con TSID + Modo de Impedancia Cartesiana compliant:
pixi run python play.py --backend gazebo --controller id --impedance --control-arms both --no-record
```

*(Nota: Si `teleop_module_sm.py` no está corriendo en YARP, `play.py` lo levantará automáticamente en segundo plano con el solver BLF seleccionado).*

### Evaluar una política

`eval_policy.py` corre una política LeRobot entrenada en cualquier plataforma, con las mismas
opciones de solver, y mide el éxito de la tarea con el criterio de robosuite
([metrics/README.md](metrics/README.md)): en algún paso y al final de cada episodio. Cada episodio
parte en home con la escena reiniciada; las cámaras, los brazos y el orden del estado salen del
checkpoint y de su dataset de entrenamiento.

```bash
pixi run python eval_policy.py --checkpoint outputs/train/<run>/checkpoints/020000/pretrained_model \
    --scene "Door Opening" --num-episodes 20
pixi run python eval_policy.py --checkpoint ... --backend gazebo --scene "Block Lifting" --controller id --impedance
```

Los resultados quedan en `outputs/eval/<run>_<backend>_<fecha>/` (`eval.log`, `summary.json`). En
el robot real el operador ubica los objetos y responde si cada episodio fue exitoso.

---

## Arquitectura

```mermaid
flowchart LR
    U["Navegador<br/>localhost:7860"] --> APP

    subgraph HUB["hub_dataset/launcher_gui (Gradio)"]
        APP["app.py<br/>inicio + pestañas"] --> REG["robots/registry.py"]
        APP --> SES["session.py<br/>sesión de grabación"]
        APP --> CUR["curation.py<br/>visualizar y curar"]
        APP --> PUSH["hub_push.py<br/>subida a Hugging Face"]
        REG --> SIM["robots/icub_sim.py"]
        REG --> RG["robots/icub_rg.py"]
    end

    SIM -- "record_fn → subproceso" --> PMJ["play.py (--backend mujoco)<br/>MuJoCo + BLF"]
    RG -- "record_fn → subproceso" --> PGZ["play.py (--backend gazebo)<br/>YARP + BLF"]
    PGZ <-->|YARP| YM["teleop_module_sm.py<br/>Whole-Body Solver"]
    YM <--> ROBOT["Gazebo / iCub Real"]
    VR["Meta Quest + BeaVR"] -. ZMQ .-> PMJ
    VR -. ZMQ .-> PGZ

    PMJ --> DS[("Dataset LeRobot<br/>data/")]
    PGZ --> DS
    DS --> CUR --> DC[("Dataset curado<br/>data_curated/")]
    DS --> PUSH
    DC --> PUSH --> HF[("Hugging Face Hub")]
```

---

## Módulos del Hub

Módulos principales ubicados en `src/hub_dataset/launcher_gui/`:

| Módulo | Función |
|---|---|
| `app.py` | Construye la interfaz gráfica (pantalla de inicio, pestañas, CSS) y conecta componentes |
| `robots/base.py`, `robots/registry.py` | Contrato base de backends y registro de robots disponibles |
| `robots/icub_sim.py`, `robots/icub_rg.py` | Formulario y lanzamiento de cada plataforma del iCub |
| `robots/solver.py` | Selector *Controller (solver)* compartido por las dos plataformas |
| `session.py` | Hilo de sesión, cola de comandos, estado y sondeo de control de episodios |
| `session_log.py` | Métricas por episodio y cuaderno de notas (`hub_session/`) |
| `curation.py` | Carga de datasets, navegación por episodios, gráficos y empaquetado curado |
| `dataset_curation.py` | Empaquetado del dataset curado (episodios, features y dimensiones eliminadas, instrucciones) |
| `hub_push.py` | Subida de datasets a Hugging Face Hub |
| `video_sync.py` | Reproductor sincronizado de las tres cámaras con las señales cinemáticas |
| `procman.py` | Gestión de procesos externos de larga duración (`teleop_module_sm.py`, `launch_sim.sh`) |
| `ports.py` | Verificación y liberación de puertos de red para VR |
| `state.py` | Estado global compartido entre los módulos del Hub |
| `ui/` | Definición de layout de pestañas, selectores de carpetas y campos numéricos |

---

## Cómo agregar un robot (Trabajo futuro)

> [!NOTE]
> Esta sección describe el diseño de la arquitectura modular multi-robot; su funcionamiento con un segundo robot físico o simulado aún está pendiente de comprobación experimental.

Para agregar una nueva plataforma robótica, se crea su backend y se registra en el Hub:

**1. Crear `hub_dataset/launcher_gui/robots/my_robot.py`:**

```python
from datetime import datetime
from pathlib import Path
import gradio as gr
from .. import session
from ..paths import ASSETS_DIR, HUB_ROOT
from .base import RobotBackend

def build_config_form() -> dict[str, gr.components.Component]:
    gr.Markdown("### 1. Parámetros de la sesión")
    repo_id = gr.Textbox(value="local/mi_robot_demo", label="Repo ID")
    num_eps = gr.Number(value=50, label="Número de episodios", precision=0, minimum=1)
    gr.Markdown("### 2. Lanzar sesión")
    launch_btn = gr.Button("LANZAR SESIÓN", variant="primary", size="lg")
    launch_status = gr.Textbox(label="Estado", interactive=False)
    return dict(repo_id=repo_id, num_eps=num_eps,
                launch_btn=launch_btn, launch_status=launch_status)

def launch(repo_id, num_eps) -> str:
    if session.state.running:
        return "Ya hay una sesión en ejecución."
    rid = repo_id.strip() if "/" in repo_id else f"local/{repo_id.strip()}"
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dataset_root = HUB_ROOT.parent / "data" / f"{rid.replace('/', '_')}_{stamp}"

    def record_fn(cmd_source, on_status):
        ...

    return session.start_session(record_fn, repo_id=rid, dataset_root=dataset_root,
                                 num_episodes=int(num_eps))

BACKEND = RobotBackend(
    id="my_robot",
    label="Mi Robot",
    image=ASSETS_DIR / "my_robot.png",
    available=True,
    build_config_form=build_config_form,
    launch=launch,
    config_input_order=["repo_id", "num_eps"],
)
```

**2. Registrarlo en `hub_dataset/launcher_gui/robots/registry.py`:**

```python
from . import icub_rg, icub_sim, my_robot

REGISTRY = [icub_sim.BACKEND, icub_rg.BACKEND, my_robot.BACKEND]
```

---

## Datos grabados

Ambos backends del iCub graban las mismas variables, garantizando compatibilidad sim-to-sim:

- **Acción (`action`):** Pose objetivo de cada mano (posición cartesiana $[x, y, z]$ y cuaternión unitario $[w, x, y, z]$ en el marco `root` del robot), nivel de cierre de pinzas ($[0.0, 1.0]$) y orientación de la cabeza.
- **Estado (`state`):**
  - Posiciones, velocidades y pares de las articulaciones.
  - Poses cartesianas de los efectores finales y comandos previos enviados.
  - Poses *ground-truth* de los objetos de la escena (en simulación).
- **Cámaras:** `left` y `right` (ojos del robot) y `front` (cámara externa de mesa), en RGB $320 \times 240$. Una tarea puede elegir otras en `scenes.yaml` (`cameras`), p. ej. la cenital `top` de Door Opening.
- **Objetos de escena ausentes en una tarea** (p. ej. sin cubo en Door Opening) se graban en cero.
- **Selección de brazos:** Solo se registran los brazos seleccionados mediante `--control-arms`.
- **Canonicidad de cuaterniones:** Parte escalar positiva ($q_w \geq 0$) para eliminar saltos de discontinuidad de signo en modelos de difusión.

---

## Estructura del repositorio

```
PMM_Final/
├── pixi.toml                    # Configuración de entornos y tareas de Pixi
├── pixi.lock                    # Archivo de bloqueo determinista de dependencias
├── play.py                      # Punto de entrada unificado para teleoperación y grabación
├── eval_policy.py               # Evaluación de políticas entrenadas
├── train_policy.py              # Script de entrenamiento LeRobot
├── replay_mujoco.py             # Reproducción de episodios grabados en MuJoCo
├── config/
│   └── control_config.yaml      # Configuración de solvers BLF, cámaras y teleoperación
├── src/                         # Código fuente unificado
│   ├── controllers/             # Solvers de Cinemática (TSIK) y Dinámica (TSID) con BLF
│   ├── interfaces/              # Módulo YARP (teleop_module_sm.py), BLFRobotBridge y VR
│   ├── plugins/                 # Plugins de LeRobot (icub_mujoco, icub_remote, icubteleop)
│   ├── hub_dataset/             # Aplicación web Gradio del Experimentation Hub
│   ├── trajectories/            # Generación de trayectorias analíticas
│   └── tools/                   # Utilidades de cuaterniones y transformadas
├── mujoco_sim/                  # Archivos de simulación MuJoCo (escenas XML, mallas OBJ)
├── gazebo_sim/                  # Archivos de simulación Gazebo (mundos SDF, modelos, plugins)
├── metrics/                     # Latencia, jitter, métricas por frame y criterios de éxito de las tareas (task_success.py)
├── docs/                        # Manual de usuario del Hub
├── docker/                      # Configuración alternativa de contenedores Docker
├── data/                        # Datasets grabados (no versionados)
└── data_curated/                # Datasets curados (no versionados)
```

---

## Documentación relacionada

| Documento | Contenido |
|---|---|
| [docs/MANUAL.es.md](docs/MANUAL.es.md) | Manual de usuario del Hub, pestaña por pestaña, con capturas |
| [docker/README.md](docker/README.md) | Contenedor Docker: volúmenes, robot real, puertos VR y actualización |
| [gazebo_sim/README.md](gazebo_sim/README.md) | Configuración de assets y entornos de simulación en Gazebo Sim |
| [metrics/README.md](metrics/README.md) | Medición de latencia, jitter y sincronización de grabaciones |
