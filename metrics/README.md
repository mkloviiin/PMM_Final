# Métricas de latencia, jitter y sincronización

Herramientas para medir el loop de teleoperación en **MuJoCo**, **Gazebo** y el **iCub real**.
Todo usa el entorno `icubenv`.

| Script | Qué hace | Cuándo |
|---|---|---|
| `frame_metrics.py` | Registra una línea por frame (periodo, tiempo por etapa, timestamps de estado y cámaras, confirmación de comandos) | **Automático** al grabar desde el Hub o con `play_*.py` |
| `latency_probe.py` | Prueba de escalón: comando → movimiento de la mano | A mano, con el robot/sim listo |
| `metrics_report.py` | Tablas (Markdown + LaTeX) y figuras para el informe | Después de grabar/probar |
| `dataset_check.py` | Integridad y sincronización de datasets ya grabados (también los antiguos) | Cuando quieras |

Todo se escribe en `hub_dataset/logs/metrics/` (fuera del dataset, no se sube a Hugging Face).
`PMM_METRICS=0` desactiva el registro; `PMM_METRICS_DIR=<carpeta>` lo redirige.

## 1. Grabar (métricas por frame)

No hay que hacer nada distinto: al grabar, se crea `hub_dataset/logs/metrics/<dataset>.jsonl` y al
guardar cada episodio aparece en el log del Hub una línea así:

```
[Metrics] ep 0 (saved): 596 frames | 29.8 Hz, period 33.6±2.4 ms (p99 37.6, max 89.2), missed 0.2 % | cmd->goToPose 10.4±5.6 ms ... | new cam frames left 100 %, right 100 %
```

En Gazebo/real hace falta que el **módulo YARP se haya reiniciado** con esta versión
(`teleop_module_sm.py` publica ahora `/teleop/cmd_ack:o` y timestamps YARP). Con un módulo
antiguo todo funciona igual, pero faltan la latencia de comandos y los timestamps.

Para la métrica de seguimiento/retardo por correlación cruzada hacen falta episodios con
movimiento real de la mano (> 5 mm), es decir, teleoperados.

## 2. Prueba de escalón

```bash
conda activate icubenv
python metrics/latency_probe.py --backend mujoco                 # sin nada más corriendo
python metrics/latency_probe.py --backend gazebo                 # sim + módulo YARP (pasos 1 y 2 del Hub)
python metrics/latency_probe.py --backend real --step 0.02       # robot + módulo YARP con --robot icub
```

Por defecto: 10 idas y vueltas de 2 cm en +z (hacia arriba), comandos a 30 Hz como el teleop, y una
pausa aleatoria antes de cada escalón para no sincronizarse con procesos periódicos. Opciones
útiles: `--reps`, `--step` (máx 5 cm), `--axis x|y|z`, `--sign -1`, `--window` (s de medición).

**Robot real:** pide confirmación antes de mover. Despeja el espacio sobre la mano derecha y ten el
paro de emergencia a mano. Ctrl+C devuelve la mano al punto de partida. La sesión de teleop/grabación
del Hub no debe estar enviando targets a la vez (detén el paso 3; el módulo YARP sí debe seguir corriendo).

## 3. Generar el informe

```bash
python metrics/metrics_report.py --logs hub_dataset/logs/hub_*.txt
python metrics/dataset_check.py data/<dataset> [data/<otro> ...] --out hub_dataset/logs/metrics/report
```

Sale en `hub_dataset/logs/metrics/report/`: `resumen.md`, `tablas.tex` (requiere `\usepackage{booktabs}`),
`resumen.json` y `fig_*.png`. Por defecto usa todos los `.jsonl` y `probe_*_trials.csv` de la carpeta;
con `--frames ...` / `--probes ...` eliges cuáles (p. ej. para excluir sesiones de prueba).

## Definiciones (para el informe)

**Loop**
- *Periodo*: tiempo entre inicios de frames consecutivos de un episodio. T₀ = 1/fps.
- *Jitter*: desviación estándar del periodo, y media de |T − T₀|.
- *Frame perdido*: T > 1,5·T₀. *Sobrecarga*: trabajo del frame (sin el sleep) > T₀.
- *Etapas*: `get_observation`, teleop (`get_action` + procesadores), `send_action` y `add_frame`.

**Comando (Gazebo/real)**: el cliente pone en cada target un envelope YARP (secuencia, t_envío);
el módulo responde por `/teleop/cmd_ack:o` con t_recepción y t_goToPose. Cliente y módulo corren
en el mismo PC, así que comparten reloj.
- *Transporte*: t_recepción − t_envío. Incluye la espera al siguiente ciclo del módulo (≤ 20 ms).
- *Hasta goToPose*: transporte + `go_to_pose_async`.
- *Banda muerta*: el módulo no reenvía `goToPose` si la mano ya está a < 1 cm / 5° del target.

**Escalón**: tiempos desde la escritura del primer comando. *Primer movimiento* = desplazamiento
> 1 mm (latencia percibida). *t10/t50/t90* y sobrepaso se miden respecto del desplazamiento
**final alcanzado**. El *error estacionario* (distancia final al target) y el *% logrado* se
reportan aparte. σ entre escalones = jitter de la respuesta.

**Sincronización**
- *Antigüedad del estado*: ahora − instante en que el módulo leyó los encoders.
- *Frame repetido*: el frame grabado reutiliza la imagen anterior porque no llegó una nueva
  (la lectura de cámara es no bloqueante).
- *Frame perdido en origen*: saltos en la secuencia del envelope de la cámara.
- *Antigüedad desde captura* y *desfase cámara − estado*: solo si la fuente comparte reloj con el PC.
  En el robot real las cámaras se fechan con el reloj del PC del robot: si no hay NTP, el informe
  marca "reloj distinto" y quedan solo las métricas relativas (estéreo, repetidos, perdidos).
- *MuJoCo*: estado y cámaras salen del mismo `mjData`, así que el desfase es 0 por construcción
  (se registra para comprobarlo).

## Observaciones de la puesta a punto (para interpretar los números)

- **MuJoCo:** el IK se recalcula solo cada `ik_recompute_steps` (60 llamadas ≈ 1 s) y entremedio
  interpola con mínimo jerk. Un comando nuevo espera a que termine la ventana actual, así que la
  latencia hasta el primer movimiento varía entre ~0,2 y ~1,2 s según la fase.
  Además, cada frame avanza `round(control_dt / sim_dt) = 7` pasos de 5 ms, es decir 35 ms de
  simulación por cada 33,3 ms reales (×1,05). Está en la tabla como "tiempo simulado / tiempo real".
- **Gazebo:** la mano se detiene a 1–2 cm del target aunque el controlador cartesiano reporte
  `desired = target`. Esto aparece como error estacionario en la prueba de escalón; en el log del
  robot real del 05-10, "hand-target" tiene una mediana de 1,4 cm.

## Criterios de éxito de las tareas (`task_success.py`)

`eval_mujoco.py` y `eval_gazebo.py` deciden el éxito con el mismo módulo, que porta el
`_check_success` de robosuite (1.5.2) para cada tarea. Cada simulador solo arma el estado: poses de
los objetos, agarres, contactos entre objetos y articulaciones. Como en los rollouts de robomimic,
un episodio cuenta como éxito si el criterio se cumple **en algún paso**. Además se guarda
`success_at_end`, porque un objeto puede cumplirlo un instante y caerse después.

| Tarea (escena) | robosuite | Criterio en el iCub |
|---|---|---|
| Block Lifting | `Lift` | centro del cubo azul > mesa + 4 cm |
| Block Stacking | `Stack` | azul > mesa + 4 cm, tocando el rojo y sin agarrar |
| Pick-and-Place | `PickPlace` | centro del cubo dentro de la caja (±6 cm en su frame, de la base a +10 cm) y sin agarrar |
| Door Opening | `Door` | bisagra > 0,3 rad |
| Two-Arm Lifting | `TwoArmLift` | fondo de la olla > mesa + 10 cm |
| Two-Arm Peg-In-Hole | `TwoArmPegInHole` | d < 3,6 cm, −12 ≤ t ≤ 14 cm, cos > 0,95 (eje de la clavija vs. agujero) |

**Agarre:** robosuite pide contacto de las dos almohadillas de la pinza. En el iCub cuenta como
agarre que la yema del pulgar y la del índice o el medio toquen el objeto, en cualquiera de las dos
manos. Son los eslabones `*_hand_{thumb,index,middle}_3`: en Gazebo tienen sensores de contacto
(`/icub/touch/<mano>_<dedo>`) y en MuJoCo se leen los contactos de esos mismos cuerpos.

**Contactos entre objetos:** en MuJoCo salen de `mjData.contact`. En Gazebo, del sensor del modelo
(`models/red-cube`, que publica en `/red-cube/contact`).

**Adaptaciones respecto de robosuite:**
- *PickPlace:* robosuite exige además que la pinza quede a más de 4,2 cm del objeto, como señal de
  que lo soltó. En el iCub el sitio de la mano está en la palma, a unos 5 cm de un cubo tomado, así
  que se usa la misma prueba de agarre que en Stack.
- *PegInHole:* la clavija mide 0,13 m, igual que en robosuite, así que t y cos no cambian. El
  agujero mide 6 cm y no 10, por lo que d se escala: 0,06 × 0,03 / 0,05.
- *Umbrales de altura:* se usan tal cual (4 y 10 cm). El cubo del iCub mide 5 cm y el de robosuite
  unos 4, así que en Lift basta levantarlo 1,5 cm, frente a 2 cm en robosuite.
