# Hub v2: prototipo de rediseño de la interfaz

Una segunda versión de la UI del Hub para comparar con la actual y elegir qué llevar a ella.
**Solo cambia la interfaz.** La sesión, los backends de robot, la curación, la subida a Hugging
Face y la vista en vivo son los de `src/hub_dataset/launcher_gui`, sin tocarlos. Se graba
exactamente igual.

```bash
pixi run python src/hub_v2/hub_v2.py      # http://localhost:7861  (HUB_V2_PORT para cambiar el puerto)
```

Puede correr al mismo tiempo que el Hub original (7860), pero **no lances sesiones desde los dos a
la vez**: usan los mismos puertos del VR y del streaming.

## Principios aplicados

| Principio | Qué significa aquí |
|---|---|
| El estado del sistema siempre a la vista | Barra superior fija en todas las pestañas: plataforma, estado de la sesión, episodios, FPS y VR |
| La navegación sigue el flujo de trabajo | Pestañas en el orden real del trabajo: Setup → Record → Review & Curate → Export |
| Reconocer en vez de recordar | Resumen en lenguaje natural antes de lanzar; resumen de cambios antes de exportar |
| Revelar de a poco | Lo técnico o poco usado plegado: FPS y carpeta, métricas, log, comandos del robot |
| Prevenir errores | Lanzar se deshabilita con una sesión en curso; aviso si falta un paso del stack; "End session" pide confirmación |
| Decir qué hacer ahora | El panel de Record dice el siguiente paso: "Ready for episode 4 — press Record (R) or A" |

## Cambios por pantalla

**Pantalla de bienvenida → eliminada.** La plataforma se elige en Setup con tres tarjetas:
MuJoCo, Gazebo y Real. Cambiarla no obliga a pasar por "Back to robot selection". Además, Gazebo
y real dejan de ser un formulario con un selector de modo adentro. Los logos quedan en la barra
superior.

**Setup** (antes: Configuration)
- Campos agrupados por la pregunta que responden: *Task* (muestra la instrucción que se graba),
  *Dataset* y *Control*.
- El nombre del dataset se sugiere a partir de la plataforma y la tarea (`local/icub_mj_block_stacking`).
  Si lo editas a mano, la sugerencia ya no lo cambia.
- El VR es un solo control: *No VR / VR over Wi-Fi / VR over USB cable*. La IP aparece solo con Wi-Fi.
- FPS y carpeta de datos van plegados en *Advanced*.
- Gazebo / real: pasos numerados (1 simulación o conexión, 2 módulo YARP, 3 sesión), cada uno con
  un indicador que se actualiza solo. El paso listo se marca en verde, y "Start session" avisa si
  falta uno.
- Resumen antes de lanzar: cuántos episodios, qué tarea, controlador, cómo se teleopera y dónde
  queda guardado.
- Al lanzar, salta solo a Record.

**Record** (antes: Episode Control)
- Una sola vista: el video grande y la bitácora a la izquierda, el panel de control a la derecha.
- Estados: *Starting…* mientras `play.py` conecta (el original marcaba "Waiting for START" desde el
  primer instante), *Ready for episode N*, *Recording episode N* (rojo, parpadeando), *Session
  finished* y *Session error* (el log se abre solo).
- Atajos de teclado **R** (grabar) y **S** (detener y guardar). No se activan mientras escribes en
  un campo.
- *Robot commands*: home, reset de escena, pausar el teleop y abrir/cerrar cada mano. Antes solo
  existían en la consola de `play.py`, así que ahora se puede grabar sin VR solo desde el Hub.
- Chuleta de los botones del control VR.
- "End session" pide confirmación, porque descarta el episodio en curso.
- Al terminar aparece *Review the recorded episodes →*, que abre el dataset recién grabado en Review.

**Review & Curate** (antes: Visualize & Curate, secciones 1–5)
- Barra compacta arriba: dataset, *Open*, *Last recorded*, ↻ y 🗑. La carpeta va plegada.
- Mientras no hay nada abierto, un mensaje dice qué hacer.
- La página larga pasa a subpestañas: *Episodes*, *Signal plots*, *Data to keep* e *Instruction*.

**Export** (antes: secciones 6–7)
- Arriba, el resumen de lo que cambia respecto del dataset grabado: episodios excluidos,
  features y dimensiones quitadas, e instrucción editada. Así revisas antes de exportar sin
  recordar lo que marcaste en otras pestañas.
- Guardar la copia local (1) y publicar en Hugging Face (2) quedan lado a lado.

## Qué llevaría primero al Hub actual (de más a menos impacto)

1. La barra de estado fija y el panel "qué hacer ahora" de Record.
2. El estado *Starting…* mientras `play.py` conecta (en `session.py` / `_bridge_play`).
3. Los botones de *Robot commands*. Son de bajo costo: `state.cmd_queue.put("h" | "n" | "8" …)`.
4. La plataforma dentro de Configuration, en vez de la pantalla de bienvenida con "Back".
5. Los pasos numerados con indicadores para Gazebo / real.
6. El resumen antes de lanzar y antes de exportar.
7. Las subpestañas en Curate.

## Notas

- **Un detalle de Gradio:** dentro de `gr.Tabs()` todo componente tiene que ir dentro de un
  `gr.Tab`. Un `gr.State` suelto entre pestañas corre el conteo, y la barra muestra una pestaña
  con el nombre de otra.
- `stack.py` repite la lógica de los botones 2.1 / 2.2 de `robots/icub_rg.py`, porque allí están
  definidos dentro del formulario. Si se adopta este diseño, conviene moverlos a funciones del
  módulo en `icub_rg.py` y usarlas desde ambos lados.
- Si el módulo YARP lo inició el otro Hub, la v2 lo muestra como "running, started outside this
  Hub", porque no conoce su solver ni sus brazos.
