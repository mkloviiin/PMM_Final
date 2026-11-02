#!/usr/bin/env bash
# Abre / cierra el Hub de experimentación desde un ícono del escritorio (sin terminal).
#
#   hub_launcher.sh            Hub cerrado: lo inicia (pixi run hub) y abre el navegador.
#                              Hub abierto: pregunta si abrirlo en el navegador o cerrarlo.
#   hub_launcher.sh install    Crea el ícono "Multimodal Experimentation Hub" en el escritorio.
#
# El Hub se reconoce por el puerto 7860 (aunque se haya abierto desde una terminal). Cerrarlo
# equivale a Ctrl+C: termina una sesión de grabación en curso; un entrenamiento sigue (corre
# aparte, ver launcher_gui/training.py) y el Hub lo retoma al volver a abrirse.

set -u

HUB_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
ROOT="$(cd "$HUB_DIR/../.." && pwd)"
PORT=7860
URL="http://localhost:$PORT"
LOG="$HUB_DIR/logs/hub_launcher.log"
TITLE="Multimodal Experimentation Hub"
PIXI="$(command -v pixi || echo "$HOME/.pixi/bin/pixi")"

notify() { notify-send -i "$HUB_DIR/assets/robot.png" "$TITLE" "$1" 2>/dev/null || true; }

hub_pid() {
    ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2
}

install_icon() {
    local desktop
    desktop="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
    local file="$desktop/experimentation-hub.desktop"
    cat > "$file" <<EOF
[Desktop Entry]
Type=Application
Name=$TITLE
Comment=Open or close the Multimodal Experimentation Hub
Exec="$HUB_DIR/hub_launcher.sh"
Icon=$HUB_DIR/assets/robot.png
Terminal=false
Categories=Science;
EOF
    chmod +x "$file" "$HUB_DIR/hub_launcher.sh"
    # GNOME (Desktop Icons) solo ejecuta lanzadores marcados como confiables
    gio set "$file" metadata::trusted true 2>/dev/null || true
    echo "Created $file"
}

start_hub() {
    mkdir -p "$(dirname "$LOG")"
    echo "===== $(date '+%F %T') start =====" >> "$LOG"
    # Sesión propia: no depende de ninguna terminal y se cierra entero con stop_hub
    (cd "$ROOT" && setsid "$PIXI" run hub >> "$LOG" 2>&1 < /dev/null &)
    notify "Starting the Hub…"
    for _ in $(seq 1 120); do
        if [ -n "$(hub_pid)" ]; then
            xdg-open "$URL" >/dev/null 2>&1 &
            return 0
        fi
        sleep 1
    done
    zenity --error --title="$TITLE" --width=600 \
        --text="The Hub did not start. Last lines of $LOG:\n\n$(tail -n 15 "$LOG" | sed 's/&/\&amp;/g; s/</\&lt;/g')"
    return 1
}

stop_hub() {
    local pid="$1" pgid target
    pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
    # Como Ctrl+C: SIGINT a todo el grupo (Hub + play.py); si no cierra, SIGTERM y al final
    # SIGKILL. Solo el grupo si su líder es el propio Hub (pixi / python), si no solo el proceso.
    target="$pid"
    if [ -n "$pgid" ] && ps -o args= -p "$pgid" 2>/dev/null | grep -qE 'pixi|python'; then
        target="-$pgid"
    fi
    for sig in INT TERM KILL; do
        kill -s "$sig" -- "$target" 2>/dev/null
        for _ in $(seq 1 20); do
            [ -z "$(hub_pid)" ] && { notify "The Hub was closed."; return 0; }
            sleep 0.5
        done
    done
    zenity --error --title="$TITLE" --text="Could not close the Hub (pid $pid)."
    return 1
}

main() {
    [ "${1:-}" = "install" ] && { install_icon; return 0; }

    local pid choice rc
    pid="$(hub_pid)"
    if [ -z "$pid" ]; then
        start_hub
        return $?
    fi

    choice="$(zenity --question --title="$TITLE" --width=420 \
        --text="The Hub is running.\n\nClosing it ends any recording session in progress. A training keeps running and the Hub picks it up when it is opened again." \
        --ok-label="Close the Hub" --cancel-label="Cancel" --extra-button="Open in browser" 2>/dev/null)"
    rc=$?
    if [ "$choice" = "Open in browser" ]; then
        xdg-open "$URL" >/dev/null 2>&1 &
    elif [ $rc -eq 0 ]; then
        stop_hub "$pid"
    fi
}

# Al hacer `source` (pruebas) solo se cargan las funciones
[ "${BASH_SOURCE[0]}" = "$0" ] && main "$@"
