# syntax=docker/dockerfile:1.7
#
# Imagen del proyecto: Hub de experimentación (Gradio) + MuJoCo + Gazebo + robot
# real vía YARP, todo sobre el entorno conda `icubenv` congelado en
# docker/conda-linux-64.lock y docker/requirements.lock.txt (ver freeze_env.sh).
#
# Construir y usar desde la raíz del repo:  docker compose build / docker compose up
# Instrucciones completas: docker/README.md

ARG UBUNTU_VERSION=22.04

# ─────────────────────────────────────────────────────────────────────────────
# Etapa 1: sistema base + entorno conda icubenv
# ─────────────────────────────────────────────────────────────────────────────
FROM ubuntu:${UBUNTU_VERSION} AS env

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Librerías GL del sistema (glvnd): el NVIDIA Container Toolkit inyecta los
# drivers y glvnd los despacha. tmux/procps/psmisc los usan start_sim.sh,
# stop_sim.sh y procman.py (tmux, pgrep, pkill).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git tmux procps psmisc iproute2 iputils-ping less nano \
        libglvnd0 libgl1 libglx0 libegl1 libgles2 libopengl0 \
        libx11-6 libx11-xcb1 libxext6 libxrender1 libxi6 libxrandr2 libxcursor1 \
        libxinerama1 libxkbcommon0 libxkbcommon-x11-0 libxcb1 libxcb-xinerama0 \
        libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-xkb1 \
        libxcb-randr0 libxcb-shape0 libxcb-xfixes0 libsm6 libice6 libfontconfig1 \
        libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*

# Miniforge solo aporta el binario de conda: las versiones del entorno las fija
# el lockfile explícito (URLs y builds exactos), no el solver.
RUN curl -fsSL -o /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh

ENV ICUBENV_PREFIX=/opt/conda/envs/icubenv

COPY docker/conda-linux-64.lock /tmp/conda-linux-64.lock
RUN /opt/conda/bin/conda create -y -p ${ICUBENV_PREFIX} --file /tmp/conda-linux-64.lock \
    && /opt/conda/bin/conda clean -afy \
    && rm /tmp/conda-linux-64.lock

# Paquetes pip con versión exacta. --no-deps: el lockfile ya trae el árbol
# completo; así pip no re-resuelve ni cambia versiones. --ignore-installed: en
# icubenv algunos paquetes pip pisan a su par de conda (p. ej. opencv-python-headless
# 4.12.0.88 sobre el de py-opencv), y pip no puede desinstalar lo que instaló conda.
# Algunos (evdev, que usa pynput) no traen wheel para Python 3.12: el compilador
# se instala solo para este paso y se quita en la misma capa.
COPY docker/requirements.lock.txt /tmp/requirements.lock.txt
RUN apt-get update && apt-get install -y --no-install-recommends build-essential linux-libc-dev \
    && ${ICUBENV_PREFIX}/bin/pip install --no-cache-dir --no-deps --ignore-installed -r /tmp/requirements.lock.txt \
    && apt-get purge -y --auto-remove build-essential linux-libc-dev \
    && rm -rf /var/lib/apt/lists/* /tmp/requirements.lock.txt

# lerobot en el commit con el que se grabaron y probaron los datasets
# (v0.5.0+18). No es el submódulo lerobot/ del repo.
ARG LEROBOT_COMMIT=017ff73fbfe46bf9a673cd9b402988dcb79151f7
RUN ${ICUBENV_PREFIX}/bin/pip install --no-cache-dir --no-deps \
        "lerobot @ git+https://github.com/huggingface/lerobot.git@${LEROBOT_COMMIT}"

# ─────────────────────────────────────────────────────────────────────────────
# Etapa 2: controlboard de gz-sim-yarp-plugins parchado (modo MIXED, ojos)
# Mismo toolchain con el que se compiló el .so original: g++ 11.4 de Ubuntu
# 22.04 + cmake 4.1.3. Las herramientas quedan solo en esta etapa.
# ─────────────────────────────────────────────────────────────────────────────
FROM env AS gzyarp-plugin

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*
RUN /opt/conda/bin/conda install -y -n base -c conda-forge cmake=4.1.3

COPY icub-lerobot-gazebo/gazebo/plugins/build_controlboard.sh \
     icub-lerobot-gazebo/gazebo/plugins/controlboard_mixed_mode.patch /tmp/plugins/
RUN PATH=/opt/conda/bin:$PATH CONDA_PREFIX=${ICUBENV_PREFIX} \
        bash /tmp/plugins/build_controlboard.sh /opt/gzyarp-patched

# ─────────────────────────────────────────────────────────────────────────────
# Etapa 3: imagen final
# ─────────────────────────────────────────────────────────────────────────────
FROM env AS final

COPY --from=gzyarp-plugin /opt/gzyarp-patched /opt/gzyarp-patched

# Usuario con el mismo UID/GID que el del host: los datasets que se escriben en
# los volúmenes quedan a su nombre y X11 lo acepta con `xhost +SI:localuser:...`.
ARG USERNAME=icub
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${USERNAME} 2>/dev/null || true \
    && useradd -m -s /bin/bash -u ${UID} -g ${GID} ${USERNAME} \
    && mkdir -p /tmp/runtime-${USERNAME} && chown ${UID}:${GID} /tmp/runtime-${USERNAME} \
    && chmod 700 /tmp/runtime-${USERNAME}

# Shells interactivas (las ventanas tmux de start_sim.sh) con conda disponible
# y icubenv activo.
RUN printf '\n# conda (icubenv)\nsource /opt/conda/etc/profile.d/conda.sh\nconda activate icubenv\n' \
        >> /etc/bash.bashrc

ENV PATH=${ICUBENV_PREFIX}/bin:/opt/conda/bin:${PATH} \
    SHELL=/bin/bash \
    PYTHONUNBUFFERED=1 \
    ICUBENV_SITE_PACKAGES=${ICUBENV_PREFIX}/lib/python3.12/site-packages \
    GZ_PATCHED_PLUGIN_DIR=/opt/gzyarp-patched \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    __GLX_VENDOR_LIBRARY_NAME=nvidia \
    __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d:${ICUBENV_PREFIX}/share/glvnd/egl_vendor.d \
    QT_X11_NO_MITSHM=1 \
    XDG_RUNTIME_DIR=/tmp/runtime-${USERNAME}

WORKDIR /workspace/PMM_Final
COPY --chown=${UID}:${GID} . /workspace/PMM_Final
# Carpetas que se montan como volumen (si no se montan, quedan dentro del contenedor).
RUN mkdir -p data data_curated outputs output hub_dataset/logs \
    && chown -R ${UID}:${GID} /workspace/PMM_Final

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER ${USERNAME}
EXPOSE 7860
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "hub_dataset/hub_experimentacion.py"]
