#!/usr/bin/env python3
"""Curacion de datasets LeRobot ya grabados: eliminar episodios, features completas,
dimensiones individuales dentro de una feature vectorial (ej. una sola señal de
`action`/`observation.state`), editar la Language Instruction y empaquetar el resultado
final en una carpeta LeRobot lista para `push_to_hub`.

No modifica el dataset original: cada operacion escribe una copia nueva.
"""

from __future__ import annotations

import copy
import logging
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.dataset_tools import (
    _copy_episodes_metadata_and_stats,
    _copy_videos,
    _write_parquet,
    delete_episodes,
    modify_tasks,
    remove_feature,
)
from lerobot.datasets.io_utils import load_stats, write_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DATA_DIR, DEFAULT_DATA_PATH
from lerobot.utils.constants import HF_LEROBOT_HOME

logger = logging.getLogger(__name__)

# Claves de estadisticas por dimension. "count" queda afuera: es un escalar por
# episodio (numero de frames), no esta indexado por dimension de la feature.
STAT_KEYS_PER_DIM = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")


def drop_feature_dimensions(
    dataset: LeRobotDataset,
    dims_to_drop: dict[str, list[str]],
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    """Elimina dimensiones nombradas individuales dentro de features vectoriales
    (ej. quitar 'rh_pos_x' de 'action' sin tocar el resto de sus 11 dimensiones).

    A diferencia de `remove_feature`/`modify_features` de lerobot (que solo pueden
    eliminar una feature completa), esto opera a nivel de indice dentro del vector.

    Args:
        dataset: Dataset LeRobot origen.
        dims_to_drop: mapa feature_name -> lista de nombres de dimension a eliminar.
        output_dir: carpeta raiz del dataset resultante. Por defecto $HF_LEROBOT_HOME/repo_id.
        repo_id: repo_id del dataset resultante. Por defecto f"{dataset.repo_id}_dimtrim".

    Returns:
        Nuevo LeRobotDataset con las dimensiones eliminadas.
    """
    if not dims_to_drop:
        raise ValueError("No dimensions to drop")

    keep_idx: dict[str, list[int]] = {}
    new_names: dict[str, list[str]] = {}

    for feature_name, drop_names in dims_to_drop.items():
        if not drop_names:
            continue
        if feature_name not in dataset.meta.features:
            raise ValueError(f"Feature '{feature_name}' not found in dataset")

        feat = dataset.meta.features[feature_name]
        if feat["dtype"] in ("video", "image"):
            raise ValueError(f"Cannot drop dimensions from a {feat['dtype']} feature: '{feature_name}'")

        names = list(feat.get("names") or [])
        if not names:
            raise ValueError(f"Feature '{feature_name}' has no named dimensions to drop")

        unknown = set(drop_names) - set(names)
        if unknown:
            raise ValueError(f"Unknown dimension names for '{feature_name}': {sorted(unknown)}")

        drop_set = set(drop_names)
        idx = [i for i, n in enumerate(names) if n not in drop_set]
        if not idx:
            raise ValueError(
                f"Cannot drop every dimension of '{feature_name}' (use remove_feature to drop it entirely)"
            )

        keep_idx[feature_name] = idx
        new_names[feature_name] = [names[i] for i in idx]

    if not keep_idx:
        raise ValueError("No dimensions to drop")

    if repo_id is None:
        repo_id = f"{dataset.repo_id}_dimtrim"
    output_dir = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / repo_id

    new_features = copy.deepcopy(dataset.meta.features)
    for feature_name, idx in keep_idx.items():
        new_features[feature_name]["shape"] = (len(idx),)
        new_features[feature_name]["names"] = new_names[feature_name]

    logger.info(f"Dropping dimensions {dims_to_drop} from dataset {dataset.repo_id}")

    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=dataset.meta.fps,
        features=new_features,
        robot_type=dataset.meta.robot_type,
        root=output_dir,
        use_videos=len(dataset.meta.video_keys) > 0,
    )

    _copy_data_with_sliced_columns(dataset, new_meta, keep_idx)

    if new_meta.video_keys:
        _copy_videos(dataset, new_meta)

    # _copy_data_with_sliced_columns already copied episodes/tasks/info via
    # _copy_episodes_metadata_and_stats, but since the *set* of feature keys is
    # unchanged (only shapes shrink), that helper copies stats verbatim/unsliced.
    # Fix that up here with an exact per-index slice (cheaper and more correct
    # than recomputing stats from scratch: each dimension's min/max/mean/std is
    # independent of the others).
    _slice_stats_in_place(new_meta.root, keep_idx)

    return LeRobotDataset(
        repo_id=repo_id,
        root=output_dir,
        image_transforms=dataset.image_transforms,
        delta_timestamps=dataset.delta_timestamps,
        tolerance_s=dataset.tolerance_s,
    )


def _copy_data_with_sliced_columns(
    dataset: LeRobotDataset,
    new_meta: LeRobotDatasetMetadata,
    keep_idx: dict[str, list[int]],
) -> None:
    """Copia los parquet de data/, recortando las columnas vectoriales en keep_idx.

    Calcado de `dataset_tools._copy_data_with_feature_changes`, pero en vez de
    eliminar columnas completas (`df.drop(columns=...)`) recorta cada array en el
    indice indicado, preservando la feature con menos dimensiones.
    """
    data_dir = dataset.root / DATA_DIR
    parquet_files = sorted(data_dir.glob("*/*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    for src_path in parquet_files:
        df = pd.read_parquet(src_path).reset_index(drop=True)

        for feature_name, idx in keep_idx.items():
            idx_arr = np.array(idx)
            df[feature_name] = df[feature_name].apply(
                lambda arr, idx_arr=idx_arr: np.asarray(arr)[idx_arr]
            )

        relative_path = src_path.relative_to(dataset.root)
        chunk_idx = int(relative_path.parts[1].split("-")[1])
        file_idx = int(relative_path.parts[2].split("-")[1].split(".")[0])

        dst_path = new_meta.root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        _write_parquet(df, dst_path, new_meta)

    _copy_episodes_metadata_and_stats(dataset, new_meta)


def _slice_stats_in_place(root: Path, keep_idx: dict[str, list[int]]) -> None:
    """Recorta meta/stats.json y las columnas stats/<feature>/<stat> de
    meta/episodes/*.parquet a los mismos indices que se conservaron en los datos."""
    stats = load_stats(root)
    if stats:
        changed = False
        for feature_name, idx in keep_idx.items():
            if feature_name not in stats:
                continue
            idx_arr = np.array(idx)
            for stat_name in STAT_KEYS_PER_DIM:
                if stat_name in stats[feature_name]:
                    stats[feature_name][stat_name] = np.asarray(stats[feature_name][stat_name])[idx_arr]
                    changed = True
        if changed:
            write_stats(stats, root)

    episodes_dir = root / "meta" / "episodes"
    for parquet_path in sorted(episodes_dir.rglob("*.parquet")):
        df = pd.read_parquet(parquet_path)
        changed = False
        for feature_name, idx in keep_idx.items():
            idx_arr = np.array(idx)
            for stat_name in STAT_KEYS_PER_DIM:
                col = f"stats/{feature_name}/{stat_name}"
                if col in df.columns:
                    df[col] = df[col].apply(lambda arr, idx_arr=idx_arr: np.asarray(arr)[idx_arr])
                    changed = True
        if changed:
            df.to_parquet(parquet_path, index=False)


def package_dataset(
    source_root: str | Path,
    source_repo_id: str,
    *,
    episodes_to_drop: list[int] | None = None,
    whole_features_to_drop: list[str] | None = None,
    dims_to_drop: dict[str, list[str]] | None = None,
    task_global: str | None = None,
    episode_tasks: dict[int, str] | None = None,
    final_repo_id: str,
    final_root: str | Path,
    on_log: Callable[[str], None] | None = None,
) -> Path:
    """Aplica ediciones pendientes a una sesion grabada y produce el dataset final.

    Encadena, en orden: eliminar episodios -> eliminar features completas ->
    eliminar dimensiones sueltas -> editar tareas (esta ultima in-place, siempre
    al final, porque es la unica operacion de lerobot que no copia el dataset).
    Nunca muta `source_root`: cada etapa escribe en una carpeta temporal nueva,
    y el resultado final se mueve (no se vuelve a copiar) a `final_root`.
    """

    def _log(msg: str) -> None:
        logger.info(msg)
        if on_log is not None:
            on_log(msg)

    source_root = Path(source_root)
    final_root = Path(final_root)
    if final_root.exists():
        raise FileExistsError(f"El destino ya existe: {final_root}")

    dims_to_drop = dict(dims_to_drop or {})
    if whole_features_to_drop:
        dims_to_drop = {k: v for k, v in dims_to_drop.items() if k not in whole_features_to_drop}

    tmp_root = final_root.parent / f"_tmp_curation_{final_root.name}"
    tmp_root.mkdir(parents=True, exist_ok=True)

    current = LeRobotDataset(repo_id=source_repo_id, root=source_root)
    cleanup_roots: list[Path] = []
    stage = 0

    try:
        if episodes_to_drop:
            _log(f"Eliminando {len(episodes_to_drop)} episodio(s)...")
            stage += 1
            next_dataset = delete_episodes(
                current,
                episodes_to_drop,
                output_dir=tmp_root / f"stage{stage}_episodes",
                repo_id=f"{final_repo_id}_stage{stage}",
            )
            if current.root != source_root:
                cleanup_roots.append(current.root)
            current = next_dataset

        if whole_features_to_drop:
            _log(f"Eliminando feature(s) completa(s): {whole_features_to_drop}")
            stage += 1
            next_dataset = remove_feature(
                current,
                whole_features_to_drop,
                output_dir=tmp_root / f"stage{stage}_features",
                repo_id=f"{final_repo_id}_stage{stage}",
            )
            if current.root != source_root:
                cleanup_roots.append(current.root)
            current = next_dataset

        if dims_to_drop:
            _log(f"Recortando dimensiones: {dims_to_drop}")
            stage += 1
            next_dataset = drop_feature_dimensions(
                current,
                dims_to_drop,
                output_dir=tmp_root / f"stage{stage}_dims",
                repo_id=f"{final_repo_id}_stage{stage}",
            )
            if current.root != source_root:
                cleanup_roots.append(current.root)
            current = next_dataset

        for root in cleanup_roots:
            shutil.rmtree(root, ignore_errors=True)

        if current.root == source_root:
            _log("Sin cambios de copia solicitados, copiando dataset base...")
            shutil.copytree(source_root, final_root)
        else:
            shutil.move(str(current.root), str(final_root))
        current = LeRobotDataset(repo_id=final_repo_id, root=final_root)

        if task_global or episode_tasks:
            _log("Aplicando cambios de Language Instruction...")
            modify_tasks(current, new_task=task_global, episode_tasks=episode_tasks)

        shutil.rmtree(tmp_root, ignore_errors=True)
        _log(f"Dataset curado en: {final_root}")
        return final_root

    except Exception:
        _log(f"Error durante el empaquetado. Directorios temporales conservados en: {tmp_root}")
        raise
