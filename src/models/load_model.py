"""Carga y cacheo del modelo YOLO11n para detección de incumplimiento de EPP.

Responsabilidad (capa Modelo): cargar el modelo entrenado una sola vez
por proceso y reutilizarlo en cada inferencia.

El modelo final utiliza únicamente dos clases:
0 - no_helmet
1 - no_gloves
"""

from __future__ import annotations

import os
from functools import lru_cache

from ultralytics import YOLO

DEFAULT_MODEL_PATH = os.environ.get(
    "EPP_MODEL_PATH",
    "models/trained/epp_no_compliance_yolo11n_final.pt",
)

DEFAULT_CLASS_NAMES: tuple[str, ...] = (
    "no_helmet",
    "no_gloves",
)


class ModelLoadError(RuntimeError):
    """Se lanza cuando el modelo no puede cargarse correctamente."""


@lru_cache(maxsize=1)
def load_model(model_path: str = DEFAULT_MODEL_PATH) -> YOLO:
    """Carga el modelo YOLO desde disco y lo mantiene en caché.

    Args:
        model_path: Ruta al archivo de pesos `.pt`.

    Returns:
        Instancia de `ultralytics.YOLO` lista para realizar inferencias.

    Raises:
        ModelLoadError: si el archivo no existe o no puede cargarse.
    """
    if not os.path.exists(model_path):
        raise ModelLoadError(
            f"No se encontró el archivo de pesos del modelo en "
            f"'{model_path}'. Verifica EPP_MODEL_PATH o coloca "
            "el modelo en esa ruta."
        )

    try:
        return YOLO(model_path)
    except Exception as exc:  # pragma: no cover
        raise ModelLoadError(
            f"Fallo al cargar el modelo YOLO desde "
            f"'{model_path}': {exc}"
        ) from exc


def get_class_names(model: YOLO | None = None) -> tuple[str, ...]:
    """Obtiene los nombres de las clases del modelo.

    Los nombres almacenados dentro del modelo tienen prioridad sobre
    los nombres predeterminados.
    """
    if model is not None and getattr(model, "names", None):
        names = model.names

        if isinstance(names, dict):
            return tuple(names[index] for index in sorted(names))

        if isinstance(names, (list, tuple)):
            return tuple(names)

    return DEFAULT_CLASS_NAMES
