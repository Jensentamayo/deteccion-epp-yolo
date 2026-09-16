"""Funciones para dibujar las detecciones de EPP sobre imágenes."""

from __future__ import annotations

import cv2
import numpy as np

from src.models.predict_model import Detection


def _normalize_class_name(class_name: str) -> str:
    """Normaliza el nombre de una clase para comparaciones."""
    return class_name.strip().lower().replace("-", "_")


def _color_for_class(class_name: str) -> tuple[int, int, int]:
    """Devuelve el color BGR correspondiente a una clase."""

    normalized = _normalize_class_name(class_name)

    if normalized in {"no_helmet", "no_gloves"}:
        return (220, 50, 50)

    if normalized in {"helmet", "gloves"}:
        return (46, 204, 113)

    if normalized == "person":
        return (160, 160, 160)

    return (46, 204, 113)


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
) -> np.ndarray:
    """Dibuja las detecciones sobre una copia de la imagen."""

    if not detections:
        return image.copy()

    annotated = image.copy()

    for detection in detections:
        color = _color_for_class(detection.class_name)

        x1 = int(detection.x1)
        y1 = int(detection.y1)
        x2 = int(detection.x2)
        y2 = int(detection.y2)

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        label = f"{detection.class_name} {detection.confidence:.2f}"

        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return annotated
