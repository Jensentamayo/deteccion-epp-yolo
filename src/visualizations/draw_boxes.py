"""Funciones para dibujar incumplimientos de EPP sobre imágenes."""

from __future__ import annotations

import cv2
import numpy as np

from src.models.predict_model import Detection

ALLOWED_CLASSES = {
    "no_helmet",
    "no_gloves",
}

RED_COLOR = (220, 50, 50)


def _normalize_class_name(class_name: str) -> str:
    """Normaliza el nombre de una clase para comparaciones."""
    return class_name.strip().lower().replace("-", "_")


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
) -> np.ndarray:
    """Dibuja únicamente detecciones de no_helmet y no_gloves.

    Las detecciones de incumplimiento se muestran en rojo.
    Cualquier otra clase recibida se ignora.
    """

    annotated = image.copy()

    for detection in detections:
        normalized = _normalize_class_name(detection.class_name)

        if normalized not in ALLOWED_CLASSES:
            continue

        x1 = int(detection.x1)
        y1 = int(detection.y1)
        x2 = int(detection.x2)
        y2 = int(detection.y2)

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            RED_COLOR,
            2,
        )

        label = f"{normalized} {detection.confidence:.2f}"

        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            RED_COLOR,
            1,
            cv2.LINE_AA,
        )

    return annotated
