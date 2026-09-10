"""Renderizado de las detecciones (bounding boxes) sobre la imagen original."""

from __future__ import annotations

import cv2
import numpy as np

from src.models.predict_model import Detection

# Verde para EPP presente, rojo para ausencia de EPP, gris por defecto.
_COLOR_BY_KEYWORD = {
    "no-": (220, 50, 50),
    "person": (160, 160, 160),
}
_DEFAULT_COLOR = (46, 204, 113)


def _color_for_class(class_name: str) -> tuple[int, int, int]:
    lowered = class_name.lower()
    for keyword, color in _COLOR_BY_KEYWORD.items():
        if lowered.startswith(keyword):
            return color
    return _DEFAULT_COLOR


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Dibuja cajas + etiquetas sobre una copia de la imagen de entrada.

    Args:
        image: Imagen RGB original (H, W, 3).
        detections: Lista de detecciones a dibujar.

    Returns:
        Copia de la imagen con las anotaciones dibujadas.
    """
    annotated = image.copy()
    for det in detections:
        color = _color_for_class(det.class_name)
        pt1 = (int(det.x1), int(det.y1))
        pt2 = (int(det.x2), int(det.y2))
        cv2.rectangle(annotated, pt1, pt2, color, thickness=2)

        label = f"{det.class_name} {det.confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_bg_pt2 = (pt1[0] + text_w + 4, pt1[1] - text_h - 8)
        cv2.rectangle(annotated, pt1, label_bg_pt2, color, thickness=-1)
        cv2.putText(
            annotated,
            label,
            (pt1[0] + 2, pt1[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated
