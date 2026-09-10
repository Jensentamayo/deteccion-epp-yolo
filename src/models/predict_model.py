"""Ejecución de inferencia del modelo YOLO11n sobre una imagen preprocesada.

Responsabilidad (capa Modelo): dado un arreglo numpy (imagen) y un umbral
de confianza, devuelve una lista estructurada de detecciones.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    """Representa una única detección de EPP sobre la imagen."""

    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class InferenceResult:
    """Resultado completo de una corrida de inferencia."""

    detections: list[Detection]
    inference_time_ms: float


def predict(
    model: YOLO,
    image: np.ndarray,
    conf_threshold: float = 0.25,
) -> InferenceResult:
    """Corre inferencia YOLO sobre una imagen ya preprocesada.

    Args:
        model: Instancia de YOLO ya cargada (ver `load_model`).
        image: Imagen en formato numpy array (H, W, 3), RGB, valores 0-255.
        conf_threshold: Confianza mínima para conservar una detección.

    Returns:
        InferenceResult con la lista de detecciones y el tiempo de inferencia.
    """
    start = time.perf_counter()
    results = model.predict(source=image, conf=conf_threshold, verbose=False)
    elapsed_ms = (time.perf_counter() - start) * 1000

    detections: list[Detection] = []
    if results:
        result = results[0]
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            detections.append(
                Detection(
                    class_name=names.get(cls_id, str(cls_id)),
                    confidence=conf,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

    return InferenceResult(detections=detections, inference_time_ms=elapsed_ms)
