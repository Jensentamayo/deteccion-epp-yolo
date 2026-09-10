"""Orquestador principal (Controlador): une preprocesamiento, modelo y tracking.

Esta es la única puerta de entrada que deben usar el servidor gRPC
y los clientes directos (tests, scripts). Devuelve una respuesta
procesada sin exponer detalles internos del modelo.
"""

from __future__ import annotations

from dataclasses import dataclass

from ultralytics import YOLO

from src.data.preprocess_img import preprocess
from src.models.predict_model import InferenceResult, predict
from src.tracking.mlflow_tracker import MLflowTracker


@dataclass(frozen=True)
class EPPDetectionResponse:
    """Respuesta procesada devuelta al cliente."""

    inference_result: InferenceResult
    num_helmets: int
    num_gloves: int
    num_vests: int
    num_goggles: int
    num_masks: int
    num_safety_shoes: int
    num_detections: int


def _count_class(result: InferenceResult, class_name: str) -> int:
    """Cuenta las detecciones pertenecientes a una clase específica."""
    return sum(
        1
        for detection in result.detections
        if detection.class_name.lower() == class_name.lower()
    )


def run_detection(
    model: YOLO,
    image_bytes: bytes,
    conf_threshold: float = 0.25,
    tracker: MLflowTracker | None = None,
) -> EPPDetectionResponse:
    """Ejecuta el pipeline completo de detección de EPP.

    Args:
        model: Modelo YOLO ya cargado.
        image_bytes: Imagen recibida como bytes.
        conf_threshold: Confianza mínima para conservar una detección.
        tracker: Tracker opcional de MLflow.

    Returns:
        EPPDetectionResponse con las detecciones y contadores por clase.
    """
    image = preprocess(image_bytes)
    result = predict(
        model,
        image,
        conf_threshold=conf_threshold,
    )

    num_helmets = _count_class(result, "helmet")
    num_gloves = _count_class(result, "gloves")
    num_vests = _count_class(result, "vest")
    num_goggles = _count_class(result, "goggles")
    num_masks = _count_class(result, "mask")
    num_safety_shoes = _count_class(result, "safety_shoe")
    num_detections = len(result.detections)

    response = EPPDetectionResponse(
        inference_result=result,
        num_helmets=num_helmets,
        num_gloves=num_gloves,
        num_vests=num_vests,
        num_goggles=num_goggles,
        num_masks=num_masks,
        num_safety_shoes=num_safety_shoes,
        num_detections=num_detections,
    )

    if tracker is not None:
        tracker.log_inference_run(
            conf_threshold=conf_threshold,
            num_detections=num_detections,
            inference_time_ms=result.inference_time_ms,
            num_violations=0,
        )

    return response
