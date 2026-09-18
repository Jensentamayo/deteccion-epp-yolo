"""Integración de preprocesamiento, inferencia y reglas de negocio de EPP."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.models.predict_model import Detection, InferenceResult, predict

# ÚNICAS CLASES PERMITIDAS EN LA APLICACIÓN.
ALLOWED_CLASSES = {
    "no_helmet",
    "no_gloves",
}


@dataclass(frozen=True)
class DetectionResponse:
    """Respuesta completa del pipeline de detección."""

    inference_result: InferenceResult
    num_helmets: int
    num_no_helmets: int
    num_gloves: int
    num_no_gloves: int
    num_vests: int = 0
    num_goggles: int = 0
    num_masks: int = 0
    num_safety_shoes: int = 0


def _filter_application_classes(
    detections: list[Detection],
) -> list[Detection]:
    """Conserva únicamente no_helmet y no_gloves."""

    return [
        detection
        for detection in detections
        if detection.class_name.strip().lower().replace("-", "_")
        in ALLOWED_CLASSES
    ]


def _scale_detections(
    detections: list[Detection],
    original_width: int,
    original_height: int,
    model_width: int,
    model_height: int,
) -> list[Detection]:
    """Escala las cajas desde las dimensiones del modelo a la imagen original."""

    scale_x = original_width / model_width
    scale_y = original_height / model_height

    return [
        Detection(
            class_name=detection.class_name,
            confidence=detection.confidence,
            x1=detection.x1 * scale_x,
            y1=detection.y1 * scale_y,
            x2=detection.x2 * scale_x,
            y2=detection.y2 * scale_y,
        )
        for detection in detections
    ]


def run_detection(
    model,
    image_bytes: bytes,
    conf_threshold: float = 0.25,
    tracker=None,
) -> DetectionResponse:
    """Ejecuta el pipeline completo de detección.

    Solo se conservan las clases:
    - no_helmet
    - no_gloves
    """

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("No se pudo decodificar la imagen recibida.")

    original_height, original_width = image.shape[:2]

    inference_result = predict(
        model=model,
        image=image,
        conf_threshold=conf_threshold,
    )

    filtered_detections = _filter_application_classes(
        inference_result.detections
    )

    model_height, model_width = image.shape[:2]

    scaled_detections = _scale_detections(
        detections=filtered_detections,
        original_width=original_width,
        original_height=original_height,
        model_width=model_width,
        model_height=model_height,
    )

    filtered_result = InferenceResult(
        detections=scaled_detections,
        inference_time_ms=inference_result.inference_time_ms,
    )

    num_no_helmets = sum(
        detection.class_name.strip().lower().replace("-", "_")
        == "no_helmet"
        for detection in scaled_detections
    )

    num_no_gloves = sum(
        detection.class_name.strip().lower().replace("-", "_")
        == "no_gloves"
        for detection in scaled_detections
    )

    if tracker is not None:
        tracker.log_inference(
            inference_time_ms=filtered_result.inference_time_ms,
            num_detections=len(scaled_detections),
            conf_threshold=conf_threshold,
            num_violations=len(scaled_detections),
        )

    return DetectionResponse(
        inference_result=filtered_result,
        num_helmets=0,
        num_no_helmets=num_no_helmets,
        num_gloves=0,
        num_no_gloves=num_no_gloves,
    )


# Compatibilidad con código y pruebas existentes.
EPPDetectionResponse = DetectionResponse
