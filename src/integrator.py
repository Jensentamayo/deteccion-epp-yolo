"""Orquestador principal (Controlador): une preprocesamiento, modelo y tracking.

Esta es la única puerta de entrada que deben usar el servidor gRPC
y los clientes directos (tests, scripts). Devuelve una respuesta
procesada sin exponer detalles internos del modelo.
"""

from __future__ import annotations

from dataclasses import dataclass

from ultralytics import YOLO

from src.data.preprocess_img import decode_image_bytes, preprocess
from src.models.predict_model import Detection, InferenceResult, predict
from src.tracking.mlflow_tracker import MLflowTracker


@dataclass(frozen=True)
class EPPDetectionResponse:
    """Respuesta procesada devuelta al cliente."""

    inference_result: InferenceResult
    num_helmets: int
    num_no_helmets: int
    num_gloves: int
    num_no_gloves: int
    num_vests: int
    num_goggles: int
    # num_masks: int
    num_safety_shoes: int
    num_detections: int


def _count_class(result: InferenceResult, class_name: str) -> int:
    """Cuenta las detecciones pertenecientes a una clase específica."""
    return sum(
        1
        for detection in result.detections
        if detection.class_name.lower() == class_name.lower()
    )


def _filter_conflicting_classes(result: InferenceResult) -> InferenceResult:
    """Resuelve conflictos entre clases opuestas usando la confianza.

    Si existen detecciones de helmet y no_helmet, se conserva la detección
    con mayor confianza.

    Si existen detecciones de gloves y no_gloves, se conserva la detección
    con mayor confianza.

    Las demás detecciones se conservan sin cambios y se mantiene el orden
    original de las detecciones.
    """
    detections = result.detections

    helmet_detections = [
        detection
        for detection in detections
        if detection.class_name.lower() == "helmet"
    ]

    no_helmet_detections = [
        detection
        for detection in detections
        if detection.class_name.lower() == "no_helmet"
    ]

    gloves_detections = [
        detection
        for detection in detections
        if detection.class_name.lower() == "gloves"
    ]

    no_gloves_detections = [
        detection
        for detection in detections
        if detection.class_name.lower() == "no_gloves"
    ]

    selected_helmet = None

    if helmet_detections and no_helmet_detections:
        best_helmet = max(
            helmet_detections,
            key=lambda detection: detection.confidence,
        )
        best_no_helmet = max(
            no_helmet_detections,
            key=lambda detection: detection.confidence,
        )

        if best_helmet.confidence >= best_no_helmet.confidence:
            selected_helmet = best_helmet
        else:
            selected_helmet = best_no_helmet

    elif helmet_detections:
        selected_helmet = max(
            helmet_detections,
            key=lambda detection: detection.confidence,
        )

    elif no_helmet_detections:
        selected_helmet = max(
            no_helmet_detections,
            key=lambda detection: detection.confidence,
        )

    selected_gloves = None

    if gloves_detections and no_gloves_detections:
        best_gloves = max(
            gloves_detections,
            key=lambda detection: detection.confidence,
        )
        best_no_gloves = max(
            no_gloves_detections,
            key=lambda detection: detection.confidence,
        )

        if best_gloves.confidence >= best_no_gloves.confidence:
            selected_gloves = best_gloves
        else:
            selected_gloves = best_no_gloves

    elif gloves_detections:
        selected_gloves = max(
            gloves_detections,
            key=lambda detection: detection.confidence,
        )

    elif no_gloves_detections:
        selected_gloves = max(
            no_gloves_detections,
            key=lambda detection: detection.confidence,
        )

    selected_detections = []

    helmet_classes = {"helmet", "no_helmet"}
    glove_classes = {"gloves", "no_gloves"}

    helmet_added = False
    gloves_added = False

    for detection in detections:
        class_name = detection.class_name.lower()

        if class_name in helmet_classes:
            if not helmet_added and detection is selected_helmet:
                selected_detections.append(selected_helmet)
                helmet_added = True
            continue

        if class_name in glove_classes:
            if not gloves_added and detection is selected_gloves:
                selected_detections.append(selected_gloves)
                gloves_added = True
            continue

        selected_detections.append(detection)

    return InferenceResult(
        detections=selected_detections,
        inference_time_ms=result.inference_time_ms,
    )


def _scale_detections(
    result: InferenceResult,
    original_shape: tuple[int, int],
    processed_shape: tuple[int, int],
) -> InferenceResult:
    """Escala las cajas al tamaño de la imagen original.

    Args:
        result: Resultado de inferencia sobre la imagen preprocesada.
        original_shape: Alto y ancho de la imagen original.
        processed_shape: Alto y ancho de la imagen usada por YOLO.

    Returns:
        Resultado con las coordenadas de las cajas adaptadas
        al tamaño de la imagen original.
    """
    original_height, original_width = original_shape
    processed_height, processed_width = processed_shape

    scale_x = original_width / processed_width
    scale_y = original_height / processed_height

    scaled_detections = [
        Detection(
            class_name=detection.class_name,
            confidence=detection.confidence,
            x1=detection.x1 * scale_x,
            y1=detection.y1 * scale_y,
            x2=detection.x2 * scale_x,
            y2=detection.y2 * scale_y,
        )
        for detection in result.detections
    ]

    return InferenceResult(
        detections=scaled_detections,
        inference_time_ms=result.inference_time_ms,
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
    original_image = decode_image_bytes(image_bytes)
    processed_image = preprocess(image_bytes)

    result = predict(
        model,
        processed_image,
        conf_threshold=conf_threshold,
    )

    result = _filter_conflicting_classes(result)

    result = _scale_detections(
        result=result,
        original_shape=original_image.shape[:2],
        processed_shape=processed_image.shape[:2],
    )

    num_helmets = _count_class(result, "helmet")
    num_no_helmets = _count_class(result, "no_helmet")
    num_gloves = _count_class(result, "gloves")
    num_no_gloves = _count_class(result, "no_gloves")
    num_vests = _count_class(result, "vest")
    num_goggles = _count_class(result, "goggles")
    # num_masks = _count_class(result, "mask")
    num_safety_shoes = _count_class(result, "safety_shoe")
    num_detections = len(result.detections)

    response = EPPDetectionResponse(
        inference_result=result,
        num_helmets=num_helmets,
        num_no_helmets=num_no_helmets,
        num_gloves=num_gloves,
        num_no_gloves=num_no_gloves,
        num_vests=num_vests,
        num_goggles=num_goggles,
        # num_masks=num_masks,
        num_safety_shoes=num_safety_shoes,
        num_detections=num_detections,
    )

    if tracker is not None:
        tracker.log_inference_run(
            conf_threshold=conf_threshold,
            num_detections=num_detections,
            inference_time_ms=result.inference_time_ms,
            num_violations=num_no_helmets + num_no_gloves,
        )

    return response
