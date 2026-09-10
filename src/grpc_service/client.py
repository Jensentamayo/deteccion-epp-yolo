"""Cliente gRPC delgado para consumir el servicio de inferencia de EPP."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

import grpc

from src.grpc_service import inference_pb2, inference_pb2_grpc

GRPC_HOST = os.environ.get("GRPC_HOST", "localhost")
GRPC_PORT = os.environ.get("GRPC_PORT", "50051")


@dataclass(frozen=True)
class ClientDetection:
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class ClientDetectResult:
    detections: list[ClientDetection]
    inference_time_ms: float
    model_version: str


def detect(image_bytes: bytes, conf_threshold: float = 0.25, address: str | None = None) -> ClientDetectResult:
    """Envía una imagen al servicio gRPC y devuelve las detecciones parseadas.

    Args:
        image_bytes: Contenido binario de la imagen a analizar.
        conf_threshold: Umbral mínimo de confianza.
        address: Dirección "host:puerto" del servidor gRPC (por defecto usa
            variables de entorno GRPC_HOST / GRPC_PORT).

    Returns:
        ClientDetectResult con las detecciones ya convertidas a tipos simples.
    """
    target = address or f"{GRPC_HOST}:{GRPC_PORT}"
    with grpc.insecure_channel(target) as channel:
        stub = inference_pb2_grpc.EPPInferenceServiceStub(channel)
        request = inference_pb2.DetectRequest(
            image_data=image_bytes,
            conf_threshold=conf_threshold,
            request_id=str(uuid.uuid4()),
        )
        response = stub.Detect(request, timeout=10)

    detections = [
        ClientDetection(
            class_name=d.class_name,
            confidence=d.confidence,
            box=(d.box.x1, d.box.y1, d.box.x2, d.box.y2),
        )
        for d in response.detections
    ]
    return ClientDetectResult(
        detections=detections,
        inference_time_ms=response.inference_time_ms,
        model_version=response.model_version,
    )
