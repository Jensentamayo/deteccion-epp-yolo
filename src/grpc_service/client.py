"""Cliente gRPC delgado para consumir el servicio de inferencia de EPP."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

import grpc

from src.grpc_service import inference_pb2, inference_pb2_grpc

GRPC_HOST = os.environ.get("GRPC_HOST", "localhost")
GRPC_PORT = os.environ.get("GRPC_PORT", "50051")
DEFAULT_TIMEOUT = 10.0


@dataclass(frozen=True)
class ClientDetection:
    """Representa una detección recibida desde el servidor gRPC."""

    class_name: str
    confidence: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class ClientDetectResult:
    """Resultado de una solicitud de detección."""

    detections: list[ClientDetection]
    inference_time_ms: float
    model_version: str
    num_helmets: int
    num_no_helmets: int
    num_gloves: int
    num_no_gloves: int


class EPPGrpcClient:
    """Cliente gRPC reutilizable para inferencia en tiempo real."""

    def __init__(
        self,
        address: str | None = None,
        timeout: float = 2.0,
    ) -> None:
        """Inicializa un canal gRPC persistente.

        Args:
            address: Dirección host:puerto del servidor.
            timeout: Tiempo máximo de espera por inferencia.
        """
        self.target = address or f"{GRPC_HOST}:{GRPC_PORT}"
        self.timeout = timeout
        self.channel = grpc.insecure_channel(self.target)
        self.stub = inference_pb2_grpc.EPPInferenceServiceStub(self.channel)

    def detect(
        self,
        image_bytes: bytes,
        conf_threshold: float = 0.25,
    ) -> ClientDetectResult:
        """Envía una imagen al servidor gRPC."""
        request = inference_pb2.DetectRequest(
            image_data=image_bytes,
            conf_threshold=conf_threshold,
            request_id=str(uuid.uuid4()),
        )

        response = self.stub.Detect(
            request,
            timeout=self.timeout,
        )

        detections = [
            ClientDetection(
                class_name=d.class_name,
                confidence=d.confidence,
                box=(
                    d.box.x1,
                    d.box.y1,
                    d.box.x2,
                    d.box.y2,
                ),
            )
            for d in response.detections
        ]

        return ClientDetectResult(
            detections=detections,
            inference_time_ms=response.inference_time_ms,
            model_version=response.model_version,
            num_helmets=response.num_helmets,
            num_no_helmets=response.num_no_helmets,
            num_gloves=response.num_gloves,
            num_no_gloves=response.num_no_gloves,
        )

    def close(self) -> None:
        """Cierra el canal gRPC."""
        self.channel.close()

    def __del__(self) -> None:
        """Intenta cerrar el canal cuando el objeto se destruye."""
        try:
            self.close()
        except Exception:
            pass


def detect(
    image_bytes: bytes,
    conf_threshold: float = 0.25,
    address: str | None = None,
) -> ClientDetectResult:
    """Envía una imagen al servicio gRPC.

    Esta función mantiene compatibilidad con el cliente original.
    Para video en tiempo real se recomienda utilizar EPPGrpcClient.
    """
    client = EPPGrpcClient(
        address=address,
        timeout=DEFAULT_TIMEOUT,
    )

    try:
        return client.detect(
            image_bytes=image_bytes,
            conf_threshold=conf_threshold,
        )
    finally:
        client.close()