"""Servidor gRPC que expone el servicio de inferencia de EPP.

Nota: `inference_pb2.py` e `inference_pb2_grpc.py` se generan localmente
a partir de `inference.proto` (ver scripts/generate_grpc.sh) y no se
versionan en el repositorio.
"""

from __future__ import annotations

import logging
import os
from concurrent import futures

import grpc

from src.grpc_service import inference_pb2, inference_pb2_grpc
from src.integrator import run_detection
from src.models.load_model import DEFAULT_MODEL_PATH, ModelLoadError, load_model
from src.tracking.mlflow_tracker import MLflowTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("epp_grpc_server")

GRPC_PORT = os.environ.get("GRPC_PORT", "50051")


class EPPInferenceServicer(inference_pb2_grpc.EPPInferenceServiceServicer):
    """Implementación del contrato definido en inference.proto."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, use_mlflow: bool = True) -> None:
        self.model_path = model_path
        self._model = None
        self._tracker = MLflowTracker() if use_mlflow else None

    @property
    def model(self):
        if self._model is None:
            self._model = load_model(self.model_path)
        return self._model

    def Detect(self, request, context):  # noqa: N802 (nombre impuesto por gRPC)
        try:
            response = run_detection(
                model=self.model,
                image_bytes=request.image_data,
                conf_threshold=request.conf_threshold or 0.25,
                tracker=self._tracker,
            )
        except ModelLoadError as exc:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details(str(exc))
            return inference_pb2.DetectResponse()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error durante la inferencia")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error interno durante la inferencia: {exc}")
            return inference_pb2.DetectResponse()

        detections_proto = [
            inference_pb2.Detection(
                class_name=det.class_name,
                confidence=det.confidence,
                box=inference_pb2.BoundingBox(x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2),
            )
            for det in response.inference_result.detections
        ]

        return inference_pb2.DetectResponse(
            detections=detections_proto,
            inference_time_ms=response.inference_result.inference_time_ms,
            model_version="yolo11n-epp-v0.1",
            request_id=request.request_id,
        )

    def HealthCheck(self, request, context):  # noqa: N802
        try:
            model = self.model
            return inference_pb2.HealthResponse(model_loaded=True, model_name=str(model.model_name))
        except ModelLoadError:
            return inference_pb2.HealthResponse(model_loaded=False, model_name="")


def serve() -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    inference_pb2_grpc.add_EPPInferenceServiceServicer_to_server(EPPInferenceServicer(), server)
    server.add_insecure_port(f"[::]:{GRPC_PORT}")
    server.start()
    logger.info("Servidor gRPC de EPP escuchando en el puerto %s", GRPC_PORT)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
