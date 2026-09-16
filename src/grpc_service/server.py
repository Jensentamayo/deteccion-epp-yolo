"""Servidor gRPC que expone el servicio de inferencia de EPP."""

from __future__ import annotations

import logging
import os
import threading
from concurrent import futures

import grpc

from src.grpc_service import inference_pb2, inference_pb2_grpc
from src.integrator import run_detection
from src.models.load_model import (
    DEFAULT_MODEL_PATH,
    ModelLoadError,
    load_model,
)
from src.tracking.mlflow_tracker import MLflowTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("epp_grpc_server")

GRPC_PORT = os.environ.get("GRPC_PORT", "50051")

MLFLOW_LOG_EVERY_N = int(
    os.environ.get("MLFLOW_LOG_EVERY_N", "30")
)


class EPPInferenceServicer(
    inference_pb2_grpc.EPPInferenceServiceServicer
):
    """Implementación del contrato definido en inference.proto."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        use_mlflow: bool = True,
    ) -> None:
        """Inicializa el servicio de inferencia."""
        self.model_path = model_path
        self._model = None

        self._tracker = (
            MLflowTracker()
            if use_mlflow
            else None
        )

        self._request_count = 0
        self._counter_lock = threading.Lock()

    @property
    def model(self):
        """Carga el modelo de manera diferida y reutilizable."""
        if self._model is None:
            self._model = load_model(self.model_path)

        return self._model

    def _should_log_mlflow(self) -> bool:
        """Determina si esta solicitud debe registrarse en MLflow."""
        with self._counter_lock:
            self._request_count += 1
            request_number = self._request_count

        return (
            self._tracker is not None
            and request_number % MLFLOW_LOG_EVERY_N == 0
        )

    def Detect(self, request, context):  # noqa: N802
        """Procesa una solicitud de detección."""
        try:
            tracker = (
                self._tracker
                if self._should_log_mlflow()
                else None
            )

            response = run_detection(
                model=self.model,
                image_bytes=request.image_data,
                conf_threshold=request.conf_threshold or 0.25,
                tracker=tracker,
            )

        except ModelLoadError as exc:
            context.set_code(
                grpc.StatusCode.UNAVAILABLE
            )
            context.set_details(str(exc))
            return inference_pb2.DetectResponse()

        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Error durante la inferencia"
            )
            context.set_code(
                grpc.StatusCode.INTERNAL
            )
            context.set_details(
                f"Error interno durante la inferencia: {exc}"
            )
            return inference_pb2.DetectResponse()

        detections_proto = [
            inference_pb2.Detection(
                class_name=det.class_name,
                confidence=det.confidence,
                box=inference_pb2.BoundingBox(
                    x1=det.x1,
                    y1=det.y1,
                    x2=det.x2,
                    y2=det.y2,
                ),
            )
            for det in response.inference_result.detections
        ]

        return inference_pb2.DetectResponse(
            detections=detections_proto,
            inference_time_ms=(
                response.inference_result.inference_time_ms
            ),
            model_version="yolo11n-epp-v0.1",
            request_id=request.request_id,
            num_helmets=response.num_helmets,
            num_no_helmets=response.num_no_helmets,
            num_gloves=response.num_gloves,
            num_no_gloves=response.num_no_gloves,
        )

    def HealthCheck(self, request, context):  # noqa: N802
        """Comprueba el estado del modelo."""
        try:
            model = self.model

            return inference_pb2.HealthResponse(
                model_loaded=True,
                model_name=str(model.model_name),
            )

        except ModelLoadError:
            return inference_pb2.HealthResponse(
                model_loaded=False,
                model_name="",
            )


def serve() -> None:
    """Inicia el servidor gRPC."""
    servicer = EPPInferenceServicer()

    logger.info("Cargando modelo YOLO11n...")

    try:
        servicer.model
    except ModelLoadError:
        logger.exception(
            "No se pudo cargar el modelo."
        )
        raise

    logger.info(
        "Modelo YOLO11n cargado correctamente."
    )

    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=8
        )
    )

    inference_pb2_grpc.add_EPPInferenceServiceServicer_to_server(
        servicer,
        server,
    )

    server.add_insecure_port(
        f"[::]:{GRPC_PORT}"
    )

    server.start()

    logger.info(
        "Servidor gRPC de EPP escuchando "
        "en el puerto %s",
        GRPC_PORT,
    )

    logger.info(
        "MLflow registra 1 de cada %s frames.",
        MLFLOW_LOG_EVERY_N,
    )

    server.wait_for_termination()


if __name__ == "__main__":
    serve()