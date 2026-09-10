"""Pruebas unitarias del servicio gRPC (Detect / HealthCheck).

Requiere que los stubs `inference_pb2` / `inference_pb2_grpc` hayan sido
generados con `scripts/generate_grpc.sh`. Si no existen, estas pruebas
se omiten automáticamente (no rompen el resto de la suite).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip(
    "src.grpc_service.inference_pb2",
    reason="Genera los stubs con scripts/generate_grpc.sh antes de correr estas pruebas.",
)

from src.grpc_service import inference_pb2  # noqa: E402
from src.grpc_service.server import EPPInferenceServicer  # noqa: E402
from src.models.load_model import ModelLoadError  # noqa: E402


@pytest.fixture
def servicer(mock_yolo_model):
    with patch("src.grpc_service.server.MLflowTracker"):
        srv = EPPInferenceServicer(use_mlflow=False)
        srv._model = mock_yolo_model
        yield srv


@pytest.fixture
def grpc_context():
    return MagicMock()


class TestDetect:
    def test_returns_detections(self, servicer, grpc_context, small_image_bytes):
        request = inference_pb2.DetectRequest(
            image_data=small_image_bytes, conf_threshold=0.25, request_id="test-1"
        )
        response = servicer.Detect(request, grpc_context)
        assert len(response.detections) == 2

    def test_preserves_request_id(self, servicer, grpc_context, small_image_bytes):
        request = inference_pb2.DetectRequest(
            image_data=small_image_bytes, conf_threshold=0.25, request_id="abc-123"
        )
        response = servicer.Detect(request, grpc_context)
        assert response.request_id == "abc-123"

    def test_sets_internal_error_on_bad_image(self, servicer, grpc_context):
        request = inference_pb2.DetectRequest(image_data=b"", conf_threshold=0.25, request_id="x")
        servicer.Detect(request, grpc_context)
        grpc_context.set_code.assert_called_once()

    def test_default_conf_threshold_applied_when_zero(self, servicer, grpc_context, small_image_bytes):
        request = inference_pb2.DetectRequest(
            image_data=small_image_bytes, conf_threshold=0.0, request_id="y"
        )
        servicer.Detect(request, grpc_context)
        _, kwargs = servicer.model.predict.call_args
        assert kwargs["conf"] == 0.25

    def test_model_load_error_sets_unavailable(self, grpc_context, small_image_bytes):
        with patch("src.grpc_service.server.MLflowTracker"):
            srv = EPPInferenceServicer(model_path="/nonexistent.pt", use_mlflow=False)
        request = inference_pb2.DetectRequest(
            image_data=small_image_bytes, conf_threshold=0.25, request_id="z"
        )
        srv.Detect(request, grpc_context)
        grpc_context.set_code.assert_called_once()


class TestHealthCheck:
    def test_reports_model_loaded_true(self, servicer, grpc_context):
        response = servicer.HealthCheck(inference_pb2.HealthRequest(), grpc_context)
        assert response.model_loaded is True

    def test_reports_model_loaded_false_when_missing(self, grpc_context):
        with patch("src.grpc_service.server.MLflowTracker"):
            srv = EPPInferenceServicer(model_path="/nonexistent.pt", use_mlflow=False)
        response = srv.HealthCheck(inference_pb2.HealthRequest(), grpc_context)
        assert response.model_loaded is False
