"""Pruebas unitarias para el cliente gRPC de detección de EPP."""

from unittest.mock import MagicMock, patch

import pytest

from src.grpc_service.client import (
    ClientDetection,
    ClientDetectResult,
    detect,
)


@pytest.fixture
def mock_response():
    """Crea una respuesta gRPC simulada con una detección."""
    response = MagicMock()
    response.inference_time_ms = 15.5
    response.model_version = "yolo11n-epp-v0.1"

    detection = MagicMock()
    detection.class_name = "helmet"
    detection.confidence = 0.95
    detection.box.x1 = 10.0
    detection.box.y1 = 20.0
    detection.box.x2 = 100.0
    detection.box.y2 = 200.0

    response.detections = [detection]
    return response


def test_detect_returns_client_detect_result(mock_response):
    """Debe devolver un ClientDetectResult."""
    with patch("src.grpc_service.client.grpc.insecure_channel"):
        channel_stub = MagicMock()
        channel_stub.Detect.return_value = mock_response

        with patch(
            "src.grpc_service.client.inference_pb2_grpc."
            "EPPInferenceServiceStub",
            return_value=channel_stub,
        ):
            result = detect(b"image")

    assert isinstance(result, ClientDetectResult)


def test_detect_uses_default_address(mock_response):
    """Debe usar localhost:50051 por defecto."""
    with patch("src.grpc_service.client.grpc.insecure_channel") as mock_channel:
        stub = MagicMock()
        stub.Detect.return_value = mock_response

        with patch(
            "src.grpc_service.client.inference_pb2_grpc."
            "EPPInferenceServiceStub",
            return_value=stub,
        ):
            detect(b"image")

    mock_channel.assert_called_once_with("localhost:50051")


def test_detect_uses_custom_address(mock_response):
    """Debe aceptar una dirección personalizada."""
    address = "192.168.1.10:6000"

    with patch("src.grpc_service.client.grpc.insecure_channel") as mock_channel:
        stub = MagicMock()
        stub.Detect.return_value = mock_response

        with patch(
            "src.grpc_service.client.inference_pb2_grpc."
            "EPPInferenceServiceStub",
            return_value=stub,
        ):
            detect(b"image", address=address)

    mock_channel.assert_called_once_with(address)


def test_detect_sends_image_bytes(mock_response):
    """Debe enviar correctamente los bytes de la imagen."""
    image_bytes = b"fake-image-data"

    with patch("src.grpc_service.client.grpc.insecure_channel"):
        stub = MagicMock()
        stub.Detect.return_value = mock_response

        with patch(
            "src.grpc_service.client.inference_pb2_grpc."
            "EPPInferenceServiceStub",
            return_value=stub,
        ):
            detect(image_bytes)

    request = stub.Detect.call_args.args[0]
    assert request.image_data == image_bytes


@pytest.mark.parametrize("threshold", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_detect_sends_conf_threshold(mock_response, threshold):
    """Debe enviar el umbral de confianza recibido."""
    with patch("src.grpc_service.client.grpc.insecure_channel"):
        stub = MagicMock()
        stub.Detect.return_value = mock_response

        with patch(
            "src.grpc_service.client.inference_pb2_grpc."
            "EPPInferenceServiceStub",
            return_value=stub,
        ):
            detect(b"image", conf_threshold=threshold)

    request = stub.Detect.call_args.args[0]
    assert request.conf_threshold == pytest.approx(threshold)


def test_detect_generates_request_id(mock_response):
    """Debe generar un identificador único para la solicitud."""
    with patch("src.grpc_service.client.grpc.insecure_channel"):
        stub = MagicMock()
        stub.Detect.return_value = mock_response

        with patch(
            "src.grpc_service.client.inference_pb2_grpc."
            "EPPInferenceServiceStub",
            return_value=stub,
        ):
            detect(b"image")

    request = stub.Detect.call_args.args[0]
    assert request.request_id
    assert len(request.request_id) == 36


def test_detect_uses_ten_second_timeout(mock_response):
    """Debe usar un timeout de 10 segundos."""
    with patch("src.grpc_service.client.grpc.insecure_channel"):
        stub = MagicMock()
        stub.Detect.return_value = mock_response

        with patch(
            "src.grpc_service.client.inference_pb2_grpc."
            "EPPInferenceServiceStub",
            return_value=stub,
        ):
            detect(b"image")

    assert stub.Detect.call_args.kwargs["timeout"] == 10


def test_detect_parses_detection(mock_response):
    """Debe convertir una detección gRPC a ClientDetection."""
    with patch("src.grpc_service.client.grpc.insecure_channel"):
        stub = MagicMock()
        stub.Detect.return_value = mock_response

        with patch(
            "src.grpc_service.client.inference_pb2_grpc."
            "EPPInferenceServiceStub",
            return_value=stub,
        ):
            result = detect(b"image")

    detection = result.detections[0]

    assert isinstance(detection, ClientDetection)
    assert detection.class_name == "helmet"
    assert detection.confidence == pytest.approx(0.95)
    assert detection.box == pytest.approx((10.0, 20.0, 100.0, 200.0))


def test_detect_returns_inference_metadata(mock_response):
    """Debe conservar tiempo de inferencia y versión del modelo."""
    with patch("src.grpc_service.client.grpc.insecure_channel"):
        stub = MagicMock()
        stub.Detect.return_value = mock_response

        with patch(
            "src.grpc_service.client.inference_pb2_grpc."
            "EPPInferenceServiceStub",
            return_value=stub,
        ):
            result = detect(b"image")

    assert result.inference_time_ms == pytest.approx(15.5)
    assert result.model_version == "yolo11n-epp-v0.1"


def test_detect_parses_multiple_detections(mock_response):
    """Debe convertir correctamente múltiples detecciones."""
    second = MagicMock()
    second.class_name = "Gloves"
    second.confidence = 0.88
    second.box.x1 = 30.0
    second.box.y1 = 40.0
    second.box.x2 = 130.0
    second.box.y2 = 140.0

    mock_response.detections.append(second)

    with patch("src.grpc_service.client.grpc.insecure_channel"):
        stub = MagicMock()
        stub.Detect.return_value = mock_response

        with patch(
            "src.grpc_service.client.inference_pb2_grpc."
            "EPPInferenceServiceStub",
            return_value=stub,
        ):
            result = detect(b"image")

    assert len(result.detections) == 2
    assert result.detections[1].class_name == "Gloves"
    assert result.detections[1].box == pytest.approx(
        (30.0, 40.0, 130.0, 140.0)
    )
