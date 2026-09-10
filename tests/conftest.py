"""Fixtures compartidas para la suite de pruebas."""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from src.models.predict_model import Detection, InferenceResult


def _make_image_bytes(width: int, height: int, color: tuple[int, int, int] = (120, 150, 180)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def small_image_bytes() -> bytes:
    return _make_image_bytes(64, 64)


@pytest.fixture
def large_image_bytes() -> bytes:
    return _make_image_bytes(2000, 1500)


@pytest.fixture
def image_bytes_factory():
    return _make_image_bytes


@pytest.fixture
def sample_image_array() -> np.ndarray:
    return np.zeros((100, 120, 3), dtype=np.uint8)


@pytest.fixture
def sample_detections() -> list[Detection]:
    return [
        Detection(class_name="helmet", confidence=0.91, x1=10, y1=10, x2=50, y2=50),
        Detection(class_name="no-gloves", confidence=0.77, x1=60, y1=20, x2=100, y2=80),
    ]


@pytest.fixture
def sample_inference_result(sample_detections) -> InferenceResult:
    return InferenceResult(detections=sample_detections, inference_time_ms=12.3)


@pytest.fixture
def mock_yolo_model(sample_detections):
    """Simula un modelo `ultralytics.YOLO` sin cargar pesos reales."""
    model = MagicMock()
    model.names = {0: "helmet", 1: "no-helmet", 2: "gloves", 3: "no-gloves", 4: "person"}
    model.model_name = "yolo11n-epp-mock"

    fake_box = MagicMock()

    def _predict(source, conf, verbose):  # noqa: ARG001
        fake_result = MagicMock()
        fake_result.names = model.names
        boxes = []
        for det in sample_detections:
            box = MagicMock()
            box.cls = [list(model.names.values()).index(det.class_name)]
            box.conf = [det.confidence]
            box.xyxy = [[det.x1, det.y1, det.x2, det.y2]]
            boxes.append(box)
        fake_result.boxes = boxes
        return [fake_result]

    model.predict.side_effect = _predict
    return model
