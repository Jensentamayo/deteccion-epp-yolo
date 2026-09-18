"""Pruebas unitarias para el dibujo de detecciones de EPP."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.predict_model import Detection
from src.visualizations.draw_boxes import (
    _normalize_class_name,
    draw_detections,
)


class TestNormalizeClassName:
    """Pruebas de normalización de nombres de clases."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("no_helmet", "no_helmet"),
            ("NO_HELMET", "no_helmet"),
            ("no-helmet", "no_helmet"),
        ],
    )
    def test_normalizes_class_name(self, value, expected):
        assert _normalize_class_name(value) == expected


class TestDrawDetections:
    """Pruebas del dibujo de cajas."""

    def test_returns_copy_of_image(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = draw_detections(image, [])

        assert result is not image
        assert result.shape == image.shape

    def test_draws_no_helmet_detection(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        detection = Detection(
            class_name="no_helmet",
            confidence=0.90,
            x1=10,
            y1=10,
            x2=50,
            y2=50,
        )

        result = draw_detections(image, [detection])

        assert np.any(result != image)

    def test_draws_no_gloves_detection(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        detection = Detection(
            class_name="no_gloves",
            confidence=0.85,
            x1=20,
            y1=20,
            x2=60,
            y2=60,
        )

        result = draw_detections(image, [detection])

        assert np.any(result != image)

    def test_normalizes_hyphenated_class(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        detection = Detection(
            class_name="no-helmet",
            confidence=0.90,
            x1=10,
            y1=10,
            x2=50,
            y2=50,
        )

        result = draw_detections(image, [detection])

        assert np.any(result != image)

    def test_ignores_unknown_class(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        detection = Detection(
            class_name="helmet",
            confidence=0.95,
            x1=10,
            y1=10,
            x2=50,
            y2=50,
        )

        result = draw_detections(image, [detection])

        assert np.array_equal(result, image)

    def test_ignores_gloves_class(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        detection = Detection(
            class_name="gloves",
            confidence=0.95,
            x1=10,
            y1=10,
            x2=50,
            y2=50,
        )

        result = draw_detections(image, [detection])

        assert np.array_equal(result, image)

    def test_draws_multiple_allowed_detections(self):
        image = np.zeros((120, 120, 3), dtype=np.uint8)

        detections = [
            Detection(
                class_name="no_helmet",
                confidence=0.90,
                x1=10,
                y1=10,
                x2=50,
                y2=50,
            ),
            Detection(
                class_name="no_gloves",
                confidence=0.80,
                x1=60,
                y1=60,
                x2=100,
                y2=100,
            ),
        ]

        result = draw_detections(image, detections)

        assert np.any(result != image)

    def test_accepts_uppercase_class_name(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        detection = Detection(
            class_name="NO_GLOVES",
            confidence=0.88,
            x1=10,
            y1=10,
            x2=50,
            y2=50,
        )

        result = draw_detections(image, [detection])

        assert np.any(result != image)

    def test_accepts_class_name_with_spaces(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        detection = Detection(
            class_name=" no_helmet ",
            confidence=0.88,
            x1=10,
            y1=10,
            x2=50,
            y2=50,
        )

        result = draw_detections(image, [detection])

        assert np.any(result != image)
