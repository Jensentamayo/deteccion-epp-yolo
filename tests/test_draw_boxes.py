"""Pruebas unitarias de src/visualizations/draw_boxes.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.predict_model import Detection
from src.visualizations.draw_boxes import _color_for_class, draw_detections


class TestColorForClass:
    @pytest.mark.parametrize(
        "class_name",
        ["no-helmet", "no-gloves", "NO-HELMET", "no-Gloves"],
    )
    def test_violation_classes_are_red(self, class_name):
        assert _color_for_class(class_name) == (220, 50, 50)

    @pytest.mark.parametrize("class_name", ["helmet", "gloves", "HELMET", "Gloves"])
    def test_compliant_classes_are_green(self, class_name):
        assert _color_for_class(class_name) == (46, 204, 113)

    def test_person_class_is_gray(self):
        assert _color_for_class("person") == (160, 160, 160)

    def test_unknown_class_defaults_to_green(self):
        assert _color_for_class("something-else") == (46, 204, 113)


class TestDrawDetections:
    def test_does_not_mutate_original_image(self, sample_image_array, sample_detections):
        original_copy = sample_image_array.copy()
        draw_detections(sample_image_array, sample_detections)
        assert np.array_equal(sample_image_array, original_copy)

    def test_output_has_same_shape(self, sample_image_array, sample_detections):
        annotated = draw_detections(sample_image_array, sample_detections)
        assert annotated.shape == sample_image_array.shape

    def test_empty_detections_returns_identical_image(self, sample_image_array):
        annotated = draw_detections(sample_image_array, [])
        assert np.array_equal(annotated, sample_image_array)

    def test_drawing_changes_pixels_when_detections_present(self, sample_image_array, sample_detections):
        annotated = draw_detections(sample_image_array, sample_detections)
        assert not np.array_equal(annotated, sample_image_array)

    @pytest.mark.parametrize("num_detections", [1, 2, 5, 10])
    def test_handles_multiple_detections(self, sample_image_array, num_detections):
        detections = [
            Detection(class_name="helmet", confidence=0.8, x1=i, y1=i, x2=i + 10, y2=i + 10)
            for i in range(num_detections)
        ]
        annotated = draw_detections(sample_image_array, detections)
        assert annotated.shape == sample_image_array.shape
