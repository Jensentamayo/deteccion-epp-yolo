"""Pruebas unitarias de src/models/predict_model.py (con modelo YOLO mockeado)."""

from __future__ import annotations

import pytest

from src.models.predict_model import Detection, InferenceResult, predict


class TestPredict:
    def test_returns_inference_result(self, mock_yolo_model, sample_image_array):
        result = predict(mock_yolo_model, sample_image_array)
        assert isinstance(result, InferenceResult)

    def test_returns_expected_number_of_detections(self, mock_yolo_model, sample_image_array):
        result = predict(mock_yolo_model, sample_image_array)
        assert len(result.detections) == 2

    def test_detection_fields_are_populated(self, mock_yolo_model, sample_image_array):
        result = predict(mock_yolo_model, sample_image_array)
        first = result.detections[0]
        assert isinstance(first, Detection)
        assert first.class_name == "helmet"
        assert 0.0 <= first.confidence <= 1.0

    def test_inference_time_is_non_negative(self, mock_yolo_model, sample_image_array):
        result = predict(mock_yolo_model, sample_image_array)
        assert result.inference_time_ms >= 0

    def test_calls_model_predict_with_conf_threshold(self, mock_yolo_model, sample_image_array):
        predict(mock_yolo_model, sample_image_array, conf_threshold=0.5)
        _, kwargs = mock_yolo_model.predict.call_args
        assert kwargs["conf"] == 0.5

    @pytest.mark.parametrize("conf", [0.1, 0.25, 0.5, 0.75, 0.9])
    def test_various_confidence_thresholds_are_passed_through(
        self, mock_yolo_model, sample_image_array, conf
    ):
        predict(mock_yolo_model, sample_image_array, conf_threshold=conf)
        _, kwargs = mock_yolo_model.predict.call_args
        assert kwargs["conf"] == conf

    def test_empty_results_returns_empty_detections(self, sample_image_array):
        from unittest.mock import MagicMock

        model = MagicMock()
        model.predict.return_value = []
        result = predict(model, sample_image_array)
        assert result.detections == []
