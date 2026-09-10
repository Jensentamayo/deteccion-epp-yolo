"""Pruebas unitarias de src/integrator.py (Controlador)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.integrator import EPPDetectionResponse, run_detection


class TestRunDetection:
    def test_returns_epp_detection_response(
        self,
        mock_yolo_model,
        small_image_bytes,
    ):
        response = run_detection(mock_yolo_model, small_image_bytes)

        assert isinstance(response, EPPDetectionResponse)

    def test_counts_helmet_correctly(
        self,
        mock_yolo_model,
        small_image_bytes,
    ):
        response = run_detection(mock_yolo_model, small_image_bytes)

        assert response.num_helmets == 1

    def test_counts_gloves_correctly(
        self,
        mock_yolo_model,
        small_image_bytes,
    ):
        response = run_detection(mock_yolo_model, small_image_bytes)

        assert response.num_gloves == 0

    def test_counts_vests_correctly(
        self,
        mock_yolo_model,
        small_image_bytes,
    ):
        response = run_detection(mock_yolo_model, small_image_bytes)

        assert response.num_vests == 0

    def test_counts_goggles_correctly(
        self,
        mock_yolo_model,
        small_image_bytes,
    ):
        response = run_detection(mock_yolo_model, small_image_bytes)

        assert response.num_goggles == 0

    def test_counts_masks_correctly(
        self,
        mock_yolo_model,
        small_image_bytes,
    ):
        response = run_detection(mock_yolo_model, small_image_bytes)

        assert response.num_masks == 0

    def test_counts_safety_shoes_correctly(
        self,
        mock_yolo_model,
        small_image_bytes,
    ):
        response = run_detection(mock_yolo_model, small_image_bytes)

        assert response.num_safety_shoes == 0

    def test_counts_total_detections(
        self,
        mock_yolo_model,
        small_image_bytes,
    ):
        response = run_detection(mock_yolo_model, small_image_bytes)

        assert response.num_detections == 2

    def test_propagates_conf_threshold_to_model(
        self,
        mock_yolo_model,
        small_image_bytes,
    ):
        run_detection(
            mock_yolo_model,
            small_image_bytes,
            conf_threshold=0.6,
        )

        _, kwargs = mock_yolo_model.predict.call_args

        assert kwargs["conf"] == 0.6

    def test_calls_tracker_when_provided(
        self,
        mock_yolo_model,
        small_image_bytes,
    ):
        tracker = MagicMock()

        run_detection(
            mock_yolo_model,
            small_image_bytes,
            tracker=tracker,
        )

        tracker.log_inference_run.assert_called_once()

    def test_does_not_call_tracker_when_none(
        self,
        mock_yolo_model,
        small_image_bytes,
    ):
        response = run_detection(
            mock_yolo_model,
            small_image_bytes,
            tracker=None,
        )

        assert response is not None

    def test_tracker_receives_expected_kwargs(
        self,
        mock_yolo_model,
        small_image_bytes,
    ):
        tracker = MagicMock()

        run_detection(
            mock_yolo_model,
            small_image_bytes,
            conf_threshold=0.4,
            tracker=tracker,
        )

        _, kwargs = tracker.log_inference_run.call_args

        assert kwargs["conf_threshold"] == 0.4
        assert kwargs["num_detections"] == 2
        assert kwargs["num_violations"] == 0

    @pytest.mark.parametrize(
        "conf",
        [0.1, 0.3, 0.5, 0.7],
    )
    def test_multiple_confidence_values_do_not_error(
        self,
        mock_yolo_model,
        small_image_bytes,
        conf,
    ):
        response = run_detection(
            mock_yolo_model,
            small_image_bytes,
            conf_threshold=conf,
        )

        assert response.inference_result is not None
