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
        response = run_detection(
            mock_yolo_model,
            small_image_bytes,
        )

        assert isinstance(response, EPPDetectionResponse)

    def test_counts_no_helmet_correctly(
        self,
        mock_yolo_model,
        small_image_bytes,
    ):
        response = run_detection(
            mock_yolo_model,
            small_image_bytes,
        )

        assert response.num_no_helmets == 0

    def test_counts_no_gloves_correctly(
        self,
        mock_yolo_model,
        small_image_bytes,
    ):
        response = run_detection(
            mock_yolo_model,
            small_image_bytes,
        )

        assert response.num_no_gloves == 1

    def test_compliant_helmet_count_is_zero(
        self,
        mock_yolo_model,
        small_image_bytes,
    ):
        response = run_detection(
            mock_yolo_model,
            small_image_bytes,
        )

        assert response.num_helmets == 0

    def test_compliant_gloves_count_is_zero(
        self,
        mock_yolo_model,
        small_image_bytes,
    ):
        response = run_detection(
            mock_yolo_model,
            small_image_bytes,
        )

        assert response.num_gloves == 0

    def test_other_legacy_counts_are_zero(
        self,
        mock_yolo_model,
        small_image_bytes,
    ):
        response = run_detection(
            mock_yolo_model,
            small_image_bytes,
        )

        assert response.num_vests == 0
        assert response.num_goggles == 0
        assert response.num_masks == 0
        assert response.num_safety_shoes == 0

    def test_counts_total_detections_from_inference_result(
        self,
        mock_yolo_model,
        small_image_bytes,
    ):
        response = run_detection(
            mock_yolo_model,
            small_image_bytes,
        )

        assert len(response.inference_result.detections) == 1

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

        tracker.log_inference.assert_called_once()

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

        _, kwargs = tracker.log_inference.call_args

        assert kwargs["inference_time_ms"] >= 0
        assert kwargs["num_detections"] == 1

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
