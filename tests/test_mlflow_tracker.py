"""Pruebas unitarias de src/tracking/mlflow_tracker.py (mlflow mockeado)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.tracking.mlflow_tracker import MLflowTracker


@pytest.fixture
def mocked_mlflow():
    with patch("src.tracking.mlflow_tracker.mlflow") as mock_mlflow:
        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        yield mock_mlflow


class TestMLflowTrackerInit:
    def test_sets_tracking_uri(self, mocked_mlflow):
        MLflowTracker(tracking_uri="http://example:5000")
        mocked_mlflow.set_tracking_uri.assert_called_once_with("http://example:5000")

    def test_sets_experiment(self, mocked_mlflow):
        MLflowTracker(experiment_name="mi-experimento")
        mocked_mlflow.set_experiment.assert_called_once_with("mi-experimento")


class TestLogInferenceRun:
    def test_starts_a_run(self, mocked_mlflow):
        tracker = MLflowTracker()
        tracker.log_inference_run(
            conf_threshold=0.25, num_detections=3, inference_time_ms=15.0, num_violations=1
        )
        mocked_mlflow.start_run.assert_called_once()

    def test_logs_conf_threshold_param(self, mocked_mlflow):
        tracker = MLflowTracker()
        tracker.log_inference_run(
            conf_threshold=0.4, num_detections=2, inference_time_ms=10.0, num_violations=0
        )
        mocked_mlflow.log_param.assert_any_call("conf_threshold", 0.4)

    def test_logs_num_detections_metric(self, mocked_mlflow):
        tracker = MLflowTracker()
        tracker.log_inference_run(
            conf_threshold=0.25, num_detections=7, inference_time_ms=10.0, num_violations=0
        )
        mocked_mlflow.log_metric.assert_any_call("num_detections", 7)

    def test_logs_environment_tag(self, mocked_mlflow):
        tracker = MLflowTracker()
        tracker.log_inference_run(
            conf_threshold=0.25,
            num_detections=1,
            inference_time_ms=10.0,
            num_violations=0,
            environment="prod",
        )
        mocked_mlflow.set_tag.assert_any_call("environment", "prod")

    @pytest.mark.parametrize("num_violations", [0, 1, 3, 10])
    def test_various_violation_counts_logged(self, mocked_mlflow, num_violations):
        tracker = MLflowTracker()
        tracker.log_inference_run(
            conf_threshold=0.25,
            num_detections=num_violations,
            inference_time_ms=5.0,
            num_violations=num_violations,
        )
        mocked_mlflow.log_metric.assert_any_call("num_violations", num_violations)


class TestLogEvaluation:
    def test_logs_all_evaluation_metrics(self, mocked_mlflow):
        tracker = MLflowTracker()
        tracker.log_evaluation(
            precision=0.9, recall=0.85, f1=0.87, map50=0.88, dataset_version="v1"
        )
        mocked_mlflow.log_metric.assert_any_call("precision", 0.9)
        mocked_mlflow.log_metric.assert_any_call("recall", 0.85)
        mocked_mlflow.log_metric.assert_any_call("f1_score", 0.87)
        mocked_mlflow.log_metric.assert_any_call("map50", 0.88)

    def test_tags_run_type_as_evaluation(self, mocked_mlflow):
        tracker = MLflowTracker()
        tracker.log_evaluation(precision=0.9, recall=0.9, f1=0.9, map50=0.9, dataset_version="v2")
        mocked_mlflow.set_tag.assert_any_call("run_type", "evaluation")
