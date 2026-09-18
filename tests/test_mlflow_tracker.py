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

class TestMLflowTrackerAdditional:
    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ([], 0.0),
            ([10.0], 10.0),
            ([10.0, 20.0], 10.0),
            ([10.0, 20.0, 30.0, 40.0, 50.0], 40.0),
            ([50.0, 10.0, 30.0, 20.0, 40.0], 40.0),
        ],
    )
    def test_calculate_p95(self, values, expected):
        assert MLflowTracker._calculate_p95(values) == expected

    def test_logs_common_params(self, mocked_mlflow):
        tracker = MLflowTracker()
        tracker.log_inference(
            inference_time_ms=20.0,
            num_detections=2,
        )
        mocked_mlflow.log_param.assert_any_call(
            "model_hub_ref",
            "ultralytics/yolo11n@main",
        )
        mocked_mlflow.log_param.assert_any_call(
            "model_revision",
            "main",
        )
        mocked_mlflow.log_param.assert_any_call(
            "preprocessing",
            "resize_max_side_1280,rgb",
        )

    def test_logs_common_tags(self, mocked_mlflow):
        tracker = MLflowTracker()
        tracker.log_inference(
            inference_time_ms=20.0,
            num_detections=2,
            environment="test",
        )
        mocked_mlflow.set_tag.assert_any_call(
            "model_license",
            "AGPL-3.0",
        )
        mocked_mlflow.set_tag.assert_any_call(
            "model_author",
            "Ultralytics",
        )
        mocked_mlflow.set_tag.assert_any_call(
            "service",
            "epp-inference-grpc",
        )
        mocked_mlflow.set_tag.assert_any_call(
            "team",
            "deteccion-epp-yolo",
        )

    def test_logs_inference_metrics(self, mocked_mlflow):
        tracker = MLflowTracker()
        tracker.log_inference(
            inference_time_ms=25.0,
            num_detections=4,
            num_violations=2,
        )
        mocked_mlflow.log_metric.assert_any_call(
            "inference_time_ms",
            25.0,
        )
        mocked_mlflow.log_metric.assert_any_call(
            "num_violations",
            2,
        )
        mocked_mlflow.log_metric.assert_any_call(
            "latency_mean_ms",
            25.0,
        )
        mocked_mlflow.log_metric.assert_any_call(
            "latency_p95_ms",
            25.0,
        )

    def test_logs_optional_evaluation_metrics(self, mocked_mlflow):
        tracker = MLflowTracker()
        tracker.log_evaluation(
            accuracy=0.95,
            precision=0.90,
            recall=0.85,
            f1=0.87,
            auc_roc=0.98,
            map50=0.76,
            map50_95=0.36,
            latency_mean_ms=43.6,
            latency_p95_ms=48.3,
            throughput_fps=22.9,
            image_size=640,
            roc_conf_threshold=0.5,
        )
        mocked_mlflow.log_metric.assert_any_call("accuracy", 0.95)
        mocked_mlflow.log_metric.assert_any_call("auc_roc", 0.98)
        mocked_mlflow.log_metric.assert_any_call("map50_95", 0.36)
        mocked_mlflow.log_metric.assert_any_call(
            "latency_mean_ms",
            43.6,
        )
        mocked_mlflow.log_metric.assert_any_call(
            "throughput_fps",
            22.9,
        )
        mocked_mlflow.log_param.assert_any_call("image_size", 640)
        mocked_mlflow.log_param.assert_any_call(
            "roc_conf_threshold",
            0.5,
        )

    def test_log_evaluation_report_reads_json(
        self,
        mocked_mlflow,
        tmp_path,
    ):
        report_path = tmp_path / "evaluation.json"
        report_path.write_text(
            '{"precision": 0.9, "recall": 0.8}',
            encoding="utf-8",
        )
        tracker = MLflowTracker()
        with patch.object(
            tracker,
            "log_evaluation",
        ) as mock_log_evaluation:
            tracker.log_evaluation_report(
                report_path,
                dataset_version="test-v1",
            )
        mock_log_evaluation.assert_called_once()
        kwargs = mock_log_evaluation.call_args.kwargs
        assert kwargs["precision"] == 0.9
        assert kwargs["recall"] == 0.8
        assert kwargs["dataset_version"] == "test-v1"

    def test_log_evaluation_report_raises_for_missing_file(
        self,
        mocked_mlflow,
        tmp_path,
    ):
        tracker = MLflowTracker()
        missing_path = tmp_path / "missing.json"
        with pytest.raises(FileNotFoundError):
            tracker.log_evaluation_report(missing_path)
