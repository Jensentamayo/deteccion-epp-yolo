"""Tracking de experimentos e inferencias con MLflow."""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path
from typing import Any

import mlflow

DEFAULT_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

DEFAULT_EXPERIMENT = "deteccion-epp-yolo"

MODEL_HUB_REF = os.environ.get(
    "EPP_MODEL_HUB_REF",
    "ultralytics/yolo11n@main",
)

MODEL_LICENSE = os.environ.get(
    "EPP_MODEL_LICENSE",
    "AGPL-3.0",
)

MODEL_AUTHOR = os.environ.get(
    "EPP_MODEL_AUTHOR",
    "Ultralytics",
)

PR_ISSUE_LINK = os.environ.get(
    "EPP_PR_ISSUE_LINK",
    "N/A",
)


class MLflowTracker:
    """Encapsula la configuración y registro de MLflow."""

    def __init__(
        self,
        tracking_uri: str = DEFAULT_TRACKING_URI,
        experiment_name: str = DEFAULT_EXPERIMENT,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        self._latencies_ms: list[float] = []
        self._inference_count = 0
        self._start_time = time.perf_counter()

    def log_inference(
        self,
        inference_time_ms: float,
        num_detections: int,
        conf_threshold: float = 0.25,
        num_violations: int = 0,
        environment: str = "dev",
    ) -> None:
        """Registra una corrida de inferencia en MLflow."""
        self._latencies_ms.append(inference_time_ms)
        self._inference_count += 1

        elapsed_seconds = time.perf_counter() - self._start_time

        throughput = (
            self._inference_count / elapsed_seconds
            if elapsed_seconds > 0
            else 0.0
        )

        latency_mean = statistics.mean(self._latencies_ms)
        latency_p95 = self._calculate_p95(self._latencies_ms)

        with mlflow.start_run():
            self._log_common_params(
                conf_threshold=conf_threshold,
            )

            mlflow.log_metric("num_detections", num_detections)
            mlflow.log_metric("num_violations", num_violations)
            mlflow.log_metric(
                "inference_time_ms",
                inference_time_ms,
            )
            mlflow.log_metric(
                "latency_mean_ms",
                latency_mean,
            )
            mlflow.log_metric(
                "latency_p95_ms",
                latency_p95,
            )
            mlflow.log_metric(
                "throughput_fps",
                throughput,
            )

            self._set_common_tags(environment=environment)

    def log_inference_run(
        self,
        conf_threshold: float,
        num_detections: int,
        inference_time_ms: float,
        num_violations: int,
        environment: str = "dev",
    ) -> None:
        """Compatibilidad con la API anterior del tracker."""
        self.log_inference(
            inference_time_ms=inference_time_ms,
            num_detections=num_detections,
            conf_threshold=conf_threshold,
            num_violations=num_violations,
            environment=environment,
        )

    def log_evaluation(
        self,
        accuracy: float | None = None,
        precision: float = 0.0,
        recall: float = 0.0,
        f1: float = 0.0,
        auc_roc: float | None = None,
        map50: float = 0.0,
        latency_mean_ms: float | None = None,
        latency_p95_ms: float | None = None,
        throughput_fps: float | None = None,
        dataset_version: str = "unknown",
        backend: str = "cpu",
        map50_95: float | None = None,
        conf_threshold: float = 0.25,
        image_size: int | None = None,
        roc_conf_threshold: float | None = None,
        evaluation_protocol: str = "object_detection",
        artifact_path: str | Path | None = None,
        model_path: str | Path | None = None,
    ) -> None:
        """Registra métricas y artefactos de una evaluación real.

        Todos los datos de la evaluación se registran dentro de una
        única corrida de MLflow.
        """
        with mlflow.start_run():
            self._log_common_params(
                conf_threshold=conf_threshold,
            )

            mlflow.log_param(
                "dataset_version",
                dataset_version,
            )
            mlflow.log_param("backend", backend)

            if image_size is not None:
                mlflow.log_param("image_size", image_size)

            if roc_conf_threshold is not None:
                mlflow.log_param(
                    "roc_conf_threshold",
                    roc_conf_threshold,
                )

            if accuracy is not None:
                mlflow.log_metric("accuracy", accuracy)

            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("map50", map50)

            if map50_95 is not None:
                mlflow.log_metric(
                    "map50_95",
                    map50_95,
                )

            if auc_roc is not None:
                mlflow.log_metric("auc_roc", auc_roc)

            if latency_mean_ms is not None:
                mlflow.log_metric(
                    "latency_mean_ms",
                    latency_mean_ms,
                )

            if latency_p95_ms is not None:
                mlflow.log_metric(
                    "latency_p95_ms",
                    latency_p95_ms,
                )

            if throughput_fps is not None:
                mlflow.log_metric(
                    "throughput_fps",
                    throughput_fps,
                )

            self._set_common_tags()
            mlflow.set_tag("run_type", "evaluation")
            mlflow.set_tag(
                "evaluation_protocol",
                evaluation_protocol,
            )

            if model_path is not None:
                model_file = Path(model_path)

                if model_file.exists():
                    mlflow.log_artifact(
                        str(model_file),
                        artifact_path="model",
                    )

            if artifact_path is not None:
                artifact = Path(artifact_path)

                if artifact.exists():
                    mlflow.log_artifact(
                        str(artifact),
                        artifact_path="evaluation",
                    )

    def log_evaluation_report(
        self,
        report_path: str | Path,
        dataset_version: str = "epp_no_compliance_v1",
    ) -> None:
        """Lee un reporte JSON y registra su evaluación en MLflow."""
        import json

        path = Path(report_path)

        if not path.exists():
            raise FileNotFoundError(
                f"No existe el reporte de evaluación: {path}"
            )

        with path.open("r", encoding="utf-8") as file:
            report: dict[str, Any] = json.load(file)

        self.log_evaluation(
            accuracy=report.get("accuracy_image_level"),
            precision=report.get("precision", 0.0),
            recall=report.get("recall", 0.0),
            f1=report.get("f1_score", 0.0),
            auc_roc=report.get("auc_roc_image_level"),
            map50=report.get("map50", 0.0),
            map50_95=report.get("map50_95"),
            latency_mean_ms=report.get("latency_mean_ms"),
            latency_p95_ms=report.get("latency_p95_ms"),
            throughput_fps=report.get("throughput_fps"),
            dataset_version=dataset_version,
            backend=report.get("backend", "cpu"),
            conf_threshold=report.get(
                "confidence_threshold",
                0.25,
            ),
            image_size=report.get("image_size"),
            roc_conf_threshold=report.get(
                "roc_conf_threshold"
            ),
            evaluation_protocol=(
                "image_level_accuracy_auc; "
                "object_detection_precision_recall_map"
            ),
            artifact_path=path,
            model_path=report.get("model"),
        )

    def log_artifact(self, artifact_path: str | Path) -> None:
        """Registra un archivo como artefacto de MLflow."""
        path = Path(artifact_path)

        if not path.exists():
            raise FileNotFoundError(
                f"No existe el artefacto: {path}"
            )

        with mlflow.start_run():
            mlflow.log_artifact(str(path))

    @staticmethod
    def _calculate_p95(values: list[float]) -> float:
        """Calcula el percentil 95 de una lista de latencias."""
        if not values:
            return 0.0

        ordered = sorted(values)
        index = int(0.95 * (len(ordered) - 1))

        return ordered[index]

    def _log_common_params(
        self,
        conf_threshold: float,
    ) -> None:
        """Registra parámetros comunes de trazabilidad."""
        mlflow.log_param(
            "model_hub_ref",
            MODEL_HUB_REF,
        )
        mlflow.log_param(
            "model_revision",
            MODEL_HUB_REF.split("@")[-1],
        )
        mlflow.log_param(
            "conf_threshold",
            conf_threshold,
        )
        mlflow.log_param(
            "preprocessing",
            "resize_max_side_1280,rgb",
        )

    @staticmethod
    def _set_common_tags(
        environment: str = "dev",
    ) -> None:
        """Registra tags comunes del proyecto."""
        mlflow.set_tag(
            "model_license",
            MODEL_LICENSE,
        )
        mlflow.set_tag(
            "model_author",
            MODEL_AUTHOR,
        )
        mlflow.set_tag(
            "environment",
            environment,
        )
        mlflow.set_tag(
            "service",
            "epp-inference-grpc",
        )
        mlflow.set_tag(
            "team",
            "deteccion-epp-yolo",
        )
        mlflow.set_tag(
            "pr_issue_link",
            PR_ISSUE_LINK,
        )
