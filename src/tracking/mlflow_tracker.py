"""Tracking de experimentos/inferencias con MLflow.

Registra por cada corrida (run) lo indicado en el enunciado del Módulo 3
para modelos MaaS (sin fine-tuning): parámetros de configuración de la
inferencia, métricas de calidad/rendimiento, artefactos y tags de
trazabilidad.
"""

from __future__ import annotations

import os

import mlflow

DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
DEFAULT_EXPERIMENT = "deteccion-epp-yolo"

MODEL_HUB_REF = os.environ.get("EPP_MODEL_HUB_REF", "ultralytics/yolo11n@main")
MODEL_LICENSE = os.environ.get("EPP_MODEL_LICENSE", "AGPL-3.0")


class MLflowTracker:
    """Encapsula la configuración y el logging de runs de MLflow."""

    def __init__(
        self,
        tracking_uri: str = DEFAULT_TRACKING_URI,
        experiment_name: str = DEFAULT_EXPERIMENT,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def log_inference_run(
        self,
        conf_threshold: float,
        num_detections: int,
        inference_time_ms: float,
        num_violations: int,
        environment: str = "dev",
    ) -> None:
        """Registra un run de inferencia con params, métricas y tags.

        Args:
            conf_threshold: Umbral de confianza usado en esta corrida.
            num_detections: Total de objetos detectados.
            inference_time_ms: Latencia de la inferencia en milisegundos.
            num_violations: Cantidad de incumplimientos de EPP detectados.
            environment: "dev" | "prod", para trazabilidad.
        """
        with mlflow.start_run():
            mlflow.log_param("model_hub_ref", MODEL_HUB_REF)
            mlflow.log_param("conf_threshold", conf_threshold)
            mlflow.log_param("preprocessing", "resize_max_side_1280,rgb")

            mlflow.log_metric("num_detections", num_detections)
            mlflow.log_metric("num_violations", num_violations)
            mlflow.log_metric("inference_time_ms", inference_time_ms)

            mlflow.set_tag("model_license", MODEL_LICENSE)
            mlflow.set_tag("environment", environment)
            mlflow.set_tag("service", "epp-inference-grpc")

    def log_evaluation(
        self,
        precision: float,
        recall: float,
        f1: float,
        map50: float,
        dataset_version: str,
    ) -> None:
        """Registra un run de evaluación offline del modelo sobre el set de validación."""
        with mlflow.start_run():
            mlflow.log_param("model_hub_ref", MODEL_HUB_REF)
            mlflow.log_param("dataset_version", dataset_version)

            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("map50", map50)

            mlflow.set_tag("model_license", MODEL_LICENSE)
            mlflow.set_tag("run_type", "evaluation")
