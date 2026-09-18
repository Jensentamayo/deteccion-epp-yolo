"""Registra en MLflow los resultados de evaluación del modelo."""

from src.tracking.mlflow_tracker import MLflowTracker

REPORT_PATH = "reports/evaluation_report.json"


def main() -> None:
    """Registra el reporte de evaluación en MLflow."""
    tracker = MLflowTracker()

    tracker.log_evaluation_report(
        report_path=REPORT_PATH,
        dataset_version="epp_no_compliance_v1",
    )

    print("Evaluación registrada correctamente en MLflow.")
    print(f"Reporte: {REPORT_PATH}")


if __name__ == "__main__":
    main()
