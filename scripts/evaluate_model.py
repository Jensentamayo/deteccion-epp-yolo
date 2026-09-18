"""Evaluación formal del modelo YOLO11n de detección de incumplimientos."""

from __future__ import annotations

import json
import time
from pathlib import Path

from sklearn.metrics import accuracy_score, roc_auc_score
from ultralytics import YOLO

MODEL_PATH = Path(
    "models/trained/epp_no_compliance_yolo11n_final.pt"
)

DATASET_YAML = Path(
    "data/processed/epp_no_compliance/data.yaml"
)

VALID_IMAGES_DIR = Path(
    "data/processed/epp_no_compliance/images/valid"
)

VALID_LABELS_DIR = Path(
    "data/processed/epp_no_compliance/labels/valid"
)

REPORT_PATH = Path(
    "reports/evaluation_report.json"
)

IMAGE_SIZE = 640
CONF_THRESHOLD = 0.05
ROC_CONF_THRESHOLD = 0.001
DEVICE = "cpu"

ALLOWED_CLASSES = {
    0,
    1,
}


def calculate_f1(
    precision: float,
    recall: float,
) -> float:
    """Calcula F1 a partir de precision y recall."""
    denominator = precision + recall

    if denominator == 0:
        return 0.0

    return 2 * precision * recall / denominator


def has_ground_truth_violation(
    label_path: Path,
) -> bool:
    """Determina si una imagen contiene algún incumplimiento."""
    if not label_path.exists():
        return False

    content = label_path.read_text(
        encoding="utf-8"
    ).strip()

    if not content:
        return False

    for line in content.splitlines():
        parts = line.split()

        if not parts:
            continue

        class_id = int(parts[0])

        if class_id in ALLOWED_CLASSES:
            return True

    return False


def evaluate_image_level_metrics(
    model: YOLO,
    image_paths: list[Path],
) -> tuple[float, float, int]:
    """Calcula accuracy y AUC-ROC a nivel de imagen."""
    ground_truth: list[int] = []
    predicted: list[int] = []
    scores: list[float] = []

    for image_path in image_paths:
        label_path = VALID_LABELS_DIR / (
            f"{image_path.stem}.txt"
        )

        actual_positive = has_ground_truth_violation(
            label_path
        )

        results = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=ROC_CONF_THRESHOLD,
            device=DEVICE,
            verbose=False,
        )

        max_confidence = 0.0

        if results:
            result = results[0]

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])

                    if class_id not in ALLOWED_CLASSES:
                        continue

                    confidence = float(box.conf[0])
                    max_confidence = max(
                        max_confidence,
                        confidence,
                    )

        predicted_positive = (
            max_confidence >= CONF_THRESHOLD
        )

        ground_truth.append(
            int(actual_positive)
        )
        predicted.append(
            int(predicted_positive)
        )
        scores.append(max_confidence)

    accuracy = accuracy_score(
        ground_truth,
        predicted,
    )

    unique_classes = len(set(ground_truth))

    auc_roc = (
        roc_auc_score(ground_truth, scores)
        if unique_classes == 2
        else 0.0
    )

    return (
        float(accuracy),
        float(auc_roc),
        len(ground_truth),
    )


def evaluate_model() -> dict[str, float | str | int]:
    """Evalúa el modelo usando el conjunto de validación."""
    print("=" * 60)
    print("EVALUACIÓN DEL MODELO YOLO11n")
    print("=" * 60)

    print(f"Modelo: {MODEL_PATH}")
    print(f"Dataset: {DATASET_YAML}")
    print(f"Device: {DEVICE}")
    print(f"Confidence: {CONF_THRESHOLD}")
    print()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No existe el modelo: {MODEL_PATH}"
        )

    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"No existe el dataset YAML: {DATASET_YAML}"
        )

    if not VALID_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"No existe el directorio: "
            f"{VALID_IMAGES_DIR}"
        )

    model = YOLO(str(MODEL_PATH))

    print("Ejecutando validación...")
    print()

    metrics = model.val(
        data=str(DATASET_YAML),
        imgsz=IMAGE_SIZE,
        conf=CONF_THRESHOLD,
        device=DEVICE,
        verbose=False,
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    f1 = calculate_f1(
        precision=precision,
        recall=recall,
    )

    image_paths = sorted(
        path
        for path in VALID_IMAGES_DIR.iterdir()
        if path.suffix.lower()
        in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    print()
    print(
        "Calculando accuracy y AUC-ROC "
        "a nivel de imagen..."
    )

    accuracy, auc_roc, evaluated_images = (
        evaluate_image_level_metrics(
            model=model,
            image_paths=image_paths,
        )
    )

    print()
    print("Midiendo latencia y throughput...")

    benchmark_paths = image_paths[:50]
    latencies: list[float] = []

    for image_path in benchmark_paths:
        start = time.perf_counter()

        model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=CONF_THRESHOLD,
            device=DEVICE,
            verbose=False,
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(elapsed_ms)

    if latencies:
        latency_mean = sum(latencies) / len(latencies)

        ordered = sorted(latencies)

        p95_index = int(
            0.95 * (len(ordered) - 1)
        )

        latency_p95 = ordered[p95_index]

        total_seconds = sum(latencies) / 1000

        throughput = (
            len(latencies) / total_seconds
            if total_seconds > 0
            else 0.0
        )
    else:
        latency_mean = 0.0
        latency_p95 = 0.0
        throughput = 0.0

    report = {
        "model": str(MODEL_PATH),
        "dataset": str(DATASET_YAML),
        "backend": DEVICE,
        "confidence_threshold": CONF_THRESHOLD,
        "roc_conf_threshold": ROC_CONF_THRESHOLD,
        "image_size": IMAGE_SIZE,
        "validation_images": len(image_paths),
        "evaluated_images": evaluated_images,
        "accuracy_image_level": accuracy,
        "auc_roc_image_level": auc_roc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "map50": map50,
        "map50_95": map50_95,
        "latency_mean_ms": latency_mean,
        "latency_p95_ms": latency_p95,
        "throughput_fps": throughput,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=4,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print("RESULTADOS")
    print("=" * 60)

    print(f"Accuracy (imagen) : {accuracy:.4f}")
    print(f"AUC-ROC (imagen)  : {auc_roc:.4f}")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"F1                : {f1:.4f}")
    print(f"mAP50             : {map50:.4f}")
    print(f"mAP50-95          : {map50_95:.4f}")
    print(
        f"Latencia media    : "
        f"{latency_mean:.2f} ms"
    )
    print(
        f"Latencia P95      : "
        f"{latency_p95:.2f} ms"
    )
    print(
        f"Throughput        : "
        f"{throughput:.4f} FPS"
    )

    print()
    print(
        f"Imágenes evaluadas: "
        f"{evaluated_images}"
    )

    print(
        f"Reporte guardado en: "
        f"{REPORT_PATH}"
    )

    return report


if __name__ == "__main__":
    evaluate_model()
