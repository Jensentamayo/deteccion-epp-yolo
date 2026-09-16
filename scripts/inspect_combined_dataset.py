
from __future__ import annotations

import random
from pathlib import Path

import cv2


DATASET = Path("data/processed/epp_combined")
OUTPUT = DATASET / "inspection"

CLASS_NAMES = [
    "helmet",
    "no_helmet",
    "Gloves",
    "no_gloves",
    "Vest",
    "Goggles",
    "Mask",
    "safety_shoe",
]

# Clases que representan cumplimiento.
COMPLIANT_CLASSES = {
    "helmet",
    "Gloves",
    "Vest",
    "Goggles",
    "Mask",
    "safety_shoe",
}

# Clases que representan incumplimiento.
NON_COMPLIANT_CLASSES = {
    "no_helmet",
    "no_gloves",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

SAMPLES_PER_SPLIT = 10
SEED = 42


def load_labels(
    label_path: Path,
) -> list[tuple[int, float, float, float, float]]:
    labels = []

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        labels.append(
            (
                class_id,
                x_center,
                y_center,
                width,
                height,
            )
        )

    return labels


def get_class_status(class_name: str) -> tuple[tuple[int, int, int], str]:
    """
    Devuelve el color BGR y el estado asociado a una clase.
    """

    if class_name in NON_COMPLIANT_CLASSES:
        return (0, 0, 255), "INCUMPLE"

    if class_name in COMPLIANT_CLASSES:
        return (0, 255, 0), "CUMPLE"

    return (255, 255, 255), "SIN EVALUAR"


def draw_labels(
    image,
    labels: list[tuple[int, float, float, float, float]],
):
    height, width = image.shape[:2]

    compliant_count = 0
    non_compliant_count = 0

    for class_id, xc, yc, w, h in labels:
        x1 = int((xc - w / 2) * width)
        y1 = int((yc - h / 2) * height)
        x2 = int((xc + w / 2) * width)
        y2 = int((yc + h / 2) * height)

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        class_name = CLASS_NAMES[class_id]
        color, status = get_class_status(class_name)

        if class_name in COMPLIANT_CLASSES:
            compliant_count += 1
        elif class_name in NON_COMPLIANT_CLASSES:
            non_compliant_count += 1

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            3,
        )

        label = f"{class_name} - {status}"

        text_y = max(y1 - 10, 25)

        cv2.putText(
            image,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    # Estado general de la imagen.
    if non_compliant_count > 0:
        general_status = "EPP INCOMPLETO"
        general_color = (0, 0, 255)
    else:
        general_status = "EPP COMPLETO"
        general_color = (0, 255, 0)

    # Fondo para el estado general.
    cv2.rectangle(
        image,
        (10, 10),
        (390, 65),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        image,
        general_status,
        (25, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        general_color,
        3,
        cv2.LINE_AA,
    )

    return image


def inspect_split(split: str, rng: random.Random) -> None:
    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split
    output_dir = OUTPUT / split

    output_dir.mkdir(parents=True, exist_ok=True)

    images = [
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if len(images) > SAMPLES_PER_SPLIT:
        images = rng.sample(images, SAMPLES_PER_SPLIT)

    print()
    print("=" * 70)
    print(f"INSPECCIONANDO: {split.upper()}")
    print("=" * 70)

    for index, image_path in enumerate(images, start=1):
        label_path = label_dir / f"{image_path.stem}.txt"

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[ERROR] No se pudo leer: {image_path}")
            continue

        if not label_path.exists():
            print(f"[ERROR] No existe etiqueta: {label_path}")
            continue

        labels = load_labels(label_path)
        annotated = draw_labels(image, labels)

        output_path = output_dir / (
            f"sample_{index:02d}_{image_path.stem}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )

        print(f"[OK] {output_path}")


def main() -> None:
    random.seed(SEED)

    if OUTPUT.exists():
        for file in OUTPUT.rglob("*.jpg"):
            file.unlink()

    OUTPUT.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print("INSPECCIÓN VISUAL DATASET EPP CONSOLIDADO")
    print("EVALUACIÓN DE CUMPLIMIENTO EPP")
    print("=" * 70)

    print()
    print("Clases que cumplen:")
    for class_name in sorted(COMPLIANT_CLASSES):
        print(f"  [VERDE] {class_name}")

    print()
    print("Clases que incumplen:")
    for class_name in sorted(NON_COMPLIANT_CLASSES):
        print(f"  [ROJO]   {class_name}")

    rng = random.Random(SEED)

    for split in ("train", "valid", "test"):
        inspect_split(split, rng)

    print()
    print("=" * 70)
    print("INSPECCIÓN COMPLETADA")
    print("=" * 70)
    print()
    print(f"Las imágenes están en: {OUTPUT}")
    print()


if __name__ == "__main__":
    main()
