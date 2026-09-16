from __future__ import annotations

from collections import Counter
from pathlib import Path


DATASET = Path("data/processed/epp_combined")

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


def validate_split(split: str) -> tuple[Counter[int], int]:
    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split

    counts: Counter[int] = Counter()

    errors = 0
    images_without_labels = 0
    labels_without_images = 0

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    images = {
        path.stem: path
        for path in image_dir.iterdir()
        if path.suffix.lower() in image_extensions
    }

    labels = {
        path.stem: path
        for path in label_dir.glob("*.txt")
    }

    for stem in images:
        if stem not in labels:
            images_without_labels += 1
            errors += 1

    for stem in labels:
        if stem not in images:
            labels_without_images += 1
            errors += 1

    for label_path in labels.values():
        for line_number, line in enumerate(
            label_path.read_text(
                encoding="utf-8"
            ).splitlines(),
            start=1,
        ):
            parts = line.strip().split()

            if len(parts) != 5:
                print(
                    f"[ERROR] {label_path}:{line_number} "
                    f"cantidad de campos incorrecta"
                )
                errors += 1
                continue

            try:
                class_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError:
                print(
                    f"[ERROR] {label_path}:{line_number} "
                    f"valores no numéricos"
                )
                errors += 1
                continue

            if not 0 <= class_id < len(CLASS_NAMES):
                print(
                    f"[ERROR] {label_path}:{line_number} "
                    f"class_id inválido: {class_id}"
                )
                errors += 1
                continue

            values = (x, y, width, height)

            if any(
                value < 0 or value > 1
                for value in values
            ):
                print(
                    f"[ERROR] {label_path}:{line_number} "
                    f"coordenadas fuera de rango: "
                    f"{values}"
                )
                errors += 1
                continue

            if width <= 0 or height <= 0:
                print(
                    f"[ERROR] {label_path}:{line_number} "
                    f"ancho/alto inválido"
                )
                errors += 1
                continue

            counts[class_id] += 1

    print()
    print("=" * 70)
    print(f"SPLIT: {split.upper()}")
    print("=" * 70)

    print(f"Imágenes:              {len(images)}")
    print(f"Etiquetas:             {len(labels)}")
    print(f"Imágenes sin etiqueta: {images_without_labels}")
    print(f"Etiquetas sin imagen:  {labels_without_images}")
    print(f"Errores encontrados:   {errors}")

    print()
    print("OBJETOS:")

    for class_id, class_name in enumerate(CLASS_NAMES):
        print(
            f"  {class_id}: "
            f"{class_name:<15} "
            f"{counts[class_id]}"
        )

    return counts, errors


def main() -> None:
    print()
    print("=" * 70)
    print("AUDITORÍA DATASET EPP CONSOLIDADO")
    print("=" * 70)

    total_counts: Counter[int] = Counter()
    total_errors = 0

    for split in ("train", "valid", "test"):
        counts, errors = validate_split(split)
        total_counts.update(counts)
        total_errors += errors

    print()
    print("=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)

    print()
    print("OBJETOS TOTALES:")

    for class_id, class_name in enumerate(CLASS_NAMES):
        print(
            f"  {class_id}: "
            f"{class_name:<15} "
            f"{total_counts[class_id]}"
        )

    print()
    print(f"TOTAL DE ERRORES: {total_errors}")

    if total_errors == 0:
        print()
        print("OK: dataset estructuralmente válido.")
    else:
        print()
        print(
            "ATENCIÓN: hay errores que deben revisarse "
            "antes de entrenar."
        )


if __name__ == "__main__":
    main()
