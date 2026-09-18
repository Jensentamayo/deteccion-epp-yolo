from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml

DATASET = Path("data/raw/complementary/street-work")


def load_classes() -> list[str]:
    yaml_path = DATASET / "data.yaml"

    with yaml_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    names = data.get("names", [])

    if isinstance(names, dict):
        return [names[index] for index in sorted(names)]

    return list(names)


def analyze_split(split: str, classes: list[str]) -> None:
    label_dir = DATASET / split / "labels"
    image_dir = DATASET / split / "images"

    print()
    print("=" * 70)
    print(f"STREET WORK - {split.upper()}")
    print("=" * 70)

    if not label_dir.exists():
        print(f"ERROR: no existe {label_dir}")
        return

    image_count = sum(
        1
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    label_files = list(label_dir.glob("*.txt"))

    counter: Counter[int] = Counter()

    for label_file in label_files:
        try:
            lines = label_file.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue

        for line in lines:
            parts = line.strip().split()

            if not parts:
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                continue

            counter[class_id] += 1

    print(f"Imágenes: {image_count}")
    print(f"Archivos de etiquetas: {len(label_files)}")
    print(f"Objetos anotados: {sum(counter.values())}")

    print()
    print("OBJETOS POR CLASE:")

    for class_id, class_name in enumerate(classes):
        print(
            f"  {class_id:>2} | "
            f"{class_name:<20} | "
            f"{counter.get(class_id, 0)}"
        )


def main() -> None:
    classes = load_classes()

    print()
    print("=" * 70)
    print("STREET WORK - CLASES")
    print("=" * 70)

    for class_id, class_name in enumerate(classes):
        print(f"  {class_id:>2} | {class_name}")

    for split in ("train", "valid", "test"):
        analyze_split(split, classes)


if __name__ == "__main__":
    main()
