from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

DATASET = Path("data/processed/inspect_protective")


def analyze_split(split: str) -> None:
    json_path = DATASET / split / "_annotations.coco.json"

    print()
    print("=" * 70)
    print(f"PROTECTIVE EQUIPMENT - {split.upper()}")
    print("=" * 70)

    if not json_path.exists():
        print(f"ERROR: no existe {json_path}")
        return

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    categories = {
        category["id"]: category["name"]
        for category in data.get("categories", [])
    }

    images = data.get("images", [])
    annotations = data.get("annotations", [])

    counts = Counter(
        annotation["category_id"]
        for annotation in annotations
        if "category_id" in annotation
    )

    print(f"Imágenes: {len(images)}")
    print(f"Anotaciones: {len(annotations)}")

    print()
    print("CLASES:")
    for category_id, name in sorted(categories.items()):
        print(f"  {category_id:>2} | {name}")

    print()
    print("OBJETOS POR CLASE:")

    for category_id, name in sorted(categories.items()):
        print(
            f"  {category_id:>2} | "
            f"{name:<20} | "
            f"{counts.get(category_id, 0)}"
        )


def main() -> None:
    for split in ("train", "valid", "test"):
        analyze_split(split)


if __name__ == "__main__":
    main()
