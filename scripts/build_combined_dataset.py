from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(".")

PPE_DATASET = ROOT / "data/raw/PPE_Detection"
PROTECTIVE_DATASET = (
    ROOT / "data/processed/inspect_protective"
)
STREET_DATASET = (
    ROOT / "data/raw/complementary/street-work"
)

OUTPUT = ROOT / "data/processed/epp_combined"

SEED = 42

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

CLASS_TO_ID = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


def normalize_class_name(name: str) -> str | None:
    """Convierte nombres de clases de los tres datasets."""
    normalized = name.strip().lower()

    aliases = {
        "helmet": "helmet",
        "no helmet": "no_helmet",
        "no_helmet": "no_helmet",

        "glove": "Gloves",
        "gloves": "Gloves",
        "no glove": "no_gloves",
        "no gloves": "no_gloves",
        "no_glove": "no_gloves",

        "vest": "Vest",

        "goggle": "Goggles",
        "goggles": "Goggles",

        "mask": "Mask",

        "safety_shoe": "safety_shoe",
        "safety shoe": "safety_shoe",
        "shoes": "safety_shoe",
    }

    return aliases.get(normalized)


def prepare_output() -> None:
    """Crea desde cero el dataset consolidado."""
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    for split in ("train", "valid", "test"):
        (OUTPUT / "images" / split).mkdir(
            parents=True,
            exist_ok=True,
        )
        (OUTPUT / "labels" / split).mkdir(
            parents=True,
            exist_ok=True,
        )


def read_yolo_classes(yaml_path: Path) -> list[str]:
    """Lee las clases de un data.yaml YOLO."""
    with yaml_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    names = data.get("names", [])

    if isinstance(names, dict):
        return [
            names[index]
            for index in sorted(names)
        ]

    return list(names)


def collect_ppe_detection() -> list[tuple[Path, list[str]]]:
    """Recolecta imágenes y etiquetas del dataset PPE_Detection."""
    items: list[tuple[Path, list[str]]] = []

    yaml_path = PPE_DATASET / "data.yaml"
    classes = read_yolo_classes(yaml_path)

    for split in ("train", "valid", "test"):
        image_dir = PPE_DATASET / split / "images"
        label_dir = PPE_DATASET / split / "labels"

        for image_path in image_dir.iterdir():
            if image_path.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
            }:
                continue

            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                continue

            lines: list[str] = []

            for line in label_path.read_text(
                encoding="utf-8"
            ).splitlines():
                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                try:
                    class_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except ValueError:
                    continue

                if not 0 <= class_id < len(classes):
                    continue

                class_name = normalize_class_name(
                    classes[class_id]
                )

                if class_name is None:
                    continue

                new_id = CLASS_TO_ID[class_name]

                lines.append(
                    f"{new_id} {x:.6f} {y:.6f} "
                    f"{w:.6f} {h:.6f}"
                )

            if lines:
                items.append(
                    (
                        image_path,
                        lines,
                    )
                )

    return items


def collect_street_work() -> list[tuple[Path, list[str]]]:
    """Recolecta imágenes y etiquetas de Street Work."""
    items: list[tuple[Path, list[str]]] = []

    yaml_path = STREET_DATASET / "data.yaml"
    classes = read_yolo_classes(yaml_path)

    for split in ("train", "valid", "test"):
        image_dir = STREET_DATASET / split / "images"
        label_dir = STREET_DATASET / split / "labels"

        for image_path in image_dir.iterdir():
            if image_path.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
            }:
                continue

            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                continue

            lines: list[str] = []

            for line in label_path.read_text(
                encoding="utf-8"
            ).splitlines():
                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                try:
                    class_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except ValueError:
                    continue

                if not 0 <= class_id < len(classes):
                    continue

                class_name = normalize_class_name(
                    classes[class_id]
                )

                if class_name is None:
                    continue

                new_id = CLASS_TO_ID[class_name]

                lines.append(
                    f"{new_id} {x:.6f} {y:.6f} "
                    f"{w:.6f} {h:.6f}"
                )

            if lines:
                items.append(
                    (
                        image_path,
                        lines,
                    )
                )

    return items


def collect_protective_equipment() -> list[tuple[Path, list[str]]]:
    """Convierte anotaciones COCO de Protective Equipment a YOLO."""
    items: list[tuple[Path, list[str]]] = []

    for split in ("train", "valid", "test"):
        split_dir = PROTECTIVE_DATASET / split
        json_path = split_dir / "_annotations.coco.json"

        if not json_path.exists():
            continue

        with json_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        categories = {
            category["id"]: category["name"]
            for category in data.get("categories", [])
        }

        images = {
            image["id"]: image
            for image in data.get("images", [])
        }

        annotations_by_image: dict[
            int,
            list[dict],
        ] = {}

        for annotation in data.get(
            "annotations",
            [],
        ):
            annotations_by_image.setdefault(
                annotation["image_id"],
                [],
            ).append(annotation)

        for image_id, image_info in images.items():
            file_name = image_info["file_name"]
            image_path = split_dir / file_name

            if not image_path.exists():
                continue

            lines: list[str] = []

            image_width = image_info["width"]
            image_height = image_info["height"]

            for annotation in annotations_by_image.get(
                image_id,
                [],
            ):
                category_name = categories.get(
                    annotation["category_id"]
                )

                if category_name is None:
                    continue

                class_name = normalize_class_name(
                    category_name
                )

                if class_name is None:
                    continue

                bbox = annotation.get("bbox")

                if not bbox or len(bbox) != 4:
                    continue

                x, y, width, height = bbox

                if image_width <= 0 or image_height <= 0:
                    continue

                center_x = (
                    x + width / 2
                ) / image_width

                center_y = (
                    y + height / 2
                ) / image_height

                normalized_width = (
                    width / image_width
                )

                normalized_height = (
                    height / image_height
                )

                new_id = CLASS_TO_ID[class_name]

                lines.append(
                    f"{new_id} "
                    f"{center_x:.6f} "
                    f"{center_y:.6f} "
                    f"{normalized_width:.6f} "
                    f"{normalized_height:.6f}"
                )

            if lines:
                items.append(
                    (
                        image_path,
                        lines,
                    )
                )

    return items


def copy_dataset(
    items: list[tuple[Path, list[str]]],
) -> Counter[str]:
    """Mezcla los datasets y crea train/valid/test."""
    random.seed(SEED)
    random.shuffle(items)

    total = len(items)

    train_end = int(total * 0.80)
    valid_end = int(total * 0.90)

    splits = [
        (
            "train",
            items[:train_end],
        ),
        (
            "valid",
            items[train_end:valid_end],
        ),
        (
            "test",
            items[valid_end:],
        ),
    ]

    object_counter: Counter[str] = Counter()

    for split, split_items in splits:
        for index, (image_path, labels) in enumerate(
            split_items
        ):
            extension = image_path.suffix.lower()

            new_stem = (
                f"{image_path.parent.parent.parent.name}_"
                f"{image_path.stem}_"
                f"{index:06d}"
            )

            destination_image = (
                OUTPUT
                / "images"
                / split
                / f"{new_stem}{extension}"
            )

            destination_label = (
                OUTPUT
                / "labels"
                / split
                / f"{new_stem}.txt"
            )

            shutil.copy2(
                image_path,
                destination_image,
            )

            destination_label.write_text(
                "\n".join(labels) + "\n",
                encoding="utf-8",
            )

            for label in labels:
                class_id = int(label.split()[0])
                class_name = CLASS_NAMES[class_id]
                object_counter[class_name] += 1

    return object_counter


def write_data_yaml() -> None:
    """Genera el data.yaml final."""
    data = {
        "path": str(OUTPUT.resolve()),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }

    with (OUTPUT / "data.yaml").open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
            allow_unicode=True,
        )


def write_report(
    original_counts: dict[str, int],
    final_counts: Counter[str],
    total_items: int,
) -> None:
    """Genera un reporte del dataset consolidado."""
    report_path = OUTPUT / "dataset_report.txt"

    lines = [
        "DATASET EPP CONSOLIDADO",
        "=" * 60,
        "",
        f"Seed: {SEED}",
        f"Total de imágenes: {total_items}",
        f"Total de clases: {len(CLASS_NAMES)}",
        "",
        "CLASES:",
    ]

    for index, name in enumerate(CLASS_NAMES):
        lines.append(f"{index}: {name}")

    lines.extend(
        [
            "",
            "OBJETOS CONSOLIDADOS:",
        ]
    )

    for class_name in CLASS_NAMES:
        lines.append(
            f"{class_name:<20} "
            f"{final_counts[class_name]}"
        )

    lines.extend(
        [
            "",
            "OBJETOS POR DATASET ANTES DE LA MEZCLA:",
        ]
    )

    for dataset_name, count in original_counts.items():
        lines.append(
            f"{dataset_name:<30} {count}"
        )

    report_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    print()
    print("=" * 70)
    print("CONSTRUYENDO DATASET EPP CONSOLIDADO")
    print("=" * 70)

    prepare_output()

    print()
    print("1. Leyendo PPE_Detection...")
    ppe_items = collect_ppe_detection()
    print(f"   Imágenes válidas: {len(ppe_items)}")

    print()
    print("2. Leyendo Protective Equipment...")
    protective_items = collect_protective_equipment()
    print(
        f"   Imágenes válidas: "
        f"{len(protective_items)}"
    )

    print()
    print("3. Leyendo Street Work...")
    street_items = collect_street_work()
    print(
        f"   Imágenes válidas: "
        f"{len(street_items)}"
    )

    all_items = (
        ppe_items
        + protective_items
        + street_items
    )

    print()
    print(
        f"4. Total de imágenes a combinar: "
        f"{len(all_items)}"
    )

    original_counts = {
        "PPE_Detection": sum(
            len(labels)
            for _, labels in ppe_items
        ),
        "Protective Equipment": sum(
            len(labels)
            for _, labels in protective_items
        ),
        "Street Work": sum(
            len(labels)
            for _, labels in street_items
        ),
    }

    final_counts = copy_dataset(all_items)

    write_data_yaml()

    write_report(
        original_counts=original_counts,
        final_counts=final_counts,
        total_items=len(all_items),
    )

    print()
    print("=" * 70)
    print("DATASET CONSOLIDADO CREADO")
    print("=" * 70)

    print()
    print(f"Ubicación: {OUTPUT}")
    print()

    print("CLASES FINALES:")

    for index, class_name in enumerate(CLASS_NAMES):
        print(
            f"  {index}: "
            f"{class_name:<20} "
            f"{final_counts[class_name]}"
        )

    print()
    print("ARCHIVOS:")
    print(f"  {OUTPUT / 'data.yaml'}")
    print(f"  {OUTPUT / 'dataset_report.txt'}")


if __name__ == "__main__":
    main()
