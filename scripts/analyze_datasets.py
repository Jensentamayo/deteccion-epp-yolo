from __future__ import annotations

from collections import Counter
from pathlib import Path

DATASETS = {
    "PPE_Detection": Path("data/raw/PPE_Detection"),
    "Protective Equipment": Path(
        "data/raw/complementary/protective-equipment-detection"
    ),
    "Street Work": Path(
        "data/raw/complementary/street-work"
    ),
}


def find_yaml(dataset_path: Path) -> Path | None:
    """Busca el archivo data.yaml del dataset."""
    candidates = list(dataset_path.rglob("data.yaml"))

    if not candidates:
        return None

    return candidates[0]


def find_label_files(dataset_path: Path) -> list[Path]:
    """Busca todos los archivos de etiquetas YOLO."""
    return list(dataset_path.rglob("*.txt"))


def analyze_labels(
    label_files: list[Path],
) -> Counter[int]:
    """Cuenta los IDs de clase encontrados en etiquetas YOLO."""
    counter: Counter[int] = Counter()

    for label_file in label_files:
        try:
            lines = label_file.read_text(
                encoding="utf-8"
            ).splitlines()
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

    return counter


def extract_names(yaml_path: Path) -> list[str]:
    """Extrae nombres de clases de un data.yaml sencillo."""
    names = []

    inside_names = False

    for line in yaml_path.read_text(
        encoding="utf-8"
    ).splitlines():

        stripped = line.strip()

        if stripped.startswith("names:"):
            inside_names = True

            value = stripped.split(
                ":", 1
            )[1].strip()

            if value.startswith("["):
                value = value.strip("[]")

                names = [
                    item.strip().strip("'\"")
                    for item in value.split(",")
                    if item.strip()
                ]

                break

            continue

        if inside_names and ":" in stripped:
            key, value = stripped.split(
                ":", 1
            )

            try:
                index = int(key.strip())
            except ValueError:
                continue

            while len(names) <= index:
                names.append("")

            names[index] = value.strip().strip("'\"")

    return names


def analyze_dataset(
    name: str,
    dataset_path: Path,
) -> None:
    """Analiza un dataset."""
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    if not dataset_path.exists():
        print(
            f"ERROR: no existe {dataset_path}"
        )
        return

    yaml_path = find_yaml(dataset_path)

    if yaml_path is None:
        print("No se encontró data.yaml.")
        return

    print(f"data.yaml: {yaml_path}")

    names = extract_names(yaml_path)

    print()
    print("CLASES:")
    for index, class_name in enumerate(names):
        print(f"  {index}: {class_name}")

    label_files = find_label_files(
        dataset_path
    )

    print()
    print(
        f"Archivos .txt encontrados: "
        f"{len(label_files)}"
    )

    counts = analyze_labels(
        label_files
    )

    print()
    print("OBJETOS POR CLASE:")

    for class_id, count in sorted(
        counts.items()
    ):
        class_name = names[class_id] if class_id < len(names) else "DESCONOCIDA"

        print(
            f"  {class_id:>2} | "
            f"{class_name:<20} | "
            f"{count}"
        )


def main() -> None:
    """Analiza todos los datasets."""
    for name, path in DATASETS.items():
        analyze_dataset(
            name=name,
            dataset_path=path,
        )


if __name__ == "__main__":
    main()
