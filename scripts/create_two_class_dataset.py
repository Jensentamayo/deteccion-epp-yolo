import shutil
from pathlib import Path

SOURCE = Path("data/processed/epp_combined")
DEST = Path("data/processed/epp_no_compliance")

# Clases originales que queremos conservar
# 1 = no_helmet
# 3 = no_gloves
CLASS_MAP = {
    1: 0,
    3: 1,
}

CLASS_NAMES = {
    0: "no_helmet",
    1: "no_gloves",
}

SPLITS = ["train", "valid", "test"]


def process_label_file(source_label: Path, destination_label: Path) -> bool:
    """
    Copia únicamente las anotaciones no_helmet y no_gloves,
    remapeando sus IDs originales a 0 y 1.

    Retorna True si el archivo contiene al menos una detección
    de las clases objetivo.
    """

    new_lines = []

    for line in source_label.read_text().splitlines():
        parts = line.split()

        if len(parts) < 5:
            continue

        try:
            original_class = int(parts[0])
        except ValueError:
            continue

        if original_class not in CLASS_MAP:
            continue

        new_class = CLASS_MAP[original_class]

        new_line = " ".join(
            [str(new_class)] + parts[1:5]
        )

        new_lines.append(new_line)

    destination_label.parent.mkdir(parents=True, exist_ok=True)
    destination_label.write_text(
        "\n".join(new_lines) + ("\n" if new_lines else "")
    )

    return bool(new_lines)


def main():
    if DEST.exists():
        raise RuntimeError(
            f"El destino ya existe: {DEST}\n"
            "No se sobrescribirá automáticamente."
        )

    total_images = 0
    total_target_images = 0
    total_no_helmet = 0
    total_no_gloves = 0

    for split in SPLITS:
        source_images = SOURCE / "images" / split
        source_labels = SOURCE / "labels" / split

        destination_images = DEST / "images" / split
        destination_labels = DEST / "labels" / split

        if not source_images.exists():
            raise FileNotFoundError(
                f"No existe el directorio de imágenes: {source_images}"
            )

        if not source_labels.exists():
            raise FileNotFoundError(
                f"No existe el directorio de labels: {source_labels}"
            )

        destination_images.mkdir(parents=True, exist_ok=True)
        destination_labels.mkdir(parents=True, exist_ok=True)

        split_images = 0
        split_target_images = 0
        split_no_helmet = 0
        split_no_gloves = 0

        for source_label in sorted(source_labels.glob("*.txt")):
            destination_label = destination_labels / source_label.name

            has_target = process_label_file(
                source_label,
                destination_label,
            )

            source_stem = source_label.stem

            matching_images = list(
                source_images.glob(f"{source_stem}.*")
            )

            if not matching_images:
                continue

            source_image = matching_images[0]
            destination_image = destination_images / source_image.name

            # Conservamos todas las imágenes.
            shutil.copy2(source_image, destination_image)

            split_images += 1

            if has_target:
                split_target_images += 1

            for line in destination_label.read_text().splitlines():
                parts = line.split()

                if not parts:
                    continue

                if parts[0] == "0":
                    split_no_helmet += 1
                elif parts[0] == "1":
                    split_no_gloves += 1

        total_images += split_images
        total_target_images += split_target_images
        total_no_helmet += split_no_helmet
        total_no_gloves += split_no_gloves

        print()
        print(f"===== {split.upper()} =====")
        print(f"Imágenes: {split_images}")
        print(f"Imágenes con clases objetivo: {split_target_images}")
        print(f"no_helmet: {split_no_helmet}")
        print(f"no_gloves: {split_no_gloves}")

    data_yaml = DEST / "data.yaml"

    data_yaml.write_text(
        """path: /home/windows_11/deteccion-epp-yolo/data/processed/epp_no_compliance
train: images/train
val: images/valid
test: images/test
nc: 2
names:
- no_helmet
- no_gloves
"""
    )

    print()
    print("=" * 50)
    print("DATASET CREADO")
    print("=" * 50)
    print(f"Ruta: {DEST}")
    print(f"Imágenes totales: {total_images}")
    print(f"Imágenes con objetivo: {total_target_images}")
    print(f"no_helmet total: {total_no_helmet}")
    print(f"no_gloves total: {total_no_gloves}")
    print()
    print("Clases:")
    print("0 -> no_helmet")
    print("1 -> no_gloves")
    print()
    print(f"YAML: {data_yaml}")


if __name__ == "__main__":
    main()
