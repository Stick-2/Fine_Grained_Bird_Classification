from pathlib import Path

import pandas as pd

import config


def _read_pairs(path):
    pairs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(" ", 1)
            pairs[int(key)] = value.strip()
    return pairs


def _build_dataframe(images, splits, labels):
    rows = []
    for image_id, image_name in images.items():
        rows.append(
            {
                "image": image_name,
                # Normalize labels to 0-based indices for CrossEntropyLoss.
                "label": int(labels[image_id]) - 1,
                "split": "train" if int(splits[image_id]) == 1 else "test",
            }
        )
    return pd.DataFrame(rows)


def main():
    root = Path(config.DATA_ROOT)
    images = _read_pairs(root / "images.txt")
    splits = _read_pairs(root / "train_test_split.txt")

    species_labels = _read_pairs(root / "image_class_labels.txt")
    df_species = _build_dataframe(images, splits, species_labels)
    df_species.to_csv(config.LABELS_SPECIES, index=False)

    family_labels = _read_pairs(root / "image_family_labels.txt")
    df_family = _build_dataframe(images, splits, family_labels)
    df_family.to_csv(config.LABELS_FAMILY, index=False)

    print(f"Saved: {config.LABELS_SPECIES} ({len(df_species)} rows)")
    print(f"Saved: {config.LABELS_FAMILY} ({len(df_family)} rows)")


if __name__ == "__main__":
    main()
