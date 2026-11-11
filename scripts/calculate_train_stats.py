"""Compute RGB normalization statistics from the new Clean training split only."""

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "metadata" / "image_manifest.csv"
OUTPUT = ROOT / "data" / "metadata" / "normalization.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError("Statistics already exist; inspect before recalculating")
    with MANIFEST.open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["new_clean_split"] == "train"]
    if not rows or any(row["version"] != "Clean" for row in rows):
        raise ValueError("The new Clean training split is missing or contains Caliper images")

    sums = np.zeros(3, dtype=np.float64)
    sums_of_squares = np.zeros(3, dtype=np.float64)
    pixels = 0
    bilinear = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    for row in rows:
        source = ROOT / row["source_path"]
        with Image.open(source) as image:
            image = image.convert("RGB").resize((224, 224), resample=bilinear)
            array = np.asarray(image, dtype=np.float64) / 255.0
        flat = array.reshape(-1, 3)
        sums += flat.sum(axis=0)
        sums_of_squares += np.square(flat).sum(axis=0)
        pixels += len(flat)

    mean = sums / pixels
    std = np.sqrt(np.maximum(sums_of_squares / pixels - np.square(mean), 0.0))
    result = {
        "source_split": "data/patient_split_clean/train",
        "source_images": len(rows),
        "resize": [224, 224],
        "color_mode": "RGB",
        "pixel_scale": "0_to_1",
        "resize_interpolation": "Pillow BILINEAR",
        "aggregation": "population statistics over all pixels in the training images",
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
