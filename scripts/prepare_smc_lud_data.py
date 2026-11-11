"""Audit SMC-LUD and create a patient-disjoint Clean image split.

Run from anywhere with Python 3.8+:
    python scripts/prepare_smc_lud_data.py

The original and published split directories are read only. Outputs are placed
under data/metadata and data/patient_split_clean after all checks pass.
"""

import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ORIGINAL = DATA / "original_data"
SEED = 24
CLASSES = ("HCC", "Hemangioma")
LEGACY = {
    "clean_ver_for_train": ("train_clean", "val_clean", "test_clean"),
    "split_for_train": ("train", "val", "test"),
}
OUTPUTS = (DATA / "metadata", DATA / "patient_split_clean")


def relative(path):
    return path.relative_to(ROOT).as_posix()


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_ids(class_name):
    filename = "HCC_patient_table.csv" if class_name == "HCC" else "Hem_patient_table.csv"
    with (ORIGINAL / class_name / filename).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [row["Subject_id"].strip() for row in rows]
    if class_name == "HCC":
        ids = ["HCC_" + patient_id for patient_id in ids]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate patient IDs in " + filename)
    return set(ids)


def main():
    if any(path.exists() for path in OUTPUTS):
        raise FileExistsError("Generated data already exists; inspect it before rerunning")

    staging = DATA / ".patient_prep_staging"
    if staging.exists():
        raise FileExistsError("Staging directory exists: " + str(staging))
    meta_stage = staging / "metadata"
    split_stage = staging / "patient_split_clean"
    meta_stage.mkdir(parents=True)
    split_stage.mkdir()

    try:
        records = []
        by_filename = {}
        patients = defaultdict(set)
        hashes = defaultdict(list)
        for class_name, prefix in (("HCC", "HCC"), ("Hemangioma", "Hem")):
            for version, folder in (
                ("Clean", prefix + "_clean (wo caliper)"),
                ("Caliper", prefix + "_caliper"),
            ):
                base = ORIGINAL / class_name / folder
                for patient_dir in sorted(base.iterdir()):
                    if not patient_dir.is_dir():
                        raise ValueError("Unexpected file in " + str(base))
                    patient_id = patient_dir.name
                    patients[class_name].add(patient_id)
                    images = sorted(path for path in patient_dir.iterdir() if path.is_file())
                    if not images:
                        raise ValueError("Empty patient folder: " + str(patient_dir))
                    for path in images:
                        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                            raise ValueError("Unexpected image type: " + str(path))
                        if path.name in by_filename:
                            raise ValueError("Duplicate image filename: " + path.name)
                        with Image.open(path) as image:
                            image.verify()
                        digest = file_hash(path)
                        record = {
                            "image_filename": path.name,
                            "patient_id": patient_id,
                            "class_label": class_name,
                            "version": version,
                            "source_path": relative(path),
                            "sha256": digest,
                            "legacy_clean_split": "",
                            "legacy_all_split": "",
                            "new_clean_split": "",
                            "metadata_status": "",
                        }
                        records.append(record)
                        by_filename[path.name] = record
                        hashes[digest].append(record)

        discrepancies = []
        for class_name in CLASSES:
            in_metadata = metadata_ids(class_name)
            in_images = patients[class_name]
            for patient_id in sorted(in_metadata - in_images):
                discrepancies.append({"class_label": class_name, "patient_id": patient_id,
                                      "issue": "metadata_without_images"})
            for patient_id in sorted(in_images - in_metadata):
                discrepancies.append({"class_label": class_name, "patient_id": patient_id,
                                      "issue": "images_without_metadata"})
            for record in records:
                if record["class_label"] == class_name:
                    record["metadata_status"] = (
                        "matched" if record["patient_id"] in in_metadata else "missing_in_metadata"
                    )

        legacy_stats = {}
        for folder, names in LEGACY.items():
            split_patients = defaultdict(set)
            split_counts = {}
            for split_name in names:
                for class_name in CLASSES:
                    files = sorted((DATA / folder / split_name / class_name).iterdir())
                    split_counts[split_name + "/" + class_name] = len(files)
                    for path in files:
                        if not path.is_file() or path.name not in by_filename:
                            raise ValueError("Legacy file missing from original: " + str(path))
                        record = by_filename[path.name]
                        if record["class_label"] != class_name:
                            raise ValueError("Legacy class mismatch: " + str(path))
                        if folder == "clean_ver_for_train" and record["version"] != "Clean":
                            raise ValueError("Caliper image in Clean folder: " + str(path))
                        field = "legacy_clean_split" if folder == "clean_ver_for_train" else "legacy_all_split"
                        if record[field]:
                            raise ValueError("Image occurs in two legacy splits: " + path.name)
                        record[field] = split_name
                        split_patients[(class_name, split_name)].add(record["patient_id"])
            overlaps = {}
            for class_name in CLASSES:
                for index, left in enumerate(names):
                    for right in names[index + 1:]:
                        overlaps[class_name + ":" + left + "_vs_" + right] = len(
                            split_patients[(class_name, left)] & split_patients[(class_name, right)]
                        )
            legacy_stats[folder] = {"image_counts": split_counts, "patient_overlap_counts": overlaps}

        if any(record["version"] == "Clean" and not record["legacy_clean_split"] for record in records):
            raise ValueError("A Clean image is absent from the published Clean split")
        if any(not record["legacy_all_split"] for record in records):
            raise ValueError("An original image is absent from the published all-image split")

        clean_split_patients = defaultdict(set)
        for class_name in CLASSES:
            clean_ids = sorted({record["patient_id"] for record in records
                                if record["class_label"] == class_name and record["version"] == "Clean"})
            random.Random(SEED + CLASSES.index(class_name)).shuffle(clean_ids)
            n_train = round(len(clean_ids) * 0.70)
            n_val = round(len(clean_ids) * 0.20)
            assignments = {
                patient_id: split for split, group in (
                    ("train", clean_ids[:n_train]),
                    ("val", clean_ids[n_train:n_train + n_val]),
                    ("test", clean_ids[n_train + n_val:]),
                ) for patient_id in group
            }
            for record in records:
                if record["class_label"] != class_name or record["version"] != "Clean":
                    continue
                split = assignments[record["patient_id"]]
                record["new_clean_split"] = split
                clean_split_patients[(split, class_name)].add(record["patient_id"])
                target = split_stage / split / class_name / record["image_filename"]
                target.parent.mkdir(parents=True, exist_ok=True)
                source = ROOT / record["source_path"]
                shutil.copy2(source, target)
                if target.stat().st_size != source.stat().st_size:
                    raise IOError("Copy size mismatch: " + str(target))

        for class_name in CLASSES:
            for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
                if clean_split_patients[(left, class_name)] & clean_split_patients[(right, class_name)]:
                    raise AssertionError("Patient overlap in generated split")

        image_fields = list(records[0])
        write_csv(meta_stage / "image_manifest.csv", image_fields,
                  sorted(records, key=lambda row: (row["class_label"], row["patient_id"], row["image_filename"])))
        write_csv(meta_stage / "metadata_discrepancies.csv",
                  ("class_label", "patient_id", "issue"), discrepancies)
        patient_rows = []
        for class_name in CLASSES:
            for patient_id in sorted(patients[class_name]):
                group = [record for record in records if record["class_label"] == class_name
                         and record["patient_id"] == patient_id]
                patient_rows.append({
                    "patient_id": patient_id,
                    "class_label": class_name,
                    "clean_images": sum(row["version"] == "Clean" for row in group),
                    "caliper_images": sum(row["version"] == "Caliper" for row in group),
                    "new_clean_split": next((row["new_clean_split"] for row in group if row["new_clean_split"]), ""),
                    "metadata_status": group[0]["metadata_status"],
                })
        write_csv(meta_stage / "patient_manifest.csv", list(patient_rows[0]), patient_rows)

        duplicate_content = [group for group in hashes.values() if len(group) > 1]
        audit = {
            "seed": SEED,
            "patient_split_ratios": {"train": 0.70, "val": 0.20, "test": 0.10},
            "source_image_counts": dict(sorted(Counter(row["class_label"] + "/" + row["version"]
                                                      for row in records).items())),
            "source_patient_counts": {class_name: len(patients[class_name]) for class_name in CLASSES},
            "metadata_discrepancy_counts": dict(Counter(row["issue"] for row in discrepancies)),
            "duplicate_content_groups": len(duplicate_content),
            "duplicate_content_cross_patient_groups": sum(
                len({row["patient_id"] for row in group}) > 1 for group in duplicate_content
            ),
            "legacy_splits": legacy_stats,
            "new_clean_split_image_counts": dict(sorted(Counter(
                row["new_clean_split"] + "/" + row["class_label"] for row in records
                if row["new_clean_split"]
            ).items())),
            "new_clean_split_patient_counts": {
                split + "/" + class_name: len(clean_split_patients[(split, class_name)])
                for split in ("train", "val", "test") for class_name in CLASSES
            },
        }
        (meta_stage / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")
        meta_stage.rename(OUTPUTS[0])
        split_stage.rename(OUTPUTS[1])
        staging.rmdir()
        print(json.dumps(audit, ensure_ascii=False, indent=2))
    except Exception:
        shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    main()
