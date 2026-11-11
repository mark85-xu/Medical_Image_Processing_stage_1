"""Reproducible patient-disjoint sampling from the existing Clean test split."""

import csv
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .models import ROOT, SPECS


MANIFEST = ROOT / "data" / "metadata" / "image_manifest.csv"
CLASS_TO_IDX = {"HCC": 0, "Hemangioma": 1}


def test_records():
    with MANIFEST.open(encoding="utf-8", newline="") as stream:
        records = [row for row in csv.DictReader(stream)
                   if row["new_clean_split"] == "test" and row["version"] == "Clean"]
    if len(records) != 254:
        raise ValueError("Expected 254 Clean test images")
    by_name = {}
    for row in records:
        name = row["image_filename"]
        if name in by_name or row["class_label"] not in CLASS_TO_IDX:
            raise ValueError("Duplicate name or unexpected label: " + name)
        original = ROOT / row["source_path"]
        split_copy = ROOT / "data" / "patient_split_clean" / "test" / row["class_label"] / name
        if not original.is_file() or not split_copy.is_file():
            raise FileNotFoundError("Test image missing: " + name)
        by_name[name] = row
    return by_name


def random_samples(by_name, seed, per_class=8):
    rng = random.Random(seed)
    selected = []
    for class_name in ("HCC", "Hemangioma"):
        grouped = defaultdict(list)
        for row in by_name.values():
            if row["class_label"] == class_name:
                grouped[row["patient_id"]].append(row)
        patients = sorted(grouped)
        rng.shuffle(patients)
        if len(patients) < per_class:
            raise ValueError("Too few patients in " + class_name)
        for patient_id in patients[:per_class]:
            row = rng.choice(sorted(grouped[patient_id], key=lambda r: r["image_filename"]))
            selected.append(dict(row, selection_group="random_stratified",
                                 selection_reason="seeded one-image-per-patient sample"))
    return selected


def manual_sample(by_name, image):
    name = Path(image).name
    if name not in by_name:
        raise ValueError("Image must belong to the existing Clean test set: " + name)
    row = by_name[name]
    if Path(image).is_absolute() and Path(image).resolve() != (ROOT / row["source_path"]).resolve() and Path(image).resolve() != (ROOT / "data" / "patient_split_clean" / "test" / row["class_label"] / name).resolve():
        raise ValueError("Path does not match the audited test image")
    return [dict(row, selection_group="manual_test_image", selection_reason="user-specified test image")]


def error_cases(by_name, excluded_patients):
    tables = {}
    for spec in SPECS:
        path = ROOT / "runs" / spec.run_dir / "test_predictions.csv"
        table = pd.read_csv(str(path)).set_index("image_filename")
        if set(table.index) != set(by_name):
            raise ValueError("Saved predictions do not match the test manifest: " + spec.name)
        tables[spec.name] = table
    used = set(excluded_patients)
    selected = []
    categories = (
        ("fixed_correct_original_wrong", lambda c: c[3] and not c[2]),
        ("original_correct_fixed_wrong", lambda c: c[2] and not c[3]),
        ("some_models_wrong", lambda c: any(c) and not all(c)),
        ("all_models_correct", lambda c: all(c)),
    )
    for category, condition in categories:
        for name in sorted(by_name):
            row = by_name[name]
            if row["patient_id"] in used:
                continue
            correct = []
            for spec in SPECS:
                pred = tables[spec.name].loc[name]
                if pred["patient_id"] != row["patient_id"] or pred["true_class"] != row["class_label"]:
                    raise ValueError("Saved prediction metadata mismatch: " + name)
                correct.append(bool(pred["true_label"] == pred["pred_label"]))
            if condition(correct):
                selected.append(dict(row, selection_group="prediction_selected",
                                     selection_reason=category))
                used.add(row["patient_id"])
                break
    return selected
