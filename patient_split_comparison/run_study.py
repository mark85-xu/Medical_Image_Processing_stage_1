"""Run 10 paired patient-split comparisons of original and spatial-fixed CustomModel.

This script reads original Clean images by manifest and never changes existing
data, model definitions, checkpoints, or previous experiments. Re-running it
resumes incomplete model runs and skips completed ones.
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parent
OUTPUT = STUDY / "outputs"
SOURCE_MANIFEST = ROOT / "data" / "metadata" / "image_manifest.csv"
SPLIT_SEEDS = tuple(range(1001, 1011))
TRAIN_SEED = 24
CLASS_TO_IDX = {"HCC": 0, "Hemangioma": 1}
MODEL_NAMES = ("original", "spatial_fixed")
MAX_EPOCHS = 100
PATIENCE = 7
BATCH_SIZE = 32
LEARNING_RATE = 3e-5
BOOTSTRAP = 1000

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import train_attention_resnet50_spatial_fixed as base  # noqa: E402
from model import CustomModel, CustomModelSpatialFixed  # noqa: E402


def args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--seed", type=int, choices=SPLIT_SEEDS,
                        help="Run just one predeclared patient split")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def read_clean_source():
    with SOURCE_MANIFEST.open(encoding="utf-8", newline="") as stream:
        records = [r for r in csv.DictReader(stream) if r["version"] == "Clean"]
    if len(records) != 2656 or len({r["image_filename"] for r in records}) != 2656:
        raise ValueError("Expected 2,656 unique Clean images")
    patients = defaultdict(set)
    for row in records:
        if row["class_label"] not in CLASS_TO_IDX or not (ROOT / row["source_path"]).is_file():
            raise ValueError("Invalid Clean source row: " + str(row))
        patients[row["class_label"]].add(row["patient_id"])
    if {key: len(value) for key, value in patients.items()} != {"HCC": 421, "Hemangioma": 323}:
        raise ValueError("Unexpected Clean patient counts")
    audit = json.loads((ROOT / "data" / "metadata" / "audit.json").read_text(encoding="utf-8"))
    if audit["duplicate_content_cross_patient_groups"] != 0:
        raise ValueError("Duplicate image content across patients")
    return records


def prepare_seed(seed, records, source_hash):
    folder = OUTPUT / "splits" / ("seed_" + str(seed))
    assignment_path = folder / "image_assignments.csv"
    stats_path = folder / "normalization.json"
    metadata_path = folder / "split_config.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["split_seed"] != seed or metadata["source_manifest_sha256"] != source_hash:
            raise ValueError("Existing split metadata differs: " + str(folder))
        rows = list(csv.DictReader(assignment_path.open(encoding="utf-8", newline="")))
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        validate_split(rows)
        if sha256(assignment_path) != metadata["assignment_sha256"]:
            raise ValueError("Assignment file changed: " + str(assignment_path))
        return rows, stats, metadata

    patient_to_split = {}
    patient_rows = []
    for class_name in CLASS_TO_IDX:
        ids = sorted({r["patient_id"] for r in records if r["class_label"] == class_name})
        random.Random(seed + CLASS_TO_IDX[class_name]).shuffle(ids)
        n_train = round(len(ids) * .70)
        n_val = round(len(ids) * .20)
        for split, group in (("train", ids[:n_train]),
                             ("val", ids[n_train:n_train + n_val]),
                             ("test", ids[n_train + n_val:])):
            for patient_id in group:
                patient_to_split[patient_id] = split
                patient_rows.append({"patient_id": patient_id, "class_label": class_name,
                                     "split": split})
    rows = [{"image_filename": r["image_filename"], "patient_id": r["patient_id"],
             "class_label": r["class_label"], "source_path": r["source_path"],
             "sha256": r["sha256"], "split": patient_to_split[r["patient_id"]]}
            for r in sorted(records, key=lambda x: (x["class_label"], x["image_filename"]))]
    validate_split(rows)
    write_csv(assignment_path, rows, list(rows[0]))
    write_csv(folder / "patient_assignments.csv", patient_rows, list(patient_rows[0]))
    stats = calculate_train_stats(rows)
    write_json(stats_path, stats)
    metadata = {
        "split_seed": seed, "training_seed": TRAIN_SEED,
        "source_manifest_sha256": source_hash,
        "assignment_sha256": sha256(assignment_path),
        "patient_counts": dict(Counter(r["split"] for r in patient_rows)),
        "image_counts": dict(Counter(r["split"] for r in rows)),
        "class_image_counts": dict(Counter(r["split"] + "/" + r["class_label"] for r in rows)),
        "normalization_sha256": sha256(stats_path),
    }
    write_json(metadata_path, metadata)
    return rows, stats, metadata


def validate_split(rows):
    if len(rows) != 2656 or len({r["image_filename"] for r in rows}) != len(rows):
        raise ValueError("Split does not contain every unique Clean image")
    patient_splits = defaultdict(set)
    class_splits = defaultdict(set)
    for row in rows:
        if row["split"] not in ("train", "val", "test"):
            raise ValueError("Unexpected split")
        patient_splits[row["patient_id"]].add(row["split"])
        class_splits[row["class_label"]].add(row["split"])
    if any(len(value) != 1 for value in patient_splits.values()):
        raise ValueError("Patient leakage within one split seed")
    if set(patient_splits) != set(r["patient_id"] for r in rows):
        raise ValueError("Patient assignment incomplete")
    if any(value != {"train", "val", "test"} for value in class_splits.values()):
        raise ValueError("A class is absent from one partition")


def calculate_train_stats(rows):
    sums = np.zeros(3, dtype=np.float64)
    squares = np.zeros(3, dtype=np.float64)
    pixels = 0
    train = [r for r in rows if r["split"] == "train"]
    for row in train:
        with Image.open(str(ROOT / row["source_path"])) as image:
            array = np.asarray(image.convert("RGB").resize((224, 224),
                               resample=Image.BILINEAR), dtype=np.float64) / 255.0
        flat = array.reshape(-1, 3)
        sums += flat.sum(axis=0)
        squares += np.square(flat).sum(axis=0)
        pixels += len(flat)
    mean = sums / pixels
    std = np.sqrt(np.maximum(squares / pixels - mean ** 2, 0.0))
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid training-only normalization")
    return {"source_split": "this split seed's train patients", "source_images": len(train),
            "resize": [224, 224], "color_mode": "RGB", "pixel_scale": "0_to_1",
            "resize_interpolation": "Pillow BILINEAR", "mean": mean.tolist(), "std": std.tolist()}


class CleanDataset(Dataset):
    def __init__(self, rows, split, transform, test=False):
        self.rows = sorted((r for r in rows if r["split"] == split),
                           key=lambda r: (CLASS_TO_IDX[r["class_label"]], r["image_filename"]))
        self.transform = transform
        self.test = test
        if not self.rows:
            raise ValueError("Empty dataset: " + split)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(str(ROOT / row["source_path"])) as image:
            tensor = self.transform(image.convert("RGB"))
        label = CLASS_TO_IDX[row["class_label"]]
        if self.test:
            return tensor, label, row["image_filename"], row["patient_id"]
        return tensor, label


def make_model(kind):
    cls = CustomModel if kind == "original" else CustomModelSpatialFixed
    return cls(num_classes=2).cuda()


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def verify_equal_initialization():
    digests = []
    for kind in MODEL_NAMES:
        base.set_seed(TRAIN_SEED)
        model = make_model(kind)
        digests.append(state_digest(model))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if digests[0] != digests[1]:
        raise ValueError("Original and fixed models did not start from identical weights")
    print("INITIAL WEIGHTS IDENTICAL: " + digests[0], flush=True)
    return digests[0]


def gpu_preflight(rows, stats):
    train_transform, _ = base.make_transforms(stats)
    dataset = CleanDataset(rows, "train", train_transform)
    images, labels = next(iter(DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)))
    for kind in MODEL_NAMES:
        base.set_seed(TRAIN_SEED)
        model = make_model(kind).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(model(images.cuda()), labels.cuda())
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite preflight loss for " + kind)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        print("GPU PREFLIGHT {}: batch={} loss={:.4f}".format(
            kind, BATCH_SIZE, loss.item()), flush=True)
        del model, optimizer, loss
        gc.collect()
        torch.cuda.empty_cache()


def run_model(seed, kind, rows, stats, split_metadata, expected_initial_hash):
    run_dir = OUTPUT / "runs" / ("seed_" + str(seed)) / kind
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "split_seed": seed, "training_seed": TRAIN_SEED, "model": kind,
        "split_assignment_sha256": split_metadata["assignment_sha256"],
        "normalization_sha256": split_metadata["normalization_sha256"],
        "source_manifest_sha256": split_metadata["source_manifest_sha256"],
        "initial_state_sha256": expected_initial_hash,
        "class_to_idx": CLASS_TO_IDX, "positive_class": "HCC", "threshold": .5,
        "patient_aggregation": "mean image HCC probability",
        "pretrained_backbone": "torchvision ResNet50 ImageNet",
        "optimizer": "Adam", "learning_rate": LEARNING_RATE, "loss": "CrossEntropyLoss",
        "max_epochs": MAX_EPOCHS, "early_stop_patience": PATIENCE,
        "batch_size": BATCH_SIZE, "bootstrap_replicates": BOOTSTRAP,
        "augmentation": {"horizontal_flip_p": .5, "vertical_flip_p": .5,
                         "rotation_degrees": 10, "brightness": .1, "contrast": .1},
        "normalization": stats, "python": sys.version.split()[0],
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "model_py_sha256": sha256(ROOT / "model.py"),
        "study_py_sha256": sha256(Path(__file__)),
    }
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Existing run has a different configuration: " + str(run_dir))
    else:
        write_json(config_path, config)
    if (run_dir / "report.md").exists():
        print("SKIP COMPLETE: {} {}".format(seed, kind), flush=True)
        return

    train_transform, eval_transform = base.make_transforms(stats)
    train_dataset = CleanDataset(rows, "train", train_transform)
    val_dataset = CleanDataset(rows, "val", eval_transform)
    test_dataset = CleanDataset(rows, "test", eval_transform, test=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)
    base.set_seed(TRAIN_SEED)
    model = make_model(kind)
    if state_digest(model) != expected_initial_hash:
        raise ValueError("Initial weights changed for " + kind)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    last_path = run_dir / "last_checkpoint.pth"
    best_path = run_dir / "best_model.pth"
    complete_path = run_dir / "training_complete.json"
    history, best_loss, bad_epochs, start_epoch = [], float("inf"), 0, 1
    if last_path.exists():
        checkpoint = torch.load(str(last_path), map_location="cuda")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        base.restore_rng(checkpoint)
        history = checkpoint["history"]
        best_loss = checkpoint["best_loss"]
        bad_epochs = checkpoint["bad_epochs"]
        start_epoch = checkpoint["epoch"] + 1
        print("RESUME {} {} from epoch {}".format(seed, kind, start_epoch), flush=True)
    if not complete_path.exists():
        for epoch in range(start_epoch, MAX_EPOCHS + 1):
            train_values = base.run_epoch(model, train_loader, criterion, optimizer)
            val_values = base.run_epoch(model, val_loader, criterion)
            entry = {"epoch": epoch}
            entry.update({"train_" + key: value for key, value in train_values.items()
                          if not key.startswith("confusion_matrix")})
            entry.update({"val_" + key: value for key, value in val_values.items()
                          if not key.startswith("confusion_matrix")})
            history.append(entry)
            if val_values["loss"] < best_loss:
                best_loss = val_values["loss"]
                bad_epochs = 0
                base.save_checkpoint(best_path, {"epoch": epoch,
                    "model_state": model.state_dict(), "validation_loss": best_loss})
            else:
                bad_epochs += 1
            base.save_checkpoint(last_path, {
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "history": history,
                "best_loss": best_loss, "bad_epochs": bad_epochs,
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
            })
            pd.DataFrame(history).to_csv(str(run_dir / "history.csv"), index=False)
            print("SEED {} {} EPOCH {}/{} train_loss={:.4f} val_loss={:.4f} val_acc={:.4f} patience={}/{}".format(
                seed, kind, epoch, MAX_EPOCHS, train_values["loss"], val_values["loss"],
                val_values["accuracy"], bad_epochs, PATIENCE), flush=True)
            if bad_epochs >= PATIENCE:
                break
        if not best_path.exists():
            raise RuntimeError("No best checkpoint for {} {}".format(seed, kind))
        write_json(complete_path, {"epochs_completed": len(history),
            "best_epoch": min(history, key=lambda r: r["val_loss"])["epoch"],
            "finished_at": datetime.now().isoformat()})

    best = torch.load(str(best_path), map_location="cuda")
    model.load_state_dict(best["model_state"])
    images = base.evaluate_test(model, test_loader)
    patients = base.patient_predictions(images)
    expected_images = sum(r["split"] == "test" for r in rows)
    expected_patients = len({r["patient_id"] for r in rows if r["split"] == "test"})
    if len(images) != expected_images or len(patients) != expected_patients:
        raise ValueError("Test count mismatch")
    image_metrics = base.metrics(images.true_label, images.prob_hcc)
    patient_metrics = base.metrics(patients.true_label, patients.prob_hcc)
    sensitivity = None
    if "Hem_87" in set(patients.patient_id):
        without_images = images[images.patient_id != "Hem_87"]
        without_patients = patients[patients.patient_id != "Hem_87"]
        sensitivity = {"excluded_patient": "Hem_87",
                       "excluded_images": len(images) - len(without_images),
                       "image": base.metrics(without_images.true_label, without_images.prob_hcc),
                       "patient": base.metrics(without_patients.true_label,
                                               without_patients.prob_hcc)}
    ci = base.bootstrap_patient_ci(patients, BOOTSTRAP, TRAIN_SEED + seed)
    images.to_csv(str(run_dir / "test_predictions.csv"), index=False)
    patients.to_csv(str(run_dir / "patient_predictions.csv"), index=False)
    write_json(run_dir / "image_metrics.json", image_metrics)
    write_json(run_dir / "patient_metrics.json", patient_metrics)
    write_json(run_dir / "hem87_sensitivity.json", sensitivity)
    write_json(run_dir / "patient_bootstrap_ci.json", ci)
    base.plot_history(history, run_dir / "training_curves.png")
    base.plot_confusion(image_metrics, run_dir / "image_confusion.png", "Image-level test")
    base.plot_confusion(patient_metrics, run_dir / "patient_confusion.png", "Patient-level test")
    base.plot_roc(images.true_label, images.prob_hcc, run_dir / "image_roc.png", "Image-level HCC ROC")
    base.plot_roc(patients.true_label, patients.prob_hcc, run_dir / "patient_roc.png", "Patient-level HCC ROC")
    lines = ["# Seed {} / {}".format(seed, kind), "",
             "Patient split shared with paired model; training seed {}.".format(TRAIN_SEED), "",
             "- Train images: {}; validation: {}; test: {}.".format(
                 len(train_dataset), len(val_dataset), len(test_dataset)),
             "- Best epoch: {}; epochs completed: {}.".format(best["epoch"], len(history)),
             "- Test image accuracy: {:.4f}; balanced accuracy: {:.4f}.".format(
                 image_metrics["accuracy"], image_metrics["balanced_accuracy"]),
             "- Test patient accuracy: {:.4f}; balanced accuracy: {:.4f}.".format(
                 patient_metrics["accuracy"], patient_metrics["balanced_accuracy"]),
             "- Test patient AUC: {:.4f}; Brier: {:.4f}.".format(
                 patient_metrics["roc_auc_hcc"], patient_metrics["brier_hcc"]), ""]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("RUN COMPLETE {} {}: image_acc={:.4f} patient_acc={:.4f} patient_bal_acc={:.4f}".format(
        seed, kind, image_metrics["accuracy"], patient_metrics["accuracy"],
        patient_metrics["balanced_accuracy"]), flush=True)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()


def paired_summary():
    rows = []
    for seed in SPLIT_SEEDS:
        pair = [OUTPUT / "runs" / ("seed_" + str(seed)) / kind for kind in MODEL_NAMES]
        if not all((folder / "report.md").exists() for folder in pair):
            continue
        configs = [json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
                   for folder in pair]
        if configs[0]["split_assignment_sha256"] != configs[1]["split_assignment_sha256"] or \
                configs[0]["normalization_sha256"] != configs[1]["normalization_sha256"] or \
                configs[0]["initial_state_sha256"] != configs[1]["initial_state_sha256"]:
            raise ValueError("Paired configurations differ at seed " + str(seed))
        predictions = [pd.read_csv(str(folder / "patient_predictions.csv")).set_index("patient_id")
                       for folder in pair]
        if not predictions[0]["true_label"].sort_index().equals(
                predictions[1]["true_label"].sort_index()):
            raise ValueError("Paired test patients or labels differ at seed " + str(seed))
        record = {"split_seed": seed, "n_test_patients": len(predictions[0])}
        for level in ("patient", "image"):
            values = [json.loads((folder / (level + "_metrics.json")).read_text(encoding="utf-8"))
                      for folder in pair]
            for key in ("accuracy", "balanced_accuracy", "hcc_sensitivity_recall",
                        "hemangioma_specificity", "roc_auc_hcc", "brier_hcc"):
                record["{}_{}_original".format(level, key)] = values[0][key]
                record["{}_{}_fixed".format(level, key)] = values[1][key]
                record["{}_{}_delta".format(level, key)] = values[1][key] - values[0][key]
        old, new = predictions
        old_correct = old.true_label == old.pred_label
        new_correct = new.true_label == new.pred_label
        record["patient_old_wrong_new_right"] = int((~old_correct & new_correct).sum())
        record["patient_old_right_new_wrong"] = int((old_correct & ~new_correct).sum())
        rows.append(record)
    summary_dir = OUTPUT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(str(summary_dir / "paired_results.csv"), index=False)
    lines = ["# 10 组患者划分：配对比较", "",
             "已完成 {}/10 组；仅将两模型均完成的划分纳入此表。".format(len(rows)), "",
             "主指标：患者级平衡准确率；差值 = 空间修正版 - 原 CustomModel。", "",
             "| 划分种子 | 测试患者 | 原版平衡准确率 | 修正版平衡准确率 | 差值 | 原错新对 | 原对新错 |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| {split_seed} | {n_test_patients} | {patient_balanced_accuracy_original:.4f} | {patient_balanced_accuracy_fixed:.4f} | {patient_balanced_accuracy_delta:+.4f} | {patient_old_wrong_new_right} | {patient_old_right_new_wrong} |".format(**row))
    if rows:
        delta = np.asarray([r["patient_balanced_accuracy_delta"] for r in rows])
        wins, ties, losses = int((delta > 0).sum()), int((delta == 0).sum()), int((delta < 0).sum())
        lines.extend(["", "- 患者平衡准确率差值均值：{:+.4f}；中位数：{:+.4f}。".format(
            float(delta.mean()), float(np.median(delta))),
            "- 修正版胜/平/负：{}/{}/{} 组。".format(wins, ties, losses)])
        if len(rows) == len(SPLIT_SEEDS):
            generator = np.random.RandomState(2401)
            resampled = [float(delta[generator.randint(0, len(delta), len(delta))].mean())
                         for _ in range(10000)]
            lines.append("- 对 10 组配对差值重采样的描述性 95% 区间：[{:+.4f}, {:+.4f}]。".format(
                *np.percentile(resampled, [2.5, 97.5])))
    lines.extend(["", "各划分共享同一来源患者；这些重复测试集并非独立外部队列。跨划分区间仅描述对划分变化的敏感程度。", ""])
    (summary_dir / "paired_report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(summary_dir / "progress.json", {"paired_splits_complete": len(rows),
        "total_planned": len(SPLIT_SEEDS), "updated_at": datetime.now().isoformat()})


def main():
    args = args_parser()
    OUTPUT.mkdir(exist_ok=True)
    records = read_clean_source()
    source_hash = sha256(SOURCE_MANIFEST)
    seeds = (args.seed,) if args.seed is not None else SPLIT_SEEDS
    prepared = {}
    for seed in seeds:
        rows, stats, metadata = prepare_seed(seed, records, source_hash)
        prepared[seed] = (rows, stats, metadata)
        print("PREPARED seed {}: {} images; patients {}; train mean {}".format(
            seed, metadata["image_counts"], metadata["patient_counts"],
            [round(v, 6) for v in stats["mean"]]), flush=True)
    if args.prepare_only:
        print("PREPARATION COMPLETE; no training started", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; use the existing yolov5 GPU environment")
    expected_hash = verify_equal_initialization()
    gpu_preflight(*prepared[seeds[0]][:2])
    for seed in seeds:
        rows, stats, metadata = prepared[seed]
        for kind in MODEL_NAMES:
            run_model(seed, kind, rows, stats, metadata, expected_hash)
        paired_summary()
    print("STUDY COMMAND COMPLETE: " + str(OUTPUT / "summary" / "paired_report.md"), flush=True)


if __name__ == "__main__":
    main()
