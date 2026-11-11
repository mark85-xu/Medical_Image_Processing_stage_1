r"""Train and evaluate the spatially corrected attention ResNet-50.

Uses the existing data/metadata and data/patient_split_clean directories. The
script is compatible with the existing Python 3.8 / PyTorch 1.8 yolov5 env and
requires no WandB or torchmetrics. It saves checkpoints, histories, test
predictions, image- and patient-level metrics, figures, and a final report.

Example (PowerShell, from any directory):
    & 'C:\Users\lhx\miniconda3\envs\yolov5\python.exe' -u `
      'F:\Medical_Image_Processing_stage_1\Ultrasound_project\scripts\train_attention_resnet50_spatial_fixed.py'

Run --preflight-only to validate data and one CUDA optimization step. A
partially finished normal run resumes automatically from last_checkpoint.pth.
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MANIFEST = DATA / "metadata" / "image_manifest.csv"
STATS = DATA / "metadata" / "normalization.json"
SPLIT = DATA / "patient_split_clean"
CLASS_TO_IDX = {"HCC": 0, "Hemangioma": 1}
SPLIT_NAMES = ("train", "val", "test")
EXPECTED_IMAGES = {"train": 1928, "val": 474, "test": 254}
EXPECTED_PATIENTS = {"train": 521, "val": 149, "test": 74}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--workers", type=int, default=0,
                        help="0 is the safest default on Windows")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--output", default="runs/attention_resnet50_spatial_fixed_patient_clean_seed24")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience, args.bootstrap) < 1 or args.workers < 0:
        parser.error("epochs, batch-size, patience, bootstrap must be positive; workers >= 0")
    return args


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metrics(labels, probabilities):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positive = (labels == 0).astype(np.int64)  # HCC is ImageFolder class 0.
    predicted_positive = (probabilities >= 0.5).astype(np.int64)
    predicted_label = np.where(predicted_positive == 1, 0, 1)
    matrix = confusion_matrix(labels, predicted_label, labels=[0, 1])
    tn = int(matrix[1, 1])  # True Hemangioma, predicted Hemangioma.
    fp = int(matrix[1, 0])  # True Hemangioma, predicted HCC.
    result = {
        "n_images_or_patients": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predicted_label)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted_label)),
        "hcc_precision": float(precision_score(positive, predicted_positive, zero_division=0)),
        "hcc_sensitivity_recall": float(recall_score(positive, predicted_positive, zero_division=0)),
        "hemangioma_specificity": float(tn / (tn + fp)) if tn + fp else None,
        "hcc_f1": float(f1_score(positive, predicted_positive, zero_division=0)),
        "macro_f1": float(f1_score(labels, predicted_label, average="macro", zero_division=0)),
        "brier_hcc": float(np.mean((probabilities - positive) ** 2)),
        "roc_auc_hcc": float(roc_auc_score(positive, probabilities))
        if len(np.unique(positive)) == 2 else None,
        "confusion_matrix_rows_true_cols_pred_HCC_Hemangioma": matrix.tolist(),
    }
    return result


def read_manifest_and_validate():
    if not MANIFEST.is_file() or not STATS.is_file():
        raise FileNotFoundError("Data manifest or normalization statistics missing")
    with MANIFEST.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 5385:
        raise ValueError("Expected 5,385 original-image manifest rows")
    by_name = {}
    patients = defaultdict(set)
    expected_names = defaultdict(set)
    for row in rows:
        name = row["image_filename"]
        if name in by_name:
            raise ValueError("Duplicate filename in manifest: " + name)
        by_name[name] = row
        split = row["new_clean_split"]
        if split:
            if split not in SPLIT_NAMES or row["version"] != "Clean":
                raise ValueError("Invalid new Clean assignment: " + name)
            patients[split].add((row["class_label"], row["patient_id"]))
            expected_names[(split, row["class_label"])].add(name)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if patients[left] & patients[right]:
            raise ValueError("Patient leakage between " + left + " and " + right)
    for split in SPLIT_NAMES:
        n_images = sum(len(expected_names[(split, cls)]) for cls in CLASS_TO_IDX)
        if n_images != EXPECTED_IMAGES[split] or len(patients[split]) != EXPECTED_PATIENTS[split]:
            raise ValueError("Unexpected image/patient count in " + split)
        for cls in CLASS_TO_IDX:
            actual = {p.name for p in (SPLIT / split / cls).iterdir() if p.is_file()}
            if actual != expected_names[(split, cls)]:
                raise ValueError("Folder and manifest differ in " + split + "/" + cls)
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    if stats["source_images"] != EXPECTED_IMAGES["train"] or stats["resize"] != [224, 224]:
        raise ValueError("Normalization statistics do not match this training split")
    if any(not math.isfinite(x) for x in stats["mean"] + stats["std"]):
        raise ValueError("Nonfinite normalization values")
    print("DATA OK: 1,928 train / 474 val / 254 test images; patients disjoint", flush=True)
    return by_name, stats


def make_transforms(stats):
    common = [transforms.Resize((224, 224))]
    train = transforms.Compose(common + [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(stats["mean"], stats["std"]),
    ])
    evaluate = transforms.Compose(common + [
        transforms.ToTensor(), transforms.Normalize(stats["mean"], stats["std"]),
    ])
    return train, evaluate


class TestDataset(Dataset):
    def __init__(self, folder, transform, by_name):
        self.images = datasets.ImageFolder(str(folder), transform=transform)
        self.by_name = by_name

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, label = self.images[index]
        path = Path(self.images.samples[index][0])
        row = self.by_name[path.name]
        return image, label, path.name, row["patient_id"]


def model_attention_resnet50_spatial_fixed():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from model import CustomModelSpatialFixed
    return CustomModelSpatialFixed(num_classes=2).cuda()


def probe_batch_size(train_set, requested):
    def probe_once(batch_size):
        loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)
        images, labels = next(iter(loader))
        model = model_attention_resnet50_spatial_fixed().train()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
        optimizer.zero_grad()
        output = model(images.cuda(non_blocking=True))
        loss = nn.CrossEntropyLoss()(output, labels.cuda(non_blocking=True))
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        value = float(loss.item())
        if not math.isfinite(value):
            raise ValueError("Nonfinite loss in GPU preflight")
        return value

    candidates = list(dict.fromkeys([requested] + [min(requested, n) for n in (16, 8, 4, 2)]))
    for batch_size in candidates:
        try:
            value = probe_once(batch_size)
            gc.collect()
            torch.cuda.empty_cache()
            print("GPU PREFLIGHT OK: batch size {}, loss {:.4f}".format(batch_size, value), flush=True)
            return batch_size
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            print("GPU out of memory at batch size {}; trying smaller".format(batch_size), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    raise RuntimeError("Even batch size 2 did not fit GPU memory")


def make_loaders(train_set, val_set, test_set, batch_size, workers):
    options = {"batch_size": batch_size, "num_workers": workers, "pin_memory": True}
    return (
        DataLoader(train_set, shuffle=True, **options),
        DataLoader(val_set, shuffle=False, **options),
        DataLoader(test_set, shuffle=False, **options),
    )


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels_all, prob_all = [], []
    for images, labels in loader:
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            output = model(images)
            loss = criterion(output, labels)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss detected during training or validation")
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        labels_all.extend(labels.cpu().numpy().tolist())
        prob_all.extend(torch.softmax(output.detach(), dim=1)[:, 0].cpu().numpy().tolist())
    result = metrics(labels_all, prob_all)
    result["loss"] = total_loss / len(loader.dataset)
    return result


def save_checkpoint(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(path))


def restore_rng(checkpoint):
    random.setstate(checkpoint["python_rng"])
    np.random.set_state(checkpoint["numpy_rng"])
    torch.set_rng_state(checkpoint["torch_rng"])
    torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])


def evaluate_test(model, loader):
    model.eval()
    records = []
    with torch.no_grad():
        for images, labels, filenames, patient_ids in loader:
            output = model(images.cuda(non_blocking=True))
            probabilities = torch.softmax(output, dim=1)[:, 0].cpu().numpy()
            for label, name, patient_id, probability in zip(labels.tolist(), filenames,
                                                            patient_ids, probabilities.tolist()):
                records.append({
                    "image_filename": name, "patient_id": patient_id,
                    "true_label": int(label), "true_class": "HCC" if label == 0 else "Hemangioma",
                    "prob_hcc": float(probability),
                    "pred_label": 0 if probability >= 0.5 else 1,
                    "pred_class": "HCC" if probability >= 0.5 else "Hemangioma",
                })
    return pd.DataFrame(records)


def patient_predictions(images):
    grouped = images.groupby("patient_id", sort=True)
    if grouped["true_label"].nunique().max() != 1:
        raise ValueError("Inconsistent patient class labels")
    result = grouped.agg(true_label=("true_label", "first"), prob_hcc=("prob_hcc", "mean"),
                         n_images=("image_filename", "count")).reset_index()
    result["true_class"] = np.where(result.true_label == 0, "HCC", "Hemangioma")
    result["pred_label"] = np.where(result.prob_hcc >= 0.5, 0, 1)
    result["pred_class"] = np.where(result.pred_label == 0, "HCC", "Hemangioma")
    return result


def bootstrap_patient_ci(patients, n_bootstrap, seed):
    generator = np.random.RandomState(seed)
    labels = patients.true_label.to_numpy()
    probabilities = patients.prob_hcc.to_numpy()
    keys = ("accuracy", "hcc_precision", "hcc_sensitivity_recall",
            "hemangioma_specificity", "hcc_f1", "roc_auc_hcc")
    draws = {key: [] for key in keys}
    for _ in range(n_bootstrap):
        indices = generator.randint(0, len(patients), size=len(patients))
        if len(np.unique(labels[indices])) < 2:
            continue
        values = metrics(labels[indices], probabilities[indices])
        for key in keys:
            if values[key] is not None:
                draws[key].append(values[key])
    return {key: {"lower_95": float(np.percentile(values, 2.5)),
                  "upper_95": float(np.percentile(values, 97.5)),
                  "valid_resamples": len(values)} if values else None
            for key, values in draws.items()}


def plot_history(history, output):
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in ((axes[0], "loss", "Cross-entropy loss"),
                           (axes[1], "accuracy", "Image accuracy")):
        ax.plot(frame.epoch, frame["train_" + key], label="train")
        ax.plot(frame.epoch, frame["val_" + key], label="validation")
        ax.set(title=title, xlabel="Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(str(output), dpi=160)
    plt.close(fig)


def plot_confusion(values, output, title):
    matrix = np.asarray(values["confusion_matrix_rows_true_cols_pred_HCC_Hemangioma"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["HCC", "Hemangioma"],
           yticklabels=["HCC", "Hemangioma"], xlabel="Predicted", ylabel="True", title=title)
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > matrix.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(str(output), dpi=160)
    plt.close(fig)


def plot_roc(labels, probabilities, output, title):
    positive = (np.asarray(labels) == 0).astype(int)
    false_positive_rate, true_positive_rate, _ = roc_curve(positive, probabilities)
    auc = roc_auc_score(positive, probabilities)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(false_positive_rate, true_positive_rate, label="AUC = {:.3f}".format(auc))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(output), dpi=160)
    plt.close(fig)


def save_report(output, config, history, image_values, patient_values, sensitivity, ci):
    lines = [
        "# SMC-LUD spatially corrected attention ResNet-50, patient-disjoint Clean data", "",
        "- Model: model.py CustomModelSpatialFixed, ImageNet-pretrained ResNet-50 backbone, two 7x7 attention branches, spatial attention, two classes",
        "- Original CustomModel is preserved; this model applies channel weights to 7x7 features before spatial attention.",
        "- The ordinary ResNet-50 baseline was randomly initialized, so differences are not a pure attention ablation.",
        "- Data: Clean images, new patient-disjoint split, seed 24", 
        "- Selection: minimum validation loss; test evaluated after selection", 
        "- Positive class: HCC (ImageFolder index 0), threshold 0.5", 
        "- Patient prediction: arithmetic mean of image-level HCC probabilities", 
        "- This split differs from the published image allocation; results are not a direct replication of Table 3.",
        "- Hem_87 lacks a patient-table row; the main test retains its folder label and sensitivity excludes it.",
        "", "## Run", "",
        "- Epochs completed: {}".format(len(history)),
        "- Best epoch: {}".format(min(history, key=lambda row: row["val_loss"])["epoch"]),
        "- Actual batch size: {}".format(config["actual_batch_size"]),
        "", "## Test results", "",
        "| Metric | Image level | Patient level |", "|---|---:|---:|",
    ]
    for key in ("accuracy", "balanced_accuracy", "hcc_precision", "hcc_sensitivity_recall",
                "hemangioma_specificity", "hcc_f1", "macro_f1", "roc_auc_hcc", "brier_hcc"):
        lines.append("| {} | {:.4f} | {:.4f} |".format(key, image_values[key], patient_values[key]))
    lines += ["", "## Hem_87 sensitivity analysis", "",
              "Excluding 5 Clean images from one test patient:", "",
              "- Image accuracy: {:.4f}".format(sensitivity["image"]["accuracy"]),
              "- Patient accuracy: {:.4f}".format(sensitivity["patient"]["accuracy"]),
              "", "## Patient bootstrap intervals", "",
              "Patients resampled with replacement; {} replicates requested.".format(config["bootstrap_replicates"]), ""]
    for key, value in ci.items():
        if value:
            lines.append("- {}: [{:.4f}, {:.4f}] ({} valid resamples)".format(
                key, value["lower_95"], value["upper_95"], value["valid_resamples"]))
    lines += ["", "See run_config.json, history.csv, predictions CSVs, metrics JSONs, and figures for details."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run with the yolov5 GPU Python interpreter")
    by_name, stats = read_manifest_and_validate()
    set_seed(args.seed)
    train_transform, eval_transform = make_transforms(stats)
    train_set = datasets.ImageFolder(str(SPLIT / "train"), transform=train_transform)
    val_set = datasets.ImageFolder(str(SPLIT / "val"), transform=eval_transform)
    test_set = TestDataset(SPLIT / "test", eval_transform, by_name)
    for dataset in (train_set, val_set, test_set.images):
        if dataset.class_to_idx != CLASS_TO_IDX:
            raise ValueError("Unexpected ImageFolder class mapping: " + str(dataset.class_to_idx))
    actual_batch_size = probe_batch_size(train_set, args.batch_size)
    if args.preflight_only:
        print("PREFLIGHT COMPLETE: no formal training or test evaluation", flush=True)
        return

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    completed = output / "report.md"
    config = {
        "model": "CustomModelSpatialFixed", "architecture_source": "model.py",
        "spatial_fix": "channel weights multiplied into both 7x7 branch feature maps",
        "backbone": "resnet50", "pretrained": True, "num_classes": 2,
        "class_to_idx": CLASS_TO_IDX, "positive_class": "HCC", "threshold": 0.5,
        "patient_aggregation": "mean HCC probability", "seed": args.seed,
        "max_epochs": args.epochs, "requested_batch_size": args.batch_size,
        "actual_batch_size": actual_batch_size, "patience": args.patience,
        "optimizer": "Adam", "learning_rate": 3e-5, "loss": "CrossEntropyLoss",
        "workers": args.workers, "bootstrap_replicates": args.bootstrap,
        "augmentation": {"horizontal_flip_p": 0.5, "vertical_flip_p": 0.5,
                         "rotation_degrees": 10, "brightness": 0.1, "contrast": 0.1},
        "normalization": stats, "manifest_sha256": sha256(MANIFEST),
        "python": sys.version.split()[0], "torch": torch.__version__,
        "torchvision": torchvision.__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "gpu": torch.cuda.get_device_name(0), "torch_cuda": torch.version.cuda,
    }
    config_path = output / "run_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old != config:
            raise ValueError("Existing output has a different configuration; use another --output")
    else:
        write_json(config_path, config)
    if completed.exists():
        print("Run already complete: " + str(completed), flush=True)
        return

    set_seed(args.seed)  # Preflight must not affect the formal run.
    train_loader, val_loader, test_loader = make_loaders(
        train_set, val_set, test_set, actual_batch_size, args.workers)
    model = model_attention_resnet50_spatial_fixed()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    criterion = nn.CrossEntropyLoss()
    last_path = output / "last_checkpoint.pth"
    best_path = output / "best_model.pth"
    history = []
    best_loss = float("inf")
    bad_epochs = 0
    start_epoch = 1
    if last_path.exists():
        checkpoint = torch.load(str(last_path), map_location="cuda")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        restore_rng(checkpoint)
        history = checkpoint["history"]
        best_loss = checkpoint["best_loss"]
        bad_epochs = checkpoint["bad_epochs"]
        start_epoch = checkpoint["epoch"] + 1
        print("RESUMING from epoch {}".format(start_epoch), flush=True)

    training_complete = output / "training_complete.json"
    if not training_complete.exists():
        for epoch in range(start_epoch, args.epochs + 1):
            train_values = run_epoch(model, train_loader, criterion, optimizer)
            val_values = run_epoch(model, val_loader, criterion)
            row = {"epoch": epoch}
            row.update({"train_" + key: value for key, value in train_values.items()
                        if not key.startswith("confusion_matrix")})
            row.update({"val_" + key: value for key, value in val_values.items()
                        if not key.startswith("confusion_matrix")})
            history.append(row)
            if val_values["loss"] < best_loss:
                best_loss = val_values["loss"]
                bad_epochs = 0
                save_checkpoint(best_path, {"epoch": epoch, "model_state": model.state_dict(),
                                            "validation_loss": best_loss})
            else:
                bad_epochs += 1
            checkpoint = {
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "history": history,
                "best_loss": best_loss, "bad_epochs": bad_epochs,
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
            }
            save_checkpoint(last_path, checkpoint)
            pd.DataFrame(history).to_csv(str(output / "history.csv"), index=False)
            print("EPOCH {}/{} train_loss={:.4f} val_loss={:.4f} val_acc={:.4f} best={:.4f} patience={}/{}".format(
                epoch, args.epochs, train_values["loss"], val_values["loss"], val_values["accuracy"],
                best_loss, bad_epochs, args.patience), flush=True)
            if bad_epochs >= args.patience:
                print("EARLY STOP at epoch {}".format(epoch), flush=True)
                break
        if not history or not best_path.exists():
            raise RuntimeError("Training ended without a best checkpoint")
        write_json(training_complete, {"epochs_completed": len(history),
                                       "best_epoch": min(history, key=lambda x: x["val_loss"])["epoch"],
                                       "finished_at": datetime.now().isoformat()})
    else:
        print("TRAINING COMPLETE: proceeding to final test", flush=True)

    best = torch.load(str(best_path), map_location="cuda")
    model.load_state_dict(best["model_state"])
    print("FINAL TEST: best validation epoch {}".format(best["epoch"]), flush=True)
    images = evaluate_test(model, test_loader)
    if len(images) != EXPECTED_IMAGES["test"]:
        raise ValueError("Test prediction count mismatch")
    patients = patient_predictions(images)
    if len(patients) != EXPECTED_PATIENTS["test"]:
        raise ValueError("Test patient count mismatch")
    image_values = metrics(images.true_label, images.prob_hcc)
    patient_values = metrics(patients.true_label, patients.prob_hcc)
    without_87_images = images[images.patient_id != "Hem_87"].copy()
    without_87_patients = patients[patients.patient_id != "Hem_87"].copy()
    if len(images) - len(without_87_images) != 5 or len(patients) - len(without_87_patients) != 1:
        raise ValueError("Hem_87 sensitivity subset differs from the audit")
    sensitivity = {
        "excluded_patient": "Hem_87", "excluded_images": 5,
        "image": metrics(without_87_images.true_label, without_87_images.prob_hcc),
        "patient": metrics(without_87_patients.true_label, without_87_patients.prob_hcc),
    }
    ci = bootstrap_patient_ci(patients, args.bootstrap, args.seed + 2401)
    images.to_csv(str(output / "test_predictions.csv"), index=False)
    patients.to_csv(str(output / "patient_predictions.csv"), index=False)
    write_json(output / "image_metrics.json", image_values)
    write_json(output / "patient_metrics.json", patient_values)
    write_json(output / "hem87_sensitivity.json", sensitivity)
    write_json(output / "patient_bootstrap_ci.json", ci)
    plot_history(history, output / "training_curves.png")
    plot_confusion(image_values, output / "image_confusion.png", "Image-level test")
    plot_confusion(patient_values, output / "patient_confusion.png", "Patient-level test")
    plot_roc(images.true_label, images.prob_hcc, output / "image_roc.png", "Image-level HCC ROC")
    plot_roc(patients.true_label, patients.prob_hcc, output / "patient_roc.png", "Patient-level HCC ROC")
    save_report(output, config, history, image_values, patient_values, sensitivity, ci)
    print("COMPLETE: " + str(output / "report.md"), flush=True)
    print("IMAGE METRICS " + json.dumps(image_values, ensure_ascii=False), flush=True)
    print("PATIENT METRICS " + json.dumps(patient_values, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
