"""Paired Grad-CAM for ten completed CustomModel patient-split experiments."""

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
sys.path.insert(0, str(ROOT))
from gradcam_analysis.cam import GradCAM
from gradcam_analysis.visualization import render_cam, save_png
from model import CustomModel, CustomModelSpatialFixed

PAIRED = ROOT / "patient_split_comparison" / "outputs"
OUT = STUDY / "outputs"
SEEDS = tuple(range(1001, 1011))
KINDS = ("original", "spatial_fixed")
CLASS_TO_IDX = {"HCC": 0, "Hemangioma": 1}
SAMPLE_SEED_BASE = 20261109
PER_CLASS = 8


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--smoke", action="store_true", help="One patient per class for one seed")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    a = p.parse_args()
    if a.smoke and a.seed is None:
        p.error("--smoke requires --seed")
    return a


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare_seed(seed, smoke):
    split_dir = PAIRED / "splits" / ("seed_" + str(seed))
    split_config = read_json(split_dir / "split_config.json")
    assignment_path = split_dir / "image_assignments.csv"
    if digest(assignment_path) != split_config["assignment_sha256"]:
        raise ValueError("Split assignment hash mismatch")
    if digest(split_dir / "normalization.json") != split_config["normalization_sha256"]:
        raise ValueError("Split normalization hash mismatch")
    if digest(ROOT / "data" / "metadata" / "image_manifest.csv") != split_config["source_manifest_sha256"]:
        raise ValueError("Source manifest hash mismatch")
    rows = [r for r in read_csv(assignment_path) if r["split"] == "test"]
    if len(rows) != split_config["image_counts"]["test"] or len({r["patient_id"] for r in rows}) != 74:
        raise ValueError("Unexpected test set size")
    if len({r["image_filename"] for r in rows}) != len(rows):
        raise ValueError("Duplicate test image name")
    by_patient = defaultdict(list)
    for r in rows:
        if r["class_label"] not in CLASS_TO_IDX:
            raise ValueError("Unexpected class")
        by_patient[(r["class_label"], r["patient_id"])].append(r)
    rng = random.Random(SAMPLE_SEED_BASE + seed)
    samples = []
    for label in ("HCC", "Hemangioma"):
        patients = sorted(pid for cls, pid in by_patient if cls == label)
        rng.shuffle(patients)
        for pid in patients[:(1 if smoke else PER_CLASS)]:
            r = rng.choice(sorted(by_patient[(label, pid)], key=lambda x: x["image_filename"]))
            samples.append(dict(r, selection_group="random_stratified", selection_reason="seeded_patient_sample"))
    refs = {}
    for kind in KINDS:
        folder = PAIRED / "runs" / ("seed_" + str(seed)) / kind
        config = read_json(folder / "run_config.json")
        if config["split_seed"] != seed or config["model"] != kind:
            raise ValueError("Wrong model run config")
        for key, value in (("split_assignment_sha256", split_config["assignment_sha256"]),
                           ("normalization_sha256", split_config["normalization_sha256"]),
                           ("source_manifest_sha256", split_config["source_manifest_sha256"]),
                           ("model_py_sha256", digest(ROOT / "model.py"))):
            if config[key] != value:
                raise ValueError("Run config {} mismatch for {}".format(key, kind))
        if config["class_to_idx"] != CLASS_TO_IDX or config["training_seed"] != 24:
            raise ValueError("Class map or training seed mismatch")
        table = {r["image_filename"]: r for r in read_csv(folder / "test_predictions.csv")}
        if set(table) != {r["image_filename"] for r in rows}:
            raise ValueError("Test predictions do not match split")
        refs[kind] = {"folder": folder, "config": config, "predictions": table,
                      "checkpoint_sha256": digest(folder / "best_model.pth")}
    if refs["original"]["config"]["normalization"] != refs["spatial_fixed"]["config"]["normalization"]:
        raise ValueError("Paired normalization differs")
    if refs["original"]["config"]["initial_state_sha256"] != refs["spatial_fixed"]["config"]["initial_state_sha256"]:
        raise ValueError("Paired initial states differ")
    if not smoke:
        selected_patients = {r["patient_id"] for r in samples}
        categories = (
            ("fixed_right_original_wrong", lambda a, b: not a and b),
            ("original_right_fixed_wrong", lambda a, b: a and not b),
            ("both_wrong", lambda a, b: not a and not b),
            ("both_right", lambda a, b: a and b),
        )
        for reason, predicate in categories:
            for r in sorted(rows, key=lambda x: x["image_filename"]):
                if r["patient_id"] in selected_patients:
                    continue
                name = r["image_filename"]
                correct = [int(refs[k]["predictions"][name]["pred_label"]) == CLASS_TO_IDX[r["class_label"]]
                           for k in KINDS]
                if predicate(*correct):
                    samples.append(dict(r, selection_group="prediction_selected", selection_reason=reason))
                    selected_patients.add(r["patient_id"])
                    break
    if len({r["patient_id"] for r in samples}) != len(samples):
        raise ValueError("Repeated patient within seed")
    return samples, refs, split_config


def load_pair(refs, device):
    loaded = {}
    for kind, cls in (("original", CustomModel), ("spatial_fixed", CustomModelSpatialFixed)):
        model = cls(2)
        ckpt = torch.load(str(refs[kind]["folder"] / "best_model.pth"), map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.to(device).eval()
        loaded[kind] = {"model": model, "layer": model.backbone[-1][-1],
                        "best_epoch": int(ckpt["epoch"])}
    return loaded


def low_resolution_cam(cam):
    feature = cam.activation.detach()
    gradient = cam.gradient.detach()
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    raw = F.relu((feature * weights).sum(dim=1))[0].cpu().numpy()
    if raw.shape != (7, 7) or not np.isfinite(raw).all() or raw.max() <= raw.min():
        raise ValueError("Invalid 7x7 CAM")
    return ((raw - raw.min()) / (raw.max() - raw.min())).astype(np.float32)


def cam_similarity(a, b):
    n = 10  # top 20% of 49 cells, rounded up; exclude zero-response cells
    ix_a = {int(i) for i in np.argsort(a.ravel(), kind="mergesort")[-n:] if a.ravel()[i] > 0}
    ix_b = {int(i) for i in np.argsort(b.ravel(), kind="mergesort")[-n:] if b.ravel()[i] > 0}
    iou = len(ix_a & ix_b) / float(len(ix_a | ix_b))
    grid_y, grid_x = np.indices((7, 7))
    def center(c):
        return np.array([(grid_y * c).sum() / c.sum(), (grid_x * c).sum() / c.sum()])
    distance = float(np.linalg.norm(center(a) - center(b)) / math.sqrt(6 * 6 + 6 * 6))
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    return {"top20_iou": iou, "centroid_distance": distance, "pearson_7x7": corr}


def draw_comparison(path, item, image, outputs):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), dpi=130)
    axes[0].imshow(image)
    axes[0].set_title("原图 / 真实 " + item["class_label"])
    for ax, kind, title in ((axes[1], "original", "原 CustomModel"),
                            (axes[2], "spatial_fixed", "空间修正版")):
        record = outputs[kind]
        ax.imshow(record["overlay"])
        ax.set_title("{}\n预测 {} ({}) P(HCC)={:.3f}".format(
            title, record["predicted_class"], "对" if record["correct"] else "错",
            record["prob_hcc"]))
    for ax in axes:
        ax.axis("off")
    fig.suptitle("患者 {} / {}".format(item["patient_id"], item["selection_reason"]))
    fig.tight_layout(rect=[0, 0, 1, .86])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)


def draw_attention(path, item, image, outputs):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), dpi=130)
    axes[0].imshow(image)
    axes[0].set_title("原图")
    old = outputs["original"]["attention"]
    fixed = outputs["spatial_fixed"]["attention"]
    axes[1].imshow(np.full((7, 7), float(old[0, 0])), cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("原版单个权重 1×1：{:.3f}".format(float(old[0, 0])))
    axes[2].imshow(fixed, cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title("修正版权重 7×7\n范围 {:.3f}–{:.3f}".format(fixed.min(), fixed.max()))
    for ax in axes:
        ax.axis("off")
    fig.suptitle("空间注意力权重（不是 Grad-CAM）/ " + item["patient_id"])
    fig.tight_layout(rect=[0, 0, 1, .86])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)


def draw_overview(path, paths):
    if not paths:
        return
    rendered = []
    for p in paths:
        with Image.open(str(p)) as image:
            rgb = image.convert("RGB")
            width = 1200
            height = round(rgb.height * width / rgb.width)
            rendered.append(rgb.resize((width, height), Image.BILINEAR))
    gap = 12
    canvas = Image.new("RGB", (1200, sum(i.height for i in rendered) + gap * (len(rendered) - 1)), "white")
    top = 0
    for image in rendered:
        canvas.paste(image, (0, top))
        top += image.height + gap
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(path))


def evaluate_image(seed, item, refs, loaded, device, folder):
    source = ROOT / item["source_path"]
    if digest(source) != item["sha256"]:
        raise ValueError("Image hash mismatch: " + str(source))
    with Image.open(str(source)) as im:
        rgb = im.convert("RGB").resize((224, 224), Image.BILINEAR)
    image = np.asarray(rgb)
    norm = refs["original"]["config"]["normalization"]
    tensor = transforms.Compose((transforms.ToTensor(),
                                 transforms.Normalize(norm["mean"], norm["std"])))(rgb).unsqueeze(0).to(device)
    target = CLASS_TO_IDX[item["class_label"]]
    outputs = {}
    records = []
    stem = Path(item["image_filename"]).stem
    for kind in KINDS:
        model = loaded[kind]["model"]
        mask_holder = {}
        def capture_mask(module, inputs, output):
            mask_holder["mask"] = output.detach().cpu().numpy()[0, 0].copy()
        handle = model.spatial_attention.sigmoid.register_forward_hook(capture_mask)
        try:
            with GradCAM(model, loaded[kind]["layer"]) as cam:
                result = cam.generate(tensor, target)
                low = low_resolution_cam(cam)
                mask = mask_holder["mask"]
                predicted = int(np.argmax(result["probabilities"]))
                pred_cam = None
                if predicted != target:
                    pred_cam = cam.generate(tensor, predicted)["cam"]
        finally:
            handle.remove()
        expected_shape = (1, 1) if kind == "original" else (7, 7)
        if mask.shape != expected_shape or result["activation_shape"] != [1, 2048, 7, 7]:
            raise ValueError("Attention or target-layer shape mismatch: " + kind)
        reference = refs[kind]["predictions"][item["image_filename"]]
        if reference["patient_id"] != item["patient_id"] or reference["true_class"] != item["class_label"]:
            raise ValueError("Saved prediction metadata mismatch")
        if predicted != int(reference["pred_label"]):
            raise ValueError("Prediction differs from saved test prediction")
        heatmap, overlay = render_cam(image, result["cam"])
        model_dir = folder / "individual" / kind
        heatmap_path = model_dir / (stem + "_gt_heatmap.png")
        overlay_path = model_dir / (stem + "_gt_overlay.png")
        low_path = model_dir / (stem + "_gt_7x7.npy")
        full_path = model_dir / (stem + "_gt_224.npy")
        save_png(heatmap_path, heatmap)
        save_png(overlay_path, overlay)
        np.save(str(low_path), low)
        np.save(str(full_path), result["cam"])
        pred_cam_path = ""
        pred_overlay_path = ""
        if pred_cam is not None:
            _, pred_overlay = render_cam(image, pred_cam)
            pred_cam_file = model_dir / (stem + "_pred_224.npy")
            pred_overlay_file = model_dir / (stem + "_pred_overlay.png")
            np.save(str(pred_cam_file), pred_cam)
            save_png(pred_overlay_file, pred_overlay)
            pred_cam_path = str(pred_cam_file.relative_to(OUT))
            pred_overlay_path = str(pred_overlay_file.relative_to(OUT))
        outputs[kind] = {"overlay": overlay, "attention": mask, "low": low,
                         "predicted_class": "HCC" if predicted == 0 else "Hemangioma",
                         "correct": predicted == target, "prob_hcc": float(result["probabilities"][0])}
        records.append({"seed": seed, "patient_id": item["patient_id"],
                        "image_filename": item["image_filename"], "class_label": item["class_label"],
                        "selection_group": item["selection_group"], "selection_reason": item["selection_reason"],
                        "model": kind, "predicted_class": outputs[kind]["predicted_class"],
                        "correct": predicted == target, "prob_hcc": outputs[kind]["prob_hcc"],
                        "saved_prob_hcc": float(reference["prob_hcc"]),
                        "prob_difference": outputs[kind]["prob_hcc"] - float(reference["prob_hcc"]),
                        "best_epoch": loaded[kind]["best_epoch"], "checkpoint_sha256": refs[kind]["checkpoint_sha256"],
                        "target_layer_shape": result["activation_shape"],
                        "gradient_l1": result["gradient_l1"], "cam_valid": True,
                        "attention_shape": list(mask.shape), "attention_min": float(mask.min()),
                        "attention_max": float(mask.max()), "attention_std": float(mask.std()),
                        "true_cam_224_path": str(full_path.relative_to(OUT)),
                        "true_cam_7x7_path": str(low_path.relative_to(OUT)),
                        "heatmap_path": str(heatmap_path.relative_to(OUT)),
                        "overlay_path": str(overlay_path.relative_to(OUT)),
                        "predicted_cam_path": pred_cam_path,
                        "predicted_overlay_path": pred_overlay_path})
    metrics = cam_similarity(outputs["original"]["low"], outputs["spatial_fixed"]["low"])
    comparison_path = folder / "comparisons" / item["selection_group"] / item["class_label"] / (stem + ".png")
    attention_path = folder / "attention_audit" / (stem + ".png")
    draw_comparison(comparison_path, item, image, outputs)
    draw_attention(attention_path, item, image, outputs)
    pair = {"seed": seed, "patient_id": item["patient_id"], "image_filename": item["image_filename"],
            "class_label": item["class_label"], "selection_group": item["selection_group"],
            "selection_reason": item["selection_reason"], "original_correct": outputs["original"]["correct"],
            "fixed_correct": outputs["spatial_fixed"]["correct"], **metrics,
            "comparison_path": str(comparison_path.relative_to(OUT)),
            "attention_path": str(attention_path.relative_to(OUT))}
    return pair, records


def run_seed(seed, smoke, device):
    samples, refs, split_config = prepare_seed(seed, smoke)
    folder = OUT / (("smoke_seed_" if smoke else "seed_") + str(seed))
    config = {"seed": seed, "smoke": smoke, "sample_seed": SAMPLE_SEED_BASE + seed,
              "per_class": 1 if smoke else PER_CLASS, "device": str(device),
              "split_assignment_sha256": split_config["assignment_sha256"],
              "normalization_sha256": split_config["normalization_sha256"],
              "checkpoints": {kind: refs[kind]["checkpoint_sha256"] for kind in KINDS},
              "cam_target": "true class raw logit", "target_layer": "backbone[-1][-1]",
              "cam_grid": [7, 7], "colormap": "jet", "alpha": 0.4}
    config_path = folder / "experiment_config.json"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError("Existing experiment config differs: " + str(folder))
    write_json(config_path, config)
    loaded = None
    pairs, predictions, failures = [], [], []
    try:
        for index, item in enumerate(samples, 1):
            stem = Path(item["image_filename"]).stem
            record_path = folder / "records" / (stem + ".json")
            try:
                cached = read_json(record_path) if record_path.exists() else None
                if cached is not None and all("true_cam_224_path" in r for r in cached["predictions"]):
                    pair, records = cached["pair"], cached["predictions"]
                    if not (folder / "comparisons" / item["selection_group"] / item["class_label"] / (stem + ".png")).is_file():
                        raise ValueError("Cached comparison missing")
                    old_map = np.load(str(folder / "individual" / "original" / (stem + "_gt_7x7.npy")))
                    fixed_map = np.load(str(folder / "individual" / "spatial_fixed" / (stem + "_gt_7x7.npy")))
                    pair.update(cam_similarity(old_map, fixed_map))
                    write_json(record_path, {"pair": pair, "predictions": records})
                else:
                    if loaded is None:
                        loaded = load_pair(refs, device)
                    pair, records = evaluate_image(seed, item, refs, loaded, device, folder)
                    write_json(record_path, {"pair": pair, "predictions": records})
                pairs.append(pair)
                predictions.extend(records)
                print("SEED {} IMAGE {}/{} {} ok".format(seed, index, len(samples), stem), flush=True)
            except Exception as exc:
                failures.append({"seed": seed, "image_filename": item["image_filename"],
                                 "error": "{}: {}".format(type(exc).__name__, exc)})
                print("SEED {} IMAGE {}/{} {} FAILED: {}".format(seed, index, len(samples), stem, exc), flush=True)
    finally:
        if loaded is not None:
            del loaded
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if pairs:
        write_csv(folder / "sample_manifest.csv", pairs, list(pairs[0]))
        write_csv(folder / "predictions.csv", predictions, list(predictions[0]))
    write_json(folder / "failures.json", failures)
    primary = [p for p in pairs if p["selection_group"] == "random_stratified"]
    if len(primary) != (2 if smoke else 16) or failures:
        raise RuntimeError("Seed {} incomplete: {} valid images, {} failures".format(seed, len(pairs), len(failures)))
    for label in ("HCC", "Hemangioma"):
        selected = [folder.parent / p["comparison_path"] for p in primary if p["class_label"] == label]
        draw_overview(folder / "overview" / (label + "_overview.png"), selected)
    examples = [folder.parent / p["comparison_path"] for p in pairs if p["selection_group"] == "prediction_selected"]
    draw_overview(folder / "overview" / "prediction_selected_overview.png", examples)
    summary = {"seed": seed, "images": len(pairs), "random_images": len(primary),
               "selected_cases": len(pairs) - len(primary), "valid_cams": 2 * len(pairs),
               "mean_top20_iou": float(np.mean([p["top20_iou"] for p in primary])),
               "mean_centroid_distance": float(np.mean([p["centroid_distance"] for p in primary])),
               "mean_pearson_7x7": float(np.mean([p["pearson_7x7"] for p in primary])),
               "random_prediction_disagreements": sum(p["original_correct"] != p["fixed_correct"] for p in primary),
               "original_1x1_all": all(r["attention_shape"] == [1, 1] for r in predictions if r["model"] == "original"),
               "fixed_7x7_all": all(r["attention_shape"] == [7, 7] for r in predictions if r["model"] == "spatial_fixed"),
               "max_abs_prob_difference": max(abs(r["prob_difference"]) for r in predictions)}
    write_json(folder / "seed_summary.json", summary)
    write_json(folder / "completion.json", {"requested_images": len(samples), "completed_images": len(pairs),
                                           "invalid_cams": 0, "failed_images": 0})
    return summary


def aggregate(summaries):
    folder = OUT / "summary"
    rows = []
    predictions = []
    for seed in SEEDS:
        seed_dir = OUT / ("seed_" + str(seed))
        if (seed_dir / "completion.json").exists():
            rows += read_csv(seed_dir / "sample_manifest.csv")
            predictions += read_csv(seed_dir / "predictions.csv")
    write_csv(folder / "seed_summary.csv", summaries, list(summaries[0]))
    if rows:
        write_csv(folder / "image_pair_metrics.csv", rows, list(rows[0]))
    primary = [r for r in rows if r["selection_group"] == "random_stratified"]
    prior = {int(r["split_seed"]): r for r in read_csv(PAIRED / "summary" / "paired_results.csv")}
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    x = [s["seed"] for s in summaries]
    axes[0].plot(x, [s["mean_top20_iou"] for s in summaries], "o-", color="#2b6cb0")
    axes[0].set_ylabel("Top 20% IoU")
    axes[1].plot(x, [s["mean_centroid_distance"] for s in summaries], "o-", color="#d97732")
    axes[1].set_ylabel("重心距离 / 图像对角线")
    axes[2].bar(x, [100 * float(prior[s["seed"]]["patient_balanced_accuracy_delta"]) for s in summaries],
                color="#438c87")
    axes[2].axhline(0, color="black", linewidth=.7)
    axes[2].set_ylabel("患者平衡准确率差值\n百分点")
    axes[2].set_xticks(x)
    axes[2].set_xlabel("患者划分种子")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(str(folder / "paired_cam_summary.png"), dpi=150)
    plt.close(fig)
    lines = ["# 十组原版与空间修正版 Grad-CAM 配对实验", "",
             "- 仅复用已训练的 20 份最佳权重；十组患者划分各抽取 HCC/血管瘤 8 名测试患者，每人一张。",
             "- 主图对真实类别 logit，在两版 ResNet-50 主干末层 7×7 特征上计算 Grad-CAM。",
             "- 空间注意力 sigmoid 权重另行审计；它不是 Grad-CAM。",
             "- 各图独立归一化，颜色深浅不能跨模型比较绝对贡献强度。", "",
             "| 种子 | 随机图像 | 另选案例 | CAM top20% IoU 均值 | 响应重心距离均值 | 患者平衡准确率差值 |",
             "|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        d = float(prior[s["seed"]]["patient_balanced_accuracy_delta"])
        lines.append("| {} | {} | {} | {:.3f} | {:.3f} | {:+.4f} |".format(
            s["seed"], s["random_images"], s["selected_cases"],
            s["mean_top20_iou"], s["mean_centroid_distance"], d))
    selected_counts = defaultdict(int)
    for r in rows:
        if r["selection_group"] == "prediction_selected":
            selected_counts[r["selection_reason"]] += 1
    class_summaries = []
    for label in ("HCC", "Hemangioma"):
        subset = [r for r in primary if r["class_label"] == label]
        if subset:
            class_summaries.append("{}：{} 张，IoU 均值 {:.3f}，重心距离均值 {:.3f}".format(
                label, len(subset), np.mean([float(r["top20_iou"]) for r in subset]),
                np.mean([float(r["centroid_distance"]) for r in subset])))
    lines.extend(["", "## 汇总与验收", "",
        "- 已完成划分：{}/10；随机展示记录 {} 条；不同患者 {} 名；额外选例 {} 条。".format(
        len(summaries), len(primary), len({r["patient_id"] for r in primary}),
        sum(s["selected_cases"] for s in summaries)),
        "- 主目标 CAM 有效数 {}；原版空间权重均为 1×1：{}；修正版均为 7×7：{}。".format(
            2 * len(rows), "是" if all(s["original_1x1_all"] for s in summaries) else "否",
            "是" if all(s["fixed_7x7_all"] for s in summaries) else "否"),
        "- 十组随机样本平均 Top 20% 热区 IoU：{:.3f}（逐组均值范围 {:.3f}–{:.3f}）；平均重心距离：{:.3f} 个 7×7 网格对角线。".format(
            np.mean([s["mean_top20_iou"] for s in summaries]),
            min(s["mean_top20_iou"] for s in summaries), max(s["mean_top20_iou"] for s in summaries),
            np.mean([s["mean_centroid_distance"] for s in summaries])),
        "- 分类别：" + "；".join(class_summaries) + "。",
        "- 另选案例：" + "；".join("{} {} 张".format(label, selected_counts[k]) for k, label in
                                 (("fixed_right_original_wrong", "修正版对、原版错"),
                                  ("original_right_fixed_wrong", "原版对、修正版错"),
                                  ("both_wrong", "两版都错"), ("both_right", "两版都对"))) + "。",
        "- 本次推理与训练时保存的测试预测类别不一致：0 条；HCC 概率最大绝对差：{:.6f}。".format(
            max(abs(float(r["prob_difference"])) for r in predictions)),
        "- 误判预测类别 CAM：{} 张；均保存数值矩阵及叠加图。".format(
            sum(bool(r.get("predicted_cam_path")) for r in predictions)),
        "- 修正版 7×7 空间权重的标准差均值 {:.3f}；原版只有 1×1 标量，无法按位置变化。".format(
            np.mean([float(r["attention_std"]) for r in predictions if r["model"] == "spatial_fixed"])),
        "", "## 输出", "",
        "- `seed_1001/` 至 `seed_1010/`：每组的三列 Grad-CAM 对照、空间注意力权重审计、两类总览、个别预测案例、逐图预测、7×7 与 224×224 CAM 矩阵及完成记录。",
        "- `summary/seed_summary.csv`：每组空间指标及验证结果；`image_pair_metrics.csv`：逐图配对指标；`paired_cam_summary.png`：十组趋势。",
        "- `smoke_seed_1001/`：正式运行之前的两图 GPU 冒烟验证，不纳入十组汇总。",
        "", "## 解释边界", "",
        "IoU、相关性和重心距离仅描述两版热图的相似程度，不能证明哪版更准确。没有病灶 ROI 真值，无法评估定位质量。十组测试集有患者重叠，不能把所有图像记录当作独立样本做显著性推断。另选案例不参与随机样本的空间指标汇总。",
        "空间权重的 7×7 变化证明修正版具备位置加权能力，并不证明该模块提高分类正确率；两版同时改变了分支特征的空间保留方式。底图的分类准确率差值来自完整测试集，CAM 相似度来自抽样图像，不能当作同一指标直接相关分析。", ""])
    (folder / "结果说明.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(folder / "progress.json", {"completed_seeds": len(summaries), "planned_seeds": 10,
                                          "random_image_records": len(primary), "valid_cams": 2 * len(rows)})


def main():
    args = arguments()
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seeds = (args.seed,) if args.seed is not None else SEEDS
    print("DEVICE {} SEEDS {} SMOKE {}".format(device, seeds, args.smoke), flush=True)
    for seed in seeds:
        run_seed(seed, args.smoke, device)
        if not args.smoke:
            complete = [read_json(OUT / ("seed_" + str(s)) / "seed_summary.json")
                        for s in SEEDS if (OUT / ("seed_" + str(s)) / "completion.json").exists()]
            aggregate(complete)
    print("GRADCAM STUDY COMPLETE", flush=True)


if __name__ == "__main__":
    main()
