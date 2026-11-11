"""Generate four-model Grad-CAM comparisons without retraining any model.

Examples:
  python -m gradcam_analysis.generate_comparison --image IMAGE_FILENAME --output gradcam_results/smoke
  python -m gradcam_analysis.generate_comparison
"""

import argparse
import csv
import json
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torchvision import transforms

from .cam import GradCAM
from .models import ROOT, SPECS, load_models, sha256
from .selection import CLASS_TO_IDX, error_cases, manual_sample, random_samples, test_records
from .visualization import ALPHA, COLORMAP, comparison, overview, render_cam, save_png


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="Test image filename or audited absolute path")
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--per-class", type=int, default=8)
    parser.add_argument("--no-error-cases", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", default="gradcam_results/main")
    args = parser.parse_args()
    if args.per_class < 1:
        parser.error("--per-class must be positive")
    return args


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_rows(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def image_and_tensor(row, normalization):
    source = ROOT / row["source_path"]
    if sha256(source) != row["sha256"]:
        raise ValueError("Image hash mismatch: " + str(source))
    with Image.open(str(source)) as image:
        rgb = image.convert("RGB").resize((224, 224), resample=Image.BILINEAR)
    array = np.asarray(rgb)
    tensor = transforms.Compose((transforms.ToTensor(),
                                 transforms.Normalize(normalization["mean"],
                                                      normalization["std"])))(rgb)
    return array, tensor.unsqueeze(0)


def reference_predictions():
    tables = {}
    for spec in SPECS:
        path = ROOT / "runs" / spec.run_dir / "test_predictions.csv"
        tables[spec.name] = pd.read_csv(str(path)).set_index("image_filename")
    return tables


def evaluate_one(item, loaded, stats, device, output, references):
    rgb, tensor = image_and_tensor(item, stats)
    tensor = tensor.to(device)
    stem = Path(item["image_filename"]).stem
    original_path = output / "individual" / "Original" / (stem + ".png")
    save_png(original_path, rgb)
    results = []
    target = CLASS_TO_IDX[item["class_label"]]
    for spec in SPECS:
        entry = loaded[spec.name]
        base = output / "individual" / spec.name
        base.mkdir(parents=True, exist_ok=True)
        heatmap_path = base / (stem + "_gt_heatmap.png")
        overlay_path = base / (stem + "_gt_overlay.png")
        cam_path = base / (stem + "_gt.npy")
        pred_cam_path = base / (stem + "_pred.npy")
        record = {
            "patient_id": item["patient_id"],
            "image_filename": item["image_filename"],
            "image_path": str((ROOT / item["source_path"]).resolve()),
            "true_label": item["class_label"],
            "model_name": spec.name,
            "cam_target": item["class_label"],
            "cam_target_mode": "ground_truth",
            "target_layer": spec.layer_name,
            "original_path": str(original_path.resolve()),
            "cam_path": "", "heatmap_path": "", "overlay_path": "",
            "predicted_cam_path": "", "predicted_overlay_path": "",
            "cam_valid": False, "cam_error": "",
            "selection_group": item["selection_group"],
            "selection_reason": item["selection_reason"],
        }
        with GradCAM(entry["model"], entry["target_layer"]) as cam:
            try:
                result = cam.generate(tensor, target)
                probabilities = result["probabilities"]
                np.save(str(cam_path), result["cam"])
                heatmap, overlay = render_cam(rgb, result["cam"])
                save_png(heatmap_path, heatmap)
                save_png(overlay_path, overlay)
                record.update(cam_valid=True, cam_path=str(cam_path.resolve()),
                              heatmap_path=str(heatmap_path.resolve()),
                              overlay_path=str(overlay_path.resolve()),
                              activation_shape=json.dumps(result["activation_shape"]),
                              gradient_l1=result["gradient_l1"],
                              raw_cam_min=result["raw_cam_min"],
                              raw_cam_max=result["raw_cam_max"])
            except (RuntimeError, ValueError) as error:
                record["cam_error"] = str(error)
                with torch.no_grad():
                    logits = entry["model"](tensor)
                    probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
            predicted_index = int(np.argmax(probabilities))
            predicted = "HCC" if predicted_index == 0 else "Hemangioma"
            expected = references[spec.name].loc[item["image_filename"]]
            if expected["true_class"] != item["class_label"]:
                raise ValueError("Saved true label differs from audited test manifest")
            record.update(predicted_label=predicted, prob_hcc=float(probabilities[0]),
                          prob_hemangioma=float(probabilities[1]),
                          confidence=float(probabilities[predicted_index]),
                          correct=bool(predicted_index == target),
                          saved_test_prob_hcc=float(expected["prob_hcc"]),
                          saved_test_predicted_label=str(expected["pred_class"]),
                          probability_difference_from_saved=float(
                              probabilities[0] - float(expected["prob_hcc"])),
                          prediction_differs_from_saved=bool(
                              predicted_index != int(expected["pred_label"])))
            if predicted_index != target:
                try:
                    predicted_cam = cam.generate(tensor, predicted_index)
                    pred_overlay = base / (stem + "_pred_overlay.png")
                    _, overlay = render_cam(rgb, predicted_cam["cam"])
                    np.save(str(pred_cam_path), predicted_cam["cam"])
                    save_png(pred_overlay, overlay)
                    record["predicted_cam_path"] = str(pred_cam_path.resolve())
                    record["predicted_overlay_path"] = str(pred_overlay.resolve())
                except (RuntimeError, ValueError) as error:
                    record["predicted_cam_error"] = str(error)
        results.append(record)
    return original_path, results


def structure_audit(loaded, stats, item, device):
    _, tensor = image_and_tensor(item, stats)
    tensor = tensor.to(device)
    target = CLASS_TO_IDX[item["class_label"]]
    rows = []
    for spec in SPECS:
        entry = loaded[spec.name]
        with GradCAM(entry["model"], entry["target_layer"]) as cam:
            result = cam.generate(tensor, target)
        rows.append({"model": spec.name, "target_layer": spec.layer_name,
                     "activation_shape": result["activation_shape"],
                     "gradient_l1": result["gradient_l1"],
                     "cam_shape": list(result["cam"].shape),
                     "cam_min": float(result["cam"].min()),
                     "cam_max": float(result["cam"].max()),
                     "best_epoch": entry["best_epoch"],
                     "checkpoint_sha256": entry["checkpoint_sha256"]})
    return rows


def report(output, samples, prediction_rows, audit, config):
    primary = [r for r in samples if r["selection_group"] == "random_stratified"]
    invalid = [r for r in prediction_rows if not r["cam_valid"]]
    lines = [
        "# 四模型 Grad-CAM 对照实验结果", "",
        "- 仅使用现有患者级 Clean 测试集、四个已训练最佳检查点；未重新训练。",
        "- 主样本：固定随机种子 {}，HCC {} 张、血管瘤 {} 张，每名患者最多一张。".format(
            config["sample_seed"], sum(r["class_label"] == "HCC" for r in primary),
            sum(r["class_label"] == "Hemangioma" for r in primary)),
        "- 额外预测结果选例与随机抽样分开记录；不用于估计模型总体性能。",
        "- CAM 目标：主图统一为真实类别 logit；误分类样本另外保存预测类别 CAM。",
        "- 目标层：各模型 ResNet 主干最后一个残差块，均保留 7×7 空间结构。",
        "- 方法：梯度空间平均→加权求和→ReLU→双线性上采样→每张 CAM 单独 min-max 归一化。",
        "- 颜色映射 {}，叠加系数 {}；颜色深浅不能跨图解释为绝对贡献强度。".format(
            config["colormap"], config["alpha"]),
        "", "## 结构与计算检查", "",
        "| 模型 | 目标层 | 特征形状 | 梯度 L1 |", "|---|---|---|---:|",
    ]
    for row in audit:
        lines.append("| {} | `{}` | `{}` | {:.6g} |".format(
            row["model"], row["target_layer"], row["activation_shape"], row["gradient_l1"]))
    lines.extend(["", "## 输出与异常", "",
                  "- 展示图像：{} 张；主目标 CAM：{} 张；无效 CAM：{} 张。".format(
                      len(samples), len(prediction_rows), len(invalid)),
                  "- 与训练时保存的测试预测相比，本次单图推理类别不同：{} 条；HCC 概率最大绝对差：{:.6f}。".format(
                      sum(bool(r["prediction_differs_from_saved"]) for r in prediction_rows),
                      max((abs(r["probability_difference_from_saved"]) for r in prediction_rows),
                          default=0.0)),
                  "- 主对照：`comparisons/`；批量图：`overview/`；原图、数值矩阵和热图：`individual/`。",
                  "- 样本、预测、配置：`sample_manifest.csv`、`predictions.csv`、`experiment_config.json`。"])
    for item in invalid:
        lines.append("- 无效：{} / {}：{}".format(item["image_filename"], item["model_name"], item["cam_error"]))
    lines.extend(["", "## 解释边界", "",
                  "本数据没有可信的病灶 ROI 标注。Grad-CAM 是指定层和类别的粗粒度响应图，不能作为病灶定位准确率或临床因果解释。原 `CustomModel` 的后续空间注意力输入为 1×1；本实验选择的是其压缩之前的主干 7×7 特征层。",
                  "", "不同模型都在同一测试图像和类别目标下展示；归一化按每张 CAM 独立进行，因此只能比较响应位置与形态，不能直接比较不同模型颜色的绝对强弱。",
                  ""])
    (output / "report" / "gradcam_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = arguments()
    device = torch.device("cuda" if (args.device == "cuda" or
                                     args.device == "auto" and torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    (output / "report").mkdir(exist_ok=True)
    by_name = test_records()
    loaded, stats = load_models(device)
    if args.image:
        samples = manual_sample(by_name, args.image)
    else:
        samples = random_samples(by_name, args.seed, args.per_class)
        if not args.no_error_cases:
            samples += error_cases(by_name, {s["patient_id"] for s in samples})
    if len({s["patient_id"] for s in samples}) != len(samples):
        raise ValueError("Selected two images from one patient")
    config = {"sample_seed": args.seed, "per_class": args.per_class,
              "image": args.image, "error_cases": not args.no_error_cases,
              "device": str(device), "torch": torch.__version__,
              "torchvision": torchvision.__version__, "python": sys.version.split()[0],
              "class_to_idx": CLASS_TO_IDX, "resize": [224, 224],
              "normalization": stats, "colormap": COLORMAP, "alpha": ALPHA,
              "cam_target": "ground truth class raw logit",
              "cam_interpolation": "bilinear align_corners=False",
              "cam_normalization": "per-image min-max after upsampling",
              "manifest_sha256": sha256(ROOT / "data" / "metadata" / "image_manifest.csv"),
              "models": [{"name": spec.name, "run_dir": spec.run_dir,
                          "target_layer": spec.layer_name,
                          "checkpoint_sha256": loaded[spec.name]["checkpoint_sha256"],
                          "best_epoch": loaded[spec.name]["best_epoch"]} for spec in SPECS]}
    config_path = output / "experiment_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Existing output has different settings; use another --output")
    write_json(config_path, config)
    audit = structure_audit(loaded, stats, samples[0], device)
    write_json(output / "target_layer_audit.json", audit)
    print("TARGET LAYERS:", json.dumps(audit, ensure_ascii=False), flush=True)
    sample_rows, predictions = [], []
    result_by_image, original_by_image = {}, {}
    references = reference_predictions()
    for number, item in enumerate(samples, start=1):
        try:
            original_path, results = evaluate_one(item, loaded, stats, device, output, references)
            category = (item["class_label"] if item["selection_group"] == "random_stratified"
                        else "Examples" if item["selection_reason"] == "all_models_correct"
                        else "Misclassified")
            compare_path = output / "comparisons" / category / (Path(item["image_filename"]).stem + ".png")
            comparison(compare_path, item, results, original_path)
            sample_rows.append(dict(item, image_path=str((ROOT / item["source_path"]).resolve()),
                                    original_path=str(original_path.resolve()),
                                    comparison_path=str(compare_path.resolve()), status="ok", error=""))
            predictions.extend(results)
            result_by_image[item["image_filename"]] = results
            original_by_image[item["image_filename"]] = original_path
            print("IMAGE {}/{} {}: {} valid CAMs".format(number, len(samples),
                  item["image_filename"], sum(r["cam_valid"] for r in results)), flush=True)
        except Exception as error:
            sample_rows.append(dict(item, image_path=str((ROOT / item["source_path"]).resolve()),
                                    original_path="", comparison_path="",
                                    status="failed", error="{}: {}".format(type(error).__name__, error)))
            print("IMAGE {}/{} FAILED {}: {}".format(number, len(samples),
                  item["image_filename"], error), flush=True)
            traceback.print_exc()
        write_rows(output / "sample_manifest.csv", sample_rows,
                   list(samples[0]) + ["image_path", "original_path", "comparison_path", "status", "error"])
        fields = ("patient_id", "image_filename", "image_path", "true_label", "model_name",
                  "predicted_label", "prob_hcc", "prob_hemangioma", "confidence", "correct",
                  "saved_test_prob_hcc", "saved_test_predicted_label",
                  "probability_difference_from_saved", "prediction_differs_from_saved",
                  "cam_target", "cam_target_mode", "target_layer", "cam_valid", "cam_error",
                  "activation_shape", "gradient_l1", "raw_cam_min", "raw_cam_max",
                  "original_path", "cam_path", "heatmap_path", "overlay_path",
                  "predicted_cam_path", "predicted_overlay_path", "predicted_cam_error",
                  "selection_group", "selection_reason")
        write_rows(output / "predictions.csv", predictions, fields)
    for class_name in ("HCC", "Hemangioma"):
        group = [r for r in samples if r["selection_group"] == "random_stratified" and
                 r["class_label"] == class_name and r["image_filename"] in result_by_image]
        overview(output / "overview" / (class_name + "_overview.png"), group,
                 result_by_image, original_by_image, class_name + "：随机患者样本")
    examples = [r for r in samples if r["selection_group"] == "prediction_selected" and
                r["image_filename"] in result_by_image]
    overview(output / "overview" / "prediction_selected_overview.png", examples,
             result_by_image, original_by_image, "预测结果选例（非随机抽样）")
    misclassified = [r for r in examples if any(
        not item["correct"] for item in result_by_image[r["image_filename"]])]
    overview(output / "overview" / "Misclassified_overview.png", misclassified,
             result_by_image, original_by_image, "至少一个模型误判的选例")
    report(output, sample_rows, predictions, audit, config)
    write_json(output / "completion.json", {
        "requested_images": len(samples), "completed_images": sum(r["status"] == "ok" for r in sample_rows),
        "failed_images": sum(r["status"] != "ok" for r in sample_rows),
        "valid_cams": sum(r["cam_valid"] for r in predictions),
        "invalid_cams": sum(not r["cam_valid"] for r in predictions),
        "selection_groups": dict(Counter(r["selection_group"] for r in samples))})
    print("COMPLETE:", output / "report" / "gradcam_summary.md", flush=True)


if __name__ == "__main__":
    main()
