"""Compare completed ResNet-18 and ResNet-50 runs on identical test cases.

Uses the saved prediction CSVs. Writes a paired comparison under runs/.
No model training or test-time threshold selection is performed.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
RUN18 = RUNS / "resnet18_patient_clean_seed24"
RUN50 = RUNS / "resnet50_patient_clean_seed24"


def load_pairs(filename, key):
    first = pd.read_csv(RUN18 / filename)
    second = pd.read_csv(RUN50 / filename)
    if first[key].duplicated().any() or second[key].duplicated().any():
        raise ValueError("Duplicate keys in " + filename)
    joined = first.merge(second, on=key, suffixes=("_18", "_50"), validate="one_to_one")
    if len(joined) != len(first) or len(joined) != len(second):
        raise ValueError("Models were not evaluated on identical " + key + " values")
    if not (joined.true_label_18 == joined.true_label_50).all():
        raise ValueError("True labels differ between model predictions")
    if key == "image_filename" and not (joined.patient_id_18 == joined.patient_id_50).all():
        raise ValueError("Patient IDs differ for the same image")
    joined["correct_18"] = (joined.true_label_18 == joined.pred_label_18).astype(int)
    joined["correct_50"] = (joined.true_label_50 == joined.pred_label_50).astype(int)
    return joined


def summary(joined):
    a = joined.correct_18.to_numpy()
    b = joined.correct_50.to_numpy()
    return {
        "n": int(len(joined)),
        "resnet18_correct": int(a.sum()),
        "resnet50_correct": int(b.sum()),
        "resnet18_accuracy": float(a.mean()),
        "resnet50_accuracy": float(b.mean()),
        "accuracy_difference_resnet50_minus_resnet18": float(b.mean() - a.mean()),
        "both_correct": int(((a == 1) & (b == 1)).sum()),
        "resnet18_only_correct": int(((a == 1) & (b == 0)).sum()),
        "resnet50_only_correct": int(((a == 0) & (b == 1)).sum()),
        "both_wrong": int(((a == 0) & (b == 0)).sum()),
    }


def paired_patient_bootstrap(images, patients, draws=5000, seed=24):
    rng = np.random.RandomState(seed)
    ids = sorted(patients.patient_id)
    image_count = np.array([int((images.patient_id_18 == patient_id).sum()) for patient_id in ids])
    image_delta_sum = np.array([
        int((images.loc[images.patient_id_18 == patient_id, "correct_50"] -
             images.loc[images.patient_id_18 == patient_id, "correct_18"]).sum())
        for patient_id in ids
    ])
    indexed_patients = patients.set_index("patient_id")
    patient_delta = np.array([
        int(indexed_patients.loc[patient_id, "correct_50"] -
            indexed_patients.loc[patient_id, "correct_18"])
        for patient_id in ids
    ])
    image_deltas, patient_deltas = [], []
    for _ in range(draws):
        sample = rng.randint(0, len(ids), size=len(ids))
        image_deltas.append(float(image_delta_sum[sample].sum() / image_count[sample].sum()))
        patient_deltas.append(float(patient_delta[sample].mean()))
    return {
        "method": "paired bootstrap resampling test patients with replacement",
        "draws": draws,
        "image_accuracy_difference_95_interval": [float(x) for x in np.percentile(image_deltas, [2.5, 97.5])],
        "patient_accuracy_difference_95_interval": [float(x) for x in np.percentile(patient_deltas, [2.5, 97.5])],
    }


def main():
    for run in (RUN18, RUN50):
        if not (run / "report.md").is_file():
            raise FileNotFoundError("Completed run missing: " + str(run))
    config18 = json.loads((RUN18 / "run_config.json").read_text(encoding="utf-8"))
    config50 = json.loads((RUN50 / "run_config.json").read_text(encoding="utf-8"))
    for field in ("seed", "actual_batch_size", "patience", "optimizer", "learning_rate",
                  "augmentation", "normalization", "manifest_sha256", "threshold"):
        if config18[field] != config50[field]:
            raise ValueError("Run configurations differ on " + field)
    images = load_pairs("test_predictions.csv", "image_filename")
    patients = load_pairs("patient_predictions.csv", "patient_id")
    if len(images) != 254 or len(patients) != 74:
        raise ValueError("Unexpected test sample counts")
    image_result = summary(images)
    patient_result = summary(patients)
    intervals = paired_patient_bootstrap(images, patients)
    result = {"image_level": image_result, "patient_level": patient_result,
              "paired_bootstrap": intervals,
              "note": "One training seed and one patient-disjoint test split; differences are descriptive."}
    (RUNS / "resnet18_vs_resnet50_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# ResNet-18 与 ResNet-50：同一测试集对照", "",
        "两次实验均使用相同的 Clean 患者级划分、种子 24、随机初始化、批大小 32、Adam 3e-5、数据增强、归一化与验证损失选模规则。仅模型骨干不同；两次训练由各自的早停规则决定结束轮次。", "",
        "| 层级 | ResNet-18 | ResNet-50 | ResNet-50 减 ResNet-18（百分点） |",
        "|---|---:|---:|---:|",
        "| 图像准确率（254 张） | {:.2%} | {:.2%} | {:+.2f} |".format(
            image_result["resnet18_accuracy"], image_result["resnet50_accuracy"],
            100 * image_result["accuracy_difference_resnet50_minus_resnet18"]),
        "| 患者准确率（74 人） | {:.2%} | {:.2%} | {:+.2f} |".format(
            patient_result["resnet18_accuracy"], patient_result["resnet50_accuracy"],
            100 * patient_result["accuracy_difference_resnet50_minus_resnet18"]),
        "", "同一测试对象上的正确性对照：", "",
        "- 图像：两者都对 {}；只有 ResNet-18 对 {}；只有 ResNet-50 对 {}；两者都错 {}。".format(
            image_result["both_correct"], image_result["resnet18_only_correct"],
            image_result["resnet50_only_correct"], image_result["both_wrong"]),
        "- 患者：两者都对 {}；只有 ResNet-18 对 {}；只有 ResNet-50 对 {}；两者都错 {}。".format(
            patient_result["both_correct"], patient_result["resnet18_only_correct"],
            patient_result["resnet50_only_correct"], patient_result["both_wrong"]),
        "", "以测试患者为单位做 5,000 次配对 bootstrap，准确率差值的 95% 区间：", "",
        "- 图像级：[{:+.2f}, {:+.2f}] 个百分点。".format(
            *[100 * x for x in intervals["image_accuracy_difference_95_interval"]]),
        "- 患者级：[{:+.2f}, {:+.2f}] 个百分点。".format(
            *[100 * x for x in intervals["patient_accuracy_difference_95_interval"]]),
        "", "这是单个训练种子、单个患者级测试划分上的描述性比较。区间反映测试患者抽样波动，不包含重新训练带来的随机种子波动，也不能直接外推至其他医院或论文原测试集。",
    ]
    (RUNS / "resnet18_vs_resnet50_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
