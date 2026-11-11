"""Create final Chinese summary and paired figures from completed study outputs."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SUMMARY = HERE / "outputs" / "summary"
FONT = Path(r"C:\Windows\Fonts\NotoSansSC-Regular.ttf")
if FONT.is_file():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False


def main():
    data = pd.read_csv(str(SUMMARY / "paired_results.csv"))
    if len(data) != 10 or data.split_seed.tolist() != list(range(1001, 1011)):
        raise ValueError("Expected all ten predeclared split seeds")
    seeds = data.split_seed.to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), dpi=160)
    for ax, prefix, title in (
        (axes[0], "patient_balanced_accuracy", "患者级平衡准确率"),
        (axes[1], "image_balanced_accuracy", "图像级平衡准确率"),
    ):
        original = data[prefix + "_original"].to_numpy()
        fixed = data[prefix + "_fixed"].to_numpy()
        for x, old, new in zip(seeds, original, fixed):
            ax.plot([x - .12, x + .12], [old, new], color="#9aa6b2", linewidth=1.3)
        ax.scatter(seeds - .12, original, s=38, color="#3274aa", label="原 CustomModel", zorder=3)
        ax.scatter(seeds + .12, fixed, s=38, color="#d57a31", label="空间修正版", zorder=3)
        ax.set_title(title)
        ax.set_ylim(min(original.min(), fixed.min()) - .007, 1.005)
        ax.set_ylabel("指标值")
        ax.set_xticks(seeds)
        ax.grid(axis="y", alpha=.25)
        ax.legend(loc="lower left")
    delta = data.patient_balanced_accuracy_delta.to_numpy()
    colors = ["#d57a31" if x > 0 else "#3274aa" if x < 0 else "#a9b1b8" for x in delta]
    axes[2].bar(seeds, delta * 100, color=colors, width=.65)
    axes[2].axhline(0, color="#4f5a65", linewidth=1)
    axes[2].set_title("患者级平衡准确率配对差值：修正版 − 原版")
    axes[2].set_ylabel("百分点")
    axes[2].set_xlabel("患者划分种子")
    axes[2].set_xticks(seeds)
    axes[2].grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(str(SUMMARY / "paired_comparison.png"), bbox_inches="tight")
    plt.close(fig)

    patient_delta = data.patient_balanced_accuracy_delta.to_numpy()
    lines = [
        "# 原 CustomModel 与空间修正版：10 组患者划分最终结果", "",
        "## 实验执行", "",
        "- 预先固定患者划分种子 1001–1010；每组按类别在患者层面约 70%/20%/10% 划分。种子 24 的先导结果不计入本次汇总。",
        "- 每组的两版模型共用相同的患者、图像及仅由该组训练集计算的标准化参数；训练种子固定 24，初始参数哈希一致。",
        "- 两版均使用 ImageNet 预训练 ResNet-50 主干、Adam 3×10⁻⁵、批大小 32、最多 100 轮、验证损失连续 7 轮不改善早停。最终测试只使用各自最低验证损失检查点。",
        "- 10 组均已完成；共 20 次训练与测试，20 份最佳权重、报告及患者预测文件齐全，错误日志为空。每组测试集为 74 名患者；图像数因患者而异。",
        "", "## 主要结果", "",
        "| 指标（10 组均值） | 原 CustomModel | 空间修正版 | 修正版 − 原版 |",
        "|---|---:|---:|---:|",
    ]
    for label, key, percent in (
        ("患者级平衡准确率（主指标）", "patient_balanced_accuracy", True),
        ("患者级准确率", "patient_accuracy", True),
        ("患者级 HCC ROC-AUC", "patient_roc_auc_hcc", False),
        ("患者级 Brier 分数，低为好", "patient_brier_hcc", False),
        ("图像级平衡准确率", "image_balanced_accuracy", True),
        ("图像级准确率", "image_accuracy", True),
        ("图像级 HCC ROC-AUC", "image_roc_auc_hcc", False),
        ("图像级 Brier 分数，低为好", "image_brier_hcc", False),
    ):
        a = float(data[key + "_original"].mean())
        b = float(data[key + "_fixed"].mean())
        if percent:
            lines.append("| {} | {:.3f}% | {:.3f}% | {:+.3f} 个百分点 |".format(
                label, a * 100, b * 100, (b - a) * 100))
        else:
            lines.append("| {} | {:.6f} | {:.6f} | {:+.6f} |".format(label, a, b, b - a))
    lines.extend([
        "", "主指标按种子逐组比较：修正版胜 1 组、持平 8 组、落后 1 组；差值中位数为 0。主指标平均差值为 {:+.4f}（{:+.3f} 个百分点），对 10 组差值重采样的描述性 95% 区间为 **[-0.0011, +0.0036]**，包含 0。".format(
            float(patient_delta.mean()), float(patient_delta.mean() * 100)),
        "", "## 结果解读", "",
        "- 种子 24 的先导实验中修正版更好，但在这 10 个新划分上，患者级平衡准确率没有出现稳定优势。患者级准确率两版通常已接近或达到 100%，存在明显的指标上限效应。",
        "- 图像级平均平衡准确率修正版反而略低；概率质量指标在患者级和图像级的方向也不完全一致。因此不能据当前数据宣布修正版在总体性能上优于原版。",
        "- 两版差异同时包括分支保留 `7×7` 空间特征以及 `SpatialAttention` 真正获得位置变化的输入；本实验不单独衡量空间注意力模块的贡献。",
        "- 十个测试集反复来自同一来源患者，彼此有重叠；上面的区间只描述对划分变化的敏感程度，不是外部医院人群的泛化置信区间。",
        "", "## 输出索引", "",
        "- `paired_results.csv`：10 组图像级、患者级指标和配对差值。",
        "- `paired_report.md`：逐种子的主指标及患者错误转换。",
        "- `paired_comparison.png`：两层级指标配对图和主指标差值图。",
        "- `../splits/seed_*/`：患者/图像划分及该组训练集标准化。",
        "- `../runs/seed_*/original/` 与 `spatial_fixed/`：配置、历史、最佳/恢复检查点、逐图像/患者预测、指标、混淆矩阵、ROC 与报告。",
        "- `../study.stdout.log`、`../study.stderr.log`：完整运行及错误日志。",
        "",
    ])
    (SUMMARY / "最终结果说明.md").write_text("\n".join(lines), encoding="utf-8")
    print(SUMMARY / "最终结果说明.md")
    print(SUMMARY / "paired_comparison.png")


if __name__ == "__main__":
    main()
