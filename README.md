# SMC-LUD 肝脏超声二分类复现实验

本项目基于 [jstak0804/Ultrasound_project](https://github.com/jstak0804/Ultrasound_project) 的模型实现，研究 SMC-LUD B 模式肝脏超声图像中的肝细胞癌（HCC）与血管瘤分类。这里记录四种模型在同一患者级划分上的单次训练，以及原始和空间修正版 `CustomModel` 在十组患者划分上的配对比较。**实验使用新建的患者级划分，不是论文原始测试划分的严格复现。**

## 仓库内容与数据边界

- `model.py`：原始 `CustomModel` 及新增 `CustomModelSpatialFixed`；原始模型未被覆盖。
- `scripts/`：数据审计、患者划分、训练集标准化、四种模型的自动训练与评估脚本。
- `patient_split_comparison/`：十组配对实验脚本及运行说明。
- `gradcam_analysis/`：四模型 Grad-CAM 分析代码；生成的热图不包含在仓库中。
- `experiment_results/`：无影像、无逐患者预测的配置、指标、训练历史和汇总报告。
- `base_line.py`、`Liver_train.py` 等：上游项目的旧入口；下述结果由 `scripts/` 下的新入口产生。

**本仓库不上传** `data/`、`runs/`、`figures/`、`gradcam_results/`、`patient_split_comparison/outputs/`，也不上传医学影像、模型检查点、Grad-CAM 图片、患者级预测明细、论文 PDF 或私人研究聊天记录。获取并合规存放 SMC-LUD 数据后，按下面的目录和脚本在本地重建。`requirements.txt` 是上游依赖清单，不能视为本次训练环境的精确锁定文件；实际环境版本见各次 `run_config.json`。

## 数据准备与共同评估流程

原始数据在本地 `data/original_data/`，分为 HCC/血管瘤及 Clean/Caliper。源数据审计共计 5,385 张图像，其中 Clean 2,656 张；本轮分类只使用 **Clean**。先由图像所在目录确定患者 ID，再建立患者与图像清单，核查重复内容和旧划分中的患者交叉。旧的图像级划分有患者跨集合，因此重新按患者、在每个类别内分配训练/验证/测试，约 70%/20%/10%。

单次实验的划分种子为 24：训练 **1,928 张/521 名患者**，验证 **474 张/149 名患者**，测试 **254 张/74 名患者**。训练、验证、测试患者互斥。RGB 图像缩放到 224×224；均值和标准差只从训练集计算。类别映射为 `HCC=0`、`Hemangioma=1`，HCC 作为 ROC-AUC 的阳性类。患者概率是该患者全部 Clean 图像的 HCC 概率均值，再以 0.5 为阈值分类。

本地数据放置方式：`data/original_data/HCC/{Clean,Caliper}/<患者目录>/...` 和 `data/original_data/Hemangioma/{Clean,Caliper}/<患者目录>/...`，以及源数据提供的患者元数据。若已经有 `data/metadata/` 与 `data/patient_split_clean/`，无需覆盖重建。数据准备脚本会检查输入并拒绝覆盖已生成目录：

```powershell
python scripts/prepare_smc_lud_data.py
python scripts/calculate_train_stats.py
```

训练均使用交叉熵、Adam（学习率 `3e-5`）、批大小 32、最多 100 轮、训练种子 24；验证损失连续 7 轮不改善则早停。只在验证集选择最低损失的检查点，随后对测试集进行图像级和患者级评估。训练增强包括水平/垂直翻转、±10° 旋转及轻微亮度/对比度扰动。实际运行环境为 Python 3.8.20、PyTorch 1.8.2+cu111、torchvision 0.9.2+cu111、RTX 4060 Laptop GPU；完整配置见结果目录。

## 四种模型：同一划分的单次训练

从项目根目录、在已配置 PyTorch GPU 的环境中运行：

```powershell
python scripts/train_resnet18_baseline.py
python scripts/train_resnet50_baseline.py
python scripts/train_attention_resnet50.py
python scripts/train_attention_resnet50_spatial_fixed.py
```

每条脚本先检查输入与 GPU，按验证损失保存最佳权重，支持中断后继续；完成后生成测试指标、混淆矩阵、ROC、预测明细和报告。本仓库仅保留不含图像和患者明细的结果摘录。

| 模型与入口 | 初始化及结构 | 完成轮次 / 最佳轮次 | 图像准确率 | 患者准确率 | 患者平衡准确率 | 患者 HCC AUC |
|---|---|---:|---:|---:|---:|---:|
| ResNet-18 · `train_resnet18_baseline.py` | 随机初始化，标准 ResNet-18 | 13 / 6 | 91.73%（233/254） | 94.59%（70/74） | 94.87% | 0.9762 |
| ResNet-50 · `train_resnet50_baseline.py` | 随机初始化，标准 ResNet-50 | 41 / 34 | 96.06%（244/254） | 97.30%（72/74） | 97.25% | 0.9985 |
| 原 `CustomModel` · `train_attention_resnet50.py` | ImageNet 预训练 ResNet-50 + 双分支注意力；空间注意力输入为 `1×1` | 23 / 16 | 98.43%（250/254） | 98.65%（73/74） | 98.44% | 1.0000 |
| 空间修正版 · `train_attention_resnet50_spatial_fixed.py` | 同类预训练主干；保留双分支 `7×7` 特征供空间注意力加权 | 19 / 12 | 99.21%（252/254） | 100%（74/74） | 100% | 1.0000 |

原版先把分支特征压成 `1×1`，使后续空间注意力无法区分不同空间位置；修正版让两条分支的通道权重乘回 `7×7` 特征图。修正版在种子 24 上结果较好，但这是一次划分的观察。普通 ResNet-50 从零初始化、注意力模型使用预训练，因此它们之间的差异**不能单独归因于注意力结构**。四次运行的完整数值见 [`experiment_results/single_runs/`](experiment_results/single_runs/)；代码和其他限制见 [`工程文件说明_训练与评估.md`](工程文件说明_训练与评估.md)。

## 十组患者划分：原版与空间修正版配对比较

执行 [`patient_split_comparison/run_study.py`](patient_split_comparison/run_study.py)，预先固定划分种子 1001–1010。每组重新按患者、类别生成约 70%/20%/10% 的 Clean 划分，并只用当组训练图像计算标准化参数。两版模型在一组内共享患者、图像、预处理、训练种子 24、训练超参数及相同初始权重；各自按最低验证损失选模，测试集各有 74 名患者。种子 24 的上述单次结果只作先导，不进入十组汇总。

```powershell
python patient_split_comparison/run_study.py --prepare-only
python patient_split_comparison/run_study.py
python patient_split_comparison/summarize_results.py
```

十组均已完成，共 **20 次训练与测试**。主指标为患者级平衡准确率；同时检查患者与图像准确率、AUC、Brier 分数等。

| 十组均值 | 原 `CustomModel` | 空间修正版 | 修正版 − 原版 |
|---|---:|---:|---:|
| 患者级平衡准确率（主指标） | 99.449% | 99.531% | +0.082 个百分点 |
| 患者级准确率 | 99.459% | 99.595% | +0.135 个百分点 |
| 患者级 HCC ROC-AUC | 0.999182 | 0.999628 | +0.000446 |
| 图像级平衡准确率 | 98.964% | 98.798% | −0.166 个百分点 |
| 图像级准确率 | 98.972% | 98.926% | −0.046 个百分点 |

主指标逐组比较为修正版 **胜 1、平 8、负 1**；平均配对差值 +0.000818，中位数 0。对十个差值重采样得到的描述性 95% 区间为 `[-0.0011, +0.0036]`，包含 0。图像级均值还略低，因此这批数据**没有显示空间修正版稳定优于原版**。十个测试集来自同一来源且有患者重叠，这个区间仅描述对划分的敏感度，不是外部医院泛化区间。修正同时改变分支特征的空间保留方式，不能单独量化 `SpatialAttention` 的贡献。

逐种子数值、患者平衡准确率配对表及文字结论见 [`experiment_results/paired_study/`](experiment_results/paired_study/)；本地运行及输出规则见 [`patient_split_comparison/README.md`](patient_split_comparison/README.md)。

## 结果边界与来源

这是 HCC/血管瘤**有标签分类**，没有病灶框或分割掩膜；Grad-CAM 热图不能据此计算病灶定位准确率。此仓库不包含原始数据和训练权重，因此仅凭仓库不能立即复跑或重新生成图像解释结果。数据源及论文请通过其正式发布渠道取得，原始工程来源见上方链接。`Hem_87` 的患者表记录缺失，主分析保留其目录标签，单次训练报告另提供排除该患者的敏感性分析。
