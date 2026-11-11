# 四种 ResNet 模型 Grad-CAM 可解释性对照实验

## 1. 目的、范围与使用的信息

本实验对四个**已经训练完成**的 HCC/血管瘤二分类模型，在同一批患者级 Clean 测试图像上生成 Grad-CAM。没有重新训练，没有修改网络结构、训练脚本或已有最佳权重。用户提供的《四种 ResNet 模型 Grad-CAM 可解释性对照实验——开发任务书》规定了样本、目标类别、输出和验收口径；上传的示例热图仅作为呈现形式参考，**没有作为本实验输入或病灶标注**。

实际使用的信息：`data/metadata/image_manifest.csv`（原图、患者、类别、Clean 与测试划分、SHA-256）、四次训练的 `run_config.json`、`best_model.pth`、`test_predictions.csv`，以及工程的 `model.py` 和现有 `Liver_train.py`、`base_line.py` 中的 Grad-CAM 计算思路。四个配置均采用 HCC=0、Hemangioma=1，相同的图像清单哈希和相同的训练集 RGB 标准化参数。数据预处理为 RGB、224×224、`ToTensor`、训练时确定的 Normalize；测试时没有随机增强。

方法参照 [PyTorch 模块前向 hook 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)、[PyTorch 双线性上采样文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html)和 [pytorch-grad-cam 项目的目标层选择说明](https://github.com/jacobgil/pytorch-grad-cam)。现有 `yolov5` 环境是 PyTorch 1.8.2+cu111、torchvision 0.9.2+cu111，未安装 `pytorch-grad-cam`；为保持环境不变，本实验使用 PyTorch 原生 hook 实现统一的标准 Grad-CAM，并复用了仓库已有实现的计算思路。仓库旧实现默认解释**预测类别**；本实验的主图统一解释**真实类别**。

## 2. 四个模型与结构审查

| 图中名称 | 实际模型 | 最佳轮次 | 主目标层 | 实测特征形状 |
|---|---|---:|---|---|
| ResNet-18 | 随机初始化的普通 ResNet-18 | 6 | `layer4[-1]` | `[1,512,7,7]` |
| ResNet-50 | 随机初始化的普通 ResNet-50 | 34 | `layer4[-1]` | `[1,2048,7,7]` |
| 原 CustomModel | ImageNet 预训练 ResNet-50 主干，仓库原注意力实现 | 16 | `backbone[-1][-1]` | `[1,2048,7,7]` |
| 空间修正版 | ImageNet 预训练主干，`CustomModelSpatialFixed` | 12 | `backbone[-1][-1]` | `[1,2048,7,7]` |

四个目标层都处在 ResNet 主干最后的残差阶段，实际梯度非零，生成的 CAM 均为有效二维图。原 `CustomModel` **在这个目标层之后**把两分支压缩至 `1×1`，使其后续 `SpatialAttention` 无法按空间位置加权。本实验没有把那个 `1×1` 注意力向量当作 CAM 特征图，而是解释其压缩之前的 `7×7` 主干特征。修正版虽然保留了 `7×7` 分支特征，但主对照也选同一语义阶段的主干末层，便于比较；因此这组 CAM 不等同于直接可视化修正版的空间注意力权重。

## 3. 样本选择

独立测试集为 254 张 Clean 图像、74 名患者，与训练和验证患者不交叉。随机主展示使用固定种子 **20261109**，分别抽取 8 名 HCC 患者和 8 名血管瘤患者，每名患者仅一张图。另从四模型已有测试预测中选取 3 张案例，独立标记为“预测结果选例”：`Hem_231`（修正版正确、原 CustomModel 错误）、`HCC_1565`（部分模型错误）、`HCC_454`（四模型均正确）。19 张图来自 19 名不同患者。没有满足条件的案例不会人为补造。

每张所选图像的源路径、患者 ID、真实类别、抽样依据及处理状态见 `gradcam_results/final/sample_manifest.csv`。读取时逐张校验原始文件 SHA-256 与清单一致，四个模型接收同一张原图与相同的输入张量。

## 4. CAM 计算和绘图过程

1. 严格加载四份 `best_model.pth`，设置 `eval()`；检查权重、类别索引、归一化和清单哈希。
2. 在指定目标层注册前向 hook 保存特征图，并在该输出张量注册梯度 hook。对**真实类别的原始 logit**反向传播；softmax 仅用于另行显示预测概率。
3. 将梯度在 `7×7` 空间上求平均，作为通道权重；加权求和特征图并执行 ReLU。使用双线性插值上采样到 `224×224`，之后对**每张热图单独**做 min-max 归一化。
4. 用固定 JET 颜色映射生成伪彩色热图，以固定透明度 **0.4** 与同一张 224×224 原图叠加。每张图形成“原图 + 四模型”五列，标出患者、真实类、CAM 目标、预测类、正误、预测置信度及 HCC 概率。
5. 若模型误判，额外以该模型**预测类别**的 logit 生成一张 CAM；它与主图的真实类别 CAM 分开保存。检查梯度缺失、非有限值、全零或常数热图，遇异常记录原因而不伪造热图。

所有 CAM 数值矩阵是 `float32`、`224×224` 的 `.npy` 文件，可供后续分析。由于每张热图独立归一化，**不同模型的红色深浅不能作为贡献强度的绝对比较**。

## 5. 运行结果与输出文件

批量运行处理 **19/19** 张图像，失败 **0** 张；四模型真实类别 CAM **76/76** 张有效；误判模型另保存 **7** 张预测类别 CAM。产生 19 张五列对照图、4 张批量总图。所有 CSV/JSON 引用的图片和矩阵路径均存在；`.npy` 尺寸均为 `224×224`，数值有限且归一化到 `[0,1]`。

最终目录：`gradcam_results/final/`。

| 文件或目录 | 内容 |
|---|---|
| `sample_manifest.csv` | 图像、患者、类别、原图路径、抽样依据、对应五列图、处理状态 |
| `predictions.csv` | 每张图的四模型预测、两类概率、置信度、正误、目标类别、目标层、CAM 有效性和各文件路径 |
| `experiment_config.json` | 种子、预处理、颜色/透明度、模型配置与权重哈希 |
| `target_layer_audit.json` | 四模型目标层实际形状、梯度强度及最佳轮次 |
| `completion.json` | 样本与 CAM 完成数、失败数 |
| `individual/Original/` | 统一尺寸的无叠加原图 |
| `individual/<模型>/` | 真实类别 CAM `.npy`、伪彩色图、叠加图；误判时还有预测类别 CAM |
| `comparisons/HCC/`、`Hemangioma/` | 随机主样本的逐图五列对照 |
| `comparisons/Misclassified/`、`Examples/` | 另选误判案例与全正确案例，和随机样本分开 |
| `overview/HCC_overview.png`、`Hemangioma_overview.png` | 两类各 8 张的五列总图 |
| `overview/Misclassified_overview.png`、`prediction_selected_overview.png` | 误判及非随机选例总图 |
| `report/gradcam_summary.md` | 机器汇总的运行摘要与异常记录 |

四模型本次单图推理的**预测类别**与先前保存的测试预测记录全部一致；HCC 概率的最大绝对差约 **0.0060**，两份数值均写入 `predictions.csv`，没有把它们强行视为完全相同。主展示组和人工选例混在一起的 19 张图**不能用于估计总体准确率**；准确率应查原有完整测试集的评估报告。

## 6. 初步视觉观察与限制

五列图显示四模型的高响应区域并不总一致。部分普通 ResNet 的 CAM 在超声扇形边缘、近场或图像外围较强；两个改进模型在一些图像上形成更局部的热区，但也存在对病灶以外组织、边界或深部区域响应较强的案例。`Hem_231` 案例中修正版预测正确、另外一些模型误判，但不能仅由热图推定其正确决策的医学原因。视觉观察需结合原图逐例阅读，不能从热图集中程度直接推断分类模型更优。

当前数据没有可信的病灶框或像素掩膜，因此**不能计算病灶定位准确率**，不能断言红色区域一定是病灶，更不能将 Grad-CAM 当作临床因果解释。四模型的主干初始化也不同：普通 ResNet 从零初始化，两个自定义模型使用 ImageNet 预训练；热图差异不只由注意力结构造成。

## 7. 代码、复现和验收

新增独立包 `gradcam_analysis/`：`models.py` 加载模型与目标层；`cam.py` 计算 Grad-CAM；`selection.py` 生成可重复的患者级样本及选例；`visualization.py` 渲染单图与总图；`generate_comparison.py` 统一调用；`README.md` 给出使用说明。未修改四个模型、其训练脚本和最佳权重。

从项目根目录运行：

```powershell
& 'C:\Users\lhx\miniconda3\envs\yolov5\python.exe' -u -m gradcam_analysis.generate_comparison --seed 20261109 --per-class 8 --output gradcam_results/final
```

单图模式可使用 `--image <当前Clean测试集图像文件名> --output <独立输出目录>`。已经分别对一张 HCC 和一张血管瘤测试图完成四模型冒烟验证；批量结果通过患者去重、目标层形状和梯度、CAM 有限值、全部路径存在、每图四模型齐全、预测类别记录一致等检查。按后续要求，`gradcam_results/` 中的 PNG 图像已单独加入 Git；`.npy` 矩阵及 CSV/JSON 明细仍留在本地，原始 `data/` 不上传。
