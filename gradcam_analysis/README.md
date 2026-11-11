# 四模型 Grad-CAM 对照程序

使用已训练的四个最佳检查点，不训练或修改模型。支持单张测试图像和固定种子的批量患者抽样。

## 运行

从项目根目录使用既有 `yolov5` 环境：

```powershell
& 'C:\Users\lhx\miniconda3\envs\yolov5\python.exe' -m gradcam_analysis.generate_comparison --image 测试集图像文件名 --output gradcam_results/smoke
& 'C:\Users\lhx\miniconda3\envs\yolov5\python.exe' -m gradcam_analysis.generate_comparison --seed 20261109 --per-class 8 --output gradcam_results/final
```

也可用 `--device cpu`，但运行更慢。`--image` 只能接受当前 Clean 测试集中的文件名或审计路径。已有输出目录的配置如与新命令不同，程序会拒绝混用。

## 数据与目标层

读取 `data/metadata/image_manifest.csv` 所标记的患者级 Clean 测试集，逐张校验源图 SHA-256。四个模型使用各自 `run_config.json` 记录的 RGB 224×224 标准化参数，并校验其一致性和类别映射 `HCC=0, Hemangioma=1`。加载 `best_model.pth`，严格检查权重与模型结构匹配。

主对照的目标层均为 ResNet 主干最后一个残差块：标准模型 `layer4[-1]`，自定义模型 `backbone[-1][-1]`。ResNet-18 预期 `[1,512,7,7]`，其余三个预期 `[1,2048,7,7]`；实际形状和梯度写入 `target_layer_audit.json`。原 `CustomModel` 后续会压成 `1×1`，本实验解释的是其压缩前的最后一层空间特征。

## 计算与输出

采用标准 Grad-CAM：对真实类别的原始 logit 求梯度，对梯度做空间平均，加权特征图、ReLU、双线性上采样至 224×224，再按单张热图最小/最大值归一化。误分类样本另计算预测类别的 CAM。使用 PyTorch 模块前向 hook 和张量梯度 hook，参考仓库已有 `GradCAM` 类；不安装额外库。全零、常数、非有限或缺失梯度均记录为无效，不用假热图替代。

默认从测试集按类别抽取 8 名 HCC 患者和 8 名血管瘤患者，每人一张图；另按四模型已有测试预测记录选取少量对照案例，单独标记为非随机样本。输出：

- `sample_manifest.csv`：图像、患者、抽样依据、处理状态；
- `predictions.csv`：四模型预测、两类概率、CAM 目标/层/有效性和文件路径；
- `experiment_config.json`、`target_layer_audit.json`、`completion.json`；
- `individual/Original/`：统一 224×224 原图；`individual/<模型>/`：归一化 CAM `.npy`、伪彩色图、叠加图及错误预测类别 CAM；
- `comparisons/`：每张图一行五列对照；`overview/`：按 HCC、血管瘤、误判案例及非随机选例生成的总图；
- `report/gradcam_summary.md`：本次运行的配置、有效热图数量和解释边界。

颜色映射固定为 JET，热图叠加系数 0.4。每张 CAM 独立归一化，因此跨模型颜色深浅不代表绝对贡献强度。没有病灶 ROI 标注，不能据此计算病灶定位准确率。
