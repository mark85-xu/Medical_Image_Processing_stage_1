# 十组配对模型 Grad-CAM 实验

仅使用 `patient_split_comparison/outputs/` 中已训练完成的十组原 `CustomModel` 与空间修正版检查点。每个划分在自己的测试患者中固定抽取 HCC/血管瘤各 8 人、每人一张图；额外预测分歧案例单独标记。两版对同一张图使用该划分训练集的同一标准化参数，并对真实类别 logit 在主干末层 `7×7` 特征上计算 Grad-CAM。误判时另存预测类别 CAM。空间注意力 sigmoid 权重只作为独立的结构审计，不称为 Grad-CAM。

从项目根目录运行，使用原有 `yolov5` 环境：

```powershell
& 'C:\Users\lhx\miniconda3\envs\yolov5\python.exe' -u patient_split_comparison/gradcam_study/run_gradcam_study.py --seed 1001 --smoke
& 'C:\Users\lhx\miniconda3\envs\yolov5\python.exe' -u patient_split_comparison/gradcam_study/run_gradcam_study.py
```

可用 `--seed 1001` 只运行指定划分。输出位于 `outputs/`，已完成图像在配置及文件检查后可复用。每张主目标 CAM 同时保存 `7×7` 原生网格和 `224×224` 上采样矩阵；误判时另存预测类别 CAM 矩阵和叠加图。`outputs/summary/` 包含十组汇总和中文报告。十组测试患者有重叠，空间图相似度只描述响应位置差异，不能作为病灶定位质量或模型优劣的指标。
