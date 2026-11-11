"""Render fixed-style Grad-CAM heatmaps and five-column comparisons."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image


ALPHA = 0.4
COLORMAP = "jet"
MODEL_TITLES = ("ResNet-18", "ResNet-50", "原 CustomModel", "空间修正版")
FONT_PATH = Path(r"C:\Windows\Fonts\NotoSansSC-Regular.ttf")
if FONT_PATH.is_file():
    font_manager.fontManager.addfont(str(FONT_PATH))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
plt.rcParams["axes.unicode_minus"] = False


def render_cam(rgb, cam):
    if rgb.shape != (224, 224, 3) or cam.shape != (224, 224):
        raise ValueError("Expected 224x224 RGB image and CAM")
    color = plt.get_cmap(COLORMAP)(cam)[..., :3]
    heatmap = np.uint8(np.clip(color * 255, 0, 255))
    overlay = np.uint8(np.clip((1 - ALPHA) * rgb.astype(np.float32) +
                               ALPHA * heatmap.astype(np.float32), 0, 255))
    return heatmap, overlay


def save_png(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(str(path))


def comparison(path, item, results, image_path, title=None):
    fig, axes = plt.subplots(1, 5, figsize=(17.5, 4.4), dpi=150)
    original = np.asarray(Image.open(str(image_path)).convert("RGB"))
    axes[0].imshow(original)
    axes[0].set_title("原始超声图", fontsize=13)
    axes[0].set_xlabel("真实：{}".format(item["class_label"]), fontsize=10)
    for index, result in enumerate(results):
        ax = axes[index + 1]
        if result["overlay_path"]:
            ax.imshow(Image.open(str(result["overlay_path"])))
        else:
            ax.imshow(original, cmap="gray")
            ax.text(.5, .5, "CAM 无效", transform=ax.transAxes, ha="center", va="center",
                    color="red", fontsize=14, bbox={"facecolor": "white", "alpha": .8})
        ax.set_title(MODEL_TITLES[index], fontsize=12)
        status = "正确" if result["correct"] else "错误"
        ax.set_xlabel("预测：{}（{}）\n置信度={:.3f}；P(HCC)={:.3f}".format(
            result["predicted_label"], status, result["confidence"],
            result["prob_hcc"]), fontsize=9)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    fig.suptitle(title or "患者 {} / 真实 {} / CAM 目标 {}".format(
        item["patient_id"], item["class_label"], item["class_label"]), fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .91], w_pad=.5)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)


def overview(path, items, result_by_image, original_by_image, name):
    if not items:
        return
    fig, axes = plt.subplots(len(items), 5, figsize=(15.5, max(3.5, 3.0 * len(items))),
                             dpi=130, squeeze=False)
    headings = ("原始图像",) + MODEL_TITLES
    for col, heading in enumerate(headings):
        axes[0, col].set_title(heading, fontsize=11)
    for row_index, item in enumerate(items):
        image_name = item["image_filename"]
        original = Image.open(str(original_by_image[image_name]))
        axes[row_index, 0].imshow(original)
        axes[row_index, 0].set_ylabel("{}\n{}".format(item["patient_id"], item["class_label"]), fontsize=9)
        for col, result in enumerate(result_by_image[image_name], start=1):
            ax = axes[row_index, col]
            if result["overlay_path"]:
                ax.imshow(Image.open(str(result["overlay_path"])))
            else:
                ax.imshow(original)
                ax.text(.5, .5, "CAM 无效", transform=ax.transAxes, ha="center", va="center",
                        color="red", fontsize=10)
            ax.set_xlabel("{}{} / 置信度={:.2f}".format(
                result["predicted_label"], "（错）" if not result["correct"] else "",
                result["confidence"]), fontsize=8)
        for ax in axes[row_index]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
    fig.suptitle(name, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, .98], h_pad=.9, w_pad=.4)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)
