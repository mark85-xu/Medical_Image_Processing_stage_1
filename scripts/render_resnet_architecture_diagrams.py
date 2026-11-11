"""Render the two trained ResNet-50 data-flow diagrams as PNG files."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
FONT = FontProperties(fname=r"C:\Windows\Fonts\NotoSansSC-Regular.ttf")
FONT_BOLD = FontProperties(fname=r"C:\Windows\Fonts\Noto Sans SC Bold (TrueType).otf")
BG = "#f7f9fc"
INK = "#213044"
BLUE = "#dceafa"
BLUE_EDGE = "#5c87b9"
TEAL = "#d9f2eb"
TEAL_EDGE = "#3c9b82"
ORANGE = "#fff0d8"
ORANGE_EDGE = "#db9448"
NEUTRAL = "#e9edf3"
NEUTRAL_EDGE = "#8594a8"


def setup(width, height, title, subtitle):
    fig, ax = plt.subplots(figsize=(width, height), dpi=170)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 17)
    ax.axis("off")
    ax.text(7, 16.5, title, ha="center", va="center", fontproperties=FONT_BOLD,
            fontsize=21, color=INK)
    ax.text(7, 15.95, subtitle, ha="center", va="center", fontproperties=FONT,
            fontsize=11, color="#56667c")
    return fig, ax


def box(ax, x, y, w, h, title, detail, fill=BLUE, edge=BLUE_EDGE, font=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.015,rounding_size=0.15",
                                linewidth=1.5, edgecolor=edge, facecolor=fill))
    ax.text(x + w / 2, y + h * .66, title, ha="center", va="center",
            fontproperties=FONT_BOLD, fontsize=font, color=INK)
    ax.text(x + w / 2, y + h * .31, detail, ha="center", va="center",
            fontproperties=FONT, fontsize=font - 1, color="#405169")


def arrow(ax, start, end, color="#66788e", width=1.8):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15,
                                 linewidth=width, color=color,
                                 connectionstyle="arc3,rad=0"))


def ordinary():
    fig, ax = setup(11.7, 14.2, "普通 ResNet-50：从输入到输出",
                    "本次基线：随机初始化主干 · 张量尺寸为 通道×高×宽")
    items = [
        (14.55, "输入超声图像", "RGB · 3×224×224", NEUTRAL, NEUTRAL_EDGE),
        (13.20, "卷积起始层", "7×7 Conv + BN + ReLU · 64×112×112", BLUE, BLUE_EDGE),
        (11.85, "最大池化", "3×3 MaxPool · 64×56×56", BLUE, BLUE_EDGE),
        (10.50, "layer1 · 3 个残差 Bottleneck", "256×56×56", BLUE, BLUE_EDGE),
        (9.15, "layer2 · 4 个残差 Bottleneck", "512×28×28", BLUE, BLUE_EDGE),
        (7.80, "layer3 · 6 个残差 Bottleneck", "1024×14×14", BLUE, BLUE_EDGE),
        (6.45, "layer4 · 3 个残差 Bottleneck", "2048×7×7", BLUE, BLUE_EDGE),
        (5.10, "全局平均池化", "每通道 7×7 → 1×1 · 2048×1×1", TEAL, TEAL_EDGE),
        (3.75, "展平 + 全连接层", "2048 → 2", TEAL, TEAL_EDGE),
        (2.40, "输出 logits", "HCC、血管瘤各 1 个分数", TEAL, TEAL_EDGE),
    ]
    for idx, (y, title, detail, fill, edge) in enumerate(items):
        box(ax, 3.25, y, 7.5, .95, title, detail, fill, edge)
        if idx:
            prev_y = items[idx - 1][0]
            arrow(ax, (7, prev_y), (7, y + .98))
    ax.text(7, 1.37, "Bottleneck：1×1 → 3×3 → 1×1 卷积，并与残差分支相加",
            ha="center", va="center", fontproperties=FONT, fontsize=10.5,
            color="#56667c")
    ax.text(7, .87, "评估时：softmax(logits) → HCC 概率；softmax 不是可训练层",
            ha="center", va="center", fontproperties=FONT, fontsize=10.5,
            color="#56667c")
    fig.savefig(OUT / "resnet50_original.png", bbox_inches="tight", pad_inches=.35)
    plt.close(fig)


def improved():
    fig, ax = setup(13.6, 14.2, "改进 ResNet-50：实际运行的数据流",
                    "本次模型：ImageNet 预训练主干 · 按 model.py 的 forward 真实张量尺寸绘制")
    shared = [
        (14.55, "输入超声图像", "RGB · 3×224×224", NEUTRAL, NEUTRAL_EDGE),
        (13.20, "ResNet-50 主干", "起始层 + layer1–4 · 2048×7×7", BLUE, BLUE_EDGE),
        (11.85, "额外 3×3 卷积", "2048×7×7 → 1024×7×7", BLUE, BLUE_EDGE),
    ]
    for idx, (y, title, detail, fill, edge) in enumerate(shared):
        box(ax, 3.25, y, 7.5, .95, title, detail, fill, edge)
        if idx:
            arrow(ax, (7, shared[idx - 1][0]), (7, y + .98))

    box(ax, .65, 9.85, 5.75, .95, "分支 1 · SEBlock", "1024×7×7 → 1024×7×7", ORANGE, ORANGE_EDGE)
    box(ax, 7.6, 9.85, 5.75, .95, "分支 2 · ChannelAttention", "仅返回权重 · 1024×1×1", ORANGE, ORANGE_EDGE)
    box(ax, .65, 8.35, 5.75, .95, "ChannelAttention", "仅返回权重 · 1024×1×1", ORANGE, ORANGE_EDGE)
    box(ax, 7.6, 8.35, 5.75, .95, "SEBlock", "1024×1×1 → 1024×1×1", ORANGE, ORANGE_EDGE)
    arrow(ax, (7, 11.85), (3.52, 10.85), ORANGE_EDGE)
    arrow(ax, (7, 11.85), (10.48, 10.85), ORANGE_EDGE)
    arrow(ax, (3.52, 9.85), (3.52, 9.33), ORANGE_EDGE)
    arrow(ax, (10.48, 9.85), (10.48, 9.33), ORANGE_EDGE)

    box(ax, 3.25, 6.85, 7.5, .95, "按通道拼接两分支", "1024×1×1 + 1024×1×1 → 2048×1×1",
        ORANGE, ORANGE_EDGE)
    arrow(ax, (3.52, 8.35), (5.5, 7.83), ORANGE_EDGE)
    arrow(ax, (10.48, 8.35), (8.5, 7.83), ORANGE_EDGE)
    box(ax, 3.25, 5.45, 7.5, .95, "SpatialAttention", "输入已为 1×1 · 输出 2048×1×1", ORANGE, ORANGE_EDGE)
    arrow(ax, (7, 6.85), (7, 6.43), ORANGE_EDGE)
    box(ax, 3.25, 4.05, 7.5, .95, "全局平均池化 + 展平", "2048×1×1 → 2048", TEAL, TEAL_EDGE)
    arrow(ax, (7, 5.45), (7, 5.03))
    box(ax, 3.25, 2.65, 7.5, .95, "全连接层 → 输出 logits", "2048 → 2 · HCC / 血管瘤", TEAL, TEAL_EDGE)
    arrow(ax, (7, 4.05), (7, 3.63))
    ax.text(7, 1.72, "两条分支共享同一组 SEBlock 和 ChannelAttention 参数，只是调用顺序不同。",
            ha="center", va="center", fontproperties=FONT, fontsize=10.5, color="#56667c")
    ax.text(7, 1.16, "注意：ChannelAttention 没有把权重乘回 7×7 特征图；空间信息在拼接前已压成 1×1。",
            ha="center", va="center", fontproperties=FONT, fontsize=10.5, color="#9b6028")
    fig.savefig(OUT / "resnet50_attention_actual.png", bbox_inches="tight", pad_inches=.35)
    plt.close(fig)


if __name__ == "__main__":
    ordinary()
    improved()
    print(OUT / "resnet50_original.png")
    print(OUT / "resnet50_attention_actual.png")
