from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from config import OUTPUT_DIR, PROJECT_DIR


def display_path(path):
    try:
        return str(Path(path).resolve().relative_to(PROJECT_DIR.resolve()))
    except ValueError:
        return str(path)


def rounded_box(ax, xy, width, height, text, fc, ec, fontsize=10, weight="normal"):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.1,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#111827",
    )
    return box


def arrow(ax, start, end, color="#1f4e79", lw=1.5, style="-|>", rad=0.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=13,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=3,
        shrinkB=3,
    )
    ax.add_patch(patch)
    return patch


def draw_mlp(ax, x, y, w, h, title, layer_labels, colors):
    rounded_box(ax, (x, y), w, h, "", "#f8fafc", "#8ab6d6")
    ax.text(x + w / 2, y + h - 0.045, title, ha="center", va="top", fontsize=10.5, fontweight="bold", color="#0b3a75")

    left = x + 0.10 * w
    right = x + 0.90 * w
    top = y + 0.70 * h
    bottom = y + 0.24 * h
    layer_x = [left + i * (right - left) / (len(layer_labels) - 1) for i in range(len(layer_labels))]

    for i in range(len(layer_x) - 1):
        for yy1 in [bottom, (bottom + top) / 2, top]:
            for yy2 in [bottom, (bottom + top) / 2, top]:
                ax.plot([layer_x[i], layer_x[i + 1]], [yy1, yy2], color="#94a3b8", lw=0.45, alpha=0.55, zorder=1)

    for lx, label, color in zip(layer_x, layer_labels, colors):
        for yy in [bottom, (bottom + top) / 2, top]:
            ax.scatter(lx, yy, s=95, color=color, edgecolor="#334155", linewidth=0.55, zorder=3)
        ax.text(lx, y + 0.10 * h, label, ha="center", va="center", fontsize=8.5, color="#334155")


def main():
    output_dir = OUTPUT_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(12.2, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.96, "DDTA_Pred neural architecture", fontsize=16, fontweight="bold", color="#111827", va="top")

    disease_box = rounded_box(
        ax,
        (0.04, 0.63),
        0.16,
        0.12,
        "Disease vector\n128-d",
        "#fff1f2",
        "#ef4444",
        fontsize=11,
        weight="bold",
    )
    herb_box = rounded_box(
        ax,
        (0.04, 0.24),
        0.16,
        0.12,
        "Herb vector\n128-d",
        "#ecfdf5",
        "#22c55e",
        fontsize=11,
        weight="bold",
    )

    draw_mlp(
        ax,
        0.26,
        0.51,
        0.26,
        0.32,
        "Treatment-vector mapper",
        ["128", "384", "384", "128"],
        ["#fecaca", "#bfdbfe", "#bfdbfe", "#bbf7d0"],
    )
    treatment_box = rounded_box(
        ax,
        (0.58, 0.61),
        0.17,
        0.14,
        "Predicted\ntreatment vector\n128-d",
        "#f0fdf4",
        "#16a34a",
        fontsize=10.5,
        weight="bold",
    )

    feature_box = rounded_box(
        ax,
        (0.28, 0.19),
        0.29,
        0.21,
        "Pairwise features\n[disease, herb,\ndisease x herb,\n|disease - herb|,\ncosine score]",
        "#f8fafc",
        "#64748b",
        fontsize=9.6,
    )

    draw_mlp(
        ax,
        0.63,
        0.16,
        0.22,
        0.30,
        "Interaction head",
        ["513", "384", "192", "1"],
        ["#ddd6fe", "#bfdbfe", "#bfdbfe", "#fde68a"],
    )

    cosine_box = rounded_box(
        ax,
        (0.79, 0.60),
        0.14,
        0.12,
        "Cosine score\nwith herb vector",
        "#eff6ff",
        "#2563eb",
        fontsize=9.8,
    )
    output_box = rounded_box(
        ax,
        (0.88, 0.30),
        0.10,
        0.17,
        "Final logit\n+\nprobability",
        "#fffbeb",
        "#f59e0b",
        fontsize=10,
        weight="bold",
    )

    arrow(ax, (0.20, 0.69), (0.26, 0.67))
    arrow(ax, (0.52, 0.67), (0.58, 0.68))
    arrow(ax, (0.75, 0.68), (0.79, 0.66))
    arrow(ax, (0.20, 0.30), (0.28, 0.30), color="#15803d")
    arrow(ax, (0.14, 0.63), (0.28, 0.39), color="#64748b", rad=-0.10)
    arrow(ax, (0.57, 0.30), (0.63, 0.30), color="#64748b")
    arrow(ax, (0.85, 0.31), (0.88, 0.36), color="#64748b")
    arrow(ax, (0.86, 0.60), (0.90, 0.47), color="#2563eb")
    arrow(ax, (0.14, 0.36), (0.80, 0.60), color="#2563eb", rad=0.18)

    ax.text(0.44, 0.47, "LayerNorm + GELU + Dropout", ha="center", fontsize=9, color="#475569")
    ax.text(0.70, 0.10, "LayerNorm + GELU + Dropout", ha="center", fontsize=9, color="#475569")
    ax.text(0.83, 0.55, "temperature-scaled", ha="center", fontsize=8.8, color="#2563eb")

    footer = rounded_box(
        ax,
        (0.08, 0.035),
        0.84,
        0.055,
        "Input: disease-herb vector pair    |    Learn: disease-specific treatment direction    |    Output: treatment association score",
        "#ffffff",
        "#cbd5e1",
        fontsize=10,
        weight="bold",
    )
    footer.set_linewidth(0.9)

    png_path = output_dir / "DDTA_Pred_neural_architecture.png"
    pdf_path = output_dir / "DDTA_Pred_neural_architecture.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"Saved: {display_path(png_path)}")
    print(f"Saved: {display_path(pdf_path)}")


if __name__ == "__main__":
    main()
