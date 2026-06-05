from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

from config import OUTPUT_DIR, PROJECT_DIR


def display_path(path):
    try:
        return str(Path(path).resolve().relative_to(PROJECT_DIR.resolve()))
    except ValueError:
        return str(path)


def main():
    output_dir = OUTPUT_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    herb_points = np.array(
        [
            [0.30, 0.34],
            [0.41, 0.35],
            [0.51, 0.29],
            [0.35, 0.49],
            [0.50, 0.44],
        ]
    )
    disease_points = np.array(
        [
            [0.65, 0.70],
            [0.79, 0.75],
            [0.87, 0.65],
            [0.69, 0.88],
            [0.91, 0.85],
        ]
    )

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "axes.linewidth": 1.1,
        }
    )

    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    arrowprops = {
        "arrowstyle": "-|>",
        "lw": 1.2,
        "color": "black",
        "shrinkA": 0,
        "shrinkB": 0,
        "mutation_scale": 12,
    }
    ax.annotate("", xy=(0.96, 0.08), xytext=(0.08, 0.08), arrowprops=arrowprops)
    ax.annotate("", xy=(0.08, 0.96), xytext=(0.08, 0.08), arrowprops=arrowprops)
    ax.text(0.99, 0.055, "x", ha="center", va="center", fontsize=12)
    ax.text(0.055, 0.99, "y", ha="center", va="center", fontsize=12)

    herb_color = "#5aa05a"
    disease_color = "#f06464"
    herb_ellipse = Ellipse(
        xy=(0.41, 0.38),
        width=0.50,
        height=0.34,
        angle=-20,
        fill=False,
        edgecolor=herb_color,
        linewidth=1.0,
        linestyle=(0, (4, 3)),
    )
    disease_ellipse = Ellipse(
        xy=(0.79, 0.77),
        width=0.47,
        height=0.34,
        angle=18,
        fill=False,
        edgecolor=disease_color,
        linewidth=1.0,
        linestyle=(0, (4, 3)),
    )
    ax.add_patch(herb_ellipse)
    ax.add_patch(disease_ellipse)

    ax.scatter(
        herb_points[:, 0],
        herb_points[:, 1],
        s=58,
        color=herb_color,
        alpha=0.9,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    ax.scatter(
        disease_points[:, 0],
        disease_points[:, 1],
        s=58,
        color=disease_color,
        alpha=0.9,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )

    ax.text(0.56, 0.99, "disease vectors", color=disease_color, fontsize=11, fontweight="bold")
    ax.text(0.25, 0.14, "herb vectors", color="#2f7d32", fontsize=11, fontweight="bold")

    png_path = output_dir / "word2vec_embedding_schematic.png"
    pdf_path = output_dir / "word2vec_embedding_schematic.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"Saved: {display_path(png_path)}")
    print(f"Saved: {display_path(pdf_path)}")


if __name__ == "__main__":
    main()
