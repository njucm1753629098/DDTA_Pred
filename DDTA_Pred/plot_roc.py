import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

from config import OUTPUT_DIR, PROJECT_DIR


def load_predictions(run_dir):
    summary_path = run_dir / "fold_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing fold summary: {summary_path}")

    summary = pd.read_csv(summary_path)
    frames = []
    for fold in summary["fold"].astype(int):
        prediction_path = run_dir / f"fold_{fold:02d}" / "best_val_predictions.csv"
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing fold prediction file: {prediction_path}")
        frame = pd.read_csv(prediction_path)
        frame["fold"] = fold
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def plot_roc(run_dir, output_dir, dpi):
    predictions = load_predictions(run_dir)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelweight": "bold",
            "font.weight": "bold",
        }
    )

    fig, ax = plt.subplots(figsize=(10.0, 7.2))
    roc_rows = []
    colors = plt.cm.tab10(np.linspace(0, 1, predictions["fold"].nunique()))
    for color, (fold, fold_df) in zip(colors, predictions.groupby("fold")):
        fpr, tpr, _ = roc_curve(fold_df["label"], fold_df["probability"])
        fold_auc = auc(fpr, tpr)
        roc_rows.append({"fold": int(fold), "roc_auc": fold_auc})
        ax.plot(fpr, tpr, linewidth=2.3, color=color, label=f"Fold {int(fold):02d} AUC={fold_auc:.3f}")

    mean_auc = float(np.mean([row["roc_auc"] for row in roc_rows]))
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1.8)
    ax.set_xlabel("False positive rate", fontsize=18, fontweight="bold")
    ax.set_ylabel("True positive rate", fontsize=18, fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(axis="both", labelsize=15, width=1.6, length=6)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    legend = ax.legend(
        loc="lower right",
        fontsize=13,
        title=f"Mean AUC = {mean_auc:.4f}",
        title_fontsize=13,
        frameon=True,
        fancybox=True,
        framealpha=0.92,
    )
    legend.get_title().set_fontweight("bold")
    for text in legend.get_texts():
        text.set_fontweight("bold")
    legend.get_frame().set_edgecolor("#b8b8b8")
    legend.get_frame().set_linewidth(1.0)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "roc_curves_only_mean_auc.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return png_path, mean_auc


def display_path(path):
    try:
        return str(Path(path).resolve().relative_to(PROJECT_DIR.resolve()))
    except ValueError:
        return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default=str(OUTPUT_DIR / "cv_runs" / "DDTA_Pred"),
        help="Directory containing fold_summary.csv and fold_*/best_val_predictions.csv.",
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR / "figures"))
    parser.add_argument("--dpi", type=int, default=900)
    args = parser.parse_args()

    png_path, mean_auc = plot_roc(Path(args.run_dir), Path(args.output_dir), args.dpi)
    print(f"Mean AUC: {mean_auc:.4f}")
    print(f"Saved: {display_path(png_path)}")


if __name__ == "__main__":
    main()
