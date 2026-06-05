import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec

from config import DEVICE, OUTPUT_DIR, PATHS, PROJECT_DIR, WORD2VEC_CONFIG
from data_builder import build_knowledge_maps, token
from model import DiseaseTreatmentVectorModel


NAME_OVERRIDES = {
    "抱茎苦荬菜": "Ixeris sonchifolia",
}


def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def pca_2d(matrix):
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def transformed_display_radius(distances, min_radius, max_radius, gamma):
    values = np.asarray(distances, dtype=float)
    min_distance = float(values.min())
    max_distance = float(values.max())
    if max_distance == min_distance:
        return np.full_like(values, min_radius, dtype=float)
    scaled = (values - min_distance) / (max_distance - min_distance)
    return min_radius + (scaled ** gamma) * (max_radius - min_radius)


def resolve_disease_name(query, disease_to_targets, word2vec):
    disease_token = token("DISEASE", query)
    if disease_token in word2vec.wv:
        return query

    lower_map = {name.lower(): name for name in disease_to_targets}
    matched = lower_map.get(query.lower())
    if matched and token("DISEASE", matched) in word2vec.wv:
        return matched

    matches = [name for name in disease_to_targets if query.lower() in name.lower()]
    matches = [name for name in matches if token("DISEASE", name) in word2vec.wv]
    if matches:
        return sorted(matches, key=len)[0]
    raise ValueError(f"Disease not found in Word2Vec vocabulary: {query}")


def load_models(model_dir):
    paths = sorted(Path(model_dir).glob("fold_*/best_model.pth"))
    if not paths:
        raise FileNotFoundError(f"No fold best_model.pth files found under {model_dir}")

    models = []
    for path in paths:
        model = DiseaseTreatmentVectorModel(vector_size=WORD2VEC_CONFIG["vector_size"]).to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        models.append(model)
    return models


def ensemble_treatment_vector(disease_vec, model_dir):
    disease_tensor = torch.tensor(disease_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    vectors = []
    with torch.no_grad():
        for model in load_models(model_dir):
            vectors.append(model.treatment_vector(disease_tensor).detach().cpu().numpy()[0])
    return normalize_vector(np.mean(vectors, axis=0))


def compute_distances(disease, model_dir):
    herbs, _, _, disease_to_targets = build_knowledge_maps()
    word2vec = Word2Vec.load(str(PATHS["word2vec"]))
    matched_disease = resolve_disease_name(disease, disease_to_targets, word2vec)
    disease_vec = word2vec.wv[token("DISEASE", matched_disease)]
    anchor_vec = ensemble_treatment_vector(disease_vec, model_dir)

    rows = []
    vectors = []
    for herb, meta in herbs.items():
        herb_token = token("HERB", herb)
        if herb_token not in word2vec.wv:
            continue
        herb_vec = normalize_vector(word2vec.wv[herb_token])
        cos_sim = cosine_similarity(anchor_vec, herb_vec)
        rows.append(
            {
                "herb": herb,
                "herb_id": meta.get("herb_id", ""),
                "english_name": meta.get("english_name", ""),
                "latin_name": meta.get("latin_name", ""),
                "cosine_similarity": cos_sim,
                "cosine_distance": 1.0 - cos_sim,
                "_vector_index": len(vectors),
            }
        )
        vectors.append(herb_vec)

    distances = pd.DataFrame(rows).sort_values("cosine_distance", ascending=True).reset_index(drop=True)
    sorted_vectors = np.vstack(vectors)[distances["_vector_index"].to_numpy(dtype=int)]
    distances = distances.drop(columns=["_vector_index"])
    return distances, sorted_vectors, normalize_vector(anchor_vec)


def display_herb_name(row):
    herb = str(row.get("herb", "")).strip()
    if herb in NAME_OVERRIDES:
        return NAME_OVERRIDES[herb]
    english_name = str(row.get("english_name", "")).strip()
    latin_name = str(row.get("latin_name", "")).strip()
    if english_name and english_name.lower() != "nan":
        return english_name
    if latin_name and latin_name.lower() != "nan":
        return latin_name
    return herb


def plot_distance(
    distances,
    herb_vectors,
    anchor_vec,
    output_dir,
    disease_name,
    top_k,
    label_top_n,
    min_display_radius,
    max_display_radius,
    display_gamma,
    dpi,
):
    plot_df = distances.copy()
    top_df = plot_df.head(top_k).copy()
    all_vectors = np.vstack([herb_vectors, anchor_vec.reshape(1, -1)])
    coords = pca_2d(all_vectors)
    raw_x = coords[:-1, 0] - coords[-1, 0]
    raw_y = coords[:-1, 1] - coords[-1, 1]
    angles = np.arctan2(raw_y, raw_x)
    missing_angle = np.abs(raw_x) + np.abs(raw_y) < 1e-12
    if missing_angle.any():
        angles[missing_angle] = np.linspace(0, 2 * np.pi, int(missing_angle.sum()), endpoint=False)

    radii = transformed_display_radius(
        plot_df["cosine_distance"],
        min_radius=min_display_radius,
        max_radius=max_display_radius,
        gamma=display_gamma,
    )
    plot_df["x"] = radii * np.cos(angles)
    plot_df["y"] = radii * np.sin(angles)
    top_df = top_df.merge(plot_df[["herb", "x", "y"]], on="herb", how="left")

    plt.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "font.size": 16,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelweight": "bold",
            "font.weight": "bold",
        }
    )

    fig, ax = plt.subplots(figsize=(13.2, 10.2))
    ax.scatter(
        plot_df["x"],
        plot_df["y"],
        s=48,
        c="#6d28d9",
        alpha=0.42,
        edgecolors="none",
        label="All herbs",
        zorder=1,
    )
    ax.scatter(
        [0.0],
        [0.0],
        s=155,
        c="#00bcd4",
        marker="D",
        edgecolor="black",
        linewidth=1.1,
        label=disease_name,
        zorder=5,
    )

    x_span = max(plot_df["x"].max() - plot_df["x"].min(), 1e-6)
    y_span = max(plot_df["y"].max() - plot_df["y"].min(), 1e-6)
    offsets = [(-0.39, 0.33), (-0.36, -0.29), (0.41, 0.30)]
    for rank, (_, row) in enumerate(top_df.iterrows(), start=1):
        if rank > label_top_n:
            continue
        dx, dy = offsets[(rank - 1) % len(offsets)]
        ax.annotate(
            f"Rank {rank}: {display_herb_name(row)}\nd={row['cosine_distance']:.3f}",
            xy=(row["x"], row["y"]),
            xytext=(row["x"] + dx * x_span, row["y"] + dy * y_span),
            fontsize=18,
            fontweight="bold",
            ha="center",
            va="center",
            arrowprops={
                "arrowstyle": "->",
                "color": "#ef4444",
                "lw": 1.5,
                "alpha": 0.9,
                "shrinkA": 2,
                "shrinkB": 4,
            },
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.74},
            zorder=6,
        )

    ax.annotate(
        disease_name,
        xy=(0.0, 0.0),
        xytext=(0.13 * x_span, 0.12 * y_span),
        fontsize=20,
        fontweight="bold",
        color="#0891b2",
        arrowprops={"arrowstyle": "->", "color": "#06b6d4", "lw": 1.8},
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.75},
        zorder=7,
    )

    ax.set_xlabel("display projection axis 1", fontsize=24, fontweight="bold")
    ax.set_ylabel("display projection axis 2", fontsize=24, fontweight="bold")
    ax.tick_params(axis="both", labelsize=18, width=1.5, length=6)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    legend = ax.legend(
        loc="best",
        fontsize=20,
        frameon=True,
        fancybox=True,
        edgecolor="#b8b8b8",
        facecolor="white",
        framealpha=0.92,
        borderpad=0.45,
        handletextpad=0.7,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")
    legend.get_frame().set_linewidth(0.6)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "diabetic_nephropathy_top3_herb_distance.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return png_path


def display_path(path):
    try:
        return str(Path(path).resolve().relative_to(PROJECT_DIR.resolve()))
    except ValueError:
        return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", default="Diabetic Nephropathy")
    parser.add_argument(
        "--model-dir",
        default=str(OUTPUT_DIR / "cv_runs" / "DDTA_Pred"),
        help="Directory containing fold_*/best_model.pth.",
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR / "figures"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--label-top-n", type=int, default=3)
    parser.add_argument("--min-display-radius", type=float, default=0.06)
    parser.add_argument("--max-display-radius", type=float, default=0.95)
    parser.add_argument("--display-gamma", type=float, default=0.45)
    parser.add_argument("--dpi", type=int, default=900)
    args = parser.parse_args()

    distances, herb_vectors, anchor_vec = compute_distances(args.disease, args.model_dir)
    png_path = plot_distance(
        distances,
        herb_vectors,
        anchor_vec,
        Path(args.output_dir),
        args.disease,
        args.top_k,
        args.label_top_n,
        args.min_display_radius,
        args.max_display_radius,
        args.display_gamma,
        args.dpi,
    )
    print(distances.head(args.top_k).to_string(index=False))
    print(f"Saved: {display_path(png_path)}")


if __name__ == "__main__":
    main()
