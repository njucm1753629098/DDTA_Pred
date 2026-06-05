from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

PATHS = {
    "herb_compound": DATA_DIR / "hit2_herbs_ingredients.csv",
    "compound_target": DATA_DIR / "hit2_ingredients_targets.csv",
    "disease_target": DATA_DIR / "disgenet_target_disease.csv",
    "ppi": DATA_DIR / "combine_score.tsv",
    "manual_herb_compound": DATA_DIR / "manual_kudiezi_compounds_template.csv",
    "manual_compound_target": DATA_DIR / "kudiezi_compounds_targets.csv",
    "word2vec": OUTPUT_DIR / "word2vec" / "tcm_disease_herb_word2vec.model",
    "samples": OUTPUT_DIR / "training_data" / "herb_disease_samples.csv",
}

WORD2VEC_CONFIG = {
    "vector_size": 128,
    "window": 5,
    "min_count": 1,
    "workers": 4,
    "epochs": 40,
    "sg": 1,
    "seed": 42,
}

SAMPLE_CONFIG = {
    "max_diseases": 800,
    "min_disease_targets": 3,
    "min_herb_targets": 2,
    "positive_quantile": 0.92,
    "negative_quantile": 0.55,
    "max_negative_ratio": 1.5,
    "hard_negative_fraction": 0.35,
    "ppi_max_pairs_per_sample": 1600,
    "random_seed": 42,
}

TRAIN_CONFIG = {
    "folds": 10,
    "epochs": 80,
    "batch_size": 128,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 12,
    "target_mean_auroc": 0.96,
    "seed": 42,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
