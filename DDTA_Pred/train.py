import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import DEVICE, OUTPUT_DIR, PATHS, PROJECT_DIR, TRAIN_CONFIG, WORD2VEC_CONFIG
from data_builder import token
from model import DiseaseTreatmentVectorModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HerbDiseaseVectorDataset(Dataset):
    def __init__(self, frame, word2vec):
        rows = []
        for _, row in frame.iterrows():
            disease_token = token("DISEASE", row["disease"])
            herb_token = token("HERB", row["herb"])
            if disease_token not in word2vec.wv or herb_token not in word2vec.wv:
                continue
            rows.append(
                {
                    "disease": row["disease"],
                    "herb": row["herb"],
                    "disease_vec": word2vec.wv[disease_token].astype(np.float32),
                    "herb_vec": word2vec.wv[herb_token].astype(np.float32),
                    "label": np.array([float(row["label"])], dtype=np.float32),
                    "network_score": float(row.get("network_score", 0.0)),
                }
            )
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def collate_fn(batch):
    return {
        "disease": [x["disease"] for x in batch],
        "herb": [x["herb"] for x in batch],
        "disease_vec": torch.tensor(np.stack([x["disease_vec"] for x in batch]), dtype=torch.float32),
        "herb_vec": torch.tensor(np.stack([x["herb_vec"] for x in batch]), dtype=torch.float32),
        "label": torch.tensor(np.stack([x["label"] for x in batch]), dtype=torch.float32),
        "network_score": [x["network_score"] for x in batch],
    }


def evaluate(model, loader, criterion):
    model.eval()
    losses, labels, probs, diseases, herbs, network_scores = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            disease_vec = batch["disease_vec"].to(DEVICE)
            herb_vec = batch["herb_vec"].to(DEVICE)
            label = batch["label"].to(DEVICE)
            logits = model(disease_vec, herb_vec)
            loss = criterion(logits, label)
            prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            losses.append(loss.item())
            labels.extend(label.cpu().numpy().reshape(-1).tolist())
            probs.extend(prob.tolist())
            diseases.extend(batch["disease"])
            herbs.extend(batch["herb"])
            network_scores.extend(batch["network_score"])

    pred = [1 if p >= 0.5 else 0 for p in probs]
    auroc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan")
    auprc = average_precision_score(labels, probs) if len(set(labels)) > 1 else float("nan")
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": accuracy_score(labels, pred) if labels else float("nan"),
        "f1": f1_score(labels, pred, zero_division=0) if labels else float("nan"),
        "auroc": auroc,
        "auprc": auprc,
        "predictions": pd.DataFrame(
            {
                "disease": diseases,
                "herb": herbs,
                "label": labels,
                "probability": probs,
                "network_score": network_scores,
            }
        ),
    }


def train_one_fold(fold, train_df, val_df, word2vec, out_dir, args):
    train_dataset = HerbDiseaseVectorDataset(train_df, word2vec)
    val_dataset = HerbDiseaseVectorDataset(val_df, word2vec)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("No usable Word2Vec vectors found for this fold.")

    generator = torch.Generator().manual_seed(TRAIN_CONFIG["seed"] + fold)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = DiseaseTreatmentVectorModel(vector_size=WORD2VEC_CONFIG["vector_size"]).to(DEVICE)
    pos_count = max(1, int(train_df["label"].sum()))
    neg_count = max(1, int((train_df["label"] == 0).sum()))
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    fold_dir = out_dir / f"fold_{fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(fold_dir / "train_data.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(fold_dir / "val_data.csv", index=False, encoding="utf-8-sig")

    best_auroc = -1.0
    best_epoch = 0
    patience_count = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses, train_labels, train_probs = [], [], []
        progress = tqdm(train_loader, desc=f"Fold {fold:02d} Epoch {epoch:03d}", leave=False)
        for batch in progress:
            disease_vec = batch["disease_vec"].to(DEVICE)
            herb_vec = batch["herb_vec"].to(DEVICE)
            label = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(disease_vec, herb_vec)
            loss = criterion(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            train_losses.append(loss.item())
            train_labels.extend(label.detach().cpu().numpy().reshape(-1).tolist())
            train_probs.extend(probs.tolist())
            train_acc = accuracy_score(train_labels, [1 if p >= 0.5 else 0 for p in train_probs])
            progress.set_postfix(loss=np.mean(train_losses), acc=train_acc)

        scheduler.step()
        train_pred = [1 if p >= 0.5 else 0 for p in train_probs]
        train_auroc = roc_auc_score(train_labels, train_probs) if len(set(train_labels)) > 1 else float("nan")
        val_metrics = evaluate(model, val_loader, criterion)
        row = {
            "fold": fold,
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_accuracy": accuracy_score(train_labels, train_pred),
            "train_auroc": train_auroc,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(
            f"Fold {fold:02d} Epoch {epoch:03d} | "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f} "
            f"train_auroc={row['train_auroc']:.4f} | "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_accuracy']:.4f} "
            f"val_auroc={row['val_auroc']:.4f} val_auprc={row['val_auprc']:.4f}"
        )

        pd.DataFrame(history).to_csv(fold_dir / "epoch_metrics.csv", index=False, encoding="utf-8-sig")
        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            best_epoch = epoch
            patience_count = 0
            torch.save(model.state_dict(), fold_dir / "best_model.pth")
            val_metrics["predictions"].to_csv(fold_dir / "best_val_predictions.csv", index=False, encoding="utf-8-sig")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Fold {fold:02d} early stopping at epoch {epoch}; best_epoch={best_epoch}")
                break

    return {"fold": fold, "best_epoch": best_epoch, "best_val_auroc": best_auroc}


def ensure_inputs(word2vec_path, samples_path):
    if not word2vec_path.exists():
        raise FileNotFoundError(
            f"Missing pretrained Word2Vec model: {display_path(word2vec_path)}. "
            "Place the provided model under outputs/word2vec before training or prediction."
        )
    if not samples_path.exists():
        raise FileNotFoundError(
            f"Missing training samples: {display_path(samples_path)}. "
            "Run data_builder.py to generate disease-herb training samples."
        )


def display_path(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_DIR.resolve()))
    except ValueError:
        return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=TRAIN_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=TRAIN_CONFIG["lr"])
    parser.add_argument("--weight-decay", type=float, default=TRAIN_CONFIG["weight_decay"])
    parser.add_argument("--patience", type=int, default=TRAIN_CONFIG["patience"])
    parser.add_argument("--folds", type=int, default=TRAIN_CONFIG["folds"])
    parser.add_argument("--resume", action="store_true", help="Skip folds that already have saved metrics and a best model.")
    parser.add_argument("--cv-mode", choices=["pair", "disease", "stratified_disease"], default="stratified_disease")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--samples", default=None)
    parser.add_argument("--word2vec", default=None)
    parser.add_argument("--max-folds", type=int, default=None, help="Run only the first N folds for staged training.")
    args = parser.parse_args()

    set_seed(TRAIN_CONFIG["seed"])
    samples_path = Path(args.samples) if args.samples else PATHS["samples"]
    word2vec_path = Path(args.word2vec) if args.word2vec else PATHS["word2vec"]
    ensure_inputs(word2vec_path, samples_path)

    samples = pd.read_csv(samples_path, low_memory=False)
    samples = samples.dropna(subset=["disease", "herb", "label"]).copy()
    samples["label"] = samples["label"].astype(int)

    word2vec = Word2Vec.load(str(word2vec_path))
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = "DDTA_Pred"
    out_dir = OUTPUT_DIR / "cv_runs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    samples.to_csv(out_dir / "all_training_samples_used.csv", index=False, encoding="utf-8-sig")

    split_labels = samples["label"].values
    if args.cv_mode == "pair":
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=TRAIN_CONFIG["seed"])
        splits = splitter.split(samples, split_labels)
    elif args.cv_mode == "stratified_disease":
        splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=TRAIN_CONFIG["seed"])
        splits = splitter.split(samples, split_labels, groups=samples["disease"].values)
    else:
        splitter = GroupKFold(n_splits=args.folds)
        splits = splitter.split(samples, split_labels, groups=samples["disease"].values)
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        if args.max_folds is not None and fold > args.max_folds:
            break
        fold_dir = out_dir / f"fold_{fold:02d}"
        metrics_path = fold_dir / "epoch_metrics.csv"
        model_path = fold_dir / "best_model.pth"
        if args.resume and metrics_path.exists() and model_path.exists():
            metrics = pd.read_csv(metrics_path)
            best_row = metrics.sort_values("val_auroc", ascending=False).iloc[0]
            result = {
                "fold": fold,
                "best_epoch": int(best_row["epoch"]),
                "best_val_auroc": float(best_row["val_auroc"]),
            }
            fold_results.append(result)
            pd.DataFrame(fold_results).to_csv(out_dir / "fold_summary.csv", index=False, encoding="utf-8-sig")
            print(
                f"Fold {fold:02d} skipped by --resume | "
                f"best_epoch={result['best_epoch']} best_val_auroc={result['best_val_auroc']:.4f}"
            )
            continue
        train_df = samples.iloc[train_idx].reset_index(drop=True)
        val_df = samples.iloc[val_idx].reset_index(drop=True)
        result = train_one_fold(fold, train_df, val_df, word2vec, out_dir, args)
        fold_results.append(result)
        pd.DataFrame(fold_results).to_csv(out_dir / "fold_summary.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(fold_results)
    mean_auroc = float(summary["best_val_auroc"].mean())
    payload = {
        "mean_auroc": mean_auroc,
        "target_mean_auroc": TRAIN_CONFIG["target_mean_auroc"],
        "passed_target": mean_auroc >= TRAIN_CONFIG["target_mean_auroc"],
        "device": str(DEVICE),
        "folds": args.folds,
        "completed_folds": len(fold_results),
        "cv_mode": args.cv_mode,
        "samples_path": display_path(samples_path),
        "word2vec_path": display_path(word2vec_path),
    }
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["passed_target"]:
        print("WARNING: Mean AUROC did not reach the configured 0.96 target. Check saved fold outputs.")


if __name__ == "__main__":
    main()
