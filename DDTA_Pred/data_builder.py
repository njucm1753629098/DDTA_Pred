import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import OUTPUT_DIR, PATHS, SAMPLE_CONFIG


def token(kind, value):
    value = "" if pd.isna(value) else str(value).strip()
    return f"{kind}::{value}"


def _read_csv(path):
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"):
        try:
            df = pd.read_csv(path, low_memory=False, encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    else:
        raise last_error
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_source_tables():
    herb_compound = _read_csv(PATHS["herb_compound"])
    compound_target = _read_csv(PATHS["compound_target"])
    disease_target = _read_csv(PATHS["disease_target"])

    if PATHS["manual_herb_compound"].exists():
        manual_hc = _read_csv(PATHS["manual_herb_compound"])
        manual_hc = manual_hc.rename(
            columns={
                "herb_name": "Chinese Character",
                "latin_name": "Latin Name",
                "compound_name": "Related Compound Name",
            }
        )
        manual_hc["Herb ID"] = manual_hc["Chinese Character"]
        manual_hc["English Name"] = manual_hc["Chinese Character"]
        manual_hc["Related Compound ID"] = manual_hc["Related Compound Name"]
        herb_compound = pd.concat([herb_compound, manual_hc], ignore_index=True)

    if PATHS["manual_compound_target"].exists():
        manual_ct = _read_csv(PATHS["manual_compound_target"])
        manual_ct = manual_ct.rename(
            columns={
                "compound_name": "Common name",
                "targets": "Gene Symbol",
            }
        )
        manual_ct["Compound ID"] = manual_ct["Common name"]
        manual_ct["Target Level"] = "manual"
        compound_target = pd.concat([compound_target, manual_ct], ignore_index=True)

    return herb_compound, compound_target, disease_target


def build_knowledge_maps():
    herb_compound, compound_target, disease_target = load_source_tables()

    compound_to_targets = defaultdict(set)
    for _, row in compound_target.iterrows():
        compound_id = str(row.get("Compound ID", "")).strip()
        compound_name = str(row.get("Common name", "")).strip()
        gene = str(row.get("Gene Symbol", "")).strip().upper()
        if not gene or gene == "NAN":
            continue
        for compound in [compound_id, compound_name]:
            if compound and compound != "nan":
                compound_to_targets[compound].add(gene)

    herbs = {}
    herb_to_compounds = defaultdict(set)
    herb_to_targets = defaultdict(set)
    for _, row in herb_compound.iterrows():
        herb_name = str(row.get("Chinese Character", "")).strip()
        if not herb_name or herb_name == "nan":
            continue
        herb_id = str(row.get("Herb ID", herb_name)).strip()
        compound_id = str(row.get("Related Compound ID", "")).strip()
        compound_name = str(row.get("Related Compound Name", "")).strip()
        herbs[herb_name] = {
            "herb_id": herb_id,
            "chinese_name": herb_name,
            "english_name": str(row.get("English Name", "")).strip(),
            "latin_name": str(row.get("Latin Name", "")).strip(),
            "pinyin": str(row.get("Chinese Pin Yin", "")).strip(),
        }
        for compound in [compound_id, compound_name]:
            if compound and compound != "nan":
                herb_to_compounds[herb_name].add(compound)
                herb_to_targets[herb_name].update(compound_to_targets.get(compound, set()))

    disease_to_targets = defaultdict(dict)
    for _, row in disease_target.iterrows():
        disease = str(row.get("disease_name", "")).strip()
        gene = str(row.get("gene_symbol", "")).strip().upper()
        if not disease or disease == "nan" or not gene or gene == "NAN":
            continue
        score = row.get("score", 0.0)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        disease_to_targets[disease][gene] = max(disease_to_targets[disease].get(gene, 0.0), score)

    return herbs, herb_to_compounds, herb_to_targets, disease_to_targets


def load_ppi_dict():
    ppi = {}
    path = PATHS["ppi"]
    if not path.exists():
        return ppi
    for chunk in pd.read_csv(path, sep="\t", chunksize=500000, low_memory=False):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        for _, row in chunk.iterrows():
            gene1 = str(row.get("Gene1", "")).strip().upper()
            gene2 = str(row.get("Gene2", "")).strip().upper()
            if not gene1 or not gene2:
                continue
            try:
                score = float(row.get("combine_score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            ppi[(gene1, gene2)] = score
            ppi[(gene2, gene1)] = score
    return ppi


def _ppi_mean(herb_targets, disease_targets, ppi, max_pairs):
    if not ppi or not herb_targets or not disease_targets:
        return 0.0
    pairs = []
    for h_gene in herb_targets:
        for d_gene in disease_targets:
            pairs.append((h_gene, d_gene))
    if len(pairs) > max_pairs:
        pairs = random.sample(pairs, max_pairs)
    values = [ppi.get(pair, 0.0) for pair in pairs]
    return float(np.mean(values)) if values else 0.0


def build_samples(herbs, herb_to_targets, disease_to_targets, ppi, samples_path=None):
    random.seed(SAMPLE_CONFIG["random_seed"])
    np.random.seed(SAMPLE_CONFIG["random_seed"])

    valid_herbs = {
        herb: set(targets)
        for herb, targets in herb_to_targets.items()
        if len(targets) >= SAMPLE_CONFIG["min_herb_targets"]
    }
    diseases = [
        (disease, target_weights)
        for disease, target_weights in disease_to_targets.items()
        if len(target_weights) >= SAMPLE_CONFIG["min_disease_targets"]
    ]
    diseases = sorted(diseases, key=lambda x: (len(x[1]), sum(x[1].values())), reverse=True)
    max_diseases = SAMPLE_CONFIG["max_diseases"]
    if max_diseases:
        diseases = diseases[:max_diseases]

    rows = []
    for disease, target_weights in tqdm(diseases, desc="Building herb-disease samples"):
        disease_targets = set(target_weights)
        disease_weight_sum = sum(target_weights.values()) or 1.0
        scored = []
        for herb, herb_targets in valid_herbs.items():
            shared = herb_targets & disease_targets
            overlap = len(shared) / max(1, len(disease_targets))
            weighted_overlap = sum(target_weights.get(g, 0.0) for g in shared) / disease_weight_sum
            ppi_score = _ppi_mean(
                herb_targets,
                disease_targets,
                ppi,
                SAMPLE_CONFIG["ppi_max_pairs_per_sample"],
            )
            ppi_norm = min(ppi_score / 1000.0, 1.0)
            network_score = 0.55 * weighted_overlap + 0.30 * overlap + 0.15 * ppi_norm
            scored.append((herb, network_score, len(shared), weighted_overlap, ppi_score))

        if not scored:
            continue
        scores = np.array([x[1] for x in scored])
        pos_thr = float(np.quantile(scores, SAMPLE_CONFIG["positive_quantile"]))
        neg_thr = float(np.quantile(scores, SAMPLE_CONFIG["negative_quantile"]))
        positives = [x for x in scored if x[1] >= pos_thr and x[2] > 0]
        easy_negatives = [x for x in scored if x[1] <= neg_thr and x[2] == 0]
        hard_negatives = [x for x in scored if neg_thr < x[1] < pos_thr and x[2] > 0]
        max_neg = int(max(1, len(positives) * SAMPLE_CONFIG["max_negative_ratio"]))
        hard_count = min(len(hard_negatives), int(max_neg * SAMPLE_CONFIG["hard_negative_fraction"]))
        easy_count = max_neg - hard_count
        negatives = []
        if hard_count > 0:
            negatives.extend(random.sample(hard_negatives, hard_count))
        if len(easy_negatives) > easy_count:
            negatives.extend(random.sample(easy_negatives, easy_count))
        else:
            negatives.extend(easy_negatives)

        for label, group in [(1, positives), (0, negatives)]:
            for herb, network_score, shared_count, weighted_overlap, ppi_score in group:
                meta = herbs.get(herb, {})
                rows.append(
                    {
                        "disease": disease,
                        "herb": herb,
                        "label": label,
                        "network_score": network_score,
                        "shared_target_count": shared_count,
                        "weighted_overlap": weighted_overlap,
                        "ppi_mean_score": ppi_score,
                        "herb_id": meta.get("herb_id", ""),
                        "herb_english_name": meta.get("english_name", ""),
                        "herb_latin_name": meta.get("latin_name", ""),
                    }
                )

    samples = pd.DataFrame(rows).drop_duplicates(["disease", "herb", "label"])
    samples_path = Path(samples_path) if samples_path else PATHS["samples"]
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(samples_path, index=False, encoding="utf-8-sig")
    return samples


def build_training_samples(samples_path=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    herbs, herb_to_compounds, herb_to_targets, disease_to_targets = build_knowledge_maps()
    ppi = load_ppi_dict()
    return build_samples(herbs, herb_to_targets, disease_to_targets, ppi, samples_path=samples_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-path", default=str(PATHS["samples"]))
    args = parser.parse_args()
    samples = build_training_samples(samples_path=Path(args.samples_path))
    print(f"Saved samples: {Path(args.samples_path)} rows={len(samples)}")


if __name__ == "__main__":
    main()
