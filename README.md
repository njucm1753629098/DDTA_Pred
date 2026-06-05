# DDTA_Pred

DDTA_Pred is a computational framework for disease-drug treatment association prediction in traditional Chinese medicine. The model uses a provided Word2Vec model to represent diseases and herbs in a shared vector space, learns disease-specific treatment vectors, and prioritizes candidate therapeutic herbs by cosine distance.

## Repository Structure

```text
DDTA_Pred/      Source code
data/           Input biomedical association data
outputs/        Provided model files, training data, cross-validation outputs, and figures
```

The pretrained Word2Vec model is at:

```text
outputs/word2vec/tcm_disease_herb_word2vec.model
```


## Data Preparation

Disease-herb training samples can be generated from the input biomedical association data:

```powershell
cd DDTA_Pred
conda run -n DR python -B .\data_builder.py
```

This step generates disease-herb training samples under:

```text
outputs/training_data/
```

## Model Training

```powershell
cd DDTA_Pred
conda run -n DR python -B .\train.py --epochs 45 --patience 8
```

The default run name is `DDTA_Pred`. Fold-level train/validation splits, epoch metrics, best validation predictions, model checkpoints, and the cross-validation summary are saved under:

```text
outputs/cv_runs/DDTA_Pred
```

## Distance Visualization

```powershell
cd DDTA_Pred
conda run -n DR python -B .\plot_distance.py
```

The script predicts the treatment vector for the query disease and visualizes the nearest herbs in the learned vector space.

## ROC Visualization

```powershell
cd DDTA_Pred
conda run -n DR python -B .\plot_roc.py
```

The script reconstructs ROC curves from saved fold-level validation predictions and reports the mean AUROC.

## Outputs

Generated figures are saved under:

```text
outputs/figures
```
